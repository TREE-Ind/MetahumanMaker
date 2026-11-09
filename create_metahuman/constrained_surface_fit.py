"""Elastic sequential QP for a fixed-topology source-surface projection.

The objective uses integrated source geometry. Linearized signed-area limits
and separating-axis inequalities oppose folds and contacts. Elastic slacks
keep bounded subproblems feasible; full nonlinear checks decide progress.
This optional external-worker module remains experimental. It provides no
continuous collision or injectivity guarantee and is not imported by the UI.
"""
import time
import numpy as np
from scipy import sparse
from create_metahuman.surface_contacts import crossings
from create_metahuman.vertex_contract import fingerprint


def _axes(points,pairs,clearance):
    if not len(pairs):return np.empty((0,34,3)),np.empty((0,34,3,3)),np.empty((0,34))
    p=points[np.asarray(pairs,dtype=int)];a,b=p[:,0],p[:,1]
    ea=np.roll(a,-1,axis=1)-a;eb=np.roll(b,-1,axis=1)-b
    na=np.cross(ea[:,0],ea[:,1]);nb=np.cross(eb[:,0],eb[:,1])
    axes=np.concatenate((na[:,None],nb[:,None],np.cross(ea[:,:,None],eb[:,None,:]).reshape(-1,9,3),np.cross(ea,na[:,None]),np.cross(eb,nb[:,None])),axis=1)
    size=np.linalg.norm(axes,axis=2);valid=size>1e-20;axes/=np.maximum(size[:,:,None],1e-30)
    axes=np.concatenate((axes,-axes),axis=1);valid=np.concatenate((valid,valid),axis=1)
    pa=np.einsum('nvc,nkc->nkv',a,axes);pb=np.einsum('nvc,nkc->nkv',b,axes)
    gap=pb[:,:,None,:]-pa[:,:,:,None];cost=np.sum(np.maximum(clearance-gap,0.),axis=(2,3));cost[~valid]=np.inf
    return axes,gap,cost


def project_hard_step(step,matrix,bounds,lower,upper,deadline):
    """Restore hard linear separators after an inexact QP subproblem.

    Alternating half-space/box projections are bounded work, not an
    optimality certificate. Complete nonlinear endpoint checks still apply.
    """
    if not len(bounds):return step,0.,0.,0
    matrix=matrix.tocsr();before=float(np.maximum(bounds-matrix@step,0.).max(initial=0))
    if before<=1e-9:return step,before,before,0
    result=step.copy();norm=np.asarray(matrix.multiply(matrix).sum(1)).ravel();passes=0
    for sweep in range(32):
        if time.monotonic()>=deadline:break
        violation=bounds-matrix@result
        if violation.max(initial=0)<=1e-9:break
        for row in np.flatnonzero(violation>1e-9)[np.argsort(violation[violation>1e-9])[::-1]]:
            start,end=matrix.indptr[row:row+2];ids=matrix.indices[start:end];values=matrix.data[start:end]
            missing=float(bounds[row]-values@result[ids])
            if missing>1e-9 and norm[row]>1e-30:
                result[ids]=np.clip(result[ids]+values*(missing/norm[row]),lower[ids],upper[ids])
        passes=sweep+1
    after=float(np.maximum(bounds-matrix@result,0.).max(initial=0))
    return (result,before,after,passes) if after<before else (step,before,before,passes)


def project_joint_step(candidate,current,free,matrix,bounds,lower,upper,source_projector,deadline):
    """Alternate fixed source/artist balls with current linear separators.

    Source centers stay fixed for this entire substep. Every returned point
    satisfies the source/artist balls; the caller still audits real contacts.
    """
    step=(candidate-current)[free].ravel();before=float(np.maximum(bounds-matrix@step,0.).max(initial=0))
    best=candidate.copy();best_residual=before;passes=0
    for sweep in range(12):
        if best_residual<=1e-9 or time.monotonic()>=deadline:break
        step,_,_,_=project_hard_step(step,matrix,bounds,lower,upper,deadline)
        projected=current.copy();projected[free]+=step.reshape(-1,3);projected=source_projector(projected)
        if projected.shape!=current.shape or not np.isfinite(projected).all() or not np.array_equal(projected[~free],current[~free]):raise ValueError('Joint projection changes hard pins or produces invalid coordinates')
        step=(projected-current)[free].ravel();residual=float(np.maximum(bounds-matrix@step,0.).max(initial=0))
        if residual<best_residual:best=projected.copy();best_residual=residual
        passes=sweep+1
    return best,before,best_residual,passes


def solve(reference,prior,triangles,projection,free,normals,minimum_area,prior_weight,laplacian,*,iterations=16,time_limit=120.,checkpoint=None,prevent_new_contacts=False,initial_points=None,contact_watchlist=None,expected_contract=None,movement_radius=None,movement_boxes=None,movement_box_origin=None,source_projector=None,contact_margin=0.,differential_target=None,differential_weight=None,joint_projection='alternating',preserve_orientations=False,compact_proposal=False,conic_segment_backtracking=False,conic_elastic_model=False,strict_generated_clearance=False,localize_contact_stalls=False,tangent_objective=None,source_angles=None,step_size_fraction=.0025,angular_secant_model=False,restore_source_angles=False,reverse_coverage=None,following_source_field=None,hard_untangle=None):
    if not conic_elastic_model:import osqp
    start=time.monotonic();deadline=start+time_limit;reference=np.asarray(reference,dtype=float);current=reference.copy();triangles=np.asarray(triangles,dtype=int)
    free=np.asarray(free,dtype=bool);ids=np.flatnonzero(free);lookup=np.full(len(reference),-1,int);lookup[ids]=np.arange(len(ids));ndof=len(ids)*3
    if initial_points is not None:
        current=np.array(initial_points,dtype=float,copy=True)
        if current.shape!=reference.shape or not np.isfinite(current).all():raise ValueError('Invalid resumed surface positions')
        if not np.array_equal(current[~free],reference[~free]):raise ValueError('Resumed surface changes hard pins')
    m=projection['matrix'];load=projection['load'];mass=projection['mass'];constant=float(projection['constant']);scale=max(float(np.mean(mass[mass>0])),1e-12)
    if differential_target is None:
        if differential_weight is not None:raise ValueError('Differential weights require a target')
        regularizer=laplacian.T@laplacian*(scale*.005);differential_load=regularizer@reference
    else:
        differential_target=np.asarray(differential_target,dtype=float);differential_weight=np.asarray(differential_weight,dtype=float)
        if differential_target.shape!=reference.shape or differential_weight.shape!=(len(reference),) or not np.isfinite(differential_target).all() or not np.isfinite(differential_weight).all() or np.any(differential_weight<0):raise ValueError('Invalid source differential objective')
        weighted=sparse.diags(differential_weight*scale);regularizer=laplacian.T@weighted@laplacian
        differential_load=laplacian.T@(differential_target*(differential_weight*scale)[:,None])
    h=(m+sparse.diags(prior_weight)+regularizer).tocsr()
    field_load=load+prior_weight[:,None]*prior+differential_load
    hf=(h[free][:,free]/scale).tocsc();pbase=sparse.kron(hf,sparse.eye(3),format='csc')
    tangent=None
    if tangent_objective is not None:
        tangent=tangent_objective['matrix'].tocsr()*scale
        if tangent.shape!=(reference.size,reference.size) or not np.isfinite(tangent.data).all():raise ValueError('Invalid tangent alignment objective')
        dofs=(ids[:,None]*3+np.arange(3)).ravel()
        pbase=(pbase+tangent[dofs][:,dofs]/scale).tocsc()
    def field_gradient(points):
        value=(h@points-field_load).ravel()
        if tangent is not None:value=value+tangent@points.ravel()
        return value.reshape(reference.shape)
    if not np.isfinite(step_size_fraction) or not 0<step_size_fraction<=.025:raise ValueError('Choose a finite proposal step fraction between zero and 0.025')
    span=float(np.linalg.norm(np.ptp(reference,axis=0)));tol=span*1e-6;clearance=tol*3;step_limit=span*step_size_fraction;total_limit=span*.01
    limits=total_limit
    artist_origin=reference
    if movement_box_origin is not None:
        artist_origin=np.asarray(movement_box_origin,dtype=float)
        if movement_boxes is None or artist_origin.shape!=reference.shape or not np.isfinite(artist_origin).all():raise ValueError('Movement box origins require matching finite coordinate budgets')
    if movement_boxes is not None:
        if movement_radius is not None:raise ValueError('Choose artist boxes or artist spheres')
        movement_boxes=np.asarray(movement_boxes,dtype=float)
        if movement_boxes.shape!=reference.shape or not np.isfinite(movement_boxes).all() or np.any(movement_boxes<0):raise ValueError('Movement boxes must match finite nonnegative coordinate budgets')
        limits=movement_boxes[free].ravel()
        if np.any(abs(current-artist_origin)>movement_boxes+1e-12):raise ValueError('Resumed surface exceeds its original per-coordinate movement boxes')
    if movement_radius is not None:
        movement_radius=np.asarray(movement_radius,dtype=float)
        if movement_radius.shape!=(len(reference),) or not np.isfinite(movement_radius).all() or np.any(movement_radius<0):raise ValueError('Movement radii must match finite nonnegative per-vertex budgets')
        # Component bounds enclose each allowed sphere. Project trial
        # endpoints onto the actual spheres before geometry/objective checks,
        # so axis-aligned corrections can use the full Euclidean budget.
        limits=np.repeat(movement_radius[free],3)
        if np.any(np.linalg.norm(current-reference,axis=1)>movement_radius+1e-12):raise ValueError('Resumed surface exceeds its per-vertex movement radius')
    elif movement_boxes is None and np.any(abs(current-reference)>total_limit+1e-12):raise ValueError('Resumed surface exceeds the original movement budget')
    contract={name:fingerprint(value) for name,value in dict(reference=reference,prior=prior,triangles=triangles,free=free,normals=normals,minimum_area=minimum_area,prior_weight=prior_weight,projection_load=load,projection_mass=mass,projection_constant=np.asarray(constant)).items()}
    for name,matrix in [('projection',m),('laplacian',laplacian)]:
        matrix=matrix.tocsr(copy=True);matrix.sort_indices()
        contract[name]={key:fingerprint(getattr(matrix,key)) for key in ('data','indices','indptr')};contract[name]['shape']=list(matrix.shape)
    if movement_radius is not None:contract['movement_radius']=fingerprint(movement_radius)
    if movement_boxes is not None:contract['movement_boxes']=fingerprint(movement_boxes)
    if movement_box_origin is not None:contract['movement_box_origin']=fingerprint(artist_origin)
    if step_size_fraction!=.0025:contract['proposal_step_fraction']=float(step_size_fraction)
    if differential_target is not None:
        contract['differential_target']=fingerprint(differential_target);contract['differential_weight']=fingerprint(differential_weight)
    if tangent is not None:
        tangent.sort_indices();contract['tangent_objective']={key:fingerprint(getattr(tangent,key)) for key in ('data','indices','indptr')}
    if not np.isfinite(contact_margin) or contact_margin<0:raise ValueError('Contact margin must be finite and nonnegative')
    if contact_margin:contract['contact_margin']=float(contact_margin)
    if source_projector is not None:contract['source_projector']=source_projector.contract
    if joint_projection not in ('alternating','conic'):raise ValueError('Unknown joint source projection method')
    if joint_projection=='conic':
        if source_projector is None or source_projector.contract.get('method') not in {'EXACT_SOURCE_WITNESS_AND_ARTIST_BOXES_V1','EXACT_SOURCE_FACET_AND_ARTIST_BOXES_V1'}:raise ValueError('Conic projection requires original artist boxes and certified source geometry')
        from .convex_surface_step import project_joint_conic
        contract['joint_projection']='conic'
    if preserve_orientations:contract['preserve_source_orientations']=True
    if restore_source_angles:
        if source_angles is None or joint_projection!='conic' or not conic_segment_backtracking:raise ValueError('Source-angle restoration requires joint conic source-angle fitting')
        contract['source_angle_restoration']='LOCAL_FIXED_SOURCE_ENVELOPE_V1'
    if compact_proposal:
        if joint_projection!='conic' or not preserve_orientations:raise ValueError('Compact proposal requires complete conic orientation enforcement')
        contract['compact_conic_proposal']=True
    if conic_segment_backtracking:
        if joint_projection!='conic':raise ValueError('Conic segment backtracking requires joint conic projection')
        contract['conic_segment_backtracking']=True
    if conic_elastic_model:
        if not conic_segment_backtracking:raise ValueError('Joint elastic optimization requires conic segment backtracking')
        contract['conic_elastic_model']=True
    if reverse_coverage is not None:
        if joint_projection!='conic' or not conic_elastic_model:raise ValueError('Reverse coverage requires the joint elastic conic model')
        coverage_matrix=reverse_coverage['matrix'].tocsr()
        contract['reverse_coverage']=dict(report=reverse_coverage['report'],shape=list(coverage_matrix.shape),
            **{name:fingerprint(value) for name,value in dict(data=coverage_matrix.data,indices=coverage_matrix.indices,indptr=coverage_matrix.indptr,target=reverse_coverage['target'],radii=reverse_coverage['radii']).items()})
    if strict_generated_clearance:
        if not prevent_new_contacts or not conic_elastic_model or contact_margin<=0:raise ValueError('Strict clearance requires joint elastic fitting, contact prevention and a positive margin')
        contract['strict_generated_clearance']=True
        contract['clearance_axis_policy']='SOURCE_QUADRATIC_HINT_V1'
        from scipy.sparse.linalg import splu
        hint_delta=(splu(hf).solve(np.asarray((field_load-h@reference)[free]/scale)) if tangent is None else
            splu(pbase).solve((-field_gradient(reference)[free]/scale).ravel()).reshape(-1,3))
        if not np.isfinite(hint_delta).all():raise ValueError('Source objective did not produce a finite separation hint')
        source_hint=reference.copy();source_hint[free]+=np.clip(hint_delta.ravel(),-limits,limits).reshape(-1,3)
        contract['source_separation_hint']=fingerprint(source_hint)
    if localize_contact_stalls:
        if not prevent_new_contacts or strict_generated_clearance:raise ValueError('Local contact holds require contact prevention without strict clearance')
        contract['localize_contact_stalls']=True
    angular=None
    if source_angles is not None:
        from .surface_alignment import angular_model,angular_values
        if not conic_elastic_model:raise ValueError('Source angle constraints require joint conic fitting')
        angular={key:value.copy() if isinstance(value,np.ndarray) else value for key,value in source_angles.items()}
        if angular_values(current,triangles,angular)[1].max(initial=0)>1e-9:raise ValueError('Initial surface violates its source angle contract')
        contract['source_angles']={key:fingerprint(value) for key,value in angular.items() if isinstance(value,np.ndarray)}
        contract['source_angles']['reserve']=float(angular['reserve'])
    if angular_secant_model:
        if angular is None:raise ValueError('Directional area secants require source angle constraints')
        contract['angular_model']='MIDPOINT_DIRECTIONAL_SECANT_WITH_EXACT_ENDPOINT_CHECK_V1'
    if following_source_field is not None:
        if angular is None:raise ValueError('Following-source angles require their original contract')
        initial_following,_=following_source_field(current)
        initial_cosine,initial_violation=angular_values(current,triangles,initial_following)
        if initial_violation.max(initial=0)>1e-9 or np.maximum(initial_following['minimum_cosine']-initial_cosine,0.).max(initial=0)>1e-7:raise ValueError('Initial source-field rebind violates its original angle contract')
        contract['source_angle_directions']=dict(method='FOLLOW_ORIGINAL_SOURCE_FACETS_V1',linearization='ORIGINAL_SOURCE_NORMAL_PARALLEL_TRANSPORT_JACOBIAN_V1',**following_source_field.contract)
    if expected_contract is not None and expected_contract!=contract:raise ValueError('Resume solver inputs changed; rebuild the source-fit checkpoint')
    p=reference[triangles];grad0=np.stack((np.cross(p[:,1]-p[:,2],normals),np.cross(p[:,2]-p[:,0],normals),np.cross(p[:,0]-p[:,1],normals)),axis=1)
    if following_source_field is not None:
        grad0[following_source_field.scope]+=following_source_field.signed_gradient_correction(reference)[following_source_field.scope]
    normal_scale=np.maximum(np.sqrt(np.sum(grad0**2,axis=(1,2))),1e-8)
    untangle=np.zeros(len(triangles),bool) if hard_untangle is None else np.asarray(hard_untangle,dtype=bool)
    if untangle.shape!=(len(triangles),) or np.any(untangle&~np.any(free[triangles],axis=1)):raise ValueError('Hard untangling must name editable target faces')
    if untangle.any():contract['hard_untangle']=fingerprint(untangle)
    diag=np.asarray(h.diagonal());contacts=crossings(current,triangles,pairs=True,tolerance=tol)
    if tangent is not None:diag=diag+tangent.diagonal().reshape(-1,3).mean(1)
    active_faces=np.any(free[triangles],axis=1)
    immutable_contacts={pair for pair in contacts if not np.any(active_faces[list(pair)])}
    def current_contacts(points):
        return immutable_contacts|crossings(points,triangles,pairs=True,tolerance=tol,active_faces=active_faces)
    allowed_contacts=contacts.copy();generated_contacts=set();prevented_trials=0
    radius_cuts={};contact_holds=set();contact_axis_rank={}
    if contact_watchlist is not None:
        watch=np.asarray(contact_watchlist)
        if watch.size:
            if not prevent_new_contacts or watch.dtype.kind not in 'iu' or watch.ndim!=2 or watch.shape[1]!=2 or watch.min()<0 or watch.max()>=len(triangles) or np.any(watch[:,0]>=watch[:,1]):raise ValueError('Invalid resumed contact watchlist')
            if np.any(triangles[watch[:,0],:,None]==triangles[watch[:,1],None,:]):raise ValueError('Contact watchlist includes shared triangle vertices')
            generated_contacts={tuple(map(int,pair)) for pair in watch}-allowed_contacts
    def geometry(points,pair_set,normal_override=None):
        p=points[triangles];cross=np.cross(p[:,1]-p[:,0],p[:,2]-p[:,0]);signed=np.einsum('ni,ni->n',cross,normals if normal_override is None else normal_override)
        violation=np.maximum(minimum_area-signed,0.)/normal_scale
        pairs=sorted(pair_set);vertices=np.asarray([triangles[list(pair)] for pair in pairs],dtype=int)
        axes,gaps,costs=_axes(points,vertices,clearance)
        soft=np.array([pair in allowed_contacts for pair in pairs],bool) if prevent_new_contacts else np.ones(len(pairs),bool)
        collision=float(np.min(costs,axis=1)[soft].sum()) if len(pairs) else 0.
        count=int(np.count_nonzero(np.min(costs,axis=1)>1e-18)) if len(pairs) else 0
        return signed,violation,collision,count,axes,gaps,costs,pairs
    def source_rms(points):
        energy=float(np.sum(points*(m@points))-2*np.sum(points*load)+constant)
        return float(np.sqrt(max(energy,0.)/max(float(mass.sum()),1e-30)))
    initial_rms=source_rms(current);rms_limit=max(initial_rms*2,span*.0015)
    def objective(points,violation,collision,penalty):
        delta=points-reference
        base=.5*np.sum(delta*(h@delta))+np.sum(delta*(h@reference-field_load))
        if tangent is not None:base+=.5*float(delta.ravel()@(tangent@delta.ravel()))+float(delta.ravel()@(tangent@reference.ravel()))
        return float(base/scale+penalty*(violation.sum()+collision))
    signed,violation,collision,clearance_count,_,_,_,_=geometry(current,contacts)
    initial=dict(reversed_faces=int((signed<=0).sum()),area_violations=int((violation>1e-8).sum()),crossings=len(contacts),clearance_pairs=clearance_count,source_rms=initial_rms,resumed=initial_points is not None,resumed_watchlist_pairs=len(generated_contacts))
    best=(initial['reversed_faces'],initial['crossings'],initial_rms,current.copy());best_fit=objective(current,violation,collision,0.);history=[];reason='iteration_limit';penalty=span*.05
    angular_hint=None;secant_stalls=0;restoration_gain=.5
    for iteration in range(iterations):
        if time.monotonic()>=deadline:reason='deadline';break
        if not ndof:reason='no_movable_vertices';break
        if following_source_field is not None:
            angular,transport_normals=following_source_field(current)
            normals[following_source_field.scope]=transport_normals[following_source_field.scope]
        if source_projector is not None:source_projector.prepare(current)
        proper=current_contacts(current);contacts|=proper
        signed,violation,collision,clearance_count,axes,gaps,costs,pairs=geometry(current,contacts)
        keep=np.min(costs,axis=1)>1e-18 if len(pairs) else np.empty(0,bool)
        contacts={pair for pair,k in zip(pairs,keep) if k}|generated_contacts
        signed,violation,collision,clearance_count,axes,gaps,costs,pairs=geometry(current,contacts)
        if not proper and not clearance_count and not np.any(violation>1e-8) and source_rms(current)<=span*1e-7:reason='geometric_limits_met';break
        rows=[];cols=[];values=[];bounds=[];hard_rows=[];orientation_rows=[];untangle_rows=[];contact_row_pairs={};axis_alternatives={}
        active=np.flatnonzero((signed<minimum_area*3)|untangle)
        p=current[triangles]
        gradients=np.stack((np.cross(p[:,1]-p[:,2],normals),np.cross(p[:,2]-p[:,0],normals),np.cross(p[:,0]-p[:,1],normals)),axis=1)
        if following_source_field is not None:
            gradients[following_source_field.scope]+=following_source_field.signed_gradient_correction(current)[following_source_field.scope]
        for face in active:
            r=len(bounds);bounds.append(float((minimum_area[face]-signed[face])/normal_scale[face]))
            for corner,vertex in enumerate(triangles[face]):
                variable=lookup[vertex]
                if variable<0:continue
                for axis in (0,1,2):rows.append(r);cols.append(variable*3+axis);values.append(float(gradients[face,corner,axis]/normal_scale[face]))
        restoring=np.flatnonzero(untangle&(signed<minimum_area))
        if len(restoring):
            # Existing folds can share vertices, so forcing every deficit to
            # shrink in the same tangent substep can exclude all useful
            # directions. Restore their total positive-part deficit instead.
            # The nonlinear endpoint must actually reduce that total, while
            # every previously outward face remains separately protected.
            r=len(bounds);bounds.append(float(restoration_gain*violation[restoring].sum()))
            hard_rows.append(r);orientation_rows.append(r);untangle_rows.append(r)
            for face in restoring:
                for corner,vertex in enumerate(triangles[face]):
                    variable=lookup[vertex]
                    if variable<0:continue
                    for axis in (0,1,2):rows.append(r);cols.append(variable*3+axis);values.append(float(gradients[face,corner,axis]/normal_scale[face]))
        if preserve_orientations:
            for face in np.flatnonzero(active_faces&(signed>0)):
                floor=min(float(minimum_area[face]),float(signed[face])*.5)
                r=len(bounds);bounds.append(float((floor-signed[face])/normal_scale[face]));hard_rows.append(r);orientation_rows.append(r)
                for corner,vertex in enumerate(triangles[face]):
                    variable=lookup[vertex]
                    if variable<0:continue
                    for axis in (0,1,2):rows.append(r);cols.append(variable*3+axis);values.append(float(gradients[face,corner,axis]/normal_scale[face]))
        for vertex,directions in radius_cuts.items():
            variable=lookup[vertex]
            for direction in directions:
                r=len(bounds);bounds.append(float(direction@(current[vertex]-reference[vertex])-movement_radius[vertex]));hard_rows.append(r)
                for axis in (0,1,2):rows.append(r);cols.append(variable*3+axis);values.append(float(-direction[axis]))
        for vertex in sorted(contact_holds):
            variable=lookup[vertex]
            if variable<0:continue
            for axis in range(3):
                for sign in (-1.,1.):
                    r=len(bounds);bounds.append(0.);hard_rows.append(r)
                    rows.append(r);cols.append(variable*3+axis);values.append(sign)
        unresolved=0;strict_pairs=0;touching_fixed_pairs=0;uncertified_separators=0
        if strict_generated_clearance:
            offset=(current-artist_origin)[free].ravel();motion_lower=np.zeros_like(current);motion_upper=np.zeros_like(current)
            motion_lower[free]=np.maximum(-step_limit,-limits-offset).reshape(-1,3);motion_upper[free]=np.minimum(step_limit,limits-offset).reshape(-1,3)
            local_hint=current.copy();local_hint[free]+=np.clip((source_hint-current)[free],motion_lower[free],motion_upper[free])
        for k,pair in enumerate(pairs):
            hard=prevent_new_contacts and pair not in allowed_contacts
            va,vb=triangles[list(pair)];strict=hard and strict_generated_clearance
            pair_axes=axes[k];pair_gaps=gaps[k];pair_costs=costs[k]
            if strict:
                hint_axes,_,hint_costs=_axes(local_hint,np.array([[va,vb]]),clearance);hint_axes=hint_axes[0]
                pa=np.einsum('vc,kc->kv',current[va],hint_axes);pb=np.einsum('vc,kc->kv',current[vb],hint_axes)
                extra_gaps=pb[:,None,:]-pa[:,:,None]
                extra_cost=np.sum(np.maximum(clearance-extra_gaps,0.),axis=(1,2));extra_cost[~np.isfinite(hint_costs[0])]=np.inf
                pair_axes=np.vstack((pair_axes,hint_axes));pair_gaps=np.concatenate((pair_gaps,extra_gaps));pair_costs=np.r_[pair_costs,extra_cost]
                low_a=np.minimum(pair_axes[:,None,:]*motion_lower[va],pair_axes[:,None,:]*motion_upper[va]).sum(2)
                high_b=np.maximum(pair_axes[:,None,:]*motion_lower[vb],pair_axes[:,None,:]*motion_upper[vb]).sum(2)
                reachable_gap=pair_gaps+high_b[:,None,:]-low_a[:,:,None]
            desired_margin=contact_margin if strict else (0. if hard else clearance)
            deficit=np.maximum(desired_margin-pair_gaps,0.)
            influence=free[va][:,None]/np.maximum(diag[va,None],1e-12)+free[vb][None,:]/np.maximum(diag[vb][None,:],1e-12)
            possible=np.all((deficit<=0)|(influence[None]>0),axis=(1,2))
            if strict:possible&=np.all(reachable_gap>=desired_margin-span*1e-12,axis=(1,2))
            weighted=np.sum(deficit,axis=(1,2));weighted[~possible]=np.inf
            weighted[~np.isfinite(pair_costs)]=np.inf
            touching=False
            if strict and not np.isfinite(weighted).any():
                # Frozen points can legitimately touch. Keep those exact
                # gaps, but require positive clearance for movable pairs.
                static_gap=np.min(np.where(influence[None]>0,np.inf,pair_gaps),axis=(1,2))
                possible=(static_gap>=-span*1e-12)&np.isfinite(pair_costs)
                possible&=np.all((influence[None]<=0)|(reachable_gap>=contact_margin-span*1e-12),axis=(1,2))
                if possible.any():weighted=np.where(possible,np.sum(deficit,axis=(1,2)),np.inf);touching=True
                else:
                    # No affine separator exists for these frozen point
                    # sets. Preserve the old envelope and rely on the full
                    # nonlinear no-new-contact gate; record this limitation.
                    deficit=np.maximum(-pair_gaps,0.);possible=np.all((deficit<=0)|(influence[None]>0),axis=(1,2))
                    weighted=np.sum(deficit,axis=(1,2));weighted[~possible|~np.isfinite(pair_costs)]=np.inf
                    strict=False;uncertified_separators+=1
            if not np.isfinite(weighted).any():unresolved+=1;continue
            if hard:
                if strict:
                    # Select a separating side from the source objective,
                    # rather than preserving an incorrect near-contact order.
                    pa=np.einsum('vc,kc->kv',local_hint[va],pair_axes);pb=np.einsum('vc,kc->kv',local_hint[vb],pair_axes)
                    hint_gap=pb[:,None,:]-pa[:,:,None]
                    wanted=np.where(influence[None]>0,contact_margin,np.minimum(contact_margin,pair_gaps))
                    need=np.maximum(wanted-hint_gap,0.)
                    score=np.sum(np.where(influence[None]>0,need**2/np.maximum(influence[None],1e-30),0.),axis=(1,2))
                    score[~np.isfinite(weighted)]=np.inf;best_score=float(score.min())
                    tied=score<=best_score+max(1e-30,best_score*1e-8)
                    chosen=int(np.argmax(np.where(tied,np.min(hint_gap,axis=(1,2)),-np.inf)))
                    if untangle.any():
                        # Contact separation is a union of halfspace choices,
                        # not one permanently authoritative axis. Keep a small
                        # ranked set of distinct original-geometry directions
                        # so a conflicting convex branch can be replaced.
                        ranked=np.lexsort((-np.min(hint_gap,axis=(1,2)),score))
                        choices=[]
                        for axis_id in ranked:
                            if not np.isfinite(score[axis_id]):continue
                            if any(float(pair_axes[axis_id]@pair_axes[old])>1-1e-8 for old in choices):continue
                            choices.append(int(axis_id))
                        if choices:
                            rank=min(contact_axis_rank.get(pair,0),len(choices)-1)
                            chosen=choices[rank];axis_alternatives[pair]=len(choices)
                else:
                    margin=np.min(pair_gaps,axis=(1,2));margin[~np.isfinite(weighted)]=-np.inf
                    chosen=int(np.argmax(margin))
            else:chosen=int(np.argmin(weighted))
            n=pair_axes[chosen]
            gap_floor=contact_margin if strict else (min(contact_margin,float(np.min(pair_gaps[chosen]))) if hard else clearance)
            strict_pairs+=int(strict);touching_fixed_pairs+=int(touching)
            for a in range(3):
                for b in range(3):
                    row_floor=min(gap_floor,float(pair_gaps[chosen,a,b])) if strict and influence[a,b]<=0 else gap_floor
                    r=len(bounds);bounds.append(float(row_floor-pair_gaps[chosen,a,b]))
                    if hard:hard_rows.append(r);contact_row_pairs[r]=pair
                    for sign,vertex in ((-1.,va[a]),(1.,vb[b])):
                        variable=lookup[vertex]
                        if variable<0:continue
                        for axis in (0,1,2):rows.append(r);cols.append(variable*3+axis);values.append(float(sign*n[axis]))
        nconstraint=len(bounds)
        c=sparse.coo_matrix((values,(rows,cols)),shape=(nconstraint,ndof)).tocsc()
        c.eliminate_zeros();nonconstant=np.asarray(c.getnnz(axis=1)).ravel()>0
        hard_mask=np.zeros(nconstraint,bool);hard_mask[hard_rows]=True;bound_array=np.asarray(bounds)
        untangle_mask=np.zeros(nconstraint,bool);untangle_mask[untangle_rows]=True
        orientation_mask=np.zeros(nconstraint,bool);orientation_mask[orientation_rows]=True
        if np.any((~nonconstant)&hard_mask&(bound_array>1e-9)):reason='infeasible_fixed_separator';break
        origin_rows=np.flatnonzero((hard_mask&~untangle_mask)[nonconstant])
        constant_constraints=int((~nonconstant).sum());c=c[nonconstant].tocsc();bounds=bound_array[nonconstant];hard_rows=np.flatnonzero(hard_mask[nonconstant]);nconstraint=len(bounds)
        # The proposal is not an accepted endpoint. In compact mode the full
        # positive-face envelope stays in c[hard_rows] for the joint conic
        # solve, without duplicating thousands of rows in the proposal QP.
        proposal_mask=~orientation_mask[nonconstant] if compact_proposal else np.ones(nconstraint,bool)
        qp_c=c[proposal_mask];qp_bounds=bounds[proposal_mask];qp_hard=hard_mask[nonconstant][proposal_mask];nproposal=len(qp_bounds)
        identity=sparse.eye(nproposal,format='csc');zero=sparse.csc_matrix((ndof,nproposal))
        constraints=sparse.vstack((sparse.hstack((qp_c,identity)),sparse.hstack((sparse.csc_matrix((nproposal,ndof)),identity)),sparse.hstack((sparse.eye(ndof),zero))),format='csc')
        delta0=(current-artist_origin)[free].ravel();lower_step=np.maximum(-step_limit,-limits-delta0);upper_step=np.minimum(step_limit,limits-delta0)
        slack_upper=np.full(nproposal,np.inf);slack_upper[qp_hard]=0.
        lower=np.r_[qp_bounds,np.zeros(nproposal),lower_step];upper=np.r_[np.full(nproposal,np.inf),slack_upper,upper_step]
        # An L1 elastic penalty avoids the ill-conditioned 1e4--1e6
        # quadratic slacks used by the initial prototype. Start from the
        # feasible zero step with exactly enough nonnegative slack.
        pqp=sparse.block_diag((pbase,identity*10.),format='csc');q=np.r_[(field_gradient(current)[free]/scale).ravel(),np.full(nproposal,penalty)]
        result=None;conic_model=None
        if conic_elastic_model:
            soft_mask=np.ones(nconstraint,bool);soft_mask[hard_rows]=False
            conic_model=dict(hessian=pbase,gradient=q[:ndof],soft_matrix=c[soft_mask],soft_bounds=bounds[soft_mask],penalty=penalty)
            if untangle.any():conic_model['strict_feasibility']=True
            if reverse_coverage is not None:
                from .reverse_coverage import affine
                conic_model['coverage_cones']=affine(current,free,reverse_coverage)
            if angular is not None:
                midpoint=(current+angular_hint)*.5 if angular_secant_model and angular_hint is not None else None
                conic_model['angle_cones']=(following_source_field.linearize(current,free,angular,linearization_points=midpoint) if following_source_field is not None else angular_model(current,triangles,free,angular,linearization_points=midpoint))
            step=np.zeros(ndof)
        else:
            solver=osqp.OSQP();solver.setup(P=sparse.triu(pqp,format='csc'),q=q,A=constraints,l=lower,u=upper,verbose=False,eps_abs=1e-7,eps_rel=1e-5,max_iter=1500,time_limit=min(12.,max(1.,deadline-time.monotonic())),polishing=False)
            solver.warm_start(x=np.r_[np.zeros(ndof),np.minimum(np.maximum(qp_bounds,0.),slack_upper)])
            result=solver.solve(raise_error=False)
            if result.x is None or not np.isfinite(result.x[:ndof]).all() or any(word in result.info.status.lower() for word in ('infeasible','non convex','unsolved')):reason='subproblem_failed';break
            step=np.clip(result.x[:ndof],lower_step,upper_step)
        if compact_proposal:
            hard_before=float(np.maximum(bounds[hard_rows]-c[hard_rows]@step,0.).max(initial=0));hard_after=hard_before;hard_passes=0
        else:step,hard_before,hard_after,hard_passes=project_hard_step(step,c[hard_rows],bounds[hard_rows],lower_step,upper_step,min(deadline,time.monotonic()+2.))
        delta=np.zeros_like(current);delta[free]=step.reshape(-1,3)
        old_merit=objective(current,violation,collision,penalty);accepted=False;trials=[];discovered=set();joint_passes=0;joint_before=0.;joint_after=0.;conic_reports=[];largest_trial_movement=0.
        segment_end=None;segment_fraction=1.;angular_restorations=[]
        next_angular_hint=None;used_angular_secant=bool(angular_secant_model and angular_hint is not None)
        fractions=(1.,.5,.25,.125,.0625,.03125,.015625,.0078125) if conic_segment_backtracking else (1.,.5,.25,.125,.0625)
        for fraction in fractions:
            if time.monotonic()>=deadline:break
            candidate=current+delta*fraction;candidate[~free]=reference[~free]
            if movement_radius is not None:
                displacement=candidate-reference;length=np.linalg.norm(displacement,axis=1)
                if fraction==1.:
                    for vertex in np.flatnonzero(free&(length>movement_radius+1e-9)):
                        direction=displacement[vertex]/length[vertex];previous=radius_cuts.setdefault(int(vertex),[])
                        if not previous or max(float(direction@old) for old in previous)<.999999:previous.append(direction.copy())
                ratio=np.minimum(1.,movement_radius/np.maximum(length,1e-300))
                candidate=reference+displacement*ratio[:,None];candidate[~free]=reference[~free]
            if source_projector is not None:
                if conic_segment_backtracking and segment_end is not None:
                    candidate=current+(segment_end-current)*(fraction/segment_fraction)
                    # A shortened restoration step need not attain the full
                    # proposal's deficit reduction. Preserve every initially
                    # feasible hard row here; measure untangling progress on
                    # the actual nonlinear surface below.
                    segment_rows=origin_rows if untangle.any() else hard_rows
                    jb=ja=float(np.maximum(bounds[segment_rows]-c[segment_rows]@((candidate-current)[free].ravel()),0.).max(initial=0));jp=0
                else:
                    candidate=source_projector(candidate)
                    if joint_projection=='conic':
                        candidate,detail=project_joint_conic(candidate,current,free,c[hard_rows],bounds[hard_rows],lower_step,upper_step,source_projector,min(deadline,time.monotonic()+(16. if conic_elastic_model else 4. if conic_segment_backtracking else 2.)),model=conic_model)
                        detail['fraction']=fraction;conic_reports.append(detail);jb=detail['hard_residual_before'];ja=detail['hard_residual_after'];jp=detail['iterations']
                        if not detail['valid']:
                            trials.append(dict(fraction=fraction,step_accepted=False,reason='joint_projection_failed',projection_status=detail['status']))
                            if conic_segment_backtracking:break
                            continue
                        if conic_segment_backtracking:
                            # The current point and projected endpoint share
                            # fixed source balls, artist boxes and halfspaces.
                            # Their segment is feasible only if the current
                            # point satisfies those same constraints as well.
                            if not strict_generated_clearance and np.max(bounds[origin_rows],initial=0)>1e-9:raise ValueError('Current surface violates the segment contact envelope')
                            segment_end=candidate.copy();segment_fraction=fraction
                    else:candidate,jb,ja,jp=project_joint_step(candidate,current,free,c[hard_rows],bounds[hard_rows],lower_step,upper_step,source_projector,min(deadline,time.monotonic()+.75))
                joint_passes+=jp;joint_before=max(joint_before,jb);joint_after=max(joint_after,ja)
                if conic_segment_backtracking:
                    ids=source_projector.ids
                    source_error=float(np.max(source_projector.source_distance(candidate)-source_projector.source_radius,initial=0))
                    artist_error=float(np.max(np.abs(candidate[ids]-source_projector.reference[ids])-source_projector.limits[ids],initial=0))
                    if ja>1e-9 or source_error>1e-10 or artist_error>1e-10:
                        trials.append(dict(fraction=fraction,step_accepted=False,reason='segment_envelope_violation',hard_residual=ja,source_residual=source_error,artist_residual=artist_error));continue
                if candidate.shape!=current.shape or not np.isfinite(candidate).all() or not np.array_equal(candidate[~free],reference[~free]):raise ValueError('Source projection changes hard pins or produces invalid coordinates')
                if movement_radius is not None and np.any(np.linalg.norm(candidate-reference,axis=1)>movement_radius+1e-12):raise ValueError('Source projection violates the original artist budget')
            largest_trial_movement=max(largest_trial_movement,float(np.linalg.norm(candidate-current,axis=1).max(initial=0)))
            trial_proper=current_contacts(candidate);trial_contacts=contacts|trial_proper
            new_pairs=trial_proper-allowed_contacts
            if prevent_new_contacts and new_pairs:
                discovered|=new_pairs;prevented_trials+=1
                trials.append(dict(fraction=fraction,crossings=len(trial_proper),new_pairs=len(new_pairs),new_pairs_sample=[list(pair) for pair in sorted(new_pairs)[:3]],step_accepted=False,reason='new_contact_constraint_generation'))
                continue
            if restore_source_angles and angular is not None and not angular_restorations:
                cosine,error=angular_values(candidate,triangles,angular)
                p=candidate[triangles];candidate_signed=np.einsum('ij,ij->i',np.cross(p[:,1]-p[:,0],p[:,2]-p[:,0]),normals)
                if error.max(initial=0)>1e-9 or np.maximum(angular['minimum_cosine']-cosine,0.).max(initial=0)>1e-7 or np.any((signed>0)&(candidate_signed<=0)):
                    from .surface_angle_restoration import restore
                    corrected,restoration=restore(candidate,current,triangles,free,angular,c[hard_rows],bounds[hard_rows],
                        lower_step,upper_step,source_projector,min(deadline,time.monotonic()+6.),orientation_normals=normals,minimum_area=minimum_area)
                    angular_restorations.append(restoration)
                    restoration['trial_fraction']=fraction
                    if restoration['valid']:
                        candidate=corrected;segment_end=candidate.copy();segment_fraction=fraction
                        largest_trial_movement=max(largest_trial_movement,float(np.linalg.norm(candidate-current,axis=1).max(initial=0)))
                        trial_proper=current_contacts(candidate);trial_contacts=contacts|trial_proper;new_pairs=trial_proper-allowed_contacts
                        if prevent_new_contacts and new_pairs:
                            discovered|=new_pairs;prevented_trials+=1
                            trials.append(dict(fraction=fraction,crossings=len(trial_proper),new_pairs=len(new_pairs),step_accepted=False,reason='restored_contact_constraint_generation'))
                            continue
            candidate_angles=angular;candidate_normals=None
            if following_source_field is not None:
                try:
                    candidate_angles,transport_normals=following_source_field(candidate)
                    candidate_normals=normals.copy();candidate_normals[following_source_field.scope]=transport_normals[following_source_field.scope]
                except ValueError as exc:
                    trials.append(dict(fraction=fraction,step_accepted=False,reason='source_field_support',diagnostic=str(exc)));continue
            ns,nv,nc,nclear,_,_,_,_=geometry(candidate,trial_contacts,normal_override=candidate_normals);rms=source_rms(candidate);merit=objective(candidate,nv,nc,penalty)
            restoration_progress=False
            if untangle.any():
                old_deficit=violation[untangle];new_deficit=nv[untangle]
                restoration_progress=bool(old_deficit.sum()>1e-9 and
                    new_deficit.sum()<=old_deficit.sum()*(1-min(.05,restoration_gain*fraction*.25)))
                if np.any(new_deficit>1e-9) and not restoration_progress:
                    trials.append(dict(fraction=fraction,step_accepted=False,reason='hard_source_untangle',unresolved_faces=int(np.count_nonzero(new_deficit>1e-9)),
                        before_deficit=float(old_deficit.sum()),after_deficit=float(new_deficit.sum())));continue
            if angular is not None:
                angle_cosine,angle_violation=angular_values(candidate,triangles,candidate_angles);angle_error=float(angle_violation.max(initial=0))
                cosine_error=float(np.maximum(candidate_angles['minimum_cosine']-angle_cosine,0.).max(initial=0))
                if angle_error>1e-9 or cosine_error>1e-7:
                    if angular_secant_model and next_angular_hint is None:next_angular_hint=candidate.copy()
                    trials.append(dict(fraction=fraction,step_accepted=False,reason='nonlinear_source_angle_violation',source_angle_residual=angle_error,source_angle_cosine_error=cosine_error,
                        movement_m=float(np.linalg.norm(candidate-current,axis=1).max()),source_field_change=float(np.linalg.norm(candidate_normals-normals,axis=1).max()) if candidate_normals is not None else 0.));continue
            if preserve_orientations and np.any((signed>0)&(ns<=0)):
                trials.append(dict(fraction=fraction,new_source_orientation_violations=int(np.count_nonzero((signed>0)&(ns<=0))),step_accepted=False,reason='new_source_orientation_violation'))
                continue
            trials.append(dict(fraction=fraction,reversed=int((ns<=0).sum()),crossings=len(trial_proper),new_pairs=len(new_pairs),source_rms=rms,merit=merit))
            if rms<=rms_limit and (merit<old_merit-1e-12 or restoration_progress):
                if candidate_normals is not None:
                    normals[following_source_field.scope]=candidate_normals[following_source_field.scope];angular=candidate_angles
                current=candidate;contacts=trial_contacts;signed=ns;violation=nv;collision=nc;clearance_count=nclear;proper=trial_proper;accepted=True;break
        reversed_count=int((signed<=0).sum());rms=source_rms(current)
        if conic_reports:
            joint_passes=sum(item['iterations'] for item in conic_reports)
            joint_before=max(item['hard_residual_before'] for item in conic_reports)
            joint_after=max(item['hard_residual_after'] for item in conic_reports)
        fit_energy=objective(current,violation,collision,0.)
        improves=(rms<best[2]) if tangent is None else (fit_energy<best_fit-1e-12)
        if reversed_count<=best[0] and len(proper)<=best[1] and (reversed_count<best[0] or len(proper)<best[1] or improves):
            best=(reversed_count,len(proper),rms,current.copy());best_fit=fit_energy
        if localize_contact_stalls and not accepted:
            repeated=discovered&generated_contacts
            for pair in repeated:contact_holds.update(int(v) for v in triangles[list(pair)].ravel() if free[v])
        added=discovered-generated_contacts;generated_contacts|=discovered
        record=dict(iteration=iteration+1,reversed_faces=reversed_count,area_violations=int((violation>1e-8).sum()),crossings=len(proper),new_pairs=len(proper-allowed_contacts),generated_contact_pairs=len(generated_contacts),added_contact_pairs=len(added),clearance_pairs=clearance_count,source_rms=rms,constraints=nconstraint,constant_constraints_removed=constant_constraints,hard_residual_before=hard_before,hard_residual_after=hard_after,hard_projection_passes=hard_passes,joint_projection_passes=joint_passes,joint_residual_before=joint_before,joint_residual_after=joint_after,unresolved_pairs=unresolved,qp_status=result.info.status if result is not None else 'not_run_direct_conic',qp_iterations=int(result.info.iter) if result is not None else 0,qp_primal_residual=float(result.info.prim_res) if result is not None else None,accepted=accepted,trials=trials)
        if untangle.any():record['restoration_gain']=restoration_gain
        if conic_reports:record['conic_projections']=conic_reports
        if angular_restorations:record['source_angle_restorations']=angular_restorations
        record.update(proposal_constraints=nproposal if result is not None else 0,orientation_constraints_in_conic=int(orientation_mask[nonconstant].sum()),proposal_solver_seconds=float(result.info.run_time) if result is not None else 0.)
        if strict_generated_clearance:record.update(strict_clearance_pairs=strict_pairs,touching_fixed_pairs=touching_fixed_pairs,uncertified_separators=uncertified_separators)
        if localize_contact_stalls:record['contact_held_vertices']=len(contact_holds)
        if angular is not None:
            cosine,error=angular_values(current,triangles,angular)
            record.update(source_angle_faces=len(angular['face_ids']),source_angle_residual=float(error.max(initial=0)),angular_secant_used=used_angular_secant)
            if angular_secant_model:
                # Two unsuccessful secant iterations restore the ordinary
                # first-order proposal model. Feasibility/merit gates never
                # depend on which model generated the candidate.
                secant_stalls=secant_stalls+1 if used_angular_secant and not accepted else 0
                angular_hint=None if secant_stalls>=2 else next_angular_hint
                if secant_stalls>=2:secant_stalls=0;record['angular_secant_fallback']=True
            # Keep original scopes and angle thresholds fixed. The optional
            # transported model rebinds directions only to original facets;
            # it never ratchets quality thresholds to intermediate endpoints.
        history.append(record);print(record,flush=True)
        if checkpoint:checkpoint(best[3],dict(initial=initial,history=history,chosen_reversed=best[0],chosen_crossings=best[1],chosen_source_rms=best[2],accepted=False,production_ready=False))
        if not accepted:
            # A solved, feasible local model with zero slack and a negligible
            # step cannot benefit from escalating its already inactive elastic
            # penalty. Still run the full final contact audit below.
            if conic_elastic_model and not discovered and largest_trial_movement<=span*1e-8 and conic_reports and all(
                item['valid'] and item.get('solver_converged',False) and abs(item.get('elastic_model_before',float('inf')))<=1e-12 and
                abs(item.get('elastic_model_after',float('inf')))<=1e-12 for item in conic_reports):
                reason='stationary_feasible_local_model';break
            if conic_elastic_model and conic_reports and not any(item['valid'] for item in conic_reports):
                if untangle.any() and strict_generated_clearance:
                    diagnostic=conic_reports[-1].get('local_infeasibility',{})
                    changed=None;retained=np.flatnonzero(nonconstant)
                    for item in diagnostic.get('top_hard_rows',[]):
                        row=int(retained[hard_rows[item['row']]])
                        pair=contact_row_pairs.get(row)
                        if pair is not None and contact_axis_rank.get(pair,0)+1<axis_alternatives.get(pair,0):
                            contact_axis_rank[pair]=contact_axis_rank.get(pair,0)+1;changed=pair;break
                    if changed is not None:
                        record['contact_branch_retry']=dict(pair=list(changed),rank=contact_axis_rank[changed]);continue
                if untangle.any() and restoration_gain>1/128 and time.monotonic()<deadline:
                    restoration_gain*=.5;record['restoration_retry']=True;continue
                reason='deadline' if time.monotonic()>=deadline else 'joint_subproblem_failed';break
            if prevent_new_contacts and discovered:
                if not added:step_limit*=.5
                continue
            if penalty<span*50:penalty*=4;continue
            reason='stalled';break
    final_contacts=crossings(best[3],triangles,pairs=True,tolerance=tol)
    if final_contacts!=current_contacts(best[3]):raise AssertionError('Active-region contact cache disagrees with the complete body audit')
    if prevent_new_contacts and final_contacts-allowed_contacts:raise AssertionError('Generated-contact solve returned new proper intersections')
    return best[3],dict(method='GENERATED_CONTACT_SOURCE_OVERLAY_QP_RESEARCH' if prevent_new_contacts else 'L1_ELASTIC_SOURCE_OVERLAY_SEPARATING_AXIS_QP_RESEARCH',initial=initial,chosen_reversed=best[0],chosen_crossings=best[1],chosen_source_rms=best[2],history=history,reason=reason,elapsed_seconds=time.monotonic()-start,
        prevent_new_contacts=prevent_new_contacts,new_pairs=len(final_contacts-allowed_contacts),hard_untangle_faces=int(untangle.sum()),generated_contact_pairs=len(generated_contacts),generated_contact_watchlist=[list(pair) for pair in sorted(generated_contacts)],generated_radius_constraints=sum(map(len,radius_cuts.values())),immutable_contact_pairs=len(immutable_contacts),active_contact_faces=int(active_faces.sum()),solver_contract=contract,prevented_trials=prevented_trials,
        contact_held_vertex_ids=sorted(contact_holds),max_movement=float(np.linalg.norm(best[3]-reference,axis=1).max()),fixed_vertices_exact=bool(np.array_equal(best[3][~free],reference[~free])),source_vertex_projection_exact=False,accepted=False,production_ready=False)
