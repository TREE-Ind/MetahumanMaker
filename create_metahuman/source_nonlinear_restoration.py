"""Nonlinear source redistribution with independently restored endpoint limits.

Internal search points are proposals. Only the polished endpoint may leave
this module, and callers must still run coverage and native storage checks.
"""
import time
import numpy as np
from scipy.optimize import minimize
from .surface_contacts import crossings
from .constrained_surface_fit import _axes
from .source_angle_contract import angular_values


def propose(reference,tri,free,normals,minimum_area,lap,following,projector,*,seconds=60.,statistics=None,progress=None,cancelled=None,reverse_projection=None,alignment_reference=None,surface_interior=None,initial_proposal=None,point_distribution=None,source_tube=False):
    stats={} if statistics is None else statistics
    began=time.monotonic();deadline=began+seconds;unit=float(np.linalg.norm(np.ptp(reference,axis=0)))*.01
    ids=np.flatnonzero(free);active=np.flatnonzero(np.any(free[tri],axis=1));angles=following.original
    if not np.array_equal(ids,projector.ids):raise ValueError('Nonlinear source observations must match the editable vertex order')
    protected=angles['face_ids'];scope=following.scope;bound=projector.limits[ids]/unit
    coherence_ids=np.flatnonzero(scope)
    from .source_alignment_energy import evaluate as alignment_energy
    from .source_direction_support import outward_corners
    stats['support_predicate']='COHERENCE_OR_OUTWARD_OBSERVED_CORNERS_V1'
    rest=reference if alignment_reference is None else alignment_reference
    rp=rest[tri]
    alignment_weights=np.linalg.norm(np.cross(rp[:,1]-rp[:,0],rp[:,2]-rp[:,0]),axis=1)
    alignment_weights*=scope&np.any(free[tri],axis=1)
    alignment_weights/=max(float(alignment_weights.sum()),1e-30)
    stats['source_alignment_objective']='FIXED_CANONICAL_AREA_MOVING_NORMAL_AGREEMENT_V1'
    tol=float(np.linalg.norm(np.ptp(reference,axis=0)))*1e-6
    contact_faces=np.any(free[tri],axis=1)
    stats['contact_scope']=dict(method='ALL_PAIRS_TOUCHING_MOVABLE_TRIANGLES',faces=int(contact_faces.sum()),fixed_coordinates_constructed_exact=True)
    allowed=crossings(reference,tri,pairs=True,tolerance=tol,active_faces=contact_faces);contacts=[];contact_axes=[]
    reverse_matrix=None
    if reverse_projection is not None:
        reverse_matrix=reverse_projection['matrix'];reverse_area=float(reverse_projection['source_area'])
        reverse_gradient=reverse_matrix@reference-reverse_projection['load']
        reverse_constant=float(np.sum(reference*(reverse_matrix@reference))-2*np.sum(reference*reverse_projection['load'])+reverse_projection['constant'])
        stats['original_source_area_reverse_objective']=True
    def area_gradient(p,direction):
        return np.stack((np.cross(p[:,1]-p[:,2],direction),np.cross(p[:,2]-p[:,0],direction),np.cross(p[:,0]-p[:,1],direction)),axis=1)
    def geometry(points):
        _,field,D,vertex_field,J=following._derivatives(points,vertex_field=True)
        n=normals.copy();n[scope]=field[scope]
        p=points[tri];C=np.cross(p[:,1]-p[:,0],p[:,2]-p[:,0])
        correction=np.einsum('fi,fkij->fkj',C,D);correction[~scope]=0
        signed=np.sum(C*n,axis=1);gradient=area_gradient(p,n)+correction
        size=np.linalg.norm(vertex_field[tri].sum(1),axis=1)
        coherence_gradient=np.einsum('fi,fkij->fkj',field,J[tri])
        outward,outward_gradient=outward_corners(points,tri[coherence_ids],vertex_field,J)
        return p,C,n,signed,gradient,correction,size,coherence_gradient,outward,outward_gradient
    p,C,n,signed,gradient,correction,size,coherence_gradient,outward,outward_gradient=geometry(reference)
    _,_,initial_D=following._derivatives(reference)
    initial_alignment,_=alignment_energy(reference,tri,n,initial_D,alignment_weights)
    distribution_matrix=None
    if point_distribution is not None:
        distribution_initial=.5*float(point_distribution['report']['before_energy'])
        if distribution_initial>unit*unit*1e-12:
            distribution_matrix=point_distribution['matrix']
            distribution_gradient=distribution_matrix@reference.ravel()
            distribution_strength=max(initial_alignment,1e-12)/(distribution_initial/(unit*unit))
            stats['tangent_point_distribution']=dict(**point_distribution['report'],balanced_strength=float(distribution_strength),fixed_reference_metric=True,proposal_only=True)
    interior_strength=0.
    if surface_interior is not None:
        initial_interior,_=surface_interior.energy(reference,projector.query)
        interior_strength=max(initial_alignment,1e-12)/max(initial_interior/(unit*unit),1e-12)
        stats['triangle_interior_objective']=dict(method='FIXED_CANONICAL_AREA_SEVEN_POINT_SOURCE_DISTANCE_V1',
            triangles=len(surface_interior.ids),samples=len(surface_interior.weights),strength=float(interior_strength))
    reverse_strength=1.
    if reverse_matrix is not None:
        initial_reverse=max(reverse_constant/(2*reverse_area*unit*unit),1e-12)
        # Compare relative progress of the two original-source objectives.
        # Without this fixed baseline scaling, dimensionless normal error
        # dominated squared metric coverage, rewarding smoother but less
        # faithful surfaces. The normalization is frozen for this operation.
        reverse_strength=max(initial_alignment,1e-12)/initial_reverse
        stats['reverse_objective_strength']=float(reverse_strength)
        stats['objective_balance']='IMMUTABLE_BASELINE_ALIGNMENT_AND_COVERAGE_V1'
    coherence_floor=np.minimum(2.4+1e-5,(size[coherence_ids]+2.4)*.5)
    scale=np.maximum(np.linalg.norm(gradient,axis=(1,2)),1e-8)
    floor=np.where(signed>0,np.minimum(minimum_area,signed*.5),minimum_area)
    flat_before,flat_supported,flat_vertices_before,_=following.flat_directions(reference,refresh=False,return_field=True)
    flat_ids=np.flatnonzero(flat_supported&np.any(free[tri],axis=1))
    flat_signed=np.sum(C*flat_before,axis=1)
    flat_floor=np.where(flat_signed>0,np.minimum(minimum_area,flat_signed*.5),minimum_area)
    flat_scale=np.maximum(np.linalg.norm(area_gradient(p,flat_before),axis=(1,2)),1e-8)
    flat_cosine=flat_signed/np.maximum(np.linalg.norm(C,axis=1),1e-30)
    quality_ids=flat_ids[flat_cosine[flat_ids]>=.25]
    quality_floor=np.minimum(.25+1e-5,(flat_cosine[quality_ids]+.25)*.5)
    initial_corner=np.einsum('fki,fi->fk',flat_vertices_before[tri[quality_ids]],C[quality_ids]/np.maximum(np.linalg.norm(C[quality_ids],axis=1)[:,None],1e-30)).min(1)
    quality_corner_floor=np.minimum(.25+1e-5,np.maximum(.25,(initial_corner+.25)*.5))
    stats['protected_flat_quality_faces']=len(quality_ids)
    artist_rows=3*len(ids) if source_tube else 0
    lambdas=np.zeros(len(ids)+artist_rows+len(active)+len(protected)+len(coherence_ids)+len(flat_ids)+len(quality_ids));rho=10.;history=[]
    stats['source_tube_parameterization']=bool(source_tube)
    last={};y=np.zeros(len(ids)*3);best=None;reason='stage_limit'
    class Deadline(BaseException):pass
    def evaluate(y):
        if cancelled and cancelled():raise RuntimeError('Nonlinear source restoration cancelled')
        if time.monotonic()>=deadline:raise Deadline()
        if 'y' in last and np.array_equal(y,last['y']):return last['value']
        points=reference.copy();points[ids]+=y.reshape(-1,3)*unit
        if source_tube:
            from .source_tube_projection import project
            raw_record=projector.query(points[ids])
            facets=following.source[following.source_triangles[raw_record['source_triangle_ids']]]
            points[ids],tube_jacobian=project(points[ids],facets,projector.source_radius)
        p,C,n,signed,signed_gradient,correction,size,coherence_gradient,outward,outward_gradient=geometry(points)
        record=projector.query(points[ids]);distance=record['residual']
        source_g=(distance-projector.source_radius)/unit
        source_grad=(points[ids]-record['surface_points'])/np.maximum(distance[:,None],1e-30)/unit
        artist_g=((np.abs(points[ids]-reference[ids])-projector.limits[ids])/unit).ravel() if source_tube else []
        orient_g=(floor[active]-signed[active])/(scale[active]*unit)
        orient_grad=-signed_gradient[active]/(scale[active,None,None]*unit)
        length=np.linalg.norm(C[protected],axis=1);cosine=angles['minimum_cosine']
        angle_g=(cosine*length-np.sum(C[protected]*n[protected],axis=1))/(scale[protected]*unit)
        direction=cosine[:,None]*C[protected]/np.maximum(length[:,None],1e-30)-n[protected]
        angle_grad=(area_gradient(p[protected],direction)-correction[protected])/(scale[protected,None,None]*unit)
        contact_g=[]
        for pair,axis in zip(contacts,contact_axes):
            a,b=tri[list(pair)];gap=(points[a]@axis)[:,None]-(points[b]@axis)[None,:]
            contact_g.extend((gap/unit).ravel())
        coherence_g=coherence_floor-size[coherence_ids]
        outward_g=.25+1e-5-outward
        use_outward=outward_g<coherence_g
        support_gradient=np.where(use_outward[:,None,None],-outward_gradient,-coherence_gradient[coherence_ids])
        coherence_g=np.minimum(coherence_g,outward_g)
        flat,flat_supported_now,flat_vertices,_=following.flat_directions(points,return_field=True)
        flat_g=(flat_floor[flat_ids]-np.sum(C[flat_ids]*flat[flat_ids],axis=1))/(flat_scale[flat_ids]*unit)
        flat_gradient=-area_gradient(p[flat_ids],flat[flat_ids])/(flat_scale[flat_ids,None,None]*unit)
        quality_length=np.maximum(np.linalg.norm(C[quality_ids],axis=1),1e-30)
        raw_corners=flat_vertices[tri[quality_ids]];raw_size=np.linalg.norm(raw_corners.sum(1),axis=1)
        corner_dot=np.einsum('fki,fi->fk',raw_corners,C[quality_ids]);worst=np.argmin(corner_dot,axis=1)
        use_corner=raw_size<2.4
        quality_normal=np.where(use_corner[:,None],raw_corners[np.arange(len(quality_ids)),worst],flat[quality_ids])
        quality_minimum=np.where(use_corner,quality_corner_floor,quality_floor)
        quality_g=(quality_minimum*quality_length-np.sum(C[quality_ids]*quality_normal,axis=1))/(flat_scale[quality_ids]*unit)
        quality_direction=quality_minimum[:,None]*C[quality_ids]/quality_length[:,None]-quality_normal
        quality_gradient=area_gradient(p[quality_ids],quality_direction)/(flat_scale[quality_ids,None,None]*unit)
        g=np.r_[source_g,artist_g,orient_g,angle_g,coherence_g,flat_g,quality_g,contact_g]
        w=np.maximum(lambdas+rho*g,0)
        displacement=points-reference;smooth=lap@displacement/unit
        value=.5e-3*float(y@y)+.5e-2*float(np.sum(smooth*smooth))
        _,_,field_D=following._derivatives(points)
        alignment_value,alignment_gradient=alignment_energy(points,tri,n,field_D,alignment_weights)
        value+=alignment_value
        if distribution_matrix is not None:
            flat_delta=displacement.ravel();distribution_delta=distribution_matrix@flat_delta
            value+=distribution_strength*(distribution_initial+float(flat_delta@distribution_gradient)+.5*float(flat_delta@distribution_delta))/(unit*unit)
        if surface_interior is not None:
            interior_value,interior_gradient=surface_interior.energy(points,projector.query)
            value+=interior_strength*interior_value/(unit*unit)
        if reverse_matrix is not None:
            reverse_delta=reverse_matrix@displacement
            value+=reverse_strength*.5*float(reverse_constant+2*np.sum(displacement*reverse_gradient)+np.sum(displacement*reverse_delta))/(reverse_area*unit*unit)
        value+=float((np.sum(w*w)-np.sum(lambdas*lambdas))/(2*rho))
        grad=1e-2*(lap.T@smooth)/unit
        grad+=alignment_gradient
        if distribution_matrix is not None:grad+=distribution_strength*(distribution_gradient+distribution_delta).reshape(-1,3)/(unit*unit)
        if surface_interior is not None:grad+=interior_strength*interior_gradient/(unit*unit)
        if reverse_matrix is not None:grad+=reverse_strength*(reverse_delta+reverse_gradient)/(reverse_area*unit*unit)
        split=len(ids);grad[ids]+=source_grad*w[:split,None]
        if source_tube:
            grad[ids]+=np.sign(points[ids]-reference[ids])*w[split:split+artist_rows].reshape(-1,3)/unit
            split+=artist_rows
        def accumulate(faces,derivatives,weights):
            for corner in range(3):np.add.at(grad,tri[faces,corner],derivatives[:,corner]*weights[:,None])
        accumulate(active,orient_grad,w[split:split+len(active)]);split+=len(active)
        accumulate(protected,angle_grad,w[split:split+len(protected)]);split+=len(protected)
        accumulate(coherence_ids,support_gradient,w[split:split+len(coherence_ids)]);split+=len(coherence_ids)
        accumulate(flat_ids,flat_gradient,w[split:split+len(flat_ids)]);split+=len(flat_ids)
        accumulate(quality_ids,quality_gradient,w[split:split+len(quality_ids)]);split+=len(quality_ids)
        for pair,axis in zip(contacts,contact_axes):
            a,b=tri[list(pair)];weights=w[split:split+9].reshape(3,3);split+=9
            np.add.at(grad,a,weights.sum(1)[:,None]*axis/unit)
            np.add.at(grad,b,-weights.sum(0)[:,None]*axis/unit)
        selected_gradient=grad[ids]
        if source_tube:selected_gradient=np.einsum('nji,nj->ni',tube_jacobian,selected_gradient)
        result=(value,selected_gradient.ravel()*unit+1e-3*y,g,points)
        last.update(y=y.copy(),value=result)
        return result
    proposal=reference.copy()
    if initial_proposal is not None:
        initial_proposal=np.asarray(initial_proposal,dtype=float)
        if initial_proposal.shape!=reference.shape or not np.isfinite(initial_proposal).all():raise ValueError('Invalid source-chart initialization')
        original_value=evaluate(y)[0];best_value=original_value;trials=[];selected_fraction=None
        for fraction in (1.,.5,.25,.125):
            # Query prospective facets only after limiting the chart to the
            # original artist box. A distant unconstrained chart facet can
            # otherwise have no intersection with that box at all.
            seed=reference.copy()
            seed[ids]+=np.clip((initial_proposal[ids]-reference[ids])*fraction,-projector.limits[ids],projector.limits[ids])
            trial=dict(fraction=fraction,source_retracted=False)
            try:
                try:
                    seed=projector.retract(seed);trial['source_retracted']=True
                except ValueError as exc:
                    # AL search already permits internally infeasible source
                    # observations. Retain the artist-bounded chart as a
                    # hypothesis and charge its full source/angle penalties;
                    # only an independently polished endpoint can be offered.
                    trial['retraction_reason']=str(exc)
                seed_y=np.clip((seed[ids]-reference[ids])/unit,-bound,bound).ravel()
                seed_value=evaluate(seed_y)[0]
                trial['objective']=float(seed_value);trials.append(trial)
                if np.isfinite(seed_value) and seed_value<best_value:
                    y=seed_y;best_value=seed_value;selected_fraction=fraction
            except ValueError as exc:trial['reason']=str(exc);trials.append(trial)
        stats['chart_initialization']=dict(selected=selected_fraction is not None,fraction=selected_fraction,
            original_objective=float(original_value),chosen_objective=float(best_value),trials=trials,original_bounds_retained=True)
    for stage in range(10):
        last.clear()
        try:
            result=minimize(lambda z:evaluate(z)[:2],y,jac=True,method='L-BFGS-B',bounds=list(zip(-bound.ravel(),bound.ravel())),
                # Update constraint multipliers within the fixed search
                # budget. A long first inner solve spent nearly the entire
                # budget improving alignment while source/angle penalties
                # still had their weakest initial weight.
                options=dict(maxiter=16,maxls=20,ftol=1e-14,gtol=1e-9,maxcor=16))
            y=result.x;last.clear();value,gradient,g,points=evaluate(y)
        except Deadline:reason='deadline';break
        proposal=points.copy()
        actual=crossings(points,tri,pairs=True,tolerance=tol,active_faces=contact_faces);new=actual-allowed;endpoint=points
        try:
            endpoint=projector.retract(points);contract,n=following(endpoint);cosine,residual=angular_values(endpoint,tri,contract)
            p=endpoint[tri];signed=np.sum(np.cross(p[:,1]-p[:,0],p[:,2]-p[:,0])*np.where(scope[:,None],n,normals),axis=1)
            endpoint_contacts=crossings(endpoint,tri,pairs=True,tolerance=tol,active_faces=contact_faces)
            feasible=bool(residual.max(initial=0)<=1e-9 and np.maximum(contract['minimum_cosine']-cosine,0).max(initial=0)<=1e-7 and
                not np.any(signed[active]<=0) and not (endpoint_contacts-allowed) and np.array_equal(endpoint[~free],reference[~free]))
            endpoint_error=float(residual.max(initial=0));back=int(np.count_nonzero(signed[active]<=0))
            if feasible:best=endpoint.copy()
        except ValueError as exc:
            feasible=False;endpoint_error=None;back=None;field_error=str(exc)
        row=dict(stage=stage+1,rho=rho,iterations=int(result.nit),objective=float(value),maximum_constraint_violation=float(g.max(initial=0)),
            source_violation_m=float(g[:len(ids)].max(initial=0)*unit),new_pairs=len(new),endpoint_angle_residual=endpoint_error,endpoint_backward_faces=back,endpoint_feasible=feasible)
        if not feasible and endpoint_error is None:row['endpoint_error']=field_error
        history.append(row);print(row,flush=True)
        stats.update(history=history,reason=reason,elapsed_seconds=time.monotonic()-began)
        if progress:progress('Redistributing the source surface: pass '+str(stage+1),(stage+1)/10)
        if best is not None:reason='feasible_endpoint';break
        lambdas=np.maximum(0,lambdas+rho*g)
        for pair in sorted(new):
            if pair not in contacts:contacts.append(pair)
        contact_axes=[]
        for pair in contacts:
            axes,gaps,costs=_axes(points,tri[np.array(pair)][None],0.)
            contact_axes.append(axes[0,int(np.argmin(costs[0]))])
        core=len(ids)+artist_rows+len(active)+len(protected)+len(coherence_ids)+len(flat_ids)+len(quality_ids);lambdas=np.r_[lambdas[:core],np.zeros(9*len(contacts))]
        rho*=5
    stats.update(history=history,reason=reason,elapsed_seconds=time.monotonic()-began,endpoint_feasible=best is not None)
    return proposal if best is None else best


def reconstruct(reference,tri,free,normals,minimum_area,lap,following,projector,*,seconds=120.,progress=None,cancelled=None,acceptance_angles=None,reverse_projection=None,alignment_reference=None,surface_interior=None,initial_proposal=None,point_distribution=None,source_tube=False):
    from .source_nonlinear_polish import polish
    began=time.monotonic();search={};refinement={}
    def update_search(label,value):
        if progress:progress(label,value*.7)
    proposal=propose(reference,tri,free,normals,minimum_area,lap,following,projector,
        seconds=min(60.,seconds*2/3),statistics=search,progress=update_search,cancelled=cancelled,reverse_projection=reverse_projection,alignment_reference=alignment_reference,surface_interior=surface_interior,initial_proposal=initial_proposal,point_distribution=point_distribution,source_tube=source_tube)
    remaining=seconds-(time.monotonic()-began)
    if remaining<=0:output=reference.copy();refinement.update(valid=False,reason='deadline')
    else:
        def update_polish(label,value):
            if progress:progress(label,.7+value*.3)
        try:
            output=polish(reference,proposal,tri,free,normals,minimum_area,following,projector,
                seconds=remaining,statistics=refinement,progress=update_polish,cancelled=cancelled,acceptance_angles=acceptance_angles)
        except ValueError as exc:
            output=reference.copy();refinement.update(valid=False,reason='unavailable_endpoint',diagnostic=str(exc))
        # A proposal can lie beyond the useful range of its local facet and
        # angle model. Start from the certified reference and approach that
        # same target through feasible steps instead of discarding it whole.
        # This shares the existing deadline and changes no endpoint limits.
        remaining=seconds-(time.monotonic()-began)
        if not refinement.get('valid') and remaining>0:
            failed_refinement=refinement;refinement={}
            try:
                output=polish(reference,reference.copy(),tri,free,normals,minimum_area,following,projector,
                    seconds=remaining,statistics=refinement,progress=update_polish,cancelled=cancelled,
                    acceptance_angles=acceptance_angles,target_points=proposal)
            except ValueError as exc:
                output=reference.copy();refinement.update(valid=False,reason='unavailable_feasible_advance',diagnostic=str(exc))
            refinement['failed_proposal_refinement']=failed_refinement
    span=float(np.linalg.norm(np.ptp(reference,axis=0)))
    before=crossings(reference,tri,pairs=True,tolerance=span*1e-6);after=crossings(output,tri,pairs=True,tolerance=span*1e-6)
    if after-before:raise AssertionError('Nonlinear restoration returned new proper intersections')
    return output,dict(method='NONLINEAR_ORIGINAL_SOURCE_RESTORATION_WITH_CONIC_POLISH_V1',initial=dict(crossings=len(before)),
        chosen_crossings=len(after),new_pairs=0,search=search,polish=refinement,elapsed_seconds=time.monotonic()-began,
        endpoint_limits_restored=bool(refinement.get('valid')),accepted=False,production_ready=False)
