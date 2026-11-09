"""Source-boundary normal continuation, explicitly an inferred shape prior."""
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve

def normal_continuation(points,triangles,metric,source_normals,observed,free):
    points=np.asarray(points,dtype=float);tri=np.asarray(triangles,dtype=np.int64);metric=np.asarray(metric,dtype=float);field=np.array(source_normals,dtype=float,copy=True);observed=np.asarray(observed,dtype=bool);free=np.asarray(free,dtype=bool);n=len(points)
    if points.shape!=(n,3) or metric.shape!=points.shape or field.shape!=points.shape or observed.shape!=(n,) or free.shape!=(n,) or not all(np.isfinite(a).all() for a in (points,metric,field)):raise ValueError('Invalid source normal continuation')
    if tri.ndim!=2 or tri.shape[1]!=3 or not len(tri) or tri.min()<0 or tri.max()>=n:raise ValueError('Invalid normal continuation triangles')
    p=points[tri];cross=np.cross(p[:,1]-p[:,0],p[:,2]-p[:,0]);native=np.zeros_like(points)
    for i in range(3):np.add.at(native,tri[:,i],cross)
    native/=np.maximum(np.linalg.norm(native,axis=1)[:,None],1e-30);field[~free&~observed]=native[~free&~observed]
    known=observed|~free;unknown=~known;length=np.linalg.norm(field[known],axis=1)
    if np.any(length<=1e-12):raise ValueError('Continuation boundary has undefined normals')
    field[known]/=length[:,None]
    edges=np.unique(np.sort(np.vstack((tri[:,[0,1]],tri[:,[1,2]],tri[:,[2,0]])),axis=1),axis=0);length=np.linalg.norm(metric[edges[:,0]]-metric[edges[:,1]],axis=1)
    if np.any(length<=0):raise ValueError('Degenerate continuation metric')
    weights=1/length;adj=sparse.coo_matrix((np.r_[weights,weights],(np.r_[edges[:,0],edges[:,1]],np.r_[edges[:,1],edges[:,0]])),shape=(n,n)).tocsr();lap=sparse.diags(np.asarray(adj.sum(1)).ravel())-adj
    from scipy.sparse.csgraph import connected_components
    _,labels=connected_components(adj,directed=False)
    if np.any(~np.isin(labels[unknown],np.unique(labels[known]))):raise ValueError('Missing normal component has no boundary')
    if unknown.any():field[unknown]=spsolve(lap[unknown][:,unknown].tocsc(),-lap[unknown][:,known]@field[known])
    length=np.linalg.norm(field,axis=1);valid=length>1e-5;field[valid]/=length[valid,None];field[~valid]=0.
    return field,dict(method='SOURCE_BOUNDARY_HARMONIC_NORMAL_PRIOR_V1',inferred_vertices=int(unknown.sum()),undefined_inferred_vertices=int((unknown&~valid).sum()),inferred_normals_are_not_source_evidence=True)


def gap_deformation_prior(reference,triangles,metric,inferred,*,strength=8.):
    """Discourage noisy deformation across an unmeasured surface strip.

    Preserve the input's displacement stencil, not absolute gap positions.
    A stencil touching inferred vertices also controls the join's slope.
    This is a shape prior; it cannot certify the missing source geometry.
    """
    import igl
    reference=np.asarray(reference,dtype=float);tri=np.asarray(triangles,dtype=np.int64);metric=np.asarray(metric,dtype=float);inferred=np.asarray(inferred,dtype=bool);n=len(reference)
    if reference.shape!=(n,3) or metric.shape!=reference.shape or inferred.shape!=(n,) or not np.isfinite(reference).all() or not np.isfinite(metric).all() or not np.isfinite(strength) or strength<=0:raise ValueError('Invalid inferred deformation prior')
    if tri.ndim!=2 or tri.shape[1]!=3 or not len(tri) or tri.min()<0 or tri.max()>=n:raise ValueError('Invalid gap deformation triangles')
    lap=igl.cotmatrix(metric,tri).tocsr();lap.setdiag(0);lap.eliminate_zeros();lap.setdiag(-np.asarray(lap.sum(1)).ravel());norm=np.asarray(abs(lap).sum(1)).ravel()
    if not np.isfinite(lap.data).all():raise ValueError('Invalid gap deformation metric')
    lap=sparse.diags(1/np.maximum(norm,1e-30))@lap;rows=np.asarray(lap[:,inferred].getnnz(axis=1)).ravel()>0;stencil=lap[rows].tocsr()
    p=metric[tri];area=np.linalg.norm(np.cross(p[:,1]-p[:,0],p[:,2]-p[:,0]),axis=1)*.5;mass=np.zeros(n)
    for i in range(3):np.add.at(mass,tri[:,i],area/3)
    edges=np.unique(np.sort(np.vstack((tri[:,[0,1]],tri[:,[1,2]],tri[:,[2,0]])),axis=1),axis=0);square=np.sum((metric[edges[:,0]]-metric[edges[:,1]])**2,axis=1);sums=np.zeros(n);degree=np.zeros(n)
    for i in range(2):np.add.at(sums,edges[:,i],square);np.add.at(degree,edges[:,i],1)
    if np.any(mass[rows]<=0) or np.any(sums[rows]<=0):raise ValueError('Gap deformation requires nondegenerate reference area')
    weight=strength*mass[rows]*degree[rows]/sums[rows];matrix=(stencil.T@sparse.diags(weight)@stencil).tocsr();load=np.asarray(matrix@reference)
    return dict(matrix=matrix,load=load,mass=np.zeros(n),constant=float(np.sum(reference*load)),report=dict(method='CANONICAL_DISPLACEMENT_STENCIL_GAP_PRIOR_V1',inferred_vertices=int(inferred.sum()),stencil_rows=int(rows.sum()),strength=float(strength),absolute_gap_positions_fixed=False,inferred_shape_is_not_source_evidence=True))


def refine(points,triangles,metric,source_normals,normal_observed,pinned,box_origin,box_limits,source_projector,projection,*,supported_faces,time_limit=40.,iterations=16,progress=None,cancelled=None,orientation_reference=None,minimum_source_cosine=.25):
    """Fit source coverage and soft continued tangents under original limits.

    Only actual source-supported faces provide hard angular constraints.
    Inferred normals guide the objective; they cannot certify source fidelity.
    The parent must run the usual native storage and source-coverage gates.
    """
    from .surface_alignment import tangent_objective,angle_contract,redistribution_objective
    from .constrained_surface_fit import solve
    points=np.asarray(points,dtype=float);tri=np.asarray(triangles,dtype=np.int64);free=~np.asarray(pinned,dtype=bool)
    field,report=normal_continuation(points,tri,metric,source_normals,normal_observed,free)
    facefield=field[tri].sum(1);length=np.linalg.norm(facefield,axis=1);directions=facefield/np.maximum(length[:,None],1e-30)
    prior=np.any(free[tri],axis=1)&np.all(np.linalg.norm(field[tri],axis=2)>0,axis=1)&(length>1.5)
    tangent=tangent_objective(points,tri,directions,prior,metric,strength=8.)
    # Normal alignment alone leaves an inherited clump of vertices on the
    # source. Redistribute only complete, coherent measured one-rings along
    # their source tangents. Missing regions retain the displacement prior;
    # source distance caps, pins and the native endpoint gates still apply.
    redistribution=redistribution_objective(points,tri,source_normals,np.asarray(normal_observed,dtype=bool),metric,free.astype(float))
    tangent['matrix']=(tangent['matrix']+redistribution['matrix']).tocsr()
    angles=angle_contract(points,tri,directions,supported_faces,metric,minimum=minimum_source_cosine,reserve=max(.5,minimum_source_cosine+1e-4),preserve_aligned=True)
    orientation=points if orientation_reference is None else np.asarray(orientation_reference,dtype=float)
    if orientation.shape!=points.shape or not np.isfinite(orientation).all():raise ValueError('Invalid fixed continuity orientation reference')
    p=orientation[tri];cross=np.cross(p[:,1]-p[:,0],p[:,2]-p[:,0]);area=np.linalg.norm(cross,axis=1);normals=cross/np.maximum(area[:,None],1e-30)
    span=float(np.linalg.norm(np.ptp(orientation,axis=0)))
    def checkpoint(_,detail):
        if cancelled and cancelled():raise RuntimeError('Surface continuity refinement cancelled')
        if progress:progress('Refining surface continuity',min(.95,len(detail.get('history',[]))/max(int(iterations),1)))
    if cancelled and cancelled():raise RuntimeError('Surface continuity refinement cancelled')
    result,numerical=solve(points,points,tri,projection,free,normals,area*.1,np.zeros(len(points)),sparse.eye(len(points)),iterations=int(iterations),time_limit=float(time_limit),checkpoint=checkpoint,prevent_new_contacts=True,source_projector=source_projector,movement_boxes=box_limits,movement_box_origin=box_origin,contact_margin=span*1e-6,differential_target=np.zeros_like(points),differential_weight=np.zeros(len(points)),joint_projection='conic',preserve_orientations=True,compact_proposal=True,conic_segment_backtracking=True,conic_elastic_model=True,localize_contact_stalls=True,tangent_objective=tangent,source_angles=angles,restore_source_angles=True,step_size_fraction=.01)
    np.testing.assert_array_equal(result[~free],points[~free])
    def energy(value):
        v=value.ravel();return float(np.sum(value*(projection['matrix']@value))-2*np.sum(projection['load']*value)+projection['constant']+v@(tangent['matrix']@v))
    before,after=energy(points),energy(result);accepted=after<before-max(abs(before)*1e-10,1e-15)
    if not accepted:result=points.copy()
    redistribution['report']['after_energy']=float(result.ravel()@(redistribution['matrix']@result.ravel()))
    report['angle_protection']=dict(method='PRESERVE_CURRENT_SUPPORTED_SOURCE_ALIGNMENT_V1',faces=len(angles['face_ids']),faces_stronger_than_reserve=int(np.count_nonzero(angles['minimum_cosine']>angles['reserve'])),cosine_roundoff_margin=1e-4,inferred_faces_are_not_source_evidence=True,requires_native_validation=True)
    numerical['objective_kind']='SOURCE_COVERAGE_NORMAL_CONTINUITY_AND_MEASURED_TANGENT_REDISTRIBUTION'
    report.update(method='SOURCE_CONSTRAINED_NORMAL_CONTINUITY_V2',normal_prior_method=report['method'],source_redistribution=redistribution['report'],accepted=accepted,before_combined_energy=before,after_combined_energy=after,soft_prior_faces=int(prior.sum()),hard_source_angle_faces=len(angles['face_ids']),source_distance_bounds_unchanged=True,pinned_coordinates_exact=True,max_movement_m=float(np.linalg.norm(result-points,axis=1).max(initial=0)),numerical=numerical)
    return result,report


def refine_with_updates(points,triangles,metric,source_normals,normal_observed,pinned,box_origin,box_limits,source_projector,projection,*,supported_faces,refresh,assess,max_passes=3,time_limit=90.,progress=None,cancelled=None,minimum_source_cosine=.25,prepare_endpoint=None):
    """Update correspondence objectives under one immutable constraint snapshot.

    Each stage is assessed against the same original source quadrature and
    normal scope, rather than against its refreshed optimization objective.
    The selected proposal still requires the parent's native transaction.
    """
    import time
    from .vertex_contract import fingerprint
    if not isinstance(max_passes,int) or not 1<=max_passes<=4 or not np.isfinite(time_limit) or time_limit<=0:raise ValueError('Choose one to four bounded continuity passes')
    original=np.asarray(points,dtype=float).copy();working=original.copy();best=None;best_score=float('inf');best_detail=None;best_valid=False;first_detail=None;rows=[];started=time.monotonic()
    def constraint_snapshot():
        return {key:fingerprint(np.asarray(value)) for key,value in dict(metric=metric,triangles=triangles,normal_observed=normal_observed,pinned=pinned,artist_origin=box_origin,artist_limits=box_limits,source_origin=source_projector.reference,source_limits=source_projector.limits,source_vertex_ids=source_projector.ids,source_radius=source_projector.source_radius).items()}
    authority=constraint_snapshot()
    gap=gap_deformation_prior(original,triangles,metric,~np.asarray(pinned,dtype=bool)&~np.asarray(normal_observed,dtype=bool))
    from .surface_sample_transport import propose
    working,transport=propose(original,triangles,metric,source_normals,normal_observed,pinned,source_projector,assess,prepare_endpoint=prepare_endpoint,time_limit=min(12.,time_limit*.2),cancelled=cancelled)
    transport_seed=working.copy()
    if constraint_snapshot()!=authority:raise ValueError('Source sample transport changed constraint authority')
    for index in range(max_passes):
        if cancelled and cancelled():raise RuntimeError('Surface continuity refinement cancelled')
        remaining=time_limit-(time.monotonic()-started)
        if remaining<2:break
        evidence={}
        if index or transport['accepted']:source_normals,projection,evidence=refresh(working)
        def update(label,value):
            if progress:progress('Continuity pass '+str(index+1)+': '+label,(index+value)/max_passes)
        stabilized=dict(matrix=projection['matrix']+gap['matrix'],load=projection['load']+gap['load'],mass=projection['mass'],constant=projection['constant']+gap['constant'])
        candidate,detail=refine(working,triangles,metric,source_normals,normal_observed,pinned,box_origin,box_limits,source_projector,stabilized,
            supported_faces=supported_faces,time_limit=min(30.,max(1.,time_limit-(time.monotonic()-started))),iterations=16,progress=update,cancelled=cancelled,orientation_reference=original,minimum_source_cosine=minimum_source_cosine)
        if prepare_endpoint is not None:
            candidate,prepared=prepare_endpoint(candidate);detail['endpoint_preparation']=prepared
        if first_detail is None:first_detail=detail
        quality=assess(candidate);valid=bool(quality['valid']);score=float(quality['score'])
        if constraint_snapshot()!=authority:raise ValueError('A refreshed continuity pass changed the original constraint authority')
        if not np.isfinite(score):raise ValueError('Invalid continuity proposal score')
        rows.append(dict(pass_index=index+1,accepted_objective=detail['accepted'],quality=quality,updated_correspondence=evidence,solver=detail))
        # Retain the first ordinary proposal as the existing native-backtrack
        # fallback. A later proposal may replace it only after fixed-scope checks.
        if best is None or valid and (not best_valid or score<best_score-1e-10):
            best=candidate.copy();best_score=score;best_detail=detail;best_valid=valid;chosen=index+1
        movement=float(np.linalg.norm(candidate-working,axis=1).max(initial=0))
        if not detail['accepted'] or movement<=max(float(np.linalg.norm(np.ptp(original,axis=0)))*1e-8,1e-12):break
        working=candidate
    if best is None:raise ValueError('Continuity refinement exceeded its budget before a proposal')
    if transport['accepted'] and (not best_valid or transport['quality']['score']<best_score):
        best=transport_seed;best_valid=True;best_score=float(transport['quality']['score']);best_detail=first_detail;chosen=0
    endpoint_search=[];endpoint_fraction=1.
    if not best_valid and prepare_endpoint is not None:
        recovery_deadline=time.monotonic()+8.;proposal=best.copy()
        for fraction in (.5,.25,.125,.0625):
            if time.monotonic()>=recovery_deadline:break
            if cancelled and cancelled():raise RuntimeError('Surface continuity refinement cancelled')
            candidate=original+(proposal-original)*fraction
            candidate,prepared=prepare_endpoint(candidate);quality=assess(candidate)
            if constraint_snapshot()!=authority:raise ValueError('Continuity endpoint recovery changed the original constraint authority')
            endpoint_search.append(dict(fraction=fraction,quality=quality,endpoint_preparation=prepared))
            if quality['valid']:
                best=candidate;best_valid=True;best_score=float(quality['score']);endpoint_fraction=fraction;break
    numerical=dict(best_detail['numerical']);numerical['initial']=first_detail['numerical']['initial'];numerical['objective_kind']='REFRESHED_SOURCE_COVERAGE_NORMALS_AND_GAP_DEFORMATION';numerical['stage_objectives_are_not_directly_comparable']=True
    report=dict(best_detail,method='FIXED_AUTHORITY_REFRESHED_SURFACE_CONTINUITY_V1',passes=rows,selected_pass=chosen,fixed_scope_valid=best_valid,requires_native_backtracking=not best_valid,
        prepared_reference_endpoint_search=endpoint_search,selected_endpoint_fraction=endpoint_fraction,
        source_sample_transport=transport,gap_deformation_prior=gap['report'],original_constraint_snapshot_preserved=True,constraint_snapshot=authority,vertex_evidence_scope_preserved=True,elapsed_seconds=time.monotonic()-started,numerical=numerical)
    return best,report
