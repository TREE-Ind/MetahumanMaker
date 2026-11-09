"""Restore immutable angular limits after nonlinear source redistribution."""
import time
import numpy as np
from scipy import sparse
from .convex_surface_step import project_joint_conic
from .source_angle_contract import angular_values
from .surface_contacts import crossings


def polish(reference,points,tri,free,normals,minimum_area,following,projector,*,seconds=30.,statistics=None,progress=None,cancelled=None,acceptance_angles=None,target_points=None):
    """Restore a proposal, or advance a valid seed toward a separate target.

    Targeted advance accepts only feasible, contact-preserving steps that
    reduce squared target distance. The caller still checks actual source
    coverage, native storage and the complete reconstructed surface.
    """
    stats={} if statistics is None else statistics;history=[];began=time.monotonic();deadline=began+seconds
    ids=np.flatnonzero(free);lookup=np.full(len(points),-1,int);lookup[ids]=np.arange(len(ids));active=np.flatnonzero(np.any(free[tri],axis=1))
    target=points.copy() if target_points is None else np.asarray(target_points,dtype=float).copy()
    if target.shape!=points.shape or not np.isfinite(target).all():raise ValueError('Invalid surface refinement target')
    advance=target_points is not None
    points=projector.retract(points);span=float(np.linalg.norm(np.ptp(reference,axis=0)))
    contact_faces=np.any(free[tri],axis=1)
    stats['contact_scope']=dict(method='ALL_PAIRS_TOUCHING_MOVABLE_TRIANGLES',faces=int(contact_faces.sum()),fixed_coordinates_checked=True)
    if not np.array_equal(points[~free],reference[~free]):raise ValueError('Surface refinement changed protected coordinates')
    allowed=crossings(reference,tri,pairs=True,tolerance=span*1e-6,active_faces=contact_faces)
    if acceptance_angles is not None:
        if not np.array_equal(acceptance_angles['face_ids'],following.original['face_ids']) or not np.array_equal(acceptance_angles['area_scale'],following.original['area_scale']):raise ValueError('Native angular acceptance must retain the original protected faces and area metric')
        if np.any(acceptance_angles['minimum_cosine']>following.original['minimum_cosine']):raise ValueError('Worker precision reserve must lie inside native angular acceptance')
        stats['native_angular_acceptance']=True
    def gradient(p,n):return np.stack((np.cross(p[:,1]-p[:,2],n),np.cross(p[:,2]-p[:,0],n),np.cross(p[:,0]-p[:,1],n)),axis=1)
    p=reference[tri];C=np.cross(p[:,1]-p[:,0],p[:,2]-p[:,0]);signed=np.sum(C*normals,axis=1)
    floor=np.where(signed>0,np.minimum(minimum_area,signed*.5),minimum_area)
    scale=np.maximum(np.linalg.norm(gradient(p,normals),axis=(1,2)),1e-8)
    def flat_field(value,refresh=True):
        return following.flat_directions(value,refresh=refresh)
    flat_before,flat_supported,flat_vertices_before,_=following.flat_directions(reference,refresh=False,return_field=True)
    flat_signed=np.sum(C*flat_before,axis=1)
    flat_ids=np.flatnonzero(flat_supported&(flat_signed>=0)&np.any(free[tri],axis=1))
    flat_floor=np.minimum(flat_signed*.5,np.linalg.norm(C,axis=1)*.05)
    flat_scale=np.maximum(np.linalg.norm(gradient(p,flat_before),axis=(1,2)),1e-8)
    flat_cosine=flat_signed/np.maximum(np.linalg.norm(C,axis=1),1e-30)
    quality_ids=np.flatnonzero(flat_supported&(flat_cosine>=.25)&np.any(free[tri],axis=1))
    quality_floor=np.minimum(.25+1e-5,(flat_cosine[quality_ids]+.25)*.5)
    quality_scale=np.maximum(np.linalg.norm(C[quality_ids],axis=1),1e-30)
    initial_corner=np.einsum('fki,fi->fk',flat_vertices_before[tri[quality_ids]],C[quality_ids]/quality_scale[:,None]).min(1)
    quality_corner_floor=np.minimum(.25+1e-5,np.maximum(.25,(initial_corner+.25)*.5))
    stats['protected_flat_quality_faces']=len(quality_ids)
    def measure(candidate):
        contract,n,supported,support_margin=following.inspect(candidate);p=candidate[tri];C=np.cross(p[:,1]-p[:,0],p[:,2]-p[:,0]);actual_normals=normals.copy();actual_normals[following.scope]=n[following.scope]
        # Tightened worker cones reserve room for storage quantization; they
        # are proposal targets. Acceptance uses the unchanged original native
        # floors supplied by the operation snapshot, not the added reserve.
        acceptance=contract if acceptance_angles is None else dict(acceptance_angles,normals=contract['normals'])
        cosine,error=angular_values(candidate,tri,acceptance)
        orientation=np.maximum(floor[active]-np.sum(C[active]*actual_normals[active],axis=1),0)/scale[active]
        back=int(np.count_nonzero(np.sum(C[active]*actual_normals[active],axis=1)<=0))
        flat,flat_support=flat_field(candidate);flat_value=np.sum(C[flat_ids]*flat[flat_ids],axis=1)
        flat_error=float(np.maximum(flat_floor[flat_ids]-flat_value,0).max(initial=0))
        flat_new=int(np.count_nonzero((flat_value<0)|~flat_support[flat_ids]))
        support_error=float(np.maximum(-support_margin[following.scope],0).max(initial=0))
        support_bad=int(np.count_nonzero(following.scope&~supported))
        quality_cosine=np.sum(C[quality_ids]*flat[quality_ids],axis=1)/np.maximum(np.linalg.norm(C[quality_ids],axis=1),1e-30)
        quality_error=float((np.maximum(.25-quality_cosine,0)+(~flat_support[quality_ids])).max(initial=0))
        quality_bad=int(np.count_nonzero((quality_cosine<.25)|~flat_support[quality_ids]))
        return contract,actual_normals,float(error.max(initial=0)),float(np.maximum(acceptance['minimum_cosine']-cosine,0).max(initial=0)),float(orientation.max(initial=0)),back,flat_error,flat_new,support_error,support_bad,quality_error,quality_bad
    current=measure(points)
    def feasible(value):
        return value[2]<=1e-9 and value[3]<=1e-7 and value[4]<=1e-9 and value[5]==0 and value[6]<=1e-9 and value[7]==0 and value[9]==0 and value[11]==0
    if advance and not feasible(current):raise ValueError('Feasible surface advance requires a certified starting point')
    if advance and crossings(points,tri,pairs=True,tolerance=span*1e-6,active_faces=contact_faces)-allowed:raise ValueError('Feasible surface advance introduces contacts at its starting point')
    def merit(value):
        # Internal refinement may trade a small residual against a much
        # larger one. Requiring every residual to fall independently rejects
        # useful nonlinear corrections. Final tolerances remain unchanged.
        return float(np.linalg.norm(np.array([value[2]/1e-9,value[3]/1e-7,value[4]/1e-9,value[6]/1e-9,
            value[8]/1e-7,value[10]/1e-7,2*value[5],2*value[7],2*value[9],2*value[11]],dtype=float)))
    def finish(valid,reason):
        if valid:following(points)  # Certify the complete, unchanged native predicate.
        stats.update(valid=valid,reason=reason,history=history,seconds=time.monotonic()-began)
        if advance:
            stats['feasible_advance']=dict(target_error_before=float(np.sum((reference-target)**2)),
                target_error_after=float(np.sum((points-target)**2)),maximum_displacement_m=float(np.linalg.norm(points-reference,axis=1).max(initial=0)))
        return points.copy() if valid else reference.copy()
    for iteration in range(8):
        if cancelled and cancelled():raise RuntimeError('Nonlinear source constraint refinement cancelled')
        if progress:progress('Restoring original surface limits: pass '+str(iteration+1),(iteration+1)/8)
        contract,n,angle_error,cos_error,orient_error,back,flat_error,flat_new,support_error,support_bad,quality_error,quality_bad=current
        if not advance and feasible(current):
            if not (crossings(points,tri,pairs=True,tolerance=span*1e-6,active_faces=contact_faces)-allowed):return finish(True,'original_limits_restored')
        if time.monotonic()>=deadline:return finish(advance and feasible(current),'deadline')
        p=points[tri];C=np.cross(p[:,1]-p[:,0],p[:,2]-p[:,0]);G=gradient(p,n)
        _,field,D,vfield,J=following._derivatives(points,vertex_field=True)
        G[following.scope]+=np.einsum('fi,fkij->fkj',C,D)[following.scope]
        rows=[];columns=[];values=[];bounds=[]
        def append(faces,derivatives,rhs):
            start=len(bounds);bounds.extend(rhs.tolist())
            for corner in range(3):
                var=lookup[tri[faces,corner]];use=var>=0
                r=np.broadcast_to(np.arange(start,start+len(faces))[:,None],(len(faces),3));c=var[:,None]*3+np.arange(3)
                rows.extend(r[use].ravel());columns.extend(c[use].ravel());values.extend(derivatives[use,corner].ravel())
        append(active,G[active]/scale[active,None,None],(floor[active]-np.sum(C[active]*n[active],axis=1))/scale[active])
        flat,flat_support=flat_field(points)
        append(flat_ids,gradient(p[flat_ids],flat[flat_ids])/flat_scale[flat_ids,None,None],
            (flat_floor[flat_ids]-np.sum(C[flat_ids]*flat[flat_ids],axis=1))/flat_scale[flat_ids])
        coherent=np.flatnonzero(following.scope);size=np.linalg.norm(vfield[tri].sum(1),axis=1)
        dsize=np.einsum('fi,fkij->fkj',field,J[tri])
        ulp=np.abs(np.spacing(points[tri].astype(np.float32)).astype(float))
        coherence_reserve=8*np.sum(np.abs(dsize)*ulp,axis=(1,2))
        from .source_direction_support import outward_corners
        outward,outward_gradient=outward_corners(points,tri[coherent],vfield,J)
        outward_reserve=8*np.sum(np.abs(outward_gradient)*ulp[coherent],axis=(1,2))
        coherence_rhs=2.4+np.maximum(1e-6,coherence_reserve[coherent])-size[coherent]
        outward_rhs=.25+np.maximum(1e-6,outward_reserve)-outward
        use_outward=outward_rhs<coherence_rhs
        append(coherent,np.where(use_outward[:,None,None],outward_gradient,dsize[coherent]),np.minimum(coherence_rhs,outward_rhs))
        stats['support_predicate']='COHERENCE_OR_OUTWARD_OBSERVED_CORNERS_V1'
        stats['support_branches']=dict(coherent=int((~use_outward).sum()),outward=int(use_outward.sum()),native_thresholds_unchanged=True)
        matrix=sparse.coo_matrix((values,(rows,columns)),shape=(len(bounds),len(ids)*3)).tocsr()
        model=dict(hessian=sparse.eye(len(ids)*3,format='csc'),gradient=(points-target)[free].ravel(),soft_matrix=sparse.csr_matrix((0,len(ids)*3)),soft_bounds=np.empty(0),penalty=0.,
            strict_feasibility=True,angle_cones=following.linearize(points,free,contract))
        if len(quality_ids):
            from .surface_alignment import angular_model
            _,_,raw_vertices,_=following.flat_directions(points,return_field=True)
            raw=raw_vertices[tri[quality_ids]];coherent_raw=np.linalg.norm(raw.sum(1),axis=1)>=2.4
            quality=dict(face_ids=np.r_[quality_ids[coherent_raw],np.repeat(quality_ids[~coherent_raw],3)],
                normals=np.vstack((flat[quality_ids[coherent_raw]],raw[~coherent_raw].reshape(-1,3))),
                minimum_cosine=np.r_[quality_floor[coherent_raw],np.repeat(quality_corner_floor[~coherent_raw],3)],
                area_scale=np.r_[quality_scale[coherent_raw],np.repeat(quality_scale[~coherent_raw],3)])
            extra=angular_model(points,tri,free,quality);combined=model['angle_cones']
            combined['matrix']=sparse.vstack((combined['matrix'],extra['matrix']),format='csr')
            for key in ('offset','normals','minimum_cosine'):combined[key]=np.concatenate((combined[key],extra[key]),axis=0)
        projector.prepare(points)
        lower=(projector.reference-projector.limits-points)[free].ravel();upper=(projector.reference+projector.limits-points)[free].ravel()
        proposed,detail=project_joint_conic(points,points,free,matrix,np.asarray(bounds),lower,upper,projector,min(deadline,time.monotonic()+8),model=model)
        before_merit=merit(current)
        row=dict(iteration=iteration+1,conic=detail,before_angle_error=angle_error,before_cosine_error=cos_error,before_feasibility_merit=before_merit,trials=[]);history.append(row)
        if not detail['valid']:return finish(advance and feasible(current),'local_projection_failed')
        accepted=False
        for fraction in (1.,.5,.25,.125,.0625):
            if cancelled and cancelled():raise RuntimeError('Nonlinear source constraint refinement cancelled')
            try:
                candidate=projector.retract(points+(proposed-points)*fraction)
                measured=measure(candidate)
                after_merit=merit(measured)
                improving=(feasible(measured) and float(np.sum((candidate-target)**2))<float(np.sum((points-target)**2))-max(span*span*1e-20,1e-30)) if advance else after_merit<before_merit*(1-1e-4*fraction)
                # Fixed/fixed triangles cannot acquire an intersection. A
                # trial that already fails merit or feasibility cannot be
                # accepted, so do not rebuild a contact search for it.
                protected_exact=np.array_equal(candidate[~free],reference[~free])
                new_pairs=crossings(candidate,tri,pairs=True,tolerance=span*1e-6,active_faces=contact_faces)-allowed if improving and protected_exact else None
                row['trials'].append(dict(fraction=fraction,angle_error=measured[2],cosine_error=measured[3],orientation_error=measured[4],flat_error=measured[6],new_flat_backward_faces=measured[7],source_support_error=measured[8],unsupported_source_faces=measured[9],flat_quality_error=measured[10],new_severe_flat_faces=measured[11],feasibility_merit=after_merit,new_pairs=None if new_pairs is None else len(new_pairs),contact_checked=new_pairs is not None,protected_exact=protected_exact))
                if improving and protected_exact and new_pairs is not None and not new_pairs:
                    points=candidate;current=measured;accepted=True
                    if not advance and feasible(measured):return finish(True,'original_limits_restored')
                    break
            except ValueError as exc:row['trials'].append(dict(fraction=fraction,error=str(exc)))
        print(row,flush=True)
        if not accepted:return finish(advance and feasible(current),'no_exact_nonlinear_progress')
    return finish(advance and feasible(current),'pass_limit')
