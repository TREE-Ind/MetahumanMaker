"""Bounded source-surface resampling before local convex refinement."""
import time
import numpy as np
from scipy import sparse


def propose(points,triangles,metric,normals,observed,pinned,projector,assess,*,prepare_endpoint=None,time_limit=12.,cancelled=None):
    """Cross original source facets without increasing distance allowances.

    A canonical positive stencil proposes tangential centroid relaxation only
    on complete, coherent measured one-rings. Fresh source queries permit facet
    changes; independent fixed-scope and new-crease checks choose an endpoint.
    No accepted result bypasses the parent's native contact/geometry transaction.
    """
    started=time.monotonic();points=np.asarray(points,dtype=float);tri=np.asarray(triangles,dtype=np.int64);metric=np.asarray(metric,dtype=float);normals=np.asarray(normals,dtype=float);observed=np.asarray(observed,dtype=bool);pinned=np.asarray(pinned,dtype=bool);n=len(points)
    if points.shape!=(n,3) or metric.shape!=points.shape or normals.shape!=points.shape or observed.shape!=(n,) or pinned.shape!=(n,) or not all(np.isfinite(a).all() for a in (points,metric,normals)) or not np.isfinite(time_limit) or time_limit<=0:raise ValueError('Invalid source sample transport')
    if tri.ndim!=2 or tri.shape[1]!=3 or not len(tri) or tri.min()<0 or tri.max()>=n:raise ValueError('Invalid transport topology')
    edges=np.unique(np.sort(np.vstack((tri[:,[0,1]],tri[:,[1,2]],tri[:,[2,0]])),axis=1),axis=0);length=np.linalg.norm(metric[edges[:,0]]-metric[edges[:,1]],axis=1)
    if np.any(length<=0):raise ValueError('Transport requires nonzero canonical edges')
    adjacency=sparse.coo_matrix((np.r_[1/length,1/length],(np.r_[edges[:,0],edges[:,1]],np.r_[edges[:,1],edges[:,0]])),shape=(n,n)).tocsr();degree=np.asarray(adjacency.sum(1)).ravel();stencil=sparse.diags(1/np.maximum(degree,1e-30))@adjacency
    norm=np.linalg.norm(normals,axis=1)
    if np.any(norm[observed]<=1e-12):raise ValueError('Measured transport normals must be nonzero')
    unit=normals/np.maximum(norm[:,None],1e-30);coherence=np.einsum('ij,ij->i',unit,stencil@unit);complete=np.asarray(stencil@observed.astype(float)).ravel()>=1-1e-12
    movable=observed&~pinned&complete&(coherence>=.8)&np.isin(np.arange(n),projector.ids)
    direction=stencil@points-points;direction-=unit*np.einsum('ij,ij->i',direction,unit)[:,None];direction[~movable]=0.
    raw_edges=np.sort(np.vstack((tri[:,[0,1]],tri[:,[1,2]],tri[:,[2,0]])),axis=1);owners=np.tile(np.arange(len(tri)),3);order=np.lexsort((raw_edges[:,1],raw_edges[:,0]));raw_edges=raw_edges[order];owners=owners[order];starts=np.r_[0,np.flatnonzero(np.any(raw_edges[1:]!=raw_edges[:-1],axis=1))+1];ends=np.r_[starts[1:],len(raw_edges)];paired=starts[ends-starts==2];adjacent=np.column_stack((owners[paired],owners[paired+1]))
    def hinges(value):
        p=value[tri];normal=np.cross(p[:,1]-p[:,0],p[:,2]-p[:,0]);normal/=np.maximum(np.linalg.norm(normal,axis=1)[:,None],1e-30)
        return np.rad2deg(np.arccos(np.clip(np.sum(normal[adjacent[:,0]]*normal[adjacent[:,1]],axis=1),-1,1)))
    initial_hinges=hinges(points);best=points.copy();best_quality=None;trials=[];selected_fraction=0.;blocked=np.zeros(n,bool)
    for fraction in (.5,.25,.125,.0625,.03125):
        if time.monotonic()-started>=time_limit or not movable.any():break
        if cancelled and cancelled():raise RuntimeError('Source sample transport cancelled')
        for localization in range(3):
            candidate=points+fraction*direction;candidate[blocked]=points[blocked];ids=projector.ids
            candidate[ids]=np.clip(candidate[ids],projector.reference[ids]-projector.limits[ids],projector.reference[ids]+projector.limits[ids])
            record=projector.query(candidate[ids]);centers=np.asarray(record['surface_points'])
            # Unreachable facets and newly creased neighborhoods retain their
            # input sampling. Neither artist boxes nor quality bounds widen.
            reachable=np.all(abs(centers-projector.reference[ids])<=projector.limits[ids],axis=1);choose=movable[ids]&~blocked[ids]&reachable
            if not np.asarray(record['observed']).all():raise ValueError('Transport needs complete original-source witnesses')
            candidate=points.copy();candidate[ids[choose]]=centers[choose]
            candidate=projector.retract(candidate);candidate[~movable|blocked]=points[~movable|blocked]
            prepared={}
            if prepare_endpoint is not None:candidate,prepared=prepare_endpoint(candidate)
            np.testing.assert_array_equal(candidate[pinned],points[pinned])
            new_hinges=hinges(candidate);bad=(initial_hinges<30.)&(new_hinges>=30.);new_sharp=int(np.count_nonzero(bad))
            if not new_sharp or localization==2 or time.monotonic()-started>=time_limit:break
            offending=np.unique(tri[adjacent[bad]]);collar=np.any(np.isin(tri,offending),axis=1);freeze=np.unique(tri[collar]);added=np.count_nonzero(~blocked[freeze]);blocked[freeze]=True
            if not added:break
        quality=assess(candidate);valid=bool(quality['valid']) and new_sharp==0
        trials.append(dict(fraction=fraction,valid=valid,quality=quality,new_sharp_hinges=new_sharp,localization_passes=localization,frozen_crease_vertices=int(np.count_nonzero(blocked&movable)),unreachable_vertices_retained=int(np.count_nonzero(movable[ids]&~reachable)),maximum_hinge_increase_degrees=float((new_hinges-initial_hinges).max(initial=0)),maximum_movement_m=float(np.linalg.norm(candidate-points,axis=1).max(initial=0)),endpoint_preparation=prepared))
        if valid and (best_quality is None or quality['score']<best_quality['score']):best=candidate.copy();best_quality=quality;selected_fraction=fraction
    projector.prepare(best)
    return best,dict(method='MEASURED_SOURCE_TANGENT_ENDPOINT_TRANSPORT_V1',accepted=best_quality is not None,selected_fraction=selected_fraction,quality=best_quality,measured_stencils=int(movable.sum()),trials=trials,source_distance_limits_unchanged=True,original_source_queries=True,requires_native_contact_validation=True,elapsed_seconds=time.monotonic()-started)
