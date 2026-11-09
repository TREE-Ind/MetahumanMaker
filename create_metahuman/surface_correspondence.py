"""Confidence-weighted conflict removal in a projected observation map."""
import numpy as np


def consistent_observations(targets,triangles,confidence,source,source_triangles,source_faces,parameter_reference):
    """Release point correspondences that prescribe severely misaligned faces.

    This does not remove source-distance evidence. The caller must retain the
    original geometric observation scope and source-offset constraints, allowing
    released vertices to redistribute on that surface instead of forcing a
    locally incompatible point map into the mesh.
    """
    targets=np.asarray(targets,dtype=float);tri=np.asarray(triangles,dtype=np.int64);confidence=np.asarray(confidence,dtype=float)
    source=np.asarray(source,dtype=float);st=np.asarray(source_triangles,dtype=np.int64);faces=np.asarray(source_faces,dtype=np.int64);metric=np.asarray(parameter_reference,dtype=float)
    n=len(targets);observed=confidence>0;keep=observed.copy()
    if targets.shape!=(n,3) or metric.shape!=targets.shape or confidence.shape!=(n,) or faces.shape!=(n,) or not np.isfinite(targets).all() or not np.isfinite(metric).all() or not np.isfinite(confidence).all() or np.any((confidence<0)|(confidence>1)):raise ValueError('Invalid observation map')
    if tri.ndim!=2 or tri.shape[1]!=3 or not len(tri) or tri.min()<0 or tri.max()>=n or source.ndim!=2 or source.shape[1]!=3 or not np.isfinite(source).all() or st.ndim!=2 or st.shape[1]!=3 or not len(st) or st.min()<0 or st.max()>=len(source) or np.any(faces[observed]<0) or np.any(faces[observed]>=len(st)):raise ValueError('Invalid observation map facets')
    field=np.zeros_like(targets);p=source[st[faces[observed]]];normal=np.cross(p[:,1]-p[:,0],p[:,2]-p[:,0]);length=np.linalg.norm(normal,axis=1)
    if np.any(length<=0):raise ValueError('Observed source normals need nondegenerate facets')
    field[observed]=normal/length[:,None];directions=field[tri].sum(1);size=np.linalg.norm(directions,axis=1)
    complete=np.all(observed[tri],axis=1)&(size>=2.4)
    q=targets[tri];cross=np.cross(q[:,1]-q[:,0],q[:,2]-q[:,0]);cosine=np.einsum('ij,ij->i',cross,directions)/np.maximum(np.linalg.norm(cross,axis=1)*size,1e-30)
    conflicts=complete&(cosine<.25);original=conflicts.copy()
    p=metric[tri];area=np.linalg.norm(np.cross(p[:,1]-p[:,0],p[:,2]-p[:,0]),axis=1)*.5
    support=np.zeros(n)
    for corner in range(3):np.add.at(support,tri[complete,corner],area[complete]/3)
    released=[]
    while conflicts.any():
        mass=np.zeros(n)
        for corner in range(3):np.add.at(mass,tri[conflicts,corner],area[conflicts]/3)
        candidates=np.unique(tri[conflicts]);cost=np.maximum(confidence[candidates]*support[candidates],1e-30)
        vertex=int(candidates[np.argmax(mass[candidates]/cost)])
        keep[vertex]=False;released.append(vertex);conflicts&=~np.any(tri==vertex,axis=1)
    if np.any(original&np.all(keep[tri],axis=1)):raise AssertionError('Conflicting complete observation triangles remain')
    return keep,dict(method='GREEDY_CONFIDENCE_WEIGHTED_OBSERVATION_MAP_CUT_V1',observed_points=int(observed.sum()),
        coherent_projected_faces=int(complete.sum()),conflicting_projected_faces=int(original.sum()),released_point_correspondences=len(released),
        remaining_complete_conflicts=0,geometric_scope_must_be_retained=True)
