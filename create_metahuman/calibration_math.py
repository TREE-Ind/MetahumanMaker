"""Numerical helpers for native evaluated-pose corrective calibration."""
import numpy as np


def apply_rest_offsets(current, offsets, linear):
    """Predict an engine pose after rest edits using UE's row-vector Jacobian."""
    current=np.asarray(current,dtype=float);offsets=np.asarray(offsets,dtype=float);linear=np.asarray(linear,dtype=float)
    if current.ndim!=2 or current.shape[1]!=3 or offsets.shape!=current.shape or linear.shape!=(len(current),3,3) or not all(np.isfinite(x).all() for x in (current,offsets,linear)):
        raise ValueError('Rest-offset prediction needs finite corresponding engine poses and Jacobians')
    return current+np.einsum('ni,nij->nj',offsets,linear)


def rest_offsets(current,desired,linear,*,minimum_singular_value=1e-4):
    """Invert the evaluated engine's per-vertex row-vector skin Jacobian."""
    current=np.asarray(current,dtype=float);desired=np.asarray(desired,dtype=float);linear=np.asarray(linear,dtype=float)
    if current.ndim!=2 or current.shape[1]!=3 or desired.shape!=current.shape or linear.shape!=(len(current),3,3):
        raise ValueError('Corrective calibration needs corresponding Nx3 poses and Nx3x3 engine skin Jacobians')
    if not all(np.isfinite(x).all() for x in (current,desired,linear)):raise ValueError('Calibration samples must be finite')
    singular=np.linalg.svd(linear,compute_uv=False)
    if minimum_singular_value<=0 or np.any(singular[:,-1]<minimum_singular_value):
        raise ValueError('A skin Jacobian is singular; retain the current rig and revise the motion sample')
    delta=desired-current
    result=np.linalg.solve(np.swapaxes(linear,1,2),delta[:,:,None])[:,:,0]
    residual=apply_rest_offsets(current,result,linear)-desired
    return result,dict(method='INVERSE_EVALUATED_ENGINE_SKIN_JACOBIAN',vertices=len(current),
        minimum_singular_value=float(singular.min()),maximum_pose_delta=float(np.linalg.norm(delta,axis=1).max(initial=0)),
        maximum_rest_offset=float(np.linalg.norm(result,axis=1).max(initial=0)),
        maximum_linear_reconstruction_error=float(np.linalg.norm(residual,axis=1).max(initial=0)))


def canonical_offsets(offsets,mesh_indices,vertex_indices,mesh,vertex_count,*,duplicate_tolerance=5e-4):
    """Fold verified engine import/DNA identities without positional aliasing.

    A native canonical vertex may have several UV-split render vertices. They
    must request the same correction within the caller's world-unit tolerance.
    Unmapped anatomy is never filled by nearest-neighbor guesses.
    """
    offsets=np.asarray(offsets,dtype=float);meshes=np.asarray(mesh_indices);vertices=np.asarray(vertex_indices)
    if offsets.ndim!=2 or offsets.shape[1]!=3 or meshes.shape!=(len(offsets),) or vertices.shape!=meshes.shape or vertex_count<=0:
        raise ValueError('Invalid engine/DNA corrective correspondence')
    if not np.isfinite(offsets).all() or meshes.dtype.kind not in 'iu' or vertices.dtype.kind not in 'iu' or duplicate_tolerance<0:
        raise ValueError('Corrective identities must be integer arrays with finite offsets')
    selected=meshes==mesh;ids=vertices[selected];values=offsets[selected]
    if not len(ids) or ids.min()<0 or ids.max()>=vertex_count:raise ValueError('Native corrective correspondence leaves the expected DNA mesh')
    counts=np.bincount(ids,minlength=vertex_count)
    if np.any(counts==0):raise ValueError('Native DNA vertex correspondence is incomplete')
    result=np.column_stack([np.bincount(ids,weights=values[:,j],minlength=vertex_count)/counts for j in range(3)])
    disagreement=float(np.linalg.norm(values-result[ids],axis=1).max(initial=0))
    if disagreement>duplicate_tolerance:raise ValueError('Split render vertices disagree on a native corrective; retain current rig')
    return result,dict(method='ENGINE_IMPORT_DNA_IDENTITIES',render_vertices=len(ids),canonical_vertices=vertex_count,
        maximum_split_vertex_disagreement=disagreement,nearest_neighbor_aliases=False)
