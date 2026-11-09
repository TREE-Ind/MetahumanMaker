"""Carry an ocular surface correction into its attached shell and lid meshes.

Bindings use exact surface witnesses. Transfer changes only the difference
between two corresponding poses, retaining the accessory's authored standoff
and native deformation rather than replacing it with an archetype accessory.
"""
import numpy as np
from .surface_oracle import SurfaceOracle


def transport_eye_displacements(source_linear,target_linear,displacements,*,coherence_tolerance=1e-4):
    """Rotate a material displacement using evaluated, coherent eye skin frames.

    Skin matrices use the native export's row-vector convention. The polar
    factor carries orientation without scaling the sculpted displacement.
    The actual deformed globe is retained; this does not assume it is rigid.
    """
    source_linear=np.asarray(source_linear,dtype=float);target_linear=np.asarray(target_linear,dtype=float);displacements=np.asarray(displacements,dtype=float)
    if source_linear.ndim!=3 or target_linear.ndim!=3 or source_linear.shape[1:]!=(3,3) or target_linear.shape[1:]!=(3,3) or not len(source_linear) or not len(target_linear) or displacements.ndim!=2 or displacements.shape[1]!=3 or not all(np.isfinite(a).all() for a in [source_linear,target_linear,displacements]):raise ValueError('Supply finite evaluated eye skin matrices and displacement vectors')
    if not np.isfinite(coherence_tolerance) or not 0<coherence_tolerance<=.01:raise ValueError('Choose a small positive eye-frame coherence tolerance')
    a=source_linear.mean(0);b=target_linear.mean(0);spread=max(float(abs(source_linear-a).max()),float(abs(target_linear-b).max()))
    if spread>coherence_tolerance:raise ValueError('The captured eye has no coherent evaluated skin frame')
    if min(np.linalg.svd(a,compute_uv=False).min(),np.linalg.svd(b,compute_uv=False).min())<1e-8 or np.linalg.det(a)<=0 or np.linalg.det(b)<=0:raise ValueError('The evaluated eye frame is singular or reflected')
    relative=np.linalg.solve(a,b);u,sigma,vt=np.linalg.svd(relative);rotation=u@vt
    if np.linalg.det(rotation)<=0:raise ValueError('The evaluated eye motion contains a reflection')
    moved=displacements@rotation
    return moved,dict(method='EVALUATED_EYE_SKIN_FRAME_POLAR_TRANSPORT',native_frame_spread=spread,relative_singular_values=sigma.tolist(),rotation_matrix=rotation.tolist(),maximum_vector_length_error=float(abs(np.linalg.norm(moved,axis=1)-np.linalg.norm(displacements,axis=1)).max(initial=0)),globe_geometry_changed=False)


def bind(surface, triangles, points, *, maximum_distance):
    surface=np.asarray(surface,dtype=float);triangles=np.asarray(triangles,dtype=np.int64);points=np.asarray(points,dtype=float)
    oracle=SurfaceOracle(surface,triangles);record=oracle.query(points,max_distance=maximum_distance)
    if not record['observed'].all():raise ValueError('An ocular accessory is outside its bounded attachment surface')
    return dict(triangles=triangles[record['source_triangle_ids']].copy(),barycentric=record['barycentric'].copy(),
                maximum_binding_distance=float(record['residual'].max(initial=0)),vertices=len(surface))


def _frames(triangles):
    edge=triangles[:,1]-triangles[:,0];normal=np.cross(edge,triangles[:,2]-triangles[:,0]);length=np.linalg.norm(edge,axis=1);area=np.linalg.norm(normal,axis=1)
    if np.any(length<=1e-12) or np.any(area<=1e-14):raise ValueError('An ocular attachment triangle collapsed in this pose')
    tangent=edge/length[:,None];normal/=area[:,None]
    return np.stack((tangent,np.cross(normal,tangent),normal),axis=-1)


def transfer(source, target, points, binding, *, maximum_offset):
    source=np.asarray(source,dtype=float);target=np.asarray(target,dtype=float);points=np.asarray(points,dtype=float);ids=np.asarray(binding['triangles']);bary=np.asarray(binding['barycentric'])
    if source.shape!=target.shape or source.shape!=(binding['vertices'],3) or points.shape!=(len(ids),3) or not all(np.isfinite(a).all() for a in (source,target,points)):
        raise ValueError('Ocular attachment needs corresponding finite poses')
    old,new=source[ids],target[ids];old_anchor=np.einsum('ni,nij->nj',bary,old);new_anchor=np.einsum('ni,nij->nj',bary,new)
    # Orthonormal transport preserves the authored thickness/offset magnitude;
    # it cannot magnify a small standoff when a lid triangle compresses.
    local=np.einsum('nji,nj->ni',_frames(old),points-old_anchor)
    result=new_anchor+np.einsum('nij,nj->ni',_frames(new),local)
    change=np.linalg.norm(result-points,axis=1)
    if not np.isfinite(maximum_offset) or maximum_offset<=0 or change.max(initial=0)>maximum_offset:
        raise ValueError('Ocular accessory correction exceeds its bounded displacement')
    return result,dict(method='POSE_DIFFERENCE_SURFACE_ATTACHMENT',maximum_correction=float(change.max(initial=0)),
                       maximum_binding_distance=binding['maximum_binding_distance'],standoff_length_preserved=True)
