"""Geometry-bound fitted eye frames, independent of Blender and materials."""
import hashlib,json
import numpy as np

PROPERTY='cm_ocular_volume'


def signature(local_points,triangles):
    points=np.asarray(local_points,dtype='<f4');faces=np.asarray(triangles,dtype='<i4')
    if points.ndim!=2 or points.shape[1]!=3 or faces.ndim!=2 or faces.shape[1]!=3 or not np.isfinite(points).all():raise ValueError('Eye profile needs finite corresponding mesh geometry')
    return hashlib.sha256(points.tobytes()+faces.tobytes()).hexdigest()


def make(world_points,triangles,matrix,basis,volume):
    matrix=np.asarray(matrix,dtype=float);basis=np.asarray(basis,dtype=float);points=np.asarray(world_points,dtype=float)
    inverse=np.linalg.inv(matrix);local=points@inverse[:3,:3].T+inverse[:3,3]
    return json.dumps(dict(schema=1,geometry=signature(local,triangles),matrix=matrix.tolist(),basis=basis.tolist(),volume=volume))


def frame(raw,world_points,triangles,matrix,basis):
    """Retain the fitted frame across aligned positive object-scale/pose edits.

    Material changes never invalidate geometry. Local sculpt/topology changes
    do; rotation is supported when the caller's current source frame follows it.
    """
    if not raw:return None
    record=json.loads(raw);matrix=np.asarray(matrix,dtype=float);basis=np.asarray(basis,dtype=float);points=np.asarray(world_points,dtype=float)
    if record.get('schema')!=1:raise ValueError('Refit this eye volume to update its saved frame format')
    inverse=np.linalg.inv(matrix);local=points@inverse[:3,:3].T+inverse[:3,3]
    if signature(local,triangles)!=record['geometry']:raise ValueError('This eye was sculpted after its volume fit; refit the eye volume before projecting color')
    old_matrix=np.asarray(record['matrix']);old_basis=np.asarray(record['basis']);linear=matrix[:3,:3]@np.linalg.inv(old_matrix[:3,:3]);offset=matrix[:3,3]-linear@old_matrix[:3,3];axes=basis.T@linear@old_basis;scale=np.diag(axes)
    if np.any(scale<=0) or not np.allclose(axes,np.diag(scale),atol=max(float(np.abs(axes).max()),1)*1e-6,rtol=0):raise ValueError('The fitted eye frame no longer follows the source axes; refit its volume after changing alignment')
    volume=record['volume'];center=(linear@(old_basis@np.asarray(volume['center']))+offset)@basis
    radius=float(volume['radii'][1])*float(scale[1])
    if not np.isfinite(center).all() or not np.isfinite(radius) or radius<=0:raise ValueError('The saved eye volume frame is invalid')
    transformed=dict(center=center.tolist(),radii=(np.asarray(volume['radii'])*scale).tolist(),powers=list(volume['powers']),
                     depth_shear=[float(volume['depth_shear'][0])*float(scale[1]/scale[0]),float(volume['depth_shear'][1])*float(scale[1]/scale[2])])
    return dict(center=center,depth_radius=radius,parameters=transformed)
