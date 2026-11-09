"""Differentiate the existing outward-corner source-support certificate."""
import numpy as np


def outward_corners(points,triangles,vertex_field,jacobians):
    """Minimum observed-corner agreement with the target face direction.

    The native predicate permits this certificate as an alternative to mean
    normal coherence. Its threshold remains 0.25; this function changes no
    scope or acceptance limit. The derivative includes the moving source
    direction as well as the target area direction.
    """
    p=points[triangles];C=np.cross(p[:,1]-p[:,0],p[:,2]-p[:,0])
    length=np.maximum(np.linalg.norm(C,axis=1),1e-30);unit=C/length[:,None]
    corners=vertex_field[triangles];cosine=np.einsum('fki,fi->fk',corners,unit)
    chosen=np.argmin(cosine,axis=1);rows=np.arange(len(p));value=cosine[rows,chosen]
    direction=(corners[rows,chosen]-value[:,None]*unit)/length[:,None]
    gradient=np.stack((np.cross(p[:,1]-p[:,2],direction),np.cross(p[:,2]-p[:,0],direction),np.cross(p[:,0]-p[:,1],direction)),axis=1)
    gradient[rows,chosen]+=np.einsum('fi,fij->fj',unit,jacobians[triangles[rows,chosen]])
    return value,gradient
