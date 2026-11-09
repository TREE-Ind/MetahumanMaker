"""Fixed-area normal agreement with derivatives of the moving source field."""
import numpy as np


def evaluate(points,triangles,normals,field_derivatives,weights):
    """Return sum(w * (1 - dot(unit area, source normal))) and its gradient.

    Weights belong to the immutable canonical reference, never the current
    triangle area. Thus a collapsing triangle cannot hide its own error.
    Source-field derivatives are with respect to each triangle corner.
    This is a proposal objective; independent endpoint checks remain required.
    """
    p=points[triangles]
    cross=np.cross(p[:,1]-p[:,0],p[:,2]-p[:,0])
    size=np.maximum(np.linalg.norm(cross,axis=1),1e-30)
    unit=cross/size[:,None]
    cosine=np.sum(unit*normals,axis=1)
    direction=(normals-cosine[:,None]*unit)/size[:,None]
    derivative=np.stack((np.cross(p[:,1]-p[:,2],direction),
        np.cross(p[:,2]-p[:,0],direction),np.cross(p[:,0]-p[:,1],direction)),axis=1)
    derivative+=np.einsum('fi,fkij->fkj',unit,field_derivatives)
    gradient=np.zeros_like(points)
    for corner in range(3):
        np.add.at(gradient,triangles[:,corner],-weights[:,None]*derivative[:,corner])
    return float(weights@(1-cosine)),gradient
