"""Per-observation projection into the unchanged original-source allowance."""
import numpy as np
from .vertex_contract import closest_triangle_points
from .source_field_derivatives import barycentric_jacobian


def project(points,facets,radii):
    """Project to triangle offset tubes and differentiate each source witness.

    Facets must be actual nearest original-source witnesses. Inside a tube
    the map is the identity. Outside, its derivative retains surface motion
    without making the optimizer pay a large off-surface penalty for a curved
    tangent step. Artist bounds and all endpoint checks remain independent.
    """
    points=np.asarray(points,dtype=float);radii=np.broadcast_to(radii,(len(points),))
    closest,_,distance=closest_triangle_points(points,facets)
    output=points.copy();jacobian=np.broadcast_to(np.eye(3),(len(points),3,3)).copy()
    outside=distance>radii
    if np.any(outside):
        raw=points[outside];tri=facets[outside];q=closest[outside];d=distance[outside];r=radii[outside]
        _,db=barycentric_jacobian(raw,tri)
        tangent=np.einsum('nki,nkj->nij',tri,db)
        normal=(raw-q)/d[:,None];ratio=r/d
        output[outside]=q+(raw-q)*ratio[:,None]
        transverse=np.eye(3)[None]-normal[:,:,None]*normal[:,None,:]
        jacobian[outside]=tangent+ratio[:,None,None]*(transverse@(np.eye(3)[None]-tangent))
    return output,jacobian
