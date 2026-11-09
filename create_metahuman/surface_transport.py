"""Intrinsic cross-section correspondence for labelled elongated mesh branches.

This is a pose/shape initializer for fingers, toes and other tube-like regions.
It does not infer anatomy or edit source geometry. Explicit proximal/distal and
transverse hints disambiguate PCA and prevent mirrored digit correspondence.
"""
import numpy as np
from .geometry import points


def frame(cloud, direction, transverse):
    cloud=points(cloud)
    axis=np.asarray(direction,dtype=float).copy()
    if axis.shape!=(3,) or not np.isfinite(axis).all() or np.linalg.norm(axis)<1e-12:
        raise ValueError('Branch longitudinal direction must be a finite nonzero vector')
    axis/=np.linalg.norm(axis)
    hint=np.asarray(transverse,dtype=float)
    if hint.shape!=(3,) or not np.isfinite(hint).all():
        raise ValueError('Branch transverse hint must be a finite three-dimensional vector')
    side=hint-axis*(hint@axis);length=np.linalg.norm(side)
    if length<1e-8:raise ValueError('Branch transverse hint must differ from its longitudinal axis')
    side/=length
    return np.column_stack((axis,side,np.cross(axis,side)))


def infer_frames(reference_cloud,source_cloud):
    """PCA defaults for an already labelled single elongated branch."""
    frames=[]
    for cloud in (reference_cloud,source_cloud):
        p=points(cloud);p=p-p.mean(0)
        _,values,vt=np.linalg.svd(p,full_matrices=False)
        if len(values)<3 or values[1]<1e-10 or values[0]<values[1]*1.15:
            raise ValueError('Branch axis is ambiguous; provide explicit transport frame hints')
        frames.append(vt.T)
    a,b=frames
    if a[:,0]@b[:,0]<0:b[:,0]*=-1
    if a[:,1]@b[:,1]<0:b[:,1]*=-1
    if abs(a[:,0]@b[:,0])<.1:
        raise ValueError('Branch directions are too different for automatic orientation; provide frame hints')
    return dict(reference_direction=a[:,0],source_direction=b[:,0],
                reference_transverse=a[:,1],source_transverse=b[:,1])


def profile(cloud,basis,*,stations=25,triangles=None):
    if not 5<=int(stations)<=256:raise ValueError('Use between 5 and 256 branch profile stations')
    cloud=points(cloud);local=cloud@basis
    if triangles is not None:
        from .profile_fit import section_profile
        triangles=np.asarray(triangles,dtype=np.int32).reshape(-1,3)
        if not len(triangles) or triangles.min()<0 or triangles.max()>=len(cloud):
            raise ValueError('Branch profile needs valid surface triangles')
        # section_profile measures horizontal triangle intersections. Rotate
        # the intrinsic axial coordinate into Z, rather than weighting the
        # silhouette by the density of imported mesh vertices.
        axial,centers,radii=section_profile(local[:,[1,2,0]],stations=int(stations),triangles=triangles)
        span=float(axial[-1]-axial[0])
        if span<1e-9:raise ValueError('Branch has no longitudinal extent')
        if np.any(np.max(radii,axis=0)<1e-10):raise ValueError('Branch surface has no usable cross-section extent')
        return {'basis':basis,'low':float(axial[0]),'span':span,
                'grid':(axial-axial[0])/span,'center':centers,'radius':radii}
    low,high=np.quantile(local[:,0],[.002,.998]);span=high-low
    if span<1e-9:raise ValueError('Branch has no longitudinal extent')
    t=(local[:,0]-low)/span
    grid=np.linspace(0,1,int(stations));centers=[];radii=[]
    minimum=max(12,min(40,len(cloud)//12))
    for station in grid:
        ids=np.flatnonzero(np.abs(t-station)<.055)
        if len(ids)<minimum:ids=np.argsort(np.abs(t-station))[:minimum]
        lower,upper=np.quantile(local[ids,1:],[.025,.975],axis=0)
        centers.append((lower+upper)*.5);radii.append((upper-lower)*.5)
    centers=np.asarray(centers);radii=np.asarray(radii)
    if np.any(np.max(radii,axis=0)<1e-10):raise ValueError('Branch surface has no usable cross-section extent')
    # Suppress sampling-density spikes along the centerline without smoothing
    # one digit into another. End stations retain their observed cap widths.
    for _ in range(2):
        centers[1:-1]=.25*centers[:-2]+.5*centers[1:-1]+.25*centers[2:]
        radii[1:-1]=.25*radii[:-2]+.5*radii[1:-1]+.25*radii[2:]
    radii=np.maximum(radii,np.max(radii,axis=0)*.06)
    return {'basis':basis,'low':float(low),'span':float(span),'grid':grid,'center':centers,'radius':radii}


def transport(query,reference_cloud,source_cloud,reference_direction,source_direction,
              reference_transverse,source_transverse,*,stations=25,axial_knots=None,
              reference_triangles=None,source_triangles=None):
    """Match longitudinal station and normalized cross-section coordinates.

    Correspondence is continuous through the branch and independent of the
    closest surface on adjacent digits. Callers blend the proximal boundary and
    run the mesh quality/collision guard before accepting a character change.
    """
    query=points(query)
    a=profile(reference_cloud,frame(reference_cloud,reference_direction,reference_transverse),stations=stations,triangles=reference_triangles)
    b=profile(source_cloud,frame(source_cloud,source_direction,source_transverse),stations=stations,triangles=source_triangles)
    local=query@a['basis'];t=(local[:,0]-a['low'])/a['span']
    source_t=t.copy()
    if axial_knots is not None:
        knots=np.asarray(axial_knots,dtype=float)
        if (knots.ndim!=2 or knots.shape[1]!=2 or len(knots)<2 or not np.isfinite(knots).all()
                or not np.all(np.diff(knots,axis=0)>0) or not np.allclose(knots[[0,-1]],[[0,0],[1,1]])):
            raise ValueError('Axial knots must increase in both columns from (0,0) to (1,1)')
        source_t=np.interp(t,knots[:,0],knots[:,1]);source_t[t<0]=t[t<0];source_t[t>1]=t[t>1]
    def sample(p,field,parameter):return np.column_stack([np.interp(parameter,p['grid'],p[field][:,i]) for i in range(2)])
    offset=(local[:,1:]-sample(a,'center',t))/sample(a,'radius',t)
    # Do not stretch unsupported root extrapolation by a distal cap ratio.
    mapped=np.column_stack((b['low']+source_t*b['span'],sample(b,'center',source_t)+offset*sample(b,'radius',source_t)))
    return mapped@b['basis'].T,{'reference_length':a['span'],'source_length':b['span'],
                                'stations':int(stations),'source_points':len(source_cloud),
                                'sampling':'TRIANGLE_SECTIONS' if reference_triangles is not None and source_triangles is not None else 'VERTEX_PROFILE'}
