"""Smooth, low-frequency cross-section initialization before local surface fitting."""
import numpy as np
from .geometry import points


def monotone(x,knots):
    knots=np.asarray(knots,dtype=float)
    if knots.ndim!=2 or knots.shape[1]!=2 or len(knots)<2 or not np.isfinite(knots).all() or not np.all(np.diff(knots,axis=0)>0):
        raise ValueError('Profile stations must increase in both reference and source coordinates')
    xp,yp=knots.T;h=np.diff(xp);d=np.diff(yp)/h;slope=np.empty(len(xp));slope[0]=d[0];slope[-1]=d[-1]
    for i in range(1,len(xp)-1):
        a,b=2*h[i]+h[i-1],h[i]+2*h[i-1];slope[i]=(a+b)/(a/d[i-1]+b/d[i])
    x=np.asarray(x,dtype=float);index=np.clip(np.searchsorted(xp,x)-1,0,len(xp)-2)
    t=(x-xp[index])/h[index];u=np.clip(t,0,1)
    y=(2*u**3-3*u**2+1)*yp[index]+(u**3-2*u**2+u)*h[index]*slope[index]+(-2*u**3+3*u**2)*yp[index+1]+(u**3-u**2)*h[index]*slope[index+1]
    y=np.where(x<xp[0],yp[0]+(x-xp[0])*slope[0],y);y=np.where(x>xp[-1],yp[-1]+(x-xp[-1])*slope[-1],y)
    return y


def section_profile(cloud,stations=61,triangles=None):
    cloud=points(cloud)
    triangles=None if triangles is None else np.asarray(triangles,dtype=int).reshape(-1,3)
    vertices=cloud if triangles is None else cloud[np.unique(triangles)]
    z=np.linspace(vertices[:,2].min(),vertices[:,2].max(),stations)
    faces=None if triangles is None else cloud[triangles]
    centers=[];radii=[];width=(z[-1]-z[0])/(stations-1)*1.3
    for station in z:
        samples=[]
        if faces is not None:
            crossing=faces[(faces[:,:,2].min(1)<=station)&(faces[:,:,2].max(1)>=station)]
            for i,j in [(0,1),(1,2),(2,0)]:
                a,b=crossing[:,i],crossing[:,j];dz=b[:,2]-a[:,2]
                valid=((a[:,2]<=station)&(b[:,2]>station)|(b[:,2]<=station)&(a[:,2]>station))&(abs(dz)>1e-12)
                samples.extend((a[valid]+(b[valid]-a[valid])*((station-a[valid,2])/dz[valid])[:,None])[:,:2])
        if len(samples)<3:
            ids=np.flatnonzero(abs(vertices[:,2]-station)<width)
            if len(ids)<12:ids=np.argsort(abs(vertices[:,2]-station))[:min(12,len(vertices))]
            samples=vertices[ids,:2]
        samples=np.asarray(samples)
        low,high=samples.min(0),samples.max(0);centers.append((low+high)*.5);radii.append((high-low)*.5)
    centers=np.array(centers);radii=np.array(radii)
    for _ in range(4):
        centers[1:-1]=.25*centers[:-2]+.5*centers[1:-1]+.25*centers[2:]
        radii[1:-1]=.25*radii[:-2]+.5*radii[1:-1]+.25*radii[2:]
    radii=np.maximum(radii,np.max(radii,axis=0)*.025)
    return z,centers,radii


def fit(query,reference,source,knots,*,reference_triangles=None,source_triangles=None):
    query=points(query);rz,rc,rr=section_profile(reference,triangles=reference_triangles);sz,sc,sr=section_profile(source,triangles=source_triangles)
    z=monotone(query[:,2],knots)
    def sample(grid,values,parameter):return np.column_stack([np.interp(parameter,grid,values[:,i]) for i in range(2)])
    normalized=(query[:,:2]-sample(rz,rc,query[:,2]))/sample(rz,rr,query[:,2])
    return np.column_stack((sample(sz,sc,z)+normalized*sample(sz,sr,z),z))


def surface_profile(cloud,triangles,*,stations=81,angles=128):
    """Sample an outer radial envelope from actual triangle/plane segments.

    Intended for an explicitly isolated, approximately star-shaped anatomical
    region. Outermost hits retain hidden inner surfaces' radial fraction during
    fitting. It cannot disambiguate adjacent anatomy or infer missing surfaces.
    """
    cloud=points(cloud);triangles=np.asarray(triangles,dtype=np.int32).reshape(-1,3)
    if not len(triangles) or triangles.min()<0 or triangles.max()>=len(cloud):
        raise ValueError('Surface profile requires valid nonempty triangles')
    if not 5<=stations<=256 or not 16<=angles<=512:raise ValueError('Invalid surface profile sampling resolution')
    z,center,box=section_profile(cloud,stations=stations,triangles=triangles)
    if z[-1]-z[0]<1e-9:raise ValueError('Surface profile has no axial extent')
    directions=np.column_stack((np.cos(np.arange(angles)*2*np.pi/angles),np.sin(np.arange(angles)*2*np.pi/angles)))
    faces=cloud[triangles];low=faces[:,:,2].min(1);high=faces[:,:,2].max(1)
    radius=[];supported=[]
    def cross(a,b):return a[...,0]*b[...,1]-a[...,1]*b[...,0]
    for index,station in enumerate(z):
        # Step just inside a cap, avoiding vertex-density fallback at an exact
        # extremum. Open/cropped sectors remain explicit unsupported samples.
        station=np.clip(station,z[0]+(z[-1]-z[0])*1e-7,z[-1]-(z[-1]-z[0])*1e-7)
        cut=faces[(low<=station)&(high>=station)];first=[];second=[]
        for face in cut:
            hits=[]
            for i,j in ((0,1),(1,2),(2,0)):
                a,b=face[i],face[j]
                if a[2]<=station<b[2] or b[2]<=station<a[2]:
                    hits.append((a+(b-a)*(station-a[2])/(b[2]-a[2]))[:2])
            if len(hits)==2:first.append(hits[0]);second.append(hits[1])
        fallback=1/np.sqrt(np.sum((directions/box[index])**2,axis=1))
        if not first:radius.append(fallback);supported.append(np.zeros(angles,bool));continue
        a=np.asarray(first)-center[index];edge=np.asarray(second)-np.asarray(first)
        denominator=cross(directions[:,None,:],edge[None,:,:])
        safe=np.where(abs(denominator)>1e-12,denominator,1.)
        distance=cross(a,edge)[None,:]/safe
        t=cross(a[None,:,:],directions[:,None,:])/safe
        valid=(abs(denominator)>1e-12)&(distance>0)&(t>=0)&(t<=1)
        measured=np.max(np.where(valid,distance,0),axis=1);support=np.any(valid,axis=1)
        radius.append(np.where(support,measured,fallback));supported.append(support)
    radius=np.asarray(radius)
    # Smooth axially, without averaging across the opposite side of a feature.
    for _ in range(3):radius[1:-1]=.25*radius[:-2]+.5*radius[1:-1]+.25*radius[2:]
    radius=np.maximum(radius,float(np.max(radius))*.005)
    return z,center,radius,np.asarray(supported)


def fit_surface(query,reference,source,knots,*,reference_triangles,source_triangles):
    """Match angular surface envelopes instead of elliptical bounding sections.

    Positive radial scale and a monotonic axial map retain the radial order of
    inner/outer skin. Callers must isolate anatomy, preserve physical seams and
    inspect topology/collisions before applying the returned coordinates.
    """
    query=points(query)
    rz,rc,rr,rs=surface_profile(reference,reference_triangles)
    sz,sc,sr,ss=surface_profile(source,source_triangles)
    z=monotone(query[:,2],knots)
    if np.any(query[:,2]<rz[0]-1e-6) or np.any(query[:,2]>rz[-1]+1e-6):
        raise ValueError('Query extends outside the sampled reference profile; expand its anatomical region')
    if np.any(z<sz[0]-1e-6) or np.any(z>sz[-1]+1e-6):
        raise ValueError('Mapped query extends outside the sampled source profile; expand its anatomical region')
    def center_at(grid,centers,at):return np.column_stack([np.interp(at,grid,centers[:,i]) for i in range(2)])
    offset=query[:,:2]-center_at(rz,rc,query[:,2]);length=np.linalg.norm(offset,axis=1)
    theta=np.mod(np.arctan2(offset[:,1],offset[:,0]),2*np.pi)
    def radius_at(grid,radius,at):
        v=np.clip((at-grid[0])/(grid[-1]-grid[0])*(len(grid)-1),0,len(grid)-1)
        i=np.minimum(v.astype(int),len(grid)-2);fz=v-i
        u=theta/(2*np.pi)*radius.shape[1];j=u.astype(int)%radius.shape[1];fu=u-np.floor(u);k=(j+1)%radius.shape[1]
        return ((1-fz)*((1-fu)*radius[i,j]+fu*radius[i,k])+fz*((1-fu)*radius[i+1,j]+fu*radius[i+1,k]))
    scale=radius_at(sz,sr,z)/radius_at(rz,rr,query[:,2])
    result=np.column_stack((center_at(sz,sc,z)+offset*scale[:,None],z))
    return result,{'solver':'POLAR_SURFACE_PROFILE_V1','reference_supported_fraction':float(rs.mean()),
        'source_supported_fraction':float(ss.mean()),'maximum_radial_scale':float(scale.max()),'review_required':True}
