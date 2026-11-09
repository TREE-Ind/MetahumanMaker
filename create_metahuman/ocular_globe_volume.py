"""Source-conditioned convex globe continuation without skin-depth imprints.

An axis-aligned mixed-power ellipsoid supplies the unobserved volume. Its
exponents are at least one, so the implicit solid is convex. Observed sclera
depth and sampled socket clearance jointly fit its dimensions. The returned
mesh retains the reference topology; final native motion still needs review.
"""
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial import ConvexHull
from .ocular_surface import FrontSurface,_sphere


def transport_convex(neutral,rotation):
    """Reparameterize gaze inside the exact neutral convex envelope.

    Retain each vertex's fractional radial depth, including shallow authored
    recesses. Every returned vertex (and therefore each triangle) remains in
    the neutral convex solid. This is a volume bound, not lid contact proof.
    """
    neutral=np.asarray(neutral,dtype=float);rotation=np.asarray(rotation,dtype=float)
    if neutral.ndim!=2 or neutral.shape[1]!=3 or len(neutral)<4 or rotation.shape!=(3,3) or not np.isfinite(neutral).all() or not np.isfinite(rotation).all() or np.max(abs(rotation.T@rotation-np.eye(3)))>1e-5 or abs(np.linalg.det(rotation)-1)>1e-5:raise ValueError('Convex gaze transport needs finite neutral geometry and a rigid rotation')
    hull=ConvexHull(neutral);center=neutral[hull.vertices].mean(0);direction=neutral-center;length=np.linalg.norm(direction,axis=1)
    if np.any(length<=np.ptp(neutral,axis=0).max()*1e-10):raise ValueError('A globe vertex lies at its interior transport center')
    direction/=length[:,None];planes=hull.equations;numerator=-(center@planes[:,:3].T+planes[:,3])
    if np.any(numerator<=0):raise ValueError('The globe transport origin must be strictly inside its envelope')
    def radius(d):
        denominator=d@planes[:,:3].T
        return np.divide(numerator[None,:],denominator,out=np.full_like(denominator,np.inf),where=denominator>1e-12).min(1)
    fraction=np.minimum(length/radius(direction),1.);turned=direction@rotation;result=center+turned*(radius(turned)*fraction)[:,None];outside=float((result@planes[:,:3].T+planes[:,3]).max());scale=float(np.ptp(neutral,axis=0).max())
    if not np.isfinite(result).all() or outside>scale*1e-8:raise ValueError('Gaze transport left the neutral globe envelope')
    return result,dict(method='NEUTRAL_CONVEX_ENVELOPE_SURFACE_TRANSPORT',maximum_outside_distance=outside,neutral_volume_fixed=True,scope='Canonical sampled pose only; compiled interpolation and lid motion require native review')


def sample_directions(direction,parameters):
    """Map spherical surface coordinates onto the fitted convex volume."""
    direction=np.array(direction,dtype=float,copy=True);center=np.asarray(parameters['center'],dtype=float);axes=np.asarray(parameters['radii'],dtype=float);powers=np.asarray(parameters['powers'],dtype=float);shear=np.asarray(parameters['depth_shear'],dtype=float)
    if direction.ndim!=2 or direction.shape[1]!=3 or center.shape!=(3,) or axes.shape!=(3,) or powers.shape!=(3,) or shear.shape!=(2,) or not all(np.isfinite(v).all() for v in (direction,center,axes,powers,shear)) or np.any(axes<=0) or np.any(powers<=1):raise ValueError('Invalid convex globe volume parameters')
    length=np.linalg.norm(direction,axis=1)
    if np.any(length<=float(axes.max())*1e-10):raise ValueError('Reference globe has a vertex at its center')
    direction/=length[:,None];lo=np.zeros(len(direction));hi=np.full(len(direction),np.linalg.norm(axes)*2)
    for _ in range(52):
        middle=(lo+hi)*.5;outside=np.sum(np.abs(direction*middle[:,None]/axes)**powers,axis=1)>1;hi=np.where(outside,middle,hi);lo=np.where(outside,lo,middle)
    candidate=center+direction*((lo+hi)*.5)[:,None]
    candidate[:,1]+=shear[0]*(candidate[:,0]-center[0])+shear[1]*(candidate[:,2]-center[2])
    return candidate


def evaluate(reference,parameters):
    """Evaluate the fitted volume on a native LOD's retained globe topology."""
    reference=np.asarray(reference,dtype=float);ref_center,_=_sphere(reference,posterior=True)
    return sample_directions(reference-ref_center,parameters)


def transport(neutral,posed,parameters):
    """Move surface coordinates with gaze while retaining the fitted volume.

    This is an inferred deformation for non-spherical stylized eyes. It keeps
    the native UVs/topology and can be compiled into native pose correctives.
    """
    neutral=np.asarray(neutral,dtype=float);posed=np.asarray(posed,dtype=float)
    if neutral.shape!=posed.shape or neutral.ndim!=2 or neutral.shape[1]!=3 or not np.isfinite(neutral).all() or not np.isfinite(posed).all():raise ValueError('Globe transport needs corresponding finite native poses')
    a=neutral-neutral.mean(0);b=posed-posed.mean(0);u,_,vh=np.linalg.svd(a.T@b);fix=np.ones(3);fix[-1]=np.linalg.det(u@vh);rotation=(u*fix)@vh
    error=float(np.linalg.norm(a@rotation-b,axis=1).max());scale=float(np.ptp(neutral,axis=0).max())
    if error>scale*1e-4:raise ValueError('Native globe sample is not rigid; separate existing eye deformations before transport')
    direction=neutral-np.asarray(parameters['center']);shear=parameters['depth_shear'];direction[:,1]-=shear[0]*direction[:,0]+shear[1]*direction[:,2]
    direction/=np.linalg.norm(direction,axis=1)[:,None]
    candidate=sample_directions(direction@rotation,parameters)
    return candidate,dict(method='NATIVE_GAZE_SURFACE_COORDINATE_TRANSPORT',native_rigid_error=error,maximum_correction_world=float(np.linalg.norm(candidate-posed,axis=1).max()),topology_preserved=True,volume_frame_preserved=True)


def front_depth(query,parameters):
    """Analytic anterior depth and projected-domain mask in the fit frame."""
    query=np.asarray(query,dtype=float);center=np.asarray(parameters['center']);r=np.asarray(parameters['radii']);powers=parameters['powers'];shear=parameters['depth_shear'];xz=query-center[[0,2]]
    domain=1-(np.abs(xz[:,0])/r[0])**powers[0]-(np.abs(xz[:,1])/r[2])**powers[2]
    depth=center[1]-r[1]*np.maximum(domain,0)**(1/powers[1])+shear[0]*xz[:,0]+shear[1]*xz[:,1]
    return depth,domain>=0


def fit(reference,observed,head,head_triangles,*,cancelled=None):
    reference=np.asarray(reference,dtype=float);observed=np.asarray(observed,dtype=float)
    if reference.ndim!=2 or reference.shape[1]!=3 or observed.ndim!=2 or observed.shape[1]!=3 or len(observed)<12 or not np.isfinite(reference).all() or not np.isfinite(observed).all():raise ValueError('Globe volume needs finite reference and observed eye surfaces')
    low=observed[:,[0,2]].min(0);high=observed[:,[0,2]].max(0);half=(high-low)*.5;origin=observed.mean(0);origin[[0,2]]=(low+high)*.5;width=float(high[0]-low[0])
    if width<=0 or half[1]<=0:raise ValueError('Observed eye aperture has no usable extent')
    pts=(observed-origin)/width;half/=width
    hull=pts[ConvexHull(pts[:,[0,2]]).vertices][:,[0,2]];edges=np.roll(hull,-1,axis=0)-hull
    xx,zz=np.meshgrid(np.linspace(-half[0]*1.65,half[0]*1.65,49),np.linspace(-half[1]*2.1,half[1]*2.1,41));q=np.c_[xx.ravel(),zz.ravel()]
    delta=q[:,None,:]-hull[None,:,:];distance=np.min((edges[None,:,0]*delta[:,:,1]-edges[None,:,1]*delta[:,:,0])/np.linalg.norm(edges,axis=1)[None,:],axis=1)
    q=q[distance<-.002];world=q*width+origin[[0,2]]
    skin=(FrontSurface(head,head_triangles)(world)-origin[1])/width
    # Variables: center-front depth, lateral radius, vertical radius, depth
    # radius, and three convex exponents (X,Z,Y). Keep the source aperture's
    # center fixed; these are hidden-volume choices, not character placement.
    start=np.array([float(np.median(pts[:,1]))-.005,half[0]*1.12,half[1]*1.35,.18,2.,2.,2.,0.,0.])
    lower=np.array([pts[:,1].min()-.08,half[0]*1.002,half[1]*1.002,.025,1.1,1.1,1.1,-1.,-1.])
    upper=np.array([pts[:,1].max()+.08,half[0]*1.6,half[1]*2.,.65,8.,8.,6.,1.,1.])
    def surface(xz,v):
        front,rx,rz,ry,px,pz,py,tx,tz=v;t=1-(np.abs(xz[:,0])/rx)**px-(np.abs(xz[:,1])/rz)**pz
        y=front+ry*(1-np.maximum(t,0)**(1/py))+tx*xz[:,0]+tz*xz[:,1]
        return y,t
    def residual(v):
        if cancelled and cancelled():raise RuntimeError('Globe volume fit cancelled')
        measured,t=surface(pts[:,[0,2]],v);hidden,domain=surface(q,v)
        missing=np.maximum(-t,0)*2
        obstruction=np.where(domain>0,np.maximum(skin+.0005-hidden,0),0)
        # Source depth dominates. Regularity selects a compact hidden volume
        # among equally faithful fits rather than an unconstrained giant sphere.
        regularity=np.r_[(v[1:3]/half-1.15)*.015,(v[3]-.18)*.025,(v[4:7]-2)*.001,v[7:]*.001]
        return np.r_[(measured-pts[:,1])*4,missing,obstruction*3,regularity]
    solved=least_squares(residual,np.clip(start,lower+1e-8,upper-1e-8),bounds=(lower,upper),loss='soft_l1',f_scale=.005,max_nfev=350,ftol=1e-10,xtol=1e-10,gtol=1e-10)
    if not solved.success:raise ValueError('Source-conditioned globe volume did not converge')
    v=solved.x;measured,domain=surface(pts[:,[0,2]],v)
    if np.any(domain<=0):raise ValueError('Convex globe volume does not cover the observed source patch')
    front,rx,rz,ry,px,pz,py,tx,tz=v;center=origin+np.array([0,(front+ry)*width,0]);axes=np.array([rx,ry,rz])*width;powers=np.array([px,py,pz])
    candidate=evaluate(reference,dict(center=center,radii=axes,powers=powers,depth_shear=[tx,tz]))
    hidden,domain=surface(q,v);deficit=np.where(domain>0,np.maximum(skin+.0005-hidden,0),0)*width
    error=(measured-pts[:,1])*width
    return candidate,dict(method='SOURCE_CONDITIONED_CONVEX_GLOBE_VOLUME',center=center.tolist(),radii=axes.tolist(),powers=powers.tolist(),depth_shear=[float(tx),float(tz)],source_depth_rms_world=float(np.sqrt(np.mean(error**2))),source_depth_max_world=float(np.max(np.abs(error))),hidden_sample_max_deficit_world=float(deficit.max(initial=0)),hidden_samples=len(q),evaluations=int(solved.nfev),source_samples=len(observed),topology_preserved=True,hidden_volume_inferred=True,scope='Parametric volume and sampled neutral clearance; native motion acceptance required')


def fit_eyes(data,options,*,progress=None,cancelled=None):
    """Fit whole source globe volumes through the normal surface-fit worker."""
    from .ocular_surface import globe_fidelity
    if not options.get('source_patches'):raise ValueError('Mark complete source eye patches before fitting globe volumes')
    basis=np.asarray(data['eye_basis']);source=np.asarray(data['source'])@basis;source_tri=np.asarray(data['source_triangles']);head=np.asarray(data['current'])@basis;tri=np.asarray(data['triangles']);ray=FrontSurface(source,source_tri)
    proof=dict(before=data['current'],planned=data['current'],ocular_movable=np.zeros(len(head),dtype=bool));reports=[]
    for eye in range(int(options['eye_count'])):
        if progress:progress('Fitting enclosed eye volume '+str(eye+1),eye/options['eye_count'])
        observed=source[np.unique(source_tri[np.asarray(data['source_eye_faces_'+str(eye)],dtype=bool)])]
        key='anatomy_eye_'+str(eye)
        if key not in data:raise ValueError('Prepare the original native eye anatomy before fitting its enclosed volume')
        candidate,volume=fit(np.asarray(data[key])@basis,observed,head,tri,cancelled=cancelled)
        volume['parameterization']='NATIVE_ANATOMY_DIRECTIONS'
        proof['ocular_eye_'+str(eye)]=candidate@basis.T;hull=observed[ConvexHull(observed[:,[0,2]]).vertices][:,[0,2]]
        reports.append(dict(eye=eye,accepted=True,solver=volume['method'],volume=volume,hidden_globe_inferred=True,exported_globe_fidelity=globe_fidelity(candidate,data['eye_triangles_'+str(eye)],ray,hull),native_motion_acceptance=False))
    return np.asarray(data['current']).copy(),reports,proof
