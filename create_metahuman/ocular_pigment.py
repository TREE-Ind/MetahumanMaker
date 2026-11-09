"""Conservative completion of a flat dark pupil hidden by source eyelids.

This is optional inferred pigment, not recovered source detail. A measured
dark/sclera silhouette must support a stable ellipse; textured or ambiguous
eyes retain their measured color instead. Observed interior is never changed.
"""
import numpy as np
from scipy.ndimage import binary_erosion, distance_transform_edt, label
from scipy.optimize import least_squares


def fit(atlas, domain, covered, marked, sclera, low, extent, transition):
    luminance = atlas[:,:,:3] @ np.array([.2126,.7152,.0722])
    base = float(sclera[:3] @ np.array([.2126,.7152,.0722]))
    if base < .08:
        return None, dict(status='retained',reason='Insufficient pupil/sclera contrast')
    # Marking identifies the eye surface, not a pigment boundary: users may
    # legitimately include dark pupil facets in their sclera selection.
    dark = domain & covered & (luminance < base*.18)
    components, count = label(dark)
    if not count:
        return None, dict(status='retained',reason='No isolated dark pigment')
    areas = np.bincount(components.ravel());areas[0]=0
    dark = components == int(areas.argmax())
    distance = distance_transform_edt(np.pad(domain,1))[1:-1,1:-1]
    boundary = dark & ~binary_erosion(dark) & (distance > transition*3)
    y,x = np.nonzero(boundary);size=atlas.shape[0];scale=float(max(extent))
    if len(x)<48 or dark.sum()<domain.sum()*.005 or dark.sum()>domain.sum()*.65:
        return None, dict(status='retained',reason='Insufficient unclipped pigment silhouette')
    points=(np.c_[x+.5,y+.5]/size*extent)/scale
    # Axis-aligned ellipse in the frontal source frame. A tilted/textured
    # silhouette is intentionally rejected by the fit residual gate.
    y0,x0=np.nonzero(dark);cloud=np.c_[x0+.5,y0+.5]/size*extent/scale
    lo,hi=cloud.min(0),cloud.max(0);initial=np.r_[(lo+hi)*.5,np.maximum((hi-lo)*.5,.02)]
    def residual(v):return (np.sqrt(np.sum(((points-v[:2])/v[2:])**2,axis=1))-1)*np.mean(v[2:])
    solution=least_squares(residual,initial,bounds=(np.r_[lo-.3,[.015,.015]],np.r_[hi+.3,[.65,.65]]),loss='soft_l1',f_scale=.002,max_nfev=250)
    center,radii=solution.x[:2],solution.x[2:];error=np.abs(residual(solution.x))
    angles=np.sort(np.mod(np.arctan2((points[:,1]-center[1])/radii[1],(points[:,0]-center[0])/radii[0]),2*np.pi))
    coverage=2*np.pi-float(np.diff(np.r_[angles,angles[0]+2*np.pi]).max())
    metrics=dict(boundary_samples=len(x),silhouette_rms_world=float(np.sqrt(np.mean(error**2))*scale),arc_degrees=float(np.degrees(coverage)))
    if not solution.success or np.sqrt(np.mean(error**2))>.008 or np.quantile(error,.95)>.018 or coverage<np.pi*.65 or radii.max()/radii.min()>2.5 or np.any(center-radii<-.3) or np.any(center+radii>extent/scale+.3):
        return None,dict(status='retained',reason='Ambiguous elliptical pigment continuation',**metrics)
    color=np.median(atlas[binary_erosion(dark)],axis=0)
    model=dict(center=(low+center*scale).tolist(),radii=(radii*scale).tolist(),color=color.tolist())
    return model,dict(status='inferred',method='OBSERVED_DARK_SILHOUETTE_ELLIPSE',observed_interior_unchanged=True,model=model,**metrics)


def evaluate(query, model, sclera, feather):
    center=np.asarray(model['center']);radii=np.asarray(model['radii'])
    radial=np.sqrt(np.sum(((query-center)/radii)**2,axis=1))
    weight=np.clip((1-radial)*radii.min()/feather+.5,0,1)
    weight=weight*weight*(3-2*weight)
    return sclera*(1-weight[:,None])+np.asarray(model['color'])*weight[:,None]
