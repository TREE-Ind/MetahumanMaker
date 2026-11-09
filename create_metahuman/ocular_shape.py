"""Geometry diagnostics for fitted globes before animated rig acceptance."""
import numpy as np
from scipy.spatial import ConvexHull,QhullError


def radial_recess(points,*,relative_threshold=.02):
    """Measure inward dents relative to the globe's own convex envelope.

    This does not alter the mesh or assert that every eye must be convex.
    Large hidden recesses need explicit motion review: rigid gaze can expose
    them even when the neutral aperture looks faithful to its source.
    """
    points=np.asarray(points,dtype=float)
    if points.ndim!=2 or points.shape[1]!=3 or len(points)<4 or not np.isfinite(points).all() or not 0<relative_threshold<1:raise ValueError('Invalid globe shape diagnostic')
    scale=float(np.ptp(points,axis=0).max())
    if scale<=0:raise ValueError('Globe has no spatial extent')
    q=(points-points.mean(0))/scale
    try:hull=ConvexHull(q)
    except QhullError as error:raise ValueError('Globe does not define a three-dimensional surface') from error
    center=q[hull.vertices].mean(0);directions=q-center
    if np.any(np.linalg.norm(directions,axis=1)<1e-12):raise ValueError('Globe contains a vertex at its interior center')
    numerator=-(hull.equations[:,:3]@center+hull.equations[:,3]);denominator=directions@hull.equations[:,:3].T
    limits=np.divide(numerator[None,:],denominator,out=np.full_like(denominator,np.inf),where=denominator>1e-12).min(1)
    if not np.isfinite(limits).all():raise ValueError('Could not measure the complete globe envelope')
    recess=np.maximum(limits-1.,0)*np.linalg.norm(directions,axis=1)*scale
    maximum=float(recess.max());ratio=maximum/scale
    return dict(method='RADIAL_CONVEX_ENVELOPE_RECESS',maximum_recess_world=maximum,maximum_recess_relative=ratio,vertices=len(points),
        vertices_over_threshold=int(np.sum(recess>relative_threshold*scale)),relative_threshold=relative_threshold,requires_motion_review=bool(ratio>relative_threshold),
        limitation='Shape diagnostic only; no collision or motion safety certificate')
