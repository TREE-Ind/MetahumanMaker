"""Smooth spatial velocity regularization of proposed character deformation.

Positive kernel weights and trilinear interpolation couple nearby skin layers.
An adaptive Euler step bound limits the grid field's spatial gradient. Linear
mesh triangles still require a separate collision/quality audit; this module
does not certify global injectivity of the exported piecewise-linear surface.
"""
import numpy as np
from scipy.ndimage import gaussian_filter,map_coordinates


def fit(reference,desired,triangles,*,fixed=None,spacing=.0125,sigma=1.2,max_steps=256,cancelled=None,return_field=False):
    reference=np.asarray(reference,dtype=float);desired=np.asarray(desired,dtype=float);triangles=np.asarray(triangles,dtype=np.int64)
    if reference.shape!=desired.shape or reference.ndim!=2 or reference.shape[1]!=3 or not np.isfinite(reference).all() or not np.isfinite(desired).all():
        raise ValueError('Volume flow requires matching finite vertex arrays')
    if not np.isfinite(spacing) or not np.isfinite(sigma) or spacing<=0 or sigma<=0 or max_steps<1:raise ValueError('Choose finite positive volume spacing, smoothing and step budget')
    if triangles.ndim!=2 or triangles.shape[1]!=3 or not len(triangles) or triangles.min()<0 or triangles.max()>=len(reference):raise ValueError('Choose valid canonical triangles')
    fixed=np.zeros(len(reference),bool) if fixed is None else np.asarray(fixed,dtype=bool)
    if fixed.shape!=(len(reference),):raise ValueError('Flow pins must match vertices')
    displacement=desired-reference;displacement[fixed]=0
    low=np.minimum(reference.min(0),desired.min(0))-spacing*5;high=np.maximum(reference.max(0),desired.max(0))+spacing*5
    shape=np.ceil((high-low)/spacing).astype(int)+1
    if np.prod(shape)>6000000:raise ValueError('Volume grid exceeds memory budget; increase spacing')
    p=reference[triangles];face_area=np.linalg.norm(np.cross(p[:,1]-p[:,0],p[:,2]-p[:,0]),axis=1)*.5
    mass=np.zeros(len(reference))
    for col in range(3):np.add.at(mass,triangles[:,col],face_area/3)
    mass/=max(mass.mean(),1e-15);mass[np.linalg.norm(displacement,axis=1)<1e-8]*=10;mass[fixed]*=10
    grid_weight=np.zeros(tuple(shape));grid_sum=np.zeros((*shape,3))
    coordinate=(reference-low)/spacing;index=np.floor(coordinate).astype(int);fraction=coordinate-index
    for x in (0,1):
        for y in (0,1):
            for z in (0,1):
                corner=np.array([x,y,z]);ids=index+corner;w=mass*np.prod(np.where(corner,fraction,1-fraction),axis=1)
                address=tuple(ids.T);np.add.at(grid_weight,address,w)
                for axis in range(3):np.add.at(grid_sum[...,axis],address,w*displacement[:,axis])
    density=gaussian_filter(grid_weight,sigma,mode='constant');velocity=np.zeros_like(grid_sum)
    for axis in range(3):
        velocity[...,axis]=gaussian_filter(grid_sum[...,axis],sigma,mode='constant')/np.maximum(density,1e-15)
    velocity[density<max(float(density.max())*1e-10,1e-15)]=0
    # Zero boundary values give a continuous extension outside the finite grid.
    for axis in range(3):
        edge=[slice(None)]*4;edge[axis]=0;velocity[tuple(edge)]=0;edge[axis]=-1;velocity[tuple(edge)]=0
    bounds=[float(np.linalg.norm(np.diff(velocity,axis=axis),axis=-1).max(initial=0)/spacing) for axis in range(3)]
    bound=float(np.linalg.norm(bounds));steps=max(8,int(np.ceil(bound*4)))
    if steps>max_steps:raise ValueError('Velocity requires too many bounded integration steps; increase smoothing')
    output=reference.copy()
    for _ in range(steps):
        if cancelled and cancelled():raise RuntimeError('Volume fit cancelled; existing character retained')
        coords=((output-low)/spacing).T
        delta=np.column_stack([map_coordinates(velocity[...,axis],coords,order=1,mode='constant',cval=0,prefilter=False) for axis in range(3)])
        output+=delta/steps;output[fixed]=reference[fixed]
    report=dict(method='AREA_WEIGHTED_SPATIAL_VELOCITY_FLOW',grid=list(map(int,shape)),steps=steps,
        gradient_bound=bound,max_step_gradient=bound/steps,pin_max_error=float(np.linalg.norm(output[fixed]-reference[fixed],axis=1).max(initial=0)),
        collision_review_required=True,production_ready=False)
    if return_field:return output,report,dict(velocity=velocity.astype(np.float32),low=low,spacing=np.array(spacing),steps=np.array(steps))
    return output,report
