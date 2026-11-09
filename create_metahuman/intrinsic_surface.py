"""Triangle-section surface correspondence in explicitly labelled limb frames.

This is a shape initializer. Region labels and joint frames must be supplied;
it does not infer anatomy or certify injectivity/collision freedom.
"""
import numpy as np
from .profile_fit import surface_profile


def frame(direction,transverse):
    axis=np.asarray(direction,dtype=float);axis=axis/np.linalg.norm(axis)
    side=np.asarray(transverse,dtype=float);side=side-axis*(axis@side);side/=np.linalg.norm(side)
    return np.column_stack((side,np.cross(axis,side),axis))


def interpolate(grid,values,z,theta=None):
    if theta is None:return np.column_stack([np.interp(z,grid,values[:,i]) for i in range(values.shape[1])])
    t=np.clip((z-grid[0])/(grid[-1]-grid[0])*(len(grid)-1),0,len(grid)-1)
    i=np.minimum(t.astype(int),len(grid)-2);f=t-i
    a=np.mod(theta,2*np.pi)/(2*np.pi)*values.shape[1];j=a.astype(int)%values.shape[1];g=a-np.floor(a);k=(j+1)%values.shape[1]
    return (1-f)*((1-g)*values[i,j]+g*values[i,k])+f*((1-g)*values[i+1,j]+g*values[i+1,k])


def fit(query,reference,reference_triangles,source,source_triangles,reference_start,reference_end,source_start,source_end,
        *,transverse=(0,1,0),stations=61,angles=96,match_extents=False):
    query=np.asarray(query,dtype=float);reference=np.asarray(reference,dtype=float);source=np.asarray(source,dtype=float)
    a=np.asarray(reference_start,dtype=float);b=np.asarray(reference_end,dtype=float);c=np.asarray(source_start,dtype=float);d=np.asarray(source_end,dtype=float)
    basis=frame(b-a,transverse);target_basis=frame(d-c,transverse)
    local=(reference-a)@basis;target=(source-c)@target_basis;p=(query-a)@basis
    rz,rc,rr,rs=surface_profile(local,reference_triangles,stations=stations,angles=angles)
    sz,sc,sr,ss=surface_profile(target,source_triangles,stations=stations,angles=angles)
    z=p[:,2]/np.linalg.norm(b-a)*np.linalg.norm(d-c)
    if match_extents:z=sz[0]+(p[:,2]-rz[0])/(rz[-1]-rz[0])*(sz[-1]-sz[0])
    offset=p[:,:2]-interpolate(rz,rc,p[:,2]);theta=np.arctan2(offset[:,1],offset[:,0])
    scale=interpolate(sz,sr,z,theta)/np.maximum(interpolate(rz,rr,p[:,2],theta),1e-10)
    positions=np.column_stack((interpolate(sz,sc,z)+offset*scale[:,None],z))@target_basis.T+c
    support=interpolate(sz,ss.astype(float),z,theta)*interpolate(rz,rs.astype(float),p[:,2],theta)
    confidence=np.clip(support,0,1)*((z>=sz[0])&(z<=sz[-1])&(p[:,2]>=rz[0])&(p[:,2]<=rz[-1]))
    confidence*=((scale>=.4)&(scale<=3.))
    return positions,confidence,{'solver':'INTRINSIC_TRIANGLE_ENVELOPE','source_supported_fraction':float(ss.mean()),
        'reference_supported_fraction':float(rs.mean()),'maximum_scale':float(scale.max(initial=0)),
        'collision_review_required':True}
