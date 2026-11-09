"""Complete explicitly marked pigment on a physical mesh graph.

Positive edge conductances give a bounded harmonic color field in linear RGB.
Only requested atlas pixels are painted; this infers missing low-frequency
pigment and does not synthesize pores, recover physical albedo or change maps.
"""
import numpy as np
from scipy import sparse
from scipy.sparse.csgraph import connected_components
from scipy.sparse.linalg import spsolve


def complete(source,region,geometry,*,unknown=None,protect=None,strength=1.):
    if geometry is None:raise ValueError('Surface color completion requires the actual mesh positions, faces and UV triangles')
    points,faces,uv=(np.asarray(v) for v in geometry)
    if points.ndim!=2 or points.shape[1]!=3 or faces.ndim!=2 or faces.shape[1]!=3 or not len(faces) or uv.shape!=(len(faces),3,2) or not np.issubdtype(faces.dtype,np.integer) or faces.min()<0 or faces.max()>=len(points) or not np.isfinite(points).all() or not np.isfinite(uv).all():raise ValueError('Surface completion topology is incomplete or invalid')
    source=np.asarray(source);region=np.asarray(region,dtype=np.float32)
    if source.dtype!=np.uint8 or source.ndim!=3 or source.shape[2] not in (3,4) or region.shape!=source.shape[:2] or not np.isfinite(region).all() or region.min()<0 or region.max()>1 or not np.isfinite(strength) or not 0<=strength<=1:raise ValueError('Supply bounded color, region and strength inputs')
    unknown=region if unknown is None else np.asarray(unknown)
    if unknown.shape!=region.shape or not np.isfinite(unknown).all() or unknown.min()<0 or unknown.max()>1:raise ValueError('Surface completion needs a matching unknown-pigment mask')
    if protect is not None:
        protect=np.asarray(protect)
        if protect.shape!=region.shape or not np.isfinite(protect).all() or protect.min()<0 or protect.max()>1:raise ValueError('Surface completion protection is invalid')
        region=region*(1-protect)
    if not np.any(region>0) or strength==0:return source.copy(),dict(method='SURFACE_HARMONIC_COLOR',changed_texels=0,inferred=True,inference_skipped=True)
    used,local=np.unique(faces,return_inverse=True);faces=local.reshape(-1,3);points=points[used];height,width=region.shape
    pixel=uv*np.array([width-1,-(height-1)])+[0,height-1]
    if pixel[:,:,0].min()<-1e-4 or pixel[:,:,0].max()>width-1+1e-4 or pixel[:,:,1].min()<-1e-4 or pixel[:,:,1].max()>height-1+1e-4:raise ValueError('Localize the skin UV tile before surface completion')
    def sample(image):
        x=pixel[:,:,0].clip(0,width-1);y=pixel[:,:,1].clip(0,height-1);ix=x.astype(int);iy=y.astype(int);jx=np.minimum(ix+1,width-1);jy=np.minimum(iy+1,height-1);tx=x-ix;ty=y-iy
        if image.ndim==3:tx=tx[:,:,None];ty=ty[:,:,None]
        return image[iy,ix]*(1-tx)*(1-ty)+image[iy,jx]*tx*(1-ty)+image[jy,ix]*(1-tx)*ty+image[jy,jx]*tx*ty
    rejected=sample(unknown)>0;marked=np.zeros(len(points),bool);np.logical_or.at(marked,faces.ravel(),rejected.ravel());color=sample(source[:,:,:3])/255; color=np.where(color<=.04045,color/12.92,((color+.055)/1.055)**2.4);trusted=~rejected;count=np.bincount(faces.ravel(),weights=trusted.ravel(),minlength=len(points));values=np.column_stack([np.bincount(faces.ravel(),weights=(color[:,:,axis]*trusted).ravel(),minlength=len(points))/np.maximum(count,1) for axis in range(3)]);marked|=count==0
    edges=np.unique(np.sort(np.vstack((faces[:,[0,1]],faces[:,[1,2]],faces[:,[2,0]])),axis=1),axis=0);length=np.linalg.norm(points[edges[:,0]]-points[edges[:,1]],axis=1)
    if np.any(length<=1e-12):raise ValueError('The skin graph has collapsed physical edges; supply canonical topology')
    weight=np.median(length)/length;adjacency=sparse.coo_matrix((np.r_[weight,weight],(np.r_[edges[:,0],edges[:,1]],np.r_[edges[:,1],edges[:,0]])),shape=(len(points),len(points))).tocsr();_,labels=connected_components(adjacency,directed=False)
    for component in np.unique(labels[marked]):
        if not np.any((labels==component)&~marked):raise ValueError('A marked skin component has no trusted boundary pigment')
    free=np.flatnonzero(marked);fixed=np.flatnonzero(~marked);lap=sparse.diags(np.asarray(adjacency.sum(1)).ravel())-adjacency
    if len(free):values[free]=spsolve(lap[free][:,free].tocsc(),-lap[free][:,fixed]@values[fixed])
    if not np.isfinite(values).all() or values.min()<-1e-7 or values.max()>1+1e-7:raise ValueError('Surface color completion failed its bounded harmonic solve')
    y,x=np.nonzero(region>0);x0,x1=int(x.min()),int(x.max())+1;y0,y1=int(y.min()),int(y.max())+1;proposal=np.zeros((y1-y0,x1-x0,3),dtype=np.float32);covered=np.zeros(proposal.shape[:2],bool);active=region[y0:y1,x0:x1]>0;conflict=0.
    for face,xy in zip(faces,pixel):
        lo=np.maximum(np.floor(xy.min(0)).astype(int),[x0,y0]);hi=np.minimum(np.ceil(xy.max(0)).astype(int)+1,[x1,y1])
        if np.any(hi<=lo):continue
        if not active[lo[1]-y0:hi[1]-y0,lo[0]-x0:hi[0]-x0].any():continue
        a,b=xy[1]-xy[0],xy[2]-xy[0];det=a[0]*b[1]-a[1]*b[0]
        if abs(det)<1e-12:continue
        xx,yy=np.meshgrid(np.arange(lo[0],hi[0]),np.arange(lo[1],hi[1]));q=np.stack((xx,yy),axis=-1)-xy[0];u=(q[:,:,0]*b[1]-q[:,:,1]*b[0])/det;v=(a[0]*q[:,:,1]-a[1]*q[:,:,0])/det;inside=(u>=-1e-7)&(v>=-1e-7)&(u+v<=1+1e-7);sy=slice(lo[1]-y0,hi[1]-y0);sx=slice(lo[0]-x0,hi[0]-x0);inside&=active[sy,sx];field=values[face[0]]*(1-u-v)[:,:,None]+values[face[1]]*u[:,:,None]+values[face[2]]*v[:,:,None];old=proposal[sy,sx];duplicate=inside&covered[sy,sx]
        if duplicate.any():conflict=max(conflict,float(np.max(np.abs(old[duplicate]-field[duplicate]))))
        old[inside]=field[inside];covered[sy,sx]|=inside
    if np.any(active&~covered):raise ValueError('A painted texel is outside the supplied completion surface; protect that surface boundary')
    if conflict>2/255:raise ValueError('Overlapping UV faces disagree on the inferred skin color')
    encoded=np.where(proposal<=.0031308,proposal*12.92,1.055*np.maximum(proposal,0)**(1/2.4)-.055);result=source.copy();amount=region[y0:y1,x0:x1,None]*strength;original=source[y0:y1,x0:x1,:3];result[y0:y1,x0:x1,:3]=np.rint(original*(1-amount)+encoded.clip(0,1)*255*amount).clip(0,255).astype(np.uint8)
    return result,dict(method='SURFACE_HARMONIC_COLOR',inferred=True,changed_texels=int(np.any(result!=source,axis=2).sum()),physical_graph_vertices=len(points),unknown_graph_vertices=len(free),unknown_color_texels=int(np.count_nonzero(unknown)),crop=[x0,y0,x1,y1],overlap_color_residual=conflict,outside_mask_unchanged=True,alpha_unchanged=True,skin_microdetail_synthesized=False)
