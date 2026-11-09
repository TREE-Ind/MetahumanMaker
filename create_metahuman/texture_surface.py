"""Continuous surface patches and visibility-aware atlas reprojection.

No scene renderer, generated depth estimator, model training or Blender APIs.
Depth and camera normals come from the submitted mesh, in a documented frame.
"""
import numpy as np
from scipy import ndimage


def unit(value):
    return value/np.maximum(np.linalg.norm(value,axis=-1,keepdims=True),1e-12)


def validate(points,faces,uv):
    points=np.asarray(points,dtype=np.float32)
    faces=np.asarray(faces,dtype=np.int32)
    uv=np.asarray(uv,dtype=np.float32)
    if points.ndim!=2 or points.shape[1]!=3 or faces.ndim!=2 or faces.shape[1]!=3 or uv.shape!=(len(faces),3,2):
        raise ValueError('Surface patches require mesh points, triangle indices and matching corner UVs')
    if not len(points) or not len(faces) or faces.min()<0 or faces.max()>=len(points):
        raise ValueError('Surface patch mesh has invalid triangle indices')
    if not np.isfinite(points).all() or not np.isfinite(uv).all():
        raise ValueError('Surface patch mesh coordinates must be finite')
    return points,faces,uv


def barycentric(tri,xy):
    a,b=tri[1]-tri[0],tri[2]-tri[0]
    determinant=a[0]*b[1]-a[1]*b[0]
    if abs(determinant)<1e-12:return None
    p=xy-tri[0]
    u=(p[...,0]*b[1]-p[...,1]*b[0])/determinant
    v=(a[0]*p[...,1]-a[1]*p[...,0])/determinant
    return np.stack([1-u-v,u,v],axis=-1)


def pixels(tri,size):
    w,h=size
    low=np.maximum(np.floor(tri.min(axis=0)).astype(int),0)
    high=np.minimum(np.ceil(tri.max(axis=0)).astype(int),[w-1,h-1])
    if np.any(high<low):return None
    y,x=np.mgrid[low[1]:high[1]+1,low[0]:high[0]+1]
    weights=barycentric(tri,np.stack([x,y],axis=-1))
    if weights is None:return None
    inside=np.min(weights,axis=-1)>=-1e-5
    return y[inside],x[inside],weights[inside]


def sample(image,uv):
    h,w=image.shape[:2]
    x=uv[...,0]*w-.5;y=(1-uv[...,1])*h-.5
    if image.ndim==2:
        return ndimage.map_coordinates(image,[y,x],order=1,mode='nearest')
    return np.stack([ndimage.map_coordinates(image[...,c],[y,x],order=1,mode='nearest')
                     for c in range(image.shape[2])],axis=-1)


class Surface:
    def __init__(self,points,faces,uv):
        self.points,self.faces,self.uv=validate(points,faces,uv)
        self.triangles=self.points[self.faces]
        cross=np.cross(self.triangles[:,1]-self.triangles[:,0],self.triangles[:,2]-self.triangles[:,0])
        self.face_normals=unit(cross)
        # Native UV charts use split vertices: smooth actual coincident mesh
        # points together so the conditioning normal does not invent chart seams.
        tolerance=max(float(np.ptp(self.points,axis=0).max())*1e-7,1e-9)
        _,weld=np.unique(np.rint(self.points/tolerance).astype(np.int64),axis=0,return_inverse=True)
        normals=np.zeros((int(weld.max())+1,3),np.float32)
        for index in range(3):np.add.at(normals,weld[self.faces[:,index]],cross)
        self.normals=unit(normals)[weld]
        self.neighbors=np.full((len(self.faces),4),-1,np.int32)
        self.neighbors[:,0]=np.arange(len(self.faces))
        edge_faces={}
        for index,tri in enumerate(weld[self.faces]):
            for slot in range(3):
                key=tuple(sorted((int(tri[slot]),int(tri[(slot+1)%3]))))
                edge_faces.setdefault(key,[]).append((index,slot+1))
        for entries in edge_faces.values():
            if len(entries)==2:
                (a,sa),(b,sb)=entries
                self.neighbors[a,sa]=b;self.neighbors[b,sb]=a

    def camera(self,coordinate,atlas_size,texels=512):
        a=self.uv[:,1]-self.uv[:,0];b=self.uv[:,2]-self.uv[:,0];p=coordinate-self.uv[:,0]
        determinant=a[:,0]*b[:,1]-a[:,1]*b[:,0]
        safe=np.abs(determinant)>1e-12
        u=np.divide(p[:,0]*b[:,1]-p[:,1]*b[:,0],determinant,out=np.zeros(len(a)),where=safe)
        v=np.divide(a[:,0]*p[:,1]-a[:,1]*p[:,0],determinant,out=np.zeros(len(a)),where=safe)
        hits=np.flatnonzero(safe&(u>=0)&(v>=0)&(u+v<=1))
        if not len(hits):raise ValueError('The chosen skin point is outside actual UV triangles')
        index=int(hits[0]);weights=np.array([1-u[index]-v[index],u[index],v[index]])
        center=weights@self.triangles[index]
        normal=unit(weights@self.normals[self.faces[index]])
        up=np.array([0,0,1.],np.float32)
        if abs(float(up@normal))>.95:up=np.array([0,1.,0],np.float32)
        right=unit(np.cross(up,normal));up=unit(np.cross(normal,right))
        basis=np.stack([right,up,normal])
        jacobian=np.linalg.solve(np.stack([a[index],b[index]]),self.triangles[index,1:]-self.triangles[index,0])
        # Match the source atlas's local physical texel density, allowing context.
        span=float(max(np.linalg.norm(jacobian[0])/atlas_size[0],
                       np.linalg.norm(jacobian[1])/atlas_size[1])*texels)
        if not np.isfinite(span) or span<=0:raise ValueError('The local UV scale is degenerate')
        return {'center':center.tolist(),'basis':basis.tolist(),'span':span,'triangle':index,
                'uv_center':np.asarray(coordinate).tolist()}

    def render(self,image,camera,size=512):
        center=np.asarray(camera['center']);basis=np.asarray(camera['basis']);span=camera['span']
        local=(self.triangles-center)@basis.T
        xy=local[...,:2]*[size/span,-size/span]+(size/2-.5)
        depth=np.full((size,size),-np.inf,np.float32)
        ids=np.full((size,size),-1,np.int32)
        uvmap=np.zeros((size,size,2),np.float32)
        normals=np.zeros((size,size,3),np.float32)
        facing=self.face_normals@basis[2]
        candidates=np.flatnonzero((xy.max(axis=1)[:,0]>=0)&(xy.min(axis=1)[:,0]<size)&
            (xy.max(axis=1)[:,1]>=0)&(xy.min(axis=1)[:,1]<size)&(facing>.05))
        for index in candidates:
            raster=pixels(xy[index],(size,size))
            if raster is None:continue
            y,x,weights=raster
            z=weights@local[index,:,2]
            visible=z>depth[y,x]
            y,x,weights,z=y[visible],x[visible],weights[visible],z[visible]
            depth[y,x]=z;ids[y,x]=index
            uvmap[y,x]=weights@self.uv[index]
            normals[y,x]=unit(weights@self.normals[self.faces[index]])@basis.T
        valid=ids>=0
        color=sample(image,uvmap);color[~valid]=0
        view_normals=normals*.5+.5;view_normals[~valid]=0
        relative_depth=np.zeros_like(depth)
        if np.any(valid):
            near=float(depth[valid].max());far=float(depth[valid].min())
            relative_depth[valid]=(depth[valid]-far)/max(near-far,1e-8)
        else:near=far=0.
        return {'color':color,'uv':uvmap,'valid':valid,'depth_world':depth,'depth':relative_depth,
                'normal':view_normals,'triangle_ids':ids,'camera':camera,
                'depth_range':[far,near],'depth_convention':'orthographic, near white, invalid black',
                'normal_convention':'camera right/up/toward-camera mapped from -1..1 to 0..1'}

    def raster_cache(self,mask,protect=None):
        """Map selected atlas texels to geometry once for an immutable solve."""
        h,w=mask.shape;rows=[];columns=[];ids=[];points=[];allow=[]
        for index,uv in enumerate(self.uv):
            raster=pixels(uv*[w,-h]+[-.5,h-.5],(w,h))
            if raster is None:continue
            y,x,weights=raster
            permitted=mask[y,x].copy()
            if protect is not None:permitted*=1-protect[y,x]
            keep=permitted>0
            if not np.any(keep):continue
            rows.append(y[keep].astype(np.int32));columns.append(x[keep].astype(np.int32))
            ids.append(np.full(int(keep.sum()),index,np.int32))
            points.append(weights[keep]@self.triangles[index]);allow.append(permitted[keep])
        if not rows:raise ValueError('No selected skin texels map to the submitted mesh')
        return {'surface':self,'mask':mask,'protect':protect,'shape':(h,w),
            'y':np.concatenate(rows),'x':np.concatenate(columns),'ids':np.concatenate(ids),
            'points':np.concatenate(points),'allowed':np.concatenate(allow)}

    def _project_cached(self,patch,render,source,mask,protect,accumulate,cache):
        if cache['surface'] is not self or cache['mask'] is not mask or cache['protect'] is not protect:
            raise ValueError('Surface projection cache belongs to different immutable inputs')
        h,w=source.shape[:2]
        if cache['shape']!=(h,w):raise ValueError('Surface cache resolution differs')
        if accumulate is None:
            sums=np.zeros_like(source) if patch is not None else None
            weights_out=np.zeros((h,w),np.float32)
        else:sums,weights_out=accumulate
        camera=render['camera'];size=render['valid'].shape[0]
        center=np.asarray(camera['center']);basis=np.asarray(camera['basis']);span=camera['span']
        visible_ids=np.unique(render['triangle_ids']);visible_ids=visible_ids[visible_ids>=0]
        visible_faces=np.zeros(len(self.faces),dtype=bool);visible_faces[visible_ids]=True
        for start in range(0,len(cache['ids']),65536):
            chunk=slice(start,start+65536)
            locations=(cache['points'][chunk]-center)@basis.T
            xy=locations[:,:2]*[size/span,-size/span]+(size/2-.5)
            inside=(xy[:,0]>=0)&(xy[:,0]<=size-1)&(xy[:,1]>=0)&(xy[:,1]<=size-1)&visible_faces[cache['ids'][chunk]]
            if not np.any(inside):continue
            xy=xy[inside];locations=locations[inside];indices=cache['ids'][chunk][inside]
            px,py=np.rint(xy).astype(int).T
            visible=render['triangle_ids'][py,px]
            same=np.any(visible[:,None]==self.neighbors[indices],axis=1)&(visible>=0)
            good=same&(np.abs(render['depth_world'][py,px]-locations[:,2])<=span/size)
            if not np.any(good):continue
            xy=xy[good];indices=indices[good]
            y=cache['y'][chunk][inside][good];x=cache['x'][chunk][inside][good]
            allowed=cache['allowed'][chunk][inside][good]
            edge=np.minimum.reduce([xy[:,0],xy[:,1],size-1-xy[:,0],size-1-xy[:,1]])
            angle=np.maximum(self.face_normals[indices]@basis[2],0.)
            weight=np.clip(edge/32,0,1)*angle**2*allowed
            if patch is not None:
                uv=np.stack([(xy[:,0]+.5)/size,1-(xy[:,1]+.5)/size],axis=-1)
                colors=sample(patch,uv)
                for channel in range(3):np.add.at(sums[...,channel],(y,x),colors[:,channel]*weight)
            np.add.at(weights_out,(y,x),weight)
        return sums,weights_out

    def project(self,patch,render,source,mask,protect=None,*,accumulate=None,cache=None):
        """Return atlas color sums and weights for visible eligible texels only."""
        if cache is not None:return self._project_cached(patch,render,source,mask,protect,accumulate,cache)
        h,w=source.shape[:2];camera=render['camera'];size=render['valid'].shape[0]
        center=np.asarray(camera['center']);basis=np.asarray(camera['basis']);span=camera['span']
        if accumulate is None:
            sums=np.zeros_like(source) if patch is not None else None
            weights_out=np.zeros((h,w),np.float32)
        else:
            sums,weights_out=accumulate
            if patch is None or sums.shape!=source.shape or weights_out.shape!=(h,w):
                raise ValueError('Surface accumulation buffers have incompatible dimensions')
        visible_ids=np.unique(render['triangle_ids']);visible_ids=visible_ids[visible_ids>=0]
        for index in visible_ids:
            tri=self.uv[index]*[w,-h]+[-.5,h-.5]
            raster=pixels(tri,(w,h))
            if raster is None:continue
            y,x,weights=raster
            allowed=mask[y,x].copy()
            if protect is not None:allowed*=1-protect[y,x]
            if not np.any(allowed>0):continue
            points=(weights@self.triangles[index]-center)@basis.T
            xy=points[:,:2]*[size/span,-size/span]+(size/2-.5)
            inside=(xy[:,0]>=0)&(xy[:,0]<=size-1)&(xy[:,1]>=0)&(xy[:,1]<=size-1)&(allowed>0)
            if not np.any(inside):continue
            y,x,points,xy,allowed=y[inside],x[inside],points[inside],xy[inside],allowed[inside]
            px,py=np.rint(xy).astype(int).T
            # Adjacent triangles share continuous skin, including split UV
            # charts. Require actual adjacency as well as depth agreement;
            # unrelated occluded eyelid/mouth surfaces cannot pass this test.
            visible=render['triangle_ids'][py,px]
            same=np.any(visible[:,None]==self.neighbors[index][None,:],axis=1)&(visible>=0)
            depth_match=np.abs(render['depth_world'][py,px]-points[:,2])<=span/size
            valid=same&depth_match
            if not np.any(valid):continue
            y,x,xy,allowed=y[valid],x[valid],xy[valid],allowed[valid]
            edge=np.minimum.reduce([xy[:,0],xy[:,1],size-1-xy[:,0],size-1-xy[:,1]])
            angle=max(float(self.face_normals[index]@basis[2]),0.)
            weight=np.clip(edge/32,0,1)*angle**2*allowed
            patch_uv=np.stack([(xy[:,0]+.5)/size,1-(xy[:,1]+.5)/size],axis=-1)
            if patch is not None:
                color=sample(patch,patch_uv)
                sums[y,x]+=color*weight[:,None]
            weights_out[y,x]+=weight
        return sums,weights_out


def plan(surface,source,mask,protect=None,*,limit=48,progress=None,span_limits=None):
    """Cover selected skin with physical patches, independent of UV charts."""
    h,w=mask.shape
    working=mask>.05
    if protect is not None:working&=protect<.95
    cameras=[]
    cache=surface.raster_cache(mask,protect)
    initial=int(working.sum())
    while int(working.sum())>=64 and len(cameras)<limit:
        distance=ndimage.distance_transform_edt(np.pad(working[::8,::8],1))[1:-1,1:-1]
        cy,cx=np.unravel_index(int(distance.argmax()),distance.shape)
        y,x=cy*8,cx*8
        if not working[y,x]:break
        try:camera=surface.camera(np.array([(x+.5)/w,1-(y+.5)/h]),(w,h))
        except ValueError:
            working[max(0,y-4):y+5,max(0,x-4):x+5]=False
            continue
        if span_limits is not None:
            camera['span']=float(np.clip(camera['span'],*span_limits))
        render=surface.render(source,camera)
        _,weights=surface.project(None,render,source,mask,protect,cache=cache)
        covered=weights>.05
        count=int(np.sum(working&covered))
        if count<64:
            working[max(0,y-4):y+5,max(0,x-4):x+5]=False
            continue
        cameras.append(camera);working[covered]=False
        if progress:progress(len(cameras),limit,'Planning surface patches')
        del render,weights
    return cameras,{'selected_texels':initial,'unplanned_texels':int(working.sum()),
                    'planned_patches':len(cameras),'patch_limit':limit}
