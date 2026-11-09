"""Bounded reference tiles for full-resolution Smart Paint color corrections."""
from collections import OrderedDict
import numpy as np

TILE = 64


class ReferencePlane:
    def __init__(self, source, snapshot, width, height, correct, linear, encoded, match_detail):
        self.source = source
        self.snapshot = snapshot
        self.width, self.height = width, height
        self.settings = correct
        self.linear, self.encoded, self.match_detail = linear, encoded, match_detail
        self.cache = OrderedDict()
        self.cache_bytes = 0
        self.max_cache_bytes = 32*1024**2

    def clear(self):
        self.cache.clear();self.cache_bytes = 0

    def sample(self, box):
        x0,x1,y0,y1 = box
        sh,sw = self.source.shape[:2]
        if (sw,sh) == (self.width,self.height):
            return self.source[y0:y1,x0:x1].copy()
        x = np.clip((np.arange(x0,x1)+.5)*sw/self.width-.5,0,sw-1)
        y = np.clip((np.arange(y0,y1)+.5)*sh/self.height-.5,0,sh-1)
        xx=x.astype(int);yy=y.astype(int)
        u=(x-xx).astype(np.float32)[None,:,None]
        v=(y-yy).astype(np.float32)[:,None,None]
        nx=np.minimum(xx+1,sw-1);ny=np.minimum(yy+1,sh-1)
        top=self.source[yy[:,None],xx]*(1-u)+self.source[yy[:,None],nx]*u
        bottom=self.source[ny[:,None],xx]*(1-u)+self.source[ny[:,None],nx]*u
        return top*(1-v)+bottom*v

    def tile(self, tx, ty):
        key=(tx,ty)
        if key in self.cache:
            self.cache.move_to_end(key)
            return self.cache[key]
        x0,y0=tx*TILE,ty*TILE;x1=min(x0+TILE,self.width);y1=min(y0+TILE,self.height)
        exposure,gamma,saturation,preserve,radius=self.settings()
        if preserve:
            # Three separable box passes have a finite support of 3*radius.
            # Include that exact halo so neighboring tiles agree at their edges.
            halo=3*int(radius)
            hx0,hx1=max(0,x0-halo),min(self.width,x1+halo)
            hy0,hy1=max(0,y0-halo),min(self.height,y1+halo)
            matched=self.match_detail(self.linear(self.snapshot[hy0:hy1,hx0:hx1,:3]),
                                      self.linear(self.sample((hx0,hx1,hy0,hy1))),int(radius))
            values=self.encoded(matched[y0-hy0:y1-hy0,x0-hx0:x1-hx0])
        else:
            values=self.sample((x0,x1,y0,y1))
        values*=np.float32(2**exposure)
        np.maximum(values,0,out=values)
        np.power(values,np.float32(1/max(gamma,.01)),out=values)
        lum=values@np.array([.2126,.7152,.0722],dtype=np.float32)
        values-=lum[...,None];values*=np.float32(saturation);values+=lum[...,None]
        np.clip(values,0,1,out=values)
        values=np.ascontiguousarray(values,dtype=np.float32)
        self.cache[key]=values;self.cache_bytes+=values.nbytes
        while self.cache_bytes>self.max_cache_bytes and len(self.cache)>1:
            _old,removed=self.cache.popitem(last=False);self.cache_bytes-=removed.nbytes
        return values

    def patch(self, box):
        x0,x1,y0,y1=box
        result=np.empty((y1-y0,x1-x0,3),dtype=np.float32)
        for ty in range(y0//TILE,(y1-1)//TILE+1):
            for tx in range(x0//TILE,(x1-1)//TILE+1):
                tile=self.tile(tx,ty)
                ax,bx=max(x0,tx*TILE),min(x1,(tx+1)*TILE)
                ay,by=max(y0,ty*TILE),min(y1,(ty+1)*TILE)
                result[ay-y0:by-y0,ax-x0:bx-x0]=tile[ay-ty*TILE:by-ty*TILE,ax-tx*TILE:bx-tx*TILE]
        return result

    def __getitem__(self, item):
        # Preserve the session.reference[y,x] diagnostic API without allocating
        # a complete corrected 8K reference plane.
        y,x=item
        scalar_y=isinstance(y,(int,np.integer));scalar_x=isinstance(x,(int,np.integer))
        if scalar_y:y=slice(int(y)%self.height,int(y)%self.height+1)
        if scalar_x:x=slice(int(x)%self.width,int(x)%self.width+1)
        y0,y1,ys=y.indices(self.height);x0,x1,xs=x.indices(self.width)
        if ys!=1 or xs!=1:raise ValueError('Reference diagnostics support unit-step slices')
        result=self.patch((x0,x1,y0,y1))
        if scalar_y:result=result[0]
        if scalar_x:result=result[0] if scalar_y else result[:,0]
        return result
