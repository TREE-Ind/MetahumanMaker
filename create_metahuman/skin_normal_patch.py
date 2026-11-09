"""Transport an exact regional normal edit onto a new UV-matched surface."""
import numpy as np


def apply(normal,before,after,progress=None):
    """Rotate changed texels by the shortest before→after direction change.

    Unchanged edit texels retain the new geometry exactly. Replaying on the
    original input preserves the artist's quantized output exactly, including
    non-unit source normals. Scratch arithmetic is bounded to 128-row bands.
    """
    if (normal.ndim!=3 or normal.shape[-1]!=3 or before.shape!=normal.shape or
        after.shape!=normal.shape or before.dtype!=np.uint8 or after.dtype!=np.uint8 or
        normal.dtype!=np.float32 or not normal.flags.writeable):
        raise ValueError('Regional skin replay needs float RGB normals and matching byte RGB edit inputs')
    for y in range(0,len(normal),128):
        current=normal[y:y+128];old=before[y:y+128];new=after[y:y+128]
        changed=np.any(old!=new,axis=2)
        if not np.any(changed):continue
        if not np.isfinite(current).all() or current.min()<0 or current.max()>1:
            raise ValueError('Regional skin replay encountered invalid geometry normals')
        a=old[changed].astype(np.float32)/255;b=new[changed].astype(np.float32)/255
        n=current[changed]
        exact=np.all(n==a,axis=1)
        a=a*2-1;b=b*2-1;n=n*2-1
        for values in (a,b,n):values/=np.maximum(np.linalg.norm(values,axis=1,keepdims=True),1e-8)
        xyz=np.cross(a,b);w=1+np.einsum('ij,ij->i',a,b)
        opposite=w<1e-6
        if np.any(opposite):
            axes=np.zeros_like(a[opposite]);axes[np.arange(len(axes)),np.argmin(abs(a[opposite]),axis=1)]=1
            xyz[opposite]=np.cross(a[opposite],axes);w[opposite]=0
        length=np.sqrt(np.einsum('ij,ij->i',xyz,xyz)+w*w)
        xyz/=np.maximum(length[:,None],1e-8);w/=np.maximum(length,1e-8)
        result=n+2*np.cross(xyz,np.cross(xyz,n)+w[:,None]*n)
        result/=np.maximum(np.linalg.norm(result,axis=1,keepdims=True),1e-8)
        result=result*.5+.5
        result[exact]=new[changed][exact].astype(np.float32)/255
        current[changed]=np.clip(result,0,1)
        if progress:progress(min(y+128,len(normal)),len(normal),'Replaying regional skin')
    return normal
