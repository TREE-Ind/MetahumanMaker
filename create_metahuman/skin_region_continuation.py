"""Continue tangent-space normals through a bounded, explicitly selected defect.

The boundary supplies the replacement direction, rather than another character's
anatomy. This fills shading only; it does not reconstruct geometric relief.
"""
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve


def remove_obsolete_relief(source, mask, *, radius=16, feather=16, detail_strength=1.):
    """Retain fine tangent detail after reconstructing the underlying surface.

    Explicitly selected coarse normal relief is replaced by the new geometry.
    Negative-Z normals cannot describe visible skin and contribute no detail.
    This is not a substitute for baking a general deformation's tangent basis.
    """
    from scipy.ndimage import gaussian_filter, distance_transform_edt
    if source.dtype!=np.uint8 or source.ndim!=3 or source.shape[2]!=3 or mask.shape!=source.shape[:2]:
        raise ValueError('Use byte RGB normals and a matching explicit repair mask')
    selected=mask>0
    if not selected.any() or radius<=0 or feather<=0 or not 0<=detail_strength<=1:raise ValueError('Select relief, positive filter sizes and a detail strength from zero to one')
    yy,xx=np.where(selected);pad=int(radius*4+feather+2)
    y0,y1=max(0,yy.min()-pad),min(len(mask),yy.max()+pad+1)
    x0,x1=max(0,xx.min()-pad),min(mask.shape[1],xx.max()+pad+1)
    original=source[y0:y1,x0:x1].astype(np.float32)/127.5-1
    valid=original[:,:,2]>.1
    weight=gaussian_filter(valid.astype(np.float32),radius)
    detail=np.zeros_like(original)
    for axis in range(2):
        coarse=gaussian_filter(original[:,:,axis]*valid,radius)/np.maximum(weight,1e-6)
        detail[:,:,axis]=np.where(valid,original[:,:,axis]-coarse,0)
    detail[:,:,:2]*=detail_strength
    length=np.linalg.norm(detail[:,:,:2],axis=2);detail[:,:,:2]/=np.maximum(length/.65,1)[:,:,None]
    detail[:,:,2]=np.sqrt(np.maximum(1-(detail[:,:,:2]**2).sum(2),0))
    alpha=np.clip(distance_transform_edt(selected[y0:y1,x0:x1])/feather,0,1)
    alpha=alpha*alpha*(3-2*alpha)
    mixed=original*(1-alpha[:,:,None])+detail*alpha[:,:,None]
    mixed/=np.maximum(np.linalg.norm(mixed,axis=2,keepdims=True),1e-8)
    result=source.copy();patch=result[y0:y1,x0:x1];local=selected[y0:y1,x0:x1]
    encoded=np.rint(np.clip(mixed*.5+.5,0,1)*255).astype(np.uint8);patch[local]=encoded[local]
    assert np.array_equal(result[~selected],source[~selected])
    return result,dict(method='REMOVE_OBSOLETE_COARSE_RELIEF',selected_texels=int(selected.sum()),
        outside_mask_bytes_exact=True,retained_detail_radius_px=radius,feather_px=feather,detail_strength=detail_strength)


def normal_reference(source,mask,*,scalar=False):
    if source.dtype!=np.uint8 or source.ndim!=3 or source.shape[2]!=3 or mask.shape!=source.shape[:2]:
        raise ValueError('Use byte RGB normals and a matching explicit repair mask')
    selected=mask>0
    if not selected.any():raise ValueError('The repair region is empty')
    yy,xx=np.where(selected)
    y0,y1=max(0,yy.min()-2),min(len(mask),yy.max()+3)
    x0,x1=max(0,xx.min()-2),min(mask.shape[1],xx.max()+3)
    region=selected[y0:y1,x0:x1].copy()
    if region[0].any() or region[-1].any() or region[:,0].any() or region[:,-1].any():
        raise ValueError('Keep an intact boundary around the repair')
    # Guard against accidental whole-atlas linear systems.
    count=int(region.sum())
    if count>1_000_000:raise ValueError('Split large defects into bounded repair regions')
    values=source[y0:y1,x0:x1].astype(np.float64)/255
    if not scalar:
        values=values*2-1
        values/=np.maximum(np.linalg.norm(values,axis=2,keepdims=True),1e-8)
    ids=np.full(region.shape,-1,dtype=np.int32);ids[region]=np.arange(count)
    y,x=np.where(region);rows=[np.arange(count)];cols=[np.arange(count)];weights=[np.full(count,4.)]
    rhs=np.zeros((count,3),dtype=np.float64)
    for dy,dx in ((-1,0),(1,0),(0,-1),(0,1)):
        neighbours=ids[y+dy,x+dx];inside=neighbours>=0
        rows.append(np.flatnonzero(inside));cols.append(neighbours[inside]);weights.append(np.full(int(inside.sum()),-1.))
        rhs[~inside]+=values[y[~inside]+dy,x[~inside]+dx]
    matrix=sparse.coo_matrix((np.concatenate(weights),(np.concatenate(rows),np.concatenate(cols))),shape=(count,count)).tocsc()
    filled=spsolve(matrix,rhs)
    if not scalar:
        filled/=np.maximum(np.linalg.norm(filled,axis=1,keepdims=True),1e-8)
        filled=filled*.5+.5
    if not np.isfinite(filled).all():raise ValueError('The repair has no stable normal continuation')
    result=source.copy();patch=result[y0:y1,x0:x1]
    patch[region]=np.rint(np.clip(filled,0,1)*255).astype(np.uint8)
    assert np.array_equal(result[~selected],source[~selected])
    return result,dict(selected_texels=count,reference='harmonic_scalar_boundary' if scalar else 'harmonic_normal_boundary',outside_mask_bytes_exact=True)
