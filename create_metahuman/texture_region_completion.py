"""Explicit local color completion with protected pixels and bounded memory.

This is inferred paint, never recovered source information. Callers supply a
semantic repair region: color alone must not decide which artist marks to erase.
The inpainting runs on a cropped region, leaving every other texel unchanged.
"""
import numpy as np


def guarded_region(mask,valid,*,radius,seam_edges=None):
    """Protect chart padding and paired UV edges, using a crop-sized distance field."""
    from PIL import Image,ImageDraw
    from scipy.ndimage import distance_transform_edt
    mask=np.asarray(mask,dtype=np.float32);valid=np.asarray(valid)
    if mask.shape!=valid.shape or mask.ndim!=2 or not np.isfinite(mask).all() or np.any((mask<0)|(mask>1)) or not np.isfinite(radius) or not 1<=radius<=64:
        raise ValueError('Atlas repair needs matching bounded masks and a valid radius')
    eligible=mask.copy();y,x=np.nonzero(mask>0)
    if not len(x):return eligible,dict(skipped_uv_border_texels=0)
    height,width=mask.shape;halo=int(np.ceil(radius*3))+3
    x0,x1=max(0,int(x.min())-halo),min(width,int(x.max())+halo+1);y0,y1=max(0,int(y.min())-halo),min(height,int(y.max())+halo+1)
    domain=Image.fromarray(np.uint8(valid[y0:y1,x0:x1]>0)*255)
    if seam_edges is not None:
        seams=np.asarray(seam_edges)
        if seams.ndim!=4 or seams.shape[1:]!=(2,2,2) or not np.isfinite(seams).all():raise ValueError('Paired UV seam edges are invalid')
        draw=ImageDraw.Draw(domain)
        for pair in seams:
            for edge in pair:
                draw.line([(float(u)*(width-1)-x0,(1-float(v))*(height-1)-y0) for u,v in edge],fill=0,width=3)
    distance=distance_transform_edt(np.pad(np.asarray(domain)>0,1))[1:-1,1:-1]
    eligible[y0:y1,x0:x1]*=np.clip((distance-radius*2-1)/radius,0,1)
    return eligible,dict(skipped_uv_border_texels=int(np.count_nonzero(eligible<mask)),guard_crop=[x0,y0,x1,y1],uv_border_protected=True)


def complete_color(source, region, *, protect=None, unknown=None, radius=5., strength=1.,method='TELEA',geometry=None):
    """Complete an explicitly approved uint8 RGB(A) repair mask using Telea.

    region/protect are HxW weights in [0,1]. Protected pixels remain exact but
    requested repairs there cannot donate contaminated color to nearby skin.
    The method does not synthesize skin microgeometry.
    """
    source=np.asarray(source);region=np.asarray(region,dtype=np.float32)
    if source.dtype!=np.uint8 or source.ndim!=3 or source.shape[2] not in (3,4) or region.shape!=source.shape[:2]:
        raise ValueError('Color completion requires uint8 RGB(A) and a matching repair mask')
    if not np.isfinite(region).all() or np.any((region<0)|(region>1)) or not np.isfinite(radius) or not 1<=radius<=64 or not np.isfinite(strength) or not 0<=strength<=1:
        raise ValueError('Completion weights and radius must be bounded and finite')
    if method=='SURFACE':
        if __package__:from .surface_color_completion import complete
        else:from surface_color_completion import complete
        return complete(source,region,geometry,unknown=unknown,protect=protect,strength=strength)
    if method!='TELEA':raise ValueError('Choose image-neighborhood or physical-surface color completion')
    active=region>0
    if unknown is not None:
        unknown=np.asarray(unknown)
        if unknown.shape!=region.shape or not np.isfinite(unknown).all() or np.any((unknown<0)|(unknown>1)):
            raise ValueError('The unknown-color mask must match the repair region')
        # A locked output pixel can still contain rejected source pigment.
        # Keep it out of the donor set even though its final bytes stay exact.
        active=active|(unknown>0)
    if protect is not None:
        protect=np.asarray(protect,dtype=np.float32)
        if protect.shape!=region.shape or not np.isfinite(protect).all() or np.any((protect<0)|(protect>1)):
            raise ValueError('The protection mask must match the repair region')
        region=region*(1-protect)
    if not np.any(region>0) or strength==0:return source.copy(),dict(method='TELEA_LOCAL_COLOR',changed_texels=0,inferred=True,inference_skipped=True)
    y,x=np.nonzero(active);pad=int(np.ceil(radius*3))+2
    y0,y1=max(0,int(y.min())-pad),min(len(source),int(y.max())+pad+1)
    x0,x1=max(0,int(x.min())-pad),min(source.shape[1],int(x.max())+pad+1)
    mask=active[y0:y1,x0:x1].astype(np.uint8)*255
    if np.any(mask[0]) or np.any(mask[-1]) or np.any(mask[:,0]) or np.any(mask[:,-1]):
        raise ValueError('The repair needs a trusted border; split UV charts explicitly')
    try:import cv2
    except ImportError as exc:raise ValueError('Local color completion needs OpenCV in the configured inference Python environment') from exc
    original=np.ascontiguousarray(source[y0:y1,x0:x1,:3]);proposal=cv2.inpaint(original,mask,radius,cv2.INPAINT_TELEA)
    weight=region[y0:y1,x0:x1,None]*strength
    result=source.copy();result[y0:y1,x0:x1,:3]=np.rint(original*(1-weight)+proposal*weight).clip(0,255).astype(np.uint8)
    return result,dict(method='TELEA_LOCAL_COLOR',changed_texels=int(np.any(result!=source,axis=2).sum()),inferred=True,
                       unknown_color_texels=int(active.sum()),locked_unknown_texels=int(np.count_nonzero(active&(region<=0))),
                       crop=[x0,y0,x1,y1],outside_mask_unchanged=True,alpha_unchanged=True)
