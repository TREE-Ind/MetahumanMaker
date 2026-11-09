"""Retune already generated texture detail and remove evidenced periodic modes.

This reuses the accepted generated residual rather than inventing procedural
pores or rerunning a model. Broad pigmentation stays anchored to the accepted
image. The existing masked spectral solver removes coherent repeating modes.
"""
import numpy as np
from scipy import ndimage
try:
    from .texture_restore_core import resize
    from .skin_reconstruction_core import reconstruct
except ImportError:
    from texture_restore_core import resize
    from skin_reconstruction_core import reconstruct


def refine(accepted, source, mask, *, gain=2.75, maximum_added=.04, radius=8., progress=None):
    accepted=np.asarray(accepted,dtype=np.float32);source=np.asarray(source,dtype=np.float32);mask=np.asarray(mask,dtype=np.float32)
    if accepted.ndim!=3 or accepted.shape[2]!=3 or source.ndim!=3 or source.shape[2]!=3 or mask.shape!=accepted.shape[:2]:
        raise ValueError('Supply linear RGB skin images and a matching explicit edit mask')
    if not 1<=gain<=4 or not 0<maximum_added<=.1 or not .5<=radius<=32:
        raise ValueError('Detail gain, additional contrast or radius is out of bounds')
    if any(not np.isfinite(value).all() for value in (accepted,source,mask)) or mask.min()<0 or mask.max()>1:
        raise ValueError('Skin inputs and masks must be finite and masks bounded')
    h,w=accepted.shape[:2]
    if not 64<=min(h,w) or max(h,w)>8192:raise ValueError('Skin refinement supports images through 8K')
    scale=max(h/source.shape[0],w/source.shape[1])
    if not 1<=scale<=4:raise ValueError('Retained source must be at the accepted resolution or up to four times smaller')
    baseline=source if source.shape==accepted.shape else resize(source,(w,h))
    residual=accepted-baseline
    del baseline
    residual-=ndimage.gaussian_filter(residual,(radius,radius,0))
    residual*=gain-1
    np.clip(residual,-maximum_added,maximum_added,out=residual)
    residual*=mask[...,None]
    proposed=np.clip(accepted+residual,0,1)
    del residual
    period=min(32.*scale,128.)
    tile=int(256*min(scale,2));overlap=tile//4
    clean,_,evidence,report=reconstruct(proposed,proposed,mask,strength=1.,method='PERIODIC',progress=progress,
        tile=tile,overlap=overlap,periodic_maximum_period=period,periodic_compensate_filter=True)
    clean[mask<=0]=accepted[mask<=0]
    report.update(detail_method='RETAINED_GENERATED_RESIDUAL',detail_gain=gain,
                  maximum_added_linear=maximum_added,detail_radius_texels=radius,
                  new_model_inference=False,geometry_modified=False,
                  periodic_maximum_period_texels=period,periodic_filter_response_compensated=True,
                  outside_mask_max_delta=float(np.abs(clean[mask<=0]-accepted[mask<=0]).max(initial=0)))
    return clean,evidence,report
