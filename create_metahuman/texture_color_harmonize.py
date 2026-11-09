"""Low-frequency color matching for an explicitly selected generated atlas.

Preserves generated grain; neither changes UVs nor infers physical reflectance.
The caller supplies a palette selection when source complexion should be kept.
"""
import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter, distance_transform_edt


def match(reference, accepted, selection, *, palette=None, edge_radius=.025):
    if reference.mode!='RGB' or accepted.mode!='RGB' or not 0<edge_radius<=.1:
        raise ValueError('Use RGB atlases and a bounded color-transition radius')
    size=reference.size;coarse=(512,512)
    src=np.asarray(accepted.resize(coarse,Image.Resampling.LANCZOS),np.float32)/255
    gen=np.asarray(reference.resize(coarse,Image.Resampling.LANCZOS),np.float32)/255
    gate=np.asarray(selection.convert('L').resize(coarse,Image.Resampling.BILINEAR),np.float32)/255
    gain=np.ones(3,np.float32)
    if palette is not None:
        chosen=(np.asarray(palette.convert('L').resize(coarse),np.float32)>127)&(gate>.95)
        if chosen.sum()<32:raise ValueError('Choose enough skin for palette matching')
        gain=np.clip(np.median(src[chosen],axis=0)/np.maximum(np.median(gen[chosen],axis=0),.01),.5,2)
    gen=np.clip(gen*gain,0,1)
    src_low=gaussian_filter(src,(8,8,0));gen_low=gaussian_filter(gen,(8,8,0))
    correction=np.clip(np.log(np.maximum(src_low,.01)/np.maximum(gen_low,.01)),-.4,.4)
    distance=distance_transform_edt(gate>.5)
    edge=np.exp(-distance/max(edge_radius*512,1))* (gate>0)
    correction*=edge[...,None]
    field=np.stack([np.asarray(Image.fromarray(correction[...,c]).resize(size,Image.Resampling.BILINEAR)) for c in range(3)],axis=2)
    full_gate=np.asarray(selection.convert('L').resize(size,Image.Resampling.BILINEAR),np.float32)/255
    original=np.asarray(reference);result=original.copy()
    for y in range(0,size[1],128):
        b=slice(y,y+128);rgb=original[b].astype(np.float32)/255
        change=(np.log(gain)[None,None,:]+field[b])*full_gate[b,...,None]
        result[b]=np.rint(np.clip(rgb*np.exp(change),0,1)*255).astype(np.uint8)
    return Image.fromarray(result),dict(method='MASKED_PALETTE_AND_BOUNDARY_MATCH',
        palette_gain=gain.tolist(),edge_radius=edge_radius,coarse_resolution=list(coarse),
        generated_high_frequency_retained=True,uv_layout_changed=False,
        outside_selection_unchanged=bool(np.array_equal(result[full_gate<=0],original[full_gate<=0])))
