"""Bound a full-atlas learned SR proposal by source color and artist selection."""
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from PIL import Image
from texture_restore_core import restore,linear,resize
import texture_sr_prior


def residual_tiles(submitted,proposal_path,target,invention,progress=None):
    """Overlap source-constrained crops; keep correction buffers tile-sized."""
    height,width=submitted.shape[:2];scale=target[0]//width
    tile=min(256,height,width);stride=max(1,tile*3//4)
    def starts(length):return sorted(set(range(0,max(1,length-tile+1),stride))|{length-tile})
    positions=[(y,x) for y in starts(height) for x in starts(width)]
    sums=np.zeros((target[1],target[0],3),np.float32)
    weights=np.zeros((target[1],target[0]),np.float32)
    axis=np.maximum(np.hanning(tile*scale),.01).astype(np.float32)
    window=axis[:,None]*axis[None,:]
    max_error=0.
    with Image.open(proposal_path) as image:
        if image.size!=target:raise ValueError('The diffusion prior returned a different atlas size')
        for index,(y,x) in enumerate(positions):
            if progress:progress(index,len(positions),'Constraining source detail')
            source=submitted[y:y+tile,x:x+tile]
            box=(x*scale,y*scale,(x+tile)*scale,(y+tile)*scale)
            proposal=np.asarray(image.crop(box).convert('RGB'),dtype=np.float32)/255
            restored,_,report=restore(source,proposal,strength=1,detail_limit=.25*invention)
            residual=linear(restored)-np.clip(resize(linear(source),(tile*scale,tile*scale)),0,1)
            region=np.s_[y*scale:(y+tile)*scale,x*scale:(x+tile)*scale]
            sums[region]+=residual*window[...,None];weights[region]+=window
            max_error=max(max_error,report['linear_source_consistency_max_error'])
    if progress:progress(len(positions),len(positions),'Constraining source detail')
    return sums,weights,dict(method='overlapping_source_tiles',tiles=len(positions),
        source_tile_size=tile,source_stride=stride,maximum_tile_consistency_error=max_error,
        full_atlas_consistency_verified=False)


def candidate(source,mask,valid,*,backend,expected,folder,strength=1,invention=.35,
              protect=None,brief='',progress=None,output_scale=1):
    if output_scale not in (1,2,4) or not 0<=invention<=1 or not 0<=strength<=1:
        raise ValueError('Invalid SR output scale or invention strength')
    h,w=source.shape[:2];target=(w*output_scale,h*output_scale)
    if h!=w or max(target)>8192 or min(source.shape[:2])<64:
        raise ValueError('PiSA-SR skin restoration needs a square atlas from 64 pixels through 8K')
    allow=np.clip(mask,0,1)*np.clip(valid,0,1)
    if protect is not None:allow*=1-np.clip(protect,0,1)
    if invention==0 or strength==0 or not np.any(allow>0):
        original=np.clip(resize(linear(source),target),0,1) if output_scale!=1 else linear(source)
        return original,np.zeros(original.shape[:2],np.float32),dict(method='LEARNED',prior='PISA',
            invention=invention,inference_skipped=True,physical_albedo_recovered=False,generated_detail_is_ground_truth=False)
    # The model sees at most a 2K context atlas. Its residual adds detail over the
    # full-resolution accepted source, rather than replacing that source's marks.
    context_size=min(w,max(32,target[0]//2),2048)
    if target[0]//context_size not in (2,4) or target[0]%context_size:
        raise ValueError('PiSA-SR context needs a 2x or 4x output ratio')
    context=np.clip(resize(linear(source),(context_size,context_size)),0,1)
    from texture_restore_core import encoded
    srgb_context=np.clip(encoded(context),0,1)
    prior=Path(folder)/'sr-prior';prior.mkdir(exist_ok=False)
    if progress:progress(0,1,'Loading local diffusion runtime')
    Image.fromarray(np.rint(srgb_context*255).astype('uint8')).save(prior/'input.png')
    report=texture_sr_prior.run(SimpleNamespace(model=str(backend),expected=expected,input=str(prior/'input.png'),output=str(prior),
        scale=target[0]//context_size,pixel=1.,semantic=invention,seed=42,tile=64,prompt=brief,progress=progress))
    # Quantized model input is the exact consistency target.
    with Image.open(prior/'input.png') as image:submitted=np.asarray(image.convert('RGB'),dtype=np.float32)/255
    result,weights,constraints=residual_tiles(submitted,prior/'proposal.png',target,invention,progress)
    original=np.clip(resize(linear(source),target),0,1) if output_scale!=1 else linear(source)
    gate=np.array(Image.fromarray(allow).resize(target,Image.Resampling.BILINEAR),dtype=np.float32)
    # Bilinear growth must not create edits outside a hard artist exclusion.
    hard=np.asarray(Image.fromarray(allow).resize(target,Image.Resampling.NEAREST),dtype=np.float32)
    gate[hard<=0]=0
    delta=np.zeros(original.shape[:2],np.float32);outside_delta=0.
    for y in range(0,target[1],128):
        band=np.s_[y:y+128];values=result[band];weight=weights[band]
        np.divide(values,weight[...,None],out=values,where=weight[...,None]>0)
        values*=strength*invention*gate[band,...,None];values+=original[band]
        np.clip(values,0,1,out=values)
        excluded=gate[band]<=0;values[excluded]=original[band][excluded]
        delta[band]=np.linalg.norm(values-original[band],axis=2)
        if np.any(excluded):outside_delta=max(outside_delta,float(np.abs(values[excluded]-original[band][excluded]).max()))
    del weights,hard,original
    texture_sr_prior.verify(Path(backend).resolve())
    return result,delta,dict(method='LEARNED',prior='PISA',invention=invention,output_scale=output_scale,
        context_resolution=[context_size]*2,output_resolution=list(target),backend=expected,
        backend_verification='worker_sha256_before_and_after',model=report,constraints=constraints,
        source_detail_additive=True,physical_albedo_recovered=False,generated_detail_is_ground_truth=False,
        outside_selected_skin_max_delta=outside_delta)
