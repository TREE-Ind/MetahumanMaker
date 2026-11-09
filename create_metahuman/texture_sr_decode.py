"""Decode VAE tiles with the model's own blending and two CPU row buffers."""
from pathlib import Path
import numpy as np
from PIL import Image


def decode(vae,latent,path,size,progress=None):
    import torch
    path=Path(path);scratch=path.with_name('decode-rgb.bin')
    if scratch.exists() or scratch.is_symlink():raise ValueError('The owned decoder scratch already exists')
    tile=int(vae.tile_latent_min_size);sample=int(vae.tile_sample_min_size)
    step=int(tile*(1-vae.tile_overlap_factor));blend=int(sample*vae.tile_overlap_factor)
    limit=sample-blend;scale=sample//tile
    if latent.shape[0]!=1 or step<=0 or sample%tile or size!=(int(latent.shape[3])*scale,int(latent.shape[2])*scale):
        raise ValueError('The installed VAE needs a compatible single-atlas tile layout')
    rows=list(range(0,int(latent.shape[2]),step));columns=list(range(0,int(latent.shape[3]),step))
    canvas=np.memmap(scratch,mode='w+',dtype='uint8',shape=(size[1],size[0],3))
    previous=[];count=0
    try:
        with torch.inference_mode():
            for row,y in enumerate(rows):
                current=[]
                for column,x in enumerate(columns):
                    values=latent[:,:,y:y+tile,x:x+tile]
                    if vae.config.use_post_quant_conv:values=vae.post_quant_conv(values)
                    # Native tiled_decode retains all decoded GPU tiles. Only
                    # the above row and current row are required for its blend.
                    values=vae.decoder(values).cpu()
                    if row:values=vae.blend_v(previous[column],values,blend)
                    if column:values=vae.blend_h(current[column-1],values,blend)
                    if not torch.isfinite(values).all():raise ValueError('Non-finite SR decode tile')
                    current.append(values)
                    height=min(limit,int(values.shape[2]),size[1]-y*scale)
                    width=min(limit,int(values.shape[3]),size[0]-x*scale)
                    pixels=values[0,:,:height,:width].clamp(-1,1).add(1).mul(127.5).round().byte().permute(1,2,0).numpy()
                    canvas[y*scale:y*scale+height,x*scale:x*scale+width]=pixels
                    count+=1
                    if progress and (count%4==0 or count==len(rows)*len(columns)):
                        progress(count,len(rows)*len(columns),'PiSA-SR: decoding skin tiles')
                previous=current
        canvas.flush()
        with Image.fromarray(canvas) as image:image.save(path)
    finally:
        canvas._mmap.close()
        scratch.unlink(missing_ok=True)
    return dict(method='row_streaming_native_vae_blend',tiles=count,decoded_float_rows_retained=2,
                scratch_bytes=size[0]*size[1]*3,cpu_blending=True)
