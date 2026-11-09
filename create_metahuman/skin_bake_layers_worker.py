"""Replay retained skin operations on a fresh geometry normal, without Blender."""
import json,sys,time,traceback
from pathlib import Path
import numpy as np
from PIL import Image
from skin_bake_layers import resolve,digest,validate_material_responses
from skin_reconstruction_core import apply_relief,repair_normals,inferred_relief
from worker_lifetime import watch_parent
from skin_material_core import height_cavity_response,retain_packed_channels
from skin_normal_patch import apply as apply_patch

def run(request):
    started=time.monotonic();root=Path(request['manifest']).resolve().parent
    if digest(request['manifest'])!=request['manifest_sha256']:raise ValueError('Retained skin layer manifest changed')
    layers=json.loads(Path(request['manifest']).read_text(encoding='utf-8'))
    validate_material_responses(layers['steps'])
    if digest(request['normal'])!=request['normal_sha256']:raise ValueError('Geometry normal changed before skin replay')
    normal=np.asarray(Image.open(request['normal']).convert('RGB'),dtype=np.float32)/255
    cavity=None;packed=None;cavity_operations=[]
    if request.get('cavity'):
        if digest(request['cavity'])!=request['cavity_sha256']:raise ValueError('Geometry cavity changed before skin replay')
        with Image.open(request['cavity']) as image:
            if image.mode in {'RGB','RGBA'}:
                packed=np.asarray(image);cavity=packed[...,0].astype(np.float32)/255
            else:cavity=np.asarray(image.convert('L'),dtype=np.float32)/255
    def load(record,size=None):
        path=resolve(root,record)
        with Image.open(path) as image:
            image=image.convert('RGB')
            if size and image.size!=size:image=image.resize(size,Image.Resampling.BILINEAR)
            return np.asarray(image,dtype=np.float32)/255
    def patch_pixels(record,size,*,plane=False):
        with Image.open(resolve(root,record)) as image:
            image=image.convert('RGB')
            if plane:image=image.getchannel('R')
            if image.size!=size:image=image.resize(size,Image.Resampling.BILINEAR)
            return np.asarray(image,dtype=np.uint8)
    progress_path=Path(request['progress'])
    def progress(done,total,phase='Replaying retained skin'):
        progress_path.write_text(json.dumps({'phase':phase,'done':done,'total':total}),encoding='utf-8')
    operations=[]
    for index,step in enumerate(layers['steps']):
        progress(index,len(layers['steps']),step['kind'])
        if step['kind']=='REPAIR':
            mask=load(step['mask'])[...,0];reference=load(step['reference'])
            normal=repair_normals(normal,reference,mask*float(step['strength']));del mask,reference
        elif step['kind']=='SURFACE_PATCH':
            size=(normal.shape[1],normal.shape[0])
            before=patch_pixels(step['before'],size);after=patch_pixels(step['after'],size)
            apply_patch(normal,before,after,progress=progress);del before,after
            if cavity is not None and step.get('cavity_response'):
                size=(cavity.shape[1],cavity.shape[0])
                before=patch_pixels(step['cavity_before'],size,plane=True);after=patch_pixels(step['cavity_after'],size,plane=True)
                for y in range(0,len(cavity),128):
                    cavity[y:y+128]+=(after[y:y+128].astype(np.float32)-before[y:y+128].astype(np.float32))/255
                np.clip(cavity,0,1,out=cavity);del before,after
                cavity_operations.append(index)
        elif step['kind']=='HEIGHT':
            height=np.load(resolve(root,step['field']),allow_pickle=False)
            if height.ndim!=2 or not np.isfinite(height).all() or abs(height).max()>16:raise ValueError('Invalid retained skin height')
            normal=apply_relief(normal,height,directx=True)
            if cavity is not None and step.get('cavity_response'):
                response=step['cavity_response']
                if response.get('method')!='SHARED_HEIGHT':raise ValueError('Unknown retained cavity response')
                field=height_cavity_response(height,float(response['strength']))
                if field.shape!=cavity.shape:field=np.asarray(Image.fromarray(field).resize((cavity.shape[1],cavity.shape[0]),Image.Resampling.BILINEAR),dtype=np.float32)
                cavity+=field;np.clip(cavity,0,1,out=cavity);del field
                cavity_operations.append(index)
            del height
        elif step['kind']=='RELIEF':
            color=load(step['color']);mask=load(step['mask'],(color.shape[1],color.shape[0]))[...,0]
            height,_gate,_detail=inferred_relief(color,mask,np.zeros_like(mask),radius=float(step['radius']),strength=float(step['strength']))
            del color,mask,_gate
            normal=apply_relief(normal,height,directx=True);del height
        elif step['kind']!='RETAIN':raise ValueError('Unknown retained skin operation')
        if step.get('quantize_after'):
            np.clip(normal,0,1,out=normal);normal*=255;np.rint(normal,out=normal);normal/=255
            if cavity is not None:cavity*=255;np.rint(cavity,out=cavity);cavity/=255
        operations.append(step['kind'])
    pixels=np.rint(np.clip(normal,0,1)*255).astype(np.uint8)
    Image.fromarray(pixels).save(request['output'])
    progress(len(operations),len(operations),'Complete')
    result={'status':'complete','sha256':digest(request['output']),'size':[normal.shape[1],normal.shape[0]],'operations':operations,'seconds':time.monotonic()-started}
    if cavity is not None:
        pixels=retain_packed_channels(cavity,packed) if packed is not None else np.rint(np.clip(cavity,0,1)*255).astype(np.uint8)
        Image.fromarray(pixels).save(request['cavity_output'])
        result.update(cavity_sha256=digest(request['cavity_output']),cavity_operations=cavity_operations,cavity_size=[cavity.shape[1],cavity.shape[0]])
    return result

if __name__=='__main__':
    request=json.loads(Path(sys.argv[1]).read_text(encoding='utf-8'));watch_parent(request['parent'])
    try:result=run(request)
    except Exception as exc:result={'status':'failed','error':str(exc),'traceback':traceback.format_exc()}
    Path(request['result']).write_text(json.dumps(result,indent=2),encoding='utf-8')
    if result['status']!='complete':raise SystemExit(1)
