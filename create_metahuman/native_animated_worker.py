"""Encode full expression maps for UE's MF_skin_animated delta inputs.

Normals decode as base + (delta * 4 - 2); colors decode in sRGB as
base + (delta - .5) * .5. Source artist maps remain unchanged.
"""
import json,sys,hashlib,traceback
from pathlib import Path
import numpy as np
from PIL import Image

def digest(path):
    with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()

def run(request):
    output=Path(request['output']).resolve();output.mkdir(exist_ok=False)
    results={}
    for item in request['items']:
        source=Path(item['source']);kind=item['kind'];mode=item.get('encoding','FULL')
        if kind not in {'CM1','CM2','CM3','WM1','WM2','WM3'} or mode not in {'FULL','DELTA'}:raise ValueError('Unsupported animated map encoding')
        if digest(source)!=item['sha256']:raise ValueError('Animated source changed')
        with Image.open(source) as image:
            if not 1<=image.width<=8192 or not 1<=image.height<=8192:raise ValueError('Animated maps must be through 8K')
            pixels=np.asarray(image.convert('RGB'));size=image.size
        target=output/source.name
        if mode=='DELTA':
            import shutil
            shutil.copy2(source,target);maximum=0.
        else:
            base=Path(item['base'])
            if digest(base)!=item['base_sha256']:raise ValueError('Animated neutral base changed')
            with Image.open(base) as image:
                image=image.convert('RGB')
                if image.size!=size:image=image.resize(size,Image.Resampling.BILINEAR)
                neutral=np.asarray(image)
            encoded=np.empty_like(pixels);maximum=0.
            for first in range(0,len(pixels),128):
                a=pixels[first:first+128].astype(np.float32)/255;b=neutral[first:first+128].astype(np.float32)/255
                if kind.startswith('WM'):
                    a=a*2-1;b=b*2-1
                    a/=np.maximum(np.linalg.norm(a,axis=2,keepdims=True),1e-8);b/=np.maximum(np.linalg.norm(b,axis=2,keepdims=True),1e-8)
                    values=(a-b)*.25+.5
                else:values=(a-b)*2+.5
                if not np.isfinite(values).all() or values.min()<-.00001 or values.max()>1.00001:raise ValueError('Expression color exceeds the native delta range; retain the source and reduce its difference from neutral')
                encoded[first:first+128]=np.rint(np.clip(values,0,1)*255).astype(np.uint8)
                maximum=max(maximum,float(abs(values-.5).max()))
            Image.fromarray(encoded).save(target)
            if digest(base)!=item['base_sha256']:raise ValueError('Animated neutral base changed while encoding')
        if digest(source)!=item['sha256']:raise ValueError('Animated source changed while encoding')
        results[kind]={'file':str(target),'sha256':digest(target),'source_sha256':item['sha256'],'base_sha256':item.get('base_sha256'),'input_encoding':mode,'encoding':'UE_ANIMATED_DELTA_V1','size':list(size),'maximum_encoded_offset':maximum}
    return {'status':'complete','maps':results}

if __name__=='__main__':
    request=json.loads(Path(sys.argv[1]).read_bytes())
    if request.get('parent'):
        from worker_lifetime import watch_parent
        watch_parent(request['parent'])
    try:result=run(request)
    except Exception as exc:result={'status':'failed','error':str(exc),'traceback':traceback.format_exc()}
    Path(request['result']).write_text(json.dumps(result,indent=2),encoding='utf-8')
    if result['status']!='complete':raise SystemExit(1)
