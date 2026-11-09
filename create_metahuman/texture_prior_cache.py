"""Bounded, verified pretrained proposals; never stores accepted character maps."""
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import uuid

import numpy as np

LIMIT=128*1024**2


def identity(source,manifest,prompt,seed,runtime_sha,initialization=None):
    pixels=np.rint(np.clip(source,0,1)*255).astype('uint8')
    inputs={'schema':1,'reference_rgb_sha256':hashlib.sha256(pixels.tobytes()).hexdigest(),
        'shape':list(pixels.shape),'prompt':prompt,'seed':seed,'steps':4,'cfg':1,'sampler':'euler',
        'output_scale':2,'vae_tiling':True,'runtime_sha256':runtime_sha,
        'weights':{x['name']:x['sha256'] for x in manifest['assets'] if x['name']!='runtime.zip'}}
    if initialization is not None:inputs['initialization']=initialization
    key=hashlib.sha256(json.dumps(inputs,sort_keys=True,separators=(',',':')).encode()).hexdigest()
    return key,inputs


def root(path):
    path=Path(path).absolute()
    if path.is_symlink() or path.resolve()!=path:
        raise ValueError('The texture prior cache must stay in its owned project directory')
    path.mkdir(parents=True,exist_ok=True)
    return path


def records(path):
    path=Path(path)
    if not path.exists():return []
    entries=[]
    for metadata in path.glob('*.json'):
        if not re.fullmatch('[a-f0-9]{64}',metadata.stem) or metadata.is_symlink():continue
        try:
            info=json.loads(metadata.read_text(encoding='utf-8'))
            image=metadata.with_suffix('.png')
            if info.get('kind')!='CM_TEXTURE_PRIOR_CACHE' or not image.is_file() or image.is_symlink():continue
            entries.append((metadata.stat().st_mtime,metadata,image,info))
        except (OSError,ValueError):continue
    return entries


def prune(path,limit=LIMIT):
    entries=records(path)
    total=sum(m.stat().st_size+p.stat().st_size for _,m,p,_ in entries)
    for _,metadata,image,_ in sorted(entries):
        if total<=limit:break
        size=metadata.stat().st_size+image.stat().st_size
        metadata.unlink();image.unlink();total-=size
    return total


def fetch(path,key,destination):
    path=root(path);metadata=path/(key+'.json');image=path/(key+'.png')
    try:
        info=json.loads(metadata.read_text(encoding='utf-8'))
        if info.get('kind')!='CM_TEXTURE_PRIOR_CACHE' or info.get('key')!=key:return None
        if image.is_symlink() or hashlib.sha256(image.read_bytes()).hexdigest()!=info['report']['output_sha256']:return None
        shutil.copy2(image,destination)
        os.utime(metadata,None)
        return dict(info['report'],cache_hit=True,cache_key=key)
    except (OSError,ValueError,KeyError):return None


def publish(path,key,inputs,image,report):
    path=root(path);token=uuid.uuid4().hex
    pending_image=path/(token+'.tmp');pending_metadata=path/(token+'.meta-tmp')
    try:
        shutil.copy2(image,pending_image)
        if hashlib.sha256(pending_image.read_bytes()).hexdigest()!=report['output_sha256']:
            raise ValueError('The pretrained proposal changed before caching')
        pending_metadata.write_text(json.dumps({'kind':'CM_TEXTURE_PRIOR_CACHE','key':key,
            'inputs':inputs,'report':report},indent=2),encoding='utf-8')
        pending_image.replace(path/(key+'.png'))
        pending_metadata.replace(path/(key+'.json'))
    finally:
        pending_image.unlink(missing_ok=True);pending_metadata.unlink(missing_ok=True)
    prune(path)


def clear(path):
    path=root(path);removed=0
    for _,metadata,image,_ in records(path):
        if metadata.resolve().parent!=path or image.resolve().parent!=path:
            raise ValueError('A cache entry left its owned directory')
        removed+=metadata.stat().st_size+image.stat().st_size
        metadata.unlink();image.unlink()
    return removed
