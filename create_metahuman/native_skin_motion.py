"""Pack sampled native shader differences without reducing neutral-map detail."""
import json,math,hashlib
from pathlib import Path
import numpy as np
from PIL import Image

def digest(path):
    with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()

def prepare(root,stage,source_sha):
    root=Path(root).resolve();stage=Path(stage);path=root/'skin-motion/manifest.json'
    if not path.exists():return None
    raw=path.read_bytes();data=json.loads(raw)
    if data.get('status')!='complete' or data.get('generator')!='CreateMetaHuman.NativeSkinMotion' or data['source_manifest_sha256']!=source_sha:raise ValueError('Animated native skin belongs to another or incomplete capture')
    poses=data['poses'];size=data['size']
    if not poses or poses[0]['pose']!='Neutral' or size not in (256,512,1024) or len({p['pose'] for p in poses})!=len(poses):raise ValueError('Invalid native skin sample layout')
    protected={}
    def read(record,role):
        item=record['maps'][role];p=(root/item['file']).resolve()
        if not p.is_relative_to(root) or digest(p)!=item['sha256']:raise ValueError('Native skin response changed')
        protected[p]=item['sha256']
        with Image.open(p) as image:
            if image.size!=(size,size):raise ValueError('Inconsistent native skin response dimensions')
            a=np.asarray(image.convert('RGB'),dtype=np.float32)/255
        if role=='basecolor':return a if item.get('linear_color',False) else np.where(a<=.04045,a/12.92,((a+.055)/1.055)**2.4)
        if role=='normal':
            a=a*2-1;return a/np.maximum(np.linalg.norm(a,axis=2,keepdims=True),1e-8)
        return a[...,:1]
    neutral={role:read(poses[0],role) for role in ('basecolor','normal','roughness')}
    if not 1<=len(poses)-1<=128:raise ValueError('Animated skin supports up to 128 non-neutral native samples')
    unique=[];indices=[];seen={}
    for pose in poses[1:]:
        signature=tuple(pose['maps'][role]['sha256'] for role in neutral)
        if signature not in seen:seen[signature]=len(unique);unique.append(pose)
        indices.append(seen[signature])
    count=len(unique)
    columns=min(8,count);rows=math.ceil(count/columns)
    from_shape=(rows*size,columns*size)
    budget=__import__('runpy').run_path(str(Path(__file__).with_name('resource_budget.py')))
    budget['require']('animated skin packing',from_shape[0]*from_shape[1]*10+size*size*96+128*1024**2)
    budget['require_disk']('animated skin packing',stage,from_shape[0]*from_shape[1]*10+64*1024**2)
    atlases={role:np.full((*from_shape,3),128,dtype=np.uint8) for role in neutral};samples=[]
    for index,pose in enumerate(unique):
        deltas={role:read(pose,role)-base for role,base in neutral.items()}
        color_scale=max(float(abs(deltas['basecolor']).max()),1e-6)
        normal_scale=max(float(abs(deltas['normal']).max()),float(abs(deltas['roughness']).max()),1e-6)
        top,left=divmod(index,columns);region=np.s_[top*size:(top+1)*size,left*size:(left+1)*size]
        for role,delta in deltas.items():
            scale=color_scale if role=='basecolor' else normal_scale
            encoded=np.rint(128+127*delta/scale).clip(1,255).astype(np.uint8)
            atlases[role][region]=encoded
        samples.append({'layer':index,'color_scale':color_scale,'normal_scale':normal_scale})
    samples=[dict(samples[index],pose=pose['pose']) for pose,index in zip(poses[1:],indices)]
    resources={}
    for role,pixels in atlases.items():
        # Opaque RGB avoids canvas alpha-premultiplication destroying numeric data.
        temporary=stage/(role+'-motion.png');Image.fromarray(pixels).save(temporary)
        sha=digest(temporary);name='textures/'+sha+'.png';target=stage/name;target.parent.mkdir(exist_ok=True);temporary.replace(target)
        resources[role]={'file':name,'sha256':sha,'bytes':target.stat().st_size,'size':[from_shape[1],from_shape[0]]}
    if path.read_bytes()!=raw or any(digest(p)!=sha for p,sha in protected.items()):raise ValueError('Native skin response changed while packing')
    return {'schema':1,'generator':'CreateMetaHuman.PortableSkinMotion','method':'Sampled native shader deltas in linear color and tangent normal space',
            'component':data['component'],'slot':data['slot'],'size':size,'columns':columns,'rows':rows,'layer_count':count,'samples':samples,'maps':resources,
            'neutral_preserved':True,'source_manifest_sha256':source_sha,'bank_sha256':data['bank_sha256'],'limits':data['limitations']}
