"""Prepare persistent numeric delta sources before native texture refresh."""
import json,os,hashlib,subprocess
from pathlib import Path

def digest(path):
    with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()

def prepare(request,folder):
    folder=Path(folder);snapshot=Path(request.get('animated_input_root',folder/'scripts')).resolve();items=[];maps={item['kind']:item for item in request['maps'] if item.get('role','HEAD')=='HEAD'}
    for kind,item in maps.items():
        if kind not in {'WM1','WM2','WM3','CM1','CM2','CM3'}:continue
        source=(snapshot/item['snapshot']).resolve()
        if not source.is_relative_to(snapshot.resolve()) or digest(source)!=item['sha256']:raise ValueError('Animated snapshot changed')
        row={'kind':kind,'source':str(source),'sha256':item['sha256'],'encoding':item.get('encoding','FULL')}
        if row['encoding']=='FULL':
            base=request.get('animated_normal_base') if kind.startswith('WM') else None
            base=base or maps.get('NORMAL' if kind.startswith('WM') else 'ALBEDO')
            if not base:raise ValueError('Animated full maps need their neutral normal/albedo in the refresh')
            path=(snapshot/base['snapshot']).resolve()
            if not path.is_relative_to(snapshot.resolve()) or digest(path)!=base['sha256']:raise ValueError('Animated neutral snapshot changed')
            row.update(base=str(path),base_sha256=base['sha256'])
        items.append(row)
    if not items:return {}
    folder.mkdir(parents=True,exist_ok=True)
    python=Path(request.get('image_python',''))
    if not python.is_file():raise ValueError('Choose the local image Python before refreshing animated maps')
    spec={'items':items,'output':str(folder/'native-inputs'),'result':str(folder/'native-animated.json'),'parent':os.getpid()}
    path=folder/'native-animated-request.json';path.write_text(json.dumps(spec,indent=2),encoding='utf-8')
    if Path(spec['output']).exists():raise ValueError('Native delta inputs already exist; inspect the original refresh result rather than dispatching it again')
    with (folder/'native-animated.log').open('w',encoding='utf-8') as log:
        completed=subprocess.run([str(python),str(Path(__file__).with_name('native_animated_worker.py')),str(path)],stdout=log,stderr=subprocess.STDOUT,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
    result=json.loads(Path(spec['result']).read_bytes())
    if completed.returncode or result.get('status')!='complete':raise ValueError(result.get('error','Native animated-map encoding failed'))
    if set(result['maps'])!={row['kind'] for row in items}:raise ValueError('Native encoding returned an incomplete map set')
    for row in result['maps'].values():
        output=Path(row['file']).resolve()
        if output.parent!=Path(spec['output']).resolve() or digest(output)!=row['sha256']:raise ValueError('Native delta output changed')
    return result['maps']
