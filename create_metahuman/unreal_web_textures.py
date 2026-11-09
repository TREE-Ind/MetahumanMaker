"""Resource-bounded extraction of native material texture pixels and metadata."""
import hashlib
import json
from pathlib import Path
import runpy
import struct
from tempfile import TemporaryDirectory


def export(source_manifest,directory,*,asset_paths=None):
    import unreal
    source_path=Path(source_manifest);source_bytes=source_path.read_bytes();source=json.loads(source_bytes.decode('utf-8-sig'));source_sha=hashlib.sha256(source_bytes).hexdigest()
    if source.get('generator')!='CreateMetaHuman.NativeWebSource' or source.get('status')!='source_extracted':
        raise ValueError('Complete the native character source extraction first')
    root=Path(directory).resolve();manifest=root/'texture-manifest.json'
    if root.exists() and any(root.iterdir()) and not manifest.is_file():
        raise ValueError('Native texture staging folder is not owned by this exporter')
    if manifest.is_file():
        previous=json.loads(manifest.read_text())
        if previous.get('generator')!='CreateMetaHuman.NativeWebTextures' or previous.get('source_actor_class')!=source['source_actor_class']:
            raise ValueError('An unrelated character texture manifest occupies the destination')
    references={}
    for part in source['skeletal_meshes']+source.get('grooms',[]):
        for slot,material in enumerate(part.get('materials',[])):
            if not material:continue
            for parameter,record in material['textures'].items():
                if record:references.setdefault(record['asset'],[]).append({'part':part['component'],'slot':slot,'parameter':parameter})
    if asset_paths is not None:
        if isinstance(asset_paths,str):raise ValueError('Texture selection must be a list of captured asset paths')
        selected=set(asset_paths)
        if not selected<=references.keys():raise ValueError('Texture selection contains assets outside the native capture')
        references={path:rows for path,rows in references.items() if path in selected}
    assets={path:unreal.load_asset(path) or unreal.find_object(None,path) for path in references}
    if any(asset is None for asset in assets.values()):raise ValueError('A referenced native texture asset is missing')
    estimate=sum(asset.blueprint_get_size_x()*asset.blueprint_get_size_y()*8 for asset in assets.values() if isinstance(asset,unreal.Texture2D))
    budget=runpy.run_path(str(Path(__file__).with_name('resource_budget.py')))
    budget['require_disk']('native web texture extraction',root,max(512*1024**2,estimate*2))
    largest=max((asset.blueprint_get_size_x()*asset.blueprint_get_size_y()*8 for asset in assets.values() if isinstance(asset,unreal.Texture2D)),default=0)
    budget['require']('native web texture extraction',512*1024**2+largest)
    root.mkdir(parents=True,exist_ok=True)
    result={'schema':1,'generator':'CreateMetaHuman.NativeWebTextures','status':'extracting','textures':{},'unresolved':[],
            'source_manifest_sha256':source_sha,
            'requested_texture_assets':sorted(references),
            'source_actor_class':source['source_actor_class'],'native_assets_modified':False}
    def checkpoint():
        temp=manifest.with_suffix('.tmp');temp.write_text(json.dumps(result,indent=2),encoding='utf-8');temp.replace(manifest)
    checkpoint()
    try:
        with TemporaryDirectory(prefix='.texture-',dir=root) as scratch:
            for asset_path,asset in assets.items():
                if not isinstance(asset,unreal.Texture2D):
                    result['unresolved'].append({'asset':asset_path,'class':asset.get_class().get_name(),'reason':'Texture2D pixel exporter only'})
                    continue
                pending=Path(scratch)/'texture.png'
                task=unreal.AssetExportTask();task.object=asset;task.filename=str(pending)
                task.exporter=unreal.TextureExporterPNG();task.automated=True;task.prompt=False;task.replace_identical=True
                if not unreal.Exporter.run_asset_export_task(task):raise RuntimeError('Could not export native texture: '+asset_path)
                with pending.open('rb') as stream:
                    header=stream.read(24);stream.seek(0);sha=hashlib.file_digest(stream,'sha256').hexdigest()
                if header[:8]!=b'\x89PNG\r\n\x1a\n':raise ValueError('Native texture export is not PNG')
                width,height=struct.unpack('>II',header[16:24])
                name=sha+'.png';target=root/name
                if target.exists():
                    with target.open('rb') as stream:old=hashlib.file_digest(stream,'sha256').hexdigest()
                    if old!=sha:raise ValueError('Native texture content-addressed file was modified')
                    pending.unlink()
                else:pending.replace(target)
                result['textures'][asset_path]={'file':name,'sha256':sha,'bytes':target.stat().st_size,
                    'width':width,'height':height,'srgb':asset.get_editor_property('srgb'),
                    'compression':str(asset.get_editor_property('compression_settings')),'references':references[asset_path]}
                checkpoint()
        if source_path.read_bytes()!=source_bytes:raise ValueError('Native capture changed during texture extraction')
        result['status']='pixels_extracted' if not result['unresolved'] else 'pixels_extracted_with_unresolved_resources';checkpoint()
        return result
    except BaseException as exc:
        result.update(status='failed',error=str(exc));checkpoint();raise
