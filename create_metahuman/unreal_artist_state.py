"""Read-only native artist-state capture, including opaque project packages.

No unsupported property is silently discarded. Reflection is an inspectable
view; compressed native packages are the preservation authority. A snapshot is
not a promise that Blender or WebGL can execute every Unreal feature.
"""
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import runpy
import uuid
import zlib

GENERATOR='CreateMetaHuman.UnrealArtistState'


def capture(character_path,directory,actor_path='',maximum_package_bytes=4*1024**3,expected_project=''):
    import unreal
    destination=Path(directory).resolve()
    project=Path(unreal.Paths.get_project_file_path()).resolve()
    if expected_project and project!=Path(expected_project).resolve():raise ValueError('Open the linked Unreal project before capturing artist edits')
    if destination.exists() and any(destination.iterdir()):raise ValueError('Choose a new artist-state snapshot directory')
    library=getattr(unreal,'CreateMHArtistLibrary',None)
    if library is None:raise ValueError('Native artist-state capture requires Bridge 0.9 or later')
    character=unreal.load_asset(character_path)
    if not isinstance(character,unreal.MetaHumanCharacter):raise ValueError('Choose an explicit MetaHuman character')
    reflected=json.loads(library.snapshot_object(character))
    if not reflected.get('objects'):raise ValueError('Native reflection returned no character state')
    if any(item['package_dirty'] for item in reflected['objects']):
        raise ValueError('Save the character before capturing its native package state; unsaved edits were not overwritten')
    actor=None;actor_state=None;materials=[];actor_packages=set()
    if actor_path:
        if actor_path=='@selected':
            selected=unreal.get_editor_subsystem(unreal.EditorActorSubsystem).get_selected_level_actors()
            if len(selected)!=1:raise ValueError('Select exactly one saved character actor in the Unreal level, or choose Creator character capture')
            actor=selected[0];actor_path=actor.get_path_name()
        else:actor=unreal.find_object(None,actor_path)
        if not isinstance(actor,unreal.Actor):raise ValueError('The explicitly selected artist actor is no longer live')
        actor_state=json.loads(library.snapshot_object(actor))
        if not actor_state.get('objects'):raise ValueError('The actor is transient; save a level actor or capture the Creator character instead')
        if any('package' not in item for item in actor_state['objects']):raise ValueError('Update the Unreal bridge to capture actor package ownership')
        if any(item['package_dirty'] for item in actor_state['objects']):raise ValueError('Save the selected actor and its level before capture; unsaved edits were not overwritten')
        for item in actor_state['objects']:
            actor_packages.update((item['package'],item['outermost_package'],item['class'].split('.')[0]))
        if not any(p.startswith('/Game/') for p in actor_packages):raise ValueError('The actor must belong to a saved project level')
        record=runpy.run_path(str(Path(__file__).with_name('unreal_web.py')))['material_record']
        for component in actor.get_components_by_class(unreal.MeshComponent):
            materials.append({'component':component.get_name(),'class':component.get_class().get_path_name(),
                'materials':[record(component.get_material(i)) for i in range(component.get_num_materials())]})
    content=Path(unreal.Paths.project_content_dir()).resolve()
    registry=unreal.AssetRegistryHelpers.get_asset_registry()
    options=unreal.AssetRegistryDependencyOptions(include_soft_package_references=True,include_hard_package_references=True)
    dirty={p.get_path_name() for p in [*unreal.EditorLoadingAndSavingUtils.get_dirty_content_packages(),*unreal.EditorLoadingAndSavingUtils.get_dirty_map_packages()]}
    pending=[character_path.split('.')[0],*actor_packages];seen=set();external=set();files=[]
    while pending:
        package=pending.pop()
        if package in seen:continue
        seen.add(package)
        if len(seen)>5000:raise ValueError('Artist-state dependency closure exceeds the bounded capture size')
        if not package.startswith('/Game/'):
            external.add(package);continue
        relative=package.removeprefix('/Game/')
        base=(content/relative).resolve()
        if not base.is_relative_to(content):raise ValueError('Invalid project package reference')
        package_files=[base.with_suffix(s) for s in ('.uasset','.umap','.uexp','.ubulk','.uptnl') if base.with_suffix(s).is_file()]
        if not package_files:raise ValueError('Referenced project package is missing on disk: '+package)
        if package in dirty:raise ValueError('Save modified dependency before capture: '+package)
        files.extend((package,p) for p in package_files)
        pending.extend(str(d) for d in registry.get_dependencies(package,options))
    size=sum(p.stat().st_size for _,p in files)
    if size>maximum_package_bytes:raise ValueError('Native package snapshot exceeds its explicit disk budget')
    budget=runpy.run_path(str(Path(__file__).with_name('resource_budget.py')))
    budget['require_disk']('native artist-state preservation',destination,size+256*1024**2)
    budget['require']('native artist-state preservation',256*1024**2)
    destination.mkdir(parents=True,exist_ok=True)
    blobs=destination.parent/'native-package-blobs';blobs.mkdir(exist_ok=True)
    report={'schema':1,'generator':GENERATOR,'status':'capturing','character':character_path,
        'engine':unreal.SystemLibrary.get_engine_version(),'project':str(Path(unreal.Paths.get_project_file_path()).resolve()),
        'captured_at':datetime.now(timezone.utc).isoformat(),'reflected':reflected,'live_components':materials,
        'actor':actor_path,'actor_reflected':actor_state,
        'capture_scope':'character_and_saved_level_actor' if actor else 'character',
        'root_packages':sorted({character_path.split('.')[0],*actor_packages}),
        'native_files':[],'external_dependencies':sorted(external),'archive_bytes_uncompressed':size,
        'application_status':'preserved; no Blender or Unreal properties have been overwritten',
        'limitations':['Engine/plugin dependencies remain versioned external requirements',
            'Selected actor capture includes its saved level and dependency closure, not the entire project',
            'Transient execution state is not a restorable native package; live material recipes are retained separately',
            'Dynamic rendering and unsupported features require explicit renderer adapters or baking']}
    for package,path in files:
        before=path.stat();sha=hashlib.sha256()
        with path.open('rb') as source:
            while block:=source.read(1024**2):sha.update(block)
        digest=sha.hexdigest();target=blobs/(digest+'.z');pending_file=blobs/(digest+'.'+uuid.uuid4().hex+'.pending')
        if not target.exists():
            compressor=zlib.compressobj(3);verify=hashlib.sha256()
            try:
                with path.open('rb') as source,pending_file.open('xb') as output:
                    while block:=source.read(1024**2):verify.update(block);output.write(compressor.compress(block))
                    output.write(compressor.flush())
                if verify.hexdigest()!=digest:raise ValueError('Native asset changed during preservation: '+package)
                pending_file.replace(target)
            finally:
                if pending_file.exists():pending_file.unlink()
        else:
            decompressor=zlib.decompressobj();verify=hashlib.sha256()
            with target.open('rb') as stream:
                while block:=stream.read(1024**2):verify.update(decompressor.decompress(block))
                verify.update(decompressor.flush())
            if verify.hexdigest()!=digest or not decompressor.eof:raise ValueError('Existing native package blob failed integrity verification')
        after=path.stat()
        if (before.st_size,before.st_mtime_ns)!=(after.st_size,after.st_mtime_ns):raise ValueError('Native asset changed during capture')
        report['native_files'].append({'package':package,'relative_file':path.relative_to(content).as_posix(),
            'sha256':digest,'bytes':before.st_size,'compressed_bytes':target.stat().st_size,'blob':'../native-package-blobs/'+target.name})
    report['status']='complete'
    output=destination/'artist-state.json';temporary=output.with_suffix('.tmp')
    temporary.write_text(json.dumps(report,indent=2),encoding='utf-8');temporary.replace(output)
    return {'status':'complete','manifest':str(output),'character':character_path,'files':len(files),
        'objects':len(reflected['objects'])+len((actor_state or {}).get('objects',[])),'native_bytes':size,'compressed_bytes':sum(r['compressed_bytes'] for r in report['native_files'])}
