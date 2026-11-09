"""Stage texture-only refreshes against a verified native head revision."""
import hashlib,json,re,shutil
from pathlib import Path
import bpy
import numpy as np


def digest(path):
    with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()


def prepare(context,folder):
    from . import head_revision,conform_geometry,resource_budget,map_provenance
    s,f=context.scene.cm_settings,context.scene.cm_features
    project=Path(bpy.path.abspath(s.project_dir)).resolve()
    history=head_revision.history(context.scene)
    record=json.loads(history.read_text(encoding='utf-8'))
    revision=Path(record['receipt']).resolve().parent
    if record.get('project_dir')!=str(project) or record.get('character_name')!=s.character_name or not revision.is_relative_to(project/'Export/.unreal-actions'):
        raise ValueError('Native revision history belongs to another Blender project')
    request_path=revision/'request.json'
    if digest(request_path)!=record.get('request_sha256'):raise ValueError('The native revision correspondence changed')
    receipt_path=revision/'receipt.json'
    if not receipt_path.is_file():receipt_path=revision/'head-revision.json'
    receipt=json.loads(receipt_path.read_text(encoding='utf-8'));original=json.loads(request_path.read_text(encoding='utf-8'))
    if receipt.get('status')!='complete' or receipt.get('character')!=f.unreal_character or receipt.get('request_sha256')!=record['request_sha256']:
        raise ValueError('No matching completed native head revision')
    expected=Path(bpy.path.abspath(f.unreal_character_project)).resolve()
    if Path(receipt.get('unreal_project_file','')).resolve()!=expected:raise ValueError('Native revision belongs to another Unreal project')
    geometry=conform_geometry.prepare(context,s,f)
    if set(geometry['vertices'])!=set(original['geometry']['vertices']):raise ValueError('Head anatomy changed; create a new head revision')
    for role,points in geometry['vertices'].items():
        before=np.asarray(original['geometry']['vertices'][role]);current=np.asarray(points)
        if before.shape!=current.shape or not np.isfinite(current).all() or not np.allclose(before,current,rtol=0,atol=1e-5):
            raise ValueError('The '+role+' geometry changed; create a new head revision before updating maps')
    scope=f.unreal_revision_update_scope
    if scope not in {'DETAIL','SKIN','ALL_HEAD'}:raise ValueError('Choose the native revision map scope')
    kinds=['NORMAL','CAVITY']
    if scope!='DETAIL':kinds+=['ALBEDO','CM1','CM2','CM3']
    if scope=='ALL_HEAD':kinds+=['WM1','WM2','WM3','ROUGHNESS']
    paths={kind:project/'Maps'/(s.character_name+'_'+kind+'.png') for kind in kinds}
    required={'NORMAL','CAVITY'}|({'ALBEDO'} if scope!='DETAIL' else set())
    if any(not paths[kind].is_file() for kind in required):raise ValueError('Save the selected head maps before refreshing Unreal')
    paths={kind:path for kind,path in paths.items() if path.is_file()}
    for group in ({'CM1','CM2','CM3'},{'WM1','WM2','WM3'}):
        present=set(paths)&group
        if present and present!=group:raise ValueError('Refresh all three '+('expression color' if 'CM1' in group else 'wrinkle normal')+' maps, or remove the incomplete set')
    for image in bpy.data.images:
        if image.filepath and Path(bpy.path.abspath(image.filepath)).resolve() in paths.values() and image.is_dirty:raise ValueError('Save selected map images before updating Unreal')
    keys=['FINAL_HEAD']+[k for k in paths if k.startswith(('WM','CM'))]
    if 'ROUGHNESS' in paths:keys+=['ROUGHNESS_HEAD']
    problems=map_provenance.problems(context.scene,keys=keys)
    if problems:raise ValueError(problems[0])
    resource_budget.require('native texture preparation',resource_budget.GIB)
    resource_budget.require_disk('native texture snapshot',project,sum(p.stat().st_size for p in paths.values())+2*resource_budget.GIB)
    snapshot=folder/'scripts';snapshot.mkdir(parents=True,exist_ok=False)
    records=[];asset_folder=f.unreal_character.rsplit('/',1)[0]+'/Maps/'
    for kind,path in paths.items():
        if not re.fullmatch(r'[A-Za-z0-9_]+\.png',path.name) or path.resolve().parent!=project/'Maps':raise ValueError('Invalid project map path')
        sha=digest(path);target=snapshot/'Maps'/path.name;target.parent.mkdir(exist_ok=True);shutil.copy2(path,target)
        if digest(target)!=sha:raise ValueError('A map changed during update preparation')
        records.append({'role':'HEAD','kind':kind,'asset':asset_folder+path.stem+'.'+path.stem,'source':str(path),'snapshot':'Maps/'+path.name,'sha256':sha})
        if kind.startswith(('WM','CM')):records[-1]['encoding']=f.unreal_wrinkle_input_encoding if kind.startswith('WM') else f.unreal_color_input_encoding
    animated_base=None
    if any(item['kind'].startswith('WM') and item.get('encoding')=='FULL' for item in records):
        from . import skin_bake_layers
        base=skin_bake_layers.native_wrinkle_baseline(context,[item['kind'] for item in records if item['kind'].startswith('WM') and item.get('encoding')=='FULL'])
        if base:
            target=snapshot/'neutral-geometry-normal.png';shutil.copy2(base,target);animated_base={'snapshot':target.name,'sha256':digest(target)}
    geometry_path=snapshot/'geometry.json';geometry_path.write_text(json.dumps(geometry),encoding='utf-8')
    for script in ('unreal_texture_update.py','unreal_revision_textures.py','unreal_roughness.py','resource_budget.py','process_ref.py','unreal_native_animated.py','native_animated_worker.py','worker_lifetime.py'):
        shutil.copy2(Path(__file__).with_name(script),snapshot/script)
    from . import roughness_response
    built=json.loads(f.unreal_build_report or '{}')
    request={'schema_version':1,'action':'UPDATE_TEXTURES','geometry_mode':'NATIVE_REVISION','scope':scope,
        'character':f.unreal_character,'expected_project':str(expected),'project_dir':str(project),'character_name':s.character_name,
        'maps':records,'geometry_sha256':digest(geometry_path),'roughness_response':roughness_response.profiles(context.scene),
        'image_python':bpy.path.abspath(f.inference_python),'animated_normal_base':animated_base,
        'existing_assembly':built.get('character')==f.unreal_character and built.get('build',{}).get('state')=='COMPLETE'}
    (folder/'request.json').write_text(json.dumps(request,indent=2),encoding='utf-8')
    return request
