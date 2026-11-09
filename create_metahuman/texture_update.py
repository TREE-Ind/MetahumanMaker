"""Prepare texture-only updates using the original character delivery inventory."""
from pathlib import Path
import hashlib
import json
import shutil
import re

import bpy
import numpy as np


def prepare(context, folder, *, native_selection=None, bundle=None, delivery=None):
    f=context.scene.cm_features
    if native_selection is not None:
        from . import native_map_update
        return native_map_update.prepare(context,folder,selection=native_selection)
    if f.unreal_revision_character and f.unreal_character==f.unreal_revision_character:
        from . import revision_texture_update
        return revision_texture_update.prepare(context,folder)
    from . import mesh_ops, resource_budget
    s, f = context.scene.cm_settings, context.scene.cm_features
    project = Path(bpy.path.abspath(s.project_dir)).resolve()
    name = re.sub(r'[^A-Za-z0-9_]', '_', s.character_name).strip('_')
    explicit_inventory=bundle is not None or delivery is not None
    if explicit_inventory and (bundle is None or delivery is None):
        raise ValueError('Supply both the export bundle and its successful delivery receipt')
    bundle = Path(bundle).resolve() if explicit_inventory else project / 'Export' / name
    if not bundle.is_relative_to(project/'Export'):
        raise ValueError('Texture inventory must belong to this character project')
    if not explicit_inventory and not (bundle/'last-character.json').is_file():
        from . import native_map_update
        return native_map_update.prepare(context,folder,selection=native_selection)
    record = ({'schema_version':1,'character_name':s.character_name,'project_dir':str(project),'receipt':delivery}
              if explicit_inventory else json.loads((bundle / 'last-character.json').read_text(encoding='utf-8')))
    if (record.get('schema_version') != 1 or record.get('character_name') != s.character_name
            or Path(record.get('project_dir', '')).resolve() != project):
        raise ValueError('The last delivery belongs to another Blender character project')
    receipt = record.get('receipt', {})
    if receipt.get('status') != 'complete' or receipt.get('character') != f.unreal_character:
        raise ValueError('No matching successful character delivery. Push this character before updating textures')
    if (not f.unreal_character_project or Path(receipt.get('unreal_project_file', '')).resolve()
            != Path(f.unreal_character_project).resolve()):
        raise ValueError('The last delivery belongs to another Unreal project')
    manifest = json.loads((bundle / 'manifest.json').read_text(encoding='utf-8'))
    if manifest.get('part') != s.part or manifest.get('name') != name:
        raise ValueError('Character part/name changed since delivery; push a new character revision')
    assets = {asset.rsplit('/', 1)[-1].split('.')[0]: asset for asset in receipt.get('texture_assets', [])}
    records = []
    sets = manifest.get('texture_sets', {s.part: manifest.get('maps', [])})
    for role, paths in sets.items():
        if role not in {'HEAD', 'BODY'}:
            continue
        for relative in paths:
            filename = Path(relative).name
            if not re.fullmatch(r'[A-Za-z0-9_]+\.png', filename):
                raise ValueError('Invalid texture filename in the delivery inventory')
            asset = assets.get(Path(filename).stem)
            if not asset:
                raise ValueError('This delivery has no ' + filename + ' asset. Push a new revision to add new maps')
            source = project / 'Maps' / filename
            if role=='HEAD' and Path(filename).stem.rsplit('_',1)[-1] in {'WM1','WM2','WM3','CM1','CM2','CM3'} and not source.is_file():continue
            if not source.is_file() or source.resolve().parent != project / 'Maps':
                raise ValueError('Missing project texture: ' + filename)
            records.append({'role': role, 'kind': Path(filename).stem.rsplit('_', 1)[-1],
                            'asset': asset, 'source': str(source), 'snapshot': 'Maps/' + filename})
            if records[-1]['kind']=='ROUGHNESS':records[-1]['allow_skin_roughness']=True
            if role=='HEAD' and records[-1]['kind'] in {'CM1','CM2','CM3'}:records[-1]['allow_expression_color']=True
            if role=='HEAD' and records[-1]['kind'] in {'WM1','WM2','WM3'}:records[-1]['allow_wrinkle_normal']=True
    # Animated maps can be introduced after the original conform. Keep
    # using that character and its verified geometry, without a new rig/build.
    if s.part!='BODY':
        from . import map_provenance
        slots=[slot for slot in ('WM1','WM2','WM3','CM1','CM2','CM3')
               if (project/'Maps'/(s.character_name+'_'+slot+'.png')).is_file()]
        problems=map_provenance.problems(context.scene,keys=slots)
        if problems:raise ValueError(problems[0])
        existing={Path(item['source']).name for item in records}
        asset_folder=f.unreal_character.rsplit('/',1)[0]+'/Maps/'
        for slot in slots:
            source=project/'Maps'/(s.character_name+'_'+slot+'.png')
            if source.name in existing or not source.is_file():continue
            if source.resolve().parent!=project/'Maps':raise ValueError('Expression color map leaves the project')
            asset=asset_folder+source.stem+'.'+source.stem
            flag='allow_wrinkle_normal' if slot.startswith('WM') else 'allow_expression_color'
            records.append({'role':'HEAD','kind':slot,'asset':asset,'source':str(source),
                            'snapshot':'Maps/'+source.name,flag:True})
    for role in (('HEAD','BODY') if s.part=='FULL_BODY' else (s.part,)):
        from . import map_provenance
        provenance=map_provenance.load(context.scene)
        key='ROUGHNESS_'+role
        if map_provenance.identity(context.scene,key) in provenance['entries']:
            problems=map_provenance.problems(context.scene,keys=[key])
            if problems:raise ValueError(problems[0])
        prefix=s.character_name+('_Body' if role=='BODY' and s.part=='FULL_BODY' else '')
        source=project/'Maps'/(prefix+'_ROUGHNESS.png')
        if not source.is_file():continue
        if source.resolve().parent!=project/'Maps':raise ValueError('Roughness map leaves the project')
        if any(Path(item['source']).name==source.name for item in records):continue
        asset=f.unreal_character.rsplit('/',1)[0]+'/Maps/'+source.stem+'.'+source.stem
        records.append({'role':role,'kind':'ROUGHNESS','asset':asset,'source':str(source),
                        'snapshot':'Maps/'+source.name,'allow_skin_roughness':True})
    if not records or len({r['asset'] for r in records}) != len(records):
        raise ValueError('The delivery inventory contains no unique texture assets')
    meshes = receipt.get('mesh_assets', [])
    targets = [('HEAD' if s.part != 'BODY' else 'BODY', s.result)]
    if s.part == 'FULL_BODY':
        targets.append(('BODY', f.body_result))
    if len(meshes) != len(targets):
        raise ValueError('The original conformed mesh inventory is incomplete; push a new revision')
    geometry = {}
    scale = context.scene.unit_settings.scale_length * 100
    for (role, obj), asset in zip(targets, meshes):
        points = mesh_ops.evaluated_coordinates(context, obj) * np.asarray([1, -1, 1]) * scale
        if len(points) != len(obj.data.vertices) or not np.isfinite(points).all():
            raise ValueError('Finish topology-changing modifiers before updating textures')
        geometry[role] = {'asset': asset, 'vertices': points.tolist()}
    required = sum(Path(r['source']).stat().st_size for r in records) + sum(len(r['vertices']) * 80 for r in geometry.values())
    resource_budget.require_disk('texture update snapshot', project, required + 2 * resource_budget.GIB)
    snapshot = folder / 'scripts'
    snapshot.mkdir(parents=True, exist_ok=False)
    for item in records:
        if item['role']=='HEAD' and item['kind'].startswith(('CM','WM')):
            item['encoding']=f.unreal_color_input_encoding if item['kind'].startswith('CM') else f.unreal_wrinkle_input_encoding
        target = snapshot / item['snapshot']
        target.parent.mkdir(exist_ok=True)
        shutil.copy2(item['source'], target)
        item['sha256'] = hashlib.sha256(target.read_bytes()).hexdigest()
    geometry_path = snapshot / 'geometry.json'
    geometry_path.write_text(json.dumps(geometry), encoding='utf-8')
    animated_base=None
    if any(item['role']=='HEAD' and item['kind'].startswith('WM') and item.get('encoding')=='FULL' for item in records):
        from . import skin_bake_layers
        base=skin_bake_layers.native_wrinkle_baseline(context,[item['kind'] for item in records if item['role']=='HEAD' and item['kind'].startswith('WM') and item.get('encoding')=='FULL'])
        if base:
            target=snapshot/'neutral-geometry-normal.png';shutil.copy2(base,target)
            animated_base={'snapshot':target.name,'sha256':hashlib.sha256(target.read_bytes()).hexdigest()}
    for script in ('unreal_texture_update.py', 'unreal_roughness.py','process_ref.py','unreal_native_animated.py','native_animated_worker.py','worker_lifetime.py'):
        shutil.copy2(Path(__file__).with_name(script), snapshot / script)
    try:
        built = json.loads(f.unreal_build_report or '{}')
    except ValueError:
        built = {}
    from . import roughness_response
    request = {'schema_version': 1, 'action': 'UPDATE_TEXTURES', 'character': f.unreal_character,
               'image_python':bpy.path.abspath(f.inference_python),'animated_normal_base':animated_base,
               'roughness_response':roughness_response.profiles(context.scene),
               'expected_project': f.unreal_character_project, 'project_dir': str(project),
               'character_name': s.character_name, 'maps': records,
               'expression_color_scope':s.part!='BODY',
               'wrinkle_normal_scope':s.part!='BODY',
               'geometry_sha256': hashlib.sha256(geometry_path.read_bytes()).hexdigest(),
               'existing_assembly': built.get('character') == f.unreal_character and built.get('build', {}).get('state') == 'COMPLETE'}
    (folder / 'request.json').write_text(json.dumps(request, indent=2), encoding='utf-8')
    return request
