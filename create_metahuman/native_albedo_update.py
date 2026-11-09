"""Stage source color for an existing native character without import history."""
import hashlib
import json
import re
import shutil
from pathlib import Path
import bpy
import numpy as np


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def prepare(context, folder, *, selection=None):
    from . import conform_geometry, mesh_ops, resource_budget, surface_identity, skin_projection_history
    s, f = context.scene.cm_settings, context.scene.cm_features
    if context.mode != 'OBJECT' or f.expression_active or f.body_editing or f.full_sculpt or f.restore_editing:
        raise ValueError('Finish the current edit and return to Neutral before applying albedo')
    if not re.fullmatch(r'/Game(?:/[A-Za-z0-9_]+)+(?:\.[A-Za-z0-9_]+)?', f.unreal_character):
        raise ValueError('Choose the linked native MetaHuman Character under /Game')
    expected = Path(bpy.path.abspath(f.unreal_character_project)).resolve()
    if expected.suffix.lower() != '.uproject' or not expected.is_file():
        raise ValueError('Choose the linked character’s Unreal project file')
    project = Path(bpy.path.abspath(s.project_dir)).resolve()
    roles = ('HEAD', 'BODY') if s.part == 'FULL_BODY' and f.skin_bake_scope == 'BOTH' else (
        f.texture_part if s.part == 'FULL_BODY' else s.part,)
    mode='NATIVE_ALBEDO' if selection is None else 'NATIVE_SKIN_MAPS'
    if selection is None:selection={role:['ALBEDO'] for role in roles}
    elif (not isinstance(selection,dict) or not selection or set(selection)-{'HEAD','BODY'}
            or any(not kinds or len(kinds)!=len(set(kinds)) or set(kinds)-{'NORMAL','CAVITY','ROUGHNESS','BENT_NORMAL_AO'} for kinds in selection.values())):
        raise ValueError('Select unique native normal, cavity or roughness channels per part')
    roles=tuple(selection)
    if 'BODY' in roles and s.part not in {'BODY','FULL_BODY'} or 'HEAD' in roles and s.part=='BODY':raise ValueError('Selected native map part is missing from this character')
    problems = skin_projection_history.problems(context.scene, kinds={'ALBEDO'} if mode=='NATIVE_ALBEDO' else {'ROUGHNESS'} if any('ROUGHNESS' in kinds for kinds in selection.values()) else set(),roles=set(roles))
    if mode=='NATIVE_SKIN_MAPS':
        from . import map_provenance
        for role in roles:
            kinds=set(selection[role]);keys=['FINAL_'+role] if kinds&{'NORMAL','CAVITY'} else []
            if 'ROUGHNESS' in kinds and not skin_projection_history.current_projection(context.scene,role,'ROUGHNESS'):
                obj,_=skin_projection_history.part(context.scene,role)
                authored=json.loads(obj.get(skin_projection_history.PROPERTY,'{}')).get('channels',{}).get('ROUGHNESS',{}).get('ownership')=='AUTHORED'
                if not authored:keys+=['ROUGHNESS_'+role]
            problems+=map_provenance.problems(context.scene,keys=keys,suffixes=tuple('_'+kind+'.png' for kind in kinds))
    if problems:
        raise ValueError(problems[0])
    resource_budget.require('native albedo preparation', 512 * resource_budget.MIB)
    geometry = conform_geometry.prepare(context, s, f) if 'HEAD' in roles else {'vertices': {}}
    if mode=='NATIVE_SKIN_MAPS':geometry['vertices']={key:value for key,value in geometry['vertices'].items() if key=='head'}
    records = []
    for role in roles:
        obj = f.body_result if role == 'BODY' and s.part == 'FULL_BODY' else s.result
        template = f.body_template if role == 'BODY' and s.part == 'FULL_BODY' else s.template
        mesh_ops.check_mesh(obj, role.title() + ' result')
        if not template or not surface_identity.matches(obj, [template]):
            raise ValueError('Keep the complete native ' + role.lower() + ' UV layout before applying albedo')
        if role == 'BODY':
            points = mesh_ops.evaluated_coordinates(context, obj)
            if len(points) != len(obj.data.vertices) or not np.isfinite(points).all():
                raise ValueError('Finish topology-changing body modifiers before applying albedo')
            geometry['vertices']['body'] = (points * [1, -1, 1] * (context.scene.unit_settings.scale_length * 100)).tolist()
        prefix = s.character_name + ('_Body' if role == 'BODY' and s.part == 'FULL_BODY' else '')
        for kind in selection[role]:
            source = project / 'Maps' / (prefix + '_'+kind+'.png')
            if not re.fullmatch(r'[A-Za-z0-9_]+\.png', source.name) or source.resolve().parent != project / 'Maps' or not source.is_file():
                raise ValueError('Save the ' + role.lower() + ' '+kind.lower()+' inside this character\'s Maps folder')
            for image in bpy.data.images:
                if image.filepath and Path(bpy.path.abspath(image.filepath)).resolve() == source.resolve() and image.is_dirty:
                    raise ValueError('Save the painted ' + role.lower() + ' '+kind.lower()+' before applying it in Unreal')
            with source.open('rb') as stream:header = stream.read(24)
            size = [int.from_bytes(header[i:i+4], 'big') for i in (16, 20)]
            if header[:8] != b'\x89PNG\r\n\x1a\n' or any(n < 1 or n > 8192 for n in size):raise ValueError('Native maps need PNG files through 8K')
            records.append(dict(role=role, kind=kind, source=str(source), snapshot='Maps/' + source.name,
                                sha256=digest(source), resolution=size))
    resource_budget.require_disk('native albedo snapshot', project,
        sum(Path(item['source']).stat().st_size for item in records) + 2 * resource_budget.GIB)
    snapshot = folder / 'scripts'
    snapshot.mkdir(parents=True, exist_ok=False)
    for item in records:
        target = snapshot / item['snapshot']
        target.parent.mkdir(exist_ok=True)
        shutil.copy2(item['source'], target)
        if digest(target) != item['sha256']:
            raise ValueError('An albedo changed while preparing its snapshot')
    geometry_path = snapshot / 'geometry.json'
    geometry_path.write_text(json.dumps(geometry), encoding='utf-8')
    for name in ('unreal_texture_update.py', 'unreal_native_albedo.py', 'unreal_revision_textures.py',
                 'unreal_import.py', 'unreal_roughness.py', 'resource_budget.py', 'process_ref.py'):
        shutil.copy2(Path(__file__).with_name(name), snapshot / name)
    if mode=='NATIVE_SKIN_MAPS':shutil.copy2(Path(__file__).with_name('unreal_native_maps.py'),snapshot/'unreal_native_maps.py')
    built = json.loads(f.unreal_build_report or '{}')
    request = dict(schema_version=1, action='UPDATE_TEXTURES', geometry_mode=mode,
        character=f.unreal_character, expected_project=str(expected), project_dir=str(project),
        character_name=s.character_name, maps=records, geometry_sha256=digest(geometry_path),
        existing_assembly=built.get('character') == f.unreal_character and built.get('build', {}).get('state') == 'COMPLETE')
    if any('ROUGHNESS' in kinds for kinds in selection.values()):
        from . import roughness_response
        request['roughness_response']=roughness_response.profiles(context.scene,roles={role for role,kinds in selection.items() if 'ROUGHNESS' in kinds})
    (folder / 'request.json').write_text(json.dumps(request, indent=2), encoding='utf-8')
    return request
