"""Refresh delivered Texture2D assets without conforming, rigging or assembling."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json
import math
import itertools


def _hash(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


EXPRESSION_LEDGER='CreateMetaHuman.ExpressionColorBindings'
WRINKLE_LEDGER='CreateMetaHuman.WrinkleNormalBindings'


def expression_plan(character,desired,scope=False,*,normal=False):
    """Plan additions/removals without mutating live Unreal struct proxies."""
    import unreal
    raw=unreal.EditorAssetLibrary.get_metadata_tag(character,WRINKLE_LEDGER if normal else EXPRESSION_LEDGER)
    ledger=json.loads(raw) if raw else {'schema':1,'owner':character.get_path_name(),'slots':{}}
    if ledger.get('schema')!=1 or ledger.get('owner')!=character.get_path_name():
        raise ValueError('Animated-map ownership belongs to another character or version')
    face=dict(character.get_editor_property('skin_settings').get_editor_property('texture_material_overrides').get_editor_property('texture_overrides').get_editor_property('face'))
    changes=[]
    for slot in (('WM1','WM2','WM3') if normal else ('CM1','CM2','CM3')):
        key=getattr(unreal.FaceTextureType,('NORMAL_ANIMATED_' if normal else 'BASECOLOR_ANIMATED_')+slot)
        current=face.get(key);path=current.get_path_name() if current else None
        owned=ledger['slots'].get(slot);target=desired.get(slot)
        if owned and (target or scope) and path!=owned['asset']:
            raise ValueError(('Wrinkle normal ' if normal else 'Expression color ')+slot+' was edited in Unreal; retain that edit before updating its managed binding')
        if target:
            if not target.startswith(character.get_path_name().rsplit('/',1)[0]+'/Maps/'):
                raise ValueError('Expression texture belongs to another character')
            if not owned:
                owned={'baseline':path,'was_present':key in face,'baseline_known':path!=target}
                ledger['slots'][slot]=owned
            owned['asset']=target
            changes.append({'slot':slot,'asset':target,'remove':False})
        elif scope and owned:
            if not owned.get('baseline_known',True):
                raise ValueError('The original '+slot+' binding predates ownership tracking; retain its verified original binding before reverting')
            baseline=owned.get('baseline')
            if baseline and not unreal.load_asset(baseline):raise ValueError('Original expression color is unavailable: '+baseline)
            changes.append({'slot':slot,'asset':baseline,'remove':not owned.get('was_present',False)})
            del ledger['slots'][slot]
    return changes,ledger


def commit_expressions(character,changes,ledger,*,normal=False):
    import unreal
    if not changes:return
    subsystem=unreal.get_editor_subsystem(unreal.MetaHumanCharacterEditorSubsystem)
    skin=character.get_editor_property('skin_settings')
    overrides=skin.get_editor_property('texture_material_overrides')
    textures=overrides.get_editor_property('texture_overrides')
    face=dict(textures.get_editor_property('face'))
    for change in changes:
        key=getattr(unreal.FaceTextureType,('NORMAL_ANIMATED_' if normal else 'BASECOLOR_ANIMATED_')+change['slot'])
        if change['remove']:face.pop(key,None)
        else:
            value=unreal.load_asset(change['asset']) if change['asset'] else None
            if change['asset'] and not isinstance(value,unreal.Texture2D):raise ValueError('Animated map is not a Texture2D')
            face[key]=value
    owned=not subsystem.is_object_added_for_editing(character)
    try:
        if owned and not subsystem.try_add_object_to_edit(character):raise RuntimeError('Creator could not edit expression color overrides')
        textures.set_editor_property('face',face)
        overrides.set_editor_property('texture_overrides',textures)
        skin.set_editor_property('texture_material_overrides',overrides)
        subsystem.commit_skin_settings(character,skin)
        # Managed roughness MICs retain explicit texture arrays. Removing a
        # FaceTextureType entry alone otherwise leaves stale CM inputs there.
        import runpy
        bindings=runpy.run_path(str(Path(__file__).with_name('unreal_roughness.py')))
        seen=set()
        for component,_,material,_,_,_ in bindings['capture'](character):
            if component.get_name()!='Face':continue
            parent=material.get_editor_property('parent')
            if not parent:continue
            managed=unreal.EditorAssetLibrary.get_metadata_tag(parent,'CreateMetaHuman.RoughnessOwner')==character.get_path_name()
            native=parent.get_editor_property('parent') if managed else parent
            if not native or not native.get_name().startswith('MI_Skin_Head_'):continue
            for change in changes:
                name=('Normal Animated Delta ' if normal else 'Basecolor Animated Delta ')+change['slot'].lower()
                texture=unreal.load_asset(change['asset']) if change['asset'] else unreal.MaterialEditingLibrary.get_material_instance_texture_parameter_value(native,name)
                if not texture:raise ValueError('Native expression fallback is missing: '+name)
                material.set_texture_parameter_value(name,texture)
                if managed and (parent.get_path_name(),name) not in seen:
                    if unreal.MaterialEditingLibrary.get_material_instance_texture_parameter_value(parent,name)!=texture:
                        unreal.MaterialEditingLibrary.set_material_instance_texture_parameter_value(parent,name,texture)
                        if not unreal.EditorAssetLibrary.save_loaded_asset(parent,False):raise RuntimeError('Could not save expression material binding')
                    seen.add((parent.get_path_name(),name))
        unreal.EditorAssetLibrary.set_metadata_tag(character,WRINKLE_LEDGER if normal else EXPRESSION_LEDGER,json.dumps(ledger,sort_keys=True))
        if not unreal.EditorAssetLibrary.save_loaded_asset(character,False):raise RuntimeError('Could not save animated-map overrides')
    finally:
        if owned:subsystem.remove_object_to_edit(character)


def _geometry_match(expected, actual, tolerance=.002):
    """FBX may reorder/weld UV copies; require both rest surfaces to match."""
    def cover(source, target):
        cells = {}
        for point in target:
            key = tuple(math.floor(x / tolerance) for x in point)
            cells.setdefault(key, []).append(point)
        maximum = 0.0
        for point in source:
            key = tuple(math.floor(x / tolerance) for x in point)
            best = float('inf')
            for offset in itertools.product((-1, 0, 1), repeat=3):
                for other in cells.get(tuple(a + b for a, b in zip(key, offset)), ()):
                    best = min(best, sum((a - b) ** 2 for a, b in zip(point, other)))
            if best > tolerance ** 2:
                raise ValueError('Accepted geometry differs from the delivered mesh; push a new character revision before updating textures')
            maximum = max(maximum, math.sqrt(best))
        return maximum
    if not expected or not actual:
        raise ValueError('Could not verify the delivered conformed mesh')
    return max(cover(expected, actual), cover(actual, expected))


def update(request_path):
    candidate=json.loads(Path(request_path).read_text(encoding='utf-8'))
    if candidate.get('geometry_mode')=='NATIVE_SOURCE_EYES':
        import runpy
        return runpy.run_path(str(Path(request_path).parent/'scripts/unreal_native_eyes.py'))['update'](request_path)
    if candidate.get('geometry_mode')=='NATIVE_ALBEDO':
        import runpy
        return runpy.run_path(str(Path(request_path).parent/'scripts/unreal_native_albedo.py'))['update'](request_path)
    if candidate.get('geometry_mode')=='NATIVE_SKIN_MAPS':
        import runpy
        return runpy.run_path(str(Path(request_path).parent/'scripts/unreal_native_maps.py'))['update'](request_path)
    if candidate.get('geometry_mode')=='NATIVE_REVISION':
        import runpy
        return runpy.run_path(str(Path(request_path).parent/'scripts/unreal_revision_textures.py'))['update'](request_path)
    import unreal
    request_path = Path(request_path).resolve()
    folder = request_path.parent
    snapshot = folder / 'scripts'
    request = json.loads(request_path.read_text(encoding='utf-8'))
    project = str(Path(unreal.Paths.get_project_file_path()).resolve())
    if request.get('schema_version') != 1 or request.get('action') != 'UPDATE_TEXTURES':
        raise ValueError('Unsupported texture update request')
    if not request.get('expected_project') or Path(request['expected_project']).resolve() != Path(project):
        raise ValueError('Open the Unreal project used for this character delivery')
    character = unreal.load_asset(request['character'])
    if not isinstance(character, unreal.MetaHumanCharacter):
        raise ValueError('The linked native MetaHuman character is missing')
    asset_folder = character.get_path_name().rsplit('/', 1)[0]
    geometry_file = snapshot / 'geometry.json'
    if _hash(geometry_file) != request['geometry_sha256']:
        raise ValueError('Texture update geometry checks changed after preparation')
    geometry = json.loads(geometry_file.read_text(encoding='utf-8'))
    checks = {}
    for role, record in geometry.items():
        asset = unreal.load_asset(record['asset'])
        if not isinstance(asset, unreal.StaticMesh) or not asset.get_path_name().startswith(asset_folder + '/'):
            raise ValueError('The original conformed ' + role + ' mesh is missing or belongs to another character')
        data = unreal.MetaHumanCharacterEditorSubsystem.get_mesh_data_for_conforming(asset)
        if not data:
            raise ValueError('Unreal could not inspect the original conformed mesh')
        actual = [(p.x, p.y, p.z) for p in data[0]]
        checks[role] = {'maximum_distance_cm': _geometry_match(record['vertices'], actual),
                        'source_vertices': len(record['vertices']), 'imported_vertices': len(actual)}
    tasks = []
    seen = set()
    for item in request['maps']:
        relative = Path(item['snapshot'])
        filename = (snapshot / relative).resolve()
        if relative.is_absolute() or not filename.is_relative_to(snapshot.resolve()) or _hash(filename) != item['sha256']:
            raise ValueError('Texture snapshot is missing, changed or outside the update folder')
        asset = unreal.load_asset(item['asset'])
        expression_color=bool(item.get('allow_expression_color') and item['role']=='HEAD' and item['kind'] in {'CM1','CM2','CM3'})
        wrinkle_normal=bool(item.get('allow_wrinkle_normal') and item['role']=='HEAD' and item['kind'] in {'WM1','WM2','WM3'})
        roughness=bool(item.get('allow_skin_roughness') and item['role'] in {'HEAD','BODY'} and item['kind']=='ROUGHNESS')
        expected=asset_folder+'/Maps/'+filename.stem+'.'+filename.stem
        if (expression_color or wrinkle_normal or roughness) and item['asset']!=expected:
            raise ValueError('Expression color target differs from its verified filename')
        if (asset is not None and not isinstance(asset,unreal.Texture2D)) or (asset is None and not (expression_color or wrinkle_normal or roughness)) or not item['asset'].startswith(asset_folder+'/Maps/'):
            raise ValueError('A delivered texture is missing or belongs to another character: ' + item['asset'])
        if item['asset'] in seen:
            raise ValueError('The texture update repeats an asset')
        seen.add(item['asset'])
        tasks.append((item, asset, filename))
    if not tasks:
        raise ValueError('No texture updates were provided')
    desired={item['kind']:item['asset'] for item,_,_ in tasks if item.get('allow_expression_color')}
    expression_changes,expression_ledger=expression_plan(character,desired,request.get('expression_color_scope',False))
    desired_wrinkles={item['kind']:item['asset'] for item,_,_ in tasks if item.get('allow_wrinkle_normal')}
    wrinkle_changes,wrinkle_ledger=expression_plan(character,desired_wrinkles,request.get('wrinkle_normal_scope',False),normal=True)
    import runpy
    bindings=runpy.run_path(str(snapshot/'unreal_roughness.py'))
    live=bindings['capture'](character)
    report = {'status': 'importing', 'action': 'UPDATE_TEXTURES', 'character': character.get_path_name(),
              'unreal_project_file': project, 'updated': [], 'warnings': [], 'geometry_checks': checks,
              'requires_reassembly': bool(request.get('existing_assembly')), 'retain_source_snapshot': False}
    def commit():
        path = folder / 'texture-progress.json'
        temporary = path.with_suffix('.tmp')
        temporary.write_text(json.dumps(report, indent=2), encoding='utf-8')
        temporary.replace(path)
    commit()
    try:
        animated=runpy.run_path(str(snapshot/'unreal_native_animated.py'))['prepare'](request,folder)
        if animated:report['animated_map_encoding']=animated
        for item, original, filename in tasks:
            native_delta=animated.get(item['kind']) if item['role']=='HEAD' else None
            if native_delta:filename=Path(native_delta['file'])
            settings = ({key: original.get_editor_property(key) for key in ('srgb', 'compression_settings')}
                        if original else {'srgb':True,'compression_settings':unreal.TextureCompressionSettings.TC_DEFAULT})
            if item.get('allow_skin_roughness'):
                settings={'srgb':False,'compression_settings':unreal.TextureCompressionSettings.TC_MASKS}
            if item.get('allow_wrinkle_normal') and not original:
                settings={'srgb':False,'compression_settings':unreal.TextureCompressionSettings.TC_BC7}
            if native_delta:settings={'srgb':False,'compression_settings':unreal.TextureCompressionSettings.TC_BC7}
            package, name = item['asset'].split('.')[0].rsplit('/', 1)
            task = unreal.AssetImportTask()
            for key, value in (('filename', str(filename)), ('destination_path', package), ('destination_name', name),
                               ('automated', True), ('save', True), ('replace_existing', True), ('replace_existing_settings', False)):
                task.set_editor_property(key, value)
            unreal.AssetToolsHelpers.get_asset_tools().import_asset_tasks([task])
            if list(task.get_editor_property('imported_object_paths')) != [item['asset']]:
                raise RuntimeError('Unreal did not refresh the expected texture: ' + item['asset'])
            texture = unreal.load_asset(item['asset'])
            for key, value in settings.items():
                if texture.get_editor_property(key) != value:
                    texture.set_editor_property(key, value)
            row = {'asset': texture.get_path_name(), 'sha256': item['sha256'], 'role': item['role'], 'kind': item['kind']}
            report['updated'].append(row)
            source = Path(item['source'])
            try:
                if _hash(source) != item['sha256']:
                    raise ValueError('Source changed during update')
                if native_delta:
                    row.update(original_source=str(source),native_sha256=native_delta['sha256'],encoding=native_delta['encoding'])
                    source=filename
                imports = texture.get_editor_property('asset_import_data')
                imports.scripted_add_filename(str(source), 0, '')
                if Path(imports.extract_filenames()[0]).resolve() != source.resolve():
                    raise RuntimeError('Unreal did not accept the persistent Reimport source')
                row['source'] = str(source)
            except Exception as exc:
                report['retain_source_snapshot'] = True
                report['warnings'].append({'asset': texture.get_path_name(), 'error': str(exc)})
            if not unreal.EditorAssetLibrary.save_loaded_asset(texture, False):
                raise RuntimeError('Texture changed in memory but could not be saved: ' + item['asset'])
            commit()
        commit_expressions(character,expression_changes,expression_ledger)
        commit_expressions(character,wrinkle_changes,wrinkle_ledger,normal=True)
        report['expression_color_changes']=expression_changes
        report['wrinkle_normal_changes']=wrinkle_changes
        roughness={item['role']:unreal.load_asset(item['asset']) for item,_,_ in tasks if item.get('allow_skin_roughness')}
        if roughness:
            report['roughness_materials']=bindings['apply'](character,roughness,request.get('roughness_response'))
            report['roughness_response']=request.get('roughness_response',{})
        def changed(component,current,key):
            parent=current.get_editor_property('parent')
            name=parent.get_name() if parent else ''
            if not (name.startswith('MI_Skin_') or name.startswith('CM_Roughness_')):return False
            role='BODY' if component.get_name()=='Body' else 'HEAD'
            for change in expression_changes:
                if role=='HEAD' and key in {'Basecolor Animated '+change['slot'].lower(),'Basecolor Animated '+change['slot'].lower()+' VT','Basecolor Animated Delta '+change['slot'].lower()}:return True
            for change in wrinkle_changes:
                if role=='HEAD' and key in {'Normal Animated '+change['slot'].lower(),'Normal Animated '+change['slot'].lower()+' VT','Normal Animated Delta '+change['slot'].lower()}:return True
            return key=='Roughness' and role in roughness
        bindings['restore'](live,changed,bindings['managed_scalar_skip'](character,roughness))
        report['runtime_material_parameters_preserved']=True
        report['status'] = 'complete'
    except Exception as exc:
        report.update(status='failed', error=str(exc), retain_source_snapshot=True)
    report['completed_at'] = datetime.now(timezone.utc).isoformat()
    commit()
    return report
