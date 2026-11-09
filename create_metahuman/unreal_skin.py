"""Local MetaHuman skin synthesis and render capture; runs inside Unreal only.

Only our disposable scratch character is edited. This never requests cloud
textures or rigging, and never changes an existing character.
"""
from pathlib import Path
import json
import hashlib
import unreal


def _write(path, value):
    pending = path.with_suffix('.tmp')
    pending.write_text(json.dumps(value, indent=2), encoding='utf-8')
    pending.replace(path)


def _cleanup(data):
    character = unreal.load_asset(data['character']) if data.get('character') else None
    subsystem = unreal.get_editor_subsystem(unreal.MetaHumanCharacterEditorSubsystem)
    if character and subsystem.is_object_added_for_editing(character):
        subsystem.remove_object_to_edit(character)
    # The unique directory was created exclusively for this request.
    directory = data.get('scratch_directory', '')
    if directory.startswith('/Game/CreateMHSkinScratch/Job_'):
        unreal.EditorAssetLibrary.delete_directory(directory)


def prepare(request_path):
    request_path = Path(request_path)
    request = json.loads(request_path.read_text(encoding='utf-8'))
    folder = request_path.parent
    directory = '/Game/CreateMHSkinScratch/Job_' + folder.name
    if unreal.EditorAssetLibrary.does_directory_exist(directory):
        raise RuntimeError('This skin request already has a scratch character; do not dispatch it again')
    full_character = request.get('full_character', True)
    data = {'scratch_directory': directory, 'label': request['label'], 'materials': {},
            'synthesis': {key: request[key] for key in ('u', 'v', 'face_texture_index')}}
    if full_character and (not hasattr(unreal, 'CreateMHSkinLibrary') or
                           not hasattr(unreal.CreateMHSkinLibrary, 'bake_surface_map')):
        raise RuntimeError('Full-character surface maps require MetaHuman Maker Bridge 0.7 or newer')
    before = {x.get_path_name() for x in unreal.ObjectIterator(unreal.MaterialInstanceDynamic)}
    try:
        tools = unreal.AssetToolsHelpers.get_asset_tools()
        character = tools.create_asset('Skin', directory, unreal.MetaHumanCharacter, unreal.MetaHumanCharacterFactoryNew())
        if not character:
            raise RuntimeError('Could not create the local skin scratch character')
        data['character'] = character.get_path_name()
        if full_character:
            character.set_editor_property('preview_material_type', unreal.MetaHumanCharacterSkinPreviewMaterial.EDITABLE)
        subsystem = unreal.get_editor_subsystem(unreal.MetaHumanCharacterEditorSubsystem)
        if not subsystem.try_add_object_to_edit(character):
            raise RuntimeError('MetaHuman character editing is unavailable')
        settings = character.get_editor_property('skin_settings')
        skin = settings.get_editor_property('skin')
        for key in ('u', 'v', 'face_texture_index'):
            skin.set_editor_property(key, request[key])
        skin.set_editor_property('body_texture_index', int(request.get('body_texture_index', 0)))
        skin.set_editor_property('show_top_underwear', bool(request.get('show_top_underwear', True)))
        data['synthesis'].update(body_texture_index=int(request.get('body_texture_index', 0)),
                                 show_top_underwear=bool(request.get('show_top_underwear', True)))
        settings.set_editor_property('skin', skin)
        subsystem.commit_skin_settings(character, settings)
        data['settings'] = str(character.get_editor_property('skin_settings'))
        textures = dict(character.get_editor_property('synthesized_face_textures'))
        if full_character:
            textures.update(character.get_editor_property('body_textures'))
        for kind, texture in textures.items():
            name = str(kind).split('.')[1].split(':')[0]
            if name not in ('BASECOLOR', 'NORMAL', 'CAVITY', 'BODY_NORMAL', 'BODY_CAVITY'):
                continue
            normal = 'NORMAL' in name
            material = tools.create_asset('Capture' + name, directory, unreal.Material, unreal.MaterialFactoryNew())
            material.set_editor_property('shading_model', unreal.MaterialShadingModel.MSM_UNLIT)
            editing = unreal.MaterialEditingLibrary
            sample = editing.create_material_expression(material, unreal.MaterialExpressionTextureSampleParameter2D)
            sample.set_editor_property('parameter_name', 'CaptureTexture')
            sample.set_editor_property('texture', texture)
            if 'CAVITY' in name:
                compression = str(texture.get_editor_property('compression_settings'))
                sampler = unreal.MaterialSamplerType.SAMPLERTYPE_MASKS if 'MASKS' in compression else (
                    unreal.MaterialSamplerType.SAMPLERTYPE_LINEAR_GRAYSCALE if 'GRAYSCALE' in compression else unreal.MaterialSamplerType.SAMPLERTYPE_LINEAR_COLOR)
                sample.set_editor_property('sampler_type', sampler)
            if normal:
                sample.set_editor_property('sampler_type', unreal.MaterialSamplerType.SAMPLERTYPE_NORMAL)
                multiply = editing.create_material_expression(material, unreal.MaterialExpressionMultiply)
                multiply.set_editor_property('const_b', .5)
                add = editing.create_material_expression(material, unreal.MaterialExpressionAdd)
                add.set_editor_property('const_b', .5)
                editing.connect_material_expressions(sample, 'RGB', multiply, 'A')
                editing.connect_material_expressions(multiply, '', add, 'A')
                editing.connect_material_property(add, '', unreal.MaterialProperty.MP_EMISSIVE_COLOR)
            else:
                editing.connect_material_property(sample, 'RGB', unreal.MaterialProperty.MP_EMISSIVE_COLOR)
            editing.recompile_material(material)
            data['materials'][name] = material.get_path_name()
        if not {'BASECOLOR', 'NORMAL'}.issubset(data['materials']):
            raise RuntimeError('The installed local skin model did not provide albedo and normals')
        if full_character:
            candidates = {}
            for mid in unreal.ObjectIterator(unreal.MaterialInstanceDynamic):
                if mid.get_path_name() in before:
                    continue
                parent = mid.get_editor_property('parent')
                name = parent.get_name() if parent else ''
                for role, prefix in [('native_albedo', 'MI_Skin_Head_UI_LOD0'),
                                     ('body_albedo', 'MI_Skin_Body_MHC')]:
                    if name.startswith(prefix):
                        candidates.setdefault(role, []).append(mid)
            data['surface_materials'] = {}
            # Both parts must include the native chest color and clothing
            # stages. Mixing raw synthesized face pixels with a final body
            # material produces a visible bib at the head/body boundary.
            for role in ('native_albedo', 'body_albedo'):
                found = candidates.get(role, [])
                if len(found) != 1:
                    raise RuntimeError('Cannot identify the scratch character native ' + role + ' material')
                source = found[0]
                data['surface_materials'][role] = {'source': source.get_path_name(),
                    'uv_origin': [1.0 if role == 'body_albedo' else 0.0, 0.0],
                    'parameters': {name: source.get_scalar_parameter_value(name) for name in
                        ('UDIM', 'rbias', 'gbias', 'bbias', 'rgain', 'ggain', 'bgain', 'Use Texture Override')}}
                for suffix in ('normal', 'roughness'):
                    data['surface_materials'][role.replace('albedo', suffix)] = dict(data['surface_materials'][role])
        _write(folder / 'native-prepared.json', data)
    except Exception:
        _cleanup(data)
        raise
    return data


def capture(request_path):
    folder = Path(request_path).parent
    data = json.loads((folder / 'native-prepared.json').read_text(encoding='utf-8'))
    try:
        world = unreal.get_editor_subsystem(unreal.UnrealEditorSubsystem).get_editor_world()
        character = unreal.load_asset(data['character'])
        textures = {str(k).split('.')[1].split(':')[0]: t
                    for k, t in character.get_editor_property('synthesized_face_textures').items()}
        body_textures = {str(k).split('.')[1].split(':')[0]: t
                         for k, t in character.get_editor_property('body_textures').items()}
        outputs = {}
        specifications = [('BASECOLOR', 'native_albedo'), ('NORMAL', 'native_normal'), ('CAVITY', 'native_cavity')]
        if data.get('surface_materials'):
            specifications.extend([('BODY_BASECOLOR', 'body_albedo'), ('BODY_NORMAL', 'body_normal'), ('BODY_CAVITY', 'body_cavity')])
            specifications.extend([('ROUGHNESS', 'native_roughness'), ('BODY_ROUGHNESS', 'body_roughness')])
        for name, role in specifications:
            normal = 'NORMAL' in name
            surface = data.get('surface_materials', {}).get(role)
            filename = role + '.png'
            path = folder / filename
            if surface:
                source = unreal.find_object(None, surface['source'])
                origin = unreal.Vector2D(*surface['uv_origin'])
                kind = 'Normal' if normal else ('Roughness' if 'ROUGHNESS' in name else 'BaseColor')
                if not source or not unreal.CreateMHSkinLibrary.bake_surface_map(source, str(path), 2048, data['scratch_directory'], origin, kind):
                    raise RuntimeError('Native material baking failed for ' + role)
            else:
                material = unreal.load_asset(data['materials'][name])
                mid = unreal.MaterialLibrary.create_dynamic_material_instance(world, material)
                mid.set_texture_parameter_value('CaptureTexture', (body_textures if name.startswith('BODY_') else textures)[name])
                target = unreal.RenderingLibrary.create_render_target2d(world, 2048, 2048, unreal.TextureRenderTargetFormat.RTF_RGBA8)
                target.set_editor_property('target_gamma', 1.0 if normal or 'CAVITY' in name else 2.2)
                unreal.RenderingLibrary.clear_render_target2d(world, target, unreal.LinearColor(0, 0, 0, 1))
                unreal.RenderingLibrary.draw_material_to_render_target(world, target, mid)
                unreal.RenderingLibrary.export_render_target(world, target, str(folder), filename)
                unreal.RenderingLibrary.release_render_target2d(target)
            if not path.is_file() or path.stat().st_size < 1024:
                raise RuntimeError('Native skin render capture did not produce a readable image')
            outputs[role] = str(path)
            outputs[role + '_sha256'] = hashlib.sha256(path.read_bytes()).hexdigest()
        _write(folder / 'native-capture.json', dict(outputs, label=data['label'], settings=data['settings'],
                                                  synthesis=data['synthesis'], status='ok',
                                                  surface_sampling=data.get('surface_materials', {})))
        return outputs
    finally:
        _cleanup(data)


def cancel(request_path):
    prepared = Path(request_path).parent / 'native-prepared.json'
    if prepared.is_file():
        _cleanup(json.loads(prepared.read_text(encoding='utf-8')))
