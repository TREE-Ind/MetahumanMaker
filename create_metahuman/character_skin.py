"""Apply matching licensed head/body skins without replacing character geometry."""
from pathlib import Path
import json
import shutil
import uuid
import bpy
from . import skin_library, skin_projection, map_provenance

SLOTS = {'head_material': 'character_previous_head_material', 'body_material': 'character_previous_body_material',
         'head_normal': 'character_previous_head_normal', 'body_normal': 'character_previous_body_normal',
         'image': 'character_previous_skin_image', 'generated_image': 'character_previous_generated_image',
         'head': 'character_previous_head', 'body': 'character_previous_body'}


def idle(context):
    from . import paint_tools
    if paint_tools.running():
        raise ValueError('Apply or cancel Smart Paint before changing character skin')
    from . import (bake_jobs, detail_jobs, wrinkle_jobs, native_body, unreal_jobs,
                   groom_jobs, skin_native_jobs, sculpt_session, photo_fit_jobs, auto_wrap_jobs)
    s, f = context.scene.cm_settings, context.scene.cm_features
    if context.mode != 'OBJECT' or f.body_editing or f.full_sculpt or f.restore_editing or sculpt_session.pending():
        raise ValueError('Finish the active edit before changing character skin')
    if any(module.running() for module in (bake_jobs, detail_jobs, wrinkle_jobs, unreal_jobs, groom_jobs,
                                          skin_library, skin_native_jobs, photo_fit_jobs,
                                          auto_wrap_jobs)) or native_body._JOB:
        raise ValueError('Wait for the current character operation before changing skin')
    if s.part != 'FULL_BODY' or not s.result or not f.body_result:
        raise ValueError('Start or load a full character first')
    if any(len(obj.data.materials) != 1 or not obj.data.materials[0] for obj in (s.result, f.body_result)):
        raise ValueError('The head and body must each have one skin material; eyes and teeth stay separate')
    if not s.character_name or any(c in s.character_name for c in '/\\:'):
        raise ValueError('Use a character name without path separators')


def _files(scene):
    name = scene.cm_settings.character_name
    return [name + ('_Body' if role == 'BODY' else '') + '_' + kind + '.png'
            for role in ('HEAD', 'BODY') for kind in ('ALBEDO', 'NORMAL', 'CAVITY', 'ROUGHNESS')]


def _snapshot(context, folder):
    s, f = context.scene.cm_settings, context.scene.cm_features
    objects = {'head': s.result, 'body': f.body_result}
    state = {'image': f.character_applied_skin_image, 'generated_image': f.generated_local_skin_image, **objects}
    for role, obj in objects.items():
        state[role + '_material'] = obj.data.materials[0]
        state[role + '_normal'] = obj.get('cm_reference_normal_image')
        for node in obj.data.materials[0].node_tree.nodes if obj.data.materials[0].use_nodes else ():
            if node.type == 'TEX_IMAGE' and node.image:
                node.image.pack()  # Preserve unsaved artist paint for Restore Previous Skin.
    metadata = {'project': str(Path(bpy.path.abspath(s.project_dir)).resolve()), 'character': s.character_name,
                'applied': f.character_skin_applied, 'generated_skin': f.generated_skin,
                'generated_preset': f.generated_skin_preset, 'generated_tone': f.generated_skin_tone,
                'invalidated': context.scene.get(map_provenance.INVALIDATED), 'files': {},
                'normal_paths': {role: obj.get('cm_reference_normal', '') for role, obj in objects.items()}}
    folder.mkdir(parents=True, exist_ok=False)
    for name in [*_files(context.scene), '.create-metahuman-bakes.json']:
        source = Path(metadata['project']) / 'Maps' / name
        target = folder / name
        if source.is_file():
            shutil.copy2(source, target)
            metadata['files'][name] = str(target)
        else:
            metadata['files'][name] = None
    state['metadata'] = metadata
    return state


def _replace_files(project, files):
    folder = project / 'Maps'
    folder.mkdir(parents=True, exist_ok=True)
    for name, source in files.items():
        if Path(name).name != name:
            raise ValueError('Invalid skin map filename')
        destination = folder / name
        if source is None:
            destination.unlink(missing_ok=True)
        else:
            pending = folder / (name + '.skin-tmp')
            shutil.copy2(source, pending)
            pending.replace(destination)


def _bind(context, state):
    s, f = context.scene.cm_settings, context.scene.cm_features
    for role, obj in (('head', s.result), ('body', f.body_result)):
        obj.data.materials[0] = state[role + '_material']
        if state[role + '_normal']:
            obj['cm_reference_normal_image'] = state[role + '_normal']
        elif 'cm_reference_normal_image' in obj:
            del obj['cm_reference_normal_image']
        obj['cm_reference_normal'] = state['metadata']['normal_paths'][role]
    f.character_applied_skin_image = state['image']
    f.character_skin_image = state['image']
    f.character_skin_applied = state['metadata']['applied']
    f.character_skin_preset = f.character_skin_applied
    if s.source and s.source == f.generated_head:
        f.generated_local_skin_image = state['generated_image']
        f.generated_skin_preset = state['metadata']['generated_preset']
        f.generated_skin_tone = state['metadata']['generated_tone']
        f.generated_skin = state['metadata']['generated_skin']
        from . import head_model
        head_model.update_skin(f, context)
    invalidated = state['metadata']['invalidated']
    if invalidated is None:
        if map_provenance.INVALIDATED in context.scene:
            del context.scene[map_provenance.INVALIDATED]
    else:
        context.scene[map_provenance.INVALIDATED] = invalidated
    if f.show_lookdev:
        from . import lookdev
        lookdev.update(f, context)


def _remember(f, state):
    previous_images = {image.as_pointer(): image for image in
                       (f.character_previous_skin_image, f.character_previous_generated_image) if image}
    for key, attribute in SLOTS.items():
        setattr(f, attribute, state[key])
    f.character_skin_previous = json.dumps(state['metadata'])
    for previous in previous_images.values():
        if previous and previous != state['image'] and previous != state['generated_image']:
            skin_library.release_unused(previous)


def _references(context):
    f = context.scene.cm_features
    if f.character_skin_preset == 'REFERENCE':
        from . import guided
        with bpy.data.libraries.load(str(guided.character_library()), link=False) as (available, loaded):
            loaded.objects = available.objects
        references = {}
        try:
            for role in ('head', 'body'):
                obj = next(obj for obj in loaded.objects if obj and obj.get('cm_library_role') == role)
                maps = {node.get('cm_map_kind'): node.image for node in obj.data.materials[0].node_tree.nodes
                        if node.type == 'TEX_IMAGE' and node.image and node.get('cm_map_kind')}
                references[role] = {'uv': skin_projection.uv_layout_signature(obj), 'maps': maps}
            return None, references
        finally:
            for obj in loaded.objects:
                if obj and not obj.users_collection:
                    bpy.data.objects.remove(obj, do_unlink=True)
    image = f.character_skin_image
    if not image or image.get('cm_skin_preset') != f.character_skin_preset:
        image = skin_library.load_cached_image(skin_library.cache_root(f), f.character_skin_preset)
    if not image:
        raise ValueError('The selected character skin is unavailable; create or select a prepared skin')
    if (not image.get('cm_native_color_eligible', True) or image.is_dirty or
        image.get('cm_gnm_pixels_hash') != skin_library.pixel_digest(image)):
        raise ValueError('This skin has GNM paint edits; wrap and transfer its painted albedo before using native full-character maps')
    references = {}
    for role, key in [('head', 'cm_native_skin_reference'), ('body', 'cm_body_skin_reference')]:
        albedo = image.get(key)
        if not albedo:
            raise ValueError('This preset has no matching full-character skin; create a new skin')
        references[role] = {'uv': albedo.get('cm_native_uv_layout'), 'maps': {'ALBEDO': albedo,
            'NORMAL': albedo.get('cm_native_normal_reference'), 'CAVITY': albedo.get('cm_native_cavity_reference')}}
        if albedo.get('cm_native_roughness_reference'):
            references[role]['maps']['ROUGHNESS'] = albedo['cm_native_roughness_reference']
    return image, references


def _material(original, images):
    material = original.copy()
    if not material.use_nodes:
        material.use_nodes = True
    nodes, links = material.node_tree.nodes, material.node_tree.links
    assigned = set()
    for node in nodes:
        kind = node.get('cm_map_kind')
        if node.type == 'TEX_IMAGE' and kind in images:
            node.image = images[kind]
            assigned.add(kind)
    if not {'ALBEDO', 'NORMAL'}.issubset(assigned):
        # Standard compatible skin shader for meshes with an untagged preview.
        nodes.clear()
        shader = nodes.new('ShaderNodeBsdfPrincipled')
        shader.inputs['Roughness'].default_value = .48
        shader.inputs['Subsurface Weight'].default_value = .06
        output = nodes.new('ShaderNodeOutputMaterial')
        links.new(shader.outputs['BSDF'], output.inputs['Surface'])
        tex = {}
        for kind, image in images.items():
            tex[kind] = nodes.new('ShaderNodeTexImage')
            tex[kind].image = image
            tex[kind]['cm_map_kind'] = kind
        links.new(tex['ALBEDO'].outputs['Color'], shader.inputs['Base Color'])
        separate = nodes.new('ShaderNodeSeparateColor')
        combine = nodes.new('ShaderNodeCombineColor')
        inverse = nodes.new('ShaderNodeMath')
        inverse.operation = 'SUBTRACT'
        inverse.inputs[0].default_value = 1
        links.new(tex['NORMAL'].outputs['Color'], separate.inputs[0])
        links.new(separate.outputs['Red'], combine.inputs['Red'])
        links.new(separate.outputs['Green'], inverse.inputs[1])
        links.new(inverse.outputs[0], combine.inputs['Green'])
        links.new(separate.outputs['Blue'], combine.inputs['Blue'])
        normal = nodes.new('ShaderNodeNormalMap')
        links.new(combine.outputs[0], normal.inputs['Color'])
        links.new(normal.outputs[0], shader.inputs['Normal'])
    from . import lookdev
    lookdev.bind_roughness(material, images.get('ROUGHNESS'))
    return material


def apply(context):
    idle(context)
    s, f = context.scene.cm_settings, context.scene.cm_features
    image, references = _references(context)
    for role, obj in (('head', s.result), ('body', f.body_result)):
        if references[role]['uv'] != skin_projection.uv_layout_signature(obj):
            raise ValueError('The selected skin does not match the ' + role + ' UV layout')
        for source in references[role]['maps'].values():
            if not isinstance(source, bpy.types.Image):
                raise ValueError('The selected skin is missing a required map')
            source.pack()
    project = Path(bpy.path.abspath(s.project_dir)).resolve()
    if shutil.disk_usage(project if project.exists() else project.parent).free < 2 * 1024**3:
        raise ValueError('Changing skin needs a 2 GiB disk reserve')
    folder = project / 'Appearance' / uuid.uuid4().hex[:12]
    folder.mkdir(parents=True, exist_ok=False)
    before = _snapshot(context, folder / 'previous')
    incoming = dict(before, metadata=dict(before['metadata']))
    incoming['image'] = image
    incoming['metadata']['applied'] = f.character_skin_preset
    incoming['metadata']['normal_paths'] = {}
    files = {}
    staged = folder / 'new'
    staged.mkdir()
    try:
        for role in ('head', 'body'):
            images = {}
            for kind, source in references[role]['maps'].items():
                name = s.character_name + ('_Body' if role == 'body' else '') + '_' + kind + '.png'
                path = staged / name
                skin_library.save_image_copy(source, path, opaque=True)
                prepared = bpy.data.images.load(str(path), check_existing=False)
                if kind != 'ALBEDO':
                    prepared.colorspace_settings.name = 'Non-Color'
                prepared.pack()
                prepared.filepath_raw = str(project / 'Maps' / name)
                images[kind] = prepared
                files[name] = str(path)
            incoming[role + '_material'] = _material(before[role + '_material'], images)
            incoming[role + '_normal'] = references[role]['maps']['NORMAL']
            incoming['metadata']['normal_paths'][role] = references[role]['maps']['NORMAL'].filepath
        if image and s.source and s.source == f.generated_head:
            incoming['generated_image'] = image
            incoming['metadata'].update(generated_skin='LOCAL', generated_preset=f.character_skin_preset, generated_tone=0)
        _replace_files(project, files)
        # Applying a legacy three-map skin must not retain a newer skin's
        # roughness on disk or in the shader.
        for role in ('head', 'body'):
            name = s.character_name + ('_Body' if role == 'body' else '') + '_ROUGHNESS.png'
            if name not in files:
                (project / 'Maps' / name).unlink(missing_ok=True)
        _bind(context, incoming)
        from .textures import skin_part
        for role in ('HEAD', 'BODY'):
            with skin_part(s, f, role):
                for kind in ('NORMAL', 'CAVITY'):
                    name = s.character_name + '_' + kind + '.png'
                    if (project / 'Maps' / name).is_file():
                        map_provenance.record_single(context.scene, project / 'Maps' / name, 'REFERENCE')
        # Base maps cannot certify existing sculpt or animated-detail outputs.
        keys = ['WM1', 'WM2', 'WM3', 'CM1', 'CM2', 'CM3']
        if s.detail:
            keys.append('FINAL_HEAD')
        if f.body_detail:
            keys.append('FINAL_BODY')
        map_provenance.invalidate_keys(context.scene, keys)
        _remember(f, before)
    except Exception:
        _replace_files(project, before['metadata']['files'])
        _bind(context, before)
        raise
    # Publication has committed. Cleanup failure must not roll back a valid skin.
    try:
        for path in staged.glob('*.png'):
            path.unlink()
        staged.rmdir()
    except OSError:
        pass
    label = image.get('cm_skin_label', 'Character Skin') if image else 'Original Local Skin'
    s.status = label + ' applied to head and body. Existing sculpt detail may need final maps rebaked.'
    return folder


def restore(context):
    idle(context)
    s, f = context.scene.cm_settings, context.scene.cm_features
    if not f.character_skin_previous:
        raise ValueError('There is no previous character skin in this project')
    state = {key: getattr(f, attribute) for key, attribute in SLOTS.items()}
    metadata = json.loads(f.character_skin_previous)
    state['metadata'] = metadata
    project = Path(bpy.path.abspath(s.project_dir)).resolve()
    if metadata['project'] != str(project) or metadata['character'] != s.character_name or state['head'] != s.result or state['body'] != f.body_result:
        raise ValueError('The previous skin belongs to a different character or project')
    for path in metadata['files'].values():
        if path and (not Path(path).resolve().is_relative_to(project / 'Appearance') or not Path(path).is_file()):
            raise ValueError('The previous skin files are unavailable')
    folder = project / 'Appearance' / uuid.uuid4().hex[:12]
    before = _snapshot(context, folder)
    try:
        _replace_files(project, metadata['files'])
        _bind(context, state)
        _remember(f, before)
    except Exception:
        _replace_files(project, before['metadata']['files'])
        _bind(context, before)
        raise
    s.status = 'Previous head and body skin restored; character geometry retained.'


class CM_OT_character_skin(bpy.types.Operator):
    bl_idname = 'cm.character_skin'
    bl_label = 'Apply Character Skin'
    bl_description = 'Apply matching head/body maps and retain the previous skin for restoration'
    action: bpy.props.EnumProperty(items=[('APPLY', 'Apply Skin', ''), ('RESTORE', 'Restore Previous Skin', '')])

    def execute(self, context):
        try:
            (restore if self.action == 'RESTORE' else apply)(context)
            context.scene.cm_features.last_action_error = False
            return {'FINISHED'}
        except Exception as exc:
            context.scene.cm_features.last_action_error = True
            context.scene.cm_settings.status = str(exc)
            self.report({'ERROR'}, str(exc))
            return {'CANCELLED'}


def draw(layout, context, *, apply_button=True):
    f = context.scene.cm_features
    from . import skin_native_jobs, bake_jobs
    layout.prop(f, 'character_skin_preset', text='Skin')
    applied = f.character_applied_skin_image
    label = applied.get('cm_skin_label', 'Character Skin') if applied else 'Original Local Skin'
    layout.label(text='Applied: ' + label)
    row = layout.row(align=True)
    row.enabled = not (skin_library.running() or skin_native_jobs.running() or bake_jobs.running())
    row.operator('cm.generate_native_skin', text='Create Skin', icon='ADD').target = 'CHARACTER'
    if apply_button:
        row.operator('cm.character_skin', text='Reapply Skin' if f.character_skin_preset == f.character_skin_applied else 'Apply Skin').action = 'APPLY'
        if f.character_skin_previous:
            previous = f.character_previous_skin_image
            label = previous.get('cm_skin_label', 'Previous Skin') if previous else 'Original Local Skin'
            layout.operator('cm.character_skin', text='Restore: ' + label, icon='LOOP_BACK').action = 'RESTORE'


CLASSES = (CM_OT_character_skin,)
