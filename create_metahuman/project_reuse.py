"""Transactional accepted-part reuse with matching anatomy and local maps."""
import json
import shutil
from contextlib import ExitStack, contextmanager
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace

import bpy
from mathutils import Matrix

from . import (
    character_project,
    map_provenance,
    reuse_details,
    reuse_wrinkles,
    sculpt_archive,
    textures,
)
from . import mesh_ops as m


def read(context, path, part):
    before = set(bpy.data.objects)
    collections = set(bpy.data.collections)
    retained = set()
    with ExitStack() as stack:
        source = path
        if bpy.data.filepath and path == Path(bpy.data.filepath).resolve():
            temporary = Path(stack.enter_context(TemporaryDirectory(prefix='cm-part-reuse-')))
            source = temporary / 'project.blend'
            shutil.copy2(path, source)
        with bpy.data.libraries.load(str(source), link=False) as (available, loaded):
            loaded.scenes = available.scenes
    try:
        candidates = []
        for scene in loaded.scenes:
            s, f = scene.cm_settings, scene.cm_features
            if part == 'HEAD' and s.part in {'HEAD', 'FULL_BODY'}:
                obj, reference = f.head_result or s.result, s.template
            elif part == 'BODY' and s.part in {'BODY', 'FULL_BODY'}:
                obj, reference = f.body_result or (s.result if s.part == 'BODY' else None), f.body_template or (s.template if s.part == 'BODY' else None)
            else:
                continue
            if obj and obj.type == 'MESH':
                candidates.append((scene, obj, reference))
        if len(candidates) > 1:
            matching = [item for item in candidates
                        if Path(bpy.path.abspath(item[0].cm_settings.project_dir)).resolve() == path.parent]
            candidates = matching or candidates
        if len(candidates) != 1:
            raise ValueError('Save a project identifying one accepted ' + part.lower() + ' before reusing it')
        scene, obj, reference = candidates[0]
        settings, features = scene.cm_settings, scene.cm_features
        if features.body_editing or features.full_sculpt or features.restore_editing or (part == 'HEAD' and features.expression_active):
            raise ValueError('Finish the saved character edit and return its head to Neutral before importing this part')
        m.check_mesh(obj, 'Saved ' + part.lower())
        m.check_mesh(reference, 'Saved template')
        if m.topology(obj) != m.topology(reference):
            raise ValueError('Saved result no longer matches its original template topology')
        accessories = {name: getattr(features, name) for name in ('eye_left', 'eye_right', 'teeth')} if part == 'HEAD' else {}
        with context.temp_override(scene=scene, view_layer=scene.view_layers[0]):
            details, current_details = reuse_details.inspect(context, scene, obj, part)
            wrinkle_state = reuse_wrinkles.inspect(context, scene, part, path)
        items = list(dict.fromkeys(item for item in [obj, reference, *accessories.values(), *details, *reuse_wrinkles.objects(wrinkle_state)] if item))
        worlds = {item: item.matrix_world.copy() for item in items}
        for item in items:
            if not item:
                continue
            m.check_mesh(item, 'Saved character part')
            if item.get('cm_individual_sculpt_part'):
                raise ValueError('Finish the individual sculpt in the saved project before reusing this part')
            if item.modifiers:
                raise ValueError('Apply live modifiers on the saved part before importing it')
            expression = item.data.shape_keys.key_blocks.get('Expression Check') if item.data.shape_keys else None
            if expression and expression.value:
                raise ValueError('Return the saved head to Neutral before importing it')
            world = worlds[item]
            item.parent = None
            item.constraints.clear()  # Preserve the stored transform, not external scene targets.
            scale = Matrix.Scale(scene.unit_settings.scale_length / context.scene.unit_settings.scale_length, 4)
            item.matrix_world = scale @ world
            if item in accessories.values():
                item['cm_accessory_template'] = reference
                if 'cm_follow_original' in item:
                    values = list(item['cm_follow_original'])
                    original = Matrix([values[i:i + 4] for i in range(0, 16, 4)])
                    item['cm_follow_original'] = [value for row in scale @ original for value in row]
            context.collection.objects.link(item)
            retained.add(item)
        reuse_details.rebind(context, obj, details, current_details)
        reuse_wrinkles.reparent(wrinkle_state)
        metadata = {'name': settings.character_name, 'folder': settings.project_dir,
                    'canonical': settings.canonical_order, 'create': settings.create_character,
                    'front': features.front_axis, 'full_head': features.full_head,
                    'detail': details[0] if details else None, 'details': details, 'wrinkles': wrinkle_state}
    finally:
        for scene in loaded.scenes:
            bpy.data.scenes.remove(scene)
        for other in set(bpy.data.objects) - before - retained:
            bpy.data.objects.remove(other, do_unlink=True)
        for collection in set(bpy.data.collections) - collections:
            bpy.data.collections.remove(collection, do_unlink=True)
    for key in ('cm_source_signature', 'cm_template_signature'):
        if key in obj:
            del obj[key]
    obj['cm_role'], obj['cm_part'] = 'RESULT', part
    obj['cm_neck_attached'] = False
    return obj, reference, accessories, metadata


@contextmanager
def maps(scene, obj, part, metadata, path):
    s = scene.cm_settings
    project = Path(bpy.path.abspath(s.project_dir)).resolve()
    project.mkdir(parents=True, exist_ok=True)
    for folder in ('Maps', 'ReferenceMaps', 'Expressions'):
        target = project / folder
        if target.exists() and target.resolve() != target:
            raise ValueError('Part reuse needs map and expression folders inside this project, without linked folders')
    with TemporaryDirectory(prefix='.part-reuse-', dir=project, ignore_cleanup_errors=True) as directory:
        stage = Path(directory)
        source = path.parent / 'Maps'
        if not source.is_dir():
            source = Path(bpy.path.abspath(metadata['folder'])) / 'Maps'
        settings = SimpleNamespace(project_dir=str(stage), character_name=s.character_name)
        images = {}
        textures.adopt_project_maps(settings, obj, part, metadata['name'], source, _images=images)
        for layer in [*metadata['details'], *reuse_wrinkles.objects(metadata['wrinkles'])]:
            textures.adopt_project_maps(settings, layer, part, metadata['name'], source, _images=images, _copy_files=False)
        normal = obj.get('cm_reference_normal_image')
        if isinstance(normal, bpy.types.Image):
            image = normal.copy()
            target = stage / 'ReferenceMaps' / (s.character_name + ('_Body' if part == 'BODY' else '') + '_SOURCE_NORMAL.png')
            target.parent.mkdir(exist_ok=True)
            image.filepath_raw = str(target)
            image.file_format = 'PNG'
            image.save()
            image.pack()
            obj['cm_reference_normal_image'] = image
            obj['cm_reference_normal'] = str(project / target.relative_to(stage))
        else:
            obj['cm_reference_normal'] = ''
        pending = {p.relative_to(stage): p for folder in ('Maps', 'ReferenceMaps')
                   for p in (stage / folder).glob('*.png')}
        pending.update(reuse_wrinkles.resources(stage, project, s.character_name, metadata['wrinkles']))
        prefix = s.character_name + ('_Body' if part == 'BODY' else '') + '_'
        kinds = ['ALBEDO', 'NORMAL', 'CAVITY', 'AO', 'ROUGHNESS']
        if part == 'HEAD':
            kinds += ['WM1', 'WM2', 'WM3', 'CM1', 'CM2', 'CM3']
        affected = set(pending) | {Path('Maps') / (prefix + kind + '.png') for kind in kinds}
        affected.add(Path('Maps/.create-metahuman-bakes.json'))
        previous = {}
        for relative in affected:
            destination = project / relative
            backup = stage / 'previous' / relative
            if destination.resolve() != destination.absolute() or destination.is_symlink() or destination.exists() and not destination.is_file():
                raise ValueError('A reused map destination must be a regular file: ' + destination.name)
            if destination.is_file():
                backup.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(destination, backup)
                previous[relative] = backup
            else:
                previous[relative] = None
        try:
            for relative in affected - {Path('Maps/.create-metahuman-bakes.json')}:
                destination = project / relative
                if relative in pending:
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    pending[relative].replace(destination)
                else:
                    destination.unlink(missing_ok=True)
            for image in bpy.data.images:
                filename = Path(bpy.path.abspath(image.filepath)).resolve() if image.filepath else None
                if filename and filename.is_relative_to(stage) and filename.relative_to(stage) in pending:
                    image.filepath_raw = str(project / filename.relative_to(stage))
            yield {p.name for p in pending if p.parent == Path('Maps')}
        except BaseException:
            for relative, backup in previous.items():
                destination = project / relative
                if backup is None:
                    destination.unlink(missing_ok=True)
                else:
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(backup, destination)
            raise


def reset(scene, part):
    s, f, g = scene.cm_settings, scene.cm_features, scene.cm_groom
    if part == 'HEAD':
        for name in ('wm1', 'wm2', 'wm3', 'wm1_mask', 'wm2_mask', 'wm3_mask', 'generated_head',
                     'generated_local_skin_image', 'head_detail_previous', 'proxy'):
            setattr(f, name, None)
    else:
        f.body_detail_previous = None
    for name in ('character_previous_head_material', 'character_previous_body_material',
                 'character_previous_head_normal', 'character_previous_body_normal',
                 'character_previous_skin_image', 'character_previous_generated_image',
                 'character_previous_head', 'character_previous_body', 'character_applied_skin_image'):
        setattr(f, name, None)
    f.character_skin_previous, f.character_skin_applied = '', 'REFERENCE'
    for name in ('native_web_capture_receipt','native_web_capture_result','native_web_source','native_web_appearance','native_web_binding','native_web_job','native_web_output','native_web_status','unreal_push_status', 'unreal_push_receipt', 'unreal_character', 'unreal_status_report','unreal_artist_snapshot','unreal_artist_receipt','unreal_artist_review',
                 'unreal_character_project', 'unreal_build_receipt', 'unreal_build_report', 'delivery_report'):
        setattr(f, name, '')
    f.unreal_assembly_ready = False
    for name in ('native_web_legacy_frame','native_web_legacy_textures','native_web_legacy_grooms'):setattr(f,name,False)
    for name in ('unreal_target', 'exchange_status', 'exchange_receipt', 'exchange_result', 'exchange_applied'):
        setattr(g, name, '')
    for pair in s.landmarks:
        for marker in (pair.source, pair.template):
            if marker and marker.type == 'EMPTY' and marker.get('cm_landmark'):
                marker.hide_set(True)
                marker.hide_render = True
    s.landmarks.clear()  # Keep old marker objects for Undo, never delete artist data here.
    f.landmarks_reviewed = False
    roles = {'HEAD_DETAIL', 'WM1', 'WM2', 'WM3'} if part == 'HEAD' else {'BODY_DETAIL'}
    table = sculpt_archive.returned(scene)
    scene[sculpt_archive.RETURNED] = json.dumps({k: v for k, v in table.items() if v.get('role') not in roles})
    if sculpt_archive.BODY_CHECKPOINT in scene:
        del scene[sculpt_archive.BODY_CHECKPOINT]


def discard_unused(before):
    while True:
        unused = [item for name, old in before.items() for item in getattr(bpy.data, name)
                  if item not in old and item.users <= int(item.use_fake_user)]
        if not unused:
            return
        bpy.data.batch_remove(ids=unused)


def import_part(context, s, f, filename, part):
    character_project.preflight(context, s, f)
    path = Path(filename).resolve()
    if part not in {'HEAD', 'BODY'} or not path.is_file():
        raise ValueError('Choose a saved head/body project')
    before = character_project._ids()
    with character_project.transaction(context):
        from . import guided, setup_paths
        guided.project_folder(s)
        setup_paths.resolve(f)
        obj, reference, accessories, metadata = read(context, path, part)
        with maps(context.scene, obj, part, metadata, path) as names:
            was_body = s.part == 'BODY'
            if part == 'HEAD':
                if was_body:
                    f.body_result, f.body_template, f.body_detail = f.body_result or s.result, f.body_template or s.template, f.body_detail or s.detail
                old = [f.head_result or (s.result if not was_body else None), s.detail if not was_body else None,
                       f.wm1, f.wm2, f.wm3, f.eye_left, f.eye_right, f.teeth, s.source, f.proxy]
                f.head_result, s.result, s.template, s.detail, s.source = obj, obj, reference, metadata['detail'], None
                for name, value in accessories.items():
                    setattr(f, name, value)
                f.front_axis, f.full_head = metadata['front'], metadata['full_head']
            else:
                old = [f.body_result or (s.result if was_body else None), f.body_detail or (s.detail if was_body else None)]
                if not was_body:
                    f.head_result = f.head_result or s.result
                f.body_result, f.body_template, f.body_detail = obj, reference, metadata['detail']
                if was_body or not f.head_result:
                    f.head_result = None
                    s.result, s.template, s.detail, s.source, s.part = obj, reference, metadata['detail'], None, 'BODY'
                    f.front_axis = metadata['front']
            retired = {item for item in old if item}
            owned = [child for item in retired for child in item.children_recursive
                     if child.get('cm_wrinkle_owner') in retired]
            for previous in dict.fromkeys(item for item in [*old, *owned] if item and item not in (obj, reference)):
                previous.hide_set(True)
                previous.hide_render = True
            reference.hide_set(True)
            reference.hide_render = True
            for item in (obj, *accessories.values()):
                if item:
                    item.hide_set(False)
                    item.hide_render = False
            if f.head_result and f.body_result:
                s.part = 'FULL_BODY'
            other = f.body_result if part == 'HEAD' else f.head_result
            if other and metadata['canonical'] != s.canonical_order and not (obj.data.uv_layers and other.data.uv_layers):
                raise ValueError('Mixed canonical/UV part reuse requires compatible UVs on both parts')
            s.canonical_order = False if other and obj.data.uv_layers and other.data.uv_layers else metadata['canonical']
            if not other:
                s.create_character = metadata['create']
            from . import native_body
            if not f.is_property_set('body_backend') and native_body.available(f):
                f.body_backend = 'NATIVE'
            reset(context.scene, part)
            reuse_wrinkles.assign(context, metadata['wrinkles'])
            detail = metadata['detail']
            if detail:
                previous = detail.get('cm_detail_previous')
                setattr(f, 'head_detail_previous' if part == 'HEAD' else 'body_detail_previous', previous if isinstance(previous, bpy.types.Object) else None)
                detail.hide_set(False)
                detail.hide_render = False
                obj.hide_set(True)
                obj.hide_render = True
            keys = ['FINAL_' + part] + (['WM1', 'WM2', 'WM3', 'CM1', 'CM2', 'CM3'] if part == 'HEAD' else [])
            ledger = map_provenance.load(context.scene)
            invalid = json.loads(context.scene.get(map_provenance.INVALIDATED, '{}'))
            for key in keys:
                identity = map_provenance.identity(context.scene, key)
                ledger['entries'].pop(identity, None)
                invalid.pop(identity, None)
            map_provenance.manifest(context.scene).write_text(json.dumps(ledger), encoding='utf-8')
            context.scene[map_provenance.INVALIDATED] = json.dumps(invalid)
            map_provenance.record(context.scene, map_provenance.basis_for_files(context.scene, names), origin='PROJECT_REUSE', outputs=names)
            m.activate(context, detail or obj)
            discard_unused(before)
            s.stage, f.ui_stage = 'REFINE', 'REFINE'
            s.status = 'Accepted ' + part.lower() + ' imported with matching anatomy, maps and available sculpt detail. Attach Head to Body to match the neck.'
    return obj


def focus(context):
    if bpy.app.background or not context.window:
        return
    from . import compare_views, guided
    window, scene = context.window, context.scene
    head, body = scene.cm_settings.result, scene.cm_features.body_result

    def finish():
        try:
            if window.scene != scene or scene.cm_settings.result != head or scene.cm_features.body_result != body:
                return None
            with bpy.context.temp_override(window=window):
                session = compare_views.recover(window)
                if session and not scene.cm_settings.source:
                    compare_views.close_views(bpy.context)
                    return .1
                if session:
                    compare_views.frame(bpy.context, session)
                else:
                    f = scene.cm_features
                    s = scene.cm_settings
                    guided.focus(bpy.context, [o for o in (s.detail or head, f.body_detail or body, f.eye_left, f.eye_right, f.teeth) if o], f.front_axis)
                    for area in window.screen.areas:
                        if area.type == 'VIEW_3D':
                            area.spaces.active.show_region_ui = True
            return None
        except (ReferenceError, RuntimeError, ValueError):
            return None

    bpy.app.timers.register(finish, first_interval=.1)
