"""Publish a fresh character without overwriting the preceding character."""
import re
import shutil
from contextlib import contextmanager
from pathlib import Path

import bpy

from . import mesh_ops as m

PROJECT_ROOT = 'cm_character_project_root'


def preflight(context, s, f, *, allow_photo_fit=False):
    from . import eye_fit_jobs
    if eye_fit_jobs.running():raise ValueError('Finish or cancel the source surface fit first')
    from . import region_projection
    if region_projection.running():raise ValueError('Finish or cancel regional surface projection first')
    from . import skin_reconstruction
    if skin_reconstruction.running():
        raise ValueError('Apply or discard the skin reconstruction before another character operation')
    from . import (
        auto_wrap_jobs,
        bake_jobs,
        detail_jobs,
        detail_mask,
        expressions,
        groom_jobs,
        lookdev,
        native_body,
        paint_tools,
        photo_fit_jobs,
        photo_review,
        scan_jobs,
        sculpt_session,
        skin_library,
        skin_native_jobs,
        unreal_jobs,
        wrinkle_jobs,
    )
    if photo_fit_jobs.running() and not allow_photo_fit:
        raise ValueError('Apply or discard the current photo fit before starting another character')
    if photo_review._VIEW:
        raise ValueError('Return to 3D from the photo overlay before starting another character operation')
    if (not s.character_name.strip() or s.character_name != s.character_name.strip()
            or re.search(r'[<>:"/\\|?*\x00-\x1f]', s.character_name)
            or s.character_name.endswith('.')
            or re.fullmatch(r'(CON|PRN|AUX|NUL|COM[1-9]|LPT[1-9])(\..*)?', s.character_name, re.IGNORECASE)):
        raise ValueError('Use a character name without path characters or trailing spaces/periods')
    if (context.mode != 'OBJECT' or f.body_editing or f.full_sculpt or f.restore_editing
            or f.expression_active or context.scene.cm_groom.scalp_editing
            or f.sculpt_session_role == 'AUTHOR' or sculpt_session.pending()
            or sculpt_session._SESSION):
        raise ValueError('Finish the active edit or sculpt session and return to Neutral before starting a character')
    if paint_tools.running() or detail_mask.running(context.scene):
        raise ValueError('Apply or cancel the current paint or mask before starting a character')
    if (any(module.running() for module in (auto_wrap_jobs, scan_jobs, bake_jobs,
            detail_jobs, wrinkle_jobs, unreal_jobs, groom_jobs, skin_library,
            skin_native_jobs, expressions)) or native_body._JOB or lookdev._JOB
            or bpy.app.is_job_running('RENDER') or bpy.app.is_job_running('OBJECT_BAKE')):
        raise ValueError('Finish or cancel the current character operation before starting another character')
    if not s.project_dir.strip():
        raise ValueError('Choose a project folder before starting a character')


@contextmanager
def output_folder(s):
    """Reserve an empty destination. Existing project contents are never reused."""
    root = Path(bpy.path.abspath(s.project_dir)).resolve()
    previous_root = s.id_data.get(PROJECT_ROOT)
    if previous_root:
        previous_root = Path(previous_root).resolve()
        if root.parent == previous_root / 'Characters':
            root = previous_root
    if root.exists() and not root.is_dir():
        raise ValueError('The project folder is a file; choose a folder')
    destination = root
    if root.exists() and any(root.iterdir()):
        parent = root / 'Characters'
        parent.mkdir(exist_ok=True)
        if parent.resolve() != parent:
            raise ValueError('The Characters folder is linked elsewhere; choose a project folder without linked subfolders')
        for index in range(1, 10001):
            destination = parent / (s.character_name + (f'-{index:03d}' if index > 1 else ''))
            try:
                destination.mkdir()
                break
            except FileExistsError:
                continue
        else:
            raise ValueError('Choose a new project folder; all character folder suffixes are occupied')
        owned = True
    else:
        owned = not root.exists()
        root.mkdir(parents=True, exist_ok=True)
    s.project_dir = str(destination)
    s.id_data[PROJECT_ROOT] = str(root)
    try:
        yield destination
    except BaseException:
        # This transaction only writes these subdirectories to a previously
        # empty destination. Do not delete unrelated files added by the artist.
        for name in ('Maps', 'ReferenceMaps', 'Appearance'):
            path = destination / name
            if path.is_dir() and not path.is_symlink() and path.resolve().parent == destination:
                shutil.rmtree(path)
        if owned:
            try:
                destination.rmdir()
            except OSError:
                pass
        raise


def _properties(group):
    """Save raw RNA storage so rollback does not trigger geometry callbacks."""
    # Blender can represent a null ID pointer as an internal IDPropertyGroup.
    # Snapshot its resolved ID/None, never that internal wrapper.
    return {p.identifier: (getattr(group, p.identifier) if p.type == 'POINTER'
                           else _stored(group.get(p.identifier)))
            for p in group.bl_rna.properties
            if not p.is_readonly and p.type != 'COLLECTION' and p.identifier != 'rna_type'}


def _stored(value):
    # Copy ID-property containers before their owning property can be cleared.
    if hasattr(value, 'to_dict'):
        return value.to_dict()
    if hasattr(value, 'to_list'):
        return value.to_list()
    return value


def _restore(group, values):
    for name, value in values.items():
        if value is None:
            group.property_unset(name)
        else:
            group[name] = value


def _ids():
    # Appending a Blender library can bring dependencies beyond objects/meshes.
    return {p.identifier: set(getattr(bpy.data, p.identifier))
            for p in bpy.data.bl_rna.properties if p.type == 'COLLECTION'}


def _clear_identity(scene):
    s, f, g = scene.cm_settings, scene.cm_features, scene.cm_groom
    # Stop accepted-head callbacks from retargeting the previous live groom.
    for name in ('main_scalp', 'groom', 'scalp', 'scalp_hair', 'scalp_brows', 'scalp_mustache', 'scalp_beard'):
        setattr(g, name, None)
    for name in ('unreal_target', 'exchange_status', 'exchange_receipt', 'exchange_result', 'exchange_applied'):
        setattr(g, name, '')
    for name in ('wm1', 'wm2', 'wm3', 'generated_head', 'head_detail_previous', 'body_detail_previous',
                 'proxy', 'generated_local_skin_image', 'wm1_mask', 'wm2_mask', 'wm3_mask',
                 'character_applied_skin_image', 'character_previous_head_material',
                 'character_previous_body_material', 'character_previous_head_normal',
                 'character_previous_body_normal', 'character_previous_skin_image',
                 'character_previous_generated_image', 'character_previous_head', 'character_previous_body'):
        setattr(f, name, None)
    for name in ('unreal_revision_source','unreal_revision_project','unreal_revision_receipt','unreal_revision_result','unreal_revision_character','native_web_capture_receipt','native_web_capture_result','native_web_source','native_web_appearance','native_web_binding','native_web_job','native_web_output','native_web_status','unreal_push_status', 'unreal_push_receipt', 'unreal_texture_receipt', 'unreal_texture_result', 'unreal_eye_receipt', 'unreal_eye_result', 'unreal_character', 'unreal_status_report','unreal_artist_snapshot','unreal_artist_receipt','unreal_artist_review',
                 'unreal_character_project', 'unreal_build_receipt', 'unreal_build_report', 'delivery_report',
                 'sculpt_session_path', 'sculpt_session_status', 'detail_update_status', 'map_bake_status', 'photo_fit_status'):
        setattr(f, name, '')
    f.unreal_assembly_ready = False
    f.unreal_revision_skin_color = False
    f.unreal_revision_update_scope = 'DETAIL'
    f.unreal_wrinkle_input_encoding='FULL';f.unreal_color_input_encoding='FULL'
    f.unreal_wrinkle_neutral_mode='AUTO';f.unreal_wrinkle_neutral_image=None
    for name in ('native_web_legacy_frame','native_web_legacy_textures','native_web_legacy_grooms'):setattr(f,name,False)
    f.landmarks_reviewed = False
    f.detail_recovery_manual = False
    s.last_error = 0
    s.landmarks.clear()  # Old marker objects belong to the preceding character.
    from . import map_provenance, sculpt_archive
    for key in (map_provenance.INVALIDATED, sculpt_archive.RETURNED, sculpt_archive.BODY_CHECKPOINT):
        if key in scene:
            del scene[key]


def start(context, s, f, create):
    preflight(context, s, f)
    with transaction(context):
        with output_folder(s):
            _clear_identity(context.scene)
            create(context, s, f)


@contextmanager
def transaction(context):
    """Restore settings, editor state and appended IDs after failed publication."""
    s,f=context.scene.cm_settings,context.scene.cm_features
    scene, g = context.scene, context.scene.cm_groom
    from . import groom_follow
    grooming = groom_follow.capture(scene)
    groups = [(group, _properties(group)) for group in (s, f, g)]
    landmarks = [(p.name, p.source, p.template) for p in s.landmarks]
    controls = [{name: getattr(item, name) for name in ('key', 'category', 'label', 'value')}
                for item in f.body_controls]
    from . import map_provenance, sculpt_archive
    flags = {key: _stored(scene.get(key)) for key in
             (map_provenance.INVALIDATED, sculpt_archive.RETURNED, sculpt_archive.BODY_CHECKPOINT, PROJECT_ROOT)}
    before = _ids()
    render_hidden = [(obj, obj.hide_render) for obj in scene.objects]
    with m.selection_state(context, on_error=True):
        try:
            yield
        except BaseException:
            # Restore references before disposing newly appended datablocks.
            for group, values in groups:
                _restore(group, values)
            groom_follow.restore(grooming)
            s.landmarks.clear()
            for name, source, template in landmarks:
                item = s.landmarks.add()
                item.name, item.source, item.template = name, source, template
            f.body_controls.clear()
            for values in controls:
                item = f.body_controls.add()
                for name, value in values.items():
                    if name == 'value':
                        item[name] = value  # Do not run the body-shaping callback.
                    else:
                        setattr(item, name, value)
            for key, value in flags.items():
                if value is None:
                    if key in scene:
                        del scene[key]
                else:
                    scene[key] = value
            for obj, hidden in render_hidden:
                obj.hide_render = hidden
            context.view_layer.update()
            added = {item for name, previous in before.items()
                     for item in getattr(bpy.data, name) if item not in previous}
            if added:
                bpy.data.batch_remove(ids=added)
                context.view_layer.update()
            raise
