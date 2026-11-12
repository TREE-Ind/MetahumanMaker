"""Cancellable local photo fitting, with explicit publication of a new source."""

import json
import hashlib
import os
import shutil
import subprocess
import time
import uuid
from pathlib import Path

import bpy
import numpy as np
from bpy.app.handlers import persistent
from bpy.props import StringProperty
from bpy_extras.io_utils import ImportHelper

from . import character_project, head_model, mesh_ops, setup_paths

_JOB = None


def running():
    return _JOB is not None


def redraw():
    for window in bpy.context.window_manager.windows:
        for area in window.screen.areas:
            area.tag_redraw()


def stamp(context):
    from .scan_jobs import geometry_stamp
    s, f = context.scene.cm_settings, context.scene.cm_features
    objects = (s.source, f.generated_head, s.result, f.body_result, s.template, f.body_template)
    models = Path(bpy.path.abspath(f.model_dir))
    resources = [(name, (models / name).stat().st_size, (models / name).stat().st_mtime_ns)
                 for name in ('gnm_head.npz', 'head_sparse_68.txt', 'face_landmarker.task')]
    return (s.project_dir, s.part, s.character_name, f.generated_head_height,
            str(models.resolve()), f.inference_python, resources,
            head_model.coefficients(f).tolist(), [(c.key, c.value) for c in f.face_controls],
            [(o.as_pointer(), geometry_stamp(context, o)) if o else None for o in objects])


def fit_stamp(context, multiview=False):
    signature = stamp(context)
    if not multiview:
        return signature
    from . import photo_color
    source = context.scene.cm_features.generated_head
    photos = [(-1, source['cm_photo_reference']), *photo_color.supporting_photos(source)]
    calibration = tuple(source.get(key) for key in ('cm_photo_fit_report', 'cm_identity', 'cm_gnm_model_scale',
                                                    'cm_head_placement'))
    return signature, calibration, [(index, image.name, hashlib.sha256(photo_color.packed_photo(image)).hexdigest(),
                        bool(image.get('cm_photo_generated_reference'))) for index, image in photos]


def start(context, filename=None, *, multiview=False):
    global _JOB
    if _JOB:
        raise ValueError('Apply or discard the current photo fit first')
    from . import photo_review
    photo_review.finish()
    s, f = context.scene.cm_settings, context.scene.cm_features
    character_project.preflight(context, s, f)
    if s.part == 'BODY':
        raise ValueError('Choose Head or Full Body before fitting a face photo')
    setup_paths.resolve(f)
    models = Path(bpy.path.abspath(f.model_dir))
    for name in ('gnm_head.npz', 'head_sparse_68.txt', 'face_landmarker.task'):
        if not (models / name).is_file():
            raise ValueError('Prepare the local photo-fitting models with tools/setup_models.py')
    source = None
    extra = {}
    if multiview:
        from . import photo_color
        obj = f.generated_head
        if not obj or not obj.get('cm_photo_fit_report') or not obj.get('cm_photo_reference'):
            raise ValueError('Fit the primary photograph and add supporting photos first')
        supports = photo_color.supporting_photos(obj)
        if not supports:
            raise ValueError('Add at least one supporting photo first')
        if s.source != obj or any(pair.source and pair.source.parent != obj for pair in s.landmarks):
            raise ValueError('Select the matching photo source and its landmarks before refining')
        coords = np.empty(len(obj.data.vertices) * 3, dtype=np.float32)
        obj.data.vertices.foreach_get('co', coords)
        if (obj.modifiers or obj.data.shape_keys
                or hashlib.sha256(coords.tobytes()).hexdigest() != obj.get('cm_gnm_generated_geometry')):
            raise ValueError('Use an unsculpted generated source; retain sculpt edits on the wrapped character')
        if not np.allclose(json.loads(obj['cm_identity']), head_model.coefficients(f), rtol=1e-6, atol=1e-6):
            raise ValueError('Update the generated source from its current controls before refining photos')
        photo_review.camera_for(context, obj)
        world = photo_color.world_landmarks(context, obj)
        inverse = np.asarray(obj.matrix_world.inverted())
        local = world @ inverse[:3, :3].T + inverse[:3, 3]
        raw = local[:, [0, 2, 1]] * [1., 1., -1.] / obj['cm_gnm_model_scale']
        extra = dict(initial_camera=json.loads(obj['cm_photo_fit_report'])['camera'], current_landmarks=raw.tolist())
    else:
        source = Path(bpy.path.abspath(filename))
        if not source.is_file() or source.stat().st_size > 40 * 1024**2:
            raise ValueError('Choose a readable face photo smaller than 40 MB')
    from . import resource_budget
    resource_budget.require('photo fitting',(1024 if multiview else 768)*resource_budget.MIB)
    folder = Path(bpy.path.abspath(s.project_dir)) / 'PhotoFits' / uuid.uuid4().hex[:12]
    folder.mkdir(parents=True)
    photo = folder / ('input' + (source.suffix.lower() if source else '.input'))
    if multiview:
        photo.write_bytes(photo_color.packed_photo(obj['cm_photo_reference']))
        extra['supporting_photos'] = []
        for index, image in supports:
            name = f'support-{index}.input'
            (folder / name).write_bytes(photo_color.packed_photo(image))
            extra['supporting_photos'].append(dict(file=name, name=image.name, slot=index,
                                                   generated=bool(image.get('cm_photo_generated_reference'))))
    else:
        shutil.copy2(source, photo)
    request = folder / 'request.json'
    initial = head_model.coefficients(f).tolist()
    request.write_text(json.dumps({'photo': str(photo.resolve()), 'models': str(models.resolve()),
                                  'initial': initial, 'parent': os.getpid(), **extra}), encoding='utf-8')
    environment = os.environ.copy()
    environment.update(OPENBLAS_NUM_THREADS='4', OMP_NUM_THREADS='4')
    with (folder / 'worker.log').open('w', encoding='utf-8') as log:
        process = subprocess.Popen([bpy.path.abspath(f.inference_python),
                                    str(Path(__file__).with_name('photo_fit_worker.py')), str(request)],
                                   stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                                   creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0), env=environment)
    _JOB = {'scene': context.scene, 'folder': folder, 'process': process,
            'stamp': fit_stamp(context, multiview), 'started': time.monotonic(), 'ready': False, 'initial': initial,
            'kind': 'MULTIVIEW' if multiview else 'FIT', 'timeout': 360 if multiview else 180}
    f.photo_fit_status = ('Fitting multiple views locally; your character is unchanged.' if multiview
                          else 'Fitting the reference locally; your character is unchanged.')
    bpy.app.timers.register(poll, first_interval=.2)


def cancel(message='Photo operation discarded; the accepted character is retained.'):
    global _JOB
    job, _JOB = _JOB, None
    if bpy.app.timers.is_registered(poll):
        bpy.app.timers.unregister(poll)
    if job:
        if job['process'].poll() is None:
            job['process'].terminate()
        if job.get('kind') == 'COLOR':
            from . import photo_color
            photo_color.discard(job)
        job['scene'].cm_features.photo_fit_status = message
        job['scene'].cm_settings.status = message
        redraw()


def poll():
    job = _JOB
    if not job:
        return None
    try:
        if job['process'].poll() is None:
            if time.monotonic() - job['started'] > job.get('timeout', 180):
                cancel('Photo operation exceeded its time limit. The owned worker was stopped.')
                return None
            return .2
        path = job['folder'] / 'result.json'
        if not path.is_file():
            raise ValueError('Photo fitting returned no result; check its worker log and local dependencies')
        report = json.loads(path.read_text(encoding='utf-8'))
        if job['process'].returncode or report.get('status') != 'complete':
            raise ValueError(report.get('error', 'Photo fitting failed'))
        if job.get('kind') == 'COLOR':
            from . import photo_color
            photo_color.ready(job, report)
            redraw()
            return None
        with bpy.context.temp_override(scene=job['scene'], view_layer=job['scene'].view_layers[0]):
            if fit_stamp(bpy.context, job.get('kind') == 'MULTIVIEW') != job['stamp']:
                raise ValueError('Character inputs changed during photo fitting; fit the reference again')
        values = np.asarray(report.get('coefficients'), dtype=float)
        camera = np.asarray(report.get('camera'), dtype=float)
        if (values.shape != (253,) or camera.shape != (6,) or not np.isfinite(values).all()
                or not np.isfinite(camera).all() or np.max(np.abs(values[:170])) > 2.50001
                or not np.array_equal(values[170:], job['initial'][170:])):
            raise ValueError('The photo fit returned invalid head parameters')
        job.update(ready=True, report=report)
        job['scene'].cm_features.photo_fit_status = (
            f"Starting shape ready: {report['before_rms_pixels']:.1f} to {report['after_rms_pixels']:.1f} px landmark error. "
            'Create the source to inspect likeness; skin and hair are separate.')
        if job.get('kind') == 'MULTIVIEW':
            job['scene'].cm_features.photo_fit_status = (
                f"Multi-view shape ready: {len(report['views']) - 1} supporting views. "
                f"Primary landmarks: {report['before_rms_pixels']:.1f} to {report['after_rms_pixels']:.1f} px. "
                'Apply the source proposal to review it before wrapping.')
        redraw()
        return None
    except Exception as exc:  # noqa: BLE001 - worker boundary must publish a usable failure status.
        cancel(str(exc))
        return None


def transfer_source_landmarks(context, original, source):
    """Carry authored anchors through the generated mesh's exact vertex correspondence."""
    from mathutils import Vector
    from mathutils.bvhtree import BVHTree
    from . import add_marker, region_projection
    pairs = context.scene.cm_settings.landmarks
    if not pairs:
        return []
    if mesh_ops.topology(original) != mesh_ops.topology(source):
        raise ValueError('The new source topology cannot retain the authored landmarks')
    old = np.empty((len(original.data.vertices), 3), dtype=np.float32)
    new = np.empty_like(old)
    original.data.vertices.foreach_get('co', old.ravel())
    source.data.vertices.foreach_get('co', new.ravel())
    original.data.calc_loop_triangles()
    triangles = np.array([triangle.vertices[:] for triangle in original.data.loop_triangles], dtype=int)
    tree = BVHTree.FromPolygons(old.tolist(), triangles.tolist(), all_triangles=True)
    changes = []
    for pair in pairs:
        if not pair.source:
            continue
        marker = pair.source
        if marker.parent != original:
            raise ValueError('A source landmark belongs to a different mesh')
        local = np.asarray(original.matrix_world.inverted() @ marker.matrix_world.translation)
        hit, _, triangle, _ = tree.find_nearest(Vector(local))
        if hit is None:
            raise ValueError('A source landmark could not be transferred')
        ids = triangles[triangle]
        bary = region_projection.barycentric(np.asarray(hit), old[ids])
        moved = local + bary @ (new[ids] - old[ids])
        replacement = add_marker(context, source, moved.tolist(), pair.name)
        changes.append((pair, marker, replacement))
    return changes


def apply(context):
    global _JOB
    job = _JOB
    if not job or not job['ready'] or context.scene != job['scene']:
        raise ValueError('Wait for a complete photo fit in this character scene')
    if job.get('kind') == 'COLOR':
        raise ValueError('Choose Apply Photo Color for this completed projection')
    s, f = context.scene.cm_settings, context.scene.cm_features
    character_project.preflight(context, s, f, allow_photo_fit=True)
    multiview = job.get('kind') == 'MULTIVIEW'
    if fit_stamp(context, multiview) != job['stamp']:
        raise ValueError('Character inputs changed; discard this fit and fit the reference again')
    previous = (f.generated_head, s.source, head_model.coefficients(f), [c.value for c in f.face_controls])
    previous_render = [(obj, obj.hide_render) for obj in dict.fromkeys(previous[:2]) if obj]
    before = set(bpy.data.user_map())
    landmark_changes = []
    try:
        with mesh_ops.selection_state(context, on_error=True):
            if multiview:
                from . import photo_color
                original = previous[0]
                supports = photo_color.supporting_photos(original)
                image = original['cm_photo_reference']
                if tuple(image.size) != (job['report']['width'], job['report']['height']):
                    raise ValueError('The primary reference changed dimensions during refinement')
            else:
                image = bpy.data.images.load(str(job['folder'] / 'reference.png'), check_existing=False)
                image.pack()
            f.generated_head = None
            head_model.update_coefficients(f, job['report']['coefficients'])
            if not multiview:
                for item in f.face_controls:
                    item.value = 0
            head_model.create_head(context, s, f, invalidate=False)
            source = s.source
            source['cm_photo_reference'] = image
            source['cm_photo_fit_report'] = json.dumps(job['report'])
            if multiview:
                source.parent = original.parent
                source.matrix_parent_inverse = original.matrix_parent_inverse.copy()
                source.rotation_mode = original.rotation_mode
                for field in ('delta_location', 'delta_scale', 'delta_rotation_euler', 'delta_rotation_quaternion'):
                    setattr(source, field, getattr(original, field).copy())
                source.matrix_world = original.matrix_world.copy()
                source['cm_front_axis'] = original.get('cm_front_axis', '-Y')
                if original.get('cm_head_placement'):
                    placement = json.loads(original['cm_head_placement'])
                    placement['placed'] = mesh_ops.transform_state(source)
                    source['cm_head_placement'] = json.dumps(placement)
                for index, supporting_image in supports:
                    source[photo_color.SUPPORT_KEY + str(index)] = supporting_image
                landmark_changes = transfer_source_landmarks(context, original, source)
                for pair, _, replacement in landmark_changes:
                    pair.source = replacement
            for old in dict.fromkeys(previous[:2]):
                if old and old != source and old.name in context.view_layer.objects:
                    old.hide_set(True)
                    old.hide_render = True
    except BaseException:
        for pair, original_marker, _ in landmark_changes:
            pair.source = original_marker
        f.generated_head, s.source = previous[:2]
        head_model.update_coefficients(f, previous[2])
        for item, value in zip(f.face_controls, previous[3]):
            item.value = value
        for obj, hidden in previous_render:
            obj.hide_render = hidden
        created = set(bpy.data.user_map()) - before
        if created:
            bpy.data.batch_remove(ids=created)
        raise
    if not multiview:
        bpy.ops.cm.clear_landmarks()
    else:
        for _, marker, _ in landmark_changes:
            marker.hide_set(True)
    f.input_mode = 'GENERATE'
    f.ui_stage = 'START'
    f.photo_fit_status = 'Photo-fitted source created. Review the overlay, then use Auto Wrap when satisfied.'
    if multiview:
        f.photo_fit_status = 'Multi-view source created. Review it, then Update Face. The wrapped character is retained.'
    s.status = f.photo_fit_status
    _JOB = None
    from . import guided
    guided.focus(context, [s.source], s.source.get('cm_front_axis', '-Y'))


@persistent
def detach(_=None):
    cancel()


@persistent
def recover(_=None):
    for scene in bpy.data.scenes:
        f = scene.cm_features
        if not (_JOB and _JOB['scene'] == scene) and f.photo_fit_status.startswith(
                ('Fitting the reference locally;', 'Fitting multiple views locally;', 'Multi-view shape ready:',
                 'Starting shape ready:', 'Photo color is projecting locally;', 'Photo color ready.')):
            f.photo_fit_status = 'The unapplied photo operation was interrupted; run it again. Your character is retained.'


def register():
    for handler in (bpy.app.handlers.load_pre, bpy.app.handlers.undo_pre):
        if detach not in handler:
            handler.append(detach)
    for handler in (bpy.app.handlers.load_post, bpy.app.handlers.undo_post, bpy.app.handlers.redo_post):
        if recover not in handler:
            handler.append(recover)


def unregister():
    detach()
    for handler in (bpy.app.handlers.load_pre, bpy.app.handlers.undo_pre):
        if detach in handler:
            handler.remove(detach)
    for handler in (bpy.app.handlers.load_post, bpy.app.handlers.undo_post, bpy.app.handlers.redo_post):
        if recover in handler:
            handler.remove(recover)


class CM_OT_fit_photo(bpy.types.Operator, ImportHelper):
    bl_idname = 'cm.fit_photo'
    bl_label = 'Fit Starting Face from Photo'
    bl_description = 'Estimate an editable starting head shape from one near-frontal neutral photo, locally'
    filter_glob: StringProperty(default='*.jpg;*.jpeg;*.png;*.webp;*.bmp;*.tif;*.tiff', options={'HIDDEN'})

    def execute(self, context):
        try:
            start(context, self.filepath)
            return {'FINISHED'}
        except Exception as exc:  # noqa: BLE001 - report failures at the Blender operator boundary.
            self.report({'ERROR'}, str(exc))
            return {'CANCELLED'}


class CM_OT_apply_photo(bpy.types.Operator):
    bl_idname = 'cm.apply_photo_fit'
    bl_label = 'Create Fitted Source'
    bl_options = {'REGISTER', 'UNDO'}  # noqa: RUF012 - Blender registration metadata.

    def execute(self, context):
        try:
            apply(context)
            return {'FINISHED'}
        except Exception as exc:  # noqa: BLE001 - report failures at the Blender operator boundary.
            self.report({'ERROR'}, str(exc))
            return {'CANCELLED'}


class CM_OT_refine_photo_views(bpy.types.Operator):
    bl_idname = 'cm.refine_photo_views'
    bl_label = 'Refine Shape from Supporting Photos'
    bl_description = 'Propose a multi-view source shape while protecting the primary likeness; generated views have lower weight'

    def execute(self, context):
        try:
            start(context, multiview=True)
            return {'FINISHED'}
        except Exception as exc:
            self.report({'ERROR'}, str(exc))
            return {'CANCELLED'}


class CM_OT_cancel_photo(bpy.types.Operator):
    bl_idname = 'cm.cancel_photo_fit'
    bl_label = 'Discard Photo Fit'

    def execute(self, context):
        cancel()
        return {'FINISHED'}


def draw(layout, context):
    f = context.scene.cm_features
    if _JOB and _JOB['scene'] == context.scene:
        from .guided import text
        text(layout, f.photo_fit_status)
        if _JOB['ready']:
            layout.operator('cm.apply_photo_color' if _JOB.get('kind') == 'COLOR' else 'cm.apply_photo_fit',
                            text='Apply Multi-View Source' if _JOB.get('kind') == 'MULTIVIEW' else 'Apply Photo Color' if _JOB.get('kind') == 'COLOR' else 'Create Fitted Source',
                            icon='USER')
        layout.operator('cm.cancel_photo_fit', text='Discard Photo Color' if _JOB.get('kind') == 'COLOR' else 'Discard Photo Fit', icon='CANCEL')
        return True
    return False


CLASSES = (CM_OT_fit_photo, CM_OT_apply_photo, CM_OT_refine_photo_views, CM_OT_cancel_photo)
