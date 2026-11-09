"""Cancellable local photo fitting, with explicit publication of a new source."""

import json
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


def start(context, filename):
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
    source = Path(bpy.path.abspath(filename))
    if not source.is_file() or source.stat().st_size > 40 * 1024**2:
        raise ValueError('Choose a readable face photo smaller than 40 MB')
    from . import resource_budget
    resource_budget.require('photo fitting',768*resource_budget.MIB)
    folder = Path(bpy.path.abspath(s.project_dir)) / 'PhotoFits' / uuid.uuid4().hex[:12]
    folder.mkdir(parents=True)
    photo = folder / ('input' + source.suffix.lower())
    shutil.copy2(source, photo)
    request = folder / 'request.json'
    initial = head_model.coefficients(f).tolist()
    request.write_text(json.dumps({'photo': str(photo.resolve()), 'models': str(models.resolve()),
                                  'initial': initial, 'parent': os.getpid()}), encoding='utf-8')
    environment = os.environ.copy()
    environment.update(OPENBLAS_NUM_THREADS='4', OMP_NUM_THREADS='4')
    with (folder / 'worker.log').open('w', encoding='utf-8') as log:
        process = subprocess.Popen([bpy.path.abspath(f.inference_python),
                                    str(Path(__file__).with_name('photo_fit_worker.py')), str(request)],
                                   stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                                   creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0), env=environment)
    _JOB = {'scene': context.scene, 'folder': folder, 'process': process,
            'stamp': stamp(context), 'started': time.monotonic(), 'ready': False, 'initial': initial}
    f.photo_fit_status = 'Fitting the reference locally; your character is unchanged.'
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
            if time.monotonic() - job['started'] > 180:
                cancel('Photo operation exceeded three minutes; the owned worker was stopped.')
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
            if stamp(bpy.context) != job['stamp']:
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
        redraw()
        return None
    except Exception as exc:  # noqa: BLE001 - worker boundary must publish a usable failure status.
        cancel(str(exc))
        return None


def apply(context):
    global _JOB
    job = _JOB
    if not job or not job['ready'] or context.scene != job['scene']:
        raise ValueError('Wait for a complete photo fit in this character scene')
    if job.get('kind') == 'COLOR':
        raise ValueError('Choose Apply Photo Color for this completed projection')
    s, f = context.scene.cm_settings, context.scene.cm_features
    character_project.preflight(context, s, f, allow_photo_fit=True)
    if stamp(context) != job['stamp']:
        raise ValueError('Character inputs changed; discard this fit and fit the reference again')
    previous = (f.generated_head, s.source, head_model.coefficients(f), [c.value for c in f.face_controls])
    previous_render = [(obj, obj.hide_render) for obj in dict.fromkeys(previous[:2]) if obj]
    before = set(bpy.data.user_map())
    try:
        with mesh_ops.selection_state(context, on_error=True):
            image = bpy.data.images.load(str(job['folder'] / 'reference.png'), check_existing=False)
            image.pack()
            f.generated_head = None
            head_model.update_coefficients(f, job['report']['coefficients'])
            for item in f.face_controls:
                item.value = 0
            head_model.create_head(context, s, f, invalidate=False)
            source = s.source
            source['cm_photo_reference'] = image
            source['cm_photo_fit_report'] = json.dumps(job['report'])
            for old in dict.fromkeys(previous[:2]):
                if old and old != source and old.name in context.view_layer.objects:
                    old.hide_set(True)
                    old.hide_render = True
    except BaseException:
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
    bpy.ops.cm.clear_landmarks()
    f.input_mode = 'GENERATE'
    f.ui_stage = 'START'
    f.photo_fit_status = 'Photo-fitted source created. Review the overlay, then use Auto Wrap when satisfied.'
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
                ('Fitting the reference locally;', 'Starting shape ready:', 'Photo color is projecting locally;', 'Photo color ready.')):
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
            layout.operator('cm.apply_photo_color' if _JOB.get('kind') == 'COLOR' else 'cm.apply_photo_fit', icon='USER')
        layout.operator('cm.cancel_photo_fit', text='Discard Photo Color' if _JOB.get('kind') == 'COLOR' else 'Discard Photo Fit', icon='CANCEL')
        return True
    return False


CLASSES = (CM_OT_fit_photo, CM_OT_apply_photo, CM_OT_cancel_photo)
