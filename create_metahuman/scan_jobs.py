"""Cancellable scan projection, with immutable geometry-only worker inputs."""

import hashlib
import json
import os
import subprocess
import time
import uuid
from pathlib import Path

import bpy
import numpy as np
from bpy.app.handlers import persistent

from . import mesh_ops as m
from . import scan_projection as scan
from .workflow import group_weights

_JOB = None
_REAPING = []


def running():
    return _JOB is not None


def parts(scene):
    s, f = scene.cm_settings, scene.cm_features
    body = s.part == 'FULL_BODY' and f.texture_part == 'BODY'
    return s.source, f.body_result if body else s.result, f.body_detail if body else s.detail


def options(scene):
    s, f = scene.cm_settings, scene.cm_features
    return {'character_name': s.character_name, 'detail_levels': s.detail_levels,
        'detail_strength': s.detail_strength, 'max_distance': s.max_distance}, {
        'use_detail_mask': f.use_detail_mask, 'neck_fade': f.neck_fade}


def state(scene):
    s, f = scene.cm_settings, scene.cm_features
    return [s.project_dir, s.part, f.texture_part, scene.frame_current, scene.frame_subframe,
        options(scene), [obj.as_pointer() if obj else 0 for obj in parts(scene)],
        f.body_editing, f.restore_editing, f.expression_active, bool(f.full_sculpt), bool(f.detail_mask_helper)]


def geometry_stamp(context, obj, *, groups=False, pending=False, transform=True):
    digest = hashlib.sha256()
    evaluated = obj.evaluated_get(context.evaluated_depsgraph_get())
    data = evaluated.to_mesh(preserve_all_data_layers=True, depsgraph=context.evaluated_depsgraph_get())
    try:
        for collection, field, width, dtype in ((data.vertices, 'co', 3, np.float32),
                (data.loops, 'vertex_index', 1, np.int32), (data.polygons, 'loop_total', 1, np.int32)):
            values = np.empty(len(collection)*width, dtype=dtype)
            collection.foreach_get(field, values)
            digest.update(values.tobytes())
        if transform:digest.update(np.asarray(evaluated.matrix_world, dtype=np.float64).tobytes())
        if groups:
            for uv in data.uv_layers:
                values = np.empty(len(uv.data)*2, dtype=np.float32)
                uv.data.foreach_get('uv', values)
                digest.update(uv.name.encode()+values.tobytes())
            for name in ('crease_edge', 'crease_vert'):
                attribute = data.attributes.get(name)
                if attribute:
                    values = np.empty(len(attribute.data), dtype=np.float32)
                    attribute.data.foreach_get('value', values)
                    digest.update(name.encode()+values.tobytes())
    finally:
        evaluated.to_mesh_clear()
    if groups:
        digest.update(json.dumps([(slot.link, slot.material.as_pointer() if slot.material else 0)
            for slot in obj.material_slots]).encode())
        for name in ('CM_Pin', 'CM_DetailMask'):
            digest.update(name.encode()+group_weights(obj, name).astype(np.float32).tobytes())
    if pending:
        digest.update(scan.shape_signature(obj).encode())
        digest.update(json.dumps([obj.get('cm_scan_applied'), obj.get('cm_scan_mask_dirty'),
            obj.get(scan.SIGNATURE), len(obj.modifiers)]).encode())
    return digest.hexdigest()


def snapshot(context, obj, label, *, pending=False, materials=False):
    evaluated = obj.evaluated_get(context.evaluated_depsgraph_get())
    data = bpy.data.meshes.new_from_object(evaluated, preserve_all_data_layers=True,
        depsgraph=context.evaluated_depsgraph_get())
    data.id_properties_clear()
    result = bpy.data.objects.new(label, data)
    result.matrix_world = evaluated.matrix_world.copy()
    try:
        if pending:
            data.vertices.foreach_set('co', scan.key_points(obj.data.shape_keys.key_blocks['Basis']).ravel())
        if materials:
            indices = np.empty(len(data.polygons), dtype=np.int32)
            data.polygons.foreach_get('material_index', indices)
            count = len(data.materials)
            data.materials.clear()
            for index in range(count):
                data.materials.append(bpy.data.materials.new(label+' Slot '+str(index)))
            data.polygons.foreach_set('material_index', indices)
        else:
            data.materials.clear()
        for group in obj.vertex_groups:
            result.vertex_groups.new(name=group.name)
        return result
    except Exception:
        bpy.data.objects.remove(result, do_unlink=True)
        bpy.data.meshes.remove(data)
        raise


def cleanup(job):
    # Only generated leaves in this unique owned job directory; never the project.
    for name in ('input.blend', 'output.blend', 'output.npz'):
        path = job['folder'] / name
        if path.is_file():
            path.unlink()


def receipt(job, status, error=''):
    value = {'status': status, 'error': error, 'worker_pid': job['process'].pid,
        'exit_code': job['process'].poll(), 'regenerate': job['request']['regenerate'],
        'seconds': round(time.monotonic()-job['started'], 3)}
    (job['folder'] / 'receipt.json').write_text(json.dumps(value, indent=2), encoding='utf-8')


def start(context):
    global _JOB
    from . import (
        bake_jobs,
        detail_jobs,
        groom_jobs,
        native_body,
        paint_tools,
        skin_library,
        skin_native_jobs,
        unreal_jobs,
        wrinkle_jobs,
    )
    s, f = context.scene.cm_settings, context.scene.cm_features
    if running() or context.mode != 'OBJECT' or f.body_editing or f.full_sculpt or f.restore_editing or f.expression_active or f.detail_mask_helper:
        raise ValueError('Finish the active edit or scan projection first')
    if any(module.running() for module in (bake_jobs, detail_jobs, groom_jobs, skin_library, skin_native_jobs, unreal_jobs, wrinkle_jobs)) or native_body._JOB or paint_tools.running():
        raise ValueError('Finish or cancel the current background operation before projecting scan detail')
    source, accepted, detail = parts(context.scene)
    m.check_mesh(source, 'Scan source')
    m.check_mesh(accepted, 'Accepted result')
    if detail:
        scan.check(detail)
    from . import resource_budget
    resource_budget.require('scan detail projection',512*resource_budget.MIB+
        resource_budget.meshes((source,accepted,detail)))
    if not bpy.app.background and context.preferences.edit.use_global_undo:
        bpy.ops.ed.undo_push(message='Start scan projection')
    scene_state = state(context.scene)
    stamps = [geometry_stamp(context, obj, groups=index>0, pending=index==2)
        for index, obj in enumerate((source, accepted, detail)) if obj]
    folder = Path(bpy.path.abspath(s.project_dir)).resolve() / 'ScanProjection' / '.jobs' / uuid.uuid4().hex[:12]
    folder.mkdir(parents=True)
    objects = {}
    snapshots = []
    try:
        for key, obj in (('source', source), ('accepted', accepted), ('detail', detail)):
            if obj:
                copied = snapshot(context, obj, 'CM Scan '+key, pending=key=='detail', materials=key!='source')
                snapshots.append(copied)
                objects[key] = copied.name
        bpy.data.libraries.write(str(folder / 'input.blend'), set(snapshots), fake_user=True, compress=False)
    except Exception:
        cleanup({'folder': folder})
        raise
    finally:
        for copied in snapshots:
            data = copied.data
            placeholders = list(data.materials)
            bpy.data.objects.remove(copied, do_unlink=True)
            bpy.data.meshes.remove(data)
            for material in placeholders:
                bpy.data.materials.remove(material)
    settings, features = options(context.scene)
    request = {'objects': objects, 'settings': settings, 'features': features, 'regenerate': bool(detail),
        'addon_root': str(Path(__file__).resolve().parents[1]), 'parent': os.getpid()}
    (folder / 'request.json').write_text(json.dumps(request, indent=2), encoding='utf-8')
    from .detail_jobs import thread_arguments
    try:
        with (folder / 'worker.log').open('w', encoding='utf-8') as log:
            process = subprocess.Popen([bpy.app.binary_path, '--background', '--factory-startup',
                *thread_arguments(), '--python-exit-code', '1', '--python', str(Path(__file__).with_name('scan_worker.py')),
                '--', str(folder / 'request.json')], stdout=log, stderr=subprocess.STDOUT,
                creationflags=subprocess.BELOW_NORMAL_PRIORITY_CLASS if os.name == 'nt' else 0)
    except Exception:
        cleanup({'folder': folder})
        raise
    _JOB = {'scene': context.scene, 'window': context.window or context.window_manager.windows[0],
        'folder': folder, 'request': request, 'process': process, 'state': scene_state,
        'stamps': stamps, 'objects': (source, accepted, detail), 'cancel': False, 'started': time.monotonic()}
    from . import process_ref
    try:
        (folder / 'ownership.json').write_text(json.dumps({'kind': 'CREATE_MH_SCAN_V1',
            'parent': process_ref.identify(os.getpid()), 'worker': process_ref.identify(process.pid)}, indent=2), encoding='utf-8')
    except OSError:
        cancel()
        bpy.app.timers.register(poll, first_interval=.2)
        raise
    s.status = 'Projecting scan detail in background. Cancel keeps the current character and sculpt.'
    f.last_action_error = False
    bpy.app.timers.register(poll, first_interval=.2)


class CM_OT_commit_scan_job(bpy.types.Operator):
    bl_idname = 'cm.commit_scan_job'
    bl_label = 'Apply Finished Scan Projection'
    bl_options = {'INTERNAL'}  # noqa: RUF012

    def execute(self, context):
        try:
            return self.commit(context)
        except Exception as exc:  # noqa: BLE001
            if _JOB:
                _JOB['error'] = str(exc)
            self.report({'ERROR'}, str(exc))
            return {'CANCELLED'}

    def commit(self, context):
        job = _JOB
        if not job or job['process'].poll() is None or job['cancel']:
            return {'CANCELLED'}
        scene = job['scene']
        if context.scene != scene or context.mode != 'OBJECT' or state(scene) != job['state']:
            raise ValueError('Character, part or projection settings changed; the previous sculpt was preserved')
        for index, (obj, expected_stamp) in enumerate(zip((obj for obj in job['objects'] if obj), job['stamps'])):
            if geometry_stamp(context, obj, groups=index>0, pending=index==2) != expected_stamp:
                raise ValueError('Source, accepted geometry or mask changed; the previous sculpt was preserved')
        s, f = scene.cm_settings, scene.cm_features
        result = job['report']
        expected = 'output.npz' if job['request']['regenerate'] else 'output.blend'
        if result.get('file') != expected:
            raise ValueError('Unexpected scan output')
        output = job['folder'] / expected
        if hashlib.sha256(output.read_bytes()).hexdigest() != result['sha256']:
            raise ValueError('Scan output changed; the previous sculpt was preserved')
        _source, accepted, detail = job['objects']
        if detail:
            # The current geometry/keys/mask stamp already matches the pristine
            # start snapshot; avoid a second dense validation pass here.
            with np.load(output, allow_pickle=False) as values:
                points = values['points'].copy()
            scan.commit(s, f, detail, points)
        else:
            with bpy.data.libraries.load(str(output), link=False) as (_library, destination):
                destination.objects = [result['object']]
            obj = destination.objects[0]
            try:
                topology = m.topology(obj)
                if len(obj.data.vertices) != result['vertices'] or topology != result['topology']:
                    raise ValueError('Scan output topology is invalid')
                scan.check(obj, topology=topology)
                from . import detail_layers
                detail_layers.bind(context, accepted, obj, scan.key_points(obj.data.shape_keys.key_blocks['Basis']), detail_topology=topology)
                indices = np.empty(len(obj.data.polygons), dtype=np.int32)
                obj.data.polygons.foreach_get('material_index', indices)
                obj.data.materials.clear()
                for material in accepted.data.materials:
                    obj.data.materials.append(material)
                for slot, original in zip(obj.material_slots, accepted.material_slots):
                    slot.link = original.link
                    if original.link == 'OBJECT':
                        slot.material = original.material
                obj.data.polygons.foreach_set('material_index', indices)
                scene.collection.objects.link(obj)
                obj.name = s.character_name + ('_Body' if s.part=='FULL_BODY' and f.texture_part=='BODY' else '') + '_Detail'
                if s.part == 'FULL_BODY' and f.texture_part == 'BODY':
                    f.body_detail = obj
                else:
                    s.detail = obj
                s.stage = 'DETAIL'
                m.activate(context, obj)
            except Exception:
                bpy.data.objects.remove(obj, do_unlink=True)
                raise
        s.status = 'Scan preview ready. Review strength and mask, then Apply Projected Detail once.'
        if not bpy.app.background and context.preferences.edit.use_global_undo:
            bpy.ops.ed.undo_push(message='Generate scan displacement')
        job['committed'] = True
        return {'FINISHED'}


def poll():
    global _JOB
    job = _JOB
    if not job:
        return None
    code = job['process'].poll()
    if code is None:
        path = job['folder'] / 'report.json'
        try:
            stage = json.loads(path.read_text(encoding='utf-8')).get('stage') if path.is_file() else None
            if stage:
                job['scene'].cm_settings.status = stage
        except (OSError, ValueError, ReferenceError):
            pass
        return .2
    status, error = 'failed', ''
    try:
        if job['cancel']:
            status = 'canceled'
            job['scene'].cm_settings.status = 'Scan projection canceled. The previous character and sculpt were retained.'
            job['scene'].cm_features.last_action_error = False
        else:
            job['report'] = json.loads((job['folder'] / 'report.json').read_text(encoding='utf-8'))
            if code or job['report'].get('status') != 'complete':
                raise ValueError(job['report'].get('error', 'Scan worker stopped; see worker.log'))
            job['scene'].cm_settings.status = 'Loading the completed scan preview…'
            with bpy.context.temp_override(window=job['window'], area=None, region=None):
                if bpy.ops.cm.commit_scan_job() != {'FINISHED'} or not job.get('committed'):
                    raise ValueError('Scan output was not committed; the previous sculpt was retained')
            status = 'complete'
            job['scene'].cm_features.last_action_error = False
    except Exception as exc:  # noqa: BLE001
        error = job.get('error') or str(exc)
        try:
            job['scene'].cm_settings.status = 'Scan projection did not apply: '+error
            job['scene'].cm_features.last_action_error = True
        except ReferenceError:
            pass
    finally:
        _JOB = None
        try:
            cleanup(job)
            receipt(job, status, error)
        except OSError:
            # A locked temporary bundle must not lock the artist workflow.
            # Its ownership journal lets project resume retry cleanup safely.
            pass
        for window in bpy.context.window_manager.windows:
            for area in window.screen.areas:
                area.tag_redraw()
    return None


def cancel():
    if _JOB:
        _JOB['cancel'] = True
        if _JOB['process'].poll() is None:
            _JOB['process'].terminate()


def reap():
    for job in list(_REAPING):
        if job['process'].poll() is not None:
            try:
                cleanup(job)
                receipt(job, 'canceled')
            except OSError:
                continue
            _REAPING.remove(job)
    return .2 if _REAPING else None


@persistent
def detach(_=None):
    global _JOB
    if _JOB:
        cancel()
        job, _JOB = _JOB, None
        if bpy.app.timers.is_registered(poll):
            bpy.app.timers.unregister(poll)
        _REAPING.append(job)
        if not bpy.app.timers.is_registered(reap):
            bpy.app.timers.register(reap, first_interval=.2, persistent=True)


class CM_OT_cancel_scan_job(bpy.types.Operator):
    bl_idname = 'cm.cancel_scan_job'
    bl_label = 'Cancel Scan Projection'

    def execute(self, context):
        cancel()
        return {'FINISHED'}


@persistent
def recover(_=None):
    """Discard abandoned owned geometry bundles; never restart their requests."""
    from . import process_ref
    for scene in bpy.data.scenes:
        root = (Path(bpy.path.abspath(scene.cm_settings.project_dir)).resolve()
            / 'ScanProjection' / '.jobs')
        if not root.is_dir():
            continue
        for ownership in root.glob('*/ownership.json'):
            try:
                folder = ownership.parent.resolve()
                if folder.parent != root.resolve():
                    continue
                owner = json.loads(ownership.read_text(encoding='utf-8'))
                if owner.get('kind') != 'CREATE_MH_SCAN_V1' or not owner.get('parent') or not owner.get('worker'):
                    continue
                if process_ref.alive(owner['parent']) or process_ref.alive(owner['worker']):
                    continue
                if not any((folder/name).is_file() for name in ('input.blend', 'output.blend', 'output.npz')):
                    continue
                cleanup({'folder': folder})
                (folder / 'receipt.json').write_text(json.dumps({'status': 'abandoned',
                    'error': 'The previous editor and worker exited. Temporary geometry was removed; the saved sculpt was retained.'}, indent=2), encoding='utf-8')
            except (OSError, ValueError, KeyError, TypeError):
                continue


def draw(layout, context):
    if not running():
        return False
    layout.label(text='Projecting Scan Detail', icon='MOD_SHRINKWRAP')
    layout.label(text='The current sculpt is retained until ready.')
    layout.operator('cm.cancel_scan_job', icon='CANCEL')
    from .guided import text
    text(layout, context.scene.cm_settings.status)
    return True


def register():
    for cls in (CM_OT_commit_scan_job, CM_OT_cancel_scan_job):
        bpy.utils.register_class(cls)
    for handlers in (bpy.app.handlers.load_pre, bpy.app.handlers.undo_pre, bpy.app.handlers.redo_pre):
        handlers.append(detach)
    bpy.app.handlers.load_post.append(recover)
    # Blender restricts bpy.data while enabling an add-on. Inspect scenes only
    # after registration has returned to the normal event loop.
    if not bpy.app.timers.is_registered(recover):
        bpy.app.timers.register(recover, first_interval=.5)


def unregister():
    detach()
    if bpy.app.timers.is_registered(recover):
        bpy.app.timers.unregister(recover)
    if recover in bpy.app.handlers.load_post:
        bpy.app.handlers.load_post.remove(recover)
    for handlers in (bpy.app.handlers.load_pre, bpy.app.handlers.undo_pre, bpy.app.handlers.redo_pre):
        if detach in handlers:
            handlers.remove(detach)
    for cls in (CM_OT_cancel_scan_job, CM_OT_commit_scan_job):
        bpy.utils.unregister_class(cls)
