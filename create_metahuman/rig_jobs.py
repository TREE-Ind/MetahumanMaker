"""Blender entry and durable observation for native full rigging."""
import json
import queue
import re
import threading
from datetime import datetime, timezone
from pathlib import Path

import bpy
from . import bridge, unreal_jobs, unreal_rig


def history(scene):
    s = scene.cm_settings
    name = re.sub(r'[^A-Za-z0-9_]', '_', s.character_name).strip('_')
    return Path(bpy.path.abspath(s.project_dir)).resolve() / 'Export' / name / 'last-rig-action.json'


def monitor(scene, folder):
    s, f = scene.cm_settings, scene.cm_features
    job = dict(scene=scene, folder=folder, snapshot=folder/'scripts', receipt=folder/'receipt.json',
               events=queue.Queue(), dispatched=threading.Event(), error=None, kind='REQUEST_RIG',
               project_dir=str(Path(bpy.path.abspath(s.project_dir)).resolve()),
               character_name=s.character_name, character=f.unreal_character, thread=threading.Thread())
    unreal_jobs._JOB = job
    if not bpy.app.timers.is_registered(unreal_jobs.poll):
        bpy.app.timers.register(unreal_jobs.poll, first_interval=.2, persistent=True)
    f.unreal_rig_receipt = str(job['receipt'])
    return job


def send(job, engine, project):
    command = 'import runpy\nrunpy.run_path(' + repr(str(job['snapshot']/'unreal_rig.py')) + ")[\"dispatch\"](" + repr(str(job['folder']/'request.json')) + ')'
    try:
        bridge.run_python(engine, command, project, on_dispatch=job['dispatched'].set)
        job['events'].put(('returned', None))
    except Exception as exc:
        job['events'].put(('error', str(exc)))


def restore(scene):
    """Observe only the original action; reopening never submits another rig."""
    path = history(scene)
    if not path.is_file():
        return False
    record = json.loads(path.read_text(encoding='utf-8'))
    s, f = scene.cm_settings, scene.cm_features
    root = Path(bpy.path.abspath(s.project_dir)).resolve()
    if (record.get('schema') != 1 or record.get('project_dir') != str(root)
            or record.get('character_name') != s.character_name
            or record.get('character') != f.unreal_character
            or Path(record.get('expected_project', '')).resolve() != Path(bpy.path.abspath(f.unreal_character_project)).resolve()):
        return False
    receipt = Path(record['receipt']).resolve()
    if receipt.name != 'receipt.json' or not receipt.is_relative_to(root/'Export/.unreal-actions'):
        raise ValueError('The rig receipt must stay inside this character project')
    folder = receipt.parent
    if unreal_rig.digest(folder/'request.json') != record['request_sha256']:
        raise ValueError('The saved rig request changed; inspect its original action before continuing')
    f.unreal_rig_receipt = str(receipt)
    if receipt.is_file():
        result = json.loads(receipt.read_text(encoding='utf-8'))
        if (result.get('character') != f.unreal_character or result.get('action') != 'REQUEST_RIG'
                or result.get('request_sha256') != record['request_sha256']):
            raise ValueError('The rig receipt does not match its original request')
        if result.get('status') in {'complete', 'failed'}:
            if not result.get('inspected_at'):
                result = dict(result, inspected_at=datetime.fromtimestamp(receipt.stat().st_mtime, timezone.utc).isoformat())
            f.unreal_rig_result = json.dumps(result)
            # A historical rig receipt must not replace a more recent status or build.
            try:
                current = json.loads(f.unreal_status_report or '{}')
            except ValueError:
                current = {}
            if unreal_jobs.receipt_time(result)>unreal_jobs.receipt_time(current):
                unreal_jobs.apply_receipt(scene, result)
            return False
    if unreal_jobs._JOB:
        if unreal_jobs._JOB['folder'] == folder:
            unreal_jobs._JOB['scene'] = scene
        return True
    if not (folder/'dispatch.json').is_file():
        f.unreal_push_status = 'Rig action prepared without confirmed dispatch. Inspect its original request before continuing.'
        return True
    job = monitor(scene, folder)
    job['dispatched'].set()
    f.unreal_push_status = 'Monitoring the original full rig request. No second request was sent.'
    return True


def start(context, *, defer=False):
    from . import character_project, conform_geometry, groom_jobs, native_geometry_state, skin_native_jobs
    if unreal_jobs.running() or groom_jobs.running() or skin_native_jobs.running():
        raise ValueError('Wait for the current Unreal operation before rigging')
    if restore(context.scene):
        raise ValueError('Observe the original full rig action before starting another')
    s, f = context.scene.cm_settings, context.scene.cm_features
    if context.mode != 'OBJECT' or f.expression_active or f.body_editing or f.full_sculpt:
        raise ValueError('Return to Neutral and finish the active edit before rigging')
    if not s.result or s.part == 'BODY':
        raise ValueError('Select the saved character head before rigging')
    character_project.preflight(context, s, f)
    native_geometry_state.guard(context)
    engine = bpy.path.abspath(s.unreal_engine_dir)
    if not (Path(engine)/'Engine/Plugins/Experimental/PythonScriptPlugin/Content/Python/remote_execution.py').is_file():
        raise ValueError('Choose the Unreal installation containing Engine/')
    if bpy.ops.cm.save_project() != {'FINISHED'}:
        raise ValueError('Save the character project before rigging')
    prepared = unreal_rig.prepare(bpy.path.abspath(s.project_dir), s.character_name, f.unreal_character,
                                 bpy.path.abspath(f.unreal_character_project), geometry=conform_geometry.prepare(context, s, f))
    folder = Path(prepared['folder'])
    f.unreal_rig_receipt = prepared['receipt']
    f.unreal_rig_result = ''
    if defer:
        f.unreal_push_status = 'Full rig action prepared. Dispatch this original request in the linked Unreal project.'
    else:
        job = monitor(context.scene, folder)
        job['thread'] = threading.Thread(target=send, args=(job, engine, s.unreal_project), name='CreateMHFullRig', daemon=True)
        f.unreal_push_status = 'Requesting the linked character’s full rig. Blender remains usable.'
        job['thread'].start()
    s.status = f.unreal_push_status
    return folder
