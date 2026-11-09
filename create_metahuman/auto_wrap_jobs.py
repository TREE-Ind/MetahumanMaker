"""One-action Auto Wrap with cancellable, immutable landmark detection."""
import json
import hashlib
import os
import subprocess
import time
import uuid
from pathlib import Path

import bpy
import numpy as np
from bpy.app.handlers import persistent
from mathutils import Matrix, Vector

from . import mesh_ops as m

_JOB=None
_PENDING_CLEANUP=set()


def running():
    return _JOB is not None


def actors(scene):
    s,f=scene.cm_settings,scene.cm_features
    return list(dict.fromkeys(obj for obj in (s.source,s.template,f.body_template,s.result,
        f.body_result,f.eye_left,f.eye_right,f.teeth,
        f.proxy if f.proxy and f.proxy.parent==s.source else None) if obj))


def landmark_geometry(context):
    from .scan_jobs import geometry_stamp
    s,f=context.scene.cm_settings,context.scene.cm_features
    return {role:geometry_stamp(context,obj) for role,obj in
        [('source',s.source),('head',s.template),*([('body',f.body_template)] if s.part=='FULL_BODY' and f.body_template else [])]}


def needs_detection(context):
    s,f=context.scene.cm_settings,context.scene.cm_features
    if not s.landmarks:return True
    if any(not p.source or not p.template or p.source.parent!=s.source or
           p.template.parent not in (s.template,f.body_template) for p in s.landmarks):return True
    saved=s.source.get('cm_auto_landmarks_state') if s.source else None
    try:return bool(saved and json.loads(saved)!=landmark_geometry(context))
    except (ValueError,TypeError):return True


def state(context,scene):
    from .scan_jobs import geometry_stamp
    from .workflow import group_weights
    s,f=scene.cm_settings,scene.cm_features
    return ([s.part,s.project_dir,s.character_name,f.front_axis,f.source_front_axis,f.landmark_kind,
        f.wrap_method,f.wrap_quality,f.conservative,f.is_mask,s.iterations,s.projection_strength,
        s.smoothing,s.landmark_radius,s.max_distance,f.expression_active,f.body_editing,
        f.full_sculpt,f.restore_editing,f.full_head,f.model_dir,f.inference_python,f.expression_reference,
        [s.source.get(key) for key in ('cm_front_axis','cm_gnm','cm_head_placement')],
        [(p.name,p.source.as_pointer() if p.source else 0,p.template.as_pointer() if p.template else 0,
          list(p.source.location) if p.source else [],list(p.template.location) if p.template else []) for p in s.landmarks]],
        [(obj.as_pointer(),geometry_stamp(context,obj,groups=True),
          [(name,hashlib.sha256(group_weights(obj,name).tobytes()).hexdigest())
           for name in ('CM_WrapMask','CM_Inner_Parts','CM_Ears') if obj.vertex_groups.get(name)]) for obj in actors(scene)])


def cleanup(job):
    path=job['folder']/'input.blend'
    try:path.unlink(missing_ok=True)
    except OSError:
        _PENDING_CLEANUP.add(path)
        if not bpy.app.timers.is_registered(reap):bpy.app.timers.register(reap,first_interval=.5,persistent=True)


def reap():
    for path in list(_PENDING_CLEANUP):
        try:path.unlink(missing_ok=True)
        except OSError:continue
        _PENDING_CLEANUP.remove(path)
    return .5 if _PENDING_CLEANUP else None


def start(context,*,wrap):
    global _JOB
    from . import scan_jobs,paint_tools,detail_mask,bake_jobs,detail_jobs,wrinkle_jobs,unreal_jobs,native_body,groom_jobs,lookdev
    from . import expressions,skin_library,skin_native_jobs
    from .features import generated_head_update
    from .setup_paths import resolve
    from . import landmark_tools
    s,f=context.scene.cm_settings,context.scene.cm_features
    m.check_inputs(s)
    if landmark_tools.active():raise ValueError('Finish editing landmark pairs before Auto Wrap')
    if not s.source.is_editable:raise ValueError('Make the Source local before automatic alignment')
    if running() or context.mode!='OBJECT' or f.body_editing or f.full_sculpt or f.restore_editing or f.expression_active:
        raise ValueError('Finish the active operation and return to Neutral before Auto Wrap')
    if any(module.running() for module in (scan_jobs,bake_jobs,detail_jobs,wrinkle_jobs,unreal_jobs,groom_jobs,expressions,skin_library,skin_native_jobs)) or native_body._JOB or lookdev._JOB or paint_tools.running() or detail_mask.running(context.scene):
        raise ValueError('Finish or cancel the current background or paint operation before Auto Wrap')
    resolve(f)
    window=context.window or next((w for w in context.window_manager.windows if w.scene==context.scene),None)
    if not window:raise ValueError('Open this character in a Blender window before starting background Auto Wrap')
    part='HEAD' if generated_head_update(s) else s.part
    if part=='FULL_BODY':m.check_mesh(f.body_template,'Body Template')
    if not Path(bpy.path.abspath(f.inference_python)).is_file():raise ValueError('Resolve the local inference Python before Auto Wrap')
    kind='POSE' if part in {'BODY','FULL_BODY'} else f.landmark_kind
    model=Path(bpy.path.abspath(f.model_dir))/('face_landmarker.task' if kind=='FACE' else 'pose_landmarker_lite.task')
    if not model.is_file():raise ValueError('Prepare the optional local landmark model, or use Manual wrapping')
    from . import resource_budget
    resource_budget.require('background Auto Wrap',768*resource_budget.MIB+
        resource_budget.meshes((s.source,s.template,f.body_template if part=='FULL_BODY' else None)))
    if not bpy.app.background and context.preferences.edit.use_global_undo:
        with context.temp_override(window=window):bpy.ops.ed.undo_push(message='Start Auto Wrap')
    stamp=state(context,context.scene)
    folder=Path(bpy.path.abspath(s.project_dir)).resolve()/'Automation'/'.jobs'/uuid.uuid4().hex[:12]
    folder.mkdir(parents=True)
    copies=[]
    names={}
    try:
        for role,obj in [('source',s.source),('template',s.template),*([('body',f.body_template)] if part=='FULL_BODY' else [])]:
            copied=scan_jobs.snapshot(context,obj,'CM Auto '+role)
            copies.append(copied);names[role]=copied.name
            for key in ('cm_front_axis','cm_gnm'):
                if key in obj:copied[key]=obj[key]
            if role=='source' and generated_head_update(s) and s.source.get('cm_head_placement'):
                frame=json.loads(s.source['cm_head_placement'])
                if m.transform_state(s.source)==frame['placed']:
                    m.restore_transform(copied,frame['fit'])
                else:copied.matrix_world=Matrix.Translation(-Vector(frame['movement']))@copied.matrix_world
        bpy.data.libraries.write(str(folder/'input.blend'),set(copies),compress=False)
    except Exception:
        cleanup({'folder':folder})
        raise
    finally:
        for obj in copies:
            data=obj.data;bpy.data.objects.remove(obj,do_unlink=True)
            if not data.users:bpy.data.meshes.remove(data)
    request={'addon_root':str(Path(__file__).resolve().parents[1]),'parent':os.getpid(),'objects':names,'part':part,
        'features':{key:getattr(f,key) for key in ('model_dir','inference_python','front_axis','source_front_axis','landmark_kind')}}
    request['features']['landmark_kind']=kind
    try:
        (folder/'request.json').write_text(json.dumps(request),encoding='utf-8')
        from .detail_jobs import thread_arguments
        with (folder/'worker.log').open('w',encoding='utf-8') as log:
            process=subprocess.Popen([bpy.app.binary_path,'--background','--factory-startup',*thread_arguments(),
                '--python-exit-code','1','--python',str(Path(__file__).with_name('auto_wrap_worker.py')),
                '--',str(folder/'request.json')],stdout=log,stderr=subprocess.STDOUT,
                creationflags=(subprocess.CREATE_NO_WINDOW|subprocess.BELOW_NORMAL_PRIORITY_CLASS) if os.name=='nt' else 0)
    except Exception:
        cleanup({'folder':folder})
        raise
    _JOB={'scene':context.scene,'window':window,'folder':folder,'process':process,'state':stamp,
        'wrap':wrap,'cancel':False,'started':time.monotonic()}
    from . import process_ref
    try:
        (folder/'ownership.json').write_text(json.dumps({'kind':'CREATE_MH_AUTO_WRAP_V1',
            'parent':process_ref.identify(os.getpid()),'worker':process_ref.identify(process.pid)}),encoding='utf-8')
    except OSError:
        cancel()
        if not bpy.app.timers.is_registered(poll):bpy.app.timers.register(poll,first_interval=.2,persistent=True)
        raise
    s.status='Detecting landmarks for Auto Wrap. The accepted character is retained.'
    f.last_action_error=False
    if not bpy.app.timers.is_registered(poll):bpy.app.timers.register(poll,first_interval=.2,persistent=True)


def publish(context,result):
    from . import add_marker
    s,f=context.scene.cm_settings,context.scene.cm_features
    pairs=result['pairs']
    matrix=Matrix(result['source_matrix'])
    if len(pairs)<3 or not np.isfinite(np.array(matrix)).all():raise ValueError('Detection returned incomplete alignment')
    previous=m.transform_state(s.source)
    axis=s.source.get('cm_front_axis')
    placement=s.source.get('cm_head_placement')
    old=[(p.name,p.source,p.template) for p in s.landmarks]
    made=[];prepared=[]
    try:
        s.source.matrix_world=matrix
        if not result.get('body_warning'):s.source['cm_front_axis']=f.front_axis
        if s.source.get('cm_gnm') and 'cm_head_placement' in s.source:del s.source['cm_head_placement']
        context.view_layer.update()
        for pair in pairs:
            parent=f.body_template if pair['part']=='BODY' else s.template
            if not parent:raise ValueError('Detected template part is unavailable')
            positions=np.array([pair['source'],pair['template']])
            if not np.isfinite(positions).all():raise ValueError('Detected point is invalid')
            marks=[]
            for obj,position in ((s.source,positions[0]),(parent,positions[1])):
                mark=add_marker(context,obj,obj.matrix_world.inverted()@Vector(position),'CM_Auto')
                made.append(mark);marks.append(mark)
            prepared.append((pair['name'],*marks))
        s.landmarks.clear()
        for name,source,template in prepared:
            pair=s.landmarks.add();pair.name=name;pair.source=source;pair.template=template
    except Exception:
        m.restore_transform(s.source,previous)
        if axis is None:
            if 'cm_front_axis' in s.source:del s.source['cm_front_axis']
        else:s.source['cm_front_axis']=axis
        if placement is not None:s.source['cm_head_placement']=placement
        s.landmarks.clear()
        for name,source,template in old:
            pair=s.landmarks.add();pair.name=name;pair.source=source;pair.template=template
        for obj in made:bpy.data.objects.remove(obj,do_unlink=True)
        raise
    for obj in set(obj for _,a,b in old for obj in (a,b) if obj):
        if obj.type=='EMPTY' and obj.get('cm_landmark'):bpy.data.objects.remove(obj,do_unlink=True)
    for index,(_,source,template) in enumerate(prepared,1):
        source.name=f'{index:02d} Source';template.name=f'{index:02d} Template'
    f.landmarks_reviewed=False
    context.view_layer.update()
    s.source['cm_auto_landmarks_state']=json.dumps(landmark_geometry(context))
    s.source['cm_face_landmarks_warning']=result.get('face_warning','')
    s.source['cm_body_landmarks_warning']=result.get('body_warning','')


def poll():
    global _JOB
    job=_JOB
    if not job:return None
    if job['process'].poll() is None:return .2
    if job.get('stopper') and job['stopper'].poll() is None:return .2
    _JOB=None
    status='failed';error=''
    try:
        scene=job.get('scene')
        if job['cancel'] or not scene:
            status='canceled'
            if scene:scene.cm_settings.status='Auto Wrap canceled. Previous character and landmarks retained.'
        else:
            window=job['window']
            if not window or window.scene!=scene:raise ValueError('Character window changed; detection was not applied')
            with bpy.context.temp_override(window=window):
                from . import landmark_tools,paint_tools,detail_mask
                if bpy.context.mode!='OBJECT' or landmark_tools.active() or paint_tools.running() or detail_mask.running(scene) or state(bpy.context,scene)!=job['state']:
                    raise ValueError('Character inputs changed during detection. Previous character and landmarks retained.')
                path=job['folder']/'result.json'
                result=json.loads(path.read_text(encoding='utf-8')) if path.is_file() else {}
                if job['process'].returncode or result.get('status')!='complete':
                    raise ValueError(result.get('error','Detection failed; see '+str(job['folder']/'worker.log')))
                publish(bpy.context,result)
                if not bpy.app.background and bpy.context.preferences.edit.use_global_undo:
                    bpy.ops.ed.undo_push(message='Auto landmarks detected')
                if job['wrap'] and not result.get('body_warning'):
                    if bpy.ops.cm.guided(action='WRAP')!={'FINISHED'}:
                        raise ValueError('Fit failed. Detected points are retained for manual correction.')
                else:
                    from .guided import focus
                    s=scene.cm_settings
                    focus(bpy.context,[s.source]+[p.source for p in s.landmarks],scene.cm_features.front_axis)
                    s.status=str(len(s.landmarks))+' pairs detected. Edit points or Wrap when ready.'
                if result.get('face_warning'):
                    scene.cm_settings.status += ' ' + result['face_warning'] + '. Review the head fit manually.'
                if result.get('body_warning'):
                    scene.cm_settings.status=result['body_warning']+'. Reference frame and accepted character retained.'
                if not bpy.app.background and bpy.context.preferences.edit.use_global_undo:
                    bpy.ops.ed.undo_push(message='Auto Wrap complete' if job['wrap'] else 'Auto detection complete')
                status='needs_review' if result.get('body_warning') else 'complete'
    except Exception as exc:
        error=str(exc)
        if job.get('scene'):
            job['scene'].cm_settings.status=error
            job['scene'].cm_features.last_action_error=True
    finally:
        cleanup(job)
        (job['folder']/'receipt.json').write_text(json.dumps({'status':status,'error':error,
            'seconds':time.monotonic()-job['started'],'worker_pid':job['process'].pid}),encoding='utf-8')
    return None


def cancel():
    if _JOB and not _JOB['cancel']:
        _JOB['cancel']=True
        process=_JOB['process']
        if process.poll() is None:
            if os.name=='nt':
                _JOB['stopper']=subprocess.Popen(['taskkill','/PID',str(process.pid),'/T','/F'],
                    stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,creationflags=subprocess.CREATE_NO_WINDOW)
            else:process.terminate()


@persistent
def detach(_=None):
    cancel()
    if _JOB:_JOB['scene']=None


@persistent
def recover(_=None):
    from . import process_ref
    try:
        root=Path(bpy.path.abspath(bpy.context.scene.cm_settings.project_dir)).resolve()/'Automation'/'.jobs'
        if not root.is_dir():return None
        for path in root.glob('*/ownership.json'):
            try:
                record=json.loads(path.read_text(encoding='utf-8'))
                if record.get('kind')!='CREATE_MH_AUTO_WRAP_V1':continue
                if process_ref.alive(record.get('parent')) or process_ref.alive(record.get('worker')):continue
                folder=path.parent.resolve()
                if folder.parent!=root:continue
                cleanup({'folder':folder})
            except (OSError,ValueError,KeyError):continue
    except (AttributeError,ReferenceError,OSError):pass
    return None


class CM_OT_cancel_auto_wrap(bpy.types.Operator):
    bl_idname='cm.cancel_auto_wrap'
    bl_label='Cancel Auto Wrap'
    def execute(self,context):
        cancel();context.scene.cm_settings.status='Canceling Auto Wrap...'
        return {'FINISHED'}


def register():
    for handlers in (bpy.app.handlers.load_pre,bpy.app.handlers.undo_pre,bpy.app.handlers.redo_pre):
        if detach not in handlers:handlers.append(detach)
    if recover not in bpy.app.handlers.load_post:bpy.app.handlers.load_post.append(recover)
    if not bpy.app.timers.is_registered(recover):bpy.app.timers.register(recover,first_interval=.5)


def unregister():
    detach()
    for handlers in (bpy.app.handlers.load_pre,bpy.app.handlers.undo_pre,bpy.app.handlers.redo_pre):
        if detach in handlers:handlers.remove(detach)
    if recover in bpy.app.handlers.load_post:bpy.app.handlers.load_post.remove(recover)
    if bpy.app.timers.is_registered(recover):bpy.app.timers.unregister(recover)


CLASSES=(CM_OT_cancel_auto_wrap,)
