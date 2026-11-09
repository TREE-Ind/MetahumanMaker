"""Blender controls for reversible native UE facial inspection."""
import ast,importlib.util,json,shutil,threading,time,uuid
from pathlib import Path
import bpy
from bpy.app.handlers import persistent
from bpy.props import StringProperty,EnumProperty,FloatProperty,IntProperty,BoolProperty,PointerProperty,CollectionProperty
from . import process_ref,expression_controls,native_face_mixer

_JOB=None
_REMOTE=None
_DIRTY=None
_SHUTTING=False
_ITEMS={}
_REMOTE_LOCK=threading.RLock()
_SYNCING=False


def running():return _JOB is not None


def sculpt_pose_items(self,context):
    from .native_pose_sculpt import pose_items
    return pose_items(self,context)


def repair_control_items(self,context):
    try:rows=json.loads(self.capture_result or '{}').get('poses',[])
    except ValueError:rows=[]
    controls=set()
    for row in rows:
        values=row.get('controls',{})
        if len(values)!=1:continue
        name=next(iter(values)).removeprefix('CTRL_expressions.')
        if name.startswith('eye'):controls.add(name)
    ordered=sorted(name for name in controls if not name.startswith('eyeBlink'))+sorted(name for name in controls if name.startswith('eyeBlink'))
    key=('repair_controls',tuple(ordered))
    if key not in _ITEMS:
        _ITEMS[key]=[('SELECT','Choose Eye Control','Select the captured control to repair')]+[(name,name,'Repair all applicable captured strengths for this control') for name in ordered]
    return _ITEMS[key]


def result(settings):
    try:return json.loads(settings.result or '{}')
    except ValueError:return {}


def poses(settings):
    raw=result(settings).get('raw_controls',[])
    return [dict(name='Neutral',controls={})]+expression_controls.inspection_poses(raw)


def pose_items(self,context):
    rows=poses(self);key=('poses',tuple(row['name'] for row in rows))
    if key not in _ITEMS:_ITEMS[key]=[(str(i),row['name'],'Evaluate this native control combination') for i,row in enumerate(rows)]
    return _ITEMS[key]


def control_items(self,context):
    names=result(self).get('raw_controls',[]);key=('controls',tuple(names))
    if key not in _ITEMS:_ITEMS[key]=[(str(i),name.removeprefix('CTRL_expressions.'),'Native raw facial control') for i,name in enumerate(names)] or [('0','No character selected','Connect an assembled Unreal character')]
    return _ITEMS[key]


def controls(settings,*,mode=None):
    mode=mode or settings.input_mode
    if mode=='MIXER':values=native_face_mixer.values(settings)
    elif mode=='JSON':
        values=json.loads(settings.controls_json)
        if not isinstance(values,dict):raise ValueError('Combined controls must be a JSON object')
    elif mode=='SINGLE':
        names=result(settings).get('raw_controls',[]);index=int(settings.control)
        if not 0<=index<len(names):raise ValueError('Choose a supported facial control')
        values={names[index]:settings.strength}
    else:
        rows=poses(settings);index=int(settings.pose)
        if not 0<=index<len(rows):raise ValueError('Choose a supported facial pose')
        values={key:value*settings.strength for key,value in rows[index]['controls'].items()}
    import math
    if any(type(key) is not str or type(value) not in (int,float) or not math.isfinite(value) or not 0<=value<=1 for key,value in values.items()):raise ValueError('Control values must be finite numbers between 0 and 1')
    return values


def queue_apply(self,context):
    global _DIRTY
    if _SYNCING:return
    if context and self.live and result(self).get('mode')=='ACTIVE':
        due=_DIRTY[1] if _DIRTY and _DIRTY[0]==context.scene else time.monotonic()+.12
        _DIRTY=(context.scene,due)
        if not bpy.app.timers.is_registered(flush):bpy.app.timers.register(flush,first_interval=.2)


def queue_look(self,context):
    if self.look_enabled:queue_apply(self,context)


def look_values(settings):
    if not settings.look_enabled:return None
    values=dict(normal=settings.look_normal)
    if (result(settings).get('look_values') or {}).get('micro') is not None:values['micro']=settings.look_micro
    return values


def flush():
    global _DIRTY
    if not _DIRTY:return None
    scene,when=_DIRTY
    if time.monotonic()<when or running():return .1
    _DIRTY=None
    if scene.name not in bpy.data.scenes:return None
    g=scene.cm_face_review
    if g.live and result(g).get('mode')=='ACTIVE':
        try:start(scene,'APPLY')
        except Exception as error:g.status=str(error)
    return .1 if _DIRTY else None


def _close_remote():
    global _REMOTE
    with _REMOTE_LOCK:
        remote,_REMOTE=_REMOTE,None
        if remote:
            try:remote['client'].stop()
            except Exception:pass


def release_transport():
    """Other add-on exchanges can take the single native command socket."""
    _close_remote()


def _idle_release(remote,stamp):
    with _REMOTE_LOCK:
        if _REMOTE is remote and remote.get('last_used')==stamp:_close_remote()


def _connection(engine,project):
    """Reuse one command connection so slider updates do not rediscover UE."""
    global _REMOTE
    key=(engine,project)
    if _REMOTE and _REMOTE['key']==key:return _REMOTE['client']
    _close_remote();path=Path(engine)/'Engine/Plugins/Experimental/PythonScriptPlugin/Content/Python/remote_execution.py'
    spec=importlib.util.spec_from_file_location('cm_face_review_remote',path);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);client=module.RemoteExecution()
    try:
        client.start();deadline=time.monotonic()+4
        while time.monotonic()<deadline:time.sleep(.1)
        nodes=[n for n in client.remote_nodes if not project or n.get('project_name')==project]
        if len(nodes)!=1:raise ValueError('Keep one matching Unreal editor open with Python Remote Execution enabled')
        client.open_command_connection(nodes[0]['node_id']);_REMOTE=dict(client=client,key=key);return client
    except Exception:client.stop();raise


def _send(job):
    folder=job['folder'];request=job['request']
    code='\n'.join(['import json,runpy,os','from pathlib import Path','folder=Path('+repr(str(folder))+')',"identity=runpy.run_path(str(folder/'process_ref.py'))['identify'](os.getpid())","(folder/'dispatch.json').write_text(json.dumps(identity))",'delegated=False','try:',"    report=runpy.run_path(str(folder/'unreal_face_review.py'))['dispatch'](folder,"+repr(request)+')',"    delegated=bool(report.get('receipt_delegated'))",'except Exception as error:',"    report=dict(status='failed',error=str(error))",'if not delegated:',"    temporary=folder/'receipt.tmp'","    temporary.write_text(json.dumps(report))","    temporary.replace(folder/'receipt.json')"])
    with _REMOTE_LOCK:
        try:
            client=_connection(request['engine'],request['project']);job['dispatched'].set();client.run_command(code,unattended=True,raise_on_failure=True)
        except Exception as error:job['error']=str(error);_close_remote()
        finally:
            if _SHUTTING:_close_remote()
            elif _REMOTE:
                stamp=time.monotonic();_REMOTE['last_used']=stamp
                timer=threading.Timer(.6,_idle_release,args=(_REMOTE,stamp));timer.daemon=True;timer.start()


def start(scene,action):
    global _JOB,_DIRTY
    from . import groom_jobs,unreal_jobs,skin_native_jobs
    if running() or groom_jobs.running() or unreal_jobs.running() or skin_native_jobs.running():raise ValueError('Wait for the current Unreal operation before changing facial review')
    s=scene.cm_settings;g=scene.cm_face_review;engine=bpy.path.abspath(s.unreal_engine_dir)
    if not (Path(engine)/'Engine/Plugins/Experimental/PythonScriptPlugin/Content/Python/remote_execution.py').is_file():raise ValueError('Choose the Unreal installation containing Engine/')
    if g.receipt:
        pending=Path(g.receipt)
        if not pending.is_file() and (pending.parent/'request.json').is_file():resume(scene);raise ValueError('Observing the preceding facial request; it will not be dispatched twice')
    request=dict(action=action,engine=engine,project=s.unreal_project)
    if action!='START':
        state=result(g)
        if state.get('mode')!='ACTIVE' and action!='UNDO_LOOK':raise ValueError('Connect the assembled Unreal character first')
        request['origin']=state['origin']
    if action=='APPLY':request.update(controls=controls(g),lod=g.lod,look=look_values(g))
    if action=='CAPTURE_LIDS':
        current=controls(g) if g.capture_set=='CURRENT' else None
        samples=expression_controls.lid_capture_poses(state['raw_controls'],g.capture_set,current)
        request.update(capture_set=g.capture_set,capture_controls=current,capture_count=len(samples))
        _DIRTY=None
    if action in {'PREPARE_LIDS','PREPARE_SCULPT','PREPARE_BANK','PREPARE_INTERVALS','PREPARE_NEUTRAL'}:
        capture=state.get('calibration_capture') or {}
        if action=='PREPARE_LIDS' and (capture.get('status')!='complete' or capture.get('capture_set') not in {'LIDS','LIDS_FINE','LIDS_DENSE','LIDS_INTERVALS','BLINK_FINE','BLINK_DENSE','GAZE_UP','GAZE_DOWN','GAZE_LEFT','GAZE_RIGHT'}):
            raise ValueError('Capture a complete blink/squint or gaze grid first')
        python=bpy.path.abspath(scene.cm_features.inference_python)
        if not Path(python).is_file():raise ValueError('Choose the local inference Python with NumPy and SciPy in Setup')
        request['python']=python;_DIRTY=None
    if action in {'PREPARE_BANK','PREPARE_INTERVALS'}:
        if g.corrective_object:raise ValueError('Finish or retain the current sculpt before preparing automatic intermediate repairs')
        capture=Path(bpy.path.abspath(g.capture_folder))/'capture.json';rows=json.loads(capture.read_text())['poses']
        if g.repair_control!='SELECT':
            if g.repair_control not in {item[0] for item in repair_control_items(g,None)}:raise ValueError('Choose an eye control from the current capture')
            selected={g.repair_control:1.}
        else:
            # Retain the captured-pose API used by existing agent recipes.
            try:selected=rows[int(g.sculpt_pose)]['controls']
            except (ValueError,IndexError):raise ValueError('Choose a captured eye pose to select the repair control') from None
        if len(selected)!=1:raise ValueError('Choose one captured eye control; the repair set uses its intermediate strengths')
        driver=next(iter(selected)).removeprefix('CTRL_expressions.')
        if not driver.startswith('eye'):raise ValueError('Choose an eye-control capture')
        base=Path(bpy.path.abspath(g.preparation_folder))/'manifest.json'
        if not base.is_file():raise ValueError('Prepare lid anatomy before preparing intermediate repairs')
        request['repair_bank']=dict(capture_folder=str(capture.parent.resolve()),base_manifest=str(base.resolve()),driver=driver,intervals=action=='PREPARE_INTERVALS')
    if action=='PREPARE_NEUTRAL':
        if g.corrective_object:raise ValueError('Finish or retain the current sculpt before fitting neutral lids')
        base=Path(bpy.path.abspath(g.preparation_folder))/'manifest.json'
        if not base.is_file():raise ValueError('Prepare lid anatomy before fitting neutral lids')
        request['neutral_fit']=dict(capture_folder=str(Path(bpy.path.abspath(g.capture_folder)).resolve()),base_manifest=str(base.resolve()))
    if action in {'STAGE_LIDS','OPEN_STAGED','COMPARE_STAGED','REVISE_BOUNDS'}:
        prepared=Path(bpy.path.abspath(g.preparation_folder)) if g.preparation_folder else None
        if not prepared or not (prepared/'manifest.json').is_file():raise ValueError('Prepare a lid correction first')
        request['preparation_folder']=str(prepared.resolve());_DIRTY=None
    if action=='COMPARE_STAGED':request.update(spacing=g.comparison_spacing,lod=g.lod,live_background=g.comparison_background)
    if action=='SAVE_LOOK':
        if not g.look_enabled:raise ValueError('Enable skin preview before saving tuning')
        request['look']=look_values(g);_DIRTY=None
    if action not in {'START','APPLY','RESTORE','SAVE_LOOK','UNDO_LOOK','CAPTURE_LIDS','PREPARE_LIDS','PREPARE_SCULPT','PREPARE_BANK','PREPARE_INTERVALS','PREPARE_NEUTRAL','STAGE_LIDS','OPEN_STAGED','COMPARE_STAGED','END_COMPARISON','FRAME_COMPARISON','REVISE_BOUNDS'}:raise ValueError('Unknown facial review operation')
    folder=Path(bpy.path.abspath(s.project_dir))/'FaceReview'/'.exchange'/uuid.uuid4().hex;folder.mkdir(parents=True,exist_ok=False)
    if action=='PREPARE_SCULPT':
        from .native_pose_sculpt import export
        request['sculpt_snapshot']=export(bpy.context,folder)
    names={'unreal_face_review','unreal_face_comparison','unreal_revision_bounds','unreal_actor_preview','process_ref','unreal_face_look','unreal_skin_normal_response','unreal_skin_repair','unreal_lod_correctives','capture_storage','expression_controls','resource_budget'}
    if action in {'STAGE_LIDS','REVISE_BOUNDS'}:names.update({'unreal_lid_staging','unreal_globe_rigidity','unreal_groom'})
    if action in {'PREPARE_LIDS','PREPARE_SCULPT','PREPARE_BANK','PREPARE_INTERVALS','PREPARE_NEUTRAL'}:
        pending=['unreal_lid_preparation','lid_calibration_worker','lid_calibration','native_pose_sculpt_solver','native_lid_repair_bank','native_interval_fields','native_neutral_lid_fit'];seen=set()
        while pending:
            name=pending.pop();path=Path(__file__).with_name(name+'.py')
            if name in seen or not path.is_file():continue
            seen.add(name);names.add(name)
            for node in ast.walk(ast.parse(path.read_text(encoding='utf-8'))):
                if isinstance(node,ast.Import):pending.extend(alias.name.split('.')[-1] for alias in node.names)
                elif isinstance(node,ast.ImportFrom):
                    if node.module:pending.append(node.module.split('.')[-1])
                    elif node.level:pending.extend(alias.name for alias in node.names)
    for name in names:shutil.copy2(Path(__file__).with_name(name+'.py'),folder/(name+'.py'))
    (folder/'request.json').write_text(json.dumps(request));job=dict(scene=scene,folder=folder,request=request,dispatched=threading.Event(),error=None)
    job['thread']=threading.Thread(target=_send,args=(job,),daemon=True);_JOB=job;g.receipt=str(folder/'receipt.json');g.status='Connecting to the selected Unreal character...' if action=='START' else 'Restoring previous playback...' if action=='RESTORE' else 'Updating the native facial pose...'
    if action=='RESTORE':_DIRTY=None
    if action=='CAPTURE_LIDS':g.status='Capturing '+str(request['capture_count'])+' native LOD0 eye poses. Previous playback is retained.'
    if action in {'PREPARE_LIDS','PREPARE_SCULPT','PREPARE_BANK','PREPARE_INTERVALS','PREPARE_NEUTRAL'}:g.status='Preparing a separate native lid correction proposal...'
    if action=='STAGE_LIDS':g.status='Staging a separate native face and Blueprint with matching groom bindings...'
    if action=='REVISE_BOUNDS':g.status='Creating and validating a separate render-bounds revision...'
    job['thread'].start()
    if not bpy.app.timers.is_registered(poll):bpy.app.timers.register(poll,first_interval=.1,persistent=True)


def poll():
    global _JOB,_SYNCING
    job=_JOB
    if not job:return None
    receipt=job['folder']/'receipt.json';scene=job.get('scene')
    if scene and (scene.name not in bpy.data.scenes or scene.cm_face_review.receipt!=str(receipt)):scene=None;job['scene']=None
    if not receipt.is_file():
        if scene and job['request']['action'] in {'PREPARE_LIDS','PREPARE_SCULPT','PREPARE_BANK','PREPARE_INTERVALS','PREPARE_NEUTRAL'}:
            try:scene.cm_face_review.status=json.loads((job['folder']/'progress.json').read_text())['phase']
            except (OSError,ValueError,KeyError):pass
        thread=job.get('thread')
        if thread and thread.is_alive():
            if scene and job['request']['action']=='CAPTURE_LIDS':
                try:
                    progress=json.loads((job['folder']/'capture/capture.json').read_text())
                    scene.cm_face_review.status='Capturing native eye poses: '+str(len(progress.get('poses',[])))+' / '+str(job['request']['capture_count'])+'.'
                except (OSError,ValueError):pass
            return .1
        dispatch=job['folder']/'dispatch.json'
        if dispatch.is_file():
            try:owner=json.loads(dispatch.read_text())
            except (OSError,ValueError):return .2
            if process_ref.alive(owner):return .2
            report=dict(status='failed',error='The original Unreal editor closed before confirming this request')
        elif job['dispatched'].is_set():
            if scene:scene.cm_face_review.status='Waiting for the original request receipt; no duplicate will be sent.'
            return .5
        else:report=dict(status='failed',error=job['error'] or 'No native execution was confirmed')
        receipt.write_text(json.dumps(report))
    try:report=json.loads(receipt.read_text())
    except (OSError,ValueError):return .1
    # The worker owns the cached socket until run_command returns.
    if job.get('thread') and job['thread'].is_alive():return .1
    if scene:
        g=scene.cm_face_review
        scene.cm_features.last_action_error=report.get('status') not in {'complete','cancelled'}
        if report.get('mode') in {'ACTIVE','RESTORED'}:
            g.result=json.dumps(report)
            if report.get('mode')=='ACTIVE':native_face_mixer.ensure(g)
            if report.get('calibration_capture'):
                changed_capture=g.capture_folder!=report['calibration_capture_folder']
                g.capture_result=json.dumps(report['calibration_capture'])
                g.capture_folder=report['calibration_capture_folder']
                if changed_capture:
                    g.repair_control='SELECT'
                    if any(item[0]=='SELECT' for item in sculpt_pose_items(g,None)):g.sculpt_pose='SELECT'
            if report.get('calibration_preparation'):
                g.preparation_result=json.dumps(report['calibration_preparation'])
                g.preparation_folder=report['calibration_preparation']['folder']
            if report.get('calibration_staging'):g.staging_result=json.dumps(report['calibration_staging'])
            if job['request']['action'] in {'START','RESTORE','SAVE_LOOK','UNDO_LOOK'}:
                _SYNCING=True
                try:
                    g.look_enabled=bool(report.get('look_preview'));skin=report.get('look_values') or {}
                    if skin.get('normal') is not None:g.look_normal=skin['normal']
                    if skin.get('micro') is not None:g.look_micro=skin['micro']
                finally:_SYNCING=False
        if report.get('status') in {'complete','cancelled'}:
            g.status=report.get('message') or ('Previous playback and skin preview restored. No assets saved.' if report['mode']=='RESTORED' else 'Reviewing '+report['actor_label']+'. Changes affect only the live Unreal actor.')
        else:g.status='Facial review: '+report.get('error','Inspect Unreal Output Log')
        scene.cm_settings.status=g.status
    _JOB=None
    for window in bpy.context.window_manager.windows:
        for area in window.screen.areas:area.tag_redraw()
    return None


def resume(scene):
    global _JOB
    path=Path(scene.cm_face_review.receipt)
    if not (path.parent/'request.json').is_file():raise ValueError('No saved facial review request to observe')
    if _JOB:
        if _JOB['folder']!=path.parent:raise ValueError('Another facial request is being observed')
        _JOB['scene']=scene
    else:
        sent=threading.Event();sent.set();_JOB=dict(scene=scene,folder=path.parent,request=json.loads((path.parent/'request.json').read_text()),dispatched=sent,thread=None,error=None)
    if not bpy.app.timers.is_registered(poll):bpy.app.timers.register(poll,first_interval=.1,persistent=True)


@persistent
def detach(*_):
    global _DIRTY
    _DIRTY=None
    if _JOB:_JOB['scene']=None


@persistent
def loaded(*_):
    scene=bpy.context.scene
    if hasattr(scene,'cm_face_review') and scene.cm_face_review.receipt:
        try:resume(scene)
        except (OSError,ValueError) as error:scene.cm_face_review.status=str(error)


class CM_FaceReviewSettings(bpy.types.PropertyGroup):
    receipt:StringProperty(options={'HIDDEN'})
    result:StringProperty(options={'HIDDEN'})
    status:StringProperty()
    capture_set:EnumProperty(name='Eye Pose Set',default='LIDS',items=[
        ('LIDS','Blink and Squint (17 poses)','Independent three-by-three grids for both eyes'),
        ('LIDS_FINE','Detailed Blink and Squint (29 poses)','Quarter-strength squint samples and three blink strengths for both eyes; captures the current saved rig'),
        ('LIDS_INTERVALS','Interval Blink and Squint (101 poses)','Sixteenth-strength sampling for anchor-preserving interval repair; streamed into one owned capture'),
        ('LIDS_DENSE','Dense Blink and Squint (53 poses)','Eighth-strength squint samples with three blink strengths for both eyes'),
        ('GAZE_UP','Upward Gaze (53 poses)','Independent squint, blink and upward-gaze grids'),
        ('GAZE_DOWN','Downward Gaze (53 poses)','Independent squint, blink and downward-gaze grids'),
        ('GAZE_LEFT','Leftward Gaze (53 poses)','Independent squint, blink and leftward-gaze grids'),
        ('GAZE_RIGHT','Rightward Gaze (53 poses)','Independent squint, blink and rightward-gaze grids'),
        ('CURRENT','Current Controls and Neutral','Capture the selected control values and neutral, retaining live playback'),
        ('BLINK_FINE','Detailed Blink Strengths (29 poses)','Quarter-strength blink samples with three squint strengths for both eyes'),
        ('BLINK_DENSE','Dense Blink Strengths (53 poses)','Eighth-strength blink samples with three squint strengths for both eyes')])
    capture_result:StringProperty(options={'HIDDEN'})
    capture_folder:StringProperty(subtype='DIR_PATH')
    preparation_result:StringProperty(options={'HIDDEN'})
    preparation_folder:StringProperty(subtype='DIR_PATH')
    staging_result:StringProperty(options={'HIDDEN'})
    corrective_object:PointerProperty(type=bpy.types.Object)
    corrective_eye_left:PointerProperty(type=bpy.types.Object,options={'HIDDEN'})
    corrective_eye_right:PointerProperty(type=bpy.types.Object,options={'HIDDEN'})
    sculpt_state:StringProperty(options={'HIDDEN'})
    sculpt_pose:EnumProperty(name='Captured Pose',items=sculpt_pose_items)
    closed_sculpt_scope:EnumProperty(name='Closed-Eye Region',items=[('LOCAL','Local Lid','Eight exterior mesh rings; preserves the existing local sculpt workflow'),('SURFACE','Physical Eye Region','A broader region measured on the neutral surface at 45% of the captured eye width. Applies when opening a full-blink sculpt; canthi and unrelated anatomy remain protected')],default='LOCAL',description='Choose coverage before opening a full-blink sculpt. Existing session masks and the 35% movement limit are retained')
    repair_control:EnumProperty(name='Repair Control',items=repair_control_items)
    repair_surface:BoolProperty(name='Untangle Captured Lid (Experimental)',default=False,description='Repair the visible lid and continue its hidden socket before preparing a separate corrective. Requires anatomy derived from captured render triangles. Neutral and globes retained; compiled motion still needs review')
    comparison_spacing:FloatProperty(name='Comparison Spacing (cm)',default=0.,min=0.,max=300.,description='0 fits spacing to the face bounds; otherwise use 20 to 300 cm')
    comparison_background:BoolProperty(name='Keep UE Preview Live',default=True,description='Temporarily permit background viewport rendering during comparison; restore the previous preference on ending, without saving configuration')
    mixer_controls:CollectionProperty(type=native_face_mixer.CM_FaceMixerControl)
    mixer_owner:StringProperty(options={'HIDDEN'})
    mixer_pair:BoolProperty(name='Link L/R',default=False,description='Subsequent slider edits also change the matching opposite control; current asymmetric values are retained')
    mixer_region:EnumProperty(name='Region',items=native_face_mixer.REGIONS,default='EYES')
    mixer_search:StringProperty(name='Find Control',description='Filter native face controls by name')
    mixer_active_only:BoolProperty(name='Active Only',default=False)
    mixer_page:IntProperty(default=0,min=0,options={'HIDDEN'})
    input_mode:EnumProperty(name='Controls',items=[('PRESET','Poses','Named inspection combinations'),('SINGLE','Individual','Inspect one supported raw control'),('JSON','Combined','Explicit JSON controls for an artist or agent'),('MIXER','Mixer','Combine native facial controls with sliders')],update=native_face_mixer.update_mode)
    pose:EnumProperty(name='Pose',items=pose_items,update=queue_apply)
    control:EnumProperty(name='Control',items=control_items,update=queue_apply)
    strength:FloatProperty(name='Strength',default=1.,min=0.,max=1.,update=queue_apply)
    controls_json:StringProperty(name='Combined Controls',default='{}',description='JSON object of native raw control names and values from 0 to 1',update=queue_apply)
    live:BoolProperty(name='Live Updates',default=True,update=queue_apply)
    look_enabled:BoolProperty(name='Preview Skin Response',default=False,update=queue_apply)
    look_normal:FloatProperty(name='Surface Detail',default=1.,min=0.,soft_max=2.,description='Native head normal response; micro skin remains a separate control',update=queue_look)
    look_micro:FloatProperty(name='Micro Skin',default=.7,min=0.,soft_max=2.,description='Native fine skin normal response',update=queue_look)
    lod:IntProperty(name='Face LOD',default=-1,min=-1,max=7,description='-1 retains the current setting; 0–7 force the selected native face LOD',update=queue_apply)


class CM_OT_native_face_review(bpy.types.Operator):
    bl_idname='cm.native_face_review'
    bl_label='Review Native Facial Controls'
    action:EnumProperty(items=[(key,key.title(),'') for key in ('START','APPLY','RESTORE','REFRESH','SAVE_LOOK','UNDO_LOOK','CAPTURE_LIDS','OPEN_CAPTURE','PREPARE_LIDS','CANCEL_PREPARE','OPEN_PREPARATION','STAGE_LIDS','OPEN_STAGED','COMPARE_STAGED','END_COMPARISON','FRAME_COMPARISON','EDIT_CAPTURED_POSE','FINISH_SCULPT','SCULPT_POSE','FRAME_SCULPT','DISCARD_SCULPT','PREPARE_SCULPT','PREPARE_BANK','PREPARE_INTERVALS','REVISE_BOUNDS','PREPARE_NEUTRAL')])
    def execute(self,context):
        try:
            if self.action in {'EDIT_CAPTURED_POSE','FINISH_SCULPT','SCULPT_POSE','FRAME_SCULPT','DISCARD_SCULPT'}:
                if running():raise ValueError('Wait for the current native review operation')
                from . import native_pose_sculpt
                if self.action=='EDIT_CAPTURED_POSE':native_pose_sculpt.begin(context)
                elif self.action=='SCULPT_POSE':native_pose_sculpt.sculpt(context)
                elif self.action=='FRAME_SCULPT':
                    framed=native_pose_sculpt.frame(context)
                    context.scene.cm_face_review.status='Corrective framed in the 3D view.' if framed else 'No 3D viewport available; corrective retained.'
                else:native_pose_sculpt.finish(context,discard=self.action=='DISCARD_SCULPT')
            elif self.action=='CANCEL_PREPARE':
                job=_JOB
                if not job or job.get('scene')!=context.scene or job['request']['action'] not in {'PREPARE_LIDS','PREPARE_SCULPT','PREPARE_BANK','PREPARE_INTERVALS','PREPARE_NEUTRAL'}:raise ValueError('No lid preparation is running for this project')
                if (job['folder']/'receipt.json').is_file():raise ValueError('Preparation already finished; checking its original result')
                (job['folder']/'cancel.flag').write_text('cancel')
                context.scene.cm_face_review.status='Cancelling the owned numerical preparation; waiting for it to stop...'
            elif self.action in {'OPEN_CAPTURE','OPEN_PREPARATION'}:
                path=context.scene.cm_face_review.capture_folder if self.action=='OPEN_CAPTURE' else context.scene.cm_face_review.preparation_folder
                if not path:raise ValueError('Capture native eye poses first' if self.action=='OPEN_CAPTURE' else 'Prepare a lid correction first')
                folder=Path(bpy.path.abspath(path))
                if not folder.is_dir():raise ValueError('The saved calibration folder is unavailable')
                bpy.ops.wm.path_open(filepath=str(folder))
            elif self.action=='REFRESH':resume(context.scene)
            else:start(context.scene,self.action)
        except Exception as error:
            context.scene.cm_face_review.status=str(error);self.report({'ERROR'},str(error));return {'CANCELLED'}
        return {'FINISHED'}


def draw(layout,context):
    g=context.scene.cm_face_review;header,box=layout.panel('cm_native_face_review',default_closed=True);header.label(text='Review Unreal Face',icon='ARMATURE_DATA')
    if box is None:return
    row=box.row();row.enabled=not running();row.operator('cm.native_face_review',text='Use Selected Unreal Character',icon='EYEDROPPER').action='START'
    state=result(g)
    if state.get('mode')=='ACTIVE':
        box.label(text=state['actor_label']);box.prop(g,'input_mode',expand=True)
        if state.get('comparison'):
            box.label(text='Comparing original and staged rig; controls drive both.')
            row=box.row();row.enabled=not running();row.operator('cm.native_face_review',text='Frame Both Faces',icon='VIEW_CAMERA').action='FRAME_COMPARISON'
            row=box.row();row.enabled=not running();row.operator('cm.native_face_review',text='End Comparison',icon='X').action='END_COMPARISON'
        if g.input_mode=='PRESET':box.prop(g,'pose')
        elif g.input_mode=='SINGLE':box.prop(g,'control')
        elif g.input_mode=='MIXER':native_face_mixer.draw(box,g)
        else:box.prop(g,'controls_json')
        if g.input_mode in {'PRESET','SINGLE'}:box.prop(g,'strength',slider=True)
        box.prop(g,'live');row=box.row();row.enabled=not running();row.operator('cm.native_face_review',text='Apply Pose').action='APPLY'
        row.operator('cm.native_face_review',text='Restore Previous Playback',icon='LOOP_BACK').action='RESTORE'
        if state.get('look_values'):
            skin=box.box();skin.enabled=not bool(state.get('comparison'));skin.label(text='Head Skin');skin.prop(g,'look_enabled')
            values=skin.column();values.enabled=g.look_enabled;values.prop(g,'look_normal',slider=True)
            if state['look_values'].get('micro') is not None:values.prop(g,'look_micro',slider=True)
            row=skin.row();row.enabled=g.look_enabled and not running();row.operator('cm.native_face_review',text='Save Skin Tuning',icon='FILE_TICK').action='SAVE_LOOK'
            skin.label(text='Saves a material copy to this character.')
        if context.scene.cm_features.advanced_ui:box.prop(g,'lod')
        capture=box.box();capture.label(text='Lid Calibration: Capture')
        options=capture.column();options.enabled=not running() and not state.get('comparison');options.prop(g,'capture_set')
        options.operator('cm.native_face_review',text='Capture Eye Poses',icon='RENDER_STILL').action='CAPTURE_LIDS'
        capture.label(text='Saved LOD0 geometry samples for calibration.')
        accepted=state.get('calibration_capture') or {}
        if accepted.get('status')=='complete' and accepted.get('capture_set')!='CURRENT':
            row=capture.row();row.enabled=not running() and not state.get('comparison');row.operator('cm.native_face_review',text='Prepare Lid Correction',icon='MOD_SMOOTH').action='PREPARE_LIDS'
        if _JOB and _JOB.get('scene')==context.scene and _JOB['request']['action'] in {'PREPARE_LIDS','PREPARE_SCULPT','PREPARE_BANK','PREPARE_INTERVALS','PREPARE_NEUTRAL'}:
            capture.operator('cm.native_face_review',text='Cancel Preparation',icon='CANCEL').action='CANCEL_PREPARE'
    elif state.get('skin_undo_available'):
        row=box.row();row.enabled=not running();row.operator('cm.native_face_review',text='Undo Skin Tuning',icon='LOOP_BACK').action='UNDO_LOOK'
    if g.capture_folder:
        try:captured=json.loads(g.capture_result or '{}')
        except ValueError:captured={}
        box.label(text=str(len(captured.get('poses',[])))+' eye poses: '+captured.get('status','unknown'))
        box.operator('cm.native_face_review',text='Open Eye Captures',icon='FILE_FOLDER').action='OPEN_CAPTURE'
    if g.preparation_folder:
        try:prepared=json.loads(g.preparation_result or '{}')
        except ValueError:prepared={}
        box.label(text='Neutral lid fit prepared.' if prepared.get('method') in {'CAPTURED_NEUTRAL_LID_FIT','CAPTURED_NEUTRAL_LID_BLINK_FIT'} else str(prepared.get('targets',0))+' corrective targets prepared.')
        if prepared.get('method') in {'CAPTURED_INTERMEDIATE_REPAIR_BANK','CAPTURED_INTERVAL_REPAIR_BANK'}:
            from .guided import text
            text(box,'Repairs: '+', '.join(format(value*100,'g')+'%' for value in prepared['repaired_strengths']))
            text(box,'Retains: '+', '.join(format(value*100,'g')+'%' for value in prepared['retained_strengths']))
        box.operator('cm.native_face_review',text='Open Prepared Correction',icon='FILE_FOLDER').action='OPEN_PREPARATION'
        if prepared.get('neutral_inspection',{}).get('valid') is False:
            from .guided import text
            warning=box.box();warning.alert=True
            warning.label(text='Neutral lid needs fitting',icon='ERROR')
            text(warning,'This correction retains the failing neutral pose. Fit the lid to the globe and recapture before accepting motion.')
        row=box.row();row.enabled=state.get('mode')=='ACTIVE' and not running() and not state.get('comparison')
        row.operator('cm.native_face_review',text='Stage Separate Rig',icon='DUPLICATE').action='STAGE_LIDS'
        box.label(text='Uses saved Blueprint settings. Original rig retained.')
        box.label(text='Save actor overrides to its Blueprint first.')
        try:staged=json.loads(g.staging_result or '{}')
        except ValueError:staged={}
        if staged.get('folder')==g.preparation_folder and staged.get('status')=='complete':
            box.label(text='Separate rig staged; motion review required.')
            row=box.row();row.enabled=state.get('mode')=='ACTIVE' and not running()
            row.operator('cm.native_face_review',text='Open Staged Rig in Unreal',icon='ARMATURE_DATA').action='OPEN_STAGED'
            row=box.row();row.enabled=state.get('mode')=='ACTIVE' and not state.get('comparison') and not running();row.operator('cm.native_face_review',text='Create Render Bounds Revision',icon='MESH_CUBE').action='REVISE_BOUNDS'
            if not state.get('comparison'):
                row=box.row();row.enabled=state.get('mode')=='ACTIVE' and not running();row.prop(g,'comparison_spacing')
                row=box.row();row.enabled=not running();row.prop(g,'comparison_background')
                row=box.row();row.enabled=state.get('mode')=='ACTIVE' and not running();row.operator('cm.native_face_review',text='Compare Staged Rig',icon='MOD_MIRROR').action='COMPARE_STAGED'
        sculpt=box.box();sculpt.label(text='Captured Pose Corrective')
        if not g.corrective_object:
            row=sculpt.column();row.enabled=bool(g.capture_folder) and not running();row.prop(g,'sculpt_pose');row.prop(g,'closed_sculpt_scope');row.operator('cm.native_face_review',text='Edit Captured Eye Pose',icon='SCULPTMODE_HLT').action='EDIT_CAPTURED_POSE'
            automatic=sculpt.box();automatic.label(text='Automatic Eye Repair');automatic.prop(g,'repair_control')
            neutral=automatic.row();neutral.enabled=state.get('mode')=='ACTIVE' and bool(g.capture_folder) and bool(g.preparation_folder) and not state.get('comparison') and not running()
            neutral.operator('cm.native_face_review',text='Fit Neutral Lids',icon='MOD_SMOOTH').action='PREPARE_NEUTRAL'
            automatic.label(text='Separate LOD0 proposal; eyes retained.')
            automatic.label(text='Independent blink compensation included.')
            if g.repair_control.startswith('eyeBlink'):
                automatic.label(text='Blink repair is experimental.')
                automatic.label(text='Inspect between captured strengths.')
            row=automatic.row();row.enabled=state.get('mode')=='ACTIVE' and bool(g.capture_folder) and bool(g.preparation_folder) and g.repair_control!='SELECT' and not state.get('comparison') and not running()
            blink_intervals=captured.get('capture_set')=='BLINK_DENSE' and g.repair_control.startswith('eyeBlink')
            if g.capture_folder and (captured.get('capture_set')=='LIDS_INTERVALS' or blink_intervals):
                row.operator('cm.native_face_review',text='Prepare Blink Interval Repairs' if blink_intervals else 'Prepare Interval Motion Repairs',icon='IPO_BEZIER').action='PREPARE_INTERVALS'
                automatic.label(text='Retains accepted quarter-strength anchors.' if blink_intervals else 'Retains accepted eighth-strength anchors.')
                if blink_intervals:
                    extra=automatic.row();extra.enabled=row.enabled;extra.operator('cm.native_face_review',text='Repair Captured Blink Strengths',icon='MOD_SMOOTH').action='PREPARE_BANK'
            else:
                row.operator('cm.native_face_review',text='Prepare Intermediate Repairs',icon='MOD_SMOOTH').action='PREPARE_BANK'
                automatic.label(text='Retains valid sampled poses and full strength.')
        else:
            sculpt.label(text=g.corrective_object.name);sculpt.label(text='Sculpt the unmasked lid. Keep topology unchanged.')
            try:closed=bool(json.loads(g.sculpt_state or '{}').get('full_blink'))
            except ValueError:closed=False
            if closed:
                sculpt.label(text='Full blink: edits change closed-eye shape.')
                sculpt.label(text='Neutral and quarter-strength poses retained.')
                sculpt.label(text='Inspect closure; open-lid untangling unavailable.')
            else:
                sculpt.prop(g,'repair_surface')
                if g.repair_surface:sculpt.label(text='Includes visible rim and socket; separate proposal only.')
                sculpt.label(text='Partial-strength edits retain the full-strength pose.')
            row=sculpt.row();row.enabled=not running()
            row.operator('cm.native_face_review',text='Resume Sculpting',icon='SCULPTMODE_HLT').action='SCULPT_POSE'
            row.operator('cm.native_face_review',text='Frame Corrective',icon='VIEWZOOM').action='FRAME_SCULPT'
            row=sculpt.row();row.enabled=state.get('mode')=='ACTIVE' and not state.get('comparison') and not running();row.operator('cm.native_face_review',text='Prepare Sculpted Correction',icon='MOD_SMOOTH').action='PREPARE_SCULPT'
            row=sculpt.row();row.enabled=not running();row.operator('cm.native_face_review',text='Keep Sculpt and Finish Editing').action='FINISH_SCULPT'
            row=sculpt.row();row.enabled=not running();row.operator('cm.native_face_review',text='Discard Temporary Corrective',icon='TRASH').action='DISCARD_SCULPT'
        sculpt.label(text='One eye control; neutral retained.')
        sculpt.label(text='Protected regions restored when preparing.')
    if g.status:
        from .guided import text
        text(box,g.status)


def register():
    global _SHUTTING
    _SHUTTING=False
    for cls in (native_face_mixer.CM_FaceMixerControl,native_face_mixer.CM_OT_native_face_mixer,CM_FaceReviewSettings,CM_OT_native_face_review):bpy.utils.register_class(cls)
    bpy.types.Scene.cm_face_review=PointerProperty(type=CM_FaceReviewSettings)
    if detach not in bpy.app.handlers.load_pre:bpy.app.handlers.load_pre.append(detach)
    if loaded not in bpy.app.handlers.load_post:bpy.app.handlers.load_post.append(loaded)
    if _JOB and not bpy.app.timers.is_registered(poll):bpy.app.timers.register(poll,first_interval=.1,persistent=True)


def unregister():
    global _SHUTTING
    _SHUTTING=True;detach()
    for callback in (poll,flush):
        if bpy.app.timers.is_registered(callback):bpy.app.timers.unregister(callback)
    for handler,callback in ((bpy.app.handlers.load_pre,detach),(bpy.app.handlers.load_post,loaded)):
        if callback in handler:handler.remove(callback)
    if not _JOB or not _JOB.get('thread') or not _JOB['thread'].is_alive():_close_remote()
    del bpy.types.Scene.cm_face_review
    for cls in (CM_OT_native_face_review,CM_FaceReviewSettings,native_face_mixer.CM_OT_native_face_mixer,native_face_mixer.CM_FaceMixerControl):bpy.utils.unregister_class(cls)
