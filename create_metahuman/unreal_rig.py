"""Stage and observe one native full-rig request without a Blender runtime."""
import hashlib,json,math,re,runpy,shutil,time,uuid
from datetime import datetime,timezone
from pathlib import Path


def digest(path):
    with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()


def prepare(project_dir,character_name,character,expected_project,*,geometry):
    root=Path(project_dir).resolve();project=Path(expected_project).resolve()
    name=re.sub(r'[^A-Za-z0-9_]','_',character_name).strip('_')
    if not root.is_dir() or not name or not character.startswith('/Game/') or not project.is_file() or project.suffix.lower()!='.uproject':raise ValueError('Choose a saved character and its existing Unreal project')
    vertices=geometry.get('vertices',{})
    if not vertices.get('head') or set(vertices)-{'head','eye_left','eye_right','teeth'}:raise ValueError('Prepare complete accepted head/anatomy correspondence before rigging')
    for points in vertices.values():
        if any(len(point)!=3 or any(not isinstance(v,(int,float)) or not math.isfinite(v) for v in point) for point in points):raise ValueError('Native rig geometry must contain finite XYZ positions')
    runpy.run_path(str(Path(__file__).with_name('resource_budget.py')))['require_disk']('native rig reserve',root,2*1024**3)
    history=root/'Export'/name/'last-rig-action.json'
    if history.is_file():
        old=json.loads(history.read_text());receipt=Path(old['receipt']).resolve()
        if old.get('character')==character and Path(old.get('expected_project','')).resolve()==project:
            if receipt.name!='receipt.json' or not receipt.is_relative_to(root/'Export/.unreal-actions'):raise ValueError('The saved rig receipt is outside this character project')
            if digest(receipt.parent/'request.json')!=old.get('request_sha256'):raise ValueError('The original rig request changed; inspect its retained action')
            result=json.loads(receipt.read_text()) if receipt.is_file() else {}
            if result.get('status') not in {'complete','failed'}:raise ValueError('Observe the original rig action before preparing another: '+str(receipt))
            if result.get('action')!='REQUEST_RIG' or result.get('character')!=character or result.get('request_sha256')!=old['request_sha256']:raise ValueError('The original rig receipt does not match its request')
    folder=root/'Export/.unreal-actions'/uuid.uuid4().hex[:12];scripts=folder/'scripts';scripts.mkdir(parents=True,exist_ok=False)
    for filename in ('unreal_rig.py','unreal_import.py','resource_budget.py','process_ref.py'):shutil.copy2(Path(__file__).with_name(filename),scripts/filename)
    request=dict(schema=1,action='REQUEST_RIG',character=character,expected_project=str(project),geometry=geometry)
    (folder/'request.json').write_text(json.dumps(request,indent=2),encoding='utf-8')
    record=dict(schema=1,project_dir=str(root),character_name=character_name,character=character,expected_project=str(project),receipt=str(folder/'receipt.json'),request_sha256=digest(folder/'request.json'))
    history.parent.mkdir(parents=True,exist_ok=True);temporary=history.with_suffix('.tmp');temporary.write_text(json.dumps(record,indent=2));temporary.replace(history)
    return dict(folder=str(folder),request=str(folder/'request.json'),receipt=str(folder/'receipt.json'))


def dispatch(request_path,*,keep_alive=False):
    import os,unreal
    request_path=Path(request_path).resolve();folder=request_path.parent;request=json.loads(request_path.read_text());receipt=folder/'receipt.json'
    if receipt.exists() or (folder/'dispatch.json').exists():raise ValueError('This rig action was already dispatched; observe its original receipt')
    if request.get('schema')!=1 or request.get('action')!='REQUEST_RIG':raise ValueError('Unsupported native rig request')
    identity=runpy.run_path(str(Path(__file__).with_name('process_ref.py')))['identify'](os.getpid())
    with (folder/'dispatch.json').open('x') as stream:json.dump(identity,stream)
    report=dict(status='scheduled',action='REQUEST_RIG',character=request['character'],request_sha256=digest(request_path),request_dispatched=False,attached_existing_request=False,editor_process=identity)
    state=dict(character=None,subsystem=None,owned=False,handle=None,last=0.,done=False,before=None,busy=False)
    def write(filename,value):
        path=folder/filename;temporary=path.with_suffix('.tmp');temporary.write_text(json.dumps(value,indent=2));temporary.replace(path)
    def inspect():return json.loads(unreal.CreateMHCharacterLibrary.inspect_character(state['character']))
    def audit(label):return runpy.run_path(str(Path(__file__).with_name('unreal_import.py')))['audit_head_conform'](state['character'],request['geometry']['vertices'],folder/(label+'.json'))
    def settings():
        character=state['character'];data=json.loads(unreal.CreateMHArtistLibrary.snapshot_object(character))
        props=next(row['properties'] for row in data['objects'] if row['object']==character.get_path_name())
        return {key:props[key]['text'] for key in ('SkinSettings','EyesSettings','MakeupSettings','HeadModelSettings','FaceEvaluationSettings','AssemblySettings') if key in props}
    def progress(phase):
        report['phase']=phase;write('progress.json',report)
    def finish(error=''):
        if state['done']:return
        state['done']=True
        report['inspected_at']=datetime.now(timezone.utc).isoformat()
        try:
            if error:raise RuntimeError(error)
            native=inspect();report['assembly']=dict(native,detailed_status_available=True)
            if native.get('rig_request_pending') or native.get('rig_state')!='RIGGED' or not native.get('has_face_dna'):raise RuntimeError('Native request ended without a rigged face DNA; inspect the MetaHuman service log')
            report['head_geometry']=audit('post-rig-geometry')
            after_settings=settings();write('artist-settings-after.json',after_settings)
            report['artist_setting_changes']=[key for key,value in state['before'].items() if after_settings.get(key)!=value]
            if not report['head_geometry']['within_native_precision'] or report['artist_setting_changes']:raise RuntimeError('Rig generated but accepted geometry or artist settings changed; review the retained native state')
            if not unreal.EditorAssetLibrary.save_loaded_asset(state['character'],False):raise RuntimeError('Rig completed in memory; save the character before closing Unreal')
            report.update(status='complete',phase='Full rig saved',saved=True,artist_settings_preserved=True)
        except Exception as exc:report.update(status='failed',error=str(exc))
        finally:
            if state['handle'] is not None:unreal.unregister_slate_post_tick_callback(state['handle']);state['handle']=None
            try:
                if state['owned'] and state['character']:state['subsystem'].remove_object_to_edit(state['character'])
            except Exception as exc:report['cleanup_warning']=str(exc)
            write('receipt.json',report)
            if keep_alive:unreal.EditorPythonScripting.set_keep_python_script_alive(False)
    def tick(delta):
        # Native initialization and geometry extraction can pump Slate while
        # still on this stack. A nested tick must not finish/release that session.
        if state['done'] or state['busy'] or time.monotonic()-state['last']<1:return
        state['busy']=True
        state['last']=time.monotonic()
        try:
            if state['character'] is None:
                project=Path(unreal.Paths.get_project_file_path()).resolve()
                if project!=Path(request['expected_project']).resolve():raise ValueError('Open the linked Unreal project')
                report['unreal_project_file']=str(project)
                character=unreal.load_asset(request['character'])
                if not isinstance(character,unreal.MetaHumanCharacter):raise ValueError('The linked MetaHuman character is missing')
                subsystem=unreal.get_editor_subsystem(unreal.MetaHumanCharacterEditorSubsystem);owned=not subsystem.is_object_added_for_editing(character)
                runpy.run_path(str(Path(__file__).with_name('resource_budget.py')))['require']('native rig initialization',(8 if owned else 2)*1024**3)
                before=json.loads(unreal.CreateMHArtistLibrary.snapshot_object(character))
                if any(row['package_dirty'] for row in before['objects']):raise ValueError('Save native artist edits before rigging')
                write('artist-before-initialization.json',before);state.update(character=character,subsystem=subsystem)
                if owned and not subsystem.try_add_object_to_edit(character):raise RuntimeError('Creator could not initialize the linked character')
                state['owned']=owned
                state['before']=settings();write('artist-settings-before.json',state['before']);report['pre_rig_geometry']=audit('pre-rig-geometry')
                if not report['pre_rig_geometry']['within_native_precision']:raise ValueError('Native geometry differs from the accepted Blender character; capture artist edits before rigging')
                native=inspect();report['assembly']=native
                if 'rig_request_pending' not in native or 'rig_state' not in native or 'has_face_dna' not in native:
                    raise RuntimeError('Native rig status is incomplete; no rig was requested. Refresh status before continuing.')
                if native.get('texture_request_pending'):raise ValueError('Finish the current native texture request before rigging')
                if native.get('rig_request_pending'):
                    report['attached_existing_request']=True;progress('Observing existing native rig request')
                elif native.get('rig_state')=='RIGGED' and native.get('has_face_dna'):
                    report['already_satisfied']=True;finish()
                else:
                    params=unreal.MetaHumanCharacterAutoRiggingRequestParams();params.rig_type=unreal.MetaHumanRigType.JOINTS_AND_BLEND_SHAPES;params.report_progress=False;params.blocking=False
                    report['request_dispatched']=True;progress('Requesting native full rig')
                    subsystem.request_auto_rigging(character,params)
                return
            native=inspect();report['assembly']=native
            if not native or 'rig_request_pending' not in native:progress('Native rig state unavailable; retaining original observer');return
            if native['rig_request_pending']:report['status']='waiting';progress('Waiting for original native rig request');return
            finish()
        except Exception as exc:
            if report['request_dispatched'] or report['attached_existing_request']:
                try:
                    native=inspect()
                    if not native or 'rig_request_pending' not in native or native.get('rig_request_pending'):
                        report['observation_warning']=str(exc);progress('Retaining original native rig observer');return
                except Exception:report['observation_warning']=str(exc);progress('Retaining original native rig observer');return
            finish(str(exc))
        finally:state['busy']=False
    if keep_alive:unreal.EditorPythonScripting.set_keep_python_script_alive(True)
    state['handle']=unreal.register_slate_post_tick_callback(tick);progress('Scheduled native rig request')
    return dict(status='scheduled',receipt=str(receipt),editor_process=identity)
