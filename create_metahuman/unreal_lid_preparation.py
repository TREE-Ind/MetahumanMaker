"""Read-only native input export and asynchronous numerical lid preparation."""
import json,runpy,subprocess
from pathlib import Path


def write(path,value):
    path=Path(path);temporary=path.with_suffix('.tmp');temporary.write_text(json.dumps(value,indent=2));temporary.replace(path)


def begin(folder,request,state):
    import unreal
    folder=Path(folder).resolve();origin=Path(state['origin'])
    if (folder/'worker-process.json').exists():raise ValueError('Observe this original preparation; do not dispatch it twice')
    active=getattr(unreal,'_cm_lid_preparations',{})
    if active:raise ValueError('Wait for the original lid preparation')
    python=request['python']
    if not Path(python).is_file():raise ValueError('Choose the configured local Python with NumPy and SciPy')
    review=runpy.run_path(str(Path(__file__).with_name('unreal_face_review.py')))
    api=review['_native'](unreal);actors=unreal.get_editor_subsystem(unreal.EditorActorSubsystem)
    actor=review['_actor'](unreal,state['actor'])
    if not actor:raise ValueError('Return to the original review actor before preparation')
    components=review['_components'](unreal,actor);face=components[state['face']];mesh=face.get_skeletal_mesh_asset()
    sculpt=request.get('sculpt_snapshot');bank=request.get('repair_bank') or request.get('neutral_fit')
    if sculpt:
        snapshot=json.loads(Path(sculpt).read_text());capture_folder=Path(snapshot['pose_file']).parent
        capture=json.loads((capture_folder/'capture.json').read_text())
        if capture.get('status')!='complete' or snapshot['source_mesh']!=mesh.get_path_name():raise ValueError('The sculpt does not belong to the connected saved face')
    elif bank:
        base=json.loads(Path(bank['base_manifest']).read_text());capture_folder=Path(bank['capture_folder'])
        capture=json.loads((capture_folder/'capture.json').read_text())
        if capture.get('status')!='complete' or base.get('source_mesh')!=mesh.get_path_name():raise ValueError('The intermediate repair set does not belong to the connected saved face')
    else:
        accepted=state.get('calibration_capture');capture_folder=Path(state.get('calibration_capture_folder',''))
        if not accepted or accepted.get('status')!='complete' or accepted.get('capture_set') not in {'LIDS','LIDS_FINE','LIDS_DENSE','LIDS_INTERVALS','BLINK_FINE','BLINK_DENSE','GAZE_UP','GAZE_DOWN','GAZE_LEFT','GAZE_RIGHT'}:raise ValueError('Capture a complete blink/squint or gaze grid first')
        capture=json.loads((capture_folder/'capture.json').read_text())
        if capture!=accepted:raise ValueError('The accepted capture changed')
    if capture.get('candidate')!=mesh.get_path_name():raise ValueError('The capture belongs to another face')
    if not capture.get('source_asset_file'):raise ValueError('Assemble and save the face as a project asset before correction preparation')
    native=unreal.CreateMHCalibrationLibrary
    if not hasattr(native,'export_embedded_calibration_reference'):raise ValueError('Install the matching native bridge 0.28 or newer before preparing lid corrections')
    helpers=runpy.run_path(str(Path(__file__).with_name('unreal_lod_correctives.py')))
    sources=helpers['source_fingerprints'](mesh)
    expected=capture.get('native_source_sha256') or {str(Path(capture['source_asset_file']).resolve()):capture['source_asset_sha256']}
    if sources!=expected:raise ValueError('The saved face or DNA changed, or the older capture lacks the separate DNA fingerprint. Capture the current rig again.')
    budget=runpy.run_path(str(Path(__file__).with_name('resource_budget.py')))
    budget['require']('lid corrective preparation',2*budget['GIB']);budget['require_disk']('lid corrective preparation',folder,2*budget['GIB'])
    if sculpt:
        write(folder/'progress.json',dict(phase='Preparing the protected sculpt from the existing sealed rig reference'))
        write(folder/'preparation-request.json',dict(sculpt_snapshot=sculpt))
    elif request.get('neutral_fit'):
        write(folder/'progress.json',dict(phase='Fitting neutral lids from the existing sealed rig reference'))
        write(folder/'preparation-request.json',dict(neutral_fit=bank))
    elif bank:
        write(folder/'progress.json',dict(phase='Checking captured intermediate strengths from the existing sealed rig reference'))
        write(folder/'preparation-request.json',dict(repair_bank=bank))
    else:
        write(folder/'progress.json',dict(phase='Exporting the selected face embedded rig reference'))
        inputs=folder/'inputs'
        if not native.export_embedded_calibration_reference(mesh,str(inputs)):raise RuntimeError('Could not export the embedded rig into this fresh preparation folder')
        if helpers['source_fingerprints'](mesh)!=sources:raise ValueError('A saved rig source changed during reference export')
        write(inputs/'source-proof.json',dict(source_files=sources,capture=str(capture_folder),source_assets_saved=0))
        write(folder/'preparation-request.json',dict(capture=str(capture_folder)))
    log=(folder/'worker.log').open('w',encoding='utf-8')
    worker=subprocess.Popen([python,str(Path(__file__).with_name('lid_calibration_worker.py')),str(folder)],stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
    try:
        identity=runpy.run_path(str(Path(__file__).with_name('process_ref.py')))['identify'](worker.pid)
        write(folder/'worker-process.json',identity)
    except Exception:
        worker.terminate();log.close();raise
    job=dict(handle=None,busy=False,cancelled=False)
    def tick(delta):
        if job['busy']:return
        if (folder/'cancel.flag').exists():
            job['cancelled']=True
            if worker.poll() is None:worker.terminate()
        if worker.poll() is None:return
        job['busy']=True
        try:
            current=review['_actor'](unreal,state['actor'])
            if not current:raise ValueError('The original actor closed during preparation; the local files were retained')
            review['_guard'](api,review['_components'](unreal,current),state['expected']);review['_look_api']()['guard'](unreal,current,state)
            if helpers['source_fingerprints'](mesh)!=sources:raise ValueError('The saved rig changed during preparation; retain the artist changes')
            if job['cancelled']:proposal=dict(status='cancelled',folder=str(folder),native_motion_accepted=False)
            else:
                proposal=json.loads((folder/'preparation-result.json').read_text())
                if worker.returncode or proposal.get('status') not in {'prepared','unchanged'}:raise RuntimeError(proposal.get('error','The numerical preparation failed'))
                if proposal['status']=='prepared':state['calibration_preparation']=proposal
            state['calibration_preparation_attempt']=proposal;write(origin/'session.json',state)
            sessions=getattr(unreal,'_cm_face_review_sessions',{});sessions[str(origin)]=state
            report=review['_report'](unreal,state,current)
            report['message']='Preparation cancelled. Previous proposal retained.' if job['cancelled'] else proposal['message'] if proposal.get('status')=='unchanged' else 'Prepared '+str(proposal['targets'])+' lid corrective targets. Rig unchanged; native staging and visual review are still required.'
            if proposal.get('status')=='prepared' and proposal.get('warnings'):
                report['message']+=' '+' '.join(proposal['warnings'])
            if proposal.get('method') in {'CAPTURED_NEUTRAL_LID_FIT','CAPTURED_NEUTRAL_LID_BLINK_FIT'}:
                report['message']='Neutral lids fitted'+(' with independent blink compensation' if proposal['method']=='CAPTURED_NEUTRAL_LID_BLINK_FIT' else '')+' in a separate proposal. Stage and review every expression before acceptance. Original rig unchanged.'
            if job['cancelled']:report['status']='cancelled'
        except Exception as error:
            state['calibration_preparation_attempt']=dict(status='failed',folder=str(folder),error=str(error));write(origin/'session.json',state)
            sessions=getattr(unreal,'_cm_face_review_sessions',{});sessions[str(origin)]=state
            try:report=review['_report'](unreal,state,actor)
            except Exception:report={}
            report.update(status='failed',error=str(error))
        finally:
            try:write(folder/'receipt.json',report)
            finally:unreal.unregister_slate_post_tick_callback(job['handle']);log.close();active.pop(str(folder),None)
    job['handle']=unreal.register_slate_post_tick_callback(tick);active[str(folder)]=(job,tick,worker);unreal._cm_lid_preparations=active
    return dict(status='queued',receipt_delegated=True)
