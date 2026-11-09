"""Run a staged 8K bake with only one Blender resident, then collect normally."""
import json,os,runpy,subprocess
from pathlib import Path


def read_job(folder):
    folder=Path(folder).resolve()
    metadata=json.loads((folder/'external.json').read_text(encoding='utf-8'))
    root=Path(metadata['project_dir']).resolve()
    if metadata.get('schema_version')!=1 or folder.parent!=root/'Bakes'/'.jobs':
        raise ValueError('External bake must belong to this character project')
    request=json.loads((folder/'request.json').read_text(encoding='utf-8'))
    if metadata['mode'] not in {'FINAL_MAPS','SKIN'} or request['mode']!=metadata['mode']:
        raise ValueError('Unsupported external bake mode')
    if Path(request['addon_root']).resolve()!=Path(__file__).resolve().parents[1]:
        raise ValueError('Use the same installed add-on location that prepared this bake')
    for key,expected in (('snapshot',folder/'snapshot.blend'),('staging',folder/'staging'),('report',folder/'report.json')):
        if Path(request[key]).resolve()!=expected:raise ValueError('External bake path does not match its owned job: '+key)
    return folder,metadata,request


def run(folder,blender):
    """Own the worker until its real process exits; never retry an old dispatch."""
    folder,metadata,request=read_job(folder)
    if (folder/'launch.json').exists() or (folder/'dispatch.json').exists() or (folder/'exit.json').exists():
        raise ValueError('External bake already dispatched. Observe or collect the original action; do not restart it.')
    source=Path(__file__).resolve().parent
    budget=runpy.run_path(str(source/'resource_budget.py'))
    budget['require']('external map bake',metadata['required_memory_bytes'])
    budget['require_disk']('external map bake reserve',folder,2*budget['GIB'])
    refs=runpy.run_path(str(source/'process_ref.py'))
    # Atomically claim this action before starting a process. Two independent
    # agents must not both pass an exists() check and launch duplicate 8K jobs.
    controller=refs['identify'](os.getpid())
    with (folder/'launch.json').open('x',encoding='utf-8') as claim:
        json.dump(dict(controller=controller),claim)
    request['parent']=os.getpid();request['addon_version']=metadata['addon_version']
    (folder/'request.json').write_text(json.dumps(request,indent=2),encoding='utf-8')
    try:
        with (folder/'worker.log').open('w',encoding='utf-8') as log:
            process=subprocess.Popen([str(blender),'--background','--factory-startup','--python-exit-code','1',
                '--python',str(source/'bake_worker.py'),'--',str(folder/'request.json')],stdout=log,stderr=subprocess.STDOUT,
                creationflags=(subprocess.CREATE_NO_WINDOW|subprocess.BELOW_NORMAL_PRIORITY_CLASS) if os.name=='nt' else 0)
    except Exception:
        # Popen did not return a process: no worker exists to observe.
        (folder/'launch.json').unlink()
        raise
    identity=refs['identify'](process.pid)
    if not identity:
        code=process.wait()
        raise RuntimeError('External bake worker exited before identification: '+str(code))
    (folder/'dispatch.json').write_text(json.dumps(dict(worker=identity,controller=controller)),encoding='utf-8')
    canceled=False
    try:code=process.wait()
    except KeyboardInterrupt:
        canceled=True
        if process.poll() is None:process.terminate()
        code=process.wait()
    result=dict(exit_code=code,worker=identity)
    if canceled:result['canceled']=True
    temporary=folder/'exit.tmp';temporary.write_text(json.dumps(result),encoding='utf-8');temporary.replace(folder/'exit.json')
    return dict(folder=str(folder),**result)


def adopt(context,folder):
    """Validate current inputs and use the ordinary atomic bake delivery path."""
    import bpy
    from . import bake_jobs,bl_info,process_ref
    if bake_jobs.running():raise ValueError('Finish the active map bake before collecting an external result')
    folder,metadata,request=read_job(folder)
    s,f=context.scene.cm_settings,context.scene.cm_features
    if (Path(bpy.path.abspath(s.project_dir)).resolve()!=Path(metadata['project_dir']).resolve()
            or s.character_name!=metadata['character_name']):raise ValueError('Open the original character project before collecting its bake')
    if list(bl_info['version'])!=metadata['addon_version']:raise ValueError('Use the add-on version that prepared this external bake')
    if (folder/'adopted.json').exists():raise ValueError('This external bake was already collected')
    observed=json.loads((folder/'exit.json').read_text(encoding='utf-8'))
    dispatch=json.loads((folder/'dispatch.json').read_text(encoding='utf-8'))
    if observed['worker']!=dispatch['worker'] or process_ref.alive(observed['worker']):
        raise ValueError('Wait for the original external bake process to exit')
    if type(observed.get('exit_code')) is not int:raise ValueError('External bake has no authoritative process exit result')
    class ObservedExit:
        def poll(self):return observed['exit_code']
    job={key:metadata[key] for key in ('mode','slots','sculpt_basis','map_basis','skin_basis','skin_plan')}
    job.update(folder=folder,scene=context.scene,destination=Path(metadata['project_dir'])/'Maps',
               map_state=tuple(metadata['map_state']),process=ObservedExit(),cancel_requested=False,open_folder=False)
    bake_jobs._JOB=job
    bake_jobs.poll()
    if not s.status.startswith('Background maps ready:'):raise ValueError(s.status)
    result=dict(status='complete',worker=observed['worker'],exit_code=observed['exit_code'],message=s.status)
    (folder/'adopted.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    return result


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser(description='Run a staged character map bake after closing its Blender editor. Reopen the original project and collect the finished result through bake_external.adopt.')
    parser.add_argument('folder',help='The owned Bakes/.jobs folder returned by bake_jobs.start(..., defer=True)')
    parser.add_argument('--blender',required=True,help='Path to the Blender executable used for preparation')
    args=parser.parse_args()
    result=run(args.folder,args.blender)
    print(json.dumps(result,indent=2),flush=True)
    raise SystemExit(result['exit_code'])
