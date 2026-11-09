"""Owned background capture-to-viewer preparation; never accesses interactive bpy."""
import hashlib
import json
import os
import runpy
import subprocess
import sys
import time
import traceback
from pathlib import Path
from tempfile import TemporaryDirectory

GENERATOR='CreateMetaHuman.NativeWebJob'
TERMINAL={'complete','failed','cancelled'}
SCRIPTS=('native_web_pipeline','native_web_geometry_worker','native_web_geometry',
         'native_web_materials','native_web_grooms','native_web_bundle','web_full_skin',
         'web_grooms','web_controls','web_lighting','web_native_materials','web_preview',
         'resource_budget','process_ref','native_pose_bank','web_native_pose_bank','native_skin_motion','web_skin_motion')


def module(name):return runpy.run_path(str(Path(__file__).with_name(name+'.py')))


def digest(path):
    with path.open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()


def write(path,value):
    pending=path.with_suffix('.pending');pending.write_text(json.dumps(value,indent=2),encoding='utf-8')
    # Windows readers/antivirus can briefly hold the current status file open.
    # Retain the complete pending record and retry the atomic replacement.
    for attempt in range(10):
        try:pending.replace(path);return
        except PermissionError:
            if attempt==9:raise
            time.sleep(.05)


class Cancelled(Exception):pass


def validate(request):
    job=Path(request['job']).resolve();source=Path(request['source']).resolve()
    if request.get('generator')!=GENERATOR or request.get('schema')!=1 or not request.get('id') or job.name!=request['id']:raise ValueError('Invalid native web job request')
    if source==job or source.is_relative_to(job) or job.is_relative_to(source):raise ValueError('Keep the native capture separate from the web job')
    if any(type(request.get(k)) is not bool for k in ('archive','legacy_component_frame','legacy_textures','legacy_grooms')):raise ValueError('Native export options must be explicit booleans')
    path=source/'source-manifest.json'
    if digest(path)!=request['source_sha256']:raise ValueError('Native capture changed after export was requested')
    appearance=request.get('appearance')
    if appearance and digest(Path(appearance))!=request['appearance_sha256']:raise ValueError('Artist bake metadata changed after export was requested')
    if request.get('expression_bank') and digest(Path(request['expression_bank']))!=request.get('expression_bank_sha256'):raise ValueError('Native face bank changed after export was requested')
    if request.get('skin_motion_sha256') and digest(source/'skin-motion/manifest.json')!=request['skin_motion_sha256']:raise ValueError('Native skin response changed after export was requested')
    return job,source


def legacy_bindings(seed,source,geometry):
    """Reuse verified legacy scalp correspondence, resolving the new target surface."""
    value=json.loads(seed.read_bytes());geo=json.loads((geometry/'geometry-manifest.json').read_bytes())
    if value.get('generator')!='CreateMetaHuman.NativeWebGroomBindings' or value.get('source_manifest_sha256')!=digest(source/'source-manifest.json'):raise ValueError('Legacy groom binding seed belongs to another capture')
    parts={p['component']:p for p in geo['parts']}
    value['legacy_binding_seed_sha256']=digest(seed)
    value['legacy_binding_original_geometry_manifest_sha256']=value['geometry_manifest_sha256']
    for row in value['bindings']:
        if row['source_surface'].get('scope')=='geometry':raise ValueError('Legacy scalp correspondence must be stored with its native capture')
        target=parts[row['target_component']].get('binding_surface')
        if not target:raise ValueError('The converted groom target has no binding surface')
        row['target_surface']={'file':target,'sha256':geo['files'][target]['sha256']}
    value['geometry_manifest_sha256']=digest(geometry/'geometry-manifest.json')
    return value


def run(request_path):
    request_path=Path(request_path).resolve();raw=request_path.read_bytes();request=json.loads(raw)
    job=Path(request['job']).resolve()
    if request_path!=job/'request.json' or job.name!=request.get('id'):raise ValueError('The export request must belong to its job folder')
    if (job/'state.json').exists():raise ValueError('This job has already started; monitor its existing process instead')
    process_ref=module('process_ref')
    state={'schema':1,'generator':GENERATOR,'id':request['id'],'request_sha256':hashlib.sha256(raw).hexdigest(),
           'status':'running','phase':'preflight','worker':process_ref['identify'](os.getpid()),'child':None,'source_sha256':request['source_sha256']}
    child=None
    def phase(name):state['phase']=name;write(job/'state.json',state);print('CM_NATIVE_WEB_PHASE '+name,flush=True)
    def cancelled():
        path=job/'cancel.json'
        return path.is_file() and json.loads(path.read_bytes()).get('id')==request['id']
    def check():
        if cancelled():raise Cancelled('Native web export cancelled; prior deliveries were retained')
    try:
        job,source=validate(request)
        phase('preflight');check()
        # Detect a missing solver interpreter before starting expensive conversion.
        import numpy,scipy
        budget=module('resource_budget');budget['require']('native web export',1024**3)
        budget['require_disk']('native web export',job,request['disk_budget'])
        phase('geometry');check()
        with (job/'geometry.log').open('w',encoding='utf-8') as log:
            command=[request['blender'],'--background','--factory-startup','--python-exit-code','1','--python',str(Path(__file__).with_name('native_web_geometry_worker.py')),'--',str(request_path)]
            child=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
            state['child']=process_ref['identify'](child.pid);write(job/'state.json',state)
            while child.poll() is None:check();time.sleep(.2)
            if child.returncode:raise RuntimeError('Geometry conversion failed; see geometry.log')
        state['child']=None;child=None;check()
        geometry=job/'geometry'
        with TemporaryDirectory(prefix='stage-',dir=job) as temporary:
            temporary=Path(temporary);phase('materials');check()
            module('native_web_materials')['prepare'](source,geometry/'geometry-manifest.json',temporary/'materials',
                appearance_manifest=request.get('appearance'),allow_legacy_textures=request['legacy_textures'],texture_limit=8192)
            phase('grooms');check();api=module('native_web_grooms');binding=request.get('binding_manifest')
            if not binding:
                binding=temporary/'bindings.json';write(binding,api['binding_plan'](source,geometry))
            else:
                if not request['legacy_grooms']:raise ValueError('A legacy binding seed requires explicit legacy groom opt-in')
                if digest(Path(binding))!=request['binding_sha256']:raise ValueError('Legacy groom bindings changed after export was requested')
                value=legacy_bindings(Path(binding),source,geometry);binding=temporary/'bindings.json';write(binding,value)
            api['prepare'](source,geometry,binding,temporary/'grooms',allow_legacy=request['legacy_grooms'])
            phase('bundle');check()
            receipt=module('native_web_bundle')['publish'](source,geometry,temporary/'materials',temporary/'grooms',request['vendor'],request['environments'],job/'bundle',name=request['name'],archive=request['archive'])
        validate(request)
        if request_path.read_bytes()!=raw or digest(source/'source-manifest.json')!=request['source_sha256']:raise ValueError('Export inputs changed during preparation; the generated delivery was not adopted')
        state.update(status='complete',phase='complete',output=str(job/'bundle'),receipt=receipt,cancel_arrived_after_commit=cancelled())
    except Cancelled as error:state.update(status='cancelled',error=str(error))
    except Exception as error:state.update(status='failed',error=str(error),traceback=traceback.format_exc());print(traceback.format_exc(),flush=True)
    finally:
        if child is not None and child.poll() is None:
            child.terminate()
            try:child.wait(timeout=15)
            except subprocess.TimeoutExpired:child.kill();child.wait()
        state['child']=None;write(job/'state.json',state)
    return state


if __name__=='__main__':
    result=run(sys.argv[1]);print(json.dumps({'status':result['status'],'output':result.get('output'),'error':result.get('error')}),flush=True)
    sys.exit(0 if result['status']=='complete' else 2 if result['status']=='cancelled' else 1)
