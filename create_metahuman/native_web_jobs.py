"""Blender controls and persistent monitoring for background native web delivery."""
import hashlib
import atexit
import json
import re
import runpy
import shutil
import subprocess
import time
import uuid
from pathlib import Path

import bpy
from bpy.app.handlers import persistent
from . import process_ref,resource_budget,setup_paths

_JOB=None
_PREVIEW=None
LABELS={'preflight':'Checking export inputs','geometry':'Converting native rigged geometry (1/4)',
        'materials':'Preparing textures and artist bakes (2/4)','grooms':'Preparing surface-bound grooms (3/4)',
        'bundle':'Writing the viewer and download ZIP (4/4)'}


def api(name):return runpy.run_path(str(Path(__file__).with_name(name+'.py')))


def write(path,value):
    pending=path.with_suffix('.pending');pending.write_text(json.dumps(value,indent=2),encoding='utf-8');pending.replace(path)


def digest(path):
    with path.open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()


def history(scene):
    s=scene.cm_settings
    name=re.sub(r'[^A-Za-z0-9_-]','_',s.character_name).strip('_') or 'Character'
    return Path(bpy.path.abspath(s.project_dir)).resolve()/'Export'/'Web'/name/'last-native-web-job.json'


def running():
    return bool(_JOB and _JOB.get('active'))


def preflight(context):
    s,f=context.scene.cm_settings,context.scene.cm_features
    if not s.character_name.strip() or s.character_name!=s.character_name.strip() or len(s.character_name)>160 or re.search(r'[<>:"/\\|?*\x00-\x1f]',s.character_name) or s.character_name.endswith('.') or re.fullmatch(r'(CON|PRN|AUX|NUL|COM[1-9]|LPT[1-9])(\..*)?',s.character_name,re.I):raise ValueError('Choose a character name without path characters, reserved device names or trailing spaces/periods')
    if not f.native_web_source.strip():raise ValueError('Choose a completed Native Capture folder containing source-manifest.json and textures')
    source=Path(bpy.path.abspath(f.native_web_source)).resolve()
    if not (source/'source-manifest.json').is_file():raise ValueError('Choose the native capture folder containing source-manifest.json')
    record=json.loads((source/'source-manifest.json').read_bytes())
    sha=digest(source/'source-manifest.json')
    if record.get('generator')!='CreateMetaHuman.NativeWebSource' or record.get('status')!='source_extracted':raise ValueError('Choose a completed native source capture')
    if not record.get('skeletal_meshes'):raise ValueError('Native capture has no rigged geometry')
    expression_bank=Path(bpy.path.abspath(f.native_web_expression_bank)).resolve() if f.native_web_expression_bank.strip() else None
    if expression_bank is None and record.get('expression_bank'):
        expression_bank=(source/record['expression_bank']['file']).resolve()
        if not expression_bank.is_relative_to(source):raise ValueError('Captured native face bank leaves its capture folder')
        if digest(expression_bank)!=record['expression_bank'].get('sha256'):raise ValueError('Captured native facial samples changed after capture')
    if expression_bank:
        bank=json.loads(expression_bank.read_bytes())
        face=next((part for part in record['skeletal_meshes'] if part['component']==f.native_web_face_component),None)
        if bank.get('status')!='complete' or not bank.get('source_mesh_unchanged') or not face or face['asset']!=bank.get('source_mesh'):raise ValueError('Native face samples do not match this completed capture')
    if any(not p.get('actor_relative_transform') for p in record['skeletal_meshes']) and not f.native_web_legacy_frame:raise ValueError('Recapture component transforms, or explicitly enable the legacy frame interpretation in Advanced')
    appearance=Path(bpy.path.abspath(f.native_web_appearance)).resolve() if f.native_web_appearance.strip() else None
    if appearance is None and (source/'appearance-manifest.json').is_file():appearance=source/'appearance-manifest.json'
    appearance_value=json.loads(appearance.read_bytes()) if appearance else None
    provisional={'generator':'CreateMetaHuman.NativeWebGeometry','source_sha256':sha,'face_component':f.native_web_face_component,'parts':[{'component':p['component']} for p in record['skeletal_meshes']]}
    planned=api('native_web_materials')['plan'](record,sha,provisional,appearance_value)
    if not (source/'textures/texture-manifest.json').is_file():raise ValueError('Extract native texture pixels for this capture before web export')
    textures=json.loads((source/'textures/texture-manifest.json').read_bytes())
    if textures.get('generator')!='CreateMetaHuman.NativeWebTextures' or textures.get('status') not in {'pixels_extracted','pixels_extracted_with_unresolved_resources'}:raise ValueError('Complete texture extraction for this capture before web export')
    legacy='source_manifest_sha256' not in textures
    if legacy and not f.native_web_legacy_textures:raise ValueError('Recapture texture provenance, or explicitly allow legacy texture metadata in Advanced')
    if not legacy and textures['source_manifest_sha256']!=sha:raise ValueError('Texture metadata belongs to another native capture')
    resources=set()
    motion_path=source/'skin-motion/manifest.json';motion_sha=None
    if motion_path.is_file():
        motion=json.loads(motion_path.read_bytes());motion_sha=digest(motion_path)
        if motion.get('status')!='complete' or motion.get('source_manifest_sha256')!=sha or not expression_bank or motion.get('bank_sha256')!=digest(expression_bank):raise ValueError('Animated skin and facial samples must belong to the same completed capture')
        for pose in motion['poses']:
            for item in pose['maps'].values():
                path=(source/item['file']).resolve()
                if not path.is_relative_to(source) or not path.is_file() or digest(path)!=item['sha256']:raise ValueError('Animated skin response is missing or changed')
                resources.add(path)
    for recipe in planned['materials'].values():
        for item in recipe['maps'].values():
            if 'asset' in item:
                pixel=textures['textures'].get(item['asset'])
                if not pixel:raise ValueError('Capture is missing required texture pixels: '+item['asset'])
                filename='textures/'+pixel['file']
            else:filename=item['bake']['file']
            path=(source/filename).resolve()
            if not path.is_relative_to(source) or not path.is_file():raise ValueError('Required texture or artist bake is missing: '+filename)
            resources.add(path)
    binding=Path(bpy.path.abspath(f.native_web_binding)).resolve() if f.native_web_legacy_grooms and f.native_web_binding.strip() else None
    if record.get('grooms') and any(g.get('asset') and not g.get('binding_source') for g in record['grooms']):
        if not f.native_web_legacy_grooms or binding is None:raise ValueError('Recapture groom binding provenance, or supply verified legacy binding metadata in Advanced')
    setup_paths.resolve(f)
    python=Path(bpy.path.abspath(f.inference_python))
    if not python.is_file():raise ValueError('Choose the local Python interpreter with NumPy and SciPy in Setup')
    environments=Path(bpy.utils.system_resource('DATAFILES',path='studiolights/world'))
    vendor=Path(__file__).with_name('web_vendor')
    api('native_web_bundle')['vendor_resources'](vendor)
    total=sum(p.stat().st_size for p in resources)
    estimate=max(256*1024**2,total*(3 if f.native_web_archive else 2)+128*1024**2)
    resource_budget.require_disk('native web export',history(context.scene).parent,estimate)
    resource_budget.require('native web export',1024**3)
    return {'source':str(source),'source_sha256':sha,'appearance':str(appearance) if appearance else None,
            'skin_motion_sha256':motion_sha,
            'appearance_sha256':digest(appearance) if appearance else None,'binding_manifest':str(binding) if binding else None,
            'binding_sha256':digest(binding) if binding else None,'face_component':f.native_web_face_component,
            'expression_bank':str(expression_bank) if expression_bank else None,'original_expression_bank':str(expression_bank) if expression_bank else None,'expression_bank_sha256':digest(expression_bank) if expression_bank else None,
            'blender':bpy.app.binary_path,'environments':str(environments),'name':s.character_name,'project':str(Path(bpy.path.abspath(s.project_dir)).resolve()),
            'archive':f.native_web_archive,'legacy_component_frame':f.native_web_legacy_frame,
            'legacy_textures':f.native_web_legacy_textures,'legacy_grooms':f.native_web_legacy_grooms,
            'disk_budget':estimate,'python':str(python),'vendor':str(vendor)}


def start(context):
    global _JOB
    from . import unreal_jobs
    if unreal_jobs.running() and unreal_jobs._JOB.get('kind')=='CAPTURE_NATIVE_WEB':raise ValueError('Wait for the current native source capture before exporting its web bundle')
    recover()
    if running():raise ValueError('A native web export is already running; monitor or cancel it before starting another')
    request=preflight(context);scene=context.scene;f=scene.cm_features
    location=history(scene);job=location.parent/'.native-jobs'/uuid.uuid4().hex[:12]
    source=Path(request['source'])
    if job.is_relative_to(source) or source.is_relative_to(job):raise ValueError('Use a native capture outside the export job folder')
    scripts=job/'scripts'/'create_metahuman';scripts.mkdir(parents=True,exist_ok=False)
    for name in api('native_web_pipeline')['SCRIPTS']:shutil.copy2(Path(__file__).with_name(name+'.py'),scripts/(name+'.py'))
    (scripts/'__init__.py').write_text('# Isolated native web worker package.\n',encoding='utf-8')
    if request.get('expression_bank'):
        bank=api('native_pose_bank')['snapshot'](request['expression_bank'],job/'native-face-bank')
        request['expression_bank']=str(bank);request['expression_bank_sha256']=digest(bank)
    shutil.copytree(request['vendor'],job/'vendor')
    request.update(schema=1,generator='CreateMetaHuman.NativeWebJob',id=job.name,job=str(job),vendor=str(job/'vendor'))
    write(job/'request.json',request)
    with (job/'worker.log').open('w',encoding='utf-8') as log:
        process=subprocess.Popen([request['python'],str(scripts/'native_web_pipeline.py'),str(job/'request.json')],stdout=log,stderr=subprocess.STDOUT,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
    launch={'id':job.name,'worker':process_ref.identify(process.pid),'request_sha256':digest(job/'request.json')};write(job/'launch.json',launch)
    write(location,{'generator':'CreateMetaHuman.NativeWebJobLink','project':str(Path(bpy.path.abspath(scene.cm_settings.project_dir)).resolve()),'name':scene.cm_settings.character_name,'job':str(job)})
    f.native_web_job=str(job);f.native_web_status='Starting native web export. Blender remains usable.'
    _JOB={'scene':scene,'job':job,'process':process,'worker':launch['worker'],'active':True}
    ensure_timer()


def ensure_timer():
    if not bpy.app.timers.is_registered(poll):bpy.app.timers.register(poll,first_interval=.2,persistent=True)


def read(job):
    request_path=job/'request.json';request=json.loads(request_path.read_bytes())
    if request.get('generator')!='CreateMetaHuman.NativeWebJob' or Path(request['job']).resolve()!=job or request.get('id')!=job.name:raise ValueError('Export job identity does not match its folder')
    state_path=job/'state.json';state=json.loads(state_path.read_bytes()) if state_path.is_file() else json.loads((job/'launch.json').read_bytes())
    if state.get('id')!=request['id'] or state.get('request_sha256')!=digest(request_path):raise ValueError('Export request changed after launch')
    return request,state


def refresh(job):
    scene=job['scene'];f=scene.cm_features;folder=job['job'];request,state=read(folder)
    job['worker']=state.get('worker')
    status=state.get('status','running');job['active']=status not in {'complete','failed','cancelled'}
    if job['active']:
        if not process_ref.alive(state.get('worker')):
            job['active']=False;f.native_web_status='The export worker stopped before completion. Logs and prior deliveries were retained.'
        elif (folder/'cancel.json').exists():f.native_web_status='Cancellation requested. The current stage is finishing safely.'
        else:f.native_web_status=LABELS.get(state.get('phase'),'Starting native web export')+'. Blender remains usable.'
    elif status=='complete':
        output=Path(state.get('output','')).resolve()
        if output!=folder/'bundle':raise ValueError('Completed export points outside its owned bundle folder')
        manifest=json.loads((output/'manifest.json').read_bytes())
        if manifest.get('generator')!='CreateMetaHuman.NativeWebBundle' or manifest.get('source_manifest_sha256')!=request['source_sha256']:raise ValueError('Completed bundle does not match the requested native capture')
        if request.get('project')!=str(Path(bpy.path.abspath(scene.cm_settings.project_dir)).resolve()) or request['name']!=scene.cm_settings.character_name:
            f.native_web_status='Export completed for the preceding character. Its delivery remains in the job folder.'
            return False
        f.native_web_output=str(output);f.native_web_output_capture=request['source'];f.native_web_status='Native web bundle ready. Open Export Folder and run Start Preview.cmd.'
        if not request['archive']:f.native_web_status='Native web preview folder ready. Run Start Preview.cmd; ZIP creation was disabled.'
    else:f.native_web_status=state.get('error','Native web export '+status)
    return job['active']


def original_worker_alive(job):
    if job.get('process') is not None and job['process'].poll() is None:return True
    return process_ref.alive(job.get('worker'))


def observation_error(job,error):
    job['active']=original_worker_alive(job)
    try:
        job['scene'].cm_features.native_web_status=('The original export worker is still running. Waiting to read its status: ' if job['active'] else 'Cannot monitor native web export: ')+str(error)
    except ReferenceError:pass
    return job['active']


def poll():
    if not _JOB:return None
    try:
        active=refresh(_JOB)
        for screen in bpy.data.screens:
            for area in screen.areas:
                if area.type=='VIEW_3D':area.tag_redraw()
        return .5 if active else None
    except (OSError,ValueError,KeyError,ReferenceError) as error:
        return .5 if observation_error(_JOB,error) else None


def cancel():
    if not running():raise ValueError('No native web export is running')
    request,_=read(_JOB['job']);write(_JOB['job']/'cancel.json',{'id':request['id']})
    _JOB['scene'].cm_features.native_web_status='Cancellation requested. Prior deliveries are retained.'


def adopt_capture(scene,receipt):
    root=Path(receipt['source']).resolve();appearance=Path(receipt['appearance']).resolve()
    if not root.is_relative_to(Path(bpy.path.abspath(scene.cm_settings.project_dir)).resolve()/'NativeWebCaptures') or appearance!=root/'appearance-manifest.json':raise ValueError('Native capture is outside this character project')
    if scene.cm_features.native_web_capture_receipt and root.name!=Path(scene.cm_features.native_web_capture_receipt).parent.name:raise ValueError('The native capture belongs to a different Unreal action')
    source=json.loads((root/'source-manifest.json').read_bytes());bakes=json.loads(appearance.read_bytes())
    sha=digest(root/'source-manifest.json')
    if source.get('status')!='source_extracted' or sha!=receipt['source_manifest_sha256'] or bakes.get('status')!='complete' or bakes.get('source_manifest_sha256')!=sha:raise ValueError('Capture receipt does not match its completed source and appearance')
    f=scene.cm_features;f.native_web_source=str(root);f.native_web_appearance=str(appearance);f.native_web_binding=''
    f.native_web_expression_bank=''
    f.native_web_legacy_frame=False;f.native_web_legacy_textures=False;f.native_web_legacy_grooms=False


def start_preview(context,*,open_browser=False):
    global _PREVIEW
    f=context.scene.cm_features
    output=Path(f.native_web_output).resolve() if f.native_web_output else None
    if not output or not (output/'StartPreview.py').is_file():raise ValueError('Complete a native web export before opening its preview')
    request,state=read(output.parent)
    if state.get('status')!='complete' or Path(state['output']).resolve()!=output:raise ValueError('The preview does not belong to a completed native web job')
    if _PREVIEW and _PREVIEW['output']==output and _PREVIEW['process'].poll() is None:
        _PREVIEW['open_browser']=open_browser
        if _PREVIEW.get('url'):
            f.native_web_url=_PREVIEW['url']
            if open_browser:bpy.ops.wm.url_open(url=_PREVIEW['url'])
            _PREVIEW['open_browser']=False
        return
    stop_preview()
    log=output.parent/'preview-server.log'
    with log.open('w',encoding='utf-8') as stream:
        process=subprocess.Popen([request['python'],str(output/'StartPreview.py')],stdout=stream,stderr=subprocess.STDOUT,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
    _PREVIEW={'scene':context.scene,'output':output,'process':process,'log':log,'started':time.monotonic(),'open_browser':open_browser}
    f.native_web_status='Starting the local web preview...'
    if not bpy.app.timers.is_registered(poll_preview):bpy.app.timers.register(poll_preview,first_interval=.2)


def poll_preview():
    if not _PREVIEW:return None
    f=_PREVIEW['scene'].cm_features
    if _PREVIEW['process'].poll() is not None:
        f.native_web_status='The local preview server stopped. The exported folder is retained.';stop_preview();return None
    with _PREVIEW['log'].open(encoding='utf-8') as stream:text=stream.read(1024)
    match=re.search(r'^Open (http://127\.0\.0\.1:[0-9]+/preview\.html)\s*$',text,re.M)
    if match:
        url=match.group(1);_PREVIEW['url']=url;f.native_web_url=url;f.native_web_status='Local web preview ready.'
        if _PREVIEW['open_browser']:bpy.ops.wm.url_open(url=url);_PREVIEW['open_browser']=False
        return None
    if time.monotonic()-_PREVIEW['started']>15:
        f.native_web_status='The local preview server did not report its address. See the export logs.';stop_preview();return None
    return .2


def stop_preview():
    global _PREVIEW
    if bpy.app.timers.is_registered(poll_preview):bpy.app.timers.unregister(poll_preview)
    close_preview_process()
    preview,_PREVIEW=_PREVIEW,None
    if preview:
        try:preview['scene'].cm_features.native_web_url=''
        except ReferenceError:pass


def close_preview_process():
    """Exit cleanup needs no access to Blender data during Python shutdown."""
    if _PREVIEW:
        process=_PREVIEW['process']
        if process.poll() is None:process.terminate()
        try:process.wait(timeout=2)
        except subprocess.TimeoutExpired:process.kill();process.wait()


@persistent
def recover(*_):
    global _JOB
    if _JOB and _JOB.get('active'):
        try:
            if refresh(_JOB):return
        except (ReferenceError,OSError,ValueError,KeyError) as error:
            if observation_error(_JOB,error):ensure_timer();return
            _JOB=None
    scene=bpy.context.scene
    if not hasattr(scene,'cm_features'):return
    f=scene.cm_features
    try:
        location=history(scene)
        if not f.native_web_job and location.is_file():
            link=json.loads(location.read_bytes())
            if link.get('generator')=='CreateMetaHuman.NativeWebJobLink' and link.get('project')==str(Path(bpy.path.abspath(scene.cm_settings.project_dir)).resolve()) and link.get('name')==scene.cm_settings.character_name:f.native_web_job=link['job']
        if not f.native_web_job:return
        job=Path(bpy.path.abspath(f.native_web_job)).resolve()
        if job.parent!=location.parent/'.native-jobs':raise ValueError('The linked export belongs to another character project')
        launch=json.loads((job/'launch.json').read_bytes())
        if launch.get('id')!=job.name or launch.get('request_sha256')!=digest(job/'request.json'):raise ValueError('The original export launch does not match its request')
        _JOB={'scene':scene,'job':job,'worker':launch.get('worker'),'active':False}
        request,_=read(job)
        if not f.native_web_source:
            f.native_web_source=request['source'];f.native_web_appearance=request.get('appearance') or '';f.native_web_binding=request.get('binding_manifest') or ''
            f.native_web_archive=request['archive'];f.native_web_face_component=request['face_component']
            f.native_web_expression_bank=request.get('original_expression_bank') or request.get('expression_bank') or ''
            f.native_web_legacy_frame=request['legacy_component_frame'];f.native_web_legacy_textures=request['legacy_textures'];f.native_web_legacy_grooms=request['legacy_grooms']
        refresh(_JOB)
        if _JOB['active']:ensure_timer()
    except (OSError,ValueError,KeyError) as error:
        if _JOB and observation_error(_JOB,error):ensure_timer()
        else:f.native_web_status='Cannot restore native web export: '+str(error)


@persistent
def detach(*_):
    global _JOB
    if bpy.app.timers.is_registered(poll):bpy.app.timers.unregister(poll)
    stop_preview()
    _JOB=None  # The owned external worker continues and can be recovered by identity.


class CM_OT_native_web(bpy.types.Operator):
    bl_idname='cm.export_native_web'
    bl_label='Export Rigged Web Bundle'
    bl_description='Build a rigged viewer from a completed native capture, textures and optional artist bakes in a background process'
    action:bpy.props.EnumProperty(items=[(n,n.title(),'') for n in ('START','CANCEL','REFRESH','FOLDER','LOGS','PREVIEW','STOP_PREVIEW','CAPTURE')],default='START')

    def execute(self,context):
        try:
            if self.action=='START':start(context)
            elif self.action=='CANCEL':cancel()
            elif self.action=='REFRESH':
                from . import unreal_jobs
                if unreal_jobs.running() and unreal_jobs._JOB.get('kind')=='CAPTURE_NATIVE_WEB':unreal_jobs.poll()
                else:unreal_jobs.restore_texture_update(context.scene,kind='CAPTURE_NATIVE_WEB');recover();poll()
            elif self.action=='PREVIEW':start_preview(context,open_browser=True)
            elif self.action=='STOP_PREVIEW':stop_preview()
            elif self.action=='CAPTURE':
                if running():raise ValueError('Finish the current web export before capturing another source')
                from . import unreal_jobs
                f=context.scene.cm_features
                unreal_jobs.character_action(context,web_capture={'size':int(f.native_web_capture_size),'detail_size':2048,'face_samples':f.native_web_capture_face_samples,'actor':'@selected' if f.native_web_capture_scope=='SELECTED' else ''})
            else:
                f=context.scene.cm_features
                path=f.native_web_output if self.action=='FOLDER' else f.native_web_job
                if not path or not Path(path).is_dir():raise ValueError('No export folder is available yet')
                bpy.ops.wm.path_open(filepath=path)
            return {'FINISHED'}
        except Exception as error:
            context.scene.cm_features.native_web_status=str(error);self.report({'ERROR'},str(error));return {'CANCELLED'}


def draw(layout,context):
    f=context.scene.cm_features
    box=layout.box();box.label(text='Rigged Native Web Bundle')
    from . import unreal_jobs
    row=box.row();row.enabled=not running() and not unreal_jobs.running() and bool(f.unreal_character or f.native_web_capture_scope=='SELECTED');row.operator('cm.export_native_web',text='Capture Latest From Unreal',icon='IMPORT').action='CAPTURE'
    box.prop(f,'native_web_capture_scope');box.prop(f,'native_web_capture_size')
    box.prop(f,'native_web_capture_face_samples')
    box.prop(f,'native_web_source')
    try:
        captured=json.loads(f.native_web_capture_result or '{}')
        if captured.get('status')=='complete' and captured.get('source')==f.native_web_source and captured.get('facial_samples',0):box.label(text=str(captured['facial_samples'])+' native face samples captured')
    except (ValueError,TypeError):pass
    box.prop(f,'native_web_archive')
    row=box.row();row.enabled=not running() and not unreal_jobs.running();row.operator('cm.export_native_web',icon='EXPORT')
    if running():box.operator('cm.export_native_web',text='Cancel Web Export',icon='CANCEL').action='CANCEL'
    if f.native_web_status:
        from .guided import text
        text(box,f.native_web_status)
    row=box.row(align=True)
    row.operator('cm.export_native_web',text='Refresh',icon='FILE_REFRESH').action='REFRESH'
    if f.native_web_output:row.operator('cm.export_native_web',text='Open Export Folder',icon='FILE_FOLDER').action='FOLDER'
    if f.native_web_output:
        current=f.native_web_source==f.native_web_output_capture
        box.operator('cm.export_native_web',text='Open Web Preview' if current else 'Open Previous Web Preview',icon='WORLD').action='PREVIEW'
        if not current:box.label(text='Export the new capture to refresh the preview.')
    if f.native_web_url:
        box.label(text=f.native_web_url)
        box.operator('cm.export_native_web',text='Stop Local Preview',icon='CANCEL').action='STOP_PREVIEW'
    if f.advanced_ui:
        box.prop(f,'native_web_appearance');box.prop(f,'native_web_expression_bank')
        box.prop(f,'native_web_face_component');box.prop(f,'native_web_legacy_frame');box.prop(f,'native_web_legacy_textures');box.prop(f,'native_web_legacy_grooms')
        if f.native_web_legacy_grooms:box.prop(f,'native_web_binding')
        if f.native_web_job:box.operator('cm.export_native_web',text='Open Export Logs',icon='FILE_TEXT').action='LOGS'
    from .guided import text
    text(box,'Uses a completed native capture. Artist bakes preserve custom eyes and freckles. Includes the lighting studio; native facial/shader equivalence remains unfinished.')


def register():
    bpy.utils.register_class(CM_OT_native_web)
    atexit.register(close_preview_process)
    for handlers,callback in ((bpy.app.handlers.load_pre,detach),(bpy.app.handlers.load_post,recover)):
        if callback not in handlers:handlers.append(callback)
    if not bpy.app.timers.is_registered(recover):bpy.app.timers.register(recover,first_interval=.1)


def unregister():
    detach()
    if bpy.app.timers.is_registered(recover):bpy.app.timers.unregister(recover)
    atexit.unregister(close_preview_process)
    for handlers,callback in ((bpy.app.handlers.load_pre,detach),(bpy.app.handlers.load_post,recover)):
        if callback in handlers:handlers.remove(callback)
    bpy.utils.unregister_class(CM_OT_native_web)
