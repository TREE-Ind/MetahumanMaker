"""Prepare immutable head revisions and recover their existing Unreal actions."""
import hashlib,json,re,shutil,threading,queue,uuid
from pathlib import Path
import bpy
from . import unreal_jobs,bridge


def digest(path):
    with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()


def history(scene):
    s=scene.cm_settings
    return Path(bpy.path.abspath(s.project_dir)).resolve()/'Export'/re.sub(r'[^A-Za-z0-9_]','_',s.character_name).strip('_')/'last-head-revision.json'


def send(job,engine,project):
    command='\n'.join([
        'import runpy,json,os','from pathlib import Path',
        'folder=Path('+repr(str(job['folder']))+')',
        "identity=runpy.run_path(str(folder/'scripts/process_ref.py'))['identify'](os.getpid())",
        "(folder/'dispatch.json').write_text(json.dumps(identity),encoding='utf-8')",
        'try:',
        "    result=runpy.run_path(str(folder/'scripts/unreal_head_revision.py'))['revise'](str(folder/'request.json'))",
        'except Exception as exc:',"    result={'status':'failed','error':str(exc)}",
        "result['action']='HEAD_REVISION'",
        "pending=folder/'receipt.tmp';pending.write_text(json.dumps(result),encoding='utf-8');pending.replace(folder/'receipt.json')"])
    try:
        bridge.run_python(engine,command,project,on_dispatch=job['dispatched'].set)
        job['events'].put(('returned',None))
    except Exception as exc:job['events'].put(('error',str(exc)))


def start(context,*,defer=False):
    from . import character_project,conform_geometry,mesh_ops,map_provenance,resource_budget,scan_projection,native_geometry_state
    s,f=context.scene.cm_settings,context.scene.cm_features
    if restore(context.scene):
        previous=json.loads(history(context.scene).read_text(encoding='utf-8'));previous_receipt=Path(previous['receipt'])
        previous_native=previous_receipt if previous_receipt.is_file() else previous_receipt.parent/'head-revision.json'
        state=json.loads(previous_native.read_text(encoding='utf-8')).get('status') if previous_native.is_file() else None
        if unreal_jobs.running() or state not in {'complete','failed'}:raise ValueError('Resolve the original head revision receipt before starting another')
    character_project.preflight(context,s,f)
    source=f.unreal_revision_source.strip() or f.unreal_character
    expected=f.unreal_revision_project.strip() or f.unreal_character_project
    if not source.startswith('/Game/'):raise ValueError('Choose a source Creator character under /Game')
    expected_path=Path(bpy.path.abspath(expected)).resolve() if expected else None
    if not expected_path or expected_path.suffix.lower()!='.uproject' or not expected_path.is_file():raise ValueError('Choose the source character’s Unreal project file')
    destination_root=f.unreal_revision_root.rstrip('/')
    if not re.fullmatch(r'/Game(?:/[A-Za-z0-9_]+)+',destination_root):raise ValueError('Choose a revision root under /Game using letters, digits and underscores')
    engine=bpy.path.abspath(s.unreal_engine_dir)
    if not (Path(engine)/'Engine/Plugins/Experimental/PythonScriptPlugin/Content/Python/remote_execution.py').is_file():raise ValueError('Choose the Unreal engine installation')
    if s.part=='BODY':raise ValueError('Select a character with a head before creating a head revision')
    scan_projection.require_applied(s.detail)
    if mesh_ops.collapsed_faces(context,s.result):raise ValueError('Repair collapsed head faces before creating a revision')
    problems=map_provenance.problems(context.scene,keys=['FINAL_HEAD'],suffixes=('_NORMAL.png','_CAVITY.png'))
    if problems:raise ValueError(' '.join(problems))
    project=Path(bpy.path.abspath(s.project_dir)).resolve()
    maps={kind:project/'Maps'/(s.character_name+'_'+kind+'.png') for kind in ('NORMAL','CAVITY')}
    if not all(path.is_file() for path in maps.values()):raise ValueError('Bake head normal and cavity maps before creating a revision')
    if f.unreal_revision_skin_color:
        albedo=project/'Maps'/(s.character_name+'_ALBEDO.png')
        if not albedo.is_file():raise ValueError('Save an accepted head albedo before including skin color')
        maps['ALBEDO']=albedo
        expression={kind:project/'Maps'/(s.character_name+'_'+kind+'.png') for kind in ('CM1','CM2','CM3')}
        present={kind:path for kind,path in expression.items() if path.is_file()}
        if present and len(present)!=3:raise ValueError('Include all three expression color maps, or remove the incomplete set before revising skin color')
        maps.update(present)
    for image in bpy.data.images:
        if image.filepath and Path(bpy.path.abspath(image.filepath)).resolve() in maps.values() and image.is_dirty:
            raise ValueError('Save the selected head map images before creating a revision')
    resource_budget.require('head revision preparation',1024**3)
    resource_budget.require_disk('head revision preparation',project,sum(p.stat().st_size for p in maps.values())+2*1024**3)
    geometry=conform_geometry.prepare(context,s,f)
    token=uuid.uuid4().hex[:12];name=re.sub(r'[^A-Za-z0-9_]','_',s.character_name).strip('_')
    folder=project/'Export/.unreal-actions'/token;snapshot=folder/'scripts';snapshot.mkdir(parents=True,exist_ok=False)
    for filename in ('unreal_head_revision.py','unreal_import.py','resource_budget.py','process_ref.py','unreal_native_animated.py','native_animated_worker.py','worker_lifetime.py','unreal_eye_isolation.py'):shutil.copy2(Path(__file__).with_name(filename),snapshot/filename)
    inputs=folder/'inputs';inputs.mkdir();records={}
    for kind,path in maps.items():
        expected_sha=digest(path);target=inputs/path.name;shutil.copy2(path,target)
        if digest(target)!=expected_sha:raise ValueError('A head map changed during revision preparation')
        records[kind]={'file':str(target),'sha256':expected_sha,'source':str(path),'encoding':f.unreal_color_input_encoding if kind.startswith('CM') else 'FULL'}
    request={'schema':1,'expected_project':str(expected_path),'source_character':source,
        'destination':destination_root+'/'+name+'_'+token+'/'+name+'_Character','geometry':geometry,'project':str(project),
        'maps':records,'image_python':bpy.path.abspath(f.inference_python),'rig_or_texture_request':False,'scope':'head geometry and selected accepted maps; retain other native artist state',
        'local_geometry_witness':native_geometry_state.capture(context,geometry['vertices'])}
    (folder/'request.json').write_text(json.dumps(request,indent=2),encoding='utf-8')
    record={'schema':1,'project_dir':str(project),'character_name':s.character_name,'request_sha256':digest(folder/'request.json'),'receipt':str(folder/'receipt.json')}
    path=history(context.scene);path.parent.mkdir(parents=True,exist_ok=True);temporary=path.with_suffix('.tmp');temporary.write_text(json.dumps(record,indent=2),encoding='utf-8');temporary.replace(path)
    f.unreal_revision_receipt=str(folder/'receipt.json');f.unreal_revision_result=''
    if defer:
        (folder/'prepared.json').write_text(json.dumps(dict(schema=1,request_sha256=record['request_sha256'],native_dispatched=False)),encoding='utf-8')
        f.unreal_push_status='Head revision prepared without native dispatch. Launch this request in the linked Unreal project after closing Blender.'
        return folder
    job=monitor(context.scene,folder);job['thread']=threading.Thread(target=send,args=(job,engine,s.unreal_project),name='CreateMHHeadRevision',daemon=True)
    f.unreal_push_status='Creating a separate head revision from native artist settings. Blender remains usable.'
    job['thread'].start()
    return folder


def monitor(scene,folder):
    s=scene.cm_settings
    job={'scene':scene,'folder':folder,'snapshot':folder/'scripts','receipt':folder/'receipt.json','events':queue.Queue(),
        'dispatched':threading.Event(),'error':None,'kind':'HEAD_REVISION','project_dir':str(Path(bpy.path.abspath(s.project_dir)).resolve()),
        'character_name':s.character_name,'thread':threading.Thread()}
    unreal_jobs._JOB=job
    if not bpy.app.timers.is_registered(unreal_jobs.poll):bpy.app.timers.register(unreal_jobs.poll,first_interval=.2,persistent=True)
    return job


def restore(scene):
    path=history(scene)
    if not path.is_file():return False
    record=json.loads(path.read_text(encoding='utf-8'));s,f=scene.cm_settings,scene.cm_features
    root=Path(bpy.path.abspath(s.project_dir)).resolve();receipt=Path(record['receipt']).resolve();folder=receipt.parent
    if record.get('schema')!=1 or record.get('project_dir')!=str(root) or record.get('character_name')!=s.character_name or not folder.is_relative_to(root/'Export/.unreal-actions'):
        raise ValueError('The head revision history belongs to another project')
    if digest(folder/'request.json')!=record['request_sha256']:raise ValueError('The saved head revision request changed')
    terminal=receipt if receipt.is_file() else folder/'head-revision.json'
    if terminal.is_file():
        result=json.loads(terminal.read_text(encoding='utf-8'))
        if result.get('status') in {'complete','failed'}:
            if result.get('status')=='complete' and result.get('request_sha256')!=record['request_sha256']:raise ValueError('The revision receipt belongs to another request')
            result['action']='HEAD_REVISION'
            general=history(scene).with_name('last-character.json')
            if result.get('status')=='complete' and general.is_file() and general.stat().st_mtime_ns>terminal.stat().st_mtime_ns:return False
            if f.unreal_revision_result!=json.dumps(result) or (result.get('status')=='complete' and f.unreal_character!=result['character']):unreal_jobs.apply_receipt(scene,result)
            f.unreal_revision_receipt=str(receipt)
            return True
    if unreal_jobs._JOB:
        if unreal_jobs._JOB['folder']==folder:unreal_jobs._JOB['scene']=scene;return True
        return False
    if not (folder/'dispatch.json').is_file():
        f.unreal_push_status=('Head revision prepared without native dispatch. Launch its original request in the linked Unreal project.' if (folder/'prepared.json').is_file() else
            'The prepared head revision has no confirmed native dispatch. Inspect its request before creating another.')
        return True
    job=monitor(scene,folder);job['dispatched'].set();f.unreal_revision_receipt=str(receipt)
    f.unreal_push_status='Monitoring the original head revision receipt. No new revision was requested.'
    return True


class CM_OT_head_revision(bpy.types.Operator):
    """Create a separate unrigged Creator head revision retaining native artist settings"""
    bl_idname='cm.create_head_revision';bl_label='Create Head Revision'
    def execute(self,context):
        try:start(context);return {'FINISHED'}
        except Exception as exc:self.report({'ERROR'},str(exc));return {'CANCELLED'}


def draw(layout,context):
    s,f=context.scene.cm_settings,context.scene.cm_features
    if not s.result or s.part=='BODY':return
    box=layout.box();box.label(text='Revise Head from Unreal')
    box.prop(f,'unreal_revision_source');box.prop(f,'unreal_revision_project');box.prop(f,'unreal_revision_root')
    box.prop(f,'unreal_revision_skin_color')
    if not f.unreal_revision_source and f.unreal_character:box.label(text='Source: linked Creator character')
    row=box.row();row.enabled=not unreal_jobs.running();row.operator('cm.create_head_revision',icon='DUPLICATE')
    box.label(text='Keeps native artist settings; updates head, normal and cavity.')
    if f.unreal_revision_skin_color:box.label(text='Also replaces albedo and available expression color maps.')
    box.label(text='Creates a separate unrigged character for Creator review.')


CLASSES=(CM_OT_head_revision,)
