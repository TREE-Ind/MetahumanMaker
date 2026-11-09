"""Immutable UV-bound skin operations retained independently of geometry bakes."""
import hashlib,json,shutil,uuid,os,subprocess,time,math
from pathlib import Path

POINTER='cm_skin_bake_layers'
ENABLED='cm_keep_skin_bake_layers'
EXPECTED='cm_skin_bake_expected_normal'
INPUT_FIELDS=('mask','reference','field','color','before','after','cavity_before','cavity_after')

def material_responses(roots):
    """Recover historical response only from an exact generated-field hash."""
    responses={}
    for root in roots:
        for path in (Path(root)/'SkinSolves').glob('*/state.json'):
            state=json.loads(path.read_text(encoding='utf-8'))
            result_path=path.parent/'result.json';request_path=path.parent/'request.json'
            if state.get('status')!='applied' or not result_path.is_file() or not request_path.is_file():continue
            result=json.loads(result_path.read_text(encoding='utf-8'));sha=result.get('outputs',{}).get('material-height.npy')
            if not sha:continue
            material=result.get('derived_material',{})
            if material.get('method')!='bounded_chroma_gated_albedo_microresponse' or material.get('shared_normal_height_field') is not True:continue
            strength=material.get('strength',0)
            if type(strength) not in (int,float) or not 0<strength<=1:continue
            responses.setdefault(sha,set()).add(float(strength))
    return responses


def enrich_material_response(steps,responses):
    for step in steps:
        if step['kind']!='HEIGHT' or 'cavity_response' in step:continue
        strengths=responses.get(step['field']['sha256'],set())
        if len(strengths)>1:raise ValueError('The retained height has ambiguous cavity strength; retain an explicit material recipe')
        if strengths:step['cavity_response']={'method':'SHARED_HEIGHT','strength':next(iter(strengths))}


def validate_material_responses(steps):
    for step in steps:
        if step['kind']=='SURFACE_PATCH':
            if not all(isinstance(step.get(key),dict) for key in ('before','after')):
                raise ValueError('Regional skin replay is missing its exact normal edit inputs')
        response=step.get('cavity_response')
        if response is None:continue
        if step['kind']=='SURFACE_PATCH' and isinstance(response,dict) and response.get('method')=='REGIONAL_PATCH':
            if not all(isinstance(step.get(key),dict) for key in ('cavity_before','cavity_after')):raise ValueError('Regional cavity replay is missing its edit inputs')
            continue
        if step['kind']!='HEIGHT' or not isinstance(response,dict) or response.get('method')!='SHARED_HEIGHT':raise ValueError('Unsupported retained cavity operation')
        strength=response.get('strength')
        if type(strength) not in (int,float) or not math.isfinite(strength) or not 0<=strength<=1:raise ValueError('Invalid retained cavity strength')


def digest(path):
    with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()

def resolve(root,record):
    root=Path(root).resolve();path=(root/record['file']).resolve()
    if not path.is_relative_to(root) or digest(path)!=record['sha256']:
        raise ValueError('A retained skin layer input changed or left its project')
    return path


def regional_recipe(root,folder,state,request,result):
    """Recover a sealed regional material as its exact accepted before/after edit."""
    recipe=state.get('normal_recipe',{});sha=state.get('outputs',{}).get(state.get('prefix','')+'_NORMAL.png')
    if (state.get('status')!='applied' or not state.get('preserve_material_region') or
        not sha or recipe.get('baseline',{}).get('sha256')!=sha or
        not recipe.get('steps') or any(step['kind']!='RETAIN' for step in recipe['steps'])):return None
    def local(record):
        path=resolve(folder,record)
        if not path.is_relative_to(root):raise ValueError('Regional material input left its source project')
        return {'file':path.relative_to(root).as_posix(),'sha256':record['sha256']}
    before=request.get('inputs',{}).get('region_normal')
    if not before or result.get('outputs',{}).get('candidate-normal.png')!=sha:
        raise ValueError('The sealed material has no verified regional normal input/output pair')
    before=local(before);after=local({'file':'candidate-normal.png','sha256':sha})
    step={'kind':'SURFACE_PATCH','before':before,'after':after,'quantize_after':True}
    old=request['inputs'].get('region_cavity');new=result.get('outputs',{}).get('candidate-cavity.png')
    if old and new:
        step.update(cavity_before=local(old),cavity_after=local({'file':'candidate-cavity.png','sha256':new}),cavity_response={'method':'REGIONAL_PATCH'})
    return dict(schema=1,baseline=before,stamp=recipe['stamp'],steps=[step])

def snapshot(project,source_projects,normal_sha,uv,role):
    """Flatten only hash-linked recipes; validate all inputs before publication."""
    project=Path(project).resolve();roots=list(dict.fromkeys([project,*[Path(p).resolve() for p in source_projects]]))
    definitions={};colors={};baselines={}
    def color(sha,path):
        if sha:colors.setdefault(sha,[]).append(Path(path))
    def add(sha,recipe,deps,origin,root,color_hint=None):
        if not sha or not recipe or recipe.get('schema')!=1:return
        stamp=recipe.get('stamp',{})
        if stamp.get('uv')!=uv or stamp.get('role')!=role:return
        baseline=recipe.get('baseline',{})
        if baseline.get('file'):baselines.setdefault(baseline['sha256'],[]).append((root,baseline))
        if recipe['baseline']['sha256']==sha and all(s['kind']=='RETAIN' for s in recipe['steps']):return
        definitions.setdefault(sha,[]).append({'recipe':recipe,'deps':deps or {},'origin':origin,'root':root,'color':color_hint})
    for root in roots:
        meta_path=root/'Maps/.create-metahuman-bakes.json'
        if meta_path.is_file():
            meta=json.loads(meta_path.read_text(encoding='utf-8'))
            for entry in meta.get('entries',{}).values():
                for name,item in entry.get('artifacts',{}).items():
                    if name.endswith('_ALBEDO.png'):color(item.get('sha256'),root/'Maps'/name)
                    if name.endswith('_NORMAL.png'):add(item.get('sha256'),item.get('skin_recipe'),item.get('dependencies'),str(meta_path),root)
        for path in sorted((root/'SkinSolves').glob('*/state.json')):
            state=json.loads(path.read_text(encoding='utf-8'));prefix=state.get('prefix','');outputs=state.get('outputs',{})
            color_sha=outputs.get(prefix+'_ALBEDO.png');color(color_sha,path.parent/'candidate-albedo.png')
            request_path=path.parent/'request.json'
            request=json.loads(request_path.read_text(encoding='utf-8')) if request_path.is_file() else {}
            for key in ('albedo','accepted'):
                record=request.get('inputs',{}).get(key)
                if record:color(record.get('sha256'),path.parent/record['file'])
            result_path=path.parent/'result.json';hint=None
            if result_path.is_file():
                result=json.loads(result_path.read_text(encoding='utf-8'))
                sha=result.get('outputs',{}).get('candidate-albedo.png')
                if sha:color(sha,path.parent/'candidate-albedo.png');hint=sha
            if request.get('color_cleanup') is False:
                hint=request.get('inputs',{}).get('albedo',{}).get('sha256')
            recovered=regional_recipe(root,path.parent,state,request,result) if result_path.is_file() else None
            add(outputs.get(prefix+'_NORMAL.png'),recovered or state.get('normal_recipe'),{prefix+'_ALBEDO.png':color_sha},str(path),root,hint)
            previous=state.get('previous_normal',{})
            add(previous.get('sha256'),previous.get('skin_recipe'),previous.get('dependencies'),str(path)+' previous',root)
        for path in (root/'SkinLayers').glob('*/bakes/*.json'):
            record=json.loads(path.read_text(encoding='utf-8'))
            add(record.get('normal_sha256'),record.get('recipe'),{},str(path),root)
    inputs={};steps=[];chain=[];seen=set()
    def keep(root,record):
        path=resolve(root,record);sha=record['sha256']
        if path.suffix.lower() not in {'.png','.npy'}:raise ValueError('Unsupported skin layer input type')
        inputs.setdefault(sha,{'source':path,'suffix':path.suffix.lower()})
        return {'file':'inputs/'+sha+path.suffix.lower(),'sha256':sha}
    def color_record(sha):
        for path in colors.get(sha,[]):
            if path.is_file() and digest(path)==sha:
                root=next((r for r in roots if path.resolve().is_relative_to(r)),None)
                return keep(root,{'file':path.resolve().relative_to(root).as_posix(),'sha256':sha})
        raise ValueError('The immutable albedo for an older relief operation is missing: '+str(sha))
    def expand(sha):
        if sha in seen:raise ValueError('Retained skin layer history contains a cycle')
        options=definitions.get(sha,[])
        if not options:return sha
        seen.add(sha);canonical={};prepared=[]
        for item in options:
            recipe=item['recipe'];bound=[]
            relief_color=next((v for k,v in item['deps'].items() if k.endswith('_ALBEDO.png') and v),None) or item['color']
            for step in recipe['steps']:
                op={k:v for k,v in step.items() if k not in INPUT_FIELDS}
                if op['kind'] not in {'REPAIR','RELIEF','HEIGHT','RETAIN','SURFACE_PATCH'}:raise ValueError('Unknown retained skin operation')
                for key in INPUT_FIELDS:
                    if key in step:op[key]=keep(item['root'],step[key])
                if op['kind']=='RELIEF' and 'color' not in op:
                    if not relief_color:raise ValueError('An older relief operation has no retained color identity')
                    op['color']=color_record(relief_color)
                bound.append(op)
            identity={'baseline':recipe['baseline']['sha256'],'steps':[{k:(v['sha256'] if isinstance(v,dict) and 'sha256' in v else v) for k,v in s.items()} for s in bound]}
            # JSON treats integer/float parameter spellings differently; values
            # are normalized here without changing stored worker parameters.
            for step in identity['steps']:
                for key in ('radius','strength'):
                    if key in step:step[key]=float(step[key])
            key=json.dumps(identity,sort_keys=True);canonical[key]=True;prepared.append((item,bound))
        if len(canonical)!=1:raise ValueError('Skin layer history is ambiguous; accepted maps were retained')
        item,bound=prepared[0]
        baseline=item['recipe']['baseline']['sha256']
        no_normal_change=all(step['kind']=='SURFACE_PATCH' and step['before']['sha256']==step['after']['sha256'] for step in bound)
        terminal=sha if baseline==sha and no_normal_change else expand(baseline)
        steps.extend(bound);chain.append({'normal_sha256':sha,'origin':item['origin']})
        return terminal
    terminal=expand(normal_sha)
    if not steps:raise ValueError('No replayable skin operations match the accepted normal map')
    enrich_material_response(steps,material_responses(roots))
    validate_material_responses(steps)
    neutral=None
    for source_root,record in baselines.get(terminal,[]):
        try:neutral=keep(source_root,record);break
        except (OSError,ValueError):continue
    if neutral is None:raise ValueError('Retain the original pre-repair normal inside its source project before keeping skin for new geometry bakes')
    folder=project/'SkinLayers'/uuid.uuid4().hex[:12];(folder/'inputs').mkdir(parents=True)
    for sha,record in inputs.items():
        destination=folder/'inputs'/(sha+record['suffix']);shutil.copy2(record['source'],destination)
        if digest(destination)!=sha:raise ValueError('A skin layer input changed while being retained')
    manifest={'schema':1,'role':role,'uv':uv,'accepted_normal_sha256':normal_sha,'geometry_baseline_sha256':terminal,'neutral_reference':neutral,'steps':steps,'chain':chain}
    path=folder/'layers.json';path.write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    return path,manifest

def retain(context,source_projects=()):
    from . import paint_session,skin_projection,skin_reconstruction,resource_budget
    role,obj=paint_session.part(context);s=context.scene.cm_settings
    normal=skin_reconstruction._normal(obj)
    if not normal or normal.is_dirty:raise ValueError('Save the current normal map before retaining its skin layers')
    project=Path(__import__('bpy').path.abspath(s.project_dir)).resolve()
    resource_budget.require_disk('retaining skin layers',project,2*resource_budget.GIB)
    path,manifest=snapshot(project,source_projects,digest(__import__('bpy').path.abspath(normal.filepath)),skin_projection.uv_layout_signature(obj),role)
    obj[POINTER]=str(path);obj[ENABLED]=True;obj[EXPECTED]=manifest['accepted_normal_sha256']
    obj['cm_skin_cavity_operations']=sum(bool(step.get('cavity_response')) for step in manifest['steps'])
    obj['cm_skin_surface_patches']=sum(step['kind']=='SURFACE_PATCH' for step in manifest['steps'])
    return path,manifest

def load(obj):
    if not obj or not obj.get(ENABLED) or not obj.get(POINTER):return None
    path=Path(obj[POINTER]).resolve();data=json.loads(path.read_text(encoding='utf-8'))
    from . import skin_projection
    if data.get('schema')!=1 or data.get('uv')!=skin_projection.uv_layout_signature(obj):
        raise ValueError('Retained skin layers do not match the current UV layout')
    validate_material_responses(data['steps'])
    if data.get('neutral_reference'):resolve(path.parent,data['neutral_reference'])
    for step in data['steps']:
        for key in INPUT_FIELDS:
            if key in step:resolve(path.parent,step[key])
    return path,data

def fingerprint(obj):
    if not obj or not obj.get(ENABLED) or not obj.get(POINTER):return None
    path=Path(obj[POINTER]);data=json.loads(path.read_text(encoding='utf-8'));stats={}
    for step in data['steps']:
        for key in INPUT_FIELDS:
            if key in step:
                record=step[key];stat=(path.parent/record['file']).stat()
                stats[record['file']]=[stat.st_size,stat.st_mtime_ns]
    if data.get('neutral_reference'):
        record=data['neutral_reference'];stat=(path.parent/record['file']).stat()
        stats[record['file']]=[stat.st_size,stat.st_mtime_ns]
    return hashlib.sha256(json.dumps([digest(path),stats],sort_keys=True).encode()).hexdigest()


def reference_normal(obj):
    """Only a missing reference may fall back to the unrepaired retained neutral."""
    loaded=load(obj)
    if not loaded:return None
    path,data=loaded
    record=data.get('neutral_reference')
    if record is None:raise ValueError('Re-retain skin layers to preserve their original neutral before making a new reference normal')
    return resolve(path.parent,record)

def preflight(scene,*,roles=None):
    import bpy
    from . import skin_reconstruction
    s,f=scene.cm_settings,scene.cm_features
    roles=roles or (('HEAD','BODY') if s.part=='FULL_BODY' else (s.part,))
    objects=[f.body_result if role=='BODY' and s.part=='FULL_BODY' else s.result for role in roles]
    for obj in objects:
        if not obj or not obj.get(ENABLED) or not obj.get(POINTER):continue
        load(obj)
        image=skin_reconstruction._normal(obj)
        if not image or image.is_dirty or digest(bpy.path.abspath(image.filepath))!=obj.get(EXPECTED):
            raise ValueError('The accepted normal changed after retaining skin layers. Retain its updated layers before baking.')

def native_wrinkle_baseline(context,slots=None):
    """Keep authored wrinkle deltas relative to geometry, not repaired skin."""
    import bpy
    from . import skin_projection,skin_reconstruction,map_provenance,skin_normal_history
    scene=context.scene;s,f=scene.cm_settings,scene.cm_features
    choice=f.unreal_wrinkle_neutral_mode
    if choice=='CURRENT':return None
    if choice=='IMAGE':
        image=f.unreal_wrinkle_neutral_image
        if not image or image.source!='FILE' or image.is_dirty or image.colorspace_settings.name!='Non-Color':
            raise ValueError('Choose a saved flat Non-Color original neutral normal for the full wrinkle maps')
        if len(image.size)!=2 or min(image.size)<1 or max(image.size)>8192:
            raise ValueError('The wrinkle neutral image must be through 8K')
        layout=image.get('cm_native_uv_layout')
        if layout and layout!=skin_projection.uv_layout_signature(s.result):raise ValueError('The wrinkle neutral image belongs to another UV layout')
        path=Path(bpy.path.abspath(image.filepath)).resolve()
        if not path.is_file():raise ValueError('Save the original wrinkle neutral image to a file first')
        return path
    if choice!='AUTO':raise ValueError('Choose a supported wrinkle neutral basis')
    loaded=load(s.result)
    if not loaded:
        project=Path(bpy.path.abspath(s.project_dir)).resolve()
        normal=project/'Maps'/(s.character_name+'_NORMAL.png')
        if not normal.is_file():return None
        sha=digest(normal)
        stamp={'uv':skin_projection.uv_layout_signature(s.result),'role':'HEAD','geometry':skin_reconstruction._geometry(context),'sculpt':map_provenance.signature(scene,'FINAL_HEAD',include_retained=False)}
        records=map_provenance.load(scene)['entries']
        artifact=records.get(map_provenance.identity(scene,'FINAL_HEAD'),{}).get('artifacts',{}).get(normal.name,{})
        if artifact.get('skin_recipe') and artifact.get('sha256')!=sha:
            raise ValueError('The accepted normal changed after its material solve; retain its updated history before updating wrinkles')
        base=skin_normal_history.original_baseline(project,sha,stamp,artifact.get('skin_recipe'))
        if base is None:return None
        image=skin_reconstruction._normal(s.result)
        if Path(bpy.path.abspath(image.filepath)).resolve()!=normal:
            raise ValueError('Use the saved head normal before resolving wrinkle skin history')
        slots=slots or [slot for slot in ('WM1','WM2','WM3') if (project/'Maps'/(s.character_name+'_'+slot+'.png')).is_file()]
        for slot in slots:
            if slot not in {'WM1','WM2','WM3'}:raise ValueError('Unknown wrinkle normal slot')
            layer=getattr(f,slot.lower());name=s.character_name+'_'+slot+'.png'
            item=records.get(map_provenance.identity(scene,slot),{}).get('artifacts',{}).get(name,{})
            if not layer or not layer.get('cm_wrinkle_native') or item.get('origin')!='BAKE' or item.get('recipe')!='FINAL':
                raise ValueError(slot+' has no tracked native neutral basis. In Advanced, choose Original Neutral Image, or Maps Include Current Skin if that is how these full maps were authored.')
        problems=map_provenance.problems(scene,keys=slots)
        if problems:raise ValueError(problems[0])
        return base
    path,layers=loaded;project=path.parent.parent.parent
    normal=Path(bpy.path.abspath(s.project_dir))/'Maps'/(s.character_name+'_NORMAL.png');sha=digest(normal)
    receipt=path.parent/'bakes'/(sha+'.json')
    if not receipt.is_file():raise ValueError('Create Final Maps for the retained skin before sending full wrinkle maps to Unreal')
    recipe=json.loads(receipt.read_bytes())['recipe']
    stamp={'uv':skin_projection.uv_layout_signature(s.result),'role':'HEAD','geometry':skin_reconstruction._geometry(context),'sculpt':map_provenance.signature(scene,'FINAL_HEAD',include_retained=False)}
    skin_normal_history.validate(project,recipe,stamp)
    return skin_normal_history.resolve(project,recipe['baseline'])


def replay(context,s):
    """Called after reference/geometry composition, before map publication."""
    import bpy
    from . import resource_budget,setup_paths
    obj=s.result;loaded=load(obj)
    if not loaded:return None
    path,layers=loaded
    project=Path(bpy.path.abspath(s.project_dir)).resolve();normal=project/'Maps'/(s.character_name+'_NORMAL.png')
    with normal.open('rb') as stream:header=stream.read(24)
    size=int.from_bytes(header[16:20],'big'),int.from_bytes(header[20:24],'big')
    cavity=project/'Maps'/(s.character_name+'_CAVITY.png')
    has_cavity=cavity.is_file() and any(step.get('cavity_response') for step in layers['steps'])
    cavity_pixels=0
    if has_cavity:
        with cavity.open('rb') as stream:header=stream.read(24)
        cavity_pixels=int.from_bytes(header[16:20],'big')*int.from_bytes(header[20:24],'big')
    resource_budget.require('retained skin layer replay',max(size[0]*size[1],cavity_pixels)*128+512*resource_budget.MIB)
    resource_budget.require_disk('retained skin layer replay',project,size[0]*size[1]*16+cavity_pixels*8+256*resource_budget.MIB)
    folder=project/'SkinLayerBakes'/uuid.uuid4().hex[:12];folder.mkdir(parents=True)
    carrier=folder/'geometry-normal.png';shutil.copy2(normal,carrier)
    request={'parent':os.getpid(),'manifest':str(path),'manifest_sha256':digest(path),'normal':str(carrier),'normal_sha256':digest(carrier),'output':str(folder/'normal.png'),'progress':str(folder/'progress.json'),'result':str(folder/'result.json')}
    if has_cavity:
        cavity_carrier=folder/'geometry-cavity.png';shutil.copy2(cavity,cavity_carrier)
        request.update(cavity=str(cavity_carrier),cavity_sha256=digest(cavity_carrier),cavity_output=str(folder/'cavity.png'))
    request_path=folder/'request.json';request_path.write_text(json.dumps(request,indent=2),encoding='utf-8')
    setup_paths.resolve(context.scene.cm_features)
    environment=os.environ.copy();environment.update(OPENBLAS_NUM_THREADS='2',OMP_NUM_THREADS='2')
    with (folder/'worker.log').open('w',encoding='utf-8') as log:
        process=subprocess.Popen([bpy.path.abspath(context.scene.cm_features.inference_python),str(Path(__file__).with_name('skin_bake_layers_worker.py')),str(request_path)],stdout=log,stderr=subprocess.STDOUT,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0),env=environment)
    while process.poll() is None:
        if context.scene.get('cm_bake_progress_file'):
            try:
                progress=json.loads((folder/'progress.json').read_text())
                Path(context.scene['cm_bake_progress_file']).write_text(json.dumps({'status':'running','slot':'retained skin detail','phase':f"{progress['phase']} {progress['done']}/{progress['total']}"}))
            except (OSError,ValueError,KeyError):pass
        time.sleep(.2)
    result=json.loads((folder/'result.json').read_text(encoding='utf-8'))
    if process.returncode or result.get('status')!='complete':raise ValueError(result.get('error','Retained skin replay failed'))
    if digest(request['output'])!=result['sha256']:raise ValueError('Replayed skin output changed')
    if request.get('cavity') and digest(request['cavity_output'])!=result.get('cavity_sha256'):raise ValueError('Replayed cavity output changed')
    shutil.copy2(request['output'],normal)
    if request.get('cavity'):shutil.copy2(request['cavity_output'],cavity)
    record={'role':layers['role'],'manifest':str(path),'manifest_sha256':request['manifest_sha256'],'normal_sha256':result['sha256'],'geometry_baseline_sha256':request['normal_sha256'],'geometry_baseline_file':str(carrier),'filename':normal.name,'operations':result['operations']}
    if request.get('cavity'):record.update(cavity_filename=cavity.name,cavity_sha256=result['cavity_sha256'],cavity_operations=result['cavity_operations'],geometry_cavity_sha256=request['cavity_sha256'])
    records=json.loads(context.scene.get('cm_bake_skin_layer_results','[]'));records.append(record);context.scene['cm_bake_skin_layer_results']=json.dumps(records)
    return record

def accept_bake(scene,records):
    """Track completed outputs without making output hashes bake inputs."""
    import bpy
    from . import skin_projection,skin_reconstruction,map_provenance
    for record in records:
        obj=scene.cm_features.body_result if record['role']=='BODY' and scene.cm_settings.part=='FULL_BODY' else scene.cm_settings.result
        path=Path(record['manifest']).resolve()
        if not obj or obj.get(POINTER)!=str(path) or digest(path)!=record['manifest_sha256']:raise ValueError('Retained skin layers changed during baking')
        layers=json.loads(path.read_text(encoding='utf-8'));project=path.parent.parent.parent
        output=project/'Maps'/record['filename']
        if digest(output)!=record['normal_sha256']:raise ValueError('Published skin normal changed')
        if record.get('cavity_sha256') and digest(project/'Maps'/record['cavity_filename'])!=record['cavity_sha256']:raise ValueError('Published skin cavity changed')
        steps=json.loads(json.dumps(layers['steps']))
        for step in steps:
            for key in INPUT_FIELDS:
                if key in step:step[key]['file']=(path.parent/step[key]['file']).relative_to(project).as_posix()
        baseline={'sha256':record['geometry_baseline_sha256']}
        if record.get('geometry_baseline_file'):
            source=Path(record['geometry_baseline_file']);target=path.parent/'bakes/baselines'/(baseline['sha256']+'.png')
            if digest(source)!=baseline['sha256']:raise ValueError('Retained geometry normal changed before publication')
            target.parent.mkdir(parents=True,exist_ok=True)
            if not target.exists():shutil.copy2(source,target)
            if digest(target)!=baseline['sha256']:raise ValueError('Retained geometry baseline changed')
            baseline['file']=target.relative_to(project).as_posix()
        with bpy.context.temp_override(scene=scene,view_layer=scene.view_layers[0]):
            stamp={'uv':skin_projection.uv_layout_signature(obj),'role':record['role'],'geometry':skin_reconstruction._geometry(bpy.context),'sculpt':map_provenance.signature(scene,'FINAL_'+record['role'],include_retained=False)}
        receipt={'normal_sha256':record['normal_sha256'],'recipe':{'schema':1,'baseline':baseline,'stamp':stamp,'steps':steps}}
        if record.get('cavity_sha256'):receipt['cavity']={key:record[key] for key in ('cavity_sha256','geometry_cavity_sha256','cavity_operations')}
        destination=path.parent/'bakes'/(record['normal_sha256']+'.json');destination.parent.mkdir(exist_ok=True)
        destination.write_text(json.dumps(receipt,indent=2),encoding='utf-8');obj[EXPECTED]=record['normal_sha256']
