"""Native DNA expression references, mapped by UV and evaluated on the current fit."""
from pathlib import Path
import json
import re
import os
import subprocess
import uuid
from types import SimpleNamespace
import bpy
import numpy as np
from bpy.props import EnumProperty, BoolProperty, StringProperty
from bpy.app.handlers import persistent
from . import geometry, mesh_ops as m
from . import ocular_visibility

_BANKS = {}
_MAPPINGS = {}
_PREPARATION = None
_PICKER_ITEMS = []  # Blender enum strings must outlive the returned list.
_SYNC_PICKER = False
_EMPTY_ITEMS = [('0','Neutral','Prepare a local reference library')]


def reference_path(f):
    if f.expression_reference:
        return Path(bpy.path.abspath(f.expression_reference))
    root=Path(bpy.path.abspath(f.model_dir))/'local-expressions'
    pointer=root/'library.json'
    if pointer.is_file():
        path=(root/json.loads(pointer.read_text())['reference']).resolve()
        if not path.is_relative_to(root.resolve()):raise ValueError('Expression library pointer is outside its local folder')
        return path
    return root/'reference.json'


def bridge_path(f):
    if f.expression_bridge_dir:return Path(bpy.path.abspath(f.expression_bridge_dir))
    # Enabling an add-on can defer scene initialization until the next timer.
    # Agent/background callers may start a workflow before that timer runs.
    from . import setup_paths
    setup_paths.resolve(f)
    default=Path(bpy.path.abspath(f.model_dir)).parent/'dist/CreateMetaHumanBridge-UE5.8'
    candidates=[default]
    # Native body shaping does not require optional GNM weights. Discover its
    # bridge independently even when a candidate model directory is incomplete.
    candidates.extend(Path(item['model_dir']).parent/'dist/CreateMetaHumanBridge-UE5.8'
                      for item in setup_paths.candidates() if item.get('model_dir'))
    for candidate in candidates:
        if (candidate/'CreateMetaHumanBridge.uplugin').is_file() and (candidate/'Binaries/Win64/UnrealEditor-CreateMetaHumanBridge.dll').is_file():
            return candidate
    return default


def running():
    return bool(_PREPARATION and _PREPARATION['process'].poll() is None)


def activate_reference(context, path):
    s,f=context.scene.cm_settings,context.scene.cm_features
    from . import auto_wrap_jobs
    if auto_wrap_jobs.running():raise ValueError('Finish or cancel Auto Wrap before switching expression references')
    if context.mode!='OBJECT' or f.expression_active or f.restore_editing or f.body_editing or f.full_sculpt:
        raise ValueError('Finish the active edit and return to Neutral before switching expression references')
    data=bank(SimpleNamespace(expression_reference=str(path)))
    if s.result and s.template:
        head=next((mesh for mesh in data['meshes'] if mesh['name']=='head_lod0_mesh'),None)
        if not head:raise ValueError('This reference contains no LOD0 head')
        uv_mapping(s.template,head)
    previous=reference_path(f)
    if previous.resolve()!=Path(path).resolve():f.expression_previous=str(previous)
    f.expression_reference=str(path)
    f.expression_index=0
    f.expression_prepared=''
    sync_picker(f)
    s.status=f'Expression reference ready: {len(data["poses"])} native poses.'
    return data


def poll_preparation():
    global _PREPARATION
    job=_PREPARATION
    if not job:return None
    code=job['process'].poll()
    if code is None:return .5
    _PREPARATION=None
    try:
        if job['cancel_file'].exists():job['cancel_file'].unlink()
        if not job.get('scene'):return None
        s,f=job['scene'].cm_settings,job['scene'].cm_features
        if job.get('canceled'):
            s.status='Expression preparation canceled. Previous reference retained.'
        elif code==0:
            pointer=json.loads((job['root']/'library.json').read_text(encoding='utf-8'))
            path=(job['root']/pointer['reference']).resolve()
            if not path.is_relative_to(job['root'].resolve()):raise ValueError('Prepared reference is outside its local folder')
            bank(SimpleNamespace(expression_reference=str(path)))
            f.expression_prepared=str(path)
            window=next((w for w in bpy.context.window_manager.windows if w.scene==job['scene']),None)
            if window and f.expression_reference==job['previous']:
                try:
                    with bpy.context.temp_override(window=window):activate_reference(bpy.context,path)
                except ValueError:
                    s.status='Expression library prepared. Finish the active edit, then choose Use Prepared Reference.'
            else:s.status='Expression library prepared. Choose Use Prepared Reference when ready.'
        else:s.status='Expression preparation failed. See '+str(job['log'])
    except (ReferenceError,OSError,ValueError) as exc:print('Expression preparation:',exc)
    for window in bpy.context.window_manager.windows:
        for area in window.screen.areas:area.tag_redraw()
    return None


@persistent
def cancel_preparation(_=None):
    if _PREPARATION and _PREPARATION['process'].poll() is None:
        _PREPARATION['canceled']=True
        _PREPARATION['cancel_file'].touch()


@persistent
def detach_preparation(_=None):
    cancel_preparation()
    if _PREPARATION:_PREPARATION['scene']=None


class CM_OT_prepare_expressions(bpy.types.Operator):
    """Prepare local DNA poses using a separate Unreal process; keep editing while it runs"""
    bl_idname='cm.prepare_expressions'
    bl_label='Prepare Native Expressions'
    custom:BoolProperty(default=False,options={'HIDDEN'})
    filepath:StringProperty(subtype='FILE_PATH')
    filter_glob:StringProperty(default='*.dna',options={'HIDDEN'})

    def invoke(self,context,event):
        if self.custom:
            context.window_manager.fileselect_add(self)
            return {'RUNNING_MODAL'}
        return self.execute(context)

    def execute(self,context):
        global _PREPARATION
        s,f=context.scene.cm_settings,context.scene.cm_features
        try:
            if _PREPARATION:raise ValueError('A native library is already being prepared')
            from . import auto_wrap_jobs
            if auto_wrap_jobs.running():raise ValueError('Finish or cancel Auto Wrap before preparing expressions')
            if context.mode!='OBJECT' or f.restore_editing or f.body_editing or f.full_sculpt:
                raise ValueError('Finish the active edit before preparing an expression reference')
            from . import setup_paths
            setup_paths.resolve(f)
            python=Path(bpy.path.abspath(f.inference_python))
            if not python.is_file():raise ValueError('Choose the local inference Python in Advanced Model Paths')
            root=(Path(bpy.path.abspath(s.project_dir))/'Expressions') if self.custom else Path(bpy.path.abspath(f.model_dir))/'local-expressions'
            dna=None
            if self.custom:
                dna=Path(bpy.path.abspath(self.filepath))
                if dna.suffix.lower()!='.dna' or not dna.is_file():raise ValueError('Choose an existing facial DNA file')
            from . import resource_budget
            resource_budget.require('native expression preparation',2*resource_budget.GIB)
            root.mkdir(parents=True,exist_ok=True)
            token=uuid.uuid4().hex[:12]
            cancel_file=root/('cancel-'+token+'.flag')
            log_path=root/('prepare-'+token+'.log')
            command=[str(python),str(Path(__file__).with_name('expression_worker.py')),
                     '--engine',bpy.path.abspath(s.unreal_engine_dir),'--bridge',str(bridge_path(f)),
                     '--destination',str(root),'--cancel-file',str(cancel_file)]
            if dna:command.extend(('--dna',str(dna)))
            with log_path.open('w',encoding='utf-8') as log:
                process=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT,
                    creationflags=(subprocess.CREATE_NO_WINDOW|subprocess.BELOW_NORMAL_PRIORITY_CLASS) if os.name=='nt' else 0)
            _PREPARATION={'process':process,'scene':context.scene,'log':log_path,'root':root,
                          'cancel_file':cancel_file,'previous':f.expression_reference}
            if not bpy.app.timers.is_registered(poll_preparation):
                bpy.app.timers.register(poll_preparation,first_interval=.5,persistent=True)
            s.status='Preparing native expressions in the background. You can keep editing.'
            return {'FINISHED'}
        except Exception as exc:
            s.status=str(exc)
            self.report({'ERROR'},str(exc))
            return {'CANCELLED'}


class CM_OT_expression_library(bpy.types.Operator):
    bl_idname='cm.expression_library'
    bl_label='Expression Reference'
    bl_options={'REGISTER','UNDO'}
    action:EnumProperty(items=[('USE','Use Prepared Reference',''),('DEFAULT','Use Bundled Reference',''),('PREVIOUS','Use Previous Reference',''),('CANCEL','Cancel Preparation','')])

    def execute(self,context):
        s,f=context.scene.cm_settings,context.scene.cm_features
        try:
            if self.action=='CANCEL':
                cancel_preparation()
                s.status='Canceling expression preparation; waiting for the owned Unreal worker to exit.'
            else:
                path=(Path(f.expression_prepared) if self.action=='USE' else Path(f.expression_previous)
                      if self.action=='PREVIOUS' else reference_path(SimpleNamespace(expression_reference='',model_dir=f.model_dir)))
                activate_reference(context,path)
            return {'FINISHED'}
        except (ValueError,OSError) as exc:
            s.status=str(exc);self.report({'ERROR'},str(exc));return {'CANCELLED'}


def binary(folder, name, dtype, shape):
    if Path(name).name != name:
        raise ValueError('Expression data must stay beside its reference manifest')
    path = folder / name
    expected = int(np.prod(shape)) * np.dtype(dtype).itemsize
    if path.stat().st_size != expected:
        raise ValueError('Expression data size does not match its manifest: '+name)
    result = np.fromfile(path, dtype=dtype).reshape(shape)
    if not np.isfinite(result).all():
        raise ValueError('Expression data contains non-finite values')
    return result


def bank(f):
    path = reference_path(f).resolve()
    if not path.is_file():
        raise ValueError('Prepare the local native expression library, or choose a reference JSON')
    stat = path.stat()
    key = (str(path), stat.st_mtime_ns, stat.st_size)
    if key not in _BANKS:
        data = json.loads(path.read_text(encoding='utf-8'))
        if data.get('schema') != 2 or data.get('coordinates') != 'UE_CENTIMETERS' or not data.get('poses'):
            raise ValueError('Use a native expression reference with poses, schema 2, UE centimeters')
        data['_folder'] = path.parent
        _BANKS.clear()
        _BANKS[key] = data
    return _BANKS[key]


def pose_items(f,context):
    try:
        data=bank(f)
        if '_items' not in data:
            inspection=set(data.get('inspection_poses',{}).values())
            order=[0]+[index for index in range(1,len(data['poses'])) if index in inspection]+[index for index in range(1,len(data['poses'])) if index not in inspection]
            data['_items']=[(str(index),re.sub(r'([a-z])([A-Z])',r'\1 \2',data['poses'][index]['name']).capitalize(),
                            'Combined native inspection pose' if index in inspection else 'Native DNA control pose',0,index)
                           for index in order]
            _PICKER_ITEMS.append(data['_items'])
        return data['_items']
    except (ValueError,OSError,AttributeError):return _EMPTY_ITEMS


def sync_picker(f):
    global _SYNC_PICKER
    _SYNC_PICKER=True
    try:f.expression_pose=str(f.expression_index)
    finally:_SYNC_PICKER=False


def update_pose(f,context):
    if _SYNC_PICKER or not context:return
    previous=f.expression_index
    try:
        f.expression_index=int(f.expression_pose)
        preview(context,context.scene.cm_settings,f)
    except Exception as exc:
        f.expression_index=previous
        sync_picker(f)
        context.scene.cm_settings.status=str(exc)


def search_pose_items(self,context):
    return pose_items(context.scene.cm_features,context) if context else _EMPTY_ITEMS


def deformation_equivalent(data, mesh, indices):
    """Prove duplicate native vertices behave identically in this fixed library."""
    indices=tuple(sorted(indices))
    cache=mesh.setdefault('_equivalent_vertices',{})
    if indices in cache:return cache[indices]
    if len(indices)<2 or not data.get('poses'):return False
    neutral=np.asarray(mesh['neutral'])
    if not all(np.array_equal(neutral[indices[0]],neutral[i]) for i in indices[1:]):return False
    count,width=len(neutral),int(mesh['influence_count'])
    weights=binary(data['_folder'],mesh['weights_file'],'<f4',(count,width))
    joints=binary(data['_folder'],mesh['joints_file'],'<u2',(count,width))
    influences=[sorted((int(joint),float(weight)) for joint,weight in zip(joints[i],weights[i]) if weight)
                for i in indices]
    if any(values!=influences[0] for values in influences[1:]):return False
    # Corrective targets can separate vertices with identical rest positions
    # and skin weights. Prove equality across every selectable stored pose too.
    for pose in data['poses']:
        reference=next((item for item in pose['meshes'] if item['index']==mesh['index']),None)
        if not reference:raise ValueError('Expression pose is missing its anatomy mesh')
        name=reference['vertices_file']
        if Path(name).name!=name:raise ValueError('Expression pose data must stay beside its manifest')
        path=data['_folder']/name
        if path.stat().st_size!=count*12:raise ValueError('Expression pose data size does not match its anatomy mesh')
        with path.open('rb') as stream:
            stream.seek(indices[0]*12);first=stream.read(12)
            for index in indices[1:]:
                stream.seek(index*12)
                if stream.read(12)!=first:return False
    cache[indices]=True
    return True


def uv_mapping(obj, mesh, tolerance=3e-6, reference_points=None, reference_data=None):
    if not obj.data.uv_layers:
        raise ValueError('Expression Check requires the unchanged MetaHuman UV map')
    uv = np.empty(len(obj.data.loops)*2, dtype=np.float32)
    obj.data.uv_layers.active.data.foreach_get('uv',uv)
    layouts = np.asarray(mesh['layouts'])
    world = neutral_coordinates(bpy.context,obj) if reference_points is not None else None
    key = (obj.as_pointer(), len(obj.data.vertices), hash(uv.tobytes()), hash(layouts.tobytes()),
           hash(np.asarray(reference_points).tobytes()) if reference_points is not None else None,
           hash(world.tobytes()) if world is not None else None,
           reference_data.setdefault('_mapping_token',object()) if reference_data is not None else None)
    if key in _MAPPINGS:
        return _MAPPINGS[key]
    buckets = {}
    for index, u, v in layouts:
        cell = tuple(np.rint(np.array([u,v])/tolerance).astype(int))
        buckets.setdefault(cell,[]).append((int(index),np.array([u,v])))
    possible = [None]*len(obj.data.vertices)
    cached = {}
    for loop, point in zip(obj.data.loops,uv.reshape(-1,2)):
        point_key=tuple(point)
        if point_key not in cached:
            cell=np.rint(point/tolerance).astype(int)
            matches={index for x in (-1,0,1) for y in (-1,0,1)
                     for index,candidate in buckets.get(tuple(cell+[x,y]),[])
                     if np.linalg.norm(point-candidate)<=tolerance}
            if not matches:
                raise ValueError('Expression reference UVs do not match '+obj.name)
            cached[point_key]=matches
        matches=cached[point_key]
        current=possible[loop.vertex_index]
        possible[loop.vertex_index]=set(matches) if current is None else current & matches
    if reference_points is not None:
        known=[i for i,values in enumerate(possible) if values and len(values)==1]
        if len(known)>=3:
            native=np.asarray([next(iter(possible[i])) for i in known],dtype=np.int32)
            fixed=np.asarray(reference_points)[native]
            if np.linalg.matrix_rank(fixed-fixed.mean(0))>=2:
                rotation,scale,translation=geometry.similarity(fixed,world[known])
                reference_points=np.asarray(reference_points)@rotation*scale+translation
        limit=max(np.linalg.norm(np.ptp(world,axis=0))*.15,1e-5)
        for vertex,values in enumerate(possible):
            if values and len(values)>1:
                candidates=np.array(sorted(values))
                distances=np.linalg.norm(reference_points[candidates]-world[vertex],axis=1)
                order=np.argsort(distances)
                if distances[order[0]]<limit and distances[order[1]]-distances[order[0]]>1e-7:
                    possible[vertex]={int(candidates[order[0]])}
                elif reference_data is not None and deformation_equivalent(reference_data,mesh,values):
                    possible[vertex]={int(candidates[0])}
    if any(values is None or len(values)!=1 for values in possible):
        missing=sum(values is None or len(values)==0 for values in possible)
        ambiguous=sum(values is not None and len(values)>1 for values in possible)
        raise ValueError(f'Expression UV correspondence for {obj.name}: {missing} missing, {ambiguous} ambiguous')
    mapping=np.array([next(iter(values)) for values in possible],dtype=np.int32)
    if len(_MAPPINGS)>16:_MAPPINGS.clear()
    _MAPPINGS[key]=mapping
    return mapping


def skin(points, joints, weights, matrices):
    """UE row-vector linear skinning, in DNA centimeters."""
    output=np.zeros_like(points,dtype=np.float64)
    for slot in range(weights.shape[1]):
        matrix=matrices[joints[:,slot]]
        transformed=np.einsum('ni,nij->nj',points,matrix[:,:3,:3])+matrix[:,3,:3]
        output+=transformed*weights[:,slot,None]
    empty=weights.sum(1)==0
    output[empty]=points[empty]
    return output


def head_pose_linear(context,s,f,*,data=None,pose=None):
    """Per-vertex row-vector world Jacobian for the active native head pose.

    Freeze the alignment during a repair stroke. It is the same global alignment
    used by preview, including the accepted head's current neutral placement.
    """
    data=bank(f) if data is None else data
    pose=data['poses'][f.expression_index%len(data['poses'])] if pose is None else pose
    head=next(mesh for mesh in data['meshes'] if mesh['name']=='head_lod0_mesh')
    indices=uv_mapping(s.template,head)
    current=neutral_coordinates(context,s.result)
    if len(current)!=len(indices):raise ValueError('Repair requires unchanged base head topology')
    source=np.asarray(head['neutral'])[indices]*[.01,-.01,.01]
    rotation,scale,translation=geometry.similarity(source,current)
    count,width=len(head['neutral']),int(head['influence_count'])
    weights=binary(data['_folder'],head['weights_file'],'<f4',(count,width))[indices]
    joints=binary(data['_folder'],head['joints_file'],'<u2',(count,width))[indices]
    matrices=binary(data['_folder'],pose['skin_matrices_file'],'<f4',(int(data['joint_count']),4,4))
    linear=np.zeros((len(indices),3,3),dtype=np.float64)
    for slot in range(width):linear+=matrices[joints[:,slot],:3,:3]*weights[:,slot,None,None]
    linear[weights.sum(1)==0]=np.eye(3)
    to_dna=rotation.T@np.diag([100,-100,100])
    from_dna=np.diag([.01,-.01,.01])@rotation
    return to_dna@linear@from_dna


def neutralize(context,s,f):
    # Keep the guard in the shared API: the generic workflow operator and
    # agent calls must obey the same edit boundaries as Expression Check.
    from . import eye_fit_jobs,paint_tools
    if eye_fit_jobs.running():
        raise ValueError('Finish or cancel the source surface fit before returning to Neutral')
    if context.mode!='OBJECT' or f.full_sculpt or f.body_editing or f.restore_editing:
        raise ValueError('Finish the active edit before returning expression review to Neutral')
    if paint_tools.running():raise ValueError('Apply or cancel Smart Paint before returning to Neutral')
    from . import expression_maps
    expression_maps.restore(s.result)
    f.expression_skin_status=''
    f.expression_active=False
    for obj in dict.fromkeys((s.result,f.head_result,f.eye_left,f.eye_right,f.teeth)):
        if obj and obj.data.shape_keys:
            key=obj.data.shape_keys.key_blocks.get('Expression Check')
            if key:key.value=0
    context.view_layer.update()
    s.status='Neutral restored. Expression previews are excluded from delivery.'


def update_strength(f,context):
    if not context or not f.expression_active:return
    s=context.scene.cm_settings
    from . import expression_maps
    expression_maps.update_strength(s.result,f.expression_strength)
    for obj in dict.fromkeys((s.result,f.head_result,f.eye_left,f.eye_right,f.teeth)):
        if obj and obj.data.shape_keys:
            key=obj.data.shape_keys.key_blocks.get('Expression Check')
            if key:key.value=f.expression_strength


def neutral_coordinates(context,obj):
    key=obj.data.shape_keys.key_blocks.get('Expression Check') if obj.data.shape_keys else None
    old=key.value if key else 0
    try:
        if key:key.value=0
        context.view_layer.update()
        return m.evaluated_coordinates(context,obj)
    finally:
        if key:key.value=old
        context.view_layer.update()


def accessory_subjects(context, f, data, rotation, scale, translation, *, allow_alias_proof=True):
    """Pair reference eyes by fitted anatomy, independent of asset side labels."""
    from itertools import permutations
    meshes={mesh['name']:mesh for mesh in data['meshes']}
    eyes=[obj for obj in (f.eye_left,f.eye_right) if obj]
    if len(set(eyes))!=len(eyes):
        raise ValueError('Choose distinct left and right eye meshes in Anatomy')
    references=[meshes[name] for name in ('eyeLeft_lod0_mesh','eyeRight_lod0_mesh') if name in meshes]
    subjects=[]
    if eyes and references:
        if len(references)<len(eyes):
            raise ValueError('The expression reference does not contain both configured eyes')
        centers=[neutral_coordinates(context,obj).mean(0) for obj in eyes]
        native=[(np.asarray(mesh['neutral'])*[.01,-.01,.01])@rotation*scale+translation for mesh in references]
        choices=[]
        for order in permutations(range(len(references)),len(eyes)):
            cost=sum(float(np.sum((center-native[index].mean(0))**2)) for center,index in zip(centers,order))
            choices.append((cost,order))
        choices.sort()
        if len(choices)>1 and abs(choices[1][0]-choices[0][0])<1e-12:
            raise ValueError('Eye positions do not distinguish anatomical sides; fit or position the eyes before previewing expressions')
        for obj,index in zip(eyes,choices[0][1]):
            mesh=references[index]
            indices=uv_mapping(obj,mesh,reference_points=native[index],reference_data=data if allow_alias_proof else None)
            subjects.append((obj,mesh,indices))
    if f.teeth and 'teeth_lod0_mesh' in meshes:
        mesh=meshes['teeth_lod0_mesh'];reference=(np.asarray(mesh['neutral'])*[.01,-.01,.01])@rotation*scale+translation
        subjects.append((f.teeth,mesh,uv_mapping(f.teeth,mesh,reference_points=reference,reference_data=data if allow_alias_proof else None)))
    return subjects


def evaluate_pose(context,s,f,data,pose):
    """Evaluate an explicit native pose without editing preview keys or scene state."""
    head=next((mesh for mesh in data['meshes'] if mesh['name']=='head_lod0_mesh'),None)
    if not head:
        raise ValueError('The expression reference contains no LOD0 head')
    m.check_mesh(s.template,'Head template')
    m.check_mesh(s.result,'Head result')
    if m.topology(s.result)!=m.topology(s.template):
        raise ValueError('Expression Check requires the unchanged accepted head topology')
    mapping=uv_mapping(s.template,head)
    source=np.asarray(head['neutral'])[mapping]*[.01,-.01,.01]
    # Follow the accepted head's position/scale, including body-height and sculpt passes.
    rotation,scale,translation=geometry.similarity(source,neutral_coordinates(context,s.result))
    matrices=binary(data['_folder'],pose['skin_matrices_file'],'<f4',(int(data['joint_count']),4,4))
    pose_meshes={mesh['index']:mesh for mesh in pose['meshes']}
    # The alias proof covers this library's stored poses. An external ad-hoc
    # pose keeps the strict geometric mapper rather than reusing that proof.
    reference_data=data if pose in data['poses'] else None
    subjects=[(s.result,head,mapping)]
    subjects.extend(accessory_subjects(context,f,data,rotation,scale,translation,allow_alias_proof=reference_data is not None))
    # Build and validate every subject before changing any expression key.
    targets=[]
    for obj,mesh,indices in subjects:
        count=len(mesh['neutral'])
        width=int(mesh['influence_count'])
        weights=binary(data['_folder'],mesh['weights_file'],'<f4',(count,width))
        joints=binary(data['_folder'],mesh['joints_file'],'<u2',(count,width))
        if joints.max(initial=0)>=len(matrices):raise ValueError('Expression joint index is out of range')
        if not np.all((np.abs(weights.sum(1)-1)<1e-4)|(weights.sum(1)==0)):
            raise ValueError('Expression skin weights are invalid')
        current=neutral_coordinates(context,obj)
        if len(current)!=len(indices):raise ValueError('Expression target topology has changed')
        dna=((current-translation)@rotation.T/scale)*[100,-100,100]
        neutral=np.asarray(mesh['neutral'])
        reference=pose_meshes[mesh['index']]
        gold=binary(data['_folder'],reference['vertices_file'],'<f4',(count,3))
        correction=gold-skin(neutral,joints,weights,matrices)
        if mesh['name'] in {'eyeLeft_lod0_mesh','eyeRight_lod0_mesh'}:
            # Imported identities can have eyes far larger and farther apart
            # than the archetype. Evaluate their motion around the fitted eye
            # frame rather than rotating them around an archetype eye center.
            eye_rotation,eye_scale,eye_translation=geometry.similarity(neutral[indices],dna)
            if not np.isfinite(eye_scale) or eye_scale<=0:
                raise ValueError('Eye anatomy does not define a positive motion scale')
            local=((dna-eye_translation)@eye_rotation.T)/eye_scale
            posed=(skin(local,joints[indices],weights[indices],matrices)+correction[indices])@eye_rotation*eye_scale+eye_translation
        else:
            posed=skin(dna,joints[indices],weights[indices],matrices)+correction[indices]
        world=(posed*[.01,-.01,.01])@rotation*scale+translation
        if not np.isfinite(world).all():raise ValueError('Expression produced non-finite geometry')
        targets.append((obj,world))
    return targets


def focus_preview(context, s, f):
    """Show accepted anatomy, excluding retained sculpt/wrinkle author meshes."""
    m.check_mesh(s.result, 'Accepted head')
    from . import compare_views
    session=compare_views.recover(context.window) if context.window else None
    if session:
        compare_views.members(context,session)
        compare_views.fit_objects(context,session['areas'],[s.result],f.front_axis)
        session['previous']=[compare_views.view_state(area) for area in session['areas']]
        session['layouts']=[compare_views.layout_state(area) for area in session['areas']]
        session['driver']=0
        return
    from .guided import focus
    focus(context, [obj for obj in (s.result, f.eye_left, f.eye_right, f.teeth) if obj], f.front_axis)


def preview(context,s,f):
    from . import eye_fit_jobs
    if eye_fit_jobs.running():raise ValueError('Finish or cancel the source surface fit before checking expressions')
    from . import workflow
    from . import paint_tools
    if paint_tools.running():raise ValueError('Apply or cancel Smart Paint before changing expressions')
    if context.mode!='OBJECT' or f.full_sculpt or f.body_editing or f.restore_editing:
        raise ValueError('Finish the current sculpt or body pass before checking expressions')
    was_active = f.expression_active
    data=bank(f)
    pose=data['poses'][f.expression_index % len(data['poses'])]
    if not pose['controls']:
        neutralize(context,s,f)
        sync_picker(f)
        return pose['name']
    targets=evaluate_pose(context,s,f,data,pose)
    # Native previews span the head, both eyes and teeth. Keep every part at
    # the preceding pose if a later key/material publication fails.
    if len({obj.data for obj,_ in targets})!=len(targets):
        raise ValueError('Expression subjects share mesh data. Make the head, eyes and teeth single-user before review.')
    snapshots=[]
    for obj,_ in targets:
        if obj.data.users-int(obj.data.use_fake_user)>1:
            raise ValueError('Make '+obj.name+' mesh data single-user before expression review; linked copies must stay unchanged')
        if not obj.is_editable or not obj.data.is_editable or (obj.data.shape_keys and not obj.data.shape_keys.is_editable):
            raise ValueError('Make '+obj.name+' local and editable before expression review')
        key=obj.data.shape_keys.key_blocks.get('Expression Check') if obj.data.shape_keys else None
        points=None
        if key:
            points=np.empty((len(key.data),3),dtype=np.float32);key.data.foreach_get('co',points.ravel())
        snapshots.append((obj,bool(obj.data.shape_keys),key,points,key.value if key else 0))
    from . import expression_maps
    material,message=expression_maps.build(context,s,f,data,pose)
    previous_materials=list(s.result.data.materials)
    try:
        for obj,world in targets:
            key=workflow.key_from_world(obj,'Expression Check',world)
            key.value=f.expression_strength
        expression_maps.commit(s.result,material)
        f.expression_active=True
        context.view_layer.update()
    except Exception:
        for obj,had_keys,key,points,value in snapshots:
            if key:
                key.data.foreach_set('co',points.ravel());key.value=value
            elif not had_keys:
                if obj.data.shape_keys:obj.shape_key_clear()
            elif obj.data.shape_keys:
                created=obj.data.shape_keys.key_blocks.get('Expression Check')
                if created:obj.shape_key_remove(created)
        for index,previous in enumerate(previous_materials):s.result.data.materials[index]=previous
        while len(s.result.data.materials)>len(previous_materials):
            s.result.data.materials.pop(index=len(s.result.data.materials)-1)
        f.expression_active=was_active
        if material and not material.users:
            material['cm_expression_committed']=False;expression_maps.discard(material)
        context.view_layer.update()
        raise
    f.expression_skin_status=message
    sync_picker(f)
    context.view_layer.update()
    s.status='Native expression: '+pose['name']+'. Return to Neutral before exporting.'
    if not was_active:
        try:focus_preview(context, s, f)
        except (ValueError,RuntimeError):s.status+=' Use Focus Face to adjust the view.'
    return pose['name']


class CM_OT_expression_preview(bpy.types.Operator):
    """Test the accepted head and accessories with a local native DNA pose"""
    bl_idname='cm.expression_preview'
    bl_label='Expression Check'
    bl_options={'REGISTER','UNDO'}
    action:EnumProperty(items=[('APPLY','Preview',''),('PREVIOUS','Previous',''),('NEXT','Next',''),('NEUTRAL','Neutral',''),('FOCUS','Focus Face','Show accepted anatomy without authoring-layer overlays')],default='APPLY')

    def execute(self,context):
        s,f=context.scene.cm_settings,context.scene.cm_features
        previous=f.expression_index
        try:
            if self.action=='FOCUS':
                if context.mode != 'OBJECT' or f.full_sculpt or f.body_editing or f.restore_editing:
                    raise ValueError('Finish the active edit before focusing expression anatomy')
                focus_preview(context,s,f)
            elif self.action=='NEUTRAL':
                neutralize(context,s,f)
            else:
                data=bank(f)
                if self.action!='APPLY':
                    order=[int(item[0]) for item in pose_items(f,context)]
                    position=order.index(f.expression_index%len(data['poses']))
                    f.expression_index=order[(position+(1 if self.action=='NEXT' else -1))%len(order)]
                preview(context,s,f)
            return {'FINISHED'}
        except Exception as exc:
            f.expression_index=previous
            s.status=str(exc)
            self.report({'ERROR'},str(exc))
            return {'CANCELLED'}


class CM_OT_choose_expression(bpy.types.Operator):
    """Search native facial poses by name and preview the selected pose"""
    bl_idname='cm.choose_expression'
    bl_label='Choose Expression'
    bl_options={'REGISTER','UNDO'}
    bl_property='pose'
    pose:EnumProperty(items=search_pose_items)

    def invoke(self,context,event):
        context.window_manager.invoke_search_popup(self)
        return {'RUNNING_MODAL'}

    def execute(self,context):
        s,f=context.scene.cm_settings,context.scene.cm_features
        previous=f.expression_index
        try:
            f.expression_index=int(self.pose)
            preview(context,s,f)
            return {'FINISHED'}
        except Exception as exc:
            f.expression_index=previous
            s.status=str(exc)
            self.report({'ERROR'},str(exc))
            return {'CANCELLED'}


def draw_preparation(layout):
    if _PREPARATION:
        progress=layout.box()
        progress.label(text='Preparing expression reference...',icon='TIME')
        row=progress.row()
        row.enabled=not _PREPARATION.get('canceled')
        row.operator('cm.expression_library',text='Cancel Preparation',icon='CANCEL').action='CANCEL'


def draw(layout,context):
    f=context.scene.cm_features
    layout.prop(f,'show_expression_check',text='Expression Check',toggle=True)
    if not f.show_expression_check:return
    box=layout.box()
    box.enabled=not (f.restore_editing or f.body_editing)
    if f.expression_prepared:
        box.operator('cm.expression_library',text='Use Prepared Reference',icon='CHECKMARK').action='USE'
    try:available=reference_path(f).is_file()
    except (ValueError,OSError):available=False
    if not available:
        if not _PREPARATION:
            box.operator('cm.prepare_expressions',icon='IMPORT')
            box.operator('cm.prepare_expressions',text='Load DNA Reference',icon='FILE_FOLDER').custom=True
            if not bridge_path(f).is_dir():box.prop(f,'expression_bridge_dir',text='Bridge Folder')
        if f.advanced_ui:box.prop(f,'expression_reference',text='Reference')
        return
    try:
        data=bank(f)
        if not f.expression_active:box.label(text='Neutral view')
        if not data.get('animated_maps') and not data.get('dna_source'):
            box.operator('cm.prepare_expressions',text='Update Native Expressions',icon='IMPORT')
        items=pose_items(f,context)
        position=next(i for i,item in enumerate(items) if int(item[0])==f.expression_index%len(data['poses']))
        selected=items[position][1]
        box.operator('cm.choose_expression',text=selected,icon='VIEWZOOM')
        box.operator('cm.expression_preview',text='Focus Face',icon='ZOOM_SELECTED').action='FOCUS'
        box.label(text=f'{position+1} / {len(data["poses"])} native poses')
    except (ValueError,OSError) as exc:
        box.label(text=str(exc),icon='ERROR')
        return
    row=box.row(align=True)
    row.operator('cm.expression_preview',text='',icon='TRIA_LEFT').action='PREVIOUS'
    row.operator('cm.expression_preview',text='Preview').action='APPLY'
    row.operator('cm.expression_preview',text='',icon='TRIA_RIGHT').action='NEXT'
    box.prop(f,'expression_strength',text='Strength')
    box.prop(f,'expression_skin',text='Wrinkles & Blood Flow')
    if f.expression_skin_status:box.label(text=f.expression_skin_status)
    elif f.expression_active and context.scene.cm_settings.result:
        if any(mat and mat.get('cm_expression_original') for mat in context.scene.cm_settings.result.data.materials):
            box.label(text='Native regional skin maps')
    box.operator('cm.expression_preview',text='Return to Neutral').action='NEUTRAL'
    box.operator('cm.inspect_eye_exposure',icon='HIDE_OFF')
    ocular_visibility.draw_last(box,context)
    corrective=sum(int(mesh.get('blendshape_count',0)) for mesh in data['meshes'])
    box.label(text='Joint + corrective deformation' if corrective else 'Joint-only reference',icon='INFO')
    header,settings=box.panel('cm_expression_library',default_closed=True)
    header.label(text='Expression Reference')
    if settings:
        source=data.get('dna_source',{}).get('path') or data.get('dna','')
        settings.label(text=Path(source.replace('\\','/')).name or 'Local DNA reference')
        row=settings.row()
        row.enabled=not _PREPARATION
        row.operator('cm.prepare_expressions',text='Load DNA Reference',icon='FILE_FOLDER').custom=True
        bundled=reference_path(SimpleNamespace(expression_reference='',model_dir=f.model_dir)).resolve()
        if reference_path(f).resolve()!=bundled:
            settings.operator('cm.expression_library',text='Use Bundled Reference').action='DEFAULT'
        if f.expression_previous and Path(f.expression_previous).resolve()!=bundled:
            settings.operator('cm.expression_library',text='Use Previous Reference').action='PREVIOUS'
    if f.advanced_ui:box.prop(f,'expression_reference',text='Reference')


def register():
    if detach_preparation not in bpy.app.handlers.load_pre:bpy.app.handlers.load_pre.append(detach_preparation)


def unregister():
    detach_preparation()
    if detach_preparation in bpy.app.handlers.load_pre:bpy.app.handlers.load_pre.remove(detach_preparation)
    if bpy.app.timers.is_registered(poll_preparation):bpy.app.timers.unregister(poll_preparation)


CLASSES=(CM_OT_expression_preview,CM_OT_choose_expression,CM_OT_prepare_expressions,CM_OT_expression_library,*ocular_visibility.CLASSES)
