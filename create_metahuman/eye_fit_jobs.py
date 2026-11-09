"""Background source-eye fitting with native, transactional three-mesh commits."""
import hashlib,json,threading,uuid
from pathlib import Path
import numpy as np
import bpy
from bpy.app.handlers import persistent
from . import mesh_ops as m

_JOB=None
_LAST_RESULT=None


def running():return _JOB is not None


def status():
    """Machine-readable state of the current or most recent runtime attempt."""
    if _JOB:return dict(job_id=_JOB['id'],kind=_JOB['kind'],running=True,status='canceling' if _JOB['cancel'].is_set() else 'running',phase=_JOB['progress'][0],progress=float(_JOB['progress'][1]))
    return dict(_LAST_RESULT) if _LAST_RESULT else dict(running=False,status='idle')


def _digest(array):return hashlib.sha256(np.asarray(array).tobytes()).hexdigest()


def _caps(obj):
    mesh=obj.data
    if not mesh.uv_layers.active:raise ValueError('The MetaHuman template needs its original UV map')
    uv=np.empty((len(mesh.loops),2));mesh.uv_layers.active.data.foreach_get('uv',uv.ravel());uv=np.rint(uv*1e7).astype(np.int64)
    parent=np.arange(len(mesh.polygons));seen={}
    def root(i):
        while parent[i]!=i:parent[i]=parent[parent[i]];i=parent[i]
        return i
    for poly in mesh.polygons:
        points=[tuple(uv[i]) for i in poly.loop_indices]
        for a,b in zip(points,points[1:]+points[:1]):
            edge=tuple(sorted((a,b)))
            if edge in seen:parent[root(poly.index)]=root(seen[edge])
            else:seen[edge]=poly.index
    regions={}
    for poly in mesh.polygons:regions.setdefault(root(poly.index),[]).append(poly)
    result=[]
    for polys in regions.values():
        if len(polys)>=len(mesh.polygons)*.5:continue
        ids=np.array(sorted({i for poly in polys for i in poly.vertices}));weights={}
        for i in ids:
            for item in mesh.vertices[int(i)].groups:weights[item.group]=weights.get(item.group,0)+item.weight
        names=[obj.vertex_groups[g].name.lower() for g in sorted(weights,key=weights.get,reverse=True)[:3]]
        if any(('eyelid' in name or 'eyecorner' in name) and 'lash' not in name for name in names):result.append(ids)
    if len(result)!=2:raise ValueError('The template must contain two identifiable MetaHuman eye-interior UV regions')
    return result


def _source(context,obj,manual,group_names=('CM_Fit_Eye_L','CM_Fit_Eye_R')):
    from . import resource_budget
    graph=context.evaluated_depsgraph_get();evaluated=obj.evaluated_get(graph);mesh=evaluated.to_mesh(preserve_all_data_layers=True,depsgraph=graph)
    try:
        mesh.calc_loop_triangles();resource_budget.require('source eye geometry',len(mesh.vertices)*96+len(mesh.loop_triangles)*160+128*1024**2)
        points=np.empty((len(mesh.vertices),3));mesh.vertices.foreach_get('co',points.ravel());matrix=np.asarray(obj.matrix_world);points=points@matrix[:3,:3].T+matrix[:3,3]
        triangles=np.empty((len(mesh.loop_triangles),3),dtype=np.int32);mesh.loop_triangles.foreach_get('vertices',triangles.ravel());colors=np.full((len(triangles),3),np.nan);images=[]
        if manual:
            if len(mesh.vertices)!=len(obj.data.vertices):raise ValueError('Apply source geometry modifiers before using painted eye vertex groups')
            masks=[]
            for name in group_names:
                group=obj.vertex_groups.get(name)
                if not group:raise ValueError('Paint source vertex groups '+', '.join(group_names)+' over the corresponding surfaces')
                weights=np.array([next((x.weight for x in v.groups if x.group==group.index),0.) for v in obj.data.vertices]);masks.append(np.all(weights[triangles]>.25,axis=1))
            return points,triangles,colors,images,masks
        if not mesh.uv_layers.active:raise ValueError('Automatic source eye detection needs the source UVs and base-color texture; use named patches for untextured eyes')
        loops=np.empty((len(mesh.loop_triangles),3),dtype=np.int32);mesh.loop_triangles.foreach_get('loops',loops.ravel())
        uv_all=np.empty((len(mesh.loops),2),dtype=np.float32);mesh.uv_layers.active.data.foreach_get('uv',uv_all.ravel());uv=np.empty((len(triangles),2),dtype=float)
        for first in range(0,len(triangles),262144):uv[first:first+262144]=uv_all[loops[first:first+262144]].astype(float).mean(1)
        materials=np.empty(len(triangles),dtype=np.int32);mesh.loop_triangles.foreach_get('material_index',materials)
        for index,mat in enumerate(mesh.materials):
            if not mat or not mat.use_nodes:continue
            shaders=[node for node in mat.node_tree.nodes if node.type=='BSDF_PRINCIPLED']
            if len(shaders)!=1:continue
            socket=shaders[0].inputs.get('Base Color')
            if not socket or len(socket.links)!=1:continue
            node=socket.links[0].from_node
            if node.type!='TEX_IMAGE' or not node.image or node.image.source=='TILED' or node.inputs['Vector'].is_linked:continue
            image=node.image;w,h=map(int,image.size)
            if not w or not h:continue
            resource_budget.require('source eye color',w*h*16+128*1024**2)
            pixels=np.empty(w*h*4,dtype=np.float32);image.pixels.foreach_get(pixels);pixels=pixels.reshape(h,w,4);selected=materials==index
            coords=np.rint((uv[selected]%1)*[w-1,h-1]).astype(int);colors[selected]=pixels[coords[:,1],coords[:,0],:3];images.append((image,_digest(pixels)))
        if not np.isfinite(colors).any():raise ValueError('Automatic eye fitting needs a directly connected base-color image; use named source patches for this material')
        return points,triangles,colors,images,None
    finally:evaluated.to_mesh_clear()


def _stamp(context,obj):
    from .scan_jobs import geometry_stamp
    from .workflow import group_weights
    extra=hashlib.sha256()
    for group in obj.vertex_groups:
        if group.name.startswith(('CM_Fit_Eye_','CM_Fit_Ear_')) or group.name.startswith('FACIAL_') and '_Ear' in group.name:extra.update(group.name.encode()+group_weights(obj,group.name).tobytes())
    for mat in obj.data.materials:
        if not mat or not mat.use_nodes:continue
        images=[(node.name,node.image.as_pointer() if node.image else 0,node.image.colorspace_settings.name if node.image else '') for node in mat.node_tree.nodes if node.type=='TEX_IMAGE']
        links=[(link.from_node.name,link.from_socket.name,link.to_node.name,link.to_socket.name) for link in mat.node_tree.links]
        extra.update(json.dumps([images,links]).encode())
    return geometry_stamp(context,obj,groups=True)+extra.hexdigest()


def _eye_reference(context,obj):
    """Keep repeated source fits from refitting their own deformed globes.

    All artist layers remain active; only this tool's owned curvature layer is
    excluded while reading its input, then restored even on evaluation errors.
    """
    key=obj.data.shape_keys.key_blocks.get('Source Eye Curvature') if obj.data.shape_keys else None
    if key is None:return m.evaluated_coordinates(context,obj)
    value=key.value
    try:
        key.value=0.;context.view_layer.update()
        return m.evaluated_coordinates(context,obj)
    finally:key.value=value;context.view_layer.update()


def prepare(context, *, manual=False):
    """Snapshot source fitting inputs without launching or publishing a job."""
    from . import character_project,setup_paths,skin_projection
    s,f=context.scene.cm_settings,context.scene.cm_features
    if running():raise ValueError('Finish or cancel the current source eye fit first')
    character_project.preflight(context,s,f);setup_paths.resolve(f)
    for obj,label in ((s.template,'Head template'),(s.result,'Accepted head'),(s.source,'Source'),(f.eye_left,'First eye'),(f.eye_right,'Second eye')):m.check_mesh(obj,label)
    if f.eye_left==f.eye_right:raise ValueError('Choose two distinct eye meshes')
    if m.topology(s.result)!=m.topology(s.template):raise ValueError('Eye fitting requires unchanged MetaHuman head topology')
    caps=_caps(s.template);reference=m.coordinates(s.template);current=m.evaluated_coordinates(context,s.result);_,w=np.unique(np.rint(reference/max(float(np.linalg.norm(np.ptp(reference,axis=0)))*1e-6,1e-9)).astype(np.int64),axis=0,return_inverse=True)
    front=np.array({'X':(1,0,0),'-X':(-1,0,0),'Y':(0,1,0),'-Y':(0,-1,0)}[f.front_axis],dtype=float);up=np.array([0.,0.,1.]);basis=np.column_stack((np.cross(up,front),-front,up))
    caps.sort(key=lambda ids:float((current[ids]@basis)[:,0].mean()))
    eyes=[f.eye_left,f.eye_right];eyes.sort(key=lambda obj:float((m.evaluated_coordinates(context,obj)@basis)[:,0].mean()))
    source,source_tri,colors,images,masks=_source(context,s.source,manual);s.result.data.calc_loop_triangles();tri=np.array([t.vertices[:] for t in s.result.data.loop_triangles],dtype=np.int64)
    data=dict(current=current,parameter_reference=reference,triangles=tri,source=source,source_triangles=source_tri,source_eye_colors=colors,eye_basis=basis)
    # Current eye Basis coordinates may already be wrapped or artist-edited.
    # Obtain a consistent pristine head/globe pair from the local native DNA
    # reference, then transport it into the template's coordinate frame.
    from types import SimpleNamespace
    from . import expressions,geometry
    try:native=expressions.bank(SimpleNamespace(expression_reference='',model_dir=f.model_dir))
    except (ValueError,OSError) as exc:raise ValueError('Prepare the original Native Expressions library from installed MetaHuman content before fitting eye anatomy') from exc
    native_head=next(mesh for mesh in native['meshes'] if mesh['name']=='head_lod0_mesh')
    head_map=expressions.uv_mapping(s.template,native_head)
    original=np.asarray(native_head['neutral'])[head_map]*[.01,-.01,.01]
    rotation,scale,translation=geometry.similarity(original,reference)
    data['anatomy_reference_head']=original@rotation*scale+translation
    anatomy={obj:(mesh,indices) for obj,mesh,indices in expressions.accessory_subjects(context,f,native,rotation,scale,translation,allow_alias_proof=False)}
    if masks is not None:
        centers=[source[np.unique(source_tri[mask])].mean(0) if mask.any() else np.full(3,np.nan) for mask in masks]
        if not np.isfinite(centers).all():raise ValueError('Both source eye masks must contain complete triangles')
        order=sorted(range(2),key=lambda i:float((centers[i]@basis)[0]));masks=[masks[i] for i in order]
    for k,ids in enumerate(caps):
        outside=np.setdiff1d(np.arange(len(reference)),ids);rim=ids[np.isin(w[ids],w[outside])]
        if len(rim)<8:raise ValueError('Template eye region has no usable attached boundary')
        data['eye_cap_'+str(k)]=ids;data['eye_rim_'+str(k)]=rim;data['eye_points_'+str(k)]=m.evaluated_coordinates(context,eyes[k])
        eyes[k].data.calc_loop_triangles();data['eye_triangles_'+str(k)]=np.array([t.vertices[:] for t in eyes[k].data.loop_triangles],dtype=np.int32)
        data['eye_parameter_reference_'+str(k)]=_eye_reference(context,eyes[k])
        if eyes[k] not in anatomy:raise ValueError('The native reference does not identify the configured eye anatomy')
        native_eye,indices=anatomy[eyes[k]]
        data['anatomy_eye_'+str(k)]=(np.asarray(native_eye['neutral'])[indices]*[.01,-.01,.01])@rotation*scale+translation
        if masks is not None:data['source_eye_faces_'+str(k)]=masks[k]
    return data,eyes,images,dict(ocular=True,eye_count=2,source_patches=manual)


def start(context, *, manual=False,refine_globes=False,globe_volume=False):
    data,eyes,images,options=prepare(context,manual=manual or refine_globes)
    if refine_globes:options['refine_globes']=True
    if globe_volume:options['globe_volume']=True
    return launch(context,data,eyes,images,options,kind='EYES')


def launch(context,data,eyes,images,options,*,kind,body=None):
    global _JOB,_LAST_RESULT
    from . import skin_projection
    if running() or kind not in {'EYES','EARS'}:raise ValueError('A source surface fit is already running or the feature type is invalid')
    s,f=context.scene.cm_settings,context.scene.cm_features
    subjects=[s.template,s.result,s.source,*eyes]+([body] if body else []);stamps=[_stamp(context,obj) for obj in subjects]
    job=dict(id=uuid.uuid4().hex[:12],kind=kind,stem=kind.lower()[:-1],body=body,scene=context.scene,window=context.window,subjects=subjects,stamps=stamps,images=images,uv=skin_projection.uv_layout_signature(s.template),eyes=eyes,data=data,project=str(Path(bpy.path.abspath(s.project_dir)).resolve()),cancel=threading.Event(),done=False,progress=('Preparing source '+kind.lower(),0.))
    python=bpy.path.abspath(f.inference_python)
    def work():
        from .surface_fit_jobs import run
        try:job['result']=run(data,options,python,job['project'],cancelled=job['cancel'].is_set,progress=lambda name,value:job.update(progress=(name,value)))
        except Exception as exc:job['error']=str(exc)
        finally:job['done']=True
    _LAST_RESULT=None;_JOB=job;job['thread']=threading.Thread(target=work,name='SourceSurfaceFit',daemon=True);job['thread'].start()
    if not bpy.app.timers.is_registered(poll):bpy.app.timers.register(poll,first_interval=.3)
    s.status='Fitting source '+kind.lower()+' in the background.'
    return job['id']


def _snapshot(obj,name):
    key=obj.data.shape_keys.key_blocks.get(name) if obj.data.shape_keys else None
    props={key:obj[key].to_list() if hasattr(obj[key],'to_list') else obj[key] for key in obj.keys() if key.startswith('cm_follow_fit_') or key=='cm_ocular_volume'}
    return dict(obj=obj,name=name,had_keys=bool(obj.data.shape_keys),key=key,coords=np.array([p.co[:] for p in key.data],dtype=np.float32) if key else None,value=float(key.value) if key else 0.,active=obj.active_shape_key_index,only=obj.show_only_shape_key,props=props)


def _restore(state):
    obj=state['obj'];key=obj.data.shape_keys.key_blocks.get(state['name']) if obj.data.shape_keys else None
    if state['key'] is not None:key.data.foreach_set('co',state['coords'].ravel());key.value=state['value']
    elif key:obj.shape_key_remove(key)
    if not state['had_keys'] and obj.data.shape_keys:obj.shape_key_clear()
    obj.active_shape_key_index=state['active'];obj.show_only_shape_key=state['only']
    for key in list(obj.keys()):
        if key.startswith('cm_follow_fit_') or key=='cm_ocular_volume':del obj[key]
    for key,value in state['props'].items():obj[key]=value
    obj.data.update()


def _apply(context,job):
    from . import native_surface_commit,workflow,skin_projection
    s,f=context.scene.cm_settings,context.scene.cm_features
    if context.mode!='OBJECT' or f.expression_active or f.body_editing or f.full_sculpt or f.restore_editing:raise ValueError('Finish the active edit and return to Neutral before applying the source eye result')
    actual=[s.template,s.result,s.source,*job['eyes']]+([f.body_result] if job['body'] else [])
    if actual!=job['subjects'] or {o for o in (f.eye_left,f.eye_right) if o}!=set(job['eyes']) or str(Path(bpy.path.abspath(s.project_dir)).resolve())!=job['project']:raise ValueError('Character or project changed while fitting; previous geometry retained')
    if [_stamp(context,obj) for obj in job['subjects']]!=job['stamps'] or skin_projection.uv_layout_signature(s.template)!=job['uv']:raise ValueError('Anatomy changed while fitting; retry from the current character')
    if job['kind']=='EYES':
        caps=_caps(s.template)
        if {tuple(ids) for ids in caps}!={tuple(job['data']['eye_cap_'+str(i)]) for i in range(2)}:raise ValueError('Template eye semantics changed while fitting; retry from the current template')
    for image,expected in job['images']:
        pixels=np.empty(len(image.pixels),dtype=np.float32);image.pixels.foreach_get(pixels)
        if _digest(pixels)!=expected:raise ValueError('Source color changed while fitting; retry from the current character')
    body,reports,proof=job['result'];ear=job['kind']=='EARS';plans=[(s.result,'Source Ear Reconstruction' if ear else 'Source Eye Opening',job['data']['current'],body,proof['surface_movable'] if ear else proof['ocular_movable'])]
    head_changed=not np.array_equal(body,job['data']['current'])
    if not head_changed:plans=[]
    if not ear:plans.extend((obj,'Source Eye Curvature',job['data']['eye_points_'+str(i)],proof['ocular_eye_'+str(i)],None) for i,obj in enumerate(job['eyes']))
    # Check every planned native write before publishing any of them.
    for obj,name,before,planned,mask in plans:native_surface_commit.commit(context,obj,name,planned,reference=before,movable=mask,publish=False)
    states=[_snapshot(obj,name) for obj,name,*_ in plans];native=[]
    if ear:states.extend(_snapshot(obj,'Source Eye Curvature') for obj in job['eyes'])
    bpy.ops.ed.undo_push(message='Before Source '+job['kind'].title()+' Fit')
    try:
        for obj,name,before,planned,mask in plans:native.append(native_surface_commit.commit(context,obj,name,planned,reference=before,movable=mask,publish=True))
        if job['eyes']:workflow.bind_accessory_fit(context,s,f)
        if not ear:
            from . import ocular_profile
            for index,obj in enumerate(job['eyes']):
                rows=[row for row in reports if row.get('eye')==index]
                volume=rows[0].get('volume') if len(rows)==1 else None
                if volume:
                    obj.data.calc_loop_triangles();tri=np.asarray([t.vertices[:] for t in obj.data.loop_triangles],dtype=np.int32)
                    obj[ocular_profile.PROPERTY]=ocular_profile.make(m.evaluated_coordinates(context,obj),tri,np.asarray(obj.matrix_world),job['data']['eye_basis'],volume)
                elif ocular_profile.PROPERTY in obj:del obj[ocular_profile.PROPERTY]
    except BaseException:
        for state in reversed(states):_restore(state)
        context.view_layer.update()
        for obj,name,before,*_ in plans:np.testing.assert_array_equal(m.evaluated_coordinates(context,obj),before)
        raise
    job['published']=True
    if not ear and f.unreal_character:
        f.unreal_push_status='Local eye geometry changed. Update the linked native character before assembling the delivery.'
    if head_changed and s.detail and s.detail!=s.result:s.detail['cm_head_update_pending']=True
    bpy.ops.ed.undo_push(message='Fit '+job['kind'].title()+' to Source')
    receipt=Path(job['project'])/'SurfaceFit'/('last-'+job['stem']+'-fit.json');receipt.write_text(json.dumps(dict(job_id=job['id'],kind=job['kind'],surface=reports,native=native),indent=2))
    from .expressions import focus_preview
    if not bpy.app.background:focus_preview(context,s,f)
    s.status='Source '+job['kind'].lower()+' fitted in reversible sculpt layers. Review the face and expressions before baking.'


def poll():
    global _JOB,_LAST_RESULT
    job=_JOB
    if not job:return None
    if not job['done']:
        if job.get('scene'):job['scene'].cm_settings.status=job['progress'][0]+f" ({job['progress'][1]*100:.0f}%)"
        if job.get('window'):
            for area in job['window'].screen.areas:area.tag_redraw()
        return .3
    try:
        scene=job.get('scene');window=job.get('window')
        if not scene or not window:return None
        if job['cancel'].is_set():raise ValueError('Source surface fit cancelled; previous geometry retained')
        if job.get('error'):raise ValueError(job['error'])
        with bpy.context.temp_override(window=window,scene=scene,view_layer=scene.view_layers[0]):_apply(bpy.context,job)
    except Exception as exc:
        job['final_error']=str(exc)
        label='Source '+job['kind'].lower()+' fit'
        if job.get('scene'):job['scene'].cm_settings.status=(label+' applied; follow-up failed: ' if job.get('published') else label+': ')+str(exc)
        print(label+':',exc)
    finally:
        outcome='complete' if job.get('published') else 'cancelled' if job['cancel'].is_set() else 'failed'
        _LAST_RESULT=dict(job_id=job['id'],kind=job['kind'],running=False,status=outcome,published=bool(job.get('published')),error=job.get('final_error',''))
        if job.get('published'):_LAST_RESULT['receipt']=str(Path(job['project'])/'SurfaceFit'/('last-'+job['stem']+'-fit.json'))
        try:(Path(job['project'])/'SurfaceFit'/('last-'+job['stem']+'-attempt.json')).write_text(json.dumps(_LAST_RESULT,indent=2))
        except OSError as exc:print('Source eye attempt receipt:',exc)
        _JOB=None
    return None


@persistent
def cancel(*_):
    if _JOB:_JOB['cancel'].set()


@persistent
def detach(*_):
    cancel()
    if _JOB:_JOB['scene']=None;_JOB['window']=None


def register():
    if detach not in bpy.app.handlers.load_pre:bpy.app.handlers.load_pre.append(detach)


def unregister():
    detach()
    if detach in bpy.app.handlers.load_pre:bpy.app.handlers.load_pre.remove(detach)


class CM_OT_fit_source_eyes(bpy.types.Operator):
    bl_idname='cm.fit_source_eyes';bl_label='Fit Eyes to Source'
    bl_description='Reconstruct eye openings and globe curvature from textured source sclera or named source eye patches in the background'
    manual:bpy.props.BoolProperty(name='Use Painted Source Eye Patches',default=False,description='Use CM_Fit_Eye_L and CM_Fit_Eye_R vertex groups instead of automatic light sclera detection')
    refine_globes:bpy.props.BoolProperty(name='Refine Globes Only',default=False,description='Keep the accepted head unchanged while fitting globe surfaces or enclosed volumes; requires painted source eye patches')
    globe_volume:bpy.props.BoolProperty(name='Fit Enclosed Globe Volume',default=False,description='Fit a source-conditioned convex globe instead of copying skin depth into its hidden surface; hidden shape is inferred and rig motion still needs review')
    def invoke(self,context,event):return context.window_manager.invoke_props_dialog(self)
    def draw(self,context):
        self.layout.prop(self,'manual')
        self.layout.prop(self,'refine_globes')
        self.layout.prop(self,'globe_volume')
        if self.manual or self.refine_globes:self.layout.label(text='Source groups: CM_Fit_Eye_L and CM_Fit_Eye_R')
    def execute(self,context):
        try:start(context,manual=self.manual,refine_globes=self.refine_globes,globe_volume=self.globe_volume);return {'FINISHED'}
        except Exception as exc:context.scene.cm_settings.status=str(exc);self.report({'ERROR'},str(exc));return {'CANCELLED'}


class CM_OT_cancel_source_eyes(bpy.types.Operator):
    bl_idname='cm.cancel_source_eyes';bl_label='Cancel Source Fit'
    def execute(self,context):cancel();context.scene.cm_settings.status='Canceling the source surface fit...';return {'FINISHED'}


CLASSES=(CM_OT_fit_source_eyes,CM_OT_cancel_source_eyes)
