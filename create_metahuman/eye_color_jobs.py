"""Transactional local source-eye appearance transfer for people and agents."""
import hashlib
import json
import os
import subprocess
import uuid
from pathlib import Path
import bpy
import numpy as np
from bpy.app.handlers import persistent

_JOB = None
_LAST = {'status':'idle','running':False}
PROPERTY = 'cm_source_eye_color'
BASELINE = 'cm_source_eye_material'


def running():return _JOB is not None


def status():
    if not _JOB:return dict(_LAST)
    path = _JOB['folder']/'report.json'
    try:report = json.loads(path.read_text()) if path.is_file() else {}
    except (OSError,ValueError):report = {}
    return dict(status='running',running=True,job_id=_JOB['id'],progress=report.get('progress',0.),receipt=str(path))


def _stamp(context, eyes):
    from . import map_provenance
    s,f = context.scene.cm_settings,context.scene.cm_features
    cache = {}
    state=dict(project=str(Path(bpy.path.abspath(s.project_dir)).resolve()),name=s.character_name,axis=f.front_axis,hidden_pigment=f.eye_hidden_pigment,
        assignments=[obj.name if obj else None for obj in (s.source,f.eye_left,f.eye_right)],
        objects=[dict(mesh=map_provenance.mesh_digest(context,obj,cache),
                        shader=map_provenance.shader_inputs(obj,cache,surface=True,fast_images=True)) for obj in (s.source,*eyes)])
    if any(obj.get('cm_ocular_volume') for obj in eyes):state['ocular_volumes']=[obj.get('cm_ocular_volume') for obj in eyes]
    return state


def _image(node, arrays, images, source):
    from . import skin_projection
    if node.type!='TEX_IMAGE' or not node.image:raise ValueError('Bake source eye shader inputs into regular images first')
    vector = node.inputs['Vector']
    if vector.is_linked:
        origin = vector.links[0]
        direct = origin.from_node
        if not ((direct.type=='TEX_COORD' and origin.from_socket.name=='UV') or
                (direct.type=='UVMAP' and direct.uv_map in {'',source.data.uv_layers.active.name})):
            raise ValueError('Source eye image must use the active source UV map without a custom mapping transform')
    image = node.image
    if image.source=='TILED' or len(image.size)!=2 or not all(image.size) or node.extension not in {'REPEAT','EXTEND'}:
        raise ValueError('Source-eye transfer needs a regular image with Repeat or Extend mapping')
    key = next((key for key,value in images.items() if value==image),None)
    if key is None:
        key = 'image_'+str(len(images));images[key]=image
        pixels = np.empty(len(image.pixels),np.float32);image.pixels.foreach_get(pixels)
        pixels = pixels.reshape(image.size[1],image.size[0],image.channels)[:,:,:3]
        if image.colorspace_settings.name=='sRGB':pixels = skin_projection.linear(pixels)
        elif image.colorspace_settings.name not in {'Linear Rec.709','Linear','Non-Color'}:
            raise ValueError('Convert the source eye image to sRGB or linear Rec.709 before projection')
        arrays[key] = pixels
    return dict(image=key,wrap=node.extension)


def _scalar(socket, arrays, images, source):
    if not socket.is_linked:return dict(value=float(socket.default_value))
    link=socket.links[0];node=link.from_node
    names={'Red':0,'Green':1,'Blue':2,'R':0,'G':1,'B':2}
    if node.type not in {'SEPARATE_COLOR','SEPRGB'} or link.from_socket.name not in names:
        raise ValueError('Source eye roughness/metallic needs a constant or RGB packed image channel; bake custom source inputs first')
    if node.type=='SEPARATE_COLOR' and node.mode!='RGB':raise ValueError('Use RGB packed source finish channels')
    if not node.inputs[0].is_linked:raise ValueError('Source eye finish channel has no image')
    result=_image(node.inputs[0].links[0].from_node,arrays,images,source)
    result['channel']=names[link.from_socket.name]
    return result


def _material(material, arrays, images, source):
    from . import eye_appearance
    if not material or not material.use_nodes:
        return dict(color=list(material.diffuse_color[:3]) if material else [.8,.8,.8],
                    roughness=dict(value=material.roughness if material else .5),
                    metallic=dict(value=material.metallic if material else 0.))
    output = next((n for n in material.node_tree.nodes if n.type=='OUTPUT_MATERIAL' and n.is_active_output),None)
    shader = eye_appearance.linked(output.inputs['Surface']) if output else None
    if not shader or shader.type != 'BSDF_PRINCIPLED':
        raise ValueError('Source-eye color needs a Principled base-color shader; bake custom source graphs first')
    result={name.lower():_scalar(shader.inputs[name],arrays,images,source) for name in ('Roughness','Metallic')}
    socket = shader.inputs['Base Color']
    if not socket.is_linked:return dict(result,color=list(socket.default_value[:3]))
    node,gain = eye_appearance.color_source(eye_appearance.linked(socket))
    if node is None:raise ValueError('Bake the source eye base color to an image before projecting its custom graph')
    result.update(_image(node,arrays,images,source),gain=gain.tolist())
    return result


def prepare(context):
    from . import character_project,eye_fit_jobs,mesh_ops,resource_budget,skin_projection,setup_paths
    s,f = context.scene.cm_settings,context.scene.cm_features
    character_project.preflight(context,s,f);setup_paths.resolve(f)
    if s.part=='BODY':raise ValueError('Choose a head or full character before projecting source eye appearance')
    if context.mode!='OBJECT' or f.expression_active or f.full_sculpt or f.body_editing or f.restore_editing:
        raise ValueError('Finish the current edit and return to Neutral before projecting source eye color')
    subjects = [f.eye_left,f.eye_right]
    if any(obj is None for obj in subjects) or len(set(subjects))!=2:
        raise ValueError('Choose two distinct fitted eye meshes')
    for obj in subjects:
        mesh_ops.check_mesh(obj,'Eye')
        if obj.data.users!=1 or not obj.data.is_editable or len(obj.data.materials)!=1:
            raise ValueError('Use a private editable eye mesh with one material on each side')
    resource_budget.require('source eye color preparation',resource_budget.GIB)
    points,triangles,_,_,masks = eye_fit_jobs._source(context,s.source,True)
    if not s.source.data.uv_layers.active:raise ValueError('The source eye surface needs its original UV map')
    s.source.data.calc_loop_triangles()
    source_triangles = np.array([t.vertices[:] for t in s.source.data.loop_triangles],dtype=np.int32)
    if not np.array_equal(source_triangles,triangles):raise ValueError('Apply source topology modifiers before projecting marked eye color')
    loops = np.array([t.loops[:] for t in s.source.data.loop_triangles],dtype=np.int32)
    source_uv = np.empty((len(s.source.data.loops),2),np.float32)
    s.source.data.uv_layers.active.data.foreach_get('uv',source_uv.ravel())
    mat_ids = np.array([s.source.data.polygons[t.polygon_index].material_index for t in s.source.data.loop_triangles],dtype=np.int32)
    front = np.array({'X':(1,0,0),'-X':(-1,0,0),'Y':(0,1,0),'-Y':(0,-1,0)}[f.front_axis],float)
    basis = np.column_stack((np.cross([0.,0.,1.],front),-front,[0.,0.,1.]))
    if any(mask.sum()<12 for mask in masks):raise ValueError('Mark complete source globe patches in CM_Fit_Eye_L and CM_Fit_Eye_R')
    order = sorted(range(2),key=lambda i:float((points[np.unique(triangles[masks[i]])]@basis)[:,0].mean()))
    subjects.sort(key=lambda obj:float((mesh_ops.evaluated_coordinates(context,obj)@basis)[:,0].mean()))
    framed=points@basis;all_faces=framed[triangles];regions=[]
    for mask in masks:
        seed=framed[np.unique(triangles[mask])];low=seed.min(0);high=seed.max(0);width=float(high[0]-low[0])
        region=(np.all(all_faces[:,:,[0,2]].min(1)<=high[[0,2]],axis=1)&
                np.all(all_faces[:,:,[0,2]].max(1)>=low[[0,2]],axis=1)&
                (all_faces[:,:,1].min(1)<=high[1]+width)&(all_faces[:,:,1].max(1)>=low[1]-width))
        regions.append(region)
    arrays={};images={};materials=[None]*max(len(s.source.data.materials),1)
    needed = np.unique(mat_ids[np.logical_or.reduce(regions)])
    for index in needed:
        material = s.source.data.materials[int(index)] if len(s.source.data.materials) else None
        materials[int(index)] = _material(material,arrays,images,s.source)
    for i,source_index in enumerate(order):
        mask=regions[source_index];obj=subjects[i];obj.data.calc_loop_triangles()
        if not obj.data.uv_layers.active:raise ValueError('Keep the native eye UV map before source color projection')
        uv=np.empty((len(obj.data.loops),2),np.float32);obj.data.uv_layers.active.data.foreach_get('uv',uv.ravel())
        coords=mesh_ops.evaluated_coordinates(context,obj)
        if len(coords)!=len(obj.data.vertices):raise ValueError('Finish topology-changing eye modifiers before color projection')
        from . import ocular_profile
        eye_triangles=np.array([t.vertices[:] for t in obj.data.loop_triangles],np.int32)
        frame=ocular_profile.frame(obj.get(ocular_profile.PROPERTY),coords,eye_triangles,np.asarray(obj.matrix_world),basis)
        if frame is not None:
            arrays['eye_volume_center_'+str(i)]=frame['center']
            arrays['eye_volume_depth_radius_'+str(i)]=np.asarray(frame['depth_radius'])
        arrays.update({'source_points_'+str(i):points[triangles[mask]]@basis,'source_uv_'+str(i):source_uv[loops[mask]],
                       'source_seed_'+str(i):framed[np.unique(triangles[masks[source_index]])],
                       'source_material_'+str(i):mat_ids[mask],'eye_points_'+str(i):coords@basis,
                       'eye_triangles_'+str(i):np.array([t.vertices[:] for t in obj.data.loop_triangles],np.int32),
                       'eye_uv_'+str(i):uv[np.array([t.loops[:] for t in obj.data.loop_triangles],np.int32)]})
    return arrays,materials,subjects


def start(context):
    global _JOB,_LAST
    from . import bake_jobs,eye_fit_jobs,photo_fit_jobs,resource_budget
    if running() or bake_jobs.running() or eye_fit_jobs.running() or photo_fit_jobs.running():
        raise ValueError('Finish the current bake or source fit before projecting eye color')
    arrays,materials,eyes = prepare(context)
    s,f = context.scene.cm_settings,context.scene.cm_features
    project = Path(bpy.path.abspath(s.project_dir)).resolve()
    resource_budget.require_disk('source-eye color snapshot',project,sum(v.nbytes for v in arrays.values())+256*resource_budget.MIB)
    folder=project/'EyeColor'/uuid.uuid4().hex[:12];folder.mkdir(parents=True)
    stamp=_stamp(context,eyes)
    np.savez_compressed(folder/'input.npz',**arrays)
    request=dict(schema=1,parent=os.getpid(),resolution=2048,materials=materials,eyes=[obj.name for obj in eyes],stamp=stamp,hidden_pigment=f.eye_hidden_pigment)
    (folder/'request.json').write_text(json.dumps(request,indent=2))
    with (folder/'worker.log').open('w',encoding='utf-8') as log:
        process=subprocess.Popen([bpy.path.abspath(f.inference_python),str(Path(__file__).with_name('eye_color_worker.py')),str(folder)],
            stdout=log,stderr=subprocess.STDOUT,creationflags=(subprocess.CREATE_NO_WINDOW|subprocess.BELOW_NORMAL_PRIORITY_CLASS) if os.name=='nt' else 0)
    _JOB=dict(id=folder.name,folder=folder,process=process,scene=context.scene,eyes=eyes,stamp=stamp,cancel=False)
    _LAST=dict(status='running',running=True,job_id=folder.name)
    bpy.app.timers.register(poll,first_interval=.2,persistent=True)
    s.status='Projecting marked source eye color in the background.'
    return folder.name


def _apply(context,job,report):
    from . import eye_appearance
    f=context.scene.cm_features
    if context.mode!='OBJECT' or f.expression_active or f.full_sculpt or f.body_editing or f.restore_editing:
        raise ValueError('Return to Neutral before applying projected eye color')
    if set(job['eyes'])!={f.eye_left,f.eye_right} or _stamp(context,job['eyes'])!=job['stamp']:
        raise ValueError('Source, eye geometry, materials or project changed; previous eye appearance retained')
    originals=[obj.data.materials[0] for obj in job['eyes']];created=[];images=[]
    previous=[{key:obj.get(key) for key in (PROPERTY,BASELINE)} for obj in job['eyes']]
    try:
        for i,(obj,original,row) in enumerate(zip(job['eyes'],originals,report['eyes'])):
            if row.get('file')!='eye_'+str(i)+'.png':raise ValueError('Unexpected projected eye file')
            path=job['folder']/row['file'];image=bpy.data.images.load(str(path),check_existing=False);images.append(image)
            image.colorspace_settings.name='sRGB';image.pack()
            if list(image.size)!=[2048,2048]:raise ValueError('Projected eye image has an invalid resolution')
            material=original.copy();created.append(material);material.use_nodes=True
            output=next((n for n in material.node_tree.nodes if n.type=='OUTPUT_MATERIAL' and n.is_active_output),None)
            shader=eye_appearance.linked(output.inputs['Surface']) if output else None
            if not shader or shader.type!='BSDF_PRINCIPLED':raise ValueError('Retain custom eye graphs in the Material editor; projection needs a Principled eye shader')
            node=material.node_tree.nodes.new('ShaderNodeTexImage');node.name='Source Eye Color';node.image=image
            uv_node=material.node_tree.nodes.new('ShaderNodeUVMap');uv_node.uv_map=obj.data.uv_layers.active.name
            material.node_tree.links.new(uv_node.outputs['UV'],node.inputs['Vector'])
            material.node_tree.links.new(node.outputs['Color'],shader.inputs['Base Color'])
            if row.get('finish_file')!='eye_'+str(i)+'_MR.png':raise ValueError('Unexpected source eye finish file')
            finish=bpy.data.images.load(str(job['folder']/row['finish_file']),check_existing=False);images.append(finish)
            finish.colorspace_settings.name='Non-Color';finish.pack()
            if list(finish.size)!=[2048,2048]:raise ValueError('Projected eye finish has an invalid resolution')
            finish_node=material.node_tree.nodes.new('ShaderNodeTexImage');finish_node.name='Source Eye Finish';finish_node.image=finish
            material.node_tree.links.new(uv_node.outputs['UV'],finish_node.inputs['Vector'])
            separate=material.node_tree.nodes.new('ShaderNodeSeparateColor');separate.mode='RGB'
            material.node_tree.links.new(finish_node.outputs['Color'],separate.inputs['Color'])
            material.node_tree.links.new(separate.outputs['Green'],shader.inputs['Roughness'])
            material.node_tree.links.new(separate.outputs['Blue'],shader.inputs['Metallic'])
            material['cm_source_eye_color']=True
            if not obj.get(BASELINE):obj[BASELINE]=original
            obj[PROPERTY]=json.dumps(dict(receipt=str(job['folder']/'report.json'),image=str(path),projection=row))
        for obj,material in zip(job['eyes'],created):obj.data.materials[0]=material
    except BaseException:
        for obj,original,props in zip(job['eyes'],originals,previous):
            obj.data.materials[0]=original
            for key,value in props.items():
                if value is None:
                    if key in obj:del obj[key]
                else:obj[key]=value
        for material in created:bpy.data.materials.remove(material)
        for image in images:
            if not image.users:bpy.data.images.remove(image)
        raise


def poll():
    global _JOB,_LAST
    job=_JOB
    if not job:return None
    if job['process'].poll() is None:return .3
    try:
        if job['cancel']:raise ValueError('Source eye projection canceled; previous materials retained')
        path=job['folder']/'report.json'
        if not path.is_file():raise ValueError('Source eye worker stopped without a result. See '+str(job['folder']/'worker.log'))
        report=json.loads(path.read_text())
        if job['process'].returncode or report.get('status')!='complete':raise ValueError(report.get('error','Source eye projection failed'))
        if job['scene']!=bpy.context.scene:raise ValueError('Character scene changed; projected eye materials were not applied')
        _apply(bpy.context,job,report)
        _LAST=dict(status='complete',running=False,published=True,job_id=job['id'],receipt=str(job['folder']/'report.json'))
        job['scene'].cm_settings.status='Source eye color and finish applied to both native UV maps. Geometry and previous materials retained.'
    except Exception as exc:
        _LAST=dict(status='canceled' if job['cancel'] else 'failed',running=False,published=False,error=str(exc),job_id=job['id'],receipt=str(job['folder']/'report.json'))
        if job['scene']==bpy.context.scene:job['scene'].cm_settings.status=str(exc)
    finally:
        _JOB=None
        try:(job['folder']/'application.json').write_text(json.dumps(_LAST,indent=2))
        except OSError as exc:print('Source eye application receipt:',exc)
    return None


def cancel():
    if _JOB:
        _JOB['cancel']=True
        if _JOB['process'].poll() is None:_JOB['process'].terminate()


@persistent
def detach(*_):
    if _JOB:_JOB['scene']=None
    cancel()


def register():
    if detach not in bpy.app.handlers.load_pre:bpy.app.handlers.load_pre.append(detach)


def unregister():
    detach()
    if detach in bpy.app.handlers.load_pre:bpy.app.handlers.load_pre.remove(detach)


class CM_OT_source_eye_color(bpy.types.Operator):
    bl_idname='cm.source_eye_color';bl_label='Project Source Eye Appearance'
    bl_description='Project marked source eye color, roughness and metallic to native eye UVs; retain geometry and previous materials'
    def execute(self,context):
        try:start(context);return {'FINISHED'}
        except Exception as exc:self.report({'ERROR'},str(exc));return {'CANCELLED'}


class CM_OT_cancel_eye_color(bpy.types.Operator):
    bl_idname='cm.cancel_eye_color';bl_label='Cancel Eye Color Projection'
    def execute(self,context):cancel();return {'FINISHED'}


class CM_OT_restore_eye_color(bpy.types.Operator):
    bl_idname='cm.restore_eye_color';bl_label='Restore Previous Eye Materials';bl_options={'REGISTER','UNDO'}
    def execute(self,context):
        global _LAST
        f=context.scene.cm_features;eyes=[f.eye_left,f.eye_right]
        if running() or len(set(eyes))!=2 or any(not obj or not obj.get(BASELINE) for obj in eyes):
            self.report({'ERROR'},'Finish projection and retain both previous eye materials before restoring');return {'CANCELLED'}
        for obj in eyes:
            obj.data.materials[0]=obj[BASELINE]
            for key in (BASELINE,PROPERTY):
                if key in obj:del obj[key]
        _LAST=dict(_LAST,status='restored',published=False,running=False)
        context.scene.cm_settings.status='Previous eye materials restored; projected images retained.'
        return {'FINISHED'}


CLASSES=(CM_OT_source_eye_color,CM_OT_cancel_eye_color,CM_OT_restore_eye_color)
