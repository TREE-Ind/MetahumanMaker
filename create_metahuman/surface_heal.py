"""Prepare masked color or roughness references for the Smart Paint workflow."""
from pathlib import Path
import bpy,json,hashlib,subprocess,os,uuid
import numpy as np
from bpy.props import FloatProperty,EnumProperty
from bpy.app.handlers import persistent
from . import paint_session,skin_projection,map_provenance

_JOB=None
CHANNELS=[('ALBEDO','Skin Color','Heal sRGB base color'),('ROUGHNESS','Roughness','Heal linear roughness data without changing color')]


def reference_key(channel):return 'cm_surface_heal_reference' if channel=='ALBEDO' else 'cm_surface_heal_roughness_reference'


def running():return _JOB is not None


def digest(path):
    with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()


def stamp(context,obj,mask,channel='ALBEDO'):
    cache={};s=context.scene.cm_settings;color=paint_session.map_image(obj,channel)
    return dict(channel=channel,geometry=map_provenance.mesh_digest(context,obj,cache),uv=skin_projection.uv_layout_signature(obj),color=map_provenance.image_digest(color,cache),mask=map_provenance.image_digest(mask,cache),project=str(Path(bpy.path.abspath(s.project_dir)).resolve()),name=s.character_name)


def start(context,*,margin=0.,feather=0.,channel='ALBEDO',component='GRAY'):
    global _JOB
    from . import bake_jobs,detail_jobs,skin_reconstruction,paint_tools,character_project,resource_budget,setup_paths,skin_library
    s,f=context.scene.cm_settings,context.scene.cm_features
    if running() or bake_jobs.running() or detail_jobs.running() or skin_reconstruction.running() or paint_tools.running() or context.mode!='OBJECT' or f.expression_active or f.body_editing or f.full_sculpt:raise ValueError('Finish the current edit, paint session or bake before preparing a heal reference')
    from . import mesh_ops
    character_project.preflight(context,s,f);setup_paths.resolve(f);role,obj=paint_session.part(context);mesh_ops.check_mesh(obj,'Skin part');color=paint_session.map_image(obj,channel);mask=f.skin_solver_mask
    if not mask:raise ValueError('Choose an explicit Heal Region mask first; white is repaired and black is retained')
    space='sRGB' if channel=='ALBEDO' else 'Non-Color'
    if color.colorspace_settings.name!=space or color.source=='TILED' or mask.source=='TILED' or color.is_float:raise ValueError('Surface heal requires a flat 8-bit '+space+' atlas and matching mask')
    if component not in {'GRAY','R','G','B'}:raise ValueError('Choose a supported roughness component')
    if tuple(mask.size)!=tuple(color.size) or not 64<=min(color.size)<=max(color.size)<=8192:raise ValueError('Use a matching repair mask at the skin atlas resolution, through 8K')
    if mask.colorspace_settings.name!='Non-Color':raise ValueError('Set the repair mask color space to Non-Color')
    if mask.get('cm_native_uv_layout') and mask['cm_native_uv_layout']!=skin_projection.uv_layout_signature(obj):raise ValueError('The repair mask belongs to a different UV layout')
    if not np.isfinite([margin,feather]).all() or min(margin,feather)<0:raise ValueError('Choose finite nonnegative surface margins')
    count=int(color.size[0])*int(color.size[1]);root=Path(bpy.path.abspath(s.project_dir)).resolve()
    resource_budget.require('surface heal reference',count*144+512*1024**2);resource_budget.require_disk('surface heal reference',root,count*16+128*1024**2)
    before=stamp(context,obj,mask,channel);points,tri,vertices,uv=skin_projection.surface(context,obj);uv-=np.floor(uv.min(0)+1e-6)
    folder=root/'ReferenceMaps'/'SurfaceHeal'/uuid.uuid4().hex[:12];folder.mkdir(parents=True)
    skin_library.save_image_copy(color,folder/'source.png');skin_library.save_image_copy(mask,folder/'mask.png')
    np.savez_compressed(folder/'surface.npz',points=points,faces=vertices[tri],uv=uv[tri])
    request=dict(schema=1,parent=os.getpid(),inputs={name:digest(folder/name) for name in ('source.png','mask.png','surface.npz')},role=role,stamp=before,margin=margin,feather=feather,channel=channel,component=component)
    (folder/'request.json').write_text(json.dumps(request,indent=2))
    environment=os.environ.copy();environment.update(OPENBLAS_NUM_THREADS='2',OMP_NUM_THREADS='2')
    with (folder/'worker.log').open('w') as log:process=subprocess.Popen([bpy.path.abspath(f.inference_python),str(Path(__file__).with_name('surface_heal_worker.py')),str(folder/'request.json')],stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,env=environment,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
    _JOB=dict(scene=context.scene,obj=obj,mask=mask,stamp=before,role=role,folder=folder,process=process,channel=channel)
    s.status='Preparing a masked surface-heal reference. Accepted skin and geometry are unchanged.'
    bpy.app.timers.register(poll,first_interval=.5)
    return folder


def poll():
    global _JOB
    job=_JOB
    if not job:return None
    if job['process'].poll() is None:return .5
    s=None;folder=job['folder']
    try:
        scene=job['scene'];s,f=scene.cm_settings,scene.cm_features
        report=json.loads((folder/'report.json').read_text())
        if job['process'].returncode or report.get('status')!='complete':raise ValueError(report.get('error','Surface heal worker stopped; inspect its retained log'))
        obj=f.body_result if s.part=='FULL_BODY' and job['role']=='BODY' else s.result
        if obj!=job['obj']:raise ValueError('The selected skin part changed; accepted skin and its reference are retained')
        with bpy.context.temp_override(scene=scene,view_layer=scene.view_layers[0]):
            if stamp(bpy.context,obj,job['mask'],job['channel'])!=job['stamp']:raise ValueError('Skin, mesh, repair mask or project changed; regenerate the reference from the current character')
        if digest(folder/'reference.png')!=report['sha256']:raise ValueError('Surface-heal reference checksum changed')
        if digest(folder/'effective-mask.png')!=report['mask_sha256']:raise ValueError('Surface-heal region checksum changed')
        image=bpy.data.images.load(str(folder/'reference.png'),check_existing=False);image.colorspace_settings.name='sRGB' if job['channel']=='ALBEDO' else 'Non-Color';image.alpha_mode=paint_session.map_image(obj,job['channel']).alpha_mode;image['cm_native_uv_layout']=job['stamp']['uv'];image['cm_surface_heal_report']=str(folder/'report.json');image['cm_surface_heal_channel']=job['channel'];image.pack()
        region=bpy.data.images.load(str(folder/'effective-mask.png'),check_existing=False);region.colorspace_settings.name='Non-Color';region['cm_native_uv_layout']=job['stamp']['uv'];region.pack()
        image['cm_surface_heal_stamp']=json.dumps(job['stamp']);image['cm_surface_heal_mask']=region;image['cm_surface_heal_mask_digest']=map_provenance.image_digest(region,{});image['cm_surface_heal_color_digest']=map_provenance.image_digest(image,{})
        obj[reference_key(job['channel'])]=image
        if job['channel']=='ALBEDO':setattr(f,'reference_skin_body' if job['role']=='BODY' else 'reference_skin',image)
        s.status='Surface-heal reference ready. Use Smart Paint to apply it to the selected region; accepted maps are unchanged.'
    except Exception as exc:
        if s is not None:s.status='Surface heal reference: '+str(exc)
        else:print('Surface heal reference:',exc)
    finally:_JOB=None
    return None


@persistent
def cancel(*_):
    global _JOB
    if _JOB:
        if _JOB['process'].poll() is None:_JOB['process'].terminate()
        _JOB=None
    if bpy.app.timers.is_registered(poll):bpy.app.timers.unregister(poll)


class CM_OT_prepare_surface_heal(bpy.types.Operator):
    bl_idname='cm.prepare_surface_heal'
    bl_label='Create Surface Heal Reference'
    bl_description='Continue nearby accepted color or roughness over an explicit repair mask using mesh connectivity; accepted maps remain unchanged until Apply'
    channel:EnumProperty(name='Heal Map',items=CHANNELS,default='ALBEDO')
    component:EnumProperty(name='Roughness Component',items=[('GRAY','Grayscale RGB','Heal all three equal scalar channels'),('R','Red','Change only red, retaining G/B/A'),('G','Green','Change only green, retaining R/B/A'),('B','Blue','Change only blue, retaining R/G/A')],default='GRAY')
    margin:FloatProperty(name='Surface Expansion',subtype='DISTANCE',min=0,max=1,default=0,description='Expand from selected mesh-vertex seeds over physical mesh edges; approximate at this mesh density')
    feather:FloatProperty(name='Surface Feather',subtype='DISTANCE',min=0,max=1,default=0,description='Smooth the expanded region over the mesh, including split UV charts')
    def invoke(self,context,event):return context.window_manager.invoke_props_dialog(self)
    def draw(self,context):
        self.layout.prop(self,'channel')
        if self.channel=='ROUGHNESS':self.layout.prop(self,'component')
        self.layout.prop(self,'margin');self.layout.prop(self,'feather')
    def execute(self,context):
        try:start(context,margin=self.margin,feather=self.feather,channel=self.channel,component=self.component);return {'FINISHED'}
        except (ValueError,OSError,RuntimeError) as exc:context.scene.cm_settings.status=str(exc);self.report({'ERROR'},str(exc));return {'CANCELLED'}


class CM_OT_cancel_surface_heal(bpy.types.Operator):
    bl_idname='cm.cancel_surface_heal'
    bl_label='Cancel Surface Heal Reference'
    def execute(self,context):cancel();context.scene.cm_settings.status='Surface heal canceled; accepted skin and reference retained.';return {'FINISHED'}


class CM_OT_apply_surface_heal(bpy.types.Operator):
    bl_idname='cm.apply_surface_heal'
    bl_label='Apply Healed Region'
    bl_description='Apply the prepared region through Smart Paint, retaining accepted pixels outside it'
    bl_options={'REGISTER','UNDO'}
    channel:EnumProperty(name='Heal Map',items=CHANNELS,default='ALBEDO')
    def execute(self,context):
        from types import SimpleNamespace
        from . import resource_budget
        s,f=context.scene.cm_settings,context.scene.cm_features;session=None
        try:
            if running():raise ValueError('Finish or cancel the surface-heal reference first')
            role,obj=paint_session.part(context);reference=obj.get(reference_key(self.channel)) if obj else None
            if not isinstance(reference,bpy.types.Image) or self.channel=='ALBEDO' and getattr(f,'reference_skin_body' if role=='BODY' else 'reference_skin')!=reference:raise ValueError('Create and select a surface-heal reference for this skin part first')
            if reference.get('cm_surface_heal_applied'):
                s.status='This healed region is already applied. Create another reference for further cleanup.';return {'FINISHED'}
            if reference.get('cm_surface_heal_channel','ALBEDO')!=self.channel:raise ValueError('The prepared reference belongs to a different map')
            region=reference.get('cm_surface_heal_mask');expected=json.loads(reference['cm_surface_heal_stamp']);current=stamp(context,obj,region,self.channel)
            if any(current[key]!=expected[key] for key in ('geometry','uv','color','project','name')):raise ValueError('Accepted skin or geometry changed; prepare a fresh surface-heal reference')
            if map_provenance.image_digest(region,{})!=reference['cm_surface_heal_mask_digest'] or map_provenance.image_digest(reference,{})!=reference['cm_surface_heal_color_digest']:raise ValueError('The prepared reference or region changed; use Smart Paint manually or regenerate')
            resource_budget.require('Apply surface heal',int(region.size[0])*int(region.size[1])*108+512*1024**2)
            controls=SimpleNamespace(paint_exposure=0,paint_gamma=1,paint_saturation=1,paint_preserve_color=False,paint_detail_radius=8)
            session=paint_session.PaintSession(context,controls=controls,channel=self.channel,reference_image=reference);rgba=paint_session.pixels(region);effective=rgba[:,:,0].copy();del rgba;session.mask[:]=effective>0
            y,x=np.nonzero(effective>0)
            if not len(x):raise ValueError('The prepared heal region is empty')
            session.compose((int(x.min()),int(x.max())+1,int(y.min()),int(y.max())+1));session.mask[:]=effective;session.finish(True)
            reference['cm_surface_heal_applied']=True
            s.status='Healed '+self.channel.lower()+' region applied through Smart Paint. Geometry and other maps retained.'
            try:
                from . import skin_projection_history
                skin_projection_history.retain(context.scene,role,(self.channel,))
            except (ValueError,OSError) as exc:s.status+=' Review the authored-color record: '+str(exc)
            return {'FINISHED'}
        except (ValueError,OSError,RuntimeError,KeyError) as exc:
            if session is not None and not session.closed:session.finish(False)
            s.status=str(exc);self.report({'ERROR'},str(exc));return {'CANCELLED'}


class CM_OT_surface_heal_brush(bpy.types.Operator):
    bl_idname='cm.surface_heal_brush'
    bl_label='Use Last Paint Region'
    bl_description='Use the region retained by Smart Paint or Paint Repair Region as the heal selection'
    bl_options={'REGISTER','UNDO'}
    def execute(self,context):
        s,f=context.scene.cm_settings,context.scene.cm_features;role,obj=paint_session.part(context)
        image=obj.get('cm_last_paint_mask') if obj else None
        if not isinstance(image,bpy.types.Image) or image.get('cm_native_uv_layout')!=skin_projection.uv_layout_signature(obj) or image.get('cm_paint_geometry')!=map_provenance.mesh_digest(context,obj):
            self.report({'ERROR'},'Retain a Smart Paint region on the current skin first');return {'CANCELLED'}
        f.skin_solver_mask=image;s.status='Last Smart Paint region selected for surface healing.';return {'FINISHED'}


def draw(layout,context):
    box=layout.box();box.label(text='Surface Material Heal')
    box.prop(context.scene.cm_features,'skin_solver_mask',text='Heal Region')
    box.operator('cm.surface_heal_brush')
    box.operator('cm.cancel_surface_heal' if running() else 'cm.prepare_surface_heal')
    role,obj=paint_session.part(context)
    if obj:
        for channel,label in [('ALBEDO','Color'),('ROUGHNESS','Roughness')]:
            reference=obj.get(reference_key(channel))
            if reference:
                if reference.get('cm_surface_heal_applied'):box.label(text=label+' heal applied.')
                else:box.operator('cm.apply_surface_heal',text='Apply '+label+' Heal').channel=channel
    box.label(text='White repairs; black retains accepted skin.')
    box.label(text='Creates a reference for Smart Paint.')


CLASSES=(CM_OT_prepare_surface_heal,CM_OT_cancel_surface_heal,CM_OT_apply_surface_heal,CM_OT_surface_heal_brush)


def register():
    for name in ('load_pre','undo_pre','redo_pre'):
        handlers=getattr(bpy.app.handlers,name)
        if cancel not in handlers:handlers.append(cancel)


def unregister():
    cancel()
    for name in ('load_pre','undo_pre','redo_pre'):
        handlers=getattr(bpy.app.handlers,name)
        if cancel in handlers:handlers.remove(cancel)
