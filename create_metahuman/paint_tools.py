"""Interactive Smart Paint with isolated stroke history and Apply/Cancel."""
import math
import bpy
import numpy as np
from bpy.app.handlers import persistent
from bpy.props import BoolProperty,FloatProperty,IntProperty
from bpy_extras import view3d_utils
from mathutils import Vector
from . import paint_session, lookdev
from .restore_brush import CM_OT_restore_brush, CM_OT_finish_repair, CM_OT_repair_oral_clearance

_ACTIVE=None
PAINT_PROPERTIES=('paint_radius','paint_opacity','mirror','paint_exposure','paint_gamma','paint_saturation',
                  'paint_preserve_color','paint_detail_radius')


def running():
    return _ACTIVE is not None


def draw(layout,context):
    if not _ACTIVE or _ACTIVE.session.scene!=context.scene:return False
    layout.label(text=('Repair Region · ' if _ACTIVE.mask_only else 'Smart Paint · ')+_ACTIVE.session.role.title(),icon='BRUSH_DATA')
    if not _ACTIVE.mask_only:layout.label(text='Reference: '+_ACTIVE.session.reference_source.name)
    if max(_ACTIVE.session.width,_ACTIVE.session.height)>4096:
        layout.label(text='8K paint: full detail after each stroke')
    for prop in PAINT_PROPERTIES:
        if _ACTIVE.mask_only and prop not in {'paint_radius','paint_opacity','mirror'}:continue
        if prop=='paint_detail_radius' and not _ACTIVE.paint_preserve_color:continue
        layout.prop(_ACTIVE,prop)
    row=layout.row(align=True)
    row.operator('cm.reference_paint_history',text='Undo Stroke',icon='LOOP_BACK').redo=False
    row.operator('cm.reference_paint_history',text='Redo Stroke',icon='LOOP_FORWARDS').redo=True
    row=layout.row(align=True)
    row.operator('cm.finish_reference_paint',text='Use Repair Region' if _ACTIVE.mask_only else 'Apply Paint',icon='CHECKMARK').accept=True
    row.operator('cm.finish_reference_paint',text='Cancel',icon='X').accept=False
    if not _ACTIVE.mask_only:
        op=layout.operator('cm.finish_reference_paint',text='Keep Brush Region',icon='BRUSH_DATA')
        op.accept=True;op.mask_only=True
    layout.label(text='Drag repairs · Shift erases · F radius')
    layout.label(text='X mirror · Ctrl+Z undo · Enter apply · Esc cancel')
    return True


class CM_OT_reference_paint(bpy.types.Operator):
    bl_idname='cm.reference_paint'
    bl_label='Smart Paint'
    bl_options={'REGISTER','UNDO','BLOCKING'}
    mask_only: BoolProperty(default=False,options={'HIDDEN'})
    paint_radius: IntProperty(name='Radius (texels)',default=64,min=1,max=1024)
    paint_opacity: FloatProperty(name='Opacity',default=.2,min=.01,max=1)
    mirror: BoolProperty(name='Mirror',default=True)
    paint_exposure: FloatProperty(name='Reference Exposure',default=0,min=-5,max=5)
    paint_gamma: FloatProperty(name='Reference Gamma',default=1,min=.1,max=4)
    paint_saturation: FloatProperty(name='Reference Saturation',default=1,min=0,max=3)
    paint_preserve_color: BoolProperty(name='Keep Local Skin Color',default=False)
    paint_detail_radius: IntProperty(name='Detail Scale (texels)',default=8,min=1,max=64)

    def invoke(self,context,event):
        global _ACTIVE
        if running() or not context.area or context.area.type!='VIEW_3D':
            self.report({'ERROR'},'Start Smart Paint in a 3D viewport after finishing the current session')
            return {'CANCELLED'}
        self.area=context.area;self.manager=context.window_manager
        self.timer=self.handler=None;self.result=None;self.down=False;self.resizing=False;self.cursor=None
        self.shading=self.area.spaces.active.shading.type
        self.render_engine=context.scene.render.engine;self.preview_engine_switched=False
        self.session=paint_session.PaintSession.__new__(paint_session.PaintSession)
        try:
            for prop in PAINT_PROPERTIES:
                setattr(self,prop,getattr(context.scene.cm_features,prop))
            self.session.__init__(context,push_undo=True,controls=self)
            self.region=next(r for r in self.area.regions if r.type=='WINDOW')
            self.area.spaces.active.show_region_ui=True
            sidebar=next((r for r in self.area.regions if r.type=='UI'),None)
            if sidebar and not sidebar.is_property_readonly('active_panel_category'):
                sidebar.active_panel_category='MetaHuman Maker'
            if context.scene.render.engine=='BLENDER_WORKBENCH':
                from .blender_compat import eevee_engine
                context.scene.render.engine=eevee_engine();self.preview_engine_switched=True
            self.area.spaces.active.shading.type='MATERIAL'
            self.area.header_text_set(('Repair Region: drag to mark skin, Shift erases, Enter keeps the mask, Esc cancels' if self.mask_only else 'Smart Paint: drag repairs, Shift erases, F radius, X mirror, Ctrl+Z undo, Enter apply, Esc cancel'))
            self.timer=self.manager.event_timer_add(1/30,window=context.window)
            self.handler=bpy.types.SpaceView3D.draw_handler_add(self.draw_cursor,(),'WINDOW','POST_PIXEL')
            _ACTIVE=self;self.manager.modal_handler_add(self)
            return {'RUNNING_MODAL'}
        except Exception as exc:
            if hasattr(self.session,'image'):self.session.finish(False)
            self.cleanup();self.report({'ERROR'},str(exc))
            return {'CANCELLED'}

    def cleanup(self):
        global _ACTIVE
        if _ACTIVE is self:_ACTIVE=None
        if self.timer:self.manager.event_timer_remove(self.timer);self.timer=None
        if self.handler:
            bpy.types.SpaceView3D.draw_handler_remove(self.handler,'WINDOW');self.handler=None
        try:
            self.area.header_text_set(None);self.area.spaces.active.shading.type=self.shading
            from .blender_compat import is_eevee
            if self.preview_engine_switched and is_eevee(self.session.scene.render.engine):
                self.session.scene.render.engine=self.render_engine
            self.area.tag_redraw()
        except ReferenceError:pass

    def request_finish(self,accept,*,mask_only=False):
        self.session.finish(accept,mask_only=mask_only)
        if accept:
            for prop in PAINT_PROPERTIES:
                setattr(self.session.f,prop,getattr(self,prop))
        self.result={'FINISHED'} if accept else {'CANCELLED'};self.area.tag_redraw()

    def inside(self,event):
        region,(x,y,w,h)=lookdev.visible_rect(self.area)
        return x<=event.mouse_x-region.x<x+w and y<=event.mouse_y-region.y<y+h

    def sample(self,xy):
        rv3d=self.area.spaces.active.region_3d
        origin=view3d_utils.region_2d_to_origin_3d(self.region,rv3d,xy)
        direction=view3d_utils.region_2d_to_vector_3d(self.region,rv3d,xy)
        point,normal,polygon,distance=self.session.bvh.ray_cast(origin,direction)
        if point is None:return None
        uv=self.session.uv_at(point,polygon)
        return (point,polygon,*uv) if uv else None

    def paint(self,xy,erase):
        hit=self.sample(xy)
        if not hit:return
        point,polygon,uv,triangle,triangle_uv=hit
        self.session.stamp_surface(point,polygon,erase)
        if self.mirror:
            reflected=Vector(point);reflected.x=2*self.session.center[0]-reflected.x
            nearest=self.session.bvh.find_nearest(reflected)
            if nearest and nearest[0] is not None:
                other=self.session.uv_at(nearest[0],nearest[2])
                if other is not None and np.linalg.norm((other[0]-uv)*[self.session.width,self.session.height])>.5:
                    self.session.stamp_surface(nearest[0],nearest[2],erase)

    def draw_cursor(self):
        if not self.cursor or bpy.context.area!=self.area:return
        import gpu
        from gpu_extras.batch import batch_for_shader
        angles=np.linspace(0,math.tau,65)
        circle=np.column_stack((np.cos(angles),np.sin(angles)))
        points=circle*18+self.cursor
        hit=self.sample(self.cursor)
        if hit:
            _,_,uv,triangle,triangle_uv=hit
            texel=(triangle_uv[1:]-triangle_uv[0])*[self.session.width,self.session.height]
            if abs(np.linalg.det(texel))>1e-8:
                radius=self.paint_radius*np.sqrt(np.linalg.norm(np.cross(triangle[1]-triangle[0],triangle[2]-triangle[0]))/abs(np.linalg.det(texel)))
                rv3d=self.area.spaces.active.region_3d
                right=rv3d.view_rotation@Vector((1,0,0));up=rv3d.view_rotation@Vector((0,1,0))
                center=hit[0]
                screen=[view3d_utils.location_3d_to_region_2d(self.region,rv3d,
                    Vector(center)+radius*(float(x)*right+float(y)*up)) for x,y in circle]
                if all(p is not None for p in screen):points=np.asarray(screen)
        shader=gpu.shader.from_builtin('UNIFORM_COLOR')
        batch=batch_for_shader(shader,'LINE_STRIP',{'pos':points.tolist()})
        shader.bind();shader.uniform_float('color',(.15,.85,1,1));batch.draw(shader)

    def modal(self,context,event):
        try:
            if self.result:
                result=self.result;self.cleanup();return result
            if context.scene!=self.session.scene or context.mode!='OBJECT':
                self.request_finish(False);self.cleanup();return {'CANCELLED'}
            if event.type=='TIMER':
                self.session.correct();self.session.publish(preview=self.down);self.area.tag_redraw();return {'RUNNING_MODAL'}
            if event.type=='LEFTMOUSE' and event.value=='RELEASE' and self.down:
                self.down=False;self.session.end();return {'RUNNING_MODAL'}
            if not self.inside(event):self.cursor=None;return {'PASS_THROUGH'}
            xy=(event.mouse_x-self.region.x,event.mouse_y-self.region.y);self.cursor=xy;f=self
            if self.resizing:
                f.paint_radius=max(1,min(1024,int(self.resize_radius+xy[0]-self.resize_x)))
                if event.type in {'LEFTMOUSE','RET','NUMPAD_ENTER'} and event.value=='PRESS':self.resizing=False
                elif event.type in {'ESC','RIGHTMOUSE'} and event.value=='PRESS':
                    f.paint_radius=self.resize_radius;self.resizing=False
                return {'RUNNING_MODAL'}
            if event.value=='PRESS':
                if event.type in {'ESC','RET','NUMPAD_ENTER'}:
                    self.request_finish(event.type!='ESC');result=self.result;self.cleanup();return result
                if event.type=='Z' and event.ctrl:self.session.undo(event.shift)
                elif event.type=='F':self.resizing=True;self.resize_x,self.resize_radius=xy[0],f.paint_radius
                elif event.type=='X':f.mirror=not f.mirror
                elif event.type=='LEFTMOUSE':
                    self.session.begin();self.down,self.last=True,xy;self.paint(xy,event.shift)
            if event.type=='MOUSEMOVE' and self.down:
                steps=min(128,max(1,int(np.linalg.norm(np.asarray(xy)-self.last)/3)))
                for point in np.linspace(self.last,xy,steps+1)[1:]:self.paint(tuple(point),event.shift)
                self.last=xy
            if event.type in {'MIDDLEMOUSE','WHEELUPMOUSE','WHEELDOWNMOUSE','NDOF_MOTION'}:return {'PASS_THROUGH'}
            return {'RUNNING_MODAL'}
        except Exception as exc:
            self.report({'ERROR'},str(exc));self.session.finish(False);self.cleanup();return {'CANCELLED'}


class CM_OT_finish_reference_paint(bpy.types.Operator):
    bl_idname='cm.finish_reference_paint'
    bl_label='Finish Smart Paint'
    accept: BoolProperty(default=True)
    mask_only: BoolProperty(default=False)

    def execute(self,context):
        if not _ACTIVE:return {'CANCELLED'}
        try:_ACTIVE.request_finish(self.accept,mask_only=self.mask_only);return {'FINISHED'}
        except Exception as exc:self.report({'ERROR'},str(exc));return {'CANCELLED'}


class CM_OT_reference_paint_history(bpy.types.Operator):
    bl_idname='cm.reference_paint_history'
    bl_label='Smart Paint Stroke History'
    redo: BoolProperty(default=False)

    def execute(self,context):
        if not _ACTIVE:return {'CANCELLED'}
        _ACTIVE.session.undo(self.redo);return {'FINISHED'}


@persistent
def cancel_active(*_):
    if _ACTIVE:
        operator=_ACTIVE;operator.request_finish(False);operator.cleanup()


@persistent
def undo_changed(*_):
    if not running() and hasattr(bpy.context.scene,'cm_features'):
        try:paint_session.synchronize_undo(bpy.context.scene)
        except Exception as exc:bpy.context.scene.cm_settings.status='Smart Paint map sync: '+str(exc)


@persistent
def load_finished(*_):
    paint_session.reset_history_cache()


def register():
    for name,callback in (('load_pre',cancel_active),('load_post',load_finished),('undo_pre',cancel_active),('redo_pre',cancel_active),('undo_post',undo_changed),('redo_post',undo_changed)):
        handlers=getattr(bpy.app.handlers,name)
        if callback not in handlers:handlers.append(callback)


def unregister():
    cancel_active()
    for name,callback in (('load_pre',cancel_active),('load_post',load_finished),('undo_pre',cancel_active),('redo_pre',cancel_active),('undo_post',undo_changed),('redo_post',undo_changed)):
        handlers=getattr(bpy.app.handlers,name)
        if callback in handlers:handlers.remove(callback)


CLASSES=(CM_OT_restore_brush,CM_OT_finish_repair,CM_OT_repair_oral_clearance,CM_OT_reference_paint,CM_OT_finish_reference_paint,CM_OT_reference_paint_history)
