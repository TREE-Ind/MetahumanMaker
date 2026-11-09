"""Explicit per-part choice between retained skin normals and sculpt geometry."""
import bpy,math
from bpy.props import EnumProperty,FloatProperty

PROPERTY='cm_normal_composition'
DISTANCE='cm_normal_projection_distance'


def projection_distance(obj,default):
    value=float(obj.get(DISTANCE,default)) if obj else float(default)
    if not math.isfinite(value) or value<=0:raise ValueError('Normal projection distance must be positive and finite')
    return value


class CM_OT_normal_projection(bpy.types.Operator):
    bl_idname='cm.normal_projection'
    bl_label='Normal Projection Distance'
    bl_description='Set a separate normal-map ray distance for this part, leaving source color transfer unchanged'
    bl_options={'REGISTER','UNDO'}
    mode:EnumProperty(items=[('CUSTOM','Separate Normal Distance','Use a shorter cage for closely fitted surfaces'),('SOURCE','Use Source Transfer Distance','Inherit the main projection distance')],default='CUSTOM')
    distance:FloatProperty(name='Distance',subtype='DISTANCE',min=.000001,default=.005)

    def invoke(self,context,event):
        s,f=context.scene.cm_settings,context.scene.cm_features
        obj=f.body_result if s.part=='FULL_BODY' and f.texture_part=='BODY' else s.result
        if obj and DISTANCE in obj:self.distance=projection_distance(obj,s.bake_distance)
        return context.window_manager.invoke_props_dialog(self)

    def draw(self,context):
        self.layout.prop(self,'mode')
        if self.mode=='CUSTOM':self.layout.prop(self,'distance')

    def execute(self,context):
        from . import bake_jobs,detail_jobs
        s,f=context.scene.cm_settings,context.scene.cm_features
        obj=f.body_result if s.part=='FULL_BODY' and f.texture_part=='BODY' else s.result
        try:
            if context.mode!='OBJECT' or bake_jobs.running() or detail_jobs.running():raise ValueError('Finish the current edit or bake before changing normal projection')
            if not obj:raise ValueError('Choose a fitted head or body first')
            if self.mode=='CUSTOM':
                if not math.isfinite(self.distance) or self.distance<=0:raise ValueError('Choose a positive normal projection distance')
                obj[DISTANCE]=self.distance
            elif DISTANCE in obj:del obj[DISTANCE]
            s.status='Normal ray distance changed for this part. Bake final maps to update it; color and sculpt geometry are retained.'
            return {'FINISHED'}
        except ValueError as exc:s.status=str(exc);self.report({'ERROR'},str(exc));return {'CANCELLED'}


def geometry_only(obj):
    mode=obj.get(PROPERTY,'COMBINE') if obj else 'COMBINE'
    if mode not in {'COMBINE','GEOMETRY_ONLY'}:raise ValueError('Choose a supported normal composition mode')
    return mode=='GEOMETRY_ONLY'


def validate(obj,detail):
    if geometry_only(obj):
        if not detail:raise ValueError('Create a sculpt layer before choosing geometry-only normals')
        if obj.get('cm_keep_skin_bake_layers') and obj.get('cm_skin_bake_layers'):raise ValueError('Disable Keep Retained Skin Layers before choosing geometry-only normals; its archive is preserved')


class CM_OT_normal_composition(bpy.types.Operator):
    bl_idname='cm.normal_composition'
    bl_label='Choose Normal Composition'
    bl_description='Choose how final normals are baked for this part; stored reference images remain unchanged'
    bl_options={'REGISTER','UNDO'}
    mode:EnumProperty(items=[('COMBINE','Skin Baseline + Sculpt','Retain source skin normals beneath sculpt detail'),('GEOMETRY_ONLY','Sculpt Geometry Only','Ignore stored skin shader normals and bake the actual sculpt surface')])

    def execute(self,context):
        from . import bake_jobs,detail_jobs
        s,f=context.scene.cm_settings,context.scene.cm_features
        obj,detail=(f.body_result,f.body_detail) if s.part=='FULL_BODY' and f.texture_part=='BODY' else (s.result,s.detail)
        try:
            if context.mode!='OBJECT' or bake_jobs.running() or detail_jobs.running():raise ValueError('Finish the current edit or bake before changing normal composition')
            if not obj:raise ValueError('Choose a fitted head or body first')
            if self.mode=='GEOMETRY_ONLY':
                if not detail:raise ValueError('Create a sculpt layer before choosing geometry-only normals')
                if obj.get('cm_keep_skin_bake_layers') and obj.get('cm_skin_bake_layers'):raise ValueError('Disable Keep Retained Skin Layers first; its archive is preserved')
            obj[PROPERTY]=self.mode
            s.status='Normal composition changed for this part. Bake final maps to update its normal; the reference image is retained.'
            return {'FINISHED'}
        except ValueError as exc:s.status=str(exc);self.report({'ERROR'},str(exc));return {'CANCELLED'}


def draw(layout,scene):
    s,f=scene.cm_settings,scene.cm_features;obj=f.body_result if s.part=='FULL_BODY' and f.texture_part=='BODY' else s.result
    if not obj:return
    row=layout.row(align=True)
    for mode,label in [('COMBINE','Skin + Sculpt'),('GEOMETRY_ONLY','Geometry Only')]:row.operator('cm.normal_composition',text=label,depress=obj.get(PROPERTY,'COMBINE')==mode).mode=mode
    layout.operator('cm.normal_projection',text='Normal Ray Distance: '+(format(obj[DISTANCE],'.4g')+' m' if DISTANCE in obj else 'Source Transfer'))


CLASSES=(CM_OT_normal_composition,CM_OT_normal_projection)
