"""Artist controls for recorded skin operations retained across final bakes."""
from pathlib import Path
import bpy
from bpy.props import StringProperty
from . import skin_bake_layers,paint_session,character_project

class CM_OT_retain_skin_bake_layers(bpy.types.Operator):
    """Retain recorded skin repairs and generated detail for future geometry bakes"""
    bl_idname='cm.retain_skin_bake_layers'
    bl_label='Retain Skin Layers for Bake'
    bl_options={'REGISTER','UNDO'}
    source_project:StringProperty(name='Source Character Project',subtype='DIR_PATH')

    def execute(self,context):
        try:
            s,f=context.scene.cm_settings,context.scene.cm_features
            character_project.preflight(context,s,f)
            role,obj=paint_session.part(context)
            source=self.source_project or obj.get('cm_skin_source_project') or context.scene.get('cm_cleanup_baseline','')
            roots=[]
            if source:
                path=Path(bpy.path.abspath(source)).resolve()
                roots=[path.parent if path.suffix.lower()=='.blend' else path]
            path,data=skin_bake_layers.retain(context,roots)
            count=sum(bool(step.get('cavity_response')) for step in data['steps'])
            s.status=f"Retained {len(data['steps'])} skin operations, including {count} microcavity response(s), for later geometry bakes."
            return {'FINISHED'}
        except Exception as exc:
            self.report({'ERROR'},str(exc));return {'CANCELLED'}

def draw(layout,context):
    role,obj=paint_session.part(context)
    if not obj:return
    box=layout.box();box.label(text='Skin Detail in Final Bakes')
    box.operator('cm.retain_skin_bake_layers',text='Retain Skin Layers for Bake')
    if obj.get(skin_bake_layers.POINTER):
        box.prop(obj,'["'+skin_bake_layers.ENABLED+'"]',text='Keep Skin Repairs in Bake')
        if 'cm_skin_cavity_operations' not in obj:box.label(text='Re-retain older layers to include microcavity.')
        else:box.label(text=str(obj['cm_skin_cavity_operations'])+' generated microcavity response(s) recorded')
        if obj.get('cm_skin_surface_patches'):box.label(text=str(obj['cm_skin_surface_patches'])+' regional surface edit(s) retained')
    box.label(text='Replays recorded normal repairs and generated microcavity.')

CLASSES=(CM_OT_retain_skin_bake_layers,)
