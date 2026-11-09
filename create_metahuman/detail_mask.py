"""Artist-painted scan inclusion mask on an isolated, disposable surface."""

import json

import bpy
import numpy as np

from . import mesh_ops as m
from .workflow import group_weights, write_group

GROUP = 'CM_DetailMask'


def target(s, f):
    body = s.part == 'FULL_BODY' and f.texture_part == 'BODY'
    detail = f.body_detail if body else s.detail
    return detail or (f.body_result if body else s.result)


def running(scene):
    return bool(scene.cm_features.detail_mask_helper)


def paint_weight(context, weight):
    settings = context.tool_settings
    brush = settings.weight_paint.brush
    if not brush:
        raise ValueError('Choose Blender’s Draw brush in the weight-paint shelf')
    brush.blend = 'MIX'
    brush.use_frontface = True
    from .blender_compat import unified_paint
    unified = unified_paint(context,'weight_paint')
    (unified if unified.use_unified_weight else brush).weight = weight


def enter(context):
    helper = context.scene.cm_features.detail_mask_helper
    if not helper:
        raise ValueError('Start Edit Scan Mask first')
    if context.object != helper or context.mode != 'PAINT_WEIGHT':
        m.activate(context, helper)
        bpy.ops.object.mode_set(mode='WEIGHT_PAINT')
    helper.vertex_groups.active_index = helper.vertex_groups[GROUP].index


def start(context):
    from . import scan_jobs
    if scan_jobs.running():
        raise ValueError('Finish or cancel scan projection before editing its mask')
    from . import (
        bake_jobs,
        detail_jobs,
        groom_jobs,
        native_body,
        paint_tools,
        skin_library,
        skin_native_jobs,
        unreal_jobs,
        wrinkle_jobs,
    )

    s, f = context.scene.cm_settings, context.scene.cm_features
    if running(context.scene):
        enter(context)
        return
    if (context.mode != 'OBJECT' or f.body_editing or f.full_sculpt or f.restore_editing
            or f.expression_active or bake_jobs.running() or detail_jobs.running()
            or wrinkle_jobs.running() or native_body._JOB or paint_tools.running()
            or unreal_jobs.running() or groom_jobs.running() or skin_library.running()
            or skin_native_jobs.running()):
        raise ValueError('Finish the current edit or background update before painting a scan mask')
    obj = target(s, f)
    m.check_mesh(obj, 'Result for scan masking')
    if obj.library or obj.data.library:
        raise ValueError('Make the result local before editing its scan mask')
    detail = f.body_detail if s.part == 'FULL_BODY' and f.texture_part == 'BODY' else s.detail
    if detail:
        from . import scan_projection
        scan_projection.check(detail)
    selected = [o.name for o in context.view_layer.objects if o.select_get()]
    active = context.view_layer.objects.active
    evaluated = obj.evaluated_get(context.evaluated_depsgraph_get())
    mesh = bpy.data.meshes.new_from_object(evaluated, preserve_all_data_layers=True,
        depsgraph=context.evaluated_depsgraph_get())
    if len(mesh.vertices) != len(obj.data.vertices):
        bpy.data.meshes.remove(mesh)
        raise ValueError('Finish topology-changing modifiers before painting a scan mask')
    helper = obj.copy()
    helper.data = mesh
    helper.name = 'Scan Mask • ' + obj.name
    helper.modifiers.clear()
    helper.constraints.clear()
    helper.animation_data_clear()
    context.scene.collection.objects.link(helper)
    helper.parent = None
    helper.matrix_world = evaluated.matrix_world.copy()
    helper.hide_render = True
    helper.hide_viewport = False
    helper.hide_select = False
    helper['cm_role'] = 'SCAN_MASK_HELPER'
    helper['cm_mask_target'] = obj
    helper['cm_mask_topology'] = m.topology(obj)
    if m.topology(helper) != helper['cm_mask_topology']:
        bpy.data.objects.remove(helper, do_unlink=True)
        raise ValueError('Finish topology-changing modifiers before painting a scan mask')
    helper['cm_mask_hidden'] = obj.hide_get()
    accepted = f.body_result if s.part == 'FULL_BODY' and f.texture_part == 'BODY' else s.result
    if accepted and accepted != obj:
        helper['cm_mask_companion'] = accepted
        helper['cm_mask_companion_hidden'] = accepted.hide_get()
    helper['cm_mask_selected'] = json.dumps(selected)
    if active:
        helper['cm_mask_active'] = active
    helper['cm_mask_role'] = 'BODY' if s.part == 'BODY' or s.part == 'FULL_BODY' and f.texture_part == 'BODY' else 'HEAD'
    # Begin with inclusion everywhere; neck and pins remain separate protections.
    write_group(helper, GROUP, group_weights(obj, GROUP, default=1))
    helper.vertex_groups[GROUP].lock_weight = False
    helper.data.use_paint_mask = False
    helper.data.use_paint_mask_vertex = False
    f.detail_mask_helper = helper
    try:
        enter(context)
        obj.hide_set(True)
        if helper.get('cm_mask_companion'):
            helper['cm_mask_companion'].hide_set(True)
        paint_weight(context, 0)
    except Exception:
        finish(context, apply=False)
        raise
    s.status = 'Paint scan mask: blue protects, red includes detail. Apply commits only mask weights; Cancel leaves the character unchanged.'


def finish(context, apply):
    s, f = context.scene.cm_settings, context.scene.cm_features
    helper = f.detail_mask_helper
    if not helper:
        raise ValueError('There is no scan-mask session to finish')
    obj = helper.get('cm_mask_target')
    if apply:
        m.check_mesh(obj, 'Original scan-mask target')
        if obj.library or obj.data.library:
            raise ValueError('Make the target local before applying its scan mask')
        if m.topology(obj) != helper.get('cm_mask_topology') or len(helper.data.vertices) != len(obj.data.vertices):
            raise ValueError('Target topology changed. Cancel this mask and paint a new one')
    if context.object and context.object.mode != 'OBJECT':
        bpy.ops.object.mode_set(mode='OBJECT')
    if apply:
        weights = group_weights(helper, GROUP)
        if not np.all(np.isfinite(weights)):
            raise ValueError('Mask contains invalid weights; cancel or repaint it')
        write_group(obj, GROUP, np.clip(weights, 0, 1))
        if obj.get('cm_scan_pending_signature') and not obj.get('cm_scan_applied', True):
            obj['cm_scan_mask_dirty'] = True
        f.use_detail_mask = True
    if obj and obj.name in context.view_layer.objects:
        obj.hide_set(bool(helper.get('cm_mask_hidden', False)))
    companion = helper.get('cm_mask_companion')
    if companion and companion.name in context.view_layer.objects:
        companion.hide_set(bool(helper.get('cm_mask_companion_hidden', False)))
    selected = json.loads(helper.get('cm_mask_selected', '[]'))
    active = helper.get('cm_mask_active')
    f.detail_mask_helper = None
    # Ordinary undoable operators own the helper lifecycle; no direct RNA edit
    # of the accepted geometry, materials, sculpt masks or shape keys is needed.
    bpy.data.objects.remove(helper, do_unlink=True)
    for item in context.view_layer.objects:
        if item.select_get():
            item.select_set(False)
    for name in selected:
        item = context.view_layer.objects.get(name)
        if item and not item.hide_get():
            item.select_set(True)
    if active and active.name in context.view_layer.objects:
        context.view_layer.objects.active = active
    s.status = ('Scan mask applied. Generate source displacement next; neck fade and pins remain enabled.'
        if apply else 'Scan mask canceled. The character and its previous mask were preserved.')


class CM_OT_detail_mask(bpy.types.Operator):
    bl_idname = 'cm.detail_mask'
    bl_label = 'Edit Scan Mask'
    bl_description = 'Paint where source displacement may affect the next sculpt layer, without editing character geometry'
    bl_options = {'REGISTER', 'UNDO'}  # noqa: RUF012
    action: bpy.props.EnumProperty(items=[(key, label, '') for key, label in (
        ('START', 'Edit',), ('RESUME', 'Resume'), ('APPLY', 'Apply'), ('CANCEL', 'Cancel'),
        ('INCLUDE', 'Include'), ('PROTECT', 'Protect'), ('FILL_INCLUDE', 'Include All'),
        ('FILL_PROTECT', 'Protect All'), ('INVERT', 'Invert'))])

    def execute(self, context):
        try:
            if self.action == 'START':
                start(context)
            elif self.action == 'RESUME':
                enter(context)
            elif self.action in {'APPLY', 'CANCEL'}:
                finish(context, apply=self.action == 'APPLY')
            else:
                enter(context)
                if self.action in {'INCLUDE', 'PROTECT'}:
                    paint_weight(context, 1 if self.action == 'INCLUDE' else 0)
                else:
                    helper = context.scene.cm_features.detail_mask_helper
                    values = group_weights(helper, GROUP)
                    values = 1-values if self.action == 'INVERT' else np.full(len(values), 1 if self.action == 'FILL_INCLUDE' else 0)
                    write_group(helper, GROUP, values)
                    helper.data.update()
            return {'FINISHED'}
        except (RuntimeError, ValueError, ReferenceError) as exc:
            context.scene.cm_settings.status = str(exc)
            self.report({'ERROR'}, str(exc))
            return {'CANCELLED'}


def draw(layout, context):
    f = context.scene.cm_features
    helper = f.detail_mask_helper
    if not helper:
        return False
    box = layout.box()
    box.label(text='Paint Scan Mask • ' + helper.get('cm_mask_role', 'HEAD'), icon='MOD_MASK')
    box.label(text='Red: include scan · Blue: protect')
    box.label(text='Only this temporary mask is editable.')
    if context.mode != 'PAINT_WEIGHT' or context.object != helper:
        box.operator('cm.detail_mask', text='Resume Mask Painting').action = 'RESUME'
    else:
        row = box.row(align=True)
        row.operator('cm.detail_mask', text='Paint Include').action = 'INCLUDE'
        row.operator('cm.detail_mask', text='Paint Protect').action = 'PROTECT'
        brush = context.tool_settings.weight_paint.brush
        if brush:
            from .blender_compat import unified_paint
            unified = unified_paint(context,'weight_paint')
            box.prop(unified if unified.use_unified_size else brush, 'size', text='Size')
            box.prop(unified if unified.use_unified_strength else brush, 'strength', text='Strength')
        box.prop(helper.data, 'use_mirror_x', text='Mirror X')
        row = box.row(align=True)
        row.operator('cm.detail_mask', text='Include All').action = 'FILL_INCLUDE'
        row.operator('cm.detail_mask', text='Protect All').action = 'FILL_PROTECT'
        box.operator('cm.detail_mask', text='Invert Mask').action = 'INVERT'
    row = box.row(align=True)
    row.operator('cm.detail_mask', text='Apply Mask', icon='CHECKMARK').action = 'APPLY'
    row.operator('cm.detail_mask', text='Cancel', icon='CANCEL').action = 'CANCEL'
    box.operator('cm.save_project', text='Save Mask Session', icon='FILE_TICK')
    return True


def register():
    bpy.utils.register_class(CM_OT_detail_mask)


def unregister():
    bpy.utils.unregister_class(CM_OT_detail_mask)
