"""Editable brightness and iris scale for the prepared local eye material."""

import bpy
import json
import numpy as np


def linked(socket):
    return socket.links[0].from_node if socket.is_linked else None


def color_source(node):
    gain = np.ones(3)
    if node and node.type == 'MIX_RGB' and node.blend_type == 'MULTIPLY':
        if (node.use_clamp or node.inputs[0].is_linked or abs(node.inputs[0].default_value - 1) > 1e-7
                or node.inputs[2].is_linked):
            return None, gain
        gain = np.asarray(node.inputs[2].default_value[:3], dtype=float)
        if not node.inputs[1].is_linked or node.inputs[1].links[0].from_socket.name != 'Color':
            return None, gain
        node = linked(node.inputs[1])
    if not node or node.type != 'TEX_IMAGE' or not node.image or not np.isfinite(gain).all() or (gain < 0).any():
        return None, gain
    return node, gain


def graph(material):
    if not material or not material.use_nodes:
        raise ValueError('Eye Appearance needs the prepared local iris/sclera material')
    nodes = material.node_tree.nodes
    output = next((n for n in nodes if n.type == 'OUTPUT_MATERIAL' and n.is_active_output), None)
    shader = linked(output.inputs['Surface']) if output else None
    mix = linked(shader.inputs['Base Color']) if shader and shader.type == 'BSDF_PRINCIPLED' else None
    if not mix or mix.type != 'MIX_RGB' or mix.blend_type != 'MIX' or mix.use_clamp:
        raise ValueError('Keep custom eye shaders in the Material editor; these controls need the local eye shader')
    sclera, _ = color_source(linked(mix.inputs[1]))
    iris, _ = color_source(linked(mix.inputs[2]))
    if any(not mix.inputs[index].is_linked or mix.inputs[index].links[0].from_socket.name != 'Color'
           for index in (1,2)):
        raise ValueError('The eye material uses a custom color channel')
    mask = linked(mix.inputs[0])
    distance = linked(mask.inputs[0]) if mask and mask.type == 'MATH' and mask.operation == 'LESS_THAN' else None
    offset = linked(iris.inputs['Vector']) if iris else None
    scale = linked(offset.inputs[0]) if offset and offset.type == 'VECT_MATH' and offset.operation == 'ADD' else None
    subtract = linked(scale.inputs[0]) if scale and scale.type == 'VECT_MATH' and scale.operation == 'SCALE' else None
    if (not sclera or not iris or not distance or distance.type != 'VECT_MATH' or distance.operation != 'DISTANCE'
            or mask.inputs[1].is_linked or not subtract or subtract.type != 'VECT_MATH'
            or subtract.operation != 'SUBTRACT' or scale.inputs['Scale'].is_linked
            or not np.isfinite(scale.inputs['Scale'].default_value)
            or not np.isfinite(mask.inputs[1].default_value) or mask.inputs[1].default_value <= 0):
        raise ValueError('The eye material has a custom iris mapping; keep it in the Material editor')
    uv = linked(distance.inputs[0])
    if (not uv or uv.type != 'TEX_COORD' or distance.inputs[0].links[0].from_socket.name != 'UV'
            or linked(subtract.inputs[0]) != uv or distance.inputs[1].is_linked
            or subtract.inputs[1].is_linked or offset.inputs[1].is_linked
            or not np.allclose(distance.inputs[1].default_value, subtract.inputs[1].default_value)
            or not np.allclose(subtract.inputs[1].default_value, offset.inputs[1].default_value)):
        raise ValueError('The eye material uses a custom iris center or UV mapping')
    return mix, mask, scale


def apply(context):
    from . import character_project, mesh_ops
    s, f = context.scene.cm_settings, context.scene.cm_features
    character_project.preflight(context, s, f)
    if s.part == 'BODY':
        raise ValueError('Choose a head or full character before adjusting eye appearance')
    if not np.isfinite((f.eye_iris_radius,f.eye_iris_exposure,f.eye_sclera_exposure)).all() or f.eye_iris_radius<=0:
        raise ValueError('Eye appearance needs finite settings and a positive iris size')
    subjects = [f.eye_left, f.eye_right]
    if any(obj is None for obj in subjects) or subjects[0] == subjects[1]:
        raise ValueError('Load both separate eye meshes before adjusting their appearance')
    originals = []
    for obj in subjects:
        mesh_ops.check_mesh(obj, 'Eye')
        if obj.name not in context.scene.objects or not obj.data.is_editable:
            raise ValueError('Use editable eye meshes belonging to the current character scene')
        if len(obj.data.materials) != 1 or obj.data.users != 1:
            raise ValueError('Give each eye its own mesh and one material before adjusting its appearance')
        graph(obj.data.materials[0])
        originals.append(obj.data.materials[0])
    copies = []
    try:
        for original in originals:
            material = original.copy()
            copies.append(material)
            mix, mask, scale = graph(material)
            old_radius = float(mask.inputs[1].default_value)
            mask.inputs[1].default_value = f.eye_iris_radius
            scale.inputs['Scale'].default_value *= old_radius / f.eye_iris_radius
            tree = material.node_tree
            for index, exposure in ((1, f.eye_sclera_exposure), (2, f.eye_iris_exposure)):
                previous = linked(mix.inputs[index])
                image, _ = color_source(previous)
                if previous.type == 'MIX_RGB' and previous.blend_type == 'MULTIPLY':
                    gain = previous
                else:
                    gain = tree.nodes.new('ShaderNodeMixRGB')
                    gain.blend_type = 'MULTIPLY'
                    gain.inputs[0].default_value = 1
                    gain.label = 'Sclera Exposure' if index == 1 else 'Iris Exposure'
                    tree.links.new(image.outputs['Color'], gain.inputs[1])
                    tree.links.new(gain.outputs[0], mix.inputs[index])
                gain.inputs[2].default_value = (2 ** exposure,) * 3 + (1,)
            material['cm_eye_appearance'] = True
        for obj, material in zip(subjects, copies):
            obj.data.materials[0] = material
    except BaseException:
        for obj, original in zip(subjects, originals):
            obj.data.materials[0] = original
        for material in copies:
            bpy.data.materials.remove(material)
        raise
    s.status = 'Eye appearance updated. Geometry and licensed source images retained.'


class CM_OT_eye_appearance(bpy.types.Operator):
    bl_idname = 'cm.eye_appearance'
    bl_label = 'Apply Eye Appearance'
    bl_description = 'Adjust both local eyes with private material copies; custom shaders are retained'
    bl_options = {'REGISTER', 'UNDO'}  # noqa: RUF012 - Blender registration metadata.

    def execute(self, context):
        try:
            apply(context)
            return {'FINISHED'}
        except Exception as exc:  # noqa: BLE001 - operator boundary.
            self.report({'ERROR'}, str(exc))
            return {'CANCELLED'}


def creator_options(features):
    from .unreal_import import eye_options
    return eye_options(dict(global_tint=list(features.eye_native_tint),
        global_saturation=features.eye_native_saturation,cornea_size=features.eye_native_size))


class CM_OT_creator_eyes(bpy.types.Operator):
    bl_idname='cm.update_creator_eyes'
    bl_label='Update Eyes in Creator'
    bl_description='Save eye tint, saturation and iris size on the linked Creator character; reassemble built characters to refresh their materials'

    def execute(self,context):
        try:
            from . import unreal_jobs
            f=context.scene.cm_features
            if not f.eye_native_enabled:raise ValueError('Enable Creator Eye Appearance first')
            unreal_jobs.character_action(context,eye_options=creator_options(f))
            return {'FINISHED'}
        except Exception as exc:
            self.report({'ERROR'},str(exc))
            return {'CANCELLED'}


CLASSES = (CM_OT_eye_appearance,CM_OT_creator_eyes)


def draw(layout, context):
    from . import eye_color_jobs
    f = context.scene.cm_features
    if (not f.eye_left or not f.eye_right) and not f.unreal_character:
        return
    box = layout.box()
    box.prop(f, 'show_eye_appearance', text='Eye Appearance', toggle=True)
    if f.show_eye_appearance:
        local=box.column();local.enabled=bool(f.eye_left and f.eye_right)
        local.label(text='Blender Preview')
        projected=all(obj and obj.get(eye_color_jobs.PROPERTY) for obj in (f.eye_left,f.eye_right))
        if projected:
            local.label(text='Projected source-eye color is active.')
            local.operator('cm.restore_eye_color',icon='LOOP_BACK')
        else:
            local.prop(f, 'eye_iris_exposure', text='Iris Exposure')
            local.prop(f, 'eye_sclera_exposure', text='Sclera Exposure')
            local.prop(f, 'eye_iris_radius', text='Iris Size')
            local.operator('cm.eye_appearance', icon='MATERIAL')
        local.prop(f,'eye_hidden_pigment')
        local.operator('cm.cancel_eye_color' if eye_color_jobs.running() else 'cm.source_eye_color',icon='IMAGE_DATA')
        local.label(text='Uses the marked source globe patches.')
        if projected:local.label(text='Local/GLB color; Creator eye shaders are separate.')
        native=box.box()
        try:source_native=json.loads(f.unreal_source_eye_result or '{}').get('source_eye_override_active',False)
        except (TypeError,ValueError):source_native=False
        if source_native:
            native.label(text='Projected source PBR eyes active in Creator.')
            native.label(text='Restore previous eyes to use native iris controls.')
        row=native.row();row.enabled=not source_native
        row.prop(f,'eye_native_enabled')
        if f.eye_native_enabled:
            controls=native.column();controls.enabled=not source_native
            controls.prop(f,'eye_native_tint')
            controls.prop(f,'eye_native_saturation')
            controls.prop(f,'eye_native_size')
            controls.label(text='Included with new Creator deliveries.')
            from . import unreal_jobs
            row=native.row();row.enabled=bool(f.unreal_character) and not unreal_jobs.running()
            row.enabled=row.enabled and not source_native
            row.operator('cm.update_creator_eyes',icon='MATERIAL')
        if f.unreal_character:
            native.operator('cm.source_eyes_creator',text='Apply Projected Eyes to Creator',icon='EXPORT')
            native.operator('cm.source_eyes_creator',text='Restore Previous Creator Eyes',icon='LOOP_BACK').restore=True
            native.prop(f,'unreal_eye_overlay_policy',text='Overlays on Assembly')
