"""Reversible camera-matched photo overlay for an accepted fitted GNM source."""

import json

import bpy
import numpy as np
from bpy.app.handlers import persistent
from bpy.props import EnumProperty
from mathutils import Matrix

from . import head_model, photo_fit_core

_VIEW = None


def camera_for(context, source):
    report = json.loads(source.get('cm_photo_fit_report', '{}'))
    image = source.get('cm_photo_reference')
    if not image or 'camera' not in report:
        raise ValueError('Create a photo-fitted source before opening its overlay')
    f = context.scene.cm_features
    model = head_model.load_model(f.model_dir)
    skin = model['vertex_groups'][list(map(str, model['vertex_group_names'])).index('skin')] > .5
    scale = source.get('cm_gnm_model_scale', f.generated_head_height / np.ptp(model['template_vertex_positions'][skin, 1]))
    transform = np.asarray(source.matrix_world)
    singular = np.linalg.svd(transform[:3, :3], compute_uv=False)
    if not np.allclose(singular, singular[0], rtol=1e-5) or np.linalg.det(transform[:3, :3]) <= 0:
        raise ValueError('Photo overlay needs a source with positive, uniform object scale')
    values = np.asarray(report['camera'])
    conversion = np.array(((1., 0., 0.), (0., 0., -1.), (0., 1., 0.)))
    axes = conversion @ photo_fit_core.rotation(values[:3]).T
    location = axes @ np.array((-values[4] / np.exp(values[3]),
                                -values[5] / np.exp(values[3]), 2.)) * scale
    world = np.eye(4)
    world[:3, :3] = transform[:3, :3] @ axes / singular[0]
    world[:3, 3] = transform[:3, :3] @ location + transform[:3, 3]
    camera = source.get('cm_photo_review_camera')
    if not camera or camera.type != 'CAMERA':
        data = bpy.data.cameras.new('Photo Fit Review')
        camera = bpy.data.objects.new('Photo Fit Review', data)
        context.scene.collection.objects.link(camera)
        source['cm_photo_review_camera'] = camera
        camera['cm_photo_review'] = True
        camera.hide_set(True)
    camera.matrix_world = Matrix(world)
    data = camera.data
    data.type = 'ORTHO'
    data.ortho_scale = scale * singular[0] / np.exp(values[3])
    data.clip_start = .001
    data.clip_end = 1000
    data.show_background_images = True
    opacity = data.background_images[0].alpha if data.background_images else .5
    data.background_images.clear()
    background = data.background_images.new()
    background.image = image
    background.alpha = opacity
    background.display_depth = 'FRONT'
    background.frame_method = 'FIT'
    return camera, report


def targets(context, target):
    s, f = context.scene.cm_settings, context.scene.cm_features
    source = f.generated_head
    if not source or s.source != source or source.name not in context.view_layer.objects:
        raise ValueError('Select the photo-fitted generated source in this scene')
    if target == 'SOURCE':
        return [source]
    if target != 'RESULT':
        raise ValueError('Choose the generated source or accepted MetaHuman')
    from .guided import signature
    head = s.result
    if s.part == 'BODY' or not head or head == source:
        raise ValueError('Wrap the photo-fitted source before comparing the MetaHuman')
    if head.get('cm_source_signature') != signature(source):
        raise ValueError('Update Face from this photo-fitted source before comparing the MetaHuman')
    result = [head] + [obj for obj in (f.eye_left, f.eye_right, f.teeth) if obj]
    if any(obj.name not in context.view_layer.objects for obj in result):
        raise ValueError('The accepted head and its accessories must be in the current view layer')
    return result


def show_target(context, target):
    state = _VIEW
    if not state or context.scene != state['scene'] or context.view_layer != state['layer']:
        raise ValueError('Return to the photo review scene before switching its comparison')
    if context.mode != 'OBJECT':
        raise ValueError('Finish the current edit before switching the photo comparison')
    visible = targets(context, target) + [state['review_camera']]
    for obj in state['layer'].objects:
        obj.hide_set(obj not in visible, view_layer=state['layer'])
        if state['space'].local_view and obj in visible:
            obj.local_view_set(state['space'], True)
    state['target'] = target


def start(context, target='SOURCE'):
    global _VIEW
    if _VIEW:
        finish()
    s, f = context.scene.cm_settings, context.scene.cm_features
    from . import character_project
    character_project.preflight(context, s, f)
    targets(context, target)
    source = f.generated_head
    area = max((a for a in context.screen.areas if a.type == 'VIEW_3D'),
               key=lambda a: a.width * a.height, default=None)
    if not area:
        raise ValueError('Open a 3D View to compare the fitted source')
    camera, report = camera_for(context, source)
    space, render = area.spaces.active, context.scene.render
    region = space.region_3d
    _VIEW = {'scene': context.scene, 'space': space, 'source': source, 'layer': context.view_layer,
             'render': tuple(getattr(render, key) for key in ('resolution_x', 'resolution_y',
                       'resolution_percentage', 'pixel_aspect_x', 'pixel_aspect_y')),
             'camera': space.camera, 'local_camera': space.use_local_camera, 'review_camera': camera,
             'view': {key: getattr(region, key).copy() if hasattr(getattr(region, key), 'copy')
                      else getattr(region, key) for key in ('view_location', 'view_rotation', 'view_distance',
                           'view_perspective', 'view_camera_zoom', 'view_camera_offset')},
             'hidden': [(obj, obj.hide_get()) for obj in context.view_layer.objects],
             'selected': [(obj, obj.select_get()) for obj in context.view_layer.objects],
             'active': context.view_layer.objects.active,
             'local': [(obj, obj.local_view_get(space)) for obj in context.view_layer.objects]
                      if space.local_view else []}
    try:
        # The review camera must be visible, including in local view, for its
        # photo background to appear. Source/result share this exact camera.
        show_target(context, target)
        render.resolution_x, render.resolution_y = report['width'], report['height']
        render.resolution_percentage = 100
        render.pixel_aspect_x = render.pixel_aspect_y = 1
        space.use_local_camera = True
        space.camera = camera
        region.view_perspective = 'CAMERA'
        region.view_camera_zoom = 0
        region.view_camera_offset = (0, 0)
        area.tag_redraw()
    except BaseException:
        finish()
        raise


@persistent
def finish(_=None):
    global _VIEW
    state, _VIEW = _VIEW, None
    if not state:
        return
    try:
        render = state['scene'].render
        for key, value in zip(('resolution_x', 'resolution_y', 'resolution_percentage',
                               'pixel_aspect_x', 'pixel_aspect_y'), state['render']):
            setattr(render, key, value)
    except ReferenceError:
        pass
    try:
        space = state['space']
        space.camera, space.use_local_camera = state['camera'], state['local_camera']
        for key, value in state['view'].items():
            setattr(space.region_3d, key, value)
        if space.local_view:
            for obj, member in state['local']:
                try:
                    obj.local_view_set(space, member)
                except ReferenceError:
                    pass
    except ReferenceError:
        pass  # Restore object visibility even if its review screen was removed.
    for obj, hidden in state['hidden']:
        try:
            if obj.name in state['layer'].objects:
                obj.hide_set(hidden, view_layer=state['layer'])
        except ReferenceError:
            pass
    for obj, selected in state['selected']:
        try:
            if obj.name in state['layer'].objects:
                obj.select_set(selected, view_layer=state['layer'])
        except ReferenceError:
            pass
    try:
        active = state['active']
        if active is None or active.name in state['layer'].objects:
            state['layer'].objects.active = active
    except ReferenceError:
        pass


def register():
    for handler in (bpy.app.handlers.load_pre, bpy.app.handlers.undo_pre, bpy.app.handlers.save_pre):
        if finish not in handler:
            handler.append(finish)


def unregister():
    finish()
    for handler in (bpy.app.handlers.load_pre, bpy.app.handlers.undo_pre, bpy.app.handlers.save_pre):
        if finish in handler:
            handler.remove(finish)


class CM_OT_photo_review(bpy.types.Operator):
    bl_idname = 'cm.photo_overlay'
    bl_label = 'Photo Fit Overlay'
    bl_description = 'Toggle a camera-matched photo overlay; returning restores the previous view and render settings'
    target: EnumProperty(items=[('AUTO', 'Toggle Source', ''),
                                ('SOURCE', 'Generated Source', ''),
                                ('RESULT', 'Accepted MetaHuman', ''),
                                ('CLOSE', 'Return to 3D', '')], default='AUTO')

    def execute(self, context):
        try:
            if self.target == 'CLOSE' or (self.target == 'AUTO' and _VIEW):
                finish()
            elif _VIEW:
                show_target(context, self.target)
            else:
                start(context, 'SOURCE' if self.target == 'AUTO' else self.target)
            return {'FINISHED'}
        except Exception as exc:  # noqa: BLE001 - report failures at the Blender operator boundary.
            self.report({'ERROR'}, str(exc))
            return {'CANCELLED'}


CLASSES = (CM_OT_photo_review,)


def draw(layout, context):
    f = context.scene.cm_features
    source = f.generated_head
    if not source or not source.get('cm_photo_fit_report'):
        return False
    box = layout.box()
    box.label(text='Compare with Reference Photo')
    state = _VIEW if _VIEW and _VIEW['scene'] == context.scene else None
    row = box.row(align=True)
    for target, label in (('SOURCE', 'Source'), ('RESULT', 'MetaHuman')):
        button = row.row(align=True)
        button.enabled = target == 'SOURCE' or bool(context.scene.cm_settings.result)
        button.operator('cm.photo_overlay', text=label,
                        depress=bool(state and state['target'] == target)).target = target
    if state:
        camera = state['review_camera']
        if camera and camera.data.background_images:
            box.prop(camera.data.background_images[0], 'alpha', text='Photo Opacity')
        box.operator('cm.photo_overlay', text='Return to 3D', icon='LOOP_BACK').target = 'CLOSE'
    elif context.scene.cm_settings.result:
        if context.scene.cm_settings.part == 'FULL_BODY' and f.body_result:
            box.prop(f, 'photo_base_skin_settings', text='Base Skin',
                     icon='TRIA_DOWN' if f.photo_base_skin_settings else 'TRIA_RIGHT', emboss=False)
            if f.photo_base_skin_settings:
                from . import character_skin
                base = box.box()
                character_skin.draw(base, context)
                base.label(text='Apply skin, then project the photo.')
                base.label(text='Skin changes replace photo color.')
        row=box.row(align=True)
        operator=row.operator('cm.project_photo_color', text='Project Photo Color', icon='IMAGE_DATA')
        operator.resolution=f.photo_color_size
        operator.strength=f.photo_color_strength
        operator.match_skin=f.photo_match_skin
        operator.blend_neck=f.photo_blend_neck
        row.prop(f,'photo_color_settings',text='',icon='PREFERENCES',toggle=True)
        if f.photo_color_settings:
            box.prop(f,'photo_color_size',text='Size')
            box.prop(f,'photo_color_strength',text='Strength')
            box.prop(f,'photo_match_skin')
            if context.scene.cm_settings.part == 'FULL_BODY' and f.photo_match_skin:
                box.prop(f,'photo_blend_neck')
    return bool(state)
