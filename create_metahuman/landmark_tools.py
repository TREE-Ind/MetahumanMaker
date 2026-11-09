"""Surface landmark placement and reversible correction across paired viewports."""

import bpy
from bpy.app.handlers import persistent
from bpy_extras import view3d_utils

from . import mesh_ops as m

_SESSION = None
COLORS = ((1., .35, .18, 1.), (.2, .8, 1., 1.), (.7, 1., .2, 1.),
          (1., .4, .85, 1.), (1., .85, .2, 1.), (.6, .5, 1., 1.))


def active(scene=None):
    return bool(_SESSION and not _SESSION.done and (scene is None or _SESSION.scene == scene))


def finish_active(context):
    if active(context.scene):
        _SESSION.finish(context)


def remove_last(context):
    s = context.scene.cm_settings
    if s.landmarks:
        if active(context.scene):
            s.landmarks.remove(len(s.landmarks)-1)
            _SESSION.record()
            _SESSION.refresh(context)
            return
        for marker in (s.landmarks[-1].source, s.landmarks[-1].template):
            if marker:
                bpy.data.objects.remove(marker, do_unlink=True)
        s.landmarks.remove(len(s.landmarks)-1)


class Session:
    def __init__(self, context):
        self.scene = context.scene
        self.window = context.window
        self.area = context.area
        self.done = False
        self.drag = None
        self.handler = None
        self.markers = set()
        self.names = {}
        self.visibility = [(obj, obj.hide_get()) for obj in context.view_layer.objects]
        self.history = []
        self.cursor = -1
        self.comparison = None

    def surfaces(self, role):
        s, f = self.scene.cm_settings, self.scene.cm_features
        if role == 'source':
            return [s.source]
        from .features import generated_head_update
        body = f.body_template if s.part=='FULL_BODY' and not generated_head_update(s) else None
        return [obj for obj in (s.template, body) if obj]

    def begin(self, context):
        from . import (
            bake_jobs,
            compare_views,
            detail_jobs,
            native_body,
            scan_jobs,
            wrinkle_jobs,
        )
        s = self.scene.cm_settings
        m.check_inputs(s)
        compare_views.check_idle(context)
        if scan_jobs.running() or detail_jobs.running() or bake_jobs.running() or native_body._JOB or wrinkle_jobs.running():
            raise ValueError('Finish or cancel the current geometry update before editing landmarks')
        for pair in s.landmarks:
            for role in ('source', 'template'):
                marker = getattr(pair, role)
                if marker:
                    if marker.parent not in self.surfaces(role):
                        raise ValueError('Existing points belong to another source/template. Load matching landmarks or clear them first')
                    self.remember(marker)
        self.comparison = compare_views.open_views(context)
        left = self.comparison['areas'][0]
        left.spaces.active.show_region_ui = True
        sidebar = next((r for r in left.regions if r.type=='UI'), None)
        if sidebar and not sidebar.is_property_readonly('active_panel_category'):
            sidebar.active_panel_category = 'MetaHuman Maker'
        self.refresh(context)
        self.record()
        self.handler = bpy.types.SpaceView3D.draw_handler_add(self.overlay, (), 'WINDOW', 'POST_PIXEL')
        def show_panel():
            if self.done:
                return
            left = self.comparison['areas'][0]
            left.spaces.active.show_region_ui = True
            self.comparison['areas'][1].spaces.active.show_region_ui = False
            sidebar = next((r for r in left.regions if r.type=='UI'), None)
            if sidebar and not sidebar.is_property_readonly('active_panel_category'):
                sidebar.active_panel_category = 'MetaHuman Maker'
            left.tag_redraw()
            return
        bpy.app.timers.register(show_panel, first_interval=.1)

    def remember(self, marker):
        self.markers.add(marker)
        self.names.setdefault(marker, marker.show_name)
        marker.show_name = False

    def expected(self):
        pairs = self.scene.cm_settings.landmarks
        return 'template' if pairs and pairs[-1].source and not pairs[-1].template else 'source'

    def snapshot(self):
        return [(pair.name, [(marker, tuple(marker.location)) if marker else None
            for marker in (pair.source, pair.template)]) for pair in self.scene.cm_settings.landmarks]

    def record(self):
        self.history = self.history[:self.cursor+1]
        self.history.append(self.snapshot())
        if len(self.history) > 128:
            self.history.pop(0)
        self.cursor = len(self.history)-1

    def restore(self, context, offset):
        position = self.cursor+offset
        if 0 <= position < len(self.history):
            self.cursor = position
            pairs = self.scene.cm_settings.landmarks
            pairs.clear()
            for name, saved in self.history[position]:
                pair = pairs.add()
                pair.name = name
                for role, value in zip(('source', 'template'), saved):
                    if value:
                        marker, point = value
                        marker.location = point
                        setattr(pair, role, marker)
            self.refresh(context)

    def refresh(self, context):
        from . import compare_views
        pairs = self.scene.cm_settings.landmarks
        visible = {marker for pair in pairs for marker in (pair.source, pair.template) if marker}
        for marker in self.markers:
            marker.hide_set(marker not in visible)
        s = self.scene.cm_settings
        for obj in (s.source, s.template):
            obj.hide_set(False)
        compare_views.members(context, self.comparison)
        message = 'Landmarks: click '+self.expected().upper()+'; right-drag points; Ctrl Z undo; Shift Ctrl Z redo; Esc finishes'
        complete = sum(bool(pair.source and pair.template) for pair in pairs)
        s.status = str(complete)+' complete pairs. Click '+self.expected()+'; right-drag to correct, or click Wrap when ready.'
        for area in self.comparison['areas']:
            area.header_text_set(message)
            area.tag_redraw()

    def viewport(self, event):
        for side, area in enumerate(self.comparison['areas']):
            if area not in list(self.window.screen.areas):
                continue
            for region in area.regions:
                overlay = region.type=='UI' and area.spaces.active.show_region_ui or region.type=='TOOLS' and area.spaces.active.show_region_toolbar
                if overlay and region.width and region.x <= event.mouse_x < region.x+region.width and region.y <= event.mouse_y < region.y+region.height:
                    return None
            region = next((r for r in area.regions if r.type == 'WINDOW'), None)
            if region and region.x <= event.mouse_x < region.x+region.width and region.y <= event.mouse_y < region.y+region.height:
                return area, region, 'template' if side == 0 else 'source'
        return None

    def ray(self, view, event, obj):
        area, region, _role = view
        coordinate = (event.mouse_x-region.x, event.mouse_y-region.y)
        rv3d = area.spaces.active.region_3d
        origin = view3d_utils.region_2d_to_origin_3d(region, rv3d, coordinate)
        direction = view3d_utils.region_2d_to_vector_3d(region, rv3d, coordinate)
        inverse = obj.matrix_world.inverted()
        hit, point, _normal, _face = obj.ray_cast(inverse @ origin, inverse.to_3x3() @ direction,
            depsgraph=bpy.context.evaluated_depsgraph_get())
        return point if hit else None

    def pick(self, view, event):
        area, region, role = view
        coordinate = (event.mouse_x-region.x, event.mouse_y-region.y)
        best, distance = None, (18*bpy.context.preferences.system.ui_scale)**2
        for pair in self.scene.cm_settings.landmarks:
            marker = getattr(pair, role)
            if not marker:
                continue
            point = view3d_utils.location_3d_to_region_2d(region, area.spaces.active.region_3d, marker.matrix_world.translation)
            if point is not None:
                gap = (point.x-coordinate[0])**2+(point.y-coordinate[1])**2
                if gap < distance:
                    best, distance = marker, gap
        return best

    def overlay(self):
        if self.done:
            return
        import blf
        import gpu
        from gpu_extras.batch import batch_for_shader
        context = bpy.context
        if context.window != self.window or context.area not in self.comparison['areas']:
            return
        role = 'template' if context.area == self.comparison['areas'][0] else 'source'
        shader = gpu.shader.from_builtin('UNIFORM_COLOR')
        gpu.state.blend_set('ALPHA')
        gpu.state.point_size_set(9*context.preferences.system.ui_scale)
        try:
            for index, pair in enumerate(self.scene.cm_settings.landmarks):
                marker = getattr(pair, role)
                if not marker:
                    continue
                point = view3d_utils.location_3d_to_region_2d(context.region, context.region_data, marker.matrix_world.translation)
                if point is None:
                    continue
                color = COLORS[index % len(COLORS)]
                shader.bind();shader.uniform_float('color', color)
                batch_for_shader(shader, 'POINTS', {'pos': [(point.x, point.y, 0)]}).draw(shader)
                blf.size(0, 12*context.preferences.system.ui_scale)
                blf.color(0, *color);blf.position(0, point.x+8, point.y+5, 0);blf.draw(0, str(index+1))
        finally:
            gpu.state.point_size_set(1)
            gpu.state.blend_set('NONE')

    def finish(self, context, *, discard_pending=True):
        if self.done:
            return
        self.done = True
        global _SESSION
        _SESSION = None
        if self.handler is not None:
            bpy.types.SpaceView3D.draw_handler_remove(self.handler, 'WINDOW')
            self.handler = None
        if self.drag:
            self.drag[0].location = self.drag[1]
            self.drag = None
        s = self.scene.cm_settings
        if discard_pending and self.expected() == 'template':
            s.landmarks.remove(len(s.landmarks)-1)
        used = {marker for pair in s.landmarks for marker in (pair.source, pair.template) if marker}
        compared = {s.source, s.template, s.result}
        for obj, hidden in self.visibility:
            try:
                if obj.name in context.view_layer.objects and obj not in compared:
                    obj.hide_set(hidden)
            except ReferenceError:
                continue
        for marker in self.markers:
            try:
                if marker in used:
                    marker.show_name = self.names[marker]
                    marker.hide_set(False)
                elif marker.name in bpy.data.objects:
                    bpy.data.objects.remove(marker, do_unlink=True)
            except ReferenceError:
                continue
        if self.comparison:
            for area in self.comparison['areas']:
                if area in list(self.window.screen.areas):
                    area.header_text_set(None)
                    area.tag_redraw()
        s.status = str(len(s.landmarks))+' complete landmark pairs. Continue in Wrap, or reopen Landmarks to refine.'

    def modal(self, context, event):
        if self.done:
            return {'CANCELLED'}
        s = self.scene.cm_settings
        view = self.viewport(event)
        if event.type == 'ESC' and event.value == 'PRESS':
            if self.drag:
                self.drag[0].location = self.drag[1];self.drag = None
                self.refresh(context)
                return {'RUNNING_MODAL'}
            self.finish(context)
            return {'FINISHED'}
        if self.drag:
            if event.type == 'MOUSEMOVE' and view and view[2] == self.drag[2]:
                point = self.ray(view, event, self.drag[0].parent)
                if point is not None:
                    self.drag[0].location = point
                    for area in self.comparison['areas']:area.tag_redraw()
            elif event.type == 'RIGHTMOUSE' and event.value == 'RELEASE':
                self.drag = None;self.record();self.refresh(context)
            return {'RUNNING_MODAL'}
        if event.value == 'PRESS' and event.ctrl and event.type in {'Z', 'Y'}:
            self.restore(context, 1 if event.shift or event.type=='Y' else -1)
            return {'RUNNING_MODAL'}
        if event.type == 'BACK_SPACE' and event.value == 'PRESS':
            if s.landmarks:
                s.landmarks.remove(len(s.landmarks)-1)
                self.record();self.refresh(context)
            return {'RUNNING_MODAL'}
        if event.type == 'RIGHTMOUSE':
            if event.value == 'PRESS' and view:
                marker = self.pick(view, event)
                if marker:
                    self.drag = (marker, tuple(marker.location), view[2])
            return {'RUNNING_MODAL'}
        if event.type == 'LEFTMOUSE' and event.value == 'PRESS':
            if not view:
                self.finish(context)
                return {'FINISHED', 'PASS_THROUGH'}
            role = self.expected()
            if view[2] != role:
                return {'RUNNING_MODAL'}
            hits = [(obj, self.ray(view, event, obj)) for obj in self.surfaces(role)]
            hits = [(obj, point) for obj, point in hits if point is not None]
            if hits:
                area, region, _ = view
                coordinate = (event.mouse_x-region.x, event.mouse_y-region.y)
                origin = view3d_utils.region_2d_to_origin_3d(region, area.spaces.active.region_3d, coordinate)
                obj, point = min(hits, key=lambda hit: ((hit[0].matrix_world @ hit[1])-origin).length)
                from . import add_marker
                if role == 'source':
                    if len(s.landmarks) >= 512:
                        return {'RUNNING_MODAL'}
                    pair = s.landmarks.add()
                else:
                    pair = s.landmarks[-1]
                marker = add_marker(context, obj, point, '')
                setattr(pair, role, marker)
                self.remember(marker);self.record();self.refresh(context)
            return {'RUNNING_MODAL'}
        if event.type == 'LEFTMOUSE':
            return {'RUNNING_MODAL'}
        return {'PASS_THROUGH'}


def invoke(operator, context, event):
    global _SESSION
    from . import auto_wrap_jobs
    if auto_wrap_jobs.running():
        operator.report({'ERROR'}, 'Finish or cancel Auto Wrap before editing landmarks')
        return {'CANCELLED'}
    if active():
        operator.report({'ERROR'}, 'Finish the current landmark session first')
        return {'CANCELLED'}
    if not context.area or context.area.type != 'VIEW_3D':
        return {'CANCELLED'}
    session = Session(context)
    _SESSION = session
    try:
        session.begin(context)
    except Exception as exc:  # noqa: BLE001
        session.finish(context, discard_pending=False)
        operator.report({'ERROR'}, str(exc))
        return {'CANCELLED'}
    operator.session = session
    context.window_manager.modal_handler_add(operator)
    return {'RUNNING_MODAL'}


@persistent
def detach(_=None):
    if active():
        _SESSION.finish(bpy.context)


def register():
    bpy.app.handlers.load_pre.append(detach)


def unregister():
    detach()
    if detach in bpy.app.handlers.load_pre:
        bpy.app.handlers.load_pre.remove(detach)
