"""Transactional neutral-head repair with inverse skinning while expressions are visible."""

import bpy
import numpy as np
from mathutils import Vector
from bpy_extras import view3d_utils
from . import geometry, mesh_ops as m, expressions
from bpy.app.handlers import persistent
from bpy.props import BoolProperty, FloatProperty

_ACTIVE = []


def physical_groups(template, f, reference):
    """Weld for brush correspondence only; never change the actual mesh/UVs."""
    if template:
        try:
            data = expressions.bank(f)
            head = next(
                mesh for mesh in data["meshes"] if mesh["name"] == "head_lod0_mesh"
            )
            indices = expressions.uv_mapping(template, head)
            if len(indices) == len(reference):
                return np.unique(indices, return_inverse=True)[1]
        except (ValueError, OSError, StopIteration):
            if f.expression_active:
                raise
    tolerance = max(np.linalg.norm(np.ptp(reference, axis=0)) * 1e-8, 1e-10)
    return np.unique(
        np.rint(reference / tolerance).astype(np.int64), axis=0, return_inverse=True
    )[1]


class RestoreSession:
    def __init__(self, context, mode=None):
        self.context = context
        self.scene = context.scene
        self.s, self.f = self.scene.cm_settings, self.scene.cm_features
        self.obj = self.s.result
        m.check_mesh(self.obj, "Head result")
        if (
            context.mode != "OBJECT"
            or self.f.full_sculpt
            or self.f.body_editing
            or self.f.restore_editing
        ):
            raise ValueError("Finish the current edit before repairing the head")
        self.original_selection = list(context.selected_objects)
        self.original_active = context.view_layer.objects.active
        self.native = bool(self.f.expression_active)
        self.mode = mode or getattr(self.f, 'restore_mode', 'RESTORE')
        if self.mode == 'SURFACE' and self.native:
            raise ValueError('Return to Neutral before smoothing on the source mesh')
        self.source_tree = None
        if self.mode == 'SURFACE':
            m.check_mesh(self.s.source, 'Reference source')
            self.source_tree = m.source_bvh(context, self.s.source)
        self.initial_collapsed = set(m.collapsed_faces(context, self.obj)) if self.mode == 'SURFACE' else set()
        self.original_neutral = expressions.neutral_coordinates(context, self.obj)
        self.original_posed = m.evaluated_coordinates(context, self.obj)
        if len(self.original_neutral) != len(self.obj.data.vertices):
            raise ValueError(
                "Repair the base head; finish topology-changing modifiers first"
            )
        template = self.s.template
        if template and m.topology(template) == m.topology(self.obj):
            reference = m.coordinates(template)
        else:
            template = None
            reference = m.coordinates(self.obj)
        self.groups = physical_groups(template, self.f, reference)
        counts = np.bincount(self.groups)

        def average(points):
            values = np.zeros((len(counts), 3))
            np.add.at(values, self.groups, points)
            return values / counts[:, None]

        clean = average(self.original_neutral)
        physical_reference = average(reference)
        # Compare displacements in the accepted head's world frame. Template
        # coordinates are local; their pose and uniform scale must not become
        # sculpt detail when an artist moves the character.
        rotation, scale, translation = geometry.similarity(physical_reference, clean)
        physical_reference = physical_reference @ rotation * scale + translation
        edges = np.unique(np.sort(self.groups[m.edges(self.obj)], axis=1), axis=0)
        edges = edges[edges[:, 0] != edges[:, 1]]
        self.physical_edges = edges
        self.group_counts = counts
        self.protected = np.zeros(len(reference), dtype=bool)
        if self.mode == 'SURFACE':
            from . import wrap_interiors
            self.protected = wrap_interiors.mouth_mask(self.obj)
        for _ in range(6):
            clean = geometry.relax_displacement(clean, physical_reference, edges, 0.5)
        clean = clean[self.groups]
        self.full_linear = (
            expressions.head_pose_linear(context, self.s, self.f)
            if self.native
            else np.broadcast_to(np.eye(3), (len(clean), 3, 3)).copy()
        )
        strength = self.f.expression_strength if self.native else 0
        self.linear = np.eye(3) + (self.full_linear - np.eye(3)) * strength
        singular = np.linalg.svd(self.linear, compute_uv=False)
        if np.any(singular[:, -1] < 1e-6) or not np.isfinite(singular).all():
            raise ValueError(
                "This pose compresses points too far for repair; return to Neutral"
            )
        self.inverse_linear = np.linalg.inv(self.linear)
        self.target_posed = np.einsum(
            "ni,nij->nj", clean - self.original_neutral, self.linear
        )
        self.delta_posed = np.zeros_like(clean)
        self.undo_stack = []
        self.redo_stack = []
        self.expression = (
            self.obj.data.shape_keys.key_blocks.get("Expression Check")
            if self.obj.data.shape_keys
            else None
        )
        self.expression_co = (
            np.array([v.co[:] for v in self.expression.data])
            if self.expression
            else None
        )
        self.expression_value = self.expression.value if self.expression else 0
        self.created_basis = not self.obj.data.shape_keys
        if self.created_basis:
            self.obj.shape_key_add(name="Basis")
        self.key = self.obj.shape_key_add(name="Restore Stroke")
        self.key.relative_key = self.obj.data.shape_keys.key_blocks[0]
        self.basis = np.array([v.co[:] for v in self.key.relative_key.data])
        self.key.data.foreach_set("co", self.basis.ravel())
        self.key.value = 1
        self.closed = False
        self.f.restore_editing = True

    def begin_stroke(self):
        self.undo_stack.append(self.delta_posed.copy())
        if len(self.undo_stack) > 20:
            self.undo_stack.pop(0)
        self.redo_stack.clear()

    def write(self):
        neutral = np.einsum("ni,nij->nj", self.delta_posed, self.inverse_linear)
        inverse = np.asarray(self.obj.matrix_world.inverted())[:3, :3].T
        self.key.data.foreach_set("co", (self.basis + neutral @ inverse).ravel())
        if self.native and self.expression:
            # The repair key stores neutral movement. The expression key supplies
            # the remaining posed movement, so it is not applied twice.
            posed_extra = np.einsum("ni,nij->nj", neutral, self.full_linear - np.eye(3))
            self.expression.data.foreach_set(
                "co", (self.expression_co + posed_extra @ inverse).ravel()
            )
        self.obj.data.update()
        self.context.view_layer.update()

    def stamp(self, center, radius, strength, mirror=False):
        if self.mode == 'SURFACE':
            centers = [np.asarray(center, dtype=float)]
            if mirror:
                point = np.array(self.obj.matrix_world.inverted() @ Vector(center))
                local = m.to_local(self.obj, self.original_neutral)
                axis = 0 if self.f.front_axis in {'Y', '-Y'} else 1
                point[axis] = 2 * local[:, axis].mean() - point[axis]
                centers.append(np.array(self.obj.matrix_world @ Vector(point)))
            self.polish_region(centers, radius, strength, iterations=2)
            return
        current = self.original_posed + self.delta_posed
        center = np.asarray(center, dtype=float)
        distance = np.linalg.norm(current - center, axis=1)
        if mirror:
            inverse = self.obj.matrix_world.inverted()
            point = np.array(inverse @ Vector(center))
            local = m.to_local(self.obj, self.original_neutral)
            axis = 0 if self.f.front_axis in {"Y", "-Y"} else 1
            point[axis] = 2 * local[:, axis].mean() - point[axis]
            mirrored = np.array(self.obj.matrix_world @ Vector(point))
            distance = np.minimum(distance, np.linalg.norm(current - mirrored, axis=1))
        weights = np.clip(1 - distance / max(float(radius), 1e-8), 0, 1) ** 2 * np.clip(
            strength, 0, 1
        )
        physical_weight = np.zeros(self.groups.max() + 1)
        np.maximum.at(physical_weight, self.groups, weights)
        weights = physical_weight[self.groups]
        self.delta_posed += (self.target_posed - self.delta_posed) * weights[:, None]
        self.write()

    def polish_region(self, centers, radius, strength=.4, *, iterations=30, source_adherence=None):
        """The Smooth on Source brush kernel, also callable for agent strokes.

        Preserve UV seam copies and oral interiors. Reprojection and smoothing
        together are capped at 0.5 mm per iteration; untouched points stay fixed.
        """
        if self.mode != 'SURFACE' or self.source_tree is None:
            raise ValueError('Start a Smooth on Source repair session first')
        adherence=float(self.f.restore_source_adherence if source_adherence is None else source_adherence)
        if not np.isfinite(adherence) or not 0 <= adherence <= 1:
            raise ValueError('Source adherence must be between zero and one')
        counts = self.group_counts
        clean = np.zeros((len(counts), 3))
        np.add.at(clean, self.groups, self.original_posed + self.delta_posed)
        clean /= counts[:, None]
        original_groups = clean.copy()
        weight = np.zeros(len(clean))
        for center in centers:
            weight = np.maximum(weight, np.clip(1 - np.linalg.norm(clean-center, axis=1)
                                / max(float(radius), 1e-8), 0, 1)**2)
        protected = np.zeros(len(clean), dtype=bool)
        np.logical_or.at(protected, self.groups, self.protected)
        weight[protected] = 0
        indices = np.flatnonzero(weight > .01)
        a, b = self.physical_edges.T
        degree = np.bincount(np.concatenate((a, b)), minlength=len(clean))
        limit = .0005 / max(self.scene.unit_settings.scale_length, 1e-12)
        strength = float(np.clip(strength, 0, 1))
        if strength == 0 or not len(indices):
            return
        for _ in range(max(1, min(int(iterations), 100))):
            mean = np.column_stack([
                (np.bincount(a, weights=clean[b, j], minlength=len(clean))
                 + np.bincount(b, weights=clean[a, j], minlength=len(clean)))
                / np.maximum(degree, 1) for j in range(3)])
            trial = clean.copy()
            for index in indices:
                point = clean[index] + (mean[index]-clean[index])*.4
                if adherence:
                    hit, _normal, _face, _distance = self.source_tree.find_nearest(point, limit*8)
                    if hit is not None:
                        point = np.asarray(hit)*adherence + point*(1-adherence)
                step = (point-clean[index])*strength*weight[index]
                step *= min(1., limit/max(float(np.linalg.norm(step)), 1e-12))
                trial[index] += step
            clean = trial
        movement = (clean-original_groups)[self.groups]
        self.delta_posed += movement
        self.write()

    def undo_stroke(self, redo=False):
        source, target = (
            (self.redo_stack, self.undo_stack)
            if redo
            else (self.undo_stack, self.redo_stack)
        )
        if source:
            target.append(self.delta_posed.copy())
            self.delta_posed = source.pop()
            self.write()

    def finish(self, accept):
        if self.closed:
            return
        self.f.restore_editing = False
        if accept:
            if self.mode == 'SURFACE' and set(m.collapsed_faces(self.context, self.obj))-self.initial_collapsed:
                self.finish(False)
                raise ValueError('Surface polish would collapse faces; reduce brush strength or radius')
            try:
                if self.native:
                    expressions.preview(self.context, self.s, self.f)
            except Exception:
                self.finish(False)
                raise
            self.s.status = "Head repair applied to neutral geometry."
        else:
            self.obj.shape_key_remove(self.key)
            if self.created_basis:
                self.obj.shape_key_clear()
            if self.expression:
                self.expression.data.foreach_set("co", self.expression_co.ravel())
                self.expression.value = self.expression_value
            self.obj.data.update()
            for obj in self.context.view_layer.objects:
                obj.select_set(False)
            for obj in self.original_selection:
                if obj.name in self.context.view_layer.objects:
                    obj.select_set(True)
            self.context.view_layer.objects.active = self.original_active
            self.s.status = (
                "Head repair canceled; the previous head and expression are restored."
            )
        self.context.view_layer.update()
        self.closed = True


def repair_oral_clearance(context, clearance_mm=1., limit_mm=5.):
    """Push recognized oral interior crossings behind the exterior skin.

    The skin, mouth boundary, UVs and topology are exact constraints. This is
    a local clearance repair, not a replacement for an anatomical cavity fit.
    """
    from mathutils.bvhtree import BVHTree
    from . import wrap_interiors, sculpt_session, bake_jobs, detail_jobs, wrinkle_jobs
    s, f = context.scene.cm_settings, context.scene.cm_features
    obj = s.result
    m.check_mesh(obj, 'Head result')
    if (context.mode != 'OBJECT' or f.expression_active or f.restore_editing
        or f.full_sculpt or f.body_editing or sculpt_session.pending()
        or any(job.running() for job in (bake_jobs, detail_jobs, wrinkle_jobs))):
        raise ValueError('Finish the active edit or bake and return to Neutral first')
    if any(mod.show_viewport and mod.type not in {'WEIGHTED_NORMAL', 'NORMAL_EDIT', 'DATA_TRANSFER'}
           for mod in obj.modifiers):
        raise ValueError('Repair the base head before geometry modifiers')
    interior = wrap_interiors.mouth_mask(obj)
    if not interior.any():
        raise ValueError('This head has no recognized oral interior skinning/UV islands')
    current = m.evaluated_coordinates(context, obj)
    faces = [p.vertices[:] for p in obj.data.polygons if not interior[list(p.vertices)].any()]
    if not faces:
        raise ValueError('The head has no separate exterior skin surface')
    tree = BVHTree.FromPolygons(current, faces)
    repaired = current.copy()
    units = .001 / max(context.scene.unit_settings.scale_length, 1e-12)
    clearance, limit = float(clearance_mm)*units, float(limit_mm)*units
    front = np.zeros(3)
    front[1 if f.front_axis in {'Y', '-Y'} else 0] = -1 if f.front_axis.startswith('-') else 1
    for index in np.flatnonzero(interior):
        hit, normal, _face, _distance = tree.find_nearest(current[index], limit)
        if hit is None:
            continue
        normal = np.asarray(normal)
        signed = float(np.dot(current[index]-hit, normal))
        if signed > -clearance and np.dot(normal, front) > .2:
            repaired[index] -= normal*min(limit, signed+clearance)
    delta = repaired-current
    changed = int((np.linalg.norm(delta, axis=1) > units*.001).sum())
    if not changed:
        return {'changed_vertices': 0, 'max_change_mm': 0., 'exterior_preserved': True}
    before_bad = set(m.collapsed_faces(context, obj))
    created_basis = not obj.data.shape_keys
    if created_basis:
        obj.shape_key_add(name='Basis')
    key = obj.shape_key_add(name='Oral Clearance Repair')
    try:
        key.relative_key = obj.data.shape_keys.key_blocks[0]
        basis = np.array([v.co[:] for v in key.relative_key.data])
        inverse = np.asarray(obj.matrix_world.inverted())[:3, :3].T
        key.data.foreach_set('co', (basis+delta@inverse).ravel())
        key.value = 1
        obj.data.update();context.view_layer.update()
        if set(m.collapsed_faces(context, obj))-before_bad:
            raise ValueError('Oral repair would collapse faces; use a smaller clearance')
        report = {'changed_vertices': changed, 'max_change_mm': float(np.linalg.norm(delta, axis=1).max()/units),
                  'exterior_preserved': True, 'shape_key': key.name}
        s.status = f'Oral clearance repaired on {changed} vertices. Review, update detail and rebake before delivery.'
        return report
    except Exception:
        obj.shape_key_remove(key)
        if created_basis:obj.shape_key_clear()
        context.view_layer.update()
        raise


class CM_OT_repair_oral_clearance(bpy.types.Operator):
    """Repair oral interior surfaces showing through the lips or philtrum"""
    bl_idname = 'cm.repair_oral_clearance'
    bl_label = 'Repair Mouth Interior Clearance'
    bl_options = {'REGISTER', 'UNDO'}
    clearance_mm: FloatProperty(name='Skin Clearance (mm)', default=1., min=.1, max=3.)

    def execute(self, context):
        try:
            result = repair_oral_clearance(context, self.clearance_mm)
            self.report({'INFO'}, f"Repaired {result['changed_vertices']} oral vertices; exterior retained")
            return {'FINISHED'}
        except Exception as exc:
            self.report({'ERROR'}, str(exc))
            return {'CANCELLED'}


class CM_OT_restore_brush(bpy.types.Operator):
    """Repair the neutral head while inspecting expressions; Enter applies and Esc restores"""

    bl_idname = "cm.restore_brush"
    bl_label = "Repair Head"
    bl_options = {"REGISTER", "UNDO", "BLOCKING"}

    def invoke(self, context, event):
        self.handler = None
        self.timer = None
        self.requested_finish = None
        self.session = None
        try:
            if not context.area or context.area.type != "VIEW_3D":
                raise ValueError("Open a 3D viewport to repair the head")
            self.area = context.area
            self.region = next(r for r in self.area.regions if r.type == "WINDOW")
            self.session = RestoreSession(context)
            m.activate(context, self.session.obj)
            self.down = False
            self.cursor = None
            self.cursor_radius = 40
            self.handler = bpy.types.SpaceView3D.draw_handler_add(
                self.draw_cursor, (), "WINDOW", "POST_PIXEL"
            )
            self.manager = context.window_manager
            self.timer = self.manager.event_timer_add(0.1, window=context.window)
            _ACTIVE.append(self)
            self.area.header_text_set(
                "Repair: LMB paint | Enter apply | Esc cancel | Ctrl+Z undo | Ctrl+Wheel size | Shift+Wheel strength | X mirror"
            )
            context.window_manager.modal_handler_add(self)
            return {"RUNNING_MODAL"}
        except Exception as exc:
            if self.session:
                self.session.finish(False)
            self.cleanup()
            self.report({"ERROR"}, str(exc))
            return {"CANCELLED"}

    def cleanup(self):
        if self in _ACTIVE:
            _ACTIVE.remove(self)
        if self.timer:
            self.manager.event_timer_remove(self.timer)
            self.timer = None
        if self.handler:
            bpy.types.SpaceView3D.draw_handler_remove(self.handler, "WINDOW")
            self.handler = None
        if getattr(self, "area", None):
            self.area.header_text_set(None)
            self.area.tag_redraw()

    def draw_cursor(self):
        if bpy.context.area != self.area or self.cursor is None:
            return
        import gpu
        from gpu_extras.batch import batch_for_shader

        angles = np.linspace(0, 2 * np.pi, 65)
        points = (
            np.column_stack([np.cos(angles), np.sin(angles)]) * self.cursor_radius
            + self.cursor
        )
        shader = gpu.shader.from_builtin("UNIFORM_COLOR")
        gpu.state.blend_set("ALPHA")
        shader.bind()
        shader.uniform_float("color", (0.25, 0.75, 1, 0.9))
        batch_for_shader(shader, "LINE_STRIP", {"pos": points.astype(np.float32)}).draw(
            shader
        )
        gpu.state.blend_set("NONE")

    def modal(self, context, event):
        try:
            if self.session.closed:
                self.cleanup()
                return {"CANCELLED"}
            if self.requested_finish is not None:
                accept = self.requested_finish
                self.session.finish(accept)
                self.cleanup()
                return {"FINISHED"} if accept else {"CANCELLED"}
            f = self.session.f
            if event.type == "WINDOW_DEACTIVATE":
                self.down = False
                return {"PASS_THROUGH"}
            sidebar = next((r for r in self.area.regions if r.type == "UI"), None)
            over_sidebar = (
                sidebar
                and self.area.spaces.active.show_region_ui
                and sidebar.x <= event.mouse_x < sidebar.x + sidebar.width
                and sidebar.y <= event.mouse_y < sidebar.y + sidebar.height
            )
            if (
                over_sidebar
                and event.type != "TIMER"
                and not (event.type == "Z" and event.ctrl)
            ):
                self.down = False
                self.cursor = None
                self.area.tag_redraw()
                return {"PASS_THROUGH"}
            if event.type in {"ESC", "RET", "NUMPAD_ENTER"} and event.value == "PRESS":
                self.session.finish(event.type != "ESC")
                self.cleanup()
                return {"CANCELLED"} if event.type == "ESC" else {"FINISHED"}
            if event.type == "Z" and event.ctrl and event.value == "PRESS":
                self.session.undo_stroke(redo=event.shift)
                self.area.tag_redraw()
                return {"RUNNING_MODAL"}
            if event.ctrl and event.type in {"WHEELUPMOUSE", "WHEELDOWNMOUSE"}:
                f.restore_radius = np.clip(
                    f.restore_radius
                    * (1.12 if event.type == "WHEELUPMOUSE" else 1 / 1.12),
                    0.001,
                    1,
                )
                return {"RUNNING_MODAL"}
            if event.shift and event.type in {"WHEELUPMOUSE", "WHEELDOWNMOUSE"}:
                f.restore_strength = np.clip(
                    f.restore_strength
                    + (0.05 if event.type == "WHEELUPMOUSE" else -0.05),
                    0.001,
                    1,
                )
                return {"RUNNING_MODAL"}
            if event.type == "X" and event.value == "PRESS":
                f.mirror = not f.mirror
                return {"RUNNING_MODAL"}
            if event.type == "LEFTMOUSE":
                self.down = event.value == "PRESS"
                if self.down:
                    self.session.begin_stroke()
            if event.type in {"MOUSEMOVE", "LEFTMOUSE"}:
                xy = (event.mouse_x - self.region.x, event.mouse_y - self.region.y)
                self.cursor = np.array(xy)
                if 0 <= xy[0] < self.region.width and 0 <= xy[1] < self.region.height:
                    view = self.area.spaces.active.region_3d
                    origin = view3d_utils.region_2d_to_origin_3d(self.region, view, xy)
                    ray = view3d_utils.region_2d_to_vector_3d(self.region, view, xy)
                    obj = self.session.obj.evaluated_get(
                        context.evaluated_depsgraph_get()
                    )
                    inverse = obj.matrix_world.inverted()
                    hit, point, _, _ = obj.ray_cast(
                        inverse @ origin, (inverse.to_3x3() @ ray).normalized()
                    )
                    if hit:
                        center = obj.matrix_world @ point
                        radius = f.restore_radius * np.linalg.norm(
                            np.ptp(self.session.original_neutral, axis=0)
                        )
                        right = view.view_matrix.inverted().to_3x3().col[0]
                        offset = view3d_utils.location_3d_to_region_2d(
                            self.region, view, center + right * radius
                        )
                        projected = view3d_utils.location_3d_to_region_2d(
                            self.region, view, center
                        )
                        if offset is not None and projected is not None:
                            self.cursor_radius = (offset - projected).length
                        if self.down:
                            self.session.stamp(
                                center, radius, f.restore_strength, f.mirror
                            )
                self.area.tag_redraw()
                return {"RUNNING_MODAL"}
            if event.type in {
                "MIDDLEMOUSE",
                "WHEELUPMOUSE",
                "WHEELDOWNMOUSE",
                "TRACKPADPAN",
                "TRACKPADZOOM",
                "NDOF_MOTION",
            }:
                return {"PASS_THROUGH"}
            return {"RUNNING_MODAL"}
        except Exception as exc:
            self.session.finish(False)
            self.cleanup()
            self.report({"ERROR"}, str(exc))
            return {"CANCELLED"}


class CM_OT_finish_repair(bpy.types.Operator):
    """Apply the neutral-head repair or restore the previous head and expression"""

    bl_idname = "cm.finish_repair"
    bl_label = "Finish Head Repair"
    accept: BoolProperty(default=True)

    def execute(self, context):
        operator = next(
            (
                operator
                for operator in _ACTIVE
                if operator.session.scene == context.scene
            ),
            None,
        )
        if operator is None:
            return {"CANCELLED"}
        operator.requested_finish = self.accept
        return {"FINISHED"}


@persistent
def cancel_active(_=None):
    for operator in list(_ACTIVE):
        try:
            operator.session.finish(False)
        finally:
            operator.cleanup()


def register():
    bpy.app.handlers.load_pre.append(cancel_active)


def unregister():
    cancel_active()
    if cancel_active in bpy.app.handlers.load_pre:
        bpy.app.handlers.load_pre.remove(cancel_active)
