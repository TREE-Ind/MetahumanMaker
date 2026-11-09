"""Interactive twelve-identity mixer with a cancelable live viewport preview."""

import math
import bpy
import numpy as np
from . import geometry, head_model, guided


class CM_OT_face_mixer(bpy.types.Operator):
    bl_idname = "cm.face_mixer"
    bl_label = "Face Mixer"
    bl_options = {"REGISTER", "UNDO", "BLOCKING"}

    def invoke(self, context, event):
        self._handler = self._timer = None
        try:
            if (
                not context.area
                or context.area.type != "VIEW_3D"
                or context.mode != "OBJECT"
            ):
                raise ValueError(
                    "Finish sculpting and open the Face Mixer from the 3D view"
                )
            self.area = context.area
            self.region = next(
                region for region in self.area.regions if region.type == "WINDOW"
            )
            self.scene = context.scene
            s, f = self.scene.cm_settings, self.scene.cm_features
            self.bank = head_model.identity_bank(f)
            self.original_head = f.generated_head
            self.original_source = s.source
            self.original_coefficients = head_model.coefficients(f).copy()
            self.had_coefficients = bool(len(f.identity_coefficients))
            self.original_mix = tuple(f.identity_mix)
            self.original_coordinates = None
            self.original_identity = None
            self.original_geometry_signature = None
            if self.original_head:
                self.original_identity = self.original_head.get("cm_identity")
                self.original_geometry_signature = self.original_head.get('cm_gnm_generated_geometry')
                self.original_coordinates = np.asarray(
                    [v.co[:] for v in self.original_head.data.vertices]
                )
            self.center = np.asarray(
                [min(250, self.region.width * 0.28), self.region.height * 0.55]
            )
            self.radius = min(140, self.region.height * 0.23)
            self.point = np.zeros(2)
            self.down = False
            self.pending = True
            self.needs_focus = not bool(self.original_head)
            self._timer = context.window_manager.event_timer_add(
                0.08, window=context.window
            )
            self._handler = bpy.types.SpaceView3D.draw_handler_add(
                self.draw_overlay, (), "WINDOW", "POST_PIXEL"
            )
            self.area.header_text_set(
                "Face Mixer: drag inside the wheel; Enter applies; Esc cancels"
            )
            context.window_manager.modal_handler_add(self)
            return {"RUNNING_MODAL"}
        except Exception as exc:
            self.cleanup(context)
            self.report({"ERROR"}, str(exc))
            return {"CANCELLED"}

    def preview(self, context):
        s, f = self.scene.cm_settings, self.scene.cm_features
        weights = geometry.identity_wheel_weights(self.point)
        f.identity_mix = weights
        head_model.update_coefficients(f, weights @ self.bank)
        head_model.create_head(context, s, f, invalidate=False)
        if self.needs_focus:
            guided.focus(context, [f.generated_head], "-Y")
            self.needs_focus = False
        self.pending = False
        self.area.tag_redraw()

    def cleanup(self, context):
        if self._handler is not None:
            bpy.types.SpaceView3D.draw_handler_remove(self._handler, "WINDOW")
            self._handler = None
        if self._timer is not None:
            context.window_manager.event_timer_remove(self._timer)
            self._timer = None
        if getattr(self, "area", None):
            self.area.header_text_set(None)
            self.area.tag_redraw()

    def finish(self, context, cancel=False):
        s, f = self.scene.cm_settings, self.scene.cm_features
        if cancel:
            f.identity_mix = self.original_mix
            head_model.update_coefficients(f, self.original_coefficients)
            if not self.had_coefficients:
                f.identity_coefficients.clear()
            if self.original_head:
                self.original_head.data.vertices.foreach_set(
                    "co", self.original_coordinates.ravel()
                )
                self.original_head.data.update()
                if self.original_identity is not None:
                    self.original_head["cm_identity"] = self.original_identity
                elif "cm_identity" in self.original_head:
                    del self.original_head["cm_identity"]
                if self.original_geometry_signature is not None:
                    self.original_head['cm_gnm_generated_geometry'] = self.original_geometry_signature
                elif 'cm_gnm_generated_geometry' in self.original_head:
                    del self.original_head['cm_gnm_generated_geometry']
            elif f.generated_head:
                temporary = f.generated_head
                mesh = temporary.data
                materials = list(mesh.materials)
                f.generated_head = None
                bpy.data.objects.remove(temporary, do_unlink=True)
                if not mesh.users:
                    bpy.data.meshes.remove(mesh)
                for material in materials:
                    if material and not material.users:
                        bpy.data.materials.remove(material)
            s.source = self.original_source
            s.status = "Face mix canceled. Previous identity restored."
        else:
            if self.pending:
                self.preview(context)
            if s.landmarks:
                bpy.ops.cm.clear_landmarks()
            s.status = "Face mix applied. Generated source ready for wrapping."
        self.cleanup(context)
        return {"CANCELLED"} if cancel else {"FINISHED"}

    def modal(self, context, event):
        try:
            if event.type in {"ESC", "RIGHTMOUSE"}:
                return self.finish(context, True)
            if event.type in {"RET", "NUMPAD_ENTER"}:
                return self.finish(context)
            if event.type == "LEFTMOUSE":
                position = np.asarray(
                    [event.mouse_x - self.region.x, event.mouse_y - self.region.y]
                )
                self.down = (
                    event.value == "PRESS"
                    and np.linalg.norm(position - self.center) <= self.radius + 12
                )
            if self.down and event.type in {"LEFTMOUSE", "MOUSEMOVE"}:
                position = np.asarray(
                    [event.mouse_x - self.region.x, event.mouse_y - self.region.y]
                )
                self.point = (position - self.center) / self.radius
                self.point /= max(np.linalg.norm(self.point), 1)
                self.pending = True
            if event.type == "TIMER" and self.pending:
                self.preview(context)
            return {"RUNNING_MODAL"}
        except Exception as exc:
            self.finish(context, True)
            self.report({"ERROR"}, str(exc))
            return {"CANCELLED"}

    def draw_overlay(self):
        if bpy.context.area != self.area:
            return
        import gpu
        import blf
        from gpu_extras.batch import batch_for_shader

        shader = gpu.shader.from_builtin("UNIFORM_COLOR")

        def disc(center, radius, color, segments=64):
            points = [
                (center[0] + math.cos(a) * radius, center[1] + math.sin(a) * radius)
                for a in np.linspace(0, 2 * math.pi, segments + 1)
            ]
            triangles = [
                p
                for i in range(segments)
                for p in (tuple(center), points[i], points[i + 1])
            ]
            shader.bind()
            shader.uniform_float("color", color)
            batch_for_shader(shader, "TRIS", {"pos": triangles}).draw(shader)

        gpu.state.blend_set("ALPHA")
        disc(self.center, self.radius + 22, (0.025, 0.03, 0.045, 0.94))
        disc(self.center, self.radius, (0.12, 0.17, 0.22, 0.95))
        weights = geometry.identity_wheel_weights(self.point)
        for index, angle in enumerate(np.arange(12) * 2 * np.pi / 12):
            point = (
                self.center + np.asarray([np.cos(angle), np.sin(angle)]) * self.radius
            )
            disc(point, 6 + weights[index] * 8, (0.2, 0.75, 0.9, 1))
            blf.position(0, point[0] - 5, point[1] - 23, 0)
            blf.size(0, 12)
            blf.color(0, 1, 1, 1, 1)
            blf.draw(0, str(index + 1))
        disc(self.center + self.point * self.radius, 7, (1, 0.65, 0.16, 1))
        blf.position(0, self.center[0] - 75, self.center[1] + self.radius + 40, 0)
        blf.size(0, 16)
        blf.color(0, 1, 1, 1, 1)
        blf.draw(0, "Face Mixer · 12 identities")
        blf.position(0, self.center[0] - 75, self.center[1] - self.radius - 42, 0)
        blf.size(0, 12)
        blf.draw(0, "Enter: Apply     Esc: Cancel")
        gpu.state.blend_set("NONE")
