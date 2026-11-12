"""Skin LookDev and isolated, asynchronous viewport-framed preview rendering."""

from pathlib import Path
from contextlib import contextmanager
import json
import os
import re
import subprocess
import uuid
import bpy
from bpy.app.handlers import persistent
import numpy as np
from mathutils import Matrix, Vector

_JOB = None
_UPDATE_SUSPENSIONS = 0
_INSPECTION_VIEWS = {}
_INSPECTION_FIELDS=('type','light','color_type','single_color','show_shadows','show_cavity','render_pass')
ENVIRONMENTS = [
    (name + ".exr", name.title(), "Blender studio environment")
    for name in (
        "studio",
        "forest",
        "courtyard",
        "interior",
        "city",
        "sunrise",
        "sunset",
        "night",
    )
]


def restore_inspection():
    for shading,values in list(_INSPECTION_VIEWS.values()):
        try:
            for key,value in values.items():setattr(shading,key,value)
        except ReferenceError:pass  # A closed editor space has no state to restore.
    _INSPECTION_VIEWS.clear()


@persistent
def _before_load(_):
    restore_inspection()


def register():
    if _before_load not in bpy.app.handlers.load_pre:bpy.app.handlers.load_pre.append(_before_load)


def inspect_update(self, context):
    """Inspect existing shading channels without editing material nodes/maps."""
    if _UPDATE_SUSPENSIONS or not context:
        return
    if not self.show_lookdev:
        restore_inspection();return
    if not context.screen:return
    for area in context.screen.areas:
        if area.type != 'VIEW_3D':continue
        shading=area.spaces.active.shading
        mode=self.lookdev_inspection
        key=shading.as_pointer()
        if mode=='COMBINED':
            previous=_INSPECTION_VIEWS.pop(key,None)
            if previous:
                for name,value in previous[1].items():setattr(shading,name,value)
        elif key not in _INSPECTION_VIEWS:
            values={name:(tuple(shading.single_color) if name=='single_color' else getattr(shading,name)) for name in _INSPECTION_FIELDS}
            _INSPECTION_VIEWS[key]=(shading,values)
        if mode=='SHAPE':
            shading.type='SOLID';shading.light='STUDIO'
            shading.color_type='SINGLE';shading.single_color=(.5,.5,.5)
            shading.show_shadows=True;shading.show_cavity=False
        else:
            shading.type='MATERIAL';shading.render_pass=mode
        area.tag_redraw()


@contextmanager
def suspend_updates():
    """Assign a group of settings without intermediate material/viewport edits."""
    global _UPDATE_SUSPENSIONS
    _UPDATE_SUSPENSIONS += 1
    try:
        yield
    finally:
        _UPDATE_SUSPENSIONS -= 1


def bind_roughness(material, image):
    """Keep native roughness detail, with one adjustable LookDev multiplier."""
    tree = material.node_tree
    for node in list(tree.nodes):
        if node.get('cm_roughness_scale') or (node.type == 'TEX_IMAGE' and node.get('cm_map_kind') == 'ROUGHNESS'):
            tree.nodes.remove(node)
    if image:
        texture = tree.nodes.new('ShaderNodeTexImage')
        texture.image = image
        texture['cm_map_kind'] = 'ROUGHNESS'
        scale = tree.nodes.new('ShaderNodeMath')
        scale.operation = 'MULTIPLY'
        scale.use_clamp = True
        scale['cm_roughness_scale'] = True
        scale.inputs[1].default_value = 1
        tree.links.new(texture.outputs['Color'], scale.inputs[0])
        for node in tree.nodes:
            if node.type == 'BSDF_PRINCIPLED':
                tree.links.new(scale.outputs[0], node.inputs['Roughness'])


def update(self, context):
    if _UPDATE_SUSPENSIONS or not context:
        return
    if not self.show_lookdev:
        restore_inspection();return
    s = context.scene.cm_settings
    for obj in (s.result, s.detail, self.body_result, self.body_detail, self.wm1, self.wm2, self.wm3):
        if not obj:
            continue
        for index, material in enumerate(list(obj.data.materials)):
            if not material or not material.use_nodes:
                continue
            if material.users > 1:
                material = material.copy()
                obj.data.materials[index] = material
            for node in material.node_tree.nodes:
                if node.type == "BSDF_PRINCIPLED":
                    node.inputs["Roughness"].default_value = self.lookdev_roughness
                    node.inputs[
                        "Subsurface Weight"
                    ].default_value = self.lookdev_subsurface
                elif node.type == "NORMAL_MAP":
                    node.inputs["Strength"].default_value = self.lookdev_normal_strength
                elif node.get('cm_roughness_scale'):
                    node.inputs[1].default_value = self.lookdev_roughness / .48
    if context.screen:
        for area in context.screen.areas:
            if area.type != "VIEW_3D":
                continue
            shading = area.spaces.active.shading
            shading.type = "MATERIAL"
            shading.use_scene_lights = shading.use_scene_world = False
            shading.use_studiolight_view_rotation = False
            shading.studio_light = self.lookdev_environment
            shading.studiolight_rotate_z = self.lookdev_rotation
            shading.studiolight_intensity = self.lookdev_intensity
            shading.studiolight_background_alpha = 0 if self.lookdev_transparent else 1
            shading.studiolight_background_blur = 0
            area.tag_redraw()
    inspect_update(self,context)


def viewport_area(context):
    if not context.screen:
        raise ValueError("Open a 3D viewport to render its framing")
    areas = sorted(
        (a for a in context.screen.areas if a.type == "VIEW_3D"), key=lambda a: a.x
    )
    if not areas:
        raise ValueError("Open a 3D viewport before rendering a preview")
    return areas[0]


def visible_rect(area):
    """Visible geometry pane in WINDOW coordinates, excluding overlapping UI."""
    window = next(region for region in area.regions if region.type == "WINDOW")
    left, bottom = window.x, window.y
    right, top = left + window.width, bottom + window.height
    space = area.spaces.active
    for region in area.regions:
        if region.type == "TOOLS" and space.show_region_toolbar and region.width > 1:
            left = max(left, region.x + region.width)
        elif region.type == "UI" and space.show_region_ui and region.width > 1:
            right = min(right, region.x)
        elif region.type in {
            "HEADER",
            "TOOL_HEADER",
            "ASSET_SHELF",
            "ASSET_SHELF_HEADER",
        }:
            if region.width < window.width * 0.5 or region.height <= 1:
                continue
            if region.y >= window.y + window.height * 0.5:
                top = min(top, region.y)
            else:
                bottom = max(bottom, region.y + region.height)
    if right <= left or top <= bottom:
        raise ValueError("The viewport is too small; expand its geometry pane")
    return window, (left - window.x, bottom - window.y, right - left, top - bottom)


def cropped_projection(projection, window_size, rectangle):
    left, bottom, width, height = rectangle
    full_width, full_height = window_size
    crop = Matrix.Identity(4)
    crop[0][0], crop[1][1] = full_width / width, full_height / height
    crop[0][3] = (full_width - 2 * left - width) / width
    crop[1][3] = (full_height - 2 * bottom - height) / height
    return crop @ projection


def preview_size(rectangle, long_side):
    width, height = rectangle[2:]
    scale = long_side / max(width, height)
    return max(1, round(width * scale)), max(1, round(height * scale))


def camera_from_view(context, scene, area, size, rectangle):
    space, view = area.spaces.active, area.spaces.active.region_3d
    window = next(region for region in area.regions if region.type == "WINDOW")
    target = cropped_projection(
        view.window_matrix.copy(), (window.width, window.height), rectangle
    )
    data = bpy.data.cameras.new("CM_PreviewCamera")
    camera = bpy.data.objects.new(data.name, data)
    scene.collection.objects.link(camera)
    camera.matrix_world = view.view_matrix.inverted()
    data.type = "PERSP" if view.is_perspective else "ORTHO"
    data.sensor_fit = "HORIZONTAL"
    data.sensor_width = 36
    data.clip_start, data.clip_end = space.clip_start, space.clip_end
    if data.type == "ORTHO":
        # Orthographic view matrices can place the origin inside the subject;
        # the viewport still draws negative-depth geometry, a render camera cannot.
        # Moving only along camera Z leaves every projected X/Y coordinate intact.
        inverse = camera.matrix_world.inverted()
        depths = [
            (inverse @ obj.matrix_world @ Vector(corner)).z
            for obj in scene.objects
            if obj.type in {"MESH", "CURVES", "CURVE"}
            for corner in obj.bound_box
        ]
        if depths:
            margin = max(data.clip_start * 2, (max(depths) - min(depths)) * .01, .001)
            distance = max(view.view_distance, max(depths) + margin)
            matrix = camera.matrix_world.copy()
            matrix.translation += matrix.to_3x3() @ Vector((0, 0, distance))
            camera.matrix_world = matrix
            data.clip_end = max(data.clip_end, distance - min(depths) + margin)
    data.lens = max(0.1, min(5000, 18 * target[0][0]))
    data.ortho_scale = 2 / max(target[0][0], 1e-8)
    graph = context.evaluated_depsgraph_get()

    def projection():
        return camera.calc_matrix_camera(graph, x=size[0], y=size[1])

    base = projection()
    if data.type == "PERSP":
        data.lens *= target[0][0] / base[0][0]
    else:
        data.ortho_scale *= base[0][0] / target[0][0]
    column = 2 if data.type == "PERSP" else 3
    base = projection()
    for axis, attribute in ((0, "shift_x"), (1, "shift_y")):
        setattr(data, attribute, 1)
        shifted = projection()
        sensitivity = shifted[axis][column] - base[axis][column]
        if abs(sensitivity) < 1e-8:
            raise ValueError("Could not reproduce the viewport camera shift")
        setattr(
            data, attribute, (target[axis][column] - base[axis][column]) / sensitivity
        )
    scene.camera = camera
    return camera, target, projection()


def preview_objects(context, area):
    s, f = context.scene.cm_settings, context.scene.cm_features
    objects = [
        s.result,
        s.detail,
        f.head_result,
        f.body_result,
        f.body_detail,
        f.eye_left,
        f.eye_right,
        f.teeth,
        s.source,
        s.template,
        f.body_template,
    ]
    for layer in (f.wm1, f.wm2, f.wm3):
        if layer:
            objects.extend([layer, *layer.children])
    objects.append(context.scene.cm_groom.groom)
    return list(
        dict.fromkeys(
            obj
            for obj in objects
            if obj
            and obj.type in {"MESH", "CURVES", "CURVE"}
            and obj.name in context.view_layer.objects
            and obj.visible_get(
                view_layer=context.view_layer, viewport=area.spaces.active
            )
        )
    )


def build_snapshot(context, folder, output, long_side):
    """Copy evaluated visible subjects and current images without editing the scene."""
    original = context.scene
    f = original.cm_features
    area = viewport_area(context)
    window, rectangle = visible_rect(area)
    objects = preview_objects(context, area)
    if not objects:
        raise ValueError("No character geometry is visible in the left viewport")
    size = preview_size(rectangle, long_side)
    scene = bpy.data.scenes.new("CM_IsolatedPreview")
    created_objects, created_data, materials, images = [], [], {}, {}
    world = None
    probe_bounds = None
    try:
        from . import blender_compat
        scene.render.engine = "CYCLES" if f.lookdev_renderer == "CYCLES" else blender_compat.eevee_engine()
        scene.cycles.samples = f.lookdev_samples
        scene.cycles.use_denoising = True
        scene.cycles.device = "CPU"
        scene.eevee.taa_render_samples = f.lookdev_samples
        scene.render.resolution_x, scene.render.resolution_y = size
        scene.render.resolution_percentage = 100
        scene.render.film_transparent = f.lookdev_transparent
        blender_compat.render_image_format(scene.render.image_settings,'PNG')
        scene.render.image_settings.color_mode = "RGBA"
        scene.render.filepath = str(output)
        # Reserve capacity for the interactive editor and Unreal during previews.
        scene.render.threads_mode = "FIXED"
        scene.render.threads = max(1, min(8, (os.cpu_count() or 2) - 2))
        scene.unit_settings.scale_length = original.unit_settings.scale_length
        scene.view_settings.view_transform = original.view_settings.view_transform
        scene.view_settings.look = original.view_settings.look
        scene.view_settings.exposure = original.view_settings.exposure
        scene.view_settings.gamma = original.view_settings.gamma
        scene.display_settings.display_device = original.display_settings.display_device
        graph = context.evaluated_depsgraph_get()
        for obj in objects:
            evaluated = obj.evaluated_get(graph)
            if obj.type == "MESH":
                data = bpy.data.meshes.new_from_object(
                    evaluated, preserve_all_data_layers=True, depsgraph=graph
                )
            else:
                data = evaluated.data.copy()
            created_data.append(data)
            clone = bpy.data.objects.new(obj.name + "_Preview", data)
            created_objects.append(clone)
            scene.collection.objects.link(clone)
            clone.matrix_world = obj.matrix_world.copy()
            for index, material in enumerate(list(data.materials)):
                if not material:
                    continue
                if material not in materials:
                    copied = material.copy()
                    materials[material] = copied
                    if copied.use_nodes:
                        for node in copied.node_tree.nodes:
                            if node.type == "TEX_IMAGE" and node.image:
                                if node.image not in images:
                                    from .textures import copy_image_pixels
                                    image = copy_image_pixels(node.image)
                                    image.pack()
                                    images[node.image] = image
                                node.image = images[node.image]
                data.materials[index] = materials[material]
        camera, target, actual = camera_from_view(context, scene, area, size, rectangle)
        created_objects.append(camera)
        created_data.append(camera.data)
        light = next(
            (
                light
                for light in context.preferences.studio_lights
                if light.type == "WORLD" and light.name == f.lookdev_environment
            ),
            None,
        )
        if not light or not Path(light.path).is_file():
            raise ValueError(
                "Selected studio environment is unavailable in this Blender install"
            )
        world = bpy.data.worlds.new("CM_PreviewWorld")
        scene.world = world
        world.use_nodes = True
        nodes, links = world.node_tree.nodes, world.node_tree.links
        texture = nodes.new("ShaderNodeTexEnvironment")
        environment = bpy.data.images.load(light.path, check_existing=False)
        environment.alpha_mode = "NONE"
        environment.pack()
        images[environment] = environment
        texture.image = environment
        # Leave the vector input unlinked so Eevee uses its native world ray
        # coordinates for both the camera and irradiance bake.
        texture.texture_mapping.rotation[2] = f.lookdev_rotation
        background = nodes.get("Background")
        background.inputs["Strength"].default_value = f.lookdev_intensity
        links.new(texture.outputs["Color"], background.inputs["Color"])
        if blender_compat.is_eevee(scene.render.engine):
            # Material Preview builds studio irradiance internally. A normal
            # Eevee render needs an irradiance volume to receive HDRI diffuse.
            clouds = []
            for obj in created_objects:
                if obj.type != "MESH":
                    continue
                positions = np.empty(len(obj.data.vertices)*3)
                obj.data.vertices.foreach_get("co",positions)
                matrix = np.asarray(obj.matrix_world)
                clouds.append(positions.reshape(-1,3) @ matrix[:3,:3].T + matrix[:3,3])
            bounds = np.concatenate(clouds) if clouds else np.empty((0,3))
            if len(bounds):
                probe_data = bpy.data.lightprobes.new("CM_PreviewIrradiance", "VOLUME")
                probe_data.resolution_x = probe_data.resolution_y = 4
                probe_data.resolution_z = 8
                probe_data.bake_samples = 32
                probe_data.capture_world = True
                probe = bpy.data.objects.new(probe_data.name, probe_data)
                scene.collection.objects.link(probe)
                probe.location = (bounds.min(0) + bounds.max(0)) * 0.5
                probe.scale = np.maximum((bounds.max(0) - bounds.min(0)) * 0.65, 0.1)
                probe_bounds = [bounds.min(0).tolist(),bounds.max(0).tolist()]
        snapshot = folder / "snapshot.blend"
        bpy.data.libraries.write(
            str(snapshot), {scene}, path_remap="ABSOLUTE", compress=True
        )
        metadata = {
            "scene": scene.name,
            "snapshot": str(snapshot),
            "output": str(output),
            "size": size,
            "rectangle": rectangle,
            "window_size": [window.width, window.height],
            "camera_type": camera.data.type,
            "target_projection": [list(row) for row in target],
            "camera_projection": [list(row) for row in actual],
            "view_matrix": [
                list(row) for row in camera.matrix_world.inverted()
            ],
            "viewport_view_matrix": [list(row) for row in area.spaces.active.region_3d.view_matrix],
            "camera_clip": [camera.data.clip_start,camera.data.clip_end],
            "subject_depths": [(camera.matrix_world.inverted() @ obj.matrix_world @ Vector(corner)).z for obj in created_objects if obj.type in {'MESH','CURVES','CURVE'} for corner in obj.bound_box],
            "subjects": [obj.name for obj in objects],
            "probe_bounds":probe_bounds,
        }
        return metadata
    finally:
        for obj in scene.objects:
            if obj not in created_objects:
                created_objects.append(obj)
                if obj.data and obj.data not in created_data:
                    created_data.append(obj.data)
        bpy.data.scenes.remove(scene)
        for obj in created_objects:
            bpy.data.objects.remove(obj, do_unlink=True)
        for data in created_data:
            collection = (
                bpy.data.meshes
                if isinstance(data, bpy.types.Mesh)
                else bpy.data.cameras
                if isinstance(data, bpy.types.Camera)
                else bpy.data.hair_curves
                if isinstance(data, bpy.types.Curves)
                else bpy.data.lightprobes
                if isinstance(data, bpy.types.LightProbe)
                else bpy.data.curves
            )
            collection.remove(data)
        for material in materials.values():
            bpy.data.materials.remove(material)
        if world:
            bpy.data.worlds.remove(world)
        for image in set(images.values()):
            bpy.data.images.remove(image)


def is_rendering():
    return _JOB is not None


def _set_status(job, message):
    try:
        job["scene"].cm_settings.status = message
    except ReferenceError:
        pass


def _poll_job():
    global _JOB
    job = _JOB
    if not job:
        return None
    if job["process"].poll() is None:
        return 0.25
    try:
        report = (
            json.loads(job["report"].read_text(encoding="utf-8"))
            if job["report"].is_file()
            else {}
        )
        if report.get("status") == "complete":
            job["scene"].cm_features.preview_output = report["output"]
            _set_status(job, "Preview saved: " + Path(report["output"]).name)
        elif job.get("cancel_requested"):
            _set_status(job, "Preview render canceled. Your character is unchanged.")
        else:
            _set_status(
                job,
                "Preview failed: "
                + report.get("error", "See " + str(job["folder"] / "render.log")),
            )
    except (OSError, ValueError, ReferenceError) as exc:
        print("MetaHuman Maker preview completion:", str(exc))
    finally:
        # Only remove the known disposable snapshot, retaining request/log/report evidence.
        _JOB = None
        try:
            (job["folder"] / "snapshot.blend").unlink(missing_ok=True)
        except OSError as exc:
            print("MetaHuman Maker preview snapshot cleanup:", str(exc))
        for window in bpy.context.window_manager.windows:
            for area in window.screen.areas:
                area.tag_redraw()
    return None


def start_render(context):
    if context.scene.cm_features.show_lookdev and context.scene.cm_features.lookdev_inspection!='COMBINED':
        raise ValueError('Switch Inspect to Beauty before rendering a studio preview')
    global _JOB
    if _JOB:
        _poll_job()
        if _JOB:
            raise ValueError(
                "A preview is rendering. Cancel it or wait for completion."
            )
    if context.mode != "OBJECT":
        raise ValueError("Finish sculpting before rendering the preview")
    s, f = context.scene.cm_settings, context.scene.cm_features
    from . import resource_budget
    resource_budget.require('preview rendering',resource_budget.GIB+int(f.lookdev_size)**2*64+
        resource_budget.meshes(context.scene.objects))
    update(f, context)
    root = Path(bpy.path.abspath(s.project_dir)) / "Preview"
    identifier = uuid.uuid4().hex[:12]
    folder = root / ".jobs" / identifier
    folder.mkdir(parents=True, exist_ok=False)
    name = re.sub(r"[^A-Za-z0-9_-]", "_", s.character_name).strip("_") or "Character"
    output = root / (name + "_" + identifier + ".png")
    metadata = build_snapshot(context, folder, output, int(f.lookdev_size))
    metadata["report"] = str(folder / "report.json")
    request = folder / "request.json"
    request.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    with (folder / "render.log").open("w", encoding="utf-8") as log:
        process = subprocess.Popen(
            [
                bpy.app.binary_path,
                "--background",
                "--factory-startup",
                metadata["snapshot"],
                "--python",
                str(Path(__file__).with_name("preview_worker.py")),
                "--",
                str(request),
            ],
            stdout=log,
            stderr=subprocess.STDOUT,
            creationflags=(subprocess.CREATE_NO_WINDOW | subprocess.BELOW_NORMAL_PRIORITY_CLASS)
            if os.name == "nt" else 0,
        )
    _JOB = {
        "process": process,
        "folder": folder,
        "report": Path(metadata["report"]),
        "scene": context.scene,
    }
    s.status = "Rendering preview in the background. You can keep editing."
    bpy.app.timers.register(_poll_job, first_interval=0.25, persistent=True)
    return process.pid


class CM_OT_preview_render(bpy.types.Operator):
    """Render a studio beauty PNG using the left viewport framing"""
    bl_idname = "cm.preview_render"
    bl_label = "Render Preview"

    def execute(self, context):
        try:
            start_render(context)
            return {"FINISHED"}
        except Exception as exc:
            context.scene.cm_settings.status = str(exc)
            self.report({"ERROR"}, str(exc))
            return {"CANCELLED"}


class CM_OT_cancel_preview(bpy.types.Operator):
    """Stop the current preview worker and preserve your character and last saved image"""
    bl_idname = "cm.cancel_preview"
    bl_label = "Cancel Preview"

    def execute(self, context):
        if _JOB and _JOB["process"].poll() is None:
            _JOB["cancel_requested"] = True
            _JOB["process"].terminate()
            _set_status(_JOB, "Canceling preview render…")
        return {"FINISHED"}


class CM_OT_open_preview(bpy.types.Operator):
    """Open the most recently completed preview image"""
    bl_idname = "cm.open_preview"
    bl_label = "Open Saved Preview"

    def execute(self, context):
        path = context.scene.cm_features.preview_output
        if not path or not Path(path).is_file():
            self.report({"ERROR"}, "Render a preview first")
            return {"CANCELLED"}
        bpy.ops.wm.path_open(filepath=path)
        return {"FINISHED"}


def draw(layout, context):
    f = context.scene.cm_features
    layout.prop(f, "show_lookdev", text="LookDev", toggle=True)
    if not f.show_lookdev:
        return
    box = layout.box()
    box.prop(f,'lookdev_inspection',text='Inspect')
    if f.lookdev_inspection!='COMBINED':
        box.label(text='Viewport inspection; materials and maps retained.')
    box.prop(f, "lookdev_skin_controls", toggle=True)
    if f.lookdev_skin_controls:
        box.prop(f, "lookdev_roughness", text="Roughness")
        box.prop(f, "lookdev_subsurface", text="Subsurface")
        box.prop(f, "lookdev_normal_strength", text="Normal Detail")
    box.prop(f, "lookdev_environment_controls", toggle=True)
    if f.lookdev_environment_controls:
        box.prop(f, "lookdev_environment", text="Studio")
        box.prop(f, "lookdev_rotation", text="Rotation")
        box.prop(f, "lookdev_intensity", text="Strength")
        box.prop(f, "lookdev_transparent", text="Transparent Background")
    box.prop(f, "lookdev_render_controls", toggle=True)
    if f.lookdev_render_controls:
        box.label(text="Current left viewport")
        box.prop(f, "lookdev_size", text="Size")
        box.prop(f, "lookdev_renderer", text="Renderer")
        box.prop(f, "lookdev_samples", text="Samples")
        if is_rendering():
            box.operator("cm.cancel_preview", text="Cancel Background Render")
        else:
            row=box.row();row.enabled=f.lookdev_inspection=='COMBINED'
            row.operator(
                "cm.preview_render", text="Render Preview", icon="RENDER_STILL"
            )
            if f.lookdev_inspection!='COMBINED':box.label(text='Switch to Beauty for a studio render.')
        if f.preview_output:
            box.operator(
                "cm.open_preview", text="Open Saved Preview", icon="IMAGE_DATA"
            )


CLASSES = (CM_OT_preview_render, CM_OT_cancel_preview, CM_OT_open_preview)


def unregister():
    global _JOB
    restore_inspection()
    if _before_load in bpy.app.handlers.load_pre:bpy.app.handlers.load_pre.remove(_before_load)
    if _JOB and _JOB["process"].poll() is None:
        _JOB["process"].terminate()
    if bpy.app.timers.is_registered(_poll_job):
        bpy.app.timers.unregister(_poll_job)
    _JOB = None
