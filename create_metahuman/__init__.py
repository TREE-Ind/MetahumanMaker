bl_info = {
    "name": "MetaHuman Maker",
    "author": "TREE Industries and contributors",
    "version": (0, 346, 0),
    "blender": (5, 2, 0),
    "location": "View3D > Sidebar > MetaHuman Maker",
    "description": "Character authoring for artists and AI agents; Unreal MetaHuman exchange",
    "category": "Mesh",
}

import json
import traceback
from pathlib import Path
import bpy
import numpy as np
from bpy.props import (
    StringProperty,
    PointerProperty,
    CollectionProperty,
    IntProperty,
    FloatProperty,
    EnumProperty,
    BoolProperty,
)
from bpy_extras.io_utils import ImportHelper, ExportHelper
from mathutils import Matrix
from . import mesh_ops, geometry, scan_projection


def mesh_poll(self, obj):
    return obj.type == "MESH"


def accepted_head_changed(settings, context):
    from . import groom_follow
    groom_follow.accepted_changed(settings, context)


class CM_Landmark(bpy.types.PropertyGroup):
    source: PointerProperty(type=bpy.types.Object)
    template: PointerProperty(type=bpy.types.Object)


class CM_Settings(bpy.types.PropertyGroup):
    source: PointerProperty(name="Source", type=bpy.types.Object, poll=mesh_poll)
    template: PointerProperty(name="Template", type=bpy.types.Object, poll=mesh_poll)
    result: PointerProperty(name="Result", type=bpy.types.Object, poll=mesh_poll, update=accepted_head_changed)
    detail: PointerProperty(name="Detail", type=bpy.types.Object, poll=mesh_poll)
    character_name: StringProperty(name="Character", default="MyCharacter")
    project_dir: StringProperty(
        name="Project Folder", subtype="DIR_PATH", default="//CreateMH_Project/"
    )
    part: EnumProperty(
        name="Template Part",
        items=[
            ("HEAD", "Head", "MetaHuman face surface only"),
            ("BODY", "Body", "MetaHuman body surface only"),
            ("FULL_BODY", "Full Body", "Separate head and body reference surfaces"),
        ],
    )
    stage: StringProperty(default="IMPORT")
    status: StringProperty(
        default="Import a source and a licensed template, or create a demo."
    )
    landmarks: CollectionProperty(type=CM_Landmark)
    iterations: IntProperty(name="Iterations", default=20, min=1, max=100)
    projection_strength: FloatProperty(
        name="Projection", default=0.65, min=0.01, max=1.0
    )
    smoothing: FloatProperty(name="Regularization", default=0.35, min=0.0, max=0.95)
    max_distance: FloatProperty(
        name="Search / diagonal", default=0.15, min=0.001, max=2.0
    )
    landmark_radius: FloatProperty(
        name="Landmark / diagonal", default=0.25, min=0.01, max=2.0
    )
    landmark_regularization: FloatProperty(
        name="Landmark Smoothing", default=1e-6, min=1e-6, max=1.0,
        description="Smooth noisy or dense landmark constraints to reduce unsupported deformation between markers",
    )
    last_error: FloatProperty(name="Surface RMS", default=0, precision=6)
    detail_levels: IntProperty(name="Subdivision Levels", min=1, max=3, default=1)
    detail_strength: FloatProperty(
        name="Scan Detail Strength", default=1.0, min=0.0, max=1.0,
        get=scan_projection.strength_get, set=scan_projection.strength_set,
    )
    map_size: EnumProperty(
        name="Map Size",
        items=[(str(s), str(s), "8K bakes can need over 8 GB RAM. Prefer 4K when memory or disk space is limited." if s == 8192 else "") for s in (256, 512, 1024, 2048, 4096, 8192)],
        default="2048",
    )
    bake_distance: FloatProperty(
        name="Bake Distance", default=0.02, min=0.00001, max=100, precision=4
    )
    web_texture_size: EnumProperty(name='GLB Texture Limit', default='2048', items=[
        (str(size), str(size), 'Maximum embedded image dimension; source images are preserved')
        for size in (512, 1024, 2048, 4096)])
    web_skin_finish: BoolProperty(name='Use Skin Finish', default=True,
        description='Normalize accepted derived roughness around Skin Finish; disable to export Blender LookDev roughness directly')
    unreal_engine_dir: StringProperty(
        name="Unreal Engine",
        subtype="DIR_PATH",
        default="C:/Program Files/Epic Games/UE_5.8/",
    )
    unreal_destination: StringProperty(
        name="Content Folder", default="/Game/CreateMetaHuman"
    )
    unreal_project: StringProperty(
        name="Unreal Project Name", description="Required if multiple editors are open"
    )
    create_character: BoolProperty(name="Create Native MetaHuman", default=True)
    canonical_order: BoolProperty(
        name="Canonical DNA Vertex Order",
        default=False,
        description="Only enable if the template uses canonical MetaHuman DNA vertex indices; otherwise match by UV",
    )
    body_height: FloatProperty(name="Height", default=1, min=0.5, max=1.5)
    body_width: FloatProperty(name="Width", default=1, min=0.5, max=1.5)
    body_depth: FloatProperty(name="Depth", default=1, min=0.5, max=1.5)
    body_waist: FloatProperty(name="Waist", default=1, min=0.5, max=1.5)
    body_shoulders: FloatProperty(name="Shoulders", default=1, min=0.5, max=1.5)
    body_hips: FloatProperty(name="Hips", default=1, min=0.5, max=1.5)


class SafeExecute:
    def execute(self, context):
        try:
            operator_id=self.bl_idname.lower().replace('_ot_','.')
            from . import eye_fit_jobs
            if eye_fit_jobs.running() and operator_id!='cm.save_project':
                raise ValueError('Finish or cancel the source surface fit before another workflow action')
            from . import skin_reconstruction
            if skin_reconstruction.running() and operator_id!='cm.save_project':
                raise ValueError('Apply or discard skin reconstruction before another workflow action')
            from . import paint_tools
            from . import landmark_tools
            if operator_id != 'cm.remove_last_landmark':
                landmark_tools.finish_active(context)
            from . import scan_jobs
            if scan_jobs.running() and operator_id != 'cm.save_project':
                raise ValueError('Finish or cancel scan projection before another workflow action')
            from . import auto_wrap_jobs
            if auto_wrap_jobs.running() and operator_id != 'cm.save_project':
                raise ValueError('Finish or cancel Auto Wrap before another workflow action')
            if paint_tools.running():
                raise ValueError('Apply or cancel Smart Paint before another workflow action')
            self.run(context, context.scene.cm_settings)
            context.scene.cm_features.last_action_error = False
            return {"FINISHED"}
        except Exception as exc:
            traceback.print_exc()
            context.scene.cm_settings.status = str(exc)
            context.scene.cm_features.last_action_error = True
            self.report({"ERROR"}, str(exc))
            return {"CANCELLED"}


class CM_OT_import(SafeExecute, bpy.types.Operator, ImportHelper):
    bl_idname = "cm.import_mesh"
    bl_label = "Import Mesh"
    bl_options = {"REGISTER", "UNDO"}
    filter_glob: StringProperty(default="*.obj;*.fbx;*.glb;*.gltf", options={"HIDDEN"})
    role: EnumProperty(
        items=[
            ("source", "Source", ""),
            ("template", "Head Template", ""),
            ("body_template", "Body Template", ""),
        ]
    )
    prepare_workflow: BoolProperty(default=False, options={'HIDDEN'})
    wrap_mode: EnumProperty(name='Character', items=[
        ('FULL_HEAD', 'Wrap Full Head', 'Head with matching eyes and teeth'),
        ('HEAD_ONLY', 'Wrap Head Only', 'Head surface without internal meshes'),
        ('FULL_BODY', 'Wrap Full Body', 'Head and body with matching internal meshes'),
    ])
    placement: EnumProperty(name='Import Placement', items=[
        ('AUTO', 'Automatic', 'Keep recognized MetaHuman UV surfaces unchanged; fit other meshes to the references'),
        ('FIT', 'Fit to References', 'Match source bounds and orientation to the references, including recognized MetaHumans'),
        ('KEEP', 'Keep Imported', 'Keep the imported size and placement; use Manual wrapping to retain this alignment'),
    ], default='AUTO')

    def invoke(self, context, event):
        if self.prepare_workflow and not self.properties.is_property_set('wrap_mode'):
            s, f = context.scene.cm_settings, context.scene.cm_features
            self.wrap_mode = 'FULL_BODY' if s.part == 'FULL_BODY' else 'FULL_HEAD' if f.full_head else 'HEAD_ONLY'
        return ImportHelper.invoke(self, context, event)

    def draw(self, context):
        if self.role == 'source' and self.prepare_workflow:
            column = self.layout.column(align=True)
            for mode in ('FULL_HEAD', 'HEAD_ONLY', 'FULL_BODY'):
                column.prop_enum(self, 'wrap_mode', mode)
            self.layout.label(text='One character mesh.')
            self.layout.label(text='Local MetaHuman references')
            self.layout.label(text='load automatically.')
            self.layout.label(text='Import Placement')
            self.layout.prop(self, 'placement', text='')

    def run(self, context, s):
        if self.role == 'source' and self.prepare_workflow:
            from . import source_import
            source_import.start(context, s, context.scene.cm_features, self.filepath, self.wrap_mode, placement=self.placement)
            return
        previous = set(bpy.data.objects)
        ext = Path(self.filepath).suffix.lower()
        if ext == ".obj":
            bpy.ops.wm.obj_import(filepath=self.filepath)
        elif ext == ".fbx":
            bpy.ops.import_scene.fbx(filepath=self.filepath, use_custom_normals=True)
        elif ext in {".glb", ".gltf"}:
            bpy.ops.import_scene.gltf(filepath=self.filepath)
        else:
            raise ValueError("Supported formats: OBJ, FBX, GLB and GLTF")
        meshes = [o for o in bpy.data.objects if o not in previous and o.type == "MESH"]
        if not meshes:
            raise ValueError("File contains no mesh")
        obj = max(meshes, key=lambda o: len(o.data.vertices))
        setattr(
            context.scene.cm_features if self.role == "body_template" else s,
            self.role,
            obj,
        )
        if self.role != "source":
            height = (
                np.ptp(mesh_ops.coordinates(obj), axis=0)[2]
                * context.scene.unit_settings.scale_length
            )
            if height > (10 if self.role == "body_template" else 2):
                obj.matrix_world = Matrix.Scale(0.01, 4) @ obj.matrix_world
            context.view_layer.update()
            if obj.modifiers:
                original = obj
                head = s.template
                s.template = obj
                try:
                    if bpy.ops.cm.prepare_template() != {"FINISHED"}:
                        raise ValueError("Could not prepare the imported template")
                    obj = s.template
                    if self.role == "body_template":
                        context.scene.cm_features.body_template = obj
                    original.hide_set(True)
                    original.hide_render = True
                finally:
                    if self.role == "body_template":
                        s.template = head
        mesh_ops.activate(context, obj)
        s.status = f"Imported {len(meshes)} mesh(es); {obj.name} is {self.role}. Check the object selector."
        from .guided import focus

        focus(context, [obj], context.scene.cm_features.front_axis)


class CM_OT_use_active(SafeExecute, bpy.types.Operator):
    bl_idname = "cm.use_active"
    bl_label = "Use Active Mesh"
    bl_options = {"REGISTER", "UNDO"}
    role: StringProperty(default="source")

    def run(self, context, s):
        mesh_ops.check_mesh(context.object, self.role)
        setattr(
            context.scene.cm_features if self.role == "body_template" else s,
            self.role,
            context.object,
        )


class CM_OT_prepare_template(SafeExecute, bpy.types.Operator):
    bl_idname = "cm.prepare_template"
    bl_label = "Prepare Static Template Copy"
    bl_description = "Copy the visible template at its current pose, applying modifiers and preserving UVs"
    bl_options = {"REGISTER", "UNDO"}

    def run(self, context, s):
        mesh_ops.check_mesh(s.template, "Template")
        original_template = s.template
        evaluated = s.template.evaluated_get(context.evaluated_depsgraph_get())
        data = bpy.data.meshes.new_from_object(
            evaluated,
            preserve_all_data_layers=True,
            depsgraph=context.evaluated_depsgraph_get(),
        )
        obj = bpy.data.objects.new(s.template.name + "_StaticTemplate", data)
        context.collection.objects.link(obj)
        obj.matrix_world = s.template.matrix_world.copy()
        from . import body_pose
        skeleton=body_pose.capture_skeleton(original_template)
        if skeleton:
            obj[body_pose.KEY]=json.dumps(skeleton)
        elif original_template.get(body_pose.KEY):
            obj[body_pose.KEY]=original_template[body_pose.KEY]
        for group in original_template.vertex_groups:
            obj.vertex_groups.new(name=group.name)
        s.template = obj
        mesh_ops.activate(context, obj)
        s.status = "Static template copied at the current pose. Confirm it contains only the intended surface."


def add_marker(context, obj, local, name):
    role = "Source" if obj == context.scene.cm_settings.source else "Template"
    name = f"{len(context.scene.cm_settings.landmarks):02d} {role}"
    marker = bpy.data.objects.new(name, None)
    context.scene.collection.objects.link(marker)
    marker.parent = obj
    marker.matrix_parent_inverse = Matrix.Identity(4)
    marker.location = local
    marker.empty_display_type = "SPHERE"
    marker.empty_display_size = max(obj.dimensions) * 0.012
    marker.show_in_front = True
    marker.show_name = True
    marker.hide_render = True
    marker["cm_landmark"] = True
    return marker


class CM_OT_show_mesh(SafeExecute, bpy.types.Operator):
    bl_idname = "cm.show_mesh"
    bl_label = "Show Workflow Mesh"
    role: StringProperty(default="result")

    def run(self, context, s):
        obj = getattr(s, self.role)
        mesh_ops.check_mesh(obj, self.role)
        for name in ("source", "template", "result", "detail"):
            item = getattr(s, name)
            if item:
                item.hide_set(item != obj)
        for pair in s.landmarks:
            for marker in (pair.source, pair.template):
                if marker:
                    marker.hide_set(True)
        mesh_ops.activate(context, obj)


class CM_OT_landmarks(bpy.types.Operator):
    bl_idname = 'cm.landmarks'
    bl_label = 'Place or Edit Landmarks'
    bl_description = 'Click Source then Template; right-drag points to correct them; Ctrl Z undoes an action; Esc finishes'
    bl_options = {'REGISTER', 'UNDO', 'BLOCKING'}

    def invoke(self, context, event):
        from . import landmark_tools
        return landmark_tools.invoke(self, context, event)

    def modal(self, context, event):
        try:
            return self.session.modal(context, event)
        except (RuntimeError, ReferenceError, ValueError) as exc:
            self.session.finish(context)
            self.report({'ERROR'}, str(exc))
            return {'CANCELLED'}


class CM_OT_remove_landmark(SafeExecute, bpy.types.Operator):
    bl_idname = 'cm.remove_last_landmark'
    bl_label = 'Remove Last Landmark Pair'
    bl_options = {'REGISTER', 'UNDO'}

    def run(self, context, s):
        from . import landmark_tools
        landmark_tools.remove_last(context)


class CM_OT_clear_landmarks(SafeExecute, bpy.types.Operator):
    bl_idname = "cm.clear_landmarks"
    bl_label = "Clear Landmarks"
    bl_options = {"REGISTER", "UNDO"}

    def run(self, context, s):
        for pair in s.landmarks:
            for marker in (pair.source, pair.template):
                if marker and marker.type=='EMPTY' and marker.get('cm_landmark'):
                    bpy.data.objects.remove(marker, do_unlink=True)
        s.landmarks.clear()


class CM_OT_landmark_file(SafeExecute, bpy.types.Operator, ExportHelper):
    bl_idname = "cm.landmark_file"
    bl_label = "Landmark File"
    filename_ext = ".json"
    filter_glob: StringProperty(default="*.json", options={"HIDDEN"})
    load: BoolProperty(default=False)

    def run(self, context, s):
        mesh_ops.check_inputs(s)
        if self.load:
            pairs = json.loads(Path(self.filepath).read_text(encoding="utf-8"))
            if not isinstance(pairs, list) or len(pairs) > 512:
                raise ValueError("Expected a list of up to 512 landmark pairs")
            # Validate the complete file before changing the scene.
            parents = []
            for pair in pairs:
                if not isinstance(pair, dict) or not all(key in pair for key in ('source', 'template')):
                    raise ValueError('Each landmark pair needs source and template coordinates')
                if not isinstance(pair.get('name', ''), str):
                    raise ValueError('Landmark names must be text')
                part = pair.get('template_part')
                if part not in {None, 'HEAD', 'BODY'}:
                    raise ValueError('Invalid landmark template part')
                if part == 'BODY':
                    target = context.scene.cm_features.body_template if s.part=='FULL_BODY' else s.template if s.part=='BODY' else None
                else:
                    target = s.template
                mesh_ops.check_mesh(target, 'Matching landmark template part')
                if part == 'HEAD' and s.part=='BODY':
                    raise ValueError('These head landmarks need a head template')
                parents.append(target)
                for role in ("source", "template"):
                    v = np.asarray(pair[role], dtype=float)
                    if v.shape != (3,) or not np.isfinite(v).all():
                        raise ValueError("Invalid landmark coordinate")
            bpy.ops.cm.clear_landmarks()
            for data, target in zip(pairs, parents):
                pair = s.landmarks.add()
                pair.name = data.get('name', '')
                for role in ("source", "template"):
                    setattr(
                        pair,
                        role,
                        add_marker(
                            context,
                            s.source if role=='source' else target,
                            data[role],
                            f"CM_{len(s.landmarks):02d}_{role}",
                        ),
                    )
        else:
            allowed = {s.template}
            if s.part=='FULL_BODY' and context.scene.cm_features.body_template:
                allowed.add(context.scene.cm_features.body_template)
            if any(p.source and p.template and (p.source.parent!=s.source or p.template.parent not in allowed) for p in s.landmarks):
                raise ValueError('Landmarks belong to another source/template; cannot save mismatched points')
            data = [
                {"source": p.source.location[:], "template": p.template.location[:],
                 'name': p.name,
                 'template_part': 'BODY' if s.part=='BODY' or (context.scene.cm_features.body_template and p.template.parent==context.scene.cm_features.body_template) else 'HEAD'}
                for p in s.landmarks
                if p.source and p.template
            ]
            Path(self.filepath).write_text(json.dumps(data, indent=2), encoding="utf-8")


class CM_OT_align(SafeExecute, bpy.types.Operator):
    bl_idname = "cm.align"
    bl_label = "Align Source"
    bl_options = {"REGISTER", "UNDO"}
    bounds: BoolProperty(default=False)

    def run(self, context, s):
        mesh_ops.align_source(context, s, self.bounds)
        s.status = (
            "Source aligned. Inspect the eyes, mouth and silhouette before wrapping."
        )


class CM_OT_wrap(SafeExecute, bpy.types.Operator):
    bl_idname = "cm.wrap"
    bl_label = "Wrap Template to Source"
    bl_options = {"REGISTER", "UNDO"}
    defer_wrinkles: BoolProperty(default=False, options={'HIDDEN'})

    def run(self, context, s):
        f = context.scene.cm_features
        from .features import generated_head_update, wrap_generated_head
        if s.source and s.source.get('cm_source_alignment')=='METAHUMAN':
            from . import native_return
            native_return.conform(context)
            return
        if not bpy.app.background:
            from . import auto_wrap_jobs
            if not auto_wrap_jobs.publishing() and f.wrap_method=='AUTO' and auto_wrap_jobs.needs_detection(context):
                auto_wrap_jobs.start(context,wrap=True)
                return
            if not auto_wrap_jobs.publishing():
                auto_wrap_jobs.start_fit(context,guided=False,defer_wrinkles=self.defer_wrinkles)
                return
        head_update = generated_head_update(s)
        if head_update:
            wrap_generated_head(context,s,f,defer_wrinkles=self.defer_wrinkles)
            return
        if f.wrap_method == "AUTO" and not len(s.landmarks):
            from .automation import auto_landmarks

            auto_landmarks(context, s, f)
        if s.part == "FULL_BODY":
            from .features import wrap_full_body

            wrap_full_body(context, s, f)
            return
        mesh_ops.wrap(context, s)
        bpy.ops.cm.show_mesh(role="result")
        s.status = f"Wrapped. Surface RMS: {s.last_error:.6f} scene units. Inspect lips, eyes and inner surfaces."
        if s.result.get('cm_wrap_recovery'):
            s.status += ' Conservative recovery used; compare likeness before continuing.'


class CM_OT_sculpt(SafeExecute, bpy.types.Operator):
    bl_idname = "cm.sculpt"
    bl_label = "Sculpt Result"
    bl_options = {"REGISTER", "UNDO"}
    detail: BoolProperty(default=False)
    part: EnumProperty(items=[('HEAD', 'Head', ''), ('EYE_LEFT', 'Left Eye', ''),
                              ('EYE_RIGHT', 'Right Eye', ''), ('TEETH', 'Teeth', ''),
                              ('BODY', 'Body', '')], default='HEAD')

    def run(self, context, s):
        from . import sculpt_parts
        sculpt_parts.start(context, s, self.part, detail=self.detail)


class CM_OT_finish_sculpt(SafeExecute, bpy.types.Operator):
    bl_idname = "cm.finish_sculpt"
    bl_label = "Finish Sculpt"
    bl_options = {"REGISTER", "UNDO"}

    def run(self, context, s):
        active = context.object
        if context.object and context.object.mode != "OBJECT":
            bpy.ops.object.mode_set(mode="OBJECT")
        from . import sculpt_parts
        sculpt_parts.finish(context, active)
        from . import wrinkles,wrinkle_jobs
        f=context.scene.cm_features
        if wrinkles.refresh_source_state(context,s,f):
            s.status='Head detail changed. Update retained wrinkles before baking or delivery.'
            if not bpy.app.background:
                wrinkle_jobs.start(context,s,f)


class CM_OT_detail(SafeExecute, bpy.types.Operator):
    bl_idname = "cm.create_detail"
    bl_label = "Create Scan Detail Layer"
    bl_options = {"REGISTER", "UNDO"}

    def run(self, context, s):
        if s.detail:
            raise ValueError(
                "A detail layer already exists. Clear its selector to create a separate new layer."
            )
        mesh_ops.create_detail(context, s)
        bpy.ops.cm.show_mesh(role="detail")
        s.status = (
            "Detail mesh created separately. Bake its normal onto the base result."
        )


class CM_OT_body(SafeExecute, bpy.types.Operator):
    bl_idname = "cm.shape_body"
    bl_label = "Update Body Proportions"
    bl_options = {"REGISTER", "UNDO"}

    def run(self, context, s):
        if s.part != "BODY":
            raise ValueError("Select Body as the template part first")
        mesh_ops.check_mesh(s.result, "Body result")
        obj = s.result
        keys = obj.data.shape_keys
        if not keys or "Wrapped" not in keys.key_blocks:
            raise ValueError("Wrap the body before shaping")
        base = np.asarray([v.co[:] for v in keys.key_blocks["Wrapped"].data])
        matrix = np.asarray(obj.matrix_world)
        base = base @ matrix[:3, :3].T + matrix[:3, 3]
        output = geometry.shape_body(
            base,
            s.body_height,
            s.body_width,
            s.body_depth,
            s.body_waist,
            s.body_shoulders,
            s.body_hips,
        )
        key = keys.key_blocks.get("Body Proportions") or obj.shape_key_add(
            name="Body Proportions"
        )
        key.relative_key = keys.key_blocks["Wrapped"]
        key.data.foreach_set("co", mesh_ops.to_local(obj, output).ravel())
        key.value = 1
        s.status = "Geometric body proportions updated from Wrapped baseline."


class CM_OT_bake(SafeExecute, bpy.types.Operator):
    bl_idname = "cm.bake"
    bl_label = "Bake Map"
    kind: EnumProperty(
        items=[
            ("ALBEDO", "Albedo", ""),
            ("NORMAL", "DirectX Normal", ""),
            ("AO", "Ambient Occlusion", ""),
        ]
    )

    def run(self, context, s):
        from . import textures,map_provenance,bake_jobs
        with textures.skin_part(s,context.scene.cm_features):
            path=mesh_ops.bake(context, s, self.kind)
            if self.kind=='NORMAL':map_provenance.record_single(context.scene,path,'NORMAL')
            bake_jobs.refresh_maps(context.scene,path.parent,[{'name':path.name}],kinds=(self.kind,))


class CM_OT_preview(SafeExecute, bpy.types.Operator):
    bl_idname = "cm.preview_material"
    bl_label = "Preview Baked Maps"
    bl_options = {"REGISTER", "UNDO"}
    paint: BoolProperty(default=False)

    def run(self, context, s):
        from .textures import preview_skin

        preview_skin(context, s, context.scene.cm_features)
        if self.paint:
            bpy.ops.object.mode_set(mode="TEXTURE_PAINT")
            s.status = "Standard Blender texture paint. Finish Sculpt/Paint, then Save Project to save your albedo."
        if context.area and context.area.type == "VIEW_3D":
            context.area.spaces.active.shading.type = "MATERIAL"


class CM_OT_save(SafeExecute, bpy.types.Operator):
    bl_idname = "cm.save_project"
    bl_label = "Save Project"

    def run(self, context, s):
        from . import project_images,skin_reconstruction
        from . import resource_budget
        root = Path(bpy.path.abspath(s.project_dir)).resolve()
        budget=resource_budget.project_save_storage(context.scene)
        resource_budget.require_disk('save character project',root,budget['required_bytes'])
        skin_reconstruction.restore_preview()
        working_images = project_images.working(context.scene)
        from . import wrinkles
        wrinkles.refresh_source_state(context,s,context.scene.cm_features)
        root.mkdir(parents=True, exist_ok=True)
        s.project_dir = str(root)
        f = context.scene.cm_features
        for obj, part in (
            (s.result, s.part),
            (f.head_result, "HEAD"),
            (f.body_result, "BODY"),
        ):
            if obj:
                obj["cm_part"] = part
                obj["cm_role"] = "RESULT"
        for image in bpy.data.images:
            dirty = image.is_dirty
            if dirty and image.packed_file:
                image.pack()
            if dirty and image in working_images and image.filepath and image.source == "FILE":
                image_path = Path(bpy.path.abspath(image.filepath)).resolve()
                if image_path.is_relative_to(root):
                    image.save()
        # A missing external texture must not prevent saving the character geometry.
        missing = []
        for image in bpy.data.images:
            if (
                not image.users
                or image.packed_file
                or image.source not in {"FILE", "GENERATED", "TILED"}
            ):
                continue
            if (
                image.source == "FILE"
                and not Path(bpy.path.abspath(image.filepath)).is_file()
            ):
                missing.append(image.name)
                continue
            try:
                image.pack()
            except RuntimeError:
                missing.append(image.name)
        bpy.ops.wm.save_as_mainfile(filepath=str(root / "project.blend"),compress=True)
        s.status = "Saved project.blend with available textures and landmark markers."
        if missing:
            s.status = f"Saved project; {len(missing)} missing/unpackable image(s). Check external texture paths."
            self.report({"WARNING"}, s.status)


class CM_OT_export(SafeExecute, bpy.types.Operator):
    bl_idname = "cm.export_bundle"
    bl_label = "Export Unreal Bundle"
    push: BoolProperty(default=False)

    def run(self, context, s):
        from .guided import check_delivery
        from . import unreal_jobs,groom_jobs,skin_native_jobs

        if unreal_jobs.running() or groom_jobs.running() or skin_native_jobs.running():
            raise ValueError('Wait for the current Unreal operation before replacing its export bundle')

        problems = check_delivery(context, s, context.scene.cm_features)
        if problems:
            raise ValueError(problems[0])
        if self.push:
            if bpy.ops.cm.save_project()!={'FINISHED'}:
                raise ValueError('Save the character project before pushing to Unreal')
            problems=check_delivery(context,s,context.scene.cm_features)
            if problems:
                raise ValueError(problems[0])
        root = mesh_ops.export_bundle(context, s)
        s.status = f"Exported {root}"
        if self.push:
            unreal_jobs.start(context, root)


class CM_OT_export_glb(SafeExecute, bpy.types.Operator):
    bl_idname = 'cm.export_static_glb'
    bl_label = 'Export Static GLB (Three.js)'
    bl_description = 'Export accepted geometry and embedded PBR skin images; animation and groom strands use separate workflows'

    def run(self, context, s):
        from . import web_export
        root = web_export.export(context)
        s.status = 'Static GLB exported: ' + str(root)


class CM_OT_demo(SafeExecute, bpy.types.Operator):
    bl_idname = "cm.create_demo"
    bl_label = "Create Synthetic Demo"
    bl_options = {"REGISTER", "UNDO"}

    def run(self, context, s):
        if context.object and context.object.mode != "OBJECT":
            bpy.ops.object.mode_set(mode="OBJECT")
        bpy.ops.cm.clear_landmarks()
        bpy.ops.mesh.primitive_uv_sphere_add(segments=32, ring_count=24)
        template = context.object
        template.name = "Demo_Template_NotMetaHuman"
        for vertex in template.data.vertices:
            vertex.co.x *= 0.08
            vertex.co.y *= 0.09
            vertex.co.z *= 0.12
        template.data.update()
        source = mesh_ops.duplicate(context, template, "Demo_Source")
        for vertex in source.data.vertices:
            x, y, z = vertex.co
            vertex.co.x *= 1.12
            vertex.co.z *= 1.08
            if y < 0:
                vertex.co.y -= 0.025 * np.exp(
                    -((x / 0.025) ** 2 + ((z - 0.015) / 0.035) ** 2)
                )
        source.data.update()
        source.data.materials.clear()
        material = bpy.data.materials.new("Demo_Source_Skin")
        material.use_nodes = True
        material.node_tree.nodes.get("Principled BSDF").inputs[
            "Base Color"
        ].default_value = (0.44, 0.19, 0.095, 1)
        source.data.materials.append(material)
        s.source, s.template = source, template
        context.scene.cm_features.wrap_method = "MANUAL"
        s.result, s.detail = None, None
        s.character_name = "DemoCharacter"
        s.create_character = False
        s.status = (
            "Synthetic demo ready. Click Wrap. This sphere is not MetaHuman topology."
        )
        from .guided import focus

        focus(context, [source], "-Y")


class CM_PT_base:
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "MetaHuman Maker"

    @classmethod
    def poll(cls, context):
        if cls.__name__ == "CM_PT_project":
            return True
        stages = {
            "CM_PT_import": "START",
            "CM_PT_wrap": "WRAP",
            "CM_PT_refine": "REFINE",
            "CM_PT_detail": "DETAIL",
            "CM_PT_bake": "TEXTURE",
            "CM_PT_export": "EXPORT",
        }
        return (
            context.scene.cm_features.advanced_ui
            and context.scene.cm_features.ui_stage == stages.get(cls.__name__)
        )


class CM_PT_project(CM_PT_base, bpy.types.Panel):
    bl_label = "MetaHuman Maker %d.%d" % bl_info['version'][:2]
    bl_idname = "CM_PT_project"

    def draw(self, context):
        from .guided import draw

        draw(self.layout, context)


class CM_PT_import(CM_PT_base, bpy.types.Panel):
    bl_label = "1 • Source and Template"
    bl_idname = "CM_PT_import"

    def draw(self, context):
        s, layout = context.scene.cm_settings, self.layout
        for role in ("source", "template"):
            box = layout.box()
            box.label(
                text="Your scan or generated mesh"
                if role == "source"
                else "MetaHuman LOD0 template"
            )
            box.prop(s, role)
            row = box.row(align=True)
            row.operator("cm.import_mesh", text="Import").role = role
            row.operator("cm.use_active", text="Use Active").role = role
        layout.prop(s, "part")
        if s.part == "FULL_BODY":
            layout.prop(context.scene.cm_features, "body_template")
        layout.label(text="Template: your licensed LOD0 surface.", icon="INFO")
        layout.operator("cm.prepare_template")
        layout.operator("cm.align", text="Rough Align by Bounds").bounds = True


class CM_PT_wrap(CM_PT_base, bpy.types.Panel):
    bl_label = "2 • Landmarks and Wrap"
    bl_idname = "CM_PT_wrap"

    def draw(self, context):
        s, layout = context.scene.cm_settings, self.layout
        if not s.source or not s.template:
            layout.label(text="Choose Source and Template in step 1.", icon="INFO")
            return
        layout.label(
            text=f"{sum(bool(p.source and p.template) for p in s.landmarks)} complete landmark pairs"
        )
        layout.operator("cm.landmarks", icon="EMPTY_AXIS")
        row = layout.row(align=True)
        row.operator("cm.landmark_file", text="Save Points").load = False
        row.operator("cm.landmark_file", text="Load Points").load = True
        layout.operator("cm.clear_landmarks")
        layout.operator("cm.align", text="Align Source from Landmarks")
        if context.scene.cm_features.advanced_ui:
            for prop in (
                "iterations",
                "projection_strength",
                "smoothing",
                "max_distance",
                "landmark_radius",
                "landmark_regularization",
            ):
                layout.prop(s, prop)
            layout.label(text="Protect vertices with group CM_Pin.", icon="INFO")
        layout.operator("cm.wrap", icon="MOD_SHRINKWRAP")


class CM_PT_refine(CM_PT_base, bpy.types.Panel):
    bl_label = "3 • Refine"
    bl_idname = "CM_PT_refine"

    def draw(self, context):
        s, layout = context.scene.cm_settings, self.layout
        if not s.result:
            layout.label(text="Wrap the Template in step 2 first.", icon="INFO")
            return
        layout.prop(s, "result")
        if s.result and s.result.data.shape_keys:
            for key in s.result.data.shape_keys.key_blocks:
                if key.name != "Basis":
                    layout.prop(key, "value", text=key.name)
        layout.operator("cm.sculpt")
        layout.operator("cm.finish_sculpt")
        if s.part == "BODY" and context.scene.cm_features.advanced_ui:
            for prop in (
                "body_height",
                "body_width",
                "body_depth",
                "body_waist",
                "body_shoulders",
                "body_hips",
            ):
                layout.prop(s, prop)
            layout.operator("cm.shape_body")


class CM_PT_detail(CM_PT_base, bpy.types.Panel):
    bl_label = "4 • Scan Detail"
    bl_idname = "CM_PT_detail"

    def draw(self, context):
        s, layout = context.scene.cm_settings, self.layout
        if not s.result:
            layout.label(text="Wrap the Template in step 2 first.", icon="INFO")
            return
        layout.prop(s, "detail_levels")
        layout.prop(s, "detail_strength")
        layout.operator("cm.create_detail")
        layout.prop(s, "detail")
        if s.detail and s.detail.data.shape_keys:
            key = s.detail.data.shape_keys.key_blocks.get("Scan Detail")
            if key:
                layout.prop(key, "value", text="Scan Detail")
        layout.operator("cm.sculpt", text="Sculpt Detail").detail = True
        layout.operator("cm.finish_sculpt")


class CM_PT_bake(CM_PT_base, bpy.types.Panel):
    bl_label = "5 • Bake Maps"
    bl_idname = "CM_PT_bake"

    def draw(self, context):
        s, layout = context.scene.cm_settings, self.layout
        if not s.result:
            layout.label(text="Wrap the Template in step 2 first.", icon="INFO")
            return
        layout.prop(s, "map_size")
        layout.prop(s, "bake_distance")
        layout.operator("cm.bake", text="Transfer Albedo").kind = "ALBEDO"
        layout.operator("cm.bake", text="Bake DirectX Normal").kind = "NORMAL"
        layout.operator("cm.bake", text="Bake Ambient Occlusion").kind = "AO"
        layout.operator("cm.preview_material")
        layout.operator("cm.preview_material", text="Paint Albedo").paint = True
        layout.operator("cm.finish_sculpt", text="Finish Sculpt / Paint")


class CM_PT_export(CM_PT_base, bpy.types.Panel):
    bl_label = "6 • Unreal 5.8"
    bl_idname = "CM_PT_export"

    def draw(self, context):
        s, layout = context.scene.cm_settings, self.layout
        if not s.result:
            layout.label(text="Wrap the Template in step 2 first.", icon="INFO")
            return
        layout.prop(s, "unreal_project")
        layout.prop(s, "create_character")
        if context.scene.cm_features.advanced_ui:
            for prop in ("unreal_engine_dir", "unreal_destination", "canonical_order"):
                layout.prop(s, prop)
        layout.operator("cm.export_bundle", icon="EXPORT")
        layout.operator(
            "cm.export_bundle", text="Push to Unreal", icon="PLAY"
        ).push = True


CLASSES = (
    CM_Landmark,
    CM_Settings,
    CM_OT_import,
    CM_OT_use_active,
    CM_OT_prepare_template,
    CM_OT_landmarks,
    CM_OT_remove_landmark,
    CM_OT_clear_landmarks,
    CM_OT_landmark_file,
    CM_OT_align,
    CM_OT_wrap,
    CM_OT_sculpt,
    CM_OT_finish_sculpt,
    CM_OT_detail,
    CM_OT_body,
    CM_OT_bake,
    CM_OT_save,
    CM_OT_export,
    CM_OT_export_glb,
    CM_OT_demo,
    CM_OT_preview,
    CM_OT_show_mesh,
    CM_PT_project,
    CM_PT_import,
    CM_PT_wrap,
    CM_PT_refine,
    CM_PT_detail,
    CM_PT_bake,
    CM_PT_export,
)


def register():
    for cls in CLASSES:
        bpy.utils.register_class(cls)
    bpy.types.Scene.cm_settings = PointerProperty(type=CM_Settings)
    from . import features, grooming, setup_paths, guided, workspace_layout, sculpt_parts, photo_fit_jobs, photo_review, photo_color, eye_appearance

    features.register()
    grooming.register()
    from . import reference_library
    reference_library.register()
    setup_paths.register()
    workspace_layout.register()
    for cls in guided.CLASSES:
        bpy.utils.register_class(cls)
    guided.register_menus()
    sculpt_parts.register()
    for cls in sculpt_parts.CLASSES:
        bpy.utils.register_class(cls)
    photo_fit_jobs.register()
    photo_review.register()
    from . import skin_reconstruction
    skin_reconstruction.register()
    from . import native_web_jobs
    native_web_jobs.register()
    from . import native_face_review
    native_face_review.register()
    for cls in (*photo_fit_jobs.CLASSES, *photo_review.CLASSES, *photo_color.CLASSES, *eye_appearance.CLASSES):
        bpy.utils.register_class(cls)


def unregister():
    from . import features, grooming, setup_paths, guided, workspace_layout, sculpt_parts, photo_fit_jobs, photo_review, photo_color, eye_appearance

    from . import skin_reconstruction
    from . import native_face_review
    native_face_review.unregister()
    from . import reference_library
    reference_library.unregister()
    from . import native_web_jobs
    native_web_jobs.unregister()
    skin_reconstruction.unregister()
    photo_review.unregister()
    photo_fit_jobs.unregister()
    for cls in reversed((*photo_fit_jobs.CLASSES, *photo_review.CLASSES, *photo_color.CLASSES, *eye_appearance.CLASSES)):
        bpy.utils.unregister_class(cls)
    sculpt_parts.unregister()
    for cls in reversed(sculpt_parts.CLASSES):
        bpy.utils.unregister_class(cls)
    workspace_layout.unregister()
    guided.unregister_menus()
    setup_paths.unregister()
    for cls in reversed(guided.CLASSES):
        bpy.utils.unregister_class(cls)
    grooming.unregister()
    features.unregister()
    del bpy.types.Scene.cm_settings
    for cls in reversed(CLASSES):
        bpy.utils.unregister_class(cls)
