"""Artist-facing workflow, view focus and actionable delivery checks."""

from pathlib import Path
import hashlib
import json
import re
import math
from datetime import datetime,timezone
import bpy
import numpy as np
from mathutils import Matrix, Vector
from bpy.props import StringProperty
from . import mesh_ops as m, setup_paths



def signature(obj):
    return hashlib.sha256(
        m.evaluated_coordinates(bpy.context, obj).tobytes()
    ).hexdigest()


def workspace():
    for item in setup_paths.candidates():
        model = Path(item.get("model_dir", ""))
        candidate = model.parent / "demo/MetaHumanHead/project.blend"
        if candidate.is_file():
            return candidate
    return None


def character_library():
    scene=getattr(bpy.context,'scene',None)
    features=getattr(scene,'cm_features',None)
    if features and getattr(features,'reference_library',''):
        selected=Path(bpy.path.abspath(features.reference_library))
        if selected.is_file() and selected.suffix.lower()=='.blend':return selected
        return None
    for item in setup_paths.candidates():
        if item.get('reference_library'):
            selected=Path(bpy.path.abspath(item['reference_library']))
            if selected.is_file() and selected.suffix.lower()=='.blend':return selected
        path = Path(item.get("model_dir", "")) / "local-character/character.blend"
        if path.is_file():
            return path
    return None


def load_body(context, s, f):
    from . import native_body
    if not f.is_property_set('body_backend') and native_body.available(f):f.body_backend='NATIVE'
    """Add a body to the existing head; keep the accepted head and its source."""
    from . import workflow, textures

    m.check_mesh(s.result, "Wrapped head")
    if s.part == "BODY" or f.body_result:
        raise ValueError(
            "This character already has a body. Shape it or load a custom body."
        )
    if f.full_sculpt or f.body_editing or context.mode != "OBJECT":
        raise ValueError("Finish the active sculpt or body pass before loading a body.")
    path = character_library()
    if not path:
        raise ValueError(
            "Prepare the local MetaHuman character library before loading its body."
        )
    with bpy.data.libraries.load(str(path), link=False) as (available, loaded):
        loaded.objects = available.objects
    reference = next(
        (o for o in loaded.objects if o and o.get("cm_library_role") == "body"), None
    )
    for obj in loaded.objects:
        if obj and obj != reference and not obj.users_collection:
            bpy.data.objects.remove(obj, do_unlink=True)
    m.check_mesh(reference, "Local body template")
    context.collection.objects.link(reference)
    reference.parent = None
    context.view_layer.update()
    head = s.result
    head_ring = workflow.neck_ring(head)
    body_ring = workflow.neck_ring(reference, False)
    h = m.evaluated_coordinates(context, head)[head_ring]
    b = m.coordinates(reference)[body_ring]
    front = reference.get("cm_front_axis", "-Y")
    angles = {"X": 0, "Y": math.pi / 2, "-X": math.pi, "-Y": -math.pi / 2}
    rotation = Matrix.Rotation(angles[f.front_axis] - angles[front], 4, "Z")
    reference.matrix_world = rotation @ reference.matrix_world
    context.view_layer.update()
    b = m.coordinates(reference)[body_ring]
    scale = 1 / context.scene.unit_settings.scale_length
    if f.proportional_scale:
        scale = np.linalg.norm(h - h.mean(0), axis=1).mean() / max(
            np.linalg.norm(b - b.mean(0), axis=1).mean(), 1e-8
        )
    transform = Matrix.Scale(float(scale), 4)
    transform.translation = Vector(h.mean(0) - b.mean(0) * scale)
    reference.matrix_world = transform @ reference.matrix_world
    context.view_layer.update()
    body = m.duplicate(context, reference, s.character_name + "_Body")
    body["cm_part"] = "BODY"
    body["cm_role"] = "RESULT"
    body["cm_body_source"] = "LIBRARY"
    body["cm_template_topology"] = m.topology(reference)
    body["cm_template_signature"] = signature(reference)
    f.body_template, f.body_result, f.head_result = reference, body, head
    s.part = "FULL_BODY"
    reference.hide_set(True)
    reference.hide_render = True
    textures.save_material_maps(s, body, "BODY")
    focus(context, [head, body, f.eye_left, f.eye_right, f.teeth], f.front_axis)
    s.status = "Body loaded and aligned. Attach Head to Body to close the neck seam."


def start_character(context, s, f):
    from . import character_project
    character_project.start(context, s, f, _start_character)


def _start_character(context, s, f):
    from . import native_body
    if not f.is_property_set('body_backend') and native_body.available(f):f.body_backend='NATIVE'
    from .features import ensure_controls

    ensure_controls(f)
    path = character_library()
    if not path:
        raise ValueError("Prepare a local textured MetaHuman character library first.")
    if f.body_editing or f.full_sculpt:
        raise ValueError(
            "Apply or cancel the current edit before starting another character."
        )
    with bpy.data.libraries.load(str(path), link=False) as (available, loaded):
        loaded.objects = available.objects
    from .reference_library import validate_objects
    roles = validate_objects(loaded.objects,require_body=True)
    for obj in loaded.objects:
        if obj and obj.get("cm_library_role"):
            context.collection.objects.link(obj)
    for role in ("head", "body"):
        m.check_mesh(roles.get(role), "Local " + role + " template")
    for obj in context.scene.objects:
        if obj not in roles.values():
            obj.hide_set(True)
            obj.hide_render = True
    s.landmarks.clear()
    s.template, f.body_template = roles["head"], roles["body"]
    s.part = "FULL_BODY"
    s.source, s.detail = None, None
    f.body_detail = None
    s.create_character, s.canonical_order = True, False
    f.front_axis = roles["head"].get("cm_front_axis", "-Y")
    for role, attribute in (("head", "head_result"), ("body", "body_result")):
        reference = roles[role]
        obj = m.duplicate(context, reference, s.character_name + "_" + role.title())
        obj.parent = None
        obj.matrix_world = reference.matrix_world.copy()
        obj["cm_part"] = role.upper()
        obj["cm_role"] = "RESULT"
        obj["cm_template_topology"] = m.topology(reference)
        obj["cm_template_signature"] = signature(reference)
        setattr(f, attribute, obj)
        reference.hide_set(True)
        reference.hide_render = True
    s.result = f.head_result
    for role in ("eye_left", "eye_right", "teeth"):
        setattr(f, role, roles.get(role))
    if f.wrap_influence != 1:
        f.wrap_influence = 1
    # Copy packed library maps into the character's own editable project folder.
    from . import textures

    for role, obj in (("HEAD", f.head_result), ("BODY", f.body_result)):
        textures.save_material_maps(s, obj, role)
    # Start from the selected matching skin, retaining the original as the
    # restorable previous appearance. Accepted head/body geometry stays intact.
    f.character_skin_applied = 'REFERENCE'
    f.character_applied_skin_image = None
    f.character_skin_previous = ''
    if f.character_skin_preset != 'REFERENCE':
        from . import character_skin
        character_skin.apply(context)
    f.ui_stage = "REFINE"
    s.stage = "REFINE"
    focus(
        context,
        [f.head_result, f.body_result, f.eye_left, f.eye_right, f.teeth],
        f.front_axis,
    )
    s.status = (
        "Textured character ready. Shape the body, sculpt, or continue to Unreal."
    )
    if not bpy.app.background and context.screen:
        for area in context.screen.areas:
            if area.type == 'VIEW_3D':
                area.spaces.active.shading.type = 'MATERIAL'
                area.spaces.active.show_region_ui = True
                sidebar = next((region for region in area.regions if region.type == 'UI'), None)
                if sidebar and not sidebar.is_property_readonly('active_panel_category'):
                    sidebar.active_panel_category = 'MetaHuman Maker'


def workspace_coverage(screen):
    from .workspace_layout import coverage
    return coverage(screen)


def ensure_workspace(context, callback=None):
    from . import workspace_layout
    return workspace_layout.start(context, callback or (lambda context: None))


def focus(context, objects, axis="-Y"):
    objects = [obj for obj in dict.fromkeys(objects) if obj and obj.name in context.scene.objects]
    if not objects or bpy.app.background or not context.window:
        return
    # Keep attached hair with its head when framing the character. Explicitly
    # hidden grooms remain hidden; source-only and unrelated mesh views keep
    # their own isolation. Live scalp references point back to the accepted head.
    from . import groom_follow
    anchors = set(objects)
    for obj in context.scene.objects:
        if obj.type != 'CURVES' or obj in anchors or obj.hide_get():
            continue
        surface = obj.data.surface
        if surface and (surface in anchors or groom_follow.source(surface) in anchors):
            objects.append(obj)
    if ensure_workspace(context, lambda ready: focus(ready, objects, axis)):
        return
    from . import compare_views
    window, scene = context.window, context.scene
    areas = [area for area in window.screen.areas if area.type == 'VIEW_3D']
    if not areas:
        return
    session = compare_views.recover(window)
    area = session['areas'][0] if session else max(areas, key=lambda area: area.width * area.height)
    region = next(r for r in area.regions if r.type == 'WINDOW')
    with context.temp_override(window=window, area=area, region=region):
        if context.object and context.object.mode != 'OBJECT':
            bpy.ops.object.mode_set(mode='OBJECT')
        space = area.spaces.active
        for obj in objects:
            obj.hide_set(False)
            if space.local_view:
                obj.local_view_set(space, True)
        bpy.ops.object.select_all(action='DESELECT')
        for obj in objects:
            obj.hide_set(False)
            obj.select_set(True)
        context.view_layer.objects.active = objects[0]
        if space.local_view:
            wanted = set(objects)
            for obj in context.view_layer.objects:
                obj.local_view_set(space, obj in wanted)
        else:
            bpy.ops.view3d.localview(frame_selected=False)
        space.show_region_ui = True
        sidebar = next((r for r in area.regions if r.type == 'UI'), None)
        if sidebar and not sidebar.is_property_readonly('active_panel_category'):
            sidebar.active_panel_category = 'MetaHuman Maker'
        space.overlay.show_floor = False
        space.overlay.show_relationship_lines = False
        compare_views.fit_objects(context, [area], objects, axis)
    # Sidebar and local-view dimensions settle after the current window event.
    # Refit only layout changes, and stop if the artist navigates the view.
    def raw_view():
        view = area.spaces.active.region_3d
        return (tuple(view.view_rotation), tuple(view.view_location), view.view_distance,
                view.view_perspective, area.spaces.active.lens)
    state = {'view': raw_view(), 'layout': compare_views.layout_state(area), 'quiet': 0, 'remaining': 6}
    workspace = window.workspace
    def settle():
        try:
            if window.scene != scene or window.workspace != workspace or area not in list(window.screen.areas):
                return None
            if not compare_views.same_view(state['view'], raw_view()):
                return None
            with bpy.context.temp_override(window=window, area=area, region=region):
                # A newly activated sidebar can expose its category only after
                # its first draw. Select the workflow again once it is writable.
                area.spaces.active.show_region_ui = True
                sidebar = next((r for r in area.regions if r.type == 'UI'), None)
                sidebar_ready = bool(sidebar and not sidebar.is_property_readonly('active_panel_category'))
                if sidebar_ready:
                    sidebar.active_panel_category = 'MetaHuman Maker'
                layout = compare_views.layout_state(area)
                if layout != state['layout']:
                    compare_views.fit_objects(bpy.context, [area], objects, axis)
                    state.update(view=raw_view(), layout=layout, quiet=0)
                else:
                    state['quiet'] += 1
            state['remaining'] -= 1
            return .1 if (state['quiet'] < 2 or not sidebar_ready) and state['remaining'] else None
        except (ReferenceError, RuntimeError, ValueError):
            return None
    bpy.app.timers.register(settle, first_interval=.1)


def focus_edit(context):
    """Frame an active edit without finishing its sculpt/paint session."""
    if context.mode == 'OBJECT' or not context.object or not context.window:
        return False
    from . import compare_views
    areas = [area for area in context.window.screen.areas if area.type == 'VIEW_3D']
    if not areas:
        return False
    area = max(areas, key=lambda area: area.width * area.height)
    region = next(region for region in area.regions if region.type == 'WINDOW')
    with context.temp_override(window=context.window, area=area, region=region):
        compare_views.fit_objects(context, [area], [context.object], context.scene.cm_features.front_axis)
    return True


def load_character_template(context, s, f, path, *, include_body=False):
    """Load the head and its matching internal meshes in scene units."""
    with bpy.data.libraries.load(str(path), link=False) as (available, loaded):
        loaded.objects = available.objects
    loaded_names = [obj.name for obj in loaded.objects if obj]
    saved = (s.template, f.front_axis, s.canonical_order, s.create_character, f.body_template)
    accessories = {role: getattr(f,role) for role in ('eye_left','eye_right','teeth')}
    visibility = [(obj, obj.hide_get(), obj.hide_render) for obj in (s.template, *accessories.values()) if obj]
    linked = []
    try:
        from .reference_library import validate_objects
        roles = validate_objects(loaded.objects,require_body=include_body)
        head = roles.get('head')
        m.check_mesh(head, 'Local head template')
        chosen = {'head': head}
        if include_body:
            m.check_mesh(roles.get('body'), 'Local body template')
            chosen['body'] = roles['body']
        if f.full_head:
            for role in ('eye_left', 'eye_right', 'teeth'):
                existing = getattr(f, role)
                if existing and not existing.get('cm_auto_accessory'):
                    continue  # Preserve the artist's explicitly assigned internal meshes.
                if roles.get(role):
                    m.check_mesh(roles[role], 'Local ' + role)
                    chosen[role] = roles[role]
        scale = Matrix.Scale(1 / context.scene.unit_settings.scale_length, 4)
        for role, obj in chosen.items():
            world = obj.matrix_world.copy()
            obj.parent = None
            obj.matrix_world = scale @ world
            context.collection.objects.link(obj)
            linked.append(obj)
            obj.hide_set(False)
            obj.hide_render = False
            if role not in {'head', 'body'}:
                obj['cm_auto_accessory'] = True
                obj['cm_accessory_template'] = head
                if 'cm_follow_original' in obj:
                    del obj['cm_follow_original']
        previous = s.template
        s.template = head
        head.name = 'MetaHuman_Head_Template'
        head['cm_template_library'] = str(path)
        f.front_axis = head.get('cm_front_axis', '-Y')
        s.canonical_order, s.create_character = False, True
        if previous and previous != head and previous.get('cm_template_library'):
            previous.hide_set(True)
            previous.hide_render = True
        for role, obj in chosen.items():
            if role == 'head':
                continue
            if role == 'body':
                f.body_template = obj
                obj.name = 'MetaHuman_Body_Template'
                obj['cm_template_library'] = str(path)
                continue
            previous = getattr(f, role)
            setattr(f, role, obj)
            if previous and previous != obj and previous.get('cm_auto_accessory'):
                previous.hide_set(True)
                previous.hide_render = True
        if not f.full_head:
            for role, previous in accessories.items():
                if previous and previous.get('cm_auto_accessory'):
                    setattr(f,role,None)
                    previous.hide_set(True)
                    previous.hide_render = True
        context.view_layer.update()
        if s.source:
            focus(context, [s.source], f.front_axis)
        else:
            focus(context, list(chosen.values()), f.front_axis)
        missing = [role for role in ('eye_left','eye_right','teeth') if f.full_head and not getattr(f,role)]
        s.status = ('Head template loaded. Missing local internal meshes: ' + ', '.join(missing)) if missing else 'Matching head template, eyes and teeth loaded.' if f.full_head else 'Head template loaded.'
    except Exception:
        s.template, f.front_axis, s.canonical_order, s.create_character, f.body_template = saved
        for role, obj in accessories.items():
            setattr(f, role, obj)
        for obj, hidden, render_hidden in visibility:
            obj.hide_set(hidden)
            obj.hide_render = render_hidden
        for obj in linked:
            bpy.data.objects.remove(obj, do_unlink=True)
        raise
    finally:
        for name in loaded_names:
            obj = bpy.data.objects.get(name)
            if obj and not obj.users_collection:
                bpy.data.objects.remove(obj, do_unlink=True)


def load_template(context, s, f):
    library = character_library()
    if library:
        return load_character_template(context, s, f, library)
    path = workspace()
    if not path:
        raise ValueError(
            "No local template library found. Import your licensed MetaHuman LOD0 head first."
        )
    with bpy.data.libraries.load(str(path), link=False) as (source, target):
        target.scenes = source.scenes[:1]
    imported = target.scenes[0]
    try:
        original = imported.cm_settings.template
        m.check_mesh(original, "Saved head template")
        obj = m.duplicate(context, original, "MetaHuman_Head_Template")
        world = original.matrix_world.copy()
        # This local preset was exported from Epic's centimeter-space reference.
        # Some FBX exports leave that scale in vertex coordinates rather than units.
        unit_scale = (
            imported.unit_settings.scale_length
            / context.scene.unit_settings.scale_length
        )
        physical_height = (
            np.ptp(m.coordinates(original), axis=0)[2]
            * imported.unit_settings.scale_length
        )
        if physical_height > 2:
            unit_scale *= 0.01
        world = Matrix.Scale(unit_scale, 4) @ world
        obj.parent = None
        obj.matrix_world = world
        front = original.get("cm_front_axis", imported.cm_features.front_axis)
        obj["cm_front_axis"] = front
        obj["cm_template_library"] = str(path)
        s.template = obj
        f.front_axis = front
        s.canonical_order = False
        s.create_character = True
    finally:
        bpy.data.scenes.remove(imported)
    s.status = "MetaHuman head template loaded. Add a source, then continue to Wrap."
    focus(context, [s.source or s.template], f.front_axis)


def align_for_detection(context, s, f):
    from .features import generated_head_update, head_source_scope
    if s.part=='FULL_BODY' and generated_head_update(s):
        with head_source_scope(s):
            return align_for_detection(context,s,f)
    m.check_inputs(s)
    if s.part == "FULL_BODY":
        if not f.body_template:
            raise ValueError(
                "Choose the Body Template before detecting full-body landmarks."
            )
    source_axis = s.source.get("cm_front_axis", f.source_front_axis)
    if source_axis == "AUTO":
        source_axis = f.front_axis
    angles = {"X": 0, "Y": math.pi / 2, "-X": math.pi, "-Y": -math.pi / 2}
    delta = angles[f.front_axis] - angles[source_axis]
    s.source.matrix_world = Matrix.Rotation(delta, 4, "Z") @ s.source.matrix_world
    s.source["cm_front_axis"] = f.front_axis
    bpy.ops.cm.clear_landmarks()
    if s.part == "FULL_BODY":
        target = np.concatenate(
            [m.coordinates(s.template), m.coordinates(f.body_template)]
        )
        source = m.coordinates(s.source)
        scale = np.ptp(target, axis=0).max() / max(np.ptp(source, axis=0).max(), 1e-9)
        transform = Matrix.Scale(float(scale), 4)
        transform.translation = Vector(
            (target.min(0) + target.max(0)) * 0.5
            - (source.min(0) + source.max(0)) * 0.5 * scale
        )
        s.source.matrix_world = transform @ s.source.matrix_world
    else:
        m.align_source(context, s, bounds=True)
    context.view_layer.update()


def check_delivery(context, s, f):
    problems = []
    if any(obj and not obj.get('cm_scan_applied', True) for obj in (s.detail, f.body_detail if s.part == 'FULL_BODY' else None)):
        problems.append('Apply or clear the pending scan displacement before delivery.')
    from . import wrinkles
    try:wrinkles.refresh_source_state(context,s,f)
    except (ValueError,RuntimeError) as exc:problems.append('Check retained wrinkle sources: '+str(exc))
    from . import map_provenance
    try:problems.extend(map_provenance.problems(context.scene))
    except (OSError,ValueError,KeyError) as exc:problems.append('Check the map bake record before delivery: '+str(exc))
    try:
        from . import skin_projection_history
        problems.extend(skin_projection_history.problems(context.scene))
    except (OSError,ValueError,KeyError) as exc:problems.append('Check source skin projection history: '+str(exc))
    if s.detail and s.detail.get('cm_head_update_pending'):
        problems.append('Update the retained head sculpt to the new face before delivery. Use Update Retained Sculpt in Refine to resume.')
    if any(obj and obj.get('cm_wrinkle_update_pending') for obj in (f.wm1,f.wm2,f.wm3)):
        problems.append('Update the retained wrinkle sculpts before delivery. Choose Update Retained Wrinkles to resume.')
    result = f.head_result if s.part == "FULL_BODY" else s.result
    template = s.template
    if not result or not template:
        return ["Wrap your Source and MetaHuman Template before delivery."]
    if f.full_sculpt:
        problems.append("Apply or cancel the combined sculpt session.")
    if f.body_editing:
        problems.append("Apply or cancel the active body pass.")
    if f.restore_editing:
        problems.append("Apply or cancel the active head repair.")
    targets = [(result, template)]
    if s.part == "FULL_BODY":
        targets.append((f.body_result, f.body_template))
    for obj, reference in targets:
        if not obj or not reference:
            problems.append("The full-body result needs both head and body meshes.")
            continue
        expected_part = (
            "BODY"
            if s.part == "FULL_BODY" and reference == f.body_template
            else "HEAD"
            if s.part == "FULL_BODY"
            else s.part
        )
        if obj.get("cm_part", expected_part) != expected_part:
            problems.append(
                f"{obj.name}: choose the matching character part or wrap its template again."
            )
        if m.topology(obj) != m.topology(reference):
            problems.append(
                f"{obj.name}: topology changed; restore the original conform topology."
            )
        if obj.get("cm_template_signature") and obj[
            "cm_template_signature"
        ] != signature(reference):
            problems.append(
                "Template changed since wrapping; wrap again before delivery."
            )
        if not np.isfinite(m.evaluated_coordinates(context, obj)).all():
            problems.append(f"{obj.name}: invalid vertex coordinates.")
        elif len(obj.data.vertices)==len(reference.data.vertices):
            error=m.geometry.split_vertex_error(m.coordinates(reference),m.evaluated_coordinates(context,obj))
            if error*context.scene.unit_settings.scale_length > 1e-5:
                problems.append(f'{obj.name}: separated UV seam vertices; wrap again or restore physical seam continuity before delivery.')
            collapsed=m.collapsed_faces(context,obj)
            if collapsed:
                problems.append(f'{obj.name}: {len(collapsed)} collapsed faces below the Unreal area threshold. Use Inspect Collapsed Faces, repair the surface, then check again. Preserve topology.')
        if s.create_character and not obj.data.uv_layers and not s.canonical_order:
            problems.append(
                f"{obj.name}: preserve MetaHuman LOD0 UVs for character conform."
            )
        if (
            s.create_character
            and not s.canonical_order
            and obj.data.uv_layers
            and reference.data.uv_layers
        ):
            actual_uv = np.empty(
                len(obj.data.uv_layers.active.data) * 2, dtype=np.float32
            )
            reference_uv = np.empty(
                len(reference.data.uv_layers.active.data) * 2, dtype=np.float32
            )
            obj.data.uv_layers.active.data.foreach_get("uv", actual_uv)
            reference.data.uv_layers.active.data.foreach_get("uv", reference_uv)
            if not np.array_equal(actual_uv, reference_uv):
                problems.append(
                    f"{obj.name}: UVs changed; restore the original MetaHuman LOD0 layout."
                )
        keys = obj.data.shape_keys
        if (
            keys
            and keys.key_blocks.get("Expression Check")
            and keys.key_blocks["Expression Check"].value
        ):
            problems.append("Return the expression preview to Neutral.")
        if obj.get("cm_source_signature") and (
            not s.source or obj["cm_source_signature"] != signature(s.source)
        ):
            problems.append(
                "Source changed since wrapping; wrap again before delivery."
            )
    return list(dict.fromkeys(problems))


def project_folder(s):
    if (
        s.project_dir in {"//CreateMH_Project/", "//CreateMH_Project", ""}
        and not bpy.data.filepath
    ):
        name = re.sub(r"[^\w .-]", "_", s.character_name).strip(" .") or "Character"
        s.project_dir = str(Path.home() / "Documents/CreateMetaHuman/Projects" / name)


class CM_OT_guided(bpy.types.Operator):
    bl_idname = "cm.guided"
    bl_label = "Character Workflow"
    bl_options = {"REGISTER", "UNDO"}
    action: StringProperty()

    @classmethod
    def description(cls, context, properties):
        if properties.action == 'UPDATE_UNREAL_TEXTURES':
            return 'Refresh existing delivered Unreal texture assets in place. Blender Undo does not revert Unreal imports; reassemble previously built copies afterward'
        if properties.action == 'DOWNLOAD_NATIVE_SOURCES':
            return 'Download the linked character\'s configured native texture sources. Tracks one UE request through completion and reopening Blender; requires Epic service access and saved UE edits. Blender Undo does not reverse native downloads'
        if properties.action == 'REQUEST_FULL_RIG':
            return 'Create the linked MetaHuman full rig in Unreal. Requires Epic service access and saved UE edits. Observes an existing request or completed rig; Blender Undo does not reverse native rigging'
        return 'Run the selected character workflow step'

    def execute(self, context):
        s, f = context.scene.cm_settings, context.scene.cm_features
        try:
            if self.action=='CHECK_ASSEMBLY_RESOURCES':
                from . import unreal_assembly
                project=Path(bpy.path.abspath(f.unreal_character_project)).resolve()
                if project.suffix.lower()!='.uproject' or not project.is_file():
                    raise ValueError('Link the character to its existing Unreal project before checking assembly resources')
                content=project.parent/'Content'
                if not content.is_dir():raise ValueError('The linked Unreal project has no Content folder')
                estimate=unreal_assembly.resource_status(content,False)
                estimate.update(project=str(project),scope='LOCAL_PRELAUNCH',checked_at=datetime.now(timezone.utc).isoformat())
                f.unreal_resource_report=json.dumps(estimate)
                s.status='Local memory and disk estimate updated. Unreal will recheck resources after loading.'
                return {'FINISHED'}
            from . import paint_tools
            from . import landmark_tools
            landmark_tools.finish_active(context)
            from . import scan_jobs
            if scan_jobs.running() and self.action != 'SAVE':
                raise ValueError('Finish or cancel scan projection before another workflow action')
            from . import auto_wrap_jobs
            if auto_wrap_jobs.running() and self.action != 'SAVE':
                raise ValueError('Finish or cancel Auto Wrap before another workflow action')
            from . import detail_mask
            if detail_mask.running(context.scene) and self.action != 'SAVE':
                raise ValueError('Apply or cancel Scan Mask before another workflow action')
            if paint_tools.running():
                raise ValueError('Apply or cancel Smart Paint before another workflow action')
            from . import skin_reconstruction
            if skin_reconstruction.running() and self.action!='SAVE':
                raise ValueError('Apply or discard skin reconstruction before another workflow action')
            project_folder(s)
            if self.action == "TEMPLATE":
                load_template(context, s, f)
            elif self.action in {"GENERATE", "GENERATE_UPDATE"}:
                from . import head_model

                setup_paths.resolve(f)
                if f.generated_head and self.action == "GENERATE":
                    f.identity_seed += 1
                head_model.create_head(
                    context, s, f, randomize=self.action == "GENERATE"
                )
                focus(context, [s.source], s.source.get("cm_front_axis", "-Y"))
            elif self.action == "CHARACTER":
                start_character(context, s, f)
            elif self.action == "LOAD_BODY":
                load_body(context, s, f)
            elif self.action == "ATTACH_BODY":
                from .workflow import attach_neck

                attach_neck(context, s, f)
                focus(
                    context,
                    [s.result, f.body_result, f.eye_left, f.eye_right, f.teeth],
                    f.front_axis,
                )
            elif self.action == "DETECT":
                if bpy.app.background:
                    from .automation import align_detect
                    align_detect(context,s,f)
                else:
                    from . import auto_wrap_jobs
                    auto_wrap_jobs.start(context,wrap=False)
            elif self.action == 'RETRY_HEAD_DETAIL':
                from . import detail_jobs
                if not detail_jobs.start(context,s,f,'SYNC_HEAD'):
                    if s.detail and 'cm_head_update_pending' in s.detail:
                        del s.detail['cm_head_update_pending']
                    s.status='Head sculpt already follows the accepted face.'
            elif self.action == 'RETRY_WRINKLES':
                from . import wrinkle_jobs
                wrinkle_jobs.start(context,s,f)
            elif self.action == "WRAP":
                if s.source and s.source.get('cm_source_alignment')=='METAHUMAN':
                    return bpy.ops.cm.conform_native_surface()
                from .features import generated_head_update
                head_update=generated_head_update(s)
                if f.wrap_method == "AUTO":
                    from . import auto_wrap_jobs
                    if auto_wrap_jobs.needs_detection(context):
                        if bpy.app.background:
                            from .automation import align_detect
                            align_detect(context,s,f)
                        else:
                            auto_wrap_jobs.start(context,wrap=True)
                            return {'FINISHED'}
                if f.body_editing or f.full_sculpt:
                    raise ValueError(
                        "Finish the current body/sculpt session before wrapping again."
                    )
                if bpy.ops.cm.wrap(defer_wrinkles=True) != {"FINISHED"}:
                    return {"CANCELLED"}
                from .detail_layers import REFERENCE
                managed_detail=bool(s.detail and (REFERENCE in s.detail.data.attributes or 'cm_detail_level' in s.detail))
                if s.detail and (not head_update or not managed_detail):
                    s.detail.hide_set(True)
                    s.detail.hide_render = True
                    s.detail = None
                if f.body_detail and not head_update:
                    f.body_detail.hide_set(True)
                    f.body_detail.hide_render = True
                    f.body_detail = None
                for field in ("wm1", "wm2", "wm3"):
                    previous = getattr(f, field)
                    if previous:
                        previous.hide_set(True)
                        previous.hide_render = True
                        for child in previous.children:
                            child.hide_set(True)
                            child.hide_render=True
                        if not head_update:
                            setattr(f, field, None)
                f.ui_stage = "REFINE"
                focus(
                    context,
                    [s.result, f.body_result if s.part == "FULL_BODY" else None]
                    + ([f.eye_left,f.eye_right,f.teeth] if f.full_head and s.part != 'BODY' else []),
                    f.front_axis,
                )
                s.status = "Wrap complete. Inspect the silhouette, eyelids and mouth; sculpt if needed."
                if head_update:
                    from . import detail_jobs
                    if detail_jobs.running():
                        s.status='Face updated. Your head sculpt is following the new shape in the background.'
                    elif any(obj and obj.get('cm_wrinkle_update_pending') for obj in (f.wm1,f.wm2,f.wm3)):
                        from . import wrinkle_jobs
                        try:wrinkle_jobs.start(context,s,f)
                        except Exception as exc:s.status='Face updated; wrinkle sculpts retained. Retry their update: '+str(exc)
            elif self.action == "FOCUS":
                if focus_edit(context):
                    return {'FINISHED'}
                objects = [s.result or s.source or s.template]
                if f.ui_stage == 'GROOM' and any((context.scene.cm_groom.groom,context.scene.cm_groom.main_scalp,context.scene.cm_groom.scalp)):
                    g=context.scene.cm_groom
                    objects=[g.main_scalp or s.result or g.scalp,g.groom]
                elif s.part == "FULL_BODY" and s.result:
                    objects += [f.body_result, f.eye_left, f.eye_right, f.teeth]
                elif s.result and s.part == 'HEAD' and f.full_head:
                    objects += [f.eye_left, f.eye_right, f.teeth]
                focus(context, objects, f.front_axis)
            elif self.action in {
                "VIEW_SOURCE",
                "VIEW_TEMPLATE",
                "VIEW_RESULT",
                "VIEW_BODY",
            }:
                role = self.action.removeprefix("VIEW_").lower()
                obj = f.body_result if role == "body" else getattr(s, role)
                markers = (
                    [getattr(p, role) for p in s.landmarks]
                    if role in {"source", "template"}
                    else []
                )
                objects = [obj] + markers
                if role == "result" and s.part == "FULL_BODY":
                    objects += [f.body_result, f.eye_left, f.eye_right, f.teeth]
                elif role == 'result' and s.part == 'HEAD' and f.full_head:
                    objects += [f.eye_left,f.eye_right,f.teeth]
                focus(context, objects, f.front_axis)
            elif self.action == "ALIGN":
                align_for_detection(context, s, f)
                focus(context, [s.source], f.front_axis)
                s.status = "Source roughly aligned. Place corresponding landmark pairs."
            elif self.action=='INSPECT_COLLAPSED':
                if context.mode!='OBJECT' or f.expression_active or f.full_sculpt or f.body_editing or f.restore_editing:
                    raise ValueError('Return to Neutral and finish the active edit before inspecting delivery geometry')
                targets=[f.head_result or s.result,f.body_result] if s.part=='FULL_BODY' else [s.result]
                affected=[(obj,m.collapsed_faces(context,obj)) for obj in targets if obj]
                affected=[(obj,faces) for obj,faces in affected if faces]
                if not affected:raise ValueError('No collapsed faces remain. Check Character again to refresh delivery status.')
                obj,faces=affected[0]
                faces=m.collapsed_faces(context,obj,require_base_topology=True)
                focus(context,[obj],f.front_axis);m.activate(context,obj)
                for vertex in obj.data.vertices:vertex.select=False
                for edge in obj.data.edges:edge.select=False
                selected=set(faces)
                for polygon in obj.data.polygons:polygon.select=polygon.index in selected
                context.tool_settings.mesh_select_mode=(False,False,True)
                bpy.ops.object.mode_set(mode='EDIT')
                s.status=f'{obj.name}: selected {len(faces)} collapsed faces. Repair without deleting geometry, return to Object Mode, then Check Character.'
            elif self.action in {"EXPORT", "PUSH", "CHECK"}:
                problems = check_delivery(context, s, f)
                f.delivery_report = (
                    "\n".join(problems) if problems else "Character checks passed."
                )
                if problems:
                    raise ValueError(problems[0])
                if self.action != "CHECK":
                    if bpy.ops.cm.export_bundle(push=self.action == "PUSH") != {
                        "FINISHED"
                    }:
                        return {"CANCELLED"}
                else:
                    s.status = (
                        "Character checks passed. Ready to export or push to Unreal."
                    )
            elif self.action in {'OPEN_CREATOR','REFRESH_UNREAL','BUILD_UNREAL'}:
                from . import unreal_jobs
                unreal_jobs.character_action(context,open_editor=self.action=='OPEN_CREATOR',build=self.action=='BUILD_UNREAL')
            elif self.action=='CAPTURE_UNREAL_ARTIST':
                from . import unreal_jobs
                unreal_jobs.character_action(context,artist_capture=True)
            elif self.action=='REVIEW_UNREAL_ARTIST':
                if not f.unreal_artist_review:raise ValueError('Capture saved UE artist edits first')
                review=json.loads(f.unreal_artist_review)
                document=bpy.data.texts.get('UE Artist Review') or bpy.data.texts.new('UE Artist Review')
                document.clear();document.write(json.dumps(review,indent=2))
                def draw_review(menu,_context):
                    menu.layout.label(text=str(review['objects'])+' preserved objects; '+str(review['native_files'])+' native files')
                    menu.layout.label(text=str(len(review['property_changes']))+' property changes; '+str(len(review['native_file_changes']))+' file changes')
                    menu.layout.label(text='Full details: UE Artist Review in Blender Text Editor')
                    menu.layout.label(text='Native edits preserved. Visual application remains separate.')
                context.window_manager.popup_menu(draw_review,title='Captured UE Artist Edits',icon='INFO')
                s.status='Review available in Blender text: UE Artist Review'
            elif self.action == 'UPDATE_UNREAL_TEXTURES':
                from . import unreal_jobs
                unreal_jobs.update_textures(context)
            elif self.action == 'APPLY_NATIVE_ALBEDO':
                from . import unreal_jobs
                unreal_jobs.update_textures(context, native_albedo=True)
            elif self.action == 'DOWNLOAD_NATIVE_SOURCES':
                from . import unreal_jobs
                unreal_jobs.character_action(context,texture_sources=True)
            elif self.action == 'REQUEST_FULL_RIG':
                from . import rig_jobs
                rig_jobs.start(context)
            elif self.action=='USE_BUILD_FACE':
                from . import unreal_jobs
                unreal_jobs.use_build_face(context.scene)
                s.status='Groom binding target set to the assembled Face mesh. Continue in Hair.'
            elif self.action == "BAKE_SKIN":
                from .textures import bake_skin
                if bpy.app.background:bake_skin(context,s,f)
                else:
                    from . import bake_jobs
                    bake_jobs.start(context,s,f,'SKIN')
            elif self.action == "SAVE":
                if bpy.ops.cm.save_project() != {"FINISHED"}:
                    return {"CANCELLED"}
            else:
                raise ValueError("Unknown workflow action")
            f.last_action_error = False
            return {"FINISHED"}
        except Exception as exc:
            f.last_action_error = True
            s.status = str(exc)
            self.report({"ERROR"}, str(exc))
            return {"CANCELLED"}


def button(layout, label, action, enabled=True, icon="NONE"):
    row = layout.row()
    row.enabled = enabled
    row.operator("cm.guided", text=label, icon=icon).action = action


def text(layout, message, width=34):
    import textwrap

    for line in textwrap.wrap(message, width=width):
        layout.label(text=line)


def draw(layout, context):
    s, f = context.scene.cm_settings, context.scene.cm_features
    from . import photo_review
    if photo_review._VIEW and photo_review.draw(layout, context):
        return
    from . import photo_fit_jobs
    if photo_fit_jobs.draw(layout, context):
        return
    from . import scan_jobs
    if scan_jobs.draw(layout, context):
        return
    from . import detail_mask
    if detail_mask.draw(layout, context):
        return
    from . import paint_tools
    if paint_tools.draw(layout, context):
        return
    from . import compare_views
    if compare_views.draw(layout, context, source_only=True):
        return
    layout.use_property_split = False
    from . import region_projection
    if region_projection.running():
        box=layout.box();box.label(text='Regional Surface Fit')
        box.operator('cm.cancel_region_projection',text='Cancel Fit',icon='CANCEL')
        text(box,s.status or 'Fitting the region; current geometry is retained until validation finishes.')
        return
    from . import expressions
    expressions.draw_preparation(layout)
    from . import auto_wrap_jobs
    if auto_wrap_jobs.running():
        box=layout.box()
        text(box,'Detecting landmarks for Auto Wrap...')
        box.operator('cm.cancel_auto_wrap',icon='CANCEL')
        return
    from . import bake_jobs, detail_jobs, unreal_jobs, skin_library, wrinkle_jobs, groom_jobs, skin_native_jobs
    pending_wrinkles=any(obj and obj.get('cm_wrinkle_update_pending') for obj in (f.wm1,f.wm2,f.wm3))
    if wrinkle_jobs.running():
        box=layout.box()
        box.operator('cm.cancel_wrinkle_update',icon='CANCEL')
        text(box,'Updating retained wrinkle sculpts...')
    elif pending_wrinkles and not detail_jobs.running() and not (s.detail and s.detail.get('cm_head_update_pending')):
        button(layout,'Update Retained Wrinkles','RETRY_WRINKLES',icon='FILE_REFRESH')
    if skin_library.running() or skin_native_jobs.running():
        box = layout.box()
        box.operator('cm.cancel_local_skin', icon='CANCEL')
        text(box, f.skin_library_status or 'Preparing local skin...')
    if unreal_jobs.running():
        box = layout.box()
        box.label(text='Push to Unreal', icon='EXPORT')
        text(box, f.unreal_push_status or 'Monitoring the original Unreal push...')
    if groom_jobs.running():
        box=layout.box()
        box.label(text='Unreal Groom Exchange',icon='EXPORT')
        text(box,context.scene.cm_groom.exchange_status)
    if detail_jobs.running():
        box=layout.box()
        box.operator('cm.cancel_detail_update',icon='CANCEL')
        text(box,f.detail_update_status or 'Updating sculpt detail...')
    if bake_jobs.running():
        box=layout.box()
        box.operator('cm.cancel_map_bake',icon='CANCEL')
        text(box,f.map_bake_status or 'Background map bake running...')
    if f.restore_editing:
        text(layout,"Repairing the neutral head. Paint on the model, then Apply or Cancel.")
        row=layout.row(align=True)
        row.operator("cm.finish_repair",text="Apply Repair",icon="CHECKMARK").accept=True
        row.operator("cm.finish_repair",text="Cancel",icon="X").accept=False
        layout.prop(f,"restore_radius",text="Size")
        layout.prop(f,"restore_strength",text="Strength")
        if f.restore_mode=='SURFACE':layout.prop(f,'restore_source_adherence')
        layout.prop(f,"mirror",text="Mirror")
        text(layout,"Ctrl+Z undoes a stroke. Enter applies; Esc cancels.")
        return
    from . import skin_brushes
    if skin_brushes.draw_active(layout,context):return
    if f.sculpt_session_role=='AUTHOR':
        from . import sculpt_session
        box=layout.box()
        box.label(text='Separate Sculpting Session',icon='SCULPTMODE_HLT')
        text(box,'Only maps and LookDev settings return to the main editor.')
        try:ready=(sculpt_session.folder_for(context.scene)/'receipts.json').is_file()
        except (ValueError,OSError):ready=False
        row=box.row()
        row.enabled=ready and not bake_jobs.running() and not detail_jobs.running() and context.mode=='OBJECT' and not sculpt_session.pending()
        row.operator('cm.sculpt_session',text='Save & Return',icon='BACK').action='RETURN'
        if not ready:text(box,'Create Final Maps before returning.')
        if f.sculpt_session_status:text(box,f.sculpt_session_status)
        stage='DETAIL'
    else:
        step=layout.row()
        selector=step.row()
        selector.enabled=context.mode=='OBJECT' and not (f.body_editing or f.full_sculpt or f.restore_editing or f.expression_active)
        selector.prop(f, "ui_stage", text="Step")
        step.operator('cm.focus_character', text='', icon='VIEWZOOM')
        stage=f.ui_stage
    if not (stage == 'START' and f.input_mode == 'GENERATE'):
        photo_review.draw(layout, context)
    if stage in {'START','WRAP','REFINE','TEXTURE','DETAIL'} and s.source and (s.result or s.template):
        if not compare_views.draw(layout, context):
            layout.operator('cm.dual_views',text='Compare Source / Result',icon='VIEW3D').action='OPEN'
    if stage in {'REFINE','GROOM','EXPORT'}:
        from . import native_face_review
        native_face_review.draw(layout,context)
    if stage in {"REFINE", "TEXTURE", "DETAIL", "EXPORT"} and not s.result:
        text(layout, "Wrap your Source and Template in step 2 before continuing.")
        return
    if stage == "START":
        layout.prop(s, "character_name")
        layout.prop(s, "project_dir", text="Save Folder")
        layout.prop(f, "input_mode", expand=True)
        from . import reference_library
        reference_library.draw(layout,context)
        box = layout.box()
        if f.input_mode == "CHARACTER":
            from . import character_skin
            character_skin.draw(box, context, apply_button=False)
            button(
                box,
                "Start Textured Full Character",
                "CHARACTER",
                bool(character_library()),
                icon="USER",
            )
            text(box, "Uses your local MetaHuman head, body and skin textures.")
            return
        elif f.input_mode == "GENERATE":
            photo = box.box()
            photo.label(text='Photo to Starting Shape')
            photo.operator('cm.fit_photo', text='Fit Starting Face from Photo', icon='IMAGE_DATA')
            text(photo, 'Use a near-frontal neutral photo. Fits starting geometry; skin and hair are separate.')
            if f.photo_fit_status:
                text(photo, f.photo_fit_status)
            if f.generated_head and f.generated_head.get('cm_photo_fit_report'):
                photo_review.draw(photo, context)
            box.label(text='Random Generation')
            box.prop(f, "identity_seed", text="Seed")
            box.prop(f, "gender", text="Gender")
            box.prop(f, "ethnicity", text="Group")
            button(
                box,
                "Generate Another Face" if f.generated_head else "Generate Face",
                "GENERATE",
                icon="USER",
            )
            box.operator("cm.face_mixer", text="Face Mixer Wheel")
            if f.generated_head:
                from . import head_model
                head_model.draw_skin(box, f)
                box.prop(f, "show_face_features", toggle=True)
                if f.show_face_features:
                    for item in f.face_controls:
                        box.prop(item, "value", text=item.label)
                    button(box, "Apply Features", "GENERATE_UPDATE")
        else:
            operator = box.operator(
                "cm.import_mesh", text="Import New Character" if s.result else "Import Character Mesh", icon="IMPORT"
            )
            operator.role = "source"
            operator.prepare_workflow = True
            if s.result:
                operator = box.operator('cm.import_mesh', text='Replace Current Source')
                operator.role = 'source'
                operator.prepare_workflow = False
        if s.source:
            box.prop(s, "source", text="Source")
        else:
            box.label(text="No source selected", icon="INFO")
        box = layout.box()
        if s.template:
            box.prop(s, "template", text="Template")
        elif workspace() and s.part != "BODY":
            button(box, "Load Local MetaHuman Head", "TEMPLATE", icon="MESH_MONKEY")
        else:
            box.operator(
                "cm.import_mesh", text="Import MetaHuman Template"
            ).role = "template"
        if not s.template:
            text(box, "Uses your licensed MetaHuman LOD0 mesh.")
        layout.prop(s, "part", text="Character")
        if s.part == "FULL_BODY":
            layout.prop(f, "body_template")
            if not f.body_template:
                layout.operator(
                    "cm.import_mesh", text="Import MetaHuman Body Template"
                ).role = "body_template"
        if not s.source and not s.template:
            layout.operator("cm.create_demo", text="Try a Geometry Demo")
        if f.advanced_ui:
            operator = layout.operator('cm.import_mesh', text='Import Source Only')
            operator.role = 'source'
            operator.prepare_workflow = False
            row = layout.row()
            row.operator("cm.use_active", text="Selected as Source").role = "source"
            row.operator("cm.use_active", text="Selected as Template").role = "template"
    elif stage == "WRAP":
        if not s.source or not s.template:
            text(layout, "Add a Source and MetaHuman Template in step 1.")
        else:
            placement = s.source.get('cm_source_alignment')
            if placement == 'METAHUMAN':
                text(layout, 'Import: MetaHuman surface recognized; original size and placement kept.')
            elif placement == 'KEEP':
                text(layout, 'Import: original size and placement kept.')
            if placement == 'METAHUMAN':
                text(layout, 'Native topology: no landmarks or surface projection needed.')
                layout.operator('cm.conform_native_surface',icon='CHECKMARK')
            else:
                header,proxy_box=layout.panel('cm_wrap_proxy',default_closed=True)
                header.label(text='Large Source / Solver Proxy')
                if proxy_box:
                    proxy_box.prop(f,'decimate_ratio')
                    button(proxy_box,'Rebuild Solver Proxy' if f.proxy else 'Create Solver Proxy','DECIMATE')
                    if f.proxy and f.proxy.type=='MESH':
                        proxy_box.label(text=f'{len(f.proxy.data.vertices):,} proxy vertices')
                        button(proxy_box,'Use Full Source','PROXY_DISABLE')
                    text(proxy_box,'The original stays available for texture and sculpt-detail projection. Rebuild after editing it.')
                layout.prop(f, "wrap_method", expand=True)
                if f.wrap_method == "AUTO":
                    layout.prop(f, "front_axis", text="Template Faces")
                    if not s.source.get("cm_front_axis"):
                        layout.prop(f, "source_front_axis", text="Source Faces")
                    button(layout, "Align & Detect Landmarks", "DETECT", icon="TRACKER")
                    text(layout, "Auto Wrap aligns, detects points and fits the character. Detect separately to inspect first.")
                    if s.landmarks:
                        layout.operator('cm.landmarks',text='Edit Detected Points')
                else:
                    button(layout, "Rough Align Source", "ALIGN")
                    layout.operator("cm.landmarks", text="Place / Edit Landmark Pairs")
                    text(layout, 'Click source then template. Right-drag points to correct; Ctrl Z undoes; Esc finishes.')
                    row=layout.row(align=True)
                    row.operator('cm.remove_last_landmark', text='Remove Last')
                    row.operator('cm.clear_landmarks', text='Clear All')
                    row=layout.row(align=True)
                    row.operator('cm.landmark_file', text='Save Points').load=False
                    row.operator('cm.landmark_file', text='Load Points').load=True
                    layout.operator("cm.align", text="Align from Landmarks")
                complete_pairs=sum(bool(p.source and p.template) for p in s.landmarks)
                pending_pairs=len(s.landmarks)-complete_pairs
                layout.label(text=f'{complete_pairs} landmark pairs'+(f' · {pending_pairs} incomplete' if pending_pairs else ''))
                from .features import generated_head_update
                if s.part=='FULL_BODY' and f.body_template and not generated_head_update(s):
                    head_count=sum(bool(p.template and p.template.parent==s.template) for p in s.landmarks)
                    body_count=sum(bool(p.template and p.template.parent==f.body_template) for p in s.landmarks)
                    layout.label(text=f'Head: {head_count} · Body: {body_count}')
                    if f.wrap_method=='MANUAL':
                        text(layout, 'Use at least three spread pairs across the whole character. Head and body fit together.')
                if s.landmarks:
                    row = layout.row(align=True)
                    row.operator("cm.guided", text="Source Points").action = "VIEW_SOURCE"
                    row.operator(
                        "cm.guided", text="Template Points"
                    ).action = "VIEW_TEMPLATE"
                layout.prop(f, "wrap_quality", text="Quality")
                button(
                    layout,
                    "Update Face" if s.result and s.source and s.source.get('cm_gnm') and s.part!='BODY' else "Auto Wrap Character" if f.wrap_method=='AUTO' else "Wrap Character",
                    "WRAP",
                    True,
                    icon="MOD_SHRINKWRAP",
                )
                if s.result:
                    text(
                        layout,
                        "Fits the new face to your current body. Body sculpt and skin materials are preserved."
                        if s.part=='FULL_BODY' and s.source and s.source.get('cm_gnm') and f.body_result
                        else "Fits the new face and carries your head sculpt forward."
                        if s.part=='HEAD' and s.source and s.source.get('cm_gnm')
                        else "Wrapping again creates a new result; previous objects are retained.",
                    )
                from . import body_pose
                body_pose.draw(layout,context)
    elif stage == "REFINE":
        if s.result:
            from . import body_pose
            body_pose.draw(layout,context)
            if s.detail and s.detail.get('cm_head_update_pending') and not detail_jobs.running():
                button(layout,'Update Retained Sculpt','RETRY_HEAD_DETAIL',icon='FILE_REFRESH')
            text(
                layout,
                "Check silhouette, eyelids, lips and the neck. Continue when the fit looks right.",
            )
            try:
                protected = json.loads(s.result.get('cm_conservative_surface', '{}')).get('limited_steps', 0)
            except (ValueError, TypeError, AttributeError):
                protected = 0
            if protected:
                text(layout, 'Conservative fitting reduced steep surface changes. If the fit is too loose, adjust points and wrap again.')
            from . import joint_wrap
            morph_objects=joint_wrap.morph_parts(context.scene)
            if morph_objects:
                row=layout.row()
                row.enabled=context.mode=='OBJECT' and not (f.body_editing or f.full_sculpt or f.restore_editing or f.expression_active)
                row.prop(f,'wrap_influence',text='Full Body Morph' if len(morph_objects)>1 else 'Body Morph' if s.part=='BODY' else 'Head Morph',slider=True)
            row = layout.row(align=True)
            source_button = row.row(align=True)
            source_button.enabled = bool(s.source)
            source_button.operator("cm.guided", text="Source").action = "VIEW_SOURCE"
            row.operator("cm.guided", text="Result").action = "VIEW_RESULT"
            if s.part == "FULL_BODY":
                row.operator("cm.guided", text="Body").action = "VIEW_BODY"
            if s.part!='BODY' and s.source:
                from . import eye_fit_jobs
                row=layout.row()
                if eye_fit_jobs.running():row.operator('cm.cancel_source_eyes',text='Cancel Surface Fit',icon='CANCEL')
                elif s.source:
                    row.enabled=context.mode=='OBJECT' and not (f.body_editing or f.full_sculpt or f.restore_editing or f.expression_active)
                    if f.eye_left and f.eye_right:row.operator('cm.fit_source_eyes',text='Fit Eyes to Source',icon='VIEWZOOM')
                    row.operator('cm.fit_source_ears',text='Reconstruct Ears',icon='MOD_SMOOTH')
                if f.eye_left and f.eye_right:
                    from . import eye_color_jobs
                    layout.operator('cm.cancel_eye_color' if eye_color_jobs.running() else 'cm.source_eye_color',text='Cancel Eye Color' if eye_color_jobs.running() else 'Project Eye Color')
            if s.source:
                from .features import draw_region_projection
                header,regional=layout.panel('cm_refine_region_fit',default_closed=True)
                header.label(text='Regional Surface Fit')
                if regional:draw_region_projection(regional,context)
            row = layout.row(align=True)
            row.enabled = not f.body_editing
            choose = row.row(align=True)
            choose.enabled = context.mode == 'OBJECT' and not (f.full_sculpt or f.restore_editing or f.expression_active)
            choose.menu('CM_MT_sculpt_parts', text='Sculpt Part', icon='SCULPTMODE_HLT')
            row.operator("cm.finish_sculpt", text="Finish Sculpt")
            layout.prop(f, "sculpt_region", text="Region")
            row = layout.row(align=True)
            row.enabled = not f.body_editing
            row.operator("cm.feature", text="Create Masks").action = "MASKS"
            row.operator("cm.feature", text="Sculpt Region").action = "SCULPT_REGION"
            if s.part == "HEAD" and not f.body_result:
                layout.prop(f, "proportional_scale")
                button(
                    layout,
                    "Load Body",
                    "LOAD_BODY",
                    bool(character_library()),
                    icon="USER",
                )
            if s.part == "FULL_BODY":
                button(
                    layout,
                    "Attach Head to Body",
                    "ATTACH_BODY",
                    not f.body_editing and not f.full_sculpt,
                )
                row = layout.row(align=True)
                row.enabled = not f.body_editing
                row.operator(
                    "cm.feature", text="Sculpt Full Model"
                ).action = "FULL_SCULPT"
                if f.full_sculpt:
                    row.operator(
                        "cm.feature", text="Apply"
                    ).action = "FULL_SCULPT_APPLY"
                    row.operator(
                        "cm.feature", text="Cancel"
                    ).action = "FULL_SCULPT_CANCEL"
            if s.part in {"BODY", "FULL_BODY"}:
                from . import native_body
                selector=layout.row()
                selector.enabled=not f.body_editing
                selector.prop(f,'body_backend',text='Generator')
                if f.body_backend=='NATIVE' and not native_body.available(f):
                    layout.prop(f,'expression_bridge_dir',text='Bridge Folder')
                    text(layout,'Choose the unpacked Bridge 0.4 folder.')
                if not f.body_editing:
                    layout.operator(
                        "cm.feature", text="Shape Body"
                    ).action = "BODY_PREVIEW"
                else:
                    if f.body_backend=='NATIVE':text(layout,f.body_native_status)
                    layout.prop(f, "body_category", text="Section")
                    items=[item for item in f.body_controls if item.category==f.body_category]
                    primary=items if f.body_backend!='NATIVE' else [item for item in items if item.key in native_body.NATIVE_KEYS]
                    for item in primary:
                        native_gender=f.body_backend=='NATIVE' and item.key=='gender'
                        layout.prop(item,"value",text='Masculine ↔ Feminine' if native_gender else item.label)
                        if native_gender:layout.label(text='− masculine    + feminine')
                    if f.body_backend=='NATIVE':
                        secondary=[item for item in items if item.key not in native_body.NATIVE_KEYS]
                        if secondary:layout.label(text='Sculpt adjustments')
                        for item in secondary:layout.prop(item,'value',text=item.label)
                    row = layout.row(align=True)
                    apply=row.row(align=True)
                    apply.enabled=f.body_backend!='NATIVE' or native_body.can_apply()
                    apply.operator("cm.feature", text="Apply Body").action = "BODY_APPLY"
                    row.operator("cm.feature", text="Cancel").action = "BODY_CANCEL"
                    row = layout.row(align=True)
                    row.operator(
                        "cm.feature", text="Reset Section"
                    ).action = "BODY_RESET"
                    row.operator(
                        "cm.feature", text="Reset All"
                    ).action = "BODY_RESET_ALL"
                row = layout.row(align=True)
                row.enabled = not f.body_editing and not f.full_sculpt
                row.operator("cm.save_custom_body", text="Save Body")
                row.operator("cm.load_custom_body", text="Load Custom Body")
        else:
            text(layout, "Wrap the character in step 2 first.")
    elif stage == "TEXTURE":
        from . import paint_session
        _, skin_target=paint_session.part(context)
        try:photo_colored=bool(paint_session.albedo(skin_target).get('cm_photo_reference'))
        except (ValueError,AttributeError):photo_colored=False
        if s.part == "FULL_BODY":
            from . import character_skin
            character_skin.draw(layout.box(), context)
            layout.prop(f, "texture_part", expand=True)
        source_textured = s.source and any(
            mat
            and mat.use_nodes
            and any(
                node.type == "TEX_IMAGE" and node.image for node in mat.node_tree.nodes
            )
            for mat in s.source.data.materials
        )
        if photo_colored:
            text(layout,'Photo color is applied. Use Smart Paint for cleanup and Detail for final normals.')
        elif s.source and not source_textured:
            text(
                layout,
                "Source has flat color. Load its UV skin image for textured skin.",
            )
            layout.operator("cm.load_image", text="Load Source Skin Image")
        layout.prop(f,'bake_device',text='Device')
        layout.prop(f, "skin_map_size", text="Size")
        if f.skin_map_size=='8192':text(layout,'Smart Paint retains 8K detail; full detail returns after each stroke.')
        if s.part=='FULL_BODY':layout.prop(f,'skin_bake_scope',text='Scope')
        layout.prop(f,'skin_bake_channels')
        if f.skin_bake_channels!='ROUGHNESS':
            layout.prop(f,'skin_replace_normal')
            if not f.skin_replace_normal:text(layout,'Keeps existing sculpt normals; creates a normal if missing.')
        layout.prop(s, "bake_distance", text="Projection Distance")
        button(
            layout,
            "Replace Color from Source" if photo_colored else "Bake Selected Skin",
            "BAKE_SKIN",
            bool(s.result and s.source),
            icon="RENDER_STILL",
        )
        text(layout,'Source baking replaces the selected skin color.')
        if f.unreal_character:
            button(layout, 'Apply Selected Albedo to Creator', 'APPLY_NATIVE_ALBEDO', bool(s.result), icon='EXPORT')
            text(layout, 'Checks native geometry; keeps the existing rig and other maps.')
        layout.operator('cm.retain_authored_skin',text='Keep Current Skin as Authored')
        row = layout.row(align=True)
        row.operator("cm.preview_material", text="Review Skin")
        row.operator("cm.preview_material", text="Paint Skin").paint = True
        layout.operator("cm.finish_sculpt", text="Finish Painting")
        box = layout.box()
        box.label(text='Repair Skin', icon='BRUSH_DATA')
        box.prop(f, 'paint_preserve_color')
        if f.paint_preserve_color:
            box.prop(f, 'paint_detail_radius')
            box.label(text='Paint over unwanted fine texture; local color stays.')
        box.operator('cm.reference_paint', text='Smart Paint')
        box.label(text='Uses the clean skin applied to this character.')
        body = s.part == 'BODY' or s.part == 'FULL_BODY' and f.texture_part == 'BODY'
        box.prop(f, 'reference_skin_body' if body else 'reference_skin', text='Override')
        box.operator('cm.load_image', text='Load Matching Reference').reference = True
        from . import surface_heal
        surface_heal.draw(layout,context)
        from . import skin_reconstruction
        skin_reconstruction.draw(layout,context)
        from . import skin_bake_layers_ui
        skin_bake_layers_ui.draw(layout,context)
    elif stage == "DETAIL":
        from . import skin_bake_layers_ui
        skin_bake_layers_ui.draw(layout,context)
        if f.sculpt_session_role!='AUTHOR':
            row=layout.row(align=True)
            row.operator('cm.sculpt_session',text='Open Sculpting Session',icon='WINDOW').action='OPEN'
            if f.sculpt_session_path:
                row.operator('cm.sculpt_session',text='Resume').action='RESUME'
            if f.sculpt_session_status:text(layout,f.sculpt_session_status)
        if s.part == "FULL_BODY":
            layout.prop(f, "texture_part", expand=True)
        selected_result = (
            f.body_result
            if s.part == "FULL_BODY" and f.texture_part == "BODY"
            else s.result
        )
        selected_detail = (
            f.body_detail
            if s.part == "FULL_BODY" and f.texture_part == "BODY"
            else s.detail
        )
        if not selected_result:
            text(layout, "Load or wrap the body in Refine before editing body detail.")
            return
        if selected_detail and selected_detail.get('cm_detail_level',1)>1:
            from . import detail_layers
            if not detail_layers.has_reference(selected_detail):
                box=layout.box()
                box.label(text='Recover Older Sculpt',icon='FILE_REFRESH')
                text(box,'Preserves sculpting baked into earlier subdivision levels.')
                field='body_detail_previous' if s.part=='BODY' or s.part=='FULL_BODY' and f.texture_part=='BODY' else 'head_detail_previous'
                box.prop(f,'detail_recovery_manual',toggle=True)
                if f.detail_recovery_manual:
                    text(box,'Choose the unchanged sculpt one level below.')
                    box.prop(f,field,text='')
                previous=selected_detail.get('cm_detail_previous')
                if isinstance(previous,bpy.types.Object):text(box,'Recorded previous layer: '+previous.name)
                if not f.detail_recovery_manual:text(box,'Finds the unchanged previous level automatically.')
                row=box.row();row.enabled=context.mode=='OBJECT' and not detail_jobs.running() and (not f.detail_recovery_manual or bool(getattr(f,field)))
                row.operator('cm.feature',text='Recover Saved Sculpt',icon='FILE_REFRESH').action='RECOVER_DETAIL'
        text(layout, "Optional: project dense source detail before the final maps.")
        extra_detail = s.source and len(
            s.source.evaluated_get(context.evaluated_depsgraph_get()).data.vertices
        ) > len(selected_result.data.vertices)
        if selected_result.get("cm_body_source") == "LIBRARY":
            extra_detail = False
        if not s.source or selected_result.get("cm_body_source") == "LIBRARY":
            text(
                layout, "Create a sculpt layer for pores, folds or fine surface detail."
            )
        elif not extra_detail:
            text(
                layout,
                "Source is lower density than the result. Extra projection is usually unnecessary; continue when ready.",
            )
        if selected_detail:
            layout.label(text='Sculpt Level: LV'+str(selected_detail.get('cm_detail_level',s.detail_levels)))
        else:
            layout.prop(s, "detail_levels", text="Subdivision")
        from . import scan_projection
        pending_scan = bool(selected_detail and not selected_detail.get('cm_scan_applied', True))
        regenerable = scan_projection.editable(selected_detail)
        if (not selected_detail and extra_detail) or regenerable:
            layout.prop(f, 'use_detail_mask', text='Use Scan Mask')
            if f.use_detail_mask:
                layout.operator('cm.detail_mask', text='Edit Scan Mask', icon='MOD_MASK').action = 'START'
                text(layout, 'Paint where scan detail is allowed. Neck fade and pins remain separate protections.')
        if not selected_detail:
            if extra_detail:
                layout.operator('cm.feature', text='Generate Source Displacement').action = 'DISPLACE'
            layout.operator(
                "cm.feature", text="Create Blank Sculpt Layer" if extra_detail else "Create Sculpt Layer"
            ).action = "SCULPT_LAYER"
        elif not selected_detail.get("cm_scan_applied", False):
            layout.prop(s, 'detail_strength', text='Scan Strength')
            if regenerable:
                layout.operator('cm.feature', text='Regenerate Source Displacement', icon='FILE_REFRESH').action = 'DISPLACE'
                if scan_projection.dirty(selected_detail, f):
                    text(layout, 'Mask or protection changed. Regenerate before applying.')
            apply = layout.row()
            apply.enabled = not regenerable or not scan_projection.dirty(selected_detail, f)
            apply.operator(
                "cm.feature", text="Apply Projected Detail"
            ).action = "APPLY_SCAN"
        if selected_detail:
            row = layout.row(align=True)
            row.enabled = not pending_scan
            sculpt = row.operator("cm.sculpt", text="Sculpt Detail")
            sculpt.detail, sculpt.part = True, f.texture_part
            row.operator("cm.finish_sculpt", text="Finish")
            row = layout.row(align=True)
            subdivide=row.row(align=True);subdivide.enabled=not pending_scan and selected_detail.get('cm_detail_level',s.detail_levels)<3
            subdivide.operator("cm.feature", text="Subdivide").action = "SUBDIVIDE_DETAIL"
            row.operator("cm.feature", text="Clear Layer").action = "CLEAR_DETAIL"
        from . import skin_brushes
        skin_brushes.draw(layout, context)
        pores = layout.column()
        pores.enabled = bool(selected_detail)
        pores.prop(f, 'add_pores')
        if f.add_pores:
            pores.prop(f, 'pore_strength')
            pores.prop(f, 'pore_spacing')
        layout.prop(f,'bake_device',text='Bake Device')
        layout.prop(s, "map_size", text="Resolution")
        if s.part=='FULL_BODY':layout.prop(f,'final_map_scope',text='Parts')
        from . import normal_composition,simple_shading
        normal_composition.draw(layout,context.scene)
        simple_shading.draw(layout,context.scene)
        final_maps = layout.row()
        final_maps.enabled = not pending_scan
        final_maps.operator(
            "cm.feature", text="Bake Final Normal & Cavity"
        ).action = "FINAL_MAPS"
        if f.texture_part == "HEAD" or s.part != "FULL_BODY":
            layout.prop(f, "show_wrinkles", toggle=True)
            if f.show_wrinkles:
                box = layout.box()
                box.prop(f, "wrinkle_slot", expand=True)
                box.prop(f,'wrinkle_map_size',text='Resolution')
                box.prop(f, "wrinkle_native_bases")
                if f.wrinkle_native_bases:
                    try:
                        from . import expressions
                        ready = bool(expressions.bank(f).get('wrinkle_poses'))
                    except (ValueError, OSError):
                        ready = False
                    if not ready:
                        from . import wrinkles
                        bridge_ready=wrinkles.native_bridge_ready(f)
                        if not bridge_ready:
                            box.prop(f,'expression_bridge_dir',text='Bridge Folder')
                            text(box,'Choose the compiled Bridge 0.5 folder.')
                        row=box.row()
                        row.enabled=bridge_ready
                        row.operator('cm.prepare_expressions', text='Prepare Native Wrinkle Bases', icon='IMPORT')
                box.prop(f, "wrinkle_protect_area")
                row=box.row(align=True)
                row.operator('cm.load_wrinkle_base',text='Load Custom WM',icon='FILE_FOLDER')
                if getattr(f,f.wrinkle_slot.lower()):
                    row.operator('cm.save_wrinkle_base',text='Save Base',icon='FILE_TICK')
                if not getattr(f, f.wrinkle_slot.lower()):
                    box.operator(
                        "cm.feature", text="Create Wrinkle Layer"
                    ).action = "WRINKLE_CREATE"
                else:
                    row = box.row(align=True)
                    row.operator(
                        "cm.feature", text="Sculpt Wrinkle"
                    ).action = "WRINKLE_SCULPT"
                    row.operator("cm.finish_sculpt", text="Finish")
                    box.operator("cm.feature", text="Update Sculpt Area").action = "WRINKLE_AREA"
                    box.operator('cm.feature',text='Reset Base',icon='FILE_REFRESH').action='WRINKLE_RESET'
                    box.operator(
                        "cm.feature", text="Bake Selected Wrinkle"
                    ).action = "WRINKLE_BAKE"
                box.prop(f, "bloodflow")
                if f.bloodflow:
                    box.prop(f, f.wrinkle_slot.lower() + "_mask", text="Region Mask")
                    box.prop(f, "bloodflow_strength", text="Tint Strength")
                    row=box.row(align=True)
                    row.operator('cm.feature',text='Update Selected CM').action='BLOODFLOW_BAKE'
                    row.operator('cm.feature',text='Update All CM').action='BLOODFLOW_ALL'
                box.operator(
                    "cm.feature", text="Bake All Wrinkles"
                ).action = "WRINKLE_ALL"
                box.prop(f,'bake_open_folder')
    elif stage == "GROOM":
        g = context.scene.cm_groom
        from .grooming import draw_lash_fit
        draw_lash_fit(layout,context)
        if context.mode=='OBJECT':
            text(layout,'Create and edit hair here. Finish character assembly in Unreal before binding and Push.')
        if g.exchange_status and not groom_jobs.running():
            text(layout,g.exchange_status)
        if g.exchange_receipt and not g.exchange_result and not groom_jobs.running():
            layout.operator('cm.groom',text='Check Previous Exchange',icon='FILE_REFRESH').action='RESUME_EXCHANGE'
        if g.scalp_editing and context.mode == 'SCULPT':
            text(layout,'Shape this growth region with the sculpt brushes, then finish.')
            layout.operator('cm.groom',text='Finish Scalp',icon='CHECKMARK').action='FINISH'
            return
        elif context.mode == 'SCULPT_CURVES':
            text(layout, "Shape the hair with Blender's sculpt brushes.")
            from .grooming import draw_population_selection
            draw_population_selection(layout)
            text(layout, 'Guide selection leaves render strands untouched; review interpolation in Unreal.')
            layout.operator('cm.groom', text='Finish Hair', icon='CHECKMARK').action = 'FINISH'
            return
        else:
            head = layout.box()
            head.label(text='Head & Growth Regions',icon='OUTLINER_OB_MESH')
            row=head.row()
            row.enabled=bool(s.result)
            row.operator('cm.groom',text='Use Character Head').action='USE_CHARACTER_HEAD'
            row=head.row(align=True)
            row.enabled=not groom_jobs.running() and not unreal_jobs.running()
            row.operator('cm.groom',text='Get Selected Head').action='SELECT_HEAD'
            row.operator('cm.import_groom_head',text='Import FBX',icon='FILE_FOLDER')
            if g.main_scalp:
                text(head,'Head: '+g.main_scalp.name)
                from . import groom_follow
                if groom_follow.modifier(g.main_scalp):
                    if g.main_scalp.get('cm_groom_follow_status'):
                        text(head,g.main_scalp['cm_groom_follow_status'])
                    else:
                        head.label(text='Live character link',icon='LINKED')
                        text(head,'Hair follows face and morph edits.')
                text(head,'Edit a growth region below to adjust coverage.')
                if g.groom and g.groom.type=='CURVES' and g.groom.data.surface!=g.main_scalp:
                    text(head,'This groom keeps its previous surface. Use Attach to Scalp to move it to this head.')
            if not g.groom:
                text(layout, "Create hair below, or get an existing groom from Unreal.")
            header, create = layout.panel('cm_groom_create', default_closed=bool(g.groom))
            header.label(text='Create Hair', icon='CURVES_DATA')
            if create:
                create.prop(g, 'name', text='Name')
                create.prop(g, 'region', text='Region')
                create.prop(g, 'scalp', text='Scalp')
                create.operator('cm.groom', text='Create Scalp Region').action = 'SCALP'
                row=create.row()
                row.enabled=bool(g.scalp)
                row.operator('cm.groom',text='Edit Scalp Region',icon='SCULPTMODE_HLT').action='EDIT_SCALP'
                row = create.row(align=True)
                row.prop(g, 'hair_length', text='Length')
                row.prop(g, 'hair_count', text='Strands')
                create.prop(g, 'hair_seed')
                row = create.row(align=True)
                row.operator('cm.groom', text='Create Hair').action = 'SEED'
                row.operator('cm.groom', text='Start Empty').action = 'EMPTY'
            edit = layout.box()
            edit.label(text='Edit & Combine')
            edit.prop(g, 'groom')
            from .grooming import draw_population_selection
            draw_population_selection(edit)
            row = edit.row(align=True)
            row.enabled = bool(g.groom and g.groom.type == 'CURVES')
            row.operator('cm.groom', text='Edit Hair', icon='SCULPTMODE_HLT').action = 'EDIT'
            row.operator('cm.groom', text='Attach to Head').action = 'ATTACH'
            row = edit.row()
            row.enabled = len([o for o in context.selected_objects if o.type in {'CURVES','CURVE'}]) >= 2
            row.operator('cm.groom', text='Combine Selected Grooms').action = 'MIX'
            text(edit, 'Select two or more grooms in the Outliner. Originals are kept.')
            row = edit.row(align=True)
            row.enabled = bool(g.groom and g.groom.type == 'CURVES')
            row.prop(g, 'simplify_ratio', text='Keep')
            row.operator('cm.groom', text='Simplify').action = 'SIMPLIFY'
            text(edit, 'Keep is applied to each group and its guides. Small groups retain at least one strand.')
            header, appearance = layout.panel('cm_groom_appearance', default_closed=True)
            header.label(text='Hair Preview', icon='MATERIAL')
            if appearance:
                appearance.prop(g, 'preview_color', text='Color')
                appearance.prop(g, 'preview_roughness', text='Roughness')
                appearance.prop(g, 'preview_strand_color')
                row = appearance.row()
                row.enabled = bool(g.groom and g.groom.type == 'CURVES')
                row.operator('cm.groom', text='Apply Preview Material', icon='MATERIAL').action = 'PREVIEW'
                text(appearance, 'Blender preview only. Tune final hair color and shading in Unreal.')
            header, exchange = layout.panel('cm_groom_exchange', default_closed=True)
            header.label(text='Unreal & Files', icon='EXPORT')
            if exchange:
                exchange.enabled=not groom_jobs.running() and not unreal_jobs.running()
                text(exchange,"Use the linked character for binding, or select an Unreal asset to import." if f.unreal_character else "Select an asset in Unreal's Content Browser, then get it here.")
                exchange.operator('cm.groom', text='Get Selected Groom', icon='IMPORT').action = 'SELECT_GROOM'
                text(exchange,'Bind to: '+g.unreal_target.rsplit('/',1)[-1].split('.')[0] if g.unreal_target else ('The linked Creator head will be prepared on Push.' if f.unreal_character else 'Choose your character\'s Face mesh as the binding target.'))
                if f.unreal_character:
                    exchange.operator('cm.groom',text='Use Linked Creator Head').action='CHARACTER_TARGET'
                exchange.operator('cm.groom', text='Use Selected Face as Binding Target').action = 'SELECT_TARGET'
                if g.last_push_result or g.exchange_result:
                    try:delivered=json.loads(g.last_push_result or g.exchange_result)
                    except ValueError:delivered={}
                    if delivered.get('status')=='complete' and delivered.get('groom') and delivered.get('binding'):
                        if delivered.get('wardrobe') and f.unreal_character:
                            exchange.prop(g,'wardrobe_slot')
                            exchange.operator('cm.groom',text='Equip Groom in Creator',icon='OUTLINER_OB_CURVES').action='EQUIP_CREATOR'
                        exchange.operator('cm.groom',text='Apply to Selected Unreal Actor',icon='OUTLINER_OB_CURVES').action='APPLY_ACTOR'
                        text(exchange,'Select one matching character in the Unreal level. Save the level after applying.')
                exchange.prop(g,'show_asset_paths',toggle=True)
                if g.show_asset_paths:
                    exchange.prop(g, 'unreal_target', text='Face Mesh')
                    exchange.operator('cm.groom', text='Get Head from Path').action = 'GRAB_HEAD'
                    exchange.prop(g, 'unreal_asset', text='Groom Asset')
                    exchange.operator('cm.groom', text='Get Groom from Path', icon='IMPORT').action = 'GRAB'
                row = exchange.row(align=True)
                row.enabled = bool(g.groom)
                row.operator('cm.groom', text='Export Unreal Alembic').action = 'ALEMBIC'
                row.operator('cm.groom', text='Export Strands').action = 'EXPORT'
                row = exchange.row()
                row.enabled = bool(g.groom)
                row.operator('cm.groom', text='Push Groom & Wardrobe', icon='EXPORT').action = 'PUSH'
                text(exchange,'Repeated pushes create a new revision. Results are selected in Unreal.')
                if g.exchange_result:
                    try:result=json.loads(g.exchange_result)
                    except ValueError:result={}
                    for label,key in [('Groom','groom'),('Binding','binding'),('Wardrobe','wardrobe')]:
                        if result.get(key):text(exchange,label+': '+result[key])
    elif stage == "EXPORT":
        layout.prop(s, "create_character")
        layout.prop(s, "unreal_project", text="Open UE Project")
        layout.prop(s, "unreal_destination", text="Content Folder")
        if f.advanced_ui and s.create_character and s.part!='BODY':
            animated=layout.box();animated.label(text='Expression Map Inputs')
            animated.prop(f,'unreal_wrinkle_input_encoding');animated.prop(f,'unreal_color_input_encoding')
            if f.unreal_wrinkle_input_encoding=='FULL':
                animated.prop(f,'unreal_wrinkle_neutral_mode')
                if f.unreal_wrinkle_neutral_mode=='IMAGE':animated.prop(f,'unreal_wrinkle_neutral_image')
        button(layout, "Check Character", "CHECK", bool(s.result), icon="CHECKMARK")
        if f.delivery_report:
            text(layout, f.delivery_report)
            if 'collapsed faces' in f.delivery_report:
                button(layout,'Inspect Collapsed Faces','INSPECT_COLLAPSED',context.mode=='OBJECT',icon='EDITMODE_HLT')
            if 'Rebake ' in f.delivery_report:
                repair=layout.box()
                repair.operator('cm.feature',text='Bake Final Normal & Cavity',icon='RENDER_STILL').action='FINAL_MAPS'
                if any((getattr(f,slot.lower()) for slot in ('WM1','WM2','WM3'))):
                    repair.operator('cm.feature',text='Bake All Wrinkles',icon='RENDER_STILL').action='WRINKLE_ALL'
                text(repair,'Using externally authored maps? Keep the current files for this geometry.')
                repair.operator('cm.use_custom_maps',text='Use Current Files as Custom Maps',icon='FILE_TICK')
        export_ready=bool(s.result) and not any(module.running() for module in (unreal_jobs,bake_jobs,detail_jobs,wrinkle_jobs,groom_jobs)) and f.delivery_report in {'','Character checks passed.','Geometry checks passed.'}
        button(layout, "Export Unreal Bundle", "EXPORT", export_ready, icon="EXPORT")
        button(
            layout, "Push to Open Unreal Project", "PUSH", export_ready, icon="PLAY"
        )
        text(layout, "Head and body skin maps are applied in Creator during import.")
        text(layout, "Push saves this project first. Repeated pushes create a new revision.")
        web = layout.box()
        web.label(text='Web / Three.js')
        web.prop(f,'web_export_kind')
        if f.web_export_kind=='NATIVE':
            from . import native_web_jobs
            native_web_jobs.draw(web,context)
        else:
            web.prop(s, 'web_texture_size')
            web.prop(s, 'web_skin_finish')
            action = web.row()
            action.enabled = bool(s.result) and not any(module.running() for module in (unreal_jobs,bake_jobs,detail_jobs,wrinkle_jobs,groom_jobs))
            action.operator('cm.export_static_glb', text='Export Static GLB', icon='EXPORT')
            text(web, 'Accepted mesh and skin images. Animation and groom strands are separate.')
        from . import head_revision
        head_revision.draw(layout,context)
        if f.unreal_push_status and not unreal_jobs.running():
            text(layout, f.unreal_push_status)
        if f.unreal_character:
            text(layout, 'Last character: ' + f.unreal_character.rsplit('/', 1)[-1].split('.')[0])
            finish=layout.box()
            finish.label(text='Finish in Creator')
            finish.prop(f,'roughness_response_mode')
            if f.roughness_response_mode!='ABSOLUTE':finish.prop(f,'skin_finish_roughness')
            button(finish,'Open in Creator','OPEN_CREATOR',not unreal_jobs.running() and not groom_jobs.running(),icon='WINDOW')
            button(finish,'Refresh Unreal Status','REFRESH_UNREAL',not unreal_jobs.running() and not groom_jobs.running(),icon='FILE_REFRESH')
            button(finish,'Create Full Rig','REQUEST_FULL_RIG',not unreal_jobs.running() and not groom_jobs.running(),icon='ARMATURE_DATA')
            text(finish,'Uses the linked Unreal character and Epic rigging service. Save UE artist edits first. Existing rigs and pending requests are reused.')
            button(finish,'Download Native Texture Sources','DOWNLOAD_NATIVE_SOURCES',not unreal_jobs.running() and not groom_jobs.running(),icon='IMPORT')
            text(finish,'Uses the source resolutions configured in Creator. Save UE artist edits first; the download can resume after reopening Blender.')
            finish.prop(f,'unreal_artist_scope')
            button(finish,'Capture UE Artist Edits','CAPTURE_UNREAL_ARTIST',not unreal_jobs.running() and not groom_jobs.running(),icon='IMPORT')
            text(finish,'Save Creator edits first. Preserves native settings/assets without replacing Blender work.')
            if f.unreal_artist_scope=='SELECTED_ACTOR':text(finish,'Select one saved UE level actor. Includes its saved level and dependencies.')
            if f.unreal_artist_snapshot:text(finish,'UE artist snapshot linked to this project.')
            if f.unreal_artist_review:button(finish,'Review Captured UE Edits','REVIEW_UNREAL_ARTIST',icon='TEXT')
            if f.unreal_character==f.unreal_revision_character:
                finish.prop(f,'unreal_revision_update_scope')
                if f.advanced_ui:
                    finish.prop(f,'unreal_wrinkle_input_encoding');finish.prop(f,'unreal_color_input_encoding')
                    if f.unreal_wrinkle_input_encoding=='FULL':
                        finish.prop(f,'unreal_wrinkle_neutral_mode')
                        if f.unreal_wrinkle_neutral_mode=='IMAGE':finish.prop(f,'unreal_wrinkle_neutral_image')
                text(finish,'Refreshes selected head maps. Geometry, rig and body stay intact.')
            else:text(finish,'Updates existing delivered texture assets. Geometry and rig stay intact.')
            from . import native_map_update
            if native_map_update.draw(finish,context.scene):text(finish,'Applies selected native detail/finish maps. Source albedos and eye overrides stay intact.')
            button(finish,'Update Textures in Unreal','UPDATE_UNREAL_TEXTURES',not unreal_jobs.running() and not groom_jobs.running(),icon='FILE_REFRESH')
            native_details=False
            saved_status=False
            resources={}
            if f.unreal_status_report:
                try:report=json.loads(f.unreal_status_report)
                except ValueError:report={}
                if report.get('character')==f.unreal_character:
                    assembly=report.get('assembly',{})
                    saved_status=assembly.get('inspection_scope')=='SAVED_CHARACTER'
                    native_details=assembly.get('detailed_status_available',False)
                    rig=assembly.get('rig_state')
                    if rig:
                        finish.label(text='Rig: '+{'RIGGED':'Ready','PENDING':'In progress','UNRIGGED':'Required'}.get(rig,rig))
                    available=assembly.get('has_texture_sources')
                    if available is not None:
                        finish.label(text='Texture sources: '+('Available' if available else 'Required'))
                    if assembly.get('texture_request_pending'):
                        text(finish,'Texture request in progress in Creator.')
                    if assembly.get('reason'):
                        text(finish,assembly['reason'])
                    resources=assembly.get('resources',{})
                    if 'required_commit_bytes' in resources:
                        gib=1024**3
                        text(finish,f"Assembly commit headroom: {resources['available_commit_bytes']/gib:.1f} GiB available; {resources['required_commit_bytes']/gib:.0f} GiB recommended.")
                        if not resources.get('editing_session_reused'):text(finish,'Includes an allowance to load the character.')
                        if resources['available_commit_bytes']<resources['required_commit_bytes'] and not f.unreal_build_allow_low_memory:
                            text(finish,'Close unused heavy applications, then refresh status before assembly.')
                        if resources.get('free_disk_bytes',0)<resources.get('disk_reserve_bytes',0):
                            text(finish,'Keep at least 10 GiB of free disk space, then refresh status.')
            from . import native_geometry_state
            native_geometry_problem=native_geometry_state.check(context)
            if native_geometry_problem:
                text(finish,native_geometry_problem)
            elif f.unreal_assembly_ready and not saved_status:
                text(finish,'Creator reports that the character is ready to assemble.')
            elif not native_details:
                text(finish,'Complete the rig and texture source requirements in Creator, then refresh the status here.')
            build_controls=finish.column()
            build_controls.enabled=not unreal_jobs.running() and not groom_jobs.running()
            build_controls.prop(f,'unreal_build_profile')
            build_controls.prop(f,'unreal_eye_overlay_policy')
            build_controls.prop(f,'unreal_build_preserve_source_textures')
            build_controls.prop(f,'unreal_build_root')
            if f.advanced_ui:build_controls.prop(f,'unreal_build_allow_low_memory')
            from . import unreal_assembly
            header,estimate_box=finish.panel('cm_assembly_local_resources',default_closed=True)
            header.label(text='Memory and Disk Estimate')
            if estimate_box:
                button(estimate_box,'Check Locally (No Unreal Launch)','CHECK_ASSEMBLY_RESOURCES',icon='FILE_REFRESH')
                try:estimate=json.loads(f.unreal_resource_report or '{}')
                except ValueError:estimate={}
                if estimate.get('project')==str(Path(bpy.path.abspath(f.unreal_character_project)).resolve()):
                    gib=1024**3
                    text(estimate_box,'Checked '+estimate['checked_at'][:19].replace('T',' ')+' UTC')
                    text(estimate_box,f"Windows commit headroom: {estimate['available_commit_bytes']/gib:.1f} GiB. Disk free: {estimate['free_disk_bytes']/gib:.1f} GiB.")
                    text(estimate_box,f"Cold character allowance inside Unreal: {estimate['required_commit_bytes']/gib:.0f} GiB; disk reserve: {estimate['disk_reserve_bytes']/gib:.0f} GiB.")
                    text(estimate_box,'Opening Unreal consumes additional memory. Its build-time checks remain authoritative.')
                    if not estimate.get('recommendation_met'):text(estimate_box,'Free resources, then check again before assembly.')
            resource_ready=unreal_assembly.resources_allow(resources,f.unreal_build_allow_low_memory)
            button(build_controls,'Assemble Character','BUILD_UNREAL',f.unreal_assembly_ready and not native_geometry_problem and resource_ready,icon='OUTLINER_OB_ARMATURE')
            text(finish,'Assembly creates a new folder revision and retains previous builds.')
            if f.unreal_build_report:
                try:built=json.loads(f.unreal_build_report)
                except ValueError:built={}
                if built.get('character')==f.unreal_character and built.get('build',{}).get('state')=='COMPLETE':
                    text(finish,'Built actor: '+built['build']['blueprint'])
                    target=built['build'].get('component_meshes',{}).get('Face')
                    if target:
                        if context.scene.cm_groom.unreal_target==target:
                            text(finish,'Hair binding is set to this assembled character.')
                        else:
                            button(finish,'Use Assembled Face for Hair','USE_BUILD_FACE',not unreal_jobs.running() and not groom_jobs.running(),icon='CURVES')
                    else:text(finish,'Choose the final Face Skeletal Mesh in Hair for groom binding.')
    if stage == "REFINE" and s.result and s.part != "BODY":
        from . import expressions

        row=layout.row(align=True)
        row.enabled=not f.body_editing
        row.operator("cm.restore_brush",text="Repair Head",icon="BRUSH_DATA")
        row.prop(f,"show_restore_controls",text="Settings",toggle=True)
        if f.show_restore_controls:
            box=layout.box()
            box.prop(f,'restore_mode')
            box.prop(f,"restore_radius",text="Size")
            box.prop(f,"restore_strength",text="Strength")
            if f.restore_mode=='SURFACE':box.prop(f,'restore_source_adherence')
            box.prop(f,"mirror",text="Mirror")
            box.operator('cm.repair_oral_clearance')
        expressions.draw(layout,context)
    if stage in {"REFINE", "TEXTURE", "DETAIL", "EXPORT"} and s.result:
        if stage == 'TEXTURE' and f.unreal_character:
            update = layout.box()
            update.label(text='Delivered Character')
            if f.unreal_character==f.unreal_revision_character:update.prop(f,'unreal_revision_update_scope')
            if f.advanced_ui and f.unreal_character==f.unreal_revision_character:
                update.prop(f,'unreal_wrinkle_input_encoding');update.prop(f,'unreal_color_input_encoding')
                if f.unreal_wrinkle_input_encoding=='FULL':
                    update.prop(f,'unreal_wrinkle_neutral_mode')
                    if f.unreal_wrinkle_neutral_mode=='IMAGE':update.prop(f,'unreal_wrinkle_neutral_image')
            update.prop(f,'roughness_response_mode')
            if f.roughness_response_mode!='ABSOLUTE':update.prop(f,'skin_finish_roughness')
            from . import native_map_update
            native_map_update.draw(update,context.scene)
            button(update,'Update Textures in Unreal','UPDATE_UNREAL_TEXTURES',not unreal_jobs.running() and not groom_jobs.running(),icon='FILE_REFRESH')
            text(update,'Updates existing assets. Push a new revision when geometry changes.')
        from . import lookdev
        from . import eye_appearance

        eye_appearance.draw(layout,context)
        lookdev.draw(layout, context)
    if f.sculpt_session_role=='AUTHOR':
        layout.operator('cm.guided',text='Save Session',icon='FILE_TICK').action='SAVE'
        if s.status:text(layout,s.status)
        return
    layout.separator()
    nav = layout.row(align=True)
    back = nav.row(align=True)
    back.enabled = stage != "START" and context.mode=='OBJECT' and not f.body_editing
    back.operator("cm.feature", text="Back to Unreal" if stage=='GROOM' else 'Back', icon="TRIA_LEFT").action = "PREVIOUS_STEP"
    next_row = nav.row(align=True)
    next_row.enabled = (
        stage != "GROOM"
        and (
            bool(s.source and s.template and s.source != s.template)
            if stage == "START"
            else bool(s.result)
            if stage == "WRAP"
            else True
        )
        and context.mode == "OBJECT"
        and not f.body_editing
    )
    next_row.operator(
        "cm.feature", text="Groom (Optional)" if stage=='EXPORT' else 'Continue', icon="TRIA_RIGHT"
    ).action = "NEXT_STEP"
    row = layout.row(align=True)
    row.operator("cm.guided", text="Save", icon="FILE_TICK").action = "SAVE"
    row.operator("cm.focus_character", text="Focus", icon="VIEWZOOM")
    row.menu('CM_MT_skin_storage',text='',icon='FILE_FOLDER')
    row.prop(f, "advanced_ui", text="Advanced", toggle=True)
    status=s.status
    pending_detail=bool(s.detail and s.detail.get('cm_head_update_pending'))
    if pending_detail and not detail_jobs.running():
        status='Head sculpt update needed. In Refine, choose Update Retained Sculpt before exporting.'
    elif pending_wrinkles and not wrinkle_jobs.running() and not detail_jobs.running():
        status='Wrinkle sculpts need updating. Choose Update Retained Wrinkles before exporting.'
    if status:
        box = layout.box()
        box.alert = f.last_action_error or (pending_detail and not detail_jobs.running())
        text(box, status)


class CM_OT_use_custom_maps(bpy.types.Operator):
    bl_idname='cm.use_custom_maps'
    bl_label='Use Current Files as Custom Maps'
    bl_description='Declare that the current map files were authored for this geometry; preserve their pixels and record them as custom assets'

    def execute(self,context):
        from . import map_provenance,bake_jobs
        try:
            if bake_jobs.running():raise ValueError('Finish or cancel the map bake before assigning custom files')
            f=context.scene.cm_features
            if context.mode!='OBJECT' or f.body_editing or f.full_sculpt or f.restore_editing or f.expression_active:
                raise ValueError('Finish the active edit before assigning custom maps')
            count=map_provenance.adopt_custom(context.scene)
            s,f=context.scene.cm_settings,context.scene.cm_features
            problems=check_delivery(context,s,f)
            f.delivery_report='\n'.join(problems) if problems else 'Character checks passed.'
            f.last_action_error=bool(problems)
            s.status=str(count)+' map set(s) assigned as custom files for this geometry.'
            return {'FINISHED'}
        except Exception as exc:
            self.report({'ERROR'},str(exc));return {'CANCELLED'}


class CM_OT_focus(bpy.types.Operator):
    bl_idname = 'cm.focus_character'
    bl_label = 'Focus Character'
    bl_description = 'Frame the working character and recover an incomplete workspace'
    bl_options = {'REGISTER'}

    def execute(self, context):
        return bpy.ops.cm.guided(action='FOCUS')


CLASSES = (CM_OT_guided,CM_OT_use_custom_maps,CM_OT_focus)


def draw_focus_menu(self, context):
    self.layout.operator('cm.focus_character', icon='VIEWZOOM')


def draw_glb_export_menu(self, context):
    self.layout.operator('cm.export_static_glb', text='MetaHuman Maker Static GLB (.glb)', icon='EXPORT')


def register_menus():
    global _FOCUS_MENU_REGISTERED
    if not globals().get('_FOCUS_MENU_REGISTERED', False):
        bpy.types.VIEW3D_MT_view.append(draw_focus_menu)
        bpy.types.TOPBAR_MT_file_export.append(draw_glb_export_menu)
        _FOCUS_MENU_REGISTERED = True


def unregister_menus():
    global _FOCUS_MENU_REGISTERED
    if globals().get('_FOCUS_MENU_REGISTERED', False):
        bpy.types.VIEW3D_MT_view.remove(draw_focus_menu)
        bpy.types.TOPBAR_MT_file_export.remove(draw_glb_export_menu)
        _FOCUS_MENU_REGISTERED = False
