"""Blender adapters for registration, detail transfer and stable mesh export."""

from contextlib import contextmanager
from pathlib import Path
import json
import numpy as np
import bpy
from mathutils import Vector, Matrix
from mathutils.bvhtree import BVHTree
from . import geometry


def transform_state(obj):
    fields=('location','scale','rotation_euler','rotation_quaternion','rotation_axis_angle')
    return dict(rotation_mode=obj.rotation_mode,**{name:list(getattr(obj,name)) for name in fields})


def restore_transform(obj,state):
    for name,value in state.items():
        setattr(obj,name,value)


def coordinates(obj, world=True):
    values = np.empty(len(obj.data.vertices) * 3, dtype=np.float64)
    obj.data.vertices.foreach_get("co", values)
    values = values.reshape(-1, 3)
    if world:
        m = np.asarray(obj.matrix_world)
        values = values @ m[:3, :3].T + m[:3, 3]
    return values


def to_local(obj, world):
    m = np.asarray(obj.matrix_world.inverted())
    return np.asarray(world) @ m[:3, :3].T + m[:3, 3]


def edges(obj):
    result = np.empty(len(obj.data.edges) * 2, dtype=np.int32)
    obj.data.edges.foreach_get("vertices", result)
    return result.reshape(-1, 2)


def topology(obj):
    return topology_data(obj.data)


def topology_data(data):
    sizes = np.empty(len(data.polygons), dtype=np.int64)
    starts = np.empty(len(data.polygons), dtype=np.int64)
    vertices = np.empty(len(data.loops), dtype=np.int64)
    data.polygons.foreach_get('loop_total', sizes)
    data.polygons.foreach_get('loop_start', starts)
    data.loops.foreach_get('vertex_index', vertices)
    return geometry.topology_arrays_hash(len(data.vertices), sizes, vertices, starts)


def include_in_local_views(context, obj):
    if context.screen:
        for area in context.screen.areas:
            if area.type == "VIEW_3D" and area.spaces.active.local_view:
                obj.local_view_set(area.spaces.active, True)


def activate(context, obj):
    if context.object and context.object.mode != "OBJECT":
        bpy.ops.object.mode_set(mode="OBJECT")
    bpy.ops.object.select_all(action="DESELECT")
    obj.hide_set(False)
    include_in_local_views(context, obj)
    obj.select_set(True)
    context.view_layer.objects.active = obj


def check_mesh(obj, label):
    if obj is None or obj.type != "MESH" or not obj.data.polygons:
        raise ValueError(f"Choose a polygon mesh for {label}")
    if abs(obj.matrix_world.determinant()) < 1e-12:
        raise ValueError(f"{label} has zero scale")


def check_inputs(settings):
    check_mesh(settings.source, "Source")
    check_mesh(settings.template, "Template")
    if settings.source == settings.template:
        raise ValueError("Source and Template must be different objects")
    if settings.template.modifiers:
        raise ValueError(
            "Use a template without modifiers so vertex order remains stable"
        )
    if len(settings.template.data.vertices) > 250000:
        raise ValueError(
            "Template exceeds 250,000 vertices; use a base-resolution template"
        )


def source_bvh(context, obj):
    evaluated = obj.evaluated_get(context.evaluated_depsgraph_get())
    mesh = evaluated.to_mesh()
    try:
        m = evaluated.matrix_world
        return BVHTree.FromPolygons(
            [m @ v.co for v in mesh.vertices], [p.vertices[:] for p in mesh.polygons]
        )
    finally:
        evaluated.to_mesh_clear()


def nearest(bvh, vertices, max_distance=float("inf"), progress=None):
    result = np.asarray(vertices).copy()
    distances = np.full(len(vertices), np.inf)
    per_vertex = np.ndim(max_distance) != 0
    if per_vertex:
        max_distance = np.asarray(max_distance, dtype=float)
        if max_distance.shape != (len(vertices),) or np.isnan(max_distance).any() or np.any(max_distance < 0):
            raise ValueError('Projection distances must match the vertex count and be nonnegative')
    for i, p in enumerate(vertices):
        hit, _, _, distance = bvh.find_nearest(Vector(p), float(max_distance[i]) if per_vertex else max_distance)
        if hit is not None:
            result[i] = hit
            distances[i] = distance
        if progress and (i+1) % 16384 == 0:
            progress(i+1, len(vertices))
    if progress:
        progress(len(vertices), len(vertices))
    return result, distances


def landmark_points(settings):
    source, template = [], []
    for pair in settings.landmarks:
        if pair.source and pair.template:
            if (
                pair.source.parent != settings.source
                or pair.template.parent != settings.template
            ):
                raise ValueError(
                    "Landmarks belong to another Source/Template; clear and place them again"
                )
            source.append(pair.source.matrix_world.translation[:])
            template.append(pair.template.matrix_world.translation[:])
    return np.asarray(source), np.asarray(template)


def align_source(context, settings, bounds=False):
    check_inputs(settings)
    source, target = landmark_points(settings)
    if bounds:
        a, b = coordinates(settings.source), coordinates(settings.template)
        size_a, size_b = np.ptp(a, axis=0), np.ptp(b, axis=0)
        scale = max(size_b) / max(max(size_a), 1e-9)
        rotation = np.eye(3)
        translation = (b.min(0) + b.max(0)) / 2 - (a.min(0) + a.max(0)) / 2 * scale
    else:
        rotation, scale, translation = geometry.similarity(source, target)
    transform = np.eye(4)
    transform[:3, :3] = rotation.T * scale
    transform[:3, 3] = translation
    settings.source.matrix_world = (
        Matrix(transform.tolist()) @ settings.source.matrix_world
    )
    if 'cm_head_placement' in settings.source:
        del settings.source['cm_head_placement']
    context.view_layer.update()


def pin_weights(obj):
    result = np.zeros(len(obj.data.vertices))
    group = obj.vertex_groups.get("CM_Pin")
    if group:
        for vertex in obj.data.vertices:
            for weight in vertex.groups:
                if weight.group == group.index:
                    result[vertex.index] = weight.weight
    return result


def duplicate(context, obj, name):
    result = obj.copy()
    result.data = obj.data.copy()
    result.name = name
    context.collection.objects.link(result)
    if result.data.shape_keys:
        key_name = result.data.shape_keys.name
        key_pointer = result.data.shape_keys.as_pointer()
        result.shape_key_clear()
        # Some Blender versions retain the copied Key; others free it here.
        # Resolve a surviving ID before touching it, never reuse a freed wrapper.
        orphan_key = bpy.data.shape_keys.get(key_name)
        if (result.data.shape_keys is None and orphan_key is not None
                and orphan_key.as_pointer() == key_pointer and orphan_key.users == 0):
            bpy.data.batch_remove(ids=(orphan_key,))
    for modifier in list(result.modifiers):
        result.modifiers.remove(modifier)
    result.hide_set(False)
    result.hide_render = False
    return result


def generated_ear_regions(context, source, template, base):
    """Restrict ear correspondence to an explicitly marked source auricle."""
    if not source.vertex_groups.get('CM_Ears'):
        return []
    groups={g.index for g in template.vertex_groups if g.name.startswith('FACIAL_') and '_Ear' in g.name}
    explicit = template.vertex_groups.get('CM_Ears')
    if not groups and not explicit:
        return []
    if explicit:
        from .workflow import group_weights
        weights=group_weights(template,'CM_Ears')
    else:
        weights=np.asarray([min(1.,sum(w.weight for w in v.groups if w.group in groups)) for v in template.data.vertices])
    evaluated=source.evaluated_get(context.evaluated_depsgraph_get())
    mesh=evaluated.to_mesh()
    try:
        if len(mesh.vertices)!=len(source.data.vertices):
            return []  # Changed source topology needs a new semantic assignment.
        group=source.vertex_groups['CM_Ears'].index
        source_weights=np.asarray([sum(w.weight for w in v.groups if w.group==group) for v in mesh.vertices])
        coordinates=np.asarray([source.matrix_world @ v.co for v in mesh.vertices])
        lateral=0 if context.scene.cm_features.front_axis in {'Y','-Y'} else 1
        marked=coordinates[source_weights>.5,lateral]
        target_marked=base[weights>.5,lateral]
        if len(marked)<16 or len(target_marked)<16:
            raise ValueError('Mark both source ears with CM_Ears weights above 0.5 before ear-restricted fitting')
        source_center=float((marked.min()+marked.max())*.5)
        target_center=float((target_marked.min()+target_marked.max())*.5)
        regions=[]
        for side in (-1,1):
            faces=[p.vertices[:] for p in mesh.polygons if (source_weights[list(p.vertices)]>.5).sum()>=2
                and side*(coordinates[list(p.vertices),lateral].mean()-source_center)>0]
            indices=np.flatnonzero((weights>.005)&(side*(base[:,lateral]-target_center)>0))
            if faces and len(indices):
                source_core=coordinates[(source_weights>.5)&(side*(coordinates[:,lateral]-source_center)>0)]
                target_core=base[(weights>.9)&(side*(base[:,lateral]-target_center)>0)]
                registered=None
                if len(source_core)>=16 and len(target_core)>=16:
                    src_low,src_high=np.percentile(source_core,[1,99],axis=0)
                    dst_low,dst_high=np.percentile(target_core,[1,99],axis=0)
                    scale=np.clip((src_high-src_low)/np.maximum(dst_high-dst_low,1e-8),.75,1.3)
                    registered=(base[indices]-(dst_low+dst_high)*.5)*scale+(src_low+src_high)*.5
                regions.append((BVHTree.FromPolygons(coordinates,faces),indices,weights[indices],registered))
        return regions
    finally:
        evaluated.to_mesh_clear()


def wrap(context, settings):
    """Publish a complete fit, or retain the previous character and editor state."""
    check_inputs(settings)
    if context.mode != 'OBJECT':
        raise ValueError('Finish the active edit before wrapping')
    features = getattr(context.scene, 'cm_features', None)
    if features and (features.expression_active or features.restore_editing or features.body_editing or features.full_sculpt):
        raise ValueError('Finish the active edit and return to Neutral before wrapping')
    previous = settings.result
    parts = (features.head_result, features.body_result) if features else ()
    accessories = [obj for obj in (features.eye_left, features.eye_right, features.teeth) if obj] if features else []
    actors = list(dict.fromkeys(obj for obj in (settings.source, settings.template, previous, *parts, *accessories) if obj))
    visibility = [(obj, obj.hide_get(), obj.hide_render) for obj in actors]
    following = [(obj, transform_state(obj), {
        key: list(obj[key]) if key == 'cm_follow_original' and key in obj else obj.get(key)
        for key in ('cm_follow_original', 'cm_follow_anchors')}) for obj in accessories]
    missing = object()
    fields = {name: getattr(settings, name, missing) for name in ('stage', 'last_error')}
    with selection_state(context, on_error=True):
        try:
            return _wrap(context, settings)
        except Exception:
            settings.result = previous
            if features:
                features.head_result, features.body_result = parts
            for obj, transforms, properties in following:
                restore_transform(obj, transforms)
                for key, value in properties.items():
                    if value is None:
                        if key in obj:del obj[key]
                    else:obj[key] = value
            for obj, hidden, render in visibility:
                obj.hide_set(hidden)
                obj.hide_render = render
            for name, value in fields.items():
                if value is missing:
                    if hasattr(settings, name):delattr(settings, name)
                else:setattr(settings, name, value)
            context.view_layer.update()
            raise


def _wrap(context, settings):
    check_inputs(settings)
    previous_result = settings.result
    base = coordinates(settings.template)
    current = base.copy()
    span = np.linalg.norm(np.ptp(base, axis=0))
    if span < 1e-9:
        raise ValueError("Template has no spatial extent")
    source_points, template_points = landmark_points(settings)
    if len(source_points):
        if len(source_points) > 512:
            raise ValueError("Use at most 512 landmark pairs")
        if len(source_points) < 3:
            raise ValueError(
                "Complete at least three landmark pairs, or clear all to use surface fit"
            )
        geometry.similarity(template_points, source_points)  # reject collinear controls
        current = geometry.rbf_displacement(
            base, template_points, source_points, span * settings.landmark_radius,
            regularization=getattr(settings, 'landmark_regularization', 1e-6),
        )
        if settings.source.get('cm_gnm'):
            current = geometry.limit_extrapolation(base, current, template_points, source_points)
    if hasattr(settings, 'local_landmarks'):
        controls, destinations = settings.local_landmarks
        # Fit clustered face controls at head scale after body placement. A
        # body-sized kernel for adjacent face points is poorly conditioned and
        # can create large unsupported offsets on the scalp and torso.
        if len(source_points):
            controls = geometry.rbf_displacement(controls, template_points,
                source_points, span * settings.landmark_radius,
                regularization=getattr(settings, 'landmark_regularization', 1e-6))
        refined = geometry.rbf_displacement(current, controls, destinations,
            settings.local_landmark_radius,
            regularization=getattr(settings, 'landmark_regularization', 1e-6))
        refined = geometry.limit_extrapolation(current, refined, controls, destinations)
        current += (refined - current) * settings.local_landmark_weights[:, None]
    features = getattr(context.scene, "cm_features", None)
    from . import wrap_proxy
    proxy = wrap_proxy.source(context,settings,features)
    bvh = source_bvh(context, proxy)
    ear_regions=generated_ear_regions(context,settings.source,settings.template,base) if settings.part in {'HEAD','FULL_BODY'} else []
    for _,indices,influence,registered in ear_regions:
        if registered is not None:
            current[indices]=current[indices]*(1-influence[:,None])+registered*influence[:,None]
    reference = current.copy()
    mesh_edges = edges(settings.template)
    interior = np.zeros(len(base), dtype=bool)
    if features and getattr(features, 'preserve_mouth_interior', True):
        if hasattr(settings, 'mouth_interior'):
            interior = settings.mouth_interior
        elif settings.part == 'HEAD':
            from .wrap_interiors import mouth_mask
            interior = mouth_mask(settings.template)
    pins = pin_weights(settings.template)
    protected_neck = False
    if (settings.part == 'HEAD' and settings.source.get('cm_gnm')
            and (settings.template.get('cm_template_library')
                 or settings.template.get('cm_library_role') == 'head')):
        # GNM is a face/neck source, not a matching MetaHuman body boundary.
        # Retain the licensed template's attachment surface and blend smoothly
        # into face fitting. Never write these constraints into artist groups.
        from .workflow import neck_ring
        ring = neck_ring(settings.template)
        neck_top = float(base[ring, 2].max())
        blend_height = max(float(np.ptp(base[:, 2])) * .22, 1e-8)
        neck = np.clip((neck_top + blend_height - base[:, 2]) / blend_height, 0, 1)
        pins = np.maximum(pins, neck)
        protected_neck = True
    if features and features.conservative:
        from .workflow import group_weights

        pins = np.maximum(pins, group_weights(settings.template, "CM_Inner_Parts"))
    if features and features.is_mask:
        from .workflow import group_weights

        mask_group = settings.template.vertex_groups.get("CM_WrapMask")
        if mask_group:
            pins = np.maximum(pins, 1 - group_weights(settings.template, "CM_WrapMask"))
        else:
            axis = 1 if features.front_axis in {"Y", "-Y"} else 0
            sign = -1 if features.front_axis.startswith("-") else 1
            center = (base[:, axis].min() + base[:, axis].max()) * 0.5
            pins = np.maximum(pins, (sign * (base[:, axis] - center) < 0).astype(float))
    # FBX duplicates vertices at UV/material boundaries. Relaxing the copies
    # independently tears the physical surface even though topology stays valid.
    # Synchronize displacement only; keep every original vertex and UV intact.
    _, weld, weld_counts = np.unique(base, axis=0, return_inverse=True, return_counts=True)
    has_splits = len(weld_counts) != len(base)
    if has_splits:
        shared_pins = np.zeros(len(weld_counts))
        np.maximum.at(shared_pins, weld, pins)
        pins = shared_pins[weld]
    limit = getattr(settings, 'projection_limits', span * settings.max_distance)
    iterations = settings.iterations
    conservative = bool(features and features.conservative)
    if features and features.wrap_quality == "FASTER":
        iterations = max(3, iterations // 2)
    if features and features.wrap_quality == "ACCURATE":
        iterations = min(100, iterations * 2)
    if conservative:
        limit *= 0.4
    surface_guard = None
    if conservative:
        # Start hard pins at their exact constraints. A backtracked step then
        # retains them, rather than interpolating from an unconstrained RBF fit.
        if has_splits:
            delta = current - base
            shared = np.column_stack([np.bincount(weld, weights=delta[:, axis], minlength=len(weld_counts))
                                      / weld_counts for axis in range(3)])
            current = base + shared[weld]
        current = base * pins[:, None] + current * (1 - pins[:, None])
        settings.template.data.calc_loop_triangles()
        triangles = np.empty((len(settings.template.data.loop_triangles), 3), dtype=np.int32)
        settings.template.data.loop_triangles.foreach_get('vertices', triangles.ravel())
        surface_guard = geometry.SurfaceStepGuard(current, triangles, weld if has_splits else None)
    oral_guard = None
    if interior.any():
        settings.template.data.calc_loop_triangles()
        oral_triangles = np.empty((len(settings.template.data.loop_triangles), 3), dtype=np.int32)
        settings.template.data.loop_triangles.foreach_get('vertices', oral_triangles.ravel())
        # Guard the head's complete triangle neighborhood. Backtracking only
        # the cavity can create a new thin face just outside its lip boundary.
        oral_region = getattr(settings, 'mouth_guard_region', np.ones(len(base), dtype=bool))
        oral_triangles = oral_triangles[np.all(oral_region[oral_triangles], axis=1)]
        floor = 8e-10 / context.scene.unit_settings.scale_length ** 2
        oral_guard = geometry.SurfaceStepGuard(current, oral_triangles,
            weld if has_splits else None, minimum_double_area=floor)
    def guard_oral(previous, proposed):
        guarded = oral_guard.limit(previous, proposed)
        guarded[~oral_region] = proposed[~oral_region]
        return guarded
    context.window_manager.progress_begin(0, iterations)
    try:
        for iteration in range(iterations):
            preceding = current
            projected, _ = nearest(bvh, current, limit)
            projected[interior] = current[interior]
            for ear_bvh, indices, influence, _ in ear_regions:
                ear, _ = nearest(ear_bvh,current[indices],limit[indices] if np.ndim(limit) else limit)
                projected[indices]=projected[indices]*(1-influence[:,None])+ear*influence[:,None]
            step = current + settings.projection_strength * (projected - current)
            current = geometry.relax_displacement(
                step,
                reference,
                mesh_edges,
                max(settings.smoothing, 0.65) if conservative else settings.smoothing,
            )
            if has_splits:
                delta = current - base
                shared = np.column_stack([np.bincount(weld, weights=delta[:,axis], minlength=len(weld_counts))
                    / weld_counts for axis in range(3)])
                current = base + shared[weld]
            # Pins are exact constraints relative to the original template, not cumulative blends.
            current = base * pins[:, None] + current * (1 - pins[:, None])
            if surface_guard is not None:
                current = surface_guard.limit(preceding, current)
            if oral_guard is not None:
                current = guard_oral(preceding, current)
            context.window_manager.progress_update(iteration + 1)
    finally:
        context.window_manager.progress_end()
    if interior.any():
        from .wrap_interiors import follow_boundary
        preceding = current
        current = follow_boundary(reference, current, mesh_edges, interior,weld=weld if has_splits else None)
        current[interior] = (base[interior] * pins[interior, None]
            + current[interior] * (1 - pins[interior, None]))
        current = guard_oral(preceding, current)
    from . import region_projection
    regional_report = []
    if region_projection.paired(settings.source, settings.template):
        allowed = (1-pins)*(~interior)
        current, regional_report = region_projection.refine_paired(context, settings.source,
            settings.template, current, iterations=min(iterations, 16), limits=allowed)
    if not np.isfinite(current).all():
        raise ValueError(
            "Solver produced invalid coordinates; adjust landmarks and distance"
        )
    result = duplicate(context, settings.template, settings.character_name + "_Wrapped")
    try:
        result.shape_key_add(name="Basis")
        key = result.shape_key_add(name="Wrapped")
        key.data.foreach_set("co", to_local(result, current).ravel())
        key.value = 1
        result["cm_template_topology"] = topology(settings.template)
        result["cm_part"] = settings.part
        result["cm_role"] = "RESULT"
        if regional_report:
            result['cm_region_projection'] = json.dumps(regional_report)
        if surface_guard is not None:
            result['cm_conservative_surface'] = json.dumps({
                'limited_steps': surface_guard.limited_steps,
                'limited_vertex_steps': surface_guard.limited_vertices,
                'iterations': iterations})
        elif 'cm_conservative_surface' in result:
            del result['cm_conservative_surface']
        if protected_neck:
            result['cm_generated_neck_protected'] = True
        if ear_regions:
            result['cm_generated_ear_regions'] = len(ear_regions)
            result['cm_source_ear_regions'] = len(ear_regions)
        if settings.part == "BODY":
            result["cm_body_source"] = "SCAN"
        from .guided import signature

        result["cm_source_signature"] = signature(settings.source)
        result["cm_template_signature"] = signature(settings.template)
        if topology(result) != topology(settings.template):
            raise RuntimeError("Topology changed unexpectedly")
        settings.result = result
        if features:
            if settings.part == "HEAD":
                features.head_result = result
            elif settings.part == "BODY":
                features.body_result = result
        projected, distances = nearest(bvh, current)
        settings.last_error = float(np.sqrt(np.mean(distances**2)))
        settings.stage = "REFINE"
        if (
            features
            and features.full_head
            and settings.part == "HEAD"
            and any((features.eye_left, features.eye_right, features.teeth))
        ):
            from .workflow import update_accessories

            update_accessories(context, settings, features)
        activate(context, result)
        settings.source.hide_render = True
        settings.template.hide_render = True
        result.hide_render = False
        if (
            previous_result
            and previous_result != result
            and previous_result.get("cm_part") == settings.part
        ):
            previous_result.hide_set(True)
            previous_result.hide_render = True
        return result
    except Exception:
        data = result.data
        bpy.data.objects.remove(result, do_unlink=True)
        if not data.users:
            bpy.data.meshes.remove(data)
        settings.result = previous_result
        raise


def evaluated_coordinates(context, obj):
    evaluated = obj.evaluated_get(context.evaluated_depsgraph_get())
    mesh = evaluated.to_mesh()
    try:
        m = evaluated.matrix_world
        return np.asarray([m @ v.co for v in mesh.vertices], dtype=np.float64)
    finally:
        evaluated.to_mesh_clear()


def refine_surface(context, settings, features):
    """Refine the accepted surface without flattening recognized oral cavities."""
    from . import character_project, workflow, wrap_interiors
    character_project.preflight(context, settings, features)
    obj = (features.body_result if features.body_result and context.active_object == features.body_result
           else settings.result)
    check_mesh(obj, 'Result')
    check_mesh(settings.source, 'Source')
    if obj == settings.source:
        raise ValueError('Source and result must be separate meshes')
    if any(modifier.show_viewport and modifier.type not in {'WEIGHTED_NORMAL', 'NORMAL_EDIT', 'DATA_TRANSFER'}
           for modifier in obj.modifiers):
        raise ValueError('Apply or disable geometry modifiers before refining the base surface')
    evaluated = obj.evaluated_get(context.evaluated_depsgraph_get())
    mesh = evaluated.to_mesh()
    try:
        if topology_data(mesh) != topology(obj):
            raise ValueError('Disable topology-changing modifiers before refining the base surface')
    finally:
        evaluated.to_mesh_clear()
    if collapsed_faces(context, obj):
        raise ValueError('Repair collapsed faces before refining the surface')
    base = evaluated_coordinates(context, obj)
    rest = coordinates(obj)
    influence = 1 - pin_weights(obj)
    if features.is_mask:
        if obj.vertex_groups.get('CM_WrapMask'):
            influence *= workflow.group_weights(obj, 'CM_WrapMask')
        else:
            axis = 1 if features.front_axis in {'Y', '-Y'} else 0
            sign = -1 if features.front_axis.startswith('-') else 1
            influence *= sign * (base[:, axis] - (base[:, axis].min() + base[:, axis].max()) * .5) >= 0
    if settings.part == 'FULL_BODY' and features.body_result and obj == features.head_result:
        influence[workflow.neck_ring(obj)] = 0
    _, families, counts = np.unique(rest, axis=0, return_inverse=True, return_counts=True)
    shared = np.ones(len(counts))
    np.minimum.at(shared, families, influence)
    influence = shared[families]
    from . import region_projection
    has_pairs = bool(region_projection.paired(settings.source, obj))
    projected, _ = nearest(source_bvh(context, settings.source), base,
        np.linalg.norm(np.ptp(base, axis=0)) * settings.max_distance)
    is_body = obj == features.body_result or settings.part == 'BODY'
    regions = generated_ear_regions(context, settings.source, obj, base) if not is_body else []
    for tree, indices, weights, _registered in regions:
        ear, _ = nearest(tree, base[indices])
        projected[indices] = projected[indices] * (1 - weights[:, None]) + ear * weights[:, None]
    interior = (wrap_interiors.mouth_mask(obj) if not is_body
                and features.preserve_mouth_interior else np.zeros(len(base), dtype=bool))
    projected[interior] = base[interior]
    candidate = base + (projected - base) * (settings.projection_strength * influence[:, None])
    regional_report = []
    if has_pairs:
        # Explicit patch labels constrain this action to those regions. Do not
        # globally project the rest of an already accepted character.
        candidate, regional_report = region_projection.refine_paired(context, settings.source,
            obj, base, iterations=min(settings.iterations, 24), limits=influence*(~interior))
    if interior.any():
        # Extend the lip boundary's displacement into the existing cavity.
        candidate = wrap_interiors.follow_boundary(base, candidate, edges(obj), interior)
        candidate[interior] = base[interior] + (candidate[interior] - base[interior]) * influence[interior, None]
    delta = candidate - base
    average = np.column_stack([np.bincount(families, weights=delta[:, axis], minlength=len(counts))
                               / counts for axis in range(3)])
    candidate = base + average[families]
    obj.data.calc_loop_triangles()
    triangles = np.empty((len(obj.data.loop_triangles), 3), dtype=np.int32)
    obj.data.loop_triangles.foreach_get('vertices', triangles.ravel())
    floor = 2.5e-10 / context.scene.unit_settings.scale_length ** 2
    guard = geometry.SurfaceStepGuard(base, triangles, families, minimum_double_area=floor)
    candidate = guard.limit(base, candidate)
    if np.linalg.norm(candidate - base, axis=1).max() < 1e-9:
        if regional_report:
            obj['cm_region_projection'] = json.dumps(regional_report)
        settings.status = 'No surface refinement needed within the current constraints.'
        return False
    from . import native_surface_commit
    native_report=native_surface_commit.commit(context,obj,'Refine',candidate,reference=base,movable=influence>0)
    obj['cm_region_projection_native']=json.dumps(native_report)
    settings.status = 'Surface refined. Review the fit and refresh changed maps before delivery.'
    if regional_report:
        obj['cm_region_projection'] = json.dumps(regional_report)
        settings.status = 'Paired regions refined; source unchanged. Review geometry before refreshing maps.'
    return True


def collapsed_faces(context, obj, *, require_base_topology=False):
    """Find evaluated faces below Creator's 1e-6 square-centimeter area floor.

    Evaluate shape keys and modifiers, and triangulate that evaluated surface.
    Use float32 centimeter positions, as Unreal's MeshDescription check does.
    Returned polygon indices are usable for selection only when topology is
    unchanged; callers must not map changed topology back onto the base mesh.
    """
    evaluated=obj.evaluated_get(context.evaluated_depsgraph_get())
    mesh=evaluated.to_mesh()
    try:
        if require_base_topology and topology_data(mesh)!=topology(obj):
            raise ValueError('Disable topology-changing modifiers before selecting collapsed faces')
        mesh.calc_loop_triangles()
        if not mesh.loop_triangles:return []
        local=np.empty((len(mesh.vertices),3),dtype=np.float64)
        mesh.vertices.foreach_get('co',local.ravel())
        matrix=np.asarray(evaluated.matrix_world)
        world=local@matrix[:3,:3].T+matrix[:3,3]
        world=np.asarray(world*(context.scene.unit_settings.scale_length*100),dtype=np.float32)
        if not np.isfinite(world).all():raise ValueError(obj.name+': invalid vertex coordinates')
        indices=np.empty((len(mesh.loop_triangles),3),dtype=np.int32)
        polygons=np.empty(len(mesh.loop_triangles),dtype=np.int32)
        mesh.loop_triangles.foreach_get('vertices',indices.ravel())
        mesh.loop_triangles.foreach_get('polygon_index',polygons)
        tri=world[indices]
        area=np.linalg.norm(np.cross(tri[:,1]-tri[:,0],tri[:,2]-tri[:,0]),axis=1)*.5
        return np.unique(polygons[area<=1e-6]).tolist()
    finally:evaluated.to_mesh_clear()


def create_detail(context, settings, project_source=True, *, subdivision_type='CATMULL_CLARK',vertex_groups=None):
    check_mesh(settings.result, "Wrapped result")
    if subdivision_type not in {'CATMULL_CLARK','SIMPLE'}:raise ValueError('Choose a supported sculpt subdivision type')
    if project_source:
        check_mesh(settings.source, "Source")
    result = duplicate(context, settings.result, settings.character_name + "_Detail")
    result.hide_render = True
    try:
        # Start from the visible wrapped shape, not the original Basis geometry.
        result.data.vertices.foreach_set(
            "co",
            to_local(result, evaluated_coordinates(context, settings.result)).ravel(),
        )
        result.data.update()
        if vertex_groups:
            from .workflow import write_group
            for name,values in vertex_groups.items():
                values=np.asarray(values,dtype=float)
                if values.shape!=(len(result.data.vertices),) or not np.isfinite(values).all() or np.any((values<0)|(values>1)):raise ValueError('Invalid source-detail support group')
                write_group(result,name,values)
        activate(context, result)
        modifier = result.modifiers.new("CM_Subdivision", "SUBSURF")
        modifier.subdivision_type=subdivision_type
        modifier.levels = settings.detail_levels
        bpy.ops.object.modifier_apply(modifier=modifier.name)
        result.shape_key_add(name="Basis")
        key = result.shape_key_add(name="Scan Detail")
        base = coordinates(result)
        from . import detail_layers
        result['cm_detail_level'] = settings.detail_levels
        result['cm_detail_subdivision_steps'] = [settings.detail_levels]
        result['cm_detail_subdivision_type']=subdivision_type
        detail_layers.bind(context, settings.result, result, to_local(result, base))
        projected = base
        if project_source:
            projected, _ = nearest(
                source_bvh(context, settings.source),
                base,
                np.linalg.norm(np.ptp(base, axis=0)) * settings.max_distance,
                progress=getattr(settings, 'scan_progress', None),
            )
        key.data.foreach_set("co", to_local(result, projected).ravel())
        key.value = settings.detail_strength if project_source else 1
        settings.detail = result
        result["cm_detail_level"] = settings.detail_levels
        result["cm_role"] = "DETAIL"
        result["cm_scan_applied"] = not project_source
        settings.stage = "DETAIL"
        return result
    except Exception:
        bpy.data.objects.remove(result, do_unlink=True)
        raise


@contextmanager
def selection_state(context, *, on_error=False):
    selected = list(context.selected_objects)
    active = context.view_layer.objects.active
    hidden = [(o, o.hide_get()) for o in context.view_layer.objects]
    local = []
    if context.screen:
        # Original datablocks can retain stale local-view flags in Blender 4.3.
        # Read the evaluated flags, including membership of hidden objects.
        depsgraph = context.evaluated_depsgraph_get()
        for area in context.screen.areas:
            space = area.spaces.active
            if area.type == "VIEW_3D" and space.local_view:
                local.extend(
                    (o, space, o.evaluated_get(depsgraph).local_view_get(space))
                    for o in context.view_layer.objects
                )
    restore = not on_error
    try:
        yield
    except BaseException:
        restore = True
        raise
    finally:
        if restore:
            # Hiding an object clears its selection. Restore visibility/local
            # membership before selecting it, including after failed creation.
            for obj, value in hidden:
                obj.hide_set(value)
            for obj, space, value in local:
                if obj.name in context.view_layer.objects:
                    obj.local_view_set(space, value)
            for obj in context.view_layer.objects:
                obj.select_set(False)
            for obj in selected:
                if obj.name in context.view_layer.objects:
                    obj.select_set(True)
            context.view_layer.objects.active = active


@contextmanager
def albedo_bake_source(source):
    """Bake base color independently of Principled specular energy compensation.

    Simple skin shaders retain their complete upstream color graph. Complex
    shader mixtures keep the existing diffuse-pass behavior rather than guessing
    which component represents their albedo. Source materials are always restored.
    """
    originals = list(source.data.materials)
    plans = []
    for material in originals:
        if not material or not material.use_nodes:
            plans.append(None)
            continue
        output = next((n for n in material.node_tree.nodes if n.type == 'OUTPUT_MATERIAL' and n.is_active_output), None)
        surface = output.inputs['Surface'] if output else None
        shader = surface.links[0].from_node if surface and surface.is_linked else None
        if shader is None or shader.type not in {'BSDF_PRINCIPLED', 'BSDF_DIFFUSE', 'EMISSION'}:
            yield 'DIFFUSE'
            return
        plans.append(shader.name)
    temporary = []
    try:
        for index, original in enumerate(originals or [None]):
            material = original.copy() if original else bpy.data.materials.new('CM_AlbedoSource')
            temporary.append(material)
            material.use_nodes = True
            nodes, links = material.node_tree.nodes, material.node_tree.links
            output = next(n for n in nodes if n.type == 'OUTPUT_MATERIAL' and n.is_active_output)
            emitter = nodes.new('ShaderNodeEmission')
            name = plans[index] if originals else None
            if name:
                shader = nodes[name]
                color = shader.inputs['Base Color' if shader.type == 'BSDF_PRINCIPLED' else 'Color']
                if color.is_linked:
                    links.new(color.links[0].from_socket, emitter.inputs['Color'])
                else:
                    emitter.inputs['Color'].default_value = color.default_value
            else:
                emitter.inputs['Color'].default_value = original.diffuse_color if original else (.8, .8, .8, 1)
            links.new(emitter.outputs[0], output.inputs['Surface'])
            if originals:
                source.data.materials[index] = material
            else:
                source.data.materials.append(material)
        yield 'EMIT'
    finally:
        for index, material in enumerate(originals):
            source.data.materials[index] = material
        if not originals:
            source.data.materials.clear()
        for material in temporary:
            bpy.data.materials.remove(material)


def bake(context, settings, kind):
    check_mesh(settings.result, "Wrapped result")
    source = (
        settings.detail if kind == "NORMAL" and settings.detail else settings.source
    )
    check_mesh(source, "Bake source")
    target = settings.result
    if target == source:
        raise ValueError("Bake source and result must differ")
    if not target.data.uv_layers:
        raise ValueError("Result needs the template's UV map before baking")
    folder = Path(bpy.path.abspath(settings.project_dir)) / "Maps"
    folder.mkdir(parents=True, exist_ok=True)
    image = bpy.data.images.new(
        f"{settings.character_name}_{kind}",
        width=int(settings.map_size),
        height=int(settings.map_size),
        alpha=False,
        float_buffer=False,
    )
    if kind != "ALBEDO":
        image.colorspace_settings.name = "Non-Color"
    # Bake into independent target materials; never insert nodes in the user's original materials.
    original_materials = list(target.data.materials)
    original_material_indices=np.empty(len(target.data.polygons),dtype=np.int32)
    target.data.polygons.foreach_get('material_index',original_material_indices)
    temporary = []
    scene = context.scene
    old_engine, old_device = scene.render.engine, scene.cycles.device
    bake_settings = scene.render.bake
    names = (
        "use_selected_to_active",
        "use_pass_direct",
        "use_pass_indirect",
        "use_pass_color",
        "cage_extrusion",
        "max_ray_distance",
        "margin",
        "normal_g",
        "use_cage",
        "cage_object",
        "normal_space",
        "normal_r",
        "normal_b",
        "use_clear",
    )
    old_bake = {n: getattr(bake_settings, n) for n in names}
    visibility = [(o, o.hide_render) for o in scene.objects]
    output = folder / f"{settings.character_name}_{kind}.png"
    completed = False
    try:
        with selection_state(context):
            activate(context, target)
            for index in range(max(1, len(original_materials))):
                material = bpy.data.materials.new("CM_BakeTarget")
                material.use_nodes = True
                node = material.node_tree.nodes.new("ShaderNodeTexImage")
                node.image = image
                material.node_tree.nodes.active = node
                temporary.append(material)
            target.data.materials.clear()
            for material in temporary:
                target.data.materials.append(material)
            for obj in scene.objects:
                if obj.type in {"MESH", "CURVE", "CURVES", "SURFACE", "META", "FONT"}:
                    obj.hide_render = obj not in (target, source)
            source.hide_set(False)
            include_in_local_views(context, source)
            source.select_set(True)
            scene.render.engine = "CYCLES"
            scene.cycles.device = "CPU"
            device_restore = None
            features = getattr(context.scene, "cm_features", None)
            if features and features.bake_device == "AUTO":
                preferences = context.preferences.addons["cycles"].preferences
                old_type = preferences.compute_device_type
                old_use = {d.id: d.use for d in preferences.devices}
                device_restore = (preferences, old_type, old_use)
                for backend in ("OPTIX", "CUDA", "HIP", "ONEAPI", "METAL"):
                    try:
                        preferences.compute_device_type = backend
                        preferences.refresh_devices()
                        available = [
                            d for d in preferences.devices if d.type == backend
                        ]
                        if available:
                            for device in preferences.devices:
                                device.use = device in available
                            scene.cycles.device = "GPU"
                            break
                    except (TypeError, RuntimeError):
                        continue
            bake_settings.use_selected_to_active = True
            bake_settings.use_pass_direct = False
            bake_settings.use_pass_indirect = False
            bake_settings.use_pass_color = True
            from . import normal_composition
            distance=normal_composition.projection_distance(target,settings.bake_distance) if kind=='NORMAL' else settings.bake_distance
            bake_settings.use_cage=False
            bake_settings.cage_object=None
            bake_settings.use_clear=True
            bake_settings.cage_extrusion = distance
            bake_settings.max_ray_distance = distance * 2
            bake_settings.margin = 12
            # Unreal expects DirectX tangent normals. Preview must invert green in Blender.
            bake_settings.normal_g = "NEG_Y"
            bake_settings.normal_space='TANGENT'
            bake_settings.normal_r='POS_X'
            bake_settings.normal_b='POS_Z'
            try:
                from . import bake_tiles
                from .bake_material_scope import worker_materials
                with worker_materials(context,source,target,kind):
                    if kind == 'ALBEDO':
                        with albedo_bake_source(source) as bake_type:
                            bake_tiles.bake(context, target, image, bake_type)
                    else:
                        bake_tiles.bake(context, target, image, {'CAVITY': 'DIFFUSE', 'NORMAL': 'NORMAL', 'AO': 'AO', 'ROUGHNESS':'ROUGHNESS'}[kind])
            finally:
                if device_restore:
                    preferences, old_type, old_use = device_restore
                    preferences.compute_device_type = old_type
                    preferences.refresh_devices()
                    for device in preferences.devices:
                        if device.id in old_use:
                            device.use = old_use[device.id]
            image.filepath_raw = str(output)
            image.file_format = "PNG"
            # A missed projection must not become an invalid (0,0,0) tangent normal.
            # Flat-color sources also have a known fallback color for uncovered texels.
            textured = any(
                mat
                and mat.use_nodes
                and any(n.type == "TEX_IMAGE" and n.image for n in mat.node_tree.nodes)
                for mat in source.data.materials
            )
            if kind == "NORMAL" or (kind == "ALBEDO" and not textured):
                from . import pixel_buffers
                with pixel_buffers.scratch((len(image.pixels)//4,4),directory=folder) as pixels:
                    image.pixels.foreach_get(pixels.ravel())
                    empty = np.all(pixels[:, :3] <= 1e-7, axis=1)
                    fallback = (0.5, 0.5, 1.0)
                    if kind == "ALBEDO":
                        colors = [
                            mat.node_tree.nodes.get("Principled BSDF")
                            .inputs["Base Color"]
                            .default_value[:3]
                            for mat in source.data.materials
                            if mat
                            and mat.use_nodes
                            and mat.node_tree.nodes.get("Principled BSDF")
                        ]
                        fallback = np.mean(colors, axis=0) if colors else (0.5, 0.5, 0.5)
                        if np.any(~empty):
                            # Match the bake buffer's encoding, which can differ from
                            # the linear color stored in the source shader.
                            fallback = np.median(pixels[~empty, :3], axis=0)
                    elif np.all(empty):
                        raise ValueError(
                            "Normal projection missed the entire source. Check alignment and increase Projection Distance."
                        )
                    pixels[empty, :3] = fallback
                    pixels[empty, 3] = 1
                    image.pixels.foreach_set(pixels.ravel())
            image.save()
            completed = True
        settings.status = f"Saved {output.name}"
        return output
    except Exception:
        bpy.data.images.remove(image)
        raise
    finally:
        scene.render.engine, scene.cycles.device = old_engine, old_device
        for name, value in old_bake.items():
            setattr(bake_settings, name, value)
        for obj, hidden in visibility:
            obj.hide_render = hidden
        target.data.materials.clear()
        for material in original_materials:
            target.data.materials.append(material)
        target.data.polygons.foreach_set('material_index',original_material_indices)
        for material in temporary:
            bpy.data.materials.remove(material)
        if completed and scene.get('cm_bake_worker') and not image.users:
            bpy.data.images.remove(image)


def export_bundle(context, settings, *, _root=None):
    from . import bake_jobs
    if bake_jobs.running():
        raise ValueError('Finish or cancel the background map bake before exporting')
    f = context.scene.cm_features
    if f.restore_editing:raise ValueError("Apply or cancel the active head repair before exporting")
    if f.body_editing:raise ValueError("Apply or cancel the active body pass before exporting")
    for obj in dict.fromkeys((settings.result,f.head_result,f.body_result,f.eye_left,f.eye_right,f.teeth)):
        if obj and obj.data.shape_keys:
            expression = obj.data.shape_keys.key_blocks.get("Expression Check")
            if expression and expression.value:
                raise ValueError("Return Expression Check to Neutral before exporting")
    if _root is None:
        from . import export_delivery
        return export_delivery.export(context, settings)
    if settings.part == "FULL_BODY":
        from .features import export_full_body

        return export_full_body(context, settings, context.scene.cm_features, root=_root)
    check_mesh(settings.result, "Wrapped result")
    if topology(settings.result) != settings.result.get("cm_template_topology", ""):
        raise ValueError(
            "Result topology changed after wrapping; restore the base topology"
        )
    from . import bridge
    import re

    name = re.sub(r"[^A-Za-z0-9_]", "_", settings.character_name).strip("_")
    if not name:
        raise ValueError("Enter a character name")
    root = Path(_root)
    root.mkdir(parents=True, exist_ok=True)
    target = settings.result
    vertices = evaluated_coordinates(context, target)
    if len(vertices) != len(target.data.vertices):
        raise ValueError(
            "Remove result modifiers that change vertex count before export"
        )
    # Explicit coordinates avoid FBX importer welding and coordinate ambiguity during conform.
    # Match this exporter's FBX round trip through Unreal: X, -Y, Z in centimeters.
    # Do not add an assumed "front" rotation; imported templates carry their own orientation.
    unit_scale = context.scene.unit_settings.scale_length * 100
    ue_vertices = vertices * np.asarray([1, -1, 1]) * unit_scale
    manifest = {
        "schema_version": 1,
        "name": name,
        "part": settings.part,
        "vertex_count": len(vertices),
        "topology_sha256": topology(target),
        "coordinate_system": "Unreal centimeters; Blender FBX round trip X,-Y,Z",
        "vertices": ue_vertices.tolist(),
        "mesh": f"{name}.fbx",
        "create_character": settings.create_character,
        "canonical_order": settings.canonical_order,
        "maps": [],
        "destination": settings.unreal_destination.rstrip("/") + "/" + name,
    }
    features = getattr(context.scene, "cm_features", None)
    if settings.part == 'HEAD' and features and features.eye_native_enabled:
        from .eye_appearance import creator_options
        manifest['native_eyes']=creator_options(features)
    if settings.part == "HEAD" and features and features.full_head:
        manifest["accessories"] = {}
        for role in ("eye_left", "eye_right", "teeth"):
            obj = getattr(features, role)
            if not obj:
                continue
            check_mesh(obj, role)
            filename = name + "_" + role + ".fbx"
            with selection_state(context):
                temporary = duplicate(context, obj, "CM_AccessoryExport")
                try:
                    values = evaluated_coordinates(context, obj)
                    if len(values) != len(temporary.data.vertices):
                        raise ValueError("Accessory modifiers changed vertex count")
                    if settings.canonical_order:
                        manifest.setdefault("accessory_vertices", {})[role] = (
                            values * np.asarray([1, -1, 1]) * unit_scale
                        ).tolist()
                    temporary.data.vertices.foreach_set(
                        "co", to_local(temporary, values).ravel()
                    )
                    temporary.data.update()
                    activate(context, temporary)
                    bpy.ops.export_scene.fbx(
                        filepath=str(root / filename),
                        use_selection=True,
                        object_types={"MESH"},
                        use_mesh_modifiers=False,
                        add_leaf_bones=False,
                        bake_anim=False,
                        mesh_smooth_type="FACE",
                        axis_forward="-Y",
                        axis_up="Z",
                    )
                finally:
                    data = temporary.data
                    bpy.data.objects.remove(temporary, do_unlink=True)
                    bpy.data.meshes.remove(data)
            manifest["accessories"][role] = filename
    with selection_state(context):
        # Export a baked duplicate to ensure current shape keys are reflected in the FBX.
        temp = duplicate(context, target, "CM_ExportTemp")
        try:
            temp.data.vertices.foreach_set("co", to_local(temp, vertices).ravel())
            temp.data.update()
            activate(context, temp)
            bpy.ops.export_scene.fbx(
                filepath=str(root / manifest["mesh"]),
                use_selection=True,
                object_types={"MESH"},
                use_mesh_modifiers=False,
                add_leaf_bones=False,
                bake_anim=False,
                mesh_smooth_type="FACE",
                axis_forward="-Y",
                axis_up="Z",
            )
        finally:
            data = temp.data
            bpy.data.objects.remove(temp, do_unlink=True)
            bpy.data.meshes.remove(data)
    import shutil

    maps = Path(bpy.path.abspath(settings.project_dir)) / "Maps"
    for path in sorted(maps.iterdir() if maps.is_dir() else ()):
        if not path.is_file() or path.suffix.lower() != '.png' or not path.name.startswith(settings.character_name + '_'):
            continue
        if path.stem.startswith(settings.character_name + "_Body_"):
            continue
        destination = root / "Maps" / path.name
        destination.parent.mkdir(exist_ok=True)
        shutil.copy2(path, destination)
        manifest["maps"].append("Maps/" + path.name)
    if settings.create_character and not settings.canonical_order and manifest.get('accessories'):
        from . import conform_geometry
        manifest['canonical_conform']=conform_geometry.prepare(context,settings,features)
    if any(Path(path).stem.endswith('_ROUGHNESS') for path in manifest['maps']):
        from . import roughness_response
        manifest['roughness_response']=roughness_response.profiles(context.scene)
        manifest.setdefault('support_files',[]).append('unreal_roughness.py')
        shutil.copy2(Path(bridge.__file__).with_name('unreal_roughness.py'),root/'unreal_roughness.py')
    animated_kinds={Path(path).stem.rsplit('_',1)[-1] for path in manifest['maps']}&{'CM1','CM2','CM3','WM1','WM2','WM3'}
    if settings.create_character and animated_kinds:
        from . import skin_bake_layers
        manifest['image_python']=bpy.path.abspath(features.inference_python)
        manifest['animated_encodings']={kind:(features.unreal_wrinkle_input_encoding if kind.startswith('WM') else features.unreal_color_input_encoding) for kind in animated_kinds}
        if any(kind.startswith('WM') and manifest['animated_encodings'][kind]=='FULL' for kind in animated_kinds):
            base=skin_bake_layers.native_wrinkle_baseline(context,[kind for kind in animated_kinds if kind.startswith('WM') and manifest['animated_encodings'][kind]=='FULL'])
            if base:
                target=root/'neutral-geometry-normal.png';shutil.copy2(base,target)
                import hashlib
                manifest['animated_normal_base']={'snapshot':target.name,'sha256':hashlib.sha256(target.read_bytes()).hexdigest()}
                manifest.setdefault('support_files',[]).append(target.name)
        for filename in ('unreal_native_animated.py','native_animated_worker.py','worker_lifetime.py'):
            shutil.copy2(Path(bridge.__file__).with_name(filename),root/filename)
            manifest.setdefault('support_files',[]).append(filename)
    (root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    shutil.copy2(
        Path(bridge.__file__).with_name("unreal_import.py"), root / "unreal_import.py"
    )
    settings.stage = "EXPORT"
    return root


def preview_material(context, settings):
    check_mesh(settings.result, "Result")
    folder = Path(bpy.path.abspath(settings.project_dir)) / "Maps"
    albedo = folder / f"{settings.character_name}_ALBEDO.png"
    if not albedo.exists():
        raise ValueError("Transfer the albedo before creating a preview material")
    material = bpy.data.materials.new(settings.character_name + "_Preview")
    material.use_nodes = True
    nodes, links = material.node_tree.nodes, material.node_tree.links
    bsdf = nodes.get("Principled BSDF")
    bsdf.inputs["Roughness"].default_value = 0.5
    color = nodes.new("ShaderNodeTexImage")
    color.name = "CM_Albedo"
    color["cm_map_kind"] = "ALBEDO"
    color.image = bpy.data.images.load(str(albedo), check_existing=False)
    color.image.alpha_mode = "NONE"
    color.location = (-600, 200)
    links.new(color.outputs["Color"], bsdf.inputs["Base Color"])
    normal_path = folder / f"{settings.character_name}_NORMAL.png"
    if normal_path.exists():
        normal = nodes.new("ShaderNodeTexImage")
        normal["cm_map_kind"] = "NORMAL"
        normal.image = bpy.data.images.load(str(normal_path), check_existing=False)
        normal.image.alpha_mode = "NONE"
        normal.image.colorspace_settings.name = "Non-Color"
        normal.location = (-800, -150)
        separate = nodes.new("ShaderNodeSeparateColor")
        separate.location = (-600, -150)
        combine = nodes.new("ShaderNodeCombineColor")
        combine.location = (-200, -150)
        invert = nodes.new("ShaderNodeMath")
        invert.operation = "SUBTRACT"
        invert.inputs[0].default_value = 1
        invert.location = (-400, -300)
        links.new(normal.outputs["Color"], separate.inputs["Color"])
        links.new(separate.outputs["Red"], combine.inputs["Red"])
        links.new(separate.outputs["Green"], invert.inputs[1])
        links.new(invert.outputs[0], combine.inputs["Green"])
        links.new(separate.outputs["Blue"], combine.inputs["Blue"])
        tangent = nodes.new("ShaderNodeNormalMap")
        tangent.location = (0, -150)
        links.new(combine.outputs["Color"], tangent.inputs["Color"])
        links.new(tangent.outputs["Normal"], bsdf.inputs["Normal"])
    roughness_path = folder / f"{settings.character_name}_ROUGHNESS.png"
    if roughness_path.exists():
        roughness = nodes.new("ShaderNodeTexImage")
        roughness["cm_map_kind"] = "ROUGHNESS"
        roughness.image = bpy.data.images.load(
            str(roughness_path), check_existing=False
        )
        roughness.image.colorspace_settings.name = "Non-Color"
        from . import lookdev
        lookdev.bind_roughness(material, roughness.image)
    # Make the albedo the paint target, not the normal image.
    nodes.active = color
    settings.result.data.materials.clear()
    settings.result.data.materials.append(material)
    for polygon in settings.result.data.polygons:
        polygon.material_index = 0
    activate(context, settings.result)
    if context.screen:
        for area in context.screen.areas:
            if area.type == "VIEW_3D":
                area.spaces.active.shading.type = "MATERIAL"
    return material
