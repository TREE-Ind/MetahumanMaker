"""Keep sculpt layers attached to their accepted surface across proportion edits."""

import hashlib
from contextlib import contextmanager
import bpy
import numpy as np
from . import mesh_ops as m

REFERENCE = ".cm_detail_reference"


def subdivision_type(detail):
    kind=detail.get('cm_detail_subdivision_type','CATMULL_CLARK')
    if kind not in {'CATMULL_CLARK','SIMPLE'}:raise ValueError('Invalid saved sculpt subdivision type; layer retained')
    return kind


def steps_for(detail):
    level = int(detail.get("cm_detail_level", 1))
    raw = detail.get("cm_detail_subdivision_steps")
    steps = [level] if raw is None else list(raw)
    if (
        not steps
        or len(steps) > 3
        or any(not isinstance(value, int) or not 1 <= value <= 3 for value in steps)
        or sum(steps) != level
    ):
        raise ValueError("Saved subdivision history is invalid; sculpt retained")
    return steps


@contextmanager
def feedback(context):
    window = context.window if not bpy.app.background else None
    workspace = context.workspace if window else None
    if window:
        window.cursor_modal_set("WAIT")
    if workspace:
        workspace.status_text_set(
            "Updating sculpt detail to follow the accepted character..."
        )
    try:
        if window:
            bpy.ops.wm.redraw_timer(type="DRAW_WIN_SWAP", iterations=1)
        yield
    finally:
        if window:
            window.cursor_modal_restore()
        if workspace:
            workspace.status_text_set(None)


def signature(context, source):
    digest = hashlib.sha256(
        np.asarray(m.evaluated_coordinates(context, source), dtype="<f4").tobytes()
    )
    for name in ("crease_edge", "crease_vert"):
        attribute = source.data.attributes.get(name)
        if attribute:
            values = np.empty(len(attribute.data), dtype="<f4")
            attribute.data.foreach_get("value", values)
            digest.update(name.encode() + values.tobytes())
    return digest.hexdigest()


def has_reference(detail):
    attribute = detail.data.attributes.get(REFERENCE)
    return bool(
        attribute
        and attribute.domain == "POINT"
        and attribute.data_type == "FLOAT_VECTOR"
    )


def read_reference(detail):
    if not has_reference(detail):
        return None
    attribute = detail.data.attributes[REFERENCE]
    result = np.empty((len(detail.data.vertices), 3), dtype=np.float32)
    attribute.data.foreach_get("vector", result.ravel())
    return result.astype(float)


def bind(context, source, detail, reference, *, detail_topology=None):
    detail["cm_detail_reference_matrix"] = (
        np.asarray(detail.matrix_world, dtype=float).ravel().tolist()
    )
    detail["cm_detail_bound_steps"] = steps_for(detail)
    detail['cm_detail_bound_subdivision_type']=subdivision_type(detail)
    attribute = detail.data.attributes.get(REFERENCE) or detail.data.attributes.new(
        REFERENCE, "FLOAT_VECTOR", "POINT"
    )
    attribute.data.foreach_set(
        "vector", np.asarray(reference, dtype=np.float32).ravel()
    )
    detail["cm_detail_source_topology"] = m.topology(source)
    detail["cm_detail_source_shape"] = signature(context, source)
    detail["cm_detail_source"] = source
    detail["cm_detail_topology"] = detail_topology or m.topology(detail)
    if 'cm_head_update_pending' in detail:
        del detail['cm_head_update_pending']


def surface(
    context, source, level, *, target_mesh=None, local_positions=None, steps=None, subdivision_type='CATMULL_CLARK'
):
    """Evaluate the pristine subdivision surface without changing selection or modes."""
    data = bpy.data.meshes.new("CM_DetailReference")
    local = (
        m.to_local(source, m.evaluated_coordinates(context, source))
        if local_positions is None
        else np.asarray(local_positions)
    )
    data.from_pydata(
        local, m.edges(source), [p.vertices[:] for p in source.data.polygons]
    )
    for name in ("crease_edge", "crease_vert"):
        attribute = source.data.attributes.get(name)
        if attribute:
            copy = data.attributes.new(name, attribute.data_type, attribute.domain)
            values = np.empty(len(attribute.data), dtype=np.float32)
            attribute.data.foreach_get("value", values)
            copy.data.foreach_set("value", values)
    return evaluate_surface(
        context,
        data,
        source.matrix_world.copy(),
        level,
        target_mesh=target_mesh,
        steps=steps,
        subdivision_type=subdivision_type,
    )


def evaluate_surface(
    context,
    data,
    matrix,
    level,
    *,
    expected_topology=None,
    target_mesh=None,
    correspondence_report=None,
    steps=None,
    subdivision_type='CATMULL_CLARK',
):
    """Evaluate and dispose an owned coarse mesh, including in a worker process."""
    probe = bpy.data.objects.new("CM_DetailReference", data)
    scene = bpy.data.scenes.new("CM_DetailReference")
    scene.collection.objects.link(probe)
    probe.matrix_world = matrix
    try:
        if subdivision_type not in {'CATMULL_CLARK','SIMPLE'}:raise ValueError('Invalid subdivision reference type')
        data.update()
        for step in steps or [int(level)]:
            modifier = probe.modifiers.new("Detail reference subdivision", "SUBSURF")
            modifier.subdivision_type=subdivision_type
            modifier.levels = int(step)
        with context.temp_override(scene=scene, view_layer=scene.view_layers[0]):
            context.view_layer.update()
            evaluated = probe.evaluated_get(context.evaluated_depsgraph_get())
            mesh = evaluated.to_mesh()
            try:
                # Dense reference evaluation must not allocate a Python Vector
                # and a list entry for every subdivided point.
                local = np.empty((len(mesh.vertices), 3), dtype=np.float64)
                mesh.vertices.foreach_get("co", local.ravel())
                transform = np.asarray(evaluated.matrix_world)
                world = local @ transform[:3, :3].T + transform[:3, 3]
                if target_mesh is not None:
                    from .subdivision_order import from_mesh, correspondence

                    mapping = correspondence(
                        from_mesh(mesh), from_mesh(target_mesh), len(data.vertices)
                    )
                    if correspondence_report is not None:
                        correspondence_report.update(
                            vertices=len(mapping),
                            remapped_vertices=int(
                                np.count_nonzero(mapping != np.arange(len(mapping)))
                            ),
                        )
                    return world[mapping]
                if expected_topology is not None:
                    from .geometry import topology_hash

                    actual = topology_hash(
                        len(mesh.vertices), [p.vertices[:] for p in mesh.polygons]
                    )
                    if actual != expected_topology:
                        raise ValueError(
                            "Subdivision vertex order differs from the sculpt; work retained. Use the same Blender thread setting as the original layer"
                        )
                return world
            finally:
                evaluated.to_mesh_clear()
    finally:
        bpy.data.objects.remove(probe, do_unlink=True)
        if not data.users:
            bpy.data.meshes.remove(data)
        bpy.data.scenes.remove(scene)


def transport(detail, before, after):
    """Fit surface deformation per vertex, retaining sculpt height along the normal."""
    count = len(before)
    edges = m.edges(detail)
    indices = np.concatenate((edges[:, 0], edges[:, 1]))
    neighbors = np.concatenate((edges[:, 1], edges[:, 0]))
    old = before[neighbors] - before[indices]
    new = after[neighbors] - after[indices]
    length = np.maximum(np.linalg.norm(old, axis=1), 1e-12)
    old, new = old / length[:, None], new / length[:, None]
    covariance, cross = np.zeros((count, 3, 3)), np.zeros((count, 3, 3))
    np.add.at(covariance, indices, old[:, :, None] * old[:, None, :])
    np.add.at(cross, indices, old[:, :, None] * new[:, None, :])
    detail.data.calc_loop_triangles()
    triangles = np.asarray([tri.vertices[:] for tri in detail.data.loop_triangles])

    def normals(points):
        values = np.cross(
            points[triangles[:, 1]] - points[triangles[:, 0]],
            points[triangles[:, 2]] - points[triangles[:, 0]],
        )
        result = np.zeros((count, 3))
        for corner in range(3):
            np.add.at(result, triangles[:, corner], values)
        return result / np.maximum(np.linalg.norm(result, axis=1), 1e-12)[:, None]

    old_normal, new_normal = normals(before), normals(after)
    weight = np.maximum(np.bincount(indices, minlength=count), 1) / 3
    covariance += (
        old_normal[:, :, None] * old_normal[:, None, :] * weight[:, None, None]
    )
    cross += old_normal[:, :, None] * new_normal[:, None, :] * weight[:, None, None]
    result = np.empty_like(covariance)
    for first in range(0, count, 32768):
        part = slice(first, first + 32768)
        result[part] = np.linalg.solve(
            covariance[part] + np.eye(3) * 1e-10, cross[part] + np.eye(3) * 1e-10
        )
    if not np.isfinite(result).all():
        raise ValueError(
            "Detail surface deformation is invalid; the sculpt was retained"
        )
    return result


def reference_for_update(context, source, detail):
    if not source or not detail:
        return None
    if REFERENCE not in detail.data.attributes and "cm_detail_level" not in detail:
        # An imported/custom bake source has no known unsculpted surface.
        # It remains a valid source; automatic following is for our own layers.
        return None
    topology = m.topology(source)
    if detail.get("cm_detail_source_topology", topology) != topology:
        raise ValueError(
            "Accepted topology changed; retain the detail layer and restore its original topology"
        )
    saved_topology = detail.get("cm_detail_topology")
    if saved_topology and saved_topology != m.topology(detail):
        raise ValueError(
            "Sculpt topology changed; retain the layer and restore its original topology"
        )
    reference = read_reference(detail)
    current = signature(context, source)
    bound_steps = list(
        detail.get("cm_detail_bound_steps", [int(detail.get("cm_detail_level", 1))])
    )
    bound_matrix = np.asarray(detail.get("cm_detail_reference_matrix", []), dtype=float)
    if (
        reference is not None
        and detail.get("cm_detail_source_shape") == current
        and bound_steps == steps_for(detail)
        and detail.get('cm_detail_bound_subdivision_type','CATMULL_CLARK')==subdivision_type(detail)
        and bound_matrix.shape == (16,)
        and np.array_equal(bound_matrix, np.asarray(detail.matrix_world).ravel())
    ):
        return None
    if reference is None:
        # LV1 legacy layers always have an unsculpted Basis. Higher legacy
        # levels may have baked artist edits in Basis, so cannot be guessed.
        if detail.get("cm_detail_level", 1) != 1:
            from .detail_recovery import describe

            return describe(context, detail)
        if not detail.data.shape_keys:
            raise ValueError(
                "The saved sculpt has no original Basis; retain it for reprojection"
            )
        reference = np.asarray(
            [v.co[:] for v in detail.data.shape_keys.key_blocks[0].data]
        )
    return reference


def prepare(context, source, detail):
    reference = reference_for_update(context, source, detail)
    if reference is None:
        return None
    with feedback(context):
        return _prepare_surface(context, source, detail, reference)


def _prepare_surface(context, source, detail, reference):
    from .detail_recovery import LegacyReference, resolve

    steps = steps_for(detail)
    if isinstance(reference, LegacyReference):

        def direct_target():
            return m.to_local(
                detail,
                surface(
                    context,
                    source,
                    int(detail.get("cm_detail_level", 1)),
                    target_mesh=detail.data,
                    subdivision_type=subdivision_type(detail),
                ),
            )

        recovered = resolve(context, reference, direct_target)
        reference, steps = recovered.reference, recovered.steps
    target = m.to_local(
        detail,
        surface(
            context,
            source,
            detail.get("cm_detail_level", 1),
            target_mesh=detail.data,
            steps=steps,
            subdivision_type=subdivision_type(detail),
        ),
    )
    plan = plan_from_target(source, detail, reference, target)
    plan["steps"] = steps
    return plan


def plan_from_target(source, detail, reference, target):
    if target.shape != reference.shape:
        raise ValueError("Detail subdivision topology changed; the sculpt was retained")
    difference = target - reference
    if np.max(np.abs(difference)) < 1e-7:
        linear = None
    else:
        linear = transport(detail, reference, target)
    updates = []
    blocks = list(detail.data.shape_keys.key_blocks) if detail.data.shape_keys else []
    for block in blocks:
        points = np.asarray([v.co[:] for v in block.data])
        offset = points - reference
        output = target + (
            np.einsum("ni,nij->nj", offset, linear) if linear is not None else offset
        )
        updates.append((block, output))
    points = m.coordinates(detail, world=False)
    offset = points - reference
    mesh = target + (
        np.einsum("ni,nij->nj", offset, linear) if linear is not None else offset
    )
    return {
        "source": source,
        "detail": detail,
        "target": target,
        "updates": updates,
        "mesh": mesh,
    }


def commit(context, plan):
    if plan is None:
        return False
    detail = plan["detail"]
    detail.data.vertices.foreach_set("co", plan["mesh"].ravel())
    for block, points in plan["updates"]:
        block.data.foreach_set("co", points.ravel())
    detail.data.update()
    if "steps" in plan:
        detail["cm_detail_subdivision_steps"] = plan["steps"]
    bind(context, plan["source"], detail, plan["target"])
    return True


def sync(context, source, detail):
    return commit(context, prepare(context, source, detail))


def sync_character(context, s, f,*,roles=None):
    pairs = [(s.result, s.detail), (f.body_result, f.body_detail)]
    if roles is not None:
        pairs=[(f.body_result,f.body_detail) if role=='BODY' and s.part=='FULL_BODY' else (s.result,s.detail) for role in roles]
    plans = [prepare(context, source, detail) for source, detail in pairs]
    for plan in plans:
        commit(context, plan)
    from . import simple_shading
    for source,detail in pairs:
        simple_shading.refresh(context,source,detail)
