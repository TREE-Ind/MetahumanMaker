"""Project, anatomy, masks and detail transactions for the expanded workflow."""

from pathlib import Path
import json
import platform
import time
import bpy
import numpy as np
from mathutils import Vector
from . import mesh_ops as m, anatomy


def group_weights(obj, name, default=0.0):
    values = np.full(len(obj.data.vertices), default, dtype=float)
    group = obj.vertex_groups.get(name)
    if group:
        values[:] = 0
        for vertex in obj.data.vertices:
            for item in vertex.groups:
                if item.group == group.index:
                    values[vertex.index] = item.weight
    return values


def write_group(obj, name, values):
    group = obj.vertex_groups.get(name) or obj.vertex_groups.new(name=name)
    group.remove(list(range(len(obj.data.vertices))))
    for weight in np.unique(np.round(values, 3)):
        indices = np.flatnonzero(np.round(values, 3) == weight).tolist()
        if weight > 0 and indices:
            group.add(indices, float(weight), "REPLACE")
    return group


def key_from_world(obj, name, vertices, relative=None):
    bpy.context.view_layer.update()
    visible = m.to_local(obj, m.evaluated_coordinates(bpy.context, obj))
    if not obj.data.shape_keys:
        obj.shape_key_add(name="Basis")
    key = obj.data.shape_keys.key_blocks.get(name)
    old_delta = np.zeros_like(visible)
    if key:
        old_delta = (
            np.array([v.co[:] for v in key.data])
            - np.array([v.co[:] for v in key.relative_key.data])
        ) * key.value
    if key is None:
        key = obj.shape_key_add(name=name)
        if relative:
            key.relative_key = obj.data.shape_keys.key_blocks[relative]
    baseline = np.array([v.co[:] for v in key.relative_key.data])
    key.data.foreach_set(
        "co", (baseline + m.to_local(obj, vertices) - visible + old_delta).ravel()
    )
    key.value = 1
    obj.data.update()
    return key


def decimate_source(context, s, f):
    from . import wrap_proxy
    return wrap_proxy.create(context,s,f)


def anatomical_masks(context, s, f):
    obj = s.result or s.template
    m.check_mesh(obj, "Head")
    world = m.evaluated_coordinates(context, obj)
    lo, hi = world.min(0), world.max(0)
    size = np.maximum(hi - lo, 1e-8)
    z = (world[:, 2] - lo[2]) / size[2]
    lateral = 0 if f.front_axis in {"Y", "-Y"} else 1
    lateral_distance = (
        np.abs(world[:, lateral] - (lo[lateral] + hi[lateral]) * 0.5) / size[lateral]
    )
    regions = {
        "Head": np.ones(len(world)),
        "Ears": ((lateral_distance > 0.35) & (z > 0.25) & (z < 0.76)).astype(float),
        "Eyes": (
            (lateral_distance > 0.07)
            & (lateral_distance < 0.35)
            & (z > 0.48)
            & (z < 0.72)
        ).astype(float),
        "Upper Orbital Rim": (
            (lateral_distance > 0.07)
            & (lateral_distance < 0.35)
            & (z > 0.64)
            & (z < 0.76)
        ).astype(float),
        "NeckRegion": np.clip((0.20 - z) / 0.12, 0, 1),
    }
    if s.source and s.source != obj:
        from mathutils.kdtree import KDTree

        available = [
            name
            for name in ("Head", "Ears", "Eyes", "Upper_Orbital_Rim", "Inner_Parts")
            if s.source.vertex_groups.get("CM_" + name)
        ]
        if available:
            source = m.coordinates(s.source)
            tree = KDTree(len(source))
            for i, p in enumerate(source):
                tree.insert(p, i)
            tree.balance()
            nearest = np.array([tree.find(p)[1] for p in world])
            for name in available:
                regions[name.replace("_", " ")] = group_weights(s.source, "CM_" + name)[
                    nearest
                ]
    # Existing imported anatomy groups take precedence over spatial estimates.
    for name, weights in regions.items():
        group_name = "CM_" + name.replace(" ", "_")
        if obj.vertex_groups.get(group_name) is None:
            write_group(obj, group_name, weights)
    if obj.vertex_groups.get('CM_DetailMask') is None:
        write_group(obj, "CM_DetailMask", np.ones(len(world)))
    s.status = "Editable region groups created. Spatial masks are estimates; refine them before detailed work."


def sculpt_region(context, s, f):
    obj = s.result
    m.check_mesh(obj, "Result")
    if obj.vertex_groups.get("CM_" + f.sculpt_region) is None:
        anatomical_masks(context, s, f)
    if obj.vertex_groups.get("CM_" + f.sculpt_region) is None:
        raise ValueError(
            "This region needs an imported or painted CM_"
            + f.sculpt_region
            + " vertex group"
        )
    weights = group_weights(obj, "CM_" + f.sculpt_region)
    m.activate(context, obj)
    mask = obj.data.attributes.get(".sculpt_mask") or obj.data.attributes.new(
        ".sculpt_mask", "FLOAT", "POINT"
    )
    mask.data.foreach_set("value", 1 - weights)
    bpy.ops.cm.sculpt()


def generate_displacement(context, s, f):
    from . import scan_projection
    if s.detail:
        scan_projection.regenerate(context, s, f)
        return
    obj = m.create_detail(context, s)
    try:
        projected = scan_projection.key_points(obj.data.shape_keys.key_blocks['Scan Detail'])
        scan_projection.write(context, s, f, obj, projected)
    except Exception:
        s.detail = None
        bpy.data.objects.remove(obj, do_unlink=True)
        raise
    s.status = 'Displacement preview ready. Adjust Scan Strength, or edit the mask and regenerate, then Apply Projected Detail once.'


def apply_scan(context, s, f):
    m.check_mesh(s.detail, "Detail")
    if s.detail.get("cm_scan_applied", True):
        raise ValueError(
            "Scan Data was already applied to this layer; clear the layer to start again"
        )
    from . import scan_projection
    if scan_projection.editable(s.detail) and scan_projection.dirty(s.detail, f):
        raise ValueError('The scan mask or protection settings changed. Regenerate Source Displacement before applying')
    if s.detail.get('cm_scan_regions'):
        from . import regional_scan
        regional_scan.validate(context,s,s.detail)
    s.detail.data.shape_keys.key_blocks["Scan Detail"].value = s.detail_strength
    s.detail["cm_scan_applied"] = True
    s.status = "Scan Data applied once; manual sculpt remains editable."


def clear_detail(context, s, f):
    if s.detail:
        obj = s.detail
        from . import simple_shading
        simple_shading.restore(context,obj)
        s.detail = None
        bpy.data.objects.remove(obj, do_unlink=True)


def neck_ring(obj, head=True):
    world = m.coordinates(obj)
    loops, welded = anatomy.geometry.welded_boundary_loops(
        world, [p.vertices[:] for p in obj.data.polygons]
    )
    explicit = np.flatnonzero(group_weights(obj, "CM_Neck") > 0.5)
    if len(explicit):
        _, unique = np.unique(welded[explicit], return_index=True)
        return explicit[np.sort(unique)]
    if not loops:
        raise ValueError(
            "No closed neck boundary; define a CM_Neck group on both meshes"
        )
    # Physical neck loops slope around the chest/back. A narrow Z band loses most
    # of their vertices, and UV split edges must never count as physical holes.
    return (
        min(loops, key=lambda ring: world[ring, 2].mean())
        if head
        else max(loops, key=lambda ring: world[ring, 2].mean())
    )


def attach_updated_head(context,s,f):
    """Place a newly fitted generated head on the unchanged accepted body."""
    from mathutils import Matrix
    from .guided import signature
    head,body=s.result,f.body_result
    h=m.evaluated_coordinates(context,head)[neck_ring(head)]
    b=m.evaluated_coordinates(context,body)[neck_ring(body,False)]
    movement=b.mean(0)-h.mean(0)
    transform=Matrix.Translation(Vector(movement))
    source_fit=m.transform_state(s.source)
    for obj in dict.fromkeys([head,s.source]+([f.eye_left,f.eye_right,f.teeth] if f.full_head else [])):
        if obj:
            obj.matrix_world=transform@obj.matrix_world
    context.view_layer.update()
    s.source['cm_head_placement']=json.dumps({'fit':source_fit,'placed':m.transform_state(s.source),'movement':movement.tolist()})
    head['cm_source_signature']=signature(s.source)
    attach_neck(context,s,f)


def attach_neck(context, s, f):
    head, body = f.head_result or s.result, f.body_result
    m.check_mesh(head, "Head result")
    m.check_mesh(body, "Body result")
    if head == body:
        raise ValueError("Head and Body must be separate meshes")
    h, b = (
        m.evaluated_coordinates(context, head),
        m.evaluated_coordinates(context, body),
    )
    hi, bi = anatomy.match_neck(h, neck_ring(head), b, neck_ring(body, False))
    output = h.copy()
    # A smooth extension of the neck correction avoids a visible step above the exact ring.
    centers = h[hi]
    delta = b[bi] - h[hi]
    radius = max(np.ptp(h[:, 2]) * 0.12, 1e-6)
    for start in range(0, len(h), 2048):
        distance = np.sum((h[start : start + 2048, None] - centers[None]) ** 2, axis=2)
        weight = np.exp(-distance / (2 * radius**2))
        denominator = np.maximum(weight.sum(1), 1e-10)
        strength = np.max(weight, axis=1)
        output[start : start + len(weight)] += (
            (weight @ delta) / denominator[:, None] * strength[:, None]
        )
    output[hi] = b[bi]
    # UV copies of a seam point must receive exactly the same correction.
    _, welded = anatomy.geometry.welded_boundary_loops(
        m.coordinates(head), [p.vertices[:] for p in head.data.polygons]
    )
    for head_index, body_index in zip(hi, bi):
        output[welded == welded[head_index]] = b[body_index]
    key_from_world(head, "Neck Seam", output)
    f.head_result = head
    head["cm_neck_attached"] = True
    s.status = f"Neck matched exactly: {len(hi)} physical points, including UV copies."


def body_preview(context, s, f):
    if context.mode!='OBJECT' or f.full_sculpt or f.restore_editing:
        raise ValueError('Finish the current sculpt or repair before shaping the body')
    if f.expression_active:
        raise ValueError('Return the expression to Neutral before shaping the body')
    if f.body_editing:
        if f.body_backend=='REGIONAL':return regional_body_update(context,s,f)
        return _body_preview(context,s,f)
    from . import sculpt_archive
    previous_layers=sculpt_archive.layers(context.scene)
    previous_table=context.scene.get(sculpt_archive.RETURNED)
    old_objects=set(obj for obj in previous_layers.values() if obj)
    old_objects|={child for obj in tuple(old_objects) for child in obj.children if child.get('cm_wrinkle_owner')==obj}
    visibility={obj:(obj.hide_get(),obj.hide_render) for obj in old_objects}
    before_ids=set(bpy.data.user_map())
    objects=set(obj for obj in (s.result,f.head_result,f.body_result,f.eye_left,f.eye_right,f.teeth) if obj and obj.type=='MESH')
    keys={obj:{key.as_pointer() for key in obj.data.shape_keys.key_blocks} if obj.data.shape_keys else set() for obj in objects}
    properties={obj:obj.get('cm_body_snapshot') for obj in objects}
    positions={obj:np.empty(len(obj.data.vertices)*3,dtype=np.float32) for obj in objects}
    for obj,points in positions.items():obj.data.vertices.foreach_get('co',points)
    values=[(item,item.value) for item in f.body_controls]
    key_ids=set(bpy.data.shape_keys)
    selected=set(context.selected_objects)
    active=context.view_layer.objects.active
    try:
        # Both the head and body can move in a body pass. Adopt their returned
        # sources before changing accepted geometry, preserving undoable receipts.
        with sculpt_archive.recover(context,rebind_maps=False) as staged:
            sculpt_archive.checkpoint_body(context.scene,staged,previous_layers,visibility,previous_table,before_ids)
            return _body_preview(context,s,f)
    except Exception:
        if sculpt_archive.BODY_CHECKPOINT in context.scene:del context.scene[sculpt_archive.BODY_CHECKPOINT]
        f.body_editing=False
        for obj in objects:
            if obj.data.shape_keys:
                for key in reversed(list(obj.data.shape_keys.key_blocks)):
                    if key.as_pointer() not in keys[obj]:obj.shape_key_remove(key)
            obj.data.vertices.foreach_set('co',positions[obj]);obj.data.update()
            previous=properties[obj]
            if previous is None:
                if 'cm_body_snapshot' in obj:del obj['cm_body_snapshot']
            else:obj['cm_body_snapshot']=previous
        for item,value in values:item.value=value
        for data in set(bpy.data.shape_keys)-key_ids:
            if not data.users:bpy.data.shape_keys.remove(data)
        for obj in context.selected_objects:obj.select_set(False)
        for obj in selected:obj.select_set(True)
        context.view_layer.objects.active=active
        context.view_layer.update()
        raise


def _body_preview(context, s, f):
    if context.mode!='OBJECT' or f.full_sculpt or f.restore_editing:
        raise ValueError('Finish the current sculpt or repair before shaping the body')
    if f.expression_active:
        raise ValueError('Return the expression to Neutral before shaping the body')
    if not f.body_editing and context.screen and not bpy.app.background:
        from .guided import focus
        focus(context,[f.body_result or s.result,f.head_result,f.eye_left,f.eye_right,f.teeth],f.front_axis)
    if f.body_backend=='NATIVE':
        from . import native_body
        return native_body.preview(context,s,f)
    return regional_body_preview(context,s,f)


def regional_body_update(context,s,f):
    """Keep the last whole-body preview if a later subject fails to update."""
    body=f.body_result or s.result
    previous_properties={}
    snapshots=[]
    for obj in dict.fromkeys(obj for obj in (body,f.head_result or s.result,f.eye_left,f.eye_right,f.teeth) if obj):
        keys=obj.data.shape_keys.key_blocks if obj.data.shape_keys else ()
        existing={key.as_pointer() for key in keys}
        saved=[]
        for key in keys:
            if key.name not in {'Body Pass','Body Pass Base','Body Follow','Body Follow Base'}:continue
            points=np.empty(len(key.data)*3,dtype=np.float32);key.data.foreach_get('co',points)
            saved.append((key,points,key.value,key.relative_key))
        snapshots.append((obj,existing,saved))
        previous_properties[obj]=obj.get('cm_body_snapshot')
    state=json.loads(body.get('cm_body_snapshot','{}'))
    controls=state.get('preview_controls',{})
    try:
        result=regional_body_preview(context,s,f)
        f.last_action_error=False
        return result
    except Exception as error:
        for obj,existing,saved in snapshots:
            if obj.data.shape_keys:
                for key in reversed(list(obj.data.shape_keys.key_blocks)):
                    if key.as_pointer() not in existing:obj.shape_key_remove(key)
            for key,points,value,relative in saved:
                key.data.foreach_set('co',points);key.value=value;key.relative_key=relative
            previous=previous_properties[obj]
            if previous is None:
                if 'cm_body_snapshot' in obj:del obj['cm_body_snapshot']
            else:obj['cm_body_snapshot']=previous
        # Raw property storage avoids recursively starting another failed solve.
        for item in f.body_controls:
            if item.key in controls:item['value']=controls[item.key]
        context.view_layer.update()
        f.last_action_error=True;s.status='Body preview retained: '+str(error)
        raise


def regional_body_preview(context, s, f):
    obj = f.body_result or s.result
    m.check_mesh(obj, "Body")
    if not f.body_editing:
        snapshot = m.evaluated_coordinates(context, obj)
        if not obj.data.shape_keys:
            obj.shape_key_add(name="Basis")
        key = obj.shape_key_add(name="Body Pass Base", from_mix=True)
        key.data.foreach_set("co", m.to_local(obj, snapshot).ravel())
        key.value = 0
        obj["cm_body_snapshot"] = json.dumps({"key": "Body Pass Base"})
        f.body_editing = True
    local = np.array(
        [v.co[:] for v in obj.data.shape_keys.key_blocks["Body Pass Base"].data]
    )
    matrix = np.asarray(obj.matrix_world)
    baseline = local @ matrix[:3, :3].T + matrix[:3, 3]
    values = {item.key: item.value for item in f.body_controls}
    # Store the pass delta relative to the snapshot, independently of pre-existing layers.
    preview = obj.data.shape_keys.key_blocks.get("Body Pass") or obj.shape_key_add(
        name="Body Pass"
    )
    preview.relative_key = obj.data.shape_keys.key_blocks["Body Pass Base"]
    from . import body_regions
    follow = {}
    output = body_regions.deform(obj, f.body_template or s.template, baseline, values, follow=follow)
    preview.data.foreach_set(
        "co", m.to_local(obj, output).ravel()
    )
    preview.value = 1
    obj.data.update()
    if s.part == "FULL_BODY" and s.result and s.result != obj:
        _body_follow(context, s, f, baseline, output, follow=follow)
    state=json.loads(obj.get('cm_body_snapshot','{}'));state['preview_controls']=values
    obj['cm_body_snapshot']=json.dumps(state)
    s.status='Body preview updated. Apply Body Pass to keep changes, or Cancel.'


def _body_follow(context, s, f, baseline, deformed, *, follow=None):
    """Keep the face rigid while the lower neck and collar follow the body."""
    head = f.head_result or s.result
    head_base = (
        head.data.shape_keys.key_blocks.get("Body Follow Base")
        if head.data.shape_keys
        else None
    )
    if head_base:
        local = np.asarray([v.co[:] for v in head_base.data])
        matrix = np.asarray(head.matrix_world)
        head_min = (local @ matrix[:3, :3].T + matrix[:3, 3])[:, 2].min()
    else:
        head_min = m.evaluated_coordinates(context, head)[:, 2].min()
    body = f.body_result
    state = json.loads(body.get('cm_body_snapshot', '{}'))
    ring = state.get('follow_neck_indices')
    if ring is None:
        try:
            ring = [int(index) for index in neck_ring(body, False)]
        except ValueError:
            # Closed custom bodies without an explicit neck group retain the
            # historical approximation. Cache only for this topology-fixed pass.
            ring = []
        state['follow_neck_indices'] = ring
        body['cm_body_snapshot'] = json.dumps(state)
    if ring:
        anchors, delta = baseline[ring], (deformed - baseline)[ring]
    else:
        top = baseline[:, 2] > head_min - np.ptp(baseline[:, 2]) * 0.01
        anchors, delta = baseline[top], (deformed - baseline)[top]
    movement = delta.mean(0)
    collar = follow.get('head_delta') if follow else None
    if collar:
        # Collar shaping is local volume, not a request to move the face/eyes.
        movement -= follow['shoulder_delta'][ring if ring else top].mean(0)
    for target in (head, f.eye_left, f.eye_right, f.teeth):
        if not target or target == f.body_result:
            continue
        if not target.data.shape_keys:
            target.shape_key_add(name="Basis")
        keys = target.data.shape_keys.key_blocks
        snapshot = keys.get("Body Follow Base")
        if snapshot is None:
            snapshot = target.shape_key_add(name="Body Follow Base", from_mix=True)
            snapshot.value = 0
        local = np.asarray([v.co[:] for v in snapshot.data])
        matrix = np.asarray(target.matrix_world)
        world = local @ matrix[:3, :3].T + matrix[:3, 3]
        output = world + movement
        if target == head:
            collar_delta = collar(output) if collar else np.zeros_like(world)
            output += collar_delta
            # A nearest surface correction locks coincident seam vertices exactly.
            # The shoulder field fades below the neck joint. The face and
            # accessories receive only the non-collar translation.
            anatomical = follow is not None and 'up' in follow
            up = follow['up'] if anatomical else np.array((0., 0., 1.))
            radius = max(np.ptp(world @ up) * 0.20, 1e-6)
            influence = np.clip(1 - ((world - anchors.mean(0)) @ up) / radius, 0, 1)
            # A sloping collar can extend beyond the broad neck band. Evaluate
            # its whole boundary so the exact seam correction is never skipped.
            indices = np.arange(len(world)) if anatomical else np.flatnonzero(influence > 0)
            for start in range(0, len(indices), 1024):
                selected = indices[start : start + 1024]
                distance = ((world[selected, None] - anchors[None]) ** 2).sum(2)
                nearest = np.argmin(distance, axis=1)
                seam = np.exp(
                    -distance[np.arange(len(selected)), nearest]
                    / max(radius * 0.04, 1e-6) ** 2
                )
                strength = np.maximum(influence[selected], seam)
                if collar:
                    local = follow['shoulder_delta'][ring if ring else top][nearest]
                    output[selected] += (delta[nearest] - local - movement) * strength[:, None]
                    output[selected] += (local - collar_delta[selected]) * seam[:, None]
                else:
                    output[selected] += (delta[nearest] - movement) * strength[:, None]
        follow_key = keys.get("Body Follow") or target.shape_key_add(name="Body Follow")
        follow_key.relative_key = snapshot
        follow_key.data.foreach_set("co", m.to_local(target, output).ravel())
        follow_key.value = 1
        target.data.update()


def body_finish(context, s, f, apply=True, *, detail_plans=None):
    from . import detail_layers
    if apply and f.body_backend == 'NATIVE':
        from . import native_body
        if not native_body.can_apply():
            raise ValueError('Wait for the current anatomical preview before applying')
    plans = detail_plans if detail_plans is not None else [detail_layers.prepare(context, source, detail) for source, detail in (
        (s.result, s.detail), (f.body_result, f.body_detail)
    )] if apply else []
    if f.body_backend=='NATIVE':
        from . import native_body
        result = native_body.finish(context,s,f,apply)
    else:
        result = regional_body_finish(context,s,f,apply)
    for plan in plans:
        detail_layers.commit(context, plan)
    from . import sculpt_archive
    sculpt_archive.finish_body_recovery(context,apply)
    return result


def regional_body_finish(context, s, f, apply=True):
    obj = f.body_result or s.result
    if not f.body_editing:
        raise ValueError("No body pass is active")
    if apply:
        key = obj.data.shape_keys.key_blocks["Body Pass"]
        key.name = "Body Pass Applied " + str(len(obj.data.shape_keys.key_blocks))
        obj.data.shape_keys.key_blocks["Body Pass Base"].name = "Body Baseline " + str(
            len(obj.data.shape_keys.key_blocks)
        )
    else:
        obj.shape_key_remove(obj.data.shape_keys.key_blocks["Body Pass"])
        obj.shape_key_remove(obj.data.shape_keys.key_blocks["Body Pass Base"])
    for target in (f.head_result or s.result, f.eye_left, f.eye_right, f.teeth):
        if not target or target == obj or not target.data.shape_keys:
            continue
        for name in ("Body Follow", "Body Follow Base"):
            key = target.data.shape_keys.key_blocks.get(name)
            if key:
                if apply:
                    key.name = (
                        name + " Applied " + str(len(target.data.shape_keys.key_blocks))
                    )
                else:
                    target.shape_key_remove(key)
    f.body_editing = False
    for item in f.body_controls:
        item.value = 0
    s.status='Body pass applied. Start another pass to continue shaping.' if apply else 'Body pass canceled; the preceding accepted shape is retained.'


def import_project(context, s, f, path, part):
    from . import project_reuse
    obj = project_reuse.import_part(context, s, f, path, part)
    project_reuse.focus(context)
    return obj


def support_report(context, s, f):
    from . import resource_budget
    try:memory_headroom=resource_budget.memory()
    except ValueError as exc:memory_headroom={'error':str(exc)}
    folder = Path(bpy.path.abspath(s.project_dir))
    folder.mkdir(parents=True, exist_ok=True)
    report = {
        "time": time.time(),
        "blender": bpy.app.version_string,
        "system": platform.platform(),
        "memory_headroom": memory_headroom,
        "stage": s.stage,
        "status": s.status,
        "objects": {
            name: {"vertices": len(o.data.vertices), "topology": m.topology(o)}
            for name in ("source", "template", "result", "detail")
            if (o := getattr(s, name))
        },
        "landmark_pairs": len(s.landmarks),
    }
    path = folder / "support-report.json"
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    s.status = "Local support report saved; mesh/texture data are not included."
    return path


def force_a_pose(context, s, f):
    from . import body_pose, native_body, detail_jobs, bake_jobs, sculpt_session
    if context.mode!='OBJECT' or f.body_editing or f.full_sculpt or sculpt_session.pending():
        raise ValueError('Finish the active edit before changing body pose')
    if native_body._JOB or detail_jobs.running() or bake_jobs.running():
        raise ValueError('Wait for the current body or map operation before changing pose')
    if f.body_detail or s.part=='BODY' and s.detail:
        raise ValueError('Use Force A Pose before creating body sculpt detail')
    obj = f.body_result or s.result
    template = f.body_template or s.template
    m.check_mesh(obj, "Body result")
    m.check_mesh(template, "Body reference")
    if m.topology(obj) != m.topology(template):
        raise ValueError("A-pose needs the unchanged reference body topology")
    manager=context.window_manager
    manager.progress_begin(0,12)
    try:output,report=body_pose.align(context,obj,template,lambda step,total:manager.progress_update(step))
    finally:manager.progress_end()
    key_from_world(obj, "Force A Pose", output)
    obj['cm_a_pose_report']=json.dumps(report)
    s.status = ('A-pose aligned '+str(report['segments'])+' limb segments using '+
                ('reference joints.' if report['joint_pivots'] else 'estimated pivots. Inspect joints.'))


def reset_a_pose(context,s,f):
    from . import native_body,detail_jobs,bake_jobs,sculpt_session
    if context.mode!='OBJECT' or f.body_editing or f.full_sculpt:
        raise ValueError('Finish the active edit before restoring body pose')
    if native_body._JOB or detail_jobs.running() or bake_jobs.running() or sculpt_session.pending():
        raise ValueError('Wait for the current body or map operation before restoring pose')
    if f.body_detail or s.part=='BODY' and s.detail:
        raise ValueError('Restore body pose before creating body sculpt detail')
    obj=f.body_result or s.result
    key=obj.data.shape_keys.key_blocks.get('Force A Pose') if obj and obj.data.shape_keys else None
    if not key:raise ValueError('This body has no Force A Pose layer')
    key.value=0;obj.data.update()
    s.status='Previous body pose restored; body proportions and other layers retained.'


def expression_check(context, s, f):
    from . import expressions
    if expressions.reference_path(f).is_file() or f.expression_reference:
        return expressions.preview(context,s,f)
    obj, template = s.result, s.template
    m.check_mesh(obj, "Result")
    if not template or not template.data.shape_keys:
        raise ValueError(
            "Expression Check needs reference expression shape keys. Import a compatible expression library first."
        )
    expressions = [
        k
        for k in template.data.shape_keys.key_blocks
        if k.name not in {"Basis", "Wrapped", "Sculpt"}
    ]
    if not expressions:
        raise ValueError(
            "No expression shapes in this template. Use a licensed expression reference or inspect expressions in MetaHuman Creator."
        )
    key = expressions[f.expression_index % len(expressions)]
    baseline = np.array([v.co[:] for v in template.data.shape_keys.key_blocks[0].data])
    delta = np.array([v.co[:] for v in key.data]) - baseline
    if len(delta) != len(obj.data.vertices):
        raise ValueError("Expression reference vertex counts do not match")
    target_key = obj.data.shape_keys.key_blocks.get(
        "Expression Check"
    ) or obj.shape_key_add(name="Expression Check")
    target_key.relative_key = obj.data.shape_keys.key_blocks[0]
    basis = np.array([v.co[:] for v in target_key.relative_key.data])
    target_key.data.foreach_set("co", (basis + delta).ravel())
    target_key.value = f.expression_strength
    obj.data.update()
    s.status = (
        "Reference expression preview: " + key.name + ". Neutralize before exporting."
    )


def accessory_anchors(template, role, before, center):
    """Fit internal meshes from their facial margins, not collapsed inner surfaces."""
    groups = set()
    for group in template.vertex_groups:
        name = group.name.lower()
        if role == 'teeth':
            matches = 'lip' in name and not any(word in name for word in ('skin','outer','lash'))
        else:
            matches = any(word in name for word in ('eyelid','eyecorner')) and 'lash' not in name
        if matches:
            groups.add(group.index)
    candidates = np.asarray([vertex.index for vertex in template.data.vertices
        if any(item.group in groups and item.weight > .05 for item in vertex.groups)], dtype=int)
    semantic = len(candidates) >= 16
    if not semantic:
        candidates = np.arange(len(before))
    distance = np.linalg.norm(before[candidates] - center, axis=1)
    return candidates[np.argsort(distance)[:512 if semantic else 128]], semantic


def bind_accessory_fit(context, s, f, *, roles=('eye_left','eye_right')):
    """Keep a custom eye fit when facial accessories subsequently follow the head.

    Store a small facial-margin reference, not a duplicate head or baked mesh.
    Call after an approved geometry fit; following remains relative to this fit.
    """
    m.check_mesh(s.template, 'Template')
    m.check_mesh(s.result, 'Result')
    before = m.coordinates(s.template)
    after = m.evaluated_coordinates(context, s.result)
    if len(before) != len(after) or m.topology(s.template) != m.topology(s.result):
        raise ValueError('Accessory binding needs the unchanged head topology')
    planned = []
    for role in roles:
        if role not in {'eye_left','eye_right','teeth'}:
            raise ValueError('Unknown facial accessory role')
        obj = getattr(f, role)
        if not obj:
            continue
        m.check_mesh(obj, role)
        original = np.asarray(obj.get('cm_follow_original', [v for row in obj.matrix_world for v in row])).reshape(4,4)
        local = m.coordinates(obj, world=False)
        center = (local @ original[:3,:3].T + original[:3,3]).mean(0)
        anchors, _ = accessory_anchors(s.template, role, before, center)
        planned.append((obj, anchors))
    if not planned:
        raise ValueError('Choose eye/teeth mesh objects in Anatomy first')
    for obj, anchors in planned:
        obj['cm_follow_fit_head'] = s.result
        obj['cm_follow_fit_topology'] = m.topology(s.result)
        obj['cm_follow_fit_anchors'] = anchors.tolist()
        obj['cm_follow_fit_positions'] = after[anchors].ravel().tolist()
        obj['cm_follow_fit_matrix'] = [v for row in obj.matrix_world for v in row]


def update_accessories(context, s, f, *, roles=('eye_left','eye_right','teeth')):
    from mathutils import Matrix
    from .geometry import similarity

    m.check_mesh(s.template, "Template")
    m.check_mesh(s.result, "Result")
    before = m.coordinates(s.template)
    after = m.evaluated_coordinates(context, s.result)
    if len(before) != len(after) or m.topology(s.template) != m.topology(s.result):
        raise ValueError("Accessory following needs the unchanged base topology")
    planned = []
    if any(role not in {'eye_left','eye_right','teeth'} for role in roles):raise ValueError('Unknown facial accessory role')
    for role in roles:
        obj = getattr(f, role)
        if not obj:
            continue
        m.check_mesh(obj, role)
        if 'cm_follow_fit_head' in obj:
            if obj['cm_follow_fit_head'] != s.result or obj.get('cm_follow_fit_topology') != m.topology(s.result):
                raise ValueError('The custom accessory fit belongs to another head or topology; bind it to the intended head first')
            anchors = np.asarray(obj['cm_follow_fit_anchors'], dtype=int)
            reference = np.asarray(obj['cm_follow_fit_positions'], dtype=float).reshape(-1,3)
            bound = np.asarray(obj['cm_follow_fit_matrix'], dtype=float).reshape(4,4)
            if anchors.ndim != 1 or len(anchors) < 3 or reference.shape != (len(anchors),3) or anchors.min()<0 or anchors.max()>=len(after) or not np.isfinite(reference).all() or not np.isfinite(bound).all():
                raise ValueError('Invalid saved accessory fitting frame; bind the fitted accessory again')
            if np.array_equal(reference, after[anchors]):
                result = Matrix(bound.tolist())
            else:
                rotation, scale, translation = similarity(reference, after[anchors])
                if not np.isfinite(scale) or scale <= 0:
                    raise ValueError('The fitted facial margins do not define a positive accessory scale')
                transform = np.eye(4)
                transform[:3,:3] = rotation.T * scale
                transform[:3,3] = translation
                result = Matrix(transform.tolist()) @ Matrix(bound.tolist())
            planned.append((obj, bound, result, True))
            continue
        original = np.asarray(obj.get('cm_follow_original', [v for row in obj.matrix_world for v in row])).reshape(4,4)
        local = m.coordinates(obj, world=False)
        bounds = local @ original[:3,:3].T + original[:3,3]
        center = bounds.mean(0)
        nearest, semantic = accessory_anchors(s.template, role, before, center)
        rotation, local_scale, _ = similarity(before[nearest], after[nearest])
        # Stylized characters can have facial margins several times larger
        # than the template. Clipping a valid similarity scale leaves small
        # eyeballs inside large fitted lids (or oversized eyes on small heads).
        if not np.isfinite(local_scale) or local_scale<=0:raise ValueError('Facial margins do not define a positive accessory scale')
        transform = np.eye(4)
        transform[:3,:3] = rotation.T * local_scale
        transform[:3,3] = after[nearest].mean(0) - before[nearest].mean(0) @ rotation * local_scale
        planned.append((obj, original, Matrix(transform.tolist()) @ Matrix(original.tolist()), semantic))
    if not planned:
        raise ValueError("Choose eye/teeth mesh objects in Anatomy first")
    for obj, original, result, semantic in planned:
        if 'cm_follow_original' not in obj:
            obj['cm_follow_original'] = original.ravel().tolist()
        if obj.matrix_world != result:
            obj.matrix_world = result
        obj['cm_follow_anchors'] = 'FACIAL_MARGIN' if semantic else 'NEAREST_SURFACE'
    s.status = f"Updated {len(planned)} accessory meshes from their saved original transforms."


def subdivide_detail(context, s, f):
    from . import scan_projection
    scan_projection.require_applied(s.detail)
    from . import detail_layers
    obj = s.detail
    m.check_mesh(obj, "Detail")
    detail_layers.sync(context, s.result, obj)
    level = obj.get("cm_detail_level", s.detail_levels)
    if level >= 3:
        raise ValueError("Detail is already at LV3")
    visible = m.evaluated_coordinates(context, obj)
    result = m.duplicate(context, obj, obj.name + "_LV" + str(level + 1))
    from . import simple_shading
    shading_source=simple_shading.detach_copy(result,allow_missing_modifier=True)
    result.data.vertices.foreach_set("co", m.to_local(result, visible).ravel())
    result.data.update()
    m.activate(context, result)
    modifier = result.modifiers.new("Additional Detail Level", "SUBSURF")
    modifier.subdivision_type=detail_layers.subdivision_type(obj)
    modifier.levels = 1
    bpy.ops.object.modifier_apply(modifier=modifier.name)
    result.shape_key_add(name="Basis")
    result.shape_key_add(name="Scan Detail").value = 1
    result["cm_scan_applied"] = obj.get("cm_scan_applied", True)
    result["cm_detail_level"] = level + 1
    result['cm_detail_subdivision_steps'] = detail_layers.steps_for(obj) + [1]
    result['cm_detail_previous'] = obj
    detail_layers.bind(context, s.result, result, m.to_local(result, detail_layers.surface(context, s.result, level + 1, target_mesh=result.data, steps=detail_layers.steps_for(result),subdivision_type=detail_layers.subdivision_type(result))))
    if shading_source is not None:
        try:simple_shading.apply(context,s.result,result,source=shading_source)
        except Exception:
            simple_shading.restore(context,result)
            data=result.data;bpy.data.objects.remove(result,do_unlink=True)
            if not data.users:bpy.data.meshes.remove(data)
            m.activate(context,obj)
            raise
    obj.hide_set(True)
    obj.hide_render = True
    s.detail = result
    s.status = f"Detail subdivided to LV{level + 1}; previous scan/sculpt surface retained in the hidden previous layer."


def full_sculpt_start(context, s, f):
    if f.body_editing:
        raise ValueError("Apply or cancel the body pass before sculpting")
    if f.full_sculpt:
        raise ValueError("Apply or cancel the current full-model sculpt first")
    objects = []
    for obj in (
        f.head_result or s.result,
        f.body_result,
        f.eye_left,
        f.eye_right,
        f.teeth,
    ):
        if obj and obj not in objects:
            objects.append(obj)
    if not objects:
        raise ValueError("Wrap a head/body before sculpting")
    vertices = []
    faces = []
    parts = []
    for obj in objects:
        m.check_mesh(obj, obj.name)
        offset = len(vertices)
        points = m.evaluated_coordinates(context, obj)
        if len(points) != len(obj.data.vertices):
            raise ValueError("Base parts must not have topology-changing modifiers")
        vertices.extend(points.tolist())
        faces.extend([[v + offset for v in p.vertices] for p in obj.data.polygons])
        parts.append({"name": obj.name, "count": len(points), "hidden": obj.hide_get()})
    mesh = bpy.data.meshes.new("CM_FullModelSculpt")
    mesh.from_pydata(vertices, [], faces)
    mesh.update()
    sculpt = bpy.data.objects.new("CM_FullModelSculpt", mesh)
    context.collection.objects.link(sculpt)
    sculpt["cm_parts"] = json.dumps(parts)
    for obj in objects:
        obj.hide_set(True)
    f.full_sculpt = sculpt
    m.activate(context, sculpt)
    bpy.ops.object.mode_set(mode="SCULPT")


def full_sculpt_finish(context, s, f, apply=True):
    obj = f.full_sculpt
    m.check_mesh(obj, "Combined Sculpt Session")
    m.activate(context, obj)
    parts = json.loads(obj["cm_parts"])
    if apply and sum(p["count"] for p in parts) != len(obj.data.vertices):
        raise ValueError(
            "Combined sculpt topology changed; cancel or restore topology before applying"
        )
    for part in parts:
        if apply and part["name"] not in bpy.data.objects:
            raise ValueError("Original part was deleted: " + part["name"])
    if apply:
        points = m.coordinates(obj)
        offset = 0
        for part in parts:
            target = bpy.data.objects[part["name"]]
            key_from_world(
                target, "Full Model Sculpt", points[offset : offset + part["count"]]
            )
            offset += part["count"]
    for part in parts:
        if part["name"] in bpy.data.objects:
            bpy.data.objects[part["name"]].hide_set(part["hidden"])
    f.full_sculpt = None
    bpy.data.objects.remove(obj, do_unlink=True)
    if s.result:
        m.activate(context, s.result)
    from .guided import focus
    focus(context,[bpy.data.objects[part['name']] for part in parts if part['name'] in bpy.data.objects and not part['hidden']],f.front_axis)
    s.status = "Full-model sculpt " + (
        "applied to separate topology-preserving parts." if apply else "canceled."
    )
