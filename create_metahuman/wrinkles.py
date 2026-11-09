"""Native MetaHuman wrinkle regions and UV-aware sculpt protection.

Masks are read from the user's optional Epic DCC content, never redistributed.
"""

from pathlib import Path
import hashlib, json
import bpy
import numpy as np
from . import mesh_ops as m


def source_stamp(context,s):
    """Evaluated head/detail geometry and UVs, independent of display materials."""
    digest=hashlib.sha256()
    for obj in (s.result,s.detail or s.result):
        if not obj:
            digest.update(b'NONE');continue
        graph=context.evaluated_depsgraph_get()
        evaluated=obj.evaluated_get(graph)
        mesh=evaluated.to_mesh(preserve_all_data_layers=True,depsgraph=graph)
        try:
            digest.update(np.asarray(evaluated.matrix_world,dtype='<f4').tobytes())
            values=np.empty(len(mesh.vertices)*3,dtype='<f4')
            mesh.vertices.foreach_get('co',values);digest.update(values.tobytes())
            for collection,field in ((mesh.loops,'vertex_index'),(mesh.polygons,'loop_total')):
                values=np.empty(len(collection),dtype='<i4')
                collection.foreach_get(field,values);digest.update(values.tobytes())
            values=np.empty(len(mesh.corner_normals)*3,dtype='<f4')
            mesh.corner_normals.foreach_get('vector',values);digest.update(values.tobytes())
            smooth=np.empty(len(mesh.polygons),dtype=np.bool_)
            mesh.polygons.foreach_get('use_smooth',smooth);digest.update(smooth.tobytes())
            uv=mesh.uv_layers.active
            if uv:
                values=np.empty(len(uv.data)*2,dtype='<f4')
                uv.data.foreach_get('uv',values);digest.update(values.tobytes())
        finally:evaluated.to_mesh_clear()
    return digest.hexdigest()


def seal_source(context,s,obj):
    obj['cm_wrinkle_source_stamp']=source_stamp(context,s)


def refresh_source_state(context,s,f):
    """Check on explicit actions, never panel redraw. Preserve every artist key.

    Legacy native layers carry their original neutral field. Compare it with
    the current evaluated source before accepting a first stamp; never label
    a legacy layer current merely because it lacks metadata.
    """
    layers=[obj for obj in (f.wm1,f.wm2,f.wm3) if obj]
    if not layers or not s.result:return False
    if context.mode!='OBJECT' or f.expression_active or f.body_editing or f.full_sculpt or f.restore_editing:
        return any(obj.get('cm_wrinkle_update_pending') for obj in layers)
    current=source_stamp(context,s)
    pending=False
    legacy_source=None
    try:
        for obj in layers:
            if obj.get('cm_wrinkle_update_pending') and not obj.get('cm_wrinkle_source_pending'):
                pending=True;continue
            previous=obj.get('cm_wrinkle_source_stamp')
            if previous==current:
                if obj.get('cm_wrinkle_source_pending'):
                    for prop in ('cm_wrinkle_source_pending','cm_wrinkle_update_pending'):
                        if prop in obj:del obj[prop]
                continue
            if previous is None:
                if legacy_source is None:
                    legacy_source=snapshot(context,s.detail or s.result,'CM_WrinkleSourceCheck')
                attribute=obj.data.attributes.get('.cm_wm_neutral')
                if attribute:
                    neutral=_read_vector(obj,'.cm_wm_neutral')
                elif not obj.get('cm_wrinkle_native') and obj.data.shape_keys:
                    neutral=np.empty((len(obj.data.vertices),3),dtype=np.float64)
                    obj.data.shape_keys.key_blocks[0].data.foreach_get('co',neutral.ravel())
                else:
                    neutral=None
                if neutral is not None:
                    matrix=np.asarray(obj.matrix_world)
                    neutral=neutral@matrix[:3,:3].T+matrix[:3,3]
                    mapped=transfer_uv(legacy_source,obj,m.evaluated_coordinates(context,legacy_source))
                    if np.max(np.abs(mapped-neutral))<2e-6:
                        obj['cm_wrinkle_source_stamp']=current
                        for prop in ('cm_wrinkle_source_pending','cm_wrinkle_update_pending'):
                            if prop in obj:del obj[prop]
                        continue
            if previous!=current:
                obj['cm_wrinkle_update_pending']=s.result
                obj['cm_wrinkle_source_pending']=True
                pending=True
    finally:
        if legacy_source:discard_layer(legacy_source)
    return pending


def native_bridge_ready(f):
    from .expressions import bridge_path
    folder=bridge_path(f)
    try:
        version=json.loads((folder/'CreateMetaHumanBridge.uplugin').read_text())['VersionName']
        return tuple(int(n) for n in version.split('.'))>=(0,5,0) and (folder/'Binaries/Win64/UnrealEditor-CreateMetaHumanBridge.dll').is_file()
    except (OSError,ValueError,KeyError):return False


def native_folder(s):
    return (
        Path(bpy.path.abspath(s.unreal_engine_dir))
        / "Engine/Plugins/MetaHuman/MetaHumanCharacter/Content/Optional/DCC/Masks"
    )


def native_mask(s, slot):
    if slot not in {"WM1", "WM2", "WM3"}:
        raise ValueError("Unknown wrinkle slot")
    folder = native_folder(s)
    files = sorted(folder.glob("head_" + slot.lower() + "_msk_*.tga"))
    if slot in {"WM1", "WM3"}:
        files += sorted(folder.glob("head_wm13_msk_*.tga"))
    expected = {"WM1": 5, "WM2": 3, "WM3": 3}[slot]
    if len(files) != expected:
        raise ValueError(
            "Install the optional MetaHuman DCC masks, or choose a custom Region Mask"
        )
    union = None
    sources = []
    for path in files:
        image = bpy.data.images.load(str(path), check_existing=False)
        try:
            image.colorspace_settings.name = "Non-Color"
            image.alpha_mode = "CHANNEL_PACKED"
            pixels = np.asarray(image.pixels[:], dtype=np.float32).reshape(
                image.size[1], image.size[0], 4
            )
            # 24-bit TGA has an opaque synthetic alpha; 32-bit alpha is a real mask.
            channels = 4 if path.read_bytes()[16] == 32 else 3
            region = pixels[..., :channels].max(axis=2)
            if union is not None and union.shape != region.shape:
                raise ValueError("Native mask dimensions do not match")
            union = region.copy() if union is None else np.maximum(union, region)
            sources.append(
                {
                    "file": path.name,
                    "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                    "channels": channels,
                }
            )
        finally:
            bpy.data.images.remove(image)
    output = bpy.data.images.new(
        "CM_" + slot + "_NativeRegion",
        width=union.shape[1],
        height=union.shape[0],
        alpha=False,
    )
    output.colorspace_settings.name = "Non-Color"
    rgba = np.ones((*union.shape, 4), dtype=np.float32)
    rgba[..., :3] = union[..., None]
    output.pixels.foreach_set(rgba.ravel())
    destination = Path(bpy.path.abspath(s.project_dir)) / "ReferenceMaps/NativeWrinkles"
    destination.mkdir(parents=True, exist_ok=True)
    output.filepath_raw = str(destination / (slot + "_Region.png"))
    output.file_format = "PNG"
    output.save()
    output.pack()
    (destination / (slot + "_sources.json")).write_text(
        json.dumps({"slot": slot, "sources": sources}, indent=2)
    )
    return output


def ensure_mask(s, f, slot):
    image = getattr(f, slot.lower() + "_mask")
    if image:
        return image
    # Native masks are valid only for the canonical face UV layout.
    from . import expressions

    data = expressions.bank(f)
    head = next(mesh for mesh in data["meshes"] if mesh["name"] == "head_lod0_mesh")
    expressions.uv_mapping(s.template, head)
    image = native_mask(s, slot)
    setattr(f, slot.lower() + "_mask", image)
    return image


def vertex_influence(obj, image):
    uv = obj.data.uv_layers.active
    if not uv:
        raise ValueError("Wrinkle layer needs head UVs")
    width, height = image.size
    if not width or not height:
        raise ValueError("Wrinkle region mask has no pixels")
    rgba = np.asarray(image.pixels[:], dtype=np.float32).reshape(height, width, 4)
    pixels = rgba[..., :3].max(axis=2)
    coordinates = np.empty(len(obj.data.loops) * 2, dtype=np.float32)
    uv.data.foreach_get("uv", coordinates)
    coordinates = coordinates.reshape(-1, 2) * [width, height] - 0.5
    lower = np.floor(coordinates).astype(int)
    fraction = coordinates - lower
    x0 = np.clip(lower[:, 0], 0, width - 1)
    y0 = np.clip(lower[:, 1], 0, height - 1)
    x1 = np.clip(lower[:, 0] + 1, 0, width - 1)
    y1 = np.clip(lower[:, 1] + 1, 0, height - 1)
    tx, ty = fraction.T
    samples = (pixels[y0, x0] * (1 - tx) + pixels[y0, x1] * tx) * (1 - ty) + (
        pixels[y1, x0] * (1 - tx) + pixels[y1, x1] * tx
    ) * ty
    vertices = np.empty(len(obj.data.loops), dtype=np.int32)
    obj.data.loops.foreach_get("vertex_index", vertices)
    result = np.zeros(len(obj.data.vertices), dtype=np.float32)
    np.maximum.at(result, vertices, samples)
    return np.clip(result, 0, 1)


def sculpt_area(s, f, obj, slot):
    m.check_mesh(obj, "Wrinkle layer")
    weights = (
        vertex_influence(obj, ensure_mask(s, f, slot))
        if f.wrinkle_protect_area
        else np.ones(len(obj.data.vertices))
    )
    mask = obj.data.attributes.get(".sculpt_mask") or obj.data.attributes.new(
        ".sculpt_mask", "FLOAT", "POINT"
    )
    mask.data.foreach_set("value", np.asarray(1 - weights, dtype=np.float32))
    obj.data.update()
    return weights


def snapshot(context, source, name):
    """Capture evaluated detail/UVs without dropping subdivision or shape layers."""
    depsgraph = context.evaluated_depsgraph_get()
    mesh = bpy.data.meshes.new_from_object(
        source.evaluated_get(depsgraph),
        preserve_all_data_layers=True,
        depsgraph=depsgraph,
    )
    obj = bpy.data.objects.new(name, mesh)
    context.collection.objects.link(obj)
    obj.matrix_world = source.matrix_world.copy()
    return obj


def uv_correspondence(reference, target):
    """Build reusable UV triangle weights once for keys, masks and skin groups."""
    from mathutils.bvhtree import BVHTree

    uv = reference.data.uv_layers.active
    target_uv = target.data.uv_layers.active
    if not uv or not target_uv:
        raise ValueError("Native wrinkle detail needs preserved head UVs")
    points = np.asarray([loop.uv[:] for loop in uv.data])
    reference.data.calc_loop_triangles()
    triangles = np.asarray(
        [triangle.loops[:] for triangle in reference.data.loop_triangles], dtype=int
    )
    corners = points[triangles]
    a, b = corners[:, 1] - corners[:, 0], corners[:, 2] - corners[:, 0]
    valid = np.abs(a[:, 0] * b[:, 1] - a[:, 1] * b[:, 0]) > 1e-16
    triangles = triangles[valid]
    tree = BVHTree.FromPolygons(
        np.column_stack((points, np.zeros(len(points)))).tolist(),
        triangles.tolist(),
        all_triangles=True,
    )
    loop_vertices = np.asarray([loop.vertex_index for loop in reference.data.loops])
    indices=[]
    weights=[]
    vertices=[]
    cache = {}
    for loop, entry in zip(target.data.loops, target_uv.data):
        key = tuple(entry.uv)
        if key not in cache:
            location, _, index, distance = tree.find_nearest((*key, 0))
            if index is None or distance > 5e-5:
                raise ValueError("Detail UVs are outside the native head layout")
            tri = triangles[index]
            p0, p1, p2 = points[tri]
            x, y = np.asarray(location[:2]) - p0
            u = p1 - p0
            v = p2 - p0
            det = u[0] * v[1] - u[1] * v[0]
            w1 = (x * v[1] - y * v[0]) / det
            w2 = (u[0] * y - u[1] * x) / det
            cache[key] = (loop_vertices[tri],np.asarray([1-w1-w2,w1,w2]))
        source,weight=cache[key]
        indices.append(source)
        weights.append(weight)
        vertices.append(loop.vertex_index)
    counts=np.bincount(vertices,minlength=len(target.data.vertices))
    if np.any(counts == 0):
        raise ValueError("Detail has vertices without UV correspondence")
    return (np.asarray(indices,dtype=np.int32),np.asarray(weights),
            np.asarray(vertices,dtype=np.int32),counts)


def transfer_uv(reference, target, values, *, mapping=None):
    """Interpolate point data in the preserved head UVs, including dense detail."""
    values=np.asarray(values)
    if mapping is None and m.topology(reference)==m.topology(target):
        return values.copy()
    shape=values.shape[1:]
    flat=values.reshape(len(values),-1)
    indices,weights,vertices,counts=uv_correspondence(reference,target) if mapping is None else mapping
    result=np.zeros((len(target.data.vertices),flat.shape[1]))
    for corner in range(3):
        np.add.at(result,vertices,flat[indices[:,corner]]*weights[:,corner,None])
    return (result/counts[:,None]).reshape((len(result),)+shape)


def _vector_attribute(obj, name, values):
    attribute = obj.data.attributes.get(name) or obj.data.attributes.new(
        name, "FLOAT_VECTOR", "POINT"
    )
    attribute.data.foreach_set("vector", np.asarray(values, dtype=np.float32).ravel())


def _read_vector(obj, name):
    attribute = obj.data.attributes.get(name)
    if (
        not attribute
        or attribute.domain != "POINT"
        or attribute.data_type != "FLOAT_VECTOR"
    ):
        raise ValueError("Wrinkle pose basis is missing: " + name)
    values = np.empty((len(obj.data.vertices), 3), dtype=np.float32)
    attribute.data.foreach_get("vector", values.ravel())
    return values.astype(float)


def add_proxies(context, owner, targets, head):
    """Attach display-only posed internals, cleaning partial creation on error."""
    created = []
    try:
        for target, world in targets:
            if target == head:continue
            proxy = snapshot(context, target, owner.name + "_" + target.name + "_Proxy")
            created.append(proxy)
            proxy.data.vertices.foreach_set("co", m.to_local(proxy, world).ravel())
            proxy.data.update()
            matrix = proxy.matrix_world.copy()
            proxy.parent = owner
            proxy.matrix_world = matrix
            proxy["cm_role"] = "WRINKLE_PROXY"
            proxy["cm_wrinkle_owner"] = owner
        return created
    except Exception:
        for obj in created:
            mesh = obj.data
            bpy.data.objects.remove(obj, do_unlink=True)
            if not mesh.users:bpy.data.meshes.remove(mesh)
        raise


def restore_proxies(context, s, f, owner):
    """Regenerate internals from accepted meshes and the retained native pose."""
    from . import expressions
    data = expressions.bank(f)
    poses = [pose for pose in data['poses'] if pose['name'] == owner.get('cm_wrinkle_pose')]
    if len(poses) != 1:
        raise ValueError('The returned wrinkle expression is absent from the native library')
    return add_proxies(context, owner, expressions.evaluate_pose(context, s, f, data, poses[0]), s.result)


def posed_layer(context, s, f, data, pose, slot):
    """Pose an evaluated sculpt snapshot and retain its inverse neutral basis."""
    from . import expressions

    if f.expression_active or f.body_editing or f.full_sculpt or f.restore_editing:
        raise ValueError(
            "Finish the current expression/body/repair before creating wrinkle bases"
        )
    targets = expressions.evaluate_pose(context, s, f, data, pose)
    head = next(world for obj, world in targets if obj == s.result)
    neutral_head = m.evaluated_coordinates(context, s.result)
    linear = expressions.head_pose_linear(context, s, f, data=data, pose=pose)
    obj = snapshot(context, s.detail or s.result, s.character_name + "_" + slot)
    try:
        neutral = m.evaluated_coordinates(context, obj)
        fields = np.column_stack((neutral_head, head, linear.reshape(len(head), 9)))
        mapped = transfer_uv(s.result, obj, fields)
        jacobian = mapped[:, 6:].reshape(-1, 3, 3)
        if np.max(np.linalg.cond(jacobian)) > 1e6:
            raise ValueError(
                "Wrinkle pose has a singular deformation; choose another base"
            )
        posed = mapped[:, 3:6] + np.einsum(
            "ni,nij->nj", neutral - mapped[:, :3], jacobian
        )
        local = m.to_local(obj, posed)
        neutral_local = m.to_local(obj, neutral)
        transform = np.asarray(obj.matrix_world)[:3, :3].T
        local_linear = transform @ jacobian @ np.linalg.inv(transform)
        obj.data.vertices.foreach_set("co", local.ravel())
        _vector_attribute(obj, ".cm_wm_neutral", neutral_local)
        _vector_attribute(obj, ".cm_wm_posed", local)
        for row in range(3):
            _vector_attribute(obj, ".cm_wm_linear_" + str(row), local_linear[:, row, :])
        obj.shape_key_add(name="Basis")
        obj.shape_key_add(name="Wrinkle Sculpt").value = 1
        obj.active_shape_key_index = 1
        obj["cm_role"] = "WRINKLE"
        obj["cm_wrinkle_native"] = True
        obj["cm_wrinkle_pose"] = pose["name"]
        obj["cm_wrinkle_slot"] = slot
        seal_source(context,s,obj)
        obj.data.update()
        add_proxies(context, obj, targets, s.result)
        return obj
    except Exception:
        discard_layer(obj)
        raise


def neutralized_copy(context, obj):
    """Remove the pose, retaining only sculpt movement in neutral coordinates."""
    if not obj.get("cm_wrinkle_native"):
        raise ValueError("This wrinkle layer has no native inverse basis")
    result = snapshot(context, obj, "CM_WrinkleNeutralBake")
    try:
        neutral = _read_vector(result, ".cm_wm_neutral")
        posed = _read_vector(result, ".cm_wm_posed")
        linear = np.stack(
            [_read_vector(result, ".cm_wm_linear_" + str(row)) for row in range(3)],
            axis=1,
        )
        current = np.asarray([v.co[:] for v in result.data.vertices])
        displacement = np.linalg.solve(
            linear.transpose(0, 2, 1), (current - posed)[..., None]
        )[..., 0]
        output = neutral + displacement
        if not np.isfinite(output).all():
            raise ValueError("Wrinkle neutralization produced non-finite geometry")
        result.data.vertices.foreach_set("co", output.ravel())
        result.data.update()
        return result
    except Exception:
        mesh = result.data
        bpy.data.objects.remove(result, do_unlink=True)
        bpy.data.meshes.remove(mesh)
        raise


def _new_layer(context, s, f, slot):
    if f.wrinkle_native_bases:
        from . import expressions

        data = expressions.bank(f)
        if slot not in data.get("wrinkle_poses", {}):
            raise ValueError("Prepare Native Wrinkle Bases first")
        return posed_layer(
            context, s, f, data, data["poses"][data["wrinkle_poses"][slot]], slot
        )
    obj = snapshot(context, s.detail or s.result, s.character_name + "_" + slot)
    obj.shape_key_add(name="Basis")
    obj.shape_key_add(name="Wrinkle Sculpt").value = 1
    obj.active_shape_key_index = 1
    obj["cm_role"] = "WRINKLE"
    obj["cm_wrinkle_slot"] = slot
    _vector_attribute(obj,'.cm_wm_neutral',np.asarray([v.co[:] for v in obj.data.vertices]))
    seal_source(context,s,obj)
    return obj


def replace_layer(context, s, f, slot, new):
    old = getattr(f, slot.lower())
    if old and old != new:
        old["cm_wrinkle_archived"] = True
        old.hide_set(True)
        old.hide_render = True
        for child in old.children:
            child.hide_set(True)
            child.hide_render = True
    setattr(f, slot.lower(), new)
    new["cm_role"] = "WRINKLE"
    new["cm_wrinkle_slot"] = slot
    m.activate(context, new)
    from .guided import focus

    focus(context, [new, *new.children])
    return new


def discard_layer(obj):
    owned = [child for child in obj.children if child.get("cm_wrinkle_owner") == obj]
    for item in owned + [obj]:
        mesh = item.data
        bpy.data.objects.remove(item, do_unlink=True)
        bpy.data.meshes.remove(mesh)


def reset_layer(context, s, f, slot):
    if context.mode != "OBJECT":
        raise ValueError("Finish sculpting before resetting a wrinkle base")
    return replace_layer(context, s, f, slot, _new_layer(context, s, f, slot))


def save_layer(context, s, f, filepath):
    obj = getattr(f, f.wrinkle_slot.lower())
    m.check_mesh(obj, "Wrinkle layer")
    if context.mode != "OBJECT":
        raise ValueError("Finish sculpting before saving a wrinkle base")
    obj["cm_wrinkle_slot"] = f.wrinkle_slot
    obj["cm_wrinkle_basis_schema"] = 1
    bpy.data.libraries.write(
        str(Path(filepath).resolve()), {obj}, path_remap="ABSOLUTE", compress=True
    )


def load_layer(context, s, f, filepath):
    """Reuse neutral fold deltas from .blend, or import a matching posed OBJ/FBX."""
    if (
        context.mode != "OBJECT"
        or f.body_editing
        or f.full_sculpt
        or f.restore_editing
        or f.expression_active
    ):
        raise ValueError(
            "Finish the current sculpt, body or expression preview before loading wrinkles"
        )
    path = Path(filepath).resolve()
    slot = f.wrinkle_slot
    containers = [
        bpy.data.objects,
        bpy.data.meshes,
        bpy.data.shape_keys,
        bpy.data.materials,
        bpy.data.images,
        bpy.data.texts,
    ]
    before = [set(items) for items in containers]
    imported = []
    new = None
    temporary = []
    try:
        with m.selection_state(context):
            if path.suffix.lower() == ".blend":
                with bpy.data.libraries.load(str(path), link=False) as (source, target):
                    if len(source.objects) != 1:
                        raise ValueError("Choose a file created with Save Wrinkle Base")
                    target.objects = source.objects
                source = target.objects[0]
                context.collection.objects.link(source)
            elif path.suffix.lower() in {".obj", ".fbx"}:
                if path.suffix.lower() == ".obj":
                    bpy.ops.wm.obj_import(filepath=str(path))
                else:
                    bpy.ops.import_scene.fbx(filepath=str(path))
                objects = [
                    obj
                    for obj in set(bpy.data.objects) - before[0]
                    if obj.type == "MESH"
                ]
                if len(objects) != 1:
                    raise ValueError("Custom wrinkle sculpt must contain one head mesh")
                source = objects[0]
            else:
                raise ValueError(
                    "Use a saved wrinkle .blend or a UV-preserving sculpted OBJ/FBX"
                )
            imported = [
                set(items) - previous for items, previous in zip(containers, before)
            ]
            m.check_mesh(source, "Custom wrinkle")
            context.view_layer.update()
            if path.suffix.lower() == ".blend":
                if source.get("cm_wrinkle_slot") != slot:
                    raise ValueError(
                        "Select "
                        + str(source.get("cm_wrinkle_slot", "the matching slot"))
                        + " before loading this saved base"
                    )
                if source.get("cm_wrinkle_basis_schema", 1) != 1:
                    raise ValueError("Unsupported custom wrinkle basis version")
            new = _new_layer(context, s, f, slot)
            if path.suffix.lower() == ".blend":
                if source.get("cm_wrinkle_native"):
                    recovered = neutralized_copy(context, source)
                    temporary.append(recovered)
                    local = np.asarray([v.co[:] for v in recovered.data.vertices])
                    delta = local - _read_vector(recovered, ".cm_wm_neutral")
                    delta = delta @ np.asarray(source.matrix_world)[:3, :3].T
                    source_matrix = np.asarray(source.matrix_world)
                    source_neutral = (
                        _read_vector(recovered, ".cm_wm_neutral")
                        @ source_matrix[:3, :3].T
                        + source_matrix[:3, 3]
                    )
                    reference = recovered
                else:
                    reference = snapshot(context, source, "CM_CustomWrinkleReference")
                    temporary.append(reference)
                    current = m.evaluated_coordinates(context, source)
                    key = (
                        source.data.shape_keys.key_blocks.get("Wrinkle Sculpt")
                        if source.data.shape_keys
                        else None
                    )
                    if not key:
                        raise ValueError("Saved wrinkle layer has no sculpt key")
                    strength = key.value
                    try:
                        key.value = 0
                        context.view_layer.update()
                        neutral = m.evaluated_coordinates(context, source)
                    finally:
                        key.value = strength
                        context.view_layer.update()
                    delta = current - neutral
                    source_neutral = neutral
                fields = transfer_uv(
                    reference, new, np.column_stack((source_neutral, delta))
                )
                new_matrix = np.asarray(new.matrix_world)
                target_neutral = (
                    (
                        _read_vector(new, ".cm_wm_neutral") @ new_matrix[:3, :3].T
                        + new_matrix[:3, 3]
                    )
                    if new.get("cm_wrinkle_native")
                    else m.evaluated_coordinates(context, new)
                )
                from .geometry import similarity

                rotation, scale, _ = similarity(fields[:, :3], target_neutral)
                world_delta = fields[:, 3:] @ rotation * scale
                local_delta = (
                    world_delta @ np.linalg.inv(np.asarray(new.matrix_world)[:3, :3]).T
                )
                if new.get("cm_wrinkle_native"):
                    linear = np.stack(
                        [
                            _read_vector(new, ".cm_wm_linear_" + str(row))
                            for row in range(3)
                        ],
                        axis=1,
                    )
                    local_delta = np.einsum("ni,nij->nj", local_delta, linear)
                key = new.data.shape_keys.key_blocks["Wrinkle Sculpt"]
                baseline = np.asarray([v.co[:] for v in key.relative_key.data])
                key.data.foreach_set("co", (baseline + local_delta).ravel())
                new.data.update()
            else:
                from .body_library import coordinates_from_uvs
                from . import workflow

                positions = coordinates_from_uvs(new, source)
                workflow.key_from_world(new, "Wrinkle Sculpt", positions)
            context.view_layer.update()
        replace_layer(context, s, f, slot, new)
        s.status = slot + " custom sculpt loaded; character geometry preserved."
        return new
    except Exception:
        if new:
            discard_layer(new)
        raise
    finally:
        for obj in temporary:
            mesh = obj.data
            bpy.data.objects.remove(obj, do_unlink=True)
            bpy.data.meshes.remove(mesh)
        if not imported:
            imported = [
                set(items) - previous for items, previous in zip(containers, before)
            ]
        for index, (items, created) in enumerate(zip(containers, imported)):
            for item in created:
                try:
                    if index == 0:
                        items.remove(item, do_unlink=True)
                    elif item.users <= int(item.use_fake_user):
                        items.remove(item)
                except ReferenceError:
                    pass  # Removing a mesh can also dispose of its Key ID.
