"""Build a new sculpt level in a worker, then publish it without replacing its archive."""

import hashlib
import json
import struct

import bpy
import numpy as np

from . import detail_layers as dl
from . import mesh_ops as m


def signature(obj):
    digest = hashlib.sha256()
    digest.update(
        json.dumps(
            [
                obj.active_shape_key_index,
                obj.show_only_shape_key,
                [g.name for g in obj.vertex_groups],
                [
                    (k.name, k.vertex_group, k.interpolation)
                    for k in obj.data.shape_keys.key_blocks
                ],
                obj.data.shape_keys.use_relative,
                obj.data.shape_keys.eval_time,
                [(uv.name, uv.active_render) for uv in obj.data.uv_layers],
                obj.data.uv_layers.active_index,
            ]
        ).encode()
    )
    for attribute in obj.data.attributes:
        digest.update(
            json.dumps([attribute.name, attribute.domain, attribute.data_type]).encode()
        )
        formats = {
            "FLOAT": ("value", 1, np.float32),
            "INT": ("value", 1, np.int32),
            "INT8": ("value", 1, np.int32),
            "INT32_2D": ("value", 2, np.int32),
            "BOOLEAN": ("value", 1, np.bool_),
            "FLOAT_VECTOR": ("vector", 3, np.float32),
            "FLOAT2": ("vector", 2, np.float32),
            "FLOAT_COLOR": ("color", 4, np.float32),
            "BYTE_COLOR": ("color", 4, np.float32),
        }
        if attribute.data_type in formats:
            field, width, dtype = formats[attribute.data_type]
            values = np.empty(len(attribute.data) * width, dtype=dtype)
            attribute.data.foreach_get(field, values)
            digest.update(values.tobytes())
        else:
            for item in attribute.data:
                digest.update(np.asarray(item.value).tobytes())
    buffer = bytearray()
    for vertex in obj.data.vertices:
        for group in vertex.groups:
            buffer.extend(struct.pack("<IIf", vertex.index, group.group, group.weight))
        if len(buffer) > 131072:
            digest.update(buffer)
            buffer.clear()
    digest.update(buffer)
    return digest.hexdigest()


def snapshot(detail, path):
    data = detail.data.copy()
    try:
        # Keep empty slots and face assignments without copying shader/image IDs.
        slots = len(data.materials)
        indices = np.empty(len(data.polygons), dtype=np.int32)
        data.polygons.foreach_get("material_index", indices)
        data.materials.clear()
        for _ in range(slots):
            data.materials.append(None)
        data.polygons.foreach_set("material_index", indices)
        for name in list(data.keys()):
            del data[name]
        if data.shape_keys:
            data.shape_keys.animation_data_clear()
        bpy.data.libraries.write(str(path), {data}, fake_user=True)
        return data.name
    finally:
        bpy.data.meshes.remove(data)


def build(context, folder, target, plan, values, steps):
    """Preserve Blender's native UV/attribute interpolation in the isolated process."""

    def stage(message):
        print("SUBDIVIDE_STAGE", message, flush=True)

    stage("load mesh snapshot")
    with bpy.data.libraries.load(str(folder / "input-subdivide.blend"), link=False) as (
        _,
        loaded,
    ):
        loaded.meshes = [target["subdivision"]["mesh"]]
    data = loaded.meshes[0]
    obj = bpy.data.objects.new("Subdivide Artist Surface", data)
    context.scene.collection.objects.link(obj)
    from mathutils import Matrix

    obj.matrix_world = Matrix(values["detail_matrix"].tolist())
    if not obj.vertex_groups:
        for name in target["subdivision"]["groups"]:
            obj.vertex_groups.new(name=name)
    if [group.name for group in obj.vertex_groups] != target["subdivision"]["groups"]:
        raise ValueError(
            "Saved sculpt mask groups do not match the subdivision snapshot"
        )
    obj.active_shape_key_index = target["subdivision"]["active_key"]
    obj.show_only_shape_key = target["subdivision"]["only_key"]
    data.vertices.foreach_set("co", plan["mesh"].ravel())
    for key, (_, points) in zip(data.shape_keys.key_blocks, plan["updates"]):
        key.data.foreach_set("co", points.ravel())
    data.update()
    stage("evaluate updated sculpt")
    context.view_layer.update()
    visible = m.evaluated_coordinates(context, obj)
    stage("copy visible artist surface")
    result = m.duplicate(context, obj, "New Sculpt Level")
    result.data.vertices.foreach_set("co", m.to_local(result, visible).ravel())
    result.data.update()
    m.activate(context, result)
    modifier = result.modifiers.new("Additional Detail Level", "SUBSURF")
    modifier.subdivision_type=target.get('subdivision_type','CATMULL_CLARK')
    modifier.levels = 1
    stage("apply subdivision")
    bpy.ops.object.modifier_apply(modifier=modifier.name)
    result.shape_key_add(name="Basis")
    result.shape_key_add(name="Scan Detail").value = 1
    from .detail_worker import mesh

    stage("evaluate pristine reference")
    creased = any(
        name in values and np.any(values[name] != 0)
        for name in ("crease_edge", "crease_vert")
    )
    if not creased:
        # The accepted neutral at the old level has already been established.
        # Factor out that prefix of the modifier history. Recreate its topology
        # without artist UV/mask/crease fields, then apply the same final step.
        source_data = mesh(values, "detail")
        source_data.vertices.foreach_set("co", plan["target"].ravel())
        reference = dl.evaluate_surface(
            context,
            source_data,
            obj.matrix_world,
            1,
            target_mesh=result.data,
            steps=[1],
            subdivision_type=target.get('subdivision_type','CATMULL_CLARK'),
        )
        reference_strategy = "verified-neutral-prefix"
    else:
        source_data = mesh(values, "source")
        for name, domain in [("crease_edge", "EDGE"), ("crease_vert", "POINT")]:
            if name in values:
                attr = source_data.attributes.new(name, "FLOAT", domain)
                attr.data.foreach_set("value", values[name])
        reference = dl.evaluate_surface(
            context,
            source_data,
            Matrix(values["source_matrix"].tolist()),
            target["level"] + 1,
            target_mesh=result.data,
            steps=steps + [1],
            subdivision_type=target.get('subdivision_type','CATMULL_CLARK'),
        )
        reference_strategy = "accepted-source-history"
    reference = m.to_local(result, reference)
    attribute = result.data.attributes.get(dl.REFERENCE)
    if attribute:
        result.data.attributes.remove(attribute)
    attribute = result.data.attributes.new(dl.REFERENCE, "FLOAT_VECTOR", "POINT")
    attribute.data.foreach_set("vector", reference.ravel())
    output = folder / "output-subdivide.blend"
    stage("save completed subdivision")
    bpy.data.libraries.write(str(output), {result.data}, fake_user=True)
    return {
        "name": output.name,
        "sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
        "mesh": result.data.name,
        "vertices": len(result.data.vertices),
        "topology": m.topology(result),
        "subdivision_steps": steps + [1],
        "reference_strategy": reference_strategy,
    }


def publish(context, job):
    target = job["targets"][0]
    old = target["detail"]
    item = job["report"]["subdivision"]
    path = job["folder"] / "output-subdivide.blend"
    if hashlib.sha256(path.read_bytes()).hexdigest() != item["sha256"]:
        raise ValueError(
            "Subdivision output checksum changed; previous sculpt retained"
        )
    with bpy.data.libraries.load(str(path), link=False) as (_, loaded):
        loaded.meshes = [item["mesh"]]
    data = loaded.meshes[0]
    result = None
    try:
        result = old.copy()
        result.data = data
        from . import simple_shading
        shading_source=simple_shading.detach_copy(result)
        result.show_only_shape_key = False
        result.active_shape_key_index = 1
        context.scene.collection.objects.link(result)
        result.name = old.name + "_LV" + str(target["level"] + 1)
        expected = len(old.data.vertices) + len(old.data.edges) + len(old.data.polygons)
        if (
            len(data.vertices) != expected
            or len(data.vertices) != item["vertices"]
            or m.topology(result) != item["topology"]
        ):
            raise ValueError(
                "Subdivision output topology is invalid; previous sculpt retained"
            )
        reference = dl.read_reference(result)
        if (
            reference is None
            or not np.isfinite(reference).all()
            or not np.isfinite(m.coordinates(result, world=False)).all()
        ):
            raise ValueError(
                "Subdivision output geometry is invalid; previous sculpt retained"
            )
        if item["subdivision_steps"] != job["report"]["files"][0][
            "subdivision_steps"
        ] + [1]:
            raise ValueError("Subdivision history is invalid; previous sculpt retained")
        if len(data.materials) != len(old.data.materials):
            raise ValueError(
                "Sculpt material slots changed during subdivision; previous sculpt retained"
            )
        for index, material in enumerate(old.data.materials):
            data.materials[index] = material
        result["cm_detail_level"] = target["level"] + 1
        result["cm_detail_subdivision_steps"] = item["subdivision_steps"]
        result["cm_detail_previous"] = old
        if "cm_detail_previous_selected" in result:
            del result["cm_detail_previous_selected"]
        dl.bind(context, target["source"], result, reference)
        if shading_source is not None:simple_shading.apply(context,target['source'],result,source=shading_source)
    except Exception:
        if result:
            helper=result.get(simple_shading.HELPER)
            if helper and helper.get('cm_simple_shading_owner')==result:simple_shading.restore(context,result)
            bpy.data.objects.remove(result, do_unlink=True)
        bpy.data.meshes.remove(data)
        raise
    s, f = context.scene.cm_settings, context.scene.cm_features
    if s.part == "FULL_BODY" and job["part"] == "BODY":
        f.body_detail = result
    else:
        s.detail = result
    old.hide_set(True)
    old.hide_render = True
    m.activate(context, result)
    s.status = (
        f"Detail subdivided to LV{target['level'] + 1}; previous sculpt retained."
    )
