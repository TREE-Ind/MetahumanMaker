"""Evaluate immutable detail geometry without loading the artist's Blender file."""

import hashlib
import json
import sys
import time
import traceback
from pathlib import Path

import bpy
import numpy as np
from mathutils import Matrix


def mesh(values, prefix):
    data = bpy.data.meshes.new(prefix)
    sizes = values[prefix + "_sizes"]
    loops = values[prefix + "_loops"]
    starts = (
        values[prefix + "_starts"]
        if prefix + "_starts" in values
        else np.concatenate(([0], np.cumsum(sizes)[:-1]))
    )
    faces = [
        loops[starts[i] : starts[i] + sizes[i]].tolist() for i in range(len(sizes))
    ]
    data.from_pydata(values[prefix + "_vertices"], values[prefix + "_edges"], faces)
    return data


def write_report(folder, data):
    temporary = folder / "report.tmp"
    temporary.write_text(json.dumps(data, indent=2))
    for attempt in range(51):
        try:
            temporary.replace(folder / "report.json")
            return
        except PermissionError:
            # Windows readers/scanners can briefly deny replacement even though
            # both sides close their handles promptly. Retry this same write.
            if attempt == 50:
                raise
            time.sleep(0.01)


def main():
    folder = Path(sys.argv[sys.argv.index("--") + 1]).resolve().parent
    request = json.loads((folder / "request.json").read_text())
    sys.path.insert(0, request["addon_root"])
    from create_metahuman import detail_layers as dl
    from create_metahuman import mesh_ops as m
    from create_metahuman.detail_recovery import LegacyReference, resolve
    from create_metahuman.worker_lifetime import watch_parent

    watch_parent(request["parent"])
    try:
        files = []
        for target in request["targets"]:
            index = target["index"]
            with np.load(folder / f"input-{index}.npz", allow_pickle=False) as values:
                detail_data = mesh(values, "detail")
                data = detail_data
                detail = bpy.data.objects.new("WorkerDetail", data)
                detail['cm_detail_subdivision_type']=target.get('subdivision_type','CATMULL_CLARK')
                bpy.context.scene.collection.objects.link(detail)
                detail.matrix_world = Matrix(values["detail_matrix"].tolist())
                for key_index in range(target["keys"]):
                    key = detail.shape_key_add(name=str(key_index))
                    key.data.foreach_set("co", values[f"key_{key_index}"].ravel())
                bpy.context.view_layer.update()
                correspondence_report = {}
                target_cache = {}

                def make_target(steps):
                    cache_key = tuple(steps)
                    if cache_key not in target_cache:
                        source_data = mesh(values, "source")
                        for name, domain in (
                            ("crease_edge", "EDGE"),
                            ("crease_vert", "POINT"),
                        ):
                            if name in values:
                                attribute = source_data.attributes.new(
                                    name, "FLOAT", domain
                                )
                                attribute.data.foreach_set("value", values[name])
                        world = dl.evaluate_surface(
                            bpy.context,
                            source_data,
                            Matrix(values["source_matrix"].tolist()),
                            target["level"],
                            target_mesh=detail_data,
                            correspondence_report=correspondence_report,
                            steps=steps,
                            subdivision_type=target.get('subdivision_type','CATMULL_CLARK'),
                        )
                        target_cache[cache_key] = m.to_local(detail, world)
                    return target_cache[cache_key]

                ancestors = []
                recovery = target.get("recovery")
                recovery_proof = []
                steps = target.get("steps", [target["level"]])
                if recovery:
                    detail["cm_detail_level"] = target["level"]
                    detail["cm_detail_subdivision_steps"] = steps
                    spec = LegacyReference(
                        [detail], recovery["parents"], set(recovery["explicit"])
                    )
                    spec.basis[0] = values["key_0"]
                    for node_index in range(1, len(recovery["levels"])):
                        prefix = f"legacy_{node_index}"
                        ancestor_data = mesh(values, prefix)
                        ancestor = bpy.data.objects.new(prefix, ancestor_data)
                        bpy.context.scene.collection.objects.link(ancestor)
                        ancestor.matrix_world = Matrix(
                            values[prefix + "_matrix"].tolist()
                        )
                        ancestor["cm_detail_level"] = recovery["levels"][node_index]
                        ancestor['cm_detail_subdivision_type']=recovery.get('types',['CATMULL_CLARK']*len(recovery['levels']))[node_index]
                        ancestor["cm_detail_subdivision_steps"] = recovery["steps"][
                            node_index
                        ]
                        for name, domain in (
                            ("crease_edge", "EDGE"),
                            ("crease_vert", "POINT"),
                        ):
                            if prefix + "_" + name in values:
                                attribute = ancestor_data.attributes.new(
                                    name, "FLOAT", domain
                                )
                                attribute.data.foreach_set(
                                    "value", values[prefix + "_" + name]
                                )
                        spec.objects.append(ancestor)
                        ancestors.append(ancestor)
                        spec.basis[node_index] = values[prefix + "_basis"]
                        spec.visible[node_index] = values[prefix + "_visible"]
                        if prefix + "_reference" in values:
                            spec.references[node_index] = values[prefix + "_reference"]
                    bpy.context.view_layer.update()
                    recovered = resolve(
                        bpy.context,
                        spec,
                        lambda: make_target([target["level"]]),
                        progress=lambda stage: write_report(
                            folder, {"status": "running", "stage": stage}
                        ),
                    )
                    reference, steps = recovered.reference, recovered.steps
                    recovery_proof = recovered.proof
                else:
                    reference = values["reference"]
                if recovery:
                    write_report(
                        folder,
                        {
                            "status": "running",
                            "stage": "Following the accepted character with recovered detail...",
                        },
                    )
                current_target = (
                    reference
                    if target.get("current_reference") and not recovery
                    else make_target(steps)
                )
                plan = dl.plan_from_target(None, detail, reference, current_target)
                arrays = {"target": plan["target"], "mesh": plan["mesh"]}
                arrays.update(
                    {
                        f"key_{i}": output
                        for i, (_, output) in enumerate(plan["updates"])
                    }
                )
                np.savez(folder / f"output-{index}.npz", **arrays)
                subdivision = None
                if target.get("subdivision"):
                    from create_metahuman import detail_subdivision

                    write_report(
                        folder,
                        {
                            "status": "running",
                            "stage": "Subdividing the sculpt in background...",
                        },
                    )
                    subdivision = detail_subdivision.build(
                        bpy.context, folder, target, plan, values, steps
                    )
                output = folder / f"output-{index}.npz"
                files.append(
                    {
                        "name": output.name,
                        "sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
                        "vertex_correspondence": correspondence_report,
                        "subdivision_steps": steps,
                        "recovered_reference": bool(recovery),
                        "recovery_proof": recovery_proof,
                    }
                )
                bpy.data.objects.remove(detail, do_unlink=True)
                bpy.data.meshes.remove(data)
                for ancestor in ancestors:
                    ancestor_data = ancestor.data
                    bpy.data.objects.remove(ancestor, do_unlink=True)
                    bpy.data.meshes.remove(ancestor_data)
        report = {"status": "complete", "files": files}
        if subdivision:
            report["subdivision"] = subdivision
    except Exception as exc:
        report = {
            "status": "failed",
            "error": str(exc),
            "traceback": traceback.format_exc(),
        }
    write_report(folder, report)
    if report["status"] != "complete":
        raise RuntimeError(report["error"])


if __name__ == "__main__":
    main()
