"""Cancellable detail-follow jobs; apply only to an unchanged editor snapshot."""

import hashlib
import json
import os
import subprocess
import sys
import uuid
from pathlib import Path

import bpy
import numpy as np
from bpy.app.handlers import persistent

from . import detail_layers as dl
from . import mesh_ops as m

_JOB = None
_REAPING = []


def running():
    return _JOB is not None


def thread_arguments():
    # Blender's subdivision vertex order can change with its global thread
    # setting. Match an explicit parent override; otherwise inherit auto.
    for index, value in enumerate(sys.argv[:-1]):
        if value in {"--threads", "-t"}:
            return ["--threads", str(int(sys.argv[index + 1]))]
    return []


def points(collection):
    values = np.empty((len(collection), 3), dtype=np.float32)
    collection.foreach_get("co", values.ravel())
    return values


def mesh_arrays(obj):
    data = obj.data
    loops = np.empty(len(data.loops), dtype=np.int32)
    sizes = np.empty(len(data.polygons), dtype=np.int32)
    starts = np.empty(len(data.polygons), dtype=np.int32)
    data.loops.foreach_get("vertex_index", loops)
    data.polygons.foreach_get("loop_total", sizes)
    data.polygons.foreach_get("loop_start", starts)
    return {
        "vertices": points(data.vertices),
        "loops": loops,
        "sizes": sizes,
        "starts": starts,
        "edges": m.edges(obj),
        "matrix": np.asarray(obj.matrix_world),
    }


def key_arrays(detail):
    return list(detail.data.shape_keys.key_blocks) if detail.data.shape_keys else []


def fingerprint(context, source, detail):
    from . import sculpt_modifiers
    digest = hashlib.sha256()
    digest.update(sculpt_modifiers.settings(detail).encode())
    digest.update(sculpt_modifiers.multires_digest(context, detail))
    digest.update(sculpt_modifiers.dependencies_digest(context, detail))
    digest.update(
        json.dumps(
            [
                detail.get("cm_detail_level"),
                dl.subdivision_type(detail),
                detail.get('cm_detail_bound_subdivision_type','CATMULL_CLARK'),
                detail.get("cm_detail_source_shape"),
                detail.get("cm_detail_source_topology"),
                detail.get("cm_detail_topology"),
                list(detail.get("cm_detail_subdivision_steps", [])),
                list(detail.get("cm_detail_bound_steps", [])),
                list(detail.get("cm_detail_reference_matrix", [])),
                detail.get("cm_detail_previous").as_pointer()
                if isinstance(detail.get("cm_detail_previous"), bpy.types.Object)
                else str(detail.get("cm_detail_previous", "")),
            ]
        ).encode()
    )
    digest.update(dl.signature(context, source).encode())
    digest.update(m.topology(source).encode())
    digest.update(m.topology(detail).encode())
    digest.update(np.asarray(detail.matrix_world, dtype="<f8").tobytes())
    digest.update(points(detail.data.vertices).tobytes())
    reference = dl.read_reference(detail)
    if reference is not None:
        digest.update(reference.astype("<f4").tobytes())
    for key in key_arrays(detail):
        digest.update(
            json.dumps([key.name, key.value, key.relative_key.name, key.mute]).encode()
        )
        digest.update(points(key.data).tobytes())
    for name in ("crease_edge", "crease_vert"):
        attribute = source.data.attributes.get(name)
        if attribute:
            values = np.empty(len(attribute.data), dtype=np.float32)
            attribute.data.foreach_get("value", values)
            digest.update(name.encode() + values.tobytes())
    return digest.hexdigest()


def scene_state(scene):
    s, f = scene.cm_settings, scene.cm_features
    return (
        s.project_dir,
        s.character_name,
        f.body_backend,
        f.body_editing,
        f.expression_active,
        bool(f.full_sculpt),
        f.restore_editing,
        tuple(
            obj.as_pointer() if obj else 0
            for obj in (s.result, s.detail, f.body_result, f.body_detail)
        ),
    )


def _write(path, data):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(data, indent=2))
    temporary.replace(path)


def start(context, s, f, action="SYNC", part="HEAD"):
    staging={}
    try:
        return _start(context,s,f,action,part,_staging=staging)
    except Exception:
        # A serialization/Popen failure has no worker to clean its carriers.
        # Never remove inputs from a process that may already be consuming them.
        process=staging.get('process')
        if staging.get('folder') and (process is None or process.poll() is not None):
            cleanup(staging)
        raise


def _start(context, s, f, action="SYNC", part="HEAD", *, _staging):
    global _JOB
    from . import bake_jobs, native_body
    from . import wrinkle_jobs
    if wrinkle_jobs.running():
        raise ValueError('Finish or cancel the retained wrinkle update first')

    if _JOB:
        raise ValueError("A detail update is running; finish or cancel it first")
    if action not in {
        "SYNC",
        "SYNC_HEAD",
        "SCULPT",
        "FINAL_MAPS",
        "BODY_APPLY",
        "RECOVER",
        "SUBDIVIDE",
    }:
        raise ValueError("Unknown detail continuation")
    if (
        context.mode != "OBJECT"
        or bake_jobs.running()
        or f.full_sculpt
        or f.restore_editing
        or f.expression_active
    ):
        raise ValueError(
            "Finish the current edit or bake before updating sculpt detail"
        )
    if f.body_editing and action != "BODY_APPLY":
        raise ValueError("Apply or cancel the body pass first")
    if (
        action == "BODY_APPLY"
        and f.body_backend == "NATIVE"
        and not native_body.can_apply()
    ):
        raise ValueError("Wait for the current anatomical preview before applying")
    pairs = [(s.result, s.detail), (f.body_result, f.body_detail)]
    if action=='FINAL_MAPS':
        from . import textures
        pairs=[(f.body_result,f.body_detail) if role=='BODY' and s.part=='FULL_BODY' else (s.result,s.detail) for role in textures.final_parts(s,f)]
    elif action == 'SYNC_HEAD':
        pairs = [pairs[0]]
    elif action in {"SCULPT", "RECOVER", "SUBDIVIDE"}:
        pairs = [pairs[1] if s.part == "FULL_BODY" and part == "BODY" else pairs[0]]
    if action in {'SCULPT', 'SUBDIVIDE', 'FINAL_MAPS'}:
        from . import scan_projection
        scan_projection.require_applied(*(detail for _source, detail in pairs))
    if action == "SUBDIVIDE":
        source, detail = pairs[0]
        m.check_mesh(source, "Accepted part")
        m.check_mesh(detail, "Sculpt layer")
        from . import simple_shading
        if simple_shading.blocking_modifiers(detail) or not detail.data.shape_keys:
            raise ValueError("Finish the active detail modifiers before subdivision")
        if int(detail.get("cm_detail_level", 1)) >= 3:
            raise ValueError("Detail is already at LV3")
    from . import resource_budget
    if any(detail for _source,detail in pairs):
        resource_budget.require('sculpt detail processing',512*resource_budget.MIB+
            resource_budget.meshes([obj for pair in pairs for obj in pair],subdivide=action=='SUBDIVIDE'))
    inputs = [
        (source, detail, dl.reference_for_update(context, source, detail))
        for source, detail in pairs
    ]
    inputs = [
        (source, detail, ref) for source, detail, ref in inputs if ref is not None
    ]
    current_reference = False
    if action == "SUBDIVIDE" and not inputs:
        current_reference = True
        inputs = [(source, detail, dl.read_reference(detail))]
        if inputs[0][2] is None:
            raise ValueError("Recover the saved sculpt reference before subdivision")
    if not inputs:
        return False
    storage=resource_budget.detail_update_storage(inputs,subdivide=action=='SUBDIVIDE')
    resource_budget.require_disk('sculpt detail processing',Path(bpy.path.abspath(s.project_dir)),storage['required_bytes'])
    folder = (
        Path(bpy.path.abspath(s.project_dir)).resolve()
        / "DetailUpdates"
        / ".jobs"
        / uuid.uuid4().hex[:12]
    )
    folder.mkdir(parents=True)
    _staging['folder']=folder
    targets = []
    from . import detail_recovery as recovery

    for index, (source, detail, reference) in enumerate(inputs):
        arrays = {f"source_{key}": value for key, value in mesh_arrays(source).items()}
        arrays["source_vertices"] = m.to_local(
            source, m.evaluated_coordinates(context, source)
        )
        arrays.update(
            {f"detail_{key}": value for key, value in mesh_arrays(detail).items()}
        )
        legacy = isinstance(reference, recovery.LegacyReference)
        if not legacy:
            arrays["reference"] = reference
        keys = key_arrays(detail)
        for key_index, key in enumerate(keys):
            arrays[f"key_{key_index}"] = points(key.data)
        recovery_data = None
        dependencies = []
        if legacy:
            recovery_data = {
                "parents": reference.parents,
                "explicit": sorted(reference.explicit),
                "levels": [recovery.level(obj) for obj in reference.objects],
                "steps": [dl.steps_for(obj) for obj in reference.objects],
                "types": [dl.subdivision_type(obj) for obj in reference.objects],
                "names": [obj.name for obj in reference.objects],
            }
            for node_index, obj in enumerate(reference.objects[1:], 1):
                prefix = f"legacy_{node_index}"
                arrays.update(
                    {
                        f"{prefix}_{name}": value
                        for name, value in mesh_arrays(obj).items()
                    }
                )
                arrays[prefix + "_basis"] = recovery.basis_for(
                    reference, node_index
                ).astype(np.float32)
                arrays[prefix + "_visible"] = recovery.visible_for(
                    context, reference, node_index
                ).astype(np.float32)
                previous_reference = dl.read_reference(obj)
                if previous_reference is not None:
                    arrays[prefix + "_reference"] = previous_reference.astype(
                        np.float32
                    )
                for name in ("crease_edge", "crease_vert"):
                    attribute = obj.data.attributes.get(name)
                    if attribute:
                        values = np.empty(len(attribute.data), dtype=np.float32)
                        attribute.data.foreach_get("value", values)
                        arrays[prefix + "_" + name] = values
                dependencies.append((obj, fingerprint(context, source, obj)))
        for name in ("crease_edge", "crease_vert"):
            attribute = source.data.attributes.get(name)
            if attribute:
                values = np.empty(len(attribute.data), dtype=np.float32)
                attribute.data.foreach_get("value", values)
                arrays[name] = values
        np.savez(folder / f"input-{index}.npz", **arrays)
        subdivision = None
        if action == "SUBDIVIDE":
            from . import detail_subdivision as subd

            subdivision = {
                "mesh": subd.snapshot(detail, folder / "input-subdivide.blend"),
                "groups": [group.name for group in detail.vertex_groups],
                "active_key": detail.active_shape_key_index,
                "only_key": detail.show_only_shape_key,
            }
        targets.append(
            {
                "source": source,
                "detail": detail,
                "stamp": fingerprint(context, source, detail),
                "keys": keys,
                "level": int(detail.get("cm_detail_level", 1)),
                "index": index,
                "topology": m.topology(detail),
                "steps": dl.steps_for(detail),
                "subdivision_type": dl.subdivision_type(detail),
                "recovery": recovery_data,
                "dependencies": dependencies,
                "recovery_identity": reference.identity() if legacy else None,
                "subdivision": subdivision,
                "subdivision_stamp": subd.signature(detail) if subdivision else None,
                "current_reference": current_reference,
            }
        )
    request = {
        "targets": [
            {
                "index": t["index"],
                "level": t["level"],
                "keys": len(t["keys"]),
                "topology": t["topology"],
                "steps": t["steps"],
                "subdivision_type": t['subdivision_type'],
                "recovery": t["recovery"],
                "subdivision": t["subdivision"],
                "current_reference": t["current_reference"],
            }
            for t in targets
        ],
        "parent": os.getpid(),
        "addon_root": str(Path(__file__).resolve().parents[1]),
        "storage_budget": storage,
    }
    _write(folder / "request.json", request)
    with (folder / "worker.log").open("w", encoding="utf-8") as log:
        process = subprocess.Popen(
            [
                bpy.app.binary_path,
                "--background",
                "--factory-startup",
                *thread_arguments(),
                "--python-exit-code",
                "1",
                "--python",
                str(Path(__file__).with_name("detail_worker.py")),
                "--",
                str(folder / "request.json"),
            ],
            stdout=log,
            stderr=subprocess.STDOUT,
            creationflags=subprocess.BELOW_NORMAL_PRIORITY_CLASS
            if os.name == "nt"
            else 0,
        )
    _staging['process']=process
    _JOB = {
        "process": process,
        "scene": context.scene,
        "state": scene_state(context.scene),
        "targets": targets,
        "action": action,
        "part": part,
        "folder": folder,
        "cancel_requested": False,
    }
    f.detail_update_status = (
        "Creating the next sculpt level in background..."
        if action == "SUBDIVIDE"
        else "Recovering the saved sculpt surface..."
        if any(t["recovery"] for t in targets)
        else "Updating sculpt surfaces in background..."
    )
    s.status = (
        "Subdivision started; Blender remains usable. Cancel retains the previous layer."
        if action == "SUBDIVIDE"
        else "Detail update started; Blender remains usable. Cancel retains the sculpt."
    )
    bpy.app.timers.register(poll, first_interval=0.2, persistent=True)
    return True


def plans_from_output(job):
    plans = []
    checksums = {item["name"]: item["sha256"] for item in job["report"]["files"]}
    metadata = {item["name"]: item for item in job["report"]["files"]}
    for target in job["targets"]:
        output = job["folder"] / f"output-{target['index']}.npz"
        if hashlib.sha256(output.read_bytes()).hexdigest() != checksums.get(
            output.name
        ):
            raise ValueError("Detail output checksum changed; the sculpt was retained")
        with np.load(output, allow_pickle=False) as values:
            arrays = {name: values[name].copy() for name in values.files}
        size = (len(target["detail"].data.vertices), 3)
        steps = metadata[output.name].get("subdivision_steps", target["steps"])
        if (
            not isinstance(steps, list)
            or not steps
            or len(steps) > 3
            or any(type(value) is not int or not 1 <= value <= 3 for value in steps)
            or sum(steps) != target["level"]
        ):
            raise ValueError(
                "Detail worker returned invalid subdivision history; sculpt retained"
            )
        if not target["recovery"] and steps != target["steps"]:
            raise ValueError(
                "Detail subdivision history changed unexpectedly; sculpt retained"
            )
        required = ["target", "mesh", *[f"key_{i}" for i in range(len(target["keys"]))]]
        if any(
            name not in arrays
            or arrays[name].shape != size
            or not np.isfinite(arrays[name]).all()
            for name in required
        ):
            raise ValueError(
                "Detail worker returned invalid geometry; the sculpt was retained"
            )
        plans.append(
            {
                "source": target["source"],
                "detail": target["detail"],
                "target": arrays["target"],
                "mesh": arrays["mesh"],
                "updates": [
                    (key, arrays[f"key_{i}"]) for i, key in enumerate(target["keys"])
                ],
                "steps": steps,
            }
        )
    return plans


def complete(job):
    from . import bake_jobs, workflow

    scene = job["scene"]
    with bpy.context.temp_override(scene=scene, view_layer=scene.view_layers[0]):
        if bpy.context.mode != "OBJECT" or scene_state(scene) != job["state"]:
            raise ValueError(
                "The editor changed during the update; work retained. Retry the action when ready"
            )
        for target in job["targets"]:
            if (
                target["source"].name not in scene.objects
                or target["detail"].name not in scene.objects
            ):
                raise ValueError(
                    "The accepted part or sculpt was removed; work retained"
                )
            if target.get("subdivision"):
                from . import detail_subdivision as subd

                from . import simple_shading
                if simple_shading.blocking_modifiers(target["detail"]):
                    raise ValueError(
                        "Sculpt modifiers changed during subdivision; previous sculpt retained"
                    )
                if subd.signature(target["detail"]) != target["subdivision_stamp"]:
                    raise ValueError(
                        "Sculpt attributes changed during subdivision; previous sculpt retained"
                    )
            if (
                fingerprint(bpy.context, target["source"], target["detail"])
                != target["stamp"]
            ):
                raise ValueError(
                    "The character or sculpt changed during the update; work retained. Retry the action"
                )
            for obj, stamp in target["dependencies"]:
                if (
                    obj.name not in scene.objects
                    or fingerprint(bpy.context, target["source"], obj) != stamp
                ):
                    raise ValueError(
                        "A previous sculpt layer changed during recovery; work retained. Retry when ready"
                    )
            if target["recovery"]:
                from .detail_recovery import describe

                if (
                    describe(bpy.context, target["detail"]).identity()
                    != target["recovery_identity"]
                ):
                    raise ValueError(
                        "The available previous layers changed during recovery; work retained"
                    )
        plans = plans_from_output(job)
        s, f = scene.cm_settings, scene.cm_features
        if job["action"] == "SUBDIVIDE":
            subd.publish(bpy.context, job)
        elif job["action"] == "BODY_APPLY":
            workflow.body_finish(bpy.context, s, f, True, detail_plans=plans)
            s.status = "Body applied; sculpt surfaces followed the accepted character."
        else:
            for plan in plans:
                dl.commit(bpy.context, plan)
            s.status = (
                "Saved sculpt reference recovered; artist detail retained."
                if job["action"] == "RECOVER"
                else "Sculpt surfaces now follow the accepted character."
            )
        if not bpy.app.background and bpy.context.preferences.edit.use_global_undo:
            bpy.ops.ed.undo_push(
                message="Apply body and detail"
                if job["action"] == "BODY_APPLY"
                else "Subdivide sculpt detail"
                if job["action"] == "SUBDIVIDE"
                else "Update sculpt detail"
            )
        if job["action"] == "SCULPT":
            if bpy.ops.cm.sculpt(detail=True, part=job["part"]) != {"FINISHED"}:
                raise ValueError(
                    "Detail updated; click Sculpt Detail to enter sculpting"
                )
        elif job["action"] == "FINAL_MAPS":
            bake_jobs.start(bpy.context, s, f, "FINAL_MAPS")
        elif job['action']=='SYNC_HEAD' and any(obj and obj.get('cm_wrinkle_update_pending') for obj in (f.wm1,f.wm2,f.wm3)):
            from . import wrinkle_jobs
            try:wrinkle_jobs.start(bpy.context,s,f)
            except Exception as exc:s.status='Head sculpt updated; wrinkle sculpts retained. Retry their update: '+str(exc)


def cleanup(job):
    for name in ("input-subdivide.blend", "output-subdivide.blend"):
        try:
            (job["folder"] / name).unlink(missing_ok=True)
        except OSError as exc:
            print("Subdivision cleanup:", exc)
    for prefix in ("input-", "output-"):
        for path in job["folder"].glob(prefix + "*.npz"):
            try:
                path.unlink(missing_ok=True)
            except OSError as exc:
                print("Detail job cleanup:", exc)


def record_exit(job):
    code = job["process"].poll()
    if code is None:
        return
    try:
        _write(
            job["folder"] / "process-exit.json",
            {
                "pid": job["process"].pid,
                "exit_code": code,
                "cancel_requested": job["cancel_requested"],
            },
        )
    except OSError as exc:
        print("Detail process exit record:", exc)


def poll():
    global _JOB
    job = _JOB
    if not job:
        return None
    code = job["process"].poll()
    if code is None:
        try:
            path = job["folder"] / "report.json"
            if path.is_file():
                stage = json.loads(path.read_text()).get("stage")
                if stage:
                    job["scene"].cm_features.detail_update_status = str(stage)
        except (OSError, ValueError, ReferenceError):
            pass
        return 0.2
    _JOB = None
    record_exit(job)
    try:
        if job["cancel_requested"]:
            job[
                "scene"
            ].cm_settings.status = "Detail update canceled; the sculpt was retained."
        else:
            path = job["folder"] / "report.json"
            report = json.loads(path.read_text()) if path.is_file() else {}
            if code or report.get("status") != "complete":
                raise RuntimeError(
                    report.get(
                        "error",
                        "Detail worker stopped (exit code "
                        + str(code)
                        + "); see "
                        + str(job["folder"] / "worker.log"),
                    )
                )
            job["report"] = report
            complete(job)
    except Exception as exc:
        try:
            job["scene"].cm_settings.status = "Detail update did not apply: " + str(exc)
        except ReferenceError:
            pass
        print("MetaHuman Maker detail update:", exc)
    finally:
        try:
            job["scene"].cm_features.detail_update_status = ""
        except ReferenceError:
            pass
        cleanup(job)
        for window in bpy.context.window_manager.windows:
            for area in window.screen.areas:
                area.tag_redraw()
    return None


def cancel():
    if _JOB:
        _JOB["cancel_requested"] = True
        if _JOB["process"].poll() is None:
            _JOB["process"].terminate()


def reap():
    for job in list(_REAPING):
        if job["process"].poll() is not None:
            record_exit(job)
            cleanup(job)
            _REAPING.remove(job)
    return 0.2 if _REAPING else None


@persistent
def cancel_active(_=None):
    global _JOB
    if _JOB:
        cancel()
        job, _JOB = _JOB, None
        if bpy.app.timers.is_registered(poll):
            bpy.app.timers.unregister(poll)
        try:
            job["scene"].cm_features.detail_update_status = ""
        except ReferenceError:
            pass
        job.pop("scene", None)
        job.pop("targets", None)
        _REAPING.append(job)
        if not bpy.app.timers.is_registered(reap):
            bpy.app.timers.register(reap, first_interval=0.2, persistent=True)


class CM_OT_cancel_detail_update(bpy.types.Operator):
    bl_idname = "cm.cancel_detail_update"
    bl_label = "Cancel Detail Update"

    def execute(self, context):
        cancel()
        return {"FINISHED"}


CLASSES = (CM_OT_cancel_detail_update,)


def register():
    for handlers in (
        bpy.app.handlers.load_pre,
        bpy.app.handlers.undo_pre,
        bpy.app.handlers.redo_pre,
    ):
        handlers.append(cancel_active)


def unregister():
    cancel_active()
    for handlers in (
        bpy.app.handlers.load_pre,
        bpy.app.handlers.undo_pre,
        bpy.app.handlers.redo_pre,
    ):
        if cancel_active in handlers:
            handlers.remove(cancel_active)
