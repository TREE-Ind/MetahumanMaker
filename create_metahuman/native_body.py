"""Asynchronous anatomical body shaping with the user's installed Unreal model."""

from pathlib import Path
import base64
import json
import os
import shutil
import subprocess
import time
import uuid
import zlib
import hashlib
import bpy
import numpy as np
from bpy.app.handlers import persistent
from . import expressions, geometry, body_ipc, mesh_ops as m

_JOB = None
_KEY = "cm_native_body_state"
# Native measurements handle proportions and girths. The remaining controls keep
# their independent regional sculpt behavior; do not present those as native solves.
MEASUREMENTS = {
    "shoulders": ("Across Shoulder", 0.17),
    "chest": ("Chest", 0.22),
    "neck": ("Neck", 0.20),
    "upper_arm": ("Bicep", 0.18),
    "forearm": ("Forearm", 0.18),
    "waist": ("Waist", 0.20),
    "hips": ("Hip", 0.18),
    "thighs": ("Thigh", 0.16),
    "calves": ("Calf", 0.18),
    "hands": ("Hand Circumference", 0.18),
    "knees": ("Knee", 0.14),
}
NATIVE_KEYS = set(MEASUREMENTS) | {
    "height",
    "mass",
    "gender",
    "definition",
    "arm_length",
    "leg_length",
    "armpit",
}


def available(f):
    bridge = expressions.bridge_path(f)
    try:
        version = json.loads((bridge / "CreateMetaHumanBridge.uplugin").read_text())[
            "VersionName"
        ]
        return (
            tuple(int(n) for n in version.split(".")) >= (0, 4, 0)
            and (
                bridge / "Binaries/Win64/UnrealEditor-CreateMetaHumanBridge.dll"
            ).is_file()
        )
    except (OSError, ValueError, KeyError):
        return False


def pack_state(obj, path):
    raw = Path(path).read_bytes()
    text = bpy.data.texts.new("CM_NativeBodyState_" + uuid.uuid4().hex[:8])
    text.use_fake_user = False  # The body's ID pointer owns the packed state.
    # Blender recalculates a text line while inserting each character. A single
    # megabyte line can stall Apply for minutes; MIME wrapping keeps this linear.
    text.from_string(base64.encodebytes(zlib.compress(raw)).decode("ascii"))
    old = obj.get(_KEY)
    obj[_KEY] = text
    if isinstance(old, bpy.types.Text) and old.users <= int(old.use_fake_user):
        bpy.data.texts.remove(old)


def unpack_state(obj, destination):
    text = obj.get(_KEY)
    if not isinstance(text, bpy.types.Text):
        return None
    compressed = base64.b64decode("".join(text.as_string().split()), validate=True)
    decoder = zlib.decompressobj()
    raw = decoder.decompress(compressed, 32 * 1024 * 1024)
    if not decoder.eof:
        raise ValueError("Packed native body state is invalid or too large")
    destination.write_bytes(raw)
    return destination


def save_custom_state(obj, filepath):
    path = Path(filepath)
    text = obj.get(_KEY)
    path.with_suffix(".cm-body.json").write_text(
        json.dumps(
            {
                "schema": 1,
                "mesh_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "state": text.as_string() if isinstance(text, bpy.types.Text) else None,
            }
        )
    )


def load_custom_state(obj, filepath):
    path = Path(filepath)
    sidecar = path.with_suffix(".cm-body.json")
    old = obj.get(_KEY)
    if _KEY in obj:
        del obj[_KEY]
    if isinstance(old, bpy.types.Text) and old.users <= int(old.use_fake_user):
        bpy.data.texts.remove(old)
    if sidecar.is_file():
        try:
            data = json.loads(sidecar.read_text())
        except (OSError, ValueError):
            return
        if (
            data.get("schema") == 1
            and data.get("mesh_sha256") == hashlib.sha256(path.read_bytes()).hexdigest()
        ):
            if not isinstance(data.get("state"), str):
                return
            try:
                encoded = "".join(data["state"].split())
                compressed = base64.b64decode(encoded, validate=True)
                decoder = zlib.decompressobj()
                decoder.decompress(compressed, 32 * 1024 * 1024)
                if not decoder.eof:
                    return
            except (ValueError, zlib.error):
                return
            text = bpy.data.texts.new("CM_NativeBodyState_" + uuid.uuid4().hex[:8])
            text.use_fake_user = False
            text.from_string("\n".join(encoded[i:i+76] for i in range(0,len(encoded),76)))
            obj[_KEY] = text


def constraints(metadata, values):
    source = {item["name"]: item for item in metadata["constraints"]}
    gender=source.get('Masculine/Feminine')
    gender_target=None
    if gender is not None:
        gender_target=(gender['min']+gender['max'])*.5+float(values.get('gender',0))*(gender['max']-gender['min'])*.5
    gender_changed=gender_target is not None and abs(gender_target-gender['actual'])>1e-5
    if not any(values.get(key,0) for key in NATIVE_KEYS-{'gender'}) and not gender_changed:return {}
    # Earlier applied native constraints remain intentional. Hold their achieved
    # values when shaping another region, rather than regressing toward the mean.
    targets = {name:item['actual'] for name,item in source.items() if item['active']}

    def relative(name, value, fraction):
        if value:
            targets[name] = source[name]["actual"] * (1 + fraction * value)

    for key, (name, fraction) in MEASUREMENTS.items():
        relative(name, values.get(key, 0), fraction)
    for key, name, sign in [
        ("mass", "Fat", 1),
        ("definition", "Muscularity", 1),
    ]:
        if values.get(key, 0):
            targets[name] = (
                source[name]["actual"]
                + values[key]
                * max(abs(source[name]["min"]), abs(source[name]["max"]))
                * sign
            )
    # Unlike girth offsets, this is a direct position on the native trait axis.
    # Positive is feminine, negative masculine; loading a pass seeds the saved
    # native position below so opening the controls does not change the body.
    if gender_target is not None:targets['Masculine/Feminine']=gender_target
    arm = values.get("arm_length", 0)
    for name in ["Upper Arm Length", "Lower Arm Length"]:
        relative(name, arm, 0.12)
    leg = source["Inseam"]["actual"] * 0.14 * values.get("leg_length", 0)
    if leg:
        targets["Inseam"] = source["Inseam"]["actual"] + leg
    armpit = values.get("armpit", 0)
    if armpit:
        targets["Shoulder Height"] = (
            source["Shoulder Height"]["actual"]
            + source["Height"]["actual"] * 0.02 * armpit
        )
    if targets or values.get("height", 0):
        targets["Height"] = (
            source["Height"]["actual"] * (1 + 0.15 * values.get("height", 0)) + leg
        )
    return {
        name: float(np.clip(value, source[name]["min"], source[name]["max"]))
        for name, value in targets.items()
    }


def _write_request(job):
    data = {
        "version": job["version"],
        "constraints": constraints(job["metadata"], job["values"]),
    }
    if body_ipc.write(job['folder']/'request.json',data):
        job["sent"] = job["version"]


def can_apply():
    return bool(_JOB and _JOB.get("metadata") and _JOB["accepted"] == _JOB["version"])


def _start(context, s, f):
    global _JOB
    if _JOB:
        raise ValueError("Finish the body pass in the other scene first")
    if not available(f):
        raise ValueError("Anatomical shaping requires the compiled Bridge 0.4 folder")
    engine = Path(bpy.path.abspath(s.unreal_engine_dir))
    editor = engine / "Engine/Binaries/Win64/UnrealEditor-Cmd.exe"
    model = (
        engine
        / "Engine/Plugins/MetaHuman/MetaHumanCharacter/Content/Body/IdentityTemplate"
    )
    if not editor.is_file() or not (model / "body_model.dna").is_file():
        raise ValueError(
            "Select Unreal with the optional MetaHuman body data installed"
        )
    from . import resource_budget
    resource_budget.require('the anatomical body worker',2*resource_budget.GIB)
    folder = (
        Path(bpy.path.abspath(s.project_dir))
        / "NativeBody"
        / ".jobs"
        / uuid.uuid4().hex[:12]
    )
    # A missing or invalid packed state must fail before changing the accepted pass.
    project = folder / "editor"
    shutil.copytree(
        expressions.bridge_path(f),
        project / "Plugins/CreateMetaHumanBridge",
        ignore=shutil.ignore_patterns("Intermediate", "*.pdb"),
    )
    uproject = project / "CMBodyWorker.uproject"
    uproject.write_text(
        json.dumps(
            {
                "FileVersion": 3,
                "Plugins": [
                    {"Name": n, "Enabled": True}
                    for n in ["PythonScriptPlugin", "CreateMetaHumanBridge"]
                ],
            }
        )
    )
    obj = f.body_result or s.result
    state = unpack_state(obj, folder / "initial.state")
    config = folder / "config.json"
    config.write_text(
        json.dumps(
            {
                "folder": str(folder),
                "model_path": str(model),
                "parent": os.getpid(),
                "state_file": str(state) if state else None,
            }
        )
    )
    runner = project / "run.py"
    runner.write_text(
        "import runpy\nrunpy.run_path("
        + repr(str(Path(__file__).with_name("native_body_server.py")))
        + ')["serve"]('
        + repr(str(config))
        + ")\n"
    )
    # The accepted pass is captured only after setup has passed all validations.
    from . import workflow

    for item in f.body_controls:
        item.value = 0
    workflow.regional_body_preview(context, s, f)
    local = np.asarray(
        [v.co[:] for v in obj.data.shape_keys.key_blocks["Body Pass Base"].data]
    )
    matrix = np.asarray(obj.matrix_world)
    baseline = local @ matrix[:3, :3].T + matrix[:3, 3]
    try:
        with (folder / "worker.log").open("w", encoding="utf-8") as log:
            process = subprocess.Popen(
                [
                    str(editor),
                    str(uproject),
                    "-run=pythonscript",
                    "-script=" + str(runner),
                    "-NullRHI",
                    "-unattended",
                    "-nosound",
                    "-nop4",
                    "-nosplash",
                    "-DDC=InstalledNoZenLocalFallback",
                ],
                stdout=log,
                stderr=subprocess.STDOUT,
                creationflags=(
                    subprocess.CREATE_NO_WINDOW | subprocess.BELOW_NORMAL_PRIORITY_CLASS
                )
                if os.name == "nt"
                else 0,
            )
    except Exception:
        workflow.regional_body_finish(context, s, f, False)
        raise
    _JOB = {
        "scene": context.scene,
        "object": obj,
        "baseline": baseline,
        "folder": folder,
        "process": process,
        "version": 0,
        "sent": -1,
        "accepted": -1,
        "values": {},
        "changed": time.monotonic(),
        "metadata": None,
    }
    f.body_native_status = "Starting anatomical model..."
    bpy.app.timers.register(_poll, first_interval=0.15, persistent=True)


def preview(context, s, f):
    if not f.body_editing:
        _start(context, s, f)
    if not _JOB or _JOB["scene"] != context.scene:
        raise ValueError(
            "The anatomical body session is unavailable; cancel and reopen it"
        )
    _JOB["values"] = {item.key: item.value for item in f.body_controls}
    _JOB["version"] += 1
    _JOB["changed"] = time.monotonic()
    f.body_native_status = "Updating anatomical preview..."


def _apply_output(job, metadata):
    from . import workflow

    scene = job["scene"]
    s, f = scene.cm_settings, scene.cm_features
    folder = job["folder"]
    points = expressions.binary(
        folder, metadata["vertices_file"], "<f4", (metadata["body_vertices"], 3)
    ) * [0.01, -0.01, 0.01]
    delta = (
        (points - job["native_base"])[job["mapping"]] @ job["rotation"] * job["scale"]
    )
    output = job["baseline"] + delta
    regional = {
        key: value for key, value in job["values"].items() if key not in NATIVE_KEYS
    }
    follow = {}
    if any(regional.values()):
        from . import body_regions
        output = body_regions.deform(job['object'], f.body_template or s.template, output, regional, follow=follow)
    obj = job["object"]
    key = obj.data.shape_keys.key_blocks["Body Pass"]
    key.data.foreach_set("co", m.to_local(obj, output).ravel())
    obj.data.update()
    bpy.context.view_layer.update()
    if s.part == "FULL_BODY":
        workflow._body_follow(bpy.context, s, f, job["baseline"], output, follow=follow)
    job["last_output"] = metadata


def _poll():
    global _JOB
    job = _JOB
    if not job:
        return None
    try:
        if job["process"].poll() is not None:
            raise RuntimeError(
                "Anatomical worker stopped. See " + str(job["folder"] / "worker.log")
            )
        scene = job["scene"]
        with bpy.context.temp_override(scene=scene, view_layer=scene.view_layers[0]):
            s, f = scene.cm_settings, scene.cm_features
            if not f.body_editing:
                if job["process"].poll() is None:
                    job["process"].terminate()
                _JOB = None
                return None
            values = {item.key: item.value for item in f.body_controls}
            if values != job["values"]:
                job.update(
                    values=values, version=job["version"] + 1, changed=time.monotonic()
                )
                f.body_native_status = "Updating anatomical preview..."
            ready = job["folder"] / "ready.json"
            if job["metadata"] is None and ready.is_file():
                metadata = json.loads((job["folder"] / "baseline.json").read_text())
                base = expressions.binary(
                    job["folder"],
                    metadata["vertices_file"],
                    "<f4",
                    (metadata["body_vertices"], 3),
                ) * [0.01, -0.01, 0.01]
                reference = f.body_template or s.template
                try:
                    mapping = expressions.uv_mapping(reference, metadata)
                except ValueError as exc:
                    raise ValueError(str(exc).replace("Expression", "Body")) from exc
                rotation, scale, translation = geometry.similarity(
                    base[mapping], job["baseline"]
                )
                job.update(
                    metadata=metadata,
                    native_base=base,
                    mapping=mapping,
                    rotation=rotation,
                    scale=scale,
                )
                gender=next((row for row in metadata['constraints'] if row['name']=='Masculine/Feminine'),None)
                control=next((row for row in f.body_controls if row.key=='gender'),None)
                if gender and control is not None and control.value==0:
                    span=gender['max']-gender['min']
                    if span>0:
                        control['value']=float(np.clip((2*gender['actual']-gender['min']-gender['max'])/span,-1,1))
                        job['values']['gender']=control.value
            if (
                job["metadata"]
                and job["sent"] != job["version"]
                and time.monotonic() - job["changed"] > 0.15
            ):
                _write_request(job)
            response = job["folder"] / "response.json"
            if response.is_file():
                message = body_ipc.read(response)
                if message is None:return 0.1
                # A completed older solve must never overwrite newer slider intent.
                if (
                    message["version"] == job["version"]
                    and job["accepted"] != job["version"]
                ):
                    if message.get("error"):
                        raise ValueError(message["error"])
                    metadata = json.loads(
                        (job["folder"] / message["output"]).read_text()
                    )
                    _apply_output(job, metadata)
                    job["accepted"] = job["version"]
                    f.body_native_status = "Anatomical preview ready"
            for window in bpy.context.window_manager.windows:
                for area in window.screen.areas:
                    area.tag_redraw()
    except Exception as exc:
        try:
            scene = job["scene"]
            with bpy.context.temp_override(
                scene=scene, view_layer=scene.view_layers[0]
            ):
                finish(bpy.context, scene.cm_settings, scene.cm_features, False)
                scene.cm_settings.status = str(exc)
        except Exception as rollback:
            print("Native body rollback:", rollback)
        return None
    return 0.1


def finish(context, s, f, apply=True):
    global _JOB
    job = _JOB
    if not job:
        raise ValueError("No anatomical body session is active")
    if apply and not can_apply():
        raise ValueError("Wait for the current anatomical preview before applying")
    if apply:
        pack_state(job["object"], job["folder"] / job["last_output"]["state_file"])
    _JOB = None
    if job["process"].poll() is None:
        job["process"].terminate()
    if bpy.app.timers.is_registered(_poll):
        bpy.app.timers.unregister(_poll)
    from . import workflow

    workflow.regional_body_finish(context, s, f, apply)
    if not apply:
        from . import sculpt_archive
        sculpt_archive.finish_body_recovery(context,False)
    f.body_native_status = ""
    s.status = (
        "Anatomical body applied."
        if apply
        else "Body pass canceled; accepted head and body restored."
    )


@persistent
def cancel_active(_=None):
    if _JOB:
        scene = _JOB["scene"]
        with bpy.context.temp_override(scene=scene, view_layer=scene.view_layers[0]):
            finish(bpy.context, scene.cm_settings, scene.cm_features, False)


@persistent
def recover_interrupted(_=None):
    """A saved preview has no matching worker after loading; restore its baseline."""
    from . import workflow

    for scene in bpy.data.scenes:
        f = scene.cm_features
        if f.body_backend != "NATIVE" or not f.body_editing:
            continue
        if _JOB and _JOB["scene"] == scene:
            continue
        with bpy.context.temp_override(scene=scene, view_layer=scene.view_layers[0]):
            workflow.regional_body_finish(bpy.context, scene.cm_settings, f, False)
            from . import sculpt_archive
            sculpt_archive.finish_body_recovery(bpy.context,False)
            f.body_native_status = ""
            scene.cm_settings.status = "Interrupted body preview recovered; accepted character restored."


def register():
    for handlers in (bpy.app.handlers.load_pre,bpy.app.handlers.undo_pre,bpy.app.handlers.redo_pre):
        handlers.append(cancel_active)
    for handlers in (bpy.app.handlers.load_post,bpy.app.handlers.undo_post,bpy.app.handlers.redo_post):
        handlers.append(recover_interrupted)


def unregister():
    cancel_active()
    for handlers in (bpy.app.handlers.load_pre,bpy.app.handlers.undo_pre,bpy.app.handlers.redo_pre):
        if cancel_active in handlers:
            handlers.remove(cancel_active)
    for handlers in (bpy.app.handlers.load_post,bpy.app.handlers.undo_post,bpy.app.handlers.redo_post):
        if recover_interrupted in handlers:
            handlers.remove(recover_interrupted)
