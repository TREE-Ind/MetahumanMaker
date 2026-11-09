"""Discover local resources without requiring per-scene path setup."""

from pathlib import Path
import json
import os
import uuid
import bpy
from bpy.app.handlers import persistent


def settings_file():
    return Path(bpy.utils.user_resource("CONFIG")) / "create_metahuman.json"


def candidates():
    try:
        saved = json.loads(settings_file().read_text(encoding="utf-8"))
    except (OSError, ValueError):
        saved = {}
    yield saved
    roots = [Path(__file__).resolve().parents[1], Path.home() / "MetaHumanMaker", Path.home() / "CreateMetaHuman"]
    if os.environ.get("CREATE_METAHUMAN_HOME"):
        roots.insert(0, Path(os.environ["CREATE_METAHUMAN_HOME"]))
    if bpy.data.filepath:
        roots.extend(Path(bpy.data.filepath).resolve().parents)
    for root in roots:
        yield {
            "model_dir": str(root / "models"),
            "inference_python": str(
                root
                / ".venv"
                / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
            ),
            "skin_solver_sr_python": str(root / '.venv-materials' / ('Scripts/python.exe' if os.name == 'nt' else 'bin/python')),
        }


def valid(name, value):
    path = Path(bpy.path.abspath(value))
    if name == 'reference_library':return path.suffix.lower()=='.blend' and path.is_file()
    return (path / "gnm_head.npz").is_file() if name == "model_dir" else path.is_file()


def resolve(f):
    """Keep valid scene choices; repair missing paths using saved setup and local discovery."""
    for paths in candidates():
        for name in ("model_dir", "inference_python", "skin_solver_sr_python", "reference_library"):
            if not valid(name, getattr(f, name)) and valid(name, paths.get(name, "")):
                setattr(f, name, paths[name])
    return valid("model_dir", f.model_dir), valid(
        "inference_python", f.inference_python
    )


def save(f):
    resolve(f)
    path = settings_file()
    path.parent.mkdir(parents=True, exist_ok=True)
    pending=path.with_name('.'+path.name+'.'+uuid.uuid4().hex+'.tmp')
    try:
        pending.write_text(
        json.dumps(
            {name: getattr(f, name) for name in ("model_dir", "inference_python", "skin_solver_sr_python", "reference_library")},
            indent=2,
        ),
        encoding="utf-8",
        )
        pending.replace(path)
    finally:
        pending.unlink(missing_ok=True)


@persistent
def repair_scenes(_):
    for scene in bpy.data.scenes:
        if hasattr(scene, "cm_features"):
            resolve(scene.cm_features)


def register():
    bpy.app.handlers.load_post.append(repair_scenes)
    # Blender restricts scene access while enabling an add-on from Preferences.
    if hasattr(bpy.data, "scenes"):
        repair_scenes(None)
    elif not bpy.app.timers.is_registered(initialize):
        bpy.app.timers.register(initialize, first_interval=0.0)


def initialize():
    repair_scenes(None)
    return None


def unregister():
    if bpy.app.timers.is_registered(initialize):
        bpy.app.timers.unregister(initialize)
    if repair_scenes in bpy.app.handlers.load_post:
        bpy.app.handlers.load_post.remove(repair_scenes)
