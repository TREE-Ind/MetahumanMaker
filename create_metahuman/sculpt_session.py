"""Separate detail authoring with acknowledged, maps-only return."""

from pathlib import Path
import hashlib, json, os, shutil, subprocess, time, uuid
import math
import bpy, numpy as np
from bpy.app.handlers import persistent
from . import process_ref, mesh_ops as m

_SESSION = None
_PENDING = None
_PENDING_STATE = None
LOOK_FIELDS = (
    "lookdev_roughness",
    "lookdev_subsurface",
    "lookdev_normal_strength",
    "lookdev_environment",
    "lookdev_rotation",
    "lookdev_intensity",
    "lookdev_transparent",
    "lookdev_size",
    "lookdev_renderer",
    "lookdev_samples",
    "lookdev_skin_controls",
    "lookdev_environment_controls",
    "lookdev_render_controls",
)


def write_json(path, data):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(data, indent=2))
    temporary.replace(path)


def read_json(path):
    return json.loads(path.read_text())


def folder_for(scene):
    return Path(bpy.path.abspath(scene.cm_features.sculpt_session_path)).resolve()


def pending():
    return _PENDING is not None


def return_state(scene):
    s, f = scene.cm_settings, scene.cm_features
    return {
        "final": stamp(scene),
        "wrinkles": {
            slot: stamp(scene, slot)
            for slot in ("WM1", "WM2", "WM3")
            if getattr(f, slot.lower())
        },
        "lookdev": {name: getattr(f, name) for name in LOOK_FIELDS},
        "project": s.project_dir,
        "name": s.character_name,
        "wrinkle_size": f.wrinkle_map_size,
        "bloodflow": f.bloodflow,
        "tint": f.bloodflow_strength,
    }


def stamp(scene, slot=None):
    s, f = scene.cm_settings, scene.cm_features
    objects = (
        [getattr(f, slot.lower())]
        if slot
        else [s.result, s.detail, f.body_result, f.body_detail]
    )
    digest = hashlib.sha256()
    with bpy.context.temp_override(scene=scene, view_layer=scene.view_layers[0]):
        for obj in objects:
            if obj:
                digest.update(m.topology(obj).encode())
                digest.update(
                    np.asarray(
                        m.evaluated_coordinates(bpy.context, obj), dtype="<f4"
                    ).tobytes()
                )
        config = (
            {}
            if slot
            else {
                "size": s.map_size,
                "distance": s.bake_distance,
                "pores": f.add_pores,
                "pore_strength": f.pore_strength,
                "pore_spacing": f.pore_spacing,
            }
        )
        digest.update(json.dumps(config, sort_keys=True).encode())
        if slot:
            mask = getattr(f, slot.lower() + "_mask")
            if mask:
                digest.update(np.asarray(mask.pixels[:], dtype="<f4").tobytes())
    return digest.hexdigest()


def bake_started(scene, mode, slots):
    if scene.cm_features.sculpt_session_role != "AUTHOR":
        return None
    return (
        {slot: stamp(scene, slot) for slot in slots}
        if mode == "WRINKLES"
        else {"FINAL": stamp(scene)}
    )


def bake_completed(job):
    scene = job["scene"]
    basis = job.get("sculpt_basis")
    if not basis or scene.cm_features.sculpt_session_role != "AUTHOR":
        return
    path = folder_for(scene) / "receipts.json"
    receipt = read_json(path) if path.is_file() else {}
    for slot, value in basis.items():
        receipt[slot] = {
            "stamp": value,
            "job": str(job["folder"]),
            "time": time.time(),
            "files": {
                p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                for p in (
                    Path(bpy.path.abspath(scene.cm_settings.project_dir)) / "Maps"
                ).glob("*.png")
                if (
                    slot == "FINAL"
                    and (p.stem.endswith("_NORMAL") or p.stem.endswith("_CAVITY"))
                )
                or (
                    slot != "FINAL"
                    and (
                        p.stem.endswith("_" + slot) or p.stem.endswith("_CM" + slot[-1])
                    )
                )
            },
        }
    write_json(path, receipt)


def _manifest(folder):
    return read_json(folder / "session.json")


def _attach(scene, folder, process=None):
    global _SESSION
    manifest = _manifest(folder)
    ready = folder / "author-ready.json"
    reference = (
        process_ref.identify(process.pid)
        if process
        else read_json(ready)["process"]
        if ready.is_file()
        else None
    )
    _SESSION = {
        "scene": scene,
        "folder": folder,
        "manifest": manifest,
        "process": reference,
        "popen": process,
        "seen": None,
        "ready": False,
    }
    write_json(
        folder / "connection.json", {"connected": True, "parent": manifest["parent"]}
    )
    response = folder / "response.json"
    if response.is_file():
        _SESSION["seen"] = read_json(response).get("request_id")
    if not bpy.app.timers.is_registered(poll_main):
        bpy.app.timers.register(poll_main, first_interval=0.25, persistent=True)


def write_snapshot(scene, filepath):
    """Pack current image buffers through temporary copies, then restore main IDs."""
    materials = {}
    images = {}
    slots = {}
    references = []
    mask_refs = {}
    feature_refs = {}
    groups = {}
    textures = {}
    modifier_textures = []
    from .sculpt_dependencies import Capture
    capture = Capture(bpy.context)
    world = scene.world
    world_copy = None

    def image_copy(image):
        if image not in images:
            copy = capture.copy(image)
            images[image] = copy
            for name in image.keys():
                reference=image.get(name)
                if isinstance(reference,bpy.types.Image):copy[name]=image_copy(reference)
        return images[image]

    def texture_copy(texture):
        if texture not in textures:
            copy=texture.copy()
            textures[texture]=copy
            if texture.type=='IMAGE' and texture.image:copy.image=image_copy(texture.image)
            if copy.use_nodes and copy.node_tree:copy_tree_images(copy.node_tree)
        return textures[texture]

    def copy_tree_images(tree):
        for node in tree.nodes:
            if node.type in {'TEX_IMAGE','TEX_ENVIRONMENT'} and node.image:
                node.image=image_copy(node.image)
            elif node.type=='GROUP' and node.node_tree:
                original=node.node_tree
                if original not in groups:
                    groups[original]=original.copy()
                    copy_tree_images(groups[original])
                node.node_tree=groups[original]

    try:
        for name in ('generated_local_skin_image','generated_skin_image'):
            image=getattr(scene.cm_features,name)
            if image:
                feature_refs[name]=image
                # RNA update callbacks would reapply the source skin and alter
                # the artist's shader. Snapshot remapping is not a UI edit.
                scene.cm_features[name]=image_copy(image)
                if getattr(scene.cm_features,name)!=images[image]:
                    raise ValueError('Cannot isolate the selected skin image')
        if world:
            world_copy = world.copy()
            scene.world = world_copy
            if world_copy.use_nodes:
                copy_tree_images(world_copy.node_tree)
        for name in ("wm1_mask", "wm2_mask", "wm3_mask"):
            image = getattr(scene.cm_features, name)
            if image:
                mask_refs[name] = image
                setattr(scene.cm_features, name, image_copy(image))
        for obj in scene.objects:
            for modifier in obj.modifiers:
                for prop in modifier.bl_rna.properties:
                    if prop.identifier=='rna_type' or prop.is_readonly or prop.type!='POINTER':continue
                    value=getattr(modifier,prop.identifier)
                    if isinstance(value,bpy.types.Texture):
                        modifier_textures.append((modifier,prop.identifier,value))
                        setattr(modifier,prop.identifier,texture_copy(value))
            if hasattr(obj.data, "materials") and obj.data not in slots:
                slots[obj.data] = list(obj.data.materials)
                for index, material in enumerate(slots[obj.data]):
                    if not material:
                        continue
                    if material not in materials:
                        copy = material.copy()
                        materials[material] = copy
                        if copy.use_nodes:
                            copy_tree_images(copy.node_tree)
                    obj.data.materials[index] = materials[material]
            image = obj.get("cm_reference_normal_image")
            if isinstance(image, bpy.types.Image):
                references.append((obj, image))
                obj["cm_reference_normal_image"] = image_copy(image)
        bpy.data.libraries.write(
            str(filepath), {scene}, path_remap="ABSOLUTE", compress=True
        )
    finally:
        for modifier,name,texture in modifier_textures:setattr(modifier,name,texture)
        for data, original in slots.items():
            for index, material in enumerate(original):
                data.materials[index] = material
        for obj, image in references:
            obj["cm_reference_normal_image"] = image
        for name, image in mask_refs.items():
            setattr(scene.cm_features, name, image)
        for name,image in feature_refs.items():
            scene.cm_features[name]=image
        scene.world = world
        if world_copy:
            bpy.data.worlds.remove(world_copy)
        for material in materials.values():
            bpy.data.materials.remove(material)
        for group in groups.values():
            bpy.data.node_groups.remove(group,do_unlink=True)
        for texture in textures.values():bpy.data.textures.remove(texture)
        bpy.data.batch_remove(ids=capture.ids)


def isolate_assets(scene, folder):
    """Give the author copy writable textures without sharing main asset paths."""
    manifest = _manifest(folder)
    project = Path(manifest["project"])
    author = Path(manifest["author_project"]).parent
    assets = author / "Assets"
    assets.mkdir(exist_ok=True)
    paths = {}
    for image in bpy.data.images:
        if image.source not in {"FILE", "GENERATED", "TILED"}:
            continue
        original = Path(bpy.path.abspath(image.filepath)) if image.filepath else None
        if (
            original
            and original.is_relative_to(project)
            and not original.is_relative_to(folder)
        ):
            destination = author / original.relative_to(project)
        elif original and original.is_relative_to(author):
            destination = original
        else:
            key = hashlib.sha256((str(original) + image.name).encode()).hexdigest()[:12]
            destination = assets / (
                key
                + "_"
                + (
                    original.name
                    if original and original.suffix
                    else image.name + ".png"
                )
            )
        destination.parent.mkdir(parents=True, exist_ok=True)
        if image.source == "TILED":
            for packed in image.packed_files:
                tile = destination.parent / Path(packed.filepath).name
                tile.write_bytes(packed.packed_file.data)
        elif image.packed_file:
            destination.write_bytes(image.packed_file.data)
        elif original and original.is_file():
            if original.resolve() != destination.resolve():
                shutil.copy2(original, destination)
        else:
            image.filepath_raw = str(destination)
            image.file_format = "PNG"
            image.save()
        image.filepath_raw = str(destination)
        if original:
            paths[str(original)] = str(destination)
    for obj in scene.objects:
        image = obj.get("cm_reference_normal_image")
        if isinstance(image, bpy.types.Image):
            obj["cm_reference_normal"] = image.filepath
        elif obj.get("cm_reference_normal") in paths:
            obj["cm_reference_normal"] = paths[obj["cm_reference_normal"]]


def start(context, s, f):
    global _SESSION
    from . import bake_jobs, detail_jobs,wrinkle_jobs,map_provenance,sculpt_archive

    if f.sculpt_session_role == "AUTHOR":
        raise ValueError("This is already a separate sculpting session")
    if _SESSION and process_ref.alive(_SESSION["process"]):
        process_ref.focus(_SESSION["process"])
        return _SESSION["folder"]
    if (
        context.mode != "OBJECT"
        or f.body_editing
        or f.full_sculpt
        or f.restore_editing
        or f.expression_active
        or bake_jobs.running()
        or detail_jobs.running()
        or wrinkle_jobs.running()
    ):
        raise ValueError(
            "Finish the active edit or map bake before opening the sculpting session"
        )
    m.check_mesh(s.result, "Accepted head")
    from . import resource_budget
    resource_budget.require('a separate sculpting editor',resource_budget.GIB+
        resource_budget.meshes((s.result,s.detail,f.body_result,f.body_detail,f.wm1,f.wm2,f.wm3)))
    project = Path(bpy.path.abspath(s.project_dir)).resolve()
    folder = project / "DetailSessions" / uuid.uuid4().hex[:12]
    author = folder / "author"
    author.mkdir(parents=True)
    f.sculpt_session_path = str(folder)
    # Recovery commits with the main save. If later window startup fails, the
    # recovered work is already safely retained in Main and can be retried.
    with sculpt_archive.recover(context):
        if bpy.ops.cm.save_project() != {"FINISHED"}:
            raise ValueError("Save the main project before opening its sculpting session")
    for name in ["Maps", "ReferenceMaps"]:
        if (project / name).is_dir():
            shutil.copytree(project / name, author / name)
    initial = {
        slot: stamp(context.scene, slot)
        for slot in ["WM1", "WM2", "WM3"]
        if getattr(f, slot.lower())
    }
    manifest = {
        "schema": 1,
        "session": folder.name,
        "parent": process_ref.identify(os.getpid()),
        "project": str(project),
        "character": s.character_name,
        "author_project": str(author / "project.blend"),
        "snapshot": str(author / "snapshot.blend"),
        "initial_wrinkles": initial,
        "accepted_geometry":map_provenance.accepted_signature(context.scene),
        "main_sculpt_sources":map_provenance.sculpt_sources_signature(context.scene),
        "main_sculpt_layers":sculpt_archive.fingerprints(context.scene),
        "main_sculpt_layers_schema":sculpt_archive.FINGERPRINT_SCHEMA,
    }
    write_json(folder / "session.json", manifest)
    scene = context.scene.copy()
    try:
        scene.name = "Sculpting Session - " + s.character_name
        scene.cm_settings.project_dir = str(author)
        scene.cm_features.sculpt_session_role = "AUTHOR"
        scene.cm_features.sculpt_session_path = str(folder)
        scene.cm_features.ui_stage = "DETAIL"
        scene.cm_features.bake_open_folder = False
        write_snapshot(scene, author / "snapshot.blend")
    finally:
        bpy.data.scenes.remove(scene)
    return _launch(context.scene, folder)


def _launch(scene, folder):
    # A previous process's ready/error marker cannot describe this launch.
    for name in ("author-ready.json", "startup-error.json"):
        (folder / name).unlink(missing_ok=True)
    with (folder / "author.log").open("a", encoding="utf-8") as log:
        process = subprocess.Popen(
            [
                bpy.app.binary_path,
                "--factory-startup",
                "--python",
                str(Path(__file__).with_name("sculpt_session_worker.py")),
                "--",
                str(folder / "session.json"),
            ],
            stdout=log,
            stderr=subprocess.STDOUT,
            creationflags=subprocess.BELOW_NORMAL_PRIORITY_CLASS
            if os.name == "nt"
            else 0,
        )
    _attach(scene, folder, process)
    scene.cm_features.sculpt_session_status = "Separate sculpting window starting..."
    return folder


def resume(context, s, f):
    folder = folder_for(context.scene)
    if not (folder / "session.json").is_file():
        project = Path(bpy.path.abspath(s.project_dir)).resolve()
        candidates = []
        for path in (project / "DetailSessions").glob("*/session.json"):
            try:
                data = read_json(path)
                if (
                    data.get("character") == s.character_name
                    and Path(data.get("project", "")) == project
                ):
                    candidates.append(path)
            except (ValueError, OSError):
                pass
        if not candidates:
            raise ValueError("No saved sculpting session is available")
        folder = max(candidates, key=lambda path: path.stat().st_mtime).parent
        f.sculpt_session_path = str(folder)
    manifest = _manifest(folder)
    if (
        manifest["character"] != s.character_name
        or Path(manifest["project"]) != Path(bpy.path.abspath(s.project_dir)).resolve()
    ):
        raise ValueError(
            "Open the original character project before resuming this session"
        )
    manifest["parent"] = process_ref.identify(os.getpid())
    write_json(folder / "session.json", manifest)
    ready = folder / "author-ready.json"
    if (
        ready.is_file()
        and read_json(ready).get("active", True)
        and process_ref.alive(read_json(ready)["process"])
    ):
        _attach(context.scene, folder)
        process_ref.focus(_SESSION["process"])
        return folder
    return _launch(context.scene, folder)


def prepare_return(context, s, f):
    global _PENDING, _PENDING_STATE
    from . import bake_jobs, detail_jobs,wrinkle_jobs
    from . import map_provenance

    if f.sculpt_session_role != "AUTHOR":
        raise ValueError("Return is available in the separate sculpting window")
    if _PENDING:
        raise ValueError("Waiting for the current return acknowledgement")
    if bake_jobs.running() or detail_jobs.running() or wrinkle_jobs.running() or context.mode != "OBJECT" or f.body_editing or f.expression_active or f.full_sculpt or f.restore_editing:
        raise ValueError("Finish sculpting and the current map bake before returning")
    map_problems=map_provenance.problems(context.scene)
    if map_problems:raise ValueError(map_problems[0])
    folder = folder_for(context.scene)
    manifest = _manifest(folder)
    if not process_ref.alive(manifest["parent"]):
        raise ValueError(
            "Main editor is unavailable. Work is retained; reopen the main project and Resume Session"
        )
    connection = (
        read_json(folder / "connection.json")
        if (folder / "connection.json").is_file()
        else {}
    )
    if (
        not connection.get("connected")
        or connection.get("parent") != manifest["parent"]
    ):
        raise ValueError(
            "Main project is disconnected. Open the original project and Resume Session; work is retained"
        )
    receipts = (
        read_json(folder / "receipts.json")
        if (folder / "receipts.json").is_file()
        else {}
    )
    if receipts.get("FINAL", {}).get("stamp") != stamp(context.scene):
        raise ValueError("Create Final Maps for the current sculpt before returning")
    for name, digest in receipts["FINAL"].get("files", {}).items():
        path = Path(bpy.path.abspath(s.project_dir)) / "Maps" / name
        if (
            not path.is_file()
            or hashlib.sha256(path.read_bytes()).hexdigest() != digest
        ):
            raise ValueError(
                "Final maps changed or are missing; Create Final Maps again"
            )
    for slot, initial in manifest["initial_wrinkles"].items():
        current = stamp(context.scene, slot)
        if current != initial and receipts.get(slot, {}).get("stamp") != current:
            raise ValueError(
                "Bake " + slot + " for its current sculpt before returning"
            )
    for slot in ["WM1", "WM2", "WM3"]:
        if (
            getattr(f, slot.lower())
            and slot not in manifest["initial_wrinkles"]
            and receipts.get(slot, {}).get("stamp") != stamp(context.scene, slot)
        ):
            raise ValueError("Bake the new " + slot + " before returning")
    if (
        Path(bpy.path.abspath(s.project_dir)).resolve()
        != Path(manifest["author_project"]).parent
    ):
        raise ValueError("Restore the session working folder before returning")
    assert bpy.ops.cm.save_project() == {"FINISHED"}
    if manifest.get('accepted_geometry') and manifest['accepted_geometry']!=map_provenance.accepted_signature(context.scene):
        raise ValueError('The accepted head/body or UVs changed in this author window. Its project was saved. Make proportion or accepted-mesh edits in Main, then open a new sculpt session before returning maps.')
    files = []
    allowed = {"NORMAL", "CAVITY", "WM1", "WM2", "WM3", "CM1", "CM2", "CM3"}
    for path in (Path(bpy.path.abspath(s.project_dir)) / "Maps").glob("*.png"):
        suffix = path.stem.removeprefix(s.character_name + "_")
        if (
            suffix in allowed
            or suffix.startswith("Body_")
            and suffix[5:] in {"NORMAL", "CAVITY"}
        ):
            files.append(
                {
                    "name": path.name,
                    "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                }
            )
    if not any(p["name"] == s.character_name + "_NORMAL.png" for p in files):
        raise ValueError("Final normal map is missing")
    from . import sculpt_archive
    request_id = uuid.uuid4().hex
    archive_path = folder / 'returns' / request_id / 'sculpt.blend'
    roles = set(sculpt_archive.roles_for_files(s.character_name, [item['name'] for item in files]).values())
    archive_receipt = sculpt_archive.write(context.scene, archive_path, roles)
    request = {
        "request_id": request_id,
        "session": manifest["session"],
        "files": files,
        "lookdev": {name: getattr(f, name) for name in LOOK_FIELDS},
        "sculpt_archive": archive_receipt,
        "accepted_geometry": map_provenance.accepted_signature(context.scene),
    }
    write_json(folder / "return.json", request)
    _PENDING_STATE = return_state(context.scene)
    _PENDING = request["request_id"]
    f.sculpt_session_status = "Waiting for the main editor to confirm map receipt..."
    return request


def check_return_target(scene,folder,manifest):
    from . import map_provenance,sculpt_archive
    expected=manifest.get('accepted_geometry')
    current=map_provenance.accepted_signature(scene)
    if expected and expected!=current:
        raise ValueError('The accepted head/body or UVs changed since this authoring session opened. Keep its maps and open a session for the updated character.')
    sources=manifest.get('main_sculpt_sources')
    if not expected or not sources:
        # Older sessions retain the original accepted meshes in their snapshot
        # or author project. Read a private Scene and dispose all appended IDs.
        owned=Path(folder).resolve()
        candidates=[Path(manifest.get('snapshot','')),Path(manifest['author_project'])]
        path=next((p for p in candidates if p.is_file() and p.resolve().is_relative_to(owned)),None)
        if not path:raise ValueError('The legacy session target cannot be verified. Keep its author project and open a new session for this character.')
        before=set(bpy.data.user_map())
        try:
            with bpy.data.libraries.load(str(path),link=False) as (source,loaded):
                loaded.scenes=list(source.scenes)
            candidates=[item for item in loaded.scenes if item.cm_features.sculpt_session_role=='AUTHOR'
                and item.cm_settings.character_name==manifest['character'] and item.cm_settings.result]
            if len(candidates)==1:reference=candidates[0]
            elif len(loaded.scenes)==1:reference=loaded.scenes[0]
            else:raise ValueError('The legacy file has ambiguous authoring targets; keep it and open a new session for this character')
            if not expected:expected=map_provenance.accepted_signature(reference)
            if path==Path(manifest.get('snapshot','')):
                sources=map_provenance.sculpt_sources_signature(reference)
        finally:
            bpy.data.batch_remove(ids=set(bpy.data.user_map())-before)
    if expected!=map_provenance.accepted_signature(scene):
        raise ValueError('The accepted head/body or UVs changed since this authoring session opened. Keep its maps and open a session for the updated character.')
    if not sources:raise ValueError('The original sculpt-source snapshot is unavailable. Keep the author project and use its maps as custom files or open a new session.')
    if sources!=map_provenance.sculpt_sources_signature(scene):
        raise ValueError('The main sculpt layers changed since this authoring session opened. Keep the author project and resolve those edits before returning maps.')
    if manifest.get('main_sculpt_layers_schema',sculpt_archive.FINGERPRINT_SCHEMA) != sculpt_archive.FINGERPRINT_SCHEMA:
        raise ValueError('This author session uses unsupported sculpt fingerprints; keep its work for explicit recovery')
    if manifest.get('main_sculpt_layers') and manifest['main_sculpt_layers'] != sculpt_archive.fingerprints(scene):
        raise ValueError('The main sculpt layers or dormant artist keys changed since this authoring session opened. Keep the author project and resolve those edits before returning maps.')
    if not manifest.get('accepted_geometry') or not manifest.get('main_sculpt_sources'):
        manifest['accepted_geometry']=expected;manifest['main_sculpt_sources']=sources
        write_json(Path(folder)/'session.json',manifest)


def receive(job, request):
    from . import bake_jobs, lookdev, detail_jobs,wrinkle_jobs
    from . import map_provenance, sculpt_archive

    scene = job["scene"]
    s, f = scene.cm_settings, scene.cm_features
    manifest = _manifest(job["folder"])
    if request["session"] != manifest["session"]:
        raise ValueError("Return belongs to another sculpting session")
    if bake_jobs.running() or detail_jobs.running() or wrinkle_jobs.running():
        raise ValueError("Finish the map bake in the main editor, then retry Return")
    if bpy.context.mode!='OBJECT' or f.body_editing or f.full_sculpt or f.restore_editing or f.expression_active:
        raise ValueError('Finish the active edit in the main editor, then retry Return')
    if s.character_name != manifest["character"] or Path(
        bpy.path.abspath(s.project_dir)
    ).resolve() != Path(manifest["project"]):
        raise ValueError(
            "The main project or character name changed; restore it before retrying Return"
        )
    check_return_target(scene,job['folder'],manifest)
    author_target=request.get('accepted_geometry')
    if (request.get('sculpt_archive') is not None or author_target is not None) and author_target!=manifest['accepted_geometry']:
        raise ValueError('Returned maps have a different or unverified accepted target. Keep the saved author project and retry Save & Return for the original target.')
    # Request IDs become directory names; never allow a request to escape its
    # owned delivery directory or overwrite an earlier accepted delivery.
    request_id = request.get('request_id', '')
    if not isinstance(request_id, str) or len(request_id) != 32 or any(ch not in '0123456789abcdef' for ch in request_id):
        raise ValueError('Invalid sculpt return request ID')
    names = [item['name'] for item in request['files']]
    if not names or len(set(names)) != len(names):
        raise ValueError('Returned maps are empty or contain duplicate filenames')
    archive_roles = sculpt_archive.roles_for_files(s.character_name, names)
    for name, value in request["lookdev"].items():
        if name not in LOOK_FIELDS:
            raise ValueError("Unknown LookDev setting")
        prop = f.bl_rna.properties[name]
        if prop.type == "ENUM" and value not in {
            item.identifier for item in prop.enum_items
        }:
            raise ValueError("Unsupported LookDev setting: " + name)
        if prop.type in {"FLOAT", "INT"} and (
            not isinstance(value, (float, int))
            or not math.isfinite(value)
            or not prop.hard_min <= value <= prop.hard_max
        ):
            raise ValueError("Invalid LookDev value: " + name)
        if prop.type == "BOOLEAN" and not isinstance(value, bool):
            raise ValueError("Invalid LookDev switch: " + name)
    delivery = job["folder"] / "deliveries" / request["request_id"]
    (delivery / "staging/Maps").mkdir(parents=True)
    archive_receipt = request.get('sculpt_archive')
    archive_path = delivery / 'sculpt.blend'
    for item in request["files"]:
        if Path(item["name"]).name != item["name"]:
            raise ValueError("Invalid returned filename")
        source = Path(manifest["author_project"]).parent / "Maps" / item["name"]
        if hashlib.sha256(source.read_bytes()).hexdigest() != item["sha256"]:
            raise ValueError("Returned map changed; retry Save & Return")
        image = bpy.data.images.load(str(source), check_existing=False)
        try:
            if not all(image.size):
                raise ValueError("Returned image has no pixel data")
        finally:
            bpy.data.images.remove(image)
        shutil.copy2(source, delivery / "staging/Maps" / item["name"])
    if archive_receipt is not None:
        if set(archive_receipt.get('roles', [])) != set(archive_roles.values()):
            raise ValueError('Returned archive does not declare the returned map groups')
        source_archive = job['folder'] / 'returns' / request_id / 'sculpt.blend'
        # Validate both before and after copying. Later author saves are irrelevant.
        if hashlib.sha256(source_archive.read_bytes()).hexdigest() != archive_receipt.get('sha256'):
            raise ValueError('Returned sculpt archive changed; retry Save & Return')
        try:
            shutil.copy2(source_archive, archive_path)
            with sculpt_archive.read(archive_path, archive_receipt) as recovered:
                if set(recovered) - set(archive_roles.values()):
                    raise ValueError('Returned archive contains unrelated sculpt roles')
        except Exception:
            archive_path.unlink(missing_ok=True)
            raise
    destination = Path(manifest["project"]) / "Maps"
    objects = list(
        dict.fromkeys(
            obj
            for obj in (
                s.result,
                s.detail,
                f.body_result,
                f.body_detail,
                f.wm1,
                f.wm2,
                f.wm3,
            )
            if obj
        )
    )
    material_slots = {obj: list(obj.data.materials) for obj in objects}
    nodes = {
        node: node.image
        for obj in objects
        for material in obj.data.materials
        if material and material.use_nodes
        for node in material.node_tree.nodes
        if node.type == "TEX_IMAGE"
    }
    settings = {name: getattr(f, name) for name in LOOK_FIELDS}
    old_show = f.show_lookdev
    old_status = (f.sculpt_session_status, s.status)
    provenance_path=map_provenance.manifest(scene)
    old_provenance=provenance_path.read_bytes() if provenance_path.exists() else None
    old_invalidation=scene.get(map_provenance.INVALIDATED)
    old_archives=scene.get(sculpt_archive.RETURNED)
    original_materials = set(bpy.data.materials)
    original_images = set(bpy.data.images)
    shader_values = {
        socket: socket.default_value
        for materials in material_slots.values()
        for material in materials
        if material and material.use_nodes
        for node in material.node_tree.nodes
        for socket in (
            [node.inputs["Roughness"], node.inputs["Subsurface Weight"]]
            if node.type == "BSDF_PRINCIPLED"
            else [node.inputs["Strength"]]
            if node.type == "NORMAL_MAP"
            else []
        )
    }
    shading_fields = (
        "type",
        "use_scene_lights",
        "use_scene_world",
        "use_studiolight_view_rotation",
        "studio_light",
        "studiolight_rotate_z",
        "studiolight_intensity",
        "studiolight_background_alpha",
        "studiolight_background_blur",
    )
    shading_values = (
        {
            area.spaces.active.shading: {
                name: getattr(area.spaces.active.shading, name)
                for name in shading_fields
            }
            for area in bpy.context.screen.areas
            if area.type == "VIEW_3D"
        }
        if bpy.context.screen
        else {}
    )
    delivered = False
    try:
        bake_jobs.deliver(
            {"folder": delivery, "destination": destination}, request["files"]
        )
        delivered = True
        bake_jobs.refresh_maps(scene, destination, request["files"])
        with bpy.context.temp_override(scene=scene, view_layer=scene.view_layers[0]):
            with lookdev.suspend_updates():
                for name in LOOK_FIELDS:
                    if name in request["lookdev"]:
                        setattr(f, name, request["lookdev"][name])
                f.show_lookdev = True
            lookdev.update(f, bpy.context)
        returned_basis=map_provenance.basis_for_files(scene,{item['name'] for item in request['files']})
        map_provenance.record(scene,returned_basis,origin='SCULPT_SESSION',outputs={item['name'] for item in request['files']})
        if archive_receipt is not None:
            sculpt_archive.record_return(scene, archive_path, archive_receipt, request['files'], manifest)
        else:
            # Retain a chronology barrier: a later unverified return must not
            # accidentally recover the sculpt behind an older cavity artifact.
            sculpt_archive.record_return(scene, None, None, request['files'], manifest)
        f.sculpt_session_status = "Maps received from sculpting session."
        s.status = (
            "Sculpted maps and LookDev settings received; accepted geometry preserved."
        )
        with bpy.context.temp_override(scene=scene, view_layer=scene.view_layers[0]):
            if bpy.ops.cm.save_project() != {"FINISHED"}:
                raise ValueError("Could not save received maps; retry Return")
    except Exception:
        archive_path.unlink(missing_ok=True)
        if old_archives is None:
            if sculpt_archive.RETURNED in scene:del scene[sculpt_archive.RETURNED]
        else:scene[sculpt_archive.RETURNED]=old_archives
        if old_provenance is None:provenance_path.unlink(missing_ok=True)
        else:provenance_path.write_bytes(old_provenance)
        if old_invalidation is None:
            if map_provenance.INVALIDATED in scene:del scene[map_provenance.INVALIDATED]
        else:scene[map_provenance.INVALIDATED]=old_invalidation
        if delivered:
            for item in request["files"]:
                backup = delivery / "backup" / item["name"]
                target = destination / item["name"]
                if backup.is_file():
                    shutil.copy2(backup, target)
                else:
                    target.unlink(missing_ok=True)
        with lookdev.suspend_updates():
            for name, value in settings.items():
                setattr(f, name, value)
            f.show_lookdev = old_show
        for obj, materials in material_slots.items():
            for index, material in enumerate(materials):
                obj.data.materials[index] = material
        for node, image in nodes.items():
            node.image = image
        for socket, value in shader_values.items():
            socket.default_value = value
        for shading, values in shading_values.items():
            for name, value in values.items():
                setattr(shading, name, value)
        f.sculpt_session_status, s.status = old_status
        for material in set(bpy.data.materials) - original_materials:
            if not material.users:
                bpy.data.materials.remove(material)
        for image in set(bpy.data.images) - original_images:
            if not image.users:
                bpy.data.images.remove(image)
        raise
    write_json(
        job["folder"] / "response.json",
        {
            "request_id": request["request_id"],
            "status": "accepted",
            "files": request["files"],
        },
    )
    job["accepted"] = request["request_id"]


def poll_main():
    global _SESSION
    job = _SESSION
    if not job:
        return None
    try:
        error = job["folder"] / "startup-error.json"
        if error.is_file():
            message = read_json(error).get(
                "error", "Could not open the sculpting window"
            )
            job["scene"].cm_features.sculpt_session_status = (
                "Sculpting session could not start: "
                + message
                + ". Saved work is retained; use Resume."
            )
            write_json(job["folder"] / "connection.json", {"connected": False})
            _SESSION = None
            return None
        if not process_ref.alive(job["process"]):
            write_json(job["folder"] / "connection.json", {"connected": False})
            job["scene"].cm_features.sculpt_session_status = (
                "Sculpted maps received. Session saved and ready to resume."
                if job.get("accepted")
                else "Sculpting window closed; saved work can be resumed."
            )
            _SESSION = None
            return None
        ready = job["folder"] / "author-ready.json"
        if ready.is_file() and not read_json(ready).get("active", True):
            write_json(job["folder"] / "connection.json", {"connected": False})
            job[
                "scene"
            ].cm_features.sculpt_session_status = (
                "Authoring window changed projects; saved work can be resumed."
            )
            _SESSION = None
            return None
        if (
            not job["ready"]
            and ready.is_file()
            and read_json(ready).get("process") == job["process"]
        ):
            job["ready"] = True
            job[
                "scene"
            ].cm_features.sculpt_session_status = "Separate sculpting session is open."
        path = job["folder"] / "return.json"
        if path.is_file():
            request = read_json(path)
            if request["request_id"] != job["seen"]:
                job["seen"] = request["request_id"]
                try:
                    receive(job, request)
                except Exception as exc:
                    write_json(
                        job["folder"] / "response.json",
                        {
                            "request_id": request["request_id"],
                            "status": "rejected",
                            "error": str(exc),
                        },
                    )
                    job["scene"].cm_features.sculpt_session_status = (
                        "Return failed: " + str(exc)
                    )
    except ReferenceError:
        _SESSION = None
        return None
    except (OSError, ValueError) as exc:
        print("Sculpt session connection:", exc)
    return 0.25


def poll_author():
    global _PENDING, _PENDING_STATE
    scene = bpy.context.scene
    f = scene.cm_features
    if f.sculpt_session_role != "AUTHOR":
        return None
    folder = folder_for(scene)
    try:
        if _PENDING and (folder / "response.json").is_file():
            response = read_json(folder / "response.json")
            if response["request_id"] == _PENDING:
                _PENDING = None
                if response["status"] == "accepted":
                    from . import bake_jobs, detail_jobs

                    if bake_jobs.running() or detail_jobs.running() or return_state(scene) != _PENDING_STATE:
                        f.sculpt_session_status = "Maps received, but this session changed while returning. Work retained; finish baking and return again."
                        bpy.ops.cm.save_project()
                        return 0.25
                    process_ref.focus(_manifest(folder)["parent"])
                    bpy.ops.wm.quit_blender("EXEC_DEFAULT")
                    return None
                f.sculpt_session_status = (
                    "Return failed; work retained. "
                    + response.get("error", "Retry Save & Return")
                )
        if (
            _PENDING
            and (folder / "connection.json").is_file()
            and not read_json(folder / "connection.json").get("connected")
        ):
            _PENDING = None
            f.sculpt_session_status = "Main project disconnected; work retained. Resume the session in the main editor, then retry Return."
    except (OSError, ValueError) as exc:
        f.sculpt_session_status = "Connection interrupted; work retained. " + str(exc)
    return 0.25


@persistent
def attach_author(_=None):
    global _PENDING, _PENDING_STATE
    _PENDING = None
    _PENDING_STATE = None
    if bpy.app.background or not hasattr(bpy.context.scene, "cm_features"):
        return
    if (
        bpy.context.scene.cm_features.sculpt_session_role == "AUTHOR"
        and not bpy.app.timers.is_registered(poll_author)
    ):
        write_json(
            folder_for(bpy.context.scene) / "author-ready.json",
            {"process": process_ref.identify(os.getpid()), "active": True},
        )
        bpy.app.timers.register(poll_author, first_interval=0.25, persistent=True)


@persistent
def detach(_=None):
    global _SESSION
    if _SESSION:
        try:
            write_json(_SESSION["folder"] / "connection.json", {"connected": False})
        except OSError:
            pass
    if (
        not bpy.app.background
        and hasattr(bpy.context.scene, "cm_features")
        and bpy.context.scene.cm_features.sculpt_session_role == "AUTHOR"
    ):
        try:
            write_json(
                folder_for(bpy.context.scene) / "author-ready.json",
                {"process": process_ref.identify(os.getpid()), "active": False},
            )
        except OSError:
            pass
    _SESSION = None
    if bpy.app.timers.is_registered(poll_main):
        bpy.app.timers.unregister(poll_main)
    if bpy.app.timers.is_registered(poll_author):
        bpy.app.timers.unregister(poll_author)


def register():
    bpy.app.handlers.load_pre.append(detach)
    bpy.app.handlers.load_post.append(attach_author)


def unregister():
    detach()
    for handlers, function in [
        (bpy.app.handlers.load_pre, detach),
        (bpy.app.handlers.load_post, attach_author),
    ]:
        if function in handlers:
            handlers.remove(function)
