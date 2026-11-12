"""Custom body interchange while retaining the accepted MetaHuman topology."""

from pathlib import Path
import shutil
from contextlib import contextmanager
from tempfile import mkdtemp
import bpy
import numpy as np
from . import mesh_ops as m, workflow


@contextmanager
def _save_stage(parent):
    stage=Path(mkdtemp(prefix='.cm-body-save-',dir=parent)).resolve()
    if stage.parent!=parent:raise ValueError('Custom body staging escaped its destination')
    recovery={'retain':False}
    try:yield stage,recovery
    finally:
        if not recovery['retain']:
            if stage.resolve().parent!=parent or stage.is_symlink():
                raise ValueError('Custom body staging moved; cleanup refused')
            shutil.rmtree(stage)


def save_custom_body(context, s, f, filepath):
    """Publish a reusable body and its native-state sidecar together."""
    from . import character_project, native_body, resource_budget
    character_project.preflight(context, s, f)
    obj=f.body_result or (s.result if s.part=='BODY' else None)
    m.check_mesh(obj, 'Load or create an accepted body before saving a custom body')
    if obj.modifiers:
        raise ValueError('Apply live body modifiers before saving a reusable custom body')
    target=Path(filepath).resolve()
    if target.suffix.lower()!='.obj':raise ValueError('Save the custom body as an OBJ file')
    sidecar=target.with_suffix('.cm-body.json')
    outputs=(target,sidecar)
    for path in outputs:
        if path.exists() and not path.is_file():raise ValueError('Custom body output must be a regular file: '+path.name)
    previous=sum(path.stat().st_size for path in outputs if path.is_file())
    state=obj.get(native_body._KEY)
    state_bytes=len(state.as_string().encode('utf-8')) if isinstance(state,bpy.types.Text) else 0
    estimate=64*resource_budget.MIB+resource_budget.meshes([obj])*2+state_bytes*2+previous
    budget=resource_budget.require_disk('save custom body',target.parent,estimate)
    target.parent.mkdir(parents=True,exist_ok=True)
    with _save_stage(target.parent) as (stage,recovery):
        staged=stage/target.name
        with m.selection_state(context):
            m.activate(context,obj)
            bpy.ops.wm.obj_export(filepath=str(staged),export_selected_objects=True,export_materials=False)
        native_body.save_custom_state(obj,staged)
        backups={path:stage/('previous-'+path.name) if path.exists() else None for path in outputs}
        for path,backup in backups.items():
            if backup:shutil.copy2(path,backup)
        published=[]
        try:
            for path in outputs:
                (stage/path.name).replace(path);published.append(path)
        except BaseException as error:
            failures=[]
            for path in reversed(published):
                backup=backups[path]
                try:
                    if backup:backup.replace(path)
                    else:path.unlink(missing_ok=True)
                except OSError as rollback_error:failures.append(str(rollback_error))
            if failures:
                recovery['retain']=True
                raise RuntimeError('Custom body save failed and could not restore all outputs. '
                    'Previous files are retained for recovery in '+str(stage)+': '+'; '.join(failures)) from error
            raise
    s.status='Custom body saved with its native state. Materials and textures stay in the character project.'
    return dict(mesh=str(target),state=str(sidecar),storage_budget=budget)


def coordinates_from_uvs(reference, source, tolerance=3e-6):
    """Recover template-order positions from UV-preserving OBJ/FBX exports."""
    m.check_mesh(reference, "Body template")
    m.check_mesh(source, "Custom body")
    if not reference.data.uv_layers or not source.data.uv_layers:
        raise ValueError("Custom body must retain its MetaHuman UV map")
    world = m.coordinates(source)
    buckets = {}
    for loop, uv in zip(source.data.loops, source.data.uv_layers.active.data):
        point = tuple(uv.uv)
        key = tuple(np.rint(np.asarray(point) / tolerance).astype(int))
        buckets.setdefault(key, []).append(
            (np.asarray(point), world[loop.vertex_index])
        )
    positions = np.zeros((len(reference.data.vertices), 3))
    counts = np.zeros(len(positions))
    spread_limit = max(np.linalg.norm(np.ptp(world, axis=0)) * 1e-5, 1e-7)
    cache = {}
    for loop, uv in zip(reference.data.loops, reference.data.uv_layers.active.data):
        point = tuple(uv.uv)
        if point not in cache:
            array = np.asarray(point)
            cell = np.rint(array / tolerance).astype(int)
            matches = [
                (np.linalg.norm(array - candidate_uv), coordinate)
                for x in (-1, 0, 1)
                for y in (-1, 0, 1)
                for candidate_uv, coordinate in buckets.get(tuple(cell + [x, y]), [])
                if np.linalg.norm(array - candidate_uv) <= tolerance
            ]
            if not matches:
                raise ValueError("Custom body UVs do not match this MetaHuman template")
            closest = min(distance for distance, _ in matches)
            selected = np.asarray(
                [
                    value
                    for distance, value in matches
                    if distance <= closest + tolerance * 0.1
                ]
            )
            if np.linalg.norm(np.ptp(selected, axis=0)) > spread_limit:
                raise ValueError(
                    "Custom body has overlapping UVs with ambiguous vertex correspondence"
                )
            cache[point] = selected.mean(0)
        positions[loop.vertex_index] += cache[point]
        counts[loop.vertex_index] += 1
    if np.any(counts == 0):
        raise ValueError(
            "Body template has isolated vertices without UV correspondence"
        )
    result = positions / counts[:, None]
    if not np.isfinite(result).all():
        raise ValueError("Custom body contains non-finite coordinates")
    return result


def load_custom_body(context, s, f, filepath):
    if f.body_editing or f.full_sculpt or context.mode != "OBJECT":
        raise ValueError(
            "Finish the current body/sculpt pass before loading a custom body"
        )
    m.check_mesh(f.body_template, "Load the base body first")
    m.check_mesh(f.body_result, "Body result")
    previous = set(bpy.data.objects)
    try:
        extension = Path(filepath).suffix.lower()
        if extension == ".obj":
            bpy.ops.wm.obj_import(filepath=filepath)
        elif extension == ".fbx":
            bpy.ops.import_scene.fbx(filepath=filepath)
        else:
            raise ValueError(
                "Use a UV-preserving OBJ or FBX body exported from this workflow"
            )
        context.view_layer.update()
        meshes = [obj for obj in set(bpy.data.objects) - previous if obj.type == "MESH"]
        if len(meshes) != 1:
            raise ValueError("Custom body file must contain exactly one body mesh")
        positions = coordinates_from_uvs(f.body_template, meshes[0])
        from .anatomy import match_neck

        head = f.head_result or s.result
        match_neck(
            m.evaluated_coordinates(context, head),
            workflow.neck_ring(head),
            positions,
            workflow.neck_ring(f.body_result, False),
        )
        workflow.key_from_world(f.body_result, "Custom Body", positions)
        from .native_body import load_custom_state
        load_custom_state(f.body_result,filepath)
        f.body_result["cm_custom_body"] = str(Path(filepath).resolve())
        context.view_layer.update()
        # Reattach the existing accepted head to this body's complete neck.
        workflow.attach_neck(context, s, f)
    finally:
        for obj in set(bpy.data.objects) - previous:
            bpy.data.objects.remove(obj, do_unlink=True)
    m.activate(context, f.body_result)
    s.status = "Custom body loaded; original topology and UVs retained, neck attached."
