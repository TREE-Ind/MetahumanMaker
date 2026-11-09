from pathlib import Path
import json
import subprocess
import bpy
import numpy as np
from mathutils import Vector
from . import mesh_ops as m


def portrait(context, obj, filename, axis="-Y", bounds=None, *, color_type="SINGLE", light="STUDIO"):
    world = m.evaluated_coordinates(context, obj)
    if bounds is not None:
        world = np.asarray(bounds, dtype=float)
    center = (world.min(0) + world.max(0)) * 0.5
    scale = max(np.ptp(world, axis=0)) * 1.15
    direction = {"X": (1, 0, 0), "-X": (-1, 0, 0), "Y": (0, 1, 0), "-Y": (0, -1, 0)}[
        axis
    ]
    scene = bpy.data.scenes.new("CM_LandmarkPortrait")
    evaluated = obj.evaluated_get(context.evaluated_depsgraph_get())
    mesh = bpy.data.meshes.new_from_object(
        evaluated, depsgraph=context.evaluated_depsgraph_get()
    )
    mesh_obj = bpy.data.objects.new("CM_PortraitMesh", mesh)
    scene.collection.objects.link(mesh_obj)
    mesh_obj.matrix_world = obj.matrix_world.copy()
    camera_data = bpy.data.cameras.new("CM_PortraitCamera")
    camera = bpy.data.objects.new("CM_PortraitCamera", camera_data)
    scene.collection.objects.link(camera)
    camera.location = Vector(center) + Vector(direction) * scale * 3
    camera.rotation_euler = (-Vector(direction)).to_track_quat("-Z", "Y").to_euler()
    camera_data.type = "ORTHO"
    camera_data.ortho_scale = scale
    scene.camera = camera
    scene.render.engine = "BLENDER_WORKBENCH"
    scene.display.shading.light = light
    scene.display.shading.color_type = color_type
    scene.display.shading.single_color = (0.65, 0.65, 0.65)
    scene.display.shading.background_type = "WORLD"
    scene.world = bpy.data.worlds.new("CM_PortraitWorld")
    scene.world.color = (0.08, 0.08, 0.08)
    scene.render.resolution_x = scene.render.resolution_y = 768
    scene.render.resolution_percentage = 100
    from .blender_compat import render_image_format
    render_image_format(scene.render.image_settings,'PNG')
    scene.render.filepath = str(filename)
    try:
        scene.view_layers[0].update()
        bpy.ops.render.render(scene=scene.name, write_still=True)
        return camera.matrix_world.copy(), scale
    finally:
        bpy.data.scenes.remove(scene)
        bpy.data.objects.remove(camera, do_unlink=True)
        bpy.data.objects.remove(mesh_obj, do_unlink=True)
        bpy.data.meshes.remove(mesh)
        bpy.data.cameras.remove(camera_data)


def auto_landmarks(context, s, f, combined=False, *, portrait_bounds=None, append=False):
    from .features import generated_head_update, head_source_scope
    if s.part=='FULL_BODY' and generated_head_update(s):
        with head_source_scope(s):
            return auto_landmarks(context,s,f,combined=combined)
    from .setup_paths import resolve

    resolve(f)
    if s.part == "FULL_BODY" and f.body_template and not combined:
        head = s.template
        body = f.body_template
        vertices = []
        faces = []
        for obj in (head, body):
            offset = len(vertices)
            vertices.extend(m.coordinates(obj).tolist())
            faces.extend(
                [
                    [v + offset for v in polygon.vertices]
                    for polygon in obj.data.polygons
                ]
            )
        mesh = bpy.data.meshes.new("CM_AutoCombinedTemplate")
        mesh.from_pydata(vertices, [], faces)
        mesh.update()
        temporary = bpy.data.objects.new("CM_AutoCombinedTemplate", mesh)
        context.collection.objects.link(temporary)
        old_kind = f.landmark_kind
        try:
            s.template = temporary
            f.landmark_kind = "POSE"
            auto_landmarks(context, s, f, combined=True)
            bvhs = {obj: m.source_bvh(context, obj) for obj in (head, body)}
            for pair in s.landmarks:
                marker = pair.template
                position = marker.matrix_world.translation.copy()
                parent = min(
                    (head, body), key=lambda obj: bvhs[obj].find_nearest(position)[3]
                )
                marker.parent = parent
                marker.matrix_parent_inverse.identity()
                marker.location = parent.matrix_world.inverted() @ position
        finally:
            s.template = head
            f.landmark_kind = old_kind
            bpy.data.objects.remove(temporary, do_unlink=True)
            bpy.data.meshes.remove(mesh)
        full_body_face_landmarks(context, s, f)
        return
    m.check_inputs(s)
    executable = Path(bpy.path.abspath(f.inference_python))
    if not executable.is_file():
        raise ValueError(
            "Choose optional inference Python (.venv/Scripts/python.exe); run tools/setup_models.py first"
        )
    folder = Path(bpy.path.abspath(s.project_dir)) / "Automation"
    folder.mkdir(parents=True, exist_ok=True)
    cameras, images = {}, {}
    for role in ("source", "template"):
        image = folder / (role + ("_face_portrait.png" if portrait_bounds else "_portrait.png"))
        cameras[role] = portrait(context, getattr(s, role), image, f.front_axis,
            bounds=portrait_bounds.get(role) if portrait_bounds else None)
        images[role] = str(image)
    request = folder / "detect-request.json"
    response = folder / "detect-result.json"
    response.unlink(missing_ok=True)
    request.write_text(
        json.dumps(
            {
                "models": bpy.path.abspath(f.model_dir),
                "kind": f.landmark_kind,
                "images": images,
            }
        )
    )
    completed = subprocess.run(
        [
            str(executable),
            str(Path(__file__).with_name("detect_worker.py")),
            str(request),
            str(response),
        ],
        capture_output=True,
        text=True,
        timeout=120,
        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
    )
    if completed.returncode or not response.exists():
        raise RuntimeError(
            "Local detector did not return a result: " + completed.stderr[-400:]
        )
    detected = json.loads(response.read_text())
    if detected.get("status") != "complete":
        raise ValueError(detected.get("error", "Landmark detection failed"))
    samples = detected["landmarks"]
    shared = sorted(set(samples["source"]) & set(samples["template"]), key=int)
    bvhs = {role: m.source_bvh(context, getattr(s, role)) for role in cameras}
    (folder / "projection-debug.json").write_text(
        json.dumps(
            {
                role: {"matrix": list(map(list, value[0])), "scale": value[1]}
                for role, value in cameras.items()
            }
        )
    )
    points = []
    for index in shared:
        pair = {}
        for role in ("source", "template"):
            matrix, scale = cameras[role]
            x, y = samples[role][index]
            origin = matrix @ Vector(((x - 0.5) * scale, (0.5 - y) * scale, 0))
            direction = matrix.to_quaternion() @ Vector((0, 0, -1))
            hit, _, _, _ = bvhs[role].ray_cast(origin, direction)
            if hit is None:
                break
            pair[role] = getattr(s, role).matrix_world.inverted() @ hit
        if len(pair) == 2:
            points.append((index, pair))
    if len(points) < 3:
        raise ValueError(
            "Fewer than three detected points hit both surfaces; use Manual"
        )
    from . import add_marker

    if not append:
        bpy.ops.cm.clear_landmarks()
    for index, positions in points:
        pair = s.landmarks.add()
        pair.name = f"{f.landmark_kind}_{index}"
        for role in ("source", "template"):
            setattr(
                pair,
                role,
                add_marker(
                    context,
                    getattr(s, role),
                    positions[role],
                    f"CM_Auto_{index}_{role}",
                ),
            )
    context.view_layer.update()
    s.status = f"{len(points)} automatic {f.landmark_kind.lower()} pairs. Inspect points and correct manually before wrapping."


def full_body_face_landmarks(context, s, f):
    """Supplement body pose controls with a face-sized detection on each head."""
    bounds = {}
    old_kind = f.landmark_kind
    try:
        for role in ('source', 'template'):
            markers = {pair.name: getattr(pair, role).matrix_world.translation
                       for pair in s.landmarks if pair.name.startswith('POSE_')}
            ears = [markers[name].z for name in ('POSE_7', 'POSE_8') if name in markers]
            shoulders = [markers[name].z for name in ('POSE_11', 'POSE_12') if name in markers]
            if not ears or not shoulders:
                raise ValueError('Head crop needs detected ears and shoulders')
            ear, shoulder = float(np.mean(ears)), float(np.mean(shoulders))
            if ear <= shoulder:
                raise ValueError('Head crop needs an upright reference')
            points = m.evaluated_coordinates(context, getattr(s, role))
            points = points[points[:, 2] >= ear - .7 * (ear - shoulder)]
            if len(points) < 3:
                raise ValueError('Head crop has no usable surface')
            bounds[role] = (points.min(0), points.max(0))
        f.landmark_kind = 'FACE'
        auto_landmarks(context, s, f, combined=True, portrait_bounds=bounds, append=True)
        s.source['cm_face_landmarks_warning'] = ''
    except (ValueError, RuntimeError) as exc:
        # Retain valid pose controls when a stylized or occluded face cannot be
        # detected. This is an explicit review requirement, not silent success.
        s.source['cm_face_landmarks_warning'] = 'Facial landmarks unavailable: ' + str(exc)
        s.status = s.source['cm_face_landmarks_warning'] + '. Review the head fit manually.'
    finally:
        f.landmark_kind = old_kind


def align_detect(context, s, f):
    """Align, detect and refine one private source/template landmark set."""
    from .features import generated_head_update,head_source_scope
    if s.part=='FULL_BODY' and generated_head_update(s):
        # Keep placement and validation in the same scope as detection. Restoring
        # FULL_BODY between these steps mistakes a generated face for a body
        # with missing knees/ankles and undoes its valid head alignment.
        with head_source_scope(s):
            return align_detect(context,s,f)
    from .guided import align_for_detection
    from mathutils import Matrix
    from .geometry import similarity
    original_matrix=s.source.matrix_world.copy()
    original_axis=s.source.get('cm_front_axis')
    if 'cm_body_landmarks_warning' in s.source:del s.source['cm_body_landmarks_warning']
    align_for_detection(context, s, f)
    auto_landmarks(context, s, f)
    context.view_layer.update()
    if s.part != "FULL_BODY":
        m.align_source(context, s)
    else:
        required={'POSE_25','POSE_26','POSE_27','POSE_28'}
        missing=required-{p.name for p in s.landmarks}
        if missing:
            # Upper-body-only similarity changes a stylized character's scale
            # and lifts its feet. Keep the imported reference frame and retain
            # local source markers for correction instead of publishing that fit.
            s.source.matrix_world=original_matrix
            if original_axis is None:
                if 'cm_front_axis' in s.source:del s.source['cm_front_axis']
            else:s.source['cm_front_axis']=original_axis
            s.source['cm_body_landmarks_warning']='Knee/ankle landmarks incomplete. Correct lower-body points before full-body wrapping'
            context.view_layer.update();f.landmarks_reviewed=False
            return
    
        # Facial detail controls must not bias the full-body placement toward
        # the head merely because there are more points on the face.
        placement = [p for p in s.landmarks if p.name.startswith('POSE_')] or list(s.landmarks)
        source = np.array([p.source.matrix_world.translation[:] for p in placement])
        target = np.array([p.template.matrix_world.translation[:] for p in placement])
        rotation, scale, translation = similarity(source, target)
        transform = Matrix.Identity(4)
        for i in range(3):
            for j in range(3):
                transform[i][j] = float(rotation[j, i] * scale)
        transform.translation = Vector(translation)
        s.source.matrix_world = transform @ s.source.matrix_world
        context.view_layer.update()
    f.landmarks_reviewed = False
