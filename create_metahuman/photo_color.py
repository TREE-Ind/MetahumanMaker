"""Owned photo-color work through the existing photo job lifecycle."""

import hashlib
import json
import os
import shutil
import subprocess
import time
import uuid
from pathlib import Path
from tempfile import TemporaryDirectory

import bpy
import numpy as np
from bpy.props import BoolProperty, EnumProperty, FloatProperty, IntProperty, StringProperty

from . import (
    character_project,
    map_provenance,
    paint_session,
    photo_review,
    skin_library,
    skin_projection,
)

SUPPORT_LIMIT = 4
SUPPORT_KEY = 'cm_photo_support_'


def supporting_photos(source):
    return [(index, image) for index in range(SUPPORT_LIMIT)
            if isinstance(image := source.get(SUPPORT_KEY + str(index)), bpy.types.Image)]


def packed_photo(image):
    if not image.packed_file:
        raise ValueError('Re-add the supporting photo to pack it in this project')
    if image.is_dirty:
        raise ValueError('Re-add an edited supporting photo before projecting it')
    return bytes(image.packed_file.data)


def world_landmarks(context, source):
    from . import head_model, photo_fit_core
    f = context.scene.cm_features
    model = head_model.load_model(f.model_dir)
    definition = np.loadtxt(Path(bpy.path.abspath(f.model_dir)) / 'head_sparse_68.txt')
    mean, modes = photo_fit_core.landmarks(model['template_vertex_positions'],
                                          model['vertex_identity_basis'], definition)
    identity = np.asarray(json.loads(source['cm_identity']), dtype=np.float64)
    raw = mean + np.tensordot(identity[:170], modes, axes=(0, 0))
    local = raw[:, [0, 2, 1]] * [1., -1., 1.] * source['cm_gnm_model_scale']
    # Use the actual generated source wherever its mesh still contains all
    # landmark contributors. This includes accepted feature-control edits.
    groups = list(map(str, model['vertex_group_names']))
    keep = np.flatnonzero(model['vertex_groups'][groups.index('skin')] > .5)
    if len(keep) == len(source.data.vertices):
        remap = np.full(len(model['template_vertex_positions']), -1, dtype=int)
        remap[keep] = np.arange(len(keep))
        ids = remap[definition[:, ::2].astype(int)]
        weights = definition[:, 1::2]
        complete = ((ids >= 0) | (weights == 0)).all(axis=1)
        coords = np.empty(len(keep) * 3, dtype=np.float32)
        source.data.vertices.foreach_get('co', coords)
        w = weights[complete]
        local[complete] = (coords.reshape(-1, 3)[ids[complete].clip(0)] * w[..., None]).sum(1) / w.sum(1)[:, None]
    matrix = np.asarray(source.matrix_world)
    return local @ matrix[:3, :3].T + matrix[:3, 3]


def base_image(image):
    """Recover the pre-photo skin, including legacy repeated-projection chains."""
    seen = set()
    while image.get('cm_photo_reference'):
        if image.as_pointer() in seen:
            raise ValueError('Photo skin history contains a cycle; apply a base skin before projecting again')
        seen.add(image.as_pointer())
        previous = image.get('cm_photo_previous_albedo')
        if not isinstance(previous, bpy.types.Image):
            raise ValueError('The original photo base skin is unavailable; apply a base skin before projecting again')
        image = previous
    return image


def eyelid_protection(context, obj):
    """Return a smooth photo mask from native eyelid joint weights."""
    return joint_feature_protection(context, obj, 'eyelid')


def joint_feature_protection(context, obj, feature):
    """Retain native cavity boundaries where source shadows can double them."""
    groups = {group.index for group in obj.vertex_groups
              if group.name.rsplit(':', 1)[-1].lower().startswith('facial_')
              and feature in group.name.lower()}
    depsgraph = context.evaluated_depsgraph_get()
    evaluated = obj.evaluated_get(depsgraph)
    mesh = evaluated.to_mesh(preserve_all_data_layers=True, depsgraph=depsgraph)
    try:
        values = np.array([sum(weight.weight for weight in vertex.groups if weight.group in groups)
                           for vertex in mesh.vertices], dtype=np.float32)
    finally:
        evaluated.to_mesh_clear()
    values = np.clip((values - .02) / .48, 0, 1)
    return values * values * (3 - 2 * values)


def skin_feature_protection(context, obj):
    return np.maximum(eyelid_protection(context, obj),
                      joint_feature_protection(context, obj, 'nostrilthickness'))


def blink_protection(context, obj):
    """Protect skin exposed by a native blink in the primary photo camera."""
    from . import expressions, mesh_ops, geometry
    s, f = context.scene.cm_settings, context.scene.cm_features
    if (not s.template or s.template.type != 'MESH' or not obj.data.uv_layers.active
            or mesh_ops.topology(obj) != mesh_ops.topology(s.template)):
        return None
    try:
        if not expressions.reference_path(f).is_file():
            return None
        data = expressions.bank(f)
        meshes = {mesh['name']: mesh for mesh in data['meshes']}
        head = meshes['head_lod0_mesh']
        neutral = np.asarray(head['neutral'])
        blink_positions = []
        for pose in data['poses']:
            controls = pose['controls']
            if (not controls or max(controls.values()) < .9
                    or any(name.rsplit('.', 1)[-1].lower() not in {'eyeblinkl', 'eyeblinkr'}
                           for name in controls)):
                continue
            sample = next(mesh for mesh in pose['meshes'] if mesh['index'] == head['index'])
            posed = expressions.binary(data['_folder'], sample['vertices_file'], '<f4', neutral.shape)
            blink_positions.append(posed)
        if not blink_positions:
            return None
        mapping = expressions.uv_mapping(s.template, head, reference_data=data)
        mesh = obj.data
        mesh.calc_loop_triangles()
        triangles = np.array([triangle.vertices[:] for triangle in mesh.loop_triangles], dtype=int)
        rotation, scale, _ = geometry.similarity(
            neutral[mapping] * [.01, -.01, .01], mesh_ops.coordinates(s.template))
        camera, _ = photo_review.camera_for(context, f.generated_head)
        basis = np.diag([.01, -.01, .01]) @ rotation * scale @ np.asarray(camera.matrix_world.inverted())[:3, :3].T
        def projected_area(positions):
            xy = (positions[mapping] @ basis)[:, :2]
            a = xy[triangles[:, 1]] - xy[triangles[:, 0]]
            b = xy[triangles[:, 2]] - xy[triangles[:, 0]]
            return np.maximum(a[:, 0] * b[:, 1] - a[:, 1] * b[:, 0], 0)
        before = projected_area(neutral)
        minimum_area = max(float(before.max(initial=0)) * 1e-9, 1e-16)
        exposed = np.zeros(len(triangles))
        for posed in blink_positions:
            after = projected_area(posed)
            expansion = 1 - before / np.maximum(after, minimum_area)
            exposed = np.maximum(exposed, np.clip((expansion - .25) / .5, 0, 1))
        values = np.zeros(len(mapping))
        np.maximum.at(values, triangles.ravel(), np.repeat(exposed, 3))
        territory = np.zeros(len(neutral))
        for name in ('eyeLeft_lod0_mesh', 'eyeRight_lod0_mesh'):
            eye = np.asarray(meshes[name]['neutral'])
            center = eye.mean(0)
            radius = max(float(np.linalg.norm(eye - center, axis=1).max()), 1e-8)
            distance = np.linalg.norm(neutral - center, axis=1)
            territory = np.maximum(territory, np.clip((2.2 * radius - distance) / (.5 * radius), 0, 1))
        values = values * values * (3 - 2 * values) * territory[mapping]
        uv = np.empty((len(mesh.loops), 2), dtype=np.float32)
        mesh.uv_layers.active.data.foreach_get('uv', uv.ravel())
        loops = np.array([triangle.loops[:] for triangle in mesh.loop_triangles], dtype=int)
        vertices = np.empty(len(mesh.loops), dtype=np.int32)
        mesh.loops.foreach_get('vertex_index', vertices)
        return uv[loops], values[vertices[loops]].astype(np.float32)
    except (OSError, ValueError, KeyError, TypeError, StopIteration):
        # Optional reference data complements the native joint-weight mask.
        return None


def stamp(context):
    from . import photo_fit_jobs
    s, f = context.scene.cm_settings, context.scene.cm_features
    cache = {}
    blink = blink_protection(context, s.result)
    return (photo_fit_jobs.stamp(context),
            map_provenance.image_digest(paint_session.albedo(s.result), cache),
            map_provenance.image_digest(base_image(paint_session.albedo(s.result)), cache),
            map_provenance.image_digest(f.generated_head['cm_photo_reference'], cache),
            hashlib.sha256(skin_feature_protection(context, s.result).astype('<f4').tobytes()).hexdigest(),
            hashlib.sha256(blink[0].tobytes() + blink[1].tobytes()).hexdigest() if blink else None,
            tuple((index, hashlib.sha256(packed_photo(image)).hexdigest())
                  for index, image in supporting_photos(f.generated_head)),
            map_provenance.image_digest(paint_session.albedo(f.body_result), cache)
            if s.part == 'FULL_BODY' and f.body_result else None,
            map_provenance.image_digest(base_image(paint_session.albedo(f.body_result)), cache)
            if s.part == 'FULL_BODY' and f.body_result else None,
            [mat.as_pointer() if mat else None for mat in s.result.data.materials])


def neck_inputs(context, head, body):
    """Keep UV copies paired through the actual welded physical neck rings."""
    from . import workflow, anatomy, mesh_ops
    head_points = mesh_ops.evaluated_coordinates(context, head)
    body_points = mesh_ops.evaluated_coordinates(context, body)
    if len(head_points) != len(head.data.vertices) or len(body_points) != len(body.data.vertices):
        raise ValueError('Blend neck color on the base head/body before subdivision modifiers')
    hi, bi = anatomy.match_neck(head_points, workflow.neck_ring(head), body_points, workflow.neck_ring(body, False))
    seam = head_points[hi]
    if np.linalg.norm(seam - body_points[bi], axis=1).max() > max(np.ptp(head_points, axis=0)) * 1e-4:
        raise ValueError('Attach the head/body neck before blending its color')
    arrays = {'points': head_points, 'seam': seam}
    for role, obj, ring in (('head', head, seam), ('body', body, body_points[bi])):
        points, _, vertices, uv = skin_projection.surface(context, obj)
        nearest = np.argmin(np.sum((points[:, None] - ring[None]) ** 2, axis=2), axis=1)
        selected = np.linalg.norm(points - ring[nearest], axis=1)[vertices] < 1e-5
        arrays[role + '_uv'] = uv[selected]
        arrays[role + '_ids'] = nearest[vertices[selected]]
    return arrays


def start(context, resolution, strength, match_skin=True, blend_neck=True, repair_hair=False):
    from . import photo_fit_jobs
    s, f = context.scene.cm_settings, context.scene.cm_features
    character_project.preflight(context, s, f)
    photo_review.targets(context, 'RESULT')
    source = f.generated_head
    image = source['cm_photo_reference']
    base = base_image(paint_session.albedo(s.result))
    if image.source == 'TILED' or base.source == 'TILED' or base.colorspace_settings.name != 'sRGB':
        raise ValueError('Photo color needs flat sRGB skin and reference images')
    if max(image.size) > 4096:
        raise ValueError('Use the normalized photo from Fit Starting Face from Photo')
    if repair_hair and not (Path(bpy.path.abspath(f.model_dir)) / 'selfie_multiclass_256x256.tflite').is_file():
        raise ValueError('Repair Hair on Skin needs the skin segmentation model; rerun optional model setup')
    from . import resource_budget
    body = bool(match_skin and s.part == 'FULL_BODY' and f.body_result)
    reference_pixels = max(min(ref.size[0] * ref.size[1], 4096 ** 2)
                           for ref in [image] + [im for _, im in supporting_photos(source)])
    resource_budget.require('photo color projection',resource_budget.maps(resolution,(s.result, f.body_result) if body else (s.result,))
                            + (6 * int(resolution) ** 2 * 16 if int(resolution) > 2048 else 0)
                            + (80 * reference_pixels if repair_hair else 0)
                            + (256 * resource_budget.MIB if body and blend_neck else 0)
                            + (384 * resource_budget.MIB if supporting_photos(source) else 0))
    signature = stamp(context)
    camera, _ = photo_review.camera_for(context, source)
    points, loops, vertices, uv = skin_projection.surface(context, s.result)
    matrix = np.asarray(camera.matrix_world.inverted())
    primary_points = points @ matrix[:3, :3].T + matrix[:3, 3]
    supports = supporting_photos(source)
    eyelids = eyelid_protection(context, s.result)
    nostrils = joint_feature_protection(context, s.result, 'nostrilthickness')
    arrays = dict(points=primary_points, triangles=vertices[loops], uv=uv[loops],
                  skin_protection=np.maximum(eyelids, nostrils),
                  eyelid_protection=eyelids, nostril_protection=nostrils)
    blink = blink_protection(context, s.result)
    if blink is not None:
        arrays.update(blink_uv=blink[0], blink_weights=blink[1])
    landmarks = world_landmarks(context, source)
    arrays['view_landmarks'] = landmarks @ matrix[:3, :3].T + matrix[:3, 3]
    parent = Path(bpy.path.abspath(s.project_dir)).resolve() / 'PhotoColors'
    if parent.resolve() != parent:
        raise ValueError('Photo color work must remain inside its project without linked folders')
    folder = parent / uuid.uuid4().hex[:12]
    folder.mkdir(parents=True, exist_ok=False)
    np.savez_compressed(folder / 'surface.npz', **arrays)
    support_names = []
    for index, supporting_image in supports:
        (folder / f'support-{index}.input').write_bytes(packed_photo(supporting_image))
        support_names.append({'slot': index, 'name': supporting_image.name,
                              'generated': bool(supporting_image.get('cm_photo_generated_reference'))})
    skin_library.save_image_copy(image, folder / 'photo.png', opaque=True)
    skin_library.save_image_copy(base, folder / 'fallback.png')
    neck_warning = ''
    if body:
        body_image = base_image(paint_session.albedo(f.body_result))
        if body_image.source == 'TILED' or body_image.colorspace_settings.name != 'sRGB':
            raise ValueError('Tone matching needs a flat sRGB body albedo')
        skin_library.save_image_copy(body_image, folder / 'body.png')
        if blend_neck:
            try:
                np.savez_compressed(folder / 'neck.npz', **neck_inputs(context, s.result, f.body_result))
            except ValueError as exc:
                neck_warning = str(exc)
    request = folder / 'request.json'
    request.write_text(json.dumps({'parent': os.getpid(), 'resolution': int(resolution), 'strength': strength,
                                  'camera_scale': camera.data.ortho_scale, 'match_skin': match_skin, 'body': body,
                                  'repair_hair': bool(repair_hair),
                                  'supporting_photos': support_names,
                                  'blend_neck': bool(body and blend_neck and not neck_warning), 'neck_warning': neck_warning,
                                  'detector': str(Path(bpy.path.abspath(f.model_dir)) / 'face_landmarker.task')}), encoding='utf-8')
    # Establish the disk-sync baseline outside the undoable Apply operator.
    # A manual undo_push nested inside that operator suppresses its final state
    # and leaves no photo-color result for Redo.
    records = [(s.result, '')] + ([(f.body_result, '_Body')] if body else [])
    for obj, suffix in records:
        original = paint_session.albedo(obj)
        target = Path(bpy.path.abspath(s.project_dir)).resolve() / 'Maps' / (s.character_name + suffix + '_ALBEDO.png')
        original.pack()
        original[paint_session.TARGET] = str(target)
        if not original.get(paint_session.REVISION):
            original[paint_session.REVISION] = uuid.uuid4().hex
        paint_session._REVISIONS[str(target)] = original[paint_session.REVISION]
    if not bpy.app.background:
        bpy.ops.ed.undo_push(message='Before Photo Color')
    environment = os.environ.copy()
    environment.update(OPENBLAS_NUM_THREADS='4', OMP_NUM_THREADS='4')
    with (folder / 'worker.log').open('w', encoding='utf-8') as log:
        process = subprocess.Popen([bpy.path.abspath(f.inference_python),
                                    str(Path(__file__).with_name('photo_color_worker.py')), str(request)],
                                   stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                                   creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0), env=environment)
    photo_fit_jobs._JOB = {'kind': 'COLOR', 'scene': context.scene, 'folder': folder,
                          'process': process, 'started': time.monotonic(), 'ready': False, 'stamp': signature,
                          'timeout': (180 + 60 * len(support_names)) * max(1, (int(resolution) / 2048) ** 2)}
    f.photo_fit_status = 'Photo color is projecting locally; the accepted skin is unchanged.'
    bpy.app.timers.register(photo_fit_jobs.poll, first_interval=.2)


def ready(job, report):
    scene = job['scene']
    window = next((w for w in bpy.context.window_manager.windows if w.scene == scene), None)
    if not window:
        raise ValueError('Return to the photo-color character and project again')
    with bpy.context.temp_override(window=window):
        if stamp(bpy.context) != job['stamp']:
            raise ValueError('Character or skin changed during projection; project photo color again')
    names = ['color.png', 'mask.png'] + (['body-color.png'] if report['matched_body'] else [])
    if report.get('clean_reference'):
        names.append('clean.png')
    if any(not (job['folder'] / name).is_file() for name in names):
        raise ValueError('Photo color returned incomplete images')
    job.update(ready=True, report=report)
    supports = report.get('supporting_views', [])
    used = sum(view['status'] == 'accepted' for view in supports)
    scene.cm_features.photo_fit_status = (
        f'Photo color ready. Used {used} of {len(supports)} supporting photos. Apply to inspect the result.'
        if supports else 'Photo color ready. Apply to inspect the face and matched skin tone; Undo retains original skin.')


def prune(job, project):
    folder = job['folder'].resolve()
    if folder.parent != (Path(project).resolve() / 'PhotoColors'):
        raise ValueError('Photo color cleanup must stay in its owned project folder')
    for name in ('surface.npz', 'neck.npz', 'photo.png', 'fallback.png', 'body.png', 'color.png', 'body-color.png', 'clean.png'):
        (folder / name).unlink(missing_ok=True)
    for index in range(SUPPORT_LIMIT):
        (folder / f'support-{index}.input').unlink(missing_ok=True)


def discard(job):
    """Release owned color buffers after the canceled worker has really exited."""
    def finish():
        if job['process'].poll() is None:
            return .2
        try:
            prune(job, job['folder'].parent.parent)
        except (OSError, ValueError) as exc:
            print('Discarded photo color input cleanup:', exc)
        return None
    if finish() is not None:
        bpy.app.timers.register(finish, first_interval=.2)


def apply(context):
    from . import photo_fit_jobs
    job = photo_fit_jobs._JOB
    if not job or job.get('kind') != 'COLOR' or not job['ready'] or job['scene'] != context.scene:
        raise ValueError('Wait for photo color in this character scene')
    s, f = context.scene.cm_settings, context.scene.cm_features
    character_project.preflight(context, s, f, allow_photo_fit=True)
    if stamp(context) != job['stamp']:
        raise ValueError('Character or skin changed; discard this result and project again')
    from . import bake_jobs
    records = [(s.result, 'color.png', s.character_name + '_ALBEDO.png')]
    if job['report']['matched_body']:
        records.append((f.body_result, 'body-color.png', s.character_name + '_Body_ALBEDO.png'))
    destination = Path(bpy.path.abspath(s.project_dir)).resolve() / 'Maps'
    previous = {obj: list(obj.data.materials) for obj, _, _ in records}
    originals = {obj: paint_session.albedo(obj) for obj, _, _ in records}
    for obj, _, name in records:
        original = originals[obj]
        original.pack()
        original[paint_session.TARGET] = str(destination / name)
        if not original.get(paint_session.REVISION):
            original[paint_session.REVISION] = uuid.uuid4().hex
        paint_session._REVISIONS[str(destination / name)] = original[paint_session.REVISION]
    copies, images, bindings = [], [], {}
    try:
        mask = bpy.data.images.load(str(job['folder'] / 'mask.png'), check_existing=False)
        images.append(mask)
        mask.colorspace_settings.name = 'Non-Color'
        mask.pack()
        mask.name = s.character_name + '_PhotoColorMask'
        if tuple(mask.size) != (job['report']['resolution'],) * 2:
            raise ValueError('Photo projection mask dimensions do not match the request')
        clean = None
        if job['report'].get('clean_reference'):
            clean = bpy.data.images.load(str(job['folder'] / 'clean.png'), check_existing=False)
            images.append(clean)
            if tuple(clean.size) != tuple(mask.size):
                raise ValueError('Photo clean reference dimensions do not match the request')
            clean.pack()
            clean.name = s.character_name + '_PhotoCleanReference'
            clean['cm_native_uv_layout'] = skin_projection.uv_layout_signature(s.result)
        for obj, filename, name in records:
            image = bpy.data.images.load(str(job['folder'] / filename), check_existing=False)
            images.append(image)
            if min(image.size) < 1 or max(image.size) > min(4096, job['report']['resolution']):
                raise ValueError('Photo color dimensions exceed the selected limit')
            if obj == s.result and tuple(image.size) != (job['report']['resolution'],) * 2:
                raise ValueError('Photo color dimensions do not match the request')
            image.pack()
            image.name = name.removesuffix('_ALBEDO.png') + '_PhotoColor'
            image['cm_photo_reference'] = f.generated_head['cm_photo_reference']
            # Undo retains the currently accepted material. Repair/projection
            # needs the original clean skin, not another baked photo layer.
            image['cm_photo_previous_albedo'] = base_image(originals[obj])
            if obj == s.result:
                image['cm_photo_projection_mask'] = mask
                image['cm_photo_hair_repair'] = bool(job['report'].get('hair_repair_enabled'))
                if job['report'].get('projection_edge_fade'):
                    image['cm_photo_projection_edge_fade'] = json.dumps(job['report']['projection_edge_fade'])
                if clean is not None:
                    image['cm_photo_clean_reference'] = clean
                    image['cm_photo_clean_uv'] = clean['cm_native_uv_layout']
                if job['report'].get('supporting_views'):
                    image['cm_photo_view_report'] = json.dumps(job['report']['supporting_views'])
            image.filepath_raw = str(destination / name)
            image[paint_session.TARGET] = str(destination / name)
            image[paint_session.REVISION] = uuid.uuid4().hex
            bindings[obj] = []
            for material in previous[obj]:
                if not material:
                    bindings[obj].append(None)
                    continue
                copy = material.copy()
                copies.append(copy)
                for node in copy.node_tree.nodes if copy.use_nodes else ():
                    if node.type == 'TEX_IMAGE' and node.image == originals[obj]:
                        node.image = image
                bindings[obj].append(copy)
        def bind():
            for obj, materials in bindings.items():
                for index, material in enumerate(materials):
                    obj.data.materials[index] = material
        with TemporaryDirectory(prefix='publish-', dir=job['folder']) as directory:
            folder = Path(directory)
            staged = folder / 'staging/Maps'
            staged.mkdir(parents=True)
            files = []
            for _, filename, name in records:
                shutil.copy2(job['folder'] / filename, staged / name)
                files.append({'name': name, 'sha256': bake_jobs.file_hash(staged / name)})
            bake_jobs.deliver({'folder': folder, 'destination': destination}, files, finalize=bind)
    except BaseException:
        for obj, materials in previous.items():
            obj.data.materials.clear()
            for material in materials:
                obj.data.materials.append(material)
        for copy in copies:
            bpy.data.materials.remove(copy)
        for image in reversed(images):
            bpy.data.images.remove(image, do_unlink=True)
        raise
    for image in images:
        if image.get(paint_session.TARGET):
            paint_session._REVISIONS[image[paint_session.TARGET]] = image[paint_session.REVISION]
    photo_fit_jobs._JOB = None
    f.photo_fit_status = 'Photo color applied. Smart Paint can repair transitions. Photo lighting is retained; skin tone matching is approximate.'
    views = job['report'].get('supporting_views', [])
    neck = job['report'].get('neck_blend', {})
    if neck.get('status') == 'complete':
        f.photo_fit_status = 'Photo color applied with the neck matched to the body. Photo lighting remains; inspect the result in 3D.'
    elif job['report'].get('neck_warning'):
        f.photo_fit_status += ' Neck color was not blended: ' + job['report']['neck_warning']
    if views:
        f.photo_fit_status += f" Used {sum(view['status'] == 'accepted' for view in views)} of {len(views)} supporting photos."
    f.generated_head['cm_photo_last_view_report'] = json.dumps(views)
    s.status = f.photo_fit_status
    try:
        prune(job, bpy.path.abspath(s.project_dir))
    except (OSError, ValueError) as exc:
        print('Photo color input cleanup:', exc)
    for area in context.screen.areas if context.screen else ():
        if area.type == 'VIEW_3D':
            area.spaces.active.shading.type = 'MATERIAL'


class CM_OT_add_supporting_photo(bpy.types.Operator):
    bl_idname = 'cm.add_supporting_photo'
    bl_label = 'Add Supporting Photo'
    bl_description = 'Add a neutral view of the same face for texture coverage'
    bl_options = {'REGISTER', 'UNDO'}
    filepath: StringProperty(subtype='FILE_PATH')
    generated: BoolProperty(name='Generated Reference', default=False,
                            description='This supporting view was generated from the primary image')
    filter_glob: StringProperty(default='*.png;*.jpg;*.jpeg;*.webp;*.tif;*.tiff', options={'HIDDEN'})

    def invoke(self, context, event):
        context.window_manager.fileselect_add(self)
        return {'RUNNING_MODAL'}

    def execute(self, context):
        image = None
        try:
            source = context.scene.cm_features.generated_head
            if not source or not source.get('cm_photo_reference'):
                raise ValueError('Fit a primary photo first')
            existing = dict(supporting_photos(source))
            slot = next((index for index in range(SUPPORT_LIMIT) if index not in existing), None)
            if slot is None:
                raise ValueError(f'Use up to {SUPPORT_LIMIT} supporting photos')
            path = Path(bpy.path.abspath(self.filepath))
            if not path.is_file() or path.stat().st_size > 32 * 1024**2:
                raise ValueError('Choose an image file up to 32 MB')
            image = bpy.data.images.load(str(path), check_existing=False)
            if min(image.size) < 1 or image.size[0] * image.size[1] > 32_000_000:
                raise ValueError('Use an image no larger than 32 megapixels')
            image.pack()
            image['cm_photo_generated_reference'] = self.generated
            packed = packed_photo(image)
            candidates = [*existing.values(), source['cm_photo_reference']]
            if any(other.packed_file and packed == bytes(other.packed_file.data) for other in candidates):
                raise ValueError('This supporting photo is already added')
            source[SUPPORT_KEY + str(slot)] = image
        except (OSError, ValueError, RuntimeError) as exc:
            if image is not None:
                bpy.data.images.remove(image)
            self.report({'ERROR'}, str(exc))
            return {'CANCELLED'}
        return {'FINISHED'}


class CM_OT_remove_supporting_photo(bpy.types.Operator):
    bl_idname = 'cm.remove_supporting_photo'
    bl_label = 'Remove Supporting Photo'
    bl_options = {'REGISTER', 'UNDO'}
    slot: IntProperty(min=0, max=SUPPORT_LIMIT - 1)

    def execute(self, context):
        source = context.scene.cm_features.generated_head
        key = SUPPORT_KEY + str(self.slot)
        if not source or not source.get(key):
            return {'CANCELLED'}
        del source[key]
        return {'FINISHED'}


class CM_OT_project_photo_color(bpy.types.Operator):
    bl_idname = 'cm.project_photo_color'
    bl_label = 'Project Photo Color'
    bl_description = ('Project visible face color from the original base skin. Apply replaces prior photo color '
                      'and paint; Undo retains it. Photo lighting remains')
    resolution: EnumProperty(items=[('1024', '1K', ''), ('2048', '2K', ''),
                                    ('4096', '4K', 'Preserve available reference detail; requires more memory')], default='2048')
    strength: FloatProperty(name='Strength', default=1, min=.01, max=1)
    match_skin: BoolProperty(name='Match Existing Skin Tone', default=True)
    blend_neck: BoolProperty(name='Blend Neck to Body', default=True,
                            description='Match the lower neck albedo to the attached body while protecting projected face color')
    repair_hair: BoolProperty(name='Repair Hair on Skin', default=False,
                             description='Infer local skin color beneath detected hair strands using nearby photo skin')

    def execute(self, context):
        try:
            start(context, self.resolution, self.strength, self.match_skin, self.blend_neck, self.repair_hair)
            return {'FINISHED'}
        except Exception as exc:  # noqa: BLE001 - operator boundary.
            self.report({'ERROR'}, str(exc))
            return {'CANCELLED'}


class CM_OT_apply_photo_color(bpy.types.Operator):
    bl_idname = 'cm.apply_photo_color'
    bl_label = 'Apply Photo Color'
    bl_options = {'REGISTER', 'UNDO'}  # noqa: RUF012 - Blender registration metadata.

    def execute(self, context):
        try:
            apply(context)
            return {'FINISHED'}
        except Exception as exc:  # noqa: BLE001 - operator boundary.
            self.report({'ERROR'}, str(exc))
            return {'CANCELLED'}


CLASSES = (CM_OT_add_supporting_photo, CM_OT_remove_supporting_photo,
           CM_OT_project_photo_color, CM_OT_apply_photo_color)
