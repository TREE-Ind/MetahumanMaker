"""Owned photo-color work through the existing photo job lifecycle."""

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
from bpy.props import BoolProperty, EnumProperty, FloatProperty

from . import (
    character_project,
    map_provenance,
    paint_session,
    photo_review,
    skin_library,
    skin_projection,
)


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


def stamp(context):
    from . import photo_fit_jobs
    s, f = context.scene.cm_settings, context.scene.cm_features
    cache = {}
    return (photo_fit_jobs.stamp(context),
            map_provenance.image_digest(paint_session.albedo(s.result), cache),
            map_provenance.image_digest(base_image(paint_session.albedo(s.result)), cache),
            map_provenance.image_digest(f.generated_head['cm_photo_reference'], cache),
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


def start(context, resolution, strength, match_skin=True, blend_neck=True):
    from . import photo_fit_jobs
    s, f = context.scene.cm_settings, context.scene.cm_features
    character_project.preflight(context, s, f)
    photo_review.targets(context, 'RESULT')
    source = f.generated_head
    image = source['cm_photo_reference']
    base = base_image(paint_session.albedo(s.result))
    if image.source == 'TILED' or base.source == 'TILED' or base.colorspace_settings.name != 'sRGB':
        raise ValueError('Photo color needs flat sRGB skin and reference images')
    if max(image.size) > 2048:
        raise ValueError('Use the normalized photo from Fit Starting Face from Photo')
    from . import resource_budget
    body = bool(match_skin and s.part == 'FULL_BODY' and f.body_result)
    resource_budget.require('photo color projection',resource_budget.maps(resolution,(s.result, f.body_result) if body else (s.result,))
                            + (256 * resource_budget.MIB if body and blend_neck else 0))
    signature = stamp(context)
    camera, _ = photo_review.camera_for(context, source)
    points, loops, vertices, uv = skin_projection.surface(context, s.result)
    matrix = np.asarray(camera.matrix_world.inverted())
    points = points @ matrix[:3, :3].T + matrix[:3, 3]
    parent = Path(bpy.path.abspath(s.project_dir)).resolve() / 'PhotoColors'
    if parent.resolve() != parent:
        raise ValueError('Photo color work must remain inside its project without linked folders')
    folder = parent / uuid.uuid4().hex[:12]
    folder.mkdir(parents=True, exist_ok=False)
    np.savez_compressed(folder / 'surface.npz', points=points, triangles=vertices[loops], uv=uv[loops])
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
                          'process': process, 'started': time.monotonic(), 'ready': False, 'stamp': signature}
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
    if any(not (job['folder'] / name).is_file() for name in names):
        raise ValueError('Photo color returned incomplete images')
    job.update(ready=True, report=report)
    scene.cm_features.photo_fit_status = 'Photo color ready. Apply to inspect the face and matched skin tone; Undo retains original skin.'


def prune(job, project):
    folder = job['folder'].resolve()
    if folder.parent != (Path(project).resolve() / 'PhotoColors'):
        raise ValueError('Photo color cleanup must stay in its owned project folder')
    for name in ('surface.npz', 'neck.npz', 'photo.png', 'fallback.png', 'body.png', 'color.png', 'body-color.png'):
        (folder / name).unlink(missing_ok=True)


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
        for obj, filename, name in records:
            image = bpy.data.images.load(str(job['folder'] / filename), check_existing=False)
            images.append(image)
            if min(image.size) < 1 or max(image.size) > 2048:
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
    neck = job['report'].get('neck_blend', {})
    if neck.get('status') == 'complete':
        f.photo_fit_status = 'Photo color applied with the neck matched to the body. Photo lighting remains; inspect the result in 3D.'
    elif job['report'].get('neck_warning'):
        f.photo_fit_status += ' Neck color was not blended: ' + job['report']['neck_warning']
    s.status = f.photo_fit_status
    try:
        prune(job, bpy.path.abspath(s.project_dir))
    except (OSError, ValueError) as exc:
        print('Photo color input cleanup:', exc)
    for area in context.screen.areas if context.screen else ():
        if area.type == 'VIEW_3D':
            area.spaces.active.shading.type = 'MATERIAL'


class CM_OT_project_photo_color(bpy.types.Operator):
    bl_idname = 'cm.project_photo_color'
    bl_label = 'Project Photo Color'
    bl_description = ('Project visible face color from the original base skin. Apply replaces prior photo color '
                      'and paint; Undo retains it. Photo lighting remains')
    resolution: EnumProperty(items=[('1024', '1K', ''), ('2048', '2K', '')], default='2048')
    strength: FloatProperty(name='Strength', default=1, min=.01, max=1)
    match_skin: BoolProperty(name='Match Existing Skin Tone', default=True)
    blend_neck: BoolProperty(name='Blend Neck to Body', default=True,
                            description='Match the lower neck albedo to the attached body while protecting projected face color')

    def execute(self, context):
        try:
            start(context, self.resolution, self.strength, self.match_skin, self.blend_neck)
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


CLASSES = (CM_OT_project_photo_color, CM_OT_apply_photo_color)
