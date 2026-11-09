"""Local licensed skin cache and owned, cancellable one-time UV preparation."""
from pathlib import Path
import hashlib
import json
import os
import shutil
import subprocess
import uuid
import sys
import bpy
from bpy.app.handlers import persistent

_JOB = None
_ITEMS = {}
EXTRA_MAP_ROLES = {'native_cavity', 'native_roughness', 'body_albedo', 'body_normal', 'body_cavity', 'body_roughness'}


def character_preset_items(f, context):
    result = [('REFERENCE', 'Original Local Skin', 'The original licensed local character skin', 0)]
    try:
        for path in sorted((cache_root(f) / 'presets').glob('*.json')):
            data = json.loads(path.read_text(encoding='utf-8'))
            if not {'native_cavity', 'body_albedo', 'body_normal', 'body_cavity'}.issubset(data.get('extra_maps', {})):
                continue
            if not data.get('native_color_eligible', True):
                continue
            key = data['preset_id']
            result.append((key, data['label'], 'Matching native head and body skin', int(key, 16)))
        packed = f.character_skin_image
        if packed and packed.get('cm_skin_preset') and all(x[0] != packed['cm_skin_preset'] for x in result):
            key = packed['cm_skin_preset']
            result.append((key, packed.get('cm_skin_label', 'Packed Skin'), 'Saved with this project', int(key, 16)))
    except (OSError, ValueError, KeyError, ReferenceError):
        pass
    signature = tuple(result)
    if signature not in _ITEMS:
        _ITEMS[signature] = [tuple(sys.intern(x) if isinstance(x, str) else x for x in item) for item in result]
    return _ITEMS[signature]


def preset_items(f, context):
    """Keep enum strings alive for Blender; stable numeric IDs survive saves."""
    result = [('REFERENCE', 'Local Reference', 'Skin prepared from your licensed local character', 0)]
    try:
        for path in sorted((cache_root(f) / 'presets').glob('*.json')):
            data = json.loads(path.read_text(encoding='utf-8'))
            key = data['preset_id']
            result.append((key, data['label'], 'Prepared local skin', int(key, 16)))
        packed = f.generated_local_skin_image
        if packed and packed.get('cm_skin_preset') and all(x[0] != packed['cm_skin_preset'] for x in result):
            key = packed['cm_skin_preset']
            result.append((key, packed.get('cm_skin_label', 'Packed Skin'), 'Saved with this project', int(key, 16)))
    except (OSError, ValueError, KeyError, ReferenceError):
        pass
    signature = tuple(result)
    if signature not in _ITEMS:
        _ITEMS[signature] = [tuple(sys.intern(x) if isinstance(x, str) else x for x in item) for item in result]
    return _ITEMS[signature]


def preset_changed(f, context):
    packed = f.generated_local_skin_image
    try:
        key = f.generated_skin_preset
        if packed and packed.get('cm_skin_preset', 'REFERENCE') == key:
            return
        image = load_cached_image(cache_root(f), key)
        if not image:
            raise ValueError('Prepare this skin before selecting it')
        f.generated_local_skin_image = image
        from . import head_model
        head_model.update_skin(f, context)
        release_unused(packed)
    except (ValueError, OSError) as exc:
        if packed:
            f.generated_skin_preset = packed.get('cm_skin_preset', 'REFERENCE')
        context.scene.cm_settings.status = str(exc)


def release_unused(image):
    """Switching presets must not accumulate unreferenced 2K pixel buffers."""
    if not image or not image.get('cm_skin_cache_owned') or image.users:
        return
    children = [image.get(key) for key in ('cm_native_skin_reference', 'cm_native_normal_reference',
                'cm_native_cavity_reference', 'cm_native_roughness_reference', 'cm_body_skin_reference')]
    bpy.data.images.remove(image)
    for child in children:
        release_unused(child)


def _write_pointer(root, data, filename):
    destination = root / filename
    destination.parent.mkdir(parents=True, exist_ok=True)
    pending = destination.with_suffix('.tmp')
    pending.write_text(json.dumps(data, indent=2), encoding='utf-8')
    pending.replace(destination)


def running():
    return _JOB is not None


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def cache_root(f):
    return Path(bpy.path.abspath(f.model_dir)).resolve() / 'gnm-skins'


def local_image(f):
    packed = f.generated_local_skin_image
    if packed:
        return packed
    root = cache_root(f)
    image = load_cached_image(root, f.generated_skin_preset)
    if image:
        f.generated_local_skin_image = image
    return image


def load_cached_image(root, preset='REFERENCE'):
    root = Path(root).resolve()
    if preset != 'REFERENCE' and (len(preset) != 7 or any(c not in '0123456789abcdef' for c in preset)):
        raise ValueError('Invalid skin preset')
    pointer = root / ('library.json' if preset == 'REFERENCE' else 'presets/' + preset + '.json')
    if not pointer.is_file():
        return None
    created = []
    try:
        data = json.loads(pointer.read_text(encoding='utf-8'))
        specifications = [('albedo', data['sha256'])]
        if data.get('native_albedo'):
            specifications.append(('native_albedo', data['native_reference']['sha256']))
        if data.get('native_normal_file'):
            if not data.get('native_albedo'):
                raise ValueError('Native normal cache has no matching albedo')
            specifications.append(('native_normal_file', data['native_normal']['sha256']))
        for role, extra in data.get('extra_maps', {}).items():
            if role in EXTRA_MAP_ROLES:
                specifications.append((role, extra['sha256']))
        paths = {}
        for key, checksum in specifications:
            relative = data[key] if key not in data.get('extra_maps', {}) else data['extra_maps'][key]['file']
            path = (root / relative).resolve()
            if not path.is_relative_to(root) or not path.is_file() or digest(path) != checksum:
                raise ValueError('The local skin cache is missing or changed; prepare it again')
            paths[key] = path
        from .head_model import validate_skin_image
        images = {}
        for key, path in paths.items():
            image = bpy.data.images.load(str(path), check_existing=False)
            created.append(image)
            validate_skin_image(image)
            if key in {'native_normal_file', 'native_cavity', 'native_roughness', 'body_normal', 'body_cavity', 'body_roughness'}:
                image.colorspace_settings.name = 'Non-Color'
            image.pack()
            image['cm_skin_cache_owned'] = True
            images[key] = image
        image = images['albedo']
        image['cm_gnm_skin'] = True
        image['cm_gnm_uv_hash'] = data['uv_sha256']
        if 'native_albedo' in images:
            native = images['native_albedo']
            native['cm_native_uv_layout'] = data['native_reference']['uv_layout']
            image['cm_native_skin_reference'] = native
            if 'native_normal_file' in images:
                native['cm_native_normal_reference'] = images['native_normal_file']
            if 'native_cavity' in images:
                native['cm_native_cavity_reference'] = images['native_cavity']
            if 'native_roughness' in images:
                native['cm_native_roughness_reference'] = images['native_roughness']
        if 'body_albedo' in images:
            body = images['body_albedo']
            body['cm_native_uv_layout'] = data['extra_maps']['body_albedo']['uv_layout']
            image['cm_body_skin_reference'] = body
            for kind in ('normal', 'cavity', 'roughness'):
                if 'body_' + kind in images:
                    body['cm_native_' + kind + '_reference'] = images['body_' + kind]
        image['cm_gnm_pixels_hash'] = pixel_digest(image) if data.get('native_color_eligible', True) else ''
        image['cm_native_color_eligible'] = data.get('native_color_eligible', True)
        image['cm_skin_preset'] = data.get('preset_id', 'REFERENCE')
        image['cm_skin_label'] = data.get('label', 'Local Reference')
        if data.get('native_synthesis'):
            image['cm_native_skin_recipe'] = json.dumps(data['native_synthesis'])
        image.name = image['cm_skin_label']
        return image
    except Exception as exc:
        for image in created:
            bpy.data.images.remove(image, do_unlink=True)
        if isinstance(exc, ValueError):
            raise
        raise ValueError('Cannot read the local skin cache; prepare it again') from exc


def start(context, resolution=2048, native=None, folder=None):
    global _JOB
    from . import skin_native_jobs
    if _JOB or skin_native_jobs.running():
        raise ValueError('Local skin preparation is already running')
    from . import guided, setup_paths
    f = context.scene.cm_features
    setup_paths.resolve(f)
    library = guided.character_library()
    if not library:
        raise ValueError('Prepare the licensed local character library first')
    from . import resource_budget
    resource_budget.require('local skin preparation',resource_budget.maps(resolution)+512*resource_budget.MIB)
    root = cache_root(f)
    root.mkdir(parents=True, exist_ok=True)
    if shutil.disk_usage(root).free < 2 * 1024**3:
        raise ValueError('Local skin preparation needs 2 GiB of free disk space')
    folder = folder or root / '.jobs' / uuid.uuid4().hex[:12]
    folder.mkdir(parents=True, exist_ok=True)
    request = {'model_dir': str(Path(bpy.path.abspath(f.model_dir)).resolve()),
               'inference_python': bpy.path.abspath(f.inference_python), 'library': str(library),
               'folder': str(folder), 'resolution': resolution, 'parent_pid': os.getpid()}
    if native:
        request.update(native)
    request_path = folder / 'request.json'
    request_path.write_text(json.dumps(request), encoding='utf-8')
    with (folder / 'worker.log').open('w', encoding='utf-8') as log:
        process = subprocess.Popen([bpy.app.binary_path, '--background', '--factory-startup', '--threads', '4',
            '--python-exit-code', '1', '--python', str(Path(__file__).with_name('skin_worker.py')), '--', str(request_path)],
            stdout=log, stderr=subprocess.STDOUT,
            creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0)
    _JOB = {'process': process, 'scene': context.scene, 'root': root, 'folder': folder, 'cancelled': False,
            'label': native.get('label') if native else None, 'native': bool(native),
            'target': native.get('target', 'GENERATED') if native else 'GENERATED'}
    f.skin_library_status = 'Preparing local skin in the background...'
    if not bpy.app.timers.is_registered(poll):
        bpy.app.timers.register(poll, first_interval=.2, persistent=True)


def pixel_digest(image):
    import numpy as np
    pixels = np.empty(len(image.pixels), dtype=np.float32)
    image.pixels.foreach_get(pixels)
    return hashlib.sha256(pixels.tobytes()).hexdigest()


def save_image_copy(source, path, *, opaque=False):
    """Image.copy() can lose unsaved file-backed pixel edits; copy the pixels."""
    import numpy as np
    from .head_model import validate_skin_image
    validate_skin_image(source)
    copy = bpy.data.images.new('CM_Skin_Save', width=source.size[0], height=source.size[1],
                               alpha=True, float_buffer=source.is_float)
    try:
        copy.colorspace_settings.name = source.colorspace_settings.name
        copy.alpha_mode = source.alpha_mode
        pixels = np.empty(len(source.pixels), dtype=np.float32)
        source.pixels.foreach_get(pixels)
        if opaque:
            pixels.reshape(-1, 4)[:, 3] = 1
        copy.pixels.foreach_set(pixels)
        copy.filepath_raw = str(path)
        copy.file_format = 'PNG'
        copy.save()
    finally:
        bpy.data.images.remove(copy)


def cancel(*_):
    from . import skin_native_jobs
    if skin_native_jobs._JOB:
        skin_native_jobs._JOB['cancelled'] = True
    if _JOB:
        _JOB['cancelled'] = True
        if _JOB['process'].poll() is None:
            _JOB['process'].terminate()


def publish(job):
    folder, root = job['folder'], job['root']
    data = json.loads((folder / 'projection.json').read_text(encoding='utf-8'))
    if data.get('schema_version') != 1 or not data.get('canonical_unchanged'):
        raise ValueError('Local skin output failed validation')
    if data['surface_transfer_fallback']['pixels'] != data['missing_skin_texels']:
        raise ValueError('Local skin still has unresolved projection gaps')
    if any(digest(path) != value for path, value in data['inputs'].items()):
        raise ValueError('Local skin inputs changed; previous cache retained')
    image = Path(data['map']).resolve()
    if not image.is_relative_to(folder.resolve()) or digest(image) != data['sha256']:
        raise ValueError('Local skin output checksum failed')
    from . import head_model
    probe = bpy.data.images.load(str(image), check_existing=False)
    try:
        head_model.validate_skin_image(probe)
        if list(probe.size) != [data['resolution'], data['resolution']]:
            raise ValueError('Local skin resolution differs from the worker report')
    finally:
        bpy.data.images.remove(probe)
    target = root / ('local-' + folder.name + '.png')
    shutil.copy2(image, target)
    reference = Path(data['native_reference']['map']).resolve()
    if not reference.is_relative_to(folder.resolve()) or digest(reference) != data['native_reference']['sha256']:
        raise ValueError('Native reference checksum failed; previous skin retained')
    native_target = root / ('native-' + folder.name + '.png')
    shutil.copy2(reference, native_target)
    pointer = dict(data, albedo=target.name, native_albedo=native_target.name, map=str(target), label='Local MetaHuman Skin')
    normal_reference = None
    if data.get('native_normal'):
        normal_reference = Path(data['native_normal']['map']).resolve()
        if not normal_reference.is_relative_to(folder.resolve()) or digest(normal_reference) != data['native_normal']['sha256']:
            raise ValueError('Native normal checksum failed; previous skin retained')
        normal_target = root / ('normal-' + folder.name + '.png')
        shutil.copy2(normal_reference, normal_target)
        pointer['native_normal_file'] = normal_target.name
    extra_sources = []
    pointer['extra_maps'] = {}
    for role, extra in data.get('extra_maps', {}).items():
        if role not in EXTRA_MAP_ROLES:
            raise ValueError('Unrecognized skin companion')
        source = Path(extra['map']).resolve()
        if not source.is_relative_to(folder.resolve()) or digest(source) != extra['sha256']:
            raise ValueError('Skin companion checksum failed')
        destination = root / (role + '-' + folder.name + '.png')
        shutil.copy2(source, destination)
        pointer['extra_maps'][role] = dict(extra, file=destination.name)
        extra_sources.append(source)
    if job.get('label'):
        pointer['preset_id'] = uuid.uuid4().hex[:7]
        pointer['label'] = job['label']
        _write_pointer(root, pointer, 'presets/' + pointer['preset_id'] + '.json')
    else:
        _write_pointer(root, pointer, 'library.json')
    job['published_preset'] = pointer.get('preset_id', 'REFERENCE')
    # The published cache is independent; retain the small proof/report, not a
    # second full-resolution albedo in each successful worker directory.
    image.unlink()
    reference.unlink()
    if normal_reference:
        normal_reference.unlink()
    for source in extra_sources:
        source.unlink()
    return target


def poll():
    global _JOB
    job = _JOB
    if not job:
        return None
    scene = job['scene']
    code = job['process'].poll()
    if code is None:
        if scene:
            try:
                scene.cm_features.skin_library_status = (job['folder'] / 'progress.txt').read_text(encoding='utf-8')
            except (OSError, ReferenceError):
                pass
        return .2
    (job['folder'] / 'process-exit.json').write_text(json.dumps({'pid': job['process'].pid, 'exit_code': code, 'cancelled': job['cancelled']}))
    try:
        if job['cancelled']:
            message = 'Local skin preparation canceled. Previous skin retained.'
        elif code:
            raise ValueError('Local skin preparation failed. See ' + str(job['folder'] / 'worker.log'))
        else:
            publish(job)
            message = 'Local GNM skin ready.'
            if scene and cache_root(scene.cm_features) == job['root']:
                from . import head_model
                image = load_cached_image(job['root'], job['published_preset'])
                head_model.validate_skin_image(image)
                generated_target = job.get('target', 'GENERATED') == 'GENERATED'
                if generated_target and scene.cm_features.generated_skin == 'LOCAL' and scene.cm_features.generated_head:
                    if image.get('cm_gnm_uv_hash') != head_model.uv_signature(scene.cm_features.generated_head):
                        raise ValueError('Local skin prepared; restore the generated head UVs before applying it')
                previous = scene.cm_features.generated_local_skin_image if generated_target else None
                if generated_target:
                    scene.cm_features.generated_local_skin_image = image
                    scene.cm_features.generated_skin_preset = job['published_preset']
                if image.get('cm_body_skin_reference'):
                    scene.cm_features.character_skin_image = image
                    scene.cm_features.character_skin_preset = job['published_preset']
                if generated_target and scene.cm_features.generated_skin == 'LOCAL' and scene.cm_features.generated_head:
                    head_model.apply_skin(scene.cm_features.generated_head, scene.cm_features)
                if not generated_target:
                    message = 'Character skin ready. Choose Apply to Character to use it.'
                release_unused(previous)
            if job.get('native'):
                # Projection proof names these inputs; keep their checksums in
                # the report, but only retain the published packed companions.
                for name in ('native_albedo.png', 'native_normal.png', 'native_cavity.png', 'native_roughness.png',
                             'body_albedo.png', 'body_normal.png', 'body_cavity.png', 'body_roughness.png'):
                    (job['folder'] / name).unlink(missing_ok=True)
        if scene:
            scene.cm_features.skin_library_status = message
            scene.cm_features.last_action_error = False
            scene.cm_settings.status = message
    except Exception as exc:
        if scene:
            scene.cm_features.skin_library_status = str(exc)
            scene.cm_features.last_action_error = True
            scene.cm_settings.status = str(exc)
    finally:
        _JOB = None
    for window in bpy.context.window_manager.windows:
        for area in window.screen.areas:
            area.tag_redraw()
    return None


@persistent
def detach(*_):
    from . import skin_native_jobs
    skin_native_jobs.detach()
    if _JOB:
        _JOB['scene'] = None


class CM_OT_prepare_local_skin(bpy.types.Operator):
    bl_idname = 'cm.prepare_local_skin'
    bl_label = 'Prepare Local Skin'
    bl_description = 'Reproject your licensed local MetaHuman skin onto GNM UVs in the background'

    def execute(self, context):
        try:
            start(context)
            return {'FINISHED'}
        except Exception as exc:
            self.report({'ERROR'}, str(exc))
            return {'CANCELLED'}


class CM_OT_cancel_local_skin(bpy.types.Operator):
    bl_idname = 'cm.cancel_local_skin'
    bl_label = 'Cancel Skin Preparation'

    def execute(self, context):
        cancel()
        context.scene.cm_features.skin_library_status = 'Cancel requested; waiting for the current operation to finish...'
        return {'FINISHED'}


class CM_OT_keep_skin(bpy.types.Operator):
    bl_idname = 'cm.keep_skin'
    bl_label = 'Keep Skin in Library'
    bl_description = 'Save the current GNM skin and its native companions as a reusable named preset'

    label: bpy.props.StringProperty(name='Name', default='My Skin')

    def invoke(self, context, event):
        return context.window_manager.invoke_props_dialog(self)

    def execute(self, context):
        try:
            if running():
                raise ValueError('Wait for skin preparation to finish')
            f = context.scene.cm_features
            from . import head_model
            image = local_image(f) if f.generated_skin == 'LOCAL' else f.generated_skin_image
            if not image or f.generated_skin == 'COLOR':
                raise ValueError('Choose a textured skin first')
            head_model.validate_skin_image(image)
            if not f.generated_head:
                raise ValueError('Generate a head before saving its skin')
            root = cache_root(f)
            root.mkdir(parents=True, exist_ok=True)
            key = uuid.uuid4().hex[:7]
            data = {'preset_id': key, 'label': self.label.strip() or 'My Skin',
                    'uv_sha256': head_model.uv_signature(f.generated_head), 'extra_maps': {}}
            if image.get('cm_native_skin_recipe'):
                data['native_synthesis'] = json.loads(image['cm_native_skin_recipe'])
            native = image.get('cm_native_skin_reference')
            companions = [('albedo', image)]
            if native:
                data['native_color_eligible'] = (not image.is_dirty and
                    image.get('cm_gnm_pixels_hash') == pixel_digest(image))
                companions.append(('native_albedo', native))
                if native.get('cm_native_normal_reference'):
                    companions.append(('native_normal_file', native['cm_native_normal_reference']))
                if native.get('cm_native_cavity_reference'):
                    companions.append(('native_cavity', native['cm_native_cavity_reference']))
                if native.get('cm_native_roughness_reference'):
                    companions.append(('native_roughness', native['cm_native_roughness_reference']))
            body = image.get('cm_body_skin_reference')
            if body:
                companions.append(('body_albedo', body))
                for kind in ('normal', 'cavity', 'roughness'):
                    if body.get('cm_native_' + kind + '_reference'):
                        companions.append(('body_' + kind, body['cm_native_' + kind + '_reference']))
            for role, source in companions:
                path = root / (key + '-' + role + '.png')
                save_image_copy(source, path)
                if role not in EXTRA_MAP_ROLES:
                    data[role] = path.name
                if role == 'albedo':
                    data['sha256'] = digest(path)
                elif role == 'native_albedo':
                    data['native_reference'] = {'sha256': digest(path), 'uv_layout': native['cm_native_uv_layout']}
                elif role == 'native_normal_file':
                    data['native_normal'] = {'sha256': digest(path)}
                else:
                    data['extra_maps'][role] = {'file': path.name, 'sha256': digest(path)}
                    if role == 'body_albedo':
                        data['extra_maps'][role]['uv_layout'] = body['cm_native_uv_layout']
            _write_pointer(root, data, 'presets/' + key + '.json')
            previous = f.generated_local_skin_image
            f.generated_local_skin_image = load_cached_image(root, key)
            f.generated_skin_preset = key
            f.generated_skin = 'LOCAL'
            release_unused(previous)
            context.scene.cm_settings.status = 'Skin saved: ' + data['label']
            return {'FINISHED'}
        except Exception as exc:
            self.report({'ERROR'}, str(exc))
            return {'CANCELLED'}


from .skin_native_jobs import CLASSES as NATIVE_CLASSES

CLASSES = (CM_OT_prepare_local_skin, CM_OT_cancel_local_skin, CM_OT_keep_skin, *NATIVE_CLASSES)


def register():
    if detach not in bpy.app.handlers.load_pre:
        bpy.app.handlers.load_pre.append(detach)


def unregister():
    cancel()
    detach()
    if detach in bpy.app.handlers.load_pre:
        bpy.app.handlers.load_pre.remove(detach)
