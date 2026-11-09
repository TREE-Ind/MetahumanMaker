"""Preserve editable wrinkle poses, masks and their local expression reference."""
import hashlib
import json
import shutil
from pathlib import Path
from types import SimpleNamespace

import bpy
import numpy as np

from . import expressions, sculpt_archive, wrinkles
from . import mesh_ops as m


def inspect(context, scene, part, project_file):
    result = {'layers': {}, 'proxies': {}, 'masks': {}, 'current': set(), 'reference': None}
    if part != 'HEAD':
        return result
    s, f = scene.cm_settings, scene.cm_features
    table = sculpt_archive.returned(scene)
    if any(entry.get('role') in {'WM1', 'WM2', 'WM3'} and not entry.get('adopted') for entry in table.values()):
        raise ValueError('Adopt returned wrinkle sculpts in their saved project and save again before reusing this head')
    layers = {slot: getattr(f, slot.lower()) for slot in ('WM1', 'WM2', 'WM3') if getattr(f, slot.lower())}
    if not layers:
        return result
    wrinkles.refresh_source_state(context, s, f)
    for slot, layer in layers.items():
        m.check_mesh(layer, 'Saved ' + slot)
        keys = layer.data.shape_keys
        if not keys or not keys.use_relative:
            raise ValueError('Saved wrinkles need relative sculpt keys before head reuse')
        if layer.get('cm_wrinkle_update_pending') and not layer.get('cm_wrinkle_source_pending'):
            raise ValueError('Finish the saved wrinkle update before reusing this head')
        if layer.get('cm_wrinkle_native'):
            for name in ('.cm_wm_neutral', '.cm_wm_posed', '.cm_wm_linear_0', '.cm_wm_linear_1', '.cm_wm_linear_2'):
                if not np.isfinite(wrinkles._read_vector(layer, name)).all():
                    raise ValueError('Saved wrinkle inverse basis contains invalid coordinates')
        if not layer.get('cm_wrinkle_source_pending'):
            result['current'].add(layer)
        children = [child for child in layer.children_recursive if child.get('cm_wrinkle_owner') == layer]
        for obj in (layer, *children):
            m.check_mesh(obj, 'Saved wrinkle or anatomy proxy')
            if obj.modifiers or obj.constraints or obj.animation_data or (obj.data.shape_keys and obj.data.shape_keys.animation_data):
                raise ValueError('Apply live wrinkle dependencies in the saved project before head reuse')
            for name, value in obj.items():
                if isinstance(value, bpy.types.Object) and value not in {layer, s.result, s.detail}:
                    raise ValueError('Saved wrinkle has an external object dependency: ' + name)
        result['proxies'][layer] = children
        mask = getattr(f, slot.lower() + '_mask')
        if mask:
            result['masks'][slot] = mask
    if any(layer.get('cm_wrinkle_native') for layer in layers.values()):
        start = str(project_file.parent)
        model_dir = Path(bpy.path.abspath(f.model_dir, start=start)).resolve()
        settings = SimpleNamespace(model_dir=str(model_dir), expression_reference=(
            bpy.path.abspath(f.expression_reference, start=start) if f.expression_reference else ''))
        reference = expressions.reference_path(settings).resolve()
        data = expressions.bank(settings)
        names = {pose['name'] for pose in data['poses']}
        if any(layer.get('cm_wrinkle_native') and layer.get('cm_wrinkle_pose') not in names for layer in layers.values()):
            raise ValueError('The saved expression reference does not contain a retained wrinkle pose')
        shared = reference.is_relative_to(model_dir)
        result['reference'] = {'path': reference, 'shared': shared}
    result['layers'] = layers
    return result


def objects(state):
    return [*state['layers'].values(), *(obj for children in state['proxies'].values() for obj in children)]


def reparent(state):
    for layer, children in state['proxies'].items():
        for child in children:
            world = child.matrix_world.copy()
            child.parent = layer
            child.matrix_world = world
    for obj in objects(state):
        obj.hide_set(True)
        obj.hide_render = True


def binary_files(value):
    if isinstance(value, dict):
        for key, item in value.items():
            if key.endswith('_file'):
                if not isinstance(item, str):
                    raise ValueError('Expression binary reference must be a filename')
                path = Path(item)
                if not item or path.is_absolute() or path.drive or '..' in path.parts or '\\' in item or ':' in item:
                    raise ValueError('Expression binary reference escapes its library')
                yield path
            else:
                yield from binary_files(item)
    elif isinstance(value, list):
        for item in value:
            yield from binary_files(item)


def resources(stage, project, character, state):
    pending = {}
    reference = state['reference']
    if reference and not reference['shared']:
        source = reference['path']
        files = {Path(source.name), *binary_files(json.loads(source.read_text(encoding='utf-8')))}
        hashes = {}
        for relative in sorted(files):
            path = source.parent / relative
            if not path.is_file() or not path.resolve().is_relative_to(source.parent):
                raise ValueError('Expression binary is missing or linked outside its library: ' + str(relative))
            hashes[str(relative)] = hashlib.sha256(path.read_bytes()).hexdigest()
        identity = hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest()
        folder = Path('Expressions') / identity
        for relative in files:
            destination = folder / relative
            existing = project / destination
            if existing.resolve() != existing.absolute():
                raise ValueError('Expression data must remain inside this project without linked paths')
            if existing.is_file() and not existing.is_symlink() and hashlib.sha256(existing.read_bytes()).hexdigest() == hashes[str(relative)]:
                continue
            staged = stage / destination
            staged.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source.parent / relative, staged)
            pending[destination] = staged
        reference['owned_path'] = project / folder / source.name
    for slot, original in list(state['masks'].items()):
        from .textures import copy_image_pixels
        image = copy_image_pixels(original)
        relative = Path('ReferenceMaps/NativeWrinkles') / (character + '_' + slot + '_Region.png')
        staged = stage / relative
        staged.parent.mkdir(parents=True, exist_ok=True)
        image.filepath_raw = str(staged)
        image.file_format = 'PNG'
        image.save()
        image.pack()
        state['masks'][slot] = image
        pending[relative] = staged
    return pending


def assign(context, state):
    s, f = context.scene.cm_settings, context.scene.cm_features
    current = wrinkles.source_stamp(context, s) if state['current'] else None
    for slot, layer in state['layers'].items():
        setattr(f, slot.lower(), layer)
        setattr(f, slot.lower() + '_mask', state['masks'].get(slot))
        if layer in state['current']:
            layer['cm_wrinkle_source_stamp'] = current
        else:
            layer['cm_wrinkle_update_pending'] = s.result
            layer['cm_wrinkle_source_pending'] = True
    if state['reference']:
        reference = state['reference']
        f.expression_reference = str(reference.get('owned_path', reference['path']))
        f.expression_index = 0
        f.expression_previous = ''
        f.expression_prepared = ''
        expressions.sync_picker(f)
    if state['layers']:
        f.show_wrinkles = True
        if f.wrinkle_slot not in state['layers']:
            f.wrinkle_slot = next(iter(state['layers']))
