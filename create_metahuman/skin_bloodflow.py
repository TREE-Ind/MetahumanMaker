"""Keep generated expression color maps tied to the accepted skin albedo."""
import json
import hashlib
from pathlib import Path

from . import map_provenance


def native_reference(scene,slot):
    """Read an explicitly retained native expression response for this face."""
    root=map_provenance.folder(scene).parent.resolve()/'ReferenceMaps'
    manifest=root/'native-expression-color.json'
    if not manifest.is_file():return None
    value=json.loads(manifest.read_text(encoding='utf-8'))
    if (value.get('schema')!=1 or value.get('encoding')!='UE_ANIMATED_DELTA_V1' or
            value.get('character')!=scene.cm_features.unreal_character):
        raise ValueError('The retained expression-color reference belongs to another native character')
    from . import skin_projection
    if value.get('uv_layout')!=skin_projection.uv_layout_signature(scene.cm_settings.result):
        raise ValueError('The retained expression-color reference has another face UV layout')
    record=value.get('maps',{}).get(slot)
    if not record:raise ValueError('The retained native expression set is incomplete: '+slot)
    path=(root/record['file']).resolve()
    if not path.is_relative_to(root.resolve()) or not path.is_file():raise ValueError('The native expression reference left this character project')
    with path.open('rb') as stream:sha=hashlib.file_digest(stream,'sha256').hexdigest()
    if sha!=record['sha256']:raise ValueError('The retained native expression response changed: '+slot)
    return dict(record,path=str(path))


def plan(scene, role):
    if role != 'HEAD':
        return []
    f, s = scene.cm_features, scene.cm_settings
    entries = map_provenance.load(scene)['entries']
    result = []
    for slot in ('CM1', 'CM2', 'CM3'):
        path = map_provenance.folder(scene)/(s.character_name+'_'+slot+'.png')
        entry = entries.get(map_provenance.identity(scene, slot), {})
        artifact = entry.get('artifacts', {}).get(path.name, {})
        generated = artifact.get('recipe') == 'BLOODFLOW_ALBEDO' or artifact.get('origin') == 'BAKE'
        if not f.bloodflow and not (path.is_file() and generated):
            continue
        if path.exists() and not generated:
            raise ValueError(slot+' is an authored or untracked map. Retain it or remove its override before generating blood flow')
        if path.is_file():
            from .skin_reconstruction import _hash
            if _hash(path)!=artifact.get('sha256'):
                raise ValueError(slot+' was edited after baking. Retain its paint before generating blood flow')
        from . import wrinkles
        wrinkles.ensure_mask(s, f, 'WM'+slot[-1])
        result.append(slot)
    return result


def stamp(scene, slots, cache):
    from .skin_reconstruction import _hash
    return {slot: {'inputs': map_provenance.signature(scene, slot, cache),
                   'output': _hash(map_provenance.paths(scene, slot)[0])
                   if map_provenance.paths(scene, slot)[0].is_file() else None}
            for slot in slots}


def record(scene, slots, previous=None):
    """Restore only affected entries; unrelated bake history stays intact."""
    if previous is not None:
        value = map_provenance.load(scene)
        for slot, entry in previous.items():
            key = map_provenance.identity(scene, slot)
            if entry is None:
                value['entries'].pop(key, None)
            else:
                value['entries'][key] = entry
        path = map_provenance.manifest(scene)
        temporary = path.with_suffix('.tmp')
        temporary.write_text(json.dumps(value, indent=2), encoding='utf-8')
        temporary.replace(path)
        return
    if not slots:return
    from .skin_reconstruction import _hash
    albedo = scene.cm_settings.character_name+'_ALBEDO.png'
    dependencies = {albedo: _hash(map_provenance.folder(scene)/albedo)}
    for slot in slots:
        name = scene.cm_settings.character_name+'_'+slot+'.png'
        reference=native_reference(scene,slot)
        details={'dependencies':dependencies}
        if reference:details.update(native_expression_sha256=reference['sha256'],native_expression_encoding='UE_ANIMATED_DELTA_V1')
        map_provenance.record(scene, {slot: map_provenance.signature(scene, slot)},
            recipe='BLOODFLOW_ALBEDO', outputs={name}, details={name:details})
