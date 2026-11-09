"""Keep generated expression color maps tied to the accepted skin albedo."""
import json

from . import map_provenance


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
        map_provenance.record(scene, {slot: map_provenance.signature(scene, slot)},
            recipe='BLOODFLOW_ALBEDO', outputs={name}, details={name: {'dependencies': dependencies}})
