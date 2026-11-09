"""Dependency-free, checksum-bound sculpt carriers for author-session recovery.

Carriers preserve editable geometry, not evaluated modifier results. Callers must
explicitly resolve unsupported live dependencies before publishing a carrier.
Loading never changes the character or links objects into its collections.
"""

from contextlib import contextmanager, ExitStack
import hashlib
import json
from pathlib import Path
import uuid

import bpy
import numpy as np
from mathutils import Matrix


SCHEMA = 1
ROLES = ("HEAD_DETAIL", "BODY_DETAIL", "WM1", "WM2", "WM3")
RETURNED = "cm_returned_sculpt_archives"
FINGERPRINT_SCHEMA = 1
BODY_CHECKPOINT = 'cm_body_returned_sculpt_recovery'


def roles_for_files(character, names):
    """Map each returned artifact to its actual sculpt-source role."""
    result = {}
    for name in names:
        suffix = name.removeprefix(character + "_")
        if not name.startswith(character + "_"):
            raise ValueError("Returned map belongs to another character")
        if suffix in {"NORMAL.png", "CAVITY.png"}:
            role = "HEAD_DETAIL"
        elif suffix in {"Body_NORMAL.png", "Body_CAVITY.png"}:
            role = "BODY_DETAIL"
        elif suffix in {prefix + str(slot) + ".png" for prefix in ("WM", "CM") for slot in (1, 2, 3)}:
            role = "WM" + suffix[2]
        else:
            raise ValueError("Unsupported returned map: " + name)
        result[name] = role
    return result


def returned(scene):
    value = json.loads(scene.get(RETURNED, '{}'))
    if not isinstance(value, dict):
        raise ValueError("Invalid returned sculpt archive table")
    return value


def record_return(scene, path, receipt, files, manifest):
    """Persist per-artifact authority; a partial return retains other artifacts."""
    roles = roles_for_files(scene.cm_settings.character_name, [item['name'] for item in files])
    value = returned(scene)
    sequence = max((entry.get('sequence', 0) for entry in value.values()), default=0) + 1
    baseline = manifest.get('main_sculpt_layers') or fingerprints(scene)
    for item in files:
        value[item['name']] = {
            'role': roles[item['name']], 'map_sha256': item['sha256'],
            'accepted_geometry': manifest['accepted_geometry'],
            'main_sculpt_sources': manifest['main_sculpt_sources'],
            'session': manifest['session'],
            'sequence': sequence, 'main_layer_stamp': baseline[roles[item['name']]],
            'fingerprint_schema': FINGERPRINT_SCHEMA,
        }
        if receipt is None:
            value[item['name']]['source_unavailable'] = True
        else:
            value[item['name']].update(path=str(Path(path).resolve()), receipt=receipt)
    scene[RETURNED] = json.dumps(value, sort_keys=True)


def layers(scene):
    s, f = scene.cm_settings, scene.cm_features
    return dict(zip(ROLES, (s.detail, f.body_detail, f.wm1, f.wm2, f.wm3)))


def fingerprints(scene):
    """Guard dormant artist keys as well as the currently evaluated surface."""
    from . import map_provenance, sculpt_modifiers
    result = {}
    with bpy.context.temp_override(scene=scene, view_layer=scene.view_layers[0]):
        for role, obj in layers(scene).items():
            if not obj:
                result[role] = None
                continue
            digest = hashlib.sha256(map_provenance.mesh_digest(bpy.context, obj).encode())
            digest.update(sculpt_modifiers.settings(obj).encode())
            digest.update(sculpt_modifiers.multires_digest(bpy.context, obj))
            digest.update(sculpt_modifiers.dependencies_digest(bpy.context,obj))
            values = np.empty(len(obj.data.vertices) * 3, dtype='<f4')
            obj.data.vertices.foreach_get('co', values)
            digest.update(values.tobytes())
            for key in obj.data.shape_keys.key_blocks if obj.data.shape_keys else ():
                digest.update(json.dumps([key.name, key.value, key.mute, key.relative_key.name,
                    key.slider_min, key.slider_max, key.interpolation, key.vertex_group]).encode())
                key.data.foreach_get('co', values)
                digest.update(values.tobytes())
            digest.update(json.dumps([obj.show_only_shape_key,
                obj.active_shape_key_index if obj.show_only_shape_key else None,
                [(group.name, group.lock_weight) for group in obj.vertex_groups]]).encode())
            for vertex in obj.data.vertices:
                digest.update(json.dumps([(weight.group, weight.weight) for weight in vertex.groups]).encode())
            reference = obj.data.attributes.get('.cm_detail_reference')
            if reference:
                reference.data.foreach_get('vector', values)
                digest.update(values.tobytes())
            for name, value in sorted(obj.items()):
                if not name.startswith(('cm_detail_', 'cm_wrinkle_')):
                    continue
                if name in {'cm_wrinkle_source_stamp','cm_wrinkle_source_pending','cm_wrinkle_update_pending'}:
                    continue  # Derived validation state, not an artist input.
                try:digest.update(json.dumps([name, _plain(value)], sort_keys=True).encode())
                except (ValueError, TypeError):pass
            result[role] = digest.hexdigest()
    return result


def _assign(scene, role, obj):
    if role == 'HEAD_DETAIL':scene.cm_settings.detail = obj
    else:setattr(scene.cm_features, 'body_detail' if role == 'BODY_DETAIL' else role.lower(), obj)


def require_legacy_sources(scene, roles, table):
    """Do not replace pre-archive author work with Main's older sculpt layers."""
    from . import map_provenance
    records=map_provenance.load(scene)['entries']
    parts=('HEAD','BODY') if scene.cm_settings.part=='FULL_BODY' else (scene.cm_settings.part,)
    keys=['FINAL_'+part for part in parts]
    if scene.cm_settings.part!='BODY':keys+=['WM1','WM2','WM3','CM1','CM2','CM3']
    for key in keys:
        entry=records.get(map_provenance.identity(scene,key),{})
        for path in map_provenance.paths(scene,key):
            if path.name in table:continue
            artifact=entry.get('artifacts',{}).get(path.name)
            if artifact is None and path.name in entry.get('files',{}):
                artifact={'origin':entry.get('origin'),'sha256':entry['files'][path.name]}
            if not artifact or artifact.get('origin',entry.get('origin'))!='SCULPT_SESSION':continue
            role=roles_for_files(scene.cm_settings.character_name,[path.name])[path.name]
            if role in roles and path.is_file() and hashlib.sha256(path.read_bytes()).hexdigest()==artifact['sha256']:
                raise ValueError('These maps came from an older sculpt session without a frozen source. Save and close its author window, then Resume Session and Save & Return with the current add-on before changing this character.')


def checkpoint_body(scene, staged, previous, visibility, table, before_ids):
    """Persist recovery ownership across a body preview, including Save/reopen."""
    if not staged:return
    stamps=fingerprints(scene)
    visible={}
    for role in staged:
        obj=previous[role]
        objects=[obj,*[child for child in obj.children if child.get('cm_wrinkle_owner')==obj]] if obj else []
        visible[role]={str(index):{'object':item,'hidden':visibility[item][0],'render':visibility[item][1]}
            for index,item in enumerate(objects)}
    scene[BODY_CHECKPOINT]={
        'roles':json.dumps(list(staged)), 'archives':table or '{}',
        'old':{role:previous[role] for role in staged if previous[role]},
        'new':{role:obj for role,obj in staged.items() if obj},
        'stamps':json.dumps({role:stamps[role] for role in staged}),
        'visibility':visible,
        'owned':{str(index):data for index,data in enumerate(set(bpy.data.user_map())-before_ids)},
    }


def finish_body_recovery(context, apply):
    """Commit recovered sources, or undo untouched adoption when body is canceled."""
    scene=context.scene
    state=scene.get(BODY_CHECKPOINT)
    if state is None:return
    if apply:
        del scene[BODY_CHECKPOINT]
        return
    roles=json.loads(state['roles'])
    previous={role:state['old'].get(role) for role in roles}
    expected={role:state['new'].get(role) for role in roles}
    stamps=json.loads(state['stamps'])
    old_table=json.loads(state['archives'])
    owned=set(state['owned'].values())
    visible={role:[(item['object'],item['hidden'],item['render']) for item in state['visibility'][role].values()] for role in roles}
    current=layers(scene)
    current_stamps=fingerprints(scene)
    restore={role for role in roles if current[role]==expected[role] and current_stamps[role]==stamps[role]}
    # Delete the IDProperty owner only after copying its references. Changed
    # sculpt work remains active, with its consumed receipt and owned assets.
    del scene[BODY_CHECKPOINT]
    for role in restore:
        _assign(scene,role,previous[role])
        for obj,hidden,render in visible[role]:
            obj.hide_set(hidden);obj.hide_render=render
        obj=expected[role]
        if obj:
            obj.hide_set(True);obj.hide_render=True
            for child in obj.children:
                if child.get('cm_wrinkle_owner')==obj:
                    child.hide_set(True);child.hide_render=True
    table={name:entry for name,entry in returned(scene).items() if entry['role'] not in restore}
    table.update({name:entry for name,entry in old_table.items() if entry['role'] in restore})
    scene[RETURNED]=json.dumps(table,sort_keys=True)
    if len(restore)==len(roles):bpy.data.batch_remove(ids=owned)
    context.view_layer.update()


@contextmanager
def recover(context, roles=ROLES, *, rebind_maps=True):
    """Stage returned layers for an edit, restoring pointers/receipts on failure.

    Accepted meshes are never replaced here. Old sculpt objects stay available
    in the project, hidden after a successful adoption. Workers remain responsible
    for transporting the recovered sculpt to a changed accepted surface.
    """
    from . import map_provenance, bake_jobs, detail_jobs, wrinkle_jobs
    scene = context.scene
    table = returned(scene)
    if scene.cm_features.sculpt_session_role!='AUTHOR':require_legacy_sources(scene,roles,table)
    latest = {}
    for name, entry in table.items():
        role = entry['role']
        if role in roles and (role not in latest or entry.get('sequence', 0) >= latest[role][1].get('sequence', 0)):
            latest[role] = (name, entry)
    pending = {role: pair for role, pair in latest.items() if not pair[1].get('adopted')}
    if not pending or scene.cm_features.sculpt_session_role == 'AUTHOR':
        yield {}
        return
    if any(entry.get('source_unavailable') for _, entry in pending.values()):
        raise ValueError('The latest returned maps have no frozen sculpt source. Save and close the author window, then Resume Session and Save & Return with the current add-on before editing this character.')
    f = scene.cm_features
    if context.mode != 'OBJECT' or f.body_editing or f.full_sculpt or f.restore_editing or f.expression_active:
        raise ValueError('Finish the active edit before recovering returned sculpt layers')
    if any(module.running() for module in (bake_jobs, detail_jobs, wrinkle_jobs)):
        raise ValueError('Finish or cancel the active worker before recovering returned sculpt layers')
    accepted = map_provenance.accepted_signature(scene)
    current = fingerprints(scene)
    folder = Path(bpy.path.abspath(scene.cm_settings.project_dir)) / 'Maps'
    for role, (name, entry) in pending.items():
        if entry['accepted_geometry'] != accepted:
            raise ValueError('Accepted geometry changed before returned sculpt recovery; keep its archive for reprojection')
        if entry.get('fingerprint_schema') != FINGERPRINT_SCHEMA:
            raise ValueError('Returned sculpt recovery uses an unsupported source fingerprint version; keep its archive for explicit recovery')
        if 'main_layer_stamp' not in entry or entry['main_layer_stamp'] != current[role]:
            raise ValueError('Main sculpt layer changed before recovery: ' + role + '. Returned work is retained in its archive.')
        path = folder / name
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != entry['map_sha256']:
            raise ValueError('Returned map changed before sculpt recovery: ' + name)
    previous = layers(scene)
    old_objects = {obj for obj in previous.values() if obj}
    old_objects |= {child for obj in tuple(old_objects) for child in obj.children if child.get('cm_wrinkle_owner') == obj}
    visibility = {obj: (obj.hide_get(), obj.hide_render) for obj in old_objects}
    previous_table = scene.get(RETURNED)
    provenance_path = map_provenance.manifest(scene)
    provenance = provenance_path.read_bytes() if provenance_path.exists() else None
    invalidated = scene.get(map_provenance.INVALIDATED)
    owned = set()
    staged = {}
    from .sculpt_dependencies import Capture
    capture = Capture(context)
    try:
        with ExitStack() as stack:
            loaded = {}
            for role, (_, entry) in pending.items():
                identity = (entry['path'], entry['receipt']['sha256'])
                if identity not in loaded:
                    loaded[identity] = stack.enter_context(read(entry['path'], entry['receipt']))
                source = loaded[identity].get(role)
                if source is None:
                    staged[role] = None
                    continue
                obj = source.copy()
                owned.add(obj)
                obj.data = source.data.copy()
                owned.add(obj.data)
                if obj.data.shape_keys:owned.add(obj.data.shape_keys)
                obj.use_fake_user = False
                obj.name = scene.cm_settings.character_name + '_Returned_' + role
                obj.matrix_world = source.matrix_world.copy()
                scene.collection.objects.link(obj)
                capture.modifiers(obj)
                old = previous[role]
                accepted_obj = scene.cm_features.body_result if role == 'BODY_DETAIL' else scene.cm_settings.result
                donor = old or accepted_obj
                if donor:
                    for material in donor.data.materials:obj.data.materials.append(material)
                    for name in ('cm_reference_normal_image', 'cm_reference_normal'):
                        value = donor.get(name)
                        if value is None and accepted_obj:value = accepted_obj.get(name)
                        if value is not None:obj[name] = value
                if role in {'HEAD_DETAIL', 'BODY_DETAIL'}:
                    obj['cm_detail_source'] = accepted_obj
                elif obj.get('cm_wrinkle_native'):
                    from . import wrinkles
                    for proxy in wrinkles.restore_proxies(context, scene.cm_settings, f, obj):
                        owned.update((proxy, proxy.data))
                staged[role] = obj
            capture.link(scene)
            owned.update(capture.ids)
        # Publish only after all carriers have passed validation and deep copy.
        for role, obj in staged.items():
            old = previous[role]
            if old:
                old.hide_set(True)
                old.hide_render = True
                for child in old.children:
                    if child.get('cm_wrinkle_owner') == old:
                        child.hide_set(True)
                        child.hide_render = True
            _assign(scene, role, obj)
            if obj:
                hidden, render = visibility.get(old, (True, True))
                obj.hide_set(hidden)
                obj.hide_render = render
                for child in obj.children:
                    if child.get('cm_wrinkle_owner') == obj:
                        child.hide_set(hidden)
                        child.hide_render = render
        context.view_layer.update()
        for entry in table.values():
            if entry['role'] in pending:entry['adopted'] = True
        scene[RETURNED] = json.dumps(table, sort_keys=True)
        keys = []
        for role in pending:
            keys += ['FINAL_HEAD' if role == 'HEAD_DETAIL' else 'FINAL_BODY'] if role in {'HEAD_DETAIL', 'BODY_DETAIL'} else [role, 'CM' + role[-1]]
        if rebind_maps:map_provenance.invalidate_keys(scene, keys)
        # Only artifacts from the adopted source can be certified for this new
        # context. A cavity from an earlier partial return remains stale.
        outputs = {name for name, entry in table.items() if entry['role'] in pending
            and entry.get('receipt', {}).get('sha256') == pending[entry['role']][1]['receipt']['sha256']
            and (folder/name).is_file() and hashlib.sha256((folder/name).read_bytes()).hexdigest() == entry['map_sha256']}
        if rebind_maps:
            map_provenance.record(scene, map_provenance.basis_for_files(scene, outputs), origin='SCULPT_SESSION', outputs=outputs)
        yield staged
    except Exception:
        owned.update(capture.ids)
        for role in staged:_assign(scene, role, previous[role])
        for obj, (hidden, render) in visibility.items():
            obj.hide_set(hidden)
            obj.hide_render = render
        if previous_table is None:
            if RETURNED in scene:del scene[RETURNED]
        else:scene[RETURNED] = previous_table
        if provenance is None:provenance_path.unlink(missing_ok=True)
        else:provenance_path.write_bytes(provenance)
        if invalidated is None:
            if map_provenance.INVALIDATED in scene:del scene[map_provenance.INVALIDATED]
        else:scene[map_provenance.INVALIDATED] = invalidated
        bpy.data.batch_remove(ids=owned)
        context.view_layer.update()
        raise


def _plain(value):
    """Keep numeric/string metadata; never serialize an ID dependency."""
    if isinstance(value, bpy.types.ID):
        raise ValueError("ID reference")
    if isinstance(value, (str, int, float, bool)):
        return value
    if hasattr(value, "items"):
        return {key: _plain(item) for key, item in value.items()}
    return [_plain(item) for item in value]


def _properties(source, target):
    omitted = []
    for name in list(target.keys()):
        del target[name]
    for name, value in source.items():
        try:
            target[name] = _plain(value)
        except (ValueError, TypeError):
            omitted.append(name)
    return omitted


def _check_source(obj, role):
    if obj.type != "MESH":
        raise ValueError(role + " must be a mesh")
    if obj.constraints:
        raise ValueError(role + ": apply live constraints before archiving")
    from . import sculpt_modifiers
    sculpt_modifiers.validate(obj, role)
    for data in (obj, obj.data, obj.data.shape_keys):
        if data and data.animation_data:
            raise ValueError(role + ": animated sculpt layers need explicit recovery")


def write(scene, path, roles=ROLES):
    """Atomically publish only selected sculpt objects and their private data.

    No source data, selection, visibility, scene pointers or filenames change.
    The receipt is returned only after a complete, compressed library write.
    """
    path = Path(path)
    roles = set(roles)
    if path.exists():
        raise ValueError("Sculpt archives are immutable; choose a new destination")
    selected = {role: obj for role, obj in layers(scene).items() if role in roles and obj}
    if set(roles) - set(ROLES):
        raise ValueError("Unknown sculpt archive role")
    for role, obj in selected.items():
        _check_source(obj, role)
    before = set(bpy.data.user_map())
    temporary = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
    receipt = {"schema": SCHEMA, "roles": sorted(roles), "objects": {}, "omitted_references": {}}
    try:
        from .sculpt_dependencies import Capture,strip_materials
        with bpy.context.temp_override(scene=scene,view_layer=scene.view_layers[0]):
            scene.view_layers[0].update()
            capture = Capture(bpy.context)
        objects = set()
        for role, source in selected.items():
            obj = source.copy()
            obj.data = source.data.copy()
            obj.name = "CM_Archive_" + role + "_" + uuid.uuid4().hex
            world = source.matrix_world.copy()
            obj.parent = None
            obj.matrix_world = world
            obj.animation_data_clear()
            strip_materials(obj.data)
            omitted = _properties(source, obj)
            omitted += ["mesh." + key for key in _properties(source.data, obj.data)]
            if obj.data.shape_keys:
                omitted += ["keys." + key for key in _properties(source.data.shape_keys, obj.data.shape_keys)]
            obj["cm_archive_role"] = role
            obj["cm_archive_schema"] = SCHEMA
            # Unlinked objects do not evaluate their world matrix after append.
            obj["cm_archive_world"] = [value for row in world for value in row]
            receipt["objects"][role] = obj.name
            receipt["omitted_references"][role] = omitted
            objects.add(obj)
        with bpy.context.temp_override(scene=scene,view_layer=scene.view_layers[0]):
            for obj in objects:capture.modifiers(obj)
        receipt['dependency_objects'] = sorted(data.name for data in capture.ids if isinstance(data,bpy.types.Object))
        # A forgotten dependency would duplicate assets or leave a mutable link.
        allowed = objects | {obj.data for obj in objects}
        allowed |= {obj.data.shape_keys for obj in objects if obj.data.shape_keys}
        allowed |= capture.ids
        users = bpy.data.user_map()
        dependencies = {dependency for dependency, owners in users.items() if owners & allowed}
        if dependencies - allowed:
            raise ValueError("Sculpt archive still references external Blender data")
        path.parent.mkdir(parents=True, exist_ok=True)
        bpy.data.libraries.write(str(temporary), objects, fake_user=True, compress=True)
        receipt["sha256"] = hashlib.sha256(temporary.read_bytes()).hexdigest()
        receipt["bytes"] = temporary.stat().st_size
        # Single process publication, and never intentionally replace an archive.
        if path.exists():
            raise ValueError("Sculpt archive destination was already published")
        temporary.rename(path)
        return receipt
    finally:
        temporary.unlink(missing_ok=True)
        bpy.data.batch_remove(ids=set(bpy.data.user_map()) - before)


@contextmanager
def read(path, receipt):
    """Load verified, private objects for inspection or a staged recovery plan.

    All appended IDs are removed on exit, including when the caller fails.
    Callers must deep-copy any data they intend to retain before leaving this
    scope, including meshes and any custom ID references they introduce.
    """
    path = Path(path)
    if receipt.get("schema") != SCHEMA:
        raise ValueError("Unsupported sculpt archive schema")
    if hashlib.sha256(path.read_bytes()).hexdigest() != receipt.get("sha256"):
        raise ValueError("Sculpt archive changed; recovery refused")
    names = receipt.get("objects", {})
    declared = receipt.get('roles', [])
    if set(declared) - set(ROLES) or not set(names) <= set(declared):
        raise ValueError("Invalid sculpt archive declared roles")
    if set(names) - set(ROLES) or len(set(names.values())) != len(names):
        raise ValueError("Invalid sculpt archive role table")
    before = set(bpy.data.user_map())
    appended = None
    try:
        with bpy.data.libraries.load(str(path), link=False) as (source, loaded):
            if not set(names.values()) <= set(source.objects):
                raise ValueError("Sculpt archive objects are missing")
            loaded.objects = list(names.values())
        appended = set(bpy.data.user_map()) - before
        result = dict(zip(names, loaded.objects))
        for role, obj in result.items():
            if not obj or obj.get("cm_archive_role") != role or obj.get("cm_archive_schema") != SCHEMA:
                raise ValueError("Sculpt archive role does not match its receipt")
            _check_source(obj, role)
            world = list(obj.get("cm_archive_world", ()))
            if len(world) != 16:
                raise ValueError("Sculpt archive transform is missing")
            obj.matrix_world = Matrix([world[index:index + 4] for index in range(0, 16, 4)])
        yield result
    finally:
        bpy.data.batch_remove(ids=appended if appended is not None else set(bpy.data.user_map()) - before)
