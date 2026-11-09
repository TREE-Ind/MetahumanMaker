"""Apply selected albedos to a verified native head/body, retaining its rig."""
import hashlib
import json
import runpy
import traceback
from datetime import datetime, timezone
from pathlib import Path


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def _fields(text):
    """Split Unreal struct text at top-level commas, preserving nested values."""
    if not text.startswith('(') or not text.endswith(')'):
        raise ValueError('Expected native struct text for the preservation comparison')
    parts = []; start = depth = 0; quoted = escaped = False
    body = text[1:-1]
    for index, char in enumerate(body):
        if quoted:
            if escaped:escaped = False
            elif char == '\\':escaped = True
            elif char == '"':quoted = False
        elif char == '"':quoted = True
        elif char == '(':depth += 1
        elif char == ')':depth -= 1
        elif char == ',' and depth == 0:
            parts.append(body[start:index]); start = index + 1
    if quoted or depth:
        raise ValueError('Unbalanced native preservation snapshot')
    if body[start:]:parts.append(body[start:])
    return dict(part.split('=', 1) for part in parts)


def _artist_skin(settings):
    result = _fields(settings.export_text())
    skin = _fields(settings.get_editor_property('skin').export_text())
    result['Skin'] = {key: value for key, value in skin.items() if key not in {'BodyBias', 'BodyGain'}}
    return result


def update(request_path):
    import unreal
    folder = Path(request_path).resolve().parent
    snapshot = folder / 'scripts'
    request = json.loads(Path(request_path).read_text(encoding='utf-8'))
    report = dict(status='preflight', action='UPDATE_TEXTURES', geometry_mode='NATIVE_ALBEDO', updated=[],
                  warnings=[], retain_source_snapshot=True, requires_reassembly=bool(request.get('existing_assembly')))
    actor = character = subsystem = None
    owned = committed = False
    old_skin = old_preview = None
    def write():
        temporary = folder / 'texture-progress.tmp'
        temporary.write_text(json.dumps(report, indent=2), encoding='utf-8')
        temporary.replace(folder / 'texture-progress.json')
    try:
        if request.get('schema_version') != 1 or request.get('action') != 'UPDATE_TEXTURES' or request.get('geometry_mode') != 'NATIVE_ALBEDO':
            raise ValueError('Unsupported native albedo request')
        project = Path(unreal.Paths.get_project_file_path()).resolve()
        if not request.get('expected_project') or project != Path(request['expected_project']).resolve():
            raise ValueError('Open the linked character’s Unreal project')
        runpy.run_path(str(snapshot / 'resource_budget.py'))['require']('native albedo application', 8 * 1024**3)
        character = unreal.load_asset(request['character'])
        if not isinstance(character, unreal.MetaHumanCharacter):
            raise ValueError('The linked native MetaHuman Character is missing')
        refresh = getattr(unreal.CreateMHCharacterLibrary, 'refresh_skin_preview', None)
        if refresh is None:
            raise ValueError('Install CreateMetaHumanBridge 0.19 or later to refresh native skin preview safely')
        report.update(character=character.get_path_name(), unreal_project_file=str(project))
        before = json.loads(unreal.CreateMHArtistLibrary.snapshot_object(character))
        if any(row['package_dirty'] for row in before['objects']):
            raise ValueError('Save native artist edits before applying albedo')
        geometry_file = snapshot / 'geometry.json'
        if digest(geometry_file) != request['geometry_sha256']:
            raise ValueError('The prepared native geometry changed')
        geometry = json.loads(geometry_file.read_text(encoding='utf-8'))['vertices']
        if not geometry or set(geometry) - {'head', 'eye_left', 'eye_right', 'teeth', 'body'}:
            raise ValueError('Invalid native geometry correspondence')
        selected = {item['role'] for item in request['maps']}
        if not selected or selected - {'HEAD', 'BODY'} or len(selected) != len(request['maps']):
            raise ValueError('Choose unique head/body albedos')
        if ('HEAD' in selected and not geometry.get('head')) or ('BODY' in selected and not geometry.get('body')):
            raise ValueError('Selected albedo has no fitted geometry witness')
        tasks = []
        for item in request['maps']:
            relative = Path(item['snapshot'])
            filename = (snapshot / relative).resolve()
            source = Path(item['source'])
            maps = (Path(request['project_dir']) / 'Maps').resolve()
            if (item.get('kind') != 'ALBEDO' or relative.is_absolute() or filename.parent != (snapshot / 'Maps').resolve()
                    or filename.suffix.lower() != '.png' or source.resolve().parent != maps
                    or digest(filename) != item['sha256'] or digest(source) != item['sha256']):
                raise ValueError('Albedo snapshot or accepted source changed')
            tasks.append((item, filename))
        subsystem = unreal.get_editor_subsystem(unreal.MetaHumanCharacterEditorSubsystem)
        report['status'] = 'initializing-native-character'
        write()
        owned = not subsystem.is_object_added_for_editing(character)
        if owned and not subsystem.try_add_object_to_edit(character):
            raise RuntimeError('Creator could not initialize the linked character')
        state = json.loads(unreal.CreateMHCharacterLibrary.inspect_character(character))
        if state.get('rig_request_pending') or state.get('texture_request_pending'):
            raise ValueError('Finish pending native rig/texture requests before applying albedo')
        audit = runpy.run_path(str(snapshot / 'unreal_import.py'))['audit_head_conform']
        head = {key: value for key, value in geometry.items() if key != 'body'}
        if head:
            measured = audit(character, head, folder / 'albedo-head-state.json')
            if measured.get('within_native_precision') is not True:
                raise ValueError('The native head differs from the fitted mesh; retain geometry edits before applying albedo')
            report['head_geometry_check'] = measured
        helpers = runpy.run_path(str(snapshot / 'unreal_revision_textures.py'))
        actor = subsystem.spawn_meta_human_actor(character, True)
        if not actor:
            raise RuntimeError('Creator did not produce a native preview actor')
        components = {role: actor.get_editor_property(name) for role, name in
                      (('HEAD', 'face_component'), ('BODY', 'body_component'))}
        if 'body' in geometry:
            points, _ = subsystem.get_mesh_data_for_conforming(components['BODY'].get_skeletal_mesh_asset())
            report['body_geometry_check'] = helpers['coverage']({'body': geometry['body']}, points)
        before_mesh = {role: helpers['mesh_digest'](subsystem, component) for role, component in components.items()}
        old_skin = character.get_editor_property('skin_settings').copy()
        old_preview = character.get_editor_property('preview_material_type')
        skin = old_skin.copy()
        overrides = skin.get_editor_property('texture_material_overrides').copy()
        texture_set = overrides.get_editor_property('texture_overrides').copy()
        materials = overrides.get_editor_property('material_overrides').copy()
        faces = dict(materials.get_editor_property('skin'))
        body_material = materials.get_editor_property('body')
        active_materials = bool(overrides.get_editor_property('enable_material_overrides'))
        lib = unreal.MaterialEditingLibrary
        # Existing custom native skin instances get private copies. Unknown
        # graphs are rejected before any bindings change, never replaced.
        custom = []
        if active_materials:
            if 'HEAD' in selected:
                custom += [('HEAD', key, value) for key, value in faces.items() if value]
            if 'BODY' in selected and body_material:
                custom.append(('BODY', None, body_material))
        for role, key, material in custom:
            if not isinstance(material, unreal.MaterialInstanceConstant) or 'Basecolor' not in {str(n) for n in lib.get_texture_parameter_names(material)}:
                raise ValueError('The artist ' + role.lower() + ' material needs a Basecolor texture adapter; it was retained')
        bindings = runpy.run_path(str(snapshot / 'unreal_roughness.py'))
        live = bindings['capture'](character)
        tools = unreal.AssetToolsHelpers.get_asset_tools()
        root = character.get_path_name().rsplit('/', 1)[0]
        imported = {}
        report['status'] = 'importing'
        write()
        for item, filename in tasks:
            # New request-owned assets retain every previous artist texture.
            name = filename.stem + '_Albedo_' + folder.name
            target = root + '/Maps/' + name + '.' + name
            if unreal.load_asset(target):
                raise ValueError('This albedo request already owns an asset; inspect its original receipt before retrying')
            task = unreal.AssetImportTask()
            for key, value in [('filename', str(filename)), ('destination_path', root + '/Maps'),
                               ('destination_name', name), ('automated', True), ('save', False), ('replace_existing', False)]:
                task.set_editor_property(key, value)
            tools.import_asset_tasks([task])
            if list(task.get_editor_property('imported_object_paths')) != [target]:
                raise RuntimeError('Unreal imported an unexpected albedo asset')
            texture = unreal.load_asset(target)
            if not isinstance(texture, unreal.Texture2D):
                raise RuntimeError('Albedo import did not create a Texture2D')
            texture.set_editor_property('srgb', True)
            texture.set_editor_property('compression_settings', unreal.TextureCompressionSettings.TC_DEFAULT)
            unreal.EditorAssetLibrary.set_metadata_tag(texture, 'CreateMetaHuman.TextureOwner', character.get_path_name())
            if digest(item['source']) != item['sha256']:
                raise ValueError('The accepted albedo changed during import')
            texture.get_editor_property('asset_import_data').scripted_add_filename(item['source'], 0, '')
            imported[item['role']] = texture
            report['updated'].append(dict(role=item['role'], kind='ALBEDO', asset=target, source=item['source'], sha256=item['sha256']))
        for role, key, material in custom:
            name = 'Albedo_' + folder.name + '_' + role + '_' + str(key or 'Body')
            if unreal.EditorAssetLibrary.does_asset_exist(root + '/Materials/' + name):
                raise ValueError('Albedo material already exists for this request; inspect its receipt')
            private = tools.duplicate_asset(name, root + '/Materials', material)
            if not private:
                raise RuntimeError('Could not retain the artist skin instance')
            lib.set_material_instance_texture_parameter_value(private, 'Basecolor', imported[role])
            if role == 'HEAD':
                faces[key] = private
            else:
                materials.set_editor_property('body', private)
            if not unreal.EditorAssetLibrary.save_loaded_asset(private, False):
                raise RuntimeError('Could not save the private albedo material')
        materials.set_editor_property('skin', faces)
        for role, texture in imported.items():
            field = 'face' if role == 'HEAD' else 'body'
            enum = unreal.FaceTextureType.BASECOLOR if role == 'HEAD' else unreal.BodyTextureType.BODY_BASECOLOR
            values = dict(texture_set.get_editor_property(field))
            values[enum] = texture
            texture_set.set_editor_property(field, values)
        overrides.set_editor_property('texture_overrides', texture_set)
        overrides.set_editor_property('material_overrides', materials)
        overrides.set_editor_property('enable_texture_overrides', True)
        skin.set_editor_property('texture_material_overrides', overrides)
        committed = True
        subsystem.commit_skin_settings(character, skin)
        character.set_editor_property('preview_material_type', unreal.MetaHumanCharacterSkinPreviewMaterial.EDITABLE)
        if not refresh(character):
            raise RuntimeError('Native skin preview could not refresh')
        bindings['restore'](live, lambda component, current, key:
            ('BODY' if component.get_name() == 'Body' else 'HEAD') in selected and key == 'Basecolor')
        actual_skin = character.get_editor_property('skin_settings').copy()
        actual_overrides = actual_skin.get_editor_property('texture_material_overrides')
        for role, texture in imported.items():
            field = 'face' if role == 'HEAD' else 'body'
            enum = unreal.FaceTextureType.BASECOLOR if role == 'HEAD' else unreal.BodyTextureType.BODY_BASECOLOR
            if dict(actual_overrides.get_editor_property('texture_overrides').get_editor_property(field)).get(enum) != texture:
                raise RuntimeError('The native albedo binding did not survive commit')
        actual_skin.set_editor_property('texture_material_overrides', old_skin.get_editor_property('texture_material_overrides'))
        # CommitSkinSettings derives the read-only BodyBias/BodyGain from U/V.
        # Compare every other field exactly, including all artist skin controls.
        # Comparing those two stale caches to freshly evaluated caches rejects
        # a valid commit on a character that never previously committed skin.
        properties = old_skin.get_editor_property('skin')
        actual_properties = actual_skin.get_editor_property('skin')
        derived = {}
        for field in ('body_bias', 'body_gain'):
            before_value = properties.get_editor_property(field)
            after_value = actual_properties.get_editor_property(field)
            derived[field] = dict(before=[before_value.x, before_value.y, before_value.z],
                                  after=[after_value.x, after_value.y, after_value.z])
        report['native_derived_body_color'] = derived
        if _artist_skin(actual_skin) != _artist_skin(old_skin):
            report['skin_difference'] = dict(before=_artist_skin(old_skin), after=_artist_skin(actual_skin))
            raise RuntimeError('An unrelated native skin setting changed')
        if {role: helpers['mesh_digest'](subsystem, component) for role, component in components.items()} != before_mesh:
            raise RuntimeError('Native geometry changed during albedo application')
        after_state = json.loads(unreal.CreateMHCharacterLibrary.inspect_character(character))
        if any(after_state.get(key) != state.get(key) for key in ('rig_state', 'has_face_dna', 'has_body_dna', 'rig_request_pending', 'texture_request_pending')):
            raise RuntimeError('The native rig or pending requests changed')
        after = json.loads(unreal.CreateMHArtistLibrary.snapshot_object(character))
        def properties(data):
            return next(row['properties'] for row in data['objects'] if row['object'] == character.get_path_name())
        before_props, after_props = properties(before), properties(after)
        changed = [key for key in before_props if key not in {'SkinSettings', 'PreviewMaterialType'}
                   and before_props[key]['text'] != after_props.get(key, {}).get('text')]
        if changed:
            raise RuntimeError('Unrelated artist settings changed: ' + ', '.join(changed))
        verified = {}
        for role, texture in imported.items():
            component = components[role]
            slots = []
            for index in range(component.get_num_materials()):
                material = component.get_material(index)
                if not material:
                    continue
                names = {str(n) for n in lib.get_texture_parameter_names(material)}
                if 'Basecolor' not in names or 'Micro Skin Details Normal' not in names:
                    continue
                if material.get_texture_parameter_value('Basecolor') != texture:
                    raise RuntimeError('The evaluated ' + role.lower() + ' skin does not use the new albedo')
                slots.append(index)
            if not slots:
                raise RuntimeError('No evaluated ' + role.lower() + ' skin slots were verified')
            verified[role] = slots
        for texture in imported.values():
            if not unreal.EditorAssetLibrary.save_loaded_asset(texture, False):
                raise RuntimeError('Could not save the native albedo texture')
        if not unreal.EditorAssetLibrary.save_loaded_asset(character, False):
            raise RuntimeError('Could not save native albedo bindings')
        report.update(status='complete', retain_source_snapshot=False, geometry_preserved=True, rig_preserved=True,
                      unrelated_artist_settings_preserved=True, unrelated_skin_settings_preserved=True,
                      evaluated_skin_slots=verified, assembly=after_state)
    except Exception as exc:
        report.update(status='failed', error=str(exc), traceback=traceback.format_exc())
        if committed and old_skin is not None:
            try:
                subsystem.commit_skin_settings(character, old_skin)
                character.set_editor_property('preview_material_type', old_preview)
                if not refresh(character):
                    raise RuntimeError('Could not refresh restored native preview mode')
                bindings['restore'](live)
                report['skin_bindings_restored'] = bool(unreal.EditorAssetLibrary.save_loaded_asset(character, False))
                if not report['skin_bindings_restored']:
                    report['restore_error'] = 'Previous bindings restored in memory; save the character before closing'
            except Exception as restore_error:
                report['restore_error'] = str(restore_error)
    finally:
        if actor:
            unreal.get_editor_subsystem(unreal.EditorActorSubsystem).destroy_actor(actor)
        if owned and character and subsystem:
            subsystem.remove_object_to_edit(character)
        report['completed_at'] = datetime.now(timezone.utc).isoformat()
        write()
    return report
