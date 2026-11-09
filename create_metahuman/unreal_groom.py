"""Runs in Unreal with the compiled CreateMetaHumanBridge editor plugin."""

from pathlib import Path
import json
import re


GROUP_PROPERTIES = {
    'rendering': ('hair_groups_rendering', 'HairGroupsRendering'),
    'physics': ('hair_groups_physics', 'HairGroupsPhysics'),
    'interpolation': ('hair_groups_interpolation', 'HairGroupsInterpolation'),
    'lod': ('hair_groups_lod', 'HairGroupsLOD'),
}


def binding_settings(binding):
    """Serializable authored skeletal binding inputs, excluding derived buffers."""
    names=('groom_binding_type','groom','source_skeletal_mesh','target_skeletal_mesh',
           'source_mesh_requested_lod','target_mesh_requested_min_lod',
           'num_interpolation_points','matching_section','target_binding_attribute')
    result={}
    for name in names:
        value=binding.get_editor_property(name)
        result[name]=(value.get_path_name() if value else None) if name in {'groom','source_skeletal_mesh','target_skeletal_mesh'} else (value if isinstance(value,int) else str(value))
    return result


def require_binding_roots(binding):
    """Reject placeholder builds that UE cannot attach to a groom component."""
    groups=list(binding.get_editor_property('group_infos'))
    if not groups or any(int(group.get_editor_property('sim_root_count'))<=0 for group in groups):
        raise RuntimeError('Native groom binding has no usable guide roots. Use a rendering-enabled Unreal editor and check groom guides and transfer settings. The existing character was retained.')
    return [dict(guides=int(group.get_editor_property('sim_root_count')),
                 curves=int(group.get_editor_property('ren_root_count')),
                 guide_lods=int(group.get_editor_property('sim_lod_count')),
                 curve_lods=int(group.get_editor_property('ren_lod_count'))) for group in groups]


def copy_binding(binding,groom,target,destination):
    """Build a private binding revision with every existing authored setting."""
    import unreal
    if not isinstance(binding,unreal.GroomBindingAsset) or not isinstance(groom,unreal.GroomAsset) or not isinstance(target,unreal.SkeletalMesh):
        raise ValueError('Choose an existing skeletal groom binding, groom and target mesh')
    if binding.get_editor_property('groom_binding_type')!=unreal.GroomBindingMeshType.SKELETAL_MESH:
        raise ValueError('This revision workflow requires a skeletal groom binding')
    if not re.fullmatch(r'/Game(?:/[A-Za-z0-9_]+)+',destination) or unreal.EditorAssetLibrary.does_asset_exist(destination):
        raise ValueError('Choose a fresh project groom binding destination')
    # A freshly revised face can still be compiling. UE defers a binding with
    # pending mesh dependencies without retaining the supplied build callback.
    # Finish native asset compilation before requesting this particular build.
    unreal.SystemLibrary.execute_console_command(None,'Editor.AsyncAssetCompilationFinishAll')
    expected=binding_settings(binding)
    fresh=unreal.GroomLibrary.create_new_groom_binding_asset_with_path(
        destination,groom,target,binding.get_editor_property('num_interpolation_points'),
        binding.get_editor_property('source_skeletal_mesh'),binding.get_editor_property('matching_section'))
    if not isinstance(fresh,unreal.GroomBindingAsset):raise RuntimeError('Could not create the private groom binding')
    # The constructor sets the protected native inputs but omits these three
    # authored settings. Their reflected notifications rebuild derived data.
    # Do not call Build again: Reset can clear roots while unchanged cached
    # keys cause that second build to skip regeneration in UE 5.8.
    extra={key:binding.get_editor_property(key) for key in
           ('source_mesh_requested_lod','target_mesh_requested_min_lod','target_binding_attribute')
           if fresh.get_editor_property(key)!=binding.get_editor_property(key)}
    if extra:fresh.set_editor_properties(extra)
    unreal.SystemLibrary.execute_console_command(None,'Editor.AsyncAssetCompilationFinishAll')
    require_binding_roots(fresh)
    expected.update(groom=groom.get_path_name(),target_skeletal_mesh=target.get_path_name())
    if binding_settings(fresh)!=expected:
        raise RuntimeError('Native groom rebuild changed authored transfer settings; original binding retained')
    if not unreal.EditorAssetLibrary.save_loaded_asset(fresh,False):raise RuntimeError('Could not save the private groom binding revision')
    return fresh


def apply_owned_revision(blueprint,component_name,groom,binding,*,owner_folder,expected_original,compile_blueprint=True):
    """Persist a reviewed groom/binding pair on an owned assembled revision.

    Retains the component's material and rendering overrides. Shared original
    groom, binding and skeletal assets are never edited or saved.
    """
    import unreal
    library=unreal.EditorAssetLibrary
    if not isinstance(blueprint,unreal.Blueprint) or not owner_folder.startswith('/Game/') or not blueprint.get_path_name().startswith(owner_folder.rstrip('/')+'/'):
        raise ValueError('Choose a blueprint inside the owned character revision folder')
    if not isinstance(groom,unreal.GroomAsset) or not isinstance(binding,unreal.GroomBindingAsset) or binding.get_editor_property('groom')!=groom:
        raise ValueError('Choose a matching reviewed groom and binding')
    subsystem=unreal.get_engine_subsystem(unreal.SubobjectDataSubsystem);data=unreal.SubobjectDataBlueprintFunctionLibrary;components=[]
    for handle in subsystem.k2_gather_subobject_data_for_blueprint(blueprint):
        value=data.get_object(data.get_data(handle))
        if value not in components:components.append(value)
    selected=[c for c in components if isinstance(c,unreal.GroomComponent) and c.get_name().removesuffix('_GEN_VARIABLE')==component_name]
    if len(selected)!=1:raise ValueError('Choose exactly one named groom component in this assembly')
    component=selected[0];old=component.get_editor_property('groom_asset');old_binding=component.get_editor_property('binding_asset');target=binding.get_editor_property('target_skeletal_mesh')
    if not old or old.get_path_name()!=expected_original or not old_binding or old_binding.get_editor_property('target_skeletal_mesh')!=target:
        raise ValueError('The artist groom or its rig changed since this revision was prepared')
    materials=[component.get_material(i) for i in range(component.get_num_materials())]
    protected={asset.get_path_name().split('.')[0] for asset in (old,old_binding,target)}
    dirty=lambda:{p.get_path_name() for p in unreal.EditorLoadingAndSavingUtils.get_dirty_content_packages()}
    before=dirty();tag='CreateMetaHuman.GroomRevision.'+component_name;old_tag=library.get_metadata_tag(blueprint,tag)
    try:
        blueprint.modify();component.modify()
        component.set_groom_asset(groom);component.set_binding_asset(binding)
        if materials!=[component.get_material(i) for i in range(component.get_num_materials())]:raise RuntimeError('Groom revision did not retain component materials')
        if compile_blueprint:unreal.BlueprintEditorLibrary.compile_blueprint(blueprint)
        if (dirty()-before)&protected:raise RuntimeError('Groom revision unexpectedly dirtied a shared source asset')
        report=dict(status='complete',blueprint=blueprint.get_path_name(),component=component_name,previous_groom=old.get_path_name(),previous_binding=old_binding.get_path_name(),groom=groom.get_path_name(),binding=binding.get_path_name(),target=target.get_path_name(),source_assets_saved=0)
        library.set_metadata_tag(blueprint,tag,json.dumps(report))
        if not library.save_loaded_asset(blueprint,False):raise RuntimeError('Could not save the owned groom revision')
        return report
    except Exception:
        if old_tag:library.set_metadata_tag(blueprint,tag,old_tag)
        else:library.remove_metadata_tag(blueprint,tag)
        # Compilation reinstantiates components; restore through fresh handles.
        for handle in subsystem.k2_gather_subobject_data_for_blueprint(blueprint):
            value=data.get_object(data.get_data(handle))
            if isinstance(value,unreal.GroomComponent) and value.get_name().removesuffix('_GEN_VARIABLE')==component_name:
                value.set_groom_asset(old);value.set_binding_asset(old_binding)
        if compile_blueprint:unreal.BlueprintEditorLibrary.compile_blueprint(blueprint)
        raise


def rebind_revision_grooms(blueprint, target, destination):
    """Rebuild face groom bindings after replacing a private assembly's mesh.

    Groom assets, authored transfer meshes and rendering settings stay intact.
    Only bindings that share the revised face skeleton are considered. The
    destination must be fresh and owned by this blueprint's package folder.
    """
    import unreal
    library=unreal.EditorAssetLibrary
    if not isinstance(blueprint,unreal.Blueprint) or not isinstance(target,unreal.SkeletalMesh):
        raise ValueError('Supply an assembled blueprint and its revised face mesh')
    owner=blueprint.get_path_name();folder=owner.rsplit('/',1)[0]
    if not destination.startswith(folder+'/') or library.does_directory_exist(destination):
        raise ValueError('Use a fresh groom binding folder within the revised assembly')
    subsystem=unreal.get_engine_subsystem(unreal.SubobjectDataSubsystem);data=unreal.SubobjectDataBlueprintFunctionLibrary
    components=[]
    for handle in subsystem.k2_gather_subobject_data_for_blueprint(blueprint):
        component=data.get_object(data.get_data(handle))
        if component not in components:components.append(component)
    faces=[c for c in components if isinstance(c,unreal.SkeletalMeshComponent) and c.get_name().removesuffix('_GEN_VARIABLE')=='Face']
    if not faces or any(c.get_skeletal_mesh_asset()!=target for c in faces):
        raise ValueError('The revised blueprint must already use this face mesh')
    pending=[];created={};skipped=[]
    for component in components:
        if not isinstance(component,unreal.GroomComponent):continue
        groom=component.get_editor_property('groom_asset');binding=component.get_editor_property('binding_asset')
        if not groom or not binding:continue
        old_target=binding.get_editor_property('target_skeletal_mesh')
        if old_target==target:continue
        if not old_target or old_target.get_editor_property('skeleton')!=target.get_editor_property('skeleton'):
            skipped.append(component.get_name());continue
        if binding.get_editor_property('groom')!=groom:raise ValueError('Groom and previous binding do not match')
        key=binding.get_path_name()
        if key not in created:
            name='Binding_'+str(len(created))+'_'+groom.get_name()
            fresh=copy_binding(binding,groom,target,destination+'/'+name)
            if not fresh or fresh.get_editor_property('target_skeletal_mesh')!=target or fresh.get_editor_property('groom')!=groom:
                raise RuntimeError('Native groom rebinding failed; previous bindings retained')
            library.set_metadata_tag(fresh,'CreateMetaHuman.GroomBindingRevision',json.dumps(dict(owner=owner,previous=key,target=target.get_path_name())))
            if not library.save_loaded_asset(fresh,False):raise RuntimeError('Could not save the revised groom binding')
            created[key]=fresh
        pending.append((component,binding,created[key]))
    try:
        for component,_,fresh in pending:component.set_binding_asset(fresh)
        unreal.BlueprintEditorLibrary.compile_blueprint(blueprint)
        if not library.save_loaded_asset(blueprint,False):raise RuntimeError('Could not save revised assembly bindings')
    except Exception:
        for component,old,_ in pending:component.set_binding_asset(old)
        unreal.BlueprintEditorLibrary.compile_blueprint(blueprint);library.save_loaded_asset(blueprint,False)
        raise
    return dict(status='complete',blueprint=owner,target=target.get_path_name(),groom_assets_modified=0,
                updated=[dict(component=c.get_name().removesuffix('_GEN_VARIABLE'),previous=old.get_path_name(),binding=fresh.get_path_name()) for c,old,fresh in pending],skipped_other_skeleton=skipped)


def asset_settings(records):
    """Resolve shared deformation settings before creating any Unreal assets."""
    resolved = None
    for record in records:
        settings = record.get('asset_settings')
        if settings is None:  # Legacy strand files keep Unreal's defaults.
            continue
        if not isinstance(settings, dict) or set(settings) != {
                'enable_global_interpolation', 'enable_simulation_cache', 'hair_interpolation_type'}:
            raise ValueError('Groom asset settings are incomplete or unsupported; grab the groom again')
        if any(type(settings[key]) is not bool for key in ('enable_global_interpolation', 'enable_simulation_cache')):
            raise ValueError('Groom interpolation and simulation-cache flags must be boolean')
        if settings['hair_interpolation_type'] not in {'NONE', 'RIGID_TRANSFORM', 'OFFSET_TRANSFORM', 'SMOOTH_TRANSFORM'}:
            raise ValueError('Unsupported groom skinning mode; grab the groom again')
        if resolved is not None and resolved != settings:
            raise ValueError('These grooms use different asset-wide deformation settings. Align Global Interpolation, Simulation Cache and Hair Interpolation Type in Unreal, then grab them again, or push them separately.')
        resolved = settings
    return dict(resolved) if resolved is not None else {}


def export_settings(groom, group_ids):
    """Native struct text retains nested curves, solver and material references."""
    info = groom.get_editor_property('hair_groups_info')
    properties = {name: groom.get_editor_property(key) for name, (key, _) in GROUP_PROPERTIES.items()}
    materials = [item.export_text() for item in groom.get_editor_property('hair_groups_materials')]
    shared = {key: bool(groom.get_editor_property(key)) for key in (
        'enable_global_interpolation', 'enable_simulation_cache')}
    shared['hair_interpolation_type'] = groom.get_editor_property('hair_interpolation_type').name
    if len(group_ids) != len(info):
        raise ValueError('Source strand groups do not match the native groom settings')
    return [dict(id=group_ids[index],
                 name=str(group.get_editor_property('group_name')),
                 materials=materials,
                 asset_settings=dict(shared),
                 **{name: values[index].export_text() for name, values in properties.items()})
            for index, group in enumerate(info)]


def apply_settings(groom, records, group_ids):
    import unreal
    if not records:
        return
    shared = asset_settings(records)
    # Apply before the group LOD array: its native property notification rebuilds
    # derived strand/interpolation data using the complete deformation settings.
    for key, value in shared.items():
        if key == 'hair_interpolation_type':
            value = getattr(unreal.GroomInterpolationType, value)
        groom.set_editor_property(key, value)
    indexed = {int(record['id']): record for record in records}
    info = groom.get_editor_property('hair_groups_info')
    materials = []
    for label, (property_name, struct_name) in GROUP_PROPERTIES.items():
        values = list(groom.get_editor_property(property_name))
        for index, group in enumerate(info):
            group_id = group_ids[index]
            record = indexed.get(group_id, {})
            if record.get(label):
                value = getattr(unreal, struct_name)()
                text = record[label]
                if label == 'rendering' and record.get('materials'):
                    # MaterialSlotName is a protected reflected property in 5.8.
                    # Native struct text exposes it without bypassing reflection.
                    match = re.search(r'\bMaterialSlotName=("(?:\\.|[^"\\])*"|[^,)]+)', text)
                    if not match:
                        raise ValueError('Groom rendering settings have no material slot')
                    token = match.group(1)
                    slot = json.loads(token) if token.startswith('"') else token
                    for text in record.get('materials', []):
                        material = unreal.HairGroupsMaterial()
                        material.import_text(text)
                        if str(material.get_editor_property('slot_name')) == slot:
                            unique = 'CM_Group%d_%s' % (group_id, slot)
                            material.set_editor_property('slot_name', unique)
                            materials.append(material)
                            text = record[label][:match.start(1)] + json.dumps(unique) + record[label][match.end(1):]
                            break
                    else:
                        text = record[label]
                value.import_text(text)
                values[index] = value
        # Whole-array physics/interpolation notifications refresh resources
        # without rebuilding their dependent interpolation bulk data in UE5.8.
        # Enabling simulation on a static groom can then assert in
        # FHairStrandsInterpolationResource. Stage via the reflected notification
        # policy, then force the final LOD rebuild even when LOD values are equal.
        if label == 'lod':
            groom.set_editor_property(property_name, values,
                                      notify_mode=unreal.PropertyAccessChangeNotifyMode.ALWAYS)
        else:
            groom.set_editor_property(property_name, values,
                                      notify_mode=unreal.PropertyAccessChangeNotifyMode.NEVER)
    if materials:
        groom.set_editor_property('hair_groups_materials', materials)


def library():
    import unreal

    cls = getattr(unreal, "CreateMHGroomLibrary", None)
    if cls is None:
        raise RuntimeError(
            "Install the compiled CreateMetaHumanBridge plugin matching UE 5.8 and restart Unreal"
        )
    return cls


def selected_asset(kind, *, details=None):
    """Resolve an explicit asset, or one selected character's Face component."""
    import unreal
    asset_type = unreal.SkeletalMesh if kind == 'HEAD' else unreal.GroomAsset
    assets = [asset for asset in unreal.EditorUtilityLibrary.get_selected_assets()
              if isinstance(asset, asset_type)]
    label = 'Face Skeletal Mesh' if kind == 'HEAD' else 'Groom'
    if len(assets)==1:
        if details is not None:details.update(source='CONTENT_BROWSER',asset=assets[0].get_path_name())
        return assets[0]
    if kind=='HEAD' and not assets:
        actors=list(unreal.get_editor_subsystem(unreal.EditorActorSubsystem).get_selected_level_actors())
        if len(actors)==1:
            faces=[component.get_skeletal_mesh_asset() for component in actors[0].get_components_by_class(unreal.SkeletalMeshComponent)
                   if component.get_name()=='Face' and isinstance(component.get_skeletal_mesh_asset(),unreal.SkeletalMesh)]
            if len(faces)==1:
                if details is not None:details.update(source='ACTOR_FACE',actor=actors[0].get_path_name(),component='Face',asset=faces[0].get_path_name())
                return faces[0]
        raise ValueError('Select one Face Skeletal Mesh asset in the Content Browser, or one character actor with a single Face component')
    raise ValueError('Select exactly one ' + label + ' asset in the Unreal Content Browser, then try again')


def selected_target():
    selection={};asset = selected_asset('HEAD',details=selection)
    return {'source_asset': asset.get_path_name(), 'source_type': 'SkeletalMesh','selection':selection}


def _head_signature(mesh):
    """Hash generated MetaHuman LOD0 geometry; UVs come from its native template."""
    import unreal
    import array
    import hashlib

    result = unreal.MetaHumanCharacterEditorSubsystem.get_mesh_data_for_conforming(mesh)
    if result is None or len(result) != 2:
        raise RuntimeError('Could not read the Creator head rest geometry')
    vertices, triangles = result
    if not vertices or not triangles:
        raise RuntimeError('The Creator head has no usable LOD0 geometry')
    digest = hashlib.sha256()
    digest.update(array.array('f', (value for v in vertices for value in (v.x, v.y, v.z))).tobytes())
    digest.update(array.array('i', triangles).tobytes())
    return digest.hexdigest(), len(vertices), len(triangles) // 3


def character_target(character_path, candidate_path=''):
    """Capture a durable rest head from an explicitly linked native character."""
    import unreal
    import hashlib

    character = unreal.load_asset(character_path)
    if not isinstance(character, unreal.MetaHumanCharacter):
        raise ValueError('Deliver this character to MetaHuman Creator before preparing its hair target')
    if unreal.get_editor_subsystem(unreal.LevelEditorSubsystem).is_in_play_in_editor():
        raise ValueError('Stop Play or Simulate before preparing a character head')
    subsystem = unreal.get_editor_subsystem(unreal.MetaHumanCharacterEditorSubsystem)
    already_editing = subsystem.is_object_added_for_editing(character)
    actor = None
    try:
        if not already_editing and not subsystem.try_add_object_to_edit(character):
            raise RuntimeError('Could not open this MetaHuman for local head preparation')
        # Native SpawnMetaHumanActor takes the character itself. No Content
        # Browser selection, actor-name guess or global transient scan is used.
        actor = subsystem.spawn_meta_human_actor(character, True)
        if not actor:
            raise RuntimeError('Creator could not provide this character\'s local preview')
        face = actor.get_editor_property('face_component')
        source = face.get_skeletal_mesh_asset() if face else None
        if not isinstance(source, unreal.SkeletalMesh):
            raise RuntimeError('The selected character pipeline did not expose a Face mesh')
        signature, vertices, triangles = _head_signature(source)
        owner = character.get_path_name()
        tags = {'CreateMH.GroomSourceCharacter': owner, 'CreateMH.GroomSourceGeometry': signature,
                'CreateMH.GroomSourceTemplate': str(character.get_editor_property('template_type')),
                'CreateMH.GroomSourceEngine': unreal.SystemLibrary.get_engine_version()}
        candidates = []
        previous_path = unreal.EditorAssetLibrary.get_metadata_tag(character, 'CreateMH.GroomRestHead')
        for source_path in dict.fromkeys(path for path in (candidate_path, previous_path) if path):
            candidate = unreal.load_asset(source_path)
            if isinstance(candidate, unreal.SkeletalMesh):
                candidates.append(candidate)
        folder = owner.split('.')[0].rsplit('/', 1)[0] + '/GroomSources'
        name = 'RestHead_' + hashlib.sha256(json.dumps(tags, sort_keys=True).encode('utf-8')).hexdigest()[:20]
        path = folder + '/' + name
        if unreal.EditorAssetLibrary.does_asset_exist(path):
            candidates.append(unreal.load_asset(path))
        mesh = None
        for candidate in candidates:
            if not isinstance(candidate, unreal.SkeletalMesh):
                continue
            if all(unreal.EditorAssetLibrary.get_metadata_tag(candidate, key) == value for key, value in tags.items()):
                if _head_signature(candidate)[0] == signature:
                    mesh = candidate
                    break
        reused = mesh is not None
        if mesh is None:
            if unreal.EditorAssetLibrary.does_asset_exist(path):
                raise ValueError('The saved grooming head no longer matches its source. Rename it in Unreal before preparing another')
            mesh = unreal.AssetToolsHelpers.get_asset_tools().duplicate_asset(name, folder, source)
            if not isinstance(mesh, unreal.SkeletalMesh):
                raise RuntimeError('Could not preserve the authored groom rest head')
            mesh.set_editor_property('asset_user_data', [])
            materials = list(mesh.get_editor_property('materials'))
            material = unreal.load_asset('/Engine/EngineMaterials/DefaultMaterial')
            if not material:
                raise RuntimeError('Could not load the binding-source reference material')
            for slot in materials:
                slot.set_editor_property('material_interface', material)
                slot.set_editor_property('overlay_material_interface', None)
            mesh.set_editor_property('materials', materials)
            for key, value in tags.items():
                unreal.EditorAssetLibrary.set_metadata_tag(mesh, key, value)
            if not unreal.EditorAssetLibrary.save_loaded_asset(mesh, False):
                raise RuntimeError('Could not save the groom rest head; inspect its asset in Unreal')
        if previous_path != mesh.get_path_name():
            unreal.EditorAssetLibrary.set_metadata_tag(character, 'CreateMH.GroomRestHead', mesh.get_path_name())
            if not unreal.EditorAssetLibrary.save_loaded_asset(character, False):
                raise RuntimeError('Rest head saved; save the character in Unreal to retain its source link')
        return {'status': 'complete', 'action': 'CHARACTER_TARGET', 'source_asset': mesh.get_path_name(),
                'source_type': 'SkeletalMesh', 'character': owner, 'geometry_signature': signature,
                'vertices': vertices, 'triangles': triangles, 'reused_source': reused}
    finally:
        try:
            if actor and not unreal.get_editor_subsystem(unreal.EditorActorSubsystem).destroy_actor(actor):
                raise RuntimeError('Could not remove the temporary head preview actor; inspect the Unreal level')
        finally:
            if not already_editing and subsystem.is_object_added_for_editing(character):
                subsystem.remove_object_to_edit(character)


def export_selected(kind, filename):
    selection={};asset = selected_asset(kind,details=selection)
    path = asset.get_path_name()
    if kind == 'HEAD':
        export_head(path, filename)
    else:
        export_groom(path, filename)
    return {'file': Path(filename).name, 'source_asset': path,
            'source_type': 'SkeletalMesh' if kind == 'HEAD' else 'GroomAsset','selection':selection}


def export_groom(asset_path, filename):
    import unreal

    groom = unreal.load_asset(asset_path)
    if not isinstance(groom, unreal.GroomAsset):
        raise ValueError("Choose an Unreal GroomAsset path")
    if not library().export_groom_json(groom, filename):
        raise RuntimeError("Groom has no editable source strand data")
    path = Path(filename)
    data = json.loads(path.read_text(encoding='utf-8-sig'))
    # Runtime HairGroupsInfo.GroupID can remain zero for multiple groups in 5.8.
    # GroomBuilder creates groups in source strand encounter order instead.
    group_ids = list(dict.fromkeys(int(curve['group']) for curve in data['curves']))
    data['group_settings'] = export_settings(groom, group_ids)
    data['source_asset'] = groom.get_path_name()
    data['engine_version'] = unreal.SystemLibrary.get_engine_version()
    path.write_text(json.dumps(data), encoding='utf-8')
    return filename


def export_head(asset_path, filename):
    import unreal

    mesh = unreal.load_asset(asset_path)
    if not isinstance(mesh, unreal.SkeletalMesh):
        raise ValueError("Choose the exported MetaHuman head SkeletalMesh asset")
    task = unreal.AssetExportTask()
    task.object = mesh
    task.filename = filename
    task.automated = True
    task.prompt = False
    task.replace_identical = True
    task.exporter = unreal.SkeletalMeshExporterFBX()
    task.options = unreal.FbxExportOption()
    task.options.level_of_detail = False
    task.options.bake_material_inputs = unreal.FbxMaterialBakeMode.DISABLED
    if not unreal.Exporter.run_asset_export_task(task):
        raise RuntimeError("Head FBX export failed")
    return filename


def apply_to_selected_actor(groom_path,binding_path):
    """Add a persistent instance component; Blueprint and native assets stay intact."""
    import unreal
    if unreal.get_editor_subsystem(unreal.LevelEditorSubsystem).is_in_play_in_editor():
        raise ValueError('Stop Play or Simulate before applying a groom to a level actor')
    groom=unreal.load_asset(groom_path);binding=unreal.load_asset(binding_path)
    if not isinstance(groom,unreal.GroomAsset) or not isinstance(binding,unreal.GroomBindingAsset):
        raise ValueError('Push a groom and binding successfully before applying them')
    if binding.get_editor_property('groom')!=groom:
        raise ValueError('This binding belongs to another groom; push the matching pair again')
    target=binding.get_editor_property('target_skeletal_mesh')
    actors=unreal.get_editor_subsystem(unreal.EditorActorSubsystem).get_selected_level_actors()
    if len(actors)!=1:raise ValueError('Select exactly one assembled character actor in the Unreal level')
    actor=actors[0]
    matches=[c for c in actor.get_components_by_class(unreal.SkeletalMeshComponent) if target and c.get_skeletal_mesh_asset()==target]
    if len(matches)!=1:raise ValueError('The selected actor does not have this binding target exactly once. Select the matching character or push a binding for its final Face mesh.')
    parent=matches[0];tag='CreateMH.Groom:'+groom.get_path_name()
    existing=[c for c in actor.get_components_by_class(unreal.GroomComponent) if c.component_has_tag(tag)]
    if len(existing)>1:raise ValueError('This actor has duplicate managed groom components; remove the duplicate in Unreal before applying again')
    if existing and existing[0].get_attach_parent()!=parent:
        raise ValueError('The managed groom was reattached manually. Restore its Face parent in Unreal before applying again.')
    if not existing and any(c.get_editor_property('groom_asset')==groom for c in actor.get_components_by_class(unreal.GroomComponent)):
        raise ValueError('This groom is already present on an artist-managed component. Edit that component in Unreal to avoid duplicate hair.')
    sub=unreal.get_engine_subsystem(unreal.SubobjectDataSubsystem)
    library=unreal.SubobjectDataBlueprintFunctionLibrary
    handles=sub.k2_gather_subobject_data_for_instance(actor)
    parent_handle=next((h for h in handles if library.get_associated_object(library.get_data(h))==parent),None)
    if parent_handle is None:raise RuntimeError('Could not locate the selected character mesh component')
    created_handle=None
    with unreal.ScopedEditorTransaction('Apply MetaHuman Maker groom'):
        actor.modify()
        try:
            if existing:component=existing[0]
            else:
                created_handle,reason=sub.add_new_subobject(unreal.AddNewSubobjectParams(parent_handle=parent_handle,new_class=unreal.GroomComponent,conform_transform_to_parent=True))
                component=library.get_associated_object(library.get_data(created_handle))
                if not isinstance(component,unreal.GroomComponent):raise RuntimeError('Could not add groom component: '+str(reason))
                component.set_editor_property('component_tags',[tag])
                # Adding an instance reruns the actor's construction script;
                # its inherited Face component has been replaced at this point.
                parent=next(c for c in actor.get_components_by_class(unreal.SkeletalMeshComponent) if c.get_skeletal_mesh_asset()==target)
                component.attach_to_component(parent,'',unreal.AttachmentRule.SNAP_TO_TARGET,unreal.AttachmentRule.SNAP_TO_TARGET,unreal.AttachmentRule.KEEP_WORLD,False)
            component.modify()
            component.set_mobility(unreal.ComponentMobility.MOVABLE)
            component.set_groom_asset(groom);component.set_binding_asset(binding)
            if component.get_attach_parent()!=parent:
                raise RuntimeError('Groom component is not attached to the binding target; inspect its attachment in Unreal')
            if component.get_editor_property('binding_asset')!=binding:
                raise RuntimeError('Unreal refused this groom binding; inspect the Output Log')
        except Exception:
            if created_handle is not None:sub.k2_delete_subobject_from_instance(handles[0],created_handle)
            raise
    return {'status':'complete','action':'APPLY_ACTOR','groom':groom.get_path_name(),'binding':binding.get_path_name(),
            'actor':actor.get_path_name(),'actor_label':actor.get_actor_label(),'component':component.get_name(),
            'target':target.get_path_name(),'reused_component':bool(existing),'level_needs_save':True}


def equip_wardrobe(character_path, item_path, slot='Hair', *, open_editor=True):
    """Register and equip a delivered item without replacing other wardrobe slots."""
    import unreal

    character = unreal.load_asset(character_path)
    item = unreal.load_asset(item_path)
    if not isinstance(character, unreal.MetaHumanCharacter):
        raise ValueError('Deliver this character to MetaHuman Creator before equipping hair')
    if not isinstance(item, unreal.MetaHumanWardrobeItem):
        raise ValueError('Push a groom with a wardrobe item before equipping it')
    collection = character.get_editor_property('internal_collection')
    if not collection or slot not in [str(name) for name in collection.get_slot_names()]:
        raise ValueError('The character pipeline has no compatible ' + slot + ' wardrobe slot')
    if not unreal.EditorAssetLibrary.save_loaded_asset(character, False):
        raise RuntimeError('Could not save the character before updating its wardrobe')
    # Creator edits a preview collection. Close it before changing the saved
    # collection, then reopen so its normal pipeline builds the groom binding.
    editor = unreal.get_editor_subsystem(unreal.AssetEditorSubsystem)
    editor.close_all_editors_for_asset(character)
    collection = character.get_editor_property('internal_collection')
    keys = [key for key in collection.get_item_keys_for_wardrobe_item(item)
            if str(collection.get_item_slot_name(key)) == slot]
    with unreal.ScopedEditorTransaction('Equip MetaHuman Maker wardrobe item'):
        character.modify()
        collection.modify()
        key = keys[0] if keys else collection.try_add_item_from_wardrobe_item(slot, item)
        if key is None:
            raise ValueError('Creator rejected this wardrobe item for the ' + slot + ' slot')
        assets = character.get_editor_property('wardrobe_individual_assets')
        name = unreal.Name(slot)
        row = assets[name] if name in assets else unreal.MetaHumanCharacterWardrobeIndividualAssets()
        items = list(row.get_editor_property('items'))
        if item not in items:
            items.append(item)
        row.set_editor_property('items', items)
        # This visible-only map cannot be replaced through set_editor_property;
        # Unreal exposes a live map whose entries can be edited individually.
        assets[name] = row
        instance = collection.get_editor_property('default_instance')
        instance.modify()
        instance.set_single_slot_selection(slot, key)
        if not unreal.EditorAssetLibrary.save_loaded_asset(character, False):
            raise RuntimeError('Wardrobe updated in memory; save the character in Unreal before closing')
    report = {'status': 'complete', 'action': 'EQUIP_CREATOR', 'character': character.get_path_name(),
              'wardrobe': item.get_path_name(), 'slot': slot, 'reused_item': bool(keys), 'saved': True}
    try:
        report['editor_opened'] = bool(editor.open_editor_for_assets([character])) if open_editor else False
    except Exception as exc:
        report['editor_opened'] = False
        report['editor_warning'] = str(exc)
    return report


def import_groom(filename, destination, target_path, revision=False, character_path=''):
    import unreal

    data = json.loads(Path(filename).read_text(encoding='utf-8-sig'))
    # Conflicting settings in a blend cannot be represented by one GroomAsset.
    # Refuse before the native builder creates or saves a partial asset.
    asset_settings(data.get('group_settings', []))
    prepared = character_target(character_path) if not target_path.strip() and character_path else None
    target = unreal.load_asset(prepared['source_asset'] if prepared else target_path)
    if not isinstance(target, unreal.SkeletalMesh):
        raise ValueError('Choose a valid target SkeletalMesh before pushing the groom')
    base = destination
    number = 2
    while any(unreal.EditorAssetLibrary.does_asset_exist(destination + suffix)
              for suffix in ('', '_Binding', '_Wardrobe')):
        if not revision:
            raise ValueError("Groom destination already exists; choose a new name")
        destination = base + '_v' + str(number).zfill(3)
        number += 1
    groom = library().import_groom_json(filename, destination)
    if not groom:
        raise RuntimeError("Could not build groom strand data")
    unreal.EditorAssetLibrary.save_loaded_asset(groom)
    report = {'groom': groom.get_path_name(), 'binding': '', 'wardrobe': '', 'status': 'importing',
              'target': target.get_path_name()}
    if prepared:
        report['prepared_source'] = prepared
    report_path = Path(filename).with_suffix('.unreal-report.json')
    report_path.write_text(json.dumps(report), encoding='utf-8')
    apply_settings(groom, data.get('group_settings', []),
                   list(dict.fromkeys(int(curve['group']) for curve in data['curves'])))
    unreal.EditorAssetLibrary.save_loaded_asset(groom)
    unreal.SystemLibrary.execute_console_command(None,'Editor.AsyncAssetCompilationFinishAll')
    binding = unreal.GroomLibrary.create_new_groom_binding_asset_with_path(
        destination + "_Binding", groom, target, 100, target, 0
    )
    if not binding:
        raise RuntimeError(
            "Groom imported, but binding failed; check scalp position and target mesh"
        )
    unreal.SystemLibrary.execute_console_command(None,'Editor.AsyncAssetCompilationFinishAll')
    report['binding'] = binding.get_path_name()
    report_path.write_text(json.dumps(report), encoding='utf-8')
    report['binding_roots']=require_binding_roots(binding)
    unreal.EditorAssetLibrary.save_loaded_asset(binding)
    folder, name = destination.rsplit("/", 1)
    item = unreal.AssetToolsHelpers.get_asset_tools().create_asset(
        name + "_Wardrobe",
        folder,
        unreal.MetaHumanWardrobeItem,
        unreal.MetaHumanWardrobeItemFactory(),
    )
    if not item:
        raise RuntimeError("Groom and binding imported; could not create wardrobe item")
    reference = unreal.EditorOnlyAssetReference()
    reference.set_editor_property("asset", binding)
    item.set_editor_property("principal_asset", reference)
    pipeline = unreal.new_object(unreal.MetaHumanGroomPipeline, outer=item)
    item.set_editor_property("pipeline", pipeline)
    unreal.EditorAssetLibrary.save_loaded_asset(item)
    report.update(wardrobe=item.get_path_name(), status='complete',
                  groups=len(groom.get_editor_property('hair_groups_info')))
    Path(filename).with_suffix(".unreal-report.json").write_text(
        json.dumps(report, indent=2)
    )
    try:
        unreal.EditorAssetLibrary.sync_browser_to_objects([item.get_path_name()])
    except Exception as exc:
        report['browser_warning'] = str(exc)
    return report
