"""Carry the add-on's authored source-eye PBR surfaces through native assembly.

Epic's standard eye baker expects its optical shader parameters. The explicit
opaque source-eye adapter uses its own UV textures and must retain that shader.
This only handles the adapter's verified ownership ledger; native optical eyes
continue through the native bake. No Creator settings are changed.
"""
import json
from pathlib import Path
import hashlib

TAG='CreateMetaHuman.SourceEyeAppearance'
ASSEMBLED_TAG='CreateMetaHuman.AssembledSourceEyes'
SOURCE_TAG='CreateMetaHuman.AssembledEyeSource'
COLOR='CM Source Eye Color'
FINISH='CM Source Eye Finish'


def refresh_maps(character,blueprint,maps,destination):
    """Refresh explicitly owned Creator and delivery eye maps together.

    Does not initialize Creator, refit anatomy or touch rig data. Existing
    materials must still match their recorded source state. The destination
    must be fresh; old texture assets remain available for rollback.
    """
    import unreal
    library=unreal.EditorAssetLibrary
    if not destination.startswith(character.get_path_name().rsplit('/',1)[0]+'/Maps/'):
        raise ValueError('Eye texture destination must belong to the linked character Maps folder')
    if set(maps)!={'eye_left','eye_right'}:raise ValueError('Supply both source eyes')
    for files in maps.values():
        if set(files)!={COLOR,FINISH} or any(not Path(path).is_file() for path in files.values()):
            raise ValueError('Supply saved source-eye color and finish images')
    previous=preserve(character,blueprint)
    if len(previous['replaced_slots'])!=2:raise ValueError('Both eyes must use the explicitly owned source-eye adapter')
    snapshots=[];imported=[]
    try:
        for row in previous['replaced_slots']:
            source=unreal.load_asset(row['source'])
            snapshots.append((source,[v.copy() for v in source.get_editor_property('texture_parameter_values')]))
            for parameter,path in maps[row['role']].items():
                filename=Path(path).resolve();name=row['role']+('_Color' if parameter==COLOR else '_Finish')
                target=destination+'/'+name+'.'+name
                if library.does_asset_exist(target):raise ValueError('Eye refresh destination already exists: '+target)
                task=unreal.AssetImportTask()
                for key,value in (('filename',str(filename)),('destination_path',destination),('destination_name',name),('automated',True),('save',False),('replace_existing',False)):
                    task.set_editor_property(key,value)
                unreal.AssetToolsHelpers.get_asset_tools().import_asset_tasks([task])
                if list(task.get_editor_property('imported_object_paths'))!=[target]:raise RuntimeError('Unexpected imported eye map destination')
                texture=unreal.load_asset(target)
                if texture.blueprint_get_size_x()!=2048 or texture.blueprint_get_size_y()!=2048:raise ValueError('Source-eye maps must retain their 2K native UV chart')
                color=parameter==COLOR;texture.set_editor_property('srgb',color)
                texture.set_editor_property('compression_settings',unreal.TextureCompressionSettings.TC_BC7 if color else unreal.TextureCompressionSettings.TC_MASKS)
                library.set_metadata_tag(texture,'CreateMetaHuman.TextureOwner',character.get_path_name())
                unreal.MaterialEditingLibrary.set_material_instance_texture_parameter_value(source,parameter,texture)
                imported.append(dict(role=row['role'],parameter=parameter,asset=target,source=str(filename),sha256=hashlib.sha256(filename.read_bytes()).hexdigest()))
        result=preserve(character,blueprint,refresh_textures=True)
        for source,_ in snapshots:
            if not library.save_loaded_asset(source,False):raise RuntimeError('Could not save refreshed Creator eye material')
        for row in imported:
            if not library.save_asset(row['asset'],False):raise RuntimeError('Could not save refreshed eye texture')
        return dict(status='complete',updated=imported,eye_material_policy=result,geometry_changed=False,rig_changed=False)
    except Exception:
        for source,parameters in snapshots:
            source.set_editor_property('texture_parameter_values',parameters);library.save_loaded_asset(source,False)
        # An assembly refresh may have completed before a later asset-save
        # failure. Restore the old two texture bindings and receipt as well.
        for row in previous['replaced_slots']:
            material=unreal.load_asset(row['material'])
            for parameter,path in row['textures'].items():
                unreal.MaterialEditingLibrary.set_material_instance_texture_parameter_value(material,parameter,unreal.load_asset(path))
            library.save_loaded_asset(material,False)
        library.set_metadata_tag(blueprint,ASSEMBLED_TAG,json.dumps(previous));library.save_loaded_asset(blueprint,False)
        raise


def preserve(character,blueprint,*,refresh_textures=False):
    import unreal
    library=unreal.EditorAssetLibrary
    raw=library.get_metadata_tag(character,TAG)
    if not raw:return dict(policy='NATIVE_EYES',replaced_slots=[])
    ledger=json.loads(raw)
    if ledger.get('schema')!=1 or ledger.get('owner')!=character.get_path_name():
        raise ValueError('Source-eye ownership does not match this character')
    overrides=character.get_editor_property('skin_settings').get_editor_property('texture_material_overrides')
    if not overrides.get_editor_property('enable_material_overrides'):
        return dict(policy='NATIVE_EYES',replaced_slots=[],reason='Source-eye overrides are disabled')
    values=dict(overrides.get_editor_property('material_overrides').get_editor_property('teeth_and_eyes'))
    selected=[]
    for role,key,slot in (
        ('eye_left',unreal.MetaHumanCharacterTeethAndEyesSlot.EYE_LEFT,'eyeLeft_shader_shader'),
        ('eye_right',unreal.MetaHumanCharacterTeethAndEyesSlot.EYE_RIGHT,'eyeRight_shader_shader')):
        material=values.get(key)
        # An artist may have replaced one of our overrides with an optical eye.
        # Leave that eye on the native pipeline instead of restoring old work.
        if not material or material.get_path_name()!=ledger.get('applied',{}).get(role):continue
        if not isinstance(material,unreal.MaterialInstanceConstant):
            raise ValueError('The source-eye adapter needs its saved material instance')
        textures={name:unreal.MaterialEditingLibrary.get_material_instance_texture_parameter_value(material,name)
                  for name in (COLOR,FINISH)}
        if not all(textures.values()):raise ValueError('Source-eye color/finish is missing; retain the authored eye maps before assembly')
        selected.append((role,slot,material,textures))
    if not selected:return dict(policy='NATIVE_EYES',replaced_slots=[],reason='Source-eye overrides were replaced by the artist')
    subsystem=unreal.get_engine_subsystem(unreal.SubobjectDataSubsystem)
    handles=subsystem.k2_gather_subobject_data_for_blueprint(blueprint)
    data=unreal.SubobjectDataBlueprintFunctionLibrary
    components=[]
    for handle in handles:
        component=data.get_object(data.get_data(handle))
        if isinstance(component,unreal.SkeletalMeshComponent) and component.get_name().removesuffix('_GEN_VARIABLE')=='Face':
            if component not in components:components.append(component)
    meshes={component.get_skeletal_mesh_asset() for component in components}
    if len(meshes)!=1 or None in meshes:raise ValueError('Assembly has no unambiguous Face mesh for its source-eye shaders')
    mesh=next(iter(meshes));slots=list(mesh.get_editor_property('materials'))
    names=[str(slot.get_editor_property('material_slot_name')) for slot in slots]
    root=blueprint.get_path_name().rsplit('/',1)[0]
    if not mesh.get_path_name().startswith(root+'/'):raise ValueError('The assembled Face mesh must belong to this build')
    for _,slot,_,_ in selected:
        if names.count(slot)!=1:raise ValueError('The assembled Face mesh has an ambiguous eye slot: '+slot)
    previous_slots=[slot.copy() for slot in slots]
    previous_mesh=[slot.get_editor_property('material_interface') for slot in previous_slots]
    previous_components=[(component,list(component.get_editor_property('override_materials'))) for component in components]
    previous_tag=library.get_metadata_tag(blueprint,ASSEMBLED_TAG)
    previous_parameters=[]
    report=dict(policy='PRESERVE_SOURCE_EYE_PBR',character=character.get_path_name(),mesh=mesh.get_path_name(),replaced_slots=[])
    try:
        # Reuse verified output instances on a repeated repair. Reject edited
        # output ownership instead of overwriting an artist's later changes.
        existing=json.loads(previous_tag) if previous_tag else None
        if existing and existing.get('character')!=character.get_path_name():raise ValueError('Assembled eye history belongs to another character')
        for role,slot,source,textures in selected:
            index=names.index(slot)
            recorded=next((row for row in (existing or {}).get('replaced_slots',[]) if row['role']==role),None)
            if recorded:
                material=unreal.load_asset(recorded['material'])
                if not material or previous_mesh[index]!=material or recorded['source']!=source.get_path_name():
                    raise ValueError('Assembled source eyes were edited; retain those artist edits before repairing again')
                changed=False
                for name,texture in textures.items():
                    actual_texture=unreal.MaterialEditingLibrary.get_material_instance_texture_parameter_value(material,name)
                    if actual_texture!=texture:
                        if (not refresh_textures or not actual_texture
                                or actual_texture.get_path_name()!=recorded.get('textures',{}).get(name)):
                            raise ValueError('Assembled eye textures differ from their owned history; retain those artist edits first')
                        changed=True
                if changed:
                    if material.get_editor_property('parent')!=source.get_editor_property('parent'):
                        raise ValueError('The assembled eye shader was edited')
                    for field in ('scalar_parameter_values','vector_parameter_values'):
                        if [v.export_text() for v in material.get_editor_property(field)]!=[v.export_text() for v in source.get_editor_property(field)]:
                            raise ValueError('Retain artist eye parameter edits before refreshing source maps')
                    # Only the adapter's two explicitly owned maps may change.
                    unrelated=lambda obj:[v.export_text() for v in obj.get_editor_property('texture_parameter_values')
                        if str(v.get_editor_property('parameter_info').get_editor_property('name')) not in (COLOR,FINISH)]
                    if unrelated(material)!=unrelated(source):raise ValueError('An unrelated eye texture was edited')
                    previous_parameters.append((material,[v.copy() for v in material.get_editor_property('texture_parameter_values')]))
                    material.set_editor_property('texture_parameter_values',[v.copy() for v in source.get_editor_property('texture_parameter_values')])
            else:
                destination=root+'/Face/Materials/MI_Source_'+role
                if library.does_asset_exist(destination):
                    material=unreal.load_asset(destination)
                    if library.get_metadata_tag(material,SOURCE_TAG)!=source.get_path_name():
                        raise ValueError('Source-eye assembly material already exists without ownership: '+destination)
                else:
                    material=library.duplicate_asset(source.get_path_name(),destination)
                    if material:library.set_metadata_tag(material,SOURCE_TAG,source.get_path_name())
                if not isinstance(material,unreal.MaterialInstanceConstant):raise RuntimeError('Could not preserve the authored eye material')
                updated=[value.copy() for value in mesh.get_editor_property('materials')]
                updated[index].set_editor_property('material_interface',material)
                mesh.set_editor_property('materials',updated)
                for component in components:component.set_material(index,material)
            if material.get_editor_property('parent')!=source.get_editor_property('parent'):
                raise ValueError('The assembled eye shader was edited; retain those changes first')
            for field in ('texture_parameter_values','scalar_parameter_values','vector_parameter_values'):
                if [value.export_text() for value in material.get_editor_property(field)]!=[value.export_text() for value in source.get_editor_property(field)]:
                    raise ValueError('The assembled eye parameters were edited; retain those changes first')
            if any(component.get_material(index)!=material for component in components):raise RuntimeError('Assembled face component does not use the authored eye shader')
            for name,texture in textures.items():
                if unreal.MaterialEditingLibrary.get_material_instance_texture_parameter_value(material,name)!=texture:
                    raise RuntimeError('An authored source-eye texture was lost')
            report['replaced_slots'].append(dict(role=role,slot=slot,index=index,source=source.get_path_name(),material=material.get_path_name(),
                previous=recorded['previous'] if recorded else (previous_mesh[index].get_path_name() if previous_mesh[index] else None),
                textures={name:value.get_path_name() for name,value in textures.items()}))
            if not library.save_loaded_asset(material,False):raise RuntimeError('Could not save assembled source-eye material')
        selected_indices={row['index'] for row in report['replaced_slots']}
        actual=list(mesh.get_editor_property('materials'))
        if any(actual[index].get_editor_property('material_interface')!=material for index,material in enumerate(previous_mesh) if index not in selected_indices):
            raise RuntimeError('An unrelated assembled material changed')
        library.set_metadata_tag(blueprint,ASSEMBLED_TAG,json.dumps(report))
        unreal.BlueprintEditorLibrary.compile_blueprint(blueprint)
        if not blueprint.generated_class() or blueprint.get_editor_property('status') not in {
                unreal.BlueprintStatus.BS_UP_TO_DATE,unreal.BlueprintStatus.BS_UP_TO_DATE_WITH_WARNINGS}:
            raise RuntimeError('Source-eye actor failed to compile')
        if not library.save_loaded_asset(mesh,False) or not library.save_loaded_asset(blueprint,False):raise RuntimeError('Could not save assembled eye bindings')
        return report
    except Exception:
        for material,parameters in previous_parameters:
            material.set_editor_property('texture_parameter_values',parameters)
            library.save_loaded_asset(material,False)
        mesh.set_editor_property('materials',previous_slots)
        for component,materials in previous_components:component.set_editor_property('override_materials',materials)
        library.set_metadata_tag(blueprint,ASSEMBLED_TAG,previous_tag)
        unreal.BlueprintEditorLibrary.compile_blueprint(blueprint)
        library.save_loaded_asset(mesh,False);library.save_loaded_asset(blueprint,False)
        raise
