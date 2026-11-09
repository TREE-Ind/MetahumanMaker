"""Retain owned packed AO repairs and explicit hidden dentition on assembly."""
import json


def preserve(character, blueprint):
    import unreal
    assets=unreal.EditorAssetLibrary;lib=unreal.MaterialEditingLibrary
    overrides=character.get_editor_property('skin_settings').get_editor_property('texture_material_overrides')
    if not overrides.get_editor_property('enable_material_overrides'):return dict(changed=False)
    source=overrides.get_editor_property('material_overrides');textures=set()
    for material in dict(source.get_editor_property('skin')).values():
        if not material:continue
        texture=lib.get_material_instance_texture_parameter_value(material,'Bent Normal')
        if texture and assets.get_metadata_tag(texture,'CreateMetaHuman.TextureOwner')==character.get_path_name():textures.add(texture)
    if len(textures)>1:raise ValueError('Assembly needs an adapter for differing per-slot packed AO maps')
    texture=next(iter(textures),None)
    teeth=dict(source.get_editor_property('teeth_and_eyes')).get(unreal.MetaHumanCharacterTeethAndEyesSlot.TEETH)
    parent=teeth;hidden=False
    while parent:
        if parent.get_name()=='M_Hide':hidden=True;break
        parent=parent.get_editor_property('parent') if isinstance(parent,unreal.MaterialInstance) else None
    if not texture and not hidden:return dict(changed=False)
    subsystem=unreal.get_engine_subsystem(unreal.SubobjectDataSubsystem);data=unreal.SubobjectDataBlueprintFunctionLibrary
    components=[]
    for handle in subsystem.k2_gather_subobject_data_for_blueprint(blueprint):
        component=data.get_object(data.get_data(handle))
        if isinstance(component,unreal.SkeletalMeshComponent) and component.get_name().removesuffix('_GEN_VARIABLE')=='Face' and component not in components:components.append(component)
    meshes={c.get_skeletal_mesh_asset() for c in components}
    if len(meshes)!=1 or None in meshes:raise ValueError('The assembled face is ambiguous')
    mesh=next(iter(meshes));root=blueprint.get_path_name().rsplit('/',1)[0]
    if not mesh.get_path_name().startswith(root+'/'):raise ValueError('Keep auxiliary repairs within this assembled character')
    slots=list(mesh.get_editor_property('materials'));changed=[]
    for i,slot in enumerate(slots):
        name=str(slot.get_editor_property('material_slot_name'));material=slot.get_editor_property('material_interface')
        if hidden and name=='teeth_shader_shader':replacement=teeth
        elif texture and material and 'Bent Normal' in {str(n) for n in lib.get_texture_parameter_names(material)}:
            if lib.get_material_instance_texture_parameter_value(material,'Bent Normal')==texture:continue
            destination=root+'/Materials/MI_CM_Aux_'+str(i);replacement=unreal.load_asset(destination)
            if replacement:
                if assets.get_metadata_tag(replacement,'CreateMetaHuman.AuxiliaryOwner')!=character.get_path_name():raise ValueError('Retain the artist-edited auxiliary material')
            else:
                replacement=unreal.AssetToolsHelpers.get_asset_tools().duplicate_asset('MI_CM_Aux_'+str(i),root+'/Materials',material)
                if not replacement:raise RuntimeError('Could not preserve the packed AO material')
                assets.set_metadata_tag(replacement,'CreateMetaHuman.AuxiliaryOwner',character.get_path_name())
            lib.set_material_instance_texture_parameter_value(replacement,'Bent Normal',texture);assets.save_loaded_asset(replacement,False)
        else:continue
        slot.set_editor_property('material_interface',replacement)
        for component in components:component.set_material(i,replacement)
        changed.append(dict(slot=name,material=replacement.get_path_name()))
    mesh.set_editor_property('materials',slots);unreal.BlueprintEditorLibrary.compile_blueprint(blueprint)
    if not assets.save_loaded_asset(mesh,False) or not assets.save_loaded_asset(blueprint,False):raise RuntimeError('Could not save auxiliary assembly bindings')
    report=dict(changed=bool(changed),slots=changed,packed_ao=texture.get_path_name() if texture else None,hidden_dentition=hidden)
    assets.set_metadata_tag(blueprint,'CreateMetaHuman.AssembledAuxiliary',json.dumps(report));assets.save_loaded_asset(blueprint,False)
    return report
