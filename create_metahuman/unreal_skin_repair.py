"""Reversible assembled skin color revisions that retain the artist shader.

No geometry, animation, shared material or Creator assets are changed. The
caller supplies a reviewed texture and a new destination for the copied MI.
"""
import json

TAG='CreateMetaHuman.SkinColorRevision'
PARAMETER='Basecolor Baked VT'


def _face(unreal,blueprint):
    subsystem=unreal.get_engine_subsystem(unreal.SubobjectDataSubsystem);library=unreal.SubobjectDataBlueprintFunctionLibrary;found=[]
    for handle in subsystem.k2_gather_subobject_data_for_blueprint(blueprint):
        obj=library.get_object(library.get_data(handle))
        if isinstance(obj,unreal.SkeletalMeshComponent) and obj.get_name().removesuffix('_GEN_VARIABLE')=='Face' and obj not in found:found.append(obj)
    if len(found)!=1:raise ValueError('Choose an assembled blueprint with exactly one Face component')
    face=found[0];names=[str(n) for n in face.get_material_slot_names()]
    if names.count('head_shader_shader')!=1:raise ValueError('The assembled head material layout is unsupported')
    return face,names.index('head_shader_shader')


def apply(blueprint,texture,*,expected_texture,destination,retain_previous=False):
    """Copy the current constant MI and change only its baked color texture.

    expected_texture is a stale-edit guard. Destination must be a new project
    asset. The original override and array length are retained for restore.
    Blueprint compilation can replace live components; reacquire for preview.
    retain_previous explicitly keeps the current artist material as the next
    undo baseline. Matching earlier repair history remains available in order.
    """
    import unreal
    lib=unreal.EditorAssetLibrary
    if not isinstance(blueprint,unreal.Blueprint) or not blueprint.get_path_name().startswith('/Game/') or not isinstance(texture,unreal.Texture2D) or not isinstance(expected_texture,unreal.Texture2D):
        raise ValueError('Choose a project blueprint and reviewed color textures')
    if not destination.startswith('/Game/') or '.' in destination or lib.does_asset_exist(destination):raise ValueError('Supply a new project material destination')
    old_history=lib.get_metadata_tag(blueprint,TAG)
    if old_history and not retain_previous:raise ValueError('Restore or explicitly retain the previous skin color revision before replacing it')
    face,index=_face(unreal,blueprint);current=face.get_material(index)
    if not isinstance(current,unreal.MaterialInstanceConstant):raise ValueError('The assembled skin needs a saved constant material instance')
    if unreal.MaterialEditingLibrary.get_material_instance_texture_parameter_value(current,PARAMETER)!=expected_texture:raise ValueError('The assembled skin changed since repair preparation')
    previous_revision=None;archived_revision=None
    if old_history:
        previous=json.loads(old_history)
        if previous.get('schema')!=1:raise ValueError('The previous skin revision history is unsupported')
        if previous.get('applied')==current.get_path_name() and previous.get('index')==index and previous.get('component')==face.get_name():
            previous_revision=previous
            if previous_revision.get('owner')!=blueprint.get_path_name():previous_revision['retained_from_owner']=previous_revision.get('owner');previous_revision['owner']=blueprint.get_path_name()
        else:archived_revision=previous  # Retain later artist edits as the new baseline.
    if texture.blueprint_get_size_x()!=expected_texture.blueprint_get_size_x() or texture.blueprint_get_size_y()!=expected_texture.blueprint_get_size_y():raise ValueError('The repaired color must retain the original atlas dimensions')
    for key in ('srgb','virtual_texture_streaming'):
        if texture.get_editor_property(key)!=expected_texture.get_editor_property(key):raise ValueError('The repaired color has incompatible '+key)
    old=list(face.get_editor_property('override_materials'));copy=lib.duplicate_asset(current.get_path_name(),destination)
    if not copy:raise RuntimeError('Could not preserve the current artist skin material')
    unreal.MaterialEditingLibrary.set_material_instance_texture_parameter_value(copy,PARAMETER,texture)
    if unreal.MaterialEditingLibrary.get_material_instance_texture_parameter_value(copy,PARAMETER)!=texture:raise RuntimeError('The copied skin did not accept its repaired color')
    record=dict(schema=1,owner=blueprint.get_path_name(),component=face.get_name(),index=index,previous=old[index].get_path_name() if index<len(old) and old[index] else None,previous_array_length=len(old),applied=copy.get_path_name(),source_material=current.get_path_name(),previous_texture=expected_texture.get_path_name(),texture=texture.get_path_name())
    if previous_revision is not None:record['previous_revision']=previous_revision
    if archived_revision is not None:record['archived_revision']=archived_revision
    if not lib.save_loaded_asset(copy,False):raise RuntimeError('Could not save the private skin material')
    try:
        face.set_material(index,copy);lib.set_metadata_tag(blueprint,TAG,json.dumps(record));unreal.BlueprintEditorLibrary.compile_blueprint(blueprint)
        if not lib.save_loaded_asset(blueprint,False):raise RuntimeError('Could not save the selected assembly')
    except Exception:
        face,index=_face(unreal,blueprint);face.set_editor_property('override_materials',old)
        if old_history:lib.set_metadata_tag(blueprint,TAG,old_history)
        else:lib.remove_metadata_tag(blueprint,TAG)
        unreal.BlueprintEditorLibrary.compile_blueprint(blueprint);raise
    return dict(status='applied',**record)


def restore(blueprint):
    """Restore the exact prior slot without overwriting subsequent artist edits."""
    import unreal
    lib=unreal.EditorAssetLibrary;raw=lib.get_metadata_tag(blueprint,TAG)
    if not raw:return dict(status='unchanged')
    record=json.loads(raw);face,index=_face(unreal,blueprint)
    if record.get('schema')!=1 or record.get('owner')!=blueprint.get_path_name() or record.get('component')!=face.get_name() or index!=record.get('index'):raise ValueError('The skin repair history does not match this assembly')
    current=face.get_material(index)
    if not current or current.get_path_name()!=record['applied']:raise ValueError('Retain the later artist material change before restoring this repair')
    previous=unreal.load_asset(record['previous']) if record['previous'] else None
    if record['previous'] and not previous:raise ValueError('The previous skin material is unavailable')
    face.set_material(index,previous);values=list(face.get_editor_property('override_materials'))
    while len(values)>record['previous_array_length'] and values[-1] is None:values.pop()
    face.set_editor_property('override_materials',values)
    if record.get('previous_revision'):lib.set_metadata_tag(blueprint,TAG,json.dumps(record['previous_revision']))
    else:lib.remove_metadata_tag(blueprint,TAG)
    unreal.BlueprintEditorLibrary.compile_blueprint(blueprint)
    if not lib.save_loaded_asset(blueprint,False):raise RuntimeError('Could not save the restored assembly')
    return dict(status='restored',blueprint=blueprint.get_path_name())
