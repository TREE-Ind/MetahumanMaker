"""Reversible normal response tuning on an assembled blueprint's private MI.

Retains native micro skin, artist textures, roughness, static permutations and
rig data. This adjusts the existing native post-bake normal response; it does
not reconstruct missing texture detail or change the shared source material.
"""
import hashlib,json,math,re,runpy
from pathlib import Path

PARAMETER='Normal Global Strength Post-Bake'
TAG='CreateMetaHuman.SkinNormalResponse'


def _face(unreal,blueprint):
    face,index=runpy.run_path(str(Path(__file__).with_name('unreal_skin_repair.py')))['_face'](unreal,blueprint)
    if not face.get_path_name().startswith(blueprint.get_path_name().split('.')[0]+'.'):raise ValueError('Create an assembly with its own Face component before tuning inherited skin')
    return face,index


def _parameters(unreal,material):
    lib=unreal.MaterialEditingLibrary
    def color(name):
        value=lib.get_material_instance_vector_parameter_value(material,name)
        return [float(value.r),float(value.g),float(value.b),float(value.a)]
    return dict(scalars={str(n):lib.get_material_instance_scalar_parameter_value(material,n) for n in lib.get_scalar_parameter_names(material)},vectors={str(n):color(n) for n in lib.get_vector_parameter_names(material)},textures={str(n):(t.get_path_name() if (t:=lib.get_material_instance_texture_parameter_value(material,n)) else None) for n in lib.get_texture_parameter_names(material)},switches={str(n):lib.get_material_instance_static_switch_parameter_value(material,n) for n in lib.get_static_switch_parameter_names(material)})


def _fingerprint(value):return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':')).encode()).hexdigest()


def apply(blueprint,strength,*,expected_material,destination,micro_strength=None,retain_previous=False):
    import unreal
    lib=unreal.EditorAssetLibrary;editing=unreal.MaterialEditingLibrary
    if not isinstance(blueprint,unreal.Blueprint) or not blueprint.get_path_name().startswith('/Game/') or not math.isfinite(strength) or not 0<=strength<=2:raise ValueError('Choose a project assembly and normal response between zero and two')
    if not re.fullmatch(r'/Game(?:/[A-Za-z0-9_]+)+',destination) or lib.does_asset_exist(destination):raise ValueError('Choose a fresh private material path')
    history=lib.get_metadata_tag(blueprint,TAG)
    if history and not retain_previous:raise ValueError('Restore or retain the previous normal response revision before replacing it')
    if micro_strength is not None and (not math.isfinite(micro_strength) or not 0<=micro_strength<=4):raise ValueError('Choose micro skin response between zero and four')
    if blueprint.get_path_name().split('.')[0] in {p.get_path_name() for p in unreal.EditorLoadingAndSavingUtils.get_dirty_content_packages()}:raise ValueError('Save the selected assembly edits before tuning its skin response')
    face,index=_face(unreal,blueprint);source=face.get_material(index)
    if not isinstance(source,unreal.MaterialInstanceConstant) or source.get_path_name()!=expected_material:raise ValueError('The saved artist skin material changed since review')
    before=_parameters(unreal,source)
    if PARAMETER not in before['scalars']:raise ValueError('This material does not expose native post-bake normal response')
    micro='Micro Skin Normal Strength'
    if micro_strength is not None and micro not in before['scalars']:raise ValueError('This material does not expose native micro skin response')
    old=list(face.get_editor_property('override_materials'));private=lib.duplicate_asset(source.get_path_name(),destination)
    if not private:raise RuntimeError('Could not retain the source skin material')
    editing.set_material_instance_scalar_parameter_value(private,PARAMETER,float(strength))
    if micro_strength is not None:editing.set_material_instance_scalar_parameter_value(private,micro,float(micro_strength))
    after=_parameters(unreal,private);expected=json.loads(json.dumps(before));expected['scalars'][PARAMETER]=after['scalars'][PARAMETER]
    if micro_strength is not None:expected['scalars'][micro]=after['scalars'][micro]
    if after!=expected or abs(after['scalars'][PARAMETER]-strength)>1e-6 or (micro_strength is not None and abs(after['scalars'][micro]-micro_strength)>1e-6):raise RuntimeError('Normal tuning changed another artist skin setting or did not accept the requested response')
    if not lib.save_loaded_asset(private,False):raise RuntimeError('Could not save the private normal response')
    record=dict(schema=1,owner=blueprint.get_path_name(),component=face.get_name(),index=index,previous=old[index].get_path_name() if index<len(old) and old[index] else None,previous_array_length=len(old),source_material=source.get_path_name(),applied=private.get_path_name(),parameter=PARAMETER,previous_strength=before['scalars'][PARAMETER],strength=after['scalars'][PARAMETER],applied_fingerprint=_fingerprint(after),other_parameters_preserved=True,geometry_changed=False)
    if micro_strength is not None:record.update(micro_strength=after['scalars'][micro],previous_micro_strength=before['scalars'][micro])
    if history:record['previous_history']=history
    try:
        face.set_material(index,private);lib.set_metadata_tag(blueprint,TAG,json.dumps(record));unreal.BlueprintEditorLibrary.compile_blueprint(blueprint)
        if not lib.save_loaded_asset(blueprint,False):raise RuntimeError('Could not save the selected normal response revision')
    except Exception:
        face,index=_face(unreal,blueprint);face.set_editor_property('override_materials',old)
        if history:lib.set_metadata_tag(blueprint,TAG,history)
        else:lib.remove_metadata_tag(blueprint,TAG)
        unreal.BlueprintEditorLibrary.compile_blueprint(blueprint);raise
    return dict(status='applied',**record)


def restore(blueprint):
    import unreal
    lib=unreal.EditorAssetLibrary;raw=lib.get_metadata_tag(blueprint,TAG)
    if not raw:return dict(status='unchanged')
    record=json.loads(raw);face,index=_face(unreal,blueprint);current=face.get_material(index)
    if record.get('schema')!=1 or record['owner']!=blueprint.get_path_name() or record['component']!=face.get_name() or record['index']!=index or not current or current.get_path_name()!=record['applied'] or _fingerprint(_parameters(unreal,current))!=record['applied_fingerprint']:raise ValueError('Retain later artist skin edits before restoring normal response')
    previous=unreal.load_asset(record['previous']) if record['previous'] else None
    if record['previous'] and not previous:raise ValueError('The retained source skin material is missing')
    face.set_material(index,previous);values=list(face.get_editor_property('override_materials'))
    while len(values)>record['previous_array_length'] and values[-1] is None:values.pop()
    face.set_editor_property('override_materials',values)
    if record.get('previous_history'):lib.set_metadata_tag(blueprint,TAG,record['previous_history'])
    else:lib.remove_metadata_tag(blueprint,TAG)
    unreal.BlueprintEditorLibrary.compile_blueprint(blueprint)
    if not lib.save_loaded_asset(blueprint,False):raise RuntimeError('Could not save restored skin response')
    return dict(status='restored',blueprint=blueprint.get_path_name())
