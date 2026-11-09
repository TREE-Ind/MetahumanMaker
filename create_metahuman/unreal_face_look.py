"""Private live skin response previews for the assembled-face review session."""
import json,math,runpy,uuid
from pathlib import Path

NORMAL='Normal Global Strength Post-Bake'
MICRO='Micro Skin Normal Strength'


def face_component(unreal,actor):
    return next(c for c in actor.get_components_by_class(unreal.SkeletalMeshComponent) if c.get_name()=='Face')


def uniforms(unreal,material):
    lib=unreal.MaterialEditingLibrary;dynamic=isinstance(material,unreal.MaterialInstanceDynamic);base=isinstance(material,unreal.Material)
    def scalar(name):return material.get_scalar_parameter_value(name) if dynamic else lib.get_material_default_scalar_parameter_value(material,name) if base else lib.get_material_instance_scalar_parameter_value(material,name)
    def vector(name):
        v=material.get_vector_parameter_value(name) if dynamic else lib.get_material_default_vector_parameter_value(material,name) if base else lib.get_material_instance_vector_parameter_value(material,name)
        return [float(v.r),float(v.g),float(v.b),float(v.a)]
    def texture(name):
        t=material.get_texture_parameter_value(name) if dynamic else lib.get_material_default_texture_parameter_value(material,name) if base else lib.get_material_instance_texture_parameter_value(material,name)
        return t.get_path_name() if t else None
    return dict(scalars={str(n):scalar(n) for n in lib.get_scalar_parameter_names(material)},vectors={str(n):vector(n) for n in lib.get_vector_parameter_names(material)},textures={str(n):texture(n) for n in lib.get_texture_parameter_names(material)})


def stable(values):
    return {kind:{key:value for key,value in rows.items() if not key.startswith(('head_wm','head_cm'))} for kind,rows in values.items()}


def info(unreal,actor):
    face=face_component(unreal,actor);names=list(map(str,face.get_material_slot_names()))
    if names.count('head_shader_shader')!=1:return None
    index=names.index('head_shader_shader');material=face.get_material(index)
    if not material:return None
    values=uniforms(unreal,material)
    if NORMAL not in values['scalars']:return None
    return dict(index=index,material=material.get_path_name(),normal=values['scalars'][NORMAL],micro=values['scalars'].get(MICRO),uniforms=values)


def objects(unreal):
    if not hasattr(unreal,'_cm_face_look_objects'):unreal._cm_face_look_objects={}
    return unreal._cm_face_look_objects


def guard(unreal,actor,state):
    look=state.get('look')
    if not look:return
    actual=info(unreal,actor)
    if state['origin'] not in objects(unreal) or not actual or actual['material']!=look['preview_material'] or stable(actual['uniforms'])!=look['expected_uniforms']:raise ValueError('Skin material changed outside this review; retain the later artist edits before restoring or saving')


def checkpoint(unreal,actor,state):
    face=face_component(unreal,actor)
    return dict(overrides=list(face.get_editor_property('override_materials')),look=json.loads(json.dumps(state.get('look'))),held=objects(unreal).get(state['origin']),info=info(unreal,actor))


def rollback(unreal,actor,state,snapshot):
    face=face_component(unreal,actor);face.set_editor_property('override_materials',snapshot['overrides']);state['look']=snapshot['look']
    if snapshot['held']:
        objects(unreal)[state['origin']]=snapshot['held'];material=snapshot['held']['preview']
        for key in [NORMAL,MICRO]:
            value=snapshot['info']['uniforms']['scalars'].get(key)
            if value is not None:material.set_scalar_parameter_value(key,value)
    else:objects(unreal).pop(state['origin'],None)


def restore(unreal,actor,state):
    if not state.get('look'):return
    guard(unreal,actor,state);look=state['look'];held=objects(unreal)[state['origin']];face=face_component(unreal,actor);face.set_material(look['index'],held['before_override']);values=list(face.get_editor_property('override_materials'))
    while len(values)>look['before_array_length'] and values[-1] is None:values.pop()
    face.set_editor_property('override_materials',values);state['look']=None;objects(unreal).pop(state['origin'],None)


def apply(unreal,actor,state,values):
    if values is None:restore(unreal,actor,state);return
    if not isinstance(values,dict) or set(values)-{'normal','micro'} or 'normal' not in values or any(type(v) not in (int,float) or not math.isfinite(v) or not 0<=v<=(2 if k=='normal' else 4) for k,v in values.items()):raise ValueError('Choose surface detail from 0 to 2 and micro skin from 0 to 4')
    current=info(unreal,actor)
    if not current or ('micro' in values and current['micro'] is None):raise ValueError('This head shader does not expose the requested native skin controls')
    if not state.get('look'):
        face=face_component(unreal,actor);original=face.get_material(current['index']);parent=original
        while isinstance(parent,unreal.MaterialInstanceDynamic):parent=parent.get_editor_property('parent')
        material=unreal.MaterialLibrary.create_dynamic_material_instance(actor.get_world(),parent)
        if not material:raise RuntimeError('Could not create a private skin preview')
        for key,value in current['uniforms']['scalars'].items():material.set_scalar_parameter_value(key,value)
        for key,value in current['uniforms']['vectors'].items():material.set_vector_parameter_value(key,unreal.LinearColor(*value))
        for key,value in current['uniforms']['textures'].items():
            if value:material.set_texture_parameter_value(key,unreal.load_asset(value) or unreal.find_object(None,value))
        before=list(face.get_editor_property('override_materials'));objects(unreal)[state['origin']]=dict(before_material=original,before_override=before[current['index']] if current['index']<len(before) else None,preview=material)
        state['look']=dict(index=current['index'],before_array_length=len(before),before_uniforms=stable(current['uniforms']),preview_material=material.get_path_name());face.set_material(current['index'],material)
    material=objects(unreal)[state['origin']]['preview'];material.set_scalar_parameter_value(NORMAL,float(values['normal']))
    if 'micro' in values:material.set_scalar_parameter_value(MICRO,float(values['micro']))
    state['look']['expected_uniforms']=stable(uniforms(unreal,material));state['look']['values']=dict(values)


def blueprint(unreal,actor):
    cls=actor.get_class();path=cls.get_path_name()
    bp=unreal.load_asset(path.removesuffix('_C')) if path.startswith('/Game/') and path.endswith('_C') else None
    if not isinstance(bp,unreal.Blueprint) or bp.generated_class()!=cls:raise ValueError('Saving skin tuning requires a saved project character blueprint')
    return bp


def saved_preflight(unreal,actor,state):
    guard(unreal,actor,state)
    if not state.get('look'):raise ValueError('Preview a skin adjustment before saving')
    bp=blueprint(unreal,actor);helper=runpy.run_path(str(Path(__file__).with_name('unreal_skin_normal_response.py')));face,index=helper['_face'](unreal,bp);material=face.get_material(index)
    if not isinstance(material,unreal.MaterialInstanceConstant) or stable(uniforms(unreal,material))!=state['look']['before_uniforms']:raise ValueError('The actor has separate skin edits; save those to its blueprint before saving this tuning')
    if bp.get_path_name().split('.')[0] in {p.get_path_name() for p in unreal.EditorLoadingAndSavingUtils.get_dirty_content_packages()}:raise ValueError('Save the character blueprint edits before saving skin tuning')
    return bp,material.get_path_name(),dict(state['look']['values'])


def save(unreal,bp,source,values):
    root=bp.get_path_name().split('.')[0].rsplit('/',1)[0];destination=root+'/Materials/MI_SkinLook_'+uuid.uuid4().hex[:12]
    return runpy.run_path(str(Path(__file__).with_name('unreal_skin_normal_response.py')))['apply'](bp,values['normal'],micro_strength=values.get('micro'),expected_material=source,destination=destination,retain_previous=True)


def undo(unreal,bp):return runpy.run_path(str(Path(__file__).with_name('unreal_skin_normal_response.py')))['restore'](bp)


def undo_preflight(unreal,actor,bp):
    helper=runpy.run_path(str(Path(__file__).with_name('unreal_skin_normal_response.py')));template,index=helper['_face'](unreal,bp);current=info(unreal,actor)
    if not current or stable(current['uniforms'])!=stable(uniforms(unreal,template.get_material(index))):raise ValueError('Retain the actor\'s separate skin edits before undoing saved tuning')
    if bp.get_path_name().split('.')[0] in {p.get_path_name() for p in unreal.EditorLoadingAndSavingUtils.get_dirty_content_packages()}:raise ValueError('Save the character blueprint edits before undoing skin tuning')


def attach_saved(unreal,actor,bp):
    helper=runpy.run_path(str(Path(__file__).with_name('unreal_skin_normal_response.py')));template,index=helper['_face'](unreal,bp);face_component(unreal,actor).set_material(index,template.get_material(index))


def undo_available(unreal,actor):
    try:return bool(unreal.EditorAssetLibrary.get_metadata_tag(blueprint(unreal,actor),'CreateMetaHuman.SkinNormalResponse'))
    except ValueError:return False
