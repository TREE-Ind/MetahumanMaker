"""Reversible optical-overlay policy for assembled opaque source eyes.

The native optical shell can paint an artificial occlusion band onto a
non-spherical opaque source eye. Its tearline can also overlap a stylized lid.
Keep both geometries and all grooms; an explicit compatibility policy changes
only the assembled Face component's two material overrides.
"""
import json

TAG='CreateMetaHuman.SourceEyeOverlayPolicy'
SLOTS=('eyeshell_shader_shader','eyeEdge_shader_shader')


def require_projected_eyes(character):
    """Fail before starting an assembly if the requested source adapter is inactive."""
    import unreal
    raw=unreal.EditorAssetLibrary.get_metadata_tag(character,'CreateMetaHuman.SourceEyeAppearance')
    ledger=json.loads(raw) if raw else {}
    override=character.get_editor_property('skin_settings').get_editor_property('texture_material_overrides')
    materials=dict(override.get_editor_property('material_overrides').get_editor_property('teeth_and_eyes'))
    if ledger.get('schema')!=1 or ledger.get('owner')!=character.get_path_name() or not override.get_editor_property('enable_material_overrides'):
        raise ValueError('Apply projected source eyes before choosing opaque source overlays, or keep native overlays')
    for role,key in [('eye_left',unreal.MetaHumanCharacterTeethAndEyesSlot.EYE_LEFT),('eye_right',unreal.MetaHumanCharacterTeethAndEyesSlot.EYE_RIGHT)]:
        material=materials.get(key)
        if not material or material.get_path_name()!=ledger.get('applied',{}).get(role):raise ValueError('Both source-eye overrides must be active for opaque source overlays')


def hidden_material_for(blueprint):
    """Reuse the native assembly's hidden shader; never create a replacement graph."""
    import unreal
    for face in _faces(unreal,blueprint):
        for index in range(len(face.get_material_slot_names())):
            material=face.get_material(index)
            if material and material.get_name().lower()=='m_hide':return material
    candidate=unreal.load_asset(blueprint.get_path_name().rsplit('/',1)[0]+'/Common/Materials/M_Hide')
    if not candidate:raise ValueError('The assembly has no native hidden material for source-eye compatibility')
    return candidate


def _faces(unreal, blueprint):
    subsystem=unreal.get_engine_subsystem(unreal.SubobjectDataSubsystem);data=unreal.SubobjectDataBlueprintFunctionLibrary;result=[]
    for handle in subsystem.k2_gather_subobject_data_for_blueprint(blueprint):
        component=data.get_object(data.get_data(handle))
        if isinstance(component,unreal.SkeletalMeshComponent) and component.get_name().removesuffix('_GEN_VARIABLE')=='Face' and component not in result:result.append(component)
    if not result:raise ValueError('Choose an assembled MetaHuman blueprint with a Face component')
    return result


def set_policy(blueprint, policy, *, hidden_material=None):
    """Apply OPAQUE_SOURCE or restore NATIVE without overwriting artist edits.

    The caller supplies a genuine hidden material when applying compatibility.
    This edits/saves only the selected project blueprint, never a mesh, groom,
    shared material, skeleton or Creator character. Compilation may replace
    live actor components; reacquire them before further preview operations.
    """
    import unreal
    if not isinstance(blueprint,unreal.Blueprint) or not blueprint.get_path_name().startswith('/Game/') or policy not in {'OPAQUE_SOURCE','NATIVE'}:
        raise ValueError('Choose a saved project blueprint and a supported eye overlay policy')
    library=unreal.EditorAssetLibrary;owner=blueprint.get_path_name();previous_tag=library.get_metadata_tag(blueprint,TAG);ledger=json.loads(previous_tag) if previous_tag else None;faces=_faces(unreal,blueprint)
    if ledger and (ledger.get('schema')!=1 or ledger.get('owner')!=owner):raise ValueError('Eye overlay history belongs to another assembly')
    if policy=='NATIVE' and not ledger:return dict(status='unchanged',policy='NATIVE',blueprint=owner)
    if policy=='OPAQUE_SOURCE' and not isinstance(hidden_material,unreal.MaterialInterface):raise ValueError('Supply a native hidden material for optical overlay compatibility')
    snapshots=[(face,list(face.get_editor_property('override_materials'))) for face in faces]
    changes=[]
    for face,values in snapshots:
        names=[str(name) for name in face.get_material_slot_names()]
        if policy=='OPAQUE_SOURCE':
            for eye_slot in ('eyeLeft_shader_shader','eyeRight_shader_shader'):
                eye=face.get_material(names.index(eye_slot))
                if not eye or not all(unreal.MaterialEditingLibrary.get_material_instance_texture_parameter_value(eye,p) for p in ('CM Source Eye Color','CM Source Eye Finish')):
                    raise ValueError('Opaque overlay compatibility requires both projected source-eye PBR materials')
        for slot in SLOTS:
            index=names.index(slot);current=values[index] if index<len(values) else None;record=None
            if ledger:
                matches=[row for row in ledger['entries'] if row['component']==face.get_name() and row['slot']==slot]
                if len(matches)!=1:raise ValueError('The eye overlay component layout changed; retain its current state')
                record=matches[0]
                if index!=record['index'] or not current or current.get_path_name()!=record['applied']:
                    raise ValueError('An artist changed an eye overlay after compatibility was applied; retain that edit before restoring')
            if policy=='OPAQUE_SOURCE':
                if record and hidden_material.get_path_name()!=record['applied']:raise ValueError('Restore the previous overlay policy before changing its hidden material')
                entry=record or dict(component=face.get_name(),slot=slot,index=index,previous=current.get_path_name() if current else None,previous_array_length=len(values),applied=hidden_material.get_path_name())
                changes.append((face,index,hidden_material,entry))
            else:
                material=unreal.load_asset(record['previous']) if record['previous'] else None
                if record['previous'] and not material:raise ValueError('A previous optical overlay material is unavailable')
                changes.append((face,index,material,record))
    if policy=='OPAQUE_SOURCE' and ledger:return dict(status='unchanged',policy=policy,blueprint=owner)
    try:
        for face,index,material,_ in changes:face.set_material(index,material)
        if policy=='NATIVE':
            for face,old in snapshots:
                values=list(face.get_editor_property('override_materials'));length=min(row['previous_array_length'] for c,_,_,row in changes if c==face)
                while len(values)>length and values[-1] is None:values.pop()
                face.set_editor_property('override_materials',values)
            library.remove_metadata_tag(blueprint,TAG)
        else:
            library.set_metadata_tag(blueprint,TAG,json.dumps(dict(schema=1,owner=owner,policy=policy,entries=[row for _,_,_,row in changes])))
        unreal.BlueprintEditorLibrary.compile_blueprint(blueprint)
        if blueprint.get_editor_property('status') not in {unreal.BlueprintStatus.BS_UP_TO_DATE,unreal.BlueprintStatus.BS_UP_TO_DATE_WITH_WARNINGS} or not library.save_loaded_asset(blueprint,False):raise RuntimeError('Could not save the source-eye overlay policy')
    except Exception:
        for face,values in snapshots:face.set_editor_property('override_materials',values)
        if previous_tag:library.set_metadata_tag(blueprint,TAG,previous_tag)
        else:library.remove_metadata_tag(blueprint,TAG)
        unreal.BlueprintEditorLibrary.compile_blueprint(blueprint);library.save_loaded_asset(blueprint,False)
        raise
    return dict(status='complete',policy=policy,blueprint=owner,changed_slots=list(SLOTS),geometry_preserved=True,grooms_preserved=True,source_materials_modified=0,actor_components_reinstanced=True)
