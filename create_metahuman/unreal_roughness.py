"""Persistent native skin material instances for projected roughness maps."""
import json
import math


RESPONSE_TAG='CreateMetaHuman.RoughnessResponse'


def response_parent(source,root,owner):
    """Copy the native inheritance chain and calibrate its final roughness UI."""
    import unreal
    lib=unreal.MaterialEditingLibrary;assets=unreal.EditorAssetLibrary
    tools=unreal.AssetToolsHelpers.get_asset_tools()
    folder=root+'/Materials/Response'
    def duplicate(original,name):
        asset=unreal.load_asset(folder+'/'+name)
        if asset:
            if (assets.get_metadata_tag(asset,'CreateMetaHuman.ResponseOwner')!=owner or
                    assets.get_metadata_tag(asset,'CreateMetaHuman.ResponseSource')!=original.get_path_name()):
                raise ValueError('Skin response asset belongs to another author: '+asset.get_path_name())
            return asset,False
        asset=tools.duplicate_asset(name,folder,original)
        if not asset:raise RuntimeError('Could not create native skin response copy')
        assets.set_metadata_tag(asset,'CreateMetaHuman.ResponseOwner',owner)
        assets.set_metadata_tag(asset,'CreateMetaHuman.ResponseSource',original.get_path_name())
        return asset,True
    def expressions(asset,cls=unreal.MaterialExpression):
        return [obj for obj in unreal.ObjectIterator(cls) if obj.get_outer()==asset]
    # Upgrade existing character-owned functions too. Returning an existing
    # parent below must not leave older characters on the multiplicative-only
    # curve, whose matte endpoint still depends on texture/cavity values.
    existing=unreal.load_asset(folder+'/CM_Skin_UI_Response')
    if existing:
        native=unreal.load_asset('/MetaHumanCharacter/Lookdev_UHM/Skin/Material_Functions/MF_skin_UI_overrides_post')
        existing,_=duplicate(native,'CM_Skin_UI_Response')
        matte_endpoint(existing)
    def clone(original):
        asset,created=duplicate(original,'CM_Response_'+original.get_name())
        if not created:return asset
        if isinstance(original,unreal.MaterialInstanceConstant):
            lib.set_material_instance_parent(asset,clone(original.get_editor_property('parent')))
        elif isinstance(original,unreal.Material):
            native=unreal.load_asset('/MetaHumanCharacter/Lookdev_UHM/Skin/Material_Functions/MF_skin_UI_overrides_post')
            function,new=duplicate(native,'CM_Skin_UI_Response')
            if new:
                nodes=expressions(function)
                parameter=next(obj for obj in nodes if isinstance(obj,unreal.MaterialExpressionScalarParameter) and
                               str(obj.get_editor_property('parameter_name'))=='Roughness UI Multiply')
                consumers=[obj for obj in nodes if parameter in lib.get_inputs_for_material_function_expression(function,obj)]
                if len(consumers)!=1 or not isinstance(consumers[0],unreal.MaterialExpressionMultiply):
                    raise ValueError('Native skin UI roughness graph is unsupported')
                gain=lib.create_material_expression_in_function(function,unreal.MaterialExpressionScalarParameter)
                gain.set_editor_property('parameter_name','CM Roughness Response Exponent');gain.set_editor_property('default_value',1.)
                power=lib.create_material_expression_in_function(function,unreal.MaterialExpressionPower)
                if not (lib.connect_material_expressions(parameter,'',power,'Base') and
                        lib.connect_material_expressions(gain,'',power,'Exp') and
                        lib.connect_material_expressions(power,'',consumers[0],'B')):
                    raise RuntimeError('Could not connect skin roughness response')
                matte_endpoint(function)
                lib.update_material_function(function)
                if not assets.save_loaded_asset(function,False):raise RuntimeError('Could not save skin response function')
            calls=[obj for obj in expressions(asset,unreal.MaterialExpressionMaterialFunctionCall)
                   if obj.get_editor_property('material_function')==native]
            if len(calls)!=1:raise ValueError('Native skin material has an unsupported UI function layout')
            if not calls[0].set_material_function(function):raise RuntimeError('Could not bind skin response function')
            lib.recompile_material(asset)
        else:raise ValueError('Unsupported native skin material parent')
        if not assets.save_loaded_asset(asset,False):raise RuntimeError('Could not save skin response parent')
        return asset
    return clone(source)


def matte_endpoint(function):
    """Preserve the glossy half; smoothly reach matte at the native UI maximum."""
    import unreal
    lib=unreal.MaterialEditingLibrary;assets=unreal.EditorAssetLibrary
    tag='CreateMetaHuman.MatteEndpoint'
    if assets.get_metadata_tag(function,tag)=='1':return
    nodes=[o for o in unreal.ObjectIterator(unreal.MaterialExpression) if o.get_outer()==function]
    parameter=next(o for o in nodes if isinstance(o,unreal.MaterialExpressionScalarParameter) and
                   str(o.get_editor_property('parameter_name'))=='Roughness UI Multiply')
    declaration=next(o for o in nodes if isinstance(o,unreal.MaterialExpressionNamedRerouteDeclaration) and
                     str(o.get_editor_property('name'))=='RoughnessModified')
    sources=[o for o in lib.get_inputs_for_material_function_expression(function,declaration) if o]
    if len(sources)!=1 or not isinstance(sources[0],unreal.MaterialExpressionMultiply):
        raise ValueError('Unsupported managed roughness endpoint graph')
    for old in nodes:
        if str(old.get_editor_property('desc')).startswith('CM matte endpoint:'):
            lib.delete_material_expression_in_function(function,old)
    def node(cls,label):
        value=lib.create_material_expression_in_function(function,cls)
        value.set_editor_property('desc','CM matte endpoint: '+label)
        return value
    subtract=node(unreal.MaterialExpressionSubtract,'upper half');subtract.set_editor_property('const_b',1.)
    divide=node(unreal.MaterialExpressionDivide,'native UI span');divide.set_editor_property('const_b',.15)
    ramp=node(unreal.MaterialExpressionSaturate,'bounded upper weight')
    ease=node(unreal.MaterialExpressionMultiply,'smooth departure from neutral')
    bounded=node(unreal.MaterialExpressionSaturate,'physical roughness')
    blend=node(unreal.MaterialExpressionLinearInterpolate,'matte endpoint');blend.set_editor_property('const_b',1.)
    connections=((parameter,subtract,'A'),(subtract,divide,'A'),(divide,ramp,''),
                 (ramp,ease,'A'),(ramp,ease,'B'),(sources[0],bounded,''),
                 (bounded,blend,'A'),(ease,blend,'Alpha'),(blend,declaration,''))
    for source,target,socket in connections:
        if not lib.connect_material_expressions(source,'',target,socket):
            raise RuntimeError('Could not connect managed matte endpoint: '+socket)
    lib.update_material_function(function)
    assets.set_metadata_tag(function,tag,'1')
    if not assets.save_loaded_asset(function,False):raise RuntimeError('Could not save matte response endpoint')


def calibrate_response(material,settings):
    """Let inferred texture variation modulate the native regional skin finish."""
    import unreal
    lib=unreal.MaterialEditingLibrary
    mode=settings.get('mode','ABSOLUTE');center=float(settings.get('center',1))
    if mode not in {'ABSOLUTE','SKIN_DETAIL'} or not math.isfinite(center) or not .02<=center<=1:
        raise ValueError('Invalid skin roughness response')
    raw=unreal.EditorAssetLibrary.get_metadata_tag(material,RESPONSE_TAG)
    prior=json.loads(raw) if raw else {'schema':1,'parameters':{}}
    if prior.get('schema')!=1:raise ValueError('Unsupported managed skin roughness response')
    names={str(name) for name in lib.get_scalar_parameter_names(material)}
    finish=settings.get('finish_roughness')
    finish_scale=1.
    if mode=='SKIN_DETAIL' and finish is not None:
        finish=float(finish)
        if not math.isfinite(finish) or not .1<=finish<=.9:
            raise ValueError('Skin finish roughness must be between 0.1 and 0.9')
        base=lib.get_material_instance_scalar_parameter_value(material,'Roughness Base')
        if not math.isfinite(base) or base<=.01:raise ValueError('Native skin roughness base is unavailable')
        finish_scale=finish/base
    records={}
    candidates=sorted(name for name in names if name.startswith(('Roughness Map Face ','Roughness Map Body')))
    for name in candidates:
        native=name.replace('Roughness Map ','Roughness ',1)
        if native not in names:continue
        current=lib.get_material_instance_scalar_parameter_value(material,name)
        old=prior['parameters'].get(name)
        baseline=old['baseline'] if old else current
        # Retain edits to an already managed regional multiplier as an artist
        # factor, rather than multiplying last delivery's calibration again.
        if old and not math.isclose(current,old['managed'],rel_tol=1e-5,abs_tol=1e-6):
            factor=old.get('factor',1)
            baseline=current/factor if abs(factor)>1e-6 else current
        factor=lib.get_material_instance_scalar_parameter_value(material,native)*finish_scale/center if mode=='SKIN_DETAIL' else 1
        desired=baseline*factor
        if not math.isfinite(desired) or desired<0:raise ValueError('Invalid authored roughness region: '+name)
        if mode=='SKIN_DETAIL' or old:
            lib.set_material_instance_scalar_parameter_value(material,name,desired)
            records[name]={'baseline':baseline,'managed':desired,'factor':factor}
    if mode=='SKIN_DETAIL' and not records:raise ValueError('Native material has no skin roughness regions')
    value={'schema':1,'mode':mode,'center':center,'parameters':records}
    if mode=='SKIN_DETAIL' and finish is not None:value['finish_roughness']=finish
    value['ui_response_exponent']=5 if mode=='SKIN_DETAIL' and finish is not None else 1
    if mode=='SKIN_DETAIL' and finish is not None:value['ui_matte_endpoint']=1
    unreal.EditorAssetLibrary.set_metadata_tag(material,RESPONSE_TAG,json.dumps(value,sort_keys=True))
    return value


def validate_refresh(material,response=None):
    """Require an existing managed skin shader; never replace an artist graph."""
    import unreal
    if not isinstance(material,unreal.MaterialInstanceConstant):raise ValueError('Head roughness refresh needs a managed material instance')
    owner=unreal.EditorAssetLibrary.get_metadata_tag(material,'CreateMetaHuman.RoughnessOwner')
    raw=unreal.EditorAssetLibrary.get_metadata_tag(material,RESPONSE_TAG)
    if not owner.startswith('/Game/'):raise ValueError('Retain authored head material overrides before refreshing managed roughness')
    prior=json.loads(raw) if raw else None
    if prior is not None and (prior.get('schema')!=1 or prior.get('mode') not in {'ABSOLUTE','SKIN_DETAIL'}):raise ValueError('Unsupported inherited roughness calibration')
    if response is not None:
        if not isinstance(response,dict) or response.get('mode','ABSOLUTE') not in {'ABSOLUTE','SKIN_DETAIL'}:raise ValueError('Invalid refreshed roughness response')
        center=float(response.get('center',1))
        if not math.isfinite(center) or not .02<=center<=1:raise ValueError('Invalid refreshed roughness center')
        if response.get('mode')=='SKIN_DETAIL' and response.get('finish_roughness') is not None:
            finish=float(response['finish_roughness'])
            if not math.isfinite(finish) or not .1<=finish<=.9:raise ValueError('Invalid refreshed skin finish')
    lib=unreal.MaterialEditingLibrary
    if 'Roughness' not in {str(name) for name in lib.get_texture_parameter_names(material)}:raise ValueError('Managed head shader has no roughness input')
    if not lib.get_material_instance_static_switch_parameter_value(material,'Use Roughness Map as Base'):raise ValueError('Enable managed projected roughness before refreshing its texture')
    # Older head revisions did not copy the calibration ledger. Keep their
    # existing finish exactly; do not invent baseline multipliers or compound it.
    settings=(response if response is not None else {key:prior[key] for key in ('mode','center','finish_roughness') if key in prior}) if prior is not None else None
    if settings and settings.get('mode')=='SKIN_DETAIL' and settings.get('finish_roughness') is not None and 'CM Roughness Response Exponent' not in {str(name) for name in lib.get_scalar_parameter_names(material)}:
        raise ValueError('This inherited skin shader needs the managed finish response before that calibration can be refreshed')
    return settings


def refresh_material(material,texture,owner,response=None):
    """Refresh a private head instance in place, retaining its parent and edits."""
    import unreal
    settings=validate_refresh(material,response)
    root=owner.rsplit('/',1)[0]
    if not material.get_path_name().startswith(root+'/Materials/') or unreal.EditorAssetLibrary.get_metadata_tag(material,'CreateMetaHuman.CharacterOwner')!=owner:
        raise ValueError('Copy inherited head materials into the character before refreshing roughness')
    lib=unreal.MaterialEditingLibrary
    result=calibrate_response(material,settings) if settings is not None else {'mode':'PRESERVE_LEGACY','calibration_recovered':False}
    lib.set_material_instance_texture_parameter_value(material,'Roughness',texture)
    if settings is not None and 'CM Roughness Response Exponent' in {str(name) for name in lib.get_scalar_parameter_names(material)}:
        lib.set_material_instance_scalar_parameter_value(material,'CM Roughness Response Exponent',result['ui_response_exponent'])
    unreal.EditorAssetLibrary.set_metadata_tag(material,'CreateMetaHuman.RoughnessOwner',owner)
    return result


def managed_scalar_skip(character,roles):
    import unreal
    owner=character.get_path_name();cache={}
    def skip(component,current,key):
        role='BODY' if component.get_name()=='Body' else 'HEAD'
        if role not in roles:return False
        parent=current.get_editor_property('parent')
        if not parent:return False
        path=parent.get_path_name()
        if path not in cache:
            raw=unreal.EditorAssetLibrary.get_metadata_tag(parent,RESPONSE_TAG)
            owned=unreal.EditorAssetLibrary.get_metadata_tag(parent,'CreateMetaHuman.RoughnessOwner')==owner
            cache[path]=(set(json.loads(raw).get('parameters',{}))|{'CM Roughness Response Exponent'}) if owned and raw else set()
        return str(key) in cache[path]
    return skip


def capture(character):
    import unreal
    root=character.get_path_name().rsplit('/',1)[0]
    live=[]
    for actor in unreal.ObjectIterator(unreal.MetaHumanCharacterEditorActor):
        if actor.is_actor_being_destroyed():continue
        components=actor.get_components_by_class(unreal.SkeletalMeshComponent)
        candidates=[component.get_material(i) for component in components for i in range(component.get_num_materials())]
        # Revisions inherit artist albedo from another asset folder. The
        # actor's native character pointer is authoritative when available;
        # texture-path ownership alone misses these actors or conflates them.
        try:actor_character=actor.get_editor_property('character')
        except Exception:actor_character=None
        if actor_character and actor_character!=character:continue
        owned_actor=actor_character==character
        owners=set()
        for candidate in candidates:
            if not isinstance(candidate,unreal.MaterialInstanceDynamic):continue
            parent=candidate.get_editor_property('parent')
            if parent:
                owners.update(value for tag in ('CreateMetaHuman.RoughnessOwner','CreateMetaHuman.CharacterOwner')
                    if (value:=unreal.EditorAssetLibrary.get_metadata_tag(parent,tag)))
        if not actor_character and owners and character.get_path_name() not in owners:continue
        if character.get_path_name() in owners:owned_actor=True
        for candidate in candidates:
            if not isinstance(candidate,unreal.MaterialInstanceDynamic):continue
            parent=candidate.get_editor_property('parent')
            if parent and (parent.get_path_name().startswith(root+'/Materials/') or
                    unreal.EditorAssetLibrary.get_metadata_tag(parent,'CreateMetaHuman.RoughnessOwner')==character.get_path_name()):owned_actor=True
            base=candidate.get_texture_parameter_value('Basecolor')
            if base and base.get_path_name().startswith(root+'/Maps/'):owned_actor=True;break
        if not owned_actor:continue
        for component in components:
            for index in range(component.get_num_materials()):
                material=component.get_material(index)
                if not isinstance(material,unreal.MaterialInstanceDynamic):continue
                live.append((component,index,material,
                    {name:material.get_texture_parameter_value(name) for name in unreal.MaterialEditingLibrary.get_texture_parameter_names(material)},
                    {name:material.get_scalar_parameter_value(name) for name in unreal.MaterialEditingLibrary.get_scalar_parameter_names(material)},
                    {name:material.get_vector_parameter_value(name) for name in unreal.MaterialEditingLibrary.get_vector_parameter_names(material)}))
    return live


def restore(live,skip=None,skip_scalar=None):
    import unreal
    for component,index,_,textures,scalars,vectors in live:
        current=component.get_material(index)
        if not isinstance(current,unreal.MaterialInstanceDynamic):continue
        for key,value in textures.items():
            if value and not (skip and skip(component,current,str(key))):current.set_texture_parameter_value(key,value)
        for key,value in scalars.items():
            if skip_scalar and skip_scalar(component,current,str(key)):
                value=unreal.MaterialEditingLibrary.get_material_instance_scalar_parameter_value(current.get_editor_property('parent'),key)
            current.set_scalar_parameter_value(key,value)
        for key,value in vectors.items():current.set_vector_parameter_value(key,value)


def apply(character, maps,responses=None):
    import unreal
    if not maps:return {}
    root=character.get_path_name().rsplit('/',1)[0]
    owner=character.get_path_name()
    live=capture(character)
    skin=character.get_editor_property('skin_settings')
    overrides=skin.get_editor_property('texture_material_overrides')
    materials=overrides.get_editor_property('material_overrides')
    faces=dict(materials.get_editor_property('skin'))
    # Refresh established private bindings without enabling/replacing any
    # overrides. Independent eye/teeth customizations must not block updating
    # a skin texture already managed by this character.
    existing=[]
    if overrides.get_editor_property('enable_material_overrides'):
        for role,texture in maps.items():
            slots=[('BODY',materials.get_editor_property('body'))] if role=='BODY' else [
                (slot,faces.get(getattr(unreal.MetaHumanCharacterSkinMaterialSlot,slot)))
                for slot in ('LOD0','LOD1','LOD2','LOD3','LOD4','LOD5TO7')]
            for slot,material in slots:
                if (not material or not material.get_path_name().startswith(root+'/Materials/') or
                    unreal.EditorAssetLibrary.get_metadata_tag(material,'CreateMetaHuman.RoughnessOwner')!=owner):
                    existing=[];break
                settings=validate_refresh(material,(responses or {}).get(role))
                existing.append((slot,material,texture,settings))
            if not existing:break
    expected=sum(1 if role=='BODY' else 6 for role in maps)
    if len(existing)==expected:
        for slot,material,texture,settings in existing:
            if settings is not None:calibrate_response(material,settings)
            unreal.MaterialEditingLibrary.set_material_instance_texture_parameter_value(material,'Roughness',texture)
            if not unreal.EditorAssetLibrary.save_loaded_asset(material,False):raise RuntimeError('Could not save refreshed skin roughness')
        def refreshed(component,current,key):
            role='BODY' if component.get_name()=='Body' else 'HEAD'
            parent=current.get_editor_property('parent')
            return (key=='Roughness' and role in maps and parent and
                unreal.EditorAssetLibrary.get_metadata_tag(parent,'CreateMetaHuman.RoughnessOwner')==owner)
        restore(live,refreshed,managed_scalar_skip(character,maps))
        return {slot:material.get_path_name() for slot,material,_,_ in existing}
    if overrides.get_editor_property('enable_material_overrides'):
        previous=list(faces.values())
        body=materials.get_editor_property('body')
        if body:previous.append(body)
        if materials.get_editor_property('teeth_and_eyes'):
            raise ValueError('Retain authored eye/teeth material overrides before enabling managed roughness')
        for material in previous:
            if material and unreal.EditorAssetLibrary.get_metadata_tag(material,'CreateMetaHuman.RoughnessOwner')!=owner:
                raise ValueError('Retain authored skin material overrides before enabling managed roughness')
    tools=unreal.AssetToolsHelpers.get_asset_tools()
    result={}
    for role,texture in maps.items():
        slots=[('BODY','MI_Skin_Body_MHC')] if role=='BODY' else [
            (slot,'MI_Skin_Head_UI_'+parent) for slot,parent in
            (('LOD0','LOD0'),('LOD1','LOD1'),('LOD2','LOD2'),('LOD3','LOD3'),('LOD4','LOD4'),('LOD5TO7','LOD57'))]
        for slot,parent_name in slots:
            parent=unreal.load_asset('/MetaHumanCharacter/Materials/'+parent_name)
            if not isinstance(parent,unreal.MaterialInstanceConstant):raise ValueError('Native skin parent is unavailable: '+parent_name)
            if 'Roughness' not in {str(name) for name in unreal.MaterialEditingLibrary.get_texture_parameter_names(parent)}:
                raise ValueError('Native skin material has no Roughness parameter')
            name='CM_Roughness_'+slot
            path=root+'/Materials/'+name
            material=unreal.load_asset(path)
            if material:
                if not isinstance(material,unreal.MaterialInstanceConstant) or unreal.EditorAssetLibrary.get_metadata_tag(material,'CreateMetaHuman.RoughnessOwner')!=owner:
                    raise ValueError('Managed roughness material name belongs to an authored asset: '+path)
            else:
                material=tools.create_asset(name,root+'/Materials',unreal.MaterialInstanceConstant,unreal.MaterialInstanceConstantFactoryNew())
                unreal.MaterialEditingLibrary.set_material_instance_parent(material,parent)
                unreal.EditorAssetLibrary.set_metadata_tag(material,'CreateMetaHuman.RoughnessOwner',owner)
            raw_response=unreal.EditorAssetLibrary.get_metadata_tag(material,RESPONSE_TAG)
            # Native preview MIDs can retain stale explicit copies of these
            # fields after a material refresh. They must not overwrite the
            # persistent calibration and become new artist baseline factors.
            managed_values={key:unreal.MaterialEditingLibrary.get_material_instance_scalar_parameter_value(material,key)
                            for key in json.loads(raw_response).get('parameters',{})} if raw_response else {}
            # A new parent otherwise loses wardrobe-authored brow/freckle
            # textures that exist only in the live native instance.
            for _,_,source,_,_,_ in live:
                source_parent=source.get_editor_property('parent')
                if source_parent not in (parent,material):continue
                for field in ('texture_parameter_values','scalar_parameter_values','vector_parameter_values'):
                    material.set_editor_property(field,source.get_editor_property(field))
                break
            for key,value in managed_values.items():
                unreal.MaterialEditingLibrary.set_material_instance_scalar_parameter_value(material,key,value)
            response=(responses or {}).get(role,{'mode':'ABSOLUTE'})
            expanded=response.get('mode')=='SKIN_DETAIL' and response.get('finish_roughness') is not None
            desired_parent=response_parent(parent,root,owner) if expanded else parent
            if material.get_editor_property('parent')!=desired_parent:
                unreal.MaterialEditingLibrary.set_material_instance_parent(material,desired_parent)
            if expanded:
                unreal.MaterialEditingLibrary.set_material_instance_scalar_parameter_value(material,'CM Roughness Response Exponent',5.)
            unreal.MaterialEditingLibrary.set_material_instance_texture_parameter_value(material,'Roughness',texture)
            # Native UI parents default to region-authored constant roughness.
            # Merely binding this texture leaves it outside the active graph.
            switch='Use Roughness Map as Base'
            if switch not in {str(key) for key in unreal.MaterialEditingLibrary.get_static_switch_parameter_names(material)}:
                raise ValueError('Native skin material cannot use a projected roughness map')
            if not unreal.MaterialEditingLibrary.get_material_instance_static_switch_parameter_value(material,switch):
                # UE 5.8's setter always returns false; verify the actual value.
                unreal.MaterialEditingLibrary.set_material_instance_static_switch_parameter_value(material,switch,True)
                if not unreal.MaterialEditingLibrary.get_material_instance_static_switch_parameter_value(material,switch):
                    raise RuntimeError('Could not enable projected skin roughness')
            calibrate_response(material,response)
            if not unreal.EditorAssetLibrary.save_loaded_asset(material,False):raise RuntimeError('Could not save roughness material')
            if slot=='BODY':materials.set_editor_property('body',material)
            else:faces[getattr(unreal.MetaHumanCharacterSkinMaterialSlot,slot)]=material
            result[slot]=material.get_path_name()
    materials.set_editor_property('skin',faces)
    overrides.set_editor_property('material_overrides',materials)
    overrides.set_editor_property('enable_material_overrides',True)
    overrides.set_editor_property('inherit_ui_params_and_src_textures',True)
    skin.set_editor_property('texture_material_overrides',overrides)
    subsystem=unreal.get_editor_subsystem(unreal.MetaHumanCharacterEditorSubsystem)
    owned=not subsystem.is_object_added_for_editing(character)
    try:
        if owned and not subsystem.try_add_object_to_edit(character):raise RuntimeError('Creator could not open skin materials')
        subsystem.commit_skin_settings(character,skin)
        # Committing skin settings rebuilds MIDs, including eyelashes. Preserve
        # the exact current wardrobe bindings, rather than resetting to parents.
        def changed(component,current,key):
            is_skin=unreal.EditorAssetLibrary.get_metadata_tag(current.get_editor_property('parent'),'CreateMetaHuman.RoughnessOwner')==owner
            role='BODY' if component.get_name()=='Body' else 'HEAD'
            return key=='Roughness' and is_skin and role in maps
        managed=managed_scalar_skip(character,maps)
        restore(live,changed,lambda component,current,key: str(key)=='CM Roughness Response Exponent' or managed(component,current,key))
        if not unreal.EditorAssetLibrary.save_loaded_asset(character,False):raise RuntimeError('Could not save roughness overrides')
    finally:
        if owned:subsystem.remove_object_to_edit(character)
    return result
