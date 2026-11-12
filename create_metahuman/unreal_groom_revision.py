"""Receipt-backed replacement of an existing assembled-character groom."""
import hashlib,json,runpy,re,os
from pathlib import Path

def _write(path,value):
    path=Path(path);temporary=path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value,indent=2),encoding='utf-8');temporary.replace(path)


def _api():return runpy.run_path(str(Path(__file__).with_name('unreal_groom.py')))
def _path(value):return value.get_path_name() if value else ''


def _object(unreal,path):
    if not path:return None
    value=unreal.find_object(None,path) or unreal.load_asset(path)
    if not value:raise ValueError('The recorded groom resource is unavailable: '+path)
    return value


def _component(unreal,actor,name):
    found=[c for c in actor.get_components_by_class(unreal.GroomComponent) if c.get_name()==name]
    if len(found)!=1:raise ValueError('Choose one existing groom component; available: '+', '.join(c.get_name() for c in actor.get_components_by_class(unreal.GroomComponent)))
    return found[0]


def _actor(unreal,path):
    found=[a for a in unreal.get_editor_subsystem(unreal.EditorActorSubsystem).get_all_level_actors() if a.get_path_name()==path]
    if len(found)==1:return found[0]
    # Selected transient editor previews are omitted by GetAllLevelActors.
    # Resolve only the recorded object in the current editor world.
    actor=unreal.find_object(None,path)
    world=unreal.get_editor_subsystem(unreal.UnrealEditorSubsystem).get_editor_world()
    if isinstance(actor,unreal.Actor) and unreal.SystemLibrary.is_valid(actor) and actor.get_world()==world:return actor
    raise ValueError('The original groom-review actor is unavailable. Reopen its level; transient previews require the original editor session.')


def _blueprint(unreal,actor):
    path=actor.get_class().get_path_name()
    return unreal.load_asset(path[:-2]) if path.startswith('/Game/') and path.endswith('_C') else None


def _default(unreal,bp,name):
    return _api()['blueprint_groom_template'](bp,name)


def _state(component):
    return dict(groom=_path(component.get_editor_property('groom_asset')),binding=_path(component.get_editor_property('binding_asset')),
                materials=[_path(m) for m in component.get_editor_property('override_materials')],
                groups=[v.export_text() for v in component.get_editor_property('groom_groups_desc')],
                use_cards=bool(component.get_editor_property('use_cards')))


def _template_settings(unreal,component,*,archetype=False):
    snapshot=getattr(getattr(unreal,'CreateMHGroomLibrary',None),'snapshot_groom_template',None)
    if not snapshot:raise ValueError('Rebuild the matching bridge to protect authored groom-template settings during Restore')
    raw=snapshot(component,archetype)
    if not raw:raise ValueError('The groom template settings could not be captured safely; retain the current edits')
    value=json.loads(raw)
    if value.get('schema')!=1 or not value.get('objects',{}).get(''):raise ValueError('The native groom template snapshot is incomplete')
    return value


def _assign(unreal,component,state):
    component.modify();component.set_binding_asset(None)
    # Stage the complete instance state before native resource initialization.
    # Property notifications can reconstruct an editor-owned component between
    # writes, especially when the replacement changes its group count.
    notify=unreal.PropertyAccessChangeNotifyMode.NEVER
    component.set_editor_property('override_materials',[_object(unreal,p) for p in state['materials']],notify_mode=notify)
    groups=[]
    for text in state['groups']:
        value=unreal.HairGroupDesc();value.import_text(text);groups.append(value)
    component.set_editor_property('groom_groups_desc',groups,notify_mode=notify)
    component.set_editor_property('use_cards',state['use_cards'],notify_mode=notify)
    component.set_groom_asset(_object(unreal,state['groom']));component.set_binding_asset(_object(unreal,state['binding']))


def _fingerprints(unreal,assets):
    packages={v.get_path_name().split('.')[0] for v in assets if v}
    dirty={p.get_path_name() for p in unreal.EditorLoadingAndSavingUtils.get_dirty_content_packages()}
    if packages&dirty:raise ValueError('Save the groom, binding and target assets before starting or finishing this review')
    result={}
    for package in packages:
        mount,relative=package.lstrip('/').split('/',1)
        if mount=='Game':directory=unreal.Paths.project_content_dir()
        elif mount=='Engine':directory=unreal.Paths.engine_content_dir()
        else:directory=unreal.PluginBlueprintLibrary.get_plugin_content_dir(mount)
        if not directory:raise ValueError('Copy editable groom resources under /Game before reviewing this content mount')
        base=Path(unreal.Paths.convert_relative_path_to_full(directory))/relative
        if not base.with_suffix('.uasset').is_file():raise ValueError('Save the groom resources before review')
        for extension in ('.uasset','.uexp','.ubulk'):
            file=base.with_suffix(extension)
            if file.is_file():
                with file.open('rb') as stream:result[str(file.resolve())]=hashlib.file_digest(stream,'sha256').hexdigest()
    return result


def preview(folder,groom_path,binding_path,component_name='Hair',*,group_policy='AUTO'):
    import unreal
    folder=Path(folder).resolve();file=folder/'preview.json'
    if file.exists():raise ValueError('Observe or restore the original groom review before creating another')
    if unreal.get_editor_subsystem(unreal.LevelEditorSubsystem).is_in_play_in_editor():raise ValueError('Stop Play or Simulate before replacing a groom')
    selected=unreal.get_editor_subsystem(unreal.EditorActorSubsystem).get_selected_level_actors()
    if len(selected)!=1:raise ValueError('Select one assembled character actor in Unreal')
    actor=selected[0];component=_component(unreal,actor,component_name);before=_state(component)
    owners=getattr(unreal,'_cm_groom_review_owners',{});owner=actor.get_path_name()+'::'+component_name
    if owner in owners:
        previous=Path(owners[owner])/'preview.json'
        active=json.loads(previous.read_text()) if previous.is_file() else {}
        if active.get('mode') in {'PREPARING','PREVIEW','SAVING','RESTORING'}:
            raise ValueError('This component already has a groom review; resume or restore its original receipt')
        owners.pop(owner,None)
    api=_api();groom=_object(unreal,groom_path);binding=_object(unreal,binding_path)
    if not isinstance(groom,unreal.GroomAsset) or not isinstance(binding,unreal.GroomBindingAsset) or binding.get_editor_property('groom')!=groom:raise ValueError('Push a matching groom and binding before preview')
    old=_object(unreal,before['groom']);old_binding=_object(unreal,before['binding'])
    if not old or not old_binding:raise ValueError('Choose an existing bound groom, or explicitly add a separate component')
    parent=component.get_attach_parent()
    target=parent.get_skeletal_mesh_asset() if isinstance(parent,unreal.SkeletalMeshComponent) else None
    if not target or old_binding.get_editor_property('target_skeletal_mesh')!=target:
        raise ValueError('The existing groom binding must match its attached skeletal mesh')
    original_binding=binding;retarget=None
    if binding.get_editor_property('target_skeletal_mesh')!=target:
        source=binding.get_editor_property('source_skeletal_mesh')
        if not isinstance(source,unreal.SkeletalMesh):raise ValueError('The pushed groom needs its persistent source rest head before preview can bind it to this face')
        inputs=_fingerprints(unreal,[groom,binding,source,target,old,old_binding])
        name=re.sub(r'[^A-Za-z0-9_]','_',groom.get_name())
        suffix=hashlib.sha256((str(folder)+'|'+target.get_path_name()).encode()).hexdigest()[:12]
        destination='/Game/CreateMetaHuman/Grooms/ReviewBindings/'+name+'_'+suffix+'_Binding'
        pending=dict(source_binding=binding_path,source_head=source.get_path_name(),target=target.get_path_name(),
                     private_binding=destination,inputs=inputs,status='PREPARING')
        _write(folder/'binding-preparation.json',pending)
        if unreal.EditorAssetLibrary.does_asset_exist(destination):
            fresh=unreal.load_asset(destination)
            owner=unreal.EditorAssetLibrary.get_metadata_tag(fresh,'MetaHumanMaker.GroomPreviewOrigin')
            signature=unreal.EditorAssetLibrary.get_metadata_tag(fresh,'MetaHumanMaker.GroomPreviewInputs')
            expected=api['binding_settings'](binding);expected.update(target_skeletal_mesh=target.get_path_name())
            if owner!=str(folder) or signature!=json.dumps(inputs,sort_keys=True) or api['binding_settings'](fresh)!=expected:
                raise ValueError('The private preview binding changed; retain it and start a fresh review')
            api['require_binding_roots'](fresh);pending['reused']=True
        else:
            fresh=api['copy_binding'](binding,groom,target,destination)
            unreal.EditorAssetLibrary.set_metadata_tag(fresh,'MetaHumanMaker.GroomPreviewOrigin',str(folder))
            unreal.EditorAssetLibrary.set_metadata_tag(fresh,'MetaHumanMaker.GroomPreviewInputs',json.dumps(inputs,sort_keys=True))
            if not unreal.EditorAssetLibrary.save_loaded_asset(fresh,False):raise RuntimeError('Could not save the private face binding')
            pending['reused']=False
        binding=fresh;binding_path=_path(binding);pending['status']='complete';_write(folder/'binding-preparation.json',pending)
        retarget=pending
    if before['groom']==groom_path and before['binding']==binding_path:raise ValueError('This groom and binding are already on the selected component')
    materials,mapping=api['revision_materials'](component,groom)
    bp=_blueprint(unreal,actor);default_component=_default(unreal,bp,component_name) if bp else None
    defaults=_state(default_component) if default_component else None
    inherited=bool(default_component and _path(default_component).split('.')[0]!=_path(bp).split('.')[0])
    parent_asset=unreal.load_asset(_path(default_component).split('.')[0]) if inherited else None
    after=dict(before,groom=groom_path,binding=binding_path,materials=[_path(v) for v in materials])
    groups,group_mapping=api['revision_group_overrides'](component,groom,policy=group_policy,with_mapping=True)
    after['groups']=[v.export_text() for v in groups]
    record=dict(schema=1,status='complete',mode='PREPARING',origin=str(folder),actor=actor.get_path_name(),actor_label=actor.get_actor_label(),component=component_name,
                unreal_project=str(Path(unreal.Paths.convert_relative_path_to_full(unreal.Paths.get_project_file_path())).resolve()),
                before=before,after=after,blueprint=_path(bp),blueprint_before=defaults,blueprint_inherited=inherited,
                blueprint_override_token=hashlib.sha256(str(folder).encode()).hexdigest()[:32] if inherited else '',
                previous_metadata=unreal.EditorAssetLibrary.get_metadata_tag(bp,'CreateMetaHuman.GroomRevision.'+component_name) if bp else '',
                material_slots=mapping,group_controls=group_mapping,group_policy=group_policy,
                binding_retarget=retarget,owned_bindings=[_path(binding)] if retarget else [],
                resources=[_path(v) for v in (old,old_binding,groom,binding,original_binding,target,parent_asset) if v],
                source_fingerprints=_fingerprints(unreal,[old,old_binding,groom,binding,original_binding,target,parent_asset]),blueprint_saved=False)
    _write(file,record)
    owners[owner]=str(folder);unreal._cm_groom_review_owners=owners
    try:
        with unreal.ScopedEditorTransaction('Preview replacement character groom'):_assign(unreal,component,after)
        observed=_state(_component(unreal,actor,component_name))
        if observed!=after:
            _write(folder/'preview-conflict.json',dict(expected=after,observed=observed))
            raise RuntimeError('Native groom preview did not retain the requested component state')
        record['mode']='PREVIEW';_write(file,record);return record
    except Exception:
        _assign(unreal,component,before);owners.pop(owner,None);record.update(mode='RESTORED',status='failed');_write(file,record);raise


def finish(origin,restore=False):
    import unreal
    if unreal.get_editor_subsystem(unreal.LevelEditorSubsystem).is_in_play_in_editor():raise ValueError('Stop Play or Simulate before finishing a groom replacement')
    folder=Path(origin).resolve();file=folder/'preview.json';record=json.loads(file.read_text())
    for key in ('before','after','blueprint_before','blueprint_after'):
        if record.get(key):record[key].pop('tags',None)
    initial_mode=record['mode']
    if initial_mode not in {'PREPARING','PREVIEW','SAVING','SAVED','RESTORING'}:raise ValueError('This groom review has already finished')
    if record.get('unreal_project'):
        actual=Path(unreal.Paths.convert_relative_path_to_full(unreal.Paths.get_project_file_path())).resolve()
        if os.path.normcase(str(actual))!=os.path.normcase(str(Path(record['unreal_project']).resolve())):
            raise ValueError('Open this groom review\'s original Unreal project before Keep or Restore')
    actor=_actor(unreal,record['actor']);component=_component(unreal,actor,record['component'])
    current=_state(component)
    if _fingerprints(unreal,[_object(unreal,p) for p in record['resources']])!=record['source_fingerprints']:
        raise ValueError('Groom resources changed after preview; retain the newer artist edit')
    api=_api();bp=_object(unreal,record['blueprint'])
    if initial_mode=='PREPARING' and current==record['before']:
        if not restore:raise ValueError('The interrupted preview was not applied; restore it before starting a fresh review')
        record['mode']='RESTORED';_write(file,record);return record
    if initial_mode=='SAVING':
        template=_default(unreal,bp,record['component']);defaults=_state(template)
        local=_path(template).split('.')[0]==_path(bp).split('.')[0]
        owns=(not record.get('blueprint_inherited') or local and unreal.EditorAssetLibrary.get_metadata_tag(template,api['OVERRIDE_OWNER'])==record.get('blueprint_override_token'))
        if local and record.get('blueprint_inherited') and not owns:raise ValueError('The interrupted save does not own this local override; retain the artist edit')
        if defaults==record['blueprint_after'] and owns:
            if current!=record['after']:raise ValueError('The groom component changed during the interrupted save')
            record.update(mode='SAVED',after=current,blueprint_saved=True)
        elif defaults==record['blueprint_before']:
            record['mode']='PREVIEW'
            if local and record.get('blueprint_inherited') and owns:record['blueprint_saved']=True
        else:raise ValueError('The blueprint groom changed during the interrupted save; retain those artist edits')
    allowed=[record['after']]+([record['before']] if initial_mode=='RESTORING' else [])
    intermediate=(initial_mode=='RESTORING' and restore and
                  all(current[k] in (record['before'][k],record['after'][k]) for k in current))
    if current not in allowed and not intermediate:
        changed=[key for key in current if current[key]!=record['after'].get(key)]
        _write(folder/'conflict.json',dict(fields=changed,expected=record['after'],observed=current))
        raise ValueError('The reviewed groom component changed ('+', '.join(changed)+'); retain those artist edits before Keep or Restore')
    if not restore and initial_mode in {'SAVED','RESTORING'}:raise ValueError('Restore the pending review, or start a fresh preview')
    if restore:
        record['mode']='RESTORING';_write(file,record)
    save_blueprint=(not restore and record['mode']!='SAVED') or (restore and record['blueprint_saved'])
    if save_blueprint:
        if not bp:raise ValueError('Preview can be restored, but saving needs a project character blueprint')
        defaults=_default(unreal,bp,record['component']);expected=record.get('blueprint_after') if restore else record['blueprint_before']
        permitted=[expected]+([record['blueprint_before']] if restore and initial_mode in {'RESTORING','SAVING'} else [])
        if _state(defaults) not in permitted:raise ValueError('The blueprint groom changed after preview; retain the newer artist edit')
        if not restore and initial_mode!='SAVING':
            inherited_now=_path(defaults).split('.')[0]!=_path(bp).split('.')[0]
            if inherited_now!=record.get('blueprint_inherited',inherited_now):raise ValueError('Blueprint override ownership changed after preview; retain the newer artist edit')
        if restore:
            previous=record['blueprint_before']
            bp.modify()
            if record.get('blueprint_inherited'):
                local=_default(unreal,bp,record['component'])
                if _path(local).split('.')[0]==_path(bp).split('.')[0]:
                    if unreal.EditorAssetLibrary.get_metadata_tag(local,api['OVERRIDE_OWNER'])!=record.get('blueprint_override_token'):
                        raise ValueError('This review does not own the local groom override; retain the artist edit')
                    expected_settings=record.get('blueprint_template_after') or record.get('blueprint_template_before') or _template_settings(unreal,local,archetype=True)
                    if _template_settings(unreal,local)!=expected_settings:
                        raise ValueError('Other authored settings on this groom override changed; retain the newer artist edits before Restore')
                    native=getattr(unreal,'CreateMHGroomLibrary',None)
                    if not native or not native.remove_blueprint_groom_override(bp,record['component'],local.get_editor_property('groom_asset'),local.get_editor_property('binding_asset')):
                        raise RuntimeError('Could not remove this review\'s local override; parent retained')
                    unreal.BlueprintEditorLibrary.compile_blueprint(bp)
                if _state(_default(unreal,bp,record['component']))!=previous:raise ValueError('The inherited groom differs from the recorded parent; retain later edits')
            else:_assign(unreal,_default(unreal,bp,record['component']),previous)
            saved=dict(status='complete',blueprint=bp.get_path_name(),restored=True,source_assets_saved=0)
            key='CreateMetaHuman.GroomRevision.'+record['component']
            if record['previous_metadata']:unreal.EditorAssetLibrary.set_metadata_tag(bp,key,record['previous_metadata'])
            else:unreal.EditorAssetLibrary.remove_metadata_tag(bp,key)
            if not unreal.EditorAssetLibrary.save_loaded_asset(bp,False):raise RuntimeError('Save the restored character blueprint in Unreal')
        else:
            groom=_object(unreal,record['after']['groom']);binding=_object(unreal,record['after']['binding'])
            materials,_=api['revision_materials'](defaults,groom)
            group_policy=record.get('group_policy','AUTO')
            groups,group_mapping=api['revision_group_overrides'](defaults,groom,policy=group_policy,with_mapping=True)
            record['blueprint_after']=dict(record['blueprint_before'],groom=_path(groom),binding=_path(binding),materials=[_path(v) for v in materials],groups=[v.export_text() for v in groups])
            record['blueprint_group_controls']=group_mapping
            if record.get('blueprint_inherited'):record['blueprint_template_before']=_template_settings(unreal,defaults)
            record['mode']='SAVING';_write(file,record)
            saved=api['apply_owned_revision'](bp,record['component'],_object(unreal,record['after']['groom']),_object(unreal,record['after']['binding']),owner_folder=bp.get_path_name().rsplit('/',1)[0],expected_original=record['blueprint_before']['groom'],compile_blueprint=False,group_policy=group_policy,override_token=record.get('blueprint_override_token',''))
            if _state(_default(unreal,bp,record['component']))!=record['blueprint_after']:raise RuntimeError('Saved blueprint groom state differs from its recorded plan')
            if record.get('blueprint_inherited'):
                record['blueprint_template_after']=_template_settings(unreal,_default(unreal,bp,record['component']))
        record['blueprint_result']=saved;record['blueprint_saved']=True
        _write(file,record)
    # Template edits can reconstruct live components. Never keep a pre-save
    # component pointer, and retain instance overrides through that rebuild.
    actor=_actor(unreal,record['actor']);component=_component(unreal,actor,record['component'])
    if restore:
        with unreal.ScopedEditorTransaction('Restore previous character groom'):_assign(unreal,component,record['before'])
        record['mode']='RESTORED'
    else:
        if _state(component)!=record['after']:_assign(unreal,component,record['after'])
        record['mode']='SAVED'
    component=_component(unreal,_actor(unreal,record['actor']),record['component'])
    if _state(component)!=(record['before'] if restore else record['after']):raise RuntimeError('The groom component did not retain the reviewed state')
    getattr(unreal,'_cm_groom_review_owners',{}).pop(record['actor']+'::'+record['component'],None)
    _write(file,record);return record


def close(origin):
    """End recovery tracking without changing actor, blueprint or asset data."""
    import unreal
    folder=Path(origin).resolve();file=folder/'preview.json';record=json.loads(file.read_text())
    record.update(closed_from=record['mode'],mode='CLOSED')
    getattr(unreal,'_cm_groom_review_owners',{}).pop(record['actor']+'::'+record['component'],None)
    _write(file,record);return record
