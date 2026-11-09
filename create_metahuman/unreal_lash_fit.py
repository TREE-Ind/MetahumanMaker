"""Native selected-character lash repair, with durable asynchronous receipts."""
import json,hashlib,runpy,subprocess,os
from pathlib import Path


def _write(path,value):
    path=Path(path);temporary=path.with_suffix('.tmp');temporary.write_text(json.dumps(value,indent=2),encoding='utf-8');temporary.replace(path)


def _digest(path):
    with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()


def _api():return runpy.run_path(str(Path(__file__).with_name('unreal_groom.py')))


def _component(unreal,actor,name):
    found=[c for c in actor.get_components_by_class(unreal.GroomComponent) if c.get_name()==name]
    if len(found)!=1:raise ValueError('The selected character needs exactly one '+name+' groom component')
    return found[0]


def _actor(unreal,path):
    actors=[a for a in unreal.get_editor_subsystem(unreal.EditorActorSubsystem).get_all_level_actors() if a.get_path_name()==path]
    if len(actors)!=1:raise ValueError('Return to the original character level before continuing this lash revision')
    return actors[0]


def _blueprint(unreal,actor):
    path=actor.get_class().get_path_name()
    if not path.startswith('/Game/') or not path.endswith('_C'):return None
    value=unreal.load_asset(path[:-2]);return value if isinstance(value,unreal.Blueprint) else None


def _default_groom(unreal,bp,name):
    subsystem=unreal.get_engine_subsystem(unreal.SubobjectDataSubsystem);data=unreal.SubobjectDataBlueprintFunctionLibrary;found=[]
    for handle in subsystem.k2_gather_subobject_data_for_blueprint(bp):
        value=data.get_object(data.get_data(handle))
        if isinstance(value,unreal.GroomComponent) and value.get_name().removesuffix('_GEN_VARIABLE')==name and value not in found:found.append(value)
    if len(found)!=1:raise ValueError('The character blueprint has no unique matching lash component')
    return found[0]


def _capture(folder,settings):
    import unreal
    if unreal.get_editor_subsystem(unreal.LevelEditorSubsystem).is_in_play_in_editor():raise ValueError('Stop Play or Simulate before fitting lashes')
    selected=unreal.get_editor_subsystem(unreal.EditorActorSubsystem).get_selected_level_actors()
    if len(selected)!=1:raise ValueError('Select one assembled character actor in Unreal')
    actor=selected[0];component=_component(unreal,actor,settings['component']);groom=component.get_editor_property('groom_asset');binding=component.get_editor_property('binding_asset')
    if not groom or not binding or binding.get_editor_property('groom')!=groom:raise ValueError('The character needs a matching existing lash groom and binding')
    target=binding.get_editor_property('target_skeletal_mesh');source=binding.get_editor_property('source_skeletal_mesh');reference=source or target
    if not isinstance(reference,unreal.SkeletalMesh):raise ValueError('This lash workflow requires a native skeletal binding reference')
    faces=[c for c in actor.get_components_by_class(unreal.SkeletalMeshComponent) if c.get_skeletal_mesh_asset()==target]
    if len(faces)!=1:raise ValueError('The lash binding must target one skeletal component on this actor')
    anatomy=Path(unreal.Paths.convert_relative_path_to_full(unreal.Paths.engine_plugins_dir()))/'MetaHuman/MetaHumanCoreTechLib/Content/ArchetypeDNA/SKM_Face.dna'
    if not anatomy.is_file():raise ValueError('Install the matching MetaHuman Creator core content before fitting native lashes')
    calibration=getattr(unreal,'CreateMHCalibrationLibrary',None);expressions=getattr(unreal,'CreateMHExpressionLibrary',None)
    if not calibration or not expressions or not calibration.export_calibration_reference(str(anatomy),str(folder/'anatomy.json')):raise ValueError('Update the project native bridge to 0.22 or newer for anatomical lash fitting')
    topology=json.loads((folder/'anatomy.json').read_text())
    if not topology.get('coordinate_system') or any('layout_positions_file' not in m for m in topology['meshes']):raise ValueError('Update the project native bridge to 0.22 or newer for anatomical lash fitting')
    if not expressions.export_engine_control_pose(reference,'{}',str(folder/'neutral.json'),faces[0] if source is None else None,True,0):raise ValueError('The lash binding reference must retain its native MetaHuman DNA identities')
    _api()['export_groom'](groom.get_path_name(),str(folder/'source-groom.json'));data=json.loads((folder/'source-groom.json').read_text());bp=_blueprint(unreal,actor);before=None
    if bp:
        defaults=_default_groom(unreal,bp,settings['component']);old=defaults.get_editor_property('groom_asset');old_binding=defaults.get_editor_property('binding_asset')
        before=dict(blueprint=bp.get_path_name(),groom=old.get_path_name() if old else None,binding=old_binding.get_path_name() if old_binding else None,metadata=unreal.EditorAssetLibrary.get_metadata_tag(bp,'CreateMetaHuman.GroomRevision.'+settings['component']))
    files={file.name:_digest(file) for file in folder.iterdir() if file.is_file() and (file.name.startswith(('anatomy.json','neutral.json')) or file.name=='source-groom.json')};owner=runpy.run_path(str(Path(__file__).with_name('process_ref.py')))['identify'](os.getpid())
    report=dict(schema=1,status='complete',actor=actor.get_path_name(),actor_label=actor.get_actor_label(),component=settings['component'],groom=groom.get_path_name(),binding=binding.get_path_name(),target=target.get_path_name(),source_mesh=source.get_path_name() if source else None,reference_mesh=reference.get_path_name(),interpolation_points=binding.get_editor_property('num_interpolation_points'),matching_section=binding.get_editor_property('matching_section'),source_groom_sha256=hashlib.sha256(json.dumps(data,sort_keys=True).encode()).hexdigest(),settings=settings,blueprint_before=before,owner=owner,inputs=files)
    report['binding_settings']=_api()['binding_settings'](binding)
    _write(folder/'capture.json',report);return report


def _stage(folder):
    import unreal
    capture=json.loads((folder/'capture.json').read_text());fit=json.loads((folder/'fit-result.json').read_text());candidate=folder/'fitted-groom.json'
    if fit.get('status')!='ready' or fit.get('capture_sha256')!=_digest(folder/'capture.json') or fit.get('output_sha256')!=_digest(candidate):raise ValueError('The fitted lash payload changed or is incomplete')
    for name,expected in capture['inputs'].items():
        path=(folder/name).resolve()
        if path.parent!=folder or _digest(path)!=expected:raise ValueError('A captured lash input changed after fitting')
    actor=_actor(unreal,capture['actor']);component=_component(unreal,actor,capture['component']);old=component.get_editor_property('groom_asset');old_binding=component.get_editor_property('binding_asset')
    if not old or old.get_path_name()!=capture['groom'] or not old_binding or old_binding.get_path_name()!=capture['binding']:raise ValueError('The artist changed the lash groom or binding during fitting; the proposal was retained')
    source=old_binding.get_editor_property('source_skeletal_mesh')
    if capture.get('binding_settings') and _api()['binding_settings'](old_binding)!=capture['binding_settings']:
        raise ValueError('The artist changed lash binding settings during fitting; retain those edits and capture a new proposal')
    if (source.get_path_name() if source else None)!=capture['source_mesh'] or old_binding.get_editor_property('matching_section')!=capture['matching_section'] or old_binding.get_editor_property('num_interpolation_points')!=capture['interpolation_points']:raise ValueError('The artist changed the binding transfer settings during fitting')
    api=_api();api['export_groom'](old.get_path_name(),str(folder/'verify-groom.json'));current=json.loads((folder/'verify-groom.json').read_text())
    if hashlib.sha256(json.dumps(current,sort_keys=True).encode()).hexdigest()!=capture['source_groom_sha256']:raise ValueError('The artist edited the source groom while fitting; keep those edits and prepare a new proposal')
    reference=unreal.load_asset(capture['reference_mesh']);faces=[c for c in actor.get_components_by_class(unreal.SkeletalMeshComponent) if c.get_skeletal_mesh_asset()==old_binding.get_editor_property('target_skeletal_mesh')]
    if len(faces)!=1 or old_binding.get_editor_property('target_skeletal_mesh').get_path_name()!=capture['target']:raise ValueError('The character binding target changed during fitting')
    if not unreal.CreateMHExpressionLibrary.export_engine_control_pose(reference,'{}',str(folder/'verify-neutral.json'),faces[0] if capture['source_mesh'] is None else None,True,0):raise ValueError('Could not verify the current binding reference')
    old_meta=json.loads((folder/'neutral.json').read_text());new_meta=json.loads((folder/'verify-neutral.json').read_text())
    for key in ('neutral_file','triangles_file','uv_file','dna_mesh_file','dna_vertex_file'):
        if _digest(folder/old_meta[key])!=_digest(folder/new_meta[key]):raise ValueError('The reference geometry or UVs changed during fitting; the source was retained')
    destination='/Game/CreateMetaHuman/Grooms/LashFit_'+folder.name;library=unreal.EditorAssetLibrary
    if library.does_directory_exist(destination):raise ValueError('This lash proposal already has native assets; inspect its original receipt')
    data=json.loads(candidate.read_text());groom=unreal.CreateMHGroomLibrary.import_groom_json(str(candidate),destination+'/Lashes')
    if not groom:raise RuntimeError('Could not build the fitted strand copy')
    _write(folder/'native-assets.json',dict(groom=groom.get_path_name()))
    api['apply_settings'](groom,data['group_settings'],list(dict.fromkeys(int(c['group']) for c in data['curves'])));assert library.save_loaded_asset(groom,False)
    binding=api['copy_binding'](old_binding,groom,old_binding.get_editor_property('target_skeletal_mesh'),destination+'/Binding')
    if not binding or not library.save_loaded_asset(binding,False):raise RuntimeError('Could not build the fitted lash binding')
    api['export_groom'](groom.get_path_name(),str(folder/'proposal-groom.json'));proposal=json.loads((folder/'proposal-groom.json').read_text());proposal_sha=hashlib.sha256(json.dumps(proposal,sort_keys=True).encode()).hexdigest()
    if component.get_editor_property('groom_asset')!=old or component.get_editor_property('binding_asset')!=old_binding:raise ValueError('The artist changed lashes during native binding creation; the new assets were retained without applying them')
    before=capture['blueprint_before'];report=dict(schema=1,status='complete',mode='PREVIEW',actor=capture['actor'],actor_label=actor.get_actor_label(),groom=groom.get_path_name(),binding=binding.get_path_name(),target=capture['target'],origin=str(folder),can_save_blueprint=bool(before and before['groom'] and before['binding']),proposal_sha256=proposal_sha,proof=fit['proof'],source_assets_saved=0)
    materials=[component.get_material(i) for i in range(component.get_num_materials())];groups=list(component.get_editor_property('groom_groups_desc'))
    try:
        with unreal.ScopedEditorTransaction('Preview fitted character lashes'):
            actor.modify();component.modify();component.set_groom_asset(groom);component.set_binding_asset(binding)
            if materials!=[component.get_material(i) for i in range(component.get_num_materials())]:raise RuntimeError('The proposal did not preserve the component material overrides')
            component.set_editor_property('groom_groups_desc',groups)
            _write(folder/'preview.json',report)
    except Exception:
        component.set_groom_asset(old);component.set_binding_asset(old_binding);component.set_editor_property('groom_groups_desc',groups);raise
    return report


def begin(folder,python,settings):
    """Export once, fit outside Unreal, then preview on the same guarded actor."""
    import unreal
    folder=Path(folder).resolve()
    if (folder/'capture.json').exists() or (folder/'worker-process.json').exists():raise ValueError('Observe the original lash request; do not dispatch it twice')
    active=getattr(unreal,'_cm_native_lash_fits',{})
    if active:raise ValueError('Wait for the active native lash fit before starting another')
    if not Path(python).is_file():raise ValueError('Choose the configured inference Python environment')
    if not .01<=settings['maximum_offset_fraction']<=.5 or not 0<=settings['clearance_fraction']<=.02:raise ValueError('Lash fitting bounds are outside the supported physical range')
    _write(folder/'progress.json',dict(phase='Capturing the selected character'))
    _capture(folder,settings);log=(folder/'worker.log').open('w',encoding='utf-8');worker=subprocess.Popen([python,str(Path(__file__).with_name('lash_fit_worker.py')),str(folder)],stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0));state=dict(worker=worker,log=log,handle=None,busy=False)
    try:
        identity=runpy.run_path(str(Path(__file__).with_name('process_ref.py')))['identify'](worker.pid);_write(folder/'worker-process.json',identity);_write(folder/'progress.json',dict(phase='Fitting roots to anatomical lid rims'))
    except Exception:
        worker.terminate();log.close();raise
    def tick(delta):
        if state['busy']:return
        if worker.poll() is None:return
        state['busy']=True
        try:
            if worker.returncode:raise RuntimeError(json.loads((folder/'fit-result.json').read_text()).get('error','The numerical lash fit failed'))
            _write(folder/'progress.json',dict(phase='Building the native lash preview'));report=_stage(folder)
        except Exception as error:
            report=dict(status='failed',error=str(error),origin=str(folder))
            if (folder/'native-assets.json').is_file():report['partial_assets']=json.loads((folder/'native-assets.json').read_text())
        finally:
            try:_write(folder/'receipt.json',report)
            finally:unreal.unregister_slate_post_tick_callback(state['handle']);log.close();active.pop(str(folder),None)
    state['handle']=unreal.register_slate_post_tick_callback(tick);active[str(folder)]=(state,tick);unreal._cm_native_lash_fits=active;return dict(status='queued',receipt_delegated=True)


def finish(origin,restore=False):
    """Keep the reviewed pair in its blueprint, or restore the preceding state."""
    import unreal
    folder=Path(origin).resolve();capture=json.loads((folder/'capture.json').read_text());preview=json.loads((folder/'preview.json').read_text());actor=_actor(unreal,capture['actor']);component=_component(unreal,actor,capture['component']);library=unreal.EditorAssetLibrary
    if preview.get('mode') not in {'PREVIEW','SAVED'}:raise ValueError('This lash proposal is no longer active')
    if component.get_editor_property('groom_asset').get_path_name()!=preview['groom'] or component.get_editor_property('binding_asset').get_path_name()!=preview['binding']:raise ValueError('Retain later artist groom changes before replacing this lash revision')
    before=capture['blueprint_before'];api=_api()
    api['export_groom'](unreal.load_asset(preview['groom']).get_path_name(),str(folder/'verify-proposal.json'));proposal=json.loads((folder/'verify-proposal.json').read_text())
    if hashlib.sha256(json.dumps(proposal,sort_keys=True).encode()).hexdigest()!=preview['proposal_sha256']:raise ValueError('The fitted groom was edited after preview; retain those artist edits before this operation')
    if not restore or preview['mode']=='SAVED':
        if not before:raise ValueError('This actor has no project blueprint to save')
        bp=unreal.load_asset(before['blueprint']);defaults=_default_groom(unreal,bp,capture['component']);expected=preview['groom'] if restore else before['groom']
        current=defaults.get_editor_property('groom_asset');current_binding=defaults.get_editor_property('binding_asset');expected_binding=preview['binding'] if restore else before['binding']
        if not current or current.get_path_name()!=expected or not current_binding or current_binding.get_path_name()!=expected_binding:raise ValueError('The blueprint lashes changed since preview; keep the newer artist edit')
        g=unreal.load_asset(before['groom'] if restore else preview['groom']);b=unreal.load_asset(before['binding'] if restore else preview['binding']);api['apply_owned_revision'](bp,capture['component'],g,b,owner_folder=bp.get_path_name().rsplit('/',1)[0],expected_original=expected,compile_blueprint=False)
        if restore:
            tag='CreateMetaHuman.GroomRevision.'+capture['component']
            if before['metadata']:library.set_metadata_tag(bp,tag,before['metadata'])
            else:library.remove_metadata_tag(bp,tag)
            library.save_loaded_asset(bp,False)
    actor=_actor(unreal,capture['actor']);component=_component(unreal,actor,capture['component']);g=unreal.load_asset(capture['groom'] if restore else preview['groom']);b=unreal.load_asset(capture['binding'] if restore else preview['binding'])
    with unreal.ScopedEditorTransaction('Restore previous lashes' if restore else 'Keep fitted character lashes'):
        actor.modify();component.modify();component.set_groom_asset(g);component.set_binding_asset(b)
    preview.update(mode='RESTORED' if restore else 'SAVED',character_blueprint_saved=bool(not restore or preview['mode']=='SAVED'));_write(folder/'preview.json',preview);return preview
