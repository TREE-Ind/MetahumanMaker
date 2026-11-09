"""Receipt-backed facial inspection, with explicit saved skin revisions."""
import json,os,runpy
from pathlib import Path


def _write(path,data):
    path=Path(path);temporary=path.with_suffix('.tmp');temporary.write_text(json.dumps(data,indent=2));temporary.replace(path)


def _native(unreal):
    api=getattr(unreal,'CreateMHExpressionLibrary',None)
    if not api or not hasattr(api,'get_face_raw_controls'):raise ValueError('Install native bridge 0.23 or newer for assembled facial review')
    return api


def _components(unreal,actor):
    return {c.get_path_name():c for c in actor.get_components_by_class(unreal.SkeletalMeshComponent) if c.get_skeletal_mesh_asset()}


def _snapshot(api,components,allow_transient=False):
    import unreal
    states={}
    actor=next(iter(components.values())).get_owner() if components else None
    sync={c.get_path_name():dict(ticking=c.is_component_tick_enabled(),forced_lod=c.get_editor_property('forced_lod'),managed=sorted(str(row.get_editor_property('name')) for row in c.get_editor_property('components_to_sync') if row.get_editor_property('sync_option')!=unreal.SyncOption.DISABLED)) for c in actor.get_components_by_class(unreal.LODSyncComponent)} if actor else {}
    for path,component in components.items():
        raw=api.capture_component_preview_state(component,allow_transient)
        if not raw:raise ValueError('Cannot safely restore '+component.get_name()+': finish custom/Sequencer animation or save its animation asset first')
        states[path]=json.loads(raw);states[path]['lod_sync']=sync;states[path]['lod_sync_driven']=any(component.get_name() in row['managed'] for row in sync.values())
    return states


def _restore(api,components,states):
    import unreal
    failed=[]
    actor=next(iter(components.values())).get_owner() if components else None
    sync={c.get_path_name():c for c in actor.get_components_by_class(unreal.LODSyncComponent)} if actor else {}
    recorded=next(iter(states.values())).get('lod_sync',{}) if states else {}
    if recorded and recorded.keys()!=sync.keys():raise ValueError('The character LODSync components changed; retain the later artist edits')
    for path in recorded:sync[path].set_component_tick_enabled(False)
    # Parent/body first, then face post-processing can consume its restored pose.
    order=sorted(states,key=lambda path:components[path].get_name()=='Face')
    for path in order:
        if not api.restore_component_preview_state(components[path],json.dumps(states[path])):failed.append(path)
    if failed:raise RuntimeError('Could not restore playback settings for '+', '.join(failed))
    # Initializing the face blueprint can change its parent's editor tick flag.
    # Finalize playback flags only after every component has initialized.
    for path,row in states.items():
        component=components[path];component.set_forced_lod(int(row['forced_lod']));component.set_editor_property('pause_anims',row['paused']);component.set_update_animation_in_editor(row['editor_tick'])
    for path,row in recorded.items():sync[path].set_editor_property('forced_lod',row['forced_lod']);sync[path].set_component_tick_enabled(row['ticking'])


def _guard(api,components,expected):
    current=_snapshot(api,components,True)
    if current.keys()!=expected.keys():raise ValueError('The character components changed during review; retain the later artist changes')
    for path,row in expected.items():
        for key,value in row.items():
            if key=='position' and row.get('playing') and row.get('rate'):continue
            if key=='forced_lod' and row.get('lod_sync_driven') and any(s['ticking'] and s['forced_lod']<0 for s in row.get('lod_sync',{}).values()):continue
            if current[path].get(key)!=value:raise ValueError('Playback changed outside this review on '+components[path].get_name()+' ('+key+': expected '+str(value)+', found '+str(current[path].get(key))+'); retain those edits before continuing')


def _look_api():return runpy.run_path(str(Path(__file__).with_name('unreal_face_look.py')))


def _comparison_api():return runpy.run_path(str(Path(__file__).with_name('unreal_face_comparison.py')))


def _actor(unreal,path):
    # EditorActorSubsystem deliberately omits transient preview actors.
    actor=unreal.find_object(None,path)
    world=unreal.get_editor_subsystem(unreal.UnrealEditorSubsystem).get_editor_world()
    return actor if isinstance(actor,unreal.Actor) and unreal.SystemLibrary.is_valid(actor) and actor.get_world()==world else None


def _report(unreal,state,actor,**extra):
    look=_look_api();skin=look['info'](unreal,actor)
    values={key:skin[key] for key in ('normal','micro')} if skin else None
    return dict(status='complete',mode=state['mode'],origin=state['origin'],actor=state['actor'],actor_label=state['actor_label'],raw_controls=state['raw_controls'],controls=state['controls'],preview_applied=state['preview_applied'],look_values=values,look_preview=bool(state.get('look')),skin_undo_available=look['undo_available'](unreal,actor),assets_saved=0,comparison=state.get('comparison'),calibration_capture=state.get('calibration_capture'),calibration_capture_folder=state.get('calibration_capture_folder',''),calibration_capture_attempt=state.get('calibration_capture_attempt'),calibration_preparation=state.get('calibration_preparation'),calibration_preparation_attempt=state.get('calibration_preparation_attempt'),calibration_staging=state.get('calibration_staging'),**extra)


def _reacquire(unreal,path):
    actors=unreal.get_editor_subsystem(unreal.EditorActorSubsystem);actor=_actor(unreal,path)
    if not actor:raise RuntimeError('The selected character could not be reacquired after compiling its skin revision')
    actors.set_selected_level_actors([actor]);return actor


def dispatch(folder,request):
    import unreal
    folder=Path(folder).resolve();api=_native(unreal)
    if unreal.get_editor_subsystem(unreal.LevelEditorSubsystem).is_in_play_in_editor():raise ValueError('Stop Play or Simulate before reviewing native facial controls')
    if getattr(unreal,'_cm_native_lash_fits',{}):raise ValueError('Wait for native lash fitting before changing the facial pose')
    if getattr(unreal,'_cm_lid_preparations',{}):raise ValueError('Wait for or cancel the original lid preparation before changing facial review')
    active=getattr(unreal,'_cm_face_review_sessions',{});action=request['action']
    if action=='START':
        for key,previous in list(active.items()):
            if not _actor(unreal,previous['actor']):
                _comparison_api()['stop'](unreal,previous);previous.update(mode='EXPIRED',comparison=None);_write(Path(previous['origin'])/'session.json',previous);active.pop(key,None)
        selected=unreal.get_editor_subsystem(unreal.EditorActorSubsystem).get_selected_level_actors()
        if len(selected)!=1:raise ValueError('Select one assembled character actor in Unreal')
        actor=selected[0];components=_components(unreal,actor);faces=[c for c in components.values() if c.get_name()=='Face']
        if active:
            if len(active)!=1:raise ValueError('Restore the active facial review before choosing another character')
            previous=next(iter(active.values()))
            if previous['actor']!=actor.get_path_name():raise ValueError('Select '+previous['actor_label']+' and reconnect, then restore its active review first')
            _guard(api,components,previous['expected'])
            _look_api()['guard'](unreal,actor,previous)
            _comparison_api()['guard'](unreal,previous)
            return _report(unreal,previous,actor,reconnected=True)
        if len(faces)!=1 or faces[0].get_editor_property('disable_post_process_blueprint'):raise ValueError('Select an assembled Face with native post-processing enabled')
        face=faces[0];controls=[name for name in api.get_face_raw_controls(face.get_skeletal_mesh_asset()) if name.startswith('CTRL_expressions.')]
        if not controls:raise ValueError('The selected face has no embedded native facial DNA controls')
        before=_snapshot(api,components);owner=runpy.run_path(str(Path(__file__).with_name('process_ref.py')))['identify'](os.getpid())
        state=dict(schema=1,mode='ACTIVE',origin=str(folder),actor=actor.get_path_name(),actor_label=actor.get_actor_label(),face=face.get_path_name(),raw_controls=controls,before=before,expected=before,owner=owner,controls={},preview_applied=False,look=None)
        _write(folder/'session.json',state);active[str(folder)]=state;unreal._cm_face_review_sessions=active
    else:
        origin=Path(request['origin']).resolve();state=json.loads((origin/'session.json').read_text())
        alive=runpy.run_path(str(Path(__file__).with_name('process_ref.py')))['alive']
        if (state.get('mode')!='ACTIVE' and action!='UNDO_LOOK') or not alive(state['owner']) or state['owner']['pid']!=os.getpid():raise ValueError('The original editor no longer owns this facial review; choose the character again')
        actor=_actor(unreal,state['actor'])
        if not actor:raise ValueError('Return to the original character level before continuing facial review')
        components=_components(unreal,actor);look=_look_api()
        # Upgrade a still-live review created before LODSync snapshots existed.
        # Capture its original instance setting before the first new pose owns it.
        if any('lod_sync' not in row for row in state['before'].values()):
            current=_snapshot(api,components,True)
            for path,row in current.items():
                state['before'][path]['lod_sync']=row['lod_sync'];state['expected'][path]['lod_sync']=row['lod_sync'];state['before'][path]['lod_sync_driven']=row['lod_sync_driven'];state['expected'][path]['lod_sync_driven']=row['lod_sync_driven']
        if action=='END_COMPARISON':
            _comparison_api()['stop'](unreal,state);_write(origin/'session.json',state);active[str(origin)]=state
            return _report(unreal,state,actor,message='Temporary comparison removed. Original character playback retained.')
        if action=='UNDO_LOOK':
            if active:raise ValueError('Restore the active facial review before undoing saved skin tuning')
            if unreal.get_editor_subsystem(unreal.EditorActorSubsystem).get_selected_level_actors()!=[actor]:raise ValueError('Select '+state['actor_label']+' to undo its saved skin tuning')
            playback=_snapshot(api,components);bp=look['blueprint'](unreal,actor);look['undo_preflight'](unreal,actor,bp);saved=look['undo'](unreal,bp);actor=_reacquire(unreal,state['actor']);look['attach_saved'](unreal,actor,bp);_restore(api,_components(unreal,actor),playback);state.update(mode='RESTORED',preview_applied=False,look=None,expected=_snapshot(api,_components(unreal,actor)));_write(origin/'session.json',state)
            report=_report(unreal,state,actor);report.update(assets_saved=1,saved_look=saved,message='Saved skin tuning undone. Current playback retained.');return report
        _guard(api,components,state['expected']);look['guard'](unreal,actor,state)
        comparison=_comparison_api();comparison['guard'](unreal,state)
        if state.get('comparison') and action not in {'APPLY','RESTORE','COMPARE_STAGED','FRAME_COMPARISON'}:raise ValueError('End the temporary comparison before changing saved rig or skin settings')
        if action=='RESTORE':
            comparison['stop'](unreal,state)
            look['restore'](unreal,actor,state)
            if state['preview_applied']:_restore(api,components,state['before'])
            state.update(mode='RESTORED',preview_applied=False,expected=_snapshot(api,components));_write(origin/'session.json',state);active.pop(str(origin),None)
        elif action=='APPLY':
            if state.get('comparison') and request.get('look'):raise ValueError('End comparison before changing skin tuning')
            controls=request['controls'];clip=api.create_control_preview_clip(components[state['face']].get_skeletal_mesh_asset(),json.dumps(controls))
            if not clip:raise ValueError('A requested control is missing or outside 0–1; the existing pose was retained')
            lod=request.get('lod',-1)
            if type(lod) is not int or not -1<=lod<=7:raise ValueError('Choose Keep Current LOD or a native face LOD from 0 to 7')
            preview=runpy.run_path(str(Path(__file__).with_name('unreal_actor_preview.py')))['preview_clip']
            checkpoint=look['checkpoint'](unreal,actor,state)
            previous={key:state[key] for key in ('expected','controls','preview_applied')}
            comparison_before=state['comparison']['expected'] if state.get('comparison') else None
            try:
                if 'look' in request:look['apply'](unreal,actor,state,request['look'])
                preview(actor,clip,0.,force_lod=None if lod==-1 else lod,lock_lod_sync=True)
                comparison['apply'](unreal,state,controls,lod)
                expected=_snapshot(api,components,True)
                state.update(expected=expected,controls=controls,preview_applied=True)
                _write(origin/'session.json',state);active[str(origin)]=state
            except Exception:
                _restore(api,components,previous['expected']);state.update(previous)
                if comparison_before:
                    candidate=comparison['actor_for'](unreal,state);_restore(api,_components(unreal,candidate),comparison_before);state['comparison']['expected']=comparison_before
                look['rollback'](unreal,actor,state,checkpoint);raise
        elif action=='COMPARE_STAGED':
            comparison['begin'](unreal,state,actor,request['preparation_folder'],request['spacing'],request.get('lod',0),request.get('live_background',True))
            _write(origin/'session.json',state);active[str(origin)]=state
            return _report(unreal,state,actor,message='Comparing the saved staged rig beside the original. Pose controls drive both; Restore Previous Playback removes the temporary copy.')
        elif action=='FRAME_COMPARISON':
            comparison['frame'](unreal,state);_write(origin/'session.json',state);active[str(origin)]=state
            return _report(unreal,state,actor,message='Framed both complete faces. Ending comparison restores the preceding viewport if this camera is still piloted.')
        elif action in {'PREPARE_LIDS','PREPARE_SCULPT','PREPARE_BANK','PREPARE_INTERVALS','PREPARE_NEUTRAL'}:
            return runpy.run_path(str(Path(__file__).with_name('unreal_lid_preparation.py')))['begin'](folder,request,state)
        elif action=='STAGE_LIDS':
            return runpy.run_path(str(Path(__file__).with_name('unreal_lid_staging.py')))['stage'](folder,request,state)
        elif action=='REVISE_BOUNDS':
            prepared=Path(request['preparation_folder']);base=json.loads((prepared/'native'/'assembly.json').read_text())
            if _look_api()['blueprint'](unreal,actor).get_path_name()!=base['source_blueprint']:raise ValueError('Connect the original character for this staged rig before revising bounds')
            reused=(prepared/'native'/'assembly-bounds.json').is_file();staged=runpy.run_path(str(Path(__file__).with_name('unreal_revision_bounds.py')))['create'](prepared)
            _guard(api,_components(unreal,actor),state['expected']);look['guard'](unreal,actor,state)
            state['calibration_staging']=staged;_write(origin/'session.json',state);active[str(origin)]=state
            report=_report(unreal,state,actor,source_assets_saved=0,message='Observed the existing bounds revision. No duplicate created.' if reused else 'Created a separate bounds revision. Neutral geometry and physics assets retained; animated motion still requires review.')
            report['assets_saved']=0 if reused else len(staged['created_assets']);return report
        elif action=='OPEN_STAGED':
            staged=runpy.run_path(str(Path(__file__).with_name('unreal_revision_bounds.py')))['read_assembly'](request['preparation_folder'])
            bp=unreal.load_asset(staged['blueprint'])
            if staged.get('status')!='complete' or not isinstance(bp,unreal.Blueprint):raise ValueError('The separate staged character is unavailable')
            unreal.EditorAssetLibrary.sync_browser_to_objects([bp.get_path_name()])
            opened=unreal.get_editor_subsystem(unreal.AssetEditorSubsystem).open_editor_for_assets([bp])
            if not opened:raise ValueError('Could not open the staged Blueprint; its saved asset was retained')
            state['calibration_staging']=staged;_write(origin/'session.json',state);active[str(origin)]=state
            return _report(unreal,state,actor,message='Opened the separate review Blueprint. Original actor retained; motion approval still required.')
        elif action=='CAPTURE_LIDS':
            recipes=runpy.run_path(str(Path(__file__).with_name('expression_controls.py')))
            samples=recipes['lid_capture_poses'](state['raw_controls'],request['capture_set'],request.get('capture_controls'))
            face=components[state['face']];mesh=face.get_skeletal_mesh_asset()
            count=unreal.get_editor_subsystem(unreal.SkeletalMeshEditorSubsystem).get_num_verts(mesh,0)
            if count<=0:raise ValueError('The selected face has no readable LOD0 vertices')
            budget=runpy.run_path(str(Path(__file__).with_name('resource_budget.py')))
            budget['require']('native eye-pose capture',max(2*budget['GIB'],count*640+512*budget['MIB']))
            estimated=count*192*len(samples)+512*budget['MIB']
            budget['require_disk']('native eye-pose capture',folder,estimated)
            output=folder/'capture'
            error=None
            try:
                capture=runpy.run_path(str(Path(__file__).with_name('unreal_lod_correctives.py')))['capture_poses'](mesh,samples,output,lods=[0],source_component=face)
                recipes['verify_lid_capture'](capture,mesh.get_path_name())
            except Exception as exc:
                error=str(exc)
                try:capture=json.loads((output/'capture.json').read_text())
                except (OSError,ValueError):capture=dict(status='failed',poses=[])
                capture.update(status='failed',error=error)
            _guard(api,components,state['expected']);look['guard'](unreal,actor,state)
            capture.update(capture_set=request['capture_set'],estimated_disk_bytes=estimated)
            if output.is_dir():_write(output/'capture.json',capture)
            state['calibration_capture_attempt']=dict(folder=str(output),capture=capture)
            if not error:state.update(calibration_capture=capture,calibration_capture_folder=str(output))
            _write(origin/'session.json',state);active[str(origin)]=state
            report=_report(unreal,state,actor)
            if error:report.update(status='failed',error=error)
            else:report['message']='Captured '+str(len(capture['poses']))+' native LOD0 eye poses. Playback retained; saved face fingerprint recorded.'
            return report
        elif action=='SAVE_LOOK':
            checkpoint=look['checkpoint'](unreal,actor,state)
            try:
                if 'look' in request:look['apply'](unreal,actor,state,request['look'])
                bp,source,values=look['saved_preflight'](unreal,actor,state)
            except Exception:look['rollback'](unreal,actor,state,checkpoint);raise
            look['restore'](unreal,actor,state)
            if state['preview_applied']:_restore(api,components,state['before'])
            state.update(mode='RESTORED',preview_applied=False);_write(origin/'session.json',state);active.pop(str(origin),None)
            try:
                saved=look['save'](unreal,bp,source,values);actor=_reacquire(unreal,state['actor']);look['attach_saved'](unreal,actor,bp);_restore(api,_components(unreal,actor),state['before']);state['expected']=_snapshot(api,_components(unreal,actor));_write(origin/'session.json',state)
            except Exception as error:
                report=_report(unreal,state,actor);report.update(status='failed',error=str(error),assets_saved=None,message='Preview restored; inspect the character material revision before retrying');return report
            report=_report(unreal,state,actor);report.update(assets_saved=2,saved_look=saved,message='Skin tuning saved to a private material copy. Previous playback restored.');return report
        else:raise ValueError('Unknown native facial review operation')
    return _report(unreal,state,actor)
