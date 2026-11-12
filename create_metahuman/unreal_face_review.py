"""Receipt-backed facial inspection, with explicit saved skin revisions."""
import json,math,os,runpy,time
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
    lod_state=json.loads(api.capture_editor_lod_state(actor)) if actor and hasattr(api,'capture_editor_lod_state') else None
    for path,component in components.items():
        raw=api.capture_component_preview_state(component,allow_transient)
        if not raw:raise ValueError('Cannot safely restore '+component.get_name()+': finish custom/Sequencer animation or save its animation asset first')
        states[path]=json.loads(raw);states[path]['lod_sync']=sync;states[path]['lod_sync_driven']=any(component.get_name() in row['managed'] for row in sync.values())
        if lod_state is not None:states[path]['component_lods']=lod_state
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
    if hasattr(api,'refresh_editor_preview_pose'):
        for path in order:
            if not api.refresh_editor_preview_pose(components[path]):raise RuntimeError('Could not evaluate restored playback on '+components[path].get_name())
    # Initializing the face blueprint can change its parent's editor tick flag.
    # Finalize playback flags only after every component has initialized.
    for path,row in states.items():
        component=components[path];component.set_forced_lod(int(row['forced_lod']));component.set_editor_property('pause_anims',row['paused']);component.set_update_animation_in_editor(row['editor_tick'])
    lod_state=next(iter(states.values())).get('component_lods') if states else None
    if lod_state is not None:
        if not api.restore_editor_lod_state(actor,json.dumps(lod_state)) or json.loads(api.capture_editor_lod_state(actor))!=lod_state:
            raise RuntimeError('Could not restore original character render/streaming LOD requests')
    for path,row in recorded.items():
        if hasattr(api,'refresh_editor_lod_sync'):
            if not api.refresh_editor_lod_sync(sync[path],row['forced_lod'],False):raise RuntimeError('Could not restore the character LODSync setting')
        else:sync[path].set_editor_property('forced_lod',row['forced_lod'])
        sync[path].set_component_tick_enabled(row['ticking'])


def _guard(api,components,expected):
    current=_snapshot(api,components,True)
    if current.keys()!=expected.keys():raise ValueError('The character components changed during review; retain the later artist changes')
    for path,row in expected.items():
        for key,value in row.items():
            if key=='position' and row.get('playing') and row.get('rate'):continue
            if key=='forced_lod' and row.get('lod_sync_driven') and any(s['ticking'] for s in row.get('lod_sync',{}).values()):continue
            if key=='component_lods':
                managed={name for sync in row.get('lod_sync',{}).values() if sync['ticking'] for name in sync['managed']}
                stable=lambda values:{path:data for path,data in values.items() if path.rsplit('.',1)[-1] not in managed}
                if stable(current[path].get(key,{}))==stable(value):continue
            if current[path].get(key)!=value:raise ValueError('Playback changed outside this review on '+components[path].get_name()+' ('+key+': expected '+str(value)+', found '+str(current[path].get(key))+'); retain those edits before continuing')


def _look_api():return runpy.run_path(str(Path(__file__).with_name('unreal_face_look.py')))


def _comparison_api():return runpy.run_path(str(Path(__file__).with_name('unreal_face_comparison.py')))


def _skin_driver_state(unreal,actor):
    """Read evaluated material uniforms; controls alone do not prove skin response."""
    skin=_look_api()['info'](unreal,actor)
    if not skin:return dict(available=False,reason='No supported native head skin material')
    values=skin['uniforms']
    drivers={name:value for name,value in values['scalars'].items()
        if name.startswith(('head_wm','head_cm'))}
    if any(not math.isfinite(value) for value in drivers.values()):
        raise ValueError('Native skin expression drivers contain non-finite values')
    return dict(available=bool(drivers),material=skin['material'],values=drivers,
        active_count=sum(abs(value)>1e-6 for value in drivers.values()),driver_count=len(drivers),
        animated_textures={name:path for name,path in values['textures'].items() if 'animated' in name.casefold()},
        scope='Evaluated material inputs, not proof of visible wrinkle or color quality')


def _camera_state(unreal):
    """Record camera inputs without changing the artist's viewport or exposure."""
    try:
        levels=unreal.get_editor_subsystem(unreal.LevelEditorSubsystem)
        actor=levels.get_pilot_level_actor()
        if not isinstance(actor,unreal.CameraActor):
            return dict(available=False,reason='The current viewport is not piloting a camera actor')
        camera=next(iter(actor.get_components_by_class(unreal.CameraComponent)),None)
        if not camera:return dict(available=False,reason='The piloted actor has no camera component')
        settings=camera.get_editor_property('post_process_settings')
        keys=('override_auto_exposure_method','override_auto_exposure_bias',
              'override_auto_exposure_apply_physical_camera_exposure',
              'auto_exposure_bias','auto_exposure_apply_physical_camera_exposure',
              'camera_iso','camera_shutter_speed')
        exposure={key:settings.get_editor_property(key) for key in keys}
        exposure['auto_exposure_method']=str(settings.get_editor_property('auto_exposure_method'))
        return dict(available=True,actor=actor.get_path_name(),label=actor.get_actor_label(),
            exact_camera_view=levels.get_exact_camera_view(),
            post_process_blend_weight=float(camera.get_editor_property('post_process_blend_weight')),
            exposure=exposure,scope='Piloted camera inputs; volumes and viewport overrides can affect final exposure')
    except Exception as error:
        return dict(available=False,reason=str(error))


def _actor(unreal,path):
    # EditorActorSubsystem deliberately omits transient preview actors.
    actor=unreal.find_object(None,path)
    world=unreal.get_editor_subsystem(unreal.UnrealEditorSubsystem).get_editor_world()
    return actor if isinstance(actor,unreal.Actor) and unreal.SystemLibrary.is_valid(actor) and actor.get_world()==world else None


def _report(unreal,state,actor,**extra):
    look=_look_api();skin=look['info'](unreal,actor)
    values={key:skin[key] for key in ('normal','micro')} if skin else None
    return dict(status='complete',mode=state['mode'],origin=state['origin'],actor=state['actor'],actor_label=state['actor_label'],raw_controls=state['raw_controls'],controls=state['controls'],preview_applied=state['preview_applied'],look_values=values,look_preview=bool(state.get('look')),skin_undo_available=look['undo_available'](unreal,actor),assets_saved=0,comparison=state.get('comparison'),shaded_capture=state.get('shaded_capture'),calibration_capture=state.get('calibration_capture'),calibration_capture_folder=state.get('calibration_capture_folder',''),calibration_capture_attempt=state.get('calibration_capture_attempt'),calibration_preparation=state.get('calibration_preparation'),calibration_preparation_attempt=state.get('calibration_preparation_attempt'),calibration_staging=state.get('calibration_staging'),**extra)


def _begin_render_warmup(unreal,api):
    """Keep captures rendering without saving viewport/performance preferences."""
    native={}
    if hasattr(api,'begin_editor_viewport_refresh') and hasattr(api,'end_editor_viewport_refresh'):
        raw=api.begin_editor_viewport_refresh()
        if not raw:raise RuntimeError('Open a level viewport before capturing its shaded view')
        native=json.loads(raw)
        if not native.get('token') or not native.get('realtime_active'):
            if native.get('token'):api.end_editor_viewport_refresh(native['token'])
            raise RuntimeError('Could not temporarily enable viewport rendering')
    try:
        shared=getattr(unreal,'_cm_shaded_render_warmup',None)
        if shared is None:
            performance=unreal.get_default_object(unreal.load_class(None,'/Script/UnrealEd.EditorPerformanceSettings'))
            before=bool(performance.get_editor_property('bThrottleCPUWhenNotForeground'))
            performance.set_editor_property('bThrottleCPUWhenNotForeground',False,notify_mode=unreal.PropertyAccessChangeNotifyMode.NEVER)
            shared=dict(users=0,performance=performance,background_before=before)
            unreal._cm_shaded_render_warmup=shared
        shared['users']+=1
        return dict(native=native,closed=False,background_before=shared['background_before'])
    except Exception:
        if native.get('token'):api.end_editor_viewport_refresh(native['token'])
        raise


def _end_render_warmup(unreal,api,warmup):
    if warmup.get('closed'):return {}
    warmup['closed']=True
    result={}
    try:
        token=warmup.get('native',{}).get('token')
        if token:
            raw=api.end_editor_viewport_refresh(token)
            if not raw:raise RuntimeError('Could not release the capture viewport override')
            result['viewport']=json.loads(raw)
    finally:
        shared=getattr(unreal,'_cm_shaded_render_warmup',None)
        if shared:
            shared['users']-=1
            if shared['users']==0:
                shared['performance'].set_editor_property('bThrottleCPUWhenNotForeground',shared['background_before'],notify_mode=unreal.PropertyAccessChangeNotifyMode.NEVER)
                result['background_restored']=True
                del unreal._cm_shaded_render_warmup
            else:result['other_captures_active']=shared['users']
    return result


def _capture_view(unreal,folder,state,actor,api):
    """Capture the current shaded viewport without switching camera or pose."""
    folder=Path(folder);file=folder/'native-view.png'
    if file.exists():raise ValueError('Observe the original shaded capture; do not overwrite it')
    budget=runpy.run_path(str(Path(__file__).with_name('resource_budget.py')))
    budget['require']('native shaded capture',256*budget['MIB'])
    budget['require_disk']('native shaded capture',folder,64*budget['MIB'])
    owner=runpy.run_path(str(Path(__file__).with_name('process_ref.py')))['identify'](os.getpid())
    if not owner:raise RuntimeError('Could not identify the Unreal process for shaded capture recovery')
    _write(folder/'capture-owner.json',owner)
    # Passing a camera makes the automation API change the pilot. Capturing
    # the current viewport instead retains the artist's framing and camera.
    active=getattr(unreal,'_cm_shaded_views',{})
    if str(folder) in active:raise ValueError('Observe the original pending shaded capture before requesting another')
    curve_key=runpy.run_path(str(Path(__file__).with_name('expression_controls.py')))['native_curve_key']
    job=dict(task=None,handle=None,started=time.monotonic())
    job['warmup']=_begin_render_warmup(unreal,api)
    def tick(delta):
        report=None
        try:
            if time.monotonic()-job['started']>60:
                raise RuntimeError('The shaded screenshot did not finish within 60 seconds; inspect the original capture receipt before retrying')
            if hasattr(api,'refresh_editor_lod_sync'):
                for sync in actor.get_components_by_class(unreal.LODSyncComponent):
                    if sync.is_component_tick_enabled() and not api.refresh_editor_lod_sync(sync):raise RuntimeError('Could not refresh native LOD streaming')
            # A control command can finish before the editor's face tick and
            # temporal lighting have rendered the new pose. Let both settle.
            if job['task'] is None:
                unreal.get_editor_subsystem(unreal.LevelEditorSubsystem).editor_invalidate_viewports()
                if time.monotonic()-job['started']<2:return
                lods=json.loads(api.capture_editor_lod_state(actor)) if hasattr(api,'capture_editor_lod_state') else {}
                requested=state.get('lod',-1)
                if requested>=0 and lods.get(state['face'],{}).get('rendered')!=requested:
                    if time.monotonic()-job['started']<30:return
                    raise RuntimeError('The requested face LOD is unavailable or overridden; inspect native LOD settings and streaming before capturing')
                job['component_lods']=lods
                face=_components(unreal,actor)[state['face']];instance=face.get_anim_instance()
                evaluated={name:float(instance.get_curve_value(curve_key(name))) for name in state['controls']} if instance else {}
                if any(name not in evaluated or not math.isfinite(evaluated[name]) or abs(evaluated[name]-value)>1e-5 for name,value in state['controls'].items()):
                    raise RuntimeError('The native face has not evaluated the requested controls; apply the pose again before capturing')
                job['evaluated_controls']=evaluated
                job['camera_state']=_camera_state(unreal)
                job['task']=unreal.AutomationLibrary.take_high_res_screenshot(1920,1080,str(file),delay=.2)
                if not job['task'] or not job['task'].is_valid_task():raise RuntimeError('Open a rendered Unreal level viewport before capturing its view')
                return
            if not job['task'].is_task_done():return
            current=_actor(unreal,state['actor'])
            if not current:raise ValueError('The original actor closed during shaded capture')
            _guard(api,_components(unreal,current),state['expected']);_look_api()['guard'](unreal,current,state)
            if not file.is_file() or file.stat().st_size==0:raise RuntimeError('Unreal completed the screenshot without an image')
            captured=dict(file=str(file.resolve()),width=1920,height=1080,controls=dict(state['controls']),evaluated_controls=job['evaluated_controls'],component_lods=job['component_lods'],native_shading=True)
            captured['skin_expression_state']=_skin_driver_state(unreal,current)
            captured['camera_state']=job['camera_state']
            # component_lods stores forced requests for recovery, not the
            # renderer's selected hair representation. Keep these separate.
            groom_api=getattr(unreal,'CreateMHGroomLibrary',None)
            if groom_api and hasattr(groom_api,'capture_groom_representation_state'):
                captured['rendered_grooms']={}
                for component in current.get_components_by_class(unreal.GroomComponent):
                    # IsRegistered is not exposed on GroomComponent in UE's
                    # Python API. The native capture checks registration and
                    # returns an empty string for unavailable render state.
                    raw=groom_api.capture_groom_representation_state(component)
                    if raw:captured['rendered_grooms'][component.get_path_name()]=json.loads(raw)
            state['shaded_capture']=captured;_write(Path(state['origin'])/'session.json',state)
            report=_report(unreal,state,current,message='Saved the current shaded Unreal view. Camera, playback and assets retained.')
        except Exception as error:report=dict(status='failed',error=str(error))
        finally:
            if report is not None:
                try:
                    try:
                        released=_end_render_warmup(unreal,api,job['warmup'])
                        if report.get('status')=='complete':
                            captured=state['shaded_capture']
                            captured['render_warmup']=dict(viewport=job['warmup']['native'],release=released,
                                background_throttle_before=job['warmup']['background_before'],
                                scope='Temporary realtime/background rendering; texture residency and temporal convergence are not certified')
                            report['shaded_capture']=captured
                            _write(Path(state['origin'])/'session.json',state)
                    except Exception as error:report=dict(status='failed',error='Shaded capture cleanup failed: '+str(error))
                    _write(folder/'receipt.json',report)
                finally:unreal.unregister_slate_post_tick_callback(job['handle']);active.pop(str(folder),None)
    try:job['handle']=unreal.register_slate_post_tick_callback(tick)
    except Exception:
        _end_render_warmup(unreal,api,job['warmup']);raise
    active[str(folder)]=(job,tick);unreal._cm_shaded_views=active
    return dict(status='queued',receipt_delegated=True)


def _reacquire(unreal,path):
    actors=unreal.get_editor_subsystem(unreal.EditorActorSubsystem);actor=_actor(unreal,path)
    if not actor:raise RuntimeError('The selected character could not be reacquired after compiling its skin revision')
    actors.set_selected_level_actors([actor]);return actor


def dispatch(folder,request):
    import unreal
    folder=Path(folder).resolve();api=_native(unreal)
    if unreal.get_editor_subsystem(unreal.LevelEditorSubsystem).is_in_play_in_editor():raise ValueError('Stop Play or Simulate before reviewing native facial controls')
    if getattr(unreal,'_cm_shaded_views',{}):raise ValueError('Wait for the original shaded-view capture')
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
        if len(faces)!=1:raise ValueError('Select the assembled character actor with one Face component; currently selected: '+actor.get_actor_label())
        if faces[0].get_editor_property('disable_post_process_blueprint'):raise ValueError('Enable native post-processing on '+actor.get_actor_label()+' / Face before facial review')
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
        if action=='CAPTURE_VIEW':return _capture_view(unreal,folder,state,actor,api)
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
            previous={key:state[key] for key in ('expected','controls','preview_applied')};previous['lod']=state.get('lod',-1)
            comparison_before=state['comparison']['expected'] if state.get('comparison') else None
            try:
                if 'look' in request:look['apply'](unreal,actor,state,request['look'])
                preview(actor,clip,0.,force_lod=None if lod==-1 else lod,lock_lod_sync=True)
                comparison['apply'](unreal,state,controls,lod)
                expected=_snapshot(api,components,True)
                state.update(expected=expected,controls=controls,preview_applied=True,lod=lod)
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
