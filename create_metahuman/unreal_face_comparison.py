"""Temporary side-by-side review of a staged face using matching raw controls."""
import array,json,math,runpy,sys,uuid
from pathlib import Path


def review_api():return runpy.run_path(str(Path(__file__).with_name('unreal_face_review.py')))


def _dot(a,b):return a.x*b.x+a.y*b.y+a.z*b.z


def _unit(unreal,value):
    length=math.sqrt(_dot(value,value))
    if length<1e-8:raise ValueError('The review face has no usable camera frame')
    return value/length


def _head_bounds(manifest):
    """Use sealed native neutral positions when cached bounds lag the shape."""
    for name in manifest['input_sha256']:
        file=Path(name)
        if file.suffix!='.json':continue
        meta=json.loads(file.read_text())
        if meta.get('source_mesh')!=manifest['source_mesh'] or meta.get('lod')!=0 or meta.get('dna_mode')!='ASSEMBLED_EMBEDDED' or not isinstance(meta.get('raw_curves'),dict):continue
        if any(abs(value)>1e-7 for key,value in meta['raw_curves'].items() if key.lower().startswith(('ctrl_expressions.','ctrl_expressions_'))):continue
        def read(key,kind):
            name=meta[key];path=file.parent/name
            if Path(name).name!=name or str(path.resolve()) not in manifest['input_sha256']:raise ValueError('The neutral head bounds need sealed native arrays')
            values=array.array(kind);values.frombytes(path.read_bytes())
            if sys.byteorder!='little':values.byteswap()
            return values
        positions=read('vertices_file','f');meshes=read('dna_mesh_file','i')
        if len(positions)!=len(meshes)*3:raise ValueError('The neutral head reference is incomplete')
        points=[positions[i*3:i*3+3] for i,index in enumerate(meshes) if index==0]
        if not points or any(not math.isfinite(v) for p in points for v in p):raise ValueError('The neutral head reference contains invalid positions')
        return [[min(p[j] for p in points) for j in range(3)],[max(p[j] for p in points) for j in range(3)]]
    raise ValueError('Keep a sealed neutral native capture for comparison framing')


def _corners(unreal,face,local_bounds):
    transform=face.get_world_transform();low,high=local_bounds
    points=[unreal.MathLibrary.transform_location(transform,unreal.Vector(x,y,z)) for x in (low[0],high[0]) for y in (low[1],high[1]) for z in (low[2],high[2])]
    origin,extent,_=unreal.SystemLibrary.get_component_bounds(face)
    # Render-bounds padding prevents culling; it is not additional geometry.
    # Retain the pose-dependent center but do not let that padding dominate a
    # face comparison camera's fit.
    extent=extent/max(1.,float(face.get_editor_property('bounds_scale')))
    if min(extent.x,extent.y,extent.z)<=0:raise ValueError('Wait for both native faces to finish updating before framing')
    points.extend(origin+unreal.Vector(extent.x*x,extent.y*y,extent.z*z) for x in (-1,1) for y in (-1,1) for z in (-1,1))
    return points


def frame(unreal,state):
    """Fit sealed head geometry plus live bounds, retaining the prior viewport."""
    candidate=guard(unreal,state)
    if not candidate:raise ValueError('Start a staged comparison first')
    review=review_api();source=review['_actor'](unreal,state['actor']);comparison=state['comparison']
    if not source:raise ValueError('Return to the original review level before framing')
    if not comparison.get('head_bounds_local'):raise ValueError('End the older comparison and start it again to use geometry-based framing')
    faces=[next(c for c in review['_components'](unreal,a).values() if c.get_name()=='Face') for a in (source,candidate)]
    points=[]
    for face in faces:
        points.extend(_corners(unreal,face,comparison['head_bounds_local']))
    center=unreal.Vector(*[(min(getattr(p,a) for p in points)+max(getattr(p,a) for p in points))*.5 for a in ('x','y','z')])
    front=_unit(unreal,faces[0].get_right_vector());up=_unit(unreal,faces[0].get_up_vector());side=_unit(unreal,faces[0].get_forward_vector())
    tan_h=36./(2*65.);tan_v=20.25/(2*65.)
    distance=max(50.,max(_dot(p-center,front)+1.15*max(abs(_dot(p-center,side))/tan_h,abs(_dot(p-center,up))/tan_v) for p in points))
    levels=unreal.get_editor_subsystem(unreal.LevelEditorSubsystem);actors=unreal.get_editor_subsystem(unreal.EditorActorSubsystem)
    camera=review['_actor'](unreal,comparison.get('camera','')) if comparison.get('camera') else None
    if camera and comparison['tag'] not in [str(t) for t in camera.get_editor_property('tags')]:raise ValueError('The comparison camera identity changed; retain it for inspection')
    pilot=levels.get_pilot_level_actor()
    if not camera or pilot!=camera:
        previous=levels.get_level_viewport_camera_info('')
        if not previous:raise ValueError('Open a level viewport before framing the comparison')
        location,rotation=previous
        comparison['viewport_before']=dict(location=[location.x,location.y,location.z],rotation=[rotation.pitch,rotation.yaw,rotation.roll],pilot=pilot.get_path_name() if pilot else '')
    location=center+front*distance;rotation=unreal.MathLibrary.find_look_at_rotation(location,center)
    if not camera:
        camera=actors.spawn_actor_from_class(unreal.CineCameraActor,location,rotation,transient=True)
        if not camera:raise RuntimeError('Could not create the transient comparison camera')
        camera.set_actor_label('MetaHuman Maker - Comparison Camera');camera.set_editor_property('tags',[unreal.Name(comparison['tag'])]);comparison['camera']=camera.get_path_name()
    camera.set_actor_location_and_rotation(location,rotation,False,False)
    component=camera.get_cine_camera_component();filmback=component.get_editor_property('filmback');filmback.set_editor_property('sensor_width',36.);filmback.set_editor_property('sensor_height',20.25);component.set_editor_property('filmback',filmback);component.set_editor_property('current_focal_length',65.)
    focus=component.get_editor_property('focus_settings');focus.set_editor_property('focus_method',unreal.CameraFocusMethod.DISABLE);component.set_editor_property('focus_settings',focus)
    levels.pilot_level_actor(camera)
    comparison['framing']=dict(method='SEALED_NEUTRAL_HEAD_AND_COMPONENT_BOUNDS',distance_cm=distance,aspect_ratio=16/9,bounds_corners=len(points))


def actor_for(unreal,state,required=True):
    comparison=state.get('comparison')
    if not comparison:return None
    actor=review_api()['_actor'](unreal,comparison['actor'])
    if not actor:
        if required:raise ValueError('The comparison actor was removed; end the comparison before continuing')
        return None
    if comparison['tag'] not in [str(tag) for tag in actor.get_editor_property('tags')]:raise ValueError('The comparison actor identity changed; retain it for inspection')
    return actor


def guard(unreal,state):
    actor=actor_for(unreal,state)
    if actor:
        review=review_api();review['_guard'](review['_native'](unreal),review['_components'](unreal,actor),state['comparison']['expected'])
    return actor


def apply(unreal,state,controls,lod):
    actor=guard(unreal,state)
    if not actor:return
    review=review_api();components=review['_components'](unreal,actor);face=next(c for c in components.values() if c.get_name()=='Face')
    clip=review['_native'](unreal).create_control_preview_clip(face.get_skeletal_mesh_asset(),json.dumps(controls))
    if not clip:raise ValueError('The staged face cannot evaluate these controls; previous poses retained')
    preview=runpy.run_path(str(Path(__file__).with_name('unreal_actor_preview.py')))['preview_clip']
    before=state['comparison']['expected']
    try:
        preview(actor,clip,0.,force_lod=None if lod==-1 else lod,lock_lod_sync=True)
        state['comparison']['expected']=review['_snapshot'](review['_native'](unreal),components,True)
    except Exception:
        review['_restore'](review['_native'](unreal),components,before);raise


def stop(unreal,state):
    comparison=state.get('comparison')
    if not comparison:return
    actor=actor_for(unreal,state,False)
    review=review_api();levels=unreal.get_editor_subsystem(unreal.LevelEditorSubsystem);actors=unreal.get_editor_subsystem(unreal.EditorActorSubsystem)
    camera=review['_actor'](unreal,comparison.get('camera','')) if comparison.get('camera') else None
    if camera:
        if comparison['tag'] not in [str(t) for t in camera.get_editor_property('tags')]:raise ValueError('The comparison camera identity changed; retain it for inspection')
        if levels.get_pilot_level_actor()==camera:
            levels.eject_pilot_level_actor();before=comparison.get('viewport_before',{});pilot=review['_actor'](unreal,before.get('pilot','')) if before.get('pilot') else None
            if pilot:levels.pilot_level_actor(pilot)
            elif before:levels.set_level_viewport_camera_info(unreal.Vector(*before['location']),unreal.Rotator(*before['rotation']),'')
        if not actors.destroy_actor(camera):raise RuntimeError('Could not remove the temporary comparison camera')
    if actor and not unreal.get_editor_subsystem(unreal.EditorActorSubsystem).destroy_actor(actor):raise RuntimeError('Could not remove the temporary comparison actor')
    if 'background_throttle_before' in comparison:
        performance=unreal.get_default_object(unreal.load_class(None,'/Script/UnrealEd.EditorPerformanceSettings'))
        if not performance.get_editor_property('bThrottleCPUWhenNotForeground'):performance.set_editor_property('bThrottleCPUWhenNotForeground',comparison['background_throttle_before'])
    state.pop('comparison',None)


def begin(unreal,state,source,prepared_folder,spacing,lod,live_background=True):
    if state.get('comparison'):guard(unreal,state);return
    if state.get('look'):raise ValueError('Save or restore the temporary skin preview before comparing saved rigs')
    if type(spacing) not in (int,float) or not math.isfinite(spacing) or (spacing!=0 and not 20<=spacing<=300):raise ValueError('Choose automatic spacing (0) or a distance between 20 and 300 cm')
    prepared=Path(prepared_folder).resolve();assembly=runpy.run_path(str(Path(__file__).with_name('unreal_revision_bounds.py')))['read_assembly'](prepared)
    helpers=runpy.run_path(str(Path(__file__).with_name('unreal_lod_correctives.py')))
    manifest,mesh=helpers['validate_sources'](prepared)
    head_bounds=_head_bounds(manifest)
    if assembly.get('status')!='complete' or assembly.get('request_sha256')!=helpers['digest'](prepared/'manifest.json'):raise ValueError('Choose the completed stage for this proposal')
    review=review_api();components=review['_components'](unreal,source);face=components[state['face']]
    if face.get_skeletal_mesh_asset()!=mesh or review['_look_api']()['blueprint'](unreal,source).get_path_name()!=assembly['source_blueprint']:raise ValueError('Connect the original saved character for this staged rig')
    bp=unreal.load_asset(assembly['blueprint'])
    if not isinstance(bp,unreal.Blueprint):raise ValueError('The separate staged Blueprint is missing')
    if bp.get_path_name().split('.')[0] in {p.get_path_name() for p in unreal.EditorLoadingAndSavingUtils.get_dirty_content_packages()}:raise ValueError('Save the staged Blueprint edits before comparison')
    budget=runpy.run_path(str(Path(__file__).with_name('resource_budget.py')));budget['require']('side-by-side native facial review',2*budget['GIB'])
    subsystem=unreal.get_editor_subsystem(unreal.EditorActorSubsystem);selected=subsystem.get_selected_level_actors();actor=None
    try:
        direction=face.get_socket_location('FACIAL_L_Eye')-face.get_socket_location('FACIAL_R_Eye')
        length=math.sqrt(direction.x**2+direction.y**2)
        if length<.01:raise ValueError('The face has no stable horizontal eye-socket frame for comparison')
        if spacing==0:
            axis=unreal.Vector(direction.x/length,direction.y/length,0.);projection=[_dot(point,axis) for point in _corners(unreal,face,head_bounds)]
            spacing=max(20.,1.15*(max(projection)-min(projection)))
            if not math.isfinite(spacing) or spacing>300:raise ValueError('This face needs more than 300 cm comparison spacing; review its scene scale first')
        offset=unreal.Vector(direction.x/length*spacing,direction.y/length*spacing,0.)
        actor=subsystem.spawn_actor_from_object(bp,source.get_actor_location()+offset,source.get_actor_rotation(),transient=True)
        if not actor:raise RuntimeError('Could not spawn the transient staged review actor')
        actor.set_actor_scale3d(source.get_actor_scale3d());actor.set_actor_label('MetaHuman Maker - Staged Lid Comparison')
        tag='CreateMH.FaceComparison.'+uuid.uuid4().hex;actor.set_editor_property('tags',list(actor.get_editor_property('tags'))+[unreal.Name(tag)])
        staged=review['_components'](unreal,actor);faces=[c for c in staged.values() if c.get_name()=='Face']
        if len(faces)!=1 or faces[0].get_skeletal_mesh_asset().get_path_name()!=assembly['mesh']:raise ValueError('The staged Blueprint no longer uses the prepared face revision')
        state['comparison']=dict(actor=actor.get_path_name(),tag=tag,source_face=mesh.get_path_name(),candidate_face=assembly['mesh'],blueprint=bp.get_path_name(),spacing_cm=spacing,head_bounds_local=head_bounds,transient=True,expected=review['_snapshot'](review['_native'](unreal),staged))
        if live_background:
            performance=unreal.get_default_object(unreal.load_class(None,'/Script/UnrealEd.EditorPerformanceSettings'));state['comparison']['background_throttle_before']=bool(performance.get_editor_property('bThrottleCPUWhenNotForeground'));performance.set_editor_property('bThrottleCPUWhenNotForeground',False)
        apply(unreal,state,state['controls'],lod)
        frame(unreal,state)
    except Exception:
        if state.get('comparison'):stop(unreal,state)
        elif actor:subsystem.destroy_actor(actor)
        state.pop('comparison',None);raise
    finally:subsystem.set_selected_level_actors(selected)
