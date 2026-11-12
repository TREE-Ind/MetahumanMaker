"""Native assembled-character clip preview for artists and agent tools."""
import math


def copy_pose_clip(source,mesh,destination,poses,*,raw_controls):
    """Create a private, labeled control review from native-validated samples.

    Uses the source clip's duration and the character's skeleton. Artist source
    animation is retained; only the new curves-only clip receives these keys.
    Returns sample times for exact posed rendering or an agent's pose picker.
    """
    import unreal,json
    poses=list(poses);known=set(raw_controls)
    if len(poses)<2 or len(poses)>64 or len({row['name'] for row in poses})!=len(poses):
        raise ValueError('Supply two to 64 distinctly named control poses')
    for row in poses:
        if not isinstance(row['name'],str) or not row['name'] or not isinstance(row['controls'],dict):raise ValueError('Each pose needs a name and raw controls')
        for key,value in row['controls'].items():
            if 'CTRL_expressions.'+key not in known or not isinstance(value,(int,float)) or not math.isfinite(value) or not 0<=value<=1:
                raise ValueError('A pose contains an unknown or unbounded native raw control: '+str(key))
    if source.get_play_length()<=0:raise ValueError('Choose a nonempty source control clip')
    clip=copy_control_clip(source,mesh,destination);library=unreal.AnimationLibrary;kind=unreal.RawCurveTrackTypes.RCT_FLOAT
    for name in library.get_animation_curve_names(clip,kind):library.remove_curve(clip,name,False)
    times=[clip.get_play_length()*i/(len(poses)-1) for i in range(len(poses))]
    for key in sorted({key for row in poses for key in row['controls']}):
        name='CTRL_expressions_'+key;library.add_curve(clip,name,kind,False)
        library.add_float_curve_keys(clip,name,times,[float(row['controls'].get(key,0.)) for row in poses])
    samples=[dict(name=row['name'],seconds=when,controls=row['controls']) for row,when in zip(poses,times)]
    unreal.EditorAssetLibrary.set_metadata_tag(clip,'CreateMetaHuman.ControlPoseSamples',json.dumps(samples))
    if not unreal.EditorAssetLibrary.save_loaded_asset(clip,False):raise RuntimeError('Could not save the private control review clip')
    return clip,samples


def copy_control_clip(source, mesh, destination):
    """Copy a curves-only control clip onto the character's own skeleton.

    A MetaHuman template clip can reference a different skeleton asset even
    when its control names match. Never alter that source clip or skeleton.
    Bone animation requires actual animation retargeting and is rejected.
    """
    import unreal
    if not isinstance(source,unreal.AnimSequence) or not isinstance(mesh,unreal.SkeletalMesh):
        raise ValueError('Supply a raw-control clip and the rigged face mesh')
    if unreal.AnimationLibrary.get_animation_track_names(source):
        raise ValueError('This clip contains bone tracks; retarget the animation before previewing it')
    library=unreal.EditorAssetLibrary
    if not destination.startswith('/Game/') or library.does_asset_exist(destination):raise ValueError('Choose a fresh project asset for the control clip')
    names=unreal.AnimationLibrary.get_animation_curve_names(source,unreal.RawCurveTrackTypes.RCT_FLOAT)
    if not names or any(not str(name).lower().startswith('ctrl_expressions_') for name in names):raise ValueError('A control clip needs native facial raw-control curves only: '+str([str(name) for name in names][:8]))
    before={str(name):unreal.AnimationLibrary.get_float_keys(source,name) for name in names}
    clip=library.duplicate_asset(source.get_path_name(),destination)
    if not clip:raise RuntimeError('Could not copy the facial control clip')
    native=getattr(unreal,'CreateMHExpressionLibrary',None)
    if not native or not hasattr(native,'prepare_control_clip'):raise ValueError('Update the native bridge to prepare character-specific control clips')
    if not native.prepare_control_clip(clip,mesh):raise RuntimeError('Native control clip preparation failed')
    if clip.get_editor_property('skeleton')!=mesh.get_editor_property('skeleton') or before!={str(name):unreal.AnimationLibrary.get_float_keys(clip,name) for name in names}:
        raise RuntimeError('The native control clip did not retain its controls and skeleton')
    if not library.save_loaded_asset(clip,False):raise RuntimeError('Could not save the character-specific control clip')
    return clip


def preview_clip(actor, animation, sample_time=0., *, force_lod=None,lock_lod_sync=False):
    """Evaluate a facial clip with its parent body in the editor.

    A blueprint recompile replaces components and resets their editor tick
    flags. Reacquire components here each time; a face-only editor tick can
    otherwise inherit a collapsed or rotated parent pose. No asset is saved.
    """
    import unreal
    if not isinstance(actor,unreal.Actor) or not isinstance(animation,unreal.AnimSequence):
        raise ValueError('Choose an assembled actor and a native animation sequence')
    sample_time=float(sample_time)
    if not math.isfinite(sample_time) or not 0<=sample_time<=animation.get_play_length():
        raise ValueError('Choose a sample inside the animation duration')
    if force_lod is not None and (type(force_lod) is not int or force_lod<0):
        raise ValueError('LOD must be a nonnegative native mesh LOD index')
    native=getattr(unreal,'CreateMHExpressionLibrary',None)
    if not native or not hasattr(native,'refresh_editor_preview_pose'):
        raise ValueError('Update the matching native bridge for immediate editor pose evaluation')
    components=actor.get_components_by_class(unreal.SkeletalMeshComponent)
    faces=[c for c in components if c.get_name()=='Face']
    if len(faces)!=1 or not faces[0].get_skeletal_mesh_asset():raise ValueError('Choose an actor with one assembled Face component')
    face=faces[0];mesh=face.get_skeletal_mesh_asset()
    if mesh.get_editor_property('skeleton')!=animation.get_editor_property('skeleton'):
        raise ValueError('The preview clip belongs to another face skeleton')
    if face.get_editor_property('disable_post_process_blueprint'):
        raise ValueError('Enable native face post-processing before previewing facial controls')
    # Force the mapped character group, rather than freezing a stale groom or
    # wardrobe LOD when only the face is changed. Keep sync ticking so its
    # requested LODs can become renderable after asynchronous streaming.
    if lock_lod_sync:
        if not hasattr(native,'refresh_editor_lod_sync'):raise ValueError('Update the matching native bridge for synchronized character LOD review')
        for sync in actor.get_components_by_class(unreal.LODSyncComponent):
            managed={str(row.get_editor_property('name')) for row in sync.get_editor_property('components_to_sync') if row.get_editor_property('sync_option')!=unreal.SyncOption.DISABLED}
            if face.get_name() not in managed:continue
            if force_lod is None and not sync.is_component_tick_enabled():continue
            global_lod=-2
            if force_lod is not None:
                mappings=sync.get_editor_property('custom_lod_mapping')
                mapping=next((list(value.get_editor_property('mapping')) for key,value in mappings.items() if str(key)==face.get_name()),None)
                choices=[i for i,value in enumerate(mapping) if value==force_lod] if mapping is not None else [force_lod]
                choices=[i for i in choices if i>=sync.get_editor_property('min_lod')]
                if not choices:raise ValueError('The native LOD mapping cannot provide this face LOD')
                global_lod=choices[0]
            sync.set_component_tick_enabled(True)
            if not native.refresh_editor_lod_sync(sync,global_lod):raise RuntimeError('Could not refresh the character LOD mapping')
    # Parent components must refresh before the face consumes their pose.
    for component in [c for c in components if c!=face]+[face]:
        component.set_update_animation_in_editor(True)
        component.set_editor_property('pause_anims',False)
        component.reinitialize_anim_nodes()
    if force_lod is not None:face.set_forced_lod(force_lod+1)
    face.play_animation(animation,False);face.set_play_rate(0.);face.set_position(sample_time,False)
    for component in [c for c in components if c!=face]+[face]:
        if component.get_skeletal_mesh_asset() and not native.refresh_editor_preview_pose(component):
            raise RuntimeError('Could not evaluate the editor pose on '+component.get_name())
    return dict(status='previewing',actor=actor.get_path_name(),mesh=mesh.get_path_name(),
                animation=animation.get_path_name(),sample_time=sample_time,
                editor_pose_dependencies=[c.get_name() for c in components],assets_saved=0)
