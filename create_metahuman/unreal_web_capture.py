"""Capture one explicit native actor/Creator character with evaluated appearance."""
import hashlib,json,runpy
from pathlib import Path


def api(name):return runpy.run_path(str(Path(__file__).with_name(name+'.py')))


def capture_actor(actor,directory,*,size=8192,detail_size=2048,character='',progress=None,face_samples=True,source_owner=None):
    import unreal
    root=Path(directory).resolve()
    if root.exists():raise ValueError('Use a new native capture folder; prior captures are retained')
    if type(face_samples) is not bool:raise ValueError('Choose whether native facial samples should be captured')
    face=next((component for component in actor.get_components_by_class(unreal.SkeletalMeshComponent) if component.get_name()=='Face' and component.get_skeletal_mesh_asset()),None)
    if face_samples and face:
        api('unreal_pose_bank')['supported'](face.get_skeletal_mesh_asset())
        if not hasattr(unreal.CreateMHExpressionLibrary,'supports_material_curve_sampling') or not unreal.CreateMHExpressionLibrary.supports_material_curve_sampling():raise ValueError('Install bridge 0.17 or newer to capture native facial samples with animated skin drivers')
        if face.get_editor_property('disable_post_process_blueprint'):raise ValueError('Enable face post-processing before capturing native facial samples, or disable the facial-sample capture option')
    if progress:progress('source',0,1)
    source=api('unreal_web')['export_actor'](actor,root)
    source.update(source_character=character,source_actor=actor.get_path_name(),source_project_file=str(Path(unreal.Paths.get_project_file_path()).resolve()),source_stage='Explicit native actor capture',cloud_requests=False)
    if face_samples and face:
        bank=api('unreal_pose_bank')['capture'](face.get_skeletal_mesh_asset(),root/'native-face-bank',progress,source_owner=source_owner,source_component=face)
        bank_path=root/'native-face-bank/bank.json'
        source['expression_bank']={'file':'native-face-bank/bank.json','sha256':hashlib.sha256(bank_path.read_bytes()).hexdigest(),'groups':len(bank['groups']),'samples':len(bank['poses'])-1,'face_component':'Face'}
        if bank.get('source_owner'):source['face_source_owner']={'asset':bank['source_owner'],'sha256':bank['source_sha256']}
    path=root/'source-manifest.json';path.write_text(json.dumps(source,indent=2),encoding='utf-8')
    appearance=api('unreal_web_appearance')['bake'](actor,root,size=size,detail_size=detail_size,progress=progress)
    # Release the private full-resolution bake inputs before allocating the
    # separate animated-response pass; a synchronous Python capture has no
    # intervening editor ticks to run the normal periodic collection.
    unreal.SystemLibrary.collect_garbage()
    motion=api('unreal_web_skin_motion')['bake'](actor,root,size=512,progress=progress) if face_samples and face else None
    sha=hashlib.sha256(path.read_bytes()).hexdigest()
    provisional={'generator':'CreateMetaHuman.NativeWebGeometry','source_sha256':sha,'face_component':'Face','parts':[{'component':p['component']} for p in source['skeletal_meshes']]}
    planned=api('native_web_materials')['plan'](source,sha,provisional,appearance)
    if progress:progress('textures',0,len(planned['required_texture_assets']))
    pixels=api('unreal_web_textures')['export'](path,root/'textures',asset_paths=planned['required_texture_assets'])
    if pixels['unresolved']:raise ValueError('Capture has unresolved texture resources needed by the browser')
    result={'status':'complete','source':str(root),'appearance':str(root/'appearance-manifest.json'),'source_manifest_sha256':sha,
            'character':character,'unreal_project_file':source['source_project_file'],'native_assets_modified':False,'cloud_requests':False,'materials_baked':len(appearance['materials']),
            'texture_assets':len(planned['required_texture_assets']),'facial_samples':source.get('expression_bank',{}).get('samples',0),'limitations':appearance['limitations']}
    if motion:
        result['skin_motion']={'status':motion['status'],'file':str(root/'skin-motion/manifest.json') if motion['status']=='complete' else None,'browser_playback':False}
        if motion['status']=='complete':result['limitations']+=motion['limitations']+['Portable playback interpolates sampled responses and does not execute the UE shader graph']
    (root/'capture-result.json').write_text(json.dumps(result,indent=2),encoding='utf-8');return result


def capture(character,directory,*,actor_path='',expected_project='',size=8192,detail_size=2048,progress=None,face_samples=True):
    import unreal
    if Path(directory).exists():raise ValueError('Choose a new native capture destination; previous captures are retained')
    if expected_project and Path(expected_project).resolve()!=Path(unreal.Paths.get_project_file_path()).resolve():raise ValueError('Open the linked Unreal project before capturing its web appearance')
    actors=unreal.get_editor_subsystem(unreal.EditorActorSubsystem)
    if actor_path:
        selected=actors.get_selected_level_actors() if actor_path=='@selected' else [unreal.find_object(None,actor_path)]
        if len(selected)!=1 or not isinstance(selected[0],unreal.Actor):raise ValueError('Select exactly one Unreal character actor to capture')
        result=capture_actor(selected[0],directory,size=size,detail_size=detail_size,progress=progress,face_samples=face_samples)
        result.update(character=character,source_character_verified=False);return result
    asset=unreal.load_asset(character)
    if not isinstance(asset,unreal.MetaHumanCharacter):raise ValueError('Link a MetaHuman character or explicitly select a level actor')
    state=json.loads(unreal.CreateMHArtistLibrary.snapshot_object(asset))
    if any(row['package_dirty'] for row in state['objects']):raise ValueError('Save Creator artist edits before capturing them for web')
    subsystem=unreal.get_editor_subsystem(unreal.MetaHumanCharacterEditorSubsystem);editing=subsystem.is_object_added_for_editing(asset);actor=None;selection=list(actors.get_selected_level_actors())
    if not editing:
        api('resource_budget')['require']('native Creator capture initialization',8*1024**3)
        api('resource_budget')['require_disk']('native Creator capture initialization',Path(directory).parent,2*1024**3)
        if progress:progress('initializing_creator',0,1)
    if not editing and not subsystem.try_add_object_to_edit(asset):raise RuntimeError('Creator could not initialize this character from the installed content')
    if progress:progress('initializing_creator',1,1)
    try:
        rig=json.loads(unreal.CreateMHCharacterLibrary.inspect_character(asset))
        if rig.get('rig_state')!='RIGGED':raise ValueError('This character needs an existing native rig before web capture; no rig request was made')
        if rig.get('rig_request_pending') or rig.get('texture_request_pending'):raise ValueError('Wait for the existing Creator rig or texture operation before capturing a stable web source')
        actor=subsystem.spawn_meta_human_actor(asset,True)
        if not actor:raise RuntimeError('Creator did not provide a transient character actor')
        return capture_actor(actor,directory,size=size,detail_size=detail_size,character=asset.get_path_name(),progress=progress,face_samples=face_samples,source_owner=asset)
    finally:
        if actor:actors.destroy_actor(actor)
        if not editing:subsystem.remove_object_to_edit(asset)
        actors.set_selected_level_actors([item for item in selection if item and not item.is_actor_being_destroyed()])


def dispatch(character,folder,options,expected_project=''):
    """Leave the remote CoreTicker callback before invoking the native baker."""
    import os,unreal
    folder=Path(folder);receipt=folder.parent/'receipt.json';handle=[None]
    if receipt.exists() or (folder.parent/'dispatch.json').exists():raise ValueError('This native capture has already been dispatched; observe its original receipt')
    identity=api('process_ref')['identify'](os.getpid())
    (folder.parent/'dispatch.json').write_text(json.dumps(identity),encoding='utf-8')
    def progress(stage,index,total):
        pending=folder.parent/'progress.pending';pending.write_text(json.dumps({'phase':stage,'completed':index,'total':total}),encoding='utf-8');pending.replace(folder.parent/'progress.json')
    def run_once(delta):
        unreal.unregister_slate_post_tick_callback(handle[0])
        report={'action':'CAPTURE_NATIVE_WEB','character':character}
        try:report.update(capture(character,options['destination'],actor_path=options.get('actor',''),expected_project=expected_project,size=options['size'],detail_size=options['detail_size'],progress=progress,face_samples=options.get('face_samples',True)))
        except Exception as error:report.update(status='failed',error=str(error))
        finally:
            pending=receipt.with_suffix('.tmp');pending.write_text(json.dumps(report),encoding='utf-8');pending.replace(receipt)
    try:
        options=dict(options)
        if options.get('actor')=='@selected':
            actors=unreal.get_editor_subsystem(unreal.EditorActorSubsystem).get_selected_level_actors()
            if len(actors)!=1:raise ValueError('Select exactly one actor in Unreal before capturing')
            options['actor']=actors[0].get_path_name()
        handle[0]=unreal.register_slate_post_tick_callback(run_once)
        progress('scheduled',0,1)
    except Exception as error:
        pending=receipt.with_suffix('.tmp');pending.write_text(json.dumps({'action':'CAPTURE_NATIVE_WEB','character':character,'status':'failed','error':str(error)}),encoding='utf-8');pending.replace(receipt)
        return {'status':'failed','receipt':str(receipt)}
    return {'status':'scheduled','receipt':str(receipt)}
