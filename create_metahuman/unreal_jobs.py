"""Network-only Unreal pushes without blocking Blender's UI.

Unreal owns import execution after dispatch. Closing/reloading Blender cannot
cancel that operation. Its receipt is written in Unreal, never inferred from a
client observation timeout. Network threads never use Blender APIs.
"""
from pathlib import Path
import json
import queue
import re
import shutil
import threading
import uuid
from datetime import datetime,timezone

import bpy
from bpy.app.handlers import persistent
from . import bridge, process_ref

_JOB = None


def receipt_time(receipt):
    """Order completed native actions using their actual observation/completion."""
    if not isinstance(receipt,dict):return 0.
    build=receipt.get('build')
    values=[receipt.get('inspected_at'),receipt.get('completed_at'),
            build.get('completed_at') if isinstance(build,dict) else None]
    times=[]
    for value in values:
        if not isinstance(value,str) or not value:continue
        try:
            parsed=datetime.fromisoformat(value.replace('Z','+00:00'))
            if parsed.tzinfo is None:parsed=parsed.replace(tzinfo=timezone.utc)
            times.append(parsed.timestamp())
        except (ValueError,OverflowError):continue
    return max(times,default=0.)


def restore_action_receipt(scene,receipt):
    """Recover action-specific data without rolling newer delivery state back."""
    f,s=scene.cm_features,scene.cm_settings
    try:current=json.loads(f.unreal_status_report or '{}')
    except ValueError:current={}
    if receipt_time(receipt)>=receipt_time(current):
        apply_receipt(scene,receipt)
        return
    # Each action still restores its own result and resources (including web
    # capture links). Its historical status must not replace a later inspection
    # or assembly, readiness, or the corresponding artist-facing message.
    names=('unreal_status_report','unreal_assembly_ready','unreal_push_status','last_action_error')
    previous={name:getattr(f,name) for name in names}
    status=s.status
    try:apply_receipt(scene,receipt)
    finally:
        for name,value in previous.items():setattr(f,name,value)
        s.status=status


def capture_progress(progress):
    labels={'scheduled':'Queued in Unreal','initializing_creator':'Opening character',
            'source':'Reading character and grooms','face_samples':'Capturing facial expressions',
            'appearance':'Baking skin and eyes','skin_motion':'Capturing animated skin',
            'textures':'Preparing texture resources'}
    label=labels.get(progress.get('phase'),'Capturing character')
    completed,total=progress.get('completed',0),progress.get('total',0)
    return label+(f' ({completed}/{total})' if type(completed) is int and type(total) is int and total>0 else '')


def running():
    return _JOB is not None


def _snapshot(bundle, folder):
    """Copy only declared inputs; never copy the surrounding project or archives."""
    data = json.loads((bundle / 'manifest.json').read_text(encoding='utf-8'))
    paths = [data['mesh'], *data.get('maps', []), *data.get('accessories', {}).values()]
    if data.get('body_mesh'):
        paths.append(data['body_mesh'])
    paths.extend(['manifest.json', 'unreal_import.py',*data.get('support_files',[])])
    inputs = []
    for relative in dict.fromkeys(paths):
        if Path(relative).is_absolute():
            raise ValueError('Export inputs must be relative to their bundle')
        source = (bundle / relative).resolve()
        if not source.is_relative_to(bundle.resolve()) or not source.is_file():
            raise ValueError('Export input is missing or outside the bundle: ' + relative)
        target = folder / relative
        if not target.resolve().is_relative_to(folder.resolve()):
            raise ValueError('Export input escapes the push snapshot: ' + relative)
        inputs.append((source, target))
    required = sum(source.stat().st_size for source, _ in inputs)
    if shutil.disk_usage(bundle).free < required + 2 * 1024**3:
        raise ValueError('Not enough disk space for this push and a 2 GiB reserve. Clear generated test files before retrying.')
    for source, target in inputs:
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
    # Import immutable copies, but retain the artist's persistent export paths
    # in native AssetImportData after the import has completed.
    data['reimport_root'] = str(bundle.resolve())
    (folder / 'manifest.json').write_text(json.dumps(data), encoding='utf-8')


def _send(job, engine, project):
    """Only ordinary Python/network/file operations are allowed in this thread."""
    try:
        _snapshot(job['bundle'], job['snapshot'])
        job['events'].put(('phase', 'Connecting to Unreal...'))
        bridge.push_bundle(engine, job['snapshot'], project,
                           receipt_path=job['receipt'], open_editor=True,
                           on_dispatch=job['dispatched'].set)
        job['events'].put(('returned', None))
    except Exception as exc:
        job['events'].put(('error', str(exc)))


def _character_action(job, engine, project, character, open_editor, expected_project, build_options=None):
    try:
        bridge.character_action(engine, character, job['snapshot'], project,
                                open_editor=open_editor, on_dispatch=job['dispatched'].set,expected_project=expected_project,build_options=build_options,
                                texture_update=job.get('kind')=='UPDATE_TEXTURES',eye_options=job.get('eye_options'),artist_destination=job.get('artist_destination',''),artist_actor=job.get('artist_actor',''),web_capture=job.get('web_capture'),texture_sources=job.get('kind')=='REQUEST_TEXTURE_SOURCES')
        job['events'].put(('returned', None))
    except Exception as exc:
        job['events'].put(('error', str(exc)))


def character_action(context, open_editor=False, build=False, eye_options=None,artist_capture=False,web_capture=None,texture_sources=False):
    global _JOB
    from . import groom_jobs, skin_native_jobs, rig_jobs
    rig_jobs.restore(context.scene)
    if _JOB or groom_jobs.running() or skin_native_jobs.running():
        raise ValueError('Wait for the current Unreal operation before checking the character')
    s, f = context.scene.cm_settings, context.scene.cm_features
    if texture_sources and (open_editor or build or eye_options is not None or artist_capture or web_capture is not None):
        raise ValueError('Download native texture sources as a separate operation')
    if texture_sources and not f.unreal_character_project:
        raise ValueError('The linked Unreal project is missing; refresh the original character delivery first')
    if texture_sources and restore_texture_update(context.scene,kind='REQUEST_TEXTURE_SOURCES'):
        raise ValueError('Resumed the original native texture-source request; wait for its receipt')
    if not f.unreal_character and not (web_capture and web_capture.get('actor')=='@selected'):
        raise ValueError('Push a character to Unreal first')
    if artist_capture and (open_editor or build or eye_options is not None):raise ValueError('Capture artist edits as a separate operation')
    if web_capture is not None and (open_editor or build or eye_options is not None or artist_capture):raise ValueError('Capture native web appearance as a separate operation')
    if web_capture is not None:
        if restore_texture_update(context.scene,kind='CAPTURE_NATIVE_WEB'):raise ValueError('The original web capture is still awaiting its Unreal receipt')
        if web_capture.get('size') not in (2048,4096,8192) or web_capture.get('detail_size') not in (512,1024,2048) or web_capture.get('actor','') not in ('','@selected'):raise ValueError('Invalid native capture quality or actor scope')
        if type(web_capture.get('face_samples',True)) is not bool:raise ValueError('Invalid native facial capture option')
    restore_build(context.scene)
    if _JOB:raise ValueError('The previous assembly is still awaiting its Unreal receipt. Do not assemble again.')
    if eye_options is not None:
        from . import character_project
        from .unreal_import import eye_options as validate_eyes
        if build or open_editor:raise ValueError('Update eye appearance as a separate action')
        if restore_texture_update(context.scene,kind='UPDATE_EYES'):
            raise ValueError('Wait for the original eye update receipt before updating again')
        character_project.preflight(context,s,f)
        eye_options=validate_eyes(eye_options)
        if bpy.ops.cm.save_project()!={'FINISHED'}:raise ValueError('Save the project before updating Creator eyes')
    build_options=None
    if build:
        from . import native_geometry_state
        native_geometry_state.guard(context)
        if context.mode!='OBJECT' or f.expression_active or f.body_editing or f.full_sculpt:
            raise ValueError('Return to Neutral and finish the active edit before assembling')
        from . import unreal_assembly
        build_options={'root':f.unreal_build_root,'name':s.character_name,'profile':f.unreal_build_profile,
                       'allow_low_memory':f.unreal_build_allow_low_memory,'preserve_source_textures':f.unreal_build_preserve_source_textures,'eye_overlay_policy':f.unreal_eye_overlay_policy}
        try:previous_build=json.loads(f.unreal_build_report or '{}')
        except ValueError:previous_build={}
        existing=unreal_assembly.linked_blueprint(build_options,previous_build,f.unreal_character)
        if existing:build_options['existing_blueprint']=existing
        unreal_assembly.options(build_options)
    engine = bpy.path.abspath(s.unreal_engine_dir)
    if not (Path(engine) / 'Engine/Plugins/Experimental/PythonScriptPlugin/Content/Python/remote_execution.py').is_file():
        raise ValueError('Choose the Unreal installation containing Engine/')
    if texture_sources:
        from . import unreal_texture_sources
        prepared=unreal_texture_sources.prepare(bpy.path.abspath(s.project_dir),s.character_name,f.unreal_character,f.unreal_character_project,s.unreal_project)
        folder=Path(prepared['folder']);snapshot=Path(prepared['snapshot'])
    else:
        folder = Path(bpy.path.abspath(s.project_dir)) / 'Export' / '.unreal-actions' / uuid.uuid4().hex[:12]
        snapshot = folder / 'scripts'
        snapshot.mkdir(parents=True, exist_ok=False)
        for name in ('unreal_character.py','unreal_assembly.py','unreal_assembly_textures.py','unreal_assembly_eyes.py','unreal_assembly_auxiliary.py','unreal_eye_overlays.py','process_ref.py','unreal_eyes.py','unreal_import.py','unreal_artist_state.py','unreal_web.py','resource_budget.py','unreal_groom.py','unreal_groom_representations.py','unreal_web_capture.py','unreal_web_appearance.py','unreal_web_bake_inputs.py','unreal_web_skin_motion.py','unreal_web_textures.py','native_web_materials.py','unreal_pose_bank.py','native_face_recipes.py'):
            shutil.copy2(Path(__file__).with_name(name),snapshot/name)
    job = {'scene': context.scene, 'folder': folder, 'snapshot': snapshot,
           'receipt': folder/'receipt.json', 'events':queue.Queue(),
           'dispatched':threading.Event(), 'error':None,
           'kind':'BUILD_CHARACTER' if build else 'OPEN_CHARACTER' if open_editor else 'CHARACTER_STATUS',
           'project_dir':str(Path(bpy.path.abspath(s.project_dir)).resolve()),'character_name':s.character_name,'character':f.unreal_character}
    if eye_options is not None:
        job.update(kind='UPDATE_EYES',eye_options=eye_options)
        f.unreal_eye_receipt=str(job['receipt']);f.unreal_eye_result=''
    if texture_sources:
        job['kind']='REQUEST_TEXTURE_SOURCES';f.unreal_sources_receipt=str(job['receipt']);f.unreal_sources_result=''
    if artist_capture:
        job.update(kind='CAPTURE_ARTIST_EDITS',artist_destination=str(Path(job['project_dir'])/'UnrealArtistEdits'/folder.name),artist_actor='@selected' if f.unreal_artist_scope=='SELECTED_ACTOR' else '')
        f.unreal_artist_receipt=str(job['receipt'])
    if web_capture is not None:
        web_capture=dict(web_capture,destination=str(Path(job['project_dir'])/'NativeWebCaptures'/folder.name))
        job.update(kind='CAPTURE_NATIVE_WEB',web_capture=web_capture)
        f.native_web_capture_receipt=str(job['receipt']);f.native_web_capture_result=''
    (folder/'request.json').write_text(json.dumps({'schema_version':1,'character':f.unreal_character,'action':job['kind'],
        'project':s.unreal_project,'expected_project':f.unreal_character_project,'build':build_options,'eyes':eye_options,'artist_actor':job.get('artist_actor',''),'web_capture':web_capture}),encoding='utf-8')
    if build:f.unreal_build_receipt=str(job['receipt'])
    if build or eye_options is not None or web_capture is not None or texture_sources:
        history=Path(job['project_dir'])/'Export'/re.sub(r'[^A-Za-z0-9_]','_',s.character_name).strip('_')/('last-texture-sources-action.json' if texture_sources else 'last-web-capture-action.json' if web_capture is not None else 'last-eye-action.json' if eye_options is not None else 'last-build-action.json')
        history.parent.mkdir(parents=True,exist_ok=True)
        record={'schema_version':1,'project_dir':job['project_dir'],'character_name':s.character_name,
                'character':f.unreal_character,'unreal_project_file':f.unreal_character_project,'receipt':str(job['receipt'])}
        temporary=history.with_suffix('.tmp');temporary.write_text(json.dumps(record,indent=2),encoding='utf-8');temporary.replace(history)
    job['thread']=threading.Thread(target=_character_action,args=(job,engine,s.unreal_project,f.unreal_character,open_editor,f.unreal_character_project,build_options),
                                   name='CreateMHCharacterStatus',daemon=True)
    _JOB=job
    f.unreal_push_status='Checking requirements and assembling in Unreal...' if build else 'Opening the last character in Creator...' if open_editor else 'Checking the last character in Unreal...'
    if eye_options is not None:f.unreal_push_status='Updating eye appearance in Creator. Blender remains usable.'
    if artist_capture:f.unreal_push_status='Preserving saved UE artist edits and native assets. Blender remains usable.'
    if web_capture is not None:f.unreal_push_status='Capturing native geometry and evaluating artist material properties. Blender remains usable.'
    if texture_sources:f.unreal_push_status='Requesting configured native texture sources. The original request will be tracked until it finishes.'
    s.status=f.unreal_push_status
    job['thread'].start()
    if not bpy.app.timers.is_registered(poll):
        bpy.app.timers.register(poll,first_interval=.2,persistent=True)


def update_textures(context, *, native_albedo=False, native_eyes=False, restore_eyes=False,native_selection=None):
    global _JOB
    from . import groom_jobs, skin_native_jobs, character_project, texture_update, guided
    if _JOB or groom_jobs.running() or skin_native_jobs.running():
        raise ValueError('Wait for the current Unreal operation before updating textures')
    s, f = context.scene.cm_settings, context.scene.cm_features
    if native_albedo and native_eyes or restore_eyes and not native_eyes:raise ValueError('Choose one native appearance operation at a time')
    if not f.unreal_character:
        raise ValueError('Push this character to Unreal before updating its textures')
    if restore_texture_update(context.scene):
        raise ValueError('Resumed the previous texture update. Wait for its receipt before another update')
    character_project.preflight(context, s, f)
    problems = [] if native_albedo or native_eyes else guided.check_delivery(context, s, f)
    if problems:
        raise ValueError(problems[0])
    engine = bpy.path.abspath(s.unreal_engine_dir)
    if not (Path(engine) / 'Engine/Plugins/Experimental/PythonScriptPlugin/Content/Python/remote_execution.py').is_file():
        raise ValueError('Choose the Unreal installation containing Engine/')
    if bpy.ops.cm.save_project() != {'FINISHED'}:
        raise ValueError('Save the character project before updating textures')
    problems = [] if native_albedo or native_eyes else guided.check_delivery(context, s, f)
    if problems:
        raise ValueError(problems[0])
    root = Path(bpy.path.abspath(s.project_dir)).resolve()
    folder = root / 'Export' / '.unreal-actions' / uuid.uuid4().hex[:12]
    if native_eyes:
        from . import native_eye_update
        native_eye_update.prepare(context,folder,restore=restore_eyes)
    elif native_albedo:
        from . import native_albedo_update
        native_albedo_update.prepare(context, folder)
    else:
        texture_update.prepare(context, folder,native_selection=native_selection)
    job = {'scene': context.scene, 'folder': folder, 'snapshot': folder / 'scripts',
           'receipt': folder / 'receipt.json', 'events': queue.Queue(), 'dispatched': threading.Event(),
           'error': None, 'kind': 'UPDATE_TEXTURES', 'project_dir': str(root), 'character_name': s.character_name,
           'character': f.unreal_character}
    history = root / 'Export' / re.sub(r'[^A-Za-z0-9_]', '_', s.character_name).strip('_') / 'last-texture-action.json'
    record = {'schema_version': 1, 'project_dir': str(root), 'character_name': s.character_name,
              'character': f.unreal_character, 'unreal_project_file': f.unreal_character_project, 'receipt': str(job['receipt'])}
    temporary = history.with_suffix('.tmp')
    history.parent.mkdir(parents=True, exist_ok=True)
    temporary.write_text(json.dumps(record, indent=2), encoding='utf-8')
    temporary.replace(history)
    job['thread'] = threading.Thread(target=_character_action,
        args=(job, engine, s.unreal_project, f.unreal_character, False, f.unreal_character_project),
        name='CreateMHTextureUpdate', daemon=True)
    _JOB = job
    f.unreal_texture_receipt = str(job['receipt'])
    f.unreal_texture_result = ''
    f.unreal_push_status = ('Applying selected albedos to the linked native character. Blender remains usable.' if native_albedo else
                           'Updating delivered textures in Unreal. Blender remains usable.')
    if native_eyes:f.unreal_push_status='Restoring previous native eye materials...' if restore_eyes else 'Applying projected source-eye PBR materials to Creator...'
    s.status = f.unreal_push_status
    job['thread'].start()
    if not bpy.app.timers.is_registered(poll):
        bpy.app.timers.register(poll, first_interval=.2, persistent=True)


def start(context, bundle):
    global _JOB
    from . import groom_jobs, skin_native_jobs
    if groom_jobs.running() or skin_native_jobs.running():
        raise ValueError('Wait for the current groom exchange before pushing a character')
    if _JOB:
        raise ValueError('A push is already being monitored. Wait for its Unreal receipt.')
    s, f = context.scene.cm_settings, context.scene.cm_features
    bundle = Path(bundle).resolve()
    folder = bundle.parent / '.unreal-jobs' / uuid.uuid4().hex[:12]
    folder.mkdir(parents=True, exist_ok=False)
    job = {'scene': context.scene, 'bundle': bundle, 'folder': folder,
           'snapshot': folder / 'bundle', 'receipt': folder / 'receipt.json',
           'events': queue.Queue(), 'dispatched': threading.Event(), 'error': None,
           'project_dir':str(Path(bpy.path.abspath(s.project_dir)).resolve()),
           'character_name':s.character_name}
    job['thread'] = threading.Thread(target=_send, args=(job, bpy.path.abspath(s.unreal_engine_dir), s.unreal_project),
                                     name='CreateMHUnrealPush', daemon=True)
    (folder / 'request.json').write_text(json.dumps({'bundle': str(bundle), 'project': s.unreal_project,
        'character_name': s.character_name, 'receipt': str(job['receipt'])}), encoding='utf-8')
    _JOB = job
    f.unreal_push_receipt = str(job['receipt'])
    f.unreal_push_status = 'Preparing the Unreal push...'
    s.status = 'Push started. Blender remains usable while Unreal imports the character.'
    job['thread'].start()
    if not bpy.app.timers.is_registered(poll):
        bpy.app.timers.register(poll, first_interval=.2, persistent=True)


def _scene(job):
    scene = job.get('scene')
    try:
        if scene and scene.name in bpy.data.scenes:
            return scene
    except ReferenceError:
        pass
    return None


def apply_receipt(scene, receipt):
    if receipt.get('action')=='REQUEST_RIG':
        f,s=scene.cm_features,scene.cm_settings
        if receipt.get('character')!=f.unreal_character:return
        f.last_action_error=receipt.get('status')!='complete'
        f.unreal_rig_result=json.dumps(receipt)
        if receipt.get('assembly'):
            f.unreal_status_report=json.dumps(receipt)
            f.unreal_assembly_ready=bool(receipt['assembly'].get('ready'))
        f.unreal_push_status=('Full rig saved. '+('Ready to assemble; local delivery checks still apply.' if f.unreal_assembly_ready else 'Download native texture sources or refresh remaining Creator requirements.')
            if receipt.get('status')=='complete' else 'Full rig action failed: '+receipt.get('error','Inspect its retained receipt and MetaHuman service log'))
        s.status=f.unreal_push_status
        return
    s, f = scene.cm_settings, scene.cm_features
    f.last_action_error = receipt.get('status') != 'complete'
    if receipt.get('action')=='HEAD_REVISION':
        f.unreal_revision_result=json.dumps(receipt)
        if receipt.get('status')=='complete':
            from . import native_geometry_state
            native_geometry_state.adopt(scene,receipt)
            f.unreal_revision_character=receipt['character'];f.unreal_character=receipt['character']
            f.unreal_revision_source='';f.unreal_revision_project=''
            f.unreal_character_project=receipt['unreal_project_file'];f.unreal_assembly_ready=False
            f.unreal_status_report=json.dumps(receipt);f.unreal_build_report=''
            f.unreal_texture_receipt='';f.unreal_texture_result='';f.unreal_eye_receipt='';f.unreal_eye_result=''
            f.native_web_capture_receipt='';f.native_web_capture_result='';f.native_web_source='';f.native_web_appearance=''
            f.unreal_artist_snapshot='';f.unreal_artist_review='';f.unreal_artist_receipt=''
            if hasattr(scene,'cm_groom'):scene.cm_groom.unreal_target=''
            f.unreal_push_status='Head revision saved with native artist settings. Open in Creator to review; the new geometry is unrigged.'
            s.stage='EXPORT'
        else:f.unreal_push_status='Head revision failed: '+receipt.get('error','Inspect its retained receipt and partial assets')
        s.status=f.unreal_push_status
        return
    if receipt.get('action')=='CAPTURE_NATIVE_WEB':
        f.native_web_capture_result=json.dumps(receipt)
        if receipt.get('status')=='complete':
            from . import native_web_jobs
            try:
                native_web_jobs.adopt_capture(scene,receipt)
                f.unreal_push_status='Native capture and artist appearance bakes are ready for rigged web export.'
            except (OSError,ValueError,KeyError) as error:
                f.last_action_error=True;f.unreal_push_status='Native capture was retained but could not be linked: '+str(error)
        else:f.unreal_push_status='Native web capture failed: '+receipt.get('error','Inspect the Unreal receipt')
        f.native_web_status=f.unreal_push_status;s.status=f.unreal_push_status
        return
    if receipt.get('action')=='CAPTURE_ARTIST_EDITS':
        if receipt.get('status')=='complete':
            from . import artist_state
            counts=artist_state.adopt_snapshot(scene,receipt['manifest'])
            f.unreal_push_status='Preserved UE artist edits: '+str(counts['objects'])+' objects and '+str(counts['files'])+' native files. Rendering adapters may still be required.'
        else:f.unreal_push_status='Artist-state capture failed: '+receipt.get('error','Inspect the receipt')
        s.status=f.unreal_push_status
        return
    if receipt.get('action')=='UPDATE_EYES':
        f.unreal_eye_result=json.dumps(receipt)
        f.unreal_push_status=('Creator eye appearance saved.'+(' Reassemble to refresh previously built eye materials.' if f.unreal_build_report else '')
            if receipt.get('status')=='complete' else 'Eye appearance update failed: '+receipt.get('error','Inspect the receipt'))
        s.status=f.unreal_push_status
        return
    if receipt.get('action') == 'UPDATE_TEXTURES':
        f.unreal_texture_result = json.dumps(receipt)
        if receipt.get('status')=='complete' and receipt.get('geometry_mode')=='NATIVE_SKIN_MAPS' and receipt.get('assembly'):
            f.unreal_status_report=json.dumps(receipt)
            f.unreal_assembly_ready=bool(receipt['assembly'].get('ready'))
        if receipt.get('geometry_mode')=='NATIVE_SOURCE_EYES':
            f.unreal_source_eye_result=json.dumps(receipt)
            if receipt.get('receipt'):f.unreal_source_eye_receipt=receipt['receipt']
        count = len(receipt.get('updated', []))
        if receipt.get('status') == 'complete':
            f.unreal_push_status = 'Updated ' + str(count) + ' textures on the existing Unreal character.'
            if receipt.get('requires_reassembly'):
                f.unreal_push_status += ' Reassemble to refresh previously built copies.'
            if receipt.get('retain_source_snapshot'):
                f.unreal_push_status += ' Some Reimport sources need repair; their snapshots were retained.'
            notes=[note for note in receipt.get('warnings',[]) if isinstance(note,str) and note.strip()]
            if notes:f.unreal_push_status+=' '+notes[0]+(' See the receipt for additional notes.' if len(notes)>1 else '')
        else:
            f.unreal_push_status = 'Texture update failed after ' + str(count) + ' imports: ' + receipt.get('error', 'Inspect the update receipt')
        if receipt.get('geometry_mode')=='NATIVE_SOURCE_EYES' and receipt.get('status')=='complete':
            f.unreal_push_status=('Previous native eye materials restored.' if receipt.get('operation')=='RESTORE_SOURCE_EYES' else
                                  'Projected source-eye PBR materials saved in Creator.')
            if receipt.get('requires_reassembly'):f.unreal_push_status+=' Reassemble to refresh previously built copies.'
        s.status = f.unreal_push_status
        return
    if receipt.get('action')=='REQUEST_TEXTURE_SOURCES':
        f.unreal_sources_result=json.dumps(receipt)
        f.unreal_status_report=json.dumps(receipt);f.unreal_assembly_ready=bool(receipt.get('assembly',{}).get('ready'))
        f.unreal_push_status=('Native texture sources saved. '+('Creator is ready to assemble; local map checks still apply.' if f.unreal_assembly_ready else 'Refresh remaining Creator requirements before assembly.')
            if receipt.get('status')=='complete' else 'Native texture-source request failed: '+receipt.get('error','Inspect the MetaHuman Message Log'))
        if receipt.get('review_required'):f.unreal_push_status+=' '+receipt['review_required']
        if receipt.get('preview_resolution_warning'):f.unreal_push_status+=' Stored source dimensions verified; native preview resolutions differ.'
        s.status=f.unreal_push_status
        return
    if receipt.get('status') != 'complete':
        if receipt.get('character')==f.unreal_character and receipt.get('assembly'):
            f.unreal_assembly_ready=bool(receipt['assembly'].get('ready'))
            f.unreal_status_report=json.dumps(receipt)
        label='Unreal action failed: ' if receipt.get('action') else 'Push failed: '
        f.unreal_push_status = label + receipt.get('error', 'Inspect the Unreal Output Log')
        s.status = f.unreal_push_status
        return
    character=receipt.get('character') or ''
    if character and character!=f.unreal_character and not receipt.get('action') and hasattr(scene,'cm_groom'):
        scene.cm_groom.unreal_target=''
        f.unreal_build_report=''
    f.unreal_character = character
    f.unreal_character_project=receipt.get('unreal_project_file','')
    state = receipt.get('assembly', {})
    f.unreal_assembly_ready = bool(state.get('ready') or state.get('can_request_build'))
    f.unreal_status_report = json.dumps(receipt)
    if f.unreal_character:
        if receipt.get('build',{}).get('state')=='COMPLETE':
            previous=f.unreal_build_report
            f.unreal_build_report=json.dumps(receipt)
            use_build_face(scene,previous=previous,only_automatic=True)
            f.unreal_push_status='Character assembled and saved: '+receipt['build']['blueprint']
        elif receipt.get('action'):
            assembly=receipt.get('assembly',{})
            if assembly.get('rig_request_pending'):
                f.unreal_push_status='Creator is rigging this character. Refresh after the request finishes.'
            elif assembly.get('texture_request_pending'):
                f.unreal_push_status='Creator is downloading texture sources. Refresh after the request finishes.'
            elif assembly.get('inspection_scope')=='SAVED_CHARACTER':
                f.unreal_push_status=assembly.get('reason') or 'Saved character inspected. Assembly will verify live readiness after loading.'
            else:
                f.unreal_push_status=('Creator reports that the character is ready to assemble.' if f.unreal_assembly_ready else
                    assembly.get('reason') or 'Creator requirements remain. Complete the rig and texture sources in Unreal, then refresh.')
            if receipt.get('editor_opened'):
                f.unreal_push_status='Character opened in Creator. '+f.unreal_push_status
            if receipt.get('editor_warning'):
                f.unreal_push_status+=' '+receipt['editor_warning']
        else:
            f.unreal_push_status = ('Character imported. Ready for assembly in Creator.' if f.unreal_assembly_ready else
                'Character imported. Complete Create Rig and Download Texture Sources in Creator before assembly.')
    else:
        f.unreal_push_status = 'Geometry and maps imported. Selected in Unreal.'
    if receipt.get('retain_source_snapshot'):
        f.unreal_push_status += ' Some Reimport paths need repair; their source files were retained.'
    s.status = f.unreal_push_status


def use_build_face(scene,previous='',only_automatic=False):
    """Bind to the real assembled face; retain artist overrides on receipt delivery."""
    f=scene.cm_features
    try:report=json.loads(f.unreal_build_report or '{}')
    except ValueError:raise ValueError('Refresh the completed character assembly first') from None
    build=report.get('build',{})
    if report.get('status')!='complete' or report.get('character')!=f.unreal_character or build.get('state')!='COMPLETE':
        raise ValueError('Assemble this character before choosing its groom binding target')
    target=build.get('component_meshes',{}).get('Face','')
    if not isinstance(target,str) or not target.startswith('/Game/'):
        if only_automatic:return False
        raise ValueError('This assembly has no recorded Face component; choose its final skeletal mesh in Hair')
    if not hasattr(scene,'cm_groom'):return False
    current=scene.cm_groom.unreal_target
    try:old=json.loads(previous or '{}')
    except ValueError:old={}
    previous_target=(old.get('build',{}).get('component_meshes',{}).get('Face','')
                     if old.get('character')==f.unreal_character else '')
    if only_automatic and current and current!=previous_target:return False
    scene.cm_groom.unreal_target=target
    return True


def poll():
    global _JOB
    job = _JOB
    if not job:
        return None
    scene = _scene(job)
    while True:
        try:
            kind, value = job['events'].get_nowait()
        except queue.Empty:
            break
        if kind == 'error':
            job['error'] = value
        elif kind == 'phase' and scene:
            scene.cm_features.unreal_push_status = value
    waiting_on_client = job['thread'].is_alive()
    if waiting_on_client and job.get('kind') in {'UPDATE_TEXTURES','UPDATE_EYES','REQUEST_TEXTURE_SOURCES','REQUEST_RIG','CAPTURE_ARTIST_EDITS','CAPTURE_NATIVE_WEB','HEAD_REVISION'} and job['receipt'].is_file():
        try:
            waiting_on_client = json.loads(job['receipt'].read_text(encoding='utf-8')).get('status') not in {'complete', 'failed'}
        except (OSError, ValueError):
            pass
    if waiting_on_client:
        if job['dispatched'].is_set() and scene:
            phase='Assembling in Unreal. Blender remains usable.'
            if job.get('kind')=='BUILD_CHARACTER':
                try:phase=json.loads((job['folder']/'progress.json').read_text())['phase']
                except (OSError,ValueError,KeyError):pass
            scene.cm_features.unreal_push_status = ('Updating textures in Unreal. Blender remains usable.' if job.get('kind')=='UPDATE_TEXTURES' else phase if job.get('kind')=='BUILD_CHARACTER' else 'Checking the character in Unreal. Blender remains usable.' if job.get('kind') else
                'Unreal is importing and opening the character. You can keep working in Blender.')
            if job.get('kind')=='UPDATE_EYES':scene.cm_features.unreal_push_status='Updating eye appearance in Creator. Blender remains usable.'
            if job.get('kind')=='REQUEST_RIG':
                phase='Observing the original native full rig request. Blender remains usable.'
                try:phase=json.loads((job['folder']/'progress.json').read_text())['phase']
                except (OSError,ValueError,KeyError):pass
                scene.cm_features.unreal_push_status=phase
            if job.get('kind')=='REQUEST_TEXTURE_SOURCES':
                phase='Requesting native texture sources. Blender remains usable.'
                try:phase=json.loads((job['folder']/'progress.json').read_text())['phase']
                except (OSError,ValueError,KeyError):pass
                scene.cm_features.unreal_push_status=phase
            if job.get('kind')=='HEAD_REVISION':
                phase='Creating a separate head revision. Blender remains usable.'
                try:phase+=' '+json.loads((job['folder']/'head-revision.json').read_text())['status']
                except (OSError,ValueError,KeyError):pass
                scene.cm_features.unreal_push_status=phase
            if job.get('kind')=='CAPTURE_ARTIST_EDITS':scene.cm_features.unreal_push_status='Preserving saved UE artist settings and native assets. Blender remains usable.'
            if job.get('kind')=='CAPTURE_NATIVE_WEB':
                phase='Capturing native web source and artist appearance. Blender remains usable.'
                try:
                    progress=json.loads((job['folder']/'progress.json').read_bytes());phase=capture_progress(progress)+'. Blender remains usable.'
                except (OSError,ValueError,KeyError):pass
                scene.cm_features.unreal_push_status=phase;scene.cm_features.native_web_status=phase
        return .2
    receipt = None
    receipt_path = job['receipt']
    if not receipt_path.is_file():
        # The importer commits this report before optional editor-window work.
        # Recover a successful import even if UE/client exits while opening UI.
        native_report = job['snapshot'] / 'unreal-import-report.json'
        if not job.get('kind') and job['dispatched'].is_set() and native_report.is_file():
            receipt_path = native_report
    if receipt_path.is_file():
        try:
            receipt = json.loads(receipt_path.read_text(encoding='utf-8'))
        except (OSError, ValueError):
            return .5
    if receipt is None and job.get('kind') == 'UPDATE_TEXTURES':
        progress = job['folder'] / 'texture-progress.json'
        if progress.is_file():
            try:
                native = json.loads(progress.read_text(encoding='utf-8'))
                if native.get('status') in {'complete', 'failed'}:
                    receipt = native
            except (OSError, ValueError):
                pass
    if receipt is None and job.get('kind')=='HEAD_REVISION':
        try:
            native=json.loads((job['folder']/'head-revision.json').read_text(encoding='utf-8'))
            if native.get('status') in {'complete','failed'}:receipt=native
        except (OSError,ValueError):pass
    if receipt is None and job['dispatched'].is_set():
        identity_path = job['folder'] / 'dispatch.json'
        try:
            identity = json.loads(identity_path.read_text(encoding='utf-8')) if identity_path.is_file() else None
        except (OSError, ValueError):
            identity = None
        if identity and not process_ref.alive(identity):
            operation='full rig request' if job.get('kind')=='REQUEST_RIG' else 'texture-source request' if job.get('kind')=='REQUEST_TEXTURE_SOURCES' else 'native web capture' if job.get('kind')=='CAPTURE_NATIVE_WEB' else 'eye update' if job.get('kind')=='UPDATE_EYES' else 'texture update' if job.get('kind')=='UPDATE_TEXTURES' else 'assembly' if job.get('kind')=='BUILD_CHARACTER' else 'push'
            receipt = {'status': 'failed', 'error': 'Unreal exited before confirming the '+operation+'. Inspect its partial assets before retrying.',
                       'editor_process': identity}
            if job.get('kind') == 'UPDATE_TEXTURES':
                try:
                    partial = json.loads((job['folder'] / 'texture-progress.json').read_text(encoding='utf-8'))
                    partial.update(receipt)
                    receipt = partial
                except (OSError, ValueError):
                    pass
        else:
            if scene:
                message='Connection ended; waiting for Unreal to confirm the original push. Do not push again.'
                if job.get('kind')=='UPDATE_TEXTURES':
                    message='Waiting for Unreal to confirm the original texture update. Do not start another update.'
                if job.get('kind')=='UPDATE_EYES':message='Waiting for Unreal to confirm the original eye appearance update.'
                if job.get('kind')=='REQUEST_TEXTURE_SOURCES':
                    message='Waiting for the original native texture-source request; no second request was sent.'
                    try:message=json.loads((job['folder']/'progress.json').read_text())['phase']
                    except (OSError,ValueError,KeyError):pass
                if job.get('kind')=='REQUEST_RIG':
                    message='Waiting for the original full rig request; no second request was sent.'
                    try:message=json.loads((job['folder']/'progress.json').read_text())['phase']
                    except (OSError,ValueError,KeyError):pass
                if job.get('kind')=='HEAD_REVISION':message='Waiting for the original head revision receipt. No second revision was requested.'
                if job.get('kind')=='CAPTURE_ARTIST_EDITS':message='Waiting for Unreal to confirm the original artist-state capture.'
                if job.get('kind')=='CAPTURE_NATIVE_WEB':message='Waiting for the original native web capture receipt. No second capture was requested.'
                if job.get('kind')=='CAPTURE_NATIVE_WEB':
                    try:
                        progress=json.loads((job['folder']/'progress.json').read_bytes());message=capture_progress(progress)+'. Blender remains usable.'
                    except (OSError,ValueError,KeyError):pass
                    scene.cm_features.native_web_status=message
                if job.get('kind')=='BUILD_CHARACTER':
                    message='Waiting for Unreal to confirm the original assembly. Do not assemble again.'
                    try:message=json.loads((job['folder']/'progress.json').read_text())['phase']+' '+message
                    except (OSError,ValueError,KeyError):pass
                scene.cm_features.unreal_push_status = message
            return .5
    if receipt is None:
        receipt = {'status': 'failed', 'error': job['error'] or 'Unreal connection could not be opened'}
    if job.get('kind'):
        receipt.setdefault('action',job['kind'])
    if job.get('kind') in {'UPDATE_TEXTURES','UPDATE_EYES','REQUEST_TEXTURE_SOURCES','REQUEST_RIG'}:
        receipt.setdefault('character', job.get('character', ''))
    if job.get('kind')=='REQUEST_RIG':
        from . import unreal_rig
        expected=unreal_rig.digest(job['folder']/'request.json')
        if receipt.get('status')=='complete' and (receipt.get('request_sha256')!=expected or receipt.get('character')!=job['character']):
            receipt=dict(status='failed',action='REQUEST_RIG',character=job['character'],error='The full rig receipt does not match its original request')
        receipt.setdefault('request_sha256',expected)
    if job['error']:
        receipt['connection_warning'] = job['error']
    if (not job.get('kind') and receipt.get('status') == 'complete'
            and not isinstance(receipt.get('reimport_sources'), dict)):
        receipt['retain_source_snapshot'] = True
        receipt['reimport_sources'] = {'repaired': [], 'warnings': [
            {'error': 'This older bundle importer did not repair Reimport paths; its source snapshot is retained.'}]}
    job['receipt'].write_text(json.dumps(receipt, indent=2), encoding='utf-8')
    (job['folder'] / 'result.json').write_text(json.dumps(receipt, indent=2), encoding='utf-8')
    if scene:
        apply_receipt(scene, receipt)
    if receipt.get('status') == 'complete':
        if job.get('kind')=='BUILD_CHARACTER' and receipt.get('build',{}).get('state')=='COMPLETE':
            history=Path(job['project_dir'])/'Export'/re.sub(r'[^A-Za-z0-9_]','_',job['character_name']).strip('_')/'last-build.json'
            record={'schema_version':1,'project_dir':job['project_dir'],'character_name':job['character_name'],'receipt':receipt}
            try:
                history.parent.mkdir(parents=True,exist_ok=True)
                temporary=history.with_suffix('.tmp');temporary.write_text(json.dumps(record,indent=2),encoding='utf-8');temporary.replace(history)
            except OSError as exc:
                if scene:scene.cm_settings.status+=' Save this project to retain build history: '+str(exc)
        # Retain the standard report for standalone export users, plus the small
        # durable receipt. The duplicate snapshot can be removed after native
        # Reimport paths are repaired; retain it when that optional step failed.
        if not job.get('kind'):
            (job['bundle'] / 'unreal-import-report.json').write_text(json.dumps(receipt, indent=2), encoding='utf-8')
            if receipt.get('character'):
                history=job['bundle']/'last-character.json'
                temporary=history.with_suffix('.tmp')
                try:
                    temporary.write_text(json.dumps({'schema_version':1,'project_dir':job['project_dir'],
                        'character_name':job['character_name'],'receipt':receipt},indent=2),encoding='utf-8')
                    temporary.replace(history)
                except OSError as exc:
                    if scene:
                        scene.cm_features.unreal_push_status+=' Delivery succeeded; save this Blender project because its delivery history could not be written: '+str(exc)
        if (not receipt.get('retain_source_snapshot')
                and job['snapshot'].resolve().is_relative_to(job['folder'].resolve())):
            shutil.rmtree(job['snapshot'])
    _JOB = None
    for window in bpy.context.window_manager.windows:
        for area in window.screen.areas:
            area.tag_redraw()
    return None


@persistent
def detach(*_):
    # The UE operation is not cancellable by terminating its network client.
    # Continue tracking its receipt without applying it to a newly loaded scene.
    if _JOB:
        _JOB['scene'] = None


def restore_delivery(scene):
    """Restore project delivery metadata without changing character meshes."""
    s=scene.cm_settings
    name=re.sub(r'[^A-Za-z0-9_]','_',s.character_name).strip('_')
    if not name:
        return False
    root=Path(bpy.path.abspath(s.project_dir)).resolve()
    history=root/'Export'/name/'last-character.json'
    if not history.is_file():
        return False
    revision_history=history.with_name('last-head-revision.json')
    if revision_history.is_file():
        revision_record=json.loads(revision_history.read_text(encoding='utf-8'));revision_path=Path(revision_record['receipt'])
        if revision_path.is_file() and revision_path.stat().st_mtime_ns>history.stat().st_mtime_ns:
            revision=json.loads(revision_path.read_text(encoding='utf-8'))
            if revision.get('status')=='complete':return False
    record=json.loads(history.read_text(encoding='utf-8'))
    if not isinstance(record,dict):
        return False
    if (record.get('schema_version')!=1 or record.get('character_name')!=s.character_name
            or Path(record.get('project_dir','')).resolve()!=root):
        return False
    receipt=record.get('receipt',{})
    if not isinstance(receipt,dict) or receipt.get('status')!='complete' or not receipt.get('character'):
        return False
    f=scene.cm_features
    if f.unreal_character==receipt['character'] and f.unreal_status_report:
        # A saved status refresh may be newer than the original import receipt.
        return False
    apply_receipt(scene,receipt)
    scene.cm_features.unreal_push_status='Last successful delivery restored. '+scene.cm_features.unreal_push_status
    return True


@persistent
def loaded(*_):
    for scene in bpy.data.scenes:
        if hasattr(scene,'cm_settings') and hasattr(scene,'cm_features'):
            scene.cm_features.unreal_resource_report=''
            try:
                restore_delivery(scene)
            except (OSError,ValueError,TypeError,KeyError):
                pass
            try:
                from . import head_revision
                head_revision.restore(scene)
            except (OSError,ValueError,TypeError,KeyError):pass
            try:
                restore_build(scene)
            except (OSError,ValueError,TypeError):pass
            try:
                from . import rig_jobs
                rig_jobs.restore(scene)
            except (OSError,ValueError,TypeError,KeyError):pass
            try:
                restore_texture_update(scene)
                restore_texture_update(scene,kind='UPDATE_EYES')
                restore_texture_update(scene,kind='REQUEST_TEXTURE_SOURCES')
                restore_texture_update(scene,kind='CAPTURE_NATIVE_WEB')
            except (OSError, ValueError, TypeError):
                pass


def restore_texture_update(scene,kind='UPDATE_TEXTURES'):
    """Observe the existing native receipt after reopening; never redispatch."""
    global _JOB
    s, f = scene.cm_settings, scene.cm_features
    eye=kind=='UPDATE_EYES';web=kind=='CAPTURE_NATIVE_WEB';sources=kind=='REQUEST_TEXTURE_SOURCES'
    prefix='unreal_sources' if sources else 'native_web_capture' if web else 'unreal_eye' if eye else 'unreal_texture'
    root = Path(bpy.path.abspath(s.project_dir)).resolve()
    name = re.sub(r'[^A-Za-z0-9_]', '_', s.character_name).strip('_')
    history = root / 'Export' / name / ('last-texture-sources-action.json' if sources else 'last-web-capture-action.json' if web else 'last-eye-action.json' if eye else 'last-texture-action.json')
    if not name or not history.is_file():
        return False
    record = json.loads(history.read_text(encoding='utf-8'))
    if (record.get('schema_version') != 1 or record.get('character_name') != s.character_name
            or Path(record.get('project_dir', '')).resolve() != root or record.get('character') != f.unreal_character
            or not f.unreal_character_project and not web
            or Path(record.get('unreal_project_file', '')).resolve() != Path(f.unreal_character_project).resolve()):
        return False
    receipt_path = Path(record.get('receipt', '')).resolve()
    if not receipt_path.is_relative_to(root / 'Export' / '.unreal-actions') or receipt_path.name != 'receipt.json':
        return False
    if receipt_path.is_file():
        if getattr(f,prefix+'_receipt') != str(receipt_path) or not getattr(f,prefix+'_result'):
            receipt = json.loads(receipt_path.read_text(encoding='utf-8'))
            if receipt.get('action') == kind and receipt.get('character') == f.unreal_character:
                setattr(f,prefix+'_receipt',str(receipt_path))
                restore_action_receipt(scene, receipt)
        return False
    if _JOB is not None:
        if Path(_JOB['receipt']).resolve() == receipt_path:
            _JOB['scene'] = scene
            return True
        return False
    dispatched = threading.Event()
    dispatched.set()
    _JOB = {'scene': scene, 'folder': receipt_path.parent, 'snapshot': receipt_path.parent / 'scripts',
            'receipt': receipt_path, 'events': queue.Queue(), 'dispatched': dispatched, 'error': None,
            'kind': kind, 'project_dir': str(root), 'character_name': s.character_name, 'character': f.unreal_character,
            'thread': threading.Thread()}
    setattr(f,prefix+'_receipt',str(receipt_path))
    f.unreal_push_status = 'Resuming the original texture update receipt. No new import was requested.'
    if eye:f.unreal_push_status='Resuming the original eye appearance receipt.'
    if sources:f.unreal_push_status='Resuming the original native texture-source request. No second request was sent.'
    if web:f.unreal_push_status='Resuming the original native web capture receipt. No new capture was requested.'
    if not bpy.app.timers.is_registered(poll):
        bpy.app.timers.register(poll, first_interval=.2, persistent=True)
    return True


def restore_build(scene):
    global _JOB
    s,f=scene.cm_settings,scene.cm_features
    root=Path(bpy.path.abspath(s.project_dir)).resolve()
    name=re.sub(r'[^A-Za-z0-9_]','_',s.character_name).strip('_')
    if not name:return False
    action=root/'Export'/name/'last-build-action.json'
    if action.is_file():
        record=json.loads(action.read_text(encoding='utf-8'))
        if isinstance(record,dict) and (record.get('schema_version')==1 and record.get('character_name')==s.character_name
            and Path(record.get('project_dir','')).resolve()==root and record.get('character')==f.unreal_character
            and (not f.unreal_character_project or Path(record.get('unreal_project_file','')).resolve()==Path(f.unreal_character_project).resolve())):
            receipt_path=Path(record.get('receipt','')).resolve()
            if not receipt_path.is_relative_to(root/'Export'/'.unreal-actions') or receipt_path.name!='receipt.json':return False
            f.unreal_build_receipt=str(receipt_path)
            if receipt_path.is_file():
                receipt=json.loads(receipt_path.read_text(encoding='utf-8'))
                if isinstance(receipt,dict) and receipt.get('character')==f.unreal_character and receipt.get('action')=='BUILD_CHARACTER':
                    if receipt.get('status')=='complete' and isinstance(receipt.get('build'),dict) and receipt['build'].get('state')=='COMPLETE':
                        restore_completed_build(scene,receipt)
                        return True
                    if receipt.get('status')=='failed':
                        try:current=json.loads(f.unreal_status_report) if f.unreal_status_report else {}
                        except ValueError:current={}
                        if not isinstance(current,dict):current={}
                        if receipt_time(receipt)>receipt_time(current):
                            apply_receipt(scene,receipt)
                        return False
            elif _JOB is not None and Path(_JOB['receipt']).resolve()==receipt_path:
                _JOB['scene']=scene
                f.unreal_push_status='Resumed monitoring the original assembly. No new build was requested.'
                return True
            elif _JOB is None:
                # Resume only the existing durable action. Never redispatch a
                # native build because the client or its observation ended.
                folder=receipt_path.parent;dispatched=threading.Event();dispatched.set()
                _JOB={'scene':scene,'folder':folder,'snapshot':folder/'scripts','receipt':receipt_path,
                      'events':queue.Queue(),'dispatched':dispatched,'error':None,'kind':'BUILD_CHARACTER',
                      'project_dir':str(root),'character_name':s.character_name,'thread':threading.Thread()}
                f.unreal_push_status='Resuming the previous assembly receipt. No new build was requested.'
                if not bpy.app.timers.is_registered(poll):bpy.app.timers.register(poll,first_interval=.2,persistent=True)
                return True
    if f.unreal_build_report:return False
    path=root/'Export'/name/'last-build.json'
    if not path.is_file():return False
    record=json.loads(path.read_text(encoding='utf-8'))
    if not isinstance(record,dict):return False
    if record.get('schema_version')!=1 or record.get('character_name')!=s.character_name or Path(record.get('project_dir','')).resolve()!=root:return False
    receipt=record.get('receipt',{})
    if not isinstance(receipt,dict) or not isinstance(receipt.get('build'),dict):return False
    if receipt.get('character')!=f.unreal_character or receipt.get('status')!='complete' or receipt['build'].get('state')!='COMPLETE':return False
    if f.unreal_character_project and Path(receipt.get('unreal_project_file','')).resolve()!=Path(f.unreal_character_project).resolve():return False
    restore_completed_build(scene,receipt)
    return True


def restore_completed_build(scene,receipt):
    """Recover delivery controls as on completion, retaining newer status/artwork."""
    f=scene.cm_features
    try:current=json.loads(f.unreal_status_report or '{}')
    except ValueError:current={}
    if not isinstance(current,dict):current={}
    if receipt_time(receipt)>=receipt_time(current):
        # In particular, an older rig receipt restored later during file load
        # must not replace a completed assembly with "Ready to assemble".
        apply_receipt(scene,receipt)
    else:
        previous=f.unreal_build_report
        f.unreal_build_report=json.dumps(receipt)
        use_build_face(scene,previous=previous,only_automatic=True)


def register():
    if detach not in bpy.app.handlers.load_pre:
        bpy.app.handlers.load_pre.append(detach)
    if loaded not in bpy.app.handlers.load_post:
        bpy.app.handlers.load_post.append(loaded)


def unregister():
    if detach in bpy.app.handlers.load_pre:
        bpy.app.handlers.load_pre.remove(detach)
    if loaded in bpy.app.handlers.load_post:
        bpy.app.handlers.load_post.remove(loaded)
    detach()
