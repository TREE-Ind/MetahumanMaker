"""Observe one native texture-source request through its actual completion."""
import json
import os
import runpy
import re
import shutil
import time
import uuid
from datetime import datetime,timezone
from pathlib import Path


def stored_sources(character):
    """Compare persistent source metadata, independently of preview GPU mips."""
    import unreal
    desired=character.get_editor_property('skin_settings').get_editor_property('desired_texture_sources_resolutions')
    face=character.get_editor_property('synthesized_face_textures_info')
    body=character.get_editor_property('high_res_body_textures_info')
    scopes=(('face_albedo',face,unreal.FaceTextureType.BASECOLOR),
            ('face_normal',face,unreal.FaceTextureType.NORMAL),
            ('face_cavity',face,unreal.FaceTextureType.CAVITY),
            *((f'face_animated_maps',face,getattr(unreal.FaceTextureType,'BASECOLOR_ANIMATED_CM'+str(i))) for i in (1,2,3)),
            *((f'face_animated_maps',face,getattr(unreal.FaceTextureType,'NORMAL_ANIMATED_WM'+str(i))) for i in (1,2,3)),
            ('body_albedo',body,unreal.BodyTextureType.BODY_BASECOLOR),
            ('body_normal',body,unreal.BodyTextureType.BODY_NORMAL),
            ('body_cavity',body,unreal.BodyTextureType.BODY_CAVITY),
            ('body_albedo',body,unreal.BodyTextureType.BODY_UNDERWEAR_BASECOLOR),
            ('body_normal',body,unreal.BodyTextureType.BODY_UNDERWEAR_NORMAL),
            ('body_albedo',body,unreal.BodyTextureType.CHEST_BASECOLOR),
            ('body_normal',body,unreal.BodyTextureType.CHEST_NORMAL),
            ('body_cavity',body,unreal.BodyTextureType.CHEST_CAVITY),
            ('body_albedo',body,unreal.BodyTextureType.CHEST_UNDERWEAR_BASECOLOR),
            ('body_normal',body,unreal.BodyTextureType.CHEST_UNDERWEAR_NORMAL),
            ('body_masks',body,unreal.BodyTextureType.BODY_UNDERWEAR_MASK))
    checks=[]
    for field,values,kind in scopes:
        value=desired.get_editor_property(field);resolution=int(value.value)
        info=values.get(kind)
        dimensions=[int(info.get_editor_property(key)) for key in ('size_x','size_y')] if info is not None else [0,0]
        checks.append(dict(setting=field,texture=str(kind),desired=resolution,stored=dimensions,matches=dimensions==[resolution,resolution]))
    return dict(matches=all(row['matches'] for row in checks),textures=checks)


def prepare(project_dir,character_name,asset_path,expected_project,project_name=''):
    """Stage one immutable action for Blender MCP or an owned Unreal worker."""
    root=Path(project_dir).resolve();project=Path(expected_project).resolve()
    if not root.is_dir():raise ValueError('Save the character project before downloading native texture sources')
    name=re.sub(r'[^A-Za-z0-9_]','_',character_name).strip('_')
    if not name or not asset_path.startswith('/Game/') or not project.is_file() or project.suffix.lower()!='.uproject':
        raise ValueError('A character name, linked native asset and existing Unreal project are required')
    budget=runpy.run_path(str(Path(__file__).with_name('resource_budget.py')))
    budget['require_disk']('native texture-source action',root,2*budget['GIB'])
    history=root/'Export'/name/'last-texture-sources-action.json'
    if history.is_file():
        previous=json.loads(history.read_text(encoding='utf-8'))
        original=Path(previous.get('receipt','')).resolve()
        if (previous.get('character')==asset_path and previous.get('unreal_project_file')==str(project)
                and original.is_relative_to(root/'Export'/'.unreal-actions') and original.name=='receipt.json'):
            if not original.is_file() or json.loads(original.read_text(encoding='utf-8')).get('status') not in {'complete','failed'}:
                raise ValueError('Observe the original native texture-source action before staging another request: '+str(original))
    folder=root/'Export'/'.unreal-actions'/uuid.uuid4().hex[:12];snapshot=folder/'scripts'
    snapshot.mkdir(parents=True,exist_ok=False)
    for filename in ('unreal_texture_sources.py','unreal_revision_textures.py','process_ref.py','resource_budget.py'):
        shutil.copy2(Path(__file__).with_name(filename),snapshot/filename)
    request=dict(schema_version=1,action='REQUEST_TEXTURE_SOURCES',character=asset_path,project=project_name,expected_project=str(project))
    (folder/'request.json').write_text(json.dumps(request,indent=2),encoding='utf-8')
    history.parent.mkdir(parents=True,exist_ok=True)
    record=dict(schema_version=1,project_dir=str(root),character_name=character_name,character=asset_path,
                unreal_project_file=str(project),receipt=str(folder/'receipt.json'))
    temporary=history.with_suffix('.tmp');temporary.write_text(json.dumps(record,indent=2),encoding='utf-8');temporary.replace(history)
    return dict(folder=str(folder),snapshot=str(snapshot),receipt=str(folder/'receipt.json'),request=str(folder/'request.json'))


def dispatch(asset_path,folder,expected_project='',*,keep_alive=False):
    import unreal
    folder=Path(folder).resolve();base=folder.parent;receipt=base/'receipt.json'
    if receipt.exists() or (base/'dispatch.json').exists():raise ValueError('Texture-source request already dispatched; observe its original receipt')
    identity=runpy.run_path(str(folder/'process_ref.py'))['identify'](os.getpid())
    (base/'dispatch.json').write_text(json.dumps(identity))
    report=dict(status='initializing',action='REQUEST_TEXTURE_SOURCES',character=asset_path,started=time.time(),
                request_dispatched=False,attached_existing_request=False,requested_rig=False)
    state=dict(handle=None,character=None,subsystem=None,actor=None,owned=False,before=None,last=0.,done=False,busy=False)
    def write(name,value):
        target=base/name;temporary=target.with_suffix('.tmp');temporary.write_text(json.dumps(value,indent=2));temporary.replace(target)
    def inspect():
        native=json.loads(unreal.CreateMHCharacterLibrary.inspect_character(state['character']))
        if native and 'rig_state' in native and 'ready' in native:native['detailed_status_available']=True
        return native
    def progress(phase):
        report['elapsed_seconds']=time.time()-report['started']
        write('progress.json',dict(phase=phase,elapsed_seconds=report['elapsed_seconds'],native_state=report.get('assembly',{}),
                                   request_dispatched=report['request_dispatched'],attached_existing_request=report['attached_existing_request']))
    def finish(error=''):
        if state['done']:return
        state['done']=True
        if state['handle'] is not None:unreal.unregister_slate_post_tick_callback(state['handle']);state['handle']=None
        try:
            if error:report.update(status='failed',error=error)
            else:
                character=state['character'];subsystem=state['subsystem'];native=inspect();report['assembly']=native
                report['stored_sources']=stored_sources(character)
                if native.get('texture_request_pending') or not native.get('has_texture_sources') or not report['stored_sources']['matches']:
                    raise RuntimeError('Native texture request ended without the current required sources. Inspect the MetaHuman Message Log for service/account details.')
                if native.get('needs_texture_download'):
                    report['preview_resolution_warning']='Stored source dimensions match every requested map; native preview texture dimensions still differ. No resolutions were changed.'
                current=json.loads(unreal.CreateMHArtistLibrary.snapshot_object(character))
                write('artist-after.json',current)
                before=state['before'];props=lambda data:next(row['properties'] for row in data['objects'] if row['object']==character.get_path_name())
                a,b=props(before),props(current)
                fields=[key for key in ('SkinSettings','EyesSettings','HeadModelSettings','BodyModelSettings','FaceEvaluationSettings','AssemblySettings','PreviewMaterialType') if key in a]
                report['artist_setting_changes']=[key for key in fields if a[key]['text']!=b.get(key,{}).get('text')]
                # Never restore an old snapshot over edits made during a cloud
                # request. Preserve and report the current native artist state.
                report['artist_settings_preserved']=not report['artist_setting_changes']
                helpers=runpy.run_path(str(folder/'unreal_revision_textures.py'))
                actor=state['actor'];components={role:actor.get_editor_property(name) for role,name in (('head','face_component'),('body','body_component'))}
                report['geometry_after']={role:helpers['mesh_digest'](subsystem,value) for role,value in components.items()}
                report['geometry_preserved']=report['geometry_after']==state['meshes']
                report['rig_preserved']=all(native.get(key)==state['native'].get(key) for key in ('rig_state','has_face_dna','has_body_dna','rig_request_pending'))
                report['source_eye_metadata_preserved']=unreal.EditorAssetLibrary.get_metadata_tag(character,'CreateMetaHuman.SourceEyeAppearance')==state['source_eye_tag']
                if not unreal.EditorAssetLibrary.save_loaded_asset(character,False):raise RuntimeError('Texture sources downloaded in memory; save the character in Unreal before closing')
                report.update(status='complete',saved=True,character=character.get_path_name(),
                              unreal_project_file=str(Path(unreal.Paths.get_project_file_path()).resolve()))
                if report['artist_setting_changes'] or not report['geometry_preserved'] or not report['rig_preserved'] or not report['source_eye_metadata_preserved']:
                    report['review_required']='Native inputs changed while downloading; inspect current artist edits and refresh delivery requirements before assembly.'
        except Exception as exc:report.update(status='failed',error=str(exc))
        finally:
            try:
                if state['actor']:unreal.get_editor_subsystem(unreal.EditorActorSubsystem).destroy_actor(state['actor'])
            except Exception as exc:report.setdefault('cleanup_warnings',[]).append(str(exc))
            try:
                if state['owned'] and state['character'] and state['subsystem']:state['subsystem'].remove_object_to_edit(state['character'])
            except Exception as exc:report.setdefault('cleanup_warnings',[]).append(str(exc))
            report['completed_at']=datetime.now(timezone.utc).isoformat();write('receipt.json',report)
            if keep_alive:unreal.EditorPythonScripting.set_keep_python_script_alive(False)
    def tick(delta):
        # Native initialization/spawn/extraction can pump Slate recursively.
        # Keep a nested observer from finishing an incomplete session.
        if state['done'] or state['busy'] or time.monotonic()-state['last']<1:return
        state['busy']=True
        state['last']=time.monotonic()
        try:
            if state['character'] is None:
                project=Path(unreal.Paths.get_project_file_path()).resolve()
                if not expected_project or project!=Path(expected_project).resolve():raise ValueError('Open the linked Unreal project before requesting texture sources')
                report['unreal_project_file']=str(project)
                budget=runpy.run_path(str(folder/'resource_budget.py'))
                budget['require_disk']('native texture-source download',unreal.Paths.project_content_dir(),16*budget['GIB'])
                character=unreal.load_asset(asset_path)
                if not isinstance(character,unreal.MetaHumanCharacter):raise ValueError('The linked native MetaHuman Character is missing')
                subsystem=unreal.get_editor_subsystem(unreal.MetaHumanCharacterEditorSubsystem)
                owned=not subsystem.is_object_added_for_editing(character)
                budget['require']('native texture-source initialization',(8 if owned else 2)*budget['GIB'])
                state.update(character=character,subsystem=subsystem)
                before=json.loads(unreal.CreateMHArtistLibrary.snapshot_object(character))
                if any(row['package_dirty'] for row in before['objects']):raise ValueError('Save native artist edits before requesting texture sources')
                write('artist-before.json',before)
                progress('Initializing the linked native character')
                if owned and not subsystem.try_add_object_to_edit(character):raise RuntimeError('Creator could not initialize the linked character')
                state['owned']=owned
                state['before']=before;state['native']=inspect();report['assembly']=state['native']
                if 'texture_request_pending' not in state['native'] or 'has_texture_sources' not in state['native']:
                    raise RuntimeError('Native texture-source status is incomplete; no download was requested. Refresh status before continuing.')
                state['source_eye_tag']=unreal.EditorAssetLibrary.get_metadata_tag(character,'CreateMetaHuman.SourceEyeAppearance')
                progress('Checking the existing rig and preview geometry')
                actor=subsystem.spawn_meta_human_actor(character,True)
                if not actor:raise RuntimeError('Creator could not inspect preview geometry before downloading textures')
                state['actor']=actor;helpers=runpy.run_path(str(folder/'unreal_revision_textures.py'))
                state['meshes']={role:helpers['mesh_digest'](subsystem,actor.get_editor_property(name)) for role,name in (('head','face_component'),('body','body_component'))}
                report['geometry_before']=state['meshes']
                resolutions=character.get_editor_property('skin_settings').get_editor_property('desired_texture_sources_resolutions')
                report['requested_resolutions']=resolutions.export_text()
                if state['native'].get('texture_request_pending'):
                    report.update(status='waiting',attached_existing_request=True);progress('Observing the existing native texture-source request')
                elif state['native'].get('has_texture_sources') and stored_sources(character)['matches']:
                    report['already_satisfied']=True;finish();return
                else:
                    if state['native'].get('rig_request_pending'):raise ValueError('Finish the current rig request before requesting texture sources')
                    params=unreal.MetaHumanCharacterTextureRequestParams();params.report_progress=False;params.blocking=False
                    report.update(status='requesting',request_dispatched=True)
                    progress('Requesting configured native texture sources')
                    subsystem.request_texture_sources(character,params)
                return
            native=inspect();report['assembly']=native
            if not native or 'texture_request_pending' not in native or 'has_texture_sources' not in native:
                progress('Native texture-source state is temporarily unavailable; retaining the original observer')
                return
            if native.get('texture_request_pending'):
                report['status']='waiting';progress('Downloading native texture sources; original request remains active')
                return
            finish()
        except Exception as exc:
            # An exception after dispatch is not evidence that the service
            # stopped. Continue observing a confirmed active native request.
            try:
                observed=inspect() if state['character'] is not None else {}
                pending=observed.get('texture_request_pending')
                if (state['character'] is not None and (not observed or 'texture_request_pending' not in observed)
                        and (report['request_dispatched'] or report['attached_existing_request'])):
                    report['observation_warning']=str(exc);progress('Native status unavailable; retaining the original request observer');return
            except Exception:
                if report['request_dispatched'] or report['attached_existing_request']:
                    report['observation_warning']=str(exc);progress('Native status unavailable; retaining the original request observer');return
                pending=False
            if pending:report['observation_warning']=str(exc);progress('Original texture-source request remains active')
            else:finish(str(exc))
        finally:state['busy']=False
    if keep_alive:unreal.EditorPythonScripting.set_keep_python_script_alive(True)
    state['handle']=unreal.register_slate_post_tick_callback(tick)
    progress('Scheduled native texture-source request')
    return dict(status='scheduled',receipt=str(receipt),editor_process=identity)
