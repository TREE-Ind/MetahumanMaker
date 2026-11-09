"""Deliver an explicitly reviewed local head repair with a native-file rollback.

No source fitting happens here: all coordinates come from the Blender add-on.
This preserves the existing asset identity and artist skin/eye settings.
"""
import json,hashlib,shutil,runpy
from pathlib import Path


def apply(request_path):
    import unreal
    request_path=Path(request_path).resolve();folder=request_path.parent
    request=json.loads(request_path.read_text());receipt=folder/'receipt.json'
    if request.get('schema')!=1 or receipt.exists():raise ValueError('Inspect the existing repair receipt before repeating delivery')
    if Path(unreal.Paths.get_project_file_path()).resolve()!=Path(request['expected_project']).resolve():raise ValueError('Open the linked Unreal project')
    c=unreal.load_asset(request['character']);assets=unreal.EditorAssetLibrary
    if not isinstance(c,unreal.MetaHumanCharacter):raise ValueError('The linked character is missing')
    editor=unreal.get_editor_subsystem(unreal.AssetEditorSubsystem);editor.close_all_editors_for_asset(c)
    sub=unreal.get_editor_subsystem(unreal.MetaHumanCharacterEditorSubsystem)
    assets.save_loaded_asset(c,False)
    native=Path(unreal.Paths.project_content_dir())/(c.get_path_name().split('.')[0].removeprefix('/Game/')+'.uasset')
    backup=folder/'before-character.uasset'
    if backup.exists():raise ValueError('A repair backup already exists; inspect the previous delivery')
    shutil.copy2(native,backup)
    artist=json.loads(unreal.CreateMHArtistLibrary.snapshot_object(c));(folder/'artist-before.json').write_text(json.dumps(artist,indent=2))
    skin=c.get_editor_property('skin_settings').export_text();eyes=c.get_editor_property('eyes_settings').export_text()
    if not sub.is_object_added_for_editing(c):
        if not sub.try_add_object_to_edit(c):raise RuntimeError('Creator could not open the linked character')
    state=json.loads(unreal.CreateMHCharacterLibrary.inspect_character(c))
    if state['rig_request_pending'] or state['texture_request_pending']:raise ValueError('Finish the pending native operation first')
    if state['rig_state']=='RIGGED':sub.remove_face_rig(c)
    p=unreal.MetaHumanCharacterFitToVerticesParams();options=p.get_editor_property('options')
    options.set_editor_property('alignment_options',unreal.MetaHumanAlignmentOptions.NONE)
    options.set_editor_property('disable_high_frequency_delta',False);options.set_editor_property('adapt_neck',False);p.set_editor_property('options',options)
    for role,field in (('head','head_vertices'),('eye_left','left_eye_vertices'),('eye_right','right_eye_vertices'),('teeth','teeth_vertices')):
        setattr(p,field,[unreal.Vector(*v) for v in request['geometry']['vertices'][role]])
    if not sub.fit_state_to_target_vertices(c,p):raise RuntimeError('Native head delivery failed; the original file backup was retained')
    sub.commit_face_state(c)
    audit=runpy.run_path(str(Path(__file__).with_name('unreal_import.py')))['audit_head_conform'](c,request['geometry']['vertices'],folder/'conform-state.json')
    if not audit['within_native_precision']:raise RuntimeError('Native positions differ from the reviewed Blender repair')
    if c.get_editor_property('skin_settings').export_text()!=skin or c.get_editor_property('eyes_settings').export_text()!=eyes:raise RuntimeError('Native artist settings changed; restore the retained backup')
    assets.save_loaded_asset(c,False)
    result=dict(status='complete',action='HEAD_REVISION',revision_mode='IN_PLACE_WITH_FILE_BACKUP',character=c.get_path_name(),unreal_project_file=request['expected_project'],local_geometry_witness=request['local_geometry_witness'],request_sha256=hashlib.sha256(request_path.read_bytes()).hexdigest(),head_geometry=audit,backup=str(backup),artist_settings_preserved=True)
    receipt.write_text(json.dumps(result,indent=2))
    # Release manual editing ownership before the durable rig action acquires it.
    sub.remove_object_to_edit(c)
    rig=runpy.run_path(str(Path(__file__).with_name('unreal_rig.py')))
    action=rig['prepare'](request_path.parents[3],c.get_name(),c.get_path_name(),request['expected_project'],geometry=request['geometry'])
    (folder/'rig-action.json').write_text(json.dumps(action,indent=2));rig['dispatch'](action['request'])
    return result
