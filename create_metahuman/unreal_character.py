"""Runs in Unreal: inspect or reopen an existing character without rebuilding it."""
from datetime import datetime, timezone
from pathlib import Path
import json
import runpy


def assembly_resources(editing):
    """Optional diagnostics must not turn a valid character status into failure."""
    import unreal
    try:
        helper=runpy.run_path(str(Path(__file__).with_name('unreal_assembly.py')))
        return helper['resource_status'](Path(unreal.Paths.project_content_dir()).resolve(),editing)
    except Exception as exc:return {'unavailable':str(exc)}


def inspect_character(asset_path, open_editor=False, expected_project=''):
    import unreal
    project=str(Path(unreal.Paths.get_project_file_path()).resolve())
    if expected_project and Path(expected_project).resolve()!=Path(project).resolve():
        raise ValueError('This character was delivered to another Unreal project. Open that project to continue.')
    character = unreal.load_asset(asset_path)
    if not isinstance(character, unreal.MetaHumanCharacter):
        raise ValueError('The last character is missing or is not a MetaHuman Character asset')
    subsystem = unreal.get_editor_subsystem(unreal.MetaHumanCharacterEditorSubsystem)
    already_editing = subsystem.is_object_added_for_editing(character)
    library = getattr(unreal,'CreateMHCharacterLibrary',None)
    if not open_editor and not already_editing and library:
        saved = json.loads(library.inspect_character(character))
        if saved.get('inspection_scope') == 'SAVED_CHARACTER':
            return {'status': 'complete', 'character': character.get_path_name(),
                    'unreal_project_file': project,
                    'inspected_at': datetime.now(timezone.utc).isoformat(),
                    'native_initializations': 0,
                    'assembly': dict(saved, detailed_status_available=True,resources=assembly_resources(False))}
    if open_editor and not already_editing and character.get_editor_property('has_face_dna_blendshapes'):
        # Creator's rigged-asset loader requires 10 GiB available virtual
        # memory. Remote execution cancels its warning dialog automatically;
        # OpenEditorForAssets can nevertheless return True. Check before
        # initializing the expensive editable character, not afterwards.
        budget = runpy.run_path(str(Path(__file__).with_name('resource_budget.py')))
        memory = budget['memory']()
        if memory and memory['available_commit_bytes'] < 10 * budget['GIB']:
            available = memory['available_commit_bytes'] / budget['GIB']
            raise ValueError(
                f'MetaHuman Creator needs at least 10 GiB free memory to open this rigged character; '
                f'{available:.1f} GiB is available. Close heavy applications and retry Open in Creator. '
                'The saved character was not changed.')
    editor_opened = False
    editor_warning = ''
    if open_editor:
        # AssetDefinition_MetaHumanCharacter owns initialization for an editor.
        # Inspecting first would initialize, unload, then initialize again.
        styles = unreal.get_default_object(unreal.load_class(None, '/Script/UnrealEd.EditorStyleSettings'))
        previous = styles.get_editor_property('AssetEditorOpenLocation')
        try:
            styles.set_editor_property('AssetEditorOpenLocation', unreal.AssetEditorOpenLocation.MAIN_WINDOW)
            accepted = unreal.get_editor_subsystem(unreal.AssetEditorSubsystem).open_editor_for_assets([character])
            editor_opened = bool(accepted and subsystem.is_object_added_for_editing(character))
            if not editor_opened:
                editor_warning = 'Creator did not open. Inspect the Unreal Output Log.'
        except Exception as exc:
            editor_warning = str(exc)
        finally:
            styles.set_editor_property('AssetEditorOpenLocation', previous)
        if not editor_opened:
            # A refused editor open must not start another costly status load.
            return {'status': 'complete', 'character': character.get_path_name(),
                    'unreal_project_file': project,
                    'inspected_at': datetime.now(timezone.utc).isoformat(),
                    'editor_opened': False, 'editor_warning': editor_warning,
                    'assembly': {'ready': False, 'detailed_status_available': False,
                                 'reason': 'Creator did not open; readiness was not inspected.'}}
    if not open_editor and not already_editing:
        # Older bridges cannot inspect a closed asset. Keep their temporary
        # status session, but refuse before allocating on a crowded machine.
        budget = runpy.run_path(str(Path(__file__).with_name('resource_budget.py')))
        memory = budget['memory']()
        required = (8 if character.get_editor_property('has_face_dna_blendshapes') else 2) * budget['GIB']
        if memory and memory['available_commit_bytes'] < required:
            raise ValueError('Not enough commit headroom to load character status. Install CreateMetaHumanBridge 0.21 or later for lightweight saved-character inspection.')
        if not subsystem.try_add_object_to_edit(character):
            raise RuntimeError('Creator could not initialize the character. Check MetaHuman Core Data in Unreal.')
    try:
        assembly = {'ready': bool(subsystem.can_build_meta_human(character, False)),
                    'has_texture_sources': bool(character.has_high_resolution_textures),
                    'detailed_status_available': False}
        if library:
            native = json.loads(library.inspect_character(character))
            if 'rig_state' in native and 'ready' in native:
                assembly.update(native,detailed_status_available=True)
        assembly['resources']=assembly_resources(subsystem.is_object_added_for_editing(character))
        report = {'status': 'complete', 'character': character.get_path_name(),
                  'unreal_project_file': project,
                  'inspected_at': datetime.now(timezone.utc).isoformat(),
                  'native_initializations': int(not already_editing),
                  'assembly': assembly}
    finally:
        # A status check must not unload a character already open in Creator.
        if not open_editor and not already_editing:
            subsystem.remove_object_to_edit(character)
    if open_editor:
        report['editor_opened'] = editor_opened
    return report
