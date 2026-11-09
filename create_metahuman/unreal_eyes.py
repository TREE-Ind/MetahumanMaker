"""Update a linked Creator character's eye materials without another import."""
from pathlib import Path
import runpy


def update(character_path, options, expected_project=''):
    import unreal
    project=str(Path(unreal.Paths.get_project_file_path()).resolve())
    if not expected_project or Path(expected_project).resolve()!=Path(project).resolve():
        raise ValueError('Open the Unreal project used for this character delivery')
    character=unreal.load_asset(character_path)
    if not isinstance(character,unreal.MetaHumanCharacter):
        raise ValueError('The linked MetaHuman Character is unavailable')
    root=Path(__file__).parent
    native=runpy.run_path(str(root/'unreal_import.py'))
    options=native['eye_options'](options)
    subsystem=unreal.get_editor_subsystem(unreal.MetaHumanCharacterEditorSubsystem)
    owned=not subsystem.is_object_added_for_editing(character)
    try:
        if owned and not subsystem.try_add_object_to_edit(character):
            raise RuntimeError('Creator could not open this character for eye editing')
        before=character.get_editor_property('eyes_settings').export_text()
        result=native['apply_eye_settings'](character,subsystem,options)
        after=character.get_editor_property('eyes_settings').export_text()
        return {'status':'complete','character':character.get_path_name(),'action':'UPDATE_EYES',
                'unreal_project_file':project,'options':result,'before':before,'after':after,'saved':True}
    finally:
        if owned:subsystem.remove_object_to_edit(character)
