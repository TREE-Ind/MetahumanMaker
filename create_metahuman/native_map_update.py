"""Stage selected native map channels using fitted geometry as correspondence."""
import json,re
from pathlib import Path
import bpy


def available(scene):
    s,f=scene.cm_settings,scene.cm_features
    name=re.sub(r'[^A-Za-z0-9_]','_',s.character_name).strip('_')
    return bool(f.unreal_character and f.unreal_character!=f.unreal_revision_character and
                not (Path(bpy.path.abspath(s.project_dir))/'Export'/name/'last-character.json').is_file())


def draw(layout,scene):
    if not available(scene):return False
    s,f=scene.cm_settings,scene.cm_features
    if s.part=='FULL_BODY':layout.prop(f,'unreal_native_map_parts')
    layout.prop(f,'unreal_native_map_channels')
    return True


def prepare(context,folder,*,selection=None):
    from . import native_albedo_update
    s,f=context.scene.cm_settings,context.scene.cm_features
    root=Path(bpy.path.abspath(s.project_dir)).resolve()
    name=re.sub(r'[^A-Za-z0-9_]','_',s.character_name).strip('_')
    history=root/'Export'/name/'last-texture-action.json'
    if history.is_file():
        previous=json.loads(history.read_text(encoding='utf-8'))
        receipt=Path(previous.get('receipt','')).resolve()
        if (previous.get('character')==f.unreal_character and receipt.is_relative_to(root/'Export'/'.unreal-actions')
                and receipt.name=='receipt.json' and not receipt.is_file()):
            raise ValueError('Resume the original texture action before preparing another: '+str(receipt))
    if selection is None:
        roles=('HEAD','BODY') if s.part=='FULL_BODY' and f.unreal_native_map_parts=='BOTH' else (
            f.unreal_native_map_parts if s.part=='FULL_BODY' else s.part,)
        kinds={'DETAIL':['NORMAL','CAVITY'],'FINISH':['ROUGHNESS'],'DETAIL_FINISH':['NORMAL','CAVITY','ROUGHNESS'],'AO':['BENT_NORMAL_AO']}[f.unreal_native_map_channels]
        selection={role:list(kinds) for role in roles}
    request=native_albedo_update.prepare(context,Path(folder),selection=selection)
    # The same durable record is understood by normal Unreal receipt resume.
    history.parent.mkdir(parents=True,exist_ok=True)
    record=dict(schema_version=1,project_dir=str(root),character_name=s.character_name,character=f.unreal_character,
                unreal_project_file=f.unreal_character_project,receipt=str(Path(folder).resolve()/'receipt.json'))
    temp=history.with_suffix('.tmp');temp.write_text(json.dumps(record,indent=2),encoding='utf-8');temp.replace(history)
    return request
