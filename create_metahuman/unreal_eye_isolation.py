"""Retain eye/teeth overrides as private material instances on a revision."""
import json

TAG='CreateMetaHuman.SourceEyeAppearance'


def isolate(materials,private,owner,source_tag):
    """Modify a copied material override struct using the caller's clone policy.

    Parent graphs and textures stay shared and read-only. Source-eye ownership
    follows the cloned active instances; historical restore state is retained.
    """
    import unreal
    assets=unreal.EditorAssetLibrary
    entries=dict(materials.get_editor_property('teeth_and_eyes'))
    owned={key:bool(value and assets.get_metadata_tag(value,'CreateMetaHuman.SourceEyeOwner')) for key,value in entries.items()}
    old={key:value.get_path_name() if value else None for key,value in entries.items()}
    entries={key:private(value) for key,value in entries.items()}
    materials.set_editor_property('teeth_and_eyes',entries)
    for key,value in entries.items():
        if value and owned[key]:
            assets.set_metadata_tag(value,'CreateMetaHuman.SourceEyeOwner',owner)
    ledger=json.loads(source_tag) if source_tag else None
    keys={'eye_left':unreal.MetaHumanCharacterTeethAndEyesSlot.EYE_LEFT,
          'eye_right':unreal.MetaHumanCharacterTeethAndEyesSlot.EYE_RIGHT}
    remapped=False
    if ledger and ledger.get('schema')==1 and all(
            old.get(key)==ledger.get('applied',{}).get(role) and entries.get(key)
            for role,key in keys.items()):
        ledger['owner']=owner
        ledger['applied']={role:entries[key].get_path_name() for role,key in keys.items()}
        remapped=True
    return json.dumps(ledger) if ledger else source_tag,dict(
        slots=len(entries),source_eye_ownership_remapped=remapped,
        materials={str(key):value.get_path_name() if value else None for key,value in entries.items()})
