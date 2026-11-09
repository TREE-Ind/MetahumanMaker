"""Executed inside the user's installed Unreal editor; exports only to local storage."""
from pathlib import Path
import json
import hashlib
import unreal


def inspection_data(dna, output, raw_controls):
    import runpy
    poses = runpy.run_path(str(Path(__file__).with_name('expression_controls.py')))['inspection_poses'](raw_controls)
    if not poses:
        return {'poses': []}
    output = Path(output)
    request = output.with_name(output.stem + '-request.json')
    request.write_text(json.dumps({'poses': poses,'metahuman_compatibility':True}), encoding='utf-8')
    if not unreal.CreateMHExpressionLibrary.export_expression_reference_json(str(dna), str(request), str(output)):
        raise RuntimeError('Native inspection pose extraction failed')
    data = json.loads(output.read_text(encoding='utf-8-sig'))
    if data.get('metahuman_compatibility') is not True:raise RuntimeError('Update MetaHuman Maker Bridge to 0.12 or later for compatible MetaHuman pose evaluation')
    if [(p['name'], p['controls']) for p in data['poses']] != [(p['name'], p['controls']) for p in poses]:
        raise RuntimeError('Native inspection poses are incomplete')
    return data


def extract_reference(output, dna=None):
    output=Path(output)
    output.parent.mkdir(parents=True,exist_ok=True)
    if dna is None:
        dna=Path(unreal.Paths.engine_plugins_dir())/'MetaHuman/MetaHumanCoreTechLib/Content/ArchetypeDNA/SKM_Face.dna'
    dna=Path(dna).resolve()
    with dna.open('rb') as stream:source_hash=hashlib.file_digest(stream,'sha256').hexdigest()
    library=getattr(unreal,'CreateMHExpressionLibrary',None)
    if library is None:
        raise RuntimeError('Expression extraction needs MetaHuman Maker Bridge 0.12 or later')
    inspect=output.with_name('inspect.json')
    request=output.with_name('request.json')
    request.write_text(json.dumps({'poses':[],'metahuman_compatibility':True}))
    if not library.export_expression_reference_json(str(dna),str(request),str(inspect)):
        raise RuntimeError('Could not inspect the local DNA reference')
    metadata=json.loads(inspect.read_text())
    if metadata.get('metahuman_compatibility') is not True:raise RuntimeError('Update MetaHuman Maker Bridge to 0.12 or later for compatible MetaHuman pose evaluation')
    if not metadata['raw_controls'] or not metadata['joint_group_count']:
        raise RuntimeError('This DNA has no facial behavior to evaluate')
    if not any(mesh['name']=='head_lod0_mesh' for mesh in metadata.get('meshes',[])):
        raise RuntimeError('Choose facial MetaHuman DNA containing a LOD0 head')
    poses=[{'name':'Neutral','controls':{}}]
    poses.extend({'name':name.removeprefix('CTRL_expressions.'),'controls':{name:1}} for name in metadata['raw_controls'])
    request.write_text(json.dumps({'poses':poses,'metahuman_compatibility':True}))
    if not library.export_expression_reference_json(str(dna),str(request),str(output)):
        raise RuntimeError('Native expression extraction failed')
    result=json.loads(output.read_text())
    if len(result['poses'])!=len(poses):raise RuntimeError('Expression extraction is incomplete')
    if result.get('animated_maps'):
        import runpy
        composites=runpy.run_path(str(Path(__file__).with_name('wrinkle_controls.py')))['derive'](result)
        request.write_text(json.dumps({'poses':composites,'metahuman_compatibility':True}))
        wrinkle_file=output.with_name('wrinkle-bases.json')
        if not library.export_expression_reference_json(str(dna),str(request),str(wrinkle_file)):
            raise RuntimeError('Native wrinkle basis extraction failed')
        wrinkle_data=json.loads(wrinkle_file.read_text())
        if len(wrinkle_data['poses'])!=3:raise RuntimeError('Native wrinkle bases are incomplete')
        result['wrinkle_poses']={slot:len(result['poses'])+i for i,slot in enumerate(['WM1','WM2','WM3'])}
        result['poses'].extend(wrinkle_data['poses'])
        result['wrinkle_basis_method']='native_driver_composite_v1'
    checks = inspection_data(dna, output.with_name('inspection-poses.json'), metadata['raw_controls'])
    result['inspection_poses'] = {pose['name']: len(result['poses']) + i for i, pose in enumerate(checks['poses'])}
    result['poses'].extend(checks['poses'])
    with dna.open('rb') as stream:final_hash=hashlib.file_digest(stream,'sha256').hexdigest()
    if final_hash!=source_hash:
        raise RuntimeError('The DNA file changed during extraction; previous library retained')
    result['dna_source']={'path':str(dna),'sha256':source_hash}
    output.write_text(json.dumps(result),encoding='utf-8')
    unreal.log('CREATE_MH_NATIVE_EXPRESSIONS_COMPLETE '+str(len(result['poses'])))
    return str(output)
