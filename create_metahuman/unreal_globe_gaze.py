"""Stage the separated native globe gaze field on a retained rig revision."""
import json,runpy
from pathlib import Path


def stage(request_folder,destination):
    import unreal
    request=Path(request_folder);manifest=json.loads((request/'manifest.json').read_text())
    if manifest.get('method')!='BLINK_INDEPENDENT_CARDINAL_GAZE_FIELDS':raise ValueError('Prepare a complete separated globe gaze field first')
    report=runpy.run_path(str(Path(__file__).with_name('unreal_lod_correctives.py')))['stage'](request,destination)
    report.update(method=manifest['method'],blink_input_excluded=not bool(manifest.get('blink_volume_solvers')),gaze_transport_mode=manifest.get('gaze_transport_mode','PRESERVE_FITTED'),native_pupil_targets_retained=True)
    mesh=unreal.load_asset(report['candidate']);unreal.EditorAssetLibrary.set_metadata_tag(mesh,'CreateMetaHuman.GlobeGazeRevision',json.dumps(report))
    if not unreal.EditorAssetLibrary.save_loaded_asset(mesh,False):raise RuntimeError('Could not save the separated gaze-field revision')
    (request/'native/gaze-stage.json').write_text(json.dumps(report,indent=2));return report
