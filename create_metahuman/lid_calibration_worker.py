"""Private standalone numerical worker; does not import Blender or Unreal."""
import os,sys,json,types,traceback
from pathlib import Path
os.environ.setdefault('OPENBLAS_NUM_THREADS','2');os.environ.setdefault('OMP_NUM_THREADS','2')
folder=Path(sys.argv[1]).resolve();package=types.ModuleType('create_metahuman');package.__path__=[str(Path(__file__).parent)]
sys.modules['create_metahuman']=package
def write(name,value):
    path=folder/name;temporary=path.with_suffix('.tmp');temporary.write_text(json.dumps(value,indent=2));temporary.replace(path)
try:
    request=json.loads((folder/'preparation-request.json').read_text())
    write('progress.json',dict(phase='Converting the protected sculpt into native corrective fields' if request.get('sculpt_snapshot') else 'Deriving lid anatomy and preparing native corrective fields'))
    if request.get('neutral_fit'):
        from create_metahuman.native_neutral_lid_fit import prepare
        report=prepare(**request['neutral_fit'],destination=folder/'prepared',progress=lambda message:write('progress.json',dict(phase=message)))
    elif request.get('repair_bank'):
        from create_metahuman.native_lid_repair_bank import prepare
        report=prepare(**request['repair_bank'],destination=folder/'prepared',progress=lambda message:write('progress.json',dict(phase=message)))
    elif request.get('sculpt_snapshot'):
        from create_metahuman.native_pose_sculpt_solver import prepare
        report=prepare(request['sculpt_snapshot'],folder/'prepared')
    else:
        from create_metahuman.lid_calibration import prepare
        report=prepare(request['capture'],folder/'inputs',folder/'prepared')
except Exception:report=dict(status='failed',error=traceback.format_exc())
write('preparation-result.json',report)
print(json.dumps(report),flush=True)
