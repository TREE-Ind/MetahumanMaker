"""Geometry-only entry point for the optional accurate regional solver."""
import os
os.environ['OPENBLAS_NUM_THREADS']='2'
os.environ['OMP_NUM_THREADS']='2'
import json,sys,types,threading,traceback,time
from pathlib import Path


def write_json(path,value):
    temporary=path.with_suffix('.tmp');temporary.write_text(json.dumps(value),encoding='utf-8')
    # Windows may briefly deny replacement while the editor reads progress.
    # A progress read must not turn an otherwise valid fit into a failure.
    for attempt in range(30):
        try:temporary.replace(path);return
        except PermissionError:
            if attempt==29:raise
            time.sleep(.01)


def main():
    request_path=Path(sys.argv[1]).resolve();folder=request_path.parent
    request=json.loads(request_path.read_text(encoding='utf-8'))
    package=types.ModuleType('create_metahuman');package.__path__=[str(Path(__file__).resolve().parent)]
    sys.modules['create_metahuman']=package
    from create_metahuman.worker_lifetime import watch_parent
    watch_parent(request['parent'])
    watchdog=threading.Timer(360. if request['options'].get('register_attachment') else 240.,lambda:os._exit(124));watchdog.daemon=True;watchdog.start()
    try:
        try:
            import numpy as np
            if request['options'].get('refine_globes') and request['options'].get('globe_volume'):from create_metahuman.ocular_globe_volume import fit_eyes as fit
            elif request['options'].get('refine_globes'):from create_metahuman.ocular_globe_refine import fit
            elif request['options'].get('ear_surface'):from create_metahuman.ear_surface import fit
            elif request['options'].get('ocular'):from create_metahuman.ocular_surface import fit
            elif request['options'].get('complete_missing'):from create_metahuman.partial_surface_fit import fit
            elif request['options'].get('reconstruct'):from create_metahuman.intrinsic_attachment_fit import fit
            else:from create_metahuman.accurate_surface import fit
        except ImportError as exc:
            raise RuntimeError('Accurate fitting needs its optional dependencies. Run Inference Python with '+str(Path(__file__).resolve())+' --install, then retry.') from exc
        with np.load(folder/'input.npz',allow_pickle=False) as archive:data={key:archive[key] for key in archive.files}
        body,reports,proof=fit(data,request['options'],
            progress=lambda label,value:write_json(folder/'progress.json',dict(label=label,value=value)))
        temporary=folder/'output-pending.npz';np.savez_compressed(temporary,body=body,**{'proof_'+key:value for key,value in proof.items()})
        temporary.replace(folder/'output.npz')
        write_json(folder/'result.json',dict(status='complete',reports=reports))
    finally:watchdog.cancel()


if __name__=='__main__':
    if sys.argv[1:] == ['--install']:
        import subprocess
        subprocess.run([sys.executable,'-m','pip','install','--disable-pip-version-check','clarabel==0.11.1','libigl==2.6.3'],check=True)
        import numpy,scipy,clarabel,igl
        print('Accurate surface fitting is ready')
        sys.exit(0)
    try:main()
    except Exception as exc:
        traceback.print_exc()
        write_json(Path(sys.argv[1]).resolve().parent/'result.json',dict(status='error',error=str(exc)))
        sys.exit(1)
