"""Install the small optional surface solver; no AI model downloads."""
import os,subprocess,sys,venv
from pathlib import Path
root=Path(__file__).resolve().parents[1];environment=root/'.venv'
if not environment.exists():venv.EnvBuilder(with_pip=True).create(environment)
python=environment/('Scripts/python.exe' if os.name=='nt' else 'bin/python')
subprocess.run([str(python),str(root/'create_metahuman/surface_fit_worker.py'),'--install'],check=True)
