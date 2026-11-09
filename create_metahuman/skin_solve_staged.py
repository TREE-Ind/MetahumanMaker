"""Prepare and dispatch one immutable skin worker independently of Blender."""
import ast
import hashlib
import json
import os
from pathlib import Path
import sys
import threading


def command(folder):
    """A pasteable command for the user's shell, with literal path quoting."""
    folder=Path(folder).resolve();plan=json.loads((folder/'staged.json').read_text())
    args=[plan['python'],str(folder/'scripts/skin_solve_staged.py'),str(folder)]
    if os.name=='nt':return '& '+' '.join("'"+arg.replace("'","''")+"'" for arg in args)
    import shlex
    return shlex.join(args)


def write_launcher(folder):
    """An interactive Windows launcher; Python paths come from JSON, not shell text."""
    folder=Path(folder)
    if os.name!='nt':return {}
    powershell=r'''$ErrorActionPreference = 'Stop'
$workerExit = 1
try {
    $job = $PSScriptRoot
    $plan = Get-Content -LiteralPath (Join-Path $job 'staged.json') -Raw | ConvertFrom-Json
    $recoveryPath = Join-Path $job 'recovery.json'
    if (-not (Test-Path -LiteralPath $recoveryPath)) { throw 'Use Save for Recovery in Blender before closing it and running this launcher.' }
    $recovery = Get-Content -LiteralPath $recoveryPath -Raw | ConvertFrom-Json
    if ($recovery.request_sha256 -ne $plan.request_sha256 -or -not (Test-Path -LiteralPath $recovery.project)) { throw 'The saved recovery project is missing or belongs to another solve.' }
    Write-Host 'Running the prepared skin solve. Keep this window open until it finishes.'
    Write-Host ('Job: ' + $job)
    $ErrorActionPreference = 'Continue'
    & $plan.python (Join-Path $job 'scripts/skin_solve_staged.py') $job 2>&1 | Tee-Object -FilePath (Join-Path $job 'launcher.log') -Append
    $workerExit = $LASTEXITCODE
    $ErrorActionPreference = 'Stop'
    if ($workerExit -eq 0) { Write-Host 'Finished. Reopen the saved project to preview and apply the candidate.' }
    else { Write-Host 'The solve did not finish. Reopen the saved project and inspect the original job; no replacement was submitted.' }
} catch { Write-Host $_.Exception.Message }
[void](Read-Host 'Press Enter to close')
exit $workerExit
'''
    batch='@echo off\r\nsetlocal DisableDelayedExpansion\r\npowershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0run-skin-solve.ps1"\r\nexit /b %errorlevel%\r\n'
    (folder/'run-skin-solve.ps1').write_text(powershell,encoding='utf-8')
    (folder/'run-skin-solve.cmd').write_bytes(batch.encode('ascii'))
    return {name:digest(folder/name) for name in ('run-skin-solve.ps1','run-skin-solve.cmd')}


def digest(path):
    with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()


def prepare(folder,python,required_memory,required_disk):
    import runpy
    folder=Path(folder).resolve();source=Path(__file__).resolve().parent
    scripts=folder/'scripts';scripts.mkdir(exist_ok=False)
    pending=['skin_reconstruction_worker','skin_solve_staged','process_ref','resource_budget'];seen=set()
    while pending:
        name=pending.pop()
        if name in seen:continue
        path=source/(name+'.py')
        if not path.is_file():continue
        seen.add(name);body=path.read_bytes();(scripts/path.name).write_bytes(body)
        for node in ast.walk(ast.parse(body)):
            if isinstance(node,ast.Import):pending.extend(alias.name.split('.')[0] for alias in node.names)
            elif isinstance(node,ast.ImportFrom) and node.module:pending.append(node.module.split('.')[0])
    plan=dict(schema=1,request_sha256=digest(folder/'request.json'),python=str(Path(python).resolve()),
        required_memory=int(required_memory),required_disk=int(required_disk),
        scripts={name+'.py':digest(scripts/(name+'.py')) for name in sorted(seen)})
    plan['producer_process']=runpy.run_path(str(source/'process_ref.py'))['identify'](os.getpid())
    if not plan['producer_process']:raise RuntimeError('Could not identify the authoring process for this staged solve')
    plan['launchers']=write_launcher(folder)
    (folder/'staged.json').write_text(json.dumps(plan,indent=2),encoding='utf-8')
    return plan


def alive(folder):
    """None means an ambiguous dispatch record; never delete its inputs yet."""
    folder=Path(folder)
    if not (folder/'dispatch.json').exists():return False
    try:
        identity=json.loads((folder/'dispatch.json').read_text())['process']
        import runpy
        return runpy.run_path(str(Path(__file__).with_name('process_ref.py')))['alive'](identity)
    except (OSError,ValueError,KeyError):return None


def dispatch(folder):
    import runpy
    folder=Path(folder).resolve();plan=json.loads((folder/'staged.json').read_text())
    request=folder/'request.json'
    if plan.get('schema')!=1 or digest(request)!=plan.get('request_sha256'):raise ValueError('The staged skin request changed')
    if json.loads(request.read_text()).get('parent') is not None:raise ValueError('This skin worker is still owned by Blender')
    if (folder/'cancel').exists() or (folder/'result.json').exists():raise ValueError('This skin action is cancelled or already has a result; observe its original files')
    if {path.name for path in (folder/'scripts').glob('*.py')}!=set(plan['scripts']):raise ValueError('The staged runtime file list changed')
    for name,sha in plan['scripts'].items():
        if Path(name).name!=name or digest(folder/'scripts'/name)!=sha:raise ValueError('The staged skin runtime changed: '+name)
    for name,sha in plan.get('launchers',{}).items():
        if Path(name).name!=name or digest(folder/name)!=sha:raise ValueError('The staged launcher changed: '+name)
    if Path(sys.executable).resolve()!=Path(plan['python']).resolve():raise ValueError('Use the staged inference Python environment')
    processes=runpy.run_path(str(folder/'scripts/process_ref.py'))
    if plan.get('producer_process') and processes['alive'](plan['producer_process']):
        raise ValueError('Close the authoring Blender before starting this prepared solve; its image buffers are still resident')
    budget=runpy.run_path(str(folder/'scripts/resource_budget.py'))
    # Refused preflight is retryable on this same prepared action. Exclusively
    # claim only after these cheap checks; parallel callers still cannot solve twice.
    budget['require']('staged skin reconstruction',plan['required_memory'])
    budget['require_disk']('staged skin reconstruction',folder,plan['required_disk'])
    identity=processes['identify'](os.getpid())
    if not identity:raise RuntimeError('Could not identify the staged worker process; no solve was started')
    with (folder/'dispatch.json').open('x',encoding='utf-8') as stream:
        json.dump(dict(process=identity,request_sha256=plan['request_sha256']),stream)
    exit_code=1
    finished=threading.Event()
    try:
        budget['require']('staged skin reconstruction',plan['required_memory'])
        budget['require_disk']('staged skin reconstruction',folder,plan['required_disk'])
        def watch_headroom():
            while not finished.wait(.5):
                try:available=budget['memory']()
                except (OSError,ValueError):continue
                if available and available['available_commit_bytes']<2*budget['GIB']:
                    try:
                        (folder/'staged-stop.json').write_text(json.dumps(dict(reason='System commit headroom fell below 2 GiB',memory=available)),encoding='utf-8')
                        (folder/'cancel').touch()
                    except OSError:os._exit(1)
                    return
        threading.Thread(target=watch_headroom,name='CMStagedSkinBudget',daemon=True).start()
        os.environ.update(OPENBLAS_NUM_THREADS='2',OMP_NUM_THREADS='2')
        sys.path.insert(0,str(folder/'scripts'))
        runpy.run_path(str(folder/'scripts/skin_reconstruction_worker.py'))['run'](request,owner=os.getppid())
        exit_code=0
    except Exception as exc:
        (folder/'result.json').write_text(json.dumps(dict(schema=1,status='failed',error=str(exc))),encoding='utf-8')
    finally:
        finished.set()
        result=json.loads((folder/'result.json').read_text())
        result.update(staged_request_sha256=plan['request_sha256'],staged_process=identity)
        temporary=folder/'result.tmp';temporary.write_text(json.dumps(result,indent=2));temporary.replace(folder/'result.json')
        (folder/'exit.json').write_text(json.dumps(dict(process=identity,exit_code=exit_code)),encoding='utf-8')
    return exit_code


if __name__=='__main__':raise SystemExit(dispatch(sys.argv[1]))
