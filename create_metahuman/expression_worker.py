"""Prepare a native expression library without occupying the interactive Blender editor."""
from pathlib import Path
import argparse
import json
import os
import shutil
import subprocess
import sys
import time
import uuid


def remove_owned(path, root):
    path,root=Path(path).resolve(),Path(root).resolve()
    if path==root or not path.is_relative_to(root):
        raise ValueError('Refusing cleanup outside the owned extraction folder')
    for attempt in range(20):
        try:
            shutil.rmtree(path)
            return
        except OSError:
            if attempt==19:raise
            time.sleep(.1)


def stop_editor(process):
    if process.poll() is None:
        if os.name=='nt':
            subprocess.run(['taskkill','/PID',str(process.pid),'/T','/F'],
                           stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,
                           creationflags=subprocess.CREATE_NO_WINDOW)
        else:process.terminate()
        try:process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill();process.wait()


def prepare(engine,bridge,destination,*,dna=None,cancel_file=None):
    engine,bridge,destination=Path(engine).resolve(),Path(bridge).resolve(),Path(destination).resolve()
    editor=engine/'Engine/Binaries/Win64/UnrealEditor-Cmd.exe'
    if not editor.is_file():raise ValueError('Select the installed Unreal Engine folder')
    descriptor=bridge/'CreateMetaHumanBridge.uplugin'
    if not descriptor.is_file() or not (bridge/'Binaries/Win64/UnrealEditor-CreateMetaHumanBridge.dll').is_file():
        raise ValueError('Select the compiled MetaHuman Maker Bridge 0.12 folder')
    if tuple(int(n) for n in json.loads(descriptor.read_text())['VersionName'].split('.'))<(0,12,0):
        raise ValueError('Compatible MetaHuman expression extraction requires Bridge 0.12 or later')
    if dna is not None:
        dna=Path(dna).resolve()
        if dna.suffix.lower()!='.dna' or not dna.is_file():
            raise ValueError('Choose an existing facial DNA file')
    identifier=uuid.uuid4().hex[:12]
    folder=destination/'libraries'/identifier
    folder.mkdir(parents=True,exist_ok=False)
    try:
        return extract(editor,bridge,destination,folder,dna,cancel_file)
    except Exception as exc:
        failure_log=destination/('failed-'+identifier+'.log')
        if (folder/'prepare.log').is_file():shutil.copy2(folder/'prepare.log',failure_log)
        # This UUID directory was created above, and its owned editor has exited.
        if folder.resolve().parent==destination/'libraries' and folder.name==identifier:
            try:remove_owned(folder,destination)
            except OSError:print('Partial extraction retained for cleanup: '+str(folder),flush=True)
        raise RuntimeError(str(exc)+'; log: '+str(failure_log)) from exc
    finally:
        if cancel_file:
            marker=Path(cancel_file).resolve()
            if marker.parent==destination and marker.name.startswith('cancel-') and marker.suffix=='.flag':
                marker.unlink(missing_ok=True)


def extract(editor,bridge,destination,folder,dna,cancel_file):
    if cancel_file and Path(cancel_file).is_file():
        raise RuntimeError('Expression preparation canceled; previous library retained')
    project=folder/'editor'
    shutil.copytree(bridge,project/'Plugins/CreateMetaHumanBridge',ignore=shutil.ignore_patterns('Intermediate','*.pdb'))
    uproject=project/'CreateMHExpressions.uproject'
    uproject.write_text(json.dumps({'FileVersion':3,'Plugins':[{'Name':name,'Enabled':True} for name in ['PythonScriptPlugin','CreateMetaHumanBridge']]}))
    runner=project/'extract.py'
    reference=folder/'reference.json'
    runner.write_text('import runpy\nrunpy.run_path('+repr(str(Path(__file__).with_name('unreal_expressions.py')))+')["extract_reference"]('+repr(str(reference))+', dna='+repr(str(dna) if dna else None)+')\n',encoding='utf-8')
    flags=(subprocess.CREATE_NO_WINDOW|subprocess.BELOW_NORMAL_PRIORITY_CLASS) if os.name=='nt' else 0
    with (folder/'prepare.log').open('w',encoding='utf-8') as log:
        process=subprocess.Popen([str(editor),str(uproject),'-run=pythonscript','-script='+str(runner),'-NullRHI','-unattended','-nosound','-nop4','-nosplash','-DDC=InstalledNoZenLocalFallback'],stdout=log,stderr=subprocess.STDOUT,creationflags=flags)
        canceled=False
        try:
            while process.poll() is None:
                if cancel_file and Path(cancel_file).is_file():
                    canceled=True
                    stop_editor(process)
                    break
                time.sleep(.2)
        finally:
            if process.poll() is None:
                stop_editor(process)
    if canceled or (cancel_file and Path(cancel_file).is_file()):
        raise RuntimeError('Expression preparation canceled; previous library retained')
    if process.returncode or not reference.is_file():
        raise RuntimeError('Native library preparation failed. See '+str(folder/'prepare.log'))
    data=json.loads(reference.read_text())
    if data.get('schema')!=2 or not data.get('poses'):raise ValueError('Native reference is incomplete')
    # The disposable editor contains no artist assets. Keep only the native log
    # and extracted library after its process has exited.
    if project.resolve().parent==folder.resolve():
        remove_owned(project,folder)
    if cancel_file and Path(cancel_file).is_file():
        raise RuntimeError('Expression preparation canceled; previous library retained')
    # Commit only after the child completed; previous successful libraries remain usable.
    pointer=destination/'library.json'
    temporary=destination/(folder.name+'.tmp')
    temporary.write_text(json.dumps({'reference':reference.relative_to(destination).as_posix()}))
    temporary.replace(pointer)
    return reference


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--engine',required=True)
    parser.add_argument('--bridge',required=True)
    parser.add_argument('--destination',required=True)
    parser.add_argument('--dna')
    parser.add_argument('--cancel-file')
    arguments=parser.parse_args()
    try:print('NATIVE_EXPRESSION_LIBRARY_READY',prepare(arguments.engine,arguments.bridge,arguments.destination,dna=arguments.dna,cancel_file=arguments.cancel_file),flush=True)
    except Exception as exc:
        print(str(exc),file=sys.stderr,flush=True)
        raise SystemExit(1)
