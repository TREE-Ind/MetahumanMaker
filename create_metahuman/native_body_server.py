"""Isolated Unreal commandlet loop for the installed native anatomical body model."""

from pathlib import Path
import ctypes
import json
import os
import time
import runpy
import unreal

_IPC=runpy.run_path(str(Path(__file__).with_name('body_ipc.py')))


def write_json(path, data):
    deadline=time.monotonic()+3
    while not _IPC['write'](path,data):
        if time.monotonic()>=deadline:raise PermissionError('Native body response remained locked: '+str(path))
        time.sleep(.05)


def parent_alive(pid):
    if os.name != "nt":
        return True
    from ctypes import wintypes

    kernel = ctypes.windll.kernel32
    kernel.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
    kernel.OpenProcess.restype = wintypes.HANDLE
    kernel.GetExitCodeProcess.argtypes = [
        wintypes.HANDLE,
        ctypes.POINTER(wintypes.DWORD),
    ]
    kernel.CloseHandle.argtypes = [wintypes.HANDLE]
    handle = kernel.OpenProcess(0x1000, False, pid)
    if not handle:
        return False
    try:
        code = wintypes.DWORD()
        return (
            bool(kernel.GetExitCodeProcess(handle, ctypes.byref(code)))
            and code.value == 259
        )
    finally:
        kernel.CloseHandle(handle)


def serve(config_path):
    config = json.loads(Path(config_path).read_text())
    folder = Path(config["folder"])
    base = {
        "model_path": config["model_path"],
        "constraints": {},
        "include_layouts": True,
    }
    if config.get("state_file"):
        base["state_file"] = config["state_file"]
    request = folder / "initial-request.json"
    request.write_text(json.dumps(base))
    output = folder / "baseline.json"
    library = getattr(unreal, "CreateMHBodyLibrary", None)
    if library is None:
        raise RuntimeError("Native body shaping requires Bridge 0.4 or later")
    if not library.evaluate_body_json(str(request), str(output)):
        raise RuntimeError("Could not initialize the native body model")
    write_json(folder / "ready.json", {"baseline": output.name})
    baseline_state = str(folder / (output.name + ".state"))
    version = -1
    while parent_alive(int(config["parent"])) and not (folder / "stop.json").exists():
        queued = folder / "request.json"
        message=_IPC['read'](queued)
        if message is not None:
            if int(message["version"]) > version:
                version = int(message["version"])
                native = {
                    "model_path": config["model_path"],
                    "state_file": baseline_state,
                    "constraints": message["constraints"],
                }
                native_request = folder / f"native-request-{version}.json"
                native_request.write_text(json.dumps(native))
                output = folder / f"result-{version}.json"
                if not library.evaluate_body_json(str(native_request), str(output)):
                    write_json(
                        folder / "response.json",
                        {
                            "version": version,
                            "error": "Native constraints could not be evaluated",
                        },
                    )
                else:
                    write_json(
                        folder / "response.json",
                        {"version": version, "output": output.name},
                    )
        time.sleep(0.1)
