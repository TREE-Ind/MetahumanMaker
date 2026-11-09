"""Stop an owned worker when its parent editor exits, without using Blender APIs."""

import ctypes, os, threading, time


def parent_alive(pid):
    if os.name != "nt":
        return os.getppid() == pid
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


def watch_parent(pid, cancel=None):
    if not isinstance(pid, int) or pid <= 0:
        raise ValueError("Worker parent PID is invalid")

    def watch():
        while parent_alive(pid) and not (cancel and os.path.exists(cancel)):
            time.sleep(0.5)
        os._exit(1)

    threading.Thread(target=watch, name="CMWorkerParentWatch", daemon=True).start()
