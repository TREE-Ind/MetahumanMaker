"""Read process identity and focus only the identified editor window."""

import ctypes, os


def identify(pid):
    if os.name != "nt":
        try:
            fields = (
                open("/proc/" + str(pid) + "/stat").read().rpartition(")")[2].split()
            )
            return {"pid": pid, "born": fields[19]} if fields[0] != "Z" else None
        except OSError:
            return None
    from ctypes import wintypes

    kernel = ctypes.windll.kernel32
    kernel.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
    kernel.OpenProcess.restype = wintypes.HANDLE
    kernel.GetExitCodeProcess.argtypes = [
        wintypes.HANDLE,
        ctypes.POINTER(wintypes.DWORD),
    ]
    kernel.GetProcessTimes.argtypes = [wintypes.HANDLE] + [
        ctypes.POINTER(wintypes.FILETIME)
    ] * 4
    kernel.CloseHandle.argtypes = [wintypes.HANDLE]
    handle = kernel.OpenProcess(0x1000, False, pid)
    if not handle:
        return None
    try:
        code = wintypes.DWORD()
        times = [wintypes.FILETIME() for _ in range(4)]
        if (
            not kernel.GetExitCodeProcess(handle, ctypes.byref(code))
            or code.value != 259
        ):
            return None
        if not kernel.GetProcessTimes(handle, *[ctypes.byref(t) for t in times]):
            return None
        return {
            "pid": pid,
            "born": str((times[0].dwHighDateTime << 32) | times[0].dwLowDateTime),
        }
    finally:
        kernel.CloseHandle(handle)


def alive(reference):
    return bool(reference) and identify(reference["pid"]) == reference


def focus(reference):
    if os.name != "nt" or not alive(reference):
        return False
    from ctypes import wintypes

    user = ctypes.windll.user32
    windows = []

    @ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HWND, wintypes.LPARAM)
    def visit(hwnd, _):
        pid = wintypes.DWORD()
        user.GetWindowThreadProcessId(hwnd, ctypes.byref(pid))
        if pid.value == reference["pid"] and user.IsWindowVisible(hwnd):
            rect = wintypes.RECT()
            user.GetWindowRect(hwnd, ctypes.byref(rect))
            windows.append(((rect.right - rect.left) * (rect.bottom - rect.top), hwnd))
        return True

    user.EnumWindows(visit, 0)
    if not windows:
        return False
    hwnd = max(windows)[1]
    user.ShowWindow(hwnd, 9)
    return bool(user.SetForegroundWindow(hwnd))
