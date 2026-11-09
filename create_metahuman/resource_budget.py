"""Cheap Windows commit-headroom checks before expensive snapshots and workers."""

import ctypes
import os
import shutil
from pathlib import Path

GIB = 1024 ** 3
MIB = 1024 ** 2


def process_memory():
    """OS peak commit/working set for the current worker, when available."""
    if os.name!='nt':return None
    class Counters(ctypes.Structure):
        _fields_=[('size',ctypes.c_uint32),('faults',ctypes.c_uint32),
            *[(name,ctypes.c_size_t) for name in ('peak_working','working','peak_paged_pool','paged_pool',
                'peak_nonpaged_pool','nonpaged_pool','commit','peak_commit')]]
    counters=Counters();counters.size=ctypes.sizeof(counters)
    query=ctypes.WinDLL('psapi').GetProcessMemoryInfo
    query.argtypes=[ctypes.c_void_p,ctypes.c_void_p,ctypes.c_uint32];query.restype=ctypes.c_int
    if not query(ctypes.c_void_p(-1),ctypes.byref(counters),counters.size):return None
    return dict(peak_commit_bytes=int(counters.peak_commit),peak_working_set_bytes=int(counters.peak_working),
                current_commit_bytes=int(counters.commit),current_working_set_bytes=int(counters.working))


class MemoryStatus(ctypes.Structure):
    _fields_ = [('length', ctypes.c_uint32), ('load', ctypes.c_uint32),
                ('total_physical', ctypes.c_uint64), ('available_physical', ctypes.c_uint64),
                ('commit_limit', ctypes.c_uint64), ('available_commit', ctypes.c_uint64),
                ('total_virtual', ctypes.c_uint64), ('available_virtual', ctypes.c_uint64),
                ('extended_virtual', ctypes.c_uint64)]


def memory():
    if os.name != 'nt':
        return None
    status = MemoryStatus()
    status.length = ctypes.sizeof(status)
    kernel = ctypes.WinDLL('kernel32', use_last_error=True)
    query = kernel.GlobalMemoryStatusEx
    query.argtypes = [ctypes.POINTER(MemoryStatus)]
    query.restype = ctypes.c_int
    if not query(ctypes.byref(status)):
        raise ValueError('Could not read Windows memory headroom before starting this operation')
    return {'available_commit_bytes': int(status.available_commit),
            'commit_limit_bytes': int(status.commit_limit),
            'available_physical_bytes': int(status.available_physical),
            'total_physical_bytes': int(status.total_physical)}


def require(operation, minimum):
    minimum = int(minimum)
    if minimum <= 0:
        raise ValueError('Memory budget must be positive')
    status = memory()
    if status and status['available_commit_bytes'] < minimum:
        raise ValueError(
            f'Not enough memory headroom for {operation}: allow about {minimum / GIB:.1f} GiB additional; '
            f"Windows currently reports {status['available_commit_bytes'] / GIB:.1f} GiB. "
            'Close heavy applications or reduce resolution/detail, then retry. The worker was not started.')
    return status


def meshes(objects, *, subdivide=False):
    """Estimate geometry/shape buffers without evaluating or copying any mesh."""
    total = 0
    for obj in dict.fromkeys(obj for obj in objects if obj is not None):
        if obj.type != 'MESH':
            continue
        keys = len(obj.data.shape_keys.key_blocks) if obj.data.shape_keys else 0
        count = len(obj.data.vertices) * (4 if subdivide else 1)
        total += count * (192 + 24 * keys)
    return total


def require_disk(operation,path,minimum):
    """Check the destination volume before creating any job directories."""
    minimum=int(minimum)
    if minimum<=0:raise ValueError('Disk budget must be positive')
    volume=Path(path).resolve()
    while not volume.exists() and volume.parent!=volume:volume=volume.parent
    try:free=shutil.disk_usage(volume).free
    except OSError as exc:raise ValueError('Could not read destination disk space before '+operation) from exc
    if free<minimum:
        raise ValueError(f'Not enough disk space for {operation}: allow about {minimum/GIB:.1f} GiB additional; '
                         f'the destination volume has {free/GIB:.1f} GiB free. Free generated files or choose another Project Folder. The worker was not started.')
    return {'available_disk_bytes':free,'required_disk_bytes':minimum,'checked_path':str(volume)}


def maps(size, objects=()):
    # Six RGBA float buffers approximate the tiled 8K bake's measured working
    # set; geometry and a worker-startup allowance are additional estimates.
    # A production 8K worker exceeded the earlier six-buffer estimate while
    # Cycles loaded frozen material images. File-backed scratch alone does not
    # justify lowering the whole-worker guard. Retain two more buffers at 8K.
    buffers=8 if int(size)>4096 else 6
    return 512 * MIB + buffers * int(size) ** 2 * 16 + meshes(objects)
