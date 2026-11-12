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
                         f'the destination volume has {free/GIB:.1f} GiB free. Free generated files or choose another Project Folder. The operation was not started.')
    return {'available_disk_bytes':free,'required_disk_bytes':minimum,'checked_path':str(volume)}


def maps(size, objects=()):
    # Six RGBA float buffers approximate the tiled 8K bake's measured working
    # set; geometry and a worker-startup allowance are additional estimates.
    # A production 8K worker exceeded the earlier six-buffer estimate while
    # Cycles loaded frozen material images. File-backed scratch alone does not
    # justify lowering the whole-worker guard. Retain two more buffers at 8K.
    buffers=8 if int(size)>4096 else 6
    return 512 * MIB + buffers * int(size) ** 2 * 16 + meshes(objects)


def scene_snapshot_storage(scene):
    """Cheap disk estimate for a scene carrier, without mesh/pixel readback.

    Packed images count their stored bytes; only generated/dirty image buffers
    need a raw estimate. Clean external images stay external in the carrier.
    The last saved file is an additional conservative size floor.
    """
    import bpy
    geometry=meshes(scene.objects)
    for obj in scene.objects:
        if obj.type=='CURVES':geometry+=len(obj.data.points)*64+len(obj.data.curves)*64
        elif obj.type=='CURVE':
            geometry+=sum((len(spline.points)+len(spline.bezier_points))*128 for spline in obj.data.splines)
    images=0
    for image in bpy.data.images:
        if image.source not in {'FILE','GENERATED','TILED'}:continue
        packed=list(image.packed_files)
        stored=sum(item.packed_file.size for item in packed)
        images+=stored
        if image.source=='GENERATED' or image.is_dirty:
            tiles=max(1,len(image.tiles)) if image.source=='TILED' else 1
            images+=int(image.size[0])*int(image.size[1])*max(4,image.channels)*(4 if image.is_float else 1)*tiles
    saved=Path(bpy.data.filepath) if bpy.data.filepath else None
    saved_bytes=saved.stat().st_size if saved and saved.is_file() else 0
    return dict(scene_estimate_bytes=max(saved_bytes,geometry+images),geometry_estimate_bytes=geometry,
                image_estimate_bytes=images,saved_file_bytes=saved_bytes,estimate=True)


def project_save_storage(scene):
    """Estimate a whole-file save, including images the save will pack."""
    import bpy
    snapshot=scene_snapshot_storage(scene)
    geometry=meshes(bpy.data.objects)
    for obj in bpy.data.objects:
        if obj.type=='CURVES':geometry+=len(obj.data.points)*64+len(obj.data.curves)*64
        elif obj.type=='CURVE':
            geometry+=sum((len(spline.points)+len(spline.bezier_points))*128 for spline in obj.data.splines)
    external=0
    for image in bpy.data.images:
        if not image.users or image.packed_file or image.is_dirty or image.source not in {'FILE','TILED'}:continue
        path=Path(bpy.path.abspath(image.filepath)) if image.filepath else None
        if image.source=='FILE' and path and path.is_file():external+=path.stat().st_size
        elif image.source=='TILED':
            external+=int(image.size[0])*int(image.size[1])*max(4,image.channels)*(4 if image.is_float else 1)*max(1,len(image.tiles))
    carrier=max(snapshot['saved_file_bytes'],geometry+snapshot['image_estimate_bytes']+external)
    return dict(required_bytes=2*carrier+256*MIB,scene_estimate_bytes=carrier,
        geometry_estimate_bytes=geometry,external_image_estimate_bytes=external,estimate=True)


def detail_update_storage(inputs, *, subdivide=False):
    """NPZ carriers plus optional next-level meshes, before serializing arrays."""
    objects=[];references=0;subdivision=0
    for source,detail,reference in inputs:
        objects.extend((source,detail))
        objects.extend(getattr(reference,'objects',()))
        references+=int(getattr(reference,'nbytes',0))
        if subdivide:
            # One Catmull-Clark step introduces an edge and face point. This
            # also covers triangular/ngon input, whose growth can exceed 4V.
            data=detail.data;count=len(data.vertices)+len(data.edges)+len(data.polygons)
            keys=len(data.shape_keys.key_blocks) if data.shape_keys else 0
            subdivision+=count*(192+24*keys)
    carriers=meshes(objects)+references
    return dict(required_bytes=2*carriers+2*subdivision+256*MIB,
                carrier_estimate_bytes=carriers,subdivision_estimate_bytes=subdivision,estimate=True)
