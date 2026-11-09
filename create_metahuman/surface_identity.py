"""Conservative native-surface recognition from complete UV triangle layouts."""
import numpy as np
from mathutils.kdtree import KDTree


def triangle_count(obj):
    sizes = np.empty(len(obj.data.polygons), dtype=np.int32)
    obj.data.polygons.foreach_get('loop_total', sizes)
    return int(np.maximum(sizes - 2, 0).sum())


def layout(objects):
    coordinates, triangles = [], []
    offset = 0
    for obj in objects:
        mesh = obj.data
        if not mesh.uv_layers.active:
            return None
        uv = np.empty((len(mesh.loops), 2), dtype=np.float32)
        mesh.uv_layers.active.data.foreach_get('uv', uv.ravel())
        if not np.isfinite(uv).all():
            return None
        mesh.calc_loop_triangles()
        indices = np.empty((len(mesh.loop_triangles), 3), dtype=np.int32)
        mesh.loop_triangles.foreach_get('loops', indices.ravel())
        coordinates.append(uv)
        triangles.append(indices + offset)
        offset += len(uv)
    uv, inverse = np.unique(np.concatenate(coordinates), axis=0, return_inverse=True)
    return uv, inverse[np.concatenate(triangles)]


def ordered(triangles):
    values = np.sort(triangles, axis=1)
    return values[np.lexsort((values[:, 2], values[:, 1], values[:, 0]))]


def matches(source, references, tolerance=3e-6):
    """Require every UV coordinate and triangle, including multiplicities.

    Spatial proportions, transforms, vertex IDs, face order and winding do not
    identify a mesh. Tessellation must agree, so ambiguous variants fall back to
    ordinary import placement rather than guessing from a name or vertex count.
    """
    if not references or not source.data.uv_layers.active:
        return False
    count = sum(triangle_count(obj) for obj in references)
    # Reject ordinary dense scans before allocating their UV/triangle arrays.
    if len(source.data.polygons) > count or triangle_count(source) != count:
        return False
    reference = layout(references)
    candidate = layout([source])
    if reference is None or candidate is None:
        return False
    uv, triangles = reference
    points, faces = candidate
    if len(points) != len(uv) or not len(uv):
        return False
    tree = KDTree(len(uv))
    for index, point in enumerate(uv):
        tree.insert((float(point[0]), float(point[1]), 0.), index)
    tree.balance()
    mapping = np.empty(len(points), dtype=np.int32)
    for index, point in enumerate(points):
        _, nearest, distance = tree.find((float(point[0]), float(point[1]), 0.))
        if nearest is None or distance > tolerance:
            return False
        mapping[index] = nearest
    if len(np.unique(mapping)) != len(uv):
        return False
    return np.array_equal(ordered(mapping[faces]), ordered(triangles))


def reference_parts(source,s,f):
    """Identify supported complete skin surfaces and optional native anatomy."""
    base=[('head',s.template)]+([('body',f.body_template)] if s.part=='FULL_BODY' else [])
    if any(not obj for _,obj in base):return []
    eyes=[(name,getattr(f,name)) for name in ('eye_left','eye_right') if getattr(f,name)]
    teeth=[('teeth',f.teeth)] if f.teeth else []
    options=[[],teeth,eyes,[*eyes,*teeth]]
    seen=set()
    for extra in options:
        parts=[*base,*extra];key=tuple(obj.as_pointer() for _,obj in parts)
        if key in seen:continue
        seen.add(key)
        if matches(source,[obj for _,obj in parts]):return parts
    return []


def recognized(source,s,f):return bool(reference_parts(source,s,f))
