"""Repair missed local skin-transfer rays using explicit surface/UV correspondence."""
import numpy as np
import hashlib
from mathutils import Vector
from mathutils.bvhtree import BVHTree
from mathutils.geometry import barycentric_transform


def uv_layout_signature(obj):
    """Identify the exact native UV assignment, including face/vertex membership."""
    if not obj.data.uv_layers.active:
        return None
    uv = np.empty((len(obj.data.uv_layers.active.data), 2), dtype=np.float32)
    obj.data.uv_layers.active.data.foreach_get('uv', uv.ravel())
    indices = np.empty(len(obj.data.loops), dtype=np.int32)
    obj.data.loops.foreach_get('vertex_index', indices)
    faces = np.empty(len(obj.data.polygons), dtype=np.int32)
    obj.data.polygons.foreach_get('loop_total', faces)
    return hashlib.sha256(np.rint(uv * 1e6).astype(np.int64).tobytes() + indices.tobytes() + faces.tobytes()).hexdigest()


def surface(context, obj):
    depsgraph = context.evaluated_depsgraph_get()
    evaluated = obj.evaluated_get(depsgraph)
    mesh = evaluated.to_mesh(preserve_all_data_layers=True, depsgraph=depsgraph)
    try:
        mesh.calc_loop_triangles()
        points = np.empty((len(mesh.vertices), 3), dtype=np.float64)
        mesh.vertices.foreach_get('co', points.ravel())
        matrix = np.asarray(obj.matrix_world)
        points = points @ matrix[:3, :3].T + matrix[:3, 3]
        triangles = np.asarray([t.loops[:] for t in mesh.loop_triangles], dtype=int)
        vertices = np.empty(len(mesh.loops), dtype=np.int32)
        mesh.loops.foreach_get('vertex_index', vertices)
        if not mesh.uv_layers.active:
            raise ValueError('Skin projection needs UVs on both surfaces')
        uv = np.empty((len(mesh.loops), 2), dtype=np.float64)
        mesh.uv_layers.active.data.foreach_get('uv', uv.ravel())
        return points, triangles, vertices, uv
    finally:
        evaluated.to_mesh_clear()


def linear(values):
    return np.where(values <= .04045, values / 12.92, ((values + .055) / 1.055) ** 2.4)


def encoded(values):
    return np.where(values <= .0031308, values * 12.92, 1.055 * np.maximum(values, 0) ** (1 / 2.4) - .055)


def sample(image, coordinates):
    width, height = image.size
    pixels = np.asarray(image.pixels[:], dtype=np.float32).reshape(height, width, 4)
    rgb = pixels[..., :3]
    if image.colorspace_settings.name == 'sRGB':
        rgb = linear(rgb)
    xy = np.asarray(coordinates) * [width, height] - .5
    cells = np.floor(xy).astype(int)
    fraction = xy - cells
    x, y = cells.T
    u, v = fraction.T
    return ((rgb[y % height, x % width] * (1 - u[:, None]) + rgb[y % height, (x + 1) % width] * u[:, None]) * (1 - v[:, None]) +
            (rgb[(y + 1) % height, x % width] * (1 - u[:, None]) + rgb[(y + 1) % height, (x + 1) % width] * u[:, None]) * v[:, None])


def repair(context, source, target, source_image, output_image, missing):
    """Fill only verified misses, sampling the nearest source triangle's real UVs.

    This is local surface transfer, not flat fill or texture synthesis. Refuse
    remote correspondences rather than painting an unrelated anatomical region.
    """
    ids = np.flatnonzero(missing)
    if not len(ids):
        return {'pixels': 0, 'maximum_distance': 0.0}
    source_points, source_triangles, source_vertices, source_uv = surface(context, source)
    points, triangles, vertices, uv = surface(context, target)
    corners = uv[triangles]
    a, b = corners[:, 1] - corners[:, 0], corners[:, 2] - corners[:, 0]
    valid = np.abs(a[:, 0] * b[:, 1] - a[:, 1] * b[:, 0]) > 1e-14
    triangles = triangles[valid]
    uv_points = np.column_stack((uv, np.zeros(len(uv))))
    target_tree = BVHTree.FromPolygons(uv_points.tolist(), triangles.tolist(), all_triangles=True)
    source_tree = BVHTree.FromPolygons(source_points.tolist(), source_vertices[source_triangles].tolist(), all_triangles=True)
    width, height = output_image.size
    coordinates = []
    distances = []
    limit = np.linalg.norm(np.ptp(source_points, axis=0)) * .02
    for index in ids:
        y, x = divmod(int(index), width)
        hit, _, triangle, distance = target_tree.find_nearest(((x + .5) / width, (y + .5) / height, 0))
        if triangle is None or distance > 1 / min(width, height):
            raise ValueError('Missing skin texel is outside the target UV surface')
        loops = triangles[triangle]
        location = barycentric_transform(hit, *[Vector(p) for p in uv_points[loops]], *[Vector(p) for p in points[vertices[loops]]])
        hit, _, triangle, distance = source_tree.find_nearest(location)
        if triangle is None or distance > limit:
            raise ValueError('Skin transfer has a distant source correspondence; inspect alignment: ' + str({'pixel': int(index), 'distance': distance, 'limit': limit, 'point': list(location)}))
        loops = source_triangles[triangle]
        source_uv_points = np.column_stack((source_uv[loops], np.zeros(3)))
        position = barycentric_transform(hit, *[Vector(p) for p in source_points[source_vertices[loops]]], *[Vector(p) for p in source_uv_points])
        coordinates.append(position[:2])
        distances.append(distance)
    colors = sample(source_image, coordinates)
    if output_image.colorspace_settings.name == 'sRGB':
        colors = encoded(colors)
    pixels = np.asarray(output_image.pixels[:], dtype=np.float32).reshape(-1, 4)
    pixels[ids, :3] = colors
    pixels[ids, 3] = 1
    output_image.pixels.foreach_set(pixels.ravel())
    return {'pixels': len(ids), 'maximum_distance': float(max(distances))}
