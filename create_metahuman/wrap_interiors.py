"""Preserve semantically identified oral cavities during outer-surface fitting."""
import numpy as np


def mouth_mask(obj):
    """Identify separate mouth UV islands from their skinning semantics.

    No vertex numbers, UV positions or character-specific bounds are assumed.
    Unknown templates return an empty mask. Physical boundary copies remain
    exterior constraints, so the original UV/material seams stay joined.
    """
    mesh = obj.data
    mask = np.zeros(len(mesh.vertices), dtype=bool)
    mouth_groups = {group.index for group in obj.vertex_groups
                    if 'mouthinterior' in group.name.lower()}
    if not mouth_groups or not mesh.uv_layers:
        return mask
    uv = np.empty((len(mesh.loops), 2), dtype=np.float64)
    mesh.uv_layers.active.data.foreach_get('uv', uv.ravel())
    keys = np.rint(uv * 1e7).astype(np.int64)
    parent = np.arange(len(mesh.polygons))

    def find(index):
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    seen = {}
    for polygon in mesh.polygons:
        points = [tuple(keys[index]) for index in polygon.loop_indices]
        for a, b in zip(points, points[1:] + points[:1]):
            edge = tuple(sorted((a, b)))
            if edge in seen:
                parent[find(polygon.index)] = find(seen[edge])
            else:
                seen[edge] = polygon.index
    islands = {}
    for polygon in mesh.polygons:
        islands.setdefault(find(polygon.index), []).append(polygon)
    interior_faces = set()
    for polygons in islands.values():
        if len(polygons) >= len(mesh.polygons) * .5:
            continue
        vertices = {index for polygon in polygons for index in polygon.vertices}
        weights = {}
        for index in vertices:
            for item in mesh.vertices[index].groups:
                weights[item.group] = weights.get(item.group, 0.) + item.weight
        dominant = sorted(weights, key=weights.get, reverse=True)[:3]
        if mouth_groups.intersection(dominant):
            interior_faces.update(polygon.index for polygon in polygons)
            mask[list(vertices)] = True
    if not interior_faces:
        return mask
    external = np.zeros(len(mesh.vertices), dtype=bool)
    for polygon in mesh.polygons:
        if polygon.index not in interior_faces:
            external[list(polygon.vertices)] = True
    coordinates = np.empty((len(mesh.vertices), 3), dtype=np.float64)
    mesh.vertices.foreach_get('co', coordinates.ravel())
    tolerance = max(float(np.linalg.norm(np.ptp(coordinates, axis=0))) * 1e-7, 1e-10)
    _, families = np.unique(np.rint(coordinates / tolerance).astype(np.int64), axis=0, return_inverse=True)
    boundary = np.zeros(int(families.max()) + 1, dtype=bool)
    np.logical_or.at(boundary, families, external)
    mask[boundary[families]] = False
    return mask


def follow_boundary(reference, fitted, edges, interior, *, tolerance=1e-8, iterations=1024, weld=None):
    """Extend boundary displacement into the cavity, retaining reference depth."""
    edges = np.asarray(edges, dtype=np.int64)
    if weld is not None:
        weld=np.asarray(weld,dtype=np.int64)
        if weld.shape!=(len(reference),) or np.any(weld<0):
            raise ValueError('Interior physical families must match reference vertices')
        _,families,counts=np.unique(weld,return_inverse=True,return_counts=True)
        def physical(points):
            return np.column_stack([np.bincount(families,weights=points[:,axis],minlength=len(counts))/counts for axis in range(3)])
        # A family touching the exterior is a fixed boundary, even if one UV
        # copy was classified as interior. Relax a single physical graph so
        # separate UV charts cannot invent different cavity displacements.
        free=np.ones(len(counts),dtype=bool);np.logical_and.at(free,families,interior)
        physical_edges=np.sort(families[edges],axis=1)
        physical_edges=np.unique(physical_edges[physical_edges[:,0]!=physical_edges[:,1]],axis=0)
        solved=follow_boundary(physical(reference),physical(fitted),physical_edges,free,
                               tolerance=tolerance,iterations=iterations)
        result=fitted.copy();selected=free[families]
        result[selected]=solved[families[selected]]
        return result
    active = edges[np.any(interior[edges], axis=1)]
    if not len(active):
        return fitted
    nodes, inverse = np.unique(active, return_inverse=True)
    local_edges = inverse.reshape(-1, 2)
    free = interior[nodes]
    delta = fitted[nodes] - reference[nodes]
    a, b = local_edges.T
    degree = np.bincount(np.concatenate((a, b)), minlength=len(nodes))
    for _ in range(iterations):
        average = np.column_stack([
            (np.bincount(a, weights=delta[b, axis], minlength=len(nodes))
             + np.bincount(b, weights=delta[a, axis], minlength=len(nodes)))
            / np.maximum(degree, 1) for axis in range(3)])
        change = float(np.max(np.abs(average[free] - delta[free]), initial=0))
        delta[free] = average[free]
        if change <= tolerance:
            break
    result = fitted.copy()
    result[nodes[free]] = reference[nodes[free]] + delta[free]
    return result
