"""Recover subdivided vertex order from anchored connectivity, then prove faces."""

import numpy as np


def from_mesh(mesh):
    edges = np.empty((len(mesh.edges), 2), dtype=np.int64)
    loops = np.empty(len(mesh.loops), dtype=np.int64)
    starts = np.empty(len(mesh.polygons), dtype=np.int64)
    sizes = np.empty(len(mesh.polygons), dtype=np.int64)
    mesh.edges.foreach_get("vertices", edges.ravel())
    mesh.loops.foreach_get("vertex_index", loops)
    mesh.polygons.foreach_get("loop_start", starts)
    mesh.polygons.foreach_get("loop_total", sizes)
    return {
        "count": len(mesh.vertices),
        "edges": edges,
        "loops": loops,
        "starts": starts,
        "sizes": sizes,
    }


def _mix(values):
    values = np.asarray(values, dtype=np.uint64).copy()
    values ^= values >> np.uint64(30)
    values *= np.uint64(0xBF58476D1CE4E5B9)
    values ^= values >> np.uint64(27)
    values *= np.uint64(0x94D049BB133111EB)
    return values ^ (values >> np.uint64(31))


def _edge_keys(data, mapping=None):
    edges = data["edges"] if mapping is None else mapping[data["edges"]]
    edges = np.sort(edges, axis=1).astype(np.uint64)
    return np.sort(edges[:, 0] * np.uint64(data["count"]) + edges[:, 1])


def _faces(data, mapping=None):
    result = {}
    for size in np.unique(data["sizes"]):
        starts = data["starts"][data["sizes"] == size]
        values = data["loops"][starts[:, None] + np.arange(size)]
        if mapping is not None:
            values = mapping[values]
        # Rotations describe the same oriented polygon; reversal does not.
        first = np.argmin(values, axis=1)
        columns = (first[:, None] + np.arange(size)) % size
        values = np.take_along_axis(values, columns, axis=1)
        rows = (
            np.ascontiguousarray(values, dtype="<i8")
            .view(np.dtype((np.void, 8 * int(size))))
            .ravel()
        )
        result[int(size)] = np.sort(rows)
    return result


def verify(evaluated, target, mapping):
    count = evaluated["count"]
    if (
        target["count"] != count
        or mapping.shape != (count,)
        or not np.array_equal(np.sort(mapping), np.arange(count))
    ):
        return False
    if not np.array_equal(_edge_keys(evaluated), _edge_keys(target, mapping)):
        return False
    a, b = _faces(evaluated), _faces(target, mapping)
    return a.keys() == b.keys() and all(np.array_equal(a[size], b[size]) for size in a)


def correspondence(evaluated, target, anchors, rounds=32):
    """Return evaluated indices in target order; coarse vertex indices are anchors.

    Blender retains the input vertices at the beginning of a subdivision mesh.
    All of those vertices are pinned as distinct anchors. Hash refinement only
    proposes a correspondence; exact edge and oriented-face checks prove it.
    Ambiguous/disconnected or changed meshes are refused, never guessed.
    """
    count = evaluated["count"]
    if target["count"] != count or not 0 < anchors <= count:
        raise ValueError("Subdivision vertex count changed; sculpt retained")
    identity = np.arange(count, dtype=np.int64)
    if verify(evaluated, target, identity):
        return identity
    fixed = _mix(np.arange(1, anchors + 1, dtype=np.uint64))
    labels = [np.full(count, np.uint64(0xD6E8FEB86659FD93)) for _ in range(2)]
    for values in labels:
        values[:anchors] = fixed
    adjacency = []
    for data in (evaluated, target):
        edges = data["edges"]
        indices = np.concatenate((edges[:, 0], edges[:, 1]))
        neighbors = np.concatenate((edges[:, 1], edges[:, 0]))
        adjacency.append((indices, neighbors))
    for _ in range(rounds):
        for index, (indices, neighbors) in enumerate(adjacency):
            sums = np.zeros(count, dtype=np.uint64)
            np.add.at(sums, indices, labels[index][neighbors])
            labels[index] = _mix(labels[index] ^ _mix(sums))
            labels[index][:anchors] = fixed
        order_a, order_b = [np.argsort(values) for values in labels]
        a, b = labels[0][order_a], labels[1][order_b]
        if np.any(a[1:] == a[:-1]) or np.any(b[1:] == b[:-1]):
            continue
        if not np.array_equal(a, b):
            raise ValueError(
                "Subdivision connectivity differs from the sculpt; work retained"
            )
        mapping = np.empty(count, dtype=np.int64)
        mapping[order_b] = order_a
        if np.array_equal(mapping[:anchors], np.arange(anchors)) and verify(
            evaluated, target, mapping
        ):
            return mapping
        raise ValueError(
            "Subdivision correspondence failed face verification; work retained"
        )
    raise ValueError("Subdivision correspondence is ambiguous; work retained")
