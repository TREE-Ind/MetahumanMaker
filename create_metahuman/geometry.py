"""Independent registration math. NumPy only; no proprietary models or solver."""

import hashlib
import numpy as np


def points(value):
    result = np.asarray(value, dtype=np.float64)
    if result.ndim != 2 or result.shape[1] != 3 or not len(result):
        raise ValueError("Expected a nonempty N x 3 point array")
    if not np.isfinite(result).all():
        raise ValueError("Geometry contains non-finite coordinates")
    return result


def scatter_displacement(canonical, physical_before, physical_after, mapping, *, locked=None):
    """Scatter a physical repair without averaging existing canonical seams.

    A locked canonical copy locks its entire physical family. This preserves
    artist offsets and exact fixed coordinates, rather than replacing seam
    copies with their mean. Existing separation is retained, not repaired.
    """
    canonical=points(canonical);before=points(physical_before);after=points(physical_after)
    mapping=np.asarray(mapping,dtype=np.int64)
    if before.shape!=after.shape or mapping.shape!=(len(canonical),) or mapping.min()<0 or mapping.max()>=len(before):
        raise ValueError('Physical displacement mapping must match canonical vertices')
    displacement=after-before
    if locked is not None:
        locked=np.asarray(locked,dtype=bool)
        if locked.shape!=(len(canonical),):raise ValueError('Locks must match canonical vertices')
        displacement=displacement.copy();displacement[np.unique(mapping[locked])]=0
    return canonical+displacement[mapping]


class SurfaceStepGuard:
    """Backtrack conservative projection locally before a triangle collapses.

    This checks each step against its preceding surface normal and retains a
    small area floor from the initial fit. It is not a self-intersection test.
    """
    def __init__(self, reference, triangles, weld=None, minimum_double_area=0.):
        self.triangles = np.asarray(triangles, dtype=np.int32).reshape(-1, 3)
        self.weld = weld
        self.minimum_double_area = float(minimum_double_area)
        normals = self.normals(reference)
        self.area = np.linalg.norm(normals, axis=1)
        self.active = self.area > max(float(self.area.max(initial=0)) * 1e-12, 1e-30)
        self.limited_steps = 0
        self.limited_vertices = 0

    def normals(self, vertices):
        values = vertices[self.triangles]
        return np.cross(values[:, 1] - values[:, 0], values[:, 2] - values[:, 0])

    def limit(self, previous, proposed):
        if not np.isfinite(proposed).all():
            raise ValueError('Conservative fitting produced non-finite geometry; previous character retained')
        normals = self.normals(previous)
        area = np.linalg.norm(normals, axis=1)
        threshold = np.maximum(np.maximum(self.area * .01, area * .02), self.minimum_double_area) * area

        def bad(candidate):
            projected = np.einsum('ij,ij->i', self.normals(candidate), normals)
            return self.active & (projected < threshold)

        direction = proposed - previous
        weight = np.ones(len(previous))
        candidate = proposed
        for _ in range(8):
            invalid = bad(candidate)
            if not invalid.any():
                break
            weight[np.unique(self.triangles[invalid])] *= .5
            if self.weld is not None:
                shared = np.ones(int(self.weld.max()) + 1)
                np.minimum.at(shared, self.weld, weight)
                weight = shared[self.weld]
            candidate = previous + direction * weight[:, None]
        else:
            # An extremely thin triangle can require an impractically small
            # step. Keep its local vertices at their preceding positions and
            # expand that patch only where its boundary still changes too far.
            for _ in range(32):
                invalid = bad(candidate)
                if not invalid.any():
                    break
                weight[np.unique(self.triangles[invalid])] = 0
                if self.weld is not None:
                    shared = np.ones(int(self.weld.max()) + 1)
                    np.minimum.at(shared, self.weld, weight)
                    weight = shared[self.weld]
                candidate = previous + direction * weight[:, None]
            # Any remaining boundary is handled together. The rest of the
            # mesh retains its full displacement whenever local repair passes.
            for _ in range(32):
                if not bad(candidate).any():
                    break
                weight *= .5
                candidate = previous + direction * weight[:, None]
            else:
                candidate = previous.copy()
                weight[:] = 0
        count = int(np.count_nonzero(weight < 1))
        self.limited_steps += bool(count)
        self.limited_vertices += count
        return candidate


def similarity(source, target):
    """Umeyama similarity without reflection; row-vector convention x @ R * s + t."""
    source, target = points(source), points(target)
    if source.shape != target.shape or len(source) < 3:
        raise ValueError("Use at least three matching landmark pairs")
    a, b = source - source.mean(0), target - target.mean(0)
    if np.linalg.matrix_rank(a, tol=1e-9) < 2 or np.linalg.matrix_rank(b, tol=1e-9) < 2:
        raise ValueError("Landmarks must be spread out, not on one line")
    u, singular, vt = np.linalg.svd(a.T @ b)
    sign = np.ones(3)
    sign[-1] = np.sign(np.linalg.det(u @ vt))
    rotation = (u * sign) @ vt
    scale = (singular * sign).sum() / (a * a).sum()
    translation = target.mean(0) - source.mean(0) @ rotation * scale
    return rotation, float(scale), translation


def rbf_displacement(vertices, controls, destinations, radius, regularization=1e-6):
    """Smooth Gaussian residual field, evaluated in bounded-memory batches."""
    vertices, controls, destinations = (
        points(vertices),
        points(controls),
        points(destinations),
    )
    if controls.shape != destinations.shape:
        raise ValueError("Landmark arrays do not match")
    if radius <= 0:
        raise ValueError("Landmark radius must be positive")
    squared = np.sum((controls[:, None] - controls[None, :]) ** 2, axis=2)
    if len(controls) > 1 and np.any(
        squared[np.triu_indices(len(controls), 1)] < radius**2 * 1e-12
    ):
        raise ValueError("Duplicate template landmarks; remove or move the duplicates")
    kernel = np.exp(-squared / (2 * radius**2))
    weights = np.linalg.solve(
        kernel + np.eye(len(kernel)) * regularization, destinations - controls
    )
    result = vertices.copy()
    for start in range(0, len(vertices), 4096):
        p = vertices[start : start + 4096]
        distance = np.sum((p[:, None] - controls[None, :]) ** 2, axis=2)
        result[start : start + len(p)] += np.exp(-distance / (2 * radius**2)) @ weights
    return result


def limit_extrapolation(vertices, deformed, controls, destinations):
    """Bound unsupported RBF overshoot without changing constrained magnitudes."""
    vertices, deformed = points(vertices), points(deformed)
    maximum = float(np.linalg.norm(points(destinations) - points(controls), axis=1).max())
    if maximum < 1e-12:
        return vertices.copy()
    delta = deformed - vertices
    magnitude = np.linalg.norm(delta, axis=1)
    limited = np.where(magnitude <= maximum, magnitude,
        maximum * (1 + np.tanh((magnitude - maximum) / maximum)))
    return vertices + delta * (limited / np.maximum(magnitude, 1e-30))[:,None]


def split_vertex_error(reference, current):
    """Largest displacement from a coincident rest-vertex group's mean."""
    reference, current = points(reference), points(current)
    if reference.shape != current.shape:
        raise ValueError('Split continuity needs matching vertex counts')
    _, inverse, counts = np.unique(reference,axis=0,return_inverse=True,return_counts=True)
    if len(counts)==len(reference):
        return 0.
    centers=np.column_stack([np.bincount(inverse,weights=current[:,axis],minlength=len(counts))/counts for axis in range(3)])
    return float(np.linalg.norm(current-centers[inverse],axis=1).max())


def neighbor_average(values, edges):
    values = np.asarray(values, dtype=np.float64)
    edges = np.asarray(edges, dtype=np.int64).reshape(-1, 2)
    total = np.zeros_like(values)
    count = np.zeros(len(values))
    if len(edges):
        a, b = edges.T
        np.add.at(total, a, values[b])
        np.add.at(total, b, values[a])
        np.add.at(count, a, 1)
        np.add.at(count, b, 1)
    isolated = count == 0
    count[isolated] = 1
    total[isolated] = values[isolated]
    return total / count[:, None]


def relax_displacement(current, reference, edges, strength):
    """Regularize displacement, retaining the reference mesh's local detail."""
    delta = current - reference
    return (
        reference + (1 - strength) * delta + strength * neighbor_average(delta, edges)
    )


def topology_hash(vertex_count, faces):
    digest = hashlib.sha256(str(vertex_count).encode())
    for face in faces:
        face = np.asarray(face, dtype="<i8")
        digest.update(np.asarray([len(face)], dtype="<i8").tobytes())
        digest.update(face.tobytes())
    return digest.hexdigest()


def topology_arrays_hash(vertex_count, sizes, vertices, starts):
    """Hash polygon buffers with the exact byte format used by topology_hash."""
    sizes = np.asarray(sizes, dtype=np.int64)
    starts = np.asarray(starts, dtype=np.int64)
    vertices = np.asarray(vertices, dtype=np.int64)
    if (
        sizes.shape != starts.shape
        or np.any(sizes < 0)
        or np.any(starts < 0)
        or np.any(starts + sizes > len(vertices))
    ):
        raise ValueError("Invalid polygon buffers")
    offsets = np.cumsum(sizes) - sizes
    count = int(sizes.sum())
    payload = np.empty(count + len(sizes), dtype="<i8")
    payload[offsets + np.arange(len(sizes))] = sizes
    loop_order = np.arange(count) + np.repeat(starts - offsets, sizes)
    positions = np.arange(count) + np.repeat(np.arange(len(sizes)) + 1, sizes)
    payload[positions] = vertices[loop_order]
    digest = hashlib.sha256(str(vertex_count).encode())
    digest.update(payload.tobytes())
    return digest.hexdigest()


def welded_boundary_loops(vertices, faces, tolerance=None):
    """Find physical open loops while ignoring FBX/UV duplicate vertices.

    Return representative vertex indices plus a weld index for every original
    vertex. This does not alter geometry, UVs, vertex order or connectivity.
    """
    vertices = points(vertices)
    tolerance = tolerance or max(np.linalg.norm(np.ptp(vertices, axis=0)) * 1e-6, 1e-9)
    _, representatives, welded = np.unique(
        np.rint(vertices / tolerance).astype(np.int64),
        axis=0,
        return_index=True,
        return_inverse=True,
    )
    edges = []
    for face in faces:
        face = welded[np.asarray(face, dtype=int)]
        for a, b in zip(face, np.roll(face, -1)):
            if a != b:
                edges.append((min(a, b), max(a, b)))
    if not edges:
        return [], welded
    unique, counts = np.unique(np.asarray(edges), axis=0, return_counts=True)
    adjacent = {}
    for a, b in unique[counts == 1]:
        adjacent.setdefault(int(a), set()).add(int(b))
        adjacent.setdefault(int(b), set()).add(int(a))
    loops = []
    pending = set(adjacent)
    while pending:
        seed = min(pending)
        component, stack = set(), [seed]
        while stack:
            vertex = stack.pop()
            if vertex in component:
                continue
            component.add(vertex)
            stack.extend(adjacent[vertex] - component)
        pending -= component
        if any(len(adjacent[v]) != 2 for v in component):
            continue  # An ambiguous/open chain is not a trustworthy neck ring.
        ordered, previous, current = [], None, seed
        while current not in ordered:
            ordered.append(current)
            following = sorted(adjacent[current] - {previous})
            previous, current = current, following[0]
        if current == seed and len(ordered) >= 3:
            loops.append(representatives[np.asarray(ordered)])
    return loops, welded


def identity_wheel_weights(point, count=12):
    """Continuous, normalized inverse-distance blend on an identity wheel."""
    point = np.asarray(point, dtype=float)
    if point.shape != (2,) or not np.isfinite(point).all() or count < 3:
        raise ValueError("Expected a finite 2D point and at least three identities")
    point = point / max(np.linalg.norm(point), 1)
    angle = np.arange(count) * 2 * np.pi / count
    anchors = np.column_stack([np.cos(angle), np.sin(angle)])
    distance = ((anchors - point) ** 2).sum(1)
    if distance.min() < 1e-12:
        weights = np.zeros(count)
        weights[np.argmin(distance)] = 1
        return weights
    weights = 1 / distance
    return weights / weights.sum()


def shape_body(
    vertices, height=1.0, width=1.0, depth=1.0, waist=1.0, shoulders=1.0, hips=1.0
):
    """Geometric proportions in Blender Z-up coordinates, without a learned body model."""
    v = points(vertices).copy()
    lo, hi = v.min(0), v.max(0)
    span = max(hi[2] - lo[2], 1e-9)
    z = (v[:, 2] - lo[2]) / span
    center = (lo + hi) * 0.5
    factors = np.ones(len(v))
    for amount, level, spread in (
        (waist, 0.55, 0.09),
        (shoulders, 0.82, 0.08),
        (hips, 0.43, 0.08),
    ):
        factors *= 1 + (amount - 1) * np.exp(-0.5 * ((z - level) / spread) ** 2)
    v[:, 0] = center[0] + (v[:, 0] - center[0]) * width * factors
    v[:, 1] = center[1] + (v[:, 1] - center[1]) * depth * factors
    v[:, 2] = lo[2] + (v[:, 2] - lo[2]) * height
    return v
