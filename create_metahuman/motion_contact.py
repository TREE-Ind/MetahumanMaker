"""Smooth anatomical contact constraints for sampled facial motion.

The neutral surface stays authoritative. Callers supply corresponding upper /
lower lid vertices and closure fractions derived from a healthy motion source.
No texture, topology or rig changes are made here; this is a pose-space repair.
Only NumPy is required, so the same solver works inside Blender.
"""

import numpy as np


def eyelid_contact(neutral, posed, pairs, desired_vectors, center, radius,
                  boundary=None, smoothing=1e-3):
    """Fit bounded smooth offsets to disjoint lid-pair contact constraints.

    All coordinates share one frame and unit. ``desired_vectors`` are upper
    minus lower at this pose, including anatomical thickness. Use disjoint
    pairs: repeated endpoints would impose contradictory position constraints.
    Returns a new surface and diagnostics, leaving input arrays untouched.
    """
    neutral = np.asarray(neutral, dtype=float)
    posed = np.asarray(posed, dtype=float)
    pairs = np.asarray(pairs, dtype=int)
    desired = np.asarray(desired_vectors, dtype=float)
    center = np.asarray(center, dtype=float)
    if neutral.shape != posed.shape or neutral.ndim != 2 or neutral.shape[1] != 3:
        raise ValueError('Contact surfaces must have matching Nx3 positions')
    if pairs.ndim != 2 or pairs.shape[1] != 2 or not len(pairs):
        raise ValueError('Contact pairs must be a nonempty Nx2 array')
    if len(np.unique(pairs)) != pairs.size or pairs.min() < 0 or pairs.max() >= len(neutral):
        raise ValueError('Contact endpoints must be distinct valid vertices')
    if desired.shape != (len(pairs), 3) or not np.isfinite(radius) or radius <= 0:
        raise ValueError('Contact vectors / anatomical radius are invalid')
    if not all(np.isfinite(a).all() for a in (neutral, posed, desired, center)):
        raise ValueError('Non-finite contact geometry')
    a, b = pairs.T
    error = desired - (posed[a] - posed[b])
    indices = np.concatenate((a, b))
    anchors = neutral[indices]
    offsets = np.concatenate((error * .5, -error * .5))
    if boundary is not None:
        boundary = np.asarray(boundary, dtype=int)
        boundary = np.setdiff1d(boundary, indices)
        anchors = np.concatenate((anchors, neutral[boundary]))
        offsets = np.concatenate((offsets, np.zeros((len(boundary), 3))))
    anchors = (anchors - center) / radius
    if np.linalg.norm(anchors[:len(indices)], axis=1).max() > 2.3:
        raise ValueError('Contact endpoints lie outside the chosen anatomical region')
    def kernel(distance):
        scaled = distance / .6
        return np.maximum(0, 1 - scaled) ** 4 * (4 * scaled + 1)
    distances = np.linalg.norm(anchors[:, None] - anchors[None, :], axis=-1)
    matrix = kernel(distances)
    matrix.flat[::len(matrix) + 1] += smoothing
    coefficients = np.linalg.solve(matrix, offsets)
    result = posed.copy()
    region = np.flatnonzero(np.linalg.norm(neutral - center, axis=1) < radius * 2.8)
    for first in range(0, len(region), 2048):
        selected = region[first:first + 2048]
        points = (neutral[selected] - center) / radius
        distance = np.linalg.norm(points[:, None] - anchors[None, :], axis=-1)
        delta = kernel(distance) @ coefficients
        limit = min(radius * .15, np.linalg.norm(offsets, axis=1).max() * 2)
        delta *= np.minimum(1, limit / np.maximum(np.linalg.norm(delta, axis=1), 1e-15))[:, None]
        # Compact C1 falloff keeps unrelated face regions exactly unchanged.
        t = np.clip((2.8 - np.linalg.norm(points, axis=1)) / .5, 0, 1)
        result[selected] += delta * (t * t * (3 - 2 * t))[:, None]
    residual = (result[a] - result[b]) - desired
    return result, {
        'pairs': len(pairs),
        'max_offset': float(np.linalg.norm(result - posed, axis=1).max()),
        'pair_error_before_p95': float(np.quantile(np.linalg.norm(error, axis=1), .95)),
        'pair_error_after_p95': float(np.quantile(np.linalg.norm(residual, axis=1), .95)),
    }


def eye_clearance(posed, eyeball, amount=1, front_axis=(0, 1, 0), thickness=.01):
    """Project penetrated anterior lid points outside a fitted spherical globe.

    Uses actual eye geometry, not section names. ``thickness`` shares the input
    unit. Partial poses interpolate the correction; neutral callers use zero.
    This protects contact poses, without altering eye geometry or UVs.
    """
    posed=np.asarray(posed,dtype=float);eyeball=np.asarray(eyeball,dtype=float)
    front=np.asarray(front_axis,dtype=float)
    if not 0<=amount<=1 or thickness<0 or front.shape!=(3,) or np.linalg.norm(front)<1e-8:
        raise ValueError('Invalid eye clearance settings')
    front=front/np.linalg.norm(front)
    # Least-squares sphere fit is independent of uneven UV sampling density.
    origin=eyeball.mean(0);local=eyeball-origin
    fit=np.linalg.lstsq(np.column_stack((2*local,np.ones(len(local)))),(local*local).sum(1),rcond=None)[0]
    center=origin+fit[:3];radius=float(np.sqrt(fit[3]+fit[:3]@fit[:3]))
    relative=posed-center;depth=relative@front
    tangent=relative-depth[:,None]*front
    tangent2=(tangent*tangent).sum(1)
    surface=np.sqrt(np.maximum(0,(radius+thickness)**2-tangent2))
    mask=(tangent2<radius**2)&(depth>radius*.1)&(depth<surface)
    result=posed.copy();offset=(surface[mask]-depth[mask])*amount
    result[mask]+=offset[:,None]*front
    return result,{'projected_vertices':int(mask.sum()),'max_offset':float(offset.max(initial=0)),'radius':radius}
