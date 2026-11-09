"""Convex source-outline interpolation without cubic-spline overshoot.

The caller supplies an ordered convex hull and feature rays. Quadratic corner
arcs stay inside that hull, retain its winding, and avoid straight triples at
constrained mesh boundaries. This module needs only NumPy.
"""
import numpy as np


def convex_outline(hull, angles, *, axes=(0, 2), center=None, rounding=.02):
    hull = np.asarray(hull, dtype=float)
    angles = np.asarray(angles, dtype=float)
    if not np.isfinite(rounding) or not 0 < rounding <= 1:
        raise ValueError('Choose outline rounding greater than zero and at most one')
    if hull.ndim != 2 or len(hull) < 3 or not np.isfinite(hull).all() or angles.ndim != 1 or not np.isfinite(angles).all():
        raise ValueError('Choose a finite ordered convex outline and feature angles')
    if len(axes) != 2 or len(set(axes)) != 2 or min(axes) < 0 or max(axes) >= hull.shape[1]:
        raise ValueError('Choose two distinct outline axes')
    projected = hull[:, axes]
    edges = np.roll(projected, -1, axis=0) - projected
    turns = edges[:, 0] * np.roll(edges[:, 1], -1) - edges[:, 1] * np.roll(edges[:, 0], -1)
    if np.all(turns < 0):
        hull = hull[::-1]
    elif not np.all(turns > 0):
        raise ValueError('Outline vertices must form a strictly convex ordered hull')
    center = hull.mean(0) if center is None else np.asarray(center, dtype=float)
    if center.shape != (hull.shape[1],) or not np.isfinite(center).all():
        raise ValueError('Choose a finite outline center')
    edge = np.roll(hull[:, axes], -1, axis=0)-hull[:, axes]
    inward = center[list(axes)]-hull[:, axes]
    if np.any(edge[:, 0]*inward[:, 1]-edge[:, 1]*inward[:, 0] <= 0):
        raise ValueError('The outline center must be strictly inside its convex hull')
    start = (np.roll(hull, 1, axis=0) + hull) * .5
    end = (hull + np.roll(hull, -1, axis=0)) * .5
    A = start - 2 * hull + end
    B = 2 * (hull - start)
    C = start - center
    scale = max(float(np.linalg.norm(np.ptp(hull[:, axes], axis=0))), 1e-30)
    result = []
    for angle in angles:
        ray = np.array([np.cos(angle), np.sin(angle)])
        def cross(value):
            q = value[:, axes]
            return q[:, 0] * ray[1] - q[:, 1] * ray[0]
        aa, bb, cc = cross(A), cross(B), cross(C)
        candidates = []
        for i in range(len(hull)):
            if abs(aa[i]) <= scale * 1e-13:
                roots = [] if abs(bb[i]) <= scale * 1e-13 else [-cc[i] / bb[i]]
            else:
                discriminant = bb[i] ** 2 - 4 * aa[i] * cc[i]
                if discriminant < -scale ** 2 * 1e-13:
                    continue
                root = np.sqrt(max(discriminant, 0.))
                # Stable quadratic formula avoids cancellation near arc ends.
                q = -.5 * (bb[i] + np.copysign(root, bb[i]))
                roots = [-bb[i] / (2 * aa[i])] if q == 0 else [q / aa[i], cc[i] / q]
            for t in roots:
                if -1e-10 <= t <= 1 + 1e-10:
                    t = float(np.clip(t, 0, 1))
                    point = start[i] + B[i] * t + A[i] * t * t
                    radius = float((point - center)[list(axes)] @ ray)
                    if radius > 0:
                        candidates.append((radius, point))
        if not candidates:
            raise ValueError('The feature ray does not intersect the rounded convex outline')
        rounded_radius, rounded = min(candidates, key=lambda item: item[0])
        polygon_hits = []
        polygon_delta = hull - center
        polygon_edges = np.roll(hull, -1, axis=0) - hull
        numerator, denominator = cross(polygon_delta), cross(polygon_edges)
        for i in range(len(hull)):
            if abs(denominator[i]) <= scale * 1e-13:
                continue
            t = -numerator[i] / denominator[i]
            if -1e-10 <= t <= 1 + 1e-10:
                point = hull[i] + np.clip(t, 0, 1) * polygon_edges[i]
                radius = float((point-center)[list(axes)] @ ray)
                if radius > 0:
                    polygon_hits.append((radius, point))
        if not polygon_hits:
            raise ValueError('The feature ray does not intersect the source hull')
        polygon_radius, polygon = min(polygon_hits, key=lambda item: item[0])
        # Interpolate reciprocal radii (a convex sum in the polar domain),
        # retaining a slight curvature on otherwise straight source edges.
        radius = 1 / ((1-rounding)/polygon_radius + rounding/rounded_radius)
        gap = polygon_radius-rounded_radius
        blend = 0. if abs(gap) < scale*1e-14 else (polygon_radius-radius)/gap
        result.append(polygon+(rounded-polygon)*blend)
    return np.asarray(result).reshape(len(angles), hull.shape[1])
