"""Orthographic photo color projection with surface visibility and UV output."""

import numpy as np


def raster(corners, width, height):
    xy = np.asarray(corners) * [width, height] - .5
    low = np.maximum(np.ceil(xy.min(0)).astype(int), 0)
    high = np.minimum(np.floor(xy.max(0)).astype(int), [width - 1, height - 1])
    if np.any(high < low):
        return None
    a, b = xy[1] - xy[0], xy[2] - xy[0]
    determinant = a[0] * b[1] - a[1] * b[0]
    if abs(determinant) < 1e-12:
        return None
    y, x = np.mgrid[low[1]:high[1] + 1, low[0]:high[0] + 1]
    delta = np.column_stack((x.ravel(), y.ravel())) - xy[0]
    u = (delta[:, 0] * b[1] - delta[:, 1] * b[0]) / determinant
    v = (a[0] * delta[:, 1] - a[1] * delta[:, 0]) / determinant
    keep = (u >= -1e-7) & (v >= -1e-7) & (u + v <= 1 + 1e-7)
    return y.ravel()[keep], x.ravel()[keep], np.column_stack((1 - u[keep] - v[keep], u[keep], v[keep]))


def sample(pixels, uv):
    height, width = pixels.shape[:2]
    xy = np.clip(uv * [width, height] - .5, 0, [width - 1, height - 1])
    lo = np.floor(xy).astype(int)
    hi = np.minimum(lo + 1, [width - 1, height - 1])
    fraction = xy - lo
    u, v = fraction.T
    if pixels.ndim == 3:
        u, v = u[:, None], v[:, None]
    return ((pixels[lo[:, 1], lo[:, 0]] * (1 - u) + pixels[lo[:, 1], hi[:, 0]] * u) * (1 - v)
            + (pixels[hi[:, 1], lo[:, 0]] * (1 - u) + pixels[hi[:, 1], hi[:, 0]] * u) * v)


def project(points, triangles, uv, photo, face_mask, fallback, scale, strength=1):
    """Arrays use bottom-up image rows and camera coordinates looking down -Z."""
    ph, pw = photo.shape[:2]
    height, width = fallback.shape[:2]
    if not 0 < strength <= 1 or not np.isfinite(scale) or scale <= 0:
        raise ValueError('Photo projection needs a positive camera scale and strength')
    if max(width, height, pw, ph) > 2048:
        raise ValueError('Photo color projection supports images up to 2K')
    if not all(np.isfinite(value).all() for value in (points, uv, photo, face_mask, fallback)):
        raise ValueError('Photo projection inputs contain non-finite values')
    screen = points[:, :2] / scale * [max(pw, ph) / pw, max(pw, ph) / ph] + .5
    zbuffer = np.full((ph, pw), -np.inf, dtype=np.float32)
    for ids in triangles:
        pixels = raster(screen[ids], pw, ph)
        if pixels is not None:
            y, x, bary = pixels
            zbuffer[y, x] = np.maximum(zbuffer[y, x], bary @ points[ids, 2])
    influence = np.zeros((height, width), dtype=np.float32)
    output = fallback.copy()
    normals = np.cross(points[triangles[:, 1]] - points[triangles[:, 0]],
                       points[triangles[:, 2]] - points[triangles[:, 0]])
    lengths = np.linalg.norm(normals, axis=1)
    facing = normals[:, 2] / np.maximum(lengths, 1e-15)
    for index in np.flatnonzero(facing > .25):
        ids = triangles[index]
        pixels = raster(uv[index], width, height)
        if pixels is None:
            continue
        y, x, bary = pixels
        projected = bary @ screen[ids]
        inside = ((projected >= 0) & (projected <= 1)).all(1)
        y, x, bary, projected = y[inside], x[inside], bary[inside], projected[inside]
        cells = np.clip((projected * [pw, ph]).astype(int), 0, [pw - 1, ph - 1])
        depth = bary @ points[ids, 2]
        tolerance = scale / max(pw, ph) * 2 / facing[index]
        nearest = zbuffer[cells[:, 1], cells[:, 0]]
        visible = np.isfinite(nearest) & (depth >= nearest - tolerance)
        angle = np.clip((facing[index] - .25) / .4, 0, 1)
        weight = sample(face_mask, projected) * visible * (angle * angle * (3 - 2 * angle)) * strength
        # Keep the strongest visible surface when UV boundaries share texels.
        keep = weight > influence[y, x]
        y, x, projected, weight = y[keep], x[keep], projected[keep], weight[keep]
        output[y, x, :3] = fallback[y, x, :3] * (1 - weight[:, None]) + sample(photo, projected)[:, :3] * weight[:, None]
        influence[y, x] = weight
    if np.count_nonzero(influence > .1) < 100:
        raise ValueError('Photo and accepted face do not overlap enough; inspect their photo comparison')
    return output, influence
