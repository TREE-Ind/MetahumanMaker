"""Local, detail-preserving neck albedo matching on aligned head/body UVs."""
import numpy as np
from photo_color_core import raster


def skin_sample(pixels, uv):
    """Match Blender's repeating skin textures, including native body tile U=1."""
    height, width = pixels.shape[:2]
    xy = uv * [width, height] - .5
    low = np.floor(xy).astype(np.int64)
    u, v = (xy - low).T
    x, y = low.T
    return ((pixels[y % height, x % width] * (1 - u[:, None]) + pixels[y % height, (x + 1) % width] * u[:, None]) * (1 - v[:, None])
            + (pixels[(y + 1) % height, x % width] * (1 - u[:, None]) + pixels[(y + 1) % height, (x + 1) % width] * u[:, None]) * v[:, None])


def seam_colors(pixels, uv, ids, count, smooth=True):
    from scipy.ndimage import gaussian_filter
    # Estimate the underlying skin color rather than matching individual pores.
    blurred = gaussian_filter(pixels[..., :3], (max(pixels.shape[:2]) / 512,) * 2 + (0,)) if smooth else pixels[..., :3]
    colors = skin_sample(blurred, uv)
    return np.asarray([np.median(colors[ids == index], axis=0) for index in range(count)])


def blend(head, body, points, triangles, uv, seam, head_uv, head_ids, body_uv, body_ids,
          protection=None, band_fraction=.65):
    """Linear RGB, bottom-up rows. Return a new head; body and inputs stay intact."""
    from scipy.spatial import cKDTree
    from scipy.ndimage import distance_transform_edt, gaussian_filter

    if not .05 <= band_fraction <= .7 or len(seam) < 3:
        raise ValueError('Neck color needs a valid seam and a bounded blend band')
    if not all(np.isfinite(a).all() for a in (head, body, points, uv, seam, head_uv, body_uv)):
        raise ValueError('Neck color inputs contain non-finite values')
    count = len(seam)
    a = seam_colors(head, head_uv, head_ids, count)
    b = seam_colors(body, body_uv, body_ids, count)
    if not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError('Every neck point needs UV samples on both skin parts')
    gains = np.log(np.clip(b / np.maximum(a, .005), .2, 5))
    distance, nearest = cKDTree(seam).query(points, k=min(4, count))
    inverse = 1 / np.maximum(distance, 1e-8) ** 2
    target = (b[nearest] * inverse[..., None]).sum(1) / inverse.sum(1)[:, None]
    # A seam-only gain cannot repair a different tone farther up the neck.
    # Remove that broad color variation locally while retaining fine detail.
    blurred = gaussian_filter(head[..., :3], (max(head.shape[:2]) / 512,) * 2 + (0,))
    colors = skin_sample(blurred, uv.reshape(-1, 2))
    native = np.zeros((len(points), 3), dtype=np.float64)
    counts = np.zeros(len(points), dtype=np.float64)
    np.add.at(native, triangles.ravel(), colors)
    np.add.at(counts, triangles.ravel(), 1)
    native /= np.maximum(counts[:, None], 1)
    vertex_gain = np.log(np.clip(target / np.maximum(native, .005), .2, 5))
    band = max(float(np.ptp(points[:, 2])) * band_fraction, 1e-8)
    weight = 1 - np.clip((distance[:, 0] / band - .5) * 2, 0, 1)
    weight = weight * weight * (3 - 2 * weight)
    height, width = head.shape[:2]
    field = np.zeros((height, width, 3), dtype=np.float32)
    coverage = np.zeros((height, width), dtype=np.float32)
    for index in np.flatnonzero(weight[triangles].max(1) > 0):
        ids = triangles[index]
        cells = raster(uv[index], width, height)
        if cells is None:
            continue
        y, x, bary = cells
        amount = bary @ weight[ids]
        keep = amount > coverage[y, x]
        y, x, bary, amount = y[keep], x[keep], bary[keep], amount[keep]
        coverage[y, x] = amount
        field[y, x] = bary @ (vertex_gain[ids] * weight[ids, None])
    affected = coverage > 0
    if not affected.any():
        raise ValueError('The neck blend does not cover any head UV texels')
    # Bilinear sampling at the mesh boundary needs a small texture gutter too.
    distance, nearest = distance_transform_edt(~affected, return_indices=True)
    gutter = (~affected) & (distance <= 2)
    field[gutter] = field[nearest[0][gutter], nearest[1][gutter]]
    if protection is not None:
        field *= (1 - np.clip(protection, 0, 1))[..., None]
    output = head.copy()
    output[..., :3] *= np.exp(field)
    before = seam_colors(head, head_uv, head_ids, count, smooth=False)
    target = seam_colors(body, body_uv, body_ids, count, smooth=False)
    after = seam_colors(output, head_uv, head_ids, count, smooth=False)
    return output, {'status': 'complete', 'method': 'LOCAL_NECK_TONE', 'seam_points': count, 'band_height': band,
                    'texels': int(np.count_nonzero(np.any(field != 0, axis=2))),
                    'median_gain': np.exp(np.median(gains, axis=0)).tolist(),
                    'linear_seam_mae_before': float(np.abs(before - target).mean()),
                    'linear_seam_mae_after': float(np.abs(after - target).mean())}
