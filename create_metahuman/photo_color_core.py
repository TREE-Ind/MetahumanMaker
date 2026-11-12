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


def feature_alignment(projected, observed, indices=None):
    """Map projected facial features onto their observed photograph positions."""
    from scipy.interpolate import RBFInterpolator

    projected, observed = np.asarray(projected), np.asarray(observed)
    if projected.shape != (68, 2) or observed.shape != (68, 2):
        raise ValueError('Feature alignment needs corresponding face landmarks')
    ids = np.arange(17, 60) if indices is None else np.asarray(indices, dtype=int)
    edge = np.array([(x, y) for x in (0., .5, 1.) for y in (0., .5, 1.)
                     if x in (0., 1.) or y in (0., 1.)])
    source = np.vstack((projected[ids], edge))
    target = np.vstack((observed[ids], edge))
    if not np.isfinite(source).all() or not np.isfinite(target).all():
        raise ValueError('Feature alignment has invalid landmarks')
    maximum = float(np.linalg.norm(projected[ids] - observed[ids], axis=1).max())
    if maximum > .09:
        raise ValueError('Photo features are too far from the accepted mesh; refine its photo fit first')
    resolution = 128
    axis = (np.arange(resolution) + .5) / resolution
    x, y = np.meshgrid(axis, axis)
    grid = np.column_stack((x.ravel(), y.ravel()))
    for smoothing in (.00001, .00003, .0001, .0003, .001, .003, .01):
        # Border anchors stay fixed. Regularization absorbs inconsistent
        # detector points instead of folding an exact triangle interpolation.
        regularization = np.r_[np.full(len(ids), smoothing), np.zeros(len(edge))]
        try:
            mapper = RBFInterpolator(source, target, kernel='thin_plate_spline',
                                     smoothing=regularization)
            mapped = mapper(grid).reshape(resolution, resolution, 2)
        except (ValueError, np.linalg.LinAlgError) as exc:
            raise ValueError('Photo features could not form a stable projection') from exc
        dy, dx = np.gradient(mapped, 1 / resolution, axis=(0, 1))
        jacobian = np.stack((dx, dy), axis=-1)
        determinant = np.linalg.det(jacobian)
        stretch = np.linalg.svd(jacobian, compute_uv=False)
        if determinant.min() < .25 or stretch.min() < .20 or stretch.max() > 4.:
            continue
        mapped = mapped.astype(np.float32)
        def warp(points):
            return sample(mapped, points)
        residual = float(np.linalg.norm(warp(projected[ids]) - observed[ids], axis=1).max())
        if residual > .025:
            raise ValueError('Photo features need a closer mesh fit before color transfer')
        return warp, {'max_feature_offset': maximum, 'regularization': smoothing,
                      'max_alignment_residual': residual,
                      'minimum_jacobian': float(determinant.min()),
                      'maximum_stretch': float(stretch.max())}
    raise ValueError('Photo alignment would fold or severely stretch the image. Refine its photo fit first')


def raster_protection(uv, weights, shape):
    """Rasterize a deformation mask independently of evaluated mesh density."""
    from scipy.ndimage import maximum_filter
    uv, weights = np.asarray(uv), np.asarray(weights)
    if (uv.ndim != 3 or uv.shape[1:] != (3, 2) or weights.shape != uv.shape[:2]
            or not np.isfinite(uv).all() or not np.isfinite(weights).all()
            or np.any((weights < 0) | (weights > 1))):
        raise ValueError('Deformation protection needs matching finite UV triangles and bounded corner weights')
    output = np.zeros(shape, dtype=np.float32)
    for corners, values in zip(uv, weights):
        pixels = raster(corners, shape[1], shape[0])
        if pixels is not None:
            y, x, bary = pixels
            output[y, x] = np.maximum(output[y, x], bary @ values)
    return np.clip(maximum_filter(output, size=3), 0, 1)


def surface_visibility(projected, depth, normal, zbuffer, scale):
    """Compare depths at matching pixel centers, with footprint coverage.

    A grazing triangle needs a slope correction, not a larger occlusion
    allowance. Enlarging depth tolerance by inverse facing can expose the
    floor behind a nostril rim or an eyelid. Bilinear footprint coverage also
    avoids hard nearest-pixel steps along those boundaries.
    """
    height, width = zbuffer.shape
    xy = np.clip(projected * [width, height] - .5, 0, [width - 1, height - 1])
    low = np.floor(xy).astype(int)
    high = np.minimum(low + 1, [width - 1, height - 1])
    fraction = xy - low
    unit = scale / max(width, height)
    result = np.zeros(len(depth), dtype=np.float32)
    for ix, iy in ((0, 0), (1, 0), (0, 1), (1, 1)):
        x = high[:, 0] if ix else low[:, 0]
        y = high[:, 1] if iy else low[:, 1]
        weight = (fraction[:, 0] if ix else 1 - fraction[:, 0]) * (fraction[:, 1] if iy else 1 - fraction[:, 1])
        center = np.column_stack(((x + .5) / width, (y + .5) / height))
        offset = (center - projected) * [width, height] * unit
        expected = depth - (offset @ normal[:2]) / normal[2]
        nearest = zbuffer[y, x]
        # Subpixel numerical/curvature allowance stays bounded at every angle.
        gap = nearest - expected
        coverage = np.clip((unit * .25 - gap) / max(unit * .20, 1e-15), 0, 1)
        coverage = coverage * coverage * (3 - 2 * coverage)
        result += weight * np.where(np.isfinite(nearest), coverage, 0)
    return result


def project_samples(points, triangles, uv, photo, face_mask, shape, scale, strength=1, warp=None,
                    vertex_protection=None, texture_protection=None):
    """Return unblended color and confidence on visible atlas texels."""
    ph, pw = photo.shape[:2]
    height, width = shape
    if not 0 < strength <= 1 or not np.isfinite(scale) or scale <= 0:
        raise ValueError('Photo projection needs a positive camera scale and strength')
    if max(width, height, pw, ph) > 4096:
        raise ValueError('Photo color projection supports images up to 4K')
    if not all(np.isfinite(value).all() for value in (points, uv, photo, face_mask)):
        raise ValueError('Photo projection inputs contain non-finite values')
    if vertex_protection is not None:
        vertex_protection = np.asarray(vertex_protection)
        if (vertex_protection.shape != (len(points),) or not np.isfinite(vertex_protection).all()
                or np.any((vertex_protection < 0) | (vertex_protection > 1))):
            raise ValueError('Skin protection must contain one bounded weight per surface vertex')
    if texture_protection is not None:
        texture_protection = np.asarray(texture_protection)
        if (texture_protection.shape != (height, width) or not np.isfinite(texture_protection).all()
                or np.any((texture_protection < 0) | (texture_protection > 1))):
            raise ValueError('Texture protection must contain bounded weights in the output UV domain')
    screen = points[:, :2] / scale * [max(pw, ph) / pw, max(pw, ph) / ph] + .5
    zbuffer = np.full((ph, pw), -np.inf, dtype=np.float32)
    for ids in triangles:
        pixels = raster(screen[ids], pw, ph)
        if pixels is not None:
            y, x, bary = pixels
            zbuffer[y, x] = np.maximum(zbuffer[y, x], bary @ points[ids, 2])
    influence = np.zeros((height, width), dtype=np.float32)
    output = np.zeros((height, width, 3), dtype=np.float32)
    normals = np.cross(points[triangles[:, 1]] - points[triangles[:, 0]],
                       points[triangles[:, 2]] - points[triangles[:, 0]])
    lengths = np.linalg.norm(normals, axis=1)
    facing = normals[:, 2] / np.maximum(lengths, 1e-15)
    smooth_normals = np.zeros_like(points)
    for corner in range(3):
        np.add.at(smooth_normals, triangles[:, corner], normals)
    smooth_normals /= np.maximum(np.linalg.norm(smooth_normals, axis=1, keepdims=True), 1e-15)
    for index in np.flatnonzero(facing > .05):
        ids = triangles[index]
        pixels = raster(uv[index], width, height)
        if pixels is None:
            continue
        y, x, bary = pixels
        projected = bary @ screen[ids]
        inside = ((projected >= 0) & (projected <= 1)).all(1)
        y, x, bary, projected = y[inside], x[inside], bary[inside], projected[inside]
        depth = bary @ points[ids, 2]
        visible = surface_visibility(projected, depth, normals[index], zbuffer, scale)
        direction = bary @ smooth_normals[ids]
        cosine = direction[:, 2] / np.maximum(np.linalg.norm(direction, axis=1), 1e-15)
        angle = np.clip((cosine - .25) / .4, 0, 1)
        sampled = projected if warp is None else np.asarray(warp(projected))
        inside_photo = np.isfinite(sampled).all(1) & ((sampled >= 0) & (sampled <= 1)).all(1)
        sampled = np.nan_to_num(sampled).clip(0, 1)
        weight = sample(face_mask, sampled) * visible * inside_photo * (angle * angle * (3 - 2 * angle)) * strength
        if vertex_protection is not None:
            weight *= 1 - bary @ vertex_protection[ids]
        if texture_protection is not None:
            weight *= 1 - texture_protection[y, x]
        # Keep the strongest visible surface when UV boundaries share texels.
        keep = weight > influence[y, x]
        y, x, sampled, weight = y[keep], x[keep], sampled[keep], weight[keep]
        output[y, x] = sample(photo, sampled)[:, :3]
        influence[y, x] = weight
    if np.count_nonzero(influence > .1) < 100:
        raise ValueError('Photo and accepted face do not overlap enough; inspect their photo comparison')
    return output, influence


def project(points, triangles, uv, photo, face_mask, fallback, scale, strength=1):
    """Arrays use bottom-up image rows and camera coordinates looking down -Z."""
    if not np.isfinite(fallback).all():
        raise ValueError('Photo projection fallback contains non-finite values')
    color, influence = project_samples(points, triangles, uv, photo, face_mask,
                                       fallback.shape[:2], scale, strength)
    output = fallback.copy()
    output[..., :3] += (color - output[..., :3]) * influence[..., None]
    return output, influence


def register_view(points, target):
    """Estimate a camera for fixed 3D landmarks from a supporting photograph."""
    from scipy.optimize import least_squares
    from photo_fit_core import project as project_landmarks, rotation

    points = np.asarray(points, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    if points.shape != (68, 3) or target.shape != (68, 2):
        raise ValueError('Supporting photos need 68 corresponding landmarks')
    if not np.isfinite(points).all() or not np.isfinite(target).all():
        raise ValueError('Supporting photo landmarks are invalid')
    eye = np.linalg.norm(target[36] - target[45])
    face_height = np.linalg.norm(target[27] - target[8])
    if face_height < .07:
        raise ValueError('The supporting photo does not resolve the face')
    profile = eye / face_height < .45
    left = np.linalg.norm(target[36] - target[39])
    right = np.linalg.norm(target[42] - target[45])
    # At profile, the detector estimates hidden-side features. Use only the
    # visible brow/eye, nose, central lips and chin for camera registration.
    if profile:
        brow = np.arange(17, 22) if left >= right else np.arange(22, 27)
        visible_eye = np.arange(36, 42) if left >= right else np.arange(42, 48)
        indices = np.r_[brow, np.arange(27, 36), visible_eye, [8, 48, 51, 54, 57]]
    else:
        indices = np.arange(17, 60)
    baseline = max(eye, face_height * .55)
    center = points[indices].mean(axis=0)
    extent = max(float(np.ptp(points[indices], axis=0).max()), 1e-8)
    normalized = (points - center) / extent
    p, q = normalized[indices], target[indices]
    weights = np.ones(len(indices))
    weights[(indices >= 17) & (indices < 27)] = .4
    weights[indices >= 48] = .45
    factor = np.sqrt(weights[:, None]) / baseline
    bounds = ([-.65, -1.65, -.85, -7, -4, -4], [.65, 1.65, .85, 4, 4, 4])
    solutions = []
    starts = (-1.3, -.65, 0., .65, 1.3) if profile else (-.65, 0., .65)
    for yaw in starts:
        camera = np.array([0., yaw, 0., 0., 0., 0.])
        flat = project_landmarks(p, camera)
        scale = np.linalg.norm(q - q.mean(axis=0)) / max(np.linalg.norm(flat - flat.mean(axis=0)), 1e-8)
        camera[3] = np.log(scale)
        camera[4:] = q.mean(axis=0) - project_landmarks(p, camera).mean(axis=0)
        solutions.append(least_squares(
            lambda v: ((project_landmarks(p, v) - q) * factor).ravel(), camera,
            bounds=bounds, loss='soft_l1', f_scale=.035, max_nfev=75))
    best = min(solutions, key=lambda result: result.cost)
    camera = best.x
    error = float(np.sqrt(np.average(
        np.sum((project_landmarks(p, camera) - q) ** 2, axis=1), weights=weights)) / baseline)
    if not best.success or error > .105 or abs(camera[1]) > 1.62 or abs(camera[0]) > .62:
        raise ValueError(f'Supporting view could not be aligned reliably ({error:.1%} landmark error)')
    scale = extent / np.exp(camera[3])
    # The camera uses stable internal features. Its color warp also follows
    # the observed jaw so neck pixels cannot drift onto the lower cheek.
    visible_jaw = np.arange(0, 9) if left >= right else np.arange(8, 17)
    feature_indices = np.unique(np.r_[indices, visible_jaw])
    return center, rotation(camera[:3]), camera[4:] * scale, scale, {
        'normalized_landmark_error': error, 'yaw_degrees': float(np.rad2deg(camera[1])),
        'profile': bool(profile), 'used_landmarks': indices.tolist(),
        'feature_landmarks': feature_indices.tolist(),
    }


def balance_view(primary, primary_weight, secondary, secondary_weight):
    """Anchor supplementary exposure to overlapping skin in the primary view."""
    from scipy.ndimage import gaussian_filter

    overlap = (primary_weight > .65) & (secondary_weight > .65)
    # Avoid dark features, deep shadow and clipped highlights in calibration.
    luma = np.array([.2126, .7152, .0722], dtype=np.float32)
    a, b = primary @ luma, secondary @ luma
    overlap &= (a > .035) & (b > .035) & (a < .78) & (b < .78)
    if np.count_nonzero(overlap) < 100:
        raise ValueError('Supporting photo has too little well-lit overlap with the primary')
    log_gain = np.log((primary[overlap] + .012) / (secondary[overlap] + .012))
    median = np.clip(np.median(log_gain, axis=0), -.65, .65)
    # Low-frequency local correction helps reject changing illumination while
    # keeping wrinkles, freckles, and pore detail in the supporting view.
    residual = np.zeros_like(primary)
    residual[overlap] = np.clip(log_gain - median, -.32, .32)
    if max(primary.shape[:2]) > 2048:
        # Illumination is a low-frequency field. Solve it on a bounded grid,
        # then apply it channel by channel to retain full-resolution detail.
        # This avoids a large-sigma convolution over every 4K color channel.
        from PIL import Image
        height, width = primary.shape[:2]
        ratio = min(1., 512 / max(height, width))
        size = (max(1, round(width * ratio)), max(1, round(height * ratio)))
        def resize(values, destination):
            return np.asarray(Image.fromarray(values).resize(destination, Image.Resampling.BILINEAR))
        support = gaussian_filter(resize(overlap.astype(np.float32), size), max(1., height * ratio / 38.))
        corrected = secondary.copy()
        for channel in range(3):
            field = gaussian_filter(resize(residual[..., channel], size), max(1., height * ratio / 38.))
            field /= np.maximum(support, .18)
            corrected[..., channel] *= np.exp(median[channel] + resize(field, (width, height)))
        return corrected, np.exp(median).tolist(), int(overlap.sum())
    sigma = max(5., primary.shape[0] / 38.)
    support = gaussian_filter(overlap.astype(np.float32), sigma)
    field = gaussian_filter(residual, (sigma, sigma, 0)) / np.maximum(support[..., None], .18)
    return secondary * np.exp(median + field), np.exp(median).tolist(), int(overlap.sum())


def merge_view(color, confidence, secondary, support, primary_weight):
    """Fill weak coverage without doubling details in the primary portrait."""
    priority = np.clip((.88 - primary_weight) / .48, 0, 1)
    priority = priority * priority * (3 - 2 * priority)
    weight = support * priority
    union = confidence + (1 - confidence) * weight
    # An already useful observation wins over uncertain overlapping samples.
    contribution = weight * (1 - confidence ** 2)
    blend = contribution / np.maximum(confidence + contribution, 1e-8)
    color += (secondary - color) * blend[..., None]
    return union


def complete_skin_tone(base, projected, confidence, triangles, uv, gain, points=None):
    """Continue observed low-frequency skin color across the mesh's UV seams."""
    from PIL import Image
    from scipy.ndimage import gaussian_filter, distance_transform_edt
    from scipy.sparse import coo_matrix, diags
    from scipy.sparse.linalg import cg

    if points is not None:
        # Native exports may split coincident seam vertices into distinct IDs.
        # Share color unknowns across those copies without editing the mesh.
        tolerance = max(float(np.ptp(points, axis=0).max()) * 1e-6, 1e-12)
        _, physical_ids = np.unique(np.rint(points / tolerance), axis=0, return_inverse=True)
        triangles = physical_ids[triangles]
    matched = base.copy()
    matched[..., :3] *= gain
    size = (min(512, base.shape[1]), min(512, base.shape[0]))
    coverage = np.zeros(size[::-1], dtype=bool)
    for triangle in uv:
        pixels = raster(triangle, *size)
        if pixels is not None:
            y, x, _ = pixels
            coverage[y, x] = True
    def coarse(values):
        if values.ndim == 2:
            return np.asarray(Image.fromarray(values).resize(size, Image.Resampling.BILINEAR))
        return np.stack([coarse(values[..., index]) for index in range(values.shape[2])], axis=-1)
    mask = coarse(confidence)
    target = coarse(projected[..., :3])
    clean = coarse(matched[..., :3])
    sigma = max(size) / 100.
    support = gaussian_filter(mask, sigma)
    observed = gaussian_filter(target * mask[..., None], (sigma, sigma, 0)) / np.maximum(support[..., None], .03)
    clean = gaussian_filter(clean * coverage[..., None], (sigma, sigma, 0)) / np.maximum(
        gaussian_filter(coverage.astype(np.float32), sigma)[..., None], .03)
    confidence_uv = sample(support, uv.reshape(-1, 2))
    observed_uv = sample(np.log(observed + .008), uv.reshape(-1, 2))
    count = int(triangles.max()) + 1
    ids = triangles.ravel()
    weight = np.clip((confidence_uv - .55) / .35, 0, 1)
    weight = weight ** 2
    observed_weight = np.bincount(ids, weights=weight, minlength=count)
    constraint = np.zeros((count, 3))
    np.add.at(constraint, ids, observed_uv * weight[:, None])
    constraint /= np.maximum(observed_weight[:, None], 1e-8)
    strength = 12 * np.minimum(observed_weight, 1)
    edges = np.unique(np.sort(np.concatenate(
        (triangles[:, [0, 1]], triangles[:, [1, 2]], triangles[:, [2, 0]])), axis=1), axis=0)
    row = np.r_[edges[:, 0], edges[:, 1]]
    col = np.r_[edges[:, 1], edges[:, 0]]
    adjacency = coo_matrix((np.ones(len(row)), (row, col)), shape=(count, count)).tocsr()
    diagonal = np.asarray(adjacency.sum(axis=1)).ravel() + strength + .00001
    system = diags(diagonal) - adjacency
    baseline = np.zeros((count, 3))
    np.add.at(baseline, ids, sample(np.log(clean + .008), uv.reshape(-1, 2)))
    baseline /= np.maximum(np.bincount(ids, minlength=count)[:, None], 1)
    field = np.zeros((count, 3))
    preconditioner = diags(1 / diagonal)
    for channel in range(3):
        field[:, channel], info = cg(system, strength * constraint[:, channel] + .00001 * baseline[:, channel],
                                    x0=baseline[:, channel], M=preconditioner, rtol=1e-6, maxiter=2000)
        if info != 0:
            raise ValueError('Photo skin-tone continuation did not converge')
    atlas = np.zeros((size[1], size[0], 3), dtype=np.float32)
    for index, vertices in enumerate(triangles):
        pixels = raster(uv[index], *size)
        if pixels is not None:
            y, x, bary = pixels
            atlas[y, x] = bary @ field[vertices]
    distance, nearest = distance_transform_edt(~coverage, return_indices=True)
    gutter = (~coverage) & (distance <= 3)
    atlas[gutter] = atlas[nearest[0][gutter], nearest[1][gutter]]
    full_size = (base.shape[1], base.shape[0])
    absolute_tone = np.stack([np.asarray(Image.fromarray(atlas[..., channel]).resize(
        full_size, Image.Resampling.BILINEAR)) for channel in range(3)], axis=-1)
    baseline_tone = np.stack([np.asarray(Image.fromarray(clean[..., channel]).resize(
        full_size, Image.Resampling.BILINEAR)) for channel in range(3)], axis=-1)
    matched[..., :3] *= np.clip((np.exp(absolute_tone) - .008) / np.maximum(baseline_tone, .008), .1, 10)
    return matched, {'method': 'SURFACE_TONE_CONTINUATION', 'constrained_vertices': int(np.count_nonzero(strength)),
                     'solves_absolute_color': True}


def uv_coverage(uv, shape):
    """Covered texel centers, retaining gaps between UV islands."""
    height, width = shape
    coverage = np.zeros((height, width), dtype=bool)
    for triangle in uv:
        pixels = raster(triangle, width, height)
        if pixels is not None:
            y, x, _ = pixels
            coverage[y, x] = True
    if not coverage.any():
        raise ValueError('Photo output has no covered UV texels')
    return coverage


def feather_projection(output, fallback, confidence, coverage, radius, *, region=None):
    """Fade only inward from unsupported texels, preserving trusted detail.

    Propagation is constrained to covered atlas pixels: another island across
    a UV gutter cannot introduce an occlusion edge. Colors are composited,
    never blurred, and confidence can only decrease. Optional anatomical
    region weights limit the blend; zero-weight texels remain byte-exact.
    """
    from scipy.ndimage import binary_dilation

    radius = int(radius)
    if (output.shape != fallback.shape or output.ndim != 3 or output.shape[2] < 3
            or confidence.shape != output.shape[:2] or coverage.shape != confidence.shape
            or coverage.dtype != bool or not 0 <= radius <= 32):
        raise ValueError('Projection feather needs matching images, confidence and UV coverage')
    if region is not None:
        region = np.asarray(region)
        if (region.shape != confidence.shape or not np.isfinite(region).all()
                or np.any((region < 0) | (region > 1))):
            raise ValueError('Projection feather region needs matching bounded weights')
    front = coverage & (confidence <= 1e-6)
    if not radius or not front.any():
        return output, confidence, dict(radius=radius, softened_texels=0)
    factor = np.ones(confidence.shape, dtype=np.float32)
    factor[front] = 0
    for step in range(1, radius + 1):
        grown = binary_dilation(front, structure=np.ones((3, 3), dtype=bool), mask=coverage)
        ring = grown & ~front
        value = step / (radius + 1)
        factor[ring] = value * value * (3 - 2 * value)
        front = grown
    if region is not None:
        factor = 1 - (1 - factor) * region
    active = coverage & (factor < 1) & (confidence > 0)
    result = output.copy()
    result[active, :3] = fallback[active, :3] + (output[active, :3] - fallback[active, :3]) * factor[active, None]
    softened = confidence.copy()
    softened[active] *= factor[active]
    return result, softened, dict(radius=radius, softened_texels=int(np.count_nonzero(active)),
                                  method='INWARD_UV_COVERAGE_FADE')


def texture_gutters(images, uv, radius=8, coverage=None):
    """Extend island colors outside their boundaries for filtered sampling."""
    from scipy.ndimage import distance_transform_edt

    if coverage is None:
        coverage = uv_coverage(uv, images[0].shape[:2])
    distance, nearest = distance_transform_edt(~coverage, return_indices=True)
    gutter = (~coverage) & (distance <= radius)
    for pixels in images:
        pixels[gutter] = pixels[nearest[0][gutter], nearest[1][gutter]]
