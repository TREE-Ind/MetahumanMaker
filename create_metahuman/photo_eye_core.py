"""Conservative observed iris-color cues; no reflectance or shader recovery."""
import numpy as np


def iris_guidance(image, landmarks):
    rgb = np.asarray(image)
    points = np.asarray(landmarks, dtype=float)
    if rgb.ndim != 3 or rgb.shape[2] != 3 or rgb.dtype != np.uint8:
        raise ValueError('Use a decoded RGB8 reference')
    if points.shape != (478, 2) or not np.isfinite(points).all():
        return dict(status='unavailable', reason='Reference has no reliable iris landmarks')
    height, width = rgb.shape[:2]
    eyes = []
    for center_index, rim, lid in (
            (468, [469, 470, 471, 472], [33, 160, 158, 133, 153, 144]),
            (473, [474, 475, 476, 477], [362, 385, 387, 263, 373, 380])):
        center = points[center_index]
        ring = points[rim]
        axes = np.column_stack(((ring[0] - ring[2]) / 2, (ring[1] - ring[3]) / 2))
        radii = np.linalg.svd(axes, compute_uv=False)
        record = dict(center_pixels=center.tolist(), status='unavailable')
        eyes.append(record)
        aperture = points[lid]
        eye_width = float(np.linalg.norm(aperture[3] - aperture[0]))
        if eye_width > 2:
            eyelid_height = (np.linalg.norm(aperture[1] - aperture[5]) +
                             np.linalg.norm(aperture[2] - aperture[4])) / 2
            record['geometry'] = dict(
                aperture_over_width=float(eyelid_height / eye_width),
                iris_diameter_over_width=float(np.linalg.norm(ring[0] - ring[2]) / eye_width))
        if radii.min() < 5 or radii.min() / max(radii.max(), 1e-8) < .25:
            record['reason'] = 'Iris is too small or too oblique to sample safely'
            continue
        low = np.maximum(0, np.floor(ring.min(0) - 1).astype(int))
        high = np.minimum([width, height], np.ceil(ring.max(0) + 2).astype(int))
        if np.any(high <= low):
            record['reason'] = 'Iris lies outside the reference'
            continue
        y, x = np.mgrid[low[1]:high[1], low[0]:high[0]]
        xy = np.column_stack((x.ravel() + .5, y.ravel() + .5))
        uv = (xy - center) @ np.linalg.inv(axes).T
        radius = np.linalg.norm(uv, axis=1)
        annulus = (radius >= .50) & (radius <= .88)
        # Clip the iris annulus to the visible lid aperture, excluding lashes/skin.
        inside = np.zeros(len(xy), dtype=bool)
        polygon = points[lid]
        for a, b in zip(polygon, np.roll(polygon, -1, axis=0)):
            if abs(b[1] - a[1]) < 1e-12:
                continue
            inside ^= ((a[1] > xy[:, 1]) != (b[1] > xy[:, 1])) & (
                xy[:, 0] < (b[0] - a[0]) * (xy[:, 1] - a[1]) / (b[1] - a[1]) + a[0])
        selected = annulus & inside
        values = rgb[y.ravel(), x.ravel()].astype(float) / 255
        luma = values @ np.array([.2126, .7152, .0722])
        if selected.sum() < 32:
            record['reason'] = 'Too little visible iris to separate pigment from pupil or lids'
            continue
        dark, bright = np.quantile(luma[selected], [.10, .80])
        # Reject dark pupil/limbal contamination and bright corneal reflections.
        selected &= (luma >= max(dark, .015)) & (luma <= min(bright, .85))
        bins = ((np.arctan2(uv[:, 1], uv[:, 0]) + np.pi) * 12 / (2 * np.pi)).astype(int) % 12
        sectors = len(np.unique(bins[selected]))
        if selected.sum() < 24 or sectors < 4:
            record['reason'] = 'Pigment sampling is dominated by occlusion or highlights'
            continue
        color = np.median(values[selected], axis=0)
        linear = np.where(color <= .04045, color / 12.92, ((color + .055) / 1.055) ** 2.4)
        record.update(status='observed', srgb=color.tolist(), linear_rgb=linear.tolist(),
                      hex='#' + ''.join(f'{int(round(v * 255)):02X}' for v in color),
                      pixels=int(selected.sum()), angular_sectors=sectors,
                      visible_annulus_fraction=float(selected.sum() / max(annulus.sum(), 1)))
    eyes.sort(key=lambda row: row['center_pixels'][0])
    return dict(status='observed' if any(e['status'] == 'observed' for e in eyes) else 'unavailable',
                method='VISIBLE_IRIS_ANNULUS', eyes={name: row for name, row in zip(('image_left', 'image_right'), eyes)},
                limitation='Observed photo color includes lighting, camera processing and reflections. It is not recovered iris albedo or a calibrated native shader setting.')


def compare_iris_observations(reference, candidate):
    """Compare photo/render cues without treating them as recovered reflectance.

    Call iris_guidance on each decoded image with its own detected landmarks.
    Image-left/right are image coordinates, so mirrored views must be corrected
    by the caller. Neither comparison nor a smaller color difference proves
    overall likeness or a valid native material override.
    """
    if any(guide.get('method') != 'VISIBLE_IRIS_ANNULUS' for guide in (reference, candidate)):
        raise ValueError('Compare two visible-iris observations with independent image landmarks')
    eyes = {}
    for side in ('image_left', 'image_right'):
        a, b = (guide.get('eyes', {}).get(side, {}) for guide in (reference, candidate))
        if a.get('status') != 'observed' or b.get('status') != 'observed':
            eyes[side] = dict(status='unavailable', reason='Both images need a visible, reliable iris sample')
            continue
        colors = np.asarray([a.get('srgb'), b.get('srgb')], dtype=float)
        if colors.shape != (2, 3) or not np.isfinite(colors).all() or np.any((colors < 0) | (colors > 1)):
            raise ValueError('Observed iris colors must be finite sRGB triples between zero and one')
        linear = np.where(colors <= .04045, colors / 12.92, ((colors + .055) / 1.055) ** 2.4)
        luma = linear @ np.array([.2126, .7152, .0722])
        row = dict(status='compared', reference_srgb=colors[0].tolist(), candidate_srgb=colors[1].tolist(),
                   mean_absolute_code_difference=float(np.mean(np.abs(colors[1] - colors[0])) * 255),
                   candidate_to_reference_luminance=float(luma[1] / luma[0]) if luma[0] > 1e-8 else None)
        geometry = {}
        for name in ('aperture_over_width', 'iris_diameter_over_width'):
            x, y = (eye.get('geometry', {}).get(name) for eye in (a, b))
            if x is not None and y is not None and np.isfinite([x, y]).all():
                geometry[name] = dict(reference=float(x), candidate=float(y), difference=float(y - x))
        if geometry:
            row['geometry'] = geometry
        eyes[side] = row
    return dict(status='compared' if any(row['status'] == 'compared' for row in eyes.values()) else 'unavailable',
                eyes=eyes, limitation='Image-space cues include lighting, exposure and shader response. Review a frontal neutral render and the actual baked eyes; parameter readback alone does not prove a visible change.')
