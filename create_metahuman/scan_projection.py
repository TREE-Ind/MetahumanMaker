"""Regenerable source projection until the artist applies the scan transaction."""

import hashlib

import numpy as np

from . import mesh_ops as m

SIGNATURE = 'cm_scan_pending_signature'


def key_points(key):
    points = np.empty((len(key.data), 3), dtype=np.float32)
    key.data.foreach_get('co', points.ravel())
    return points


def shape_signature(obj, *, topology=None):
    keys = obj.data.shape_keys
    if not keys or [k.name for k in keys.key_blocks] != ['Basis', 'Scan Detail']:
        raise ValueError('This layer has authored shape keys. Apply or clear it deliberately before generating new scan data')
    digest = hashlib.sha256((topology or m.topology(obj)).encode())
    for key in keys.key_blocks:
        digest.update(key_points(key).tobytes())
    return digest.hexdigest()


def editable(obj):
    return bool(obj and not obj.get('cm_scan_applied', True) and obj.get(SIGNATURE))


def require_applied(*objects):
    if any(obj and not obj.get('cm_scan_applied', True) for obj in objects):
        raise ValueError('Apply Projected Detail once, or clear the pending scan layer, before sculpting, subdividing or baking final maps')


def check(obj, *, topology=None):
    if not editable(obj):
        raise ValueError('This sculpt layer is applied or predates regenerable scan data. Apply or clear the existing layer deliberately')
    from . import simple_shading
    if simple_shading.blocking_modifiers(obj) or shape_signature(obj, topology=topology) != obj[SIGNATURE]:
        raise ValueError('The pending layer has manual edits or modifiers. They were preserved; apply this layer or clear it deliberately')


def dirty(obj, f):
    return bool(obj.get('cm_scan_mask_dirty')
        or bool(obj.get('cm_scan_mask_used')) != f.use_detail_mask
        or abs(float(obj.get('cm_scan_neck_fade', f.neck_fade))-f.neck_fade) > 1e-7)


def write(context, s, f, obj, projected):
    from .workflow import group_weights

    keys = obj.data.shape_keys.key_blocks
    base = key_points(keys['Basis']).astype(np.float64)
    matrix = np.asarray(obj.matrix_world, dtype=np.float64)
    world = base @ matrix[:3, :3].T + matrix[:3, 3]
    weights = group_weights(obj, 'CM_DetailMask', default=1) if f.use_detail_mask else np.ones(len(base))
    height = np.ptp(world[:, 2])
    fade = np.clip((world[:, 2]-world[:, 2].min()) / max(height*f.neck_fade, 1e-9), 0, 1)
    weights *= fade * (1-m.pin_weights(obj))
    output = base + (projected-base)*weights[:, None]
    if not np.all(np.isfinite(output)):
        raise ValueError('Source projection contains invalid coordinates; the previous displacement was preserved')
    commit(s, f, obj, output)


def commit(s, f, obj, output):
    keys = obj.data.shape_keys.key_blocks
    if output.shape != (len(obj.data.vertices), 3) or not np.isfinite(output).all():
        raise ValueError('Invalid scan output; the previous displacement was preserved')
    # All calculation completes before replacing the pending output.
    keys['Scan Detail'].data.foreach_set('co', output.astype(np.float32).ravel())
    keys['Scan Detail'].value = s.detail_strength
    obj[SIGNATURE] = shape_signature(obj)
    obj['cm_scan_mask_used'] = f.use_detail_mask
    obj['cm_scan_neck_fade'] = f.neck_fade
    obj['cm_scan_mask_dirty'] = False
    obj.data.update()


def regenerate(context, s, f):
    if context.mode != 'OBJECT':
        raise ValueError('Apply or cancel mask painting and finish sculpting before regenerating scan data')
    obj = s.detail
    check(obj)
    m.check_mesh(s.source, 'Scan source')
    if obj.get('cm_scan_regions'):
        from . import regional_scan
        regional_scan.regenerate(context,s,f,obj)
        return
    base = key_points(obj.data.shape_keys.key_blocks['Basis']).astype(np.float64)
    matrix = np.asarray(obj.matrix_world, dtype=np.float64)
    world = base @ matrix[:3, :3].T + matrix[:3, 3]
    projected, _ = m.nearest(m.source_bvh(context, s.source), world,
        np.linalg.norm(np.ptp(world, axis=0))*s.max_distance,
        progress=getattr(s, 'scan_progress', None))
    write(context, s, f, obj, m.to_local(obj, projected))
    s.status = 'Scan displacement regenerated from the current source and mask. Review it, then Apply Projected Detail once.'


def strength_layer(settings):
    scene = settings.id_data
    if not hasattr(scene, 'cm_features'):
        return None
    f = scene.cm_features
    return f.body_detail if settings.part == 'FULL_BODY' and f.texture_part == 'BODY' else settings.detail


def strength_get(settings):
    obj = strength_layer(settings)
    if editable(obj):
        return obj.data.shape_keys.key_blocks['Scan Detail'].value
    return float(settings.get('detail_strength', 1.0))


def strength_set(settings, value):
    value = min(1.0, max(0.0, float(value)))
    settings['detail_strength'] = value
    obj = strength_layer(settings)
    if editable(obj):
        obj.data.shape_keys.key_blocks['Scan Detail'].value = value
