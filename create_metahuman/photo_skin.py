"""A clean, tone-matched Smart Paint reference for photo-colored skin."""

import bpy
import numpy as np

from . import paint_session, skin_projection


def reference(obj, active, role):
    if not active.get('cm_photo_reference'):
        return None
    layout = skin_projection.uv_layout_signature(obj)
    cached = active.get('cm_photo_clean_reference')
    if isinstance(cached, bpy.types.Image):
        if active.get('cm_photo_clean_uv',cached.get('cm_native_uv_layout')) != layout:
            raise ValueError('Photo skin UVs changed; project its color again before Smart Paint')
        return cached
    if role == 'BODY':
        # The body received only a global tone correction, no photo patch.
        # PaintSession retains this image as its clean reference before editing.
        return active
    base = active.get('cm_photo_previous_albedo')
    mask = active.get('cm_photo_projection_mask')
    if not isinstance(base, bpy.types.Image) or not isinstance(mask, bpy.types.Image):
        return None
    if active.colorspace_settings.name != 'sRGB' or base.colorspace_settings.name != 'sRGB':
        raise ValueError('Photo skin repair needs sRGB color images')
    width, height = active.size
    if min(width, height) < 1 or max(width, height) > 4096:
        raise ValueError('Photo skin repair supports images up to 4K')
    clean = skin_projection.linear(paint_session.resample(base, width, height))
    current = skin_projection.linear(paint_session.pixels(active)[..., :3])
    influence = paint_session.resample(mask, width, height)[..., 0]
    if not all(np.isfinite(value).all() for value in (clean,current,influence)):
        raise ValueError('Photo skin contains non-finite color or mask pixels')
    valid = influence > max(float(influence.max()) * .95, .001)
    if np.count_nonzero(valid) < 100:
        raise ValueError('The photo projection mask has too little face coverage for a clean repair reference')
    old, new = clean[valid], current[valid]
    luminance = new @ [.2126, .7152, .0722]
    low, high = np.percentile(luminance, [25, 75])
    selected = (luminance >= low) & (luminance <= high)
    old_median, new_median = np.median(old[selected], axis=0), np.median(new[selected], axis=0)
    gain = np.clip(new_median / np.maximum(old_median, .005), .1, 32)
    chroma = old_median / max(float(old_median.sum()), 1e-8)
    distance = np.linalg.norm(clean / np.maximum(clean.sum(2, keepdims=True), 1e-8) - chroma, axis=2)
    amount = 1 - np.clip((distance - .05) / .13, 0, 1)
    amount = amount * amount * (3 - 2 * amount)
    clean *= 1 + (gain - 1) * amount[..., None]
    pixels = np.ones((height, width, 4), dtype=np.float32)
    pixels[..., :3] = skin_projection.encoded(np.clip(clean, 0, 1))
    image = bpy.data.images.new(active.name + '_CleanReference', width=width, height=height, alpha=True)
    try:
        image.colorspace_settings.name = 'sRGB'
        image.pixels.foreach_set(pixels.ravel())
        image.file_format = 'PNG'
        image.pack()
        image['cm_native_uv_layout'] = layout
        image['cm_photo_reference'] = active['cm_photo_reference']
        image['cm_photo_repair_gain'] = gain.tolist()
        active['cm_photo_clean_reference'] = image
        return image
    except BaseException:
        bpy.data.images.remove(image, do_unlink=True)
        raise
