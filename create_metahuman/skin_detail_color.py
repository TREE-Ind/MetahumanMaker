"""Replace fine color texture with a reference while retaining local skin tone."""
import numpy as np


def _box(values, radius, axis):
    """Separable box pass with bounded prefix-sum scratch space."""
    result = np.empty_like(values)
    other = 1 - axis
    for start in range(0, values.shape[other], 64):
        selection = [slice(None), slice(None), slice(None)]
        selection[other] = slice(start, start + 64)
        block = values[tuple(selection)]
        pad = [(0, 0)] * 3
        pad[axis] = (radius, radius)
        padded = np.pad(block, pad, mode='edge')
        prefix = np.cumsum(padded, axis=axis, dtype=np.float64)
        shape = list(prefix.shape)
        shape[axis] = 1
        prefix = np.concatenate((np.zeros(shape, dtype=np.float64), prefix), axis=axis)
        first = [slice(None)] * 3
        last = [slice(None)] * 3
        first[axis] = slice(None, -(2 * radius + 1))
        last[axis] = slice(2 * radius + 1, None)
        result[tuple(selection)] = (prefix[tuple(last)] - prefix[tuple(first)]) / (2 * radius + 1)
    return result


def low_frequency(values, radius):
    result = values
    for _ in range(3):
        result = _box(_box(result, radius, 1), radius, 0)
    return result


def match_detail(source, reference, radius):
    """Linear RGB; source supplies tone, reference supplies bounded fine detail.

    The artist's paint mask controls where this replacement is used. This is
    frequency separation, not reconstruction of missing source information.
    """
    if source.shape != reference.shape or source.ndim != 3 or source.shape[2] != 3:
        raise ValueError('Skin detail matching needs equally sized RGB images')
    if not 1 <= radius <= 64 or int(radius) != radius:
        raise ValueError('Skin detail scale must be between 1 and 64 texels')
    if not np.isfinite(source).all() or not np.isfinite(reference).all():
        raise ValueError('Skin detail matching needs finite color pixels')
    tone = low_frequency(source, radius)
    # Reference pigment (lip color, freckles, undertones) must not repaint the
    # source palette. Transfer only bounded luminance contrast, not RGB ratios.
    luminance = reference @ np.asarray([.2126, .7152, .0722], dtype=np.float32)
    clean = low_frequency(luminance[..., None], radius)[..., 0]
    np.maximum(clean, .005, out=clean)
    np.divide(luminance, clean, out=clean)
    np.clip(clean, .85, 1.15, out=clean)
    tone *= clean[..., None]
    np.clip(tone, 0, 1, out=tone)
    return tone
