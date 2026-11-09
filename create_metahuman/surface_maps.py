"""Linear tangent-space map operations independent of Blender."""

import numpy as np


def reorient_normals(base, detail,*,out=None):
    """Reoriented normal mapping: preserve base pores beneath sculpted detail."""
    base, detail = np.asarray(base), np.asarray(detail)
    if not base.ndim or base.shape != detail.shape or base.shape[-1] not in (3, 4):
        raise ValueError("Normal maps must have matching RGB or RGBA dimensions")
    # Blender pixels are float32. Keep that precision rather than promoting
    # two complete 8K maps and every arithmetic temporary to float64. Preserve
    # double precision for callers that explicitly supply other numeric arrays.
    dtype = np.float32 if base.dtype == detail.dtype == np.float32 else np.float64
    channels = base.shape[-1]
    source = base.reshape(-1, channels)
    sculpt = detail.reshape(-1, channels)
    result = np.empty(base.shape, dtype=dtype) if out is None else np.asarray(out)
    if result.shape!=base.shape or result.dtype!=dtype or not result.flags.c_contiguous or not result.flags.writeable:raise ValueError('Normal output must be writable contiguous pixels of the matching shape and precision')
    output = result.reshape(-1, channels)
    # Bound arithmetic scratch space independently of image dimensions.
    for first in range(0, len(source), 65536):
        last = first + 65536
        a, b = source[first:last], sculpt[first:last]
        if not np.isfinite(a).all() or not np.isfinite(b).all():
            raise ValueError('Normal maps contain non-finite pixels')
        t = a[:, :3].astype(dtype, copy=True)
        u = b[:, :3].astype(dtype, copy=True)
        t *= 2
        t -= 1
        t[:, 2] += 1
        u *= 2
        u -= 1
        u[:, :2] *= -1
        dot = np.einsum('ij,ij->i', t, u)
        dot /= np.maximum(t[:, 2], 1e-8)
        t *= dot[:, None]
        t -= u
        magnitude = np.linalg.norm(t, axis=1)
        np.maximum(magnitude, 1e-8, out=magnitude)
        t /= magnitude[:, None]
        t *= .5
        t += .5
        block = output[first:last]
        block[:, :3] = t
        if channels == 4:
            block[:, 3] = b[:, 3]
        np.clip(block, 0, 1, out=block)
    return result


def native_expression_color_in_place(pixels, delta):
    """Retarget native sRGB expression deltas onto a new linear-light albedo.

    Native animated color decodes as base + (delta - .5) * .5 in sRGB.
    Retaining this field preserves expression-specific crease/pigment changes,
    instead of replacing them with a nearly neutral regional red tint.
    """
    pixels,delta=np.asarray(pixels),np.asarray(delta)
    if (pixels.shape[:-1]!=delta.shape[:-1] or pixels.shape[-1] not in (3,4) or
            delta.shape[-1] not in (3,4) or pixels.dtype.kind!='f' or not pixels.flags.c_contiguous):
        raise ValueError('Native expression color needs matching floating-point albedo and delta pixels')
    output=pixels.reshape(-1,pixels.shape[-1]);reference=delta.reshape(-1,delta.shape[-1])
    for first in range(0,len(output),65536):
        block=output[first:first+65536,:3];change=reference[first:first+65536,:3]
        if not np.isfinite(block).all() or not np.isfinite(change).all() or np.any((change<0)|(change>1)):
            raise ValueError('Native expression color contains invalid pixels')
        encoded=np.where(block<=.0031308,block*12.92,1.055*np.maximum(block,0)**(1/2.4)-.055)
        encoded+=(change-.5)*.5;np.clip(encoded,0,1,out=encoded)
        block[:]=np.where(encoded<=.04045,encoded/12.92,((encoded+.055)/1.055)**2.4)
    return pixels


def tint_bloodflow_in_place(pixels, influence, strength):
    """Apply the artistic CM channel tint in the caller's pixel color domain."""
    pixels, influence = np.asarray(pixels), np.asarray(influence)
    if pixels.ndim < 2 or pixels.shape[-1] not in (3, 4) or influence.shape != pixels.shape[:-1]:
        raise ValueError('Blood-flow pixels and mask dimensions do not match')
    if pixels.dtype.kind != 'f' or not pixels.flags.c_contiguous or not np.isfinite(strength):
        raise ValueError('Blood-flow tint needs contiguous floating-point pixels and finite strength')
    output = pixels.reshape(-1, pixels.shape[-1])
    mask = influence.reshape(-1)
    for first in range(0, len(output), 65536):
        last = first + 65536
        block, weights = output[first:last], mask[first:last]
        if not np.isfinite(block).all() or not np.isfinite(weights).all():
            raise ValueError('Blood-flow pixels or mask contain non-finite values')
        factor = np.empty(len(block), dtype=np.float64)
        np.multiply(weights, 1., out=factor, dtype=np.float64)
        factor *= strength
        factor += 1
        np.multiply(block[:, 0], factor, out=block[:, 0])
        np.multiply(weights, -.35, out=factor, dtype=np.float64)
        factor *= strength
        factor += 1
        np.multiply(block[:, 1], factor, out=block[:, 1])
        np.multiply(block[:, 2], factor, out=block[:, 2])
    return pixels
