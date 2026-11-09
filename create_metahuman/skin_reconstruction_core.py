"""Local skin reconstruction from source identity and a UV-matched skin prior.

This implementation combines local spectral evidence, source-color anchors and
screened gradient reconstruction. It does not estimate physical albedo from a
single lit image. SciPy is required in the external inference environment only.
"""
import numpy as np
from scipy import fft, ndimage


def periodic_component(log_color, region, *, maximum_period=32., compensate_filter=False):
    """Fit only two coherent repeating modes, leaving the source residual intact.

    Unlike reference reconstruction, this never replaces the broadband skin
    signal with a low-pass field. A weighted sinusoidal fit estimates narrow
    periodic contamination; irregular pores, scars and creases remain in the
    residual. Spectral evidence can still mistake intentionally periodic detail
    for an artifact, so removal remains masked and adjustable.
    """
    if not 8<=maximum_period<=128:raise ValueError('Periodic analysis supports periods from 8 through 128 texels')
    h,w=region.shape
    valid=region>.05
    if float(valid.mean())<.5:
        return np.zeros_like(log_color),0.,8.,0.,[]
    # Atlas padding and protected holes are not observations of skin. Black
    # UV borders create strong Fourier peaks of their own; extend nearby skin
    # into that analysis context while still fitting/publicizing only the mask.
    analysis=log_color
    if not valid.all():
        indices=ndimage.distance_transform_edt(~valid,return_distances=False,return_indices=True)
        analysis=log_color.copy()
        analysis[~valid]=log_color[tuple(indices[:,~valid])]
    lum=analysis@np.asarray([.2126,.7152,.0722],np.float32)
    sigma=3.*maximum_period/32.
    fine=lum-ndimage.gaussian_filter(lum,sigma)
    window=np.hanning(h)[:,None]*np.hanning(w)[None,:]
    power=np.abs(fft.fft2(fine*window))**2
    fy=fft.fftfreq(h)[:,None];fx=fft.fftfreq(w)[None,:]
    radius=np.sqrt(fy*fy+fx*fx);angle=np.arctan2(fy,fx)%np.pi
    valid=(radius>=1/maximum_period)&(radius<=1/4)&((fy>0)|((fy==0)&(fx>0)))
    ratios=np.zeros_like(power)
    edges=np.linspace(1/maximum_period,1/4,17)
    for low,high in zip(edges[:-1],edges[1:]):
        band=valid&(radius>=low)&(radius<high)
        if np.any(band):ratios[band]=power[band]/max(float(np.median(power[band])),1e-8)
    first=np.unravel_index(np.argmax(ratios),ratios.shape)
    difference=np.abs(angle-angle[first]);difference=np.minimum(difference,np.pi-difference)
    other=np.where(valid&(difference>np.pi/6),ratios,0)
    second=np.unravel_index(np.argmax(other),other.shape)
    paired=min(float(ratios[first]),float(other[second]))
    score=float(np.clip((paired-15)/60,0,1))
    period=float(1/max(float(radius[first]),1e-8))
    if score<=0:return np.zeros_like(log_color),score,period,paired,[]
    yy,xx=np.mgrid[:h,:w]
    log_power=np.log(np.maximum(power,1e-20))
    basis=[];frequencies=[]
    for py,px in (first,second):
        offsets=[]
        for axis,length in ((0,h),(1,w)):
            a=[py,px];b=[py,px];a[axis]=(a[axis]-1)%length;b[axis]=(b[axis]+1)%length
            low,mid,high=log_power[tuple(a)],log_power[py,px],log_power[tuple(b)]
            denominator=low-2*mid+high
            offsets.append(float(np.clip(.5*(low-high)/denominator,-.5,.5)) if denominator<-.0001 else 0.)
        vy=float(fy[py,0])+offsets[0]/h;vx=float(fx[0,px])+offsets[1]/w
        frequencies.append([vy,vx])
        phase=2*np.pi*(vy*yy+vx*xx)
        basis.extend((np.cos(phase),np.sin(phase)))
    design=np.stack(basis,axis=2).reshape(-1,4).astype(np.float32)
    weights=(window*np.clip(region,0,1)).ravel().astype(np.float32)
    total=max(float(weights.sum()),1e-8)
    design-=(design*weights[:,None]).sum(0)/total
    highpass=analysis-ndimage.gaussian_filter(analysis,(sigma,sigma,0))
    weighted=design*weights[:,None]
    gram=design.T@weighted
    gram.flat[::5]+=total*1e-5
    coefficients=np.linalg.solve(gram,weighted.T@highpass.reshape(-1,3))
    if compensate_filter:
        # Detection uses a high-pass signal. Recover the fitted sinusoid's
        # amplitude before subtracting it, rather than retaining the filter's
        # low-frequency tail as a visible coarse weave after upscaling.
        response=[max(1-np.exp(-2*np.pi**2*sigma**2*(vy*vy+vx*vx)),.01) for vy,vx in frequencies]
        coefficients/=np.repeat(response,2)[:,None]
    component=(design@coefficients).reshape(h,w,3)
    np.clip(component,-.15,.15,out=component)
    return component,score,period,paired,frequencies


def periodic_evidence(luminance):
    """Require narrow spectral peaks in two independent directions.

    Scores are artifact evidence, not calibrated probabilities. A single wrinkle
    direction is insufficient. Isotropic stochastic skin detail is penalized by
    the radial power baseline.
    """
    h, w = luminance.shape
    signal = luminance - ndimage.gaussian_filter(luminance, 3)
    window = np.hanning(h)[:, None] * np.hanning(w)[None, :]
    power = np.abs(fft.fft2(signal * window)) ** 2
    fy = fft.fftfreq(h)[:, None]
    fx = fft.fftfreq(w)[None, :]
    radius = np.sqrt(fx * fx + fy * fy)
    angle = np.arctan2(fy, fx) % np.pi
    valid = (radius >= 1/32) & (radius <= 1/4) & (fy > 0)
    ratios = np.zeros_like(power)
    for low, high in zip(np.linspace(1/32, 1/4, 17)[:-1], np.linspace(1/32, 1/4, 17)[1:]):
        band = valid & (radius >= low) & (radius < high)
        if np.any(band):
            baseline = max(float(np.median(power[band])), 1e-8)
            ratios[band] = power[band] / baseline
    first = np.unravel_index(np.argmax(ratios), ratios.shape)
    difference = np.abs(angle - angle[first])
    difference = np.minimum(difference, np.pi - difference)
    second = float(np.max(np.where(valid & (difference > np.pi/6), ratios, 0)))
    paired = min(float(ratios[first]), second)
    score = float(np.clip((paired - 15) / 60, 0, 1))
    period = float(1 / max(float(radius[first]), 1e-8))
    return score, period, paired


def screened(source, target, weight):
    """Minimize ||u-source||² + weight ||gradient(u-target)||².

    An orthonormal cosine basis provides Neumann boundaries on each overlapping
    tile. Source color anchors the low frequencies; target gradients control
    fine detail. No dense matrix or full-frame sparse factorization is built.
    """
    h, w = source.shape[:2]
    eigenvalues = (2-2*np.cos(np.pi*np.arange(h)/h))[:, None]
    eigenvalues = eigenvalues + (2-2*np.cos(np.pi*np.arange(w)/w))[None, :]
    if source.ndim == 3:
        eigenvalues = eigenvalues[..., None]
    residual = fft.dctn(source-target, axes=(0, 1), norm='ortho', workers=2)
    residual /= 1 + weight*eigenvalues
    return target + fft.idctn(residual, axes=(0, 1), norm='ortho', workers=2)


def local_evidence(luminance):
    candidates = [periodic_evidence(luminance)]
    h, w = luminance.shape
    if min(h, w) >= 128:
        for y in (0, h//2):
            for x in (0, w//2):
                patch = luminance[y:y+h//2, x:x+w//2]
                if min(patch.shape) >= 64:
                    candidates.append(periodic_evidence(patch))
    # Two supporting subwindows prevent one isolated mole/corner from making
    # an entire tile high confidence. Smaller windows tolerate UV distortion.
    local = sorted(candidates[1:], reverse=True)
    supported = local[1] if len(local) >= 2 else candidates[0]
    return max(candidates[0], supported)


def _starts(length, tile, overlap):
    if length <= tile:
        return [0]
    values = list(range(0, length-tile+1, tile-overlap))
    if values[-1] != length-tile:
        values.append(length-tile)
    return values


def reconstruct(source, reference, mask, *, strength=1., tile=256, overlap=64,
                normal=None, reference_normal=None, progress=None, cleanup_floor=0.,method='PERIODIC',
                periodic_maximum_period=32.,periodic_compensate_filter=False):
    if source.shape != reference.shape or source.ndim != 3 or source.shape[2] != 3:
        raise ValueError('Skin reconstruction requires matching linear RGB arrays')
    h, w = source.shape[:2]
    if method not in {'PERIODIC','REFERENCE'}:raise ValueError('Unknown skin cleanup method')
    if min(h, w) < 64 or max(h, w) > 8192:
        raise ValueError('Skin reconstruction supports images from 64 pixels through 8K')
    if mask.shape != (h, w) or not 0 <= strength <= 1 or not 0 <= cleanup_floor <= 1:
        raise ValueError('Invalid skin reconstruction mask or strength')
    if not 64 <= tile <= 1024 or not 0 < overlap < tile:
        raise ValueError('Invalid reconstruction tile size or overlap')
    if not all(np.isfinite(v).all() for v in (source, reference, mask)):
        raise ValueError('Skin reconstruction inputs contain non-finite values')
    if (normal is None) != (reference_normal is None):
        raise ValueError('Normal reconstruction requires source and reference normals together')
    if normal is not None and (normal.shape != source.shape or reference_normal.shape != source.shape
                              or not np.isfinite(normal).all() or not np.isfinite(reference_normal).all()):
        raise ValueError('Normal maps must have matching finite RGB arrays')
    output = np.zeros_like(source, dtype=np.float32)
    weights = np.zeros((h, w), dtype=np.float32)
    confidence = np.zeros((h, w), dtype=np.float32)
    normals = np.zeros_like(output) if normal is not None else None
    tiles = [(y, x) for y in _starts(h, tile, overlap) for x in _starts(w, tile, overlap)]
    evidence = []
    for index, (y, x) in enumerate(tiles):
        box = (slice(y, min(y+tile, h)), slice(x, min(x+tile, w)))
        src = np.maximum(source[box], .0001)
        ref = np.maximum(reference[box], .0001)
        region = np.clip(mask[box], 0, 1)
        log_color = np.log(src)
        lum = np.log(src @ np.asarray([.2126, .7152, .0722], dtype=np.float32))
        component=None;frequencies=[]
        if method=='PERIODIC' and float(region.max())>=.001:
            component,score,period,paired,frequencies=periodic_component(log_color,region,
                maximum_period=periodic_maximum_period,compensate_filter=periodic_compensate_filter)
        else:
            score, period, paired = local_evidence(lum) if float(region.max()) >= .001 else (0.,8.,0.)
        # Explicit artist override for blurred/non-periodic damaged areas. This
        # is never automatic evidence: callers must supply a repair mask.
        if method=='REFERENCE':score = max(score, cleanup_floor)
        if float(region.max()) < .001:
            score = 0.
        evidence.append({'x': x, 'y': y, 'score': score, 'period_texels': period,
            'paired_peak_ratio': paired,'removed_modes':frequencies})
        # Fade tile boundaries, including arbitrarily small edge images, while
        # keeping strictly positive coverage at the outer image boundary.
        feather = np.maximum(np.hanning(src.shape[0]), .02)[:, None] * np.maximum(np.hanning(src.shape[1]), .02)[None, :]
        feather = feather.astype(np.float32)
        candidate = source[box]
        influence = np.zeros_like(region)
        if score > 0 and strength > 0:
            scale = float(np.clip(period*.65, 2, 16))
            if method=='PERIODIC':
                solved=log_color-component
            else:
                coarse = ndimage.gaussian_filter(log_color, (scale, scale, 0))
                ref_lum = np.log(ref @ np.asarray([.2126, .7152, .0722], dtype=np.float32))
                fine = ref_lum - ndimage.gaussian_filter(ref_lum, scale)
                target = coarse + np.clip(fine, -.15, .15)[..., None]*.6
                solved = screened(log_color, target, max(32., period*period*2.))
            structural = ndimage.gaussian_filter(lum, max(scale/2, 1))
            gy, gx = np.gradient(structural)
            gate = 1/(1+((gx*gx+gy*gy)**.5/.035)**4)
            influence = (score*strength*region*gate).astype(np.float32)
            candidate = source[box] + (np.exp(solved)-source[box])*influence[..., None]
        output[box] += candidate*feather[..., None]
        weights[box] += feather
        confidence[box] += influence*feather
        if normals is not None:
            original = normal[box]*2-1
            clean = reference_normal[box]*2-1
            coarse_n = ndimage.gaussian_filter(original, (4, 4, 0))
            fine_n = clean-ndimage.gaussian_filter(clean, (4, 4, 0))
            solved_n = screened(original, coarse_n+fine_n*.3, 12.)
            value = original+(solved_n-original)*influence[..., None]
            value[..., 2] = np.maximum(value[..., 2], 1e-5)
            value /= np.maximum(np.linalg.norm(value, axis=2, keepdims=True), 1e-8)
            normals[box] += value*feather[..., None]
        if progress:
            progress(index+1, len(tiles))
    output /= weights[..., None]
    confidence /= weights
    np.clip(output, 0, 1, out=output)
    # Untouched pixels are bit-exact in the working float representation.
    output[confidence <= 0] = source[confidence <= 0]
    if normals is not None:
        normals /= weights[..., None]
        normals /= np.maximum(np.linalg.norm(normals, axis=2, keepdims=True), 1e-8)
        normals = normals*.5+.5
        normals[confidence <= 0] = normal[confidence <= 0]
    return output, normals, confidence, {'tiles': evidence, 'tile_count': len(tiles),
        'active_tiles': sum(v['score'] > 0 for v in evidence), 'mean_influence': float(confidence.mean()),
        'method':method,'reference_detail_transplanted':method=='REFERENCE',
        'confidence_is_probability': False, 'physical_albedo_recovered': False}


def inferred_relief(color, mask, artifact, *, radius=3., strength=.25):
    """Art-directed, band-limited height; this is not physical shape recovery.

    Reject chromatic edges and regions still showing paired periodic peaks.
    Work in log luminance to reduce dependence on the source palette. Broad
    lighting and facial shape are excluded; a soft limit prevents dark pigment
    and isolated texel spikes from producing deep pits. Height units are
    source-image texels, and derivatives are scaled when delivering at 8K.
    """
    if color.ndim != 3 or color.shape[2] != 3 or mask.shape != color.shape[:2] or artifact.shape != mask.shape:
        raise ValueError('Relief color, region and artifact masks must match')
    if not .5 <= radius <= 32 or not 0 <= strength <= 1:
        raise ValueError('Invalid inferred relief radius or strength')
    if not all(np.isfinite(v).all() for v in (color, mask, artifact)):
        raise ValueError('Relief inputs contain non-finite values')
    rgb = np.log(np.maximum(color, .001))
    lum = rgb @ np.asarray([.2126, .7152, .0722], dtype=np.float32)
    band = ndimage.gaussian_filter(lum, max(radius*.3, .5))-ndimage.gaussian_filter(lum, radius*2)
    chroma = rgb-lum[..., None]
    chroma -= ndimage.gaussian_filter(chroma, (radius*2, radius*2, 0))
    pigment_gate = 1/(1+(np.linalg.norm(chroma, axis=2)/.045)**4)
    # Evidence is recomputed after color cleanup, rather than reusing original
    # confidence as proof that a checker was successfully removed.
    reject = np.zeros_like(mask, dtype=np.float32)
    cover = np.zeros_like(mask, dtype=np.float32)
    for y in _starts(mask.shape[0], 256, 64):
        for x in _starts(mask.shape[1], 256, 64):
            box = (slice(y,y+256),slice(x,x+256))
            score = local_evidence(lum[box])[0]
            shape = lum[box].shape
            feather = np.maximum(np.hanning(shape[0]),.02)[:,None]*np.maximum(np.hanning(shape[1]),.02)[None,:]
            reject[box] += score*feather
            cover[box] += feather
    reject /= np.maximum(cover, 1e-8)
    # Erode/soften region edges so mask boundaries never become a raised rim.
    region = np.minimum(np.clip(mask,0,1), ndimage.gaussian_filter(np.clip(mask,0,1),radius*2))
    gate = region*pigment_gate*(1-reject)**2
    height = (.35*radius*strength*np.tanh(band/.08)*gate).astype(np.float32)
    return height, gate.astype(np.float32), {'inferred_not_recovered':True,
        'radius_source_texels':radius,'strength':strength,
        'mean_residual_periodic_evidence':float(reject.mean()),
        'height_max_source_texels':float(np.abs(height).max()),
        'original_artifact_mean':float(np.mean(artifact))}


def apply_relief(normal, height, *, directx=True, progress=None):
    """RNM composition onto native-resolution normals, with bounded scratch.

    UV image rows run downwards. DirectX tangent Y follows that convention;
    OpenGL uses its opposite. Preserve original bytes where height is flat.
    """
    from surface_maps import reorient_normals
    if normal.ndim != 3 or normal.shape[2] != 3 or height.ndim != 2 or min(height.shape)<2:
        raise ValueError('Relief requires RGB tangent normals and a height plane')
    if not np.isfinite(height).all():raise ValueError('Invalid relief height')
    h,w = normal.shape[:2]
    sy,sx = h/height.shape[0],w/height.shape[1]
    result = np.empty_like(normal)
    for y in range(0,h,128):
        end = min(y+128,h)
        yy = (np.arange(max(y-1,0),min(end+1,h),dtype=np.float32)+.5)/sy-.5
        xx = (np.arange(w,dtype=np.float32)+.5)/sx-.5
        plane = ndimage.map_coordinates(height,[np.broadcast_to(yy[:,None],(len(yy),w)),np.broadcast_to(xx[None,:],(len(yy),w))],order=1,mode='nearest')
        gy,gx = np.gradient(plane)
        offset = 1 if y else 0
        gx,gy = gx[offset:offset+end-y]*sx,gy[offset:offset+end-y]*sy
        detail = np.stack((-gx, -gy if directx else gy, np.ones_like(gx)),axis=2)
        detail /= np.linalg.norm(detail,axis=2,keepdims=True)
        detail = detail*.5+.5
        block = reorient_normals(normal[y:end],detail)
        untouched = (gx==0)&(gy==0)
        block[untouched] = normal[y:end][untouched]
        result[y:end] = block
        if progress:progress(end,h)
    return result


def repair_normals(normal, reference, mask):
    """Replace damaged shading only under an artist mask, at native resolution."""
    h,w=normal.shape[:2]
    output=normal.copy()
    for y in range(0,h,128):
        end=min(y+128,h)
        yy=np.arange(y,end,dtype=np.float32)+.5
        xx=np.arange(w,dtype=np.float32)+.5
        def sample(plane):
            coords=[np.broadcast_to((yy[:,None]*plane.shape[0]/h-.5),(end-y,w)),
                    np.broadcast_to((xx[None,:]*plane.shape[1]/w-.5),(end-y,w))]
            return ndimage.map_coordinates(plane,coords,order=1,mode='nearest')
        amount=np.clip(sample(mask),0,1)
        clean=np.stack([sample(reference[:,:,c]) for c in range(3)],axis=2)*2-1
        original=normal[y:end]*2-1
        value=original+(clean-original)*amount[:,:,None]
        value/=np.maximum(np.linalg.norm(value,axis=2,keepdims=True),1e-8)
        output[y:end]=value*.5+.5
        output[y:end][amount<=0]=normal[y:end][amount<=0]
    return output
