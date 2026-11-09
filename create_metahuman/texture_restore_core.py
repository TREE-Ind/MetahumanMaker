"""Source-constrained texture upscaling with an optional learned proposal.

The learned proposal is a prior, not recovered ground truth. Retain source color,
limit its fine residual, suppress periodic regions and UV boundaries, and check
the result against the original texels. This module does not generate PBR depth.
"""
import numpy as np
from PIL import Image
from scipy import ndimage,fft


def linear(rgb):
    return np.where(rgb<=.04045,rgb/12.92,((rgb+.055)/1.055)**2.4).astype(np.float32)


def encoded(rgb):
    return np.where(rgb<=.0031308,rgb*12.92,1.055*np.maximum(rgb,0)**(1/2.4)-.055)


def resize(values,size):
    """Resize linear channels independently without a full-frame coordinate grid."""
    return np.stack([np.asarray(Image.fromarray(values[:,:,c]).resize(size,Image.Resampling.BICUBIC))
                     for c in range(values.shape[2])],axis=2).astype(np.float32)


def reduce(values,scale):
    h,w=values.shape[:2]
    return values.reshape(h//scale,scale,w//scale,scale,3).mean(axis=(1,3))


def artifact_spectrum(source):
    """Exclude corroborated repeating directions from source consistency only.

    A broad structure lock can otherwise reintroduce a checker whose period
    overlaps real wrinkles. Paired narrow peaks supply a masked exception,
    not a claim that all periodic skin content is unwanted.
    """
    from skin_reconstruction_core import periodic_component
    h,w=source.shape[:2]
    _,score,period,ratio,frequencies=periodic_component(
        np.log(np.maximum(linear(source),.005)),np.ones((h,w),np.float32))
    transfer=np.ones((h,w),np.float32)
    if score>=.1 and len(frequencies)==2:
        fy=np.arange(h,dtype=np.float32)[:,None]/(2*h)
        fx=np.arange(w,dtype=np.float32)[None,:]/(2*w)
        for vy,vx in frequencies:
            radius=np.hypot(vy,vx)
            width=max(2/min(h,w),radius*.12)
            distance=np.sqrt((fy-abs(vy))**2+(fx-abs(vx))**2)/width
            fade=np.clip(distance-1,0,1)
            transfer*=.5-.5*np.cos(np.pi*fade)
    return transfer,{'paired_spectral_score':score,'paired_peak_ratio':ratio,
        'estimated_period':period,'frequencies':frequencies,
        'detected_pattern_exception':bool(score>=.1 and len(frequencies)==2),
        'notched_source_modes':int(np.sum(transfer==0))}


def prepare_model_input(source,*,valid=None,mask=None,protect=None):
    """Remove selected coherent pattern bands from model context, not identity.

    This conditioning image is never published as the character albedo. The
    accepted source remains the constraint for the final learned residual.
    """
    transfer,report=artifact_spectrum(source)
    if not report['detected_pattern_exception']:return source.copy(),report
    values=linear(source);logs=np.log(np.maximum(values,.005))
    coefficients=fft.dctn(logs,axes=(0,1),norm='ortho')
    contamination=fft.idctn(coefficients*(1-transfer[:,:,None]),axes=(0,1),norm='ortho')
    np.clip(contamination,-.35,.35,out=contamination)
    gate=np.ones(source.shape[:2],np.float32)
    if valid is not None:
        distance=ndimage.distance_transform_edt(np.pad(valid>.5,1))[1:-1,1:-1]
        gate*=np.clip(valid,0,1)*np.clip((distance-2)/6,0,1)
    if mask is not None:gate*=np.clip(mask,0,1)
    if protect is not None:gate*=1-np.clip(protect,0,1)
    prepared=np.clip(encoded(np.exp(logs-contamination*gate[:,:,None])),0,1)
    prepared[gate<=0]=source[gate<=0]
    report.update(context_only=True,conditioning_max_delta=float(np.abs(prepared-source).max()))
    return prepared,report


def _trusted(error,artifact,trusted_band=.025,spectrum=None):
    if artifact is None:return error
    h,w=error.shape[:2]
    fy=np.arange(h,dtype=np.float32)[:,None]/(2*h)
    fx=np.arange(w,dtype=np.float32)[None,:]/(2*w)
    radius=np.sqrt(fy*fy+fx*fx)
    # A genuine excluded band is necessary: repeatedly applying a Gaussian
    # consistency filter would eventually restore the unwanted pattern too.
    fade=np.clip((radius-trusted_band)/trusted_band,0,1)
    transfer=(.5+.5*np.cos(np.pi*fade))[:,:,None]
    transfer[radius>=2*trusted_band]=0
    if spectrum is not None:transfer*=spectrum[:,:,None]
    low=fft.idctn(fft.dctn(error,axes=(0,1),norm='ortho')*transfer,axes=(0,1),norm='ortho')
    return error*(1-artifact[:,:,None])+low*artifact[:,:,None]


def _project(values,source,scale,allow=None,steps=16,artifact=None,trusted_band=.025,spectrum=None):
    # Smooth back-projection avoids introducing a visible 2x2 block correction
    # grid. Protected regions retain the already constrained baseline.
    for iteration in range(steps):
        error=_trusted(source-reduce(values,scale),artifact,trusted_band,spectrum)
        if float(np.abs(error).max())<2e-5:break
        update=resize(error,(values.shape[1],values.shape[0]))
        if allow is not None:update*=allow[:,:,None]
        values+=update
        np.clip(values,0,1,out=values)
    error=_trusted(source-reduce(values,scale),artifact,trusted_band,spectrum)
    return iteration+1,float(np.abs(error).max())


def restore(source,proposal,*,valid=None,protect=None,strength=.35,detail_limit=.08,
            periodic=None,artifact_repair=False,trusted_band=.025,pattern_exception=False):
    """Return an RGB upscale, generated-residual map, and factual diagnostics.

    Arrays are sRGB 0..1. Valid/protect/periodic maps use source dimensions;
    periodic values are evidence, not probabilities. The output must be an
    integer 2x or 4x scale. Protected regions use interpolation of accepted data.
    """
    if source.ndim!=3 or source.shape[2]!=3 or proposal.ndim!=3 or proposal.shape[2]!=3:
        raise ValueError('Texture restoration requires RGB source and proposal')
    h,w=source.shape[:2]
    scale=proposal.shape[0]//h
    if scale not in (2,4) or proposal.shape[:2]!=(h*scale,w*scale):
        raise ValueError('The learned proposal must be a matching 2x or 4x upscale')
    if not 0<=strength<=1 or not 0<=detail_limit<=.25:
        raise ValueError('Invalid texture detail controls')
    if not .01<=trusted_band<=.15:
        raise ValueError('Trusted structure band must be .01...15 cycles per source texel')
    for mask in (valid,protect,periodic):
        if mask is not None and (mask.shape!=(h,w) or not np.isfinite(mask).all()):
            raise ValueError('Texture masks must match the source and contain finite values')
    if not all(np.isfinite(v).all() and v.min()>=0 and v.max()<=1 for v in (source,proposal)):
        raise ValueError('Texture inputs must be finite RGB values in 0..1')
    original=linear(source)
    baseline=resize(original,(w*scale,h*scale));np.clip(baseline,0,1,out=baseline)
    baseline_steps,baseline_error=_project(baseline,original,scale)
    learned=linear(proposal)
    sigma=scale*8 if artifact_repair else max(scale,2)
    low_base=ndimage.gaussian_filter(baseline,(sigma,sigma,0))
    low_learned=ndimage.gaussian_filter(learned,(sigma,sigma,0))
    learned*=np.clip(low_base/np.maximum(low_learned,.001),.5,2.)
    difference=learned-baseline
    difference-=ndimage.gaussian_filter(difference,(sigma,sigma,0))
    limit=detail_limit*np.maximum(low_base,.025)
    if artifact_repair and periodic is not None:
        contaminated=(periodic>=.25)[:,:,None]
        # Removing a strong checker needs room to subtract the existing source
        # oscillation. The ordinary synthesis cap alone traps that oscillation
        # in the output. Only explicitly selected contaminated regions receive
        # this replacement budget; the learned residual itself remains bounded.
        expanded=np.abs(baseline-low_base)+limit
        limit=np.where(np.repeat(np.repeat(contaminated,scale,axis=0),scale,axis=1),expanded,limit)
    np.clip(difference,-limit,limit,out=difference)
    gate=np.ones((h,w),np.float32)
    if valid is not None:
        inside=np.clip(valid,0,1)
        distance=ndimage.distance_transform_edt(np.pad(inside>.5,1))[1:-1,1:-1]
        gate*=inside*np.clip((distance-2)/6,0,1)
    if protect is not None:gate*=1-np.clip(protect,0,1)
    artifact=None
    spectrum=None;spectral_report=None
    if periodic is not None:
        if artifact_repair:
            # Strict texel consistency would force the unwanted checker back
            # into the output. In explicitly detected regions constrain broad
            # identity/color bands, while allowing the contaminated fine band
            # to be replaced. This is inferred replacement, never recovery.
            artifact=(periodic>=.25).astype(np.float32)
            if pattern_exception:spectrum,spectral_report=artifact_spectrum(source)
        else:gate*=1-.85*np.clip(periodic,0,1)
    allow=np.repeat(np.repeat(gate,scale,axis=0),scale,axis=1)
    result=baseline+difference*(strength*allow[:,:,None])
    np.clip(result,0,1,out=result)
    steps,error=_project(result,original,scale,allow,artifact=artifact,trusted_band=trusted_band,spectrum=spectrum)
    generated=np.linalg.norm(result-baseline,axis=2)
    # No new generated detail is allowed inside protected/UV-excluded blocks.
    result[allow<=0]=baseline[allow<=0]
    generated[allow<=0]=0
    return np.clip(encoded(result),0,1),generated,{
        'scale':scale,'strength':strength,'detail_limit':detail_limit,
        'linear_source_consistency_max_error':error,
        'consistency_mode':'artifact_band_relaxed' if artifact is not None else 'strict_source_texels',
        'artifact_mode_trusted_band_cyc_per_source_texel':[0,2*trusted_band] if artifact is not None else None,
        'raw_source_max_error':float(np.abs(original-reduce(result,scale)).max()),
        'baseline_consistency_max_error':baseline_error,
        'back_projection_steps':steps,'baseline_projection_steps':baseline_steps,
        'generated_residual_max_linear':float(generated.max()),
        'generated_detail_is_ground_truth':False,
        'pattern_exception':spectral_report,
        'new_normal_roughness_or_cavity_generated':False}
