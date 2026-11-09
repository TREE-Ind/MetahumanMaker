"""Apply pretrained detail proposals to shared, continuous surface patches."""
import numpy as np
import shutil
from PIL import Image

from texture_surface import Surface,plan,sample
from texture_restore_core import restore,linear,prepare_model_input,resize
from texture_restore_backend import fingerprint
from texture_edit_prior import propose


def candidate(source,mask,valid,geometry,*,backend,expected,folder,strength=1,
              invention=.35,brief='Fine natural skin grain',protect=None,
              preserve_structure=True,patterns=False,progress=None,output_scale=1):
    if output_scale not in (1,2):raise ValueError('Surface output scale must be 1 or 2')
    if fingerprint(backend,'EDIT',progress)!=expected:
        raise ValueError('The editing backend changed after submission')
    original=linear(source)
    h,w=source.shape[:2]
    if output_scale==2:
        original=np.clip(resize(original,(w*2,h*2)),0,1)
    if invention==0 or strength==0:
        return original,np.zeros(original.shape[:2],np.float32),{
            'method':'LEARNED','layout':'SURFACE','invention':invention,'inference_skipped':True,
            'physical_albedo_recovered':False,'generated_detail_is_ground_truth':False}
    surface=Surface(*geometry)
    head_scale=float(np.median(np.ptp(surface.points,axis=0)))
    spans=(head_scale*.08,head_scale*.15) if output_scale==2 else None
    cameras,planning=plan(surface,source,mask*valid,protect,limit=128 if output_scale==2 else 48,progress=progress,span_limits=spans)
    if not cameras:raise ValueError('No visible surface patches cover the selected unprotected skin')
    if output_scale==2:
        def upscale(value):return np.asarray(Image.fromarray(value).resize((w*2,h*2),Image.Resampling.BILINEAR),dtype=np.float32)
        output_mask,output_valid=upscale(mask),upscale(valid)
        output_protect=upscale(protect) if protect is not None else None
    else:output_mask,output_valid,output_protect=mask,valid,protect
    mapped=output_mask*output_valid
    del output_mask,output_valid
    sums=np.zeros_like(original);weights=np.zeros(original.shape[:2],np.float32)
    projection_cache=surface.raster_cache(mapped,output_protect) if output_scale==2 else None
    details=[]
    for index,camera in enumerate(cameras):
        from resource_budget import require_disk,require,GIB,MIB
        require_disk('surface texture patch',folder,8*GIB+128*MIB)
        if progress:progress(index,len(cameras),'Generating surface detail')
        render=surface.render(source,camera)
        protected=sample(protect,render['uv']) if protect is not None else None
        context=render['color'];conditioning=None
        if patterns:
            context,conditioning=prepare_model_input(context,valid=render['valid'],protect=protected)
        if index==0:
            for name,key in [('color','color'),('normal','normal'),('depth','depth')]:
                Image.fromarray(np.rint(np.clip(render[key],0,1)*255).astype('uint8')).save(folder/('surface-guide-'+name+'.png'))
        prompt=('Restore this exact continuous surface patch of skin. Preserve the position and shape '
            'of existing creases, scars and pigmentation. Remove checker patterns and projection smearing. '
            'Flat unlit base color, no new shadows or highlights. Permitted detail: '+brief)
        prior_folder=folder/('surface-prior-'+str(index))
        require('surface detail model',5*GIB)
        prior,prior_report=propose(context,prior_folder,backend=backend,prompt=prompt,verified=expected,
            cache=folder.parent/'.prior-cache')
        shutil.copy2(prior_folder/'inference.log',folder/'editing-model.log')
        if index==0:shutil.copy2(prior_folder/'proposal.png',folder/'surface-prior-preview.png')
        if prior_folder.resolve().parent!=folder.resolve() or prior_folder.is_symlink():
            raise ValueError('The model scratch folder left its owned texture job')
        shutil.rmtree(prior_folder)
        band=.025+(.075 if preserve_structure else .025)*(1-invention)
        # Solve the continuous context before masking atlas publication. Strict
        # source constraints just outside a small repair selection otherwise
        # feed contaminated frequencies back through the patch-wide transform.
        result,_,info=restore(render['color'],prior,valid=render['valid'].astype(np.float32),
            protect=protected,strength=strength*invention,detail_limit=.025+.225*invention,
            periodic=render['valid'].astype(np.float32),artifact_repair=True,trusted_band=band,pattern_exception=patterns)
        projection=surface.render(source,camera,size=512*output_scale) if output_scale==2 else render
        if output_scale==2:
            surface.project(linear(result),projection,original,mapped,output_protect,accumulate=(sums,weights),cache=projection_cache)
        else:
            patch_sums,patch_weights=surface.project(linear(result),projection,original,mapped,output_protect)
            sums+=patch_sums;weights+=patch_weights
        details.append({'camera':camera,'depth_range':render['depth_range'],'prior':prior_report,'constraints':info,
                        'model_context':conditioning})
        if progress:progress(index+1,len(cameras),'Restoring surface patches')
    covered=weights>0
    if output_scale==2:np.divide(sums,weights[...,None],out=sums,where=covered[...,None])
    else:sums[covered]/=weights[covered,None]
    eligible=np.clip(mapped,0,1)
    if output_protect is not None:eligible*=1-np.clip(output_protect,0,1)
    if output_scale==2:
        blend=weights.copy();blend/=.25;np.clip(blend,0,1,out=blend);blend*=eligible
        sums-=original;sums*=blend[...,None];sums+=original;result=sums
    else:
        blend=eligible*np.clip(weights/.25,0,1)
        result=original+(sums-original)*blend[...,None]
    result[blend<=0]=original[blend<=0]
    if fingerprint(backend,'EDIT',progress)!=expected:
        raise ValueError('The editing backend changed during surface reconstruction')
    if output_scale==2:
        evidence=weights
        for y in range(0,len(result),128):evidence[y:y+128]=np.linalg.norm(result[y:y+128]-original[y:y+128],axis=2)
    else:evidence=np.linalg.norm(result-original,axis=2)
    return np.clip(result,0,1),evidence,{
        'method':'LEARNED','prior':'EDIT','layout':'SURFACE','invention':invention,
        'output_scale':output_scale,'output_resolution':[w*output_scale,h*output_scale],
        'learned_2x_detail_retained':output_scale==2,
        'backend':expected,'backend_verification':'worker_sha256_before_and_after',
        'preserve_structure':preserve_structure,'planning':planning,'patches':details,
        'eligible_coverage_fraction':float(np.sum(covered&(eligible>0))/max(np.sum(eligible>0),1)),
        'camera_depth_from_actual_mesh':True,'camera_normals_from_actual_mesh':True,
        'guides_used_as_trained_controlnet':False,'physical_albedo_recovered':False,
        'continuous_context_constraints':True,'selection_applied_at_reprojection':True,
        'generated_detail_is_ground_truth':False,'shared_surface_reprojection':True,
        'outside_selected_skin_max_delta':float(np.abs(result[eligible<=0]-original[eligible<=0]).max())
            if np.any(eligible<=0) else 0.}
