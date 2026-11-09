"""Run the reconstruction solver on owned image inputs, without Blender/network."""
import hashlib
import json
import shutil
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from skin_reconstruction_core import reconstruct, inferred_relief, apply_relief, repair_normals
from worker_lifetime import watch_parent


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def load(folder, record, size=None,*,plane=False):
    path = (folder / record['file']).resolve()
    if not path.is_relative_to(folder) or digest(path) != record['sha256']:
        raise ValueError('Skin solver input is outside its job or has changed')
    image = Image.open(path).convert('RGB')
    if plane:image=image.getchannel('R')
    if min(image.size) < 64 or max(image.size) > 8192:
        raise ValueError('Skin solver inputs must be from 64 pixels through 8K')
    if size and image.size != size:
        image = image.resize(size, Image.Resampling.BILINEAR)
    return np.asarray(image, dtype=np.float32)/255


def linear(values):
    return np.where(values <= .04045, values/12.92, ((values+.055)/1.055)**2.4)


def save(values, path, color=False):
    if color:
        values = np.where(values <= .0031308, values*12.92, 1.055*np.maximum(values, 0)**(1/2.4)-.055)
    Image.fromarray(np.rint(np.clip(values, 0, 1)*255).astype(np.uint8)).save(path)


def run(request_path,*,owner=None):
    folder = request_path.resolve().parent
    request = json.loads(request_path.read_text(encoding='utf-8'))
    if request.get('schema') != 1:
        raise ValueError('Unsupported skin solver request')
    if request.get('parent'):
        watch_parent(request['parent'],folder/'cancel')
    elif owner:
        watch_parent(owner,folder/'cancel')
    started = time.monotonic()
    from color_completion_worker import eligible,run as complete_color_only
    if eligible(request):
        complete_color_only(folder,request,started)
        return
    color_cleanup = bool(request.get('color_cleanup',True))
    method=request.get('method','REFERENCE')
    source = load(folder, request['inputs']['albedo'])
    seam_geometry=None
    size = (source.shape[1], source.shape[0])
    reference = load(folder, request['inputs']['reference'], None if method=='GENERATIVE' else size) if color_cleanup and method in {'REFERENCE','GENERATIVE'} else source
    if request['inputs'].get('mask'):
        mask = load(folder, request['inputs']['mask'], size,plane=True)
    else:
        mask = (source.max(2) > .01).astype(np.float32)
    normal = reference_normal = None
    protection=color_protection=anatomy_color_mask=None;anatomy_report=None
    relief_strength = float(request.get('relief_strength', 0))
    surface_repair = bool(request.get('surface_repair', False))
    steps=request.get('normal_steps',[])
    native_normal = bool(steps or relief_strength or surface_repair)
    if relief_strength and not request['inputs'].get('normal'):
        raise ValueError('Inferred relief requires an existing tangent normal map')
    if request['inputs'].get('normal') and not (steps and request['inputs'].get('retained_normal')):
        normal = load(folder, request['inputs']['normal'], None if native_normal else size)
        if not steps and (not native_normal or surface_repair):
            reference_normal = load(folder, request['inputs']['reference_normal'], None if native_normal else size)
    phase='Restoring color'
    def progress(done, total, phase_name=None):
        temp = folder / 'progress.tmp'
        try:
            temp.write_text(json.dumps({'status':'running','phase':phase_name or phase,'tiles_done':done,'tiles_total':total}))
            temp.replace(folder / 'progress.json')
        except OSError:
            pass  # Windows readers may briefly hold the status file open.
    if color_cleanup and method in {'LEARNED','SEAMS','GENERATIVE','MATERIAL','LOCAL_FILL'}:
        if (method in {'LEARNED','LOCAL_FILL'} and not request['inputs'].get('mask')) or not request['inputs'].get('valid'):
            raise ValueError('Texture restoration needs its repair/UV coverage inputs')
        record=request['inputs']['valid']
        uv_path=(folder/record['file']).resolve()
        if not uv_path.is_relative_to(folder) or digest(uv_path)!=record['sha256']:
            raise ValueError('UV coverage input changed after submission')
        with np.load(uv_path,allow_pickle=False) as data:
            triangles=data['triangles']
            anatomy_weights=data['anatomy_protection'] if 'anatomy_protection' in data else None
            anatomy_color=data['anatomy_color_match'] if 'anatomy_color_match' in data else None
            anatomy_neck=data['anatomy_neck_anchor'] if 'anatomy_neck_anchor' in data else None
            surface_geometry=(data['points'],data['faces'],triangles) if 'points' in data and 'faces' in data else None
            completion_geometry=(data['points'],data['completion_faces'],data['completion_triangles']) if 'points' in data and 'completion_faces' in data and 'completion_triangles' in data else surface_geometry
            if 'seam_edges' in data and 'seam_centers' in data:
                seam_geometry=(data['seam_edges'],data['seam_centers'])
        if triangles.ndim!=3 or triangles.shape[1:]!=(3,2) or not np.isfinite(triangles).all():
            raise ValueError('UV coverage needs finite triangles')
        coverage=Image.new('L',size);draw=ImageDraw.Draw(coverage)
        for tri in triangles:
            draw.polygon([(float(u)*(size[0]-1),(1-float(v))*(size[1]-1)) for u,v in tri],fill=255)
        valid=np.asarray(coverage,dtype=np.float32)/255
        coverage.save(folder/'valid.png');coverage.close()
        protection=load(folder,request['inputs']['protect'],size,plane=True) if request['inputs'].get('protect') else None
        color_protection=protection
        if request.get('anatomy_protection'):
            if anatomy_weights is None or anatomy_weights.shape!=(len(triangles),3) or not np.isfinite(anatomy_weights).all() or anatomy_weights.min()<0 or anatomy_weights.max()>1:
                raise ValueError('Body anatomy protection snapshot is missing or invalid')
            from skin_anatomy import rasterize
            anatomy=rasterize(triangles,anatomy_weights,size)
            save(anatomy,folder/'anatomy-protect.png')
            if anatomy_color is not None and anatomy_neck is not None:
                if any(v.shape!=anatomy_weights.shape or not np.isfinite(v).all() or v.min()<0 or v.max()>1 for v in (anatomy_color,anatomy_neck)):
                    raise ValueError('Invalid anatomical color protection fields')
                anatomy_color_mask=rasterize(triangles,anatomy_color,size)
                neck=rasterize(triangles,anatomy_neck,size)
                color_protection=np.maximum(color_protection,neck) if color_protection is not None else neck
            else:color_protection=np.maximum(color_protection,anatomy) if color_protection is not None else anatomy
            protection=np.maximum(protection,anatomy) if protection is not None else anatomy
            anatomy_report=dict(request['anatomy_protection'],fully_protected_source_texels=int(np.count_nonzero(anatomy>=1)),
                                protected_source_texels=int(np.count_nonzero(anatomy>0)))
            del anatomy,anatomy_weights
    if color_cleanup and float(request.get('strength',1))==0 and request['inputs'].get('accepted') and request.get('output_scale',1)==1:
        normals=None;influence=np.zeros(mask.shape,np.float32)
        # The publication blend below normally replaces the source base.
        # Start from source here so its strength-zero result is accepted exactly.
        color=linear(source)
        report={'method':method,'inference_skipped':True,'zero_strength':True}
    elif color_cleanup and method=='LOCAL_FILL':
        from texture_region_completion import complete_color,guarded_region
        radius=float(request.get('fill_radius',7))
        if not np.isfinite(radius) or not 1<=radius<=64:raise ValueError('Color fill radius must be from 1 through 64 texels')
        progress(0,1,'Completing selected color')
        # Protect atlas chart borders instead of using padding as skin donors.
        completion_method=request.get('fill_method','TELEA');completion_valid=valid
        if completion_method=='SURFACE' and completion_geometry is not None and completion_geometry is not surface_geometry:
            coverage=Image.new('L',size);draw=ImageDraw.Draw(coverage)
            for triangle in completion_geometry[2]:draw.polygon([(float(u)*(size[0]-1),(1-float(v))*(size[1]-1)) for u,v in triangle],fill=255)
            completion_valid=np.asarray(coverage,dtype=np.float32)/255;coverage.close()
        eligible,guard=guarded_region(mask,completion_valid,radius=radius,seam_edges=seam_geometry[0] if seam_geometry is not None else None,
                                     geometry=completion_geometry if completion_method=='SURFACE' else None)
        proposal,report=complete_color(np.rint(source*255).astype(np.uint8),eligible,unknown=mask,
            radius=radius,strength=float(request.get('strength',1)),method=completion_method,geometry=completion_geometry)
        color=linear(proposal.astype(np.float32)/255);normals=None
        influence=np.max(np.abs(proposal.astype(np.float32)/255-source),axis=2)
        report.update(method='LOCAL_FILL',completion_method=report['method'],physical_albedo_recovered=False,
            generated_detail_is_ground_truth=False,**guard)
        del proposal,eligible
    elif color_cleanup and method=='MATERIAL':
        color=linear(source);normals=None;influence=np.zeros(mask.shape,np.float32)
        report=dict(method='MATERIAL',material_from_current_albedo=True,model_inference_performed=False,
            source_albedo_sha256=request['inputs']['albedo']['sha256'],physical_albedo_recovered=False,
            generated_detail_is_ground_truth=False)
    elif color_cleanup and method=='GENERATIVE':
        from texture_restore_core import resize
        target=(size[0]*int(request.get('output_scale',1)),size[1]*int(request.get('output_scale',1)))
        color=linear(reference)
        if color.shape[:2]!=(target[1],target[0]):color=np.clip(resize(color,target),0,1)
        baseline=linear(source)
        if baseline.shape!=color.shape:baseline=np.clip(resize(baseline,target),0,1)
        region=np.asarray(Image.fromarray(mask*valid).resize(target,Image.Resampling.BILINEAR),dtype=np.float32)
        coverage_report=None
        anatomy_color_report=None
        if request.get('role')=='BODY':
            from generated_coverage import fill_exterior
            coverage=Image.new('L',target);draw=ImageDraw.Draw(coverage)
            for tri in triangles:draw.polygon([(float(u)*(target[0]-1),(1-float(v))*(target[1]-1)) for u,v in tri],fill=255)
            domain=np.asarray(coverage,dtype=np.float32)/255
            selected=region.copy()
            if protection is not None:selected*=1-np.asarray(Image.fromarray(protection).resize(target,Image.Resampling.BILINEAR),dtype=np.float32)
            coverage_report=fill_exterior(color,domain,selected)
            del coverage,domain,selected
            if anatomy_color_mask is not None:
                from skin_anatomy import match_color
                color_weights=np.asarray(Image.fromarray(anatomy_color_mask).resize(target,Image.Resampling.BILINEAR),dtype=np.float32)
                anatomy_color_report=match_color(color,baseline,color_weights)
                del color_weights
        color-=baseline;color*=region[...,None]*float(request.get('strength',1));color+=baseline
        del baseline,region,reference
        normals=None;influence=np.zeros(color.shape[:2],np.float32)
        report={'method':'GENERATIVE','generated_detail_is_ground_truth':False,'physical_albedo_recovered':False,
            'generated_input_sha256':request['inputs']['reference']['sha256'],'source_structure_lock':False}
        if coverage_report:report['generated_uv_coverage']=coverage_report
        if anatomy_color_report:report['anatomy_color_match']=anatomy_color_report
    elif color_cleanup and method=='SEAMS':
        if seam_geometry is None:raise ValueError('Seam matching needs actual paired mesh edges')
        color=linear(source);normals=None;influence=np.zeros(mask.shape,np.float32)
        report={'method':'SEAMS','physical_albedo_recovered':False,'generated_detail_is_ground_truth':False}
    elif color_cleanup and method=='LEARNED':
        from texture_restore_backend import candidate
        config=request['learned'];color,normals=None,None
        parameters=dict(backend=config['backend'],expected=config['expected'],folder=folder,
            strength=float(request.get('strength',.65)),invention=float(config.get('invention',.35)),
            protect=protection,progress=progress,brief=config.get('brief','Fine organic skin detail'),
            preserve_structure=bool(config.get('preserve_structure',True)))
        parameters['patterns']=bool(config.get('patterns',False))
        if config.get('prior')=='PISA':
            from texture_sr_restore import candidate as sr_candidate
            color,influence,report=sr_candidate(source,mask,valid,backend=config['backend'],expected=config['expected'],folder=folder,
                strength=parameters['strength'],invention=parameters['invention'],protect=protection,
                brief=parameters['brief'],progress=progress,output_scale=int(request.get('output_scale',1)))
        elif config.get('prior')=='EDIT' and config.get('projection')=='SURFACE':
            if surface_geometry is None:raise ValueError('Surface editing needs a new snapshot of the target mesh')
            from texture_surface_restore import candidate as surface_candidate
            color,influence,report=surface_candidate(source,mask,valid,surface_geometry,**parameters,
                output_scale=int(request.get('output_scale',1)))
        else:
            color,influence,report=candidate(source,mask,valid,**parameters,
                gpu=int(config.get('gpu',-1)),prior=config.get('prior','RESTORE'))
    elif color_cleanup:
        linear_source=linear(source)
        color, normals, influence, report = reconstruct(linear_source, linear(reference) if reference is not source else linear_source, mask,
            strength=float(request.get('strength', 1)), normal=None if native_normal else normal,
            reference_normal=None if native_normal else reference_normal, cleanup_floor=float(request.get('cleanup_floor',0)),
            method=method,tile=128 if method=='PERIODIC' else 256,progress=progress)
    else:
        color, normals = linear(source), None
        influence = np.zeros(source.shape[:2],dtype=np.float32)
        report = {'color_cleanup':False,'color_retained':True,'physical_albedo_recovered':False}
    if color.shape[:2]!=source.shape[:2]:
        from texture_restore_core import resize,encoded
        size=(color.shape[1],color.shape[0])
        def plane(values):return np.asarray(Image.fromarray(values).resize(size,Image.Resampling.BILINEAR),dtype=np.float32)
        mask=plane(mask)
        source=encoded(np.clip(resize(linear(source),size),0,1))
        if seam_geometry is not None:
            coverage=Image.new('L',size);draw=ImageDraw.Draw(coverage)
            for tri in triangles:draw.polygon([(float(u)*(size[0]-1),(1-float(v))*(size[1]-1)) for u,v in tri],fill=255)
            valid=np.asarray(coverage,dtype=np.float32)/255
            coverage.save(folder/'valid.png');coverage.close()
            protection=plane(protection) if protection is not None else None
            color_protection=plane(color_protection) if color_protection is not None else None
        report['native_atlas_resolution_retained']=False
    accepted_record=request['inputs'].get('accepted')
    accepted=linear(load(folder,accepted_record)) if accepted_record else linear(source)
    if accepted.shape[:2]!=color.shape[:2]:
        from texture_restore_core import resize
        accepted=np.clip(resize(accepted,size),0,1)
    if color_cleanup and accepted_record:
        # The source is a selected repair input, not the publication baseline.
        # Subtract only the unselected source base: the solver has already
        # applied its mask/strength to the actual cleanup residual. Multiplying
        # that residual by the mask again would incorrectly square feathering.
        mix=np.clip(mask,0,1)*float(request.get('strength',1))
        color+=(accepted-linear(source))*(1-mix[...,None])
        np.clip(color,0,1,out=color)
        color[mix<=0]=accepted[mix<=0]
        if not np.any(mix>0):influence.fill(0)
        report['publication_base']='accepted_albedo'
        report['accepted_base_sha256']=accepted_record['sha256']
        report['source_selection_strength']=float(request.get('strength',1))
    else:report['publication_base']='submitted_input_legacy'
    if request['inputs'].get('protect') and protection is None:
        protection=load(folder,request['inputs']['protect'],size,plane=True)
        color_protection=protection
    if color_protection is not None:
        color+=(accepted-color)*color_protection[...,None]
        color[color_protection>=1]=accepted[color_protection>=1]
        report['accepted_paint_protected']=True
    if anatomy_report:report['anatomy_protection']=anatomy_report
    shared_height=None
    if request.get('surface_balance'):
        from skin_material_core import split_surface
        shared_height,split_report=split_surface(color,mask*float(request.get('strength',1)),balance=float(request['surface_balance']),
            height_strength=float(request.get('relief_strength',.7)),radius=float(request.get('relief_radius',2))*int(request.get('output_scale',1)),
            protect=protection,valid=valid,progress=progress)
        np.save(folder/'material-height.npy',shared_height,allow_pickle=False)
        report['surface_redistribution']=split_report
    if seam_geometry is not None:
        from texture_restore_core import encoded
        from texture_seams import inspect,reconcile
        if not report.get('inference_skipped') and (method!='LOCAL_FILL' or request.get('fill_method')=='SURFACE'):
            eligible=mask*valid
            if color_protection is not None:eligible*=1-color_protection
            budget=.25*float(request.get('strength',1))
            if method=='LEARNED':budget*=float(request['learned'].get('invention',.35))
            report['seam_reconciliation']=reconcile(color,*seam_geometry,eligible=eligible,
                domain=valid,max_delta=budget,progress=progress,evidence=influence)
        seams=inspect(np.clip(encoded(accepted),0,1),np.clip(encoded(color),0,1),*seam_geometry,eligible=mask)
        report['uv_seams']=seams
        report['seam_comparison_base']=report['publication_base']
        # Existing source seams are measured separately, not claimed repaired.
        introduced=seams.get('seam_regression',{})
        report['new_seam_guard_passed']=bool(not len(seam_geometry[0]) or
            (seams.get('continuity_measured') and introduced.get('p95',1)<=2/255 and introduced.get('max',1)<=8/255))
        if (request.get('surface_balance') or (method=='LOCAL_FILL' and request.get('fill_method')=='SURFACE')) and not report['new_seam_guard_passed']:
            from texture_seams import retain_regressed_anchors
            previous=np.clip(encoded(accepted),0,1);updated=np.clip(encoded(color),0,1)
            report['seam_anchor_fallback']=retain_regressed_anchors(previous,updated,*seam_geometry)
            seams=inspect(previous,updated,*seam_geometry,eligible=mask)
            color=linear(updated);del previous,updated
            report['uv_seams']=seams;introduced=seams.get('seam_regression',{})
            report['new_seam_guard_passed']=bool(seams.get('continuity_measured') and introduced.get('p95',1)<=2/255 and introduced.get('max',1)<=8/255)
    save(color,folder/'candidate-albedo.png',color=True)
    reused_normal=False
    if (steps and request['inputs'].get('retained_normal') and accepted_record and
            np.array_equal(np.asarray(Image.open(folder/'candidate-albedo.png').convert('RGB')),
                           np.asarray(Image.open(folder/accepted_record['file']).convert('RGB')))):
        progress(0,1,'Retaining accepted shading')
        record=request['inputs']['retained_normal']
        retained=(folder/record['file']).resolve()
        if not retained.is_relative_to(folder) or digest(retained)!=record['sha256']:
            raise ValueError('The retained normal changed after submission')
        shutil.copy2(retained,folder/'candidate-normal.png')
        reused_normal=True
        report['normal_recipe_reused_unchanged_albedo']=True
        report['normal_steps']=[{'kind':step['kind']} for step in steps]
        with Image.open(retained) as image:report['normal_resolution']=list(image.size)
    # Color/model generation and shading replay are sequential. Do not keep
    # two additional 8K RGB publication baselines alive during normal baking.
    untouched_delta=0.
    for y in range(0,len(color),128):
        clear=mask[y:y+128]<=0
        if np.any(clear):
            untouched_delta=max(untouched_delta,float(np.abs(color[y:y+128][clear]-accepted[y:y+128][clear]).max()))
    del accepted,source
    def accepted_surface(kind,packed=False,target=None):
        record=request['inputs']['region_'+kind];path=(folder/record['file']).resolve()
        if not path.is_relative_to(folder) or digest(path)!=record['sha256']:
            raise ValueError('The accepted regional material changed after submission')
        with Image.open(path) as image:
            if packed:
                if image.mode not in ('RGB','RGBA'):return None
            else:image=image.convert('RGB') if kind=='normal' else image.convert('RGB').getchannel('R')
            target=target or size
            if image.size!=target:image=image.resize(target,Image.Resampling.BILINEAR)
            return np.asarray(image,dtype=np.uint8)
    def retain_region(values,kind,restore_only=False):
        from skin_material_core import preserve_region_base
        target=(values.shape[1],values.shape[0])
        def gate(value):
            if value is None or value.shape==values.shape[:2]:return value
            return np.asarray(Image.fromarray(value).resize(target,Image.Resampling.BILINEAR),dtype=np.float32)
        detail=preserve_region_base(values,accepted_surface(kind,target=target),gate(mask),protect=gate(protection),
            valid=gate(valid),strength=float(request.get('strength',1)),normal=kind=='normal',restore_only=restore_only)
        detail['resolution']=list(target)
        report.setdefault('regional_material_preservation',{})[kind]=detail
    if steps and not reused_normal:
        phase='Replaying shading'
        if normal is None:normal=load(folder,request['inputs']['normal'])
        if normal is None:raise ValueError('Skin-normal replay requires its clean baseline')
        if request.get('preserve_material_region'):
            with Image.open(folder/request['inputs']['region_normal']['file']) as image:accepted_size=image.size
            target=(max(size[0],normal.shape[1],accepted_size[0]),max(size[1],normal.shape[0],accepted_size[1]))
            if normal.shape[:2]!=(target[1],target[0]):normal=load(folder,request['inputs']['normal'],target)
        normals=normal
        if request.get('preserve_material_region'):retain_region(normals,'normal')
        report['normal_recipe_replayed_from_baseline']=True
        report['normal_steps']=[]
        for index,step in enumerate(steps):
            if step['kind']=='RETAIN':
                detail={'kind':'RETAIN','immutable_accepted_surface':True}
            elif step['kind']=='SURFACE_PATCH':
                from skin_normal_patch import apply as apply_patch
                def patch_input(key):
                    record=request['inputs'][step[key]];path=(folder/record['file']).resolve()
                    if not path.is_relative_to(folder) or digest(path)!=record['sha256']:raise ValueError('The retained regional normal changed')
                    with Image.open(path) as image:
                        image=image.convert('RGB');target=(normals.shape[1],normals.shape[0])
                        if image.size!=target:image=image.resize(target,Image.Resampling.BILINEAR)
                        return np.asarray(image,dtype=np.uint8)
                before,after=patch_input('before'),patch_input('after')
                normals=apply_patch(normals,before,after,progress=progress);del before,after
                detail={'kind':'SURFACE_PATCH','method':'shortest_arc_normal_transport'}
            elif step['kind']=='HEIGHT':
                if step['field']=='generated_height':
                    if shared_height is None:raise ValueError('The shared generated material height is missing')
                    field=shared_height
                else:
                    record=request['inputs'][step['field']];path=(folder/record['file']).resolve()
                    if not path.is_relative_to(folder) or digest(path)!=record['sha256']:raise ValueError('The retained material height changed')
                    field=np.load(path,allow_pickle=False)
                if field.ndim!=2 or field.shape!=color.shape[:2] or not np.isfinite(field).all() or abs(field).max()>16:
                    raise ValueError('Invalid retained material height')
                normals=apply_relief(normals,field,directx=bool(request.get('directx',True)),progress=progress)
                detail={'kind':'HEIGHT','immutable_material_field':True,'height_max_output_texels':float(abs(field).max())}
            elif step['kind']=='REPAIR':
                region=load(folder,request['inputs'][step['mask']],size,plane=True)
                ref=load(folder,request['inputs'][step['reference']])
                normals=repair_normals(normals,ref,region*float(step['strength']))
                del ref
                detail={'kind':'REPAIR','strength':step['strength']}
            elif step['kind']=='RELIEF':
                layer_color=load(folder,request['inputs'][step['color']]) if step.get('color') else color
                layer_size=(layer_color.shape[1],layer_color.shape[0])
                region=load(folder,request['inputs'][step['mask']],layer_size,plane=True)
                evidence=np.zeros_like(region) if step.get('color') else influence
                height,gate,detail=inferred_relief(layer_color,region,evidence,
                    radius=float(step['radius']),strength=float(step['strength']))
                normals=apply_relief(normals,height,directx=bool(request.get('directx',True)),progress=progress)
                save(gate,folder/'relief-mask.png')
                detail.update(kind='RELIEF')
                del height,gate,layer_color,evidence
            else:raise ValueError('Unknown skin-normal replay operation')
            if step.get('quantize_after'):
                np.clip(normals,0,1,out=normals)
                normals*=255;np.rint(normals,out=normals);normals/=255
            report['normal_steps'].append(detail)
            progress(index+1,len(steps))
        report['normal_resolution']=[normal.shape[1],normal.shape[0]]
    if surface_repair and not steps:
        if not request['inputs'].get('normal_mask'):
            raise ValueError('Surface repair requires a separate damaged-normal mask')
        normal_mask = load(folder,request['inputs']['normal_mask'],size,plane=True)
        normals = repair_normals(normal,reference_normal,normal_mask*float(request.get('strength',1)))
        report['normal_surface_repair'] = 'artist-mask reference replacement'
    if relief_strength and not steps:
        height, gate, relief_report = inferred_relief(color,mask,influence,
            radius=float(request.get('relief_radius',3)),strength=relief_strength)
        normals = apply_relief(normals if normals is not None else normal,height,directx=bool(request.get('directx',True)),progress=progress)
        save(gate,folder/'relief-mask.png')
        report['relief'] = relief_report
        report['normal_resolution'] = [normal.shape[1],normal.shape[0]]
    if normals is not None:
        if request.get('preserve_material_region'):retain_region(normals,'normal',restore_only=True)
        progress(0,1,'Saving skin normal')
        save(normals,folder/'candidate-normal.png')
    # Full-resolution expression colors do not use normals. Release both the
    # replay baseline and result before allocating their 8K RGB tint buffers.
    normals=normal=None
    bloodflow=request.get('bloodflow',{})
    flow_files=[]
    for slot in bloodflow.get('slots',[]):
        progress(len(flow_files),len(bloodflow['slots']),'Generating expression colors')
        if slot not in {'CM1','CM2','CM3'}:raise ValueError('Invalid blood-flow slot')
        # Use the final quantized albedo, matching the actual delivered base.
        # Full albedo resolution avoids the old DEFAULT 512px color bottleneck.
        from surface_maps import tint_bloodflow_in_place,native_expression_color_in_place
        pixels=linear(np.asarray(Image.open(folder/'candidate-albedo.png').convert('RGB'),dtype=np.float32)/255)
        native_reference=request['inputs'].get('native_expression_'+slot)
        if native_reference:native_expression_color_in_place(pixels,load(folder,native_reference,size))
        region=load(folder,request['inputs']['bloodflow_'+slot],size).mean(2)
        tint_bloodflow_in_place(pixels,region,float(bloodflow['strength']))
        name='candidate-'+slot+'.png'
        save(pixels,folder/name,color=True)
        flow_files.append(name)
        del pixels,region
    if flow_files:
        report['bloodflow']={'slots':bloodflow['slots'],'resolution':list(size),
            'source_albedo_sha256':digest(folder/'candidate-albedo.png'),
            'physiological_recovery':False,
            'method':'native_expression_retarget' if any(request['inputs'].get('native_expression_'+slot) for slot in bloodflow['slots']) else 'native_region_tint'}
    material_files=[]
    if request.get('material_strength'):
        normals=normal=None
        from skin_material_core import derive
        def gray(key):
            record=request['inputs'][key];path=(folder/record['file']).resolve()
            if not path.is_relative_to(folder) or digest(path)!=record['sha256']:raise ValueError('Material baseline changed')
            with Image.open(path) as image:
                image=image.convert('RGB').getchannel('R')
                if request.get('preserve_material_region'):
                    with Image.open(folder/request['inputs']['region_'+key.removeprefix('material_')]['file']) as accepted_image:
                        target=(max(size[0],image.width,accepted_image.width),max(size[1],image.height,accepted_image.height))
                    if image.size!=target:image=image.resize(target,Image.Resampling.BILINEAR)
                return np.asarray(image,dtype=np.float32)/255
        rough_base,cavity_base=gray('material_roughness'),gray('material_cavity')
        if request.get('preserve_material_region'):
            retain_region(rough_base,'roughness');retain_region(cavity_base,'cavity')
        rough,cavity,material_report=derive(color,mask,rough_base,cavity_base,
            strength=float(request['material_strength']),protect=protection,progress=progress,height=shared_height)
        for kind,value in (('roughness',rough),('cavity',cavity)):
            name='candidate-'+kind+'.png'
            if request.get('preserve_material_region'):retain_region(value,kind,restore_only=True)
            packed=accepted_surface(kind,packed=True,target=(value.shape[1],value.shape[0])) if request.get('preserve_material_region') else None
            if packed is not None:
                from skin_material_core import retain_packed_channels
                Image.fromarray(retain_packed_channels(value,packed)).save(folder/name)
                report.setdefault('retained_packed_material_channels',{})[kind]='GBA' if packed.shape[2]==4 else 'GB'
                del packed
            else:save(value,folder/name)
            material_files.append(name)
        report['derived_material']=dict(material_report,source_albedo_sha256=digest(folder/'candidate-albedo.png'))
        del rough,cavity
    save(influence, folder / 'artifact-evidence.png')
    files = ['candidate-albedo.png','artifact-evidence.png'] + (['candidate-normal.png'] if (folder/'candidate-normal.png').is_file() else []) + flow_files+material_files
    if shared_height is not None:files.append('material-height.npy')
    if anatomy_report:files.append('anatomy-protect.png')
    if not reused_normal and ((relief_strength and not steps) or any(step['kind']=='RELIEF' for step in steps)):files.append('relief-mask.png')
    report.update(status='ready', schema=1, seconds=time.monotonic()-started,
        output_resolution=[color.shape[1],color.shape[0]],
        inputs=request['inputs'], outputs={name:digest(folder/name) for name in files},
        untouched_mask_max_delta=untouched_delta,
        maps_added=(['ALBEDO'] if color_cleanup else [])+(['NORMAL'] if (folder/'candidate-normal.png').is_file() else [])+bloodflow.get('slots',[])+(['ROUGHNESS','CAVITY'] if material_files else []),
        roughness_synthesized=bool(material_files), cavity_synthesized=bool(material_files), lighting_removal_verified=False)
    from resource_budget import process_memory
    report['worker_memory']=process_memory()
    temp = folder / 'result.tmp'
    temp.write_text(json.dumps(report, indent=2), encoding='utf-8')
    temp.replace(folder / 'result.json')


if __name__ == '__main__':
    try:
        run(Path(sys.argv[1]))
    except Exception as exc:
        path = Path(sys.argv[1]).resolve().parent / 'result.json'
        path.write_text(json.dumps({'schema':1,'status':'failed','error':str(exc)}),encoding='utf-8')
        raise
