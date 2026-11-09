"""Bounded color-only path for the existing skin candidate/publication workflow."""
import hashlib,json,time
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw
from texture_region_completion import complete_color,guarded_region


def eligible(request):
    inputs=request.get('inputs',{})
    return (request.get('color_cleanup',True) and request.get('method')=='LOCAL_FILL'
            and request.get('fill_method','TELEA')!='SURFACE'
            and inputs.get('accepted',{}).get('sha256')==inputs.get('albedo',{}).get('sha256')
            and not inputs.get('normal') and not request.get('normal_steps')
            and not request.get('relief_strength') and not request.get('surface_repair')
            and not request.get('material_strength') and not request.get('surface_balance')
            and not request.get('bloodflow',{}).get('slots') and request.get('output_scale',1)==1)


def run(folder,request,started):
    """Do not allocate the joint solver's full float color/normal atlas planes."""
    folder=Path(folder)
    def digest(path):
        with path.open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()
    def path(record):
        result=(folder/record['file']).resolve()
        if not result.is_relative_to(folder.resolve()) or digest(result)!=record['sha256']:raise ValueError('Color completion input changed or is outside its job')
        return result
    def plane(record,size):
        with Image.open(path(record)) as image:
            image=image.convert('RGB').getchannel('R')
            if image.size!=size:image=image.resize(size,Image.Resampling.BILINEAR)
            return np.asarray(image,dtype=np.float32)/255
    inputs=request['inputs']
    if not inputs.get('mask') or not inputs.get('valid'):raise ValueError('Local completion needs explicit repair and UV masks')
    path(inputs['accepted'])
    with Image.open(path(inputs['albedo'])) as image:
        size=image.size
        if min(size)<64 or max(size)>8192:raise ValueError('Color completion supports skin atlases from 64 texels through 8K')
        source=np.asarray(image.convert('RGB')).copy()
    mask=plane(inputs['mask'],size);protect=plane(inputs['protect'],size) if inputs.get('protect') else None
    method=request.get('fill_method','TELEA');geometry=None
    with np.load(path(inputs['valid']),allow_pickle=False) as data:
        triangles=data['completion_triangles'] if method=='SURFACE' and 'completion_triangles' in data else data['triangles'];seams=data['seam_edges'] if 'seam_edges' in data else None
        if method=='SURFACE':
            if 'points' not in data or 'faces' not in data:raise ValueError('Recapture actual mesh geometry for surface color completion')
            geometry=(data['points'],data['completion_faces'] if 'completion_faces' in data else data['faces'],triangles)
    if triangles.ndim!=3 or triangles.shape[1:]!=(3,2) or not np.isfinite(triangles).all():raise ValueError('Color completion needs valid mesh UV triangles')
    coverage=Image.new('L',size);draw=ImageDraw.Draw(coverage)
    for tri in triangles:draw.polygon([(float(u)*(size[0]-1),(1-float(v))*(size[1]-1)) for u,v in tri],fill=255)
    valid=np.asarray(coverage);coverage.save(folder/'valid.png')
    radius=float(request.get('fill_radius',7));strength=float(request.get('strength',1))
    region,guard=guarded_region(mask,valid,radius=radius,seam_edges=seams)
    progress=folder/'progress.json';progress.write_text(json.dumps(dict(status='running',phase='Completing selected color',tiles_done=0,tiles_total=1)))
    result,report=complete_color(source,region,unknown=mask,radius=radius,strength=strength,method=method,geometry=geometry)
    if protect is not None:
        # Match the joint solver's linear-color protection blend, without
        # allocating any full float RGB atlas. Enabling derived maps must not
        # change how feathered artist protection affects the color candidate.
        for y in range(0,len(source),128):
            weight=protect[y:y+128,:,None]
            if not np.any(weight):continue
            old=source[y:y+128].astype(np.float32)/255;new=result[y:y+128].astype(np.float32)/255
            old=np.where(old<=.04045,old/12.92,((old+.055)/1.055)**2.4)
            new=np.where(new<=.04045,new/12.92,((new+.055)/1.055)**2.4)
            new+=(old-new)*weight
            encoded=np.where(new<=.0031308,new*12.92,1.055*np.maximum(new,0)**(1/2.4)-.055)
            result[y:y+128]=np.rint(encoded.clip(0,1)*255).astype(np.uint8)
            fixed=protect[y:y+128]>=1;result[y:y+128][fixed]=source[y:y+128][fixed]
    evidence=np.zeros(mask.shape,np.uint8);outside=guard_delta=protected_delta=0
    for y in range(0,len(source),128):
        delta=np.max(np.abs(result[y:y+128].astype(np.int16)-source[y:y+128]),axis=2);evidence[y:y+128]=delta
        clear=mask[y:y+128]<=0
        if clear.any():outside=max(outside,int(delta[clear].max()))
        fixed=region[y:y+128]<=0
        if fixed.any():guard_delta=max(guard_delta,int(delta[fixed].max()))
        if protect is not None:
            fixed=protect[y:y+128]>=1
            if fixed.any():protected_delta=max(protected_delta,int(delta[fixed].max()))
    if outside or guard_delta or protected_delta:raise ValueError('Color completion changed a protected texel; retain the accepted map')
    Image.fromarray(result).save(folder/'candidate-albedo.png');Image.fromarray(evidence).save(folder/'artifact-evidence.png')
    files=['candidate-albedo.png','artifact-evidence.png']
    report.update(status='ready',schema=1,method='LOCAL_FILL',completion_method=report['method'],
        changed_texels=int(np.count_nonzero(evidence)),
        color_only_bounded_path=True,seconds=time.monotonic()-started,inputs=inputs,
        outputs={name:digest(folder/name) for name in files},maps_added=['ALBEDO'],
        publication_base='accepted_albedo',accepted_base_sha256=inputs['accepted']['sha256'],
        source_selection_strength=strength,untouched_mask_max_delta=0.,accepted_paint_protected=protect is not None,
        new_seam_guard_passed=True,uv_seams=dict(method='UNCHANGED_UV_SEAM_BANDS',guard_band_max_byte_delta=guard_delta),
        roughness_synthesized=False,cavity_synthesized=False,physical_albedo_recovered=False,
        generated_detail_is_ground_truth=False,lighting_removal_verified=False,**guard)
    from resource_budget import process_memory
    report['worker_memory']=process_memory();temp=folder/'result.tmp';temp.write_text(json.dumps(report,indent=2));temp.replace(folder/'result.json')
