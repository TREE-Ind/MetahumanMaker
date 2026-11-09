"""Preserve fitted gaze shapes with drivers that do not depend on blinking.

Derives rest-space globe fields from native open-gaze evaluations. Each
cardinal gaze has its own scalar driver, so blink/squint cannot change its
shape coefficients. Native joint motion and pupil targets remain available.
"""
import hashlib,json,re,uuid
from pathlib import Path
import numpy as np


def digest(path):
    with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()


def prepare(reference_file,capture_folder,base_dna,base_mesh,base_asset_file,neutral_capture,destination,*,radius=.35,maximum_offset_cm=15.,channel_namespace=None,transport_mode='PRESERVE_FITTED',blink_captures=None):
    reference_file=Path(reference_file).resolve();capture_folder=Path(capture_folder).resolve();base_dna=Path(base_dna).resolve();base_asset_file=Path(base_asset_file).resolve();neutral_capture=Path(neutral_capture).resolve();destination=Path(destination).resolve()
    reference=json.loads(reference_file.read_text());capture=json.loads((capture_folder/'capture.json').read_text());base_meta=json.loads(neutral_capture.read_text())
    if destination.exists() or reference.get('coordinates')!='PRESERVED_DNA' or capture.get('status')!='complete' or base_meta.get('source_mesh')!=base_mesh:raise ValueError('Supply complete gaze training, current rigid-globe DNA and a fresh output folder')
    if not np.isfinite(radius) or radius<=0 or not np.isfinite(maximum_offset_cm) or maximum_offset_cm<=0:raise ValueError('Choose positive gaze interpolation and offset bounds')
    if transport_mode not in {'PRESERVE_FITTED','FIXED_CONVEX_ENVELOPE'}:raise ValueError('Choose fitted gaze reproduction or fixed neutral volume transport')
    if blink_captures is not None and (transport_mode!='FIXED_CONVEX_ENVELOPE' or set(blink_captures)!={.5,1.}):raise ValueError('Fixed-volume blinking needs native half/full blink captures')
    namespace=channel_namespace or 'CM_globe_gaze_'+uuid.uuid4().hex[:8]
    if not re.fullmatch(r'CM_[A-Za-z0-9_]{1,48}',namespace):raise ValueError('Choose a short CM_ namespace for the new gaze revision')
    meshes={m['name']:m for m in reference['meshes']};inputs={};asset_sha=digest(base_asset_file);poses={(row['lod'],row['direction'],row['value']):row for row in capture['poses']};directions=('Up','Down','Left','Right');values=(.25,.5,.75,1.);expected={(lod,d,v) for lod in range(reference['lod_count']) for d in directions for v in values}
    if len(poses)!=len(capture['poses']) or set(poses)!=expected:raise ValueError('Capture all four cardinal gaze samples at each native LOD')
    def record(path):
        path=Path(path).resolve();sha=digest(path)
        if str(path) in inputs and inputs[str(path)]!=sha:raise ValueError('A gaze-calibration input changed while preparing the fields')
        inputs[str(path)]=sha;return path
    def binary(parent,name,dtype):
        if Path(name).name!=name:raise ValueError('Native arrays must stay beside their metadata')
        return np.fromfile(record(parent/name),dtype=dtype)
    for path in (reference_file,capture_folder/'capture.json',base_dna,neutral_capture):record(path)
    mids=binary(neutral_capture.parent,base_meta['dna_mesh_file'],'<i4');vids=binary(neutral_capture.parent,base_meta['dna_vertex_file'],'<i4');rest=binary(neutral_capture.parent,base_meta['neutral_file'],'<f4').reshape(-1,3);head=meshes['head_lod0_mesh'];use=mids==head['index'];ids=vids[use];count=np.bincount(ids,minlength=head['vertex_count'])
    if len(count)!=head['vertex_count'] or np.any(count==0):raise ValueError('The native neutral lacks complete canonical head identities')
    canonical=np.column_stack([np.bincount(ids,weights=rest[use,j],minlength=len(count))/count for j in range(3)]);raw=binary(reference_file.parent,head['positions_file'],'<f4').reshape(-1,3);transform=np.linalg.lstsq(np.c_[raw,np.ones(len(raw))],canonical,rcond=None)[0];frame_error=float(np.linalg.norm(np.c_[raw,np.ones(len(raw))]@transform-canonical,axis=1).max())
    if frame_error>.0005 or abs(np.linalg.det(transform[:3]))<1e-12:raise ValueError('The captured native rest geometry does not match its DNA coordinate frame')
    for side in ('Left','Right'):
        mesh=meshes[f'eye{side}_lod0_mesh'];selected=mids==mesh['index'];ids=vids[selected];count=np.bincount(ids,minlength=mesh['vertex_count'])
        if len(count)!=mesh['vertex_count'] or np.any(count==0):raise ValueError('The base neutral omits canonical globe vertices')
        points=np.column_stack([np.bincount(ids,weights=rest[selected,j],minlength=len(count))/count for j in range(3)]);source=binary(reference_file.parent,mesh['positions_file'],'<f4').reshape(-1,3);error=float(np.linalg.norm(np.c_[source,np.ones(len(source))]@transform-points,axis=1).max())
        if error>.0005:raise ValueError('The fitted globe reference does not match the base neutral; retain the later artist geometry')
        frame_error=max(frame_error,error)
    fields={};native_ids={};maxima=[]
    for lod in range(reference['lod_count']):
        for direction in directions:
            for value in values:
                row=poses[lod,direction,value];file=Path(row['file']).resolve()
                if file.parent!=capture_folder:raise ValueError('A gaze sample leaves its capture folder')
                meta=json.loads(record(file).read_text());wanted={'eyeLook'+direction+side:value for side in ('L','R')}
                if row['controls']!=wanted or meta['lod']!=lod or meta['source_mesh']!=capture['source_mesh']:raise ValueError('A gaze capture has mismatched controls, rig or LOD')
                curves={name.lower().removeprefix('ctrl_expressions_'):weight for name,weight in meta['raw_curves'].items()}
                if any(not np.isfinite(weight) for weight in curves.values()) or any(abs(curves.get(name.lower(),float('inf'))-weight)>1e-6 for name,weight in wanted.items()) or any(abs(weight)>1e-6 for name,weight in curves.items() if name.startswith(('eyeblink','eyesquintinner'))):raise ValueError('Gaze training must contain the requested gaze with neutral blink and squint')
                weights={name.lower():weight for name,weight in meta['morphs'].items()}
                mid=binary(file.parent,meta['dna_mesh_file'],'<i4');vid=binary(file.parent,meta['dna_vertex_file'],'<i4')
                if lod not in native_ids:native_ids[lod]=(mid,vid)
                elif not np.array_equal(native_ids[lod][0],mid) or not np.array_equal(native_ids[lod][1],vid):raise ValueError('Native gaze vertex identities changed')
                for side in ('Left','Right'):
                    name=f'eye{side}_lod{lod}_mesh';mesh=meshes[name];field=np.zeros((mesh['vertex_count'],3));selected=mid==mesh['index'];counts=np.bincount(vid[selected],minlength=mesh['vertex_count'])
                    if len(counts)!=mesh['vertex_count'] or np.any(counts==0):raise ValueError('Native gaze samples omit canonical globe vertices')
                    for target in mesh['targets']:
                        if not target['channel'].startswith('CM_'):continue
                        weight=weights.get((name+'__'+target['channel']).lower(),0.)
                        if not np.isfinite(weight):raise ValueError('A native globe driver weight is not finite')
                        if abs(weight)<1e-10:continue
                        indices=binary(reference_file.parent,target['indices_file'],'<u4');delta=binary(reference_file.parent,target['deltas_file'],'<f4').reshape(-1,3)
                        if len(indices)!=len(delta) or (len(indices) and indices.max()>=len(field)):raise ValueError('A source globe corrective leaves its canonical mesh')
                        field[indices]+=delta*weight
                    if transport_mode=='FIXED_CONVEX_ENVELOPE':
                        from .ocular_globe_volume import transport_convex
                        from .calibration_math import canonical_offsets,rest_offsets
                        posed=binary(file.parent,meta['vertices_file'],'<f4').reshape(-1,3);linear=binary(file.parent,meta['skin_linear_file'],'<f4').reshape(-1,3,3);n=mesh['vertex_count'];selected=mid==mesh['index'];raw=binary(reference_file.parent,mesh['positions_file'],'<f4').reshape(-1,3);neutral=raw@transform[:3]+transform[3]
                        old_field=field[vid[selected]]@transform[:3];rigid=posed[selected]-np.einsum('ni,nij->nj',old_field,linear[selected]);canonical=canonical_offsets(rigid,np.full(selected.sum(),mesh['index'],dtype=np.int32),vid[selected],mesh['index'],n)[0];a=neutral-neutral.mean(0);b=canonical-canonical.mean(0);u,_,vh=np.linalg.svd(a.T@b);fix=np.ones(3);fix[-1]=np.linalg.det(u@vh);rotation=(u*fix)@vh
                        if np.linalg.norm(a@rotation-b,axis=1).max()>.0005:raise ValueError('Separate nonrigid globe effects before fixed-volume gaze calibration')
                        desired,proof=transport_convex(neutral,rotation);offset,_=rest_offsets(rigid,desired[vid[selected]],linear[selected]);folded,_=canonical_offsets(offset,np.full(selected.sum(),mesh['index'],dtype=np.int32),vid[selected],mesh['index'],n);field=folded@np.linalg.inv(transform[:3])
                    maximum=float(np.linalg.norm(field@transform[:3],axis=1).max(initial=0))
                    if not np.isfinite(field).all() or maximum>maximum_offset_cm:raise ValueError('A gaze shape exceeds its physical bound')
                    fields[lod,side,direction,value]=field;maxima.append(maximum)
    if blink_captures is not None:
        from .ocular_globe_volume import transport_convex
        from .calibration_math import canonical_offsets
        for value,blink_file in blink_captures.items():
            blink_file=Path(blink_file).resolve();meta=json.loads(record(blink_file).read_text());curves={name.lower().removeprefix('ctrl_expressions_'):weight for name,weight in meta['raw_curves'].items()}
            if meta['lod']!=0 or any(abs(curves.get('eyeblink'+side,float('inf'))-value)>1e-6 for side in ['l','r']) or any(abs(weight)>1e-6 for name,weight in curves.items() if name.startswith('eyelook')):raise ValueError('Supply native half/full blinks with neutral gaze')
            mid=binary(blink_file.parent,meta['dna_mesh_file'],'<i4');vid=binary(blink_file.parent,meta['dna_vertex_file'],'<i4');posed=binary(blink_file.parent,meta['vertices_file'],'<f4').reshape(-1,3);rest=binary(blink_file.parent,meta['neutral_file'],'<f4').reshape(-1,3);linear=binary(blink_file.parent,meta['skin_linear_file'],'<f4').reshape(-1,3,3)
            for side in ['Left','Right']:
                mesh=meshes[f'eye{side}_lod0_mesh'];use=mid==mesh['index'];n=mesh['vertex_count'];neutral=binary(reference_file.parent,mesh['positions_file'],'<f4').reshape(-1,3)@transform[:3]+transform[3];captured=canonical_offsets(rest,mid,vid,mesh['index'],n)[0];points=canonical_offsets(posed,mid,vid,mesh['index'],n)[0]
                if np.linalg.norm(captured-neutral,axis=1).max()>.0005:raise ValueError('The blink neutral does not match the fixed globe revision')
                a=neutral-neutral.mean(0);b=points-points.mean(0);u,_,vh=np.linalg.svd(a.T@b);fix=np.ones(3);fix[-1]=np.linalg.det(u@vh);rotation=(u*fix)@vh;translation=points.mean(0)-neutral.mean(0)@rotation
                if np.linalg.norm(a@rotation-b,axis=1).max()>.0005 or np.max(abs(linear[use]-rotation))>1e-5:raise ValueError('Blink input must have separated rigid globe motion before surface transport')
                for lod in range(reference['lod_count']):
                    mesh=meshes[f'eye{side}_lod{lod}_mesh'];neutral=binary(reference_file.parent,mesh['positions_file'],'<f4').reshape(-1,3)@transform[:3]+transform[3];desired,_=transport_convex(neutral,rotation);field=((desired-(neutral@rotation+translation))@rotation.T)@np.linalg.inv(transform[:3]);maximum=float(np.linalg.norm(field@transform[:3],axis=1).max())
                    if not np.isfinite(field).all() or maximum>maximum_offset_cm:raise ValueError('A blink volume field exceeds its physical bound')
                    fields[lod,side,'Blink',value]=field;maxima.append(maximum)
    if any(digest(Path(name))!=sha for name,sha in inputs.items()) or digest(base_asset_file)!=asset_sha:raise ValueError('A gaze-calibration source changed before the request was sealed')
    destination.mkdir(parents=True);root_targets=[];lower_targets=[];patches=[];solvers=[]
    for side in ('Left','Right'):
        suffix=side[0]
        for direction in (*directions,'Blink') if blink_captures is not None else directions:
            samples=(.5,1.) if direction=='Blink' else values
            rows=[dict(name=f'{namespace}_pose_{suffix}_{direction.lower()}_000',values=[0.])]
            for value in samples:
                step=round(value*100);channel=f'{namespace}_{suffix}_{direction.lower()}_{step:03d}';rows.append(dict(name=f'{namespace}_pose_{suffix}_{direction.lower()}_{step:03d}',values=[value],channel=channel))
                for lod in range(reference['lod_count']):
                    name=f'eye{side}_lod{lod}_mesh';mesh=meshes[name];field=fields[lod,side,direction,value];chosen=np.flatnonzero(np.linalg.norm(field,axis=1)>1e-7);filename=f'{side}_{direction}_{step:03d}_LOD{lod}_dna.f32'
                    if lod==0 and not len(chosen):chosen=np.array([0],dtype=int)
                    if not len(chosen):continue
                    field[chosen].astype('<f4').tofile(destination/filename);target=dict(mesh=name,channel=channel,lod=lod,indices=chosen.tolist(),file=filename)
                    (root_targets if lod==0 else lower_targets).append(target)
                    mid,vid=native_ids[lod];selected=np.flatnonzero(mid==mesh['index']);compiled=field[vid[selected]]@transform[:3];active=np.linalg.norm(compiled,axis=1)>1e-7
                    if active.any():
                        filename=f'{side}_{direction}_{step:03d}_LOD{lod}_compiled.f32';compiled[active].astype('<f4').tofile(destination/filename);patches.append(dict(morph=name+'__'+channel,lod=lod,indices=selected[active].tolist(),file=filename))
            solvers.append(dict(name=f'{namespace}_solver_{suffix}_{direction.lower()}',radius=(.5 if direction=='Blink' else .25) if transport_mode=='FIXED_CONVEX_ENVELOPE' else float(radius),function='LINEAR' if transport_mode=='FIXED_CONVEX_ENVELOPE' else 'GAUSSIAN',inputs=['CTRL_expressions.eyeBlink'+suffix if direction=='Blink' else 'CTRL_expressions.eyeLook'+direction+suffix],poses=rows))
    # Five values on each scalar driver avoid the native packed-quaternion
    # conversion ambiguity, and excluded controls cannot alter its weights.
    (destination/'pose-request.json').write_text(json.dumps(dict(schema='CM_DNA_POSE_CORRECTIVES_V1',maximum_offset=maximum_offset_cm,targets=root_targets,solvers=solvers)))
    (destination/'driven-request.json').write_text(json.dumps(dict(schema='CM_DNA_DRIVEN_TARGETS_V1',maximum_offset=maximum_offset_cm,activate_owned_drivers=True,targets=lower_targets)))
    (destination/'compiled-request.json').write_text(json.dumps(dict(schema='CM_COMPILED_CORRECTIVE_V1',maximum_offset_cm=maximum_offset_cm,allow_new_dna_targets=True,patches=patches)))
    prepared={file.name:digest(file) for file in destination.iterdir() if file.is_file()};report=dict(schema='CM_NATIVE_LOD_CORRECTIVE_REQUEST_V1',status='prepared',source_mesh=base_mesh,source_dna=str(base_dna),source_asset_file=str(base_asset_file),source_asset_sha256=asset_sha,targets=len(root_targets)+len(lower_targets),lods=list(range(reference['lod_count'])),input_sha256=inputs,prepared_sha256=prepared,coordinate_transform=transform.tolist(),coordinate_frame_error_cm=frame_error,method='BLINK_INDEPENDENT_CARDINAL_GAZE_FIELDS',channel_namespace=namespace,gaze_only_solvers=len(solvers),maximum_rest_field_cm=max(maxima),native_assets_modified=False,native_motion_accepted=False)
    report.update(gaze_transport_mode=transport_mode,kernel='LINEAR' if transport_mode=='FIXED_CONVEX_ENVELOPE' else 'GAUSSIAN')
    report.update(gaze_only_solvers=8,blink_volume_solvers=2 if blink_captures is not None else 0)
    (destination/'manifest.json').write_text(json.dumps(report,indent=2));return report
