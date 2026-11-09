"""Seal a paired native expression cleanup and its ocular accessory fields.

The neutral, topology, UVs, joints and existing expression fields are retained.
New complementary drivers fade the cleanup as a separate control increases.
Native assembled motion/appearance acceptance remains a separate operation.
"""
import json,re,uuid
from pathlib import Path
import numpy as np
from .native_lid_contact import digest,native_exterior_triangles
from .native_corrective_gates import complementary_family
from .calibration_math import canonical_offsets,rest_offsets
from .ocular_attachments import bind


def prepare(capture_folder,solution_folder,reference_file,structure_file,source_dna,source_asset_file,coordinate_transform,destination,*,driver='eyeSquintInner',fade='eyeBlink',accessory_indices=(6,7),maximum_offset_cm=15.,maximum_attachment_cm=3.,allow_unfingerprinted_capture=False):
    capture_folder=Path(capture_folder).resolve();solution_folder=Path(solution_folder).resolve();reference_file=Path(reference_file).resolve();structure_file=Path(structure_file).resolve();source_dna=Path(source_dna).resolve();source_asset_file=Path(source_asset_file).resolve();destination=Path(destination).resolve();matrix=np.asarray(coordinate_transform,dtype=float)
    capture=json.loads((capture_folder/'capture.json').read_text());reference=json.loads(reference_file.read_text())
    if destination.exists() or capture.get('status')!='complete' or reference.get('coordinates')!='PRESERVED_DNA' or matrix.shape!=(4,3) or not np.isfinite(matrix).all() or np.linalg.svd(matrix[:3],compute_uv=False).min()<1e-8:raise ValueError('Choose a fresh destination and complete native expression captures, DNA anatomy and transform')
    captured_sha=capture.get('source_asset_sha256')
    if captured_sha:
        if Path(capture['source_asset_file']).resolve()!=source_asset_file or digest(source_asset_file)!=captured_sha:raise ValueError('The saved source face changed after these poses were captured; capture the current revision again')
    elif not allow_unfingerprinted_capture:raise ValueError('Capture the saved source face with the current bridge workflow before preparing expression cleanup; older captures lack a source fingerprint')
    if driver==fade or any(not re.fullmatch(r'eye[A-Za-z0-9_]+',x) for x in [driver,fade]) or any(not np.isfinite(x) or x<=0 for x in [maximum_offset_cm,maximum_attachment_cm]):raise ValueError('Choose distinct paired eye controls and positive physical bounds')
    meshes={m['index']:m for m in reference['meshes']};indices=[0,*accessory_indices]
    if len(set(indices))!=len(indices) or any(i not in meshes or 0 not in meshes[i]['lods'] for i in indices):raise ValueError('Choose distinct native LOD0 head/accessory meshes')
    inputs={str(p):digest(p) for p in [capture_folder/'capture.json',reference_file,structure_file,source_dna]};frames={};neutral=None;states=[]
    def record(path):
        path=Path(path).resolve();sha=digest(path)
        if str(path) in inputs and inputs[str(path)]!=sha:raise ValueError('A native cleanup input changed')
        inputs[str(path)]=sha;return path
    def frame(row):
        file=Path(row['file']).resolve()
        if file.parent!=capture_folder or row['lod']!=0:raise ValueError('Expression samples must belong to this LOD0 capture folder')
        meta=json.loads(record(file).read_text())
        if meta['source_mesh']!=capture['candidate'] or meta['lod']!=0 or meta.get('dna_mode')!='ASSEMBLED_EMBEDDED':raise ValueError('A sample belongs to another native rig')
        def read(key,dtype,shape=None):
            name=meta[key]
            if Path(name).name!=name:raise ValueError('Native sample arrays must remain beside their metadata')
            data=np.fromfile(record(capture_folder/name),dtype=dtype);return data.reshape(shape) if shape else data
        return dict(meta=meta,file=file,mid=read('dna_mesh_file','<i4'),vid=read('dna_vertex_file','<i4'),triangles=read('triangles_file','<u4',(-1,3)),pose=read('vertices_file','<f4',(-1,3)),linear=read('skin_linear_file','<f4',(-1,3,3)))
    for row in capture['poses']:
        controls={k.removeprefix('CTRL_expressions.'):v for k,v in row['controls'].items()}
        if not controls:
            if neutral is not None:raise ValueError('Keep exactly one neutral reference sample')
            neutral=frame(row);continue
        wanted={driver+s for s in ['L','R']}
        if set(controls)!=wanted or controls[driver+'L']!=controls[driver+'R'] or type(controls[driver+'L']) not in (int,float) or not 0<controls[driver+'L']<=1:raise ValueError('Capture only neutral and equally activated left/right samples of the requested driver')
        current=frame(row);curves={k.lower().removeprefix('ctrl_expressions_'):v for k,v in current['meta']['raw_curves'].items()}
        if any(abs(curves.get(k.lower(),float('inf'))-v)>1e-6 for k,v in controls.items()) or any(abs(v)>1e-6 for k,v in curves.items() if k not in {x.lower() for x in wanted}):raise ValueError('A native sample evaluated unexpected raw controls')
        states.append((float(controls[driver+'L']),row,current))
    if neutral is None or not states or len({s[0] for s in states})!=len(states) or 4*len(states)*len(indices)>64 or 4*len(indices)>16:raise ValueError('Capture neutral and distinct driver strengths within the native target/solver limits')
    mid=neutral['mid'];vid=neutral['vid']
    if not len(mid)==len(vid)==len(neutral['pose'])==len(neutral['linear']):raise ValueError('The neutral native identities are incomplete')
    def canonical(values,index):return canonical_offsets(values,mid,vid,index,meshes[index]['vertex_count'])[0]
    rest={i:canonical(neutral['pose'],i) for i in indices};structure=np.load(structure_file)
    triangles,attachment_topology=native_exterior_triangles(neutral['triangles'],mid,vid,structure['outer_triangles'])
    bindings={i:bind(rest[0],triangles,rest[i],maximum_distance=maximum_attachment_cm) for i in indices if i!=0};fields={};reports=[]
    for value,row,current in states:
        if not np.array_equal(current['mid'],mid) or not np.array_equal(current['vid'],vid) or not np.array_equal(current['triangles'],neutral['triangles']) or current['pose'].shape!=neutral['pose'].shape:raise ValueError('Native render/canonical correspondence changed between samples')
        folder=solution_folder/row['name'];solution=json.loads(record(folder/'solution.json').read_text());file=record(folder/'fields.npz')
        if solution.get('status')!='solved' or solution['source_mesh']!=capture['candidate'] or Path(solution['capture_file']).resolve()!=current['file'] or digest(file)!=solution['fields_sha256']:raise ValueError('A cleanup field belongs to another native pose')
        for path,sha in solution['input_sha256'].items():
            if digest(Path(path))!=sha or (path in inputs and inputs[path]!=sha):raise ValueError('A cleanup source changed after its numerical solve')
            inputs[path]=sha
        head=canonical(current['pose'],0);data=np.load(file)
        for side in ['left','right']:
            field=data[side]
            if field.shape!=rest[0].shape or not np.isfinite(field).all():raise ValueError('A cleanup field has the wrong canonical head identity')
            fields[0,side,value]=field;selected=mid==0;posed_delta=np.einsum('ni,nij->nj',field[vid[selected]],current['linear'][selected]);delta,_=canonical_offsets(posed_delta,np.zeros(selected.sum(),dtype=np.int32),vid[selected],0,len(head))
            for i,binding in bindings.items():
                pose_delta=np.einsum('ni,nij->nj',binding['barycentric'],delta[binding['triangles']]);use=mid==i;desired=current['pose'][use]+pose_delta[vid[use]];offset,proof=rest_offsets(current['pose'][use],desired,current['linear'][use]);folded,split=canonical_offsets(offset,np.full(use.sum(),i,dtype=np.int32),vid[use],i,meshes[i]['vertex_count']);fields[i,side,value]=folded;reports.append(dict(mesh=meshes[i]['name'],side=side,value=value,skin=proof,split=split,attachment='ADDITIONAL_HEAD_DISPLACEMENT',maximum_binding_distance_cm=binding['maximum_binding_distance']))
    if any(digest(Path(path))!=sha for path,sha in inputs.items()):raise ValueError('A native cleanup input changed before request sealing')
    source_sha=digest(source_asset_file);destination.mkdir(parents=True);inverse=np.linalg.inv(matrix[:3]);targets=[];solvers=[];patches=[];namespace='CM_lidpol_'+uuid.uuid4().hex[:8]
    for i in indices:
        for side,suffix in [('left','L'),('right','R')]:
            samples=[]
            for step,(value,_,_) in enumerate(sorted(states,key=lambda item:item[0])):
                field=fields[i,side,value]
                if not np.isfinite(field).all() or np.linalg.norm(field,axis=1).max()>maximum_offset_cm:raise ValueError('An expression cleanup exceeds its physical offset bound')
                ids=np.flatnonzero(np.linalg.norm(field,axis=1)>1e-7)
                if not len(ids):ids=np.array([0])
                name=f'm{i}_{side}_{step:02d}_dna.f32';(field[ids]@inverse).astype('<f4').tofile(destination/name);samples.append(dict(value=value,indices=ids.tolist(),file=name))
            family=complementary_family(destination,samples,mesh=meshes[i]['name'],driver='CTRL_expressions.'+driver+suffix,gate='CTRL_expressions.'+fade+suffix,namespace=f'{namespace}_m{i}_{suffix}');targets.extend(family['targets']);solvers.extend(family['solvers'])
    for target in targets:
        i=next(i for i in indices if meshes[i]['name']==target['mesh']);canonical_field=np.zeros((meshes[i]['vertex_count'],3));canonical_field[target['indices']]=np.fromfile(destination/target['file'],dtype='<f4').reshape(-1,3)@matrix[:3];selected=np.flatnonzero(mid==i);compiled=canonical_field[vid[selected]];active=np.linalg.norm(compiled,axis=1)>1e-7
        if active.any():
            name=target['channel']+'_compiled.f32';compiled[active].astype('<f4').tofile(destination/name);patches.append(dict(morph=target['mesh']+'__'+target['channel'],lod=0,indices=selected[active].tolist(),file=name))
    (destination/'pose-request.json').write_text(json.dumps(dict(schema='CM_DNA_POSE_CORRECTIVES_V1',maximum_offset=maximum_offset_cm/np.linalg.svd(matrix[:3],compute_uv=False).min(),targets=targets,solvers=solvers)))
    (destination/'compiled-request.json').write_text(json.dumps(dict(schema='CM_COMPILED_CORRECTIVE_V1',maximum_offset_cm=maximum_offset_cm,allow_new_dna_targets=True,patches=patches)))
    prepared={file.name:digest(file) for file in destination.iterdir() if file.is_file()};report=dict(schema='CM_NATIVE_LOD_CORRECTIVE_REQUEST_V1',status='prepared',source_mesh=capture['candidate'],source_dna=str(source_dna),source_asset_file=str(source_asset_file),source_asset_sha256=source_sha,source_fingerprint_at_capture=bool(captured_sha),legacy_capture_explicitly_adopted=not bool(captured_sha),targets=len(targets),lods=[0],method='PAIRED_EXPRESSION_CLEANUP_WITH_COMPLEMENTARY_FADE',driver=driver,fade=fade,accessory_indices=list(accessory_indices),accessory_reports=reports,attachment_topology=attachment_topology,coordinate_transform=matrix.tolist(),input_sha256=inputs,prepared_sha256=prepared,minimum_bridge='0.27.0',neutral_geometry_changed=False,globe_geometry_changed=False,native_motion_accepted=False)
    if any(digest(Path(path))!=sha for path,sha in inputs.items()) or digest(source_asset_file)!=source_sha:raise ValueError('A native cleanup source changed during request sealing')
    (destination/'manifest.json').write_text(json.dumps(report,indent=2));return report
