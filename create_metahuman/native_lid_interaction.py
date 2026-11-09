"""Prepare intrinsic, axis-preserving blink/expression interaction fields.

The current rig supplies all target poses. No character-specific vertex lists
or archetype facial proportions are introduced. Native appearance acceptance
is required separately, especially for gaze and other combined expressions.
"""
import json,re,uuid
from pathlib import Path
import numpy as np
from scipy import sparse
from scipy.sparse.csgraph import dijkstra
from .native_lid_contact import digest,native_exterior_triangles
from .native_corrective_gates import interaction_family,write_pose_groups
from .calibration_math import canonical_offsets,rest_offsets
from .ocular_attachments import bind


def prepare(capture_folder,reference_file,structure_file,source_dna,source_asset_file,coordinate_transform,destination,*,driver='eyeSquintInner',closure='eyeBlink',core_fraction=.16,halo_fraction=.45,maximum_offset_cm=15.,maximum_attachment_cm=3.,accessory_indices=(6,7)):
    capture_folder=Path(capture_folder).resolve();reference_file=Path(reference_file).resolve();structure_file=Path(structure_file).resolve();source_dna=Path(source_dna).resolve();source_asset_file=Path(source_asset_file).resolve();destination=Path(destination).resolve();matrix=np.asarray(coordinate_transform,dtype=float)
    capture=json.loads((capture_folder/'capture.json').read_text());reference=json.loads(reference_file.read_text())
    if destination.exists() or capture.get('status')!='complete' or not capture.get('source_asset_sha256') or reference.get('coordinates')!='PRESERVED_DNA':raise ValueError('Choose a fresh destination and a complete fingerprinted native capture')
    source_sha=digest(source_asset_file)
    if Path(capture['source_asset_file']).resolve()!=source_asset_file or source_sha!=capture['source_asset_sha256']:raise ValueError('The saved source face changed after interaction capture')
    if matrix.shape!=(4,3) or not np.isfinite(matrix).all() or np.linalg.svd(matrix[:3],compute_uv=False).min()<1e-8:raise ValueError('Supply a finite invertible preserved-DNA coordinate transform')
    if driver==closure or any(not re.fullmatch(r'eye[A-Za-z0-9_]+',x) for x in [driver,closure]) or not np.isfinite([core_fraction,halo_fraction,maximum_offset_cm,maximum_attachment_cm]).all() or not 0<core_fraction<halo_fraction<=.5 or min(maximum_offset_cm,maximum_attachment_cm)<=0:raise ValueError('Choose distinct eye controls and bounded intrinsic core/halo dimensions')
    meshes={m['index']:m for m in reference['meshes']};indices=[0,*accessory_indices]
    if len(set(indices))!=len(indices) or any(i not in meshes or 0 not in meshes[i]['lods'] for i in indices):raise ValueError('Choose distinct canonical LOD0 head/accessory meshes')
    inputs={str(p):digest(p) for p in [capture_folder/'capture.json',reference_file,structure_file,source_dna]}
    def record(path):
        path=Path(path).resolve();sha=digest(path)
        if str(path) in inputs and inputs[str(path)]!=sha:raise ValueError('An interaction input changed')
        inputs[str(path)]=sha;return path
    def load(row):
        file=Path(row['file']).resolve()
        if file.parent!=capture_folder or row['lod']!=0:raise ValueError('Keep interaction samples in their captured LOD0 folder')
        meta=json.loads(record(file).read_text())
        if meta['source_mesh']!=capture['candidate'] or meta['lod']!=0 or meta.get('dna_mode')!='ASSEMBLED_EMBEDDED':raise ValueError('An interaction sample belongs to another rig')
        def read(key,dtype,shape=None):
            name=meta[key]
            if Path(name).name!=name:raise ValueError('Native arrays must remain beside their metadata')
            data=np.fromfile(record(capture_folder/name),dtype=dtype);return data.reshape(shape) if shape else data
        controls={k.removeprefix('CTRL_expressions.'):v for k,v in row['controls'].items()};curves={k.lower().removeprefix('ctrl_expressions_'):v for k,v in meta['raw_curves'].items()}
        if any(type(v) not in (int,float) or not np.isfinite(v) or not 0<v<=1 or abs(curves.get(k.lower(),float('inf'))-v)>1e-6 for k,v in controls.items()) or any(abs(v)>1e-6 for k,v in curves.items() if k not in {x.lower() for x in controls}):raise ValueError('An interaction sample evaluated unexpected raw controls')
        return dict(name=row['name'],controls=controls,mid=read('dna_mesh_file','<i4'),vid=read('dna_vertex_file','<i4'),triangles=read('triangles_file','<u4',(-1,3)),pose=read('vertices_file','<f4',(-1,3)),linear=read('skin_linear_file','<f4',(-1,3,3)))
    frames=[load(row) for row in capture['poses']];neutral=[f for f in frames if not f['controls']]
    if len(neutral)!=1:raise ValueError('Keep exactly one neutral interaction reference')
    neutral=neutral[0];mid=neutral['mid'];vid=neutral['vid'];grids={side:{(0.,0.):neutral} for side in ['L','R']}
    def canonical(values,index):return canonical_offsets(values,mid,vid,index,meshes[index]['vertex_count'])[0]
    for frame in frames:
        if not np.array_equal(frame['mid'],mid) or not np.array_equal(frame['vid'],vid) or not np.array_equal(frame['triangles'],neutral['triangles']) or frame['pose'].shape!=neutral['pose'].shape or frame['linear'].shape!=(len(mid),3,3) or not np.isfinite(frame['pose']).all() or not np.isfinite(frame['linear']).all():raise ValueError('Native topology, correspondence or skin frames changed between samples')
        if not frame['controls']:continue
        sides=[side for side in grids if set(frame['controls'])<={driver+side,closure+side}]
        if len(sides)!=1:raise ValueError('Capture each eye independently, including both single-control axes')
        side=sides[0];point=(float(frame['controls'].get(driver+side,0)),float(frame['controls'].get(closure+side,0)))
        if point in grids[side]:raise ValueError('Interaction control points must be distinct')
        grids[side][point]=frame
    for grid in grids.values():
        axes=[sorted({p[i] for p in grid}) for i in range(2)]
        if any(a[0]!=0 or a[-1]!=1 for a in axes) or set(grid)!={(a,b) for a in axes[0] for b in axes[1]} or (2*len(grid))%4==0:raise ValueError('Capture a complete independent grid through full activation, with a UE-compatible scalar knot count')
        if any(s*b<1/64 for s,b in grid if s and b):raise ValueError('Choose interaction knots whose nonzero strength product is at least 1/64 for the native output-scale bound')
    interior_count=sum(sum(s>0 and b>0 for s,b in g) for g in grids.values())
    if interior_count*len(indices)>512 or 2*len(indices)>128:raise ValueError('The interaction grid exceeds the grouped native request limits')
    rest={i:canonical(neutral['pose'],i) for i in [*indices,3,4]};structure=np.load(structure_file)
    triangles,topology=native_exterior_triangles(neutral['triangles'],mid,vid,structure['outer_triangles'])
    edges=np.unique(np.sort(np.concatenate((triangles[:,[0,1]],triangles[:,[1,2]],triangles[:,[2,0]])),axis=1),axis=0);length=np.linalg.norm(rest[0][edges[:,0]]-rest[0][edges[:,1]],axis=1)
    if np.any(length<=0):raise ValueError('The intrinsic interaction surface contains collapsed edges')
    graph=sparse.coo_matrix((np.r_[length,length],(np.r_[edges[:,0],edges[:,1]],np.r_[edges[:,1],edges[:,0]])),shape=(len(rest[0]),len(rest[0]))).tocsr();weights={};cores={};domains=[]
    for side,role,index in [('L','left',3),('R','right',4)]:
        width=float(np.ptp(rest[index][:,0]));core=core_fraction*width;halo=halo_fraction*width;margin=structure[role+'_margin'];distance=dijkstra(graph,directed=False,indices=margin,min_only=True,limit=halo)
        t=np.clip((distance-core)/(halo-core),0,1);weight=1-t*t*(3-2*t);inner=structure[role+'_interior'];weight[inner]=1;weights[side]=weight;cores[side]=weight==1
        if width<=0 or not np.isfinite(weight).all() or not np.all(weight[margin]==1):raise ValueError('The captured eye has no complete intrinsic interaction core')
        domains.append(dict(side=side,core_radius_cm=core,halo_radius_cm=halo,core_vertices=int(cores[side].sum()),affected_vertices=int((weight>0).sum())))
    if np.any(cores['L']&cores['R']):raise ValueError('The two anatomical interaction cores overlap; refine their physical dimensions')
    weights['L'][cores['R']]=0;weights['R'][cores['L']]=0;total=np.maximum(1,weights['L']+weights['R'])
    for side in weights:weights[side]/=total
    bindings={i:bind(rest[0],triangles,rest[i],maximum_distance=maximum_attachment_cm) for i in indices if i!=0};fields={};reports=[]
    for side,grid in grids.items():
        for (s,b),frame in sorted(grid.items()):
            if not s or not b:continue
            head=canonical(frame['pose'],0);blink=canonical(grid[0.,b]['pose'],0);squint=canonical(grid[s,0.]['pose'],0)
            target=blink+(1-b)*(squint-rest[0]);delta=(target-head)*weights[side][:,None];desired=head+delta;use=mid==0
            offset,proof=rest_offsets(frame['pose'][use],desired[vid[use]],frame['linear'][use]);field,split=canonical_offsets(offset,np.zeros(use.sum(),dtype=np.int32),vid[use],0,len(head));fields[0,side,s,b]=field
            core_error=float(np.linalg.norm(desired[cores[side]]-blink[cores[side]],axis=1).max(initial=0)) if b==1 else None
            if core_error is not None and core_error>1e-8:raise ValueError('The full-blink core did not retain its closed reference')
            reports.append(dict(side=side,driver=s,closure=b,maximum_core_error_cm=core_error,skin=proof,split=split))
            for i,binding in bindings.items():
                pose_delta=np.einsum('ni,nij->nj',binding['barycentric'],delta[binding['triangles']]);use=mid==i;desired_accessory=frame['pose'][use]+pose_delta[vid[use]];offset,_=rest_offsets(frame['pose'][use],desired_accessory,frame['linear'][use]);fields[i,side,s,b]=canonical_offsets(offset,np.full(use.sum(),i,dtype=np.int32),vid[use],i,meshes[i]['vertex_count'])[0]
    if any(not np.isfinite(field).all() or np.linalg.norm(field,axis=1).max()>maximum_offset_cm for field in fields.values()):raise ValueError('An interaction field exceeds its physical offset bound')
    if any(digest(file)!=sha for file,sha in inputs.items()) or digest(source_asset_file)!=source_sha:raise ValueError('An interaction source changed before request creation')
    destination.mkdir(parents=True);inverse=np.linalg.inv(matrix[:3]);targets=[];solvers=[];patches=[];namespace='CM_lidint_'+uuid.uuid4().hex[:8]
    for i in indices:
        for side,grid in grids.items():
            samples=[]
            for step,(s,b) in enumerate(sorted(grid)):
                row=dict(values=[s,b])
                if s and b:
                    field=fields[i,side,s,b];ids=np.flatnonzero(np.linalg.norm(field,axis=1)>1e-7)
                    if not len(ids):ids=np.array([0])
                    name=f'm{i}_{side}_{step:02d}_dna.f32';(field[ids]@inverse).astype('<f4').tofile(destination/name);row.update(indices=ids.tolist(),file=name)
                samples.append(row)
            family=interaction_family(destination,samples,mesh=meshes[i]['name'],drivers=['CTRL_expressions.'+driver+side,'CTRL_expressions.'+closure+side],namespace=f'{namespace}_m{i}_{side}');targets.extend(family['targets']);solvers.extend(family['solvers'])
    for target in targets:
        i=next(i for i in indices if meshes[i]['name']==target['mesh']);canonical_field=np.zeros((meshes[i]['vertex_count'],3));canonical_field[target['indices']]=np.fromfile(destination/target['file'],dtype='<f4').reshape(-1,3)@matrix[:3];use=np.flatnonzero(mid==i);compiled=canonical_field[vid[use]];active=np.linalg.norm(compiled,axis=1)>1e-7
        if active.any():
            name=target['channel']+'_compiled.f32';compiled[active].astype('<f4').tofile(destination/name);patches.append(dict(morph=target['mesh']+'__'+target['channel'],lod=0,indices=use[active].tolist(),file=name))
    pose_requests=write_pose_groups(destination,targets,solvers,maximum_offset_cm/np.linalg.svd(matrix[:3],compute_uv=False).min())
    (destination/'compiled-request.json').write_text(json.dumps(dict(schema='CM_COMPILED_CORRECTIVE_V1',maximum_offset_cm=maximum_offset_cm,allow_new_dna_targets=True,patches=patches)))
    prepared={f.name:digest(f) for f in destination.iterdir() if f.is_file()};report=dict(schema='CM_NATIVE_LOD_CORRECTIVE_REQUEST_V1',status='prepared',source_mesh=capture['candidate'],source_dna=str(source_dna),source_asset_file=str(source_asset_file),source_asset_sha256=source_sha,targets=len(targets),lods=[0],method='INTRINSIC_AXIS_PRESERVING_LID_INTERACTION',driver=driver,closure=closure,domains=domains,pose_reports=reports,attachment_topology=topology,coordinate_transform=matrix.tolist(),input_sha256=inputs,prepared_sha256=prepared,minimum_bridge='0.27.0',neutral_geometry_changed=False,globe_geometry_changed=False,zero_on_either_axis=True,native_motion_accepted=False)
    report['pose_requests']=pose_requests
    if any(digest(file)!=sha for file,sha in inputs.items()) or digest(source_asset_file)!=source_sha:raise ValueError('An interaction source changed during request sealing')
    (destination/'manifest.json').write_text(json.dumps(report,indent=2));return report
