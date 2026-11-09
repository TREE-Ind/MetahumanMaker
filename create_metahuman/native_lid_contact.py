"""Prepare private LOD0 blink/gaze contact fields from native rig captures.

The neutral and artist materials are retained. This is a calibration backend;
compiled native motion and appearance still require review before promotion.
"""
import hashlib,json,re,shutil,uuid
from pathlib import Path
import numpy as np
from .ocular_lid_contact import clear_contact
from .calibration_math import rest_offsets,canonical_offsets


def digest(file):
    with Path(file).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()


def native_exterior_triangles(triangles,mesh_ids,vertex_ids,reference_sheet):
    """Use captured render facets with the reference's exterior vertex domain.

    DNA and imported render meshes can triangulate the same quads differently.
    When every supplied facet exists in the capture, retain that exact subset.
    Legacy DNA sheets use their exterior vertex domain instead; clearance on
    DNA diagonals alone does not certify the rendered surface. Boundary-crossing
    facets stay excluded rather than pulling socket vertices into the aperture.
    """
    triangles=np.asarray(triangles);mesh_ids=np.asarray(mesh_ids);vertex_ids=np.asarray(vertex_ids);reference_sheet=np.asarray(reference_sheet)
    if triangles.ndim!=2 or triangles.shape[1]!=3 or not np.issubdtype(triangles.dtype,np.integer) or not len(triangles) or triangles.min()<0 or triangles.max()>=len(mesh_ids):raise ValueError('Supply complete native render triangles')
    if mesh_ids.ndim!=1 or vertex_ids.shape!=mesh_ids.shape or not np.issubdtype(vertex_ids.dtype,np.integer) or np.any(vertex_ids<0):raise ValueError('Supply exact native canonical vertex identities')
    if reference_sheet.ndim!=2 or reference_sheet.shape[1]!=3 or not np.issubdtype(reference_sheet.dtype,np.integer) or not len(reference_sheet) or reference_sheet.min()<0:raise ValueError('Supply the reference exterior anatomical domain')
    head=vertex_ids[triangles[np.all(mesh_ids[triangles]==0,axis=1)]]
    if not len(head) or np.any(np.diff(np.sort(head,axis=1),axis=1)==0):raise ValueError('The captured native head has missing or degenerate facets')
    reference={tuple(sorted(map(int,t))) for t in reference_sheet}
    native_keys=[tuple(sorted(map(int,t))) for t in head]
    exact=reference.issubset(set(native_keys))
    selected=head[np.asarray([key in reference for key in native_keys])] if exact else head[np.all(np.isin(head,np.unique(reference_sheet)),axis=1)]
    if not len(selected):raise ValueError('The native head has no corresponding exterior facets')
    return selected,dict(method='CAPTURED_NATIVE_RENDER_TRIANGLES',native_head_triangles=len(head),selected_exterior_triangles=len(selected),reference_exterior_triangles=len(reference_sheet),identical_reference_triangles=sum(tuple(sorted(map(int,t))) in reference for t in selected),excluded_socket_vertices_retained=True,boundary_policy='EXACT_CANONICAL_FACETS' if exact else 'ALL_VERTICES_IN_REFERENCE_EXTERIOR_DOMAIN',canonical_identities_preserved=True,position_based_matching=False)


def solve_pose(capture_file,structure_file,destination,*,maximum_fraction=.35,clearance_fraction=.005,contact_radius_fraction=None,halo_radius_fraction=None,fairing_strength=0.,visibility_mode='NEUTRAL_MATERIAL'):
    capture_file=Path(capture_file).resolve();structure_file=Path(structure_file).resolve();destination=Path(destination).resolve();inputs={str(capture_file):digest(capture_file),str(structure_file):digest(structure_file)};meta=json.loads(capture_file.read_text())
    if meta.get('lod')!=0 or destination.exists():raise ValueError('Choose a fresh native LOD0 contact solution')
    def read(key,dtype):
        name=meta[key]
        if Path(name).name!=name:raise ValueError('Native arrays must stay beside their capture')
        path=capture_file.parent/name;inputs[str(path)]=digest(path);return np.fromfile(path,dtype=dtype)
    mid=read('dna_mesh_file','<i4');vid=read('dna_vertex_file','<i4');pose=read('vertices_file','<f4').reshape(-1,3);neutral=read('neutral_file','<f4').reshape(-1,3);normals=read('neutral_normals_file','<f4').reshape(-1,3);linear=read('skin_linear_file','<f4').reshape(-1,3,3)
    if not len(mid)==len(vid)==len(pose)==len(neutral)==len(normals)==len(linear) or np.any(vid<0):raise ValueError('Incomplete native contact identities')
    def canonical(index,values):return canonical_offsets(values,mid,vid,index,int(vid[mid==index].max())+1)[0]
    frame=np.array([1.,-1.,1.]);head=canonical(0,pose)*frame;rest=canonical(0,neutral)*frame;normal=canonical(0,normals)*frame;structure=np.load(structure_file)
    faces,topology=native_exterior_triangles(read('triangles_file','<u4').reshape(-1,3),mid,vid,structure['outer_triangles']);cross=np.cross(rest[faces[:,1]]-rest[faces[:,0]],rest[faces[:,2]]-rest[faces[:,0]]);dot=np.einsum('ij,ij->i',cross,normal[faces].sum(1))
    if np.median(dot[abs(dot)>1e-15])<0:faces=faces[:,[0,2,1]]
    fields={};proofs=[];selected=mid==0
    for side,index in [('left',3),('right',4)]:
        corrected,proof=clear_contact(head,faces,structure[side+'_margin'],canonical(index,pose)*frame,neutral=rest,neutral_eye=canonical(index,neutral)*frame,maximum_fraction=maximum_fraction,clearance_fraction=clearance_fraction,contact_radius_fraction=contact_radius_fraction,halo_radius_fraction=halo_radius_fraction,fairing_strength=fairing_strength,visibility_mode=visibility_mode)
        desired=corrected[vid[selected]]*frame;offset,skin_proof=rest_offsets(pose[selected],desired,linear[selected]);field,split=canonical_offsets(offset,np.zeros(selected.sum(),dtype=np.int32),vid[selected],0,len(head));fields[side]=field;proofs.append(dict(side=side,contact=proof,skin=skin_proof,split=split))
    if any(digest(path)!=sha for path,sha in inputs.items()):raise ValueError('A contact input changed during calibration')
    destination.mkdir(parents=True);np.savez_compressed(destination/'fields.npz',**fields)
    report=dict(status='solved',source_mesh=meta['source_mesh'],capture_file=str(capture_file),input_sha256=inputs,fields_sha256=digest(destination/'fields.npz'),render_topology=topology,eyes=proofs,native_motion_accepted=False)
    (destination/'solution.json').write_text(json.dumps(report,indent=2));return report


def prepare(capture_folder,solution_folder,reference_file,base_dna,source_asset_file,coordinate_transform,destination,*,radius=.55,maximum_offset_cm=15.):
    capture_folder=Path(capture_folder).resolve();solution_folder=Path(solution_folder).resolve();reference_file=Path(reference_file).resolve();base_dna=Path(base_dna).resolve();source_asset_file=Path(source_asset_file).resolve();destination=Path(destination).resolve();capture=json.loads((capture_folder/'capture.json').read_text());reference=json.loads(reference_file.read_text());matrix=np.asarray(coordinate_transform,dtype=float)
    if destination.exists() or capture.get('status')!='complete' or reference.get('coordinates')!='PRESERVED_DNA' or matrix.shape!=(4,3) or not np.isfinite(matrix).all() or not np.isfinite(radius) or radius<=0 or not np.isfinite(maximum_offset_cm) or maximum_offset_cm<=0:raise ValueError('Supply a fresh destination and complete native contact training')
    inverse=np.linalg.inv(matrix[:3]);inputs={str(path):digest(path) for path in [reference_file,base_dna,capture_folder/'capture.json']};fields={};identities=None
    expected={'Blink'+str(b)+d+str(g) for b in [50,100] for d in ['Up','Down','Left','Right'] for g in [50,100]}|{'Blink50','Blink100'}
    if len(capture['poses'])!=18 or {row['name'] for row in capture['poses']}!=expected:raise ValueError('Capture the complete blink/cardinal-gaze grid')
    for row in capture['poses']:
        if row['lod']!=0 or Path(row['file']).resolve().parent!=capture_folder:raise ValueError('A contact sample belongs to another LOD or capture folder')
        folder=solution_folder/row['name'];report=json.loads((folder/'solution.json').read_text());inputs[str(folder/'solution.json')]=digest(folder/'solution.json');path=folder/'fields.npz'
        if report['status']!='solved' or report['source_mesh']!=capture['candidate'] or Path(report['capture_file']).resolve()!=Path(row['file']).resolve() or digest(path)!=report['fields_sha256']:raise ValueError('A contact solution does not match the requested native rig')
        for file,sha in report['input_sha256'].items():
            if file in inputs and inputs[file]!=sha:raise ValueError('Contact solutions used different versions of a shared input')
            inputs[file]=sha
        inputs[str(path)]=digest(path);data=np.load(path);fields[row['name']]={side:data[side] for side in ['left','right']}
        meta=json.loads(Path(row['file']).read_text());mid=np.fromfile(capture_folder/meta['dna_mesh_file'],dtype='<i4');vid=np.fromfile(capture_folder/meta['dna_vertex_file'],dtype='<i4')
        match=re.fullmatch(r'Blink(50|100)(?:(Up|Down|Left|Right)(50|100))?',row['name']);blink=int(match[1])/100.;wanted={'eyeBlink'+s:blink for s in ['L','R']}
        if match[2]:wanted.update({'eyeLook'+match[2]+s:int(match[3])/100. for s in ['L','R']})
        curves={name.lower().removeprefix('ctrl_expressions_'):value for name,value in meta['raw_curves'].items()}
        if row['controls']!=wanted or meta['source_mesh']!=capture['candidate'] or any(abs(curves.get(name.lower(),float('inf'))-value)>1e-6 for name,value in wanted.items()) or any(abs(value)>1e-6 for name,value in curves.items() if name.startswith(('eyeblink','eyelook','eyesquint')) and name not in {key.lower() for key in wanted}):raise ValueError('A contact capture has different native blink/gaze controls')
        if identities is None:identities=(mid,vid)
        elif not np.array_equal(mid,identities[0]) or not np.array_equal(vid,identities[1]):raise ValueError('Contact training vertex identities changed')
    if any(digest(path)!=sha for path,sha in inputs.items()):raise ValueError('Contact training changed before sealing the request')
    head=next(m for m in reference['meshes'] if m['name']=='head_lod0_mesh');namespace='CM_lid_contact_'+uuid.uuid4().hex[:8];targets=[];patches=[];solvers=[];destination.mkdir(parents=True);mid,vid=identities;selected=np.flatnonzero(mid==head['index'])
    def target(channel,field):
        if field.shape!=(head['vertex_count'],3) or not np.isfinite(field).all() or np.linalg.norm(field,axis=1).max()>maximum_offset_cm:raise ValueError('A contact field leaves its canonical or physical bounds')
        ids=np.flatnonzero(np.linalg.norm(field,axis=1)>1e-7)
        if not len(ids):ids=np.array([0])
        name=channel+'_dna.f32';(field[ids]@inverse).astype('<f4').tofile(destination/name);targets.append(dict(mesh=head['name'],lod=0,channel=channel,indices=ids.tolist(),file=name));compiled=field[vid[selected]];use=np.linalg.norm(compiled,axis=1)>1e-7
        if use.any():
            name=channel+'_compiled.f32';compiled[use].astype('<f4').tofile(destination/name);patches.append(dict(morph=head['name']+'__'+channel,lod=0,indices=selected[use].tolist(),file=name))
    for side,suffix in [('left','L'),('right','R')]:
        base=[]
        for b in [0,50,100]:
            row=dict(name=f'{namespace}_{suffix}_base_{b}',values=[b/100.])
            if b:
                channel=f'{namespace}_{suffix}_blink_{b}';target(channel,fields['Blink'+str(b)][side]);row.update(channel=channel,multipliers=['CTRL_expressions.eyeBlink'+suffix],scale=100./b)
            base.append(row)
        solvers.append(dict(name=f'{namespace}_{suffix}_blink',radius=radius,inputs=['CTRL_expressions.eyeBlink'+suffix],poses=base))
        for direction in ['Up','Down','Left','Right']:
            poses=[]
            for b in [0,50,100]:
                for g in [0,50,100]:
                    row=dict(name=f'{namespace}_{suffix}_{direction}_{b}_{g}',values=[b/100.,g/100.])
                    if b and g:
                        channel=row['name'];target(channel,fields['Blink'+str(b)+direction+str(g)][side]-fields['Blink'+str(b)][side]);row.update(channel=channel,multipliers=['CTRL_expressions.eyeBlink'+suffix,'CTRL_expressions.eyeLook'+direction+suffix],scale=10000./(b*g))
                    poses.append(row)
            solvers.append(dict(name=f'{namespace}_{suffix}_{direction}',radius=radius,inputs=['CTRL_expressions.eyeBlink'+suffix,'CTRL_expressions.eyeLook'+direction+suffix],poses=poses))
    request=dict(schema='CM_DNA_POSE_CORRECTIVES_V1',maximum_offset=maximum_offset_cm/np.linalg.svd(matrix[:3],compute_uv=False).min(),targets=targets,solvers=solvers)
    (destination/'pose-request.json').write_text(json.dumps(request));(destination/'compiled-request.json').write_text(json.dumps(dict(schema='CM_COMPILED_CORRECTIVE_V1',maximum_offset_cm=maximum_offset_cm,allow_new_dna_targets=True,patches=patches)))
    prepared={path.name:digest(path) for path in destination.iterdir() if path.is_file()};report=dict(schema='CM_NATIVE_LOD_CORRECTIVE_REQUEST_V1',status='prepared',source_mesh=capture['candidate'],source_dna=str(base_dna),source_asset_file=str(source_asset_file),source_asset_sha256=digest(source_asset_file),targets=len(targets),lods=[0],input_sha256=inputs,prepared_sha256=prepared,method='NEUTRAL_VISIBLE_BLINK_GAZE_LID_CONTACT',channel_namespace=namespace,coordinate_transform=matrix.tolist(),native_motion_accepted=False)
    (destination/'manifest.json').write_text(json.dumps(report,indent=2));return report


def rebase_globe_sibling(request_folder,old_globe_folder,new_globe_folder,new_asset_file,destination):
    """Reuse head fields across sealed, globe-only forks of the same base.

    This checks operation provenance, not arbitrary artist edits. Both native
    revisions must append only globe targets to the same saved source DNA.
    Head evaluation is retained; contact with the new globes needs new review.
    """
    request_folder=Path(request_folder).resolve();destination=Path(destination).resolve();new_asset_file=Path(new_asset_file).resolve();request=json.loads((request_folder/'manifest.json').read_text());inputs=dict(request['input_sha256']);forks=[]
    if destination.exists() or request.get('method')!='NEUTRAL_VISIBLE_BLINK_GAZE_LID_CONTACT' or request['lods']!=[0]:raise ValueError('Rebase a prepared head contact request to a fresh folder')
    def record(file):
        file=Path(file).resolve();sha=digest(file)
        if str(file) in inputs and inputs[str(file)]!=sha:raise ValueError('A shared calibration input changed')
        inputs[str(file)]=sha;return json.loads(file.read_text())
    record(request_folder/'manifest.json')
    for file,sha in request['prepared_sha256'].items():
        path=request_folder/file
        if path.parent!=request_folder or digest(path)!=sha:raise ValueError('The original contact payload changed')
    for folder in [old_globe_folder,new_globe_folder]:
        folder=Path(folder).resolve();manifest=record(folder/'manifest.json');stage=record(folder/'native/stage.json')
        if stage['status']!='complete' or stage['request_sha256']!=digest(folder/'manifest.json') or stage['source']!=manifest['source_mesh']:raise ValueError('A globe fork lacks matching native staging provenance')
        for file,sha in manifest['prepared_sha256'].items():
            path=folder/file
            if path.parent!=folder or digest(path)!=sha:raise ValueError('A globe fork payload changed')
        targets=record(folder/'pose-request.json')['targets']+record(folder/'driven-request.json')['targets'];patches=record(folder/'compiled-request.json')['patches']
        if not targets or any(not re.fullmatch(r'eye(?:Left|Right)_lod[0-7]_mesh',row['mesh']) for row in targets) or any(not re.fullmatch(r'eye(?:Left|Right)_lod[0-7]_mesh__CM_[A-Za-z0-9_]+',row['morph']) for row in patches):raise ValueError('Only globe geometry may differ between these source forks')
        for file,sha in manifest['input_sha256'].items():
            if file in inputs and inputs[file]!=sha:raise ValueError('The globe forks used different shared inputs')
            inputs[file]=sha
        inputs[stage['dna']]=digest(stage['dna']);forks.append((manifest,stage))
    old,previous=forks[0];new,current=forks[1]
    if any(old[key]!=new[key] for key in ['source_mesh','source_dna','source_asset_sha256','coordinate_transform']) or request['source_mesh']!=previous['candidate'] or Path(request['source_dna']).resolve()!=Path(previous['dna']).resolve():raise ValueError('Head contact reuse requires globe-only siblings of the exact same saved rig')
    pose=record(request_folder/'pose-request.json');compiled=record(request_folder/'compiled-request.json')
    if any(row['mesh']!='head_lod0_mesh' for row in pose['targets']) or any(not row['morph'].startswith('head_lod0_mesh__CM_') or row['lod']!=0 for row in compiled['patches']):raise ValueError('Only LOD0 head fields can be reused by this operation')
    if any(digest(file)!=sha for file,sha in inputs.items()):raise ValueError('A sealed source changed before contact reuse')
    destination.mkdir(parents=True)
    for file in request['prepared_sha256']:shutil.copyfile(request_folder/file,destination/file)
    request.update(source_mesh=current['candidate'],source_dna=current['dna'],source_asset_file=str(new_asset_file),source_asset_sha256=digest(new_asset_file),input_sha256=inputs,prepared_sha256={file:digest(destination/file) for file in request['prepared_sha256']},rebased_from=str(request_folder),reuse_method='SEALED_GLOBE_ONLY_SIBLINGS',head_evaluation_preserved_by_globe_only_operations=True,new_globe_contact_certified=False,native_motion_accepted=False)
    request['preserved_globe_motion']=dict(gaze_transport_mode=new.get('gaze_transport_mode','PRESERVE_FITTED'),blink_input_excluded=not bool(new.get('blink_volume_solvers')),native_pupil_targets_retained=True)
    (destination/'manifest.json').write_text(json.dumps(request,indent=2));return request
