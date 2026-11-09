"""Convert posed-head edits into sealed fields retaining native blink endpoints."""
import json,uuid
from pathlib import Path
import numpy as np
from .calibration_math import canonical_offsets,rest_offsets
from .native_lid_contact import digest
from .native_corrective_gates import complementary_family,endpoint_family
from .ocular_attachments import bind


def _encode_small_fields(field):
    """Preserve small authored deltas through UE's nearly-zero morph filter.

    Mesh-description evaluation can discard vectors whose every component is
    below its near-zero tolerance, independently of the morph build threshold.
    Two identically driven, non-small fields reconstruct those values at runtime.
    The carrier is numeric encoding only; it has no independent control or pose.
    Input/output units are the captured native centimeters.
    """
    field=np.asarray(field,dtype=float);small=(np.linalg.norm(field,axis=1)>1e-7)&(np.max(abs(field),axis=1)<2e-4)
    if not small.any():return [field],dict(method='DIRECT',compensated_vertices=0)
    primary=field.copy();cancel=np.zeros_like(field);primary[small,0]+=.001;cancel[small,0]=-.001
    decoded=primary.astype(np.float32).astype(float)+cancel.astype(np.float32).astype(float)
    error=float(np.max(np.linalg.norm(decoded-field.astype(np.float32).astype(float),axis=1)))
    if error>1e-8:raise ValueError('The compensated corrective encoding exceeds its numeric reconstruction budget')
    return [primary,cancel],dict(method='PAIRED_SUBTHRESHOLD_FIELDS',compensated_vertices=int(small.sum()),carrier_cm=.001,maximum_float32_decode_error_cm=error)


def prepare(snapshot_file,destination):
    snapshot_file=Path(snapshot_file).resolve();destination=Path(destination).resolve();snapshot=json.loads(snapshot_file.read_text())
    if destination.exists():raise ValueError('Keep the original sculpt preparation; choose a fresh destination')
    base_file=Path(snapshot['base_manifest']);base=json.loads(base_file.read_text());inputs=dict(base['input_sha256']);inputs.update(snapshot['input_sha256']);inputs[str(snapshot_file)]=digest(snapshot_file)
    edited=Path(snapshot['edited_file']);inputs[str(edited)]=snapshot['edited_sha256']
    def check():
        if any(digest(path)!=sha for path,sha in inputs.items()):raise ValueError('A sealed sculpt or native input changed')
    check();file=Path(snapshot['pose_file']);meta=json.loads(file.read_text())
    if meta['source_mesh']!=base['source_mesh'] or meta['lod']!=0 or meta.get('dna_mode')!='ASSEMBLED_EMBEDDED':raise ValueError('The sculpt belongs to a different native face or LOD')
    controls=snapshot['controls']
    if len(controls)!=1:raise ValueError('Author one captured eye-control corrective at a time')
    driver,value=next(iter(controls.items()));driver=driver.removeprefix('CTRL_expressions.')
    partial_blink=driver.startswith('eyeBlink')
    full_blink=partial_blink and float(value)==1.
    if not driver.startswith('eye') or driver[-1] not in 'LR' or not 1/64<=float(value)<=1:raise ValueError('Use a captured eye control from 1/64 through full strength')
    if full_blink and snapshot.get('surface_repair'):raise ValueError('Open-lid untangling cannot validate a full-blink corrective; retain the sculpt for closed-eye review')
    if full_blink:
        curves={name.lower().replace('.','_'):float(weight) for name,weight in meta.get('raw_curves',{}).items() if name.lower().startswith(('ctrl_expressions.','ctrl_expressions_'))}
        wanted='ctrl_expressions_'+driver.lower()
        if abs(curves.get(wanted,0.)-1.)>1e-6 or any(abs(weight)>1e-6 for name,weight in curves.items() if name!=wanted):raise ValueError('Use a captured single-control full blink, not a relabeled intermediate or mixed pose')
    def read(key,dtype,shape):
        path=file.parent/meta[key]
        if path.parent!=file.parent or str(path.resolve()) not in inputs:raise ValueError('A native sculpt array is not sealed')
        return np.fromfile(path,dtype=dtype).reshape(shape)
    mid=read('dna_mesh_file','<i4',(-1,));vid=read('dna_vertex_file','<i4',(-1,));pose=read('vertices_file','<f4',(-1,3));linear=read('skin_linear_file','<f4',(-1,3,3))
    def canonical(index):return canonical_offsets(pose,mid,vid,index,int(vid[mid==index].max())+1)[0]
    head=canonical(0);desired=np.fromfile(edited,dtype='<f4').reshape(-1,3);protected_file=Path(snapshot['folder'])/'protected.npz'
    if str(protected_file.resolve()) not in inputs:raise ValueError('The corrective protection mask is not sealed')
    with np.load(protected_file,allow_pickle=False) as protection:protected=protection['protected'].copy()
    if desired.shape!=head.shape or protected.shape!=(len(head),) or protected.dtype!=np.bool_ or not np.isfinite(desired).all():raise ValueError('The sculpt or protection mask no longer matches the native topology')
    restored=int(np.count_nonzero(np.linalg.norm(desired[protected]-head[protected],axis=1)>1e-5));desired[protected]=head[protected]
    delta=desired-head;delta[np.linalg.norm(delta,axis=1)<5e-5]=0;desired=head+delta
    globe=canonical(3 if driver.endswith('L') else 4);bound=float(np.ptp(globe[:,0]))*.35
    if float(np.linalg.norm(delta,axis=1).max())>bound:raise ValueError('The posed sculpt exceeds 35% of the captured eye width; reduce the correction')
    with np.load(snapshot['structure']) as structure:
        triangles=structure['outer_triangles'].copy();side='left' if driver.endswith('L') else 'right'
        margin=structure[side+'_margin'].copy();interior=structure[side+'_interior'].copy()
    repair_proof=None
    if snapshot.get('surface_repair'):
        if base.get('anatomy_topology')!='CAPTURED_RENDER_TRIANGLES_AND_UVS':raise ValueError('Prepare a fresh lid grid with render-derived anatomy before untangling')
        capture_file=file.parent/'capture.json'
        if str(capture_file.resolve()) not in inputs:raise ValueError('Keep the sealed capture index for the neutral repair reference')
        rows=json.loads(capture_file.read_text())['poses'];neutral_rows=[row for row in rows if not row['controls']]
        if len(neutral_rows)!=1:raise ValueError('The lid repair needs one sealed neutral capture')
        neutral_file=Path(neutral_rows[0]['file']).resolve()
        if neutral_file.parent!=file.parent or str(neutral_file) not in inputs:raise ValueError('The neutral repair reference is not sealed beside this pose')
        neutral_meta=json.loads(neutral_file.read_text())
        if neutral_meta['source_mesh']!=meta['source_mesh'] or neutral_meta['lod']!=0:raise ValueError('The neutral repair reference belongs to another rig')
        def neutral_array(key,dtype,shape):
            name=neutral_meta[key];path=neutral_file.parent/name
            if Path(name).name!=name or str(path.resolve()) not in inputs:raise ValueError('A neutral repair array is not sealed')
            return np.fromfile(path,dtype=dtype).reshape(shape)
        if not np.array_equal(neutral_array('dna_mesh_file','<i4',(-1,)),mid) or not np.array_equal(neutral_array('dna_vertex_file','<i4',(-1,)),vid):raise ValueError('The repair capture identities changed')
        neutral_points=neutral_array('vertices_file','<f4',(-1,3));neutral_head=canonical_offsets(neutral_points,mid,vid,0,len(head))[0]
        native_tri=read('triangles_file','<u4',(-1,3));native_head=vid[native_tri[np.all(mid[native_tri]==0,axis=1)]]
        from .lid_surface_repair import repair
        frame=np.asarray([1.,-1.,1.]);corrected,repair_proof=repair(neutral_head*frame,desired*frame,native_head,triangles,margin,interior,globe*frame)
        desired=corrected*frame;delta=desired-head;protected=np.all(delta==0,axis=1)
        if float(np.linalg.norm(delta,axis=1).max())>bound:raise ValueError('The combined sculpt/repair exceeds 35 percent of the captured eye width')
        # Include any artist edit retained outside the automatic repair collar
        # when checking the actual head field that will be exported.
        from .surface_contacts import crossings
        active=np.any(np.linalg.norm(delta,axis=1)[native_head]>1e-7,axis=1)
        remaining=crossings(desired,native_head,tolerance=float(np.linalg.norm(np.ptp(head,axis=0)))*1e-6,active_faces=active)
        if remaining:raise ValueError('The combined sculpt/repair retains proper head intersections; retain the current rig')
        repair_proof['exported_head_remaining_proper_intersections']=remaining
    if not np.any(delta):raise ValueError('Sculpt the unprotected lid before preparing a correction')
    reference=Path(base['source_dna']).with_name('reference.json')
    if str(reference.resolve()) not in inputs:raise ValueError('Keep the sealed native rig reference beside its DNA')
    index_rows={row['index']:row for row in json.loads(reference.read_text())['meshes']}
    fields={};proofs=[]
    for index in (0,6,7):
        use=mid==index
        if index==0:goal=desired[vid[use]]
        else:
            points=canonical(index);binding=bind(head,triangles,points,maximum_distance=max(3.,bound*2))
            transferred=np.einsum('ni,nij->nj',binding['barycentric'],delta[binding['triangles']]);goal=pose[use]+transferred[vid[use]]
        offsets,skin=rest_offsets(pose[use],goal,linear[use]);field,split=canonical_offsets(offsets,np.full(use.sum(),index,dtype=np.int32),vid[use],index,index_rows[index]['vertex_count'])
        if index==0:field[protected]=0
        predicted=pose[use]+np.einsum('ni,nij->nj',field[vid[use]],linear[use]);error=float(np.linalg.norm(predicted-goal,axis=1).max())
        if error>5e-4:raise ValueError('Canonical sculpt reconstruction exceeds native precision; retain the current rig')
        skin['maximum_canonical_reconstruction_error_cm']=error
        fields[index]=field;proofs.append(dict(mesh=index,skin=skin,split=split))
    matrix=np.asarray(base['coordinate_transform']);inverse=np.linalg.inv(matrix[:3]);destination.mkdir(parents=True)
    targets=[];solvers=[];patches=[];namespace='CM_sculpt_'+uuid.uuid4().hex[:8];encodings=[];encoded_maximum=0.
    for index,field in fields.items():
        if partial_blink and not full_blink:encoded,encoding=[field],dict(method='ENCODE_IN_ENDPOINT_FAMILY')
        else:encoded,encoding=_encode_small_fields(field)
        encodings.append(dict(mesh=index,**encoding))
        for part,payload in enumerate(encoded):
            ids=np.flatnonzero(np.linalg.norm(payload,axis=1)>1e-7)
            if not len(ids):continue
            encoded_maximum=max(encoded_maximum,float(np.linalg.norm(payload[ids],axis=1).max()))
            name=f'm{index}_sculpt_{part}.f32';(payload[ids]@inverse).astype('<f4').tofile(destination/name)
            sample=dict(value=float(value),file=name,indices=ids.tolist())
            if full_blink:
                family=complementary_family(destination,[sample],mesh=index_rows[index]['name'],driver='CTRL_expressions.'+driver,gate=None,radius=.25,namespace=namespace+'_m'+str(index)+'_p'+str(part),zero_values=[.25,.5,.75],multiply_driver=False)
            elif partial_blink:
                family=endpoint_family(destination,sample,mesh=index_rows[index]['name'],driver='CTRL_expressions.'+driver,namespace=namespace+'_m'+str(index)+'_p'+str(part))
                encodings.append(dict(mesh=index,endpoint_encoding=family['encoding']))
            else:
                family=complementary_family(destination,[sample],mesh=index_rows[index]['name'],driver='CTRL_expressions.'+driver,gate='CTRL_expressions.eyeBlink'+driver[-1],namespace=namespace+'_m'+str(index)+'_p'+str(part),zero_values=[1.] if float(value)<1. else [])
            targets.extend(family['targets']);solvers.extend(family['solvers'])
    for target in targets:
        index=next(i for i,row in index_rows.items() if row['name']==target['mesh']);field=np.zeros((index_rows[index]['vertex_count'],3));field[target['indices']]=np.fromfile(destination/target['file'],dtype='<f4').reshape(-1,3)@matrix[:3]
        encoded_maximum=max(encoded_maximum,float(np.linalg.norm(field,axis=1).max()))
        use=np.flatnonzero(mid==index);compiled=field[vid[use]];active=np.linalg.norm(compiled,axis=1)>1e-7;name=target['channel']+'_compiled.f32';compiled[active].astype('<f4').tofile(destination/name)
        patches.append(dict(morph=target['mesh']+'__'+target['channel'],lod=0,indices=use[active].tolist(),file=name))
    max_offset=encoded_maximum*1.001+1e-4
    (destination/'pose-request.json').write_text(json.dumps(dict(schema='CM_DNA_POSE_CORRECTIVES_V1',maximum_offset=max_offset/np.linalg.svd(matrix[:3],compute_uv=False).min(),targets=targets,solvers=solvers)))
    (destination/'compiled-request.json').write_text(json.dumps(dict(schema='CM_COMPILED_CORRECTIVE_V1',maximum_offset_cm=max_offset,allow_new_dna_targets=True,patches=patches)))
    manifest=dict(schema='CM_NATIVE_LOD_CORRECTIVE_REQUEST_V1',status='prepared',source_mesh=base['source_mesh'],source_dna=base['source_dna'],source_asset_file=base['source_asset_file'],source_asset_sha256=base['source_asset_sha256'],native_source_sha256=base.get('native_source_sha256',{}),targets=len(targets),lods=[0],method='PROTECTED_CAPTURED_POSE_SCULPT',coordinate_transform=matrix.tolist(),input_sha256=inputs,prepared_sha256={p.name:digest(p) for p in destination.iterdir() if p.is_file()},neutral_geometry_changed=False,globe_geometry_changed=False,native_motion_accepted=False,minimum_bridge='0.27.0',driver=driver,sculpt_strength=value,fade_control='eyeBlink'+driver[-1],protected_vertices=int(protected.sum()),protected_edits_restored=restored,pose_proofs=proofs)
    if repair_proof:
        manifest['method']='RENDER_CAPTURED_POSE_LID_REPAIR';manifest['surface_repair']=repair_proof
    manifest['retain_full_strength_pose']=float(value)<1.
    if full_blink:
        manifest.update(method='PROTECTED_CAPTURED_FULL_BLINK_SCULPT',strength_interpolation='TERMINAL_SCALAR_HAT',retained_strengths=[0.,.25,.5,.75],fade_control=None,closed_eye_geometry_review_required=True)
    elif partial_blink:manifest['strength_interpolation']='ENDPOINT_PRESERVING_SCALAR_HAT'
    manifest['field_encoding']=encodings
    check()
    (destination/'manifest.json').write_text(json.dumps(manifest,indent=2))
    result=dict(status='prepared',folder=str(destination),targets=len(targets),source_mesh=base['source_mesh'],method=manifest['method'],structure_file=snapshot['structure'],retain_full_strength_pose=float(value)<1.,native_motion_accepted=False)
    if full_blink:result.update(strength_interpolation='TERMINAL_SCALAR_HAT',retained_strengths=[0.,.25,.5,.75],closed_eye_geometry_review_required=True)
    elif partial_blink:result['strength_interpolation']='ENDPOINT_PRESERVING_SCALAR_HAT'
    if repair_proof:result['surface_repair']=repair_proof
    (destination/'preparation-summary.json').write_text(json.dumps(result,indent=2));return result
