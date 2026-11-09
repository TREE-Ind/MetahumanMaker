"""Prepare a bounded neutral lid refit from sealed native geometry.

Changes LOD0 head and attached eye-edge/lash neutrals. Optional native blink
fields cancel each eye's rest edit from quarter strength onward, retaining the
other eye's neutral fit. Native globe arrays, joints, skin weights and existing
expression targets are retained. This is a separate proposal, not acceptance of
expressions or lower-LOD transitions.
"""
import json,uuid
from pathlib import Path
import numpy as np
from .calibration_math import canonical_offsets,rest_offsets
from .native_lid_contact import digest
from .lid_surface_repair import repair,inspect_pose
from .ocular_attachments import bind
from .native_corrective_gates import complementary_family,write_pose_groups


def prepare(capture_folder,base_manifest,destination,*,progress=None,compensate_blinks=True):
    capture_folder=Path(capture_folder).resolve();base_file=Path(base_manifest).resolve();destination=Path(destination).resolve()
    if type(compensate_blinks) is not bool:raise ValueError('Choose whether to prepare independent blink compensation')
    if destination.exists():raise ValueError('Retain the previous neutral fit; choose a fresh destination')
    base=json.loads(base_file.read_text());inputs=dict(base['input_sha256']);inputs[str(base_file)]=digest(base_file)
    def check():
        if any(digest(path)!=sha for path,sha in inputs.items()):raise ValueError('A sealed neutral-fit input changed')
    def sealed(path):
        path=Path(path).resolve()
        if str(path) not in inputs or digest(path)!=inputs[str(path)]:raise ValueError('A neutral-fit input is not sealed')
        return path
    check()
    if base.get('anatomy_topology')!='CAPTURED_RENDER_TRIANGLES_AND_UVS':raise ValueError('Prepare current render-derived lid anatomy first')
    capture=json.loads(sealed(capture_folder/'capture.json').read_text());rows=[row for row in capture['poses'] if not row['controls'] and row['lod']==0]
    if capture.get('status')!='complete' or capture.get('candidate')!=base['source_mesh'] or len(rows)!=1:raise ValueError('Keep one sealed neutral capture of the connected saved face')
    file=sealed(rows[0]['file'])
    if file.parent!=capture_folder:raise ValueError('Keep neutral metadata beside the captured arrays')
    meta=json.loads(file.read_text())
    if meta.get('source_mesh')!=base['source_mesh'] or meta.get('dna_mode')!='ASSEMBLED_EMBEDDED':raise ValueError('The neutral capture belongs to another native rig')
    def array(key,dtype,shape):
        name=meta[key]
        if Path(name).name!=name:raise ValueError('Keep neutral arrays beside their metadata')
        return np.fromfile(sealed(file.parent/name),dtype=dtype).reshape(shape)
    mid=array('dna_mesh_file','<i4',(-1,));vid=array('dna_vertex_file','<i4',(-1,));pose=array('vertices_file','<f4',(-1,3));rest=array('neutral_file','<f4',(-1,3));linear=array('skin_linear_file','<f4',(-1,3,3));tri=array('triangles_file','<u4',(-1,3))
    reference_file=sealed(Path(base['source_dna']).with_name('reference.json'));meshes={row['index']:row for row in json.loads(reference_file.read_text())['meshes']}
    def canonical(index,points=pose):return canonical_offsets(points,mid,vid,index,meshes[index]['vertex_count'])[0]
    summary=json.loads((base_file.parent/'preparation-summary.json').read_text());structure=sealed(summary['structure_file']);anatomy=np.load(structure)
    triangles=vid[tri[np.all(mid[tri]==0,axis=1)]];head=canonical(0);frame=np.array([1.,-1.,1.]);candidate=head*frame;before={};proofs={};side_deltas={};notify=progress or (lambda message:None)
    for side,index in [('left',3),('right',4)]:
        eye=canonical(index)*frame
        before[side]=inspect_pose(head*frame,candidate,triangles,anatomy['outer_triangles'],anatomy[side+'_margin'],anatomy[side+'_interior'],eye)
        if before[side]['valid']:continue
        preceding=candidate.copy()
        candidate,proof=repair(head*frame,candidate,triangles,anatomy['outer_triangles'],anatomy[side+'_margin'],anatomy[side+'_interior'],eye,preserve_projection=True,progress=lambda text:notify(side.title()+': '+text))
        side_deltas[side]=(candidate-preceding)*frame
        proof['neutral_geometry_changed']=True;proofs[side]=proof;check()
    final={side:inspect_pose(head*frame,candidate,triangles,anatomy['outer_triangles'],anatomy[side+'_margin'],anatomy[side+'_interior'],canonical(index)*frame) for side,index in [('left',3),('right',4)]}
    if not all(row['valid'] for row in final.values()):raise ValueError('Both neutral lids must pass after the complete fit; preceding proposal retained')
    visible=np.unique(anatomy['outer_triangles'])
    projection_error=float(np.linalg.norm((candidate-head*frame)[visible][:,[0,2]],axis=1).max())
    if projection_error>2e-4:raise ValueError('The neutral fit changed the visible projected lid shape beyond native precision')
    if not proofs:return dict(status='unchanged',message='Both captured neutral lids already pass. Previous proposal retained.',source_mesh=base['source_mesh'],native_motion_accepted=False)
    delta=candidate*frame-head;matrix=np.asarray(base['coordinate_transform']);inverse=np.linalg.inv(matrix[:3]);payloads=[];reconstruction=[];maximum=0.;compensation={}
    for index in (0,6,7):
        if index not in meshes or not np.any(mid==index):raise ValueError('Keep complete head, eye-edge and eyelash identities for neutral fitting')
        points=canonical(index);use=mid==index
        if index==0:
            transferred=delta;side_transfers=side_deltas
        else:
            binding=bind(head,triangles,points,maximum_distance=3.)
            transferred=np.einsum('ni,nij->nj',binding['barycentric'],delta[binding['triangles']])
            side_transfers={side:np.einsum('ni,nij->nj',binding['barycentric'],values[binding['triangles']]) for side,values in side_deltas.items()}
        offsets,skin=rest_offsets(pose[use],pose[use]+transferred[vid[use]],linear[use]);field,_=canonical_offsets(offsets,np.full(use.sum(),index,dtype=np.int32),vid[use],index,meshes[index]['vertex_count'])
        path=sealed(reference_file.parent/meshes[index]['positions_file']);original=np.fromfile(path,dtype='<f4').reshape(-1,3)
        if np.linalg.norm(original@matrix[:3]+matrix[3]-canonical(index,rest),axis=1).max()>5e-4:raise ValueError('The native neutral reference and captured rest geometry disagree')
        output=(original+field@inverse).astype('<f4');effective=(output-original)@matrix[:3]
        predicted=pose[use]+np.einsum('ni,nij->nj',effective[vid[use]],linear[use]);error=float(np.linalg.norm(predicted-(pose[use]+transferred[vid[use]]),axis=1).max())
        if error>2e-4:raise ValueError('Quantized neutral reconstruction exceeds native precision')
        if compensate_blinks:
            compensation[index]={}
            for side,values in side_transfers.items():
                side_rest,_=rest_offsets(pose[use],pose[use]+values[vid[use]],linear[use])
                side_field,_=canonical_offsets(side_rest,np.full(use.sum(),index,dtype=np.int32),vid[use],index,meshes[index]['vertex_count'])
                compensation[index][side]=-side_field
            residual=effective+sum(compensation[index].values())
            if np.linalg.norm(residual,axis=1).max()>2e-4:raise ValueError('Independent eye compensation does not reconstruct the complete neutral fit')
        maximum=max(maximum,float(np.linalg.norm(output-original,axis=1).max()));payloads.append((meshes[index]['name'],output));reconstruction.append(dict(mesh=index,maximum_error_cm=error,skin=skin))
    check();destination.mkdir(parents=True);neutral=[]
    for number,(name,points) in enumerate(payloads):
        filename='neutral-'+str(number)+'.f32';points.tofile(destination/filename);neutral.append(dict(mesh=name,file=filename))
    request=dict(schema='CM_DNA_CALIBRATION_V1',maximum_offset=maximum*1.001+1e-4,neutral=neutral,correctives=[])
    (destination/'calibration-request.json').write_text(json.dumps(request,indent=2))
    targets=[];solvers=[];pose_requests=[]
    if compensate_blinks:
        namespace='CM_neutral_'+uuid.uuid4().hex[:8];bound=0.
        for index,members in compensation.items():
            for side,field in members.items():
                ids=np.flatnonzero(np.linalg.norm(field,axis=1)>1e-7)
                if not len(ids):continue
                name=f'compensation-{index}-{side}.f32';(field[ids]@inverse).astype('<f4').tofile(destination/name)
                rows=[dict(value=value,file=name,indices=ids.tolist()) for value in (.25,.5,.75,1.)]
                plan=complementary_family(destination,rows,mesh=meshes[index]['name'],driver='CTRL_expressions.eyeBlink'+('L' if side=='left' else 'R'),gate=None,radius=.25,namespace=namespace+f'_m{index}_'+side[0],multiply_driver=False)
                targets.extend(plan['targets']);solvers.extend(plan['solvers']);bound=max(bound,float(np.linalg.norm(field[ids],axis=1).max()))
        if not targets:raise ValueError('The neutral fit has no nonzero blink compensation')
        pose_requests=write_pose_groups(destination,targets,solvers,(bound*1.001+1e-4)/np.linalg.svd(matrix[:3],compute_uv=False).min())
        patches=[]
        for target in targets:
            index=next(i for i,row in meshes.items() if row['name']==target['mesh']);field=np.zeros((meshes[index]['vertex_count'],3));field[target['indices']]=np.fromfile(destination/target['file'],dtype='<f4').reshape(-1,3)@matrix[:3]
            use=np.flatnonzero(mid==index);compiled=field[vid[use]];active=np.linalg.norm(compiled,axis=1)>1e-7;name=target['channel']+'-compiled.f32';compiled[active].astype('<f4').tofile(destination/name)
            patches.append(dict(morph=target['mesh']+'__'+target['channel'],lod=0,indices=use[active].tolist(),file=name))
        (destination/'compiled-request.json').write_text(json.dumps(dict(schema='CM_COMPILED_CORRECTIVE_V1',maximum_offset_cm=bound*1.001+1e-4,allow_new_dna_targets=True,patches=patches)))
    manifest=dict(base,status='prepared',method='CAPTURED_NEUTRAL_LID_BLINK_FIT' if compensate_blinks else 'CAPTURED_NEUTRAL_LID_FIT',targets=len(targets),lods=[0],pose_requests=pose_requests,neutral_geometry_changed=True,globe_geometry_changed=False,neutral_fit=dict(before=before,final=final,repairs=proofs,reconstruction=reconstruction,maximum_visible_projection_error_cm=projection_error,blink_compensation=bool(compensate_blinks),compensation_full_strength=.25),input_sha256=inputs,prepared_sha256={p.name:digest(p) for p in destination.iterdir() if p.is_file()},native_motion_accepted=False)
    # A previous expression bank supplies sealed anatomy, not the diagnostics
    # or driver metadata of this different proposal.
    for key in ('neutral_inspection','warnings','driver','repaired_strengths','retained_strengths','bank_inspections','bank_repairs','field_encoding','anchor_strengths','strength_interpolation','independent_blink_fade','pose_proofs'):
        manifest.pop(key,None)
    check();(destination/'manifest.json').write_text(json.dumps(manifest,indent=2))
    result=dict(status='prepared',folder=str(destination),targets=len(targets),source_mesh=base['source_mesh'],method=manifest['method'],structure_file=str(structure),neutral_fit=manifest['neutral_fit'],neutral_geometry_changed=True,globe_geometry_changed=False,native_motion_accepted=False,warnings=['Neutral head and eye attachments change at LOD0. Blink compensation retains fitted neutral at zero and removes each side contribution from quarter strength onward. Review expressions and LOD transitions, then recapture the staged rig.'] if compensate_blinks else ['Neutral head and eye attachments change at LOD0. Review every expression and LOD transition, then capture the staged rig again.'])
    (destination/'preparation-summary.json').write_text(json.dumps(result,indent=2));return result
