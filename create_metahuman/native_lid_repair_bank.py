"""Prepare jointly driven intermediate repairs from one sealed native capture."""
import json,re,uuid
from pathlib import Path
import numpy as np
from .calibration_math import canonical_offsets
from .native_lid_contact import digest
from .lid_surface_repair import inspect_pose
from .native_pose_sculpt_solver import prepare as prepare_single,_encode_small_fields
from .native_corrective_gates import complementary_family,endpoint_bank,write_pose_groups


def prepare(capture_folder,base_manifest,driver,destination,*,progress=None,intervals=False):
    capture_folder=Path(capture_folder).resolve();base_file=Path(base_manifest).resolve();destination=Path(destination).resolve()
    if destination.exists():raise ValueError('Retain the original repair bank; choose a fresh destination')
    if not re.fullmatch(r'eye[A-Za-z0-9_]+[LR]',driver):raise ValueError('Choose one captured eye control')
    partial_blink=driver.startswith('eyeBlink')
    base=json.loads(base_file.read_text());inputs=dict(base['input_sha256']);inputs[str(base_file)]=digest(base_file)
    if base.get('anatomy_topology')!='CAPTURED_RENDER_TRIANGLES_AND_UVS':raise ValueError('Prepare current render-derived lid anatomy first')
    def check():
        if any(digest(path)!=sha for path,sha in inputs.items()):raise ValueError('A sealed repair-bank source changed')
    def sealed(path):
        path=Path(path).resolve()
        if str(path) not in inputs or digest(path)!=inputs[str(path)]:raise ValueError('A repair-bank input is not sealed')
        return path
    check();capture=json.loads(sealed(capture_folder/'capture.json').read_text())
    if capture.get('status')!='complete' or capture.get('candidate')!=base['source_mesh']:raise ValueError('Keep the complete capture of the connected saved face')
    summary=json.loads((base_file.parent/'preparation-summary.json').read_text());structure=sealed(summary.get('structure_file',base_file.parent.parent/'inputs/lid-structure.npz'))
    anatomy=np.load(structure);side='left' if driver.endswith('L') else 'right';eye_index=3 if side=='left' else 4;frame=np.asarray([1.,-1.,1.]);notify=progress or (lambda message:None)
    def load(row):
        file=sealed(row['file'])
        if file.parent!=capture_folder or row['lod']!=0:raise ValueError('Repair-bank samples must be captured LOD0 poses')
        meta=json.loads(file.read_text())
        if meta.get('source_mesh')!=base['source_mesh'] or meta.get('dna_mode')!='ASSEMBLED_EMBEDDED':raise ValueError('A repair-bank sample belongs to a different rig')
        def array(key,dtype,shape):
            name=meta[key]
            if Path(name).name!=name:raise ValueError('Keep captured arrays beside their metadata')
            return np.fromfile(sealed(file.parent/name),dtype=dtype).reshape(shape)
        mid=array('dna_mesh_file','<i4',(-1,));vid=array('dna_vertex_file','<i4',(-1,));pose=array('vertices_file','<f4',(-1,3));tri=array('triangles_file','<u4',(-1,3))
        head=canonical_offsets(pose,mid,vid,0,int(vid[mid==0].max())+1)[0];eye=canonical_offsets(pose,mid,vid,eye_index,int(vid[mid==eye_index].max())+1)[0]
        return file,head,eye,vid[tri[np.all(mid[tri]==0,axis=1)]]
    neutral_rows=[row for row in capture['poses'] if not row['controls']]
    if len(neutral_rows)!=1:raise ValueError('Keep exactly one neutral repair-bank sample')
    _,neutral,neutral_eye,triangles=load(neutral_rows[0]);samples={}
    # Every bank keeps its zero-strength anchor. Inspect that actual captured
    # pose too: valid intermediates cannot establish clearance near an invalid
    # neutral when the correction continuously tends to zero there.
    neutral_proof=inspect_pose(neutral*frame,neutral*frame,triangles,anatomy['outer_triangles'],anatomy[side+'_margin'],anatomy[side+'_interior'],neutral_eye*frame)
    warnings=[] if neutral_proof['valid'] else [
        'The captured neutral lid fails geometry checks. This bank retains neutral; '
        'fit the neutral lid to the globe and recapture before accepting motion.']
    for row in capture['poses']:
        controls={key.removeprefix('CTRL_expressions.'):value for key,value in row['controls'].items()}
        if len(controls)==1 and driver in controls and 1/64<=float(controls[driver])<1:
            value=float(controls[driver])
            if value in samples:raise ValueError('Keep distinct captured strengths for the repair bank')
            samples[value]=row
    if not samples or len(samples)>(15 if intervals else 7):raise ValueError('Choose a compatible capture: Dense supplies seven intermediate strengths; interval repair needs fifteen')
    anchor_strengths=None
    if intervals:
        strengths=np.asarray([0.,*sorted(samples),1.])
        expected=9 if partial_blink else 17
        if len(strengths)!=expected or not np.allclose(strengths,np.linspace(0,1,expected),atol=1e-12,rtol=0):raise ValueError('Capture Dense Blink Strengths for blink intervals, or Interval Blink and Squint for other eye controls')
        anchor_strengths=strengths[::2].tolist()
    inspections=[dict(strength=0.,anchor=True,**neutral_proof)];pending=[]
    for value,row in sorted(samples.items()):
        notify('Checking '+driver+' at '+str(value));file,head,eye,tri=load(row)
        if not np.array_equal(tri,triangles):raise ValueError('Native render topology changed between bank samples')
        proof=inspect_pose(neutral*frame,head*frame,tri,anatomy['outer_triangles'],anatomy[side+'_margin'],anatomy[side+'_interior'],eye*frame)
        inspections.append(dict(strength=value,**proof))
        if not proof['valid']:
            if intervals and value in anchor_strengths:raise ValueError('An interval anchor fails geometry checks at '+str(value)+('; stage the quarter-strength blink repairs and capture that saved candidate first' if partial_blink else '; repair the Dense captured poses first'))
            pending.append((value,file,head))
    if intervals:
        full=[row for row in capture['poses'] if {key.removeprefix('CTRL_expressions.'):value for key,value in row['controls'].items()}=={driver:1.}]
        if len(full)!=1:raise ValueError('Keep exactly one full-strength interval anchor')
        _,head,eye,tri=load(full[0])
        if not np.array_equal(tri,triangles):raise ValueError('The full-strength render topology changed')
        # A closed blink intentionally collapses the aperture projection. Retain
        # that source endpoint exactly; do not call it a passing open-lid pose.
        proof=(dict(valid=None,retained_closed_endpoint=True,native_motion_accepted=False) if partial_blink else
               inspect_pose(neutral*frame,head*frame,tri,anatomy['outer_triangles'],anatomy[side+'_margin'],anatomy[side+'_interior'],eye*frame))
        inspections.append(dict(strength=1.,anchor=True,**proof))
        if not partial_blink and not proof['valid']:raise ValueError('Repair the full-strength anchor before preparing interval motion')
    inspection_file=destination.parent/'bank-inspection.json';inspection_file.write_text(json.dumps(inspections,indent=2))
    if not pending:return dict(status='unchanged',source_mesh=base['source_mesh'],message=' '.join(['All captured intermediate strengths pass the endpoint checks. Previous proposal retained.',*warnings]),neutral_inspection=neutral_proof,warnings=warnings,native_motion_accepted=False)
    work=destination.parent/'bank-authoring';work.mkdir(exist_ok=False);children=[];all_strengths=sorted([*samples,1.]);matrix=np.asarray(base['coordinate_transform']);inverse=np.linalg.inv(matrix[:3])
    reference=sealed(Path(base['source_dna']).with_name('reference.json'));meshes={row['index']:row for row in json.loads(reference.read_text())['meshes']};names={row['name']:index for index,row in meshes.items()}
    for number,(value,file,head) in enumerate(pending):
        notify('Repairing '+driver+' at '+str(value)+' ('+str(number+1)+' of '+str(len(pending))+')')
        folder=work/('sample-'+str(number));folder.mkdir();edited=folder/'edited.f32';head.astype('<f4').tofile(edited);protection=folder/'protected.npz';np.savez_compressed(protection,protected=np.ones(len(head),dtype=bool))
        snapshot=dict(folder=str(folder),pose_file=str(file),base_manifest=str(base_file),structure=str(structure),source_mesh=base['source_mesh'],controls={driver:value},surface_repair=True,edited_file=str(edited),edited_sha256=digest(edited),input_sha256={str(protection):digest(protection)})
        snapshot_file=folder/'sculpt.json';snapshot_file.write_text(json.dumps(snapshot,indent=2));proposal=prepare_single(snapshot_file,folder/'prepared');child=Path(proposal['folder']);manifest=json.loads((child/'manifest.json').read_text());request=json.loads((child/'pose-request.json').read_text());fields={}
        for target in request['targets']:
            if '_on_' not in target['channel']:continue
            index=names[target['mesh']];field=fields.setdefault(index,np.zeros((meshes[index]['vertex_count'],3)))
            file=child/target['file'];field[target['indices']]+=np.fromfile(file,dtype='<f4').reshape(-1,3)@matrix[:3];inputs[str(file)]=digest(file)
        inputs.update(manifest['input_sha256']);inputs[str(child/'manifest.json')]=digest(child/'manifest.json');children.append((value,fields,proposal['surface_repair']))
    check();destination.mkdir();targets=[];solvers=[];encodings=[];payloads={};encoded_maximum=0.;namespace='CM_bank_'+uuid.uuid4().hex[:8]
    if intervals:
        from .native_interval_fields import family as interval_family
        midpoint_strengths=[float(value) for value in strengths[1::2]]
        for index in sorted({index for _,fields,_ in children for index in fields}):
            midpoint=np.zeros((len(midpoint_strengths),meshes[index]['vertex_count'],3))
            for value,fields,_ in children:midpoint[midpoint_strengths.index(value)]=fields[index]
            family=interval_family(destination,midpoint,knots=anchor_strengths,mesh=meshes[index]['name'],driver='CTRL_expressions.'+driver,gate=None if partial_blink else 'CTRL_expressions.eyeBlink'+driver[-1],coordinate_linear=matrix[:3],namespace=namespace+f'_m{index}')
            targets.extend(family['targets']);solvers.extend(family['solvers']);encodings.extend(dict(mesh=index,**row) for row in family['field_encoding']);encoded_maximum=max(encoded_maximum,family['maximum_encoded_offset_cm'])
    else:
        for value,fields,_ in children:
            for index,field in fields.items():
                encoded,proof=_encode_small_fields(field);encodings.append(dict(strength=value,mesh=index,**proof))
                for part,payload in enumerate(encoded):payloads.setdefault((index,part),{})[value]=payload
        for (index,part),fields in sorted(payloads.items()):
            rows=[]
            for number,(value,field) in enumerate(sorted(fields.items())):
                ids=np.flatnonzero(np.linalg.norm(field,axis=1)>1e-7)
                if not len(ids):continue
                name=f'm{index}_p{part}_s{number}.f32';(field[ids]@inverse).astype('<f4').tofile(destination/name);rows.append(dict(value=value,file=name,indices=ids.tolist()));encoded_maximum=max(encoded_maximum,float(np.linalg.norm(field[ids],axis=1).max()))
            if not rows:continue
            zero_values=[value for value in all_strengths if value not in {row['value'] for row in rows}]
            if partial_blink:
                family=endpoint_bank(destination,rows,mesh=meshes[index]['name'],driver='CTRL_expressions.'+driver,namespace=namespace+f'_m{index}_p{part}',zero_values=zero_values)
            else:
                family=complementary_family(destination,rows,mesh=meshes[index]['name'],driver='CTRL_expressions.'+driver,gate='CTRL_expressions.eyeBlink'+driver[-1],namespace=namespace+f'_m{index}_p{part}',zero_values=zero_values,multiply_driver=False)
            targets.extend(family['targets']);solvers.extend(family['solvers'])
    meta=json.loads(sealed(neutral_rows[0]['file']).read_text());mid=np.fromfile(sealed(capture_folder/meta['dna_mesh_file']),dtype='<i4');vid=np.fromfile(sealed(capture_folder/meta['dna_vertex_file']),dtype='<i4');patches=[]
    for target in targets:
        index=names[target['mesh']];field=np.zeros((meshes[index]['vertex_count'],3));field[target['indices']]=np.fromfile(destination/target['file'],dtype='<f4').reshape(-1,3)@matrix[:3]
        encoded_maximum=max(encoded_maximum,float(np.linalg.norm(field,axis=1).max()))
        # Canonical fields map directly to the sealed render identities.
        use=np.flatnonzero(mid==index);compiled=field[vid[use]];active=np.linalg.norm(compiled,axis=1)>1e-7;name=target['channel']+'_compiled.f32';compiled[active].astype('<f4').tofile(destination/name);patches.append(dict(morph=target['mesh']+'__'+target['channel'],lod=0,indices=use[active].tolist(),file=name))
    bound=encoded_maximum*1.001+1e-4
    pose_requests=write_pose_groups(destination,targets,solvers,bound/np.linalg.svd(matrix[:3],compute_uv=False).min())
    (destination/'compiled-request.json').write_text(json.dumps(dict(schema='CM_COMPILED_CORRECTIVE_V1',maximum_offset_cm=bound,allow_new_dna_targets=True,patches=patches)))
    repaired=[value for value,_,_ in children];retained=[value for value in all_strengths if value not in repaired]
    manifest=dict(base,schema='CM_NATIVE_LOD_CORRECTIVE_REQUEST_V1',status='prepared',targets=len(targets),lods=[0],method='CAPTURED_INTERVAL_REPAIR_BANK' if intervals else 'CAPTURED_INTERMEDIATE_REPAIR_BANK',driver=driver,input_sha256=inputs,prepared_sha256={p.name:digest(p) for p in destination.iterdir() if p.is_file()},repaired_strengths=repaired,retained_strengths=[0.,*retained],field_encoding=encodings,bank_inspections=inspections,bank_repairs=[dict(strength=v,proof=proof) for v,_,proof in children],neutral_geometry_changed=False,globe_geometry_changed=False,native_motion_accepted=False)
    manifest.update(pose_requests=pose_requests,strength_interpolation='ANCHOR_PRESERVING_QUADRATIC_INTERVAL_RESIDUAL' if intervals else 'LINEAR_NEUTRAL_KNOT_WITHOUT_EXTRA_DRIVER_MULTIPLIER')
    manifest.update(neutral_inspection=neutral_proof,warnings=warnings)
    if partial_blink and not intervals:manifest['strength_interpolation']='ENDPOINT_PRESERVING_SCALAR_BLINK_BANK'
    if partial_blink and intervals:manifest['independent_blink_fade']=False
    if intervals:manifest['anchor_strengths']=anchor_strengths
    check();(destination/'manifest.json').write_text(json.dumps(manifest,indent=2));result=dict(status='prepared',folder=str(destination),targets=len(targets),source_mesh=base['source_mesh'],method=manifest['method'],structure_file=str(structure),driver=driver,repaired_strengths=repaired,retained_strengths=[0.,*retained],native_motion_accepted=False)
    result.update(neutral_inspection=neutral_proof,warnings=warnings)
    if partial_blink:result['strength_interpolation']=manifest['strength_interpolation']
    (destination/'preparation-summary.json').write_text(json.dumps(result,indent=2));return result
