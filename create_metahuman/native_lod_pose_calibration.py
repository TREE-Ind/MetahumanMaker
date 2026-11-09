"""Calibrate lower LODs to complete evaluated native facial motion.

Independent side fields model unilateral motion. A separate bilateral field
models the additional interaction, avoiding a sparse full-control solver that
loses its correction when the other eye changes state. The native bridge still
has to validate, stage and visually review the resulting private candidate.
"""
import hashlib,json
from pathlib import Path
import numpy as np
from .calibration_math import rest_offsets,canonical_offsets,apply_rest_offsets
from .lod_surface_transfer import transport_evaluated_motion
from .ocular_pose_domain import states,motion_channel,interaction_channel


def prepare(capture_folder,reference_file,binding_folder,domain_file,coordinate_transform,destination,*,maximum_offset_cm=15.,duplicate_tolerance_cm=.02,target_refiner=None):
    capture_folder=Path(capture_folder).resolve();reference_file=Path(reference_file).resolve();binding_folder=Path(binding_folder).resolve();domain_file=Path(domain_file).resolve();destination=Path(destination).resolve()
    capture=json.loads((capture_folder/'capture.json').read_text());reference=json.loads(reference_file.read_text());domain=json.loads(domain_file.read_text());matrix=np.asarray(coordinate_transform,dtype=float)
    if capture.get('status')!='complete' or reference.get('coordinates')!='PRESERVED_DNA' or matrix.shape!=(4,3) or not np.isfinite(matrix).all():raise ValueError('Supply complete native captures, DNA anatomy and their coordinate transform')
    if destination.exists():raise ValueError('Choose a fresh evaluated-motion calibration folder')
    if not np.isfinite(maximum_offset_cm) or maximum_offset_cm<=0:raise ValueError('Choose a positive native corrective bound')
    inverse=np.linalg.inv(matrix[:3]);source_mesh=capture['candidate'];names=[row['name'] for row in domain];lookup={name:i for i,name in enumerate(names)};neutral_index=lookup.get('Neutral')
    if len(names)!=61 or len(lookup)!=61 or neutral_index is None:raise ValueError('Capture the complete factorized eye-control domain first')
    meshes={m['name']:m for m in reference['meshes']};inputs={};rows=[];targets=[]
    def digest(file):
        with file.open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()
    def read_file(file,dtype):
        inputs[str(file)]=digest(file);return np.fromfile(file,dtype=dtype)
    def frame(lod,name,linear=False):
        file=capture_folder/f'LOD{lod}_{name}.json';meta=json.loads(file.read_text());inputs[str(file)]=digest(file)
        if meta.get('source_mesh')!=source_mesh or meta.get('lod')!=lod:raise ValueError('A pose capture belongs to another rig or LOD')
        def path(key):
            value=meta[key]
            if Path(value).name!=value:raise ValueError('Native capture binaries must remain beside their pose metadata')
            return capture_folder/value
        positions=read_file(path('vertices_file'),'<f4').reshape(-1,3)
        if not np.isfinite(positions).all() or len(positions)!=meta['vertex_count']:raise ValueError('A native pose has invalid positions')
        return meta,positions,read_file(path('skin_linear_file'),'<f4').reshape(-1,3,3) if linear else None
    def identities(meta):
        mids=read_file(capture_folder/meta['dna_mesh_file'],'<i4');vids=read_file(capture_folder/meta['dna_vertex_file'],'<i4')
        if len(mids)!=meta['vertex_count'] or len(vids)!=len(mids):raise ValueError('Native pose identities are incomplete')
        return mids,vids
    def canonical(values,select,ids,count):
        chosen=ids[select];number=np.bincount(chosen,minlength=count)
        if chosen.min()<0 or chosen.max()>=count or np.any(number==0):raise ValueError('Native pose lacks complete canonical correspondence')
        result=np.column_stack([np.bincount(chosen,weights=values[select,j],minlength=count)/number for j in range(3)])
        if np.linalg.norm(values[select]-result[chosen],axis=1).max(initial=0)>.0005:raise ValueError('Native split vertices disagree before calibration')
        return result
    inputs[str(reference_file)]=digest(reference_file);inputs[str(domain_file)]=digest(domain_file);inputs[str(capture_folder/'capture.json')]=digest(capture_folder/'capture.json')
    high_meta,_,_=frame(0,'Neutral');high_mids,high_vids=identities(high_meta);families={}
    for mesh in reference['meshes']:
        if 0 in mesh.get('lods',[]) and any((binding_folder/(str(m['index'])+'-binding.npz')).exists() for m in reference['meshes'] if m['name'].split('_lod')[0]==mesh['name'].split('_lod')[0] and m['index']!=mesh['index']):
            families[mesh['name']]=mesh
    high={name:[] for name in families}
    for name in names:
        _,positions,_=frame(0,name)
        for family,mesh in families.items():high[family].append(canonical(positions,high_mids==mesh['index'],high_vids,mesh['vertex_count']))
    high={name:np.asarray(values) for name,values in high.items()};destination.mkdir(parents=True)
    refinements=[]
    if target_refiner is not None:
        for file,expected in getattr(target_refiner,'input_sha256',{}).items():
            if digest(Path(file))!=expected:raise ValueError('An anatomical refinement input changed')
            inputs[file]=expected
    for lod in range(1,reference['lod_count']):
        meta,neutral,_=frame(lod,'Neutral');mids,vids=identities(meta);positions=[];linear=[]
        for name in names:
            _,v,j=frame(lod,name,True);positions.append(v);linear.append(j)
        positions=np.asarray(positions);linear=np.asarray(linear)
        goals={};selected_meshes=[]
        for mesh in reference['meshes']:
            file=binding_folder/(str(mesh['index'])+'-binding.npz');family=mesh['name'].split('_lod')[0]+'_lod0_mesh'
            if lod not in mesh.get('lods',[]) or family not in high or not file.exists():continue
            inputs[str(file)]=digest(file);data=np.load(file);binding={k:data[k].item() if data[k].ndim==0 else data[k] for k in data.files};select=mids==mesh['index'];n=mesh['vertex_count'];neutral_canonical=canonical(neutral,select,vids,n);desired=[]
            for pose in high[family]:
                goal,_=transport_evaluated_motion(high[family][neutral_index],pose,neutral_canonical,binding,duplicate_tolerance=duplicate_tolerance_cm);desired.append(goal)
            goals[mesh['name']]=np.asarray(desired);selected_meshes.append(mesh)
        if target_refiner is not None:
            goals,proof=target_refiner(lod,goals,names);refinements.append(dict(lod=lod,proof=proof))
        for mesh in selected_meshes:
            select=mids==mesh['index'];n=mesh['vertex_count'];desired=np.asarray(goals[mesh['name']]);goal=desired[:,vids[select]];j=linear[:,select];current=positions[:,select]
            rest,diagnostic=rest_offsets(current.reshape(-1,3),goal.reshape(-1,3),j.reshape(-1,3,3));rest=rest.reshape(len(names),-1,3);folded=[];maximum_split=0.
            for offsets in rest:
                value,proof=canonical_offsets(offsets,np.full(select.sum(),mesh['index'],dtype=int),vids[select],mesh['index'],n);folded.append(value);maximum_split=max(maximum_split,proof['maximum_split_vertex_disagreement'])
            folded=np.asarray(folded);neutral_error=float(np.linalg.norm(rest[neutral_index],axis=1).max(initial=0))
            if neutral_error>.0001:raise ValueError('Evaluated motion would change the target neutral geometry')
            rebuilt=apply_rest_offsets(current.reshape(-1,3),folded[:,vids[select]].reshape(-1,3),j.reshape(-1,3,3));maximum_error=float(np.linalg.norm(rebuilt-goal.reshape(-1,3),axis=1).max(initial=0))
            if maximum_error>.001:raise ValueError('Native skinning cannot represent the requested canonical motion within its calibration bound')
            for state in states()[1:]:
                state=state['name'];left=folded[lookup['L'+state]];right=folded[lookup['R'+state]];interaction=folded[lookup['Both'+state]]-left-right
                for channel,field in [(motion_channel('L',state),left),(motion_channel('R',state),right),(interaction_channel(state),interaction)]:
                    if np.linalg.norm(field,axis=1).max(initial=0)>maximum_offset_cm:raise ValueError('A fitted native pose field exceeds its physical bound')
                    chosen=np.flatnonzero(np.linalg.norm(field,axis=1)>1e-7)
                    if not len(chosen):continue
                    name=str(mesh['index'])+'_'+channel+'.f32';(field[chosen]@inverse).astype('<f4').tofile(destination/name);targets.append(dict(mesh=mesh['name'],lod=lod,channel=channel,indices=chosen.tolist(),file=name))
            np.savez_compressed(destination/(str(mesh['index'])+'-desired.npz'),positions=desired,names=np.asarray(names));rows.append(dict(mesh=mesh['name'],lod=lod,poses=len(names),maximum_inverse_skin_error_cm=maximum_error,maximum_native_split_disagreement_cm=maximum_split,neutral_rest_error_cm=neutral_error,minimum_skin_singular_value=diagnostic['minimum_singular_value']));print('Calibrated full motion '+mesh['name'],flush=True)
    guide=dict(schema='CM_LOD_CORRECTIVE_GUIDES_V1',targets=targets,input_sha256=inputs,native_motion_accepted=False)
    (destination/'guides.json').write_text(json.dumps(guide,separators=(',',':')));report=dict(status='prepared',meshes=rows,refinements=refinements,source_mesh=source_mesh,targets=len(targets),method='FACTORIZED_COMPLETE_EVALUATED_MOTION',native_assets_modified=False,native_motion_accepted=False);(destination/'report.json').write_text(json.dumps(report,indent=2));return report
