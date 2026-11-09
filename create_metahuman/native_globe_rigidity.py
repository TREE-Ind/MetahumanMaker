"""Prepare retirement of generated globe shape drivers, retaining native pupils.

Rigidly skinned globes should not acquire shape changes from lid/gaze RBF
interpolation. This removes generated CM_ geometry from recognized native eye
meshes; it does not alter skinning, joint behavior, native pupil controls or lids.
The compiled morphs must be retired on a separate mesh copy by the native stage.
"""
import hashlib,json,re
from pathlib import Path
import numpy as np


def digest(path):
    with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()


def prepare(reference_file,source_dna,source_mesh,source_asset_file,destination,*,maximum_offset=15.):
    reference_file=Path(reference_file).resolve();source_dna=Path(source_dna).resolve();source_asset_file=Path(source_asset_file).resolve();destination=Path(destination).resolve();reference=json.loads(reference_file.read_text())
    if reference.get('schema')!='CM_DNA_CALIBRATION_REFERENCE_V1' or reference.get('coordinates')!='PRESERVED_DNA':raise ValueError('Export the current complete native calibration reference first')
    if destination.exists() or not re.fullmatch(r'/Game(?:/[A-Za-z0-9_]+)+\.[A-Za-z0-9_]+',source_mesh):raise ValueError('Choose a fresh request and a saved project face mesh')
    if not np.isfinite(maximum_offset) or maximum_offset<=0:raise ValueError('Choose a positive corrective offset bound')
    inputs={str(path):digest(path) for path in (reference_file,source_dna,source_asset_file)};plans=[];retire=[];preserved=[];pairs=set()
    def binary(name,dtype):
        if Path(name).name!=name:raise ValueError('Corrective arrays must stay beside their reference')
        path=reference_file.parent/name;inputs[str(path)]=digest(path);return np.fromfile(path,dtype=dtype)
    for mesh in reference['meshes']:
        match=re.fullmatch(r'eye(Left|Right)_lod([0-7])_mesh',mesh['name'])
        if not match:continue
        pairs.add((match[1],int(match[2])))
        for target in mesh['targets']:
            if not target['channel'].startswith('CM_'):
                preserved.append(dict(mesh=mesh['name'],channel=target['channel']));continue
            name=mesh['name']+'__'+target['channel'];retire.append(name)
            if 'indices_file' not in target or 'deltas_file' not in target:raise ValueError('The current bridge must export generated corrective arrays')
            ids=binary(target['indices_file'],'<u4');values=binary(target['deltas_file'],'<f4').reshape(-1,3)
            if len(ids)!=len(values) or len(ids)!=target['vertices'] or not np.isfinite(values).all() or (len(ids) and (ids.max()>=mesh['vertex_count'] or len(np.unique(ids))!=len(ids))):raise ValueError('A generated globe corrective has invalid canonical data')
            if np.linalg.norm(values,axis=1).max(initial=0)>maximum_offset:raise ValueError('A generated globe corrective exceeds the permitted offset bound')
            if len(ids) and np.any(values):plans.append((mesh['name'],target['channel'],ids,values))
    expected={(side,lod) for side in ('Left','Right') for lod in range(reference['lod_count'])}
    if pairs!=expected or not plans or len(retire)!=len(set(retire)):raise ValueError('Supply complete paired native eye LODs with generated globe correctives')
    destination.mkdir(parents=True);rows=[]
    for index,(mesh,channel,ids,values) in enumerate(plans):
        name=f'retire_{index:04d}.f32';(-values).astype('<f4').tofile(destination/name);rows.append(dict(mesh=mesh,channel=channel,indices=ids.tolist(),file=name))
    request=dict(schema='CM_DNA_CALIBRATION_V1',maximum_offset=float(maximum_offset),neutral=[],correctives=rows)
    (destination/'dna-request.json').write_text(json.dumps(request));prepared={file.name:digest(file) for file in destination.iterdir() if file.is_file()}
    report=dict(schema='CM_RIGID_GLOBE_REVISION_V1',status='prepared',source_mesh=source_mesh,source_dna=str(source_dna),source_asset_file=str(source_asset_file),reference=str(reference_file),retire_morphs=retire,preserved_native_eye_targets=preserved,lods=sorted({lod for _,lod in pairs}),input_sha256=inputs,prepared_sha256=prepared,neutral_geometry_changed=False,joints_changed=False,skin_weights_changed=False,lids_changed=False,native_motion_accepted=False)
    (destination/'manifest.json').write_text(json.dumps(report,indent=2));return report
