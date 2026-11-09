"""Prepare a private native LOD corrective revision from verified guide fields.

No native asset is changed here. The bridge must validate driver activation,
stage a mesh copy, and evaluate the resulting poses before publication.
"""
import hashlib,json,shutil
from pathlib import Path
import numpy as np


def prepare(guide_file,reference_file,captures,coordinate_transform,destination,*,source_dna,maximum_offset_cm=15.,pose_drivers=None):
    guide_file=Path(guide_file).resolve();reference_file=Path(reference_file).resolve();destination=Path(destination).resolve()
    guide=json.loads(guide_file.read_text());reference=json.loads(reference_file.read_text());matrix=np.asarray(coordinate_transform,dtype=float)
    if guide.get('schema')!='CM_LOD_CORRECTIVE_GUIDES_V1' or reference.get('schema')!='CM_DNA_CALIBRATION_REFERENCE_V1' or reference.get('coordinates')!='PRESERVED_DNA':raise ValueError('Supply native DNA-space corrective guides and their calibration reference')
    if matrix.shape!=(4,3) or not np.isfinite(matrix).all() or not np.isfinite(maximum_offset_cm) or maximum_offset_cm<=0:raise ValueError('Supply a finite native coordinate transform and corrective bound')
    singular=np.linalg.svd(matrix[:3],compute_uv=False)
    if singular.min()<1e-8:raise ValueError('The native coordinate transform is singular')
    if destination.exists():raise ValueError('Choose a fresh private native revision request folder')
    meshes={m['name']:m for m in reference['meshes']};loaded={};source_mesh=None;patches=[];targets=[];inputs={};empty=[];seen=set()
    def digest(file):
        with file.open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()
    source_dna=Path(source_dna).resolve()
    if not source_dna.is_file():raise ValueError('Export the current native source DNA before staging lower LODs')
    inputs[str(source_dna)]=digest(source_dna)
    def local_file(parent,name):
        if Path(name).name!=name:raise ValueError('Native corrective input files must stay beside their manifest')
        file=(parent/name).resolve()
        if file.parent!=parent:raise ValueError('Native corrective inputs cannot leave their manifest folder')
        return file
    for key,path in captures.items():
        lod=int(key);path=Path(path).resolve();meta=json.loads(path.read_text())
        if lod!=meta.get('lod') or not 1<=lod<reference['lod_count']:raise ValueError('A native LOD capture has the wrong identity')
        if source_mesh is None:source_mesh=meta['source_mesh']
        if meta['source_mesh']!=source_mesh:raise ValueError('Native captures belong to different character revisions')
        mids_file=local_file(path.parent,meta['dna_mesh_file']);vids_file=local_file(path.parent,meta['dna_vertex_file']);mids=np.fromfile(mids_file,dtype='<i4');vids=np.fromfile(vids_file,dtype='<i4')
        if len(mids)!=meta['vertex_count'] or len(vids)!=len(mids):raise ValueError('Native import/DNA identities are incomplete')
        loaded[lod]=(mids,vids);inputs[str(path)]=digest(path);inputs[str(mids_file)]=digest(mids_file);inputs[str(vids_file)]=digest(vids_file)
    plans=[]
    for row in guide['targets']:
        name=row['mesh'];channel=row['channel'];lod=row['lod'];mesh=meshes.get(name);key=(name,channel)
        if mesh is None or lod not in mesh.get('lods',[]) or lod not in loaded or not channel.startswith('CM_') or key in seen:raise ValueError('A guide has a missing, duplicate or unowned native LOD target')
        seen.add(key);file=local_file(guide_file.parent,row['file']);values=np.fromfile(file,dtype='<f4')
        ids=np.asarray(row['indices'])
        if ids.ndim!=1 or (len(ids) and ids.dtype.kind not in 'iu') or len(ids)*3!=len(values) or not np.isfinite(values).all():raise ValueError('A native guide field is invalid')
        if not len(ids):empty.append(dict(mesh=name,lod=lod,channel=channel));continue
        if ids.min()<0 or ids.max()>=mesh['vertex_count'] or len(np.unique(ids))!=len(ids):raise ValueError('A guide leaves its canonical DNA vertex domain')
        raw=values.reshape(-1,3);cm=raw@matrix[:3]
        if np.linalg.norm(cm,axis=1).max(initial=0)>maximum_offset_cm:raise ValueError('A native LOD corrective exceeds its physical offset bound')
        mids,vids=loaded[lod];selected=np.flatnonzero(mids==mesh['index'])
        if not len(selected) or vids[selected].min()<0 or vids[selected].max()>=mesh['vertex_count'] or np.any(np.bincount(vids[selected],minlength=mesh['vertex_count'])==0):raise ValueError('A LOD capture omits canonical vertices of the requested mesh')
        canonical=np.zeros((mesh['vertex_count'],3));canonical[ids]=cm;compiled=canonical[vids[selected]];active=np.linalg.norm(compiled,axis=1)>1e-7
        if not active.any():empty.append(dict(mesh=name,lod=lod,channel=channel));continue
        plans.append((row,file,selected[active],compiled[active]));inputs[str(file)]=digest(file)
    if not plans:raise ValueError('There are no nonzero lower-LOD targets to stage')
    destination.mkdir(parents=True)
    for i,(row,file,indices,values) in enumerate(plans):
        raw_name=f'target_{i:04d}_dna.f32';compiled_name=f'target_{i:04d}_rest_cm.f32';shutil.copyfile(file,destination/raw_name);values.astype('<f4').tofile(destination/compiled_name)
        targets.append(dict(mesh=row['mesh'],channel=row['channel'],lod=row['lod'],indices=row['indices'],file=raw_name));patches.append(dict(morph=row['mesh']+'__'+row['channel'],lod=row['lod'],indices=indices.tolist(),file=compiled_name))
    raw_bound=float(maximum_offset_cm/singular.min())
    (destination/'dna-request.json').write_text(json.dumps(dict(schema='CM_DNA_CALIBRATION_V1',maximum_offset=raw_bound,neutral=[],correctives=[])))
    (destination/'driven-request.json').write_text(json.dumps(dict(schema='CM_DNA_DRIVEN_TARGETS_V1',maximum_offset=raw_bound,activate_owned_drivers=True,targets=targets)))
    (destination/'compiled-request.json').write_text(json.dumps(dict(schema='CM_COMPILED_CORRECTIVE_V1',maximum_offset_cm=maximum_offset_cm,allow_new_dna_targets=True,patches=patches)))
    if pose_drivers is not None:
        if pose_drivers.get('schema')!='CM_DNA_POSE_CORRECTIVES_V1' or any(t.get('file')!='driver_zero.f32' or t.get('indices')!=[0] for t in pose_drivers.get('targets',[])):raise ValueError('New LOD drivers must retain zero geometry on the source LOD')
        np.zeros((1,3),dtype='<f4').tofile(destination/'driver_zero.f32');(destination/'pose-request.json').write_text(json.dumps(pose_drivers))
    inputs[str(guide_file)]=digest(guide_file);inputs[str(reference_file)]=digest(reference_file)
    inputs.update(guide.get('input_sha256',{}))
    prepared={file.name:digest(file) for file in destination.iterdir() if file.suffix=='.f32' or file.name.endswith('-request.json')}
    report=dict(schema='CM_NATIVE_LOD_CORRECTIVE_REQUEST_V1',status='prepared',source_mesh=source_mesh,source_dna=str(source_dna),targets=len(targets),lods=sorted({t['lod'] for t in targets}),empty_targets=empty,input_sha256=inputs,prepared_sha256=prepared,coordinate_transform=matrix.tolist(),native_assets_modified=False,native_motion_accepted=False)
    (destination/'manifest.json').write_text(json.dumps(report,indent=2));return report
