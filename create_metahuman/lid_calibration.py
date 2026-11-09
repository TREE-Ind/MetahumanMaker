"""Prepare native lid proposals from verified captures and embedded rig inputs."""
import hashlib,json
from pathlib import Path
import numpy as np
from .calibration_math import canonical_offsets
from .native_lid_anatomy import visibility_calibrated


def prepare(capture_folder,input_folder,destination,*,core_fraction=.16,halo_fraction=.45):
    capture_folder=Path(capture_folder).resolve();input_folder=Path(input_folder).resolve();destination=Path(destination).resolve()
    capture=json.loads((capture_folder/'capture.json').read_text())
    exported=json.loads((input_folder/'embedded-reference.json').read_text())
    if capture.get('status')!='complete' or exported.get('status')!='complete' or exported.get('source_mesh')!=capture.get('candidate'):
        raise ValueError('Use complete captures and the embedded reference of the same saved face')
    source=Path(capture['source_asset_file']).resolve()
    def digest(file):
        with Path(file).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()
    if digest(source)!=capture['source_asset_sha256']:
        raise ValueError('The saved face changed after capture; retain it and capture the new revision')
    recipe=capture.get('capture_set')
    gazes={'GAZE_UP':'eyeLookUp','GAZE_DOWN':'eyeLookDown','GAZE_LEFT':'eyeLookLeft','GAZE_RIGHT':'eyeLookRight'}
    if recipe not in {'LIDS','LIDS_FINE','LIDS_DENSE','LIDS_INTERVALS','BLINK_FINE','BLINK_DENSE'} and recipe not in gazes:raise ValueError('Capture a complete lid or gaze grid before preparing a correction')
    reference_file=input_folder/exported['reference'];dna=input_folder/exported['dna']
    reference=json.loads(reference_file.read_text());meshes={row['index']:row for row in reference['meshes']}
    if reference.get('coordinates')!='PRESERVED_DNA' or any(i not in meshes for i in (0,3,4)):
        raise ValueError('Use complete preserved native head and eye anatomy')
    neutral=[row for row in capture['poses'] if not row['controls']]
    if len(neutral)!=1:raise ValueError('Keep exactly one neutral calibration sample')
    meta_file=Path(neutral[0]['file']).resolve();meta=json.loads(meta_file.read_text())
    if meta_file.parent!=capture_folder or meta.get('dna_binding')!=exported.get('dna_binding'):
        raise ValueError('The captured and exported embedded rig identities differ')
    proof_file=input_folder/'source-proof.json'
    proof=json.loads(proof_file.read_text()) if proof_file.is_file() else {'source_files':{str(source):capture['source_asset_sha256']}}
    if any(digest(file)!=sha for file,sha in proof['source_files'].items()):raise ValueError('A saved native calibration source changed')
    read_proof={}
    def read(folder,name,dtype,shape=None):
        if Path(name).name!=name:raise ValueError('Calibration arrays must stay beside their metadata')
        file=(folder/name).resolve();read_proof[str(file)]=digest(file);data=np.fromfile(file,dtype=dtype)
        return data.reshape(shape) if shape else data
    mids=read(capture_folder,meta['dna_mesh_file'],'<i4');vids=read(capture_folder,meta['dna_vertex_file'],'<i4')
    pose=read(capture_folder,meta['vertices_file'],'<f4',(-1,3))
    canonical={i:canonical_offsets(pose,mids,vids,i,meshes[i]['vertex_count'])[0] for i in (0,3,4)}
    reference_head=read(input_folder,meshes[0]['positions_file'],'<f4',(-1,3))
    system=np.column_stack((reference_head,np.ones(len(reference_head))))
    matrix,_,rank,_=np.linalg.lstsq(system,canonical[0],rcond=None)
    error=float(np.linalg.norm(system@matrix-canonical[0],axis=1).max())
    if rank!=4 or not np.isfinite(matrix).all() or np.linalg.svd(matrix[:3],compute_uv=False).min()<1e-8 or error>.002:
        raise ValueError('Embedded DNA and captured neutral do not share an accurate affine frame (maximum error '+str(error)+' cm); retain the rig for inspection')
    # DNA and render meshes can choose different diagonals in the same quad.
    # Aperture cuts must follow the tessellation being displayed and corrected.
    if not meta.get('uv_file'):raise ValueError('Capture the neutral with native UVs before deriving visible lid anatomy')
    render_triangles=read(capture_folder,meta['triangles_file'],'<u4',(-1,3))
    render_uv=read(capture_folder,meta['uv_file'],'<f4',(-1,2))
    if render_uv.shape!=(len(mids),2) or not np.isfinite(render_uv).all():raise ValueError('The captured UVs do not match the native render vertices')
    if not len(render_triangles) or render_triangles.max()>=len(mids):raise ValueError('The captured render triangles do not match the native identities')
    def anatomy(index):
        selected=np.flatnonzero(mids==index);lookup=np.full(len(mids),-1,dtype=np.int64);lookup[selected]=np.arange(len(selected))
        triangles=render_triangles[np.all(mids[render_triangles]==index,axis=1)]
        if not len(triangles):raise ValueError('The captured neutral is missing a required anatomical render surface')
        return dict(positions=canonical[index]*[1.,-1.,1.],layout_positions=vids[selected],
                    layout_uv=render_uv[selected],triangles=lookup[triangles])
    head=anatomy(0);eyes={}
    for side,index in (('left',3),('right',4)):
        eye=anatomy(index);eyes[side]=dict(positions=eye['positions'],triangles=eye['layout_positions'][eye['triangles']])
    # UV charts identify a socket seed, not the visible eyelid edge. Continuing
    # that seed directly into the motion core includes hidden socket returns.
    # Use the saved neutral head/globe silhouette to choose a complete native
    # topology loop. A missing/closed silhouette must fail rather than silently
    # changing the anatomical correspondence.
    structure=visibility_calibrated(head,eyes);structure_file=input_folder/'lid-structure.npz'
    if structure_file.exists():raise ValueError('This preparation already has anatomy; observe its original result')
    arrays={'outer_triangles':structure['outer_triangles']}
    for side,row in structure['eyes'].items():arrays.update({side+'_margin':row['margin'],side+'_interior':row['interior']})
    np.savez_compressed(structure_file,**arrays)
    options=dict(core_fraction=core_fraction,halo_fraction=halo_fraction)
    if recipe in {'LIDS','LIDS_FINE','LIDS_DENSE','LIDS_INTERVALS','BLINK_FINE','BLINK_DENSE'}:
        from .native_lid_interaction import prepare as solver
    else:
        from .native_gaze_interaction import prepare as solver
        options['gaze']=gazes[recipe]
    result=solver(capture_folder,reference_file,structure_file,dna,source,matrix,destination,**options)
    if any(digest(file)!=sha for file,sha in proof['source_files'].items()):raise ValueError('A saved native calibration source changed during preparation')
    manifest_file=destination/'manifest.json';manifest=json.loads(manifest_file.read_text())
    if any(digest(file)!=sha for file,sha in read_proof.items()):raise ValueError('A native coordinate/anatomy input changed during preparation')
    manifest['input_sha256'].update(read_proof)
    manifest['native_source_sha256']=proof['source_files'];manifest['input_sha256'].update(proof['source_files'])
    anatomy_proof={side:dict(margin_vertices=len(row['margin']),interior_vertices=len(row['interior']),ring=row['ring'],visibility=row['proof']) for side,row in structure['eyes'].items()}
    manifest['anatomy_method']=structure['method'];manifest['anatomy_eyes']=anatomy_proof
    manifest['anatomy_topology']='CAPTURED_RENDER_TRIANGLES_AND_UVS'
    if proof_file.is_file():manifest['input_sha256'][str(proof_file)]=digest(proof_file)
    manifest_file.write_text(json.dumps(manifest,indent=2))
    report=dict(status='prepared',folder=str(destination),capture_set=recipe,targets=result['targets'],source_mesh=capture['candidate'],
                source_asset_sha256=capture['source_asset_sha256'],coordinate_transform=matrix.tolist(),maximum_mapping_error_cm=error,
                anatomy_method=structure['method'],anatomy_eyes=anatomy_proof,anatomy_topology=manifest['anatomy_topology'],native_motion_accepted=False)
    (destination/'preparation-summary.json').write_text(json.dumps(report,indent=2))
    return report
