"""Numerical lash fitting from a receipt-backed native capture; no Blender import."""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS','2')
import json,sys,hashlib,types
from pathlib import Path


def digest(path):
    with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()


def run(folder):
    import numpy as np
    root=Path(__file__).resolve().parent;package=types.ModuleType('_cm_lash_math');package.__path__=[str(root)];sys.modules['_cm_lash_math']=package
    from _cm_lash_math.native_lid_anatomy import visibility_calibrated
    from _cm_lash_math.groom_lid_fit import fit
    folder=Path(folder).resolve();capture=json.loads((folder/'capture.json').read_text())
    if capture.get('schema')!=1 or capture.get('status')!='complete':raise ValueError('Supply a complete native lash capture')
    for name,expected in capture['inputs'].items():
        path=(folder/name).resolve()
        if path.parent!=folder or digest(path)!=expected:raise ValueError('A native lash input changed or left its capture')
    if capture.get('owner'):
        from process_ref import alive
        from worker_lifetime import watch_parent
        if not alive(capture['owner']):raise ValueError('The original native editor no longer owns this capture')
        watch_parent(capture['owner']['pid'],folder/'cancel')
    reference=json.loads((folder/'anatomy.json').read_text());system=reference.get('coordinate_system')
    directions=dict(Left=[-1,0,0],Right=[1,0,0],Up=[0,0,1],Down=[0,0,-1],Front=[0,-1,0],Back=[0,1,0])
    if not system or any(system.get(key) not in directions for key in ('x','y','z')):raise ValueError('Update the native bridge to 0.22 or newer for DNA coordinate metadata')
    frame=np.asarray([directions[system[key]] for key in ('x','y','z')],dtype=float)
    if abs(np.linalg.det(frame))!=1:raise ValueError('The anatomical coordinate axes are not independent')
    meshes={m['name']:m for m in reference['meshes']}
    def binary(name,dtype):
        if name not in capture['inputs']:raise ValueError('A native binary is outside the sealed capture')
        return np.fromfile(folder/name,dtype=dtype)
    def chart(name):
        m=meshes[name];return dict(positions=binary(m['positions_file'],'<f4').reshape(-1,3)@frame,layout_positions=binary(m['layout_positions_file'],'<i4'),layout_uv=binary(m['layout_uv_file'],'<f4').reshape(-1,2),triangles=binary(m['layout_triangles_file'],'<u4').reshape(-1,3))
    head=chart('head_lod0_mesh');eyes={role:chart(name) for role,name in [('left','eyeLeft_lod0_mesh'),('right','eyeRight_lod0_mesh')]}
    for eye in eyes.values():eye['triangles']=eye['layout_positions'][eye['triangles']]
    anatomy=visibility_calibrated(head,eyes);meta=json.loads((folder/'neutral.json').read_text())
    if any(key not in meta for key in ('dna_mesh_file','dna_vertex_file','neutral_file','uv_file')):raise ValueError('The binding reference export lacks canonical DNA identities; request its skin-matrix capture')
    mid=binary(meta['dna_mesh_file'],'<i4');vid=binary(meta['dna_vertex_file'],'<i4');positions=binary(meta['neutral_file'],'<f4').reshape(-1,3);uv=binary(meta['uv_file'],'<f4').reshape(-1,2)
    def canonical(mesh):
        use=mid==mesh['index'];ids=vid[use];count=np.bincount(ids,minlength=mesh['vertex_count'])
        if not len(ids) or ids.min()<0 or ids.max()>=mesh['vertex_count'] or len(count)!=mesh['vertex_count'] or np.any(count==0):raise ValueError('The native reference lacks complete anatomical vertex identities')
        value=np.column_stack([np.bincount(ids,weights=positions[use,j],minlength=len(count))/count for j in range(3)])
        if np.linalg.norm(positions[use]-value[ids],axis=1).max(initial=0)>.0001:raise ValueError('Native split positions disagree in the binding reference')
        return value,use
    native_head,use=canonical(meshes['head_lod0_mesh']);rims={};widths=[];eye_centers=[]
    for role,eye_name in [('left','eyeLeft_lod0_mesh'),('right','eyeRight_lod0_mesh')]:
        eye,_=canonical(meshes[eye_name]);widths.append(float(np.ptp(eye[:,0])));eye_centers.append(eye.mean(0));margin=anatomy['eyes'][role]['margin'];coordinates=[]
        for vertex in margin:
            values=np.unique(uv[use&(vid==vertex)],axis=0)
            if len(values)!=1:raise ValueError('A visible lid rim has ambiguous native UV charts; preserve the original groom')
            coordinates.append(values[0])
        rims[role]=dict(positions=native_head[margin],uv=np.asarray(coordinates))
    scale=min(widths)
    if scale<=0:raise ValueError('The native eye reference has no physical width')
    outward=np.mean(eye_centers,axis=0)-native_head.mean(0);outward[2]=0
    if np.linalg.norm(outward)<scale*.001:raise ValueError('The binding reference has no identifiable anterior eye direction')
    settings=capture['settings'];source=json.loads((folder/'source-groom.json').read_text());result,proof=fit(source,rims,maximum_root_offset=scale*settings['maximum_offset_fraction'],clearance=scale*settings['clearance_fraction'],outward=outward,rebuild_guides=settings['rebuild_guides'])
    result_file=folder/'fitted-groom.json';result_file.write_text(json.dumps(result),encoding='utf-8');report=dict(schema=1,status='ready',capture_sha256=digest(folder/'capture.json'),output_sha256=digest(result_file),proof=proof,anatomy='PRISTINE_NATIVE_VISIBILITY_BOUNDARY',reference_mesh=capture['reference_mesh'],source_assets_saved=0)
    (folder/'fit-result.json').write_text(json.dumps(report,indent=2));return report


if __name__=='__main__':
    folder=Path(sys.argv[1]).resolve()
    try:run(folder)
    except BaseException as error:
        (folder/'fit-result.json').write_text(json.dumps(dict(schema=1,status='failed',error=str(error)),indent=2));raise
