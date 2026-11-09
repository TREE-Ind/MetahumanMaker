"""Read actual compiled poses, compare calibration anchors and export review geometry."""
import json
from pathlib import Path
import numpy as np


def inspect(capture_folder,reference_file,desired_folder,destination,*,pose_map,source_captures=None,tolerance_mm=.005):
    capture_folder=Path(capture_folder);destination=Path(destination);desired_folder=Path(desired_folder);reference=json.loads(Path(reference_file).read_text());manifest=json.loads((capture_folder/'capture.json').read_text())
    if manifest.get('status')!='complete' or destination.exists() or tolerance_mm<=0:raise ValueError('Supply complete native captures, a fresh review folder and positive comparison tolerance')
    meshes={m['name']:m for m in reference['meshes']};rows=[];destination.mkdir(parents=True)
    def load(folder,name):
        meta=json.loads((folder/(name+'.json')).read_text());read=lambda key,dtype:np.fromfile(folder/meta[key],dtype=dtype)
        return meta,read('vertices_file','<f4').reshape(-1,3),read('dna_mesh_file','<i4'),read('dna_vertex_file','<i4')
    for lod in sorted({row['lod'] for row in manifest['poses']}):
        meta,neutral,mids,vids=load(capture_folder,f'LOD{lod}_Neutral');triangles=np.fromfile(capture_folder/meta['triangles_file'],dtype='<u4').reshape(-1,3);normal=np.fromfile(capture_folder/meta['neutral_normals_file'],dtype='<f4').reshape(-1,3)*[1,-1,1];dataset={};selectors={}
        for role,prefix in [('head','head'),('left','eyeLeft'),('right','eyeRight')]:
            mesh=meshes[f'{prefix}_lod{lod}_mesh'];use=np.flatnonzero(mids==mesh['index']);selectors[role]=use;lookup=np.full(len(mids),-1);lookup[use]=np.arange(len(use));tri=lookup[triangles[np.all(mids[triangles]==mesh['index'],axis=1)]];v=neutral[use]*[.01,-.01,.01];cross=np.cross(v[tri[:,1]]-v[tri[:,0]],v[tri[:,2]]-v[tri[:,0]]);dot=np.einsum('ij,ij->i',cross,normal[use][tri].sum(1))
            if np.median(dot[np.abs(dot)>1e-16])<0:tri=tri[:,[0,2,1]]
            dataset['triangles' if role=='head' else role+'_triangles']=tri
            if role!='head':dataset[role+'_uv']=np.fromfile(capture_folder/meta['uv_file'],dtype='<f4').reshape(-1,2)[use][tri].reshape(-1,2)
        for row in [r for r in manifest['poses'] if r['lod']==lod]:
            name=row['name'];meta,v,mid,vid=load(capture_folder,f'LOD{lod}_{name}')
            if meta['lod']!=lod or meta['source_mesh']!=manifest['candidate'] or not np.array_equal(mid,mids) or not np.array_equal(vid,vids) or not np.isfinite(v).all():raise ValueError('Native pose identity changed during review')
            for role,use in selectors.items():dataset[name+'_'+role]=v[use]*[.01,-.01,.01]
            target_name=pose_map.get(name);errors=[]
            if target_name is not None:
                if lod==0 and source_captures is not None:
                    _,old,oldmid,oldvid=load(Path(source_captures),'LOD0_'+target_name)
                    if not np.array_equal(mid,oldmid) or not np.array_equal(vid,oldvid):raise ValueError('High-LOD preservation comparison changed native identities')
                    errors.append(dict(mesh='ENTIRE_LOD0',maximum_error_mm=float(np.linalg.norm(v-old,axis=1).max(initial=0)*10)))
                elif lod>0:
                    for mesh in reference['meshes']:
                        file=desired_folder/(str(mesh['index'])+'-desired.npz')
                        if lod not in mesh.get('lods',[]) or not file.exists():continue
                        goal=np.load(file);index=goal['names'].tolist().index(target_name);use=mids==mesh['index'];expected=goal['positions'][index][vids[use]];errors.append(dict(mesh=mesh['name'],maximum_error_mm=float(np.linalg.norm(v[use]-expected,axis=1).max(initial=0)*10)))
            rows.append(dict(lod=lod,pose=name,anchor=target_name,comparisons=errors,within_tolerance=all(e['maximum_error_mm']<=tolerance_mm for e in errors) if errors else None))
        np.savez_compressed(destination/f'LOD{lod}.npz',**dataset)
    report=dict(status='complete',anchors_within_tolerance=all(row['within_tolerance'] is not False for row in rows),tolerance_mm=tolerance_mm,poses=rows,native_visual_acceptance=False)
    (destination/'review-provenance.json').write_text(json.dumps(dict(status='complete',native_capture_geometry=True,synthetic_geometry_changes=False,source_mesh=manifest['candidate'])))
    (destination/'verification.json').write_text(json.dumps(report,indent=2));return report
