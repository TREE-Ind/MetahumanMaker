"""Publish an explicit normal/cavity repair without rebuilding geometry or color."""
import json,hashlib,shutil,uuid
from pathlib import Path
import bpy


def apply(context,*,normal,cavity,mask,expected_geometry,expected_uv):
    from . import mesh_ops,skin_projection,skin_normal_history,map_provenance,bake_jobs,skin_reconstruction
    from .vertex_contract import fingerprint
    s=context.scene.cm_settings;f=context.scene.cm_features
    if context.mode!='OBJECT' or f.expression_active or skin_reconstruction.running():
        raise ValueError('Finish the current edit before publishing regional shading')
    obj=s.result
    if fingerprint(mesh_ops.evaluated_coordinates(context,obj))!=expected_geometry or skin_projection.uv_layout_signature(obj)!=expected_uv:
        raise ValueError('The shading repair belongs to different geometry or UVs')
    root=Path(bpy.path.abspath(s.project_dir)).resolve();prefix=s.character_name
    folder=root/'ReferenceMaps/ShadingRepairs'/uuid.uuid4().hex[:12];folder.mkdir(parents=True)
    def digest(path):
        with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()
    # Keep the complete inputs and pre-edit maps for rollback and normal replay.
    sources={kind:(root/'Maps'/(prefix+'_'+kind+'.png')) for kind in ('NORMAL','CAVITY')}
    for kind,path in sources.items():shutil.copy2(path,folder/('before-'+kind.lower()+'.png'))
    paths={'NORMAL':Path(normal).resolve(),'CAVITY':Path(cavity).resolve()}
    for path in (*paths.values(),Path(mask).resolve()):
        if not path.is_relative_to(root):raise ValueError('Retain regional shading references inside the project')
    # Validate pixels in the image Python: accepting a reference must never
    # overwrite unselected texels, alpha, or the atlas dimensions.
    import subprocess
    check="import sys,numpy as n;from PIL import Image;m=n.asarray(Image.open(sys.argv[1]).convert('L'))>0;" \
          "a=n.asarray(Image.open(sys.argv[2]).convert('RGBA'));b=n.asarray(Image.open(sys.argv[3]).convert('RGBA'));" \
          "assert a.shape==b.shape and a.shape[:2]==m.shape;assert n.array_equal(a[~m],b[~m]);assert n.array_equal(a[:,:,3],b[:,:,3])"
    for kind,path in paths.items():
        subprocess.run([bpy.path.abspath(f.inference_python),'-c',check,str(mask),str(sources[kind]),str(path)],check=True)
    key='FINAL_HEAD';entry=map_provenance.load(context.scene)['entries'].get(map_provenance.identity(context.scene,key),{})
    prior=entry.get('artifacts',{}).get(prefix+'_NORMAL.png',{}).get('skin_recipe')
    recipe=json.loads(json.dumps(prior)) if prior else {'schema':1,'baseline':skin_normal_history.reference(root,folder/'before-normal.png'),'steps':[],'stamp':{k:v for k,v in skin_reconstruction._stamp(context,with_normal=True).items() if k in ('role','geometry','uv','sculpt')}}
    for kind,path in paths.items():shutil.copy2(path,folder/('after-'+kind.lower()+'.png'))
    recipe['steps'].append(dict(kind='SURFACE_PATCH',before=skin_normal_history.reference(root,folder/'before-normal.png'),after=skin_normal_history.reference(root,folder/'after-normal.png'),cavity_before=skin_normal_history.reference(root,folder/'before-cavity.png'),cavity_after=skin_normal_history.reference(root,folder/'after-cavity.png')))
    publication=folder/'publish';(publication/'staging/Maps').mkdir(parents=True)
    files=[]
    for kind,path in paths.items():
        name=prefix+'_'+kind+'.png';shutil.copy2(path,publication/'staging/Maps'/name);files.append(dict(name=name,sha256=digest(path)))
    def finalize():
        bake_jobs.refresh_maps(context.scene,root/'Maps',files,kinds=('NORMAL','CAVITY'))
        details={prefix+'_NORMAL.png':{'skin_recipe':recipe},prefix+'_CAVITY.png':{'surface_repair':str(folder)}}
        map_provenance.record(context.scene,{key:map_provenance.signature(context.scene,key)},origin='CUSTOM',outputs={item['name'] for item in files},details=details)
    bake_jobs.deliver(dict(folder=publication,destination=root/'Maps'),files,finalize=finalize)
    report=dict(status='applied',geometry_changed=False,albedo_changed=False,maps=files,mask=str(mask),expected_geometry=expected_geometry,uv=expected_uv)
    (folder/'report.json').write_text(json.dumps(report,indent=2));return folder
