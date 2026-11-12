"""Compute unchanged surface-fit mathematics in an owned Blender worker."""
import json,os,sys,traceback
from pathlib import Path
import bpy,numpy as np
from mathutils import Vector


def main(folder):
    request=json.loads((folder/'request.json').read_text())
    sys.path.insert(0,request['addon_root'])
    import create_metahuman
    create_metahuman.register()
    from create_metahuman import mesh_ops as m,wrap_fit,wrap_proxy,joint_wrap,add_marker,scan_jobs
    from create_metahuman.worker_lifetime import watch_parent
    watch_parent(request['parent'])
    with bpy.data.libraries.load(str(folder/'input.blend'),link=False) as (_,loaded):loaded.objects=list(request['objects'].values())
    for obj in loaded.objects:bpy.context.scene.collection.objects.link(obj)
    objects={role:bpy.data.objects[name] for role,name in request['objects'].items()}
    scene=bpy.context.scene;scene.unit_settings.scale_length=request['units']
    s,f=scene.cm_settings,scene.cm_features;s.source=objects['source'];s.template=objects['template'];s.part=request['part'];s.project_dir=str(folder)
    for key,value in request['settings'].items():setattr(s,key,value)
    for key,value in request['features'].items():setattr(f,key,value)
    f.full_head=False
    if 'body' in objects:f.body_template=objects['body']
    bpy.context.view_layer.update()
    if 'proxy' in objects:
        proxy=objects['proxy'];f.proxy=proxy;proxy[wrap_proxy.OWNER]=1
        proxy[wrap_proxy.SOURCE]=wrap_proxy.stamp(bpy.context,s.source);proxy[wrap_proxy.GEOMETRY]=wrap_proxy.stamp(bpy.context,proxy)
    for row in request['pairs']:
        target=objects[row['role']];pair=s.landmarks.add();pair.name=row['name']
        pair.source=add_marker(bpy.context,s.source,s.source.matrix_world.inverted()@Vector(row['source']),'source')
        pair.template=add_marker(bpy.context,target,target.matrix_world.inverted()@Vector(row['template']),'template')
        pair.source['_cm_fit_marker_world']=row['source']
        pair.template['_cm_fit_marker_world']=row['template']
    bpy.context.view_layer.update()
    if wrap_fit.raw_stamp(s.template)!=request['template_stamp'] or scan_jobs.geometry_stamp(bpy.context,s.source)!=request['source_stamp']:
        raise ValueError('Worker geometry does not match the frozen fit contract')
    if s.part=='FULL_BODY':
        head,body,report,_=joint_wrap.solve(bpy.context,s,f)
        arrays=dict(head=head,body=body);result=dict(kind='JOINT',report=report)
    else:
        fit=m.solve_wrap(bpy.context,s)
        arrays=dict(points=fit['points']);result=dict(kind='SURFACE',surface_rms=fit['surface_rms'],properties=fit['properties'])
    temporary=folder/'fit.tmp.npz';np.savez(temporary,**arrays);temporary.replace(folder/'fit.npz')
    result.update(status='complete',schema=1,sha256=wrap_fit.digest(folder/'fit.npz'),contract={key:request[key] for key in ('part','source_stamp','template_stamp','body_stamp')})
    (folder/'result.json').write_text(json.dumps(result))


if __name__=='__main__':
    folder=Path(sys.argv[sys.argv.index('--')+1]).resolve()
    try:main(folder)
    except Exception:
        error=traceback.format_exc();traceback.print_exc()
        (folder/'result.json').write_text(json.dumps(dict(status='failed',error=error)))
        raise
