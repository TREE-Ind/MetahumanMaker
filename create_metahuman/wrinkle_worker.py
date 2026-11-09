"""Owned wrinkle-retarget worker. No user project or map files are overwritten."""
from pathlib import Path
import hashlib
import json
import sys
import traceback
import bpy
import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import create_metahuman
create_metahuman.register()
from create_metahuman import wrinkle_retarget as ret

folder=Path(sys.argv[sys.argv.index('--')+1]).resolve()
request=json.loads((folder/'request.json').read_text())
try:
    bpy.ops.wm.open_mainfile(filepath=str(folder/'input.blend'))
    scene=bpy.data.scenes[request['scene']]
    bpy.context.window.scene=scene
    s,f=scene.cm_settings,scene.cm_features
    plans=ret.prepare_all(bpy.context,s,f)
    objects=[]
    slots=[]
    materials={}
    for slot,old,new in plans:
        slots.append({'slot':slot,'old':old.name,'new':new.name})
        for obj in (new,*new.children):
            obj['cm_wrinkle_job_id']=request['id']
            materials[obj.name]=[material.name if material else None for material in obj.data.materials]
            face_materials=np.empty(len(obj.data.polygons),dtype=np.int32)
            obj.data.polygons.foreach_get('material_index',face_materials)
            obj.data.materials.clear()  # Reuse original shaders on delivery, never duplicate packed skins.
            obj.data.polygons.foreach_set('material_index',face_materials)
            objects.append(obj)
    output=folder/'output.blend'
    bpy.data.libraries.write(str(output),set(objects),compress=True,path_remap='ABSOLUTE')
    report={'status':'complete','id':request['id'],'slots':slots,'objects':[obj.name for obj in objects],
            'materials':materials,'sha256':hashlib.sha256(output.read_bytes()).hexdigest()}
except Exception as exc:
    report={'status':'error','error':str(exc),'traceback':traceback.format_exc()}
(folder/'report.json').write_text(json.dumps(report,indent=2))
if report['status']!='complete':
    raise RuntimeError(report['error'])
