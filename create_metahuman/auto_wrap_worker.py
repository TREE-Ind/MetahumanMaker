"""Detect aligned head/body landmarks from a geometry-only private scene."""
import json
import sys
import traceback
from pathlib import Path

import bpy


def main():
    folder=Path(sys.argv[sys.argv.index('--')+1]).resolve().parent
    request=json.loads((folder/'request.json').read_text(encoding='utf-8'))
    sys.path.insert(0,request['addon_root'])
    import create_metahuman
    create_metahuman.register()
    from create_metahuman import automation
    from create_metahuman.worker_lifetime import watch_parent
    watch_parent(request['parent'])
    with bpy.data.libraries.load(str(folder/'input.blend'),link=False) as (incoming,outgoing):
        outgoing.objects=list(request['objects'].values())
    for obj in outgoing.objects:bpy.context.scene.collection.objects.link(obj)
    s,f=bpy.context.scene.cm_settings,bpy.context.scene.cm_features
    s.project_dir=str(folder)
    s.source=bpy.data.objects[request['objects']['source']]
    s.template=bpy.data.objects[request['objects']['template']]
    if request['objects'].get('body'):f.body_template=bpy.data.objects[request['objects']['body']]
    s.part=request['part']
    for key,value in request['features'].items():setattr(f,key,value)
    automation.align_detect(bpy.context,s,f)
    pairs=[]
    for pair in s.landmarks:
        pairs.append({'name':pair.name,'source':list(pair.source.matrix_world.translation),
            'template':list(pair.template.matrix_world.translation),
            'part':'BODY' if pair.template.parent==f.body_template else 'HEAD'})
    result={'status':'complete','pairs':pairs,'source_matrix':list(map(list,s.source.matrix_world)),
        'face_warning':s.source.get('cm_face_landmarks_warning',''),
        'body_warning':s.source.get('cm_body_landmarks_warning','')}
    (folder/'result.json').write_text(json.dumps(result),encoding='utf-8')


if __name__=='__main__':
    try:main()
    except Exception as exc:
        traceback.print_exc()
        folder=Path(sys.argv[sys.argv.index('--')+1]).resolve().parent
        (folder/'result.json').write_text(json.dumps({'status':'failed','error':str(exc)}),encoding='utf-8')
        raise
