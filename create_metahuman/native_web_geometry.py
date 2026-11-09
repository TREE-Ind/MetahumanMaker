"""Reusable native FBX-to-glTF conversion in a dedicated Blender worker.

This exports rigged geometry and a groom binding surface. It is one stage of
native web delivery, not a claim of UE shader / RigLogic or animation parity.
"""
import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import bpy
import numpy as np
from mathutils import Matrix,Quaternion

from . import resource_budget

GENERATOR='CreateMetaHuman.NativeWebGeometry'
DIRECT_CORRECTIVES={'eye_blink_L','eye_blink_R','brow_raise_L','brow_raise_R',
    'brow_down_L','brow_down_R','eye_widen_L','eye_widen_R',
    'mouth_cornerPull_left','mouth_cornerPull_right','mouth_cornersDown_L',
    'mouth_cornersDown_R','nose_wrinkle_L','nose_wrinkle_R'}


def input_file(root,relative,sha256=None):
    path=(root/relative).resolve()
    if not path.is_relative_to(root) or not path.is_file():
        raise ValueError('Native source file is missing or outside its capture folder: '+str(relative))
    if sha256:
        with path.open('rb') as stream:actual=hashlib.file_digest(stream,'sha256').hexdigest()
        if actual!=sha256:raise ValueError('Native source file changed: '+str(relative))
    return path


def component_matrix(value):
    """Actor-space Unreal centimetres -> Blender metres, including rotation/scale."""
    translation=np.asarray(value['translation_cm'],dtype=float)
    rotation=np.asarray(value['rotation_xyzw'],dtype=float)
    scale=np.asarray(value['scale'],dtype=float)
    if translation.shape!=(3,) or rotation.shape!=(4,) or scale.shape!=(3,) or not all(np.isfinite(a).all() for a in (translation,rotation,scale)):
        raise ValueError('Invalid captured component transform')
    if abs(np.linalg.norm(rotation)-1)>1e-3 or np.any(np.abs(scale)<1e-8):
        raise ValueError('Captured component has an invalid rotation or singular scale')
    q=Quaternion((rotation[3],rotation[0],rotation[1],rotation[2]));flip=np.diag([1,-1,1])
    linear=flip@np.asarray(q.to_matrix())@np.diag(scale)@flip
    result=Matrix(linear.tolist()).to_4x4();result.translation=translation*[.01,-.01,.01]
    return result


def convert(source_root,destination,*,face_component='Face',facial_mode='DIRECT',legacy_component_frame=False,expression_bank=None):
    """Create a new owned geometry folder; never change the capture or user scene.

    Launch in a dedicated background Blender process. ALL_RAW retains every
    exported corrective; DIRECT preserves the bounded inspection subset, which
    does not execute RigLogic. Legacy transform interpretation requires an
    explicit caller opt-in and is reported in the result.
    """
    if not bpy.app.background:raise ValueError('Native web conversion must run in a separate background Blender process')
    if type(legacy_component_frame) is not bool:raise ValueError('Legacy component-frame opt-in must be a boolean')
    if facial_mode not in {'DIRECT','ALL_RAW','NATIVE_SAMPLED'}:raise ValueError('Choose DIRECT, ALL_RAW or NATIVE_SAMPLED facial controls')
    if (facial_mode=='NATIVE_SAMPLED')!=(expression_bank is not None):raise ValueError('Native sampled geometry requires an explicit native expression bank')
    source_root=Path(source_root).resolve();destination=Path(destination).absolute()
    if destination.exists():raise ValueError('Choose a new native geometry destination; existing work is retained')
    if destination.resolve()!=destination or destination.is_relative_to(source_root):
        raise ValueError('Geometry staging must be separate from the native source without linked folders')
    manifest_path=input_file(source_root,'source-manifest.json');source_bytes=manifest_path.read_bytes()
    source=json.loads(source_bytes.decode('utf-8-sig'))
    if source.get('generator')!='CreateMetaHuman.NativeWebSource' or source.get('status')!='source_extracted':
        raise ValueError('Choose a completed native web source capture')
    parts=source.get('skeletal_meshes',[])
    if not parts or len({p['component'] for p in parts})!=len(parts):raise ValueError('Native source has missing or duplicate skeletal components')
    for part in parts:
        input_file(source_root,part['fbx']['file'],part['fbx']['sha256'])
        if 'actor_relative_transform' not in part and not legacy_component_frame:
            raise ValueError('Recapture component transforms with the current add-on before converting '+part['component'])
    estimate=max(1024**3,sum(p['fbx']['bytes'] for p in parts)*24)
    resource_budget.require('native rigged web geometry conversion',estimate)
    destination.parent.mkdir(parents=True,exist_ok=True)
    resource_budget.require_disk('native rigged web geometry conversion',destination.parent,max(256*1024**2,estimate//3))
    bpy.ops.wm.read_factory_settings(use_empty=True)
    report={'schema':1,'generator':GENERATOR,'source_sha256':hashlib.sha256(source_bytes).hexdigest(),
        'facial_mode':facial_mode,'legacy_component_frame':legacy_component_frame,'face_component':face_component,
        'geometry':'RIGGED_ASSET_REFERENCE_POSE','coordinates':'meters, Y up; actor space',
        'source_actor_transform':source.get('actor_transform'),'mesh':'character.glb','parts':[],
        'limitations':['Native RigLogic is not executed','Actor animation and live poses are not converted','Materials are placeholders for subsequent native material preparation']}
    with TemporaryDirectory(prefix='.native-geometry-',dir=destination.parent) as temporary:
        stage=Path(temporary);objects=[]
        for part in parts:
            before=set(bpy.data.objects)
            bpy.ops.import_scene.fbx(filepath=str(input_file(source_root,part['fbx']['file'])),use_anim=False,use_image_search=False,automatic_bone_orientation=False)
            imported=set(bpy.data.objects)-before;objects.extend(imported)
            transform=component_matrix(part.get('actor_relative_transform',part['transform']))
            for obj in imported:
                if obj.parent not in imported:obj.matrix_world=transform@obj.matrix_world
            bpy.context.view_layer.update()
            meshes=[obj for obj in imported if obj.type=='MESH']
            if len(meshes)!=1:raise ValueError('Expected one native LOD0 mesh per component: '+part['component'])
            obj=meshes[0];lod=part['lod_sections'][0]
            if len(obj.data.vertices)!=lod['vertices']:raise ValueError('Native LOD0 vertex count changed: '+part['component'])
            slots=list(range(len(part['materials'])))
            if len(obj.data.materials)!=len(slots):slots=sorted(set(lod['materials']))
            if len(obj.data.materials)!=len(slots):raise ValueError('Native material-slot correspondence is ambiguous: '+part['component'])
            alias='Face' if part['component']==face_component else part['component']
            for index,slot in enumerate(slots):
                name='CMNative_'+alias+'_'+str(slot);material=bpy.data.materials.get(name) or bpy.data.materials.new(name);material.use_nodes=True;obj.data.materials[index]=material
            keys=list(obj.data.shape_keys.key_blocks) if obj.data.shape_keys else []
            original_shapes=[key.name for key in keys[1:]]
            if part['component']==face_component and facial_mode=='DIRECT':
                for key in keys[1:]:
                    if key.name.split('__')[-1] not in DIRECT_CORRECTIVES:obj.shape_key_remove(key)
            if part['component']==face_component and facial_mode=='NATIVE_SAMPLED':
                from . import native_pose_bank
                report['native_expression_bank']=native_pose_bank.install(obj,part,expression_bank,transform,stage,source.get('face_source_owner'))
            obj.name='CM_Web_Face' if part['component']==face_component else 'CM_Web_'+part['component']
            armatures=[m.object for m in obj.modifiers if m.type=='ARMATURE' and m.object]
            if len(armatures)!=1:raise ValueError('Native component has no unique skeletal rig: '+part['component'])
            arm=armatures[0];bones=[b.name for b in arm.data.bones];lookup={name:i for i,name in enumerate(bones)}
            weights=np.zeros((len(obj.data.vertices),12),np.float32);joints=np.zeros((len(obj.data.vertices),12),np.int32)
            for vertex in obj.data.vertices:
                influences=sorted(((g.weight,lookup[obj.vertex_groups[g.group].name]) for g in vertex.groups if obj.vertex_groups[g.group].name in lookup and g.weight>0),reverse=True)
                if not influences or len(influences)>12:raise ValueError('Native vertex must have one to twelve influences')
                for index,(weight,joint) in enumerate(influences):weights[vertex.index,index]=weight;joints[vertex.index,index]=joint
            surface_file=None
            if part['component']==face_component or any(g.get('asset') and g.get('target_component')==part['component'] for g in source.get('grooms',[])):
                obj.data.calc_loop_triangles();triangles=[t for t in obj.data.loop_triangles if slots[t.material_index]==0]
                if not triangles or not obj.data.uv_layers.active:raise ValueError('Native face has no UV skin surface for groom binding')
                surface_file='face-bind-surface.npz' if part['component']==face_component else 'surface-'+hashlib.sha256(part['component'].encode()).hexdigest()[:16]+'.npz'
                np.savez_compressed(stage/surface_file,positions=np.array([obj.matrix_world@v.co for v in obj.data.vertices],np.float32),
                    triangles=np.array([t.vertices for t in triangles],np.int32),uvs=np.array([[obj.data.uv_layers.active.data[i].uv[:] for i in t.loops] for t in triangles],np.float32),weights=weights,joints=joints,bones=np.array(bones))
            report['parts'].append({'component':part['component'],'node':obj.name,'vertices':len(obj.data.vertices),'bones':len(bones),
                'native_material_slots':slots,'source_correctives':original_shapes,'exported_correctives':[k.name for k in obj.data.shape_keys.key_blocks[1:]] if obj.data.shape_keys else [],
                'actor_relative_transform':part.get('actor_relative_transform'),'binding_surface':surface_file,'max_influences':int((weights>0).sum(1).max())})
        for obj in bpy.context.selected_objects:obj.select_set(False)
        morph_bytes=sum(len(obj.data.vertices)*max(0,len(obj.data.shape_keys.key_blocks)-1)*24 for obj in objects if obj.type=='MESH' and obj.data.shape_keys)
        resource_budget.require('native web morph serialization',max(512*1024**2,morph_bytes*3))
        resource_budget.require_disk('native web morph serialization',stage,max(128*1024**2,morph_bytes*2))
        for obj in objects:obj.select_set(True)
        bpy.context.view_layer.objects.active=next(o for o in objects if o.type=='ARMATURE')
        result=bpy.ops.export_scene.gltf(filepath=str(stage/'character.glb'),export_format='GLB',use_selection=True,
            export_skins=True,export_morph=True,export_animations=False,export_materials='EXPORT',export_extras=False,
            export_vertex_color='ACTIVE',export_all_vertex_colors=True,export_all_influences=True,will_save_settings=False)
        if result!={'FINISHED'}:raise RuntimeError('Native rigged geometry export did not finish')
        report['binding_sources']={}
        for key,record in source.get('binding_surfaces',{}).items():
            before=set(bpy.data.objects)
            bpy.ops.import_scene.fbx(filepath=str(input_file(source_root,record['fbx']['file'],record['fbx']['sha256'])),use_anim=False,use_image_search=False,automatic_bone_orientation=False)
            imported=[o for o in set(bpy.data.objects)-before if o.type=='MESH']
            if len(imported)!=1:raise ValueError('Groom source FBX must have exactly one LOD0 surface')
            obj=imported[0];obj.data.calc_loop_triangles();slots=list(range(record['material_count']))
            if len(obj.data.materials)!=len(slots):slots=sorted(set(record['lod_material_slots']))
            if len(obj.data.materials)!=len(slots):raise ValueError('Groom source material correspondence is ambiguous')
            triangles=[t for t in obj.data.loop_triangles if slots[t.material_index]==record['material_slot']]
            if not triangles or not obj.data.uv_layers.active:raise ValueError('Groom source has no matching UV surface')
            filename='binding-source-'+key+'.npz'
            np.savez_compressed(stage/filename,positions=np.array([obj.matrix_world@v.co for v in obj.data.vertices],np.float32),
                triangles=np.array([t.vertices for t in triangles],np.int32),uvs=np.array([[obj.data.uv_layers.active.data[i].uv[:] for i in t.loops] for t in triangles],np.float32))
            report['binding_sources'][key]={'file':filename,'asset':record['asset'],'sha256':hashlib.sha256((stage/filename).read_bytes()).hexdigest()}
        if manifest_path.read_bytes()!=source_bytes:raise ValueError('Native capture changed during conversion; previous delivery retained')
        for part in parts:input_file(source_root,part['fbx']['file'],part['fbx']['sha256'])
        for record in source.get('binding_surfaces',{}).values():input_file(source_root,record['fbx']['file'],record['fbx']['sha256'])
        report['files']={p.name:{'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()} for p in stage.iterdir() if p.is_file()}
        (stage/'geometry-manifest.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
        stage.replace(destination)
    return report
