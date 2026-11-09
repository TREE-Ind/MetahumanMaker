"""Evaluate artist skin response using the native face bank's material drivers."""
import hashlib,json,runpy,math,re
from pathlib import Path

def digest(path):
    with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()

def neutral_parameters(root,source):
    reference=source.get('expression_bank')
    if not reference:return {}
    root=Path(root).resolve();path=(root/reference['file']).resolve()
    if not path.is_relative_to(root) or digest(path)!=reference['sha256']:raise ValueError('Native neutral skin face bank changed')
    bank=json.loads(path.read_bytes());item=bank['poses'][0]
    if item['name']!='Neutral':raise ValueError('Native skin bank has no neutral sample')
    sample=Path(item['file']).resolve()
    if not sample.is_relative_to(path.parent) or digest(sample)!=bank['files'][sample.name]['sha256']:raise ValueError('Native neutral skin sample changed')
    return json.loads(sample.read_bytes()).get('material_scalars',{})

def bake(actor,source_root,*,size=512,progress=None):
    import unreal
    root=Path(source_root).resolve();source_path=root/'source-manifest.json';source_bytes=source_path.read_bytes();source=json.loads(source_bytes)
    reference=source.get('expression_bank')
    if not reference:return None
    bank_path=(root/reference['file']).resolve()
    if not bank_path.is_relative_to(root) or digest(bank_path)!=reference['sha256']:raise ValueError('Native skin face bank changed')
    bank=json.loads(bank_path.read_bytes())
    if bank.get('status')!='complete':raise ValueError('Complete native facial sampling before skin response capture')
    if size not in (256,512,1024):raise ValueError('Choose supported animated skin sampling dimensions')
    face=next((c for c in actor.get_components_by_class(unreal.SkeletalMeshComponent) if c.get_name()=='Face'),None)
    if not face:raise ValueError('Native skin motion requires the captured face component')
    record=runpy.run_path(str(Path(__file__).with_name('unreal_web.py')))['material_record']
    live=face.get_material(0);before=record(live);names=set(before['scalars'])
    poses=[]
    for item in bank['poses']:
        if not re.fullmatch(r'[A-Za-z0-9_]{1,160}',item['name']):raise ValueError('Invalid native skin pose name')
        path=Path(item['file']).resolve()
        if not path.is_relative_to(bank_path.parent) or digest(path)!=bank['files'][path.name]['sha256']:raise ValueError('Native skin sample changed')
        data=json.loads(path.read_bytes());values=data.get('material_scalars')
        if values is None:raise ValueError('Recapture the native face bank using bridge 0.17 or newer for evaluated skin drivers')
        if not isinstance(values,dict) or any(not key.startswith('head_wm') or type(value) not in (int,float) or not math.isfinite(value) for key,value in values.items()):raise ValueError('Invalid native material driver values')
        if poses and set(values)!=set(poses[0][1]):raise ValueError('Native skin driver coverage changes between samples')
        if set(values)-names:raise ValueError('The artist skin graph does not expose the captured native expression drivers')
        poses.append((item['name'],values))
    if not poses or poses[0][0]!='Neutral' or not poses[0][1]:return {'status':'unsupported','reason':'No native animated skin drivers in this face material'}
    budget=runpy.run_path(str(Path(__file__).with_name('resource_budget.py')))
    budget['require_disk']('native animated skin capture',root,len(poses)*size*size*24+64*1024**2)
    budget['require']('native animated skin capture',512*1024**2+size*size*48)
    folder=root/'skin-motion';folder.mkdir(exist_ok=False)
    result={'schema':1,'generator':'CreateMetaHuman.NativeSkinMotion','status':'baking','size':size,'component':'Face','slot':0,
            'source_manifest_sha256':hashlib.sha256(source_bytes).hexdigest(),'bank_sha256':digest(bank_path),'poses':[],
            'limitations':['Sampled artist material response; combined expressions approximate nonlinear shader interactions','Animated response resolution is independent of the full-resolution neutral skin maps']}
    receipt=folder/'manifest.json'
    def write():receipt.write_text(json.dumps(result,indent=2),encoding='utf-8')
    write();actors=unreal.get_editor_subsystem(unreal.EditorActorSubsystem);selection=list(actors.get_selected_level_actors());private_actor=None;seen={};bake_inputs=None
    try:
        unreal.SystemLibrary.collect_garbage()
        private_actor=actors.spawn_actor_from_class(unreal.SkeletalMeshActor,unreal.Vector(),transient=True)
        private=private_actor.get_component_by_class(unreal.SkeletalMeshComponent);private.set_skeletal_mesh_asset(face.get_skeletal_mesh_asset())
        private.set_world_transform(face.get_world_transform(),False,False)
        primitive=face.get_editor_property('custom_primitive_data')
        if primitive.get_editor_property('data'):private.set_editor_property('custom_primitive_data',primitive)
        parent=live
        while isinstance(parent,unreal.MaterialInstanceDynamic):parent=parent.get_editor_property('parent')
        material=unreal.MaterialLibrary.create_dynamic_material_instance(actor.get_world(),parent)
        for key,value in before['scalars'].items():material.set_scalar_parameter_value(key,value)
        for key,value in before['vectors'].items():material.set_vector_parameter_value(key,unreal.LinearColor(*value))
        for key,value in before['textures'].items():
            if value:material.set_texture_parameter_value(key,unreal.load_asset(value['asset']) or unreal.find_object(None,value['asset']))
        bake_inputs=runpy.run_path(str(Path(__file__).with_name('unreal_web_bake_inputs.py')))['BakeInputs'](root)
        material,bake_metadata=bake_inputs.material(material,before,actor.get_world())
        normal_material,normal_metadata=bake_inputs.material(material,before,actor.get_world(),separate_micro=True)
        result['bake_inputs']=bake_metadata
        private.set_material(0,material)
        for index,(name,values) in enumerate(poses):
            if progress:progress('skin_motion',index,len(poses))
            signature=json.dumps(values,sort_keys=True,separators=(',',':'));maps=seen.get(signature)
            if maps is None:
                for key,value in values.items():
                    material.set_scalar_parameter_value(key,value)
                    normal_material.set_scalar_parameter_value(key,value)
                maps={}
                for role in ('basecolor','normal','roughness'):
                    private.set_material(0,normal_material if role=='normal' else material)
                    target=folder/(name+'_'+role+'.png')
                    if not unreal.CreateMHArtistLibrary.bake_material_property(private,0,role,size,str(target)):raise RuntimeError('Native skin response bake failed: '+name+' / '+role)
                    metadata=json.loads(Path(str(target)+'.json').read_bytes())
                    if not metadata.get('material_parameters_snapshotted') or not metadata.get('texture_residency_waited'):raise ValueError('Update the native bridge before sampling skin appearance')
                    maps[role]={'file':target.relative_to(root).as_posix(),'sha256':digest(target),'size':[size,size],'linear_color':metadata['linear_color']}
                seen[signature]=maps
            result['poses'].append({'pose':name,'material_scalars':values,'maps':maps});write()
        if record(live)!=before or source_path.read_bytes()!=source_bytes or digest(bank_path)!=reference['sha256']:raise ValueError('Artist skin or native source changed during response capture')
        result.update(status='complete',source_material_preserved=True,unique_parameter_states=len(seen));write()
        if progress:progress('skin_motion',len(poses),len(poses))
        return result
    except Exception as exc:result.update(status='failed',error=str(exc));write();raise
    finally:
        if private_actor:actors.destroy_actor(private_actor)
        if bake_inputs:bake_inputs.close()
        actors.set_selected_level_actors([item for item in selection if item and not item.is_actor_being_destroyed()])
