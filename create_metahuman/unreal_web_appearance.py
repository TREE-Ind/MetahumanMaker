"""Read-only native property baking over captured LOD0 surfaces, with eye bake modes."""
import hashlib,json,re,runpy,uuid
from pathlib import Path

GENERATOR='CreateMetaHuman.NativeWebAppearance'
PROPERTIES={'basecolor':'basecolor','normal':'normal','roughness':'roughness','metallic':'metallic','specular':'specular','cavity':'ambient_occlusion','emissive':'emissive_color','opacity':'opacity_mask'}


def digest(path):
    with path.open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()


def api(name):return runpy.run_path(str(Path(__file__).with_name(name+'.py')))


def material_hash(record):return hashlib.sha256(json.dumps(record,sort_keys=True,allow_nan=False).encode()).hexdigest()


def plan(source,size=8192,detail_size=2048):
    if size not in (256,512,1024,2048,4096,8192) or detail_size not in (256,512,1024,2048):raise ValueError('Choose supported native bake dimensions')
    if source.get('generator')!='CreateMetaHuman.NativeWebSource' or source.get('status')!='source_extracted':raise ValueError('Complete native source extraction before appearance baking')
    family=api('native_web_materials')['family'];rows=[];unbaked=[]
    for part in source['skeletal_meshes']:
        for slot in sorted(set(part['lod_sections'][0]['materials'])):
            material=part['materials'][slot]
            if not material:raise ValueError('Captured rendered surface has no material')
            kind=family(material)
            if kind in ('HIDDEN','LASH','EYE_SHELL','TEARLINE'):
                unbaked.append({'component':part['component'],'slot':slot,'family':kind});continue
            roles=['iris','sclera','iris_normal','sclera_normal'] if kind=='EYE' else ['basecolor','normal','roughness','metallic','specular','cavity']+(['emissive'] if kind is None else [])
            rows.append({'component':part['component'],'slot':slot,'family':kind or 'PBR','size':size if kind in ('SKIN',None) else detail_size,'roles':roles,'material_sha256':material_hash(material)})
    return {'materials':rows,'unbaked_families':unbaked}


def opaque(material):
    return surface(material)[0]=='OPAQUE'


def surface(material):
    import unreal
    current=material;blend=None;clip=None
    while isinstance(current,unreal.MaterialInstance):
        if not isinstance(current,unreal.MaterialInstanceDynamic):
            overrides=current.get_editor_property('base_property_overrides')
            if blend is None and overrides.get_editor_property('override_blend_mode'):blend=overrides.get_editor_property('blend_mode')
            if clip is None and overrides.get_editor_property('override_opacity_mask_clip_value'):clip=overrides.get_editor_property('opacity_mask_clip_value')
        current=current.get_editor_property('parent')
    if not isinstance(current,unreal.Material):raise ValueError('Material has no native surface definition')
    if blend is None:blend=current.get_editor_property('blend_mode')
    if clip is None:clip=current.get_editor_property('opacity_mask_clip_value')
    mode='OPAQUE' if 'OPAQUE' in str(blend) else 'MASKED' if 'MASKED' in str(blend) else 'UNSUPPORTED'
    return mode,float(clip)


def bake(actor,source_root,*,size=8192,detail_size=2048,progress=None):
    import unreal
    root=Path(source_root).resolve();path=root/'source-manifest.json';raw=path.read_bytes();source=json.loads(raw)
    planned=plan(source,size,detail_size);destination=root/'appearance-manifest.json'
    if destination.exists() or (root/'appearance').exists():raise ValueError('Appearance outputs already exist; retain them and capture into a new folder')
    if not hasattr(unreal,'CreateMHArtistLibrary'):raise ValueError('Install the MetaHuman Maker native bridge before appearance baking')
    if not hasattr(unreal.CreateMHArtistLibrary,'supports_stable_material_baking') or not unreal.CreateMHArtistLibrary.supports_stable_material_baking():raise ValueError('Install bridge 0.16 or newer for current-parameter appearance baking with resident textures')
    components={c.get_name():c for c in actor.get_components_by_class(unreal.SkeletalMeshComponent)}
    source_parts={p['component']:p for p in source['skeletal_meshes']};record=api('unreal_web')['material_record']
    neutral=api('unreal_web_skin_motion')['neutral_parameters'](root,source) if source.get('expression_bank') else {}
    def verify():
        if path.read_bytes()!=raw:raise ValueError('Source capture changed while baking appearance')
        for row in planned['materials']:
            component=components.get(row['component'])
            if not component or component.get_skeletal_mesh_asset().get_path_name()!=source_parts[row['component']]['asset'] or material_hash(record(component.get_material(row['slot'])))!=row['material_sha256']:raise ValueError('Live geometry or material settings changed after native capture')
    verify()
    for row in planned['materials']:
        mode,clip=surface(components[row['component']].get_material(row['slot']))
        if mode=='MASKED' and row['family']=='SKIN':row.update(surface=mode,alpha_test=clip);row['roles'].append('opacity')
        elif mode!='OPAQUE':raise ValueError('Automatic property baking needs an opaque surface or masked native skin: '+row['component']+' slot '+str(row['slot'])+'. Transparent and masked custom graphs need a dedicated adapter')
    estimate=sum(row['size']**2*4*len(row['roles']) for row in planned['materials'])
    api('resource_budget')['require_disk']('native appearance baking',root,estimate+256*1024**2)
    api('resource_budget')['require']('native appearance baking',512*1024**2+max((row['size']**2*32 for row in planned['materials']),default=0))
    result={'schema':1,'generator':GENERATOR,'status':'baking','source_manifest_sha256':hashlib.sha256(raw).hexdigest(),'pose':'ASSET_REFERENCE_POSE','materials':[],
        'native_assets_modified':False,'unbaked_families':planned['unbaked_families'],
        'limitations':['Static material properties only; native view/time effects, WPO, transparency and subsurface models are not reproduced','Native eye bake modes preserve iris/sclera parameters; browser eye shading remains approximate']}
    def checkpoint():
        pending=destination.with_suffix('.pending');pending.write_text(json.dumps(result,indent=2),encoding='utf-8');pending.replace(destination)
    checkpoint();private_actor=None;private_assets=[];bake_inputs=None
    subsystem=unreal.get_editor_subsystem(unreal.EditorActorSubsystem);world=actor.get_world()
    selection=list(subsystem.get_selected_level_actors())
    try:
        bake_inputs=api('unreal_web_bake_inputs')['BakeInputs'](root)
        private_actor=subsystem.spawn_actor_from_class(unreal.SkeletalMeshActor,unreal.Vector(),transient=True)
        private=private_actor.get_component_by_class(unreal.SkeletalMeshComponent)
        for index,row in enumerate(planned['materials']):
            verify()
            if progress:progress('appearance',index,len(planned['materials']))
            original=components[row['component']];slot=row['slot'];live=original.get_material(slot);captured=source_parts[row['component']]['materials'][slot]
            private.set_skeletal_mesh_asset(original.get_skeletal_mesh_asset());private.set_world_transform(original.get_world_transform(),False,False)
            primitive=original.get_editor_property('custom_primitive_data')
            if primitive.get_editor_property('data'):private.set_editor_property('custom_primitive_data',primitive)
            private.set_material(slot,live)
            if neutral and row['component']=='Face' and slot==0 and row['family']=='SKIN':
                parent=live
                while isinstance(parent,unreal.MaterialInstanceDynamic):parent=parent.get_editor_property('parent')
                material=unreal.MaterialLibrary.create_dynamic_material_instance(world,parent);private_assets.append(material)
                for key,value in captured['scalars'].items():material.set_scalar_parameter_value(key,value)
                for key,value in captured['vectors'].items():material.set_vector_parameter_value(key,unreal.LinearColor(*value))
                for key,value in captured['textures'].items():
                    if value:material.set_texture_parameter_value(key,unreal.load_asset(value['asset']) or unreal.find_object(None,value['asset']))
                for key,value in neutral.items():material.set_scalar_parameter_value(key,value)
                private.set_material(slot,material)
            bake_metadata={};normal_material=None;normal_metadata={}
            if row['family']=='SKIN':
                resolved,bake_metadata=bake_inputs.material(private.get_material(slot),captured,world)
                normal_material,normal_metadata=bake_inputs.material(resolved,captured,world,separate_micro=True)
                private.set_material(slot,resolved)
            output={'component':row['component'],'slot':slot,'maps':{},'source_material_sha256':row['material_sha256']}
            if row.get('surface')=='MASKED':output.update(surface='MASKED',alpha_test=row['alpha_test'])
            if row['family']=='PBR':output.update(kind='PBR',surface='OPAQUE')
            result['materials'].append(output)
            for role in row['roles']:
                if row['family']=='SKIN':private.set_material(slot,normal_material if role=='normal' else resolved)
                if row['family']=='EYE':
                    mode='iris' if role.startswith('iris') else 'sclera';normal=role.endswith('_normal')
                    parent=live
                    while isinstance(parent,unreal.MaterialInstanceDynamic):parent=parent.get_editor_property('parent')
                    name='Eye_'+str(index)+'_'+role
                    mic=unreal.new_object(unreal.MaterialInstanceConstant,name='CM_Web_'+name+'_'+uuid.uuid4().hex)
                    if not mic:raise RuntimeError('Could not create private eye bake settings')
                    private_assets.append(mic);lib=unreal.MaterialEditingLibrary;lib.set_material_instance_parent(mic,parent)
                    if not {'Bake Iris ON','Bake Sclera ON'}<=captured['switches'].keys():raise ValueError('This eye shader has no supported iris/sclera bake modes')
                    for key,value in captured['switches'].items():lib.set_material_instance_static_switch_parameter_value(mic,key,value)
                    lib.set_material_instance_static_switch_parameter_value(mic,'Bake Iris ON',mode=='iris');lib.set_material_instance_static_switch_parameter_value(mic,'Bake Sclera ON',mode=='sclera');lib.update_material_instance(mic)
                    material=unreal.MaterialLibrary.create_dynamic_material_instance(world,mic)
                    for key,value in captured['scalars'].items():material.set_scalar_parameter_value(key,value)
                    for key,value in captured['vectors'].items():material.set_vector_parameter_value(key,unreal.LinearColor(*value))
                    for key,value in captured['textures'].items():
                        if value:
                            texture=live.get_texture_parameter_value(key) if isinstance(live,unreal.MaterialInstanceDynamic) else lib.get_material_default_texture_parameter_value(live,key) if isinstance(live,unreal.Material) else lib.get_material_instance_texture_parameter_value(live,key)
                            material.set_texture_parameter_value(key,texture)
                    material.set_scalar_parameter_value('Bake Basecolor Iris',int(mode=='iris' and not normal));material.set_scalar_parameter_value('Bake Normal Iris',int(mode=='iris' and normal));material.set_scalar_parameter_value('Bake Normal Sclera',int(mode=='sclera' and normal));private.set_material(slot,material)
                    prop='normal' if normal else 'basecolor'
                else:prop=PROPERTIES[role]
                filename='appearance/'+hashlib.sha256(row['component'].encode()).hexdigest()[:12]+'_'+str(slot)+'_'+role+'.png';target=root/filename
                if not unreal.CreateMHArtistLibrary.bake_material_property(private,slot,prop,row['size'],str(target)):raise RuntimeError('Native appearance bake failed: '+row['component']+' / '+role)
                metadata=json.loads(Path(str(target)+'.json').read_bytes())
                output['maps'][role]={'file':filename,'sha256':digest(target),'bytes':target.stat().st_size,'size':[row['size'],row['size']],**metadata,**(normal_metadata if role=='normal' else bake_metadata)};checkpoint()
        verify();result['status']='complete';checkpoint();return result
    except BaseException as error:result.update(status='failed',error=str(error));checkpoint();raise
    finally:
        if private_actor:subsystem.destroy_actor(private_actor)
        if bake_inputs:bake_inputs.close()
        subsystem.set_selected_level_actors([item for item in selection if item and not item.is_actor_being_destroyed()])
        # Eye material instances are transient and leave no content assets to delete.
        private_assets.clear()
