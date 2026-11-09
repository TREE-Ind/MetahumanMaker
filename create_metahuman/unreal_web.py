"""Read a processed Unreal actor into an explicit native-to-web source bundle.

This is an extraction stage, not a claim that UE material graphs, RigLogic or
groom simulation execute in a browser. Subsequent conversion must resolve the
recorded resources and verify those runtime features independently.
"""
import hashlib
import json
from pathlib import Path
import re
import runpy


GENERATOR='CreateMetaHuman.NativeWebSource'


def transform(value):
    t=value.translation;r=value.rotation;s=value.scale3d
    return {'translation_cm':[t.x,t.y,t.z],'rotation_xyzw':[r.x,r.y,r.z,r.w],'scale':[s.x,s.y,s.z]}


def material_record(material):
    import unreal
    lib=unreal.MaterialEditingLibrary
    if not material:return None
    dynamic=isinstance(material,unreal.MaterialInstanceDynamic)
    base=isinstance(material,unreal.Material)
    def scalar(name):
        return material.get_scalar_parameter_value(name) if dynamic else (lib.get_material_default_scalar_parameter_value(material,name) if base else lib.get_material_instance_scalar_parameter_value(material,name))
    def vector(name):
        return material.get_vector_parameter_value(name) if dynamic else (lib.get_material_default_vector_parameter_value(material,name) if base else lib.get_material_instance_vector_parameter_value(material,name))
    def texture(name):
        return material.get_texture_parameter_value(name) if dynamic else (lib.get_material_default_texture_parameter_value(material,name) if base else lib.get_material_instance_texture_parameter_value(material,name))
    vectors={}
    for name in lib.get_vector_parameter_names(material):
        v=vector(name);vectors[str(name)]=[v.r,v.g,v.b,v.a]
    textures={}
    for name in lib.get_texture_parameter_names(material):
        image=texture(name)
        textures[str(name)]={'asset':image.get_path_name(),'class':image.get_class().get_name()} if image else None
    chain=[];parent=material
    while isinstance(parent,unreal.MaterialInstance):
        chain.append(parent.get_path_name());parent=parent.get_editor_property('parent')
    if parent:chain.append(parent.get_path_name())
    switches=material
    while isinstance(switches,unreal.MaterialInstanceDynamic):switches=switches.get_editor_property('parent')
    mode,cutoff=runpy.run_path(str(Path(__file__).with_name('unreal_web_appearance.py')))['surface'](material)
    return {'asset':material.get_path_name(),'inheritance':chain,
            'surface':{'mode':mode,'opacity_mask_clip_value':cutoff},
            'scalars':{str(n):scalar(n) for n in lib.get_scalar_parameter_names(material)},
            'vectors':vectors,'textures':textures,
            'switches':{str(n):(lib.get_material_default_static_switch_parameter_value(switches,n) if isinstance(switches,unreal.Material)
                               else lib.get_material_instance_static_switch_parameter_value(switches,n))
                        for n in lib.get_static_switch_parameter_names(switches)},
            'web_shader_compiled':False}


def export_actor(actor,directory,*,animations=()):
    import unreal
    command=unreal.SystemLibrary.get_command_line().lower()
    if '-nullrhi' in command or ('-run=' in command and '-allowcommandletrendering' not in command):
        raise ValueError('Native skeletal FBX capture needs a rendering editor; commandlets must use -AllowCommandletRendering without -NullRHI')
    destination=Path(directory).resolve()
    def actor_transform(component):
        return transform(unreal.MathLibrary.make_relative_transform(component.get_world_transform(),actor.get_actor_transform()))
    prior={}
    if destination.exists() and any(destination.iterdir()):
        manifest=destination/'source-manifest.json'
        prior=json.loads(manifest.read_text(encoding='utf-8')) if manifest.is_file() else {}
        if prior.get('generator')!=GENERATOR or prior.get('status')!='failed' or prior.get('source_actor_class')!=actor.get_class().get_path_name():
            raise ValueError('Choose an empty owned native-web staging folder; existing files are retained')
    destination.mkdir(parents=True,exist_ok=True)
    report={'schema':1,'generator':GENERATOR,'source_actor_class':actor.get_class().get_path_name(),
            'actor_transform':transform(actor.get_actor_transform()),
            'geometry_pose':'ASSET_REFERENCE_POSE',
            'engine':unreal.SystemLibrary.get_engine_version(),'coordinates':'Unreal centimeters, Z up',
            'status':'extracting','web_ready':False,'skeletal_meshes':[],'grooms':[],'animations':[],'binding_surfaces':{},
            'missing_capabilities':['portable material graph equivalents','RigLogic/driven appearance runtime',
                                    'animated groom binding and hair shader','browser conversion and visual acceptance']}
    def checkpoint():
        (destination/'source-manifest.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    checkpoint()
    def export_fbx(asset,filename):
        for record in prior.get('skeletal_meshes',[])+prior.get('animations',[]):
            old=record['fbx']
            if old['file']!=filename:continue
            path=destination/filename
            with path.open('rb') as stream:sha=hashlib.file_digest(stream,'sha256').hexdigest()
            if record['asset']!=asset.get_path_name() or sha!=old['sha256']:
                raise ValueError('A previous source export changed; choose another staging folder')
            return old
        if (destination/filename).exists():raise ValueError('An unrecorded file occupies the extraction destination')
        task=unreal.AssetExportTask();task.object=asset;task.filename=str(destination/filename)
        task.automated=True;task.prompt=False;task.replace_identical=False
        task.exporter=unreal.SkeletalMeshExporterFBX() if isinstance(asset,unreal.SkeletalMesh) else unreal.AnimSequenceExporterFBX()
        options=unreal.FbxExportOption();options.level_of_detail=False
        options.export_morph_targets=True;options.export_preview_mesh=False
        options.bake_material_inputs=unreal.FbxMaterialBakeMode.DISABLED;task.options=options
        if not unreal.Exporter.run_asset_export_task(task):raise RuntimeError('Native FBX extraction failed: '+asset.get_path_name())
        path=destination/filename
        if not path.is_file() or path.stat().st_size<64:raise RuntimeError('Native FBX extraction is empty')
        with path.open('rb') as stream:sha=hashlib.file_digest(stream,'sha256').hexdigest()
        return {'file':filename,'bytes':path.stat().st_size,'sha256':sha}
    try:
        for component in actor.get_components_by_class(unreal.SkeletalMeshComponent):
            mesh=component.get_skeletal_mesh_asset()
            if not mesh:continue
            name=re.sub('[^A-Za-z0-9_]','_',component.get_name())
            editor=unreal.get_editor_subsystem(unreal.SkeletalMeshEditorSubsystem)
            lods=[{'lod':lod,'vertices':editor.get_num_verts(mesh,lod),
                   'materials':[editor.get_lod_material_slot(mesh,lod,i) for i in range(editor.get_num_sections(mesh,lod))]}
                  for lod in range(editor.get_lod_count(mesh))]
            record={'component':component.get_name(),'asset':mesh.get_path_name(),
                    'skeleton':mesh.get_editor_property('skeleton').get_path_name(),
                    'transform':transform(component.get_relative_transform()),
                    'actor_relative_transform':actor_transform(component),
                    'materials':[material_record(component.get_material(i)) for i in range(component.get_num_materials())],
                    'fbx':export_fbx(mesh,name+'.fbx'),'morph_export_requested':True,'lod_sections':lods}
            package=mesh.get_path_name().split('.')[0]
            if package.startswith('/Game/'):
                asset_file=Path(unreal.Paths.project_content_dir()).resolve()/(package.removeprefix('/Game/')+'.uasset')
                dirty={p.get_path_name() for p in unreal.EditorLoadingAndSavingUtils.get_dirty_content_packages()}
                if package in dirty:raise ValueError('Save the native skeletal mesh before capturing its geometry: '+package)
                if asset_file.is_file():
                    with asset_file.open('rb') as stream:record['asset_sha256']=hashlib.file_digest(stream,'sha256').hexdigest()
            report['skeletal_meshes'].append(record);checkpoint()
        if not report['skeletal_meshes']:raise ValueError('Select an assembled actor containing skeletal meshes')
        groom_export=runpy.run_path(str(Path(__file__).with_name('unreal_groom.py')))['export_groom']
        for component in actor.get_components_by_class(unreal.GroomComponent):
            groom=component.get_editor_property('groom_asset')
            record={'component':component.get_name(),'asset':groom.get_path_name() if groom else None,
                    'transform':transform(component.get_relative_transform()),'actor_relative_transform':actor_transform(component),'visible':component.is_visible(),'status':'not_assigned'}
            if groom:
                name=re.sub('[^A-Za-z0-9_]','_',component.get_name())+'.groom.json'
                groom_export(groom.get_path_name(),str(destination/name))
                record.update(file=name,status='source_strands_extracted',
                    sha256=hashlib.sha256((destination/name).read_bytes()).hexdigest(),bytes=(destination/name).stat().st_size,
                    materials=[material_record(component.get_material(i)) for i in range(component.get_num_materials())])
                binding=component.get_editor_property('binding_asset')
                record['native_binding_asset']=binding.get_path_name() if binding else None
                record['web_binding_resolved']=False
                parent=component.get_attach_parent()
                while parent and not isinstance(parent,unreal.SkeletalMeshComponent):parent=parent.get_attach_parent()
                target_mesh=binding.get_editor_property('target_skeletal_mesh') if binding else None
                candidates=[c for c in actor.get_components_by_class(unreal.SkeletalMeshComponent) if target_mesh and c.get_skeletal_mesh_asset()==target_mesh]
                target=candidates[0] if len(candidates)==1 else parent
                if not target or target.get_owner()!=actor:raise ValueError('Groom target is not uniquely captured on this actor: '+component.get_name())
                record['target_component']=target.get_name()
                original=binding.get_editor_property('source_skeletal_mesh') if binding else None
                original=original or target.get_skeletal_mesh_asset()
                section=int(binding.get_editor_property('matching_section')) if binding else 0
                editor=unreal.get_editor_subsystem(unreal.SkeletalMeshEditorSubsystem)
                if section<0 or section>=editor.get_num_sections(original,0):raise ValueError('Groom source section is outside native LOD0')
                key=hashlib.sha256((original.get_path_name()+':'+str(section)).encode()).hexdigest()[:16]
                if key not in report['binding_surfaces']:
                    prior_mesh=next((p for p in report['skeletal_meshes'] if p['asset']==original.get_path_name()),None)
                    report['binding_surfaces'][key]={'asset':original.get_path_name(),'fbx':prior_mesh['fbx'] if prior_mesh else export_fbx(original,'BindingSource_'+key+'.fbx'),
                        'material_count':len(original.get_editor_property('materials')),'lod_material_slots':[editor.get_lod_material_slot(original,0,i) for i in range(editor.get_num_sections(original,0))],
                        'material_slot':editor.get_lod_material_slot(original,0,section)}
                record['binding_source']=key
                record['component_group_settings']=[]
                for index,desc in enumerate(component.get_editor_property('groom_groups_desc')):
                    values={'index':index,'raw':desc.export_text()}
                    for field in ('hair_width','hair_root_scale','hair_tip_scale','hair_length_scale'):
                        values[field]=float(desc.get_editor_property(field));values[field+'_override']=bool(desc.get_editor_property(field+'_override'))
                    record['component_group_settings'].append(values)
            report['grooms'].append(record);checkpoint()
        for index,path in enumerate(animations):
            asset=unreal.load_asset(path)
            if not isinstance(asset,unreal.AnimSequence):raise ValueError('Choose an explicit AnimSequence asset: '+path)
            report['animations'].append({'asset':path,'fbx':export_fbx(asset,'Animation_'+str(index)+'.fbx')})
        report['status']='source_extracted';checkpoint()
        return report
    except BaseException as exc:
        report.update(status='failed',error=str(exc));checkpoint();raise
