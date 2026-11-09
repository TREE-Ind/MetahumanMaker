"""Selected-part source texture baking without replacing accepted sculpt maps."""
import hashlib
import json
import shutil
import uuid
from pathlib import Path
import bpy
from . import map_provenance,mesh_ops,skin_projection,textures


def plan(s,f):
    roles=('HEAD','BODY') if s.part=='FULL_BODY' and f.skin_bake_scope=='BOTH' else (
        f.texture_part if s.part=='FULL_BODY' else s.part,)
    folder=Path(bpy.path.abspath(s.project_dir))/'Maps'
    result=[]
    for role in roles:
        obj=f.body_result if role=='BODY' and s.part=='FULL_BODY' else s.result
        mesh_ops.check_mesh(obj,role.title()+' result')
        if role=='BODY' and s.part=='FULL_BODY' and obj.get('cm_body_source')=='LIBRARY':
            raise ValueError('The library body has no matched source; choose Head or fit a full-body source first')
        prefix=s.character_name+('_Body' if role=='BODY' and s.part=='FULL_BODY' else '')
        if Path(prefix).name!=prefix:raise ValueError('Enter a character name without directory separators')
        result.append({'role':role,'prefix':prefix,'albedo':f.skin_bake_channels!='ROUGHNESS',
            'roughness':f.skin_bake_channels in {'ROUGHNESS','ALL'},
            'normal':bool(f.skin_bake_channels!='ROUGHNESS' and (f.skin_replace_normal or not (folder/(prefix+'_NORMAL.png')).is_file()))})
    return result


def preflight(context,s,f):
    from . import character_project,skin_reconstruction,paint_session
    character_project.preflight(context,s,f)
    if skin_reconstruction.running() or getattr(paint_session,'_SESSION',None):
        raise ValueError('Finish the current skin repair or paint session before source baking')
    mesh_ops.check_mesh(s.source,'Texture source')
    for item in plan(s,f):
        obj=f.body_result if item['role']=='BODY' and s.part=='FULL_BODY' else s.result
        if not obj.data.uv_layers:raise ValueError('The selected skin part needs template UVs')


def signature(scene,items=None):
    s,f=scene.cm_settings,scene.cm_features
    cache={};items=plan(s,f) if items is None else items;folder=Path(bpy.path.abspath(s.project_dir))/'Maps'
    with bpy.context.temp_override(scene=scene,view_layer=scene.view_layers[0]):
        value={'plan':items,'size':f.skin_map_size,'distance':s.bake_distance,
               'source':map_provenance.mesh_digest(bpy.context,s.source,cache),
               'materials':map_provenance.shader_inputs(s.source,cache,surface=True,fast_images=True),'tone':f.generated_skin_tone,'parts':[]}
        for item in items:
            obj=f.body_result if item['role']=='BODY' and s.part=='FULL_BODY' else s.result
            files={}
            for kind in ('ALBEDO','NORMAL','ROUGHNESS'):
                path=folder/(item['prefix']+'_'+kind+'.png')
                if path.is_file():
                    with path.open('rb') as stream:files[kind]=hashlib.file_digest(stream,'sha256').hexdigest()
            value['parts'].append({'mesh':map_provenance.mesh_digest(bpy.context,obj,cache),
                'uv':skin_projection.uv_layout_signature(obj),'files':files})
    return hashlib.sha256(json.dumps(value,sort_keys=True).encode()).hexdigest()


def finish(scene,destination,files,items):
    """Roll back shader/metadata changes if atomic map delivery cannot finalize."""
    import numpy as np
    s,f=scene.cm_settings,scene.cm_features
    objects=[f.body_result if item['role']=='BODY' and s.part=='FULL_BODY' else s.result for item in items]
    properties={obj:{key:obj.get(key) for key in ('cm_skin_source_albedo','cm_albedo_source',
                'cm_reference_normal_image','cm_reference_normal','cm_normal_source','cm_source_skin_receipt')} for obj in objects}
    materials={obj:list(obj.data.materials) for obj in objects}
    faces={}
    for obj in objects:
        faces[obj]=np.empty(len(obj.data.polygons),dtype=np.int32)
        obj.data.polygons.foreach_get('material_index',faces[obj])
    nodes=[(node,node.image) for obj in objects for mat in obj.data.materials if mat and mat.use_nodes
           for node in mat.node_tree.nodes if node.type=='TEX_IMAGE']
    trees={mat.node_tree:(set(mat.node_tree.nodes),[(link.from_socket,link.to_socket) for link in mat.node_tree.links])
           for obj in objects for mat in obj.data.materials if mat and mat.use_nodes}
    source=f.skin_solver_source;manifest=map_provenance.manifest(scene)
    old_manifest=manifest.read_bytes() if manifest.is_file() else None
    invalidated=scene.get(map_provenance.INVALIDATED)
    created=[]
    try:
        _finish(scene,destination,files,items,created)
        from . import skin_projection_history
        skin_projection_history.record(scene,items)
    except BaseException:
        for obj,slots in materials.items():
            obj.data.materials.clear()
            for mat in slots:obj.data.materials.append(mat)
            obj.data.polygons.foreach_set('material_index',faces[obj])
        for node,image in nodes:node.image=image
        for tree,(previous,links) in trees.items():
            for node in list(tree.nodes):
                if node not in previous:tree.nodes.remove(node)
            tree.links.clear()
            for source_socket,target_socket in links:tree.links.new(source_socket,target_socket)
        for obj,values in properties.items():
            for key,value in values.items():
                if value is None:
                    if key in obj:del obj[key]
                else:obj[key]=value
        f.skin_solver_source=source
        if old_manifest is None:manifest.unlink(missing_ok=True)
        else:manifest.write_bytes(old_manifest)
        if invalidated is None:
            if map_provenance.INVALIDATED in scene:del scene[map_provenance.INVALIDATED]
        else:scene[map_provenance.INVALIDATED]=invalidated
        for image,path in created:
            if image and not image.users:bpy.data.images.remove(image)
            path.unlink(missing_ok=True)
        raise


def _finish(scene,destination,files,items,created):
    from . import bake_jobs
    s,f=scene.cm_settings,scene.cm_features
    changed={item['name'] for item in files}
    # Refresh only produced maps in existing authored materials.
    bake_jobs.refresh_maps(scene,destination,files,kinds=('ALBEDO','NORMAL','ROUGHNESS'))
    with bpy.context.temp_override(scene=scene,view_layer=scene.view_layers[0]):
      for item in items:
        role=item['role'];obj=f.body_result if role=='BODY' and s.part=='FULL_BODY' else s.result
        if item['prefix']+'_ALBEDO.png' in changed:
            retained=obj.get('cm_skin_source_albedo')
            if retained and f.skin_solver_source==retained:f.skin_solver_source=None
            if 'cm_skin_source_albedo' in obj:del obj['cm_skin_source_albedo']
        normal=item['prefix']+'_NORMAL.png'
        with textures.skin_part(s,f,role):
            _bind_transferred_maps(obj,destination,item['prefix'],changed)
            native=textures.generated_reference_albedo(s,f) if role=='HEAD' and s.template else None
            if item['prefix']+'_ALBEDO.png' in changed:obj['cm_albedo_source']='LOCAL_NATIVE_UV' if native else 'SOURCE_PROJECTION'
            if normal in changed:
                parent=destination.parent/'ReferenceMaps';parent.mkdir(exist_ok=True)
                if parent.resolve()!=parent:raise ValueError('Reference maps must remain inside the character project')
                path=parent/(item['prefix']+'_'+uuid.uuid4().hex[:12]+'_NORMAL.png')
                created.append((None,path))
                shutil.copy2(destination/normal,path)
                image=bpy.data.images.load(str(path),check_existing=False)
                created[-1]=(image,path)
                image.colorspace_settings.name='Non-Color';image.pack();image['cm_owned_skin_baseline']=True
                obj['cm_reference_normal_image']=image;obj['cm_reference_normal']=str(path)
                obj['cm_normal_source']='LOCAL_NATIVE_UV' if role=='HEAD' and s.template and textures.generated_reference_normal(bpy.context,s,f) else 'SOURCE_PROJECTION'
            # A first projection may have no image node to refresh yet.
            if item['prefix']+'_ALBEDO.png' in changed and not any(n.type=='TEX_IMAGE' and n.image and n.get('cm_map_kind')=='ALBEDO'
                       for mat in obj.data.materials if mat and mat.use_nodes for n in mat.node_tree.nodes):
                mesh_ops.preview_material(bpy.context,s)
            elif normal in changed:
                _bind_missing_normal(obj,destination/normal)
            roughness=item['prefix']+'_ROUGHNESS.png'
            if roughness in changed:
                from . import lookdev
                image=bpy.data.images.load(str(destination/roughness),check_existing=False)
                image.colorspace_settings.name='Non-Color';image.alpha_mode='NONE';image.pack()
                for material in obj.data.materials:
                    if not material or not material.use_nodes:continue
                    scales=[node.inputs[1].default_value for node in material.node_tree.nodes if node.get('cm_roughness_scale')]
                    lookdev.bind_roughness(material,image)
                    if scales:
                        for node in material.node_tree.nodes:
                            if node.get('cm_roughness_scale'):node.inputs[1].default_value=scales[0]
        if normal in changed:
            key='FINAL_'+role
            map_provenance.record(scene,{key:map_provenance.signature(scene,key)},origin='BAKE',
                outputs={normal},recipe='SOURCE_GNM_NATIVE' if obj.get('cm_normal_source')=='LOCAL_NATIVE_UV' else 'SOURCE_SKIN_NORMAL')


def _bind_transferred_maps(obj,destination,prefix,changed):
    """Retarget managed template map nodes on a private shader copy.

    Fresh wraps inherit template image names. Matching only the new character's
    filenames leaves those nodes showing the template skin after a successful
    bake. Preserve authored shader structure and all untagged image nodes.
    """
    images={}
    for index,material in enumerate(list(obj.data.materials)):
        if not material or not material.use_nodes:continue
        replacements={node.name:node.get('cm_map_kind') for node in material.node_tree.nodes
            if node.type=='TEX_IMAGE' and node.get('cm_map_kind') in {'ALBEDO','NORMAL','ROUGHNESS'}
            and prefix+'_'+node['cm_map_kind']+'.png' in changed
            and (not node.image or Path(bpy.path.abspath(node.image.filepath)).name!=prefix+'_'+node['cm_map_kind']+'.png')}
        if not replacements:continue
        private=material.copy();obj.data.materials[index]=private
        for name,kind in replacements.items():
            if kind not in images:
                image=bpy.data.images.load(str(destination/(prefix+'_'+kind+'.png')),check_existing=False)
                image.colorspace_settings.name='sRGB' if kind=='ALBEDO' else 'Non-Color'
                image.alpha_mode='NONE';image.pack();images[kind]=image
            private.node_tree.nodes[name].image=images[kind]


def _bind_missing_normal(obj,path):
    """Add a first delivered normal without rebuilding an authored color shader."""
    image=None
    for mat in obj.data.materials:
        if not mat or not mat.use_nodes:continue
        nodes,links=mat.node_tree.nodes,mat.node_tree.links
        if any(n.type=='TEX_IMAGE' and n.image and Path(bpy.path.abspath(n.image.filepath)).name==path.name for n in nodes):continue
        shaders=[node for node in nodes if node.type=='BSDF_PRINCIPLED']
        if not shaders:continue
        if image is None:
            image=bpy.data.images.load(str(path),check_existing=False)
            image.colorspace_settings.name='Non-Color';image.alpha_mode='NONE';image.pack()
        texture=nodes.new('ShaderNodeTexImage');texture.image=image;texture['cm_map_kind']='NORMAL'
        separate=nodes.new('ShaderNodeSeparateColor');combine=nodes.new('ShaderNodeCombineColor')
        invert=nodes.new('ShaderNodeMath');invert.operation='SUBTRACT';invert.inputs[0].default_value=1
        tangent=nodes.new('ShaderNodeNormalMap')
        links.new(texture.outputs['Color'],separate.inputs['Color'])
        links.new(separate.outputs['Red'],combine.inputs['Red'])
        links.new(separate.outputs['Green'],invert.inputs[1])
        links.new(invert.outputs[0],combine.inputs['Green'])
        links.new(separate.outputs['Blue'],combine.inputs['Blue'])
        links.new(combine.outputs['Color'],tangent.inputs['Color'])
        for shader in shaders:links.new(tangent.outputs['Normal'],shader.inputs['Normal'])
