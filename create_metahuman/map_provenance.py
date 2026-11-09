"""Relate derived maps to the actual character inputs used to bake them."""
from pathlib import Path
import hashlib
import json
import time
import bpy
import numpy as np
from .buffer_digest import sha256_buffer

INVALIDATED='cm_invalidated_bakes'


def folder(scene):
    return Path(bpy.path.abspath(scene.cm_settings.project_dir))/'Maps'


def manifest(scene):
    return folder(scene)/'.create-metahuman-bakes.json'


def identity(scene,key):
    return json.dumps([scene.cm_settings.character_name,key],ensure_ascii=False)


def prefix(scene,key):
    s=scene.cm_settings
    return s.character_name+('_Body' if key in {'FINAL_BODY','ROUGHNESS_BODY'} and s.part=='FULL_BODY' else '')


def paths(scene,key):
    kinds=('NORMAL','CAVITY') if key.startswith('FINAL_') else ('ROUGHNESS',) if key.startswith('ROUGHNESS_') else (key,)
    return [folder(scene)/(prefix(scene,key)+'_'+kind+'.png') for kind in kinds]


def load(scene):
    path=manifest(scene)
    if not path.exists():return {'schema':1,'entries':{}}
    value=json.loads(path.read_text(encoding='utf-8'))
    if value.get('schema')!=1 or not isinstance(value.get('entries'),dict):
        raise ValueError('Unsupported map bake record; retain it and rebake affected maps')
    return value


def image_digest(image,cache):
    if not image:return None
    ident=image.as_pointer()
    if ident not in cache:
        pixels=np.empty(len(image.pixels),dtype=np.float32)
        image.pixels.foreach_get(pixels)
        metadata=json.dumps([list(image.size),image.colorspace_settings.name,image.alpha_mode]).encode()
        cache[ident]=sha256_buffer(pixels,metadata)
    return cache[ident]


def mesh_digest(context,obj,cache=None):
    if not obj:return None
    key=('mesh',obj.as_pointer())
    if cache is not None and key in cache:return cache[key]
    graph=context.evaluated_depsgraph_get();evaluated=obj.evaluated_get(graph)
    data=evaluated.to_mesh(preserve_all_data_layers=True,depsgraph=graph)
    try:
        digest=hashlib.sha256()
        matrix=np.asarray(evaluated.matrix_world,dtype='<f4');digest.update(matrix.tobytes())
        points=np.empty((len(data.vertices),3),dtype=np.float32)
        data.vertices.foreach_get('co',points.ravel());digest.update(points.tobytes())
        for collection,field in ((data.loops,'vertex_index'),(data.polygons,'loop_total'),(data.polygons,'material_index')):
            values=np.empty(len(collection),dtype='<i4');collection.foreach_get(field,values);digest.update(values.tobytes())
        normals=np.empty(len(data.corner_normals)*3,dtype='<f4')
        data.corner_normals.foreach_get('vector',normals);digest.update(normals.tobytes())
        for uv in data.uv_layers:
            values=np.empty(len(uv.data)*2,dtype='<f4');uv.data.foreach_get('uv',values)
            digest.update(json.dumps([uv.name,uv.active_render]).encode()+values.tobytes())
        digest.update(str(data.uv_layers.active_index).encode())
        for name in ('.cm_wm_neutral','.cm_wm_posed',*[f'.cm_wm_linear_{i}' for i in range(3)]):
            attribute=data.attributes.get(name)
            if attribute:
                values=np.empty(len(attribute.data)*3,dtype='<f4')
                attribute.data.foreach_get('vector',values);digest.update(name.encode()+values.tobytes())
        result=digest.hexdigest()
        if cache is not None:cache[key]=result
        return result
    finally:evaluated.to_mesh_clear()


def shader_inputs(obj,cache,*,surface=False,fast_images=False):
    def texture(image):
        if not image or not fast_images or image.is_dirty or image.source!='FILE':
            return image_digest(image,cache)
        key=('encoded_image',image.as_pointer())
        if key not in cache:
            if image.packed_file:
                digest=hashlib.sha256(image.packed_file.data)
            else:
                path=Path(bpy.path.abspath(image.filepath,library=image.library))
                if not path.is_file():return image_digest(image,cache)
                with path.open('rb') as stream:digest=hashlib.file_digest(stream,'sha256')
            digest.update(json.dumps([image.source,image.colorspace_settings.name,image.alpha_mode]).encode())
            cache[key]=digest.hexdigest()
        return cache[key]
    def scalar(value):
        if isinstance(value,bpy.types.ID):return [value.bl_rna.identifier,value.name]
        if isinstance(value,(str,int,float,bool)) or value is None:return value
        try:return [scalar(item) for item in value]
        except TypeError:return str(value)
    def tree_state(tree,seen):
        if tree.as_pointer() in seen:return 'recursive group'
        seen=seen|{tree.as_pointer()}
        def inputs(node):
            if node.type.startswith('BSDF_') and not surface:return [socket for socket in node.inputs if 'Normal' in socket.name or socket.name=='Tangent']
            if node.type=='OUTPUT_MATERIAL':return [socket for socket in node.inputs if socket.name in {'Surface','Displacement'}]
            return node.inputs
        roots=[node for node in tree.nodes if node.type=='OUTPUT_MATERIAL' and node.is_active_output]
        active=set()
        def visit(node):
            if node in active:return
            active.add(node)
            for socket in inputs(node):
                for link in socket.links:visit(link.from_node)
        for node in roots:visit(node)
        # Shader groups have no Material Output; keep their complete internal
        # graph, conservatively, rather than losing a normal dependency.
        if not roots:active=set(tree.nodes)
        return [{'type':node.bl_idname,'name':node.name,
            'inputs':[(socket.identifier,scalar(socket.default_value)) for socket in inputs(node) if hasattr(socket,'default_value')],
            'links':[(socket.identifier,link.from_node.name,link.from_socket.identifier) for socket in inputs(node) for link in socket.links],
            'properties':{name:scalar(getattr(node,name)) for name in ('space','uv_map','attribute_name','operation',
                'blend_type','interpolation','extension','projection','noise_dimensions','vector_type') if hasattr(node,name)},
            'image':texture(node.image) if node.type=='TEX_IMAGE' else None,
            'group':tree_state(node.node_tree,seen) if node.type=='GROUP' and node.node_tree else None}
            for node in tree.nodes if node in active]
    evaluated=obj.evaluated_get(bpy.context.evaluated_depsgraph_get())
    return [tree_state(mat.node_tree,set()) if mat.use_nodes else list(mat.diffuse_color)
            for mat in evaluated.data.materials if mat]


def target(scene,key):
    s,f=scene.cm_settings,scene.cm_features
    if key=='FINAL_BODY' and s.part=='FULL_BODY':return f.body_result,f.body_detail
    return s.result,s.detail


def accepted_signature(scene):
    s,f=scene.cm_settings,scene.cm_features
    with bpy.context.temp_override(scene=scene,view_layer=scene.view_layers[0]):
        value=[s.part,mesh_digest(bpy.context,s.result)]
        if s.part=='FULL_BODY':value.append(mesh_digest(bpy.context,f.body_result))
    return hashlib.sha256(json.dumps(value).encode()).hexdigest()


def sculpt_sources_signature(scene):
    s,f=scene.cm_settings,scene.cm_features
    with bpy.context.temp_override(scene=scene,view_layer=scene.view_layers[0]):
        cache={}
        value=[mesh_digest(bpy.context,obj,cache) for obj in (s.detail,f.body_detail,f.wm1,f.wm2,f.wm3)]
    return hashlib.sha256(json.dumps(value).encode()).hexdigest()


def signature(scene,key,cache=None,*,recipe='FINAL',include_retained=True):
    cache={} if cache is None else cache
    s,f=scene.cm_settings,scene.cm_features
    if key.startswith('ROUGHNESS_'):
        albedo=folder(scene)/(prefix(scene,key)+'_ALBEDO.png')
        values={'albedo':hashlib.sha256(albedo.read_bytes()).hexdigest() if albedo.is_file() else None,
                'role':key.rsplit('_',1)[-1]}
        return hashlib.sha256(json.dumps(values,sort_keys=True).encode()).hexdigest()
    if key.startswith('CM'):
        albedo=folder(scene)/(s.character_name+'_ALBEDO.png')
        values={'color_resolution':'SAVED_ALBEDO','tint':f.bloodflow_strength,
            'mask':image_digest(getattr(f,'wm'+key[-1]+'_mask'),cache),
            'albedo':hashlib.sha256(albedo.read_bytes()).hexdigest() if albedo.is_file() else None}
        from . import skin_bloodflow
        reference=skin_bloodflow.native_reference(scene,key)
        if reference:values['native_expression_sha256']=reference['sha256']
        return hashlib.sha256(json.dumps(values,sort_keys=True).encode()).hexdigest()
    head,detail=target(scene,key)
    source_size=None
    if recipe in {'SOURCE_SKIN_NORMAL','SOURCE_GNM_NATIVE'}:
        path=folder(scene)/(prefix(scene,key)+'_NORMAL.png')
        with path.open('rb') as stream:header=stream.read(24)
        if header[:8]!=b'\x89PNG\r\n\x1a\n':raise ValueError('Source normal needs a valid delivered PNG')
        source_size=[int.from_bytes(header[16:20],'big'),int.from_bytes(header[20:24],'big')]
    with bpy.context.temp_override(scene=scene,view_layer=scene.view_layers[0]):
        if recipe in {'REFERENCE','NATIVE_COPY','GNM_NATIVE_COPY','SOURCE_GNM_NATIVE'}:
            graph=bpy.context.evaluated_depsgraph_get();evaluated=head.evaluated_get(graph)
            mesh=evaluated.to_mesh(preserve_all_data_layers=True,depsgraph=graph)
            try:
                digest=hashlib.sha256()
                for uv in mesh.uv_layers:
                    values=np.empty(len(uv.data)*2,dtype='<f4');uv.data.foreach_get('uv',values)
                    digest.update(json.dumps([uv.name,uv.active_render]).encode()+values.tobytes())
                ref=head.get('cm_reference_normal_image')
                digest.update(json.dumps([recipe,s.map_size if recipe=='NATIVE_COPY' else None,
                    image_digest(ref,cache) if isinstance(ref,bpy.types.Image) else None]).encode())
                if recipe in {'GNM_NATIVE_COPY','SOURCE_GNM_NATIVE'}:
                    from . import textures
                    native=textures.generated_reference_normal(bpy.context,s,f)
                    values={'source':mesh_digest(bpy.context,s.source,cache),
                        'normal_inputs':shader_inputs(s.source,cache) if s.source else None,
                        'native':image_digest(native,cache) if native else None,'size':source_size or s.map_size}
                    digest.update(json.dumps(values,sort_keys=True).encode())
                return digest.hexdigest()
            finally:evaluated.to_mesh_clear()
        values={'target':mesh_digest(bpy.context,head,cache),'distance':s.bake_distance,
                'size':source_size or (s.map_size if key.startswith('FINAL_') else f.wrinkle_map_size),'recipe':recipe}
        if key.startswith('FINAL_') and recipe not in {'CAVITY','FINAL_CAVITY'}:
            from . import normal_composition
            values['distance']=normal_composition.projection_distance(head,s.bake_distance)
        geometry_only=bool(head and head.get('cm_normal_composition')=='GEOMETRY_ONLY' and key.startswith('FINAL_') and recipe in {'FINAL','FINAL_NORMAL','GENERATED_FINAL_NORMAL'})
        if geometry_only:values['normal_composition']='GEOMETRY_ONLY'
        if key.startswith('FINAL_') and recipe in {'FINAL','CAVITY','FINAL_CAVITY'}:values['cavity_method']='PHYSICAL_ADJACENCY_V1'
        generated=False
        generated_normal=(key=='FINAL_HEAD' and recipe in {'FINAL','FINAL_NORMAL','GENERATED_FINAL_NORMAL'}) or (
            key in {'WM1','WM2','WM3'} and getattr(f,key.lower()) and getattr(f,key.lower()).get('cm_wrinkle_native'))
        if generated_normal and not geometry_only:
            from . import textures
            generated=textures.generated_skin_source(s,f)
            if generated:
                native=textures.generated_reference_normal(bpy.context,s,f)
                values['generated_skin']={'source':mesh_digest(bpy.context,s.source,cache),
                    'normal_inputs':shader_inputs(s.source,cache),
                    'native':image_digest(native,cache) if native else None}
        if key.startswith('FINAL_'):
            cavity=recipe in {'CAVITY','FINAL_CAVITY'}
            source=(detail or head) if cavity else s.source if recipe in {'SKIN_NORMAL','SOURCE_SKIN_NORMAL'} else detail or (s.source if recipe=='NORMAL' or (head and head.get('cm_body_source')!='LIBRARY') else None)
            pores=bool(f.add_pores and detail and recipe not in {'NORMAL','SKIN_NORMAL','SOURCE_SKIN_NORMAL','CAVITY'})
            values.update(source=mesh_digest(bpy.context,source,cache),pores=pores,
                pore_strength=f.pore_strength if pores else None,
                pore_spacing=f.pore_spacing if pores else None,
                cavity=f.cavity_radius if recipe not in {'NORMAL','SKIN_NORMAL','SOURCE_SKIN_NORMAL','FINAL_NORMAL','GENERATED_FINAL_NORMAL'} else None)
        else:
            layer=getattr(f,key.lower())
            values.update(source=mesh_digest(bpy.context,layer,cache))
            if layer and not layer.get('cm_wrinkle_native'):values['normal_inputs']=shader_inputs(layer,cache)
        uses_reference=(bool(detail) and recipe not in {'NORMAL','SKIN_NORMAL','SOURCE_SKIN_NORMAL','CAVITY','FINAL_CAVITY','GENERATED_FINAL_NORMAL'}) if key.startswith('FINAL_') else bool(getattr(f,key.lower()) and getattr(f,key.lower()).get('cm_wrinkle_native'))
        if geometry_only:uses_reference=False
        if generated and key in {'WM1','WM2','WM3'}:uses_reference=False
        reference=head.get('cm_reference_normal_image') if head and uses_reference else None
        if isinstance(reference,bpy.types.Image):values['reference']=image_digest(reference,cache)
        else:
            path=Path(head.get('cm_reference_normal','')) if head and uses_reference else None
            values['reference']=hashlib.sha256(path.read_bytes()).hexdigest() if path and path.is_file() else None
        if include_retained and not geometry_only and key.startswith('FINAL_') and recipe not in {'CAVITY','FINAL_CAVITY','SKIN_NORMAL','SOURCE_SKIN_NORMAL','SOURCE_GNM_NATIVE'}:
            from . import skin_bake_layers
            retained=skin_bake_layers.fingerprint(head)
            if retained is not None:values['retained_skin_layers']=retained
        # When normals are projected without a reference skin, shader normal
        # inputs can contribute too. Geometry-only detail excludes that shader.
        if key.startswith('FINAL_') and source and recipe not in {'CAVITY','FINAL_CAVITY'}:
            if not values.get('reference') and not generated and not geometry_only:
                values['normal_inputs']=shader_inputs(source,cache)
        return hashlib.sha256(json.dumps(values,sort_keys=True).encode()).hexdigest()


def capture(scene,mode,slots=()):
    s=scene.cm_settings
    if mode=='FINAL_MAPS':
        from . import textures
        parts=textures.final_parts(s,scene.cm_features)
    else:parts=('HEAD','BODY') if s.part=='FULL_BODY' else (s.part,)
    keys=['CM'+slot[-1] for slot in slots] if mode=='BLOODFLOW' else list(slots) if mode=='WRINKLES' else ['FINAL_'+part for part in
        parts]
    if mode=='WRINKLES' and scene.cm_features.bloodflow:keys+=['CM'+slot[-1] for slot in slots]
    cache={}
    return {key:signature(scene,key,cache) for key in keys}


def prepare(scene,mode,slots=()):
    from . import textures,wrinkles
    s,f=scene.cm_settings,scene.cm_features
    if mode=='BLOODFLOW':
        for slot in slots:wrinkles.ensure_mask(s,f,slot)
    elif mode=='WRINKLES':
        if f.bloodflow:
            for slot in slots:wrinkles.ensure_mask(s,f,slot)
        if any(getattr(f,slot.lower()).get('cm_wrinkle_native') for slot in slots):
            textures.reference_normal(s)
    else:
        for role in (textures.final_parts(s,f) if mode=='FINAL_MAPS' else ('HEAD','BODY') if s.part=='FULL_BODY' else (s.part,)):
            with textures.skin_part(s,f,role):
                if s.detail and s.result.get('cm_normal_composition')!='GEOMETRY_ONLY':textures.reference_normal(s)
    return capture(scene,mode,slots)


def validate(scene,basis):
    cache={}
    if any(signature(scene,key,cache)!=value for key,value in basis.items()):
        raise ValueError('Bake inputs changed while the worker ran; previous maps retained. Bake again when ready.')


def record(scene,basis,*,origin='BAKE',outputs=None,recipe='FINAL',details=None):
    value=load(scene)
    invalidated=json.loads(scene.get(INVALIDATED,'{}'))
    for key,stamp in basis.items():
        files={path.name:hashlib.sha256(path.read_bytes()).hexdigest() for path in paths(scene,key)
            if path.is_file() and (outputs is None or path.name in outputs)}
        if files:
            ident=identity(scene,key);previous=value['entries'].get(ident,{})
            artifacts=dict(previous.get('artifacts',{}))
            if not artifacts:
                artifacts={name:{'sha256':sha,'stamp':previous['stamp'],'recipe':'FINAL','origin':previous.get('origin','BAKE')}
                    for name,sha in previous.get('files',{}).items()}
            cache={}
            for name,sha in files.items():
                method=recipe
                if origin!='BAKE':method='CONTEXT'
                elif recipe=='FINAL' and key.startswith('FINAL_'):
                    obj,detail=target(scene,key)
                    method='REFERENCE' if obj and not detail and obj.get('cm_body_source')=='LIBRARY' else (
                        'FINAL_CAVITY' if name.endswith('_CAVITY.png') else 'FINAL_NORMAL')
                    if method=='FINAL_NORMAL' and key=='FINAL_HEAD':
                        from . import textures
                        if textures.generated_skin_source(scene.cm_settings,scene.cm_features):method='GENERATED_FINAL_NORMAL'
                artifacts[name]={'sha256':sha,'stamp':signature(scene,key,cache,recipe=method),
                    'recipe':method,'origin':origin,'time':time.time()}
                if details and name in details:
                    for field in ('dependencies','skin_recipe'):
                        if field in details[name]:
                            artifacts[name][field]=json.loads(json.dumps(details[name][field]))
            value['entries'][ident]={'key':key,'stamp':stamp,'files':{name:item['sha256'] for name,item in artifacts.items()},
                'artifacts':artifacts,'time':time.time(),'origin':origin}
            pending=invalidated.get(ident,[])
            if pending is True:pending=[path.name for path in paths(scene,key) if path.is_file()]
            if not pending:pending=[]
            if not isinstance(pending,list):raise ValueError('Invalid pending-map file list; retain it and check the project')
            remaining=[name for name in pending if name not in files]
            if remaining:invalidated[ident]=remaining
            else:invalidated.pop(ident,None)
    path=manifest(scene);path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_suffix('.tmp')
    previous=scene.get(INVALIDATED)
    try:
        scene[INVALIDATED]=json.dumps(invalidated)
        temporary.write_text(json.dumps(value,indent=2),encoding='utf-8');temporary.replace(path)
    except Exception:
        if previous is None:
            if INVALIDATED in scene:del scene[INVALIDATED]
        else:scene[INVALIDATED]=previous
        raise


def invalidate_keys(scene,keys):
    invalidated=json.loads(scene.get(INVALIDATED,'{}'))
    for key in keys:
        # Untracked library skin maps remain external reference assets. Existing
        # custom wrinkle outputs and head-detail outputs must follow a new face.
        if key=='FINAL_HEAD' and not scene.cm_settings.detail:continue
        if any(path.is_file() for path in paths(scene,key)):
            invalidated[identity(scene,key)]=[path.name for path in paths(scene,key) if path.is_file()]
    scene[INVALIDATED]=json.dumps(invalidated)


def invalidate_face(scene):
    invalidate_keys(scene,['FINAL_HEAD','WM1','WM2','WM3'])


def adopt_custom(scene):
    from . import wrinkles
    wrinkles.refresh_source_state(bpy.context,scene.cm_settings,scene.cm_features)
    from . import detail_jobs,wrinkle_jobs
    s,f=scene.cm_settings,scene.cm_features
    if detail_jobs.running() or wrinkle_jobs.running() or (s.detail and s.detail.get('cm_head_update_pending')) or any(
        obj and obj.get('cm_wrinkle_update_pending') for obj in (f.wm1,f.wm2,f.wm3)):
        raise ValueError('Finish the retained sculpt updates before assigning custom maps')
    if not scene.cm_settings.result:raise ValueError('Wrap the character before assigning custom maps')
    keys=['FINAL_'+part for part in (('HEAD','BODY') if scene.cm_settings.part=='FULL_BODY' else (scene.cm_settings.part,))]
    if scene.cm_settings.part!='BODY':keys+=['WM1','WM2','WM3','CM1','CM2','CM3']
    cache={}
    basis={key:signature(scene,key,cache) for key in keys if any(path.is_file() for path in paths(scene,key))}
    if not basis:raise ValueError('No current map files to use; bake or import maps first')
    record(scene,basis,origin='CUSTOM')
    return len(basis)


def basis_for_files(scene,names):
    keys=['FINAL_'+part for part in (('HEAD','BODY') if scene.cm_settings.part=='FULL_BODY' else (scene.cm_settings.part,))]
    if scene.cm_settings.part!='BODY':keys+=['WM1','WM2','WM3','CM1','CM2','CM3']
    cache={}
    return {key:signature(scene,key,cache) for key in keys if any(path.name in names for path in paths(scene,key))}


def sculpt_needs_normal(scene,key,cache):
    _,detail=target(scene,key)
    if not detail:return False
    ident=('sculpt_normal',detail.as_pointer())
    if ident not in cache:
        reference=detail.data.attributes.get('.cm_detail_reference')
        if not reference or scene.cm_features.add_pores:cache[ident]=True
        else:
            with bpy.context.temp_override(scene=scene,view_layer=scene.view_layers[0]):
                evaluated=detail.evaluated_get(bpy.context.evaluated_depsgraph_get())
                data=evaluated.to_mesh()
                try:
                    if len(data.vertices)!=len(reference.data):cache[ident]=True
                    else:
                        points=np.empty(len(data.vertices)*3,dtype=np.float32)
                        baseline=np.empty_like(points)
                        data.vertices.foreach_get('co',points);reference.data.foreach_get('vector',baseline)
                        cache[ident]=bool(np.max(np.abs(points-baseline),initial=0)>1e-6)
                finally:evaluated.to_mesh_clear()
    return cache[ident]


def problems(scene, *, keys=None, suffixes=None, ignore_dependencies=False, cache=None):
    result=[]
    invalidated=json.loads(scene.get(INVALIDATED,'{}'))
    records=load(scene)['entries'];cache={} if cache is None else cache
    if keys is None:
        keys=['FINAL_'+part for part in (('HEAD','BODY') if scene.cm_settings.part=='FULL_BODY' else (scene.cm_settings.part,))]
        if scene.cm_settings.part!='BODY':keys+=['WM1','WM2','WM3','CM1','CM2','CM3']
    for key in keys:
        ident=identity(scene,key);entry=records.get(ident)
        label={'FINAL_HEAD':'head normal/cavity','FINAL_BODY':'body normal/cavity'}.get(key,
            key+' blood-flow' if key.startswith('CM') else key+' wrinkle')
        if invalidated.get(ident) and (entry or any(path.is_file() for path in paths(scene,key))):
            result.append('Rebake '+label+' maps after updating the face.');continue
        if not entry:continue  # Existing external/reference maps have no invented bake history.
        artifacts=entry.get('artifacts') or {name:{'sha256':sha,'stamp':entry['stamp'],'recipe':'FINAL'} for name,sha in entry['files'].items()}
        for name,item in artifacts.items():
            if suffixes and not name.endswith(suffixes):continue
            if Path(name).name!=name:raise ValueError('Invalid map filename in bake record')
            if key.startswith('FINAL_') and name.endswith('_NORMAL.png') and item['recipe'] in {'REFERENCE','NATIVE_COPY','GNM_NATIVE_COPY','SKIN_NORMAL','SOURCE_SKIN_NORMAL','SOURCE_GNM_NATIVE'} and sculpt_needs_normal(scene,key,cache):
                result.append('Rebake '+label+' maps: '+name+' does not include the current sculpt. Use Bake Final Normal & Cavity.');break
            if item['stamp']!=signature(scene,key,cache,recipe=item['recipe']):
                result.append('Rebake '+label+' maps: '+name+' has outdated geometry, UVs or bake inputs.');break
            if not ignore_dependencies:
                changed=False
                for dependency,sha in item.get('dependencies',{}).items():
                    if Path(dependency).name!=dependency:raise ValueError('Invalid derived-map dependency')
                    source=folder(scene)/dependency
                    if not source.is_file() or hashlib.sha256(source.read_bytes()).hexdigest()!=sha:
                        result.append('Refresh skin shading: '+name+' was inferred from an earlier '+dependency+'.')
                        changed=True;break
                if changed:break
            path=folder(scene)/name
            if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest()!=item['sha256']:
                result.append('Rebake '+label+' maps: a recorded output is missing or changed.');break
    return result


def record_single(scene,path,recipe):
    """Record one scoped head/body output without certifying adjacent files."""
    s,f=scene.cm_settings,scene.cm_features
    scoped_body=s.part=='FULL_BODY' and s.result==f.body_result
    name=s.character_name
    try:
        if scoped_body and name.endswith('_Body'):s.character_name=name[:-5]
        key='FINAL_BODY' if scoped_body or s.part=='BODY' else 'FINAL_HEAD'
        record(scene,{key:signature(scene,key,recipe=recipe)},outputs={Path(path).name},recipe=recipe)
    finally:s.character_name=name
