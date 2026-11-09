"""Owned, cancellable skin reconstruction with staged publication and revert."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import uuid

import bpy
from bpy.app.handlers import persistent

from . import paint_session, map_provenance, skin_library, skin_projection
from . import skin_normal_history, skin_bloodflow

_JOB = None
_PREVIEW = None


@persistent
def restore_preview(*_):
    global _PREVIEW
    preview,_PREVIEW=_PREVIEW,None
    if not preview:return
    for obj,hidden in preview.get('source_visibility',()):
        try:obj.hide_set(hidden)
        except ReferenceError:pass
    for node,original in preview['nodes']:
        try:
            if node.image==preview['image']:node.image=original
        except ReferenceError:pass
    try:
        if preview['image'].users==0:bpy.data.images.remove(preview['image'])
    except ReferenceError:pass
    try:
        preview['scene'].cm_features.skin_solver_status='Showing accepted color. The candidate is retained for inspection, Apply or Discard.'
    except ReferenceError:pass


def preview_color(context):
    global _PREVIEW
    if not _JOB or not _JOB.get('ready') or _JOB['scene']!=context.scene:
        raise ValueError('Prepare a color candidate for this character first')
    if not _JOB['state'].get('color',True):raise ValueError('This candidate has no color changes')
    restore_preview()
    role,obj,color,_=_images(context)
    image=bpy.data.images.load(str(_JOB['folder']/'candidate-albedo.png'),check_existing=False)
    image.colorspace_settings.name='sRGB'
    s,f=context.scene.cm_settings,context.scene.cm_features
    objects=(s.result,s.detail) if role=='HEAD' else (f.body_result,f.body_detail)
    nodes={node.as_pointer():node for target in objects if target for material in target.data.materials
           if material and material.use_nodes for node in material.node_tree.nodes
           if node.type=='TEX_IMAGE' and node.image==color}
    if not nodes:
        bpy.data.images.remove(image)
        raise ValueError('The accepted skin has no color texture nodes to preview')
    _PREVIEW={'scene':context.scene,'obj':obj,'image':image,'color':color,
              'nodes':[(node,node.image) for node in nodes.values()]}
    from . import compare_views
    source=s.source
    if source and source!=obj and not compare_views._SESSIONS:
        _PREVIEW['source_visibility']=[(source,source.hide_get())];source.hide_set(True)
    for node in nodes.values():node.image=image
    f.skin_solver_status='Previewing candidate color on skin. Accepted files and normals are retained; Apply or Discard when ready.'


def preview_generated(context):
    global _PREVIEW
    if running():raise ValueError('Inspect the staged candidate while its material solve is active')
    restore_preview()
    role,obj,color,_=_images(context)
    image=generated_reference(context.scene,role,obj)
    if not image:raise ValueError('Choose a generated UV-matched albedo first')
    s,f=context.scene.cm_settings,context.scene.cm_features
    objects=(s.result,s.detail) if role=='HEAD' else (f.body_result,f.body_detail)
    nodes={node.as_pointer():node for target in objects if target for material in target.data.materials
        if material and material.use_nodes for node in material.node_tree.nodes
        if node.type=='TEX_IMAGE' and node.image==color}
    if not nodes:raise ValueError('The accepted skin has no color texture nodes')
    _PREVIEW={'scene':context.scene,'obj':obj,'image':image,'color':color,'nodes':[(node,node.image) for node in nodes.values()]}
    from . import compare_views
    if s.source and s.source!=obj and not compare_views._SESSIONS:
        _PREVIEW['source_visibility']=[(s.source,s.source.hide_get())];s.source.hide_set(True)
    for node in nodes.values():node.image=image
    f.skin_solver_status='Previewing generated input only; derived maps are unchanged. Show Accepted Color restores the original.'


def running():
    return _JOB is not None


def _hash(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def retained_material_input(scene,role,accepted):
    """Control-only tuning reuses the unredistributed input; paint starts anew."""
    f=scene.cm_features
    if not f.skin_solver_last:return None
    previous=_folder(scene,f.skin_solver_last)
    seen=set()
    while True:
        if previous in seen:raise ValueError('The material input history contains a cycle')
        seen.add(previous);state=json.loads((previous/'state.json').read_text())
        if state['stamp']['role']==role:break
        if not state.get('previous_solver_last'):return None
        previous=_folder(scene,state['previous_solver_last'])
    if (state.get('status')!='applied'
            or _hash(accepted)!=state.get('outputs',{}).get(accepted.name)):
        return None
    request=json.loads((previous/'request.json').read_text())
    if request.get('method')!='MATERIAL':return None
    if state.get('staged'):
        from . import skin_solve_staged
        plan=json.loads((previous/'staged.json').read_text())
        if skin_solve_staged.digest(previous/'request.json')!=plan['request_sha256']:
            raise ValueError('The retained material request changed; inspect its original input before tuning')
    record=request['inputs']['albedo'];path=(previous/record['file']).resolve()
    if not path.is_relative_to(previous) or _hash(path)!=record['sha256']:
        raise ValueError('The retained material color input changed; keep its original job files for control-only tuning')
    return path,record['sha256']


def initialized_roughness_nodes(scene,state,*,remove=False):
    """Bind a newly created roughness map only to unlinked managed skin inputs."""
    filename=state['prefix']+'_ROUGHNESS.png'
    if filename not in state.get('initialized_materials',[]):return
    s,f=scene.cm_settings,scene.cm_features
    objects=(f.body_result,f.body_detail) if state['stamp']['role']=='BODY' and s.part=='FULL_BODY' else (s.result,s.detail)
    materials={mat.as_pointer():mat for obj in objects if obj for mat in obj.data.materials if mat and mat.use_nodes}
    image=None
    for mat in materials.values():
        tree=mat.node_tree
        if remove:
            for node in list(tree.nodes):
                if node.get('cm_solver_initialized_roughness')==filename:tree.nodes.remove(node)
            continue
        if any(node.get('cm_solver_initialized_roughness')==filename for node in tree.nodes):continue
        inputs=[node.inputs['Roughness'] for node in tree.nodes if node.type=='BSDF_PRINCIPLED' and not node.inputs['Roughness'].is_linked]
        if not inputs:continue  # Retain an authored shader input.
        if any(obj not in objects and mat in obj.data.materials[:] for obj in bpy.data.objects if obj.type=='MESH'):continue
        if image is None:
            image=bpy.data.images.load(str(Path(bpy.path.abspath(s.project_dir))/'Maps'/filename),check_existing=False)
            image.colorspace_settings.name='Non-Color';image.alpha_mode='NONE';image.pack()
        texture=tree.nodes.new('ShaderNodeTexImage');texture.image=image;texture['cm_map_kind']='ROUGHNESS'
        texture['cm_solver_initialized_roughness']=filename
        channel=tree.nodes.new('ShaderNodeSeparateColor');channel['cm_solver_initialized_roughness']=filename
        scale=tree.nodes.new('ShaderNodeMath');scale.operation='MULTIPLY';scale.use_clamp=True
        scale['cm_roughness_scale']=True;scale['cm_solver_initialized_roughness']=filename
        scale.inputs[1].default_value=f.lookdev_roughness/.48 if f.show_lookdev else 1
        tree.links.new(texture.outputs['Color'],channel.inputs['Color'])
        tree.links.new(channel.outputs['Red'],scale.inputs[0])
        for socket in inputs:tree.links.new(scale.outputs[0],socket)


def _geometry(context,cache=None):
    s, f = context.scene.cm_settings, context.scene.cm_features
    return [map_provenance.mesh_digest(context, o,cache) for o in (s.result, f.body_result) if o]


def _images(context):
    role, obj = paint_session.part(context)
    color = paint_session.albedo(obj)
    if _PREVIEW and _PREVIEW['scene']==context.scene and _PREVIEW['obj']==obj:
        if all(node.image==_PREVIEW['image'] for node,_ in _PREVIEW['nodes']):
            color=_PREVIEW['color']
        else:
            restore_preview();color=paint_session.albedo(obj)
    f=context.scene.cm_features
    reference = paint_session.reference(context, role, obj) if f.skin_solver_color and f.skin_solver_method=='REFERENCE' else None
    if f.skin_solver_color and f.skin_solver_method=='GENERATIVE':
        reference=generated_reference(context.scene,role,obj)
        if not reference:raise ValueError('Choose the generated UV-matched '+role.lower()+' albedo texture')
    if not reference and f.skin_solver_color and f.skin_solver_method=='REFERENCE':
        raise ValueError('Choose a UV-matched reference skin before reconstructing')
    if any(i.source == 'TILED' or i.colorspace_settings.name != 'sRGB' for i in (color, reference) if i):
        raise ValueError('Reconstruction requires flat sRGB albedo and reference images')
    known_layout=reference.get('cm_native_uv_layout') if reference else None
    if known_layout and known_layout!=skin_projection.uv_layout_signature(obj):
        raise ValueError('The reference skin uses a different UV layout')
    return role, obj, color, reference


def generated_reference(scene,role,obj):
    f=scene.cm_features
    if role!='BODY':return f.skin_solver_generated_albedo
    if f.skin_solver_generated_body_albedo:return f.skin_solver_generated_body_albedo
    legacy=f.skin_solver_generated_albedo
    if legacy and legacy.get('cm_native_uv_layout')==skin_projection.uv_layout_signature(obj):return legacy
    return None


def paint_protection(scene,role):
    return scene.cm_features.skin_solver_body_protect if role=='BODY' else scene.cm_features.skin_solver_protect


def _normal(obj):
    def feeds_normal(node,seen=None):
        seen=set() if seen is None else seen
        if node in seen:return False
        seen.add(node)
        return any(link.to_node.type=='NORMAL_MAP' or feeds_normal(link.to_node,seen)
                   for output in node.outputs for link in output.links)
    images={n.image.as_pointer():n.image for mat in obj.data.materials if mat and mat.use_nodes
            for n in mat.node_tree.nodes if n.type=='TEX_IMAGE' and n.image and
            (n.get('cm_map_kind')=='NORMAL' or n.name in {'CM_NormalTex','CM_Normal'})}
    if not images:
        # Older preview materials identify normals by their Normal Map link.
        images={n.image.as_pointer():n.image for mat in obj.data.materials if mat and mat.use_nodes
                for n in mat.node_tree.nodes if n.type=='TEX_IMAGE' and n.image and
                feeds_normal(n)}
    if len(images)!=1:raise ValueError('Prepare one tangent normal map for this skin part')
    image=next(iter(images.values()))
    if image.source=='TILED' or image.colorspace_settings.name!='Non-Color' or image.is_dirty:
        raise ValueError('Save the flat Non-Color normal map before reconstructing')
    return image


def _stamp(context, with_normal=False, *, legacy=False, cache=None,legacy_color_scope=False,bloodflow=(),legacy_resolution=False):
    role, obj, color, reference = _images(context)
    cache = {} if cache is None else cache
    f = context.scene.cm_features
    if legacy:reference=paint_session.reference(context,role,obj)
    mask=f.skin_solver_mask if legacy or f.skin_solver_color or f.skin_solver_relief else None
    state={'role':role, 'geometry':_geometry(context,cache),
        'uv':skin_projection.uv_layout_signature(obj),
        'images':[map_provenance.image_digest(i, cache) if i else None for i in (color, reference, mask)]}
    if not legacy_resolution:state['output_resolution']=f.skin_solver_output
    if f.skin_solver_color and f.skin_solver_material_strength:
        root=Path(bpy.path.abspath(context.scene.cm_settings.project_dir)).resolve()
        prefix=context.scene.cm_settings.character_name+('_Body' if role=='BODY' and context.scene.cm_settings.part=='FULL_BODY' else '')
        state['derived_material']={'strength':f.skin_solver_material_strength,
            'baselines':{kind:_hash(root/'Maps'/(prefix+'_'+kind+'.png')) for kind in ('ROUGHNESS','CAVITY') if (root/'Maps'/(prefix+'_'+kind+'.png')).is_file()}}
    if f.skin_solver_color and f.skin_solver_method in {'GENERATIVE','MATERIAL'} and f.skin_solver_surface_balance:
        state['surface_balance']=f.skin_solver_surface_balance
    if role=='BODY' and f.skin_solver_color and f.skin_solver_method=='GENERATIVE':
        state['preserve_body_anatomy']=f.skin_solver_preserve_body_anatomy
        if f.skin_solver_preserve_body_anatomy:
            from . import skin_anatomy
            state['anatomy_weights']=skin_anatomy.vertex_weights(obj)[1]['weight_signature']
    if f.skin_solver_color and (not legacy_color_scope or f.skin_solver_source or f.skin_solver_method in {'LEARNED','SEAMS'}):
        state['source']=map_provenance.image_digest(f.skin_solver_source if f.skin_solver_method not in {'SEAMS','MATERIAL','LOCAL_FILL'} else None,cache)
        state['protect']=map_provenance.image_digest(paint_protection(context.scene,role),cache)
    if with_normal:
        image=_normal(obj)
        state['normal']={'file':str(Path(bpy.path.abspath(image.filepath)).resolve()),
                         'sha256':_hash(bpy.path.abspath(image.filepath)),'size':list(image.size)}
        state['surface_mask']=map_provenance.image_digest(f.skin_solver_surface_mask if legacy or f.skin_solver_surface_repair else None,cache)
        state['sculpt']=map_provenance.signature(context.scene,'FINAL_'+role,cache,include_retained=False)
    if bloodflow:state['bloodflow']=skin_bloodflow.stamp(context.scene,bloodflow,cache)
    return json.loads(json.dumps(state))


def _folder(scene, value):
    root = Path(bpy.path.abspath(scene.cm_settings.project_dir)).resolve()
    path = Path(value).resolve()
    if path.parent != root/'SkinSolves' or path.parent.resolve() != root/'SkinSolves':
        raise ValueError('The skin solve belongs to another project or linked folder')
    return path


def paint_region(context):
    role,obj=paint_session.part(context)
    image=obj.get('cm_last_paint_mask') if obj else None
    if not isinstance(image,bpy.types.Image) or image.get('cm_paint_role')!=role:
        raise ValueError('Apply Smart Paint on this skin part first to retain its brush region')
    if image.get('cm_native_uv_layout')!=skin_projection.uv_layout_signature(obj) or image.get('cm_paint_geometry')!=map_provenance.mesh_digest(context,obj):
        raise ValueError('The painted region belongs to earlier geometry or UVs; paint the current skin first')
    return image


def repair_painted(context):
    """Turn the retained brush region into a normal-only staged candidate."""
    f=context.scene.cm_features
    if running() or f.skin_solver_job:
        raise ValueError('Apply or discard the current skin candidate first')
    image=paint_region(context)
    settings={'skin_solver_color':False,'skin_solver_surface_repair':True,
        'skin_solver_surface_mask':image,'skin_solver_relief':0.,'skin_solver_floor':0.,
        'skin_solver_mask':None,'skin_solver_strength':1.}
    previous={key:getattr(f,key) for key in settings}
    try:
        for key,value in settings.items():setattr(f,key,value)
        start(context)
    except BaseException:
        for key,value in previous.items():setattr(f,key,value)
        raise


def retain_projection(context,obj,image):
    """Keep one immutable, project-owned input before any color reconstruction."""
    current=obj.get('cm_skin_source_albedo')
    if isinstance(current,bpy.types.Image):return current
    root=Path(bpy.path.abspath(context.scene.cm_settings.project_dir)).resolve()
    parent=root/'SourceMaps';parent.mkdir(exist_ok=True)
    if parent.resolve()!=parent:raise ValueError('Source projections must remain inside the character project')
    path=parent/(context.scene.cm_settings.character_name+'_'+uuid.uuid4().hex[:12]+'_ALBEDO.png')
    skin_library.save_image_copy(image,path)
    retained=bpy.data.images.load(str(path),check_existing=False)
    retained.colorspace_settings.name='sRGB';retained.pack()
    retained['cm_native_uv_layout']=skin_projection.uv_layout_signature(obj)
    retained['cm_source_geometry']=map_provenance.mesh_digest(context,obj)
    obj['cm_skin_source_albedo']=retained
    return retained


def complete_painted_color(context):
    """Build a reviewable color candidate from the retained Smart Paint mask."""
    f=context.scene.cm_features
    if running() or f.skin_solver_job:raise ValueError('Apply or discard the current skin candidate first')
    image=paint_region(context)
    settings=dict(skin_solver_color=True,skin_solver_method='LOCAL_FILL',skin_solver_mask=image,
                  skin_solver_source=None,skin_solver_output='KEEP',skin_solver_surface_repair=False,
                  skin_solver_relief=0.,skin_solver_floor=0.,skin_solver_surface_balance=0.)
    previous={key:getattr(f,key) for key in settings}
    try:
        for key,value in settings.items():setattr(f,key,value)
        start(context)
    except BaseException:
        for key,value in previous.items():setattr(f,key,value)
        raise


def refresh_normals(context):
    f=context.scene.cm_features
    previous={key:getattr(f,key) for key in ('skin_solver_color','skin_solver_relief','skin_solver_surface_repair')}
    try:
        f.skin_solver_color=False;f.skin_solver_relief=0;f.skin_solver_surface_repair=False
        start(context,refresh=True)
    except BaseException:
        for key,value in previous.items():setattr(f,key,value)
        raise


def start(context,*,refresh=False,defer=False,surface_reference=None):
    global _JOB
    from . import character_project, resource_budget, setup_paths
    s, f = context.scene.cm_settings, context.scene.cm_features
    if f.skin_solver_job:
        raise ValueError('Resume or discard the existing skin candidate first')
    character_project.preflight(context, s, f)
    setup_paths.resolve(f)
    role, obj, color, reference = _images(context)
    bloodflow=skin_bloodflow.plan(context.scene,role) if f.skin_solver_color else []
    learned=bool(f.skin_solver_color and f.skin_solver_method=='LEARNED')
    generative=bool(f.skin_solver_color and f.skin_solver_method=='GENERATIVE')
    current_material=bool(f.skin_solver_color and f.skin_solver_method=='MATERIAL')
    local_fill=bool(f.skin_solver_color and f.skin_solver_method=='LOCAL_FILL')
    if local_fill and not f.skin_solver_mask:raise ValueError('Paint or select an explicit color completion mask first')
    balance=bool(f.skin_solver_surface_balance and (generative or current_material))
    material_strength=f.skin_solver_material_strength if f.skin_solver_color else 0
    if current_material and not (material_strength or f.skin_solver_relief):
        raise ValueError('Set Surface Detail or Derive Skin Material before building the current albedo material')
    if balance and (not f.skin_solver_relief or not material_strength):
        raise ValueError('Set Surface Detail and Derive Skin Material above zero before moving shading into surface maps')
    diffusion_sr=learned and f.skin_solver_prior=='PISA'
    worker_python=bpy.path.abspath(f.skin_solver_sr_python if diffusion_sr else f.inference_python)
    if not Path(worker_python).is_file():raise ValueError('Choose an existing Diffusion Python environment before using PiSA-SR' if diffusion_sr else 'Choose the inference Python environment')
    upscale=bool(((learned and (diffusion_sr or (f.skin_solver_prior=='EDIT' and f.skin_solver_projection=='SURFACE'))) or generative) and
                 f.skin_solver_output=='8192' and max(color.size)<8192)
    if upscale and learned and not diffusion_sr and tuple(color.size)!=(4096,4096):raise ValueError('The learned surface 8K upscale currently requires a 4K skin atlas')
    if diffusion_sr and (color.size[0]!=color.size[1] or color.size[0] not in (256,512,1024,2048,4096,8192)):
        raise ValueError('Diffusion SR needs a square power-of-two skin atlas from 256 through 8192')
    if upscale and diffusion_sr and color.size[0] not in (2048,4096):
        raise ValueError('Choose a 2K or 4K skin atlas for an 8K diffusion upscale; smaller previews can keep their current size')
    if upscale and (color.size[0]!=color.size[1] or 8192%color.size[0]):
        raise ValueError('8K reconstruction needs a square atlas whose resolution divides 8192')
    output_scale=8192//color.size[0] if upscale else 1
    seam_repair=bool(f.skin_solver_color and f.skin_solver_method=='SEAMS')
    backend=None;backend_hashes={}
    if learned:
        if not f.skin_solver_mask:
            raise ValueError('Choose an explicit reconstruction mask for local detail restoration')
        editing=f.skin_solver_prior=='EDIT'
        backend=(Path(bpy.path.abspath(f.model_dir)).resolve()/'albedo-superresolution/pisa-sr' if diffusion_sr else
                 Path(bpy.path.abspath(f.model_dir)).resolve()/'texture-restoration'/('flux2-klein4b' if editing else 'ncnn'))
        if diffusion_sr:
            from .texture_sr_prior import expected
            backend_hashes=expected(backend)
        elif editing:
            from .texture_backend_identity import expected_editor
            backend_hashes=expected_editor(backend)
        else:
            for name in ('realesrgan-ncnn-vulkan.exe','models/realesrgan-x4plus.bin','models/realesrgan-x4plus.param'):
                if not (backend/name).is_file():
                    raise ValueError('Install the selected optional texture backend in '+str(backend)+' first')
                backend_hashes[name]=_hash(backend/name)
    if f.skin_solver_color:
        # Replay an accepted color-dependent normal recipe in the same candidate,
        # so accepting repaired color does not leave inferred shading stale.
        entry=map_provenance.load(context.scene)['entries'].get(map_provenance.identity(context.scene,'FINAL_'+role),{})
        prefix=s.character_name+('_Body' if role=='BODY' and s.part=='FULL_BODY' else '')
        artifact=entry.get('artifacts',{}).get(prefix+'_NORMAL.png',{})
        if artifact.get('dependencies') and artifact.get('skin_recipe'):refresh=True
    with_normal=bool(refresh or f.skin_solver_relief or f.skin_solver_surface_repair or balance)
    if not f.skin_solver_color and not with_normal:
        raise ValueError('Enable color cleanup or a normal repair operation')
    if ((f.skin_solver_color and f.skin_solver_method=='REFERENCE' and f.skin_solver_floor) or f.skin_solver_relief) and not f.skin_solver_mask:
        raise ValueError('Choose an explicit skin repair mask for forced cleanup or normal changes')
    if f.skin_solver_surface_repair and not f.skin_solver_surface_mask:
        raise ValueError('Choose a separate damaged-normal mask to retain sculpt detail elsewhere')
    if min(color.size) < 64 or max(color.size) > 8192:
        raise ValueError('Reconstruction supports delivered skin maps through 8K')
    source=f.skin_solver_source if f.skin_solver_color and f.skin_solver_source and not seam_repair and not current_material and not local_fill else color
    if source!=color and (source.source=='TILED' or source.colorspace_settings.name!='sRGB' or
        tuple(source.size)!=tuple(color.size) or source.get('cm_native_uv_layout')!=skin_projection.uv_layout_signature(obj)):
        raise ValueError('Use an original projection with this character UV layout, color space and map size')
    prefix = s.character_name+('_Body' if role=='BODY' and s.part=='FULL_BODY' else '')
    accepted=Path(bpy.path.abspath(s.project_dir)).resolve()/'Maps'/(prefix+'_ALBEDO.png')
    if not accepted.is_file() or Path(bpy.path.abspath(color.filepath)).resolve()!=accepted:
        raise ValueError('Transfer or save this skin into the character project before reconstruction')
    if color.is_dirty:
        if bpy.ops.cm.save_project()!={'FINISHED'}:
            raise ValueError('Save the current skin paint before reconstructing')
    material_input=retained_material_input(context.scene,role,accepted) if current_material else None
    # SR uses overlapping source tiles plus full output/weight/evidence buffers;
    # the heavier global correction intermediates are no longer atlas-sized.
    color_only_fill=local_fill and not with_normal and not material_strength and not bloodflow and not balance
    color_budget=color.size[0]*color.size[1]*output_scale**2*(64 if color_only_fill else 80 if diffusion_sr else 160 if local_fill or learned else 100 if f.skin_solver_color else 32)
    # The real 8K albedo workflow peaked at 9.8 GiB process commit. Retain an
    # extra GiB above the original model allowance rather than understate it.
    model_budget=5*resource_budget.GIB if diffusion_sr else 0
    surface_budget=2*resource_budget.GIB if balance else 0
    worker_memory=color_budget+512*resource_budget.MIB+model_budget+surface_budget
    resource_budget.require('skin reconstruction preparation' if defer else 'skin reconstruction',3*resource_budget.GIB if defer else worker_memory)
    normal=_normal(obj) if with_normal else None
    # Reuse only inside this synchronous, immutable preparation call. Never
    # retain a cache across edits, Apply, timers or separate operations.
    preparation_cache={}
    signature=_stamp(context,with_normal,cache=preparation_cache,bloodflow=bloodflow)
    root = Path(bpy.path.abspath(s.project_dir)).resolve()
    previous_normal={};normal_recipe=None
    if normal:
        expected=accepted.with_name(prefix+'_NORMAL.png')
        if Path(bpy.path.abspath(normal.filepath)).resolve()!=expected or max(normal.size)>8192:
            raise ValueError('Use the saved project tangent normal map, through 8K')
        entry=map_provenance.load(context.scene)['entries'].get(map_provenance.identity(context.scene,'FINAL_'+role),{})
        previous_normal=entry.get('artifacts',{}).get(prefix+'_NORMAL.png',{})
        normal_recipe=previous_normal.get('skin_recipe') or skin_normal_history.recover(root,_hash(expected),signature)
        if normal_recipe:
            normal_recipe=json.loads(json.dumps(normal_recipe))
            skin_normal_history.validate(root,normal_recipe,signature)
            if normal_recipe.get('previous_albedo_sha256') and not previous_normal.get('dependencies'):
                previous_normal=dict(previous_normal,dependencies={prefix+'_ALBEDO.png':normal_recipe['previous_albedo_sha256']},
                    skin_recipe=json.loads(json.dumps(normal_recipe)))
        if refresh and not normal_recipe:
            raise ValueError('No retained skin-normal recipe matches this map; retain or rebake its clean baseline first')
        problems=map_provenance.problems(context.scene,keys=['FINAL_'+role],suffixes=('_NORMAL.png',),
            ignore_dependencies=bool(normal_recipe),cache=preparation_cache)
        if problems:raise ValueError(problems[0])
        worker_memory=max(worker_memory,max(color_budget,normal.size[0]*normal.size[1]*96)+512*resource_budget.MIB+model_budget+surface_budget)
        if not defer:resource_budget.require('joint skin and normal reconstruction',worker_memory)
    reference_normal=None
    if f.skin_solver_surface_repair:
        if surface_reference is not None:
            if (surface_reference.colorspace_settings.name!='Non-Color' or
                tuple(surface_reference.size)!=tuple(normal.size) or
                surface_reference.get('cm_native_uv_layout')!=skin_projection.uv_layout_signature(obj)):
                raise ValueError('A surface repair reference needs matching UVs, resolution and non-color normal data')
            reference_normal=surface_reference
        else:
            template=f.body_template if role=='BODY' and s.part=='FULL_BODY' else s.template
            if not template or skin_projection.uv_layout_signature(template)!=skin_projection.uv_layout_signature(obj):
                raise ValueError('Surface repair needs the matching template UV layout')
            reference_normal=_normal(template)
    root = Path(bpy.path.abspath(s.project_dir)).resolve()
    worker_disk=color.size[0]*color.size[1]*output_scale**2*32+(normal.size[0]*normal.size[1]*12 if normal else 0)+9*resource_budget.GIB
    resource_budget.require_disk('skin reconstruction',root,worker_disk)
    parent = root/'SkinSolves'
    parent.mkdir(parents=True,exist_ok=True)
    if parent.resolve() != parent:
        raise ValueError('Skin solves must remain in the character project')
    folder = parent/uuid.uuid4().hex[:12]
    folder.mkdir()
    material_baselines={}
    initialized_materials=[]
    def material_path(kind):
        path=root/'Maps'/(prefix+'_'+kind+'.png')
        if kind=='ROUGHNESS' and not path.is_file():
            path=folder/'initial-roughness.png'
            if not path.is_file():
                image=bpy.data.images.new('CM_Initial_Roughness',width=color.size[0],height=color.size[1],alpha=False)
                try:
                    image.colorspace_settings.name='Non-Color'
                    value=f.skin_finish_roughness
                    image.generated_color=(value,value,value,1)
                    image.filepath_raw=str(path);image.file_format='PNG';image.save()
                finally:bpy.data.images.remove(image)
                initialized_materials.append(prefix+'_ROUGHNESS.png')
        return path
    if balance:
        last=_folder(context.scene,f.skin_solver_last) if f.skin_solver_last else None
        previous=json.loads((last/'state.json').read_text()) if last else None
        seen=set()
        while previous and previous['stamp']['role']!=role:
            pointer=previous.get('previous_solver_last')
            if not pointer or pointer in seen:previous=None;break
            seen.add(pointer);last=_folder(context.scene,pointer)
            previous=json.loads((last/'state.json').read_text())
        if previous and (previous.get('status')!='applied' or previous['stamp']['geometry']!=signature['geometry'] or previous['stamp']['uv']!=signature['uv']):
            raise ValueError('Retain the current material before rebalancing changed geometry or UVs')
        for kind in ('NORMAL','ROUGHNESS','CAVITY'):
            if previous and previous.get('material_baselines'):
                record=previous['material_baselines'][kind]
                path=skin_normal_history.resolve(root,record)
            elif previous and json.loads((last/'request.json').read_text()).get('method')=='GENERATIVE':
                name=prefix+'_'+kind+'.png'
                if _hash(root/'Maps'/name)!=previous['outputs'].get(name):raise ValueError('Retain edited material maps before redistributing their generated detail')
                path=last/'publish/backup'/name
                expected=(previous.get('previous_normal',{}).get('sha256') if kind=='NORMAL' else
                          previous.get('previous_cavity',{}).get('sha256') if kind=='CAVITY' else
                          previous['stamp'].get('derived_material',{}).get('baselines',{}).get('ROUGHNESS'))
                if not expected or _hash(path)!=expected:raise ValueError('The original material baseline is unavailable: '+kind)
            else:path=material_path(kind)
            if path.parent==root/'Maps':
                # Publication replaces accepted Maps. Original material inputs
                # must remain immutable for the next regional redistribution.
                retained=folder/('baseline-'+kind.lower()+'.png')
                shutil.copy2(path,retained);path=retained
            material_baselines[kind]=skin_normal_history.reference(root,path)
        normal_recipe={'schema':1,'baseline':material_baselines['NORMAL'],
            'stamp':{key:signature[key] for key in ('role','geometry','uv','sculpt')},
            'steps':[{'kind':'HEIGHT','field':{'file':(folder/'material-height.npy').relative_to(root).as_posix(),'sha256':None},'quantize_after':True,
                      'cavity_response':{'method':'SHARED_HEIGHT','strength':material_strength}}]}
    if f.skin_solver_color:retain_projection(context,obj,source)
    inputs = {}
    uv_info={}
    if learned or seam_repair or generative or current_material or local_fill:
        # Rasterize actual UV triangles. Empty atlas/background is never treated
        # as skin merely because its padding happens to contain a color.
        from . import skin_uv
        uv_info=skin_uv.write(obj,folder/'valid.npz',context,
            protect_anatomy=bool(role=='BODY' and generative and f.skin_solver_preserve_body_anatomy))
        inputs['valid']={'file':'valid.npz','sha256':_hash(folder/'valid.npz')}
    snapshots={}
    flow_masks=tuple(('bloodflow_'+slot,getattr(f,'wm'+slot[-1]+'_mask')) for slot in bloodflow)
    for key, image in (('albedo', source),
                       ('accepted',color if f.skin_solver_color else None),
                       ('protect',paint_protection(context.scene,role) if f.skin_solver_color else None),
                       ('reference', reference if f.skin_solver_color and f.skin_solver_method in {'REFERENCE','GENERATIVE'} else None),
                       ('mask', f.skin_solver_mask if f.skin_solver_color or f.skin_solver_relief else None),
                       ('normal_mask',f.skin_solver_surface_mask if f.skin_solver_surface_repair else None),*flow_masks):
        if image:
            if key=='albedo' and material_input:
                path,sha=material_input;shutil.copy2(path,folder/'albedo.png')
                if _hash(folder/'albedo.png')!=sha:raise ValueError('Retained material input changed while preparing the solve')
                inputs[key]=dict(file='albedo.png',sha256=sha)
                continue
            pointer=image.as_pointer()
            if pointer in snapshots:
                inputs[key]=dict(snapshots[pointer])
                continue
            filename = key+'.png'
            skin_library.save_image_copy(image, folder/filename)
            inputs[key] = {'file':filename,'sha256':_hash(folder/filename)}
            snapshots[pointer]=inputs[key]
    if material_strength:
        if not f.skin_solver_color:raise ValueError('Material derivation needs an albedo operation')
        for kind in ('ROUGHNESS','CAVITY'):
            path=skin_normal_history.resolve(root,material_baselines[kind]) if material_baselines else material_path(kind)
            if not path.is_file():raise ValueError('Retain a source '+kind+' map before deriving skin material')
            name='material_'+kind.lower()+'.png';shutil.copy2(path,folder/name)
            inputs['material_'+kind.lower()]={'file':name,'sha256':_hash(folder/name)}
    if balance:
        for kind in ('NORMAL','ROUGHNESS','CAVITY'):
            source_path=root/'Maps'/(prefix+'_'+kind+'.png')
            if not source_path.is_file() and kind=='ROUGHNESS':source_path=material_path(kind)
            name='region_'+kind.lower()+'.png'
            shutil.copy2(source_path,folder/name)
            inputs['region_'+kind.lower()]={'file':name,'sha256':_hash(folder/name)}
    if normal:
        # A new regional repair edits the currently accepted surface. Seal it
        # as the baseline so earlier repairs/relief remain exact outside this
        # brush, rather than replacing an earlier relief step or replaying it
        # against newly painted color. Previous provenance remains in state for
        # Revert; explicit refresh without a repair still replays its recipe.
        regional_repair=bool(f.skin_solver_surface_repair and not balance)
        baseline=(Path(bpy.path.abspath(normal.filepath)) if regional_repair or not normal_recipe
                  else skin_normal_history.resolve(root,normal_recipe['baseline']))
        shutil.copy2(baseline,folder/'normal.png')
        inputs['normal']={'file':'normal.png','sha256':_hash(folder/'normal.png')}
        if normal_recipe and not f.skin_solver_surface_repair and not f.skin_solver_relief:
            shutil.copy2(bpy.path.abspath(normal.filepath),folder/'retained-normal.png')
            inputs['retained_normal']={'file':'retained-normal.png','sha256':_hash(folder/'retained-normal.png')}
    if reference_normal:
        skin_library.save_image_copy(reference_normal,folder/'reference_normal.png')
        inputs['reference_normal']={'file':'reference_normal.png','sha256':_hash(folder/'reference_normal.png')}
    normal_steps=[]
    if normal:
        if normal_recipe is None or regional_repair:
            normal_recipe={'schema':1,'baseline':skin_normal_history.reference(root,folder/'normal.png'),
                'steps':[],'stamp':{key:signature[key] for key in ('role','geometry','uv','sculpt')}}
        if upscale:
            for step in normal_recipe['steps']:
                if step['kind']=='RELIEF':step['radius']*=output_scale
        if f.skin_solver_surface_repair and not balance:
            normal_recipe['steps'].append({'kind':'REPAIR','strength':f.skin_solver_strength,
                'mask':skin_normal_history.reference(root,folder/inputs['normal_mask']['file']),
                'reference':skin_normal_history.reference(root,folder/inputs['reference_normal']['file'])})
        if f.skin_solver_relief and not balance:
            step={'kind':'RELIEF','strength':f.skin_solver_relief,'radius':f.skin_solver_relief_radius*output_scale,
                'mask':skin_normal_history.reference(root,folder/inputs['mask']['file'])}
            positions=[i for i,old in enumerate(normal_recipe['steps']) if old['kind']=='RELIEF' and 'color' not in old]
            if positions:
                step['quantize_after']=normal_recipe['steps'][positions[0]].get('quantize_after',False)
                normal_recipe['steps'][positions[0]]=step
                normal_recipe['steps']=[old for i,old in enumerate(normal_recipe['steps']) if i not in positions[1:]]
            else:normal_recipe['steps'].append(step)
        normal_recipe.pop('previous_albedo_sha256',None)
        if normal_recipe['steps']:normal_recipe['steps'][-1]['quantize_after']=True
        for index,step in enumerate(normal_recipe['steps']):
            fields=('mask','reference','field','color','before','after','cavity_before','cavity_after')
            local={key:value for key,value in step.items() if key not in fields}
            for field in fields:
                if field not in step:continue
                if field=='field' and step[field].get('sha256') is None:
                    local[field]='generated_height';continue
                name=f'step_{index}_{field}'
                source_path=skin_normal_history.resolve(root,step[field])
                destination=folder/(name+source_path.suffix)
                shutil.copy2(source_path,destination)
                inputs[name]={'file':destination.name,'sha256':_hash(destination)}
                local[field]=name
            normal_steps.append(local)
    request = {'schema':1,'role':role,'parent':None if defer else os.getpid(),'strength':f.skin_solver_strength,'inputs':inputs,
        'cleanup_floor':f.skin_solver_floor,'relief_strength':f.skin_solver_relief,
        'relief_radius':f.skin_solver_relief_radius,'surface_repair':f.skin_solver_surface_repair,
        'color_cleanup':f.skin_solver_color,'method':f.skin_solver_method,'normal_steps':normal_steps,'output_scale':output_scale}
    request['bloodflow']={'slots':bloodflow,'strength':f.bloodflow_strength}
    request['material_strength']=material_strength
    if local_fill:
        request['fill_radius']=f.skin_solver_fill_radius;request['fill_method']=f.skin_solver_fill_method
    if current_material:request['material_albedo_basis']='RETAINED_INPUT' if material_input else 'CURRENT_COLOR'
    request['surface_balance']=f.skin_solver_surface_balance if balance else 0
    request['preserve_material_region']=bool(balance)
    if uv_info.get('anatomy'):request['anatomy_protection']=uv_info['anatomy']
    if learned:
        request['learned']={'backend':str(backend),'expected':backend_hashes,
            'gpu':f.skin_solver_gpu,'invention':f.skin_solver_invention,
            'prior':f.skin_solver_prior,'brief':f.skin_solver_brief,
            'projection':f.skin_solver_projection,'preserve_structure':f.skin_solver_preserve_structure}
        request['learned']['patterns']=f.skin_solver_patterns
    (folder/'request.json').write_text(json.dumps(request,indent=2),encoding='utf-8')
    state = {'schema':1,'stamp':signature,'prefix':prefix,'project':str(root),'status':'running',
        'normal':with_normal,'color':f.skin_solver_color,'normal_recipe':normal_recipe,
        'previous_normal':previous_normal,'cooperative_cancel':True,'signature_version':4}
    entries=map_provenance.load(context.scene)['entries']
    state['bloodflow']=bloodflow
    state['derived_material']=bool(material_strength)
    state['preserve_material_region']=bool(balance)
    state['previous_solver_last']=f.skin_solver_last
    state['initialized_materials']=initialized_materials
    if material_baselines:state['material_baselines']=material_baselines
    state['previous_roughness']=map_provenance.load(context.scene)['entries'].get(map_provenance.identity(context.scene,'ROUGHNESS_'+role))
    state['previous_cavity']=map_provenance.load(context.scene)['entries'].get(map_provenance.identity(context.scene,'FINAL_'+role),{}).get('artifacts',{}).get(prefix+'_CAVITY.png',{})
    state['previous_bloodflow']={slot:entries.get(map_provenance.identity(context.scene,slot)) for slot in bloodflow}
    (folder/'state.json').write_text(json.dumps(state,indent=2),encoding='utf-8')
    if defer:
        from . import skin_solve_staged
        skin_solve_staged.prepare(folder,worker_python,worker_memory,worker_disk)
        state.update(staged=True,status='prepared')
        (folder/'state.json').write_text(json.dumps(state,indent=2),encoding='utf-8')
        _JOB=dict(scene=context.scene,folder=folder,state=state,ready=False)
        f.skin_solver_job=str(folder)
        f.skin_solver_status='Skin solve staged. Save this project and close its authoring Blender before dispatching the original staged worker.'
        return folder
    environment = os.environ.copy()
    environment.update(OPENBLAS_NUM_THREADS='2', OMP_NUM_THREADS='2')
    with (folder/'worker.log').open('w',encoding='utf-8') as log:
        process = subprocess.Popen([worker_python,str(Path(__file__).with_name('skin_reconstruction_worker.py')),str(folder/'request.json')],
            stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,
            creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0),env=environment)
    _JOB = {'scene':context.scene,'folder':folder,'process':process,'state':state,'ready':False}
    f.skin_solver_job = str(folder)
    f.skin_solver_status = 'Reconstructing skin locally. The accepted maps are unchanged.'
    bpy.app.timers.register(poll,first_interval=.5)


def poll():
    job = _JOB
    if not job:
        return None
    if job.get('state',{}).get('staged'):
        from . import skin_solve_staged
        folder=job['folder'];plan=json.loads((folder/'staged.json').read_text())
        if skin_solve_staged.digest(folder/'request.json')!=plan['request_sha256']:
            job['scene'].cm_features.skin_solver_status='Staged request changed. Inspect the original files; no worker was resubmitted.'
            return None
        if (folder/'exit.json').is_file():
            result=json.loads((folder/'result.json').read_text())
            if result.get('staged_request_sha256')!=plan['request_sha256']:
                job['scene'].cm_features.skin_solver_status='The staged result belongs to another request.'
                return None
            _read_result(job);return None
        active=skin_solve_staged.alive(folder)
        if not (folder/'dispatch.json').exists():
            job['scene'].cm_features.skin_solver_status='Skin solve prepared; its original staged worker has not been dispatched.'
            return 1.
        if active is False:
            job['scene'].cm_features.skin_solver_status='The staged worker exited before confirming a candidate. Inspect its original result/cancel files; no new solve was submitted.'
            return None
    if job.get('state',{}).get('staged') or job['process'].poll() is None:
        try:
            progress = json.loads((job['folder']/'progress.json').read_text())
            job['scene'].cm_features.skin_solver_status = f"{progress.get('phase','Reconstructing skin')}: {progress['tiles_done']}/{progress['tiles_total']}"
        except (OSError, ValueError, KeyError):
            pass
        return .5
    _read_result(job)
    return None


def _read_result(job):
    try:
        result = json.loads((job['folder']/'result.json').read_text())
        if result.get('status') != 'ready':
            raise ValueError(result.get('error','The solver did not return a candidate'))
        if 'candidate-albedo.png' not in result.get('outputs',{}):
            raise ValueError('The solver did not return its color map')
        if job['state'].get('normal') and 'candidate-normal.png' not in result.get('outputs',{}):
            raise ValueError('The solver did not return its normal map')
        for name, sha in result['outputs'].items():
            if Path(name).name != name or _hash(job['folder']/name) != sha:
                raise ValueError('Solver output changed or is incomplete')
        for step in job['state'].get('normal_recipe',{}).get('steps',[]) if job['state'].get('normal_recipe') else []:
            if step['kind']=='HEIGHT' and step['field'].get('sha256') is None:
                step['field']['sha256']=result['outputs']['material-height.npy']
                (job['folder']/'state.json').write_text(json.dumps(job['state'],indent=2),encoding='utf-8')
        if job['state'].get('preserve_material_region'):
            # The finished map includes accepted patches and repairs outside
            # this edit. Seal that exact result as the next replay baseline,
            # avoiding repeated copies/replay of every previous height field.
            recipe=job['state']['normal_recipe']
            if not recipe.get('neutral_baseline'):
                # Replay seals a regional result, but native expression deltas
                # still need the untouched neutral that preceded this material.
                recipe['neutral_baseline']=job['state']['material_baselines']['NORMAL']
            if recipe['steps'][0]['kind']=='HEIGHT':job['state']['surface_field']=recipe['steps'][0]['field']
            root=job['folder'].parent.parent
            recipe['baseline']=skin_normal_history.reference(root,job['folder']/'candidate-normal.png',result['outputs']['candidate-normal.png'])
            recipe['steps']=[{'kind':'RETAIN'}]
            (job['folder']/'state.json').write_text(json.dumps(job['state'],indent=2),encoding='utf-8')
        job.update(ready=True,result=result)
        job['scene'].cm_features.skin_solver_status = (
            'Skin candidate ready. Inspect the color and evidence maps, then Apply or Discard.'
            if job['state'].get('color',True) else
            'Shading candidate ready. Inspect Normal, then Apply or Discard. The color map is retained.')
    except (OSError, ValueError, KeyError) as exc:
        job['scene'].cm_features.skin_solver_status = 'Skin solver stopped: '+str(exc)+'. Discard to retry.'


def apply(context):
    global _JOB
    from . import bake_jobs
    restore_preview()
    job = _JOB
    if not job or job['scene'] != context.scene or not job['ready']:
        raise ValueError('Wait for the skin candidate in this character')
    if job['result'].get('new_seam_guard_passed') is False:
        raise ValueError('The candidate introduces UV seam discontinuities. Discard and reduce the repair region or invention')
    if _stamp(context,job['state'].get('normal',False),legacy=job['state'].get('signature_version',1)<2,
              legacy_color_scope=job['state'].get('signature_version',1)<3,
              legacy_resolution=job['state'].get('signature_version',1)<4,
              bloodflow=job['state'].get('bloodflow',())) != job['state']['stamp']:
        raise ValueError('Character, skin or reference changed. Discard and reconstruct again')
    folder = _folder(context.scene,str(job['folder']))
    filename = job['state']['prefix']+'_ALBEDO.png'
    role=_images(context)[0]
    expected=context.scene.cm_settings.character_name+('_Body' if role=='BODY' and context.scene.cm_settings.part=='FULL_BODY' else '')+'_ALBEDO.png'
    if filename!=expected or Path(filename).name!=filename:
        raise ValueError('The candidate targets another skin map')
    source = folder/'candidate-albedo.png'
    if _hash(source) != job['result']['outputs']['candidate-albedo.png']:
        raise ValueError('The skin candidate changed after reconstruction')
    publication = folder/'publish'
    (publication/'staging/Maps').mkdir(parents=True,exist_ok=False)
    files,kinds=[],[]
    if job['state'].get('color',True):
        shutil.copy2(source,publication/'staging/Maps'/filename)
        files.append({'name':filename,'sha256':_hash(source)})
        kinds.append('ALBEDO')
    if job['state'].get('normal'):
        source=folder/'candidate-normal.png'
        if _hash(source)!=job['result']['outputs'].get(source.name):
            raise ValueError('The normal candidate changed after reconstruction')
        filename=job['state']['prefix']+'_NORMAL.png'
        shutil.copy2(source,publication/'staging/Maps'/filename)
        files.append({'name':filename,'sha256':_hash(source)})
        kinds.append('NORMAL')
    for slot in job['state'].get('bloodflow',()):
        source=folder/('candidate-'+slot+'.png')
        if _hash(source)!=job['result']['outputs'].get(source.name):
            raise ValueError('The blood-flow candidate changed after reconstruction')
        filename=job['state']['prefix']+'_'+slot+'.png'
        shutil.copy2(source,publication/'staging/Maps'/filename)
        files.append({'name':filename,'sha256':_hash(source)})
        kinds.append(slot)
    if job['state'].get('derived_material'):
        for kind in ('ROUGHNESS','CAVITY'):
            source=folder/('candidate-'+kind.lower()+'.png')
            if _hash(source)!=job['result']['outputs'].get(source.name):raise ValueError('The derived material changed after preparation')
            filename=job['state']['prefix']+'_'+kind+'.png'
            shutil.copy2(source,publication/'staging/Maps'/filename)
            files.append({'name':filename,'sha256':_hash(source)});kinds.append(kind)
    scene = context.scene
    nodes = [(n,n.image) for obj in (scene.cm_settings.result,scene.cm_settings.detail,scene.cm_features.body_result,scene.cm_features.body_detail) if obj
             for mat in obj.data.materials if mat and mat.use_nodes for n in mat.node_tree.nodes if n.type=='TEX_IMAGE']
    try:
        def finalize():
            bake_jobs.refresh_maps(scene,folder.parent.parent/'Maps',files,kinds=tuple(kinds))
            skin_bloodflow.record(scene,job['state'].get('bloodflow',()))
            if job['state'].get('derived_material'):
                dependencies={job['state']['prefix']+'_ALBEDO.png':_hash(folder.parent.parent/'Maps'/(job['state']['prefix']+'_ALBEDO.png'))}
                key='ROUGHNESS_'+role;name=job['state']['prefix']+'_ROUGHNESS.png'
                map_provenance.record(scene,{key:map_provenance.signature(scene,key)},recipe='ALBEDO_MATERIAL',
                    outputs={name},details={name:{'dependencies':dependencies}})
                key='FINAL_'+role;name=job['state']['prefix']+'_CAVITY.png'
                map_provenance.record(scene,{key:map_provenance.signature(scene,key)},origin='CUSTOM',
                    outputs={name},details={name:{'dependencies':dependencies,'material_source':job['result'].get('derived_material')}})
            if 'NORMAL' in kinds:
                key='FINAL_'+role
                recipe=job['state'].get('normal_recipe')
                detail={'skin_recipe':recipe} if recipe else {}
                if recipe and any(step['kind']=='RELIEF' and 'color' not in step for step in recipe['steps']):
                    detail['dependencies']={job['state']['prefix']+'_ALBEDO.png':
                        _hash(folder.parent.parent/'Maps'/(job['state']['prefix']+'_ALBEDO.png'))}
                elif not recipe:
                    # A ready candidate saved by an older release can still be
                    # published; its exact job chain is recovered on Refresh.
                    request=json.loads((folder/'request.json').read_text(encoding='utf-8'))
                    detail=dict(job['state'].get('previous_normal',{}))
                    if request.get('relief_strength'):
                        detail['dependencies']={job['state']['prefix']+'_ALBEDO.png':
                            _hash(folder.parent.parent/'Maps'/(job['state']['prefix']+'_ALBEDO.png'))}
                map_provenance.record(scene,{key:map_provenance.signature(scene,key)},origin='CUSTOM',
                    outputs={job['state']['prefix']+'_NORMAL.png'},details={job['state']['prefix']+'_NORMAL.png':detail})
            initialized_roughness_nodes(scene,job['state'])
        bake_jobs.deliver({'folder':publication,'destination':folder.parent.parent/'Maps'},files,
            finalize=finalize)
    except BaseException:
        initialized_roughness_nodes(scene,job['state'],remove=True)
        for node, image in nodes:
            node.image = image
        raise
    job['state'].update(status='applied',outputs={v['name']:v['sha256'] for v in files})
    job['state']['published_image']=map_provenance.image_digest(_images(context)[2],{})
    (folder/'state.json').write_text(json.dumps(job['state'],indent=2),encoding='utf-8')
    scene.cm_features.skin_solver_last = str(folder)
    scene.cm_features.skin_solver_job = ''
    scene.cm_features.skin_solver_status = 'Skin maps applied. Inspect in 3D; Revert retains the prior maps. Update Unreal Textures when ready.'
    scene.cm_settings.status = scene.cm_features.skin_solver_status
    _JOB = None


def revert(context):
    from . import bake_jobs, character_project
    s, f = context.scene.cm_settings, context.scene.cm_features
    character_project.preflight(context,s,f)
    folder = _folder(context.scene,f.skin_solver_last)
    state = json.loads((folder/'state.json').read_text())
    if state.get('status')!='applied' or _geometry(context)!=state['stamp']['geometry']:
        raise ValueError('The character changed since this solve; retain its maps and history')
    if state['stamp'].get('sculpt') and map_provenance.signature(context.scene,'FINAL_'+state['stamp']['role'],include_retained=False)!=state['stamp']['sculpt']:
        raise ValueError('The sculpt changed since this solve; retain its maps and history')
    role,obj=paint_session.part(context)
    if role!=state['stamp']['role']:
        raise ValueError('Choose the skin part used by the last solve before reverting')
    if json.loads(json.dumps(map_provenance.image_digest(paint_session.albedo(obj),{})))!=state['published_image']:
        raise ValueError('The skin has unsaved edits after Apply; retain the current paint')
    if state.get('normal'):_normal(obj)
    destination = folder.parent.parent/'Maps'
    for name, sha in state['outputs'].items():
        allowed=[state['prefix']+'_'+kind+'.png' for kind in ('ALBEDO','NORMAL',*state.get('bloodflow',()),*(('ROUGHNESS','CAVITY') if state.get('derived_material') else ()))]
        if Path(name).name!=name or name not in allowed:
            raise ValueError('Invalid skin reconstruction map history')
        if _hash(destination/name)!=sha:
            raise ValueError('The skin was edited after Apply; retain the current paint')
    publication = folder/'revert'
    (publication/'staging/Maps').mkdir(parents=True,exist_ok=False)
    files = []
    for name in state['outputs']:
        backup=folder/'publish/backup'/name
        if backup.is_file():
            shutil.copy2(backup,publication/'staging/Maps'/name)
            files.append({'name':name,'sha256':_hash(publication/'staging/Maps'/name)})
        elif name in {state['prefix']+'_'+slot+'.png' for slot in state.get('bloodflow',())} or name in state.get('initialized_materials',[]):
            files.append({'name':name,'delete':True})
        else:raise ValueError('The prior skin map backup is missing')
    nodes=[(n,n.image) for obj in (s.result,s.detail,f.body_result,f.body_detail) if obj
           for mat in obj.data.materials if mat and mat.use_nodes for n in mat.node_tree.nodes if n.type=='TEX_IMAGE']
    try:
        def finalize():
            bake_jobs.refresh_maps(context.scene,destination,files,kinds=('ALBEDO','NORMAL',*state.get('bloodflow',()),'ROUGHNESS','CAVITY'))
            skin_bloodflow.record(context.scene,(),previous=state.get('previous_bloodflow',{}))
            if state.get('derived_material'):
                skin_bloodflow.record(context.scene,(),previous={'ROUGHNESS_'+state['stamp']['role']:state.get('previous_roughness')})
                key='FINAL_'+state['stamp']['role'];name=state['prefix']+'_CAVITY.png'
                map_provenance.record(context.scene,{key:map_provenance.signature(context.scene,key)},origin='CUSTOM',
                    outputs={name},details={name:state.get('previous_cavity',{})})
            if state.get('normal'):
                key='FINAL_'+state['stamp']['role']
                map_provenance.record(context.scene,{key:map_provenance.signature(context.scene,key)},
                    origin='CUSTOM',outputs={state['prefix']+'_NORMAL.png'},
                    details={state['prefix']+'_NORMAL.png':state.get('previous_normal',{})})
            initialized_roughness_nodes(context.scene,state,remove=True)
        bake_jobs.deliver({'folder':publication,'destination':destination},files,
            finalize=finalize)
    except BaseException:
        for node,image in nodes:node.image=image
        raise
    state['status']='reverted'
    (folder/'state.json').write_text(json.dumps(state,indent=2),encoding='utf-8')
    f.skin_solver_last=''
    prior=state.get('previous_solver_last')
    if prior:
        try:
            prior_folder=_folder(context.scene,prior)
            previous=json.loads((prior_folder/'state.json').read_text())
            if previous.get('status')=='applied' and all(Path(name).name==name and _hash(destination/name)==sha for name,sha in previous['outputs'].items()):
                f.skin_solver_last=prior
        except (OSError,ValueError,KeyError):pass
    s.status=f.skin_solver_status='Previous skin maps restored. Update Unreal Textures to restore the native copy too.'


def discard():
    global _JOB
    restore_preview()
    job, _JOB = _JOB, None
    scene = job['scene'] if job else bpy.context.scene
    if job and job.get('state',{}).get('staged'):(job['folder']/'cancel').touch()
    if job and job.get('process') and job['process'].poll() is None:
        if job.get('state',{}).get('cooperative_cancel'):
            (job['folder']/'cancel').touch()
        else:job['process'].terminate()
    scene.cm_features.skin_solver_job=''
    scene.cm_features.skin_solver_status='Skin candidate discarded; accepted maps retained.'
    if job:
        def prune():
            if job.get('state',{}).get('staged'):
                from . import skin_solve_staged
                if skin_solve_staged.alive(job['folder']) is not False:return .2
            process=job.get('process')
            if process and process.poll() is None:return .2
            folder=_folder(scene,str(job['folder']))
            state=job.get('state',{})
            state['status']='discarded'
            (folder/'state.json').write_text(json.dumps(state,indent=2),encoding='utf-8')
            from . import skin_storage
            skin_storage.prune(folder.parent.parent,folder,worker_stopped=True)
            return None
        if prune() is not None:bpy.app.timers.register(prune,first_interval=.2)


@persistent
def detach(*_):
    global _JOB
    restore_preview()
    if bpy.app.timers.is_registered(poll):bpy.app.timers.unregister(poll)
    if _JOB and _JOB.get('process') and _JOB['process'].poll() is None:
        if _JOB.get('state',{}).get('cooperative_cancel'):
            (_JOB['folder']/'cancel').touch()
        else:_JOB['process'].terminate()
    _JOB=None


@persistent
def recover(*_):
    global _JOB
    scene=bpy.context.scene
    if not hasattr(scene,'cm_features') or not scene.cm_features.skin_solver_job:
        return
    try:
        folder=_folder(scene,scene.cm_features.skin_solver_job)
        state=json.loads((folder/'state.json').read_text())
        if state.get('status') in {'applied','reverted','discarded'}:
            raise ValueError('This saved candidate has already been '+state['status'])
        _JOB={'scene':scene,'folder':folder,'state':state,'ready':False}
        if state.get('staged'):
            if poll() is not None and not bpy.app.timers.is_registered(poll):bpy.app.timers.register(poll,first_interval=.5)
        else:_read_result(_JOB)
    except (OSError, ValueError) as exc:
        scene.cm_features.skin_solver_status='Skin candidate cannot resume: '+str(exc)


def save_staged(context):
    if not _JOB or not _JOB['state'].get('staged'):
        raise ValueError('Prepare a low-memory skin solve first')
    if bpy.ops.cm.save_project()!={'FINISHED'}:
        raise ValueError('The solve is prepared but recovery was not saved. Use Save for Recovery before closing Blender.')
    from . import skin_solve_staged
    folder=_JOB['folder'];project=Path(bpy.data.filepath).resolve()
    if not project.is_file():raise ValueError('The prepared project was not saved; keep Blender open and retry Save for Recovery')
    (folder/'recovery.json').write_text(json.dumps(dict(project=str(project),request_sha256=skin_solve_staged.digest(folder/'request.json')),indent=2),encoding='utf-8')
    context.scene.cm_features.skin_solver_status='Saved for recovery. Open Job Folder, close this Blender, then run run-skin-solve.cmd. Reopen project.blend to preview and apply.'


class CM_OT_reconstruct_skin(bpy.types.Operator):
    bl_idname='cm.reconstruct_skin'
    bl_label='Reconstruct Skin'
    defer:bpy.props.BoolProperty(name='Stage Without Launching',default=False,options={'HIDDEN'})
    action:bpy.props.EnumProperty(items=[(x,x.title(),'') for x in ('START','PREPARE','SAVE_STAGED','COPY_COMMAND','OPEN_JOB','SETUP_MATERIAL','PAINT_NORMALS','REFRESH_NORMALS','APPLY','DISCARD','REVERT','COLOR','NORMAL','EVIDENCE','PREVIEW','GENERATED_PREVIEW','ACCEPTED','CLEAR_CACHE','PAINT_COLOR')],default='START')

    def execute(self,context):
        try:
            if self.action=='START':start(context,defer=self.defer)
            elif self.action=='PREPARE':
                start(context,defer=True)
                save_staged(context)
            elif self.action=='SAVE_STAGED':save_staged(context)
            elif self.action in {'COPY_COMMAND','OPEN_JOB'}:
                if not _JOB:raise ValueError('There is no active skin job')
                folder=_JOB['folder']
                if self.action=='OPEN_JOB':bpy.ops.wm.path_open(filepath=str(folder))
                else:
                    if not _JOB['state'].get('staged'):raise ValueError('Only a prepared low-memory solve has a separate worker command')
                    from . import skin_solve_staged
                    context.window_manager.clipboard=skin_solve_staged.command(folder)
                    self.report({'INFO'},'Worker command copied for PowerShell' if os.name=='nt' else 'Worker command copied for your terminal')
            elif self.action=='SETUP_MATERIAL':
                if running():raise ValueError('Apply or discard the current candidate before preparing another material')
                f=context.scene.cm_features
                f.skin_solver_color=True;f.skin_solver_method='MATERIAL';f.skin_solver_output='KEEP'
                if not f.skin_solver_relief:f.skin_solver_relief=.2
                if not f.skin_solver_material_strength:f.skin_solver_material_strength=.5
                if not f.skin_solver_surface_balance:f.skin_solver_surface_balance=.65
                f.skin_solver_status='Material setup uses current albedo without model inference. Choose the repair/protection masks, adjust surface controls, then Reconstruct Skin Maps.'
            elif self.action=='PAINT_NORMALS':repair_painted(context)
            elif self.action=='PAINT_COLOR':complete_painted_color(context)
            elif self.action=='REFRESH_NORMALS':refresh_normals(context)
            elif self.action=='APPLY':apply(context)
            elif self.action=='DISCARD':discard()
            elif self.action=='REVERT':
                f=context.scene.cm_features;previous=f.texture_part
                try:
                    if context.scene.cm_settings.part=='FULL_BODY':
                        state=json.loads((_folder(context.scene,f.skin_solver_last)/'state.json').read_text())
                        f.texture_part=state['stamp']['role']
                    revert(context)
                except BaseException:
                    f.texture_part=previous;raise
            elif self.action=='PREVIEW':preview_color(context)
            elif self.action=='GENERATED_PREVIEW':preview_generated(context)
            elif self.action=='ACCEPTED':restore_preview()
            elif self.action=='CLEAR_CACHE':
                if running():raise ValueError('Finish or discard the skin candidate before clearing its model cache')
                from . import texture_prior_cache
                path=Path(bpy.path.abspath(context.scene.cm_settings.project_dir)).resolve()/'SkinSolves/.prior-cache'
                removed=texture_prior_cache.clear(path)
                context.scene.cm_features.skin_solver_status=f'Cleared {removed/1024**2:.1f} MiB of cached model previews. Accepted maps retained.'
            else:
                if not _JOB or not _JOB['ready']:raise ValueError('Wait for the skin candidate')
                name={'COLOR':'candidate-albedo.png','NORMAL':'candidate-normal.png','EVIDENCE':'artifact-evidence.png'}[self.action]
                if not (_JOB['folder']/name).is_file():raise ValueError('This candidate has no normal changes')
                image=bpy.data.images.load(str(_JOB['folder']/name),check_existing=True)
                image.colorspace_settings.name='sRGB' if self.action=='COLOR' else 'Non-Color'
                area=next((a for a in context.screen.areas if a.type=='IMAGE_EDITOR'),None)
                if area is None:area=next((a for a in context.screen.areas if a.type=='PROPERTIES'),None)
                if area is None:raise ValueError('Open an Image Editor to inspect the candidate; the accepted skin is unchanged')
                area.type='IMAGE_EDITOR'
                area.spaces.active.image=image
            return {'FINISHED'}
        except Exception as exc:
            self.report({'ERROR'},str(exc))
            return {'CANCELLED'}


class CM_MT_skin_storage(bpy.types.Menu):
    bl_label='Skin Job Storage'
    def draw(self,context):
        self.layout.operator('cm.skin_storage',text='Inspect Discarded Job Storage',icon='VIEWZOOM').clean=False
        self.layout.operator('cm.skin_storage',text='Clean Discarded Buffers',icon='TRASH').clean=True
        self.layout.separator()
        self.layout.label(text='Accepted/reverted jobs and job records are retained.')


class CM_OT_skin_storage(bpy.types.Operator):
    bl_idname='cm.skin_storage'
    bl_label='Skin Job Storage'
    bl_description='Inspect or remove working buffers from jobs already discarded; retain accepted/revert history and job records'
    clean:bpy.props.BoolProperty(default=False,options={'HIDDEN'})
    summary:bpy.props.StringProperty(options={'HIDDEN'})

    def project(self,context):
        value=context.scene.cm_settings.project_dir
        if not value.strip():raise ValueError('Choose the saved character project folder first')
        root=Path(bpy.path.abspath(value)).resolve()
        if not root.is_dir():raise ValueError('The character project folder is missing')
        return root

    def protected(self):
        paths=[bpy.path.abspath(image.filepath,library=image.library) for image in bpy.data.images if image.users and image.filepath]
        for scene in bpy.data.scenes:
            if not hasattr(scene,'cm_features'):continue
            f=scene.cm_features
            paths.extend(bpy.path.abspath(path) for path in (f.skin_solver_job,f.skin_solver_last) if path)
        return paths

    def invoke(self,context,event):
        if not self.clean:return self.execute(context)
        try:
            if running():raise ValueError('Finish or discard the active skin job first')
            from . import skin_storage
            report=skin_storage.inspect(self.project(context),self.protected())
            if not report['eligible_files']:return self.execute(context)
            self.summary=f"{report['eligible_files']} files, {report['eligible_bytes']/1024**2:.1f} MiB in {len(report['jobs'])} discarded jobs"
        except Exception as exc:
            self.report({'ERROR'},str(exc));return {'CANCELLED'}
        return context.window_manager.invoke_props_dialog(self,width=480)

    def draw(self,context):
        self.layout.label(text=self.summary)
        self.layout.label(text='Permanently remove discarded input/candidate images and scratch.')
        self.layout.label(text='Accepted/reverted jobs and request/result/log records are retained.')

    def execute(self,context):
        try:
            if running():raise ValueError('Finish or discard the active skin job first')
            from . import skin_storage
            root=self.project(context)
            report=skin_storage.clean(root,self.protected()) if self.clean else skin_storage.inspect(root,self.protected())
            folder=root/'SkinSolves';folder.mkdir(exist_ok=True)
            path=folder/('storage-cleanup-'+uuid.uuid4().hex[:12]+'.json' if self.clean else 'storage-inspection.json')
            path.write_text(json.dumps(report,indent=2),encoding='utf-8')
            size=report['removed_bytes'] if self.clean else report['eligible_bytes']
            context.scene.cm_features.skin_solver_status=(f'Removed {size/1024**2:.1f} MiB of discarded buffers.' if self.clean else
                f"{size/1024**2:.1f} MiB can be removed from {len(report['jobs'])} discarded jobs. Accepted/revert history is retained.")
            context.scene.cm_settings.status=context.scene.cm_features.skin_solver_status
            self.report({'INFO'},context.scene.cm_features.skin_solver_status)
            return {'FINISHED'}
        except Exception as exc:
            self.report({'ERROR'},str(exc));return {'CANCELLED'}


def draw(layout,context):
    f=context.scene.cm_features
    box=layout.box()
    box.label(text='Skin Reconstruction (Experimental)')
    if _JOB:
        from .guided import text
        text(box,f.skin_solver_status)
        if _JOB['state'].get('staged'):
            box.operator('cm.reconstruct_skin',text='Open Job Folder',icon='FILE_FOLDER').action='OPEN_JOB'
            if not (_JOB['folder']/'dispatch.json').exists() and not _JOB.get('ready'):
                box.operator('cm.reconstruct_skin',text='Save for Recovery',icon='FILE_TICK').action='SAVE_STAGED'
                box.operator('cm.reconstruct_skin',text='Copy Worker Command').action='COPY_COMMAND'
                text(box,'Close this Blender before running the prepared worker. On Windows, double-click run-skin-solve.cmd in the job folder; otherwise use the copied command. Reopen project.blend to recover the candidate.')
        if _JOB.get('ready'):
            color=_JOB['state'].get('color',True)
            if color:
                box.operator('cm.reconstruct_skin',text='Show Accepted Color' if _PREVIEW else 'Preview Color on Skin').action='ACCEPTED' if _PREVIEW else 'PREVIEW'
                row=box.row(align=True)
                row.operator('cm.reconstruct_skin',text='Inspect Color').action='COLOR'
                row.operator('cm.reconstruct_skin',text='Inspect Evidence').action='EVIDENCE'
            if _JOB['state'].get('normal'):box.operator('cm.reconstruct_skin',text='Inspect Normal').action='NORMAL'
            box.operator('cm.reconstruct_skin',text='Apply Skin Maps' if color else 'Apply Shading Repair').action='APPLY'
        box.operator('cm.reconstruct_skin',text='Discard').action='DISCARD'
    else:
        role,obj=paint_session.part(context)
        box.operator('cm.reconstruct_skin',text='Material from Current Albedo').action='SETUP_MATERIAL'
        box.operator('cm.reconstruct_skin',text='Refresh Skin Shading from Current Color').action='REFRESH_NORMALS'
        if obj:box.operator('cm.reference_paint',text='Paint Repair Region',icon='BRUSH_DATA').mask_only=True
        if obj and isinstance(obj.get('cm_last_paint_mask'),bpy.types.Image):
            box.operator('cm.reconstruct_skin',text='Complete Color in Painted Region').action='PAINT_COLOR'
            box.operator('cm.reconstruct_skin',text='Repair Shading in Painted Region').action='PAINT_NORMALS'
            box.label(text='Prepares normals; keeps the color map.')
        box.prop(f,'skin_solver_color')
        if f.skin_solver_color and role=='HEAD':
            box.prop(f,'bloodflow',text='Keep Expression Color Maps in Sync')
            if f.bloodflow:box.prop(f,'bloodflow_strength')
        if f.skin_solver_color:
            box.prop(f,'skin_solver_method',text='Method')
            if f.skin_solver_method not in {'SEAMS','MATERIAL','LOCAL_FILL'}:box.prop(f,'skin_solver_source')
            box.prop(f,'skin_solver_body_protect' if role=='BODY' else 'skin_solver_protect')
            if f.skin_solver_method=='SEAMS':box.label(text='Balances current color at paired UV edges.')
            if f.skin_solver_method=='LOCAL_FILL':
                box.prop(f,'skin_solver_fill_method')
                box.prop(f,'skin_solver_fill_radius',text='Seam Guard Radius' if f.skin_solver_fill_method=='SURFACE' else 'Color Fill Radius')
                box.label(text='Fills selected color from surrounding skin.')
                box.label(text='Protected paint and UV borders are retained.')
            if f.skin_solver_method=='MATERIAL':
                box.prop(f,'skin_solver_surface_balance',slider=True)
                box.label(text='Uses current albedo; no image generation or upscale.')
                box.label(text='Control-only tuning retains its original color input.')
            if f.skin_solver_method=='GENERATIVE':
                box.prop(f,'skin_solver_generated_body_albedo' if role=='BODY' else 'skin_solver_generated_albedo')
                if role=='BODY':box.prop(f,'skin_solver_preserve_body_anatomy')
                box.prop(f,'skin_solver_output')
                box.prop(f,'skin_solver_surface_balance',slider=True)
                box.label(text='Generated input must keep the exact UV layout.')
                row=box.row(align=True)
                row.operator('cm.reconstruct_skin',text='Preview Generated Color').action='GENERATED_PREVIEW'
                row.operator('cm.reconstruct_skin',text='Show Accepted Color').action='ACCEPTED'
            box.prop(f,'skin_solver_material_strength',slider=True)
            if f.skin_solver_method=='LEARNED':
                box.prop(f,'skin_solver_prior')
                if f.skin_solver_prior=='PISA':
                    box.prop(f,'skin_solver_sr_python')
                    box.prop(f,'skin_solver_brief')
                    box.prop(f,'skin_solver_output')
                    box.label(text='Adds bounded local diffusion detail over source color.')
                    box.label(text='Uses existing PiSA-SR models and CUDA; no downloads.')
                if f.skin_solver_prior=='EDIT':
                    box.prop(f,'skin_solver_brief')
                    box.prop(f,'skin_solver_projection')
                    if f.skin_solver_projection=='SURFACE':box.prop(f,'skin_solver_output')
                    box.prop(f,'skin_solver_preserve_structure')
                    box.prop(f,'skin_solver_patterns')
                    box.operator('cm.reconstruct_skin',text='Clear Cached Model Previews').action='CLEAR_CACHE'
                box.prop(f,'skin_solver_invention',slider=True)
                box.label(text='Inferred detail; inspect the candidate before Apply.')
                box.label(text='White mask repairs; accepted paint stays protected.')
        box.prop(f,'skin_solver_strength')
        if f.skin_solver_color or f.skin_solver_relief:box.prop(f,'skin_solver_mask')
        if f.skin_solver_color and f.skin_solver_method=='REFERENCE':box.prop(f,'skin_solver_floor')
        box.prop(f,'skin_solver_surface_repair')
        if f.skin_solver_surface_repair:box.prop(f,'skin_solver_surface_mask')
        box.prop(f,'skin_solver_relief',text='Surface Detail' if f.skin_solver_method in {'GENERATIVE','MATERIAL'} and f.skin_solver_surface_balance else 'Inferred Relief')
        if f.skin_solver_relief:box.prop(f,'skin_solver_relief_radius')
        run=box.operator('cm.reconstruct_skin',text='Reconstruct Skin Maps')
        run.action='START';run.defer=False
        box.operator('cm.reconstruct_skin',text='Prepare Low-Memory Solve').action='PREPARE'
        if f.skin_solver_last:box.operator('cm.reconstruct_skin',text='Revert Last Solve').action='REVERT'
        row=box.row(align=True)
        row.operator('cm.skin_storage',text='Inspect Discarded Storage',icon='VIEWZOOM').clean=False
        row.operator('cm.skin_storage',text='Clean Discarded Buffers',icon='TRASH').clean=True
        if f.skin_solver_status:
            from .guided import text
            text(box,f.skin_solver_status)


def register():
    bpy.utils.register_class(CM_OT_reconstruct_skin)
    bpy.utils.register_class(CM_OT_skin_storage)
    bpy.utils.register_class(CM_MT_skin_storage)
    for handlers,callback in ((bpy.app.handlers.load_pre,detach),(bpy.app.handlers.undo_pre,detach),
                              (bpy.app.handlers.load_post,recover),(bpy.app.handlers.undo_post,recover),
                              (bpy.app.handlers.save_pre,restore_preview)):
        if callback not in handlers:handlers.append(callback)


def unregister():
    detach()
    for handlers,callback in ((bpy.app.handlers.load_pre,detach),(bpy.app.handlers.undo_pre,detach),
                              (bpy.app.handlers.load_post,recover),(bpy.app.handlers.undo_post,recover),
                              (bpy.app.handlers.save_pre,restore_preview)):
        for existing in list(handlers):
            if getattr(existing,'__module__','')==__name__:handlers.remove(existing)
    registered=bpy.types.Operator.bl_rna_get_subclass_py('CM_OT_reconstruct_skin')
    if registered:bpy.utils.unregister_class(registered)
    registered=bpy.types.Operator.bl_rna_get_subclass_py('CM_OT_skin_storage')
    if registered:bpy.utils.unregister_class(registered)
    registered=bpy.types.Menu.bl_rna_get_subclass_py('CM_MT_skin_storage')
    if registered:bpy.utils.unregister_class(registered)
