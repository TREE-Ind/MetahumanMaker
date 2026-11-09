"""Native regional expression drivers applied to authored WM/CM preview maps."""
from pathlib import Path
import bpy
import numpy as np
from .animated_regions import REGIONS

ORIGINAL='cm_expression_original'
STRENGTH='cm_expression_strength'
WEIGHT='cm_expression_weights'
_MASKS={}


def pixels(image):
    values=np.empty(len(image.pixels),dtype=np.float32);image.pixels.foreach_get(values)
    return values.reshape(image.size[1],image.size[0],4)


def region_data(s):
    from .wrinkles import native_folder
    folder=native_folder(s)
    paths={name:folder/name for name,channel in REGIONS.values()}
    key=tuple((str(path),path.stat().st_mtime_ns,path.stat().st_size) for path in sorted(paths.values()))
    if key in _MASKS:return _MASKS[key]
    result={};shape=None
    for filename,path in paths.items():
        bits=path.read_bytes()[16]
        if bits not in (24,32) or bits==24 and any(channel==3 for name,channel in REGIONS.values() if name==filename):
            raise ValueError('Expression maps require the original DCC mask channels')
        image=bpy.data.images.load(str(path),check_existing=False)
        try:
            image.colorspace_settings.name='Non-Color';image.alpha_mode='CHANNEL_PACKED'
            values=pixels(image)
            if shape is not None and values.shape!=shape:raise ValueError('Native mask sizes do not match')
            shape=values.shape;result[filename]=values
        finally:bpy.data.images.remove(image)
    _MASKS.clear();_MASKS[key]=result
    return result


def weights(s,data,pose,available):
    names=data.get('animated_maps',[]);values=pose.get('animated_maps',[])
    if len(names)!=len(values) or not names:raise ValueError('Rebuild the native expression library to preview animated maps')
    if not np.isfinite(values).all() or min(values)<0 or max(values)>1.00001:
        raise ValueError('The native expression map drivers are invalid')
    masks=region_data(s)
    h,w=next(iter(masks.values())).shape[:2]
    output={kind:np.zeros((h,w,4),dtype=np.float32) for kind in ('NORMAL','COLOR')}
    for name,value in zip(names,values):
        family,separator,region=name.partition('.')
        if not separator or not family.startswith(('head_wm','head_cm')):continue
        slot=int(family[7])
        kind='NORMAL' if family.endswith('_normal') else 'COLOR'
        label=('WM' if kind=='NORMAL' else 'CM')+str(slot)
        if label not in available:continue
        if region not in REGIONS:raise ValueError('Unsupported native expression region: '+region)
        filename,channel=REGIONS[region]
        np.maximum(output[kind][...,slot-1],masks[filename][...,channel]*min(float(value),1),out=output[kind][...,slot-1])
    for value in output.values():
        value[...,:3]/=np.maximum(1,value[...,:3].sum(axis=2))[...,None]
        value[...,3]=1
    return output


def base_material(obj):
    if len(obj.data.materials)!=1 or not obj.data.materials[0]:return None
    material=obj.data.materials[0]
    return material.get(ORIGINAL) or material


def tagged_image(material,kind):
    nodes=[n for n in material.node_tree.nodes if n.type=='TEX_IMAGE' and n.image and
           (n.get('cm_map_kind')==kind or kind=='ALBEDO' and n.name=='CM_Albedo')]
    if len(nodes)==1:return nodes[0]
    if nodes:return None
    # Licensed templates can predate our map tags. Follow the actual shader
    # input rather than guessing from filenames or the order of image nodes.
    found={};visited=set()
    def upstream(socket):
        for link in socket.links:
            node=link.from_node
            if node.as_pointer() in visited:continue
            visited.add(node.as_pointer())
            if node.type=='TEX_IMAGE' and node.image:found[node.as_pointer()]=node
            else:
                for item in node.inputs:
                    if item.is_linked:upstream(item)
    for shader in material.node_tree.nodes:
        if shader.type=='BSDF_PRINCIPLED':upstream(shader.inputs['Base Color' if kind=='ALBEDO' else 'Normal'])
    return next(iter(found.values())) if len(found)==1 else None


def asset(path,normal):
    stamp=path.stat()
    for image in bpy.data.images:
        if image.get('cm_expression_asset')==str(path) and image.get('cm_expression_mtime')==str(stamp.st_mtime_ns) and image.get('cm_expression_bytes')==stamp.st_size:
            return image
    image=bpy.data.images.load(str(path),check_existing=False)
    image.colorspace_settings.name='Non-Color' if normal else 'sRGB'
    image.alpha_mode='NONE'
    image['cm_expression_asset']=str(path);image['cm_expression_mtime']=str(stamp.st_mtime_ns)
    image['cm_expression_bytes']=stamp.st_size;image.pack()
    return image


def discard(material):
    if not material or material.users or not material.get(ORIGINAL):return
    # Accepted IDs may still be referenced by Blender's memfile undo history.
    # Removing their node trees/images here can leave dangling restored links.
    # Keep these small previews until Blender discards unused data on reopen.
    if material.get('cm_expression_committed'):return
    images=[n.image for n in material.node_tree.nodes if n.type=='TEX_IMAGE' and n.image and n.image.get(WEIGHT)]
    bpy.data.materials.remove(material)
    for image in set(images):
        if not image.users:bpy.data.images.remove(image)


def restore(obj):
    if not obj:return
    for index,material in enumerate(list(obj.data.materials)):
        original=material.get(ORIGINAL) if material else None
        if isinstance(original,bpy.types.Material):
            material['cm_expression_committed']=True
            obj.data.materials[index]=original;discard(material)


def build(context,s,f,data,pose):
    """Build privately before touching geometry or accepted material bindings."""
    if not f.expression_skin:return None,'Animated skin preview off'
    original=base_material(s.result)
    if not original or not original.use_nodes:return None,'Use one skin material to preview animated maps'
    sources={kind:tagged_image(original,kind) for kind in ('ALBEDO','NORMAL')}
    if not any(sources.values()):return None,'Use image-based head skin to preview WM/CM maps'
    folder=Path(bpy.path.abspath(s.project_dir))/'Maps'
    files={kind:folder/(s.character_name+'_'+kind+'.png') for kind in ('WM1','WM2','WM3','CM1','CM2','CM3')}
    files={kind:path for kind,path in files.items() if path.is_file() and sources['NORMAL' if kind.startswith('WM') else 'ALBEDO']}
    if not files:return None,'Bake WM/CM maps to preview expression-driven skin'
    if not data.get('animated_maps') or not pose.get('animated_maps'):
        return None,'Update Native Expressions for skin drivers'
    from .wrinkles import native_folder
    if not all((native_folder(s)/name).is_file() for name,channel in REGIONS.values()):
        return None,'Install optional MetaHuman DCC masks for skin preview'
    fields=weights(s,data,pose,files)
    material=original.copy();material.name=s.character_name+'_ExpressionSkin';material[ORIGINAL]=original
    tree=material.node_tree;nodes,links=tree.nodes,tree.links
    try:
        strength=nodes.new('ShaderNodeValue');strength[STRENGTH]=True
        strength.outputs[0].default_value=f.expression_strength;strength.label='Expression strength'
        for kind,source_kind,prefix in (('NORMAL','NORMAL','WM'),('COLOR','ALBEDO','CM')):
            slots=[i for i in (1,2,3) if prefix+str(i) in files]
            if not slots:continue
            source=tagged_image(material,source_kind)
            consumers=[link.to_socket for link in list(source.outputs['Color'].links)]
            if not consumers:continue
            image=bpy.data.images.new('CM_Expression_'+kind,width=fields[kind].shape[1],height=fields[kind].shape[0],alpha=False)
            image.colorspace_settings.name='Non-Color';image[WEIGHT]=kind
            image.pixels.foreach_set(fields[kind].ravel());image.pack()
            texture=nodes.new('ShaderNodeTexImage');texture.image=image
            separate=nodes.new('ShaderNodeSeparateColor');links.new(texture.outputs['Color'],separate.inputs['Color'])
            current=source.outputs['Color']
            for slot in slots:
                label=prefix+str(slot)
                tex=nodes.new('ShaderNodeTexImage');tex.image=asset(files[label],kind=='NORMAL');tex.label=label
                tex['cm_expression_slot']=label
                delta=nodes.new('ShaderNodeVectorMath');delta.operation='SUBTRACT'
                links.new(tex.outputs['Color'],delta.inputs[0]);links.new(source.outputs['Color'],delta.inputs[1])
                factor=nodes.new('ShaderNodeMath');factor.operation='MULTIPLY'
                links.new(separate.outputs[slot-1],factor.inputs[0]);links.new(strength.outputs[0],factor.inputs[1])
                weighted=nodes.new('ShaderNodeVectorMath');weighted.operation='SCALE'
                links.new(delta.outputs['Vector'],weighted.inputs[0]);links.new(factor.outputs[0],weighted.inputs['Scale'])
                add=nodes.new('ShaderNodeVectorMath');add.operation='ADD'
                links.new(current,add.inputs[0]);links.new(weighted.outputs['Vector'],add.inputs[1]);current=add.outputs['Vector']
            for socket in consumers:links.new(current,socket)
        material['cm_expression_maps']=', '.join(files)
        return material,'Native skin: '+str(sum(k.startswith('WM') for k in files))+' WM + '+str(sum(k.startswith('CM') for k in files))+' CM'
    except Exception:
        discard(material)
        raise


def commit(obj,material):
    previous=obj.data.materials[0] if obj.data.materials else None
    if material:
        material['cm_expression_committed']=True
        if previous and previous.get(ORIGINAL):previous['cm_expression_committed']=True
        obj.data.materials[0]=material
        discard(previous)
    else:restore(obj)


def update_strength(obj,value):
    if not obj:return
    for material in obj.data.materials:
        if material and material.get(ORIGINAL):
            for node in material.node_tree.nodes:
                if node.get(STRENGTH):node.outputs[0].default_value=value


def toggle(f,context):
    if not context or not f.expression_active:return
    from . import expressions
    s=context.scene.cm_settings
    try:
        data=expressions.bank(f);pose=data['poses'][f.expression_index%len(data['poses'])]
        material,message=build(context,s,f,data,pose);commit(s.result,material)
        f.expression_skin_status=message
    except Exception as exc:
        restore(s.result);f.expression_skin_status=str(exc)
