"""Static local sculpt modifiers and their worker/recovery state guards."""

import hashlib
import json

import bpy
import numpy as np


# These stacks carry all of their editable data in the object/mesh. Modifiers
# with external targets, caches, node groups or textures need explicit capture.
LOCAL = {'SUBSURF', 'MULTIRES', 'SMOOTH', 'CORRECTIVE_SMOOTH', 'LAPLACIANSMOOTH',
         'DECIMATE', 'TRIANGULATE', 'WELD', 'SOLIDIFY', 'BEVEL', 'WEIGHTED_NORMAL'}
DEPENDENT = {'SHRINKWRAP','MIRROR','ARRAY','DISPLACE','SIMPLE_DEFORM','CAST','WARP',
             'HOOK','LATTICE','CURVE','DATA_TRANSFER','BOOLEAN','MESH_DEFORM','SURFACE_DEFORM'}


def validate(obj, role):
    for modifier in obj.modifiers:
        if modifier.type not in LOCAL | DEPENDENT:
            raise ValueError(role + ': this modifier needs dependency capture before return: ' + modifier.name)
        if modifier.type == 'MULTIRES' and modifier.is_external:
            raise ValueError(role + ': pack external Multires displacement before returning its editable sculpt')
        for prop in modifier.bl_rna.properties:
            if prop.identifier != 'rna_type' and prop.type == 'POINTER':
                value = getattr(modifier, prop.identifier, None)
                if isinstance(value,bpy.types.ID) and not isinstance(value,(bpy.types.Object,bpy.types.Texture,bpy.types.Collection)):
                    raise ValueError(role + ': modifier references uncaptured data: ' + modifier.name)


def settings(obj):
    """Include disabled/render-only settings, which evaluated meshes omit."""
    result = [(modifier.type, _settings(modifier)) for modifier in obj.modifiers]
    return json.dumps(result, sort_keys=True)


def dependencies_digest(context, obj):
    """Protect shaping assets even when their modifier is disabled or zeroed."""
    from . import mesh_ops as m,map_provenance
    digest=hashlib.sha256()
    visited={}
    def visit(value):
        if value in visited:
            digest.update(('REF'+str(visited[value])).encode())
            return
        visited[value]=len(visited)
        digest.update(json.dumps([value.bl_rna.identifier,value.name_full,
            value.library.filepath if value.library else None]).encode())
        if isinstance(value,bpy.types.Object):
            if value.type in {'MESH','CURVE','SURFACE'}:
                if value.type=='MESH':digest.update(map_provenance.mesh_digest(context,value).encode())
                else:digest.update(np.asarray(m.evaluated_coordinates(context,value),dtype='<f4').tobytes())
            digest.update(np.asarray(value.matrix_world,dtype='<f8').tobytes())
            if value.type=='LATTICE':
                digest.update(json.dumps([(list(point.co_deform),point.weight_softbody) for point in value.data.points]).encode())
            if value.type in {'CURVE','SURFACE'}:
                for spline in value.data.splines:
                    digest.update(json.dumps([spline.type,spline.order_u,spline.use_cyclic_u,
                        [(list(point.co),point.radius,point.tilt) for point in spline.points],
                        [(list(point.co),list(point.handle_left),list(point.handle_right),point.radius,point.tilt)
                         for point in spline.bezier_points]]).encode())
            for modifier in value.modifiers:references(modifier)
        elif isinstance(value,bpy.types.Texture):
            digest.update(json.dumps(_settings(value),sort_keys=True).encode())
            if value.type=='IMAGE' and value.image:visit(value.image)
        elif isinstance(value,bpy.types.Image):
            digest.update(json.dumps([list(value.size),value.colorspace_settings.name,value.alpha_mode]).encode())
            pixels=np.empty(len(value.pixels),dtype='<f4')
            value.pixels.foreach_get(pixels)
            digest.update(pixels.tobytes())
        elif isinstance(value,bpy.types.Collection):
            for child in sorted(value.objects,key=lambda item:item.name_full):visit(child)
            for child in sorted(value.children,key=lambda item:item.name_full):visit(child)
        elif isinstance(value,bpy.types.NodeTree):
            digest.update(json.dumps(sorted((link.from_node.name,link.from_socket.identifier,
                link.to_node.name,link.to_socket.identifier) for link in value.links)).encode())
            for node in value.nodes:
                digest.update(json.dumps([node.bl_idname,node.name,_settings(node)],sort_keys=True).encode())
                for socket in node.inputs:
                    item=getattr(socket,'default_value',None)
                    if isinstance(item,bpy.types.ID):visit(item)
                    elif isinstance(item,(int,float,str,bool)):
                        digest.update(json.dumps([socket.identifier,item]).encode())
                    elif item is not None:
                        try:digest.update(json.dumps([socket.identifier,list(item)]).encode())
                        except TypeError:pass
                if getattr(node,'node_tree',None):visit(node.node_tree)
                if getattr(node,'image',None):visit(node.image)
    def references(struct):
        for prop in struct.bl_rna.properties:
            if prop.identifier=='rna_type' or prop.type!='POINTER':continue
            value=getattr(struct,prop.identifier)
            if isinstance(value,bpy.types.ID):
                digest.update(prop.identifier.encode())
                visit(value)
        try:items=list(struct.items())
        except (AttributeError,TypeError):items=[]
        for name,value in sorted(items):
            if isinstance(value,bpy.types.ID):
                digest.update(name.encode())
                visit(value)
    for modifier in obj.modifiers:references(modifier)
    return digest.digest()


def _settings(struct):
    values = {}
    for prop in struct.bl_rna.properties:
        if prop.identifier in {'rna_type', 'show_expanded', 'is_active'}:
            continue
        if isinstance(struct,bpy.types.Node) and prop.identifier in {'select','location','width','height','hide',
            'show_options','show_preview','show_texture','use_custom_color','color','label'}:
            continue
        if prop.is_readonly and prop.identifier not in {'custom_profile','points','is_bound','is_bind',
            'falloff_curve','curves','color_ramp','elements','vertex_indices'}:
            continue
        value = getattr(struct, prop.identifier)
        if prop.type in {'BOOLEAN', 'INT', 'FLOAT', 'STRING', 'ENUM'}:
            values[prop.identifier] = list(value) if getattr(prop, 'is_array', False) else sorted(value) if isinstance(value, set) else value
        elif prop.type == 'POINTER':
            values[prop.identifier] = ([value.name_full, value.library.filepath if value.library else None] if isinstance(value, bpy.types.ID)
                else _settings(value) if value is not None else None)
        elif prop.type == 'COLLECTION':
            values[prop.identifier] = [_settings(item) for item in value]
    return values


def multires_digest(context, obj):
    """Detect sculpt edits hidden above the current viewport subdivision level.

    Probe the full stored displacement surface, without later modifiers hiding
    it. Restore viewport settings even when evaluation fails. No source geometry
    or selection changes; use only on explicit save/update/return actions.
    """
    from . import mesh_ops as m
    digest = hashlib.sha256()
    modifiers = list(obj.modifiers)
    for index, modifier in enumerate(modifiers):
        if modifier.type != 'MULTIRES':continue
        previous = [(item, item.show_viewport) for item in modifiers[index:]]
        level = modifier.levels
        try:
            modifier.levels = modifier.total_levels
            modifier.show_viewport = True
            for item in modifiers[index + 1:]:item.show_viewport = False
            context.view_layer.update()
            digest.update(np.asarray(m.evaluated_coordinates(context, obj), dtype='<f4').tobytes())
        finally:
            modifier.levels = level
            for item, enabled in previous:item.show_viewport = enabled
            context.view_layer.update()
    return digest.digest()
