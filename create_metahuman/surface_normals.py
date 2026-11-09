"""Dynamic smoothing across canonical UV splits without welding export meshes.

Helpers share original mesh data (including live shape keys), and weld only
their evaluated normal source. Data Transfer writes normals after shape keys;
assigning posed normals to the undeformed Basis would rotate them twice.
"""
import bpy
from mathutils import Matrix


def ensure(context,head,body,distance=1e-6):
    if head==body or any(o is None or o.type!='MESH' for o in (head,body)):
        raise ValueError('Choose separate canonical head and body meshes')
    if distance<=0:raise ValueError('Normal source weld distance must be positive')
    owner=head.name
    collection=bpy.data.collections.get('CM Surface Normal Helpers')
    if collection is None:
        collection=bpy.data.collections.new('CM Surface Normal Helpers');context.scene.collection.children.link(collection)
    helpers=[]
    for role,target in (('body',body),('connected',head)):
        helper=next((o for o in collection.objects if o.get('cm_normal_owner')==owner and o.get('cm_normal_role')==role),None)
        if helper is None:
            helper=bpy.data.objects.new('CM '+role.title()+' Normal Source',target.data);collection.objects.link(helper)
        helper.data=target.data;helper.parent=target;helper.matrix_parent_inverse=Matrix.Identity(4);helper.matrix_basis=Matrix.Identity(4)
        helper['cm_normal_owner']=owner;helper['cm_normal_role']=role;helper.hide_render=True;helper.hide_select=True;helper.hide_set(True);helpers.append(helper)
    body_input,source=helpers
    modifier=source.modifiers.get('Connected normal geometry') or source.modifiers.new('Connected normal geometry','NODES')
    group=modifier.node_group
    if group is None:
        group=bpy.data.node_groups.new('CM Connected Normal Geometry '+owner,'GeometryNodeTree');modifier.node_group=group
        group.interface.new_socket(name='Geometry',in_out='INPUT',socket_type='NodeSocketGeometry')
        group.interface.new_socket(name='Geometry',in_out='OUTPUT',socket_type='NodeSocketGeometry')
    group.nodes.clear();nodes=group.nodes;links=group.links
    entry=nodes.new('NodeGroupInput');exit=nodes.new('NodeGroupOutput')
    info=nodes.new('GeometryNodeObjectInfo');info.transform_space='RELATIVE';info.inputs['Object'].default_value=body_input
    join=nodes.new('GeometryNodeJoinGeometry');merge=nodes.new('GeometryNodeMergeByDistance');merge.inputs['Distance'].default_value=distance
    smooth=nodes.new('GeometryNodeSetShadeSmooth');smooth.domain='FACE';smooth.inputs['Shade Smooth'].default_value=True
    links.new(entry.outputs['Geometry'],join.inputs['Geometry']);links.new(info.outputs['Geometry'],join.inputs['Geometry'])
    links.new(join.outputs['Geometry'],merge.inputs['Geometry']);links.new(merge.outputs['Geometry'],smooth.inputs['Geometry']);links.new(smooth.outputs['Geometry'],exit.inputs['Geometry'])
    for obj in (head,body):
        transfer=obj.modifiers.get('CM Dynamic Surface Normals') or obj.modifiers.new('CM Dynamic Surface Normals','DATA_TRANSFER')
        transfer.object=source;transfer.use_loop_data=True;transfer.data_types_loops={'CUSTOM_NORMAL'};transfer.loop_mapping='POLYINTERP_NEAREST';transfer.use_object_transform=True
    context.view_layer.update()
    return source
