"""Live grooming references that retain a stable native binding rest surface."""

import bpy
import numpy as np

from . import mesh_ops as m

INDEX = 'cm_groom_head_index'
REST = 'cm_groom_follow_rest'
TAG = 'cm_groom_follow'
TRANSFORM = 'CM_CharacterHeadTransform'


def follow_transform(obj, head):
    constraint = obj.constraints.get(TRANSFORM) or obj.constraints.new('COPY_TRANSFORMS')
    constraint.name = TRANSFORM
    constraint.target = head
    return constraint


def modifier(obj):
    return next((mod for mod in obj.modifiers if mod.type == 'NODES' and mod.node_group
                 and mod.node_group.get(TAG)), None) if obj else None


def source(obj):
    mod = modifier(obj)
    node = mod.node_group.nodes.get('Character Head') if mod else None
    return node.inputs['Object'].default_value if node else None


def objects(scene):
    g = scene.cm_groom
    return list(dict.fromkeys(obj for obj in (g.main_scalp, g.scalp_hair, g.scalp_brows,
        g.scalp_mustache, g.scalp_beard) if obj and modifier(obj)))


def capture(scene):
    """Keep live binding targets when a scene publication may roll back."""
    state=[]
    for obj in objects(scene):
        node=modifier(obj).node_group.nodes.get('Character Head')
        constraint=obj.constraints.get(TRANSFORM)
        state.append((obj,node,node.inputs['Object'].default_value if node else None,
                      constraint,constraint.target if constraint else None,
                      {key:obj.get(key) for key in ('cm_groom_previous_head',
                       'cm_groom_previous_target','cm_groom_follow_status')}))
    return state


def restore(state):
    for obj,node,head,constraint,target,properties in state:
        if node:node.inputs['Object'].default_value=head
        if constraint:constraint.target=target
        for key,value in properties.items():
            if value is None:
                if key in obj:del obj[key]
            else:obj[key]=value


def link(context, obj, head, *, subset=False):
    """Sample head positions by preserved vertex identity in the proxy's frame.

    Offsets leave its original geometry and rest_position untouched, so native
    Hair Curves keeps the same rest-space binding during identity edits.
    """
    m.check_mesh(head, 'Accepted character head')
    if obj == head:
        raise ValueError('The grooming reference must be separate from the character head')
    if subset and INDEX not in obj.data.attributes:
        raise ValueError('Recreate this growth region to link it to the character')
    if not subset and m.topology(obj) != m.topology(head):
        raise ValueError('The grooming reference and character need identical base topology')
    if modifier(obj):
        raise ValueError('This grooming reference is already linked')
    index = obj.data.attributes.get(INDEX) or obj.data.attributes.new(INDEX, 'INT', 'POINT')
    if not subset:
        index.data.foreach_set('value', np.arange(len(obj.data.vertices), dtype=np.int32))
    rest = obj.data.attributes.get(REST) or obj.data.attributes.new(REST, 'FLOAT_VECTOR', 'POINT')
    positions = m.coordinates(obj, world=False).astype(np.float32)
    if subset:
        mapping = np.empty(len(index.data), dtype=np.int32)
        index.data.foreach_get('value', mapping)
        head_positions = m.evaluated_coordinates(context, head)
        if np.any(mapping < 0) or np.any(mapping >= len(head_positions)):
            raise ValueError('Growth-region vertex correspondence is incomplete; recreate the region')
        positions = m.to_local(obj, head_positions[mapping]).astype(np.float32)
    rest.data.foreach_set('vector', positions.ravel())
    group = bpy.data.node_groups.new('CM_CharacterHeadFollow', 'GeometryNodeTree')
    group[TAG] = True
    mod = constraint = None
    try:
        group.interface.new_socket(name='Geometry', in_out='INPUT', socket_type='NodeSocketGeometry')
        group.interface.new_socket(name='Geometry', in_out='OUTPUT', socket_type='NodeSocketGeometry')
        nodes, links = group.nodes, group.links
        incoming = nodes.new('NodeGroupInput')
        outgoing = nodes.new('NodeGroupOutput')
        info = nodes.new('GeometryNodeObjectInfo')
        info.name = 'Character Head'
        info.transform_space = 'RELATIVE'
        info.inputs['Object'].default_value = head
        position = nodes.new('GeometryNodeInputPosition')
        sample = nodes.new('GeometryNodeSampleIndex')
        sample.data_type, sample.domain = 'FLOAT_VECTOR', 'POINT'
        sample.clamp = False
        indices = nodes.new('GeometryNodeInputNamedAttribute')
        indices.data_type = 'INT'
        indices.inputs['Name'].default_value = INDEX
        original = nodes.new('GeometryNodeInputNamedAttribute')
        original.data_type = 'FLOAT_VECTOR'
        original.inputs['Name'].default_value = REST
        delta = nodes.new('ShaderNodeVectorMath')
        delta.operation = 'SUBTRACT'
        move = nodes.new('GeometryNodeSetPosition')
        links.new(info.outputs['Geometry'], sample.inputs['Geometry'])
        links.new(position.outputs['Position'], sample.inputs['Value'])
        links.new(indices.outputs['Attribute'], sample.inputs['Index'])
        links.new(sample.outputs['Value'], delta.inputs[0])
        links.new(original.outputs['Attribute'], delta.inputs[1])
        links.new(incoming.outputs['Geometry'], move.inputs['Geometry'])
        links.new(delta.outputs['Vector'], move.inputs['Offset'])
        links.new(move.outputs['Geometry'], outgoing.inputs['Geometry'])
        mod = obj.modifiers.new('Follow Character Head', 'NODES')
        mod.node_group = group
        obj[TAG] = True
        obj['cm_groom_head_topology'] = m.topology(head)
        if not subset:
            constraint = follow_transform(obj, head)
        return mod
    except Exception:
        if constraint:
            obj.constraints.remove(constraint)
        if mod:
            obj.modifiers.remove(mod)
        bpy.data.node_groups.remove(group)
        raise


def accepted_changed(settings, _context):
    """Retarget only owned live references; incompatible changes keep old hair."""
    scene = settings.id_data
    if not hasattr(scene, 'cm_groom'):
        return
    main, head = scene.cm_groom.main_scalp, settings.result
    if not main or not modifier(main):
        return
    if not head or head.get('cm_groom_reference') or head.type != 'MESH':
        main['cm_groom_follow_status'] = 'Character head unavailable; grooming keeps its previous head.'
        return
    if source(main) == head:
        main['cm_groom_follow_status'] = ''
        return
    if m.topology(head) != main.get('cm_groom_head_topology'):
        main['cm_groom_follow_status'] = 'Head topology changed. Grooming keeps its previous head; use a compatible head or start a new grooming head.'
        return
    nodes = [modifier(obj).node_group.nodes.get('Character Head') for obj in objects(scene)]
    if any(node is None for node in nodes):
        main['cm_groom_follow_status'] = 'A head-follow modifier was edited. Restore it or create a new grooming head.'
        return
    previous_head = source(main)
    previous_target = scene.cm_groom.unreal_target
    restored_target = (main.get('cm_groom_previous_target', '')
                       if main.get('cm_groom_previous_head') == head.name else '')
    for node in nodes:
        node.inputs['Object'].default_value = head
    constraint = main.constraints.get(TRANSFORM)
    if constraint:
        constraint.target = head
    # A replacement identity needs its matching assembled Face mesh selected
    # again, rather than silently binding the groom to an older UE revision.
    scene.cm_groom.unreal_target = restored_target
    main['cm_groom_previous_head'] = previous_head.name if previous_head else ''
    main['cm_groom_previous_target'] = previous_target
    main['cm_groom_follow_status'] = ''
