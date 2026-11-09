"""Images whose external files belong to the working character's save."""
import bpy


def working(scene):
    s, f, g = scene.cm_settings, scene.cm_features, scene.cm_groom
    objects = {obj for obj in (s.source, s.template, s.result, s.detail, f.proxy,
               f.head_result, f.body_result, f.body_template, f.body_detail, f.full_sculpt,
               f.eye_left, f.eye_right, f.teeth, f.wm1, f.wm2, f.wm3,
               g.groom, g.main_scalp, g.scalp, g.scalp_hair, g.scalp_brows,
               g.scalp_mustache, g.scalp_beard) if obj}
    images, trees, visited = set(), [], set()
    if scene.world and scene.world.use_nodes:
        trees.append(scene.world.node_tree)
    for obj in objects:
        for value in obj.values():
            if isinstance(value, bpy.types.Image):
                images.add(value)
        for slot in obj.material_slots:
            if slot.material and slot.material.use_nodes:
                trees.append(slot.material.node_tree)
        for modifier in obj.modifiers:
            if modifier.type == 'NODES' and modifier.node_group:
                trees.append(modifier.node_group)
    while trees:
        tree = trees.pop()
        if tree in visited:
            continue
        visited.add(tree)
        for node in tree.nodes:
            image = getattr(node, 'image', None)
            if isinstance(image, bpy.types.Image):
                images.add(image)
            group = getattr(node, 'node_tree', None)
            if group:
                trees.append(group)
            for socket in node.inputs:
                value = getattr(socket, 'default_value', None)
                if isinstance(value, bpy.types.Image):
                    images.add(value)
    return images
