"""Keep unrelated snapshot shaders out of Cycles' global shader synchronization."""
from contextlib import contextmanager
import bpy


@contextmanager
def worker_materials(context,source,target,kind):
    # Snapshot-only: never change an artist's live material graph for memory use.
    if not context.scene.get('cm_bake_worker'):
        yield
        return
    saved=[];copies=[];slots=list(source.data.materials);world=context.scene.world
    try:
        if kind=='NORMAL':
            for index,original in enumerate(slots):
                if not original or not original.use_nodes:continue
                material=original.copy();copies.append(material);tree=material.node_tree
                outputs=[n for n in tree.nodes if n.type=='OUTPUT_MATERIAL' and n.is_active_output]
                # A direct Principled surface is the common glTF/preview case.
                # Mixed/custom closures retain all their original dependencies.
                if len(outputs)==1 and len(outputs[0].inputs['Surface'].links)==1:
                    shader=outputs[0].inputs['Surface'].links[0].from_node
                    if shader.type=='BSDF_PRINCIPLED':
                        for name in ('Base Color','Roughness','Metallic'):
                            for link in list(shader.inputs[name].links):tree.links.remove(link)
                        reachable=set(outputs);pending=list(outputs)
                        while pending:
                            node=pending.pop()
                            for socket in node.inputs:
                                for link in socket.links:
                                    if link.from_node not in reachable:reachable.add(link.from_node);pending.append(link.from_node)
                        for node in list(tree.nodes):
                            if node not in reachable:tree.nodes.remove(node)
                source.data.materials[index]=material
        allowed=set(source.data.materials)|set(target.data.materials)
        # Evaluated geometry can introduce material slots through modifiers.
        graph=context.evaluated_depsgraph_get()
        for obj in (source,target):
            evaluated=obj.evaluated_get(graph)
            if hasattr(evaluated.data,'materials'):allowed.update(evaluated.data.materials)
        for material in bpy.data.materials:
            if material not in allowed and material.use_nodes:
                saved.append(material);material.use_nodes=False
        context.scene.world=None
        print('BAKE_MATERIAL_SCOPE',kind,'inactive_shaders',len(saved),'source_materials',len(slots),flush=True)
        yield
    finally:
        context.scene.world=world
        for material in saved:material.use_nodes=True
        # Slot assignment preserves polygon material indices; clearing the
        # collection would reset multi-material sources to slot zero.
        for index,material in enumerate(slots):source.data.materials[index]=material
        for material in copies:bpy.data.materials.remove(material)
