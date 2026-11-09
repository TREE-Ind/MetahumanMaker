"""Private, frozen geometry targets and packed displacement textures.

Only shape dependencies are copied. A mesh target is evaluated once into private
geometry, excluding its rig, materials, parents and the rest of its project.
"""

import bpy
import numpy as np
from mathutils import Matrix


def plain(value):
    if isinstance(value, bpy.types.ID):raise ValueError('ID reference')
    if isinstance(value, (str, int, float, bool)):return value
    if hasattr(value, 'items'):return {key:plain(item) for key,item in value.items()}
    return [plain(item) for item in value]


def clean_properties(data):
    for name,value in list(data.items()):
        try:plain(value)
        except (ValueError, TypeError):del data[name]


def strip_materials(data):
    if not hasattr(data,'materials'):return
    indices = np.empty(len(data.polygons),dtype=np.int32) if isinstance(data,bpy.types.Mesh) else None
    if indices is not None:data.polygons.foreach_get('material_index',indices)
    data.materials.clear()
    if indices is not None:data.polygons.foreach_set('material_index',indices)


class Capture:
    def __init__(self, context):
        self.context = context
        self.cache = {}
        self.ids = set()
        self.dependencies = None

    def own(self, data):
        self.ids.add(data)
        return data

    def copy(self, source):
        if source in self.ids:return source
        if source in self.cache:return self.cache[source]
        if isinstance(source,bpy.types.Object):return self.object(source)
        if isinstance(source,bpy.types.Texture):return self.texture(source)
        if isinstance(source,bpy.types.Image):return self.image(source)
        if isinstance(source,bpy.types.Collection):return self.collection(source)
        raise ValueError('Sculpt dependency capture is not ready for '+source.bl_rna.identifier+': '+source.name)

    def object(self, source):
        if source.type not in {'MESH','EMPTY','LATTICE','CURVE','SURFACE'}:
            raise ValueError('Unsupported sculpt target type: '+source.type)
        # Only an unlinked library dependency is immutable. A helper linked in
        # the author's scene may have been edited since the previous return.
        captured = bool(source.get('cm_archive_dependency')) and not source.users_scene
        graph = self.context.evaluated_depsgraph_get()
        evaluated = source if captured else source.evaluated_get(graph)
        if source.type=='MESH':
            data = self.own(source.data.copy() if captured else bpy.data.meshes.new_from_object(
                evaluated,preserve_all_data_layers=True,depsgraph=graph))
        else:
            data = self.own(evaluated.data.copy()) if evaluated.data else None
        obj = self.own(bpy.data.objects.new('CM_Frozen_'+source.name,data))
        self.cache[source] = obj
        if data:
            data.animation_data_clear()
            clean_properties(data)
            strip_materials(data)
            if data.shape_keys:
                self.own(data.shape_keys)
                data.shape_keys.animation_data_clear()
                clean_properties(data.shape_keys)
        for name,value in source.items():
            try:obj[name]=plain(value)
            except (ValueError,TypeError):pass
        for group in source.vertex_groups:
            obj.vertex_groups.new(name=group.name).lock_weight=group.lock_weight
        world = source.get('cm_archive_world') if captured else None
        matrix = Matrix([list(world)[i:i+4] for i in range(0,16,4)]) if world is not None else evaluated.matrix_world.copy()
        obj.matrix_world = matrix
        obj['cm_archive_world'] = [value for row in matrix for value in row]
        obj['cm_archive_dependency'] = True
        # Invisible in render, but retain dependency-graph evaluation. Recovered
        # helpers are linked to a dedicated collection; never hide_viewport=True.
        obj.hide_render = True
        obj.display_type = 'WIRE'
        if source.type=='MESH':
            render = source.get('cm_archive_render_mesh')
            if captured and isinstance(render,bpy.types.Mesh):render=self.own(render.copy())
            elif not captured:
                render_source=self.switch_source(source)
                if render_source:
                    render=self.render_mesh(render_source)
                    if render is None:
                        graph=self.context.evaluated_depsgraph_get()
                        render=self.own(bpy.data.meshes.new_from_object(render_source.evaluated_get(graph),preserve_all_data_layers=True,depsgraph=graph))
                    if render_source.matrix_world!=matrix:
                        render.transform(matrix.inverted() @ render_source.matrix_world)
                    strip_materials(render);clean_properties(render)
                else:render=self.render_mesh(source)
            else:render=None
            if render is not None:self.render_switch(obj,render,matrix)
        return obj

    def switch_source(self, obj):
        """Recognize only our exact render switch; never assume edited nodes match."""
        if len(obj.modifiers)!=1:return None
        modifier=obj.modifiers[0]
        if modifier.type!='NODES' or not modifier.show_viewport or not modifier.show_render:return None
        group=modifier.node_group
        if not group or not group.get('cm_archive_render_switch'):return None
        types=['NodeGroupInput','NodeGroupOutput','GeometryNodeIsViewport','GeometryNodeObjectInfo','GeometryNodeSwitch']
        if sorted(node.bl_idname for node in group.nodes)!=sorted(types):return None
        nodes={node.bl_idname:node for node in group.nodes}
        info=nodes['GeometryNodeObjectInfo'];switch=nodes['GeometryNodeSwitch']
        if info.transform_space!='RELATIVE' or info.inputs['As Instance'].default_value or switch.input_type!='GEOMETRY':return None
        target=info.inputs['Object'].default_value
        if not target or target.type!='MESH' or target.data!=obj.get('cm_archive_render_mesh'):return None
        links={(link.from_node.bl_idname,link.from_socket.name,link.to_node.bl_idname,link.to_socket.name) for link in group.links}
        expected={('NodeGroupInput','Geometry','GeometryNodeSwitch','True'),
            ('GeometryNodeIsViewport','Is Viewport','GeometryNodeSwitch','Switch'),
            ('GeometryNodeObjectInfo','Geometry','GeometryNodeSwitch','False'),
            ('GeometryNodeSwitch','Output','NodeGroupOutput','Geometry')}
        return target if links==expected else None

    def render_mesh(self, source):
        """Sample render subdivision/visibility without changing the source asset.

        Store a separate frozen mesh when render and viewport stacks differ.
        Geometry Nodes' render-context condition requires an actual render graph
        and is refused here rather than treating its viewport output as final.
        """
        if self.dependencies is None:
            self.dependencies={}
            for dependency,owners in bpy.data.user_map().items():
                for owner in owners:self.dependencies.setdefault(owner,set()).add(dependency)
        pending=[source];seen=set();objects=[]
        while pending:
            data=pending.pop()
            if data in seen:continue
            seen.add(data)
            if isinstance(data,bpy.types.Object):objects.append(data)
            if isinstance(data,bpy.types.NodeTree) and any(node.bl_idname=='GeometryNodeIsViewport' for node in data.nodes):
                raise ValueError('This sculpt target switches geometry by render context; capture its render graph before returning')
            pending.extend(dependency for dependency in self.dependencies.get(data,())
                if not (isinstance(data,bpy.types.Object) and data.parent_type=='OBJECT' and dependency==data.parent))
        saved=[]
        for obj in objects:
            for modifier in obj.modifiers:
                values={}
                if modifier.show_viewport!=modifier.show_render:values['show_viewport']=modifier.show_viewport
                if modifier.type in {'SUBSURF','MULTIRES'} and modifier.levels!=modifier.render_levels:values['levels']=modifier.levels
                if values:saved.append((modifier,values))
        settings=self.context.scene.render
        simplify=settings.simplify_subdivision
        simplified=settings.use_simplify and simplify!=settings.simplify_subdivision_render
        if not saved and not simplified:return None
        try:
            for modifier,values in saved:
                modifier.show_viewport=modifier.show_render
                if 'levels' in values:modifier.levels=modifier.render_levels
            if simplified:settings.simplify_subdivision=settings.simplify_subdivision_render
            self.context.view_layer.update()
            graph=self.context.evaluated_depsgraph_get()
            mesh=self.own(bpy.data.meshes.new_from_object(source.evaluated_get(graph),preserve_all_data_layers=True,depsgraph=graph))
            clean_properties(mesh)
            strip_materials(mesh)
            return mesh
        finally:
            for modifier,values in saved:
                for name,value in values.items():setattr(modifier,name,value)
            settings.simplify_subdivision=simplify
            self.context.view_layer.update()

    def render_switch(self, obj, mesh, matrix):
        """Keep the two frozen surfaces selectable by Blender's actual context."""
        render_obj=self.own(bpy.data.objects.new('CM_Render_'+obj.name,mesh))
        render_obj.parent=obj
        render_obj.matrix_parent_inverse=Matrix.Identity(4)
        render_obj.matrix_basis=Matrix.Identity(4)
        render_obj.hide_render=True
        render_obj.display_type='WIRE'
        group=self.own(bpy.data.node_groups.new('Frozen sculpt viewport-render geometry','GeometryNodeTree'))
        group['cm_archive_render_switch']=True
        group.interface.new_socket(name='Geometry',in_out='INPUT',socket_type='NodeSocketGeometry')
        group.interface.new_socket(name='Geometry',in_out='OUTPUT',socket_type='NodeSocketGeometry')
        inputs=group.nodes.new('NodeGroupInput')
        outputs=group.nodes.new('NodeGroupOutput')
        is_viewport=group.nodes.new('GeometryNodeIsViewport')
        info=group.nodes.new('GeometryNodeObjectInfo')
        info.transform_space='RELATIVE'
        info.inputs['Object'].default_value=render_obj
        switch=group.nodes.new('GeometryNodeSwitch');switch.input_type='GEOMETRY'
        group.links.new(is_viewport.outputs['Is Viewport'],switch.inputs['Switch'])
        group.links.new(inputs.outputs['Geometry'],switch.inputs['True'])
        group.links.new(info.outputs['Geometry'],switch.inputs['False'])
        group.links.new(switch.outputs['Output'],outputs.inputs['Geometry'])
        modifier=obj.modifiers.new('Frozen viewport and render surfaces','NODES')
        modifier.node_group=group
        obj['cm_archive_render_mesh']=mesh

    def image(self, source):
        if source.source not in {'FILE','GENERATED'}:
            raise ValueError('Sculpt image tiles/movies/sequences need explicit capture: '+source.name)
        # Image.copy can omit a clean packed buffer and later try to reload its
        # original filepath. Capture the live pixels into a fresh generated ID.
        pixels = np.empty(len(source.pixels),dtype=np.float32)
        source.pixels.foreach_get(pixels)
        width,height = source.size
        if not width or not height or not pixels.size:
            raise ValueError('Sculpt displacement image has no pixels: '+source.name)
        channels = pixels.size // (width*height)
        if pixels.size != width*height*channels or channels not in {1,2,3,4}:
            raise ValueError('Unsupported sculpt displacement pixel layout: '+source.name)
        if channels!=4:
            values=pixels.reshape(-1,channels)
            rgba=np.ones((len(values),4),dtype=np.float32)
            rgba[:,:3]=values[:,:1] if channels<3 else values[:,:3]
            if channels==2:rgba[:,3]=values[:,1]
            pixels=rgba.ravel()
        image = self.own(bpy.data.images.new('CM_Frozen_'+source.name,width,height,alpha=True,float_buffer=source.is_float))
        self.cache[source] = image
        image.colorspace_settings.name=source.colorspace_settings.name
        image.alpha_mode=source.alpha_mode
        image.file_format='OPEN_EXR' if source.is_float else 'PNG'
        image.pixels.foreach_set(pixels)
        for name,value in source.items():
            try:image[name]=plain(value)
            except (ValueError,TypeError):pass
        image.pack()
        image.filepath_raw=source.filepath_raw
        if not image.packed_file:
            raise ValueError('Could not pack sculpt displacement image: '+source.name)
        return image

    def texture(self, source):
        if source.type not in {'NONE','BLEND','CLOUDS','DISTORTED_NOISE','IMAGE','MAGIC','MARBLE','MUSGRAVE','NOISE','STUCCI','VORONOI','WOOD'}:
            raise ValueError('This sculpt texture needs external data capture: '+source.name)
        if source.use_nodes:
            raise ValueError('Legacy node-based displacement textures need explicit capture: '+source.name)
        texture = self.own(source.copy())
        self.cache[source] = texture
        texture.animation_data_clear()
        clean_properties(texture)
        if source.type=='IMAGE' and source.image:texture.image=self.copy(source.image)
        self.remap_pointers(texture)
        return texture

    def collection(self, source):
        collection = self.own(bpy.data.collections.new('CM_Frozen_'+source.name))
        self.cache[source] = collection
        for obj in source.objects:collection.objects.link(self.copy(obj))
        for child in source.children:collection.children.link(self.copy(child))
        return collection

    def remap_pointers(self, struct):
        for prop in struct.bl_rna.properties:
            if prop.identifier=='rna_type' or prop.is_readonly or prop.type!='POINTER':continue
            value = getattr(struct,prop.identifier)
            if isinstance(value,bpy.types.ID):setattr(struct,prop.identifier,self.copy(value))
        # Modifier input IDProperties (for example Geometry Nodes) are not RNA
        # pointers. Never let an unhandled pointer pull original project assets.
        if hasattr(struct,'items'):
            try:items=list(struct.items())
            except TypeError:items=[]
            for name,value in items:
                if isinstance(value,bpy.types.ID):struct[name]=self.copy(value)

    def modifiers(self, obj):
        for modifier in obj.modifiers:
            # Hook's object setter recomputes its bind inverse. A private target
            # has the same transform, so retain the artist's existing binding.
            inverse=modifier.matrix_inverse.copy() if modifier.type=='HOOK' else None
            self.remap_pointers(modifier)
            if inverse is not None:modifier.matrix_inverse=inverse

    def link(self, scene):
        """Link only helper objects; return the collection for transaction ownership."""
        objects = [data for data in self.ids if isinstance(data,bpy.types.Object)]
        if not objects:return None
        collection = self.own(bpy.data.collections.new('Recovered sculpt references'))
        scene.collection.children.link(collection)
        for obj in objects:
            collection.objects.link(obj)
            obj.hide_set(True,view_layer=scene.view_layers[0])
        return collection
