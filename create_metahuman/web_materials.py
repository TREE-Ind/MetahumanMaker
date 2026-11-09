"""Temporary portable PBR materials for static character GLB delivery."""

import bpy
import numpy as np
from pathlib import Path


def linked(socket):
    return socket.links[0].from_node if socket.is_linked else None


def uv_image(node):
    if not node or node.type != 'TEX_IMAGE' or not node.image:
        return False
    vector = node.inputs['Vector']
    return not vector.is_linked or (vector.links[0].from_socket.name == 'UV'
                                   and vector.links[0].from_node.type in {'TEX_COORD', 'UVMAP'})


def dx_image(normal):
    combined = linked(normal.inputs['Color'])
    if not combined or combined.type != 'COMBINE_COLOR' or combined.mode != 'RGB':
        return None
    separate = linked(combined.inputs['Red'])
    inverse = linked(combined.inputs['Green'])
    if (not separate or separate.type != 'SEPARATE_COLOR' or separate.mode != 'RGB'
            or linked(combined.inputs['Blue']) != separate
            or not inverse or inverse.type != 'MATH' or inverse.operation != 'SUBTRACT'
            or inverse.inputs[0].is_linked or abs(inverse.inputs[0].default_value - 1) > 1e-8
            or linked(inverse.inputs[1]) != separate
            or combined.inputs['Red'].links[0].from_socket.name != 'Red'
            or combined.inputs['Blue'].links[0].from_socket.name != 'Blue'
            or inverse.inputs[1].links[0].from_socket.name != 'Green'):
        return None
    image = linked(separate.inputs['Color'])
    return image if uv_image(image) else None


def sample(pixels, u, v, extension):
    height, width = pixels.shape[:2]
    x, y = u * width - .5, v * height - .5
    x0, y0 = np.floor(x).astype(int), np.floor(y).astype(int)
    fx, fy = x - x0, y - y0
    wrap = extension == 'REPEAT'
    a, b = (x0 % width, (x0 + 1) % width) if wrap else (np.clip(x0, 0, width - 1), np.clip(x0 + 1, 0, width - 1))
    c, d = (y0 % height, (y0 + 1) % height) if wrap else (np.clip(y0, 0, height - 1), np.clip(y0 + 1, 0, height - 1))
    result = (pixels[c, a] * (1 - fx[:, None]) + pixels[c, b] * fx[:, None]) * (1 - fy)
    result += (pixels[d, a] * (1 - fx[:, None]) + pixels[d, b] * fx[:, None]) * fy
    if extension == 'CLIP':
        result[(u < 0) | (u > 1) | (v < 0) | (v > 1)] = 0
    return result


class Materials:
    def __init__(self, limit):
        self.limit = limit
        self.images = []
        self.materials = []
        self.cache = {}
        self.material_cache = {}
        self.calibrated = {}

    def image(self, original, *, normal=False, multiplier=1, limit=None, channel=None):
        limit = limit or self.limit
        key = (original.as_pointer(), normal, multiplier, limit, channel)
        if key in self.cache:
            return self.cache[key]
        if original.source not in {'FILE', 'GENERATED'} or not len(original.pixels):
            raise ValueError('GLB export needs readable single-image textures: ' + original.name)
        dirty=original.is_dirty
        image = (bpy.data.images.new('CM_Web_Copy',width=original.size[0],height=original.size[1],
                                    alpha=original.channels==4,float_buffer=original.is_float)
                 if dirty else original.copy())
        self.images.append(image)
        image.name = 'CM_Web_' + original.name
        image.colorspace_settings.name=original.colorspace_settings.name
        if normal or channel is not None:
            # Color-space changes can reload the original file buffer. Apply
            # them before resizing or editing the private image's pixels.
            image.colorspace_settings.name = 'Non-Color'
        if dirty:
            pixels=np.empty(len(original.pixels),dtype=np.float32)
            original.pixels.foreach_get(pixels)
            image.pixels.foreach_set(pixels)
            image.update()
            del pixels
        width, height = image.size
        ratio = min(1., limit / max(width, height))
        if ratio < 1:
            image.scale(max(1, round(width * ratio)), max(1, round(height * ratio)))
        if normal or multiplier != 1 or channel is not None:
            pixels = np.empty(len(image.pixels), dtype=np.float32)
            image.pixels.foreach_get(pixels)
            pixels = pixels.reshape(-1, image.channels)
            if not np.isfinite(pixels).all():
                raise ValueError('Texture contains invalid pixels: ' + original.name)
            if normal:
                if image.channels < 3:
                    raise ValueError('Normal texture must contain RGB channels')
                pixels[:, 1] = 1 - pixels[:, 1]
            if channel is not None:
                if channel not in range(min(4,image.channels)):
                    raise ValueError('Texture is missing the selected scalar channel')
                scalar=pixels[:,channel].copy()
                pixels[:,:min(3,image.channels)]=scalar[:,None]
            if multiplier != 1:
                pixels[:, :min(3, image.channels)] *= multiplier
                np.clip(pixels, 0, 1, out=pixels)
            image.pixels.foreach_set(pixels.ravel())
            image.update()
        # Resize alone does not mark Blender 4.3 image buffers dirty. Repack
        # the private buffer so glTF cannot reuse old packed/file bytes.
        image.pixels[0] = image.pixels[0]
        image.file_format = 'PNG'
        image.filepath_raw = ''
        image.pack()
        self.cache[key] = image
        return image

    def eye(self, color):
        """Flatten the prepared local eye's UV iris/sclera shader to one map."""
        if not color or color.type != 'MIX_RGB' or color.blend_type != 'MIX':
            return None
        from . import eye_appearance,skin_projection
        sclera, sclera_gain = eye_appearance.color_source(linked(color.inputs[1]))
        iris, iris_gain = eye_appearance.color_source(linked(color.inputs[2]))
        mask = linked(color.inputs[0])
        if not uv_image(sclera) or not iris or iris.type != 'TEX_IMAGE' or not iris.image:
            return None
        if not mask or mask.type != 'MATH' or mask.operation != 'LESS_THAN':
            return None
        distance = linked(mask.inputs[0])
        if not distance or distance.type != 'VECT_MATH' or distance.operation != 'DISTANCE':
            return None
        uv = linked(distance.inputs[0])
        if not uv or uv.type != 'TEX_COORD' or distance.inputs[0].links[0].from_socket.name != 'UV':
            return None
        if sclera.inputs['Vector'].is_linked and linked(sclera.inputs['Vector']) != uv:
            return None
        offset = linked(iris.inputs['Vector'])
        scale = linked(offset.inputs[0]) if offset and offset.type == 'VECT_MATH' and offset.operation == 'ADD' else None
        subtract = linked(scale.inputs[0]) if scale and scale.type == 'VECT_MATH' and scale.operation == 'SCALE' else None
        if (not subtract or subtract.type != 'VECT_MATH' or subtract.operation != 'SUBTRACT'
                or linked(subtract.inputs[0]) != uv
                or any(socket.is_linked for socket in (distance.inputs[1], mask.inputs[1],
                           subtract.inputs[1], scale.inputs['Scale'], offset.inputs[1]))):
            return None
        if sclera.interpolation != 'Linear' or iris.interpolation != 'Linear':
            return None
        if sclera.extension not in {'REPEAT', 'EXTEND', 'CLIP'} or iris.extension not in {'REPEAT', 'EXTEND', 'CLIP'}:
            return None
        radius = float(mask.inputs[1].default_value)
        center = np.asarray(distance.inputs[1].default_value)
        shift = np.asarray(subtract.inputs[1].default_value)
        zoom = float(scale.inputs['Scale'].default_value)
        translate = np.asarray(offset.inputs[1].default_value)
        a = self.image(sclera.image, limit=min(self.limit, 1024))
        b = self.image(iris.image, limit=min(self.limit, 1024))
        if a.colorspace_settings.name != b.colorspace_settings.name or a.channels != 4 or b.channels != 4:
            return None
        data = []
        for image in (a, b):
            pixels = np.empty(len(image.pixels), dtype=np.float32)
            image.pixels.foreach_get(pixels)
            data.append(pixels.reshape(image.size[1], image.size[0], 4))
        width, height = a.size
        output = np.empty((height, width, 4), dtype=np.float32)
        u = (np.arange(width) + .5) / width
        for row in range(height):
            v = (row + .5) / height
            base = sample(data[0], u, v, sclera.extension)
            overlay = sample(data[1], (u - shift[0]) * zoom + translate[0],
                             (v - shift[1]) * zoom + translate[1], iris.extension)
            if a.colorspace_settings.name == 'sRGB':
                base[:,:3]=skin_projection.encoded(np.clip(skin_projection.linear(base[:,:3])*sclera_gain,0,1))
                overlay[:,:3]=skin_projection.encoded(np.clip(skin_projection.linear(overlay[:,:3])*iris_gain,0,1))
            else:
                base[:,:3]=np.clip(base[:,:3]*sclera_gain,0,1)
                overlay[:,:3]=np.clip(overlay[:,:3]*iris_gain,0,1)
            inside = (u - center[0]) ** 2 + (v - center[1]) ** 2 + center[2] ** 2 < radius ** 2
            output[row] = np.where(inside[:, None], overlay, base)
        image = bpy.data.images.new('CM_Web_Eye', width, height, alpha=False)
        self.images.append(image)
        image.colorspace_settings.name = a.colorspace_settings.name
        image.pixels.foreach_set(output.ravel())
        image.update()
        return image

    def material(self, original,*,role=None,profile=None):
        key=(original,role)
        if key in self.material_cache:
            return self.material_cache[key]
        material = bpy.data.materials.new('CM_Web_' + (original.name if original else 'Default'))
        self.materials.append(material)
        material.use_nodes = True
        shader = material.node_tree.nodes.get('Principled BSDF')
        if original is None or not original.use_nodes:
            shader.inputs['Base Color'].default_value = original.diffuse_color if original else (.5, .5, .5, 1)
            self.material_cache[key] = material
            return material
        tree = original.node_tree
        output = next((n for n in tree.nodes if n.type == 'OUTPUT_MATERIAL' and n.is_active_output), None)
        source = linked(output.inputs['Surface']) if output else None
        if not source or source.type != 'BSDF_PRINCIPLED':
            raise ValueError('Use a Principled image-based material before GLB export: ' + original.name)
        for socket in source.inputs:
            if socket.name in shader.inputs and hasattr(socket, 'default_value') and not socket.is_linked:
                shader.inputs[socket.name].default_value = socket.default_value
        material.use_backface_culling = original.use_backface_culling
        material.surface_render_method = original.surface_render_method
        nodes, links = material.node_tree.nodes, material.node_tree.links

        def texture(image, socket, original_node=None):
            node = nodes.new('ShaderNodeTexImage')
            node.image = image
            if original_node:
                node.extension = original_node.extension
                node.interpolation = original_node.interpolation
                uv = linked(original_node.inputs['Vector'])
                if uv and uv.type == 'UVMAP':
                    mapping = nodes.new('ShaderNodeUVMap')
                    mapping.uv_map = uv.uv_map
                    links.new(mapping.outputs['UV'], node.inputs['Vector'])
            links.new(node.outputs['Color'], socket)
            return node

        color = linked(source.inputs['Base Color'])
        if color:
            if uv_image(color):
                texture(self.image(color.image), shader.inputs['Base Color'], color)
            else:
                eye = self.eye(color)
                if not eye:
                    raise ValueError('Flatten the complex Base Color shader before GLB export: ' + original.name)
                texture(eye, shader.inputs['Base Color'])
        normal = linked(source.inputs['Normal'])
        if normal:
            converted = dx_image(normal) if normal.type == 'NORMAL_MAP' else None
            image_node = linked(normal.inputs['Color']) if normal.type == 'NORMAL_MAP' else None
            if converted:
                image_node = converted
                image = self.image(converted.image, normal=True)
            elif normal.type == 'NORMAL_MAP' and uv_image(image_node):
                image = self.image(image_node.image)
            else:
                raise ValueError('Flatten the complex normal shader before GLB export: ' + original.name)
            if normal.type != 'NORMAL_MAP' or normal.space != 'TANGENT' or normal.inputs['Strength'].is_linked:
                raise ValueError('GLB export needs a tangent-space normal map: ' + original.name)
            tangent = nodes.new('ShaderNodeNormalMap')
            tangent.inputs['Strength'].default_value = normal.inputs['Strength'].default_value
            tangent.uv_map = normal.uv_map
            texture(image, tangent.inputs['Color'], image_node)
            links.new(tangent.outputs['Normal'], shader.inputs['Normal'])
        rough = linked(source.inputs['Roughness'])
        if rough:
            factor = 1.
            if rough.type == 'MATH' and rough.operation == 'MULTIPLY' and rough.get('cm_roughness_scale'):
                if rough.inputs[1].is_linked:
                    raise ValueError('Flatten linked roughness scaling before GLB export: '+original.name)
                factor = float(rough.inputs[1].default_value)
                rough = linked(rough.inputs[0])
            channel=None
            if rough and rough.type=='SEPARATE_COLOR' and rough.mode=='RGB':
                output=source.inputs['Roughness'].links[0].from_socket
                outer=linked(source.inputs['Roughness'])
                if outer and outer.type=='MATH':output=outer.inputs[0].links[0].from_socket
                channel={'Red':0,'Green':1,'Blue':2}.get(output.name)
                if channel is None:raise ValueError('Unsupported roughness channel: '+output.name)
                rough=linked(rough.inputs['Color'])
            if not uv_image(rough):
                raise ValueError('Flatten the complex roughness shader before GLB export: ' + original.name)
            # Calibrate only the accepted derived map. An unrelated artist
            # shader on the same mesh must keep its own absolute response.
            gain=1.
            if profile and Path(bpy.path.abspath(rough.image.filepath)).resolve()==Path(profile['path']):
                gain=profile['finish_roughness']/profile['center']
                if channel is None:channel=0  # Native managed skin reads R, not RGB luminance.
                self.calibrated[material.name]={'role':role,'center':profile['center'],
                                               'finish_roughness':profile['finish_roughness'],'gain':gain}
            texture(self.image(rough.image, multiplier=factor*gain,channel=channel), shader.inputs['Roughness'], rough)
        for socket in source.inputs:
            if socket.is_linked and socket.name not in {'Base Color', 'Normal', 'Roughness'}:
                raise ValueError('Flatten linked ' + socket.name + ' before GLB export: ' + original.name)
        self.material_cache[key] = material
        return material

    def close(self):
        for material in self.materials:
            bpy.data.materials.remove(material)
        for image in self.images:
            bpy.data.images.remove(image)
