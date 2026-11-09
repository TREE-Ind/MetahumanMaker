"""Bound native bake-pixel allocations with overlapping UV raster tiles."""
from pathlib import Path
import json
import bpy
import numpy as np
from contextlib import contextmanager
from contextlib import ExitStack


def progress(scene, label):
    filename = scene.get('cm_bake_progress_file') if scene.get('cm_bake_worker') else None
    if not filename:
        return
    path = Path(filename)
    try:
        data = json.loads(path.read_text(encoding='utf-8')) if path.is_file() else {}
        data.update(status='running', phase=label)
        temporary = path.with_suffix('.tmp')
        temporary.write_text(json.dumps(data), encoding='utf-8')
        temporary.replace(path)
    except (OSError, ValueError):
        pass  # Progress failure must not discard a valid bake.


@contextmanager
def raster_tile(context, target):
    """Rasterize a single offset UV tile without changing delivery UVs."""
    original = target.data
    layer = next((uv for uv in original.uv_layers if uv.active_render), original.uv_layers.active)
    if layer is None:
        raise ValueError('Baking needs target UVs')
    coordinates = np.empty((len(original.loops), 2), dtype=np.float64)
    layer.data.foreach_get('uv', coordinates.ravel())
    offset = np.floor(coordinates.min(0) + 1e-6)
    if not np.any(offset) or np.any(coordinates.max(0) > offset + 1 + 1e-6):
        yield
        return
    private = original.copy()
    try:
        private.uv_layers[layer.name].data.foreach_set('uv', (coordinates - offset).ravel())
        private.update()
        target.data = private
        context.view_layer.update()
        yield
    finally:
        target.data = original
        context.view_layer.update()
        bpy.data.meshes.remove(private)


def bake(context, target, image, kind, *, tile_size=None):
    # The native per-texel bake allocation is separate from assembly scratch.
    # Smaller 8K tiles bound that allocation without reducing output resolution.
    if tile_size is None:tile_size=2048 if max(image.size)>4096 else 4096
    with raster_tile(context, target):
        _bake(context, target, image, kind, tile_size=tile_size)


def _bake(context, target, image, kind, *, tile_size=4096):
    width, height = image.size
    if max(width, height) <= tile_size:
        bpy.ops.object.bake(type=kind)
        return
    if tile_size < 1:
        raise ValueError('Bake tile size must be positive')
    original = target.data
    private = original.copy()
    nodes = [material.node_tree.nodes.active for material in private.materials
             if material and material.use_nodes]
    if not nodes or any(not node or node.type != 'TEX_IMAGE' for node in nodes):
        bpy.data.meshes.remove(private)
        raise ValueError('Tiled baking needs an active image node in every target material')
    bindings = [(node, node.image) for node in nodes]
    tile = None
    buffers=ExitStack()
    try:
        layer = next((uv for uv in private.uv_layers if uv.active_render), private.uv_layers.active)
        if layer is None:
            raise ValueError('Tiled baking needs target UVs')
        coordinates = np.empty((len(private.loops), 2), dtype=np.float64)
        layer.data.foreach_get('uv', coordinates.ravel())
        render_name = layer.name
        active = private.uv_layers.active_index
        # Keep all existing UV layers and their active/render settings intact.
        # Only the explicit raster UV override changes for each tile.
        raster = private.uv_layers.new(name='CM_BakeTileUV')
        private.uv_layers.active_index = active
        private.uv_layers[render_name].active_render = True
        target.data = private
        from . import pixel_buffers
        scratch_root=Path(bpy.path.abspath(context.scene.cm_settings.project_dir))
        output=buffers.enter_context(pixel_buffers.scratch((height,width,4),directory=scratch_root))
        pad = max(context.scene.render.bake.margin + 2, 2)
        stride = tile_size - 2 * pad
        if stride < 1:
            raise ValueError('Bake tile size must exceed twice the bake margin')
        total = ((width + stride - 1) // stride) * ((height + stride - 1) // stride)
        index = 0
        for y in range(0, height, stride):
            for x in range(0, width, stride):
                w, h = min(stride, width - x), min(stride, height - y)
                # A fixed power-of-two canvas also avoids rescaling UVs by
                # fractional image dimensions at the tile boundaries.
                tw = th = tile_size
                uv = (coordinates * [width, height] - [x - pad, y - pad]) / [tw, th]
                raster.data.foreach_set('uv', uv.ravel())
                private.update()
                tile = bpy.data.images.new('CM_BakeTile', width=tw, height=th, alpha=False)
                tile.colorspace_settings.name = image.colorspace_settings.name
                for node, _ in bindings:
                    node.image = tile
                context.view_layer.update()
                index += 1
                progress(context.scene, image.name + ' tile ' + str(index) + '/' + str(total))
                bpy.ops.object.bake(type=kind, uv_layer=raster.name)
                values = np.empty((th, tw, 4), dtype=np.float32)
                tile.pixels.foreach_get(values.ravel())
                output[y:y+h, x:x+w] = values[pad:pad+h, pad:pad+w]
                for node, original_image in bindings:
                    node.image = original_image
                context.view_layer.update()
                bpy.data.images.remove(tile)
                tile = None
                del values
        image.pixels.foreach_set(output.ravel())
    finally:
        for node, original_image in bindings:
            node.image = original_image
        if tile is not None:
            bpy.data.images.remove(tile)
        target.data = original
        context.view_layer.update()
        bpy.data.meshes.remove(private)
        buffers.close()
