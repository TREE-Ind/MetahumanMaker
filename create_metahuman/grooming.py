"""Hair Curves editing, scalp attachment and native Unreal strand exchange."""

from pathlib import Path
import json
import bpy
from bpy_extras.io_utils import ImportHelper
import numpy as np
from bpy.props import (
    PointerProperty,
    StringProperty,
    FloatProperty,
    FloatVectorProperty,
    IntProperty,
    EnumProperty,
)
from . import mesh_ops as m, workflow


def preview_material(groom, color, roughness, *, strand_color=True):
    """Apply a local Blender preview without changing exchange materials."""
    if not groom or groom.type != 'CURVES' or not groom.is_editable or not groom.data.is_editable:
        raise ValueError('Choose a local Hair Curves groom for preview shading')
    if groom.mode != 'OBJECT':
        raise ValueError('Finish editing the groom before changing preview shading')
    color = np.asarray(color, dtype=float)
    if color.shape != (3,) or not np.isfinite(color).all() or np.any((color < 0) | (color > 1)):
        raise ValueError('Preview color must contain three finite values between zero and one')
    if not np.isfinite(roughness) or not 0 <= roughness <= 1:
        raise ValueError('Preview roughness must be between zero and one')
    previous = groom.data
    # Keep the first source/linked data intact. Later preview changes can reuse
    # our single-user copy instead of copying every strand for each adjustment.
    reuse = bool(previous.get('cm_groom_preview_private') and previous.users == 1 and not previous.use_fake_user)
    private = previous if reuse else previous.copy()
    previous_materials = list(private.materials)
    material = None
    try:
        material = bpy.data.materials.new(groom.name + '_Preview')
        material['cm_groom_preview'] = True
        material.diffuse_color = (*color, 1)
        material.roughness = roughness
        material.use_nodes = True
        nodes = material.node_tree.nodes
        shader = nodes.get('Principled BSDF')
        shader.inputs['Base Color'].default_value = (*color, 1)
        shader.inputs['Roughness'].default_value = roughness
        if strand_color and 'groom_color' in private.attributes:
            attribute = nodes.new('ShaderNodeAttribute')
            attribute.attribute_name = 'groom_color'
            multiply = nodes.new('ShaderNodeMixRGB')
            multiply.blend_type = 'MULTIPLY'
            multiply.inputs[0].default_value = 1
            multiply.inputs[2].default_value = (*color, 1)
            material.node_tree.links.new(attribute.outputs['Color'], multiply.inputs[1])
            material.node_tree.links.new(multiply.outputs['Color'], shader.inputs['Base Color'])
        # Preserve slot count and indices: replacing the preview must not alter
        # per-curve group/material assignments or an independently linked groom.
        if private.materials:
            for index in range(len(private.materials)):
                private.materials[index] = material
        else:
            private.materials.append(material)
        groom.data = private
        private['cm_groom_preview_private'] = True
    except Exception:
        groom.data = previous
        if reuse:
            for index, old in enumerate(previous_materials):private.materials[index] = old
            while len(private.materials) > len(previous_materials):private.materials.pop(index=len(private.materials)-1)
        elif not private.users:
            bpy.data.hair_curves.remove(private)
        if material is not None and not material.users:
            bpy.data.materials.remove(material)
        raise
    for old in set(mat for mat in previous_materials if mat):
        if old.get('cm_groom_preview') and not old.users and not old.use_fake_user:
            bpy.data.materials.remove(old)
    return material


def groom_arrays(payload, scale):
    """Validate and assemble transport arrays before allocating Blender IDs."""
    if not isinstance(payload, dict) or payload.get('schema_version', 1) != 1:
        raise ValueError('Use a version 1 groom strand file')
    if payload.get('coordinates', 'Unreal centimeters') != 'Unreal centimeters':
        raise ValueError('Groom coordinates must be Unreal centimeters')
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError('Groom import requires a positive scene unit scale')
    curves = payload.get('curves')
    if not isinstance(curves, list) or not curves:
        raise ValueError('Groom needs curves with at least two points')
    lengths = []
    for index, curve in enumerate(curves):
        points = curve.get('points') if isinstance(curve, dict) else None
        if not isinstance(points, (list, tuple)) or len(points) < 2:
            raise ValueError('Strand ' + str(index + 1) + ' needs at least two points')
        lengths.append(len(points))
    total = sum(lengths)
    if total > np.iinfo(np.int32).max:
        raise ValueError('Groom point count exceeds Blender curve indexing capacity')
    positions = np.empty((total, 3), dtype=np.float64)
    widths = np.empty(total, dtype=np.float64)
    groups = np.empty(len(curves), dtype=np.int32)
    guides = np.empty(len(curves), dtype=np.int32)
    uv = np.zeros((len(curves), 3), dtype=np.float64)
    colors = np.ones((total, 4), dtype=np.float64) if any('colors' in c for c in curves) else None
    limit = np.finfo(np.float32).max

    def array(value, shape, field, index):
        try:
            values = np.asarray(value, dtype=np.float64)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError('Strand ' + str(index + 1) + ': invalid ' + field) from exc
        if values.shape != shape:
            raise ValueError('Strand ' + str(index + 1) + ': ' + field + ' must have shape ' + str(shape))
        return values

    def integer(value, field, index, maximum):
        if not isinstance(value, (int, np.integer)) or not 0 <= value <= maximum:
            raise ValueError('Strand ' + str(index + 1) + ': invalid ' + field)
        return value

    first = 0
    for index, (curve, count) in enumerate(zip(curves, lengths)):
        last = first + count
        positions[first:last] = array(curve['points'], (count, 3), 'points', index)
        values = array(curve.get('widths', np.full(count, .01)), (count,), 'widths', index)
        widths[first:last] = values
        groups[index] = integer(curve.get('group', 0), 'group', index, np.iinfo(np.int32).max)
        guides[index] = integer(curve.get('guide', 0), 'guide flag', index, 1)
        uv[index, :2] = array(curve.get('root_uv', [0, 0]), (2,), 'root UV', index)
        if colors is not None and 'colors' in curve:
            colors[first:last, :3] = array(curve['colors'], (count, 3), 'colors', index)
        first = last
    with np.errstate(over='ignore', invalid='ignore'):
        positions /= [scale, -scale, scale]
        widths /= scale * 2
    for field, values in [('points', positions), ('widths', widths), ('root UVs', uv), ('colors', colors)]:
        if values is not None and (not np.isfinite(values).all() or np.any(np.abs(values) > limit)):
            raise ValueError('Groom ' + field + ' must be finite and fit Blender precision at this scene scale')
    if np.any(widths < 0):
        raise ValueError('Groom widths must be nonnegative')
    records = payload.get('group_settings', [])
    if not isinstance(records, list):
        raise ValueError('Groom group settings must be a list')
    seen = set()
    for record in records:
        group = record.get('id') if isinstance(record, dict) else None
        if not isinstance(group, int) or not 0 <= group <= np.iinfo(np.int32).max or group in seen:
            raise ValueError('Groom settings need unique nonnegative integer group IDs')
        seen.add(group)
    try:
        settings = json.dumps(records, allow_nan=False) if records else ''
    except (ValueError, TypeError, OverflowError) as exc:
        raise ValueError('Groom group settings are invalid') from exc
    return lengths, positions, widths, groups, guides, uv, colors, settings


def import_curves(context, payload, name="Imported_Groom"):
    arrays = groom_arrays(payload, context.scene.unit_settings.scale_length * 100)
    lengths, points, widths, groups, guides, uv, colors, settings = arrays
    data = obj = None
    with m.selection_state(context, on_error=True):
        try:
            data = bpy.data.hair_curves.new(name)
            data.add_curves(lengths)
            data.position_data.foreach_set('vector', points.ravel())
            for field, values in [('groom_group_id', groups), ('groom_guide', guides)]:
                attribute = data.attributes.new(field, 'INT', 'CURVE')
                attribute.data.foreach_set('value', values)
            attribute = data.attributes.new('groom_root_uv', 'FLOAT_VECTOR', 'CURVE')
            attribute.data.foreach_set('vector', uv.ravel())
            radius = data.attributes.get('radius') or data.attributes.new('radius', 'FLOAT', 'POINT')
            radius.data.foreach_set('value', widths)
            if colors is not None:
                attribute = data.attributes.new('groom_color', 'FLOAT_COLOR', 'POINT')
                attribute.data.foreach_set('color', colors.ravel())
            obj = bpy.data.objects.new(name, data)
            if settings:
                obj['cm_groom_unreal_groups'] = settings
            (context.collection or context.scene.collection).objects.link(obj)
            m.activate(context, obj)
            return obj
        except Exception:
            if obj is not None:
                bpy.data.objects.remove(obj, do_unlink=True)
            if data is not None and not data.users:
                bpy.data.hair_curves.remove(data)
            raise



def curve_attribute(data, name, domain, kinds, count, default, *, width=1):
    attribute = data.attributes.get(name)
    if attribute is None:
        return np.full((count, width) if width > 1 else count, default, dtype=np.float64)
    if attribute.domain != domain or attribute.data_type not in kinds or len(attribute.data) != count:
        raise ValueError('Groom attribute ' + name + ' must use ' + domain + ' domain and ' + '/'.join(kinds))
    values = np.empty((count, width) if width > 1 else count, dtype=np.float64)
    field = 'color' if width == 4 else 'vector' if width > 1 else 'value'
    attribute.data.foreach_get(field, values.ravel())
    return values


def export_curves(context, objects):
    curves = []
    groups = []
    next_group = 0
    unit = context.scene.unit_settings.scale_length * 100
    for obj in objects:
        if not obj or obj.type not in {"CURVES", "CURVE"}:
            continue
        first_curve = len(curves)
        evaluated = obj.evaluated_get(context.evaluated_depsgraph_get())
        data = evaluated.data
        matrix = np.asarray(evaluated.matrix_world)
        if not np.isfinite(matrix).all():
            raise ValueError(obj.name + ': groom transform must be finite before export')
        singular = np.linalg.svd(matrix[:3,:3],compute_uv=False)
        volume = abs(np.linalg.det(matrix[:3,:3]))
        uniform = singular.max()-singular.min() <= max(singular.max(),1e-12)*1e-6
        if obj.type == "CURVES":
            if not isinstance(data, bpy.types.Curves):
                raise ValueError(obj.name + ': modifiers must output Hair Curves before export')
            count = len(data.curves)
            if not count:
                continue
            offsets = np.empty(len(data.curve_offset_data), dtype=np.int32)
            data.curve_offset_data.foreach_get('value', offsets)
            lengths = np.diff(offsets)
            if len(lengths) != count or np.any(lengths < 2):
                raise ValueError(obj.name + ': every exported strand needs at least two points')
            points = np.empty((len(data.points), 3), dtype=np.float64)
            data.position_data.foreach_get('vector', points.ravel())
            world = (points @ matrix[:3, :3].T + matrix[:3, 3]) * [unit, -unit, unit]
            if not np.isfinite(world).all():
                raise ValueError(obj.name + ': groom contains non-finite evaluated positions')
            group_ids = curve_attribute(data, 'groom_group_id', 'CURVE', ('INT',), count, 0)
            guides = curve_attribute(data, 'groom_guide', 'CURVE', ('INT', 'BOOLEAN'), count, 0)
            if np.any(group_ids < 0) or np.any((guides != 0) & (guides != 1)):
                raise ValueError(obj.name + ': groom groups must be nonnegative and guide flags must be 0 or 1')
            uv = curve_attribute(data, 'groom_root_uv', 'CURVE', ('FLOAT_VECTOR',), count, 0, width=3)
            radii = curve_attribute(data, 'radius', 'POINT', ('FLOAT',), len(points), .01 / (unit * 2)) * unit * 2
            if np.any(radii < 0) or not np.isfinite(radii).all() or not np.isfinite(uv).all():
                raise ValueError(obj.name + ': groom widths and root UVs must be finite, with nonnegative widths')
            colors = None
            if 'groom_color' in data.attributes:
                colors = curve_attribute(data, 'groom_color', 'POINT', ('FLOAT_COLOR', 'BYTE_COLOR'), len(points), 1, width=4)
                if not np.isfinite(colors).all():
                    raise ValueError(obj.name + ': groom colors must be finite')
            if uniform:
                width_scale = volume ** (1 / 3)
            else:
                tangent = np.empty_like(points)
                tangent[1:-1] = (points[2:] - points[:-2]) * .5
                tangent[offsets[:-1]] = points[offsets[:-1] + 1] - points[offsets[:-1]]
                tangent[offsets[1:] - 1] = points[offsets[1:] - 1] - points[offsets[1:] - 2]
                magnitude = np.linalg.norm(tangent, axis=1)
                stretch = np.linalg.norm(tangent @ matrix[:3, :3].T, axis=1) / np.maximum(magnitude, 1e-12)
                stretch = np.where(magnitude > 1e-12, stretch, volume ** (1 / 3))
                width_scale = np.sqrt(volume / np.maximum(stretch, 1e-12))
            radii *= width_scale
            # RNA access is complete before constructing the transport records.
            for index, (first, last) in enumerate(zip(offsets[:-1], offsets[1:])):
                item = {'points': world[first:last].tolist(), 'widths': radii[first:last].tolist(),
                        'group': int(group_ids[index]), 'guide': int(guides[index]), 'root_uv': uv[index, :2].tolist()}
                if colors is not None:
                    item['colors'] = colors[first:last, :3].tolist()
                curves.append(item)
        else:
            for spline in data.splines:
                values = (
                    spline.bezier_points if spline.type == "BEZIER" else spline.points
                )
                points = np.array([p.co[:3] for p in values])
                world = (points @ matrix[:3, :3].T + matrix[:3, 3]) * [
                    unit,
                    -unit,
                    unit,
                ]
                curves.append(
                    {
                        "points": world.tolist(),
                        "widths": [0.01] * len(world),
                        "group": 0,
                        "guide": 0,
                        "root_uv": [0, 0],
                    }
                )
        settings = {int(item['id']): item for item in json.loads(obj.get('cm_groom_unreal_groups', '[]'))}
        # Different objects often both start with group 0. Keep their native
        # settings independent when the objects are combined or pushed together.
        group_map = {}
        for curve in curves[first_curve:]:
            previous = int(curve['group'])
            if previous not in group_map:
                group_map[previous] = next_group
                groups.append({**settings.get(previous, {}), 'id': next_group,
                               'source_object': obj.name, 'source_group': previous})
                next_group += 1
            curve['group'] = group_map[previous]
    if not curves:
        raise ValueError("Select at least one Curves groom")
    from . import unreal_groom
    unreal_groom.asset_settings(groups)
    return {"schema_version": 1, "coordinates": "Unreal centimeters", "curves": curves,
            'group_settings': groups}


def copy_mix_materials(context, objects, combined):
    """Retain evaluated Blender material slots and each exported strand's index.

    These preview materials are separate from the native exchange group settings.
    Link existing material IDs without changing their node graphs; later Preview
    adjustments already make the mixed groom's shading private.
    """
    materials=[];lookup={};indices=[]
    for obj in objects:
        evaluated=obj.evaluated_get(context.evaluated_depsgraph_get())
        data=evaluated.data
        if obj.type=='CURVES':
            count=len(data.curves)
            values=np.asarray(curve_attribute(data,'material_index','CURVE',('INT',),count,0),dtype=np.int32)
        else:
            count=len(data.splines);values=np.asarray([spline.material_index for spline in data.splines],dtype=np.int32)
        if not count:continue
        slots=[slot.material for slot in evaluated.material_slots] or [None]
        if np.any(values<0) or np.any(values>=len(slots)):
            raise ValueError(obj.name+': a strand refers to a missing preview material slot')
        remap=[]
        for material in slots:
            if material not in lookup:lookup[material]=len(materials);materials.append(material)
            remap.append(lookup[material])
        indices.extend(np.asarray(remap,dtype=np.int32)[values].tolist())
    if len(indices)!=len(combined.data.curves):
        raise ValueError('The mixed preview materials do not match the exported strands')
    for material in materials:combined.data.materials.append(material)
    attribute=combined.data.attributes.get('material_index') or combined.data.attributes.new('material_index','INT','CURVE')
    attribute.data.foreach_set('value',indices)


def select_population(context, groom, population):
    """Select whole native curves from transport guide flags, without reshaping."""
    if population not in {'GUIDES', 'STRANDS', 'ALL'}:
        raise ValueError('Choose guides, render strands or all hair')
    if not groom or groom.type != 'CURVES' or not groom.is_editable or not groom.data.is_editable:
        raise ValueError('Choose a local Hair Curves groom')
    if context.mode not in {'OBJECT', 'SCULPT_CURVES'} or (context.mode == 'SCULPT_CURVES' and context.object != groom):
        raise ValueError('Finish the active edit before selecting another groom')
    data = groom.data
    count = len(data.curves)
    if not count:
        raise ValueError('Add strands before selecting hair')
    selected = np.ones(count, dtype=bool)
    if population != 'ALL':
        guides = curve_attribute(data, 'groom_guide', 'CURVE', ('INT', 'BOOLEAN'), count, 0)
        if np.any((guides != 0) & (guides != 1)):
            raise ValueError('Guide flags must be 0 or 1')
        selected = guides == (1 if population == 'GUIDES' else 0)
    total = int(np.count_nonzero(selected))
    if not total:
        raise ValueError('This groom has no ' + ('guides' if population == 'GUIDES' else 'render strands'))
    if data.users > 1:
        if context.mode != 'OBJECT':
            raise ValueError('Finish Hair, then select the population again to keep linked copies independent')
        data = data.copy()
        groom.data = data
    selection = data.attributes.get('.selection')
    if selection and (selection.domain != 'CURVE' or selection.data_type != 'BOOLEAN'):
        data.attributes.remove(selection)
        selection = None
    if selection is None:
        selection = data.attributes.new('.selection', 'BOOLEAN', 'CURVE')
    selection.data.foreach_set('value', selected)
    data.selection_domain = 'CURVE'
    data.update_tag()
    context.view_layer.update()
    return total, count


def draw_population_selection(layout):
    row = layout.row(align=True)
    for label, population in [('Guides', 'GUIDES'), ('Strands', 'STRANDS'), ('All', 'ALL')]:
        row.operator('cm.groom', text=label).action = 'SELECT_' + population


def seed_curves(context, scalp, settings):
    """Distribute roots over surface area, including the whole extracted region."""
    m.check_mesh(scalp, "Scalp")
    if not scalp.data.uv_layers:
        raise ValueError("Scalp needs a UV map before creating hair")
    world = m.evaluated_coordinates(context, scalp)
    if len(world) != len(scalp.data.vertices):
        raise ValueError("Use a scalp without topology-changing modifiers")
    scalp.data.calc_loop_triangles()
    triangles = np.array([tri.vertices[:] for tri in scalp.data.loop_triangles], dtype=int)
    if not len(triangles):
        raise ValueError("Scalp has no faces for hair roots")
    if scalp.vertex_groups.get("CM_Scalp"):
        mask = workflow.group_weights(scalp, "CM_Scalp") > 0.5
    elif scalp.get("cm_groom_region"):
        mask = np.ones(len(world), dtype=bool)
    else:
        from . import groom_regions
        mask = groom_regions.masks(context,scalp,world)[0][settings.region]
    triangles = triangles[np.all(mask[triangles], axis=1)]
    vertices = world[triangles]
    cross = np.cross(vertices[:, 1] - vertices[:, 0], vertices[:, 2] - vertices[:, 0])
    areas = np.linalg.norm(cross, axis=1)
    valid = areas > 1e-16
    vertices, cross, areas = vertices[valid], cross[valid], areas[valid]
    if not len(areas):
        raise ValueError("No usable faces in this region; create a scalp or paint CM_Scalp weights")
    rng = np.random.default_rng(settings.hair_seed)
    chosen = rng.choice(len(areas), settings.hair_count, p=areas / areas.sum())
    uv = rng.random((settings.hair_count, 2))
    root = np.sqrt(uv[:, 0])
    weights = np.column_stack((1 - root, root * (1 - uv[:, 1]), root * uv[:, 1]))
    roots = np.einsum('ni,nij->nj', weights, vertices[chosen])
    normals = cross[chosen] / areas[chosen, None]
    unit = context.scene.unit_settings.scale_length * 100
    payload = {"curves": [
        {"points": ((point + np.linspace(0, settings.hair_length, 6)[:, None] * normal)
                    * [unit, -unit, unit]).tolist(), "group": 0, "guide": 0}
        for point, normal in zip(roots, normals)
    ]}
    groom = import_curves(context, payload, settings.name)
    try:
        regions=[getattr(settings,'scalp_'+name) for name in ('hair','brows','mustache','beard')]
        target=settings.main_scalp if settings.main_scalp and scalp in regions else scalp
        attach(context, groom, target)
    except Exception:
        data = groom.data
        bpy.data.objects.remove(groom, do_unlink=True)
        if not data.users:
            bpy.data.hair_curves.remove(data)
        raise
    return groom


def attach(context, groom, scalp):
    from . import groom_binding
    return groom_binding.attach(context, groom, scalp)


def simplify_curves(groom, ratio):
    """Thin each render/guide population independently on private curve data."""
    if not groom or groom.type != 'CURVES':
        raise ValueError('Choose a Hair Curves groom')
    if not groom.is_editable or not groom.data.is_editable:
        raise ValueError('Make the groom local before simplifying')
    if groom.mode != 'OBJECT':
        raise ValueError('Finish editing the groom before simplifying')
    if not np.isfinite(ratio) or not 0 < ratio <= 1:
        raise ValueError('Keep Strands must be greater than zero and at most one')
    data = groom.data
    count = len(data.curves)
    if not count:
        raise ValueError('Add strands before simplifying this groom')
    groups = curve_attribute(data, 'groom_group_id', 'CURVE', ('INT',), count, 0)
    guides = curve_attribute(data, 'groom_guide', 'CURVE', ('INT', 'BOOLEAN'), count, 0)
    if np.any(groups < 0) or np.any((guides != 0) & (guides != 1)):
        raise ValueError('Groom groups must be nonnegative and guide flags must be 0 or 1')
    keep = np.zeros(count, dtype=bool)
    # A sparse guide population must not disappear when interleaved with a much
    # denser render population. Also retain small independent material groups.
    populations, category = np.unique(np.column_stack((groups, guides)), axis=0, return_inverse=True)
    for index in range(len(populations)):
        members = np.flatnonzero(category == index)
        wanted = max(1, int(len(members) * ratio))
        keep[members[np.linspace(0, len(members) - 1, wanted, dtype=int)]] = True
    removed = np.flatnonzero(~keep)
    if not len(removed):
        return count, count
    private = data.copy()
    try:
        private.remove_curves(indices=removed.tolist())
        groom.data = private
    except Exception:
        groom.data = data
        if not private.users:
            bpy.data.hair_curves.remove(private)
        raise
    # Retain the original datablock for linked objects and Blender Undo.
    return count, len(private.curves)


def create_scalp(context, base, region):
    m.check_mesh(base, "Head for scalp")
    points = m.evaluated_coordinates(context, base)
    if len(points) != len(base.data.vertices):
        raise ValueError('Use a head without topology-changing modifiers for scalp extraction')
    from . import groom_regions
    masks, method = groom_regions.masks(context,base,points)
    mask = masks[region]
    polygon_indices = [
        p.index
        for p in base.data.polygons
        if all(mask[i] for i in p.vertices)
    ]
    if not polygon_indices:
        raise ValueError(
            "No polygons in this scalp region; sculpt/select a more suitable head"
        )
    scalp = m.duplicate(context, base, "Scalp_" + region)
    scalp.hide_render = True
    scalp.data.vertices.foreach_set("co", m.to_local(scalp, points).ravel())
    m.activate(context, scalp)
    import bmesh

    bm = bmesh.new()
    try:
        bm.from_mesh(scalp.data)
        bm.faces.ensure_lookup_table()
        keep = set(polygon_indices)
        bmesh.ops.delete(
            bm,
            geom=[p for p in bm.faces if p.index not in keep],
            context="FACES",
        )
        bm.to_mesh(scalp.data)
    except Exception:
        data = scalp.data
        bpy.data.objects.remove(scalp, do_unlink=True)
        if not data.users:
            bpy.data.meshes.remove(data)
        raise
    finally:
        bm.free()
    scalp.data.update()
    scalp['cm_groom_region'] = region
    scalp['cm_groom_region_method'] = method
    scalp.display_type = 'WIRE'
    from . import groom_follow
    if groom_follow.modifier(base):
        try:
            groom_follow.link(context, scalp, groom_follow.source(base), subset=True)
        except Exception:
            data = scalp.data
            bpy.data.objects.remove(scalp, do_unlink=True)
            if not data.users:
                bpy.data.meshes.remove(data)
            raise
    return scalp


def reference_head(context, head, *, follow=False):
    """Freeze the visible head, preserving native facial weights and UVs."""
    m.check_mesh(head,'Head for grooming')
    data = bpy.data.meshes.new_from_object(head.evaluated_get(context.evaluated_depsgraph_get()),
                                         depsgraph=context.evaluated_depsgraph_get())
    scalp = bpy.data.objects.new('MetaHuman_Groom_Head',data)
    context.collection.objects.link(scalp)
    scalp.matrix_world = head.matrix_world.copy()
    for group in head.vertex_groups:
        scalp.vertex_groups.new(name=group.name)
    scalp.hide_render = True
    scalp['cm_groom_reference'] = True
    material = next((mat for mat in bpy.data.materials if mat.get('cm_groom_reference')),None)
    if material is None:
        material = bpy.data.materials.new('CM_GroomReference')
        material['cm_groom_reference'] = True
        material.diffuse_color = (.35,.35,.35,1)
        material.use_nodes = True
        shader = material.node_tree.nodes.get('Principled BSDF')
        shader.inputs['Base Color'].default_value = (.35,.35,.35,1)
        shader.inputs['Roughness'].default_value = .65
    data.materials.clear();data.materials.append(material)
    if follow:
        from . import groom_follow
        try:
            groom_follow.link(context, scalp, head)
        except Exception:
            bpy.data.objects.remove(scalp, do_unlink=True)
            if not data.users:
                bpy.data.meshes.remove(data)
            raise
    return scalp


def setup_head(context, scalp):
    regions = {}
    try:
        if not scalp.data.uv_layers:
            raise ValueError('The grooming head needs UVs for root attachment')
        for region in ('HAIR','BROWS','MUSTACHE','BEARD'):
            regions[region] = create_scalp(context,scalp,region)
            regions[region].hide_set(True)
    except Exception:
        for obj in [*regions.values(),scalp]:
            data=obj.data;bpy.data.objects.remove(obj,do_unlink=True)
            if not data.users:bpy.data.meshes.remove(data)
        raise
    g=context.scene.cm_groom
    old=g.main_scalp
    g.main_scalp=scalp
    for region,obj in regions.items():setattr(g,'scalp_'+region.lower(),obj)
    g.scalp=regions[g.region]
    # A different head must not silently keep the preceding character's target.
    g.unreal_target=scalp.get('cm_unreal_mesh','')
    if old and old!=scalp and old.get('cm_groom_reference'):
        old.hide_set(True)
    m.activate(context,scalp)
    focus_groom(context)
    return scalp


def focus_groom(context, editing=None):
    from . import guided
    s,f,g=context.scene.cm_settings,context.scene.cm_features,context.scene.cm_groom
    visible=s.result if s.result and not s.result.get('cm_groom_reference') else g.main_scalp
    # Agent actions and imported saved scenes can enter grooming from another
    # step. Keep Finish Hair/Scalp visible while native brushes own the mode.
    f.ui_stage='GROOM'
    if g.main_scalp and g.main_scalp.get('cm_groom_reference'):
        # A wire reference is useful over the accepted character, but becomes
        # unreadable when the reference itself is the only visible head.
        g.main_scalp.display_type='SOLID' if visible==g.main_scalp else 'WIRE'
    objects=[editing or g.groom or visible,visible]
    if g.main_scalp:
        objects.extend(obj for obj in context.scene.objects if obj.type=='CURVES'
                       and obj.data.surface==g.main_scalp and not obj.hide_get())
    if visible==s.result:
        objects.extend([f.eye_left,f.eye_right,f.teeth])
    guided.focus(context,list(dict.fromkeys(obj for obj in objects if obj)),f.front_axis)
    if editing or g.groom or visible:
        m.activate(context,editing or g.groom or visible)


def head_setup_idle(context):
    from . import groom_jobs, unreal_jobs, sculpt_session, native_body, bake_jobs, detail_jobs
    if context.mode!='OBJECT' or context.scene.cm_groom.scalp_editing or sculpt_session.pending():
        raise ValueError('Finish the current edit before changing the grooming head')
    if any(job.running() for job in (groom_jobs,unreal_jobs,bake_jobs,detail_jobs)) or native_body._JOB:
        raise ValueError('Wait for the current character operation before changing the grooming head')


def import_head(context, path):
    """Keep a static full-head scalp; discard only the temporary FBX objects."""
    before = set(bpy.data.objects)
    previous_materials = set(bpy.data.materials)
    previous_images = set(bpy.data.images)
    scalp = None
    try:
        bpy.ops.import_scene.fbx(filepath=str(path), use_image_search=False, use_anim=False)
        meshes = [o for o in bpy.data.objects if o not in before and o.type == 'MESH']
        if not meshes:
            raise ValueError('Unreal head FBX contained no mesh')
        head = max(meshes, key=lambda o: len(o.data.vertices))
        m.activate(context, head)
        bpy.ops.object.mode_set(mode='EDIT')
        bpy.ops.mesh.select_all(action='SELECT')
        bpy.ops.mesh.separate(type='MATERIAL')
        bpy.ops.object.mode_set(mode='OBJECT')
        meshes = [o for o in bpy.data.objects if o not in before and o.type == 'MESH']
        head = max(meshes, key=lambda o: len(o.data.vertices))
        scalp = reference_head(context,head)
    finally:
        if context.mode != 'OBJECT':
            bpy.ops.object.mode_set(mode='OBJECT')
        for obj in list(bpy.data.objects):
            if obj not in before and obj != scalp:
                data = obj.data
                bpy.data.objects.remove(obj, do_unlink=True)
                if data and not data.users and isinstance(data, (bpy.types.Mesh, bpy.types.Armature)):
                    bpy.data.batch_remove(ids=[data])
        for material in list(bpy.data.materials):
            if material not in previous_materials and not material.users:
                bpy.data.materials.remove(material)
        for image in list(bpy.data.images):
            if image not in previous_images and not image.users:
                bpy.data.images.remove(image)
    return setup_head(context,scalp)


class CM_OT_import_groom_head(bpy.types.Operator,ImportHelper):
    bl_idname='cm.import_groom_head'
    bl_label='Import Grooming Head (FBX)'
    bl_description='Import an exported MetaHuman head and prepare four editable growth regions'
    bl_options={'REGISTER','UNDO'}
    filename_ext='.fbx'
    filter_glob:StringProperty(default='*.fbx',options={'HIDDEN'})

    def execute(self,context):
        try:
            head_setup_idle(context)
            import_head(context,self.filepath)
            context.scene.cm_features.last_action_error=False
            context.scene.cm_settings.status='Head and four growth regions ready. Choose the matching Face mesh in Unreal before Push.'
            return {'FINISHED'}
        except Exception as exc:
            context.scene.cm_features.last_action_error=True
            context.scene.cm_settings.status=str(exc)
            self.report({'ERROR'},str(exc))
            return {'CANCELLED'}


def region_changed(settings, context):
    scalp = getattr(settings, 'scalp_' + settings.region.lower(), None)
    if scalp:
        settings.scalp = scalp


class CM_GroomSettings(bpy.types.PropertyGroup):
    groom: PointerProperty(name="Groom", type=bpy.types.Object)
    scalp: PointerProperty(name="Scalp", type=bpy.types.Object)
    main_scalp: PointerProperty(name='Full Head Scalp', type=bpy.types.Object)
    scalp_hair: PointerProperty(type=bpy.types.Object)
    scalp_brows: PointerProperty(type=bpy.types.Object)
    scalp_mustache: PointerProperty(type=bpy.types.Object)
    scalp_beard: PointerProperty(type=bpy.types.Object)
    scalp_editing: bpy.props.BoolProperty(default=False)
    exchange_status: StringProperty(name='Unreal Exchange')
    exchange_receipt: StringProperty(subtype='FILE_PATH')
    exchange_result: StringProperty()
    last_push_result: StringProperty()
    exchange_applied: StringProperty()
    lash_fit_receipt: StringProperty(subtype='FILE_PATH')
    lash_fit_result: StringProperty()
    lash_component: StringProperty(name='Lash Component',default='Eyelashes')
    lash_rebuild_guides: bpy.props.BoolProperty(name='Rebuild Guides from Strands',default=False,
        description='Propose replacement guides sampled from authored visible strands; the original groom is retained')
    lash_maximum_offset: FloatProperty(name='Root Fit Reach',default=.2,min=.01,max=.5,subtype='FACTOR',
        description='Maximum root movement as a fraction of eye width in the binding reference')
    lash_fit_options: bpy.props.BoolProperty(name='Fit Options',default=False)
    show_asset_paths: bpy.props.BoolProperty(name='Use Asset Paths', default=False)
    unreal_asset: StringProperty(name="Unreal Groom Asset", default="/Game/MyGroom")
    unreal_target: StringProperty(
        name="Target Skeletal Mesh", default=""
    )
    name: StringProperty(name="Groom Name", default="CustomHair")
    wardrobe_slot: EnumProperty(
        name='Creator Section', default='Hair',
        description='Equip the last delivered groom in this wardrobe section; other sections retain their selections',
        items=[('Hair', 'Hair', ''), ('Eyebrows', 'Eyebrows', ''),
               ('Mustache', 'Mustache', ''), ('Beard', 'Beard', ''),
               ('Eyelashes', 'Eyelashes', '')],
    )
    simplify_ratio: FloatProperty(name="Keep Strands", default=0.5, min=0.01, max=1)
    hair_length: FloatProperty(
        name="Initial Hair Length", default=0.04, min=0.0001, max=2
    )
    hair_count: IntProperty(name="Initial Strands", default=100, min=1, max=100000)
    hair_seed: IntProperty(name="Seed", default=42, min=0)
    preview_color: FloatVectorProperty(name='Preview Color', subtype='COLOR', size=3,
        default=(.045, .018, .008), min=0, max=1)
    preview_roughness: FloatProperty(name='Preview Roughness', default=.45, min=0, max=1)
    preview_strand_color: bpy.props.BoolProperty(name='Use Strand Colors', default=True,
        description='Multiply preview color by the imported per-point groom colors when available')
    region: EnumProperty(
        name="Scalp Region",
        update=region_changed,
        items=[
            ("HAIR", "Hair", ""),
            ("BROWS", "Eyebrows", ""),
            ("MUSTACHE", "Mustache", ""),
            ("BEARD", "Beard", ""),
        ],
    )


class CM_OT_groom(bpy.types.Operator):
    bl_idname = "cm.groom"
    bl_label = "Groom Action"
    bl_options = {"REGISTER", "UNDO"}
    action: StringProperty()

    @classmethod
    def description(cls, context, properties):
        if properties.action == 'ATTACH':
            return 'Attach the placed hair to the full head in its rest pose; roots then follow head movement and deformation'
        if properties.action == 'USE_CHARACTER_HEAD':
            return 'Use a live grooming reference for the accepted character; reuse existing linked hair and growth regions'
        if properties.action == 'SEED':
            return 'Create editable strands on this growth region and attach them to the full head'
        if properties.action == 'MIX':
            return 'Combine selected grooms while retaining their group settings; keep the original grooms'
        if properties.action == 'SIMPLIFY':
            return 'Reduce each group and its guides separately; retain at least one strand per population and keep linked copies independent'
        if properties.action == 'PREVIEW':
            return 'Replace Blender preview shading on this groom; keep geometry, linked copies and Unreal exchange material settings'
        if properties.action in {'SELECT_GUIDES', 'SELECT_STRANDS', 'SELECT_ALL'}:
            return 'Select whole guides, render strands or all hair for native brushes; keep guide flags, groups and attachment data'
        if properties.action=='EQUIP_CREATOR':
            return 'Equip the last delivered groom in the chosen Creator section; preserve the selections in other sections'
        if properties.action=='CHARACTER_TARGET':
            return 'Prepare a persistent binding source from this project\'s linked MetaHuman Creator character; reuse its unchanged rest head without selecting a mesh'
        if properties.action=='PUSH':
            return 'Send hair, binding and wardrobe to Unreal; prepare the linked Creator head automatically when no Face target is set'
        if properties.action=='APPLY_ACTOR':
            return 'Apply the last delivered groom and binding to one matching actor selected in Unreal; keep existing hair components and save the level afterward'
        if properties.action=='FIT_LASHES':
            return 'Fit a copy of the selected Unreal character lash groom to anatomical lid rims and preview it; keep the original groom and artist controls'
        if properties.action=='KEEP_LASH_FIT':return 'Save the reviewed lash groom and binding in the character blueprint'
        if properties.action=='REVERT_LASH_FIT':return 'Restore the preceding actor lashes and any blueprint change made by this repair; preserve later artist edits'
        return 'Run the selected grooming action'

    def execute(self, context):
        s, g = (
            context.scene.cm_settings,
            context.scene.cm_groom,
        )
        try:
            context.scene.cm_features.last_action_error = False
            if self.action in {'FIT_LASHES','KEEP_LASH_FIT','REVERT_LASH_FIT'}:
                from . import groom_jobs
                groom_jobs.start(context,self.action);return {'FINISHED'}
            from . import paint_tools
            if paint_tools.running():
                raise ValueError('Apply or cancel Smart Paint before editing a groom')
            from . import groom_jobs
            if self.action == 'RESUME_EXCHANGE':
                groom_jobs.resume(context)
                return {'FINISHED'}
            if self.action == 'APPLY_EXCHANGE':
                groom_jobs.apply_import(context)
                return {'FINISHED'}
            if self.action == "FINISH":
                if context.mode != 'OBJECT':
                    bpy.ops.object.mode_set(mode='OBJECT')
                s.status = 'Scalp editing finished. Create hair on the adjusted region.' if g.scalp_editing else "Hair editing finished. Export or push the groom when ready."
                g.scalp_editing = False
                return {'FINISHED'}
            if self.action in {'SELECT_GUIDES', 'SELECT_STRANDS', 'SELECT_ALL'}:
                total, count = select_population(context, g.groom, self.action.removeprefix('SELECT_'))
                s.status = 'Selected %d of %d hair curves. Native brushes affect the selection.' % (total, count)
                return {'FINISHED'}
            if context.mode != 'OBJECT':
                raise ValueError("Finish the active edit before another groom action")
            if self.action=='USE_CHARACTER_HEAD':
                head_setup_idle(context)
                if s.part not in {'HEAD','FULL_BODY'}:
                    raise ValueError('Use an accepted head or full character for grooming')
                head=s.result
                m.check_mesh(head,'Accepted character head')
                if context.scene.cm_features.expression_active:
                    raise ValueError('Return to Neutral before creating a grooming head')
                from . import groom_follow
                if g.main_scalp and groom_follow.modifier(g.main_scalp):
                    groom_follow.accepted_changed(s, context)
                    if groom_follow.source(g.main_scalp) == head:
                        focus_groom(context)
                        s.status = 'Using the linked character head. Existing hair and edited growth regions are kept.'
                        return {'FINISHED'}
                setup_head(context,reference_head(context,head,follow=True))
                s.status='Head and four growth regions follow this character live. ' + ('Push will prepare the linked Creator head.' if context.scene.cm_features.unreal_character else 'Choose the matching Face mesh in Unreal before Push.')
                return {'FINISHED'}
            folder = Path(bpy.path.abspath(s.project_dir)) / "Grooms"
            folder.mkdir(parents=True, exist_ok=True)
            if self.action in {'GRAB_HEAD', 'SELECT_HEAD', 'SELECT_GROOM', 'SELECT_TARGET','APPLY_ACTOR','EQUIP_CREATOR','CHARACTER_TARGET'}:
                groom_jobs.start(context, self.action)
                return {'FINISHED'}
            elif self.action == "SCALP":
                base = (g.main_scalp or s.result) if g.scalp and g.scalp.get('cm_groom_region') else g.scalp or g.main_scalp or s.result
                g.scalp = create_scalp(context, base, g.region)
                setattr(g, 'scalp_' + g.region.lower(), g.scalp)
            elif self.action == 'EDIT_SCALP':
                m.check_mesh(g.scalp, 'Scalp region')
                focus_groom(context,g.scalp)
                m.activate(context, g.scalp)
                bpy.ops.object.mode_set(mode='SCULPT')
                g.scalp_editing = True
                s.status = 'Shape the growth region, then choose Finish Scalp.'
                return {'FINISHED'}
            elif self.action == "EMPTY":
                if not g.scalp:
                    g.scalp = s.result
                m.check_mesh(g.scalp, "Scalp")
                data = bpy.data.hair_curves.new(g.name)
                obj = bpy.data.objects.new(g.name, data)
                context.collection.objects.link(obj)
                data.surface = g.main_scalp or g.scalp
                data.surface_uv_map = (
                    data.surface.data.uv_layers.active.name if data.surface.data.uv_layers else ""
                )
                g.groom = obj
                m.activate(context, obj)
                focus_groom(context)
            elif self.action == "SEED":
                scalp = g.scalp or s.result
                g.groom = seed_curves(context, scalp, g)
                focus_groom(context)
            elif self.action == "EDIT":
                if not g.groom or g.groom.type != 'CURVES':
                    raise ValueError("Create or choose a Hair Curves groom first")
                focus_groom(context,g.groom)
                m.activate(context, g.groom)
                # Native brush strokes change the datablock directly. Detach a
                # linked copy before entering Sculpt so an earlier version stays intact.
                if g.groom.data.users > 1:
                    g.groom.data = g.groom.data.copy()
                bpy.ops.object.mode_set(mode='SCULPT_CURVES')
                s.status = "Sculpt the hair with Blender's brushes, then choose Finish Hair."
                return {'FINISHED'}
            elif self.action == 'PREVIEW':
                preview_material(g.groom, g.preview_color, g.preview_roughness,
                                 strand_color=g.preview_strand_color)
                s.status = 'Blender groom preview updated. Unreal material settings are kept; tune final hair shading in Unreal.'
                return {'FINISHED'}
            elif self.action == "MIX":
                objects = [o for o in context.selected_objects if o.type in {'CURVES', 'CURVE'}]
                if len(objects) < 2:
                    raise ValueError("Select at least two grooms to combine; originals are kept")
                payload = export_curves(context, objects)
                combined = import_curves(context, payload, g.name + '_Mix')
                try:
                    copy_mix_materials(context,objects,combined)
                    if g.main_scalp or s.result or g.scalp:
                        attach(context, combined, g.main_scalp or s.result or g.scalp)
                except Exception:
                    data = combined.data
                    bpy.data.objects.remove(combined, do_unlink=True)
                    if not data.users:
                        bpy.data.hair_curves.remove(data)
                    raise
                g.groom = combined
                for original in objects:
                    original.hide_set(True)
                    original.hide_render = True
                s.status = 'Combined %d grooms. Originals are kept in the Outliner.' % len(objects)
                return {'FINISHED'}
            elif self.action == "ATTACH":
                attach(context, g.groom, g.main_scalp or s.result or g.scalp)
                s.status = 'Attached '+str(len(g.groom.data.curves))+' strands. Hair follows head deformation and motion.'
                return {'FINISHED'}
            elif self.action == "SIMPLIFY":
                before, after = simplify_curves(g.groom, g.simplify_ratio)
                s.status = 'Simplified %s: %d to %d strands. Groups and guides retained; linked copies unchanged.' % (g.groom.name, before, after)
                return {'FINISHED'}
            elif self.action == "ALEMBIC":
                if not g.groom:
                    raise ValueError("Choose a groom")
                from . import groom_alembic
                objects = [obj for obj in context.selected_objects if obj.type in {'CURVE', 'CURVES'}] or [g.groom]
                groom_alembic.export(context, objects, folder / (g.name + '.abc'))
            elif self.action in {"EXPORT", "PUSH"}:
                objects = [
                    o for o in context.selected_objects if o.type in {"CURVE", "CURVES"}
                ] or [g.groom]
                payload = export_curves(context, objects)
                if self.action == 'PUSH':
                    groom_jobs.start(context, self.action, payload)
                    return {'FINISHED'}
                path = folder / (g.name + ".json")
                path.write_text(json.dumps(payload))
            elif self.action == "GRAB":
                groom_jobs.start(context, self.action)
                return {'FINISHED'}
            else:
                raise ValueError("Unknown groom action")
            s.status = "Groom action complete: " + self.action
            return {"FINISHED"}
        except Exception as exc:
            context.scene.cm_features.last_action_error = True
            self.report({"ERROR"}, str(exc))
            s.status = str(exc)
            return {"CANCELLED"}


def draw_lash_fit(layout,context):
    from . import groom_jobs,unreal_jobs
    g=context.scene.cm_groom;header,box=layout.panel('cm_native_lash_fit',default_closed=True);header.label(text='Repair Unreal Eyelashes',icon='OUTLINER_OB_CURVES')
    if box is None:return
    box.label(text='Select the character actor in Unreal.')
    controls=box.column();controls.enabled=not groom_jobs.running() and not unreal_jobs.running()
    controls.prop(g,'lash_rebuild_guides')
    controls.operator('cm.groom',text='Fit Selected Character Lashes',icon='MOD_CURVE').action='FIT_LASHES'
    controls.prop(g,'lash_fit_options',toggle=True)
    if g.lash_fit_options:
        controls.prop(g,'lash_maximum_offset');controls.prop(g,'lash_component')
    try:result=json.loads(g.lash_fit_result or '{}')
    except ValueError:result={}
    if result.get('mode')=='PREVIEW':
        row=controls.row();row.enabled=bool(result.get('can_save_blueprint'));row.operator('cm.groom',text='Keep in Character Blueprint',icon='CHECKMARK').action='KEEP_LASH_FIT'
    if result.get('mode') in {'PREVIEW','SAVED'}:controls.operator('cm.groom',text='Restore Previous Lashes',icon='LOOP_BACK').action='REVERT_LASH_FIT'


class CM_PT_groom(bpy.types.Panel):
    bl_label = "Groom • Create, Edit & Unreal"
    bl_idname = "CM_PT_groom"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "MetaHuman Maker"

    @classmethod
    def poll(cls, context):
        return (
            context.scene.cm_features.advanced_ui
            and context.scene.cm_features.ui_stage == "GROOM"
        )

    def draw(self, context):
        g = context.scene.cm_groom
        l = self.layout
        draw_lash_fit(l,context)
        for prop in ("groom", "scalp", "name", "region", "hair_length", "hair_count", "hair_seed"):
            l.prop(g, prop)
        for label, action in [
            ('Use Current Character Head','USE_CHARACTER_HEAD'),
            ('Get Selected Head from Unreal', 'SELECT_HEAD'),
            ('Get Selected Groom from Unreal', 'SELECT_GROOM'),
            ('Use Selected Face as Binding Target', 'SELECT_TARGET'),
            ("Grab MetaHuman Head from Unreal", "GRAB_HEAD"),
            ("Create Editable Region Scalp", "SCALP"),
            ("Create Empty Hair", "EMPTY"),
            ("Seed Editable Strands", "SEED"),
            ("Edit Hair", "EDIT"),
            ("Finish Hair", "FINISH"),
            ("Combine Selected Grooms", "MIX"),
            ("Attach to Scalp", "ATTACH"),
        ]:
            l.operator("cm.groom", text=label).action = action
        l.operator('cm.import_groom_head',text='Import Head (FBX)',icon='FILE_FOLDER')
        l.prop(g, "simplify_ratio")
        draw_population_selection(l)
        l.operator("cm.groom", text="Simplify Groom").action = "SIMPLIFY"
        for prop in ('preview_color', 'preview_roughness', 'preview_strand_color'):
            l.prop(g, prop)
        l.operator('cm.groom', text='Apply Preview Material').action = 'PREVIEW'
        l.prop(g, "unreal_asset")
        l.prop(g, "unreal_target")
        l.prop(g, 'wardrobe_slot')
        for label, action in [
            ("Grab Groom from Unreal", "GRAB"),
            ("Export Unreal Alembic", "ALEMBIC"),
            ("Export Strand JSON", "EXPORT"),
              ("Push Groom + Binding + Wardrobe", "PUSH"),
              ("Equip Groom in Creator", "EQUIP_CREATOR"),
              ("Use Linked Creator Head", "CHARACTER_TARGET"),
        ]:
            l.operator("cm.groom", text=label).action = action


CLASSES = (CM_GroomSettings, CM_OT_import_groom_head, CM_OT_groom, CM_PT_groom)


def register():
    for cls in CLASSES:
        bpy.utils.register_class(cls)
    bpy.types.Scene.cm_groom = PointerProperty(type=CM_GroomSettings)
    from . import groom_jobs
    groom_jobs.register()


def unregister():
    from . import groom_jobs
    groom_jobs.unregister()
    del bpy.types.Scene.cm_groom
    for cls in reversed(CLASSES):
        bpy.utils.unregister_class(cls)
