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


def remember_native_roots(context,obj,native_roots):
    """Retain exact native anchors across Blender's centimeter conversion."""
    original=np.asarray(native_roots,dtype=float);roots=original.astype(np.float32)
    if roots.shape!=(len(obj.data.curves),3) or not np.isfinite(roots).all():raise ValueError('Native groom roots must be finite float32 positions')
    unit=context.scene.unit_settings.scale_length*100
    offsets=np.empty(len(obj.data.curve_offset_data),np.int32);obj.data.curve_offset_data.foreach_get('value',offsets)
    local=np.empty((len(obj.data.points),3),np.float32);obj.data.position_data.foreach_get('vector',local.ravel())
    expected=(original/[unit,-unit,unit]).astype(np.float32)
    if not np.array_equal(local[offsets[:-1]],expected):raise ValueError('The imported groom roots changed before native anchor caching')
    for name,values in (('cm_native_root_position',roots),('cm_import_root_position',expected)):
        attribute=obj.data.attributes.get(name) or obj.data.attributes.new(name,'FLOAT_VECTOR','CURVE')
        if attribute.domain!='CURVE' or attribute.data_type!='FLOAT_VECTOR':raise ValueError('Native groom root cache attributes have an incompatible type')
        attribute.data.foreach_set('vector',values.ravel())
    obj['cm_native_root_unit']=float(unit);obj['cm_native_root_matrix']=json.dumps(np.asarray(obj.matrix_world).tolist())


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
            remember_native_roots(context,obj,[curve['points'][0] for curve in payload['curves']])
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
            if (obj.get('cm_native_root_unit')==unit and obj.get('cm_native_root_matrix')
                    and np.array_equal(matrix,np.asarray(json.loads(obj['cm_native_root_matrix'])))
                    and data.attributes.get('cm_native_root_position') and data.attributes.get('cm_import_root_position')):
                original=curve_attribute(data,'cm_import_root_position','CURVE',('FLOAT_VECTOR',),count,0,width=3)
                native=curve_attribute(data,'cm_native_root_position','CURVE',('FLOAT_VECTOR',),count,0,width=3)
                unchanged=np.all(points[offsets[:-1]]==original,axis=1)
                world[offsets[:-1][unchanged]]=native[unchanged]
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


def select_population(context, groom, population, group_id=None):
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
    if group_id is not None:
        if type(group_id) is not int or group_id < 0:
            raise ValueError('Choose a nonnegative native groom group ID')
        groups = curve_attribute(data, 'groom_group_id', 'CURVE', ('INT',), count, 0)
        selected &= groups == group_id
    if population != 'ALL':
        guides = curve_attribute(data, 'groom_guide', 'CURVE', ('INT', 'BOOLEAN'), count, 0)
        if np.any((guides != 0) & (guides != 1)):
            raise ValueError('Guide flags must be 0 or 1')
        selected &= guides == (1 if population == 'GUIDES' else 0)
    total = int(np.count_nonzero(selected))
    if not total:
        raise ValueError('No matching hair curves in this population' + (f' and group {group_id}' if group_id is not None else ''))
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


def remove_selected_curves(context, groom):
    """Remove whole selected curves on a private copy, retaining all survivors."""
    if context.mode != 'OBJECT':raise ValueError('Finish Hair before removing selected curves')
    if not groom or groom.type != 'CURVES' or not groom.is_editable or not groom.data.is_editable:
        raise ValueError('Choose an editable Hair Curves groom')
    data=groom.data;count=len(data.curves)
    selection=data.attributes.get('.selection')
    if not selection or selection.domain!='CURVE' or selection.data_type!='BOOLEAN':
        raise ValueError('Select whole hair curves with Guides, Strands or All first')
    selected=np.empty(count,bool);selection.data.foreach_get('value',selected)
    ids=np.flatnonzero(selected)
    if not len(ids):raise ValueError('Select curves to remove first')
    if len(ids)==count:raise ValueError('Keep at least one curve; choose a group or smaller selection')
    from . import resource_budget
    resource_budget.require('remove selected groom curves',len(data.points)*64+64*resource_budget.MIB)
    private=data.copy()
    try:
        private.remove_curves(indices=ids.tolist())
        if len(private.curves)!=count-len(ids):raise ValueError('Groom curve removal was incomplete')
        groom.data=private
    except Exception:
        bpy.data.hair_curves.remove(private);raise
    if not data.users:bpy.data.hair_curves.remove(data)
    context.view_layer.update()
    return dict(removed_curves=len(ids),retained_curves=len(private.curves))


def draw_population_selection(layout):
    g=bpy.context.scene.cm_groom
    layout.prop(g,'selection_group_enabled')
    if g.selection_group_enabled:
        layout.prop(g,'selection_group')
        groom=g.groom
        if groom and groom.type=='CURVES':
            attribute=groom.data.attributes.get('groom_group_id')
            if attribute and attribute.domain=='CURVE' and attribute.data_type=='INT':
                values=np.empty(len(groom.data.curves),np.int32);attribute.data.foreach_get('value',values)
                ids=np.unique(values)
                layout.label(text='Available: '+', '.join(str(v) for v in ids[:16])+(' ...' if len(ids)>16 else ''))
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


def draw_crown_shape(layout,context):
    g=context.scene.cm_groom
    if g.wardrobe_slot!='Hair':return
    header,body=layout.panel('cm_crown_shape',default_closed=True)
    header.label(text='Crown Shape')
    if body:
        body.enabled=context.mode=='OBJECT' and bool(g.groom and g.groom.type=='CURVES')
        body.prop(g,'crown_lift');body.prop(g,'crown_lean')
        if g.crown_lean:body.prop(g,'crown_direction')
        body.operator('cm.groom',text='Apply Crown Shape').action='CROWN_LIFT'
        body.label(text='Root anchors stay fixed. Undo reverses each application.')


def shape_crown_lift(context,groom,scale,*,lean=0.,direction=(0.,1.,0.)):
    """Shape crown lift and sweep in world space around exact strand roots."""
    if context.mode!='OBJECT':raise ValueError('Finish Hair before adjusting crown lift')
    if not groom or groom.type!='CURVES' or not len(groom.data.curves):raise ValueError('Choose an editable Hair Curves groom')
    if not np.isfinite(scale) or not .1<=scale<=2:raise ValueError('Choose crown lift between 0.1 and 2')
    direction=np.asarray(direction,dtype=float)
    if not np.isfinite(lean) or not -60<=lean<=60:raise ValueError('Choose crown lean between -60 and 60 degrees')
    if direction.shape!=(3,) or not np.isfinite(direction).all():raise ValueError('Choose a finite crown sweep direction')
    direction=direction.copy();direction[2]=0
    if lean and np.linalg.norm(direction)<1e-8:raise ValueError('Crown sweep needs a horizontal direction')
    if np.linalg.norm(direction)>0:direction/=np.linalg.norm(direction)
    if scale==1 and lean==0:return dict(changed_points=0,roots_preserved=True)
    from . import resource_budget
    data=groom.data
    resource_budget.require('shape groom crown',len(data.points)*160+len(data.curves)*96+64*resource_budget.MIB)
    offsets=np.empty(len(data.curve_offset_data),np.int32);data.curve_offset_data.foreach_get('value',offsets)
    lengths=np.diff(offsets)
    if np.any(lengths<2):raise ValueError('Every strand needs at least two points')
    points=np.empty((len(data.points),3),np.float64);data.position_data.foreach_get('vector',points.ravel())
    if not np.isfinite(points).all():raise ValueError('Groom contains non-finite positions')
    matrix=np.asarray(groom.matrix_world,dtype=np.float64)[:3,:3]
    inverse_up=np.linalg.solve(matrix,np.array([0.,0.,1.]))
    roots=points[offsets[:-1]];height=roots@matrix[2]
    span=float(np.ptp(height))
    if span<1e-8:raise ValueError('Crown shaping needs roots distributed over the scalp')
    weight=np.clip(((height-height.min())/span-.35)/.30,0,1);weight=weight*weight*(3-2*weight)
    indices=np.repeat(np.arange(len(lengths)),lengths)
    lift=(points-roots[indices])@matrix[2]
    candidate=points+((scale-1)*weight[indices]*lift)[:,None]*inverse_up
    # Only positive lift sweeps: hanging lengths retain their existing route.
    # The direction is horizontal in world space, independent of object scale.
    if lean:
        sweep=np.linalg.solve(matrix,direction)
        candidate+=(np.tan(np.deg2rad(lean))*weight[indices]*np.maximum(lift,0))[:,None]*sweep
    candidate[offsets[:-1]]=roots
    private=data.copy()
    try:
        private.position_data.foreach_set('vector',candidate.ravel())
        actual=np.empty_like(points);private.position_data.foreach_get('vector',actual.ravel())
        if not np.array_equal(actual[offsets[:-1]],roots):raise ValueError('Crown shaping changed a root anchor')
        changed=int(np.any(actual!=points,axis=1).sum())
        private['cm_crown_lift']=float(scale);private['cm_crown_lean']=float(lean);groom.data=private
    except Exception:
        bpy.data.hair_curves.remove(private);raise
    if not data.users:bpy.data.hair_curves.remove(data)
    return dict(changed_points=changed,roots_preserved=True,curves=len(lengths),crown_lift=float(scale),crown_lean=float(lean))


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
    replacement_component: StringProperty(name='Actor Groom Component',default='',description='Leave blank to use the Creator Section name, such as Hair; enter a custom existing actor component name when needed')
    replacement_origin: StringProperty(name='Replacement Review Folder',subtype='DIR_PATH')
    replacement_result: StringProperty()
    replacement_group_defaults: bpy.props.BoolProperty(name='Use Incoming Group Settings',default=False,description='Use the pushed groom asset settings instead of carrying component-level width, shadow and other group overrides from the existing hair')
    clearance_head: PointerProperty(name='Collider Head',type=bpy.types.Object)
    clearance_region: EnumProperty(name='Clearance Region',items=[('EARS','Ears','Native ear weights or CM_Ears'),('HEAD','Whole Head','All head surfaces'),('CUSTOM','Vertex Group','A weighted collision region')],default='EARS')
    clearance_group: StringProperty(name='Collision Group',default='CM_Ears')
    clearance_route: EnumProperty(name='Hair Route',items=[('SURFACE','Nearest Surface','Resolve sampled penetration against nearby surfaces'),('TUCK_EARS','Behind Ears','Guide strands beyond the rear ear surface along the selected head-local direction')],default='SURFACE')
    clearance_back_axis: FloatVectorProperty(name='Head-local Back Direction',size=3,default=(0,1,0),subtype='DIRECTION',description='Direction toward the back of the head in the collider object local space; standard native heads use +Y')
    clearance_mm: FloatProperty(name='Clearance (mm)',default=.12,min=0,max=5)
    clearance_root_mm: FloatProperty(name='Protect Root Length (mm)',default=1.,min=0,max=20)
    clearance_offset_mm: FloatProperty(name='Maximum Movement (mm)',default=30.,min=.1,max=100)
    clearance_length: FloatProperty(name='Length Change Limit',default=.15,min=0,max=1,subtype='FACTOR')
    clearance_accept_partial: bpy.props.BoolProperty(name='Accept Partial Repair',default=False,description='Explicitly apply a bounded proposal with remaining sampled violations for local visual review')
    clearance_receipt: StringProperty(subtype='FILE_PATH')
    clearance_status: StringProperty()
    clearance_report: StringProperty(options={'HIDDEN'})
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
    fit_lod_meshes: bpy.props.BoolProperty(name='Fit Cards to Strand Edits',default=True,
        description='Fit private imported card and helmet meshes to position-only strand edits using the configured numerical Python; failed fits retain reported strand LOD fallbacks')
    simulation_policy: EnumProperty(name='Push Physics',default='SOURCE',items=[
        ('SOURCE','Keep Captured Settings','Retain captured per-group physics and LOD overrides; newly authored hair uses Unreal defaults'),
        ('STATIC','Static','Disable simulation for every group and authored LOD on the new revision'),
        ('RODS','Groom Rods','Enable the built-in rods solver for every group and authored LOD on the new revision'),
        ('SPRINGS','Groom Springs','Enable the built-in springs solver for every group and authored LOD on the new revision')],
        description='Only the new pushed revision changes; existing source assets retain their physics. Review motion and collision in Unreal')
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
    crown_lift: FloatProperty(name='Crown Lift',default=1.,min=.1,max=2.,description='Scale upper scalp strands vertically around fixed roots; below 1 lowers a tall crown')
    crown_lean: FloatProperty(name='Crown Sweep',default=0.,min=-60.,max=60.,description='Sweep upward crown lengths horizontally, in degrees; roots and hanging lengths stay fixed')
    crown_direction: FloatVectorProperty(name='Sweep Direction',default=(0.,1.,0.),subtype='DIRECTION',description='World direction for crown sweep; vertical component is ignored. Positive Y is behind a standard imported head')
    selection_group_enabled: bpy.props.BoolProperty(name='Filter Selection by Group',default=False)
    selection_group: IntProperty(name='Groom Group ID',default=0,min=0,
        description='Select only this imported groom group; guides and render strands keep their separate populations')
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


class CM_OT_recover_groom_review(bpy.types.Operator,ImportHelper):
    """Link a saved groom review without changing Blender or Unreal assets"""
    bl_idname='cm.recover_groom_review'
    bl_label='Recover Saved Groom Review'
    filename_ext='.json'
    filter_glob: StringProperty(default='preview.json',options={'HIDDEN'})
    review: EnumProperty(name='Review',items=lambda self,context:getattr(self,'_recovery_items',[('BROWSE','Choose review file...','Browse for a saved preview.json')]))

    def invoke(self,context,event):
        from . import groom_jobs
        try:choices=groom_jobs.find_replacement_reviews(context)
        except OSError:choices=[]
        self._recovery_paths={str(i):item['path'] for i,item in enumerate(choices)}
        self._recovery_items=[(str(i),item['label'],item['description']) for i,item in enumerate(choices)]+[('BROWSE','Choose review file...','Browse for another saved preview.json')]
        if choices:
            self.filepath='';self.review='0';return context.window_manager.invoke_props_dialog(self,width=580)
        return self.browse(context)

    def browse(self,context):
        self.filepath=str(Path(bpy.path.abspath(context.scene.cm_settings.project_dir))/'Grooms/.exchange')+'/'
        context.window_manager.fileselect_add(self);return {'RUNNING_MODAL'}

    def draw(self,context):
        self.layout.prop(self,'review',text='Review')
        self.layout.label(text='Unreal validates the saved sources before Keep or Restore.')

    def execute(self,context):
        from . import groom_jobs
        try:
            if not self.filepath:
                if self.review=='BROWSE':return self.browse(context)
                self.filepath=self._recovery_paths[self.review]
            groom_jobs.recover_replacement(context,self.filepath);return {'FINISHED'}
        except (OSError,ValueError,TypeError,KeyError,AttributeError) as exc:
            self.report({'ERROR'},str(exc));return {'CANCELLED'}


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
        if properties.action == 'CROWN_LIFT':
            return 'Adjust crown height and horizontal sweep around fixed roots; keep widths, groups, guides and root UVs. Repeated applications accumulate; Undo restores the previous shape'
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
        if properties.action=='REMOVE_SELECTED':
            return 'Remove whole selected hair curves on a private data copy; linked grooms stay unchanged and Undo restores the curves'
        if properties.action=='KEEP_LASH_FIT':return 'Save the reviewed lash groom and binding in the character blueprint'
        if properties.action=='REVERT_LASH_FIT':return 'Restore the preceding actor lashes and any blueprint change made by this repair; preserve later artist edits'
        if properties.action=='PREVIEW_REPLACE':return 'Preview the last pushed groom on the selected character existing component; keep the preceding groom and do not add duplicate hair'
        if properties.action=='KEEP_REPLACE':return 'Save the reviewed groom in the existing character blueprint while preserving its material and group controls'
        if properties.action=='RESTORE_REPLACE':return 'Restore the recorded preceding actor groom and any blueprint change made by this review; retain later artist edits'
        return 'Run the selected grooming action'

    def execute(self, context):
        s, g = (
            context.scene.cm_settings,
            context.scene.cm_groom,
        )
        try:
            context.scene.cm_features.last_action_error = False
            from . import groom_clearance
            if self.action=='CANCEL_CLEARANCE':
                groom_clearance.cancel();g.clearance_status='Clearance cancelled/discarded; original hair retained.';return {'FINISHED'}
            if self.action in {'SOLVE_CLEARANCE','APPLY_CLEARANCE','REVERT_CLEARANCE','RESUME_CLEARANCE'}:
                {'SOLVE_CLEARANCE':groom_clearance.start,'APPLY_CLEARANCE':groom_clearance.apply,'REVERT_CLEARANCE':groom_clearance.revert,'RESUME_CLEARANCE':groom_clearance.resume}[self.action](context)
                return {'FINISHED'}
            if groom_clearance.pending():raise ValueError('Apply or discard the original clearance proposal before other groom operations')
            if self.action in {'FIT_LASHES','KEEP_LASH_FIT','REVERT_LASH_FIT'}:
                from . import groom_jobs
                groom_jobs.start(context,self.action);return {'FINISHED'}
            from . import paint_tools
            if paint_tools.running():
                raise ValueError('Apply or cancel Smart Paint before editing a groom')
            from . import groom_jobs
            if self.action in {'PREVIEW_REPLACE','KEEP_REPLACE','RESTORE_REPLACE','CLOSE_REPLACE'}:
                groom_jobs.start(context,self.action);return {'FINISHED'}
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
                total, count = select_population(context, g.groom, self.action.removeprefix('SELECT_'),
                                                g.selection_group if g.selection_group_enabled else None)
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
            elif self.action == 'REMOVE_SELECTED':
                result=remove_selected_curves(context,g.groom)
                s.status='Removed %d curves; retained %d. Undo restores the previous groom.' % (result['removed_curves'],result['retained_curves'])
            elif self.action == 'CROWN_LIFT':
                result=shape_crown_lift(context,g.groom,g.crown_lift,lean=g.crown_lean,direction=g.crown_direction)
                s.status='Crown shape adjusted: '+str(result['changed_points'])+' points. Root anchors retained; review silhouette and scalp clearance before Push.'
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


def draw_replacement(layout,context):
    from . import groom_jobs,unreal_jobs
    g=context.scene.cm_groom;record={};delivered={}
    if g.replacement_origin:
        file=Path(bpy.path.abspath(g.replacement_origin))/'preview.json'
        try:
            if file.is_file():record=json.loads(file.read_text())
        except (OSError,ValueError):pass
    try:delivered=json.loads(g.last_push_result or g.exchange_result or '{}')
    except ValueError:pass
    active=record.get('mode') in {'PREPARING','PREVIEW','SAVING','SAVED','RESTORING'}
    if not active:
        row=layout.row();row.enabled=not groom_jobs.running() and not unreal_jobs.running()
        row.operator('cm.recover_groom_review',text='Recover Saved Groom Review',icon='FILE_FOLDER')
    if not active and not (delivered.get('status')=='complete' and delivered.get('groom') and delivered.get('binding')):return
    box=layout.box();box.label(text='Replace Existing Character Groom',icon='OUTLINER_OB_CURVES')
    box.enabled=not groom_jobs.running() and not unreal_jobs.running()
    if active:
        box.label(text='Component: '+record.get('component','Hair'))
        if record.get('mode') in {'PREVIEW','SAVING'} and record.get('blueprint'):
            box.operator('cm.groom',text='Save Groom in Blueprint',icon='FILE_TICK').action='KEEP_REPLACE'
        box.operator('cm.groom',text='Restore Previous Groom',icon='LOOP_BACK').action='RESTORE_REPLACE'
        try:failed=json.loads(g.exchange_result or '{}').get('status')=='failed'
        except ValueError:failed=False
        if failed or g.show_asset_paths:
            box.operator('cm.groom',text='Close Review / Retain Current Edits',icon='CANCEL').action='CLOSE_REPLACE'
        if record.get('mode')=='SAVED':
            from .guided import text
            text(box,'Saved in blueprint. Use Equip Groom in Creator to update its wardrobe too.')
        if record.get('mode')=='SAVED' and delivered.get('status')=='complete' and delivered.get('binding') and (delivered.get('groom'),delivered.get('binding'))!=(record['after']['groom'],record['after']['binding']):
            box.operator('cm.groom',text='Preview Next Replacement',icon='HIDE_OFF').action='PREVIEW_REPLACE'
    else:
        box.label(text='Select one character actor in Unreal.')
        box.label(text='Replace: '+(g.replacement_component.strip() or g.wardrobe_slot))
        box.operator('cm.groom',text='Preview Replacement on Character',icon='HIDE_OFF').action='PREVIEW_REPLACE'
    if not active or (record.get('mode')=='SAVED' and (delivered.get('groom'),delivered.get('binding'))!=(record.get('after',{}).get('groom'),record.get('after',{}).get('binding'))):
        box.prop(g,'replacement_group_defaults')
    if g.show_asset_paths:
        box.prop(g,'replacement_component');box.prop(g,'replacement_origin')


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


def draw_clearance(layout,context):
    from . import groom_clearance
    g=context.scene.cm_groom
    box=layout.box();box.label(text='Surface Clearance · Local Proposal',icon='MOD_PHYSICS')
    options=box.column();options.enabled=not groom_clearance.pending()
    options.prop(g,'clearance_head');options.prop(g,'clearance_region')
    options.prop(g,'clearance_route')
    if g.clearance_route=='TUCK_EARS':options.prop(g,'clearance_back_axis')
    if g.clearance_region=='CUSTOM':options.prop(g,'clearance_group')
    for name in ('clearance_mm','clearance_root_mm','clearance_offset_mm','clearance_length'):options.prop(g,name)
    box.label(text='Review strand placement after Apply; Revert restores geometry.')
    if groom_clearance.ready():
        import json
        report=json.loads(g.clearance_report or '{}')
        if report.get('unresolved_curves'):box.prop(g,'clearance_accept_partial')
        row=box.row();row.enabled=not report.get('unresolved_curves') or g.clearance_accept_partial
        row.operator('cm.groom',text='Apply Local Proposal',icon='CHECKMARK').action='APPLY_CLEARANCE'
    elif not groom_clearance.pending():box.operator('cm.groom',text='Solve Clearance',icon='MOD_PHYSICS').action='SOLVE_CLEARANCE'
    if groom_clearance.pending():box.operator('cm.groom',text='Cancel / Discard',icon='CANCEL').action='CANCEL_CLEARANCE'
    elif g.clearance_receipt:box.operator('cm.groom',text='Resume Saved Proposal',icon='FILE_REFRESH').action='RESUME_CLEARANCE'
    if g.groom and g.groom.get(groom_clearance.LAYERS):
        row=box.row();row.enabled=not groom_clearance.pending()
        row.operator('cm.groom',text='Revert Clearance Geometry',icon='LOOP_BACK').action='REVERT_CLEARANCE'
    if g.clearance_status:
        from .guided import text
        text(box,g.clearance_status)


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
        draw_clearance(l,context)
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
        l.operator('cm.groom',text='Remove Selected Curves').action='REMOVE_SELECTED'
        l.operator("cm.groom", text="Simplify Groom").action = "SIMPLIFY"
        draw_crown_shape(l,context)
        for prop in ('preview_color', 'preview_roughness', 'preview_strand_color'):
            l.prop(g, prop)
        l.operator('cm.groom', text='Apply Preview Material').action = 'PREVIEW'
        l.prop(g, "unreal_asset")
        l.prop(g, "unreal_target")
        l.prop(g, 'wardrobe_slot')
        l.prop(g, 'fit_lod_meshes')
        l.prop(g, 'simulation_policy')
        l.label(text='Review hair motion in Unreal Simulate.')
        for label, action in [
            ("Grab Groom from Unreal", "GRAB"),
            ("Export Unreal Alembic", "ALEMBIC"),
            ("Export Strand JSON", "EXPORT"),
              ("Push Groom + Binding + Wardrobe", "PUSH"),
              ("Equip Groom in Creator", "EQUIP_CREATOR"),
              ("Use Linked Creator Head", "CHARACTER_TARGET"),
        ]:
            l.operator("cm.groom", text=label).action = action


        draw_replacement(l,context)


CLASSES = (CM_GroomSettings, CM_OT_import_groom_head, CM_OT_recover_groom_review, CM_OT_groom, CM_PT_groom)


def register():
    for cls in CLASSES:
        bpy.utils.register_class(cls)
    bpy.types.Scene.cm_groom = PointerProperty(type=CM_GroomSettings)
    from . import groom_jobs
    groom_jobs.register()
    from . import groom_clearance
    groom_clearance.register()


def unregister():
    from . import groom_clearance
    groom_clearance.unregister()
    from . import groom_jobs
    groom_jobs.unregister()
    del bpy.types.Scene.cm_groom
    for cls in reversed(CLASSES):
        bpy.utils.unregister_class(cls)
