"""Keep a reused part's editable sculpt and explicit previous-level chain."""
import bpy
import numpy as np

from . import detail_layers, scan_projection, sculpt_archive
from . import mesh_ops as m


def inspect(context, scene, source, part):
    s, f = scene.cm_settings, scene.cm_features
    detail = (s.detail if part == 'HEAD' else f.body_detail or (s.detail if s.part == 'BODY' else None))
    role = 'HEAD_DETAIL' if part == 'HEAD' else 'BODY_DETAIL'
    if any(entry.get('role') == role and not entry.get('adopted')
           for entry in sculpt_archive.returned(scene).values()):
        raise ValueError('Adopt the returned sculpt in its saved project and save again before reusing this part')
    layers, current = [], set()
    item = detail
    while item:
        if item in layers or len(layers) >= 3:
            raise ValueError('Saved detail has a cyclic or unsupported previous-level chain')
        m.check_mesh(item, 'Saved sculpt detail')
        scan_projection.require_applied(item)
        from . import simple_shading
        if simple_shading.blocking_modifiers(item) or item.constraints or item.animation_data or (item.data.shape_keys and item.data.shape_keys.animation_data):
            raise ValueError('Apply live detail dependencies in the saved project before reusing this part')
        if item.get('cm_head_update_pending'):
            raise ValueError('Finish the saved detail update and save again before reusing this part')
        if item.get('cm_detail_source') not in (None, source):
            raise ValueError('Saved detail belongs to another accepted surface; update it before part reuse')
        for name, value in item.items():
            owned_helper=name==simple_shading.HELPER and item.get(simple_shading.ENABLED) and isinstance(value,bpy.types.Object) and value.get('cm_simple_shading_owner')==item
            if isinstance(value, bpy.types.Object) and not owned_helper and name not in {'cm_detail_source', 'cm_detail_previous'}:
                raise ValueError('Saved detail has an external object dependency: ' + name)
        if 'cm_detail_level' in item:
            detail_layers.steps_for(item)
            if int(item['cm_detail_level']) > 1 and not detail_layers.has_reference(item):
                raise ValueError('Update the legacy detail layer in its saved project and save again before part reuse')
            if item.get('cm_detail_source_topology', m.topology(source)) != m.topology(source):
                raise ValueError('Saved detail no longer matches its accepted part topology')
            if item.get('cm_detail_topology', m.topology(item)) != m.topology(item):
                raise ValueError('Saved sculpt topology changed; restore it before part reuse')
            matrix = np.asarray(item.get('cm_detail_reference_matrix', []))
            if (detail_layers.has_reference(item)
                    and item.get('cm_detail_source_shape') == detail_layers.signature(context, source)
                    and list(item.get('cm_detail_bound_steps', [])) == detail_layers.steps_for(item)
                    and matrix.shape == (16,) and np.array_equal(matrix, np.asarray(item.matrix_world).ravel())):
                current.add(item)
        layers.append(item)
        previous = item.get('cm_detail_previous')
        item = previous if isinstance(previous, bpy.types.Object) else None
    if detail and 'cm_detail_level' in detail and not detail_layers.has_reference(detail):
        # Resolve eligibility while the original scene still contains its history.
        detail_layers.reference_for_update(context, source, detail)
    return layers, current


def rebind(context, source, layers, current):
    context.view_layer.update()
    stamp = detail_layers.signature(context, source) if current else None
    for layer in layers:
        layer['cm_detail_source'] = source
        if layer in current:
            # Unit conversion changes world-space signatures, but neither the
            # stored local reference nor any authored shape-key coordinates.
            layer['cm_detail_source_shape'] = stamp
            layer['cm_detail_reference_matrix'] = np.asarray(layer.matrix_world).ravel().tolist()
        layer.hide_set(True)
        layer.hide_render = True
        from . import simple_shading
        if layer.get(simple_shading.ENABLED):simple_shading.apply(context,source,layer,source=layer.get(simple_shading.SOURCE,'GEOMETRY'))
