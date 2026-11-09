"""Stage wrinkle sculpts on a changed identity without touching accepted layers."""

import numpy as np
from . import mesh_ops as m, wrinkles as w, detail_layers as dl


def points(data):
    result = np.empty((len(data), 3), dtype=np.float64)
    data.foreach_get('co', result.ravel())
    return result


def linear(obj):
    return np.stack([w._read_vector(obj, '.cm_wm_linear_' + str(row))
                     for row in range(3)], axis=1)


def prepare(context, s, f, old, slot, *, data=None):
    """Return a private replacement; caller owns publication or disposal.

    Retain individual relative keys, including edits to the posed Basis. Unpose
    each key, transport its neutral displacement through the identity change,
    and apply the new expression basis. Evaluated key strength is never baked
    into the transferred offsets. The old layer and slot pointers are read-only.
    """
    m.check_mesh(old, 'Retained wrinkle layer')
    keys = old.data.shape_keys
    if old.modifiers or not keys or not keys.use_relative:
        raise ValueError('Retained wrinkles require relative sculpt keys without modifiers')
    if keys.animation_data:
        raise ValueError('Keep the animated wrinkle source; automatic key animation transfer is not ready')
    if context.mode != 'OBJECT':
        raise ValueError('Finish sculpting before updating retained wrinkles')
    if slot not in {'WM1', 'WM2', 'WM3'}:
        raise ValueError('Unknown retained wrinkle slot')
    if s.detail and s.detail.get('cm_head_update_pending'):
        raise ValueError('Update the retained head sculpt before updating wrinkles')
    new = None
    try:
        native = bool(old.get('cm_wrinkle_native'))
        if native:
            from . import expressions
            data = expressions.bank(f) if data is None else data
            poses = [pose for pose in data['poses'] if pose['name'] == old.get('cm_wrinkle_pose')]
            if len(poses) != 1:
                raise ValueError('The retained wrinkle expression is absent from the native library')
            new = w.posed_layer(context, s, f, data, poses[0], slot)
            old_neutral = w._read_vector(old, '.cm_wm_neutral')
            old_posed = w._read_vector(old, '.cm_wm_posed')
            old_linear = linear(old)
        else:
            new = w.snapshot(context, s.detail or s.result, s.character_name + '_' + slot)
            old_neutral = w._read_vector(old,'.cm_wm_neutral') if old.data.attributes.get('.cm_wm_neutral') else points(keys.key_blocks[0].data)
            old_posed = old_neutral
            old_linear = None
            w._vector_attribute(new,'.cm_wm_neutral',points(new.data.vertices))
            w.seal_source(context,s,new)
        old_matrix, new_matrix = np.asarray(old.matrix_world), np.asarray(new.matrix_world)
        same_topology=m.topology(old)==m.topology(new)
        mapping=None if same_topology else w.uv_correspondence(old,new)
        def transfer(values):
            return w.transfer_uv(old,new,values,mapping=mapping)
        neutral_world = old_neutral @ old_matrix[:3, :3].T + old_matrix[:3, 3]
        target_neutral = (w._read_vector(new, '.cm_wm_neutral') @ new_matrix[:3, :3].T
                          + new_matrix[:3, 3]) if native else m.evaluated_coordinates(context, new)
        mapped_neutral = transfer(neutral_world)
        deformation = dl.transport(new, mapped_neutral, target_neutral)
        new_posed = w._read_vector(new, '.cm_wm_posed') if native else m.to_local(new, target_neutral)
        new_linear = linear(new) if native else None
        inverse_matrix = np.linalg.inv(new_matrix[:3, :3]).T
        # Drop only the freshly generated empty keys; the artist source stays intact.
        new.shape_key_clear()
        for source in keys.key_blocks:
            delta = points(source.data) - old_posed
            if native:
                delta = np.linalg.solve(old_linear.transpose(0, 2, 1), delta[..., None])[..., 0]
            delta = transfer(delta @ old_matrix[:3, :3].T)
            delta = np.einsum('ni,nij->nj', delta, deformation) @ inverse_matrix
            if native:
                delta = np.einsum('ni,nij->nj', delta, new_linear)
            output = new_posed + delta
            if not np.isfinite(output).all():
                raise ValueError('Retained wrinkle transfer produced invalid geometry')
            key = new.shape_key_add(name=source.name)
            key.data.foreach_set('co', output.ravel())
            for name in ('slider_min', 'slider_max', 'value', 'mute', 'interpolation'):
                setattr(key, name, getattr(source, name))
        for source, key in zip(keys.key_blocks, new.data.shape_keys.key_blocks):
            key.relative_key = new.data.shape_keys.key_blocks[source.relative_key.name]
        # Match the Basis in the raw mesh too, for downstream mesh snapshots.
        new.data.vertices.foreach_set('co', points(new.data.shape_keys.key_blocks[0].data).ravel())
        # Region protection and artist groups follow the same UV correspondence.
        mask = old.data.attributes.get('.sculpt_mask')
        if mask:
            values = np.empty(len(mask.data), dtype=np.float32)
            mask.data.foreach_get('value', values)
            mapped = transfer(values)
            target = new.data.attributes.get('.sculpt_mask') or new.data.attributes.new('.sculpt_mask', 'FLOAT', 'POINT')
            target.data.foreach_set('value', np.clip(mapped, 0, 1))
        memberships={group.index:([],[]) for group in old.vertex_groups}
        for vertex in old.data.vertices:
            for membership in vertex.groups:
                if membership.group in memberships:
                    indices,weights=memberships[membership.group]
                    indices.append(vertex.index);weights.append(membership.weight)
        # The snapshot owns its mesh. Clear inherited deform weights once,
        # rather than scanning a dense surface once per native bone group.
        new.vertex_groups.clear()
        for group in old.vertex_groups:
            indices,weights=memberships[group.index]
            if same_topology:
                mapped_indices=np.asarray(indices,dtype=int)
                mapped_weights=np.asarray(weights)
            elif indices:
                values=np.zeros(len(old.data.vertices))
                values[indices]=weights
                mapped=transfer(values)
                mapped_indices=np.flatnonzero(mapped>0)
                mapped_weights=mapped[mapped_indices]
            else:
                mapped_indices=np.empty(0,dtype=int)
                mapped_weights=np.empty(0)
            target = new.vertex_groups.new(name=group.name)
            for index,weight in zip(mapped_indices,mapped_weights):
                target.add([int(index)],float(max(0,min(weight,1))),'REPLACE')
            target.lock_weight = group.lock_weight
        for source, key in zip(keys.key_blocks, new.data.shape_keys.key_blocks):
            key.vertex_group = source.vertex_group
        new.active_shape_key_index = min(old.active_shape_key_index, len(keys.key_blocks) - 1)
        new['cm_role'], new['cm_wrinkle_slot'] = 'WRINKLE', slot
        new['cm_wrinkle_retarget_source'] = old.name
        new.hide_set(True)
        new.hide_render = True
        for child in new.children:
            child.hide_set(True)
            child.hide_render = True
        new.data.update()
        return new
    except Exception:
        if new:
            w.discard_layer(new)
        raise


def prepare_all(context, s, f, *, data=None):
    """Stage all assigned slots or dispose all staged replacements on failure."""
    plans = []
    try:
        for slot in ('WM1', 'WM2', 'WM3'):
            old = getattr(f, slot.lower())
            if old:
                plans.append((slot, old, prepare(context, s, f, old, slot, data=data)))
        return plans
    except Exception:
        for _, _, new in plans:
            w.discard_layer(new)
        raise
