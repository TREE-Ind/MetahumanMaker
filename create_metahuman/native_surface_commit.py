"""Transactional, evaluated validation for artist/agent surface-fit commits.

Uses only Blender and NumPy. Optional external research solvers are not
required. Unchanged vertices retain their native key storage, and every
commit checks proper contacts over the complete target mesh.
"""
import numpy as np
from . import mesh_ops as m,workflow


def commit(context,target,name,planned,*,reference=None,movable=None,validator=None,property_updates=None,publish=True):
    from . import region_projection as rp
    if any(modifier.show_viewport and modifier.type not in {'WEIGHTED_NORMAL','NORMAL_EDIT','DATA_TRANSFER'} for modifier in target.modifiers):raise ValueError('Apply or disable geometry modifiers before committing a surface fit')
    updates={} if property_updates is None else dict(property_updates)
    old_properties={key:(key in target,target.get(key)) for key in updates}
    before=m.evaluated_coordinates(context,target).copy();planned=np.asarray(planned,dtype=float)
    if planned.shape!=before.shape or not np.isfinite(planned).all():raise ValueError('Invalid surface-fit output; previous geometry retained')
    if reference is not None and not np.array_equal(before,reference):raise ValueError('Surface changed before commit; previous geometry retained')
    changed=np.any(planned!=before,axis=1)
    if movable is not None:
        movable=np.asarray(movable,dtype=bool)
        if movable.shape!=(len(before),) or np.any(changed&~movable):raise ValueError('Surface fit changes protected vertices; previous geometry retained')
    fixed=~changed
    target.data.calc_loop_triangles();tri=np.array([t.vertices[:] for t in target.data.loop_triangles],dtype=np.int32)
    rest=m.coordinates(target);weld_tolerance=max(float(np.linalg.norm(np.ptp(rest,axis=0)))*1e-6,1e-9)
    _,weld=np.unique(np.rint(rest/weld_tolerance).astype(np.int64),axis=0,return_inverse=True)
    collision_tolerance=float(np.linalg.norm(np.ptp(before,axis=0)))*1e-6
    previous_contacts=rp.crossings(before,tri,weld,pairs=True,tolerance=collision_tolerance)
    had_keys=bool(target.data.shape_keys);old=target.data.shape_keys.key_blocks.get(name) if had_keys else None
    saved=np.array([p.co[:] for p in old.data],dtype=np.float32) if old else None
    old_value=float(old.value) if old else 0.;key=None
    old_active=target.active_shape_key_index;old_only=target.show_only_shape_key
    visible=m.to_local(target,before)
    baseline=np.array([p.co[:] for p in old.relative_key.data]) if old else None
    old_delta=(saved.astype(float)-baseline)*old_value if old else np.zeros_like(before)
    def restore():
        nonlocal key
        for prop,(existed,value) in old_properties.items():
            if existed:target[prop]=value
            elif prop in target:del target[prop]
        if old is not None:
            old.data.foreach_set('co',saved.ravel());old.value=old_value
        elif key is not None and target.data.shape_keys:target.shape_key_remove(key);key=None
        if not had_keys and target.data.shape_keys:target.shape_key_clear()
        target.active_shape_key_index=old_active;target.show_only_shape_key=old_only
        target.data.update();context.view_layer.update()
        if not np.array_equal(m.evaluated_coordinates(context,target),before):raise RuntimeError('Surface-fit rollback could not restore the original evaluated surface')
    try:
        key=workflow.key_from_world(target,name,planned)
        if baseline is None:baseline=np.array([p.co[:] for p in key.relative_key.data])
        value=old_value if old_value!=0 else 1.
        co=(baseline+(m.to_local(target,planned)-visible+old_delta)/value).astype(np.float32)
        co[fixed]=saved[fixed] if old is not None and old_value!=0 else baseline[fixed]
        key.data.foreach_set('co',co.ravel());key.value=value;target.data.update();context.view_layer.update()
        actual=m.evaluated_coordinates(context,target)
        if not np.isfinite(actual).all() or not np.array_equal(actual[fixed],before[fixed]):raise ValueError('Stored fit moves protected vertices')
        # Account for local float32 storage and the active relative-key sum;
        # actual error is reported, never presented as bit-exact free motion.
        local_ulp=float(np.max(np.abs(np.spacing(visible.astype(np.float32))),initial=0))
        active=sum(k.value!=0 and not k.mute for k in target.data.shape_keys.key_blocks)
        linear=np.asarray(target.matrix_world,dtype=float)[:3,:3]
        tolerance=max(float(np.linalg.norm(np.ptp(before,axis=0)))*2e-7,local_ulp*float(np.linalg.norm(linear,ord=2))*(active+2),1e-10)
        error=float(np.linalg.norm(actual-planned,axis=1).max(initial=0))
        if error>tolerance:raise ValueError('Stored fit differs from the requested surface beyond native precision')
        if m.collapsed_faces(context,target):raise ValueError('Stored fit produces collapsed faces')
        expected=planned[tri];observed=actual[tri]
        expected_normal=np.cross(expected[:,1]-expected[:,0],expected[:,2]-expected[:,0]);actual_normal=np.cross(observed[:,1]-observed[:,0],observed[:,2]-observed[:,0])
        if np.any(np.einsum('ni,ni->n',expected_normal,actual_normal)<=0):raise ValueError('Stored fit reverses a requested face orientation')
        actual_contacts=rp.crossings(actual,tri,weld,pairs=True,tolerance=collision_tolerance)
        new_pairs=actual_contacts-previous_contacts
        if new_pairs:raise ValueError('Stored fit introduces '+str(len(new_pairs))+' new proper intersections')
        extra=validator(actual.copy()) if validator is not None else None
        result=dict(native_commit_checked=True,layer=name,layer_value=float(key.value),
            protected_vertices=int(fixed.sum()),protected_coordinates_exact=True,
            crossings_before=len(previous_contacts),crossings_after=len(actual_contacts),new_pairs=0,
            max_world_storage_error=error,world_storage_tolerance=tolerance,planned_face_orientation_preserved=True,collapsed_faces=0)
        if extra is not None:result['surface_validation']=extra
        result['published']=bool(publish)
        if publish:
            for prop,value in updates.items():target[prop]=value
            if updates:result['transactional_properties_updated']=sorted(updates)
        else:restore()
        return result
    except BaseException as exc:
        restore()
        if isinstance(exc,Exception):raise ValueError(str(exc)+'; previous geometry retained') from exc
        raise
