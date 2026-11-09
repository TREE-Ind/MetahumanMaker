"""Transactional native Hair Curves attachment and export root metadata."""
import bpy
import numpy as np
from . import mesh_ops as m


def attach(context,groom,scalp):
    m.check_mesh(scalp,'Full head scalp')
    if not groom or groom.type!='CURVES':raise ValueError('Choose a Hair Curves groom')
    if context.mode!='OBJECT':raise ValueError('Finish the active edit before attaching hair')
    if not scalp.data.uv_layers.active:raise ValueError('The full head needs a UV map for root attachment')
    if not groom.is_editable or not groom.data.is_editable or not scalp.is_editable:
        raise ValueError('Make the groom and head local before attaching')
    if not len(groom.data.curves):raise ValueError('Add strands before attaching this groom')
    if abs(groom.matrix_world.determinant())<1e-12 or abs(scalp.matrix_world.determinant())<1e-12:
        raise ValueError('The groom and head must have nonzero scale')
    parent=scalp
    while parent:
        if parent==groom:raise ValueError('The head cannot be a child of the groom being attached')
        parent=parent.parent
    values=np.empty(len(groom.data.attributes['position'].data)*3,dtype=np.float32)
    groom.data.attributes['position'].data.foreach_get('vector',values)
    if not np.isfinite(values).all():raise ValueError('Hair contains invalid point positions')
    previous=groom.data
    original_parent=groom.parent
    original_inverse=groom.matrix_parent_inverse.copy();original_world=groom.matrix_world.copy()
    rest=scalp.add_rest_position_attribute
    modifiers=set(groom.modifiers)
    # Work on a private copy: shared grooms stay independent and a native failure
    # cannot leave half-moved roots or partially replaced UV metadata.
    private=previous.copy();groom.data=private
    try:
        # Blender's native operator snaps in reference space, writes surface UVs,
        # sets the parent while keeping transforms, prepares rest_position and
        # ensures exactly one surface-deformation node without replacing artist
        # modifiers. Explicit context keeps the user's selection untouched.
        with context.temp_override(object=scalp,active_object=scalp,
                                   selected_objects=[groom],selected_editable_objects=[groom]):
            result=bpy.ops.curves.surface_set()
        if result!={'FINISHED'}:raise ValueError('Blender could not attach this groom to the head')
        uv=private.attributes.get('surface_uv_coordinate')
        if not uv or uv.data_type!='FLOAT2' or len(uv.data)!=len(private.curves):
            raise ValueError('Blender did not produce complete root attachment UVs')
        roots=np.empty((len(private.curves),2),dtype=np.float32);uv.data.foreach_get('vector',roots.ravel())
        if not np.isfinite(roots).all():raise ValueError('Root attachment produced invalid UVs')
        exported=private.attributes.get('groom_root_uv')
        if exported and (exported.data_type!='FLOAT_VECTOR' or exported.domain!='CURVE'):
            private.attributes.remove(exported);exported=None
        if not exported:exported=private.attributes.new('groom_root_uv','FLOAT_VECTOR','CURVE')
        exported.data.foreach_set('vector',np.column_stack((roots,np.zeros(len(roots),dtype=np.float32))).ravel())
        for modifier in groom.modifiers:
            if modifier not in modifiers and modifier.type=='NODES':modifier.name='CM_SurfaceBinding'
        groom['cm_attached_scalp']=scalp.name;groom['cm_binding_version']=2
        context.view_layer.update()
    except Exception:
        groom.data=previous;groom.parent=original_parent
        groom.matrix_parent_inverse=original_inverse;groom.matrix_world=original_world
        for modifier in list(groom.modifiers):
            if modifier not in modifiers:groom.modifiers.remove(modifier)
        scalp.add_rest_position_attribute=rest
        if not private.users:bpy.data.hair_curves.remove(private)
        raise
    # Retain the old datablock for Blender's undo history; no artist data is
    # removed during the operation. Unused copies disappear on normal reopening.
    return groom
