"""Preserve coarse smooth shading through SIMPLE tessellation and sculpt bending."""
import hashlib,json
import bpy
import numpy as np
from bpy.props import EnumProperty,BoolProperty
from . import mesh_ops as m

ENABLED='cm_simple_base_shading'
STAMP='cm_simple_shading_stamp'
SOURCE='cm_simple_shading_source'
HELPER='cm_simple_shading_helper'
MODIFIER='CM SIMPLE Shading After Deformation'


def managed_modifier(obj,modifier):
    helper=obj.get(HELPER)
    return bool(obj.get(ENABLED) and helper and helper.get('cm_simple_shading_owner')==obj and modifier.name==MODIFIER and modifier.type=='DATA_TRANSFER' and modifier.object==helper and modifier.use_loop_data and set(modifier.data_types_loops)=={'CUSTOM_NORMAL'} and modifier.loop_mapping=='TOPOLOGY' and not any(getattr(modifier,name,False) for name in ('use_vert_data','use_edge_data','use_poly_data')))


def blocking_modifiers(obj):
    return [modifier for modifier in obj.modifiers if not managed_modifier(obj,modifier)]


def detach_copy(obj,*,allow_missing_modifier=False):
    """Drop only copied shading links; never remove the previous layer's helper."""
    if not obj.get(ENABLED):return None
    helper=obj.get(HELPER);modifier=obj.modifiers.get(MODIFIER)
    if not helper or helper.get('cm_simple_shading_owner')==obj:raise ValueError('Expected copied shading owned by a previous layer')
    if (not modifier and not allow_missing_modifier) or (modifier and (modifier.type!='DATA_TRANSFER' or modifier.object!=helper)):raise ValueError('Copied sculpt shading dependency changed; previous layer retained')
    source=obj.get(SOURCE,'GEOMETRY')
    if modifier:obj.modifiers.remove(modifier)
    for key in (ENABLED,STAMP,HELPER,SOURCE):
        if key in obj:del obj[key]
    return source


def unit(values):
    lengths=np.linalg.norm(values,axis=1,keepdims=True)
    if np.any(lengths<1e-12):raise ValueError('Cannot transfer shading through a collapsed or isolated surface')
    return values/lengths


def corner_order(dense,target,mapping):
    """Map loops by exact oriented faces, retaining split/hard-edge shading."""
    from . import subdivision_order
    a,b=subdivision_order.from_mesh(dense),subdivision_order.from_mesh(target);result=np.full(len(b['loops']),-1,dtype=np.int64)
    for size in np.unique(b['sizes']):
        sa=a['starts'][a['sizes']==size];sb=b['starts'][b['sizes']==size]
        va=a['loops'][sa[:,None]+np.arange(size)];vb=mapping[b['loops'][sb[:,None]+np.arange(size)]]
        ra,rb=np.argmin(va,axis=1),np.argmin(vb,axis=1)
        def rows(values,rotation):
            canonical=np.take_along_axis(values,(rotation[:,None]+np.arange(size))%size,axis=1)
            return np.ascontiguousarray(canonical,dtype='<i8').view(np.dtype((np.void,8*int(size)))).ravel()
        ka,kb=rows(va,ra),rows(vb,rb);ia,ib=np.argsort(ka),np.argsort(kb)
        if not np.array_equal(ka[ia],kb[ib]) or np.any(ka[ia][1:]==ka[ia][:-1]):raise ValueError('Ambiguous oriented sculpt faces; split shading retained')
        offsets=(np.arange(size)[None,:]-rb[ib,None]+ra[ia,None])%size
        result[(sb[ib,None]+np.arange(size)).ravel()]=(sa[ia,None]+offsets).ravel()
    if np.any(result<0):raise ValueError('Incomplete sculpt corner correspondence')
    return result


def signature(context,base,detail):
    from . import map_provenance,detail_layers
    points=m.evaluated_coordinates(context,detail)
    reference=detail_layers.read_reference(detail)
    if reference is None:raise ValueError('This sculpt has no bound subdivision reference')
    digest=hashlib.sha256(points.tobytes()+reference.tobytes())
    digest.update(json.dumps(['SIMPLE_BEND_V1',detail.get(SOURCE,'GEOMETRY'),map_provenance.mesh_digest(context,base),m.topology(detail),detail_layers.steps_for(detail)]).encode())
    return digest.hexdigest()


def field(context,base,detail,*,source='GEOMETRY'):
    from . import detail_layers,subdivision_order
    if detail_layers.subdivision_type(detail)!='SIMPLE' or detail.get('cm_detail_source')!=base:raise ValueError('Use a bound SIMPLE sculpt for base shading')
    if any(mod.show_viewport or mod.show_render for mod in blocking_modifiers(detail)):raise ValueError('Apply or disable sculpt modifiers before transferring base shading')
    reference=detail_layers.read_reference(detail)
    if reference is None:raise ValueError('This sculpt has no bound subdivision reference')
    current=m.to_local(detail,m.evaluated_coordinates(context,detail))
    if current.shape!=reference.shape:raise ValueError('Sculpt topology changed; shading retained')
    evaluated=base.evaluated_get(context.evaluated_depsgraph_get());mesh=evaluated.to_mesh()
    try:
        positions=np.empty((len(mesh.vertices),3));mesh.vertices.foreach_get('co',positions.ravel());faces=[p.vertices[:] for p in mesh.polygons]
        if source=='GEOMETRY':
            if any(not p.use_smooth for p in mesh.polygons):raise ValueError('Use Base Render Shading to preserve hard edges on the base')
            values=np.empty_like(positions);mesh.vertices.foreach_get('normal',values.ravel());ids=np.empty(len(mesh.loops),dtype=np.int32);mesh.loops.foreach_get('vertex_index',ids);normals=values[ids]
        elif source=='RENDER':
            normals=np.empty((len(mesh.corner_normals),3));mesh.corner_normals.foreach_get('vector',normals.ravel())
        else:raise ValueError('Choose geometric or render base normals')
    finally:evaluated.to_mesh_clear()
    data=bpy.data.meshes.new('CM_ShadingReference');probe=None;dense=None;ev=None
    try:
        data.from_pydata(positions,[],faces);data.attributes.new('base_shading','FLOAT_VECTOR','CORNER').data.foreach_set('vector',normals.ravel())
        probe=bpy.data.objects.new('CM_ShadingReference',data);context.scene.collection.objects.link(probe)
        for step in detail_layers.steps_for(detail):
            modifier=probe.modifiers.new('Shading subdivision','SUBSURF');modifier.subdivision_type='SIMPLE';modifier.levels=step
        context.view_layer.update();ev=probe.evaluated_get(context.evaluated_depsgraph_get());dense=ev.to_mesh()
        mapping=subdivision_order.correspondence(subdivision_order.from_mesh(dense),subdivision_order.from_mesh(detail.data),len(positions))
        loops=corner_order(dense,detail.data,mapping)
        original=np.empty((len(dense.vertices),3));dense.vertices.foreach_get('co',original.ravel());normals=np.empty((len(dense.loops),3));dense.attributes['base_shading'].data.foreach_get('vector',normals.ravel());original=original[mapping];normals=normals[loops]
    finally:
        if dense is not None:ev.to_mesh_clear()
        if probe is not None:bpy.data.objects.remove(probe,do_unlink=True)
        if not data.users:bpy.data.meshes.remove(data)
    transform=np.linalg.inv(np.asarray(detail.matrix_world))@np.asarray(base.matrix_world)
    original=original@transform[:3,:3].T+transform[:3,3];normals=unit(normals@np.linalg.inv(transform[:3,:3]))
    tolerance=max(float(np.ptp(reference,axis=0).max())*1e-7,1e-9)
    if not np.allclose(original,reference,atol=tolerance,rtol=0):raise ValueError('The base no longer matches this sculpt reference; update the sculpt before transferring shading')
    detail.data.calc_loop_triangles();tri=np.asarray([t.vertices[:] for t in detail.data.loop_triangles])
    def geometry(points):
        corners=points[tri];area=np.cross(corners[:,1]-corners[:,0],corners[:,2]-corners[:,0]);values=np.zeros_like(points)
        for k in range(3):np.add.at(values,tri[:,k],area)
        return unit(values)
    old,new=geometry(original),geometry(current);cross=np.cross(old,new);dot=np.sum(old*new,axis=1)
    if np.any(dot<-.99):raise ValueError('Sculpt normals reverse across the reference; repair the fold before transferring shading')
    ids=np.empty(len(detail.data.loops),dtype=np.int32);detail.data.loops.foreach_get('vertex_index',ids);cross,dot=cross[ids],dot[ids]
    result=unit(normals+np.cross(cross,normals)+np.cross(cross,np.cross(cross,normals))/(1+dot[:,None]))
    if not np.isfinite(result).all():raise ValueError('Invalid sculpt shading; previous normals retained')
    return result


def apply(context,base,detail,*,replace_custom=False,source='GEOMETRY'):
    if not detail:raise ValueError('Create a SIMPLE sculpt layer first')
    from . import resource_budget
    resource_budget.require('SIMPLE sculpt shading',len(detail.data.vertices)*640+256*1024**2)
    if not detail.get(ENABLED) and detail.data.has_custom_normals and not replace_custom:raise ValueError('The sculpt has authored normals. Explicitly choose Replace Custom Normals to archive and replace them')
    normals=field(context,base,detail,source=source);mesh=detail.data
    helper=detail.get(HELPER)
    if helper and (helper.get('cm_simple_shading_owner')!=detail or helper.type!='MESH'):raise ValueError('Sculpt shading helper ownership changed; work retained')
    modifier=detail.modifiers.get(MODIFIER)
    if modifier and (modifier.type!='DATA_TRANSFER' or modifier.object!=helper):raise ValueError('Sculpt shading modifier is not owned by this layer')
    # Custom normals are encoded relative to mesh geometry. Encode on evaluated
    # positions, then transfer AFTER shape keys, avoiding a second deformation.
    data=bpy.data.meshes.new('CM SIMPLE Evaluated Shading')
    try:
        current=m.to_local(detail,m.evaluated_coordinates(context,detail))
        data.from_pydata(current,[],[p.vertices[:] for p in mesh.polygons])
        smooth=np.empty(len(mesh.polygons),dtype=bool);mesh.polygons.foreach_get('use_smooth',smooth);data.polygons.foreach_set('use_smooth',smooth)
        data.normals_split_custom_set(normals);data.update()
    except BaseException:
        bpy.data.meshes.remove(data);raise
    if helper:
        if helper.name not in context.scene.objects:context.scene.collection.objects.link(helper)
        previous=helper.data;helper.data=data
        if not previous.users:bpy.data.meshes.remove(previous)
    else:
        helper=bpy.data.objects.new('CM SIMPLE Shading Source',data);context.scene.collection.objects.link(helper);helper['cm_simple_shading_owner']=detail;detail[HELPER]=helper
    helper.matrix_world=detail.matrix_world;helper.hide_render=True;helper.hide_select=True;helper.hide_set(True)
    modifier=modifier or detail.modifiers.new(MODIFIER,'DATA_TRANSFER');modifier.object=helper;modifier.use_loop_data=True;modifier.data_types_loops={'CUSTOM_NORMAL'};modifier.loop_mapping='TOPOLOGY';modifier.use_object_transform=True;modifier.show_viewport=True;modifier.show_render=True
    context.view_layer.update();detail[ENABLED]=True;detail[SOURCE]=source;detail[STAMP]=signature(context,base,detail)
    return len(normals)


def refresh(context,base,detail):
    if detail and detail.get(ENABLED):
        if context.scene.cm_features.add_pores:raise ValueError('Restore geometric sculpt shading before adding live pore displacement, or apply the pores before transferring base shading')
        modifier=detail.modifiers.get(MODIFIER)
        if not detail.get(HELPER) or not modifier or not modifier.show_viewport or not modifier.show_render or signature(context,base,detail)!=detail.get(STAMP):apply(context,base,detail,source=detail.get(SOURCE,'GEOMETRY'))


def restore(context,detail):
    if not detail or not detail.get(ENABLED):return
    helper=detail.get(HELPER);modifier=detail.modifiers.get(MODIFIER)
    if helper and helper.get('cm_simple_shading_owner')!=detail:raise ValueError('Sculpt shading helper ownership changed; work retained')
    if modifier:
        if modifier.type!='DATA_TRANSFER' or modifier.object!=helper:raise ValueError('Sculpt shading modifier ownership changed; work retained')
        detail.modifiers.remove(modifier)
    if helper:
        mesh=helper.data;bpy.data.objects.remove(helper,do_unlink=True)
        if not mesh.users:bpy.data.meshes.remove(mesh)
    for key in (ENABLED,STAMP,HELPER,SOURCE):
        if key in detail:del detail[key]
    context.view_layer.update()


class CM_OT_simple_shading(bpy.types.Operator):
    bl_idname='cm.simple_shading'
    bl_label='SIMPLE Sculpt Shading'
    bl_description='Interpolate base smooth normals and rotate them with sculpt bending; no coordinates change'
    bl_options={'REGISTER','UNDO'}
    mode:EnumProperty(items=[('BASE','Follow Base Shading','Preserve smooth base shading through SIMPLE subdivision'),('GEOMETRIC','Restore Sculpt Shading','Restore the original sculpt normal field')])
    replace_custom:BoolProperty(name='Replace Custom Normals',default=False)
    normal_source:EnumProperty(name='Base Normal Source',items=[('GEOMETRY','Base Geometry','Use smooth geometric base normals, excluding normal-transfer modifiers'),('RENDER','Base Render Shading','Preserve the evaluated base corner normals, including hard edges and authored normal transfers')],default='GEOMETRY')
    def execute(self,context):
        from . import bake_jobs,detail_jobs
        s,f=context.scene.cm_settings,context.scene.cm_features
        base,detail=(f.body_result,f.body_detail) if s.part=='FULL_BODY' and f.texture_part=='BODY' else (s.result,s.detail)
        try:
            if context.mode!='OBJECT' or bake_jobs.running() or detail_jobs.running():raise ValueError('Finish the active edit or bake before changing sculpt shading')
            if self.mode=='BASE':apply(context,base,detail,replace_custom=self.replace_custom,source=self.normal_source)
            else:restore(context,detail)
            s.status='Sculpt shading updated without moving vertices. Refresh after sculpt edits; final baking refreshes managed shading automatically.'
            return {'FINISHED'}
        except (ValueError,RuntimeError,np.linalg.LinAlgError) as exc:s.status=str(exc);self.report({'ERROR'},str(exc));return {'CANCELLED'}


def draw(layout,scene):
    s,f=scene.cm_settings,scene.cm_features;detail=f.body_detail if s.part=='FULL_BODY' and f.texture_part=='BODY' else s.detail
    if not detail or detail.get('cm_detail_subdivision_type')!='SIMPLE':return
    row=layout.row(align=True)
    row.operator('cm.simple_shading',text='Follow Base Geometry',depress=bool(detail.get(ENABLED)) and detail.get(SOURCE)=='GEOMETRY').mode='BASE'
    if detail.get(ENABLED):row.operator('cm.simple_shading',text='Restore Shading').mode='GEOMETRIC'
    op=layout.operator('cm.simple_shading',text='Use Base Render Shading',depress=bool(detail.get(ENABLED)) and detail.get(SOURCE)=='RENDER');op.mode='BASE';op.normal_source='RENDER'


CLASSES=(CM_OT_simple_shading,)
