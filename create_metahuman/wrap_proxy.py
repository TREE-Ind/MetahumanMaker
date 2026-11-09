"""Replaceable solver-only geometry; the full source stays authoritative."""
import bpy
import numpy as np
from . import mesh_ops as m, resource_budget

SOURCE='cm_proxy_source_geometry'
GEOMETRY='cm_proxy_geometry'
OWNER='cm_solver_proxy_version'


def stamp(context,obj):
    from .scan_jobs import geometry_stamp
    return geometry_stamp(context,obj,transform=False)


def source(context,settings,features):
    original=settings.source
    proxy=features.proxy if features else None
    if not proxy or proxy.parent!=original:return original
    m.check_mesh(proxy,'Solver proxy')
    identity=np.eye(4)
    if (proxy.get(OWNER)!=1 or proxy.get(SOURCE)!=stamp(context,original)
            or proxy.get(GEOMETRY)!=stamp(context,proxy)
            or not np.allclose(np.asarray(proxy.matrix_basis),identity,rtol=0,atol=1e-7)
            or not np.allclose(np.asarray(proxy.matrix_parent_inverse),identity,rtol=0,atol=1e-7)):
        raise ValueError('Solver proxy is out of date or edited. Rebuild Solver Proxy, or choose Use Full Source before wrapping.')
    return proxy


def create(context,s,f):
    from . import character_project
    character_project.preflight(context,s,f)
    if context.mode!='OBJECT':raise ValueError('Return to Object Mode before rebuilding a solver proxy')
    m.check_mesh(s.source,'Source')
    resource_budget.require('build solver proxy',256*resource_budget.MIB+3*resource_budget.meshes([s.source]))
    previous=f.proxy
    reusable=(previous and previous.parent==s.source and previous.get(OWNER)==1
              and not previous.modifiers and not previous.constraints and not previous.vertex_groups
              and previous.get(GEOMETRY)==stamp(context,previous)
              and previous not in (s.source,s.template,s.result,s.detail,f.head_result,f.body_result,f.body_template,f.body_detail,f.eye_left,f.eye_right,f.teeth))
    before=set(bpy.data.meshes);candidate=None
    try:
        with m.selection_state(context):
            evaluated=s.source.evaluated_get(context.evaluated_depsgraph_get())
            data=bpy.data.meshes.new_from_object(evaluated,depsgraph=context.evaluated_depsgraph_get())
            candidate=bpy.data.objects.new(s.source.name+'_WrapProxy',data)
            context.collection.objects.link(candidate);candidate.parent=s.source
            candidate.matrix_parent_inverse.identity();candidate.matrix_basis.identity()
            m.activate(context,candidate)
            modifier=candidate.modifiers.new('Solver decimation','DECIMATE');modifier.ratio=f.decimate_ratio
            if bpy.ops.object.modifier_apply(modifier=modifier.name)!={'FINISHED'}:
                raise ValueError('Could not build the reduced solver mesh; the previous proxy is retained')
            if not candidate.data.polygons:raise ValueError('Proxy ratio removed all faces; increase Proxy Ratio')
            context.view_layer.update()
            source_stamp=stamp(context,s.source);proxy_stamp=stamp(context,candidate)
            if reusable:
                old_data=previous.data;previous.data=candidate.data
                previous.matrix_parent_inverse.identity();previous.matrix_basis.identity()
                bpy.data.objects.remove(candidate,do_unlink=True);candidate=None
                proxy=previous
                if not old_data.users:bpy.data.meshes.remove(old_data)
            else:
                proxy=candidate;candidate=None
            proxy[OWNER]=1;proxy[SOURCE]=source_stamp;proxy[GEOMETRY]=proxy_stamp
            proxy['cm_full_source']=s.source.name;proxy['cm_proxy_ratio']=float(f.decimate_ratio)
            proxy.hide_set(True);proxy.hide_render=True;f.proxy=proxy
    finally:
        if candidate:bpy.data.objects.remove(candidate,do_unlink=True)
        for data in set(bpy.data.meshes)-before:
            if not data.users:bpy.data.meshes.remove(data)
    s.status=f'Solver proxy ready: {len(proxy.data.vertices):,} vertices. Full source retained for textures and sculpt detail.'
    return proxy


def disable(context,s,f):
    from . import character_project
    character_project.preflight(context,s,f)
    f.proxy=None
    s.status='Wrapping uses the full source. Previous proxy objects remain available for reuse.'
