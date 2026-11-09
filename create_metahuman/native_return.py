"""Direct native UV correspondence, without landmarks or surface projection."""
from collections import defaultdict
from itertools import permutations
import json
import bpy
import numpy as np
from mathutils.kdtree import KDTree
from . import mesh_ops as m,geometry,surface_identity,workflow


def arrays(context,objects):
    uv=[];loops=[];vertices=[];points=[];materials=[];polygons=[];segments=[];lo=vo=po=0
    for obj in objects:
        mesh=obj.data
        if not mesh.uv_layers.active:raise ValueError('Native return requires the original active UV layout')
        xyz=m.evaluated_coordinates(context,obj)
        if len(xyz)!=len(mesh.vertices):raise ValueError('Apply topology-changing modifiers before native return')
        mesh.calc_loop_triangles();u=np.empty((len(mesh.loops),2),dtype=np.float32);mesh.uv_layers.active.data.foreach_get('uv',u.ravel())
        lt=np.array([t.loops[:] for t in mesh.loop_triangles],dtype=np.int64);vt=np.array([t.vertices[:] for t in mesh.loop_triangles],dtype=np.int64)
        uv.append(u);loops.append(lt+lo);vertices.append(vt+vo);points.append(xyz)
        materials.extend(mesh.polygons[t.polygon_index].material_index for t in mesh.loop_triangles)
        polygons.extend(t.polygon_index+po for t in mesh.loop_triangles)
        segments.append((vo,vo+len(xyz),po,po+len(mesh.polygons)));lo+=len(u);vo+=len(xyz);po+=len(mesh.polygons)
    uv,index=np.unique(np.concatenate(uv),axis=0,return_inverse=True)
    return dict(uv=uv,uv_tri=index[np.concatenate(loops)],tri=np.concatenate(vertices),points=np.concatenate(points),
        materials=np.asarray(materials),polygons=np.asarray(polygons),segments=segments)


def correspondence(context,source,references):
    from . import resource_budget
    resource_budget.require('native surface correspondence',512*1024**2)
    if not surface_identity.matches(source,references):raise ValueError('The complete native UV triangle layout does not match; use the ordinary wrap workflow')
    a=arrays(context,[source]);b=arrays(context,references)
    tree=KDTree(len(b['uv']))
    for i,point in enumerate(b['uv']):tree.insert((float(point[0]),float(point[1]),0.),i)
    tree.balance();mapping=np.array([tree.find((float(p[0]),float(p[1]),0.))[1] for p in a['uv']])
    au=mapping[a['uv_tri']];bu=b['uv_tri'];ao=np.argsort(au,axis=1);bo=np.argsort(bu,axis=1)
    at=np.take_along_axis(a['tri'],ao,axis=1);bt=np.take_along_axis(b['tri'],bo,axis=1)
    groups=[]
    for uvtri in (au,bu):
        rows=defaultdict(list)
        for i,key in enumerate(map(tuple,np.sort(uvtri,axis=1))):rows[key].append(i)
        groups.append(rows)
    anchors=[(groups[0][key][0],ids[0]) for key,ids in groups[1].items() if len(ids)==1 and len(set(key))==3]
    if len(anchors)<3:raise ValueError('The UV layout has no unambiguous native alignment anchors')
    anchors=np.asarray(anchors)[::max(1,len(anchors)//4096)]
    rotation,scale,translation=geometry.similarity(b['points'][bt[anchors[:,1]]].reshape(-1,3),a['points'][at[anchors[:,0]]].reshape(-1,3))
    predicted=b['points']@rotation*scale+translation;tri_map=np.full(len(bt),-1,dtype=np.int64);deferred=[]
    tolerance=max(float(np.linalg.norm(np.ptp(a['points'],axis=0)))*1e-6,1e-8)
    for key,reference_ids in groups[1].items():
        source_ids=groups[0][key]
        if len(reference_ids)!=len(source_ids):raise ValueError('Native UV triangle multiplicities changed')
        if len(set(key))!=3:
            if len(source_ids)!=1:raise ValueError('Overlapping degenerate UV triangles need separate native parts')
            deferred.append((reference_ids[0],source_ids[0],key));continue
        if len(source_ids)==1:tri_map[reference_ids[0]]=source_ids[0];continue
        if len(source_ids)>7:raise ValueError('Too many overlapping native UV surfaces; import separate native parts')
        source_xyz=a['points'][at[source_ids]];expected=predicted[bt[reference_ids]]
        costs=np.sum((expected[:,None,:,:]-source_xyz[None,:,:,:])**2,axis=(2,3))
        choices=sorted((float(sum(costs[i,j] for i,j in enumerate(order))),order) for order in permutations(range(len(source_ids))))
        if len(choices)>1 and choices[1][0]-choices[0][0]<=tolerance*tolerance:
            if not np.all(np.linalg.norm(source_xyz-source_xyz[0],axis=2)<=tolerance):raise ValueError('Overlapping native UV surfaces have ambiguous anatomical positions')
        for i,j in enumerate(choices[0][1]):tri_map[reference_ids[i]]=source_ids[j]
    result=np.full(len(b['points']),-1,dtype=np.int64);maximum=0.
    valid=np.flatnonzero(tri_map>=0)
    for target,original in zip(bt[valid].ravel(),at[tri_map[valid]].ravel()):
        if result[target]>=0:
            delta=float(np.linalg.norm(a['points'][result[target]]-a['points'][original]));maximum=max(maximum,delta)
            if delta>tolerance:raise ValueError('Native UV correspondence is inconsistent across shared vertices; source retained')
        else:result[target]=original
    # A few native triangles have two equal UV corners. Resolve their corner
    # order from the already-proven neighboring vertices, never spatial guesses.
    for reference_id,source_id,key in deferred:
        target=bt[reference_id];original=at[source_id];known=result[target]>=0;choices=[]
        for order in permutations(range(3)):
            if tuple(np.asarray(key)[list(order)])!=key:continue
            candidate=original[list(order)]
            if not np.any(known) or np.any(np.linalg.norm(a['points'][candidate[known]]-a['points'][result[target[known]]],axis=1)>tolerance):continue
            choices.append(candidate)
        if not choices or any(np.any(np.linalg.norm(a['points'][candidate]-a['points'][choices[0]],axis=1)>tolerance) for candidate in choices[1:]):raise ValueError('Degenerate UV corners cannot be proved by neighboring native vertices')
        at[source_id]=choices[0];tri_map[reference_id]=source_id
        for target_id,source_vertex in zip(target,choices[0]):
            if result[target_id]<0:result[target_id]=source_vertex
    if np.any(result<0):raise ValueError('Native return cannot map loose or unused vertices')
    material=np.full(sum(len(o.data.polygons) for o in references),-1,dtype=np.int32)
    for poly,slot in zip(b['polygons'],a['materials'][tri_map]):
        if material[poly]>=0 and material[poly]!=slot:raise ValueError('Source material boundaries cut through native polygons; export compatible material sections')
        material[poly]=slot
    return [dict(coordinates=a['points'][result[start:end]],materials=material[ps:pe]) for start,end,ps,pe in b['segments']],dict(method='NATIVE_UV_TRIANGLE_CORRESPONDENCE',source_vertices=len(a['points']),max_split_difference=maximum)


def conform(context):
    from . import character_project
    s,f=context.scene.cm_settings,context.scene.cm_features;character_project.preflight(context,s,f)
    m.check_inputs(s)
    if s.part not in {'HEAD','FULL_BODY'}:raise ValueError('Native return supports head and full-body imports')
    roles=surface_identity.reference_parts(s.source,s,f)
    if not roles:raise ValueError('The complete native UV triangle layout does not match; use the ordinary wrap workflow')
    references=[o for _,o in roles]
    plans,report=correspondence(context,s.source,references)
    report['source_parts']=[role for role,_ in roles]
    report['prepared_accessories']=[role for role in ('eye_left','eye_right','teeth') if getattr(f,role) and role not in report['source_parts']]
    state={name:getattr(f,name) for name in ('head_result','body_result','eye_left','eye_right','teeth','body_detail','wm1','wm2','wm3')}
    previous=s.result;detail=s.detail;stage=s.stage;ui=f.ui_stage;created=[];hidden=[]
    try:
        for (role,reference),plan in zip(roles,plans):
            result=m.duplicate(context,reference,s.character_name+'_Native_'+role);created.append((role,result))
            for name in list(result.keys()):
                if name.startswith('cm_follow_'):del result[name]
            result.shape_key_add(name='Basis');key=result.shape_key_add(name='Wrapped');key.data.foreach_set('co',m.to_local(result,plan['coordinates']).ravel());key.value=1
            result.data.materials.clear()
            for mat in s.source.data.materials:result.data.materials.append(mat)
            for poly,slot in zip(result.data.polygons,plan['materials']):poly.material_index=int(slot)
            result['cm_role']='RESULT';result['cm_part']='BODY' if role=='body' else 'HEAD';result['cm_template_topology']=m.topology(reference);result['cm_native_return']=json.dumps(report)
            from .guided import signature
            result['cm_source_signature']=signature(s.source);result['cm_template_signature']=signature(reference)
            if role=='body':result['cm_body_source']='SCAN'
            actual=m.evaluated_coordinates(context,result)
            if np.max(np.linalg.norm(actual-plan['coordinates'],axis=1))>max(np.linalg.norm(np.ptp(actual,axis=0))*2e-6,1e-7):raise ValueError('Native result storage lost source positions')
        supplied={role for role,_ in created}
        missing=[role for role in ('eye_left','eye_right','teeth') if role not in supplied and getattr(f,role)]
        for role in missing:
            result=m.duplicate(context,getattr(f,role),s.character_name+'_Prepared_'+role);created.append((role,result))
            for name in list(result.keys()):
                if name.startswith('cm_follow_'):del result[name]
        outputs=dict(created);s.result=outputs['head'];f.head_result=s.result
        if 'body' in outputs:f.body_result=outputs['body']
        for role in ('eye_left','eye_right','teeth'):
            if role in outputs:setattr(f,role,outputs[role])
        # Missing anatomy follows the returned head from prepared templates;
        # anatomy actually present in the export retains its source positions.
        if missing:workflow.update_accessories(context,s,f,roles=tuple(missing))
        bound=tuple(role for role in ('eye_left','eye_right','teeth') if role in outputs)
        if bound:workflow.bind_accessory_fit(context,s,f,roles=bound)
        active={s.result,f.body_result,f.eye_left,f.eye_right,f.teeth}
        for obj in dict.fromkeys(o for o in [previous,detail,*state.values(),*references,s.source] if o and o not in active):
            hidden.append((obj,obj.hide_get(),obj.hide_render));obj.hide_set(True);obj.hide_render=True
        s.detail=None;f.body_detail=None
        for name in ('wm1','wm2','wm3'):setattr(f,name,None)
        s.stage='REFINE';f.ui_stage='REFINE';s.last_error=0.;m.activate(context,s.result)
        s.status='Native geometry conformed directly; no landmarks or projection. Imported materials retained.'
        if missing:s.status+=' Missing anatomy follows the prepared templates: '+', '.join(missing)+'.'
        if not bpy.app.background:
            try:
                from . import compare_views,guided
                session=compare_views.recover(context.window)
                if session:
                    compare_views.members(context,session);compare_views.fit_objects(context,session['areas'],list(outputs.values()),f.front_axis)
                    session['previous']=[compare_views.view_state(a) for a in session['areas']];session['layouts']=[compare_views.layout_state(a) for a in session['areas']]
                else:guided.focus(context,list(outputs.values()),f.front_axis)
            except (ValueError,RuntimeError):s.status+=' Use Frame Both to review the returned character.'
        return outputs,report
    except Exception:
        s.result=previous;s.detail=detail;s.stage=stage;f.ui_stage=ui
        for name,value in state.items():setattr(f,name,value)
        for obj,viewport,render in hidden:obj.hide_set(viewport);obj.hide_render=render
        for _,obj in created:
            mesh=obj.data;bpy.data.objects.remove(obj,do_unlink=True)
            if not mesh.users:bpy.data.meshes.remove(mesh)
        raise


class CM_OT_conform_native_surface(bpy.types.Operator):
    bl_idname='cm.conform_native_surface';bl_label='Conform Native MetaHuman';bl_options={'REGISTER','UNDO'}
    bl_description='Reuse complete native UV correspondence and source vertex positions without landmark detection or projection'
    def execute(self,context):
        try:conform(context);return {'FINISHED'}
        except Exception as exc:context.scene.cm_settings.status=str(exc);self.report({'ERROR'},str(exc));return {'CANCELLED'}


CLASSES=(CM_OT_conform_native_surface,)
