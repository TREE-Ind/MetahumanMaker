"""Project permitted source skin onto a separate, shape-preserving sculpt layer."""
import hashlib,json,re
from contextlib import contextmanager,nullcontext
import bpy
import numpy as np
from mathutils.bvhtree import BVHTree
from . import mesh_ops as m,source_regions,region_projection as rp,scan_projection,detail_layers as dl
from .vertex_contract import fingerprint

SUPPORT='CM_SourceProjectionSupported'
ORIGIN='cm_region_projection_origin'


@contextmanager
def extension_transaction(obj):
    """Restore the applied layer if committing a calculated extension fails."""
    from .workflow import write_group
    names=('cm_scan_regions','cm_scan_applied',scan_projection.SIGNATURE,'cm_scan_mask_used','cm_scan_neck_fade','cm_scan_mask_dirty','cm_scan_original_source','cm_scan_region_signature','cm_regional_scan_report','cm_scan_region_schema')
    properties={name:obj[name] for name in names if name in obj};had_origin=ORIGIN in obj.data.attributes;origin=projection_origin(obj);support=rp.weights(obj,SUPPORT);key=obj.data.shape_keys.key_blocks['Scan Detail'];points=scan_projection.key_points(key);value=key.value
    try:yield
    except Exception:
        for name in names:
            if name in properties:obj[name]=properties[name]
            elif name in obj:del obj[name]
        attribute=obj.data.attributes.get(ORIGIN)
        if had_origin:attribute.data.foreach_set('vector',origin.astype(np.float32).ravel())
        elif attribute:obj.data.attributes.remove(attribute)
        write_group(obj,SUPPORT,support);key.data.foreach_set('co',points.ravel());key.value=value;obj.data.update()
        raise


def inherited_support(context,base,obj,support):
    """Interpolate a policy on an isolated copy using the layer's exact subdivision."""
    from .workflow import write_group
    mesh=base.data.copy();temporary=bpy.data.objects.new('CM_RegionalSupport',mesh);context.collection.objects.link(temporary);temporary.matrix_world=base.matrix_world.copy()
    try:
        temporary.shape_key_clear();mesh.vertices.foreach_set('co',m.to_local(base,m.evaluated_coordinates(context,base)).astype(np.float32).ravel())
        # Deform indices belong to the object, while weights live in its copied mesh.
        for group in base.vertex_groups:temporary.vertex_groups.new(name=group.name)
        write_group(temporary,SUPPORT,support)
        for step in dl.steps_for(obj):
            modifier=temporary.modifiers.new('Regional support','SUBSURF');modifier.subdivision_type=dl.subdivision_type(obj);modifier.levels=step;modifier.render_levels=step
        context.view_layer.update();evaluated=temporary.evaluated_get(context.evaluated_depsgraph_get());result=evaluated.to_mesh()
        try:
            if m.topology_data(result)!=m.topology(obj):raise ValueError('The sculpt topology changed; inherited source support cannot be reconstructed')
            index=temporary.vertex_groups[SUPPORT].index;values=np.zeros(len(result.vertices))
            for vertex in result.vertices:
                for item in vertex.groups:
                    if item.group==index:values[vertex.index]=item.weight;break
            return values
        finally:evaluated.to_mesh_clear()
    finally:
        bpy.data.objects.remove(temporary,do_unlink=True)
        if not mesh.users:bpy.data.meshes.remove(mesh)


def projection_origin(obj):
    attribute=obj.data.attributes.get(ORIGIN)
    if attribute:
        if attribute.domain!='POINT' or attribute.data_type!='FLOAT_VECTOR':raise ValueError('Invalid regional projection history; existing sculpt retained')
        points=np.empty((len(obj.data.vertices),3),dtype=np.float32);attribute.data.foreach_get('vector',points.ravel());return points.astype(float)
    return scan_projection.key_points(obj.data.shape_keys.key_blocks['Basis']).astype(float)


def check_surface(before,after,tri,weld):
    moved=np.any(before!=after,axis=1);active=np.any(moved[tri],axis=1)
    if not active.any():return dict(edited_faces=0,new_proper_contacts=0,orientations_preserved=True)
    p=before[tri[active]];q=after[tri[active]];old=np.cross(p[:,1]-p[:,0],p[:,2]-p[:,0]);new=np.cross(q[:,1]-q[:,0],q[:,2]-q[:,0]);area=np.linalg.norm(old,axis=1);size=np.linalg.norm(new,axis=1)
    if np.any(area<=0) or np.any(size<area*.1) or np.any(np.einsum('ij,ij->i',old,new)<=0):raise ValueError('Regional projection would fold or collapse the sculpt surface; projection retained')
    tolerance=float(np.linalg.norm(np.ptp(before,axis=0)))*1e-6
    initial=rp.crossings(before,tri,weld,pairs=True,tolerance=tolerance,active_faces=active);final=rp.crossings(after,tri,weld,pairs=True,tolerance=tolerance,active_faces=active)
    if final-initial:raise ValueError('Regional projection would add sculpt intersections; projection retained')
    return dict(edited_faces=int(active.sum()),new_proper_contacts=0,before_edited_contacts=len(initial),after_edited_contacts=len(final),minimum_area_ratio=float((size/area).min()),orientations_preserved=True)


def limit_projection(origin,proposal,basis,matrix,alpha,tri,weld):
    """Backtrack only physical vertex families incident to threatened faces."""
    world=origin@matrix[:3,:3].T+matrix[:3,3];p=world[tri];old=np.cross(p[:,1]-p[:,0],p[:,2]-p[:,0]);area=np.linalg.norm(old,axis=1)
    delta=proposal-origin;fraction=np.ones(int(weld.max())+1);iterations=0
    for iteration in range(33):
        output=(origin+delta*fraction[weld,None]).astype(np.float32).astype(float)
        visible=basis+(output-basis)*alpha[:,None];visible=visible@matrix[:3,:3].T+matrix[:3,3]
        p=visible[tri];cross=np.cross(p[:,1]-p[:,0],p[:,2]-p[:,0]);changed=np.any(visible!=world,axis=1);active=np.any(changed[tri],axis=1)
        invalid=active&((area<=0)|(np.linalg.norm(cross,axis=1)<area*.1)|(np.einsum('ij,ij->i',old,cross)<=0))
        if not invalid.any():return output,visible,dict(local_backtracking_iterations=iterations,limited_physical_vertices=int((fraction<1).sum()),minimum_local_fraction=float(fraction.min()))
        if iteration==32:break
        affected=np.unique(weld[tri[invalid]]);fraction[affected]*=.5;iterations+=1
    raise ValueError('No orientation-preserving regional projection was found; existing sculpt retained')


def source_data(context,source,name):
    source.data.calc_loop_triangles();tri=np.empty((len(source.data.loop_triangles),3),dtype=np.int64);source.data.loop_triangles.foreach_get('vertices',tri.ravel());points=m.evaluated_coordinates(context,source)
    evaluated=source.evaluated_get(context.evaluated_depsgraph_get());mesh=evaluated.to_mesh()
    try:
        if m.topology_data(mesh)!=m.topology(source):raise ValueError('Disable topology-changing source modifiers before regional detail projection')
    finally:evaluated.to_mesh_clear()
    allowed=np.all(rp.weights(source,name)[tri]>.25,axis=1);exact=source_regions.triangle_mask(source,name)
    if exact is not None:allowed&=exact
    ids=np.flatnonzero(allowed)
    if not len(ids):raise ValueError('Mark complete permitted source faces for '+name)
    return points,tri,ids


def support_from_fit(context,source,base,points,tri,faces):
    from . import source_budget
    raw=base.get(source_budget.PROPERTY)
    if not raw:raise ValueError('Fit this marked region with Accurate Fit before projecting source detail')
    ledger=json.loads(raw);contract=ledger['context'];geometry=contract['geometry'];rest=m.coordinates(base);tolerance=max(float(np.linalg.norm(np.ptp(rest,axis=0)))*1e-6,1e-9);_,weld=np.unique(np.rint(rest/tolerance).astype(np.int64),axis=0,return_inverse=True)
    expected=dict(source=fingerprint(points),source_triangles=fingerprint(tri),weld=fingerprint(weld),budget_target_matrix=fingerprint(np.asarray(base.matrix_world,dtype=float)))
    if any(geometry.get(k)!=v for k,v in expected.items()):raise ValueError('The source or fit frame changed; refit the marked region before projecting detail')
    entry=ledger['policies'].get(source_budget._key(faces))
    if entry is None:raise ValueError('Accurately fit this exact source-face region once before projecting detail')
    support=source_budget.observed_support(entry,contract['vertices'])
    if not support.any():raise ValueError('This region has distance limits but no verified source observations. Run Accurate Fit on this exact marked region before projecting detail')
    return support[weld].astype(float)


def signature(context,s,obj,region):
    name=rp.PREFIX+region;points,tri,faces=source_data(context,s.source,name)
    payload=dict(source=fingerprint(points),triangles=fingerprint(tri),faces=fingerprint(faces),region_weights=fingerprint(rp.weights(obj,name)),support=fingerprint(rp.weights(obj,SUPPORT)),pins=fingerprint(m.pin_weights(obj)),detail_matrix=fingerprint(np.asarray(obj.matrix_world)),accepted_shape=dl.signature(context,obj['cm_detail_source']))
    regions=json.loads(obj.get('cm_scan_regions','[]'))
    if len(regions)>1:
        payload['composed_regions']={item:dict(faces=fingerprint(source_data(context,s.source,rp.PREFIX+item)[2]),weights=fingerprint(rp.weights(obj,rp.PREFIX+item))) for item in regions}
        payload['projection_origin']=fingerprint(projection_origin(obj))
    if obj.get('cm_scan_region_schema',1)>=2:
        key=obj.data.shape_keys.key_blocks['Scan Detail']
        payload['parameters']=dict(mask=fingerprint(rp.weights(obj,'CM_DetailMask')),use_mask=context.scene.cm_features.use_detail_mask,strength=float(key.value),muted=key.mute,key_group=key.vertex_group)
    return hashlib.sha256(json.dumps(payload,sort_keys=True).encode()).hexdigest()


def validate(context,s,obj):
    if obj.get('cm_scan_original_source')!=s.source:raise ValueError('The original source changed; regenerate the regional source detail')
    region=json.loads(obj['cm_scan_regions'])[-1]
    if signature(context,s,obj,region)!=obj.get('cm_scan_region_signature'):raise ValueError('Source geometry or regional masks changed; regenerate source detail before applying')


def regenerate(context,s,f,obj,*,region=None,origin=None,support=None):
    extending=region is not None
    if not extending:scan_projection.check(obj)
    region=region or json.loads(obj['cm_scan_regions'])[-1];name=rp.PREFIX+region;points,tri,faces=source_data(context,s.source,name)
    local=projection_origin(obj) if origin is None else origin;matrix=np.asarray(obj.matrix_world);world=local@matrix[:3,:3].T+matrix[:3,3]
    basis=scan_projection.key_points(obj.data.shape_keys.key_blocks['Basis']).astype(float)
    # Use the original physical families even if an earlier region moved them.
    _,weld=np.unique(basis,axis=0,return_inverse=True);count=np.bincount(weld);unique=np.column_stack([np.bincount(weld,weights=world[:,axis])/count for axis in range(3)])
    if np.max(np.linalg.norm(world-unique[weld],axis=1))>1e-7:raise ValueError('The sculpt has split coincident copies; repair its seams before regional projection')
    weights=rp.weights(obj,name);weights*=(rp.weights(obj,SUPPORT) if support is None else support)>=1-1e-6
    if f.use_detail_mask:weights*=rp.weights(obj,'CM_DetailMask')
    weights*=1-m.pin_weights(obj)
    shared=np.ones(len(unique));np.minimum.at(shared,weld,weights)
    obj.data.calc_loop_triangles();target_tri=np.empty((len(obj.data.loop_triangles),3),dtype=np.int64);obj.data.loop_triangles.foreach_get('vertices',target_tri.ravel());physical=weld[target_tri];p=unique[physical];cross=np.cross(p[:,1]-p[:,0],p[:,2]-p[:,0]);normals=np.zeros_like(unique)
    for corner in range(3):np.add.at(normals,physical[:,corner],cross)
    normals/=np.maximum(np.linalg.norm(normals,axis=1)[:,None],1e-30)
    edges=np.unique(np.sort(np.vstack((physical[:,[0,1]],physical[:,[1,2]],physical[:,[2,0]])),axis=1),axis=0);length=np.linalg.norm(unique[edges[:,0]]-unique[edges[:,1]],axis=1);reach=np.zeros(len(unique))
    for corner in (0,1):np.maximum.at(reach,edges[:,corner],length*.5)
    reach=np.minimum(reach,float(np.linalg.norm(np.ptp(world,axis=0)))*s.max_distance);tree=BVHTree.FromPolygons(points.tolist(),tri[faces].tolist(),all_triangles=True);projected=unique.copy();used=[];offset=[];rejected=0
    for index in np.flatnonzero(shared>0):
        hit,normal,face,distance=tree.find_nearest(unique[index],float(reach[index]))
        if hit is None or float(np.asarray(normal)@normals[index])<.25:rejected+=1;continue
        projected[index]+=(np.asarray(hit)-unique[index])*shared[index];used.append(int(faces[face]));offset.append(float(distance))
    if not used:raise ValueError('No facing, source-supported samples are reachable in this region')
    output=m.to_local(obj,projected[weld]);output[shared[weld]==0]=local[shared[weld]==0]
    native=output.astype(np.float32).astype(float);alpha=np.full(len(local),float(s.detail_strength));key=obj.data.shape_keys.key_blocks['Scan Detail']
    if key.vertex_group:alpha*=rp.weights(obj,key.vertex_group)
    if key.mute:alpha[:]=0.
    output,visible,limiter=limit_projection(local,native,basis,matrix,alpha,target_tri,weld);native=output
    endpoint=check_surface(world,visible,target_tri,weld);endpoint.update(limiter)
    with extension_transaction(obj) if extending else nullcontext():
        if extending:
            from .workflow import write_group
            # All source queries and geometry checks finish before changing the layer.
            attribute=obj.data.attributes.get(ORIGIN) or obj.data.attributes.new(ORIGIN,'FLOAT_VECTOR','POINT');attribute.data.foreach_set('vector',local.astype(np.float32).ravel())
            write_group(obj,SUPPORT,support);obj['cm_scan_regions']=json.dumps(json.loads(obj['cm_scan_regions'])+[region]);obj['cm_scan_applied']=False
        obj['cm_scan_region_schema']=2
        # commit() exposes the requested strength; include it in the source stamp.
        scan_projection.commit(s,f,obj,output);cache=signature(context,s,obj,region)
        obj['cm_scan_original_source']=s.source;obj['cm_scan_region_signature']=cache
        report=dict(method='SOURCE_AUTHORITY_SIMPLE_DETAIL_PROJECTION_V2',region=region,source_faces=len(faces),projected_physical_vertices=len(used),unsupported_or_unreachable=rejected,maximum_projection_m=max(offset),mean_projection_m=float(np.mean(offset)),uv_copies_shared=True,base_conform_mesh_unchanged=True,source_observation_support_inherited=True,endpoint_geometry=endpoint)
        if extending:report['previous_sculpt_preserved_outside_region']=bool(np.array_equal(native[shared[weld]==0],local[shared[weld]==0]));report['composed_regions']=json.loads(obj['cm_scan_regions'])
        obj['cm_regional_scan_report']=json.dumps(report)
    s.status='Regional source detail ready. Review it, then Apply Projected Detail.'


class CM_OT_project_region_detail(bpy.types.Operator):
    bl_idname='cm.project_region_detail'
    bl_label='Project Region onto Sculpt Layer'
    bl_description='Create a shape-preserving LV1 sculpt layer and project only the marked, source-supported region; keeps the conform mesh unchanged'
    bl_options={'REGISTER','UNDO'}

    def execute(self,context):
        from . import textures,detail_jobs,bake_jobs,resource_budget
        s,f=context.scene.cm_settings,context.scene.cm_features;created=None;previous_stage=s.stage;previous_ui=f.ui_stage;previous_level=s.detail_levels;previous_active=context.active_object
        try:
            if context.mode!='OBJECT' or rp.running() or detail_jobs.running() or bake_jobs.running():raise ValueError('Finish the current edit or geometry/bake job before projecting detail')
            if context.active_object is None or context.active_object not in (s.result,f.body_result,s.detail,f.body_detail):raise ValueError('Select the accepted head, body or its sculpt layer')
            body=context.active_object in (f.body_result,f.body_detail) and f.body_result is not None
            region=re.sub(r'[^A-Za-z0-9_]+','_',f.projection_region.strip()).strip('_');name=rp.PREFIX+region
            if not region:raise ValueError('Choose a marked source/body region')
            with textures.skin_part(s,f,'BODY' if body else 'HEAD'):
                base=s.result;m.check_mesh(base,'Accepted result');m.check_mesh(s.source,'Original source')
                if not base.vertex_groups.get(name) or not s.source.vertex_groups.get(name):raise ValueError('Mark matching source and result regions first')
                if s.detail:
                    obj=s.detail;regions=json.loads(obj.get('cm_scan_regions','[]'))
                    if scan_projection.editable(obj) and regions and regions[-1]==region:regenerate(context,s,f,obj)
                    else:
                        if not regions or not obj.get('cm_scan_applied',False):raise ValueError('Apply the pending region before adding another region')
                        if region in regions:raise ValueError('This region is already applied; use the sculpt tools to refine it')
                        from . import simple_shading
                        if simple_shading.blocking_modifiers(obj) or scan_projection.shape_signature(obj)!=obj.get(scan_projection.SIGNATURE):raise ValueError('This layer has authored changes; they were retained. Use the sculpt workflow')
                        key=obj.data.shape_keys.key_blocks['Scan Detail']
                        if key.mute or key.vertex_group or abs(key.value-1)>1e-7 or abs(s.detail_strength-1)>1e-7:raise ValueError('Use full projected detail strength before composing another region')
                        if dl.reference_for_update(context,base,obj) is not None:raise ValueError('Update the sculpt to the accepted base before projecting another region')
                        validate(context,s,obj)
                        points,tri,faces=source_data(context,s.source,name);support=support_from_fit(context,s.source,base,points,tri,faces)
                        resource_budget.require('regional source detail',512*resource_budget.MIB+resource_budget.meshes([base,s.source],subdivide=True))
                        support=inherited_support(context,base,obj,support);origin=scan_projection.key_points(key).astype(float)
                        regenerate(context,s,f,obj,region=region,origin=origin,support=support)
                else:
                    points,tri,faces=source_data(context,s.source,name);support=support_from_fit(context,s.source,base,points,tri,faces)
                    resource_budget.require('regional source detail',512*resource_budget.MIB+resource_budget.meshes([base,s.source],subdivide=True));s.detail_levels=1
                    created=m.create_detail(context,s,project_source=False,subdivision_type='SIMPLE',vertex_groups={SUPPORT:support});created['cm_scan_regions']=json.dumps([region]);created['cm_scan_applied']=False;created[scan_projection.SIGNATURE]=scan_projection.shape_signature(created)
                    regenerate(context,s,f,created)
            f.texture_part='BODY' if body else 'HEAD';f.ui_stage='DETAIL';s.stage='DETAIL'
            from . import compare_views
            session=compare_views.recover(context.window) if context.window else None
            if session:
                try:compare_views.members(context,session)
                except (ReferenceError,RuntimeError):pass
            return {'FINISHED'}
        except Exception as exc:
            if created:
                if f.body_detail==created:f.body_detail=None
                if s.detail==created:s.detail=None
                mesh=created.data;bpy.data.objects.remove(created,do_unlink=True)
                if not mesh.users:bpy.data.meshes.remove(mesh)
            if previous_active and previous_active.name in context.view_layer.objects:m.activate(context,previous_active)
            s.stage=previous_stage;f.ui_stage=previous_ui;s.status=str(exc);self.report({'ERROR'},str(exc));return {'CANCELLED'}
        finally:s.detail_levels=previous_level


CLASSES=(CM_OT_project_region_detail,)
