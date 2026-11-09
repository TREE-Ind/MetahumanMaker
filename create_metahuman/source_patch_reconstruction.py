"""Reparameterize a bounded folded skin patch onto an observed source surface.

Unlike depth-only projection, interior vertices can move tangentially, allowing
an obsolete recess to unfold. Vertex order, topology and UVs are retained.
Callers must review the proposal before committing it to a character.
"""
import numpy as np


def restore(context, history):
    """Restore a committed repair only while the head still matches its result.

    Texture publication has its own backup; this restores vertex and shape-key
    coordinates, rather than silently replacing subsequent artist edits.
    """
    import bpy,json
    from pathlib import Path
    from . import mesh_ops as m
    from .vertex_contract import fingerprint
    obj=context.scene.cm_settings.result;history=Path(history).resolve()
    root=Path(bpy.path.abspath(context.scene.cm_settings.project_dir)).resolve()
    if not history.is_relative_to(root):raise ValueError('Use this character project\'s repair history')
    report=json.loads((history/'repair.json').read_text())
    if context.mode!='OBJECT' or context.scene.cm_features.expression_active:raise ValueError('Return to Neutral and Object Mode first')
    if fingerprint(m.evaluated_coordinates(context,obj))!=report['geometry_after']:raise ValueError('The head changed after this repair; retain the subsequent artist edits')
    keys=list(obj.data.shape_keys.key_blocks) if obj.data.shape_keys else []
    if [key.name for key in keys]!=report['shape_keys']:raise ValueError('The shape-key layout changed after this repair')
    with np.load(history/'rollback.npz',allow_pickle=False) as saved:
        obj.data.vertices.foreach_set('co',saved['basis'].astype(np.float32).ravel())
        for i,key in enumerate(keys):key.data.foreach_set('co',saved['key_'+str(i)].astype(np.float32).ravel())
    if obj.data.shape_keys:obj.data.shape_keys.update_tag()
    obj.data.update();obj.update_tag(refresh={'DATA'});context.view_layer.update()
    if fingerprint(m.evaluated_coordinates(context,obj))!=report['geometry_before']:raise RuntimeError('Restored coordinates differ from the original repair input')
    report['status']='geometry_restored';(history/'repair.json').write_text(json.dumps(report,indent=2));return report


def apply(context, proposed, *, expected_geometry, history):
    """Commit a reviewed proposal with an exact mesh/shape-key rollback file."""
    import bpy,json
    from pathlib import Path
    from . import mesh_ops as m
    from .vertex_contract import fingerprint
    s,f=context.scene.cm_settings,context.scene.cm_features;obj=s.result
    if context.mode!='OBJECT' or f.expression_active:raise ValueError('Return to Neutral and Object Mode before committing a source repair')
    before=m.evaluated_coordinates(context,obj);proposed=np.asarray(proposed,dtype=np.float64)
    if fingerprint(before)!=expected_geometry:raise ValueError('The reviewed source repair belongs to a different head')
    if proposed.shape!=before.shape or not np.isfinite(proposed).all():raise ValueError('Keep finite positions and the complete vertex order')
    history=Path(history).resolve();root=Path(bpy.path.abspath(s.project_dir)).resolve()
    if not history.is_relative_to(root):raise ValueError('Keep source repair history inside the character project')
    history.mkdir(parents=True,exist_ok=False)
    original=m.coordinates(obj,False).copy();keys=list(obj.data.shape_keys.key_blocks) if obj.data.shape_keys else []
    snapshots=[np.array([v.co[:] for v in key.data],dtype=np.float32) for key in keys]
    np.savez_compressed(history/'rollback.npz',basis=original,**{'key_'+str(i):v for i,v in enumerate(snapshots)})
    delta=m.to_local(obj,proposed)-m.to_local(obj,before)
    def set_positions(basis,key_values):
        obj.data.vertices.foreach_set('co',np.asarray(basis,dtype=np.float32).ravel())
        for key,values in zip(keys,key_values):key.data.foreach_set('co',np.asarray(values,dtype=np.float32).ravel())
        if obj.data.shape_keys:obj.data.shape_keys.update_tag()
        obj.data.update();obj.update_tag(refresh={'DATA'});context.view_layer.update()
    try:
        set_positions(original+delta,[values+delta for values in snapshots])
        actual=m.evaluated_coordinates(context,obj);np.testing.assert_allclose(actual,proposed,atol=2e-7)
        unchanged=np.all(proposed==before,axis=1);np.testing.assert_array_equal(actual[unchanged],before[unchanged])
        report=dict(status='applied',geometry_before=expected_geometry,geometry_after=fingerprint(actual),
                    shape_keys=[key.name for key in keys],outside_patch_exact=True,rollback='rollback.npz')
        (history/'repair.json').write_text(json.dumps(report,indent=2));return report
    except BaseException:
        set_positions(original,snapshots);raise


def reconstruct(points, edges, target_raycast, source_raycast, view, scale,
                region, *, lower_limit=None, max_move=.04, triangles=None):
    points=np.asarray(points,dtype=np.float64);view=np.asarray(view,dtype=np.float64)
    region=np.asarray(region,dtype=float)
    if points.ndim!=2 or points.shape[1]!=3 or not np.isfinite(points).all():raise ValueError('Use finite world-space mesh positions')
    if view.shape!=(4,4) or not np.isfinite(view).all() or not np.isfinite(scale) or scale<=0:raise ValueError('Use a finite orthographic view and positive scale')
    if region.shape!=(4,) or not np.isfinite(region).all() or np.any(region[2:]<=0):raise ValueError('Use a bounded ellipse with positive radii')
    local=(points-view[:3,3])@view[:3,:3]
    screen=np.column_stack((local[:,0]/scale+.5,.5-local[:,1]/scale))
    radius=np.linalg.norm((screen-np.asarray(region[:2]))/np.asarray(region[2:]),axis=1)
    inside=radius<1
    if lower_limit is not None:inside &= screen[:,1]<lower_limit
    # Solve on physical vertices so a UV split cannot create a repair seam.
    _,first,weld=np.unique(np.round(points,7),axis=0,return_index=True,return_inverse=True)
    physical=points[first].copy();uv=local[first,:2].copy()
    edges=np.unique(np.sort(weld[np.asarray(edges,dtype=np.int32)],axis=1),axis=0)
    edges=edges[edges[:,0]!=edges[:,1]];adj=[[] for _ in physical]
    for a,b in edges:adj[a].append(b);adj[b].append(a)
    candidate=inside[first];selected=np.zeros(len(physical),bool);todo=[];forward=-view[:3,2]
    for i in np.flatnonzero(candidate):
        j=first[i];origin=view[:3,3]+view[:3,0]*local[j,0]+view[:3,1]*local[j,1]
        hit=target_raycast(origin,forward)
        if hit is not None and abs(float((physical[i]-hit)@forward))<.001:todo.append(int(i))
    while todo:
        i=todo.pop()
        if selected[i]:continue
        selected[i]=True;todo.extend(j for j in adj[i] if candidate[j] and not selected[j])
    ids=np.flatnonzero(selected)
    if not len(ids) or len(ids)>10000:raise ValueError('Select a bounded connected skin patch')
    a,b=edges.T;src=np.r_[a,b];dst=np.r_[b,a];keep=selected[src];src=src[keep];dst=dst[keep]
    degree=np.bincount(src,minlength=len(physical))
    if np.any(degree[ids]==0):raise ValueError('The patch contains isolated vertices')
    # Positive harmonic weights unfold the interior in the viewing plane.
    # The unchanged surrounding vertices provide all boundary conditions.
    residual=0.
    for iteration in range(6000):
        mean=np.column_stack([np.bincount(src,weights=uv[dst,k],minlength=len(physical)) for k in range(2)])
        updated=mean[ids]/degree[ids,None]
        residual=float(abs(updated-uv[ids]).max());uv[ids]=updated
        if residual<1e-9:break
    if residual>1e-7:raise ValueError('Patch parameterization did not converge')
    proposed=physical.copy();missed=[]
    for i in ids:
        origin=view[:3,3]+view[:3,0]*uv[i,0]+view[:3,1]*uv[i,1]
        hit=source_raycast(origin,forward)
        if hit is None:missed.append(int(i))
        else:proposed[i]=hit
    if missed:raise ValueError('The source does not cover the complete selected patch')
    # A short boundary collar prevents an abrupt step where source and target
    # surfaces differ. The obsolete recess remains fully reconstructed inside.
    distance=np.full(len(physical),100000,dtype=np.int32);distance[~selected]=0
    for _ in range(5):
        updated=distance.copy();np.minimum.at(updated,a,distance[b]+1);np.minimum.at(updated,b,distance[a]+1);distance=updated
    blend=np.clip((distance-1)/4,0,1).astype(float);blend=blend*blend*(3-2*blend)
    displacement=(proposed-physical)*blend[:,None]
    maximum=float(np.linalg.norm(displacement,axis=1).max())
    if maximum>max_move:raise ValueError('Source reconstruction exceeds the allowed local movement')
    result=points+displacement[weld]
    np.testing.assert_array_equal(result[~selected[weld]],points[~selected[weld]])
    chart=None
    if triangles is not None:
        faces=np.asarray(triangles,dtype=np.int32);faces=faces[selected[weld[faces]].all(1)]
        projected=((result-view[:3,3])@view[:3,:3])[:,:2][faces]
        e=projected[:,1]-projected[:,0];f=projected[:,2]-projected[:,0]
        areas=e[:,0]*f[:,1]-e[:,1]*f[:,0]
        tolerance=scale*scale*1e-12
        if not len(areas) or np.any(areas<=tolerance):raise ValueError('The reconstructed patch folds or collapses in its source view; expand the boundary')
        chart=dict(triangles=len(faces),folded=0,collapsed=0,minimum_double_area_m2=float(areas.min()))
    return result,dict(method='SOURCE_SUPPORTED_HARMONIC_PATCH',changed_vertices=np.flatnonzero(selected[weld]).tolist(),
        physical_vertices=int(len(ids)),iterations=iteration+1,parameter_residual_m=residual,
        max_movement_m=maximum,outside_patch_exact=True,source_hits=int(len(ids)),
        topology_unchanged=True,proposal_requires_review=True,chart_validation=chart,
        region=region.tolist(),view=view.tolist(),view_scale=float(scale),boundary_collar_rings=4)
