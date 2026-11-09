"""Local source projection and connected fairing through explicit view masks.

The fairing operation retains fixed boundary positions and protects anatomical
landmarks. Rear surfaces are excluded by visibility and mesh connectivity.
"""
import numpy as np


def project_depth(points,source_raycast,target_raycast,view,scale,region,*,depth_reserve=.04,visible_depth=.012,max_move=.025):
    points=np.asarray(points,dtype=np.float64);view=np.asarray(view,dtype=np.float64)
    local=(points-view[:3,3])@view[:3,:3]
    screen=np.column_stack((local[:,0]/scale+.5,.5-local[:,1]/scale))
    center=np.asarray(region[:2]);radii=np.asarray(region[2:])
    if np.any(radii<=0) or not 0<depth_reserve<=1:raise ValueError('Use a bounded region and a positive retained depth')
    radius=np.linalg.norm((screen-center)/radii,axis=1);selected=radius<1
    weights=np.clip((1-radius)/.25,0,1);weights=weights*weights*(3-2*weights)
    forward=-view[:3,2];result=points.copy();changed=[]
    for index in np.flatnonzero(selected):
        origin=view[:3,3]+view[:3,0]*local[index,0]+view[:3,1]*local[index,1]
        target=target_raycast(origin,forward);source=source_raycast(origin,forward)
        if target is None or source is None:continue
        if (points[index]-target)@forward>visible_depth:continue
        movement=float((source-points[index])@forward)*(1-depth_reserve)*weights[index]
        if abs(movement)>max_move:raise ValueError(f'The selected source correction exceeds the local movement limit: {movement:.6f}m at vertex {index}')
        result[index]+=forward*movement
        if abs(movement)>1e-10:changed.append(int(index))
    if not changed:raise ValueError('No source-supported depth correction was found in the selected region')
    np.testing.assert_array_equal(result[~selected],points[~selected])
    return result,dict(method='BOUNDED_SOURCE_DEPTH_PATCH',changed_vertices=changed,max_movement_m=float(np.linalg.norm(result-points,axis=1).max()),outside_region_exact=True,screen_coordinates_preserved=True,retained_depth_fraction=depth_reserve)


def fair_visible_patch(points,edges,target_raycast,view,scale,region,*,depth_reserve=.005,protected=(),protected_below=None):
    """Fair the visible connected patch, including its attached recess lining.

    Connectivity within the selection keeps a mouth interior or rear scalp
    from being pulled through the front merely because it shares screen XY.
    UV-split copies share their original coincident-position witness.
    """
    points=np.asarray(points,dtype=np.float64);view=np.asarray(view,dtype=np.float64)
    local=(points-view[:3,3])@view[:3,:3];screen=np.column_stack((local[:,0]/scale+.5,.5-local[:,1]/scale))
    radius=np.linalg.norm((screen-np.asarray(region[:2]))/np.asarray(region[2:]),axis=1);inside=radius<1
    for hole in protected:
        inside &= np.linalg.norm((screen-np.asarray(hole[:2]))/np.asarray(hole[2:]),axis=1)>=1
    if protected_below is not None:inside &= screen[:,1]<protected_below
    forward=-view[:3,2];edges=np.asarray(edges,dtype=np.int32)
    _,inverse=np.unique(np.round(points,7),axis=0,return_inverse=True)
    first={};extra=[]
    for i,k in enumerate(inverse):
        if k in first:extra.append((first[k],i))
        else:first[k]=i
    if extra:edges=np.concatenate((edges,np.asarray(extra,dtype=np.int32)))
    edges=np.unique(np.sort(edges,axis=1),axis=0);adj=[[] for _ in points]
    for a,b in edges:adj[a].append(b);adj[b].append(a)
    seeds=[]
    for i in np.flatnonzero(inside):
        origin=view[:3,3]+view[:3,0]*local[i,0]+view[:3,1]*local[i,1]
        hit=target_raycast(origin,forward)
        if hit is not None and (points[i]-hit)@forward<.001:seeds.append(int(i))
    selected=np.zeros(len(points),bool);todo=list(seeds)
    while todo:
        i=todo.pop()
        if selected[i]:continue
        selected[i]=True;todo.extend(j for j in adj[i] if inside[j] and not selected[j])
    ids=np.flatnonzero(selected)
    if not len(ids) or len(ids)>3500:raise ValueError('Use a small connected surface patch for local fairing')
    index={int(v):i for i,v in enumerate(ids)}
    border=sorted({j for i in ids for j in adj[i] if not selected[j]})
    rows=np.r_[ids,border];matrix=np.zeros((len(rows),len(ids)));rhs=np.zeros((len(rows),3))
    boundary=0
    for row,i in enumerate(rows):
        if selected[i]:matrix[row,index[i]]=len(adj[i])
        else:rhs[row]-=len(adj[i])*points[i]
        for j in adj[i]:
            if selected[j]:matrix[row,index[j]]-=1
            else:rhs[row]+=points[j];boundary+=1
    if not boundary:raise ValueError('Keep an intact fixed boundary around a surface correction')
    # Boundary rows constrain the entering tangent as well as position,
    # avoiding a visible crease around the repaired region.
    solved=np.linalg.solve(matrix.T@matrix,matrix.T@rhs)
    result=points.copy();result[ids]=solved*(1-depth_reserve)+points[ids]*depth_reserve
    if np.linalg.norm(result-points,axis=1).max()>scale*max(region[2:])*.75:
        raise ValueError('The proposed fairing exceeds the size of the selected feature')
    np.testing.assert_array_equal(result[~selected],points[~selected])
    return result,dict(method='CONNECTED_VISIBLE_BIHARMONIC_FAIRING',changed_vertices=ids.tolist(),boundary_connections=boundary,max_movement_m=float(np.linalg.norm(result-points,axis=1).max()),outside_region_exact=True,screen_coordinates_preserved=False,retained_shape_fraction=depth_reserve)


def flatten_protrusion(points,target_raycast,view,scale,region,*,protected=(),protected_below=None,max_move=.015):
    """Depress a protrusion to a robust quadratic fit of surrounding skin.

    This is an explicit design edit, not source reconstruction. Only outward
    excess is removed. Protected features and unselected vertices stay exact.
    """
    points=np.asarray(points,dtype=np.float64);view=np.asarray(view,dtype=np.float64)
    local=(points-view[:3,3])@view[:3,:3];screen=np.column_stack((local[:,0]/scale+.5,.5-local[:,1]/scale))
    xy=(screen-np.asarray(region[:2]))/np.asarray(region[2:]);radius=np.linalg.norm(xy,axis=1);visible=np.zeros(len(points),bool);forward=-view[:3,2]
    for i in np.flatnonzero(radius<1.5):
        origin=view[:3,3]+view[:3,0]*local[i,0]+view[:3,1]*local[i,1];hit=target_raycast(origin,forward)
        visible[i]=hit is not None and (points[i]-hit)@forward<.001
    weight=np.clip((1-radius)/.25,0,1)
    for hole in protected:
        r=np.linalg.norm((screen-np.asarray(hole[:2]))/np.asarray(hole[2:]),axis=1);weight*=np.clip((r-1)/.5,0,1)
    if protected_below is not None:weight*=np.clip((protected_below-screen[:,1])/(8/768),0,1)
    weight=weight*weight*(3-2*weight)
    ring=visible&(radius>=1)&(radius<1.5)
    if protected_below is not None:ring &= screen[:,1]<protected_below
    if ring.sum()<20:raise ValueError('Choose a protrusion with sufficient visible surrounding skin')
    u,v=xy.T;design=np.column_stack((np.ones(len(points)),u,v,u*u,u*v,v*v))
    a=design[ring];b=local[ring,2];fit=np.linalg.lstsq(a,b,rcond=None)[0]
    for _ in range(5):
        residual=abs(a@fit-b);limit=max(float(np.median(residual))*2,1e-5);w=np.minimum(1,limit/np.maximum(residual,1e-12));fit=np.linalg.lstsq(a*w[:,None],b*w,rcond=None)[0]
    target=design@fit;delta=np.minimum(target-local[:,2],0)*weight
    delta[~visible]=0
    if abs(delta).max()>max_move:raise ValueError('Flattening exceeds the local design movement limit')
    changed=np.flatnonzero(abs(delta)>1e-10);result=points.copy();result[changed]+=view[:3,2]*delta[changed,None]
    if not len(changed):raise ValueError('No outward protrusion was found above the surrounding-skin fit')
    return result,dict(method='PROTECTED_SURROUNDING_SKIN_FLATTEN',changed_vertices=changed.tolist(),max_movement_m=float(abs(delta).max()),outside_region_exact=True,screen_coordinates_preserved=True,design_edit=True,surrounding_samples=int(ring.sum()))
