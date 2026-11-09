"""Source-region inference driven body refinement proposals (external worker).

Produces a proposal, never a production acceptance. Blender must guard and
review the complete physical mesh before committing. No character bounds.
"""
import numpy as np
from .semantic_correspondence import region_faces,normal_motion_support
from .surface_oracle import SurfaceOracle
from .global_registration import vertex_normals
from .surface_fairing import fair


def project_region_normals(base,current,triangles,source,source_triangles,influence,*,source_face_ids=None,max_distance,stiffness=.3,selection='nearest',cancelled=None):
    """Conform a registered part's thickness without changing its station.

    Signed normal-line intersections avoid sending forearm vertices to a hand
    through tangential nearest-point motion. Influence defines the allowed
    part and its attachment collar; zero influence is an exact hard boundary.
    Source witnesses are exact, while fairing relaxes them for coherent skin.
    Full-mesh contact acceptance is deliberately left to the caller.
    """
    current=np.asarray(current,dtype=float);influence=np.asarray(influence,dtype=float)
    if influence.shape!=(len(current),) or not np.isfinite(influence).all() or np.any((influence<0)|(influence>1)):raise ValueError('Normal fitting requires a bounded registered part mask')
    ids=np.flatnonzero(influence>.01)
    if not len(ids):raise ValueError('No supported vertices in the selected part')
    normals=vertex_normals(current,triangles);oracle=SurfaceOracle(source,source_triangles,source_face_ids)
    record=oracle.ray_query(current[ids],normals[ids],max_distance=max_distance,normal_limit=.2,selection=selection,cancelled=cancelled)
    observed=record['observed'];confidence=np.zeros(len(current));goal=current.copy()
    if observed.any():
        clearance=oracle.boundary_clearance(record['surface_points'][observed],cutoff=max_distance*.1,cancelled=cancelled)
        support=np.clip(clearance/(max_distance*.05),0,1)**2
        selected=ids[observed];confidence[selected]=record['confidence'][observed]*influence[selected]**2*support
        goal[selected]=current[selected]+(record['surface_points'][observed]-current[selected])*influence[selected,None]
    free=influence>.01
    result,report=fair(base,triangles,goal,confidence*20.,~free,stiffness,reference=current,retention=.02,preserve_reference_differentials=True)
    result[~free]=current[~free]
    report.update(method='REGISTERED_PART_SIGNED_NORMAL_CONFORMANCE_V1',queried_vertices=len(ids),observed_vertices=int(observed.sum()),
        source_witnesses_exact=True,source_geometry_changed=False,outside_unchanged=bool(np.array_equal(result[~free],current[~free])),production_ready=False)
    return result,report


def place_aligned_extent_region(base,current,source_region,influence):
    """Initialize an already oriented child region from its observed extents.

    A parent frame need not describe a child's proportions. This positive
    diagonal placement preserves the child's neutral orientation and blends
    through its native rig influence. Complete source region coverage and
    aligned axes are preconditions, not inferred guarantees. Review the join.
    """
    base=np.asarray(base,dtype=float);current=np.asarray(current,dtype=float);source=np.asarray(source_region,dtype=float);influence=np.asarray(influence,dtype=float)
    if base.shape!=current.shape or base.ndim!=2 or base.shape[1]!=3 or source.ndim!=2 or source.shape[1]!=3 or not len(source) or influence.shape!=(len(base),) or not all(np.isfinite(x).all() for x in (base,current,source,influence)) or np.any((influence<0)|(influence>1)):raise ValueError('Invalid aligned child region')
    ids=influence>.05
    if not ids.any():raise ValueError('Choose a supported child rig region')
    low,high=base[ids].min(0),base[ids].max(0);a,b=source.min(0),source.max(0)
    if np.any(high-low<=1e-12) or np.any(b-a<=1e-12):raise ValueError('Region extents must span all three axes')
    scale=(b-a)/(high-low);translation=a-low*scale
    posed=base*scale+translation
    return current+(posed-current)*influence[:,None],dict(method='ALIGNED_CHILD_EXTENT_PLACEMENT_V1',
        scale=scale.tolist(),translation=translation.tolist(),source_extent=[a.tolist(),b.tolist()],production_ready=False)


def fit_region_frame(clean,targets,weights,*,condition_limit=5.,axis_aligned=False):
    """Robust orientation-preserving affine placement for a clean rig region.

    The low-dimensional transform cannot fold the interior of the clean
    region. This says nothing about its join to adjacent regions. Targets
    must be observed correspondences, not nearest-vertex coordinate copies.
    """
    clean=np.asarray(clean,dtype=float);targets=np.asarray(targets,dtype=float);weights=np.asarray(weights,dtype=float)
    if clean.ndim!=2 or clean.shape[1]!=3 or targets.shape!=clean.shape or weights.shape!=(len(clean),) or not all(np.isfinite(x).all() for x in (clean,targets,weights)) or np.any(weights<0):raise ValueError('Invalid frame observations')
    if np.count_nonzero(weights)>3 and np.linalg.matrix_rank(clean[weights>0]-np.average(clean,axis=0,weights=weights))<3:raise ValueError('Frame observations do not span a volume')
    if np.count_nonzero(weights)<4 or not np.isfinite(condition_limit) or condition_limit<1:raise ValueError('Frame fitting needs four supported observations and a valid condition limit')
    original=weights.copy();center=np.average(clean,axis=0,weights=weights)
    scale=max(float(np.sqrt(np.average(np.sum((clean-center)**2,axis=1),weights=weights))),1e-12)
    design=np.column_stack(((clean-center)/scale,np.ones(len(clean))))
    for _ in range(6):
        if axis_aligned:
            transform=np.zeros((4,3))
            for axis in range(3):
                columns=design[:,[axis,3]]
                solved=np.linalg.lstsq(columns*np.sqrt(weights[:,None]),targets[:,axis]*np.sqrt(weights),rcond=None)[0]
                transform[axis,axis]=max(float(solved[0]),scale*.01);transform[3,axis]=solved[1]
        else:transform=np.linalg.lstsq(design*np.sqrt(weights[:,None]),targets*np.sqrt(weights[:,None]),rcond=None)[0]
        residual=np.linalg.norm(design@transform-targets,axis=1)
        order=np.argsort(residual);cumulative=np.cumsum(original[order])
        median=residual[order[np.searchsorted(cumulative,cumulative[-1]*.5)]]
        robust_scale=max(float(median),scale*1e-4)
        weights=original/(1+(residual/(2*robust_scale))**2)
    linear=transform[:3]/scale;u,s,v=np.linalg.svd(linear)
    if s.max()<1e-10:raise ValueError('Source observations collapse the requested region frame')
    correction=np.ones(3)
    if np.linalg.det(u@v)<0:correction[-1]=-1
    clipped=np.maximum(s,s.max()/condition_limit)
    linear=u@np.diag(clipped*correction)@v
    translation=np.average(targets-clean@linear,axis=0,weights=weights)
    result=clean@linear+translation
    return linear,translation,dict(method='ROBUST_POSITIVE_REGION_AFFINE_V1',
        determinant=float(np.linalg.det(linear)),condition_number=float(clipped.max()/clipped.min()),
        axis_aligned=bool(axis_aligned),
        supported=int((original>0).sum()),weighted_target_rms=float(np.sqrt(np.average(np.sum((result-targets)**2,axis=1),weights=original))))


def reconstruct_rig_regions(base,current,triangles,source,source_triangles,influences,transfer,*,
                            columns,max_distance,region_names=None,iterations=1000,axis_aligned_frame=False,plant_axis=None):
    """Replace damaged rig regions with posed clean differentials and a stitch.

    Source labels, skin weights and connectivity supply all region selection;
    no world-axis or character-specific boxes are used. No contact acceptance
    is performed here: the complete reconstructed mesh needs Blender review.
    """
    from . import region_reconstruction as rr
    base=np.asarray(base,dtype=float);current=np.asarray(current,dtype=float);triangles=np.asarray(triangles,dtype=np.int64)
    influences=np.asarray(influences,dtype=float)
    if base.shape!=current.shape or influences.ndim!=2 or influences.shape[0]!=len(base) or not np.isfinite(influences).all() or np.any((influences<0)|(influences>1)):raise ValueError('Rig reconstruction arrays must match canonical order')
    _,weld=np.unique(np.round(base,6),axis=0,return_inverse=True);count=np.bincount(weld)
    def physical(values):return np.column_stack([np.bincount(weld,weights=values[:,i])/count for i in range(3)])
    edges=np.unique(np.sort(weld[np.vstack((triangles[:,[0,1]],triangles[:,[1,2]],triangles[:,[2,0]]))],axis=1),axis=0);edges=edges[edges[:,0]!=edges[:,1]]
    reference=physical(current);output=reference.copy();normal=vertex_normals(current,triangles);reports=[]
    surface=base[triangles];area=np.linalg.norm(np.cross(surface[:,1]-surface[:,0],surface[:,2]-surface[:,0]),axis=1)
    vertex_area=np.zeros(len(base))
    for corner in range(3):np.add.at(vertex_area,triangles[:,corner],area/3)
    for column in columns:
        faces=region_faces(transfer,column);weights=influences[:,column]
        ids=np.flatnonzero(weights>.01)
        if not len(faces) or not len(ids):raise ValueError('Requested rig region has no source support')
        oracle=SurfaceOracle(source,source_triangles,faces)
        record=oracle.query(current[ids],max_distance=max_distance,query_normals=normal[ids],normal_limit=.2)
        area_weight=vertex_area[ids]/max(float(vertex_area[ids].mean()),1e-300)
        mass=record['confidence']*weights[ids]**2*area_weight;observed=record['observed']
        if observed.any():
            clearance=oracle.boundary_clearance(record['surface_points'][observed],cutoff=max_distance)
            mass[observed]*=transfer['confidence'][record['source_triangle_ids'][observed]]*np.clip(clearance/(max_distance*.05),0,1)**2
        linear,translation,frame=fit_region_frame(base[ids],record['surface_points'],mass,axis_aligned=axis_aligned_frame)
        if plant_axis is not None:
            if plant_axis not in (0,1,2):raise ValueError('Choose a planted world axis from zero to two')
            source_ids=np.unique(oracle.triangles[oracle.face_ids])
            floor=float(np.quantile(oracle.vertices[source_ids,plant_axis],.001))
            placed_min=float(np.min((base[ids]@linear)[:,plant_axis]))
            translation[plant_axis]=floor-placed_min
            frame.update(plant_axis=int(plant_axis),source_floor=floor)
        posed=base@linear+translation
        pw=np.zeros(len(count));np.maximum.at(pw,weld,weights)
        core,patch=rr.semantic_patch(pw,edges,core_threshold=.8)
        output,stitch_report=rr.stitch(output,physical(posed),edges,core,patch,iterations=iterations)
        reports.append(dict(region=str(column) if region_names is None else str(region_names[column]),frame=frame,stitch=stitch_report))
    result=current+(output-reference)[weld]
    return result,dict(method='CLEAN_RIG_REGION_RECONSTRUCTION_V1',regions=reports,coordinate_boxes_used=False,
        area_weighted_frame=True,
        production_ready=False,collision_review_required=True)


def reinitialize(base,triangles,registered,*,pinned,confidence=None,stiffness=2000.):
    """Reconstruct low-frequency placement from a clean canonical prior.

    This is a restart, not an incremental projection of a folded mesh. A
    positive-determinant affine placement and screened displacement solve
    retain clean reference differentials. Registration goals with extreme
    edge stretch lose confidence. Caller locks remain exact. Full collision
    and anatomical review are required; no injectivity guarantee is implied.
    """
    base=np.asarray(base,dtype=float);registered=np.asarray(registered,dtype=float)
    triangles=np.asarray(triangles,dtype=np.int64);pinned=np.asarray(pinned,dtype=bool)
    if base.shape!=registered.shape or base.ndim!=2 or base.shape[1]!=3 or not np.isfinite(base).all() or not np.isfinite(registered).all():raise ValueError('Clean and registered vertex orders must match')
    if pinned.shape!=(len(base),):raise ValueError('Locks must match canonical vertices')
    mass=np.ones(len(base)) if confidence is None else np.array(confidence,dtype=float,copy=True)
    if mass.shape!=pinned.shape or not np.isfinite(mass).all() or np.any(mass<0):raise ValueError('Invalid restart observation confidence')
    if triangles.ndim!=2 or triangles.shape[1]!=3 or not len(triangles) or triangles.min()<0 or triangles.max()>=len(base):raise ValueError('Invalid canonical triangles')
    # Equal-area samples avoid high facial tessellation dominating whole-body
    # scale. Robust affine fitting rejects local spikes in the old result.
    p=base[triangles];area=np.linalg.norm(np.cross(p[:,1]-p[:,0],p[:,2]-p[:,0]),axis=1)
    vertex_area=np.zeros(len(base))
    for column in range(3):np.add.at(vertex_area,triangles[:,column],area/3)
    center=np.average(base,axis=0,weights=vertex_area);design=np.column_stack((base-center,np.ones(len(base))))
    weight=vertex_area.copy()
    for _ in range(5):
        matrix=np.linalg.lstsq(design*np.sqrt(weight[:,None]),registered*np.sqrt(weight[:,None]),rcond=None)[0]
        residual=np.linalg.norm(design@matrix-registered,axis=1)
        scale=max(float(np.median(residual)),np.linalg.norm(np.ptp(registered,axis=0))*1e-5)
        weight=vertex_area/(1+(residual/(scale*2))**2)
    u,s,v=np.linalg.svd(matrix[:3]);correction=np.ones(3)
    if np.linalg.det(u@v)<0:correction[-1]=-1
    matrix[:3]=u@np.diag(np.clip(s,.05,20)*correction)@v
    prior=design@matrix
    edges=np.unique(np.sort(np.vstack((triangles[:,[0,1]],triangles[:,[1,2]],triangles[:,[2,0]])),axis=1),axis=0)
    clean_length=np.linalg.norm(prior[edges[:,0]]-prior[edges[:,1]],axis=1)
    old_length=np.linalg.norm(registered[edges[:,0]]-registered[edges[:,1]],axis=1)
    valid=clean_length>np.linalg.norm(np.ptp(prior,axis=0))*1e-10
    ratio=np.ones(len(edges));ratio[valid]=old_length[valid]/clean_length[valid]
    damage=np.maximum(ratio,1/np.maximum(ratio,1e-12));trust=np.ones(len(base))
    edge_trust=np.minimum(1.,(2/np.maximum(damage,1))**4)
    np.minimum.at(trust,edges[:,0],edge_trust);np.minimum.at(trust,edges[:,1],edge_trust)
    mass*=trust
    result,report=fair(base,triangles,registered,mass,pinned,stiffness,reference=prior,
        retention=.001,preserve_reference_differentials=True)
    report.update(method='CLEAN_CANONICAL_DIFFERENTIAL_REINITIALIZATION_V1',
        affine_determinant=float(np.linalg.det(matrix[:3])),
        downweighted_vertices=int((trust<.5).sum()),locks_exact=bool(np.array_equal(result[pinned],registered[pinned])),
        production_ready=False,collision_review_required=True)
    return result,report


def propose(base,current,triangles,source,source_triangles,influences,transfer,*,
            pinned,max_distance,region_names=None,graph_rings=5,stiffness=4.,
            cancelled=None):
    current=np.asarray(current,dtype=float);triangles=np.asarray(triangles,dtype=np.int64)
    influences=np.asarray(influences,dtype=float);pinned=np.asarray(pinned,dtype=bool)
    if influences.ndim!=2 or influences.shape[0]!=len(current) or not np.isfinite(influences).all() or np.any((influences<0)|(influences>1)):
        raise ValueError('Influences must match target vertices')
    if pinned.shape!=(len(current),):raise ValueError('Pins must match target vertices')
    if transfer['influences'].shape!=(len(source_triangles),influences.shape[1]):raise ValueError('Source transfer columns must match target region order')
    if not np.isfinite(max_distance) or max_distance<=0 or int(graph_rings)!=graph_rings or graph_rings<0:raise ValueError('Choose finite search distance and whole graph rings')
    names=list(range(influences.shape[1])) if region_names is None else list(region_names)
    if len(names)!=influences.shape[1]:raise ValueError('Region names must match influence columns')
    normal=vertex_normals(current,triangles);blend=np.zeros_like(current);mass=np.zeros(len(current))
    reverse_oracle=SurfaceOracle(current,triangles)
    free=np.zeros(len(current),bool);observations={};evaluation=[]
    for column,name in enumerate(names):
        if cancelled and cancelled():raise RuntimeError('Anatomical refinement cancelled')
        faces=region_faces(transfer,column);weight=influences[:,column]**2
        ids=np.flatnonzero((weight>.01)&~pinned)
        if not len(faces) or not len(ids):
            observations[str(name)]={'supported':0,'vertices':len(ids),'source_faces':len(faces)};continue
        oracle=SurfaceOracle(source,source_triangles,faces)
        record=oracle.query(current[ids],max_distance=max_distance,query_normals=normal[ids],normal_limit=.1,cancelled=cancelled)
        observed=record['observed'];source_confidence=np.zeros(len(ids))
        boundary_confidence=np.zeros(len(ids))
        motion_confidence=normal_motion_support(current[ids],record['surface_points'],normal[ids],slide_tolerance=max_distance*.05)
        reciprocal_confidence=np.zeros(len(ids))
        if observed.any():
            clearance=oracle.boundary_clearance(record['surface_points'][observed],cutoff=max_distance,cancelled=cancelled)
            # A distant closest point on a cut edge is not evidence of an
            # anatomical station. Require interior support proportional to
            # displacement, with a small scale-relative transition collar.
            collar=np.maximum(max_distance*.05,record['residual'][observed]*.5)
            boundary_confidence[observed]=np.clip(clearance/collar,0,1)**2
            # The source point should map back near the same registered skin
            # location. A cap or another body part often wins a one-way query
            # but fails this reciprocal check near missing observations.
            source_faces=oracle.vertices[oracle.triangles[record['source_triangle_ids'][observed]]]
            source_normals=np.cross(source_faces[:,1]-source_faces[:,0],source_faces[:,2]-source_faces[:,0])
            reverse=reverse_oracle.query(record['surface_points'][observed],max_distance=max_distance,
                query_normals=source_normals,normal_limit=.1,cancelled=cancelled)
            roundtrip=np.linalg.norm(reverse['surface_points']-current[ids[observed]],axis=1)
            radius=np.maximum(max_distance*.05,record['residual'][observed]*.25)
            reciprocal_confidence[observed]=reverse['observed']*np.clip(1-roundtrip/radius,0,1)**2
        eligible=observed&(boundary_confidence>.05)&(motion_confidence>.25)&(reciprocal_confidence>.05)
        source_confidence[eligible]=transfer['confidence'][record['source_triangle_ids'][eligible]]
        w=weight[ids]*record['confidence']*source_confidence*boundary_confidence*motion_confidence*reciprocal_confidence
        blend[ids]+=record['surface_points']*w[:,None];mass[ids]+=w
        free[ids[w>1e-4]]=True
        observations[str(name)]=dict(supported=int(eligible.sum()),vertices=len(ids),source_faces=len(faces),
            boundary_rejected=int((observed&(boundary_confidence<=.05)).sum()),
            sliding_rejected=int((observed&(motion_confidence<=.25)).sum()),
            reciprocal_rejected=int((observed&(reciprocal_confidence<=.05)).sum()),
            before_rms=float(np.sqrt(np.mean(record['residual'][eligible]**2))) if eligible.any() else None)
        evaluation.append((str(name),oracle,ids,eligible))
    edges=np.unique(np.sort(np.vstack((triangles[:,[0,1]],triangles[:,[1,2]],triangles[:,[2,0]])),axis=1),axis=0)
    # Pins are barriers and remain exact after physical seam welding in fair().
    edges=edges[~np.any(pinned[edges],axis=1)]
    free&=~pinned
    for _ in range(int(graph_rings)):free[np.unique(edges[np.any(free[edges],axis=1)])]=True
    goal=current.copy();seen=mass>1e-8;goal[seen]=blend[seen]/mass[seen,None]
    goal[~free]=current[~free]
    result,report=fair(base,triangles,goal,np.clip(mass,0,1)*10.,~free,stiffness,reference=current,retention=.05,
        preserve_reference_differentials=True)
    # Existing registrations can have slightly separated UV seam copies.
    # Physical averaging is useful for the solve but must not silently move
    # locked canonical vertices when the displacement is scattered back.
    result[~free]=current[~free]
    report['canonical_pins_exact']=bool(np.array_equal(result[~free],current[~free]))
    for name,oracle,ids,eligible in evaluation:
        after=oracle.query(result[ids],max_distance=max_distance,query_normals=normal[ids],normal_limit=.1,cancelled=cancelled)
        residual=after['residual'][eligible]
        observations[name]['after_rms_same_supported_vertices']=float(np.sqrt(np.mean(np.nan_to_num(residual,nan=max_distance)**2))) if eligible.any() else None
    report.update(method='INFERRED_ANATOMY_SCREENED_SURFACE_REFINEMENT_V1',regions=observations,
        coordinate_boxes_used=False,collision_review_required=True,production_ready=False)
    return result,report
