"""Paired-region, bidirectional surface registration without editing the source.

CM_Fit_<name> vertex groups pair artist/agent-labelled patches. This is a
local refinement, not a substitute for articulated pose/landmark alignment.
Only NumPy and Blender's BVH are required; no model downloads or dense NxN arrays.
"""
import json
import re
import threading
import hashlib
import time
from bpy.app.handlers import persistent
import bpy
import numpy as np
from mathutils import Vector
from mathutils.bvhtree import BVHTree
from . import geometry, mesh_ops as m

PREFIX = 'CM_Fit_'
_JOB=None
_PENDING='cm_region_projection_pending'


def crossings(points, triangles, weld=None, *, pairs=False,tolerance=None,active_faces=None):
    """Count proper nonadjacent triangle crossings, excluding physical seams.

    BVH supplies broad-phase candidates only. Float64 segment tests avoid the
    scale-dependent results observed with mathutils ray/triangle tests. Plane
    penetration and edge clearance must exceed a physical tolerance (default
    one millionth of the mesh span). Coplanar overlaps and boundary-only
    contacts are not proper crossings.
    """
    points=geometry.points(points);triangles=np.asarray(triangles,dtype=np.int32).reshape(-1,3)
    tolerance=float(np.linalg.norm(np.ptp(points,axis=0))*1e-6 if tolerance is None else tolerance)
    if not np.isfinite(tolerance) or tolerance<0:raise ValueError('Crossing tolerance must be finite and nonnegative')
    tree=BVHTree.FromPolygons(points.tolist(),triangles.tolist(),all_triangles=True)
    physical=triangles if weld is None else np.asarray(weld)[triangles]
    found=set()
    if active_faces is None:
        candidates=np.asarray(tree.overlap(tree),dtype=np.int32).reshape(-1,2)
        candidates=candidates[candidates[:,0]<candidates[:,1]]
    else:
        active=np.asarray(active_faces,dtype=bool)
        if active.shape!=(len(triangles),):raise ValueError('Active collision faces must match triangles')
        ids=np.flatnonzero(active)
        if not len(ids):return set() if pairs else 0
        query=BVHTree.FromPolygons(points.tolist(),triangles[ids].tolist(),all_triangles=True)
        candidates=np.asarray(query.overlap(tree),dtype=np.int32).reshape(-1,2)
        if len(candidates):
            candidates[:,0]=ids[candidates[:,0]];candidates=candidates[candidates[:,0]!=candidates[:,1]];candidates=np.unique(np.sort(candidates,axis=1),axis=0)
    if len(candidates):
        # Most BVH self-overlaps are adjacent triangles. Filter shared physical
        # vertices in NumPy instead of doing thousands of tiny set operations
        # on every backtracking step.
        shared=np.any(physical[candidates[:,0],:,None]==physical[candidates[:,1],None,:],axis=(1,2))
        candidates=candidates[~shared]
    if len(candidates):
        first,second=points[triangles[candidates[:,0]]],points[triangles[candidates[:,1]]]
        hit=np.zeros(len(candidates),bool)
        def dot(a,b):return np.einsum('ij,ij->i',a,b)
        for edges,face in ((first,second),(second,first)):
            e1,e2=face[:,1]-face[:,0],face[:,2]-face[:,0]
            normal=np.cross(e1,e2);area=np.linalg.norm(normal,axis=1)
            lengths=np.column_stack((np.linalg.norm(e2,axis=1),np.linalg.norm(e1,axis=1),np.linalg.norm(e2-e1,axis=1)))
            for i in range(3):
                origin=edges[:,i];direction=edges[:,(i+1)%3]-origin
                p=np.cross(direction,e2);det=dot(e1,p)
                scale=np.linalg.norm(e1,axis=1)*np.linalg.norm(e2,axis=1)*np.linalg.norm(direction,axis=1)
                valid=abs(det)>np.maximum(scale*1e-12,1e-300)
                inverse=np.zeros_like(det);np.divide(1.,det,out=inverse,where=valid)
                delta=origin-face[:,0];u=dot(delta,p)*inverse;q=np.cross(delta,e1)
                v=dot(direction,q)*inverse;t=dot(e2,q)*inverse
                eps=1e-9
                plane_start=dot(delta,normal);plane_end=dot(delta+direction,normal)
                depth=tolerance*area
                straddles=((plane_start>depth)&(plane_end<-depth))|((plane_start<-depth)&(plane_end>depth))
                interior=np.all(np.column_stack((u,v,1-u-v))*area[:,None]>tolerance*lengths,axis=1)
                hit|=valid&straddles&interior&(u>eps)&(v>eps)&(u+v<1-eps)&(t>eps)&(t<1-eps)
        found={tuple(map(int,pair)) for pair in candidates[hit]}
    return found if pairs else len(found)


def weights(obj, name):
    group = obj.vertex_groups.get(name)
    out = np.zeros(len(obj.data.vertices))
    if group:
        for vertex in obj.data.vertices:
            out[vertex.index] = min(1., sum(w.weight for w in vertex.groups if w.group == group.index))
    return out


def separate_contacts(reference,triangles,allowed,*,movable=None,max_distance=None,iterations=12,cancelled=None):
    """Resolve small new penetrations while retaining inherited contacts.

    Candidate vertex/plane separations must reduce the new crossing count,
    preserve triangle orientation/quality and stay within a displacement budget.
    This is a bounded cleanup, not a substitute for anatomical registration.
    """
    reference=geometry.points(reference);current=reference.copy();triangles=np.asarray(triangles,dtype=np.int32)
    span=float(np.linalg.norm(np.ptp(reference,axis=0)));clearance=max(span*2e-6,1e-12)
    budget=float(span*.002 if max_distance is None else max_distance)
    if not np.isfinite(budget) or budget<=0:raise ValueError('Contact repair needs a positive displacement budget')
    movable=np.ones(len(reference),bool) if movable is None else np.asarray(movable,dtype=bool)
    if movable.shape!=(len(reference),):raise ValueError('Contact mobility mask must match physical vertices')
    collision_tolerance=span*1e-6
    allowed=set(allowed);remaining=crossings(current,triangles,pairs=True,tolerance=collision_tolerance)-allowed;before=len(remaining)
    quality=triangle_quality(reference,triangles)
    p=reference[triangles];normals=np.cross(p[:,1]-p[:,0],p[:,2]-p[:,0])
    for _ in range(max(1,int(iterations))):
        if cancelled and cancelled():raise RuntimeError('Contact repair cancelled; previous mesh retained')
        if not remaining:break
        best=None
        for pair in sorted(remaining):
            for face_id,plane_id in (pair,pair[::-1]):
                ids=triangles[face_id];plane=current[triangles[plane_id]]
                normal=np.cross(plane[1]-plane[0],plane[2]-plane[0]);length=np.linalg.norm(normal)
                if length<1e-20:continue
                normal/=length;distance=(current[ids]-plane[0])@normal
                for sign in (-1,1):
                    active=sign*distance<clearance;changed=ids[active]
                    if not len(changed) or not movable[changed].all():continue
                    candidate=current.copy();candidate[changed]+=normal*(sign*clearance-distance[active])[:,None]
                    displacement=np.linalg.norm(candidate-reference,axis=1)
                    if displacement.max()>budget:continue
                    local=np.any(np.isin(triangles,changed),axis=1)
                    q=candidate[triangles[local]];n=np.cross(q[:,1]-q[:,0],q[:,2]-q[:,0])
                    dot=np.einsum('ij,ij->i',n,normals[local])
                    if np.any(dot<.1*np.linalg.norm(n,axis=1)*np.linalg.norm(normals[local],axis=1)):continue
                    if np.any(triangle_quality(candidate,triangles[local])<quality[local]*.35):continue
                    new=crossings(candidate,triangles,pairs=True,tolerance=collision_tolerance)-allowed
                    score=float(np.sum(displacement**2))
                    if len(new)<len(remaining) and (best is None or (len(new),score)<best[0]):
                        best=((len(new),score),candidate,new)
            if best is not None and not best[2]:break
        if best is None:break
        _,current,remaining=best
    return current,{'solver':'BOUNDED_CONTACT_SEPARATION_V1','new_crossings_before':before,
        'new_crossings_after':len(remaining),'accepted':not remaining,
        'max_movement':float(np.linalg.norm(current-reference,axis=1).max()),'review_required':True}


def paired(source, target):
    return sorted(g.name for g in source.vertex_groups
                  if g.name.startswith(PREFIX) and target.vertex_groups.get(g.name))


def normals(points, triangles):
    p = points[triangles]
    face = np.cross(p[:, 1]-p[:, 0], p[:, 2]-p[:, 0])
    result = np.zeros_like(points)
    for col in range(3):np.add.at(result, triangles[:, col], face)
    return result / np.maximum(np.linalg.norm(result, axis=1)[:, None], 1e-20)


def barycentric(point, triangle):
    a, b, c = triangle
    u, v, w = b-a, c-a, point-a
    uu, uv, vv = u@u, u@v, v@v
    det = uu*vv-uv*uv
    if abs(det) < 1e-24:return np.array([1., 0., 0.])
    y, z = (vv*(w@u)-uv*(w@v))/det, (uu*(w@v)-uv*(w@u))/det
    value = np.maximum([1-y-z, y, z], 0)
    return value / max(sum(value), 1e-20)


def compatible_nearest(tree,point,normal,distance,span,source_normals,normal_cosine,*,cancelled=None):
    """Nearest supported facing surface within the existing bounded search.

    Grow the radius only when needed. Source face IDs index precomputed NumPy
    normals, avoiding one Python/NumPy dot call per candidate triangle.
    """
    origin=Vector(point);hit=tree.find_nearest(origin,distance)
    if hit[0] is None:return None
    if np.dot(hit[1],normal)>=normal_cosine:return hit
    limit=min(distance,max(hit[3]*3.,span*.10))
    radius=min(limit,max(hit[3]*1.25,span*.003,1e-9))
    while True:
        if cancelled and cancelled():raise RuntimeError('Regional projection cancelled; previous geometry retained')
        candidates=tree.find_nearest_range(origin,radius)
        if candidates:
            face_ids=np.fromiter((c[2] for c in candidates),dtype=np.int32,count=len(candidates))
            supported=np.einsum('ij,j->i',source_normals[face_ids],normal)>=normal_cosine
            if supported.any():
                distances=np.fromiter((c[3] for c in candidates),dtype=float,count=len(candidates))
                return candidates[int(np.argmin(np.where(supported,distances,np.inf)))]
        if radius>=limit:return None
        radius=min(limit,radius*2.)


def triangle_quality(points,triangles):
    p=points[triangles]
    area=np.linalg.norm(np.cross(p[:,1]-p[:,0],p[:,2]-p[:,0]),axis=1)
    lengths=sum(np.sum((p[:,i]-p[:,(i+1)%3])**2,axis=1) for i in range(3))
    return 2*np.sqrt(3)*area/np.maximum(lengths,1e-30)


def limit_step(current,candidate,triangles,reference_quality,guard,allowed_crossings,*,weld=None,orientation=None,minimum_quality_fraction=.35,collision_tolerance=None,diagnostics=None):
    """Limit physical collisions and distortion for ICP and shape initialization."""
    if not np.isfinite(minimum_quality_fraction) or not 0<minimum_quality_fraction<=1:raise ValueError('Choose a quality fraction between zero and one')
    if collision_tolerance is None:
        if not hasattr(guard,'collision_tolerance'):guard.collision_tolerance=float(np.linalg.norm(np.ptp(current,axis=0))*1e-6)
        collision_tolerance=guard.collision_tolerance
    if not np.isfinite(collision_tolerance) or collision_tolerance<0:raise ValueError('Choose a finite nonnegative collision tolerance')
    candidate=guard.limit(current,candidate);direction=candidate-current
    previous=current[triangles];previous_normal=np.cross(previous[:,1]-previous[:,0],previous[:,2]-previous[:,0]);previous_area=np.linalg.norm(previous_normal,axis=1)
    threshold=np.maximum(np.maximum(guard.area*.01,previous_area*.02),guard.minimum_double_area)*previous_area
    def invalid_area(values):
        p=values[triangles];normal=np.cross(p[:,1]-p[:,0],p[:,2]-p[:,0])
        return guard.active&(np.einsum('ij,ij->i',normal,previous_normal)<threshold)
    if diagnostics is not None:diagnostics.update(collision_tolerance=collision_tolerance,initial_area_guard_max_movement=float(np.linalg.norm(direction,axis=1).max(initial=0)))
    step_weight=np.ones(len(current));collision_limited=distortion_limited=False
    for backtrack in range(32):
        new_pairs=crossings(candidate,triangles,weld,pairs=True,tolerance=collision_tolerance)-allowed_crossings
        distorted=np.flatnonzero(triangle_quality(candidate,triangles)<reference_quality*minimum_quality_fraction)
        # Freezing collision vertices changes neighboring triangles too. The
        # initial area/orientation check must remain true after every repair.
        bad_area=invalid_area(candidate);distorted=np.union1d(distorted,np.flatnonzero(bad_area))
        if diagnostics is not None:diagnostics.update(correction_passes=backtrack,last_new_crossings=len(new_pairs),last_area_failures=int(bad_area.sum()),last_distorted_faces=len(distorted))
        if orientation is not None:
            p=candidate[triangles];n=np.cross(p[:,1]-p[:,0],p[:,2]-p[:,0])
            n/=np.maximum(np.linalg.norm(n,axis=1)[:,None],1e-20)
            oriented=np.linalg.norm(orientation,axis=1)>.5
            distorted=np.union1d(distorted,np.flatnonzero(oriented&(np.einsum('ij,ij->i',n,orientation)<.5)))
        if not new_pairs and not len(distorted):break
        collision_limited|=bool(new_pairs);distortion_limited|=bool(len(distorted))
        face_ids=np.union1d(np.asarray(list(new_pairs),dtype=int).ravel(),distorted)
        vertex_ids=np.unique(triangles[face_ids])
        if backtrack<8:step_weight[vertex_ids]*=.5
        else:step_weight[vertex_ids]=0
        if weld is not None:
            shared=np.ones(int(weld.max())+1);np.minimum.at(shared,weld,step_weight);step_weight=shared[weld]
        candidate=guard.limit(current,current+direction*step_weight[:,None])
    else:
        # The last local freeze may already have repaired its boundary. Do
        # not discard all other safe motion without checking that last field.
        final_pairs=crossings(candidate,triangles,weld,pairs=True,tolerance=collision_tolerance)-allowed_crossings
        final_bad=triangle_quality(candidate,triangles)<reference_quality*minimum_quality_fraction
        final_bad|=invalid_area(candidate)
        if orientation is not None:
            p=candidate[triangles];n=np.cross(p[:,1]-p[:,0],p[:,2]-p[:,0]);n/=np.maximum(np.linalg.norm(n,axis=1)[:,None],1e-20)
            final_bad|=(np.linalg.norm(orientation,axis=1)>.5)&(np.einsum('ij,ij->i',n,orientation)<.5)
        if final_pairs or np.any(final_bad):candidate=current.copy()
    if diagnostics is not None:diagnostics['final_max_movement']=float(np.linalg.norm(candidate-current,axis=1).max(initial=0))
    return candidate,collision_limited,distortion_limited


def solve(reference, triangles, source, source_triangles, influence, *, iterations=16,
          distance=None, regularization=1.6, reverse_strength=.4, normal_cosine=.8,
          weld=None, cancelled=None, progress=None):
    """Symmetric ICP with local rigid edge regularization and exact boundary pins.

    Reverse samples use equal spatial cells rather than raw source vertex density.
    Front/back correspondence rejects opposing normals. SurfaceStepGuard retains
    triangle orientation/area; it does not certify absence of self intersections.
    """
    rest = geometry.points(reference).copy(); current = rest.copy()
    if not 0<normal_cosine<1:raise ValueError('Surface normal compatibility must be between zero and one')
    source = geometry.points(source); triangles = np.asarray(triangles, dtype=np.int32)
    source_triangles = np.asarray(source_triangles, dtype=np.int32)
    influence = np.clip(np.asarray(influence, dtype=float), 0, 1)
    if influence.shape != (len(rest),) or not np.isfinite(influence).all():
        raise ValueError('Region weights must be finite and match the result')
    active = influence > 0
    if not active.any() or not len(source_triangles):raise ValueError('Mark nonempty source and result regions')
    span = float(np.linalg.norm(np.ptp(source[np.unique(source_triangles)], axis=0)))
    distance = float(distance if distance is not None else span*.3)
    if distance <= 0 or not np.isfinite(distance):raise ValueError('Region search distance must be positive')
    if weld is not None:
        _, weld = np.unique(weld, return_inverse=True)
        counts = np.bincount(weld);shared = np.ones(len(counts))
        np.minimum.at(shared, weld, influence);influence = shared[weld];active = influence > 0
    tree = BVHTree.FromPolygons(source.tolist(), source_triangles.tolist(), all_triangles=True)
    patch = triangles[np.all(active[triangles], axis=1)]
    if not len(patch):raise ValueError('Result region must include complete faces')
    edge = np.unique(np.sort(np.vstack((triangles[:, [0,1]],triangles[:, [1,2]],triangles[:, [2,0]])), axis=1), axis=0)
    edge = edge[np.any(active[edge], axis=1)];a,b = edge.T
    degree = np.bincount(edge.ravel(), minlength=len(rest)).astype(float)
    rest_edges = rest[a]-rest[b]
    # Area-balanced surface samples, capped to avoid high-poly source overhead.
    surf = source[source_triangles];centers = surf.mean(1)
    sn = np.cross(surf[:,1]-surf[:,0],surf[:,2]-surf[:,0]);sn /= np.maximum(np.linalg.norm(sn,axis=1)[:,None],1e-20)
    cell = max(span/42, 1e-9)
    _, sample = np.unique(np.floor(centers/cell).astype(np.int64), axis=0, return_index=True)
    sample = sample[np.linspace(0,len(sample)-1,min(len(sample),4096),dtype=int)]
    samples, sample_normals = centers[sample], sn[sample]
    guard = geometry.SurfaceStepGuard(rest, triangles, weld)
    guard.collision_tolerance=float(np.linalg.norm(np.ptp(rest,axis=0))*1e-6)
    reference_quality=triangle_quality(rest,triangles)
    anchor_normals=normals(rest,triangles)
    rp=rest[triangles];orientation=np.cross(rp[:,1]-rp[:,0],rp[:,2]-rp[:,0])
    orientation/=np.maximum(np.linalg.norm(orientation,axis=1)[:,None],1e-20)
    def measure(points):
        forward=[]
        for p in points[active]:forward.append(tree.find_nearest(Vector(p))[3])
        reverse=[]
        mesh=BVHTree.FromPolygons(points.tolist(),patch.tolist(),all_triangles=True)
        for p in samples:reverse.append(mesh.find_nearest(Vector(p))[3])
        return {'forward_rms':float(np.sqrt(np.mean(np.square(forward)))),
                'reverse_rms':float(np.sqrt(np.mean(np.square(reverse)))),
                'forward_p95':float(np.quantile(forward,.95)),
                'reverse_p95':float(np.quantile(reverse,.95)),
                'source_coverage':float(np.mean(np.asarray(reverse)<=span*.01))}
    initial_metrics=measure(current)
    allowed_crossings=crossings(current,triangles,weld,pairs=True,tolerance=guard.collision_tolerance)
    initial_metrics['crossings']=len(allowed_crossings)
    collision_limited=0;distortion_limited=0
    history=[]
    for iteration in range(max(1,int(iterations))):
        if cancelled and cancelled():raise RuntimeError('Regional projection cancelled; previous geometry retained')
        if progress:progress(iteration/max(int(iterations),1))
        vn = normals(current, triangles)*.35+anchor_normals*.65
        vn/=np.maximum(np.linalg.norm(vn,axis=1)[:,None],1e-20)
        if weld is not None:
            vn=np.column_stack([np.bincount(weld,weights=vn[:,i],minlength=len(counts)) for i in range(3)])[weld]
            vn/=np.maximum(np.linalg.norm(vn,axis=1)[:,None],1e-20)
        data = np.zeros_like(rest);mass = np.zeros(len(rest)); forward_dist=[]
        for index in np.flatnonzero(active):
            if cancelled and cancelled():raise RuntimeError('Regional projection cancelled; previous geometry retained')
            hit=compatible_nearest(tree,current[index],vn[index],distance,span,sn,normal_cosine,cancelled=cancelled)
            if hit is None:continue
            point,normal,_,dist=hit
            confidence=float(np.dot(normal,vn[index]))**2
            data[index]=np.asarray(point)*confidence;mass[index]=confidence;forward_dist.append(dist)
        # Balanced reverse term makes unrepresented source volume visible to
        # the solve; forward-only ICP can score well after shrinking fingers.
        target_tree=BVHTree.FromPolygons(current.tolist(),patch.tolist(),all_triangles=True)
        reverse_dist=[];reverse_mass=np.zeros(len(rest));reverse_delta=np.zeros_like(rest)
        for point,normal in zip(samples,sample_normals):
            if cancelled and cancelled():raise RuntimeError('Regional projection cancelled; previous geometry retained')
            hit,face_normal,face,dist=target_tree.find_nearest(Vector(point),distance)
            if hit is None or np.dot(normal,face_normal)<normal_cosine:continue
            ids=patch[face];bary=barycentric(np.asarray(hit),current[ids])
            anchored=bary@anchor_normals[ids];anchored/=max(np.linalg.norm(anchored),1e-20)
            if normal@anchored<normal_cosine:continue
            reverse_dist.append(dist)
            np.add.at(reverse_mass,ids,bary)
            np.add.at(reverse_delta,ids,bary[:,None]*(point-np.asarray(hit)))
        received=reverse_mass>1e-12
        # Normalize coverage per target vertex so dense local tessellation
        # cannot dominate the fit; reverse displacements remain in world units.
        data[received]+=reverse_strength*(current[received]+reverse_delta[received]/reverse_mass[received,None])
        mass[received]+=reverse_strength
        supported=mass>0
        data[supported]/=mass[supported,None];data[~supported]=current[~supported]
        # Local ARAP rotations preserve connected surface structure while the
        # data term changes proportions. Batched 3x3 SVD has linear memory.
        cur_edges=current[a]-current[b]
        covariance=np.zeros((len(rest),3,3))
        outer=cur_edges[:,:,None]*rest_edges[:,None,:]
        np.add.at(covariance,a,outer);np.add.at(covariance,b,outer)
        u,_,vt=np.linalg.svd(covariance)
        rotation=u@vt;bad=np.linalg.det(rotation)<0
        u[bad,:,-1]*=-1;rotation=u@vt
        desired=np.einsum('nij,nj->ni',(rotation[a]+rotation[b])*.5,rest_edges)
        rhs_edge=np.zeros_like(rest);np.add.at(rhs_edge,a,desired);np.add.at(rhs_edge,b,-desired)
        # Jacobi local/global solve with exact inactive boundary and smooth
        # influence. No all-vertex dense system or SciPy dependency.
        candidate=current.copy();strength=regularization*(1-.45*iteration/max(iterations-1,1))
        data_weight=.45*degree*np.minimum(mass,1.6)*influence
        denominator=np.maximum(strength*degree+data_weight,1e-20)
        for _ in range(20):
            neighbors=np.zeros_like(rest);np.add.at(neighbors,a,candidate[b]);np.add.at(neighbors,b,candidate[a])
            proposal=(strength*(neighbors+rhs_edge)+data_weight[:,None]*data)/denominator[:,None]
            candidate[active]=.65*proposal[active]+.35*candidate[active]
            candidate[~active]=rest[~active]
        candidate=current+(candidate-current)*influence[:,None]
        if weld is not None:
            delta=candidate-rest
            candidate=rest+np.column_stack([np.bincount(weld,weights=delta[:,i],minlength=len(counts))/counts for i in range(3)])[weld]
        candidate,collision,distortion=limit_step(current,candidate,triangles,reference_quality,
            guard,allowed_crossings,weld=weld,orientation=orientation)
        collision_limited+=collision;distortion_limited+=distortion
        current=candidate
        history.append({'forward_rms':float(np.sqrt(np.mean(np.square(forward_dist)))) if forward_dist else None,
                        'reverse_rms':float(np.sqrt(np.mean(np.square(reverse_dist)))) if reverse_dist else None,
                        'forward_coverage':len(forward_dist)/max(int(active.sum()),1),
                        'reverse_coverage':len(reverse_dist)/max(len(samples),1)})
    final_metrics=measure(current)
    final_metrics['crossings']=crossings(current,triangles,weld,tolerance=guard.collision_tolerance)
    final_metrics['degraded_triangle_fraction']=float(np.mean(triangle_quality(current,triangles)<reference_quality*.2))
    before_error=initial_metrics['forward_rms']+initial_metrics['reverse_rms']
    after_error=final_metrics['forward_rms']+final_metrics['reverse_rms']
    measurable_progress=after_error<before_error-max(before_error*1e-6,span*1e-8)
    already_aligned=before_error<=span*.002
    accepted=(final_metrics['forward_rms']+final_metrics['reverse_rms'] <=
              initial_metrics['forward_rms']+initial_metrics['reverse_rms'] and
              final_metrics['crossings']<=initial_metrics['crossings'] and final_metrics['degraded_triangle_fraction']<.005 and
              (measurable_progress or already_aligned))
    return (current if accepted else rest),{'solver':'PAIRED_SYMMETRIC_ARAP_V1','iterations':len(history),'source_samples':len(samples),
                    'accepted':accepted,'before':initial_metrics,'after':final_metrics,
                    'measurable_progress':bool(measurable_progress),'already_aligned':bool(already_aligned),
                    'reason':'IMPROVED' if measurable_progress else ('ALREADY_ALIGNED' if already_aligned else 'NO_SAFE_PROGRESS'),
                    'max_movement':float(np.linalg.norm(current-rest,axis=1).max()),
                    'normal_cosine':float(normal_cosine),
                    'supported_result_fraction':history[-1]['forward_coverage'],
                    'unobserved_surface_policy':'RETAIN_LOCAL_STRUCTURE',
                    'review_required':bool(final_metrics['crossings'] or final_metrics['source_coverage']<.95
                                           or final_metrics['forward_p95']>span*.02),
                    'limited_steps':guard.limited_steps,'collision_limited_steps':collision_limited,
                    'distortion_limited_steps':distortion_limited,'history':history}


def prepare_pairs(context, source, target, current=None, *, limits=None, names=None):
    """Copy all Blender-owned data on the main thread before numerical work."""
    available=paired(source,target)
    names=available if names is None else list(names)
    if len(names)!=len(set(names)) or any(name not in available for name in names):
        raise ValueError('Choose unique paired region names present on source and result')
    if not names:raise ValueError('Mark matching CM_Fit_ regions on the source and result')
    current=m.evaluated_coordinates(context,target) if current is None else np.asarray(current).copy()
    rest=m.coordinates(target)
    tolerance=max(float(np.linalg.norm(np.ptp(rest,axis=0)))*1e-6,1e-9)
    _,weld=np.unique(np.rint(rest/tolerance).astype(np.int64),axis=0,return_inverse=True)
    for obj in (source,target):
        evaluated=obj.evaluated_get(context.evaluated_depsgraph_get());mesh=evaluated.to_mesh()
        try:
            if m.topology_data(mesh)!=m.topology(obj):
                raise ValueError('Disable topology-changing modifiers before paired region fitting: '+obj.name)
        finally:evaluated.to_mesh_clear()
    target.data.calc_loop_triangles();triangles=np.array([t.vertices[:] for t in target.data.loop_triangles])
    source.data.calc_loop_triangles();source_triangles=np.array([t.vertices[:] for t in source.data.loop_triangles])
    source_points=m.evaluated_coordinates(context,source)
    if len(source_points)!=len(source.data.vertices):raise ValueError('Paired fitting requires source base topology; disable topology-changing modifiers')
    tasks=[]
    for name in names:
        sw=weights(source,name);tw=weights(target,name)*(1-m.pin_weights(target))
        if limits is not None:tw*=limits
        shared=np.ones(int(weld.max())+1);np.minimum.at(shared,weld,tw);tw=shared[weld]
        from . import source_regions
        source_mask=np.all(sw[source_triangles]>.25,axis=1)
        exact=source_regions.triangle_mask(source,name)
        if exact is not None:source_mask&=exact
        source_face_ids=np.flatnonzero(source_mask)
        selected=source_triangles[source_face_ids]
        if not len(selected):raise ValueError('Source region has no complete faces: '+name)
        # Solve only the labelled patch and its pinned one-ring boundary.
        # Running batched rotations over the entire character for each finger
        # wastes memory/time and makes an interactive refinement appear hung.
        local_faces=triangles[np.any(tw[triangles]>0,axis=1)]
        ids=np.unique(local_faces)
        # UV copies must share the graph, not merely their final positions.
        # Independent seam rotations/Laplacians followed by averaging create
        # artificial crease boundaries even when positional continuity passes.
        families,local_faces=np.unique(weld[local_faces],return_inverse=True)
        local_faces=local_faces.reshape(-1,3)
        local_faces=local_faces[(local_faces[:,0]!=local_faces[:,1])&
                                (local_faces[:,0]!=local_faces[:,2])&
                                (local_faces[:,1]!=local_faces[:,2])]
        # Include every original copy of a participating physical vertex.
        ids=np.flatnonzero(np.isin(weld,families))
        scatter=np.searchsorted(families,weld[ids])
        physical_influence=np.ones(len(families));np.minimum.at(physical_influence,scatter,tw[ids])
        source_ids,selected=np.unique(selected,return_inverse=True)
        selected=selected.reshape(-1,3)
        if not len(ids):raise ValueError('Result region is empty or fully pinned: '+name)
        tasks.append(dict(name=name,ids=ids,scatter=scatter,triangles=local_faces,source=source_points[source_ids],
                          source_triangles=selected,influence=physical_influence,source_face_ids=source_face_ids))
    return current,tasks


def initialize_branch(reference,task,*,cancelled=None,progress=None,frames=None):
    """Guarded intrinsic shape initialization for one explicitly paired branch."""
    from . import surface_transport
    reference=reference.copy();triangles=task['triangles'];influence=task['influence']
    cloud=task['source'][np.unique(task['source_triangles'])]
    if np.count_nonzero(influence>.25)<12:raise ValueError('Branch needs at least twelve supported result vertices')
    hints=frames or surface_transport.infer_frames(reference[influence>.25],cloud)
    profile_faces=triangles[np.all(influence[triangles]>.25,axis=1)]
    if not len(profile_faces):raise ValueError('Branch needs complete supported result faces')
    desired,details=surface_transport.transport(reference,reference,task['source'],**hints,
        reference_triangles=profile_faces,source_triangles=task['source_triangles'])
    desired=reference+(desired-reference)*influence[:,None]
    quality=triangle_quality(reference,triangles);guard=geometry.SurfaceStepGuard(reference,triangles)
    guard.collision_tolerance=float(np.linalg.norm(np.ptp(reference,axis=0))*1e-6)
    allowed=crossings(reference,triangles,pairs=True,tolerance=guard.collision_tolerance);current=reference.copy();limited=0
    for step in range(8):
        if cancelled and cancelled():raise RuntimeError('Branch initialization cancelled; previous geometry retained')
        if progress:progress(step/8)
        proposal=current+(desired-current)/(8-step)
        current,collision,distortion=limit_step(current,proposal,triangles,quality,guard,allowed)
        limited+=collision or distortion
    tree=BVHTree.FromPolygons(task['source'].tolist(),task['source_triangles'].tolist(),all_triangles=True)
    def error(p):
        return float(np.sqrt(np.mean([tree.find_nearest(Vector(v))[3]**2 for v in p[influence>0]])))
    before,after=error(reference),error(current)
    accepted=after<=before
    return (current if accepted else reference),dict(solver='INTRINSIC_BRANCH_TRANSPORT_V1',accepted=accepted,
        before={'forward_rms':before,'crossings':len(allowed)},
        after={'forward_rms':after,'crossings':crossings(current,triangles,tolerance=guard.collision_tolerance)},
        review_required=True,limited_steps=limited,profile=details)


def polish_region(reference,task,*,iterations=8,cancelled=None,progress=None):
    """Bounded Taubin fairing for an accepted patch, with physical boundaries."""
    reference=reference.copy();current=reference.copy();triangles=task['triangles'];influence=task['influence']
    edges=np.unique(np.sort(np.vstack((triangles[:,[0,1]],triangles[:,[1,2]],triangles[:,[2,0]])),axis=1),axis=0)
    quality=triangle_quality(reference,triangles);guard=geometry.SurfaceStepGuard(reference,triangles)
    guard.collision_tolerance=float(np.linalg.norm(np.ptp(reference,axis=0))*1e-6)
    allowed=crossings(reference,triangles,pairs=True,tolerance=guard.collision_tolerance);limited=0
    span=float(np.linalg.norm(np.ptp(task['source'],axis=0)))
    for step in range(max(1,int(iterations))):
        if cancelled and cancelled():raise RuntimeError('Regional polish cancelled; previous geometry retained')
        if progress:progress(step/max(int(iterations),1))
        candidate=current+.35*(geometry.neighbor_average(current,edges)-current)*influence[:,None]
        candidate-=.36*(geometry.neighbor_average(candidate,edges)-candidate)*influence[:,None]
        delta=candidate-reference;length=np.linalg.norm(delta,axis=1)
        delta*=np.minimum(1.,span*.015/np.maximum(length,1e-20))[:,None]
        candidate,collision,distortion=limit_step(current,reference+delta,triangles,quality,guard,allowed)
        limited+=collision or distortion;current=candidate
    tree=BVHTree.FromPolygons(task['source'].tolist(),task['source_triangles'].tolist(),all_triangles=True)
    def error(p):return float(np.sqrt(np.mean([tree.find_nearest(Vector(v))[3]**2 for v in p[influence>0]])))
    before,after=error(reference),error(current)
    accepted=after<=max(before*1.05,before+span*.002)
    return (current if accepted else reference),dict(solver='BOUNDED_REGION_FAIRING_V1',accepted=accepted,
        before={'forward_rms':before,'crossings':len(allowed)},
        after={'forward_rms':after,'crossings':crossings(current,triangles,tolerance=guard.collision_tolerance)},review_required=True,
        limited_steps=limited,max_displacement=float(np.linalg.norm(current-reference,axis=1).max()))


def run_prepared(current,tasks,*,iterations=16,cancelled=None,progress=None,mode='REFINE',frames=None,global_guard=None):
    """Numerical work only: no bpy data, context, operators or UI calls."""
    if mode=='POLISH' and global_guard is not None:
        from . import regional_polish
        return regional_polish.run(current,tasks,global_guard,run_prepared,iterations=iterations,cancelled=cancelled,progress=progress)
    current=current.copy();reports=[]
    for index,task in enumerate(tasks):
        def report_progress(value):
            if progress:progress(task['name'],(index+value)/len(tasks))
        raw=current[task['ids']];scatter=task['scatter'];counts=np.bincount(scatter)
        physical=np.column_stack([np.bincount(scatter,weights=raw[:,i])/counts for i in range(3)])
        if mode=='TRANSPORT':
            fitted,report=initialize_branch(physical,task,cancelled=cancelled,progress=report_progress,
                                           frames=(frames or {}).get(task['name']))
        elif mode=='POLISH':
            fitted,report=polish_region(physical,task,iterations=iterations,cancelled=cancelled,progress=report_progress)
        else:
            fitted,report=solve(physical,task['triangles'],task['source'],task['source_triangles'],
                task['influence'],iterations=iterations,cancelled=cancelled,progress=report_progress)
        current[task['ids']]=raw+(fitted-physical)[scatter]
        report['region']=task['name'];report['physical_vertices']=len(physical)
        report['uv_vertices']=len(raw);reports.append(report)
    return current,reports


def refine_paired(context, source, target, current=None, *, iterations=16, limits=None, names=None):
    """Synchronous API for background wrapping, scripts and batch processing."""
    if not paired(source,target):return current,[]
    current,tasks=prepare_pairs(context,source,target,current,limits=limits,names=names)
    return run_prepared(current,tasks,iterations=iterations)


def running():return _JOB is not None


def cancel():
    if _JOB:_JOB['cancel'].set()


def stamp(context,source,target):
    from .scan_jobs import geometry_stamp
    from . import source_regions
    digest=hashlib.sha256()
    for obj in (source,target):
        digest.update(str(obj.as_pointer()).encode());digest.update(geometry_stamp(context,obj,groups=True).encode())
        digest.update(m.coordinates(obj).tobytes())
        digest.update(str(obj.get('cm_transport_frames','')).encode())
        digest.update(str(obj.get('cm_source_distance_budget','')).encode())
        digest.update(source_regions.stamp(obj))
        for group in obj.vertex_groups:
            if group.name.startswith(PREFIX) or group.name in {'CM_WrapMask','CM_Inner_Parts'}:
                digest.update(group.name.encode());digest.update(weights(obj,group.name).tobytes())
    return digest.hexdigest()


def start(context,target,*,names=None,mode='REFINE',reconstruct=False,repair_target='AUTO',source_observation='NEAREST',observation_rings=2.,complete_missing=False,rebuild_missing=False,register_attachment=False,curvature_polish=False,curvature_model='CURVATURE',attachment_model='AUTO'):
    """Keep the viewport responsive; retain geometry until a checked commit."""
    global _JOB
    from . import character_project,resource_budget,wrap_interiors,workflow
    character_project.preflight(context,context.scene.cm_settings,context.scene.cm_features)
    s,f=context.scene.cm_settings,context.scene.cm_features
    if running():raise ValueError('Finish or cancel the current regional projection')
    m.check_mesh(s.source,'Source');m.check_mesh(target,'Result')
    if target==s.source:raise ValueError('Source and result must be separate meshes; source geometry is fixed')
    if mode not in {'REFINE','TRANSPORT','POLISH','ACCURATE'}:raise ValueError('Unknown regional projection mode')
    if reconstruct and mode!='ACCURATE':raise ValueError('Attachment reconstruction requires Accurate Fit')
    if complete_missing and (mode!='ACCURATE' or reconstruct):raise ValueError('Partial-surface completion requires Accurate Fit without attachment reconstruction')
    if rebuild_missing and not complete_missing:raise ValueError('Missing-shape reconstruction requires partial-surface completion')
    if curvature_polish and (mode!='ACCURATE' or not rebuild_missing or reconstruct):raise ValueError('Source curvature polish requires Accurate Fit with missing-shape reconstruction')
    if curvature_model not in {'CURVATURE','NORMAL_FLOW'}:raise ValueError('Choose CURVATURE or NORMAL_FLOW surface refinement')
    if register_attachment and (not rebuild_missing or target!=f.body_result or names is None or len(names)!=2):raise ValueError('Coarse pelvis registration requires two paired body regions with missing-shape reconstruction')
    if repair_target not in {'AUTO','GAP','ALIGNMENT'}:raise ValueError('Choose AUTO, GAP or ALIGNMENT for attachment repair')
    if attachment_model not in {'AUTO','SOURCE_NORMALS','SOURCE_TRANSPORT','CHART_ONLY'}:raise ValueError('Choose AUTO, SOURCE_NORMALS, SOURCE_TRANSPORT or CHART_ONLY attachment fitting')
    if attachment_model!='AUTO' and (mode!='ACCURATE' or not reconstruct):raise ValueError('Attachment model selection requires Accurate Fit with attachment reconstruction')
    if source_observation not in {'NEAREST','NORMAL_LINE'}:raise ValueError('Choose NEAREST or NORMAL_LINE source correspondence')
    if not np.isfinite(observation_rings) or not .5<=observation_rings<=12:raise ValueError('Choose observation reach between 0.5 and 12 local edges')
    if any(modifier.show_viewport and modifier.type not in {'WEIGHTED_NORMAL','NORMAL_EDIT','DATA_TRANSFER'}
           for modifier in target.modifiers):
        raise ValueError('Apply or disable geometry modifiers before refining the base surface')
    limits=np.ones(len(target.data.vertices))
    if target==f.body_result or s.part=='BODY':pass
    else:
        if f.preserve_mouth_interior:limits[wrap_interiors.mouth_mask(target)]=0
        if s.part=='FULL_BODY':limits[workflow.neck_ring(target)]=0
    if f.is_mask and target.vertex_groups.get('CM_WrapMask'):limits*=weights(target,'CM_WrapMask')
    if mode=='ACCURATE' and s.part=='FULL_BODY':limits[workflow.neck_ring(target)]=0
    current,tasks=prepare_pairs(context,s.source,target,limits=limits,names=names)
    resource_budget.require('regional surface projection',128*resource_budget.MIB+sum(
        value.nbytes for task in tasks for value in task.values() if isinstance(value,np.ndarray))*12)
    accurate_data=None;companion=None
    if mode=='ACCURATE':
        from pathlib import Path
        from . import setup_paths,surface_fit_jobs
        setup_paths.resolve(f);python=Path(bpy.path.abspath(f.inference_python))
        if not python.is_file():raise ValueError('Install optional accurate fitting with tools/setup_surface_solver.py first')
        project=Path(bpy.path.abspath(s.project_dir)).resolve()
        resource_budget.require_disk('accurate regional fitting',project,128*resource_budget.MIB)
        accurate_data,companion=surface_fit_jobs.prepare(context,s.source,target,current,tasks)
        if reconstruct:
            from . import native_storage
            accurate_data.update(native_storage.snapshot(context,target,'Accurate Regional Fit'))
        if register_attachment:
            from .body_pose import pelvis_attachment_frame
            accurate_data['attachment_frame']=pelvis_attachment_frame(f.body_template)
        resource_budget.require('accurate regional fitting',256*resource_budget.MIB+sum(a.nbytes for a in accurate_data.values())*64)
        options=dict(regions=[task['name'] for task in tasks],iterations=48 if register_attachment else min(s.iterations,16),search_distance=s.max_distance,reconstruct=bool(reconstruct),repair_target=repair_target,attachment_model=attachment_model,source_observation=source_observation,observation_rings=float(observation_rings),complete_missing=bool(complete_missing),rebuild_missing=bool(rebuild_missing),register_attachment=bool(register_attachment),curvature_polish=bool(curvature_polish),curvature_model=curvature_model)
    from . import regional_native_budget,regional_attempt
    local_budget=regional_native_budget.snapshot(context,s.source,target,current,tasks,for_polish=mode=='POLISH') if mode!='ACCURATE' else None
    polish_guard=regional_native_budget.contact_snapshot(target) if mode=='POLISH' else None
    if polish_guard is not None:polish_guard['source_budget']=local_budget
    if not bpy.app.background and context.preferences.edit.use_global_undo:
        window=context.window or next((w for w in context.window_manager.windows if w.scene==context.scene),None)
        if not window:raise ValueError('Open the character window before fitting a region')
        with context.temp_override(window=window):bpy.ops.ed.undo_push(message='Start Regional Fit')
    job={'scene':context.scene,'source':s.source,'target':target,'stamp':stamp(context,s.source,target),
         'cancel':threading.Event(),'progress':('Starting',0.),'complete':False,'error':None,'mode':mode,'reconstruct':bool(reconstruct),'complete_missing':bool(complete_missing),'register_attachment':bool(register_attachment)}
    job['curvature_polish']=bool(curvature_polish)
    job['curvature_model']=curvature_model
    job['regions']=[task['name'] for task in tasks];job['input_reference']=current
    if mode=='POLISH':job['native_tasks']=tasks
    if register_attachment:job.update(attachment_template=f.body_template,attachment_frame=accurate_data['attachment_frame'].copy())
    if companion:
        from .scan_jobs import geometry_stamp
        job.update(companion=companion,companion_stamp=geometry_stamp(context,companion,groups=True))
    iterations=min(s.iterations,24)
    frames=json.loads(target.get('cm_transport_frames','{}'))
    def work():
        started=time.monotonic()
        try:
            if mode=='ACCURATE':
                output,reports,proof=surface_fit_jobs.run(accurate_data,options,python,project,cancelled=job['cancel'].is_set,
                    progress=lambda name,value:job.update(progress=(name,value)))
                job['result']=(output,reports)
                if proof:job['native_validator']=lambda actual:surface_fit_jobs.validate_native(actual,proof,reports)
                if proof:
                    job['native_retractor']=lambda candidate:surface_fit_jobs.retract_native(candidate,proof)
                    if 'source_budget_context' in proof:
                        from . import source_budget
                        previous=bytes(accurate_data['source_budget_previous']).decode()
                        job['native_properties']={source_budget.PROPERTY:source_budget.updated(previous,proof,certify_observations=True)}
                job['input_reference']=current
            else:
                job['result']=run_prepared(current,tasks,iterations=iterations,cancelled=job['cancel'].is_set,
                    progress=lambda name,value:job.update(progress=(name,value)),mode=mode,frames=frames,global_guard=polish_guard)
                job['native_validator']=regional_native_budget.validator(local_budget,current,job['result'][0])
                if mode=='POLISH':job['native_properties']=regional_native_budget.properties(local_budget,current,job['result'][0])
        except Exception as exc:job['error']=str(exc)
        finally:job['elapsed_seconds']=time.monotonic()-started;job['complete']=True
    job['thread']=threading.Thread(target=work,name='CreateMH Regional Projection',daemon=True)
    _JOB=job;context.scene[_PENDING]=True;regional_attempt.publish(job,'RUNNING');job['thread'].start();bpy.app.timers.register(poll,first_interval=.2)
    s.status='Regional projection running; viewport remains interactive. Source is fixed.'


def poll():
    global _JOB
    from . import regional_attempt
    job=_JOB
    if not job:return None
    if not job['complete']:
        name,value=job['progress']
        job['scene'].cm_settings.status='Regional projection: '+name+' '+str(round(value*100))+'%'
        return .25
    try:
        scene=job['scene'];s,f=scene.cm_settings,scene.cm_features
        if job['cancel'].is_set():raise ValueError('Regional projection cancelled; previous geometry retained')
        if job['error']:raise ValueError(job['error'])
        window=next((w for w in bpy.context.window_manager.windows if w.scene==scene),None)
        if not window:raise ValueError('Character window changed; previous geometry retained')
        with bpy.context.temp_override(window=window,scene=scene,view_layer=scene.view_layers[0]):
            if bpy.context.mode!='OBJECT' or s.source!=job['source'] or job['target'] not in (s.result,f.head_result,f.body_result):
                raise ValueError('Character or edit mode changed; projection discarded')
            if stamp(bpy.context,job['source'],job['target'])!=job['stamp']:
                raise ValueError('Geometry or region masks changed during projection; result discarded')
            if job.get('companion'):
                from .scan_jobs import geometry_stamp
                if geometry_stamp(bpy.context,job['companion'],groups=True)!=job['companion_stamp']:raise ValueError('Companion surface changed; accurate fit discarded')
            if job.get('register_attachment'):
                from .body_pose import pelvis_attachment_frame
                if f.body_template!=job['attachment_template'] or not np.array_equal(pelvis_attachment_frame(f.body_template),job['attachment_frame']):raise ValueError('Reference anatomy changed; attachment registration discarded')
            output,reports=job['result'];target=job['target']
            for report in reports:report['job_wall_seconds']=job.get('elapsed_seconds')
            if job['mode']=='POLISH' and job.get('input_reference') is not None and np.array_equal(output,job['input_reference']):
                for report in reports:report['accepted']=False;report['reason']='Polish found no safe geometric change.'
            if not any(report['accepted'] for report in reports):
                target['cm_region_projection']=json.dumps(reports)
                target['cm_region_projection_'+job['mode'].lower()]=json.dumps(reports)
                s.status=('Accurate fit retained the current surface; no verified improvement was found.' if job['mode']=='ACCURATE'
                    else 'No safe movement found. Adjust the matching region masks or landmarks before retrying.')
                if job.get('reconstruct'):s.status='Attachment repair retained the current surface. '+reports[0].get('reason','No verified local improvement was found.')
                if job.get('complete_missing'):s.status='Partial-surface completion retained the current surface. '+reports[0].get('reason','No verified improvement was found.')
                regional_attempt.publish(job,'REJECTED')
                return None
            from . import native_surface_commit
            key_name={'REFINE':'Regional Projection','TRANSPORT':'Branch Initialization','POLISH':'Regional Polish','ACCURATE':'Accurate Regional Fit'}[job['mode']]
            if job['mode'] in {'ACCURATE','POLISH'}:
                from . import native_fit_step
                reference=job.get('input_reference')
                if reference is None:reference=m.evaluated_coordinates(bpy.context,target).copy()
                native_report=native_fit_step.commit(bpy.context,target,key_name,output,reference=reference,validator=job.get('native_validator'),retractor=job.get('native_retractor'),property_updates=job.get('native_properties'))
            else:
                native_report=native_surface_commit.commit(bpy.context,target,key_name,output,reference=job.get('input_reference'),validator=job.get('native_validator'),property_updates=job.get('native_properties'));native_report['step_fraction']=1.
            job['native_committed']=True
            job['native_report']=native_report
            target['cm_region_projection_native']=json.dumps(native_report)
            if job['mode']=='POLISH' and job.get('native_tasks'):
                from . import regional_native_budget
                regional_native_budget.polish_metrics(bpy.context,target,job['input_reference'],job['native_tasks'],reports,native_report)
            if job['mode']=='ACCURATE':
                bounds=native_report['surface_validation']['surface_distance_upper_bounds']
                for report in reports:
                    report['planned_after']=report['after'];report['planned_surface_measurement']=report.pop('surface_measurement')
                    report['after']=dict(forward_rms=bounds['forward']['rms_m'],reverse_rms=bounds['reverse']['rms_m'],
                        crossings=native_report['crossings_after'],distance_upper_bounds=True)
                    report['native_step_fraction']=native_report['step_fraction'];report['native_surface_measurement']=bounds
                    if 'source_alignment' in native_report['surface_validation']:
                        report['planned_source_alignment']=report.get('source_alignment')
                        report['source_alignment']=native_report['surface_validation']['source_alignment']
                    if 'observed_surface_bounds' in native_report['surface_validation']:
                        report['planned_observed_after']=report.get('observed_after')
                        report['observed_after']=native_report['surface_validation']['observed_surface_bounds']
            target['cm_region_projection_native']=json.dumps(native_report)
            target['cm_region_projection']=json.dumps(reports)
            target['cm_region_projection_'+job['mode'].lower()]=json.dumps(reports)
            regional_attempt.publish(job,'COMMITTED',native=native_report)
            rejected=sum(not r['accepted'] for r in reports)
            review=sum(r['review_required'] for r in reports)
            s.status='Regional projection finished; '+str(rejected)+' rejected, '+str(review)+' need review. Inspect neutral geometry before baking.'
            if job.get('reconstruct'):
                alignment=native_report.get('surface_validation',{}).get('source_alignment',{})
                remaining=alignment.get('after_severe_faces')
                if remaining is not None:
                    s.status=('Attachment repair applied; '+str(remaining)+' severe triangles remain in checked regions. Inspect shape before baking.'
                        if remaining else 'Attachment repair applied. Inspect shape and detail before baking.')
            if job.get('register_attachment') and not job.get('curvature_polish'):
                registration=native_report['surface_validation']['attachment_registration']
                s.status='Hip attachment draft: remaining height offset '+format(registration['after_height_gap_m']*1000,'.1f')+' mm. Inspect shape before baking.'
            if job.get('curvature_polish'):
                target['cm_geometry_review_required']=True
                s.status='Experimental curvature draft. Inspect seams and shape before baking; Undo restores the previous surface and limits.'
                if job.get('curvature_model')=='NORMAL_FLOW':s.status='Surface continuity draft. Inspect inferred joins and skin coverage before baking.'
            scene.pop(_PENDING,None)
            if not bpy.app.background and bpy.context.preferences.edit.use_global_undo:
                bpy.ops.ed.undo_push(message={'REFINE':'Refine Region','TRANSPORT':'Initialize Branch Shape','POLISH':'Polish Region','ACCURATE':'Accurate Fit Region'}[job['mode']])
    except Exception as exc:
        try:
            published=bool(job.get('native_committed'))
            job['scene'].cm_settings.status=('Fit committed, but report refresh failed: ' if published else '')+str(exc)
            regional_attempt.publish(job,'COMMITTED_WITH_REPORT_ERROR' if published else 'CANCELLED' if job['cancel'].is_set() else 'FAILED',error=exc,native=job.get('native_report'))
        except ReferenceError:pass
    finally:
        try:job['scene'].pop(_PENDING,None)
        except ReferenceError:pass
        _JOB=None
    return None


@persistent
def reset_on_load(_):
    global _JOB
    job=_JOB;cancel();_JOB=None
    if job:
        try:
            from . import regional_attempt
            regional_attempt.publish(job,'CANCELLED',error='Regional fit interrupted by loading or history.')
            job['scene'].pop(_PENDING,None)
        except ReferenceError:pass
    if bpy.app.timers.is_registered(poll):bpy.app.timers.unregister(poll)


@persistent
def recover_history(_=None):
    """Clear restored progress markers; worker jobs are not history state."""
    for scene in bpy.data.scenes:
        if _JOB and _JOB['scene']==scene:continue
        s=getattr(scene,'cm_settings',None)
        f=getattr(scene,'cm_features',None)
        from . import regional_attempt
        if s and f:
            for target in {obj for obj in (s.result,f.head_result,f.body_result) if obj}:regional_attempt.recover(target)
        pending=bool(scene.pop(_PENDING,None))
        if s and (pending or s.status.startswith(('Regional projection running','Regional projection:'))):
            s.status='Regional fit interrupted or reverted; current surface retained.'


def register():
    for handlers in (bpy.app.handlers.load_pre,bpy.app.handlers.undo_pre,bpy.app.handlers.redo_pre):
        if reset_on_load not in handlers:handlers.append(reset_on_load)
    for handlers in (bpy.app.handlers.load_post,bpy.app.handlers.undo_post,bpy.app.handlers.redo_post):
        if recover_history not in handlers:handlers.append(recover_history)
    if not bpy.app.timers.is_registered(recover_history):bpy.app.timers.register(recover_history,first_interval=.1)


def unregister():
    reset_on_load(None)
    if bpy.app.timers.is_registered(recover_history):bpy.app.timers.unregister(recover_history)
    for handlers in (bpy.app.handlers.load_pre,bpy.app.handlers.undo_pre,bpy.app.handlers.redo_pre):
        if reset_on_load in handlers:handlers.remove(reset_on_load)
    for handlers in (bpy.app.handlers.load_post,bpy.app.handlers.undo_post,bpy.app.handlers.redo_post):
        if recover_history in handlers:handlers.remove(recover_history)


class CM_OT_seed_anatomical_region(bpy.types.Operator):
    bl_idname='cm.seed_anatomical_region'
    bl_label='Use Anatomical Limb Region'
    bl_description='Replace the selected body result region from its reference skeleton, preserving attachment edges and hard pins; mark the corresponding source patch separately'
    bl_options={'REGISTER','UNDO'}
    root:bpy.props.EnumProperty(name='Limb',items=[(part+'_'+side,label+' '+side.upper(),'')
        for part,label in (('upperarm','Whole Arm'),('upperarm_only','Upper Arm and Shoulder'),('lowerarm','Forearm and Hand'),('hand','Hand'),
                           ('thigh','Whole Leg'),('thigh_only','Thigh, Hip Blend and Knee'),('thigh_surface','Thigh Surface and Knee'),('knee','Knee Transition'),('calf','Calf and Foot'),('foot','Foot')) for side in ('l','r')],default='hand_l')
    attachment_rings:bpy.props.IntProperty(name='Attachment Collar Rings',description='Blend the fitting region into neighboring attachment topology while protecting distal anatomy',default=4,min=0,max=12)

    def execute(self,context):
        from . import workflow,body_pose,bake_jobs
        s,f=context.scene.cm_settings,context.scene.cm_features;target=context.active_object
        try:
            if running() or bake_jobs.running():raise ValueError('Finish or cancel the current geometry/bake job before changing region masks')
            if context.mode!='OBJECT':raise ValueError('Return to Object Mode before creating an anatomical region')
            if not target or target!=f.body_result or target.type!='MESH':raise ValueError('Select the wrapped body result')
            template=f.body_template
            if not template or m.topology(template)!=m.topology(target):raise ValueError('Use the canonical body result before topology-changing detail')
            shoulder=self.root.startswith('upperarm_only_');thigh=self.root.startswith('thigh_only_');knee=self.root.startswith('knee_');surface=self.root.startswith('thigh_surface_')
            if surface:values=body_pose.thigh_surface_region(template,self.root[-1],protected=m.pin_weights(target)>=.999)
            elif shoulder:values=body_pose.upperarm_shoulder_region(template,self.root[-1],protected=m.pin_weights(target)>=.999,attachment_rings=self.attachment_rings)
            elif thigh:values=body_pose.thigh_knee_region(template,self.root[-1],protected=m.pin_weights(target)>=.999,attachment_rings=self.attachment_rings)
            elif knee:values=body_pose.knee_region(template,self.root[-1],protected=m.pin_weights(target)>=.999)
            else:values=body_pose.subtree_weights(template,[self.root],mode='ANATOMICAL',protected=m.pin_weights(target)>=.999)[self.root]
            target.data.calc_loop_triangles()
            triangles=np.array([tuple(t.vertices) for t in target.data.loop_triangles],dtype=np.int64).reshape(-1,3)
            if not np.any(np.all(values[triangles]>.01,axis=1)):raise ValueError('No complete supported faces in this anatomical region')
            name=('ThighSurface' if surface else 'Shoulder' if shoulder else 'ThighKnee' if thigh else self.root.rsplit('_',1)[0].title())+'_'+self.root[-1].upper()
            workflow.write_group(target,PREFIX+name,values);f.projection_region=name
            target['cm_fit_region_'+name]=json.dumps(dict(method='DOMINANT_REFERENCE_SKELETON_SURFACE' if surface else 'REFERENCE_SKELETON_TERRITORY_WITH_ATTACHMENT_COLLAR',root=self.root,source_labels_inferred=False,distal_anatomy_protected=shoulder or thigh or knee or surface,attachment_rings=self.attachment_rings if shoulder or thigh else 0))
            s.status='Created '+name+' on the body. Mark the matching source patch, then fit this region.'
            return {'FINISHED'}
        except Exception as exc:
            s.status=str(exc);self.report({'ERROR'},str(exc));return {'CANCELLED'}


class CM_OT_mark_fit_region(bpy.types.Operator):
    bl_idname='cm.mark_fit_region'
    bl_label='Mark Selected Fit Region'
    bl_description='Mark exact selected source faces or selected result vertices using the region name; replaces this region mask only'
    bl_options={'REGISTER','UNDO'}

    def execute(self,context):
        from . import workflow
        s,f=context.scene.cm_settings,context.scene.cm_features
        obj=context.active_object
        try:
            if running():raise ValueError('Finish or cancel regional projection before changing its masks')
            if obj not in (s.source,s.result,f.head_result,f.body_result,s.template,f.body_template) or not obj or obj.type!='MESH':
                raise ValueError('Select the source, result or template mesh')
            if context.mode!='EDIT_MESH':raise ValueError('In Edit Mode select the vertices for this region first')
            name=re.sub(r'[^A-Za-z0-9_]+','_',f.projection_region.strip()).strip('_')
            if not name:raise ValueError('Give the paired region a name, such as Hand_L')
            import bmesh
            bm=bmesh.from_edit_mesh(obj.data);bm.verts.ensure_lookup_table();bm.verts.index_update()
            selected=[v.index for v in bm.verts if v.select]
            if len(selected)<3:raise ValueError('Select at least three vertices forming complete faces')
            source_faces=None
            if obj==s.source:
                bm.faces.ensure_lookup_table();bm.faces.index_update()
                source_faces=np.asarray([face.index for face in bm.faces if face.select],dtype=np.int64)
                if not len(source_faces):raise ValueError('Select complete source faces for this region')
            values=np.zeros(len(obj.data.vertices));values[selected]=1
            # VertexGroup.add is unavailable in Edit Mode. Return to Edit Mode
            # afterwards so marking the second patch remains a familiar action.
            bpy.ops.object.mode_set(mode='OBJECT')
            old_group=weights(obj,PREFIX+name) if obj.vertex_groups.get(PREFIX+name) else None
            try:
                workflow.write_group(obj,PREFIX+name,values)
                if source_faces is not None:
                    from . import source_regions
                    source_regions.write(obj,PREFIX+name,source_faces)
            except BaseException:
                if old_group is not None:workflow.write_group(obj,PREFIX+name,old_group)
                elif obj.vertex_groups.get(PREFIX+name):obj.vertex_groups.remove(obj.vertex_groups[PREFIX+name])
                raise
            finally:bpy.ops.object.mode_set(mode='EDIT')
            s.status=('Marked '+str(len(source_faces))+' exact source faces for '+name+'. ' if source_faces is not None else 'Marked '+name+' on '+obj.name+'. ')+'Mark the matching patch, then return to Object Mode and refine.'
            return {'FINISHED'}
        except Exception as exc:
            s.status=str(exc);self.report({'ERROR'},str(exc));return {'CANCELLED'}


class CM_OT_cancel_region_projection(bpy.types.Operator):
    bl_idname='cm.cancel_region_projection'
    bl_label='Cancel Regional Projection'
    def execute(self,context):cancel();return {'FINISHED'}


class CM_OT_fit_named_region(bpy.types.Operator):
    bl_idname='cm.fit_named_region'
    bl_label='Fit Named Region'
    bl_description='Fit the named paired patch on the selected head/body result'
    mode:bpy.props.EnumProperty(items=[('REFINE','Refine','Conservative paired surface fitting'),
        ('ACCURATE','Accurate Fit','Fit source shape and detail together in a cancellable external worker'),
        ('TRANSPORT','Initialize Branch Shape','Intrinsic cross-section mapping for a labelled finger, toe or limb'),
        ('POLISH','Polish Region','Bounded fairing with fixed boundaries and source-distance limits')],default='REFINE')
    reconstruct:bpy.props.BoolProperty(name='Repair Attachment',description='Reconstruct the worst supported local source patch using an oriented intrinsic chart and constrained fitting',default=False,options={'HIDDEN'})
    complete_missing:bpy.props.BoolProperty(name='Complete Partial Surface',description='Fit reliable source observations and continue their displacement through missing coverage; inferred regions require artist review',default=False,options={'HIDDEN'})
    rebuild_missing:bpy.props.BoolProperty(name='Rebuild Missing Shape',description='Infer screened-bending positions through missing source coverage while retaining broad volume and supported detail',default=False,options={'HIDDEN'})
    curvature_polish:bpy.props.BoolProperty(name='Source Curvature Polish',description='Match measured source curvature and surface coverage without another coarse alignment',default=False,options={'HIDDEN'})
    curvature_model:bpy.props.EnumProperty(name='Surface Refinement',items=(('CURVATURE','Source Curvature','Match source-derived curvature'),('NORMAL_FLOW','Surface Continuity','Continue source normals through unobserved areas as a soft shape prior')),default='CURVATURE',options={'HIDDEN'})
    attachment_model:bpy.props.EnumProperty(name='Attachment Fit Model',items=[('AUTO','Automatic','Compare source-following and chart fitting'),('SOURCE_NORMALS','Preserve Source Angles','Fit fixed source tangent directions and preserve initially aligned faces'),('CHART_ONLY','Chart Fitting','Fit chart and source coverage without additional source-angle constraints'),('SOURCE_TRANSPORT','Follow Source Surface','Transport original angle limits along the original source facets')],default='AUTO',options={'HIDDEN'})
    repair_target:bpy.props.EnumProperty(name='Repair Target',items=[('AUTO','Automatic','Prioritize supported folds, otherwise source gaps'),('GAP','Source Gap','Repair the largest supported source gap'),('ALIGNMENT','Surface Fold','Repair a coherent source-normal alignment defect')],default='AUTO',options={'HIDDEN'})
    source_observation:bpy.props.EnumProperty(name='Source Correspondence',items=[('NEAREST','Nearest Facing Surface','Nearest facing source facets'),('NORMAL_LINE','Normal-Line Surface','Facing intersections along surface normals; preserve surface station across gaps')],default='NEAREST')
    observation_rings:bpy.props.FloatProperty(name='Observation Reach (Edges)',description='Fade distant observations using the median local edge length; initialize large shape differences before refinement',default=2.,min=.5,max=12.)
    def execute(self,context):
        s,f=context.scene.cm_settings,context.scene.cm_features
        try:
            target=context.active_object
            if target not in (s.result,f.head_result,f.body_result):raise ValueError('Select the head or body result first')
            name=PREFIX+re.sub(r'[^A-Za-z0-9_]+','_',f.projection_region.strip()).strip('_')
            start(context,target,names=[name],mode=self.mode,reconstruct=self.reconstruct,repair_target=self.repair_target,attachment_model=self.attachment_model,source_observation=self.source_observation,observation_rings=self.observation_rings,complete_missing=self.complete_missing,rebuild_missing=self.rebuild_missing,curvature_polish=self.curvature_polish,curvature_model=self.curvature_model)
            return {'FINISHED'}
        except Exception as exc:s.status=str(exc);self.report({'ERROR'},str(exc));return {'CANCELLED'}


class CM_OT_align_pelvis_attachment(bpy.types.Operator):
    bl_idname='cm.align_pelvis_attachment'
    bl_label='Align Paired Hips'
    bl_description='Coarse source-backed pelvis attachment registration using the reference rig and both paired thigh regions; inspect the anatomical proposal before final baking'
    left_region:bpy.props.StringProperty(name='Left Region',default='ThighKnee_L')
    right_region:bpy.props.StringProperty(name='Right Region',default='ThighKnee_R')
    curvature_polish:bpy.props.BoolProperty(name='Polish Existing Surface',description='Preserve current attachment anchors and match source curvature instead of another coarse registration',default=False,options={'HIDDEN'})
    curvature_model:bpy.props.EnumProperty(name='Surface Refinement',items=(('CURVATURE','Source Curvature','Match source-derived curvature'),('NORMAL_FLOW','Surface Continuity','Continue source normals through unobserved areas as a soft shape prior')),default='CURVATURE',options={'HIDDEN'})
    def execute(self,context):
        s,f=context.scene.cm_settings,context.scene.cm_features
        try:
            target=context.active_object
            if target!=f.body_result:raise ValueError('Select the canonical body result first')
            if not f.body_template or m.topology(f.body_template)!=m.topology(target):raise ValueError('Pelvis attachment registration requires the matching canonical body template')
            names=[PREFIX+re.sub(r'[^A-Za-z0-9_]+','_',name.strip()).strip('_') for name in (self.left_region,self.right_region)]
            if len(set(names))!=2:raise ValueError('Choose two distinct paired thigh regions')
            if any(not obj or not obj.vertex_groups.get(name) for obj in (s.source,target) for name in names):raise ValueError('Mark both matching thigh regions on source and body before aligning their attachment')
            start(context,target,names=names,mode='ACCURATE',complete_missing=True,rebuild_missing=True,register_attachment=True,source_observation='NORMAL_LINE',curvature_polish=self.curvature_polish,curvature_model=self.curvature_model)
            return {'FINISHED'}
        except Exception as exc:s.status=str(exc);self.report({'ERROR'},str(exc));return {'CANCELLED'}


class CM_OT_review_source_region(bpy.types.Operator):
    bl_idname='cm.review_source_region'
    bl_label='Review Source Region'
    bl_description='Report exact permitted source components or select them for inspection; never removes geometry or changes fitting masks'
    bl_options={'REGISTER','UNDO'}
    action:bpy.props.EnumProperty(name='Review',items=[('REPORT','Inspect Components','Report all connected source patches without changing selection'),('ISLANDS','Select Small Patches','Select small disconnected source patches for review, without changing fit masks'),('COMPONENT','Select Component','Select one original source component by its stable polygon ID')],default='REPORT')
    area_fraction:bpy.props.FloatProperty(name='Relative Patch Area',description='Review patches up to this fraction of the largest connected patch; size does not determine validity',default=.01,min=0.,max=1.)
    component_id:bpy.props.IntProperty(name='Component ID',description='Smallest original polygon ID of the component, from the review report',default=0,min=0)
    stitch_exact_seams:bpy.props.BoolProperty(name='Connect Exact Seams',description='Treat edges with byte-exact matching endpoint coordinates as connected for this review only; no mesh welding or proximity threshold',default=True)

    def execute(self,context):
        from . import source_regions,source_region_review,bake_jobs
        from .vertex_contract import fingerprint
        s,f=context.scene.cm_settings,context.scene.cm_features;source=s.source
        try:
            if running() or bake_jobs.running():raise ValueError('Finish or cancel the geometry/bake job before reviewing source regions')
            if context.mode!='OBJECT':raise ValueError('Return to Object Mode before reviewing source regions')
            if not source or source.type!='MESH':raise ValueError('Choose a source mesh first')
            name=PREFIX+re.sub(r'[^A-Za-z0-9_]+','_',f.projection_region.strip()).strip('_')
            if name==PREFIX or not source.vertex_groups.get(name):raise ValueError('Choose a marked source region')
            source.data.calc_loop_triangles();tri=np.array([t.vertices[:] for t in source.data.loop_triangles],dtype=np.int64).reshape(-1,3);polygons=np.array([t.polygon_index for t in source.data.loop_triangles],dtype=np.int64)
            points=m.evaluated_coordinates(context,source)
            if len(points)!=len(source.data.vertices):raise ValueError('Disable topology-changing source modifiers before reviewing fitting authority')
            evaluated=source.evaluated_get(context.evaluated_depsgraph_get());mesh=evaluated.to_mesh()
            try:
                if m.topology_data(mesh)!=m.topology(source):raise ValueError('Disable topology-changing source modifiers before reviewing fitting authority')
            finally:evaluated.to_mesh_clear()
            selected=np.all(weights(source,name)[tri]>.25,axis=1);exact=source_regions.triangle_mask(source,name)
            if exact is not None:selected&=exact
            report=source_region_review.components(points,tri,polygons,selected,stitch_exact_seams=self.stitch_exact_seams)
            report.update(region=name,source_object=source.name,coordinates=fingerprint(points),triangles=fingerprint(tri),policy=fingerprint(selected),has_exact_face_authority=exact is not None)
            rows=report['components'];chosen=[]
            if self.action=='ISLANDS':chosen=[r for r in rows[1:] if r['area_fraction_of_largest']<=self.area_fraction]
            elif self.action=='COMPONENT':
                chosen=[r for r in rows if r['component_id']==self.component_id]
                if not chosen:raise ValueError('Component ID not found; inspect this region again')
            ids=sorted({i for row in chosen for i in row['polygon_ids']})
            report.update(action=self.action,selected_component_ids=[r['component_id'] for r in chosen],selected_polygons=len(ids))
            # No selection/context change when there is nothing to review.
            if ids:
                if not context.view_layer.objects.get(source.name) or source.hide_get():raise ValueError('Make the source visible in the current view layer before selecting patches')
                for obj in context.selected_objects:obj.select_set(False)
                source.select_set(True);context.view_layer.objects.active=source
                mask=np.zeros(len(source.data.polygons),dtype=bool);mask[ids]=True
                source.data.polygons.foreach_set('select',mask)
                source.data.vertices.foreach_set('select',np.zeros(len(source.data.vertices),dtype=bool));source.data.edges.foreach_set('select',np.zeros(len(source.data.edges),dtype=bool))
                context.tool_settings.mesh_select_mode=(False,False,True);bpy.ops.object.mode_set(mode='EDIT')
            source['cm_source_region_review']=json.dumps(report,separators=(',',':'))
            s.status=str(len(rows))+' source patches in '+name.removeprefix(PREFIX)+'. '+(str(len(ids))+' faces selected for review. ' if ids else '')+'Fit masks unchanged; small patches may be valid skin.'
            return {'FINISHED'}
        except Exception as exc:s.status=str(exc);self.report({'ERROR'},str(exc));return {'CANCELLED'}


CLASSES=(CM_OT_seed_anatomical_region,CM_OT_mark_fit_region,CM_OT_cancel_region_projection,CM_OT_fit_named_region,CM_OT_align_pelvis_attachment,CM_OT_review_source_region)
