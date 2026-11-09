"""Oriented original-facet disk correspondence for attachment reconstruction.

Tutte initialization and libigl SLIM supply local chart optimization. The
result is a proposal; 3D bounds, endpoint contacts and native storage still
require independent validation. No global registration guarantee is made.
"""
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve
from scipy.sparse.csgraph import dijkstra,connected_components


def topology(faces,n):
    edges=np.vstack((faces[:,[0,1]],faces[:,[1,2]],faces[:,[2,0]]))
    unique,index,count=np.unique(np.sort(edges,axis=1),axis=0,return_index=True,return_counts=True)
    if np.any(count>2) or n-len(unique)+len(faces)!=1:raise ValueError('Patch is not a manifold disk')
    directed=edges[index[count==1]];out={int(a):int(b) for a,b in directed}
    if len(out)!=len(directed):raise ValueError('Patch boundary is not an oriented simple loop')
    loop=[int(directed[0,0])]
    for _ in range(len(directed)-1):
        if loop[-1] not in out:raise ValueError('Patch has inconsistent orientation')
        loop.append(out[loop[-1]])
    if len(set(loop))!=len(directed) or out[loop[-1]]!=loop[0]:raise ValueError('Patch has multiple boundary loops')
    adj=sparse.coo_matrix((np.ones(len(unique)*2),(np.r_[unique[:,0],unique[:,1]],np.r_[unique[:,1],unique[:,0]])),shape=(n,n)).tocsr()
    if connected_components(adj,directed=False,return_labels=False)!=1:raise ValueError('Patch has disconnected components')
    lap=sparse.diags(np.asarray(adj.sum(1)).ravel())-adj
    return np.asarray(loop,dtype=np.int64),lap,unique


def harmonic(lap,boundary,values):
    fixed=np.zeros(lap.shape[0],bool);fixed[boundary]=True;out=np.zeros((len(fixed),2));out[boundary]=values
    out[~fixed]=spsolve(lap[~fixed][:,~fixed].tocsc(),-(lap[~fixed][:,fixed]@out[fixed]))
    return out


def signed_areas(uv,faces):
    a=uv[faces[:,1]]-uv[faces[:,0]];b=uv[faces[:,2]]-uv[faces[:,0]]
    return a[:,0]*b[:,1]-a[:,1]*b[:,0]


def boundary_is_simple(uv,loop):
    """Reject nonadjacent crossings/touches in the optimized chart boundary."""
    points=uv[loop];n=len(points);i,j=np.triu_indices(n,1);keep=(j!=i+1)&~((i==0)&(j==n-1));i=i[keep];j=j[keep]
    a=points[i];b=points[(i+1)%n];c=points[j];d=points[(j+1)%n];scale=max(float(np.linalg.norm(np.ptp(points,axis=0))),1.)
    epsilon=scale*1e-12;area_epsilon=epsilon*scale
    overlap=np.all(np.maximum(np.minimum(a,b),np.minimum(c,d))<=np.minimum(np.maximum(a,b),np.maximum(c,d))+epsilon,axis=1)
    def cross(x,y):return x[:,0]*y[:,1]-x[:,1]*y[:,0]
    signs=[]
    for value in (cross(b-a,c-a),cross(b-a,d-a),cross(d-c,a-c),cross(d-c,b-c)):
        signs.append(np.where(abs(value)<=area_epsilon,0,np.sign(value)))
    return not bool(np.any(overlap&(signs[0]*signs[1]<=0)&(signs[2]*signs[3]<=0)))


def propose(reference,triangles,free,source,source_triangles,source_face_ids,seed,*,rings=6,maximum_rings=20,parameter_reference=None,cancelled=None,correspondence=None,prefer_source_boundary=True):
    reference=np.asarray(reference,dtype=float);source=np.asarray(source,dtype=float);triangles=np.asarray(triangles);source_triangles=np.asarray(source_triangles);source_face_ids=np.asarray(source_face_ids);free=np.asarray(free,dtype=bool)
    for vertices,faces in ((reference,triangles),(source,source_triangles)):
        if vertices.ndim!=2 or vertices.shape[1]!=3 or not len(vertices) or not np.isfinite(vertices).all():raise ValueError('Charts require finite N by 3 vertices')
        if faces.dtype.kind not in 'iu' or faces.ndim!=2 or faces.shape[1]!=3 or not len(faces) or faces.min()<0 or faces.max()>=len(vertices):raise ValueError('Charts require valid original triangle indices')
    if source_face_ids.dtype.kind not in 'iu' or source_face_ids.ndim!=1 or not len(source_face_ids) or source_face_ids.min()<0 or source_face_ids.max()>=len(source_triangles):raise ValueError('Choose original source-patch faces')
    if free.shape!=(len(reference),) or not 0<=seed<len(reference) or not free[seed]:raise ValueError('Chart seed must belong to the movable region')
    if not isinstance(rings,int) or not isinstance(maximum_rings,int) or not 2<=rings<=maximum_rings<=32:raise ValueError('Choose chart radii between two and 32 graph rings')
    if parameter_reference is not None:
        parameter_reference=np.asarray(parameter_reference,dtype=float)
        if parameter_reference.shape!=reference.shape or not np.isfinite(parameter_reference).all():raise ValueError('Intrinsic reference must match the canonical vertex order')
    attempts=[]
    # A convex replacement of a crossed closest-source boundary is useful as
    # a last resort, but is not evidence of a good physical join. First grow
    # the immutable-boundary collar until its actual source witnesses admit
    # an oriented disk. Do not prefer a small, synthetically ordered collar
    # merely because its 2D chart can be made injective.
    for registration in ((False,True) if prefer_source_boundary else (True,)):
        for size in range(rings,maximum_rings+1,2):
            if cancelled and cancelled():raise RuntimeError('Attachment reconstruction cancelled')
            try:
                transport={}
                result,receipt=_propose(reference,triangles,free,source,source_triangles,source_face_ids,seed,rings=size,parameter_reference=parameter_reference,allow_boundary_registration=registration,correspondence=transport)
                receipt['domain_attempts']=attempts+[dict(rings=size,registration_allowed=registration,valid=True)]
                receipt['boundary_search']='EXACT_SOURCE_COLLAR_BEFORE_REGISTERED_FALLBACK' if prefer_source_boundary else 'REGISTERED_BOUNDARY_COMPARISON'
                if correspondence is not None:correspondence.clear();correspondence.update(transport)
                return result,receipt
            except ValueError as exc:attempts.append(dict(rings=size,registration_allowed=registration,valid=False,reason=str(exc)))
    raise ValueError('No valid intrinsic repair disk: '+str(attempts))


def _propose(reference,triangles,free,source,source_triangles,source_face_ids,seed,*,rings,parameter_reference,allow_boundary_registration=True,correspondence=None):
    import igl
    from create_metahuman.vertex_contract import witnesses,audit_witnesses
    ref=np.asarray(reference);tri=np.asarray(triangles);free=np.asarray(free,dtype=bool)
    edges=np.unique(np.sort(np.vstack((tri[:,[0,1]],tri[:,[1,2]],tri[:,[2,0]])),axis=1),axis=0)
    selected=np.zeros(len(ref),bool);selected[seed]=True
    for _ in range(rings):
        selected[np.unique(edges[np.any(selected[edges],axis=1)])]=True;selected&=free
    target_ids=np.unique(tri[np.all(selected[tri],axis=1)]);lookup=np.full(len(ref),-1,int);lookup[target_ids]=np.arange(len(target_ids))
    target_faces=lookup[tri[np.all(selected[tri],axis=1)]]
    
    try:boundary,lap,_=topology(target_faces,len(target_ids))
    except ValueError as exc:raise ValueError('Target collar: '+str(exc)) from exc
    boundary_ids=target_ids[boundary]
    source_tree=igl.AABB();source_tree.init(source,source_triangles[source_face_ids])
    _,near,_=source_tree.squared_distance(source,source_triangles[source_face_ids],ref[np.r_[seed,boundary_ids]])
    near=source_face_ids[near];record=witnesses(ref[boundary_ids],source,source_triangles,near[1:]);audit_witnesses(record,source,source_triangles)
    vertices,first,weld=np.unique(source,axis=0,return_index=True,return_inverse=True);sf=weld[source_triangles[source_face_ids]]
    valid=np.all(np.diff(np.sort(sf,axis=1),axis=1)!=0,axis=1);sf=sf[valid];original_faces=source_face_ids[valid]
    se=np.unique(np.sort(np.vstack((sf[:,[0,1]],sf[:,[1,2]],sf[:,[2,0]])),axis=1),axis=0)
    cost=np.linalg.norm(vertices[se[:,0]]-vertices[se[:,1]],axis=1)
    graph=sparse.coo_matrix((np.r_[cost,cost],(np.r_[se[:,0],se[:,1]],np.r_[se[:,1],se[:,0]])),shape=(len(vertices),len(vertices))).tocsr()
    center=weld[source_triangles[near[0],0]];distance=dijkstra(graph,directed=False,indices=center)
    bound_distance=distance[weld[source_triangles[near[1:]]]]
    radius=float(np.max(bound_distance))*1.15
    if not np.isfinite(radius) or radius<=0:raise ValueError('Boundary has no connected original-source neighborhood')
    crop=np.all(distance[sf]<=radius,axis=1);source_ids=np.unique(sf[crop]);slookup=np.full(len(vertices),-1,int);slookup[source_ids]=np.arange(len(source_ids));faces=slookup[sf[crop]];global_faces=original_faces[crop]
    
    local_source=vertices[source_ids]
    cut_receipt=None;domain_receipt=None
    try:loop,slap,_=topology(faces,len(local_source))
    except ValueError as exc:
        from .chart_cut import open_boundaries
        from .chart_domain import prepare as prepare_domain
        # Cuts affect only this chart. All source triangles, positions and
        # barycentric corner identities remain authoritative and unchanged.
        _,witness_local,_=source_tree.squared_distance(source,source_triangles[source_face_ids],ref[target_ids])
        try:
            local_source,faces,global_faces,domain_receipt=prepare_domain(local_source,faces,global_faces,near)
            try:loop,slap,_=topology(faces,len(local_source))
            except ValueError:
                protected=np.isin(global_faces,np.r_[near,source_face_ids[witness_local]])
                local_source,faces,cut_receipt=open_boundaries(local_source,faces,protected)
                loop,slap,_=topology(faces,len(local_source))
        except ValueError as cut_error:raise ValueError('Source disk: '+str(exc)+'; '+str(cut_error)) from cut_error
    length=np.linalg.norm(local_source[loop]-np.roll(local_source[loop],-1,axis=0),axis=1);angle=np.r_[0,np.cumsum(length[:-1])]/length.sum()*2*np.pi
    uv=harmonic(slap,loop,np.column_stack((np.cos(angle),np.sin(angle))))
    sa=signed_areas(uv,faces)
    if np.any(sa<=1e-14):raise ValueError('Original-source chart is folded or degenerate')
    face_map={int(face):i for i,face in enumerate(global_faces)}
    if any(int(face) not in face_map for face in near[1:]):raise ValueError('Source disk does not cover the target boundary witnesses')
    boundary_uv=np.einsum('ni,nij->nj',record['barycentric'],uv[faces[[face_map[int(face)] for face in near[1:]]]])
    target_uv=harmonic(lap,boundary,boundary_uv);area=signed_areas(target_uv,target_faces);sign=1 if np.median(area)>0 else -1
    boundary_mode='EXACT_CLOSEST_SOURCE_UV';boundary_error=0.
    if np.any(area<=1e-14):
        if not allow_boundary_registration and not boundary_is_simple(boundary_uv,np.arange(len(boundary_uv))):raise ValueError('Closest-source boundary crosses itself; expand the physical repair boundary')
        # A folded input can have a self-crossing closest-source boundary.
        # Register an ordered convex loop in the original-source chart.
        # This changes interior correspondence only: physical boundary
        # vertices remain fixed and the caller must audit the 3D join.
        center_uv=boundary_uv.mean(0);center_complex=complex(*center_uv)
        observation=(boundary_uv[:,0]+1j*boundary_uv[:,1])-center_complex
        lengths=np.linalg.norm(ref[boundary_ids]-np.roll(ref[boundary_ids],-1,axis=0),axis=1)
        theta=np.r_[0,np.cumsum(lengths[:-1])]/lengths.sum()*2*np.pi
        candidates=[]
        for direction in (1,):
            unit=np.exp(1j*theta*direction);phase=np.angle(np.sum(observation*np.conj(unit)));unit*=np.exp(1j*phase)
            radius_fit=min(float(np.maximum(np.real(observation*np.conj(unit)).mean(),1e-4)),(1-np.linalg.norm(center_uv))*.95)
            curve=center_complex+radius_fit*unit;error=float(np.mean(abs(curve-(boundary_uv[:,0]+1j*boundary_uv[:,1]))**2))
            candidates.append((error,np.column_stack((curve.real,curve.imag))))
        error,ordered=min(candidates,key=lambda item:item[0]);boundary_error=float(np.sqrt(error));boundary_mode='SOURCE_CHART_CONVEX_LOOP_REGISTRATION'
        target_uv=harmonic(lap,boundary,ordered);area=signed_areas(target_uv,target_faces);sign=1 if np.median(area)>0 else -1
    if np.any(area<=1e-14):raise ValueError('Boundary-constrained target chart is folded or degenerate')
    _,near_local,_=source_tree.squared_distance(source,source_triangles[source_face_ids],ref[target_ids])
    all_faces=source_face_ids[near_local];all_record=witnesses(ref[target_ids],source,source_triangles,all_faces)
    usable=np.array([int(face) in face_map for face in all_faces]);usable&=all_record['residual']<=max(np.linalg.norm(np.ptp(ref,axis=0))*.002,float(np.median(all_record['residual'])))
    # Being close to the source is insufficient evidence of correspondence
    # inside a fold: both sides of a pleat can be almost on the same facet.
    # Do not anchor that contradictory ordering into the injective chart.
    # This only removes proposal anchors; source envelopes and final gates
    # still constrain every moved vertex independently.
    observed=source[source_triangles[all_faces]]
    field=np.cross(observed[:,1]-observed[:,0],observed[:,2]-observed[:,0]);field/=np.maximum(np.linalg.norm(field,axis=1)[:,None],1e-30)
    face_field=field[target_faces].sum(1)
    current=ref[target_ids][target_faces];cross=np.cross(current[:,1]-current[:,0],current[:,2]-current[:,0])
    cosine=np.einsum('ij,ij->i',cross,face_field)/np.maximum(np.linalg.norm(cross,axis=1)*np.linalg.norm(face_field,axis=1),1e-30)
    contradictory=(np.linalg.norm(face_field,axis=1)>=2.4)&(cosine<.25)
    unsafe=np.zeros(len(target_ids),bool);unsafe[np.unique(target_faces[contradictory])]=True
    local_edges=np.unique(np.sort(np.vstack((target_faces[:,[0,1]],target_faces[:,[1,2]],target_faces[:,[2,0]])),axis=1),axis=0)
    unsafe[np.unique(local_edges[np.any(unsafe[local_edges],axis=1)])]=True
    rejected_anchors=int((usable&unsafe).sum());usable&=~unsafe
    anchors=np.flatnonzero(usable).astype(np.int32)
    if len(anchors)<3:raise ValueError('Source chart lacks reliable original-source anchors')
    anchor_uv=np.einsum('ni,nij->nj',all_record['barycentric'][anchors],uv[faces[[face_map[int(all_faces[i])] for i in anchors]]])
    rest=ref[target_ids] if parameter_reference is None else np.asarray(parameter_reference)[target_ids]
    rest_area=np.linalg.norm(np.cross(rest[target_faces[:,1]]-rest[target_faces[:,0]],rest[target_faces[:,2]]-rest[target_faces[:,0]]),axis=1).sum()
    rest=(rest-rest.mean(0))*np.sqrt(area.sum()/rest_area)
    boundary_receipt=None
    if not allow_boundary_registration:
        from .chart_boundary import solve as solve_boundary
        target_uv,boundary_receipt=solve_boundary(rest,target_faces,target_uv,boundary,boundary_uv)
        boundary_mode='EXACT_SOURCE_BOUNDARY_VERIFIED_INTERIOR';boundary_error=0.
    else:
        slim=igl.slim_precompute(np.asfortranarray(rest),np.asfortranarray(target_faces,dtype=np.int32),np.asfortranarray(target_uv),igl.MappingEnergyType.SYMMETRIC_DIRICHLET,anchors,np.asfortranarray(anchor_uv),10000.)
        target_uv=igl.slim_solve(slim,30)
    area=signed_areas(target_uv,target_faces)
    if not np.isfinite(target_uv).all() or np.any(area<=1e-14):raise ValueError('Source-anchored injective optimization failed its orientation gate')
    if not boundary_is_simple(target_uv,boundary):raise ValueError('Optimized target chart has a self-crossing boundary')
    uv3=np.column_stack((uv,np.zeros(len(uv))));target3=np.column_stack((target_uv,np.zeros(len(target_uv))));tree=igl.AABB();tree.init(uv3,faces)
    sq,which,_=tree.squared_distance(uv3,faces,target3)
    if np.max(sq)>1e-16:raise ValueError('Target chart leaves the original-source disk')
    witness=witnesses(target3,uv3,faces,which);bary=witness['barycentric'];mapped=np.einsum('ni,nij->nj',bary,source[source_triangles[global_faces[which]]])
    output=ref.copy();interior=np.ones(len(target_ids),bool);interior[boundary]=False;output[target_ids[interior]]=mapped[interior]
    if correspondence is not None:
        correspondence.update(source_uv=uv.copy(),source_chart_triangles=faces.copy(),source_face_ids=global_faces.copy(),
            target_uv=target_uv.copy(),target_chart_triangles=target_faces.copy(),target_vertex_ids=target_ids.copy(),
            target_face_ids=np.flatnonzero(np.all(selected[tri],axis=1)))
    return output,dict(method='ORIGINAL_FACET_INTRINSIC_DISK_PROPOSAL_V1',seed=int(seed),rings=rings,
        target_vertices=len(target_ids),target_interior_vertices=int(interior.sum()),source_vertices=len(local_source),source_faces=len(faces),source_chart_cut=cut_receipt,source_chart_domain=domain_receipt,
        source_geodesic_radius_m=radius,minimum_target_uv_area=float(np.min(area*sign)),minimum_source_uv_area=float(sa.min()),
        boundary_mode=boundary_mode,boundary_uv_registration_rms=boundary_error,boundary_continuation=boundary_receipt,
        source_anchor_vertices=len(anchors),source_anchor_uv_rms=float(np.sqrt(np.mean(np.sum((target_uv[anchors]-anchor_uv)**2,axis=1)))),
        source_anchor_policy='BOUNDARY_FIXED_INTERIOR_ANCHORS_DIAGNOSTIC_ONLY' if boundary_receipt else 'COHERENT_ALIGNMENT_WITH_ONE_RING_FOLD_EXCLUSION',contradictory_source_faces=int(contradictory.sum()),excluded_fold_anchors=rejected_anchors,
        optimized_boundary_simple=True,
        source_face_ids=global_faces[which],source_barycentric=bary,target_vertex_ids=target_ids,interior=interior,
        accepted=False,review_required=True)
