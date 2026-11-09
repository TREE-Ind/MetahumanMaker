"""Auditable anatomy-scoped nonrigid registration in an external SciPy worker.

The locally affine stiffness baseline follows Amberg, Romdhani and Vetter
(CVPR 2007). Extensions here are explicit triangle witnesses, area-balanced
reverse coverage, immutable region scopes, exact positional pin elimination,
and measured acceptance. This module makes no state-of-the-art or global
injectivity claim. Collision review is required before committing its output.
"""
import time
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import lsmr
from scipy.spatial import cKDTree
from .surface_oracle import SurfaceOracle
from .vertex_contract import audit_witnesses


def vertex_normals(vertices,triangles):
    p=vertices[triangles];normal=np.cross(p[:,1]-p[:,0],p[:,2]-p[:,0]);result=np.zeros_like(vertices)
    for col in range(3):np.add.at(result,triangles[:,col],normal)
    return result/np.maximum(np.linalg.norm(result,axis=1)[:,None],1e-300)


def barycentric_rows(triangles,face_ids,barycentric,n):
    return sparse.coo_matrix((barycentric.ravel(),(np.repeat(np.arange(len(face_ids)),3),triangles[face_ids].ravel())),
        shape=(len(face_ids),n)).tocsr()


def local_area_step(current,proposed,triangles,pins,iterations=24):
    """Reduce displacement only around failing faces; no collision guarantee."""
    old=current[triangles];old_normal=np.cross(old[:,1]-old[:,0],old[:,2]-old[:,0])
    area2=np.einsum('ij,ij->i',old_normal,old_normal)
    weight=np.ones(len(current));weight[pins]=0
    direction=proposed-current
    for _ in range(iterations):
        candidate=current+direction*weight[:,None]
        p=candidate[triangles];normal=np.cross(p[:,1]-p[:,0],p[:,2]-p[:,0])
        bad=(area2>=1e-36)&(np.einsum('ij,ij->i',normal,old_normal)<=area2*.03)
        if not bad.any():return candidate,weight,True
        weight[np.unique(triangles[bad])]*=.5
    # A partially limited field can still be made valid by the outer global
    # line search. Never report a failed limiter as a successful zero step.
    return current+direction*weight[:,None],weight,False


def fit_patch(initial,triangles,source_vertices,source_triangles,movable,*,observed=None,contact_guard=False,**options):
    """Solve one connected registration patch with an exact fixed boundary.

    Extract complete incident triangles and their fixed outside vertices,
    solve the local system, then scatter without changing the full vertex
    order. Supply physical welded geometry when UV seam copies are present.
    This wrapper does not provide collision acceptance or infer anatomy.
    """
    initial=np.asarray(initial,dtype=float);triangles=np.asarray(triangles,dtype=np.int64)
    movable=np.asarray(movable,dtype=bool)
    observed=movable.copy() if observed is None else np.asarray(observed,dtype=bool)
    if initial.ndim!=2 or initial.shape[1]!=3 or movable.shape!=(len(initial),) or observed.shape!=movable.shape or not movable.any():raise ValueError('Choose a matching nonempty registration patch')
    if triangles.ndim!=2 or triangles.shape[1]!=3 or not len(triangles) or triangles.min()<0 or triangles.max()>=len(initial):raise ValueError('Invalid complete target triangles')
    selected=triangles[np.any(movable[triangles],axis=1)]
    ids,local_tri=np.unique(selected,return_inverse=True);local_tri=local_tri.reshape(-1,3)
    local=initial[ids];pins=np.flatnonzero(~movable[ids]);outer=observed[ids]&movable[ids]
    if 'pin_ids' in options or 'pin_targets' in options or 'outer' in options:raise ValueError('Patch boundary and observed mask supply registration constraints')
    if options.get('axis_constraints'):
        lookup=np.full(len(initial),-1,int);lookup[ids]=np.arange(len(ids));mapped=[]
        for constraint in options['axis_constraints']:
            indices=np.asarray(constraint['vertex_ids'],dtype=np.int64)
            if indices.ndim!=1 or np.any(indices<0) or np.any(indices>=len(initial)) or np.any(lookup[indices]<0):raise ValueError('Axis constraint lies outside the registration patch')
            mapped.append(dict(constraint,vertex_ids=lookup[indices]))
        options['axis_constraints']=mapped
    guard=None
    if contact_guard:
        if 'candidate_guard' in options:raise ValueError('Choose the patch contact guard or a caller guard')
        from .surface_contacts import ContactStepGuard
        guard=ContactStepGuard(initial,triangles,movable=movable)
        def full_contact_guard(current,proposed):
            complete_current=initial.copy();complete_current[ids]=current
            complete_proposed=initial.copy();complete_proposed[ids]=proposed
            return guard(complete_current,complete_proposed)[ids]
        full_contact_guard.description='PORTABLE_FULL_TARGET_ENDPOINT_CONTACT_GUARD'
        options['candidate_guard']=full_contact_guard
    solved,witness,report=fit(local,local_tri,source_vertices,source_triangles,
        pin_ids=pins,outer=outer,**options)
    result=initial.copy();result[ids]=solved
    if not np.array_equal(result[~movable],initial[~movable]):raise RuntimeError('Patch fitting changed its fixed boundary')
    report.update(patch_vertices=len(ids),movable_vertices=int(movable.sum()),
        fixed_boundary_vertices=len(pins),outside_unchanged=True,
        local_to_full_vertex_ids=ids.tolist())
    if guard is not None:
        report['contact_guard']=dict(calls=guard.calls,limited_calls=guard.limited_calls,
            inherited_crossing_pairs=len(guard.allowed),collision_tolerance=guard.tolerance)
    return result,report


def fit(initial,triangles,source_vertices,source_triangles,*,source_face_ids=None,outer=None,
        pin_ids=None,pin_targets=None,landmark_matrix=None,landmark_targets=None,
        stiffness=(20.,5.,1.,.2,.05),reverse_samples=768,reverse_strength=.3,
        landmark_weight=30.,max_distance=.15,max_linear_iterations=240,graph_nodes=None,cancelled=None,progress=None,candidate_guard=None,axis_constraints=None):
    initial=np.asarray(initial,dtype=np.float64);triangles=np.asarray(triangles,dtype=np.int64)
    source_vertices=np.asarray(source_vertices,dtype=np.float64);source_triangles=np.asarray(source_triangles,dtype=np.int64)
    n=len(initial)
    if initial.shape!=(n,3) or not np.isfinite(initial).all() or triangles.ndim!=2 or triangles.shape[1]!=3 or triangles.min()<0 or triangles.max()>=n:
        raise ValueError('Invalid canonical registration geometry')
    outer=np.ones(n,bool) if outer is None else np.asarray(outer,dtype=bool)
    if outer.shape!=(n,) or not outer.any():raise ValueError('Mark observed exterior template vertices')
    pins=np.asarray([] if pin_ids is None else pin_ids,dtype=np.int64)
    targets=initial[pins].copy() if pin_targets is None else np.asarray(pin_targets,dtype=float)
    if pins.ndim!=1 or targets.shape!=(len(pins),3) or len(np.unique(pins))!=len(pins) or np.any(pins<0) or np.any(pins>=n):
        raise ValueError('Invalid positional pin constraints')
    if not np.isfinite(targets).all() or not np.array_equal(initial[pins],targets):
        raise ValueError('Initialize pinned vertices at their exact targets before fitting')
    if reverse_samples<1 or max_distance<=0 or not np.isfinite(max_distance):raise ValueError('Choose positive sampling and finite distance bounds')
    if not len(stiffness) or np.any(np.asarray(stiffness)<=0):raise ValueError('Choose positive stiffness stages')
    oracle=SurfaceOracle(source_vertices,source_triangles,source_face_ids)
    selected=oracle.face_ids
    source_faces=source_vertices[source_triangles[selected]]
    source_area=np.linalg.norm(np.cross(source_faces[:,1]-source_faces[:,0],source_faces[:,2]-source_faces[:,0]),axis=1)*.5
    if source_area.sum()<=0:raise ValueError('Source region has no surface area')
    quantiles=(np.arange(reverse_samples)+.5)/reverse_samples*source_area.sum()
    sample_ids=selected[np.searchsorted(np.cumsum(source_area),quantiles)]
    sample_points=source_vertices[source_triangles[sample_ids]].mean(1)
    p=initial[triangles];face_area=np.linalg.norm(np.cross(p[:,1]-p[:,0],p[:,2]-p[:,0]),axis=1)*.5
    area=np.zeros(n)
    for col in range(3):np.add.at(area,triangles[:,col],face_area/3)
    area/=max(float(area[outer].mean()),1e-300)
    target_faces=np.flatnonzero(np.all(outer[triangles],axis=1))
    if not len(target_faces):raise ValueError('Observed template region needs complete exterior faces')
    center=initial.mean(0);span=float(np.linalg.norm(np.ptp(initial,axis=0)))
    if span<1e-12:raise ValueError('Canonical mesh has no usable extent')
    rest=(initial-center)/span;h=np.column_stack((rest,np.ones(n)))
    node_ids=np.arange(n)
    if graph_nodes is not None and graph_nodes<n:
        if graph_nodes<4:raise ValueError('A deformation graph needs at least four nodes')
        # Partition of unity reproduces an affine field exactly. All hard pins
        # are graph nodes; their interpolation rows have exactly one nonzero.
        cells=np.floor((rest-rest.min(0))*float(graph_nodes)**(1/3)).astype(np.int64)
        _,representatives=np.unique(cells,axis=0,return_index=True)
        node_ids=np.unique(np.concatenate((representatives,pins)))
        distance,neighbors=cKDTree(rest[node_ids]).query(rest,k=min(8,len(node_ids)))
        # A nearest-k inverse-distance blend alone jumps when the kth node
        # changes. Vanishing support weight makes that transition continuous.
        support=np.maximum(distance[:,-1:],1e-12)
        weights=np.maximum(0,1-distance/support)**2/np.maximum(distance,1e-12)**2
        weights/=np.maximum(weights.sum(1)[:,None],1e-300)
        exact=distance[:,0]<1e-12;weights[exact]=0;weights[exact,0]=1
        rows=np.broadcast_to(np.arange(n)[:,None,None],(n,neighbors.shape[1],4)).ravel()
        cols=(neighbors[:,:,None]*4+np.arange(4)).ravel()
        D=sparse.coo_matrix(((weights[:,:,None]*h[:,None,:]).ravel(),(rows,cols)),shape=(n,len(node_ids)*4)).tocsr()
    else:
        neighbors=np.arange(n)[:,None]
        D=sparse.coo_matrix((h.ravel(),(np.repeat(np.arange(n),4),np.arange(n*4))),shape=(n,n*4)).tocsr()
    nodes=len(node_ids)
    edges=np.unique(np.sort(np.vstack((triangles[:,[0,1]],triangles[:,[1,2]],triangles[:,[2,0]])),axis=1),axis=0)
    edges=np.unique(np.sort(neighbors[:,0][edges],axis=1),axis=0)
    edges=edges[edges[:,0]!=edges[:,1]]
    incidence=sparse.coo_matrix((np.tile([1.,-1.],len(edges)),(np.repeat(np.arange(len(edges)),2),edges.ravel())),shape=(len(edges),nodes)).tocsr()
    S=sparse.kron(incidence,sparse.diags([1.,1.,1.,.5]),format='csr')
    # Eliminate pinned translations algebraically. Each pinned transform still
    # has a free 3x3 linear part, while D_i X_i equals its requested point.
    pin_nodes=np.searchsorted(node_ids,pins)
    removed=pin_nodes*4+3;free=np.setdiff1d(np.arange(nodes*4),removed);lookup=np.full(nodes*4,-1,dtype=np.int64);lookup[free]=np.arange(len(free))
    rows=list(free);cols=list(np.arange(len(free)));values=[1.]*len(free)
    for vertex,node in zip(pins,pin_nodes):
        rows.extend([node*4+3]*3);cols.extend(lookup[node*4+np.arange(3)]);values.extend(-rest[vertex])
    P=sparse.coo_matrix((values,(rows,cols)),shape=(nodes*4,len(free))).tocsr()
    C=np.zeros((nodes*4,3));C[removed]=(targets-center)/span
    identity=np.tile(np.vstack((np.eye(3),np.zeros((1,3)))),(nodes,1))
    current=initial.copy();field=identity.copy();history=[];started=time.monotonic()
    landmark_matrix=None if landmark_matrix is None else sparse.csr_matrix(landmark_matrix)
    if landmark_matrix is not None:
        landmark_targets=np.asarray(landmark_targets,dtype=float)
        if landmark_matrix.shape[1]!=n or landmark_targets.shape!=(landmark_matrix.shape[0],3):raise ValueError('Invalid barycentric landmark constraints')
    constraints=[]
    for constraint in axis_constraints or ():
        axis=constraint['axis'];indices=np.asarray(constraint['vertex_ids'],dtype=np.int64);values=np.asarray(constraint['targets'],dtype=float);weight=float(constraint.get('weight',30.))
        if axis not in (0,1,2) or indices.ndim!=1 or not len(indices) or values.shape!=indices.shape or np.any(indices<0) or np.any(indices>=n) or not np.isfinite(values).all() or not np.isfinite(weight) or weight<=0:raise ValueError('Invalid axis constraints')
        rows=sparse.coo_matrix((np.ones(len(indices)),(np.arange(len(indices)),indices)),shape=(len(indices),n)).tocsr()
        constraints.append((axis,indices,values,weight,rows))
    def observe(vertices):
        record=oracle.query(vertices,max_distance=max_distance,query_normals=vertex_normals(vertices,triangles),normal_limit=.1,cancelled=cancelled)
        record['confidence'][~outer]=0;record['observed'][~outer]=False;record['source_triangle_ids'][~outer]=-1;record['barycentric'][~outer]=0;record['residual'][~outer]=np.nan
        return record
    def measure(vertices,record):
        forward=np.where(record['observed'],np.nan_to_num(record['residual'],nan=max_distance),max_distance)
        # Excluding hidden vertices prevents an unsupported oral cavity from
        # being scored as if it were an observable part of a closed source scan.
        rms=float(np.sqrt(np.average(forward[outer]**2,weights=area[outer])))
        reverse=SurfaceOracle(vertices,triangles,target_faces).query(sample_points,max_distance=max_distance,cancelled=cancelled)
        distances=np.where(reverse['observed'],np.nan_to_num(reverse['residual'],nan=max_distance),max_distance)
        reverse_rms=float(np.sqrt(np.mean(distances**2)))
        landmark_rms=0. if landmark_matrix is None else float(np.sqrt(np.mean(np.sum((landmark_matrix@vertices-landmark_targets)**2,axis=1))))
        axis_rms=0. if not constraints else float(np.sqrt(np.mean(np.concatenate([(vertices[indices,axis]-values)**2 for axis,indices,values,weight,rows in constraints]))))
        return {'forward_rms':rms,'reverse_rms':reverse_rms,'landmark_rms':landmark_rms,'axis_constraint_rms':axis_rms,'score':rms+reverse_strength*reverse_rms+.5*landmark_rms+.5*axis_rms,
            'observed_fraction':float(record['observed'][outer].mean())},reverse
    record=observe(current);initial_metrics,reverse=measure(current,record);metrics=initial_metrics
    for stage,alpha in enumerate(stiffness):
        if cancelled and cancelled():raise RuntimeError('Global registration cancelled; previous character retained')
        robust=1/(1+(np.nan_to_num(record['residual'],nan=max_distance)/(max_distance*.3))**2)
        weight=np.sqrt(record['confidence']*area*robust)
        retention=np.sqrt(np.where(outer,.005,.5))
        blocks=[S*np.sqrt(alpha),D.multiply(weight[:,None]),D.multiply(retention[:,None]),sparse.eye(nodes*4,format='csr')*.003]
        rhs=[np.zeros((S.shape[0],3)),(record['surface_points']-center)/span*weight[:,None],
             (current-center)/span*retention[:,None],identity*.003]
        valid=reverse['observed']
        if valid.any():
            R=barycentric_rows(triangles,reverse['source_triangle_ids'][valid],reverse['barycentric'][valid],n)
            reverse_weight=np.sqrt(reverse_strength*float(area[outer].sum())/max(int(valid.sum()),1))
            blocks.append((R@D)*reverse_weight);rhs.append((sample_points[valid]-center)/span*reverse_weight)
        if landmark_matrix is not None:
            blocks.append((landmark_matrix@D)*landmark_weight);rhs.append((landmark_targets-center)/span*landmark_weight)
        A=sparse.vstack(blocks,format='csr');B=np.vstack(rhs)
        matrix=(A@P).tocsr();B-=A@C
        column_scale=1/np.maximum(np.sqrt(np.asarray(matrix.power(2).sum(0)).ravel()),1e-12)
        matrix=matrix@sparse.diags(column_scale,format='csr');unknown=np.empty((len(free),3));linear=[]
        for axis in range(3):
            if cancelled and cancelled():raise RuntimeError('Global registration cancelled; previous character retained')
            axis_matrix=matrix;axis_rhs=B[:,axis]
            for constrained_axis,indices,values,weight,rows in constraints:
                if constrained_axis!=axis:continue
                operator=(rows@D)*weight
                extra=(operator@P)@sparse.diags(column_scale,format='csr')
                target=(values-center[axis])/span*weight-operator@C[:,axis]
                axis_matrix=sparse.vstack((axis_matrix,extra),format='csr');axis_rhs=np.r_[axis_rhs,target]
            # Rebalance after adding strong support-plane constraints.
            axis_scale=1/np.maximum(np.sqrt(np.asarray(axis_matrix.power(2).sum(0)).ravel()),1e-12)
            axis_matrix=axis_matrix@sparse.diags(axis_scale,format='csr')
            total_scale=column_scale*axis_scale
            solved=lsmr(axis_matrix,axis_rhs,atol=1e-6,btol=1e-6,maxiter=max_linear_iterations,x0=field[free,axis]/total_scale)
            unknown[:,axis]=solved[0]*total_scale
            linear.append({'stop':int(solved[1]),'iterations':int(solved[2]),'residual_norm':float(solved[3])})
        proposed_field=P@unknown+C;proposed=(D@proposed_field)*span+center;proposed[pins]=targets
        proposed,local_weight,local_valid=local_area_step(current,proposed,triangles,pins)
        old=current[triangles];old_normal=np.cross(old[:,1]-old[:,0],old[:,2]-old[:,0]);old_area=np.linalg.norm(old_normal,axis=1)
        accepted=False;fraction=1.
        for _ in range(9):
            candidate=current+(proposed-current)*fraction;candidate[pins]=targets
            if candidate_guard is not None:
                candidate=np.asarray(candidate_guard(current,candidate),dtype=float)
                if candidate.shape!=current.shape or not np.isfinite(candidate).all() or not np.array_equal(candidate[pins],targets):
                    raise ValueError('Candidate guard changed exact pins or returned invalid geometry')
            # A guard can freeze the entire field. Re-querying both surfaces
            # at identical coordinates is expensive and cannot improve fit.
            # Smaller steps still get a chance to find a valid trajectory.
            if np.array_equal(candidate,current):
                fraction*=.5
                continue
            p=candidate[triangles];normal=np.cross(p[:,1]-p[:,0],p[:,2]-p[:,0]);dot=np.einsum('ij,ij->i',normal,old_normal)
            valid_area=(old_area<1e-18)|(dot>old_area**2*.03)
            if valid_area.all():
                candidate_record=observe(candidate);candidate_metrics,candidate_reverse=measure(candidate,candidate_record)
                if candidate_metrics['score']<metrics['score']-span*1e-8:
                    current=candidate;field=field+(proposed_field-field)*fraction
                    record=candidate_record;metrics=candidate_metrics;reverse=candidate_reverse;accepted=True;break
            fraction*=.5
        entry={'stage':stage,'stiffness':float(alpha),'accepted':accepted,'step_fraction':fraction if accepted else 0.,
            'local_area_valid':local_valid,'locally_limited_fraction':float(np.mean(local_weight<.999)),
            'metrics':metrics,'linear_solves':linear,'linear_converged':all(r['stop'] in (0,1,2,4,5) for r in linear)}
        history.append(entry)
        if progress:progress(entry)
    audit=audit_witnesses(record,source_vertices,source_triangles)
    report={'solver':'AUDITABLE_LOCALLY_AFFINE_REGISTRATION_PROTOTYPE_V1','baseline_attribution':'Amberg et al. CVPR 2007',
        'before':initial_metrics,'after':metrics,'history':history,'witness_audit':audit,'deformation_graph_nodes':nodes,
        'pin_max_error':float(np.linalg.norm(current[pins]-targets,axis=1).max(initial=0)),
        'elapsed_seconds':time.monotonic()-started,'collision_review_required':True,
        'geometry_guard':'LOCAL_TRIANGLE_AREA_AND_ORIENTATION_ONLY' if candidate_guard is None else getattr(candidate_guard,'description','CALLER_SUPPLIED_CANDIDATE_GUARD'),
        'production_ready':False}
    return current,record,report
