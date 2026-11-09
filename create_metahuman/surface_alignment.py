"""Source-normal tangent energy for the external constrained surface fitter."""
import numpy as np
from scipy import sparse


from .source_angle_contract import angle_contract,angular_values


def angular_model(points,triangles,free,contract,*,linearization_points=None):
    """Convex area cone with a fixed current offset and optional secant slope.

    A triangle's area vector is quadratic in positions. Its Jacobian at the
    midpoint of a proposed direction reproduces that direction's exact area
    change. The optional slope is a proposal model, not a changed constraint;
    callers must retain exact nonlinear checks at every accepted endpoint.
    """
    points=np.asarray(points);linearization=points if linearization_points is None else np.asarray(linearization_points,dtype=float)
    if linearization.shape!=points.shape or not np.isfinite(linearization).all():raise ValueError('Invalid angular model linearization points')
    ids=contract['face_ids'];faces=np.asarray(triangles)[ids];p=linearization[faces];n=len(points);count=len(ids)
    lookup=np.full(n,-1,dtype=np.int64);lookup[free]=np.arange(np.count_nonzero(free));rows=[];columns=[];values=[]
    def skew(w):
        x,y,z=w.T;zero=np.zeros(len(w))
        return np.stack((zero,-z,y,z,zero,-x,-y,x,zero),axis=1).reshape(-1,3,3)
    for corner,edge in enumerate((p[:,2]-p[:,1],p[:,0]-p[:,2],p[:,1]-p[:,0])):
        variables=lookup[faces[:,corner]];movable=variables>=0
        r=np.broadcast_to(np.arange(count)[:,None,None]*3+np.arange(3)[None,:,None],(count,3,3))
        c=np.broadcast_to(variables[:,None,None]*3+np.arange(3)[None,None,:],(count,3,3))
        v=skew(edge)/contract['area_scale'][:,None,None]
        rows.append(r[movable].ravel());columns.append(c[movable].ravel());values.append(v[movable].ravel())
    B=sparse.coo_matrix((np.concatenate(values),(np.concatenate(rows),np.concatenate(columns))),shape=(3*count,3*np.count_nonzero(free))).tocsr();B.eliminate_zeros()
    q=points[faces];offset=np.cross(q[:,1]-q[:,0],q[:,2]-q[:,0])/contract['area_scale'][:,None]
    return dict(matrix=B,offset=offset,normals=contract['normals'],minimum_cosine=contract['minimum_cosine'])


def tangent_objective(reference,triangles,normals,supported,parameter_reference,*,strength=2.):
    """Penalize edge components across coherent original-source tangent planes.

    Canonical area/edge ratios keep inherited skinny faces from suppressing
    their own alignment weight. This soft quadratic guides the proposal;
    independent source-distance, orientation and contact gates still apply.
    """
    reference=np.asarray(reference,dtype=float);metric=np.asarray(parameter_reference,dtype=float)
    triangles=np.asarray(triangles,dtype=np.int64);normals=np.asarray(normals,dtype=float);supported=np.asarray(supported,dtype=bool)
    n=len(reference)
    if reference.shape!=(n,3) or metric.shape!=reference.shape or not np.isfinite(reference).all() or not np.isfinite(metric).all():raise ValueError('Invalid alignment metric')
    if triangles.ndim!=2 or triangles.shape[1]!=3 or not len(triangles) or triangles.min()<0 or triangles.max()>=n or normals.shape!=(len(triangles),3) or supported.shape!=(len(triangles),):raise ValueError('Invalid source-normal face scope')
    if not np.isfinite(normals).all() or not np.isfinite(strength) or strength<=0:raise ValueError('Choose finite normals and positive alignment strength')
    faces=triangles[supported];directions=normals[supported];length=np.linalg.norm(directions,axis=1)
    if np.any(length<1e-12):raise ValueError('Supported alignment normals must be nonzero')
    directions=directions/length[:,None]
    p=metric[faces];area=np.linalg.norm(np.cross(p[:,1]-p[:,0],p[:,2]-p[:,0]),axis=1)*.5
    edge_square=sum(np.sum((p[:,a]-p[:,b])**2,axis=1) for a,b in ((0,1),(1,2),(2,0)))
    if np.any(area<=0) or np.any(edge_square<=0):raise ValueError('Alignment needs nondegenerate canonical facets')
    weights=strength*area/edge_square
    edges=np.concatenate([faces[:,pair] for pair in ((0,1),(1,2),(2,0))])
    direction=np.tile(directions,(3,1))*np.sqrt(np.tile(weights,3))[:,None]
    rows=np.repeat(np.arange(len(edges)),6);columns=(edges[:,:,None]*3+np.arange(3)).reshape(-1)
    values=np.stack((-direction,direction),axis=1).reshape(-1)
    B=sparse.coo_matrix((values,(rows,columns)),shape=(len(edges),3*n)).tocsr()
    matrix=(B.T@B).tocsr()
    return dict(matrix=matrix,report=dict(method='CANONICAL_AREA_SOURCE_TANGENT_QUADRATIC_V1',supported_faces=len(faces),strength=float(strength),
        before_energy=float(np.sum((B@reference.ravel())**2)),reference_metric='CANONICAL_AREA_OVER_EDGE_SQUARE'))


def redistribution_objective(reference,triangles,normals,eligible,parameter_reference,influence,*,strength=8.):
    """Redistribute observed samples along source tangent planes.

    A positive canonical-edge stencil measures tangential centroid imbalance.
    Unlike displacement fairing toward an inherited projected point map, its
    target is zero tangential imbalance. The normal component is excluded;
    original source-distance, artist, contact and nonlinear checks must still
    constrain the actual fitted geometry. This is a distribution prior, not
    evidence for missing source shape or an injective parameterization.
    """
    reference=np.asarray(reference,dtype=float);metric=np.asarray(parameter_reference,dtype=float)
    tri=np.asarray(triangles,dtype=np.int64);normals=np.asarray(normals,dtype=float)
    eligible=np.asarray(eligible,dtype=bool);influence=np.asarray(influence,dtype=float);n=len(reference)
    if reference.shape!=(n,3) or metric.shape!=reference.shape or normals.shape!=reference.shape or eligible.shape!=(n,) or influence.shape!=(n,) or not all(np.isfinite(a).all() for a in (reference,metric,normals,influence)) or np.any((influence<0)|(influence>1)) or not np.isfinite(strength) or strength<=0:raise ValueError('Invalid source redistribution arrays')
    if tri.ndim!=2 or tri.shape[1]!=3 or not len(tri) or tri.min()<0 or tri.max()>=n:raise ValueError('Invalid source redistribution triangles')
    length=np.linalg.norm(normals,axis=1)
    if np.any(length[eligible]<1e-12):raise ValueError('Supported redistribution normals must be nonzero')
    unit=normals/np.maximum(length[:,None],1e-30)
    edges=np.unique(np.sort(np.vstack((tri[:,[0,1]],tri[:,[1,2]],tri[:,[2,0]])),axis=1),axis=0)
    distance=np.linalg.norm(metric[edges[:,0]]-metric[edges[:,1]],axis=1)
    if np.any(distance<=0):raise ValueError('Source redistribution requires nonzero canonical edges')
    weight=1/distance
    adjacency=sparse.coo_matrix((np.r_[weight,weight],(np.r_[edges[:,0],edges[:,1]],np.r_[edges[:,1],edges[:,0]])),shape=(n,n)).tocsr()
    degree=np.asarray(adjacency.sum(1)).ravel();stencil=sparse.diags(1/np.maximum(degree,1e-30))@adjacency
    # Require complete supported stencils and coherent source normals. No
    # artificial tangent plane is extrapolated over a missing observation.
    support=np.asarray(stencil@eligible.astype(float)).ravel()
    average=stencil@unit
    coherence=np.einsum('ij,ij->i',unit,average)
    ids=np.flatnonzero(eligible&(support>=1-1e-12)&(coherence>=.8)&(influence>0))
    lap=(sparse.eye(n)-stencil).tocsr()[ids]
    p=metric[tri];area=np.linalg.norm(np.cross(p[:,1]-p[:,0],p[:,2]-p[:,0]),axis=1)*.5;mass=np.zeros(n)
    for corner in range(3):np.add.at(mass,tri[:,corner],area/3)
    edge_square=np.zeros(n)
    # Weighted average squared canonical edge length makes the energy
    # independent of character units, without using deformed triangle areas.
    np.add.at(edge_square,edges[:,0],weight*distance**2);np.add.at(edge_square,edges[:,1],weight*distance**2)
    edge_square/=np.maximum(degree,1e-30)
    weights=strength*mass[ids]*influence[ids]/np.maximum(edge_square[ids],1e-30)
    projector=(np.eye(3)[None,:,:]-unit[ids,:,None]*unit[ids,None,:])*np.sqrt(weights)[:,None,None]
    blocks=sparse.bsr_matrix((projector,np.arange(len(ids)),np.arange(len(ids)+1)),shape=(3*len(ids),3*len(ids)),blocksize=(3,3))
    B=blocks@sparse.kron(lap,sparse.eye(3),format='csr')
    matrix=(B.T@B).tocsr()
    return dict(matrix=matrix,report=dict(method='CANONICAL_POSITIVE_SOURCE_TANGENT_REDISTRIBUTION_V1',supported_stencils=len(ids),strength=float(strength),before_energy=float(np.sum((B@reference.ravel())**2)),source_distance_evidence_unchanged=True,normal_curvature_penalized=False))
