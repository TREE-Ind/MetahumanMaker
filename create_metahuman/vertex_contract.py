"""Portable, ordered geometry contracts and float64 source-surface witnesses.

Hashes certify arrays and order, not anatomical accuracy. Source witnesses
certify a point on a named triangle, not that the correspondence is semantic.
"""
import hashlib
import numpy as np


def fingerprint(array):
    array=np.ascontiguousarray(array)
    digest=hashlib.sha256()
    digest.update(array.dtype.str.encode('ascii'));digest.update(str(array.shape).encode('ascii'))
    digest.update(array.tobytes(order='C'))
    return {'sha256':digest.hexdigest(),'dtype':array.dtype.str,'shape':list(array.shape)}


def closest_triangle_points(query,triangles):
    """Float64 closest points and barycentric coordinates, one triangle/query.

    Evaluate the plane interior and three bounded edges. Degenerate triangles
    use their edges, and a collapsed edge is treated as its endpoint.
    """
    query=np.asarray(query,dtype=np.float64);triangles=np.asarray(triangles,dtype=np.float64)
    if query.ndim!=2 or query.shape[1]!=3 or triangles.shape!=(len(query),3,3):
        raise ValueError('Expected N query points and N corresponding triangles')
    if not np.isfinite(query).all() or not np.isfinite(triangles).all():raise ValueError('Nonfinite surface coordinates')
    a,b,c=triangles[:,0],triangles[:,1],triangles[:,2];ab=b-a;ac=c-a;delta=query-a
    def dot(x,y):return np.einsum('ij,ij->i',x,y)
    aa,bb,cc=dot(ab,ab),dot(ab,ac),dot(ac,ac);det=aa*cc-bb*bb
    valid=det>np.maximum(aa*cc*1e-14,1e-300);safe=np.where(valid,det,1.)
    u=(cc*dot(delta,ab)-bb*dot(delta,ac))/safe
    v=(aa*dot(delta,ac)-bb*dot(delta,ab))/safe
    bary=np.column_stack((1-u-v,u,v));inside=valid&np.all(bary>=0,axis=1)
    points=np.einsum('ni,nij->nj',bary,triangles)
    cost=np.where(inside,np.sum((points-query)**2,axis=1),np.inf)
    for i,j in ((0,1),(1,2),(2,0)):
        edge=triangles[:,j]-triangles[:,i];length=dot(edge,edge)
        t=np.clip(dot(query-triangles[:,i],edge)/np.maximum(length,1e-300),0,1)
        candidate=np.zeros_like(bary);candidate[:,i]=1-t;candidate[:,j]=t
        p=np.einsum('ni,nij->nj',candidate,triangles);distance=np.sum((p-query)**2,axis=1)
        take=distance<cost;bary[take]=candidate[take];points[take]=p[take];cost[take]=distance[take]
    return points,bary,np.sqrt(cost)


def witnesses(query,source_vertices,source_triangles,triangle_ids,confidence=None):
    query=np.asarray(query,dtype=np.float64);source_vertices=np.asarray(source_vertices,dtype=np.float64)
    source_triangles=np.asarray(source_triangles,dtype=np.int64);ids=np.asarray(triangle_ids,dtype=np.int64)
    if ids.shape!=(len(query),) or np.any(ids<-1) or np.any(ids>=len(source_triangles)):
        raise ValueError('Witness IDs must name a source triangle or -1 for unsupported')
    confidence=np.ones(len(query)) if confidence is None else np.asarray(confidence,dtype=float)
    if confidence.shape!=ids.shape or not np.isfinite(confidence).all() or np.any((confidence<0)|(confidence>1)):
        raise ValueError('Witness confidence must be finite and between zero and one')
    supported=(ids>=0)&(confidence>0)
    points=query.copy();bary=np.zeros((len(query),3));distance=np.full(len(query),np.nan)
    if supported.any():
        p,b,d=closest_triangle_points(query[supported],source_vertices[source_triangles[ids[supported]]])
        points[supported]=p;bary[supported]=b;distance[supported]=d
    ids=ids.copy();ids[~supported]=-1;confidence=confidence.copy();confidence[~supported]=0
    return dict(vertex_ids=np.arange(len(query),dtype=np.int64),source_triangle_ids=ids,
        barycentric=bary,surface_points=points,residual=distance,confidence=confidence,observed=supported)


def resolve_facet_ties(query,source_vertices,source_triangles,baseline_ids,candidate_ids):
    """Choose stable source facets at numerically indistinguishable features.

    This resolves normal-field ambiguity only; it does not change distance
    witnesses or relax a source-offset bound. Retain a baseline facet when it
    is strictly closer in float64, or the two facets share a geometric vertex
    and have indistinguishable closest points and squared distances. Equal
    distances to separate surface sheets are deliberately not a feature tie.
    """
    query=np.asarray(query,dtype=float);source=np.asarray(source_vertices,dtype=float);tri=np.asarray(source_triangles)
    old=np.asarray(baseline_ids);new=np.asarray(candidate_ids)
    if query.ndim!=2 or query.shape[1]!=3 or source.ndim!=2 or source.shape[1]!=3 or not np.isfinite(query).all() or not np.isfinite(source).all():raise ValueError('Invalid source facet tie coordinates')
    if tri.dtype.kind not in 'iu' or tri.ndim!=2 or tri.shape[1]!=3 or not len(tri) or tri.min()<0 or tri.max()>=len(source) or any(ids.dtype.kind not in 'iu' or ids.shape!=(len(query),) or np.any((ids<0)|(ids>=len(tri))) for ids in (old,new)):raise ValueError('Invalid source facet tie indices')
    result=new.copy();ids=np.flatnonzero(old!=new)
    report=dict(method='SHARED_SOURCE_FEATURE_FLOAT64_TIE_RESOLUTION_V1',different_facet_queries=len(ids),shared_feature_ties=0,strictly_closer_baseline=0,source_distance_bounds_unchanged=True)
    if not len(ids):return result,report
    a=source[tri[old[ids]]];b=source[tri[new[ids]]]
    pa,_,da=closest_triangle_points(query[ids],a);pb,_,db=closest_triangle_points(query[ids],b)
    # Squared-distance roundoff is measured in source units. Its square-root
    # bounds the positional ambiguity near an edge/vertex. No world-axis or
    # character-specific length enters this policy.
    scale=2*float(np.linalg.norm(source-source.mean(0),axis=1).max(initial=0));error=128*np.finfo(float).eps*scale**2
    shared=np.any(np.all(a[:,:,None,:]==b[:,None,:,:],axis=-1),axis=(1,2))
    tie=shared&(np.abs(da**2-db**2)<=error)&(np.sum((pa-pb)**2,axis=1)<=error)
    closer=da**2<db**2-error
    result[ids[tie|closer]]=old[ids[tie|closer]]
    report.update(shared_feature_ties=int(tie.sum()),strictly_closer_baseline=int(closer.sum()),squared_distance_roundoff_bound=error)
    return result,report


def audit_witnesses(record,source_vertices,source_triangles):
    ids=np.asarray(record['source_triangle_ids']);observed=np.asarray(record['observed'],dtype=bool)
    n=len(ids)
    if not np.array_equal(record['vertex_ids'],np.arange(n,dtype=np.int64)):raise ValueError('Canonical vertex order changed')
    if not np.array_equal(observed,(ids>=0)&(record['confidence']>0)):raise ValueError('Inconsistent observation flags')
    bary=np.asarray(record['barycentric'])[observed]
    if np.any(bary<0) or not np.allclose(bary.sum(1),1.,rtol=0,atol=2e-14):raise ValueError('Invalid surface barycentrics')
    vertices=np.asarray(source_vertices,dtype=np.float64);triangles=np.asarray(source_triangles,dtype=np.int64)
    expected=np.einsum('ni,nij->nj',bary,vertices[triangles[ids[observed]]])
    actual=np.asarray(record['surface_points'])[observed]
    if not np.array_equal(expected,actual):raise ValueError('Witness surface points do not reconstruct byte for byte')
    return {'canonical_order_exact':True,'witness_reconstruction_exact':True,'vertices':n,
        'observed':int(observed.sum()),'unsupported':int((~observed).sum()),
        'source_vertices':fingerprint(vertices),'source_triangles':fingerprint(triangles),
        'vertex_ids':fingerprint(record['vertex_ids']),'source_triangle_ids':fingerprint(ids)}


def quantization_receipt(planned_world,matrix_world,observed_local):
    """Distinguish exact float32 storage from world-space roundoff."""
    planned_world=np.asarray(planned_world,dtype=np.float64);matrix=np.asarray(matrix_world,dtype=np.float64)
    inverse=np.linalg.inv(matrix);expected=(planned_world@inverse[:3,:3].T+inverse[:3,3]).astype(np.float32)
    actual=np.asarray(observed_local,dtype=np.float32)
    if actual.shape!=expected.shape:raise ValueError('Stored vertex count/order does not match plan')
    world=actual.astype(np.float64)@matrix[:3,:3].T+matrix[:3,3]
    return {'storage_bytes_exact':actual.tobytes()==expected.tobytes(),'planned_float32':fingerprint(expected),
        'observed_float32':fingerprint(actual),'world_quantization_max':float(np.linalg.norm(world-planned_world,axis=1).max(initial=0))}
