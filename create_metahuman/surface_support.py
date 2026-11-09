"""Local observation reach for refinement of a registered physical surface."""
import numpy as np


def footprint_support(oracle,points,residual,scale,*,cancelled=None):
    """Require surrounding source surface even for zero-distance witnesses.

    Distance alone cannot distinguish skin from a cut rim. Half a local
    physical edge is the minimum sampling footprint; remote observations
    additionally need half their projection distance. UV splits are welded
    by the oracle, so this excludes geometric holes rather than UV seams.
    """
    points=np.asarray(points,dtype=float);residual=np.asarray(residual,dtype=float);scale=np.asarray(scale,dtype=float)
    if points.shape!=(len(residual),3) or scale.shape!=residual.shape or np.any(scale<=0) or not np.isfinite(scale).all() or not np.isfinite(residual).all() or np.any(residual<0):raise ValueError('Invalid source observation footprint')
    radius=np.maximum(scale,residual)*.5
    if not len(radius):return np.zeros(0,bool)
    clearance=oracle.boundary_clearance(points,cutoff=float(radius.max()),cancelled=cancelled)
    return clearance>=radius


def reciprocal_support(reference,triangles,vertex_ids,surface_points,*,rings=2):
    """Reject observations whose inverse belongs to a different target patch.

    Facing rays alone can jump across a missing source region or to a nearby
    limb. The inverse nearest triangle must lie within a bounded number of
    physical graph edges of the vertex that requested the observation.
    This is correspondence evidence, not a source-hole filling operation.
    """
    import igl
    from scipy import sparse
    reference=np.asarray(reference,dtype=float);triangles=np.asarray(triangles,dtype=np.int64)
    ids=np.asarray(vertex_ids,dtype=np.int64);points=np.asarray(surface_points,dtype=float)
    if reference.ndim!=2 or reference.shape[1]!=3 or points.shape!=(len(ids),3) or ids.ndim!=1 or not np.isfinite(reference).all() or not np.isfinite(points).all():raise ValueError('Invalid reciprocal observation arrays')
    if triangles.ndim!=2 or triangles.shape[1]!=3 or not len(triangles) or triangles.min()<0 or triangles.max()>=len(reference) or ids.min(initial=0)<0 or ids.max(initial=0)>=len(reference):raise ValueError('Invalid reciprocal observation indices')
    if not isinstance(rings,int) or not 0<=rings<=8:raise ValueError('Choose zero to eight inverse support rings')
    if not len(ids):return np.zeros(0,bool),dict(method='PHYSICAL_GRAPH_RECIPROCAL_SUPPORT_V1',observations=0,accepted=0,rejected=0,rings=rings)
    tree=igl.AABB();tree.init(reference,triangles)
    try:_,nearest,_=tree.squared_distance(reference,triangles,points)
    finally:del tree
    edges=np.unique(np.sort(np.vstack((triangles[:,[0,1]],triangles[:,[1,2]],triangles[:,[2,0]])),axis=1),axis=0)
    adjacency=sparse.coo_matrix((np.ones(2*len(edges)),(np.r_[edges[:,0],edges[:,1]],np.r_[edges[:,1],edges[:,0]])),shape=(len(reference),len(reference))).tocsr()
    accepted=np.zeros(len(ids),bool)
    for row,vertex in enumerate(ids):
        visited={int(vertex)};front=visited.copy();destination=set(map(int,triangles[nearest[row]]))
        for step in range(rings+1):
            if visited&destination:accepted[row]=True;break
            if step==rings:break
            neighbors=set()
            for current in front:neighbors.update(map(int,adjacency.indices[adjacency.indptr[current]:adjacency.indptr[current+1]]))
            front=neighbors-visited;visited|=front
            if not front:break
    return accepted,dict(method='PHYSICAL_GRAPH_RECIPROCAL_SUPPORT_V1',observations=len(ids),accepted=int(accepted.sum()),rejected=int((~accepted).sum()),rings=rings)


def observation_scale(points,triangles):
    """Median incident edge length, independent of character orientation/units."""
    points=np.asarray(points,dtype=float);triangles=np.asarray(triangles,dtype=np.int64)
    edges=np.unique(np.sort(np.vstack((triangles[:,[0,1]],triangles[:,[1,2]],triangles[:,[2,0]])),axis=1),axis=0)
    lengths=np.linalg.norm(points[edges[:,0]]-points[edges[:,1]],axis=1);valid=(edges[:,0]!=edges[:,1])&(lengths>0)
    edges=edges[valid];lengths=lengths[valid];vertices=np.r_[edges[:,0],edges[:,1]];order=np.argsort(vertices)
    vertices=vertices[order];lengths=np.r_[lengths,lengths][order];ids,starts=np.unique(vertices,return_index=True)
    scale=np.zeros(len(points))
    for vertex,values in zip(ids,np.split(lengths,starts[1:])):scale[vertex]=np.median(values)
    return scale


def reach_confidence(residual,scale,*,rings=2.):
    """Fade remote observations before they drag a missing patch to distant skin.

    This is a local refinement trust region, not source-hole classification.
    Larger initial shape/pose differences require branch initialization first.
    """
    if not np.isfinite(rings) or not .5<=rings<=12:raise ValueError('Choose observation reach between 0.5 and 12 local edges')
    residual=np.asarray(residual,dtype=float);scale=np.asarray(scale,dtype=float)
    if residual.shape!=scale.shape or np.any(scale<0) or not np.isfinite(scale).all():raise ValueError('Observation distances and local scales must match')
    fraction=residual/np.maximum(scale*rings,1e-30)
    confidence=np.clip(2*(1-fraction),0,1);confidence[~np.isfinite(residual)|(scale<=0)]=0
    return confidence*confidence*(3-2*confidence)


def observation_fade(observed,triangles,*,rings=4):
    """Blend movement through graph distance from missing observation borders."""
    observed=np.asarray(observed,dtype=bool);triangles=np.asarray(triangles,dtype=np.int64)
    if not isinstance(rings,int) or rings<1:raise ValueError('Observation collars require positive whole rings')
    edges=np.unique(np.sort(np.vstack((triangles[:,[0,1]],triangles[:,[1,2]],triangles[:,[2,0]])),axis=1),axis=0)
    distance=np.full(len(observed),rings,dtype=int);distance[~observed]=0;front=~observed
    for step in range(1,rings):
        vertices=np.unique(edges[np.any(front[edges],axis=1)])
        front=np.zeros(len(observed),bool);front[vertices]=True;front&=distance>step;distance[front]=step
    fade=distance/rings
    return fade*fade*(3-2*fade)
