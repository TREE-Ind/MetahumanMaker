"""Infer source anatomy from a registered, rig-labelled target surface.

External worker: NumPy/SciPy only. Exact triangle witnesses transfer influence
by barycentric interpolation, without axis, side, size or vertex-order rules.
This is a registration-dependent inference, not semantic ground truth.
"""
import numpy as np
from .surface_oracle import SurfaceOracle
from .vertex_contract import audit_witnesses,fingerprint


def normal_motion_support(query,targets,normals,*,slide_tolerance):
    """Downweight nearest-surface matches that mainly slide along the skin.

    Registration/pose alignment must establish tangential placement first.
    This refinement confidence is invariant to world-axis orientation.
    """
    query=np.asarray(query,dtype=float);targets=np.asarray(targets,dtype=float);normals=np.asarray(normals,dtype=float)
    if query.ndim!=2 or query.shape[1]!=3 or targets.shape!=query.shape or normals.shape!=query.shape or not all(np.isfinite(x).all() for x in (query,targets,normals)):
        raise ValueError('Motion support arrays must be matching finite N by 3 arrays')
    if not np.isfinite(slide_tolerance) or slide_tolerance<=0:raise ValueError('Choose finite positive sliding tolerance')
    length=np.linalg.norm(normals,axis=1);unit=normals/np.maximum(length[:,None],1e-300)
    delta=targets-query;normal_distance=np.einsum('ij,ij->i',delta,unit)
    tangent=np.linalg.norm(delta-normal_distance[:,None]*unit,axis=1)
    allowance=np.maximum(slide_tolerance,abs(normal_distance)*.3)
    support=1/(1+(tangent/allowance)**4)
    support[length<1e-12]=0
    return support


def infer_source_regions(target, target_triangles, influences, source,
                         source_triangles, *, max_distance, face_mask=None,
                         normal_limit=None, cancelled=None,correspondence=None):
    """Return soft source-face labels and an auditable reverse correspondence.

    Each eligible source facet is sampled at its centroid. Registered target
    triangles supply rig weights. Optional normal filtering is appropriate
    only for reliable orientation; a folded initial fit must not cut holes
    in coarse anatomical masks by rejecting opposing skin walls.
    A returned ``correspondence`` can reweight another influence set without
    repeating geometry queries. Exact geometry/order and query-setting hashes
    must match; changed geometry is rejected rather than silently reused.
    Unsupported facets stay unlabelled;
    no nearest-vertex shortcut or forced winner fills missing observations.
    Source garments can be excluded with a caller-supplied face mask. Samples
    do not certify all of a large facet, so artists may override the labels.
    """
    target=np.asarray(target,dtype=float);source=np.asarray(source,dtype=float)
    st=np.asarray(source_triangles,dtype=np.int64);tt=np.asarray(target_triangles,dtype=np.int64)
    weights=np.asarray(influences,dtype=float)
    if weights.ndim!=2 or weights.shape[0]!=len(target) or not weights.shape[1] or not np.isfinite(weights).all() or np.any((weights<0)|(weights>1)):
        raise ValueError('Rig influences must be finite N by K values between zero and one')
    if source.ndim!=2 or source.shape[1]!=3 or not np.isfinite(source).all() or st.ndim!=2 or st.shape[1]!=3 or not len(st) or st.min()<0 or st.max()>=len(source):
        raise ValueError('Invalid source triangle surface')
    if not np.isfinite(max_distance) or max_distance<=0:raise ValueError('Choose a finite positive registration support distance')
    mask=np.ones(len(st),bool) if face_mask is None else np.asarray(face_mask,dtype=bool)
    if mask.shape!=(len(st),) or not mask.any():raise ValueError('Choose eligible source facets')
    ids=np.flatnonzero(mask);surface=source[st[ids]];query=surface.mean(axis=1)
    normal=np.cross(surface[:,1]-surface[:,0],surface[:,2]-surface[:,0])
    normal/=np.maximum(np.linalg.norm(normal,axis=1)[:,None],1e-300)
    contract=dict(target=fingerprint(target),target_triangles=fingerprint(tt),source=fingerprint(source),
        source_triangles=fingerprint(st),eligible_faces=fingerprint(ids),max_distance=float(max_distance),normal_limit=normal_limit)
    reused=correspondence is not None
    if reused:
        if correspondence.get('contract')!=contract:raise ValueError('Cached anatomy correspondence no longer matches the registered surfaces or query settings')
        record=correspondence['witnesses']
        if len(record['source_triangle_ids'])!=len(ids):raise ValueError('Cached source facet order changed')
    else:
        oracle=SurfaceOracle(target,tt)
        record=oracle.query(query,max_distance=max_distance,query_normals=None if normal_limit is None else normal,
                            normal_limit=.1 if normal_limit is None else normal_limit,cancelled=cancelled)
    audit=audit_witnesses(record,target,tt)
    observed=record['observed'];local=np.zeros((len(ids),weights.shape[1]))
    local[observed]=np.einsum('ni,nik->nk',record['barycentric'][observed],
        weights[tt[record['source_triangle_ids'][observed]]])
    # Barycentric sums can exceed one by an ulp on fully weighted faces.
    np.clip(local,0.,1.,out=local)
    transferred=np.zeros((len(st),weights.shape[1]));transferred[ids]=local
    confidence=np.zeros(len(st))
    # Separation from the registered target is evidence uncertainty, rather
    # than an unreported hard assignment to the nearest anatomical label.
    confidence[ids[observed]]=record['confidence'][observed]*np.maximum(0.,
        1-(record['residual'][observed]/max_distance)**2)
    report=dict(method='BARYCENTRIC_REGISTERED_ANATOMY_TRANSFER_V1',eligible_faces=len(ids),
        observed_faces=int(observed.sum()),unobserved_faces=int((~observed).sum()),
        max_distance=float(max_distance),normal_limit=None if normal_limit is None else float(normal_limit),
        normal_filter_applied=normal_limit is not None,
        correspondence_reused=reused,
        coordinate_boxes_used=False,source_mutated=False,witness_audit=audit,
        limitations=['Requires a reasonably registered target',
                    'Centroid samples may straddle anatomical boundaries',
                    'Rig influence is inferred anatomy, not ground truth'])
    return dict(influences=transferred,confidence=confidence,source_face_ids=ids,
                witnesses=record,report=report,correspondence=dict(contract=contract,witnesses=record))


def region_faces(transfer, column, *, minimum_influence=.08, minimum_confidence=.05):
    """Select observed source facets for a region, allowing overlapping labels."""
    weights=np.asarray(transfer['influences']);confidence=np.asarray(transfer['confidence'])
    if not isinstance(column,(int,np.integer)) or not 0<=column<weights.shape[1]:raise ValueError('Invalid anatomical region column')
    if not 0<=minimum_influence<1 or not 0<=minimum_confidence<1:raise ValueError('Region thresholds must lie between zero and one')
    return np.flatnonzero((weights[:,column]>minimum_influence)&(confidence>minimum_confidence))


def region_from_attachment_curve(vertices,triangles,curve,distal_face_ids,*,
                                 excluded_face_ids=(),anchors=12,margin=None,cancelled=None):
    """Trace a source region from an ordered, registered attachment curve.

    Requires a closed genus-zero source. Paths follow original physical edges;
    source coordinates, facets and UV copies are never edited. A simple loop,
    two complementary face components, complete distal containment and no
    excluded facets are required. This transfers a supplied anatomical prior;
    it does not recognize anatomy independently of the registration.
    """
    from scipy import sparse
    from scipy.spatial import cKDTree
    from scipy.sparse.csgraph import dijkstra,connected_components
    vertices=np.asarray(vertices,dtype=float);triangles=np.asarray(triangles)
    curve=np.asarray(curve,dtype=float)
    if vertices.ndim!=2 or vertices.shape[1]!=3 or not np.isfinite(vertices).all():raise ValueError('Invalid source vertices')
    if triangles.dtype.kind not in 'iu' or triangles.ndim!=2 or triangles.shape[1]!=3 or not len(triangles) or triangles.min()<0 or triangles.max()>=len(vertices):raise ValueError('Invalid ordered source triangles')
    triangles=triangles.astype(np.int64,copy=False)
    if curve.ndim!=2 or curve.shape[1]!=3 or not np.isfinite(curve).all() or int(anchors)!=anchors or not 3<=anchors<=len(curve):raise ValueError('Provide a densely sampled finite closed attachment curve and at least three anchors')
    def face_ids(values,required=False):
        values=np.asarray(values)
        if values.size==0 and not required:return np.empty(0,dtype=int)
        if values.dtype.kind not in 'iu' or values.ndim!=1 or not len(values) or values.min()<0 or values.max()>=len(triangles):raise ValueError('Region constraints must name original source face indices')
        return values.astype(np.int64,copy=False)
    distal=face_ids(distal_face_ids,True);excluded=face_ids(excluded_face_ids)
    physical,representatives,weld=np.unique(vertices,axis=0,return_index=True,return_inverse=True);tri=weld[triangles]
    directed=np.vstack((tri[:,[0,1]],tri[:,[1,2]],tri[:,[2,0]]))
    edges,inverse,counts=np.unique(np.sort(directed,axis=1),axis=0,return_inverse=True,return_counts=True)
    if np.any(counts!=2) or len(np.unique(tri))-len(edges)+len(tri)!=2:raise ValueError('Attachment tracing requires a closed manifold genus-zero source')
    owners=np.tile(np.arange(len(tri)),3)[np.argsort(inverse)].reshape(-1,2)
    lengths=np.linalg.norm(physical[edges[:,0]]-physical[edges[:,1]],axis=1);midpoints=physical[edges].mean(1)
    curve_lengths=np.linalg.norm(np.roll(curve,-1,axis=0)-curve,axis=1)
    if np.any(curve_lengths<=max(float(np.linalg.norm(np.ptp(curve,axis=0)))*1e-12,1e-15)):raise ValueError('Attachment curve has collapsed segments')
    arc=np.r_[0.,np.cumsum(curve_lengths)];arc/=arc[-1]
    curve_anchor_ids=np.searchsorted(arc[:-1],np.arange(anchors)/anchors)%len(curve)
    if len(np.unique(curve_anchor_ids))!=anchors:raise ValueError('Resample the attachment curve more uniformly before tracing')
    anchor_points=curve[curve_anchor_ids]
    witness=SurfaceOracle(vertices,triangles).query(anchor_points,cancelled=cancelled)
    if not witness['observed'].all():raise ValueError('Attachment anchors have no source support')
    source_anchors=cKDTree(physical).query(witness['surface_points'])[1]
    if len(np.unique(source_anchors))!=anchors:raise ValueError('Attachment anchors project ambiguously onto the source')
    if margin is None:margin=max(float(np.linalg.norm(np.ptp(curve,axis=0)))*.15,float(witness['residual'].max())*2,float(np.median(lengths))*8)
    if not np.isfinite(margin) or margin<=0:raise ValueError('Choose a finite positive curve search margin')
    used=set();loop=[]
    for k in range(anchors):
        if cancelled and cancelled():raise RuntimeError('Source attachment tracing cancelled')
        first=int(source_anchors[k]);last=int(source_anchors[(k+1)%anchors])
        ia=int(curve_anchor_ids[k]);ib=int(curve_anchor_ids[(k+1)%anchors])
        indices=np.arange(ia,ib+1) if ib>ia else np.r_[np.arange(ia,len(curve)),np.arange(ib+1)]
        line=curve[indices];samples=np.vstack([line[:-1]*(1-t)+line[1:]*t for t in np.linspace(0,1,8)])
        nearby=cKDTree(samples).query(midpoints)[0];keep=nearby<margin
        blocked=np.zeros(len(physical),bool);blocked[list(used)]=True;blocked[source_anchors]=True;blocked[[first,last]]=False
        keep&=~np.any(blocked[edges],axis=1);edge=edges[keep]
        weight=lengths[keep]*(1+(nearby[keep]/max(margin*.2,1e-6))**2)
        graph=sparse.coo_matrix((np.r_[weight,weight],(np.r_[edge[:,0],edge[:,1]],np.r_[edge[:,1],edge[:,0]])),shape=(len(physical),len(physical))).tocsr()
        distance,previous=dijkstra(graph,directed=False,indices=first,return_predecessors=True)
        if not np.isfinite(distance[last]):raise ValueError('Attachment curve cannot be embedded without crossing earlier paths')
        path=[last]
        while path[-1]!=first:path.append(int(previous[path[-1]]))
        path=path[::-1];loop.extend(path[:-1]);used.update(path)
    loop=np.asarray(loop,dtype=int)
    if len(np.unique(loop))!=len(loop):raise ValueError('Source attachment curve is not simple')
    cut={tuple(x) for x in np.sort(np.column_stack((loop,np.roll(loop,-1))),axis=1)}
    pairs=owners[np.asarray([tuple(edge) not in cut for edge in edges])]
    graph=sparse.coo_matrix((np.ones(len(pairs)*2),(np.r_[pairs[:,0],pairs[:,1]],np.r_[pairs[:,1],pairs[:,0]])),shape=(len(tri),len(tri))).tocsr()
    components,labels=connected_components(graph,directed=False)
    if components!=2:raise ValueError('Source attachment curve did not create two complementary surface components')
    component=int(np.bincount(labels[distal]).argmax());selected=np.flatnonzero(labels==component)
    if not np.isin(distal,selected).all():raise ValueError('Attachment boundary cuts the required distal region')
    if np.isin(excluded,selected).any():raise ValueError('Attachment boundary includes an excluded anatomical region')
    return dict(face_ids=selected,boundary_vertex_ids=representatives[loop],curve_anchor_indices=curve_anchor_ids,
        source_anchor_vertex_ids=representatives[source_anchors],witnesses=witness,
        contract=dict(source_vertices=fingerprint(vertices),source_triangles=fingerprint(triangles),registered_curve=fingerprint(curve)),
        report=dict(method='REGISTERED_ATTACHMENT_EDGE_CURVE_REGION_V1',source_faces=len(selected),boundary_vertices=len(loop),anchors=int(anchors),
            curve_margin=float(margin),anchor_source_rms=float(np.sqrt(np.mean(witness['residual']**2))),source_mutated=False,
            distal_fully_contained=True,excluded_facets=0,source_region_review_required=True))
