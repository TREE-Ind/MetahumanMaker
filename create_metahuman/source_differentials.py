"""Source-witness anchored quadratic differential targets for surface fitting.

Local polynomial fitting follows the point-set surface family (Alexa et al.,
Visualization 2001). This module estimates source differentials for a target
one-ring; it does not establish correspondence, injectivity or acceptance.
SciPy is required only by an external numerical worker.
"""
import numpy as np
from scipy import sparse
from scipy.spatial import cKDTree
from .vertex_contract import audit_witnesses,fingerprint


def source_differentials(positions,triangles,source,source_triangles,vertex_ids,
                         record,eligible_face_ids,*,neighbors=96,differential_operator=None):
    """Estimate normal differentials in the caller's actual stencil units.

    The default is the normalized one-ring Laplacian. An explicit sparse
    operator must annihilate constants; its sign, weights and full row support
    are applied to source-derived heights, without assuming uniform neighbors.
    """
    positions=np.asarray(positions,dtype=float);triangles=np.asarray(triangles)
    source=np.asarray(source,dtype=float);source_triangles=np.asarray(source_triangles)
    ids=np.asarray(vertex_ids);eligible=np.asarray(eligible_face_ids)
    if positions.ndim!=2 or positions.shape[1]!=3 or not np.isfinite(positions).all():raise ValueError('Invalid target positions')
    if triangles.dtype.kind not in 'iu' or triangles.ndim!=2 or triangles.shape[1]!=3 or triangles.min()<0 or triangles.max()>=len(positions):raise ValueError('Invalid target triangles')
    if ids.dtype.kind not in 'iu' or ids.ndim!=1 or not len(ids) or len(np.unique(ids))!=len(ids) or ids.min()<0 or ids.max()>=len(positions):raise ValueError('Choose unique physical target IDs')
    if eligible.dtype.kind not in 'iu' or eligible.ndim!=1 or not len(eligible) or eligible.min()<0 or eligible.max()>=len(source_triangles):raise ValueError('Choose original source facet IDs')
    if isinstance(neighbors,bool) or not isinstance(neighbors,int) or not 12<=neighbors<=256:raise ValueError('Choose 12 to 256 source samples per patch')
    proof=audit_witnesses(record,source,source_triangles)
    if len(record['vertex_ids'])!=len(ids) or not np.all(record['observed']) or not np.isin(record['source_triangle_ids'],eligible).all():raise ValueError('Source patch does not support these witnesses')
    facets=source[source_triangles[eligible]];centroids=facets.mean(1)
    cross=np.cross(facets[:,1]-facets[:,0],facets[:,2]-facets[:,0]);area=np.linalg.norm(cross,axis=1)
    normals=cross/np.maximum(area[:,None],1e-300)
    witness_facets=source[source_triangles[record['source_triangle_ids']]]
    normal=np.cross(witness_facets[:,1]-witness_facets[:,0],witness_facets[:,2]-witness_facets[:,0]);normal/=np.maximum(np.linalg.norm(normal,axis=1)[:,None],1e-300)
    axis=np.eye(3)[np.argmin(np.abs(normal),axis=1)];u=np.cross(normal,axis);u/=np.maximum(np.linalg.norm(u,axis=1)[:,None],1e-300);v=np.cross(normal,u)
    centers=record['surface_points'];k=min(neighbors,len(eligible))
    distance,samples=cKDTree(centroids).query(centers,k=k)
    distance=distance.reshape(len(ids),k);samples=samples.reshape(len(ids),k)
    radius=np.maximum(distance[:,-1],np.linalg.norm(np.ptp(source,axis=0))*1e-9)
    delta=centroids[samples]-centers[:,None]
    x=np.einsum('nki,ni->nk',delta,u)/radius[:,None];y=np.einsum('nki,ni->nk',delta,v)/radius[:,None]
    height=np.einsum('nki,ni->nk',delta,normal)/radius[:,None]
    design=np.stack((x,y,x*x,x*y,y*y),axis=2)
    facing=np.einsum('nki,ni->nk',normals[samples],normal)>.35
    weights=np.exp(-2*(distance/radius[:,None])**2)*area[samples]*facing
    weights/=np.maximum(weights.sum(1)[:,None],1e-300)
    gram=np.einsum('nki,nk,nkj->nij',design,weights,design)
    eigen=np.linalg.eigvalsh(gram);supported=(facing.sum(1)>=12)&(eigen[:,0]>np.maximum(eigen[:,-1]*1e-7,1e-15))
    safe=gram.copy();safe[~supported]=np.eye(5)
    rhs=np.einsum('nki,nk,nk->ni',design,weights,height);rhs[~supported]=0
    coefficients=np.linalg.solve(safe,rhs[...,None])[...,0]
    residual=np.sqrt(np.sum(weights*(np.einsum('nki,ni->nk',design,coefficients)-height)**2,axis=1))
    supported&=np.isfinite(coefficients).all(1)&(residual<.04)
    # Confidence is explicitly zero where a smooth local patch is unsupported.
    confidence=np.where(supported,np.clip(1-residual/.04,0,1)*record['confidence'],0.)
    edges=np.unique(np.sort(np.vstack((triangles[:,[0,1]],triangles[:,[1,2]],triangles[:,[2,0]])),axis=1),axis=0);edges=edges[edges[:,0]!=edges[:,1]]
    adjacency=sparse.coo_matrix((np.ones(len(edges)*2),(np.r_[edges[:,0],edges[:,1]],np.r_[edges[:,1],edges[:,0]])),shape=(len(positions),len(positions))).tocsr()
    degree=np.asarray(adjacency.sum(1)).ravel();lap=sparse.eye(len(positions))-sparse.diags(1/np.maximum(degree,1))@adjacency
    if differential_operator is not None:
        lap=sparse.csr_matrix(differential_operator,dtype=float)
        if lap.shape!=(len(positions),len(positions)) or not np.isfinite(lap.data).all():raise ValueError('Invalid source differential operator')
        row_sum=np.asarray(lap[ids].sum(1)).ravel();scale=np.asarray(abs(lap[ids]).sum(1)).ravel()
        if np.any(abs(row_sum)>np.maximum(scale,1)*1e-12):raise ValueError('Source differential operator must annihilate constants')
    target=np.zeros((len(ids),3))
    for row,vertex in enumerate(ids):
        ring=adjacency.indices[adjacency.indptr[vertex]:adjacency.indptr[vertex+1]]
        if differential_operator is not None:
            columns=lap.indices[lap.indptr[vertex]:lap.indptr[vertex+1]];ring=columns[columns!=vertex]
        if not len(ring):confidence[row]=0;continue
        offsets=positions[np.r_[vertex,ring]]-centers[row]
        a=offsets@u[row]/radius[row];b=offsets@v[row]/radius[row]
        basis=np.column_stack((a,b,a*a,a*b,b*b))
        # Reject extrapolation beyond the sampled patch rather than inventing
        # source curvature across a gap, feature or neighboring limb.
        if np.max(np.hypot(a,b))>2.5:confidence[row]=0;continue
        heights=(basis@coefficients[row])*radius[row]
        if differential_operator is None:target[row]=(heights[0]-heights[1:].mean())*normal[row]
        else:
            weights=np.asarray(lap[vertex,np.r_[vertex,ring]].toarray()).ravel()
            target[row]=float(weights@heights)*normal[row]
    target[confidence==0]=0
    return dict(vertex_ids=ids.copy(),target=target,confidence=confidence,
        radius=radius,fit_rms=residual*radius,coefficients=coefficients,laplacian=lap,
        contract=dict(positions=fingerprint(positions),triangles=fingerprint(triangles),
            source=fingerprint(source),source_triangles=fingerprint(source_triangles),
            eligible_source_faces=fingerprint(eligible),witness=proof,
            differential_operator=dict(shape=list(lap.shape),data=fingerprint(lap.data),indices=fingerprint(lap.indices),indptr=fingerprint(lap.indptr))),
        report=dict(method='SOURCE_WITNESS_QUADRATIC_DIFFERENTIALS_V1',operator='UNIFORM_ONE_RING' if differential_operator is None else 'EXPLICIT_CONSTANT_ANNIHILATING_STENCIL',vertices=len(ids),supported=int((confidence>0).sum()),
            fit_rms_max_m=float((residual*radius)[confidence>0].max(initial=0)),source_mutated=False))
