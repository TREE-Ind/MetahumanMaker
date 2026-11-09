"""Piecewise derivatives of the original source's interpolated normal field.

Active source facets/features stay fixed within a local proposal. Exact source
queries and nonlinear endpoint checks remain the acceptance authority.
"""
import numpy as np
from .vertex_contract import closest_triangle_points


def barycentric_jacobian(points,facets):
    _,bary,_=closest_triangle_points(points,facets)
    a=facets[:,1]-facets[:,0];b=facets[:,2]-facets[:,0]
    aa=np.sum(a*a,axis=1);ab=np.sum(a*b,axis=1);bb=np.sum(b*b,axis=1);det=aa*bb-ab*ab
    result=np.zeros((len(points),3,3));positive=bary>1e-10
    inside=np.all(positive,axis=1)&(det>np.maximum(aa*bb*1e-14,1e-300))
    u=(bb[inside,None]*a[inside]-ab[inside,None]*b[inside])/det[inside,None]
    v=(aa[inside,None]*b[inside]-ab[inside,None]*a[inside])/det[inside,None]
    result[inside]=np.stack((-u-v,u,v),axis=1)
    for i,j in ((0,1),(1,2),(2,0)):
        use=(positive.sum(1)==2)&positive[:,i]&positive[:,j]
        edge=facets[use,j]-facets[use,i];gradient=edge/np.maximum(np.sum(edge*edge,axis=1)[:,None],1e-300)
        result[use,i]=-gradient;result[use,j]=gradient
    return bary,result


def normal_jacobians(points,source,source_triangles,faces,target_triangles,corner_normals,*,return_vertex_field=False):
    """Per-vertex normal derivatives, followed by normalized face averages."""
    faces=np.asarray(faces,dtype=np.int64);valid=faces>=0;field=np.zeros_like(points,dtype=float);derivative=np.zeros((len(points),3,3))
    facets=np.asarray(source)[np.asarray(source_triangles)[faces[valid]]];corners=np.asarray(corner_normals)[faces[valid]]
    bary,db=barycentric_jacobian(np.asarray(points)[valid],facets)
    blended=np.einsum('ni,nij->nj',bary,corners);length=np.linalg.norm(blended,axis=1);unit=blended/np.maximum(length[:,None],1e-30)
    P=(np.eye(3)[None]-unit[:,:,None]*unit[:,None,:])/np.maximum(length[:,None,None],1e-30)
    derivative[valid]=P@np.einsum('nki,nkj->nij',corners,db);field[valid]=unit
    tri=np.asarray(target_triangles);summed=field[tri].sum(1);size=np.linalg.norm(summed,axis=1);normals=summed/np.maximum(size[:,None],1e-30)
    P=(np.eye(3)[None]-normals[:,:,None]*normals[:,None,:])/np.maximum(size[:,None,None],1e-30)
    result=(normals,P[:,None]@derivative[tri])
    return (*result,field,derivative) if return_vertex_field else result


def transported_model(points,triangles,free,contract,faces,source,source_triangles,corner_normals,*,linearization_points=None,field_derivatives=None):
    """Linearize area after rotating the endpoint field to its current frame.

    For source direction n and area C, d(R C) = dC +
    [n C^T - (n dot C) I] dn. Rotation preserves area length to first order.
    """
    from scipy import sparse
    from .surface_alignment import angular_model
    model=angular_model(points,triangles,free,contract,linearization_points=linearization_points)
    normals,D=normal_jacobians(points,source,source_triangles,faces,triangles,corner_normals) if field_derivatives is None else field_derivatives
    ids=contract['face_ids'];tri=np.asarray(triangles)[ids];n=normals[ids];p=np.asarray(points)[tri]
    if np.max(np.abs(n-contract['normals']),initial=0)>1e-10:raise ValueError('Transport derivative differs from the current original-source field')
    C=np.cross(p[:,1]-p[:,0],p[:,2]-p[:,0])/contract['area_scale'][:,None]
    Q=n[:,:,None]*C[:,None,:]-np.sum(n*C,axis=1)[:,None,None]*np.eye(3)[None]
    lookup=np.full(len(points),-1,dtype=np.int64);lookup[free]=np.arange(np.count_nonzero(free));rows=[];columns=[];values=[]
    correction=Q[:,None]@D[ids]
    for corner in range(3):
        variables=lookup[tri[:,corner]];use=variables>=0
        r=np.broadcast_to(np.arange(len(ids))[:,None,None]*3+np.arange(3)[None,:,None],(len(ids),3,3))
        c=np.broadcast_to(variables[:,None,None]*3+np.arange(3)[None,None,:],(len(ids),3,3))
        rows.append(r[use].ravel());columns.append(c[use].ravel());values.append(correction[use,corner].ravel())
    B=sparse.coo_matrix((np.concatenate(values),(np.concatenate(rows),np.concatenate(columns))),shape=model['matrix'].shape).tocsr()
    model['matrix']=(model['matrix']+B).tocsr()
    model['source_field_derivatives']=dict(method='ORIGINAL_SOURCE_NORMAL_PARALLEL_TRANSPORT_JACOBIAN_V1',faces=len(ids),proposal_only=True,nonlinear_checks_required=True)
    return model
