"""Conservative supporting-plane contact for selected anterior skin triangles.

Every supplied triangle is placed outside a supporting plane of the actual
globe's convex hull. Shared closed-lid seams retain one polygonal depth field.
This certifies only the selected sheet, not whole-character/self collision.
"""
import numpy as np
from scipy import sparse
from scipy.spatial import ConvexHull


def clear_skin(eye,skin,triangles,*,clearance,maximum_offset,seam=None,direction=(0.,-1.,0.),support_triangles=None,fixed_vertices=None):
    import clarabel
    eye=np.asarray(eye,dtype=float);skin=np.asarray(skin,dtype=float);tri=np.asarray(triangles,dtype=int);direction=np.asarray(direction,dtype=float)
    if eye.ndim!=2 or skin.ndim!=2 or eye.shape[1]!=3 or skin.shape[1]!=3 or not np.isfinite(eye).all() or not np.isfinite(skin).all() or not np.isfinite(maximum_offset) or maximum_offset<=0 or not np.isfinite(clearance) or clearance<0:raise ValueError('Convex contact needs finite geometry and physical bounds')
    if tri.ndim!=2 or tri.shape[1]!=3 or not len(tri) or tri.min()<0 or tri.max()>=len(skin) or direction.shape!=(3,) or not np.isfinite(direction).all() or np.linalg.norm(direction)<=0:raise ValueError('Choose a valid anterior sheet and contact direction')
    support=tri if support_triangles is None else np.asarray(support_triangles,dtype=int)
    if support.ndim!=2 or support.shape[1]!=3 or not len(support) or support.min()<0 or support.max()>=len(skin):raise ValueError('Choose a valid contact smoothing halo')
    fixed=np.empty(0,dtype=int) if fixed_vertices is None else np.unique(np.asarray(fixed_vertices,dtype=int))
    if fixed.ndim!=1 or (len(fixed) and (fixed.min()<0 or fixed.max()>=len(skin))):raise ValueError('Protected contact vertices leave the skin mesh')
    direction=direction/np.linalg.norm(direction);hull=ConvexHull(eye);planes=hull.equations;den=planes[:,:3]@direction;planes=planes[den>1e-7];den=den[den>1e-7]
    if not len(planes):raise ValueError('The globe has no supporting plane in the correction direction')
    chosen=[];requirements=[]
    for start in range(0,len(tri),128):
        corners=skin[tri[start:start+128]];signed=np.einsum('tvi,fi->tvf',corners,planes[:,:3])+planes[:,3];required=np.maximum((clearance-signed)/den,0.);index=np.argmin(required.max(axis=1),axis=1);chosen.extend(index.tolist());requirements.append(required[np.arange(len(index)),:,index])
    chosen=np.asarray(chosen);requirements=np.vstack(requirements);seam_ids=np.asarray(seam['vertices'],dtype=int) if seam else np.empty(0,dtype=int);active=np.union1d(np.union1d(tri.ravel(),support.ravel()),seam_ids);lookup=np.full(len(skin),-1,dtype=int);lookup[active]=np.arange(len(active));lower=np.zeros(len(active));np.maximum.at(lower,lookup[tri].ravel(),requirements.ravel())
    if lower.max(initial=0)>maximum_offset+1e-10:raise ValueError(f'Anterior convex contact needs {lower.max():.6g}, exceeding its physical correction bound {maximum_offset:.6g}')
    if np.intersect1d(fixed,seam_ids).size:raise ValueError('A shared contact seam cannot also be fixed')
    nonseam=np.setdiff1d(active,np.union1d(seam_ids,fixed));columns={int(v):i for i,v in enumerate(nonseam)};rows=[];cols=[];values=[];variables=len(nonseam)
    for vertex,column in columns.items():rows.append(lookup[vertex]);cols.append(column);values.append(1.)
    if seam:
        knots=np.asarray(seam['knots']);parameters=np.asarray(seam['parameters']);interval=np.clip(np.searchsorted(knots,parameters,side='right')-1,0,len(knots)-2);t=(parameters-knots[interval])/(knots[interval+1]-knots[interval])
        for vertex,i,weight in zip(seam_ids,interval,t):rows.extend([lookup[vertex]]*2);cols.extend([variables+int(i),variables+int(i)+1]);values.extend([1-weight,weight])
        variables+=len(knots)
    if not variables:raise ValueError('Protected vertices leave no contact degrees of freedom')
    P=sparse.coo_matrix((values,(rows,cols)),shape=(len(active),variables)).tocsc();edges=np.unique(np.sort(np.vstack((support[:,[0,1]],support[:,[1,2]],support[:,[2,0]])),axis=1),axis=0);edges=lookup[edges];D=sparse.coo_matrix((np.tile([1.,-1.],len(edges)),(np.repeat(np.arange(len(edges)),2),edges.ravel())),shape=(len(edges),len(active))).tocsc();Q=P.T@(sparse.eye(len(active),format='csc')+2*D.T@D)@P;A=sparse.vstack((-P,P)).tocsc();bounds=np.r_[-lower,np.full(len(active),maximum_offset)]
    settings=clarabel.DefaultSettings();settings.verbose=False;settings.tol_feas=1e-10;settings.tol_gap_abs=1e-11
    solution=clarabel.DefaultSolver(sparse.triu(Q).tocsc(),np.zeros(variables),A,bounds,[clarabel.NonnegativeConeT(len(bounds))],settings).solve()
    if str(solution.status)!='Solved':raise ValueError('Bounded anterior convex contact did not converge')
    offset=np.asarray(P@np.asarray(solution.x)).ravel();result=skin.copy();result[active]+=offset[:,None]*direction;normal=planes[chosen,:3];minimum=np.min(np.einsum('tvi,ti->tv',result[tri],normal)+planes[chosen,3,None],axis=1)
    if minimum.min(initial=clearance)<clearance-1e-8 or offset.max(initial=0)>maximum_offset+1e-8:raise ValueError('The final skin triangles do not satisfy their convex contact certificate')
    return result,dict(method='ANTERIOR_CONVEX_SUPPORT_PLANES',triangles=len(tri),active_vertices=len(active),depth_variables=variables,shared_seam=seam is not None,maximum_correction=float(offset.max(initial=0)),minimum_plane_clearance=float(minimum.min()),clearance=clearance,globe_vertices_changed=False,scope='Selected anterior sheet outside actual globe convex hull; not whole-motion or self-collision certification')
