"""Conservative surface projection through overlapping triangle charts.

External NumPy/SciPy worker only. Integration cells are intersections of source
and target chart triangles, so tiny target basis functions cannot disappear
between source samples. Original meshes and their orders are never edited.
"""
import numpy as np
from scipy import sparse
from scipy.spatial import cKDTree
from .vertex_contract import fingerprint


def _cross(a,b):return a[...,0]*b[...,1]-a[...,1]*b[...,0]


def _mesh(uv,triangles):
    uv=np.asarray(uv,dtype=float);triangles=np.asarray(triangles)
    if uv.ndim!=2 or uv.shape[1]!=2 or not np.isfinite(uv).all():raise ValueError('Choose finite two-dimensional chart coordinates')
    if triangles.dtype.kind not in 'iu' or triangles.ndim!=2 or triangles.shape[1]!=3 or not len(triangles) or triangles.min()<0 or triangles.max()>=len(uv):raise ValueError('Invalid ordered chart triangles')
    triangles=triangles.astype(np.int64,copy=False);p=uv[triangles];area=_cross(p[:,1]-p[:,0],p[:,2]-p[:,0])
    if np.any(area<=0):raise ValueError('Chart triangles must be positively oriented and nondegenerate')
    return uv,triangles,p,area


def _pairs(target,source,cancelled):
    sl=source.min(1);sh=source.max(1);sc=source.mean(1);sr=np.linalg.norm(source-sc[:,None],axis=2).max(1)
    tl=target.min(1);th=target.max(1);tc=target.mean(1);tr=np.linalg.norm(target-tc[:,None],axis=2).max(1)
    floor=max(float(np.linalg.norm(np.ptp(sc,axis=0)))*1e-12,1e-15);bins=np.floor(np.log2(np.maximum(sr,floor)/floor)).astype(int);parts=[]
    for group in np.unique(bins):
        ids=np.flatnonzero(bins==group);tree=cKDTree(sc[ids]);radius=float(sr[ids].max())
        for start in range(0,len(target),128):
            if cancelled and cancelled():raise RuntimeError('Source overlay cancelled')
            query=np.arange(start,min(start+128,len(target)));found=tree.query_ball_point(tc[query],tr[query]+radius+floor)
            lengths=np.asarray([len(x) for x in found])
            if not lengths.sum():continue
            first=np.repeat(query,lengths);second=ids[np.concatenate(found).astype(int)]
            keep=np.all(tl[first]<=sh[second]+floor,axis=1)&np.all(sl[second]<=th[first]+floor,axis=1)
            if keep.any():parts.append(np.column_stack((first[keep],second[keep])))
    return np.vstack(parts) if parts else np.empty((0,2),dtype=int)


def _polygons(source_local,inverse_source_local):
    """Intersect each source triangle with the reference target simplex."""
    ref=np.array([[0.,0.],[1.,0.],[0.,1.]])
    delta=ref[None]-source_local[:,None,0,:]
    eta=np.einsum('nvi,nji->nvj',delta,inverse_source_local)
    in_source=np.all(eta>=-1e-10,axis=2)&(eta.sum(2)<=1+1e-10)
    in_target=np.all(source_local>=-1e-10,axis=2)&(source_local.sum(2)<=1+1e-10)
    r=(np.roll(ref,-1,axis=0)-ref)[None,:,None,:]
    s=(np.roll(source_local,-1,axis=1)-source_local)[:,None,:,:]
    qp=source_local[:,None,:,:]-ref[None,:,None,:]
    denominator=_cross(r,s);valid=abs(denominator)>1e-14*np.linalg.norm(r,axis=3)*np.linalg.norm(s,axis=3)
    t=np.divide(_cross(qp,s),denominator,out=np.zeros_like(denominator),where=valid)
    u=np.divide(_cross(qp,r),denominator,out=np.zeros_like(denominator),where=valid)
    valid&=(t>=-1e-10)&(t<=1+1e-10)&(u>=-1e-10)&(u<=1+1e-10)
    intersections=ref[None,:,None,:]+t[:,:,:,None]*r
    points=np.concatenate((np.broadcast_to(ref,(len(source_local),3,2)),source_local,intersections.reshape(-1,9,2)),axis=1)
    mask=np.column_stack((in_source,in_target,valid.reshape(-1,9)));count=mask.sum(1)
    center=np.where(mask[:,:,None],points,0).sum(1)/np.maximum(count[:,None],1)
    angle=np.arctan2(points[:,:,1]-center[:,None,1],points[:,:,0]-center[:,None,0]);angle[~mask]=np.inf
    order=np.argsort(angle,axis=1,kind='stable')
    return np.take_along_axis(points,order[:,:,None],axis=1),count


def assemble_surface_projection(target_uv,target_triangles,source_uv,source_triangles,source_vertices,*,target_vertex_ids=None,target_vertex_count=None,cancelled=None):
    """Assemble an L2 projection using positive triangle-overlap integration.

    Both charts must be injective oriented triangulations. This function checks
    triangle orientation, not global chart injectivity. Optional target IDs
    gather chart seam copies into shared physical unknowns. Returned matrix,
    load, mass and constant evaluate the integrated geometry error. Coverage
    outside the shared chart domain is reported, not silently extrapolated.
    """
    tu,tt,tp,ta=_mesh(target_uv,target_triangles);su,st,sp,sa=_mesh(source_uv,source_triangles)
    source_vertices=np.asarray(source_vertices,dtype=float)
    if source_vertices.shape!=(len(su),3) or not np.isfinite(source_vertices).all():raise ValueError('Source positions must match the source chart vertices')
    ids=np.arange(len(tu)) if target_vertex_ids is None else np.asarray(target_vertex_ids)
    if ids.dtype.kind not in 'iu' or ids.shape!=(len(tu),) or ids.min()<0:raise ValueError('Invalid physical target vertex IDs')
    if target_vertex_count is not None and (isinstance(target_vertex_count,(bool,np.bool_)) or not isinstance(target_vertex_count,(int,np.integer))):
        raise ValueError('Target vertex count must be an integer')
    count=int(ids.max())+1 if target_vertex_count is None else int(target_vertex_count)
    if count<=ids.max():raise ValueError('Target vertex count excludes chart vertices')
    xyz=source_vertices[st];source_area=np.linalg.norm(np.cross(xyz[:,1]-xyz[:,0],xyz[:,2]-xyz[:,0]),axis=1)*.5
    source_normal=np.cross(xyz[:,1]-xyz[:,0],xyz[:,2]-xyz[:,0])/np.maximum(source_area[:,None]*2,1e-300)
    pairs=_pairs(tp,sp,cancelled);rows=[];columns=[];entries=[];load=np.zeros((count,3));mass=np.zeros(count);constant=0.;covered=0.;cells=0
    face_area=np.zeros(len(tt));face_normal_integral=np.zeros((len(tt),3))
    for start in range(0,len(pairs),2048):
        if cancelled and cancelled():raise RuntimeError('Source overlay cancelled')
        ti,si=pairs[start:start+2048].T;n=tp[ti];s=sp[si]
        dn=np.stack((n[:,1]-n[:,0],n[:,2]-n[:,0]),axis=2);inv_n=np.linalg.inv(dn)
        ds=np.stack((s[:,1]-s[:,0],s[:,2]-s[:,0]),axis=2);inv_s=np.linalg.inv(ds)
        local=np.einsum('nvi,nji->nvj',s-n[:,None,0],inv_n)
        inv_local=inv_s@dn
        polygon,lengths=_polygons(local,inv_local)
        matrix=np.zeros((len(ti),3,3));vector=np.zeros_like(matrix);lumped=np.zeros((len(ti),3));energy=np.zeros(len(ti));pair_area=np.zeros(len(ti))
        world_density=source_area[si]*2*ta[ti]/sa[si]
        for k in range(1,14):
            active=np.flatnonzero(lengths>k+1)
            if not len(active):continue
            q=np.stack((polygon[active,0],polygon[active,k],polygon[active,k+1]),axis=1)
            area=_cross(q[:,1]-q[:,0],q[:,2]-q[:,0])*.5
            good=area>1e-20;active=active[good];q=q[good];area=area[good]*world_density[active]
            if not len(active):continue
            basis=np.stack((1-q[:,:,0]-q[:,:,1],q[:,:,0],q[:,:,1]),axis=2);basis=np.clip(basis,0,1);basis/=basis.sum(2)[:,:,None]
            # Source barycentrics use original UV differences for stability,
            # never large extrapolated world-space coordinates.
            delta=n[active,None,0]-s[active,None,0]+np.einsum('nij,nvj->nvi',dn[active],q)
            eta=np.einsum('nvi,nji->nvj',delta,inv_s[active]);sb=np.stack((1-eta[:,:,0]-eta[:,:,1],eta[:,:,0],eta[:,:,1]),axis=2)
            if sb.min()<-1e-6 or sb.max()>1+1e-6:raise ValueError('Overlay intersection lost source containment')
            sb=np.clip(sb,0,1);sb/=sb.sum(2)[:,:,None];world=np.einsum('nvi,nij->nvj',sb,xyz[si[active]])
            bsum=basis.sum(1);xsum=world.sum(1)
            matrix[active]+=(np.einsum('nvi,nvj->nij',basis,basis)+bsum[:,:,None]*bsum[:,None,:])*(area/12)[:,None,None]
            vector[active]+=(np.einsum('nvi,nvj->nij',basis,world)+bsum[:,:,None]*xsum[:,None,:])*(area/12)[:,None,None]
            lumped[active]+=bsum*(area/3)[:,None]
            energy[active]+=(np.sum(world**2,axis=(1,2))+np.sum(xsum**2,axis=1))*(area/12)
            pair_area[active]+=area;cells+=len(active)
        physical=ids[tt[ti]]
        rows.append(np.repeat(physical,3,axis=1).ravel());columns.append(np.tile(physical,(1,3)).ravel());entries.append(matrix.ravel())
        for corner in range(3):
            np.add.at(load,physical[:,corner],vector[:,corner]);np.add.at(mass,physical[:,corner],lumped[:,corner])
        np.add.at(face_area,ti,pair_area);np.add.at(face_normal_integral,ti,pair_area[:,None]*source_normal[si])
        constant+=float(energy.sum());covered+=float(pair_area.sum())
    matrix=sparse.coo_matrix((np.concatenate(entries) if entries else [],(np.concatenate(rows) if rows else [],np.concatenate(columns) if columns else [])),shape=(count,count)).tocsr();matrix.eliminate_zeros()
    total_area=float(source_area.sum())
    return dict(matrix=matrix,load=load,mass=mass,constant=constant,target_face_area=face_area,target_face_normal_integral=face_normal_integral,
        contract=dict(target_uv=fingerprint(tu),target_triangles=fingerprint(tt),target_vertex_ids=fingerprint(ids),source_uv=fingerprint(su),source_triangles=fingerprint(st),source_vertices=fingerprint(source_vertices)),
        report=dict(method='POSITIVE_CHART_OVERLAY_L2_PROJECTION_V1',candidate_pairs=len(pairs),integration_triangles=cells,source_area=total_area,covered_area=covered,
            covered_source_fraction=covered/max(total_area,1e-300),observed_target_vertices=int((mass>0).sum()),partition_unity_max_error=float(np.max(abs(matrix@np.ones(count)-mass),initial=0)),source_mutated=False))


def projection_rms(projection,positions):
    positions=np.asarray(positions,dtype=float);matrix=projection['matrix'];load=projection['load']
    if positions.shape!=load.shape or not np.isfinite(positions).all():raise ValueError('Projection positions must share the physical target order')
    energy=float(np.sum(positions*(matrix@positions))-2*np.sum(positions*load)+projection['constant'])
    return float(np.sqrt(max(energy,0.)/max(float(projection['mass'].sum()),1e-300)))
