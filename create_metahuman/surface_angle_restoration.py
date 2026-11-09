"""Local nonlinear source-angle restoration for a measured joint proposal.

Source facets and artist/contact envelopes remain fixed. A successful local
correction is still only a proposal: the caller must repeat actual contact,
orientation, merit and native checks before committing geometry.
"""
import time
import numpy as np
from scipy import sparse
from .surface_alignment import angular_model, angular_values
from .source_envelope import SourceFacetBoxes
from .vertex_contract import witnesses, fingerprint
from .convex_surface_step import project_joint_conic


def restore(candidate,current,triangles,free,contract,matrix,bounds,lower,upper,projector,deadline,*,passes=4,orientation_normals=None,minimum_area=None,collar_rings=1):
    candidate=np.asarray(candidate,dtype=float);current=np.asarray(current,dtype=float)
    free=np.asarray(free,dtype=bool);triangles=np.asarray(triangles,dtype=np.int64)
    matrix=sparse.csr_matrix(matrix);bounds=np.asarray(bounds);lower=np.asarray(lower);upper=np.asarray(upper)
    started=time.monotonic();original=candidate.copy();point=original.copy();history=[]
    if not isinstance(collar_rings,int) or not 1<=collar_rings<=4:raise ValueError('Choose one to four local correction collar rings')
    def errors(points):
        cosine,residual=angular_values(points,triangles,contract)
        return residual,np.maximum(contract['minimum_cosine']-cosine,0.)
    residual,cosine_error=errors(point);bad=(residual>1e-9)|(cosine_error>1e-7)
    orientation_ids=np.empty(0,dtype=np.int64);orientation_scale=np.empty(0);orientation_floor=np.empty(0)
    def orientation_errors(points):
        p=points[triangles[orientation_ids]];area=np.cross(p[:,1]-p[:,0],p[:,2]-p[:,0])
        signed=np.einsum('ij,ij->i',area,orientation_normals[orientation_ids]) if len(orientation_ids) else np.empty(0)
        return np.maximum(orientation_floor-signed/orientation_scale,0.)
    if orientation_normals is not None:
        orientation_normals=np.asarray(orientation_normals,dtype=float);minimum_area=np.asarray(minimum_area,dtype=float)
        if orientation_normals.shape!=(len(triangles),3) or minimum_area.shape!=(len(triangles),) or not np.isfinite(orientation_normals).all() or not np.isfinite(minimum_area).all():raise ValueError('Invalid restoration orientation envelope')
        p=current[triangles];cross=np.cross(p[:,1]-p[:,0],p[:,2]-p[:,0]);signed=np.einsum('ij,ij->i',cross,orientation_normals)
        orientation_ids=np.flatnonzero((signed>0)&np.any(free[triangles],axis=1));orientation_scale=np.linalg.norm(cross[orientation_ids],axis=1)
        orientation_floor=np.minimum(minimum_area[orientation_ids],signed[orientation_ids]*.5)/orientation_scale
    orientation_error=orientation_errors(point);bad_orientation=orientation_error>1e-9
    report=dict(method='LOCAL_JOINT_SOURCE_ANGLE_RESTORATION_V2',valid=False,initial_bad_faces=int(bad.sum()),initial_bad_orientation_faces=int(bad_orientation.sum()),
                initial_residual=float(residual.max(initial=0)),history=history,source_facets_refreshed=False,collar_rings=collar_rings)
    def finish(reason,valid=False):
        result=point if valid else original
        report.update(reason=reason,valid=bool(valid),elapsed_seconds=time.monotonic()-started,
                      maximum_correction_m=float(np.linalg.norm(result-original,axis=1).max(initial=0)))
        return result,report
    if not bad.any() and not bad_orientation.any():return finish('already_feasible',True)
    if projector.contract.get('method')!='EXACT_SOURCE_FACET_AND_ARTIST_BOXES_V1':return finish('unsupported_source_envelope')
    local=np.zeros(len(point),bool);local[np.unique(triangles[contract['face_ids'][bad]])]=True
    local[np.unique(triangles[orientation_ids[bad_orientation]])]=True
    for _ in range(collar_rings):
        local[np.unique(triangles[np.any(local[triangles],axis=1)])]=True
        local&=free
    if not local.any():return finish('fixed_angle_violation')
    affected=np.any(local[triangles[contract['face_ids']]],axis=1)
    if np.any(bad&~affected):return finish('fixed_angle_violation')
    orientation_affected=np.any(local[triangles[orientation_ids]],axis=1)
    if np.any(bad_orientation&~orientation_affected):return finish('fixed_orientation_violation')
    orientation_contract=dict(face_ids=orientation_ids[orientation_affected],normals=orientation_normals[orientation_ids[orientation_affected]] if len(orientation_ids) else np.empty((0,3)),
        area_scale=orientation_scale[orientation_affected],minimum_cosine=np.full(int(orientation_affected.sum()),.1))
    local_contract={key:value[affected] for key,value in contract.items() if key in ('face_ids','normals','area_scale','minimum_cosine')}
    global_ids=np.flatnonzero(free);lookup=np.full(len(point),-1,int);lookup[global_ids]=np.arange(len(global_ids))
    dofs=(lookup[np.flatnonzero(local),None]*3+np.arange(3)).ravel()
    local_matrix=matrix[:,dofs].tocsr();active_rows=np.asarray(local_matrix.getnnz(axis=1)).ravel()>0
    source_mask=local[projector.ids];source_ids=projector.ids[source_mask];face_ids=projector.witness['source_triangle_ids'][source_mask].copy()
    def frozen_query(points):return witnesses(points,projector.source_vertices,projector.source_triangles,face_ids)
    local_projector=SourceFacetBoxes(projector.reference,projector.limits,source_ids,frozen_query,
        source_radius=np.broadcast_to(projector.source_radius,(len(projector.ids),))[source_mask],
        query_contract=dict(vertices=fingerprint(projector.source_vertices),triangles=fingerprint(projector.source_triangles)),
        source_vertices=projector.source_vertices,source_triangles=projector.source_triangles)
    local_projector.prepare(current)
    np.testing.assert_array_equal(local_projector.facets,projector.facets[source_mask])
    report.update(local_vertices=int(local.sum()),affected_angle_faces=int(affected.sum()),affected_orientation_faces=int(orientation_affected.sum()),local_source_constraints=len(source_ids))
    for _ in range(int(passes)):
        if time.monotonic()>=deadline:return finish('deadline')
        shift=(point-current)[free].ravel();remaining=bounds-matrix@shift
        if np.max(remaining[~active_rows],initial=0)>1e-9:return finish('fixed_separator_violation')
        model=dict(hessian=sparse.eye(len(dofs),format='csc'),gradient=(point-original)[local].ravel(),
                   soft_matrix=sparse.csr_matrix((0,len(dofs))),soft_bounds=np.empty(0),penalty=0.,
                   angle_cones=angular_model(point,triangles,local,local_contract))
        hard=local_matrix[active_rows];rhs=remaining[active_rows]
        if len(orientation_contract['face_ids']):
            oriented=angular_model(point,triangles,local,orientation_contract);count=len(orientation_contract['face_ids']);normals=orientation_contract['normals']
            dot=sparse.coo_matrix((normals.ravel(),(np.repeat(np.arange(count),3),np.arange(3*count))),shape=(count,3*count)).tocsr()
            hard=sparse.vstack((hard,dot@oriented['matrix']),format='csr')
            rhs=np.r_[rhs,orientation_floor[orientation_affected]-np.einsum('ij,ij->i',normals,oriented['offset'])]
        corrected,detail=project_joint_conic(point,point,local,hard,rhs,
            lower[dofs]-shift[dofs],upper[dofs]-shift[dofs],local_projector,deadline,model=model)
        history.append(detail)
        if not detail['valid']:return finish('local_projection_failed')
        point=corrected;residual,cosine_error=errors(point);orientation_error=orientation_errors(point)
        detail.update(nonlinear_residual=float(residual.max(initial=0)),nonlinear_cosine_error=float(cosine_error.max(initial=0)),nonlinear_orientation_residual=float(orientation_error.max(initial=0)))
        if residual.max(initial=0)<=1e-9 and cosine_error.max(initial=0)<=1e-7 and orientation_error.max(initial=0)<=1e-9:
            displacement=(point-current)[free].ravel()
            hard=float(np.maximum(bounds-matrix@displacement,0.).max(initial=0))
            box=float(max(np.max(lower-displacement,initial=0),np.max(displacement-upper,initial=0)))
            source=float(np.max(projector.source_distance(point)-projector.source_radius,initial=0))
            valid=hard<=1e-9 and box<=1e-10 and source<=1e-10 and np.array_equal(point[~local],original[~local])
            report.update(final_residual=float(residual.max(initial=0)),final_orientation_residual=float(orientation_error.max(initial=0)),hard_residual=hard,box_residual=box,source_residual=source,untouched_coordinates_exact=bool(np.array_equal(point[~local],original[~local])))
            return finish('restored' if valid else 'global_envelope_violation',valid)
    return finish('pass_limit')
