"""Constrained canonical bending and original-source redistribution.

Observed positions are immutable by default. With an explicit source
projector they can slide within their original source-distance limits.
Pinned coordinates always remain exact. The inferred join is not source
evidence; the parent must validate coverage and native storage.
"""
import numpy as np
from scipy import sparse
from .vertex_contract import witnesses, fingerprint
from .source_envelope import SourceFacetBoxes
from .surface_alignment import angle_contract
from .constrained_surface_fit import solve


def complete(points,triangles,observed,pinned,metric,box_origin,box_limits,normals,*,source_vertices,source_triangles,supported_faces=None,time_limit=30.,iterations=16,progress=None,cancelled=None,source_projector=None,differential_target=None,differential_confidence=None,positional_objective=None):
    import igl
    from scipy.sparse.csgraph import connected_components
    points=np.asarray(points,dtype=float);tri=np.asarray(triangles,dtype=np.int64);metric=np.asarray(metric,dtype=float)
    observed=np.asarray(observed,dtype=bool);pinned=np.asarray(pinned,dtype=bool);n=len(points)
    origin=np.asarray(box_origin,dtype=float).copy();limits=np.asarray(box_limits,dtype=float).copy();normals=np.asarray(normals,dtype=float)
    if points.shape!=(n,3) or metric.shape!=points.shape or observed.shape!=(n,) or pinned.shape!=(n,) or origin.shape!=points.shape or limits.shape!=points.shape or not all(np.isfinite(v).all() for v in (points,metric,origin,limits,normals)) or np.any(limits<0):raise ValueError('Invalid inferred completion snapshot')
    if tri.ndim!=2 or tri.shape[1]!=3 or not len(tri) or tri.min()<0 or tri.max()>=n or normals.shape!=(len(tri),3):raise ValueError('Invalid inferred completion triangles')
    sliding=source_projector is not None
    free=~pinned if sliding else ~(observed|pinned)
    report=dict(method='SOURCE_CONSTRAINED_CANONICAL_REDISTRIBUTION_V1' if sliding else 'JOINT_CANONICAL_UNSUPPORTED_BENDING_V1',inferred_vertices=int((free&~observed).sum()),source_sliding_vertices=int((free&observed).sum()),inferred_shape_is_not_source_evidence=True)
    if not free.any():return points.copy(),dict(report,accepted=False,reason='no_unsupported_vertices')
    if np.any(np.abs(points[free]-origin[free])>limits[free]+1e-12):raise ValueError('Inferred completion starts outside the original artist boxes')
    origin[~free]=points[~free];limits[~free]=0.
    p=metric[tri];area=np.linalg.norm(np.cross(p[:,1]-p[:,0],p[:,2]-p[:,0]),axis=1)*.5;mass=np.zeros(n)
    for corner in range(3):np.add.at(mass,tri[:,corner],area/3)
    if np.any(mass[free]<=0):raise ValueError('Inferred completion requires nondegenerate canonical area')
    lap=igl.cotmatrix(metric,tri).tocsr()
    if not np.isfinite(lap.data).all():raise ValueError('Invalid canonical completion metric')
    adjacency=lap.copy();adjacency.setdiag(0);adjacency.eliminate_zeros();_,labels=connected_components(adjacency,directed=False)
    if np.any(~np.isin(labels[free],np.unique(labels[~free]))):raise ValueError('Every inferred component needs a source-observed or pinned boundary')
    inverse=np.divide(1.,mass,out=np.zeros(n),where=mass>0)
    active_rows=np.asarray(lap[:,free].getnnz(axis=1)).ravel()>0;stencil=lap[active_rows]
    desired=np.zeros_like(points)
    if (differential_target is None)!=(differential_confidence is None):raise ValueError('Provide both source curvature targets and confidence')
    if differential_target is not None:
        target=np.asarray(differential_target,dtype=float);confidence=np.asarray(differential_confidence,dtype=float)
        if target.shape!=points.shape or confidence.shape!=(n,) or not np.isfinite(target).all() or not np.isfinite(confidence).all() or np.any((confidence<0)|(confidence>1)):raise ValueError('Invalid source curvature completion target')
        desired=target*confidence[:,None]
        report.update(source_curvature_vertices=int(np.count_nonzero(confidence)),source_curvature_is_source_derived_not_inferred_gap_bending=True)
    bending=(stencil.T@sparse.diags(inverse[active_rows])@stencil).tocsr()
    normalization=max(float(bending.diagonal()[free].max()),1e-30);weight=float(mass[mass>0].mean())
    objective=dict(matrix=bending*(weight/normalization),load=np.asarray(stencil.T@(inverse[active_rows,None]*desired[active_rows]))*(weight/normalization),mass=mass,
        constant=float(np.sum(inverse[active_rows,None]*desired[active_rows]**2))*(weight/normalization))
    if positional_objective is not None:
        extra=positional_objective
        if extra['matrix'].shape!=(n,n) or np.shape(extra['load'])!=(n,3) or np.shape(extra['mass'])!=(n,) or not np.isfinite(extra['matrix'].data).all() or not np.isfinite(extra['load']).all() or not np.isfinite(extra['mass']).all() or np.any(extra['mass']<0) or not np.isfinite(extra['constant']):raise ValueError('Invalid positional source objective')
        objective=dict(matrix=objective['matrix']+extra['matrix'],load=objective['load']+extra['load'],mass=objective['mass']+extra['mass'],constant=objective['constant']+float(extra['constant']))
        report['positional_source_objective_included']=True
    supported=np.zeros(len(tri),bool) if supported_faces is None else np.asarray(supported_faces,dtype=bool).copy()
    if supported.shape!=(len(tri),):raise ValueError('Invalid completion source-normal scope')
    supported&=np.any(free[tri],axis=1)
    angles=angle_contract(points,tri,normals,supported,metric)
    # The default gap phase has no movable source positions. Optional source
    # sliding requires the original positional constraints for every movable
    # observation; inferred positions never acquire invented witnesses.
    source=np.asarray(source_vertices,dtype=float);st=np.asarray(source_triangles,dtype=np.int64);empty=np.empty(0,dtype=np.int64)
    def query(query_points):return witnesses(query_points,source,st,empty)
    projector=source_projector if sliding else SourceFacetBoxes(points,limits,empty,query,source_radius=np.empty(0),
        query_contract=dict(vertices=fingerprint(source),triangles=fingerprint(st)),source_vertices=source,source_triangles=st)
    if sliding:
        if not np.array_equal(np.sort(projector.ids),np.flatnonzero(observed&free)):raise ValueError('Sliding requires the complete movable original observed-source constraint scope')
        projector.prepare(points)
    p=points[tri];current_area=np.linalg.norm(np.cross(p[:,1]-p[:,0],p[:,2]-p[:,0]),axis=1);span=float(np.linalg.norm(np.ptp(points,axis=0)))
    def checkpoint(_,detail):
        if cancelled and cancelled():raise RuntimeError('Inferred completion cancelled')
        if progress:progress('Completing unsupported shape',min(.95,len(detail.get('history',[]))/max(int(iterations),1)))
    if cancelled and cancelled():raise RuntimeError('Inferred completion cancelled')
    result,numerical=solve(points,points,tri,objective,free,normals,np.maximum(current_area*.1,span**2*1e-12),np.zeros(n),lap,
        iterations=int(iterations),time_limit=float(time_limit),checkpoint=checkpoint,prevent_new_contacts=True,movement_boxes=limits,movement_box_origin=origin,
        source_projector=projector,contact_margin=span*1e-6,differential_target=np.zeros_like(points),differential_weight=np.zeros(n),
        joint_projection='conic',preserve_orientations=True,compact_proposal=True,conic_segment_backtracking=True,conic_elastic_model=True,
        localize_contact_stalls=True,source_angles=angles,restore_source_angles=True,step_size_fraction=.01)
    np.testing.assert_array_equal(result[~free],points[~free])
    if np.any(np.abs(result[free]-origin[free])>limits[free]+1e-10):raise ValueError('Inferred completion exceeds its original artist boxes')
    def energy(value):return float(np.sum((stencil@value-desired[active_rows])**2*inverse[active_rows,None]))
    before,after=energy(points),energy(result);accepted=after<before-max(abs(before)*1e-10,1e-12)
    if positional_objective is not None:
        def combined(value):return energy(value)*(weight/normalization)+float(np.sum(value*(extra['matrix']@value))-2*np.sum(extra['load']*value)+extra['constant'])
        cb,ca=combined(points),combined(result);accepted=ca<cb-max(abs(cb)*1e-10,1e-15);report.update(before_combined_energy=cb,after_combined_energy=ca)
    if not accepted:result=points.copy()
    # The shared solver's quadratic RMS is bending energy here, not a
    # measurement of source fit. Keep that distinction explicit in reports.
    numerical['objective_kind']='SOURCE_CURVATURE_AND_INFERRED_BENDING_NOT_SOURCE_DISTANCE' if differential_target is not None else 'CANONICAL_BENDING_NOT_SOURCE_DISTANCE'
    if positional_objective is not None:numerical['objective_kind']='JOINT_SOURCE_POSITION_CURVATURE_AND_INFERRED_BENDING'
    report.update(accepted=accepted,before_bending_energy=before,after_bending_energy=after,observed_coordinates_exact=not sliding,pinned_coordinates_exact=True,source_distance_bounds_unchanged=True,
        max_movement_m=float(np.linalg.norm(result-points,axis=1).max(initial=0)),source_angle_faces=len(angles['face_ids']),numerical=numerical)
    return result,report
