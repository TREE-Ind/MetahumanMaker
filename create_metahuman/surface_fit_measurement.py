"""Independent area-weighted, two-way surface measurements for a worker.

Requires optional libigl in the external environment. Source-to-target
coverage uses fixed original-source triangle quadrature, not chart identity
error. Measurements do not certify semantic correspondence or injectivity.
"""
import numpy as np
from .vertex_contract import fingerprint


class UnsupportedSurfaceCoverage(ValueError):
    """A local patch has no independently measurable reverse coverage."""


def barycentric_quadratic(vertices,physical,bary,weight,samples,body_count):
    """Quadratic observations with companion contributions held constant."""
    from scipy import sparse
    rows=np.repeat(np.arange(len(samples)),3);columns=physical.ravel();values=bary.ravel();body=columns<body_count
    B=sparse.coo_matrix((values[body],(rows[body],columns[body])),shape=(len(samples),body_count)).tocsr()
    fixed=np.zeros_like(samples)
    for corner in range(3):
        mask=physical[:,corner]>=body_count
        fixed[mask]+=bary[mask,corner,None]*vertices[physical[mask,corner]]
    target=samples-fixed;weighted=sparse.diags(weight)
    return dict(matrix=(B.T@weighted@B).tocsr(),load=B.T@(weight[:,None]*target),mass=np.asarray(B.T@weight).ravel(),constant=float(np.sum(weight[:,None]*target**2)))


def measure_surface_fit(before,after,triangles,source,source_triangles,movable,source_face_ids,*,companion_vertices=None,companion_triangles=None,forward_vertices=None,reverse_source_face_ids=None):
    import igl
    before=np.asarray(before,dtype=float);after=np.asarray(after,dtype=float);triangles=np.asarray(triangles);source=np.asarray(source,dtype=float);source_triangles=np.asarray(source_triangles);movable=np.asarray(movable,dtype=bool);source_ids=np.asarray(source_face_ids)
    if before.ndim!=2 or before.shape[1]!=3 or after.shape!=before.shape or movable.shape!=(len(before),) or not movable.any() or not np.isfinite(before).all() or not np.isfinite(after).all():raise ValueError('Invalid surface fit arrays')
    if triangles.dtype.kind not in 'iu' or triangles.ndim!=2 or triangles.shape[1]!=3 or triangles.min()<0 or triangles.max()>=len(before):raise ValueError('Invalid target triangles')
    if source.ndim!=2 or source.shape[1]!=3 or not np.isfinite(source).all() or source_triangles.dtype.kind not in 'iu' or source_triangles.ndim!=2 or source_triangles.shape[1]!=3 or source_triangles.min()<0 or source_triangles.max()>=len(source):raise ValueError('Invalid original source')
    if source_ids.dtype.kind not in 'iu' or source_ids.ndim!=1 or not len(source_ids) or source_ids.min()<0 or source_ids.max()>=len(source_triangles) or len(np.unique(source_ids))!=len(source_ids):raise ValueError('Choose unique original source facets')
    fixed_reverse=None if reverse_source_face_ids is None else np.asarray(reverse_source_face_ids)
    if fixed_reverse is not None and (fixed_reverse.dtype.kind not in 'iu' or fixed_reverse.ndim!=1 or not len(fixed_reverse) or len(np.unique(fixed_reverse))!=len(fixed_reverse) or not np.isin(fixed_reverse,source_ids).all()):raise ValueError('Frozen reverse scope must name unique facets within the original source policy')
    if not np.array_equal(before[~movable],after[~movable]):raise ValueError('Measurement scope excludes moved vertices')
    forward_mask=movable if forward_vertices is None else np.asarray(forward_vertices,dtype=bool)
    if forward_mask.shape!=movable.shape or not forward_mask.any() or np.any(forward_mask&~movable):raise ValueError('Forward observations must be a nonempty subset of movable vertices')
    target_before=before;target_after=after;target_tri=triangles
    active=np.any(movable[triangles],axis=1)
    if companion_vertices is not None:
        cv=np.asarray(companion_vertices,dtype=float);ct=np.asarray(companion_triangles)
        if cv.ndim!=2 or cv.shape[1]!=3 or not np.isfinite(cv).all() or ct.dtype.kind not in 'iu' or ct.ndim!=2 or ct.shape[1]!=3 or ct.min()<0 or ct.max()>=len(cv):raise ValueError('Invalid fixed companion surface')
        target_before=np.vstack((before,cv));target_after=np.vstack((after,cv));target_tri=np.vstack((triangles,ct+len(before)));active=np.r_[active,np.zeros(len(ct),bool)]
    elif companion_triangles is not None:raise ValueError('Companion triangles require their vertices')
    trees=[]
    def tree(vertices,faces):
        value=igl.AABB();value.init(vertices,faces);trees.append(value);return value
    def stats(distance,weight):
        return dict(rms_m=float(np.sqrt(np.average(distance**2,weights=weight))),max_m=float(distance.max()))
    try:
        source_faces=source_triangles[source_ids];original=tree(source,source_faces)
        p=before[triangles];area2=np.linalg.norm(np.cross(p[:,1]-p[:,0],p[:,2]-p[:,0]),axis=1);mass=np.zeros(len(before))
        for k in range(3):np.add.at(mass,triangles[:,k],area2/6)
        weight=mass[forward_mask]
        if weight.sum()<=0:raise ValueError('Movable target surface has no area')
        forward=[np.sqrt(np.maximum(original.squared_distance(source,source_faces,points[forward_mask])[0],0)) for points in (before,after)]
        old_tree=tree(target_before,target_tri);new_tree=tree(target_after,target_tri)
        facets=source[source_faces]
        if fixed_reverse is None:
            centroids=facets.mean(1);_,nearest,_=old_tree.squared_distance(target_before,target_tri,centroids);selected=active[nearest]
        else:selected=np.isin(source_ids,fixed_reverse)
        eligible=source_ids[selected]
        if not len(eligible):raise UnsupportedSurfaceCoverage('No original source surface is covered by this active target region')
        facets=facets[selected];area=np.linalg.norm(np.cross(facets[:,1]-facets[:,0],facets[:,2]-facets[:,0]),axis=1)*.5
        bary=np.array([[2/3,1/6,1/6],[1/6,2/3,1/6],[1/6,1/6,2/3]])
        samples=np.einsum('ki,nij->nkj',bary,facets).reshape(-1,3);weights=np.repeat(area/3,3)
        reverse=[np.sqrt(np.maximum(value.squared_distance(vertices,target_tri,samples)[0],0)) for value,vertices in ((old_tree,target_before),(new_tree,target_after))]
        contract=dict(before=fingerprint(before),after=fingerprint(after),triangles=fingerprint(triangles),movable=fingerprint(movable),forward_vertices=fingerprint(forward_mask),source=fingerprint(source),source_triangles=fingerprint(source_triangles),source_face_ids=fingerprint(source_ids),measured_source_face_ids=fingerprint(eligible))
        if companion_vertices is not None:contract.update(companion_vertices=fingerprint(cv),companion_triangles=fingerprint(ct))
        return dict(method='FIXED_SOURCE_AREA_QUADRATURE_BIDIRECTIONAL_V1' if forward_vertices is None else 'PARTIAL_OBSERVATION_FIXED_SOURCE_AREA_QUADRATURE_V1',forward_scope='MOVABLE_VERTICES' if forward_vertices is None else 'EXPLICIT_OBSERVED_VERTICES',reverse_scope='BASELINE_ACTIVE_TARGET_COVERAGE' if fixed_reverse is None else 'EXPLICIT_ORIGINAL_SOURCE_FACETS',forward_before=stats(forward[0],weight),forward_after=stats(forward[1],weight),reverse_before=stats(reverse[0],weights),reverse_after=stats(reverse[1],weights),source_area_m2=float(area.sum()),source_faces=len(eligible),source_samples=len(samples),measured_source_face_ids=eligible,contract=contract)
    finally:trees.clear()


def assemble_reverse_projection(before,triangles,source,source_triangles,movable,source_face_ids,*,companion_vertices=None,companion_triangles=None,observation_mask=None,patch_ids=None,source_policies=None,allow_inferred_surface=False,reverse_source_face_ids=None,coverage_metric=None):
    """Area-balanced source-to-target barycentric observations.

    Target barycentrics are fixed at the initial closest surface points.
    The source quadrature and its active scope match measure_surface_fit.
    A fixed companion contributes constants, never movable unknowns.
    Optional partial-observation loads require fully observed, coherently
    facing triangles whose corner policies all permit the source facet. Unsupported rows
    retain zero weight; they remain in independent coverage measurements.
    Explicit inferred-surface mode can load a paired, facing surface sample
    through unknown target vertices. It never relabels those vertices as
    position observations; original distance and endpoint gates still apply.
    """
    import igl
    from scipy import sparse
    from .vertex_contract import closest_triangle_points
    if allow_inferred_surface and observation_mask is None:raise ValueError('Inferred surface observations require explicit paired source policies')
    measured=measure_surface_fit(before,before,triangles,source,source_triangles,movable,source_face_ids,companion_vertices=companion_vertices,companion_triangles=companion_triangles,reverse_source_face_ids=reverse_source_face_ids)
    before=np.asarray(before,dtype=float);triangles=np.asarray(triangles);source=np.asarray(source,dtype=float);source_triangles=np.asarray(source_triangles)
    selected=measured['measured_source_face_ids'];facets=source[source_triangles[selected]]
    area=np.linalg.norm(np.cross(facets[:,1]-facets[:,0],facets[:,2]-facets[:,0]),axis=1)*.5
    quadrature=np.array([[2/3,1/6,1/6],[1/6,2/3,1/6],[1/6,1/6,2/3]])
    samples=np.einsum('ki,nij->nkj',quadrature,facets).reshape(-1,3);weight=np.repeat(area/3,3)
    vertices=before;faces=triangles
    if companion_vertices is not None:
        vertices=np.vstack((before,companion_vertices));faces=np.vstack((triangles,np.asarray(companion_triangles)+len(before)))
    tree=igl.AABB();tree.init(vertices,faces)
    try:distance,face_ids,_=tree.squared_distance(vertices,faces,samples)
    finally:del tree
    physical=faces[face_ids];points,bary,residual=closest_triangle_points(samples,vertices[physical])
    np.testing.assert_allclose(residual**2,distance,rtol=1e-8,atol=1e-14)
    partial_report={}
    if observation_mask is not None:
        observed=np.asarray(observation_mask,dtype=bool);patch=np.asarray(patch_ids)
        if observed.shape!=(len(before),) or patch.shape!=observed.shape or patch.dtype.kind not in 'iu' or not isinstance(source_policies,dict) or (not allow_inferred_surface and np.any(observed&~np.asarray(movable,dtype=bool))) or np.any(observed&(patch<0)):raise ValueError('Partial reverse projection needs observed movable vertices and paired source policies')
        supported=np.all(physical<len(before),axis=1);indices=np.flatnonzero(supported)
        vertex_supported=np.any(np.asarray(movable,dtype=bool)[physical[indices]],axis=1) if allow_inferred_surface else np.all(observed[physical[indices]],axis=1)
        supported[indices]&=vertex_supported
        observed_rows=supported.copy();indices=np.flatnonzero(supported)
        owners=patch[physical[indices]];same_owner=np.zeros(len(samples),bool);same_owner[indices]=np.all(owners==owners[:,0,None],axis=1)&(owners[:,0]>=0)
        supported[indices]&=np.all(owners>=0,axis=1);indices=np.flatnonzero(supported)
        owners=patch[physical[indices]];source_rows=np.repeat(selected,3)
        # Different labels need not imply disjoint authority. A shared-joint
        # triangle is eligible only when every corner's original policy
        # permits this exact source facet. No policy or evidence mask expands.
        for owner in np.unique(owners):
            rows=indices[np.any(owners==owner,axis=1)]
            if int(owner) not in source_policies:raise ValueError('Missing source policy for partial reverse projection')
            policy=np.asarray(source_policies[int(owner)])
            if policy.dtype.kind not in 'iu' or policy.ndim!=1 or not len(policy) or policy.min()<0 or policy.max()>=len(source_triangles):raise ValueError('Invalid source policy for partial reverse projection')
            supported[rows]&=np.isin(source_rows[rows],policy)
        paired_rows=supported.copy()
        # A reversed target sheet is not a reliable reverse correspondence.
        # Its reconstruction is still guided by source observations and the
        # alignment objective, without anchoring unrelated skin to the fold.
        t=vertices[physical];normal=np.cross(t[:,1]-t[:,0],t[:,2]-t[:,0])
        s=np.repeat(np.cross(facets[:,1]-facets[:,0],facets[:,2]-facets[:,0]),3,axis=0)
        cosine=np.einsum('ij,ij->i',normal,s)/np.maximum(np.linalg.norm(normal,axis=1)*np.linalg.norm(s,axis=1),1e-30)
        supported&=cosine>=.25
        weight[~supported]=0
        partial_report=dict(method='POLICY_INTERSECTION_FACING_REVERSE_PROJECTION_V2',supported_samples=int(supported.sum()),
            excluded_unobserved_samples=int((~observed_rows).sum()),excluded_policy_samples=int((observed_rows&~paired_rows).sum()),
            excluded_facing_samples=int((paired_rows&~supported).sum()),fit_source_area_m2=float(weight.sum()),
            inferred_vertices_loaded=bool(allow_inferred_surface and np.any(supported&~np.all(np.r_[observed,False][np.minimum(physical,len(before))],axis=1))),
            surface_observations_do_not_relabel_vertex_evidence=True,measurement_scope_unchanged=True,
            source_policy_test='ORIGINAL_SOURCE_FACET_INTERSECTION_AT_ALL_CORNERS',source_policies_unchanged=True,mixed_policy_samples=int((supported&~same_owner).sum()),mixed_policy_source_area_m2=float(weight[supported&~same_owner].sum()))
        if allow_inferred_surface:
            partial_report['method']='POLICY_INTERSECTION_SURFACE_OBSERVATIONS_V2'
            partial_report['excluded_nonmoving_or_companion_samples']=partial_report.pop('excluded_unobserved_samples')
    elif patch_ids is not None or source_policies is not None:raise ValueError('Paired reverse projection requires an observation mask')
    normalization={}
    if coverage_metric is not None:
        # Match the length-squared units of the canonical tangent objectives.
        # Raw area quadrature times squared offset has length-fourth units and
        # can be overwhelmed by the normal/distribution priors at fine sampling.
        # Only proposal weights change; independent source-area measurements,
        # source policies and every distance constraint remain unmodified.
        metric=np.asarray(coverage_metric,dtype=float)
        if metric.shape!=before.shape or not np.isfinite(metric).all():raise ValueError('Invalid reverse coverage metric')
        p=metric[triangles];square=sum(np.sum((p[:,a]-p[:,b])**2,axis=1) for a,b in ((0,1),(1,2),(2,0)))/3
        if not np.isfinite(square).all() or np.any(square<=0):raise ValueError('Coverage metric requires nonzero triangle edge scale')
        target_body=face_ids<len(triangles);measured_rows=target_body.copy()
        if observation_mask is not None:
            measured_rows[target_body]&=np.all(observed[physical[target_body]],axis=1)
        scale=np.ones(len(samples));scale[measured_rows]=1/square[face_ids[measured_rows]]
        active=weight>0;weight*=scale
        normalization=dict(method='MEASURED_CANONICAL_MEAN_EDGE_SQUARED_REVERSE_COVERAGE_V2',metric=fingerprint(metric),source_area_measurement_unchanged=True,source_quadrature_unchanged=True,proposal_only=True,scaled_samples=int(np.count_nonzero(active&measured_rows)),inferred_sample_weights_unchanged=True,unscaled_inferred_samples=int(np.count_nonzero(active&target_body&~measured_rows)),minimum_scale=float(scale[active&measured_rows].min()) if np.any(active&measured_rows) else 0.,maximum_scale=float(scale[active&measured_rows].max(initial=0)))
    quadratic=barycentric_quadratic(vertices,physical,bary,weight,samples,len(before))
    return dict(**quadratic,source_area=float(area.sum()),
        contract=measured['contract'],source_face_ids=selected,target_triangle_ids=face_ids,target_barycentric=bary,sample_weights=weight,source_samples=samples,
        report=dict(method='FIXED_SOURCE_AREA_REVERSE_BARYCENTRIC_PROJECTION_V1',source_faces=len(selected),source_samples=len(samples),source_area_m2=float(area.sum()),initial_reverse_rms_m=measured['reverse_before']['rms_m'],source_mutated=False,partial_observations=partial_report,
            source_scope_fingerprint=fingerprint(selected),target_triangle_fingerprint=fingerprint(face_ids),target_barycentric_fingerprint=fingerprint(bary),reverse_scope=measured['reverse_scope'],coverage_normalization=normalization))
