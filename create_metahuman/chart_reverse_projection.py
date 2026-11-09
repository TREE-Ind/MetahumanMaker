"""Transport fixed original-source quadrature through an oriented repair chart."""
import numpy as np
from .vertex_contract import fingerprint,closest_triangle_points
from .surface_fit_measurement import barycentric_quadratic


def assemble(before,triangles,source,source_triangles,movable,chart,baseline,*,companion_vertices=None,companion_triangles=None):
    """Change proposal correspondence only; retain independent coverage scope.

    Source samples inside the verified target chart use its inverse mapping.
    Outside rows retain their original closest-surface observations. Source
    positions, quadrature weights, policies and acceptance measurements do not
    change. Fixed companion observations never become movable vertices.
    """
    import igl
    before=np.asarray(before,dtype=float);triangles=np.asarray(triangles)
    source=np.asarray(source,dtype=float);source_triangles=np.asarray(source_triangles)
    selected=baseline['source_face_ids'];samples=baseline['source_samples'];weight=baseline['sample_weights']
    quadrature=np.array([[2/3,1/6,1/6],[1/6,2/3,1/6],[1/6,1/6,2/3]])
    expected=np.einsum('ki,nij->nkj',quadrature,source[source_triangles[selected]]).reshape(-1,3)
    if not np.array_equal(samples,expected):raise ValueError('Chart transport requires unchanged original-source quadrature')
    source_ids=chart['source_face_ids'];source_uv=chart['source_uv'];source_faces=chart['source_chart_triangles']
    target_uv=chart['target_uv'];target_faces=chart['target_chart_triangles'];target_ids=chart['target_vertex_ids'];target_face_ids=chart['target_face_ids']
    if len(np.unique(source_ids))!=len(source_ids) or len(source_faces)!=len(source_ids):raise ValueError('Source chart loses original facet identity')
    if not np.array_equal(target_ids[target_faces],triangles[target_face_ids]):raise ValueError('Target chart loses canonical face-corner identity')
    lookup={int(face):index for index,face in enumerate(source_ids)}
    source_rows=np.flatnonzero(np.isin(selected,source_ids));rows=(source_rows[:,None]*3+np.arange(3)).ravel()
    face_ids=baseline['target_triangle_ids'].copy();bary=baseline['target_barycentric'].copy()
    fixed_companion=face_ids>=len(triangles)
    transferred=np.zeros(len(samples),bool)
    if len(rows):
        local=np.array([lookup[int(face)] for face in selected[source_rows]],dtype=np.int64)
        uv_samples=np.einsum('ki,nij->nkj',quadrature,source_uv[source_faces[local]]).reshape(-1,2)
        uv3=np.column_stack((target_uv,np.zeros(len(target_uv))));samples3=np.column_stack((uv_samples,np.zeros(len(uv_samples))))
        tree=igl.AABB();tree.init(uv3,target_faces)
        try:distance,which,_=tree.squared_distance(uv3,target_faces,samples3)
        finally:del tree
        _,inverse_bary,residual=closest_triangle_points(samples3,uv3[target_faces[which]])
        inside=(distance<=1e-18)&(residual<=1e-9)&~fixed_companion[rows]&np.any(np.asarray(movable,dtype=bool)[target_ids[target_faces[which]]],axis=1)
        if not np.isfinite(inverse_bary).all() or np.any(inverse_bary<-1e-10) or not np.allclose(inverse_bary.sum(1),1.,atol=1e-12,rtol=0):raise ValueError('Invalid inverse repair-chart barycentrics')
        use=rows[inside];face_ids[use]=target_face_ids[which[inside]];bary[use]=inverse_bary[inside];transferred[use]=True
    vertices=before;faces=triangles
    if companion_vertices is not None:
        vertices=np.vstack((before,companion_vertices));faces=np.vstack((triangles,np.asarray(companion_triangles)+len(before)))
    physical=faces[face_ids]
    if np.any(transferred&np.any(physical>=len(before),axis=1)):raise ValueError('Chart transport changes a fixed companion observation')
    if not np.array_equal(face_ids[fixed_companion],baseline['target_triangle_ids'][fixed_companion]) or not np.array_equal(bary[fixed_companion],baseline['target_barycentric'][fixed_companion]):raise ValueError('Chart transport reassigns an original fixed companion observation')
    quadratic=barycentric_quadratic(vertices,physical,bary,weight,samples,len(before))
    report=dict(baseline['report']);report.update(method='ORIGINAL_SOURCE_CHART_INVERSE_REVERSE_PROJECTION_V1',
        chart_transferred_samples=int(transferred.sum()),nearest_fallback_samples=int((~transferred).sum()),
        transferred_source_area_m2=float(weight[transferred].sum()),source_quadrature_exact=True,source_weights_exact=True,
        fixed_companion_samples=int(fixed_companion.sum()),fixed_companion_bindings_exact=True,
        independent_measurement_scope_unchanged=True,proposal_only=True,target_triangle_fingerprint=fingerprint(face_ids),target_barycentric_fingerprint=fingerprint(bary))
    return dict(baseline,**quadratic,target_triangle_ids=face_ids,target_barycentric=bary,report=report)
