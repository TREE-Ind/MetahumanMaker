"""Inspect source authority without classifying or discarding small patches."""
import numpy as np


def components(points, triangles, polygon_ids, selected, *, stitch_exact_seams=True):
    """Return stable, edge-connected components of an exact triangle policy.

    Exact-coordinate seam matching is optional and changes only this graph.
    Nearby cloth and skin never join through a proximity tolerance.
    Component IDs are their smallest original polygon ID. No size implies
    validity, confidence, semantic class or permission to delete a component.
    """
    points=np.asarray(points,dtype=float);tri=np.asarray(triangles);polys=np.asarray(polygon_ids);selected=np.asarray(selected)
    if points.ndim!=2 or points.shape[1]!=3 or not np.isfinite(points).all():raise ValueError('Invalid source coordinates')
    if tri.ndim!=2 or tri.shape[1]!=3 or tri.dtype.kind not in 'iu' or not len(tri) or tri.min()<0 or tri.max()>=len(points):raise ValueError('Invalid source triangles')
    if polys.shape!=(len(tri),) or polys.dtype.kind not in 'iu' or polys.min()<0 or selected.shape!=(len(tri),) or selected.dtype.kind!='b':raise ValueError('Invalid source policy')
    ids=np.flatnonzero(selected)
    if not len(ids):raise ValueError('Source region has no complete permitted faces')
    # A polygon cannot be partly permitted: Blender selects original polygons,
    # and all triangles belonging to one polygon share the same authority.
    if np.any(np.isin(polys[~selected],polys[selected])):raise ValueError('Source policy splits an original polygon')
    faces=tri[ids];parent=np.arange(len(ids))
    def root(i):
        while parent[i]!=i:parent[i]=parent[parent[i]];i=parent[i]
        return i
    def join(a,b):
        a=root(a);b=root(b)
        if a!=b:parent[max(a,b)]=min(a,b)
    graph_faces=faces
    seam_copies=0
    if stitch_exact_seams:
        unique,inverse=np.unique(points,axis=0,return_inverse=True);graph_faces=inverse[faces];seam_copies=len(points)-len(unique)
    edges=np.sort(np.concatenate((graph_faces[:,[0,1]],graph_faces[:,[1,2]],graph_faces[:,[2,0]])),axis=1);owners=np.tile(np.arange(len(ids)),3)
    order=np.lexsort((edges[:,1],edges[:,0]));edges=edges[order];owners=owners[order]
    shared=np.flatnonzero(np.all(edges[1:]==edges[:-1],axis=1))
    for i in shared:join(int(owners[i]),int(owners[i+1]))
    # Triangulation diagonals normally connect polygon triangles. Explicitly
    # join a polygon too, including pathological degenerate triangulations.
    order=np.argsort(polys[ids],kind='stable');same=np.flatnonzero(polys[ids][order][1:]==polys[ids][order][:-1])
    for i in same:join(int(order[i]),int(order[i+1]))
    roots=np.array([root(i) for i in range(len(ids))]);p=points[faces];area=np.linalg.norm(np.cross(p[:,1]-p[:,0],p[:,2]-p[:,0]),axis=1)*.5
    rows=[]
    for label in np.unique(roots):
        mask=roots==label;polygon=np.unique(polys[ids[mask]]);coords=p[mask].reshape(-1,3)
        rows.append(dict(component_id=int(polygon.min()),polygon_ids=polygon.tolist(),polygons=len(polygon),triangles=int(mask.sum()),area_m2=float(area[mask].sum()),bounds_min_m=coords.min(0).tolist(),bounds_max_m=coords.max(0).tolist()))
    rows.sort(key=lambda r:(-r['area_m2'],r['component_id']))
    largest=rows[0]['area_m2']
    for row in rows:row['area_fraction_of_largest']=row['area_m2']/largest if largest>0 else 0.
    return dict(method='EXACT_POLICY_EDGE_CONNECTED_COMPONENT_REVIEW_V1',components=rows,component_count=len(rows),source_triangles=int(selected.sum()),source_polygons=len(np.unique(polys[ids])),source_area_m2=float(area.sum()),small_components_are_not_invalid=True,source_authority_changed=False,connectivity='EXACT_COORDINATE_EDGE_OR_ORIGINAL_POLYGON' if stitch_exact_seams else 'SHARED_INDEXED_EDGE_OR_ORIGINAL_POLYGON',exact_seam_copies=seam_copies,mesh_welded=False)
