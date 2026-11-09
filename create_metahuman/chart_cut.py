"""Witness-avoiding cuts in temporary source charts; original facets stay exact."""
import numpy as np
from scipy import sparse
from scipy.sparse.csgraph import connected_components,dijkstra


def open_boundaries(vertices,faces,protected_faces):
    """Open a connected genus-zero patch along a boundary-connecting edge tree.

    No source face is removed. Duplicated chart vertices retain an exact map
    to their original positions and ordered face corners. The caller must
    independently verify disk topology and the final source correspondence.
    """
    import igl
    vertices=np.asarray(vertices,dtype=float);faces=np.asarray(faces,dtype=np.int64)
    protected_faces=np.asarray(protected_faces,dtype=bool)
    if protected_faces.shape!=(len(faces),):raise ValueError('Chart protection must identify original source faces')
    half=np.stack((faces,faces[:,[1,2,0]]),axis=-1)
    edges,inverse,count=np.unique(np.sort(half.reshape(-1,2),axis=1),axis=0,return_inverse=True,return_counts=True)
    if np.any(count>2):raise ValueError('Cannot chart-cut a nonmanifold source patch')
    boundary=edges[count==1];ids=np.unique(boundary)
    if not len(ids):raise ValueError('Source chart has no open boundary')
    graph=sparse.coo_matrix((np.ones(len(boundary)*2),(np.r_[boundary[:,0],boundary[:,1]],np.r_[boundary[:,1],boundary[:,0]])),shape=(len(vertices),len(vertices))).tocsr()
    if np.any(np.asarray(graph.sum(1)).ravel()[ids]!=2):raise ValueError('Source chart boundaries are not disjoint simple loops')
    number,labels=connected_components(graph[ids][:,ids],directed=False)
    if not 2<=number<=4 or len(vertices)-len(edges)+len(faces)!=2-number:raise ValueError('Chart cuts support two to four genus-zero boundary loops')
    full=sparse.coo_matrix((np.ones(len(edges)*2),(np.r_[edges[:,0],edges[:,1]],np.r_[edges[:,1],edges[:,0]])),shape=graph.shape).tocsr()
    if connected_components(full,directed=False,return_labels=False)!=1:raise ValueError('Source chart patch is disconnected')
    loops=sorted((ids[labels==index] for index in range(number)),key=len,reverse=True)
    protected=np.zeros(len(vertices),bool);protected[np.unique(faces[protected_faces])]=True
    usable=~np.any(protected[edges],axis=1)
    safe=edges[usable];cost=np.linalg.norm(vertices[safe[:,0]]-vertices[safe[:,1]],axis=1)
    if np.any(cost<=0):raise ValueError('Chart-cut graph contains zero-length edges')
    routes=sparse.coo_matrix((np.r_[cost,cost],(np.r_[safe[:,0],safe[:,1]],np.r_[safe[:,1],safe[:,0]])),shape=graph.shape).tocsr()
    connected=set(int(v) for v in loops[0] if not protected[v]);cuts=set();total_length=0.
    if not connected:raise ValueError('Source witnesses leave no safe outer chart-cut endpoint')
    for loop in loops[1:]:
        distance,previous,_=dijkstra(routes,directed=False,indices=sorted(connected),min_only=True,return_predecessors=True)
        end=int(loop[np.argmin(distance[loop])])
        if not np.isfinite(distance[end]):raise ValueError('No boundary-connecting chart cut avoids the source witnesses')
        total_length+=float(distance[end]);path=[end]
        while path[-1] not in connected:
            parent=int(previous[path[-1]])
            if parent<0 or len(path)>len(vertices):raise ValueError('Invalid source chart-cut predecessor chain')
            cuts.add(tuple(sorted((path[-1],parent))));path.append(parent)
        connected.update(path);connected.update(int(v) for v in loop if not protected[v])
    flags=np.array([tuple(edge) in cuts for edge in edges],bool)[inverse].reshape(len(faces),3)
    if np.any(flags[protected_faces]):raise ValueError('Chart cut touches a protected source witness face')
    opened,triangles,mapping=igl.cut_mesh(vertices,faces,flags)
    if not np.array_equal(opened,vertices[mapping]) or not np.array_equal(mapping[triangles],faces):raise ValueError('Chart cut changed original source geometry or face-corner order')
    return opened,triangles,dict(method='WITNESS_AVOIDING_SOURCE_CHART_CUT_V1',boundary_loops=number,cut_edges=len(cuts),cut_length_m=total_length,duplicated_vertices=len(opened)-len(vertices),protected_source_faces=int(protected_faces.sum()),original_face_corners_exact=True,source_faces_removed=0)
