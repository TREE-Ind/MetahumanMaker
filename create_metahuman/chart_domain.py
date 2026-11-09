"""Temporary source-chart domains with exact original face-corner identity."""
import numpy as np
from scipy import sparse
from scipy.sparse.csgraph import connected_components


def prepare(vertices,faces,original_faces,required_faces):
    """Split vertex-only fan contacts and retain the seed's face component.

    This acts only on a cropped chart, never on the source asset. Required
    boundary witnesses must survive in one component. Nonmanifold edges and
    any change in original ordered face corners remain unsupported.
    """
    import igl
    vertices=np.asarray(vertices,dtype=float);faces=np.asarray(faces,dtype=np.int64)
    original_faces=np.asarray(original_faces);required_faces=np.asarray(required_faces)
    if len(original_faces)!=len(faces) or not len(required_faces) or not np.isin(required_faces,original_faces).all():raise ValueError('Chart domain must retain its original seed and boundary facets')
    if not igl.is_edge_manifold(faces):raise ValueError('Source chart has nonmanifold edges')
    old_vertices=vertices;old_faces=faces;old_original=original_faces
    manifold=np.asarray(igl.is_vertex_manifold(faces),dtype=bool)
    split_count=int(np.count_nonzero(~manifold))
    if split_count:
        vertices,faces,mapping=igl.split_nonmanifold(vertices,faces)
        if not np.array_equal(vertices,old_vertices[mapping]) or not np.array_equal(mapping[faces],old_faces):raise ValueError('Chart fan splitting changed original positions or face order')
    edges=np.unique(np.sort(np.vstack((faces[:,[0,1]],faces[:,[1,2]],faces[:,[2,0]])),axis=1),axis=0)
    graph=sparse.coo_matrix((np.ones(len(edges)*2),(np.r_[edges[:,0],edges[:,1]],np.r_[edges[:,1],edges[:,0]])),shape=(len(vertices),len(vertices))).tocsr()
    number,labels=connected_components(graph,directed=False)
    seed=int(np.flatnonzero(original_faces==required_faces[0])[0]);owner=labels[faces[seed,0]]
    keep=labels[faces[:,0]]==owner
    if not np.isin(required_faces,original_faces[keep]).all():raise ValueError('Source boundary witnesses span disconnected face components')
    used=np.unique(faces[keep]);lookup=np.full(len(vertices),-1,int);lookup[used]=np.arange(len(used))
    output_vertices=vertices[used];output_faces=lookup[faces[keep]];output_original=original_faces[keep]
    if not np.array_equal(output_vertices[output_faces],old_vertices[old_faces[keep]]):raise ValueError('Chart domain changed original source face corners')
    receipt=None
    if split_count or not keep.all():
        receipt=dict(method='ORIGINAL_CORNER_MANIFOLD_CHART_DOMAIN_V1',split_nonmanifold_vertices=split_count,
            vertex_copies_after_split=len(vertices)-len(old_vertices),face_components=number,
            omitted_other_component_chart_faces=int(np.count_nonzero(~keep)),retained_source_faces=len(output_faces),
            required_source_witness_faces_retained=int(len(np.unique(required_faces))),original_source_face_corners_exact=True,source_asset_unchanged=True)
    return output_vertices,output_faces,output_original,receipt
