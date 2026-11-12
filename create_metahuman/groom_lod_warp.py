"""Local rigid MLS proposals for native groom card/helmet rest meshes.

No Blender or Unreal objects enter the solver. Source UVs/topology are not
changed; the native caller owns private asset creation after reviewing gates.
"""
import numpy as np


def _root_constraints(vertices,triangles,roots,targets):
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components
    from scipy.spatial import cKDTree
    edges=np.unique(np.sort(np.concatenate((triangles[:,[0,1]],triangles[:,[1,2]],triangles[:,[2,0]])),axis=1),axis=0)
    a,b=edges.T
    adjacency=coo_matrix((np.ones(2*len(a)),(np.r_[a,b],np.r_[b,a])),shape=(len(vertices),len(vertices))).tocsr()
    _,components=connected_components(adjacency,directed=False)
    roots,ids=np.unique(roots,axis=0,return_index=True);targets=targets[ids]
    tree=cKDTree(roots);distance,closest=tree.query(vertices,k=1,workers=1)
    nearest=np.full(int(components.max())+1,np.inf);np.minimum.at(nearest,components,distance)
    pinned=distance<=nearest[components]+.025
    anchors=vertices.copy()
    if not np.array_equal(roots,targets):
        # Root placement is itself a local rigid transfer, so rotating a
        # complete groom also rotates the offset of its card roots correctly.
        d,near=tree.query(vertices[pinned],k=min(8,len(roots)),workers=1)
        if d.ndim==1:d=d[:,None];near=near[:,None]
        weights=1/(d*d+np.maximum(d[:,-1,None]*.01,1e-8)**2)
        weights/=weights.sum(axis=1,keepdims=True)
        p,q=roots[near],targets[near]
        pc=np.einsum('nk,nki->ni',weights,p);qc=np.einsum('nk,nki->ni',weights,q)
        covariance=np.einsum('nk,nki,nkj->nij',weights,p-pc[:,None],q-qc[:,None])
        covariance+=np.maximum(np.linalg.norm(covariance,axis=(1,2))*1e-9,1e-15)[:,None,None]*np.eye(3)
        u,_,vh=np.linalg.svd(covariance);reflected=np.linalg.det(vh.transpose(0,2,1)@u.transpose(0,2,1))<0
        vh[reflected,-1]*=-1;rotations=vh.transpose(0,2,1)@u.transpose(0,2,1)
        anchors[pinned]=qc+np.einsum('nij,nj->ni',rotations,vertices[pinned]-pc)
    return pinned,anchors


def _regularized(source, desired, triangles, confidence, pins, anchors, strength):
    """Local/global ARAP with fixed root placements and positional observations."""
    from scipy.sparse import coo_matrix, diags
    from scipy.sparse.linalg import splu
    from scipy.spatial import cKDTree
    edges=np.unique(np.sort(np.concatenate((triangles[:,[0,1]],triangles[:,[1,2]],triangles[:,[2,0]])),axis=1),axis=0)
    edges=edges[edges[:,0]!=edges[:,1]]
    a,b=edges.T;count=len(source)
    adjacency=coo_matrix((np.ones(2*len(a)),(np.r_[a,b],np.r_[b,a])),shape=(count,count)).tocsr()
    degree=np.asarray(adjacency.sum(axis=1)).ravel()
    data=np.maximum(degree,1)*strength*confidence
    system=diags(degree+data)-adjacency
    pinned=pins.copy()
    pinned|=degree==0
    free=~pinned
    current=desired.copy();current[pinned]=anchors[pinned];rotations=np.broadcast_to(np.eye(3),(count,3,3)).copy()
    if not free.any():return current,rotations
    solve=splu(system[free][:,free].tocsc())
    boundary=system[free][:,pinned]@anchors[pinned]
    original=source[a]-source[b]
    for _ in range(16):
        deformed=current[a]-current[b]
        covariance=np.zeros((count,3,3))
        contribution=original[:,:,None]*deformed[:,None,:]
        np.add.at(covariance,a,contribution);np.add.at(covariance,b,contribution)
        covariance+=np.maximum(np.linalg.norm(covariance,axis=(1,2))*1e-8,1e-15)[:,None,None]*np.eye(3)
        u,_,vh=np.linalg.svd(covariance);rotation=vh.transpose(0,2,1)@u.transpose(0,2,1)
        reflected=np.linalg.det(rotation)<0;vh[reflected,-1]*=-1
        rotations=vh.transpose(0,2,1)@u.transpose(0,2,1)
        edge=np.einsum('nij,nj->ni',(rotations[a]+rotations[b])*.5,original)
        rhs=data[:,None]*desired
        np.add.at(rhs,a,edge);np.add.at(rhs,b,-edge)
        candidate=solve.solve(rhs[free]-boundary)
        change=float(np.max(np.abs(candidate-current[free]),initial=0))
        current[free]=candidate
        if change<1e-6:break
    # Project the few remaining edge-length violations instead of relaxing
    # their acceptance limits. Fixed root placements retain zero mobility.
    original_length=np.linalg.norm(original,axis=1)
    mobility=np.where(pinned,0.,1/np.maximum(data,1e-8))
    for _ in range(80):
        vector=current[a]-current[b];length=np.linalg.norm(vector,axis=1)
        desired_length=np.clip(length,original_length*.5001,original_length*1.4999)
        active=(original_length>1e-8)&(np.abs(desired_length-length)>1e-7)
        if not active.any():break
        aa,bb=a[active],b[active]
        correction=vector[active]*((length[active]-desired_length[active])/np.maximum(length[active],1e-20))[:,None]
        total=mobility[aa]+mobility[bb]
        correction/=np.maximum(total[:,None],1e-20)
        delta=np.zeros_like(current);counts=np.zeros(count)
        np.add.at(delta,aa,-correction*mobility[aa,None]);np.add.at(delta,bb,correction*mobility[bb,None])
        np.add.at(counts,aa,1);np.add.at(counts,bb,1)
        current=(current+.8*delta/np.maximum(counts[:,None],1)).astype(np.float32).astype(np.float64)
    covariance=np.zeros((count,3,3));deformed=current[a]-current[b]
    contribution=original[:,:,None]*deformed[:,None,:]
    np.add.at(covariance,a,contribution);np.add.at(covariance,b,contribution)
    covariance+=np.maximum(np.linalg.norm(covariance,axis=(1,2))*1e-8,1e-15)[:,None,None]*np.eye(3)
    u,_,vh=np.linalg.svd(covariance);reflection=np.linalg.det(vh.transpose(0,2,1)@u.transpose(0,2,1))<0
    vh[reflection,-1]*=-1;rotations=vh.transpose(0,2,1)@u.transpose(0,2,1)
    return current.astype(np.float32).astype(np.float64),rotations.astype(np.float32).astype(np.float64)


def warp(source, target, vertices, triangles, *, neighbors=32, chunk=512,
         source_uv=None, vertex_uv=None, root_positions=None):
    from scipy.spatial import cKDTree

    source, target, vertices = [np.asarray(v, dtype=np.float64) for v in (source, target, vertices)]
    triangles = np.asarray(triangles, dtype=np.int64)
    if (source.ndim != 2 or source.shape[1:] != (3,) or source.shape != target.shape
            or len(source) < 4 or not len(vertices) or not len(triangles)
            or vertices.ndim != 2 or vertices.shape[1:] != (3,)
            or triangles.ndim != 2 or triangles.shape[1:] != (3,)
            or not all(np.isfinite(v).all() for v in (source, target, vertices))
            or np.any(triangles < 0) or np.any(triangles >= len(vertices))):
        raise ValueError('LOD warp needs finite corresponding controls and an indexed triangle mesh')
    if not 4 <= int(neighbors) <= 128 or not 1 <= int(chunk) <= 4096:
        raise ValueError('Unsupported LOD warp neighborhood or batch size')
    embedded, queries = source, vertices
    semantic = source_uv is not None and vertex_uv is not None
    if semantic:
        source_uv, vertex_uv = np.asarray(source_uv), np.asarray(vertex_uv)
        if (source_uv.shape != (len(source), 2) or vertex_uv.shape != (len(vertices), 2)
                or not np.isfinite(source_uv).all() or not np.isfinite(vertex_uv).all()):
            raise ValueError('Scalp correspondence needs finite root UVs for every control and mesh vertex')
        scale = max(float(np.ptp(source, axis=0).max()), 1e-7)
        if root_positions is not None:
            root_positions=np.asarray(root_positions)
            if root_positions.shape!=source.shape or not np.isfinite(root_positions).all():
                raise ValueError('Root positions must correspond to all strand controls')
            unique,ids=np.unique(np.round(source_uv,6),axis=0,return_index=True)
            if len(unique)>4:
                distances,near=cKDTree(unique).query(unique,k=2,workers=1)
                world=np.linalg.norm(root_positions[ids]-root_positions[ids[near[:,1]]],axis=1)
                usable=(distances[:,1]>1e-5)&(world>1e-6)&(world<scale*.25)
                if usable.any():scale=float(np.clip(np.median(world[usable]/distances[usable,1]),scale*.1,scale*5))
        embedded = np.column_stack((source, source_uv * scale))
        queries = np.column_stack((vertices, vertex_uv * scale))
    tree = cKDTree(embedded)
    k = min(int(neighbors), len(source))
    result = vertices.copy()
    rotations = np.broadcast_to(np.eye(3), (len(vertices), 3, 3)).copy()
    residual = np.zeros(len(vertices)); radius = np.zeros(len(vertices)); nearest = np.zeros(len(vertices))
    for start in range(0, len(vertices), int(chunk)):
        stop = min(len(vertices), start + int(chunk)); query = vertices[start:stop]
        distances, ids = tree.query(queries[start:stop], k=k, workers=1)
        p, q = source[ids], target[ids]
        h = np.maximum(distances[:, -1], 1e-7)
        weights = 1 / (distances ** 2 + (h[:, None] * .05) ** 2)
        weights /= weights.sum(axis=1, keepdims=True)
        pc = np.einsum('nk,nki->ni', weights, p); qc = np.einsum('nk,nki->ni', weights, q)
        a, b = p - pc[:, None], q - qc[:, None]
        covariance = np.einsum('nk,nki,nkj->nij', weights, a, b)
        # A small identity preference suppresses unconstrained twist in a
        # locally almost-collinear lock without introducing scale/shear.
        regularizer = np.maximum(np.linalg.norm(covariance, axis=(1, 2)) * 1e-7, 1e-15)
        covariance += regularizer[:, None, None] * np.eye(3)
        u, _, vh = np.linalg.svd(covariance)
        rotation = vh.transpose(0, 2, 1) @ u.transpose(0, 2, 1)
        reflected = np.linalg.det(rotation) < 0
        vh[reflected, -1] *= -1
        rotation = vh.transpose(0, 2, 1) @ u.transpose(0, 2, 1)
        fitted = qc + np.einsum('nij,nj->ni', rotation, query - pc)
        unchanged = np.max(np.abs(q - p), axis=(1, 2)) <= 1e-10
        fitted[unchanged] = query[unchanged]; rotation[unchanged] = np.eye(3)
        result[start:stop] = fitted; rotations[start:stop] = rotation
        errors = np.einsum('nij,nkj->nki', rotation, a) - b
        residual[start:stop] = np.sqrt(np.einsum('nk,nki,nki->n', weights, errors, errors))
        spatial = np.linalg.norm(p - query[:, None], axis=2)
        radius[start:stop] = np.maximum(spatial.max(axis=1), 1e-7)
        nearest[start:stop] = spatial.min(axis=1)
    pins=np.zeros(len(vertices),dtype=bool);anchors=result.copy()
    if root_positions is not None:
        root_positions=np.asarray(root_positions)
        if root_positions.shape!=source.shape or not np.isfinite(root_positions).all():
            raise ValueError('Strand roots must correspond to every control')
        root_samples=np.linalg.norm(source-root_positions,axis=1)<1e-7
        if root_samples.any():
            pins,anchors=_root_constraints(vertices,triangles,root_positions[root_samples],target[root_samples])
            result[pins]=anchors[pins]
    result=result.astype(np.float32).astype(np.float64)
    rotations=rotations.astype(np.float32).astype(np.float64)
    proposal=result.copy()
    def geometry_quality(positions,frames):
        old=np.cross(vertices[triangles[:,1]]-vertices[triangles[:,0]],vertices[triangles[:,2]]-vertices[triangles[:,0]])
        new=np.cross(positions[triangles[:,1]]-positions[triangles[:,0]],positions[triangles[:,2]]-positions[triangles[:,0]])
        expected=np.einsum('ntij,nj->nti',frames[triangles],old).mean(axis=1)
        lengths=np.linalg.norm(old,axis=1);usable=lengths>max(float(np.ptp(vertices,axis=0).max())**2*1e-12,1e-16)
        cosine=np.einsum('ni,ni->n',expected,new)/np.maximum(np.linalg.norm(expected,axis=1)*np.linalg.norm(new,axis=1),1e-30)
        area=np.linalg.norm(new,axis=1)/np.maximum(lengths,1e-30)
        edges=np.concatenate((triangles[:,[0,1]],triangles[:,[1,2]],triangles[:,[2,0]]))
        old_edge=np.linalg.norm(vertices[edges[:,1]]-vertices[edges[:,0]],axis=1)
        ratio=np.linalg.norm(positions[edges[:,1]]-positions[edges[:,0]],axis=1)/np.maximum(old_edge,1e-30)
        valid=old_edge>1e-8
        return usable&((cosine<.2)|(area<.1)),valid&((ratio<.5)|(ratio>1.5)),ratio,valid
    initial_moved=np.linalg.norm(proposal-vertices,axis=1)>1e-5
    initial_incoherent=initial_moved&(residual>np.maximum(radius*.35,.002))
    initial_extrapolated=initial_moved&(nearest>radius*.75)
    initial_triangles,initial_edges,_,_=geometry_quality(result,rotations)
    regularized=False;capture=1.;spill=0.;attempts=[]
    if initial_incoherent.any() or initial_extrapolated.any() or initial_triangles.any() or initial_edges.any():
        confidence=1/(1+(residual/np.maximum(radius*.35,.002))**2)
        confidence=np.clip(confidence,.05,1)
        requested=np.linalg.norm(proposal-vertices)
        for strength in (64.,16.,4.,1.,.25):
            candidate,frames=_regularized(vertices,proposal,triangles,confidence,pins,anchors,strength)
            bad_tri,bad_edge,_,_=geometry_quality(candidate,frames)
            retained=1-float(np.linalg.norm(candidate-proposal))/max(float(requested),1e-12)
            outside=~initial_moved
            extra=float(np.linalg.norm(candidate[outside]-vertices[outside],axis=1).max(initial=0))
            attempts.append(dict(strength=strength,motion_capture=retained,spill_cm=extra,
                invalid_triangles=int(bad_tri.sum()),strained_edges=int(bad_edge.sum())))
            if (not bad_tri.any() and not bad_edge.any() and retained>=.8
                    and extra<=max(.05,float(np.linalg.norm(proposal-vertices,axis=1).max())*.1)):
                result,rotations=candidate,frames;regularized=True;capture=retained;spill=extra;break
    displacement = np.linalg.norm(result - vertices, axis=1)
    control_delta = np.linalg.norm(target - source, axis=1)
    source_normal = np.cross(vertices[triangles[:, 1]] - vertices[triangles[:, 0]], vertices[triangles[:, 2]] - vertices[triangles[:, 0]])
    target_normal = np.cross(result[triangles[:, 1]] - result[triangles[:, 0]], result[triangles[:, 2]] - result[triangles[:, 0]])
    expected = np.einsum('ntij,nj->nti', rotations[triangles], source_normal).mean(axis=1)
    lengths = np.linalg.norm(source_normal, axis=1)
    usable = lengths > max(float(np.ptp(vertices, axis=0).max()) ** 2 * 1e-12, 1e-16)
    cosine = np.einsum('ni,ni->n', expected, target_normal) / np.maximum(np.linalg.norm(expected, axis=1) * np.linalg.norm(target_normal, axis=1), 1e-30)
    area_ratio = np.linalg.norm(target_normal, axis=1) / np.maximum(lengths, 1e-30)
    edges = np.concatenate((triangles[:, [0, 1]], triangles[:, [1, 2]], triangles[:, [2, 0]]))
    old_edge = np.linalg.norm(vertices[edges[:, 1]] - vertices[edges[:, 0]], axis=1)
    new_edge = np.linalg.norm(result[edges[:, 1]] - result[edges[:, 0]], axis=1)
    ratio = new_edge / np.maximum(old_edge, 1e-30); valid_edges = old_edge > 1e-8
    moved = displacement > 1e-5
    incoherent = moved & (residual > np.maximum(radius * .35, .002))
    extrapolated = moved & (nearest > radius * .75)
    bad_triangles = usable & ((cosine < .2) | (area_ratio < .1))
    bad_edges = valid_edges & ((ratio < .5) | (ratio > 1.5))
    excessive = displacement > max(float(control_delta.max()) * 2, 1e-5)
    accepted = not ((not regularized and (incoherent.any() or extrapolated.any())) or bad_triangles.any() or bad_edges.any() or excessive.any())
    method=('ROOT_UV_MLS_ARAP' if semantic else 'LOCAL_MLS_ARAP') if regularized else ('ROOT_UV_RIGID_MLS' if semantic else 'LOCAL_RIGID_MLS')
    report = dict(accepted=bool(accepted),method=method,vertices=len(vertices),controls=len(source),
                  changed_vertices=int(moved.sum()),max_displacement_cm=float(displacement.max(initial=0)),
                  max_control_displacement_cm=float(control_delta.max(initial=0)),
                  regularized=regularized,motion_capture=capture,maximum_spill_cm=spill,
                  regularization_attempts=attempts,
                  local_fit_incoherent_vertices=int(incoherent.sum()),local_fit_extrapolated_vertices=int(extrapolated.sum()),
                  invalid_triangles=int(bad_triangles.sum()),strained_edges=int(bad_edges.sum()),
                  excessive_vertices=int(excessive.sum()),min_edge_ratio=float(ratio[valid_edges].min(initial=1)),
                  max_edge_ratio=float(ratio[valid_edges].max(initial=1)))
    return result, rotations, report


def main(input_path, output_path):
    import json
    from pathlib import Path

    request = json.loads(Path(input_path).read_text())
    if request.get('schema') != 1:
        raise ValueError('Unsupported groom LOD warp input')
    source = np.asarray(request['source_controls']); target = np.asarray(request['target_controls'])
    rows = []
    for mesh in request['meshes']:
        positions, rotations, report = warp(source, target, mesh['positions'], mesh['triangles'],
            source_uv=request.get('control_root_uv'), vertex_uv=mesh.get('root_uv'),
            root_positions=request.get('control_root_positions'))
        rows.append(dict(source=mesh['source'],lod=mesh['lod'],signature=mesh['signature'],vertex_ids=mesh['vertex_ids'],
                         positions=positions.tolist(),rotations=rotations.tolist(),report=report))
    if not rows:raise ValueError('LOD deformation request has no meshes')
    result = dict(request, status='complete',accepted=all(row['report']['accepted'] for row in rows),meshes=rows)
    for key in ('source_controls','target_controls','control_root_uv','control_root_positions'):result.pop(key,None)
    path = Path(output_path);temporary=path.with_suffix('.tmp')
    temporary.write_text(json.dumps(result,separators=(',',':')));temporary.replace(path)


if __name__ == '__main__':
    import sys
    main(sys.argv[1], sys.argv[2])
