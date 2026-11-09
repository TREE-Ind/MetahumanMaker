"""Stage actual mesh coverage and paired UV seams for texture workers."""
import numpy as np


def native_layout(positions,layout_positions,layout_uv,layout_triangles):
    """Describe native UV coverage/seams using exact canonical identities."""
    positions=np.asarray(positions);ids=np.asarray(layout_positions);uv=np.asarray(layout_uv);tri=np.asarray(layout_triangles)
    if positions.ndim!=2 or positions.shape[1]!=3 or ids.ndim!=1 or not np.issubdtype(ids.dtype,np.integer) or not len(ids) or ids.min()<0 or ids.max()>=len(positions) or uv.shape!=(len(ids),2) or tri.ndim!=2 or tri.shape[1]!=3 or not len(tri) or not np.issubdtype(tri.dtype,np.integer) or tri.min()<0 or tri.max()>=len(ids) or not np.isfinite(positions).all() or not np.isfinite(uv).all():raise ValueError('Supply complete native canonical positions and layout topology')
    uv,origin=local_tile(uv);faces=ids[tri];edges={}
    for face,coordinates in zip(faces,uv[tri]):
        center=coordinates.mean(0)
        for i,j in ((0,1),(1,2),(2,0)):
            a,b=int(face[i]),int(face[j]);pair=coordinates[[i,j]] if a<b else coordinates[[j,i]];edges.setdefault(tuple(sorted((a,b))),[]).append((pair,center))
    pairs=[];centers=[];opened=ambiguous=0
    for entries in edges.values():
        if len(entries)==1:opened+=1
        elif len(entries)!=2:ambiguous+=1
        elif not np.allclose(entries[0][0],entries[1][0],rtol=0,atol=1e-6):pairs.append([entries[0][0],entries[1][0]]);centers.append([entries[0][1],entries[1][1]])
    return dict(points=positions,faces=faces,triangles=uv[tri],uv_tile_origin=origin,seam_edges=np.asarray(pairs,dtype=np.float32).reshape(-1,2,2,2),seam_centers=np.asarray(centers,dtype=np.float32).reshape(-1,2,2),open_edges=np.int32(opened),ambiguous_edges=np.int32(ambiguous),canonical_identity=True)


def local_tile(values):
    """Localize one PNG tile without changing the character's mesh UVs."""
    if values.ndim!=2 or values.shape[1]!=2 or not len(values) or not np.isfinite(values).all():
        raise ValueError('Texture reconstruction needs finite target UVs')
    origin=np.floor(values.min(axis=0)+1e-6).astype(np.int32)
    if np.any(values.max(axis=0)>origin+1+1e-6):
        raise ValueError('This part spans multiple UV tiles; select a single skin tile for PNG reconstruction')
    return values-origin.astype(values.dtype) if np.any(origin) else values,origin


def write(obj, path, context=None,*,protect_anatomy=False):
    mesh=obj.data
    if not mesh.uv_layers.active:
        raise ValueError('Texture reconstruction requires the target skin UV layout')
    uv=mesh.uv_layers.active.data
    mesh.calc_loop_triangles()
    values=np.empty((len(uv),2),dtype=np.float32)
    uv.foreach_get('uv',values.ravel())
    values,tile_origin=local_tile(values)
    tri_loops=np.empty((len(mesh.loop_triangles),3),dtype=np.int32)
    mesh.loop_triangles.foreach_get('loops',tri_loops.ravel())
    triangles=values[tri_loops]
    faces=np.empty_like(tri_loops)
    mesh.loop_triangles.foreach_get('vertices',faces.ravel())
    points=np.empty((len(mesh.vertices),3),dtype=np.float32)
    mesh.vertices.foreach_get('co',points.ravel())
    matrix=np.asarray(obj.matrix_world,dtype=np.float32)
    world=points@matrix[:3,:3].T+matrix[:3,3]
    if context is not None:
        from .mesh_ops import evaluated_coordinates
        world=np.asarray(evaluated_coordinates(context,obj),dtype=np.float32)
        if world.shape!=points.shape:
            raise ValueError('Apply or disable topology-changing modifiers before surface texture reconstruction')
    tolerance=max(float(np.ptp(points,axis=0).max())*1e-7,1e-9)
    welded=np.rint(points/tolerance).astype(np.int64)
    labels=[tuple(row) for row in welded.tolist()]
    vertex_indices=np.empty(len(mesh.loops),dtype=np.int32)
    mesh.loops.foreach_get('vertex_index',vertex_indices)
    starts=np.empty(len(mesh.polygons),dtype=np.int32)
    totals=np.empty_like(starts)
    mesh.polygons.foreach_get('loop_start',starts)
    mesh.polygons.foreach_get('loop_total',totals)
    # Match the former float64 polygon means without one NumPy call per face.
    centers_by_face=(np.add.reduceat(values.astype(np.float64),starts,axis=0)/totals[:,None]).tolist() if len(starts) else []
    uv_rows=values.tolist()
    edges={}
    for start,total,center in zip(starts.tolist(),totals.tolist(),centers_by_face):
        loops=list(range(start,start+total))
        for index,a in enumerate(loops):
            b=loops[(index+1)%len(loops)]
            pa=labels[vertex_indices[a]];pb=labels[vertex_indices[b]]
            key=tuple(sorted((pa,pb)))
            coords=[uv_rows[a],uv_rows[b]]
            if pa>pb:coords.reverse()
            edges.setdefault(key,[]).append((coords,center))
    pairs=[];centers=[];open_edges=ambiguous_edges=0
    for entries in edges.values():
        if len(entries)==1:open_edges+=1
        elif len(entries)!=2:ambiguous_edges+=1
        elif not all(abs(a-b)<=1e-6 for left,right in zip(entries[0][0],entries[1][0]) for a,b in zip(left,right)):
            pairs.append([entries[0][0],entries[1][0]])
            centers.append([entries[0][1],entries[1][1]])
    extras={};anatomy=None
    if protect_anatomy:
        from . import skin_anatomy
        head=context.scene.cm_settings.result if context is not None else None
        head_points=evaluated_coordinates(context,head) if head and head!=obj else None
        fields,anatomy=skin_anatomy.protection(obj,world,faces,welded,
            head_points=np.asarray(head_points,dtype=np.float32) if head_points is not None else None)
        extras.update(fields)
    np.savez_compressed(path,triangles=triangles,uv_tile_origin=tile_origin,**extras,
        points=world,faces=faces,
        seam_edges=np.asarray(pairs,dtype=np.float32).reshape(-1,2,2,2),
        seam_centers=np.asarray(centers,dtype=np.float32).reshape(-1,2,2),
        open_edges=np.int32(open_edges),ambiguous_edges=np.int32(ambiguous_edges),weld_tolerance=tolerance)
    return {'seam_edges':len(pairs),'open_edges':open_edges,'ambiguous_edges':ambiguous_edges,'anatomy':anatomy,
            'uv_tile_origin':tile_origin.tolist(),
            'weld_tolerance':tolerance}
