"""Locate native lid boundaries and the external skin sheet at every LOD.

UV chart connectivity separates external skin from hidden socket tissue.
Canonical DNA edges retain physical connectivity across texture/normal splits.
Eye volumes identify the two boundary loops; character-specific vertex lists
and positional vertex welding are unnecessary.
"""
import numpy as np
from scipy import sparse
from scipy.sparse.csgraph import connected_components


def select(head,eyes):
    positions=np.asarray(head['positions'],dtype=float);ids=np.asarray(head['layout_positions']);uv=np.asarray(head['layout_uv'],dtype=float);tri=np.asarray(head['triangles'])
    if not eyes or positions.ndim!=2 or positions.shape[1]!=3 or uv.shape!=(len(ids),2) or tri.ndim!=2 or tri.shape[1]!=3 or not np.isfinite(positions).all() or not np.isfinite(uv).all():raise ValueError('Supply finite native head charts and eye anatomy')
    keys={};vertices=[]
    for vertex,coordinate in zip(ids,uv):
        key=(int(vertex),coordinate.tobytes())
        if key not in keys:keys[key]=len(keys)
        vertices.append(keys[key])
    chart_tri=np.asarray(vertices)[tri];edges=np.vstack((chart_tri[:,[0,1]],chart_tri[:,[1,2]],chart_tri[:,[2,0]]));graph=sparse.coo_matrix((np.ones(len(edges)*2),(np.r_[edges[:,0],edges[:,1]],np.r_[edges[:,1],edges[:,0]])),shape=(len(keys),len(keys))).tocsr();count,labels=connected_components(graph,directed=False);face_labels=labels[chart_tri[:,0]]
    corners=uv[tri];edge=corners[:,1:]-corners[:,:1];area=np.abs(edge[:,0,0]*edge[:,1,1]-edge[:,0,1]*edge[:,1,0])*.5;areas=np.bincount(face_labels,weights=area,minlength=count);physical=ids[tri]
    def boundaries(faces):
        edges,n=np.unique(np.sort(np.vstack((faces[:,[0,1]],faces[:,[1,2]],faces[:,[2,0]])),axis=1),axis=0,return_counts=True);boundary=edges[n==1];neighbors={}
        for a,b in boundary:neighbors.setdefault(int(a),set()).add(int(b));neighbors.setdefault(int(b),set()).add(int(a))
        unseen=set(neighbors);loops=[]
        while unseen:
            seed=min(unseen);group={seed};queue=[seed]
            while queue:
                for vertex in neighbors[queue.pop()]:
                    if vertex not in group:group.add(vertex);queue.append(vertex)
            unseen-=group
            if len(group)<3 or any(len(neighbors[v])!=2 for v in group):continue
            ordered=[seed];previous=None;current=seed
            while True:
                following=sorted(neighbors[current]-({previous} if previous is not None else set()))[0]
                if following==seed:break
                ordered.append(following);previous,current=current,following
                if len(ordered)>len(group):raise ValueError('Native lid boundary did not close')
            if len(ordered)==len(group):loops.append(np.asarray(ordered,dtype=np.int64))
        return loops
    selected=None
    for component in np.argsort(areas)[::-1]:
        if areas[component]<=0:continue
        faces=physical[face_labels==component];loops=boundaries(faces)
        if len(loops)<len(eyes):continue
        chosen={};used=set()
        for role,eye in eyes.items():
            points=np.asarray(eye['positions'],dtype=float);center=points.mean(0);width=float(np.ptp(points,axis=0).max());distances=[float(np.linalg.norm(positions[loop].mean(0)-center)) for loop in loops];order=np.argsort(distances)
            candidate=next((int(i) for i in order if int(i) not in used and distances[i]<=width*1.5),None)
            if candidate is None:break
            chosen[role]=(loops[candidate],distances[candidate]);used.add(candidate)
        if len(chosen)==len(eyes):selected=(int(component),faces,chosen);break
    if selected is None:raise ValueError('Native chart topology does not identify a separate boundary for each eye')
    component,faces,chosen=selected;result={}
    all_edges=np.sort(np.stack((physical[:,[0,1]],physical[:,[1,2]],physical[:,[2,0]]),axis=1),axis=2)
    for role,(margin,distance) in chosen.items():
        boundary={tuple(sorted((int(a),int(b)))) for a,b in zip(margin,np.roll(margin,-1))};adjacent=[]
        for i,edges in enumerate(all_edges):
            if face_labels[i]!=component and any(tuple(map(int,edge)) in boundary for edge in edges):adjacent.append(face_labels[i])
        socket=np.unique(physical[np.isin(face_labels,np.unique(adjacent))]) if adjacent else np.empty(0,dtype=np.int64)
        result[role]=dict(margin=margin,interior=np.union1d(socket,margin),anatomical_center_distance=distance)
    return dict(outer_triangles=faces,eyes=result,outer_uv_component=component,uv_components=count,method='NATIVE_CHART_TOPOLOGY_AND_EYE_BOUNDARIES')


def visibility_calibrated(head,eyes):
    """Refine chart seeds to the pristine anatomical open-eye silhouette.

    Supply physical X-right/Z-up/negative-Y-forward anatomy. A coarse LOD
    whose visibility falls outside the seeded topology is rejected rather
    than silently using a texture-chart boundary as the visible lid rim.
    """
    from .ocular_surface import _aperture_margin
    seed=select(head,eyes);positions=np.asarray(head['positions']);faces=np.asarray(head['layout_positions'])[np.asarray(head['triangles'])]
    edges=np.unique(np.sort(np.vstack((faces[:,[0,1]],faces[:,[1,2]],faces[:,[2,0]])),axis=1),axis=0)
    adjacency=sparse.coo_matrix((np.ones(len(edges)*2),(np.r_[edges[:,0],edges[:,1]],np.r_[edges[:,1],edges[:,0]])),shape=(len(positions),len(positions))).tocsr();result={};hidden=[]
    for role,eye in eyes.items():
        lid=seed['eyes'][role];margin,interior,ring,proof=_aperture_margin(positions,faces,adjacency,lid['interior'],lid['margin'],np.asarray(eye['positions']),np.asarray(eye['triangles']))
        result[role]=dict(margin=margin,interior=interior,ring=ring,proof=proof);hidden.extend(np.setdiff1d(interior,margin).tolist())
    return dict(outer_triangles=faces[~np.any(np.isin(faces,hidden),axis=1)],eyes=result,method='NATIVE_PRISTINE_VISIBILITY_BOUNDARY',native_motion_accepted=False)
