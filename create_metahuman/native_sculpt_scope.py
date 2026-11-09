"""Physical, density-independent support for captured closed-eye sculpting."""
import heapq
import numpy as np


def physical_support(reference, sheet, margin, eye_width, *, fraction=.45):
    """Return complete exterior triangles within a neutral surface distance.

    This selects a new authoring region before a session is sealed. It never
    edits an existing protection mask or changes the corrective movement bound.
    Uses only NumPy and the standard library so Blender needs no worker imports.
    """
    p=np.asarray(reference,dtype=float);tri=np.asarray(sheet,dtype=int);rim=np.unique(np.asarray(margin,dtype=int))
    radius=float(eye_width)*float(fraction)
    if (p.ndim!=2 or p.shape[1]!=3 or not np.isfinite(p).all()
            or tri.ndim!=2 or tri.shape[1]!=3 or not len(tri)
            or tri.min()<0 or tri.max()>=len(p) or not len(rim)
            or rim.min()<0 or rim.max()>=len(p)
            or not np.isfinite(radius) or radius<=0):
        raise ValueError('Physical sculpt coverage requires finite neutral anatomy and positive eye width')
    edges=np.unique(np.sort(np.vstack((tri[:,[0,1]],tri[:,[1,2]],tri[:,[2,0]])),axis=1),axis=0)
    length=np.linalg.norm(p[edges[:,0]]-p[edges[:,1]],axis=1)
    if np.any(length<=0):raise ValueError('The neutral exterior contains collapsed sculpt edges')
    adjacency=[[] for _ in p]
    for (a,b),weight in zip(edges,length):
        adjacency[a].append((int(b),float(weight)));adjacency[b].append((int(a),float(weight)))
    distance=np.full(len(p),np.inf);distance[rim]=0.;queue=[(0.,int(i)) for i in rim];heapq.heapify(queue)
    while queue:
        value,node=heapq.heappop(queue)
        if value!=distance[node]:continue
        for neighbor,weight in adjacency[node]:
            proposed=value+weight
            if proposed<=radius and proposed<distance[neighbor]:
                distance[neighbor]=proposed;heapq.heappush(queue,(proposed,neighbor))
    patch=tri[np.all(distance[tri]<=radius,axis=1)]
    support=np.unique(patch)
    if not len(patch) or not np.isin(rim,support).all():
        raise ValueError('Physical coverage does not contain the complete lid margin; retain the local sculpt region')
    return support,dict(method='NEUTRAL_SURFACE_DISTANCE',eye_width_cm=float(eye_width),
                        radius_cm=radius,vertices=len(support),triangles=len(patch),
                        maximum_distance_cm=float(distance[support].max()))
