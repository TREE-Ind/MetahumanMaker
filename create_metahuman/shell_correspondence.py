"""Ordered ray-shell correspondence for explicitly aligned anatomical frames.

The first or farthest hit alone cannot distinguish a neck from an overhanging
chin. Match the oriented crossing's order and keep unmatched layers unknown.
Blender BVH supplies ray candidates; source triangle IDs remain traceable.
"""
import numpy as np
from mathutils import Vector


def layers(tree,origin,direction,*,limit=2.,epsilon=2e-6,max_hits=20):
    origin=np.asarray(origin,dtype=float);direction=np.asarray(direction,dtype=float)
    direction/=max(np.linalg.norm(direction),1e-20);cursor=origin.copy();hits=[]
    for _ in range(max_hits):
        remaining=limit-float((cursor-origin)@direction)
        if remaining<=0:break
        point,normal,face,_=tree.ray_cast(Vector(cursor),Vector(direction),remaining)
        if point is None:break
        point=np.array(point);normal=np.array(normal);distance=float((point-origin)@direction)
        facing=float(normal@direction)
        if abs(facing)>.05:hits.append((distance,point,face,1 if facing>0 else -1))
        cursor=point+direction*epsilon
    return hits


def match(reference_tree,source_tree,query,reference_origin,source_origin,source_direction,*,limit=2.):
    query=np.asarray(query,dtype=float);origin=np.asarray(reference_origin,dtype=float);direction=query-origin
    length=float(np.linalg.norm(direction))
    if length<1e-8:return None
    reference=layers(reference_tree,origin,direction,limit=limit)
    if not reference:return None
    index=min(range(len(reference)),key=lambda i:np.linalg.norm(reference[i][1]-query))
    if np.linalg.norm(reference[index][1]-query)>1e-4:return None
    sign=reference[index][3];ordinal=sum(hit[3]==sign for hit in reference[:index+1])-1
    source=[hit for hit in layers(source_tree,source_origin,source_direction,limit=limit) if hit[3]==sign]
    if ordinal>=len(source):return None
    return source[ordinal][1],source[ordinal][2],dict(orientation=sign,ordinal=ordinal,
        reference_layers=len(reference),source_oriented_layers=len(source))
