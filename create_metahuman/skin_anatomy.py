"""Geometry-anchored protection for body material invention."""
import hashlib
import heapq
import re
import numpy as np


def vertex_weights(obj):
    groups={group.index:group.name for group in obj.vertex_groups}
    extremities={index for index,name in groups.items() if re.match(r'^(hand|foot|ball|thumb|index|middle|ring|pinky|.*toe)(_|$)',name,re.I)}
    neck={index for index,name in groups.items() if re.match(r'^(neck|head)(_|$)',name,re.I)}
    if not extremities:
        raise ValueError('Body anatomy protection needs retained hand and foot bone groups; disable it for a different body layout')
    values=np.zeros((len(obj.data.vertices),2),np.float32)
    for vertex in obj.data.vertices:
        for item in vertex.groups:
            if item.group in extremities:values[vertex.index,0]+=item.weight
            if item.group in neck:values[vertex.index,1]+=item.weight
    np.clip(values,0,1,out=values)
    if not np.any(values[:,0]>.05):raise ValueError('Retain the body hand/foot skin weights before protecting their anatomy')
    names=sorted(groups[index] for index in extremities|neck)
    signature=hashlib.sha256(values.tobytes()+'\n'.join(names).encode()).hexdigest()
    return values,{'bone_groups':names,'weight_signature':signature}


def protection(obj,points,faces,welded,*,head_points=None):
    weights,report=vertex_weights(obj)
    _,inverse=np.unique(welded,axis=0,return_inverse=True)
    count=int(inverse.max())+1
    canonical=np.empty(count,np.int32);canonical[inverse]=np.arange(len(inverse))
    positions=points[canonical]
    edges=np.sort(np.concatenate((inverse[faces[:,[0,1]]],inverse[faces[:,[1,2]]],inverse[faces[:,[2,0]]])),axis=1)
    edges,counts=np.unique(edges,axis=0,return_counts=True)
    neck_weight=np.zeros(count,np.float32);np.maximum.at(neck_weight,inverse,weights[:,1])
    extent=float(np.ptp(points,axis=0).max())
    boundary=np.unique(edges[counts==1]);seeds=boundary[neck_weight[boundary]>.03];anchor='neck_bone_weights'
    if head_points is not None:
        selected=[]
        for index in boundary:
            if np.min(np.linalg.norm(head_points-positions[index],axis=1))<=extent*1e-4:selected.append(index)
        seeds=np.asarray(selected,np.int32);anchor='matching_head_surface_boundary'
    elif not len(seeds):
        seeds=boundary[positions[boundary,2]>=points[:,2].max()-extent*.1];anchor='upper_body_open_boundary'
    if not len(seeds):raise ValueError('No matching open neck boundary was found; use a compatible head/body pair or disable anatomy protection')
    lock=extent*.003;feather=extent*.03
    graph=[[] for _ in range(count)]
    lengths=np.linalg.norm(positions[edges[:,0]]-positions[edges[:,1]],axis=1)
    for (a,b),length in zip(edges.tolist(),lengths.tolist()):
        if a!=b:graph[a].append((b,length));graph[b].append((a,length))
    distance=np.full(count,np.inf);distance[seeds]=0
    queue=[(0.,int(index)) for index in seeds];heapq.heapify(queue)
    while queue:
        value,index=heapq.heappop(queue)
        if value>distance[index]:continue
        for other,length in graph[index]:
            proposed=value+length
            if proposed<=lock+feather and proposed<distance[other]:
                distance[other]=proposed;heapq.heappush(queue,(proposed,other))
    neck=np.clip(1-(distance[inverse]-lock)/feather,0,1).astype(np.float32)
    neck=neck*neck*(3-2*neck)
    extremities=np.clip(weights[:,0]*4,0,1)
    extremities=extremities*extremities*(3-2*extremities)
    values=np.maximum(extremities,neck)
    report.update(neck_boundary_vertices=len(seeds),neck_lock_world_units=lock,neck_feather_world_units=feather,
                  neck_anchor=anchor,
                  protected_vertices=int(np.count_nonzero(values)),fully_protected_vertices=int(np.count_nonzero(values>=1)),
                  method='retained_bone_weights_and_welded_surface_distance')
    return {'anatomy_protection':values[faces],'anatomy_color_match':extremities[faces],
            'anatomy_neck_anchor':neck[faces]},report


def match_color(color,source,weight):
    """Retain source anatomy while transporting only the broad skin palette."""
    import cv2
    from scipy import ndimage
    h,w=weight.shape;size=(min(512,w),min(512,h))
    support=weight*(color.max(axis=2)>.02)
    low_weight=ndimage.gaussian_filter(cv2.resize(support,size,interpolation=cv2.INTER_AREA),3)
    valid=low_weight>.001;gains=[]
    for channel in range(3):
        a=ndimage.gaussian_filter(cv2.resize(source[...,channel]*support,size,interpolation=cv2.INTER_AREA),3)
        b=ndimage.gaussian_filter(cv2.resize(color[...,channel]*support,size,interpolation=cv2.INTER_AREA),3)
        ratio=np.ones_like(a);ratio[valid]=np.clip(b[valid]/np.maximum(a[valid],1e-5),.65,1.65)
        gains.append([float(ratio[valid].min()),float(ratio[valid].max())] if valid.any() else [1.,1.])
        field=cv2.resize(ratio,(w,h),interpolation=cv2.INTER_LINEAR)
        for y in range(0,h,128):
            original=source[y:y+128,:,channel]*field[y:y+128]
            block=color[y:y+128,:,channel]
            block+=(np.clip(original,0,1)-block)*weight[y:y+128]
        del field
    return {'method':'source_detail_with_normalized_coarse_palette_transport','field_size':list(size),
            'gain_bounds':[.65,1.65],'channel_gain_ranges':gains,
            'fully_preserved_detail_texels':int(np.count_nonzero(weight>=1)),
            'absolute_color_lock':False,'invented_anatomy':False}


def rasterize(triangles,weights,size):
    """Interpolate protection across each UV face without stepped neck bands."""
    from PIL import Image,ImageDraw
    width,height=size
    result=np.zeros((height,width),np.float32)
    full=Image.new('L',size);draw=ImageDraw.Draw(full)
    for tri,values in zip(triangles,weights):
        if not np.any(values>0):continue
        points=tri*np.array([width-1,-(height-1)],np.float32)+[0,height-1]
        if np.all(values>=1):
            draw.polygon([tuple(p) for p in points],fill=255);continue
        low=np.maximum(np.floor(points.min(axis=0)).astype(int),0)
        high=np.minimum(np.ceil(points.max(axis=0)).astype(int),[width-1,height-1])
        if np.any(high<low):continue
        a,b,c=points
        denominator=(b[1]-c[1])*(a[0]-c[0])+(c[0]-b[0])*(a[1]-c[1])
        if abs(denominator)<1e-8:continue
        y,x=np.mgrid[low[1]:high[1]+1,low[0]:high[0]+1]
        u=((b[1]-c[1])*(x-c[0])+(c[0]-b[0])*(y-c[1]))/denominator
        v=((c[1]-a[1])*(x-c[0])+(a[0]-c[0])*(y-c[1]))/denominator
        w=1-u-v;inside=(u>=-1e-5)&(v>=-1e-5)&(w>=-1e-5)
        field=np.clip(u*values[0]+v*values[1]+w*values[2],0,1).astype(np.float32);field[~inside]=0
        block=result[low[1]:high[1]+1,low[0]:high[0]+1];np.maximum(block,field,out=block)
    np.maximum(result,np.asarray(full,dtype=np.float32)/255,out=result)
    return result
