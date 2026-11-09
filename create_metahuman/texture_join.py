"""Geometry-coupled head/body color continuity, with the head held fixed.

This module operates on explicit atlas/geometry snapshots, without Blender.
It transports a bounded broad color correction along the body surface;
it does not copy tangent-space normal RGB between incompatible tangent frames.
"""
import numpy as np
from scipy import sparse
from scipy.sparse.csgraph import dijkstra
from scipy.sparse.linalg import spsolve


def correspondence(head,body):
    points=np.concatenate((head['points'],body['points']))
    if not np.isfinite(points).all():raise ValueError('Join geometry must be finite')
    tolerance=max(float(np.ptp(points,axis=0).max())*1e-6,1e-9)
    _,labels=np.unique(np.rint(points/tolerance).astype(np.int64),axis=0,return_inverse=True)
    split=len(head['points']);head_labels=labels[:split];body_labels=labels[split:]
    def boundary(data,ids):
        edges={}
        for face,uv in zip(data['faces'],data['triangles']):
            for a,b in ((0,1),(1,2),(2,0)):
                left,right=int(ids[face[a]]),int(ids[face[b]])
                if left==right:continue
                coords=uv[[a,b]] if left<right else uv[[b,a]]
                edges.setdefault(tuple(sorted((left,right))),[]).append((coords,uv.mean(axis=0)))
        return {key:values[0] for key,values in edges.items() if len(values)==1}
    a,b=boundary(head,head_labels),boundary(body,body_labels)
    keys=sorted(set(a)&set(b))
    if len(keys)<3:raise ValueError('No unambiguous shared head/body boundary was found')
    return {'edges':np.asarray([[a[k][0],b[k][0]] for k in keys],np.float32),
            'centers':np.asarray([[a[k][1],b[k][1]] for k in keys],np.float32),
            'keys':np.asarray(keys,np.int32),'body_labels':body_labels,'tolerance':tolerance}


def measurements(head_color,body_color,join):
    from texture_seams import samples,read_color
    # The same normalized edge correspondence is sampled independently at each
    # atlas resolution. Their pixel grids need not have equal dimensions.
    # samples() chooses counts from atlas pixels, so use the larger grid to
    # generate a common parameterization, then transform its positions.
    width=max(head_color.shape[1],body_color.shape[1]);height=max(head_color.shape[0],body_color.shape[0])
    common=samples(join['edges'],join['centers'],(width,height))
    def grid(image):
        ratio=np.array([image.shape[1]/width,image.shape[0]/height])
        return (common+.5)*ratio-.5
    hp,bp=grid(head_color),grid(body_color)
    left=read_color(head_color,hp)[:,0];right=read_color(body_color,bp)[:,1]
    difference=np.abs(left-right).mean(axis=1)
    return {'paired_edges':len(join['edges']),'paired_samples':len(common),
            'encoded_difference_mean':float(difference.mean()),'encoded_difference_p95':float(np.percentile(difference,95)),
            'encoded_difference_max':float(difference.max()),'head_changed':False,'full_surface_visual_approval':False}


def balance_color(head_color,body_color,head,body,*,width=.04,max_delta=.12,protect=None):
    """Return an encoded RGB body candidate; all geometry and head pixels stay fixed.

    width is a fraction of the body's world-space extent. A surface Laplacian
    extends low-frequency boundary mismatch into a finite body band, avoiding
    atlas-distance leakage into unrelated UV islands. Artist protection wins.
    """
    from texture_seams import samples,read_color
    from texture_restore_core import linear,encoded
    if not .005<=width<=.1 or not 0<=max_delta<=.25:raise ValueError('Invalid join correction bounds')
    for image in (head_color,body_color):
        if image.ndim!=3 or image.shape[2]!=3 or not np.isfinite(image).all() or image.min()<0 or image.max()>1:
            raise ValueError('Join color needs finite encoded RGB in 0..1')
    if protect is not None and (protect.shape!=body_color.shape[:2] or not np.isfinite(protect).all() or protect.min()<0 or protect.max()>1):
        raise ValueError('Join protection must match the body atlas in 0..1')
    join=correspondence(head,body);before=measurements(head_color,body_color,join)
    labels=join['body_labels'];unique,inverse=np.unique(labels,return_inverse=True)
    index={int(label):i for i,label in enumerate(unique)}
    positions=np.zeros((len(unique),3),np.float64);positions[inverse]=body['points']
    faces=inverse[body['faces']]
    edges=np.unique(np.sort(np.concatenate((faces[:,[0,1]],faces[:,[1,2]],faces[:,[2,0]])),axis=1),axis=0)
    edges=edges[edges[:,0]!=edges[:,1]]
    lengths=np.linalg.norm(positions[edges[:,0]]-positions[edges[:,1]],axis=1)
    rows=np.r_[edges[:,0],edges[:,1]];cols=np.r_[edges[:,1],edges[:,0]]
    distances=sparse.coo_matrix((np.r_[lengths,lengths],(rows,cols)),shape=(len(unique),len(unique))).tocsr()
    values=np.zeros((len(unique),3),np.float64);counts=np.zeros(len(unique))
    for k,edge,centers in zip(join['keys'],join['edges'],join['centers']):
        # Per-edge means deliberately retain each atlas's fine color texture.
        # They correct palette mismatch, not uncorrelated freckles or pores.
        hp=samples(edge[None],centers[None],(head_color.shape[1],head_color.shape[0]))
        bp=samples(edge[None],centers[None],(body_color.shape[1],body_color.shape[0]))
        delta=linear(read_color(head_color,hp)[:,0]).mean(axis=0)-linear(read_color(body_color,bp)[:,1]).mean(axis=0)
        for label in k:
            i=index[int(label)];values[i]+=delta;counts[i]+=1
    seeds=np.flatnonzero(counts);values[seeds]/=counts[seeds,None];np.clip(values,-max_delta,max_delta,out=values)
    band=float(np.ptp(body['points'],axis=0).max())*width
    distance=dijkstra(distances,directed=False,indices=seeds,min_only=True,limit=band)
    free=np.flatnonzero((distance>0)&(distance<band));fixed=np.ones(len(unique),bool);fixed[free]=False
    weights=1/np.maximum(lengths,band*1e-5)
    adjacency=sparse.coo_matrix((np.r_[weights,weights],(rows,cols)),shape=distances.shape).tocsr()
    laplace=sparse.diags(np.asarray(adjacency.sum(axis=1)).ravel())-adjacency
    if len(free):
        values[free]=spsolve(laplace[free][:,free].tocsc(),-laplace[free][:,fixed]@values[fixed])
    np.clip(values,-max_delta,max_delta,out=values)
    candidate=linear(body_color);height,width_px=body_color.shape[:2];affected=np.zeros((height,width_px),bool)
    field=values[inverse][body['faces']]
    for uv,delta in zip(body['triangles'],field):
        if np.max(np.abs(delta))<1e-8:continue
        p=uv*np.array([width_px,-height])+[-.5,height-.5]
        low=np.maximum(np.floor(p.min(axis=0)).astype(int),0);high=np.minimum(np.ceil(p.max(axis=0)).astype(int),[width_px-1,height-1])
        if np.any(high<low):continue
        a,b,c=p;den=(b[1]-c[1])*(a[0]-c[0])+(c[0]-b[0])*(a[1]-c[1])
        if abs(den)<1e-9:continue
        y,x=np.mgrid[low[1]:high[1]+1,low[0]:high[0]+1]
        u=((b[1]-c[1])*(x-c[0])+(c[0]-b[0])*(y-c[1]))/den
        v=((c[1]-a[1])*(x-c[0])+(a[0]-c[0])*(y-c[1]))/den;w=1-u-v
        inside=(u>=-1e-7)&(v>=-1e-7)&(w>=-1e-7)
        region=np.s_[low[1]:high[1]+1,low[0]:high[0]+1]
        # Adjacent triangles can share a pixel center. Apply the field once.
        inside&=~affected[region]
        correction=u[...,None]*delta[0]+v[...,None]*delta[1]+w[...,None]*delta[2]
        if protect is not None:correction*=1-protect[region][...,None]
        block=candidate[region];block[inside]+=correction[inside];affected[region]|=inside
    np.clip(candidate,0,1,out=candidate);candidate=encoded(candidate).astype(np.float32)
    # No round-trip drift is permitted outside this explicit correction band.
    candidate[~affected]=body_color[~affected]
    if protect is not None:candidate[protect>=1]=body_color[protect>=1]
    after=measurements(head_color,candidate,join)
    return candidate,{'before':before,'after':after,'surface_band_world_units':band,
        'max_linear_correction':max_delta,'changed_texels':int(np.any(candidate!=body_color,axis=2).sum()),
        'fixed_head':True,'geometry_changed':False,'normal_maps_changed':False,
        'method':'bounded_surface_laplacian_palette_transport','visually_approved':False}
