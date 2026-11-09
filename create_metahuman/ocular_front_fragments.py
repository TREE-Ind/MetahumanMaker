"""Worker-only visible frontal fragments with original-vertex interpolation.

Folded lids have anterior-facing triangles behind other lid triangles. Those
hidden folds must not be pulled over the globe along with the visible surface.
"""
import numpy as np
from scipy import sparse
from .ocular_overlay import _clip


def _area(p):
    return float(np.sum(p[:,0]*np.roll(p[:,1],-1)-p[:,1]*np.roll(p[:,0],-1))) if len(p)>=3 else 0.


def _positive(p,f,epsilon=0.):
    result=[]
    for a,b in zip(p,np.roll(p,-1,axis=0)):
        fa=float(f(a)-epsilon);fb=float(f(b)-epsilon)
        if fa>=0:result.append(a)
        if (fa>=0)!=(fb>=0):result.append(a+(b-a)*(fa/(fa-fb)))
    return np.asarray(result,dtype=float).reshape(-1,2)


def _subtract(p,cut):
    intersection=p
    for a,b in zip(cut,np.roll(cut,-1,axis=0)):
        intersection=_clip(intersection,a,b)
        if _area(intersection)<=1e-13:return [p]
    if abs(_area(p)-_area(intersection))<=1e-13:return []
    pieces=[];remaining=p
    for a,b in zip(cut,np.roll(cut,-1,axis=0)):
        outside=_clip(remaining,b,a)
        if _area(outside)>1e-13:pieces.append(outside)
        remaining=_clip(remaining,a,b)
        if len(remaining)<3:break
    return pieces


def visible_fragments(skin,contact,occluders):
    """Partition selected triangles by nearer projected skin, not ray samples.

    Return virtual front-surface triangles and a sparse map to original skin
    vertices. Frame X-right/Z-up/negative-Y-forward; input triangles must have
    positive projected area. The result concerns this frontal view only.
    """
    skin=np.asarray(skin,dtype=float);contact=np.asarray(contact,dtype=int);occluders=np.asarray(occluders,dtype=int)
    projections=skin[occluders][:,:,[0,2]];area=np.array([_area(t) for t in projections]);occluders=occluders[area>1e-12];projections=projections[area>1e-12]
    low=projections.min(1);high=projections.max(1);matrices=np.concatenate((projections,np.ones((*projections.shape[:2],1))),axis=2);planes=np.linalg.solve(matrices,skin[occluders,1,None])[...,0]
    points=[];faces=[];rows=[];cols=[];weights=[];removed=0
    for ids in contact:
        t=skin[ids][:,[0,2]]
        if _area(t)<=1e-12:continue
        T=np.c_[t,np.ones(3)];inv=np.linalg.inv(T);plane=inv@skin[ids,1];parts=[t]
        candidates=np.flatnonzero(np.all(low<=t.max(0)+1e-12,axis=1)&np.all(high>=t.min(0)-1e-12,axis=1))
        for k in candidates:
            if np.array_equal(np.sort(ids),np.sort(occluders[k])):continue
            difference=plane-planes[k]
            if np.max(np.c_[projections[k],np.ones(3)]@difference)<=1e-10:continue
            cut=_positive(projections[k],lambda q:np.r_[q,1.]@difference,1e-10)
            if _area(cut)<=1e-13:continue
            new=[]
            for part in parts:new.extend(_subtract(part,cut))
            removed+=max(0,len(parts)-len(new));parts=new
            if not parts:break
        for part in parts:
            bary=np.c_[part,np.ones(len(part))]@inv;start=len(points);points.extend(bary@skin[ids]);rows.extend(np.repeat(np.arange(start,start+len(part)),3));cols.extend(np.tile(ids,len(part)));weights.extend(bary.ravel())
            faces.extend((start,start+i,start+i+1) for i in range(1,len(part)-1))
    if not faces:raise ValueError('The selected lid patch has no visible frontal surface')
    mapping=sparse.coo_matrix((weights,(rows,cols)),shape=(len(points),len(skin))).tocsr()
    return np.asarray(points),np.asarray(faces,dtype=int),mapping,dict(method='EXACT_FRONTAL_SKIN_ENVELOPE',contact_triangles=len(contact),visible_fragment_triangles=len(faces),occluder_triangles=len(occluders))


def material_topology(triangles,mapping,*,tolerance=1e-10):
    """Join virtual vertices by original-vertex interpolation, never position.

    Fragment extraction intentionally duplicates each polygon's corners. A
    curvature operator needs shared adjacency across their material edges.
    The original contact-witness map is retained separately by the caller.
    """
    if not np.isfinite(tolerance) or not 0<tolerance<=1e-8:raise ValueError('Choose a small finite interpolation identity tolerance')
    mapping=mapping.tocsr(copy=True);mapping.sum_duplicates();mapping.sort_indices();cache={};unique=[];inverse=np.empty(mapping.shape[0],dtype=int)
    for row in range(mapping.shape[0]):
        first,last=mapping.indptr[row:row+2];ids=mapping.indices[first:last];weights=mapping.data[first:last];quantized=np.rint(weights/tolerance).astype(np.int64);key=tuple((int(i),int(w)) for i,w in zip(ids,quantized) if w)
        if key not in cache:cache[key]=len(unique);unique.append(row)
        inverse[row]=cache[key]
    compact=mapping[np.asarray(unique,dtype=int)];difference=mapping-compact[inverse];error=float(abs(difference.data).max(initial=0))
    if error>2*tolerance:raise ValueError('Virtual material vertices disagree beyond the interpolation identity tolerance')
    triangles=inverse[np.asarray(triangles,dtype=int)];triangles=triangles[(triangles[:,0]!=triangles[:,1])&(triangles[:,1]!=triangles[:,2])&(triangles[:,0]!=triangles[:,2])]
    return triangles,compact,dict(method='CANONICAL_INTERPOLATION_VERTEX_IDENTITIES',fragment_vertices=mapping.shape[0],material_vertices=len(unique),maximum_weight_disagreement=error,position_based_welding=False)
