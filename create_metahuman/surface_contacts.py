"""Portable proper triangle-crossing guard for external SciPy workers.

Radius-tier centroid trees conservatively enumerate intersecting bounding
spheres; exact boxes and float64 segment/triangle tests narrow the candidates.
Coplanar overlaps and boundary-only contact are not proper crossings. This is
an endpoint constraint, not continuous collision detection or injectivity.
"""
import numpy as np
from scipy.spatial import cKDTree
from . import geometry


def _candidates(points,triangles,tolerance,active_faces=None):
    p=points[triangles];low=p.min(1);high=p.max(1);centers=p.mean(1)
    radii=np.linalg.norm(p-centers[:,None],axis=2).max(1)
    floor=max(float(np.linalg.norm(np.ptp(points,axis=0)))*1e-10,1e-15)
    bins=np.floor(np.log2(np.maximum(radii,floor)/max(float(radii.max(initial=0)),floor))).astype(int)
    chunks=[];query_ids=np.arange(len(triangles)) if active_faces is None else np.flatnonzero(active_faces)
    for group in np.unique(bins):
        ids=np.flatnonzero(bins==group);tree=cKDTree(centers[ids]);radius=float(radii[ids].max())
        for start in range(0,len(query_ids),256):
            queries=query_ids[start:start+256];found=tree.query_ball_point(centers[queries],radii[queries]+radius+tolerance)
            lengths=np.fromiter((len(x) for x in found),int,count=len(found))
            if not lengths.sum():continue
            first=np.repeat(queries,lengths);second=ids[np.concatenate(found).astype(int)]
            selected=first<second if active_faces is None else first!=second
            first=first[selected];second=second[selected]
            selected=np.all(low[first]<=high[second],axis=1)&np.all(low[second]<=high[first],axis=1)
            if selected.any():chunks.append(np.column_stack((first[selected],second[selected])))
    result=np.vstack(chunks) if chunks else np.empty((0,2),int)
    return result if active_faces is None else np.unique(np.sort(result,axis=1),axis=0)


def crossings(points,triangles,weld=None,*,pairs=False,tolerance=None,active_faces=None):
    points=geometry.points(points);triangles=np.asarray(triangles,dtype=np.int64).reshape(-1,3)
    if not len(triangles) or triangles.min()<0 or triangles.max()>=len(points):raise ValueError('Invalid crossing triangles')
    tolerance=float(np.linalg.norm(np.ptp(points,axis=0))*1e-6 if tolerance is None else tolerance)
    if not np.isfinite(tolerance) or tolerance<0:raise ValueError('Invalid physical collision tolerance')
    physical=triangles if weld is None else np.asarray(weld)[triangles]
    if active_faces is not None:
        active_faces=np.asarray(active_faces,dtype=bool)
        if active_faces.shape!=(len(triangles),):raise ValueError('Active collision faces must match triangles')
    candidates=_candidates(points,triangles,tolerance,active_faces)
    if len(candidates):
        shared=np.any(physical[candidates[:,0],:,None]==physical[candidates[:,1],None,:],axis=(1,2))
        candidates=candidates[~shared]
    found=set()
    if len(candidates):
        first,second=points[triangles[candidates[:,0]]],points[triangles[candidates[:,1]]]
        hit=np.zeros(len(candidates),bool)
        def dot(a,b):return np.einsum('ij,ij->i',a,b)
        for edges,face in ((first,second),(second,first)):
            e1,e2=face[:,1]-face[:,0],face[:,2]-face[:,0]
            normal=np.cross(e1,e2);area=np.linalg.norm(normal,axis=1)
            lengths=np.column_stack((np.linalg.norm(e2,axis=1),np.linalg.norm(e1,axis=1),np.linalg.norm(e2-e1,axis=1)))
            for i in range(3):
                origin=edges[:,i];direction=edges[:,(i+1)%3]-origin
                p=np.cross(direction,e2);det=dot(e1,p)
                scale=np.linalg.norm(e1,axis=1)*np.linalg.norm(e2,axis=1)*np.linalg.norm(direction,axis=1)
                valid=abs(det)>np.maximum(scale*1e-12,1e-300)
                inverse=np.zeros_like(det);np.divide(1.,det,out=inverse,where=valid)
                delta=origin-face[:,0];u=dot(delta,p)*inverse;q=np.cross(delta,e1)
                v=dot(direction,q)*inverse;t=dot(e2,q)*inverse
                plane_start=dot(delta,normal);plane_end=dot(delta+direction,normal);depth=tolerance*area
                straddles=((plane_start>depth)&(plane_end<-depth))|((plane_start<-depth)&(plane_end>depth))
                interior=np.all(np.column_stack((u,v,1-u-v))*area[:,None]>tolerance*lengths,axis=1)
                hit|=valid&straddles&interior&(u>1e-9)&(v>1e-9)&(u+v<1-1e-9)&(t>1e-9)&(t<1-1e-9)
        found={tuple(map(int,pair)) for pair in candidates[hit]}
    return found if pairs else len(found)


class ContactStepGuard:
    """Locally reduce proposed motion before accepting new crossing pairs."""
    description='PORTABLE_ENDPOINT_CONTACT_AND_LOCAL_AREA_GUARD'

    def __init__(self,reference,triangles,*,weld=None,movable=None):
        self.triangles=np.asarray(triangles,dtype=np.int64);self.weld=weld
        self.movable=None if movable is None else np.asarray(movable,dtype=bool)
        if self.movable is not None and self.movable.shape!=(len(reference),):raise ValueError('Contact mobility must match complete vertices')
        self.active_faces=None if self.movable is None else np.any(self.movable[self.triangles],axis=1)
        self.tolerance=float(np.linalg.norm(np.ptp(reference,axis=0))*1e-6)
        self.allowed=crossings(reference,self.triangles,weld,pairs=True,tolerance=self.tolerance,active_faces=self.active_faces)
        self.area=geometry.SurfaceStepGuard(reference,self.triangles if self.active_faces is None else self.triangles[self.active_faces],weld)
        self.calls=0;self.limited_calls=0

    def __call__(self,current,proposed):
        if self.movable is not None:
            proposed=np.array(proposed,copy=True);proposed[~self.movable]=current[~self.movable]
        self.calls+=1;candidate=self.area.limit(current,proposed);direction=candidate-current
        weight=np.ones(len(current));limited=False
        for step in range(24):
            new=crossings(candidate,self.triangles,self.weld,pairs=True,tolerance=self.tolerance,active_faces=self.active_faces)-self.allowed
            if not new:
                self.limited_calls+=int(limited);return candidate
            limited=True;ids=np.unique(self.triangles[np.asarray(list(new),int).ravel()])
            if step<8:weight[ids]*=.5
            else:weight[ids]=0
            if self.weld is not None:
                shared=np.ones(int(np.max(self.weld))+1);np.minimum.at(shared,self.weld,weight);weight=shared[self.weld]
            candidate=self.area.limit(current,current+direction*weight[:,None])
        self.limited_calls+=1
        return current.copy()
