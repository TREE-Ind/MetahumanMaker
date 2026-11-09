"""Exact axis-ray surface intervals for volume-aware template transport.

External NumPy/SciPy worker. Endpoints use original triangle IDs and
barycentrics; no fixed-k centroid approximation or source mutation. Missing
or open columns remain unsupported instead of becoming a single skin sheet.
"""
import numpy as np
from scipy.spatial import cKDTree


def _cap_boundary_loops(vertices,triangles):
    """Append query-only caps to simple consistently oriented boundary loops."""
    directed=np.vstack((triangles[:,[0,1]],triangles[:,[1,2]],triangles[:,[2,0]]))
    _,first,counts=np.unique(np.sort(directed,axis=1),axis=0,return_index=True,return_counts=True)
    boundary=directed[first[counts==1]]
    if not len(boundary):return vertices,triangles
    starts,nstart=np.unique(boundary[:,0],return_counts=True);ends,nend=np.unique(boundary[:,1],return_counts=True)
    if not np.array_equal(starts,ends) or np.any(nstart!=1) or np.any(nend!=1):raise ValueError('Reference boundary is not a collection of oriented simple loops')
    following=dict(map(tuple,boundary));remaining=set(following);centers=[];caps=[]
    while remaining:
        start=min(remaining);ring=[];current=start
        while current not in ring:
            ring.append(current);remaining.discard(current);current=int(following[current])
        if current!=start or len(ring)<3:raise ValueError('Invalid reference boundary loop')
        center=len(vertices)+len(centers);centers.append(vertices[ring].mean(0))
        caps.extend((following[vertex],vertex,center) for vertex in ring)
    return np.vstack((vertices,centers)),np.vstack((triangles,np.asarray(caps,int)))


class AxisIntervals:
    def __init__(self,vertices,triangles,axis,face_ids=None,*,require_orientation=False,cap_boundaries=False):
        self.vertices=np.array(vertices,dtype=float,copy=True);self.triangles=np.array(triangles,dtype=np.int64,copy=True)
        if self.vertices.ndim!=2 or self.vertices.shape[1]!=3 or not np.isfinite(self.vertices).all() or self.triangles.ndim!=2 or self.triangles.shape[1]!=3 or not len(self.triangles) or self.triangles.min()<0 or self.triangles.max()>=len(self.vertices):raise ValueError('Invalid interval surface')
        if axis not in (0,1,2):raise ValueError('Choose a ray axis from zero to two')
        if cap_boundaries:
            if face_ids is not None:raise ValueError('Cap the complete reference, not a clipped face region')
            self.vertices,self.triangles=_cap_boundary_loops(self.vertices,self.triangles)
        self.axis=axis;self.plane=[i for i in range(3) if i!=axis]
        self.require_orientation=bool(require_orientation)
        ids=np.arange(len(self.triangles)) if face_ids is None else np.asarray(face_ids,dtype=np.int64)
        if ids.ndim!=1 or not len(ids) or ids.min()<0 or ids.max()>=len(self.triangles):raise ValueError('Choose valid interval faces')
        surface=self.vertices[self.triangles[ids]];projected=surface[:,:,self.plane]
        e1=projected[:,1]-projected[:,0];e2=projected[:,2]-projected[:,0]
        det=e1[:,0]*e2[:,1]-e1[:,1]*e2[:,0]
        valid=abs(det)>np.maximum(np.linalg.norm(e1,axis=1)*np.linalg.norm(e2,axis=1)*1e-12,1e-300)
        self.ids=ids[valid];self.projected=projected[valid];self.det=det[valid]
        self.normals=np.cross(surface[valid,1]-surface[valid,0],surface[valid,2]-surface[valid,0])
        self.normals/=np.maximum(np.linalg.norm(self.normals,axis=1)[:,None],1e-300)
        self.span=float(np.linalg.norm(np.ptp(surface.reshape(-1,3),axis=0)));self.tolerance=max(self.span*1e-10,1e-14)
        centers=self.projected.mean(1);radii=np.linalg.norm(self.projected-centers[:,None],axis=2).max(1)
        self.tiers=[]
        if len(radii):
            floor=max(self.span*1e-9,1e-15);bins=np.floor(np.log2(np.maximum(radii,floor)/floor)).astype(int)
            for group in np.unique(bins):
                local=np.flatnonzero(bins==group);self.tiers.append((local,cKDTree(centers[local]),float(radii[local].max())))
        self.vertices.flags.writeable=False;self.triangles.flags.writeable=False

    def query(self,points,cancelled=None,*,rank_hint=None,count_hint=None):
        points=np.asarray(points,dtype=float)
        if points.ndim!=2 or points.shape[1]!=3 or not np.isfinite(points).all():raise ValueError('Invalid interval query points')
        n=len(points);low=np.full(n,np.nan);high=low.copy();lower_ids=np.full(n,-1,int);upper_ids=lower_ids.copy()
        lower_bary=np.zeros((n,3));upper_bary=lower_bary.copy();observed=np.zeros(n,bool);has_span=np.zeros(n,bool);oriented=np.zeros(n,bool)
        interval_rank=np.full(n,-1,int);interval_count=np.zeros(n,int)
        for start in range(0,n,64):
            if cancelled and cancelled():raise RuntimeError('Interval query cancelled')
            projected=points[start:start+64][:,self.plane];candidates=[[] for _ in projected]
            for ids,tree,radius in self.tiers:
                found=tree.query_ball_point(projected,radius+self.tolerance)
                for i,selected in enumerate(found):
                    if len(selected):candidates[i].append(ids[np.asarray(selected,int)])
            for i,parts in enumerate(candidates):
                if not parts:continue
                ids=np.concatenate(parts);face=self.projected[ids]
                a=face[:,1]-face[:,0];b=face[:,2]-face[:,0];delta=projected[i]-face[:,0]
                u=(delta[:,0]*b[:,1]-delta[:,1]*b[:,0])/self.det[ids]
                v=(a[:,0]*delta[:,1]-a[:,1]*delta[:,0])/self.det[ids]
                inside=(u>=-1e-9)&(v>=-1e-9)&(u+v<=1+1e-9)
                ids=ids[inside];bary=np.column_stack((1-u[inside]-v[inside],u[inside],v[inside]))
                if not len(ids):continue
                bary=np.maximum(bary,0);bary/=bary.sum(1)[:,None]
                heights=np.einsum('ni,ni->n',bary,self.vertices[self.triangles[self.ids[ids]],self.axis])
                # Shared triangle edges produce duplicate hits. Pair unique
                # crossings in depth order rather than spanning empty gaps
                # between different toes or other nonconvex surface lobes.
                order=np.lexsort((self.ids[ids],heights));unique=np.r_[True,np.diff(heights[order])>self.span*1e-8]
                order=order[unique];j=start+i
                if len(order)<2 or len(order)%2:continue
                pairs=order.reshape(-1,2);count=len(pairs)
                if rank_hint is not None and count_hint is not None and count_hint[j]==count and 0<=rank_hint[j]<count:
                    rank=int(rank_hint[j])
                else:
                    lower=heights[pairs[:,0]];upper=heights[pairs[:,1]];position=points[j,self.axis]
                    distance=np.maximum(np.maximum(lower-position,position-upper),0)
                    rank=int(np.lexsort((abs((lower+upper)*.5-position),distance))[0])
                lo,hi=map(int,pairs[rank]);interval_rank[j]=rank;interval_count[j]=count
                if heights[hi]-heights[lo]<=self.span*1e-5:continue
                has_span[j]=True
                oriented[j]=self.normals[ids[lo],self.axis]<-.05 and self.normals[ids[hi],self.axis]>.05
                if self.require_orientation and not oriented[j]:continue
                low[j]=heights[lo];high[j]=heights[hi];lower_ids[j]=self.ids[ids[lo]];upper_ids[j]=self.ids[ids[hi]]
                lower_bary[j]=bary[lo];upper_bary[j]=bary[hi];observed[j]=True
        return dict(low=low,high=high,observed=observed,has_span=has_span,oriented=oriented,interval_rank=interval_rank,interval_count=interval_count,lower_triangle_ids=lower_ids,
            upper_triangle_ids=upper_ids,lower_barycentric=lower_bary,upper_barycentric=upper_bary)


def transport_axis(reference,triangles,source,source_triangles,influence,*,axis,source_face_ids,reference_face_ids=None,cancelled=None,cap_reference=False,max_axis_scale=8.):
    """Map shell depth between paired surfaces, preserving layer ordering.

    Endpoints can be on different surface components. Call on anatomically
    scoped regions; this is not a substitute for articulation or visibility
    inference. Endpoint/quality review is required after reconstruction.
    """
    reference=np.asarray(reference,dtype=float);influence=np.asarray(influence,dtype=float)
    if influence.shape!=(len(reference),) or not np.isfinite(influence).all() or np.any((influence<0)|(influence>1)):raise ValueError('Invalid interval transport influence')
    if not np.isfinite(max_axis_scale) or max_axis_scale<1:raise ValueError('Choose an interval scale bound of at least one')
    ids=np.flatnonzero(influence>0);points=reference[ids]
    # Anatomical labels select admissible endpoints, not the ray mesh. Cutting
    # the triangles first opens the proximal joint and corrupts ray parity.
    native=AxisIntervals(reference,triangles,axis,cap_boundaries=cap_reference).query(points,cancelled)
    target=AxisIntervals(source,source_triangles,axis).query(points,cancelled,
        rank_hint=native['interval_rank'],count_hint=native['interval_count'])
    valid=native['observed']&target['observed'];fraction=np.zeros(len(ids))
    if reference_face_ids is not None:
        valid&=np.isin(native['lower_triangle_ids'],reference_face_ids)&np.isin(native['upper_triangle_ids'],reference_face_ids)
    valid&=np.isin(target['lower_triangle_ids'],source_face_ids)&np.isin(target['upper_triangle_ids'],source_face_ids)
    ratio=np.ones(len(ids));ratio[valid]=(target['high'][valid]-target['low'][valid])/(native['high'][valid]-native['low'][valid])
    too_scaled=valid&((ratio>max_axis_scale)|(ratio<1/max_axis_scale));valid&=~too_scaled
    fraction[valid]=np.clip((points[valid,axis]-native['low'][valid])/(native['high'][valid]-native['low'][valid]),0,1)
    desired=points[:,axis].copy();desired[valid]=target['low'][valid]+fraction[valid]*(target['high'][valid]-target['low'][valid])
    result=reference.copy();result[ids,axis]+=(desired-points[:,axis])*influence[ids]
    return result,dict(method='PAIRED_SURFACE_INTERVAL_TRANSPORT_V1',axis=int(axis),queried=len(ids),supported=int(valid.sum()),
        native_supported=int(native['observed'].sum()),source_supported=int(target['observed'].sum()),
        native_oriented=int(native['oriented'].sum()),source_oriented=int(target['oriented'].sum()),
        complete_surface_rays=True,
        reference_query_caps=bool(cap_reference),scale_rejected=int(too_scaled.sum()),
        unsupported=int((~valid).sum()),source_mutated=False,production_ready=False),dict(vertex_ids=ids,valid=valid,fraction=fraction,native=native,source=target)
