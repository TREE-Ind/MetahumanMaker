"""Float64 triangle-surface queries without fixed-k centroid approximations.

Triangles are partitioned by bounding-sphere size. For each size tier, a
centroid tree retrieves every sphere that could beat a known vertex-distance
upper bound. Exact triangle distance then chooses the witness. This avoids
missing large sparse triangles among densely tessellated nearby features.
SciPy is an external-worker dependency, never imported by add-on registration.
"""
import numpy as np
from scipy.spatial import cKDTree
from .vertex_contract import closest_triangle_points,witnesses


class SurfaceOracle:
    def __init__(self,vertices,triangles,face_ids=None):
        if face_ids is not None and np.asarray(face_ids).dtype.kind=='b':
            raise ValueError('Surface face IDs must be integer indices; convert a face mask with flatnonzero')
        self.vertices=np.array(vertices,dtype=np.float64,copy=True)
        self.triangles=np.array(triangles,dtype=np.int64,copy=True)
        self.face_ids=np.arange(len(triangles),dtype=np.int64) if face_ids is None else np.array(face_ids,dtype=np.int64,copy=True)
        if self.vertices.ndim!=2 or self.vertices.shape[1]!=3 or self.triangles.ndim!=2 or self.triangles.shape[1]!=3:
            raise ValueError('Surface oracle needs vertex and triangle arrays')
        if not len(self.face_ids) or self.face_ids.min()<0 or self.face_ids.max()>=len(self.triangles):
            raise ValueError('Choose nonempty valid source faces')
        if self.triangles.min()<0 or self.triangles.max()>=len(self.vertices) or not np.isfinite(self.vertices).all():
            raise ValueError('Invalid source surface')
        for array in (self.vertices,self.triangles,self.face_ids):array.flags.writeable=False
        surface=self.vertices[self.triangles[self.face_ids]]
        centers=surface.mean(1);radii=np.linalg.norm(surface-centers[:,None],axis=2).max(1)
        self.radii=radii;radii.flags.writeable=False
        span=float(np.linalg.norm(np.ptp(surface.reshape(-1,3),axis=0)))
        self.tolerance=max(span*1e-12,1e-14)
        normals=np.cross(surface[:,1]-surface[:,0],surface[:,2]-surface[:,0])
        self.normals=normals/np.maximum(np.linalg.norm(normals,axis=1)[:,None],1e-300)
        self.center_tree=cKDTree(centers)
        self.vertex_tree=cKDTree(self.vertices[np.unique(self.triangles[self.face_ids])])
        positive=np.maximum(radii,max(span*1e-9,1e-15))
        bins=np.floor(np.log2(positive/positive.min())).astype(int)
        self.tiers=[]
        for group in np.unique(bins):
            ids=np.flatnonzero(bins==group)
            self.tiers.append((ids,cKDTree(centers[ids]),float(radii[ids].max())))

    def nearest_ids(self,query,*,max_distance=np.inf,cancelled=None,batch_size=32,query_normals=None,normal_limit=.2):
        query=np.asarray(query,dtype=np.float64)
        if query.ndim!=2 or query.shape[1]!=3 or not np.isfinite(query).all():raise ValueError('Invalid query positions')
        if max_distance<=0 or np.isnan(max_distance):raise ValueError('Surface search distance must be positive')
        query_normals=None if query_normals is None else np.asarray(query_normals,dtype=float)
        if query_normals is not None and (query_normals.shape!=query.shape or not np.isfinite(query_normals).all()):raise ValueError('Invalid query normals')
        if query_normals is not None:query_normals=query_normals/np.maximum(np.linalg.norm(query_normals,axis=1)[:,None],1e-300)
        if not np.isfinite(normal_limit) or not -1<=normal_limit<=1:raise ValueError('Normal limit must lie between -1 and one')
        result=np.full(len(query),-1,dtype=np.int64)
        for start in range(0,len(query),batch_size):
            if cancelled and cancelled():raise RuntimeError('Surface query cancelled')
            points=query[start:start+batch_size]
            upper=np.minimum(self.vertex_tree.query(points,k=1)[0],max_distance)
            normals=None if query_normals is None else query_normals[start:start+batch_size]
            if normals is not None:
                # A non-facing nearest vertex is not a valid upper bound for
                # the nearest facing triangle. Seed with actual facing facet
                # distances, then search every sphere capable of beating it.
                k=min(32,len(self.face_ids));seed=self.center_tree.query(points,k=k)[1].reshape(len(points),k)
                seed_tri=self.vertices[self.triangles[self.face_ids[seed]]]
                repeated=np.broadcast_to(points[:,None,:],(len(points),k,3)).reshape(-1,3)
                _,_,seed_distance=closest_triangle_points(repeated,seed_tri.reshape(-1,3,3))
                agreement=np.einsum('nki,ni->nk',self.normals[seed],normals)
                seed_distance=seed_distance.reshape(len(points),k)
                upper=np.minimum(np.min(np.where(agreement>=normal_limit,seed_distance,np.inf),axis=1),max_distance)
            candidates=[[] for _ in points]
            for ids,tree,radius in self.tiers:
                found=tree.query_ball_point(points,upper+radius+self.tolerance,workers=1)
                for i,selected in enumerate(found):
                    if len(selected):candidates[i].append(ids[np.asarray(selected,dtype=int)])
            for i,parts in enumerate(candidates):
                if not parts:continue
                ids=np.concatenate(parts);best=(np.inf,np.iinfo(np.int64).max)
                if normals is not None:
                    ids=ids[(self.normals[ids]@normals[i])>=normal_limit]
                    if not len(ids):continue
                # Bound temporary triangle arrays even for poor initialization.
                for offset in range(0,len(ids),32768):
                    local=ids[offset:offset+32768];global_ids=self.face_ids[local]
                    repeated=np.broadcast_to(points[i],(len(local),3))
                    _,_,distance=closest_triangle_points(repeated,self.vertices[self.triangles[global_ids]])
                    order=np.lexsort((global_ids,distance));index=order[0]
                    proposed=(float(distance[index]),int(global_ids[index]))
                    if proposed<best:best=proposed
                if best[0]<=max_distance+self.tolerance:result[start+i]=best[1]
        return result

    def query(self,query,*,max_distance=np.inf,query_normals=None,normal_limit=.2,cancelled=None):
        ids=self.nearest_ids(query,max_distance=max_distance,cancelled=cancelled,query_normals=query_normals,normal_limit=normal_limit)
        confidence=(ids>=0).astype(float)
        if query_normals is not None:
            normal=np.asarray(query_normals,dtype=float)
            if normal.shape!=np.asarray(query).shape:raise ValueError('Query normals must match positions')
            normal=normal/np.maximum(np.linalg.norm(normal,axis=1)[:,None],1e-300)
            valid=ids>=0
            faces=self.vertices[self.triangles[ids[valid]]]
            n=np.cross(faces[:,1]-faces[:,0],faces[:,2]-faces[:,0]);n/=np.maximum(np.linalg.norm(n,axis=1)[:,None],1e-300)
            agreement=np.einsum('ij,ij->i',n,normal[valid])
            confidence[valid]=np.where(agreement>=normal_limit,np.clip(agreement,0,1)**2,0)
        return witnesses(query,self.vertices,self.triangles,ids,confidence)

    def ray_query(self,origins,directions,*,max_distance,normal_limit=.2,selection='nearest',cancelled=None):
        """Nearest facing hit along a bounded signed normal line, in float64.

        Complete bounding-sphere candidates are filtered against the segment
        before triangle intersection. Original facet IDs and barycentrics are
        retained. A negative signed distance contracts an oversized surface;
        no tangential nearest-point slide or fixed-k candidate shortcut occurs.
        ``outer`` chooses the greatest signed facing hit and requires a source
        patch that excludes neighboring anatomical parts along the line.
        The selected surface and supplied directions must identify the intended
        anatomical wall; this is a local fitting primitive, not pose alignment.
        """
        origins=np.asarray(origins,dtype=float);directions=np.asarray(directions,dtype=float)
        if origins.ndim!=2 or origins.shape[1]!=3 or directions.shape!=origins.shape or not np.isfinite(origins).all() or not np.isfinite(directions).all():raise ValueError('Normal-line queries require matching finite N by 3 arrays')
        if not np.isfinite(max_distance) or max_distance<=0 or not np.isfinite(normal_limit) or not 0<=normal_limit<=1:raise ValueError('Choose positive finite ray distance and a facing threshold between zero and one')
        if selection not in {'nearest','outer'}:raise ValueError('Choose nearest or outer facing normal-line intersection')
        length=np.linalg.norm(directions,axis=1);unit=directions/np.maximum(length[:,None],1e-300)
        ids=np.full(len(origins),-1,dtype=np.int64);bary=np.zeros((len(origins),3));confidence=np.zeros(len(origins));signed=np.full(len(origins),np.nan)
        radius=max(tier[2] for tier in self.tiers)
        for start in range(0,len(origins),32):
            if cancelled and cancelled():raise RuntimeError('Normal-line projection cancelled')
            candidates=self.center_tree.query_ball_point(origins[start:start+32],max_distance+radius+self.tolerance)
            for offset,selected in enumerate(candidates):
                i=start+offset
                if not len(selected) or length[i]<1e-12:continue
                local=np.asarray(selected,dtype=int);center=self.center_tree.data[local]-origins[i]
                station=np.clip(center@unit[i],-max_distance,max_distance)
                keep=np.linalg.norm(center-station[:,None]*unit[i],axis=1)<=self.radii[local]+self.tolerance
                local=local[keep];agreement=self.normals[local]@unit[i];local=local[agreement>=normal_limit]
                if not len(local):continue
                gid=self.face_ids[local];t=self.vertices[self.triangles[gid]];e1=t[:,1]-t[:,0];e2=t[:,2]-t[:,0]
                p=np.cross(np.broadcast_to(unit[i],e2.shape),e2);det=np.einsum('ij,ij->i',e1,p)
                scale=np.linalg.norm(e1,axis=1)*np.linalg.norm(e2,axis=1);valid=abs(det)>np.maximum(scale*1e-14,1e-300)
                inv=np.divide(1.,det,out=np.zeros_like(det),where=valid);delta=origins[i]-t[:,0]
                u=np.einsum('ij,ij->i',delta,p)*inv;q=np.cross(delta,e1);v=(q@unit[i])*inv;distance=np.einsum('ij,ij->i',e2,q)*inv
                valid&=(u>=-1e-12)&(v>=-1e-12)&(u+v<=1+1e-12)&(abs(distance)<=max_distance+self.tolerance)
                if not valid.any():continue
                eligible=np.flatnonzero(valid);cost=abs(distance[eligible]) if selection=='nearest' else -distance[eligible]
                best=eligible[np.lexsort((gid[eligible],cost))[0]]
                value=np.clip([1-u[best]-v[best],u[best],v[best]],0.,1.);value/=value.sum()
                ids[i]=gid[best];bary[i]=value;confidence[i]=float(np.clip(self.normals[local[best]]@unit[i],0,1)**2);signed[i]=distance[best]
        observed=ids>=0;points=origins.copy();points[observed]=np.einsum('ni,nij->nj',bary[observed],self.vertices[self.triangles[ids[observed]]])
        residual=np.full(len(origins),np.nan);residual[observed]=np.linalg.norm(points[observed]-origins[observed],axis=1)
        return dict(vertex_ids=np.arange(len(origins),dtype=np.int64),source_triangle_ids=ids,barycentric=bary,
            surface_points=points,residual=residual,confidence=confidence,observed=observed,signed_distance=signed)

    def boundary_clearance(self,points,*,cutoff,cancelled=None):
        """Exact distance to selected-surface open edges, capped at cutoff.

        Coincident source vertices are welded for connectivity only, so UV
        seams are not misidentified as holes. Region selection edges also
        count as boundaries: neither is evidence for continuing skin beyond
        the observed patch. Closed surfaces return infinity.
        """
        points=np.asarray(points,dtype=float)
        if points.ndim!=2 or points.shape[1]!=3 or not np.isfinite(points).all():raise ValueError('Invalid boundary query')
        if not np.isfinite(cutoff) or cutoff<=0:raise ValueError('Choose finite positive boundary cutoff')
        if not hasattr(self,'_boundary_segments'):
            vertices,inverse=np.unique(self.vertices,axis=0,return_inverse=True)
            tri=inverse[self.triangles[self.face_ids]]
            edges=np.sort(np.vstack((tri[:,[0,1]],tri[:,[1,2]],tri[:,[2,0]])),axis=1)
            edges=edges[edges[:,0]!=edges[:,1]]
            edges,counts=np.unique(edges,axis=0,return_counts=True)
            segments=vertices[edges[counts==1]]
            self._boundary_segments=segments
            if len(segments):
                centers=segments.mean(1)
                self._boundary_tree=cKDTree(centers)
                self._boundary_radius=float(np.linalg.norm(segments[:,1]-segments[:,0],axis=1).max())*.5
        segments=self._boundary_segments
        if not len(segments):return np.full(len(points),np.inf)
        result=np.full(len(points),float(cutoff))
        for start in range(0,len(points),128):
            if cancelled and cancelled():raise RuntimeError('Source boundary query cancelled')
            batch=points[start:start+128]
            candidates=self._boundary_tree.query_ball_point(batch,cutoff+self._boundary_radius+self.tolerance)
            for i,ids in enumerate(candidates):
                if not len(ids):continue
                edge=segments[ids];direction=edge[:,1]-edge[:,0]
                t=np.clip(np.einsum('ij,ij->i',batch[i]-edge[:,0],direction)/np.maximum(np.einsum('ij,ij->i',direction,direction),1e-300),0,1)
                distance=np.linalg.norm(batch[i]-edge[:,0]-direction*t[:,None],axis=1)
                result[start+i]=min(cutoff,float(distance.min()))
        return result
