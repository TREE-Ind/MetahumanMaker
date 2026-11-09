"""Transport original angular limits along the immutable observed source.

The initial scope and minimum cosines stay fixed. Only the directions follow
fresh original-source facet witnesses; a coherent face cannot lose support.
"""
import numpy as np
from .vertex_contract import resolve_facet_ties,fingerprint,closest_triangle_points


def corner_normals(source,triangles):
    source=np.asarray(source);triangles=np.asarray(triangles)
    # Exact position aliases share a shading field; source coordinates and
    # original triangle identities remain untouched, including UV seams.
    _,weld=np.unique(source,axis=0,return_inverse=True);physical=weld[triangles]
    p=source[triangles];cross=np.cross(p[:,1]-p[:,0],p[:,2]-p[:,0]);unit=cross/np.maximum(np.linalg.norm(cross,axis=1)[:,None],1e-30)
    n=len(triangles);face=np.tile(np.arange(n),3)
    a=np.r_[physical[:,0],physical[:,1],physical[:,2]];b=np.r_[physical[:,1],physical[:,2],physical[:,0]]
    ca=np.r_[3*np.arange(n),3*np.arange(n)+1,3*np.arange(n)+2];cb=np.r_[3*np.arange(n)+1,3*np.arange(n)+2,3*np.arange(n)]
    swap=a>b;lo=np.minimum(a,b);hi=np.maximum(a,b);first=np.where(swap,cb,ca);second=np.where(swap,ca,cb)
    order=np.lexsort((hi,lo));lo=lo[order];hi=hi[order];first=first[order];second=second[order];face=face[order]
    starts=np.r_[0,np.flatnonzero((lo[1:]!=lo[:-1])|(hi[1:]!=hi[:-1]))+1];counts=np.diff(np.r_[starts,len(lo)])
    paired=starts[counts==2];smooth=np.sum(unit[face[paired]]*unit[face[paired+1]],axis=1)>=np.cos(np.pi/4);paired=paired[smooth]
    links=np.vstack((np.column_stack((first[paired],first[paired+1])),np.column_stack((second[paired],second[paired+1]))))
    labels=np.arange(n*3)
    for _ in range(64):
        old=labels.copy();np.minimum.at(labels,links[:,0],old[links[:,1]]);np.minimum.at(labels,links[:,1],old[links[:,0]]);labels=labels[labels]
        if np.array_equal(labels,old):break
    else:raise ValueError('Source corner-normal components did not converge')
    normal=np.zeros((n*3,3));np.add.at(normal,labels,np.repeat(cross,3,axis=0));normal/=np.maximum(np.linalg.norm(normal,axis=1)[:,None],1e-30)
    return normal[labels].reshape(n,3,3)


def directions(points,source,triangles,faces,target_triangles,*,smooth_normals=None,return_field=False):
    faces=np.asarray(faces,dtype=np.int64);field=np.zeros((len(faces),3),dtype=float);valid=faces>=0
    source=np.asarray(source);triangles=np.asarray(triangles);facets=triangles[faces[valid]];p=source[facets]
    if smooth_normals is None:smooth_normals=corner_normals(source,triangles)
    corners=np.asarray(smooth_normals)[faces[valid]]
    _,bary,_=closest_triangle_points(np.asarray(points)[valid],p)
    blended=np.einsum('ni,nij->nj',bary,corners)
    field[valid]=blended/np.maximum(np.linalg.norm(blended,axis=1)[:,None],1e-30)
    summed=field[np.asarray(target_triangles)].sum(1);size=np.linalg.norm(summed,axis=1)
    result=(summed/np.maximum(size[:,None],1e-30),np.all(valid[np.asarray(target_triangles)],axis=1)&(size>=2.4))
    return (*result,field,valid) if return_field else result


def rebind(points,target_triangles,source,source_triangles,before_faces,after_faces,original,scope,*,smooth_normals=None,reference=None,branch_caps=None,faces_only=False,return_support=False):
    faces=np.asarray(after_faces,dtype=np.int64).copy();ids=np.flatnonzero((np.asarray(before_faces)>=0)&(faces>=0))
    faces[ids],_=resolve_facet_ties(np.asarray(points)[ids],source,source_triangles,np.asarray(before_faces)[ids],faces[ids])
    if reference is not None and branch_caps is not None:
        active=ids[(np.asarray(branch_caps)[ids]>0)&(faces[ids]!=np.asarray(before_faces)[ids])]
        if len(active):
            old_facets=np.asarray(source)[np.asarray(source_triangles)[np.asarray(before_faces)[active]]]
            old_origin,_,_=closest_triangle_points(np.asarray(reference)[active],old_facets)
            old_now,_,distance=closest_triangle_points(np.asarray(points)[active],old_facets)
            new_now,_,_=closest_triangle_points(np.asarray(points)[active],np.asarray(source)[np.asarray(source_triangles)[faces[active]]])
            motion=np.linalg.norm(np.asarray(points)[active]-np.asarray(reference)[active],axis=1)
            old_motion=np.linalg.norm(old_now-old_origin,axis=1);jump=np.linalg.norm(new_now-old_origin,axis=1)
            epsilon=max(float(np.linalg.norm(np.ptp(reference,axis=0)))*1e-10,1e-14)
            retain=(jump>4*np.maximum(motion,old_motion)+epsilon)&(distance<=np.asarray(branch_caps)[active])
            faces[active[retain]]=np.asarray(before_faces)[active[retain]]
    if faces_only:return faces
    normals,coherent,field,observed=directions(points,source,source_triangles,faces,target_triangles,smooth_normals=smooth_normals,return_field=True)
    # Curved source patches can spread the three observed directions beyond
    # the initial mean-coherence cutoff while every direction still certifies
    # an outward-facing target. Retain the original scope in that case; never
    # drop a face or lower its original angular floor.
    lost=np.asarray(scope)&~coherent
    margin=None
    if return_support:
        observed_faces=np.all(observed[np.asarray(target_triangles)],axis=1)
        margin=np.where(observed_faces,np.linalg.norm(field[np.asarray(target_triangles)].sum(1),axis=1)-2.4,-1.)
    if np.any(lost):
        tri=np.asarray(target_triangles)[lost];p=np.asarray(points)[tri]
        cross=np.cross(p[:,1]-p[:,0],p[:,2]-p[:,0]);size=np.linalg.norm(cross,axis=1)
        unit=cross/np.maximum(size[:,None],1e-30)
        outward=(np.all(observed[tri],axis=1)&(size>0)&
                 np.all(np.einsum('fki,fi->fk',field[tri],unit)>=.25,axis=1))
        coherent[lost]=outward
        if return_support:
            corner_margin=np.min(np.einsum('fki,fi->fk',field[tri],unit),axis=1)-.25
            corner_margin=np.where(np.all(observed[tri],axis=1)&(size>0),corner_margin,-1.)
            margin[lost]=np.maximum(margin[lost],corner_margin)
    if return_support:return dict(original,normals=normals[original['face_ids']]),normals,coherent,margin
    if np.any(np.asarray(scope)&~coherent):raise ValueError('Source-following surface loses its original coherent observation scope')
    return dict(original,normals=normals[original['face_ids']]),normals


def original_contract(reference,target_triangles,source,source_triangles,before_faces,eligible,movable,metric,*,smooth_normals=None):
    from .source_angle_contract import angle_contract
    faces=np.asarray(before_faces).copy();faces[~np.asarray(eligible,dtype=bool)]=-1
    normals,coherent=directions(reference,source,source_triangles,faces,target_triangles,smooth_normals=smooth_normals)
    scope=coherent&np.any(np.asarray(movable)[np.asarray(target_triangles)],axis=1)
    return normals,scope,angle_contract(reference,target_triangles,normals,scope,metric,preserve_aligned=True)


class FollowingField:
    def __init__(self,source,source_triangles,triangles,before_faces,ids,query,original,scope,policy_faces,reference,branch_caps):
        self.source=source;self.source_triangles=source_triangles;self.triangles=triangles;self.before_faces=before_faces.copy();self.ids=ids.copy();self.query=query;self.original=original;self.scope=scope.copy()
        self.policy=set(int(face) for face in policy_faces)
        self.smooth_normals=corner_normals(source,source_triangles)
        self.reference=reference.copy();self.branch_caps=branch_caps.copy()
        if any(int(face) not in self.policy for face in self.before_faces[self.ids]):raise ValueError('Original source branch leaves its paired facet policy')
        self.contract={key:fingerprint(value) for key,value in dict(source=source,source_triangles=source_triangles,triangles=triangles,before_faces=before_faces,ids=ids,scope=scope,policy_faces=np.asarray(policy_faces,dtype=np.int64),reference=reference,branch_caps=branch_caps).items()}

    def __call__(self,points):
        return self._bind(points)

    def inspect(self,points):
        """Measure the same support predicate without certifying the endpoint."""
        return self._bind(points,return_support=True)

    def flat_directions(self,points,*,refresh=True,return_field=False):
        """Original-facet field used by the independent alignment measure."""
        if not hasattr(self,'_flat_source_normals'):
            p=self.source[self.source_triangles];cross=np.cross(p[:,1]-p[:,0],p[:,2]-p[:,0])
            self._flat_source_normals=cross/np.maximum(np.linalg.norm(cross,axis=1)[:,None],1e-30)
        faces=self.before_faces.copy();observed=faces>=0
        if refresh:
            chosen=self.query(points[self.ids])['source_triangle_ids']
            faces[self.ids],_=resolve_facet_ties(points[self.ids],self.source,self.source_triangles,faces[self.ids],chosen)
        vertex=np.zeros_like(points);vertex[observed]=self._flat_source_normals[faces[observed]]
        summed=vertex[self.triangles].sum(1);size=np.linalg.norm(summed,axis=1)
        p=points[self.triangles];C=np.cross(p[:,1]-p[:,0],p[:,2]-p[:,0]);area=np.linalg.norm(C,axis=1)
        outward=(area>0)&np.all(np.einsum('fki,fi->fk',vertex[self.triangles],C/np.maximum(area[:,None],1e-30))>=.25,axis=1)
        result=(summed/np.maximum(size[:,None],1e-30),np.all(observed[self.triangles],axis=1)&((size>=2.4)|outward))
        return (*result,vertex,observed) if return_field else result

    def _bind(self,points,*,return_support=False):
        chosen=self.query(points[self.ids])['source_triangle_ids']
        if any(int(face) not in self.policy for face in chosen):raise ValueError('Source-field transport leaves the original facet policy')
        faces=self.before_faces.copy();faces[self.ids]=chosen
        return rebind(points,self.triangles,self.source,self.source_triangles,self.before_faces,faces,self.original,self.scope,smooth_normals=self.smooth_normals,reference=self.reference,branch_caps=self.branch_caps,return_support=return_support)

    def linearize(self,points,free,contract,*,linearization_points=None):
        from .source_field_derivatives import transported_model
        faces,normals,D=self._derivatives(points)
        return transported_model(points,self.triangles,free,contract,faces,self.source,self.source_triangles,self.smooth_normals,linearization_points=linearization_points,field_derivatives=(normals,D))

    def _derivatives(self,points,*,vertex_field=False):
        from .source_field_derivatives import normal_jacobians
        cache=getattr(self,'_derivative_cache',None)
        if cache is not None and np.array_equal(cache[0],points):return cache[1:] if vertex_field else cache[1:4]
        chosen=self.query(points[self.ids])['source_triangle_ids']
        if any(int(face) not in self.policy for face in chosen):raise ValueError('Source-field linearization leaves the original facet policy')
        faces=self.before_faces.copy();faces[self.ids]=chosen
        faces=rebind(points,self.triangles,self.source,self.source_triangles,self.before_faces,faces,self.original,self.scope,reference=self.reference,branch_caps=self.branch_caps,faces_only=True)
        normals,D,field,J=normal_jacobians(points,self.source,self.source_triangles,faces,self.triangles,self.smooth_normals,return_vertex_field=True)
        self._derivative_cache=(np.array(points,copy=True),faces,normals,D,field,J)
        return (faces,normals,D,field,J) if vertex_field else (faces,normals,D)

    def signed_gradient_correction(self,points):
        _,_,D=self._derivatives(points)
        p=np.asarray(points)[self.triangles];area=np.cross(p[:,1]-p[:,0],p[:,2]-p[:,0])
        return np.einsum('fi,fkij->fkj',area,D)
