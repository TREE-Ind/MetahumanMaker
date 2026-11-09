"""Bounded sampled globe/skin occlusion correction for the numerical worker."""
import numpy as np
from scipy import sparse
from scipy.optimize import minimize,LinearConstraint,Bounds
from scipy.spatial import Delaunay


def constrain_hidden_globe(eye,triangles,head,head_triangles,hull,width):
    from .ocular_surface import FrontSurface
    eye=np.asarray(eye,dtype=float);triangles=np.asarray(triangles,dtype=np.int64)
    corners=eye[triangles];xz=corners[:,:,[0,2]]
    area=(xz[:,1,0]-xz[:,0,0])*(xz[:,2,1]-xz[:,0,1])-(xz[:,1,1]-xz[:,0,1])*(xz[:,2,0]-xz[:,0,0])
    front=triangles[area>width*width*1e-12]
    if not len(front):raise ValueError('Globe has no front-facing projected triangles')
    bary=np.array([[1/3,1/3,1/3],[.5,.5,0],[.5,0,.5],[0,.5,.5],
        [2/3,1/6,1/6],[1/6,2/3,1/6],[1/6,1/6,2/3]])
    points=np.einsum('sj,fjk->fsk',bary,eye[front]).reshape(-1,3)
    faces=np.repeat(front,len(bary),axis=0);weights=np.tile(bary,(len(front),1))
    edge=np.roll(hull,-1,axis=0)-hull;delta=points[:,None,:][:,:,[0,2]]-hull[None,:,:]
    inset=np.min((edge[None,:,0]*delta[:,:,1]-edge[None,:,1]*delta[:,:,0])/np.linalg.norm(edge,axis=1)[None,:],axis=1)
    outside=inset<-width*.001;points=points[outside];faces=faces[outside];weights=weights[outside]
    # A separate frontal grid catches boundary crossings between triangle
    # quadrature points. Retain original triangle/barycentric witnesses so the
    # depth constraints act on the actual exported mesh, without remeshing.
    lo=hull.min(0);hi=hull.max(0);span=hi-lo;lo-=span*.12;hi+=span*.12
    xx,zz=np.meshgrid(np.linspace(lo[0],hi[0],97),np.linspace(lo[1],hi[1],73));q=np.c_[xx.ravel(),zz.ravel()]
    delta=q[:,None,:]-hull[None,:,:];sd=np.min((edge[None,:,0]*delta[:,:,1]-edge[None,:,1]*delta[:,:,0])/np.linalg.norm(edge,axis=1)[None,:],axis=1)
    q=q[(sd<-width*.001)&(Delaunay(eye[:,[0,2]]).find_simplex(q)>=0)]
    grid_y,grid_faces,grid_bary=FrontSurface(eye,triangles).sample(q)
    points=np.vstack((points,np.c_[q[:,0],grid_y,q[:,1]]));faces=np.vstack((faces,triangles[grid_faces]));weights=np.vstack((weights,grid_bary))
    if not len(points):raise ValueError('No hidden globe samples are available for clearance inspection')
    skin=FrontSurface(head,head_triangles)(points[:,[0,2]])
    clearance=width*.0005;required=(skin+clearance-points[:,1])/width;violated=required>0
    output=eye.copy();maximum=0.;iterations=0
    if violated.any():
        ff=faces[violated];ww=weights[violated];rhs=required[violated]
        edges=np.unique(np.sort(np.vstack((triangles[:,[0,1]],triangles[:,[1,2]],triangles[:,[2,0]])),axis=1),axis=0)
        active=np.unique(ff)
        for _ in range(2):active=np.union1d(active,edges[np.any(np.isin(edges,active),axis=1)].ravel())
        if len(active)>2048:raise ValueError('Globe clearance region exceeds the bounded dense solve; reduce eye topology before fitting')
        lookup=np.full(len(eye),-1,dtype=int);lookup[active]=np.arange(len(active));n=len(active)
        A=sparse.coo_matrix((ww.ravel(),(np.repeat(np.arange(len(ff)),3),lookup[ff].ravel())),shape=(len(ff),n)).toarray()
        local=edges[np.all(np.isin(edges,active),axis=1)];local=lookup[local]
        adj=sparse.coo_matrix((np.ones(len(local)*2),(np.r_[local[:,0],local[:,1]],np.r_[local[:,1],local[:,0]])),shape=(n,n)).tocsr()
        lap=sparse.diags(np.asarray(adj.sum(1)).ravel())-adj;Q=(sparse.eye(n)+.1*(lap.T@lap)).toarray()
        result=minimize(lambda d:.5*float(d@Q@d),np.zeros(n),jac=lambda d:Q@d,
            method='SLSQP',bounds=Bounds(0.,.1),constraints=LinearConstraint(A,rhs,np.inf),
            options=dict(maxiter=100,ftol=1e-12))
        if not result.success or np.min(A@result.x-rhs)<-1e-8:raise ValueError('Could not clear the hidden globe within its bounded depth correction')
        output[active,1]+=result.x*width;maximum=float(result.x.max()*width);iterations=int(result.nit)
    actual=np.einsum('ij,ij->i',output[faces,1],weights)
    remaining=int(np.sum(actual<skin+clearance-width*1e-8))
    if remaining:raise ValueError('Hidden globe still violates sampled skin clearance')
    return output,dict(method='POSITIVE_DEPTH_QUADRATIC_OCCLUSION_V1',outside_samples=len(points),
        violated_before=int(violated.sum()),violated_after=remaining,maximum_depth_correction=maximum,
        clearance=clearance,iterations=iterations,frontal_grid=[97,73],grid_witnesses=len(q),
        limitation='Seven samples per front triangle plus a frontal grid outside the source outline; not a continuous collision certificate')
