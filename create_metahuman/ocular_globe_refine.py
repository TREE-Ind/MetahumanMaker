"""Observed depth refinement with frontal triangle-overlay lid clearance.

Uses marked source eye surfaces. Hidden geometry remains inferred. Overlay
constraints cover selected frontal triangle sheets; other grid constraints
remain sampled. Neither establishes whole-character motion safety.
"""
import time
import numpy as np
from scipy import sparse
from scipy.spatial import ConvexHull,Delaunay
from .ocular_surface import FrontSurface,globe_fidelity


def fit(data,options,*,progress=None,cancelled=None):
    if not options.get('source_patches'):raise ValueError('Paint both source eye patches before refining observed globe surfaces')
    basis=np.asarray(data['eye_basis']);src=np.asarray(data['source'])@basis;source_tri=np.asarray(data['source_triangles'])
    source=FrontSurface(src,source_tri);skin=FrontSurface(np.asarray(data['current'])@basis,np.asarray(data['triangles']))
    reports=[];proof={'before':data['current'],'planned':data['current'], 'ocular_movable':np.zeros(len(data['current']),dtype=bool)}
    def check_cancel():
        if cancelled and cancelled():raise RuntimeError('Eye refinement cancelled')
    for k in range(int(options['eye_count'])):
        check_cancel();mask=np.asarray(data['source_eye_faces_'+str(k)],dtype=bool)
        start=time.monotonic();eye=np.asarray(data['eye_points_'+str(k)])@basis;tri=np.asarray(data['eye_triangles_'+str(k)]);pts=src[np.unique(source_tri[mask])]
        if len(pts)<12:raise ValueError('Paint complete source eye patches with at least twelve vertices per eye')
        h=pts[ConvexHull(pts[:,[0,2]]).vertices][:,[0,2]];width=np.ptp(h[:,0]);edge=np.roll(h,-1,axis=0)-h
        def inset(q):
            d=q[:,None,:]-h[None,:,:]
            return np.min((edge[None,:,0]*d[:,:,1]-edge[None,:,1]*d[:,:,0])/np.linalg.norm(edge,axis=1)[None,:],axis=1)
        lo=h.min(0);hi=h.max(0);span=hi-lo;lo-=span*.12;hi+=span*.12
        x,z=np.meshgrid(np.linspace(lo[0],hi[0],97),np.linspace(lo[1],hi[1],73));q=np.c_[x.ravel(),z.ravel()];q=q[Delaunay(eye[:,[0,2]]).find_simplex(q)>=0]
        y,ids,bary=FrontSurface(eye,tri).sample(q);dist=inset(q);outside=dist<-width*.001;inside=dist>width*.03
        if inside.sum()<24:raise ValueError('The source eye patch and accepted globe do not share enough observed area; fit the eye opening first')
        corners=eye[tri];xz=corners[:,:,[0,2]];u=xz[:,1]-xz[:,0];v=xz[:,2]-xz[:,0];area=u[:,0]*v[:,1]-u[:,1]*v[:,0];front=tri[area>width*width*1e-12]
        qb=np.array([[1/3]*3,[.5,.5,0],[.5,0,.5],[0,.5,.5],[2/3,1/6,1/6],[1/6,2/3,1/6],[1/6,1/6,2/3]])
        quadr=np.einsum('sj,fjk->fsk',qb,eye[front]).reshape(-1,3);ff=np.repeat(front,len(qb),axis=0);ww=np.tile(qb,(len(front),1));keep=inset(quadr[:,[0,2]])<-width*.001
        hiddenq=np.vstack((q[outside],quadr[keep][:,[0,2]]));hiddeny=np.r_[y[outside],quadr[keep,1]];hiddenf=np.vstack((tri[ids[outside]],ff[keep]));hiddenb=np.vstack((bary[outside],ww[keep]))
        targets=(source(q[inside])-y[inside])/width;rhs=(skin(hiddenq)+width*.0005-hiddeny)/width
        from .ocular_overlay import anterior_lid_sheet,depth_constraints
        sheet,sheet_triangles,sheet_report=anterior_lid_sheet(data,k)
        overlay,overlay_rhs,overlay_report=depth_constraints(eye,tri,sheet,sheet_triangles,clearance=width*.0005,cancelled=cancelled)
        active=np.unique(np.r_[hiddenf.ravel(),tri[ids[inside]].ravel(),overlay.indices]);lookup=np.full(len(eye),-1);lookup[active]=np.arange(len(active));n=len(active)
        if n>2048 or len(hiddenf)>32768 or overlay.shape[0]>262144:raise ValueError('Observed globe refinement exceeds the bounded ocular solve; reduce globe topology first')
        def matrix(f,b):return sparse.coo_matrix((b.ravel(),(np.repeat(np.arange(len(f)),3),lookup[f].ravel())),shape=(len(f),n)).tocsr()
        A=sparse.vstack((matrix(hiddenf,hiddenb),overlay[:,active])).tocsc();rhs=np.r_[rhs,overlay_rhs/width];B=matrix(tri[ids[inside]],bary[inside])
        edges=np.unique(np.sort(np.vstack((tri[:,[0,1]],tri[:,[1,2]],tri[:,[2,0]])),axis=1),axis=0);edges=edges[np.all(lookup[edges]>=0,axis=1)];edges=lookup[edges]
        adj=sparse.coo_matrix((np.ones(len(edges)*2),(np.r_[edges[:,0],edges[:,1]],np.r_[edges[:,1],edges[:,0]])),shape=(n,n)).tocsr();L=sparse.diags(np.asarray(adj.sum(1)).ravel())-adj
        regularity=.1
        Q=(B.T@B+regularity*sparse.eye(n)+regularity*L.T@L).tocsc();c=np.asarray(B.T@targets).ravel()
        bound=.05
        import clarabel
        settings=clarabel.DefaultSettings();settings.verbose=False;settings.tol_feas=1e-9;settings.tol_gap_abs=1e-10;settings.tol_gap_rel=1e-9
        constraint=sparse.vstack((-A,sparse.eye(n),-sparse.eye(n))).tocsc();right=np.r_[-rhs,np.full(2*n,bound)]
        result=clarabel.DefaultSolver(sparse.triu(Q).tocsc(),-c,constraint,right,[clarabel.NonnegativeConeT(len(right))],settings).solve()
        check_cancel();offset=np.asarray(result.x)
        if str(result.status)!='Solved' or np.min(A@offset-rhs)<-1e-8:raise ValueError('Observed globe refinement did not converge with frontal triangle clearance; previous geometry retained')
        candidate=eye.copy();candidate[active,1]+=offset*width
        if cancelled and cancelled():raise RuntimeError('Eye refinement cancelled')
        before=globe_fidelity(eye,tri,source,h);after=globe_fidelity(candidate,tri,source,h)
        row=dict(eye=k,solver='OBSERVED_DEPTH_WITH_FRONTAL_TRIANGLE_OVERLAY',seconds=time.monotonic()-start,success=True,message=str(result.status),iterations=int(result.iterations),variables=n,observed_samples=int(inside.sum()),hidden_samples=len(rhs),hidden_min_slack=float(np.min(A@offset-rhs)),max_movement_world=float(np.max(abs(offset))*width),before=before,after=after,
            anterior_sheet=sheet_report,overlay={key:value for key,value in overlay_report.items() if not isinstance(value,np.ndarray)},
            clearance_scope='Selected anterior neutral skin/globe triangle depth; not a whole-motion collision certificate')
        if after['depth_rms_world']>before['depth_rms_world']:raise ValueError('Observed globe depth did not improve; previous geometry retained')
        from .ocular_shape import radial_recess
        row['globe_shape']=radial_recess(candidate)
        row['accepted']=True;proof['ocular_eye_'+str(k)]=candidate@basis.T;reports.append(row)
        if progress:progress('Refining observed eye surface '+str(k+1),(k+1)/options['eye_count'])
    return np.asarray(data['current']).copy(),reports,proof
