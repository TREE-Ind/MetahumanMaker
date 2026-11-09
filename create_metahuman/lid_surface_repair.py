"""Experimental, bounded repair of one captured frontal lid pose.

Inputs use canonical native vertex identities and X-right/Z-up/negative-Y-front
coordinates in the same physical units. Supply actual render triangles and a
visible-aperture cut derived on those triangles. This is an endpoint proposal,
not a compiled-motion, global injectivity or continuous-collision certificate.
The smooth-determinant objective is independently implemented from the geometric
ideas described at https://arxiv.org/abs/2102.03069 . No reference code is bundled.
"""
import numpy as np
from scipy import sparse
from scipy.optimize import minimize
from scipy.sparse.csgraph import dijkstra
from .ocular_lid_contact import _hull_triangles
from .ocular_motion import continue_socket
from .ocular_overlay import depth_constraints
from .surface_contacts import crossings


def _normal(points,faces):
    return np.cross(points[faces[:,1]]-points[faces[:,0]],points[faces[:,2]]-points[faces[:,0]])


def _edges(faces):
    return np.unique(np.sort(np.vstack((faces[:,[0,1]],faces[:,[1,2]],faces[:,[2,0]])),axis=1),axis=0,return_counts=True)


def _region(neutral,sheet,margin,width,halo_fraction):
    edges,_=_edges(sheet);length=np.linalg.norm(neutral[edges[:,0]]-neutral[edges[:,1]],axis=1)
    if np.any(length<=0):raise ValueError('The native exterior contains collapsed edges')
    graph=sparse.coo_matrix((np.r_[length,length],(np.r_[edges[:,0],edges[:,1]],np.r_[edges[:,1],edges[:,0]])),shape=(len(neutral),len(neutral))).tocsr()
    distance=dijkstra(graph,directed=False,indices=margin,min_only=True,limit=width*halo_fraction);domain=np.flatnonzero(distance<=width*halo_fraction)
    faces=sheet[np.all(np.isin(sheet,domain),axis=1)]
    if not len(faces) or not np.isin(margin,np.unique(faces)).all():raise ValueError('The physical collar does not contain the complete lid margin')
    edges,counts=_edges(faces);outer=np.setdiff1d(np.unique(edges[counts==1]),margin)
    if not len(outer):raise ValueError('The native lid collar has no fixed outer boundary')
    return faces,outer,edges


def inspect_pose(neutral,posed,triangles,sheet,margin,interior,eye,*,halo_fraction=.45):
    """Inspect one sealed native pose before deciding whether it needs repair.

    Uses the repair's full collar/socket scope, not only displaced vertices.
    Passing this sampled endpoint does not certify the motion between samples.
    """
    width=float(np.ptp(eye[:,0]))
    if not np.isfinite(width) or width<=0:raise ValueError('The captured globe has no physical width')
    faces,_,_=_region(neutral,sheet,margin,width,halo_fraction)
    active=np.any(np.isin(triangles,np.union1d(np.unique(faces),interior)),axis=1)
    contacts=crossings(posed,triangles,active_faces=active)
    reversed_count=int((np.einsum('ij,ij->i',_normal(neutral,faces),_normal(posed,faces))<=0).sum())
    _,deficit,_=depth_constraints(eye,_hull_triangles(eye),posed,faces,clearance=0.)
    clearance=float(-deficit.max());simple=_simple_boundary(posed[margin][:,[0,2]])
    return dict(valid=contacts==0 and reversed_count==0 and clearance>=0 and simple,proper_crossings=contacts,reversed_triangles=reversed_count,clearance=clearance,simple_boundary=bool(simple),native_motion_accepted=False)


def _curve(points,margin):
    edge=np.linalg.norm(points[np.roll(margin,-1)]-points[margin],axis=1);previous=np.roll(edge,1)
    if np.any(edge<=0):raise ValueError('The native aperture has a collapsed boundary edge')
    spacing=float(np.median(edge));a=2*spacing*spacing/(previous*(previous+edge));b=2*spacing*spacing/(edge*(previous+edge))
    columns=np.column_stack((np.roll(margin,1),margin,np.roll(margin,-1)))
    return sparse.coo_matrix((np.column_stack((a,-a-b,b)).ravel(),(np.repeat(np.arange(len(margin)),3),columns.ravel())),shape=(len(margin),len(points))).tocsr()


def _simple_boundary(rim):
    following=np.roll(rim,-1,axis=0);n=len(rim)
    def cross(a,b):return a[...,0]*b[...,1]-a[...,1]*b[...,0]
    for i in range(n):
        ids=np.arange(i+2,n)
        if i==0:ids=ids[ids!=n-1]
        if not len(ids):continue
        a,b=rim[i],following[i];c,d=rim[ids],following[ids]
        if np.any((cross(b-a,c-a)*cross(b-a,d-a)<0)&(cross(d-c,a-c)*cross(d-c,b-c)<0)):return False
    return True


def _rim(reference,target,margin,fixed,width,curve,limit,check):
    """Fair an ordered aperture while retaining its neutral edge directions."""
    import clarabel
    r=reference[margin][:,[0,2]]/width;p=target[margin][:,[0,2]]/width
    n=len(margin);following=np.roll(np.arange(n),-1);edge=r[following]-r;length=np.linalg.norm(edge,axis=1)
    if np.any(length<=1e-12):raise ValueError('The native aperture has a collapsed frontal boundary edge')
    edge/=length[:,None]
    D=sparse.coo_matrix((np.r_[-np.ones(n),np.ones(n)],(np.r_[np.arange(n),np.arange(n)],np.r_[np.arange(n),following])),shape=(n,n)).tocsr()
    G=sparse.hstack((sparse.diags(edge[:,0])@D,sparse.diags(edge[:,1])@D)).tocsr()
    C=curve[:,margin];H=(100*sparse.eye(n)+10*C.T@C).tocsc();Q=sparse.block_diag((H,H)).tocsc();q=np.r_[-100*p[:,0],-100*p[:,1]]
    pins=np.flatnonzero(np.isin(margin,fixed));constant=np.r_[pins,pins+n];free=np.setdiff1d(np.arange(2*n),constant);z=p.T.flatten()
    identity=sparse.eye(2*n,format='csr');A=sparse.vstack((-G,identity,-identity)).tocsc()
    bounds=np.r_[-.02*length,z+limit/np.sqrt(2),-z+limit/np.sqrt(2)]-A[:,constant]@z[constant]
    settings=clarabel.DefaultSettings();settings.verbose=False;settings.tol_feas=1e-10;settings.tol_gap_abs=1e-10
    solved=clarabel.DefaultSolver(sparse.triu(Q[free][:,free]).tocsc(),q[free]+Q[free][:,constant]@z[constant],A[:,free],bounds,[clarabel.NonnegativeConeT(len(bounds))],settings).solve();check()
    if str(solved.status) not in {'Solved','AlmostSolved'}:raise ValueError('The bounded aperture edge-order constraints are infeasible; retain the preceding rig')
    x=np.asarray(solved.x)
    if not np.isfinite(x).all() or np.min(bounds-A[:,free]@x)<-1e-8:raise ValueError('The aperture boundary failed its actual constraint residuals')
    z[free]=x;rim=z.reshape(2,n).T
    # Edge order alone is not a simplicity certificate: independently reject
    # crossings between nonadjacent boundary segments before collar solving.
    if not _simple_boundary(rim):raise ValueError('The bounded aperture boundary still crosses; retain the preceding rig')
    answer=target.copy();answer[np.ix_(margin,[0,2])]=rim*width
    return answer,dict(method='BOUNDED_ANATOMICAL_EDGE_ORDER',status=str(solved.status),minimum_edge_ratio=float(np.min((G@z)/length)),maximum_movement=float(np.linalg.norm(rim-p,axis=1).max()*width),proper_boundary_crossings=0)


def _refine_projection(reference,origin,points,faces,fixed,margin,limit,check,progress):
    """Joint rim/collar SQP fallback with unchanged physical box constraints.

    A fixed rim can make its adjacent triangle fan impossible to unfold. This
    step frees the rim except for the canthi, constrains anatomical edge order,
    and linearizes all signed triangle areas. Actual areas and aperture
    simplicity govern step acceptance; linear feasibility alone never passes.
    Coordinates are already normalized by physical globe width.
    """
    import clarabel
    p=points.copy();n=len(p);vertices=np.unique(faces);free=np.setdiff1d(vertices,fixed);columns=(free[:,None]*2+np.arange(2)).ravel()
    def cross(a,b):return a[...,0]*b[...,1]-a[...,1]*b[...,0]
    area=cross(reference[faces[:,1]]-reference[faces[:,0]],reference[faces[:,2]]-reference[faces[:,0]])
    def ratios(x):return cross(x[faces[:,1]]-x[faces[:,0]],x[faces[:,2]]-x[faces[:,0]])/area
    def merit(x):return float(np.sum(np.maximum(.1-ratios(x),0)**2))
    edges,_=_edges(faces);adj=sparse.coo_matrix((np.ones(2*len(edges)),(np.r_[edges[:,0],edges[:,1]],np.r_[edges[:,1],edges[:,0]])),shape=(n,n)).tocsr();lap=(sparse.diags(np.asarray(adj.sum(1)).ravel())-adj)[vertices]
    K=sparse.kron(lap[:,free],sparse.eye(2),format='csc');L=sparse.kron(lap,sparse.eye(2),format='csc');Q=(K.T@K+.01*sparse.eye(len(columns))).tocsc()
    following=np.roll(margin,-1);edge=reference[following]-reference[margin];length=np.linalg.norm(edge,axis=1);edge/=length[:,None]
    E=sparse.coo_matrix((np.stack((-edge,edge),axis=1).ravel(),(np.repeat(np.arange(len(margin)),4),(np.stack((margin,following),axis=1)[:,:,None]*2+np.arange(2)).ravel())),shape=(len(margin),2*n)).tocsc();Ef=E[:,columns]
    lower=origin[free]-limit/np.sqrt(2);upper=origin[free]+limit/np.sqrt(2);identity=sparse.eye(len(columns),format='csc');history=[]
    for iteration in range(12):
        check();ratio=ratios(p)
        if np.all(ratio>=.099999):break
        progress('Refining the aperture and collar together')
        a=p[faces[:,1]]-p[faces[:,0]];b=p[faces[:,2]]-p[faces[:,0]];g1=np.column_stack((b[:,1],-b[:,0]));g2=np.column_stack((-a[:,1],a[:,0]));gradient=np.stack((-g1-g2,g1,g2),axis=1)/area[:,None,None]
        G=sparse.coo_matrix((gradient.ravel(),(np.repeat(np.arange(len(faces)),6),(faces[:,:,None]*2+np.arange(2)).ravel())),shape=(len(faces),2*n)).tocsc()[:,columns]
        q=np.asarray(K.T@(L@(p-origin).ravel())).ravel()+.01*(p[free]-origin[free]).ravel()
        A=sparse.vstack((-G,-Ef,identity,-identity)).tocsc();bounds=np.r_[ratio-.1,E@p.ravel()-.02*length,(upper-p[free]).ravel(),(p[free]-lower).ravel()]
        settings=clarabel.DefaultSettings();settings.verbose=False;settings.tol_feas=1e-9;settings.tol_gap_abs=1e-10;settings.tol_gap_rel=1e-9
        solved=clarabel.DefaultSolver(sparse.triu(Q).tocsc(),q,A,bounds,[clarabel.NonnegativeConeT(len(bounds))],settings).solve();check()
        if str(solved.status) not in {'Solved','AlmostSolved'}:raise ValueError('The joint aperture/collar constraints are infeasible; retain the preceding rig')
        x=np.asarray(solved.x)
        if not np.isfinite(x).all() or np.min(bounds-A@x)<-1e-8:raise ValueError('The joint aperture/collar step failed its actual constraint residuals')
        step=x.reshape(-1,2);old=merit(p);alpha=1.
        while alpha>=1/1024:
            check();candidate=p.copy();candidate[free]+=alpha*step
            if merit(candidate)<old and _simple_boundary(candidate[margin]):break
            alpha*=.5
        if alpha<1/1024:raise ValueError('The bounded joint aperture/collar refinement stalled; retain the preceding rig')
        p=candidate;ratio=ratios(p);history.append(dict(status=str(solved.status),step_fraction=alpha,reversed=int((ratio<=0).sum()),minimum_ratio=float(ratio.min())))
    ratio=ratios(p)
    if np.any(ratio<=1e-6) or not _simple_boundary(p[margin]) or np.any(p[free]<lower-1e-10) or np.any(p[free]>upper+1e-10):raise ValueError('The bounded joint aperture/collar still folds; retain the preceding rig')
    return p,free,dict(method='JOINT_APERTURE_COLLAR_AREA_SQP',iterations=len(history),steps=history,reversed=0,minimum_ratio=float(ratio.min()),minimum_edge_ratio=float(np.min((E@p.ravel())/length)),proper_boundary_crossings=0)


def _project(reference,target,faces,fixed,width,curve,margin,limit,check,progress):
    source_origin=target[:,[0,2]].copy()/width;anchors=fixed.copy()
    target,rim_report=_rim(reference,target,margin,fixed,width,curve,limit,check)
    fixed=np.union1d(fixed,margin)
    r=reference[:,[0,2]]/width;p=target[:,[0,2]].copy()/width;origin=p.copy()
    free=np.setdiff1d(np.unique(faces),fixed)
    if not len(free):raise ValueError('The lid has no movable projected surface')
    dm=np.stack((r[faces[:,1]]-r[faces[:,0]],r[faces[:,2]]-r[faces[:,0]]),axis=2)
    if np.any(abs(np.linalg.det(dm))<1e-16):raise ValueError('The neutral lid has a collapsed projection; retain the pose for manual review')
    inverse=np.linalg.inv(dm);weight=abs(np.linalg.det(dm));weight/=weight.sum();history=[]
    def evaluate(x,epsilon):
        check();p[free]=x.reshape(-1,2)
        ds=np.stack((p[faces[:,1]]-p[faces[:,0]],p[faces[:,2]]-p[faces[:,0]]),axis=2)
        j=ds@inverse;det=np.linalg.det(j);trace=np.sum(j*j,axis=(1,2));h=np.hypot(det,epsilon)
        chi=(det+h)*.5;negative=det<0;chi[negative]=epsilon*epsilon/(2*(h[negative]-det[negative]))
        numerator=trace+det*det+1.;delta=p[free]-origin[free];rim_delta=p[margin]-origin[margin];curvature=curve@p
        energy=float(np.sum(weight*numerator/chi)+100*np.sum(delta*delta)/len(free)+100*np.sum(rim_delta*rim_delta)/len(margin)+10*np.sum(curvature*curvature)/len(margin))
        cofactor=np.empty_like(j);cofactor[:,0,0]=j[:,1,1];cofactor[:,1,1]=j[:,0,0];cofactor[:,0,1]=-j[:,1,0];cofactor[:,1,0]=-j[:,0,1]
        coefficient=2*det/chi-numerator*(.5*(1+det/h))/(chi*chi)
        derivative=weight[:,None,None]*(2*j/chi[:,None,None]+coefficient[:,None,None]*cofactor)
        columns=derivative@np.swapaxes(inverse,1,2);gradient=np.zeros_like(p)
        np.add.at(gradient,faces[:,1],columns[:,:,0]);np.add.at(gradient,faces[:,2],columns[:,:,1]);np.add.at(gradient,faces[:,0],-columns.sum(2))
        gradient[free]+=200*delta/len(free);gradient[margin]+=200*rim_delta/len(margin);gradient+=20*(curve.T@curvature)/len(margin)
        if not np.isfinite(energy) or not np.isfinite(gradient).all():raise ValueError('The projected lid objective became nonfinite')
        return energy,gradient[free].ravel()
    x=p[free].ravel().copy();bounds=np.stack((x-limit/np.sqrt(2),x+limit/np.sqrt(2)),axis=1)
    for epsilon in (.1,.03,.01,.003,.001,.0001):
        progress('Untangling the visible lid projection')
        result=minimize(evaluate,x,args=(epsilon,),jac=True,method='L-BFGS-B',bounds=bounds,options=dict(maxiter=450,ftol=1e-12,gtol=1e-8,maxls=40,maxcor=20))
        x=result.x;evaluate(x,epsilon)
        ds=np.stack((p[faces[:,1]]-p[faces[:,0]],p[faces[:,2]]-p[faces[:,0]]),axis=2);ratio=np.linalg.det(ds@inverse)
        history.append(dict(epsilon=epsilon,iterations=int(result.nit),reversed=int((ratio<=0).sum()),minimum_ratio=float(ratio.min())))
    if history[-1]['reversed'] or history[-1]['minimum_ratio']<=1e-6:
        p,free,refinement=_refine_projection(r,source_origin,p,faces,anchors,margin,limit,check,progress);history.append(refinement)
    history[0]['aperture_boundary']=rim_report
    answer=target.copy();answer[np.ix_(free,[0,2])]=p[free]*width
    return answer,history


def _conic_depth_margin(Q,q,G,B,deficit,upper_depth,limit,p,faces,free,
                        initial_dot,base_dot,row_scale,movable_face,uncertainty):
    """Solve the orientation error bound against final edge lengths directly.

    With X/Z fixed, the normal dot product is affine in vertex Y. Each final
    edge length is a norm of an affine Y difference and fixed X/Z length.
    Auxiliary second-order cones bound those norms, giving the same native
    position-error constraint without a fixed-point approximation.
    """
    import clarabel
    n=len(free);selected=np.flatnonzero(movable_face);m=len(selected)
    if not n or not m:raise ValueError('No movable lid faces for the conic depth solve')
    local=faces[selected];lookup=np.full(len(p),-1,dtype=int);lookup[free]=np.arange(n)
    differences=[];edge_values=[]
    for corner in (1,2):
        columns=lookup[local[:,[0,corner]]].ravel();rows=np.repeat(np.arange(m),2);weights=np.tile([-1.,1.],m);keep=columns>=0
        differences.append(sparse.coo_matrix((weights[keep],(rows[keep],columns[keep])),shape=(m,n)).tocsr())
        edge_values.append(p[local[:,corner]]-p[local[:,0]])
    D=sparse.vstack(differences,format='coo');edges=np.vstack(edge_values);count=2*m
    extra=sparse.coo_matrix((np.full(m,2*uncertainty),(selected,np.arange(m))),shape=(len(faces),m)).tocsr()
    orientation=sparse.hstack((-G,extra,extra),format='csc')
    zero=sparse.csc_matrix((B.shape[0],count));identity=sparse.hstack((sparse.eye(n),sparse.csc_matrix((n,count))),format='csc')
    nonnegative=sparse.vstack((orientation,sparse.hstack((B,zero)),identity,-identity),format='csc')
    bounds=np.r_[(initial_dot-base_dot)*row_scale-movable_face*4*uncertainty**2,-deficit,upper_depth,np.full(n,limit)]
    soc=sparse.coo_matrix((np.r_[-np.ones(count),-D.data],
        (np.r_[3*np.arange(count),3*D.row+1],np.r_[n+np.arange(count),D.col])),shape=(3*count,n+count)).tocsc()
    soc_bounds=np.column_stack((np.zeros(count),edges[:,1],np.linalg.norm(edges[:,[0,2]],axis=1))).ravel()
    A=sparse.vstack((nonnegative,soc),format='csc');b=np.r_[bounds,soc_bounds]
    P=sparse.block_diag((Q,sparse.csc_matrix((count,count))),format='csc');objective=np.r_[q,np.zeros(count)]
    settings=clarabel.DefaultSettings();settings.verbose=False;settings.tol_feas=1e-10;settings.tol_gap_abs=1e-10;settings.tol_gap_rel=1e-10
    solved=clarabel.DefaultSolver(sparse.triu(P).tocsc(),objective,A,b,
        [clarabel.NonnegativeConeT(len(bounds)),*[clarabel.SecondOrderConeT(3) for _ in range(count)]],settings).solve()
    if str(solved.status) not in {'Solved','AlmostSolved'}:raise ValueError('The final-edge lid safety constraints are infeasible; retain the preceding rig')
    solution=np.asarray(solved.x);slack=b-A@solution;cone=slack[len(bounds):].reshape(-1,3)
    residual=min(float(slack[:len(bounds)].min()),float((cone[:,0]-np.linalg.norm(cone[:,1:],axis=1)).min()))
    if not np.isfinite(solution).all() or residual<-1e-8:raise ValueError('The conic lid depth proposal failed its actual constraint residuals')
    return solution[:n],dict(method='FINAL_EDGE_SECOND_ORDER_ORIENTATION_BOUND',status=str(solved.status),
        edge_cones=count,minimum_constraint_slack=residual,iterations=int(solved.iterations))


def repair(neutral,posed,triangles,sheet,margin,interior,eye,*,halo_fraction=.45,maximum_fraction=.35,cancelled=None,progress=None,preserve_projection=False):
    """Return a checked pose proposal and proof; never mutate caller arrays.

    The surrounding collar and frontal canthus positions are fixed. A canthus
    already inside the globe can move forward in depth to clear it, within the
    same physical bounds. A smooth-determinant
    projection objective and a constrained depth solve repair the visible sheet;
    the native neutral socket supplies its hidden continuation. Final checks use
    float32 positions, the full head for proper crossing pairs involving changed
    faces, and a freshly computed frontal globe overlay. Coplanar overlaps and
    boundary-only contact are outside the proper-crossing test's scope.
    """
    try:import clarabel
    except ImportError:raise ValueError('Install clarabel in the configured inference Python before using captured-lid untangling') from None
    neutral=np.asarray(neutral,dtype=float);posed=np.asarray(posed,dtype=float);eye=np.asarray(eye,dtype=float)
    triangles=np.asarray(triangles);sheet=np.asarray(sheet);margin=np.asarray(margin);interior=np.asarray(interior)
    if neutral.ndim!=2 or neutral.shape[1]!=3 or posed.shape!=neutral.shape or eye.ndim!=2 or eye.shape[1]!=3 or not all(np.isfinite(v).all() for v in (neutral,posed,eye)):
        raise ValueError('Supply finite corresponding native head and globe positions')
    for faces in (triangles,sheet):
        if faces.ndim!=2 or faces.shape[1]!=3 or not len(faces) or not np.issubdtype(faces.dtype,np.integer) or faces.min()<0 or faces.max()>=len(neutral):raise ValueError('Supply canonical native render triangles')
    for ids in (margin,interior):
        if ids.ndim!=1 or not len(ids) or not np.issubdtype(ids.dtype,np.integer) or ids.min()<0 or ids.max()>=len(neutral) or len(np.unique(ids))!=len(ids):raise ValueError('Supply distinct canonical aperture and socket vertices')
    if len(margin)<4 or not np.isin(margin,interior).all():raise ValueError('The socket must contain the complete ordered native lid margin')
    if not np.isfinite([halo_fraction,maximum_fraction]).all() or not 0<halo_fraction<=.5 or not 0<maximum_fraction<=.35:raise ValueError('Choose a bounded physical collar and at most 35 percent movement')
    keys={tuple(sorted(map(int,t))) for t in triangles}
    if any(tuple(sorted(map(int,t))) not in keys for t in sheet):raise ValueError('Derive lid anatomy on the captured render tessellation, not DNA diagonals')
    def check():
        if cancelled and cancelled():raise RuntimeError('Lid repair canceled; preceding rig retained')
    notify=progress or (lambda message:None);check();width=float(np.ptp(eye[:,0]))
    if width<=0:raise ValueError('The captured globe has no physical width')
    faces,outer,edges=_region(neutral,sheet,margin,width,halo_fraction)
    canthi=margin[[np.argmin(neutral[margin,0]),np.argmax(neutral[margin,0])]];fixed=np.union1d(outer,canthi)
    if len(np.unique(canthi))!=2:raise ValueError('The native aperture has no distinct horizontal canthi')
    curve=_curve(neutral,margin);reference_normals=_normal(neutral,faces);normal_size=np.linalg.norm(reference_normals,axis=1)
    if np.any(normal_size<=0):raise ValueError('The native lid has a collapsed reference triangle')
    if preserve_projection:
        # Neutral fitting should not fair an already accepted aperture shape.
        # Retain its projected rim/collar; the constrained depth solve below
        # still enforces orientation, movement and actual globe clearance.
        result=posed.copy();history=[]
    else:
        result,history=_project(neutral,posed,faces,fixed,width,curve,margin,min(.15,maximum_fraction*.5),check,notify)
    check();notify('Solving lid depth and globe clearance');r=neutral/width;p=result/width;e=eye/width
    _,deficit,witness=depth_constraints(e,_hull_triangles(e),p,faces,clearance=.0005,cancelled=cancelled)
    # A witness located at a pinned canthus cannot be repaired by moving other
    # vertices. Release only its Y coordinate when actual captured penetration
    # (not just the clearance cushion) proves the anchor is infeasible.
    contact_faces=faces[witness['skin_faces']];penetrating=deficit>.0005+1e-10
    depth_canthi=np.unique(contact_faces[(witness['skin_barycentric']>1-1e-8)&penetrating[:,None]&np.isin(contact_faces,canthi)])
    depth_fixed=np.union1d(outer,np.setdiff1d(canthi,depth_canthi));free=np.setdiff1d(np.unique(faces),depth_fixed)
    adj=sparse.coo_matrix((np.ones(2*len(edges)),(np.r_[edges[:,0],edges[:,1]],np.r_[edges[:,1],edges[:,0]])),shape=(len(neutral),len(neutral))).tocsr();degree=np.asarray(adj.sum(1)).ravel()
    lap=(sparse.eye(len(neutral))-sparse.diags(1/np.maximum(1,degree))@adj)[np.unique(faces)];K=lap[:,free];R=curve[:,free];delta=p[:,1]-r[:,1]
    Q=(K.T@K+.003*sparse.eye(len(free))+10*R.T@R).tocsc();q=np.asarray(K.T@(lap@delta)+10*R.T@(curve@p[:,1])).ravel()
    rn=_normal(r,faces);pn=_normal(p,faces);a=p[faces[:,1]]-p[faces[:,0]];b=p[faces[:,2]]-p[faces[:,0]]
    g1=rn[:,0]*b[:,2]-rn[:,2]*b[:,0];g2=-rn[:,0]*a[:,2]+rn[:,2]*a[:,0]
    G=sparse.coo_matrix((np.column_stack((-g1-g2,g1,g2)).ravel(),(np.repeat(np.arange(len(faces)),3),faces.ravel())),shape=(len(faces),len(neutral))).tocsr()[:,free]
    row_scale=1/np.maximum(np.linalg.norm(rn,axis=1),1e-14);G=sparse.diags(row_scale)@G
    base_dot=.025*np.sum(rn*rn,axis=1);initial_dot=np.einsum('ij,ij->i',pn,rn);movable_face=np.any(np.isin(faces,free),axis=1)
    # A positive dot alone can be lost when native field reconstruction moves
    # vertices by a few float32 ULPs. Bound the cross-product change when each
    # triangle vertex moves within this physical position-error ball.
    native_position_budget_cm=2e-4;uncertainty=native_position_budget_cm/width
    def required_dot(points):
        first=points[faces[:,1]]-points[faces[:,0]];second=points[faces[:,2]]-points[faces[:,0]]
        area_error=2*uncertainty*(np.linalg.norm(first,axis=1)+np.linalg.norm(second,axis=1))+4*uncertainty*uncertainty
        return base_dot+movable_face*np.linalg.norm(rn,axis=1)*area_error
    required=required_dot(p)
    B=sparse.coo_matrix((witness['skin_barycentric'].ravel(),(np.repeat(np.arange(len(deficit)),3),faces[witness['skin_faces']].ravel())),shape=(len(deficit),len(neutral))).tocsr()[:,free]
    deficit-=.0005*(1-np.clip(np.asarray(B.sum(1)).ravel(),0,1));limit=min(.15,maximum_fraction*.5)
    upper_depth=np.full(len(free),limit);upper_depth[np.isin(free,depth_canthi)]=0.
    A=sparse.vstack((-G,B,sparse.eye(len(free)),-sparse.eye(len(free)))).tocsc()
    settings=clarabel.DefaultSettings();settings.verbose=False;settings.tol_feas=1e-9;settings.tol_gap_abs=1e-10
    depth_margin=None
    for depth_iteration in range(4):
        lower=(required-initial_dot)*row_scale;bounds=np.r_[-lower,-deficit,upper_depth,np.full(len(free),limit)]
        solved=clarabel.DefaultSolver(sparse.triu(Q).tocsc(),q,A,bounds,[clarabel.NonnegativeConeT(len(bounds))],settings).solve();check()
        if str(solved.status) not in {'Solved','AlmostSolved'}:raise ValueError('The bounded lid depth/clearance constraints are infeasible; retain the preceding rig')
        dy=np.asarray(solved.x)
        if not np.isfinite(dy).all() or np.min(bounds-A@dy)<-1e-8:raise ValueError('The depth proposal failed its actual constraint residuals')
        candidate=p.copy();candidate[free,1]+=dy;actual_required=required_dot(candidate)
        if np.min((np.einsum('ij,ij->i',_normal(candidate,faces),rn)-actual_required)*row_scale)>=-1e-9:break
        required=np.maximum(required,actual_required)
    else:
        check();notify('Solving native-position safety against final edge lengths')
        dy,depth_margin=_conic_depth_margin(Q,q,G,B,deficit,upper_depth,limit,p,faces,free,
            initial_dot,base_dot,row_scale,movable_face,uncertainty)
        candidate=p.copy();candidate[free,1]+=dy;check()
        if np.min((np.einsum('ij,ij->i',_normal(candidate,faces),rn)-required_dot(candidate))*row_scale)<-1e-9:
            raise ValueError('The final-edge depth solve did not meet the native-position safety bound')
    result[free,1]+=dy*width;result,socket=continue_socket(neutral,result,triangles,interior,margin)
    changed=np.linalg.norm(result-posed,axis=1)>1e-7;result[changed]=result[changed].astype(np.float32).astype(float);result[~changed]=posed[~changed]
    if not np.array_equal(result[depth_fixed],posed[depth_fixed]) or not np.array_equal(result[canthi][:,[0,2]],posed[canthi][:,[0,2]]) or not np.isfinite(result).all():raise ValueError('The repair changed a fixed lid anchor')
    maximum=float(np.linalg.norm(result-posed,axis=1).max())
    check();notify('Checking the complete head for new proper intersections');active=np.any(changed[triangles],axis=1);tolerance=float(np.linalg.norm(np.ptp(posed,axis=0)))*1e-6
    after=crossings(result,triangles,pairs=True,tolerance=tolerance,active_faces=active)
    if after:
        candidate,alternate=continue_socket(neutral,result,triangles,interior,margin,footprint=True)
        moved=np.linalg.norm(candidate-posed,axis=1)>1e-7;candidate[moved]=candidate[moved].astype(np.float32).astype(float);candidate[~moved]=posed[~moved]
        remaining=crossings(candidate,triangles,pairs=True,tolerance=tolerance,active_faces=np.any(moved[triangles],axis=1));check()
        if len(remaining)<len(after):
            alternate['preceding_proper_intersections']=len(after);result=candidate;socket=alternate;changed=moved;after=remaining
            maximum=float(np.linalg.norm(result-posed,axis=1).max())
    if maximum>maximum_fraction*width:raise ValueError('Socket continuation exceeds the requested physical movement limit')
    if after:
        notify('Untangling the hidden socket footprint with its rim fixed')
        try:
            candidate,alternate=continue_socket(neutral,result,triangles,interior,margin,injective=True,cancelled=cancelled)
            moved=np.linalg.norm(candidate-posed,axis=1)>1e-7;candidate[moved]=candidate[moved].astype(np.float32).astype(float);candidate[~moved]=posed[~moved]
            remaining=crossings(candidate,triangles,pairs=True,tolerance=tolerance,active_faces=np.any(moved[triangles],axis=1));check()
            movement=float(np.linalg.norm(candidate-posed,axis=1).max())
            if len(remaining)<len(after) and movement<=maximum_fraction*width:
                alternate['preceding_proper_intersections']=len(after);result=candidate;socket=alternate;changed=moved;after=remaining;maximum=movement
        except ValueError as error:
            notify('Socket footprint retained: '+str(error))
    before=crossings(posed,triangles,pairs=True,tolerance=tolerance,active_faces=np.any(changed[triangles],axis=1))
    if after:raise ValueError('The proposed lid retains '+str(len(after))+' proper head intersections; preceding rig retained')
    final_normals=_normal(result,faces)
    if np.any(np.einsum('ij,ij->i',reference_normals,final_normals)<=0):raise ValueError('The quantized visible lid still reverses relative to its native reference')
    if not _simple_boundary(result[margin][:,[0,2]]):raise ValueError('The quantized aperture boundary still crosses; retain the preceding rig')
    _,final_deficit,final_witness=depth_constraints(eye,_hull_triangles(eye),result,faces,clearance=0.,cancelled=cancelled);clearance=float(-final_deficit.max());check()
    if clearance<0:raise ValueError('The quantized lid does not clear the captured globe')
    report=dict(status='proposal',method='RENDER_LID_UNTANGLE_AND_SOCKET_CONTINUATION',projection=history,depth_status=depth_margin['status'] if depth_margin else str(solved.status),depth_iterations=depth_iteration+1+(1 if depth_margin else 0),depth_margin=depth_margin,native_position_budget_cm=native_position_budget_cm,socket=socket,
                changed_vertices=int(changed.sum()),maximum_movement=maximum,eye_width=width,clearance=clearance,clearance_witnesses=len(final_deficit),
                removed_proper_intersections=len(before),remaining_proper_intersections=0,visible_reversed_triangles=0,fixed_boundary_exact=True,canthi_exact=bool(np.array_equal(result[canthi],posed[canthi])),canthi_projection_exact=True,contact_depth_canthi=len(depth_canthi),maximum_canthi_depth_change=float(np.max(abs(result[canthi,1]-posed[canthi,1]))),float32_quantized=True,
                neutral_geometry_changed=False,globe_geometry_changed=False,native_motion_accepted=False,scope='Single captured frontal lid endpoint; proper crossings involving changed faces and frontal globe clearance only')
    return result,report
