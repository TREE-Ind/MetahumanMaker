"""Fixed-boundary planar embeddings for a triangulated anatomical socket cap."""
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve
from scipy.optimize import minimize


def injective_footprint(neutral,posed,faces,interior,margin,*,cancelled=None):
    """Untangle the cap projection without moving its rim or exterior vertices.

    A convex Tutte embedding supplies the reference metric. A smooth determinant
    barrier maps that disk to the actual simple aperture, keeping its boundary
    fixed. The caller must quantize and check the full 3D head afterward.
    """
    def check():
        if cancelled and cancelled():raise RuntimeError('Socket footprint cancelled; preceding rig retained')
    ids=np.unique(interior);boundary=np.unique(margin);lookup=np.full(len(posed),-1,dtype=int);lookup[ids]=np.arange(len(ids))
    selected=faces[np.all(np.isin(faces,ids),axis=1)];tri=lookup[selected];b=lookup[boundary];free=np.setdiff1d(np.arange(len(ids)),b)
    if not len(tri) or not len(free):raise ValueError('The socket cap needs interior triangles and a fixed rim')
    edges,counts=np.unique(np.sort(np.vstack((tri[:,[0,1]],tri[:,[1,2]],tri[:,[2,0]])),axis=1),axis=0,return_counts=True)
    rim=edges[counts==1]
    if np.any(counts>2) or len(ids)-len(edges)+len(tri)!=1 or set(rim.ravel())!=set(b):raise ValueError('Injective continuation needs a single manifold disk cap')
    neighbors={int(i):[] for i in b}
    for i,j in rim:neighbors[int(i)].append(int(j));neighbors[int(j)].append(int(i))
    if any(len(row)!=2 for row in neighbors.values()):raise ValueError('The socket rim is not one simple topology loop')
    ordered=[int(b.min())];previous=-1
    while True:
        following=next(i for i in sorted(neighbors[ordered[-1]]) if i!=previous)
        if following==ordered[0]:break
        if following in ordered:raise ValueError('The socket rim contains disconnected loops')
        previous=ordered[-1];ordered.append(following)
    if len(ordered)!=len(b):raise ValueError('The socket boundary is disconnected')
    ordered=np.asarray(ordered);target=np.asarray(posed[ids][:,[0,2]],dtype=float);center=target[b].mean(0);scale=float(np.linalg.norm(np.ptp(target[b],axis=0)))
    if not np.isfinite(scale) or scale<=0:raise ValueError('The socket aperture has no projected extent')
    target=(target-center)/scale
    # Reuse the exterior repair's proper boundary check without changing it.
    from .lid_surface_repair import _simple_boundary
    if not _simple_boundary(target[ordered]):raise ValueError('Repair the aperture boundary before continuing its socket')
    ring=target[ordered];following=np.roll(ring,-1,axis=0);area=.5*np.sum(ring[:,0]*following[:,1]-ring[:,1]*following[:,0])
    if abs(area)<1e-12:raise ValueError('The projected socket aperture has zero area')
    length=np.linalg.norm(neutral[ids[ordered]]-neutral[ids[np.roll(ordered,-1)]],axis=1)
    if np.any(length<=0):raise ValueError('The native socket rim has collapsed edges')
    angle=np.r_[0.,np.cumsum(length[:-1])]/length.sum()*2*np.pi*np.sign(area);reference=np.zeros_like(target)
    reference[ordered]=np.column_stack((np.cos(angle),np.sin(angle)))*np.sqrt(abs(area)/np.pi)
    graph=sparse.coo_matrix((np.ones(2*len(edges)),(np.r_[edges[:,0],edges[:,1]],np.r_[edges[:,1],edges[:,0]])),shape=(len(ids),len(ids))).tocsr();lap=sparse.diags(np.asarray(graph.sum(1)).ravel())-graph
    reference[free]=spsolve(lap[free][:,free].tocsc(),-lap[free][:,b]@reference[b]);check()
    dm=np.stack((reference[tri[:,1]]-reference[tri[:,0]],reference[tri[:,2]]-reference[tri[:,0]]),axis=2);signed=np.linalg.det(dm)
    if np.any(abs(signed)<1e-14) or not (np.all(signed>0) or np.all(signed<0)):raise ValueError('The convex reference socket did not produce a valid disk embedding')
    inverse=np.linalg.inv(dm);weights=abs(signed)/abs(signed).sum();p=target.copy();origin=p.copy();history=[]
    def evaluate(x,epsilon):
        check();p[free]=x.reshape(-1,2);ds=np.stack((p[tri[:,1]]-p[tri[:,0]],p[tri[:,2]]-p[tri[:,0]]),axis=2);j=ds@inverse;det=np.linalg.det(j);trace=np.sum(j*j,axis=(1,2));h=np.hypot(det,epsilon)
        chi=(det+h)*.5;negative=det<0;chi[negative]=epsilon**2/(2*(h[negative]-det[negative]));numerator=trace+det*det+1.;delta=p[free]-origin[free]
        energy=float(np.sum(weights*numerator/chi)+np.sum(delta*delta)/len(free))
        cofactor=np.empty_like(j);cofactor[:,0,0]=j[:,1,1];cofactor[:,1,1]=j[:,0,0];cofactor[:,0,1]=-j[:,1,0];cofactor[:,1,0]=-j[:,0,1]
        derivative=weights[:,None,None]*(2*j/chi[:,None,None]+(2*det/chi-numerator*.5*(1+det/h)/chi**2)[:,None,None]*cofactor)
        columns=derivative@np.swapaxes(inverse,1,2);gradient=np.zeros_like(p);np.add.at(gradient,tri[:,1],columns[:,:,0]);np.add.at(gradient,tri[:,2],columns[:,:,1]);np.add.at(gradient,tri[:,0],-columns.sum(2));gradient[free]+=2*delta/len(free)
        if not np.isfinite(energy) or not np.isfinite(gradient).all():raise ValueError('The socket determinant objective became nonfinite')
        return energy,gradient[free].ravel()
    bounds=list(zip(np.tile(target[b].min(0),len(free)),np.tile(target[b].max(0),len(free))));x=p[free].ravel()
    for epsilon in (.1,.03,.01,.003,.001,.0001,.00001):
        solved=minimize(evaluate,x,args=(epsilon,),jac=True,method='L-BFGS-B',bounds=bounds,options=dict(maxiter=450,ftol=1e-12,gtol=1e-8,maxls=40,maxcor=20));x=solved.x;evaluate(x,epsilon)
        ratio=np.linalg.det(np.stack((p[tri[:,1]]-p[tri[:,0]],p[tri[:,2]]-p[tri[:,0]]),axis=2)@inverse);history.append(dict(epsilon=epsilon,iterations=int(solved.nit),reversed=int((ratio<=0).sum()),minimum_ratio=float(ratio.min())))
    if ratio.min()<=1e-6 or not np.array_equal(p[b],target[b]):raise ValueError('The bounded socket footprint still folds; preceding rig retained')
    result=np.array(posed,copy=True);result[np.ix_(ids[free],[0,2])]=p[free]*scale+center
    result[np.ix_(ids[free],[0,2])]=result[np.ix_(ids[free],[0,2])].astype(np.float32).astype(float)
    decoded=(result[ids][:,[0,2]]-center)/scale
    ratio=np.linalg.det(np.stack((decoded[tri[:,1]]-decoded[tri[:,0]],decoded[tri[:,2]]-decoded[tri[:,0]]),axis=2)@inverse)
    if ratio.min()<=1e-6:raise ValueError('The float32 socket footprint still folds; preceding rig retained')
    return result,dict(method='POSITIVE_JACOBIAN_SOCKET_FOOTPRINT',interior_vertices=len(free),triangles=len(tri),fixed_boundary_exact=True,minimum_area_ratio=float(ratio.min()),float32_quantized=True,steps=history,maximum_projected_change=float(np.linalg.norm((p-origin)*scale,axis=1).max()),scope='Planar cap only; full quantized 3D intersection checks required')
