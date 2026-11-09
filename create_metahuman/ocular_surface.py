"""Geometry-only source-eye reconstruction for the optional numerical worker."""
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve
from scipy.spatial import ConvexHull,Delaunay
from scipy.interpolate import RBFInterpolator
from .source_contours import convex_outline
from .surface_charts import boundary_loop,ordered_angles


def _area(points, faces):
    a=points[faces[:,1]]-points[faces[:,0]];b=points[faces[:,2]]-points[faces[:,0]]
    return a[:,0]*b[:,1]-a[:,1]*b[:,0]


def _sphere(points, *, posterior=False):
    points=np.asarray(points,dtype=float)
    if posterior:points=points[points[:,1]>=np.median(points[:,1])]
    origin=points.mean(0);q=points-origin
    solution=np.linalg.lstsq(np.c_[q*2,np.ones(len(q))],np.sum(q*q,axis=1),rcond=None)[0]
    center=origin+solution[:3];radius=float(np.median(np.linalg.norm(points-center,axis=1)))
    if not np.isfinite(radius) or radius<=0:raise ValueError('Eye surface does not define a finite positive radius')
    return center,radius


def _anterior_margin(reference, faces, adjacency, cap, rim):
    """Find the lid's front ridge beyond the posterior UV socket boundary.

    Grow complete topological rings from the socket interior. The first front
    depth minimum is the neutral lid margin; character scale and placement do
    not enter this search. Refuse ambiguous anatomy rather than fitting the
    posterior UV seam as though it were the visible opening.
    """
    seen=np.zeros(len(reference),bool);seen[cap]=True
    ring=np.zeros(len(reference),bool);ring[rim]=True;rows=[]
    for step in range(18):
        ids=np.flatnonzero(ring)
        if len(ids)<8:break
        rows.append((float(reference[ids,1].mean()),ids,seen.copy()))
        ring=(adjacency@ring.astype(float)>0)&~seen;seen|=ring
    peak=next((i for i in range(2,len(rows)-1)
               if rows[i][0]<rows[i-1][0] and rows[i][0]<=rows[i+1][0]),None)
    if peak is None:raise ValueError('Template has no identifiable anterior eyelid margin')
    _,_,interior=rows[peak]
    patch=faces[np.all(interior[faces],axis=1)]
    edges,count=np.unique(np.sort(np.vstack((patch[:,[0,1]],patch[:,[1,2]],patch[:,[2,0]])),axis=1),axis=0,return_counts=True)
    boundary=edges[count==1];margin,degree=np.unique(boundary,return_counts=True)
    if len(margin)<8 or np.any(degree!=2):raise ValueError('Anterior eyelid margin is not a closed manifold loop')
    # A graph-distance frontier is not itself a mesh boundary: diagonal
    # triangles can leave extra tips in it. Constrain the actual patch loop.
    return boundary_loop(patch),np.unique(patch),peak


def _aperture_margin(reference, faces, adjacency, cap, rim, eye, eye_triangles):
    """Calibrate the lid loop from pristine native globe/skin visibility.

    The most anterior skin ridge is outside the anatomical aperture. Mapping
    that ridge onto the source opening changes which lid vertices are visible
    and how the inherited rig moves them. Use native neutral occlusion to pick
    the closest complete topological loop, independent of the source identity.
    """
    n=len(reference);distance=np.full(n,-1,dtype=np.int32);distance[rim]=0
    seen=np.zeros(n,bool);seen[cap]=True;ring=np.zeros(n,bool);ring[rim]=True
    interiors=[seen.copy()]
    for step in range(1,18):
        ring=(adjacency@ring.astype(float)>0)&~seen;seen|=ring
        distance[ring]=step;interiors.append(seen.copy())
    lo=eye[:,[0,2]].min(0);hi=eye[:,[0,2]].max(0)
    x,z=np.meshgrid(np.linspace(lo[0],hi[0],97),np.linspace(lo[1],hi[1],73));q=np.c_[x.ravel(),z.ravel()]
    domain=Delaunay(eye[:,[0,2]]).find_simplex(q)>=0
    globe=np.full(len(q),np.inf);globe[domain]=FrontSurface(eye,eye_triangles)(q[domain])
    skin,hit,bary=FrontSurface(reference,faces).sample(q)
    tolerance=max(float(np.linalg.norm(hi-lo))*1e-6,1e-10)
    visible=(globe<skin-tolerance).reshape(x.shape)
    neighbor=np.zeros_like(visible);neighbor[1:]|=visible[:-1];neighbor[:-1]|=visible[1:];neighbor[:,1:]|=visible[:,:-1];neighbor[:,:-1]|=visible[:,1:]
    edge=np.flatnonzero((~visible&neighbor&domain.reshape(x.shape)).ravel())
    if visible.sum()<24 or len(edge)<24:raise ValueError('The original native reference has no identifiable open-eye silhouette')
    d=distance[faces[hit[edge]]];valid=np.all(d>=0,axis=1)
    if valid.sum()<len(edge)*.9:raise ValueError('Native eye silhouette does not belong to the expected lid topology')
    sampled=np.sum(d[valid]*bary[edge[valid]],axis=1);median=float(np.median(sampled));step=int(np.floor(median+.5))
    if not 1<=step<len(interiors)-2:raise ValueError('Native eye aperture lies outside the supported eyelid collar')
    patch=faces[np.all(interiors[step][faces],axis=1)]
    loop=boundary_loop(patch)
    return loop,np.unique(patch),step,dict(method='NATIVE_NEUTRAL_OCCLUSION',grid=[97,73],visible_samples=int(visible.sum()),boundary_samples=len(edge),median_topological_ring=median,selected_ring=step,scope='Original native head and globe; source character does not choose anatomical correspondence')


class FrontSurface:
    def __init__(self, points, triangles):
        self.points=np.asarray(points)[np.asarray(triangles)]
        xy=self.points[:,:,[0,2]];self.low=xy.min(1);self.high=xy.max(1)

    def __call__(self, query):
        return self.sample(query)[0]

    def sample(self, query):
        query=np.asarray(query);low=query.min(0);high=query.max(0)
        selected=np.all(self.low<=high,axis=1)&np.all(self.high>=low,axis=1)
        pts=self.points[selected];lo=self.low[selected];hi=self.high[selected];indices=np.flatnonzero(selected);result=[];face_ids=[];bary=[]
        for q in query:
            candidates=np.all(lo<=q+1e-10,axis=1)&np.all(hi>=q-1e-10,axis=1)
            s=pts[candidates];ids=indices[candidates];xy=s[:,:,[0,2]];a=xy[:,1]-xy[:,0];b=xy[:,2]-xy[:,0];r=q-xy[:,0]
            det=a[:,0]*b[:,1]-a[:,1]*b[:,0];valid=abs(det)>1e-18
            s=s[valid];ids=ids[valid];a=a[valid];b=b[valid];r=r[valid];det=det[valid]
            u=(r[:,0]*b[:,1]-r[:,1]*b[:,0])/det;v=(a[:,0]*r[:,1]-a[:,1]*r[:,0])/det
            inside=(u>=-1e-8)&(v>=-1e-8)&(u+v<=1+1e-8)
            y=s[:,0,1]+u*(s[:,1,1]-s[:,0,1])+v*(s[:,2,1]-s[:,0,1])
            if not inside.any():raise ValueError('The source does not cover the eye reconstruction region; refine alignment or source eye patches')
            hit=np.flatnonzero(inside)[np.argmin(y[inside])];result.append(float(y[hit]));face_ids.append(ids[hit]);bary.append([1-u[hit]-v[hit],u[hit],v[hit]])
        return np.array(result),np.array(face_ids,dtype=np.int64),np.array(bary)


class EyeSurface:
    """Smooth source sclera continuation, independent of surrounding lid skin.

    Normalized local height fitting retains the marked eye's surface while a
    quadratic polynomial prior continues curvature outside its observed edge.
    The continuation is inferred, not an observation of the hidden globe.
    """
    def __init__(self,points):
        points=np.asarray(points,dtype=float);self.origin=points.mean(0)
        self.scale=float(np.ptp(points[:,[0,2]],axis=0).max())
        if self.scale<=0:raise ValueError('The source eye patch has zero projected extent')
        xy=(points[:,[0,2]]-self.origin[[0,2]])/self.scale
        xy,indices=np.unique(np.rint(xy*1e9).astype(np.int64),axis=0,return_index=True)
        if len(indices)<12:raise ValueError('The source eye patch has too few distinct projected points')
        self.field=RBFInterpolator(xy.astype(float)/1e9,(points[indices,1]-self.origin[1])/self.scale,
            kernel='thin_plate_spline',degree=2,smoothing=1e-5,neighbors=None if len(indices)<=1024 else 96)

    def __call__(self,query):
        return self.field((np.asarray(query)-self.origin[[0,2]])/self.scale)*self.scale+self.origin[1]


class ObservedEyeSurface:
    """Use measured frontal depth inside the marked eye's closed outline.

    Sclera selections commonly omit an iris or pupil. Fitting only their
    vertices invents the middle of an otherwise observed eye. Read that
    middle from the complete source, keeping the smooth prior for hidden
    continuation. A narrow, scale-relative blend avoids a depth step at the
    inferred outline. The convex outline has the same selection semantics as
    the lid fit; it is not a semantic classifier for arbitrary source meshes.
    """
    def __init__(self, points, source_surface):
        self.prior=EyeSurface(points);self.source=source_surface
        self.hull=np.asarray(points)[ConvexHull(np.asarray(points)[:,[0,2]]).vertices][:,[0,2]]
        self.edge=np.roll(self.hull,-1,axis=0)-self.hull
        self.width=float(np.ptp(self.hull[:,0]))

    def __call__(self, query):
        query=np.asarray(query);result=self.prior(query)
        delta=query[:,None,:]-self.hull[None,:,:]
        inset=np.min((self.edge[None,:,0]*delta[:,:,1]-self.edge[None,:,1]*delta[:,:,0])/np.linalg.norm(self.edge,axis=1)[None,:],axis=1)
        inside=inset>0.
        if inside.any():
            weight=np.clip(inset[inside]/(self.width*.02),0.,1.)
            weight=weight*weight*(3.-2.*weight)
            result[inside]=result[inside]*(1.-weight)+self.source(query[inside])*weight
        return result


def globe_fidelity(eye, triangles, source_surface, hull):
    """Inspect the exported triangle surface, including unmarked eye centers.

    Finite frontal samples are a fidelity diagnostic, not a collision proof.
    Values use input world units and do not assume the scene's unit scale.
    """
    hull=np.asarray(hull);edge=np.roll(hull,-1,axis=0)-hull
    lo=hull.min(0);hi=hull.max(0);width=float(np.ptp(hull[:,0]))
    x,z=np.meshgrid(np.linspace(lo[0],hi[0],41),np.linspace(lo[1],hi[1],31));q=np.c_[x.ravel(),z.ravel()]
    delta=q[:,None,:]-hull[None,:,:]
    inset=np.min((edge[None,:,0]*delta[:,:,1]-edge[None,:,1]*delta[:,:,0])/np.linalg.norm(edge,axis=1)[None,:],axis=1)
    q=q[inset>width*.03]
    if not len(q):raise ValueError('The selected eye has no usable interior fidelity samples')
    error=FrontSurface(eye,triangles)(q)-source_surface(q)
    return dict(method='FRONTAL_TRIANGLE_DEPTH_GRID',grid=[41,31],samples=len(q),
        outline_inset_fraction=.03,depth_rms_world=float(np.sqrt(np.mean(error**2))),
        depth_max_world=float(np.max(abs(error))),limitation='Finite frontal depth samples; excludes the outline transition and hidden globe')


def _chart(reference, current, faces, adjacency, rim, desired, *, cancelled=None, progress=None, region=None):
    import igl
    n=len(reference);center=reference[rim].mean(0);width=float(np.ptp(reference[rim,0]))
    if width<=0:raise ValueError('Template eye boundary has zero width')
    r=(reference[:,[0,2]]-center[[0,2]])/[width*1.3,width*1.4]
    if region is None:region=(np.linalg.norm(r,axis=1)<1)&(reference[:,1]<center[1]+width*2.5)
    else:
        region=np.asarray(region,dtype=bool)
        if region.shape!=(n,):raise ValueError('Surface chart region must match the complete reference mesh')
    selected=np.all(region[faces],1);raw=faces[selected];ids=np.unique(raw)
    lookup=np.full(n,-1);lookup[ids]=np.arange(len(ids));ff=lookup[raw].astype(np.int32)
    edges=np.vstack((ff[:,[0,1]],ff[:,[1,2]],ff[:,[2,0]]));ee,count=np.unique(np.sort(edges,axis=1),axis=0,return_counts=True)
    border={tuple(row) for row in ee[count==1]};directed={int(i):int(j) for i,j in edges if tuple(sorted((i,j))) in border}
    if not directed:raise ValueError('Eye region has no topological boundary')
    order=[min(directed)]
    while directed[order[-1]]!=order[0]:
        order.append(directed[order[-1]])
        if len(order)>len(directed):raise ValueError('Eye region boundary is not a simple loop')
    if len(order)!=len(directed):raise ValueError('Eye region is not a single disk')
    order=np.array(order);q=reference[ids[order]];lengths=np.linalg.norm(np.roll(q,-1,axis=0)-q,axis=1)
    phase=np.r_[0,np.cumsum(lengths[:-1])]/lengths.sum()*2*np.pi
    target=current[ids[order]][:,[0,2]];origin=target.mean(0);radius=np.ptp(target,axis=0)*.5
    startangle=np.arctan2(*(target[0]-origin)[::-1]);circle=origin+np.c_[np.cos(phase+startangle),np.sin(phase+startangle)]*radius
    outer=np.zeros(len(ids),bool);outer[order]=True;adj=adjacency[ids][:,ids];L=sparse.diags(np.asarray(adj.sum(1)).ravel())-adj
    U=np.zeros((len(ids),2));U[order]=circle
    U[~outer]=spsolve(L[~outer][:,~outer].tocsc(),-L[~outer][:,outer]@U[outer])
    if np.any(_area(U,ff)<=0):raise ValueError('Could not establish an unfolded eye region')
    fix=outer|np.isin(ids,rim);fixed=np.flatnonzero(fix).astype(np.int32);bc=current[ids[fixed]][:,[0,2]].copy()
    for i,p in zip(rim,desired):
        k=np.flatnonzero(ids[fixed]==i)
        if len(k):bc[k[0]]=p[[0,2]]
    scale=width*3;V=np.asfortranarray(reference[ids]/width);F=np.asfortranarray(ff);sol=(U-origin)/scale;bounds=(bc-origin)/scale;start=sol[fixed].copy()
    for stage in range(1,21):
        if cancelled and cancelled():raise RuntimeError('Eye reconstruction cancelled')
        if progress:progress(stage/23)
        sb=np.asfortranarray(start+(bounds-start)*(stage/20))
        solver=igl.slim_precompute(V,F,np.asfortranarray(sol),igl.MappingEnergyType.SYMMETRIC_DIRICHLET,fixed,sb,1e9);sol=igl.slim_solve(solver,30)
    for step,penalty in enumerate((1e10,1e11,1e12),21):
        if cancelled and cancelled():raise RuntimeError('Eye reconstruction cancelled')
        if progress:progress(step/23)
        solver=igl.slim_precompute(V,F,np.asfortranarray(sol),igl.MappingEnergyType.SYMMETRIC_DIRICHLET,fixed,np.asfortranarray(bounds),penalty);sol=igl.slim_solve(solver,30)
    xy=np.asarray(sol)*scale+origin;error=float(np.linalg.norm(xy[fixed]-bc,axis=1).max())
    correction=np.zeros_like(xy);correction[fixed]=bc-xy[fixed]
    correction[~fix]=spsolve(L[~fix][:,~fix].tocsc(),-L[~fix][:,fix]@correction[fix]);xy+=correction;xy[fixed]=bc
    if error>width*.001 or np.any(_area(xy,ff)<=0):raise ValueError('Source eye constraints could not be fitted without folds (residual '+str(error)+', nonpositive '+str(int(np.sum(_area(xy,ff)<=0)))+'); previous geometry retained')
    return ids,xy,ids[order],dict(planar_faces=len(ff),nonpositive_faces=0,penalty_residual=error,boundary_residual=0.)


def fit(data, options, *, progress=None, cancelled=None):
    basis=np.asarray(data['eye_basis']);to_frame=lambda p:np.asarray(p)@basis
    reference=to_frame(data['parameter_reference']);current=to_frame(data['current']);source=to_frame(data['source']);tri=np.asarray(data['triangles']);source_tri=np.asarray(data['source_triangles'])
    tolerance=max(float(np.linalg.norm(np.ptp(reference,axis=0)))*1e-6,1e-9)
    _,w=np.unique(np.rint(reference/tolerance).astype(np.int64),axis=0,return_inverse=True);n=int(w.max()+1);count=np.bincount(w)
    physical=lambda p:np.column_stack([np.bincount(w,weights=p[:,j])/count for j in range(3)])
    b=physical(reference);v=physical(current);out=v.copy();faces=w[tri]
    edges=np.unique(np.sort(np.vstack((faces[:,[0,1]],faces[:,[1,2]],faces[:,[2,0]])),axis=1),axis=0);edges=edges[edges[:,0]!=edges[:,1]]
    adj=sparse.coo_matrix((np.ones(len(edges)*2),(np.r_[edges[:,0],edges[:,1]],np.r_[edges[:,1],edges[:,0]])),shape=(n,n)).tocsr()
    samples=source[source_tri].mean(1);ray=FrontSurface(source,source_tri);active=np.zeros(n,bool);proof={};reports=[]
    for k in range(int(options['eye_count'])):
        if cancelled and cancelled():raise RuntimeError('Eye reconstruction cancelled')
        cap=np.unique(w[data['eye_cap_'+str(k)]]);rim=np.unique(w[data['eye_rim_'+str(k)]])
        chart_center=b[rim].mean(0);chart_width=float(np.ptp(b[rim,0]))
        if chart_width<=0:raise ValueError('Template eye boundary has zero width')
        chart_radius=(b[:,[0,2]]-chart_center[[0,2]])/[chart_width*1.3,chart_width*1.4]
        chart_region=(np.linalg.norm(chart_radius,axis=1)<1)&(b[:,1]<chart_center[1]+chart_width*2.5)
        if 'anatomy_reference_head' not in data or 'anatomy_eye_'+str(k) not in data:raise ValueError('Eye fitting needs an original native anatomy reference; prepare Native Expressions and retry')
        anatomy=physical(to_frame(data['anatomy_reference_head']))
        rim,cap,margin_ring,anatomy_report=_aperture_margin(anatomy,faces,adj,cap,rim,to_frame(data['anatomy_eye_'+str(k)]),np.asarray(data['eye_triangles_'+str(k)]))
        center=v[rim].mean(0);width=np.ptp(v[rim,0]);delta=samples-center
        roi=(abs(delta[:,0])<width*1.35)&(abs(delta[:,2])<width*.75)&(abs(delta[:,1])<width*.65)
        if options.get('source_patches'):
            selected=np.asarray(data['source_eye_faces_'+str(k)],dtype=bool)
        else:
            rgb=data['source_eye_colors'];selected=roi&(rgb.min(1)>.48)&(np.ptp(rgb,axis=1)<.22)
            if selected.sum()<24 or selected.sum()>max(int(roi.sum()*.7),24):raise ValueError('Automatic sclera boundaries are not distinguishable; mark the source eye patches for this character')
        points=source[np.unique(source_tri[selected])]
        if len(points)<12:raise ValueError('The selected source eye patch is too small')
        hull=points[ConvexHull(points[:,[0,2]]).vertices]
        angles,angle_report=ordered_angles(b[rim]);desired=convex_outline(hull,angles)
        ids,xy,outer,report=_chart(b,v,faces,adj,rim,desired,region=chart_region,cancelled=cancelled,progress=(lambda value:progress('Unfolding source eye '+str(k+1),(k+value)/options['eye_count'])) if progress else None)
        changed=np.setdiff1d(ids,outer)
        if np.any(active[changed]):raise ValueError('Eye reconstruction regions overlap; refine the template or alignment')
        out[ids[:,None],[0,2]]=xy;out[changed,1]=ray(out[changed][:,[0,2]])
        inside=np.setdiff1d(cap,rim)
        h=hull[:,[0,2]];edge=np.roll(h,-1,axis=0)-h
        d=out[inside,None,:][:,:,[0,2]]-h[None,:,:]
        inset=np.min((edge[None,:,0]*d[:,:,1]-edge[None,:,1]*d[:,:,0])/np.linalg.norm(edge,axis=1)[None,:],axis=1)
        # Recess the socket from the actual almond-shaped outline. An ellipse
        # leaves some inner-lid vertices on the sclera at its tapered corners.
        eye_width=float(np.ptp(desired[:,0]));out[inside,1]+=eye_width*.3*(-np.expm1(-np.maximum(inset,0)/(eye_width*.1)))
        active[changed]=True
        eye_surface=ObservedEyeSurface(points,ray)
        volume=None
        if options.get('globe_volume'):
            from .ocular_globe_volume import fit as fit_volume
            eye,volume=fit_volume(to_frame(data['anatomy_eye_'+str(k)]),points,out,faces,cancelled=cancelled)
            volume['parameterization']='NATIVE_ANATOMY_DIRECTIONS'
            contact=volume
        else:
            eye=to_frame(data['eye_parameter_reference_'+str(k)]);c0,r0=_sphere(eye,posterior=True);c1,r1=_sphere(points);eye=(eye-c0)*(r1/r0)+c1
            facing=np.clip((c1[1]-eye[:,1])/r1,0,1);ii=np.flatnonzero(facing>.1);eye_surface=ObservedEyeSurface(points,ray);depth=eye_surface(eye[ii][:,[0,2]])
            h=hull[:,[0,2]];edge=np.roll(h,-1,axis=0)-h;d=eye[ii,None,:][:,:,[0,2]]-h[None,:,:];sd=np.min((edge[None,:,0]*d[:,:,1]-edge[None,:,1]*d[:,:,0])/np.linalg.norm(edge,axis=1)[None,:],axis=1)
            # A smooth one-sided barrier retreats under skin outside the opening.
            # Large constant edge offsets shift the visible intersection inward and
            # expose coarse socket chords as scallops around an otherwise good fit.
            transition=r1*.002;clearance=transition*np.logaddexp(0.,-sd/transition)
            # The unobserved continuation cannot protrude through measured skin.
            # Bound it before blending with the posterior hemisphere; unrestricted
            # polynomial extrapolation can otherwise emerge through the cheek.
            skin_depth=ray(eye[ii][:,[0,2]]);blend=np.clip(-sd/(eye_width*.02),0.,1.);blend=blend*blend*(3-2*blend)
            depth=depth*(1-blend)+skin_depth*blend
            depth=np.maximum(depth,skin_depth)
            weight=np.clip((facing[ii]-.1)/.4,0,1);weight=weight*weight*(3-2*weight)
            eye[ii,1]=eye[ii,1]*(1-weight)+(depth+clearance)*weight
            if 'eye_triangles_'+str(k) in data:
                from .ocular_contact import constrain_hidden_globe
                eye,contact=constrain_hidden_globe(eye,np.asarray(data['eye_triangles_'+str(k)]),out,faces,h,eye_width)
            else:raise ValueError('Globe topology is required for source eye clearance; update the fitting input')
        fidelity=globe_fidelity(eye,np.asarray(data['eye_triangles_'+str(k)]),ray,h)
        proof['ocular_eye_'+str(k)]=eye@basis.T
        residual=eye_surface(points[:,[0,2]])-points[:,1]
        report.update(source_faces=int(selected.sum()),accepted=True,solver='NATIVE_APERTURE_SOURCE_EYE_SURFACE',eye=k,anterior_margin_ring=margin_ring,anterior_margin_vertices=len(rim),boundary_order=angle_report,
            anatomical_aperture=anatomy_report,globe_surface='SOURCE_CONDITIONED_CONVEX_VOLUME' if volume else 'MEASURED_SOURCE_INTERIOR_WITH_SMOOTH_HIDDEN_CONTINUATION',hidden_globe_inferred=True,interior_outline='CONVEX_MARKED_PATCH',source_eye_vertex_depth_rms=float(np.sqrt(np.mean(residual**2))),source_eye_vertex_depth_max=float(np.max(abs(residual))),exported_globe_fidelity=fidelity,hidden_globe_clearance=contact)
        if volume:report['volume']=volume
        reports.append(report)
    planned=np.asarray(data['current']).copy();planned[active[w]]=(out[w[active[w]]])@basis.T
    proof.update(before=data['current'],planned=planned,ocular_movable=active[w])
    return planned,reports,proof
