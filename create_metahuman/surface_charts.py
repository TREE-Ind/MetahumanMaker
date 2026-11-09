"""Anatomical disk-chart surface correspondence, external SciPy worker.

Positive mean-value harmonic charts follow established surface
parameterization methods (see libigl's harmonic parameterization tutorial).
This prototype composes charts and retains original source triangle witnesses.
Chart validity does not certify anatomical correspondence or 3D chord quality.
"""
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import splu
from scipy.sparse.csgraph import dijkstra
from scipy.ndimage import gaussian_filter1d
from .surface_oracle import SurfaceOracle
from .vertex_contract import fingerprint


def disk_region_faces(vertices,triangles,face_ids):
    """Regularize a coarse region on a closed genus-zero source surface.

    Keep the largest connected component, fill enclosed complement components,
    and grow past vertex-only boundary pinches. Only original facets are
    selected: coordinates, triangulation and vertex order never change.
    This is a topology operation, not an anatomical segmentation model.
    DiskChart still validates the resulting patch before any correspondence.
    """
    from scipy.sparse.csgraph import connected_components
    vertices=np.asarray(vertices,dtype=float);triangles=np.asarray(triangles,dtype=np.int64)
    _,weld=np.unique(vertices,axis=0,return_inverse=True);tri=weld[triangles]
    n=int(weld.max())+1;nf=len(tri);chosen=np.zeros(nf,bool);chosen[np.asarray(face_ids,dtype=np.int64)]=True
    if not chosen.any() or chosen.all():raise ValueError('Choose a proper nonempty source region')
    edge=np.sort(np.vstack((tri[:,[0,1]],tri[:,[1,2]],tri[:,[2,0]])),axis=1)
    unique,inverse,counts=np.unique(edge,axis=0,return_inverse=True,return_counts=True)
    if np.any(counts!=2):raise ValueError('Disk region regularization requires a closed manifold source')
    order=np.argsort(inverse);owners=np.tile(np.arange(nf),3)[order].reshape(-1,2)
    adjacency=sparse.coo_matrix((np.ones(2*len(owners)),(owners.ravel(),owners[:,::-1].ravel())),shape=(nf,nf)).tocsr()
    incidence=sparse.coo_matrix((np.ones(tri.size),(tri.ravel(),np.repeat(np.arange(nf),3))),shape=(n,nf)).tocsr()
    initial=int(chosen.sum());pinches=0
    def largest(mask):
        ids=np.flatnonzero(mask)
        if not len(ids):raise ValueError('Region exhausted its exterior during topology regularization')
        _,labels=connected_components(adjacency[ids][:,ids],directed=False)
        keep=np.bincount(labels).argmax();result=np.zeros(nf,bool);result[ids[labels==keep]]=True
        return result
    chosen=largest(chosen)
    for _ in range(32):
        chosen=~largest(~chosen)
        boundary=unique[chosen[owners].sum(1)==1]
        degree=np.bincount(boundary.ravel(),minlength=n);bad=(degree!=0)&(degree!=2)
        if not bad.any():break
        pinches+=int(bad.sum())
        grown=np.asarray(incidence[bad].sum(0)).ravel()>0
        chosen|=grown
    else:raise ValueError('Source region boundary could not be regularized without manual labels')
    chosen=largest(chosen)
    return np.flatnonzero(chosen),dict(method='CLOSED_SURFACE_DISK_REGION_REGULARIZATION_V1',
        input_faces=initial,output_faces=int(chosen.sum()),resolved_vertex_pinches=pinches,
        source_geometry_unchanged=True,anatomy_review_required=True)


class DiskChart:
    def __init__(self,vertices,triangles,face_ids,*,forward=(0.,-1.,0.),optimizer='harmonic',iterations=40,distal_point=None,boundary_parameter=None):
        vertices=np.asarray(vertices,dtype=float);triangles=np.asarray(triangles,dtype=np.int64);face_ids=np.asarray(face_ids,dtype=np.int64)
        if face_ids.ndim!=1 or not len(face_ids) or face_ids.min()<0 or face_ids.max()>=len(triangles):raise ValueError('Choose a nonempty anatomical disk')
        self.vertex_ids,inverse=np.unique(triangles[face_ids],return_inverse=True)
        self.vertices=vertices[self.vertex_ids].copy();self.triangles=inverse.reshape(-1,3);self.face_ids=face_ids.copy();n=len(self.vertices)
        directed=np.vstack((self.triangles[:,[0,1]],self.triangles[:,[1,2]],self.triangles[:,[2,0]]))
        edges,first,counts=np.unique(np.sort(directed,axis=1),axis=0,return_index=True,return_counts=True)
        if np.any(counts>2) or n-len(edges)+len(face_ids)!=1:raise ValueError('Anatomical patch must be a manifold disk')
        boundary=directed[first[counts==1]];starts,nstart=np.unique(boundary[:,0],return_counts=True);ends,nend=np.unique(boundary[:,1],return_counts=True)
        if not len(boundary) or not np.array_equal(starts,ends) or np.any(nstart!=1) or np.any(nend!=1):raise ValueError('Anatomical boundary is not one simple loop')
        following=dict(map(tuple,boundary));ring=[];current=int(boundary[0,0])
        while current not in ring:
            ring.append(current);current=int(following[current])
        if current!=ring[0] or len(ring)!=len(boundary):raise ValueError('Anatomical patch has multiple boundary loops')
        ring=np.asarray(ring,int);direction=np.asarray(forward,dtype=float)
        center=self.vertices[ring].mean(0);anchor=int(np.argmax((self.vertices[ring]-center)@direction));ring=np.roll(ring,-anchor)
        lengths=np.linalg.norm(self.vertices[np.roll(ring,-1)]-self.vertices[ring],axis=1)
        if lengths.sum()<=1e-12:raise ValueError('Collapsed anatomical boundary')
        phase=np.r_[0.,np.cumsum(lengths[:-1])]/lengths.sum()
        if boundary_parameter is not None:
            supplied=np.asarray(boundary_parameter,dtype=float)
            if supplied.shape!=(len(vertices),) or not np.isfinite(supplied).all():raise ValueError('Boundary parameters must match the original chart vertices')
            phase=supplied[self.vertex_ids[ring]]
            if np.any((phase<0)|(phase>=1)) or not np.isclose(np.mod(np.roll(phase,-1)-phase,1.).sum(),1.,atol=1e-10):
                raise ValueError('Boundary parameters must traverse the convex circle once in oriented order')
        self.boundary=ring;boundary_uv=np.column_stack((np.cos(2*np.pi*phase),np.sin(2*np.pi*phase)))
        rows=[];columns=[];values=[]
        p=self.vertices[self.triangles]
        for corner in range(3):
            a=p[:,(corner+1)%3]-p[:,corner];b=p[:,(corner+2)%3]-p[:,corner]
            la=np.linalg.norm(a,axis=1);lb=np.linalg.norm(b,axis=1)
            half=np.linalg.norm(np.cross(a,b),axis=1)/np.maximum(la*lb+np.einsum('ij,ij->i',a,b),1e-300)
            for offset,length in ((1,la),(2,lb)):
                rows.append(self.triangles[:,corner]);columns.append(self.triangles[:,(corner+offset)%3]);values.append(half/np.maximum(length,1e-300))
        W=sparse.coo_matrix((np.concatenate(values),(np.concatenate(rows),np.concatenate(columns))),shape=(n,n)).tocsr()
        degree=np.asarray(W.sum(1)).ravel()
        if np.any(degree<=0) or not np.isfinite(degree).all():raise ValueError('Degenerate anatomical chart triangles')
        L=sparse.eye(n,format='csr')-sparse.diags(1/degree)@W
        self.harmonic_matrix=L
        fixed=np.zeros(n,bool);fixed[ring]=True;free=~fixed
        uv=np.zeros((n,2));uv[ring]=boundary_uv
        uv[free]=splu(L[free][:,free].tocsc()).solve(-L[free][:,fixed]@uv[fixed])
        def signed_areas(values):
            t=values[self.triangles]
            return (t[:,1,0]-t[:,0,0])*(t[:,2,1]-t[:,0,1])-(t[:,1,1]-t[:,0,1])*(t[:,2,0]-t[:,0,0])
        initial_area=signed_areas(uv)
        if np.any(initial_area<=0):
            raise ValueError(f'Harmonic initialization is numerically singular: {int(np.count_nonzero(initial_area<=0))} invalid facets, minimum area {float(initial_area.min())}')
        if optimizer not in ('harmonic','slim'):raise ValueError('Unknown disk chart optimizer')
        if optimizer=='slim':
            try:import igl
            except ImportError as exc:raise RuntimeError('SLIM chart optimization requires optional libigl in the external worker') from exc
            area=np.linalg.norm(np.cross(p[:,1]-p[:,0],p[:,2]-p[:,0]),axis=1).sum()*.5
            normalized=(self.vertices-self.vertices.mean(0))*np.sqrt(np.pi/area)
            state=igl.slim_precompute(np.asfortranarray(normalized),np.asfortranarray(self.triangles,dtype=np.int32),
                np.asfortranarray(uv),igl.SYMMETRIC_DIRICHLET,ring.astype(np.int32),np.asfortranarray(boundary_uv),1e8)
            for _ in range(max(1,int(iterations))):
                candidate=np.asarray(igl.slim_solve(state,1)).copy()
                if not np.isfinite(candidate).all() or np.any(signed_areas(candidate)<=0):
                    raise ValueError('SLIM optimizer lost chart orientation; previous geometry is retained')
                uv=candidate
        # A distal geodesic anchor counters the very different distribution of
        # calf length versus toe detail in the two parameterizations.
        edge_length=np.linalg.norm(self.vertices[edges[:,0]]-self.vertices[edges[:,1]],axis=1)
        graph=sparse.coo_matrix((np.r_[edge_length,edge_length],(np.r_[edges[:,0],edges[:,1]],np.r_[edges[:,1],edges[:,0]])),shape=(n,n)).tocsr()
        distance=dijkstra(graph,directed=False,indices=ring,min_only=True)
        if not np.isfinite(distance).all():raise ValueError('Anatomical disk is disconnected')
        self.distal=int(np.argmax(distance)) if distal_point is None else int(np.argmin(np.linalg.norm(self.vertices-np.asarray(distal_point,dtype=float),axis=1)))
        z=uv[:,0]+1j*uv[:,1];a=z[self.distal]
        self.anchor=a
        self.phase=float(np.angle((z[ring[0]]-a)/(1-np.conj(a)*z[ring[0]])))
        # Recenter query coordinates analytically. Moving the triangle vertices
        # through a Mobius transform and reconnecting them by straight chords
        # can flip skinny triangles even though the analytic map is conformal.
        t=uv[self.triangles];signed=(t[:,1,0]-t[:,0,0])*(t[:,2,1]-t[:,0,1])-(t[:,1,1]-t[:,0,1])*(t[:,2,0]-t[:,0,0])
        if not np.isfinite(uv).all() or np.any(signed<=0):
            raise ValueError('Anatomical chart has folded or collapsed triangles '
                f'(optimizer={optimizer}, invalid={int(np.count_nonzero(signed<=0))}, '
                f'minimum_area={float(signed.min())}, distal_radius={float(abs(a))})')
        self.uv=uv;self.report=dict(vertices=n,triangles=len(face_ids),boundary_vertices=len(ring),optimizer=optimizer,
            boundary_strategy='supplied_monotone_circle_phase' if boundary_parameter is not None else 'arc_length_circle',
            distal_strategy='geodesic' if distal_point is None else 'supplied_anatomical_landmark',
            distal_vertex_id=int(self.vertex_ids[self.distal]),minimum_signed_uv_area=float(signed.min()),uv=fingerprint(uv))

    def normalized_queries(self,uv):
        uv=np.asarray(uv);z=uv[:,0]+1j*uv[:,1];a=self.anchor
        z=(z-a)/(1-np.conj(a)*z)*np.exp(-1j*self.phase)
        return np.column_stack((z.real,z.imag))

    def original_queries(self,uv):
        uv=np.asarray(uv);z=(uv[:,0]+1j*uv[:,1])*np.exp(1j*self.phase);a=self.anchor
        z=(z+a)/(1+np.conj(a)*z)
        return np.column_stack((z.real,z.imag))


def constrain_chart_landmarks(native,target,native_points,source_points,*,iterations=60):
    """Optimize an oriented native chart against interior surface landmarks.

    SLIM keeps the native chart locally oriented while fitting source-chart
    coordinates. Source points use exact triangle barycentrics; native points
    snap to existing canonical vertices, with the snap error reported. This
    supplies correspondence constraints, not 3D collision or rig acceptance.
    """
    import igl
    native_points=np.asarray(native_points,dtype=float);source_points=np.asarray(source_points,dtype=float)
    if native_points.ndim!=2 or native_points.shape[1]!=3 or native_points.shape!=source_points.shape or len(native_points)<2 or not np.isfinite(native_points).all() or not np.isfinite(source_points).all():
        raise ValueError('Provide at least two paired finite interior surface landmarks')
    nearest=np.argmin(np.linalg.norm(native.vertices[:,None,:]-native_points[None,:,:],axis=2),axis=0)
    if len(np.unique(nearest))!=len(nearest) or np.isin(nearest,native.boundary).any():raise ValueError('Interior landmarks must snap to distinct non-boundary canonical vertices')
    source_witness=SurfaceOracle(target.vertices,target.triangles).query(source_points,max_distance=float(np.linalg.norm(np.ptp(target.vertices,axis=0))*.05))
    if not source_witness['observed'].all():raise ValueError('An interior source landmark lies outside the selected source patch')
    goals=np.einsum('ni,nij->nj',source_witness['barycentric'],target.uv[target.triangles[source_witness['source_triangle_ids']]])
    boundary=target.original_queries(native.normalized_queries(native.uv[native.boundary]))
    boundary/=np.linalg.norm(boundary,axis=1)[:,None]
    constraints=np.r_[native.boundary,nearest].astype(np.int32);targets=np.vstack((boundary,goals))
    p=native.vertices[native.triangles];area=np.linalg.norm(np.cross(p[:,1]-p[:,0],p[:,2]-p[:,0]),axis=1).sum()*.5
    metric=(native.vertices-native.vertices.mean(0))*np.sqrt(np.pi/area)
    # Start inside the requested convex boundary. Asking the barrier optimizer
    # to move an unrelated boundary and interior landmarks simultaneously can
    # stall with positive areas but very large landmark errors.
    fixed=np.zeros(len(native.vertices),bool);fixed[native.boundary]=True;free=~fixed
    uv=np.zeros_like(native.uv);uv[native.boundary]=boundary
    uv[free]=splu(native.harmonic_matrix[free][:,free].tocsc()).solve(-native.harmonic_matrix[free][:,fixed]@uv[fixed])
    def check_orientation(values):
        t=values[native.triangles]
        signed=(t[:,1,0]-t[:,0,0])*(t[:,2,1]-t[:,0,1])-(t[:,1,1]-t[:,0,1])*(t[:,2,0]-t[:,0,0])
        if not np.isfinite(values).all() or np.any(signed<=0):raise ValueError('Landmark chart optimization lost orientation; previous character retained')
        return signed
    check_orientation(uv)
    initial_landmarks=uv[nearest].copy();stages=10 if iterations else 0
    history=[]
    for stage in range(stages):
        stage_goals=initial_landmarks+(goals-initial_landmarks)*(stage+1)/stages
        targets=np.vstack((boundary,stage_goals))
        state=igl.slim_precompute(np.asfortranarray(metric),np.asfortranarray(native.triangles,dtype=np.int32),
            np.asfortranarray(uv),igl.SYMMETRIC_DIRICHLET,constraints,np.asfortranarray(targets),1e8)
        for _ in range(max(1,int(iterations)//stages)):
            uv=np.asarray(igl.slim_solve(state,1)).copy();check_orientation(uv)
        history.append(float(np.linalg.norm(uv[nearest]-stage_goals,axis=1).max()))
    signed=check_orientation(uv)
    residual=np.linalg.norm(uv[nearest]-goals,axis=1)
    boundary_residual=np.linalg.norm(uv[native.boundary]-boundary,axis=1)
    # Positive triangle areas alone do not establish that soft constraints
    # converged. A stalled barrier solve must never become a character map.
    tolerance=1e-4
    if max(float(residual.max()),float(boundary_residual.max()))>tolerance:
        raise ValueError('Landmark chart constraints did not converge '
            f'(interior error={float(residual.max()):.6g}, boundary error={float(boundary_residual.max()):.6g}); previous character retained')
    return uv,dict(method='INTERIOR_LANDMARK_CONSTRAINED_SLIM_CHART_V2',landmarks=len(nearest),iterations=int(iterations),
        initialization='POSITIVE_HARMONIC_TARGET_BOUNDARY',continuation_stage_max_errors=history,
        native_local_vertex_ids=nearest.tolist(),native_snap_max_error=float(np.linalg.norm(native.vertices[nearest]-native_points,axis=1).max()),
        landmark_uv_max_error=float(residual.max()),minimum_signed_uv_area=float(signed.min()),
        source_surface_points=source_witness['surface_points'].tolist(),source_local_triangle_ids=source_witness['source_triangle_ids'].tolist(),
        source_barycentric=source_witness['barycentric'].tolist(),three_dimensional_review_required=True)


def map_disk_patch(base,current,triangles,source,source_triangles,native_face_ids,source_face_ids,*,forward=(0.,-1.,0.),collar_rings=6,chart_optimizer='harmonic',native_distal_point=None,source_distal_point=None,native_child_influence=None,source_child_influence=None,seam_mode='source_differentials',paired_landmarks=None):
    """Map a complete disk patch to source geometry, keeping a seam collar.

    The intrinsic distal anchor is not a semantic toe landmark. Rig behavior
    and 3D triangle quality require review even when source points are exact.
    """
    from .surface_fairing import fair,harmonic_displacement
    base=np.asarray(base,dtype=float);current=np.asarray(current,dtype=float);source=np.asarray(source,dtype=float)
    native_vertices,native_weld=np.unique(base,axis=0,return_inverse=True)
    source_vertices,source_weld=np.unique(source,axis=0,return_inverse=True)
    native_tri=native_weld[np.asarray(triangles,dtype=np.int64)];source_tri=source_weld[np.asarray(source_triangles,dtype=np.int64)]
    native=DiskChart(native_vertices,native_tri,native_face_ids,forward=forward,optimizer=chart_optimizer,distal_point=native_distal_point)
    target=DiskChart(source_vertices,source_tri,source_face_ids,forward=forward,optimizer=chart_optimizer,distal_point=source_distal_point)
    native_query=native.normalized_queries(native.uv)
    child_alignment=None
    if (native_child_influence is None)!=(source_child_influence is None):raise ValueError('Supply both native-vertex and source-face child influences')
    if native_child_influence is not None:
        nc=np.asarray(native_child_influence,dtype=float);sc=np.asarray(source_child_influence,dtype=float)
        if nc.shape!=(len(base),) or sc.shape!=(len(source_triangles),) or not np.isfinite(nc).all() or not np.isfinite(sc).all() or np.any((nc<-1e-12)|(nc>1+1e-12)) or np.any((sc<-1e-12)|(sc>1+1e-12)):
            raise ValueError('Child anatomical influence arrays must match their original surface order')
        nc=np.clip(nc,0.,1.);sc=np.clip(sc,0.,1.)
        nw=np.zeros(len(native_vertices));np.maximum.at(nw,native_weld,nc);nw=nw[native.vertex_ids]
        sf=np.asarray(source_face_ids,dtype=np.int64);st=source_tri[sf];sp=source_vertices[st]
        mass=np.linalg.norm(np.cross(sp[:,1]-sp[:,0],sp[:,2]-sp[:,0]),axis=1)
        sw=np.zeros(len(source_vertices));sm=np.zeros(len(source_vertices))
        for corner in range(3):
            np.add.at(sw,st[:,corner],mass*sc[sf]);np.add.at(sm,st[:,corner],mass)
        sw=(sw/np.maximum(sm,1e-300))[target.vertex_ids]
        def contour_radius(chart,weights):
            if weights[chart.distal]<.5:raise ValueError('Distal landmark must lie inside the labelled child region')
            t=chart.triangles;edges=np.unique(np.sort(np.vstack((t[:,[0,1]],t[:,[1,2]],t[:,[2,0]])),axis=1),axis=0)
            crossing=(weights[edges[:,0]]<.5)!=(weights[edges[:,1]]<.5);edges=edges[crossing]
            if len(edges)<3:raise ValueError('Child region lacks an internal anatomical contour')
            alpha=(.5-weights[edges[:,0]])/(weights[edges[:,1]]-weights[edges[:,0]])
            uv=chart.uv[edges[:,0]]*(1-alpha[:,None])+chart.uv[edges[:,1]]*alpha[:,None]
            radius=np.linalg.norm(chart.normalized_queries(uv),axis=1)
            length=np.linalg.norm(chart.vertices[edges[:,1]]-chart.vertices[edges[:,0]],axis=1)
            order=np.argsort(radius);q=np.cumsum(length[order]);value=float(radius[order[np.searchsorted(q,q[-1]*.5)]])
            if not 1e-6<value<1-1e-6:raise ValueError('Child contour is too close to the chart centre or outer boundary')
            return value
        nr=contour_radius(native,nw);sr=contour_radius(target,sw);r=np.linalg.norm(native_query,axis=1)
        mapped_r=np.where(r<=nr,r*sr/nr,sr+(r-nr)*(1-sr)/(1-nr))
        native_query*=np.divide(mapped_r,r,out=np.ones_like(r),where=r>1e-14)[:,None]
        child_alignment=dict(method='RIG_CHILD_CONTOUR_RADIAL_STATION_V1',native_radius=nr,source_radius=sr,
            limitation='Median contour radius; does not align individual fingers or the complete wrist curve')
    def radial_mass(chart):
        p=chart.vertices[chart.triangles];area=np.linalg.norm(np.cross(p[:,1]-p[:,0],p[:,2]-p[:,0]),axis=1)
        mass=np.zeros(len(chart.vertices))
        for corner in range(3):np.add.at(mass,chart.triangles[:,corner],area/3)
        radius=np.linalg.norm(chart.normalized_queries(chart.uv),axis=1);order=np.argsort(radius)
        cdf=(np.cumsum(mass[order])-.5*mass[order])/mass.sum()
        return radius,radius[order],cdf
    radius,native_r,native_cdf=radial_mass(native);_,source_r,source_cdf=radial_mass(target)
    # Conformal charts can compress a long thin calf by many orders of
    # magnitude. Balance radial surface mass before looking up source points.
    # A smoothed monotone log-radius map preserves distal/boundary anchors.
    positive=radius[radius>1e-14];knots=np.linspace(np.log(max(float(positive.min()),1e-14)),0,512)
    quantile=np.interp(np.exp(knots),native_r,native_cdf)
    mapped_radius=np.interp(quantile,source_cdf,source_r)
    log_mapped=gaussian_filter1d(np.log(np.maximum(mapped_radius,1e-14)),2.)
    log_mapped=np.maximum.accumulate(log_mapped);log_mapped[-1]=0
    adjusted=np.exp(np.interp(np.log(np.maximum(radius,1e-14)),knots,log_mapped));adjusted[radius<1e-14]=0
    adjusted[native.boundary]=1
    if chart_optimizer!='slim':native_query=native_query*(adjusted/np.maximum(radius,1e-300))[:,None]
    t=native_query[native.triangles];area=(t[:,1,0]-t[:,0,0])*(t[:,2,1]-t[:,0,1])-(t[:,1,1]-t[:,0,1])*(t[:,2,0]-t[:,0,0])
    if chart_optimizer!='slim' and np.any(area<=0):raise ValueError('Radial mass balancing folded the native chart')
    source_queries=target.original_queries(native_query)
    landmark_report=None
    if paired_landmarks is not None:
        if chart_optimizer!='slim' or child_alignment is not None:raise ValueError('Interior landmarks require SLIM charts without the experimental radial child warp')
        source_queries,landmark_report=constrain_chart_landmarks(native,target,paired_landmarks['native_points'],paired_landmarks['source_points'])
    uv_source=np.column_stack((target.uv,np.zeros(len(target.uv))));uv_query=np.column_stack((source_queries,np.zeros(len(source_queries))))
    witness=SurfaceOracle(uv_source,target.triangles).query(uv_query,max_distance=.03)
    if not witness['observed'].all():raise ValueError('Native disk extends outside the source chart')
    mapped=np.einsum('ni,nij->nj',witness['barycentric'],target.vertices[target.triangles[witness['source_triangle_ids']]])
    if landmark_report is not None:
        anchor_ids=np.asarray(landmark_report['native_local_vertex_ids'],dtype=int)
        landmark_report['source_surface_max_error']=float(np.linalg.norm(mapped[anchor_ids]-np.asarray(landmark_report['source_surface_points']),axis=1).max())
    counts=np.bincount(native_weld);physical_current=np.column_stack([np.bincount(native_weld,weights=current[:,i])/counts for i in range(3)])
    local_current=physical_current[native.vertex_ids]
    edge=np.unique(np.sort(np.vstack((native.triangles[:,[0,1]],native.triangles[:,[1,2]],native.triangles[:,[2,0]])),axis=1),axis=0)
    collar=np.zeros(len(mapped),bool);collar[native.boundary]=True
    for _ in range(int(collar_rings)):collar[np.unique(edge[np.any(collar[edge],axis=1)])]=True
    pinned=~collar;pinned[native.boundary]=True;goal=mapped.copy();goal[native.boundary]=local_current[native.boundary]
    if seam_mode=='harmonic_displacement':stitched,seam=harmonic_displacement(local_current,goal,native.triangles,pinned)
    elif seam_mode=='source_differentials':
        stitched,seam=fair(native.vertices,native.triangles,goal,np.zeros(len(mapped)),pinned,1.,reference=mapped,
            retention=1e-5,preserve_reference_differentials=True)
    else:raise ValueError('Unknown anatomical chart seam method')
    desired=physical_current.copy();desired[native.vertex_ids]=stitched
    touched=np.zeros(len(native_vertices),bool);touched[native.vertex_ids]=True
    output=current.copy();output[touched[native_weld]]=desired[native_weld[touched[native_weld]]]
    report=dict(method='ANATOMICAL_DISK_CHART_CORRESPONDENCE_PROTOTYPE_V1',native=native.report,source=target.report,
        anchor_transfer='INTERIOR_LANDMARK_SLIM_CHART' if landmark_report is not None else 'ANALYTIC_MOBIUS_QUERY_COMPOSITION',
        child_alignment=child_alignment,seam_mode=seam_mode,
        interior_landmarks=landmark_report,
        radial_surface_mass_balanced=chart_optimizer!='slim',balanced_minimum_uv_area=float(area.min()),
        collar_vertices=int(collar.sum()),source_core_vertices=int((~collar).sum()),outside_unchanged=True,
        max_chart_residual=float(witness['residual'].max()),seam=seam,production_ready=False,review_required=True)
    record=dict(native_physical_vertex_ids=native.vertex_ids,source_triangle_ids=target.face_ids[witness['source_triangle_ids']],
        barycentric=witness['barycentric'],mapped_source_points=mapped,core=~collar)
    return output,report,record


def boundary_loop(faces):
    faces=np.asarray(faces,dtype=np.int64)
    if faces.ndim!=2 or faces.shape[1]!=3 or not len(faces):raise ValueError('Choose a nonempty triangle region')
    edges=np.vstack((faces[:,[0,1]],faces[:,[1,2]],faces[:,[2,0]]));unique,count=np.unique(np.sort(edges,axis=1),axis=0,return_counts=True)
    border={tuple(e) for e in unique[count==1]};directed={int(i):int(j) for i,j in edges if tuple(sorted((i,j))) in border}
    if len(directed)!=len(border) or len(set(directed.values()))!=len(directed):raise ValueError('The surface region boundary is not a simple loop')
    if not directed:raise ValueError('The surface region has no boundary')
    loop=[min(directed)]
    while directed[loop[-1]]!=loop[0]:
        loop.append(directed[loop[-1]])
        if len(loop)>len(directed):raise ValueError('The surface region boundary is not a disk')
    if len(loop)!=len(directed):raise ValueError('The surface region has multiple boundary components')
    return np.asarray(loop,dtype=np.int64)

def ordered_angles(points, *, axes=(0,2), minimum_fraction=.01):
    """Correct projected angular reversals while preserving topological order."""
    points=np.asarray(points,dtype=float)
    if points.ndim!=2 or len(points)<3 or not np.isfinite(points).all() or not 0<minimum_fraction<1:raise ValueError('Choose a finite ordered surface boundary')
    q=points[:,axes].copy();q-=q.mean(0);span=np.ptp(q,axis=0)
    if np.any(span<=0):raise ValueError('The boundary does not span a projection plane')
    q/=span*.5;raw=np.arctan2(q[:,1],q[:,0]);delta=(np.roll(raw,-1)-raw+np.pi)%(2*np.pi)-np.pi;winding=int(round(float(delta.sum()/(2*np.pi))))
    if abs(winding)!=1:raise ValueError('The projected boundary does not have a single winding')
    oriented=raw*winding;steps=(np.roll(oriented,-1)-oriented+np.pi)%(2*np.pi)-np.pi
    if np.all(steps>0):return raw,dict(angular_reversals=0,max_angular_correction=0.)
    start=(int(np.argmax(steps))+1)%len(raw);order=np.roll(np.arange(len(raw)),-start);angles=np.unwrap(oriented[order]);spacing=2*np.pi/len(raw)*minimum_fraction;values=angles-np.arange(len(raw))*spacing;weights=np.ones(len(raw));weights[[0,-1]]=1e12;blocks=[]
    for i,(value,weight) in enumerate(zip(values,weights)):
        blocks.append([i,i,float(weight),float(weight*value)])
        while len(blocks)>1 and blocks[-2][3]/blocks[-2][2]>blocks[-1][3]/blocks[-1][2]:
            b=blocks.pop();a=blocks.pop();blocks.append([a[0],b[1],a[2]+b[2],a[3]+b[3]])
    corrected=np.empty(len(raw))
    for first,last,weight,value in blocks:corrected[first:last+1]=value/weight
    corrected+=np.arange(len(raw))*spacing
    if np.any(np.diff(corrected)<=0) or corrected[-1]-corrected[0]>=2*np.pi:raise ValueError('Could not order the surface boundary without overlap')
    result=np.empty(len(raw));result[order]=corrected*winding
    return result,dict(angular_reversals=int((steps<=0).sum()),max_angular_correction=float(np.max(abs(corrected-angles))))

def collar_weights(adjacency, region, core):
    """Blend a reconstructed feature into its unchanged collar with zero end slope."""
    from scipy import sparse
    from scipy.sparse.linalg import spsolve
    region=np.asarray(region,dtype=bool);core=np.asarray(core,dtype=bool)
    if region.shape!=core.shape or adjacency.shape!=(len(region),len(region)) or np.any(core&~region):raise ValueError('Feature core must lie inside its complete reconstruction region')
    outer=region&(np.asarray(adjacency@(~region).astype(float)).ravel()>0)
    if np.any(core&outer):raise ValueError('Feature core reaches the reconstruction boundary; expand its collar')
    free=region&~core&~outer;value=np.zeros(len(region));value[core]=1
    L=sparse.diags(np.asarray(adjacency.sum(1)).ravel())-adjacency
    if free.any():value[free]=spsolve(L[free][:,free].tocsc(),-L[free][:,~free]@value[~free])
    if not np.isfinite(value).all() or value.min()<-1e-8 or value.max()>1+1e-8:raise ValueError('Could not construct a bounded feature collar')
    value=np.clip(value,0,1)
    return value*value*(3-2*value)
