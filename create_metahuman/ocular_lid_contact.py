"""Bounded anatomical lid contact with a fixed outer smoothing boundary."""
import numpy as np
from .ocular_motion import collar_vertices
from scipy import sparse
from scipy.spatial import ConvexHull
from .ocular_overlay import depth_constraints
from .ocular_front_fragments import visible_fragments,material_topology


def _hull_triangles(points):
    hull=ConvexHull(points);tri=hull.simplices.copy();cross=np.cross(points[tri[:,1]]-points[tri[:,0]],points[tri[:,2]]-points[tri[:,0]]);flip=np.einsum('ij,ij->i',cross,hull.equations[:,:3])<0;tri[flip]=tri[flip][:,[0,2,1]]
    return tri


def _clear_projected(globe,skin,contact,support,fixed,clearance,maximum,seam,reference_skin,reference_globe,fairing_strength=0.,visibility_mode='NEUTRAL_MATERIAL'):
    """Exact projected triangle overlaps, with a bounded smooth skin field."""
    import clarabel
    tri=_hull_triangles(globe)
    # The return inside an open socket is intentionally behind the globe.
    # Calibrate the exterior material surface in neutral before moving lids.
    if visibility_mode=='POSE_EXTERIOR':
        # The caller supplies the pristine external sheet, excluding the socket
        # cap. Motion can expose material that was hidden by the neutral globe;
        # retain its current skin envelope instead of freezing neutral visibility.
        visible,visible_tri,interpolation,visibility=visible_fragments(skin,contact,support)
        reference_proof=None
    else:
        combined=np.vstack((reference_skin,reference_globe));occluders=np.vstack((support,_hull_triangles(reference_globe)+len(skin)))
        _,reference_tri,reference_map,reference_proof=visible_fragments(combined,contact,occluders);reference_map=reference_map[:,:len(skin)]
        material_skin=reference_map@skin;combined=np.vstack((material_skin,skin));visible,visible_tri,interpolation,visibility=visible_fragments(combined,reference_tri,support+len(material_skin));interpolation=interpolation[:,:len(material_skin)]@reference_map
    _,rhs,witness=depth_constraints(globe,tri,visible,visible_tri,clearance=clearance)
    active=np.unique(support);lookup=np.full(len(skin),-1,dtype=int);lookup[active]=np.arange(len(active))
    seam_ids=np.asarray(seam['vertices'],dtype=int) if seam else np.empty(0,dtype=int)
    if np.setdiff1d(seam_ids,active).size:raise ValueError('The selected sheet does not contain the complete closed lid seam')
    if np.intersect1d(fixed,seam_ids).size:raise ValueError('A shared lid seam cannot be fixed')
    free=np.setdiff1d(active,np.union1d(fixed,seam_ids));rows=list(lookup[free]);cols=list(range(len(free)));values=[1.]*len(free);variables=len(free)
    if seam:
        knots=np.asarray(seam['knots']);parameters=np.asarray(seam['parameters']);interval=np.clip(np.searchsorted(knots,parameters,side='right')-1,0,len(knots)-2);t=(parameters-knots[interval])/(knots[interval+1]-knots[interval])
        for vertex,i,weight in zip(seam_ids,interval,t):rows.extend([lookup[vertex]]*2);cols.extend([variables+int(i),variables+int(i)+1]);values.extend([1-weight,weight])
        variables+=len(knots)
    P=sparse.coo_matrix((values,(rows,cols)),shape=(len(active),variables)).tocsc()
    corners=visible_tri[witness['skin_faces']];V=sparse.coo_matrix((witness['skin_barycentric'].ravel(),(np.repeat(np.arange(len(rhs)),3),corners.ravel())),shape=(len(rhs),len(visible))).tocsr();B=(V@interpolation)[:,active]
    edges=np.unique(np.sort(np.vstack((support[:,[0,1]],support[:,[1,2]],support[:,[2,0]])),axis=1),axis=0);spacing=float(np.median(np.linalg.norm(reference_skin[edges[:,1]]-reference_skin[edges[:,0]],axis=1)));edges=lookup[edges]
    D=sparse.coo_matrix((np.tile([1.,-1.],len(edges)),(np.repeat(np.arange(len(edges)),2),edges.ravel())),shape=(len(edges),len(active))).tocsc()
    Q=P.T@(sparse.eye(len(active),format='csc')+2*D.T@D)@P;q=np.zeros(variables);curvature=None;topology=None
    if fairing_strength:
        material_tri,material_map,topology=material_topology(visible_tri,interpolation)
        visible_edges=np.unique(np.sort(np.vstack((material_tri[:,[0,1]],material_tri[:,[1,2]],material_tri[:,[2,0]])),axis=1),axis=0)
        adjacency=sparse.coo_matrix((np.ones(2*len(visible_edges)),(np.r_[visible_edges[:,0],visible_edges[:,1]],np.r_[visible_edges[:,1],visible_edges[:,0]])),shape=(material_map.shape[0],material_map.shape[0])).tocsr();degree=np.asarray(adjacency.sum(1)).ravel();valid=degree>0
        lap=sparse.diags(valid.astype(float))-sparse.diags(np.divide(1,degree,out=np.zeros_like(degree),where=valid))@adjacency;curvature=(lap@material_map[:,active]).tocsr();C=curvature@P;initial=np.asarray(lap@(material_map@(skin[:,1]-reference_skin[:,1]))).ravel();weight=fairing_strength/max(spacing,1e-8)**4;Q=Q+weight*C.T@C;q=-weight*np.asarray(C.T@initial).ravel()
    lower=-maximum if fairing_strength else 0.;A=sparse.vstack((-B@P,-P,P)).tocsc();bounds=np.r_[-rhs,np.full(len(active),-lower),np.full(len(active),maximum)]
    settings=clarabel.DefaultSettings();settings.verbose=False;settings.tol_feas=1e-10;settings.tol_gap_abs=1e-11
    solution=clarabel.DefaultSolver(sparse.triu(Q).tocsc(),q,A,bounds,[clarabel.NonnegativeConeT(len(bounds))],settings).solve()
    if str(solution.status) not in {'Solved','AlmostSolved','InsufficientProgress'}:raise ValueError(f'Bounded projected lid contact did not converge ({solution.status}); greatest initial depth deficit {rhs.max():.6g}')
    offset=np.asarray(P@np.asarray(solution.x)).ravel();residual=np.asarray(B@offset).ravel()-rhs
    # A narrow clipped polygon can stop interior-point progress before the
    # requested objective gap. Accept feasibility only after checking every
    # actual geometric witness and bound; solver status is not a certificate.
    if not np.isfinite(offset).all() or residual.min()<-1e-8 or offset.min()<lower-1e-8 or offset.max()>maximum+1e-8:raise ValueError(f'Projected lid contact failed its final clearance bounds ({solution.status}, residual {residual.min():.6g})')
    result=skin.copy();result[active,1]-=offset
    proof=dict(method='NEUTRAL_VISIBLE_LID_CONTACT',solver_status=str(solution.status),neutral_visibility=reference_proof,visibility=visibility,overlapping_triangle_pairs=witness['overlapping_triangle_pairs'],witnesses=len(rhs),minimum_depth_clearance=float(residual.min()+clearance),maximum_correction=float(offset.max()),clearance=clearance,shared_seam=seam is not None,scope='Neutral-visible anterior skin fragments versus convex globe frontal projection; not whole-motion or self-collision certification')
    proof['visibility_mode']=visibility_mode
    if visibility_mode=='POSE_EXTERIOR':proof.update(method='POSE_EXTERIOR_LID_CONTACT',scope='Current frontal envelope of supplied exterior lid sheet versus convex globe; not whole-motion, socket-return, or self-collision certification')
    if fairing_strength:proof.update(fairing_strength=fairing_strength,minimum_correction=float(offset.min()),neutral_edge_spacing=spacing,visible_depth_curvature_before=float(np.sum(initial**2)),visible_depth_curvature_after=float(np.sum((initial-curvature@offset)**2)),curvature_reference='NEUTRAL_DISPLACEMENT',material_topology=topology,two_sided_movement=True,projected_positions_unchanged=True)
    return result,proof


def seam_parameters(neutral,margin):
    neutral=np.asarray(neutral,dtype=float);margin=np.asarray(margin,dtype=int);first,last=sorted((int(np.argmin(neutral[margin,0])),int(np.argmax(neutral[margin,0]))));arcs=[margin[first:last+1],np.r_[margin[last:],margin[:first+1]]];parameters={}
    for arc in arcs:
        if neutral[arc[0],0]>neutral[arc[-1],0]:arc=arc[::-1]
        distance=np.r_[0.,np.cumsum(np.linalg.norm(np.diff(neutral[arc],axis=0),axis=1))]
        if distance[-1]<=0 or len(arc)<3:raise ValueError('The native lid seam has no usable arc length')
        parameters.update(zip(map(int,arc),map(float,distance/distance[-1])))
    return dict(vertices=margin,parameters=np.asarray([parameters[int(v)] for v in margin]),knots=np.linspace(0.,1.,min(map(len,arcs))))


def clear_contact(skin,sheet,margin,eye,*,neutral,neutral_eye,closed=False,contact_rings=3,halo_rings=6,clearance_fraction=.0005,maximum_fraction=.1,contact_radius_fraction=None,halo_radius_fraction=None,fairing_strength=0.,visibility_mode='NEUTRAL_MATERIAL'):
    """Resolve contact in lid depth without pulling socket returns onto the globe.

    Physical frame is X-right/Z-up/negative-Y-forward. Contact applies to the
    topological lid band, with a larger smooth halo pinned at its outer edge.
    The actual globe convex hull supplies projected-overlap constraints. This
    certifies the selected anterior triangles, not all poses/self-collisions.
    Contact alone moves skin anteriorly; optional displacement fairing permits
    bounded depth changes in either direction within the same contact constraints.
    POSE_EXTERIOR includes currently exposed exterior material. The caller must
    exclude the socket cap; neutral-material visibility remains the default.
    """
    if visibility_mode not in {'NEUTRAL_MATERIAL','POSE_EXTERIOR'}:raise ValueError('Choose neutral-material or posed-exterior visibility')
    skin=np.asarray(skin,dtype=float);sheet=np.asarray(sheet,dtype=int);margin=np.asarray(margin,dtype=int);eye=np.asarray(eye,dtype=float)
    if skin.ndim!=2 or skin.shape[1]!=3 or eye.ndim!=2 or eye.shape[1]!=3 or not np.isfinite(skin).all() or not np.isfinite(eye).all():raise ValueError('Supply finite skin and globe positions')
    if sheet.ndim!=2 or sheet.shape[1]!=3 or not len(sheet) or sheet.min()<0 or sheet.max()>=len(skin) or margin.ndim!=1 or len(margin)<4 or margin.min()<0 or margin.max()>=len(skin):raise ValueError('Supply the native external sheet and canonical lid margin')
    if type(contact_rings) is not int or type(halo_rings) is not int or not 1<=contact_rings<halo_rings<=8 or not 0<=clearance_fraction<maximum_fraction<=.5:raise ValueError('Choose a bounded contact band and a larger smoothing halo')
    if not np.isfinite(fairing_strength) or not 0<=fairing_strength<=1:raise ValueError('Choose a finite visible-surface fairing strength from zero through one')
    width=float(np.ptp(eye[:,0]))
    if width<=0:raise ValueError('The globe has no physical width')
    physical=contact_radius_fraction is not None or halo_radius_fraction is not None
    if physical:
        if contact_radius_fraction is None or halo_radius_fraction is None or not np.isfinite([contact_radius_fraction,halo_radius_fraction]).all() or not 0<contact_radius_fraction<halo_radius_fraction<=.5:raise ValueError('Choose a physical contact radius and a larger halo, each at most half the globe width')
        from .ocular_motion import intrinsic_collar
        band,band_proof=intrinsic_collar(neutral,sheet,margin,radius=width*contact_radius_fraction);halo,halo_proof=intrinsic_collar(neutral,sheet,margin,radius=width*halo_radius_fraction)
    else:band=collar_vertices(sheet,margin,rings=contact_rings);halo=collar_vertices(sheet,margin,rings=halo_rings)
    support=sheet[np.all(np.isin(sheet,halo),axis=1)];contact=sheet[np.all(np.isin(sheet,band),axis=1)];cross=np.cross(skin[contact[:,1]]-skin[contact[:,0]],skin[contact[:,2]]-skin[contact[:,0]]);contact=contact[cross[:,1]<-width*width*1e-12]
    if not len(contact):raise ValueError('The native lid band has no anterior contact triangles')
    # A retriangulated anatomical cut can have inner boundary vertices beside
    # the original margin. Pin only the outer halo, not these aperture vertices.
    edges,counts=np.unique(np.sort(np.vstack((support[:,[0,1]],support[:,[1,2]],support[:,[2,0]])),axis=1),axis=0,return_counts=True);fixed=np.setdiff1d(np.unique(edges[counts==1]),band)
    if not len(fixed):raise ValueError('The lid smoothing halo has no fixed outer boundary')
    neutral=np.asarray(neutral,dtype=float);neutral_eye=np.asarray(neutral_eye,dtype=float)
    if neutral.shape!=skin.shape or neutral_eye.shape!=eye.shape or not np.isfinite(neutral).all() or not np.isfinite(neutral_eye).all():raise ValueError('Supply corresponding neutral head and globe geometry')
    seam=None
    if closed:
        if neutral is None:raise ValueError('A closed contact seam needs corresponding neutral anatomy')
        neutral=np.asarray(neutral,dtype=float)
        if neutral.shape!=skin.shape or not np.isfinite(neutral).all():raise ValueError('Neutral seam anatomy does not match the current head')
        seam=seam_parameters(neutral,margin)
    origin=eye.mean(0);normalized=(skin-origin)/width;globe=(eye-origin)/width
    corrected,proof=_clear_projected(globe,normalized,contact,support,fixed,clearance_fraction,maximum_fraction,seam,(neutral-origin)/width,(neutral_eye-origin)/width,fairing_strength=fairing_strength,visibility_mode=visibility_mode)
    result=skin.copy();active=np.unique(support);result[active]=skin[active]+(corrected[active]-normalized[active])*width
    if not np.array_equal(result[fixed],skin[fixed]):raise ValueError('The contact solver changed its fixed boundary')
    proof.update(method='ANATOMICAL_LID_BAND_WITH_FIXED_HALO',contact_rings=contact_rings,halo_rings=halo_rings,contact_triangles=len(contact),fixed_vertices=len(fixed),fixed_boundary_policy='OUTSIDE_CONTACT_BAND',maximum_depth_change=float(np.max(np.abs(result[:,1]-skin[:,1]))),globe_geometry_changed=False,native_motion_accepted=False)
    if physical:proof.update(method='PHYSICAL_LID_BAND_WITH_FIXED_HALO',contact_domain=band_proof,halo_domain=halo_proof)
    return result,proof
