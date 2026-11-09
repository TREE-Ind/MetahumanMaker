"""Topology-based ocular pose repairs, independent of character identity.

The anterior depth solve covers projected triangle overlaps only. It does not
certify arbitrary 3D motion or replace native evaluated-pose acceptance.
"""
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve


def collar_vertices(faces,vertices,*,rings=2):
    """Grow a bounded topological repair collar without spatial aliases."""
    faces=np.asarray(faces,dtype=int);vertices=np.unique(np.asarray(vertices,dtype=int))
    if not isinstance(rings,int) or not 0<=rings<=8 or faces.ndim!=2 or faces.shape[1]!=3 or not len(vertices):raise ValueError('Invalid ocular collar')
    for _ in range(rings):vertices=np.unique(faces[np.any(np.isin(faces,vertices),axis=1)])
    return vertices


def intrinsic_collar(points,faces,margin,*,radius):
    """Select a physical surface-distance collar, independent of mesh density.

    Measures the actual neutral sheet; no nearest-neighbor welding or traversal
    through a hidden socket is introduced. Returns only complete patch vertices.
    """
    from scipy.sparse.csgraph import dijkstra
    points=np.asarray(points,dtype=float);faces=np.asarray(faces,dtype=int);margin=np.unique(np.asarray(margin,dtype=int))
    if points.ndim!=2 or points.shape[1]!=3 or not np.isfinite(points).all() or faces.ndim!=2 or faces.shape[1]!=3 or not len(faces) or faces.min()<0 or faces.max()>=len(points) or not len(margin) or margin.min()<0 or margin.max()>=len(points) or not np.isfinite(radius) or radius<=0:raise ValueError('A physical lid collar requires finite anatomy, a complete margin and positive radius')
    edges=np.unique(np.sort(np.vstack((faces[:,[0,1]],faces[:,[1,2]],faces[:,[2,0]])),axis=1),axis=0);length=np.linalg.norm(points[edges[:,1]]-points[edges[:,0]],axis=1)
    if np.any(length<=0):raise ValueError('A physical collar cannot traverse collapsed anatomical edges')
    graph=sparse.coo_matrix((np.r_[length,length],(np.r_[edges[:,0],edges[:,1]],np.r_[edges[:,1],edges[:,0]])),shape=(len(points),len(points))).tocsr();distance=dijkstra(graph,directed=False,indices=margin,min_only=True,limit=float(radius));selected=np.flatnonzero(distance<=radius);patch=faces[np.all(np.isin(faces,selected),axis=1)]
    if not len(patch):raise ValueError('The physical collar has no complete surface triangles')
    support=np.unique(patch)
    if not np.isin(margin,support).all():raise ValueError('Increase the physical collar radius to include the complete lid margin')
    return support,dict(method='INTRINSIC_SURFACE_DISTANCE_COLLAR',radius=float(radius),vertices=len(support),triangles=len(patch),maximum_distance=float(distance[support].max()),physical_coordinates_preserved=True)


def projected_support(skin,faces,eye):
    """Select external-sheet faces over a posed globe, excluding rear skin.

    The caller supplies an anatomical external sheet with hidden socket faces
    already removed. Bounds follow the posed globe rather than a fixed ring.
    """
    skin=np.asarray(skin,dtype=float);faces=np.asarray(faces,dtype=int);eye=np.asarray(eye,dtype=float)
    corners=skin[faces];xz=corners[:,:,[0,2]];low=eye[:,[0,2]].min(0);high=eye[:,[0,2]].max(0)
    overlap=np.all(xz.min(1)<=high,axis=1)&np.all(xz.max(1)>=low,axis=1)
    anterior=np.mean(corners[:,:,1],axis=1)<float(np.mean(eye[:,1])+np.ptp(eye[:,0])*.4)
    selected=faces[overlap&anterior]
    if not len(selected):raise ValueError('No external lid support covers the posed globe')
    return np.unique(selected),selected


def follow_gaze(neutral,reference_eye,posed_eye,faces,collar,margin,*,fraction=.3):
    """Transfer a fraction of rigid globe rotation to the aperture collar.

    Scale follows the actual globe and neutral aperture. The outer collar is
    fixed and the displacement is continued harmonically from the margin.
    Clearance against the posed globe must be solved separately afterwards.
    """
    from scipy.spatial.transform import Rotation
    neutral=np.asarray(neutral,dtype=float);eye=np.asarray(reference_eye,dtype=float);posed=np.asarray(posed_eye,dtype=float);faces=np.asarray(faces,dtype=int)
    if eye.shape!=posed.shape or not 0<=fraction<=1 or not np.isfinite(eye).all() or not np.isfinite(posed).all():raise ValueError('Invalid gaze-follow samples')
    a=eye-eye.mean(0);b=posed-posed.mean(0);u,_,vh=np.linalg.svd(a.T@b);fix=np.ones(3);fix[-1]=np.linalg.det(u@vh);rotation=(u*fix)@vh
    error=float(np.linalg.norm(a@rotation-b,axis=1).max())
    width=float(np.ptp(eye[:,0]))
    if width<=0 or error>width*1e-4:raise ValueError('Gaze follow needs a corresponding rigid globe sample')
    posterior=eye[eye[:,1]>=np.median(eye[:,1])];origin=posterior.mean(0);q=posterior-origin
    center=origin+np.linalg.lstsq(np.c_[2*q,np.ones(len(q))],np.sum(q*q,axis=1),rcond=None)[0][:3]
    turn=Rotation.from_rotvec(Rotation.from_matrix(rotation).as_rotvec()*fraction).as_matrix()
    collar=np.unique(collar);margin=np.unique(margin);patch=faces[np.all(np.isin(faces,collar),axis=1)]
    edges,counts=np.unique(np.sort(np.vstack((patch[:,[0,1]],patch[:,[1,2]],patch[:,[2,0]])),axis=1),axis=0,return_counts=True)
    boundary=np.unique(edges[counts==1]);outer=np.setdiff1d(boundary,margin);pins=np.union1d(outer,margin);free=np.setdiff1d(collar,pins)
    if not len(outer) or not np.isin(margin,collar).all():raise ValueError('Aperture collar has no fixed outer boundary')
    adjacency=sparse.coo_matrix((np.ones(len(edges)*2),(np.r_[edges[:,0],edges[:,1]],np.r_[edges[:,1],edges[:,0]])),shape=(len(neutral),len(neutral))).tocsr();lap=sparse.diags(np.asarray(adjacency.sum(1)).ravel())-adjacency
    delta=np.zeros_like(neutral);moved=(neutral[margin]-center)@turn+center
    # Keep canthi fixed while allowing the central upper/lower lids to follow.
    t=(neutral[margin,0]-neutral[margin,0].min())/np.ptp(neutral[margin,0]);weight=np.sin(np.pi*t)**2
    delta[margin]=(moved-neutral[margin])*weight[:,None]
    delta[free]=spsolve(lap[free][:,free].tocsc(),-lap[free][:,pins]@delta[pins])
    if not np.isfinite(delta).all():raise ValueError('Gaze-follow collar is disconnected')
    return neutral+delta,dict(method='RIGID_GLOBE_ROTATION_APERTURE_FOLLOW',rotation_fraction=fraction,rigid_sample_error=error,maximum_movement=float(np.linalg.norm(delta,axis=1).max()))


def continue_socket(neutral,posed,faces,interior,margin,*,footprint=False,injective=False,cancelled=None):
    """Continue the posed aperture displacement through its hidden socket.

    Preserve the neutral socket shape with a harmonic displacement field.
    This removes inherited socket motion that can fold through a resized eye.
    The supplied anatomical margin and all exterior vertices remain exact.
    An optional harmonic X/Z footprint removes residual projected folds in the
    hidden cap. It retains the neutral depth displacement continuation; callers
    must still check the complete resulting surface for intersections.
    """
    neutral=np.asarray(neutral,dtype=float);posed=np.asarray(posed,dtype=float);faces=np.asarray(faces,dtype=int)
    interior=np.unique(np.asarray(interior,dtype=int));margin=np.unique(np.asarray(margin,dtype=int))
    if neutral.shape!=posed.shape or neutral.ndim!=2 or neutral.shape[1]!=3 or not np.isfinite(neutral).all() or not np.isfinite(posed).all():raise ValueError('Socket continuation needs finite corresponding poses')
    if not len(margin) or not np.isin(margin,interior).all() or interior.min()<0 or interior.max()>=len(neutral):raise ValueError('Invalid anatomical socket boundary')
    patch=faces[np.all(np.isin(faces,interior),axis=1)]
    if not len(patch):raise ValueError('Socket boundary has no interior surface')
    edges=np.unique(np.sort(np.vstack((patch[:,[0,1]],patch[:,[1,2]],patch[:,[2,0]])),axis=1),axis=0)
    free=np.setdiff1d(interior,margin);n=len(neutral)
    adjacency=sparse.coo_matrix((np.ones(len(edges)*2),(np.r_[edges[:,0],edges[:,1]],np.r_[edges[:,1],edges[:,0]])),shape=(n,n)).tocsr()
    lap=sparse.diags(np.asarray(adjacency.sum(1)).ravel())-adjacency
    delta=spsolve(lap[free][:,free].tocsc(),-lap[free][:,margin]@(posed[margin]-neutral[margin]))
    if not np.isfinite(delta).all():raise ValueError('Socket is disconnected from its anatomical margin')
    result=posed.copy();result[free]=neutral[free]+delta
    if footprint or injective:
        result[np.ix_(free,[0,2])]=spsolve(lap[free][:,free].tocsc(),-lap[free][:,margin]@posed[margin][:,[0,2]])
        if not np.isfinite(result).all():raise ValueError('The socket footprint is disconnected from its anatomical margin')
    embedding=None
    if injective:
        from .socket_embedding import injective_footprint
        result,embedding=injective_footprint(neutral,result,faces,interior,margin,cancelled=cancelled)
    return result,dict(method='INJECTIVE_SOCKET_CONTINUATION' if injective else 'HARMONIC_SOCKET_FOOTPRINT' if footprint else 'HARMONIC_SOCKET_DISPLACEMENT',interior_vertices=len(free),boundary_vertices=len(margin),embedding=embedding,maximum_change=float(np.linalg.norm(result-posed,axis=1).max()))


def close_aperture(neutral,anatomy,faces,collar,margin,volume,*,upper=.15,lower=.5):
    """Close upper/lower aperture arcs over the fitted globe for a squint."""
    from scipy.interpolate import PchipInterpolator
    from .ocular_globe_volume import front_depth
    neutral=np.asarray(neutral,dtype=float);anatomy=np.asarray(anatomy,dtype=float);faces=np.asarray(faces,dtype=int);margin=np.asarray(margin,dtype=int);collar=np.unique(collar)
    if not 0<=upper<1 or not 0<=lower<1 or upper+lower>=1:raise ValueError('A partial aperture must retain a positive opening')
    first,last=sorted((int(np.argmin(anatomy[margin,0])),int(np.argmax(anatomy[margin,0]))));arcs=[margin[first:last+1],np.r_[margin[last:],margin[:first+1]]];high,low=sorted(arcs,key=lambda a:float(anatomy[a,2].mean()),reverse=True)
    def curve(ids):
        x,index=np.unique(neutral[ids,0],return_index=True)
        return PchipInterpolator(x,neutral[ids[index],2],extrapolate=True)
    hi,lo=curve(high),curve(low);x=neutral[margin,0];gap=np.maximum(hi(x)-lo(x),0);t=(x-x.min())/np.ptp(x);taper=np.sin(np.pi*t)**2
    goal=neutral[margin].copy();goal[:,2]+=np.where(np.isin(margin,high),-upper,lower)*gap*taper
    depth,inside=front_depth(goal[:,[0,2]],volume);goal[inside,1]=np.minimum(goal[inside,1],depth[inside]-np.ptp(x)*.0005)
    patch=faces[np.all(np.isin(faces,collar),axis=1)];edges,counts=np.unique(np.sort(np.vstack((patch[:,[0,1]],patch[:,[1,2]],patch[:,[2,0]])),axis=1),axis=0,return_counts=True);outer=np.setdiff1d(np.unique(edges[counts==1]),margin);pins=np.union1d(outer,margin);free=np.setdiff1d(collar,pins)
    adjacency=sparse.coo_matrix((np.ones(len(edges)*2),(np.r_[edges[:,0],edges[:,1]],np.r_[edges[:,1],edges[:,0]])),shape=(len(neutral),len(neutral))).tocsr();lap=sparse.diags(np.asarray(adjacency.sum(1)).ravel())-adjacency
    delta=np.zeros_like(neutral);delta[margin]=goal-neutral[margin];delta[free]=spsolve(lap[free][:,free].tocsc(),-lap[free][:,pins]@delta[pins])
    if not np.isfinite(delta).all():raise ValueError('Aperture closure collar is disconnected')
    return neutral+delta,dict(method='VOLUME_GUIDED_PARTIAL_APERTURE',upper_fraction=upper,lower_fraction=lower,maximum_movement=float(np.linalg.norm(delta,axis=1).max()))


def fitted_blink_pose(current,open_pose,anatomy,faces,regions,*,closure):
    """Close fitted lids at the requested gaze, including their socket tissue.

    Regions supply anatomical collars, margins, interiors, fitted globe frames,
    and evaluated globe geometry. The caller supplies corresponding physical
    vertices; render splits and native DNA identities stay outside this solve.
    This is a pose target, requiring native interpolation validation afterward.
    """
    current=np.asarray(current,dtype=float);open_pose=np.asarray(open_pose,dtype=float)
    if current.shape!=open_pose.shape or not np.isfinite(closure) or not 0<closure<=1:
        raise ValueError('Fitted blink requires corresponding poses and positive closure')
    target=current.copy();reports=[]
    for region in regions:
        collar=np.asarray(region['collar']);margin=np.asarray(region['margin']);interior=np.asarray(region['interior']);sheet=np.asarray(region['sheet']);volume=region['volume'];globe=np.asarray(region['globe']);width=float(np.ptp(globe[:,0]))
        goal,blink=blink_aperture(open_pose,anatomy,faces,collar,margin,volume,closure=closure,meeting_fraction=region.get('meeting_fraction'),closed_anatomy=region.get('closed_anatomy'),shared_seam=region.get('shared_seam',False))
        target,join=continue_pose_patch(target,goal,faces,np.union1d(collar,interior),np.union1d(margin,interior),reference=goal)
        target,fair=fair_lid_surface(target,sheet,collar,margin,volume,maximum_offset=width*.15*closure,screening=.005,enforce_globe=region.get('fair_contact',True))
        ids,selected=projected_support(target,sheet,globe)
        if region.get('convex_contact',False):
            from .ocular_convex_contact import clear_skin
            target,contact=clear_skin(globe,target,selected,clearance=width*.0005,maximum_offset=width*.3,seam=blink.get('shared_seam') if closure==1. else None)
        else:target,contact=anterior_clearance(globe,region['globe_triangles'],target,selected,ids,clearance=width*.0005,maximum_offset=width*.3,include_folded=True)
        target,socket=continue_socket(open_pose,target,faces,interior,margin)
        target,recess=recess_socket(target,interior,margin,volume,maximum_offset=volume['radii'][0])
        reports.append(dict(blink=blink,join=join,fairing=fair,contact=contact,socket=socket,recess=recess))
    if not reports:raise ValueError('Fitted blink needs at least one anatomical eye region')
    return target,dict(method='GAZE_CONDITIONED_FITTED_BLINK',regions=reports,native_interpolation_accepted=False)


def blink_meeting_fraction(reference,closed,margin):
    """Infer lid travel from corresponding native neutral and blink anatomy.

    Fit one dimensionless meeting fraction to the reference seam. Aperture
    height weights the fit naturally; nearly closed canthi do not dominate it.
    No source character dimensions or hand-picked closing ratio are used.
    """
    from scipy.interpolate import PchipInterpolator
    reference=np.asarray(reference,dtype=float);closed=np.asarray(closed,dtype=float);margin=np.asarray(margin,dtype=int)
    if reference.shape!=closed.shape or reference.ndim!=2 or reference.shape[1]!=3 or not np.isfinite(reference).all() or not np.isfinite(closed).all() or len(margin)<6 or len(np.unique(margin))!=len(margin) or margin.min()<0 or margin.max()>=len(reference):raise ValueError('Native blink calibration needs corresponding finite anatomy and an ordered aperture')
    first,last=sorted((int(np.argmin(reference[margin,0])),int(np.argmax(reference[margin,0]))));arcs=[margin[first:last+1],np.r_[margin[last:],margin[:first+1]]];upper,lower=sorted(arcs,key=lambda ids:float(reference[ids,2].mean()),reverse=True);parameter=np.linspace(0,1,65)
    def samples(ids,points):
        if reference[ids[0],0]>reference[ids[-1],0]:ids=ids[::-1]
        distance=np.r_[0.,np.cumsum(np.linalg.norm(np.diff(reference[ids],axis=0),axis=1))]
        if distance[-1]<=0:raise ValueError('Native blink aperture has a collapsed arc')
        values,selected=np.unique(distance/distance[-1],return_index=True)
        if len(values)<3:raise ValueError('Native blink aperture has too few distinct samples')
        return PchipInterpolator(values,points[ids[selected]],axis=0,extrapolate=False)(parameter)
    high,low=samples(upper,reference),samples(lower,reference);posed_high,posed_low=samples(upper,closed),samples(lower,closed)
    gap=high[:,2]-low[:,2];width=float(np.ptp(reference[margin,0]));valid=gap>width*1e-4
    if width<=0 or valid.sum()<8:raise ValueError('Native blink calibration has no usable aperture height')
    displacement=(posed_high[:,2]+posed_low[:,2])*.5-low[:,2]
    raw=float(gap[valid]@displacement[valid]/(gap[valid]@gap[valid]));fraction=float(np.clip(raw,0,1))
    return fraction,dict(method='NATIVE_MOTION_LID_TRAVEL',fraction=fraction,unbounded_fraction=raw,bound_applied=raw!=fraction,samples=int(valid.sum()),
        seam_fit_relative_rms=float(np.sqrt(np.mean((displacement[valid]-fraction*gap[valid])**2))/width),
        reference_closed_gap_relative=float(np.median((posed_high[valid,2]-posed_low[valid,2])/gap[valid])))


def blink_aperture(neutral,anatomy,faces,collar,margin,volume,*,closure=1.,meeting_fraction=None,closed_anatomy=None,shared_seam=False):
    """Close an anatomical aperture over its globe, retaining a finite seam.

    Both arcs approach the same source-conditioned meeting curve. Unlike a
    squint taper, a full blink closes the entire aperture, including near its
    canthi. Biharmonic displacement keeps the surrounding transition smooth.
    Actual triangle clearance and posed rig acceptance remain separate.
    """
    from scipy.interpolate import PchipInterpolator
    from .ocular_globe_volume import front_depth
    neutral=np.asarray(neutral,dtype=float);anatomy=np.asarray(anatomy,dtype=float);faces=np.asarray(faces,dtype=int);collar=np.unique(collar);margin=np.asarray(margin,dtype=int)
    meeting_reference=None
    if meeting_fraction is None:
        if closed_anatomy is None:raise ValueError('Supply native closed-lid anatomy or an explicit lid meeting fraction')
        meeting_fraction,meeting_reference=blink_meeting_fraction(anatomy,closed_anatomy,margin)
    if neutral.shape!=anatomy.shape or neutral.ndim!=2 or neutral.shape[1]!=3 or not np.isfinite(neutral).all() or not np.isfinite(anatomy).all() or not np.isfinite(closure) or not 0<=closure<=1 or not np.isfinite(meeting_fraction) or not 0<=meeting_fraction<=1:
        raise ValueError('Blink needs finite corresponding anatomy and a bounded closure')
    if len(margin)<4 or not np.isin(margin,collar).all():raise ValueError('Blink needs a complete ordered margin inside its collar')
    first,last=sorted((int(np.argmin(anatomy[margin,0])),int(np.argmax(anatomy[margin,0]))));arcs=[margin[first:last+1],np.r_[margin[last:],margin[:first+1]]];upper,lower=sorted(arcs,key=lambda ids:float(anatomy[ids,2].mean()),reverse=True)
    if neutral[upper[0],0]>neutral[upper[-1],0]:upper=upper[::-1]
    if neutral[lower[0],0]>neutral[lower[-1],0]:lower=lower[::-1]
    def curve(ids):
        distance=np.r_[0.,np.cumsum(np.linalg.norm(np.diff(neutral[ids],axis=0),axis=1))]
        if distance[-1]<=0:raise ValueError('A blink arc has no usable extent')
        parameter=distance/distance[-1];unique,selected=np.unique(parameter,return_index=True)
        if len(unique)<3:raise ValueError('A blink arc has too few distinct points')
        return PchipInterpolator(unique,neutral[ids[selected]],axis=0,extrapolate=False),parameter
    high,upper_t=curve(upper);low,lower_t=curve(lower);parameter=np.zeros(len(margin));lookup_margin={int(v):i for i,v in enumerate(margin)}
    knots=None
    if shared_seam:
        # Both arcs must represent the SAME polygonal seam. Independent
        # samples of a curved seam leave a real hole on coarse native LODs.
        if len(upper)<=len(lower):
            paired=np.rint(np.linspace(0,len(lower)-1,len(upper))).astype(int);knots=(upper_t+lower_t[paired])*.5;lower_t=np.interp(lower_t,lower_t[paired],knots);upper_t=knots.copy()
        else:
            paired=np.rint(np.linspace(0,len(upper)-1,len(lower))).astype(int);knots=(lower_t+upper_t[paired])*.5;upper_t=np.interp(upper_t,upper_t[paired],knots);lower_t=knots.copy()
    for arc,values in ((upper,upper_t),(lower,lower_t)):
        for vertex,value in zip(arc,values):parameter[lookup_margin[int(vertex)]]=value
    x=neutral[margin,0];width=float(np.ptp(x));high_points,low_points=high(parameter),low(parameter);gap=np.linalg.norm(high_points-low_points,axis=1)
    if width<=0 or gap.max(initial=0)<=width*1e-5:raise ValueError('The fitted eye opening is already collapsed')
    # Use bounded arc parameters, never extrapolated world-X polynomials.
    # Fitted stylized margins need not be monotone in world X at their canthi.
    seam=low_points+meeting_fraction*(high_points-low_points);clearance=np.minimum(width*.0002,gap*.001)
    if shared_seam:
        seam_knots=low(knots)+meeting_fraction*(high(knots)-low(knots));seam=np.column_stack([np.interp(parameter,knots,seam_knots[:,axis]) for axis in range(3)]);clearance=np.zeros_like(gap)
    seam[:,2]+=np.where(np.isin(margin,upper),.5,-.5)*clearance
    goal=neutral[margin]+closure*(seam-neutral[margin])
    if shared_seam and closure==1.:goal=seam.copy()
    depth,inside=front_depth(goal[:,[0,2]],volume);moving=inside&(gap>width*1e-5)
    goal[moving,1]+=closure*(np.minimum(goal[moving,1],depth[moving]-width*.0005)-goal[moving,1])
    patch=faces[np.all(np.isin(faces,collar),axis=1)];edges,counts=np.unique(np.sort(np.vstack((patch[:,[0,1]],patch[:,[1,2]],patch[:,[2,0]])),axis=1),axis=0,return_counts=True);boundary=np.setdiff1d(np.unique(edges[counts==1]),margin)
    # Pin two boundary rings so the moving patch does not develop a sharp
    # slope where it meets the unchanged surrounding skin.
    outer=np.setdiff1d(np.unique(edges[np.any(np.isin(edges,boundary),axis=1)]),margin);pins=np.union1d(outer,margin);free=np.setdiff1d(collar,pins)
    if not len(boundary) or not len(free):raise ValueError('Blink collar has no free skin transition')
    adjacency=sparse.coo_matrix((np.ones(len(edges)*2),(np.r_[edges[:,0],edges[:,1]],np.r_[edges[:,1],edges[:,0]])),shape=(len(neutral),len(neutral))).tocsr();lap=sparse.diags(np.asarray(adjacency.sum(1)).ravel())-adjacency
    # Restrict the operator to the collar before squaring: unrelated face
    # vertices cannot become accidental zero-displacement constraints.
    local=lap[collar][:,collar];energy=(local.T@local+local*.2+sparse.eye(len(collar))*1e-8).tocsr();lookup=np.full(len(neutral),-1);lookup[collar]=np.arange(len(collar));f,p=lookup[free],lookup[pins]
    delta=np.zeros_like(neutral);delta[margin]=goal-neutral[margin]
    import clarabel
    Q=sparse.triu(energy[f][:,f]*2).tocsc();A=sparse.vstack((sparse.eye(len(f)),-sparse.eye(len(f)))).tocsc();settings=clarabel.DefaultSettings();settings.verbose=False;settings.tol_feas=1e-10;settings.tol_gap_abs=1e-11
    for axis in range(3):
        prescribed=delta[pins,axis];bound=np.r_[np.full(len(f),prescribed.max(initial=0)),np.full(len(f),-prescribed.min(initial=0))]
        solve=clarabel.DefaultSolver(Q,np.asarray(2*energy[f][:,p]@prescribed).ravel(),A,bound,[clarabel.NonnegativeConeT(len(bound))],settings).solve()
        if str(solve.status)!='Solved':raise ValueError('The bounded blink transition did not converge')
        delta[free,axis]=solve.x
    if not np.isfinite(delta).all():raise ValueError('Blink transition could not be solved')
    return neutral+delta,dict(method='VOLUME_GUIDED_BIHARMONIC_BLINK',closure=closure,meeting_fraction=meeting_fraction,meeting_reference=meeting_reference,seam_gap_world=0. if shared_seam else width*.0002,maximum_movement=float(np.linalg.norm(delta,axis=1).max()),arc_parameterization='SHARED_POLYGONAL_SEAM' if shared_seam else 'BOUNDED_ARCLENGTH',shared_seam=dict(vertices=margin.tolist(),parameters=parameter.tolist(),knots=knots.tolist()) if shared_seam else None,displacement_bounds='PRESCRIBED_MARGIN_COMPONENT_RANGE')


def fair_lid_surface(posed,faces,support,margin,volume,*,maximum_offset,screening=.03,enforce_globe=True):
    """Fair an external posed lid while retaining its aperture and outer join.

    Unlike displacement smoothing, absolute-surface curvature minimization can
    remove inherited folds. Bounds limit each coordinate's movement; a final
    depth constraint keeps free vertices anterior to the fitted globe. Triangle
    overlap and expression interpolation still require separate acceptance.
    """
    import clarabel
    from .ocular_globe_volume import front_depth
    posed=np.asarray(posed,dtype=float);faces=np.asarray(faces,dtype=int);support=np.unique(support);margin=np.unique(margin)
    if posed.ndim!=2 or posed.shape[1]!=3 or not np.isfinite(posed).all() or not np.isfinite(maximum_offset) or maximum_offset<=0 or not np.isfinite(screening) or screening<=0:
        raise ValueError('Lid fairing requires finite positions and positive bounds')
    if not len(support) or support.min()<0 or support.max()>=len(posed) or not np.isin(margin,support).all():raise ValueError('Lid fairing needs a bounded anatomical support')
    patch=faces[np.all(np.isin(faces,support),axis=1)]
    if not len(patch):raise ValueError('The external lid support has no complete triangles')
    edges,counts=np.unique(np.sort(np.vstack((patch[:,[0,1]],patch[:,[1,2]],patch[:,[2,0]])),axis=1),axis=0,return_counts=True)
    # The aperture is an inner boundary. Treating its neighbours as fixed
    # freezes the very lid rows that need fairing, especially on lower LODs.
    boundary=np.setdiff1d(np.unique(edges[counts==1]),margin);outer=np.setdiff1d(np.unique(edges[np.any(np.isin(edges,boundary),axis=1)]),margin)
    pins=np.union1d(outer,margin);free=np.setdiff1d(support,pins)
    if not len(boundary):raise ValueError('The external lid support has no fairing boundary')
    if not len(free):
        return posed.copy(),dict(method='FIXED_BOUNDARY_LID_SURFACE',free_vertices=0,fixed_vertices=len(pins),maximum_change=0.,performed=False,scope='No smoothing degrees of freedom; triangle contact remains required')
    lookup=np.full(len(posed),-1);lookup[support]=np.arange(len(support));local_edges=lookup[edges];n=len(support)
    adjacency=sparse.coo_matrix((np.ones(len(edges)*2),(np.r_[local_edges[:,0],local_edges[:,1]],np.r_[local_edges[:,1],local_edges[:,0]])),shape=(n,n)).tocsr()
    degree=np.asarray(adjacency.sum(1)).ravel()
    if np.any(degree==0):raise ValueError('The lid fairing support is disconnected')
    lap=sparse.eye(n)-sparse.diags(1/degree)@adjacency;curvature=(lap.T@lap).tocsr();energy=curvature+sparse.eye(n)*screening;f=lookup[free]
    Q=sparse.triu(2*energy[f][:,f]).tocsc();A=sparse.vstack((sparse.eye(len(f)),-sparse.eye(len(f)))).tocsc();local=posed[support]-posed[support].mean(0);gradient=2*curvature@local
    settings=clarabel.DefaultSettings();settings.verbose=False;settings.tol_feas=1e-10;settings.tol_gap_abs=1e-11
    result=posed.copy();limit=maximum_offset/np.sqrt(3)
    # Solve X/Z first, then constrain anterior depth at the resulting X/Z.
    for axis in (0,2,1):
        upper=np.full(len(f),limit);lower=np.full(len(f),-limit)
        if axis==1 and enforce_globe:
            depth,inside=front_depth(result[free][:,[0,2]],volume);padding=float(volume['radii'][0])*.001
            upper[inside]=np.minimum(upper[inside],depth[inside]-padding-posed[free[inside],1])
            if np.any(upper<lower):raise ValueError('Globe clearance exceeds the bounded lid fairing region')
        solution=clarabel.DefaultSolver(Q,np.asarray(gradient[f,axis]).ravel(),A,np.r_[upper,-lower],[clarabel.NonnegativeConeT(2*len(f))],settings).solve()
        if str(solution.status)!='Solved':raise ValueError('The bounded lid fairing did not converge')
        result[free,axis]+=np.asarray(solution.x)
    before=float(np.sum((lap@local)**2));after=float(np.sum((lap@(result[support]-posed[support].mean(0)))**2))
    return result,dict(method='BOUNDED_ABSOLUTE_LID_FAIRING',free_vertices=len(free),fixed_vertices=len(pins),maximum_change=float(np.linalg.norm(result-posed,axis=1).max()),curvature_energy_before=before,curvature_energy_after=after,aperture_and_outer_join_unchanged=True,globe_depth_constraint=enforce_globe,scope='External selected vertex constraints; not whole-motion triangle collision certification')


def continue_pose_patch(current,desired,faces,support,core,*,reference=None):
    """Join an aperture repair to the actual posed surrounding face.

    Core vertices match the repair exactly; the external support boundary
    remains on the native expression. Hidden core boundaries take precedence.
    """
    current=np.asarray(current,dtype=float);desired=np.asarray(desired,dtype=float);faces=np.asarray(faces,dtype=int);support=np.unique(support);core=np.unique(core)
    rest=current if reference is None else np.asarray(reference,dtype=float)
    if rest.shape!=current.shape or not np.isfinite(rest).all():raise ValueError('Pose repair reference must match the face')
    if current.shape!=desired.shape or not np.isin(core,support).all():raise ValueError('Pose repair core must belong to its support')
    patch=faces[np.all(np.isin(faces,support),axis=1)];edges,counts=np.unique(np.sort(np.vstack((patch[:,[0,1]],patch[:,[1,2]],patch[:,[2,0]])),axis=1),axis=0,return_counts=True)
    boundary=np.setdiff1d(np.unique(edges[counts==1]),core);pins=np.union1d(core,boundary);free=np.setdiff1d(support,pins)
    if not len(boundary):raise ValueError('Pose repair needs an external boundary')
    adjacency=sparse.coo_matrix((np.ones(len(edges)*2),(np.r_[edges[:,0],edges[:,1]],np.r_[edges[:,1],edges[:,0]])),shape=(len(current),len(current))).tocsr();lap=sparse.diags(np.asarray(adjacency.sum(1)).ravel())-adjacency
    delta=np.zeros_like(current);delta[core]=desired[core]-rest[core];delta[boundary]=current[boundary]-rest[boundary];delta[free]=spsolve(lap[free][:,free].tocsc(),-lap[free][:,pins]@delta[pins])
    if not np.isfinite(delta).all():raise ValueError('Pose repair support is disconnected')
    result=current.copy();result[support]=rest[support]+delta[support]
    return result,dict(method='NATIVE_POSE_BOUNDARY_CONTINUATION',reference='CURRENT_POSE' if reference is None else 'REST_DIFFERENTIAL_SHAPE',core_vertices=len(core),boundary_vertices=len(boundary),maximum_correction=float(np.linalg.norm(result-current,axis=1).max()))


def recess_socket(posed,interior,margin,volume,*,maximum_offset):
    """Keep the inferred hidden socket behind an opaque convex globe.

    The visible margin stays exact. This is an occlusion repair for hidden
    socket geometry, not a certificate of physical tissue/globe separation.
    """
    posed=np.asarray(posed,dtype=float);free=np.setdiff1d(interior,margin);center=np.asarray(volume['center']);r=np.asarray(volume['radii']);shear=np.abs(volume['depth_shear'])
    limit=float(center[1]+shear[0]*r[0]+shear[1]*r[2]+r[1]*.001)
    offset=np.maximum(limit-posed[free,1],0)
    if not np.isfinite(offset).all() or maximum_offset<=0 or offset.max(initial=0)>maximum_offset:raise ValueError('Hidden socket recess exceeds its bounded correction: '+str(float(offset.max(initial=0)))+' > '+str(maximum_offset))
    result=posed.copy();result[free,1]+=offset
    return result,dict(method='POSTERIOR_SOCKET_OCCLUSION',vertices=len(free),maximum_retreat=float(offset.max(initial=0)),visible_margin_unchanged=True)


def anterior_clearance(eye,eye_triangles,skin,skin_triangles,movable,*,clearance,maximum_offset,include_folded=False):
    """Pull the selected anterior lid forward using exact overlay witnesses.

    Fixed X/Z coordinates make every overlapping triangle constraint linear.
    A bounded smooth depth correction preserves other anatomy and the globe.
    """
    from .ocular_overlay import depth_constraints
    import clarabel
    skin=np.asarray(skin,dtype=float);tri=np.asarray(skin_triangles,dtype=int);active=np.unique(np.asarray(movable,dtype=int))
    if not np.isfinite(maximum_offset) or maximum_offset<=0 or not len(active) or active.min()<0 or active.max()>=len(skin):raise ValueError('Invalid bounded lid correction region')
    folded=0
    if include_folded:
        # Include selected external-sheet folds in the depth constraints.
        # Reorientation is for projection math only; mesh winding is retained.
        xz=skin[tri][:,:,[0,2]];a=xz[:,1]-xz[:,0];b=xz[:,2]-xz[:,0];reverse=a[:,0]*b[:,1]-a[:,1]*b[:,0]<0
        folded=int(reverse.sum());tri=tri.copy();tri[reverse]=tri[reverse][:,[0,2,1]]
    _,rhs,proof=depth_constraints(eye,eye_triangles,skin,tri,clearance=clearance)
    vertices=tri[proof['skin_faces']];weights=proof['skin_barycentric'];lookup=np.full(len(skin),-1);lookup[active]=np.arange(len(active));indices=lookup[vertices]
    rows=np.repeat(np.arange(len(rhs)),3);valid=indices.ravel()>=0
    A=sparse.coo_matrix((weights.ravel()[valid],(rows[valid],indices.ravel()[valid])),shape=(len(rhs),len(active))).tocsc()
    edges=np.unique(np.sort(np.vstack((tri[:,[0,1]],tri[:,[1,2]],tri[:,[2,0]])),axis=1),axis=0);edges=lookup[edges];edges=edges[np.all(edges>=0,axis=1)]
    D=sparse.coo_matrix((np.tile([1.,-1.],len(edges)),(np.repeat(np.arange(len(edges)),2),edges.ravel())),shape=(len(edges),len(active))).tocsr()
    Q=sparse.eye(len(active),format='csc')+2*(D.T@D)
    constraints=sparse.vstack((A,sparse.eye(len(active)),-sparse.eye(len(active)))).tocsc();bound=np.r_[-rhs,np.zeros(len(active)),np.full(len(active),maximum_offset)]
    settings=clarabel.DefaultSettings();settings.verbose=False;settings.tol_feas=1e-10;settings.tol_gap_abs=1e-11
    solution=clarabel.DefaultSolver(sparse.triu(Q).tocsc(),np.zeros(len(active)),constraints,bound,[clarabel.NonnegativeConeT(len(bound))],settings).solve()
    offset=np.asarray(solution.x)
    if str(solution.status)!='Solved' or np.max(A@offset+rhs)>1e-8:raise ValueError('Lid clearance exceeds its bounded region; retain current native rig')
    result=skin.copy();result[active,1]+=offset
    return result,dict(method='ANTERIOR_TRIANGLE_OVERLAY_LID_DEPTH',witnesses=len(rhs),reoriented_projection_faces=folded,maximum_depth_change=float(np.max(np.abs(offset))),maximum_constraint_residual=float(np.max(A@offset+rhs)),scope='Selected frontal sheets; not a whole-motion collision certificate')
