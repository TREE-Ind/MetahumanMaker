"""Bilateral attachment proposals from source-backed anatomical sections.

This is a coarse registration hypothesis, not a semantic anatomy classifier.
Correspondence points lie on the supplied original triangle meshes. No character IDs,
world axes, dimensions or hand-picked vertex indices enter the detector.
"""
import numpy as np
from .vertex_contract import fingerprint


def _section(vertices,triangles,face_ids,frame,membership):
    vertices=np.asarray(vertices,dtype=float);triangles=np.asarray(triangles,dtype=np.int64)
    origin,lateral,up=frame;_,weld=np.unique(vertices,axis=0,return_inverse=True)
    p=vertices[triangles[face_ids]];signed=(p-origin)@lateral
    scale=2*np.linalg.norm(vertices-vertices.mean(0),axis=1).max();tol=scale*1e-10
    active=(signed.min(1)<=tol)&(signed.max(1)>=-tol)
    nodes={};positions=[];segments=[];parents=[];barycentrics=[];labels=[]
    for row in np.flatnonzero(active):
        ids=triangles[face_ids[row]];distance=signed[row];endpoints={}
        if np.all(np.abs(distance)<=tol):continue
        for a,b in ((0,1),(1,2),(2,0)):
            for corner in (a,b):
                if abs(distance[corner])<=tol:
                    key=(int(weld[ids[corner]]),)*2;weights=np.zeros(3);weights[corner]=1;endpoints[key]=weights
            if distance[a]*distance[b]<0 and abs(distance[a])>tol and abs(distance[b])>tol:
                t=float(distance[a]/(distance[a]-distance[b]));weights=np.zeros(3);weights[a]=1-t;weights[b]=t
                key=tuple(sorted((int(weld[ids[a]]),int(weld[ids[b]]))));endpoints[key]=weights
        if len(endpoints)!=2:continue
        edge=[];bary=[]
        for key,weights in endpoints.items():
            if key not in nodes:nodes[key]=len(positions);positions.append(weights@vertices[ids])
            edge.append(nodes[key]);bary.append(weights)
        segments.append(edge);parents.append(int(face_ids[row]));barycentrics.append(bary);labels.append(membership[row])
    if not segments:raise ValueError('No supported anatomical section intersects this paired region')
    points=np.asarray(positions);edges=np.asarray(segments);parents=np.asarray(parents);barycentrics=np.asarray(barycentrics);labels=np.asarray(labels)
    _,first,inverse=np.unique(np.sort(edges,axis=1),axis=0,return_index=True,return_inverse=True)
    merged=np.zeros((len(first),labels.shape[1]),bool);np.logical_or.at(merged,inverse,labels);order=np.argsort(first);unique=first[order]
    edges=edges[unique];parents=parents[unique];barycentrics=barycentrics[unique];labels=merged[order]
    roots=np.arange(len(points))
    def root(i):
        while roots[i]!=i:roots[i]=roots[roots[i]];i=roots[i]
        return i
    for a,b in edges:roots[root(int(a))]=root(int(b))
    components=np.array([root(int(a)) for a in edges[:,0]]);choices=[]
    for component in np.unique(components):
        ids=np.flatnonzero(components==component)
        if not np.all(np.any(labels[ids],axis=0)):continue
        length=float(np.linalg.norm(points[edges[ids,0]]-points[edges[ids,1]],axis=1).sum())
        choices.append((length,ids))
    choices.sort(key=lambda item:item[0],reverse=True)
    if not choices:raise ValueError('The attachment section must connect both paired anatomical regions')
    if len(choices)>1 and choices[1][0]>.8*choices[0][0]:raise ValueError('Multiple comparable bilateral sections; refine the source masks before registering')
    rows=choices[0][1];height=(points[edges[rows]]-origin)@up;row,corner=np.unravel_index(np.argmin(height),height.shape);row=int(rows[row])
    face=int(parents[row]);bary=barycentrics[row,corner];point=np.einsum('ni,nij->nj',bary[None,:],vertices[triangles[face]][None,:,:])[0]
    return dict(point=point,triangle_id=face,barycentric=bary,section_length=choices[0][0],bilateral_components=len(choices),height=float((point-origin)@up),curve_points=points[edges[rows]],curve_faces=parents[rows],curve_barycentric=barycentrics[rows])


def _basin_anchor(section,vertices,triangles,frame,window):
    """Integrate a quadratic height-window density along original sections."""
    p=section['curve_points'];edge=p[:,1]-p[:,0];length=np.linalg.norm(edge,axis=1)
    h=(p-frame[0])@frame[2];minimum=section['height'];a=1-(h[:,0]-minimum)/window;b=-(h[:,1]-h[:,0])/window
    lo=np.zeros(len(p));hi=np.ones(len(p));positive=b>1e-15;negative=b<-1e-15
    lo[positive]=np.clip(-a[positive]/b[positive],0,1);hi[negative]=np.clip(-a[negative]/b[negative],0,1)
    valid=(hi>lo)&(length>0)&((a>0)|positive)
    def integral(t):return a*a*t+a*b*t*t+b*b*t**3/3
    def moment(t):return a*a*t*t/2+2*a*b*t**3/3+b*b*t**4/4
    mass=np.where(valid,np.maximum(integral(hi)-integral(lo),0)*length,0);first=np.where(valid,(moment(hi)-moment(lo))*length,0)
    if mass.sum()<=0:raise ValueError('Anatomical section has no supported lowest basin')
    center=np.sum(p[:,0]*mass[:,None]+edge*first[:,None],axis=0)/mass.sum()
    t=np.clip(np.einsum('ij,ij->i',center-p[:,0],edge)/np.maximum(length**2,1e-30),lo,hi)
    distance=np.sum((p[:,0]+edge*t[:,None]-center)**2,axis=1);distance[~valid]=np.inf;row=int(np.argmin(distance))
    bary=(1-t[row])*section['curve_barycentric'][row,0]+t[row]*section['curve_barycentric'][row,1];face=int(section['curve_faces'][row])
    point=np.einsum('ni,nij->nj',bary[None,:],vertices[triangles[face]][None,:,:])[0]
    section.update(point=point,triangle_id=face,barycentric=bary,basin_centroid=center,basin_mass=float(mass.sum()),basin_window=float(window))


def _basin_controls(section,frame,window,*,samples=9):
    """Sample the lowest section envelope across its intrinsic depth span.

    Segment/plane intersections retain original facet barycentrics. Clipping
    by height before measuring the span keeps distant torso geometry out of
    the basin. Sampling does not depend on imported vertex density.
    """
    p=section['curve_points'];origin,lateral,up=frame;depth=np.cross(lateral,up)
    height=(p-origin)@up;cap=section['height']+float(window)
    change=height[:,1]-height[:,0];lo=np.zeros(len(p));hi=np.ones(len(p))
    rising=change>1e-15;falling=change<-1e-15
    hi[rising]=np.minimum(1,(cap-height[rising,0])/change[rising])
    lo[falling]=np.maximum(0,(cap-height[falling,0])/change[falling])
    valid=(hi>=lo)&(height.min(1)<=cap)
    if not valid.any():raise ValueError('No supported coarse basin envelope')
    edge=p[:,1]-p[:,0];axial=(p-origin)@depth;step=axial[:,1]-axial[:,0]
    a=axial[:,0]+step*lo;b=axial[:,0]+step*hi
    low=np.minimum(a,b);high=np.maximum(a,b)
    lower=float(low[valid].min());upper=float(high[valid].max());span=upper-lower
    if span<=window*1e-8:raise ValueError('The attachment basin has no supported depth extent')
    tolerance=span*1e-10;points=[];faces=[];barycentrics=[]
    for station in np.linspace(lower,upper,int(samples)):
        eligible=valid&(low<=station+tolerance)&(high>=station-tolerance)
        t=np.where(change<0,hi,lo).copy();moving=abs(step)>span*1e-12
        t[moving]=(station-axial[moving,0])/step[moving]
        t=np.clip(t,lo,hi);h=height[:,0]+t*change;h[~eligible]=np.inf
        if not eligible.any():raise ValueError('The coarse basin envelope has an unsupported depth gap')
        row=int(np.argmin(h));value=float(t[row]);points.append(p[row,0]+value*edge[row])
        faces.append(section['curve_faces'][row]);barycentrics.append((1-value)*section['curve_barycentric'][row,0]+value*section['curve_barycentric'][row,1])
    return dict(points=np.asarray(points),triangle_ids=np.asarray(faces,dtype=np.int64),barycentric=np.asarray(barycentrics),depth_span=float(span))


def propose(reference,triangles,source,source_triangles,influences,source_policies,frame):
    """Match the lowest basins of unique bilateral anatomical sections."""
    reference=np.asarray(reference,dtype=float);source=np.asarray(source,dtype=float);tri=np.asarray(triangles,dtype=np.int64);st=np.asarray(source_triangles,dtype=np.int64);frame=np.asarray(frame,dtype=float);influences=np.asarray(influences,dtype=float)
    if reference.ndim!=2 or reference.shape[1]!=3 or source.ndim!=2 or source.shape[1]!=3 or frame.shape!=(3,3) or influences.shape!=(2,len(reference)) or not all(np.isfinite(a).all() for a in (reference,source,frame,influences)) or np.any((influences<0)|(influences>1)):raise ValueError('Invalid bilateral attachment arrays')
    if not np.allclose(frame[1:]@frame[1:].T,np.eye(2),rtol=0,atol=1e-8):raise ValueError('Attachment frame needs orthogonal unit lateral/up directions')
    if len(source_policies)!=2:raise ValueError('Attachment registration needs two paired source regions')
    for policy in source_policies:
        if np.asarray(policy).dtype.kind not in 'iu' or np.asarray(policy).ndim!=1 or not len(policy):raise ValueError('Attachment source policies must name original triangles')
    for points,faces in ((reference,tri),(source,st)):
        if faces.ndim!=2 or faces.shape[1]!=3 or not len(faces) or faces.min()<0 or faces.max()>=len(points):raise ValueError('Invalid attachment triangles')
    free=np.max(influences,axis=0)>1e-4;target_faces=np.flatnonzero(np.all(free[tri],axis=1))
    if not len(target_faces):raise ValueError('Select a fully movable bilateral attachment region')
    target_membership=np.stack([np.any(values[tri[target_faces]]>1e-4,axis=1) for values in influences],axis=1)
    source_faces=np.unique(np.concatenate(list(source_policies)))
    if not len(source_faces) or source_faces.min()<0 or source_faces.max()>=len(st):raise ValueError('Invalid paired source attachment policies')
    sides=[]
    for policy in source_policies:
        p=source[st[policy]];area=np.linalg.norm(np.cross(p[:,1]-p[:,0],p[:,2]-p[:,0]),axis=1)
        if area.sum()<=0:raise ValueError('Attachment source policies need nonzero surface area')
        sides.append(float(np.average((p.mean(1)-frame[0])@frame[1],weights=area)))
    if sides[0]*sides[1]>=0:raise ValueError('Paired source thighs must lie on opposite sides of the reference anatomical plane')
    source_membership=np.stack([np.isin(source_faces,ids) for ids in source_policies],axis=1)
    target=_section(reference,tri,target_faces,frame,target_membership);original=_section(source,st,source_faces,frame,source_membership)
    window=min(target['section_length'],original['section_length'])*.02
    _basin_anchor(target,reference,tri,frame,window);_basin_anchor(original,source,st,frame,window)
    guide_window=min(target['section_length'],original['section_length'])*.1
    controls=dict(target=_basin_controls(target,frame,guide_window),source=_basin_controls(original,frame,guide_window))
    return dict(target=target,source=original,controls=controls,report=dict(method='RIG_FRAME_BILATERAL_SECTION_ATTACHMENT_V3',
        source_point=original['point'].tolist(),target_point=target['point'].tolist(),source_triangle_id=original['triangle_id'],source_barycentric=original['barycentric'].tolist(),target_triangle_id=target['triangle_id'],target_barycentric=target['barycentric'].tolist(),
        gap_m=float(np.linalg.norm(original['point']-target['point'])),height_gap_m=target['height']-original['height'],source_section_length_m=original['section_length'],target_section_length_m=target['section_length'],
        source_snapshot=fingerprint(source),source_topology=fingerprint(st),frame=frame.tolist(),source_policy_signed_centroids_m=sides,
        anchor_policy='LOWEST_BASIN_ARCLENGTH_CENTROID_PROJECTED_TO_SECTION_V1',basin_window_m=window,source_basin_centroid=original['basin_centroid'].tolist(),target_basin_centroid=target['basin_centroid'].tolist(),
        guide_policy='PAIRED_BASIN_LOWER_ENVELOPE_DEPTH_STATIONS_V1',guide_window_m=guide_window,
        guide_controls={key:{name:value.tolist() if isinstance(value,np.ndarray) else value for name,value in control.items()} for key,control in controls.items()},anatomical_hypothesis_requires_review=True))


def coherent_guide(reference,triangles,influence,attachment):
    """A smooth, boundary-pinned coarse correspondence guide, not a commit.

    A Gaussian residual field spreads paired basin controls across the paired
    region. Boundary samples are selected deterministically in physical vertex
    order, without world-axis bins. Displacement magnitude is bounded by the
    largest control shift. Actual surface fitting still
    needs source witnesses and the complete geometric/native acceptance gates.
    """
    from .geometry import rbf_displacement
    reference=np.asarray(reference,dtype=float);tri=np.asarray(triangles,dtype=np.int64);influence=np.asarray(influence,dtype=float)
    if reference.ndim!=2 or reference.shape[1]!=3 or influence.shape!=(len(reference),) or not np.isfinite(reference).all() or not np.isfinite(influence).all() or np.any((influence<0)|(influence>1)):raise ValueError('Invalid attachment guide arrays')
    free=influence>1e-4;edges=np.unique(np.sort(np.vstack((tri[:,[0,1]],tri[:,[1,2]],tri[:,[2,0]])),axis=1),axis=0)
    mixed=free[edges[:,0]]!=free[edges[:,1]];boundary=np.unique(edges[mixed][~free[edges[mixed]]])
    if not len(boundary):raise ValueError('Coarse attachment registration needs a protected surrounding boundary')
    origins=attachment['controls']['target']['points'];destinations=attachment['controls']['source']['points'];gap=float(np.linalg.norm(destinations-origins,axis=1).max())
    if gap<=1e-12:return reference.copy(),dict(method='BOUNDARY_PINNED_BASIN_CONTOUR_GUIDE_V2',maximum_guide_displacement_m=0.,protected_coordinates_exact=True)
    radius=max(gap,attachment['target']['section_length']*.25);threshold=radius**2*1e-12
    if np.min(np.sum((reference[boundary,None]-origins[None])**2,axis=2))<=threshold:raise ValueError('An attachment contour control coincides with a protected boundary')
    sample=boundary[np.linspace(0,len(boundary)-1,min(128,len(boundary)),dtype=int)];pins=[]
    for vertex in sample:
        if not pins or np.min(np.sum((reference[pins]-reference[vertex])**2,axis=1))>threshold:pins.append(int(vertex))
    shifts=destinations-origins
    spacing=np.linalg.norm(origins[:,None]-origins[None],axis=2);np.fill_diagonal(spacing,np.inf)
    steps=max(16,min(128,int(np.ceil(16*gap/max(float(spacing.min()),radius*1e-3)))))
    guide=reference.copy();maximum_speed=0.
    # Advect through a time-varying smooth field rather than applying a large
    # displacement once. Moving controls follow straight source-backed paths;
    # this numerical guide still requires the actual fitter's safety checks.
    for step in range(steps):
        moving=origins+shifts*(step/steps)
        controls=np.vstack((moving,reference[pins]));destinations=np.vstack((moving+shifts/steps,reference[pins]))
        raw=rbf_displacement(guide,controls,destinations,radius,regularization=1e-5)
        displacement=raw-guide;length=np.linalg.norm(displacement,axis=1)
        displacement*=np.minimum(1,(gap/steps)/np.maximum(length,1e-30))[:,None]*influence[:,None]
        guide+=displacement;guide[~free]=reference[~free]
        maximum_speed=max(maximum_speed,float(np.linalg.norm(displacement,axis=1).max()*steps))
    return guide,dict(method='BOUNDARY_PINNED_BASIN_CONTOUR_FLOW_V3',radius_m=radius,boundary_control_points=len(pins),contour_control_points=len(origins),maximum_control_shift_m=gap,integration_steps=steps,maximum_speed_m=maximum_speed,
        maximum_guide_displacement_m=float(np.linalg.norm(guide-reference,axis=1).max()),protected_coordinates_exact=True,
        displacement_magnitude_bounded=True,guide_is_not_source_evidence=True)
