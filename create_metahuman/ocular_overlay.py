"""Linear depth clearance on overlays of selected frontal skin/globe triangles.

Unlike a ray grid, overlay vertices include projected edge intersections. The
depth difference is affine on each overlay polygon, so its extrema lie there.
The caller must select the actual anterior lid sheet; this is not a general
3D collision certificate or a substitute for testing animated geometry.
"""
import numpy as np
from scipy import sparse


def anterior_lid_sheet(data,eye):
    """Select the external sheet using the pristine native aperture topology."""
    from .ocular_surface import _aperture_margin
    basis=np.asarray(data['eye_basis'],dtype=float);tri=np.asarray(data['triangles'],dtype=np.int32)
    reference=np.asarray(data['parameter_reference'],dtype=float)
    tolerance=max(float(np.linalg.norm(np.ptp(reference,axis=0)))*1e-6,1e-9)
    _,weld=np.unique(np.rint(reference/tolerance).astype(np.int64),axis=0,return_inverse=True)
    count=np.bincount(weld);physical=lambda p:np.column_stack([np.bincount(weld,weights=p[:,j])/count for j in range(3)])
    faces=weld[tri];edges=np.unique(np.sort(np.vstack((faces[:,[0,1]],faces[:,[1,2]],faces[:,[2,0]])),axis=1),axis=0);edges=edges[edges[:,0]!=edges[:,1]]
    adjacency=sparse.coo_matrix((np.ones(len(edges)*2),(np.r_[edges[:,0],edges[:,1]],np.r_[edges[:,1],edges[:,0]])),shape=(len(count),len(count))).tocsr()
    cap_raw=np.asarray(data['eye_cap_'+str(eye)],dtype=int);cap=np.unique(weld[cap_raw])
    outside=np.setdiff1d(np.arange(len(reference)),cap_raw);rim=np.intersect1d(cap,weld[outside])
    anatomy=physical(np.asarray(data['anatomy_reference_head'])@basis)
    margin,interior,ring,calibration=_aperture_margin(anatomy,faces,adjacency,cap,rim,
        np.asarray(data['anatomy_eye_'+str(eye)])@basis,np.asarray(data['eye_triangles_'+str(eye)]))
    hidden=np.zeros(len(count),bool);hidden[np.setdiff1d(interior,margin)]=True
    selected=~np.any(hidden[faces],axis=1)
    return np.asarray(data['current'])@basis,tri[selected],dict(native_aperture_ring=ring,
        excluded_socket_vertices=int(hidden.sum()),selection='Pristine native aperture; excludes hidden socket cap',calibration=calibration)


def _clip(polygon,a,b,epsilon=1e-12):
    edge=b-a
    def side(p):return edge[0]*(p[1]-a[1])-edge[1]*(p[0]-a[0])
    result=[]
    for p,q in zip(polygon,np.roll(polygon,-1,axis=0)):
        sp,sq=side(p),side(q);inside=sp>=-epsilon
        if inside:result.append(p)
        if inside!=(sq>=-epsilon):
            denominator=sp-sq
            if abs(denominator)>1e-30:result.append(p+(q-p)*(sp/denominator))
    return np.asarray(result,dtype=float).reshape(-1,2)


def depth_constraints(eye,eye_triangles,skin,skin_triangles,*,clearance=0.,cancelled=None):
    """Return A, rhs for eye Y offsets, with fixed selected anterior skin.

    Input frame: X right, Z up, negative Y forward. Front-facing triangle
    projections must have positive signed XZ area. Vertices and offsets share
    input world units. Only projected polygons of nonzero area contribute.
    """
    eye=np.asarray(eye,dtype=float);skin=np.asarray(skin,dtype=float)
    et=np.asarray(eye_triangles,dtype=np.int32);st=np.asarray(skin_triangles,dtype=np.int32)
    if (eye.ndim!=2 or skin.ndim!=2 or eye.shape[1]!=3 or skin.shape[1]!=3
            or not np.isfinite(eye).all() or not np.isfinite(skin).all()
            or not np.isfinite(clearance) or clearance<0):raise ValueError('Invalid ocular clearance surfaces')
    if et.ndim!=2 or st.ndim!=2 or et.shape[1]!=3 or st.shape[1]!=3 or not len(et) or not len(st):raise ValueError('Clearance needs nonempty triangle meshes')
    if et.min()<0 or st.min()<0 or et.max()>=len(eye) or st.max()>=len(skin):raise ValueError('Clearance triangle index is outside its mesh')
    origin=eye.mean(0);scale=float(np.linalg.norm(np.ptp(eye[:,[0,2]],axis=0)))
    if scale<=1e-12:raise ValueError('Globe has zero frontal extent')
    e=(eye-origin)/scale;s=(skin-origin)/scale
    def area(v):return (v[:,1,0]-v[:,0,0])*(v[:,2,1]-v[:,0,1])-(v[:,1,1]-v[:,0,1])*(v[:,2,0]-v[:,0,0])
    ex=e[et][:,:,[0,2]];sx=s[st][:,:,[0,2]]
    ef=np.flatnonzero(area(ex)>1e-12);sf=np.flatnonzero(area(sx)>1e-12)
    sx=sx[sf];lo=sx.min(1);hi=sx.max(1)
    rows=[];columns=[];values=[];rhs=[];eye_faces=[];skin_faces=[];barys=[];skin_barys=[];witnesses=[];pairs=0
    for face in ef:
        if cancelled and cancelled():raise RuntimeError('Ocular clearance cancelled')
        tri=ex[face];candidates=np.flatnonzero(np.all(lo<=tri.max(0)+1e-12,axis=1)&np.all(hi>=tri.min(0)-1e-12,axis=1))
        E=np.column_stack((tri[1]-tri[0],tri[2]-tri[0]));Einv=np.linalg.inv(E)
        for k in candidates:
            polygon=tri.copy();clip=sx[k]
            for a,b in zip(clip,np.roll(clip,-1,axis=0)):
                polygon=_clip(polygon,a,b)
                if len(polygon)<3:break
            if len(polygon)<3:continue
            signed=np.sum(polygon[:,0]*np.roll(polygon[:,1],-1)-polygon[:,1]*np.roll(polygon[:,0],-1))
            if signed<1e-14:continue
            pairs+=1;skin_face=int(sf[k]);S=np.column_stack((clip[1]-clip[0],clip[2]-clip[0]))
            eb=(polygon-tri[0])@Einv.T;eb=np.c_[1-eb.sum(1),eb]
            sb=np.linalg.solve(S,(polygon-clip[0]).T).T;sb=np.c_[1-sb.sum(1),sb]
            delta=(sb@s[st[skin_face],1]-eb@e[et[face],1])*scale+clearance
            first=len(rhs);rows.extend(np.repeat(np.arange(first,first+len(polygon)),3));columns.extend(np.tile(et[face],len(polygon)));values.extend(eb.ravel());rhs.extend(delta)
            eye_faces.extend([int(face)]*len(polygon));skin_faces.extend([skin_face]*len(polygon));barys.extend(eb);skin_barys.extend(sb);witnesses.extend(polygon*scale+origin[[0,2]])
    if not rhs:raise ValueError('Selected frontal lid and globe surfaces have no overlapping projection')
    A=sparse.coo_matrix((values,(rows,columns)),shape=(len(rhs),len(eye))).tocsr()
    return A,np.asarray(rhs),dict(method='FRONTAL_TRIANGLE_OVERLAY_DEPTH',overlapping_triangle_pairs=pairs,
        witnesses=len(rhs),clearance=clearance,violations_before=int(np.sum(np.asarray(rhs)>1e-10)),
        maximum_depth_deficit=float(np.max(rhs)),eye_faces=np.asarray(eye_faces),skin_faces=np.asarray(skin_faces),
        eye_barycentric=np.asarray(barys),skin_barycentric=np.asarray(skin_barys),projection_points=np.asarray(witnesses))
