"""NumPy-only source-angle contracts shared by workers and native validation."""
import numpy as np


def attachment_contract(reference,triangles,source,source_triangles,source_faces,eligible,movable,metric):
    """Original, coherent source normals; never derive authority from a chart."""
    triangles=np.asarray(triangles,dtype=np.int64);source_faces=np.asarray(source_faces,dtype=np.int64)
    eligible=np.asarray(eligible,dtype=bool);movable=np.asarray(movable,dtype=bool)
    observed=eligible&(source_faces>=0);field=np.zeros_like(reference)
    facets=np.asarray(source)[np.asarray(source_triangles)[source_faces[observed]]]
    normals=np.cross(facets[:,1]-facets[:,0],facets[:,2]-facets[:,0]);normals/=np.maximum(np.linalg.norm(normals,axis=1)[:,None],1e-30)
    field[observed]=normals;face_field=field[triangles].sum(1);size=np.linalg.norm(face_field,axis=1)
    supported=np.all(observed[triangles],axis=1)&np.any(movable[triangles],axis=1)&(size>=2.4)
    directions=face_field/np.maximum(size[:,None],1e-30)
    contract=angle_contract(reference,triangles,directions,supported,metric,preserve_aligned=True)
    return directions,supported,contract


def angle_contract(reference,triangles,normals,supported,metric,*,minimum=.25,reserve=.5,preserve_aligned=False):
    """Protect initially sound source-facing triangles, with angular reserve."""
    reference=np.asarray(reference,dtype=float);tri=np.asarray(triangles,dtype=np.int64);normals=np.asarray(normals,dtype=float);metric=np.asarray(metric,dtype=float);supported=np.asarray(supported,dtype=bool)
    if metric.shape!=reference.shape or reference.ndim!=2 or reference.shape[1]!=3 or tri.ndim!=2 or tri.shape[1]!=3 or normals.shape!=(len(tri),3) or supported.shape!=(len(tri),) or not 0<minimum<=reserve<1:raise ValueError('Invalid source angle contract')
    if not len(tri) or tri.min()<0 or tri.max()>=len(reference) or not all(np.isfinite(a).all() for a in (reference,metric,normals)):raise ValueError('Invalid source angle geometry')
    p=reference[tri];cross=np.cross(p[:,1]-p[:,0],p[:,2]-p[:,0]);length=np.linalg.norm(cross,axis=1);unit=normals/np.maximum(np.linalg.norm(normals,axis=1)[:,None],1e-30)
    cosine=np.einsum('ij,ij->i',cross,unit)/np.maximum(length,1e-30);ids=np.flatnonzero(supported&(length>0)&(cosine>=minimum))
    p=metric[tri[ids]];scale=np.linalg.norm(np.cross(p[:,1]-p[:,0],p[:,2]-p[:,0]),axis=1)
    if not np.isfinite(scale).all() or np.any(scale<=0):raise ValueError('Source angle cones require nondegenerate canonical faces')
    targets=np.maximum(minimum,np.clip(cosine[ids],-1,1)-np.minimum(1e-4,(cosine[ids]-minimum)*.5))
    if not preserve_aligned:targets=np.minimum(reserve,targets)
    return dict(face_ids=ids,normals=unit[ids],area_scale=scale,minimum_cosine=targets,reserve=float(reserve),preserve_aligned=bool(preserve_aligned))


def angular_values(points,triangles,contract):
    p=np.asarray(points)[np.asarray(triangles)[contract['face_ids']]]
    area=np.cross(p[:,1]-p[:,0],p[:,2]-p[:,0])/contract['area_scale'][:,None]
    signed=np.einsum('ij,ij->i',area,contract['normals']);length=np.linalg.norm(area,axis=1)
    return signed/np.maximum(length,1e-30),np.maximum(contract['minimum_cosine']*length-signed,0.)


def _native_uncertainty(reference,triangles,contract):
    p=np.asarray(reference)[np.asarray(triangles)[contract['face_ids']]]
    e1=p[:,1]-p[:,0];e2=p[:,2]-p[:,0];cross=np.cross(e1,e2);area=np.linalg.norm(cross,axis=1)
    unit=cross/np.maximum(area[:,None],1e-30);cosine=np.sum(unit*contract['normals'],axis=1)
    gradient=(contract['normals']-cosine[:,None]*unit)/np.maximum(area[:,None],1e-30)
    g1=np.cross(e2,gradient);g2=np.cross(gradient,e1);g=np.stack((-g1-g2,g1,g2),axis=1)
    ulp=np.abs(np.spacing(p.astype(np.float32)).astype(float))
    # First-order cosine sensitivity vanishes on aligned skin. Keep the
    # second-order normal perturbation instead of pinning every small face.
    linear=4*np.sum(np.abs(g)*ulp,axis=(1,2))
    positional=np.linalg.norm(ulp,axis=2).max(1)
    angular=4*positional*(np.linalg.norm(e1,axis=1)+np.linalg.norm(e2,axis=1))/np.maximum(area,1e-30)
    return cosine,linear+angular**2


def native_precision_reserve(reference,triangles,contract):
    """Tighten worker cones inside the unchanged original native contract.

    Skinny faces amplify native coordinate rounding. Use half of the angular
    slack already present at the starting mesh, capped by a scale-aware float32
    estimate. The original surface stays feasible; no public limit increases.
    """
    cosine,uncertainty=_native_uncertainty(reference,triangles,contract)
    estimate=np.maximum(1e-6,uncertainty)
    reserve=np.minimum(estimate,np.maximum(cosine-contract['minimum_cosine'],0.)*.5)
    inner=dict(contract,minimum_cosine=contract['minimum_cosine']+reserve)
    return inner,dict(method='INNER_ORIGINAL_SOURCE_ANGLE_NATIVE_PRECISION_RESERVE_V1',faces=len(reserve),
        reserved_faces=int((reserve>0).sum()),maximum_cosine_reserve=float(reserve.max(initial=0)),original_native_limits_unchanged=True)


def native_precision_pins(reference,triangles,contract):
    """Retain source-aligned stars whose initial angular slack is sub-ULP.

    These faces cannot safely spend their existing reserve on a float32 edit.
    Pin all their corners so the native transaction retains exact key storage.
    """
    cosine,uncertainty=_native_uncertainty(reference,triangles,contract)
    tri=np.asarray(triangles)[contract['face_ids']]
    unstable=uncertainty>np.maximum(cosine-contract['minimum_cosine'],0.)*.5
    return np.unique(tri[unstable]),dict(method='ORIGINAL_SOURCE_ANGLE_NATIVE_PRECISION_PINS_V1',faces=int(unstable.sum()),vertices=int(len(np.unique(tri[unstable]))),original_native_limits_unchanged=True)


