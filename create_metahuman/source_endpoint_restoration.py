"""Repair endpoint folds against freshly queried original-source normals.

The fitting field can become stale as correspondences slide across the source.
This phase refreshes that field only between local constrained corrections.
Original source-distance, artist and anchor limits remain unchanged. Full
native orientation, coverage and contact validation is still mandatory.
"""
import time
import numpy as np
from scipy import sparse
from .surface_defects import analyze,angle_quality
from .surface_angle_restoration import restore


def repair(points,triangles,free,metric,orientation_normals,eligible,source,source_triangles,face_query,projector,*,anchors=None,time_limit=8.,passes=3,cancelled=None,minimum_cosine=.25,reserve_cosine=.5,preserve_aligned=False):
    original=np.array(points,dtype=float,copy=True);tri=np.asarray(triangles,dtype=np.int64);metric=np.asarray(metric,dtype=float)
    free=np.array(free,dtype=bool,copy=True);eligible=np.asarray(eligible,dtype=bool);n=len(original)
    if original.shape!=(n,3) or free.shape!=(n,) or eligible.shape!=(n,) or metric.shape!=original.shape:raise ValueError('Invalid source-endpoint restoration layout')
    if not 0<minimum_cosine<=reserve_cosine<1 or not np.isfinite(time_limit) or time_limit<=0:raise ValueError('Invalid source-normal restoration thresholds or time budget')
    if anchors is not None:
        anchors=np.asarray(anchors,dtype=np.int64)
        if anchors.ndim!=1 or np.any(anchors<0) or np.any(anchors>=n):raise ValueError('Invalid source-endpoint contour anchors')
        free[anchors]=False
    deadline=time.monotonic()+float(time_limit);history=[];best=original.copy()
    def field(value):
        if cancelled and cancelled():raise RuntimeError('Endpoint source-frame restoration cancelled')
        faces=np.asarray(face_query(value),dtype=np.int64)
        if faces.shape!=(n,) or np.any(faces[eligible]<0) or np.any(faces[eligible]>=len(source_triangles)):raise ValueError('Invalid endpoint source-normal witnesses')
        summary,detail=analyze(value,tri,eligible,source,source_triangles,faces,parameter_reference=metric,minimum_cosine=minimum_cosine,find_clusters=False)
        return faces,summary,detail
    faces,initial,detail=field(best);scope=detail['face_supported'].copy();p=metric[tri];reference_area=np.linalg.norm(np.cross(p[:,1]-p[:,0],p[:,2]-p[:,0]),axis=1)*.5
    def score_field(detail):
        cosine=np.where(detail['face_supported'][scope],detail['face_cosine'][scope],-1.)
        return (int(np.count_nonzero(~detail['face_supported'][scope])),int(np.count_nonzero(cosine<0)),int(np.count_nonzero(cosine<minimum_cosine)),float(np.sum(reference_area[scope]*np.maximum(minimum_cosine-cosine,0.)**2)))
    def angles(detail):return angle_quality(np.where(detail['face_supported'][scope],detail['face_cosine'][scope],-1.))
    def readable(summary):
        row=dict(summary)
        if minimum_cosine!=.25:row['below_threshold_faces']=row.pop('severe_alignment_faces')
        return row
    initial_score=score_field(detail);score=initial_score;current=best.copy()
    report=dict(method='FRESH_SOURCE_ENDPOINT_NORMAL_QUALITY_V1' if minimum_cosine!=.25 else 'FRESH_SOURCE_ENDPOINT_NORMAL_RESTORATION_V1',before=readable(initial),history=history,
        minimum_cosine=float(minimum_cosine),reserve_cosine=float(reserve_cosine),preserve_aligned_faces=bool(preserve_aligned),fixed_supported_faces=int(scope.sum()),
        before_angle_quality=angles(detail),
        source_distance_limits_unchanged=True,artist_limits_unchanged=True,requires_native_contact_check=True)
    if not initial['severe_alignment_faces']:return original,dict(report,accepted=False,reason='already_within_quality_threshold' if minimum_cosine!=.25 else 'already_source_facing',after=readable(initial),after_angle_quality=angles(detail))
    for _ in range(int(passes)):
        if time.monotonic()>=deadline:break
        ids=np.flatnonzero(scope&detail['face_supported']);vectors=np.zeros_like(current)
        sp=source[source_triangles[faces[eligible]]];sn=np.cross(sp[:,1]-sp[:,0],sp[:,2]-sp[:,0]);vectors[eligible]=sn/np.maximum(np.linalg.norm(sn,axis=1)[:,None],1e-30)
        directions=vectors[tri[ids]].sum(1);directions/=np.linalg.norm(directions,axis=1)[:,None]
        mp=metric[tri[ids]];area_scale=np.linalg.norm(np.cross(mp[:,1]-mp[:,0],mp[:,2]-mp[:,0]),axis=1)
        cosine=detail['face_cosine'][ids]
        protected=np.maximum(minimum_cosine,cosine-np.minimum(1e-4,(cosine-minimum_cosine)*.5))
        if preserve_aligned:protected=np.maximum(protected,minimum_cosine+1e-4)
        if not preserve_aligned:protected=np.minimum(reserve_cosine,protected)
        minimum=np.where(cosine<minimum_cosine,reserve_cosine,protected)
        contract=dict(face_ids=ids,normals=directions,area_scale=area_scale,minimum_cosine=minimum)
        projector.prepare(current)
        p=current[tri];area=np.linalg.norm(np.cross(p[:,1]-p[:,0],p[:,2]-p[:,0]),axis=1)
        for rings in ((1,2,3) if preserve_aligned else (1,)):
            if time.monotonic()>=deadline:break
            candidate,correction=restore(current,current,tri,free,contract,sparse.csr_matrix((0,int(free.sum())*3)),np.empty(0),
                (projector.reference-projector.limits-current)[free].ravel(),(projector.reference+projector.limits-current)[free].ravel(),
                projector,deadline,passes=6,orientation_normals=orientation_normals,minimum_area=area*.1,collar_rings=rings)
            history.append(correction)
            if correction['valid']:break
        if not history:break
        correction=history[-1]
        if not correction['valid']:break
        np.testing.assert_array_equal(candidate[~free],original[~free])
        faces,summary,detail=field(candidate);new_score=score_field(detail)
        correction['fresh_source_alignment']=readable(summary)
        if new_score<score:best=candidate.copy();score=new_score
        if new_score[:3]==(0,0,0):break
        if new_score>=score and np.array_equal(candidate,current):break
        current=candidate
    _,final,final_detail=field(best);accepted=score<initial_score
    report.update(accepted=accepted,reason='source_frame_improved' if accepted else 'no_bounded_endpoint_improvement',after=readable(final),
        after_angle_quality=angles(final_detail),remaining_bad_face_ids=np.flatnonzero(scope&((~final_detail['face_supported'])|(final_detail['face_cosine']<minimum_cosine)))[:64].tolist(),
        maximum_correction_m=float(np.linalg.norm(best-original,axis=1).max(initial=0)),pinned_coordinates_exact=True,
        elapsed_seconds=float(time_limit)-(deadline-time.monotonic()))
    return best if accepted else original,report
