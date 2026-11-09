"""Incremental landmark fitting from a clean, already placed physical mesh.

Work on physical vertices, then scatter to canonical UV copies in the caller.
This avoids freezing existing folds into a refinement's reference metric.
"""
import numpy as np
from . import geometry,region_projection as rp


def fit(reference,triangles,control_ids,destinations,*,pin_ids=None,pin_targets=None,
        steps=24,radius=.15,regularization=.02,cancelled=None,progress=None):
    reference=geometry.points(reference).copy();triangles=np.asarray(triangles,dtype=np.int32).reshape(-1,3)
    ids=np.asarray(control_ids,dtype=np.int32);destinations=geometry.points(destinations)
    if ids.ndim!=1 or len(ids)!=len(destinations) or not len(triangles) or triangles.min()<0 or triangles.max()>=len(reference):
        raise ValueError('Landmark flow needs valid physical triangles and matching landmark bindings')
    pins=np.asarray([] if pin_ids is None else pin_ids,dtype=np.int32)
    pin_targets=np.empty((0,3)) if pin_targets is None else geometry.points(pin_targets)
    if pins.ndim!=1 or pin_targets.shape!=(len(pins),3):raise ValueError('Flow pins and destinations must match')
    all_ids=np.concatenate((ids,pins))
    if not len(all_ids) or all_ids.min()<0 or all_ids.max()>=len(reference) or len(np.unique(all_ids))!=len(all_ids):
        raise ValueError('Bind each landmark/pin to a unique valid physical vertex')
    if not 1<=int(steps)<=256 or not np.isfinite(regularization) or regularization<0:
        raise ValueError('Invalid flow steps or regularization')
    goals=np.vstack((destinations,pin_targets));current=reference.copy()
    quality=rp.triangle_quality(reference,triangles);guard=geometry.SurfaceStepGuard(reference,triangles)
    tolerance=float(np.linalg.norm(np.ptp(reference,axis=0))*1e-6);guard.collision_tolerance=tolerance
    allowed=rp.crossings(reference,triangles,pairs=True,tolerance=tolerance);history=[]
    for step in range(int(steps)):
        if cancelled and cancelled():raise RuntimeError('Landmark initialization cancelled; previous character retained')
        remaining=int(steps)-step
        next_goals=current[all_ids]+(goals-current[all_ids])/remaining
        candidate=geometry.rbf_displacement(current,current[all_ids],next_goals,radius,regularization)
        if len(pins):candidate[pins]=reference[pins]+(pin_targets-reference[pins])*(step+1)/int(steps)
        candidate,collision,distortion=rp.limit_step(current,candidate,triangles,quality,guard,allowed)
        current=candidate
        history.append({'landmark_rms':float(np.sqrt(np.mean(np.sum((current[ids]-destinations)**2,axis=1)))),
            'collision_limited':bool(collision),'distortion_limited':bool(distortion)})
        if progress:progress((step+1)/int(steps))
    pin_error=float(np.linalg.norm(current[pins]-pin_targets,axis=1).max()) if len(pins) else 0.
    after_pairs=rp.crossings(current,triangles,pairs=True,tolerance=tolerance)
    return current,{'solver':'GUARDED_INCREMENTAL_LANDMARK_FLOW_V1','before_landmark_rms':float(np.sqrt(np.mean(np.sum((reference[ids]-destinations)**2,axis=1)))),
        'after_landmark_rms':history[-1]['landmark_rms'],'crossings_before':len(allowed),
        'crossings_after':len(after_pairs),'introduced_crossings':len(after_pairs-allowed),'pin_max_error':pin_error,
        'limited_steps':sum(r['collision_limited'] or r['distortion_limited'] for r in history),
        'steps':int(steps),'history':history,'review_required':True}
