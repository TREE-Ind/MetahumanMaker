"""Template-backed Poisson stitching for damaged physical mesh regions.

Inputs are physical vertices/edges, not disconnected UV copies. Callers retain
their scatter map and apply the returned displacement to canonical vertices.
The clean reference must already be registered to the desired feature pose.
"""
import numpy as np
from . import geometry


def semantic_patch(weights, edges, *, core_threshold=.65, support_threshold=.005,
                   rings=6, protected=None):
    """Select an attachment repair from rig influence and physical connectivity.

    No coordinates, character dimensions, side convention or vertex IDs are
    assumed. Call separately for each anatomical influence family. Protected
    vertices are barriers, so expansion cannot cross an artist-locked region.
    Supply welded weights/edges rather than disconnected UV seam copies.
    """
    weights=np.asarray(weights,dtype=float)
    if weights.ndim!=1 or not len(weights) or not np.isfinite(weights).all() or np.any((weights<0)|(weights>1)):
        raise ValueError('Semantic weights must be finite values between zero and one')
    if not 0<=support_threshold<core_threshold<=1 or int(rings)!=rings or rings<0:
        raise ValueError('Choose ordered influence thresholds and nonnegative whole graph rings')
    edges=np.asarray(edges,dtype=np.int32).reshape(-1,2)
    if not len(edges) or edges.min()<0 or edges.max()>=len(weights):raise ValueError('Invalid semantic repair graph')
    locked=np.zeros(len(weights),bool) if protected is None else np.asarray(protected,dtype=bool)
    if locked.shape!=weights.shape:raise ValueError('Protected mask must match physical vertices')
    core=(weights>core_threshold)&~locked
    patch=(weights>support_threshold)&~locked
    if not core.any():raise ValueError('Semantic region has no unlocked repair core')
    traversable=edges[~np.any(locked[edges],axis=1)]
    for _ in range(int(rings)):
        patch[np.unique(traversable[np.any(patch[traversable],axis=1)])]=True
    return core,patch


def select_reconstruction(reference,proposed,triangles,edges,*,rings=6,iterations=16,cancelled=None):
    """Retain reconstructed regions while reverting new endpoint contacts.

    Unlike an incremental orientation guard, this allows an already folded
    reference to unfold. New crossing pairs or extreme quality loss grow a
    rejected core; harmonic transition weights join it to retained proposal
    regions. Complete physical geometry is required. No continuous collision,
    orientation, anatomical fidelity or injectivity guarantee is made.
    """
    from . import region_projection as rp
    reference=geometry.points(reference);proposed=geometry.points(proposed)
    if proposed.shape!=reference.shape:raise ValueError('Reconstruction vertex orders must match')
    triangles=np.asarray(triangles,dtype=np.int32).reshape(-1,3);edges=np.asarray(edges,dtype=np.int32).reshape(-1,2)
    if not len(triangles) or triangles.min()<0 or triangles.max()>=len(reference) or not len(edges) or edges.min()<0 or edges.max()>=len(reference):raise ValueError('Invalid complete reconstruction graph')
    if int(rings)!=rings or rings<1 or int(iterations)!=iterations or iterations<1:raise ValueError('Choose positive whole reconstruction limits')
    tolerance=float(np.linalg.norm(np.ptp(reference,axis=0))*1e-6)
    allowed=rp.crossings(reference,triangles,pairs=True,tolerance=tolerance)
    quality=np.minimum(rp.triangle_quality(reference,triangles),rp.triangle_quality(proposed,triangles))*.02
    core=np.zeros(len(reference),bool);weight=np.ones(len(reference));result=proposed.copy();history=[]
    for step in range(int(iterations)):
        if cancelled and cancelled():raise RuntimeError('Reconstruction selection cancelled')
        new=rp.crossings(result,triangles,pairs=True,tolerance=tolerance)-allowed
        bad=np.flatnonzero(rp.triangle_quality(result,triangles)<quality)
        history.append(dict(new_pairs=len(new),distorted_faces=len(bad)))
        if not new and not len(bad):break
        faces=np.union1d(np.asarray(list(new),int).ravel(),bad)
        core[np.unique(triangles[faces])]=True
        patch=core.copy()
        for _ in range(int(rings)):patch[np.unique(edges[np.any(patch[edges],axis=1)])]=True
        weight[core]=0;weight[~patch]=1
        for _ in range(80):
            average=geometry.neighbor_average(weight[:,None],edges)[:,0]
            weight[patch&~core]=average[patch&~core]
        result=reference+(proposed-reference)*weight[:,None]
    else:
        # Never expose the last unchecked blend if the iteration budget ends.
        result=reference.copy();weight[:]=0
    after=rp.crossings(result,triangles,pairs=True,tolerance=tolerance)
    if after-allowed:raise RuntimeError('Reconstruction selection failed final crossing check')
    changed=np.linalg.norm(proposed-reference,axis=1)>1e-10
    return result,dict(method='HARMONIC_RECONSTRUCTION_SELECTION_V1',crossings_before=len(allowed),
        crossings_after=len(after),new_crossing_pairs=len(after-allowed),collision_tolerance=tolerance,
        rejected_core_vertices=int(core.sum()),retained_full_vertices=int((weight>.999).sum()),
        retained_mean_fraction=float(weight.mean()),changed_vertices=int(changed.sum()),
        changed_region_retained_mean_fraction=float(weight[changed].mean()) if changed.any() else 1.,
        history=history,production_ready=False,
        geometry_guard='ENDPOINT_NEW_CROSSING_PAIRS_AND_QUALITY_ONLY',review_required=True)


def guard_repair(reference, proposed, triangles, patch, *, minimum_quality_fraction=.35,weld=None):
    """Retain safe local repair motion instead of halving an entire character.

    Complete triangles are required, including fixed surroundings. Optional
    ``weld`` labels physical families of canonical UV copies; checks then use
    the actual canonical coordinates without averaging existing seam offsets.
    Existing proper crossings are grandfathered; new pairs are forbidden.
    This is an endpoint guard, not continuous collision detection or proof of
    anatomical fidelity. No source or Blender data is mutated.
    """
    from . import region_projection as rp
    reference=geometry.points(reference);proposed=geometry.points(proposed)
    patch=np.asarray(patch,dtype=bool)
    if proposed.shape!=reference.shape or patch.shape!=(len(reference),):raise ValueError('Repair arrays must share physical vertex order')
    if weld is not None:
        weld=np.asarray(weld)
        if weld.shape!=patch.shape:raise ValueError('Physical families must match canonical vertices')
        _,weld=np.unique(weld,return_inverse=True)
        family_patch=np.ones(int(weld.max())+1,bool);np.logical_and.at(family_patch,weld,patch)
        patch=family_patch[weld]
    triangles=np.asarray(triangles,dtype=np.int32).reshape(-1,3)
    if not len(triangles) or triangles.min()<0 or triangles.max()>=len(reference):raise ValueError('Invalid complete repair triangles')
    candidate=reference.copy();candidate[patch]=proposed[patch]
    tolerance=float(np.linalg.norm(np.ptp(reference,axis=0))*1e-6)
    allowed=rp.crossings(reference,triangles,weld,pairs=True,tolerance=tolerance)
    diagnostics={}
    result,collision_limited,distortion_limited=rp.limit_step(reference,candidate,triangles,
        rp.triangle_quality(reference,triangles),geometry.SurfaceStepGuard(reference,triangles,weld),allowed,weld=weld,
        minimum_quality_fraction=minimum_quality_fraction,collision_tolerance=tolerance,diagnostics=diagnostics)
    after=rp.crossings(result,triangles,weld,pairs=True,tolerance=tolerance)
    if after-allowed:raise RuntimeError('Repair guard failed its final crossing audit')
    if not np.array_equal(result[~patch],reference[~patch]):raise RuntimeError('Repair moved a locked vertex')
    movement=np.linalg.norm(result-reference,axis=1)
    return result,dict(solver='LOCAL_ATTACHMENT_REPAIR_GUARD_V1',accepted=bool(np.any(movement>0)),
        max_movement=float(movement.max(initial=0)),crossings_before=len(allowed),crossings_after=len(after),
        new_crossing_pairs=len(after-allowed),collision_tolerance=tolerance,
        collision_limited=collision_limited,distortion_limited=distortion_limited,
        outside_unchanged=True,diagnostics=diagnostics,review_required=True)


def finalize_mapped_patch(reference,proposed,triangles,edges,core,*,core_repair_distance,cancelled=None):
    """Resolve small chord contacts in mapped skin and backtrack its join.

    The core is a source-mapped surface; its contact correction has a strict
    displacement budget. Harmonic selection then retracts unsafe join motion.
    Any changed source points need a new fidelity check. Physical geometry and
    complete surrounding triangles are required; inherited crossings remain.
    """
    from . import region_projection as rp
    reference=geometry.points(reference);proposed=geometry.points(proposed);core=np.asarray(core,dtype=bool)
    if proposed.shape!=reference.shape or core.shape!=(len(reference),):raise ValueError('Mapped patch arrays must share vertex order')
    triangles=np.asarray(triangles,dtype=np.int32).reshape(-1,3)
    tolerance=float(np.linalg.norm(np.ptp(proposed,axis=0))*1e-6)
    before=rp.crossings(reference,triangles,pairs=True,tolerance=tolerance)
    current=rp.crossings(proposed,triangles,pairs=True,tolerance=tolerance)
    core_pairs={pair for pair in current-before if np.all(core[triangles[list(pair)]])}
    corrected=proposed.copy();correction=dict(accepted=True,new_crossings_before=0,new_crossings_after=0,max_movement=0.)
    if core_pairs:
        corrected,correction=rp.separate_contacts(proposed,triangles,current-core_pairs,movable=core,
            max_distance=core_repair_distance,iterations=6,cancelled=cancelled)
        if not correction['accepted']:
            return reference.copy(),dict(accepted=False,reason='Mapped core contacts exceed bounded correction',core_correction=correction,review_required=True)
    collar=(~core)&(np.linalg.norm(proposed-reference,axis=1)>1e-10)
    result,join_correction=rp.separate_contacts(corrected,triangles,before,movable=collar,
        max_distance=core_repair_distance*5,iterations=12,cancelled=cancelled)
    if join_correction['accepted']:
        return result,dict(accepted=True,core_correction=correction,join_correction=join_correction,
            source_core_preserved_during_join=True,production_ready=False,review_required=True)
    result,selection=select_reconstruction(reference,corrected,triangles,edges,cancelled=cancelled)
    motion=np.linalg.norm(corrected-reference,axis=1)
    retained=np.linalg.norm(result-reference,axis=1)
    meaningful=bool(np.any(retained>1e-8) or not np.any(motion>1e-8))
    return result,dict(accepted=meaningful,reason=None if meaningful else 'Join selection rejected the entire mapped proposal',
        core_correction=correction,join_correction=join_correction,join_selection=selection,production_ready=False,review_required=True)


def repair_semantic(reference,clean,edges,triangles,weights,*,protected=None,
                    core_threshold=.65,support_threshold=.005,rings=6,
                    minimum_quality_fraction=.35,**stitch_options):
    """Reconstruct a rig-labelled attachment and limit unsafe local motion.

    ``clean`` must already be registered using trustworthy source evidence.
    This API deliberately does not invent that registration from rig weights.
    Selection is character independent; outside and protected vertices stay
    fixed. Core anchors may be relaxed by the safety guard, reported as RMS.
    """
    core,patch=semantic_patch(weights,edges,protected=protected,
        core_threshold=core_threshold,support_threshold=support_threshold,rings=rings)
    proposal,reconstruction=stitch(reference,clean,edges,core,patch,**stitch_options)
    result,report=guard_repair(reference,proposal,triangles,patch,
        minimum_quality_fraction=minimum_quality_fraction)
    target=clean if stitch_options.get('anchors') is None else stitch_options['anchors']
    report.update(selection='RIG_INFLUENCE_AND_CONNECTIVITY',core_vertices=int(core.sum()),
        patch_vertices=int(patch.sum()),reconstruction=reconstruction,
        core_anchor_rms=float(np.sqrt(np.mean((result[core]-np.asarray(target)[core])**2))))
    return result,report


def stitch(reference,clean,edges,core,patch,*,iterations=600,cancelled=None,anchors=None,
           targets=None,confidence=None,screening=1.):
    reference=geometry.points(reference);clean=geometry.points(clean)
    if reference.shape!=clean.shape:raise ValueError('Clean and current region vertex orders must match')
    anchors=clean if anchors is None else geometry.points(anchors)
    if anchors.shape!=clean.shape:raise ValueError('Repair anchors must match the physical vertex order')
    if (targets is None)!=(confidence is None):raise ValueError('Surface targets and confidence must be provided together')
    if not np.isfinite(screening) or screening<0:raise ValueError('Surface screening must be finite and nonnegative')
    if targets is not None:
        targets=geometry.points(targets);confidence=np.asarray(confidence,dtype=float)
        if targets.shape!=clean.shape or confidence.shape!=(len(reference),) or not np.isfinite(confidence).all() or np.any(confidence<0):
            raise ValueError('Surface constraints must be finite, nonnegative and match physical vertices')
    core=np.asarray(core,dtype=bool);patch=np.asarray(patch,dtype=bool)
    if core.shape!=(len(reference),) or patch.shape!=core.shape or not core.any():
        raise ValueError('Choose a nonempty repair core and matching patch masks')
    if np.any(core&~patch):raise ValueError('Repair core must be inside the stitch patch')
    edges=np.asarray(edges,dtype=np.int32).reshape(-1,2)
    if not len(edges) or edges.min()<0 or edges.max()>=len(reference):raise ValueError('Invalid repair graph')
    # Only solve the selected neighborhood and its fixed one-ring boundary.
    local_edges=edges[np.any(patch[edges],axis=1)]
    ids,mapping=np.unique(local_edges,return_inverse=True);local_edges=mapping.reshape(-1,2)
    active=patch[ids];fixed_core=core[ids];free=active&~fixed_core
    original=reference[ids];goal=clean[ids];current=original.copy()
    differential=goal-geometry.neighbor_average(goal,local_edges)
    mass=np.zeros(len(ids)) if confidence is None else confidence[ids]*screening
    degree=np.bincount(local_edges.ravel(),minlength=len(ids))
    current[fixed_core]=anchors[ids][fixed_core]
    for step in range(max(1,int(iterations))):
        if cancelled and cancelled():raise RuntimeError('Region reconstruction cancelled; previous surface retained')
        proposal=geometry.neighbor_average(current,local_edges)+differential
        if targets is not None:
            proposal=(proposal*degree[:,None]+targets[ids]*mass[:,None])/np.maximum(degree+mass,1e-20)[:,None]
        current[free]=.8*proposal[free]+.2*current[free]
    if not np.isfinite(current).all():raise ValueError('Region reconstruction produced nonfinite coordinates')
    result=reference.copy();result[ids[active]]=current[active]
    residual=current-geometry.neighbor_average(current,local_edges)-differential
    if targets is not None:residual=(residual*degree[:,None]+mass[:,None]*(current-targets[ids]))/np.maximum(degree+mass,1e-20)[:,None]
    report={'solver':'CANONICAL_POISSON_STITCH_V1' if targets is None else 'SCREENED_CANONICAL_POISSON_STITCH_V1','iterations':max(1,int(iterations)),
            'solver_vertices':len(ids),'core_vertices':int(fixed_core.sum()),
            'free_vertices':int(free.sum()),'residual_rms':float(np.sqrt(np.mean(residual[free]**2))) if free.any() else 0.,
            'screened_surface_vertices':int(np.count_nonzero(mass[free])),
            'outside_unchanged':bool(np.array_equal(result[~patch],reference[~patch]))}
    return result,report


def guarded_stitch(reference,clean,edges,core,patch,triangles,*,minimum_fraction=.01,**kwargs):
    """Backtrack a complete repair field without introducing triangle crossings.

    Use the complete physical mesh triangles, including fixed surroundings.
    Partial steps intentionally relax core anchors; the report exposes that
    residual instead of claiming an exact reconstruction after backtracking.
    This function does not mutate Blender objects or establish source fidelity.
    """
    from . import region_projection as rp
    if not 0<minimum_fraction<=1:raise ValueError('Minimum repair fraction must be between zero and one')
    reference=geometry.points(reference)
    triangles=np.asarray(triangles,dtype=np.int32).reshape(-1,3)
    if not len(triangles) or triangles.min()<0 or triangles.max()>=len(reference):
        raise ValueError('Guarded reconstruction needs complete valid physical triangles')
    proposed,report=stitch(reference,clean,edges,core,patch,**kwargs)
    tolerance=float(np.linalg.norm(np.ptp(reference,axis=0))*1e-6)
    allowed=rp.crossings(reference,triangles,pairs=True,tolerance=tolerance)
    quality=rp.triangle_quality(reference,triangles)
    old=reference[triangles];old_normal=np.cross(old[:,1]-old[:,0],old[:,2]-old[:,0])
    length=np.linalg.norm(old_normal,axis=1);fraction=1.;result=reference.copy()
    for _ in range(13):
        if fraction<minimum_fraction:
            fraction=0.;break
        if kwargs.get('cancelled') and kwargs['cancelled']():
            raise RuntimeError('Region reconstruction cancelled; previous surface retained')
        candidate=reference+(proposed-reference)*fraction
        p=candidate[triangles];normal=np.cross(p[:,1]-p[:,0],p[:,2]-p[:,0])
        dot=np.einsum('ij,ij->i',old_normal,normal)
        valid=(rp.triangle_quality(candidate,triangles)>=quality*.35)&(dot>=length*np.linalg.norm(normal,axis=1)*.1)
        if valid.all() and not (rp.crossings(candidate,triangles,pairs=True,tolerance=tolerance)-allowed):
            result=candidate;break
        fraction*=.5
    else:fraction=0.
    target=clean if kwargs.get('anchors') is None else kwargs['anchors']
    report.update(solver='GUARDED_CANONICAL_POISSON_STITCH_V1',accepted=fraction>0,accepted_fraction=fraction,
        crossings_before=len(allowed),crossings_after=rp.crossings(result,triangles,tolerance=tolerance),
        core_anchor_rms=float(np.sqrt(np.mean((result[core]-target[core])**2))),
        max_movement=float(np.linalg.norm(result-reference,axis=1).max()),review_required=True)
    return result,report


def reconstruct_surface(reference,clean,edges,core,patch,triangles,*,targets,confidence,
                        screenings=(0.,.03,.1,.3),iterations=800,anchors=None,cancelled=None):
    """Choose a screened repair using surface support and measured crossings.

    This is an untangling repair, unlike conservative guarded_stitch: crossing
    pairs may change, but their total cannot increase. Even a successful repair
    still requires visual review. Supply complete physical mesh triangles.
    """
    from . import region_projection as rp
    reference=geometry.points(reference);triangles=np.asarray(triangles,dtype=int).reshape(-1,3)
    targets=geometry.points(targets);confidence=np.asarray(confidence,dtype=float)
    patch=np.asarray(patch,dtype=bool)
    if targets.shape!=reference.shape or confidence.shape!=(len(reference),) or not np.isfinite(confidence).all() or np.any(confidence<0):
        raise ValueError('Surface constraints must match physical mesh vertices')
    if not len(triangles) or triangles.min()<0 or triangles.max()>=len(reference):raise ValueError('Invalid reconstruction triangles')
    if not len(screenings):raise ValueError('Choose at least one surface screening strength')
    measured=confidence*patch
    if not measured.any():raise ValueError('Surface reconstruction needs supported patch observations')
    def error(p):return float(np.sqrt(np.average(np.sum((p-targets)**2,axis=1),weights=measured)))
    tolerance=float(np.linalg.norm(np.ptp(reference,axis=0))*1e-6)
    before=rp.crossings(reference,triangles,tolerance=tolerance);before_error=error(reference);quality=rp.triangle_quality(reference,triangles)
    result=reference.copy();best=None;trials=[]
    for strength in screenings:
        if cancelled and cancelled():raise RuntimeError('Region reconstruction cancelled; previous surface retained')
        candidate,solver=stitch(reference,clean,edges,core,patch,iterations=iterations,anchors=anchors,
            targets=targets,confidence=confidence,screening=strength,cancelled=cancelled)
        crossings=rp.crossings(candidate,triangles,tolerance=tolerance);fit_error=error(candidate)
        valid=bool(crossings<=before and (crossings<before or fit_error<=before_error) and
                   np.all(rp.triangle_quality(candidate,triangles)>=quality*.02))
        trials.append({'screening':float(strength),'crossings':crossings,'surface_target_rms':fit_error,'valid':valid})
        if valid and (best is None or fit_error<best['surface_target_rms']):
            result=candidate;best=dict(trials[-1],solver=solver)
    return result,{'solver':'SELECTED_SCREENED_SURFACE_RECONSTRUCTION_V1','accepted':best is not None,
        'before':{'crossings':before,'surface_target_rms':before_error},'selected':best,
        'surface_target_improved':bool(best and best['surface_target_rms']<=before_error),
        'trials':trials,'review_required':True,'outside_unchanged':bool(np.array_equal(result[~patch],reference[~patch]))}
