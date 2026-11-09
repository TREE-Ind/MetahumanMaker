"""External-worker screened biharmonic surface fairing with exact hard pins."""
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import splu


def rebuild_unobserved_shape(goal,triangles,observed,pinned,parameter_reference):
    """Screened bending of unsupported positions, with exact evidence.

    The canonical metric avoids making an inherited compressed/folded patch
    its own fairness prior. Observed and artist-pinned positions remain exact.
    This proposes missing shape; it does not certify unseen source anatomy.
    """
    import igl
    goal=np.asarray(goal,dtype=float);metric=np.asarray(parameter_reference,dtype=float)
    triangles=np.asarray(triangles,dtype=np.int64);observed=np.asarray(observed,dtype=bool);pinned=np.asarray(pinned,dtype=bool)
    n=len(goal)
    if goal.shape!=(n,3) or metric.shape!=goal.shape or observed.shape!=(n,) or pinned.shape!=(n,) or not np.isfinite(goal).all() or not np.isfinite(metric).all():raise ValueError('Invalid missing-shape arrays')
    if triangles.ndim!=2 or triangles.shape[1]!=3 or not len(triangles) or triangles.min()<0 or triangles.max()>=n:raise ValueError('Invalid missing-shape triangles')
    fixed=observed|pinned;free=~fixed;output=goal.copy()
    if not free.any():return output,dict(method='CANONICAL_METRIC_SCREENED_BENDING_V1',inferred=0,max_movement=0.,evidence_max_error=0.)
    facets=metric[triangles];area=np.linalg.norm(np.cross(facets[:,1]-facets[:,0],facets[:,2]-facets[:,0]),axis=1)*.5;mass=np.zeros(n)
    for corner in range(3):np.add.at(mass,triangles[:,corner],area/3)
    if np.any(mass[free]<=0):raise ValueError('Missing-shape vertices require nonzero canonical area')
    lap=igl.cotmatrix(metric,triangles).tocsr()
    if not np.isfinite(lap.data).all():raise ValueError('Missing-shape canonical metric contains degenerate faces')
    # Squared Laplace-Beltrami energy is positive semidefinite even when a
    # non-Delaunay canonical edge has a negative cotangent coefficient.
    inverse=np.zeros(n);inverse[mass>0]=1/mass[mass>0]
    bending=(lap.T@sparse.diags(inverse)@lap).tocsc()
    from .surface_support import observation_scale
    length=observation_scale(metric,triangles)*4
    if np.any(length[free]<=0):raise ValueError('Missing-shape vertices require nonzero canonical edge scale')
    screening=np.zeros(n);screening[free]=mass[free]/length[free]**4
    # Preserve broad unsupported volume rather than allowing a large hole
    # to collapse to the minimum-area/minimum-bending join. Four canonical
    # edges define a local, scale-independent band of shape reconstruction.
    matrix=(bending+sparse.diags(screening)).tocsc()
    A=matrix[free][:,free].tocsc()
    from scipy.sparse.csgraph import connected_components
    adjacency=lap.copy();adjacency.setdiag(0);adjacency.eliminate_zeros()
    _,labels=connected_components(adjacency,directed=False)
    if np.any(~np.isin(labels[free],np.unique(labels[fixed]))):raise ValueError('Every missing-shape component needs a supported or pinned boundary')
    output[free]=splu(A).solve(screening[free,None]*goal[free]-bending[free][:,fixed]@goal[fixed])
    if not np.isfinite(output).all():raise ValueError('Nonfinite inferred shape; previous geometry retained')
    np.testing.assert_array_equal(output[fixed],goal[fixed])
    return output,dict(method='CANONICAL_METRIC_SCREENED_BENDING_V1',scale_edges=4.,inferred=int(free.sum()),max_movement=float(np.linalg.norm(output-goal,axis=1).max()),evidence_max_error=0.,inferred_shape_is_not_source_evidence=True)


def screened_displacement(reference,goal,triangles,confidence,pinned,*,stiffness=4.,retention=.02,iterations=4):
    """Robust soft observations on a positive displacement graph.

    Positive graph conductances and diagonal observation/retention terms
    give a convex local blend of observed displacement and fixed boundaries.
    Inconsistent observations receive Huber weights based on local edges.
    This smooths deformation evidence; it does not reconstruct unseen detail.
    """
    from .surface_support import observation_scale
    reference=np.asarray(reference,dtype=float);goal=np.asarray(goal,dtype=float);triangles=np.asarray(triangles,dtype=np.int64)
    confidence=np.asarray(confidence,dtype=float);pinned=np.asarray(pinned,dtype=bool);n=len(reference)
    if reference.ndim!=2 or reference.shape[1]!=3 or goal.shape!=reference.shape or confidence.shape!=(n,) or pinned.shape!=(n,) or not np.isfinite(reference).all() or not np.isfinite(goal).all() or not np.isfinite(confidence).all() or np.any((confidence<0)|(confidence>1)):raise ValueError('Invalid displacement continuation arrays')
    if triangles.ndim!=2 or triangles.shape[1]!=3 or not len(triangles) or triangles.min()<0 or triangles.max()>=n:raise ValueError('Invalid continuation triangles')
    if not np.isfinite(stiffness) or stiffness<=0 or not np.isfinite(retention) or retention<=0 or not isinstance(iterations,int) or not 1<=iterations<=12:raise ValueError('Invalid continuation stiffness or iteration count')
    facets=reference[triangles];area=np.linalg.norm(np.cross(facets[:,1]-facets[:,0],facets[:,2]-facets[:,0]),axis=1);mass=np.zeros(n)
    for corner in range(3):np.add.at(mass,triangles[:,corner],area/6)
    positive=mass[mass>0]
    if not len(positive):raise ValueError('Displacement continuation requires nonzero surface area')
    mass=np.maximum(mass,float(np.median(positive))*1e-6)
    edges=np.unique(np.sort(np.vstack((triangles[:,[0,1]],triangles[:,[1,2]],triangles[:,[2,0]])),axis=1),axis=0);edges=edges[edges[:,0]!=edges[:,1]]
    degree=np.bincount(edges.ravel(),minlength=n);conductance=np.sqrt(mass[edges[:,0]]*mass[edges[:,1]]/np.maximum(degree[edges[:,0]]*degree[edges[:,1]],1))
    adjacency=sparse.coo_matrix((np.r_[conductance,conductance],(np.r_[edges[:,0],edges[:,1]],np.r_[edges[:,1],edges[:,0]])),shape=(n,n)).tocsr()
    lap=sparse.diags(np.asarray(adjacency.sum(1)).ravel())-adjacency
    scale=observation_scale(reference,triangles);raw=goal-reference;displacement=np.zeros_like(reference);robust=np.ones(n);free=~pinned;observed=(confidence>0)&free
    if not observed.any():raise ValueError('Displacement continuation requires movable source observations')
    for _ in range(iterations):
        weight=confidence*robust*mass;matrix=(sparse.diags(weight+retention*mass)+stiffness*lap).tocsc()
        displacement[free]=splu(matrix[free][:,free].tocsc()).solve((weight[:,None]*raw)[free])
        error=np.linalg.norm(displacement-raw,axis=1);robust=np.minimum(1.,np.maximum(scale*.5,1e-12)/np.maximum(error,1e-30))
    output=reference+displacement;output[pinned]=reference[pinned]
    if not np.isfinite(output).all():raise ValueError('Nonfinite displacement continuation')
    return output,dict(method='ROBUST_POSITIVE_SCREENED_DISPLACEMENT_V1',vertices=n,observed=int(observed.sum()),inferred=int((free&~observed).sum()),iterations=iterations,
        stiffness=stiffness,retention=retention,source_targets_are_soft=True,max_displacement=float(np.linalg.norm(displacement,axis=1).max()),
        max_anchor_displacement=float(np.linalg.norm(raw[observed],axis=1).max()),collision_review_required=True)


def harmonic_displacement(reference,goal,triangles,pinned):
    """Join registered anchors with a positive-weight displacement field.

    Inputs use one physical vertex per position. Every connected component
    must have an anchor. Free displacement is a convex combination of anchor
    displacement, avoiding the overshoot of a biharmonic seam correction.
    This does not establish anatomical correspondence or collision freedom.
    """
    from scipy.sparse.csgraph import connected_components
    reference=np.asarray(reference,dtype=float);goal=np.asarray(goal,dtype=float)
    triangles=np.asarray(triangles,dtype=np.int64);pinned=np.asarray(pinned,dtype=bool)
    if reference.ndim!=2 or reference.shape[1]!=3 or goal.shape!=reference.shape or pinned.shape!=(len(reference),) or not np.isfinite(reference).all() or not np.isfinite(goal).all():
        raise ValueError('Harmonic collar arrays must be finite and share physical vertex order')
    if triangles.ndim!=2 or triangles.shape[1]!=3 or not len(triangles) or triangles.min()<0 or triangles.max()>=len(reference):raise ValueError('Invalid collar triangles')
    edges=np.unique(np.sort(np.vstack((triangles[:,[0,1]],triangles[:,[1,2]],triangles[:,[2,0]])),axis=1),axis=0)
    edges=edges[edges[:,0]!=edges[:,1]];n=len(reference)
    adjacency=sparse.coo_matrix((np.ones(2*len(edges)),(np.r_[edges[:,0],edges[:,1]],np.r_[edges[:,1],edges[:,0]])),shape=(n,n)).tocsr()
    components,labels=connected_components(adjacency,directed=False)
    if len(np.unique(labels[pinned]))!=components:raise ValueError('Each collar component requires a fixed anchor')
    degree=np.asarray(adjacency.sum(1)).ravel();L=sparse.diags(degree)-adjacency
    displacement=np.zeros_like(reference);displacement[pinned]=goal[pinned]-reference[pinned];free=~pinned
    if free.any():displacement[free]=splu(L[free][:,free].tocsc()).solve(-L[free][:,pinned]@displacement[pinned])
    output=reference+displacement;output[pinned]=goal[pinned]
    if not np.isfinite(output).all():raise ValueError('Nonfinite harmonic collar result')
    return output,dict(method='POSITIVE_HARMONIC_DISPLACEMENT_COLLAR_V1',vertices=n,pins=int(pinned.sum()),
        pin_max_error=0.,max_displacement=float(np.linalg.norm(displacement,axis=1).max()),
        max_anchor_displacement=float(np.linalg.norm(displacement[pinned],axis=1).max()),collision_review_required=True)


def fair(base,triangles,goal,confidence,pinned,lam,reference=None,retention=None,fairness_weights=None,
         preserve_reference_differentials=False):
    base=np.asarray(base,dtype=float);goal=np.asarray(goal,dtype=float);triangles=np.asarray(triangles,dtype=np.int64)
    confidence=np.asarray(confidence,dtype=float);pinned=np.asarray(pinned,dtype=bool)
    if base.shape!=goal.shape or base.ndim!=2 or base.shape[1]!=3 or confidence.shape!=(len(base),) or pinned.shape!=confidence.shape:
        raise ValueError('Surface observations and pins must match canonical vertices')
    if not np.isfinite(goal).all() or not np.isfinite(confidence).all() or np.any(confidence<0) or not np.isfinite(lam) or lam<=0:
        raise ValueError('Choose finite surface targets, nonnegative confidence and positive fairness')
    _,weld=np.unique(np.round(base,6),axis=0,return_inverse=True);counts=np.bincount(weld);n=len(counts)
    def physical(values):return np.column_stack([np.bincount(weld,weights=values[:,i])/counts for i in range(3)])
    target=physical(goal);weight=np.bincount(weld,weights=confidence)/counts
    fixed=np.zeros(n,bool);np.logical_or.at(fixed,weld,pinned)
    edges=np.unique(np.sort(weld[np.vstack((triangles[:,[0,1]],triangles[:,[1,2]],triangles[:,[2,0]]))],axis=1),axis=0);edges=edges[edges[:,0]!=edges[:,1]]
    adjacency=sparse.coo_matrix((np.ones(len(edges)*2),(np.r_[edges[:,0],edges[:,1]],np.r_[edges[:,1],edges[:,0]])),shape=(n,n)).tocsr()
    degree=np.asarray(adjacency.sum(1)).ravel();L=sparse.eye(n)-sparse.diags(1/np.maximum(degree,1))@adjacency
    retention=(.03 if reference is not None else 1e-7) if retention is None else float(retention)
    if not np.isfinite(retention) or retention<=0:raise ValueError('Retention must be finite and positive')
    prior=target if reference is None else physical(reference)
    if preserve_reference_differentials and reference is None:raise ValueError('Differential-preserving fairing requires a registered reference')
    if fairness_weights is None:regularizer=L.T@L
    else:
        fairness_weights=np.asarray(fairness_weights,dtype=float)
        if fairness_weights.shape!=(len(base),) or not np.isfinite(fairness_weights).all() or np.any(fairness_weights<0):raise ValueError('Fairness weights must match canonical vertices')
        regional=np.bincount(weld,weights=fairness_weights)/counts
        regularizer=L.T@sparse.diags(regional)@L
    matrix=(sparse.diags(weight+retention)+lam*regularizer).tocsc();rhs=target*weight[:,None]+prior*retention
    # Smooth the displacement field, rather than flattening the character's
    # existing shape in areas with weak or absent surface observations.
    if preserve_reference_differentials:rhs+=lam*(regularizer@prior)
    free=~fixed;rhs=rhs[free]-matrix[free][:,fixed]@target[fixed]
    result=target.copy()
    if free.any():result[free]=splu(matrix[free][:,free]).solve(rhs)
    if not np.isfinite(result).all():raise ValueError('Nonfinite fairing result; previous character retained')
    output=result[weld]
    if preserve_reference_differentials:
        output=np.asarray(reference,dtype=float)+(result-prior)[weld]
        output[pinned]=goal[pinned]
    return output,dict(vertices=n,pins=int(fixed.sum()),max_movement=float(np.linalg.norm(result-target,axis=1).max()),
        preserve_reference_differentials=bool(preserve_reference_differentials),
        pin_max_error=float(np.linalg.norm(result[fixed]-target[fixed],axis=1).max(initial=0)),collision_review_required=True)
