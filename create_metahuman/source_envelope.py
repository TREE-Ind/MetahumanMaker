"""Conservative movement budgets derived from an immutable source surface."""
import numpy as np


def unobserved_motion_envelope(reference,metric,edges,free,observed,influence,limits):
    """One fixed, boundary-derived motion policy for genuinely missing shape.

    Original source-supported boxes remain exact. Only missing vertices gain
    uncertainty with geodesic distance from source-supported/fixed boundaries.
    The envelope uses the original snapshot, never a generated shape goal or
    an intermediate solver endpoint. Artist influence fades its extent.
    """
    from scipy import sparse
    from scipy.sparse.csgraph import dijkstra
    reference=np.asarray(reference,dtype=float);metric=np.asarray(metric,dtype=float);edges=np.asarray(edges,dtype=np.int64)
    free=np.asarray(free,dtype=bool);observed=np.asarray(observed,dtype=bool);influence=np.asarray(influence,dtype=float);limits=np.asarray(limits,dtype=float)
    n=len(reference)
    if reference.shape!=(n,3) or metric.shape!=reference.shape or limits.shape!=reference.shape or free.shape!=(n,) or observed.shape!=(n,) or influence.shape!=(n,) or not all(np.isfinite(a).all() for a in (reference,metric,limits,influence)) or np.any(limits<0) or np.any((influence<0)|(influence>1)) or np.any(observed&~free):raise ValueError('Invalid missing-shape envelope')
    if edges.ndim!=2 or edges.shape[1]!=2 or not len(edges) or edges.min()<0 or edges.max()>=n:raise ValueError('Invalid missing-shape graph')
    unknown=free&~observed;output=limits.copy()
    canonical=np.linalg.norm(metric[edges[:,0]]-metric[edges[:,1]],axis=1);actual=np.linalg.norm(reference[edges[:,0]]-reference[edges[:,1]],axis=1)
    valid=(canonical>0)&(actual>0);supported=valid&np.all(observed[edges],axis=1)
    scale=float(np.median(actual[supported]/canonical[supported])) if supported.any() else 1.
    anchors=np.flatnonzero(observed|~free);seed=np.where(free[anchors],limits[anchors].max(1),0.)
    rows=np.r_[edges[valid,0],edges[valid,1],np.full(len(anchors),n)];columns=np.r_[edges[valid,1],edges[valid,0],anchors]
    costs=np.r_[canonical[valid]*scale,canonical[valid]*scale,seed]
    graph=sparse.coo_matrix((costs,(rows,columns)),shape=(n+1,n+1)).tocsr()
    distance=dijkstra(graph,directed=True,indices=n)[:n]
    if not np.isfinite(distance[unknown]).all():raise ValueError('Each missing-shape component needs a source-supported or fixed boundary')
    output[unknown]=np.maximum(output[unknown],(distance[unknown]*influence[unknown])[:,None])
    if not np.array_equal(output[~unknown],limits[~unknown]):raise AssertionError('Missing-shape envelope changed supported or fixed budgets')
    return output,dict(method='FIXED_GEODESIC_UNOBSERVED_MOTION_ENVELOPE_V1',metric_scale=scale,missing_vertices=int(unknown.sum()),
        expanded_missing_vertices=int(np.any(output[unknown]>limits[unknown],axis=1).sum()),source_supported_limits_unchanged=True,fixed_limits_unchanged=True,
        before_max_missing_limit_m=float(limits[unknown].max(initial=0)),after_max_missing_limit_m=float(output[unknown].max(initial=0)),derived_from_inferred_goal=False,budgets_recentered=False)


def source_reachable_boxes(reference,vertex_ids,surface_points,free,edges,*,baseline,margin=.002,taper_rings=4):
    """Fix artist boxes around original centers that can reach source witnesses.

    This defines an explicit new policy, not an iterative budget increase.
    The caller must certify the supplied points against the original source.
    Reachability alone does not prove a collision-free path or correspondence.
    """
    reference=np.asarray(reference,dtype=float);ids=np.asarray(vertex_ids);points=np.asarray(surface_points,dtype=float);free=np.asarray(free,dtype=bool);edges=np.asarray(edges)
    if reference.ndim!=2 or reference.shape[1]!=3 or not np.isfinite(reference).all():raise ValueError('Invalid artist centers')
    n=len(reference)
    if ids.ndim!=1 or ids.dtype.kind not in 'iu' or np.any(ids<0) or np.any(ids>=n) or len(np.unique(ids))!=len(ids):raise ValueError('Invalid source witness vertex IDs')
    if points.shape!=(len(ids),3) or not np.isfinite(points).all() or free.shape!=(n,) or not np.all(free[ids]):raise ValueError('Source reachability requires observed movable vertices')
    if edges.ndim!=2 or edges.shape[1]!=2 or edges.dtype.kind not in 'iu' or np.any(edges<0) or np.any(edges>=n):raise ValueError('Invalid artist mesh edges')
    if not np.isfinite(baseline) or baseline<=0 or not np.isfinite(margin) or margin<0 or not isinstance(taper_rings,int) or not 0<=taper_rings<=32:raise ValueError('Invalid reachability policy')
    radii=np.full(n,float(baseline));radii[ids]=np.maximum(baseline,np.max(np.abs(points-reference[ids]),axis=1)+margin)
    for _ in range(taper_rings):
        grown=radii.copy();np.maximum.at(grown,edges[:,0],radii[edges[:,1]]-baseline*.25);np.maximum.at(grown,edges[:,1],radii[edges[:,0]]-baseline*.25)
        radii[free]=grown[free]
    return np.repeat(radii[:,None],3,axis=1)


def source_offset_radii(distances,*,maximum_offset,maximum_motion,safety=.95):
    """Bound source distance after any movement inside the returned balls.

    Distance to a fixed surface is 1-Lipschitz: d(new,S) <= d(old,S)
    + ||new-old||. This needs no planar/normal approximation and remains
    valid across curved facets and patch boundaries. Infinite input distance
    means unsupported and receives zero motion. A point already outside the
    envelope is pinned, not silently certified as repaired.
    """
    distances=np.asarray(distances,dtype=float)
    if np.isnan(distances).any() or np.any(distances<0):raise ValueError('Source distances must be nonnegative or positive infinity')
    if not np.isfinite(maximum_offset) or maximum_offset<=0 or not np.isfinite(safety) or not 0<safety<=1:raise ValueError('Choose a positive finite source offset and safety between zero and one')
    motion=np.broadcast_to(np.asarray(maximum_motion,dtype=float),distances.shape)
    if not np.isfinite(motion).all() or np.any(motion<0):raise ValueError('Maximum motion must be finite and nonnegative')
    return np.minimum(motion,np.maximum(maximum_offset-distances,0.)*safety)


def project_ball_intersection(points,center_a,radius_a,center_b,radius_b):
    """Closest points inside both artist and source-witness balls.

    A source witness must lie on the immutable source. Its ball certifies
    source offset, while the other ball bounds total artist displacement.
    Neither center is silently reset. Empty intersections are rejected.
    """
    points=np.asarray(points,dtype=float);a=np.asarray(center_a,dtype=float);b=np.asarray(center_b,dtype=float)
    if points.ndim!=2 or points.shape[1]!=3 or a.shape!=points.shape or b.shape!=points.shape or not all(np.isfinite(v).all() for v in (points,a,b)):raise ValueError('Ball projection needs matching finite N by 3 coordinates')
    ra=np.broadcast_to(np.asarray(radius_a,dtype=float),(len(points),));rb=np.broadcast_to(np.asarray(radius_b,dtype=float),(len(points),))
    if not np.isfinite(ra).all() or not np.isfinite(rb).all() or np.any(ra<0) or np.any(rb<0):raise ValueError('Ball radii must be finite and nonnegative')
    axis=b-a;distance=np.linalg.norm(axis,axis=1);tolerance=32*np.finfo(float).eps*np.maximum(1.,np.maximum(abs(a).max(1),abs(b).max(1)))
    if np.any(distance>ra+rb+tolerance):raise ValueError('Artist and source-witness balls have no feasible intersection')
    da=points-a;db=points-b;la=np.linalg.norm(da,axis=1);lb=np.linalg.norm(db,axis=1)
    pa=a+da*np.minimum(1.,ra/np.maximum(la,1e-300))[:,None];pb=b+db*np.minimum(1.,rb/np.maximum(lb,1e-300))[:,None]
    good_a=np.linalg.norm(pa-b,axis=1)<=rb+tolerance;good_b=np.linalg.norm(pb-a,axis=1)<=ra+tolerance
    choose_a=good_a&(~good_b|(np.sum((pa-points)**2,axis=1)<=np.sum((pb-points)**2,axis=1)))
    output=np.where(choose_a[:,None],pa,pb);circle=~good_a&~good_b
    if circle.any():
        ids=np.flatnonzero(circle);unit=axis[ids]/distance[ids,None]
        height=(ra[ids]**2-rb[ids]**2+distance[ids]**2)/(2*distance[ids]);center=a[ids]+unit*height[:,None]
        radius=np.sqrt(np.maximum(ra[ids]**2-height**2,0.));direction=points[ids]-center;direction-=unit*np.einsum('ni,ni->n',direction,unit)[:,None]
        length=np.linalg.norm(direction,axis=1);flat=length<1e-14
        if flat.any():
            basis=np.eye(3)[np.argmin(abs(unit[flat]),axis=1)];direction[flat]=np.cross(unit[flat],basis);length[flat]=np.linalg.norm(direction[flat],axis=1)
        output[ids]=center+direction/length[:,None]*radius[:,None]
    if np.any(np.linalg.norm(output-a,axis=1)>ra+tolerance*4) or np.any(np.linalg.norm(output-b,axis=1)>rb+tolerance*4):raise ValueError('Ball projection lost feasibility')
    return output


class SourceWitnessBalls:
    """External-worker constraint with an injected exact source query.

    The query returns original-facet witnesses for the selected physical
    vertices. Fresh source centers permit tangential motion while the artist
    centers remain fixed throughout the operation.
    """
    def __init__(self,reference,radii,vertex_ids,query,*,source_radius,query_contract):
        from .vertex_contract import fingerprint
        vertex_ids=np.asarray(vertex_ids)
        if vertex_ids.dtype.kind not in 'iu':raise ValueError('Source-witness vertex IDs must be integer indices')
        self.reference=np.array(reference,dtype=float,copy=True);self.radii=np.asarray(radii,dtype=float).copy();self.ids=vertex_ids.astype(np.int64,copy=True);self.query=query
        if self.reference.ndim!=2 or self.reference.shape[1]!=3 or not np.isfinite(self.reference).all() or self.radii.shape!=(len(self.reference),) or not np.isfinite(self.radii).all() or np.any(self.radii<0) or self.ids.ndim!=1 or len(np.unique(self.ids))!=len(self.ids) or (len(self.ids) and (self.ids.min()<0 or self.ids.max()>=len(self.reference))):raise ValueError('Invalid source-witness vertex layout')
        radius=np.asarray(source_radius,dtype=float)
        if radius.ndim>1 or (radius.ndim==1 and radius.shape!=(len(self.ids),)) or not np.isfinite(radius).all() or np.any(radius<=0):raise ValueError('Choose finite positive source-witness radii')
        self.source_radius=float(radius) if radius.ndim==0 else radius.copy();self.witness=None;self.projections=0
        encoded_radius=self.source_radius if radius.ndim==0 else fingerprint(self.source_radius)
        self.contract=dict(method='EXACT_SOURCE_WITNESS_AND_ARTIST_BALLS_V1',reference=fingerprint(self.reference[self.ids]),radii=fingerprint(self.radii[self.ids]),vertex_ids=fingerprint(self.ids),source_radius=encoded_radius,query=query_contract)

    def prepare(self,positions):
        record=self.query(np.asarray(positions)[self.ids])
        centers=np.asarray(record['surface_points'],dtype=float)
        if centers.shape!=(len(self.ids),3) or not np.isfinite(centers).all() or not np.asarray(record['observed']).all():raise ValueError('Source query did not produce complete finite witnesses')
        if np.any(np.linalg.norm(positions[self.ids]-centers,axis=1)>self.source_radius+1e-10):raise ValueError('Starting patch lies outside its source-witness envelope')
        self.witness=record;self.centers=centers.copy()

    def __call__(self,positions):
        if self.witness is None:raise ValueError('Prepare source witnesses before projection')
        output=np.array(positions,dtype=float,copy=True)
        output[self.ids]=project_ball_intersection(output[self.ids],self.reference[self.ids],self.radii[self.ids],self.centers,self.source_radius)
        self.projections+=1
        return output

    def source_distance(self,positions):
        if self.witness is None:raise ValueError('Prepare source witnesses before measuring distance')
        return np.linalg.norm(np.asarray(positions)[self.ids]-self.centers,axis=1)


def project_box_ball(points,lower,upper,centers,radii):
    """Euclidean projection onto an axis box intersected with a source ball.

    The active box constraints are handled by clipping. A monotone scalar
    Lagrange multiplier then enforces the ball without alternating drift.
    The feasible side of the root bracket is always returned.
    """
    points=np.asarray(points,dtype=float);centers=np.asarray(centers,dtype=float)
    lower=np.broadcast_to(np.asarray(lower,dtype=float),points.shape);upper=np.broadcast_to(np.asarray(upper,dtype=float),points.shape)
    radii=np.broadcast_to(np.asarray(radii,dtype=float),(len(points),))
    if points.ndim!=2 or points.shape[1]!=3 or centers.shape!=points.shape or not all(np.isfinite(v).all() for v in (points,centers,lower,upper,radii)) or np.any(lower>upper) or np.any(radii<0):raise ValueError('Invalid box/source-ball intersection')
    nearest=np.clip(centers,lower,upper);minimum=np.linalg.norm(nearest-centers,axis=1)
    tolerance=64*np.finfo(float).eps*np.maximum(1.,np.max(abs(centers),axis=1))
    if np.any(minimum>radii+tolerance):raise ValueError('Artist box and source ball have no feasible intersection')
    output=np.clip(points,lower,upper);outside=np.linalg.norm(output-centers,axis=1)>radii
    tangent=outside&(radii-minimum<=tolerance);output[tangent]=nearest[tangent]
    ids=np.flatnonzero(outside&~tangent)
    if len(ids):
        p=points[ids];c=centers[ids];lo=lower[ids];hi=upper[ids];r=radii[ids]
        def at(value):return np.clip(c+(p-c)/(1+value[:,None]),lo,hi)
        left=np.zeros(len(ids));right=np.ones(len(ids))
        for _ in range(64):
            bad=np.linalg.norm(at(right)-c,axis=1)>r
            if not bad.any():break
            right[bad]*=2
        if np.any(np.linalg.norm(at(right)-c,axis=1)>r+tolerance[ids]):raise ValueError('Could not bracket the box/source-ball projection')
        for _ in range(64):
            mid=(left+right)*.5;bad=np.linalg.norm(at(mid)-c,axis=1)>r
            left=np.where(bad,mid,left);right=np.where(bad,right,mid)
        output[ids]=at(right)
    if np.any(np.linalg.norm(output-centers,axis=1)>radii+tolerance*4):raise ValueError('Box/source-ball projection lost feasibility')
    return output


def facet_box_anchor(points,lower,upper,facets,radii):
    """Find certified anchors in boxes intersecting triangle offsets.

    A ball about the nearest surface point is a useful fast certificate, but
    its failure does not imply that the entire triangle offset misses the
    box. Dykstra projections handle that case using the two convex sets.
    Only an independently measured feasible point is returned; convergence
    failure is not a claim that the geometric intersection is empty.
    """
    from .vertex_contract import closest_triangle_points
    points=np.asarray(points,dtype=float);facets=np.asarray(facets,dtype=float)
    lower=np.broadcast_to(np.asarray(lower,dtype=float),points.shape);upper=np.broadcast_to(np.asarray(upper,dtype=float),points.shape)
    radii=np.broadcast_to(np.asarray(radii,dtype=float),(len(points),))
    if points.ndim!=2 or points.shape[1]!=3 or facets.shape!=(len(points),3,3) or not all(np.isfinite(v).all() for v in (points,facets,lower,upper,radii)) or np.any(lower>upper) or np.any(radii<0):raise ValueError('Invalid box/source-facet intersection')
    centers,_,_=closest_triangle_points(points,facets)
    tolerance=128*np.finfo(float).eps*np.maximum(1.,np.max(abs(facets),axis=(1,2)))
    fast=np.linalg.norm(np.clip(centers,lower,upper)-centers,axis=1)<=radii
    output=np.empty_like(points)
    if fast.any():output[fast]=project_box_ball(points[fast],lower[fast],upper[fast],centers[fast],radii[fast])
    ids=np.flatnonzero(~fast)
    if len(ids):
        x=points[ids].copy();box_dual=np.zeros_like(x);tube_dual=np.zeros_like(x)
        lo=lower[ids];hi=upper[ids];f=facets[ids];r=radii[ids]
        for _ in range(256):
            before=x+box_dual;box=np.clip(before,lo,hi);box_dual=before-box
            before=box+tube_dual;surface,_,distance=closest_triangle_points(before,f)
            x=surface+(before-surface)*np.minimum(1.,r/np.maximum(distance,1e-300))[:,None]
            tube_dual=before-x
            candidate=np.clip(x,lo,hi)
            feasible=closest_triangle_points(candidate,f)[2]<=r+tolerance[ids]
            if feasible.all():break
        if not feasible.all():raise ValueError('Source facet/artist box anchor did not converge; intersection not certified')
        output[ids]=candidate
    if np.any(closest_triangle_points(output,facets)[2]>radii+tolerance*4):raise ValueError('Source facet/artist box anchor lost feasibility')
    return output


class SourceWitnessBoxes(SourceWitnessBalls):
    """Original artist boxes plus fixed per-vertex source-offset budgets."""
    def __init__(self,reference,limits,vertex_ids,query,*,source_radius,query_contract):
        from .vertex_contract import fingerprint
        limits=np.broadcast_to(np.asarray(limits,dtype=float),np.asarray(reference).shape).copy()
        if not np.isfinite(limits).all() or np.any(limits<0):raise ValueError('Invalid original artist box')
        super().__init__(reference,np.linalg.norm(limits,axis=1),vertex_ids,query,source_radius=source_radius,query_contract=query_contract)
        self.limits=limits;self.contract.update(method='EXACT_SOURCE_WITNESS_AND_ARTIST_BOXES_V1',limits=fingerprint(limits[self.ids]))

    def __call__(self,positions):
        if self.witness is None:raise ValueError('Prepare source witnesses before projection')
        output=np.array(positions,dtype=float,copy=True)
        output[self.ids]=project_box_ball(output[self.ids],self.reference[self.ids]-self.limits[self.ids],self.reference[self.ids]+self.limits[self.ids],self.centers,self.source_radius)
        self.projections+=1
        return output


class SourceFacetBoxes(SourceWitnessBoxes):
    """Artist boxes intersected with offsets of original source triangles.

    A whole triangle, rather than one frozen point on it, certifies each
    local convex constraint. This permits tangential redistribution without
    increasing the original source-distance allowance. Facets refresh only
    between outer iterations, never during a line search.
    """
    def __init__(self,reference,limits,vertex_ids,query,*,source_radius,query_contract,source_vertices,source_triangles):
        from .vertex_contract import fingerprint
        super().__init__(reference,limits,vertex_ids,query,source_radius=source_radius,query_contract=query_contract)
        self.source_vertices=np.array(source_vertices,dtype=float,copy=True);self.source_triangles=np.array(source_triangles,dtype=np.int64,copy=True)
        expected=dict(vertices=fingerprint(self.source_vertices),triangles=fingerprint(self.source_triangles))
        if any(query_contract.get(key)!=value for key,value in expected.items()):raise ValueError('Source facets do not match the immutable query geometry')
        self.source_vertices.flags.writeable=False;self.source_triangles.flags.writeable=False
        self.contract['method']='EXACT_SOURCE_FACET_AND_ARTIST_BOXES_V1'

    def prepare(self,positions):
        from .vertex_contract import audit_witnesses
        super().prepare(positions)
        audit_witnesses(self.witness,self.source_vertices,self.source_triangles)
        self.facets=self.source_vertices[self.source_triangles[self.witness['source_triangle_ids']]]
        self.anchor=facet_box_anchor(np.asarray(positions)[self.ids],self.reference[self.ids]-self.limits[self.ids],
            self.reference[self.ids]+self.limits[self.ids],self.facets,self.source_radius)

    def source_distance(self,positions):
        from .vertex_contract import closest_triangle_points
        if self.witness is None:raise ValueError('Prepare original source facets before measuring distance')
        return closest_triangle_points(np.asarray(positions)[self.ids],self.facets)[2]

    def retract(self,positions):
        """Project an outside proposal into the original source/artist envelope.

        Unlike prepare(), the input need not already satisfy source distance.
        This is an independent endpoint query, not a facet refresh inside an
        existing convex subproblem. Original centers and budgets stay fixed.
        """
        from .vertex_contract import audit_witnesses
        positions=np.asarray(positions,dtype=float)
        if positions.shape!=self.reference.shape or not np.isfinite(positions).all():raise ValueError('Invalid source retraction proposal')
        record=self.query(positions[self.ids]);audit_witnesses(record,self.source_vertices,self.source_triangles)
        if not np.asarray(record['observed']).all():raise ValueError('Source retraction requires complete original-source witnesses')
        centers=np.asarray(record['surface_points'],dtype=float)
        facets=self.source_vertices[self.source_triangles[record['source_triangle_ids']]]
        anchor=facet_box_anchor(positions[self.ids],self.reference[self.ids]-self.limits[self.ids],self.reference[self.ids]+self.limits[self.ids],facets,self.source_radius)
        self.witness=record;self.centers=centers.copy();self.facets=facets;self.anchor=anchor
        return self(positions)

    def __call__(self,positions):
        from .vertex_contract import closest_triangle_points
        if self.witness is None:raise ValueError('Prepare original source facets before projection')
        output=np.array(positions,dtype=float,copy=True)
        lower=self.reference[self.ids]-self.limits[self.ids];upper=self.reference[self.ids]+self.limits[self.ids]
        points=np.clip(output[self.ids],lower,upper)
        radius=np.broadcast_to(self.source_radius,(len(self.ids),));outside=closest_triangle_points(points,self.facets)[2]>radius
        # Near a triangle's offset plane, an anchor-to-point segment can lie
        # entirely on the radius boundary. Floating distance noise then makes
        # bisection choose an arbitrary tangential position, even when the
        # required correction is only a few ulps. Project along the nearest
        # surface direction first, retaining tangential correspondence.
        # Box clipping can require another pass; the established feasible
        # anchor remains the fallback for genuinely difficult intersections.
        magnitude=np.maximum(1.,np.maximum(np.max(np.abs(points),axis=1),np.max(np.abs(self.facets),axis=(1,2))))
        roundoff=128*np.finfo(float).eps*magnitude
        for _ in range(4):
            if not outside.any():break
            selected=np.flatnonzero(outside);surface,_,distance=closest_triangle_points(points[selected],self.facets[selected])
            projected=surface+(points[selected]-surface)*(radius[selected]/np.maximum(distance,1e-300))[:,None]
            points[selected]=np.clip(projected,lower[selected],upper[selected])
            outside=closest_triangle_points(points,self.facets)[2]>radius+roundoff
        if outside.any():
            anchor=self.anchor[outside];direction=points[outside]-anchor;facets=self.facets[outside];r=radius[outside]
            low=np.zeros(len(r));high=np.ones(len(r))
            for _ in range(56):
                mid=(low+high)*.5;distance=closest_triangle_points(anchor+direction*mid[:,None],facets)[2]
                feasible=distance<=r+roundoff[outside];low=np.where(feasible,mid,low);high=np.where(feasible,high,mid)
            points[outside]=anchor+direction*low[:,None]
        output[self.ids]=points;self.projections+=1
        if np.any(self.source_distance(output)>radius+1e-12):raise ValueError('Original source facet envelope lost feasibility')
        return output


def contract(current,source,tree,direction,influence,*,margin=.001,normal_limit=.2,max_distance=None,cancelled=None):
    from mathutils import Vector
    from .geometry import points
    current=points(current);source=points(source)
    direction=np.asarray(direction,dtype=float)
    if direction.shape!=(3,) or not np.isfinite(direction).all() or np.linalg.norm(direction)<1e-12:
        raise ValueError('Source envelope direction must be a finite nonzero vector')
    direction=direction/np.linalg.norm(direction)
    influence=np.asarray(influence,dtype=float)
    if influence.shape!=(len(current),) or not np.isfinite(influence).all():raise ValueError('Envelope weights must match the result')
    if margin<0 or not np.isfinite(margin):raise ValueError('Envelope margin must be finite and nonnegative')
    if not 0<=normal_limit<=1:raise ValueError('Envelope normal limit must be between zero and one')
    span=float(np.linalg.norm(np.ptp(source,axis=0)))
    limit=span*.2 if max_distance is None else float(max_distance)
    if limit<=0 or not np.isfinite(limit):raise ValueError('Envelope distance limit must be finite and positive')
    plane=float(np.max(source@direction)+max(span*.05,margin*2))
    result=current.copy();hits=missing=remote=0
    for i in np.flatnonzero(influence>0):
        if cancelled and cancelled():raise RuntimeError('Envelope fitting cancelled; previous surface retained')
        p=current[i];origin=p+direction*(plane-p@direction)
        hit,normal,_,_=tree.ray_cast(Vector(origin),Vector(-direction))
        if hit is None or normal@Vector(direction)<normal_limit:missing+=1;continue
        excess=float((p-np.asarray(hit))@direction-margin)
        if excess<=0:continue
        if excess>limit:remote+=1;continue
        result[i]-=direction*excess*np.clip(influence[i],0,1);hits+=1
    return result,{'solver':'SCOPED_SOURCE_ENVELOPE_V1','hits':hits,'unsupported_rays':missing,
                   'remote_rays_retained':remote,'direction':direction.tolist(),'margin':float(margin)}
