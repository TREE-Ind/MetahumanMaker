"""Joint source/artist/contact feasibility for an external numerical worker.

Clarabel solves the convex local projection or an elastic quadratic model.
The optional model Hessian must be positive semidefinite. Original source centers stay
fixed throughout the substep. Nonlinear mesh and native-host checks remain
the caller's responsibility; no global injectivity claim is made here.
"""
import time
import numpy as np
from scipy import sparse


def project_joint_conic(candidate,current,free,matrix,bounds,lower,upper,source_projector,deadline,*,model=None):
    """Return a measured feasible local step, or retain current coordinates.

    A supplied model contains hessian, gradient, soft_matrix, soft_bounds
    and penalty in physical displacement units. Time-limited elastic primals
    retain their real solver status and pass the same feasibility checks;
    callers must verify nonlinear merit, orientation and actual intersections.
    """
    import clarabel
    start=time.monotonic();current=np.asarray(current,dtype=float);candidate=np.asarray(candidate,dtype=float);free=np.asarray(free,dtype=bool)
    if current.ndim!=2 or current.shape[1]!=3 or candidate.shape!=current.shape or free.shape!=(len(current),) or not np.isfinite(current).all() or not np.isfinite(candidate).all():raise ValueError('Invalid joint projection coordinates')
    method=source_projector.contract.get('method');facet_envelope=method=='EXACT_SOURCE_FACET_AND_ARTIST_BOXES_V1'
    if method not in {'EXACT_SOURCE_WITNESS_AND_ARTIST_BOXES_V1','EXACT_SOURCE_FACET_AND_ARTIST_BOXES_V1'} or source_projector.witness is None:raise ValueError('Conic projection requires prepared source geometry and original artist boxes')
    movable=np.flatnonzero(free);ndof=3*len(movable);lookup=np.full(len(current),-1,int);lookup[movable]=np.arange(len(movable))
    ids=source_projector.ids;variables=lookup[ids]
    if np.any(variables<0):raise ValueError('Source constraints include fixed vertices')
    matrix=sparse.csr_matrix(matrix);bounds=np.asarray(bounds,dtype=float);lower=np.asarray(lower,dtype=float);upper=np.asarray(upper,dtype=float)
    if matrix.shape!=(len(bounds),ndof) or bounds.ndim!=1 or lower.shape!=(ndof,) or upper.shape!=lower.shape or not all(np.isfinite(a).all() for a in (bounds,lower,upper,matrix.data)) or np.any(lower>upper):raise ValueError('Invalid joint contact/box constraints')
    lower=lower.copy();upper=upper.copy()
    artist_lower=source_projector.reference[ids]-source_projector.limits[ids]-current[ids]
    artist_upper=source_projector.reference[ids]+source_projector.limits[ids]-current[ids]
    lower.reshape(-1,3)[variables]=np.maximum(lower.reshape(-1,3)[variables],artist_lower)
    upper.reshape(-1,3)[variables]=np.minimum(upper.reshape(-1,3)[variables],artist_upper)
    if np.any(lower>upper):raise ValueError('Step bounds and original artist boxes do not intersect')
    step=(candidate-current)[free].ravel();before=float(np.maximum(bounds-matrix@step,0.).max(initial=0))
    centers=source_projector.centers;radii=np.broadcast_to(source_projector.source_radius,(len(ids),))
    metadata=dict(method='JOINT_SOURCE_BOX_CONTACT_SOCP_V1',solver_version=clarabel.__version__,status='NotRun',valid=False,iterations=0,hard_residual_before=before)
    def finish(points,status,valid,iterations=0):
        delta=(points-current)[free].ravel();hard=float(np.maximum(bounds-matrix@delta,0.).max(initial=0))
        metadata.update(status=str(status),valid=bool(valid),iterations=int(iterations),hard_residual_after=hard,elapsed_seconds=time.monotonic()-start)
        return points,metadata
    if time.monotonic()>=deadline:return finish(current.copy(),'Deadline',False)
    if not ndof:return finish(current.copy(),'NoMovableVertices',True)
    unit=max(float(np.linalg.norm(np.ptp(current,axis=0)))*.001,1e-6)
    row_norm=np.sqrt(np.asarray(matrix.multiply(matrix).sum(1)).ravel());active=row_norm>1e-15
    if np.any((~active)&(bounds>1e-12)):return finish(current.copy(),'InfeasibleFixedSeparator',False)
    hard=sparse.diags(1/row_norm[active])@matrix[active]
    linear=sparse.vstack((-hard,sparse.eye(ndof),-sparse.eye(ndof)),format='csc')
    rhs=np.r_[-bounds[active]/(row_norm[active]*unit),upper/unit,-lower/unit]
    # For each source point: (r, current + delta - source_center) in SOC(4).
    rows=(np.arange(len(ids))[:,None]*4+np.arange(1,4)).ravel();columns=(variables[:,None]*3+np.arange(3)).ravel()
    soc=sparse.coo_matrix((-np.ones(len(rows)),(rows,columns)),shape=(4*len(ids),ndof)).tocsc()
    soc_rhs=np.column_stack((radii/unit,(current[ids]-centers)/unit)).ravel()
    nfacet=2*len(ids) if facet_envelope else 0;nbase=ndof+nfacet
    if facet_envelope:
        facets=source_projector.facets;edge1=(facets[:,1]-facets[:,0])/unit;edge2=(facets[:,2]-facets[:,0])/unit
        facet_columns=np.r_[np.repeat(np.arange(len(ids))*2,3),np.repeat(np.arange(len(ids))*2+1,3)]
        facet_soc=sparse.coo_matrix((np.r_[edge1.ravel(),edge2.ravel()],(np.tile(rows,2),facet_columns)),shape=(4*len(ids),nfacet)).tocsc()
        soc=sparse.hstack((soc,facet_soc),format='csc')
        soc_rhs=np.column_stack((radii/unit,(current[ids]-facets[:,0])/unit)).ravel()
        # a >= 0, b >= 0, a+b <= 1 certifies a point on each
        # immutable original source triangle: v0 + a*(v1-v0)+b*(v2-v0).
        face_index=np.arange(len(ids));bary_rows=np.r_[3*face_index,3*face_index+1,3*face_index+2,3*face_index+2]
        bary_columns=ndof+np.r_[2*face_index,2*face_index+1,2*face_index,2*face_index+1]
        bary=sparse.coo_matrix((np.r_[-np.ones(len(ids)*2),np.ones(len(ids)*2)],(bary_rows,bary_columns)),shape=(len(ids)*3,nbase)).tocsc()
        linear=sparse.vstack((sparse.hstack((linear,sparse.csc_matrix((linear.shape[0],nfacet)))),bary),format='csc')
        rhs=np.r_[rhs,np.tile([0.,0.,1.],len(ids))]
        metadata.update(source_envelope='ORIGINAL_TRIANGLE_OFFSET',facet_variables=nfacet)
    A=sparse.vstack((linear,soc),format='csc');b=np.r_[rhs,soc_rhs]
    cones=[clarabel.NonnegativeConeT(len(rhs))]+[clarabel.SecondOrderConeT(4) for _ in ids]
    objective=sparse.block_diag((sparse.eye(ndof),sparse.csc_matrix((nfacet,nfacet))),format='csc');linear_objective=np.r_[-step/unit,np.zeros(nfacet)]
    soft=None;soft_bounds=None;nsoft=0;nangle=0;nangle_rows=0;nc=0
    if model is not None:
        hessian=sparse.csc_matrix(model['hessian']);gradient=np.asarray(model['gradient'],dtype=float)
        soft=sparse.csc_matrix(model['soft_matrix']);soft_bounds=np.asarray(model['soft_bounds'],dtype=float);penalty=float(model['penalty'])
        if hessian.shape!=(ndof,ndof) or gradient.shape!=(ndof,) or soft.shape!=(len(soft_bounds),ndof) or soft_bounds.ndim!=1 or not all(np.isfinite(v).all() for v in (hessian.data,gradient,soft.data,soft_bounds)) or not np.isfinite(penalty) or penalty<0:raise ValueError('Invalid joint elastic objective')
        symmetry=hessian-hessian.T
        if np.max(np.abs(symmetry.data),initial=0)>1e-10:raise ValueError('Joint elastic Hessian must be symmetric')
        nsoft=len(soft_bounds)
        # Both displacement and elastic slacks use the same length unit.
        # Divide the physical objective by unit**2, preserving its minimizer.
        objective=sparse.block_diag((hessian,sparse.csc_matrix((nfacet,nfacet)),sparse.eye(nsoft)*10.),format='csc')
        linear_objective=np.r_[gradient/unit,np.zeros(nfacet),np.full(nsoft,penalty/unit)]
        soft_extended=sparse.hstack((soft,sparse.csc_matrix((nsoft,nfacet))),format='csc')
        elastic=sparse.vstack((sparse.hstack((-soft_extended,-sparse.eye(nsoft))),sparse.hstack((sparse.csc_matrix((nsoft,nbase)),-sparse.eye(nsoft)))),format='csc')
        padded=sparse.hstack((A,sparse.csc_matrix((A.shape[0],nsoft))),format='csc')
        A=sparse.vstack((elastic,padded),format='csc');b=np.r_[-soft_bounds/unit,np.zeros(nsoft),b]
        cones=[clarabel.NonnegativeConeT(nsoft*2)]+cones
        metadata.update(method='JOINT_ELASTIC_SOURCE_FACET_CONTACT_SOCP_V1' if facet_envelope else 'JOINT_ELASTIC_SOURCE_BOX_CONTACT_SOCP_V1',elastic_constraints=nsoft)
    angle=None
    if model is not None and model.get('angle_cones') is not None:
        angle=model['angle_cones'];B=sparse.csr_matrix(angle['matrix']);offset=np.asarray(angle['offset'],dtype=float);direction=np.asarray(angle['normals'],dtype=float);cosine=np.asarray(angle['minimum_cosine'],dtype=float);nangle=len(cosine)
        if B.shape!=(3*nangle,ndof) or offset.shape!=(nangle,3) or direction.shape!=offset.shape or not all(np.isfinite(v).all() for v in (B.data,offset,direction,cosine)) or np.any((cosine<=0)|(cosine>1)) or np.any(abs(np.linalg.norm(direction,axis=1)-1)>1e-8):raise ValueError('Invalid affine source-normal cones')
        if nangle:
            direction_size=np.linalg.norm(direction,axis=1);effective_cosine=cosine/direction_size
            if np.all(effective_cosine<1):
                # n.C >= c*|C| is exactly equivalent (0<c<1) to
                # |C_tangent| <= sqrt(1-c*c)/c * (n.C). Removing the
                # repeated large longitudinal component avoids subtracting
                # nearly equal numbers in narrow, almost-aligned cones.
                unit_direction=direction/direction_size[:,None]
                axis=np.eye(3)[np.argmin(np.abs(unit_direction),axis=1)]
                t1=np.cross(unit_direction,axis);t1/=np.linalg.norm(t1,axis=1)[:,None];t2=np.cross(unit_direction,t1)
                tangent=np.stack((unit_direction*(np.sqrt(1-effective_cosine*effective_cosine)/effective_cosine)[:,None],t1,t2),axis=1)
                rows=np.broadcast_to(np.arange(3*nangle).reshape(nangle,3,1),(nangle,3,3))
                columns=np.broadcast_to(np.arange(nangle)[:,None,None]*3+np.arange(3)[None,None,:],(nangle,3,3))
                transform=sparse.coo_matrix((tangent.ravel(),(rows.ravel(),columns.ravel())),shape=(3*nangle,3*nangle)).tocsr()
                angular=-(transform@B)*unit;angular_rhs=np.asarray(transform@offset.ravel()).ravel();dimension=3
                metadata['source_angle_cone_representation']='EQUIVALENT_TANGENT_FRAME_SOC3_V1'
            else:
                D=sparse.coo_matrix(((direction/cosine[:,None]).ravel(),(np.repeat(np.arange(nangle),3),np.arange(3*nangle))),shape=(nangle,3*nangle)).tocsr()
                order=np.column_stack((np.arange(nangle),nangle+np.arange(3*nangle).reshape(-1,3))).ravel()
                angular=-sparse.vstack((D@B,B),format='csr')[order]*unit
                angular_rhs=np.column_stack((np.sum(direction*offset,axis=1)/cosine,offset)).ravel();dimension=4
                metadata['source_angle_cone_representation']='FULL_AREA_SOC4'
            nangle_rows=dimension*nangle
            angular=sparse.hstack((angular,sparse.csc_matrix((nangle_rows,objective.shape[0]-ndof))),format='csc')
            A=sparse.vstack((A,angular),format='csc');b=np.r_[b,angular_rhs]
            cones+=[clarabel.SecondOrderConeT(dimension) for _ in range(nangle)]
        metadata['source_angle_cones']=nangle
        if angle.get('source_field_derivatives'):metadata['source_field_derivatives']=angle['source_field_derivatives']
        def angle_error(value):
            area=offset+(B@value).reshape(-1,3)
            return float(np.maximum(cosine*np.linalg.norm(area,axis=1)-np.sum(direction*area,axis=1),0.).max(initial=0))
    coverage=None
    if model is not None and model.get('coverage_cones') is not None:
        coverage=model['coverage_cones'];C=sparse.csr_matrix(coverage['matrix']);co=np.asarray(coverage['offset'],dtype=float);cr=np.asarray(coverage['radii'],dtype=float);nc=len(cr)
        if C.shape!=(3*nc,ndof) or co.shape!=(nc,3) or cr.shape!=(nc,) or not all(np.isfinite(value).all() for value in (C.data,co,cr)) or np.any(cr<=0):raise ValueError('Invalid reverse-coverage cones')
        order=np.column_stack((np.arange(nc),nc+np.arange(3*nc).reshape(-1,3))).ravel()
        covered=sparse.vstack((sparse.csr_matrix((nc,ndof)),-C),format='csr')[order]
        covered=sparse.hstack((covered,sparse.csc_matrix((4*nc,objective.shape[0]-ndof))),format='csc')
        A=sparse.vstack((A,covered),format='csc');b=np.r_[b,np.column_stack((cr/unit,co/unit)).ravel()]
        cones+=[clarabel.SecondOrderConeT(4) for _ in range(nc)]
        metadata['reverse_coverage_cones']=nc
        def coverage_error(value):return float(np.maximum(np.linalg.norm(co+(C@value).reshape(-1,3),axis=1)-cr,0.).max(initial=0))
    settings=clarabel.DefaultSettings();settings.verbose=False;settings.max_iter=100;settings.max_threads=1
    settings.time_limit=max(.001,deadline-time.monotonic());settings.tol_feas=1e-10;settings.tol_gap_abs=1e-10;settings.tol_gap_rel=1e-10
    if model is not None and model.get('strict_feasibility'):
        settings.max_iter=200;settings.tol_feas=1e-14;settings.tol_gap_abs=1e-14;settings.tol_gap_rel=1e-14
        settings.iterative_refinement_abstol=1e-14;settings.iterative_refinement_reltol=1e-14
        metadata['strict_feasibility_tolerance']=1e-14
    solver=clarabel.DefaultSolver(sparse.triu(objective,format='csc'),linear_objective,A,b,cones,settings);result=solver.solve()
    status=str(result.status)
    converged=status in ('Solved','AlmostSolved')
    # An inexact elastic step can still be useful. Preserve the solver's
    # actual status and accept only an independently measured feasible
    # primal; the caller must also require actual nonlinear merit descent.
    inexact=model is not None and status in ('MaxTime','MaxIterations')
    metadata.update(solver_converged=converged,inexact_primal=inexact,solver_primal_residual=getattr(result,'r_prim',None),solver_dual_residual=getattr(result,'r_dual',None))
    if status in ('PrimalInfeasible','AlmostPrimalInfeasible') and result.z is not None:
        # This diagnoses only the current convex proposal. It is not proof
        # that the artist's nonlinear surface repair is globally impossible.
        z=np.asarray(result.z,dtype=float)
        if z.shape==b.shape and np.isfinite(z).all():
            sizes=[('elastic',2*nsoft),('hard_halfspaces',int(active.sum())),('movement_boxes',2*ndof),
                   ('source_barycentric',3*len(ids) if facet_envelope else 0),('source_distance',4*len(ids)),
                   ('source_angles',nangle_rows),('reverse_coverage',4*nc)]
            offset=0;groups={}
            for name,size in sizes:
                group=z[offset:offset+size];groups[name]=dict(dual_l1=float(np.abs(group).sum()),rhs_dot=float(b[offset:offset+size]@group));offset+=size
            if offset!=len(z):raise AssertionError('Conic diagnostic row groups disagree with the assembled problem')
            hard_start=2*nsoft;weights=z[hard_start:hard_start+int(active.sum())];top=np.argsort(np.abs(weights))[-8:][::-1]
            ray_scale=max(float(np.abs(z).max(initial=0)),1e-300)
            metadata['local_infeasibility']=dict(scope='CURRENT_AFFINE_PROPOSAL_ONLY',rhs_dot=float(b@z),
                stationarity_max=float(np.abs(A.T@z).max(initial=0)),ray_scale=ray_scale,
                normalized_rhs_dot=float(b@(z/ray_scale)),normalized_stationarity_max=float(np.abs(A.T@(z/ray_scale)).max(initial=0)),groups=groups,
                top_hard_rows=[dict(row=int(np.flatnonzero(active)[k]),dual=float(weights[k])) for k in top])
    if (not converged and not inexact) or result.x is None:return finish(current.copy(),status,False,result.iterations)
    delta=np.asarray(result.x[:ndof])*unit
    if delta.shape!=(ndof,) or not np.isfinite(delta).all():return finish(current.copy(),'InvalidSolution',False,result.iterations)
    points=current.copy();points[free]+=np.clip(delta,lower,upper).reshape(-1,3)
    # Remove tiny cone/box solver residuals without refreshing source centers.
    raw_points=points.copy()
    metadata['raw_candidate_source_residual_m']=float(np.max(source_projector.source_distance(points)-radii,initial=0))
    metadata['raw_candidate_hard_residual']=float(np.maximum(bounds-matrix@((points-current)[free].ravel()),0.).max(initial=0))
    if angle is not None:metadata['raw_candidate_angle_residual']=angle_error((points-current)[free].ravel())
    points=source_projector(points);delta=(points-current)[free].ravel()
    metadata['source_projection_displacement_m']=float(np.linalg.norm(points-raw_points,axis=1).max(initial=0))
    hard_error=float(np.maximum(bounds-matrix@delta,0.).max(initial=0))
    if model is not None and hard_error>1e-9:
        # Restore the existing tolerance by shortening toward the feasible
        # current surface. This does not relax any halfspace or source bound.
        # Both endpoints already lie in the convex source/artist envelope.
        current_feasible=(np.max(bounds,initial=0)<=9e-10 and np.all(lower<=0) and np.all(upper>=0)
            and np.max(source_projector.source_distance(current)-radii,initial=0)<=1e-10
            and np.max(np.abs(current[ids]-source_projector.reference[ids])-source_projector.limits[ids],initial=0)<=1e-10)
        if current_feasible:
            residual=bounds-matrix@delta;bad=residual>9e-10
            fraction=float(np.clip(np.min((9e-10-bounds[bad])/(residual[bad]-bounds[bad]),initial=1.),0,1))
            metadata.update(raw_candidate_hard_residual=hard_error,hard_feasibility_backtrack_fraction=fraction)
            points=current+(points-current)*fraction;delta=(points-current)[free].ravel()
            hard_error=float(np.maximum(bounds-matrix@delta,0.).max(initial=0))
    angular_error=0.
    if angle is not None:
        angular_error=angle_error(delta)
        if angular_error>1e-9 and angle_error(np.zeros(ndof))<=9e-10 and np.max(bounds,initial=0)<=9e-10 and np.all(lower<=0) and np.all(upper>=0) and np.max(source_projector.source_distance(current)-radii,initial=0)<=1e-10:
            original=delta.copy();low=0.;high=1.
            for _ in range(40):
                middle=(low+high)*.5
                if angle_error(original*middle)<=9e-10:low=middle
                else:high=middle
            points=current.copy();points[free]+=original.reshape(-1,3)*low;delta=(points-current)[free].ravel()
            angular_error=angle_error(delta);hard_error=float(np.maximum(bounds-matrix@delta,0.).max(initial=0));metadata['source_angle_backtrack_fraction']=low
        metadata['source_angle_residual']=angular_error
    box_error=float(max(np.max(lower-delta,initial=0),np.max(delta-upper,initial=0)))
    source_error=float(np.max(source_projector.source_distance(points)-radii,initial=0))
    metadata.update(candidate_hard_residual=hard_error,box_residual=box_error,source_residual=source_error,protected_coordinates_exact=bool(np.array_equal(points[~free],current[~free])))
    if model is not None:
        def energy(value):
            slack=np.maximum(soft_bounds-soft@value,0.)
            return float(.5*value@(hessian@value)+gradient@value+penalty*slack.sum()+5.*np.sum(slack**2))
        metadata.update(elastic_model_before=energy(np.zeros(ndof)),elastic_model_after=energy(delta))
    coverage_residual=coverage_error(delta) if coverage is not None else 0.
    if coverage is not None:metadata['reverse_coverage_residual_m']=coverage_residual
    # A transported area cone is a local surrogate. Its small affine-model
    # error need not reject an endpoint before the exact curved-source test.
    # Hard boxes/source/coverage and all caller/native tolerances stay fixed.
    affine_angle_tolerance=1e-8 if model is not None and model.get('strict_feasibility') and angle is not None and angle.get('source_field_derivatives') else 1e-9
    metadata['affine_angle_tolerance']=affine_angle_tolerance
    valid=hard_error<=1e-9 and angular_error<=affine_angle_tolerance and box_error<=1e-10 and source_error<=1e-10 and coverage_residual<=1e-10 and np.array_equal(points[~free],current[~free])
    return finish(points if valid else current.copy(),status,valid,result.iterations)
