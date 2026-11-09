"""External original-source attachment reconstruction with native proof output."""
import time
import numpy as np


def fit(data,options,*,progress=None,cancelled=None):
    """Compare bounded repair candidates against the same immutable snapshot."""
    from .surface_defects import repair_clusters
    model=options.get('attachment_model','AUTO')
    if model not in {'AUTO','SOURCE_NORMALS','SOURCE_TRANSPORT','CHART_ONLY'}:raise ValueError('Choose AUTO, SOURCE_NORMALS, SOURCE_TRANSPORT or CHART_ONLY attachment fitting')
    initial_model=('SOURCE_TRANSPORT' if 'native_storage_coords' in data else 'SOURCE_NORMALS') if model=='AUTO' else model
    comparison_model='CHART_ONLY' if model=='AUTO' else initial_model
    started=time.monotonic();best=None;best_score=None;attempts=[];limit=4;pending=[(0,6,'CHART','PREFER_SOURCE',initial_model)];queued=set(pending)
    while pending and len(attempts)<limit:
        if cancelled and cancelled():raise RuntimeError('Attachment reconstruction cancelled')
        if attempts and time.monotonic()-started>115:break
        rank,rings,reverse_model,boundary_policy,fit_model=pending.pop(0);attempt_number=len(attempts)
        candidate_options=dict(options,_repair_candidate_rank=rank,_repair_chart_rings=rings,_repair_reverse_model=reverse_model,_repair_boundary_policy=boundary_policy,_repair_fit_model=fit_model)
        def update(label,value):
            if progress:progress('Repair candidate '+str(attempt_number+1)+': '+label,min(.95,(attempt_number+value)/limit))
        result=_fit_patch(data,candidate_options,progress=update,cancelled=cancelled)
        report=result[1][0];alignment=report.get('source_alignment',{})
        row=dict(candidate_rank=rank,requested_rings=rings,reverse_model=reverse_model,boundary_policy=boundary_policy,fit_model=report.get('fit_model',fit_model),requested_fit_model=fit_model,source_normal_objective=report.get('source_normal_objective'),chart_boundary=report.get('chart',{}).get('boundary_continuation'),accepted=report['accepted'],seed=report.get('chart',{}).get('seed'),
            reverse_projection=report.get('reverse_projection'),source_alignment=alignment,reason=report.get('reason'),elapsed_seconds=report.get('elapsed_seconds'))
        numerical=report.get('numerical',{})
        row.update(numerical_method=numerical.get('method'),endpoint_limits_restored=numerical.get('endpoint_limits_restored'),
            refinement_reason=numerical.get('polish',{}).get('reason'),native_storage=report.get('native_storage'),
            nonlinear_search_seconds=numerical.get('search',{}).get('elapsed_seconds'),refinement_seconds=numerical.get('polish',{}).get('seconds'))
        attempts.append(row)
        score=(alignment.get('after_back_faces',float('inf')),alignment.get('after_severe_reference_area_m2',float('inf')),alignment.get('after_energy',float('inf')))
        if report.get('repair_target')!='ALIGNMENT':score=(sum(report.get('after',{}).get(d+'_rms',float('inf')) for d in ('forward','reverse')),)+score
        if best is None or report['accepted'] and not best[1][0]['accepted'] or bool(report['accepted'])==bool(best[1][0]['accepted']) and score<best_score:
            best=result;best_score=score
        if progress and not report['accepted']:
            progress('Candidate '+str(attempt_number+1)+' retained current surface: '+report.get('reason','No verified improvement'),min(.95,len(attempts)/limit))
        if report.get('repair_target')!='ALIGNMENT':
            if reverse_model=='CHART':pending=[(rank,rings,'NEAREST','REGISTERED',comparison_model)];queued.add(pending[0]);continue
            break
        if report['accepted'] and numerical.get('endpoint_limits_restored') and alignment.get('after_back_faces',0)<alignment.get('before_back_faces',0):
            # A verified nonlinear fold repair is ready for native review.
            # Do not spend the remaining slots on a weaker chart hypothesis
            # after resolving original reversed faces with fixed limits.
            break
        if report['accepted'] and numerical.get('endpoint_limits_restored'):
            clusters=repair_clusters(report.get('source_defects',{'clusters':[]}))
            local_area=clusters[rank].get('reference_area_m2',0) if rank<len(clusters) else 0
            resolved_area=alignment.get('before_severe_reference_area_m2',0)-alignment.get('after_severe_reference_area_m2',0)
            if local_area>0 and resolved_area>=.25*local_area:
                # A broad surface repair can make meaningful progress without
                # removing a reversed triangle. It is ready for native review
                # under the same complete acceptance gates.
                break
        if report['accepted'] and alignment.get('after_back_faces')==0 and alignment.get('after_severe_faces')==0:break
        if len(attempts)==1:
            for next_rank in range(1,min(limit,len(repair_clusters(report.get('source_defects',{'clusters':[]}))))):
                pair=(next_rank,6,'CHART','PREFER_SOURCE',initial_model);pending.append(pair);queued.add(pair)
        # A formally accepted small patch can leave most of its original
        # fold intact. Spend one existing candidate slot on a wider join,
        # rather than treating any tiny improvement as adequate coverage.
        clusters=repair_clusters(report.get('source_defects',{'clusters':[]}))
        cluster=clusters[rank] if rank<len(clusters) else {}
        local_back=cluster.get('back_facing_faces',0)
        unresolved_back=local_back and alignment.get('after_back_faces',0)>alignment.get('before_back_faces',0)-local_back
        # A broad pinched region can contain no reversed triangles. Give it
        # the same nonlinear redistribution path when the small chart leaves
        # most of its original severe reference area unresolved. This only
        # chooses a proposal; every existing endpoint gate still applies.
        unresolved_area=cluster.get('reference_area_m2',0)>0 and alignment.get('after_severe_reference_area_m2',float('inf'))>alignment.get('before_severe_reference_area_m2',0)-.25*cluster.get('reference_area_m2',0)
        widen_needed=rings==6 and (unresolved_back or unresolved_area)
        if widen_needed:
            # Nonlinear restoration uses observed source distances rather
            # than chart-inverse fitting. Its widened preparation uses the
            # registered chart collar, with all endpoint limits unchanged.
            pair=(rank,12,'NEAREST','REGISTERED',fit_model) if fit_model=='SOURCE_TRANSPORT' else (rank,12,reverse_model,boundary_policy,fit_model)
            if pair not in queued:pending.insert(0,pair);queued.add(pair)
        # Chart inverse transport is a competing hypothesis, not authority
        # over the previous nearest-surface model. Compare the same wider
        # patch (or the first patch when no widening is needed) on the same
        # original snapshot, within the existing candidate/time limits.
        if reverse_model=='CHART' and (rings>=12 or rank==0 and not widen_needed):
            pair=(rank,rings,'NEAREST','REGISTERED',comparison_model)
            if pair not in queued:pending.insert(0,pair);queued.add(pair)
        # A valid 2D chart can still join poorly in 3D. Give a rejected or
        # severely shortened repair a wider free collar, using the same
        # original snapshot, source data and public movement budgets.
        if not report['accepted'] or report.get('alignment_endpoint_fraction',1.)<.5 or alignment.get('after_back_faces')==0 and alignment.get('after_severe_faces',0)>0:
            pair=(rank,rings+2,reverse_model,boundary_policy,fit_model)
            if pair[1]<=10 and pair not in queued:pending.append(pair);queued.add(pair)
        if not report['accepted']:
            # Failed widening must not exhaust the search before other regions
            # receive a first attempt. Preserve priority within both cohorts.
            visited={row['candidate_rank'] for row in attempts}
            pending.sort(key=lambda candidate:candidate[0] in visited)
    chosen=best[1][0];chosen['candidate_search']=attempts;chosen['candidate_search_seconds']=time.monotonic()-started
    chosen['candidate_schedule']=dict(method='UNVISITED_AFTER_REJECTION_V1',attempted_regions=sorted({row['candidate_rank'] for row in attempts}),candidate_limit=limit,search_budget_seconds=115)
    if progress:progress('Ready for native reconstruction checks',1.)
    return best


def _fit_patch(data,options,*,progress=None,cancelled=None):
    import igl
    from scipy import sparse
    from .intrinsic_disk import propose
    from .constrained_surface_fit import solve
    from .source_envelope import SourceFacetBoxes,source_reachable_boxes
    from .vertex_contract import witnesses,audit_witnesses,fingerprint
    from .surface_fit_measurement import measure_surface_fit,assemble_reverse_projection,UnsupportedSurfaceCoverage
    from .surface_defects import analyze,compare,repair_clusters
    from .surface_oracle import SurfaceOracle
    from .surface_support import observation_scale,reach_confidence
    started=time.monotonic()
    def check():
        if cancelled and cancelled():raise RuntimeError('Attachment reconstruction cancelled')
        if time.monotonic()-started>210:raise RuntimeError('Attachment reconstruction exceeded its preparation budget')
    def update(label,value):
        check()
        if progress:progress(label,value)
    raw=np.asarray(data['current'],dtype=float);weld=np.asarray(data['weld'],dtype=np.int64);count=np.bincount(weld)
    if raw.ndim!=2 or raw.shape[1]!=3 or weld.shape!=(len(raw),) or np.any(count==0) or not np.isfinite(raw).all():raise ValueError('Invalid reconstruction snapshot')
    def physical(points):return np.column_stack([np.bincount(weld,weights=points[:,axis])/count for axis in range(3)])
    reference=physical(raw);n=len(reference);tri=weld[np.asarray(data['triangles'],dtype=np.int64)];tri=tri[np.all(np.diff(np.sort(tri,axis=1),axis=1)!=0,axis=1)]
    source=np.asarray(data['source'],dtype=float);source_tri=np.asarray(data['source_triangles'],dtype=np.int64);span=float(np.linalg.norm(np.ptp(reference,axis=0)));guard=max(span*2.5e-7,1e-10)
    regions=options['regions'];influence=np.zeros(n);assignment=np.full(n,-1,int)
    for index,_ in enumerate(regions):
        values=np.asarray(data['influence_'+str(index)],dtype=float)
        if values.shape!=(n,) or not np.isfinite(values).all() or np.any((values<0)|(values>1)):raise ValueError('Invalid reconstruction region weights')
        use=values>influence;influence[use]=values[use];assignment[use]=index
    supported=influence>1e-4;all_ids=np.flatnonzero(supported)
    if not len(all_ids):raise ValueError('This region is empty or fully pinned')
    parameter_reference=physical(np.asarray(data.get('parameter_reference',raw)))
    local_scale=observation_scale(reference,tri)
    update('Finding the attachment source',.03);distances=np.full(n,-np.inf);targets=reference.copy();source_field_faces=np.full(n,-1,dtype=np.int64);trees=[]
    try:
        for index,_ in enumerate(regions):
            check();faces=np.asarray(data['source_faces_'+str(index)],dtype=np.int64);tree=igl.AABB();tree.init(source,source_tri[faces]);trees.append((tree,faces))
            ids=all_ids[assignment[all_ids]==index]
            if not len(ids):continue
            _,local,_=tree.squared_distance(source,source_tri[faces],reference[ids]);record=witnesses(reference[ids],source,source_tri,faces[local]);audit_witnesses(record,source,source_tri)
            oracle=SurfaceOracle(source,source_tri,faces)
            residual=record['residual'];clearance=oracle.boundary_clearance(record['surface_points'],cutoff=max(float(residual.max())*.5,span*5e-6),cancelled=cancelled)
            observed=((residual<=span*5e-6)|(clearance>=residual*.5))&(reach_confidence(residual,local_scale[ids],rings=float(options.get('observation_rings',2.)))>.01)
            good=ids[observed];targets[good]=record['surface_points'][observed];distances[good]=residual[observed];source_field_faces[good]=record['source_triangle_ids'][observed]
            del oracle
        supported&=source_field_faces>=0;all_ids=np.flatnonzero(supported)
        if not len(all_ids):
            return raw.copy(),[dict(solver='INTRINSIC_ATTACHMENT_RECONSTRUCTION_V1',accepted=False,review_required=True,regions=regions,reason='No interior source support was found. Review source masks or use a supported patch.')],{}
        policy=options.get('repair_target','AUTO')
        if policy not in {'AUTO','GAP','ALIGNMENT'}:raise ValueError('Choose AUTO, GAP or ALIGNMENT for attachment repair')
        defects,_=analyze(reference,tri,supported,source,source_tri,source_field_faces,parameter_reference=parameter_reference)
        clusters=repair_clusters(defects)
        strategy='ALIGNMENT' if policy!='GAP' and clusters else 'GAP'
        patch_model=options.get('_repair_fit_model','SOURCE_NORMALS')
        if strategy=='GAP' and patch_model=='SOURCE_TRANSPORT':patch_model='SOURCE_NORMALS'
        if policy=='ALIGNMENT' and not clusters:
            return raw.copy(),[dict(solver='INTRINSIC_ATTACHMENT_RECONSTRUCTION_V1',accepted=False,review_required=True,regions=regions,reason='No coherent surface-fold region was found inside these masks.',source_defects=defects)],{}
        rank=int(options.get('_repair_candidate_rank',0))
        if strategy=='ALIGNMENT' and not 0<=rank<len(clusters):raise ValueError('Repair candidate is outside the coherent defect list')
        seed=clusters[rank]['seed'] if strategy=='ALIGNMENT' else int(np.argmax(distances));region=int(assignment[seed]);tree,faces=trees[region]
        eligible=supported&(assignment==region)
        update('Building an oriented source chart',.12)
        correspondence={}
        try:goal,chart=propose(reference,tri,eligible,source,source_tri,faces,seed,rings=int(options.get('_repair_chart_rings',6)),parameter_reference=parameter_reference,cancelled=cancelled,correspondence=correspondence,prefer_source_boundary=options.get('_repair_boundary_policy','PREFER_SOURCE')!='REGISTERED')
        except ValueError as exc:
            return raw.copy(),[dict(solver='INTRINSIC_ATTACHMENT_RECONSTRUCTION_V1',accepted=False,review_required=True,regions=regions,selected_region=regions[region],repair_target=strategy,source_defects=defects,chart=dict(seed=seed),reason='No supported disk was found inside these masks. Enlarge the matching source/result patches.',diagnostic=str(exc))],{}
        free=np.zeros(n,bool);free[chart['target_vertex_ids'][chart['interior']]]=True
        precision_pins=None
        if patch_model=='SOURCE_NORMALS':
            from .source_angle_contract import attachment_contract,native_precision_pins
            _,_,initial_angles=attachment_contract(reference,tri,source,source_tri,source_field_faces,supported,free,parameter_reference)
            pins,precision_pins=native_precision_pins(reference,tri,initial_angles)
            free[pins]=False
        ids=np.flatnonzero(free)
        if not len(ids):
            return raw.copy(),[dict(solver='INTRINSIC_ATTACHMENT_RECONSTRUCTION_V1',accepted=False,review_required=True,regions=regions,repair_target=strategy,source_defects=defects,reason='This chart has no editable native-precision source-angle support. Enlarge the matching patch.',native_precision_pins=precision_pins)],{}
        def query(points):
            check();_,local,_=tree.squared_distance(source,source_tri[faces],points)
            return witnesses(points,source,source_tri,faces[local])
        from .source_surface_interior import InteriorSamples
        interior=InteriorSamples(reference,tri,free,parameter_reference)
        interior_before=interior.measure(reference,query)
        def measure_complete(candidate):
            result=measure_surface_fit(reference,candidate,tri,source,source_tri,free,faces,**companion)
            result.update(interior_before=interior_before,interior_after=interior.measure(candidate,query))
            return result
        initial=query(reference[ids]);audit_witnesses(initial,source,source_tri)
        edges=np.unique(np.sort(np.vstack((tri[:,[0,1]],tri[:,[1,2]],tri[:,[2,0]])),axis=1),axis=0)
        # Bounds are fixed from the original operation snapshot, not the
        # unconstrained chart proposal or an intermediate solver endpoint.
        public_limits=source_reachable_boxes(reference,all_ids,targets[all_ids],supported,edges,baseline=span*.01,margin=span*.001)*influence[:,None]
        limits=np.maximum(public_limits-guard,public_limits*.5);offset=np.maximum(initial['residual'],span*5e-6)+guard*2
        from .source_budget import constrain
        offset,source_radius=constrain(data,ids,np.full(len(ids),region),offset,offset-guard)
        projector=SourceFacetBoxes(reference,limits,ids,query,source_radius=source_radius,query_contract=dict(vertices=fingerprint(source),triangles=fingerprint(source_tri)),source_vertices=source,source_triangles=source_tri)
        adj=sparse.coo_matrix((np.ones(len(edges)*2),(np.r_[edges[:,0],edges[:,1]],np.r_[edges[:,1],edges[:,0]])),shape=(n,n)).tocsr();lap=sparse.eye(n)-sparse.diags(1/np.maximum(np.asarray(adj.sum(1)).ravel(),1))@adj
        facets=reference[tri];cross=np.cross(facets[:,1]-facets[:,0],facets[:,2]-facets[:,0]);area=np.linalg.norm(cross,axis=1);normals=cross/np.maximum(area[:,None],1e-30)
        chart_facets=source[source_tri[chart['source_face_ids']]];vn=np.cross(chart_facets[:,1]-chart_facets[:,0],chart_facets[:,2]-chart_facets[:,0]);vn/=np.maximum(np.linalg.norm(vn,axis=1)[:,None],1e-30)
        field=np.zeros_like(reference);field[chart['target_vertex_ids']]=vn;face_field=field[tri].sum(1);usable=np.all(np.isin(tri,chart['target_vertex_ids']),axis=1)&(np.linalg.norm(face_field,axis=1)>.5);normals[usable]=face_field[usable]/np.linalg.norm(face_field[usable],axis=1)[:,None]
        mass=np.zeros(n)
        for corner in range(3):np.add.at(mass,tri[:,corner],area/6)
        weights=mass*free
        companion=dict(companion_vertices=data['companion'],companion_triangles=data['companion_triangles']) if 'companion' in data else {}
        try:reverse=assemble_reverse_projection(reference,tri,source,source_tri,free,faces,**companion)
        except UnsupportedSurfaceCoverage as exc:
            return raw.copy(),[dict(solver='INTRINSIC_ATTACHMENT_RECONSTRUCTION_V1',accepted=False,review_required=True,
                regions=regions,selected_region=regions[region],repair_target=strategy,source_defects=defects,
                chart={key:value for key,value in chart.items() if not isinstance(value,np.ndarray)},
                reason='This chart has no independently measurable source coverage. Try another supported chart.',
                diagnostic=str(exc))],{}
        from .reverse_coverage import contract as coverage_contract
        coverage=coverage_contract(reference,tri,reverse,reserve=span*2.5e-7,follow=bool(options.get('_repair_follow_coverage',False)),**companion)
        if options.get('_repair_reverse_model','CHART')=='CHART':
            from .chart_reverse_projection import assemble as chart_reverse
            reverse=chart_reverse(reference,tri,source,source_tri,free,correspondence,reverse,**companion)
        projection=dict(matrix=sparse.diags(weights)+2*reverse['matrix'],load=goal*weights[:,None]+2*reverse['load'],
            mass=weights+2*reverse['mass'],constant=float(np.sum(weights[:,None]*goal**2)+2*reverse['constant']))
        tangent=None;angles=None;normal_objective=None;following=None;public_angles=None
        fit_model=patch_model
        if fit_model in {'SOURCE_NORMALS','SOURCE_TRANSPORT'}:
            from .source_angle_contract import attachment_contract,native_precision_reserve
            from .surface_alignment import tangent_objective
            if fit_model=='SOURCE_TRANSPORT':
                from .following_source_field import original_contract
                original_normals,normal_scope,angles=original_contract(reference,tri,source,source_tri,source_field_faces,supported,free,parameter_reference)
            else:original_normals,normal_scope,angles=attachment_contract(reference,tri,source,source_tri,source_field_faces,supported,free,parameter_reference)
            # The source-normal model must use its original observed field for
            # hard signed-area planes too. A chart is only a proposal and can
            # point elsewhere at a folded join, opposing its own objective.
            normals[normal_scope]=original_normals[normal_scope]
            public_angles=angles
            angles,precision=native_precision_reserve(reference,tri,angles)
            tangent=tangent_objective(reference,tri,original_normals,normal_scope,parameter_reference)
            if fit_model=='SOURCE_TRANSPORT':
                from .following_source_field import FollowingField
                branch_caps=np.zeros(n);branch_caps[ids]=np.maximum(offset-guard,0.)
                following=FollowingField(source,source_tri,tri,source_field_faces,ids,query,angles,normal_scope,faces,reference,branch_caps)
            normal_objective=dict(**(tangent['report'] if tangent is not None else dict(method='ORIGINAL_SOURCE_FIELD_TRANSPORT_V1')),protected_source_angle_faces=len(angles['face_ids']),
                original_source_field=True,original_source_area_planes=True,native_precision_pins=precision_pins,native_precision_reserve=precision,preserve_initially_aligned=True,native_revalidation_required=True)
            if following is not None:
                normal_objective['tangent_method']=normal_objective['method'];normal_objective['method']='ORIGINAL_SOURCE_FIELD_TRANSPORT_V1'
        update('Constraining the attachment repair',.3)
        nonlinear_requested=bool(options.get('_repair_nonlinear_restore',int(chart.get('rings',0))>=12))
        nonlinear_restore=bool(nonlinear_requested and following is not None and strategy=='ALIGNMENT' and
            clusters[rank].get('reference_area_m2',0)>0)
        untangle=(normal_scope&(np.sum(cross*normals,axis=1)<=0)) if following is not None and (options.get('_repair_hard_untangle',False) or nonlinear_restore and clusters[rank].get('back_facing_faces',0)>0) else None
        def checkpoint(points,receipt):update('Attachment repair: step '+str(len(receipt['history'])),min(.9,.3+len(receipt['history'])*.025))
        if nonlinear_restore:
            from .source_nonlinear_restoration import reconstruct
            from .surface_alignment import redistribution_objective
            _,_,_,vertex_field,_=following._derivatives(reference,vertex_field=True)
            distribution=redistribution_objective(reference,tri,vertex_field,
                (following.before_faces>=0)&(np.linalg.norm(vertex_field,axis=1)>1e-12),parameter_reference,free.astype(float),strength=1.)
            # Inherited caps can erase the original planning reserve. Search
            # inside those same caps so every stored canonical seam copy has
            # room for native rounding, not only its physical-family mean.
            projector.source_radius=np.minimum(projector.source_radius,np.maximum(offset-guard,offset*.5))
            projector.contract['source_radius']=fingerprint(projector.source_radius)
            fitted,numerical=reconstruct(reference,tri,free,normals.copy(),np.maximum(area*.1,span**2*1e-12),lap,following,projector,
                progress=lambda label,value:update(label,.3+.6*value),cancelled=cancelled,acceptance_angles=public_angles,reverse_projection=reverse,alignment_reference=parameter_reference,surface_interior=interior,initial_proposal=goal,point_distribution=distribution,source_tube=bool(options.get('_repair_source_tube',True)))
            normal_objective['proposal_method']=numerical['method']
        else:fitted,numerical=solve(reference,reference,tri,projection,free,normals.copy(),np.maximum(area*.1,span**2*1e-12),weights*.01,lap,
            iterations=20,time_limit=60,checkpoint=checkpoint,prevent_new_contacts=True,source_projector=projector,movement_boxes=limits,contact_margin=span*1e-6,
            differential_target=lap@goal,differential_weight=np.full(n,.05),joint_projection='conic',preserve_orientations=True,compact_proposal=True,conic_segment_backtracking=True,conic_elastic_model=True,strict_generated_clearance=bool(options.get('_repair_strict_clearance',False)),localize_contact_stalls=not bool(options.get('_repair_strict_clearance',False)),tangent_objective=tangent,source_angles=angles,angular_secant_model=angles is not None,reverse_coverage=coverage,following_source_field=following,hard_untangle=untangle)
        storage_body=None;storage_prediction=None;storage_report=None
        if following is not None and 'native_storage_coords' in data:
            from .native_storage import from_snapshot
            from .native_storage_repair import repair as repair_storage
            from .source_budget import storage_constraints
            storage_field=FollowingField(source,source_tri,tri,source_field_faces,ids,query,public_angles,normal_scope,faces,reference,branch_caps)
            storage_body,storage_prediction,storage_report=repair_storage(from_snapshot(data),raw+(fitted-reference)[weld],weld,data['triangles'],tri,free,storage_field,public_angles,cancelled=cancelled,time_limit=30.,source_caps=offset,artist_limits=public_limits,artist_reference=reference,persistent_source=storage_constraints(data,ids,weld))
            if storage_body is not None:fitted=physical(storage_prediction)
        update('Checking the complete reconstructed surface',.94)
        measured=measure_complete(fitted)
        def distances_improve(value):return all(value[d+'_after']['rms_m']<value[d+'_before']['rms_m'] and value[d+'_after']['max_m']<=value[d+'_before']['max_m']+span*5e-7 for d in ('forward','reverse','interior'))
        def alignment_improves(value):return value['after_energy']<value['before_energy']-1e-10 and value['new_back_faces']==0 and value['new_severe_faces']==0
        accepted=distances_improve(measured)
        after_field_faces=source_field_faces.copy();after_field_faces[ids]=query(fitted[ids])['source_triangle_ids']
        alignment=compare(reference,fitted,tri,supported,source,source_tri,source_field_faces,after_field_faces,parameter_reference=parameter_reference)
        accepted&=bool(alignment['new_back_faces']==0 and alignment['new_severe_faces']==0 and alignment['after_energy']<=alignment['before_energy']+1e-10)
        if untangle is not None:accepted&=bool(alignment['after_back_faces']<alignment['before_back_faces'])
        storage_ok=storage_report is None or bool(storage_report.get('accepted'))
        accepted&=storage_ok
        endpoint_fraction=1.;endpoint_attempts=[]
        if strategy=='ALIGNMENT':
            accepted&=alignment_improves(alignment)
            if not accepted and storage_ok:
                from .surface_contacts import crossings
                tolerance=span*1e-6;allowed=None
                before_signed=np.einsum('ij,ij->i',cross,normals)
                for fraction in (.5,.25,.125,.0625):
                    check();candidate=reference+(fitted-reference)*fraction
                    attempt=dict(fraction=fraction,accepted=False);endpoint_attempts.append(attempt)
                    # Chords between valid points can leave a curved source.
                    # Retract the shortened proposal onto original facets
                    # inside the same immutable source/artist envelope.
                    chord=candidate.copy()
                    try:candidate=projector.retract(candidate)
                    except ValueError as exc:attempt.update(reason='SOURCE_RETRACTION',diagnostic=str(exc));continue
                    attempt['source_retraction_m']=float(np.linalg.norm(candidate[ids]-chord[ids],axis=1).max(initial=0))
                    record=query(candidate[ids])
                    if np.any(np.abs(candidate[ids]-reference[ids])>limits[ids]+1e-12):attempt['reason']='ARTIST_BOUNDS';continue
                    if np.any(record['residual']>offset-guard+1e-12):attempt['reason']='SOURCE_ENVELOPE';continue
                    candidate_angles=angles;candidate_normals=normals
                    if following is not None:
                        try:candidate_angles,transport_normals=following(candidate)
                        except ValueError as exc:attempt.update(reason='SOURCE_FIELD_SUPPORT',diagnostic=str(exc));continue
                        candidate_normals=normals.copy();candidate_normals[following.scope]=transport_normals[following.scope]
                    cp=candidate[tri];signed=np.einsum('ij,ij->i',np.cross(cp[:,1]-cp[:,0],cp[:,2]-cp[:,0]),candidate_normals)
                    if np.any((before_signed>0)&(signed<=0)):attempt['reason']='FACE_ORIENTATION';continue
                    if angles is not None:
                        from .source_angle_contract import angular_values
                        cosine,violation=angular_values(candidate,tri,candidate_angles)
                        if violation.max(initial=0)>1e-9 or np.maximum(candidate_angles['minimum_cosine']-cosine,0.).max(initial=0)>1e-7:attempt['reason']='ORIGINAL_SOURCE_ANGLES';continue
                    af=source_field_faces.copy();af[ids]=record['source_triangle_ids']
                    candidate_alignment=compare(reference,candidate,tri,supported,source,source_tri,source_field_faces,af,parameter_reference=parameter_reference)
                    if untangle is not None and candidate_alignment['after_back_faces']>=candidate_alignment['before_back_faces']:attempt['reason']='SOURCE_UNTANGLE';continue
                    if not alignment_improves(candidate_alignment):attempt['reason']='SOURCE_ALIGNMENT';continue
                    candidate_measurement=measure_complete(candidate)
                    if not distances_improve(candidate_measurement):attempt['reason']='SURFACE_COVERAGE';continue
                    if allowed is None:allowed=crossings(reference,tri,pairs=True,tolerance=tolerance)
                    contacts=crossings(candidate,tri,pairs=True,tolerance=tolerance)
                    if contacts-allowed:attempt['reason']='NEW_CONTACTS';continue
                    fitted=candidate;alignment=candidate_alignment;measured=candidate_measurement;accepted=True;endpoint_fraction=fraction;attempt['accepted']=True
                    numerical['alignment_endpoint_fraction']=fraction;numerical['chosen_crossings']=len(contacts);numerical['endpoint_scope']='CONIC_FULL_STEP_DIAGNOSTICS_WITH_VERIFIED_SOURCE_RETRACTED_BACKTRACK'
                    break
        measured_faces=measured.pop('measured_source_face_ids');body=(storage_body if storage_body is not None and endpoint_fraction==1. else raw+(fitted-reference)[weld]) if accepted else raw.copy()
        chart_arrays={key:chart.pop(key) for key in ('source_face_ids','source_barycentric','target_vertex_ids','interior')}
        report=dict(solver='INTRINSIC_ATTACHMENT_RECONSTRUCTION_V1',accepted=bool(accepted),review_required=True,regions=regions,selected_region=regions[region],chart=chart,repair_target=strategy,fit_model=fit_model,requested_fit_model=options.get('_repair_fit_model','SOURCE_NORMALS'),source_alignment=alignment,source_defects=defects,alignment_endpoint_fraction=endpoint_fraction,alignment_endpoint_mode='SOURCE_FACET_RETRACTION',alignment_endpoint_attempts=endpoint_attempts,
            before=dict(forward_rms=measured['forward_before']['rms_m'],reverse_rms=measured['reverse_before']['rms_m'],crossings=numerical['initial']['crossings']),
            after=dict(forward_rms=measured['forward_after']['rms_m'],reverse_rms=measured['reverse_after']['rms_m'],crossings=numerical['chosen_crossings']),surface_measurement=measured,reverse_projection=reverse['report'],reverse_coverage=coverage['report'],source_normal_objective=normal_objective,native_storage=storage_report,numerical=numerical,elapsed_seconds=time.monotonic()-started)
        checks={}
        for direction in ('forward','reverse','interior'):
            checks[direction+'_rms_improves']=measured[direction+'_after']['rms_m']<measured[direction+'_before']['rms_m']
            checks[direction+'_maximum_preserved']=measured[direction+'_after']['max_m']<=measured[direction+'_before']['max_m']+span*5e-7
        checks.update(no_new_backward_faces=alignment['new_back_faces']==0,
            no_new_severe_faces=alignment['new_severe_faces']==0,
            supported_alignment_preserved=alignment['after_energy']<=alignment['before_energy']+1e-10)
        if strategy=='ALIGNMENT':checks['supported_alignment_improves']=alignment['after_energy']<alignment['before_energy']-1e-10
        if untangle is not None:checks['backward_face_count_reduced']=alignment['after_back_faces']<alignment['before_back_faces']
        if storage_report is not None:checks['native_storage_verified']=storage_ok
        if bool(accepted)!=all(checks.values()):raise AssertionError('Attachment acceptance report disagrees with its measured endpoint')
        report['acceptance_checks']={key:bool(value) for key,value in checks.items()}
        report['rejection_gates']=[key for key,value in checks.items() if not value]
        if not accepted:
            labels=dict(forward_rms_improves='target-to-source RMS did not improve',reverse_rms_improves='source-to-target RMS did not improve',
                forward_maximum_preserved='target-to-source maximum gap increased',reverse_maximum_preserved='source-to-target maximum gap increased',
                interior_rms_improves='triangle-interior source RMS did not improve',interior_maximum_preserved='triangle-interior maximum source gap increased',
                no_new_backward_faces='new backward-facing faces appeared',no_new_severe_faces='new severe source-alignment defects appeared',supported_alignment_preserved='supported alignment worsened',
                supported_alignment_improves='supported alignment did not improve',backward_face_count_reduced='no existing backward-facing face was removed',
                native_storage_verified='predicted native storage did not pass source support, angle, or contact checks')
            failures=(storage_report or {}).get('final_failures',{})
            failure_names={'unsupported_faces':'source-observation face','angular_faces':'source-angle constraint',
                'flat_quality_faces':'flat-source quality constraint','source_bound_canonical_vertices':'source-distance bound','persistent_source_bound_canonical_vertices':'inherited source-distance bound','artist_bound_canonical_vertices':'movement bound','new_contacts':'new intersection'}
            details=[str(int(failures[key]))+' '+name+('s' if failures[key]!=1 else '') for key,name in failure_names.items() if failures.get(key,0)]
            if details:labels['native_storage_verified']='Blender rounding failed '+', '.join(details)
            report['reason']='Repair retained the original: '+'; '.join(labels[key] for key in report['rejection_gates'])+'.'
        proof={}
        if accepted:
            record=query(fitted[ids]);audit_witnesses(record,source,source_tri)
            proof=dict(weld=weld,reference=reference,planned=body,selected=ids,limits=public_limits,maximum_offset=offset,triangles=tri,normals=normals,before=raw,mass=mass,
                source_face_ids=faces,measured_source_faces=measured_faces,patch_ids=np.zeros(len(ids),dtype=np.int64),policy_faces_0=faces,**{'witness_'+key:value for key,value in record.items()},**{'chart_'+key:value for key,value in chart_arrays.items()})
            proof.update(alignment_before_faces=source_field_faces,alignment_eligible=supported,alignment_parameter_reference=parameter_reference,alignment_required=np.asarray(strategy=='ALIGNMENT'),alignment_non_regression=np.asarray(True))
            proof['alignment_severe_non_regression']=np.asarray(True)
            proof['interior_source_required']=np.asarray(True)
            if angles is not None:
                proof['attachment_source_angles']=np.asarray(True)
                proof['attachment_source_planes']=np.asarray(True)
            if following is not None:proof['attachment_following_source']=np.asarray(True)
            if untangle is not None:proof['alignment_back_reduction_required']=np.asarray(True)
            if storage_body is not None and endpoint_fraction==1.:proof['native_storage_prediction']=storage_prediction
            if companion:proof.update(companion=data['companion'],companion_triangles=data['companion_triangles'])
        update('Ready for native reconstruction checks',1.)
        return body,[report],proof
    finally:trees.clear()
