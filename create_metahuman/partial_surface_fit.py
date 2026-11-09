"""Source-observed fitting with bounded displacement continuation across gaps.

Missing observations are inferred deformation, never measured source detail.
Only the normal external worker imports this optional numerical backend.
"""
import time
import numpy as np


def fit(data,options,*,progress=None,cancelled=None):
    """Escalate tangent regularization only when endpoint quality needs it."""
    if options.get('curvature_polish') and not (options.get('complete_missing') and options.get('rebuild_missing')):raise ValueError('Surface refinement requires partial-surface missing-shape reconstruction')
    if not options.get('rebuild_missing',False):return _fit_candidate(data,options,progress=progress,cancelled=cancelled)
    started=time.monotonic();best=None;best_score=None;attempts=[]
    strengths=(2.,) if options.get('curvature_polish',False) else (8.,) if options.get('register_attachment',False) else (2.,8.)
    for index,strength in enumerate(strengths):
        if cancelled and cancelled():raise RuntimeError('Partial-surface fitting cancelled')
        if index and time.monotonic()-started>115:break
        def update(label,value):
            if progress:progress('Shape candidate '+str(index+1)+': '+label,(index+value)/len(strengths))
        result=_fit_candidate(data,dict(options,_tangent_strength=strength,_candidate_time_limit=120. if options.get('register_attachment',False) else 70.),progress=update,cancelled=cancelled)
        report=result[1][0];alignment=report.get('source_alignment',{})
        quality=bool(alignment) and alignment['new_back_faces']==0 and alignment['after_energy']<=alignment['before_energy']+1e-10 and alignment['after_severe_reference_area_m2']<=alignment['before_severe_reference_area_m2']+1e-12
        attempts.append(dict(tangent_strength=strength,accepted=report['accepted'],endpoint_alignment_quality=quality,source_alignment=alignment,
            before=report.get('before'),after=report.get('after'),reason=report.get('reason')))
        score=(not(report['accepted'] and quality),not report['accepted'],alignment.get('new_back_faces',float('inf')),alignment.get('after_severe_reference_area_m2',float('inf')),alignment.get('after_energy',float('inf')))
        if best is None or score<best_score:best=result;best_score=score
        if report['accepted'] and quality:break
    best[1][0]['tangent_candidate_search']=attempts;best[1][0]['candidate_search_seconds']=time.monotonic()-started
    if progress:progress('Ready for native reconstruction checks',1.)
    return best


def _fit_candidate(data,options,*,progress=None,cancelled=None):
    import igl
    from scipy import sparse
    from .surface_oracle import SurfaceOracle
    from .surface_support import observation_scale,reach_confidence,reciprocal_support,footprint_support
    from .surface_fairing import screened_displacement,rebuild_unobserved_shape
    from .source_envelope import SourceFacetBoxes,source_reachable_boxes,unobserved_motion_envelope
    from .surface_fit_measurement import measure_surface_fit
    from .constrained_surface_fit import solve
    from .vertex_contract import witnesses,audit_witnesses,fingerprint
    from .surface_defects import analyze,compare
    started=time.monotonic()
    def check():
        if cancelled and cancelled():raise RuntimeError('Partial-surface fitting cancelled')
        if time.monotonic()-started>(300 if options.get('register_attachment',False) else 210):raise RuntimeError('Partial-surface fitting exceeded its time budget')
    def update(label,value):
        check()
        if progress:progress(label,value)
    raw=np.asarray(data['current'],dtype=float);weld=np.asarray(data['weld'],dtype=np.int64);count=np.bincount(weld)
    if raw.ndim!=2 or raw.shape[1]!=3 or weld.shape!=(len(raw),) or np.any(count==0) or not np.isfinite(raw).all():raise ValueError('Invalid partial-surface snapshot')
    reference=np.column_stack([np.bincount(weld,weights=raw[:,axis])/count for axis in range(3)]);n=len(reference)
    rest=np.asarray(data.get('parameter_reference',raw),dtype=float)
    if rest.shape!=raw.shape or not np.isfinite(rest).all():raise ValueError('Invalid canonical normal-field reference')
    parameter_reference=np.column_stack([np.bincount(weld,weights=rest[:,axis])/count for axis in range(3)])
    tri=weld[np.asarray(data['triangles'],dtype=np.int64)];tri=tri[np.all(np.diff(np.sort(tri,axis=1),axis=1)!=0,axis=1)]
    source=np.asarray(data['source'],dtype=float);st=np.asarray(data['source_triangles'],dtype=np.int64);span=float(np.linalg.norm(np.ptp(reference,axis=0)));guard=max(span*2.5e-7,1e-10)
    regions=options['regions'];influence=np.zeros(n);assignment=np.full(n,-1,int)
    for i,_ in enumerate(regions):
        values=np.asarray(data['influence_'+str(i)],dtype=float)
        if values.shape!=(n,) or not np.isfinite(values).all() or np.any((values<0)|(values>1)):raise ValueError('Invalid partial-surface region weights')
        use=values>influence;influence[use]=values[use];assignment[use]=i
    p=reference[tri];cross=np.cross(p[:,1]-p[:,0],p[:,2]-p[:,0]);area=np.linalg.norm(cross,axis=1);mass=np.zeros(n);vn=np.zeros_like(reference)
    for corner in range(3):np.add.at(mass,tri[:,corner],area/6);np.add.at(vn,tri[:,corner],cross)
    normals=cross/np.maximum(area[:,None],1e-30)
    edges=np.unique(np.sort(np.vstack((tri[:,[0,1]],tri[:,[1,2]],tri[:,[2,0]])),axis=1),axis=0)
    adj=sparse.coo_matrix((np.ones(2*len(edges)),(np.r_[edges[:,0],edges[:,1]],np.r_[edges[:,1],edges[:,0]])),shape=(n,n)).tocsr();degree=np.maximum(np.asarray(adj.sum(1)).ravel(),1)
    lap=sparse.eye(n)-sparse.diags(1/degree)@adj
    for _ in range(3):vn=(vn+adj@vn/degree[:,None])/2
    free=(influence>1e-4)&(mass>0);selected=np.flatnonzero(free)
    if not len(selected):raise ValueError('This partial-surface region is empty or fully pinned')
    scale=observation_scale(reference,tri);targets=reference.copy();confidence=np.zeros(n);geometric_observed=np.zeros(n,bool);observed_faces=np.full(n,-1,int);normal_faces=np.full(n,-1,int);trees=[];policies={};reciprocal_reports=[]
    try:
        update('Finding reliable original-source observations',.03)
        for i,_ in enumerate(regions):
            check();ids=selected[assignment[selected]==i];faces=np.asarray(data['source_faces_'+str(i)],dtype=np.int64);policies[i]=faces
            tree=igl.AABB();tree.init(source,st[faces]);trees.append((tree,faces))
            if not len(ids):continue
            oracle=SurfaceOracle(source,st,faces)
            # Position rays can miss on an inherited fold. A separately
            # supported nearest-source normal still protects its neighbors.
            _,near,_=tree.squared_distance(source,st[faces],reference[ids]);near_record=witnesses(reference[ids],source,st,faces[near]);audit_witnesses(near_record,source,st)
            residual=near_record['residual']
            normal_support=footprint_support(oracle,near_record['surface_points'],residual,scale[ids],cancelled=cancelled)&(reach_confidence(residual,scale[ids],rings=float(options.get('observation_rings',2.)))>.01)
            normal_faces[ids[normal_support]]=near_record['source_triangle_ids'][normal_support]
            record=oracle.ray_query(reference[ids],vn[ids],normal_limit=.1,max_distance=span*float(options.get('search_distance',.15)),cancelled=cancelled)
            audit_witnesses(record,source,st);support=reach_confidence(record['residual'],scale[ids],rings=float(options.get('observation_rings',2.)))
            observed=record['observed'].copy();valid=np.flatnonzero(observed)
            if len(valid):
                residual=record['residual'][valid]
                clearance=oracle.boundary_clearance(record['surface_points'][valid],cutoff=max(float(residual.max())*.5,span*5e-6),cancelled=cancelled)
                observed[valid]&=(residual<=span*5e-6)|(clearance>=residual*.5)
            quality=record['confidence']*support;observed&=quality>.25
            if options.get('rebuild_missing',False):
                valid=np.flatnonzero(observed)
                mutual,reciprocal=reciprocal_support(reference,tri,ids[valid],record['surface_points'][valid])
                observed[valid]&=mutual;reciprocal['region']=regions[i];reciprocal_reports.append(reciprocal)
            good=ids[observed]
            targets[good]=record['surface_points'][observed];confidence[good]=quality[observed];observed_faces[good]=record['source_triangle_ids'][observed]
            geometric_observed[good]=True
            # A valid source-distance constraint is not automatically a good
            # point correspondence. Keep its geometric authority even when
            # insufficient surrounding surface makes its XYZ target unsafe.
            footprint=footprint_support(oracle,record['surface_points'][observed],record['residual'][observed],scale[good],cancelled=cancelled)
            confidence[good[~footprint]]=0.
            inherited=data.get('inherited_source_caps_'+str(i))
            if inherited is not None:
                retained=ids[(np.asarray(inherited)[ids]>0)&~geometric_observed[ids]]
                if len(retained):
                    _,near,_=tree.squared_distance(source,st[faces],reference[retained]);record=witnesses(reference[retained],source,st,faces[near]);audit_witnesses(record,source,st)
                    targets[retained]=record['surface_points'];observed_faces[retained]=record['source_triangle_ids'];geometric_observed[retained]=True
        attachment=None;anchor_ids=np.empty(0,dtype=np.int64);polish=bool(options.get('curvature_polish',False));normal_quality_ready=False
        polish_model=options.get('curvature_model','CURVATURE')
        if polish_model not in {'CURVATURE','NORMAL_FLOW'}:raise ValueError('Choose CURVATURE or NORMAL_FLOW surface refinement')
        if options.get('register_attachment',False) and polish:
            from .attachment_alignment import propose
            existing=propose(reference,tri,source,st,np.stack([data['influence_'+str(i)] for i in range(2)]),[policies[i] for i in range(2)],data['attachment_frame'])
            binding=existing['controls']['target'];anchor_ids=np.unique(tri[binding['triangle_ids']][binding['barycentric']>1e-8])
        if options.get('register_attachment',False) and not polish:
            from .attachment_alignment import propose,coherent_guide
            if len(regions)!=2 or 'attachment_frame' not in data:raise ValueError('Attachment registration requires two paired regions and a reference rig frame')
            attachment=propose(reference,tri,source,st,np.stack([data['influence_'+str(i)] for i in range(2)]),[policies[i] for i in range(2)],data['attachment_frame'])
            binding=attachment['controls']['target'];anchor_ids=np.unique(tri[binding['triangle_ids']][binding['barycentric']>1e-8])
            guide,guide_report=coherent_guide(reference,tri,influence,attachment)
            anchor_records=[]
            retargeted=0
            for i,(tree,faces) in enumerate(trees):
                ids=selected[(assignment[selected]==i)&((confidence[selected]>0)|np.isin(selected,anchor_ids))]
                if not len(ids):continue
                _,local,_=tree.squared_distance(source,st[faces],guide[ids]);record=witnesses(guide[ids],source,st,faces[local]);audit_witnesses(record,source,st)
                anchor=np.isin(ids,anchor_ids);good=(record['residual']<=np.maximum(scale[ids]*2,guard))|anchor
                valid=np.flatnonzero(good)
                if len(valid):
                    oracle=SurfaceOracle(source,st,faces)
                    good[valid]&=footprint_support(oracle,record['surface_points'][valid],record['residual'][valid],scale[ids[valid]],cancelled=cancelled)|anchor[valid]
                targets[ids[good]]=record['surface_points'][good];observed_faces[ids[good]]=record['source_triangle_ids'][good];confidence[ids[anchor]]=1.
                geometric_observed[ids[anchor]]=True
                retargeted+=int(good.sum())
                anchor_records.extend(dict(physical_vertex=int(vertex),source_triangle_id=int(face),barycentric=bary.tolist(),point=point.tolist()) for vertex,face,bary,point in zip(ids[anchor],record['source_triangle_ids'][anchor],record['barycentric'][anchor],record['surface_points'][anchor]))
            attachment['report'].update(anchor_physical_vertices=anchor_ids.tolist(),observation_policy='COARSE_SECTION_REGISTRATION_NOT_LOCAL_RAY_SUPPORT',artist_budget_policy='ORIGINAL_SOURCE_ANCHOR_REACHABILITY',normal_reference_policy='ORIGINAL_NEAREST_SUPPORTED_SOURCE')
            attachment['report']['source_anchor_witnesses']=anchor_records
            attachment['report'].update(coherent_guide=guide_report,retargeted_source_correspondences=retargeted,original_geometric_observation_scope_retained=True)
        observed=geometric_observed;observation_ids=np.flatnonzero(observed);map_report=None
        if not len(observation_ids):return raw.copy(),[dict(solver='PARTIAL_SOURCE_DISPLACEMENT_CONTINUATION_V1',accepted=False,review_required=True,regions=regions,reason='No reliable original-source observations support this patch.')],{}
        if options.get('rebuild_missing',False):
            from .surface_correspondence import consistent_observations
            keep,map_report=consistent_observations(targets,tri,confidence,source,st,observed_faces,parameter_reference)
            confidence[~keep]=0
        point_observed=confidence>0
        if not point_observed.any():return raw.copy(),[dict(solver='PARTIAL_SOURCE_SHAPE_COMPLETION_V1',accepted=False,review_required=True,regions=regions,reason='No consistent point correspondences support reconstruction.',observation_map=map_report)],{}
        update('Continuing deformation through missing observations',.15)
        boundary=reference+(targets-reference)*influence[:,None] if options.get('rebuild_missing',False) else targets
        goal,continuation=screened_displacement(reference,boundary,tri,confidence,~free)
        if attachment is not None:continuation['attachment_registration']=attachment['report']
        if options.get('rebuild_missing',False):
            # The observation data term, inferred boundary and differential
            # field must agree on the same original-source positions.
            goal[point_observed]=boundary[point_observed]
            goal,shape_report=rebuild_unobserved_shape(goal,tri,point_observed,~free,parameter_reference)
            unknown_position=free&~point_observed
            goal[unknown_position]=reference[unknown_position]+(goal[unknown_position]-reference[unknown_position])*influence[unknown_position,None]
            continuation['shape_reconstruction']=shape_report
            continuation['reciprocal_observations']=reciprocal_reports
            continuation['reconstruction_boundary']='ORIGINAL_SOURCE_WITH_ARTIST_INFLUENCE'
            continuation['proposal_boundary_error_m']=float(np.linalg.norm(goal[point_observed]-boundary[point_observed],axis=1).max(initial=0))
            full=point_observed&(influence>=1.-1e-12)
            continuation['proposal_full_strength_source_error_m']=float(np.linalg.norm(goal[full]-targets[full],axis=1).max(initial=0))
            continuation['observation_map']=map_report
        # Bounds are derived only from certified original-source anchors.
        # Inferred goals are deliberately not called source witnesses.
        public_limits=source_reachable_boxes(reference,observation_ids,targets[observed],free,edges,baseline=span*.01,margin=span*.001)*influence[:,None]
        if options.get('rebuild_missing',False):
            public_limits,envelope=unobserved_motion_envelope(reference,parameter_reference,edges,free,observed,influence,public_limits)
            continuation['motion_envelope']=envelope
        limits=np.maximum(public_limits-guard,public_limits*.5)
        def query(points):
            check();face_ids=np.full(len(points),-1,dtype=np.int64)
            for i,(tree,faces) in enumerate(trees):
                mask=assignment[observation_ids]==i
                if mask.any():_,local,_=tree.squared_distance(source,st[faces],points[mask]);face_ids[mask]=faces[local]
            return witnesses(points,source,st,face_ids)
        initial=query(reference[observation_ids]);source_contract=audit_witnesses(initial,source,st)
        offset=np.maximum(initial['residual'],span*5e-6)+guard*2
        from .source_budget import constrain
        offset,source_radius=constrain(data,observation_ids,assignment[observation_ids],offset,offset-guard)
        projector=SourceFacetBoxes(reference,limits,observation_ids,query,source_radius=source_radius,query_contract=dict(vertices=fingerprint(source),triangles=fingerprint(st)),source_vertices=source,source_triangles=st)
        companion=dict(companion_vertices=data['companion'],companion_triangles=data['companion_triangles']) if 'companion' in data else {}
        source_ids=np.unique(np.concatenate(list(policies.values())))
        observed_weight=mass*confidence*2;completion_weight=mass*(free&~point_observed)*(0. if polish else .5)
        if options.get('rebuild_missing',False):observed_weight*=influence
        weights=observed_weight+completion_weight
        data_goal=goal.copy()
        if options.get('rebuild_missing',False):
            # Reconstruction may replace missing shape, but must not replace
            # actual source observations with its smoothed inference.
            data_goal[point_observed]=targets[point_observed]
            continuation['observed_data_target']='ORIGINAL_SOURCE_WITNESSES'
        projection=dict(matrix=sparse.diags(weights),load=weights[:,None]*data_goal,mass=weights,constant=float(np.sum(weights[:,None]*data_goal**2)))
        if attachment is not None:
            binding=attachment['controls']['target'];count=len(binding['points'])
            row=sparse.coo_matrix((binding['barycentric'].ravel(),(np.repeat(np.arange(count),3),tri[binding['triangle_ids']].ravel())),shape=(count,n)).tocsr()
            strength=float(mass[free].sum())*.1/count;destination=attachment['controls']['source']['points']
            projection=dict(matrix=projection['matrix']+strength*(row.T@row),load=projection['load']+strength*(row.T@destination),mass=projection['mass']+strength*np.asarray(row.T@np.ones(count)).ravel(),constant=projection['constant']+strength*float(np.sum(destination**2)))
            continuation['attachment_registration'].update(landmark_area_weight=strength,landmark_count=count,total_landmark_area_weight=strength*count)
            # A point landmark alone can concentrate motion into a pucker.
            # Distribute the coarse hypothesis over canonical surface area;
            # it remains a soft inference term, not measured source evidence.
            p=parameter_reference[tri];canonical_area=np.linalg.norm(np.cross(p[:,1]-p[:,0],p[:,2]-p[:,0]),axis=1)*.5
            canonical_mass=np.zeros(n)
            for corner in range(3):np.add.at(canonical_mass,tri[:,corner],canonical_area/3)
            guide_weight=canonical_mass*influence*free
            if guide_weight.sum()<=0:raise ValueError('Attachment guide requires nonzero movable canonical area')
            guide_weight*=4*float(mass[free].sum())/guide_weight.sum()
            projection=dict(matrix=projection['matrix']+sparse.diags(guide_weight),load=projection['load']+guide_weight[:,None]*guide,
                mass=projection['mass']+guide_weight,constant=projection['constant']+float(np.sum(guide_weight[:,None]*guide**2)))
            continuation['attachment_registration']['distributed_guide_objective']=dict(method='CANONICAL_AREA_COARSE_GUIDE_V1',
                vertices=int(np.count_nonzero(guide_weight)),total_area_weight=float(guide_weight.sum()),
                guide_is_not_source_evidence=True,source_and_artist_constraints_unchanged=True)
        if options.get('rebuild_missing',False):
            from .surface_fit_measurement import assemble_reverse_projection
            source_point_projection=projection
            reverse=assemble_reverse_projection(reference,tri,source,st,free,source_ids,observation_mask=observed,patch_ids=assignment,source_policies=policies,allow_inferred_surface=polish,coverage_metric=parameter_reference if polish and polish_model=='NORMAL_FLOW' else None,**companion)
            projection=dict(matrix=projection['matrix']+2*reverse['matrix'],load=projection['load']+2*reverse['load'],mass=projection['mass']+2*reverse['mass'],constant=projection['constant']+2*reverse['constant'])
            continuation['source_reverse_objective']=reverse['report']
        normal_eligible=(normal_faces>=0)&free;normal_ids=np.flatnonzero(normal_eligible)
        field=np.zeros_like(reference);facets=source[st[normal_faces[normal_eligible]]];sn=np.cross(facets[:,1]-facets[:,0],facets[:,2]-facets[:,0]);sn/=np.maximum(np.linalg.norm(sn,axis=1)[:,None],1e-30);field[normal_eligible]=sn
        complete=np.all(normal_eligible[tri],axis=1);directions=field[tri[complete]].sum(1);size=np.linalg.norm(directions,axis=1);valid=size>=2.4;normals[np.flatnonzero(complete)[valid]]=directions[valid]/size[valid,None]
        tangent=None;angles=None
        supported=np.zeros(len(tri),bool);supported[np.flatnonzero(complete)[valid]]=True
        differential_target=np.asarray(lap@goal)
        if options.get('rebuild_missing',False) and not polish:
            from .source_differentials import source_differentials
            curvature_reports=[]
            for index,faces in policies.items():
                ids=normal_ids[assignment[normal_ids]==index]
                if not len(ids):continue
                record=witnesses(reference[ids],source,st,normal_faces[ids]);audit_witnesses(record,source,st)
                differential=source_differentials(reference,tri,source,st,ids,record,faces)
                # This fitter and source_differentials use the same uniform
                # one-ring Laplacian. Replace its supported differential,
                # including a zero tangential centroid imbalance.
                # Retaining the old tangential target can preserve a distorted
                # parameterization even after its normal curvature is repaired.
                # Unsupported completion targets are unchanged.
                blend=differential['confidence']*influence[ids]
                differential_target[ids]+=blend[:,None]*(differential['target']-differential_target[ids])
                curvature_reports.append(dict(region=regions[index],**differential['report']))
            continuation['source_curvature_objective']=dict(method='SUPPORTED_SOURCE_FULL_DIFFERENTIAL_BLEND_V2',
                regions=curvature_reports,unsupported_targets_retained=True,tangential_targets_retained=False,
                supported_tangent_target='ZERO_CENTROID_IMBALANCE',
                source_distance_evidence_unchanged=True)
            from .surface_alignment import tangent_objective,angle_contract,redistribution_objective
            supported=np.zeros(len(tri),bool);supported[np.flatnonzero(complete)[valid]]=True
            tangent=tangent_objective(reference,tri,normals,supported,parameter_reference,strength=float(options.get('_tangent_strength',2.)))
            continuation['source_tangent_objective']=tangent['report']
            redistribution=redistribution_objective(reference,tri,field,normal_eligible,parameter_reference,influence)
            tangent['matrix']=(tangent['matrix']+redistribution['matrix']).tocsr()
            continuation['source_redistribution_objective']=redistribution['report']
            angles=angle_contract(reference,tri,normals,supported,parameter_reference)
            continuation['source_angle_constraints']=dict(method='AFFINE_AREA_NORMAL_CONES_WITH_NONLINEAR_CHECK_V1',faces=len(angles['face_ids']),minimum_source_cosine=.25,reserve_cosine=.5,bounds_policy='FIXED_ORIGINAL_SNAPSHOT',proposal_model='DIRECTIONAL_MIDPOINT_SECANT' if attachment is not None else 'CURRENT_JACOBIAN')
        update('Fitting observed skin and its displacement continuation',.25)
        def checkpoint(points,receipt):update('Partial surface: step '+str(len(receipt['history'])),min(.9,.3+len(receipt['history'])*.035))
        if polish:fitted=reference.copy()
        else:
            fitted,numerical=solve(reference,reference,tri,projection,free,normals,np.maximum(area*.1,span**2*1e-12),mass*.05,lap,iterations=min(int(options.get('iterations',16)),48 if attachment is not None else 20),time_limit=min(120.,float(options.get('_candidate_time_limit',120.))),checkpoint=checkpoint,step_size_fraction=.01 if attachment is not None else .0025,
                prevent_new_contacts=True,source_projector=projector,movement_boxes=limits,contact_margin=span*1e-6,differential_target=differential_target,differential_weight=np.full(n,2.),joint_projection='conic',preserve_orientations=True,compact_proposal=True,conic_segment_backtracking=True,conic_elastic_model=True,localize_contact_stalls=True,tangent_objective=tangent,source_angles=angles,angular_secant_model=attachment is not None,restore_source_angles=attachment is not None)
        if attachment is not None or polish:
            from .surface_completion import complete
            # Correspondences may slide on immutable source facets instead
            # of freezing inherited sample clumps into the reconstructed join.
            pinned=~free;pinned=pinned.copy();pinned[anchor_ids]=True
            mobile=~pinned[observation_ids];mobile_ids=observation_ids[mobile]
            fixed_queries=fitted[observation_ids].copy()
            def mobile_query(points):
                positions=fixed_queries.copy();positions[mobile]=points
                record=query(positions)
                return witnesses(points,source,st,record['source_triangle_ids'][mobile])
            mobile_projector=SourceFacetBoxes(reference,limits,mobile_ids,mobile_query,source_radius=source_radius[mobile],
                query_contract=dict(vertices=fingerprint(source),triangles=fingerprint(st)),source_vertices=source,source_triangles=st)
            from .source_endpoint_restoration import repair
            endpoint_projector=projector
            quality_options=dict(minimum_cosine=.5,reserve_cosine=.6,preserve_aligned=True) if polish and polish_model=='NORMAL_FLOW' else {}
            def endpoint_source_faces(points):
                faces=normal_faces.copy()
                for index,(tree,policy) in enumerate(trees):
                    ids=normal_ids[assignment[normal_ids]==index]
                    if len(ids):_,local,_=tree.squared_distance(source,st[policy],points[ids]);faces[ids]=policy[local]
                if quality_options:
                    from .vertex_contract import resolve_facet_ties
                    faces[normal_ids],_=resolve_facet_ties(points[normal_ids],source,st,normal_faces[normal_ids],faces[normal_ids])
                return faces
            curvature_options={}
            if polish and polish_model=='CURVATURE':
                operator=igl.cotmatrix(parameter_reference,tri).tocsr();targets=np.zeros_like(reference);certainty=np.zeros(n);curvature=[]
                for index,faces in policies.items():
                    ids=normal_ids[assignment[normal_ids]==index]
                    if not len(ids):continue
                    tree,policy=trees[index];_,local,_=tree.squared_distance(source,st[policy],fitted[ids]);record=witnesses(fitted[ids],source,st,policy[local]);audit_witnesses(record,source,st)
                    result=source_differentials(fitted,tri,source,st,ids,record,faces,differential_operator=operator)
                    targets[ids]=result['target'];certainty[ids]=result['confidence']*influence[ids];curvature.append(dict(region=regions[index],**result['report']))
                curvature_options=dict(differential_target=targets,differential_confidence=certainty,positional_objective=projection)
                continuation['source_curvature_completion']=dict(regions=curvature,vertex_observation_scope_unchanged=True,internal_anchor_vertices=anchor_ids.tolist())
            if polish and polish_model=='NORMAL_FLOW':
                from .surface_normal_continuation import refine_with_updates
                fitted,preparation=repair(fitted,tri,free,parameter_reference,normals,normal_eligible,source,st,endpoint_source_faces,projector,
                    anchors=anchor_ids,time_limit=8.,passes=3,cancelled=cancelled,**quality_options)
                continuation['source_normal_preparation']=preparation
                normal_quality_ready=preparation.get('fixed_supported_faces',0)>0 and preparation['after'].get('below_threshold_faces')==0
                prepared_faces=endpoint_source_faces(fitted)
                if normal_quality_ready:
                    facets=source[st[prepared_faces[normal_eligible]]];cross=np.cross(facets[:,1]-facets[:,0],facets[:,2]-facets[:,0]);field[normal_eligible]=cross/np.maximum(np.linalg.norm(cross,axis=1)[:,None],1e-30)
                precision_radius=source_radius.copy();precision_radius[mobile]=np.minimum(precision_radius[mobile],np.maximum(offset[mobile]-guard,offset[mobile]*.5))
                precision_mobile=SourceFacetBoxes(reference,limits,mobile_ids,mobile_query,source_radius=precision_radius[mobile],query_contract=dict(vertices=fingerprint(source),triangles=fingerprint(st)),source_vertices=source,source_triangles=st)
                endpoint_projector=SourceFacetBoxes(reference,limits,observation_ids,query,source_radius=precision_radius,query_contract=dict(vertices=fingerprint(source),triangles=fingerprint(st)),source_vertices=source,source_triangles=st)
                def prepare_quality_endpoint(points):
                    retracted=precision_mobile.retract(points)
                    polished,detail=repair(retracted,tri,free,parameter_reference,normals,normal_eligible,source,st,endpoint_source_faces,endpoint_projector,
                        anchors=anchor_ids,time_limit=2.,passes=3,cancelled=cancelled,**quality_options)
                    detail['source_precision_retraction_m']=float(np.linalg.norm(retracted-points,axis=1).max(initial=0));detail['public_source_limits_unchanged']=True
                    return polished,detail
                continuation['native_precision_envelope']=dict(guard_m=guard,moving_source_vertices=len(mobile_ids),source_radius=fingerprint(precision_radius),public_source_limits_unchanged=True)
                frozen_source_faces=reverse['source_face_ids'].copy()
                baseline_measurement=measure_surface_fit(reference,reference,tri,source,st,free,source_ids,forward_vertices=observed,reverse_source_face_ids=frozen_source_faces,**companion)
                last_normal_faces=prepared_faces.copy() if normal_quality_ready else normal_faces.copy();last_reverse_faces=reverse['target_triangle_ids'].copy()
                def source_normal_faces(points):
                    return endpoint_source_faces(points)
                def refresh_objective(points):
                    nonlocal last_normal_faces,last_reverse_faces
                    faces=source_normal_faces(points);new_field=field.copy();facets=source[st[faces[normal_eligible]]];cross=np.cross(facets[:,1]-facets[:,0],facets[:,2]-facets[:,0]);new_field[normal_eligible]=cross/np.maximum(np.linalg.norm(cross,axis=1)[:,None],1e-30)
                    current_reverse=assemble_reverse_projection(points,tri,source,st,~pinned,source_ids,observation_mask=observed,patch_ids=assignment,source_policies=policies,allow_inferred_surface=True,reverse_source_face_ids=frozen_source_faces,coverage_metric=parameter_reference,**companion)
                    np.testing.assert_array_equal(current_reverse['source_face_ids'],frozen_source_faces)
                    combined=dict(matrix=source_point_projection['matrix']+2*current_reverse['matrix'],load=source_point_projection['load']+2*current_reverse['load'],mass=source_point_projection['mass']+2*current_reverse['mass'],constant=source_point_projection['constant']+2*current_reverse['constant'])
                    evidence=dict(source_samples=current_reverse['report'],changed_normal_witnesses=int(np.count_nonzero(faces[normal_eligible]!=last_normal_faces[normal_eligible])),changed_reverse_target_faces=int(np.count_nonzero(current_reverse['target_triangle_ids']!=last_reverse_faces)),normal_vertex_scope=fingerprint(normal_ids),normal_witness_facets=fingerprint(faces[normal_ids]),frozen_source_scope=fingerprint(frozen_source_faces),original_point_observations_unchanged=True)
                    last_normal_faces=faces;last_reverse_faces=current_reverse['target_triangle_ids'].copy()
                    return new_field,combined,evidence
                def assess_continuity(points):
                    measured=measure_surface_fit(reference,points,tri,source,st,free,source_ids,forward_vertices=observed,reverse_source_face_ids=frozen_source_faces,**companion)
                    quality=dict(valid=True,score=0.,distances={})
                    for direction in ('forward','reverse'):
                        before=baseline_measurement[direction+'_before'];after=measured[direction+'_after'];quality['distances'][direction]=dict(before=before,after=after)
                        quality['valid']&=after['rms_m']<before['rms_m'] and after['max_m']<=before['max_m']+span*5e-7
                        quality['score']+=(after['rms_m']/max(before['rms_m'],span*1e-9))**2
                    if supported.any():
                        alignment=compare(reference,points,tri,normal_eligible,source,st,normal_faces,source_normal_faces(points),parameter_reference=parameter_reference)
                        quality['source_alignment']=alignment
                        quality['valid']&=alignment['new_back_faces']==0 and alignment['after_energy']<=alignment['before_energy']+1e-10 and alignment['after_severe_reference_area_m2']<=alignment['before_severe_reference_area_m2']+1e-12
                        if normal_quality_ready:quality['valid']&=float(np.cos(np.deg2rad(alignment['after_angle_quality']['maximum_degrees'])))>=.5-1e-7
                        quality['score']+=alignment['after_energy']/max(alignment['before_energy'],1e-12)
                    p=points[tri];signed=np.einsum('ij,ij->i',np.cross(p[:,1]-p[:,0],p[:,2]-p[:,0]),normals)
                    before=reference[tri];original_signed=np.einsum('ij,ij->i',np.cross(before[:,1]-before[:,0],before[:,2]-before[:,0]),normals)
                    quality['original_reference_reversals']=int(np.count_nonzero((original_signed>0)&(signed<=0)));quality['valid']&=quality['original_reference_reversals']==0
                    return quality
                fitted,completion=refine_with_updates(fitted,tri,parameter_reference,field,normal_eligible,pinned,reference,limits,mobile_projector,projection,supported_faces=supported,refresh=refresh_objective,assess=assess_continuity,
                    progress=lambda label,value:update(label,.9+.03*value),cancelled=cancelled,minimum_source_cosine=.5 if normal_quality_ready else .25,prepare_endpoint=prepare_quality_endpoint)
            else:
                fitted,completion=complete(fitted,tri,observed,pinned,parameter_reference,reference,limits,normals,
                    source_vertices=source,source_triangles=st,supported_faces=supported,time_limit=30. if polish else 60.,iterations=16 if polish else 32,source_projector=mobile_projector,**curvature_options,
                    progress=lambda label,value:update(label,.9+.03*value),cancelled=cancelled)
            continuation['inferred_completion']=completion
            if polish:numerical=completion['numerical']
            update('Restoring endpoint source-facing detail',.93)
            fitted,endpoint=repair(fitted,tri,free,parameter_reference,normals,normal_eligible,source,st,endpoint_source_faces,endpoint_projector,
                anchors=anchor_ids,time_limit=8.,passes=3,cancelled=cancelled,**quality_options)
            continuation['source_endpoint_restoration']=endpoint
        if options.get('rebuild_missing',False) and not polish:
            continuation['source_redistribution_objective']['after_energy']=float(fitted.ravel()@(redistribution['matrix']@fitted.ravel()))
        if attachment is not None:
            binding=attachment['target'];point=binding['barycentric']@fitted[tri[binding['triangle_id']]]
            continuation['attachment_registration']['planned_bound_landmark_gap_m']=float(np.linalg.norm(point-attachment['source']['point']))
            after_attachment=propose(fitted,tri,source,st,np.stack([data['influence_'+str(i)] for i in range(2)]),[policies[i] for i in range(2)],data['attachment_frame'])
            continuation['attachment_registration'].update(planned_section_gap_after_m=after_attachment['report']['gap_m'],planned_section_height_gap_after_m=after_attachment['report']['height_gap_m'])
        update('Checking source coverage and inferred movement',.94)
        measured=measure_surface_fit(reference,fitted,tri,source,st,free,source_ids,forward_vertices=observed,**companion)
        before_record=initial;after_record=query(fitted[observation_ids])
        def observed_stats(record):return dict(rms_m=float(np.sqrt(np.average(record['residual']**2,weights=mass[observation_ids]))),max_m=float(record['residual'].max()))
        ob,oa=observed_stats(before_record),observed_stats(after_record)
        normal_summary,_=analyze(reference,tri,normal_eligible,source,st,normal_faces,parameter_reference=parameter_reference,find_clusters=False)
        alignment=None
        if normal_summary['supported_faces']:
            after_faces=normal_faces.copy()
            for i,(tree,faces) in enumerate(trees):
                ids=normal_ids[assignment[normal_ids]==i]
                if len(ids):_,local,_=tree.squared_distance(source,st[faces],fitted[ids]);after_faces[ids]=faces[local]
            alignment=compare(reference,fitted,tri,normal_eligible,source,st,normal_faces,after_faces,parameter_reference=parameter_reference)
        accepted=all(measured[d+'_after']['rms_m']<measured[d+'_before']['rms_m'] and (polish or measured[d+'_after']['max_m']<=measured[d+'_before']['max_m']+span*5e-7) for d in ('forward','reverse'))
        accepted&=oa['rms_m']<ob['rms_m'] and (polish or oa['max_m']<=ob['max_m']+span*5e-7)
        if attachment is not None:accepted&=after_attachment['report']['gap_m']<attachment['report']['gap_m']-guard and abs(after_attachment['report']['height_gap_m'])<=abs(attachment['report']['height_gap_m'])+guard
        measured_faces=measured.pop('measured_source_face_ids');body=raw+(fitted-reference)[weld] if accepted else raw.copy()
        report=dict(solver='PARTIAL_SOURCE_SHAPE_COMPLETION_V1' if options.get('rebuild_missing',False) else 'PARTIAL_SOURCE_DISPLACEMENT_CONTINUATION_V1',completion_mode='REBUILD_SHAPE' if options.get('rebuild_missing',False) else 'PRESERVE_DETAIL',accepted=bool(accepted),review_required=True,regions=regions,source_observation='NORMAL_LINE',source_contract=source_contract,coverage_policy='INDEPENDENT_FIXED_SOURCE_ENDPOINT_GATE',forward_measurement_scope='FIXED_OBSERVED_VERTICES',observed_physical_vertices=len(observation_ids),inferred_physical_vertices=int((free&~observed).sum()),continuation=continuation,
            before=dict(forward_rms=measured['forward_before']['rms_m'],reverse_rms=measured['reverse_before']['rms_m'],crossings=numerical['initial']['crossings']),
            after=dict(forward_rms=measured['forward_after']['rms_m'],reverse_rms=measured['reverse_after']['rms_m'],crossings=numerical['chosen_crossings']),observed_before=ob,observed_after=oa,surface_measurement=measured,numerical=numerical,elapsed_seconds=time.monotonic()-started,
            limitation='Missing-source vertices use inferred shape or displacement, not observed source anatomy.')
        report['source_evidence']=dict(encoding='PACKBITS_LITTLE_HEX',canonical_vertices=len(raw),snapshot=fingerprint(raw),triangles=fingerprint(data['triangles']),
            observed=np.packbits(observed[weld],bitorder='little').tobytes().hex(),inferred=np.packbits((free&~observed)[weld],bitorder='little').tobytes().hex())
        report['source_evidence'].update(point_correspondence=np.packbits(point_observed[weld],bitorder='little').tobytes().hex(),
            source_supported_unanchored=np.packbits((observed&~point_observed)[weld],bitorder='little').tobytes().hex())
        report.update(point_correspondence_physical_vertices=int(point_observed.sum()),source_supported_unanchored_vertices=int((observed&~point_observed).sum()),geometric_observation_scope_preserved=True)
        if polish:report.update(solver='SOURCE_CONSTRAINED_CURVATURE_POLISH_V1',completion_mode='SOURCE_CURVATURE_POLISH',native_maximum_gap_validation_required=True)
        if polish and polish_model=='NORMAL_FLOW':report.update(solver='SOURCE_CONSTRAINED_NORMAL_CONTINUITY_V1',completion_mode='SOURCE_NORMAL_CONTINUITY')
        if alignment is not None:report.update(source_alignment=alignment,source_alignment_policy='NO_NEW_BACK_FACES_AND_NONINCREASING_ENERGY_AND_SEVERE_AREA' if options.get('rebuild_missing',False) else 'NO_NEW_BACK_FACES_AND_NONINCREASING_FIXED_SCOPE_ENERGY')
        if not accepted:report['reason']='No bounded continuation improved observed skin and both complete source distances.'
        proof={}
        if accepted:
            projector.prepare(fitted);record=projector.witness;audit_witnesses(record,source,st)
            proof=dict(weld=weld,reference=reference,planned=body,selected=selected,constrained_ids=observation_ids,forward_ids=observation_ids,limits=public_limits,maximum_offset=offset,triangles=tri,normals=normals,before=raw,mass=mass,source_face_ids=source_ids,measured_source_faces=measured_faces,patch_ids=assignment[observation_ids],observed_ids=observation_ids,observed_before=np.array([ob['rms_m'],ob['max_m']]),**{'witness_'+key:value for key,value in record.items()})
            if polish:proof['internal_pin_ids']=anchor_ids
            for i,faces in policies.items():proof['policy_faces_'+str(i)]=faces
            if alignment is not None:proof.update(alignment_before_faces=normal_faces,alignment_eligible=normal_eligible,alignment_parameter_reference=parameter_reference,alignment_required=np.asarray(False),alignment_non_regression=np.asarray(True),alignment_severe_non_regression=np.asarray(bool(options.get('rebuild_missing',False))),alignment_ids=normal_ids,alignment_patch_ids=assignment[normal_ids])
            if alignment is not None and polish and polish_model=='NORMAL_FLOW' and (normal_quality_ready or endpoint.get('fixed_supported_faces',0)>0 and endpoint['after'].get('below_threshold_faces')==0):proof['alignment_quality_minimum_cosine']=np.asarray(.5)
            if companion:proof.update(companion=data['companion'],companion_triangles=data['companion_triangles'])
            if attachment is not None:proof.update(attachment_frame=data['attachment_frame'],attachment_influences=np.stack([data['influence_'+str(i)] for i in range(2)]),attachment_before_gap=np.asarray(attachment['report']['gap_m']),attachment_before_height_gap=np.asarray(attachment['report']['height_gap_m']))
            if attachment is not None:
                for side in ('source','target'):
                    for key in ('triangle_ids','barycentric','points'):proof['attachment_'+side+'_control_'+key]=attachment['controls'][side][key]
        update('Ready for native partial-surface checks',1.)
        return body,[report],proof
    finally:trees.clear()
