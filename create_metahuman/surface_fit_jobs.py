"""Snapshot, run and validate accurate fitting without blocking the editor."""
import json,os,shutil,subprocess,time,uuid
from pathlib import Path
import numpy as np


def prepare(context,source,target,current,tasks):
    from . import mesh_ops as m
    rest=m.coordinates(target);tolerance=max(float(np.linalg.norm(np.ptp(rest,axis=0)))*1e-6,1e-9)
    _,weld=np.unique(np.rint(rest/tolerance).astype(np.int64),axis=0,return_inverse=True)
    target.data.calc_loop_triangles();source.data.calc_loop_triangles()
    data=dict(current=current,parameter_reference=rest,weld=weld,triangles=np.array([t.vertices[:] for t in target.data.loop_triangles],dtype=np.int64),
        source=m.evaluated_coordinates(context,source),source_triangles=np.array([t.vertices[:] for t in source.data.loop_triangles],dtype=np.int64))
    data['budget_target_matrix']=np.asarray(target.matrix_world,dtype=float)
    for index,task in enumerate(tasks):
        families,scatter=np.unique(weld[task['ids']],return_inverse=True);np.testing.assert_array_equal(scatter,task['scatter'])
        values=np.zeros(int(weld.max())+1);values[families]=task['influence']
        data['influence_'+str(index)]=values;data['source_faces_'+str(index)]=task['source_face_ids']
    from . import source_budget
    data.update(source_budget.snapshot(target.get(source_budget.PROPERTY,''),data))
    s,f=context.scene.cm_settings,context.scene.cm_features
    companion=f.head_result if target==f.body_result else (f.body_result if s.part=='FULL_BODY' else None)
    if companion==target:companion=None
    if companion:
        evaluated=companion.evaluated_get(context.evaluated_depsgraph_get());mesh=evaluated.to_mesh()
        try:
            mesh.calc_loop_triangles();data['companion_triangles']=np.array([t.vertices[:] for t in mesh.loop_triangles],dtype=np.int64)
            data['companion']=m.evaluated_coordinates(context,companion)
        finally:evaluated.to_mesh_clear()
    return data,companion


def _cleanup(folder,parent):
    folder=folder.resolve();parent=parent.resolve()
    if folder.parent!=parent or len(folder.name)!=12 or any(c not in '0123456789abcdef' for c in folder.name):raise ValueError('Invalid owned surface-fit cache path')
    for attempt in range(8):
        try:
            if folder.exists():shutil.rmtree(folder)
            return
        except PermissionError:
            if attempt==7:raise
            time.sleep(.05*(attempt+1))


def run(data,options,python,project,*,cancelled=None,progress=None):
    cache=Path(project).resolve()/'SurfaceFit'/'.jobs';cache.mkdir(parents=True,exist_ok=True)
    folder=cache/uuid.uuid4().hex[:12];folder.mkdir();process=None
    try:
        if cancelled and cancelled():raise RuntimeError('Accurate fit cancelled; previous surface retained')
        np.savez_compressed(folder/'input.npz',**data)
        request=dict(parent=os.getpid(),options=options)
        (folder/'request.json').write_text(json.dumps(request),encoding='utf-8')
        flags=(subprocess.CREATE_NO_WINDOW|subprocess.BELOW_NORMAL_PRIORITY_CLASS) if os.name=='nt' else 0
        with (folder/'worker.log').open('w',encoding='utf-8') as log:
            if cancelled and cancelled():raise RuntimeError('Accurate fit cancelled; previous surface retained')
            process=subprocess.Popen([str(python),str(Path(__file__).with_name('surface_fit_worker.py')),str(folder/'request.json')],
                stdout=log,stderr=subprocess.STDOUT,creationflags=flags)
            started=time.monotonic()
            while process.poll() is None:
                if cancelled and cancelled():raise RuntimeError('Accurate fit cancelled; previous surface retained')
                if time.monotonic()-started>(360 if options.get('register_attachment',False) else 250):raise RuntimeError('Accurate fit exceeded its time budget; previous surface retained')
                if progress and (folder/'progress.json').exists():
                    try:
                        value=json.loads((folder/'progress.json').read_text(encoding='utf-8'));progress(value['label'],value['value'])
                    except (OSError,ValueError):pass
                time.sleep(.15)
        if cancelled and cancelled():raise RuntimeError('Accurate fit cancelled; previous surface retained')
        result_path=folder/'result.json'
        if not result_path.is_file():raise RuntimeError('Accurate-fit worker stopped without a result; previous surface retained')
        result=json.loads(result_path.read_text(encoding='utf-8'))
        if result.get('status')!='complete':raise RuntimeError(result.get('error','Accurate fit failed'))
        if process.returncode!=0:raise RuntimeError('Accurate-fit worker exited unsuccessfully; previous surface retained')
        with np.load(folder/'output.npz',allow_pickle=False) as archive:
            body=archive['body'];proof={key.removeprefix('proof_'):archive[key] for key in archive.files if key.startswith('proof_')}
        if body.shape!=data['current'].shape or not np.isfinite(body).all():raise ValueError('Invalid accurate-fit worker output')
        reports=result['reports']
        if any(report.get('accepted') for report in reports):
            np.testing.assert_array_equal(proof['before'],data['current']);np.testing.assert_array_equal(proof['planned'],body)
            proof.update(source=data['source'],source_triangles=data['source_triangles'])
            if 'source_budget_context' in data:
                proof['source_budget_context']=data['source_budget_context'];proof['source_budget_previous']=data['source_budget_previous']
        # Keep one small receipt. Geometry snapshots are disposable, not
        # accumulating full-character backups in the user's project.
        receipt=cache.parent/'last-fit.json';temporary=receipt.with_suffix('.tmp')
        temporary.write_text(json.dumps(result,indent=2),encoding='utf-8');temporary.replace(receipt)
        return body,reports,proof
    finally:
        if process is not None and process.poll() is None:
            process.terminate()
            try:process.wait(timeout=3)
            except subprocess.TimeoutExpired:process.kill();process.wait(timeout=3)
        _cleanup(folder,cache)


def retract_native(candidate,proof):
    """Retract a shortened proposal inside its original source/artist budgets.

    Curved-source chords need not satisfy the endpoint offset cap. This
    creates a new checked endpoint, never new budgets or source evidence.
    Only observed vertices move during retraction; inferred positions retain
    the proposed shortened deformation. Native validation remains mandatory.
    """
    from .source_envelope import SourceFacetBoxes
    from .vertex_contract import witnesses,fingerprint
    from mathutils import Vector
    from mathutils.bvhtree import BVHTree
    weld=proof['weld'];count=np.bincount(weld);ids=proof.get('constrained_ids',proof['selected'])
    physical=np.column_stack([np.bincount(weld,weights=candidate[:,axis])/count for axis in range(3)])
    projector=proof.get('_native_retractor')
    if projector is None:
        cache=proof.setdefault('_native_source_trees',{})
        def query(points):
            chosen=np.full(len(ids),-1,dtype=np.int64)
            for index in np.unique(proof['patch_ids']):
                mask=proof['patch_ids']==index;faces=proof['policy_faces_'+str(index)]
                if int(index) not in cache:cache[int(index)]=BVHTree.FromPolygons(proof['source'].tolist(),proof['source_triangles'][faces].tolist(),all_triangles=True)
                hits=[cache[int(index)].find_nearest(Vector(point))[2] for point in points[mask]]
                if any(hit is None for hit in hits):raise ValueError('Could not retract the shortened source-surface endpoint')
                nearest=np.asarray(hits,dtype=np.int64)
                chosen[mask]=faces[nearest]
            return witnesses(points,proof['source'],proof['source_triangles'],chosen)
        span=float(np.linalg.norm(np.ptp(proof['before'],axis=0)));guard=max(span*2.5e-7,1e-10)
        limits=np.maximum(proof['limits']-guard,proof['limits']*.5)
        radius=np.maximum(proof['maximum_offset']-guard,proof['maximum_offset']*.5)
        projector=SourceFacetBoxes(proof['reference'],limits,ids,query,source_radius=radius,
            query_contract=dict(vertices=fingerprint(proof['source']),triangles=fingerprint(proof['source_triangles'])),source_vertices=proof['source'],source_triangles=proof['source_triangles'])
        proof['_native_retractor']=projector
    retracted=projector.retract(physical)
    pins=proof.get('internal_pin_ids',np.empty(0,dtype=np.int64))
    retracted[pins]=physical[pins]
    output=proof['before']+(retracted-proof['reference'])[weld]
    return output,dict(method='ORIGINAL_SOURCE_FACET_NATIVE_RETRACTION_V1',source_constrained_physical_vertices=len(ids),
        max_correction_m=float(np.linalg.norm(output-candidate,axis=1).max(initial=0)),budgets_recentered=False,internal_pins_preserved=True)


def validate_native(actual,proof,reports):
    """Repeat source/artist/area bounds inside the native rollback transaction."""
    from .vertex_contract import audit_witnesses,witnesses,closest_triangle_points,fingerprint
    from mathutils import Vector
    from mathutils.bvhtree import BVHTree
    weld=proof['weld'];selected=proof['selected'];count=np.bincount(weld)
    if 'alignment_quality_minimum_cosine' in proof and 'alignment_before_faces' not in proof:raise ValueError('Source-normal quality certification requires a fixed original alignment scope')
    if bool(proof.get('alignment_back_reduction_required',False)) and 'alignment_before_faces' not in proof:raise ValueError('Backward-face reduction requires the original alignment scope')
    if bool(proof.get('attachment_source_angles',False)) and 'alignment_before_faces' not in proof:raise ValueError('Attachment source angles require the original source-normal scope')
    if bool(proof.get('attachment_source_planes',False)) and not bool(proof.get('attachment_source_angles',False)):raise ValueError('Attachment source planes require the original source-angle contract')
    follows=bool(proof.get('attachment_following_source',False));following_scope=None
    if follows:
        if not bool(proof.get('attachment_source_planes',False)):raise ValueError('Source-field transport requires original source area planes')
        from .following_source_field import original_contract,corner_normals
        if '_following_corner_normals' not in proof:proof['_following_corner_normals']=corner_normals(proof['source'],proof['source_triangles'])
        moving=np.zeros(len(proof['reference']),bool);moving[selected]=True
        _,following_scope,_=original_contract(proof['reference'],proof['triangles'],proof['source'],proof['source_triangles'],proof['alignment_before_faces'],proof['alignment_eligible'],moving,proof['alignment_parameter_reference'],smooth_normals=proof['_following_corner_normals'])
    if actual.shape!=proof['planned'].shape or not np.isfinite(actual).all():raise ValueError('Invalid stored accurate-fit surface')
    editable=np.isin(weld,selected);np.testing.assert_array_equal(actual[~editable],proof['before'][~editable])
    pins=proof.get('internal_pin_ids',np.empty(0,dtype=np.int64))
    if pins.ndim!=1 or pins.dtype.kind not in 'iu' or not np.isin(pins,selected).all():raise ValueError('Invalid internal surface anchors')
    fixed_internal=np.isin(weld,pins);np.testing.assert_array_equal(actual[fixed_internal],proof['before'][fixed_internal])
    constrained=proof.get('constrained_ids',selected);forward_ids=proof.get('forward_ids',selected)
    for ids,parents in ((constrained,selected),(forward_ids,constrained)):
        if ids.dtype.kind not in 'iu' or ids.ndim!=1 or not len(ids) or len(np.unique(ids))!=len(ids) or not np.all(np.isin(ids,parents)):raise ValueError('Invalid native source-constraint scope')
    source_editable=np.isin(weld,constrained)
    record={key.removeprefix('witness_'):value for key,value in proof.items() if key.startswith('witness_')}
    audit_witnesses(record,proof['source'],proof['source_triangles'])
    caps=np.zeros(len(proof['reference']));caps[constrained]=proof['maximum_offset']
    artist=np.abs(actual[editable]-proof['reference'][weld[editable]])
    if np.any(artist>proof['limits'][weld[editable]]+1e-12):raise ValueError('Stored accurate fit exceeds its artist movement bounds')
    def physical(points):return np.column_stack([np.bincount(weld,weights=points[:,axis])/count for axis in range(3)])
    def signed(points):
        p=physical(points)[proof['triangles']]
        return np.einsum('ni,ni->n',np.cross(p[:,1]-p[:,0],p[:,2]-p[:,0]),proof['normals'])
    reversal=(signed(proof['before'])>0)&(signed(actual)<=0)
    if following_scope is not None:reversal&=~following_scope
    if np.any(reversal):raise ValueError('Stored accurate fit adds a source-reference face reversal')
    # BVH supplies candidate facets only. Exact float64 closest points on
    # those original facets provide upper bounds on the true distances.
    # This works for a shortened native step without assuming that the
    # full planned endpoint's nearest facets remain nearest along its path.
    cache=proof.setdefault('_native_source_trees',{})
    current=physical(actual);chosen=np.full(len(current),-1,dtype=np.int64)
    def nearest_ids(tree,points):
        ids=np.fromiter(((-1 if (hit:=tree.find_nearest(Vector(point)))[2] is None else int(hit[2])) for point in points),dtype=np.int64,count=len(points))
        if np.any(ids<0):raise ValueError('Could not certify a native source-surface witness')
        return ids
    for index in np.unique(proof['patch_ids']):
        mask=proof['patch_ids']==index;faces=proof['policy_faces_'+str(index)]
        if int(index) not in cache:cache[int(index)]=BVHTree.FromPolygons(proof['source'].tolist(),proof['source_triangles'][faces].tolist(),all_triangles=True)
        chosen[constrained[mask]]=faces[nearest_ids(cache[int(index)],current[constrained[mask]])]
    native_witness=witnesses(actual[source_editable],proof['source'],proof['source_triangles'],chosen[weld[source_editable]])
    audit_witnesses(native_witness,proof['source'],proof['source_triangles']);source_distance=native_witness['residual']
    if np.any(source_distance>caps[weld[source_editable]]+1e-12):raise ValueError('Stored accurate fit exceeds its source-detail bounds')
    from . import source_budget
    persistent=source_budget.validate_native(actual,proof)
    _,_,forward=closest_triangle_points(current[forward_ids],proof['source'][proof['source_triangles'][chosen[forward_ids]]])
    facets=proof['source'][proof['source_triangles'][proof['measured_source_faces']]]
    area=np.linalg.norm(np.cross(facets[:,1]-facets[:,0],facets[:,2]-facets[:,0]),axis=1)*.5
    bary=np.array([[2/3,1/6,1/6],[1/6,2/3,1/6],[1/6,1/6,2/3]])
    samples=np.einsum('ki,nij->nkj',bary,facets).reshape(-1,3);reverse_weight=np.repeat(area/3,3)
    target=current;triangles=proof['triangles']
    if 'companion' in proof:
        triangles=np.vstack((triangles,proof['companion_triangles']+len(target)));target=np.vstack((target,proof['companion']))
    tree=BVHTree.FromPolygons(target.tolist(),triangles.tolist(),all_triangles=True)
    _,_,reverse=closest_triangle_points(samples,target[triangles[nearest_ids(tree,samples)]])
    upper=dict(forward=dict(rms_m=float(np.sqrt(np.average(forward**2,weights=proof['mass'][forward_ids]))),max_m=float(forward.max())),
        reverse=dict(rms_m=float(np.sqrt(np.average(reverse**2,weights=reverse_weight))),max_m=float(reverse.max())))
    error=float(np.linalg.norm(actual-proof['planned'],axis=1).max(initial=0))
    span=float(np.linalg.norm(np.ptp(proof['before'],axis=0)))
    for report in reports:
        measured=report['surface_measurement']
        for direction in ('forward','reverse'):
            if upper[direction]['rms_m']>=measured[direction+'_before']['rms_m']:raise ValueError('Stored accurate fit does not prove a source-surface improvement')
            if upper[direction]['max_m']>measured[direction+'_before']['max_m']+span*5e-7:raise ValueError('Stored accurate fit increases the worst source-surface gap')
    alignment_report={}
    if bool(proof.get('interior_source_required',False)):
        from .source_surface_interior import InteriorSamples,native_query
        policies=np.unique(proof['patch_ids'])
        if len(policies)!=1:raise ValueError('Triangle-interior proof needs one immutable source policy')
        index=int(policies[0]);policy=proof['policy_faces_'+str(index)]
        movable=np.zeros(len(current),bool);movable[selected]=True
        interior=InteriorSamples(proof['reference'],proof['triangles'],movable,proof['alignment_parameter_reference'])
        def interior_query(points):return native_query(cache[index],proof['source'],proof['source_triangles'],policy,points)
        interior_before=interior.measure(proof['reference'],interior_query)
        interior_after=interior.measure(current,interior_query)
        if interior_after['rms_m']>=interior_before['rms_m']:raise ValueError('Stored accurate fit does not improve triangle-interior source fidelity')
        if interior_after['max_m']>interior_before['max_m']+span*5e-7:raise ValueError('Stored accurate fit increases the worst triangle-interior source gap')
        alignment_report['triangle_interior_source']=dict(method='INDEPENDENT_NATIVE_SEVEN_POINT_CANONICAL_AREA_SOURCE_V1',
            original_policy_reconstructed=True,fixed_canonical_area=True,before=interior_before,after=interior_after)
    if 'attachment_frame' in proof:
        from .attachment_alignment import propose
        section=propose(current,proof['triangles'],proof['source'],proof['source_triangles'],proof['attachment_influences'],[proof['policy_faces_0'],proof['policy_faces_1']],proof['attachment_frame'])['report']
        guard=max(span*2.5e-7,1e-10)
        if section['gap_m']>=float(proof['attachment_before_gap'])-guard or abs(section['height_gap_m'])>abs(float(proof['attachment_before_height_gap']))+guard:raise ValueError('Stored attachment registration does not improve its anatomical section correspondence')
        alignment_report['attachment_registration']=dict(before_gap_m=float(proof['attachment_before_gap']),after_gap_m=section['gap_m'],before_height_gap_m=float(proof['attachment_before_height_gap']),after_height_gap_m=section['height_gap_m'],native_section_checked=True,anchor_policy=section['anchor_policy'],anatomy_review_required=True)
        controls={}
        for side,points,triangles in (('source',proof['source'],proof['source_triangles']),('target',proof['reference'],proof['triangles'])):
            ids=proof['attachment_'+side+'_control_triangle_ids'];bary=proof['attachment_'+side+'_control_barycentric'];positions=proof['attachment_'+side+'_control_points']
            if ids.dtype.kind not in 'iu' or ids.ndim!=1 or len(ids)<3 or ids.min()<0 or ids.max()>=len(triangles) or bary.shape!=(len(ids),3) or positions.shape!=bary.shape or not np.isfinite(bary).all() or not np.isfinite(positions).all() or np.any(bary<-1e-12) or np.any(bary>1+1e-12) or not np.allclose(bary.sum(1),1,rtol=0,atol=1e-12):raise ValueError('Invalid native attachment contour bindings')
            reconstructed=np.einsum('ni,nij->nj',bary,points[triangles[ids]])
            if np.linalg.norm(reconstructed-positions,axis=1).max()>max(span*1e-10,1e-12):raise ValueError('Attachment contour bindings do not reconstruct their original triangles')
            if side=='source' and not np.isin(ids,np.union1d(proof['policy_faces_0'],proof['policy_faces_1'])).all():raise ValueError('Attachment source contours leave the paired source policy')
            if side=='target' and not np.isin(triangles[ids],selected).all():raise ValueError('Attachment target contours include protected vertices')
            controls[side]=(ids,bary,positions)
        ids,bary,before_points=controls['target'];destination=controls['source'][2]
        if before_points.shape!=destination.shape:raise ValueError('Attachment contour counts must match')
        after_points=np.einsum('ni,nij->nj',bary,current[proof['triangles'][ids]])
        before_errors=np.linalg.norm(before_points-destination,axis=1);after_errors=np.linalg.norm(after_points-destination,axis=1)
        before_rms=float(np.sqrt(np.mean(before_errors**2)));after_rms=float(np.sqrt(np.mean(after_errors**2)))
        if after_rms>=before_rms-guard or after_errors.max()>before_errors.max()+guard:raise ValueError('Stored attachment registration does not improve its original contour bindings')
        alignment_report['attachment_registration'].update(contour_controls=len(ids),before_contour_rms_m=before_rms,after_contour_rms_m=after_rms,
            before_contour_max_m=float(before_errors.max()),after_contour_max_m=float(after_errors.max()),native_contour_bindings_checked=True,original_source_contours_verified=True)
    if 'observed_ids' in proof:
        ids=proof['observed_ids'];baseline=proof['observed_before']
        if ids.dtype.kind not in 'iu' or ids.ndim!=1 or not len(ids) or len(np.unique(ids))!=len(ids) or not np.all(np.isin(ids,constrained)) or not np.array_equal(ids,forward_ids) or baseline.shape!=(2,) or not np.isfinite(baseline).all():raise ValueError('Invalid observed partial-surface scope')
        _,_,distance=closest_triangle_points(current[ids],proof['source'][proof['source_triangles'][chosen[ids]]])
        observed=dict(rms_m=float(np.sqrt(np.average(distance**2,weights=proof['mass'][ids]))),max_m=float(distance.max()))
        if observed['rms_m']>=baseline[0] or observed['max_m']>baseline[1]+span*5e-7:raise ValueError('Stored partial-surface completion does not improve its fixed observed-source scope')
        alignment_report.update(observed_surface_bounds=observed,observed_source_scope_checked=True,inferred_vertices_require_review=True,source_bounds_scope='FIXED_OBSERVED_VERTICES',source_constrained_physical_vertices=len(constrained))
    if 'alignment_before_faces' in proof:
        from .surface_defects import compare
        after_faces=proof['alignment_before_faces'].copy()
        if 'alignment_ids' in proof:
            ids=proof['alignment_ids'];patches=proof['alignment_patch_ids']
            if ids.dtype.kind not in 'iu' or ids.ndim!=1 or patches.shape!=ids.shape or len(np.unique(ids))!=len(ids) or not np.all(np.isin(ids,selected)):raise ValueError('Invalid native source-normal scope')
            for index in np.unique(patches):
                mask=patches==index;faces=proof['policy_faces_'+str(index)]
                if int(index) not in cache:cache[int(index)]=BVHTree.FromPolygons(proof['source'].tolist(),proof['source_triangles'][faces].tolist(),all_triangles=True)
                after_faces[ids[mask]]=faces[nearest_ids(cache[int(index)],current[ids[mask]])]
        else:after_faces[selected]=chosen[selected]
        alignment=compare(proof['reference'],current,proof['triangles'],proof['alignment_eligible'],proof['source'],proof['source_triangles'],
            proof['alignment_before_faces'],after_faces,parameter_reference=proof['alignment_parameter_reference'])
        if bool(proof['alignment_required']) and not (alignment['after_energy']<alignment['before_energy']-1e-10 and alignment['new_back_faces']==0):
            raise ValueError('Stored attachment repair does not improve the original surface-fold measurement')
        if bool(proof.get('alignment_non_regression',False)):
            if alignment['new_back_faces'] or alignment['after_energy']>alignment['before_energy']+1e-10:raise ValueError('Stored completion adds a source-normal fold or worsens supported alignment')
            alignment_report['source_alignment_non_regression_checked']=True
        if bool(proof.get('alignment_severe_non_regression',False)):
            if alignment['new_severe_faces']:raise ValueError('Stored completion adds a severe source-alignment defect')
            alignment_report['source_alignment_severe_non_regression_checked']=True
        if bool(proof.get('alignment_severe_non_regression',False)):
            if alignment['after_severe_reference_area_m2']>alignment['before_severe_reference_area_m2']+1e-12:raise ValueError('Stored reconstruction increases severely misaligned source-supported area')
            alignment_report['source_alignment_severe_area_checked']=True
        alignment_report['source_alignment']=alignment
        if bool(proof.get('alignment_back_reduction_required',False)):
            if alignment['after_back_faces']>=alignment['before_back_faces']:raise ValueError('Stored attachment repair did not remove a supported backward-facing face')
            alignment_report['source_backward_face_reduction_checked']=True
        if bool(proof.get('attachment_source_angles',False)):
            from .source_angle_contract import attachment_contract,angular_values
            movable=np.zeros(len(proof['reference']),bool);movable[selected]=True
            if follows:
                directions,scope,contract=original_contract(proof['reference'],proof['triangles'],proof['source'],proof['source_triangles'],proof['alignment_before_faces'],proof['alignment_eligible'],movable,proof['alignment_parameter_reference'],smooth_normals=proof['_following_corner_normals'])
            else:directions,scope,contract=attachment_contract(proof['reference'],proof['triangles'],proof['source'],proof['source_triangles'],proof['alignment_before_faces'],proof['alignment_eligible'],movable,proof['alignment_parameter_reference'])
            if bool(proof.get('attachment_source_planes',False)):
                if np.max(np.abs(proof['normals'][scope]-directions[scope]),initial=0)>1e-12:raise ValueError('Attachment area planes differ from their original source field')
                alignment_report['attachment_source_planes']=dict(original_field_reconstructed=True,faces=int(scope.sum()))
            if follows:
                from .following_source_field import rebind
                for index in np.unique(proof['patch_ids']):
                    use=proof['patch_ids']==index
                    if not np.isin(proof['alignment_before_faces'][constrained[use]],proof['policy_faces_'+str(index)]).all():raise ValueError('Original source branches leave their paired facet policy')
                branch_caps=np.zeros(len(proof['reference']));branch_caps[constrained]=np.maximum(proof['maximum_offset']-max(float(np.linalg.norm(np.ptp(proof['reference'],axis=0)))*2.5e-7,1e-10),0.)
                contract,transport_normals=rebind(current,proof['triangles'],proof['source'],proof['source_triangles'],proof['alignment_before_faces'],after_faces,contract,scope,smooth_normals=proof['_following_corner_normals'],reference=proof['reference'],branch_caps=branch_caps)
                facets=current[proof['triangles']];cross=np.cross(facets[:,1]-facets[:,0],facets[:,2]-facets[:,0])
                transport_signed=np.sum(cross*transport_normals,axis=1)
                if np.any(scope&(signed(proof['before'])>0)&(transport_signed<=0)):raise ValueError('Stored source-field transport reverses an originally source-facing triangle')
                alignment_report['source_field_transport']=dict(original_thresholds_reconstructed=True,original_scope_preserved=True,endpoint_original_facets_verified=True)
            cosine,violation=angular_values(current,proof['triangles'],contract)
            maximum_error=float(violation.max(initial=0));cosine_error=float(np.maximum(contract['minimum_cosine']-cosine,0.).max(initial=0))
            if maximum_error>1e-9 or cosine_error>1e-7:raise ValueError('Stored attachment fit violates its original source-angle constraints')
            alignment_report['attachment_source_angles']=dict(original_field_reconstructed=True,faces=len(contract['face_ids']),
                preserve_initially_aligned=True,maximum_normalized_residual=maximum_error,maximum_cosine_violation=cosine_error)
        if 'alignment_quality_minimum_cosine' in proof:
            threshold=float(proof['alignment_quality_minimum_cosine'])
            if not 0<threshold<1:raise ValueError('Invalid native source-quality angle threshold')
            cosine=float(np.cos(np.deg2rad(alignment['after_angle_quality']['maximum_degrees'])))
            if cosine<threshold-1e-7:raise ValueError('Stored surface does not retain its source-normal quality correction')
            alignment_report['source_alignment_quality_checked']=True
            alignment_report['source_alignment_quality_minimum_cosine']=threshold
    if 'native_storage_prediction' in proof:
        predicted=proof['native_storage_prediction']
        if predicted.shape!=actual.shape or not np.isfinite(predicted).all():raise ValueError('Invalid native storage prediction')
        alignment_report['native_storage_prediction']=dict(whole_array_exact=bool(np.array_equal(actual,predicted)),maximum_error_m=float(np.linalg.norm(actual-predicted,axis=1).max(initial=0)),native_checks_independent=True)
    return dict(**alignment_report,persistent_source_budget=persistent,artist_bounds_checked=True,original_source_witnesses_checked=True,source_bounds_checked=True,
        source_reference_orientation_preserved=True,two_way_surface_improvement_proven=True,native_difference_from_full_plan=error,
        surface_distance_upper_bounds=upper,native_source_faces=fingerprint(native_witness['source_triangle_ids']),
        maximum_source_witness_distance=float(source_distance.max(initial=0)),protected_coordinates_exact=True)
