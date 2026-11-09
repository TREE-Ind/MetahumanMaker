"""Source-certified paired fitting in an optional external Python process.

Only geometry and masks cross the process boundary. No Blender data or
character-specific indices, charts, file paths or pretrained models are used.
"""
import time
import numpy as np
from scipy import sparse
import igl
from .vertex_contract import fingerprint,witnesses,audit_witnesses
from .source_envelope import SourceWitnessBoxes,SourceFacetBoxes,source_reachable_boxes
from .source_differentials import source_differentials
from .surface_oracle import SurfaceOracle
from .surface_fit_measurement import assemble_reverse_projection,measure_surface_fit
from .constrained_surface_fit import solve
from .surface_support import observation_scale,reach_confidence,observation_fade


def fit(data,options,*,progress=None,cancelled=None):
    started=time.monotonic()
    def check():
        if cancelled and cancelled():raise RuntimeError('Accurate fit cancelled; previous surface retained')
        if time.monotonic()-started>210:raise RuntimeError('Accurate fit preparation exceeded its time budget')
    def update(label,value):
        check()
        if progress:progress(label,value)
    raw=np.asarray(data['current'],dtype=float);weld=np.asarray(data['weld'],dtype=np.int64)
    source=np.asarray(data['source'],dtype=float);source_tri=np.asarray(data['source_triangles'],dtype=np.int64)
    tri=weld[np.asarray(data['triangles'],dtype=np.int64)]
    tri=tri[np.all(np.sort(tri,axis=1)[:,1:]!=np.sort(tri,axis=1)[:,:-1],axis=1)]
    if raw.ndim!=2 or raw.shape[1]!=3 or weld.shape!=(len(raw),) or not np.isfinite(raw).all() or not len(tri):raise ValueError('Invalid accurate-fit snapshot')
    count=np.bincount(weld)
    if not len(count) or np.any(count==0):raise ValueError('Physical vertex families must be contiguous')
    def physical(points):return np.column_stack([np.bincount(weld,weights=points[:,axis])/count for axis in range(3)])
    reference=physical(raw);n=len(reference);span=float(np.linalg.norm(np.ptp(reference,axis=0)))
    if not span>0:raise ValueError('Accurate fitting needs a nonempty surface')
    edges=np.unique(np.sort(np.vstack((tri[:,[0,1]],tri[:,[1,2]],tri[:,[2,0]])),axis=1),axis=0)
    p=reference[tri];cross=np.cross(p[:,1]-p[:,0],p[:,2]-p[:,0]);area=np.linalg.norm(cross,axis=1)
    face_normals=cross/np.maximum(area[:,None],1e-30);mass=np.zeros(n);vertex_normals=np.zeros_like(reference)
    for corner in range(3):
        np.add.at(mass,tri[:,corner],area/6);np.add.at(vertex_normals,tri[:,corner],cross)
    adj=sparse.coo_matrix((np.ones(len(edges)*2),(np.r_[edges[:,0],edges[:,1]],np.r_[edges[:,1],edges[:,0]])),shape=(n,n)).tocsr()
    degree=np.asarray(adj.sum(1)).ravel();lap=sparse.eye(n)-sparse.diags(1/np.maximum(degree,1))@adj
    for _ in range(3):vertex_normals=(vertex_normals+(adj@vertex_normals)/np.maximum(degree[:,None],1))/2
    influence=np.zeros(n);assignment=np.full(n,-1,int);regions=options['regions']
    for index,region in enumerate(regions):
        values=np.asarray(data['influence_'+str(index)],dtype=float)
        if values.shape!=(n,) or not np.isfinite(values).all() or np.any((values<0)|(values>1)):raise ValueError('Invalid paired fit weights')
        use=values>influence;assignment[use]=index;influence[use]=values[use]
    local_scale=observation_scale(reference,tri);support_strength=np.zeros(n)
    candidate_ids=np.flatnonzero(influence>1e-4)
    if not len(candidate_ids):raise ValueError('This region is empty or fully pinned')
    targets=reference.copy();confidence=np.zeros(n);query_face=np.full(n,-1,int);partitions=[]
    update('Finding source detail',.03)
    for index,region in enumerate(regions):
        check();ids=candidate_ids[assignment[candidate_ids]==index]
        if not len(ids):continue
        faces=np.asarray(data['source_faces_'+str(index)],dtype=np.int64)
        oracle=SurfaceOracle(source,source_tri,faces)
        observation=options.get('source_observation','NEAREST')
        if observation not in {'NEAREST','NORMAL_LINE'}:raise ValueError('Choose NEAREST or NORMAL_LINE source correspondence')
        query=oracle.ray_query if observation=='NORMAL_LINE' else oracle.query
        arguments=dict(directions=vertex_normals[ids]) if observation=='NORMAL_LINE' else dict(query_normals=vertex_normals[ids])
        record=query(reference[ids],**arguments,normal_limit=.1,
            max_distance=span*float(options.get('search_distance',.15)),cancelled=lambda:bool(cancelled and cancelled()) or time.monotonic()-started>90)
        audit_witnesses(record,source,source_tri);observed=record['observed'].copy()
        # A selected/open source rim is not evidence for skin continuing
        # through a missing or intentionally excluded surface. Prevent a
        # nearest-facing query from pulling hidden anatomy down to that rim.
        observed_ids=np.flatnonzero(observed)
        if len(observed_ids):
            residual=record['residual'][observed_ids]
            clearance=oracle.boundary_clearance(record['surface_points'][observed_ids],cutoff=max(float(residual.max())*.5,span*5e-6),cancelled=cancelled)
            observed[observed_ids]&=(residual<=span*5e-6)|(clearance>=residual*.5)
        support=reach_confidence(record['residual'],local_scale[ids],rings=float(options.get('observation_rings',2.)))
        observed&=support>.01;good=ids[observed];support_strength[good]=support[observed]
        targets[good]=record['surface_points'][observed];confidence[good]=record['confidence'][observed];query_face[good]=record['source_triangle_ids'][observed]
        partitions.append((index,faces))
        del oracle
    support_strength*=observation_fade((influence>1e-4)&(confidence>0),tri)
    free=(influence>1e-4)&(confidence>0)&(mass>0)&(support_strength>.01);selected=np.flatnonzero(free)
    if not len(selected):raise ValueError('No matching source surface was found; adjust the paired masks or alignment')
    trees=[]
    for index,faces in partitions:
        mask=assignment[selected]==index
        if not mask.any():continue
        facets=source_tri[faces];tree=igl.AABB();tree.init(source,facets)
        trees.append(dict(mask=mask,tree=tree,faces=facets,global_faces=faces));del tree
    def query(points):
        check();faces=np.full(len(points),-1,dtype=np.int64)
        for patch in trees:
            _,local,_=patch['tree'].squared_distance(source,patch['faces'],points[patch['mask']])
            faces[patch['mask']]=patch['global_faces'][local]
        return witnesses(points,source,source_tri,faces)
    try:
        initial=query(reference[selected]);source_proof=audit_witnesses(initial,source,source_tri)
        public_limits=source_reachable_boxes(reference,selected,targets[selected],free,edges,baseline=span*.01,margin=span*.001)
        public_limits[free]*=(influence[free]*support_strength[free])[:,None]
        guard=max(span*2.5e-7,1e-10)
        solver_limits=np.maximum(public_limits-guard,public_limits*.5)
        offset=np.maximum(initial['residual'],span*5e-6)+guard*2
        from .source_budget import constrain
        offset,source_radius=constrain(data,selected,assignment[selected],offset,offset-guard)
        envelope=options.get('source_envelope','FACET')
        if envelope not in {'POINT','FACET'}:raise ValueError('Unknown source envelope')
        extra=dict(source_vertices=source,source_triangles=source_tri) if envelope=='FACET' else {}
        projector=(SourceFacetBoxes if envelope=='FACET' else SourceWitnessBoxes)(reference,solver_limits,selected,query,source_radius=source_radius,
            query_contract=dict(vertices=fingerprint(source),triangles=fingerprint(source_tri)),**extra)
        update('Matching surface shape',.18)
        differential_target=np.asarray(lap@reference);differential_weight=np.full(n,.005)
        for index,faces in partitions:
            ids=selected[assignment[selected]==index]
            if not len(ids):continue
            record=witnesses(reference[ids],source,source_tri,query_face[ids])
            differential=source_differentials(reference,tri,source,source_tri,ids,record,faces)
            weight=differential['confidence']*.15
            differential_target[ids]=(differential_target[ids]*.005+weight[:,None]*differential['target'])/(.005+weight[:,None])
            differential_weight[ids]+=weight
        source_ids=np.unique(np.concatenate([faces for _,faces in partitions]))
        companion={}
        if 'companion' in data:companion=dict(companion_vertices=data['companion'],companion_triangles=data['companion_triangles'])
        reverse=assemble_reverse_projection(reference,tri,source,source_tri,free,source_ids,**companion)
        weights=mass*confidence*influence*2
        projection=dict(matrix=sparse.diags(weights)+2*reverse['matrix'],load=weights[:,None]*targets+2*reverse['load'],
            mass=weights+2*reverse['mass'],constant=float(np.sum(weights[:,None]*targets**2)+2*reverse['constant']))
        # Source normals guide well-observed faces; boundary faces retain
        # their original direction where a complete source field is absent.
        field=np.zeros_like(reference);facets=source[source_tri[query_face[selected]]]
        normals=np.cross(facets[:,1]-facets[:,0],facets[:,2]-facets[:,0]);normals/=np.maximum(np.linalg.norm(normals,axis=1)[:,None],1e-30)
        field[selected]=normals;complete=np.all(free[tri],axis=1)
        directions=field[tri[complete]].sum(1);directions/=np.maximum(np.linalg.norm(directions,axis=1)[:,None],1e-30)
        valid=np.linalg.norm(directions,axis=1)>.5;face_normals[np.flatnonzero(complete)[valid]]=directions[valid]
        minimum=np.maximum(area*.1,span**2*1e-12)
        def checkpoint(points,receipt):
            steps=len(receipt.get('history',[]));update('Accurate fit: step '+str(steps),min(.9,.25+steps/max(int(options.get('iterations',16)),1)*.6))
        fitted,solver_report=solve(reference,reference,tri,projection,free,face_normals,minimum,mass*.05,lap,
            iterations=int(options.get('iterations',16)),time_limit=120.,checkpoint=checkpoint,prevent_new_contacts=True,
            initial_points=reference.copy(),source_projector=projector,movement_boxes=solver_limits,
            contact_margin=span*1e-6,differential_target=differential_target,differential_weight=differential_weight,
            joint_projection='conic',preserve_orientations=True,compact_proposal=True,
            conic_segment_backtracking=True,conic_elastic_model=True,localize_contact_stalls=envelope=='FACET')
        update('Checking the complete surface',.94)
        measured=measure_surface_fit(reference,fitted,tri,source,source_tri,free,source_ids,**companion)
        accepted=all(measured[d+'_after']['rms_m']<measured[d+'_before']['rms_m'] for d in ('forward','reverse'))
        accepted&=all(measured[d+'_after']['max_m']<=measured[d+'_before']['max_m']+span*5e-7 for d in ('forward','reverse'))
        measured_faces=measured.pop('measured_source_face_ids')
        body=raw+(fitted-reference)[weld] if accepted else raw.copy()
        report=dict(solver='PAIRED_JOINT_ELASTIC_SURFACE_V1',accepted=bool(accepted),review_required=True,
            regions=regions,physical_vertices=n,free_physical_vertices=len(selected),uv_vertices=len(raw),
            unsupported_vertices=int(len(candidate_ids)-len(selected)),source_contract=source_proof,source_envelope=envelope,source_observation=observation,observation_rings=float(options.get('observation_rings',2.)),
            before=dict(forward_rms=measured['forward_before']['rms_m'],reverse_rms=measured['reverse_before']['rms_m'],crossings=solver_report['initial']['crossings']),
            after=dict(forward_rms=measured['forward_after']['rms_m'],reverse_rms=measured['reverse_after']['rms_m'],crossings=solver_report['chosen_crossings']),
            surface_measurement=measured,numerical=solver_report,elapsed_seconds=time.monotonic()-started)
        proof={}
        if accepted:
            projector.prepare(fitted);record=projector.witness;audit_witnesses(record,source,source_tri)
            proof=dict(weld=weld,reference=reference,planned=body,selected=selected,limits=public_limits,maximum_offset=offset,
                triangles=tri,normals=face_normals,before=raw,mass=mass,source_face_ids=source_ids,measured_source_faces=measured_faces,
                patch_ids=assignment[selected],**{'witness_'+key:value for key,value in record.items()})
            for index,faces in partitions:proof['policy_faces_'+str(index)]=faces
            if companion:proof.update(companion=data['companion'],companion_triangles=data['companion_triangles'])
        update('Ready for native checks',1.)
        return body,[report],proof
    finally:
        for patch in trees:patch.pop('tree',None)
        trees.clear()
