"""Bounded, unpublished native storage-cell search for source-following fits."""
import time
import numpy as np
from .source_angle_contract import angular_values
from .surface_contacts import crossings


def repair(predictor,planned,weld,canonical_triangles,physical_triangles,free,angle_field,contract,*,cancelled=None,time_limit=20.,source_caps=None,artist_limits=None,artist_reference=None,persistent_source=()):
    began=time.monotonic();weld=np.asarray(weld);count=np.bincount(weld);tri=np.asarray(canonical_triangles)
    def physical(points):return np.column_stack([np.bincount(weld,weights=points[:,axis])/count for axis in range(3)])
    tolerance=float(np.linalg.norm(np.ptp(predictor.before,axis=0)))*1e-6
    active=np.any(np.asarray(free)[weld][tri],axis=1)
    allowed=crossings(predictor.before,tri,weld,pairs=True,tolerance=tolerance,active_faces=active)
    flat_quality_scope=np.zeros(len(physical_triangles),bool)
    if hasattr(angle_field,'flat_directions'):
        reference_points=physical(predictor.before);flat_before,flat_supported=angle_field.flat_directions(reference_points,refresh=False)
        p=reference_points[physical_triangles];C=np.cross(p[:,1]-p[:,0],p[:,2]-p[:,0])
        cosine=np.sum(C*flat_before,axis=1)/np.maximum(np.linalg.norm(C,axis=1),1e-30)
        flat_quality_scope=flat_supported&(cosine>=.25)&np.any(np.asarray(free)[physical_triangles],axis=1)
    trials=0;unchanged_trials=0;duplicate_trials=0
    def evaluate(stored,*,base_actual=None,base_contacts=None,copies=None,audit=None,base_score=None,base_bad=None):
        nonlocal unchanged_trials
        actual=predictor.evaluate(stored);points=physical(actual)
        if base_actual is not None and base_score is not None and np.array_equal(actual,base_actual):
            unchanged_trials+=1
            return base_score,base_bad.copy(),actual,base_contacts.copy()
        unsupported=np.zeros(len(physical_triangles),bool);support_excess=0.
        if hasattr(angle_field,'inspect'):
            angles,_,supported,margin=angle_field.inspect(points)
            unsupported=np.asarray(angle_field.scope)&~supported
            support_excess=float(np.maximum(-margin[unsupported],0.).sum())
        else:angles,_=angle_field(points)
        if not np.array_equal(angles['face_ids'],contract['face_ids']) or not np.array_equal(angles['minimum_cosine'],contract['minimum_cosine']):raise ValueError('Storage repair changed its original angular scope or floors')
        cosine,error=angular_values(points,physical_triangles,angles);cos_error=np.maximum(angles['minimum_cosine']-cosine,0.)
        angular_bad=(error>1e-9)|(cos_error>1e-7)
        bad=unsupported.copy();bad[angles['face_ids'][angular_bad]]=True
        flat_bad=np.zeros(len(physical_triangles),bool);flat_excess=0.
        if flat_quality_scope.any():
            flat,flat_supported,flat_vertices,_=angle_field.flat_directions(points,return_field=True)
            p=points[physical_triangles];C=np.cross(p[:,1]-p[:,0],p[:,2]-p[:,0])
            flat_cosine=np.sum(C*flat,axis=1)/np.maximum(np.linalg.norm(C,axis=1),1e-30)
            flat_bad=flat_quality_scope&((flat_cosine<.25)|~flat_supported)
            bad|=flat_bad
            selected_tri=physical_triangles[flat_quality_scope];raw=flat_vertices[selected_tri]
            unit=C[flat_quality_scope]/np.maximum(np.linalg.norm(C[flat_quality_scope],axis=1)[:,None],1e-30)
            outward_margin=np.einsum('fki,fi->fk',raw,unit).min(1)-.25
            support_margin=np.maximum(np.linalg.norm(raw.sum(1),axis=1)-2.4,outward_margin)
            quality_margin=np.minimum(flat_cosine[flat_quality_scope]-.25,support_margin)
            flat_excess=float(np.maximum(-quality_margin,0).sum())
        position_excess=0.;persistent_bad=np.zeros(len(actual),bool)
        for indices,caps,query in persistent_source:
            excess=np.maximum(query(actual[indices])-caps-1e-12,0.)
            persistent_bad[indices[excess>0]]=True;position_excess+=float(excess.sum())
        if persistent_bad.any():
            failed_vertices=np.zeros(len(points),bool);failed_vertices[weld[persistent_bad]]=True
            bad|=np.any(failed_vertices[np.asarray(physical_triangles)],axis=1)
        source_bad_count=artist_bad_count=0
        if source_caps is not None:
            from .vertex_contract import witnesses
            selected=np.asarray(angle_field.ids);chosen=angle_field.query(points[selected])['source_triangle_ids']
            face_ids=np.full(len(points),-1,dtype=np.int64);face_ids[selected]=chosen
            caps=np.zeros(len(points));caps[selected]=source_caps;canonical=np.flatnonzero(np.isin(weld,selected))
            record=witnesses(actual[canonical],angle_field.source,angle_field.source_triangles,face_ids[weld[canonical]])
            excess=np.maximum(record['residual']-caps[weld[canonical]]-1e-12,0.)
            source_bad_count=int(np.count_nonzero(excess>0))
            bad_vertices=np.zeros(len(points),bool);bad_vertices[weld[canonical[excess>0]]]=True
            position_excess+=float(excess.sum())
            if artist_limits is not None:
                artist=np.maximum(np.abs(actual[canonical]-artist_reference[weld[canonical]])-artist_limits[weld[canonical]]-1e-12,0.)
                artist_bad_count=int(np.count_nonzero(np.any(artist>0,axis=1)))
                bad_vertices[weld[canonical[np.any(artist>0,axis=1)]]]=True;position_excess+=float(artist.sum())
            bad|=np.any(bad_vertices[np.asarray(physical_triangles)],axis=1)
        local=None
        if base_actual is not None:
            unchanged=np.ones(len(actual),bool);unchanged[copies]=False
            if np.array_equal(actual[unchanged],base_actual[unchanged]):
                affected=np.zeros(len(actual),bool);affected[copies]=True;local=np.any(affected[tri],axis=1)
        if local is None:contacts=crossings(actual,tri,weld,pairs=True,tolerance=tolerance,active_faces=active)-allowed
        else:
            contacts={pair for pair in base_contacts if not local[pair[0]] and not local[pair[1]]}
            contacts|=crossings(actual,tri,weld,pairs=True,tolerance=tolerance,active_faces=local)-allowed
        score=(int(bad.sum())+len(contacts),len(contacts),int(bad.sum()),float(np.maximum(error-1e-9,0.).sum()+np.maximum(cos_error-1e-7,0.).sum()+support_excess+position_excess+flat_excess))
        if audit is not None:audit.update(unsupported_faces=int(unsupported.sum()),angular_faces=int(angular_bad.sum()),flat_quality_faces=int(flat_bad.sum()),source_bound_canonical_vertices=source_bad_count,persistent_source_bound_canonical_vertices=int(persistent_bad.sum()),artist_bound_canonical_vertices=artist_bad_count,new_contacts=len(contacts))
        return score,bad,actual,contacts
    stored=predictor.encode(planned)
    initial_audit={}
    try:initial,bad,actual,contacts=evaluate(stored,audit=initial_audit)
    except ValueError as exc:return None,None,dict(accepted=False,reason=str(exc),trials=0)
    cell_lower=stored.copy();cell_upper=stored.copy()
    local=predictor.local(actual).astype(np.float32)
    delta=stored-predictor.baseline.astype(np.float32)
    # A shape-key coordinate can have much finer spacing than the visible
    # relative-key sum. Search actual output cells too, rather than spending
    # all trials on key-bit changes that cannot affect a stored vertex.
    quantum=np.maximum(np.abs(np.spacing(stored)).astype(float),
        np.maximum(np.abs(np.spacing(delta)).astype(float),np.abs(np.spacing(local)).astype(float)/abs(predictor.value)))
    cell_lower=(stored.astype(float)-8*quantum).astype(np.float32)
    cell_upper=(stored.astype(float)+8*quantum).astype(np.float32)
    best=initial;changes=[]
    for iteration in range(12):
        if not best[0] or time.monotonic()-began>=time_limit:break
        vertices=np.unique(np.asarray(physical_triangles)[bad])
        if contacts:vertices=np.union1d(vertices,weld[tri[np.asarray(sorted(contacts)).ravel()]].ravel())
        vertices=vertices[np.asarray(free)[vertices]];winner=None;finished=False;seen=set()
        for kind,steps in (('KEY_ULP',1),('OUTPUT_CELL',1),('OUTPUT_CELL',2),('OUTPUT_CELL',4),('OUTPUT_CELL',8)):
            for vertex in vertices:
                copies=np.flatnonzero(weld==vertex)
                for axis in range(3):
                    for sign in (-np.inf,np.inf):
                        if cancelled and cancelled():raise RuntimeError('Native storage repair cancelled')
                        if time.monotonic()-began>=time_limit:finished=True;break
                        proposal=stored.copy();values=proposal[copies,axis]
                        if kind=='KEY_ULP':values=np.nextafter(values,np.float32(sign))
                        else:values=(values.astype(float)+np.copysign(steps*quantum[copies,axis],sign)).astype(np.float32)
                        proposal[copies,axis]=np.clip(values,cell_lower[copies,axis],cell_upper[copies,axis])
                        if np.array_equal(proposal[copies,axis],stored[copies,axis]):continue
                        token=(int(vertex),axis,proposal[copies,axis].tobytes())
                        if token in seen:duplicate_trials+=1;continue
                        seen.add(token)
                        try:score,failed,prediction,new_contacts=evaluate(proposal,base_actual=actual,base_contacts=contacts,copies=copies,base_score=best,base_bad=bad)
                        except ValueError:continue
                        trials+=1
                        if score<best and (winner is None or score<winner[0]):winner=(score,failed,prediction,proposal,new_contacts,int(vertex),axis,sign,steps,kind)
                        if score[:3]<best[:3]:finished=True;break
                    if finished:break
                if finished:break
            if finished:break
        if winner is None:break
        best,bad,actual,stored,contacts,vertex,axis,sign,steps,kind=winner
        changes.append(dict(vertex=vertex,axis=axis,direction='UP' if sign>0 else 'DOWN',steps=steps,step_kind=kind))
    final_audit={};audited,audited_bad,actual,audited_contacts=evaluate(stored,audit=final_audit)
    if audited!=best:raise AssertionError('Native storage search disagrees with its final complete audit')
    report=dict(method='BOUNDED_NATIVE_STORAGE_CELL_REPAIR_V4',accepted=not bool(best[0]),source_support_checked=hasattr(angle_field,'inspect'),initial=list(initial),after=list(best),initial_failures=initial_audit,final_failures=final_audit,maximum_output_cell_radius=8,cell_scale='MAX_KEY_DELTA_VISIBLE_FLOAT32_SPACING',changes=changes,trials=trials,unchanged_native_trials=unchanged_trials,duplicate_trials_skipped=duplicate_trials,elapsed_seconds=time.monotonic()-began,native_validation_required=True)
    if best[0]:return None,None,report
    # Local trial caching is an optimization, never the final authority.
    if audited_bad.any() or audited_contacts:raise AssertionError('Storage-cell trial cache disagrees with the complete endpoint audit')
    report['complete_endpoint_audit']=True
    report['canonical_source_bounds_checked']=source_caps is not None
    report['persistent_source_policies_checked']=len(persistent_source)
    changed=np.any(planned!=predictor.before,axis=1);encoded=predictor.plan(stored,changed)
    np.testing.assert_array_equal(predictor(encoded),actual)
    report['maximum_plan_adjustment_m']=float(np.linalg.norm(encoded-planned,axis=1).max())
    return encoded,actual,report
