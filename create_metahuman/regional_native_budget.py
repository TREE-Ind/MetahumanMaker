"""Saved source limits for local fitting modes without an external solver."""
import numpy as np


def contact_snapshot(target):
    from . import mesh_ops as m
    rest=m.coordinates(target);tolerance=max(float(np.linalg.norm(np.ptp(rest,axis=0)))*1e-6,1e-9)
    _,weld=np.unique(np.rint(rest/tolerance).astype(np.int64),axis=0,return_inverse=True)
    target.data.calc_loop_triangles()
    return dict(weld=weld,triangles=np.array([t.vertices[:] for t in target.data.loop_triangles],dtype=np.int64))


def snapshot(context,source,target,current,tasks,*,for_polish=False):
    from . import source_budget,surface_fit_jobs
    if not target.get(source_budget.PROPERTY) and not for_polish:return None
    data,_=surface_fit_jobs.prepare(context,source,target,current,tasks)
    output={key:data[key] for key in ('source','source_triangles','weld','source_budget_context','source_budget_previous')}
    if for_polish:
        from mathutils import Vector
        from mathutils.bvhtree import BVHTree
        from .vertex_contract import witnesses,audit_witnesses
        weld=data['weld'];count=np.bincount(weld)
        physical=np.column_stack([np.bincount(weld,weights=current[:,axis])/count for axis in range(3)])
        spread=np.zeros(len(physical));np.maximum.at(spread,weld,np.linalg.norm(current-physical[weld],axis=1))
        span=float(np.linalg.norm(np.ptp(physical,axis=0)));previous=bytes(data['source_budget_previous']).decode()
        output['original_budget_previous']=data['source_budget_previous'];output['policy_proofs']=[]
        for index,task in enumerate(tasks):
            families=np.unique(weld[task['ids']]);ids=families[task['influence']>0];faces=task['source_face_ids']
            if not len(ids):continue
            tree=BVHTree.FromPolygons(data['source'].tolist(),data['source_triangles'][faces].tolist(),all_triangles=True)
            chosen=faces[np.array([tree.find_nearest(Vector(point))[2] for point in physical[ids]],dtype=np.int64)]
            record=witnesses(physical[ids],data['source'],data['source_triangles'],chosen);audit_witnesses(record,data['source'],data['source_triangles'])
            caps=np.maximum(record['residual']+spread[ids],span*5e-6)+span*5e-7
            caps,_=source_budget.constrain(data,ids,np.full(len(ids),index),caps,caps)
            proof=dict(source_budget_context=data['source_budget_context'],selected=ids,maximum_offset=caps,
                patch_ids=np.zeros(len(ids),dtype=np.int64),policy_faces_0=faces)
            output['policy_proofs'].append(proof)
            previous=source_budget.updated(previous,proof)
        output['pending_ledger']=previous
        output['source_budget_previous']=np.frombuffer(previous.encode(),dtype=np.uint8).copy()
        output['operation_source_caps']=True
    return output


def properties(data,before,output):
    if data is None or not data.get('operation_source_caps'):return None
    from . import source_budget
    changed=np.unique(data['weld'][np.any(output!=before,axis=1)])
    previous=bytes(data['original_budget_previous']).decode()
    for original in data['policy_proofs']:
        use=np.isin(original['selected'],changed)
        if not use.any():continue
        proof=dict(original,selected=original['selected'][use],maximum_offset=original['maximum_offset'][use],patch_ids=original['patch_ids'][use])
        previous=source_budget.updated(previous,proof)
    return {source_budget.PROPERTY:previous} if previous else None


def validator(data,before,output):
    if data is None:return None
    from . import source_budget
    proof=dict(data,selected=np.unique(data['weld'][np.any(output!=before,axis=1)]))
    return lambda actual:dict(persistent_source_budget=source_budget.validate_native(actual,proof),prior_budget_enforced=True,
        original_operation_source_caps=bool(data.get('operation_source_caps')))


def polish_metrics(context,target,before,tasks,reports,native):
    from mathutils import Vector
    from mathutils.bvhtree import BVHTree
    from . import mesh_ops as m,region_projection as rp
    after=m.evaluated_coordinates(context,target)
    for task,report in zip(tasks,reports):
        scatter=task['scatter'];counts=np.bincount(scatter)
        def physical(points):return np.column_stack([np.bincount(scatter,weights=points[task['ids'],axis])/counts for axis in range(3)])
        old=physical(before);current=physical(after);use=task['influence']>0
        tree=BVHTree.FromPolygons(task['source'].tolist(),task['source_triangles'].tolist(),all_triangles=True)
        def rms(points):return float(np.sqrt(np.mean([tree.find_nearest(Vector(point))[3]**2 for point in points[use]])))
        tolerance=float(np.linalg.norm(np.ptp(old,axis=0))*1e-6)
        report['planned_before']=report['before'];report['planned_after']=report['after']
        report['before']=dict(forward_rms=rms(old),crossings=rp.crossings(old,task['triangles'],tolerance=tolerance))
        report['after']=dict(forward_rms=rms(current),crossings=rp.crossings(current,task['triangles'],tolerance=tolerance))
        report['native_step_fraction']=native['step_fraction'];report['native_metrics_scope']='ORIGINAL_OPERATION_REGION'
        report['planned_max_displacement']=report.get('max_displacement');report['max_displacement']=float(np.linalg.norm(current-old,axis=1).max(initial=0))
