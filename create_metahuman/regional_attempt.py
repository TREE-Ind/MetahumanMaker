"""Latest regional job state, separate from the last stored geometry report."""
import json,time,uuid

PROPERTY='cm_region_projection_attempt'


def read(target):
    try:
        value=json.loads(target.get(PROPERTY,'{}'))
        return value if isinstance(value,dict) else {}
    except (ValueError,TypeError):return {}


def publish(job,status,*,error=None,native=None):
    meta=job.setdefault('attempt_meta',dict(id=uuid.uuid4().hex,started_at=time.time(),mode=job['mode'],regions=job.get('regions',[])))
    value=dict(meta,status=status,updated_at=time.time(),candidate_published=bool(job.get('native_committed')))
    if error:value['error']=str(error)
    if job.get('result'):
        value['worker_regions']=[dict(region=row.get('region',row.get('selected_region')),accepted=bool(row.get('accepted')),review_required=bool(row.get('review_required')),reason=row.get('reason')) for row in job['result'][1]]
    if native:
        value['native']=dict(checked=bool(native.get('native_commit_checked')),step_fraction=native.get('step_fraction'),crossings_after=native.get('crossings_after'),protected_coordinates_exact=native.get('protected_coordinates_exact'),surface_validation=native.get('surface_validation'))
    job['target'][PROPERTY]=json.dumps(value)
    return value


def recover(target):
    value=read(target)
    if value.get('status')!='RUNNING':return False
    value.update(status='INTERRUPTED',candidate_published=False,updated_at=time.time(),error='Worker state is not restored by project loading or history.')
    target[PROPERTY]=json.dumps(value);return True
