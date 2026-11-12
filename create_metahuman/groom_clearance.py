"""Asynchronous local groom clearance with bounded proposals and sparse reverts."""
import hashlib,json,os,threading,uuid
from pathlib import Path
import bpy
import numpy as np
from mathutils import Vector
from mathutils.bvhtree import BVHTree
from bpy.app.handlers import persistent
from . import mesh_ops,resource_budget,process_ref

_JOB=None
_RETIRED=[]
LAYERS='cm_groom_clearance_layers'


def pending():return _JOB is not None
def running():return bool(_JOB and not _JOB['done'].is_set())
def ready():return bool(_JOB and _JOB.get('ready'))
def digest(file):return hashlib.sha256(Path(file).read_bytes()).hexdigest()


def write_json(file,value):
    """Expose saved requests and terminal receipts only after a full write."""
    file=Path(file);temporary=file.with_name(file.name+'.tmp')
    with temporary.open('w',encoding='utf-8') as stream:
        json.dump(value,stream,indent=2,allow_nan=False)
        stream.write('\n');stream.flush();os.fsync(stream.fileno())
    temporary.replace(file)

def array_hash(*values):
    h=hashlib.sha256()
    for value in values:h.update(np.ascontiguousarray(value).tobytes())
    return h.hexdigest()


def settings(g):
    return dict(clearance=g.clearance_mm/1000,root_guard=g.clearance_root_mm/1000,
                maximum_offset=g.clearance_offset_mm/1000,maximum_length_change=g.clearance_length,
                stiffness=12.,passes=10,route=g.clearance_route,back_axis=list(g.clearance_back_axis))


def normalized_settings(value):
    result=dict(value);result.setdefault('route','SURFACE');result.setdefault('back_axis',[0.,1.,0.])
    return result


def inputs(context,groom,head,region,group):
    from . import paint_tools,sculpt_session
    f=context.scene.cm_features
    if paint_tools.running() or sculpt_session.pending() or f.body_editing or f.full_sculpt or f.restore_editing:
        raise ValueError('Finish the active paint, body or sculpt session before groom clearance')
    if context.mode!='OBJECT' or context.scene.cm_features.expression_active:
        raise ValueError('Finish the active edit and return to Neutral before groom clearance')
    if not groom or groom.type!='CURVES' or not groom.is_editable or not groom.data.is_editable:
        raise ValueError('Choose local editable Hair Curves for clearance')
    mesh_ops.check_mesh(head,'Clearance head')
    scale=context.scene.unit_settings.scale_length
    if not np.isfinite(scale) or scale<=0:raise ValueError('Use a positive scene unit scale')
    local=np.empty((len(groom.data.points),3),np.float32);groom.data.position_data.foreach_get('vector',local.ravel())
    offsets=np.empty(len(groom.data.curve_offset_data),np.int32);groom.data.curve_offset_data.foreach_get('value',offsets)
    if len(local)<2 or np.any(np.diff(offsets)<2):raise ValueError('Every strand needs at least two points')
    evaluated=groom.evaluated_get(context.evaluated_depsgraph_get()).data
    if not isinstance(evaluated,bpy.types.Curves) or len(evaluated.points)!=len(local):raise ValueError('Apply topology-changing hair modifiers on a copy before clearance')
    current=np.empty_like(local);evaluated.position_data.foreach_get('vector',current.ravel())
    if not np.allclose(current,local,rtol=0,atol=1e-7):raise ValueError('Apply the evaluated hair deformation on a copy before clearance')
    matrix=np.asarray(groom.matrix_world)
    if not np.isfinite(matrix).all() or abs(np.linalg.det(matrix[:3,:3]))<1e-12:raise ValueError('The groom needs a finite invertible transform')
    points=(local@matrix[:3,:3].T+matrix[:3,3])*scale
    depsgraph=context.evaluated_depsgraph_get();obj=head.evaluated_get(depsgraph);mesh=obj.to_mesh(preserve_all_data_layers=True,depsgraph=depsgraph)
    try:
        mesh.calc_loop_triangles();vertices=np.empty((len(mesh.vertices),3),np.float64);mesh.vertices.foreach_get('co',vertices.ravel())
        transform=np.asarray(obj.matrix_world)
        if not np.isfinite(transform).all() or abs(np.linalg.det(transform[:3,:3]))<1e-12:raise ValueError('The collider needs a finite invertible transform')
        vertices=(vertices@transform[:3,:3].T+transform[:3,3])*scale
        triangles=np.array([t.vertices[:] for t in mesh.loop_triangles],dtype=np.int32)
        if np.linalg.det(transform[:3,:3])<0:triangles=triangles[:,[0,2,1]]
        if region=='HEAD':weights=np.ones(len(vertices))
        else:
            names={v.index for v in head.vertex_groups if (v.name==group if region=='CUSTOM' else v.name=='CM_Ears')}
            if region=='EARS' and not names:names={v.index for v in head.vertex_groups if v.name.startswith('FACIAL_') and '_Ear' in v.name}
            if not names:raise ValueError('Use a native ear-weighted head, or choose a custom vertex group')
            weights=np.array([min(1.,sum(w.weight for w in v.groups if w.group in names)) for v in mesh.vertices])
        if np.count_nonzero(weights>.1)<16:raise ValueError('The clearance region has too few weighted vertices')
        axes=transform[:3,:3].copy()
        axes/=np.maximum(np.linalg.norm(axes,axis=0),1e-12)
    finally:obj.to_mesh_clear()
    if not np.isfinite(points).all() or not np.isfinite(vertices).all():raise ValueError('Clearance inputs contain invalid positions')
    stamp=array_hash(local,offsets,matrix,vertices,triangles,weights,np.array([scale]))
    origin=vertices.mean(0);points-=origin;vertices-=origin
    selected=vertices[weights>.1];bounds=[]
    populations=[selected]
    if region=='EARS':
        center=selected.mean(0);axis=np.linalg.svd(selected-center,full_matrices=False)[2][0]
        populations=[selected[((selected-center)@axis)*side>=0] for side in (-1,1)]
    for population in populations:bounds.append((population.min(0)-.008,population.max(0)+.008))
    return dict(points=points,lengths=np.diff(offsets),vertices=vertices,triangles=triangles,weights=weights,
                bounds=bounds,axes=axes,origin=origin,scale=scale,stamp=stamp,offsets=offsets)


def start(context):
    global _JOB
    from . import groom_jobs,groom_clearance_core
    if pending() or groom_jobs.running() or any(t.is_alive() for t in _RETIRED):raise ValueError('Finish the original groom operation before starting clearance')
    g=context.scene.cm_groom;head=g.clearance_head or g.main_scalp or context.scene.cm_settings.result;groom=g.groom
    previous=Path(bpy.path.abspath(g.clearance_receipt)) if g.clearance_receipt else None
    if previous and not previous.is_file() and (previous.parent/'request.json').is_file():
        owner=json.loads((previous.parent/'request.json').read_text()).get('owner')
        if process_ref.alive(owner):raise ValueError('The original Blender process still owns this clearance request')
    count=len(groom.data.points) if groom and groom.type=='CURVES' else 0
    resource_budget.require('groom surface clearance',512*resource_budget.MIB+count*128)
    parent=Path(bpy.path.abspath(context.scene.cm_settings.project_dir)).resolve()/'GroomClearance'/'.jobs'
    resource_budget.require_disk('groom surface clearance',parent.parent,128*resource_budget.MIB+count*32)
    data=inputs(context,groom,head,g.clearance_region,g.clearance_group);options=settings(g)
    if options['route']=='TUCK_EARS' and g.clearance_region!='EARS':
        raise ValueError('Choose the Ears region for behind-ear routing')
    if options['route']=='TUCK_EARS' and np.linalg.norm(options['back_axis'])<1e-6:
        raise ValueError('Choose a nonzero head-local direction toward the back of the head')
    folder=parent/uuid.uuid4().hex;folder.mkdir(parents=True)
    manifest=dict(schema=1,groom=groom.name,head=head.name,region=g.clearance_region,group=g.clearance_group,
                  source_stamp=data['stamp'],settings=options,root_pins=True,owner=process_ref.identify(os.getpid()))
    write_json(folder/'request.json',manifest)
    job=dict(scene=context.scene,groom=groom,head=head,data=data,options=options,manifest=manifest,folder=folder,
             cancel=threading.Event(),done=threading.Event(),ready=False,progress=('Preparing surface queries',0,1))
    def work():
        try:
            tree=BVHTree.FromPolygons(data['vertices'].tolist(),data['triangles'].tolist(),all_triangles=True)
            weights=data['weights'][data['triangles']].mean(1)
            ray_length=max(float(np.linalg.norm(np.ptp(data['vertices'],axis=0)))*2,.01)
            epsilon=max(ray_length*1e-6,1e-7)
            directions=[Vector(data['axes']@v).normalized() for v in ((1,.271,.13),(-.37,1,.11),(1,-.618,-.05))]
            def nearest(point):
                q,n,i,d=tree.find_nearest(Vector(point))
                return (q,n,weights[i]) if q is not None else None
            def inside(point):
                votes=0
                for direction in directions:
                    p=Vector(point);hits=0
                    for _ in range(32):
                        hit,n,i,d=tree.ray_cast(p,direction,ray_length)
                        if hit is None:break
                        hits+=1;p=hit+direction*epsilon
                    votes+=hits%2
                return votes>=2
            baseline_depths={}
            def safety(point,reference):
                p=Vector(point);q,normal,index,distance=tree.find_nearest(p)
                if q is None or distance<1e-6 or (p-q).dot(normal)>=0 or not inside(point):return None
                key=tuple(reference)
                if key not in baseline_depths:
                    r=Vector(reference);old,old_normal,_,depth=tree.find_nearest(r)
                    baseline_depths[key]=-float(depth) if old is not None and (r-old).dot(old_normal)<0 and inside(reference) else 0.
                allowed=baseline_depths[key]
                if -distance>=allowed-1e-6:return None
                return q,(q-p)/distance,allowed
            routing=None
            if options['route']=='TUCK_EARS':
                # Use the rear exit of the strongly weighted ear surface,
                # rather than the nearest concha surface. The same constraint
                # also routes already-visible strands out of the ear silhouette.
                faces=data['triangles'][weights>.8]
                if len(faces)<16:raise ValueError('The ear region needs strong native weights for routing')
                envelope=BVHTree.FromPolygons(data['vertices'].tolist(),faces.tolist(),all_triangles=True)
                back=Vector(data['axes']@np.asarray(options['back_axis']));back.normalize()
                eligible={}
                def routing(point,reference):
                    # A projected rear envelope must not catch distant temple
                    # hair merely sharing the ear's lateral/vertical coordinates.
                    # Keep eligibility attached to the original strand sample
                    # throughout relaxation so a contact cannot vanish midway.
                    key=tuple(reference)
                    if key not in eligible:
                        q,normal,index,distance=tree.find_nearest(Vector(reference))
                        eligible[key]=bool(q is not None and weights[index]>.5 and
                            (distance<=.002 or ((Vector(reference)-q).dot(normal)<0 and inside(reference))))
                    if not eligible[key]:return None
                    origin=Vector(point)-back*ray_length;last=None
                    for _ in range(32):
                        hit,normal,index,distance=envelope.ray_cast(origin,back,ray_length*2)
                        if hit is None:break
                        last=hit;origin=hit+back*epsilon
                    return (last,back) if last is not None else None
            solver_options={k:v for k,v in options.items() if k not in {'route','back_axis'}}
            result,report=groom_clearance_core.repair(data['points'],data['lengths'],nearest,inside,data['bounds'],
                cancelled=job['cancel'].is_set,progress=lambda *v:job.update(progress=v),routing=routing,safety=safety,**solver_options)
            if job['cancel'].is_set():raise InterruptedError('Groom clearance cancelled; original geometry retained')
            ids=np.flatnonzero(np.any(result!=data['points'],axis=1))
            world=result[ids]+data['origin']
            with (folder/'proposal.tmp').open('wb') as stream:np.savez_compressed(stream,indices=ids,physical_positions=world)
            (folder/'proposal.tmp').replace(folder/'proposal.npz')
            report.update(status='proposed',source_stamp=data['stamp'],proposal_sha256=digest(folder/'proposal.npz'),settings=options)
            if job['cancel'].is_set():raise InterruptedError('Groom clearance cancelled; original geometry retained')
            write_json(folder/'report.json',report);job.update(report=report)
        except Exception as error:
            job.update(error=str(error),cancelled=isinstance(error,InterruptedError))
            write_json(folder/'report.json',dict(status='cancelled' if job['cancelled'] else 'failed',error=str(error)))
        finally:job['done'].set()
    job['thread']=threading.Thread(target=work,daemon=True);_JOB=job;job['thread'].start()
    g.clearance_accept_partial=False
    g.clearance_receipt=str(folder/'report.json');g.clearance_status='Solving a root-pinned groom proposal; the original hair is retained.'
    if not bpy.app.timers.is_registered(poll):bpy.app.timers.register(poll,first_interval=.2,persistent=True)


def poll():
    global _JOB
    job=_JOB
    if not job:return None
    scene=job.get('scene')
    if not scene:
        if not job['done'].is_set():return .2
        _JOB=None;return None
    g=scene.cm_groom
    if not job['done'].is_set():
        phase,count,total=job['progress'];g.clearance_status=f'{phase}: {count:,} / {total:,}'
        return .2
    if job.get('error'):
        g.clearance_status=job['error'];scene.cm_features.last_action_error=not job.get('cancelled',False);_JOB=None;return None
    job['ready']=True;report=job['report'];g.clearance_report=json.dumps(report)
    g.clearance_status=f"Prepared {report['changed_curves']:,} strand repairs; {len(report['unresolved_curves']):,} strands still have sampled clearance violations."
    return None


def apply(context):
    global _JOB
    job=_JOB;g=context.scene.cm_groom
    if not job or not job.get('ready') or job['scene']!=context.scene:raise ValueError('Solve a clearance proposal first')
    report=job['report']
    if report['unresolved_curves'] and not g.clearance_accept_partial:raise ValueError('Inspect the report and explicitly accept a partial repair, or discard and change its limits')
    head=g.clearance_head or g.main_scalp or context.scene.cm_settings.result
    if g.groom!=job['groom'] or head!=job['head'] or settings(g)!=job['options'] or (g.clearance_region,g.clearance_group)!=(job['manifest']['region'],job['manifest']['group']):raise ValueError('Clearance subjects or settings changed; solve a fresh proposal')
    current=inputs(context,g.groom,head,g.clearance_region,g.clearance_group)
    if current['stamp']!=job['data']['stamp']:raise ValueError('Hair or collider changed; retain the artist edits and solve a fresh proposal')
    file=job['folder']/'proposal.npz'
    if digest(file)!=report['proposal_sha256']:raise ValueError('The clearance proposal changed')
    proposal=np.load(file,allow_pickle=False);ids=proposal['indices'];physical=proposal['physical_positions']
    groom=g.groom;previous=groom.data;local=np.empty((len(previous.points),3),np.float32);previous.position_data.foreach_get('vector',local.ravel())
    inverse=np.asarray(groom.matrix_world.inverted());after=(physical/current['scale'])@inverse[:3,:3].T+inverse[:3,3]
    before=local[ids].copy();after=after.astype(np.float32);local[ids]=after
    changed_roots=np.intersect1d(ids,current['offsets'][:-1])
    if len(changed_roots):raise RuntimeError('A clearance proposal attempted to move strand roots')
    layer=job['folder']/('applied-'+uuid.uuid4().hex+'.npz')
    with layer.with_suffix('.tmp').open('wb') as stream:np.savez_compressed(stream,indices=ids,before=before,after=after)
    layer.with_suffix('.tmp').replace(layer)
    old_layers=groom.get(LAYERS);old_report=groom.get('cm_groom_clearance_report')
    layers=json.loads(old_layers or '[]');layers.append(dict(file=str(layer),sha256=digest(layer),topology=array_hash(current['offsets']),points=len(local),report=str(job['folder']/'report.json')))
    private=previous.copy()
    try:
        private.position_data.foreach_set('vector',local.ravel());private.update_tag();groom.data=private
        groom[LAYERS]=json.dumps(layers);groom['cm_groom_clearance_report']=json.dumps(report)
        context.view_layer.update()
    except Exception:
        groom.data=previous
        for key,value in ((LAYERS,old_layers),('cm_groom_clearance_report',old_report)):
            if value is not None:groom[key]=value
            elif key in groom:del groom[key]
        if not private.users:bpy.data.hair_curves.remove(private)
        raise
    g.clearance_status='Applied local clearance geometry. Inspect it before Unreal delivery; roots and other attributes retained.';_JOB=None


def resume(context):
    global _JOB
    if pending() or any(t.is_alive() for t in _RETIRED):raise ValueError('Observe the original clearance operation first')
    g=context.scene.cm_groom;file=Path(bpy.path.abspath(g.clearance_receipt)) if g.clearance_receipt else None
    if not file or not (file.parent/'request.json').is_file():raise ValueError('Choose a saved clearance report first')
    manifest=json.loads((file.parent/'request.json').read_text())
    manifest['settings']=normalized_settings(manifest['settings'])
    if not file.is_file():
        if process_ref.alive(manifest.get('owner')):raise ValueError('The original Blender process is still working; observe that request')
        raise ValueError('The original process stopped without a completed proposal; retain its files and start a fresh solve')
    report=json.loads(file.read_text());head=g.clearance_head or g.main_scalp or context.scene.cm_settings.result
    if report.get('status')!='proposed':raise ValueError(report.get('error','This clearance request has no applicable proposal'))
    if not g.groom or not head or g.groom.name!=manifest['groom'] or head.name!=manifest['head']:raise ValueError('Select the original groom and head before resuming')
    data=inputs(context,g.groom,head,manifest['region'],manifest['group'])
    if data['stamp']!=manifest['source_stamp'] or settings(g)!=manifest['settings'] or (g.clearance_region,g.clearance_group)!=(manifest['region'],manifest['group']):raise ValueError('Clearance inputs changed; solve a fresh proposal')
    if digest(file.parent/'proposal.npz')!=report['proposal_sha256']:raise ValueError('The saved proposal changed')
    done=threading.Event();done.set()
    _JOB=dict(scene=context.scene,groom=g.groom,head=head,data=data,options=manifest['settings'],manifest=manifest,
              folder=file.parent,cancel=threading.Event(),done=done,ready=True,report=report)
    g.clearance_accept_partial=False
    poll()


def revert(context):
    g=context.scene.cm_groom;groom=g.groom
    if pending() or context.mode!='OBJECT' or not groom or groom.type!='CURVES':raise ValueError('Finish the pending operation and choose the groom in Object Mode')
    layers=json.loads(groom.get(LAYERS,'[]'))
    if not layers:raise ValueError('This groom has no saved clearance geometry layer')
    record=layers[-1];file=Path(record['file'])
    if not file.is_file() or digest(file)!=record['sha256']:raise ValueError('Retain this project cache to revert the saved clearance geometry')
    offsets=np.empty(len(groom.data.curve_offset_data),np.int32);groom.data.curve_offset_data.foreach_get('value',offsets)
    if len(groom.data.points)!=record['points'] or array_hash(offsets)!=record['topology']:raise ValueError('Groom topology changed; retain the artist edits')
    saved=np.load(file,allow_pickle=False);ids=saved['indices'];local=np.empty((len(groom.data.points),3),np.float32);groom.data.position_data.foreach_get('vector',local.ravel())
    if not np.array_equal(local[ids],saved['after']):raise ValueError('The repaired points were edited afterward; retain those artist edits')
    local[ids]=saved['before'];previous=groom.data;private=previous.copy()
    old_layers=groom.get(LAYERS);old_report=groom.get('cm_groom_clearance_report')
    try:
        private.position_data.foreach_set('vector',local.ravel());private.update_tag();groom.data=private
        groom[LAYERS]=json.dumps(layers[:-1])
        report=Path(layers[-2]['report']) if len(layers)>1 and layers[-2].get('report') else None
        if report and report.is_file():groom['cm_groom_clearance_report']=report.read_text()
        elif 'cm_groom_clearance_report' in groom:del groom['cm_groom_clearance_report']
        context.view_layer.update()
    except Exception:
        groom.data=previous
        for key,value in ((LAYERS,old_layers),('cm_groom_clearance_report',old_report)):
            if value is not None:groom[key]=value
            elif key in groom:del groom[key]
        if not private.users:bpy.data.hair_curves.remove(private)
        raise
    g.clearance_status='Previous clearance positions restored; current materials and other attributes retained.'


@persistent
def cancel(*_):
    global _JOB
    job=_JOB
    if not job:return
    job['cancel'].set()
    if job.get('thread') and job['thread'].is_alive():_RETIRED.append(job['thread'])
    job['scene']=job['groom']=job['head']=None;_JOB=None


def register():
    for name in ('load_pre','undo_pre','redo_pre'):
        handlers=getattr(bpy.app.handlers,name)
        if cancel not in handlers:handlers.append(cancel)


def unregister():
    cancel()
    if bpy.app.timers.is_registered(poll):bpy.app.timers.unregister(poll)
    for name in ('load_pre','undo_pre','redo_pre'):
        handlers=getattr(bpy.app.handlers,name)
        if cancel in handlers:handlers.remove(cancel)
