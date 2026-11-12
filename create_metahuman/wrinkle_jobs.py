"""Stage and publish retained wrinkle identities through an owned Blender worker."""
from pathlib import Path
import hashlib
import json
import subprocess
import traceback
import uuid
import bpy
from bpy.app.handlers import persistent
from . import detail_jobs,detail_subdivision as subd

_JOB=None
_REAPING=[]


def running():
    return _JOB is not None


def stamp(context,scene):
    s,f=scene.cm_settings,scene.cm_features
    objects=[obj for obj in (s.result,s.detail,f.body_result,f.eye_left,f.eye_right,f.teeth,f.wm1,f.wm2,f.wm3) if obj]
    result=[detail_jobs.scene_state(scene)]
    for obj in objects:
        result.append((obj.as_pointer(),detail_jobs.fingerprint(context,s.result,obj),
                       subd.signature(obj) if obj.data.shape_keys else None,
                       obj.get('cm_wrinkle_pose'),obj.get('cm_wrinkle_native'),
                       [(material.as_pointer() if material else 0) for material in obj.data.materials],
                       [(key.slider_min,key.slider_max,key.interpolation,key.vertex_group) for key in obj.data.shape_keys.key_blocks]
                       if obj.data.shape_keys else None,
                       [(group.name,group.lock_weight) for group in obj.vertex_groups]))
    return result


def cleanup(job):
    folder=job['folder'].resolve()
    for name in ('input.blend','output.blend'):
        path=(folder/name).resolve()
        if path.parent!=folder or path.is_symlink():
            raise ValueError('Wrinkle job cleanup target escaped its private folder')
        path.unlink(missing_ok=True)


def start(context,s,f):
    global _JOB
    from . import bake_jobs
    if running() or detail_jobs.running() or bake_jobs.running():
        raise ValueError('Finish or cancel the current sculpt or map update first')
    if context.mode!='OBJECT' or f.body_editing or f.full_sculpt or f.restore_editing or f.expression_active:
        raise ValueError('Finish the active sculpt, body or expression edit first')
    if s.detail and s.detail.get('cm_head_update_pending'):
        raise ValueError('Update the retained head sculpt before updating wrinkles')
    slots=[(slot,getattr(f,slot.lower())) for slot in ('WM1','WM2','WM3') if getattr(f,slot.lower())]
    if not slots:
        return False
    from . import mesh_ops
    mesh_ops.check_mesh(s.result,'Accepted head')
    from . import resource_budget
    resource_budget.require('retained wrinkle updates',512*resource_budget.MIB+
        resource_budget.meshes([s.result,s.detail,*[obj for _,obj in slots]]))
    root=Path(bpy.path.abspath(s.project_dir)).resolve()
    storage=resource_budget.scene_snapshot_storage(context.scene)
    base=s.detail or s.result
    output=sum(max(len(old.data.vertices),len(base.data.vertices))*(192+24*len(old.data.shape_keys.key_blocks))
        for _,old in slots if old.data.shape_keys)
    storage['output_estimate_bytes']=output
    storage['required_bytes']=max(2*resource_budget.GIB,
        2*max(storage['scene_estimate_bytes'],output)+512*resource_budget.MIB)
    resource_budget.require_disk('retained wrinkle updates',root,storage['required_bytes'])
    folder=root/'WrinkleUpdates'/'.jobs'/uuid.uuid4().hex[:12]
    folder.mkdir(parents=True)
    ident=folder.name
    request={'id':ident,'scene':context.scene.name,'storage_budget':storage}
    (folder/'request.json').write_text(json.dumps(request,indent=2))
    job={'folder':folder,'id':ident,'scene':context.scene,'cancel':False,'slots':slots}
    pending_before=[(old,'cm_wrinkle_update_pending' in old,old.get('cm_wrinkle_update_pending')) for _,old in slots]
    try:
        for _,old in slots:
            old['cm_wrinkle_update_pending']=s.result
        job['stamp']=stamp(context,context.scene)
        bpy.data.libraries.write(str(folder/'input.blend'),{context.scene},compress=True,path_remap='ABSOLUTE')
        with (folder/'worker.log').open('w') as log:
            job['process']=subprocess.Popen([bpy.app.binary_path,'--background','--factory-startup',
                *detail_jobs.thread_arguments(),'--python-exit-code','1','--python',
                str(Path(__file__).with_name('wrinkle_worker.py')),'--',str(folder)],
                stdout=log,stderr=subprocess.STDOUT,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
        _JOB=job
        s.status='Updating retained wrinkles in the background. Cancel preserves the original layers.'
        bpy.app.timers.register(poll,first_interval=.2,persistent=True)
        return True
    except Exception:
        for old,existed,value in pending_before:
            if existed:old['cm_wrinkle_update_pending']=value
            elif 'cm_wrinkle_update_pending' in old:del old['cm_wrinkle_update_pending']
        process=job.get('process')
        if process is None or process.poll() is not None:cleanup(job)
        raise


def publish(job,report):
    scene=job['scene']
    with bpy.context.temp_override(scene=scene,view_layer=scene.view_layers[0]):
        if bpy.context.mode!='OBJECT' or stamp(bpy.context,scene)!=job['stamp']:
            raise ValueError('Character or wrinkle layers changed during the update; originals retained. Retry when ready.')
        output=job['folder']/'output.blend'
        if report.get('id')!=job['id'] or hashlib.sha256(output.read_bytes()).hexdigest()!=report.get('sha256'):
            raise ValueError('Retained wrinkle output does not match its owned request')
        expected={slot:old for slot,old in job['slots']}
        if len(report['slots'])!=len(expected) or {item['slot'] for item in report['slots']}!=set(expected):
            raise ValueError('Retained wrinkle output is missing a slot')
        before_objects=set(bpy.data.objects)
        before_meshes=set(bpy.data.meshes)
        plans=[]
        snapshots=[]
        from . import map_provenance
        invalidated_before=scene.get(map_provenance.INVALIDATED)
        try:
            with bpy.data.libraries.load(str(output),link=False) as (_,loaded):
                loaded.objects=list(report['objects'])
            by_name=dict(zip(report['objects'],loaded.objects))
            for name,obj in by_name.items():
                if not obj or obj.get('cm_wrinkle_job_id')!=job['id']:
                    raise ValueError('Unexpected retained wrinkle output object')
                scene.collection.objects.link(obj)
                for name_material in report['materials'][name]:
                    material=bpy.data.materials.get(name_material) if name_material else None
                    if name_material and not material:
                        raise ValueError('A wrinkle material was removed; original layers retained')
                    obj.data.materials.append(material)
            for item in report['slots']:
                slot=item['slot'];old=expected[slot];new=by_name[item['new']]
                if item['old']!=old.name or new.get('cm_wrinkle_slot')!=slot:
                    raise ValueError('Retained wrinkle output changed its source slot')
                if not new.data.shape_keys or not len(new.data.vertices):
                    raise ValueError('Retained wrinkle output has no sculpt keys')
                plans.append((slot,old,new))
            # All outputs are loaded and checked before any slot is changed.
            f=scene.cm_features
            snapshots=[(slot,old,old.get('cm_wrinkle_archived'),
                [(obj,obj.hide_get(),obj.hide_render) for obj in (old,*old.children)]) for slot,old,_ in plans]
            for slot,old,new in plans:
                old['cm_wrinkle_archived']=True
                old.hide_set(True);old.hide_render=True
                for child in old.children:
                    child.hide_set(True);child.hide_render=True
                setattr(f,slot.lower(),new)
                new.hide_set(True);new.hide_render=True
                if 'cm_wrinkle_update_pending' in new:
                    del new['cm_wrinkle_update_pending']
            map_provenance.invalidate_keys(scene,[slot for slot,_,_ in plans])
            scene.cm_settings.status='Retained wrinkle sculpts now follow the accepted face and head detail. Bake their maps before delivery.'
            if not bpy.app.background and bpy.context.preferences.edit.use_global_undo:
                bpy.ops.ed.undo_push(message='Update retained wrinkle sculpts')
        except Exception:
            if invalidated_before is None:
                if map_provenance.INVALIDATED in scene:del scene[map_provenance.INVALIDATED]
            else:scene[map_provenance.INVALIDATED]=invalidated_before
            for slot,old,archived,visibility in snapshots:
                setattr(scene.cm_features,slot.lower(),old)
                if archived is None:
                    if 'cm_wrinkle_archived' in old:del old['cm_wrinkle_archived']
                else:old['cm_wrinkle_archived']=archived
                for obj,hidden,render_hidden in visibility:
                    obj.hide_set(hidden);obj.hide_render=render_hidden
            for obj in set(bpy.data.objects)-before_objects:
                bpy.data.objects.remove(obj,do_unlink=True)
            for mesh in set(bpy.data.meshes)-before_meshes:
                if not mesh.users:
                    bpy.data.meshes.remove(mesh)
            raise


def finish(job):
    code=job['process'].poll()
    (job['folder']/'process-exit.json').write_text(json.dumps({'pid':job['process'].pid,'exit_code':code,'cancelled':job['cancel']}))
    cleanup(job)


def poll():
    global _JOB
    job=_JOB
    if not job:return None
    if job['process'].poll() is None:return .2
    _JOB=None
    try:
        if job['cancel']:
            job['scene'].cm_settings.status='Wrinkle update cancelled; original sculpts retained. Retry when ready.'
        else:
            path=job['folder']/'report.json'
            report=json.loads(path.read_text()) if path.is_file() else {}
            if job['process'].returncode or report.get('status')!='complete':
                raise ValueError(report.get('error','Wrinkle worker stopped; original layers retained'))
            publish(job,report)
    except Exception as exc:
        job['scene'].cm_settings.status='Wrinkle update did not apply: '+str(exc)
        (job['folder']/'delivery-error.txt').write_text(traceback.format_exc())
        print('Retained wrinkle update:',exc)
    finally:
        finish(job)
        for window in bpy.context.window_manager.windows:
            for area in window.screen.areas:area.tag_redraw()
    return None


def cancel():
    if _JOB:
        _JOB['cancel']=True
        if _JOB['process'].poll() is None:_JOB['process'].terminate()


def reap():
    for job in list(_REAPING):
        if job['process'].poll() is not None:
            finish(job);_REAPING.remove(job)
    return .2 if _REAPING else None


@persistent
def detach(*_):
    global _JOB
    if _JOB:
        cancel()
        job,_JOB=_JOB,None
        job.pop('scene',None);job.pop('slots',None);job.pop('stamp',None)
        _REAPING.append(job)
        if bpy.app.timers.is_registered(poll):bpy.app.timers.unregister(poll)
        if not bpy.app.timers.is_registered(reap):bpy.app.timers.register(reap,first_interval=.2,persistent=True)


def register():
    for handlers in (bpy.app.handlers.load_pre,bpy.app.handlers.undo_pre,bpy.app.handlers.redo_pre):
        if detach not in handlers:handlers.append(detach)


def unregister():
    detach()
    for handlers in (bpy.app.handlers.load_pre,bpy.app.handlers.undo_pre,bpy.app.handlers.redo_pre):
        if detach in handlers:handlers.remove(detach)


class CM_OT_cancel_wrinkle_update(bpy.types.Operator):
    bl_idname='cm.cancel_wrinkle_update'
    bl_label='Cancel Wrinkle Update'

    def execute(self,context):
        cancel()
        return {'FINISHED'}


CLASSES=(CM_OT_cancel_wrinkle_update,)
