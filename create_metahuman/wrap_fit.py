"""Immutable surface-fit inputs and one-use guarded publication buffers."""
from contextlib import contextmanager
import hashlib,json
from pathlib import Path
import bpy,numpy as np
from mathutils import Matrix,Vector
from . import mesh_ops as m,scan_jobs,wrap_proxy

_PREPARED=None
SETTINGS=('iterations','projection_strength','smoothing','max_distance','landmark_radius','landmark_regularization','character_name')
FEATURES=('front_axis','wrap_method','wrap_quality','conservative','is_mask','preserve_mouth_interior','retain_generated_ear_detail')
FLAGS=('cm_gnm','cm_front_axis','cm_template_library','cm_library_role')


def raw_stamp(obj):
    return hashlib.sha256(m.coordinates(obj).tobytes()+m.topology(obj).encode()).hexdigest()


def digest(path):
    with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()


def snapshot(context,folder):
    from .features import generated_head_update
    from . import resource_budget
    s,f=context.scene.cm_settings,context.scene.cm_features
    part='HEAD' if generated_head_update(s) else s.part
    proxy=wrap_proxy.source(context,s,f)
    selected=[s.source,s.template]+([f.body_template] if part=='FULL_BODY' else [])
    if proxy!=s.source:selected.append(proxy)
    resource_budget.require('background surface fit',768*resource_budget.MIB+3*resource_budget.meshes(selected))
    resource_budget.require_disk('background surface fit',folder,3*resource_budget.meshes(selected)+128*resource_budget.MIB)
    folder.mkdir(parents=True,exist_ok=False)
    copies={};shadows=[]
    try:
        # Evaluate a private shadow at the same fit-frame transform that the
        # generated-head publication wrapper will use. Parent/constraint links
        # are read in this scene; only static geometry enters the worker.
        shadow=s.source.copy();context.scene.collection.objects.link(shadow);shadows.append(shadow)
        placement=s.source.get('cm_head_placement') if generated_head_update(s) else None
        if placement:
            frame=json.loads(placement)
            if m.transform_state(s.source)==frame['placed']:m.restore_transform(shadow,frame['fit'])
            else:shadow.matrix_world=Matrix.Translation(-Vector(frame['movement']))@shadow.matrix_world
        context.view_layer.update()
        frozen=scan_jobs.snapshot(context,shadow,'CM Fit Source');context.scene.collection.objects.link(frozen)
        if not shadow.parent and not shadow.constraints:m.restore_transform(frozen,m.transform_state(shadow))
        copies['source']=frozen
        frozen['_cm_fit_original_vertices']=len(s.source.data.vertices)
        for role,obj in [('template',s.template),*([('body',f.body_template)] if part=='FULL_BODY' else [])]:
            copied=m.duplicate(context,obj,'CM Fit '+role)
            world=obj.matrix_world.copy();copied.parent=None;copied.constraints.clear();copied.animation_data_clear()
            if obj.parent or obj.constraints:copied.matrix_world=world
            copied.id_properties_clear();copied.data.id_properties_clear();copied.data.materials.clear()
            copies[role]=copied
            for key in FLAGS:
                if key in obj:copied[key]=obj[key]
        for key in FLAGS:
            if key in s.source:frozen[key]=s.source[key]
        if proxy!=s.source:
            copied=scan_jobs.snapshot(context,proxy,'CM Fit Proxy');context.scene.collection.objects.link(copied)
            copied.parent=frozen;copied.matrix_parent_inverse.identity();copied.matrix_basis.identity();copies['proxy']=copied
        context.view_layer.update()
        pairs=[]
        for pair in s.landmarks:
            if not pair.source or not pair.template:
                if part=='FULL_BODY':raise ValueError('Complete each landmark pair before wrapping')
                continue
            allowed=(s.template,f.body_template) if part=='FULL_BODY' else (s.template,)
            if pair.source.parent!=s.source or pair.template.parent not in allowed:raise ValueError('Landmarks belong to another Source/Template')
            # Evaluate the original marker transform at the future fit frame;
            # multiplying its world position by an inverse loses precision even
            # when the source frame has not moved.
            marker=pair.source.copy();context.scene.collection.objects.link(marker)
            marker.parent=shadow;shadows.append(marker);context.view_layer.update()
            pairs.append(dict(name=pair.name,source=list(marker.matrix_world.translation),
                template=list(pair.template.matrix_world.translation),role='body' if pair.template.parent!=s.template else 'template'))
        request=dict(schema=1,part=part,objects={key:obj.name for key,obj in copies.items()},
            settings={key:getattr(s,key) for key in SETTINGS},features={key:getattr(f,key) for key in FEATURES},
            units=context.scene.unit_settings.scale_length,pairs=pairs,
            source_stamp=scan_jobs.geometry_stamp(context,frozen),template_stamp=raw_stamp(s.template),
            body_stamp=raw_stamp(f.body_template) if part=='FULL_BODY' else None)
        bpy.data.libraries.write(str(folder/'input.blend'),set(copies.values()),compress=False)
        return request
    finally:
        for obj in reversed(list(copies.values())):
            data=obj.data;bpy.data.objects.remove(obj,do_unlink=True)
            if not data.users:bpy.data.meshes.remove(data)
        for obj in reversed(shadows):bpy.data.objects.remove(obj,do_unlink=True)
        context.view_layer.update()


def load(folder):
    report=json.loads((folder/'result.json').read_text())
    if report.get('status')!='complete':raise ValueError(report.get('error','Surface fit failed'))
    if report.get('schema')!=1 or digest(folder/'fit.npz')!=report.get('sha256'):raise ValueError('Surface-fit result is incomplete or changed')
    with np.load(folder/'fit.npz',allow_pickle=False) as arrays:
        points={name:arrays[name].copy() for name in arrays.files}
    if any(value.ndim!=2 or value.shape[1]!=3 or not np.isfinite(value).all() for value in points.values()):raise ValueError('Invalid surface-fit buffers')
    report['arrays']=points
    return report


@contextmanager
def prepared(context,payload):
    global _PREPARED
    if _PREPARED is not None:raise ValueError('A surface fit is already being published')
    s,f=context.scene.cm_settings,context.scene.cm_features
    record=dict(scene=context.scene,source=s.source,template=s.template,body=f.body_template,payload=payload,used=False)
    _PREPARED=record
    try:
        yield
        if not record['used']:raise ValueError('The completed fit was not consumed by its original wrap action')
    finally:_PREPARED=None


def consume(context,settings,kind):
    record=_PREPARED
    if record is None:return None
    payload=record['payload'];contract=payload['contract'];f=context.scene.cm_features
    if record['used'] or context.scene!=record['scene'] or settings.source!=record['source'] or settings.template!=record['template']:
        raise ValueError('Surface-fit publication target changed')
    if kind!=payload['kind'] or settings.part!=contract['part']:raise ValueError('Surface-fit publication mode changed')
    if raw_stamp(settings.template)!=contract['template_stamp'] or scan_jobs.geometry_stamp(context,settings.source)!=contract['source_stamp']:
        raise ValueError('Surface-fit frame changed before publication')
    if kind=='JOINT' and (f.body_template!=record['body'] or raw_stamp(f.body_template)!=contract['body_stamp']):raise ValueError('Body template changed before publication')
    record['used']=True
    if kind=='JOINT':
        from .guided import signature
        return payload['arrays']['head'],payload['arrays']['body'],payload['report'],signature(settings.source)
    return dict(points=payload['arrays']['points'],surface_rms=payload['surface_rms'],properties=payload['properties'])
