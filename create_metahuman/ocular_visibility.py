"""Geometry-only frontal eye exposure inspection for artists and agents.

Sample the actual evaluated head and globe surfaces. This is a bounded frontal
measurement, not a collision certificate or a test of shader transparency.
"""
from contextlib import contextmanager
from datetime import datetime,timezone
import json
import bpy
import numpy as np
from mathutils import Vector
from mathutils.bvhtree import BVHTree
from .vertex_contract import fingerprint

REPORT_KEY='cm_ocular_visibility_report'


@contextmanager
def preview_neutral(context,objects):
    keys=[]
    try:
        for obj in objects:
            key=obj.data.shape_keys.key_blocks.get('Expression Check') if obj.data.shape_keys else None
            if key:keys.append((key,key.value));key.value=0
        context.view_layer.update()
        yield
    finally:
        for key,value in keys:key.value=value
        context.view_layer.update()


def _surface(context,obj,basis,origin):
    evaluated=obj.evaluated_get(context.evaluated_depsgraph_get());mesh=evaluated.to_mesh()
    try:
        if not mesh.vertices or not mesh.polygons:raise ValueError('Eye inspection requires nonempty evaluated meshes')
        points=np.empty((len(mesh.vertices),3),dtype=float);mesh.vertices.foreach_get('co',points.ravel())
        matrix=np.asarray(evaluated.matrix_world);world=points@matrix[:3,:3].T+matrix[:3,3]
        points=(world-origin)@basis
        mesh.calc_loop_triangles();triangles=np.empty((len(mesh.loop_triangles),3),dtype=np.int32);mesh.loop_triangles.foreach_get('vertices',triangles.ravel())
        tree=BVHTree.FromPolygons(points.tolist(),triangles.tolist(),all_triangles=True)
        return dict(tree=tree,low=points.min(0),high=points.max(0),fingerprint=fingerprint(world))
    finally:evaluated.to_mesh_clear()


def inspect(context,*,columns=65,rows=49):
    """Return exposure counts without changing geometry, materials or the pose.

    Neutral means the same evaluated scene with Expression Check disabled.
    Other artist shape keys and armature poses remain part of the comparison.
    """
    s,f=context.scene.cm_settings,context.scene.cm_features
    head=s.result;eyes=[f.eye_left,f.eye_right]
    if context.mode!='OBJECT':raise ValueError('Finish the current edit before inspecting eye exposure')
    if not head or any(not eye for eye in eyes):raise ValueError('Choose a head and both eye meshes in Anatomy')
    if len({head,*eyes})!=3:raise ValueError('Eye inspection requires distinct head and globe meshes')
    if any(obj.type!='MESH' for obj in (head,*eyes)):raise ValueError('Eye inspection requires mesh objects')
    if not (9<=columns<=257 and 9<=rows<=257):raise ValueError('Eye sampling dimensions must be between 9 and 257')
    from . import resource_budget
    resource_budget.require('frontal eye inspection',512*1024**2)
    unit_scale=float(context.scene.unit_settings.scale_length)
    front=np.asarray({'X':(1,0,0),'-X':(-1,0,0),'Y':(0,1,0),'-Y':(0,-1,0)}[f.front_axis],dtype=float)
    basis=np.column_stack((np.cross([0.,0.,1.],front),-front,[0.,0.,1.]));origin=np.asarray(head.matrix_world.translation,dtype=float)
    objects=[head,*eyes]
    current=[_surface(context,obj,basis,origin) for obj in objects]
    with preview_neutral(context,objects):neutral=[_surface(context,obj,basis,origin) for obj in objects]
    low=np.min([q['low'] for q in [*current,*neutral]],axis=0);high=np.max([q['high'] for q in [*current,*neutral]],axis=0)
    extent=float(np.linalg.norm(high-low))
    if not np.isfinite(extent) or extent<=0:raise ValueError('Eye inspection requires a finite character extent')
    tolerance=max(extent*1e-6,1e-8);start=float(low[1]-extent);distance=float(high[1]-start+extent);direction=Vector((0,1,0))
    def visible(head_surface,eye_surface,x,z):
        ray=Vector((float(x),start,float(z)))
        hit=eye_surface['tree'].ray_cast(ray,direction,distance)
        if hit[0] is None:return False,None
        skin=head_surface['tree'].ray_cast(ray,direction,distance)
        if skin[0] is None:return True,None
        clearance=float(skin[3]-hit[3])
        return clearance>tolerance,clearance
    result=[]
    for i,eye in enumerate(eyes,1):
        lo=np.minimum(current[i]['low'],neutral[i]['low']);hi=np.maximum(current[i]['high'],neutral[i]['high'])
        if np.any(hi[[0,2]]<=lo[[0,2]]):raise ValueError('Eye projection has zero area')
        counts=[0,0];maximum=0.;no_skin=0
        for z in np.linspace(lo[2],hi[2],rows):
            for x in np.linspace(lo[0],hi[0],columns):
                before,_=visible(neutral[0],neutral[i],x,z)
                after,gap=visible(current[0],current[i],x,z)
                counts[0]+=int(before);counts[1]+=int(after)
                if after:
                    if gap is None:no_skin+=1
                    else:maximum=max(maximum,gap)
        result.append(dict(object=eye.name,neutral_visible_samples=counts[0],current_visible_samples=counts[1],
            visible_ratio=counts[1]/counts[0] if counts[0] else None,maximum_visible_depth_m=maximum*unit_scale,
            samples_without_skin_hit=no_skin,grid=[columns,rows]))
    pose='Neutral'
    if f.expression_active:
        from . import expressions
        data=expressions.bank(f);pose=data['poses'][f.expression_index%len(data['poses'])]['name']
    return dict(schema=1,method='FRONTAL_EVALUATED_SURFACE_RAYS',created=datetime.now(timezone.utc).isoformat(),
        pose=pose,strength=float(f.expression_strength) if f.expression_active else 0.,front_axis=f.front_axis,
        tolerance_m=tolerance*unit_scale,eyes=result,
        current_geometry={obj.name:row['fingerprint'] for obj,row in zip(objects,current)},
        neutral_geometry={obj.name:row['fingerprint'] for obj,row in zip(objects,neutral)},
        limitations=['Frontal finite sampling; small gaps can fall between samples',
            'Geometry only; opacity, eyelashes and shading are not evaluated',
            'Zero exposed samples does not certify closure, collision safety or animation quality'])


def draw_last(layout,context):
    raw=context.scene.get(REPORT_KEY,'')
    if not raw:return
    try:report=json.loads(raw)
    except (TypeError,ValueError):return
    box=layout.box();box.label(text='Last frontal inspection: '+report['pose'])
    for i,row in enumerate(report['eyes'],1):
        ratio=row['visible_ratio'];text=f'{ratio:.1%} of neutral exposure' if ratio is not None else 'No exposed neutral samples'
        box.label(text=f'Eye {i}: {text}')
    box.label(text='Sampled geometry; inspect motion visually too.',icon='INFO')


class CM_OT_inspect_eye_exposure(bpy.types.Operator):
    bl_idname='cm.inspect_eye_exposure';bl_label='Inspect Eye Exposure'
    bl_description='Measure the previewed eyes against the actual skin surface from the configured front view'
    def execute(self,context):
        try:
            report=inspect(context);context.scene[REPORT_KEY]=json.dumps(report)
            context.scene.cm_settings.status='Frontal eye inspection recorded. This is sampled geometry, not an animation certificate.'
            return {'FINISHED'}
        except Exception as exc:
            context.scene.cm_settings.status=str(exc);self.report({'ERROR'},str(exc));return {'CANCELLED'}


CLASSES=(CM_OT_inspect_eye_exposure,)
