"""Artist/agent entry point for marked source outer-ear reconstruction."""
import numpy as np
import bpy
from . import eye_fit_jobs as jobs,mesh_ops as m,workflow


def prepare(context,*,method='INTRINSIC'):
    from . import character_project,setup_paths
    if method not in {'INTRINSIC','PROJECTED'}:raise ValueError('Unknown ear reconstruction method')
    s,f=context.scene.cm_settings,context.scene.cm_features
    character_project.preflight(context,s,f);setup_paths.resolve(f)
    for obj,label in ((s.template,'Head template'),(s.result,'Accepted head'),(s.source,'Source')):m.check_mesh(obj,label)
    if m.topology(s.template)!=m.topology(s.result):raise ValueError('Ear reconstruction requires unchanged MetaHuman head topology')
    names=('CM_Fit_Ear_L','CM_Fit_Ear_R');source,st,_,images,_=jobs._source(context,s.source,True,group_names=names)
    current=m.evaluated_coordinates(context,s.result);reference=m.coordinates(s.template);front=np.array({'X':(1,0,0),'-X':(-1,0,0),'Y':(0,1,0),'-Y':(0,-1,0)}[f.front_axis],dtype=float);basis=np.column_stack((np.cross([0.,0.,1.],front),-front,[0.,0.,1.]))
    s.result.data.calc_loop_triangles();tri=np.array([t.vertices[:] for t in s.result.data.loop_triangles],dtype=np.int64)
    data=dict(current=current,parameter_reference=reference,triangles=tri,source=source,source_triangles=st,eye_basis=basis)
    cores=[]
    for side in ('L','R'):
        groups=[g.name for g in s.template.vertex_groups if g.name.startswith('FACIAL_'+side+'_Ear')]
        if not groups:raise ValueError('The template needs its native MetaHuman ear skinning groups')
        weights=sum((workflow.group_weights(s.template,name) for name in groups),np.zeros(len(reference)))
        if not np.any(weights>.95):raise ValueError('The native ear core could not be identified')
        cores.append(weights)
    cores.sort(key=lambda weights:float((current[weights>.95]@basis)[:,0].mean()))
    sources=[workflow.group_weights(s.source,name) for name in names]
    if any(not np.any(w>.25) for w in sources):raise ValueError('Mark both source ear surfaces before reconstructing ears')
    sources.sort(key=lambda weights:float((source[weights>.25]@basis)[:,0].mean()))
    has=[s.result.vertex_groups.get(name) is not None for name in names]
    if any(has) and not all(has):raise ValueError('Mark both result ear regions, including the supporting head collar')
    regions=[workflow.group_weights(s.result,name) for name in names] if all(has) else [np.asarray(w>0,dtype=float) for w in cores]
    if any(not np.any(w>.05) for w in regions):raise ValueError('Both result ear regions must include a supporting collar')
    regions.sort(key=lambda weights:float((current[weights>.05]@basis)[:,0].mean()))
    midpoint=float((current@basis)[:,0].mean());signs=[]
    for i,(core,region,patch) in enumerate(zip(cores,regions,sources)):
        sign=float(np.sign((current[core>.95]@basis)[:,0].mean()-midpoint));signs.append(sign)
        data['ear_weights_'+str(i)]=core;data['ear_region_'+str(i)]=region;data['source_ear_'+str(i)]=patch;data['ear_sign_'+str(i)]=np.array(sign)
    if set(signs)!={-1.,1.}:raise ValueError('The native ear regions do not define distinct head sides')
    body=f.body_result if f.body_result and f.body_result!=s.result else None
    if body:data['body_current']=m.evaluated_coordinates(context,body)
    eyes=list(dict.fromkeys(o for o in (f.eye_left,f.eye_right) if o))
    return data,eyes,images,dict(ear_surface=True,ear_method=method),body


def start(context,*,method='INTRINSIC'):
    data,eyes,images,options,body=prepare(context,method=method)
    return jobs.launch(context,data,eyes,images,options,kind='EARS',body=body)


def status():return jobs.status()
def cancel():jobs.cancel()


class CM_OT_fit_source_ears(bpy.types.Operator):
    bl_idname='cm.fit_source_ears';bl_label='Reconstruct Source Ears'
    bl_description='Map complete curved source ears and their supporting head patches with native geometric validation'
    method:bpy.props.EnumProperty(name='Method',items=(('INTRINSIC','Curved Surface','Preserve the full source ear surface and reconstruct its supporting collar'),('PROJECTED','Projected Outline','Legacy side projection for intentionally flat or limited sources')),default='INTRINSIC')
    def invoke(self,context,event):return context.window_manager.invoke_props_dialog(self)
    def draw(self,context):
        self.layout.prop(self,'method')
        self.layout.label(text='Source groups: CM_Fit_Ear_L and CM_Fit_Ear_R')
        self.layout.label(text='Result regions include the ears and supporting collar.')
        self.layout.label(text='Use Regional Surface Fit to mark or edit these regions.')
    def execute(self,context):
        try:start(context,method=self.method);return {'FINISHED'}
        except Exception as exc:context.scene.cm_settings.status=str(exc);self.report({'ERROR'},str(exc));return {'CANCELLED'}


CLASSES=(CM_OT_fit_source_ears,)
