"""Reference-pose limb alignment using corresponding skin and joint pivots."""
from pathlib import Path
import hashlib
import json
import re
import numpy as np
from mathutils import Vector
from . import mesh_ops as m

KEY='cm_pose_reference_skeleton'
CHAINS=(('upperarm','lowerarm','hand'),('thigh','calf','foot'))
REGION_CHAINS=(('clavicle','upperarm','lowerarm','hand'),('thigh','calf','foot'))


def coordinate_signature(obj):
    points=np.empty((len(obj.data.vertices),3),dtype=np.float64)
    obj.data.vertices.foreach_get('co',points.ravel())
    return hashlib.sha256(np.rint(points*1e6).astype('<i8').tobytes()).hexdigest()


def capture_skeleton(obj, world=False):
    armature=obj.find_armature()
    if not armature:return None
    matrix=armature.matrix_world if world else obj.matrix_world.inverted()@armature.matrix_world
    return {'bones':{bone.name:{'head':list(matrix@bone.head),
                   'parent':bone.parent.name if bone.parent else ''}
                    for bone in armature.pose.bones}}


def reference_skeleton(template):
    if template.get(KEY):return json.loads(template[KEY])
    live=capture_skeleton(template)
    if live:return live
    if template.get('cm_library_role')=='body':
        from . import guided
        library=guided.character_library()
        path=Path(library).parent/'body-pose.json' if library else None
        if path and path.is_file():
            data=json.loads(path.read_text(encoding='utf-8'))
            if data.get('coordinates')==coordinate_signature(template) and data.get('topology')==m.topology(template):
                return data
    return None


def _name(name):return name.rsplit(':',1)[-1].lower()


def _members(root, bones):
    side=root[-1]
    chain=next(chain for chain in REGION_CHAINS if root[:-2] in chain)
    descendants=chain[chain.index(root[:-2]):]
    prefixes=list(descendants)
    if 'hand' in descendants:prefixes+=['index','middle','ring','pinky','thumb','metacarpal']
    if 'foot' in descendants:prefixes+=['ball','toe']
    result=set()
    for name in bones:
        if name.endswith('_'+side) and any(name==part+'_'+side or name.startswith(part+'_') for part in prefixes):
            result.add(name)
        current=name;seen=set()
        while current and current not in seen:
            if current==root:result.add(name);break
            seen.add(current);current=bones.get(current,{}).get('parent','')
    return result


_PRIMARY_JOINT=re.compile(r'^(?:pelvis|head|spine_\d+|neck_\d+)$|^(?:clavicle|upperarm|lowerarm|hand|thigh|calf|foot|ball|toe)_[lr]$|^(?:index|middle|ring|pinky|thumb)_(?:\d+|metacarpal)_[lr]$')


def _anatomical_support(points,bones,roots):
    """Smooth primary-skeleton territory, independent of pose-helper labels.

    Distributed corrective influences are retained inside the appropriate
    limb territory and attenuated where a competing primary segment is nearer.
    This geometric ownership prior requires a canonical reference skeleton;
    it is not an anatomical classifier for an unregistered source scan.
    """
    primary={name for name,data in bones.items() if _PRIMARY_JOINT.fullmatch(name) and 'head' in data}
    if len(primary)<4:raise ValueError('Anatomical limb regions require the canonical reference skeleton')
    membership={root:_members(root,bones)&primary for root in roots}
    inside={root:np.full(len(points),np.inf) for root in roots};outside={root:np.full(len(points),np.inf) for root in roots}
    segments=[]
    for name in sorted(primary):
        position=np.asarray(bones[name]['head'],dtype=float)
        segments.append((name,position,position))
        parent=bones[name].get('parent','');seen={name}
        while parent and parent not in primary and parent not in seen:
            seen.add(parent);parent=bones.get(parent,{}).get('parent','')
        if parent in primary:segments.append((parent,np.asarray(bones[parent]['head'],dtype=float),position))
    for owner,a,b in segments:
        direction=b-a;length=float(direction@direction)
        fraction=np.clip((points-a)@direction/max(length,1e-30),0,1)
        distance=np.linalg.norm(points-a-fraction[:,None]*direction,axis=1)
        for root in roots:
            destination=inside[root] if owner in membership[root] else outside[root]
            np.minimum(destination,distance,out=destination)
    epsilon=max(float(np.linalg.norm(np.ptp(points,axis=0)))*1e-9,1e-12)
    result={}
    for root in roots:
        if not np.isfinite(inside[root]).all() or not np.isfinite(outside[root]).all():raise ValueError('Reference skeleton lacks limb or surrounding anatomy')
        relative=(outside[root]-inside[root])/np.maximum(np.minimum(inside[root],outside[root]),epsilon)
        fraction=np.clip(.5+.75*relative,0,1)
        result[root]=fraction*fraction*(3-2*fraction)
    return result


def _attachment_fade(template,points,*,protected=None,rings=6):
    """Lock physical open boundaries and blend out through graph distance."""
    if int(rings)!=rings or rings<1:raise ValueError('Choose positive whole attachment collar rings')
    _,weld=np.unique(points,axis=0,return_inverse=True);n=int(weld.max())+1
    template.data.calc_loop_triangles()
    triangles=weld[np.array([tuple(t.vertices) for t in template.data.loop_triangles],dtype=np.int64).reshape(-1,3)]
    edges=np.sort(np.vstack((triangles[:,[0,1]],triangles[:,[1,2]],triangles[:,[2,0]])),axis=1)
    edges,counts=np.unique(edges,axis=0,return_counts=True);edges_valid=edges[:,0]!=edges[:,1]
    boundary=np.zeros(n,bool);boundary[np.unique(edges[(counts==1)&edges_valid])]=True;edges=edges[edges_valid]
    if protected is not None:
        protected=np.asarray(protected,dtype=bool)
        if protected.shape!=(len(points),):raise ValueError('Attachment pins must match template vertex order')
        np.logical_or.at(boundary,weld,protected)
    distance=np.full(n,int(rings),dtype=int);distance[boundary]=0;visited=boundary.copy();front=boundary.copy()
    for step in range(1,int(rings)):
        neighbors=np.unique(edges[np.any(front[edges],axis=1)]);front=np.zeros(n,bool);front[neighbors]=True;front&=~visited
        distance[front]=step;visited|=front
    fade=distance.astype(float)/rings;fade=fade*fade*(3-2*fade)
    return fade[weld]


def subtree_weights(template,roots,*,mode='DEFORMATION',protected=None,protection_rings=6):
    """Collect native limb subtrees, including corrective and distal bones.

    ANATOMICAL mode bounds distributed corrective influences using primary
    reference-skeleton territories and locks physical open attachment edges.
    Optional protected vertex families receive the same graph collar. Helper
    labels alone are not anatomy. Closed UV seams are welded before selection.
    DEFORMATION retains the complete families used by pose alignment.
    TERRITORY selects primary-skeleton ownership without multiplying by
    deformation/corrective weights, while retaining attachment protection.
    Namespaces are normalized just as in pose alignment. This lets solvers
    select a complete calf/ankle/foot or forearm/hand from anatomy rather than
    hand-authored coordinate regions. Source geometry is never involved.
    """
    roots=[_name(root) for root in roots]
    if mode not in {'DEFORMATION','ANATOMICAL','TERRITORY'}:raise ValueError('Choose DEFORMATION, ANATOMICAL or TERRITORY influence mode')
    if mode=='DEFORMATION' and protected is not None:raise ValueError('Attachment protection requires anatomical or territory mode')
    if not roots or len(set(roots))!=len(roots):raise ValueError('Choose distinct rig subtree roots')
    if any(root[:-2] not in {part for chain in REGION_CHAINS for part in chain} for root in roots):raise ValueError('Choose a clavicle, upperarm, lowerarm, hand, thigh, calf or foot subtree root')
    skeleton=reference_skeleton(template) or {}
    bones={_name(name):dict(data,parent=_name(data.get('parent',''))) for name,data in skeleton.get('bones',{}).items()}
    names={group.index:_name(group.name) for group in template.vertex_groups}
    all_names={**{name:{} for name in names.values()},**bones}
    if any(root not in all_names for root in roots):raise ValueError('Requested root is absent from the template rig')
    if mode=='TERRITORY':
        points=np.empty((len(template.data.vertices),3),dtype=np.float64);template.data.vertices.foreach_get('co',points.ravel())
        support=_anatomical_support(points,bones,roots)
        fade=_attachment_fade(template,points,protected=protected,rings=protection_rings)
        return {root:support[root]*fade for root in roots}
    members={root:_members(root,all_names) for root in roots}
    membership={index:[root for root in roots if name in members[root]] for index,name in names.items()}
    result={root:np.zeros(len(template.data.vertices)) for root in roots}
    for vertex in template.data.vertices:
        for weight in vertex.groups:
            for root in membership.get(weight.group,()):result[root][vertex.index]+=weight.weight
    result={root:np.clip(values,0,1) for root,values in result.items()}
    if mode=='ANATOMICAL':
        points=np.empty((len(template.data.vertices),3),dtype=np.float64);template.data.vertices.foreach_get('co',points.ravel())
        support=_anatomical_support(points,bones,roots)
        fade=_attachment_fade(template,points,protected=protected,rings=protection_rings)
        result={root:result[root]*support[root]*fade for root in roots}
    return result


def attachment_region(template,values,*,protected=None,rings=4,core_threshold=.25):
    """Extend a skeleton territory with a physical, geodesic blend collar.

    A fitting region needs room on both sides of an attachment. Deformation
    ownership alone can freeze the torso neighbor of an erroneous armpit.
    This only selects vertices; it neither changes geometry nor relaxes pins.
    """
    if not isinstance(rings,int) or not 0<=rings<=32:raise ValueError('Choose between zero and 32 attachment rings')
    if not 0<core_threshold<1:raise ValueError('Choose a territory core threshold between zero and one')
    points=np.empty((len(template.data.vertices),3),dtype=np.float64);template.data.vertices.foreach_get('co',points.ravel())
    values=np.asarray(values,dtype=float)
    if values.shape!=(len(points),) or not np.isfinite(values).all() or np.any((values<0)|(values>1)):raise ValueError('Attachment territory must contain one finite weight per vertex')
    locked=np.zeros(len(points),bool) if protected is None else np.asarray(protected,dtype=bool)
    if locked.shape!=values.shape:raise ValueError('Attachment pins must match the template')
    _,weld=np.unique(points,axis=0,return_inverse=True);n=int(weld.max())+1
    frozen=np.zeros(n,bool);np.logical_or.at(frozen,weld,locked)
    ownership=np.zeros(n);np.maximum.at(ownership,weld,values);ownership[frozen]=0
    if rings:
        template.data.calc_loop_triangles();tri=weld[np.array([t.vertices[:] for t in template.data.loop_triangles],dtype=np.int64)]
        edges=np.unique(np.sort(np.vstack((tri[:,[0,1]],tri[:,[1,2]],tri[:,[2,0]])),axis=1),axis=0)
        distance=np.full(n,rings+1,dtype=int);front=(ownership>=core_threshold)&~frozen;distance[front]=0
        for step in range(1,rings+1):
            neighbors=np.unique(edges[np.any(front[edges],axis=1)])
            front=np.zeros(n,bool);front[neighbors]=True;front&=(distance>step)&~frozen;distance[front]=step
        blend=np.clip(1-distance/(rings+1),0,1);blend=blend*blend*(3-2*blend)
        ownership=np.maximum(ownership,blend);ownership[frozen]=0
    return ownership[weld]


def joint_station_protection(template,root,start,end,*,minimum_station):
    """Intersect distal ownership with a station along named reference joints."""
    if not 0<=minimum_station<=1:raise ValueError('Choose a reference station between zero and one')
    skeleton=reference_skeleton(template) or {};bones={_name(name):data for name,data in skeleton.get('bones',{}).items()}
    try:a=np.asarray(bones[_name(start)]['head'],dtype=float);b=np.asarray(bones[_name(end)]['head'],dtype=float)
    except KeyError as exc:raise ValueError('Distal protection requires the named reference joints') from exc
    if a.shape!=(3,) or b.shape!=(3,) or not np.isfinite(a).all() or not np.isfinite(b).all():raise ValueError('Reference protection joints must have finite three-dimensional positions')
    direction=b-a;length=float(direction@direction)
    if length<=1e-20:raise ValueError('Reference protection joints must be distinct')
    points=np.empty((len(template.data.vertices),3),dtype=np.float64);template.data.vertices.foreach_get('co',points.ravel())
    station=(points-a)@direction/length
    territory=subtree_weights(template,[root],mode='TERRITORY')[root]
    return (territory>.05)&(station>=minimum_station)


def distal_arm_protection(template,side,*,proximal_station=.8):
    """Protect forearm territory beyond the reference upper-arm station.

    Smooth skeleton territories have nonzero tails near the axilla. They
    must not turn an upper-arm attachment into a frozen forearm vertex.
    The reference shoulder-to-elbow station excludes those proximal tails.
    """
    if side not in {'l','r'} or not 0<=proximal_station<=1:raise ValueError('Choose a side and upper-arm station between zero and one')
    return joint_station_protection(template,'lowerarm_'+side,'upperarm_'+side,'lowerarm_'+side,minimum_station=proximal_station)


def distal_leg_protection(template,side,*,calf_station=.2):
    """Keep the calf/foot fixed while leaving the knee transition movable."""
    if side not in {'l','r'}:raise ValueError('Choose left or right anatomy')
    return joint_station_protection(template,'calf_'+side,'calf_'+side,'foot_'+side,minimum_station=calf_station)


def pelvis_attachment_frame(template):
    """World-space section origin and lateral/up axes from the reference rig."""
    skeleton=reference_skeleton(template) or {};bones={_name(name):data for name,data in skeleton.get('bones',{}).items()}
    def position(name):
        if name not in bones:raise ValueError('Pelvis attachment registration requires pelvis, spine and bilateral thigh reference joints')
        value=np.asarray(template.matrix_world@Vector(bones[name]['head']),dtype=float)
        if value.shape!=(3,) or not np.isfinite(value).all():raise ValueError('Nonfinite attachment reference joint')
        return value
    left,right=position('thigh_l'),position('thigh_r');lateral=left-right;length=float(np.linalg.norm(lateral))
    if length<=1e-12:raise ValueError('Bilateral thigh reference joints must be distinct')
    lateral/=length;up=position('spine_01')-position('pelvis');up-=lateral*float(up@lateral);length=float(np.linalg.norm(up))
    if length<=1e-12:raise ValueError('Pelvis and spine must define an independent up direction')
    return np.stack(((left+right)*.5,lateral,up/length))


def _attachment_territory(template,root,distal,protected,protection_rings,attachment_rings):
    if protected is not None:
        protected=np.asarray(protected,dtype=bool)
        if protected.shape!=distal.shape:raise ValueError('Protected vertices must match the reference body')
        distal=distal|protected
    values=subtree_weights(template,[root],mode='TERRITORY',protected=distal,protection_rings=protection_rings)[root]
    expanded=attachment_region(template,values,protected=distal,rings=attachment_rings)
    points=np.empty((len(template.data.vertices),3),dtype=np.float64);template.data.vertices.foreach_get('co',points.ravel())
    return np.maximum(values,expanded*_attachment_fade(template,points,protected=distal,rings=protection_rings))


def upperarm_shoulder_region(template,side,*,protected=None,protection_rings=2,attachment_rings=4):
    """Select the shoulder/upper arm by ownership, locking distal anatomy.

    Skin deformation weights are not region membership: tiny corrective
    tails must not become distant fit vertices or near-zero movement caps.
    Clavicle territory includes the attachment; lower-arm territory and its
    graph collar keep the already fitted forearm and hand fixed.
    """
    if side not in {'l','r'}:raise ValueError('Choose left or right anatomy')
    distal=distal_arm_protection(template,side)
    return _attachment_territory(template,'clavicle_'+side,distal,protected,protection_rings,attachment_rings)


def thigh_knee_region(template,side,*,protected=None,protection_rings=2,attachment_rings=4):
    """Select a thigh, hip blend and knee, preserving distal calf/foot anatomy.

    Reference joints define the knee transition, independently of skin-weight
    tails, source dimensions or character-specific vertex positions.
    """
    if side not in {'l','r'}:raise ValueError('Choose left or right anatomy')
    return _attachment_territory(template,'thigh_'+side,distal_leg_protection(template,side),protected,protection_rings,attachment_rings)


def thigh_surface_region(template,side,*,protected=None):
    """Dominant leg surface without the shared pelvis/hip attachment collar.

    Ambiguous ownership between a leg and competing primary bones is fixed.
    This distinguishes local surface cleanup from attachment reconstruction;
    it uses the canonical rig, not source coordinates or character IDs.
    """
    if side not in {'l','r'}:raise ValueError('Choose left or right anatomy')
    skeleton=reference_skeleton(template) or {};bones={_name(name):dict(data,parent=_name(data.get('parent',''))) for name,data in skeleton.get('bones',{}).items()}
    points=np.empty((len(template.data.vertices),3),dtype=np.float64);template.data.vertices.foreach_get('co',points.ravel())
    root='thigh_'+side;ownership=_anatomical_support(points,bones,[root])[root]
    dominant=np.clip((ownership-.5)*2,0,1)
    distal=distal_leg_protection(template,side)
    if protected is not None:
        protected=np.asarray(protected,dtype=bool)
        if protected.shape!=distal.shape:raise ValueError('Protected vertices must match the reference body')
        distal|=protected
    return dominant*_attachment_fade(template,points,protected=distal,rings=2)


def knee_region(template,side,*,protected=None):
    """A reference-joint transition band; keep the hip and distal leg fixed."""
    if side not in {'l','r'}:raise ValueError('Choose left or right anatomy')
    skeleton=reference_skeleton(template) or {};bones={_name(name):data for name,data in skeleton.get('bones',{}).items()}
    try:a=np.asarray(bones['thigh_'+side]['head'],dtype=float);b=np.asarray(bones['calf_'+side]['head'],dtype=float)
    except KeyError as exc:raise ValueError('Knee refinement requires thigh and calf reference joints') from exc
    direction=b-a;length=float(direction@direction)
    if length<=1e-20:raise ValueError('Knee reference joints must be distinct')
    points=np.empty((len(template.data.vertices),3),dtype=np.float64);template.data.vertices.foreach_get('co',points.ravel())
    station=(points-a)@direction/length
    fade=np.minimum(np.clip((station-.55)/.25,0,1),np.clip((1.35-station)/.2,0,1));fade=fade*fade*(3-2*fade)
    return thigh_knee_region(template,side,protected=protected,attachment_rings=0)*fade


def align(context, obj, template, progress=None):
    reference=m.coordinates(template)
    output=m.evaluated_coordinates(context,obj)
    if output.shape!=reference.shape or len(output)!=len(obj.data.vertices):
        raise ValueError('Set the pose on the base body before topology-changing modifiers or body detail')
    skeleton=reference_skeleton(template)
    bones={_name(name):{'head':data['head'],'parent':_name(data['parent'])}
           for name,data in (skeleton or {}).get('bones',{}).items()}
    names={group.index:_name(group.name) for group in obj.vertex_groups}
    all_names={**{name:{} for name in names.values()},**bones}
    roots=[bone+'_'+side for side in ('l','r') for chain in CHAINS for bone in chain]
    members={root:_members(root,all_names) for root in roots}
    exact={root:np.zeros(len(output)) for root in roots}
    subtree={root:np.zeros(len(output)) for root in roots}
    membership={index:[root for root in roots if name in members[root]] for index,name in names.items()}
    for vertex in obj.data.vertices:
        for weight in vertex.groups:
            name=names.get(weight.group)
            if name in exact:exact[name][vertex.index]+=weight.weight
            for root in membership.get(weight.group,()):subtree[root][vertex.index]+=weight.weight
    for root in roots:subtree[root]=np.clip(subtree[root],0,1)
    processed=0;joints_used=0;angles=[]
    for step,root in enumerate(roots):
        if progress:progress(step,len(roots))
        weight=exact[root]
        if np.count_nonzero(weight>.25)<8:
            chain=next(chain for chain in CHAINS if root[:-2] in chain)
            index=chain.index(root[:-2])
            weight=subtree[root]-(subtree[chain[index+1]+'_'+root[-1]] if index+1<len(chain) else 0)
        ids=np.flatnonzero(weight>max(.15,float(weight.max())*.65))
        if len(ids)<8:continue
        w=weight[ids]**2;w/=w.sum()
        a,b=reference[ids],output[ids]
        ca,cb=w@a,w@b
        x,y=a-ca,b-cb
        if np.linalg.matrix_rank(x)<2 or np.linalg.matrix_rank(y)<2:continue
        u,_,vt=np.linalg.svd((x*w[:,None]).T@y)
        sign=np.ones(3);sign[-1]=np.sign(np.linalg.det(u@vt))
        rotation=(u*sign)@vt
        if root in bones:
            pivot_reference=np.asarray(template.matrix_world@Vector(bones[root]['head']))
            affine,_,rank,_=np.linalg.lstsq(x*np.sqrt(w[:,None]),y*np.sqrt(w[:,None]),rcond=1e-5)
            pivot=(pivot_reference-ca)@affine+cb if rank==3 else cb
            if rank==3:joints_used+=1
        else:
            # Legacy templates keep a conservative proximal surface estimate.
            lateral=int(np.argmax(np.ptp(reference[:,:2],axis=0)))
            near=np.abs(a[:,lateral]-np.mean(reference[:,lateral])) if root.startswith(('upperarm','lowerarm','hand')) else -a[:,2]
            pivot=b[np.argmin(near)]
        angle=float(np.arccos(np.clip((np.trace(rotation)-1)*.5,-1,1)))
        if angle>1e-6:
            transformed=(output-pivot)@rotation.T+pivot
            output+=(transformed-output)*subtree[root][:,None]
        angles.append(angle);processed+=1
    if not processed:
        raise ValueError('A-pose needs the reference body skin groups. Preserve upperarm, lowerarm, hand, thigh, calf and foot weights.')
    return output,{'segments':processed,'joint_pivots':joints_used==processed,'reference_joint_segments':joints_used,
                   'largest_rotation_degrees':float(np.degrees(max(angles)))}


def draw(layout,context):
    s,f=context.scene.cm_settings,context.scene.cm_features
    if s.part not in {'BODY','FULL_BODY'}:return
    obj=f.body_result or s.result
    if not obj:return
    row=layout.row(align=True)
    row.enabled=context.mode=='OBJECT' and not f.body_editing and not f.full_sculpt and not f.body_detail and not(s.part=='BODY' and s.detail)
    row.operator('cm.feature',text='Force A Pose').action='A_POSE'
    if obj.data.shape_keys and obj.data.shape_keys.key_blocks.get('Force A Pose'):
        row.operator('cm.feature',text='Restore Pose').action='A_POSE_RESET'
    layout.label(text='Set limb pose before creating body detail.')
