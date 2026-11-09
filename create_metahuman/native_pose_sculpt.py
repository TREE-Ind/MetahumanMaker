"""Author a private canonical head correction from a captured native pose."""
import hashlib,json,uuid
from pathlib import Path
import bpy
import numpy as np
from .calibration_math import canonical_offsets

ITEMS={}
OWNER='cm_native_pose_session'
def digest(path):
    with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()

def pose_items(settings,context):
    try:
        rows=json.loads((Path(bpy.path.abspath(settings.capture_folder))/'capture.json').read_text())['poses']
        choices=[];blink_choices=[];closed_choices=[]
        for i,row in enumerate(rows):
            if len(row['controls'])!=1:continue
            name,value=next(iter(row['controls'].items()));name=name.removeprefix('CTRL_expressions.')
            if name.startswith('eye') and name[-1] in 'LR' and 1/64<=float(value)<=1:
                from .native_face_mixer import label
                closed=name.startswith('eyeBlink') and float(value)==1.
                item=(str(i),label(name)+' '+format(float(value)*100,'g')+'%'+(' - Closed Eye' if closed else ''),'Changes full closure; neutral and quarter-strength anchors are retained' if closed else 'Edit a private copy; partial blink retains neutral and full closure')
                (closed_choices if closed else blink_choices if name.startswith('eyeBlink') else choices).append(item)
        rows=choices+blink_choices
        if rows or closed_choices:rows.append(('SELECT','Choose Captured Eye Pose','Choose a pose from this capture; earlier capture selections are retained only with their original data'))
        # Preserve existing dynamic enum ordinals, including SELECT. Closed
        # endpoints were previously omitted and are appended after old choices.
        rows.extend(closed_choices)
    except (OSError,ValueError,KeyError):rows=[]
    key=tuple(rows) or (('NONE','Capture an eye control or partial blink first',''),)
    return ITEMS.setdefault(key,list(key))

def topology(mesh):
    values=np.empty(len(mesh.loops),dtype=np.int32);mesh.loops.foreach_get('vertex_index',values)
    sizes=np.empty(len(mesh.polygons),dtype=np.int32);mesh.polygons.foreach_get('loop_total',sizes)
    return hashlib.sha256(values.tobytes()+sizes.tobytes()).hexdigest()

def current(context):
    g=context.scene.cm_face_review;obj=g.corrective_object
    if not obj or obj.type!='MESH' or obj.name not in context.view_layer.objects:
        raise ValueError('Open a captured corrective in the current view layer first')
    return obj,json.loads(g.sculpt_state or '{}')


def object_mode(context,obj):
    if context.mode=='OBJECT':return
    if context.mode!='SCULPT' or context.view_layer.objects.active!=obj:
        raise ValueError('Finish the other active edit before changing the corrective session')
    if bpy.ops.object.mode_set(mode='OBJECT')!={'FINISHED'}:
        raise ValueError('Could not leave corrective sculpt mode')


def frame(context):
    obj,_=current(context)
    if context.mode!='OBJECT' and (context.mode!='SCULPT' or context.view_layer.objects.active!=obj):
        raise ValueError('Finish the other active edit before framing the corrective')
    # A background agent has no viewport; framing is optional there.
    for window in context.window_manager.windows:
        if window.scene!=context.scene:continue
        for area in window.screen.areas:
            if area.type!='VIEW_3D':continue
            region=next((r for r in area.regions if r.type=='WINDOW'),None)
            if not region:continue
            selected=list(context.selected_objects);active=context.view_layer.objects.active
            try:
                if context.mode=='OBJECT':
                    for value in selected:value.select_set(False)
                    obj.select_set(True);context.view_layer.objects.active=obj
                with context.temp_override(window=window,area=area,region=region):
                    bpy.ops.view3d.view_selected(use_all_regions=False)
            finally:
                if context.mode=='OBJECT':
                    obj.select_set(False)
                    for value in selected:value.select_set(True)
                    context.view_layer.objects.active=active
            return True
    return False


def sculpt(context):
    obj,_=current(context)
    if context.mode=='SCULPT' and context.view_layer.objects.active==obj:return
    object_mode(context,obj)
    if obj.modifiers or obj.data.shape_keys:
        raise ValueError('Keep the captured corrective without modifiers or shape keys')
    obj.hide_set(False)
    for selected in context.selected_objects:selected.select_set(False)
    obj.select_set(True);context.view_layer.objects.active=obj
    if bpy.ops.object.mode_set(mode='SCULPT')!={'FINISHED'}:
        raise ValueError('Could not enter corrective sculpt mode')
    frame(context)
    context.scene.cm_face_review.status='Sculpt the unmasked lid. Prepare Correction leaves Sculpt mode automatically.'


def finish(context,discard=False):
    obj,state=current(context)
    g=context.scene.cm_face_review
    token=state.get('session_token')
    owned=[value for value in (obj,g.corrective_eye_left,g.corrective_eye_right) if value and token and value.get(OWNER)==token]
    if discard:
        if not token or obj not in owned:
            raise ValueError('This older session has no ownership record; keep its sculpt and remove unwanted objects manually')
        if any(len(value.users_scene)>1 for value in owned):
            raise ValueError('A corrective object is linked to another scene; retain it before ending this session')
    object_mode(context,obj)
    if discard:
        # Artist-created children are not part of the temporary capture.
        for value in owned:
            for child in list(value.children):
                if child not in owned:
                    world=child.matrix_world.copy();child.parent=None;child.matrix_world=world
        meshes={value.data for value in owned if value.type=='MESH'}
        for value in owned:bpy.data.objects.remove(value,do_unlink=True)
        for mesh in meshes:
            if mesh.users==0:bpy.data.meshes.remove(mesh)
    g.corrective_object=None;g.corrective_eye_left=None;g.corrective_eye_right=None;g.sculpt_state=''
    for value in context.selected_objects:value.select_set(False)
    for name in state.get('previous_selection',[]):
        value=context.view_layer.objects.get(name)
        if value and not value.hide_get() and not value.hide_select:value.select_set(True)
    active=context.view_layer.objects.get(state.get('previous_active',''))
    context.view_layer.objects.active=active if active and active.select_get() else next(iter(context.selected_objects),None)
    g.status='Temporary corrective discarded. Source rig and captured inputs retained.' if discard else 'Corrective geometry retained. Editing session finished.'


def begin(context):
    if context.mode!='OBJECT':raise ValueError('Finish the current edit before opening a captured pose')
    f=context.scene.cm_features
    if f.full_sculpt or f.body_editing or f.restore_editing:raise ValueError('Finish the active character edit before opening a captured pose')
    g=context.scene.cm_face_review
    if g.corrective_object:raise ValueError('Retain or finish the current corrective before opening another pose')
    if not g.preparation_folder:raise ValueError('Prepare the captured lid grid first to establish its native rig reference')
    base=Path(bpy.path.abspath(g.preparation_folder))/'manifest.json';manifest=json.loads(base.read_text())
    capture=Path(bpy.path.abspath(g.capture_folder));rows=json.loads((capture/'capture.json').read_text())
    if rows.get('status')!='complete' or rows.get('candidate')!=manifest['source_mesh']:raise ValueError('Use a complete capture of the prepared source face')
    if g.sculpt_pose in {'NONE','SELECT'}:raise ValueError('Choose a single-control captured pose')
    row=rows['poses'][int(g.sculpt_pose)];controls=row['controls']
    if len(controls)!=1 or not 0<float(next(iter(controls.values())))<=1:raise ValueError('This corrective editor requires one active captured facial control')
    file=Path(row['file']).resolve();meta=json.loads(file.read_text())
    if file.parent!=capture.resolve() or meta['source_mesh']!=manifest['source_mesh'] or meta['lod']!=0:raise ValueError('The captured pose identity changed')
    proofs={str(file):digest(file),str(base.resolve()):digest(base),str((capture/'capture.json').resolve()):digest(capture/'capture.json')}
    def read(key,dtype,shape=None):
        name=meta[key]
        if Path(name).name!=name:raise ValueError('Capture arrays must remain beside their metadata')
        path=capture/name;proofs[str(path.resolve())]=digest(path);data=np.fromfile(path,dtype=dtype)
        return data.reshape(shape) if shape else data
    mid=read('dna_mesh_file','<i4');vid=read('dna_vertex_file','<i4');posed=read('vertices_file','<f4',(-1,3));tri=read('triangles_file','<u4',(-1,3));read('skin_linear_file','<f4',(-1,3,3))
    count=int(vid[mid==0].max())+1;head=canonical_offsets(posed,mid,vid,0,count)[0]
    faces=vid[tri[np.all(mid[tri]==0,axis=1)]]
    faces=faces[np.all(np.diff(np.sort(faces,axis=1),axis=1)>0,axis=1)]
    structure=Path(json.loads((base.parent/'preparation-summary.json').read_text()).get('structure_file',base.parent.parent/'inputs/lid-structure.npz'))
    if not structure.is_file():raise ValueError('Keep the original lid preparation anatomy beside its inputs')
    proofs[str(structure.resolve())]=digest(structure)
    driver=next(iter(controls)).removeprefix('CTRL_expressions.');role='left' if driver.endswith('L') else 'right' if driver.endswith('R') else None
    if not role or not driver.startswith('eye'):raise ValueError('Choose a captured left or right eye control for lid corrective editing')
    value=float(next(iter(controls.values())));closed=driver.startswith('eyeBlink') and value==1.
    if driver.startswith('eyeBlink') and not 1/64<=value<=1:raise ValueError('Choose a captured blink from 1/64 through full closure')
    with np.load(structure) as anatomy:margin=anatomy[role+'_margin'].copy();outer=anatomy['outer_triangles'].copy();interior=anatomy[role+'_interior'].copy()
    support=np.unique(margin)
    for _ in range(8):support=np.unique(outer[np.any(np.isin(outer,support),axis=1)])
    scope_proof=dict(method='LOCAL_EIGHT_RINGS',vertices=len(support))
    if closed and g.closed_sculpt_scope=='SURFACE':
        # Measure the undeformed sheet: a closed blink compresses surface
        # distances and must not enlarge coverage merely by changing expression.
        neutral_rows=[item for item in rows['poses'] if not item['controls']]
        if len(neutral_rows)!=1:raise ValueError('Physical eye coverage needs one neutral pose in the same capture')
        neutral_file=Path(neutral_rows[0]['file']).resolve()
        if neutral_file.parent!=capture.resolve():raise ValueError('Keep the neutral pose beside this captured eye pose')
        neutral_meta=json.loads(neutral_file.read_text())
        if neutral_meta['source_mesh']!=manifest['source_mesh'] or neutral_meta['lod']!=0:raise ValueError('The neutral sculpt reference belongs to another face or LOD')
        proofs[str(neutral_file)]=digest(neutral_file)
        def neutral_read(key,dtype,shape=None):
            name=neutral_meta[key]
            if Path(name).name!=name:raise ValueError('Neutral sculpt arrays must remain beside their capture')
            path=capture/name;proofs[str(path.resolve())]=digest(path);array=np.fromfile(path,dtype=dtype)
            return array.reshape(shape) if shape else array
        if not np.array_equal(neutral_read('dna_mesh_file','<i4'),mid) or not np.array_equal(neutral_read('dna_vertex_file','<i4'),vid):raise ValueError('Neutral and posed sculpt vertex identities differ')
        reference=canonical_offsets(neutral_read('vertices_file','<f4',(-1,3)),mid,vid,0,count)[0]
        eye_index=3 if role=='left' else 4;eye=canonical_offsets(posed,mid,vid,eye_index,int(vid[mid==eye_index].max())+1)[0]
        from .native_sculpt_scope import physical_support
        support,scope_proof=physical_support(reference,outer,margin,float(np.ptp(eye[:,0])))
    protected=np.ones(count,bool);protected[support]=False;protected[margin]=True
    if closed:
        # A closed-eye correction must allow editing the actual lid line and
        # its hidden return, while keeping the anatomical canthi fixed.
        protected[interior]=False
        protected[margin[[np.argmin(head[margin,0]),np.argmax(head[margin,0])]]]=True
    folder=Path(bpy.path.abspath(context.scene.cm_settings.project_dir))/'FaceReview'/'Sculpts'/uuid.uuid4().hex;folder.mkdir(parents=True,exist_ok=False)
    head.astype('<f4').tofile(folder/'source.f32');np.savez_compressed(folder/'protected.npz',protected=protected)
    for path in (folder/'source.f32',folder/'protected.npz'):proofs[str(path.resolve())]=digest(path)
    previous_selection=[value.name for value in context.selected_objects];previous_active=context.view_layer.objects.active
    token=uuid.uuid4().hex
    mesh=bpy.data.meshes.new('Native Pose Corrective');obj=None;references=[]
    try:
        mesh.from_pydata((head*[.01,-.01,.01]).tolist(),[],faces[:,[0,2,1]].tolist());mesh.update()
        mask=mesh.attributes.new('.sculpt_mask','FLOAT','POINT');mask.data.foreach_set('value',protected.astype(np.float32))
        obj=bpy.data.objects.new('Corrective - '+row['name'],mesh);context.scene.collection.objects.link(obj);obj.location.x=2.5;obj[OWNER]=token
        for polygon in mesh.polygons:polygon.use_smooth=True
        for index in (3,4):
            points=canonical_offsets(posed,mid,vid,index,int(vid[mid==index].max())+1)[0]
            eye_faces=vid[tri[np.all(mid[tri]==index,axis=1)]]
            eye_mesh=bpy.data.meshes.new('Captured Eye Reference');eye_mesh.from_pydata((points*[.01,-.01,.01]).tolist(),[],eye_faces[:,[0,2,1]].tolist());eye_mesh.update()
            eye=bpy.data.objects.new('Captured Eye '+str(index),eye_mesh);context.scene.collection.objects.link(eye);eye.parent=obj;eye.hide_select=True;eye[OWNER]=token;references.append(eye)
            for polygon in eye_mesh.polygons:polygon.use_smooth=True
        state=dict(schema=2,session_token=token,previous_selection=previous_selection,previous_active=previous_active.name if previous_active else '',pose_file=str(file),base_manifest=str(base.resolve()),structure=str(structure.resolve()),input_sha256=proofs,folder=str(folder),vertex_count=count,topology=topology(mesh),controls=controls,source_mesh=manifest['source_mesh'],full_blink=closed,sculpt_scope=scope_proof)
        (folder/'session.json').write_text(json.dumps(state,indent=2));g.sculpt_state=json.dumps(state);g.corrective_object=obj
        g.corrective_eye_left,g.corrective_eye_right=references
        for selected in context.selected_objects:selected.select_set(False)
        obj.select_set(True);context.view_layer.objects.active=obj
        sculpt(context)
        if closed:
            g.repair_surface=False
            g.status='Full blink: sculpt the lid line and socket. Canthi protected; neutral and quarter-strength anchors retained. Closed-eye geometry needs visual review.'
    except Exception:
        if obj and context.view_layer.objects.active==obj and context.mode=='SCULPT':bpy.ops.object.mode_set(mode='OBJECT')
        g.corrective_object=None;g.corrective_eye_left=None;g.corrective_eye_right=None;g.sculpt_state=''
        for eye in references:
            eye_mesh=eye.data;bpy.data.objects.remove(eye,do_unlink=True)
            if eye_mesh.users==0:bpy.data.meshes.remove(eye_mesh)
        if obj:bpy.data.objects.remove(obj,do_unlink=True)
        if mesh.users==0:bpy.data.meshes.remove(mesh)
        for name in previous_selection:
            value=context.view_layer.objects.get(name)
            if value:value.select_set(True)
        context.view_layer.objects.active=previous_active
        raise

def export(context,folder):
    g=context.scene.cm_face_review;obj=g.corrective_object;state=json.loads(g.sculpt_state or '{}')
    if not obj or obj.type!='MESH' or obj.modifiers or obj.data.shape_keys or len(obj.data.vertices)!=state.get('vertex_count') or topology(obj.data)!=state.get('topology'):raise ValueError('Keep the corrective mesh topology unchanged, without modifiers or shape keys')
    if any(digest(path)!=sha for path,sha in state['input_sha256'].items()):raise ValueError('A native corrective input changed; retain the sculpt and recapture the current rig')
    if state.get('full_blink') and g.repair_surface:raise ValueError('Turn off open-lid untangling for a full-blink corrective; closed-eye geometry requires separate review')
    object_mode(context,obj)
    values=np.empty(len(obj.data.vertices)*3,dtype=np.float32);obj.data.vertices.foreach_get('co',values);values=values.reshape(-1,3)*[100.,-100.,100.]
    if not np.isfinite(values).all():raise ValueError('The sculpt contains nonfinite positions')
    file=Path(folder)/'sculpted.f32';values.astype('<f4').tofile(file)
    state.update(edited_file=str(file),edited_sha256=digest(file),surface_repair=bool(g.repair_surface))
    snapshot=Path(folder)/'sculpt.json';snapshot.write_text(json.dumps(state,indent=2));return str(snapshot)
