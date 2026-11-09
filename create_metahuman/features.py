"""Expanded workflow properties, operators and sidebar panels."""

from pathlib import Path
from contextlib import contextmanager
import traceback
import bpy
import numpy as np
from bpy.props import (
    StringProperty,
    PointerProperty,
    FloatProperty,
    IntProperty,
    BoolProperty,
    EnumProperty,
    CollectionProperty,
    FloatVectorProperty,
)
from bpy_extras.io_utils import ImportHelper, ExportHelper
from . import anatomy, workflow, mesh_ops as m, textures, head_model, automation
from .paint_tools import CLASSES as PAINT_CLASSES
from . import paint_tools
from . import skin_bake_layers_ui
from . import head_revision
from . import expression_maps
from .face_mixer import CM_OT_face_mixer
from .wrinkle_ops import CLASSES as WRINKLE_CLASSES
from .sculpt_session_ops import CLASSES as SCULPT_SESSION_CLASSES
from . import sculpt_session
from . import lookdev, expressions, restore_brush, native_body, bake_jobs, detail_jobs
from . import unreal_jobs
from . import skin_library
from . import skin_brushes
from . import detail_mask
from . import scan_jobs
from . import region_projection
from . import regional_scan
from . import normal_composition
from . import simple_shading
from . import surface_heal
from . import eye_fit_jobs
from . import eye_color_jobs
from . import native_eye_update
from . import native_return
from . import ear_fit
from . import skin_projection_history
from . import landmark_tools
from . import joint_wrap
from . import auto_wrap_jobs
from . import character_skin
from . import wrinkle_jobs
from .workflow_steps import ITEMS as WORKFLOW_ITEMS, ORDER as WORKFLOW_ORDER


def generated_head_update(s):
    return bool(s.part in {'HEAD','FULL_BODY'} and s.source and s.source.get('cm_gnm')
        and (s.part=='FULL_BODY' or s.result))


@contextmanager
def head_source_scope(s):
    part=s.part
    try:
        if generated_head_update(s):
            s.part='HEAD'
        yield
    finally:
        s.part=part


def wrap_generated_head(context,s,f,*,defer_wrinkles=False):
    from . import sculpt_archive
    # The face edit invalidates these maps itself. Avoid rewriting disk receipts
    # for its intermediate sculpt selection, which Blender Undo cannot rewind.
    with sculpt_archive.recover(context, ('HEAD_DETAIL', 'WM1', 'WM2', 'WM3'), rebind_maps=False):
        return _wrap_generated_head(context,s,f,defer_wrinkles=defer_wrinkles)


def _wrap_generated_head(context,s,f,*,defer_wrinkles=False):
    """Replace only the head; restore accepted state if fitting/attachment fails."""
    if f.body_editing or f.full_sculpt or f.restore_editing:
        raise ValueError('Apply or cancel the active body, sculpt or head repair before updating the face.')
    if f.expression_active:
        raise ValueError('Return Expression Check to Neutral before updating the face.')
    if detail_jobs.running() or bake_jobs.running() or wrinkle_jobs.running():
        raise ValueError('Finish or cancel the detail update or map bake before updating the face.')
    if s.part=='FULL_BODY' and not f.body_result:
        raise ValueError('Wrap the generated source as Head, then Load Body before updating a full character.')
    if f.body_result:
        workflow.neck_ring(f.body_result,False)  # Validate before replacing any accepted geometry.
    previous, previous_head=s.result,f.head_result
    objects=list(dict.fromkeys(obj for obj in (s.source,s.template,previous,f.eye_left,f.eye_right,f.teeth) if obj))
    snapshots=[(obj,m.transform_state(obj),obj.hide_get(),obj.hide_render) for obj in objects]
    placement=s.source.get('cm_head_placement')
    pending=s.detail.get('cm_head_update_pending') if s.detail else None
    from . import map_provenance
    previous_maps=context.scene.get(map_provenance.INVALIDATED)
    wrinkle_pending=[(obj,obj.get('cm_wrinkle_update_pending')) for obj in (f.wm1,f.wm2,f.wm3) if obj]
    internals=[]
    for obj in (f.eye_left,f.eye_right,f.teeth):
        if obj and f.full_head:
            original=list(obj['cm_follow_original']) if 'cm_follow_original' in obj else None
            anchors=obj.get('cm_follow_anchors')
            values=[(key,key.value) for key in obj.data.shape_keys.key_blocks if key.name.startswith('Body Follow Applied ')] if obj.data.shape_keys else []
            internals.append((obj,original,anchors,values))
    new_result=None
    try:
        if placement:
            import json
            from mathutils import Matrix, Vector
            frame=json.loads(placement)
            if m.transform_state(s.source)==frame['placed']:
                m.restore_transform(s.source,frame['fit'])
            else:
                s.source.matrix_world=Matrix.Translation(-Vector(frame['movement']))@s.source.matrix_world
            del s.source['cm_head_placement']
            context.view_layer.update()
        for _,_,_,values in internals:
            for key,_ in values:
                key.value=0  # The new body-frame placement replaces the old generated follow offset.
        with head_source_scope(s):
            if f.wrap_method=='AUTO' and not len(s.landmarks):
                automation.auto_landmarks(context,s,f)
            new_result=m.wrap(context,s)
            if previous and m.topology(previous)==m.topology(new_result):
                new_result.data.materials.clear()
                for material in previous.data.materials:
                    new_result.data.materials.append(material)
                old_uv=previous.data.uv_layers.active
                new_uv=new_result.data.uv_layers.active
                if old_uv and new_uv and len(old_uv.data)==len(new_uv.data):
                    old_values=np.empty(len(old_uv.data)*2,dtype=np.float32)
                    new_values=np.empty_like(old_values)
                    old_uv.data.foreach_get('uv',old_values);new_uv.data.foreach_get('uv',new_values)
                    if np.array_equal(old_values,new_values):
                        # Keep the immutable skin baseline. Falling back to the
                        # last final normal would count old sculpt detail twice.
                        for name in ('cm_reference_normal_image','cm_reference_normal','cm_normal_source','cm_albedo_source'):
                            if name in previous:new_result[name]=previous[name]
            if f.body_result:
                workflow.attach_updated_head(context,s,f)
        for obj,_ in wrinkle_pending:
            obj['cm_wrinkle_update_pending']=new_result
        if s.detail:
            if detail_jobs.start(context,s,f,'SYNC_HEAD'):
                s.detail['cm_head_update_pending']=new_result
            elif 'cm_head_update_pending' in s.detail:
                del s.detail['cm_head_update_pending']
        if wrinkle_pending and not detail_jobs.running() and not defer_wrinkles:
            wrinkle_jobs.start(context,s,f)
        map_provenance.invalidate_face(context.scene)
        return new_result
    except Exception:
        if wrinkle_jobs.running():
            wrinkle_jobs.detach()
        if detail_jobs.running() and any(target['source']==new_result for target in detail_jobs._JOB['targets']):
            detail_jobs.cancel_active()
        s.result,f.head_result=previous,previous_head
        if previous_maps is None:
            if map_provenance.INVALIDATED in context.scene:del context.scene[map_provenance.INVALIDATED]
        else:context.scene[map_provenance.INVALIDATED]=previous_maps
        for obj,value in wrinkle_pending:
            if value is None:
                if 'cm_wrinkle_update_pending' in obj:del obj['cm_wrinkle_update_pending']
            else:obj['cm_wrinkle_update_pending']=value
        for obj,transforms,hidden,render_hidden in snapshots:
            # Restore native components, avoiding a second lossy matrix decomposition.
            m.restore_transform(obj,transforms)
            obj.hide_set(hidden)
            obj.hide_render=render_hidden
        for obj,original,anchors,values in internals:
            for key,value in values:
                key.value=value
            for name,value in (('cm_follow_original',original),('cm_follow_anchors',anchors)):
                if value is None:
                    if name in obj:
                        del obj[name]
                else:
                    obj[name]=value
        if placement is None:
            if 'cm_head_placement' in s.source:
                del s.source['cm_head_placement']
        else:
            s.source['cm_head_placement']=placement
        if s.detail:
            if pending is None:
                if 'cm_head_update_pending' in s.detail:
                    del s.detail['cm_head_update_pending']
            else:
                s.detail['cm_head_update_pending']=pending
        if new_result and new_result!=previous:
            mesh=new_result.data
            bpy.data.objects.remove(new_result,do_unlink=True)
            if not mesh.users:
                bpy.data.meshes.remove(mesh)
        context.view_layer.update()
        raise


def wrap_full_body(context, s, f):
    if generated_head_update(s):
        return wrap_generated_head(context, s, f)
    from . import joint_wrap
    return joint_wrap.wrap(context, s, f)


def export_full_body(context, s, f, *, root=None):
    import shutil, json
    from tempfile import TemporaryDirectory

    if root is None:
        return m.export_bundle(context, s)

    m.check_mesh(f.head_result or s.result, "Head Result")
    m.check_mesh(f.body_result, "Body Result")
    from . import roughness_response
    responses=roughness_response.profiles(context.scene)
    original = (s.part, s.result, s.character_name)
    name = s.character_name
    try:
        s.part = "HEAD"
        s.result = f.head_result or s.result
        head = m.export_bundle(context, s, _root=root)
        s.part = "BODY"
        s.result = f.body_result
        s.character_name = name + "_Body"
        with TemporaryDirectory(prefix='.body-', dir=head) as temporary:
            body = m.export_bundle(context, s, _root=Path(temporary))
            data = json.loads((head / "manifest.json").read_text())
            body_data = json.loads((body / "manifest.json").read_text())
            from . import export_delivery
            export_delivery.merge_support(body,head,body_data,data)
            shutil.copy2(body / body_data["mesh"], head / body_data["mesh"])
            data.update(
                part="FULL_BODY",
                body_mesh=body_data["mesh"],
                body_vertices=body_data["vertices"],
                body_vertex_count=body_data["vertex_count"],
                body_topology_sha256=body_data["topology_sha256"],
                body_canonical_order=body_data.get("canonical_order", False),
            )
            for relative in body_data.get("maps", []):
                destination = head / relative
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(body / relative, destination)
                if relative not in data["maps"]:
                    data["maps"].append(relative)
        data["texture_sets"] = {
            "HEAD": [
                p for p in data["maps"] if not Path(p).stem.startswith(name + "_Body_")
            ],
            "BODY": body_data.get("maps", []),
        }
        data['roughness_response']=responses
        if f.character_applied_skin_image:
            recipe = f.character_applied_skin_image.get('cm_native_skin_recipe')
            if recipe:
                data['native_skin'] = json.loads(recipe)
        (head / "manifest.json").write_text(json.dumps(data), encoding="utf-8")
        return head
    finally:
        s.part, s.result, s.character_name = original


def update_body(self, context):
    if (
        context
        and hasattr(context.scene, "cm_features")
        and context.scene.cm_features.body_editing
    ):
        workflow.body_preview(
            context, context.scene.cm_settings, context.scene.cm_features
        )


class CM_BodyControl(bpy.types.PropertyGroup):
    key: StringProperty()
    category: StringProperty()
    label: StringProperty()
    value: FloatProperty(min=-1, max=1, update=update_body)


class CM_IdentityCoefficient(bpy.types.PropertyGroup):
    index: IntProperty()
    value: FloatProperty(min=-3, max=3)


def previous_detail_poll(self,obj):
    from .detail_recovery import _compatible
    scene=self.id_data
    if not hasattr(scene,'cm_settings'):return False
    s=scene.cm_settings
    detail=self.body_detail if s.part=='FULL_BODY' and self.texture_part=='BODY' else s.detail
    try:return bool(detail and not obj.modifiers and _compatible(obj,detail))
    except (TypeError,ValueError,ReferenceError):return False


class CM_Features(bpy.types.PropertyGroup):
    projection_region: StringProperty(name='Fit Region',default='Hand_L',
        description='Matching patch name on the source and result/template; select vertices in Edit Mode and mark each patch')
    web_export_kind: EnumProperty(name='Export From',items=(('NATIVE','Processed Unreal character','Rigged geometry, captured materials and grooms'),('STATIC','Accepted Blender mesh','Static geometry and embedded PBR images')),default='NATIVE')
    native_web_source: StringProperty(name='Native Capture',subtype='DIR_PATH',description='Completed native capture folder containing source-manifest.json and extracted textures')
    native_web_appearance: StringProperty(name='Artist Appearance Bakes',subtype='FILE_PATH',description='Optional source-bound appearance manifest; otherwise uses appearance-manifest.json in the capture folder')
    native_web_archive: BoolProperty(name='Create Download ZIP',default=True,description='Include a portable ZIP as well as the served preview folder; requires extra disk space')
    native_web_face_component: StringProperty(name='Face Component',default='Face')
    native_web_expression_bank: StringProperty(name='Native Face Samples',subtype='FILE_PATH',description='Optional completed native face bank.json matching this captured face; exports evaluated joint-and-corrective samples instead of direct corrective controls')
    native_web_legacy_frame: BoolProperty(name='Legacy Component Frame',default=False,description='Explicitly interpret old parent-relative capture transforms as actor-relative; only use when the frames are known to match')
    native_web_legacy_textures: BoolProperty(name='Legacy Texture Metadata',default=False)
    native_web_legacy_grooms: BoolProperty(name='Legacy Groom Bindings',default=False)
    native_web_binding: StringProperty(name='Verified Groom Bindings',subtype='FILE_PATH')
    native_web_job: StringProperty(subtype='DIR_PATH')
    native_web_output: StringProperty(subtype='DIR_PATH')
    native_web_status: StringProperty(options={'SKIP_SAVE'})
    native_web_url: StringProperty(options={'SKIP_SAVE'})
    native_web_capture_receipt: StringProperty(subtype='FILE_PATH')
    native_web_capture_result: StringProperty()
    native_web_output_capture: StringProperty(subtype='DIR_PATH')
    native_web_capture_scope: EnumProperty(name='Capture From',items=(('CHARACTER','Linked Creator character','Save Creator edits first; uses an existing native rig'),('SELECTED','Selected Unreal level actor','Capture exactly one selected actor and its current component materials')),default='CHARACTER')
    native_web_capture_size: EnumProperty(name='Surface Bake Size',description='Skin and opaque custom materials; eyes and mouth use 2K maps',items=[(str(n),str(n),'') for n in (2048,4096,8192)],default='8192')
    native_web_capture_face_samples: BoolProperty(name='Capture Native Facial Samples',default=True,description='Capture character-specific mouth shapes, blinks, expressions and native skin responses automatically; needs a saved DNA-bound face and bridge 0.17 or newer')
    skin_library_status: StringProperty(options={'SKIP_SAVE'})
    unreal_push_status: StringProperty()
    unreal_push_receipt: StringProperty(subtype='FILE_PATH')
    unreal_texture_receipt: StringProperty(subtype='FILE_PATH')
    eye_hidden_pigment: EnumProperty(name='Hidden Eye Color',items=[
        ('SCLERA_ONLY','Measured Sclera','Continue measured sclera behind lids; preserve the source pupil cutoff'),
        ('FIT_DARK_ELLIPSE','Complete Simple Pupil','Infer hidden dark pigment from a well-supported observed elliptical silhouette; measured interior stays unchanged and ambiguous eyes retain their source')],default='SCLERA_ONLY')
    unreal_source_eye_receipt: StringProperty(subtype='FILE_PATH')
    unreal_source_eye_result: StringProperty()
    unreal_sources_receipt: StringProperty(subtype='FILE_PATH')
    unreal_sources_result: StringProperty()
    unreal_rig_receipt: StringProperty(subtype='FILE_PATH')
    unreal_rig_result: StringProperty()
    unreal_texture_result: StringProperty()
    unreal_eye_receipt: StringProperty(subtype='FILE_PATH')
    unreal_eye_result: StringProperty()
    unreal_character: StringProperty()
    unreal_revision_source: StringProperty(name='Source Character',description='Saved Creator asset path; blank uses the linked character')
    unreal_revision_project: StringProperty(name='Unreal Project',subtype='FILE_PATH',description='Source .uproject; blank uses the linked character project')
    unreal_revision_root: StringProperty(name='Revision Folder',default='/Game/CreateMHRevisions')
    unreal_revision_skin_color: BoolProperty(name='Include Skin Color',default=False,description='Replace head albedo and available CM1/CM2/CM3 expression color maps with accepted Blender maps; keep native eye, freckle and skin controls')
    unreal_revision_update_scope: EnumProperty(name='Refresh Maps',items=(('DETAIL','Normal and Cavity','Refresh only head normal and cavity'),('SKIN','Skin Color and Detail','Also refresh head albedo and available expression color maps'),('ALL_HEAD','All Available Head Maps','Also include available wrinkle normals and roughness; body maps stay unchanged')),default='DETAIL')
    unreal_wrinkle_input_encoding: EnumProperty(name='Wrinkle Input',items=(('FULL','Full Normal Maps','Add-on sculpt bakes; convert against the neutral geometry normal'),('DELTA','Native Delta Maps','Already encoded native WM textures; preserve their numeric data')),default='FULL')
    unreal_wrinkle_neutral_mode: EnumProperty(name='Wrinkle Neutral',items=(('AUTO','Automatic','Recover pre-repair neutral from verified skin history for tracked native wrinkle bakes'),('CURRENT','Maps Include Current Skin','These full expression maps already contain the current neutral skin; subtract the current normal'),('IMAGE','Original Neutral Image','Use the saved neutral normal with which the full expression maps were authored')),default='AUTO')
    unreal_wrinkle_neutral_image: PointerProperty(name='Original Neutral Normal',type=bpy.types.Image)
    unreal_color_input_encoding: EnumProperty(name='Expression Color Input',items=(('FULL','Full Color Maps','Add-on CM bakes; convert against the accepted albedo'),('DELTA','Native Delta Maps','Already encoded native CM textures; preserve their numeric data')),default='FULL')
    unreal_revision_receipt: StringProperty(subtype='FILE_PATH')
    unreal_revision_result: StringProperty()
    unreal_revision_character: StringProperty()
    unreal_assembly_ready: BoolProperty(default=False)
    unreal_status_report: StringProperty()
    unreal_artist_snapshot: StringProperty(subtype='FILE_PATH')
    unreal_artist_receipt: StringProperty(subtype='FILE_PATH')
    unreal_artist_scope: EnumProperty(name='Capture',items=(
        ('CHARACTER','Creator character','Preserve the saved MetaHuman character and linked native assets'),
        ('SELECTED_ACTOR','Character + selected UE actor','Also preserve the single selected saved level actor, its components, Blueprint, level and dependencies')),default='CHARACTER')
    unreal_artist_review: StringProperty()
    unreal_character_project: StringProperty()
    unreal_native_map_parts: EnumProperty(name='Native Map Parts',items=[('BOTH','Head and Body',''),('HEAD','Head',''),('BODY','Body','')],default='BOTH')
    unreal_build_preserve_source_textures: BoolProperty(name='Preserve Source Texture Resolution',default=True,
        description='Keep source LOD0 albedo resolution and animated source-map sizes through 8K using private assembly settings; distant LOD presets remain in place')
    unreal_native_map_channels: EnumProperty(name='Native Map Channels',items=[('DETAIL','Normal and Cavity',''),('FINISH','Roughness',''),('DETAIL_FINISH','Normal, Cavity and Roughness',''),('AO','Repaired Bent-normal AO','Apply the prepared packed bent-normal/AO map while retaining native skin shading')],default='DETAIL')
    detail_update_status: StringProperty(options={'SKIP_SAVE'})
    detail_recovery_manual: BoolProperty(name='Choose Previous Layer',default=False,options={'SKIP_SAVE'})
    head_detail_previous: PointerProperty(name='Previous Layer',type=bpy.types.Object,poll=previous_detail_poll)
    body_detail_previous: PointerProperty(name='Previous Layer',type=bpy.types.Object,poll=previous_detail_poll)
    ui_stage: EnumProperty(
        name="Workflow",
        items=WORKFLOW_ITEMS,
        default="START",
    )
    advanced_ui: BoolProperty(name="Advanced Controls", default=False)
    show_lookdev: BoolProperty(name="LookDev", default=False, update=lookdev.update)
    lookdev_skin_controls: BoolProperty(name="Skin Shader",default=False)
    lookdev_environment_controls: BoolProperty(name="Environment",default=False)
    lookdev_render_controls: BoolProperty(name="Render",default=True)
    lookdev_environment: EnumProperty(
        name="Environment",
        items=lookdev.ENVIRONMENTS,
        default="studio.exr",
        update=lookdev.update,
    )
    lookdev_rotation: FloatProperty(
        name="Rotation",
        subtype="ANGLE",
        default=0,
        min=-6.2832,
        max=6.2832,
        update=lookdev.update,
    )
    lookdev_intensity: FloatProperty(
        name="Strength", default=1, min=0.01, max=10, update=lookdev.update
    )
    lookdev_transparent: BoolProperty(
        name="Transparent Background", default=False, update=lookdev.update
    )
    lookdev_roughness: FloatProperty(
        name="Roughness", default=0.48, min=0.02, max=1, update=lookdev.update
    )
    lookdev_subsurface: FloatProperty(
        name="Subsurface", default=0.06, min=0, max=1, update=lookdev.update
    )
    lookdev_normal_strength: FloatProperty(
        name="Normal Detail", default=1, min=0, max=3, update=lookdev.update
    )
    lookdev_size: EnumProperty(
        name="Long Side",
        items=[("1024", "1K", ""), ("2048", "2K", ""), ("4096", "4K", "")],
        default="1024",
    )
    lookdev_renderer: EnumProperty(
        name="Renderer",
        items=[
            ("EEVEE", "Eevee", "Fast preview"),
            ("CYCLES", "Cycles CPU", "Path-traced preview"),
        ],
        default="EEVEE",
    )
    lookdev_samples: IntProperty(name="Samples", default=32, min=8, max=256)
    preview_output: StringProperty(name="Last Preview", subtype="FILE_PATH")
    input_mode: EnumProperty(
        items=[
            ("SCAN", "Import Mesh", "Use an existing scan or sculpt"),
            ("GENERATE", "Generate Face", "Generate a local GNM source"),
            (
                "CHARACTER",
                "Full Character",
                "Start with your local textured MetaHuman templates",
            ),
        ]
    )
    source_front_axis: EnumProperty(
        name="Source Faces",
        items=[
            ("AUTO", "Same as Template", ""),
            ("X", "+X", ""),
            ("-X", "-X", ""),
            ("Y", "+Y", ""),
            ("-Y", "-Y", ""),
        ],
    )
    landmarks_reviewed: BoolProperty(default=False)
    last_action_error: BoolProperty(default=False)
    delivery_report: StringProperty()
    proxy: PointerProperty(type=bpy.types.Object)
    body_template: PointerProperty(name="Body Template", type=bpy.types.Object)
    body_result: PointerProperty(name="Body Result", type=bpy.types.Object)
    body_detail: PointerProperty(name="Body Detail", type=bpy.types.Object)
    head_result: PointerProperty(name="Head Result", type=bpy.types.Object)
    wrap_influence: FloatProperty(name='Wrap Morph', min=0, max=1, default=1,
        get=joint_wrap.morph_get, set=joint_wrap.morph_set,
        description='Blend from the original template to the fitted character; head and body move together')
    texture_part: EnumProperty(
        name="Skin Part", items=[("HEAD", "Head", ""), ("BODY", "Body", "")]
    )
    skin_map_size: EnumProperty(name='Texture Resolution',items=[(str(size),str(size),'8K needs extra memory. Smart Paint retains 8K output with a fast preview while dragging.' if size==8192 else '') for size in (512,1024,2048,4096,8192)],default='4096',
        description='Source skin resolution, independent of the final sculpt normal/cavity resolution')
    skin_bake_scope: EnumProperty(name='Bake Parts',items=[('SELECTED','Selected Part','Use the Head/Body selector'),
        ('BOTH','Head and Body','Project skin onto both fitted parts')],default='SELECTED')
    skin_bake_channels: EnumProperty(name='Transfer Channels',default='COLOR',items=[
        ('COLOR','Color','Transfer color and optionally source normals'),
        ('ROUGHNESS','Roughness Only','Transfer source shader roughness while retaining accepted color, normals and cavity'),
        ('ALL','Color and Roughness','Transfer color and roughness, and optionally source normals')])
    skin_replace_normal: BoolProperty(name='Replace Normal from Source',default=False,
        description='Replace existing delivered normals with source-transfer normals; otherwise preserve them. A missing normal is created automatically')
    skin_brush: EnumProperty(
        name='Skin Brush',
        items=skin_brushes.PRESETS,
        default='PoresFine',
    )
    detail_mask_helper: PointerProperty(type=bpy.types.Object)
    full_sculpt: PointerProperty(name="Combined Sculpt Session", type=bpy.types.Object)
    eye_left: PointerProperty(name="Left Eye", type=bpy.types.Object)
    eye_right: PointerProperty(name="Right Eye", type=bpy.types.Object)
    show_eye_appearance: BoolProperty(name='Eye Appearance',default=False)
    eye_iris_exposure: FloatProperty(name='Iris Exposure',default=3,min=-3,max=6)
    eye_sclera_exposure: FloatProperty(name='Sclera Exposure',default=1,min=-3,max=3)
    eye_iris_radius: FloatProperty(name='Iris Size',default=.16,min=.08,max=.28)
    eye_native_enabled: BoolProperty(name='Creator Eye Appearance',default=False)
    eye_native_tint: FloatVectorProperty(name='Iris Tint',subtype='COLOR',size=3,min=0,max=1,default=(.65,.7,.75))
    eye_native_saturation: FloatProperty(name='Iris Saturation',min=0,max=4,default=0)
    eye_native_size: FloatProperty(name='Iris Size',min=.145,max=.185,default=.165,precision=3)
    teeth: PointerProperty(name="Teeth", type=bpy.types.Object)
    full_head: BoolProperty(name="Include Eyes and Teeth", default=True)
    wrap_method: EnumProperty(
        name="Landmarks",
        items=[
            ("MANUAL", "Manual", "Place editable pairs"),
            ("AUTO", "Automatic", "Detect facial or pose landmarks locally"),
        ],
        default="AUTO",
    )
    wrap_quality: EnumProperty(
        name="Wrap Quality",
        items=[
            ("CUSTOM", "Balanced", "Use the solver settings"),
            ("FASTER", "Faster", "Shorter solve"),
            ("ACCURATE", "Accurate", "More refinement iterations"),
        ],
    )
    conservative: BoolProperty(name="Conservative Wrap")
    preserve_mouth_interior: BoolProperty(name="Preserve Mouth Interior", default=True,
        description="Keep recognized oral cavity UV islands behind the lips while fitting the outer skin")
    is_mask: BoolProperty(name="Frontal Face Mask")
    decimate_ratio: FloatProperty(name="Proxy Ratio", default=0.25, min=0.01, max=1)
    front_axis: EnumProperty(
        name="Front Axis",
        items=[
            ("-Y", "-Y", "Blender convention"),
            ("Y", "+Y", ""),
            ("X", "+X", "Common Unreal FBX convention"),
            ("-X", "-X", ""),
        ],
    )
    landmark_kind: EnumProperty(
        name="Detector",
        items=[
            ("FACE", "Face", "Face landmarks"),
            ("POSE", "Full Body", "Pose landmarks"),
        ],
    )
    reference_library: StringProperty(
        name="Reference Library", subtype="FILE_PATH",
        description="Local licensed head/body template library; independent of optional inference models",
    )
    model_dir: StringProperty(
        name="Local Model Folder",
        subtype="DIR_PATH",
        default=str(Path(__file__).resolve().parents[1] / "models"),
    )
    inference_python: StringProperty(
        name="Inference Python",
        subtype="FILE_PATH",
        default=str(Path(__file__).resolve().parents[1] / ".venv/Scripts/python.exe"),
    )
    sculpt_region: EnumProperty(
        name="Sculpt Region",
        items=[
            (n, n.replace("_", " "), "")
            for n in (
                "Head",
                "Ears",
                "Eyes",
                "Upper_Orbital_Rim",
                "Inner_Parts",
                "NeckRegion",
            )
        ],
    )
    use_detail_mask: BoolProperty(name="Use Detail Mask", default=False)
    neck_fade: FloatProperty(
        name="Neck Fade / Height", default=0.12, min=0.001, max=0.5
    )
    restore_mode: EnumProperty(name='Repair Mode', items=[
        ('RESTORE', 'Restore Shape', 'Relax local fitting damage while retaining template anatomy'),
        ('SURFACE', 'Smooth on Source', 'Smooth local folds and reproject onto the reference mesh; Neutral only')], default='RESTORE')
    restore_radius: FloatProperty(
        name="Restore Radius / Diagonal", default=0.08, min=0.001, max=1
    )
    restore_editing: BoolProperty(default=False,options={'HIDDEN','SKIP_SAVE'})
    show_restore_controls: BoolProperty(name="Repair Settings",default=False)
    restore_strength: FloatProperty(
        name="Restore Strength", default=0.25, min=0.001, max=1
    )
    restore_source_adherence: FloatProperty(
        name='Follow Source', default=.9, min=0, max=1,
        description='How strongly smoothing projects back to the reference mesh. Reduce to repair folds present in the source; zero smooths without projection'
    )
    mirror: BoolProperty(name="Mirror", default=True)
    cavity_radius: FloatProperty(
        name="Cavity Scale / Diagonal", default=0.01, min=0.00001, max=0.1
    )
    body_controls: CollectionProperty(type=CM_BodyControl)
    body_editing: BoolProperty(default=False)
    body_backend: EnumProperty(name="Generator",items=[('NATIVE','Anatomical','Use the installed native body model for physique and measurements'),('REGIONAL','Regional','Independent regional sculpt controls')],default='REGIONAL')
    body_native_status: StringProperty(options={'SKIP_SAVE'})
    proportional_scale: BoolProperty(
        name="Proportional Scale",
        default=True,
        description="Fit the body's neck size to the current head before attaching",
    )
    body_category: EnumProperty(
        name="Body Section",
        items=[
            (n, l, "")
            for n, l in [
                ("main", "Main"),
                ("upper", "Upper"),
                ("arms", "Arms"),
                ("lower", "Lower"),
                ("hands_feet", "Hands & Feet"),
                ("details", "Details"),
            ]
        ],
    )
    reference_skin: PointerProperty(name="Reference Skin", type=bpy.types.Image)
    reference_skin_body: PointerProperty(name="Body Reference Skin", type=bpy.types.Image)
    skin_solver_mask: PointerProperty(name='Reconstruction Mask',type=bpy.types.Image,
        description='Optional image in this skin UV layout: white allows reconstruction, black protects')
    skin_solver_strength: FloatProperty(name='Cleanup Strength',default=1,min=0,max=1)
    skin_solver_color: BoolProperty(name='Clean Up Color',default=True,
        description='Disable to retain the accepted albedo while repairing its normal shading')
    skin_solver_method: EnumProperty(name='Cleanup Method',items=[
        ('PERIODIC','Preserve Source Detail','Subtract coherent repeating patterns without replacing source skin detail',0,0),
        ('LEARNED','Local Detail Restoration','Optional local upscale prior inside an explicit repair mask; inferred detail, preview before Apply',0,1),
        ('REFERENCE','Replace Damaged Skin','Reconstruct explicitly damaged regions from a clean reference; can soften source detail',0,2),
        ('SEAMS','Match UV Seam Colors','Balance actual paired mesh UV edges in the accepted color map without generating skin texture',0,3),
        ('GENERATIVE','Generative Albedo Material','Use a generated UV-matched albedo as the material source, retaining geometry and protected paint',0,4),
        ('MATERIAL','Material from Current Albedo','Build bounded skin surface maps from the accepted albedo without a model or replacement image; optionally redistribute its fine shading into a shared surface field',0,5),
        ('LOCAL_FILL','Complete Painted Color','Infer local color from surrounding skin inside an explicit mask, retaining protected paint and UV borders; preview before Apply',0,6)],default='PERIODIC')
    skin_solver_fill_method: EnumProperty(name='Color Completion',items=[
        ('SURFACE','Follow Skin Surface','Carry trusted boundary pigment through the marked region using actual mesh distances; infers smooth pigment, not pores'),
        ('TELEA','Image Neighborhood','Complete pigment from nearby texture pixels')],default='SURFACE')
    skin_solver_fill_radius: IntProperty(name='Color Fill Radius (texels)',default=7,min=1,max=64,
        description='Image-fill neighborhood and UV-border guard at the current atlas resolution; completes pigment without recovering microdetail')
    skin_solver_generated_albedo: PointerProperty(name='Generated Albedo',type=bpy.types.Image,
        description='Generated texture with this exact UV layout; generation may invent skin details')
    skin_solver_generated_body_albedo: PointerProperty(name='Generated Body Albedo',type=bpy.types.Image,
        description='Generated texture matching the body UV layout; stored separately from the head reference')
    skin_solver_preserve_body_anatomy: BoolProperty(name='Protect Hands, Feet & Neck',default=True,
        description='Keep source hand/foot detail with broad color matching, retain their shading maps, and anchor the neck to the head boundary during generated skin edits')
    skin_solver_material_strength: FloatProperty(name='Derive Skin Material',default=0,min=0,max=1,
        description='Artistic albedo-driven roughness and microcavity over retained source maps; not measured reflectance')
    skin_solver_surface_balance: FloatProperty(name='Move Shading into Surface Maps',default=0,min=0,max=1,
        description='Separate fine achromatic shading from generated color and use one shared height field for normals, roughness and cavity; retains crisp pigmentation')
    skin_solver_sr_python: StringProperty(name='Diffusion Python',subtype='FILE_PATH',
        description='Existing Python environment with Torch CUDA, Diffusers, Transformers, NumPy, SciPy and Pillow; no installation or downloads are performed')
    skin_solver_invention: FloatProperty(name='Texture Invention',default=.35,min=0,max=1,
        description='0 retains the chosen input; 1 permits the strongest bounded local model replacement. Broad color, UV borders and protected paint stay constrained')
    skin_solver_prior: EnumProperty(name='Detail Prior',items=[
        ('RESTORE','Fast Restoration','Local Real-ESRGAN upscale prior'),
        ('PISA','Diffusion Super Resolution','Local PiSA-SR atlas detail prior with source color, repair mask and protected paint constraints; requires existing optional CUDA models'),
        ('EDIT','Reference Editing','Optional quantized FLUX.2 Klein 4B reference editor; slower and more inventive')],default='RESTORE')
    skin_solver_brief: StringProperty(name='Skin Detail Brief',default='Fine irregular organic skin pores and natural skin grain',
        description='Describe permitted skin detail for the optional reference editor; texture invention still bounds replacement')
    skin_solver_output: EnumProperty(name='Output Resolution',default='KEEP',items=[
        ('KEEP','Keep Current','Repair at the current delivered resolution'),
        ('8192','8K Learned Upscale','PiSA-SR supports 2K/4K inputs and already-8K atlases; reference surface editing requires a 4K input')])
    skin_solver_projection: EnumProperty(name='Editing Space',items=[
        ('SURFACE','Continuous Surface','Edit physical skin patches and reproject through visible mesh triangles'),
        ('UV','UV Patches','Edit overlapping atlas crops directly')],default='SURFACE')
    skin_solver_preserve_structure: BoolProperty(name='Preserve Source Structure',default=True,
        description='Retain broader source detail at lower invention; high invention permits crease changes. Protect important marks explicitly with paint')
    skin_solver_patterns: BoolProperty(name='Suppress Repeating Artifacts',default=True,
        description='Suppress detected checker directions in model context and exclude them from the source structure lock; use explicit repair/protection masks')
    skin_solver_gpu: IntProperty(name='Restoration GPU',default=-1,min=-1,max=15,
        description='NCNN device index; -1 automatically selects a Vulkan GPU')
    skin_solver_source: PointerProperty(name='Original Skin Projection',type=bpy.types.Image,
        description='Optional preserved albedo already projected onto this character UV layout')
    skin_solver_protect: PointerProperty(name='Keep Accepted Paint',type=bpy.types.Image,
        description='White retains the currently accepted color when recovering an original projection')
    skin_solver_body_protect: PointerProperty(name='Keep Accepted Body Paint',type=bpy.types.Image,
        description='Body UV mask: white retains accepted body color and generated material baselines; separate from head protection')
    skin_solver_floor: FloatProperty(name='Masked Cleanup Minimum',default=0,min=0,max=1,
        description='Force repair of blurred or non-periodic damage inside an explicit mask; may replace source detail')
    skin_solver_relief: FloatProperty(name='Inferred Relief',default=0,min=0,max=1,
        description='Art-directed relief from cleaned color, not recovered physical depth; zero preserves normals')
    skin_solver_relief_radius: FloatProperty(name='Relief Scale (source texels)',default=3,min=.5,max=16)
    skin_solver_surface_repair: BoolProperty(name='Repair Masked Normals',default=False,
        description='Replace damaged normals with the matching template skin only inside an explicit repair mask')
    skin_solver_surface_mask: PointerProperty(name='Damaged Normal Mask',type=bpy.types.Image,
        description='White replaces damaged shading; black retains the existing sculpt normal')
    skin_solver_job: StringProperty(subtype='DIR_PATH')
    skin_solver_last: StringProperty(subtype='DIR_PATH')
    skin_solver_status: StringProperty()
    paint_radius: IntProperty(name="Paint Radius (pixels)", default=64, min=1, max=1024)
    paint_opacity: FloatProperty(name="Paint Opacity", default=0.2, min=0.01, max=1)
    paint_preserve_color: BoolProperty(name='Keep Local Skin Color', default=False,
        description='Replace fine color detail from the reference while retaining broad source color. Paint only the areas to clean')
    paint_detail_radius: IntProperty(name='Detail Scale (texels)', default=8, min=1, max=64)
    paint_exposure: FloatProperty(name="Reference Exposure", default=0, min=-5, max=5)
    paint_gamma: FloatProperty(name="Reference Gamma", default=1, min=0.1, max=4)
    paint_saturation: FloatProperty(
        name="Reference Saturation", default=1, min=0, max=3
    )
    bake_device: EnumProperty(
        name="Bake Device",
        items=[
            ("CPU", "CPU", ""),
            ("AUTO", "Auto GPU", "OptiX, CUDA, HIP, oneAPI, then CPU"),
        ],
        default="AUTO",
    )
    wrinkle_slot: EnumProperty(
        name="Wrinkle Slot", items=[(n, n, "") for n in ("WM1", "WM2", "WM3")]
    )
    show_wrinkles: BoolProperty(name="Wrinkle Maps", default=False)
    map_bake_status: StringProperty(options={'SKIP_SAVE'})
    sculpt_session_role: EnumProperty(items=[('NONE','Main',''),('AUTHOR','Sculpting Session','')],default='NONE',options={'HIDDEN'})
    sculpt_session_path: StringProperty(subtype='DIR_PATH')
    sculpt_session_status: StringProperty(options={'SKIP_SAVE'})
    bake_open_folder: BoolProperty(name="Open Folder When Finished", default=True)
    final_map_scope: EnumProperty(name='Final Map Parts',default='BOTH',items=[
        ('BOTH','Head and Body','Bake final normal and cavity maps for both parts'),
        ('HEAD','Head Only','Preserve body maps and bake the accepted head'),
        ('BODY','Body Only','Preserve head maps and bake the accepted body')])
    wrinkle_protect_area: BoolProperty(name="Protect Sculpt Area", default=True)
    wrinkle_native_bases: BoolProperty(name="Native Expression Bases", default=True)
    wrinkle_map_size: EnumProperty(name='Wrinkle Resolution',items=[('DEFAULT','Default (1K Normal / Full Color)','1K animated normals; expression colors retain the albedo resolution')]+[(str(size),str(size),'8K bakes can need over 8 GB RAM. Prefer 4K when memory or disk space is limited.' if size == 8192 else '') for size in (256,512,1024,2048,4096,8192)],default='DEFAULT')
    wm1: PointerProperty(type=bpy.types.Object)
    wm2: PointerProperty(type=bpy.types.Object)
    wm3: PointerProperty(type=bpy.types.Object)
    show_expression_check: BoolProperty(name="Expression Check", default=False)
    expression_active: BoolProperty(default=False,options={'HIDDEN'})
    expression_skin: BoolProperty(name='Wrinkles & Blood Flow',default=True,update=expression_maps.toggle)
    expression_skin_status: StringProperty(options={'SKIP_SAVE'})
    unreal_build_root: StringProperty(name='Build Folder',default='/Game/MetaHumans')
    roughness_response_mode: EnumProperty(name='Skin Roughness',items=[
        ('AUTO','Automatic','Derived albedo material adds detail to native skin regions; authored roughness stays absolute'),
        ('SKIN_DETAIL','Native Skin + Detail','Keep native region differences; normalize texture variation around the selected skin finish'),
        ('ABSOLUTE','Absolute Map','Use texture values as absolute roughness with native map-region multipliers')],default='AUTO')
    skin_finish_roughness: FloatProperty(name='Skin Finish',default=.4,min=.1,max=.9,
        description='Base roughness for derived skin in Unreal before Creator adjustment: lower is shinier. Preserves texture detail and native region differences; applied on texture delivery')
    unreal_build_profile: EnumProperty(name='Assembly',items=[('CINEMATIC','UE Cinematic','Full native cinematic pipeline'),('HIGH','UE Optimized High','Native optimized high quality'),('MEDIUM','UE Optimized Medium','Native optimized medium quality'),('LOW','UE Optimized Low','Native optimized low quality')],default='CINEMATIC')
    unreal_eye_overlay_policy: EnumProperty(name='Source Eye Overlays',items=[
        ('NATIVE','Keep Native','Retain native optical shell and tearline materials'),
        ('OPAQUE_SOURCE','Opaque Source','For projected opaque eyes: hide native optical shell/tearline overlays that can create false bands; preserve their geometry and all grooms')],default='NATIVE')
    unreal_build_receipt: StringProperty(subtype='FILE_PATH')
    unreal_build_report: StringProperty()
    unreal_resource_report: StringProperty(options={'SKIP_SAVE'},description='Local prelaunch memory and disk estimate; native assembly rechecks after loading')
    unreal_build_allow_low_memory: BoolProperty(name='Allow Low Memory',default=False,
        description='Explicitly attempt assembly below Unreal\'s 10 GiB available-commit recommendation, matching its native confirmation option; the hard commit floor remains')
    expression_reference: StringProperty(name="Native Expression Reference",subtype="FILE_PATH")
    expression_prepared: StringProperty(name="Prepared Expression Reference",subtype="FILE_PATH")
    expression_previous: StringProperty(name="Previous Expression Reference",subtype="FILE_PATH")
    expression_bridge_dir: StringProperty(name="Compiled Bridge Folder",subtype="DIR_PATH")
    expression_index: IntProperty(name="Expression", default=0, min=0)
    expression_pose: EnumProperty(name="Pose",items=expressions.pose_items,update=expressions.update_pose)
    expression_strength: FloatProperty(
        name="Expression Strength", default=1, min=0, max=1, update=expressions.update_strength
    )
    sync_views: BoolProperty(name="Synchronize Views", default=True)
    generated_head: PointerProperty(type=bpy.types.Object)
    face_controls: CollectionProperty(type=CM_BodyControl)
    show_face_features: BoolProperty(name="Face Features", default=False)
    photo_fit_status: StringProperty(name='Photo Fit Status')
    photo_color_settings: BoolProperty(name='Photo Color Settings',default=False)
    photo_base_skin_settings: BoolProperty(name='Base Skin',default=False,
        description='Choose matching head and body skin before projecting photo color')
    photo_color_size: EnumProperty(name='Photo Color Size',items=[('1024','1K',''),('2048','2K','')],default='2048')
    photo_color_strength: FloatProperty(name='Photo Color Strength',default=1,min=.01,max=1)
    photo_match_skin: BoolProperty(name='Match Existing Skin Tone',default=True)
    photo_blend_neck: BoolProperty(name='Blend Neck to Body',default=True,
        description='Match the attached lower neck to the body while protecting projected face color')
    generated_head_height: FloatProperty(
        name="Generated Head Height", default=0.34, min=0.05, max=1
    )
    identity_seed: IntProperty(name="Random Seed", default=17, min=0)
    identity_coefficients: CollectionProperty(type=CM_IdentityCoefficient)
    coefficient_page: IntProperty(name="Coefficient Page", default=0, min=0, max=25)
    gender: FloatProperty(name="GNM Gender (F–M)", default=0.5, min=0, max=1)
    ethnicity: EnumProperty(
        name="GNM Sampling Group",
        items=[
            ("0", "Middle Eastern", "GNM training category"),
            ("1", "Asian", "GNM training category"),
            ("2", "White", "GNM training category"),
            ("3", "Black", "GNM training category"),
        ],
        default="2",
    )
    identity_mix: FloatVectorProperty(
        name="12 Identity Mix", size=12, default=(0,) * 12, min=-3, max=3
    )
    skin_color: FloatVectorProperty(
        name="Preview Skin Tone",
        size=3,
        subtype="COLOR",
        default=(0.5, 0.28, 0.18),
        min=0,
        max=1,
        update=head_model.update_skin,
    )
    generated_skin: EnumProperty(name='Skin', items=[
        ('COLOR', 'Plain Color', 'Lightweight solid-color preview'),
        ('LOCAL', 'Local Skin Library', 'Your licensed local skins prepared for GNM UVs'),
        ('IMAGE', 'Skin Image', 'Albedo using the generated head GNM UV layout'),
    ], default='COLOR', update=head_model.update_skin)
    generated_skin_image: PointerProperty(name='GNM Skin', type=bpy.types.Image, update=head_model.update_skin)
    generated_local_skin_image: PointerProperty(type=bpy.types.Image)
    generated_skin_preset: EnumProperty(name='Preset', items=skin_library.preset_items, update=skin_library.preset_changed)
    character_skin_preset: EnumProperty(name='Character Skin', items=skin_library.character_preset_items)
    character_skin_image: PointerProperty(type=bpy.types.Image)
    character_applied_skin_image: PointerProperty(type=bpy.types.Image)
    character_skin_applied: StringProperty(default='REFERENCE')
    character_skin_previous: StringProperty()
    character_previous_head_material: PointerProperty(type=bpy.types.Material)
    character_previous_body_material: PointerProperty(type=bpy.types.Material)
    character_previous_head_normal: PointerProperty(type=bpy.types.Image)
    character_previous_body_normal: PointerProperty(type=bpy.types.Image)
    character_previous_skin_image: PointerProperty(type=bpy.types.Image)
    character_previous_generated_image: PointerProperty(type=bpy.types.Image)
    character_previous_head: PointerProperty(type=bpy.types.Object)
    character_previous_body: PointerProperty(type=bpy.types.Object)
    generated_skin_tone: FloatProperty(name='Tone', description='Lighten or darken the selected skin in exposure stops',
        default=0, min=-2, max=2, update=head_model.update_skin)
    add_pores: BoolProperty(name="Add Procedural Base Pores", default=False)
    pore_strength: FloatProperty(
        name="Pore Depth (scene units)", default=0.00002, min=0, max=0.001, precision=6
    )
    pore_spacing: FloatProperty(
        name="Pore Spacing", default=0.0008, min=0.00005, max=0.01, precision=5
    )
    bloodflow: BoolProperty(name="Generate CM Blood Flow", default=False)
    bloodflow_strength: FloatProperty(
        name="Blood Flow Tint", default=0.04, min=0, max=0.3
    )
    wm1_mask: PointerProperty(name="WM1 / CM1 Region Mask", type=bpy.types.Image)
    wm2_mask: PointerProperty(name="WM2 / CM2 Region Mask", type=bpy.types.Image)
    wm3_mask: PointerProperty(name="WM3 / CM3 Region Mask", type=bpy.types.Image)


def ensure_controls(f):
    if len(f.body_controls) != len(anatomy.BODY_CONTROLS):
        f.body_controls.clear()
        for category, key, label in anatomy.BODY_CONTROLS:
            item = f.body_controls.add()
            item.category, item.key, item.label = category, key, label


class CM_OT_feature(bpy.types.Operator):
    bl_idname = "cm.feature"
    bl_label = "Workflow Action"
    bl_options = {"REGISTER", "UNDO"}
    action: StringProperty()

    def execute(self, context):
        s, f = context.scene.cm_settings, context.scene.cm_features
        ensure_controls(f)
        try:
            if eye_fit_jobs.running():raise ValueError('Finish or cancel the source surface fit before another workflow action')
            landmark_tools.finish_active(context)
            if scan_jobs.running():
                raise ValueError('Finish or cancel scan projection before another workflow action')
            if auto_wrap_jobs.running():
                raise ValueError('Finish or cancel Auto Wrap before another workflow action')
            if detail_mask.running(context.scene):
                raise ValueError('Apply or cancel Scan Mask before another workflow action')
            if paint_tools.running():
                raise ValueError('Apply or cancel Smart Paint before another workflow action')
            from . import skin_reconstruction
            if skin_reconstruction.running():
                raise ValueError('Apply or discard skin reconstruction before another workflow action')
            from . import setup_paths

            setup_paths.resolve(f)
            if self.action == 'DISPLACE' and not bpy.app.background:
                scan_jobs.start(context)
                return {'FINISHED'}
            if self.action=='SUBDIVIDE_DETAIL' and not bpy.app.background:
                detail_jobs.start(context,s,f,'SUBDIVIDE','BODY' if s.part=='BODY' or s.part=='FULL_BODY' and f.texture_part=='BODY' else 'HEAD')
                return {'FINISHED'}
            if self.action == 'RECOVER_DETAIL':
                if detail_jobs.running() or context.mode!='OBJECT':
                    raise ValueError('Finish or cancel the current update or sculpt before recovering its surface')
                body=s.part=='BODY' or s.part=='FULL_BODY' and f.texture_part=='BODY'
                detail=f.body_detail if s.part=='FULL_BODY' and body else s.detail
                m.check_mesh(detail,'Saved sculpt')
                previous=(f.body_detail_previous if body else f.head_detail_previous) if f.detail_recovery_manual else None
                if f.detail_recovery_manual and not previous:
                    raise ValueError('Choose the previous sculpt level, or turn off Choose Previous Layer for automatic recovery')
                if previous:
                    detail['cm_detail_previous']=previous
                    detail['cm_detail_previous_selected']=True
                elif detail.get('cm_detail_previous_selected'):
                    del detail['cm_detail_previous_selected']
                    if 'cm_detail_previous' in detail:del detail['cm_detail_previous']
                if not detail_jobs.start(context,s,f,'RECOVER','BODY' if body else 'HEAD'):
                    s.status='This sculpt already has a current saved surface reference.'
                return {'FINISHED'}
            if self.action == 'WRINKLE_BAKE' and not bpy.app.background:
                bake_jobs.start(context,s,f,'WRINKLES',[f.wrinkle_slot])
                return {'FINISHED'}
            if self.action in {'BLOODFLOW_BAKE','BLOODFLOW_ALL'}:
                bake_jobs.start(context,s,f,'BLOODFLOW',[f.wrinkle_slot] if self.action=='BLOODFLOW_BAKE' else ['WM1','WM2','WM3'])
                return {'FINISHED'}
            if self.action in {"NEXT_STEP", "PREVIOUS_STEP"}:
                if context.mode!='OBJECT' or f.body_editing:
                    raise ValueError('Finish the active edit before changing workflow steps')
                steps = WORKFLOW_ORDER
                offset = 1 if self.action == "NEXT_STEP" else -1
                f.ui_stage = steps[
                    max(0, min(len(steps) - 1, steps.index(f.ui_stage) + offset))
                ]
                return {"FINISHED"}
            if self.action == "SAVE_SETUP":
                setup_paths.save(f)
                s.status = "Local model setup saved for new projects."
                return {"FINISHED"}
            from . import wrap_proxy
            actions = {
                "PROXY_DISABLE": wrap_proxy.disable,
                "DECIMATE": workflow.decimate_source,
                "MASKS": workflow.anatomical_masks,
                "SCULPT_REGION": workflow.sculpt_region,
                "DISPLACE": workflow.generate_displacement,
                "APPLY_SCAN": workflow.apply_scan,
                "CLEAR_DETAIL": workflow.clear_detail,
                "ATTACH": workflow.attach_neck,
                "BODY_PREVIEW": workflow.body_preview,
                "A_POSE": workflow.force_a_pose,
                "A_POSE_RESET": workflow.reset_a_pose,
                "SUPPORT": workflow.support_report,
                "AUTO": automation.auto_landmarks,
                "EXPRESSION": workflow.expression_check,
                "CAVITY": textures.bake_cavity,
                "ALPHAS": textures.save_brush_alphas,
                "WRINKLE_CREATE": textures.wrinkle_layer,
                "WRINKLE_BAKE": textures.bake_wrinkle,
                "ACCESSORIES": workflow.update_accessories,
                "SUBDIVIDE_DETAIL": workflow.subdivide_detail,
                "FULL_SCULPT": workflow.full_sculpt_start,
            }
            if self.action in actions:
                if self.action in {
                    "DISPLACE",
                    "APPLY_SCAN",
                    "CLEAR_DETAIL",
                    "SUBDIVIDE_DETAIL",
                    "CAVITY",
                }:
                    with textures.skin_part(s, f):
                        actions[self.action](context, s, f)
                else:
                    actions[self.action](context, s, f)
            elif self.action == "SCULPT_LAYER":
                with textures.skin_part(s, f):
                    if s.detail:
                        raise ValueError("A sculpt layer already exists for this part")
                    m.create_detail(context, s, project_source=False)
                    s.status = "Sculpt layer ready. Sculpt Detail adds skin detail without changing the conform mesh."
            elif self.action in {"GNM_DEFAULT", "GNM_RANDOM", "GNM_UPDATE", "GNM_MIX"}:
                if self.action == "GNM_RANDOM":
                    f.identity_seed += 1
                if self.action == "GNM_MIX":
                    head_model.mix_identities(context, s, f)
                else:
                    head_model.create_head(context, s, f, self.action == "GNM_RANDOM")
            elif self.action in {"BODY_APPLY", "BODY_CANCEL"}:
                if self.action == 'BODY_APPLY' and not bpy.app.background:
                    if detail_jobs.start(context, s, f, 'BODY_APPLY'):
                        return {'FINISHED'}
                if self.action == 'BODY_CANCEL' and detail_jobs.running():
                    detail_jobs.cancel()
                workflow.body_finish(context, s, f, self.action == "BODY_APPLY")
            elif self.action in {"FULL_SCULPT_APPLY", "FULL_SCULPT_CANCEL"}:
                workflow.full_sculpt_finish(
                    context, s, f, self.action == "FULL_SCULPT_APPLY"
                )
            elif self.action == "BODY_RESET":
                for item in f.body_controls:
                    if item.category == f.body_category:
                        item.value = 0
            elif self.action == "BODY_RESET_ALL":
                for item in f.body_controls:
                    item.value = 0
            elif self.action == "WRINKLE_ALL":
                slots=[slot for slot in ('WM1','WM2','WM3') if getattr(f,slot.lower())]
                if not bpy.app.background:bake_jobs.start(context,s,f,'WRINKLES',slots)
                else:
                    for slot in slots:textures.bake_wrinkle(context,s,f,slot)
            elif self.action == "WRINKLE_SCULPT":
                obj = getattr(f, f.wrinkle_slot.lower())
                m.check_mesh(obj, f.wrinkle_slot)
                from . import wrinkles, guided
                wrinkles.sculpt_area(s, f, obj, f.wrinkle_slot)
                guided.focus(context, [obj,*obj.children])
                m.activate(context, obj)
                obj.active_shape_key_index = 1
                bpy.ops.object.mode_set(mode="SCULPT")
            elif self.action == "WRINKLE_AREA":
                from . import wrinkles
                obj = getattr(f, f.wrinkle_slot.lower())
                m.check_mesh(obj, f.wrinkle_slot)
                wrinkles.sculpt_area(s, f, obj, f.wrinkle_slot)
                s.status = f.wrinkle_slot + " sculpt area updated from its region mask."
            elif self.action == 'WRINKLE_RESET':
                from . import wrinkles
                wrinkles.reset_layer(context,s,f,f.wrinkle_slot)
                s.status = f.wrinkle_slot + ' base reset; the previous sculpt is archived.'
            elif self.action == "ADOPT_SOURCE":
                m.check_inputs(s)
                if m.topology(s.source) != m.topology(s.template):
                    raise ValueError(
                        "Exact template vertex order/connectivity required for direct adoption"
                    )
                vertices = m.evaluated_coordinates(context, s.source)
                obj = m.duplicate(context, s.template, s.character_name + "_Conformed")
                obj.shape_key_add(name="Basis")
                workflow.key_from_world(obj, "Wrapped", vertices)
                obj["cm_template_topology"] = m.topology(s.template)
                obj["cm_part"] = s.part
                s.result = obj
                s.status = (
                    "Compatible source topology adopted directly, without projection."
                )
            elif self.action == "NEUTRAL":
                expressions.neutralize(context,s,f)
            elif self.action == "REFINE":
                target=f.body_result if f.body_result and context.active_object==f.body_result else s.result
                if target and region_projection.paired(s.source,target) and not bpy.app.background:
                    region_projection.start(context,target)
                else:m.refine_surface(context, s, f)
            elif self.action == "FINAL_MAPS":
                if not bpy.app.background:bake_jobs.start(context,s,f,'FINAL_MAPS')
                else:textures.final_maps(context, s, f)
            else:
                raise ValueError("Unknown workflow action: " + self.action)
            return {"FINISHED"}
        except Exception as exc:
            traceback.print_exc()
            s.status = str(exc)
            self.report({"ERROR"}, str(exc))
            return {"CANCELLED"}


class CM_OT_project_file(bpy.types.Operator, ImportHelper):
    bl_idname = "cm.project_file"
    bl_label = "Open / Import Project"
    filter_glob: StringProperty(default="*.blend", options={"HIDDEN"})
    part: EnumProperty(
        items=[
            ("RESUME", "Resume Project", ""),
            ("HEAD", "Import Head", ""),
            ("BODY", "Import Body", ""),
        ]
    )

    def execute(self, context):
        try:
            if self.part == "RESUME":
                bpy.ops.wm.open_mainfile(filepath=self.filepath, load_ui=False)
            else:
                workflow.import_project(
                    context,
                    context.scene.cm_settings,
                    context.scene.cm_features,
                    self.filepath,
                    self.part,
                )
            return {"FINISHED"}
        except Exception as exc:
            self.report({"ERROR"}, str(exc))
            return {"CANCELLED"}


class CM_OT_load_image(bpy.types.Operator, ImportHelper):
    bl_idname = "cm.load_image"
    bl_label = "Load Skin Image"
    filter_glob: StringProperty(
        default="*.png;*.jpg;*.jpeg;*.tif;*.tga;*.exr", options={"HIDDEN"}
    )
    reference: BoolProperty(default=False)

    def execute(self, context):
        try:
            s, f = context.scene.cm_settings, context.scene.cm_features
            if paint_tools.running():
                raise ValueError('Apply or cancel Smart Paint before loading another image')
            if not self.reference and s.source and s.source == f.generated_head and s.source.get('cm_gnm'):
                return bpy.ops.cm.load_generated_skin(filepath=self.filepath)
            image = bpy.data.images.load(self.filepath, check_existing=True)
            if self.reference:
                if s.part == 'BODY' or s.part == 'FULL_BODY' and f.texture_part == 'BODY':
                    f.reference_skin_body = image
                else:
                    f.reference_skin = image
            else:
                m.check_mesh(s.source, "Source")
                material = bpy.data.materials.new(s.source.name + "_Albedo")
                material.use_nodes = True
                node = material.node_tree.nodes.new("ShaderNodeTexImage")
                node.image = image
                material.node_tree.links.new(
                    node.outputs["Color"],
                    material.node_tree.nodes.get("Principled BSDF").inputs[
                        "Base Color"
                    ],
                )
                s.source.data.materials.clear()
                s.source.data.materials.append(material)
                for p in s.source.data.polygons:
                    p.material_index = 0
            return {"FINISHED"}
        except Exception as exc:
            self.report({"ERROR"}, str(exc))
            return {"CANCELLED"}


class CM_OT_load_generated_skin(bpy.types.Operator, ImportHelper):
    """Load and pack a skin albedo with the generated head's GNM UV layout"""
    bl_idname = 'cm.load_generated_skin'
    bl_label = 'Choose GNM Skin Image'
    bl_options = {'REGISTER', 'UNDO'}
    filter_glob: StringProperty(default='*.png;*.jpg;*.jpeg;*.tif;*.tiff;*.tga;*.exr', options={'HIDDEN'})

    def execute(self, context):
        f = context.scene.cm_features
        try:
            m.check_mesh(f.generated_head, 'Generate a GNM head first')
            if not f.generated_head.get('cm_gnm'):
                raise ValueError('Choose the generated GNM head')
            image = bpy.data.images.load(self.filepath, check_existing=True)
            head_model.validate_skin_image(image)
            image.pack()
            f.generated_skin_image = image
            f.generated_skin = 'IMAGE'
            context.scene.cm_settings.status = 'GNM skin loaded. Tone updates immediately; the image stays with the saved head.'
            return {'FINISHED'}
        except Exception as exc:
            self.report({'ERROR'}, str(exc))
            return {'CANCELLED'}


class CM_OT_custom_body(bpy.types.Operator, ExportHelper):
    bl_idname = "cm.save_custom_body"
    bl_label = "Save Custom Body"
    filename_ext = ".obj"
    filter_glob: StringProperty(default="*.obj", options={"HIDDEN"})

    def execute(self, context):
        s, f = context.scene.cm_settings, context.scene.cm_features
        try:
            obj = f.body_result or s.result
            m.check_mesh(obj, "Body")
            with m.selection_state(context):
                m.activate(context, obj)
                bpy.ops.wm.obj_export(
                    filepath=self.filepath,
                    export_selected_objects=True,
                    export_materials=False,
                )
            native_body.save_custom_state(obj,self.filepath)
            return {"FINISHED"}
        except Exception as exc:
            self.report({"ERROR"}, str(exc))
            return {"CANCELLED"}


class CM_OT_load_custom_body(bpy.types.Operator, ImportHelper):
    bl_idname = "cm.load_custom_body"
    bl_label = "Load Custom Body"
    bl_options = {"REGISTER", "UNDO"}
    filter_glob: StringProperty(default="*.obj;*.fbx", options={"HIDDEN"})

    def execute(self, context):
        try:
            from .body_library import load_custom_body

            load_custom_body(
                context,
                context.scene.cm_settings,
                context.scene.cm_features,
                self.filepath,
            )
            return {"FINISHED"}
        except Exception as exc:
            context.scene.cm_settings.status = str(exc)
            self.report({"ERROR"}, str(exc))
            return {"CANCELLED"}


def button(layout, label, action):
    layout.operator("cm.feature", text=label).action = action


class Panel:
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "MetaHuman Maker"

    @classmethod
    def poll(cls, context):
        stages = {
            "CM_PT_advanced": "WRAP",
            "CM_PT_anatomy": "REFINE",
            "CM_PT_region_projection": "REFINE",
            "CM_PT_detail_tools": "DETAIL",
            "CM_PT_generation": "START",
            "CM_PT_projects": "START",
        }
        return (
            context.scene.cm_features.advanced_ui
            and context.scene.cm_features.ui_stage == stages.get(cls.__name__)
        )


def draw_region_projection(layout,context):
    f=context.scene.cm_features
    box=layout.box();box.prop(f,'projection_region',text='Region')
    if context.active_object==f.body_result and f.body_result:
        box.operator_menu_enum('cm.seed_anatomical_region','root',text='Use Anatomical Limb Region')
    box.operator('cm.mark_fit_region')
    review=box.row(align=True);review.enabled=context.mode=='OBJECT' and not region_projection.running()
    review.operator('cm.review_source_region',text='Inspect Source').action='REPORT'
    review.operator('cm.review_source_region',text='Select Small Patches').action='ISLANDS'
    source=context.scene.cm_settings.source
    if source and source.get('cm_source_region_review'):
        import json
        try:
            inspection=json.loads(source['cm_source_region_review'])
            if inspection['region']=='CM_Fit_'+f.projection_region:
                box.label(text=str(inspection['component_count'])+' connected source patches',icon='INFO')
                if f.advanced_ui:
                    for patch in inspection['components'][:8]:
                        select=box.operator('cm.review_source_region',text='Review patch '+str(patch['component_id'])+' ('+str(patch['polygons'])+' faces)')
                        select.action='COMPONENT';select.component_id=patch['component_id'];select.stitch_exact_seams=inspection['connectivity']=='EXACT_COORDINATE_EDGE_OR_ORIGINAL_POLYGON'
                box.label(text='Patch size does not determine validity.')
        except (ValueError,KeyError,TypeError):pass
    actions=box.column();active=context.active_object
    actions.enabled=bool(active and active in (context.scene.cm_settings.result,f.head_result,f.body_result) and context.mode=='OBJECT' and not region_projection.running())
    initialize=actions.operator('cm.fit_named_region',text='Initialize Branch');initialize.mode='TRANSPORT';initialize.reconstruct=False;initialize.complete_missing=False;initialize.rebuild_missing=False
    row=actions.row(align=True)
    quick=row.operator('cm.fit_named_region',text='Quick Fit');quick.mode='REFINE';quick.reconstruct=False;quick.complete_missing=False;quick.rebuild_missing=False
    accurate=row.operator('cm.fit_named_region',text='Accurate Fit');accurate.mode='ACCURATE';accurate.reconstruct=False;accurate.complete_missing=False;accurate.source_observation='NEAREST';accurate.observation_rings=2.;accurate.rebuild_missing=False
    conform=actions.operator('cm.fit_named_region',text='Normal-Line Fit');conform.mode='ACCURATE';conform.reconstruct=False;conform.complete_missing=False;conform.source_observation='NORMAL_LINE';conform.observation_rings=2.;conform.rebuild_missing=False
    completion=actions.operator('cm.fit_named_region',text='Complete Partial Surface');completion.mode='ACCURATE';completion.reconstruct=False;completion.complete_missing=True;completion.source_observation='NORMAL_LINE';completion.observation_rings=2.;completion.rebuild_missing=False
    if f.advanced_ui:
        missing=actions.operator('cm.fit_named_region',text='Rebuild Missing Shape');missing.mode='ACCURATE';missing.reconstruct=False;missing.complete_missing=True;missing.rebuild_missing=True;missing.source_observation='NORMAL_LINE';missing.observation_rings=2.
        if active and active==f.body_result:
            alignment=actions.operator('cm.align_pelvis_attachment',text='Align Paired Hips');alignment.curvature_polish=False;alignment.curvature_model='CURVATURE'
        if active and active==f.body_result:
            hip_polish=actions.operator('cm.align_pelvis_attachment',text='Experimental Hip Curvature');hip_polish.curvature_polish=True;hip_polish.curvature_model='CURVATURE'
            hip_flow=actions.operator('cm.align_pelvis_attachment',text='Refine Paired Hip Continuity');hip_flow.curvature_polish=True;hip_flow.curvature_model='NORMAL_FLOW'
    if f.advanced_ui:
        curvature=actions.operator('cm.fit_named_region',text='Experimental Source Curvature');curvature.mode='ACCURATE';curvature.reconstruct=False;curvature.complete_missing=True;curvature.rebuild_missing=True;curvature.curvature_polish=True;curvature.curvature_model='CURVATURE';curvature.source_observation='NORMAL_LINE'
        flow=actions.operator('cm.fit_named_region',text='Refine Surface Continuity');flow.mode='ACCURATE';flow.reconstruct=False;flow.complete_missing=True;flow.rebuild_missing=True;flow.curvature_polish=True;flow.curvature_model='NORMAL_FLOW';flow.source_observation='NORMAL_LINE'
    repair=actions.operator('cm.fit_named_region',text='Repair Attachment');repair.mode='ACCURATE';repair.reconstruct=True;repair.complete_missing=False;repair.repair_target='AUTO';repair.observation_rings=2.;repair.rebuild_missing=False
    polish=actions.operator('cm.fit_named_region',text='Polish This Region');polish.mode='POLISH';polish.reconstruct=False;polish.complete_missing=False;polish.rebuild_missing=False
    actions.operator('cm.project_region_detail',text='Project Source Detail in Region')
    for action in (initialize,quick,accurate,conform,completion,repair,polish):action.curvature_polish=False;action.curvature_model='CURVATURE'
    if f.advanced_ui:missing.curvature_polish=False;missing.curvature_model='CURVATURE'
    box.label(text='Mark source faces and result vertices')
    box.label(text='with the same region name.')
    box.label(text='Fit the selected result in Object Mode.')
    target=f.body_result if context.active_object==f.body_result else context.scene.cm_settings.result
    from . import regional_attempt
    attempt=regional_attempt.read(target) if target else {}
    latest=attempt.get('status')
    if latest in {'FAILED','CANCELLED','INTERRUPTED','COMMITTED_WITH_REPORT_ERROR'}:
        box.label(text='Latest '+str(attempt.get('mode','fit')).lower()+': '+latest.lower().replace('_',' '),icon='ERROR' if latest in {'FAILED','COMMITTED_WITH_REPORT_ERROR'} else 'INFO')
        box.label(text='Candidate not applied.' if not attempt.get('candidate_published') else 'Fit applied; inspect report error.',icon='INFO')
        if attempt.get('error'):box.label(text=attempt['error'][:160])
    elif latest=='RUNNING':box.label(text='Regional fit running...',icon='TIME')
    if target and target.get('cm_region_projection') and latest not in {'RUNNING','FAILED','CANCELLED','INTERRUPTED','COMMITTED_WITH_REPORT_ERROR'}:
        import json
        try:
            reports=json.loads(target['cm_region_projection'])
            review=sum(bool(r.get('review_required')) for r in reports)
            retained=sum(not r.get('accepted',False) for r in reports)
            box.label(text=str(review)+(' patch needs' if review==1 else ' patches need')+' review.',icon='INFO')
            if retained:box.label(text=str(retained)+(' patch unchanged.' if retained==1 else ' patches unchanged.'),icon='INFO')
            inferred=sum(int(r.get('inferred_physical_vertices',0)) for r in reports if r.get('accepted'))
            if inferred:box.label(text='Inferred surface: '+str(inferred)+' vertices.',icon='INFO')
            if f.advanced_ui:
                for report in reports:
                    continuity=report.get('continuation',{}).get('inferred_completion',{})
                    sampling=continuity.get('source_sample_transport',{})
                    if sampling.get('accepted'):box.label(text='Source sampling redistributed.')
                    elif sampling.get('trials'):box.label(text='Source sampling retained after quality checks.',icon='INFO')
                    if continuity.get('passes'):
                        selected='source transport' if continuity['selected_pass']==0 else str(continuity['selected_pass'])
                        box.label(text='Continuity passes: '+str(len(continuity['passes']))+'; selected '+selected)
                    if continuity.get('selected_endpoint_fraction',1)<1:box.label(text='Recovery step: '+format(continuity['selected_endpoint_fraction']*100,'.0f')+'%')
                    quality=report.get('source_alignment',{}).get('after_angle_quality',{})
                    if quality.get('faces',0):
                        box.label(text='Supported faces: 95% within '+format(quality['p95_degrees'],'.1f')+' degrees')
                        box.label(text='Peak direction error: '+format(quality['maximum_degrees'],'.1f')+' degrees')
                        previous=report.get('source_alignment',{}).get('before_angle_quality',{})
                        if previous and quality['p99_degrees']>previous['p99_degrees']+.1:
                            box.label(text='99th percentile worsened by '+format(quality['p99_degrees']-previous['p99_degrees'],'.1f')+' degrees',icon='INFO')
        except (TypeError,ValueError):pass
    if region_projection.running():box.operator('cm.cancel_region_projection')


class CM_PT_region_projection(Panel,bpy.types.Panel):
    bl_label='Surface Correspondence'
    bl_idname='CM_PT_region_projection'
    def draw(self,context):draw_region_projection(self.layout,context)


class CM_PT_advanced(Panel, bpy.types.Panel):
    bl_label = "Workflow • Preparation & Automation"
    bl_idname = "CM_PT_advanced"

    def draw(self, context):
        f = context.scene.cm_features
        l = self.layout
        l.prop(f, "full_head")
        l.prop(f, "front_axis")
        l.prop(f, "wrap_method")
        l.prop(f, "wrap_quality")
        if not f.advanced_ui:
            return
        l.prop(f, "landmark_kind")
        button(l, "Detect Landmarks Locally", "AUTO")
        l.prop(f, "conservative")
        l.prop(f, "preserve_mouth_interior")
        l.prop(f, "is_mask")
        l.prop(f, "decimate_ratio")
        button(l, "Create Solver Proxy", "DECIMATE")
        l.prop(f, "proxy")
        l.operator("cm.load_image", text="Replace Source Albedo")
        button(l, "Adopt Compatible Source Topology", "ADOPT_SOURCE")
        button(l, "Refine Surface Fit", "REFINE")
        draw_region_projection(l,context)
        l.prop(f, "model_dir")
        l.prop(f, "inference_python")
        button(l, "Remember Model Paths", "SAVE_SETUP")
        l.operator("cm.dual_views")
        l.prop(f, "sync_views")


class CM_PT_anatomy(Panel, bpy.types.Panel):
    bl_label = "Workflow • Anatomy & Body"
    bl_idname = "CM_PT_anatomy"
    bl_options = {"DEFAULT_CLOSED"}

    def draw(self, context):
        f = context.scene.cm_features
        l = self.layout
        for name in (
            "head_result",
            "body_template",
            "body_result",
            "eye_left",
            "eye_right",
            "teeth",
        ):
            l.prop(f, name)
        button(l, "Update Eyes & Teeth from Head", "ACCESSORIES")
        l.operator('cm.cancel_source_eyes' if eye_fit_jobs.running() else 'cm.fit_source_eyes')
        if not eye_fit_jobs.running():l.operator('cm.fit_source_ears')
        button(l, "Sculpt Full Model Together", "FULL_SCULPT")
        row = l.row(align=True)
        button(row, "Apply Full Sculpt", "FULL_SCULPT_APPLY")
        button(row, "Cancel", "FULL_SCULPT_CANCEL")
        button(l, "Attach Head to Body", "ATTACH")
        button(l, "Force A Pose", "A_POSE")
        button(l, "Create Editable Region Masks", "MASKS")
        l.prop(f, "sculpt_region")
        button(l, "Sculpt Selected Region", "SCULPT_REGION")
        l.prop(f, "restore_mode")
        l.prop(f, "restore_radius")
        l.prop(f, "restore_strength")
        if f.restore_mode=='SURFACE':l.prop(f,'restore_source_adherence')
        l.prop(f, "mirror")
        l.operator("cm.restore_brush")
        l.operator('cm.repair_oral_clearance')
        l.prop(f, "expression_index")
        l.prop(f, "expression_strength")
        button(l, "Check Reference Expression", "EXPRESSION")
        button(l, "Return to Neutral", "NEUTRAL")
        button(l, "Start Body Pass / Preview", "BODY_PREVIEW")
        l.prop(f, "body_category")
        for item in f.body_controls:
            if item.category == f.body_category:
                l.prop(item, "value", text='Masculine ↔ Feminine' if f.body_backend=='NATIVE' and item.key=='gender' else item.label)
        row = l.row(align=True)
        button(row, "Apply Body Pass", "BODY_APPLY")
        button(row, "Cancel", "BODY_CANCEL")
        button(l, "Reset Section", "BODY_RESET")
        l.operator("cm.save_custom_body")


class CM_PT_detail_tools(Panel, bpy.types.Panel):
    bl_label = "Workflow • Detail, Paint & Wrinkles"
    bl_idname = "CM_PT_detail_tools"
    bl_options = {"DEFAULT_CLOSED"}

    def draw(self, context):
        f = context.scene.cm_features
        l = self.layout
        l.prop(f, "use_detail_mask")
        l.prop(f, "neck_fade")
        button(l, "Generate Displacement", "DISPLACE")
        button(l, "Apply Scan Data Once", "APPLY_SCAN")
        button(l, "Clear Detail Layer", "CLEAR_DETAIL")
        button(l, "Subdivide Current Detail (up to LV3)", "SUBDIVIDE_DETAIL")
        l.prop(f, "bake_device")
        l.prop(f, "cavity_radius")
        button(l, "Bake Curvature Cavity", "CAVITY")
        button(l, "Create 14 Skin Alphas", "ALPHAS")
        l.prop(f, "add_pores")
        l.prop(f, "pore_strength")
        l.prop(f, "pore_spacing")
        normal_composition.draw(l,context.scene)
        simple_shading.draw(l,context.scene)
        button(l, "Create Final Normal + Cavity Maps", "FINAL_MAPS")
        body = context.scene.cm_settings.part == 'BODY' or context.scene.cm_settings.part == 'FULL_BODY' and f.texture_part == 'BODY'
        l.prop(f, 'reference_skin_body' if body else 'reference_skin')
        l.operator("cm.load_image", text="Load Reference Skin").reference = True
        for prop in (
            "paint_radius",
            "paint_opacity",
            "paint_exposure",
            "paint_gamma",
            "paint_saturation",
        ):
            l.prop(f, prop)
        l.operator("cm.reference_paint")
        l.prop(f, 'paint_preserve_color')
        if f.paint_preserve_color:l.prop(f, 'paint_detail_radius')
        l.prop(f, "wrinkle_slot")
        button(l, "Create Wrinkle Sculpt Layer", "WRINKLE_CREATE")
        button(l, "Sculpt Selected Wrinkle Layer", "WRINKLE_SCULPT")
        l.operator("cm.finish_sculpt", text="Apply / Finish Wrinkle Sculpt")
        l.prop(f, f.wrinkle_slot.lower() + "_mask")
        l.prop(f, "bloodflow")
        l.prop(f, "bloodflow_strength")
        button(l, "Bake Selected Wrinkle", "WRINKLE_BAKE")
        button(l, "Bake All Wrinkle Slots", "WRINKLE_ALL")
        button(l, "Update Selected Expression Color", "BLOODFLOW_BAKE")
        button(l, "Update All Expression Colors", "BLOODFLOW_ALL")


class CM_PT_generation(Panel, bpy.types.Panel):
    bl_label = "GNM • Head Generation"
    bl_idname = "CM_PT_generation"
    bl_options = {"DEFAULT_CLOSED"}

    def draw(self, context):
        f = context.scene.cm_features
        l = self.layout
        if not f.advanced_ui:
            l.prop(f, "identity_seed")
            button(l, "Generate a New Face", "GNM_RANDOM")
            if f.generated_head:
                head_model.draw_skin(l, f)
            l.label(text="Creates your Source; keep a MetaHuman Template.", icon="INFO")
            return
        for prop in (
            "model_dir",
            "generated_head_height",
            "identity_seed",
            "gender",
            "ethnicity",
        ):
            l.prop(f, prop)
        head_model.draw_skin(l, f)
        button(l, "Generate Default GNM Head", "GNM_DEFAULT")
        button(l, "Random Semantic Face", "GNM_RANDOM")
        l.operator("cm.face_mixer", text="Face Mixer Wheel")
        l.prop(f, "identity_mix")
        button(l, "Apply 12 Identity Mix", "GNM_MIX")
        for item in f.face_controls:
            l.prop(item, "value", text=item.label)
        l.prop(f, "coefficient_page")
        for item in list(f.identity_coefficients)[
            f.coefficient_page * 10 : f.coefficient_page * 10 + 10
        ]:
            l.prop(item, "value", text="Identity " + str(item.index + 1))
        button(l, "Update Generated Head / Source", "GNM_UPDATE")


class CM_PT_projects(Panel, bpy.types.Panel):
    bl_label = "Projects • Reuse & Diagnostics"
    bl_idname = "CM_PT_projects"
    bl_options = {"DEFAULT_CLOSED"}

    def draw(self, context):
        l = self.layout
        for part, label in [
            ("RESUME", "Resume Project"),
            ("HEAD", "Import Head from Project"),
            ("BODY", "Import Body from Project"),
        ]:
            l.operator("cm.project_file", text=label).part = part
        button(l, "Export Local Support Report", "SUPPORT")


class CM_OT_dual_views(bpy.types.Operator):
    bl_idname = "cm.dual_views"
    bl_label = "Source / Result Comparison"
    bl_description = "Compare source and result with linked navigation; close returns to one result view"
    action: EnumProperty(items=[('TOGGLE','Toggle',''),('OPEN','Open',''),('CLOSE','Close',''),('FRAME','Frame Both','')], default='TOGGLE')

    def execute(self, context):
        from . import compare_views
        try:
            current = compare_views.recover(context.window) if context.window else None
            if self.action == 'CLOSE' or self.action == 'TOGGLE' and current:
                compare_views.close_views(context)
            else:
                session = compare_views.open_views(context)
                if self.action == 'FRAME':
                    compare_views.check_idle(context)
                    compare_views.frame(context, session)
            return {'FINISHED'}
        except Exception as exc:
            context.scene.cm_settings.status = str(exc)
            self.report({'ERROR'}, str(exc))
            return {'CANCELLED'}


CLASSES = (
    CM_BodyControl,
    CM_IdentityCoefficient,
    CM_Features,
    CM_OT_feature,
    CM_OT_project_file,
    CM_OT_dual_views,
    CM_OT_load_image,
    CM_OT_load_generated_skin,
    CM_OT_custom_body,
    CM_OT_load_custom_body,
    CM_OT_face_mixer,
    *region_projection.CLASSES,
    *regional_scan.CLASSES,
    *normal_composition.CLASSES,
    *simple_shading.CLASSES,
    *surface_heal.CLASSES,
    *eye_fit_jobs.CLASSES,
    *eye_color_jobs.CLASSES,
    *native_eye_update.CLASSES,
    *native_return.CLASSES,
    *ear_fit.CLASSES,
    *skin_projection_history.CLASSES,
    *skin_library.CLASSES,
    *character_skin.CLASSES,
    *lookdev.CLASSES,
    *bake_jobs.CLASSES,
    *detail_jobs.CLASSES,
    *wrinkle_jobs.CLASSES,
    *WRINKLE_CLASSES,
    *SCULPT_SESSION_CLASSES,
    *expressions.CLASSES,
    *auto_wrap_jobs.CLASSES,
    *PAINT_CLASSES,
    *skin_bake_layers_ui.CLASSES,
    *head_revision.CLASSES,
    CM_PT_advanced,
    CM_PT_region_projection,
    CM_PT_anatomy,
    CM_PT_detail_tools,
    CM_PT_generation,
    CM_PT_projects,
)


def register():
    for cls in CLASSES:
        bpy.utils.register_class(cls)
    bpy.types.Scene.cm_features = PointerProperty(type=CM_Features)
    from . import skin_brushes
    skin_brushes.register()
    detail_mask.register()
    scan_jobs.register()
    landmark_tools.register()
    restore_brush.register()
    paint_tools.register()
    surface_heal.register()
    eye_fit_jobs.register()
    eye_color_jobs.register()
    from . import compare_views
    compare_views.register()
    native_body.register()
    bake_jobs.register()
    detail_jobs.register()
    wrinkle_jobs.register()
    sculpt_session.register()
    unreal_jobs.register()
    skin_library.register()
    expressions.register()
    auto_wrap_jobs.register()
    region_projection.register()


def unregister():
    eye_color_jobs.unregister()
    eye_fit_jobs.unregister()
    surface_heal.unregister()
    region_projection.unregister()
    auto_wrap_jobs.unregister()
    landmark_tools.unregister()
    scan_jobs.unregister()
    if detail_mask.running(bpy.context.scene):
        detail_mask.finish(bpy.context, apply=False)
    detail_mask.unregister()
    from . import skin_brushes
    skin_brushes.unregister()
    from . import compare_views
    compare_views.unregister()
    paint_tools.unregister()
    wrinkle_jobs.unregister()
    skin_library.unregister()
    unreal_jobs.unregister()
    sculpt_session.unregister()
    detail_jobs.unregister()
    bake_jobs.unregister()
    native_body.unregister()
    restore_brush.unregister()
    lookdev.unregister()
    expressions.unregister()
    del bpy.types.Scene.cm_features
    for cls in reversed(CLASSES):
        bpy.utils.unregister_class(cls)
