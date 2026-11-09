"""Prepare private reference libraries from explicitly assigned licensed templates."""
import hashlib
import json
import os
from pathlib import Path
import uuid

import bpy
import numpy as np
from bpy_extras.io_utils import ExportHelper, ImportHelper
from mathutils import Matrix, Vector

from . import mesh_ops as m, resource_budget, setup_paths


REFERENCE_ROLES=('head','body','eye_left','eye_right','teeth')


def validate_objects(objects, *, require_body=False):
    """Check loaded references before replacing any active character pointers."""
    found={}
    for obj in objects:
        if not obj:continue
        role=obj.get('cm_library_role')
        if role not in REFERENCE_ROLES:continue
        if role in found:raise ValueError('Reference library contains more than one '+role+'; save one mesh per role')
        found[role]=obj
    for role in ('head','body') if require_body else ('head',):
        if role not in found:
            raise ValueError('Reference library has no tagged '+role+' template. Assign licensed templates and use Save Assigned Templates as Library.')
    resource_budget.require('validate reference geometry',128*resource_budget.MIB+resource_budget.meshes(found.values()))
    for role,obj in found.items():
        m.check_mesh(obj,role+' reference')
        if not obj.data.polygons or not obj.data.uv_layers:
            raise ValueError(role+' reference needs surface faces and its original UV layout')
        if not np.isfinite(np.asarray(obj.matrix_world)).all() or not np.isfinite(m.coordinates(obj)).all():
            raise ValueError(role+' reference has invalid coordinates; repair it before saving the library')
    return found


def inspect(filename):
    """Inspect library copies without linking them or retaining appended IDs."""
    path=Path(filename).resolve()
    if path.suffix.lower()!='.blend' or not path.is_file():raise ValueError('Choose an existing reference .blend library')
    resource_budget.require('inspect reference library',512*resource_budget.MIB+4*path.stat().st_size)
    before=set(bpy.data.user_map())
    try:
        with bpy.data.libraries.load(str(path),link=False) as (available,loaded):
            loaded.objects=available.objects
        found=validate_objects(loaded.objects)
        return dict(file=str(path),roles=list(found),vertices={role:len(obj.data.vertices) for role,obj in found.items()},
                    missing=[role for role in REFERENCE_ROLES if role not in found])
    finally:
        added=set(bpy.data.user_map())-before
        if added:bpy.data.batch_remove(ids=added)


def select(context,filename):
    report=inspect(filename)
    f=context.scene.cm_features
    names=('reference_library','model_dir','inference_python','skin_solver_sr_python')
    previous={name:getattr(f,name) for name in names}
    try:
        f.reference_library=report['file'];setup_paths.save(f)
    except (OSError,ValueError,RuntimeError):
        for name,value in previous.items():setattr(f,name,value)
        raise
    return report


def roles(context):
    s,f=context.scene.cm_settings,context.scene.cm_features
    if context.mode!='OBJECT' or f.full_sculpt or f.body_editing or f.restore_editing:
        raise ValueError('Finish the active edit and return to Object Mode before saving reference templates')
    result={'head':s.template,'body':f.body_template}
    result.update({key:getattr(f,key) for key in ('eye_left','eye_right','teeth') if getattr(f,key)})
    if not result['head'] or not result['body']:
        raise ValueError('Assign the licensed neutral LOD0 Head Template and Body Template first')
    if len(set(result.values()))!=len(result):raise ValueError('Assign a different mesh to each reference role')
    for role,obj in result.items():
        m.check_mesh(obj,role+' reference')
        if not len(obj.data.polygons) or not obj.data.uv_layers:
            raise ValueError(role+' reference needs surface faces and its original UV layout')
    return result


def _trees(material):
    pending=[material.node_tree] if material.use_nodes and material.node_tree else []
    seen=set()
    while pending:
        tree=pending.pop()
        if tree in seen:continue
        seen.add(tree);yield tree
        pending.extend(node.node_tree for node in tree.nodes if node.type=='GROUP' and node.node_tree)


def _tag_maps(material):
    if not material.use_nodes:return
    for node in material.node_tree.nodes:
        if node.type!='BSDF_PRINCIPLED':continue
        for socket,kind in [('Base Color','ALBEDO'),('Roughness','ROUGHNESS'),('Metallic','METALLIC'),('Normal','NORMAL')]:
            if socket not in node.inputs or not node.inputs[socket].is_linked:continue
            source=node.inputs[socket].links[0].from_node
            if kind=='NORMAL' and source.type=='NORMAL_MAP' and source.inputs['Color'].is_linked:
                source=source.inputs['Color'].links[0].from_node
            if source.type=='TEX_IMAGE' and source.image:source['cm_map_kind']=kind


def save(context,filename):
    """Write packed static copies; keep source objects, images and project intact."""
    filename=Path(filename).resolve();receipt=filename.with_suffix('.reference.json')
    if filename.suffix.lower()!='.blend' or filename.exists() or receipt.exists():
        raise ValueError('Choose a fresh .blend filename; previous libraries are retained')
    selected=roles(context);materials={mat for obj in selected.values() for mat in obj.data.materials if mat}
    images={node.image for mat in materials for tree in _trees(mat) for node in tree.nodes if node.type=='TEX_IMAGE' and node.image}
    if any(image.source not in {'FILE','GENERATED'} for image in images):
        raise ValueError('Reference packing supports still FILE/GENERATED images; bake tiled, sequence or movie textures first')
    image_bytes=sum(int(image.size[0])*int(image.size[1])*16 for image in images)
    resource_budget.require('save reference library',512*resource_budget.MIB+3*image_bytes+resource_budget.meshes(selected.values()))
    resource_budget.require_disk('save reference library',filename,32*resource_budget.MIB+2*image_bytes+resource_budget.meshes(selected.values()))
    scale=float(context.scene.unit_settings.scale_length)
    if not np.isfinite(scale) or scale<=0:raise ValueError('Choose a valid scene unit scale')
    from . import body_pose
    skeleton=body_pose.capture_skeleton(selected['body'],world=True)
    if not skeleton and selected['body'].get(body_pose.KEY):
        skeleton=json.loads(selected['body'][body_pose.KEY])
        for bone in skeleton['bones'].values():
            bone['head']=list(selected['body'].matrix_world@Vector(bone['head']))
    if skeleton:
        for bone in skeleton['bones'].values():bone['head']=(np.asarray(bone['head'])*scale).tolist()
    points={role:m.evaluated_coordinates(context,obj)*scale for role,obj in selected.items()}
    if any(len(points[role])!=len(obj.data.vertices) or not np.isfinite(points[role]).all() for role,obj in selected.items()):
        raise ValueError('Reference modifiers must preserve original vertex identities and finite coordinates')
    before=set(bpy.data.user_map());copies={};group_copies={};image_copies={};material_copies={}
    pending=filename.with_name('.'+filename.stem+'-'+uuid.uuid4().hex[:12]+'.blend')
    def private_tree(tree):
        if tree in group_copies:return group_copies[tree]
        value=tree.copy();group_copies[tree]=value;value.animation_data_clear();fill_tree(value);return value
    def fill_tree(tree):
        for node in tree.nodes:
            if node.type=='GROUP' and node.node_tree:node.node_tree=private_tree(node.node_tree)
            if node.type=='TEX_IMAGE' and node.image:
                original=node.image
                if original not in image_copies:
                    from .textures import copy_image_pixels
                    private=copy_image_pixels(original);image_copies[original]=private
                    private.pack()
                    if not private.packed_file:raise ValueError('Could not pack reference image '+original.name)
                node.image=image_copies[original]
    def private_material(original):
        if original not in material_copies:
            value=original.copy();material_copies[original]=value;value.animation_data_clear()
            for key in list(value.keys()):del value[key]
            if value.use_nodes and value.node_tree:
                value.node_tree.animation_data_clear();fill_tree(value.node_tree);_tag_maps(value)
        return material_copies[original]
    try:
        with m.selection_state(context):
            for role,original in selected.items():
                obj=m.duplicate(context,original,'Local_MetaHuman_'+role);copies[role]=obj
                obj.parent=None;obj.animation_data_clear()
                for constraint in list(obj.constraints):obj.constraints.remove(constraint)
                for key in list(obj.keys()):del obj[key]
                for key in list(obj.data.keys()):del obj.data[key]
                obj.matrix_world=Matrix.Identity(4)
                obj.data.vertices.foreach_set('co',points[role].astype(np.float32).ravel());obj.data.update()
                indices=[polygon.material_index for polygon in obj.data.polygons]
                assigned=list(original.data.materials);obj.data.materials.clear()
                for mat in assigned:obj.data.materials.append(private_material(mat) if mat else None)
                for polygon,index in zip(obj.data.polygons,indices):polygon.material_index=index
                obj['cm_library_role']=role;obj['cm_front_axis']=context.scene.cm_features.front_axis
                if role=='body' and skeleton:
                    skeleton.update(coordinates=body_pose.coordinate_signature(obj),topology=m.topology(obj))
                    obj[body_pose.KEY]=json.dumps(skeleton)
            filename.parent.mkdir(parents=True,exist_ok=True)
            bpy.data.libraries.write(str(pending),set(copies.values()),fake_user=True,compress=True,path_remap='ABSOLUTE')
            with bpy.data.libraries.load(str(pending),link=False) as (available,_):
                if set(available.objects)!={obj.name for obj in copies.values()}:
                    raise ValueError('A reference shader depends on other scene objects; use standalone texture materials before packing')
            if os.name=='nt':pending.rename(filename)
            else:os.link(pending,filename);pending.unlink()
            report=dict(schema=1,status='complete',file=str(filename),sha256=hashlib.sha256(filename.read_bytes()).hexdigest(),
                roles={role:dict(object=obj.name,vertices=len(obj.data.vertices),topology=m.topology(obj)) for role,obj in copies.items()},
                packed_images=len(image_copies),body_pose_included=bool(skeleton),coordinates='meters; original world alignment',source_objects_modified=False)
            receipt.write_text(json.dumps(report,indent=2),encoding='utf-8')
    finally:
        added=set(bpy.data.user_map())-before
        if added:bpy.data.batch_remove(ids=added)
        pending.unlink(missing_ok=True)
    f=context.scene.cm_features;f.reference_library=str(filename)
    setup_paths.save(f)
    return report


class CM_OT_save_reference_library(bpy.types.Operator,ExportHelper):
    bl_idname='cm.save_reference_library'
    bl_label='Save Assigned Templates as Library'
    bl_description='Save private packed copies of assigned licensed head/body references; keep original geometry, materials and current project'
    filename_ext='.blend'
    filter_glob:bpy.props.StringProperty(default='*.blend',options={'HIDDEN'})

    def invoke(self,context,event):
        try:roles(context)
        except ValueError as error:self.report({'ERROR'},str(error));return {'CANCELLED'}
        if not self.filepath:
            folder=Path(bpy.utils.user_resource('DATAFILES'))/'create_metahuman/references'
            path=folder/'character.blend';index=2
            while path.exists():path=folder/('character_'+str(index)+'.blend');index+=1
            self.filepath=str(path)
        return ExportHelper.invoke(self,context,event)

    def execute(self,context):
        try:
            report=save(context,bpy.path.abspath(self.filepath))
            context.scene.cm_settings.status='Reference library saved with '+str(report['packed_images'])+' packed images. Guided import is ready.'
        except (ValueError,RuntimeError,OSError,TypeError) as error:
            self.report({'ERROR'},str(error));return {'CANCELLED'}
        return {'FINISHED'}


class CM_OT_select_reference_library(bpy.types.Operator,ImportHelper):
    bl_idname='cm.select_reference_library'
    bl_label='Choose Reference Library'
    bl_description='Use a local library prepared from licensed head/body templates; remember it independently of optional models'
    filename_ext='.blend'
    filter_glob:bpy.props.StringProperty(default='*.blend',options={'HIDDEN'})

    def execute(self,context):
        try:
            report=select(context,bpy.path.abspath(self.filepath))
            missing=report['missing']
            context.scene.cm_settings.status=('Reference library ready: '+', '.join(report['roles'])+'.'+
                (' Missing: '+', '.join(missing)+'. These workflows need matching templates.' if missing else ' Head, body, eyes and teeth are available.'))
        except (ValueError,RuntimeError,OSError,TypeError) as error:
            self.report({'ERROR'},str(error));return {'CANCELLED'}
        return {'FINISHED'}


def draw(layout,context):
    f=context.scene.cm_features
    missing=bool(f.reference_library) and not setup_paths.valid('reference_library',f.reference_library)
    header,box=layout.panel('cm_reference_setup',default_closed=bool(f.reference_library) and not missing)
    header.label(text='Reference Library',icon='FILE_FOLDER')
    if missing:header.label(text='File missing',icon='ERROR')
    if not box:return
    box.prop(f,'reference_library',text='Library')
    box.operator('cm.select_reference_library',icon='FILEBROWSER')
    if missing:box.label(text='Locate the library again, or save assigned templates.',icon='INFO')
    box.label(text='Or import your licensed neutral LOD0 templates:')
    box.prop(context.scene.cm_settings,'template',text='Head Template');box.prop(f,'body_template',text='Body Template')
    for role,label in [('eye_left','Left Eye'),('eye_right','Right Eye'),('teeth','Teeth')]:box.prop(f,role,text=label)
    box.prop(f,'front_axis',text='Templates Face')
    row=box.row();row.enabled=bool(context.scene.cm_settings.template and f.body_template) and context.mode=='OBJECT'
    row.operator('cm.save_reference_library',icon='FILE_TICK')
    box.label(text='Packed copies stay local. Original UVs are required.')
    box.label(text='Manual wrapping needs no inference-model download.')


CLASSES=(CM_OT_save_reference_library,CM_OT_select_reference_library)


def register():
    for cls in CLASSES:bpy.utils.register_class(cls)


def unregister():
    for cls in reversed(CLASSES):bpy.utils.unregister_class(cls)
