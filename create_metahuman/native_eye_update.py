"""Stage projected source-eye PBR appearance for an existing native character."""
import hashlib
import json
import shutil
from pathlib import Path
import bpy
from bpy.props import BoolProperty


def digest(path):
    with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()


def prepare(context,folder,*,restore=False):
    from . import character_project,conform_geometry,eye_color_jobs,eye_appearance,resource_budget
    s,f=context.scene.cm_settings,context.scene.cm_features
    character_project.preflight(context,s,f)
    if context.mode!='OBJECT' or f.expression_active or f.full_sculpt or f.body_editing or f.restore_editing or eye_color_jobs.running():
        raise ValueError('Finish the current edit and return to Neutral before updating native source eyes')
    if not f.unreal_character.startswith('/Game/') or not f.unreal_character_project:
        raise ValueError('Link this character to its native MetaHuman asset and Unreal project first')
    expected=Path(bpy.path.abspath(f.unreal_character_project)).resolve()
    if expected.suffix.lower()!='.uproject' or not expected.is_file():raise ValueError('Choose the linked Unreal project file')
    project=Path(bpy.path.abspath(s.project_dir)).resolve()
    records=[];geometry=None
    if not restore:
        if s.part=='BODY':raise ValueError('Choose a head or full character before applying source eyes')
        resource_budget.require('native source-eye preparation',resource_budget.GIB)
        geometry=conform_geometry.prepare(context,s,f)
        for role in ('eye_left','eye_right'):
            name=geometry['anatomy_assignment'].get(role)
            obj=bpy.data.objects.get(name) if name else None
            if obj is None or not obj.get(eye_color_jobs.PROPERTY):raise ValueError('Project Source Eye Appearance locally before applying it in Creator')
            state=json.loads(obj[eye_color_jobs.PROPERTY]);color=Path(state['image']).resolve();parent=color.parent
            if not parent.is_relative_to(project/'EyeColor'):raise ValueError('The projected eye color belongs to another character project')
            request=json.loads((parent/'request.json').read_text());report=json.loads((parent/'report.json').read_text())
            if report.get('status')!='complete' or obj.name not in request['eyes']:raise ValueError('The source-eye projection is incomplete')
            subjects=[bpy.data.objects.get(n) for n in request['eyes']]
            if set(subjects)!={f.eye_left,f.eye_right}:raise ValueError('Eye assignments changed; project source eye appearance again')
            current=json.loads(json.dumps(eye_color_jobs._stamp(context,subjects)));before=request['stamp'];index=request['eyes'].index(obj.name)
            if (any(current[key]!=before[key] for key in ('project','name','axis','assignments')) or current['objects'][0]!=before['objects'][0]
                    or any(a['mesh']!=b['mesh'] for a,b in zip(current['objects'][1:],before['objects'][1:]))):
                raise ValueError('Source or eye geometry changed since projection; refresh source eye appearance first')
            row=report['eyes'][index];finish=parent/row['finish_file']
            material=obj.data.materials[0]
            output=next((n for n in material.node_tree.nodes if n.type=='OUTPUT_MATERIAL' and n.is_active_output),None)
            shader=eye_appearance.linked(output.inputs['Surface']) if output else None
            if not shader or shader.type!='BSDF_PRINCIPLED':raise ValueError('The local source-eye PBR graph changed; refresh its projection')
            for socket,path in (('Base Color',color),('Roughness',finish),('Metallic',finish)):
                node=eye_appearance.linked(shader.inputs[socket])
                if socket!='Base Color':node=eye_appearance.linked(node.inputs[0]) if node and node.type=='SEPARATE_COLOR' else None
                if not node or node.type!='TEX_IMAGE' or not node.image or Path(bpy.path.abspath(node.image.filepath)).resolve()!=path:
                    raise ValueError('Local eye PBR bindings were edited; retain that artwork before refreshing source eyes')
                if node.image.is_dirty:raise ValueError('Save painted eye images before applying them in Unreal')
            for key in ('Transmission Weight','Coat Weight','Anisotropic IOR Level'):
                socket=shader.inputs.get(key)
                if socket and (socket.is_linked or abs(socket.default_value)>1e-8):raise ValueError('Native source-eye adapter supports opaque isotropic PBR; retain custom optical graphs separately')
            if shader.inputs['Normal'].is_linked:raise ValueError('Bake or retain authored eye normals before applying the source PBR adapter')
            for key in ('IOR','Specular IOR Level'):
                if shader.inputs[key].is_linked:raise ValueError('Native source-eye adapter needs constant IOR/specular controls')
            ior=float(shader.inputs['IOR'].default_value);level=float(shader.inputs['Specular IOR Level'].default_value)
            specular=2*level*((ior-1)/(ior+1))**2/.08
            if not 0<=specular<=1:raise ValueError('The local eye reflectance exceeds the native opaque PBR adapter range')
            for kind,path in (('ALBEDO',color),('FINISH',finish)):
                if not path.is_file() or path.parent!=parent:raise ValueError('Projected eye image is missing or outside its job')
                records.append(dict(role=role,kind=kind,source=str(path),snapshot='Maps/'+role+'_'+kind+'.png',
                                    sha256=digest(path),specular=specular))
    resource_budget.require_disk('native source-eye snapshot',project,sum(Path(row['source']).stat().st_size for row in records)+resource_budget.GIB)
    snapshot=folder/'scripts';snapshot.mkdir(parents=True,exist_ok=False)
    for row in records:
        target=snapshot/row['snapshot'];target.parent.mkdir(exist_ok=True);shutil.copy2(row['source'],target)
        if digest(target)!=row['sha256']:raise ValueError('Projected eye image changed during preparation')
    if geometry:
        (snapshot/'geometry.json').write_text(json.dumps(geometry),encoding='utf-8')
    for name in ('unreal_texture_update.py','unreal_native_eyes.py','unreal_native_albedo.py','unreal_revision_textures.py',
                 'unreal_import.py','unreal_roughness.py','resource_budget.py','process_ref.py'):
        shutil.copy2(Path(__file__).with_name(name),snapshot/name)
    built=json.loads(f.unreal_build_report or '{}')
    request=dict(schema_version=1,action='UPDATE_TEXTURES',geometry_mode='NATIVE_SOURCE_EYES',mode='RESTORE' if restore else 'APPLY',
        character=f.unreal_character,expected_project=str(expected),project_dir=str(project),character_name=s.character_name,maps=records,
        geometry_sha256=digest(snapshot/'geometry.json') if geometry else None,
        existing_assembly=built.get('character')==f.unreal_character and built.get('build',{}).get('state')=='COMPLETE')
    (folder/'request.json').write_text(json.dumps(request,indent=2),encoding='utf-8')
    return request


class CM_OT_source_eyes_creator(bpy.types.Operator):
    bl_idname='cm.source_eyes_creator';bl_label='Apply Projected Eyes to Creator'
    bl_description='Apply projected opaque eye color/finish as private native materials, or restore previous native eye overrides'
    restore:BoolProperty(default=False,options={'HIDDEN'})
    def execute(self,context):
        try:
            from . import unreal_jobs
            unreal_jobs.update_textures(context,native_eyes=True,restore_eyes=self.restore)
            return {'FINISHED'}
        except Exception as exc:self.report({'ERROR'},str(exc));return {'CANCELLED'}


CLASSES=(CM_OT_source_eyes_creator,)
