"""Track the fit used for source-projected color and roughness without owning paint."""
import hashlib,json
from pathlib import Path
import bpy
from . import map_provenance,skin_projection

PROPERTY='cm_source_skin_receipt'


def part(scene,role):
    s,f=scene.cm_settings,scene.cm_features
    if role not in {'HEAD','BODY'}:raise ValueError('Choose a head or body skin part')
    obj=f.body_result if s.part=='FULL_BODY' and role=='BODY' else s.result
    prefix=s.character_name+('_Body' if s.part=='FULL_BODY' and role=='BODY' else '')
    if Path(prefix).name!=prefix:raise ValueError('Use a character name without directory separators')
    return obj,prefix


def digest(path):
    with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()


def inputs(scene,obj,*,native=False,cache=None):
    cache={} if cache is None else cache;s,f=scene.cm_settings,scene.cm_features
    with bpy.context.temp_override(scene=scene,view_layer=scene.view_layers[0]):
        value=dict(uv=skin_projection.uv_layout_signature(obj),source_shader=map_provenance.shader_inputs(s.source,cache,surface=True,fast_images=True) if s.source else None,native=native)
        if native:value['tone']=f.generated_skin_tone
        else:value.update(target=map_provenance.mesh_digest(bpy.context,obj,cache),source=map_provenance.mesh_digest(bpy.context,s.source,cache),distance=s.bake_distance)
    return hashlib.sha256(json.dumps(value,sort_keys=True).encode()).hexdigest()


def record(scene,items):
    folder=map_provenance.folder(scene);cache={}
    for item in items:
        obj,prefix=part(scene,item['role']);old=json.loads(obj.get(PROPERTY,'{}'))
        channels=dict(old.get('channels',{})) if old.get('prefix')==prefix and old.get('schema')==1 else {}
        for kind in ('ALBEDO','ROUGHNESS'):
            if not item.get('albedo' if kind=='ALBEDO' else 'roughness'):continue
            path=folder/(prefix+'_'+kind+'.png')
            if not path.is_file():raise ValueError('Projected skin output is missing: '+path.name)
            native=kind=='ALBEDO' and obj.get('cm_albedo_source')=='LOCAL_NATIVE_UV'
            channels[kind]=dict(ownership='SOURCE',sha256=digest(path),native=native,inputs=inputs(scene,obj,native=native,cache=cache))
        obj[PROPERTY]=json.dumps(dict(schema=1,prefix=prefix,role=item['role'],channels=channels))


def retain(scene,role,kinds=('ALBEDO','ROUGHNESS')):
    obj,prefix=part(scene,role)
    if not obj:raise ValueError('Choose a fitted skin part first')
    previous=json.loads(obj.get(PROPERTY,'{}'));channels=dict(previous.get('channels',{})) if previous.get('schema')==1 and previous.get('prefix')==prefix else {}
    found=False
    for kind in kinds:
        if kind not in {'ALBEDO','ROUGHNESS'}:raise ValueError('Unsupported authored skin channel')
        path=map_provenance.folder(scene)/(prefix+'_'+kind+'.png')
        if path.is_file():channels[kind]=dict(ownership='AUTHORED',sha256=digest(path));found=True
    if not found:raise ValueError('Save the selected skin channels into the project first')
    obj[PROPERTY]=json.dumps(dict(schema=1,prefix=prefix,role=role,channels=channels))


def current_projection(scene,role,kind):
    """Prove that saved pixels still belong to the recorded source transfer."""
    obj,prefix=part(scene,role)
    if not obj:return False
    record=json.loads(obj.get(PROPERTY,'{}'))
    entry=record.get('channels',{}).get(kind,{})
    path=map_provenance.folder(scene)/(prefix+'_'+kind+'.png')
    return bool(record.get('schema')==1 and record.get('prefix')==prefix and entry.get('ownership')=='SOURCE'
        and path.is_file() and entry.get('sha256')==digest(path)
        and entry.get('inputs')==inputs(scene,obj,native=entry.get('native',False)))


def problems(scene, *, kinds=None,roles=None):
    s=scene.cm_settings;result=[];cache={};folder=map_provenance.folder(scene)
    for role in (('HEAD','BODY') if s.part=='FULL_BODY' else (s.part,)):
        if roles is not None and role not in roles:continue
        obj,prefix=part(scene,role)
        if not obj or not obj.get(PROPERTY):continue  # Legacy maps are untracked, not certified fresh.
        receipt=json.loads(obj[PROPERTY])
        if receipt.get('schema')!=1 or receipt.get('prefix')!=prefix:
            result.append('Review the source skin record for '+role.lower()+'. Refresh Source Skin or keep the current skin as authored.');continue
        for kind,entry in receipt.get('channels',{}).items():
            if kind not in {'ALBEDO','ROUGHNESS'}:raise ValueError('Invalid source skin channel in projection record')
            if kinds is not None and kind not in kinds:continue
            path=folder/(prefix+'_'+kind+'.png')
            if not path.is_file():result.append('Missing '+role.lower()+' '+kind.lower()+' map.');continue
            if entry.get('ownership')=='AUTHORED':continue
            # A changed file may be accepted paint or external artwork. Do not
            # reinterpret it as an untouched projection or replace its pixels.
            if digest(path)!=entry.get('sha256'):continue
            if inputs(scene,obj,native=entry.get('native',False),cache=cache)!=entry.get('inputs'):
                result.append(role.title()+' '+kind.lower()+' projection predates the current fit/source. Refresh Source Skin or keep the current skin as authored.')
    return result


class CM_OT_retain_authored_skin(bpy.types.Operator):
    bl_idname='cm.retain_authored_skin'
    bl_label='Keep Current Skin as Authored'
    bl_description='Retain the selected part color/roughness through source-fit changes; no image pixels are changed'
    bl_options={'REGISTER','UNDO'}

    def execute(self,context):
        from . import bake_jobs
        s,f=context.scene.cm_settings,context.scene.cm_features
        try:
            if bake_jobs.running():raise ValueError('Finish the skin bake before retaining authored maps')
            role=f.texture_part if s.part=='FULL_BODY' else s.part;obj,prefix=part(context.scene,role)
            if not obj:raise ValueError('Choose a fitted skin part first')
            channels={kind:dict(ownership='AUTHORED',sha256=digest(path)) for kind in ('ALBEDO','ROUGHNESS') if (path:=map_provenance.folder(context.scene)/(prefix+'_'+kind+'.png')).is_file()}
            if not channels:raise ValueError('Save this part skin into the project first')
            obj[PROPERTY]=json.dumps(dict(schema=1,prefix=prefix,role=role,channels=channels));s.status='Current skin retained as authored. Source-fit changes will not request replacing it.';return {'FINISHED'}
        except (ValueError,OSError) as exc:s.status=str(exc);self.report({'ERROR'},str(exc));return {'CANCELLED'}


CLASSES=(CM_OT_retain_authored_skin,)
