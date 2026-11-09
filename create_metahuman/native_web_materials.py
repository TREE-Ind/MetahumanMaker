"""Portable resources from a captured UE actor; unknown shaders require PBR bakes."""
import copy
import hashlib
import json
import math
from pathlib import Path
import runpy
import shutil
import struct
from tempfile import TemporaryDirectory

GENERATOR='CreateMetaHuman.NativeWebMaterials'
COLOR_MAPS={'basecolor','iris','sclera','emissive'}
MAPS={'basecolor','normal','roughness','metallic','specular','packed','scatter','cavity',
      'micro_normal','micro_mask','iris','sclera','iris_normal','sclera_normal','opacity','emissive'}

def digest(path):
    with path.open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()

def checked(root,name,sha):
    path=(root/name).resolve()
    if not path.is_relative_to(root) or not path.is_file():raise ValueError('Missing or out-of-capture material resource: '+str(name))
    if not isinstance(sha,str) or len(sha)!=64 or digest(path)!=sha:raise ValueError('Captured material resource changed: '+str(name))
    return path

def family(record):
    chain=' '.join(record.get('inheritance',[])).lower();keys=record.get('textures',{})
    if 'm_hide' in chain:return 'HIDDEN'
    if 'Iris' in keys:return 'EYE'
    if 'teeth' in chain:return 'TEETH'
    if 'eyelashes' in chain:return 'LASH'
    if 'occlusion' in chain:return 'EYE_SHELL'
    if 'lacrimal' in chain:return 'TEARLINE'
    if 'skin_unified' in chain:return 'SKIN'
    return None

def bindings(kind,record):
    baked=record.get('switches',{}).get('Use Baked Material') is True
    if kind=='SKIN':
        primary={'basecolor':'Basecolor Baked','normal':'Normal Baked','packed':'SRMF Baked','scatter':'Scatter Baked'} if baked else {'basecolor':'Basecolor','normal':'Normal','roughness':'Roughness'}
        return {**primary,'cavity':'Cavity','micro_normal':'Micro Skin Details Normal','micro_mask':'Micro Skin Details Mask'}
    if kind=='TEETH':return {'basecolor':'Basecolor Baked','normal':'Normal Baked','packed':'SRM Baked','specular':'SRM Baked'} if baked else {'basecolor':'Basecolor','normal':'Normal'}
    if kind=='EYE':return {'iris':'Iris','sclera':'Sclera Basecolor','iris_normal':'Iris Normal','sclera_normal':'Sclera Normal'}
    if kind=='LASH':return {'opacity':'Texture'}
    return {}

def plan(source,source_sha,geometry,appearance=None):
    """Resolve families and exact texture requirements before pixel extraction."""
    if source.get('generator')!='CreateMetaHuman.NativeWebSource' or source.get('status')!='source_extracted':raise ValueError('Complete the native source capture first')
    if geometry.get('generator')!='CreateMetaHuman.NativeWebGeometry' or geometry.get('source_sha256')!=source_sha:raise ValueError('Geometry belongs to another source capture')
    if appearance is not None and (appearance.get('generator')!='CreateMetaHuman.NativeWebAppearance' or appearance.get('status')!='complete' or appearance.get('source_manifest_sha256')!=source_sha):raise ValueError('Native appearance bakes belong to another or incomplete capture')
    overrides={}
    for row in (appearance or {}).get('materials',[]):
        key=(row['component'],row['slot'])
        if key in overrides:raise ValueError('Duplicate native appearance material')
        if set(row.get('maps',{}))-MAPS:raise ValueError('Unknown native appearance map role')
        overrides[key]=row
    source_parts={p['component']:p for p in source['skeletal_meshes']};recipes={};used=set();required=set()
    for part in geometry['parts']:
        component=part['component']
        if component not in source_parts:raise ValueError('Converted component is absent from the capture')
        native=source_parts[component];alias='Face' if component==geometry.get('face_component') else component
        for slot in sorted(set(native['lod_sections'][0]['materials'])):
            record=native['materials'][slot];override=overrides.get((component,slot),{});used.add((component,slot))
            if not record:raise ValueError('Capture has no material recipe for '+component+' slot '+str(slot))
            kind=family(record)
            if kind is None:
                if override.get('kind')!='PBR' or override.get('surface')!='OPAQUE' or not {'basecolor','roughness','metallic'}<=set(override.get('maps',{})):raise ValueError('Material needs verified opaque PBR bakes or a supported MetaHuman shader: '+component+' slot '+str(slot)+' '+record['asset'])
                kind='PBR'
            allowed=({'basecolor','normal','roughness','metallic','specular','packed','scatter','cavity','micro_normal','micro_mask','opacity'} if kind=='SKIN' else
                     {'basecolor','normal','roughness','metallic','packed','specular','cavity','emissive'} if kind in {'TEETH','PBR'} else
                     {'iris','sclera','iris_normal','sclera_normal'} if kind=='EYE' else {'opacity'} if kind=='LASH' else set())
            if set(override.get('maps',{}))-allowed:raise ValueError('Bake map is not supported by this material family: '+component+' slot '+str(slot))
            mapping={}
            for role,parameter in bindings(kind,record).items():
                if role=='packed' and 'roughness' in override.get('maps',{}):continue
                if role in {'micro_normal','micro_mask'} and 'normal' in override.get('maps',{}) and not override['maps']['normal'].get('micro_details_separate'):continue
                asset=record.get('textures',{}).get(parameter)
                if asset and role not in override.get('maps',{}):mapping[role]={'asset':asset['asset']};required.add(asset['asset'])
            for role,row in override.get('maps',{}).items():mapping[role]={'bake':copy.deepcopy(row)}
            needed={'basecolor'} if kind in {'SKIN','TEETH','PBR'} else {'iris','sclera','iris_normal','sclera_normal'} if kind=='EYE' else {'opacity'} if kind=='LASH' else set()
            if not needed<=mapping.keys():raise ValueError('Native material is missing required inputs: '+component+' slot '+str(slot))
            name='CMNative_'+alias+'_'+str(slot)
            if name in recipes:raise ValueError('Native material aliases collide: '+name)
            recipes[name]={'kind':kind,'source':record['asset'],'scalars':copy.deepcopy(record.get('scalars',{})),
                'vectors':copy.deepcopy(record.get('vectors',{})),'switches':copy.deepcopy(record.get('switches',{})),
                'maps':mapping,'native_record':copy.deepcopy(record),'unsupported':['complete UE shading graph and driven appearance parity']}
            if kind=='PBR':recipes[name]['unsupported'].append('native WPO, time effects, transparency and non-PBR lighting models')
            if override.get('surface')=='MASKED':
                cutoff=override.get('alpha_test')
                if kind!='SKIN' or 'opacity' not in mapping or type(cutoff) not in (int,float) or not math.isfinite(cutoff) or not 0<=cutoff<=1:raise ValueError('Masked skin needs a baked opacity mask and valid cutoff')
                recipes[name].update(surface='MASKED',alpha_test=cutoff)
    if set(overrides)-used:raise ValueError('Appearance bakes reference an absent rendered material')
    return {'materials':recipes,'required_texture_assets':sorted(required)}

def prepare(source_root,geometry_manifest,destination,*,appearance_manifest=None,allow_legacy_textures=False,texture_limit=8192):
    """Publish a new portable material directory, leaving captured images intact."""
    if type(allow_legacy_textures) is not bool or type(texture_limit) is not int or texture_limit<1:raise ValueError('Invalid material preparation options')
    root=Path(source_root).resolve();destination=Path(destination).absolute()
    if destination.exists() or destination.resolve()!=destination or destination.is_relative_to(root):raise ValueError('Choose a new separate material destination without linked folders')
    source_path=root/'source-manifest.json';source_bytes=source_path.read_bytes();source_sha=hashlib.sha256(source_bytes).hexdigest();source=json.loads(source_bytes)
    geometry_path=Path(geometry_manifest);geometry_bytes=geometry_path.read_bytes();geometry=json.loads(geometry_bytes)
    appearance_path=Path(appearance_manifest) if appearance_manifest else None
    appearance_bytes=appearance_path.read_bytes() if appearance_path else None
    appearance=json.loads(appearance_bytes) if appearance_bytes is not None else None
    planned=plan(source,source_sha,geometry,appearance)
    texture_path=root/'textures/texture-manifest.json';texture_bytes=texture_path.read_bytes();textures=json.loads(texture_bytes)
    if textures.get('generator')!='CreateMetaHuman.NativeWebTextures' or textures.get('status') not in {'pixels_extracted','pixels_extracted_with_unresolved_resources'}:raise ValueError('Complete native texture extraction first')
    legacy='source_manifest_sha256' not in textures
    if legacy and not allow_legacy_textures:raise ValueError('Recapture texture provenance or explicitly opt in to legacy texture metadata')
    if not legacy and textures['source_manifest_sha256']!=source_sha:raise ValueError('Texture pixels belong to another source capture')
    if textures.get('source_actor_class')!=source.get('source_actor_class'):raise ValueError('Texture actor identity does not match the capture')
    dependencies={}
    for name,recipe in planned['materials'].items():
        for value in recipe['scalars'].values():
            if type(value) not in (int,float) or not math.isfinite(value):raise ValueError('Invalid captured scalar in '+name)
        for role,item in list(recipe['maps'].items()):
            if 'asset' in item:
                native=textures['textures'].get(item['asset'])
                if not native:raise ValueError('Required texture pixels are missing: '+item['asset'])
                relative='textures/'+native['file'];sha=native['sha256'];srgb=native['srgb'];baked=False;expected=[native['width'],native['height']]
            else:
                native=item['bake'];relative=native['file'];sha=native['sha256'];srgb=native.get('srgb',role in COLOR_MAPS and not native.get('linear_color',False));baked=True;expected=native.get('size')
            path=checked(root,relative,sha)
            if type(srgb) is not bool:raise ValueError('Native texture color space must be explicit boolean metadata')
            with path.open('rb') as stream:header=stream.read(24)
            if header[:8]!=b'\x89PNG\r\n\x1a\n' or header[12:16]!=b'IHDR':raise ValueError('Native material resource is not PNG: '+relative)
            size=list(struct.unpack('>II',header[16:24]))
            if min(size)<1 or max(size)>texture_limit or expected is not None and list(expected)!=size:raise ValueError('Native texture dimensions are invalid or exceed the limit: '+relative)
            filename='textures/'+sha+'.png';dependencies[filename]=(path,sha)
            mapped={'file':filename,'size':size,'srgb':bool(srgb),'sha256':sha,'native_artist_bake':baked}
            if baked and role=='normal' and native.get('micro_details_separate'):mapped['micro_details_separate']=True
            if baked and native.get('virtual_inputs_resolved'):mapped['virtual_inputs_resolved']=True
            if 'asset' in item:mapped['source']=item['asset']
            if role=='emissive':
                scale=native.get('emissive_scale',1)
                if type(scale) not in (int,float) or not math.isfinite(scale) or scale<0:raise ValueError('Native emissive scale is invalid')
                mapped['emissive_scale']=scale
            recipe['maps'][role]=mapped
    budget=runpy.run_path(str(Path(__file__).with_name('resource_budget.py')))
    destination.parent.mkdir(parents=True,exist_ok=True)
    budget['require_disk']('portable native material preparation',destination.parent,max(32*1024**2,sum(p.stat().st_size for p,_ in dependencies.values())+16*1024**2))
    result={'schema':1,'generator':GENERATOR,'source_manifest_sha256':source_sha,'geometry_manifest_sha256':hashlib.sha256(geometry_bytes).hexdigest(),
        'legacy_texture_provenance':legacy,'materials':planned['materials'],'texture_limit':texture_limit,'full_shader_parity':False,
        'baked_materials':[name for name,r in planned['materials'].items() if any(m['native_artist_bake'] for m in r['maps'].values())]}
    result['source_metadata']={k:copy.deepcopy(v) for k,v in source.items() if k not in {'skeletal_meshes','grooms','animations'}}
    if appearance is not None:result['native_appearance_record']=copy.deepcopy(appearance)
    if appearance_bytes is not None:result['appearance_manifest_sha256']=hashlib.sha256(appearance_bytes).hexdigest()
    with TemporaryDirectory(prefix='.native-materials-',dir=destination.parent) as temporary:
        stage=Path(temporary);(stage/'textures').mkdir()
        for filename,(path,sha) in dependencies.items():
            target=stage/filename;shutil.copy2(path,target)
            if digest(target)!=sha:raise ValueError('Native texture changed during material publication')
        motion=runpy.run_path(str(Path(__file__).with_name('native_skin_motion.py')))['prepare'](root,stage,source_sha)
        if motion:
            name='CMNative_'+('Face' if motion['component']==geometry.get('face_component') else motion['component'])+'_'+str(motion['slot'])
            if name not in result['materials'] or result['materials'][name]['kind']!='SKIN':raise ValueError('Animated skin does not match the exported face material')
            result['materials'][name]['animated_skin']=motion
            result['sampled_skin_response']=True
        payload=json.dumps(result,indent=2,allow_nan=False)
        (stage/'native-materials.json').write_text(payload,encoding='utf-8')
        runpy.run_path(str(Path(__file__).with_name('web_native_materials.py')))['write'](stage/'native-materials.js')
        runpy.run_path(str(Path(__file__).with_name('web_skin_motion.py')))['write'](stage/'native-skin-motion.js')
        if source_path.read_bytes()!=source_bytes or texture_path.read_bytes()!=texture_bytes or geometry_path.read_bytes()!=geometry_bytes:raise ValueError('Capture or geometry changed during material preparation')
        if appearance_path and appearance_path.read_bytes()!=appearance_bytes:raise ValueError('Appearance bakes changed during material preparation')
        stage.replace(destination)
    return result
