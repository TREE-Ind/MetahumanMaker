"""Publish prepared native stages as a self-contained, locally served web bundle."""
import hashlib
import json
import posixpath
import re
import runpy
import shutil
import struct
import zipfile
from pathlib import Path, PurePosixPath
from tempfile import TemporaryDirectory

GENERATOR='CreateMetaHuman.NativeWebBundle'
THREE_VERSION='0.186.1'
LIMITATIONS=[
    'Browser skin, eye and hair shaders approximate the native Unreal graphs.',
    'Direct corrective controls do not execute native RigLogic expressions.',
    'Strand ribbons use surface-frame binding; native RBF, simulation and groom LODs are not reproduced.',
    'Driven wrinkle and blood-flow animation is not implemented in this viewer.',
    'Actor animation, live poses and arbitrary material graphs are not converted.',
    'Geometry GLB alone omits runtime textures, full-weight skinning and strand rendering.',
]


def digest(path):
    with path.open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()


def relative(name):
    if not isinstance(name,str) or not name or '\\' in name or ':' in name or '?' in name or '#' in name or any(p in ('','.', '..') for p in name.split('/')) or PurePosixPath(name).is_absolute():raise ValueError('Invalid bundle resource path: '+str(name))
    return name


def resource(root,name,record):
    name=relative(name);path=(root/name).resolve()
    if not path.is_relative_to(root) or not path.is_file() or digest(path)!=record.get('sha256') or 'bytes' in record and path.stat().st_size!=record['bytes']:raise ValueError('Bundle input is missing or changed: '+name)
    return path


def module(name):
    return runpy.run_path(str(Path(__file__).with_name(name+'.py')))


def document(path,generator):
    raw=path.read_bytes();value=json.loads(raw)
    if value.get('generator')!=generator:raise ValueError('Unexpected native web stage: '+str(path))
    return value,hashlib.sha256(raw).hexdigest()


def vendor_resources(root):
    manifest=json.loads((root/'manifest.json').read_bytes())
    if manifest.get('package')!='three' or manifest.get('version')!=THREE_VERSION:raise ValueError('Use the pinned Three.js '+THREE_VERSION+' vendor cache')
    entries=manifest['files'];required={'LICENSE','build/three.module.js','examples/jsm/loaders/GLTFLoader.js','examples/jsm/controls/OrbitControls.js','examples/jsm/environments/RoomEnvironment.js','examples/jsm/loaders/EXRLoader.js','examples/jsm/lights/RectAreaLightUniformsLib.js'}
    if not required.issubset(entries):raise ValueError('Viewer vendor cache is incomplete')
    result={'vendor/manifest.json':(root/'manifest.json',digest(root/'manifest.json'))}
    for name,record in entries.items():
        if record.get('source')!='https://cdn.jsdelivr.net/npm/three@'+THREE_VERSION+'/'+name:raise ValueError('Unexpected viewer dependency origin')
        path=resource((root/'three').resolve(),name,record);result['vendor/three/'+name]=(path,record['sha256'])
        if name.endswith('.js'):
            code=re.sub(r'/\*.*?\*/','',path.read_text(encoding='utf-8'),flags=re.S)
            dependencies=re.findall(r'''^\s*(?:import|export)\s+(?:\{[^}]*\}|\*[^\n]*|\w+)\s+from\s*['"]([^'"]+)['"]''',code,flags=re.M)
            for dependency in dependencies:
                if dependency=='three':target='build/three.module.js'
                elif dependency=='three/tsl':target='build/three.tsl.js'
                elif dependency.startswith('three/addons/'):target='examples/jsm/'+dependency.removeprefix('three/addons/')
                elif dependency.startswith('.'):target=posixpath.normpath(posixpath.join(posixpath.dirname(name),dependency))
                else:raise ValueError('Unsupported viewer module dependency: '+dependency)
                if target not in entries:raise ValueError('Missing viewer module dependency: '+target)
    return result


def glb_materials(path):
    with path.open('rb') as stream:
        header=stream.read(20)
        if len(header)!=20 or struct.unpack('<III',header[:12])!=(0x46546c67,2,path.stat().st_size):raise ValueError('Geometry is not a complete version-2 GLB')
        size,kind=struct.unpack('<II',header[12:])
        if kind!=0x4e4f534a or size>path.stat().st_size-20:raise ValueError('Invalid GLB document')
        value=json.loads(stream.read(size))
    if any(b.get('uri') for b in value.get('buffers',[])) or any(i.get('uri') for i in value.get('images',[])):raise ValueError('Native geometry must embed its resources')
    names=[m['name'] for m in value.get('materials',[])]
    if len(set(names))!=len(names):raise ValueError('Geometry has ambiguous material names')
    return set(names)


def publish(source_root,geometry_root,materials_root,grooms_root,vendor_root,environment_root,destination,*,name='Character',archive=True):
    """Publish current renderer code plus verified prepared resources, without Blender."""
    roots=[Path(p).resolve() for p in (source_root,geometry_root,materials_root,grooms_root,vendor_root,environment_root)]
    source_root,geometry_root,materials_root,grooms_root,vendor_root,environment_root=roots
    destination=Path(destination).absolute()
    if destination.exists() or destination.resolve()!=destination or any(destination.is_relative_to(r) or r.is_relative_to(destination) for r in roots):raise ValueError('Choose a new separate bundle destination without linked folders')
    if not isinstance(name,str) or not name.strip() or len(name)>160 or any(ord(c)<32 for c in name) or type(archive) is not bool:raise ValueError('Choose a character name and explicit archive option')
    source,source_sha=document(source_root/'source-manifest.json','CreateMetaHuman.NativeWebSource')
    geometry,geometry_sha=document(geometry_root/'geometry-manifest.json','CreateMetaHuman.NativeWebGeometry')
    materials,materials_sha=document(materials_root/'native-materials.json','CreateMetaHuman.NativeWebMaterials')
    grooms,grooms_sha=document(grooms_root/'grooms.json','CreateMetaHuman.NativeWebGrooms')
    if source.get('status')!='source_extracted' or geometry.get('source_sha256')!=source_sha:raise ValueError('Complete geometry conversion for this native capture first')
    for value in (materials,grooms):
        if value.get('source_manifest_sha256')!=source_sha or value.get('geometry_manifest_sha256')!=geometry_sha:raise ValueError('Prepared web stages belong to different captures or geometry')
    dependencies={}
    mesh=relative(geometry['mesh'])
    dependencies['character.glb']=(resource(geometry_root,mesh,geometry['files'][mesh]),geometry['files'][mesh]['sha256'])
    sampled=geometry.get('facial_mode')=='NATIVE_SAMPLED'
    if sampled:
        descriptor='native-expression-bank.json'
        if descriptor not in geometry['files']:raise ValueError('Sampled native geometry has no expression bank descriptor')
        dependencies[descriptor]=(resource(geometry_root,descriptor,geometry['files'][descriptor]),geometry['files'][descriptor]['sha256'])
    material_names=glb_materials(dependencies['character.glb'][0])
    if material_names!=set(materials['materials']):raise ValueError('Prepared materials do not cover the exported geometry exactly')
    for recipe in materials['materials'].values():
        motion=recipe.get('animated_skin')
        if motion:
            if not sampled:raise ValueError('Sampled skin response needs its native facial controller')
            bank=json.loads(dependencies['native-expression-bank.json'][0].read_bytes())
            if {s['pose'] for g in bank['groups'] for s in g['samples']}!={s['pose'] for s in motion['samples']}:raise ValueError('Skin response and facial sample groups differ')
        for item in recipe.get('animated_skin',{}).get('maps',{}).values():
            if item['file']!='textures/'+item['sha256']+'.png':raise ValueError('Animated skin atlas is not content-addressed')
            dependencies[relative(item['file'])]=(resource(materials_root,item['file'],item),item['sha256'])
        for item in recipe['maps'].values():
            if item:
                if item['file']!='textures/'+item['sha256']+'.png':raise ValueError('Prepared material resource is not content-addressed')
                dependencies[relative(item['file'])]=(resource(materials_root,item['file'],item),item['sha256'])
    for record in grooms['grooms']:
        filename=relative(record['file'])
        if '/' in filename or not filename.endswith('.strands.bin') or filename not in grooms.get('files',{}):raise ValueError('Reprepare grooms with output resource checksums before publication')
        if record['target_material'] not in material_names:raise ValueError('Prepared groom target is absent from geometry')
        dependencies[filename]=(resource(grooms_root,filename,grooms['files'][filename]),grooms['files'][filename]['sha256'])
    captured={g['component'] for g in source.get('grooms',[]) if g.get('asset')}
    if {g['component'] for g in grooms['grooms']}!=captured:raise ValueError('Prepared strands do not cover every assigned native groom')
    dependencies.update(vendor_resources(vendor_root))
    lighting=module('web_lighting')
    for filename in (*[n+'.exr' for n in lighting['ENVIRONMENTS']],'license.txt'):
        path=(environment_root/filename).resolve()
        if not path.is_relative_to(environment_root) or not path.is_file():raise ValueError('Missing local studio environment: '+filename)
        dependencies['lighting/'+filename]=(path,digest(path))
    inputs=[(source_root/'source-manifest.json',source_sha),(geometry_root/'geometry-manifest.json',geometry_sha),(materials_root/'native-materials.json',materials_sha),(grooms_root/'grooms.json',grooms_sha)]
    total=sum(p.stat().st_size for p,_ in dependencies.values())
    budget=module('resource_budget');budget['require_disk']('native web bundle publication',destination.parent,total*(2 if archive else 1)+64*1024**2)
    destination.parent.mkdir(parents=True,exist_ok=True)
    with TemporaryDirectory(prefix='.native-bundle-',dir=destination.parent) as temporary:
        stage=Path(temporary)
        for filename,(path,sha) in dependencies.items():
            target=stage/filename;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(path,target)
            if digest(target)!=sha:raise ValueError('Bundle input changed during publication: '+filename)
        for filename,root in (('geometry-manifest.json',geometry_root),('native-materials.json',materials_root),('grooms.json',grooms_root)):shutil.copy2(root/filename,stage/filename)
        for output,code in (('full-skin.js','web_full_skin'),('native-materials.js','web_native_materials'),('grooms.js','web_grooms'),('face-controls.js','web_controls')):module(code)['write'](stage/output)
        module('web_skin_motion')['write'](stage/'native-skin-motion.js')
        (stage/'lighting-studio.js').write_text(lighting['JAVASCRIPT'],encoding='utf-8')
        (stage/'lighting/manifest.json').write_text(json.dumps({'environments':list(lighting['ENVIRONMENTS']),'license':'CC0','creator':'Greg Zaal / Poly Haven','source':'Blender studio environment distribution'},indent=2),encoding='utf-8')
        (stage/'load-character.js').write_text("import {loadNativeCharacter} from './full-skin.js';\nimport {nativeMaterialFactory} from './native-materials.js';\nimport {loadGrooms} from './grooms.js';\nexport async function loadCharacter(scene){const configure=await nativeMaterialFactory(new URL('./native-materials.json',import.meta.url));const gltf=await loadNativeCharacter(scene,new URL('./character.glb',import.meta.url),configure);gltf.scene.userData.grooms=await loadGrooms(gltf.scene,new URL('./grooms.json',import.meta.url));return gltf.scene}\n",encoding='utf-8')
        if sampled:
            module('web_native_pose_bank')['write'](stage/'native-pose-bank.js')
            loader=(stage/'load-character.js').read_text(encoding='utf-8')
            loader="import {loadNativePoseBank} from './native-pose-bank.js';\n"+loader.replace('return gltf.scene}',"await loadNativePoseBank(gltf.scene,new URL('./native-expression-bank.json',import.meta.url));return gltf.scene}")
            (stage/'load-character.js').write_text(loader,encoding='utf-8')
        page=module('web_preview')['HTML'].replace('__MODEL__','character.glb').replace('https://cdn.jsdelivr.net/npm/three@'+THREE_VERSION+'/build/three.module.js','./vendor/three/build/three.module.js').replace('https://cdn.jsdelivr.net/npm/three@'+THREE_VERSION+'/examples/jsm/','./vendor/three/examples/jsm/')
        page=page.replace("import { loadCharacter }","import {createFaceControls} from './face-controls.js';\nimport { loadCharacter }")
        page=page.replace('const model=await loadCharacter(scene);','const model=await loadCharacter(scene);createFaceControls(model);')
        face=next((p['node'] for p in geometry['parts'] if p['component']==geometry.get('face_component','Face')),None)
        page=page.replace("model.getObjectByName('CM_Web_Head')",'model.getObjectByName('+json.dumps(face)+')' if face else 'null')
        page=page.replace('Download GLB','Geometry GLB')
        if archive:page=page.replace('>Geometry GLB</a>','>Geometry GLB</a><a href="Character-WebBundle.zip" download>Download web bundle</a>')
        page=page.replace('Static preview; animation and groom strands are separate.','Rigged preview with direct corrective controls. Native shader, RigLogic and groom equivalence remain unfinished.')
        if sampled:page=page.replace('Rigged preview with direct corrective controls.','Rigged preview with natively evaluated facial samples.')
        (stage/'preview.html').write_text(lighting['install_html'](page),encoding='utf-8')
        capabilities={'rigged':True,'all_skin_weights':True,'facial_mode':geometry.get('facial_mode'),'full_riglogic':False,'native_shaders':False,'native_groom_rbf':False,'native_groom_simulation':False,'driven_appearance':False,'local_lighting_environments':len(lighting['ENVIRONMENTS']),'source_groom_components':len(captured),'rendered_groom_groups':len(grooms['grooms'])}
        limitations=list(LIMITATIONS)
        capabilities['sampled_skin_response']=bool(sampled and materials.get('sampled_skin_response'))
        capabilities['driven_appearance']=capabilities['sampled_skin_response']
        if capabilities['sampled_skin_response']:
            limitations=[item for item in limitations if item!='Driven wrinkle and blood-flow animation is not implemented in this viewer.']
            limitations.append('Native color, tangent-normal and roughness responses are sampled at their recorded resolution; interpolation and combined expressions approximate nonlinear UE shading.')
        if sampled:
            capabilities['native_sampled_expressions']=True
            limitations[1]='Facial poses use character-specific native samples; combined poses approximate nonlinear RigLogic interactions.'
        manifest={'schema':1,'generator':GENERATOR,'name':name.strip(),'status':'published','browser_verified':False,'source_manifest_sha256':source_sha,'geometry_manifest_sha256':geometry_sha,'material_manifest_sha256':materials_sha,'groom_manifest_sha256':grooms_sha,'mesh':'character.glb','preview':'preview.html','capabilities':capabilities,'limitations':limitations,'provenance_files':['geometry-manifest.json','native-materials.json','grooms.json'],'geometry_provenance_only_files':[f for f in geometry['files'] if f not in {mesh,'native-expression-bank.json'}]}
        (stage/'manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
        (stage/'README.txt').write_text(name.strip()+' — native character web preview\n\nRun Start Preview.cmd or: python StartPreview.py\nOpen the localhost URL printed by the server. Keep all files together.\nAll viewer libraries and lighting environments are included; no CDN is required.\nServe over HTTP; opening preview.html directly from disk will not work.\nLighting studio: presets, HDR environments, key/fill/rim lights and saved setups.\n\nKnown limitations:\n'+'\n'.join('- '+v for v in limitations)+'\n\nOnly runtime dependencies are included. Geometry binding surfaces named in the\nprovenance manifest remain in the authoring stage, not this browser delivery.\nUse this character and its bundled assets subject to their applicable licenses.\n',encoding='utf-8')
        (stage/'StartPreview.py').write_text("from functools import partial\nfrom http.server import SimpleHTTPRequestHandler,ThreadingHTTPServer\nfrom pathlib import Path\nserver=ThreadingHTTPServer(('127.0.0.1',0),partial(SimpleHTTPRequestHandler,directory=str(Path(__file__).resolve().parent)))\nprint(f'Open http://127.0.0.1:{server.server_port}/preview.html',flush=True)\ntry: server.serve_forever()\nexcept KeyboardInterrupt: pass\nfinally: server.server_close()\n",encoding='utf-8')
        (stage/'Start Preview.cmd').write_bytes(b'@echo off\r\ncd /d "%~dp0"\r\nwhere py >nul 2>nul\r\nif errorlevel 1 (python StartPreview.py) else (py -3 StartPreview.py)\r\npause\r\n')
        checks={p.relative_to(stage).as_posix():{'bytes':p.stat().st_size,'sha256':digest(p)} for p in sorted(stage.rglob('*')) if p.is_file()}
        (stage/'bundle-checksums.json').write_text(json.dumps(checks,indent=2),encoding='utf-8')
        archive_record=None
        if archive:
            path=stage/'Character-WebBundle.zip'
            with zipfile.ZipFile(path,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=3) as zipped:
                for filename in (*checks,'bundle-checksums.json'):zipped.write(stage/filename,filename)
            with zipfile.ZipFile(path) as zipped:
                if zipped.testzip() is not None:raise ValueError('Web bundle archive verification failed')
            archive_record={'file':path.name,'bytes':path.stat().st_size,'sha256':digest(path)}
        for path,sha in (*inputs,*dependencies.values()):
            if digest(path)!=sha:raise ValueError('Input changed during web bundle publication')
        receipt={'generator':GENERATOR,'destination':str(destination),'runtime_files':len(checks)+1,'runtime_bytes':sum(c['bytes'] for c in checks.values()),'archive':archive_record,'browser_verified':False,'full_native_parity':False}
        (stage/'publication.json').write_text(json.dumps(receipt,indent=2),encoding='utf-8')
        stage.replace(destination)
    return receipt
