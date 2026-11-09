"""Read a complete native face bank without creating persistent Unreal assets."""
import hashlib
import json
from pathlib import Path
import runpy
import time


def api(name):return runpy.run_path(str(Path(__file__).with_name(name+'.py')))


def digest(path):
    with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()


def supported(mesh):
    import unreal
    library=getattr(unreal,'CreateMHExpressionLibrary',None)
    if library is None or not hasattr(library,'supports_isolated_pose_evaluation') or not library.supports_isolated_pose_evaluation():
        raise ValueError('Install MetaHuman Maker Bridge 0.15 or newer to capture native facial samples without changing live wrinkle materials, or disable that capture option')
    names={data.get_class().get_name() for data in mesh.get_editor_property('asset_user_data') if data}
    if not names.intersection({'DNAAsset','DNAAssetUserData'}):
        raise ValueError('The selected face has no native DNA binding; disable native facial samples for an approximate export')


def capture(mesh,directory,progress=None,source_owner=None,source_component=None):
    import unreal
    supported(mesh)
    root=Path(directory).resolve()
    if root.exists():raise ValueError('Choose a new native face-bank folder; previous samples are retained')
    package=mesh.get_path_name().split('.')[0]
    fingerprint_scope='SKELETAL_MESH_PACKAGE'
    if not package.startswith('/Game/'):
        if source_owner is None or not isinstance(source_owner,unreal.MetaHumanCharacter):raise ValueError('A transient face needs its linked saved Creator character; use linked-character capture or a saved assembled face')
        package=source_owner.get_path_name().split('.')[0];fingerprint_scope='CREATOR_OWNER_PACKAGE'
        if not package.startswith('/Game/'):raise ValueError('Save the linked Creator character before capturing its transient face')
    if package in {p.get_path_name() for p in unreal.EditorLoadingAndSavingUtils.get_dirty_content_packages()}:
        raise ValueError('Save the native face mesh before capturing facial samples')
    source=Path(unreal.Paths.project_content_dir()).resolve()/(package.removeprefix('/Game/')+'.uasset')
    if not source.is_file():raise ValueError('The native face or linked Creator package is not saved')
    budget=api('resource_budget');budget['require']('native facial sampling',1024**3);budget['require_disk']('native facial sampling',root.parent,128*1024**2)
    root.mkdir(parents=True)
    report={'schema':1,'generator':'CreateMetaHuman.NativeFaceBank','status':'running','source_mesh':mesh.get_path_name(),
            'source_sha256':digest(source),'source_fingerprint_scope':fingerprint_scope,'cloud_requests':False,'poses':[],'groups':[],'files':{}}
    if fingerprint_scope=='CREATOR_OWNER_PACKAGE':report['source_owner']=source_owner.get_path_name()
    receipt=root/'bank.json';shared={};started=time.monotonic();definitions=api('native_face_recipes')['recipes']();total=1+4*len(definitions)
    def write():
        pending=receipt.with_suffix('.pending');pending.write_text(json.dumps(report,indent=2),encoding='utf-8');pending.replace(receipt)
    write()
    def evaluate(name,controls):
        path=root/(name+'.json')
        if progress:progress('face_samples',len(report['poses']),total)
        if not unreal.CreateMHExpressionLibrary.export_engine_control_pose(mesh,json.dumps(controls,allow_nan=False),str(path),source_component):
            raise RuntimeError('Native facial evaluation failed: '+name+'; inspect the Unreal log')
        data=json.loads(path.read_bytes());actual={key.lower():value for key,value in data['raw_curves'].items()}
        if data.get('material_instances_isolated') is not True:raise ValueError('Native facial sampling did not isolate live materials; update the Unreal bridge')
        if data.get('dna_mode')!='ASSEMBLED_EMBEDDED' or data.get('source_mesh')!=report['source_mesh']:
            raise ValueError('Native sample identity does not match the captured face')
        if any(abs(actual.get(('CTRL_expressions_'+key).lower(),float('inf'))-value)>1e-6 for key,value in controls.items()):
            raise ValueError('Native facial controls were not evaluated: '+name)
        if report.get('dna_binding') and report['dna_binding']!=data['dna_binding']:raise ValueError('The face DNA binding changed during sampling')
        report.update(dna_binding=data['dna_binding'],vertices=data['vertex_count'],bones=len(data['bones']))
        # Reference positions and UVs are identical for every sample. Keep one
        # verified copy rather than repeating ~0.7 MB seventy-seven times.
        for key in ('neutral_file','uv_file'):
            filename=data[key];file=root/filename
            if key not in shared:shared[key]=(filename,digest(file))
            else:
                retained,sha=shared[key]
                if digest(file)!=sha:raise ValueError('Native reference geometry or UVs changed during sampling')
                if file.name!=retained:
                    data[key]=retained
                    if file.parent!=root:raise ValueError('Unexpected native sample output path')
                    file.unlink()
        path.write_text(json.dumps(data,separators=(',',':')),encoding='utf-8')
        for filename in (path.name,data['vertices_file'],data['neutral_file'],data['uv_file']):
            file=root/filename
            if file.resolve().parent!=root or not file.is_file():raise ValueError('Native sample data leaves its capture folder')
            report['files'][filename]={'sha256':digest(file),'bytes':file.stat().st_size}
        report['poses'].append({'name':name,'file':str(path),'controls':controls,'active_morphs':sum(abs(value)>1e-6 for value in data['morphs'].values())})
        write();unreal.SystemLibrary.collect_garbage();return data
    try:
        neutral=evaluate('Neutral',{});available=set(neutral['raw_controls'])
        for name,controls in definitions:
            missing={'CTRL_expressions.'+key for key in controls}-available
            if missing:raise ValueError('This native rig lacks the controls required for '+name+': '+', '.join(sorted(missing)))
        for name,controls in definitions:
            group={'name':name,'samples':[]}
            for level,strength in enumerate((.25,.5,.75,1.),1):
                pose=name+'_'+str(level);evaluate(pose,{key:value*strength for key,value in controls.items()});group['samples'].append({'strength':strength,'pose':pose})
            report['groups'].append(group);write()
        if digest(source)!=report['source_sha256']:raise ValueError('The native source mesh changed during sampling')
        report.update(status='complete',source_mesh_unchanged=True,elapsed_seconds=time.monotonic()-started)
        if progress:progress('face_samples',total,total)
    except Exception as exc:report.update(status='failed',error=str(exc));raise
    finally:write()
    return report
