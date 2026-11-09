"""Bind natively evaluated whole-face samples to an exact FBX neutral surface.

Runs inside the isolated Blender web-geometry worker. Samples already contain
joint motion and correctives; facial joints remain at their reference transforms.
"""
import hashlib
import json
from pathlib import Path
import shutil
import runpy


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream,'sha256').hexdigest()


def snapshot(bank_path,destination):
    """Freeze the native bank and its data inside one owned export job."""
    bank_path=Path(bank_path).resolve();root=bank_path.parent;destination=Path(destination).resolve()
    if destination.exists() or destination.is_relative_to(root):raise ValueError('Choose a new separate face-bank snapshot folder')
    bank=json.loads(bank_path.read_bytes());original_sha=digest(bank_path)
    if bank.get('status')!='complete' or not bank.get('source_mesh_unchanged'):raise ValueError('Native face samples are incomplete')
    for relative,record in bank.get('files',{}).items():
        path=(root/relative).resolve()
        if not path.is_relative_to(root) or not path.is_file() or digest(path)!=record['sha256']:raise ValueError('Native face sample data changed after capture')
    resources={}
    for pose in bank['poses']:
        path=Path(pose['file']).resolve()
        if not path.is_relative_to(root) or not path.is_file():raise ValueError('Native pose leaves its source bank')
        record=json.loads(path.read_bytes())
        for item in (path,*(path.parent/record[key] for key in ('vertices_file','neutral_file','uv_file'))):
            item=item.resolve()
            if not item.is_relative_to(root) or not item.is_file():raise ValueError('Native pose data leaves its source bank')
            resources[item.relative_to(root).as_posix()]=(item,digest(item))
    budget=runpy.run_path(str(Path(__file__).with_name('resource_budget.py')))
    budget['require_disk']('native face sample snapshot',destination.parent,sum(path.stat().st_size for path,_ in resources.values())+16*1024**2)
    destination.mkdir(parents=True)
    for relative,(path,sha) in resources.items():
        target=destination/relative;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(path,target)
        if digest(target)!=sha:raise ValueError('Native pose changed while taking its export snapshot')
    for pose in bank['poses']:pose['file']=str(destination/Path(pose['file']).resolve().relative_to(root))
    bank.update(original_bank_sha256=original_sha,files={relative:{'sha256':sha} for relative,(_,sha) in resources.items()})
    if digest(bank_path)!=original_sha or any(digest(path)!=sha for path,sha in resources.values()):raise ValueError('Native source bank changed while staging')
    target=destination/'bank.json';target.write_text(json.dumps(bank,indent=2),encoding='utf-8');return target


def install(obj,part,bank_path,component_transform,stage,source_owner=None):
    import numpy as np
    from mathutils import Vector
    from mathutils.kdtree import KDTree
    bank_path=Path(bank_path).resolve();bank=json.loads(bank_path.read_bytes())
    if bank.get('status')!='complete' or not bank.get('source_mesh_unchanged'):
        raise ValueError('Choose a completed native face bank with preserved source geometry')
    if bank.get('source_mesh')!=part['asset']:
        raise ValueError('The native face bank belongs to a different skeletal mesh')
    if bank.get('source_owner') and (not source_owner or source_owner.get('asset')!=bank['source_owner'] or source_owner.get('sha256')!=bank.get('source_sha256')):
        raise ValueError('The transient face bank belongs to a different saved Creator character')
    if part.get('asset_sha256') and part['asset_sha256']!=bank.get('source_sha256'):
        raise ValueError('The native face mesh changed after its pose bank was evaluated')
    poses=bank.get('poses',[]);groups=bank.get('groups',[])
    if not poses or poses[0]['name']!='Neutral' or not groups or len({p['name'] for p in poses})!=len(poses):
        raise ValueError('Native face bank has missing or ambiguous samples')
    root=bank_path.parent;count=int(bank['vertices'])
    for relative,record in bank.get('files',{}).items():
        path=(root/relative).resolve()
        if not path.is_relative_to(root) or not path.is_file() or digest(path)!=record['sha256']:raise ValueError('Staged native face data changed')
    if count!=len(obj.data.vertices):raise ValueError('Native face bank vertex count does not match the exported reference mesh')
    files={};records={};points={};uvs=None;reference=None;sections=None
    def checked(path):
        path=Path(path).resolve()
        if not path.is_relative_to(root) or not path.is_file():raise ValueError('Native pose data is missing or outside its bank')
        files[str(path.relative_to(root))]=digest(path);return path
    for pose in poses:
        path=checked(pose['file']);record=json.loads(path.read_bytes());name=pose['name']
        if record.get('source_mesh')!=bank['source_mesh'] or record.get('dna_binding')!=bank.get('dna_binding') or record.get('dna_mode')!='ASSEMBLED_EMBEDDED':
            raise ValueError('Native sample identity/DNA binding mismatch: '+name)
        actual={key.lower():value for key,value in record['raw_curves'].items()}
        if any(abs(actual.get(('CTRL_expressions_'+key).lower(),float('inf'))-value)>1e-6 for key,value in pose['controls'].items()):
            raise ValueError('Native sample controls were not evaluated: '+name)
        data=checked(path.parent/record['vertices_file']);positions=np.fromfile(data,dtype='<f4').reshape(-1,3)
        ref=checked(path.parent/record['neutral_file']);neutral=np.fromfile(ref,dtype='<f4').reshape(-1,3)
        uvfile=checked(path.parent/record['uv_file']);uv=np.fromfile(uvfile,dtype='<f4').reshape(-1,2)
        if positions.shape!=(count,3) or neutral.shape!=(count,3) or uv.shape!=(count,2) or not all(np.isfinite(a).all() for a in (positions,neutral,uv)):
            raise ValueError('Native sample has invalid geometry: '+name)
        if reference is None:reference=neutral;uvs=uv;sections=record['sections']
        elif not np.array_equal(reference,neutral) or not np.array_equal(uvs,uv) or sections!=record['sections']:
            raise ValueError('Native sample reference surface changed: '+name)
        points[name]=positions;records[name]=record
    if np.linalg.norm(points['Neutral']-reference,axis=1).max()>1e-3:
        raise ValueError('Native neutral evaluation does not preserve this character reference surface')
    slots=np.full(count,-1,np.int32)
    for section in sections:
        start=int(section['start']);end=start+int(section['count'])
        if start<0 or end>count or np.any(slots[start:end]!=-1):raise ValueError('Native face material sections overlap or exceed their surface')
        slots[start:end]=int(section['material'])
    if np.any(slots<0):raise ValueError('Native material sections do not cover the reference surface')
    # FBX converts Unreal's left-handed centimeters into Blender's right-handed
    # meters. Compare in component space to avoid actor placement assumptions.
    transform=np.asarray(component_transform,dtype=np.float64)
    native=np.column_stack((reference*[.01,-.01,.01],np.ones(count)))@transform.T
    tree=KDTree(count)
    for index,point in enumerate(native[:,:3]):tree.insert(point,index)
    tree.balance();tolerance=3e-6
    world=np.asarray(obj.matrix_world,dtype=np.float64)
    local=np.array([v.co[:] for v in obj.data.vertices],np.float64)
    vertices=np.column_stack((local,np.ones(count)))@world.T
    candidates={i:{index for _,index,_ in tree.find_range(Vector(point[:3]),tolerance)} for i,point in enumerate(vertices)}
    if any(not values for values in candidates.values()):
        worst=max(min(distance for _,_,distance in [tree.find(Vector(point[:3]))]) for point in vertices)
        raise ValueError('Native reference geometry does not match the exported FBX (max nearest distance '+str(worst)+' m)')
    uv_layer=obj.data.uv_layers.active
    if not uv_layer:raise ValueError('Native pose binding requires the original face UVs')
    material_slots=list(range(len(part['materials'])))
    if len(obj.data.materials)!=len(material_slots):material_slots=sorted(set(part['lod_sections'][0]['materials']))
    for polygon in obj.data.polygons:
        slot=material_slots[polygon.material_index]
        for loop_index in polygon.loop_indices:
            loop=obj.data.loops[loop_index];uv=uv_layer.data[loop_index].uv
            expected=np.array([uv.x,1.-uv.y])
            candidates[loop.vertex_index]&={index for index in candidates[loop.vertex_index] if slots[index]==slot and np.max(np.abs(uvs[index]-expected))<3e-6}
    mapping=np.zeros(count,np.int32);ambiguous=0
    for vertex,choices in candidates.items():
        if not choices:raise ValueError('Native position/UV/material correspondence failed at vertex '+str(vertex))
        options=sorted(choices);selected=options[0]
        if len(options)>1:
            ambiguous+=1
            for name,data in points.items():
                if np.max(np.abs(data[options]-data[selected]))>1e-5:
                    raise ValueError('Coincident vertices have different native pose behavior: '+str(vertex)+' / '+name)
        mapping[vertex]=selected
    # Bake fully evaluated facial deformation once; retaining raw correctives
    # beside these samples would double-apply part of the rig response.
    if obj.data.shape_keys:
        for key in list(obj.data.shape_keys.key_blocks)[1:]:obj.shape_key_remove(key)
    if not obj.data.shape_keys:obj.shape_key_add(name='Basis')
    delta_transform=np.linalg.inv(world[:3,:3])@transform[:3,:3]@np.diag([.01,-.01,.01])
    targets={}
    for index,pose in enumerate(poses[1:],1):
        name=pose['name'];target='CMNativePose_'+str(index).zfill(3);targets[name]=target
        delta=(points[name]-points['Neutral'])[mapping]@delta_transform.T
        key=obj.shape_key_add(name=target);key.data.foreach_set('co',(local+delta).astype(np.float32).ravel());key.value=0.
    descriptor={'schema':1,'generator':'CreateMetaHuman.NativePoseBank','method':'Character-specific native joint-and-corrective samples; piecewise interpolation',
        'source_mesh':bank['source_mesh'],'source_mesh_sha256':bank['source_sha256'],'dna_binding':bank['dna_binding'],'bank_sha256':digest(bank_path),
        'neutral_preserved':True,'facial_bones':'REFERENCE_POSE','groups':[{'name':g['name'],'samples':[dict(s,target=targets[s['pose']]) for s in g['samples']]} for g in groups],
        'correspondence':{'vertices':count,'max_reference_error_m':float(np.linalg.norm(vertices[:,:3]-native[mapping,:3],axis=1).max()),'equivalent_coincident_vertices':ambiguous,'position_uv_material_checked':True},
        'limitations':['Interpolation of evaluated samples does not execute arbitrary RigLogic controls.','Combined mouth, blink and emotion samples approximate nonlinear interactions.','Driven skin appearance is not included in this pose bank.']}
    if bank.get('source_owner'):descriptor.update(source_owner=bank['source_owner'],source_fingerprint_scope=bank['source_fingerprint_scope'])
    (Path(stage)/'native-expression-bank.json').write_text(json.dumps(descriptor,indent=2),encoding='utf-8')
    for relative,sha in files.items():
        if digest(root/relative)!=sha:raise ValueError('Native pose bank changed during conversion')
    return descriptor
