"""Reusable, source-bound strand preparation; native groom RBF remains separate."""
import copy,hashlib,json,re,runpy
from pathlib import Path
from tempfile import TemporaryDirectory
import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation

GENERATOR='CreateMetaHuman.NativeWebGrooms'

def digest(path):
    with path.open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()

def checked(root,record):
    path=(root/record['file']).resolve()
    if not path.is_relative_to(root) or not path.is_file() or digest(path)!=record['sha256']:raise ValueError('Groom resource is missing, changed or outside its capture')
    return path

def matrix(value):
    if value is None:raise ValueError('Groom preparation needs captured actor-space transforms')
    t=np.asarray(value['translation_cm'],float);q=np.asarray(value['rotation_xyzw'],float);s=np.asarray(value['scale'],float)
    if t.shape!=(3,) or q.shape!=(4,) or s.shape!=(3,) or not all(np.isfinite(x).all() for x in (t,q,s)) or abs(np.linalg.norm(q)-1)>.001 or np.any(abs(s)<1e-8):raise ValueError('Invalid groom component transform')
    flip=np.diag([1,-1,1]);result=np.eye(4);result[:3,:3]=flip@Rotation.from_quat(q).as_matrix()@np.diag(s)@flip;result[:3,3]=t*[.01,-.01,.01];return result

def surface(path):
    with np.load(path,allow_pickle=False) as data:result={k:np.array(data[k]) for k in ('positions','triangles','uvs')}
    p,t,u=(result[k] for k in ('positions','triangles','uvs'))
    if p.ndim!=2 or p.shape[1]!=3 or t.ndim!=2 or t.shape[1]!=3 or not len(t) or u.shape!=(len(t),3,2) or not np.issubdtype(t.dtype,np.integer) or t.min()<0 or t.max()>=len(p) or not np.isfinite(p).all() or not np.isfinite(u).all():raise ValueError('Invalid groom binding surface')
    return result

def closest(points,triangles):
    _,ids=cKDTree(triangles.mean(1)).query(points,k=min(16,len(triangles)))
    if ids.ndim==1:ids=ids[:,None]
    tri=triangles[ids];a=tri[:,:,0];u=tri[:,:,1]-a;v=tri[:,:,2]-a;q=points[:,None,:]-a
    dot=lambda x,y:np.einsum('...i,...i->...',x,y)
    uu=dot(u,u);uv=dot(u,v);vv=dot(v,v);qu=dot(q,u);qv=dot(q,v);den=uu*vv-uv*uv;good=abs(den)>1e-18;den=np.where(good,den,1)
    b=(qu*vv-qv*uv)/den;c=(qv*uu-qu*uv)/den;bary=np.stack((1-b-c,b,c),axis=-1).clip(0,1);bary/=np.maximum(bary.sum(-1,keepdims=True),1e-20)
    projected=np.einsum('nki,nkij->nkj',bary,tri);distance=np.linalg.norm(projected-points[:,None,:],axis=-1);distance[~good]=np.inf
    chosen=distance.argmin(1);rows=np.arange(len(points))
    if not np.isfinite(distance[rows,chosen]).all():raise ValueError('Degenerate groom binding surface')
    return ids[rows,chosen],bary[rows,chosen],distance[rows,chosen]

def frames(tri):
    u=tri[:,1]-tri[:,0];u/=np.maximum(np.linalg.norm(u,axis=-1,keepdims=True),1e-15)
    n=np.cross(u,tri[:,2]-tri[:,0]);n/=np.maximum(np.linalg.norm(n,axis=-1,keepdims=True),1e-15);return np.stack((u,np.cross(n,u),n),axis=-1)

def bind_data(groom,data,source,target,target_transform,groom_transform,target_material):
    """Return every rendered group, preserving each strand and its width profile."""
    if data.get('source_asset')!=groom['asset']:raise ValueError('Captured strands belong to another groom asset')
    curves=[c for c in data['curves'] if not c['guide'] and len(c['points'])>=2]
    if not curves:raise ValueError('The captured groom contains no rendered strands')
    st=source['positions'][source['triangles']];tt=target['positions'][target['triangles']]
    roots=np.asarray([c['points'][0] for c in curves],float)*[.01,-.01,.01]
    sid,sb,sd=closest(roots,st);uv=np.einsum('ni,nij->nj',sb,source['uvs'][sid]);tid,tb,td=closest(uv,target['uvs'])
    if np.quantile(sd,.95)>.02 or np.quantile(td,.95)>.005:raise ValueError('Groom scalp correspondence is too uncertain')
    sf=frames(st[sid]);tf=frames(tt[tid]);sa=np.einsum('ni,nij->nj',sb,st[sid]);ta=np.einsum('ni,nij->nj',tb,tt[tid])
    inverse=np.linalg.inv(target_transform);target_local=tt@inverse[:3,:3].T+inverse[:3,3];local_frames=frames(target_local[tid])
    target_vectors=np.einsum('ij,njk->nik',target_transform[:3,:3],local_frames);delta=groom_transform@inverse
    settings={int(s['id']):(i,s) for i,s in enumerate(data.get('group_settings',[]))};records=[];payloads={}
    for group in sorted({int(c['group']) for c in curves}):
        selected=[i for i,c in enumerate(curves) if int(c['group'])==group]
        if group not in settings:raise ValueError('Native groom group settings are missing')
        group_index,setting=settings[group];render=setting.get('rendering','')
        direct_material=re.search(r'\bMaterial=([^,)]*)',render)
        if direct_material and direct_material.group(1)!='None':raise ValueError('Recapture the direct groom-group material override before web preparation')
        def number(name,default):
            match=re.search(r'\b'+name+r'=([0-9.eE+-]+)',render);return float(match.group(1)) if match else default
        width=number('HairWidth',0) if 'HairWidth_Override=True' in render else None
        root_scale=number('HairRootScale',1);tip_scale=number('HairTipScale',1);length=1
        overrides={int(d['index']):d for d in groom.get('component_group_settings',[])}.get(group_index,{})
        for key in ('hair_width','hair_root_scale','hair_tip_scale','hair_length_scale'):
            if overrides.get(key+'_override'):
                value=float(overrides[key])
                if key=='hair_width':width=value
                elif key=='hair_root_scale':root_scale=value
                elif key=='hair_tip_scale':tip_scale=value
                else:length=value
        if not all(np.isfinite(x) and x>=0 for x in (root_scale,tip_scale,length)) or length>1 or width is not None and (not np.isfinite(width) or width<0):raise ValueError('Invalid native groom width or length settings')
        slot=re.search(r'\bMaterialSlotName="([^"]+)"',render)
        names=[re.search(r'\bSlotName="([^"]+)"',text) for text in setting.get('materials',[])]
        indices=[i for i,name in enumerate(names) if name and slot and name.group(1)==slot.group(1)]
        materials=groom.get('materials',[])
        material_index=indices[0] if len(indices)==1 else 0 if len(materials)==1 else None
        if material_index is None or material_index>=len(materials) or not materials[material_index]:raise ValueError('Native groom group material is ambiguous')
        material=materials[material_index];scalars={name.casefold():value for name,value in material['scalars'].items()}
        melanin=float(scalars.get('hairmelanin',.5));roughness=float(scalars.get('hairroughness',.5))
        if not 0<=melanin<=1 or not 0<=roughness<=1:raise ValueError('Invalid native hair appearance parameters')
        maximum=max(float(np.max(curves[i]['widths'])) for i in selected);points=[];widths=[];curve_records=[];offset=0
        width_scale=float(abs(np.linalg.det(groom_transform[:3,:3]))**(1/3))
        for i in selected:
            curve=curves[i];p=np.asarray(curve['points'],float)*[.01,-.01,.01];w=np.asarray(curve['widths'],float)
            if p.shape!=(len(w),3) or not np.isfinite(p).all() or not np.isfinite(w).all():raise ValueError('Invalid native strand positions or widths')
            w=np.maximum(w,0)
            arc=np.concatenate(([0],np.cumsum(np.linalg.norm(np.diff(p,axis=0),axis=1))))
            if arc[-1]<1e-12:raise ValueError('Native groom contains a zero-length rendered strand')
            coord=arc/arc[-1];render_count=int(np.searchsorted(coord,length,side='right'))
            local=(p-sa[i])@sf[i];bound=local@target_vectors[i].T+ta[i];bound=bound@delta[:3,:3].T+delta[:3,3]
            points.append(((bound-ta[i])@tf[i]).astype('<f4'))
            # UE normalizes radii against the group maximum, not each curve.
            if width is not None:w=w/max(maximum,1e-12)*width
            widths.extend(w*.01*width_scale*(root_scale+(tip_scale-root_scale)*coord))
            curve_records.append({'triangle':int(tid[i]),'bary':tb[i].tolist(),'offset':offset,'count':len(p),'render_count':render_count});offset+=len(p)
        anchors=target['positions'][:,[0,2,1]].copy();anchors[:,2]*=-1
        arrays={'points':np.concatenate(points).astype('<f4'),'widths':np.asarray(widths,dtype='<f4'),'anchors':anchors.astype('<f4'),'triangles':target['triangles'].astype('<u4')}
        binary=bytearray();layout={}
        for name,array in arrays.items():layout[name]={'offset':len(binary),'count':int(array.size),'type':str(array.dtype)};binary.extend(array.tobytes())
        slug=re.sub('[^A-Za-z0-9_-]','_',groom['component']);filename=slug+'-'+hashlib.sha256(groom['component'].encode()).hexdigest()[:8]+'-g'+str(group)+'.strands.bin';payloads[filename]=bytes(binary)
        records.append({'name':groom['component'] if len(settings)==1 else groom['component']+' / '+setting.get('name',str(group)),
            'component':groom['component'],'group':group,'file':filename,'layout':layout,'curves':curve_records,'target_material':target_material,
            'visible':groom.get('visible',True) and length>0,'color':[max(.015,.8*(1-melanin)**3)]*3,'roughness':roughness,
            'source_curves':sum(int(c['group'])==group for c in data['curves']),'render_curves':len(selected),'points':offset,
            'source_root_distance_p95_m':float(np.quantile(sd[selected],.95)),'uv_distance_p95':float(np.quantile(td[selected],.95)),
            'width_normalization':'native group maximum','component_settings':copy.deepcopy(overrides),'native_group_settings':copy.deepcopy(setting),
            'negative_width_samples_clamped':sum(int(np.count_nonzero(np.asarray(curves[i]['widths'])<0)) for i in selected),
            'native_material':copy.deepcopy(material),'nonuniform_width_approximation':bool(np.ptp(np.linalg.svd(groom_transform[:3,:3],compute_uv=False))>1e-5),
            'binding':'surface-frame approximation; native RBF and simulation are not reproduced'})
    return records,payloads

def prepare(source_root,geometry_root,binding_manifest,destination,*,allow_legacy=False):
    root=Path(source_root).resolve();geometry_root=Path(geometry_root).resolve();destination=Path(destination).absolute()
    if type(allow_legacy) is not bool or destination.exists() or destination.resolve()!=destination or destination.is_relative_to(root) or destination.is_relative_to(geometry_root):raise ValueError('Choose a new separate groom destination and explicit legacy option')
    source_bytes=(root/'source-manifest.json').read_bytes();source=json.loads(source_bytes);geometry_bytes=(geometry_root/'geometry-manifest.json').read_bytes();geometry=json.loads(geometry_bytes)
    binding_path=Path(binding_manifest);binding_bytes=binding_path.read_bytes();bindings=json.loads(binding_bytes);source_sha=hashlib.sha256(source_bytes).hexdigest()
    if source.get('generator')!='CreateMetaHuman.NativeWebSource' or source.get('status')!='source_extracted' or geometry.get('generator')!='CreateMetaHuman.NativeWebGeometry' or geometry.get('source_sha256')!=source_sha:raise ValueError('Groom geometry does not match a completed native capture')
    if bindings.get('generator')!='CreateMetaHuman.NativeWebGroomBindings' or bindings.get('source_manifest_sha256')!=source_sha or bindings.get('geometry_manifest_sha256')!=hashlib.sha256(geometry_bytes).hexdigest():raise ValueError('Groom binding metadata belongs to another capture or geometry')
    groom_rows=[g for g in source.get('grooms',[]) if g.get('asset')];lookup={b['component']:b for b in bindings['bindings']}
    if len(lookup)!=len(bindings['bindings']) or set(lookup)!={g['component'] for g in groom_rows}:raise ValueError('Binding metadata must cover every captured groom exactly once')
    parts={p['component']:p for p in geometry['parts']};resources=[];records=[];cache={}
    for groom in groom_rows:
        row=lookup[groom['component']];sha=groom.get('sha256')
        if sha is None and not allow_legacy:raise ValueError('Recapture groom provenance before preparing this legacy source')
        if sha and row.get('groom_sha256',sha)!=sha:raise ValueError('Groom binding uses different source strands')
        path=checked(root,{'file':groom['file'],'sha256':sha or row['groom_sha256']});resources.append((path,sha or row['groom_sha256']))
        source_base=geometry_root if row['source_surface'].get('scope')=='geometry' else root
        resources.extend((checked(base,item),item['sha256']) for base,item in ((source_base,row['source_surface']),(geometry_root,row['target_surface'])))
    budget=runpy.run_path(str(Path(__file__).with_name('resource_budget.py')));budget['require']('native groom preparation',512*1024**2+max((p.stat().st_size*16 for p,_ in resources),default=0))
    destination.parent.mkdir(parents=True,exist_ok=True);budget['require_disk']('native groom preparation',destination.parent,max(128*1024**2,sum(p.stat().st_size for p,_ in resources)*2))
    with TemporaryDirectory(prefix='.native-grooms-',dir=destination.parent) as temporary:
        stage=Path(temporary)
        for groom in groom_rows:
            row=lookup[groom['component']]
            if row['target_component'] not in parts:raise ValueError('Groom target component was not converted')
            part=parts[row['target_component']]
            alias='Face' if row['target_component']==geometry.get('face_component','Face') else row['target_component']
            valid_materials={'CMNative_'+alias+'_'+str(slot) for slot in part['native_material_slots']}
            if row['target_material'] not in valid_materials:raise ValueError('Groom target material is not on its converted component')
            def load(base,item):
                path=checked(base,item)
                if path not in cache:cache[path]=surface(path)
                return cache[path]
            target_value=part.get('actor_relative_transform');groom_value=groom.get('actor_relative_transform')
            if allow_legacy:
                target_value=target_value or next(p['transform'] for p in source['skeletal_meshes'] if p['component']==row['target_component']);groom_value=groom_value or groom['transform']
            data=json.loads((root/groom['file']).read_text(encoding='utf-8-sig'))
            source_base=geometry_root if row['source_surface'].get('scope')=='geometry' else root
            bound,payloads=bind_data(groom,data,load(source_base,row['source_surface']),load(geometry_root,row['target_surface']),matrix(target_value),matrix(groom_value),row['target_material'])
            records.extend(bound)
            for name,payload in payloads.items():(stage/name).write_bytes(payload)
        result={'schema':1,'generator':GENERATOR,'source_manifest_sha256':source_sha,'geometry_manifest_sha256':hashlib.sha256(geometry_bytes).hexdigest(),
            'binding_manifest_sha256':hashlib.sha256(binding_bytes).hexdigest(),'legacy_capture':allow_legacy,'grooms':records,'native_rbf':False,'native_simulation':False}
        result['native_records']={g['component']:copy.deepcopy(g) for g in groom_rows}
        result['binding_record']=copy.deepcopy(bindings)
        result['files']={p.name:{'bytes':p.stat().st_size,'sha256':digest(p)} for p in stage.iterdir() if p.is_file()}
        (stage/'grooms.json').write_text(json.dumps(result,separators=(',',':'),allow_nan=False),encoding='utf-8')
        runpy.run_path(str(Path(__file__).with_name('web_grooms.py')))['write'](stage/'grooms.js')
        if (root/'source-manifest.json').read_bytes()!=source_bytes or (geometry_root/'geometry-manifest.json').read_bytes()!=geometry_bytes or binding_path.read_bytes()!=binding_bytes:raise ValueError('Groom capture changed during preparation')
        for path,sha in resources:
            if digest(path)!=sha:raise ValueError('Groom source changed during preparation')
        stage.replace(destination)
    return result

def binding_plan(source_root,geometry_root):
    """Resolve fresh captures directly; old captures need explicit binding records."""
    root=Path(source_root);geo=Path(geometry_root);source_path=root/'source-manifest.json';geometry_path=geo/'geometry-manifest.json'
    source=json.loads(source_path.read_bytes());geometry=json.loads(geometry_path.read_bytes());sha=digest(source_path)
    if geometry.get('source_sha256')!=sha:raise ValueError('Geometry belongs to another capture')
    parts={p['component']:p for p in geometry['parts']};rows=[]
    for groom in source.get('grooms',[]):
        if not groom.get('asset'):continue
        if not groom.get('sha256') or groom.get('binding_source') not in geometry.get('binding_sources',{}) or groom.get('target_component') not in parts:raise ValueError('Recapture and convert groom binding provenance first')
        part=parts[groom['target_component']];filename=part.get('binding_surface')
        if not filename or filename not in geometry['files']:raise ValueError('Converted groom target has no binding surface')
        alias='Face' if groom['target_component']==geometry.get('face_component','Face') else groom['target_component']
        rows.append({'component':groom['component'],'groom_sha256':groom['sha256'],'source_surface':{'scope':'geometry',**geometry['binding_sources'][groom['binding_source']]},
            'target_component':groom['target_component'],'target_material':'CMNative_'+alias+'_0',
            'target_surface':{'file':filename,'sha256':geometry['files'][filename]['sha256']}})
    return {'schema':1,'generator':'CreateMetaHuman.NativeWebGroomBindings','source_manifest_sha256':sha,'geometry_manifest_sha256':digest(geometry_path),'bindings':rows}
