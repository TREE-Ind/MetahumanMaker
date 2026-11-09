"""Portable strand ribbons bound to the exported facial surface.

This surface-frame binding is a browser approximation of native groom RBF
deformation. Original strands/settings remain in the native source bundle.
"""
import json
import re
from pathlib import Path


def bind(folder):
    """Legacy workspace helper; reusable deliveries use native_web_grooms.prepare."""
    import numpy as np
    from scipy.spatial import cKDTree
    folder=Path(folder)
    source=np.load(folder/'groom-source-surface.npz')
    target=np.load(folder/'face-bind-surface.npz')
    st=source['positions'][source['triangles']];tt=target['positions'][target['triangles']]
    suv=source['uvs'];tuv=target['uvs']
    source_tree=cKDTree(st.mean(1));target_tree=cKDTree(tuv.mean(1))

    def closest(points,triangles,tree):
        _,ids=tree.query(points,k=16)
        tri=triangles[ids];a=tri[:,:,0];u=tri[:,:,1]-a;v=tri[:,:,2]-a;q=points[:,None,:]-a
        dot=lambda x,y:np.einsum('...i,...i->...',x,y)
        uu=dot(u,u);uv=dot(u,v);vv=dot(v,v);qu=dot(q,u);qv=dot(q,v)
        den=uu*vv-uv*uv;good=abs(den)>1e-18;den=np.where(good,den,1)
        b=(qu*vv-qv*uv)/den;c=(qv*uu-qu*uv)/den
        bary=np.stack((1-b-c,b,c),axis=-1).clip(0,1);bary/=np.maximum(bary.sum(-1,keepdims=True),1e-20)
        projected=np.einsum('nki,nkij->nkj',bary,tri)
        distance=np.linalg.norm(projected-points[:,None,:],axis=-1);distance[~good]=np.inf
        chosen=distance.argmin(1);rows=np.arange(len(points))
        if not np.isfinite(distance[rows,chosen]).all():raise ValueError('Degenerate groom binding surface')
        return ids[rows,chosen],bary[rows,chosen],distance[rows,chosen]

    def frames(tri):
        u=tri[:,1]-tri[:,0];u/=np.maximum(np.linalg.norm(u,axis=-1,keepdims=True),1e-15)
        n=np.cross(u,tri[:,2]-tri[:,0]);n/=np.maximum(np.linalg.norm(n,axis=-1,keepdims=True),1e-15)
        v=np.cross(n,u)
        return np.stack((u,v,n),axis=-1)

    manifest=json.loads((folder/'source-manifest.json').read_text());records=[]
    for groom in manifest['grooms']:
        if not groom['asset']:continue
        data=json.loads((folder/groom['file']).read_text(encoding='utf-8-sig'))
        curves=[c for c in data['curves'] if not c['guide'] and len(c['points'])>=2]
        roots=np.array([c['points'][0] for c in curves],np.float64)*[.01,-.01,.01]
        sid,sb,sd=closest(roots,st,source_tree)
        uv=np.einsum('ni,nij->nj',sb,suv[sid])
        tid,tb,td=closest(uv,tuv,target_tree)
        if np.quantile(sd,.95)>.02 or np.quantile(td,.95)>.005:raise ValueError('Groom scalp correspondence is too uncertain')
        sf=frames(st[sid]);tf=frames(tt[tid]);sa=np.einsum('ni,nij->nj',sb,st[sid]);ta=np.einsum('ni,nij->nj',tb,tt[tid])
        points=[];widths=[];curve_records=[];offset=0
        settings=data.get('group_settings',[])
        render=settings[0].get('rendering','') if settings else ''
        def setting(name,default):
            match=re.search(r'\b'+name+r'=([0-9.eE+-]+)',render)
            return float(match.group(1)) if match else default
        override_width=setting('HairWidth',0) if 'HairWidth_Override=True' in render else 0
        root_scale=setting('HairRootScale',1);tip_scale=setting('HairTipScale',1)
        for i,curve in enumerate(curves):
            p=np.asarray(curve['points'],np.float64)*[.01,-.01,.01]
            local=(p-sa[i])@sf[i]
            points.append(local.astype('<f4'))
            w=np.array(curve['widths'],np.float64)
            if override_width:w=w/max(float(w.max()),1e-12)*override_width
            widths.extend(w*.01*np.linspace(root_scale,tip_scale,len(w)))
            curve_records.append({'triangle':int(tid[i]),'bary':tb[i].tolist(),'offset':offset,'count':len(p)})
            offset+=len(p)
        anchors=target['positions'][:,[0,2,1]].copy();anchors[:,2]*=-1
        arrays={'points':np.concatenate(points).astype('<f4'),'widths':np.array(widths,dtype='<f4'),
                'anchors':anchors.astype('<f4'),'triangles':target['triangles'].astype('<u4')}
        binary=bytearray();layout={}
        for name,array in arrays.items():
            layout[name]={'offset':len(binary),'count':int(array.size),'type':str(array.dtype)};binary.extend(array.tobytes())
        filename=groom['component']+'.strands.bin';(folder/filename).write_bytes(binary)
        material=next((m for m in groom['materials'] if m and m['asset'].endswith('_Hair')),groom['materials'][0])
        melanin=material['scalars'].get('HairMelanin',.5)
        records.append({'name':groom['component'],'file':filename,'layout':layout,'curves':curve_records,
            'color':[max(.015,.8*(1-melanin)**3)]*3,'roughness':material['scalars'].get('HairRoughness',.5),
            'source_curves':len(data['curves']),'render_curves':len(curves),'points':offset,
            'source_root_distance_p95_m':float(np.quantile(sd,.95)),'uv_distance_p95':float(np.quantile(td,.95)),
            'binding':'surface-frame approximation; native RBF and simulation are not reproduced'})
    (folder/'grooms.json').write_text(json.dumps({'schema':1,'grooms':records},separators=(',',':')))
    return [{k:v for k,v in r.items() if k not in ('curves','layout')} for r in records]


JAVASCRIPT=r'''import * as THREE from 'three';
export async function loadGrooms(model,url){
 const response=await fetch(url);if(!response.ok)throw Error('Groom manifest unavailable');
 const descriptor=await response.json(),meshes=[],surfaces=new Map();model.updateMatrixWorld(true);
 if(!Array.isArray(descriptor.grooms))throw Error('Invalid groom manifest');
 if(!descriptor.grooms.length)return {systems:[],update(){}};
 model.traverse(o=>{if(o.isSkinnedMesh)meshes.push(o)});
 const rest=new THREE.Vector3(),key=v=>[v.x,v.y,v.z].map(x=>Math.round(x*1e5)).join(',');
 function surface(name){
  if(!surfaces.has(name)){
   const face=meshes.find(o=>o.material.name===name);if(!face)throw Error('Groom target surface is missing: '+name);
   const index=new Map();face.skeleton.update();
   for(let i=0;i<face.geometry.attributes.position.count;i++){face.getVertexPosition(i,rest);rest.applyMatrix4(face.matrixWorld);model.worldToLocal(rest);index.set(key(rest),i)}
   surfaces.set(name,{face,index});
  }
  return surfaces.get(name);
 }
 const systems=[];
 for(const record of descriptor.grooms){
  const {face,index}=surface(record.target_material??'CMNative_Face_0');
  const raw=await fetch(new URL(record.file,url)).then(r=>{if(!r.ok)throw Error('Missing strands');return r.arrayBuffer()});
  const read=n=>{const a=record.layout[n];return n==='triangles'?new Uint32Array(raw,a.offset,a.count):new Float32Array(raw,a.offset,a.count)};
  const local=read('points'),widths=read('widths'),anchors=read('anchors'),triangles=read('triangles');
  const positions=new Float32Array(record.points*6),tangents=new Float32Array(record.points*6),sides=new Float32Array(record.points*2),width=new Float32Array(record.points*2),indices=[];
  const needed=new Set();for(const curve of record.curves)for(let j=0;j<3;j++)needed.add(triangles[curve.triangle*3+j]);
  const links=new Map();for(const a of needed){rest.fromArray(anchors,a*3);let i=index.get(key(rest));
   if(i===undefined){
    const q=[rest.x,rest.y,rest.z].map(x=>Math.round(x*1e5));let best=Infinity;
    for(let x=-1;x<=1;x++)for(let y=-1;y<=1;y++)for(let z=-1;z<=1;z++){
     const candidate=index.get([q[0]+x,q[1]+y,q[2]+z].join(','));if(candidate===undefined)continue;
     const v=new THREE.Vector3();face.getVertexPosition(candidate,v);v.applyMatrix4(face.matrixWorld);model.worldToLocal(v);const distance=v.distanceToSquared(rest);
     if(distance<best){best=distance;i=candidate}
    }
   }
   if(i===undefined)throw Error('Groom anchor does not match exported face: '+rest.toArray()+' mesh bounds '+JSON.stringify(new THREE.Box3().setFromObject(face)));
   links.set(a,i)}
  for(const c of record.curves)for(let i=0;i<c.count;i++){
   const p=c.offset+i;sides[p*2]=-1;sides[p*2+1]=1;width[p*2]=width[p*2+1]=Math.max(0,widths[p])*.5;
   if(i+1<(c.render_count??c.count)){const a=p*2;indices.push(a,a+1,a+2,a+1,a+3,a+2)}
  }
  const geometry=new THREE.BufferGeometry();geometry.setAttribute('position',new THREE.BufferAttribute(positions,3).setUsage(THREE.DynamicDrawUsage));
  geometry.setAttribute('cmTangent',new THREE.BufferAttribute(tangents,3).setUsage(THREE.DynamicDrawUsage));
  geometry.setAttribute('normal',geometry.attributes.cmTangent);
  geometry.setAttribute('cmSide',new THREE.BufferAttribute(sides,1));geometry.setAttribute('cmWidth',new THREE.BufferAttribute(width,1));geometry.setIndex(indices);
  const material=new THREE.MeshPhysicalMaterial({color:new THREE.Color(...record.color),roughness:record.roughness,side:THREE.DoubleSide});
  const viewToLocal=new THREE.Matrix3(),viewMatrix=new THREE.Matrix4();
  material.userData.cmGroomViewToLocal=viewToLocal;
  material.onBeforeCompile=shader=>{
   shader.uniforms.cmGroomViewToLocal={value:viewToLocal};
   shader.vertexShader='attribute vec3 cmTangent;attribute float cmSide;attribute float cmWidth;uniform mat3 cmGroomViewToLocal;\n'+shader.vertexShader;
   shader.vertexShader=shader.vertexShader.replace('#include <project_vertex>',`#include <project_vertex>
    vec3 cmT=mat3(modelViewMatrix)*cmTangent;cmT/=max(length(cmT),.000001);
    vec3 cmAcross=cross(cmT,-mvPosition.xyz/max(length(mvPosition.xyz),.000001));cmAcross/=max(length(cmAcross),.000001);
    float cmRadiusScale=1.0/max(length(cmGroomViewToLocal*cmAcross),.000001);
    mvPosition.xyz+=cmAcross*cmSide*cmWidth*cmRadiusScale;
    gl_Position=projectionMatrix*mvPosition;`);
   shader.fragmentShader=shader.fragmentShader.replace('#include <normal_fragment_begin>',`#include <normal_fragment_begin>
    normal=vViewPosition/max(length(vViewPosition),.000001);`);
  };
  material.customProgramCacheKey=()=> 'CMStrandRibbon2';
  const object=new THREE.Mesh(geometry,material);object.name=record.name;object.frustumCulled=false;model.add(object);
  object.visible=record.visible!==false;
  object.onBeforeRender=(renderer,scene,camera)=>{viewMatrix.multiplyMatrices(camera.matrixWorldInverse,object.matrixWorld);viewToLocal.setFromMatrix4(viewMatrix).invert()};
  systems.push({record,local,triangles,links,positions,tangents,geometry,object,face});
 }
 function update(){
  model.updateMatrixWorld(true);for(const {face} of surfaces.values())face.skeleton.update();
  const caches=new Map();
  function reader(face){if(!caches.has(face))caches.set(face,new Map());const cache=caches.get(face);return i=>{if(!cache.has(i)){const v=new THREE.Vector3();face.getVertexPosition(i,v);v.applyMatrix4(face.matrixWorld);model.worldToLocal(v);cache.set(i,v)}return cache.get(i)}}
  const u=new THREE.Vector3(),v=new THREE.Vector3(),n=new THREE.Vector3(),anchor=new THREE.Vector3(),p=new THREE.Vector3(),t=new THREE.Vector3();
  for(const s of systems){
   const get=reader(s.face);
   for(const c of s.record.curves){
    const a=get(s.links.get(s.triangles[c.triangle*3])),b=get(s.links.get(s.triangles[c.triangle*3+1])),d=get(s.links.get(s.triangles[c.triangle*3+2]));
    u.subVectors(b,a).normalize();n.crossVectors(u,t.subVectors(d,a)).normalize();v.crossVectors(n,u);
    anchor.copy(a).multiplyScalar(c.bary[0]).addScaledVector(b,c.bary[1]).addScaledVector(d,c.bary[2]);
    for(let j=0;j<c.count;j++){const i=c.offset+j,k=i*3;p.copy(anchor).addScaledVector(u,s.local[k]).addScaledVector(v,s.local[k+1]).addScaledVector(n,s.local[k+2]);
     // Store model-local points so scene placement cannot apply a second time.
     p.toArray(s.positions,i*6);p.toArray(s.positions,i*6+3);
    }
    for(let j=0;j<c.count;j++){const i=c.offset+j,prev=Math.max(c.offset,i-1)*6,next=Math.min(c.offset+c.count-1,i+1)*6;
     t.fromArray(s.positions,next).sub(p.fromArray(s.positions,prev)).normalize();t.toArray(s.tangents,i*6);t.toArray(s.tangents,i*6+3)}
   }
   s.geometry.attributes.position.needsUpdate=true;s.geometry.attributes.cmTangent.needsUpdate=true;
  }
 }
 update();
 return {systems,update};
}
'''


def write(path):
    Path(path).write_text(JAVASCRIPT,encoding='utf-8')
