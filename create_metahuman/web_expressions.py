"""Character-specific, natively evaluated expression samples for web previews.

Samples include both skinning and correctives. Interpolation is bounded to one
authored expression curve; it is not arbitrary multi-control RigLogic execution.
"""
from pathlib import Path


def correspondence(vertices,loop_vertices,uvs,mesh,reference,folder,tolerance=3e-6):
    """Resolve native layout identities; reject unproven coincident vertices."""
    import numpy as np
    points=np.asarray(mesh['neutral'])*[.01,-.01,.01]
    buckets={};cache={};possible={};maximum=0
    for index,u,v in mesh['layouts']:
        key=tuple(np.rint(np.array([u,v])/tolerance).astype(int))
        buckets.setdefault(key,[]).append((int(index),np.array([u,v])))
    for vertex,uv in zip(loop_vertices,uvs):
        key=tuple(uv)
        if key not in cache:
            cell=np.rint(np.array(uv)/tolerance).astype(int)
            cache[key]={i for x in (-1,0,1) for y in (-1,0,1) for i,q in buckets.get(tuple(cell+[x,y]),[]) if np.linalg.norm(q-uv)<=tolerance}
        matches={i for i in cache[key] if np.linalg.norm(points[i]-vertices[vertex])<2e-5}
        possible[vertex]=matches if vertex not in possible else possible[vertex]&matches
    resolved={}
    for vertex,choices in possible.items():
        if not choices:raise ValueError('Native expression layout/geometry mismatch: '+mesh['name']+' vertex '+str(vertex))
        candidates=sorted(choices)
        if len(candidates)>1:
            for pose in reference['poses']:
                record=next(m for m in pose['meshes'] if m['index']==mesh['index'])
                values=np.memmap(Path(folder)/record['vertices_file'],dtype='<f4',mode='r',shape=(len(points),3))
                if not all(np.array_equal(values[candidates[0]],values[i]) for i in candidates[1:]):
                    raise ValueError('Coincident vertices have different native expression behavior')
        chosen=candidates[0];resolved[vertex]=chosen
        maximum=max(maximum,float(np.linalg.norm(points[chosen]-vertices[vertex])))
    return resolved,maximum


JAVASCRIPT=r'''export async function loadExpressionSamples(model,url){
 const response=await fetch(url);if(!response.ok)throw Error('Native expression samples unavailable');const descriptor=await response.json();
 const meshes=[];model.traverse(o=>{if(o.isSkinnedMesh&&o.morphTargetDictionary)meshes.push(o)});
 const expected=descriptor.groups.flatMap(g=>g.samples.map(s=>s.target));
 if(!meshes.length||expected.some(n=>!meshes.some(o=>o.morphTargetDictionary[n]!==undefined)))throw Error('Native expression geometry is incomplete');
 let active='Neutral',strength=0;
 function apply(name,value){
  if(!Number.isFinite(value)||value<0||value>1)throw Error('Expression strength outside 0..1');
  const group=descriptor.groups.find(g=>g.name===name);if(name!=='Neutral'&&!group)throw Error('Unknown native expression');
  for(const o of meshes)o.morphTargetInfluences.fill(0);
  const maps=new Float32Array(descriptor.animated_maps.length);
  if(group&&value>0){
   const samples=[{strength:0,animated_maps:descriptor.neutral_maps},...group.samples];
   const upper=samples.findIndex(s=>s.strength>=value),a=samples[Math.max(0,upper-1)],b=samples[upper];
   const t=(value-a.strength)/(b.strength-a.strength);
   for(const [sample,weight] of [[a,1-t],[b,t]]){
    if(sample.target)for(const o of meshes){const i=o.morphTargetDictionary[sample.target];if(i!==undefined)o.morphTargetInfluences[i]=weight}
    for(let i=0;i<maps.length;i++)maps[i]+=(sample.animated_maps[i]??0)*weight;
   }
  }else maps.set(descriptor.neutral_maps);
  active=name;strength=value;model.userData.animatedMaps={names:descriptor.animated_maps,values:maps};
  model.updateMatrixWorld(true);model.userData.grooms?.update();
 }
 const controller={groups:descriptor.groups.map(g=>g.name),apply,get active(){return active},get strength(){return strength},method:descriptor.method,calibrated:descriptor.calibrated===true};
 model.userData.expressionController=controller;apply('Neutral',0);return controller;
}
'''


def write(path):
    Path(path).write_text(JAVASCRIPT,encoding='utf-8')


def descriptor(reference):
    method=reference.get('expression_method','Native joint-and-corrective samples')
    result={'schema':1,'source_character':reference['source_character'],'dna_sha256':reference['dna_sha256'],'calibrated':bool(reference.get('calibration')),
            'method':method+'; piecewise interpolation at strengths 0/.25/.5/.75/1; one expression at a time',
            'animated_maps':reference['animated_maps'],'neutral_maps':reference['poses'][0]['animated_maps'],
            'groups':[{'name':g['name'],'samples':[dict(s,target='CMExpr_'+str(s['pose']).zfill(3),animated_maps=reference['poses'][s['pose']]['animated_maps']) for s in g['samples']]} for g in reference['expression_groups']]}
    if reference.get('calibration'):result['calibration']=reference['calibration']
    return result
