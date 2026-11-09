"""Browser interpolation of character-specific, fully evaluated native poses."""
from pathlib import Path

JAVASCRIPT=r'''export async function loadNativePoseBank(model,url){
 const response=await fetch(url);if(!response.ok)throw Error('Native face bank unavailable');const descriptor=await response.json();
 if(descriptor.generator!=='CreateMetaHuman.NativePoseBank'||descriptor.facial_bones!=='REFERENCE_POSE'||!descriptor.neutral_preserved)throw Error('Unsupported native face bank');
 const groups=new Map(),targets=new Set();
 for(const group of descriptor.groups){
  if(groups.has(group.name)||!group.samples.length)throw Error('Ambiguous native expression group');
  let previous=0;for(const sample of group.samples){if(!(sample.strength>previous&&sample.strength<=1)||typeof sample.target!=='string'||targets.has(sample.target))throw Error('Invalid native pose sample');previous=sample.strength;targets.add(sample.target);}
  if(previous!==1)throw Error('Native expression requires a full-strength sample');groups.set(group.name,group);
 }
 const meshes=[];model.traverse(o=>{if(o.isSkinnedMesh&&o.morphTargetDictionary){const indices=[...targets].map(t=>o.morphTargetDictionary[t]).filter(i=>i!==undefined);if(indices.length)meshes.push({object:o,indices});}});
 if(!meshes.length||[...targets].some(t=>!meshes.some(m=>m.object.morphTargetDictionary[t]!==undefined)))throw Error('Native pose geometry is incomplete');
 let signature='',revision=0,active='Neutral',strength=0,applied={};
 function commit(values){
  const ordered=Object.entries(values).filter(([,v])=>v>1e-5).sort(([a],[b])=>a.localeCompare(b));
  const next=ordered.map(([n,v])=>n+':'+v.toFixed(5)).join('|');if(next===signature)return;signature=next;
  for(const {object,indices} of meshes)for(const i of indices)object.morphTargetInfluences[i]=0;
  const skinWeights={};
  for(const [name,value] of ordered){
   const group=groups.get(name);if(!group)throw Error('Unknown native expression '+name);
   const samples=[{strength:0},...group.samples],upper=samples.findIndex(s=>s.strength>=value),a=samples[Math.max(0,upper-1)],b=samples[upper];
   const t=(value-a.strength)/(b.strength-a.strength);
   for(const [sample,weight] of [[a,1-t],[b,t]])if(sample.target&&weight>0){
    if(sample.pose)skinWeights[sample.pose]=(skinWeights[sample.pose]??0)+weight;
    for(const {object} of meshes){const i=object.morphTargetDictionary[sample.target];if(i!==undefined)object.morphTargetInfluences[i]+=weight;}
   }
  }
  const updated=new Set();model.traverse(o=>{for(const m of (Array.isArray(o.material)?o.material:[o.material]))if(m?.userData.cmSkinMotion&&!updated.has(m)){m.userData.cmSkinMotion.setWeights(skinWeights);updated.add(m);}});
  applied=Object.fromEntries(ordered);revision++;model.updateMatrixWorld(true);
 }
 function setState({mouthWeights={},blink=0,strength:gain=1,expressionWeights={}}={}){
  const values={},entries=Object.entries(mouthWeights).filter(([name])=>name!=='sil');
  if(!Number.isFinite(gain)||gain<0||gain>1.5||!Number.isFinite(blink))throw Error('Invalid native face strength');
  const total=entries.reduce((sum,[,v])=>sum+Math.max(0,v),0);
  for(const [name,weight] of entries){if(!Number.isFinite(weight)||!groups.has(name))throw Error('Invalid native mouth shape');values[name]=Math.min(1,Math.max(0,weight)*gain/Math.max(1,total));}
  for(const [name,value] of Object.entries(expressionWeights)){if(!groups.has(name)||!Number.isFinite(value))throw Error('Invalid native expression');values[name]=Math.min(1,Math.max(0,value));}
  if(groups.has('Blink'))values.Blink=Math.min(1,Math.max(0,blink));commit(values);
 }
 function apply(name,value){if(!Number.isFinite(value)||value<0||value>1||name!=='Neutral'&&!groups.has(name))throw Error('Invalid native pose');active=name;strength=value;commit(name==='Neutral'?{}:{[name]:value});model.userData.grooms?.update();}
 const controller={groups:[...groups.keys()],apply,setState,get active(){return active},get strength(){return strength},get revision(){return revision},get state(){return {...applied}},method:descriptor.method,nativeSamples:true,descriptor};
 model.userData.nativeFaceController=controller;model.userData.expressionController=controller;return controller;
}
'''


def write(path):
    Path(path).write_text(JAVASCRIPT,encoding='utf-8')
