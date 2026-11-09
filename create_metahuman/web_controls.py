"""Capability-based face controls for the native web viewer."""

JAVASCRIPT=r'''import * as THREE from 'three';
export function createFaceControls(model) {
 const panel=document.createElement('aside');panel.id='face-controls';
 panel.style.cssText='position:fixed;right:16px;top:90px;width:240px;max-height:72vh;overflow:auto;padding:16px;background:#171b22ed;border-radius:10px;z-index:2';
 const title=document.createElement('strong');title.textContent='Face controls';panel.append(title);
 const note=document.createElement('p');note.style.cssText='font-size:12px;color:#b8c2d0';panel.append(note);
 if(model.userData.grooms?.systems.length){
  const label=document.createElement('label'),toggle=document.createElement('input');toggle.type='checkbox';toggle.checked=model.userData.grooms.systems.some(s=>s.object.visible);
  label.style.cssText='display:block;margin:12px 0';
  toggle.onchange=()=>{for(const s of model.userData.grooms.systems)s.object.visible=toggle.checked};
  label.append(toggle,document.createTextNode(' Show grooms'));panel.append(label);
 }
 const controls=[],bones=new Set();model.traverse(o=>{if(o.isSkinnedMesh)for(const b of o.skeleton.bones)bones.add(b)});
 const rest=new Map([...bones].map(b=>[b,b.quaternion.clone()]));
 function slider(name,id,min,max,step,apply,neutral=0){
  const label=document.createElement('label');label.style.cssText='display:block;margin:14px 0';
  const text=document.createElement('span'),output=document.createElement('output');text.textContent=name+' ';output.textContent=String(neutral);
  const input=document.createElement('input');Object.assign(input,{type:'range',id,min,max,step,value:neutral});input.dataset.neutral=String(neutral);input.style.width='100%';
  input.addEventListener('input',()=>{const value=Number(input.value);output.textContent=value.toFixed(2);apply(value);model.updateMatrixWorld(true);if(id==='face-head-turn'||id==='face-jaw'||id.startsWith('face-morph-'))model.userData.grooms?.update()});
  label.append(text,output,input);panel.append(label);controls.push(input);return input;
 }
 const jaw=[...bones].filter(b=>b.name==='FACIAL_C_Jaw');
 const expression=model.userData.expressionController;let expressionSelect=null;
 if(expression){
  const label=document.createElement('label');label.textContent='Expression';label.style.display='block';
  expressionSelect=document.createElement('select');expressionSelect.id='face-expression';expressionSelect.style.cssText='display:block;width:100%;margin:8px 0;padding:6px;background:#394453;color:white;border:0;border-radius:5px';
  for(const name of ['Neutral',...expression.groups]){const option=document.createElement('option');option.value=option.textContent=name;expressionSelect.append(option)}
  label.append(expressionSelect);panel.append(label);
  const input=slider('Expression strength','face-expression-strength',0,1,.01,value=>expression.apply(expressionSelect.value,value));
  expressionSelect.onchange=()=>{input.value=expressionSelect.value==='Neutral'?0:1;input.dispatchEvent(new Event('input'))};
 }
 if(jaw.length&&!expression)slider('Jaw open','face-jaw',0,1,.01,value=>{
  for(const b of jaw)b.quaternion.copy(rest.get(b)).multiply(new THREE.Quaternion().setFromAxisAngle(new THREE.Vector3(1,0,0),value*.25));
 });
 const head=[...bones].filter(b=>b.name==='head');
 if(head.length)slider('Head turn','face-head-turn',-1,1,.01,value=>{
  model.updateMatrixWorld(true);
  for(const b of head){const axis=new THREE.Vector3(0,1,0).applyQuaternion(b.parent.getWorldQuaternion(new THREE.Quaternion()).invert());
   b.quaternion.copy(rest.get(b)).premultiply(new THREE.Quaternion().setFromAxisAngle(axis,value*.5));}
 });
 // Only expose expression morphs that actually exist in the exported asset.
 const morphs=[];model.traverse(o=>{if(o.morphTargetDictionary&&o.morphTargetInfluences)morphs.push(o)});
 const names=[...new Set(morphs.flatMap(o=>Object.keys(o.morphTargetDictionary)))];
 const expressions=new Map();
 for(const name of names.filter(n=>/blink|smile|brow.*raise|mouth_cornerPull|mouth_cornersDown|eye_widen/i.test(n))){
  const semantic=name.split('__').pop();if(!expressions.has(semantic))expressions.set(semantic,[]);expressions.get(semantic).push(name);
 }
 for(const [semantic,targets] of (expression?[]:[...expressions].slice(0,12))){
  const label=semantic.replace(/_/g,' ').replace(/([a-z])([A-Z])/g,'$1 $2');
  slider(label,'face-morph-'+controls.length,0,1,.01,value=>{for(const o of morphs)for(const name of targets){const i=o.morphTargetDictionary[name];if(i!==undefined)o.morphTargetInfluences[i]=value}});
 }
 const skin=new Set();model.traverse(o=>{if(o.isMesh)for(const m of Array.isArray(o.material)?o.material:[o.material])if(m.userData.cmNative?.kind==='SKIN')skin.add(m)});
 if(skin.size){
  const heading=document.createElement('strong');heading.textContent='Skin preview';panel.append(heading);
  slider('Skin roughness','skin-roughness',0,1,.01,value=>{for(const m of skin){m.userData.cmSkinFinish=value;if(m.userData.cmSkinUniform)m.userData.cmSkinUniform.value=value}},.5);
  const normal=new Map([...skin].map(m=>[m,m.normalScale.clone()]));
  slider('Normal detail','skin-normal-detail',0,2,.01,value=>{for(const m of skin)m.normalScale.copy(normal.get(m)).multiplyScalar(value)},1);
 }
 const reset=document.createElement('button');reset.textContent='Reset to neutral';reset.id='face-reset';
 reset.onclick=()=>{if(expressionSelect)expressionSelect.value='Neutral';for(const input of controls){input.value=input.dataset.neutral;input.dispatchEvent(new Event('input'))}};
 reset.disabled=!controls.length;panel.append(reset);
 note.textContent=expression?(expression.calibrated?'Calibrated reference motion preserves the neutral face and differs from native UE animation. Preview one expression at a time.':'Select an expression and adjust its strength. Preview one expression at a time.'):jaw.length?'Native joints and direct corrective previews. Full RigLogic expressions and driven skin maps are pending.':'Facial controls require a rigged export or facial morph targets.';
 document.body.append(panel);return {panel,controls,reset};
}
'''


def write(path):
    path.write_text(JAVASCRIPT,encoding='utf-8')
