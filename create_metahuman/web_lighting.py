"""Portable character lighting studio with local CC0 environments."""
import json
from pathlib import Path
import shutil

ENVIRONMENTS=('studio','courtyard','interior','forest','city','sunrise','sunset','night')

JAVASCRIPT=r'''import * as THREE from 'three';
import {EXRLoader} from 'three/addons/loaders/EXRLoader.js';
import {RectAreaLightUniformsLib} from 'three/addons/lights/RectAreaLightUniformsLib.js';

export async function createLightingStudio({scene,renderer,model}){
 RectAreaLightUniformsLib.init();
 const presets={
  'Neutral studio':{environment:'studio',environmentIntensity:.65,key:3,fill:1.5,rim:2,azimuth:-35,elevation:25,softness:1,temperature:6500,exposure:0,rotation:0},
  'Beauty portrait':{environment:'studio',environmentIntensity:.55,key:5,fill:3,rim:3,azimuth:-25,elevation:30,softness:1.4,temperature:5800,exposure:-.2,rotation:0},
  'Dramatic rim':{environment:'studio',environmentIntensity:.12,key:4,fill:.15,rim:12,azimuth:-65,elevation:20,softness:.65,temperature:6500,exposure:0,rotation:45},
  'Overcast outdoors':{environment:'courtyard',environmentIntensity:1,key:1,fill:1,rim:.5,azimuth:-35,elevation:50,softness:1.8,temperature:7200,exposure:0,rotation:80},
  'Golden hour':{environment:'sunset',environmentIntensity:.7,key:4,fill:.45,rim:6,azimuth:-65,elevation:12,softness:.5,temperature:3600,exposure:-.2,rotation:150},
  'Night portrait':{environment:'night',environmentIntensity:.25,key:2,fill:.25,rim:4,azimuth:-45,elevation:15,softness:.7,temperature:8500,exposure:0,rotation:0}
 };
 const environments=['studio','courtyard','interior','forest','city','sunrise','sunset','night'];
 let state={...presets['Neutral studio'],background:'#20252d',showEnvironment:false,blur:.45,preset:'Neutral studio'},serial=0,current=null;
 const box=new THREE.Box3().setFromObject(model),center=box.getCenter(new THREE.Vector3()),height=Math.max(.1,box.getSize(new THREE.Vector3()).y);
 const target=new THREE.Vector3(center.x,box.min.y+height*.77,center.z);
 const rig=new THREE.Group();rig.name='Character lighting studio';scene.add(rig);
 const lights=['Key','Fill','Rim'].map(name=>{const light=new THREE.RectAreaLight(0xffffff,1,height*.6,height);light.name=name+' softbox';rig.add(light);return light});
 const pmrem=new THREE.PMREMGenerator(renderer);pmrem.compileEquirectangularShader();
 const loader=new EXRLoader();const inputs=new Map(),fields=new Map();
 const panel=document.createElement('details');panel.id='lighting-studio';panel.open=innerWidth>1000;
 panel.style.cssText='position:fixed;left:16px;top:90px;width:260px;max-height:72vh;overflow:auto;padding:14px;background:#171b22ed;border-radius:10px;z-index:2;box-sizing:border-box';
 const title=document.createElement('summary');title.textContent='Lighting studio';title.style.cssText='cursor:pointer;font-weight:700';panel.append(title);
 const message=document.createElement('p');message.setAttribute('role','status');message.style.cssText='font-size:12px;color:#b8c2d0';panel.append(message);
 function label(name){const node=document.createElement('label');node.style.cssText='display:block;margin:12px 0;font-size:12px';node.append(document.createTextNode(name));panel.append(node);return node}
 function select(name,id,values,change){const node=label(name),input=document.createElement('select');input.id=id;input.style.cssText='display:block;width:100%;margin-top:6px;padding:5px;background:#394453;color:white;border:0;border-radius:5px';for(const value of values){const option=document.createElement('option');option.value=option.textContent=value;input.append(option)}input.onchange=()=>change(input.value);node.append(input);return input}
 function slider(name,key,min,max,step){const node=label(name+' '),output=document.createElement('output'),input=document.createElement('input');Object.assign(input,{type:'range',id:'studio-'+key,min,max,step});input.style.cssText='display:block;width:100%';node.append(output,input);fields.set(key,{input,output});input.oninput=()=>setState({[key]:Number(input.value)},true);return input}
 const preset=select('Preset','studio-preset',['Custom',...Object.keys(presets)],value=>{if(value!=='Custom')applyPreset(value)});preset.options[0].disabled=true;
 const environment=select('Environment','studio-environment',environments,value=>setState({environment:value},true));
 slider('Exposure (EV)','exposure',-3,3,.1);slider('Environment strength','environmentIntensity',0,3,.05);slider('Environment rotation','rotation',-180,180,1);
 const advanced=document.createElement('details'),advancedTitle=document.createElement('summary');advancedTitle.textContent='Softboxes and background';advancedTitle.style.cursor='pointer';advanced.append(advancedTitle);panel.append(advanced);
 const before=panel.children.length;
 slider('Key light','key',0,15,.1);slider('Fill light','fill',0,15,.1);slider('Rim light','rim',0,20,.1);slider('Key direction','azimuth',-180,180,1);slider('Key elevation','elevation',-20,80,1);slider('Softbox size','softness',.2,2,.05);slider('Key temperature (K)','temperature',2500,10000,100);
 for(const node of [...panel.children].slice(before))advanced.append(node);
 const backgroundLabel=label('Background '),background=document.createElement('input');background.type='color';background.id='studio-background';background.oninput=()=>setState({background:background.value},true);backgroundLabel.append(background);advanced.append(backgroundLabel);
 const toggleLabel=label(''),toggle=document.createElement('input');toggle.type='checkbox';toggle.id='studio-showEnvironment';toggle.onchange=()=>setState({showEnvironment:toggle.checked},true);toggleLabel.append(toggle,document.createTextNode(' Show environment'));advanced.append(toggleLabel);
 slider('Background blur','blur',0,1,.05);advanced.append(panel.lastElementChild);
 const reset=document.createElement('button');reset.textContent='Reset lighting';reset.id='studio-reset';reset.onclick=()=>applyPreset('Neutral studio');panel.append(reset);
 const download=document.createElement('button');download.textContent='Save lighting';download.id='studio-save';download.style.marginLeft='8px';download.onclick=()=>{const blob=new Blob([JSON.stringify(getState(),null,2)],{type:'application/json'}),url=URL.createObjectURL(blob),a=document.createElement('a');a.href=url;a.download='character-lighting.json';a.click();setTimeout(()=>URL.revokeObjectURL(url),1000)};panel.append(download);
 const restoreLabel=label('Load saved lighting'),restore=document.createElement('input');restore.type='file';restore.accept='.json,application/json';restore.id='studio-load';restore.style.width='100%';restore.onchange=async()=>{try{const file=restore.files[0];if(!file)return;if(file.size>16384)throw Error('Lighting file is too large');const value=JSON.parse(await file.text());if(value.schema!==1||!value.lighting)throw Error('Choose a saved character lighting file');await setState(value.lighting,true)}catch(error){message.textContent=error.message}finally{restore.value=''}};restoreLabel.append(restore);
 document.body.append(panel);
 function temperature(k){const t=k/100;let r=t<=66?255:329.698727446*(t-60)**-.1332047592,g=t<=66?99.4708025861*Math.log(t)-161.1195681661:288.1221695283*(t-60)**-.0755148492,b=t>=66?255:t<=19?0:138.5177312231*Math.log(t-10)-305.0447927307;return new THREE.Color().setRGB(...[r,g,b].map(v=>THREE.MathUtils.clamp(v/255,0,1)),THREE.SRGBColorSpace)}
 function place(light,azimuth,elevation,distance){const a=THREE.MathUtils.degToRad(azimuth),e=THREE.MathUtils.degToRad(elevation);light.position.set(Math.sin(a)*Math.cos(e),Math.sin(e),Math.cos(a)*Math.cos(e)).multiplyScalar(distance).add(target);light.lookAt(target)}
 function update(){
  renderer.toneMappingExposure=2**state.exposure;scene.environmentIntensity=state.environmentIntensity;scene.environmentRotation.y=THREE.MathUtils.degToRad(state.rotation);scene.backgroundRotation.copy(scene.environmentRotation);scene.backgroundBlurriness=state.blur;scene.backgroundIntensity=state.environmentIntensity;
  scene.background=state.showEnvironment&&current?current.source:new THREE.Color(state.background);
  for(let i=0;i<3;i++){lights[i].intensity=state[['key','fill','rim'][i]];lights[i].width=height*.6*state.softness;lights[i].height=height*state.softness}
  lights[0].color.copy(temperature(state.temperature));lights[1].color.set(0xffffff);lights[2].color.set(0xffffff);
  place(lights[0],state.azimuth,state.elevation,height*1.15);place(lights[1],-state.azimuth*.6,15,height*1.3);place(lights[2],state.azimuth+170,30,height*.9);
  environment.value=state.environment;preset.value=state.preset;
  for(const [key,field] of fields){field.input.value=state[key];field.output.textContent=Number(state[key]).toFixed(key==='temperature'||key==='rotation'?0:2)}
  background.value=state.background;toggle.checked=state.showEnvironment;
  if(current)message.textContent=state.preset;
  const legacy=document.querySelector('#light');if(legacy){legacy.min=-3;legacy.max=3;legacy.step=.1;legacy.value=state.exposure;legacy.oninput=()=>setState({exposure:Number(legacy.value)},true);legacy.parentNode.firstChild.textContent='Exposure '}
 }
 async function loadEnvironment(name){
  const ticket=++serial;message.textContent='Loading '+name+' lighting…';
  if(!inputs.has(name)){const promise=loader.loadAsync(new URL('./lighting/'+name+'.exr',import.meta.url)).then(source=>{source.mapping=THREE.EquirectangularReflectionMapping;return {source,filtered:pmrem.fromEquirectangular(source)}}).catch(error=>{inputs.delete(name);throw error});inputs.set(name,promise)}
  try{const result=await inputs.get(name);if(ticket!==serial)return;current=result;scene.environment=result.filtered.texture;update();message.textContent=state.preset==='Custom'?'Custom lighting':state.preset}
  catch(error){if(ticket!==serial)return;message.textContent='Environment unavailable; retained previous lighting. '+error.message;throw error}
 }
 async function setState(values,custom=false){
  const next={...state};const ranges={exposure:[-3,3],environmentIntensity:[0,3],rotation:[-180,180],key:[0,15],fill:[0,15],rim:[0,20],azimuth:[-180,180],elevation:[-20,80],softness:[.2,2],temperature:[2500,10000],blur:[0,1]};
  for(const [key,value] of Object.entries(values)){
   if(key in ranges){if(!Number.isFinite(value)||value<ranges[key][0]||value>ranges[key][1])throw Error('Invalid lighting value: '+key);next[key]=value}
   else if(key==='environment'){if(!environments.includes(value))throw Error('Unknown lighting environment');next[key]=value}
   else if(key==='background'){if(!/^#[a-f0-9]{6}$/i.test(value))throw Error('Choose a background color');next[key]=value}
   else if(key==='showEnvironment'){if(typeof value!=='boolean')throw Error('Invalid environment visibility');next[key]=value}
  }
  const changed=next.environment!==state.environment;next.preset=custom?'Custom':state.preset;state=next;update();if(changed||!current)await loadEnvironment(state.environment);return getState();
 }
 function getState(){return {schema:1,lighting:{...state}}}
 async function applyPreset(name){if(!Object.hasOwn(presets,name))throw Error('Unknown studio preset');state.preset=name;await setState(presets[name]);message.textContent=name;return getState()}
 await applyPreset('Neutral studio');
 return {getState,setState,applyPreset,presets:Object.keys(presets),environments,lights,panel,dispose(){rig.removeFromParent();panel.remove();for(const promise of inputs.values())promise.then(r=>{r.source.dispose();r.filtered.dispose()}).catch(()=>{});pmrem.dispose()}};
}
'''


def write(path,environment_root=None):
    path=Path(path)
    if environment_root is None:
        import bpy
        environment_root=bpy.utils.system_resource('DATAFILES',path='studiolights/world')
    source=Path(environment_root);target=path.parent/'lighting'
    for name in (*[n+'.exr' for n in ENVIRONMENTS],'license.txt'):
        if not (source/name).is_file():raise ValueError('Blender studio environment is missing: '+name)
    target.mkdir(exist_ok=True)
    for name in (*[n+'.exr' for n in ENVIRONMENTS],'license.txt'):shutil.copy2(source/name,target/name)
    (target/'manifest.json').write_text(json.dumps({'environments':list(ENVIRONMENTS),'license':'CC0','creator':'Greg Zaal / Poly Haven','source':'Blender studio environment distribution'},indent=2))
    path.write_text(JAVASCRIPT,encoding='utf-8')


def install_html(page):
    if "from './lighting-studio.js'" in page:return page
    page=page.replace("import { loadCharacter }", "import {createLightingStudio} from './lighting-studio.js';\nimport { loadCharacter }")
    page=page.replace("const model=await loadCharacter(scene);", "const model=await loadCharacter(scene);const studio=await createLightingStudio({scene,renderer,model});")
    page=page.replace('window.characterPreview={model,scene,camera,renderer,controls,','window.characterPreview={model,scene,camera,renderer,controls,studio,')
    page=page.replace("document.querySelector('#light').oninput=e=>{renderer.toneMappingExposure=Number(e.target.value)};",'')
    return page
