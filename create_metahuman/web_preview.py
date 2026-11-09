"""Small static Three.js viewer delivered alongside a character GLB."""

HTML='''<!doctype html>
<html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Character preview</title>
<link rel="icon" href="data:,">
<style>
body{margin:0;background:#20252d;color:#eee;font:14px system-ui}canvas{display:block}
nav{position:fixed;top:16px;left:16px;display:flex;gap:10px;align-items:center;flex-wrap:wrap;background:#171b22df;padding:12px;border-radius:10px}
button,a{background:#394453;color:white;border:0;border-radius:6px;padding:8px 12px;cursor:pointer;text-decoration:none}
footer{position:fixed;bottom:16px;left:16px;max-width:85vw;background:#171b22df;padding:10px;border-radius:8px}
</style>
<nav><strong id="name">Character</strong><button id="body" disabled>Full character</button><button id="face" disabled>Face</button>
<label>Light <input id="light" type="range" min="0.3" max="2" value="1" step="0.05"></label><a href="__MODEL__" download>Download GLB</a></nav>
<footer id="status">Loading character…</footer>
<script type="importmap">{"imports":{"three":"https://cdn.jsdelivr.net/npm/three@0.186.1/build/three.module.js","three/addons/":"https://cdn.jsdelivr.net/npm/three@0.186.1/examples/jsm/"}}</script>
<script type="module">
import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { RoomEnvironment } from 'three/addons/environments/RoomEnvironment.js';
import { loadCharacter } from './load-character.js';
const status=document.querySelector('#status');
try {
 const renderer=new THREE.WebGLRenderer({antialias:true,preserveDrawingBuffer:true});
 renderer.setPixelRatio(Math.min(devicePixelRatio,2));renderer.setSize(innerWidth,innerHeight);
 renderer.toneMapping=THREE.ACESFilmicToneMapping;document.body.append(renderer.domElement);
 const scene=new THREE.Scene();scene.background=new THREE.Color('#20252d');
 const pmrem=new THREE.PMREMGenerator(renderer),room=new RoomEnvironment();
 const environment=pmrem.fromScene(room);scene.environment=environment.texture;room.dispose();pmrem.dispose();
 const camera=new THREE.PerspectiveCamera(35,innerWidth/innerHeight,.001,1000);
 const controls=new OrbitControls(camera,renderer.domElement);controls.enableDamping=true;
 const model=await loadCharacter(scene);const full=new THREE.Box3().setFromObject(model);
 const head=model.getObjectByName('CM_Web_Head');
 function frame(box){const center=box.getCenter(new THREE.Vector3()),size=box.getSize(new THREE.Vector3());
  const span=Math.max(size.y,size.x/camera.aspect,size.z*.6);
  const distance=span*.62/Math.tan(THREE.MathUtils.degToRad(camera.fov/2));
  controls.target.copy(center);camera.position.copy(center).add(new THREE.Vector3(0,0,distance));
  camera.near=Math.max(.001,distance/1000);camera.far=Math.max(100,distance*100);camera.updateProjectionMatrix();controls.update();}
 const body=()=>frame(full),face=()=>frame(head?new THREE.Box3().setFromObject(head):full);
 body();document.querySelector('#body').disabled=false;document.querySelector('#face').disabled=!head;
 document.querySelector('#body').onclick=body;document.querySelector('#face').onclick=face;
 document.querySelector('#light').oninput=e=>{renderer.toneMappingExposure=Number(e.target.value)};
 const manifest=await fetch('./manifest.json').then(r=>{if(!r.ok)throw Error('Manifest unavailable');return r.json()});
 document.querySelector('#name').textContent=manifest.name;
 status.textContent='Drag to orbit · Right-drag to pan · Scroll to zoom · Static preview; animation and groom strands are separate.';
 window.addEventListener('resize',()=>{camera.aspect=innerWidth/innerHeight;camera.updateProjectionMatrix();renderer.setSize(innerWidth,innerHeight)});
 renderer.setAnimationLoop(()=>{controls.update();renderer.render(scene,camera)});
 // A bounded inspection hook also makes this preview usable by browser agents.
 window.characterPreview={model,scene,camera,renderer,controls,frameBody:body,frameFace:face,ready:true};
}catch(error){status.textContent='Preview could not load. Serve this folder over HTTP and check the network connection. '+error.message;console.error(error);window.characterPreview={ready:false,error:error.message}}
</script></html>'''


def write(path,model):
    from . import web_lighting
    web_lighting.write(path.with_name('lighting-studio.js'))
    path.write_text(web_lighting.install_html(HTML.replace('__MODEL__',model)),encoding='utf-8')
