"""Browser transport for all glTF skin-weight sets, including MetaHuman's 12."""

JAVASCRIPT=r'''import * as THREE from 'three';
import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js';

function sourceGLB(buffer) {
 const view=new DataView(buffer);
 if(view.byteLength<20||view.getUint32(0,true)!==0x46546c67||view.getUint32(4,true)!==2||view.getUint32(8,true)!==buffer.byteLength)throw Error('Expected a self-contained version-2 GLB');
 let json,binary;for(let offset=12;offset<buffer.byteLength;){const size=view.getUint32(offset,true),type=view.getUint32(offset+4,true);offset+=8;if(offset+size>buffer.byteLength)throw Error('Invalid GLB chunk');
  if(type===0x4e4f534a)json=JSON.parse(new TextDecoder().decode(new Uint8Array(buffer,offset,size)));
  if(type===0x004e4942)binary={offset,size};offset+=size;}
 if(!json||!binary)throw Error('GLB is missing its embedded skin data');
 function attribute(id,joints) {
  const a=json.accessors[id],b=json.bufferViews[a.bufferView];
  if(a.type!=='VEC4'||a.sparse||b.buffer!==undefined&&b.buffer!==0)throw Error('Unsupported skin accessor');
  const bytes={5121:1,5123:2,5126:4}[a.componentType];if(!bytes||joints&&a.componentType===5126)throw Error('Unsupported skin component type');
  const stride=b.byteStride||bytes*4,start=(b.byteOffset||0)+(a.byteOffset||0),count=a.count;
  if(!Number.isInteger(count)||count<1||start<0||start+(count-1)*stride+4*bytes>binary.size)throw Error('Skin accessor exceeds its buffer');
  const result=joints?new Uint16Array(count*4):new Float32Array(count*4);
  const read=a.componentType===5121?'getUint8':a.componentType===5123?'getUint16':'getFloat32';
  const divisor=a.normalized?(a.componentType===5121?255:65535):1;
  for(let i=0;i<count;i++)for(let c=0;c<4;c++)result[i*4+c]=view[read](binary.offset+start+i*stride+c*bytes,true)/divisor;
  return new THREE.BufferAttribute(result,4);
 }
 return {json,attribute};
}

function installMesh(mesh,sets) {
 const count=mesh.geometry.attributes.position.count;
 for(const set of sets)if(set.indices.count!==count||set.weights.count!==count)throw Error('Skin attributes do not match the mesh');
 let extraMax=0;
 for(let i=0;i<count;i++){
  let sum=0,extra=0;
  for(let s=0;s<sets.length;s++)for(let c=0;c<4;c++){
   const w=sets[s].weights.array[i*4+c],j=sets[s].indices.array[i*4+c];
   if(!Number.isFinite(w)||w<0||j>=mesh.skeleton.bones.length)throw Error('Invalid native skin influence');sum+=w;if(s)extra+=w;
  }
  if(sum<=0)throw Error('Unweighted vertex in native skin');extraMax=Math.max(extraMax,extra/sum);
  for(const set of sets)for(let c=0;c<4;c++)set.weights.array[i*4+c]/=sum;
 }
 mesh.geometry.setAttribute('skinIndex',sets[0].indices);mesh.geometry.setAttribute('skinWeight',sets[0].weights);
 for(let s=1;s<sets.length;s++){mesh.geometry.setAttribute('cmSkinIndex'+s,sets[s].indices);mesh.geometry.setAttribute('cmSkinWeight'+s,sets[s].weights);}
 let declarations='',position='',normal='';
 for(let s=1;s<sets.length;s++){
  declarations+=`attribute vec4 cmSkinIndex${s}; attribute vec4 cmSkinWeight${s};\n`;
  for(const c of ['x','y','z','w']){
   position+=`skinned += getBoneMatrix(cmSkinIndex${s}.${c}) * skinVertex * cmSkinWeight${s}.${c};\n`;
   normal+=`skinMatrix += cmSkinWeight${s}.${c} * getBoneMatrix(cmSkinIndex${s}.${c});\n`;
  }
 }
 const materials=Array.isArray(mesh.material)?mesh.material:[mesh.material];
 mesh.material=materials.map(original=>{
  const material=original.clone(),previous=original.onBeforeCompile,key=original.customProgramCacheKey();
  if(original.userData.cmSkinMotion){
   const motion=original.userData.cmSkinMotion;material.userData.cmSkinMotion=motion;motion.retain();
   material.addEventListener('dispose',()=>motion.release());
  }
  material.onBeforeCompile=function(shader,renderer){previous.call(this,shader,renderer);
   const skin=THREE.ShaderChunk.skinning_vertex.replace('transformed =',position+'transformed =');
   const norm=THREE.ShaderChunk.skinnormal_vertex.replace('skinMatrix = bindMatrixInverse',normal+'skinMatrix = bindMatrixInverse');
   if(!skin.includes(position)||!norm.includes(normal))throw Error('Unsupported Three.js skin shader layout');
   shader.vertexShader=declarations+shader.vertexShader.replace('#include <skinning_vertex>',skin).replace('#include <skinnormal_vertex>',norm);
   mesh.userData.cmFullSkinCompiled=sets.length;
  };
  material.customProgramCacheKey=()=>key+'CMFullSkin'+sets.length;return material;
 });
 if(!Array.isArray(mesh.material)||materials.length===1)mesh.material=mesh.material[0];
 const base=new THREE.Vector4(),value=new THREE.Vector4(),matrix=new THREE.Matrix4();
 mesh.applyBoneTransform=function(index,target){
  base.set(target.x,target.y,target.z,target.isVector4?target.w:1).applyMatrix4(this.bindMatrix);
  target.isVector4?target.set(0,0,0,0):target.set(0,0,0);
  for(const set of sets)for(let c=0;c<4;c++){
   const w=set.weights.array[index*4+c];if(!w)continue;const j=set.indices.array[index*4+c];
   matrix.multiplyMatrices(this.skeleton.bones[j].matrixWorld,this.skeleton.boneInverses[j]);
   target.addScaledVector(value.copy(base).applyMatrix4(matrix),w);
  }
  if(target.isVector4)target.w=base.w;return target.applyMatrix4(this.bindMatrixInverse);
 };
 mesh.normalizeSkinWeights=()=>{}; // All sets were normalized together above.
 mesh.userData.cmFullSkin={sets:sets.length,maxInfluences:sets.length*4,extraWeightMax:extraMax};
}

export async function loadNativeCharacter(scene,url,configureMaterial=null) {
 const response=await fetch(url);if(!response.ok)throw Error('Native character download failed: '+response.status);
 const buffer=await response.arrayBuffer(),source=sourceGLB(buffer);
 const loader=new GLTFLoader();
 // Three.js normalizes its first four weights during loading. Give it a copy;
 // restore every set from the immutable source bytes after parsing.
 const gltf=await loader.parseAsync(buffer.slice(0),new URL('.',new URL(url,location.href)).href);
 const pending=[];gltf.scene.traverse(mesh=>{if(mesh.isSkinnedMesh)pending.push(mesh)});
 for(const mesh of pending){
  const association=gltf.parser.associations.get(mesh);
  if(!association||association.meshes===undefined||association.primitives===undefined)throw Error('Native skin primitive could not be identified');
  const attributes=source.json.meshes[association.meshes].primitives[association.primitives].attributes,sets=[];
  for(let i=0;attributes['JOINTS_'+i]!==undefined;i++){
   if(attributes['WEIGHTS_'+i]===undefined)throw Error('Missing native weight set');
   sets.push({indices:source.attribute(attributes['JOINTS_'+i],true),weights:source.attribute(attributes['WEIGHTS_'+i],false)});
  }
  if(configureMaterial)await configureMaterial(mesh,association);
  installMesh(mesh,sets);
 }
 gltf.scene.updateMatrixWorld(true);scene.add(gltf.scene);return gltf;
}
'''


def write(path):
    path.write_text(JAVASCRIPT,encoding='utf-8')
