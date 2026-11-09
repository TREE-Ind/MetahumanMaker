"""Sampled native material response for the browser's facial sample controller."""
JAVASCRIPT=r'''import * as THREE from 'three';
export async function attachSkinMotion(material,descriptor,url){
 if(descriptor.generator!=='CreateMetaHuman.PortableSkinMotion'||!descriptor.neutral_preserved)throw Error('Unsupported native skin response');
 const {size,columns,samples}=descriptor,count=samples.length,layers=descriptor.layer_count??count;
 if(!Number.isInteger(size)||size<1||count<1||count>128||!Number.isInteger(columns)||columns<1)throw Error('Invalid animated skin layout');
 const records=new Map(samples.map(s=>[s.pose,s]));if(records.size!==count)throw Error('Duplicate animated skin sample');
 if(!Number.isInteger(layers)||layers<1||layers>count||samples.some(s=>!Number.isInteger(s.layer)||s.layer<0||s.layer>=layers||!Number.isFinite(s.color_scale)||s.color_scale<=0||!Number.isFinite(s.normal_scale)||s.normal_scale<=0))throw Error('Invalid animated skin samples');
 async function arrayTexture(record,red=false){
  const response=await fetch(new URL(record.file,url));if(!response.ok)throw Error('Animated skin atlas unavailable');
  const bitmap=await createImageBitmap(await response.blob(),{premultiplyAlpha:'none',colorSpaceConversion:'none'});
  if(bitmap.width!==record.size[0]||bitmap.height!==record.size[1])throw Error('Animated skin atlas dimensions changed');
  const canvas=document.createElement('canvas');canvas.width=bitmap.width;canvas.height=bitmap.height;
  const context=canvas.getContext('2d',{willReadFrequently:true});context.drawImage(bitmap,0,0);bitmap.close();
  const source=context.getImageData(0,0,canvas.width,canvas.height).data,channels=red?1:4,pixels=new Uint8Array(size*size*layers*channels),width=canvas.width;
  for(let layer=0;layer<layers;layer++){
   const left=(layer%columns)*size,top=Math.floor(layer/columns)*size;
   for(let y=0;y<size;y++){
    const start=((top+y)*width+left)*4,destination=(layer*size*size+y*size)*channels;
    if(red){for(let x=0;x<size;x++)pixels[destination+x]=source[start+x*4];}
    else pixels.set(source.subarray(start,start+size*4),destination);
   }
  }
  canvas.width=canvas.height=1;
  const texture=new THREE.DataArrayTexture(pixels,size,size,layers);texture.format=red?THREE.RedFormat:THREE.RGBAFormat;
  texture.type=THREE.UnsignedByteType;texture.colorSpace=THREE.NoColorSpace;texture.magFilter=THREE.LinearFilter;texture.minFilter=THREE.LinearMipmapLinearFilter;texture.generateMipmaps=true;
  texture.wrapS=texture.wrapT=THREE.ClampToEdgeWrapping;texture.needsUpdate=true;return texture;
 }
 // Load sequentially to bound decoded-atlas memory during character startup.
 const color=await arrayTexture(descriptor.maps.basecolor),normal=await arrayTexture(descriptor.maps.normal),roughness=await arrayTexture(descriptor.maps.roughness,true);
 const active=Array.from({length:count},()=>new THREE.Vector4()),uniformCount={value:0};let lastWeights={},references=0,disposed=false;
 const state={samples:count,method:descriptor.method,retain(){references++},release(){if(references>0)references--;this.dispose()},dispose(){if(references||disposed)return;disposed=true;color.dispose();normal.dispose();roughness.dispose()},setWeights(weights={}){
  if(Object.entries(weights).some(([name,value])=>!records.has(name)||!Number.isFinite(value)||value<0))throw Error('Invalid native skin weights');
  const entries=Object.entries(weights).filter(([,value])=>value>1e-6);
  if(entries.length>count)throw Error('Invalid native skin weights');
  entries.forEach(([name,value],index)=>{const s=records.get(name);active[index].set(s.layer,value,s.color_scale,s.normal_scale);});
  uniformCount.value=entries.length;lastWeights=Object.fromEntries(entries);
 },get weights(){return {...lastWeights}}};material.userData.cmSkinMotion=state;
 const previous=material.onBeforeCompile,key=material.customProgramCacheKey.bind(material);
 material.onBeforeCompile=function(shader,...args){
  previous.call(this,shader,...args);
  Object.assign(shader.uniforms,{cmMotionColor:{value:color},cmMotionNormal:{value:normal},cmMotionRoughness:{value:roughness},cmMotionActive:{value:active},cmMotionCount:uniformCount});
  shader.fragmentShader=`uniform highp sampler2DArray cmMotionColor,cmMotionNormal,cmMotionRoughness;
   uniform vec4 cmMotionActive[${count}];uniform int cmMotionCount;
   void cmEvaluateSkin(vec2 uv,out vec3 colorDelta,out vec3 normalDelta,out float roughDelta){
    colorDelta=vec3(0);normalDelta=vec3(0);roughDelta=0.0;
    for(int i=0;i<${count};i++){
     if(i>=cmMotionCount)break;vec4 p=cmMotionActive[i];vec3 address=vec3(uv,p.x);
     colorDelta+=(texture(cmMotionColor,address).rgb*255.0-128.0)/127.0*p.z*p.y;
     normalDelta+=(texture(cmMotionNormal,address).rgb*255.0-128.0)/127.0*p.w*p.y;
     roughDelta+=(texture(cmMotionRoughness,address).r*255.0-128.0)/127.0*p.w*p.y;
    }
   }\n`+shader.fragmentShader;
  shader.fragmentShader=shader.fragmentShader.replace('#include <color_fragment>',`vec3 cmColorDelta,cmNormalDelta;float cmRoughDelta;cmEvaluateSkin(vMapUv,cmColorDelta,cmNormalDelta,cmRoughDelta);\n#include <color_fragment>`);
  shader.fragmentShader=shader.fragmentShader.replace('#include <alphamap_fragment>',`diffuseColor.rgb=clamp(diffuseColor.rgb+cmColorDelta,0.0,1.0);\n#include <alphamap_fragment>`);
  shader.fragmentShader=shader.fragmentShader.replace('#include <roughnessmap_fragment>',`#include <roughnessmap_fragment>\nroughnessFactor=clamp(roughnessFactor+cmRoughDelta,0.0,1.0);`);
  shader.fragmentShader=shader.fragmentShader.replace('#include <normal_fragment_maps>',THREE.ShaderChunk.normal_fragment_maps);
  const normalSample='vec3 mapN = texture2D( normalMap, vNormalMapUv ).xyz * 2.0 - 1.0;';
  if(!shader.fragmentShader.includes(normalSample))throw Error('Native skin normal shader changed');
  shader.fragmentShader=shader.fragmentShader.replace(normalSample,normalSample+'mapN=normalize(mapN+cmNormalDelta);');
  this.userData.cmSkinMotionCompiled=true;
 };
 material.customProgramCacheKey=()=>key()+':CMNativeSkinMotion1:'+count;
 material.addEventListener('dispose',()=>state.dispose());
 return state;
}
'''

def write(path):path.write_text(JAVASCRIPT,encoding='utf-8')
