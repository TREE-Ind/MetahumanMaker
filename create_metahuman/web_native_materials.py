"""Native baked material inputs for the rigged browser renderer.

Preserves the source recipe separately; these initial PBR/eye implementations
do not claim to reproduce Unreal's complete skin and eye shading models.
"""

JAVASCRIPT=r'''import * as THREE from 'three';
export async function nativeMaterialFactory(url) {
 const response=await fetch(url);if(!response.ok)throw Error('Native material recipes are unavailable');
 const recipes=await response.json(),cache=new Map(),textures=new Map(),loader=new THREE.TextureLoader();
 async function texture(record,color=false){
  if(!record)return null;const path=new URL(record.file,url).href,srgb=color&&record.srgb!==false,key=path+':'+srgb;
  if(!textures.has(key))textures.set(key,loader.loadAsync(path).then(t=>{t.flipY=false;t.wrapS=t.wrapT=THREE.RepeatWrapping;t.colorSpace=srgb?THREE.SRGBColorSpace:THREE.NoColorSpace;return t}));
  return textures.get(key);
 }
 async function create(name){
  const recipe=recipes.materials[name];if(!recipe)throw Error('Native material section has no recipe: '+name);
  const maps=recipe.maps,scalars=recipe.scalars;
  const material=new THREE.MeshPhysicalMaterial({color:0xffffff,metalness:0,roughness:.5});material.name=name;
  material.userData.cmNative={kind:recipe.kind,source:recipe.source,approximation:true,unsupported:recipe.unsupported};
  if(recipe.kind==='HIDDEN'){material.visible=false;return material;}
  if(recipe.kind==='SKIN'||recipe.kind==='TEETH'||recipe.kind==='PBR'){
   material.map=await texture(maps.basecolor,true);material.normalMap=await texture(maps.normal);
   material.normalScale.set(1,-1).multiplyScalar(recipe.kind==='PBR'||maps.normal?.native_artist_bake?1:(scalars['Normal Global Strength Post-Bake']??1));
   material.roughnessMap=await texture(maps.packed??maps.roughness);material.metalnessMap=maps.packed?material.roughnessMap:await texture(maps.metallic);
   material.aoMap=await texture(maps.cavity);material.aoMapIntensity=maps.cavity&&recipe.kind==='SKIN'?.35:1;
   material.roughness=recipe.kind==='PBR'||(maps.packed??maps.roughness)?.native_artist_bake?1:(scalars['Roughness Global Multiply Post-Bake']??1);material.metalness=material.metalnessMap?1:0;
   if(maps.emissive){material.emissiveMap=await texture(maps.emissive,true);material.emissive.setHex(0xffffff);material.emissiveIntensity=maps.emissive.emissive_scale??1;if(!Number.isFinite(material.emissiveIntensity)||material.emissiveIntensity<0)throw Error('Native emissive scale is invalid')}
   // Native SRMF is specular R / roughness G / metallic B / fuzz A.
   // Keep specular/scatter resources available for the later SSS shader port.
   material.userData.cmNative.packed=maps.packed;material.userData.cmNative.scatter=maps.scatter;
   if(recipe.kind==='TEETH'||recipe.kind==='PBR'&&maps.specular){
    const specular=await texture(maps.specular);
    const depth=recipe.kind==='TEETH'&&recipe.switches?.['Use Depth Offset']===true?(scalars['Depth Offset']??0):0;
    if(!Number.isFinite(depth))throw Error('Native mouth depth offset is invalid');
    material.userData.cmNative.mouthDepthOffsetCm=depth;
    material.onBeforeCompile=function(shader){
     if(specular){
      shader.uniforms.cmMouthSpecular={value:specular};
      shader.fragmentShader='uniform sampler2D cmMouthSpecular;\n'+shader.fragmentShader;
      shader.fragmentShader=shader.fragmentShader.replace('#include <lights_physical_fragment>',`#include <lights_physical_fragment>
       material.specularColor=mix(vec3(.08*texture2D(cmMouthSpecular,vMapUv).r),diffuseColor.rgb,metalnessFactor);
      `);
     }
     // Native M_teeth_unified uses normalize(WorldPosition-CameraPosition)*
     // Depth Offset as WPO. Evaluate after full skinning, in world metres,
     // then return the vector to local space before Three projects the vertex.
     if(depth!==0){
      const uniform={value:depth*.01};shader.uniforms.cmMouthDepth=uniform;this.userData.cmMouthDepthUniform=uniform;
      shader.vertexShader='uniform float cmMouthDepth;\n'+shader.vertexShader;
      shader.vertexShader=shader.vertexShader.replace('#include <project_vertex>',`
       vec3 cmMouthWorld=(modelMatrix*vec4(transformed,1.0)).xyz;
       vec3 cmMouthRay=cmMouthWorld-cameraPosition;
       vec3 cmMouthOffset=cmMouthRay/max(length(cmMouthRay),.000001)*cmMouthDepth;
       transformed+=(inverse(modelMatrix)*vec4(cmMouthOffset,0.0)).xyz;
       #include <project_vertex>
      `);
     }
     this.userData.cmMouthShaderCompiled=true;
    };
    material.customProgramCacheKey=()=> 'CMNativeMouth2:'+Boolean(specular)+':'+(depth!==0);
   }
   if(recipe.kind==='SKIN'){
    const skinSpecular=await texture(maps.specular??maps.packed);
    const micro=maps.normal?.native_artist_bake&&!maps.normal.micro_details_separate?null:await texture(maps.micro_normal),microMask=micro?await texture(maps.micro_mask):null;
    material.userData.cmSkinFinish=.5;
    material.onBeforeCompile=function(shader){
     const uniform={value:this.userData.cmSkinFinish??.5};this.userData.cmSkinUniform=uniform;
     shader.uniforms.cmSkinFinish=uniform;shader.uniforms.cmSkinPacked={value:skinSpecular};
     shader.fragmentShader='uniform float cmSkinFinish;uniform sampler2D cmSkinPacked;\n'+shader.fragmentShader;
     shader.fragmentShader=shader.fragmentShader.replace('#include <roughnessmap_fragment>',`#include <roughnessmap_fragment>
      float cmGain=pow(.85+.3*cmSkinFinish,5.0);
      float cmMatte=max(0.0,2.0*cmSkinFinish-1.0);
      roughnessFactor=mix(clamp(roughnessFactor*cmGain,0.0,1.0),1.0,cmMatte*cmMatte);
     `);
     if(skinSpecular)shader.fragmentShader=shader.fragmentShader.replace('#include <lights_physical_fragment>',`#include <lights_physical_fragment>
      float cmSpecular=texture2D(cmSkinPacked,vRoughnessMapUv).r;
      material.specularColor=mix(vec3(.08*cmSpecular),diffuseColor.rgb,metalnessFactor);
     `);
     if(micro&&this.normalMap){
      shader.uniforms.cmMicroNormal={value:micro};shader.uniforms.cmMicroMask={value:microMask};
      shader.uniforms.cmMicroStrength={value:scalars['Micro Skin Normal Strength']??.35};
      shader.uniforms.cmMicroTile={value:scalars['Micro Skin Tiling']??40};
      shader.uniforms.cmMicroChannels={value:new THREE.Vector4(...['A','B','C','D'].map(c=>scalars['Micro Mask'+c+' Strength']??1))};
      shader.fragmentShader='uniform sampler2D cmMicroNormal;uniform sampler2D cmMicroMask;uniform float cmMicroStrength;uniform float cmMicroTile;uniform vec4 cmMicroChannels;\n'+shader.fragmentShader;
      const sample='vec3 mapN = texture2D( normalMap, vNormalMapUv ).xyz * 2.0 - 1.0;';
      const mask=microMask?'clamp(dot(texture2D(cmMicroMask,vNormalMapUv),cmMicroChannels),0.0,1.0)':'1.0';
      const blend=sample+`vec3 cmDetail=texture2D(cmMicroNormal,vNormalMapUv*cmMicroTile).xyz*2.0-1.0;
       cmDetail.xy*=cmMicroStrength*${mask};cmDetail=normalize(cmDetail);
       vec3 cmT=mapN+vec3(0,0,1),cmU=cmDetail*vec3(-1,-1,1);
       mapN=normalize(cmT*dot(cmT,cmU)/max(cmT.z,.001)-cmU);`;
      shader.fragmentShader=shader.fragmentShader.replace('#include <normal_fragment_maps>',THREE.ShaderChunk.normal_fragment_maps.replace(sample,blend));
     }
    };
    material.customProgramCacheKey=()=> 'CMNativeSkin3:'+Boolean(skinSpecular)+':'+Boolean(micro)+':'+Boolean(microMask);
   }
  }else if(recipe.kind==='EYE'){
   material.map=await texture(maps.sclera,true);material.normalMap=await texture(maps.sclera_normal);
   material.normalScale.set(1,-1);material.roughness=scalars['Cornea Roughness']??.075;
   material.ior=1.376;material.clearcoat=1;material.clearcoatRoughness=material.roughness;
   const iris=await texture(maps.iris,true),irisNormal=await texture(maps.iris_normal);
   const radius=Math.max(.01,scalars['Cornea Size']??.165);
   material.onBeforeCompile=shader=>{
    shader.uniforms.cmIris={value:iris};shader.uniforms.cmIrisNormal={value:irisNormal};shader.uniforms.cmIrisRadius={value:radius};
    shader.fragmentShader='uniform sampler2D cmIris;uniform sampler2D cmIrisNormal;uniform float cmIrisRadius;\n'+shader.fragmentShader;
    shader.fragmentShader=shader.fragmentShader.replace('#include <map_fragment>',`
     vec2 cmIrisUV=(vMapUv-vec2(.5))/(2.0*cmIrisRadius)+vec2(.5);
     float cmIrisMask=1.0-smoothstep(cmIrisRadius-.008,cmIrisRadius+.008,length(vMapUv-vec2(.5)));
     diffuseColor*=mix(texture2D(map,vMapUv),texture2D(cmIris,cmIrisUV),cmIrisMask);
    `);
    shader.fragmentShader=shader.fragmentShader.replace('#include <normal_fragment_maps>',THREE.ShaderChunk.normal_fragment_maps.replace(
     'vec3 mapN = texture2D( normalMap, vNormalMapUv ).xyz * 2.0 - 1.0;',
     'vec3 mapN = mix(texture2D(normalMap,vNormalMapUv).xyz,texture2D(cmIrisNormal,cmIrisUV).xyz,cmIrisMask)*2.0-1.0;'));
   };
   material.customProgramCacheKey=()=> 'CMNativeEye1';
  }else if(recipe.kind==='LASH'){
   material.color.setHex(0x17120f);material.alphaMap=await texture(maps.opacity);material.alphaTest=.35;material.side=THREE.DoubleSide;material.roughness=.6;
  }else if(recipe.kind==='EYE_SHELL'){
   material.color.setHex(0x241b17);material.transparent=true;material.opacity=.12;material.depthWrite=false;material.roughness=1;
  }else if(recipe.kind==='TEARLINE'){
   material.color.setHex(0xffffff);material.transparent=true;material.opacity=.22;material.depthWrite=false;material.roughness=.03;
  }else{throw Error('No browser implementation for native material kind: '+recipe.kind);}
  if(recipe.surface==='MASKED'){
   material.alphaMap=await texture(maps.opacity);material.alphaTest=recipe.alpha_test;
   // UE property-bake RGB is valid even when the exported basecolor alpha is
   // zero. Coverage comes exclusively from the separately evaluated mask.
   const previous=material.onBeforeCompile,key=material.customProgramCacheKey.bind(material);
   material.onBeforeCompile=function(shader,...args){
    previous.call(this,shader,...args);
    shader.fragmentShader=shader.fragmentShader.replace('#include <map_fragment>',THREE.ShaderChunk.map_fragment.replace('diffuseColor *= sampledDiffuseColor;','diffuseColor.rgb *= sampledDiffuseColor.rgb;'));
   };
   material.customProgramCacheKey=()=>key()+':CMNativeMask1';
  }
  if(recipe.animated_skin){const {attachSkinMotion}=await import('./native-skin-motion.js');await attachSkinMotion(material,recipe.animated_skin,url);}
  return material;
 }
 return async mesh=>{
  const original=Array.isArray(mesh.material)?mesh.material:[mesh.material],result=[];
  for(const m of original){if(!cache.has(m.name))cache.set(m.name,create(m.name));result.push(await cache.get(m.name));}
  mesh.material=Array.isArray(mesh.material)?result:result[0];
 };
}
'''


def write(path):
    path.write_text(JAVASCRIPT,encoding='utf-8')
