"""Keep source texture resolution in an isolated native assembly pipeline."""
from pathlib import Path
import re


def configure(character,params,profile,preserve=True):
    import unreal
    settings_class=unreal.load_class(None,'/Script/MetaHumanCharacterPalette.MetaHumanCharacterPaletteProjectSettings')
    if not settings_class:raise ValueError('MetaHuman pipeline settings are unavailable')
    settings=unreal.get_default_object(settings_class)
    pipelines=dict(settings.get_editor_property('DefaultCharacterLegacyPipelines'))
    quality=getattr(unreal.MetaHumanQualityLevel,profile)
    cls=pipelines.get(quality)
    if not cls:raise ValueError('The configured native '+profile.lower()+' pipeline is missing')
    if not isinstance(cls,unreal.Class):cls=unreal.load_class(None,str(cls))
    if not cls:raise ValueError('Could not load the configured assembly pipeline')
    original=unreal.get_default_object(cls).get_editor_property('EditorPipeline')
    if not preserve:return None,dict(policy='NATIVE_PRESET',pipeline_class=cls.get_path_name(),material_baking_required=bool(original.get_editor_property('bBakeMaterials')))
    before=original.get_editor_property('MaterialBakingOptions').export_text()
    before_caps=original.get_editor_property('MaxTextureResolutions').export_text()
    pipeline=unreal.new_object(cls)
    editor=pipeline.get_editor_property('EditorPipeline')
    if not editor or editor==original:raise ValueError('Native assembly did not create private pipeline settings')
    options=editor.get_editor_property('MaterialBakingOptions')
    caps=editor.get_editor_property('MaxTextureResolutions')
    baking_text=options.export_text();caps_text=caps.export_text()
    skin=character.get_editor_property('SkinSettings');overrides=skin.get_editor_property('TextureMaterialOverrides')
    texture_overrides=overrides.get_editor_property('TextureOverrides')
    enabled=bool(overrides.get_editor_property('bEnableTextureOverrides'))
    face_info=character.get_editor_property('SynthesizedFaceTexturesInfo')
    body_info=character.get_editor_property('HighResBodyTexturesInfo')
    def dimension(info):
        return max(int(info.get_editor_property('SizeX')),int(info.get_editor_property('SizeY'))) if info else 0
    def texture_size(texture):
        if not texture:return 0
        data=texture.get_editor_property('asset_import_data')
        for filename in data.extract_filenames() if data else []:
            path=Path(filename)
            if not path.is_file() or path.suffix.lower()!='.png':continue
            with path.open('rb') as stream:header=stream.read(24)
            if header[:8]==b'\x89PNG\r\n\x1a\n':return max(int.from_bytes(header[i:i+4],'big') for i in (16,20))
        size=max(texture.blueprint_get_size_x(),texture.blueprint_get_size_y())
        # Native skin instances also expose valid small constant textures
        # (e.g. the 32px flat-gray inherited Basecolor placeholder). They do
        # not mean compilation is pending. Real override sizes are included
        # separately above; an uninitialized texture has no dimensions.
        if size<=0:raise ValueError('A native source texture is still being prepared; retry assembly after texture preparation')
        return size
    face_values=dict(texture_overrides.get_editor_property('Face')) if enabled else {}
    body_values=dict(texture_overrides.get_editor_property('Body')) if enabled else {}
    head_size=max(dimension(face_info.get(unreal.FaceTextureType.BASECOLOR)),texture_size(face_values.get(unreal.FaceTextureType.BASECOLOR)))
    body_size=max(dimension(body_info.get(unreal.BodyTextureType.BODY_BASECOLOR)),texture_size(body_values.get(unreal.BodyTextureType.BODY_BASECOLOR)))
    # Inheritance-disabled custom skin instances can provide the effective
    # albedo without a native texture override. Include their actual inputs.
    if overrides.get_editor_property('bEnableMaterialOverrides'):
        materials=overrides.get_editor_property('MaterialOverrides')
        candidates=list(dict(materials.get_editor_property('Skin')).values())
        for material in candidates:
            if material:head_size=max(head_size,texture_size(unreal.MaterialEditingLibrary.get_material_instance_texture_parameter_value(material,'Basecolor')))
        material=materials.get_editor_property('Body')
        if material:body_size=max(body_size,texture_size(unreal.MaterialEditingLibrary.get_material_instance_texture_parameter_value(material,'Basecolor')))
    changes=[]
    def resolution(size):
        if size>8192:raise ValueError('Preserve Source supports native texture inputs through 8K')
        size=next((n for n in (128,256,512,1024,2048,4096,8192) if n>=size),8192)
        return size
    # These native settings are not BlueprintType structs. Their generic
    # Python nested views alias the parent; change only exact exported pairs
    # and let Unreal parse/validate the complete private settings value.
    for label,name,size in [('HEAD','T_Head_BC',head_size),('BODY','T_Body_BC',body_size)]:
        if not size:continue
        pattern=r'(\("'+re.escape(name)+r'",\s*Res)(\d+)(\))'
        matches=list(re.finditer(pattern,baking_text))
        if len(matches)!=1:raise ValueError('Native material output needs a resolution adapter: '+name)
        old=int(matches[0].group(2));new=max(old,resolution(size))
        if new>old:
            baking_text=re.sub(pattern,lambda m:m.group(1)+str(new)+m.group(3),baking_text)
            changes.append(dict(scope=label,texture=name,before='Res'+str(old),after='Res'+str(new)))
    for name,old_text in re.findall(r'\(([A-Za-z0-9_]+),\s*Res(\d+)\)',caps_text):
        key=getattr(unreal.FaceTextureType,name.upper(),None)
        if key is None:continue
        size=max(dimension(face_info.get(key)),texture_size(face_values.get(key)))
        if not size:continue
        old=int(old_text);new=max(old,resolution(size))
        if new>old:
            caps_text=re.sub(r'(\('+re.escape(name)+r',\s*Res)'+old_text+r'(\))',lambda m:m.group(1)+str(new)+m.group(2),caps_text)
            changes.append(dict(scope='SOURCE_FACE',texture=name,before='Res'+old_text,after='Res'+str(new)))
    options.import_text(baking_text);caps.import_text(caps_text)
    editor.set_editor_property('MaterialBakingOptions',options);editor.set_editor_property('MaxTextureResolutions',caps)
    if options.export_text()!=baking_text or caps.export_text()!=caps_text:raise RuntimeError('Native assembly settings did not retain source resolutions')
    if (original.get_editor_property('MaterialBakingOptions').export_text()!=before or
            original.get_editor_property('MaxTextureResolutions').export_text()!=before_caps):raise RuntimeError('An original native pipeline preset changed')
    return pipeline,dict(policy='PRESERVE_SOURCE',pipeline_class=cls.get_path_name(),private_pipeline=pipeline.get_path_name(),
        material_baking_required=bool(editor.get_editor_property('bBakeMaterials')),source_albedo_sizes=dict(HEAD=head_size,BODY=body_size),
        changes=changes,original_preset_preserved=True,configured_baking=options.export_text(),configured_source_limits=caps.export_text())
