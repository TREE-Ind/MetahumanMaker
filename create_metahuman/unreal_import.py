"""Run inside UE 5.8. Standalone copy is included with each export bundle."""

import json
from pathlib import Path
import re
import math
import struct


def audit_head_conform(character, vertices, output_file):
    """Measure the native face against supplied canonical vertices, in UE cm.

    Older bridges report an unavailable measurement, never a preservation claim.
    The tolerance allows 32 float32 ULPs at the supplied coordinate scale.
    This checks evaluated rest vertices, not rig motion or source-fit quality.
    """
    import unreal
    library=getattr(unreal,'CreateMHCharacterLibrary',None)
    export=getattr(library,'export_face_state',None)
    if export is None:return dict(geometry_preservation_checked=False,reason='Native face-state measurement requires bridge 0.18 or later')
    output_file=Path(output_file)
    if not export(character,str(output_file)):raise RuntimeError('Native head geometry measurement failed')
    metadata=json.loads(output_file.read_text(encoding='utf-8-sig'))
    if metadata.get('schema')!='CM_FACE_STATE_V1' or metadata.get('coordinate_system')!='UE_CM' or metadata.get('dtype')!='<f4':raise ValueError('Unsupported native face-state snapshot')
    binary=output_file.parent/metadata['vertices_file']
    if binary.parent.resolve()!=output_file.parent.resolve():raise ValueError('Invalid native snapshot buffer path')
    raw=binary.read_bytes()
    if len(raw)!=metadata['bytes']:raise ValueError('Incomplete native snapshot buffer')
    roles={'head_lod0_mesh':'head','eyeLeft_lod0_mesh':'eye_left','eyeRight_lod0_mesh':'eye_right','teeth_lod0_mesh':'teeth'}
    result=dict(geometry_preservation_checked=True,within_native_precision=True,snapshot=str(output_file),parts={},
        tolerance_policy='32_FLOAT32_ULPS_AT_TARGET_COORDINATE_SCALE',high_frequency_delta_scale=metadata['high_frequency_delta_scale'])
    for mesh in metadata['meshes']:
        role=roles.get(mesh['name'])
        if role not in vertices:continue
        target=vertices[role];count=len(target)
        if mesh['vertex_count']!=count or not count:raise ValueError('Native vertex count changed for '+role)
        offset=mesh['byte_offset'];end=offset+count*12
        if offset<0 or end>len(raw):raise ValueError('Invalid native vertex buffer range')
        actual=struct.iter_unpack('<fff',raw[offset:end]);errors=[]
        scale=max(abs(v) for point in target for v in point)
        tolerance=max(1e-5,math.ldexp(1.,math.frexp(scale)[1]-24)*32.)
        for point,wanted in zip(actual,target):
            error=math.dist(point,wanted)
            if not math.isfinite(error):raise ValueError('Nonfinite native geometry measurement')
            errors.append(error)
        maximum=max(errors);passed=maximum<=tolerance
        result['parts'][role]=dict(vertices=count,rms_mm=math.sqrt(math.fsum(e*e for e in errors)/count)*10.,max_mm=maximum*10.,tolerance_mm=tolerance*10.,within_native_precision=passed)
        result['within_native_precision']&=passed
    if set(result['parts'])!=set(vertices):raise ValueError('Native snapshot is missing supplied head parts')
    return result


IRIS_OPTION_FIELDS = {
    'primary_color_hue': ('primary_color_u', 0., 1.),
    'secondary_color_hue': ('secondary_color_u', 0., 1.),
    'secondary_color_value': ('secondary_color_v', 0., 1.),
    'iris_color_blend': ('color_blend', 0., 1.),
    'iris_blend_softness': ('color_blend_softness', 0., 1.),
    'iris_rotation': ('rotation', 0., 1.),
    'iris_shadow_details': ('shadow_details', 0., 1.),
    'limbal_ring_size': ('limbal_ring_size', .6, .85),
    'limbal_ring_softness': ('limbal_ring_softness', .02, .15),
}


def eye_options(value):
    if (not isinstance(value, dict) or not {'global_tint','global_saturation','cornea_size'}<=set(value)
            or set(value)-({'global_tint','global_saturation','cornea_size','pupil_dilation','primary_color_value'} | set(IRIS_OPTION_FIELDS))):
        raise ValueError('Eye appearance needs tint, saturation and iris size')
    tint=value['global_tint'];saturation=value['global_saturation'];size=value['cornea_size']
    if (not isinstance(tint,(list,tuple)) or len(tint)!=3
            or any(type(v) not in (int,float) or not math.isfinite(v) or not 0<=v<=1 for v in tint)
            or type(saturation) not in (int,float) or not math.isfinite(saturation) or not 0<=saturation<=4
            or type(size) not in (int,float) or not math.isfinite(size) or not .145-1e-6<=size<=.185+1e-6):
        raise ValueError('Eye appearance contains an invalid color, saturation or iris size')
    result=dict(global_tint=list(tint),global_saturation=float(saturation),cornea_size=max(.145,min(.185,float(size))))
    if 'pupil_dilation' in value:
        pupil=value['pupil_dilation']
        if type(pupil) not in (int,float) or not math.isfinite(pupil) or not .85-1e-6<=pupil<=1.2+1e-6:
            raise ValueError('Choose native pupil dilation between 0.85 and 1.2')
        result['pupil_dilation']=max(.85,min(1.2,float(pupil)))
    if 'primary_color_value' in value:
        lightness=value['primary_color_value']
        if type(lightness) not in (int,float) or not math.isfinite(lightness) or not 0<=lightness<=1:
            raise ValueError('Choose iris lightness between zero and one')
        result['primary_color_value']=float(lightness)
    for name, (_, lower, upper) in IRIS_OPTION_FIELDS.items():
        if name not in value:continue
        number=value[name]
        if type(number) not in (int,float) or not math.isfinite(number) or not lower-1e-6<=number<=upper+1e-6:
            raise ValueError(f'Choose {name.replace("_", " ")} between {lower:g} and {upper:g}')
        result[name]=max(lower,min(upper,float(number)))
    return result


def apply_eye_settings(character, subsystem, value):
    import unreal
    value=eye_options(value)
    original=character.get_editor_property('eyes_settings').copy()
    settings=original.copy()
    for name in ('eye_left','eye_right'):
        eye=settings.get_editor_property(name)
        iris=eye.get_editor_property('iris');cornea=eye.get_editor_property('cornea')
        iris.set_editor_property('global_tint',unreal.LinearColor(*value['global_tint'],1))
        iris.set_editor_property('global_saturation',value['global_saturation'])
        if 'primary_color_value' in value:iris.set_editor_property('primary_color_v',value['primary_color_value'])
        for key,(property_name,_,_) in IRIS_OPTION_FIELDS.items():
            if key in value:iris.set_editor_property(property_name,value[key])
        cornea.set_editor_property('size',value['cornea_size'])
        eye.set_editor_property('iris',iris);eye.set_editor_property('cornea',cornea)
        if 'pupil_dilation' in value:
            pupil=eye.get_editor_property('pupil');pupil.set_editor_property('dilation',value['pupil_dilation'])
            eye.set_editor_property('pupil',pupil)
        settings.set_editor_property(name,eye)
    try:
        subsystem.commit_eyes_settings(character,settings)
        if not unreal.EditorAssetLibrary.save_loaded_asset(character,False):
            raise RuntimeError('Could not save Creator eye appearance')
    except Exception:
        subsystem.commit_eyes_settings(character,original)
        unreal.EditorAssetLibrary.save_loaded_asset(character,False)
        raise
    return value


def _path(root, relative):
    result = (root / relative).resolve()
    if not result.is_relative_to(root.resolve()) or not result.is_file():
        raise ValueError("Bundle contains a missing or out-of-folder file: " + relative)
    return result


def _accessory_vertices(manifest):
    """Canonical fitting uses explicit arrays rather than the imported FBX order."""
    result = {}
    if not manifest.get('create_character') or not manifest.get('canonical_order') or manifest['part'] == 'BODY':
        return result
    for role in manifest.get('accessories', {}):
        if role not in {'eye_left', 'eye_right', 'teeth'}:
            raise ValueError('Unknown head accessory: ' + role)
        points = manifest.get('accessory_vertices', {}).get(role)
        if not points:
            raise ValueError('Canonical head delivery needs explicit ' + role + ' vertices. Re-export this character with the current add-on.')
        if any(not isinstance(p, (list, tuple)) or len(p) != 3 or
               any(not isinstance(v, (int, float)) or not math.isfinite(v) for v in p) for p in points):
            raise ValueError('Invalid canonical accessory coordinates: ' + role)
        result[role] = points
    return result


def _mapped_conform(manifest):
    value=manifest.get('canonical_conform')
    if value is None:
        if manifest.get('create_character') and not manifest.get('canonical_order') and manifest.get('accessories'):
            raise ValueError('Complete-head UV delivery needs explicit native correspondence. Re-export this character with the current add-on to preserve eyes and teeth.')
        return None
    if not isinstance(value,dict) or value.get('method')!='LOCAL_UV_CORRESPONDENCE':
        raise ValueError('Unsupported canonical conform correspondence; re-export the character')
    vertices=value.get('vertices',{})
    if not isinstance(vertices,dict) or set(vertices)!={'head',*manifest.get('accessories',{})}:
        raise ValueError('Canonical conform anatomy does not match the delivered parts')
    for role,points in vertices.items():
        if not isinstance(points,list) or not points or any(not isinstance(p,list) or len(p)!=3 or
                any(type(v) not in (int,float) or not math.isfinite(v) for v in p) for p in points):
            raise ValueError('Invalid canonical conform vertices: '+role)
    return vertices


def repair_import_sources(asset_paths, snapshot_root, source_root):
    """Keep imported mesh/texture Reimport paths valid after snapshot cleanup."""
    import unreal
    snapshot_root, source_root = Path(snapshot_root).resolve(), Path(source_root).resolve()
    repaired, warnings = [], []
    for path in asset_paths:
        asset = unreal.load_asset(path)
        if not isinstance(asset, (unreal.StaticMesh, unreal.SkeletalMesh, unreal.Texture)):
            continue
        try:
            data = asset.get_editor_property('asset_import_data')
            if not data:
                raise ValueError('Imported asset has no source-file metadata')
            changed = []
            filenames = list(data.extract_filenames())
            if not filenames:
                raise ValueError('Imported asset has no source filename')
            for index, filename in enumerate(filenames):
                filename = Path(filename).resolve()
                if not filename.is_relative_to(snapshot_root):
                    continue
                relative = filename.relative_to(snapshot_root)
                source = _path(source_root, relative.as_posix())
                data.scripted_add_filename(str(source), index, '')
                changed.append(str(source))
            if changed:
                if not unreal.EditorAssetLibrary.save_loaded_asset(asset, False):
                    raise RuntimeError('Could not save repaired source-file metadata')
                repaired.append({'asset': str(path), 'sources': changed})
        except Exception as exc:
            warnings.append({'asset': str(path), 'error': str(exc)})
    return {'repaired': repaired, 'warnings': warnings}


def import_bundle(manifest_path):
    import unreal

    manifest_path = Path(manifest_path).resolve()
    root = manifest_path.parent
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != 1:
        raise ValueError("Unsupported MetaHuman Maker bundle schema")
    canonical_accessories = _accessory_vertices(manifest)
    if 'native_eyes' in manifest:eye_options(manifest['native_eyes'])
    mapped_conform=_mapped_conform(manifest)
    name = manifest["name"]
    destination = manifest["destination"]
    if not re.fullmatch(r"[A-Za-z0-9_]+", name) or not re.fullmatch(
        r"/Game(?:/[A-Za-z0-9_]+)+", destination
    ):
        raise ValueError("Invalid Unreal asset name or destination")
    # Refuse overwrites: a push must never silently replace a previous character or its maps.
    if unreal.EditorAssetLibrary.does_directory_exist(destination):
        if unreal.EditorAssetLibrary.list_assets(
            destination, recursive=True, include_folder=False
        ):
            raise RuntimeError(
                "Destination already contains assets. Choose a new character name or folder."
            )
    mesh_file = _path(root, manifest["mesh"])
    maps = [_path(root, p) for p in manifest.get("maps", [])]
    vertices = manifest["vertices"]
    if len(vertices) != manifest["vertex_count"]:
        raise ValueError("Vertex count mismatch in manifest")
    if manifest.get("create_character"):
        optional = (
            Path(unreal.Paths.engine_plugins_dir())
            / "MetaHuman/MetaHumanCharacter/Content/Optional"
        )
        if (
            not (optional / "TextureSynthesis").is_dir()
            or not (optional / "BodyTextures").is_dir()
        ):
            raise RuntimeError(
                "MetaHuman Core Data is missing. In Epic Games Launcher, open UE 5.8 Options and install MetaHuman Creator Core Data. Disable Create Native MetaHuman to import only geometry and maps."
            )
    created = []
    report = {
        "mesh_assets": [],
        "texture_assets": [],
        "character": None,
        "status": "importing",
        "unreal_project_file": str(Path(unreal.Paths.get_project_file_path()).resolve()),
        "texture_overrides": {},
    }
    try:
        animated={}
        if manifest.get('create_character') and any(path.stem.rsplit('_',1)[-1] in {'WM1','WM2','WM3','CM1','CM2','CM3'} for path in maps):
            import runpy
            from hashlib import file_digest
            records=[]
            head_paths=set(manifest.get('texture_sets',{}).get('HEAD',manifest.get('maps',[])))
            for path in maps:
                if path.relative_to(root).as_posix() not in head_paths:continue
                with path.open('rb') as stream:sha=file_digest(stream,'sha256').hexdigest()
                kind=path.stem.rsplit('_',1)[-1]
                records.append({'kind':kind,'snapshot':str(path.relative_to(root)),'sha256':sha,'encoding':manifest.get('animated_encodings',{}).get(kind,'FULL')})
            animated_request={'maps':records,'image_python':manifest.get('image_python',''),'animated_input_root':str(root),'animated_normal_base':manifest.get('animated_normal_base')}
            animated=runpy.run_path(str(root/'unreal_native_animated.py'))['prepare'](animated_request,root.parent/('native-'+root.name))
            report['animated_map_encoding']=animated
        options = unreal.FbxImportUI()
        options.import_mesh = True
        options.import_as_skeletal = False
        options.import_materials = False
        options.import_textures = False
        options.automated_import_should_detect_type = False
        options.mesh_type_to_import = unreal.FBXImportType.FBXIT_STATIC_MESH
        options.static_mesh_import_data.combine_meshes = True
        options.static_mesh_import_data.generate_lightmap_u_vs = False
        options.static_mesh_import_data.remove_degenerates = False
        mesh_task = unreal.AssetImportTask()
        mesh_task.filename = str(mesh_file)
        mesh_task.destination_path = destination
        mesh_task.destination_name = name + "_Conformed"
        mesh_task.automated = True
        mesh_task.save = True
        mesh_task.options = options
        unreal.AssetToolsHelpers.get_asset_tools().import_asset_tasks([mesh_task])
        report["mesh_assets"] = list(mesh_task.imported_object_paths)
        created.extend(report["mesh_assets"])
        if not created:
            raise RuntimeError("Unreal did not import the FBX; inspect the Output Log")
        body_asset = None
        accessories = {}
        for role, filename in manifest.get("accessories", {}).items():
            task = unreal.AssetImportTask()
            task.filename = str(_path(root, filename))
            task.destination_path = destination
            task.destination_name = name + "_" + role
            task.automated = True
            task.save = True
            task.options = options
            unreal.AssetToolsHelpers.get_asset_tools().import_asset_tasks([task])
            accessory_paths = list(task.imported_object_paths)
            if not accessory_paths:
                raise RuntimeError("Accessory import failed: " + role)
            created.extend(accessory_paths)
            accessories[role] = unreal.load_asset(accessory_paths[0])
        if manifest["part"] == "FULL_BODY":
            body_task = unreal.AssetImportTask()
            body_task.filename = str(_path(root, manifest["body_mesh"]))
            body_task.destination_path = destination
            body_task.destination_name = name + "_BodyConformed"
            body_task.automated = True
            body_task.save = True
            body_task.options = options
            unreal.AssetToolsHelpers.get_asset_tools().import_asset_tasks([body_task])
            body_paths = list(body_task.imported_object_paths)
            if not body_paths:
                raise RuntimeError("Body FBX import failed")
            created.extend(body_paths)
            report["mesh_assets"].extend(body_paths)
            body_asset = unreal.load_asset(body_paths[0])
        imported_textures = {}
        for path in maps:
            native_delta=animated.get(path.stem.rsplit('_',1)[-1])
            task = unreal.AssetImportTask()
            task.filename = native_delta['file'] if native_delta else str(path)
            task.destination_path = destination + "/Maps"
            task.automated = True
            task.save = True
            unreal.AssetToolsHelpers.get_asset_tools().import_asset_tasks([task])
            paths = list(task.imported_object_paths)
            if not paths:
                raise RuntimeError("Texture import failed: " + path.name)
            created.extend(paths)
            report["texture_assets"].extend(paths)
            for asset_path in paths:
                texture = unreal.load_asset(asset_path)
                imported_textures[path.name] = texture
                if native_delta:
                    texture.set_editor_property('compression_settings',unreal.TextureCompressionSettings.TC_BC7)
                    texture.set_editor_property('srgb',False)
                elif path.stem.endswith(("_NORMAL", "_WM1", "_WM2", "_WM3")):
                    texture.set_editor_property(
                        "compression_settings",
                        unreal.TextureCompressionSettings.TC_NORMALMAP,
                    )
                    texture.set_editor_property("srgb", False)
                    texture.set_editor_property("flip_green_channel", False)
                elif path.stem.endswith(("_AO", "_CAVITY", "_ROUGHNESS")):
                    texture.set_editor_property("srgb", False)
                unreal.EditorAssetLibrary.save_loaded_asset(texture)
        if manifest.get("create_character"):
            if not hasattr(unreal, "MetaHumanCharacterEditorSubsystem"):
                raise RuntimeError(
                    "Enable MetaHuman Creator and install its Core Data for UE 5.8"
                )
            factory_class = getattr(unreal, "MetaHumanCharacterFactoryNew", None)
            if factory_class is None:
                raise RuntimeError(
                    "MetaHuman Character factory is unavailable in this engine build"
                )
            character = unreal.AssetToolsHelpers.get_asset_tools().create_asset(
                name + "_Character",
                destination,
                unreal.MetaHumanCharacter,
                factory_class(),
            )
            if character is None:
                raise RuntimeError("Could not create MetaHuman Character")
            created.append(character.get_path_name())
            subsystem = unreal.get_editor_subsystem(
                unreal.MetaHumanCharacterEditorSubsystem
            )
            if not subsystem.try_add_object_to_edit(character):
                raise RuntimeError(
                    "MetaHuman Core Data could not initialize the character"
                )
            try:
                if manifest.get('native_skin'):
                    # Set local synthesis drivers before conforming. Changing
                    # face detail afterwards can also update native face state.
                    recipe = manifest['native_skin']
                    settings = character.get_editor_property('skin_settings')
                    properties = settings.get_editor_property('skin')
                    for key in ('u', 'v', 'face_texture_index', 'body_texture_index', 'show_top_underwear'):
                        if key in recipe:
                            properties.set_editor_property(key, recipe[key])
                    settings.set_editor_property('skin', properties)
                    subsystem.commit_skin_settings(character, settings)
                    report['native_skin'] = {key: recipe[key] for key in
                        ('u', 'v', 'face_texture_index', 'body_texture_index', 'show_top_underwear') if key in recipe}
                vectors = [unreal.Vector(*v) for v in vertices]
                if manifest["part"] in {"BODY", "FULL_BODY"}:
                    body_vertices = manifest.get("body_vertices", vertices)
                    canonical = manifest.get(
                        "body_canonical_order", manifest.get("canonical_order", False)
                    )
                    if canonical:
                        body_vectors = [unreal.Vector3f(*v) for v in body_vertices]
                    else:
                        body_code, body_vectors = (
                            subsystem.get_mesh_for_body_conforming_from_template(
                                character,
                                body_asset
                                or unreal.load_asset(report["mesh_assets"][0]),
                                unreal.load_asset(report["mesh_assets"][0])
                                if body_asset
                                else None,
                                True,
                            )
                        )
                        if body_code != unreal.ImportErrorCode.SUCCESS:
                            raise RuntimeError(
                                "Body UV correspondence failed: " + str(body_code)
                            )
                    if not subsystem.conform_body_to_target(
                        character, body_vectors, [], True, True
                    ):
                        raise RuntimeError(
                            "Body conform failed; use a compatible MetaHuman body in A-pose"
                        )
                    subsystem.commit_body_state(character)
                if manifest["part"] in {"HEAD", "FULL_BODY"}:
                    if manifest.get("canonical_order", False) or mapped_conform:
                        params = unreal.MetaHumanCharacterFitToVerticesParams()
                        # These coordinates already include the artist's wrap,
                        # cleanup and head/body placement. Creator's defaults
                        # discard vertex detail and realign/adapt that work.
                        options=params.get_editor_property('options')
                        options.set_editor_property('alignment_options',unreal.MetaHumanAlignmentOptions.NONE)
                        # Native HF evaluation smooths the fitted head and adds
                        # a stock skin variant. The supplied source already owns
                        # its geometry detail; avoid processing that detail twice.
                        options.set_editor_property('disable_high_frequency_delta',True)
                        options.set_editor_property('adapt_neck',False)
                        params.set_editor_property('options',options)
                        params.head_vertices = [unreal.Vector(*v) for v in mapped_conform['head']] if mapped_conform else vectors
                        for role, field in (('eye_left', 'left_eye_vertices'), ('eye_right', 'right_eye_vertices'), ('teeth', 'teeth_vertices')):
                            points=mapped_conform.get(role) if mapped_conform else canonical_accessories.get(role)
                            if points is not None:setattr(params, field, [unreal.Vector(*v) for v in points])
                        success = subsystem.fit_state_to_target_vertices(
                            character, params
                        )
                        if not success:
                            raise RuntimeError(
                                "Head fitting failed. Template must have canonical MetaHuman vertex order."
                            )
                        report['head_conform'] = {'method': 'LOCAL_UV_CORRESPONDENCE' if mapped_conform else 'CANONICAL_VERTICES', 'accessories': {
                            role: {'vertex_count': len(points)} for role, points in (mapped_conform or canonical_accessories).items() if role!='head'}}
                        # These options request preservation; they do not prove
                        # the native conform or subsequent body/rig alignment
                        # retained every supplied vertex. Keep evidence separate
                        # from the requested behavior in artist/agent receipts.
                        report['head_conform']['fit_options']={
                            'alignment':'NONE','disable_high_frequency_delta':True,'adapt_neck':False}
                        target=mapped_conform or dict(canonical_accessories,head=vertices)
                        measured=audit_head_conform(character,target,root/'head-conform-state.json')
                        report['head_conform'].update(measured)
                        if measured.get('within_native_precision') is False:raise RuntimeError('Native head conform differs from supplied geometry; inspect head-conform-state.json before rigging')
                    else:
                        params = unreal.ImportFromTemplateParams()
                        params.match_vertices_by_u_vs = True
                        params.use_eye_meshes = bool(
                            accessories.get("eye_left") and accessories.get("eye_right")
                        )
                        params.use_teeth_mesh = bool(accessories.get("teeth"))
                        result = subsystem.import_from_template(
                            character,
                            unreal.load_asset(created[0]),
                            accessories.get("eye_left"),
                            accessories.get("eye_right"),
                            accessories.get("teeth"),
                            params,
                        )
                        if result != unreal.ImportErrorCode.SUCCESS:
                            raise RuntimeError(
                                "Head conform failed: "
                                + str(result)
                                + ". Use an unmodified MetaHuman LOD0 UV layout."
                            )
                        report['head_conform'] = {'method': 'TEMPLATE_UVS', 'accessories': {
                            role: {'asset': asset.get_path_name()} for role, asset in accessories.items()}}
                    subsystem.commit_face_state(character)
                elif manifest["part"] != "BODY":
                    raise ValueError("Expected HEAD or BODY template part")
                # Apply face and body maps through UE 5.8's public skin override structure.
                skin = character.get_editor_property("skin_settings")
                overrides = skin.get_editor_property("texture_material_overrides")
                texture_set = overrides.get_editor_property("texture_overrides")
                sets = manifest.get(
                    "texture_sets", {manifest["part"]: manifest.get("maps", [])}
                )
                kinds = {
                    "HEAD": {
                        "ALBEDO": unreal.FaceTextureType.BASECOLOR,
                        "NORMAL": unreal.FaceTextureType.NORMAL,
                        "CAVITY": unreal.FaceTextureType.CAVITY,
                        "WM1": unreal.FaceTextureType.NORMAL_ANIMATED_WM1,
                        "WM2": unreal.FaceTextureType.NORMAL_ANIMATED_WM2,
                        "WM3": unreal.FaceTextureType.NORMAL_ANIMATED_WM3,
                        "CM1": unreal.FaceTextureType.BASECOLOR_ANIMATED_CM1,
                        "CM2": unreal.FaceTextureType.BASECOLOR_ANIMATED_CM2,
                        "CM3": unreal.FaceTextureType.BASECOLOR_ANIMATED_CM3,
                    },
                    "BODY": {
                        "ALBEDO": unreal.BodyTextureType.BODY_BASECOLOR,
                        "NORMAL": unreal.BodyTextureType.BODY_NORMAL,
                        "CAVITY": unreal.BodyTextureType.BODY_CAVITY,
                    },
                }
                for part, paths in sets.items():
                    if part not in kinds:
                        continue
                    replacements = {}
                    for relative in paths:
                        kind = Path(relative).stem.rsplit("_", 1)[-1]
                        texture = imported_textures.get(Path(relative).name)
                        if kind in kinds[part] and texture:
                            replacements[kinds[part][kind]] = texture
                    if replacements:
                        texture_set.set_editor_property(
                            "face" if part == "HEAD" else "body", replacements
                        )
                        report["texture_overrides"][part] = [
                            str(k) for k in replacements
                        ]
                if report["texture_overrides"]:
                    overrides.set_editor_property("texture_overrides", texture_set)
                    overrides.set_editor_property("enable_texture_overrides", True)
                    skin.set_editor_property("texture_material_overrides", overrides)
                    subsystem.commit_skin_settings(character, skin)
                    character.set_editor_property(
                        "preview_material_type",
                        unreal.MetaHumanCharacterSkinPreviewMaterial.EDITABLE,
                    )
                subsystem.assemble_for_preview(character)
                roughness={part:imported_textures[Path(relative).name] for part,paths in sets.items()
                    for relative in paths if Path(relative).stem.endswith('_ROUGHNESS') and Path(relative).name in imported_textures}
                if roughness:
                    import runpy
                    report['roughness_materials']=runpy.run_path(str(root/'unreal_roughness.py'))['apply'](character,roughness,manifest.get('roughness_response'))
                    report['roughness_response']=manifest.get('roughness_response',{})
                if manifest.get('native_eyes'):
                    report['native_eyes']=apply_eye_settings(character,subsystem,manifest['native_eyes'])
                if manifest.get('native_skin'):
                    actual = character.get_editor_property('skin_settings').get_editor_property('skin')
                    report['native_skin'] = {key: actual.get_editor_property(key) for key in
                        ('u', 'v', 'face_texture_index', 'body_texture_index', 'show_top_underwear')}
                report['assembly'] = {
                    'ready': bool(subsystem.can_build_meta_human(character, False)),
                    'has_texture_sources': bool(character.has_high_resolution_textures),
                    'preview_only': True,
                }
                unreal.EditorAssetLibrary.save_loaded_asset(character)
                report["character"] = character.get_path_name()
            finally:
                subsystem.remove_object_to_edit(character)
        sources = repair_import_sources(created, root, manifest.get('reimport_root', root))
        report['reimport_sources'] = sources
        if sources['warnings']:
            # Do not lose source files or misreport a completed character import
            # when only optional Reimport metadata could not be updated.
            report['retain_source_snapshot'] = True
        report["status"] = "complete"
        unreal.log("MetaHuman Maker: imported " + destination)
    except Exception as exc:
        report["status"] = "failed"
        report["error"] = str(exc)
        report["partial_assets"] = created
        # Preserve imported geometry and textures for inspection instead of destroying assets.
        raise
    finally:
        (root / "unreal-import-report.json").write_text(
            json.dumps(report, indent=2), encoding="utf-8"
        )
    return report


if __name__ == "__main__":
    import sys

    import_bundle(
        sys.argv[1] if len(sys.argv) > 1 else Path(__file__).with_name("manifest.json")
    )
