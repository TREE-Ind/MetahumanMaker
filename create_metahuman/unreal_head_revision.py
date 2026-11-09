"""Create a separate, locally conformed Creator revision from native artist state.

This never requests a rig or texture sources. It retains partial new assets on
failure and never saves or edits the source character or its material assets.
"""
import hashlib,json,math,re,shutil,traceback,runpy
from pathlib import Path


def digest(path):
    with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()


def revise(request_path):
    import unreal
    request_path=Path(request_path).resolve();request=json.loads(request_path.read_text(encoding='utf-8'))
    receipt=request_path.parent/'head-revision.json'
    if receipt.exists():raise ValueError('This revision already has a receipt. Inspect the original action or prepare a new revision request.')
    report={'status':'preflight','created':[],'rig_or_texture_requested':False,'request_sha256':digest(request_path)}
    subsystem=unreal.get_editor_subsystem(unreal.MetaHumanCharacterEditorSubsystem)
    owned=False;character=None
    def write():
        temporary=receipt.with_suffix('.tmp');temporary.write_text(json.dumps(report,indent=2),encoding='utf-8');temporary.replace(receipt)
    def snapshot(obj):
        value=json.loads(unreal.CreateMHArtistLibrary.snapshot_object(obj))
        # UObject outer enumeration has no stable order. Preserve every
        # record/property while comparing identity by its full object path.
        value['objects']=sorted(value.get('objects',[]),key=lambda row:row['object'])
        return value
    try:
        if request.get('schema')!=1 or request.get('rig_or_texture_request') is not False:
            raise ValueError('Unsupported head revision request')
        actual=Path(unreal.Paths.get_project_file_path()).resolve()
        if not request.get('expected_project') or actual!=Path(request['expected_project']).resolve():
            raise ValueError('Open the linked Unreal project before creating a revision')
        destination=request['destination']
        if not re.fullmatch(r'/Game(?:/[A-Za-z0-9_]+)+',destination):raise ValueError('Invalid revision destination')
        folder,name=destination.rsplit('/',1)
        if unreal.EditorAssetLibrary.does_directory_exist(folder) or unreal.EditorAssetLibrary.does_asset_exist(destination):
            raise ValueError('Choose an unused revision folder; existing assets were retained')
        if shutil.disk_usage(actual.parent).free<2*1024**3:raise ValueError('Head revision needs 2 GiB of disk headroom')
        budget=runpy.run_path(str(Path(__file__).with_name('resource_budget.py')))
        budget['require']('native head revision',8*1024**3)
        vertices=request['geometry']['vertices']
        if set(vertices)-{'head','eye_left','eye_right','teeth'} or not vertices.get('head'):raise ValueError('Unsupported head correspondence')
        for role,points in vertices.items():
            if not isinstance(points,list) or any(not isinstance(p,list) or len(p)!=3 or any(type(v) not in (int,float) or not math.isfinite(v) for v in p) for p in points):
                raise ValueError('Invalid native vertices for '+role)
        maps=request['maps']
        kinds={'NORMAL':'NORMAL','CAVITY':'CAVITY','ALBEDO':'BASECOLOR',
               'CM1':'BASECOLOR_ANIMATED_CM1','CM2':'BASECOLOR_ANIMATED_CM2','CM3':'BASECOLOR_ANIMATED_CM3'}
        if not {'NORMAL','CAVITY'}.issubset(maps) or set(maps)-set(kinds):raise ValueError('Head revision expects normal/cavity and optional accepted skin color maps')
        expression=set(maps)&{'CM1','CM2','CM3'}
        if expression and (expression!={'CM1','CM2','CM3'} or 'ALBEDO' not in maps):raise ValueError('Expression color requires albedo and all three CM maps')
        project=Path(request['project']).resolve()
        if request.get('project_sha256') and digest(project/'project.blend')!=request['project_sha256']:raise ValueError('The prepared candidate project changed')
        for kind,record in maps.items():
            path=Path(record['file']).resolve()
            owned_inputs=request_path.parent/'inputs'
            if not (path.is_relative_to(project/'Maps') or path.is_relative_to(owned_inputs)) or digest(path)!=record['sha256']:raise ValueError('A prepared revision map changed')
        animated_request={'maps':[dict(record,kind=kind,snapshot=str(Path(record['file']).resolve().relative_to(request_path.parent))) for kind,record in maps.items()],
                          'image_python':request.get('image_python',''),'animated_input_root':str(request_path.parent)}
        animated=runpy.run_path(str(Path(__file__).with_name('unreal_native_animated.py')))['prepare'](animated_request,request_path.parent)
        if animated:report['animated_map_encoding']=animated
        source=unreal.load_asset(request['source_character'])
        if not isinstance(source,unreal.MetaHumanCharacter):raise ValueError('The source Creator character is missing')
        original=snapshot(source)
        if any(o['package_dirty'] for o in original['objects']):raise ValueError('Save native artist edits before creating a revision')
        source_state=json.loads(unreal.CreateMHCharacterLibrary.inspect_character(source))
        source_acquired=False
        try:
            # Bridge 0.21 can read pending flags on a closed source. Avoid
            # allocating its rig merely to copy persistent artist settings.
            if not source_state:
                if not subsystem.is_object_added_for_editing(source):
                    if not subsystem.try_add_object_to_edit(source):raise RuntimeError('Creator could not inspect the source character')
                    source_acquired=True
                source_state=json.loads(unreal.CreateMHCharacterLibrary.inspect_character(source))
            if source_state['rig_request_pending'] or source_state['texture_request_pending']:raise ValueError('Finish pending source rig or texture requests first')
        finally:
            if source_acquired:subsystem.remove_object_to_edit(source)
        report['source_inspection_scope']=source_state.get('inspection_scope','LEGACY_SESSION')
        report['source_native_initializations']=int(source_acquired)
        original=snapshot(source)
        if any(o['package_dirty'] for o in original['objects']):raise ValueError('Native initialization changed the source; review and save it before creating a revision')
        # Native packages remain the authority for opaque artist state.
        package=source.get_path_name().split('.')[0].removeprefix('/Game/')
        source_file=Path(unreal.Paths.project_content_dir()).resolve()/(package+'.uasset')
        source_hash=digest(source_file)
        # Loading a saved pipeline can leave unreachable owned specifications.
        # Clone compilation collects these later. Establish the baseline after
        # synchronous engine GC, while the Python reference keeps source alive.
        # Never exempt missing objects from the final preservation comparison.
        unreal.collect_garbage()
        stable=snapshot(source)
        before_gc={row['object']:row for row in original['objects']}
        after_gc={row['object']:row for row in stable['objects']}
        if (set(after_gc)-set(before_gc) or
                any(before_gc[path]!=row for path,row in after_gc.items()) or
                digest(source_file)!=source_hash):
            raise RuntimeError('Source state changed while establishing the revision baseline')
        report['source_baseline_gc']={'synchronous':True,
            'collected_objects':sorted(set(before_gc)-set(after_gc)),
            'retained_objects_unchanged':True,'source_file_unchanged':True}
        original=stable
        (request_path.parent/'source-before.json').write_text(json.dumps(original,indent=2),encoding='utf-8')
        tools=unreal.AssetToolsHelpers.get_asset_tools()
        character=tools.duplicate_asset(name,folder,source)
        if not isinstance(character,unreal.MetaHumanCharacter):raise RuntimeError('Could not duplicate the source Creator character')
        # Existing expression maps are inherited artist baselines, not maps
        # owned by this new character's future expression-update operations.
        unreal.EditorAssetLibrary.remove_metadata_tag(character,'CreateMetaHuman.ExpressionColorBindings')
        report['created'].append(character.get_path_name());report.update(status='conforming',character=character.get_path_name(),source=source.get_path_name());write()
        owned=not subsystem.is_object_added_for_editing(character)
        if owned and not subsystem.try_add_object_to_edit(character):raise RuntimeError('Creator could not initialize the revision')
        state=json.loads(unreal.CreateMHCharacterLibrary.inspect_character(character))
        if state['rig_request_pending'] or state['texture_request_pending']:raise ValueError('Finish pending native requests first')
        if state['rig_state']=='RIGGED':subsystem.remove_face_rig(character)
        params=unreal.MetaHumanCharacterFitToVerticesParams()
        options=params.get_editor_property('options')
        options.set_editor_property('alignment_options',unreal.MetaHumanAlignmentOptions.NONE)
        # A revision inherits deliberate native artist evaluation settings.
        # Fresh imports disable the extra stock HF layer; do not silently
        # change an existing artist layer here. Measure the resulting geometry
        # below so an inherited HF layer cannot masquerade as an exact fit.
        options.set_editor_property('disable_high_frequency_delta',False)
        options.set_editor_property('adapt_neck',False)
        params.set_editor_property('options',options)
        for role,field in [('head','head_vertices'),('eye_left','left_eye_vertices'),('eye_right','right_eye_vertices'),('teeth','teeth_vertices')]:
            if role in vertices:setattr(params,field,[unreal.Vector(*point) for point in vertices[role]])
        if not subsystem.fit_state_to_target_vertices(character,params):raise RuntimeError('Native head conform failed')
        subsystem.commit_face_state(character)
        report['status']='importing-maps';write()
        source_face=dict(source.get_editor_property('skin_settings').get_editor_property('texture_material_overrides').get_editor_property('texture_overrides').get_editor_property('face'))
        replacements={}
        for kind,record in maps.items():
            task=unreal.AssetImportTask();path=Path(animated[kind]['file'] if kind in animated else record['file']);asset_name=path.stem
            for key,value in [('filename',str(path)),('destination_path',folder+'/Maps'),('destination_name',asset_name),('automated',True),('save',False),('replace_existing',False)]:task.set_editor_property(key,value)
            tools.import_asset_tasks([task]);paths=list(task.get_editor_property('imported_object_paths'))
            if len(paths)!=1:raise RuntimeError('Native map import did not return one texture')
            texture=unreal.load_asset(paths[0])
            if not isinstance(texture,unreal.Texture2D):raise RuntimeError('Native map import did not create Texture2D')
            native_key=getattr(unreal.FaceTextureType,kinds[kind]);inherited=source_face.get(native_key)
            color=kind in {'ALBEDO','CM1','CM2','CM3'}
            texture.set_editor_property('srgb',bool(inherited.get_editor_property('srgb')) if inherited and color else color)
            compression='TC_NORMALMAP' if kind=='NORMAL' else ('TC_DEFAULT' if kind in {'ALBEDO','CM1','CM2','CM3'} else 'TC_MASKS')
            if kind in animated:
                texture.set_editor_property('srgb',False);compression='TC_BC7'
            texture.set_editor_property('compression_settings',getattr(unreal.TextureCompressionSettings,compression))
            if kind=='NORMAL':texture.set_editor_property('flip_green_channel',False)
            if not unreal.EditorAssetLibrary.save_loaded_asset(texture,False):raise RuntimeError('Could not save new revision texture')
            replacements[getattr(unreal.FaceTextureType,kinds[kind])]=texture;report['created'].append(texture.get_path_name())
        # Start from the complete native skin settings, preserving freckles,
        # tone, roughness, body and expression overrides rather than defaults.
        skin=source.get_editor_property('skin_settings').copy()
        retained_skin={key:skin.get_editor_property(key).export_text() for key in ('skin','freckles')}
        overrides=skin.get_editor_property('texture_material_overrides').copy()
        textures=overrides.get_editor_property('texture_overrides').copy()
        face=dict(textures.get_editor_property('face'));old_to_new={}
        for key,texture in replacements.items():
            if face.get(key):old_to_new[face[key].get_path_name()]=texture
            face[key]=texture
        textures.set_editor_property('face',face);overrides.set_editor_property('texture_overrides',textures)
        overrides.set_editor_property('enable_texture_overrides',True)
        # Make artist material overrides private before replacing any texture
        # parameters. Shared parent graphs stay read-only.
        materials=overrides.get_editor_property('material_overrides').copy();copied={}
        def private(material):
            if not material:return material
            path=material.get_path_name()
            if path in copied:return copied[path]
            if not isinstance(material,unreal.MaterialInstanceConstant):raise ValueError('This artist override needs an explicit material adapter: '+path)
            target=tools.duplicate_asset('Revision_'+str(len(copied))+'_'+material.get_name(),folder+'/Materials',material)
            if not target:raise RuntimeError('Could not retain artist material override')
            copied[path]=target;report['created'].append(target.get_path_name())
            unreal.EditorAssetLibrary.set_metadata_tag(target,'CreateMetaHuman.CharacterOwner',character.get_path_name())
            parameters=[value.copy() for value in target.get_editor_property('texture_parameter_values')]
            for value in parameters:
                texture=value.get_editor_property('parameter_value')
                if texture and texture.get_path_name() in old_to_new:value.set_editor_property('parameter_value',old_to_new[texture.get_path_name()])
            target.set_editor_property('texture_parameter_values',parameters)
            if unreal.EditorAssetLibrary.get_metadata_tag(material,'CreateMetaHuman.RoughnessOwner')==source.get_path_name():
                unreal.EditorAssetLibrary.set_metadata_tag(target,'CreateMetaHuman.RoughnessOwner',character.get_path_name())
                calibration=unreal.EditorAssetLibrary.get_metadata_tag(material,'CreateMetaHuman.RoughnessResponse')
                if calibration:unreal.EditorAssetLibrary.set_metadata_tag(target,'CreateMetaHuman.RoughnessResponse',calibration)
            if not unreal.EditorAssetLibrary.save_loaded_asset(target,False):raise RuntimeError('Could not save private artist material')
            return target
        materials.set_editor_property('skin',{key:private(value) for key,value in dict(materials.get_editor_property('skin')).items()})
        materials.set_editor_property('body',private(materials.get_editor_property('body')))
        eye_helpers=runpy.run_path(str(Path(__file__).with_name('unreal_eye_isolation.py')))
        eye_tag,report['eye_material_isolation']=eye_helpers['isolate'](materials,private,character.get_path_name(),
            unreal.EditorAssetLibrary.get_metadata_tag(source,eye_helpers['TAG']))
        for material in copied.values():
            if not unreal.EditorAssetLibrary.save_loaded_asset(material,False):raise RuntimeError('Could not save isolated artist materials')
        if eye_tag:unreal.EditorAssetLibrary.set_metadata_tag(character,eye_helpers['TAG'],eye_tag)
        overrides.set_editor_property('material_overrides',materials);skin.set_editor_property('texture_material_overrides',overrides)
        subsystem.commit_skin_settings(character,skin);subsystem.assemble_for_preview(character)
        applied_face=dict(character.get_editor_property('skin_settings').get_editor_property('texture_material_overrides').get_editor_property('texture_overrides').get_editor_property('face'))
        if any(applied_face.get(key)!=value for key,value in replacements.items()):raise RuntimeError('A selected revision texture binding was not retained')
        if any(character.get_editor_property('skin_settings').get_editor_property(key).export_text()!=value for key,value in retained_skin.items()):
            raise RuntimeError('Native skin or freckles changed during revision')
        after=snapshot(character)
        original_props=next(o['properties'] for o in original['objects'] if o['object']==source.get_path_name())
        revision_props=next(o['properties'] for o in after['objects'] if o['object']==character.get_path_name())
        preserved={key:original_props[key]['text']==revision_props[key]['text'] for key in ('EyesSettings','HeadModelSettings','FaceEvaluationSettings') if key in original_props}
        if len(preserved)!=3 or not all(preserved.values()):raise RuntimeError('A native artist appearance setting changed during revision')
        audit=runpy.run_path(str(Path(__file__).with_name('unreal_import.py')))['audit_head_conform']
        report['head_geometry']=audit(character,vertices,request_path.parent/'head-conform-state.json')
        report['native_high_frequency_policy']='RETAIN_ARTIST_EVALUATION_SETTINGS'
        if report['head_geometry'].get('within_native_precision') is False:
            report.setdefault('warnings',[]).append('Retained native artist evaluation layers change the supplied head geometry; inspect head_geometry before final baking.')
        source_after=snapshot(source)
        (request_path.parent/'source-after.json').write_text(json.dumps(source_after,indent=2),encoding='utf-8')
        if source_after!=original or digest(source_file)!=source_hash:
            old={row['object']:row for row in original['objects']};new={row['object']:row for row in source_after['objects']}
            changed=[]
            for path in sorted(set(old)|set(new)):
                if path not in old or path not in new:changed.append(path+': object membership');continue
                for key in sorted(set(old[path])|set(new[path])):
                    if key=='properties':
                        for field in sorted(set(old[path][key])|set(new[path][key])):
                            if old[path][key].get(field)!=new[path][key].get(field):changed.append(path+'.'+field)
                    elif old[path].get(key)!=new[path].get(key):changed.append(path+': '+key)
            report['source_changed_fields']=changed;report['source_file_unchanged']=digest(source_file)==source_hash
            raise RuntimeError('The source character changed during revision')
        if not unreal.EditorAssetLibrary.save_loaded_asset(character,False):raise RuntimeError('Could not save native revision')
        final_state=json.loads(unreal.CreateMHCharacterLibrary.inspect_character(character))
        report.update(status='complete',action='HEAD_REVISION',local_geometry_witness=request.get('local_geometry_witness'),unreal_project_file=str(actual),assembly=dict(final_state,detailed_status_available=True),native_settings_preserved=preserved,source_unchanged=True,source_package_sha256=source_hash,
            conform_options={'alignment':'NONE','disable_high_frequency_delta':False,'adapt_neck':False},
            material_overrides_copied=len(copied),state=json.loads(unreal.CreateMHCharacterLibrary.inspect_character(character)),
            map_sha256={kind:record['sha256'] for kind,record in maps.items()},map_bindings={kind:replacements[getattr(unreal.FaceTextureType,kinds[kind])].get_path_name() for kind in maps},limitations=['New head geometry is unrigged; no rig or texture sources requested','Shared native shader graphs remain read-only','Actor-specific edits are not copied by a Creator character revision'])
    except Exception as exc:report.update(status='failed',error=str(exc),traceback=traceback.format_exc())
    finally:
        if owned and character:subsystem.remove_object_to_edit(character)
        write()
    return report
