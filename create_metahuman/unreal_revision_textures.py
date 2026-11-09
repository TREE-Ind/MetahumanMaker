"""Refresh owned native-revision maps without conforming or requesting a rig."""
import hashlib,itertools,json,math,runpy,traceback
from datetime import datetime,timezone
from pathlib import Path

KINDS={'NORMAL':('NORMAL','Normal'),'CAVITY':('CAVITY','Cavity'),'ALBEDO':('BASECOLOR','Basecolor'),
       **{k:('BASECOLOR_ANIMATED_'+k,'Basecolor Animated Delta '+k.lower()) for k in ('CM1','CM2','CM3')},
       **{k:('NORMAL_ANIMATED_'+k,'Normal Animated Delta '+k.lower()) for k in ('WM1','WM2','WM3')}}


def digest(path):
    with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()


def coverage(vertices,points,tolerance=.002):
    cells={}
    for p in points:
        value=(p.x,p.y,p.z);key=tuple(math.floor(v/tolerance) for v in value);cells.setdefault(key,[]).append(value)
    checks={}
    for role,values in vertices.items():
        if not values:raise ValueError('Empty native '+role+' correspondence')
        maximum=0.
        for point in values:
            if len(point)!=3 or any(not math.isfinite(v) for v in point):raise ValueError('Invalid native correspondence')
            key=tuple(math.floor(v/tolerance) for v in point);best=float('inf')
            for offset in itertools.product((-1,0,1),repeat=3):
                for other in cells.get(tuple(a+b for a,b in zip(key,offset)),[]):best=min(best,sum((a-b)**2 for a,b in zip(point,other)))
            if best>tolerance**2:raise ValueError('Native '+role+' differs from the prepared revision; retain UE geometry edits before refreshing maps')
            maximum=max(maximum,math.sqrt(best))
        checks[role]={'maximum_distance_cm':maximum,'vertices':len(values),'scope':'submitted canonical anatomy coverage in composite native preview'}
    return checks


def mesh_digest(subsystem,component):
    points,indices=subsystem.get_mesh_data_for_conforming(component.get_skeletal_mesh_asset())
    return hashlib.sha256(json.dumps([[(p.x,p.y,p.z) for p in points],list(indices)],separators=(',',':')).encode()).hexdigest()


def update(request_path):
    import unreal
    request_path=Path(request_path).resolve();folder=request_path.parent;snapshot=folder/'scripts'
    request=json.loads(request_path.read_text(encoding='utf-8'))
    report={'status':'preflight','action':'UPDATE_TEXTURES','geometry_mode':'NATIVE_REVISION','updated':[],
        'warnings':[],'retain_source_snapshot':True,'requires_reassembly':bool(request.get('existing_assembly'))}
    actor=None;owned=False;character=None
    def commit():
        temporary=folder/'texture-progress.tmp';temporary.write_text(json.dumps(report,indent=2),encoding='utf-8');temporary.replace(folder/'texture-progress.json')
    try:
        if request.get('schema_version')!=1 or request.get('action')!='UPDATE_TEXTURES' or request.get('geometry_mode')!='NATIVE_REVISION':raise ValueError('Unsupported native revision map update')
        project=Path(unreal.Paths.get_project_file_path()).resolve()
        if not request.get('expected_project') or project!=Path(request['expected_project']).resolve():raise ValueError('Open the linked Unreal project')
        report.update(character=request['character'],unreal_project_file=str(project))
        budget=runpy.run_path(str(snapshot/'resource_budget.py'));budget['require']('native texture refresh',4*1024**3)
        character=unreal.load_asset(request['character'])
        if not isinstance(character,unreal.MetaHumanCharacter):raise ValueError('The native revision is missing')
        native_before=json.loads(unreal.CreateMHArtistLibrary.snapshot_object(character))
        if any(o['package_dirty'] for o in native_before['objects']):raise ValueError('Save native artist edits before refreshing maps')
        root=character.get_path_name().rsplit('/',1)[0]
        geometry_file=snapshot/'geometry.json'
        if digest(geometry_file)!=request['geometry_sha256']:raise ValueError('The prepared native correspondence changed')
        geometry=json.loads(geometry_file.read_text(encoding='utf-8'))
        if set(geometry['vertices'])-{'head','eye_left','eye_right','teeth'} or not geometry['vertices'].get('head'):raise ValueError('Unsupported native anatomy correspondence')
        subsystem=unreal.get_editor_subsystem(unreal.MetaHumanCharacterEditorSubsystem)
        owned=not subsystem.is_object_added_for_editing(character)
        if owned and not subsystem.try_add_object_to_edit(character):raise RuntimeError('Creator could not inspect the revision')
        state=json.loads(unreal.CreateMHCharacterLibrary.inspect_character(character))
        if state['rig_request_pending'] or state['texture_request_pending']:raise ValueError('Finish pending rig/texture requests before refreshing maps')
        actor=subsystem.spawn_meta_human_actor(character,True)
        face_component=actor.get_editor_property('face_component');body_component=actor.get_editor_property('body_component')
        points,_=subsystem.get_mesh_data_for_conforming(face_component.get_skeletal_mesh_asset())
        report['geometry_checks']=coverage(geometry['vertices'],points)
        before_mesh={'head':mesh_digest(subsystem,face_component),'body':mesh_digest(subsystem,body_component)}
        skin=character.get_editor_property('skin_settings').copy();controls={key:skin.get_editor_property(key).export_text() for key in ('skin','freckles')}
        overrides=skin.get_editor_property('texture_material_overrides').copy();textures=overrides.get_editor_property('texture_overrides').copy();face=dict(textures.get_editor_property('face'))
        materials=overrides.get_editor_property('material_overrides').copy();faces=dict(materials.get_editor_property('skin'))
        selected={item['kind'] for item in request['maps']}
        scope=request.get('scope');allowed={'NORMAL','CAVITY'}
        if scope in {'SKIN','ALL_HEAD'}:allowed|={'ALBEDO','CM1','CM2','CM3'}
        if scope=='ALL_HEAD':allowed|={'WM1','WM2','WM3','ROUGHNESS'}
        if scope not in {'DETAIL','SKIN','ALL_HEAD'} or not {'NORMAL','CAVITY'}.issubset(selected) or selected-allowed:raise ValueError('Map selection differs from the requested scope')
        for group in ({'CM1','CM2','CM3'},{'WM1','WM2','WM3'}):
            if selected&group and selected&group!=group:raise ValueError('Incomplete expression map set')
        bindings=runpy.run_path(str(snapshot/'unreal_roughness.py'))
        if 'ROUGHNESS' in selected:
            if not faces or any(not material for material in faces.values()):raise ValueError('Head roughness refresh needs existing managed skin overrides')
            for material in faces.values():bindings['validate_refresh'](material,request.get('roughness_response',{}).get('HEAD'))
        tasks=[];seen=set()
        for item in request['maps']:
            relative=Path(item['snapshot']);filename=(snapshot/relative).resolve();kind=item['kind']
            if item.get('role')!='HEAD' or relative.is_absolute() or not filename.is_relative_to((snapshot/'Maps').resolve()) or filename.suffix.lower()!='.png' or digest(filename)!=item['sha256']:raise ValueError('Invalid or changed head texture snapshot')
            expected=root+'/Maps/'+filename.stem+'.'+filename.stem
            if item['asset']!=expected or kind in seen:raise ValueError('Repeated or unowned head texture target')
            seen.add(kind);current=unreal.load_asset(expected)
            if current and not isinstance(current,unreal.Texture2D):raise ValueError('The map destination is not a texture')
            if current:
                bound=face.get(getattr(unreal.FaceTextureType,KINDS[kind][0])) if kind in KINDS else None
                if current!=bound and unreal.EditorAssetLibrary.get_metadata_tag(current,'CreateMetaHuman.TextureOwner')!=character.get_path_name():raise ValueError('The destination texture is an unbound artist asset; retain it before refreshing')
            tasks.append((item,filename,current))
        if not tasks:raise ValueError('No head textures selected')
        report['status']='encoding-animated-maps';commit()
        animated=runpy.run_path(str(snapshot/'unreal_native_animated.py'))['prepare'](request,folder)
        if animated:report['animated_map_encoding']=animated
        live=bindings['capture'](character)
        imported={};tools=unreal.AssetToolsHelpers.get_asset_tools();report['status']='importing';commit()
        for item,filename,current in tasks:
            kind=item['kind'];native=face.get(getattr(unreal.FaceTextureType,KINDS[kind][0])) if kind in KINDS else current
            normal=kind in {'NORMAL','WM1','WM2','WM3'};color=kind=='ALBEDO' or kind.startswith('CM')
            srgb=bool(native.get_editor_property('srgb')) if native and color else color
            compression=unreal.TextureCompressionSettings.TC_NORMALMAP if normal else (unreal.TextureCompressionSettings.TC_DEFAULT if color else unreal.TextureCompressionSettings.TC_MASKS)
            native_delta=animated.get(kind)
            if native_delta:
                filename=Path(native_delta['file']);srgb=False;compression=unreal.TextureCompressionSettings.TC_BC7;normal=False
            task=unreal.AssetImportTask()
            for key,value in [('filename',str(filename)),('destination_path',root+'/Maps'),('destination_name',filename.stem),('automated',True),('save',False),('replace_existing',bool(current)),('replace_existing_settings',False)]:task.set_editor_property(key,value)
            tools.import_asset_tasks([task])
            if list(task.get_editor_property('imported_object_paths'))!=[item['asset']]:raise RuntimeError('Unreal imported an unexpected head texture')
            texture=unreal.load_asset(item['asset']);texture.set_editor_property('srgb',srgb);texture.set_editor_property('compression_settings',compression)
            if normal:texture.set_editor_property('flip_green_channel',False)
            unreal.EditorAssetLibrary.set_metadata_tag(texture,'CreateMetaHuman.TextureOwner',character.get_path_name())
            row={'asset':texture.get_path_name(),'kind':kind,'role':'HEAD','sha256':item['sha256']};report['updated'].append(row)
            source=Path(item['source'])
            if digest(source)==item['sha256']:
                if native_delta:
                    row.update(original_source=str(source),native_sha256=native_delta['sha256'],encoding=native_delta['encoding'])
                    source=filename
                imports=texture.get_editor_property('asset_import_data');imports.scripted_add_filename(str(source),0,'')
                if Path(imports.extract_filenames()[0]).resolve()!=source.resolve():raise RuntimeError('Persistent texture source was not retained')
                row['source']=str(source)
            else:raise ValueError('The accepted source map changed during native refresh')
            if not unreal.EditorAssetLibrary.save_loaded_asset(texture,False):raise RuntimeError('Could not save refreshed head texture')
            imported[kind]=texture;commit()
        # Only head material overrides are changed. Shared artist overrides
        # receive private copies; shared parent graphs remain read-only.
        private={}
        for key,material in faces.items():
            if not material:continue
            if not isinstance(material,unreal.MaterialInstanceConstant):raise ValueError('This head material needs an explicit refresh adapter')
            path=material.get_path_name()
            if path in private:faces[key]=private[path];continue
            own=path.startswith(root+'/Materials/') and unreal.EditorAssetLibrary.get_metadata_tag(material,'CreateMetaHuman.CharacterOwner')==character.get_path_name()
            target=material if own else tools.duplicate_asset('Refresh_'+str(len(private))+'_'+material.get_name(),root+'/Materials',material)
            if not target:raise RuntimeError('Could not retain the artist head material')
            unreal.EditorAssetLibrary.set_metadata_tag(target,'CreateMetaHuman.CharacterOwner',character.get_path_name())
            if target!=material:
                for tag in ('CreateMetaHuman.RoughnessOwner',bindings['RESPONSE_TAG']):
                    value=unreal.EditorAssetLibrary.get_metadata_tag(material,tag)
                    if value:unreal.EditorAssetLibrary.set_metadata_tag(target,tag,value)
            parameters={str(name) for name in unreal.MaterialEditingLibrary.get_texture_parameter_names(target)}
            for kind,texture in imported.items():
                name=KINDS[kind][1] if kind in KINDS else 'Roughness'
                if name in parameters:unreal.MaterialEditingLibrary.set_material_instance_texture_parameter_value(target,name,texture)
            if 'ROUGHNESS' in imported:
                calibration=bindings['refresh_material'](target,imported['ROUGHNESS'],character.get_path_name(),request.get('roughness_response',{}).get('HEAD'))
                if calibration['mode']=='PRESERVE_LEGACY':
                    warning='Legacy roughness calibration history is unavailable; existing native finish parameters were preserved without recalibration'
                    if warning not in report['warnings']:report['warnings'].append(warning)
                report.setdefault('roughness_materials',{})[str(key)]=target.get_path_name()
            if not unreal.EditorAssetLibrary.save_loaded_asset(target,False):raise RuntimeError('Could not save the private head material')
            private[path]=target;faces[key]=target
        for kind,texture in imported.items():
            if kind in KINDS:face[getattr(unreal.FaceTextureType,KINDS[kind][0])]=texture
        textures.set_editor_property('face',face);overrides.set_editor_property('texture_overrides',textures);overrides.set_editor_property('enable_texture_overrides',True)
        materials.set_editor_property('skin',faces);overrides.set_editor_property('material_overrides',materials);skin.set_editor_property('texture_material_overrides',overrides)
        subsystem.commit_skin_settings(character,skin)
        changed_names={KINDS[kind][1] if kind in KINDS else 'Roughness' for kind in imported}
        refreshed_materials={material.get_path_name() for material in faces.values() if material}
        def changed(component,current,key):
            parent=current.get_editor_property('parent')
            return component.get_name()=='Face' and parent and parent.get_path_name() in refreshed_materials and key in changed_names
        bindings['restore'](live,changed,bindings['managed_scalar_skip'](character,{'HEAD':imported['ROUGHNESS']} if 'ROUGHNESS' in imported else {}))
        after_skin=character.get_editor_property('skin_settings')
        if any(after_skin.get_editor_property(key).export_text()!=value for key,value in controls.items()):raise RuntimeError('Native skin or freckles changed during refresh')
        after_mesh={'head':mesh_digest(subsystem,actor.get_editor_property('face_component')),'body':mesh_digest(subsystem,actor.get_editor_property('body_component'))}
        if after_mesh!=before_mesh:raise RuntimeError('Native geometry changed during texture-only refresh')
        after_state=json.loads(unreal.CreateMHCharacterLibrary.inspect_character(character))
        if any(after_state.get(key)!=state.get(key) for key in ('rig_state','rig_request_pending','texture_request_pending')):raise RuntimeError('Native rig/request state changed during refresh')
        native_after=json.loads(unreal.CreateMHArtistLibrary.snapshot_object(character))
        before_props=next(o['properties'] for o in native_before['objects'] if o['object']==character.get_path_name())
        after_props=next(o['properties'] for o in native_after['objects'] if o['object']==character.get_path_name())
        preserved={key:before_props[key]['text']==after_props[key]['text'] for key in ('EyesSettings','HeadModelSettings','FaceEvaluationSettings') if key in before_props}
        if len(preserved)!=3 or not all(preserved.values()):raise RuntimeError('Native artist settings changed during refresh')
        actual_face=dict(after_skin.get_editor_property('texture_material_overrides').get_editor_property('texture_overrides').get_editor_property('face'))
        if any(actual_face.get(getattr(unreal.FaceTextureType,KINDS[kind][0]))!=texture for kind,texture in imported.items() if kind in KINDS):raise RuntimeError('A refreshed face override was not retained')
        verified=[];component=actor.get_editor_property('face_component')
        for index in range(component.get_num_materials()):
            material=component.get_material(index)
            if not material:continue
            names={str(name) for name in unreal.MaterialEditingLibrary.get_texture_parameter_names(material)}
            if 'Micro Skin Details Normal' not in names or 'Basecolor' not in names:continue
            for kind,texture in imported.items():
                name=KINDS[kind][1] if kind in KINDS else 'Roughness'
                if name not in names or material.get_texture_parameter_value(name)!=texture:raise RuntimeError('The evaluated skin material did not use refreshed '+kind)
            verified.append(index)
        if not verified:raise RuntimeError('No evaluated skin material was verified after refresh')
        if not unreal.EditorAssetLibrary.save_loaded_asset(character,False):raise RuntimeError('Could not save the native map bindings')
        report.update(status='complete',retain_source_snapshot=False,geometry_preserved=True,rig_preserved=True,native_settings_preserved=preserved,
            runtime_material_parameters_preserved=True,evaluated_skin_slots=verified,map_bindings={kind:texture.get_path_name() for kind,texture in imported.items()})
    except Exception as exc:report.update(status='failed',error=str(exc),traceback=traceback.format_exc())
    finally:
        if actor:unreal.get_editor_subsystem(unreal.EditorActorSubsystem).destroy_actor(actor)
        if owned and character:subsystem.remove_object_to_edit(character)
        report['completed_at']=datetime.now(timezone.utc).isoformat();commit()
    return report
