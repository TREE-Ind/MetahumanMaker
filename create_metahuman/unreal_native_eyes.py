"""Native source-eye PBR overrides with reversible artist-state ownership."""
import hashlib
import json
import runpy
import traceback
from datetime import datetime,timezone
from pathlib import Path

TAG='CreateMetaHuman.SourceEyeAppearance'
COLOR='CM Source Eye Color'
FINISH='CM Source Eye Finish'
SPECULAR='CM Source Eye Specular'
ROLES=('eye_left','eye_right')


def digest(path):
    with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()


def _path(value):return value.get_path_name() if value else None


def _other_materials(materials,keys):
    value=materials.copy();entries=dict(value.get_editor_property('teeth_and_eyes'))
    for key in keys.values():entries.pop(key,None)
    value.set_editor_property('teeth_and_eyes',entries)
    return value.export_text()


def _surface(unreal,tools,root,token,textures):
    lib=unreal.MaterialEditingLibrary;name='M_SourceEyes_'+token
    if unreal.EditorAssetLibrary.does_asset_exist(root+'/Materials/'+name):raise ValueError('Source-eye request assets already exist; inspect the original receipt')
    material=tools.create_asset(name,root+'/Materials',unreal.Material,unreal.MaterialFactoryNew())
    if not material:raise RuntimeError('Could not create the source-eye surface')
    for parameter,texture,sampler,outputs in (
        (COLOR,textures['eye_left']['ALBEDO'],unreal.MaterialSamplerType.SAMPLERTYPE_COLOR,[('RGB',unreal.MaterialProperty.MP_BASE_COLOR)]),
        (FINISH,textures['eye_left']['FINISH'],unreal.MaterialSamplerType.SAMPLERTYPE_MASKS,
         [('G',unreal.MaterialProperty.MP_ROUGHNESS),('B',unreal.MaterialProperty.MP_METALLIC)])):
        node=lib.create_material_expression(material,unreal.MaterialExpressionTextureSampleParameter2D)
        node.set_editor_property('parameter_name',parameter);node.set_editor_property('texture',texture);node.set_editor_property('sampler_type',sampler)
        for channel,property in outputs:
            if not lib.connect_material_property(node,channel,property):raise RuntimeError('Could not connect source-eye '+parameter)
    specular=lib.create_material_expression(material,unreal.MaterialExpressionScalarParameter)
    specular.set_editor_property('parameter_name',SPECULAR);specular.set_editor_property('default_value',.5)
    if not lib.connect_material_property(specular,'',unreal.MaterialProperty.MP_SPECULAR):raise RuntimeError('Could not connect source-eye specular')
    lib.recompile_material(material)
    if not unreal.EditorAssetLibrary.save_loaded_asset(material,False):raise RuntimeError('Could not save source-eye surface')
    return material


def update(request_path):
    import unreal
    folder=Path(request_path).resolve().parent;snapshot=folder/'scripts';request=json.loads(Path(request_path).read_text())
    mode=request.get('mode');report=dict(status='preflight',action='UPDATE_TEXTURES',geometry_mode='NATIVE_SOURCE_EYES',
        operation='RESTORE_SOURCE_EYES' if mode=='RESTORE' else 'APPLY_SOURCE_EYES',updated=[],warnings=[],retain_source_snapshot=True,
        requires_reassembly=bool(request.get('existing_assembly')))
    actor=character=subsystem=None;owned=committed=False;old_skin=None;old_tag=''
    def write():
        temporary=folder/'texture-progress.tmp';temporary.write_text(json.dumps(report,indent=2));temporary.replace(folder/'texture-progress.json')
    try:
        if request.get('schema_version')!=1 or request.get('action')!='UPDATE_TEXTURES' or request.get('geometry_mode')!='NATIVE_SOURCE_EYES' or mode not in {'APPLY','RESTORE'}:
            raise ValueError('Unsupported native source-eye request')
        project=Path(unreal.Paths.get_project_file_path()).resolve()
        if not request.get('expected_project') or project!=Path(request['expected_project']).resolve():raise ValueError('Open the linked Unreal project')
        character=unreal.load_asset(request['character'])
        if not isinstance(character,unreal.MetaHumanCharacter):raise ValueError('The linked native MetaHuman Character is missing')
        subsystem=unreal.get_editor_subsystem(unreal.MetaHumanCharacterEditorSubsystem)
        additional=2 if subsystem.is_object_added_for_editing(character) else 8
        runpy.run_path(str(snapshot/'resource_budget.py'))['require']('native source-eye appearance',additional*1024**3)
        library=unreal.CreateMHCharacterLibrary;refresh=getattr(library,'refresh_skin_preview',None)
        if refresh is None:raise ValueError('Install CreateMetaHumanBridge 0.19 or later for native source-eye preview')
        owner=character.get_path_name();root=owner.rsplit('/',1)[0]
        report.update(character=owner,unreal_project_file=str(project))
        before=json.loads(unreal.CreateMHArtistLibrary.snapshot_object(character))
        if any(row['package_dirty'] for row in before['objects']):raise ValueError('Save native artist edits before changing source-eye materials')
        old_tag=unreal.EditorAssetLibrary.get_metadata_tag(character,TAG)
        old_ledger=json.loads(old_tag) if old_tag else None
        if old_ledger and old_ledger.get('owner')!=owner:raise ValueError('Source-eye history belongs to another character')
        if mode=='RESTORE' and not old_ledger:raise ValueError('This character has no previous source-eye material state to restore')
        subsystem=unreal.get_editor_subsystem(unreal.MetaHumanCharacterEditorSubsystem)
        owned=not subsystem.is_object_added_for_editing(character)
        report['status']='initializing-native-character';write()
        if owned and not subsystem.try_add_object_to_edit(character):raise RuntimeError('Creator could not initialize the linked character')
        native=json.loads(library.inspect_character(character))
        if native.get('rig_request_pending') or native.get('texture_request_pending'):raise ValueError('Finish pending native requests before changing source eyes')
        if mode=='APPLY':
            geometry_path=snapshot/'geometry.json'
            if digest(geometry_path)!=request['geometry_sha256']:raise ValueError('Native eye geometry witness changed')
            geometry=json.loads(geometry_path.read_text())['vertices']
            if not {'head','eye_left','eye_right'}<=set(geometry) or set(geometry)-{'head','eye_left','eye_right','teeth'}:raise ValueError('Complete head/eye correspondence is required')
            measured=runpy.run_path(str(snapshot/'unreal_import.py'))['audit_head_conform'](character,geometry,folder/'eye-state.json')
            if measured.get('within_native_precision') is not True:raise ValueError('The native head/eyes differ from the fitted character; retain geometry edits before applying source eyes')
            report['geometry_check']=measured
        helpers=runpy.run_path(str(snapshot/'unreal_revision_textures.py'))
        comparison=runpy.run_path(str(snapshot/'unreal_native_albedo.py'))
        bindings=runpy.run_path(str(snapshot/'unreal_roughness.py'))
        actor=subsystem.spawn_meta_human_actor(character,True)
        if not actor:raise RuntimeError('Creator did not produce a preview actor')
        face=actor.get_editor_property('face_component');body=actor.get_editor_property('body_component')
        components={'head':face,'body':body};before_mesh={role:helpers['mesh_digest'](subsystem,value) for role,value in components.items()}
        slot_names=[str(name) for name in face.get_material_slot_names()]
        slots={role:slot_names.index(name) for role,name in zip(ROLES,('eyeLeft_shader_shader','eyeRight_shader_shader'))}
        keys=dict(zip(ROLES,(unreal.MetaHumanCharacterTeethAndEyesSlot.EYE_LEFT,unreal.MetaHumanCharacterTeethAndEyesSlot.EYE_RIGHT)))
        old_skin=character.get_editor_property('skin_settings').copy();skin=old_skin.copy()
        overrides=skin.get_editor_property('texture_material_overrides').copy()
        materials=overrides.get_editor_property('material_overrides').copy()
        previous=dict(materials.get_editor_property('teeth_and_eyes'));current=dict(previous)
        previous_enabled=bool(overrides.get_editor_property('enable_material_overrides'))
        previous_inherit=bool(overrides.get_editor_property('inherit_ui_params_and_src_textures'))
        others=_other_materials(materials,keys)
        live=bindings['capture'](character)
        if mode=='APPLY':
            dormant=(any(materials.get_editor_property('skin').values()) or bool(materials.get_editor_property('body')) or
                     any(value for key,value in current.items() if key not in keys.values()))
            if not previous_enabled and dormant:raise ValueError('Review existing dormant skin/other material overrides before enabling source-eye overrides; previous artist entries were retained')
            records=request.get('maps',[])
            if {(row.get('role'),row.get('kind')) for row in records}!={(role,kind) for role in ROLES for kind in ('ALBEDO','FINISH')} or len(records)!=4:
                raise ValueError('Source-eye handoff needs color and finish for both eyes')
            source_root=(Path(request['project_dir'])/'EyeColor').resolve();tasks=[]
            for row in records:
                relative=Path(row['snapshot']);filename=(snapshot/relative).resolve();source=Path(row['source']).resolve()
                if (relative.is_absolute() or filename.parent!=(snapshot/'Maps').resolve() or not source.is_relative_to(source_root)
                        or digest(filename)!=row['sha256'] or digest(source)!=row['sha256']):raise ValueError('Source-eye snapshot changed or leaves the character project')
                tasks.append((row,filename))
            report['status']='importing-source-eye-textures';write()
            tools=unreal.AssetToolsHelpers.get_asset_tools();imported={role:{} for role in ROLES};specular={}
            for row,filename in tasks:
                name='SourceEye_'+row['role']+'_'+row['kind']+'_'+folder.name;target=root+'/Maps/'+name+'.'+name
                if unreal.load_asset(target):raise ValueError('Source-eye request assets already exist; inspect the original receipt')
                task=unreal.AssetImportTask()
                for key,value in [('filename',str(filename)),('destination_path',root+'/Maps'),('destination_name',name),('automated',True),('save',False),('replace_existing',False)]:task.set_editor_property(key,value)
                tools.import_asset_tasks([task])
                if list(task.get_editor_property('imported_object_paths'))!=[target]:raise RuntimeError('Unexpected source-eye texture destination')
                texture=unreal.load_asset(target);color=row['kind']=='ALBEDO'
                texture.set_editor_property('srgb',color)
                texture.set_editor_property('compression_settings',unreal.TextureCompressionSettings.TC_BC7 if color else unreal.TextureCompressionSettings.TC_MASKS)
                imports=texture.get_editor_property('asset_import_data');imports.scripted_add_filename(str(source_root/Path(row['source']).resolve().relative_to(source_root)),0,'')
                if Path(imports.extract_filenames()[0]).resolve()!=Path(row['source']).resolve():raise RuntimeError('Persistent source-eye Reimport path was not retained')
                unreal.EditorAssetLibrary.set_metadata_tag(texture,'CreateMetaHuman.TextureOwner',owner)
                imported[row['role']][row['kind']]=texture;specular[row['role']]=float(row['specular'])
                report['updated'].append(dict(role=row['role'],kind=row['kind'],asset=target,source=row['source'],sha256=row['sha256']))
            parent=_surface(unreal,tools,root,folder.name,imported);selected={}
            for role in ROLES:
                name='MI_SourceEyes_'+role+'_'+folder.name
                material=tools.create_asset(name,root+'/Materials',unreal.MaterialInstanceConstant,unreal.MaterialInstanceConstantFactoryNew())
                if not material:raise RuntimeError('Could not create a source-eye material instance')
                lib=unreal.MaterialEditingLibrary;lib.set_material_instance_parent(material,parent)
                lib.set_material_instance_texture_parameter_value(material,COLOR,imported[role]['ALBEDO'])
                lib.set_material_instance_texture_parameter_value(material,FINISH,imported[role]['FINISH'])
                lib.set_material_instance_scalar_parameter_value(material,SPECULAR,specular[role])
                unreal.EditorAssetLibrary.set_metadata_tag(material,'CreateMetaHuman.SourceEyeOwner',owner)
                if not unreal.EditorAssetLibrary.save_loaded_asset(material,False):raise RuntimeError('Could not save source-eye material')
                current[keys[role]]=material;selected[role]=material
            ledger=dict(schema=1,owner=owner,previous_enabled=previous_enabled,previous_inherit=previous_inherit,other_materials=others,previous_tag=old_tag,
                previous={role:dict(present=keys[role] in previous,asset=_path(previous.get(keys[role]))) for role in ROLES},
                applied={role:material.get_path_name() for role,material in selected.items()},receipt=str(folder/'receipt.json'))
            overrides.set_editor_property('enable_material_overrides',True)
            if not previous_enabled:
                # Previously inactive material overrides implicitly inherited
                # native skin UI/source textures. Keep that effective behavior
                # when enabling only our two eye entries.
                overrides.set_editor_property('inherit_ui_params_and_src_textures',True)
            new_tag=json.dumps(ledger)
        else:
            if old_ledger.get('schema')!=1 or any(_path(current.get(keys[role]))!=old_ledger['applied'].get(role) for role in ROLES):
                raise ValueError('Source-eye overrides were edited in Unreal; retain those eye edits before restoring the previous state')
            enabled=previous_enabled
            if previous_enabled and not old_ledger['previous_enabled'] and others==old_ledger['other_materials']:enabled=False
            dormant_only=not old_ledger['previous_enabled'] and enabled
            for role in ROLES:
                original=old_ledger['previous'][role]
                if original['present'] and not dormant_only:
                    asset=unreal.load_asset(original['asset']) if original['asset'] else None
                    if original['asset'] and not asset:raise ValueError('Previous native eye material is missing: '+original['asset'])
                    current[keys[role]]=asset
                else:current.pop(keys[role],None)
            if dormant_only:report['warnings'].append('Other material overrides changed after source-eye application; they remain enabled. Previous dormant eye entries are retained in this receipt.')
            overrides.set_editor_property('enable_material_overrides',enabled)
            if not enabled and previous_enabled and not old_ledger['previous_enabled'] and others==old_ledger['other_materials']:
                overrides.set_editor_property('inherit_ui_params_and_src_textures',old_ledger['previous_inherit'])
            new_tag=old_ledger.get('previous_tag','');report['restored_eye_materials']=old_ledger['previous']
        materials.set_editor_property('teeth_and_eyes',current)
        overrides.set_editor_property('material_overrides',materials);skin.set_editor_property('texture_material_overrides',overrides)
        committed=True;subsystem.commit_skin_settings(character,skin)
        if not refresh(character):raise RuntimeError('Native source-eye preview did not refresh')
        # Preserve every live component parameter outside the selected eye pair.
        bindings['restore']([row for row in live if not (row[0]==face and row[1] in slots.values())])
        actual=character.get_editor_property('skin_settings').copy();actual_override=actual.get_editor_property('texture_material_overrides').copy()
        actual_materials=actual_override.get_editor_property('material_overrides').copy();values=dict(actual_materials.get_editor_property('teeth_and_eyes'))
        if any(values.get(keys[role])!=current.get(keys[role]) or (keys[role] in values)!=(keys[role] in current) for role in ROLES):raise RuntimeError('Native source-eye material bindings were not retained')
        normalized=dict(values)
        for role in ROLES:
            if keys[role] in previous:normalized[keys[role]]=previous[keys[role]]
            else:normalized.pop(keys[role],None)
        actual_materials.set_editor_property('teeth_and_eyes',normalized);actual_override.set_editor_property('material_overrides',actual_materials)
        actual_override.set_editor_property('enable_material_overrides',previous_enabled)
        actual_override.set_editor_property('inherit_ui_params_and_src_textures',previous_inherit)
        actual.set_editor_property('texture_material_overrides',actual_override)
        if comparison['_artist_skin'](actual)!=comparison['_artist_skin'](old_skin):raise RuntimeError('An unrelated artist skin/material setting changed')
        if {role:helpers['mesh_digest'](subsystem,value) for role,value in components.items()}!=before_mesh:raise RuntimeError('Native geometry changed during source-eye application')
        after_native=json.loads(library.inspect_character(character))
        if any(after_native.get(key)!=native.get(key) for key in ('rig_state','has_face_dna','has_body_dna','rig_request_pending','texture_request_pending')):raise RuntimeError('The native rig or request state changed')
        after=json.loads(unreal.CreateMHArtistLibrary.snapshot_object(character))
        def properties(data):return next(row['properties'] for row in data['objects'] if row['object']==owner)
        a,b=properties(before),properties(after)
        changed=[key for key in a if key!='SkinSettings' and a[key]['text']!=b.get(key,{}).get('text')]
        if changed:raise RuntimeError('Unrelated persistent native settings changed: '+', '.join(changed))
        if mode=='APPLY':
            verified={}
            for role,index in slots.items():
                material=face.get_material(index)
                if not material or material.get_texture_parameter_value(COLOR)!=imported[role]['ALBEDO'] or material.get_texture_parameter_value(FINISH)!=imported[role]['FINISH']:
                    raise RuntimeError('Evaluated native '+role+' does not use projected color/finish')
                if abs(material.get_scalar_parameter_value(SPECULAR)-specular[role])>1e-6:raise RuntimeError('Native eye specular differs from local PBR')
                verified[role]=dict(slot=index,material=selected[role].get_path_name(),specular=specular[role])
            report['evaluated_eye_slots']=verified
            for maps in imported.values():
                for texture in maps.values():
                    if not unreal.EditorAssetLibrary.save_loaded_asset(texture,False):raise RuntimeError('Could not save projected native eye texture')
        unreal.EditorAssetLibrary.set_metadata_tag(character,TAG,new_tag)
        if not unreal.EditorAssetLibrary.save_loaded_asset(character,False):raise RuntimeError('Could not save native source-eye bindings')
        active=False
        if new_tag:
            restored_ledger=json.loads(new_tag)
            stored=character.get_editor_property('skin_settings').get_editor_property('texture_material_overrides')
            eye_values=dict(stored.get_editor_property('material_overrides').get_editor_property('teeth_and_eyes'))
            active=bool(stored.get_editor_property('enable_material_overrides')) and restored_ledger.get('owner')==owner and all(
                _path(eye_values.get(keys[role]))==restored_ledger.get('applied',{}).get(role) for role in ROLES)
        report.update(status='complete',retain_source_snapshot=False,geometry_preserved=True,rig_preserved=True,receipt=str(folder/'receipt.json'),
            unrelated_artist_settings_preserved=True,unrelated_skin_materials_preserved=True,
            source_eye_override_active=active,
            surface_model='Opaque PBR source eyes; native shell/tearline retained',assembly=after_native)
    except Exception as exc:
        report.update(status='failed',error=str(exc),traceback=traceback.format_exc())
        if committed and old_skin is not None:
            try:
                subsystem.commit_skin_settings(character,old_skin)
                if not refresh(character):raise RuntimeError('Could not refresh restored eye materials')
                bindings['restore'](live);unreal.EditorAssetLibrary.set_metadata_tag(character,TAG,old_tag)
                report['previous_eye_state_restored']=bool(unreal.EditorAssetLibrary.save_loaded_asset(character,False))
            except Exception as restore_error:report['restore_error']=str(restore_error)
    finally:
        if actor:unreal.get_editor_subsystem(unreal.EditorActorSubsystem).destroy_actor(actor)
        if owned and character and subsystem:subsystem.remove_object_to_edit(character)
        report['completed_at']=datetime.now(timezone.utc).isoformat();write()
    return report
