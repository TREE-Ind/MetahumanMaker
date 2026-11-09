"""Refresh native head/body detail and finish without a legacy mesh inventory."""
import hashlib,json,runpy,traceback
from datetime import datetime,timezone
from pathlib import Path

PARAMS={'NORMAL':'Normal','CAVITY':'Cavity','ROUGHNESS':'Roughness','BENT_NORMAL_AO':'Bent Normal'}


def digest(path):
    with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()


def update(request_path):
    import unreal
    folder=Path(request_path).resolve().parent;snapshot=folder/'scripts'
    request=json.loads(Path(request_path).read_text(encoding='utf-8'))
    report=dict(status='preflight',action='UPDATE_TEXTURES',geometry_mode='NATIVE_SKIN_MAPS',updated=[],
                warnings=[],retain_source_snapshot=True,requires_reassembly=bool(request.get('existing_assembly')))
    actor=character=subsystem=None;owned=committed=False;old_skin=old_preview=None;live=[]
    assets=unreal.EditorAssetLibrary;lib=unreal.MaterialEditingLibrary
    bindings=runpy.run_path(str(snapshot/'unreal_roughness.py'))
    def write():
        temp=folder/'texture-progress.tmp';temp.write_text(json.dumps(report,indent=2),encoding='utf-8');temp.replace(folder/'texture-progress.json')
    try:
        if request.get('schema_version')!=1 or request.get('action')!='UPDATE_TEXTURES' or request.get('geometry_mode')!='NATIVE_SKIN_MAPS':raise ValueError('Unsupported native skin-map request')
        project=Path(unreal.Paths.get_project_file_path()).resolve()
        if not request.get('expected_project') or project!=Path(request['expected_project']).resolve():raise ValueError('Open the linked native character project')
        budget=runpy.run_path(str(snapshot/'resource_budget.py'))
        budget['require']('native skin-map application',8*budget['GIB'])
        budget['require_disk']('native skin-map import',unreal.Paths.project_content_dir(),
            sum(int(item['resolution'][0])*int(item['resolution'][1])*8 for item in request['maps'])+2*budget['GIB'])
        character=unreal.load_asset(request['character'])
        if not isinstance(character,unreal.MetaHumanCharacter):raise ValueError('The linked asset is not a native MetaHuman Character')
        refresh=getattr(unreal.CreateMHCharacterLibrary,'refresh_skin_preview',None)
        if refresh is None:raise ValueError('Install CreateMetaHumanBridge 0.19 or later')
        report.update(character=character.get_path_name(),unreal_project_file=str(project))
        before=json.loads(unreal.CreateMHArtistLibrary.snapshot_object(character))
        if any(row['package_dirty'] for row in before['objects']):raise ValueError('Save UE artist edits before refreshing native maps')
        (folder/'artist-before.json').write_text(json.dumps(before,indent=2),encoding='utf-8')
        geometry_file=snapshot/'geometry.json'
        if digest(geometry_file)!=request['geometry_sha256']:raise ValueError('The prepared native geometry changed')
        geometry=json.loads(geometry_file.read_text())['vertices']
        if not geometry or set(geometry)-{'head','eye_left','eye_right','teeth','body'}:raise ValueError('Invalid native geometry witness')
        tasks=[];selected={}
        for item in request['maps']:
            role,kind=item.get('role'),item.get('kind')
            if role not in {'HEAD','BODY'} or kind not in PARAMS or kind in selected.get(role,set()):raise ValueError('Select unique supported native map channels')
            selected.setdefault(role,set()).add(kind)
            relative=Path(item['snapshot']);filename=(snapshot/relative).resolve();source=Path(item['source']).resolve()
            if (relative.is_absolute() or filename.parent!=(snapshot/'Maps').resolve() or source.parent!=(Path(request['project_dir'])/'Maps').resolve()
                    or source.suffix.lower()!='.png' or digest(filename)!=item['sha256'] or digest(source)!=item['sha256']):raise ValueError('A native map or its accepted source changed')
            tasks.append((item,filename))
        if not tasks or any(not geometry.get('head' if role=='HEAD' else 'body') for role in selected):raise ValueError('Selected native maps need complete fitted geometry')
        subsystem=unreal.get_editor_subsystem(unreal.MetaHumanCharacterEditorSubsystem)
        owned=not subsystem.is_object_added_for_editing(character);report['status']='initializing-native-character';write()
        if owned and not subsystem.try_add_object_to_edit(character):raise RuntimeError('Creator could not initialize this character')
        state=json.loads(unreal.CreateMHCharacterLibrary.inspect_character(character))
        if state.get('rig_request_pending') or state.get('texture_request_pending'):raise ValueError('Finish pending native requests before updating maps')
        head={key:value for key,value in geometry.items() if key!='body'}
        if head:
            measured=runpy.run_path(str(snapshot/'unreal_import.py'))['audit_head_conform'](character,head,folder/'maps-head-state.json')
            if measured.get('within_native_precision') is not True:raise ValueError('Native head differs from the prepared fitted geometry')
            report['head_geometry_check']=measured
        helpers=runpy.run_path(str(snapshot/'unreal_revision_textures.py'))
        actor=subsystem.spawn_meta_human_actor(character,True)
        if not actor:raise RuntimeError('Creator could not inspect native preview geometry')
        components={role:actor.get_editor_property(name) for role,name in (('HEAD','face_component'),('BODY','body_component'))}
        if 'body' in geometry:
            points,_=subsystem.get_mesh_data_for_conforming(components['BODY'].get_skeletal_mesh_asset())
            report['body_geometry_check']=helpers['coverage']({'body':geometry['body']},points)
        report['geometry_before']={role:helpers['mesh_digest'](subsystem,component) for role,component in components.items()}
        old_skin=character.get_editor_property('skin_settings').copy();old_preview=character.get_editor_property('preview_material_type')
        skin=old_skin.copy();overrides=skin.get_editor_property('texture_material_overrides').copy()
        old_overrides=old_skin.get_editor_property('texture_material_overrides')
        texture_set=overrides.get_editor_property('texture_overrides').copy();materials=overrides.get_editor_property('material_overrides').copy()
        faces=dict(materials.get_editor_property('skin'));old_faces=dict(faces);old_body=materials.get_editor_property('body')
        active=bool(overrides.get_editor_property('enable_material_overrides'))
        if not active and any('ROUGHNESS' in kinds for kinds in selected.values()) and (any(faces.values()) or old_body or materials.get_editor_property('teeth_and_eyes')):
            raise ValueError('Review disabled artist material overrides before enabling projected roughness')
        live=bindings['capture'](character);root=character.get_path_name().rsplit('/',1)[0];owner=character.get_path_name()
        tools=unreal.AssetToolsHelpers.get_asset_tools();plans=[]
        for role,kinds in selected.items():
            entries=list(faces.items()) if role=='HEAD' else [(None,old_body)]
            if 'ROUGHNESS' in kinds:
                entries=([(getattr(unreal.MetaHumanCharacterSkinMaterialSlot,slot),faces.get(getattr(unreal.MetaHumanCharacterSkinMaterialSlot,slot)))
                    for slot in ('LOD0','LOD1','LOD2','LOD3','LOD4','LOD5TO7')] if role=='HEAD' else [(None,old_body)])
            for slot,current in entries:
                material=current if active else None
                if not material and 'ROUGHNESS' not in kinds:continue
                if not material:
                    label=str(slot).split('.')[1].split(':')[0] if role=='HEAD' else 'BODY'
                    parent_name='MI_Skin_Body_MHC' if role=='BODY' else 'MI_Skin_Head_UI_'+('LOD57' if label=='LOD5TO7' else label)
                    material=unreal.load_asset('/MetaHumanCharacter/Materials/'+parent_name)
                names={str(n) for n in lib.get_texture_parameter_names(material)} if isinstance(material,unreal.MaterialInstanceConstant) else set()
                if not {'Basecolor','Micro Skin Details Normal',*[PARAMS[k] for k in kinds]}<=names:
                    raise ValueError('The artist '+role.lower()+' material needs a native skin-map adapter; it was retained')
                if 'ROUGHNESS' in kinds and 'Use Roughness Map as Base' not in {str(n) for n in lib.get_static_switch_parameter_names(material)}:
                    raise ValueError('The native skin material cannot activate a projected roughness map')
                plans.append((role,slot,material))
        imported={};report['status']='importing';write()
        for item,filename in tasks:
            name=filename.stem+'_Native_'+folder.name;path=root+'/Maps/'+name+'.'+name
            if assets.does_asset_exist(path):raise ValueError('This action already owns native map assets; inspect its original receipt')
            task=unreal.AssetImportTask()
            for key,value in [('filename',str(filename)),('destination_path',root+'/Maps'),('destination_name',name),('automated',True),('save',False),('replace_existing',False)]:task.set_editor_property(key,value)
            tools.import_asset_tasks([task])
            if list(task.get_editor_property('imported_object_paths'))!=[path]:raise RuntimeError('Unexpected native texture import result')
            texture=unreal.load_asset(path)
            if not isinstance(texture,unreal.Texture2D):raise RuntimeError('Native map import did not produce a Texture2D')
            texture.set_editor_property('srgb',False)
            texture.set_editor_property('compression_settings',unreal.TextureCompressionSettings.TC_NORMALMAP if item['kind']=='NORMAL' else unreal.TextureCompressionSettings.TC_MASKS)
            if item['kind']=='NORMAL':texture.set_editor_property('flip_green_channel',False)
            if digest(item['source'])!=item['sha256']:raise ValueError('Accepted map changed during import')
            texture.get_editor_property('asset_import_data').scripted_add_filename(item['source'],0,'')
            assets.set_metadata_tag(texture,'CreateMetaHuman.TextureOwner',owner)
            imported[(item['role'],item['kind'])]=texture
            report['updated'].append(dict(role=item['role'],kind=item['kind'],asset=path,source=item['source'],sha256=item['sha256']))
        private_slots=[]
        for index,(role,slot,material) in enumerate(plans):
            name='NativeMaps_'+folder.name+'_'+role+'_'+str(index)
            if assets.does_asset_exist(root+'/Materials/'+name):raise ValueError('This action already owns its skin material')
            private=tools.duplicate_asset(name,root+'/Materials',material)
            if not private:raise RuntimeError('Could not copy the artist skin material')
            for tag,value in assets.get_metadata_tag_values(material).items():assets.set_metadata_tag(private,str(tag),value)
            if assets.get_metadata_tag(private,'CreateMetaHuman.RoughnessOwner'):
                assets.set_metadata_tag(private,'CreateMetaHuman.RoughnessOwner',owner)
            # Preserve live artist texture/scalar/vector arrays before adding
            # selected channels to an instance based on the stock UI parent.
            for _,_,source,_,_,_ in live:
                if source.get_editor_property('parent')!=material:continue
                for field in ('texture_parameter_values','scalar_parameter_values','vector_parameter_values'):private.set_editor_property(field,source.get_editor_property(field))
                break
            assets.set_metadata_tag(private,'CreateMetaHuman.CharacterOwner',owner)
            for kind in selected[role]:lib.set_material_instance_texture_parameter_value(private,PARAMS[kind],imported[(role,kind)])
            if 'ROUGHNESS' in selected[role]:
                response=request.get('roughness_response',{}).get(role,{'mode':'ABSOLUTE'})
                if response.get('mode')=='SKIN_DETAIL' and response.get('finish_roughness') is not None:
                    parent=private.get_editor_property('parent')
                    if assets.get_metadata_tag(parent,'CreateMetaHuman.ResponseOwner')!=owner:
                        # Keep response-graph copies within this action; an
                        # upgrade must not edit other artist skin instances.
                        parent=bindings['response_parent'](parent,root+'/Materials/NativeMaps_'+folder.name,owner)
                    lib.set_material_instance_parent(private,parent)
                lib.set_material_instance_static_switch_parameter_value(private,'Use Roughness Map as Base',True)
                if not lib.get_material_instance_static_switch_parameter_value(private,'Use Roughness Map as Base'):raise RuntimeError('Projected roughness did not activate')
                calibration=bindings['calibrate_response'](private,response)
                if 'CM Roughness Response Exponent' in {str(n) for n in lib.get_scalar_parameter_names(private)}:
                    lib.set_material_instance_scalar_parameter_value(private,'CM Roughness Response Exponent',calibration['ui_response_exponent'])
                assets.set_metadata_tag(private,'CreateMetaHuman.RoughnessOwner',owner)
                report.setdefault('roughness_response',{}).setdefault(role,[]).append(calibration)
            if not assets.save_loaded_asset(private,False):raise RuntimeError('Could not save the private skin instance')
            if role=='HEAD':faces[slot]=private
            else:materials.set_editor_property('body',private)
            private_slots.append((role,slot,private))
        materials.set_editor_property('skin',faces)
        for (role,kind),texture in imported.items():
            if kind in {'ROUGHNESS','BENT_NORMAL_AO'}:continue
            field='face' if role=='HEAD' else 'body';values=dict(texture_set.get_editor_property(field))
            enum=getattr(unreal.FaceTextureType,kind) if role=='HEAD' else getattr(unreal.BodyTextureType,'BODY_'+kind)
            values[enum]=texture;texture_set.set_editor_property(field,values)
        overrides.set_editor_property('texture_overrides',texture_set);overrides.set_editor_property('material_overrides',materials)
        if any(kind!='ROUGHNESS' for role,kind in imported):overrides.set_editor_property('enable_texture_overrides',True)
        if private_slots:
            overrides.set_editor_property('enable_material_overrides',True)
            if not active:overrides.set_editor_property('inherit_ui_params_and_src_textures',True)
        skin.set_editor_property('texture_material_overrides',overrides)
        committed=True;subsystem.commit_skin_settings(character,skin)
        character.set_editor_property('preview_material_type',unreal.MetaHumanCharacterSkinPreviewMaterial.EDITABLE)
        if not refresh(character):raise RuntimeError('Could not refresh native skin preview')
        def changed(component,current,key):
            role='BODY' if component.get_name()=='Body' else 'HEAD'
            names={str(n) for n in lib.get_texture_parameter_names(current)}
            return 'Micro Skin Details Normal' in names and key in {PARAMS[k] for k in selected.get(role,set())}
        bindings['restore'](live,changed,bindings['managed_scalar_skip'](character,{role for role,kinds in selected.items() if 'ROUGHNESS' in kinds}))
        verified={}
        for role,kinds in selected.items():
            slots=[];component=components[role]
            for index in range(component.get_num_materials()):
                material=component.get_material(index)
                if not material or 'Micro Skin Details Normal' not in {str(n) for n in lib.get_texture_parameter_names(material)}:continue
                for kind in kinds:
                    if material.get_texture_parameter_value(PARAMS[kind])!=imported[(role,kind)]:raise RuntimeError('Evaluated '+role.lower()+' skin did not use '+kind.lower())
                slots.append(index)
            if not slots:raise RuntimeError('No evaluated '+role.lower()+' skin material was verified')
            verified[role]=slots
        actual=character.get_editor_property('skin_settings').copy();normalized=actual.get_editor_property('texture_material_overrides').copy()
        norm_textures=normalized.get_editor_property('texture_overrides').copy()
        for role,kinds in selected.items():
            field='face' if role=='HEAD' else 'body';values=dict(norm_textures.get_editor_property(field));old=dict(old_overrides.get_editor_property('texture_overrides').get_editor_property(field))
            for kind in kinds-{'ROUGHNESS','BENT_NORMAL_AO'}:
                enum=getattr(unreal.FaceTextureType,kind) if role=='HEAD' else getattr(unreal.BodyTextureType,'BODY_'+kind)
                if values.get(enum)!=imported[(role,kind)]:raise RuntimeError('A native texture override was not retained')
                if enum in old:values[enum]=old[enum]
                else:values.pop(enum,None)
            norm_textures.set_editor_property(field,values)
        normalized.set_editor_property('texture_overrides',norm_textures)
        norm_materials=normalized.get_editor_property('material_overrides').copy();norm_faces=dict(norm_materials.get_editor_property('skin'))
        for role,slot,private in private_slots:
            if role=='HEAD':
                if norm_faces.get(slot)!=private:raise RuntimeError('A private native head material was not retained')
                if slot in old_faces:norm_faces[slot]=old_faces[slot]
                else:norm_faces.pop(slot,None)
            else:
                if norm_materials.get_editor_property('body')!=private:raise RuntimeError('A private native body material was not retained')
                norm_materials.set_editor_property('body',old_body)
        norm_materials.set_editor_property('skin',norm_faces);normalized.set_editor_property('material_overrides',norm_materials)
        for flag in ('enable_texture_overrides','enable_material_overrides','inherit_ui_params_and_src_textures'):normalized.set_editor_property(flag,old_overrides.get_editor_property(flag))
        if normalized.export_text()!=old_overrides.export_text():raise RuntimeError('An unrelated native texture, eye or material override changed')
        actual.set_editor_property('texture_material_overrides',old_overrides)
        artist=runpy.run_path(str(snapshot/'unreal_native_albedo.py'))['_artist_skin']
        if artist(actual)!=artist(old_skin):raise RuntimeError('Unrelated native skin settings changed')
        report['geometry_after']={role:helpers['mesh_digest'](subsystem,component) for role,component in components.items()}
        if report['geometry_after']!=report['geometry_before']:raise RuntimeError('Geometry changed during texture-only refresh')
        after_state=json.loads(unreal.CreateMHCharacterLibrary.inspect_character(character))
        if 'ready' in after_state and 'rig_state' in after_state:after_state['detailed_status_available']=True
        if any(after_state.get(k)!=state.get(k) for k in ('rig_state','has_face_dna','has_body_dna','rig_request_pending','texture_request_pending','has_texture_sources')):raise RuntimeError('Native rig or source state changed')
        character.set_editor_property('preview_material_type',old_preview)
        if not refresh(character):raise RuntimeError('Could not restore the artist preview mode')
        bindings['restore'](live,changed,bindings['managed_scalar_skip'](character,{r for r,k in selected.items() if 'ROUGHNESS' in k}))
        after=json.loads(unreal.CreateMHArtistLibrary.snapshot_object(character));(folder/'artist-after.json').write_text(json.dumps(after,indent=2),encoding='utf-8')
        props=lambda data:next(row['properties'] for row in data['objects'] if row['object']==owner)
        a,b=props(before),props(after)
        changes=[key for key in a if key!='SkinSettings' and a[key]['text']!=b.get(key,{}).get('text')]
        if changes:raise RuntimeError('Unrelated native artist settings changed: '+', '.join(changes))
        for texture in imported.values():
            if not assets.save_loaded_asset(texture,False):raise RuntimeError('Could not save a native texture')
        if not assets.save_loaded_asset(character,False):raise RuntimeError('Could not save native skin-map bindings')
        report.update(status='complete',retain_source_snapshot=False,geometry_preserved=True,rig_preserved=True,
            unrelated_artist_settings_preserved=True,unselected_overrides_preserved=True,evaluated_skin_slots=verified,assembly=after_state)
    except Exception as exc:
        report.update(status='failed',error=str(exc),traceback=traceback.format_exc())
        if committed and old_skin is not None:
            try:
                subsystem.commit_skin_settings(character,old_skin);character.set_editor_property('preview_material_type',old_preview)
                if not refresh(character):raise RuntimeError('Could not refresh restored native preview')
                bindings['restore'](live)
                report['skin_bindings_restored']=bool(assets.save_loaded_asset(character,False))
            except Exception as error:report['restore_error']=str(error)
    finally:
        try:
            if actor:unreal.get_editor_subsystem(unreal.EditorActorSubsystem).destroy_actor(actor)
            if owned and character and subsystem:subsystem.remove_object_to_edit(character)
        except Exception as error:report.setdefault('warnings',[]).append('Preview cleanup: '+str(error))
        report['completed_at']=datetime.now(timezone.utc).isoformat();write()
    return report
