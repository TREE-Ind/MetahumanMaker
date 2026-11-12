"""Local native character assembly. No rigging or texture-service requests."""
from pathlib import Path
from datetime import datetime,timezone
import ctypes
import json
import re
import shutil

GIB=1024**3
EXPORT_QUALITY_TAG='MHExportQuality'


def budgets(content):
    class MemoryStatus(ctypes.Structure):
        _fields_=[('length',ctypes.c_ulong),('load',ctypes.c_ulong),
                  *[(name,ctypes.c_ulonglong) for name in ('total','available','page_total','page_available','virtual_total','virtual_available','extended')]]
    memory=MemoryStatus();memory.length=ctypes.sizeof(memory)
    if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(memory)):
        raise RuntimeError('Could not read available system memory before assembly')
    # WindowsPlatformMemory::GetStats clamps AvailableVirtual to available
    # page-file commit. Creator checks that value, not AvailablePhysical.
    return {'free_disk_bytes':shutil.disk_usage(content).free,'available_memory_bytes':memory.available,
            'available_commit_bytes':min(memory.page_available,memory.virtual_available),
            'available_address_space_bytes':memory.virtual_available,
            'minimum_commit_bytes':2*GIB,'recommended_commit_bytes':10*GIB,
            'initialization_reserve_bytes':8*GIB,'disk_reserve_bytes':10*GIB}


def commit_requirement(limits,editing,allow_low_memory=False):
    """Keep a native build reserve after acquiring a cold editing session."""
    floor=limits['minimum_commit_bytes']
    build=floor if allow_low_memory else limits['recommended_commit_bytes']
    return build+(0 if editing else limits['initialization_reserve_bytes'])


def resource_status(content,editing):
    """Cheap recommendation for status panels; never initializes a character."""
    limits=budgets(content)
    required=commit_requirement(limits,editing)
    return dict(limits,required_commit_bytes=required,editing_session_reused=bool(editing),
        recommendation_met=limits['available_commit_bytes']>=required and limits['free_disk_bytes']>=limits['disk_reserve_bytes'])


def resources_allow(status,allow_low_memory=False):
    """Use cached diagnostics for the button; the real action always rechecks."""
    if not all(key in status for key in ('minimum_commit_bytes','recommended_commit_bytes','initialization_reserve_bytes','available_commit_bytes','free_disk_bytes','disk_reserve_bytes')):
        return True  # Older receipts have no resource observation.
    required=commit_requirement(status,status.get('editing_session_reused',False),allow_low_memory)
    return status['available_commit_bytes']>=required and status['free_disk_bytes']>=status['disk_reserve_bytes']


def options(value):
    root=value.get('root','/Game/MetaHumans').rstrip('/')
    if not re.fullmatch(r'/Game(?:/[A-Za-z0-9_]+)+',root):
        raise ValueError('Choose a build folder under /Game using letters, numbers and underscores')
    name=re.sub(r'[^A-Za-z0-9_]','_',value.get('name','')).strip('_')
    if not name or len(name)>80:raise ValueError('Use a character build name between 1 and 80 letters, numbers or underscores')
    profile=value.get('profile','CINEMATIC')
    if profile not in {'CINEMATIC','HIGH','MEDIUM','LOW'}:raise ValueError('Unknown MetaHuman assembly profile')
    if type(value.get('preserve_source_textures',True)) is not bool:raise ValueError('Choose a supported source texture resolution policy')
    if type(value.get('allow_low_memory',False)) is not bool:raise ValueError('Low-memory assembly must be an explicit boolean choice')
    if value.get('eye_overlay_policy','NATIVE') not in {'NATIVE','OPAQUE_SOURCE'}:raise ValueError('Choose a supported assembled eye-overlay policy')
    existing=value.get('existing_blueprint','')
    if existing:
        if not isinstance(existing,str):raise ValueError('Choose a saved assembled Blueprint to update')
        target,separator,object_name=existing.rpartition('.')
        folder,_,asset_name=target.rpartition('/')
        parent,_,chosen=folder.rpartition('/')
        if (not separator or parent!=root or asset_name!='BP_'+chosen or object_name!=asset_name or
                not re.fullmatch(re.escape(name)+r'(?:_v\d{3,})?',chosen)):
            raise ValueError('The existing assembly must match this character name and build folder')
    return root,name,profile


def linked_blueprint(value,receipt,character):
    """Reuse only the completed assembly linked to this exact Creator asset."""
    if (receipt.get('status')!='complete' or receipt.get('character')!=character or
            receipt.get('build',{}).get('state')!='COMPLETE'):return ''
    candidate=receipt['build'].get('blueprint','')
    if not candidate:return ''
    proposed=dict(value,existing_blueprint=candidate)
    try:options(proposed)
    except ValueError:return ''
    return candidate


def actor_meshes(blueprint):
    """Read assembled component templates without spawning or editing actors."""
    import unreal
    subsystem=unreal.get_engine_subsystem(unreal.SubobjectDataSubsystem)
    library=unreal.SubobjectDataBlueprintFunctionLibrary
    components={}
    for handle in subsystem.k2_gather_subobject_data_for_blueprint(blueprint):
        component=library.get_object(library.get_data(handle))
        if not isinstance(component,unreal.SkeletalMeshComponent):continue
        mesh=component.get_skeletal_mesh_asset()
        if not mesh:continue
        name=component.get_name().removesuffix('_GEN_VARIABLE')
        paths=components.setdefault(name,set());paths.add(mesh.get_path_name())
    # Unreal can return the same inherited component twice. A genuinely
    # conflicting component name is not a safe basis for an automatic binding.
    return {name:next(iter(paths)) for name,paths in components.items() if len(paths)==1}


def selected_grooms(character,output_folder):
    """Read active wardrobe selections before native assembly replaces assets.

    Stored selections below an unselected parent are deliberately ignored.
    Source bindings and rest meshes must survive replacement of the output.
    This function does not modify or save any assets.
    """
    import unreal
    collection=character.get_editor_property('internal_collection')
    if not collection:raise ValueError('The character has no wardrobe collection to inspect')
    instance=collection.get_editor_property('default_instance')
    if not instance:raise ValueError('The wardrobe collection has no default instance')
    pending=[]
    for entry in instance.get_slot_selection_data():
        selection=entry.get_editor_property('selection')
        parent=selection.get_editor_property('parent_item_path')
        keys=list(parent.get_editor_property('parent_items'))
        last=parent.get_editor_property('item')
        if not last.is_null():keys.append(last)
        pending.append((tuple(key.export_text() for key in keys),selection))
    containers={():collection};records=[];issues=[];ignored=0
    prefix=output_folder.rstrip('/').casefold()+'/'
    def inside(asset):return asset.get_path_name().split('.')[0].casefold().startswith(prefix)
    while pending:
        deferred=[];advanced=False
        for parent,selection in pending:
            if parent not in containers:deferred.append((parent,selection));continue
            advanced=True;palette=containers[parent];key=selection.get_editor_property('selected_item')
            if key.is_null():continue
            slot=str(selection.get_editor_property('slot_name'))
            # Python USTRUCT wrappers do not implement native key equality.
            if not any(key.export_text()==candidate.export_text() for candidate in palette.get_all_item_keys()):
                issues.append(slot+': selected wardrobe entry is missing');continue
            value=key.get_editor_property('principal_asset_or_external_wardrobe_item').get_editor_property('asset')
            if not value:
                issues.append(slot+': selected wardrobe asset is missing');continue
            principal=value
            if isinstance(value,unreal.MetaHumanWardrobeItem):
                containers[parent+(key.export_text(),)]=value
                principal=value.get_editor_property('principal_asset').get_editor_property('asset')
                if not principal:
                    issues.append(slot+': selected wardrobe principal asset is missing');continue
            if not isinstance(principal,unreal.GroomBindingAsset):continue
            groom=principal.get_editor_property('groom')
            source=principal.get_editor_property('source_skeletal_mesh')
            target=principal.get_editor_property('target_skeletal_mesh')
            row=dict(slot=slot,selection=key.export_text(),parent=list(parent),wardrobe=value.get_path_name(),
                     binding=principal.get_path_name(),groom=groom.get_path_name() if groom else '',
                     source=source.get_path_name() if source else '',target=target.get_path_name() if target else '')
            records.append(row)
            if not isinstance(groom,unreal.GroomAsset) or not source or not target:
                issues.append(slot+': groom binding needs a groom, source rest mesh and target mesh')
            elif any(inside(asset) for asset in (value,principal,groom,source,target)):
                issues.append(slot+': groom input is inside the assembly output; keep its wardrobe item, groom, binding and rest meshes outside '+output_folder)
        if not advanced:
            ignored=len(deferred);break
        pending=deferred
    fingerprint=getattr(getattr(unreal,'CreateMHGroomLibrary',None),'fingerprint_groom',None)
    compare=getattr(getattr(unreal,'CreateMHGroomLibrary',None),'compare_groom_groups',None)
    if records and (not callable(fingerprint) or not callable(compare)):
        issues.append('Install the current CreateMetaHumanBridge with groom fingerprints before assembly')
    elif records:
        fingerprints={}
        for row in records:
            if not row['groom']:continue
            if row['groom'] not in fingerprints:
                fingerprints[row['groom']]=json.loads(fingerprint(unreal.load_asset(row['groom'])))
            data=fingerprints[row['groom']]
            if data.get('schema')!=1 or data.get('status')!='complete' or not data.get('groups'):
                issues.append(row['slot']+': groom source data could not be fingerprinted')
            else:row['geometry']=data['groups']
    return dict(grooms=records,issues=issues,inactive_selections=ignored,
                scope='Active selected skeletal groom inputs; other wardrobe types are not verified')


def verify_assembled_grooms(blueprint,expected,*,native_stock_retarget=False):
    """Require each selected source groom to reach a distinct bound component."""
    if not expected['grooms']:
        return dict(verified=[],selected_grooms=0,scope='No selected skeletal grooms')
    import unreal
    subsystem=unreal.get_engine_subsystem(unreal.SubobjectDataSubsystem)
    library=unreal.SubobjectDataBlueprintFunctionLibrary
    components=[];seen=set();meshes=set(actor_meshes(blueprint).values())
    for handle in subsystem.k2_gather_subobject_data_for_blueprint(blueprint):
        component=library.get_object(library.get_data(handle))
        if not isinstance(component,unreal.GroomComponent) or component.get_path_name() in seen:continue
        seen.add(component.get_path_name())
        groom=component.get_editor_property('groom_asset');binding=component.get_editor_property('binding_asset')
        if not groom or not binding or binding.get_editor_property('groom')!=groom:continue
        target=binding.get_editor_property('target_skeletal_mesh')
        if not target or target.get_path_name() not in meshes:continue
        groups=list(binding.get_editor_property('group_infos'))
        if not groups or any(int(group.get_editor_property('sim_root_count'))<=0 for group in groups):continue
        components.append((component,groom,binding))
    matches=[];fingerprints={};comparisons={};adaptations=set()
    for row in expected['grooms']:
        source=unreal.load_asset(row['groom'])
        if not isinstance(source,unreal.GroomAsset):raise RuntimeError('Assembly lost a selected source groom: '+row['groom'])
        source_data=json.loads(unreal.CreateMHGroomLibrary.fingerprint_groom(source))
        if source_data.get('status')!='complete' or source_data.get('groups')!=row['geometry']:
            raise RuntimeError('Assembly changed a selected source groom: '+row['groom'])
        candidates=[]
        for index,(component,groom,binding) in enumerate(components):
            path=groom.get_path_name()
            if path not in fingerprints:
                fingerprints[path]=json.loads(unreal.CreateMHGroomLibrary.fingerprint_groom(groom))
            data=fingerprints[path]
            pair=(row['groom'],path)
            if pair not in comparisons:
                comparisons[pair]=(data.get('status')=='complete' and len(data.get('groups',[]))==len(row['geometry']) and all(
                    json.loads(unreal.CreateMHGroomLibrary.compare_groom_groups(source,index,groom,index)).get('matches') is True
                    for index in range(len(row['geometry']))))
                # The native editor pipeline explicitly bakes an RBF head fit
                # into installed preset grooms. Their positions are expected to
                # change; curve layout, guide flags, widths, colors and root UVs
                # must still match every source group. This allowance is used
                # only after this action verified a fresh native actor update.
                stock=(native_stock_retarget and row['groom'].startswith('/MetaHumanCharacter/Optional/Grooms/')
                    and row.get('source','').startswith('/MetaHumanCharacter/Optional/Grooms/GroomMesh/')
                    and row.get('target','').startswith('/MetaHumanCharacter/Optional/Grooms/GroomMesh/'))
                if (not comparisons[pair] and stock and data.get('status')=='complete'
                        and len(data.get('groups',[]))==len(row['geometry'])):
                    attributes=[json.loads(unreal.CreateMHGroomLibrary.compare_groom_groups(source,i,groom,i,False))
                        for i in range(len(row['geometry']))]
                    if all(value.get('matches') is True for value in attributes):
                        comparisons[pair]=True;adaptations.add(pair)
            if comparisons[pair]:
                candidates.append(index)
        # Prefer the native slot name, while allowing pipeline-specific names.
        candidates.sort(key=lambda i:components[i][0].get_name().removesuffix('_GEN_VARIABLE')!=row['slot'])
        matches.append(candidates)
    occupied={}
    def assign(row,visited):
        for candidate in matches[row]:
            if candidate in visited:continue
            visited.add(candidate)
            if candidate not in occupied or assign(occupied[candidate],visited):
                occupied[candidate]=row;return True
        return False
    for row in range(len(matches)):
        if not assign(row,set()):
            raise RuntimeError('Assembly omitted or changed the selected '+expected['grooms'][row]['slot']+' groom. After a head revision, push the custom groom again against the current linked character, equip that wardrobe item, then assemble. Inspect the wardrobe validation log; the result is not marked complete.')
    result=[]
    for component_index,row_index in sorted(occupied.items(),key=lambda item:item[1]):
        component,groom,binding=components[component_index]
        adapted=(expected['grooms'][row_index]['groom'],groom.get_path_name()) in adaptations
        result.append(dict(slot=expected['grooms'][row_index]['slot'],component=component.get_name(),
                           groom=groom.get_path_name(),binding=binding.get_path_name(),
                           source_groom=expected['grooms'][row_index]['groom'],geometry_verified=not adapted,source_unchanged=True,
                           attributes_verified=True,native_stock_retargeted=adapted,
                           position_tolerance_cm=None if adapted else .001,
                           position_scope='Native RBF head adaptation; inspect roots and silhouette' if adapted else 'Source coordinates within native roundtrip tolerance'))
    return dict(verified=result,selected_grooms=len(expected['grooms']),
                scope='Bound components and ordered strand attributes; source positions compared except for native preset RBF head adaptation. Not animated or visual acceptance.')


def assemble(asset_path,build_options,expected_project='',progress_path=None):
    import unreal
    # Load the sibling status helper from the immutable action snapshot.
    import runpy
    inspect=runpy.run_path(str(Path(__file__).with_name('unreal_character.py')))['inspect_character']
    root,name,profile=options(build_options)
    project=Path(unreal.Paths.get_project_file_path()).resolve()
    if expected_project and project!=Path(expected_project).resolve():raise ValueError('Open the linked Unreal project before assembly')
    report={'status':'failed','action':'BUILD_CHARACTER','character':asset_path,
            'unreal_project_file':str(project),'inspected_at':datetime.now(timezone.utc).isoformat(),
            'native_initializations':0}
    content=Path(unreal.Paths.project_content_dir()).resolve()
    limits=budgets(content)
    report['build']={'profile':profile,'requested_root':root,'requested_name':name,'budgets':limits,'state':'NOT_STARTED','native_build_api_called':False}
    if limits['free_disk_bytes']<limits['disk_reserve_bytes']:
        report.update(status='failed',error='Assembly needs at least 10 GiB of free disk space; free generated files or choose another project volume')
        return report
    character=unreal.load_asset(asset_path)
    if not isinstance(character,unreal.MetaHumanCharacter):raise ValueError('The linked native character is missing')
    overlay_api=None
    if build_options.get('eye_overlay_policy','NATIVE')=='OPAQUE_SOURCE':
        overlay_api=runpy.run_path(str(Path(__file__).with_name('unreal_eye_overlays.py')))
        overlay_api['require_projected_eyes'](character)
    subsystem=unreal.get_editor_subsystem(unreal.MetaHumanCharacterEditorSubsystem)
    editing=subsystem.is_object_added_for_editing(character)
    acquired=False
    previous=None
    previous_quality=''
    quality_tag_cleared=False
    try:
        # A recommendation override does not disable the commit-headroom floor.
        # Take ownership once; inspect_character reuses this editing session.
        override=build_options.get('allow_low_memory',False)
        required=commit_requirement(limits,editing,override)
        limits['available_commit_bytes']=budgets(content)['available_commit_bytes']
        limits['required_commit_bytes']=required
        limits['editing_session_reused']=bool(editing)
        limits['memory_measure']='Available commit/address-space headroom (UE AvailableVirtual on Windows)'
        if override:report['build']['memory_warning']='Assembly below the native 10 GiB commit recommendation was explicitly allowed; hard commit floor retained'
        if limits['available_commit_bytes']<required:
            report.update(error=f"Assembly needs about {required/GIB:.0f} GiB of available commit headroom{' including character loading' if not editing else ''}; {limits['available_commit_bytes']/GIB:.1f} GiB is available. No new editing session or build was started.")
            return report
        existing=build_options.get('existing_blueprint','')
        ordinal=1;chosen=name
        if existing:
            previous=unreal.load_asset(existing)
            if isinstance(previous,unreal.Blueprint):
                previous_quality=unreal.EditorAssetLibrary.get_metadata_tag(previous,EXPORT_QUALITY_TAG)
            if not previous_quality:
                raise ValueError('The linked assembly is missing or is not a generated MetaHuman Blueprint')
            owner=unreal.EditorAssetLibrary.get_metadata_tag(previous,'MetaHumanMaker.SourceCharacter')
            if owner and owner!=character.get_path_name():raise ValueError('The existing assembly belongs to another Creator character')
            target=existing.rsplit('/',1)[0];chosen=target.rsplit('/',1)[1]
        else:
            while True:
                target=root+'/'+chosen
                disk=content/target.removeprefix('/Game/')
                if not unreal.EditorAssetLibrary.does_directory_exist(target) and not disk.exists():break
                ordinal+=1;chosen=name+'_v'+str(ordinal).zfill(3)
        blueprint_path=target+'/BP_'+chosen
        wardrobe=selected_grooms(character,target)
        report['build']['wardrobe_preflight']=wardrobe
        if wardrobe['issues']:
            raise ValueError('Assembly wardrobe preflight: '+'; '.join(wardrobe['issues'])+'. No build was dispatched.')
        if not editing:
            if not subsystem.try_add_object_to_edit(character):raise RuntimeError('Creator could not initialize the character for assembly')
            acquired=True;report['native_initializations']=1
        # Loading rigged character data consumes part of the cold allowance.
        # Recheck the native build threshold in the actual editing session.
        remaining=budgets(content)
        limits['build_commit_available_bytes']=remaining['available_commit_bytes']
        required=commit_requirement(limits,True,override)
        limits['build_commit_required_bytes']=required
        if remaining['available_commit_bytes']<required:
            report.update(status='failed',error=f"Character loading left {remaining['available_commit_bytes']/GIB:.1f} GiB commit headroom; assembly needs {required/GIB:.0f} GiB. No build was started.")
            return report
        inspected=inspect(asset_path,expected_project=expected_project)
        # Inspection reuses this session; its own zero-count must not erase
        # the initialization already owned by this assembly request.
        inspected.pop('native_initializations',None)
        report.update(inspected)
        report['action']='BUILD_CHARACTER'
        if not report['assembly']['ready']:
            report.update(status='failed',error=report['assembly'].get('reason') or 'Complete rigging and texture sources in Creator before assembly')
            return report
        # Recheck after taking ownership; a stale Blender readiness indicator is
        # never authorization to build an unrigged or incomplete character.
        if not subsystem.can_build_meta_human(character,False):
            report.update(status='failed',error='Character readiness changed. Refresh Creator status before assembly')
            return report
        report['build'].update(state='BUILDING',output_folder=target,blueprint=blueprint_path,updated_existing=bool(existing))
        if progress_path:
            Path(progress_path).write_text(json.dumps({'phase':'Assembling '+chosen+' in Unreal...',
                'output_folder':target,'profile':profile}),encoding='utf-8')
        params=unreal.MetaHumanCharacterEditorBuildParameters()
        params.pipeline_type=(unreal.MetaHumanDefaultPipelineType.CINEMATIC if profile=='CINEMATIC' else unreal.MetaHumanDefaultPipelineType.OPTIMIZED)
        params.pipeline_quality=getattr(unreal.MetaHumanQualityLevel,profile)
        params.absolute_build_path=root;params.name_override=chosen
        # Match Creator's animation-system selection, including the native AnimBP
        # default. The public build struct itself otherwise defaults to None.
        settings=character.get_editor_property('assembly_settings')
        animation=str(settings.get_editor_property('animation_system_name'))
        params.animation_system_name=animation if animation not in {'','None'} else 'AnimBP'
        # A fresh, per-build Common folder copies the plugin dependencies into
        # the game project without touching another character's shared assets.
        params.common_folder_path=target+'/Common'
        report['build'].update(animation_system=str(params.animation_system_name),common_folder=params.common_folder_path)
        params.enable_wardrobe_item_validation=True
        textures=runpy.run_path(str(Path(__file__).with_name('unreal_assembly_textures.py')))
        private_pipeline,report['build']['texture_policy']=textures['configure'](character,params,profile,build_options.get('preserve_source_textures',True))
        if report['build']['texture_policy'].get('material_baking_required') and re.search(r'(?:^|\s)-nullrhi(?:\s|$)',unreal.SystemLibrary.get_command_line(),re.I):
            raise ValueError('This native assembly pipeline bakes materials and needs rendering enabled. Restart the build editor without NullRHI; no native build was requested.')
        available=budgets(content)
        limits['dispatch_commit_available_bytes']=available['available_commit_bytes']
        limits['dispatch_disk_available_bytes']=available['free_disk_bytes']
        if available['free_disk_bytes']<limits['disk_reserve_bytes']:
            report['build']['state']='NOT_STARTED'
            report.update(status='failed',error='Assembly preparation left less than the 10 GiB disk reserve. No build was dispatched.')
            return report
        if available['available_commit_bytes']<required:
            report['build']['state']='NOT_STARTED'
            report.update(status='failed',error=f"Assembly preparation left {available['available_commit_bytes']/GIB:.1f} GiB commit headroom; {required/GIB:.0f} GiB is required. No build was dispatched.")
            return report
        current_wardrobe=selected_grooms(character,target)
        report['build']['wardrobe_dispatch_check']=current_wardrobe
        if current_wardrobe['issues'] or current_wardrobe['grooms']!=wardrobe['grooms']:
            raise ValueError('Selected groom inputs changed during assembly preparation. Refresh Creator and retry; no build was dispatched.')
        # UE 5.8 writes this tag after UpdateActorBlueprint. On a rebuild, the
        # previous tag would otherwise satisfy the check even if native assembly
        # returned early. Clear it in memory just before dispatch and never
        # save the removal as a separate operation.
        if private_pipeline:
            native=getattr(unreal.CreateMHCharacterLibrary,'build_with_pipeline',None)
            if native is None:raise ValueError('Install CreateMetaHumanBridge 0.20 or later for source-resolution assembly')
            request=dict(pipeline_type=int(params.pipeline_type.value),pipeline_quality=int(params.pipeline_quality.value),
                root=str(params.absolute_build_path),name=str(params.name_override),common=str(params.common_folder_path),animation=str(params.animation_system_name))
        if previous is not None:
            quality_tag_cleared=True
            unreal.EditorAssetLibrary.remove_metadata_tag(previous,EXPORT_QUALITY_TAG)
            if unreal.EditorAssetLibrary.get_metadata_tag(previous,EXPORT_QUALITY_TAG):
                raise RuntimeError('Could not verify the next native rebuild. No rebuild was dispatched.')
        report['build']['native_build_api_called']=True
        if private_pipeline:
            if not native(character,private_pipeline,json.dumps(request),False):raise RuntimeError('The native assembly request was rejected')
        else:
            subsystem.build_meta_human(character,params)
        blueprint=unreal.load_asset(blueprint_path)
        if not isinstance(blueprint,unreal.Blueprint):
            raise RuntimeError('Native assembly did not produce its actor Blueprint. Inspect the MetaHuman Message Log; partial assets were retained at '+target)
        if existing and blueprint.get_path_name()!=existing:raise RuntimeError('Native assembly returned a different Blueprint while updating the linked character')
        quality_marker=unreal.EditorAssetLibrary.get_metadata_tag(blueprint,EXPORT_QUALITY_TAG)
        if not quality_marker:
            raise RuntimeError('Native assembly did not reach its final actor update. Inspect the MetaHuman Message Log; partial assets were retained at '+target)
        report['build']['native_actor_update_verified']=True
        report['build']['groom_verification']=verify_assembled_grooms(blueprint,wardrobe,
            native_stock_retarget=report['build'].get('native_actor_update_verified') is True)
        unreal.EditorAssetLibrary.set_metadata_tag(blueprint,'MetaHumanMaker.SourceCharacter',character.get_path_name())
        eyes=runpy.run_path(str(Path(__file__).with_name('unreal_assembly_eyes.py')))
        report['build']['eye_material_policy']=eyes['preserve'](character,blueprint)
        auxiliary=runpy.run_path(str(Path(__file__).with_name('unreal_assembly_auxiliary.py')))
        report['build']['auxiliary_skin_policy']=auxiliary['preserve'](character,blueprint)
        if overlay_api:
            report['build']['source_eye_overlay_policy']=overlay_api['set_policy'](blueprint,'OPAQUE_SOURCE',hidden_material=overlay_api['hidden_material_for'](blueprint))
        unreal.BlueprintEditorLibrary.compile_blueprint(blueprint)
        status=blueprint.get_editor_property('status')
        if status not in {unreal.BlueprintStatus.BS_UP_TO_DATE,unreal.BlueprintStatus.BS_UP_TO_DATE_WITH_WARNINGS} or not blueprint.generated_class():
            raise RuntimeError('The assembled Blueprint did not compile successfully. Inspect '+blueprint_path)
        registry=unreal.AssetRegistryHelpers.get_asset_registry()
        skeletal=[]
        for item in registry.get_assets_by_path(target,recursive=True):
            if str(item.asset_class_path.asset_name)=='SkeletalMesh':skeletal.append(str(item.package_name)+'.'+str(item.asset_name))
        if not skeletal:raise RuntimeError('Native assembly produced no skeletal meshes under '+target)
        if not unreal.EditorAssetLibrary.save_directory(target,only_if_is_dirty=False,recursive=True):
            raise RuntimeError('Assembly output could not be fully saved. Inspect '+target+' before retrying')
        report['build'].update(state='COMPLETE',blueprint=blueprint.get_path_name(),skeletal_meshes=skeletal,
            blueprint_status=str(status),native_export_quality=str(quality_marker),completed_at=datetime.now(timezone.utc).isoformat())
        try:report['build']['component_meshes']=actor_meshes(blueprint)
        except Exception as exc:report['build']['component_warning']='Could not identify groom binding targets: '+str(exc)
        try:unreal.EditorAssetLibrary.sync_browser_to_objects([blueprint.get_path_name()])
        except Exception as exc:report['build']['browser_warning']=str(exc)
        report['status']='complete'
        return report
    except Exception as exc:
        report.update(status='failed',error=str(exc))
        if report['build']['state']=='BUILDING':report['build']['state']='FAILED'
        return report
    finally:
        try:
            if quality_tag_cleared:
                # Resolve the current object because native assembly can replace
                # the Blueprint at this path. Preserve a newly written marker.
                current=unreal.load_asset(existing)
                if (isinstance(current,unreal.Blueprint) and
                        not unreal.EditorAssetLibrary.get_metadata_tag(current,EXPORT_QUALITY_TAG)):
                    unreal.EditorAssetLibrary.set_metadata_tag(current,EXPORT_QUALITY_TAG,previous_quality)
        finally:
            if acquired:subsystem.remove_object_to_edit(character)
