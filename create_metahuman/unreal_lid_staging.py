"""Stage a prepared lid proposal as a separate saved character assembly."""
import json,re,runpy,uuid
from pathlib import Path


def write(path,value):
    path=Path(path);temporary=path.with_suffix('.tmp');temporary.write_text(json.dumps(value,indent=2));temporary.replace(path)


def stage(folder,request,state):
    import unreal
    prepared=Path(request['preparation_folder']).resolve()
    helpers=runpy.run_path(str(Path(__file__).with_name('unreal_lod_correctives.py')))
    manifest,mesh=helpers['validate_sources'](prepared)
    review=runpy.run_path(str(Path(__file__).with_name('unreal_face_review.py')))
    actors=unreal.get_editor_subsystem(unreal.EditorActorSubsystem)
    actor=review['_actor'](unreal,state['actor'])
    if not actor:raise ValueError('Return to the original review actor before staging')
    face=review['_components'](unreal,actor)[state['face']]
    if face.get_skeletal_mesh_asset()!=mesh:raise ValueError('The proposal belongs to another face; connect its source character')
    if state.get('look'):raise ValueError('Save or restore the temporary skin preview before staging the saved character')
    look=review['_look_api']();bp=look['blueprint'](unreal,actor);library=unreal.EditorAssetLibrary
    package=bp.get_path_name().split('.')[0]
    if package in {p.get_path_name() for p in unreal.EditorLoadingAndSavingUtils.get_dirty_content_packages()}:raise ValueError('Save the character Blueprint edits before staging')
    subsystem=unreal.get_engine_subsystem(unreal.SubobjectDataSubsystem);data=unreal.SubobjectDataBlueprintFunctionLibrary
    objects=[data.get_object(data.get_data(h)) for h in subsystem.k2_gather_subobject_data_for_blueprint(bp)]
    faces=list(dict.fromkeys(c for c in objects if isinstance(c,unreal.SkeletalMeshComponent) and c.get_name().removesuffix('_GEN_VARIABLE')=='Face'))
    if len(faces)!=1 or faces[0].get_skeletal_mesh_asset()!=mesh:raise ValueError('Save the captured Face mesh to the source Blueprint before staging')
    bp_file=Path(unreal.Paths.convert_relative_path_to_full(unreal.Paths.project_content_dir()))/(package.removeprefix('/Game/')+'.uasset')
    fingerprint=helpers['digest'](bp_file);sealed=helpers['digest'](prepared/'manifest.json')
    plan_file=prepared/'native-staging.json'
    if plan_file.exists():
        plan=json.loads(plan_file.read_text())
        if plan['request_sha256']!=sealed or plan['source_blueprint']!=bp.get_path_name() or plan['source_blueprint_sha256']!=fingerprint:raise ValueError('The saved Blueprint or proposal changed since staging; retain the existing candidate and prepare the current character again')
    else:
        name=re.sub('[^A-Za-z0-9_]','_',mesh.get_name())[:48]
        destination='/Game/CreateMetaHuman/LidCorrections/'+name+'_'+uuid.uuid4().hex[:12]
        plan=dict(request_sha256=sealed,source_blueprint=bp.get_path_name(),source_blueprint_sha256=fingerprint,
                  mesh=destination+'/SKM_Face',blueprint=destination+'/BP_LidReview')
        write(plan_file,plan)
    receipt=prepared/'native'/'assembly.json'
    reused=receipt.is_file()
    if reused:
        assembly=json.loads(receipt.read_text())
        if assembly.get('status')!='complete' or assembly.get('blueprint')!=plan['blueprint']+'.BP_LidReview' or not library.does_asset_exist(assembly['blueprint']):raise ValueError('The original staged assembly is missing; inspect its retained receipt before preparing another revision')
        assembly=runpy.run_path(str(Path(__file__).with_name('unreal_revision_bounds.py')))['read_assembly'](prepared)
    else:
        stage_file=prepared/'native'/'stage.json'
        if stage_file.is_file():
            native=json.loads(stage_file.read_text());target=unreal.load_asset(native['candidate'])
            if native.get('status')!='complete' or native.get('request_sha256')!=sealed or native['candidate'].split('.')[0]!=plan['mesh'] or not isinstance(target,unreal.SkeletalMesh):raise ValueError('The retained native stage does not match this proposal')
            tagged=json.loads(library.get_metadata_tag(target,'CreateMetaHuman.LODCorrectiveRevision') or '{}')
            if tagged.get('request_sha256')!=sealed:raise ValueError('The retained candidate has a different corrective revision')
        else:native=helpers['stage'](prepared,plan['mesh'])
        if library.does_asset_exist(plan['blueprint']):raise ValueError('A partial review assembly already exists. Retain it for inspection; this action will not overwrite it.')
        assembly=runpy.run_path(str(Path(__file__).with_name('unreal_globe_rigidity.py')))['assemble'](native,bp.get_path_name(),plan['blueprint'],metadata_tag='CreateMetaHuman.LidCorrectiveReview')
        assembly.update(request_sha256=sealed,folder=str(prepared),settings_scope='saved_blueprint',level_instance_overrides_copied=False)
        write(receipt,assembly)
    review['_guard'](review['_native'](unreal),review['_components'](unreal,actor),state['expected']);look['guard'](unreal,actor,state)
    sources=helpers['source_fingerprints'](mesh)
    if (manifest.get('native_source_sha256') and sources!=manifest['native_source_sha256']) or helpers['digest'](bp_file)!=fingerprint or package in {p.get_path_name() for p in unreal.EditorLoadingAndSavingUtils.get_dirty_content_packages()}:raise RuntimeError('A source changed during staging; inspect the retained candidate before approval')
    state['calibration_staging']=assembly;write(Path(state['origin'])/'session.json',state)
    getattr(unreal,'_cm_face_review_sessions',{})[state['origin']]=state
    report=review['_report'](unreal,state,actor)
    report.update(assets_saved=0 if reused else 2+len({row['binding'] for row in assembly['grooms']['updated']}),source_assets_saved=0,staged_assets_retained=True)
    report['message']='Separate rig staged with saved Blueprint settings. Original actor retained; review eyelid motion before approval.'
    return report
