"""Retire generated globe morphs on an explicit, separate native rig copy."""
import hashlib,json,re,runpy
from pathlib import Path


def digest(path):
    with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()


def stage(request_folder,destination):
    import unreal
    request=Path(request_folder).resolve();manifest=json.loads((request/'manifest.json').read_text());library=unreal.EditorAssetLibrary
    if manifest.get('schema')!='CM_RIGID_GLOBE_REVISION_V1' or manifest.get('status')!='prepared':raise ValueError('Prepare a complete generated-globe retirement request first')
    if not re.fullmatch(r'/Game(?:/[A-Za-z0-9_]+)+',destination) or library.does_asset_exist(destination):raise ValueError('Choose a fresh project skeletal-mesh revision')
    api=unreal.CreateMHCalibrationLibrary
    if not hasattr(api,'retire_generated_morphs'):raise ValueError('Update the native bridge to 0.24 or newer for batched generated-morph retirement')
    budget=runpy.run_path(str(Path(__file__).with_name('resource_budget.py')));budget['require']('native rigid globe revision',8*budget['GIB']);budget['require_disk']('native rigid globe revision',request,2*budget['GIB'])
    for name,expected in manifest['input_sha256'].items():
        if digest(Path(name))!=expected:raise ValueError('A captured rig source changed; prepare the revision again')
    for name,expected in manifest['prepared_sha256'].items():
        path=(request/name).resolve()
        if path.parent!=request or digest(path)!=expected:raise ValueError('A prepared globe-retirement payload changed')
    source=unreal.load_asset(manifest['source_mesh']);package=manifest['source_mesh'].split('.')[0]
    actual_asset=Path(unreal.Paths.convert_relative_path_to_full(unreal.Paths.project_content_dir()))/(package.removeprefix('/Game/')+'.uasset')
    if actual_asset.resolve()!=Path(manifest['source_asset_file']).resolve():raise ValueError('The saved source package belongs to another project')
    dirty=lambda:{p.get_path_name() for p in unreal.EditorLoadingAndSavingUtils.get_dirty_content_packages()}
    if package in dirty():raise ValueError('Save the source face edits before preparing another revision')
    names=manifest['retire_morphs']
    if not names or len(set(names))!=len(names) or any(not re.fullmatch(r'eye(?:Left|Right)_lod[0-7]_mesh__CM_[A-Za-z0-9_]+',name) for name in names):raise ValueError('Only explicitly named generated globe morphs can be retired')
    original={m.get_name():m for m in source.get_editor_property('morph_targets')};present=[name for name in names if name in original]
    if not present:raise ValueError('The saved mesh has no matching generated globe morphs')
    out=request/'native'
    if out.exists():raise ValueError('Observe the original native revision instead of staging it twice')
    out.mkdir();dna=out/'rigid-globes.dna'
    if not api.write_calibration_dna(manifest['source_dna'],str(request/'dna-request.json'),str(dna)):raise RuntimeError('Could not retire the generated globe DNA deltas')
    target=library.duplicate_asset(source.get_path_name(),destination)
    if not target:raise RuntimeError('Could not create the private rigid-globe revision')
    (out/'partial-assets.json').write_text(json.dumps(dict(candidate=target.get_path_name())))
    clones={m.get_name():m for m in target.get_editor_property('morph_targets')}
    if any(name not in clones or clones[name]==original[name] or clones[name].get_outer()!=target for name in present):raise ValueError('The revision does not own its morph subobjects; the source was retained')
    # Removing the source attributes as well as compiled morphs prevents a later
    # native rebuild from silently restoring the discarded generated shapes.
    if not api.retire_generated_morphs(source,target,present):raise RuntimeError('Could not retire the generated globe morphs in one native rebuild')
    if not api.apply_calibration_to_mesh_copy(source,target,str(dna)):raise RuntimeError('Could not bind the rigid-globe DNA on its private revision')
    remaining={m.get_name() for m in target.get_editor_property('morph_targets')}
    if remaining!=set(original)-set(present):raise RuntimeError('The native revision did not preserve unrelated morph targets')
    if package in dirty():raise RuntimeError('The source face was unexpectedly dirtied; do not publish this revision')
    report=dict(status='complete',source=source.get_path_name(),candidate=target.get_path_name(),dna=str(dna),retired_compiled_morphs=present,already_absent_morphs=[name for name in names if name not in original],lods=manifest['lods'],source_assets_saved=0,neutral_geometry_changed=False,joints_changed=False,skin_weights_changed=False,lids_changed=False,native_motion_accepted=False)
    library.set_metadata_tag(target,'CreateMetaHuman.RigidGlobeRevision',json.dumps(report))
    if not library.save_loaded_asset(target,False):raise RuntimeError('Could not save the separate rigid-globe revision')
    (out/'stage.json').write_text(json.dumps(report,indent=2));return report


def assemble(stage_report,source_blueprint,destination,*,metadata_tag='CreateMetaHuman.RigidGlobeReview'):
    """Create a separate assembled review with matching groom bindings."""
    import unreal
    library=unreal.EditorAssetLibrary
    if stage_report.get('status')!='complete' or not re.fullmatch(r'/Game(?:/[A-Za-z0-9_]+)+',destination) or library.does_asset_exist(destination):raise ValueError('Choose a completed rigid-globe stage and a fresh assembly destination')
    source=unreal.load_asset(source_blueprint);mesh=unreal.load_asset(stage_report['candidate'])
    if not isinstance(source,unreal.Blueprint) or not isinstance(mesh,unreal.SkeletalMesh):raise ValueError('Choose an assembled blueprint and the staged face')
    subsystem=unreal.get_engine_subsystem(unreal.SubobjectDataSubsystem);data=unreal.SubobjectDataBlueprintFunctionLibrary
    def faces(bp):
        values=[data.get_object(data.get_data(h)) for h in subsystem.k2_gather_subobject_data_for_blueprint(bp)]
        return list(dict.fromkeys(c for c in values if isinstance(c,unreal.SkeletalMeshComponent) and c.get_name().removesuffix('_GEN_VARIABLE')=='Face'))
    before=faces(source)
    if len(before)!=1 or before[0].get_skeletal_mesh_asset().get_path_name()!=stage_report['source']:raise ValueError('The assembled source face differs from the captured rig')
    bp=library.duplicate_asset(source.get_path_name(),destination);selected=faces(bp)
    if len(selected)!=1 or selected[0] in before:raise RuntimeError('The assembly does not own a unique face component')
    selected[0].set_skeletal_mesh_asset(mesh)
    grooms=runpy.run_path(str(Path(__file__).with_name('unreal_groom.py')))['rebind_revision_grooms'](bp,mesh,destination.rsplit('/',1)[0]+'/GroomBindings')
    unreal.BlueprintEditorLibrary.compile_blueprint(bp)
    report=dict(status='complete',blueprint=bp.get_path_name(),mesh=mesh.get_path_name(),source_blueprint=source.get_path_name(),grooms=grooms,source_assets_saved=0,native_motion_accepted=False)
    library.set_metadata_tag(bp,metadata_tag,json.dumps(report))
    if not library.save_loaded_asset(bp,False):raise RuntimeError('Could not save the separate rigid-globe review assembly')
    return report
