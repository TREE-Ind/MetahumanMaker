"""Separate saved assembly revisions with verified neutral render bounds."""
import json,math,runpy,uuid
from pathlib import Path

TAG='CreateMetaHuman.BoundsRevision'

def api(name):return runpy.run_path(str(Path(__file__).with_name(name+'.py')))

def write(path,value):
    path=Path(path);temporary=path.with_suffix('.tmp');temporary.write_text(json.dumps(value,indent=2));temporary.replace(path)

def read_assembly(prepared):
    native=Path(prepared)/'native';base=json.loads((native/'assembly.json').read_text());variant=native/'assembly-bounds.json'
    if not variant.is_file():return base
    result=json.loads(variant.read_text())
    if result.get('schema')!='CM_BOUNDS_ASSEMBLY_V1' or result.get('status')!='complete' or result.get('base_blueprint')!=base['blueprint'] or result.get('base_mesh')!=base['mesh'] or result.get('source_blueprint')!=base['source_blueprint'] or result.get('geometry_changed') is not False:raise ValueError('The retained bounds revision does not match this staged rig')
    return result

def default_faces(unreal,bp):
    subsystem=unreal.get_engine_subsystem(unreal.SubobjectDataSubsystem);data=unreal.SubobjectDataBlueprintFunctionLibrary
    values=[data.get_object(data.get_data(h)) for h in subsystem.k2_gather_subobject_data_for_blueprint(bp)]
    return list(dict.fromkeys(c for c in values if isinstance(c,unreal.SkeletalMeshComponent) and c.get_name().removesuffix('_GEN_VARIABLE')=='Face'))

def file_for(unreal,asset):
    package=asset.get_path_name().split('.')[0]
    if not package.startswith('/Game/'):raise ValueError('Use saved project assemblies for a bounds revision')
    return Path(unreal.Paths.convert_relative_path_to_full(unreal.Paths.project_content_dir()))/(package.removeprefix('/Game/')+'.uasset')

def fit_blueprint(unreal,bp,source_blueprint):
    native=unreal.CreateMHCalibrationLibrary;actors=unreal.get_editor_subsystem(unreal.EditorActorSubsystem);selected=actors.get_selected_level_actors();actor=None
    if bp==source_blueprint or bp.get_path_name().split('.')[0]==source_blueprint.get_path_name().split('.')[0]:raise ValueError('Fit bounds only on a separate Blueprint copy')
    templates=default_faces(unreal,bp)
    if len(templates)!=1 or templates[0] in default_faces(unreal,source_blueprint):raise ValueError('The bounds revision needs one independently owned Face component')
    def spawn():
        value=actors.spawn_actor_from_object(bp,unreal.Vector(),transient=True)
        if not value:raise RuntimeError('Could not create the owned bounds-validation actor')
        face=next(c for c in value.get_components_by_class(unreal.SkeletalMeshComponent) if c.get_name()=='Face');mesh=face.get_skeletal_mesh_asset()
        clip=unreal.CreateMHExpressionLibrary.create_control_preview_clip(mesh,'{}')
        if not clip:actors.destroy_actor(value);raise ValueError('The staged face has no readable native neutral pose')
        api('unreal_actor_preview')['preview_clip'](value,clip,0.,force_lod=0,lock_lod_sync=True)
        if not native.refresh_preview_component_bounds(face):actors.destroy_actor(value);raise RuntimeError('Could not refresh the owned preview bounds')
        return value,face
    def corners(face,geometry):
        low,high=geometry['geometry_min'],geometry['geometry_max'];transform=face.get_world_transform()
        return [unreal.MathLibrary.transform_location(transform,unreal.Vector(x,y,z)) for x in (low[0],high[0]) for y in (low[1],high[1]) for z in (low[2],high[2])]
    def measured(face,points):
        origin,extent,_=unreal.SystemLibrary.get_component_bounds(face)
        if min(extent.x,extent.y,extent.z)<=1e-6:raise ValueError('The native physics/mesh bounds have a collapsed axis; retain this revision for manual review')
        ratios=[abs(getattr(p,a)-getattr(origin,a))/getattr(extent,a) for p in points for a in ('x','y','z')]
        return max(ratios),dict(origin=[origin.x,origin.y,origin.z],extent=[extent.x,extent.y,extent.z])
    try:
        actor,face=spawn();geometry=json.loads(native.inspect_geometry_bounds(face.get_skeletal_mesh_asset()));points=corners(face,geometry)
        ratio,before=measured(face,points);authored=float(face.get_editor_property('bounds_scale'));scale=authored if ratio<=1.00001 else authored*ratio*1.03
        if not math.isfinite(scale) or scale<=0 or scale>max(8.,authored):raise ValueError('This character needs unusually large render bounds; inspect its scale and physics asset before delivery')
        actors.destroy_actor(actor);actor=None
        templates[0].set_editor_property('bounds_scale',scale);unreal.BlueprintEditorLibrary.compile_blueprint(bp)
        actor,face=spawn()
        if abs(face.get_editor_property('bounds_scale')-scale)>1e-5:raise ValueError('The construction script overrides Face bounds scale; retain the copy for manual review')
        remaining,after=measured(face,corners(face,geometry))
        if remaining>1.00001:raise ValueError('The revised component still excludes neutral geometry; inherited/physics bounds require manual review')
        return dict(status='complete',component='Face',authored_scale=authored,bounds_scale=scale,before=before,after=after,maximum_remaining_ratio=remaining,neutral_only=True,physics_assets_modified=0)
    finally:
        if actor and unreal.SystemLibrary.is_valid(actor):actors.destroy_actor(actor)
        actors.set_selected_level_actors(selected)

def create(prepared_folder):
    import unreal
    prepared=Path(prepared_folder).resolve();native_folder=prepared/'native';library=unreal.EditorAssetLibrary
    helpers=api('unreal_lod_correctives');helpers['validate_sources'](prepared)
    original=json.loads((native_folder/'assembly.json').read_text())
    if (native_folder/'assembly-bounds.json').is_file():
        result=read_assembly(prepared)
        if not library.does_asset_exist(result['blueprint']):raise ValueError('The retained bounds revision is missing; inspect its receipt before creating another')
        return result
    native=unreal.CreateMHCalibrationLibrary
    if not hasattr(native,'inspect_geometry_bounds') or not hasattr(native,'refresh_preview_component_bounds'):raise ValueError('Install native bridge 0.29 or newer for verified render-bounds revisions')
    source=unreal.load_asset(original['blueprint']);mesh=unreal.load_asset(original['mesh'])
    if original.get('status')!='complete' or not isinstance(source,unreal.Blueprint) or not isinstance(mesh,unreal.SkeletalMesh):raise ValueError('Stage a complete separate rig before creating its bounds revision')
    if source.get_path_name().split('.')[0] in {p.get_path_name() for p in unreal.EditorLoadingAndSavingUtils.get_dirty_content_packages()}:raise ValueError('Save the staged Blueprint edits before creating its bounds revision')
    source_file=file_for(unreal,source);source_sha=helpers['digest'](source_file);sources=helpers['source_fingerprints'](mesh)
    geometry=json.loads(native.inspect_geometry_bounds(mesh))
    budget=api('resource_budget');budget['require']('native bounds revision',2*budget['GIB']);budget['require_disk']('native bounds revision',prepared,512*budget['MIB'])
    plan_file=native_folder/'bounds-plan.json'
    if plan_file.is_file():
        plan=json.loads(plan_file.read_text())
        if plan['source_sha256']!=source_sha or plan['mesh_sources']!=sources:raise ValueError('A staged source changed after bounds revision began; retain its partial copy for inspection')
    else:
        destination=source.get_path_name().split('.')[0].rsplit('/',1)[0]+'/Bounds_'+uuid.uuid4().hex[:12]
        plan=dict(source_sha256=source_sha,mesh_sources=sources,destination=destination);write(plan_file,plan)
    destination=plan['destination'];mesh_report=None;target=mesh
    if library.does_directory_exist(destination):raise ValueError('A partial bounds revision already exists; inspect the retained plan instead of overwriting it')
    if any(row['outside_imported'] or row['outside_extended'] for row in geometry['lods']):
        target=library.duplicate_asset(mesh.get_path_name(),destination+'/SKM_Face')
        if not target or not native.refresh_mesh_copy_bounds(mesh,target):raise RuntimeError('Could not refresh bounds on the private face copy')
        mesh_report=json.loads(native.inspect_geometry_bounds(target))
        if mesh_report['position_sha1']!=geometry['position_sha1'] or any(row['outside_imported'] or row['outside_extended'] for row in mesh_report['lods']):raise RuntimeError('The private mesh bounds failed geometry-preservation/containment checks')
        if not library.save_loaded_asset(target,False):raise RuntimeError('Could not save the private bounds-only face copy')
        assembled=api('unreal_globe_rigidity')['assemble'](dict(status='complete',candidate=target.get_path_name(),source=mesh.get_path_name()),source.get_path_name(),destination+'/BP_BoundsReview',metadata_tag=TAG)
        bp=unreal.load_asset(assembled['blueprint'])
    else:
        bp=library.duplicate_asset(source.get_path_name(),destination+'/BP_BoundsReview');assembled=dict(blueprint=bp.get_path_name(),grooms=original['grooms']) if bp else None
        if not bp:raise RuntimeError('Could not duplicate the saved staged Blueprint')
    fitted=fit_blueprint(unreal,bp,source)
    if helpers['digest'](source_file)!=source_sha or helpers['source_fingerprints'](mesh)!=sources:raise RuntimeError('A source changed during bounds validation; inspect the retained private revision')
    result=dict(original,schema='CM_BOUNDS_ASSEMBLY_V1',status='complete',blueprint=bp.get_path_name(),mesh=target.get_path_name(),base_blueprint=original['blueprint'],base_mesh=original['mesh'],geometry_changed=False,bounds=fitted,source_geometry=geometry,mesh_bounds=mesh_report,source_assets_saved=0,native_motion_accepted=False)
    result['grooms']=dict(assembled['grooms'],blueprint=bp.get_path_name())
    result['created_assets']=sorted({bp.get_path_name(),*([target.get_path_name()] if target!=mesh else []),*([row['binding'] for row in assembled['grooms']['updated']] if target!=mesh else [])})
    library.set_metadata_tag(bp,TAG,json.dumps(result))
    if not library.save_loaded_asset(bp,False):raise RuntimeError('Could not save the verified private bounds revision')
    write(native_folder/'assembly-bounds.json',result);return result
