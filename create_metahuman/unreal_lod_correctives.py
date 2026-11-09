"""Stage explicit lower-LOD correctives on a separate native rig revision."""
import hashlib,json,re,runpy
from pathlib import Path


def digest(file):
    with Path(file).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()


def source_fingerprints(mesh):
    """Track saved project mesh/DNA packages, including split bulk payloads."""
    import unreal
    packages={mesh.get_path_name().split('.')[0]}
    for data in mesh.get_editor_property('asset_user_data'):
        if not data:continue
        if data.get_class().get_name()=='DNAAssetUserData':
            dna=data.get_editor_property('dna_asset')
            if dna:packages.add(dna.get_path_name().split('.')[0])
    dirty={p.get_path_name() for p in unreal.EditorLoadingAndSavingUtils.get_dirty_content_packages()}
    if packages&dirty:raise ValueError('Save the face and embedded DNA edits before calibration')
    content=Path(unreal.Paths.convert_relative_path_to_full(unreal.Paths.project_content_dir()))
    result={}
    for package in packages:
        if not package.startswith('/Game/'):continue
        base=content/package.removeprefix('/Game/')
        if not base.with_suffix('.uasset').is_file():raise ValueError('Save the face and DNA project assets before calibration')
        for extension in ('.uasset','.uexp','.ubulk'):
            file=base.with_suffix(extension)
            if file.is_file():result[str(file.resolve())]=digest(file)
    return result


def _pose_plans(request,manifest):
    """Validate all sealed pose groups before any native output is created."""
    names=manifest.get('pose_requests')
    if names is None:names=['pose-request.json'] if (request/'pose-request.json').is_file() else []
    if not isinstance(names,list) or len(names)>8 or any(not isinstance(name,str) for name in names) or len(set(names))!=len(names):raise ValueError('Supply up to eight distinct sealed native pose requests')
    plans=[];channels=set();solvers=set();poses=set()
    for name in names:
        if not isinstance(name,str) or Path(name).name!=name or not name.endswith('.json') or name not in manifest['prepared_sha256']:raise ValueError('Pose groups must be sealed JSON files beside their request')
        data=json.loads((request/name).read_text())
        if data.get('schema')!='CM_DNA_POSE_CORRECTIVES_V1' or not 1<=len(data.get('targets',[]))<=64 or not 1<=len(data.get('solvers',[]))<=16:raise ValueError('Each native pose group must satisfy the target and solver limits')
        def distinct(values,seen):
            for value in values:
                if value in seen:raise ValueError('Native pose groups must use distinct channels, solvers and pose names')
                seen.add(value)
        distinct((row['mesh']+'__'+row['channel'] for row in data['targets']),channels);distinct((row['name'] for row in data['solvers']),solvers);distinct((p['name'] for row in data['solvers'] for p in row['poses']),poses)
        plans.append((name,data))
    if len(plans)>1 and not (request/'driven-request.json').is_file() and sum(len(data['targets']) for _,data in plans)!=manifest.get('targets'):raise ValueError('The grouped target count does not match the sealed manifest')
    if manifest.get('method') in {'CAPTURED_NEUTRAL_LID_FIT','CAPTURED_NEUTRAL_LID_BLINK_FIT'}:
        compensated=manifest['method']=='CAPTURED_NEUTRAL_LID_BLINK_FIT'
        if (request/'driven-request.json').exists():raise ValueError('Keep neutral fitting separate from lower-LOD target changes')
        if manifest.get('lods')!=[0] or manifest.get('neutral_geometry_changed') is not True or manifest.get('globe_geometry_changed') is not False:
            raise ValueError('A neutral lid fit must contain explicit LOD0 neutral geometry')
        if compensated:
            allowed={'head_lod0_mesh','eyelashes_lod0_mesh','eyeEdge_lod0_mesh'}
            if not plans or sum(len(data['targets']) for _,data in plans)!=manifest.get('targets') or 'compiled-request.json' not in manifest['prepared_sha256']:
                raise ValueError('Seal the complete native blink compensation and compiled fields')
            for _,data in plans:
                if any(row['mesh'] not in allowed for row in data['targets']):raise ValueError('Blink compensation may change only head and eye attachments')
                for solver in data['solvers']:
                    if solver['inputs'] not in [['CTRL_expressions.eyeBlinkL'],['CTRL_expressions.eyeBlinkR']] or solver.get('function')!='LINEAR' or solver.get('radius')!=.25 or solver.get('gate_values') or [p['values'] for p in solver['poses']]!=[[0.],[.25],[.5],[.75],[1.]] or any(p.get('multipliers') for p in solver['poses']):
                        raise ValueError('Use independent quarter-strength scalar blink compensation')
        elif plans or manifest.get('targets')!=0 or (request/'compiled-request.json').exists():raise ValueError('Keep neutral-only fitting separate from expression target changes')
        if 'calibration-request.json' not in manifest['prepared_sha256']:raise ValueError('Seal the neutral calibration request')
        data=json.loads((request/'calibration-request.json').read_text())
        if data.get('schema')!='CM_DNA_CALIBRATION_V1' or data.get('correctives')!=[] or data.get('neutral_joints') or {row['mesh'] for row in data.get('neutral',[])}!={'head_lod0_mesh','eyelashes_lod0_mesh','eyeEdge_lod0_mesh'} or len(data['neutral'])!=3:
            raise ValueError('Neutral fitting may change only the head and its two eye attachments')
    elif manifest.get('globe_geometry_changed') is False and not manifest.get('preserved_globe_motion'):
        allowed={'head_lod0_mesh','eyelashes_lod0_mesh','eyeEdge_lod0_mesh'}
        if not plans or any(row['mesh'] not in allowed for _,data in plans for row in data['targets']):raise ValueError('Only explicit head/lid accessory fields can inherit unchanged globe motion')
    return plans


def validate_sources(request_folder):
    """Check sealed inputs and the current saved mesh/DNA before native writes."""
    import unreal
    request=Path(request_folder).resolve();manifest=json.loads((request/'manifest.json').read_text())
    if manifest.get('schema')!='CM_NATIVE_LOD_CORRECTIVE_REQUEST_V1' or manifest.get('status')!='prepared' or not manifest.get('prepared_sha256'):raise ValueError('Prepare a complete sealed native LOD corrective request first')
    native=manifest.get('native_source_sha256',{})
    for file,expected in manifest['input_sha256'].items():
        path=Path(file).resolve()
        package=path.suffix in {'.uasset','.uexp','.ubulk'} and native.get(file)==expected
        if (path.suffix not in {'.dna','.json','.f32','.i32','.u32','.npz'} and not package) or digest(path)!=expected:raise ValueError('A native corrective source changed; prepare the revision again')
    for file,expected in manifest['prepared_sha256'].items():
        path=(request/file).resolve()
        if path.parent!=request or digest(path)!=expected:raise ValueError('A prepared native corrective payload changed')
    source=unreal.load_asset(manifest['source_mesh'])
    if not isinstance(source,unreal.SkeletalMesh):raise ValueError('The captured source face is missing')
    package=source.get_path_name().split('.')[0]
    if manifest.get('source_asset_file'):
        current=Path(unreal.Paths.convert_relative_path_to_full(unreal.Paths.project_content_dir()))/(package.removeprefix('/Game/')+'.uasset')
        if current.resolve()!=Path(manifest['source_asset_file']).resolve() or digest(current)!=manifest['source_asset_sha256']:raise ValueError('The saved source rig changed or belongs to another project')
    if native and source_fingerprints(source)!=native:raise ValueError('The face or embedded DNA changed; capture and prepare the current saved rig again')
    if package in {p.get_path_name() for p in unreal.EditorLoadingAndSavingUtils.get_dirty_content_packages()}:raise ValueError('Save the source face edits before staging another native revision')
    return manifest,source


def stage(request_folder,destination):
    import unreal
    request=Path(request_folder).resolve();library=unreal.EditorAssetLibrary
    manifest,source=validate_sources(request)
    if not re.fullmatch(r'/Game(?:/[A-Za-z0-9_]+)+',destination) or library.does_asset_exist(destination):raise ValueError('Choose a fresh project skeletal-mesh revision')
    budget=runpy.run_path(str(Path(__file__).with_name('resource_budget.py')));budget['require']('native LOD corrective staging',8*budget['GIB'])
    calibrated=manifest.get('method') in {'CAPTURED_NEUTRAL_LID_FIT','CAPTURED_NEUTRAL_LID_BLINK_FIT'}
    staging_bytes=Path(manifest['source_dna']).stat().st_size*(2 if manifest.get('method')=='CAPTURED_NEUTRAL_LID_BLINK_FIT' else 1)+64*budget['MIB']
    if manifest.get('source_asset_file'):staging_bytes+=Path(manifest['source_asset_file']).stat().st_size
    budget['require_disk']('native LOD corrective staging',request,2*budget['GIB']+staging_bytes)
    package=source.get_path_name().split('.')[0]
    plans=_pose_plans(request,manifest);driven=(request/'driven-request.json').is_file()
    neutral=manifest.get('method')=='CAPTURED_NEUTRAL_LID_FIT'
    if not driven and not neutral and (manifest.get('lods')!=[0] or not plans):raise ValueError('A root-only revision needs LOD0 pose drivers')
    out=request/'native'
    if out.exists():raise ValueError('This native request has already been staged; retain it and prepare a fresh request')
    out.mkdir();dna=out/'corrected.dna';api=unreal.CreateMHCalibrationLibrary
    base=manifest['source_dna'];receipts=[];generated=[]
    if calibrated:
        calibrated_dna=out/('calibrated.dna' if plans else 'corrected.dna')
        if not api.write_calibration_dna(base,str(request/'calibration-request.json'),str(calibrated_dna)):raise RuntimeError('Native neutral-fit validation failed; the source rig was retained')
        base=str(calibrated_dna)
        if plans:generated.append(calibrated_dna)
    for index,(name,poses) in enumerate(plans):
        output=out/('corrected.dna' if index==len(plans)-1 and not driven else 'drivers.dna' if len(plans)==1 else f'drivers-{index:02d}.dna')
        if not api.write_pose_corrective_dna(base,str(request/name),str(output)):raise RuntimeError('Native factorized pose-driver validation failed; the source rig was retained')
        base=str(output);gates=[pose for solver in poses['solvers'] for pose in solver['poses'] if pose.get('multipliers')]
        linear=sum(solver.get('function')=='LINEAR' for solver in poses['solvers'])
        external_gates=sum(len(solver.get('gate_values',{})) for solver in poses['solvers'])
        if external_gates:
            receipt=json.loads(Path(base+'.drivers.json').read_text())
            if not receipt.get('native_driver_centers_valid') or receipt.get('external_gate_controls')!=external_gates:raise RuntimeError('Install bridge 0.27 or later for native gates independent of interpolation coordinates')
        if linear:
            receipt=json.loads(Path(base+'.drivers.json').read_text())
            if not receipt.get('native_driver_centers_valid') or receipt.get('linear_solvers')!=linear:raise RuntimeError('Install bridge 0.26 or later for validated linear pose interpolation')
        if gates:
            receipt=json.loads(Path(base+'.drivers.json').read_text())
            if not receipt.get('native_driver_centers_valid') or receipt.get('multiplicatively_gated_poses')!=len(gates) or receipt.get('native_zero_gate_samples')!=sum(len(pose['multipliers']) for pose in gates):raise RuntimeError('This bridge did not validate native multiplicative pose gates; install bridge 0.25 or later before staging')
        receipts.append(dict(request=name,request_sha256=digest(request/name),receipt=json.loads(Path(base+'.drivers.json').read_text())))
        # Previous groups are embedded in the newly validated DNA. Keep their
        # small receipts, and release only our own intermediate DNA payloads.
        if generated:
            previous=generated[-1]
            if previous.parent!=out or not (previous.name.startswith('drivers') or previous.name=='calibrated.dna'):raise RuntimeError('Unexpected native intermediate path')
            previous.unlink()
        generated.append(output)
    if driven and not api.write_driven_mesh_targets_dna(base,str(request/'driven-request.json'),str(dna)):raise RuntimeError('Native LOD target/driver validation failed; the source rig was retained')
    target=library.duplicate_asset(source.get_path_name(),destination)
    if not target:raise RuntimeError('Could not create the private native rig revision')
    (out/'partial-assets.json').write_text(json.dumps(dict(candidate=target.get_path_name(),source=source.get_path_name())))
    if not api.apply_calibration_to_mesh_copy(source,target,str(dna)):raise RuntimeError('Could not bind the revised LOD DNA; the source rig was retained')
    if not neutral and not api.apply_compiled_corrective_offsets(source,target,str(request/'compiled-request.json')):raise RuntimeError('Could not compile the revised LOD morphs; the source rig was retained')
    if package in {p.get_path_name() for p in unreal.EditorLoadingAndSavingUtils.get_dirty_content_packages()}:raise RuntimeError('The source face was unexpectedly dirtied; inspect it before publishing any revision')
    report=dict(status='complete',candidate=target.get_path_name(),source=source.get_path_name(),dna=str(dna),request_sha256=digest(request/'manifest.json'),targets=manifest['targets'],lods=manifest['lods'],native_motion_accepted=False,source_assets_saved=0)
    if len(plans)>1:
        report['pose_driver_groups']=receipts
        (out/'pose-groups.json').write_text(json.dumps(dict(groups=receipts,final_dna=str(dna),final_dna_sha256=digest(dna)),indent=2))
    library.set_metadata_tag(target,'CreateMetaHuman.LODCorrectiveRevision',json.dumps(report))
    if manifest.get('preserved_globe_motion'):
        lineage=dict(manifest['preserved_globe_motion'],candidate=target.get_path_name(),inherited_globe_revision=source.get_path_name(),dna=str(dna),native_motion_accepted=False)
        library.set_metadata_tag(target,'CreateMetaHuman.GlobeGazeRevision',json.dumps(lineage))
    elif manifest.get('globe_geometry_changed') is False:
        raw=library.get_metadata_tag(source,'CreateMetaHuman.GlobeGazeRevision')
        if raw:
            lineage=json.loads(raw);lineage.update(candidate=target.get_path_name(),inherited_globe_revision=source.get_path_name(),dna=str(dna),native_motion_accepted=False)
            library.set_metadata_tag(target,'CreateMetaHuman.GlobeGazeRevision',json.dumps(lineage));report['globe_motion_preserved']=True
    if not library.save_loaded_asset(target,False):raise RuntimeError('Could not save the private native LOD revision')
    (out/'stage.json').write_text(json.dumps(report,indent=2));return report


def capture_poses(mesh,samples,destination,*,lods,source_component=None):
    """Capture complete requested native LOD/control combinations, read-only."""
    import unreal
    destination=Path(destination).resolve();samples=list(samples);lods=list(lods)
    if destination.exists() or not samples or len(samples)>128 or len({row['name'] for row in samples})!=len(samples) or any(not row['name'].isalnum() for row in samples):raise ValueError('Choose a fresh capture folder and distinct named control samples')
    if not lods or len(set(lods))!=len(lods) or any(type(lod) is not int or lod<0 for lod in lods):raise ValueError('Choose distinct native face LOD indices')
    package=mesh.get_path_name().split('.')[0];source_file=None;source_sha=None
    dirty=lambda:package in {p.get_path_name() for p in unreal.EditorLoadingAndSavingUtils.get_dirty_content_packages()}
    if dirty():raise ValueError('Save the source face edits before capturing calibration poses')
    if package.startswith('/Game/'):
        source_file=Path(unreal.Paths.convert_relative_path_to_full(unreal.Paths.project_content_dir()))/(package.removeprefix('/Game/')+'.uasset')
        if not source_file.is_file():raise ValueError('Save the source face as a project asset before calibration capture')
        source_sha=digest(source_file)
    source_files=source_fingerprints(mesh)
    destination.mkdir(parents=True);report=dict(schema=2,status='capturing',candidate=mesh.get_path_name(),poses=[],source_assets_saved=0,source_asset_file=str(source_file.resolve()) if source_file else None,source_asset_sha256=source_sha,source_fingerprint_available=source_sha is not None,native_source_sha256=source_files)
    storage=runpy.run_path(str(Path(__file__).with_name('capture_storage.py')))['CaptureStorage'](destination)
    for lod in lods:
        for row in samples:
            name=f"LOD{lod}_{row['name']}";file=destination/(name+'.json')
            if not unreal.CreateMHExpressionLibrary.export_engine_control_pose(mesh,json.dumps(row['controls']),str(file),source_component,True,lod):
                report.update(status='failed',failed_pose=name);(destination/'capture.json').write_text(json.dumps(report,indent=2));raise RuntimeError('Native pose capture failed: '+name)
            storage.add(file);report['storage']=storage.report()
            report['poses'].append(dict(name=row['name'],lod=lod,controls=row['controls'],file=str(file)));(destination/'capture.json').write_text(json.dumps(report,indent=2))
    if dirty() or (source_file and digest(source_file)!=source_sha) or source_fingerprints(mesh)!=source_files:
        report.update(status='failed',source_changed_during_capture=True);(destination/'capture.json').write_text(json.dumps(report,indent=2));raise ValueError('The source face changed during capture; retain the samples for diagnosis and capture a fresh revision')
    report['status']='complete';(destination/'capture.json').write_text(json.dumps(report,indent=2));return report
