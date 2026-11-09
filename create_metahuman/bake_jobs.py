"""Owned background map baking with staged, reversible map delivery."""

from pathlib import Path
import hashlib, json, os, shutil, subprocess, uuid
import bpy
from bpy.app.handlers import persistent

_JOB = None
_REAPING = []


def file_hash(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def stage_maps(source, destination, mode, name, part,selected=None):
    """Copy only this bake's input maps, never unrelated characters or slots."""
    if mode not in {'WRINKLES', 'FINAL_MAPS','SKIN','BLOODFLOW'}:
        raise ValueError('Unknown map bake mode')
    prefixes = [name]
    if mode in {'FINAL_MAPS','SKIN'} and part == 'FULL_BODY':
        prefixes.append(name + '_Body')
    if selected is not None:prefixes=[prefix for prefix in prefixes if prefix in selected]
    kinds = ('ALBEDO',) if mode=='BLOODFLOW' else ('ALBEDO', 'NORMAL','ROUGHNESS') if mode=='SKIN' else ('ALBEDO','NORMAL') if mode=='WRINKLES' else ('ALBEDO', 'NORMAL', 'CAVITY')
    copied = []
    for prefix in prefixes:
        for kind in kinds:
            filename = prefix + '_' + kind + '.png'
            if Path(filename).name != filename:
                raise ValueError('Enter a character name without directory separators before baking')
            path = source / filename
            if path.is_file():
                shutil.copy2(path, destination / filename)
                copied.append(filename)
    return copied


def running():
    return _JOB is not None


def _clean_directory(job, name):
    folder = job["folder"].resolve()
    path = (folder / name).resolve()
    if path.parent != folder:
        raise ValueError("Bake cleanup path is outside its owned job")
    if path.is_dir():
        shutil.rmtree(path)


def deliver(job, files, finalize=None):
    destination = job["destination"]
    destination.mkdir(parents=True, exist_ok=True)
    incoming = job["folder"] / "incoming"
    backup = job["folder"] / "backup"
    incoming.mkdir()
    backup.mkdir()
    existed = set()
    changed = []
    try:
        for item in files:
            name = item["name"]
            if Path(name).name != name or not name.lower().endswith(".png"):
                raise ValueError("Invalid bake output filename")
            source = job["folder"] / "staging/Maps" / name
            if not item.get('delete') and file_hash(source) != item["sha256"]:
                raise ValueError("Bake output checksum changed")
            if not item.get('delete'):
                shutil.copy2(source, incoming / name)
            if (destination / name).exists():
                shutil.copy2(destination / name, backup / name)
                existed.add(name)
        for item in files:
            name = item["name"]
            if item.get('delete'):
                (destination / name).unlink(missing_ok=True)
            else:
                (incoming / name).replace(destination / name)
            changed.append(name)
        if finalize:finalize()
    except Exception:
        for name in reversed(changed):
            if name in existed:
                (backup / name).replace(destination / name)
            else:
                (destination / name).unlink(missing_ok=True)
        raise


def refresh_maps(scene, destination, files, *, kinds=('NORMAL', 'CAVITY')):
    """Refresh requested map nodes, retaining shaders and unrelated painted maps.

    Background detail bakes keep the default normal/cavity scope. A direct map
    bake explicitly names its output so an existing packed albedo can refresh
    without rebuilding the artist's material.
    """
    s, f = scene.cm_settings, scene.cm_features
    changed = {item["name"] for item in files if not item.get('delete')}
    loaded, plans, copies, data_copies = {}, [], {}, []
    parts=((s.result,'BODY' if s.part=='BODY' else 'HEAD'),
           (s.detail,'BODY' if s.part=='BODY' else 'HEAD'),
           (f.body_result,'BODY'),(f.body_detail,'BODY'))
    for obj,role in parts:
        if not obj:
            continue
        prefix=s.character_name+('_Body' if role=='BODY' and s.part=='FULL_BODY' else '')
        for slot,material in enumerate(obj.data.materials):
            if not material or not material.use_nodes:
                continue
            updates={}
            for node in material.node_tree.nodes:
                if node.type != "TEX_IMAGE" or not node.image:
                    continue
                kind=node.get('cm_map_kind')
                if kind in kinds:
                    name=prefix+'_'+kind+'.png'
                else:
                    name=Path(bpy.path.abspath(node.image.filepath)).name
                    if not any(name==prefix+'_'+kind+'.png' for kind in kinds):continue
                if name not in changed:continue
                updates[node.name]=name
            if updates:
                if not obj.is_editable or not obj.data.is_editable:
                    raise ValueError('Make the skin part local before applying baked maps')
                plans.append((obj,slot,material,role,updates))
    try:
        # Resolve every output before changing material bindings. Semantic map
        # tags survive a character rename; part prefixes prevent a head-only
        # bake from updating the body. Copies retain artist node graphs while
        # keeping templates and linked objects outside this delivery untouched.
        for obj,slot,material,role,updates in plans:
            for name in updates.values():
                if name not in loaded:
                    image = bpy.data.images.load(
                        str(destination / name), check_existing=False
                    )
                    image.colorspace_settings.name = (
                        "sRGB" if name.endswith(('_ALBEDO.png','_CM1.png','_CM2.png','_CM3.png')) else "Non-Color"
                    )
                    image.alpha_mode = "NONE"
                    image.pack()
                    loaded[name] = image
            key=(material.as_pointer(),role)
            if key not in copies:
                copy=material.copy();copies[key]=copy
                for node_name,name in updates.items():copy.node_tree.nodes[node_name].image=loaded[name]
        old_slots={obj:list(obj.data.materials) for obj,_,_,_,_ in plans}
        for obj in old_slots:
            if obj.data.users>1:
                previous=obj.data;private=previous.copy();data_copies.append((obj,previous,private));obj.data=private
        for obj,slot,material,role,updates in plans:
            obj.data.materials[slot]=copies[(material.as_pointer(),role)]
    except Exception:
        if 'old_slots' in locals():
            for obj,materials in old_slots.items():
                obj.data.materials.clear()
                for material in materials:obj.data.materials.append(material)
        for obj,previous,private in reversed(data_copies):
            obj.data=previous
            if not private.users:bpy.data.meshes.remove(private)
        for material in copies.values():
            if not material.users:bpy.data.materials.remove(material)
        for image in loaded.values():
            if not image.users:bpy.data.images.remove(image)
        raise


def disk_budget(context,s,f,mode,slots):
    """Estimate snapshot, staged inputs, outputs, incoming copies and backups."""
    from . import resource_budget as budget
    project=Path(bpy.path.abspath(s.project_dir)).resolve();maps=project/'Maps'
    prefixes=[s.character_name]
    if mode in {'FINAL_MAPS','SKIN'} and s.part=='FULL_BODY':prefixes.append(s.character_name+'_Body')
    if mode=='FINAL_MAPS':
        from . import textures
        prefixes=[s.character_name+('_Body' if role=='BODY' and s.part=='FULL_BODY' else '') for role in textures.final_parts(s,f)]
    if mode=='SKIN':
        from . import skin_bake
        prefixes=[item['prefix'] for item in skin_bake.plan(s,f)]
    kinds=('ALBEDO',) if mode=='BLOODFLOW' else ('ALBEDO','NORMAL','CAVITY') if mode=='FINAL_MAPS' else ('ALBEDO','NORMAL')
    inputs=[prefix+'_'+kind+'.png' for prefix in prefixes for kind in kinds]
    if any(Path(name).name!=name for name in inputs):raise ValueError('Enter a character name without directory separators before baking')
    input_bytes=sum((maps/name).stat().st_size for name in inputs if (maps/name).is_file())
    if mode=='SKIN':
        from . import skin_bake
        outputs={item['prefix']+'_'+kind+'.png':int(f.skin_map_size)
                 for item in skin_bake.plan(s,f) for kind in ('ALBEDO','NORMAL','ROUGHNESS')
                 if item.get({'ALBEDO':'albedo','NORMAL':'normal','ROUGHNESS':'roughness'}[kind])}
    elif mode=='FINAL_MAPS':
        outputs={prefix+'_'+kind+'.png':int(s.map_size) for prefix in prefixes for kind in ('NORMAL','CAVITY')}
    else:
        size=1024 if f.wrinkle_map_size=='DEFAULT' else int(f.wrinkle_map_size)
        outputs={} if mode=='BLOODFLOW' else {s.character_name+'_'+slot+'.png':size for slot in slots}
        if f.bloodflow or mode=='BLOODFLOW':
            albedo=maps/(s.character_name+'_ALBEDO.png')
            with albedo.open('rb') as stream:header=stream.read(24)
            if header[:8]!=b'\x89PNG\r\n\x1a\n':raise ValueError('Blood flow needs a saved PNG albedo')
            color_size=max(int.from_bytes(header[16:20],'big'),int.from_bytes(header[20:24],'big'))
            outputs.update({s.character_name+'_CM'+slot[-1]+'.png':color_size for slot in slots})
    # A conservative uncompressed 16-bit RGBA PNG allowance plus headers.
    output_bytes=sum(size*size*8+65536 for size in outputs.values())
    backup_bytes=sum((maps/name).stat().st_size for name in outputs if (maps/name).is_file())
    image_bytes=0
    source_images=[getattr(f,slot.lower()+'_mask') for slot in slots] if mode=='BLOODFLOW' else bpy.data.images
    for image in source_images:
        if not image:continue
        if image.type in {'RENDER_RESULT','COMPOSITING'}:continue
        encoded=image.packed_file.size if image.packed_file else 0
        if not encoded and image.filepath:
            source=Path(bpy.path.abspath(image.filepath,library=image.library))
            if source.is_file():encoded=source.stat().st_size
        resident=0
        # Asking for size can decode a lazy image. Budgeting must not load an
        # unrelated 8K texture before the snapshot has even been accepted.
        if image.has_data:
            width,height=image.size
            tiles=len(image.tiles) if image.source=='TILED' else 1
            resident=width*height*(16 if image.is_float else 8)*tiles
        image_bytes+=max(encoded,resident)
    snapshot_bytes=64*budget.MIB+image_bytes+(0 if mode=='BLOODFLOW' else budget.meshes(context.scene.objects))
    if mode!='BLOODFLOW' and bpy.data.filepath and Path(bpy.data.filepath).is_file():snapshot_bytes=max(snapshot_bytes,Path(bpy.data.filepath).stat().st_size)
    reserve=2*budget.GIB
    required=snapshot_bytes+input_bytes+2*output_bytes+backup_bytes+reserve
    # Serial RNM composition reads the reference once and overwrites its one
    # file-backed detail plane. Native image/worker RAM remains budgeted separately.
    scratch_size=(int(s.map_size) if mode=='FINAL_MAPS' else
                  int(f.skin_map_size) if mode=='SKIN' else
                  1024 if mode=='WRINKLES' and f.wrinkle_map_size=='DEFAULT' else
                  int(f.wrinkle_map_size) if mode=='WRINKLES' else 0)
    scratch_bytes=scratch_size**2*16 if scratch_size>4096 else 0
    required+=scratch_bytes
    return dict(snapshot_bytes=snapshot_bytes,staged_input_bytes=input_bytes,output_bytes=output_bytes,scratch_bytes=scratch_bytes,
                backup_bytes=backup_bytes,reserve_bytes=reserve,
                **budget.require_disk('map baking',project,required))


def start(context, s, f, mode, slots=None, *, defer=False):
    global _JOB
    from . import resource_budget
    if mode=='SKIN':
        from . import skin_bake
        skin_bake.preflight(context,s,f)
        if not defer:resource_budget.require('source skin baking',resource_budget.maps(f.skin_map_size,(s.source,s.result,f.body_result)))
    if mode=='BLOODFLOW':
        from . import textures
        albedo=textures.require_saved_bloodflow_albedo(s)
        with albedo.open('rb') as stream:header=stream.read(24)
        width,height=int.from_bytes(header[16:20],'big'),int.from_bytes(header[20:24],'big')
        if header[:8]!=b'\x89PNG\r\n\x1a\n' or not 1<=width<=8192 or not 1<=height<=8192:raise ValueError('Expression color needs a saved head PNG through 8K')
        if not defer:resource_budget.require('expression-color baking',width*height*64+512*resource_budget.MIB)
    elif mode=='WRINKLES' and f.bloodflow:
        from . import textures
        textures.require_saved_bloodflow_albedo(s)
    if not defer and not _JOB and mode in {'FINAL_MAPS','WRINKLES'}:
        size=s.map_size if mode=='FINAL_MAPS' else 1024 if f.wrinkle_map_size=='DEFAULT' else f.wrinkle_map_size
        resource_budget.require('map baking',resource_budget.maps(size,(s.result,s.detail,f.body_result,f.body_detail,f.wm1,f.wm2,f.wm3)))
    if defer:
        if mode not in {'FINAL_MAPS','SKIN'}:raise ValueError('External staging currently supports final maps and source skin')
        size=int(f.skin_map_size if mode=='SKIN' else s.map_size)
        # Freezing packed images still needs source/copy pixel buffers, even
        # though the bake itself will run only after this Blender has closed.
        resource_budget.require('map snapshot preparation',512*resource_budget.MIB+4*size*size*16+
            resource_budget.meshes((s.source,s.result,s.detail,f.body_result,f.body_detail)))
    if mode == 'FINAL_MAPS':
        from . import normal_composition,textures
        final_roles=textures.final_parts(s,f)
        for role in final_roles:
            with textures.skin_part(s,f,role):normal_composition.validate(s.result,s.detail)
        from . import skin_bake_layers
        skin_bake_layers.preflight(context.scene,roles=final_roles)
        from . import scan_projection
        scan_projection.require_applied(s.detail if 'HEAD' in final_roles or s.part!='FULL_BODY' else None,
            f.body_detail if s.part=='FULL_BODY' and 'BODY' in final_roles else None)
    from . import detail_jobs
    from . import wrinkle_jobs
    from . import wrinkles
    if mode!='BLOODFLOW':wrinkles.refresh_source_state(context,s,f)
    if wrinkle_jobs.running() or mode in {'WRINKLES','FINAL_MAPS'} and any(obj and obj.get('cm_wrinkle_update_pending') for obj in (f.wm1,f.wm2,f.wm3)):
        raise ValueError('Update the retained wrinkle sculpts before baking maps')
    if detail_jobs.running():
        raise ValueError('Finish or cancel the detail update before starting a bake')
    if _JOB:
        raise ValueError("A map bake is already running; finish or cancel it first")
    if (
        context.mode != "OBJECT"
        or f.body_editing
        or f.full_sculpt
        or f.restore_editing
        or f.expression_active
    ):
        raise ValueError(
            "Finish the current sculpt, body or expression preview before baking"
        )
    if mode not in {"WRINKLES", "FINAL_MAPS","SKIN","BLOODFLOW"}:
        raise ValueError("Unknown map bake mode")
    slots = list(slots or [])
    if mode=='BLOODFLOW':
        if s.part=='BODY' or not s.result or not slots or len(set(slots))!=len(slots) or set(slots)-{'WM1','WM2','WM3'}:raise ValueError('Choose head expression-color slots to bake')
    if mode == "WRINKLES":
        if not slots:
            raise ValueError("Create a wrinkle base before baking")
        for slot in slots:
            if slot not in {"WM1", "WM2", "WM3"} or not getattr(f, slot.lower()):
                raise ValueError("Create the selected wrinkle base first")
    project = Path(bpy.path.abspath(s.project_dir)).resolve()
    disk=disk_budget(context,s,f,mode,slots)
    if mode == 'FINAL_MAPS':
        from . import detail_layers
        if not defer and not bpy.app.background and detail_jobs.start(context, s, f, 'FINAL_MAPS'):
            return detail_jobs._JOB['folder']
        detail_layers.sync_character(context, s, f,roles=final_roles)
    from . import sculpt_session
    from . import map_provenance
    map_basis=map_provenance.prepare(context.scene,mode,slots) if mode!='SKIN' else {}
    skin_plan=skin_bake.plan(s,f) if mode=='SKIN' else None
    skin_basis=skin_bake.signature(context.scene,skin_plan) if mode=='SKIN' else None
    map_state=(s.project_dir,s.character_name,s.part)
    sculpt_basis = sculpt_session.bake_started(context.scene,mode,slots) if mode not in {'SKIN','BLOODFLOW'} else None
    folder = project / "Bakes" / ".jobs" / uuid.uuid4().hex[:12]
    folder.mkdir(parents=True)
    staging = folder / "staging"
    maps = staging / "Maps"
    maps.mkdir(parents=True)
    destination = project / "Maps"
    staged_inputs = stage_maps(destination, maps, mode, s.character_name, s.part,
        selected=[item['prefix'] for item in skin_plan] if skin_plan else
            [s.character_name+('_Body' if role=='BODY' and s.part=='FULL_BODY' else '') for role in final_roles] if mode=='FINAL_MAPS' else None)
    snapshot = folder / "snapshot.blend"
    scene = bpy.data.scenes.new('CM_ExpressionColor') if mode=='BLOODFLOW' else context.scene.copy()
    try:
        scene.cm_settings.project_dir = str(staging)
        if mode=='BLOODFLOW':
            scene.cm_settings.character_name=s.character_name;scene.cm_settings.part='HEAD'
            scene.cm_features.bloodflow_strength=f.bloodflow_strength
            for slot in slots:setattr(scene.cm_features,slot.lower()+'_mask',getattr(f,slot.lower()+'_mask'))
        sculpt_session.write_snapshot(scene,snapshot)
        scene_name = scene.name
    finally:
        bpy.data.scenes.remove(scene)
    request = {
        "mode": mode,
        "slots": slots,
        "snapshot": str(snapshot),
        "scene": scene_name,
        "report": str(folder / "report.json"),
        "staging": str(staging),
        "addon_root": str(Path(__file__).resolve().parents[1]),
        "parent": 0 if defer else os.getpid(),
        "staged_inputs": staged_inputs,
        "disk_budget":disk,
        "skin_plan":skin_plan,
        "snapshot_scope":'EXPRESSION_MASKS_ONLY' if mode=='BLOODFLOW' else 'CHARACTER_SCENE',
        "snapshot_bytes":snapshot.stat().st_size,
    }
    request_path = folder / "request.json"
    request_path.write_text(json.dumps(request, indent=2))
    if defer:
        from . import bl_info
        size=f.skin_map_size if mode=='SKIN' else s.map_size
        metadata=dict(schema_version=1,addon_version=list(bl_info['version']),project_dir=str(project),
            character_name=s.character_name,mode=mode,slots=slots,sculpt_basis=sculpt_basis,
            map_basis=map_basis,map_state=list(map_state),skin_basis=skin_basis,skin_plan=skin_plan,
            required_memory_bytes=resource_budget.maps(size,(s.source,s.result,s.detail,f.body_result,f.body_detail)))
        (folder/'external.json').write_text(json.dumps(metadata,indent=2),encoding='utf-8')
        f.map_bake_status='External bake prepared; launch it after closing Blender. Existing maps are retained.'
        s.status=f.map_bake_status
        return folder
    with (folder / "worker.log").open("w", encoding="utf-8") as log:
        process = subprocess.Popen(
            [
                bpy.app.binary_path,
                "--background",
                "--factory-startup",
                "--python-exit-code",
                "1",
                "--python",
                str(Path(__file__).with_name("bake_worker.py")),
                "--",
                str(request_path),
            ],
            stdout=log,
            stderr=subprocess.STDOUT,
            creationflags=(
                subprocess.CREATE_NO_WINDOW | subprocess.BELOW_NORMAL_PRIORITY_CLASS
            )
            if os.name == "nt"
            else 0,
        )
    _JOB = {
        "folder": folder,
        "process": process,
        "scene": context.scene,
        "destination": destination,
        "mode": mode,
        "slots":slots,
        "cancel_requested": False,
        "open_folder": f.bake_open_folder,
        "sculpt_basis": sculpt_basis,
        "map_basis":map_basis,
        "map_state":map_state,
        "skin_basis":skin_basis,"skin_plan":skin_plan,
    }
    f.map_bake_status = "Baking snapshot in background..."
    s.status = "Background map bake started; you can keep editing."
    bpy.app.timers.register(poll, first_interval=0.2, persistent=True)
    return folder


def poll():
    global _JOB
    job = _JOB
    if not job:
        return None
    code = job["process"].poll()
    if code is None:
        try:
            features = job["scene"].cm_features
            report_path = job["folder"] / "report.json"
            if report_path.is_file():
                report = json.loads(report_path.read_text())
                if report.get("status") == "running":
                    features.map_bake_status = "Background bake: " + str(
                        report.get("slot", "maps")
                    )
                    if report.get('phase'):
                        features.map_bake_status += ' / ' + str(report['phase'])
        except ReferenceError:
            cancel_active()
            return None
        except (ValueError, OSError):
            pass
        return 0.2
    cleanup = False
    try:
        s = job["scene"].cm_settings
        if job["cancel_requested"]:
            s.status = "Map bake canceled; previous maps retained."
            cleanup = True
        else:
            report_path = job["folder"] / "report.json"
            report = (
                json.loads(report_path.read_text()) if report_path.is_file() else {}
            )
            if code or report.get("status") != "complete":
                raise RuntimeError(
                    report.get(
                        "error",
                        "Bake worker stopped. See " + str(job["folder"] / "worker.log"),
                    )
                )
            from . import map_provenance
            if 'map_basis' in job:
                try:
                    if (s.project_dir,s.character_name,s.part)!=job['map_state']:
                        raise ValueError('Project changed while baking; previous maps retained. Bake again in the intended project.')
                    map_provenance.validate(job['scene'],job['map_basis'])
                except Exception:
                    cleanup=True  # Untouched destination; stale snapshots can be discarded.
                    raise
            if job['mode']=='SKIN':
                from . import skin_bake
                if skin_bake.signature(job['scene'],job['skin_plan'])!=job['skin_basis']:
                    cleanup=True
                    raise ValueError('Source, selected skin, paint or bake settings changed; previous maps retained. Bake again when ready.')
                expected={item['prefix']+'_'+kind+'.png' for item in job['skin_plan']
                    for kind in ('ALBEDO','NORMAL','ROUGHNESS') if item.get(
                        {'ALBEDO':'albedo','NORMAL':'normal','ROUGHNESS':'roughness'}[kind],kind=='ALBEDO')}
                if not {item['name'] for item in report['files']}<=expected:
                    raise ValueError('Skin worker returned maps outside its selected scope')
                finalize=lambda:skin_bake.finish(job['scene'],job['destination'],report['files'],job['skin_plan'])
            else:
                if job['mode']=='FINAL_MAPS':
                    expected={s.character_name+('_Body' if key=='FINAL_BODY' and s.part=='FULL_BODY' else '')+'_'+kind+'.png'
                        for key in job['map_basis'] for kind in ('NORMAL','CAVITY')}
                    if not {item['name'] for item in report['files']}<=expected:
                        raise ValueError('Final-map worker returned files outside its prepared part scope')
                if job['mode']=='BLOODFLOW':
                    expected={s.character_name+'_CM'+slot[-1]+'.png' for slot in job['slots']}
                    if {item['name'] for item in report['files']}!=expected:raise ValueError('Expression-color worker returned an incomplete or unrelated map set')
                def finalize():
                    if 'map_basis' in job:map_provenance.record(job['scene'],job['map_basis'])
                    if report.get('skin_layers'):
                        from . import skin_bake_layers
                        skin_bake_layers.accept_bake(job['scene'],report['skin_layers'])
            deliver(job, report["files"],finalize=finalize)
            if job["mode"] == "FINAL_MAPS":
                refresh_maps(job["scene"], job["destination"], report["files"])
            elif job['mode']=='BLOODFLOW':refresh_maps(job['scene'],job['destination'],report['files'],kinds=('CM1','CM2','CM3'))
            from . import sculpt_session
            sculpt_session.bake_completed(job)
            s.status = (
                "Background maps ready: "
                + str(len(report["files"]))
                + " file(s) updated."
            )
            cleanup = True
            if job["open_folder"] and not bpy.app.background and os.name == "nt":
                os.startfile(str(job["destination"]))
    except Exception as exc:
        try:
            job["scene"].cm_settings.status = "Map bake failed; " + str(exc)
        except ReferenceError:
            pass
        print("MetaHuman Maker background bake:", exc)
    finally:
        try:
            job["scene"].cm_features.map_bake_status = ""
        except ReferenceError:
            pass
        _JOB = None
        if cleanup:
            _cleanup(job)
        scene=job.get('scene')
        if scene:
            try:
                if scene.cm_features.ui_stage=='EXPORT':
                    from . import guided
                    with bpy.context.temp_override(scene=scene,view_layer=scene.view_layers[0]):
                        problems=guided.check_delivery(bpy.context,scene.cm_settings,scene.cm_features)
                    scene.cm_features.delivery_report='\n'.join(problems) if problems else 'Character checks passed.'
                    scene.cm_features.last_action_error=bool(problems)
            except (ReferenceError,ValueError,OSError,KeyError) as exc:
                try:
                    scene.cm_features.delivery_report='Check character before delivery: '+str(exc)
                    scene.cm_features.last_action_error=True
                except ReferenceError:pass
        for window in bpy.context.window_manager.windows:
            for area in window.screen.areas:
                area.tag_redraw()
    return None


def cancel():
    if _JOB:
        _JOB["cancel_requested"] = True
        if _JOB["process"].poll() is None:
            _JOB["process"].terminate()


def _cleanup(job):
    try:
        (job["folder"] / "snapshot.blend").unlink(missing_ok=True)
    except OSError as exc:
        print("Bake snapshot cleanup:", exc)
    for name in ["staging", "incoming", "backup"]:
        try:
            _clean_directory(job, name)
        except OSError as exc:
            print("Bake cleanup:", exc)


def reap():
    for job in list(_REAPING):
        if job["process"].poll() is not None:
            _cleanup(job)
            _REAPING.remove(job)
    return 0.2 if _REAPING else None


@persistent
def cancel_active(_=None):
    global _JOB
    if _JOB:
        cancel()
        job = _JOB
        _JOB = None
        if bpy.app.timers.is_registered(poll):
            bpy.app.timers.unregister(poll)
        try:
            job["scene"].cm_features.map_bake_status = ""
        except ReferenceError:
            pass
        # Windows may still hold output files immediately after TerminateProcess.
        # Reap the same owned handle after it exits, including across file loads.
        job.pop("scene", None)
        _REAPING.append(job)
        if not bpy.app.timers.is_registered(reap):
            bpy.app.timers.register(reap, first_interval=0.2, persistent=True)


class CM_OT_cancel_map_bake(bpy.types.Operator):
    bl_idname = "cm.cancel_map_bake"
    bl_label = "Cancel Background Bake"

    def execute(self, context):
        cancel()
        return {"FINISHED"}


CLASSES = (CM_OT_cancel_map_bake,)


def register():
    for handlers in (bpy.app.handlers.load_pre,bpy.app.handlers.undo_pre,bpy.app.handlers.redo_pre):
        handlers.append(cancel_active)


def unregister():
    cancel_active()
    for handlers in (bpy.app.handlers.load_pre,bpy.app.handlers.undo_pre,bpy.app.handlers.redo_pre):
        if cancel_active in handlers:
            handlers.remove(cancel_active)
