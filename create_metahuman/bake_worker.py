"""Run map operations against an immutable snapshot in a separate Blender."""

from pathlib import Path
import json, sys, traceback
import bpy


def main():
    request = json.loads(Path(sys.argv[sys.argv.index("--") + 1]).read_text())
    if request.get("parent"):
        import runpy

        runpy.run_path(str(Path(__file__).with_name("worker_lifetime.py")))[
            "watch_parent"
        ](request["parent"])
    report = Path(request["report"])

    def write(data):
        temporary = report.with_suffix(".tmp")
        temporary.write_text(json.dumps(data, indent=2))
        temporary.replace(report)

    try:
        sys.path.insert(0, request["addon_root"])
        import create_metahuman

        if request.get('addon_version') and list(create_metahuman.bl_info['version'])!=request['addon_version']:
            raise ValueError('External bake add-on version changed after preparation; retain the snapshot and install the original version')
        create_metahuman.register()
        bpy.ops.wm.open_mainfile(filepath=request["snapshot"])
        scene = bpy.data.scenes[request["scene"]]
        scene['cm_bake_worker']=True
        scene['cm_bake_progress_file']=str(report)
        if bpy.context.window:
            bpy.context.window.scene = scene
        with bpy.context.temp_override(scene=scene, view_layer=scene.view_layers[0]):
            from create_metahuman import textures
            from create_metahuman.bake_jobs import file_hash

            s, f = scene.cm_settings, scene.cm_features
            folder = Path(request["staging"]) / "Maps"
            before = {
                p.name: file_hash(p)
                for p in folder.glob("*.png")
            }
            if request["mode"] == "WRINKLES":
                skin_refreshed=False
                for slot in request["slots"]:
                    write({"status": "running", "slot": slot})
                    # The frozen head/source and resolution are common to this
                    # batch. Refresh their baseline once, not once per slot.
                    needs_skin=bool(getattr(f,slot.lower()).get('cm_wrinkle_native') and textures.generated_skin_source(s,f))
                    textures.bake_wrinkle(bpy.context, s, f, slot,refresh_skin=not skin_refreshed)
                    skin_refreshed=skin_refreshed or needs_skin
            elif request['mode']=='BLOODFLOW':
                for slot in request['slots']:
                    write({'status':'running','slot':'CM'+slot[-1]})
                    textures.bake_bloodflow(bpy.context,s,f,slot)
            elif request['mode']=='SKIN':
                write({'status':'running','slot':'selected source skin'})
                textures.bake_skin(bpy.context,s,f)
            else:
                textures.final_maps(bpy.context, s, f)
            files = []
            for path in folder.glob("*.png"):
                digest = file_hash(path)
                if digest != before.get(path.name):
                    files.append({"name": path.name, "sha256": digest})
            write({"status": "complete", "files": files,
                   "skin_layers":json.loads(scene.get('cm_bake_skin_layer_results','[]'))})
        print("BACKGROUND_MAP_BAKE_COMPLETE", flush=True)
    except Exception as exc:
        write(
            {"status": "failed", "error": str(exc), "traceback": traceback.format_exc()}
        )
        raise


if __name__ == "__main__":
    main()
