"""Open a private sculpting project in its own interactive Blender window."""

from pathlib import Path
import sys
import shutil
import bpy

root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root))
import create_metahuman

create_metahuman.register()
from create_metahuman import sculpt_session, process_ref


def initialize(folder):
    manifest = sculpt_session.read_json(folder / "session.json")
    if manifest.get("author_initialized"):
        bpy.ops.wm.open_mainfile(filepath=manifest["author_project"])
    else:
        # The isolated snapshot is a scene library, not a saved window layout.
        # Append it into this window, then save a regular, resumable Blender file.
        original = bpy.context.scene
        old_objects = list(original.objects)
        snapshot = manifest.get("snapshot")
        if not snapshot:
            snapshot = str(Path(manifest["author_project"]).with_name("snapshot.blend"))
            shutil.copy2(manifest["author_project"], snapshot)
        with bpy.data.libraries.load(snapshot, link=False) as (source, target):
            target.scenes = source.scenes[:1]
        if not target.scenes or not target.scenes[0]:
            raise ValueError("The sculpting snapshot contains no scene")
        bpy.context.window.scene = target.scenes[0]
        bpy.data.batch_remove(ids=[original, *old_objects])
    s, f = bpy.context.scene.cm_settings, bpy.context.scene.cm_features
    f.sculpt_session_role = "AUTHOR"
    f.sculpt_session_path = str(folder)
    f.ui_stage = "DETAIL"
    from create_metahuman import detail_jobs
    sculpt_session.isolate_assets(bpy.context.scene, folder)
    if not manifest.get("author_initialized"):
        bpy.ops.wm.save_as_mainfile(filepath=manifest["author_project"])
        manifest["author_initialized"] = True
        sculpt_session.write_json(folder / "session.json", manifest)
    sculpt_session.write_json(
        folder / "author-ready.json",
        {
            "process": process_ref.identify(__import__("os").getpid()),
            "active": True,
            "project": manifest["author_project"],
        },
    )
    sculpt_session.attach_author()
    try:
        detail_jobs.start(bpy.context,s,f)
    except Exception as exc:
        s.status='Saved sculpt needs recovery: '+str(exc)

    def arrange():
        from create_metahuman import guided

        pending = False
        for area in bpy.context.screen.areas:
            if area.type == "VIEW_3D":
                area.spaces.active.show_region_ui = True
                area.spaces.active.shading.type = "MATERIAL"
                region = next((r for r in area.regions if r.type == "UI"), None)
                if region and not region.is_property_readonly("active_panel_category"):
                    region.active_panel_category = "MetaHuman Maker"
                else:
                    pending = True
                area.tag_redraw()
        guided.focus(
            bpy.context,
            [
                s.result if detail_jobs.running() else s.detail or s.result,
                (f.body_result if detail_jobs.running() else f.body_detail or f.body_result) if s.part == "FULL_BODY" else None,
                f.eye_left,
                f.eye_right,
                f.teeth,
            ],
        )
        f.sculpt_session_status = (
            "Private sculpting copy. Create Final Maps, then Save & Return."
        )
        return 0.25 if pending else None

    bpy.app.timers.register(arrange, first_interval=0.5)
    print("SEPARATE_SCULPT_SESSION_READY", folder, flush=True)


folder = Path(sys.argv[sys.argv.index("--") + 1]).resolve().parent
try:
    initialize(folder)
except Exception as exc:
    import traceback

    traceback.print_exc()
    sculpt_session.write_json(folder / "startup-error.json", {"error": str(exc)})

    def close_failed_startup():
        bpy.ops.wm.quit_blender("EXEC_DEFAULT")

    bpy.app.timers.register(close_failed_startup, first_interval=0.25)
