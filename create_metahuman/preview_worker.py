"""Render an isolated preview snapshot in a separate Blender process."""

from pathlib import Path
import json
import sys
import traceback
import bpy


def write_report(path, data):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(data, indent=2), encoding="utf-8")
    temporary.replace(path)


def main():
    request_path = Path(sys.argv[sys.argv.index("--") + 1])
    request = json.loads(request_path.read_text(encoding="utf-8"))
    report = Path(request["report"])
    try:
        scene = bpy.data.scenes[request["scene"]]
        if bpy.context.window:
            bpy.context.window.scene = scene
        scene.view_layers[0].update()
        if scene.render.engine in {'BLENDER_EEVEE_NEXT','BLENDER_EEVEE'} and any(
            obj.type == "LIGHT_PROBE" for obj in scene.objects
        ):
            print("CM_PREVIEW_BAKE_ENVIRONMENT", flush=True)
            with bpy.context.temp_override(scene=scene,view_layer=scene.view_layers[0]):
                print("CM_PREVIEW_PROBE_CONTEXT",bpy.context.scene.name,len(bpy.context.view_layer.objects),flush=True)
                result=bpy.ops.object.lightprobe_cache_bake("EXEC_DEFAULT",subset="ALL")
                if result!={"FINISHED"}:
                    raise RuntimeError("Could not bake Eevee environment illumination")
            scene.view_layers[0].update()
        print("CM_PREVIEW_RENDER_START", flush=True)
        bpy.ops.render.render(scene=scene.name, write_still=True)
        output = Path(scene.render.filepath)
        if not output.is_file() or output.stat().st_size == 0:
            raise RuntimeError("Renderer did not write the preview image")
        write_report(
            report,
            {
                "status": "complete",
                "output": str(output),
                "width": scene.render.resolution_x,
                "height": scene.render.resolution_y,
            },
        )
        print("CM_PREVIEW_RENDER_DONE", flush=True)
    except Exception as exc:
        write_report(
            report,
            {
                "status": "failed",
                "error": str(exc),
                "traceback": traceback.format_exc(),
            },
        )
        raise


if __name__ == "__main__":
    main()
