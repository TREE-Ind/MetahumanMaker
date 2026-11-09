"""Load Epic's remote client from the user's engine; do not redistribute it."""

import importlib.util
import json
from pathlib import Path
import time


def push_bundle(engine_dir, bundle, project_name="", receipt_path=None, open_editor=False, on_dispatch=None):
    script = Path(bundle) / "unreal_import.py"
    lines = [
            "import json, runpy, unreal",
            "from pathlib import Path",
            "manifest = Path(" + json.dumps(str(Path(bundle) / "manifest.json")) + ")",
            "data = json.loads(manifest.read_text(encoding='utf-8'))",
            "base = data['destination']",
            "destination = base",
            "revision = 2",
            "while unreal.EditorAssetLibrary.does_directory_exist(destination):",
            "    destination = base + '_v' + str(revision).zfill(3)",
            "    revision += 1",
            "data['destination'] = destination",
            "push_manifest = manifest.with_name('unreal-push-manifest.json')",
            "push_manifest.write_text(json.dumps(data), encoding='utf-8')",
            "report = runpy.run_path("
            + json.dumps(str(script))
            + ')["import_bundle"](push_manifest)',
            "assets = [report['character']] if report.get('character') else report.get('mesh_assets', [])",
    ]
    def receipt_lines(indent=''):
        return [indent + line for line in [
            'receipt = Path(' + json.dumps(str(receipt_path)) + ')',
            "pending = receipt.with_suffix('.tmp')",
            "pending.write_text(json.dumps(report), encoding='utf-8')",
            'pending.replace(receipt)',
        ]]
    if receipt_path:
        identity_script = str(Path(__file__).with_name('process_ref.py'))
        lines[0:0] = [
            'import os, json, runpy', 'from pathlib import Path',
            'identity = runpy.run_path(' + json.dumps(identity_script) + ')["identify"](os.getpid())',
            'Path(' + json.dumps(str(Path(receipt_path).with_name('dispatch.json'))) + ').write_text(json.dumps(identity), encoding="utf-8")',
        ]
        # Commit import success before any optional editor-window work.
        lines.extend(receipt_lines())
    if open_editor:
        lines.extend([
            "if report.get('character'):",
            "    styles = unreal.get_default_object(unreal.load_class(None, '/Script/UnrealEd.EditorStyleSettings'))",
            "    old_location = styles.get_editor_property('AssetEditorOpenLocation')",
            '    try:',
            "        styles.set_editor_property('AssetEditorOpenLocation', unreal.AssetEditorOpenLocation.MAIN_WINDOW)",
            "        report['editor_opened'] = bool(unreal.get_editor_subsystem(unreal.AssetEditorSubsystem).open_editor_for_assets([unreal.load_asset(report['character'])]))",
            '    except Exception as exc:',
            "        report['editor_warning'] = str(exc)",
            '    finally:',
            "        styles.set_editor_property('AssetEditorOpenLocation', old_location)",
        ])
    # Reveal a successful Creator handoff directly. Browser synchronization can
    # raise a floating window over it; retain that fallback for mesh-only pushes
    # or an editor that could not open, without changing committed import success.
    lines.extend([
        "report['browser_revealed'] = False",
        "if assets and not report.get('editor_opened', False):",
        '    try:',
        '        unreal.EditorAssetLibrary.sync_browser_to_objects(assets)',
        "        report['browser_revealed'] = True",
        '    except Exception as exc:',
        "        report['browser_warning'] = str(exc)",
    ])
    if receipt_path:
        # Write from Unreal itself: a lost client connection is not proof that
        # import stopped, and must never cause an automatic duplicate push.
        lines = ['import json, runpy, unreal', 'from pathlib import Path', 'report = {}', 'try:'] + ['    ' + line for line in lines] + [
            'except Exception as exc:',
            "    report = dict(report, status='failed', error=str(exc))",
            '    raise',
            'finally:',
        ] + receipt_lines('    ')
    command = '\n'.join(lines)
    compile(command, '<MetaHuman Maker Unreal Push>', 'exec')
    return run_python(engine_dir, command, project_name, on_dispatch=on_dispatch)


def character_action(engine_dir, character, folder, project_name='', open_editor=False, on_dispatch=None, expected_project='', build_options=None, texture_update=False, eye_options=None,artist_destination='',artist_actor='',web_capture=None,texture_sources=False):
    folder = Path(folder)
    if texture_sources:
        command='import runpy\nrunpy.run_path('+repr(str(folder/'unreal_texture_sources.py'))+")[\"dispatch\"]("+repr(character)+','+repr(str(folder))+',expected_project='+repr(expected_project)+')'
        compile(command,'<MetaHuman Maker Texture Sources>','exec')
        return run_python(engine_dir,command,project_name,on_dispatch=on_dispatch)
    if web_capture is not None:
        command='import runpy\nrunpy.run_path('+repr(str(folder/'unreal_web_capture.py'))+")[\"dispatch\"]("+repr(character)+','+repr(str(folder))+','+repr(web_capture)+',expected_project='+repr(expected_project)+')'
        compile(command,'<MetaHuman Maker Native Web Capture>','exec')
        return run_python(engine_dir,command,project_name,on_dispatch=on_dispatch)
    action = 'UPDATE_TEXTURES' if texture_update else 'BUILD_CHARACTER' if build_options is not None else 'OPEN_CHARACTER' if open_editor else 'CHARACTER_STATUS'
    call=("runpy.run_path(str(folder / 'unreal_assembly.py'))['assemble']("+json.dumps(character)+
          ', build_options='+repr(build_options)+', expected_project='+json.dumps(expected_project)+
          ", progress_path=str(folder.parent / 'progress.json'))" if build_options is not None else
          "runpy.run_path(str(folder / 'unreal_character.py'))['inspect_character']("+json.dumps(character)+
          ', open_editor='+repr(open_editor)+', expected_project='+json.dumps(expected_project)+')')
    if texture_update:
        call = "runpy.run_path(str(folder / 'unreal_texture_update.py'))['update'](str(folder.parent / 'request.json'))"
    if eye_options is not None:
        action='UPDATE_EYES'
        call="runpy.run_path(str(folder / 'unreal_eyes.py'))['update']("+repr(character)+','+repr(eye_options)+', expected_project='+repr(expected_project)+')'
    if artist_destination:
        action='CAPTURE_ARTIST_EDITS'
        call="runpy.run_path(str(folder / 'unreal_artist_state.py'))['capture']("+repr(character)+','+repr(str(artist_destination))+', expected_project='+repr(expected_project)+', actor_path='+repr(artist_actor)+')'
    lines = [
        'import json, runpy, os', 'from pathlib import Path',
        'folder = Path(' + json.dumps(str(folder)) + ')',
        "receipt = folder.parent / 'receipt.json'",
        'report = ' + repr({'character': character, 'action': action}),
        'try:',
        "    identity = runpy.run_path(str(folder / 'process_ref.py'))['identify'](os.getpid())",
        "    (folder.parent / 'dispatch.json').write_text(json.dumps(identity), encoding='utf-8')",
        '    result = '+call,
        '    report.update(result)',
        'except Exception as exc:',
        "    report.update(status='failed', error=str(exc))",
        'finally:',
        "    pending = receipt.with_suffix('.tmp')",
        "    pending.write_text(json.dumps(report), encoding='utf-8')",
        '    pending.replace(receipt)',
    ]
    command = '\n'.join(lines)
    compile(command, '<MetaHuman Maker Character Status>', 'exec')
    return run_python(engine_dir, command, project_name, on_dispatch=on_dispatch)


def run_python(engine_dir, command, project_name="", on_dispatch=None):
    # Unreal exposes one command socket. Yield an idle live-review lease before
    # another add-on workflow connects; never interrupt an in-flight command.
    import sys
    review=sys.modules.get(__package__+'.native_face_review')
    if review:review.release_transport()
    client = (
        Path(engine_dir)
        / "Engine/Plugins/Experimental/PythonScriptPlugin/Content/Python/remote_execution.py"
    )
    if not client.is_file():
        raise ValueError(
            "Choose the Unreal install root containing Engine/, e.g. UE_5.8"
        )
    spec = importlib.util.spec_from_file_location("cm_ue_remote", client)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    remote = module.RemoteExecution()
    try:
        remote.start()
        deadline = time.monotonic() + 4
        nodes = []
        while time.monotonic() < deadline:
            nodes = remote.remote_nodes
            if project_name:
                nodes = [n for n in nodes if n.get("project_name") == project_name]
            time.sleep(0.1)
        if not nodes:
            raise RuntimeError(
                "No matching Unreal editor. Enable Python Remote Execution and keep UE open."
            )
        if len(nodes) > 1:
            raise RuntimeError(
                "Multiple Unreal editors found. Set Unreal Project Name to choose one."
            )
        remote.open_command_connection(nodes[0]["node_id"])
        if on_dispatch:
            on_dispatch()
        return remote.run_command(command, unattended=True, raise_on_failure=True)
    finally:
        remote.stop()
