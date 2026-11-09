"""Stage the complete character before replacing a previous export delivery."""
import json
import re
import shutil
from pathlib import Path
from tempfile import TemporaryDirectory

import bpy


def files(manifest):
    if manifest.get('format') == 'GLB':
        names = {'manifest.json', manifest['mesh'], *manifest.get('support_files', [])}
    else:
        names = {'manifest.json', 'unreal_import.py', manifest['mesh'],*manifest.get('support_files',[])}
    names.update(manifest.get('maps', []))
    names.update(manifest.get('accessories', {}).values())
    if manifest.get('body_mesh'):
        names.add(manifest['body_mesh'])
    result = set()
    for name in names:
        path = Path(name)
        if not name or path.is_absolute() or path.drive or '..' in path.parts or '\\' in name or ':' in name:
            raise ValueError('Export manifest contains an invalid file path')
        result.add(path)
    return result


def checked(root, relative):
    path = root / relative
    if path.resolve() != path.absolute():
        raise ValueError('Export files must remain inside the bundle without linked paths')
    if path.exists() and not path.is_file():
        raise ValueError('Export destination must be a regular file: ' + str(relative))
    return path


def merge_support(source, destination, source_manifest, destination_manifest):
    """Keep dependencies declared by either part of a staged full-body export."""
    files(source_manifest)  # Validate paths before touching either directory.
    declared=list(destination_manifest.get('support_files',[]))
    for name in source_manifest.get('support_files',[]):
        relative=Path(name)
        origin=checked(source,relative);target=checked(destination,relative)
        if not origin.is_file():raise ValueError('Body export did not produce support file '+name)
        if target.is_file():
            if origin.read_bytes()!=target.read_bytes():
                raise ValueError('Head/body support file conflict: '+name)
        else:
            target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(origin,target)
        if name not in declared:declared.append(name)
    if declared:destination_manifest['support_files']=declared


def publish(stage, destination):
    current = json.loads((stage / 'manifest.json').read_text(encoding='utf-8'))
    pending = files(current)
    previous_manifest = checked(destination, Path('manifest.json'))
    previous = files(json.loads(previous_manifest.read_text(encoding='utf-8'))) if previous_manifest.is_file() else set()
    for relative in pending:
        if not checked(stage, relative).is_file():
            raise ValueError('Export did not produce ' + str(relative))
    affected = pending | previous
    targets = {relative: checked(destination, relative) for relative in affected}
    with TemporaryDirectory(prefix='.previous-', dir=stage.parent) as directory:
        backup = Path(directory)
        existed = set()
        for relative, target in targets.items():
            if target.is_file():
                copy = backup / relative
                copy.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(target, copy)
                existed.add(relative)
        # The manifest is the commit record. Publish it after all its artifacts.
        order = sorted(affected - {Path('manifest.json')}) + [Path('manifest.json')]
        try:
            for relative in order:
                target = targets[relative]
                if relative in pending:
                    target.parent.mkdir(parents=True, exist_ok=True)
                    (stage / relative).replace(target)
                else:
                    target.unlink(missing_ok=True)
        except BaseException:
            for relative in order:
                target = targets[relative]
                if relative in existed:
                    shutil.copy2(backup / relative, target)
                else:
                    target.unlink(missing_ok=True)
            raise


def export(context, settings):
    from . import mesh_ops

    name = re.sub(r'[^A-Za-z0-9_]', '_', settings.character_name).strip('_')
    if not name:
        raise ValueError('Enter a character name')
    parent = Path(bpy.path.abspath(settings.project_dir)).resolve() / 'Export'
    destination = parent / name
    if destination.resolve() != destination.absolute():
        raise ValueError('Export folder must remain inside this project without linked folders')
    destination.mkdir(parents=True, exist_ok=True)
    stage_before = settings.stage
    try:
        with TemporaryDirectory(prefix='.export-', dir=parent) as directory:
            stage = Path(directory)
            mesh_ops.export_bundle(context, settings, _root=stage)
            publish(stage, destination)
    except BaseException:
        settings.stage = stage_before
        raise
    settings.stage = 'EXPORT'
    return destination
