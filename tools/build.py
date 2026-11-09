"""Build the Blender add-on and its native helper without character assets."""

from pathlib import Path
import hashlib
import json
import zipfile
import ast
import argparse
from package_files import files, PACKAGE_SLUG, BRIDGE_PACKAGE_NAME

parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--addon-only',action='store_true',help='Keep already published native bridge archives unchanged')
args=parser.parse_args()

root = Path(__file__).resolve().parents[1]
destination = root / "dist"
destination.mkdir(exist_ok=True)
tree = ast.parse((root / 'create_metahuman/__init__.py').read_text(encoding='utf-8'))
metadata = next(ast.literal_eval(node.value) for node in tree.body if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'bl_info' for t in node.targets))
version = '.'.join(map(str, metadata['version']))
output = destination / (PACKAGE_SLUG + '-' + version + '.zip')
for name in ('LICENSE','THIRD_PARTY_NOTICES.md'):
    if (root/name).read_bytes()!=(root/'create_metahuman'/name).read_bytes():
        raise RuntimeError('Project/add-on distribution notices differ: '+name)
with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED) as archive:
    for path in files(root / "create_metahuman"):
        archive.write(path, path.relative_to(root))
report = {
    "file": output.name,
    "sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
    "bytes": output.stat().st_size,
}
(destination / "build.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
print(json.dumps(report, indent=2))

# Package an already compiled editor bridge. Run UAT BuildPlugin first.
plugin = destination / "CreateMetaHumanBridge-UE5.8"
if not args.addon_only and (plugin / "Binaries/Win64/UnrealEditor-CreateMetaHumanBridge.dll").is_file():
    bridge_version = json.loads((plugin / 'CreateMetaHumanBridge.uplugin').read_text(encoding='utf-8'))['VersionName']
    output = destination / (BRIDGE_PACKAGE_NAME + '-UE5.8-' + bridge_version + '.zip')
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(plugin.rglob("*")):
            relative = path.relative_to(plugin)
            if (
                path.is_file()
                and "Intermediate" not in relative.parts
                and path.suffix.lower() != ".pdb"
                and relative.as_posix() not in {'LICENSE','THIRD_PARTY_NOTICES.md'}
            ):
                if relative.as_posix() == 'CreateMetaHumanBridge.uplugin':
                    archive.write(root / 'plugins/CreateMetaHumanBridge/CreateMetaHumanBridge.uplugin', Path('CreateMetaHumanBridge') / relative)
                else:
                    archive.write(path, Path("CreateMetaHumanBridge") / relative)
        for name in ('LICENSE','THIRD_PARTY_NOTICES.md'):
            archive.write(root/name,Path('CreateMetaHumanBridge')/name)
    report = {
        "file": output.name,
        "sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
        "bytes": output.stat().st_size,
    }
    (destination / "bridge-build.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, indent=2))
