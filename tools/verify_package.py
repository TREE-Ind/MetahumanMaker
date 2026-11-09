"""Check packaged Python/native files against source and installed folders."""

import hashlib
import json
import sys
import zipfile
from package_files import files
from pathlib import Path


def main():
    root = Path(__file__).resolve().parents[1]
    archive_path = Path(sys.argv[1]).resolve()
    installations = [Path(arg).resolve() for arg in sys.argv[2:]]
    source = {
        path.relative_to(root / 'create_metahuman').as_posix(): path.read_bytes()
        for path in files(root / "create_metahuman")
    }
    with zipfile.ZipFile(archive_path) as archive:
        names = {name.removeprefix('create_metahuman/') for name in archive.namelist()}
        if names != set(source):
            raise RuntimeError("Package file list differs from source")
        for name, data in source.items():
            if archive.read("create_metahuman/" + name) != data:
                raise RuntimeError("Package differs from source: " + name)
    for folder in installations:
        if {path.relative_to(folder).as_posix() for path in files(folder)} != set(source):
            raise RuntimeError("Installed file list differs: " + str(folder))
        for name, data in source.items():
            if (folder / name).read_bytes() != data:
                raise RuntimeError("Installed file differs: " + str(folder / name))
    report = {
        "archive": str(archive_path),
        "sha256": hashlib.sha256(archive_path.read_bytes()).hexdigest(),
        "python_files": sum(name.endswith('.py') for name in source),
        "native_files": sum(name.startswith('native/') for name in source),
        "viewer_files": sum(name.startswith('web_vendor/') for name in source),
        "matching_installations": [str(path) for path in installations],
    }
    (archive_path.parent / "package-verification.json").write_text(
        json.dumps(report, indent=2)
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
