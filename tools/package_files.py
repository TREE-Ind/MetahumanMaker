"""Explicit add-on distribution inventory, shared by build and verification."""
from pathlib import Path
import hashlib,json

PROJECT_NAME = 'MetaHuman Maker'
PACKAGE_SLUG = 'metahuman_maker'
BRIDGE_PACKAGE_NAME = 'MetaHumanMakerBridge'

NATIVE_FILES = ('cm_groom_alembic.exe', 'groom_alembic_writer.cpp',
                'LICENSE-Alembic.txt', 'LICENSE-Imath.txt', 'README.md')


def files(root,*,native_binary=True):
    result = sorted(Path(root).glob('*.py'))
    for name in ('LICENSE','THIRD_PARTY_NOTICES.md'):
        path=Path(root)/name
        if not path.is_file():raise RuntimeError('Required distribution notice missing: '+str(path))
        result.append(path)
    for name in NATIVE_FILES:
        if name=='cm_groom_alembic.exe' and not native_binary:continue
        path = Path(root) / 'native' / name
        if not path.is_file():
            raise RuntimeError('Required native distribution file missing: ' + str(path))
        result.append(path)
    vendor=Path(root)/'web_vendor'
    manifest=vendor/'manifest.json'
    data=json.loads(manifest.read_bytes())
    result.append(manifest)
    for name,record in data['files'].items():
        path=(vendor/'three'/name).resolve()
        if not path.is_relative_to(vendor.resolve()) or not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest()!=record['sha256']:raise RuntimeError('Missing or changed viewer distribution file: '+name)
        result.append(path)
    return result
