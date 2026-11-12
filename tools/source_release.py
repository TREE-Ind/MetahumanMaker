"""Audit/build an explicit source archive without private workspace artifacts."""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import re
import stat
import zipfile

from package_files import files, PACKAGE_SLUG, PROJECT_NAME


TOOLS = (
    'build.py', 'build_groom_alembic.py', 'package_files.py', 'verify_package.py',
    'source_release.py', 'setup_surface_solver.py', 'setup_models.py',
    'provision_optional.py',
    'convert_gnm_decoder.py', 'provision_material_decoder.py',
    'provision_material_sr.py', 'provision_texture_editor.py',
    'material_decomposition.py', 'material_superresolution.py',
)
EXCLUDED = (
    'build/', 'dist/', 'test-output/', 'demo/', 'research/', 'models/',
    '.venv*/', '.env*', 'web/', 'assets/', 'renders/', 'tests/ue_project/',
    'plugins/**/Binaries/', 'plugins/**/Intermediate/',
    'create_metahuman/native/cm_groom_alembic.exe',
)
PUBLIC_FILES = (
    'CONTRIBUTING.md', 'CODE_OF_CONDUCT.md', 'SECURITY.md', '.gitattributes', '.editorconfig',
    '.github/workflows/source-checks.yml',
    '.github/ISSUE_TEMPLATE/bug_report.yml', '.github/ISSUE_TEMPLATE/feature_request.yml',
    '.github/ISSUE_TEMPLATE/config.yml', '.github/pull_request_template.md',
    'docs/quickstart.md', 'docs/release-notes.md', 'tests/release_source.py',
    'docs/installation.md', 'docs/workflows.md', 'docs/agents.md', 'docs/troubleshooting.md',
    'docs/known-issues.md', 'docs/development.md', 'docs/releasing.md', 'docs/launch.md', 'docs/media/README.md',
)
MEDIA_FILES = ('hero.jpg','blender-workflow.jpg','agent-workflow.gif',
               'skin-reconstruction.jpg','skin-maps.jpg','genie-expressions.jpg',
               'unreal-editable.jpg','tree-mark.svg')
MEDIA_TYPES={'.jpg':b'\xff\xd8\xff','.png':b'\x89PNG\r\n\x1a\n','.gif':b'GIF8'}


def check_media(name,body):
    signature=MEDIA_TYPES.get(Path(name).suffix.lower())
    if not signature or not body.startswith(signature) or len(body)>4*1024**2:
        raise ValueError('Invalid or oversized documentation media: '+name)


def check_text(name, body, private_roots=()):
    """Reject private paths and selected credential forms without echoing values."""
    text=body.decode('utf-8')
    normalized=text.replace('\\\\','/').replace('\\','/').casefold()
    if any(str(value).replace('\\','/').casefold() in normalized for value in private_roots):
        raise ValueError('An included file contains a local absolute path: '+name)
    patterns=(
        r'-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----',
        r'\bgithub_pat_[A-Za-z0-9_]{20,}\b',
        r'\bgh[pousr]_[A-Za-z0-9]{30,}\b',
        r'\bAKIA[0-9A-Z]{16}\b',
        r'\bsk-proj-[A-Za-z0-9_-]{24,}\b',
    )
    if any(re.search(pattern,text) for pattern in patterns):
        raise ValueError('An included file contains a possible credential: '+name)
    if name.endswith('.py'):ast.parse(body,filename=name)


def digest(data):
    return hashlib.sha256(data).hexdigest()


def public_bytes(path):
    """Match the public Git checkout's LF text while retaining vendored bytes."""
    path=Path(path);body=path.read_bytes()
    return body if 'web_vendor' in path.parts or path.suffix.lower() in MEDIA_TYPES else body.replace(b'\r\n',b'\n')


def inventory(root):
    root=Path(root).resolve()
    selected={}

    def add(path,name=None):
        path=Path(path);resolved=path.resolve()
        if not resolved.is_relative_to(root) or not path.is_file():
            raise ValueError('Missing or out-of-workspace release source: '+str(path))
        for part in (path,*path.parents):
            if part==root:break
            if part.is_symlink() or getattr(part.lstat(),'st_file_attributes',0)&getattr(stat,'FILE_ATTRIBUTE_REPARSE_POINT',0x400):
                raise ValueError('Release inventory does not follow linked paths: '+str(path))
        name=name or path.relative_to(root).as_posix()
        if name in selected or name.startswith('/') or '..' in Path(name).parts:
            raise ValueError('Duplicate or invalid release path: '+name)
        selected[name]=path

    for path in files(root/'create_metahuman',native_binary=False):add(path)
    for name in ('LICENSE','THIRD_PARTY_NOTICES.md','.gitignore'):add(root/name)
    for name in PUBLIC_FILES:add(root/name)
    for name in MEDIA_FILES:add(root/'docs/media'/name)
    public_readme=root/'release/README.md'
    add(public_readme if public_readme.is_file() else root/'README.md','README.md')
    add(root/'tests/native_assembly_memory.py')
    add(root/'tests/photo_multiview.py')
    add(root/'tests/photo_eyes.py')
    add(root/'tests/photo_visibility.py')
    add(root/'tests/groom_group_mapping.py')
    add(root/'tests/groom_lod_warp.py')
    for name in ('marigold','texture-editor','pisa-sr'):add(root/'release/models'/(name+'.json'))
    for name in TOOLS:add(root/'tools'/name)
    plugin=root/'plugins/CreateMetaHumanBridge'
    add(plugin/'CreateMetaHumanBridge.uplugin')
    source=plugin/'Source'
    for path in sorted(source.rglob('*')):
        if path.is_file() and path.suffix in {'.cpp','.h','.cs'}:add(path)
    if not any(name.endswith('.Build.cs') for name in selected):
        raise ValueError('The bridge source inventory is incomplete')
    license_text=(root/'LICENSE').read_bytes()
    if not license_text.decode('utf-8').replace('\r\n','\n').startswith('MIT License\n'):
        raise ValueError('The approved MIT project license is missing')
    for name in ('LICENSE','THIRD_PARTY_NOTICES.md'):
        if (root/name).read_bytes()!=(root/'create_metahuman'/name).read_bytes():
            raise ValueError('Project/add-on notice copies differ: '+name)
    # Check included text only. Never open .env.local or other excluded paths.
    private_roots={str(Path.home()),str(root)}
    records={}
    for name,path in sorted(selected.items()):
        body=public_bytes(path)
        if Path(name).suffix.lower() in MEDIA_TYPES:check_media(name,body)
        else:check_text(name,body,private_roots)
        records[name]={'bytes':len(body),'sha256':digest(body)}
    metadata=ast.parse((root/'create_metahuman/__init__.py').read_text(encoding='utf-8'))
    info=next(ast.literal_eval(node.value) for node in metadata.body
        if isinstance(node,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='bl_info' for t in node.targets))
    assert info['name'] == PROJECT_NAME, 'Public product identity differs from add-on metadata'
    report={'schema':1,'project':PROJECT_NAME,'version':'.'.join(map(str,info['version'])),'license':'MIT',
        'text_normalization':'LF for public text; vendored viewer bytes retained',
        'scope':'Original project code; separately licensed dependencies retain their notices',
        'file_count':len(records),'total_bytes':sum(item['bytes'] for item in records.values()),
        'files':records,'excluded':list(EXCLUDED),
        'limitations':['No claim of complete feature/platform parity or independent legal review',
            'Source build requires separately installed Blender/UE/SDKs and a local native helper build']}
    return selected,report


def stage(destination,selected,report):
    """Write an exact public checkout into a fresh directory; no Git/network side effects."""
    destination=Path(destination).absolute()
    if destination.exists() or destination.is_symlink():
        raise ValueError('Choose a fresh source staging directory; existing paths are retained')
    for parent in destination.parents:
        if parent.exists() and (parent.is_symlink() or getattr(parent.lstat(),'st_file_attributes',0)&getattr(stat,'FILE_ATTRIBUTE_REPARSE_POINT',0x400)):
            raise ValueError('Source staging does not follow linked directories')
    destination.mkdir(parents=True,exist_ok=False)
    for name,path in sorted(selected.items()):
        body=public_bytes(path)
        if digest(body)!=report['files'][name]['sha256']:
            raise ValueError('Source changed during staging: '+name)
        target=destination/name;target.parent.mkdir(parents=True,exist_ok=True)
        with target.open('xb') as output:output.write(body)
    (destination/'SOURCE_MANIFEST.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    actual={path.relative_to(destination).as_posix() for path in destination.rglob('*') if path.is_file()}
    if actual!=set(selected)|{'SOURCE_MANIFEST.json'}:raise ValueError('The staged checkout has an unexpected inventory')
    for name,record in report['files'].items():
        if digest((destination/name).read_bytes())!=record['sha256']:raise ValueError('Staged source differs: '+name)
    return {'directory':str(destination.resolve()),'file_count':len(actual),'verified':True}


def archive(root,selected,report):
    destination=Path(root)/'dist';destination.mkdir(exist_ok=True)
    target=destination/(PACKAGE_SLUG+'-source-'+report['version']+'.zip')
    def entry(archive,name,data):
        info=zipfile.ZipInfo(name,(2026,1,1,0,0,0))
        info.compress_type=zipfile.ZIP_DEFLATED
        info.external_attr=0o100644<<16
        archive.writestr(info,data)
    # Exclusive creation preserves previous review artifacts.
    with zipfile.ZipFile(target,'x') as output:
        for name,path in sorted(selected.items()):
            body=public_bytes(path)
            if digest(body)!=report['files'][name]['sha256']:
                raise ValueError('Source changed during packaging: '+name)
            entry(output,name,body)
        entry(output,'SOURCE_MANIFEST.json',(json.dumps(report,indent=2)+'\n').encode('utf-8'))
    with zipfile.ZipFile(target) as check:
        if set(check.namelist())!=set(selected)|{'SOURCE_MANIFEST.json'}:
            raise ValueError('The source archive has an unexpected inventory')
        for name,record in report['files'].items():
            body=check.read(name)
            if len(body)!=record['bytes'] or digest(body)!=record['sha256']:
                raise ValueError('The source archive differs from its inventory: '+name)
    return {'file':target.name,'bytes':target.stat().st_size,'sha256':digest(target.read_bytes())}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive',action='store_true',help='Create a source ZIP after the inventory audit')
    parser.add_argument('--stage',type=Path,help='Write a fresh public checkout after the inventory audit')
    args=parser.parse_args()
    root=Path(__file__).resolve().parents[1]
    selected,report=inventory(root)
    if args.archive:report['archive']=archive(root,selected,report)
    if args.stage:report['stage']=stage(args.stage,selected,report)
    destination=root/'dist';destination.mkdir(exist_ok=True)
    (destination/'source-release-audit.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({k:v for k,v in report.items() if k!='files'},indent=2))


if __name__=='__main__':main()
