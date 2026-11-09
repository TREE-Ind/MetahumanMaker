"""Pinned optional model setup. Default: size/disk plan, no downloads or writes."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys
import urllib.request
import uuid
import zipfile

GIB=1024**3
PROFILES=('marigold','texture-editor','pisa-sr')


def contained(root,name):
    path=(root/name).resolve()
    if Path(name).is_absolute() or not path.is_relative_to(root) or path==root:
        raise ValueError('Invalid optional asset path')
    return path


def verified(path,record):
    if not path.is_file() or path.stat().st_size!=record['bytes']:return False
    with path.open('rb') as stream:
        if record.get('sha256'):return hashlib.file_digest(stream,'sha256').hexdigest()==record['sha256']
        data=stream.read()
    return hashlib.sha1(b'blob '+str(len(data)).encode()+b'\0'+data).hexdigest()==record['git_sha1']


def plan(root,profile):
    spec=json.loads((root/'release/models'/(profile+'.json')).read_text(encoding='utf-8'))
    if spec.get('schema')!=1:raise ValueError('Unsupported optional model manifest')
    destination=contained(root,spec['destination'])
    seen=set();missing=resident=0
    for record in spec['assets']:
        path=contained(destination,record['file'])
        if record['file'] in seen or record['bytes']<=0:raise ValueError('Duplicate or invalid optional asset')
        seen.add(record['file'])
        if not record['source'].startswith('https://'):raise ValueError('Optional assets require HTTPS')
        sha=record.get('sha256');git=record.get('git_sha1')
        if not (sha and len(sha)==64 or git and len(git)==40):raise ValueError('Missing pinned asset identity')
        if path.exists():resident+=record['bytes']
        else:missing+=record['bytes']
    volume=destination
    while not volume.exists():volume=volume.parent
    free=shutil.disk_usage(volume).free
    reserve=int(spec['reserve_gib']*GIB)
    extraction=512*1024**2 if spec.get('extract_runtime') else 0
    report=dict(profile=profile,destination=spec['destination'],asset_count=len(seen),
        missing_bytes=missing,existing_bytes_to_verify=resident,reserve_bytes=reserve,
        extraction_allowance_bytes=extraction,free_bytes=free,
        sufficient_disk=free>=missing+reserve+extraction,downloads_started=False)
    return spec,destination,report


def provision(spec,destination,report):
    if not report['sufficient_disk']:raise ValueError('Not enough disk space for the missing assets plus the retained reserve; no download started')
    # Verify existing files before creating any manifest or replacing anything.
    for record in spec['assets']:
        path=contained(destination,record['file'])
        if path.exists() and not verified(path,record):
            raise ValueError('Existing optional asset differs from the pin; it was retained: '+record['file'])
    destination.mkdir(parents=True,exist_ok=True)
    for record in spec['assets']:
        path=contained(destination,record['file'])
        if path.exists():continue
        path.parent.mkdir(parents=True,exist_ok=True)
        pending=path.with_name(path.name+'.cm-download-'+uuid.uuid4().hex)
        try:
            request=urllib.request.Request(record['source'],headers={'User-Agent':'CreateMetaHuman-model-setup'})
            count=0
            with urllib.request.urlopen(request,timeout=90) as source,pending.open('xb') as output:
                while chunk:=source.read(4*1024**2):
                    count+=len(chunk)
                    if count>record['bytes']:raise ValueError('Optional asset exceeded its pinned size')
                    if shutil.disk_usage(destination).free<len(chunk)+report['reserve_bytes']:
                        raise ValueError('Stopping optional download to retain disk reserve')
                    output.write(chunk)
            if not verified(pending,record):raise ValueError('Optional asset failed pinned verification: '+record['file'])
            if path.exists():raise ValueError('Optional asset appeared during download; existing data retained')
            # Hard-link publication is exclusive and stays on this volume.
            path.hardlink_to(pending)
            print('VERIFIED '+record['file'],flush=True)
        finally:pending.unlink(missing_ok=True)
    runtime=spec.get('extract_runtime')
    if runtime:
        source=contained(destination,runtime);target=destination/'runtime'
        # Keep the small verified archive for recovery/reverification. Extract
        # only within the owned runtime folder and bound the expanded size.
        with zipfile.ZipFile(source) as archive:
            if sum(item.file_size for item in archive.infolist())>report['extraction_allowance_bytes']:
                raise ValueError('Runtime archive exceeds its extraction allowance')
            for item in archive.infolist():
                path=contained(target,item.filename)
                if (item.external_attr>>16)&0o170000==0o120000:raise ValueError('Runtime archive contains a symbolic link')
                if item.is_dir():continue
                body=archive.read(item)
                if path.exists():
                    if path.read_bytes()!=body:raise ValueError('Existing runtime file differs; it was retained: '+item.filename)
                    continue
                path.parent.mkdir(parents=True,exist_ok=True)
                with path.open('xb') as output:output.write(body)
    manifest=dict(spec['manifest'],status='verified-local-not-benchmarked')
    path=destination/'provision.json';temporary=destination/('provision-'+uuid.uuid4().hex+'.tmp')
    try:
        temporary.write_text(json.dumps(manifest,indent=2)+'\n',encoding='utf-8');temporary.replace(path)
    finally:temporary.unlink(missing_ok=True)
    return dict(report,downloads_started=True,verified=True)


def main(default=None):
    parser=argparse.ArgumentParser(description=__doc__)
    if default is None:parser.add_argument('profile',choices=PROFILES)
    parser.add_argument('--download',action='store_true',help='Download/verify pinned assets after the disk check')
    args=parser.parse_args();profile=default or args.profile
    root=Path(__file__).resolve().parents[1]
    spec,destination,report=plan(root,profile)
    if args.download:report=provision(spec,destination,report)
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
