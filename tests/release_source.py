"""Offline public-source boundary and packaging checks; standard library only."""
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
import unittest
import zipfile
import posixpath
from urllib.parse import unquote

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'tools'))
import source_release as release


class SourceRelease(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.selected,cls.report=release.inventory(ROOT)

    def test_public_inventory_and_links(self):
        banned={'.blend','.blend1','.dna','.fbx','.glb','.exe','.dll','.pdb','.pch','.h5','.safetensors','.pt'}
        for name,path in self.selected.items():
            self.assertNotIn(Path(name).suffix,banned,name)
            self.assertNotIn(Path(name).parts[0],{'build','dist','test-output','demo','models','.venv'},name)
            self.assertFalse(any(part.startswith('.env') for part in Path(name).parts),name)
            if name.endswith('.md'):
                text=path.read_text(encoding='utf-8')
                links=re.findall(r'\[[^\]]+\]\(([^)]+)\)',text)+re.findall(r'(?:src|href)="([^"]+)"',text)
                for link in links:
                    if '://' in link or link.startswith('#'):continue
                    target=posixpath.normpath(posixpath.join(Path(name).parent.as_posix(),unquote(link.split('#',1)[0])))
                    self.assertIn(target,self.selected,(name,link))

    def test_documentation_media_retains_binary_bytes(self):
        for name,path in self.selected.items():
            if Path(name).suffix in release.MEDIA_TYPES:
                self.assertEqual(release.public_bytes(path),path.read_bytes(),name)
                release.check_media(name,path.read_bytes())
        with self.assertRaises(ValueError):release.check_media('docs/media/example.jpg',b'not an image')

    def test_sensitive_text_rejected_without_value_disclosure(self):
        probes=[
            'github_'+'pat_'+'x'*40,
            'gh'+'p_'+'a'*36,
            '-----BEGIN '+'PRIVATE KEY-----',
            'AK'+'IA'+'A'*16,
            'sk-'+'proj-'+'x'*32,
        ]
        for secret in probes:
            with self.assertRaises(ValueError) as caught:release.check_text('sample.txt',secret.encode())
            self.assertNotIn(secret,str(caught.exception))
        with self.assertRaises(ValueError):release.check_text('sample.txt',str(ROOT).encode(),[ROOT])

    def test_portable_line_endings(self):
        with tempfile.TemporaryDirectory(prefix='cm-public-lines-') as temporary:
            path=Path(temporary)/'example.py';path.write_bytes(b'x=1\r\n')
            self.assertEqual(release.public_bytes(path),b'x=1\n')
            vendor=Path(temporary)/'web_vendor'/'example.js';vendor.parent.mkdir();vendor.write_bytes(b'x=1\r\n')
            self.assertEqual(release.public_bytes(vendor),b'x=1\r\n')

    def test_stage_archive_and_reinventory(self):
        with tempfile.TemporaryDirectory(prefix='cm-public-source-') as temporary:
            staged=Path(temporary)/'source'
            release.stage(staged,self.selected,self.report)
            selected,report=release.inventory(staged)
            self.assertEqual(self.report['files'],report['files'])
            with self.assertRaises(ValueError):release.stage(staged,self.selected,self.report)
            artifact=release.archive(staged,selected,report)
            with zipfile.ZipFile(staged/'dist'/artifact['file']) as archive:
                self.assertEqual(set(archive.namelist()),set(selected)|{'SOURCE_MANIFEST.json'})
                self.assertEqual(json.loads(archive.read('SOURCE_MANIFEST.json'))['files'],report['files'])
            with self.assertRaises(FileExistsError):release.archive(staged,selected,report)
            if shutil.which('git'):
                def git(*args):
                    return subprocess.run(['git','-C',str(staged),*args],check=True,capture_output=True).stdout
                git('init');git('config','core.autocrlf','false');git('add','--all')
                tracked=set(git('ls-files','-z').decode().strip('\0').split('\0'))
                self.assertEqual(tracked,set(selected)|{'SOURCE_MANIFEST.json'})
                tree=git('write-tree').decode().strip()
                import io
                with zipfile.ZipFile(io.BytesIO(git('archive','--format=zip',tree))) as snapshot:
                    for name,record in report['files'].items():
                        self.assertEqual(release.digest(snapshot.read(name)),record['sha256'],name)


if __name__=='__main__':unittest.main(verbosity=2)
