"""Share identical immutable arrays inside one newly owned native capture."""
import hashlib,json,os,uuid
from pathlib import Path

ARRAY_KEYS=('vertices_file','neutral_file','neutral_normals_file','triangles_file','uv_file','dna_mesh_file','dna_vertex_file','skin_linear_file')


class CaptureStorage:
    def __init__(self,folder):
        self.folder=Path(folder).resolve();self.known={};self.files=0;self.linked=0;self.linked_bytes=0;self.fallbacks=0

    def add(self,metadata):
        metadata=Path(metadata)
        if metadata.resolve().parent!=self.folder:raise ValueError('Capture metadata must remain inside its owned folder')
        values=json.loads(metadata.read_text())
        for key in ARRAY_KEYS:
            name=values.get(key)
            if name is None:continue
            if not isinstance(name,str) or Path(name).name!=name:raise ValueError('Capture arrays must remain beside their metadata')
            path=self.folder/name
            if path.is_symlink() or path.resolve().parent!=self.folder or not path.is_file():raise ValueError('Capture sharing requires a regular owned array')
            size=path.stat().st_size;fingerprint=(key,size,hashlib.sha256(path.read_bytes()).hexdigest());self.files+=1
            first=self.known.get(fingerprint)
            if first is None or not first.is_file():self.known[fingerprint]=path;continue
            if first==path or os.path.samefile(first,path):continue
            temporary=self.folder/('.capture-link-'+uuid.uuid4().hex)
            try:
                # Replace only the new array's directory entry after linking a
                # proven byte-identical array. Unsupported filesystems retain
                # the original copy; no user asset or external path is touched.
                os.link(first,temporary);os.replace(temporary,path)
                self.linked+=1;self.linked_bytes+=size
            except OSError:self.fallbacks+=1
            finally:
                if temporary.exists():temporary.unlink()

    def report(self):
        return dict(method='IDENTICAL_ARRAY_HARDLINKS_WITH_COPY_FALLBACK',arrays_seen=self.files,linked_arrays=self.linked,linked_payload_bytes=self.linked_bytes,copy_fallbacks=self.fallbacks,paths_and_bytes_preserved=True)
