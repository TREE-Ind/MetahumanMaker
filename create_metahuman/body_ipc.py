"""Nonblocking JSON handoff for the native body worker on Windows."""
import json
from pathlib import Path


def read(path):
    try:return json.loads(Path(path).read_text(encoding='utf-8'))
    except (FileNotFoundError,PermissionError):return None


def write(path,value):
    path=Path(path);temporary=path.with_name(path.name+'.tmp')
    try:
        temporary.write_text(json.dumps(value),encoding='utf-8')
        temporary.replace(path)
    except PermissionError:
        # A reader or Windows scanner can briefly deny replace/open. The
        # caller retries without marking this request as sent or complete.
        return False
    return True
