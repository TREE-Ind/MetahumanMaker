"""Cheap editor preflight; full model-byte verification belongs in the worker."""
import hashlib
import json
from pathlib import Path
import re


def expected_editor(backend):
    backend=Path(backend).resolve()
    manifest=json.loads((backend/'provision.json').read_text(encoding='utf-8'))
    if manifest.get('status')!='verified-local-not-benchmarked':
        raise ValueError('Finish provisioning the optional texture editor first')
    assets={record['name']:record for record in manifest['assets']}
    result={}
    for name in ('flux-2-klein-4b-Q4_0.gguf','Qwen3-4B-Q4_K_M.gguf',
                 'full_encoder_small_decoder.safetensors'):
        record=assets.get(name,{})
        sha=record.get('sha256','')
        path=(backend/name).resolve()
        if not path.is_relative_to(backend) or not re.fullmatch('[0-9a-f]{64}',sha):
            raise ValueError('Invalid texture editor manifest: '+name)
        if not path.is_file() or path.stat().st_size!=record.get('size'):
            raise ValueError('Texture editor weight is missing or incomplete: '+name)
        # These are expected hashes, not evidence that current bytes passed.
        # The cancellable worker compares every actual byte before inference.
        result[name]=sha
    runtime=backend/'runtime/sd-cli.exe'
    if not runtime.resolve().is_relative_to(backend) or not runtime.is_file():
        raise ValueError('The local texture editor runtime is missing')
    with runtime.open('rb') as stream:
        result['runtime/sd-cli.exe']=hashlib.file_digest(stream,'sha256').hexdigest()
    return result
