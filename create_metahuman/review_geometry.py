"""Load a checked review endpoint without substituting an unapplied solve.

Review eligibility is not production acceptance. Native backtracking may
publish a different endpoint than the external numerical candidate.
"""
import json
import re
from pathlib import Path
import numpy as np
from .vertex_contract import fingerprint


def load_review_geometry(folder, prefix, source_vertices, source_triangles):
    if not isinstance(prefix,str) or not re.fullmatch(r'[A-Za-z0-9_-]+',prefix):
        raise ValueError('Review must name a local artifact prefix')
    folder=Path(folder)
    receipt=json.loads((folder/(prefix+'-candidate.json')).read_text())
    audit=json.loads((folder/(prefix+'-audit.json')).read_text())
    with np.load(folder/(prefix+'-candidate.npz'),allow_pickle=False) as candidate:
        planned=np.array(candidate['body'],copy=True)
        initial=np.array(candidate['before'],copy=True)
    if planned.ndim!=2 or planned.shape[1]!=3 or initial.shape!=planned.shape or not np.isfinite(planned).all() or not np.isfinite(initial).all():
        raise ValueError('Review has invalid geometry')
    if not receipt.get('reason') or not receipt.get('prevent_new_contacts') or receipt.get('new_pairs')!=0 or not audit.get('generated_contact_endpoint_gate_passed') or audit.get('new_pairs')!=0 or audit.get('collapsed_float32_faces')!=0:
        raise ValueError('Review has not passed its terminal geometry audit')
    if receipt.get('output_body')!=fingerprint(planned) or receipt.get('input_body')!=fingerprint(initial):
        raise ValueError('Review planned geometry contract changed')
    source=dict(vertices=fingerprint(source_vertices),triangles=fingerprint(source_triangles))
    if audit.get('source_contract')!=source:
        raise ValueError('Review original source changed')
    native_path=folder/(prefix+'-native-review.npz')
    preview_path=folder/(prefix+'-preview.json')
    preview=json.loads(preview_path.read_text()) if preview_path.exists() else {}
    basis='planned'
    body=planned
    if native_path.exists():
        with np.load(native_path,allow_pickle=False) as native:
            body=np.array(native['body'],copy=True)
            before=np.array(native['before'],copy=True)
            fraction=float(native['fraction'])
            planned_hash=str(native['planned_body_sha256'])
        if body.shape!=planned.shape or before.shape!=planned.shape or not np.isfinite(body).all() or not np.isfinite(before).all():
            raise ValueError('Native review has invalid geometry')
        if planned_hash!=fingerprint(planned)['sha256'] or preview.get('planned_output_body')!=fingerprint(planned):
            raise ValueError('Native review does not match its audited solve')
        if preview.get('native_output_body')!=fingerprint(body) or preview.get('native_input_body')!=fingerprint(before):
            raise ValueError('Native review geometry changed or has no bound host receipt')
        if preview.get('candidate')!=prefix or preview.get('new_pairs_vs_immediate_input')!=0 or not preview.get('protected_native_coordinates_exact'):
            raise ValueError('Native review has not passed its host protection gate')
        if preview.get('parts',{}).get('body',{}).get('collapsed_faces')!=0:
            raise ValueError('Native review has no noncollapsed host proof')
        if not np.isfinite(fraction) or not 0<fraction<=1 or preview.get('native_step_fraction')!=fraction:
            raise ValueError('Native review step fraction changed')
        basis='native'
    elif preview.get('native_output_body') is not None:
        raise ValueError('Checked native review geometry is missing')
    return dict(body=body,receipt=receipt,audit=audit,
                contract=dict(prefix=prefix,basis=basis,body=fingerprint(body),planned_body=fingerprint(planned),source=source))
