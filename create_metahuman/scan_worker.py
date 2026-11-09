"""Project geometry-only snapshots without opening the artist's textured scene."""

import hashlib
import json
import sys
import time
import traceback
from pathlib import Path
from types import SimpleNamespace

import bpy
import numpy as np


def report(folder, value):
    path = folder / 'report.tmp'
    path.write_text(json.dumps(value, indent=2), encoding='utf-8')
    for attempt in range(51):
        try:
            path.replace(folder / 'report.json')
            return
        except PermissionError:
            if attempt == 50:
                raise
            time.sleep(.01)


def main():
    folder = Path(sys.argv[sys.argv.index('--')+1]).resolve().parent
    request = json.loads((folder / 'request.json').read_text(encoding='utf-8'))
    sys.path.insert(0, request['addon_root'])
    from create_metahuman import mesh_ops as m
    from create_metahuman import scan_projection as scan
    from create_metahuman import workflow
    from create_metahuman.worker_lifetime import watch_parent
    watch_parent(request['parent'])
    try:
        report(folder, {'status': 'running', 'stage': 'Loading scan geometry…'})
        with bpy.data.libraries.load(str(folder / 'input.blend'), link=False) as (_source, destination):
            destination.objects = list(request['objects'].values())
        objects = dict(zip(request['objects'], destination.objects))
        for obj in objects.values():
            bpy.context.scene.collection.objects.link(obj)
        s = SimpleNamespace(**request['settings'])
        s.scan_progress = lambda done, total: report(folder, {'status': 'running',
            'stage': 'Projecting scan detail: '+str(round(100*done/max(total, 1)))+'%',
            'progress': done/max(total, 1)})
        f = SimpleNamespace(**request['features'])
        s.result, s.source = objects['accepted'], objects['source']
        s.detail = objects.get('detail')
        report(folder, {'status': 'running', 'stage': 'Projecting source onto the sculpt surface…'})
        if s.detail:
            obj = s.detail
            obj.shape_key_add(name='Basis')
            obj.shape_key_add(name='Scan Detail').value = s.detail_strength
            obj['cm_scan_applied'] = False
            obj[scan.SIGNATURE] = scan.shape_signature(obj)
        workflow.generate_displacement(bpy.context, s, f)
        obj = s.detail
        if request['regenerate']:
            output = folder / 'output.npz'
            np.savez(output, points=scan.key_points(obj.data.shape_keys.key_blocks['Scan Detail']))
        else:
            output = folder / 'output.blend'
            # The parent binds this to its accepted object after validation.
            # Remove the worker-only ID dependency from the output bundle.
            if 'cm_detail_source' in obj:
                del obj['cm_detail_source']
            bpy.data.libraries.write(str(output), {obj}, fake_user=True, compress=False)
        report(folder, {'status': 'complete', 'file': output.name,
            'sha256': hashlib.sha256(output.read_bytes()).hexdigest(),
            'object': obj.name, 'vertices': len(obj.data.vertices), 'topology': m.topology(obj)})
    except Exception as exc:
        report(folder, {'status': 'failed', 'error': str(exc), 'traceback': traceback.format_exc()})
        raise


if __name__ == '__main__':
    main()
