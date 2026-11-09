"""Single-frame groom export with Unreal coordinates and typed Alembic attributes."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import struct
import subprocess
import sys
from tempfile import TemporaryDirectory

import numpy as np


def export(context, objects, destination):
    from . import grooming, resource_budget
    helper = Path(__file__).with_name('native') / 'cm_groom_alembic.exe'
    if sys.platform != 'win32' or not helper.is_file():
        raise ValueError('Unreal Alembic export needs the packaged Windows writer; reinstall the complete add-on')
    payload = grooming.export_curves(context, objects)
    curves = payload['curves']
    counts = np.asarray([len(curve['points']) for curve in curves], dtype='<i4')
    total = int(counts.sum())
    if not len(curves) or np.any(counts < 2) or len(curves) > 10000000 or total > 100000000:
        raise ValueError('The groom exceeds the Alembic writer limits or contains an empty strand')
    resource_budget.require('groom Alembic export', total * 64 + len(curves) * 64)
    destination = Path(destination).resolve()
    if destination.suffix.lower() != '.abc':
        raise ValueError('Choose an .abc groom filename')
    resource_budget.require_disk('groom Alembic export', destination.parent,
                                 resource_budget.GIB + total * 128)
    destination.parent.mkdir(parents=True, exist_ok=True)
    names = [destination.name, destination.name + '.json']
    colors = any('colors' in curve for curve in curves)
    report = {'schema_version': 1, 'coordinates': 'Unreal centimeters', 'axis': 'Z up',
              'curves': len(curves), 'points': total,
              'guides': sum(int(curve['guide']) for curve in curves),
              'groups': payload.get('group_settings', []),
              'attributes': ['groom_group_id', 'groom_guide', 'groom_root_uv', 'groom_width']
                            + (['groom_color'] if colors else []),
              'import': 'Enable Alembic Groom Importer; import with rotation 0 and scale 1.'}
    with TemporaryDirectory(prefix='.groom-alembic-', dir=destination.parent) as directory:
        folder = Path(directory)
        buffer = folder / 'input.bin'
        with buffer.open('wb') as stream:
            stream.write(b'CMGABC01' + struct.pack('<III', len(curves), total, int(colors)))
            stream.write(counts.tobytes())
            for key, shape in [('points', 3), ('widths', 1)]:
                for curve in curves:
                    values = np.asarray(curve[key], dtype='<f4')
                    if values.size != len(curve['points']) * shape or not np.isfinite(values).all():
                        raise ValueError('Invalid Alembic ' + key)
                    stream.write(values.tobytes())
            for key in ('group', 'guide'):
                stream.write(np.asarray([curve[key] for curve in curves], dtype='<i4').tobytes())
            stream.write(np.asarray([curve['root_uv'] for curve in curves], dtype='<f4').tobytes())
            if colors:
                for curve in curves:
                    values = np.asarray(curve.get('colors', [[1, 1, 1]] * len(curve['points'])), dtype='<f4')
                    if values.shape != (len(curve['points']), 3) or not np.isfinite(values).all():
                        raise ValueError('Invalid Alembic strand colors')
                    stream.write(values.tobytes())
        staged = folder / destination.name
        result = subprocess.run([str(helper), str(buffer), str(staged)], capture_output=True,
                                text=True, timeout=180,
                                creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
        if result.returncode or not staged.is_file() or staged.stat().st_size < 64:
            raise ValueError('Groom Alembic writer failed: ' + (result.stderr or result.stdout).strip())
        report.update(sha256=hashlib.sha256(staged.read_bytes()).hexdigest(), bytes=staged.stat().st_size)
        (folder / names[1]).write_text(json.dumps(report, indent=2), encoding='utf-8')
        # Stage both files before replacing either; retain prior exports on failure.
        existed, changed = set(), []
        for index, name in enumerate(names):
            if (destination.parent / name).exists():
                shutil.copy2(destination.parent / name, folder / ('old-' + str(index)))
                existed.add(index)
        try:
            for index, name in enumerate(names):
                os.replace(folder / name, destination.parent / name)
                changed.append(index)
        except BaseException:
            for index in reversed(changed):
                if index in existed:
                    os.replace(folder / ('old-' + str(index)), destination.parent / names[index])
                else:
                    (destination.parent / names[index]).unlink(missing_ok=True)
            raise
    return report
