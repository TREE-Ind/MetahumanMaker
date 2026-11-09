"""Optional, offline Marigold lighting estimates; never writes character maps.

Run in the separately provisioned Torch environment, not Blender's Python.
Outputs are linear estimates, not measured reflectance or geometric detail.
"""
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'create_metahuman'))

REVISION = '08c3930bb641abf786ba44ce92547507ebefbc16'
REPOSITORY = 'prs-eth/marigold-iid-lighting-v1-1'


def verify(folder):
    manifest = json.loads((folder / 'provision.json').read_text(encoding='utf-8'))
    if manifest.get('revision') != REVISION or manifest.get('repo') != REPOSITORY:
        raise ValueError('Material model does not match the pinned lighting checkpoint')
    for record in manifest['files']:
        relative = Path(record['path'])
        path = (folder / relative).resolve()
        if relative.is_absolute() or not path.is_relative_to(folder):
            raise ValueError('Invalid model file path')
        if record.get('lfs'):
            with path.open('rb') as stream:
                actual = hashlib.file_digest(stream, 'sha256').hexdigest()
            expected = record['lfs']['oid']
        else:
            data = path.read_bytes()
            actual = hashlib.sha1(b'blob ' + str(len(data)).encode() + b'\0' + data).hexdigest()
            expected = record['oid']
        if actual != expected:
            raise ValueError('Material model file changed: ' + str(relative))


def run(request_path):
    request_path = Path(request_path).resolve()
    request = json.loads(request_path.read_text(encoding='utf-8'))
    folder = request_path.parent
    if shutil.disk_usage(folder).free < 8 * 1024**3 + 128 * 1024**2:
        raise RuntimeError('Retain at least 8 GiB free before material estimation')
    if request.get('parent_pid'):
        from worker_lifetime import watch_parent
        watch_parent(request['parent_pid'], request.get('cancel'))
    model = Path(request['model']).resolve()
    verify(model)
    # This worker must not resolve missing weights or populate another hub cache.
    os.environ['HF_HUB_OFFLINE'] = '1'
    os.environ['TRANSFORMERS_OFFLINE'] = '1'
    os.environ['HF_HUB_DISABLE_TELEMETRY'] = '1'
    import numpy as np
    from PIL import Image
    import torch
    from diffusers import MarigoldIntrinsicsPipeline
    from texture_restore_core import encoded

    torch.set_num_threads(4)
    if not torch.cuda.is_available():
        raise RuntimeError('This optional material worker requires CUDA')
    report = {'schema': 1, 'model': REPOSITORY, 'revision': REVISION,
              'license': 'CreativeML Open RAIL++-M', 'status': 'loading',
              'output_space': 'linear', 'physical_ground_truth': False,
              'applied_to_character': False, 'cases': []}
    report_path = folder / 'material-report.json'

    def save_report():
        report_path.write_text(json.dumps(report, indent=2), encoding='utf-8')

    save_report()
    started = time.monotonic()
    try:
        pipe = MarigoldIntrinsicsPipeline.from_pretrained(
            str(model), variant='fp16', torch_dtype=torch.float16,
            use_safetensors=True, local_files_only=True)
        names = list(pipe.target_properties['target_names'])
        if names != ['albedo', 'shading', 'residual']:
            raise ValueError('Unexpected material modalities')
        pipe.enable_model_cpu_offload(gpu_id=0)
        pipe.vae.enable_tiling()
        report['load_seconds'] = time.monotonic() - started
        report['status'] = 'running'
        save_report()
        for index, entry in enumerate(request['images']):
            source = Path(entry).resolve()
            with Image.open(source) as image:
                if min(image.size) < 64 or max(image.size) > 1024:
                    raise ValueError('Material input must be a 64..1024 pixel surface patch')
                image = image.convert('RGB')
                began = time.monotonic()
                result = pipe(image, num_inference_steps=4, ensemble_size=1,
                              processing_resolution=768, match_input_resolution=True,
                              batch_size=1, generator=torch.Generator(device='cpu').manual_seed(42))
            values = result.prediction
            if values.shape != (3, image.height, image.width, 3) or not np.isfinite(values).all():
                raise ValueError('Invalid intrinsic output dimensions or nonfinite values')
            target = folder / ('material-' + str(index))
            target.mkdir(exist_ok=True)
            np.savez_compressed(target / 'linear.npz', **dict(zip(names, values)))
            for name, value in zip(names, values):
                rgb = np.rint(np.clip(encoded(value), 0, 1) * 255).astype('uint8')
                Image.fromarray(rgb).save(target / (name + '-srgb-preview.png'))
            with source.open('rb') as stream:
                source_sha = hashlib.file_digest(stream, 'sha256').hexdigest()
            report['cases'].append({'source': str(source), 'source_sha256': source_sha,
                                    'output': target.name, 'seconds': time.monotonic() - began,
                                    'shape': list(values.shape)})
            save_report()
            print('MATERIAL_ESTIMATE', index, report['cases'][-1]['seconds'], flush=True)
        report.update(status='ready', seconds=time.monotonic() - started,
                      peak_cuda_bytes=torch.cuda.max_memory_allocated())
        save_report()
    except BaseException as exc:
        report.update(status='failed', error=str(exc))
        save_report()
        raise


if __name__ == '__main__':
    run(sys.argv[1])
