"""Local photo inference and constrained fitting; no Blender or network calls."""

import json
import sys
import time
from pathlib import Path

import numpy as np
from photo_fit_core import MEDIAPIPE_68, fit
from worker_lifetime import watch_parent


def run(request, folder):
    import mediapipe as mp
    from mediapipe.tasks import python
    from mediapipe.tasks.python import vision
    from PIL import Image, ImageOps

    start = time.perf_counter()
    with Image.open(request['photo']) as original:
        if original.width * original.height > 32_000_000:
            raise ValueError('Use a photo no larger than 32 megapixels')
        image = ImageOps.exif_transpose(original).convert('RGB')
        image.thumbnail((2048, 2048))
        image.save(folder / 'reference.png')
    width, height = image.size
    options = vision.FaceLandmarkerOptions(
        base_options=python.BaseOptions(model_asset_path=str(Path(request['models']) / 'face_landmarker.task')),
        num_faces=2, min_face_detection_confidence=.6, min_face_presence_confidence=.6)
    with vision.FaceLandmarker.create_from_options(options) as detector:
        detected = detector.detect(mp.Image.create_from_file(str(folder / 'reference.png')))
    if len(detected.face_landmarks) != 1:
        raise ValueError('Use a photo containing exactly one clearly visible face')
    pixels = np.asarray([(p.x * width, p.y * height) for p in detected.face_landmarks[0]])
    eye = np.linalg.norm(pixels[33] - pixels[263])
    if eye < 80:
        raise ValueError('Use a closer face photograph with at least 80 pixels between the outer eye corners')
    if np.linalg.norm(pixels[13] - pixels[14]) / eye > .08:
        raise ValueError('Use a neutral, closed-mouth reference; an open mouth or smile would distort identity fitting')
    target = pixels[list(MEDIAPIPE_68)]
    target = (target - [width / 2, height / 2]) * [1, -1] / max(width, height)
    models = Path(request['models'])
    with np.load(models / 'gnm_head.npz', allow_pickle=False) as model:
        result = fit(model['template_vertex_positions'], model['vertex_identity_basis'],
                     np.loadtxt(models / 'head_sparse_68.txt'), target, request.get('initial'))
    result.update(width=width, height=height, seconds=time.perf_counter() - start,
                  reference='reference.png', status='complete',
                  before_rms_pixels=result['before_rms'] * max(width, height),
                  after_rms_pixels=result['after_rms'] * max(width, height))
    return result


if __name__ == '__main__':
    path = Path(sys.argv[1])
    request = json.loads(path.read_text(encoding='utf-8'))
    watch_parent(request['parent'])
    try:
        result = run(request, path.parent)
    except Exception as exc:  # noqa: BLE001 - always emit the owned worker's failure receipt.
        result = {'status': 'failed', 'error': str(exc)}
    (path.parent / 'result.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    if result['status'] != 'complete':
        raise SystemExit(1)
