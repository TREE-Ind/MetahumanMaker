"""Local photo face mask and visible color transfer, without Blender or network."""

import json
import sys
import time
from pathlib import Path

import numpy as np
from photo_color_core import project
from worker_lifetime import watch_parent


def linear(values):
    return np.where(values <= .04045, values / 12.92, ((values + .055) / 1.055) ** 2.4)


def tone(pixels, gain, chroma):
    rgb = pixels[..., :3]
    normalized = rgb / np.maximum(rgb.sum(2, keepdims=True), 1e-8)
    distance = np.linalg.norm(normalized - chroma, axis=2)
    amount = 1 - np.clip((distance - .05) / .13, 0, 1)
    amount = amount * amount * (3 - 2 * amount)
    result = pixels.copy()
    result[..., :3] *= 1 + (gain - 1) * amount[..., None]
    return result


def save(pixels, path):
    from PIL import Image
    rgb = pixels[..., :3]
    pixels = pixels.copy()
    pixels[..., :3] = np.where(rgb <= .0031308, rgb * 12.92, 1.055 * np.maximum(rgb, 0) ** (1 / 2.4) - .055)
    Image.fromarray(np.rint(np.clip(pixels[::-1], 0, 1) * 255).astype(np.uint8)).save(path)


def run(request, folder):
    import mediapipe as mp
    from mediapipe.tasks import python
    from mediapipe.tasks.python import vision
    from PIL import Image, ImageDraw, ImageFilter

    start = time.perf_counter()
    image = Image.open(folder / 'photo.png').convert('RGB')
    width, height = image.size
    options = vision.FaceLandmarkerOptions(
        base_options=python.BaseOptions(model_asset_path=request['detector']), num_faces=2,
        min_face_detection_confidence=.6, min_face_presence_confidence=.6)
    with vision.FaceLandmarker.create_from_options(options) as detector:
        faces = detector.detect(mp.Image.create_from_file(str(folder / 'photo.png'))).face_landmarks
    if len(faces) != 1:
        raise ValueError('Photo color needs exactly one clearly visible face')
    # Ordered MediaPipe face oval. Keep hair, ears, clothing and background out.
    oval = (10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365,
            379, 378, 400, 377, 152, 148, 176, 149, 150, 136, 172, 58, 132, 93,
            234, 127, 162, 21, 54, 103, 67, 109)
    mask = Image.new('L', image.size)
    ImageDraw.Draw(mask).polygon([(faces[0][i].x * width, faces[0][i].y * height) for i in oval], fill=255)
    radius = max(2, round(min(width, height) * .008))
    mask = mask.filter(ImageFilter.MinFilter(2 * radius + 1)).filter(ImageFilter.GaussianBlur(radius))
    fallback = Image.open(folder / 'fallback.png').convert('RGBA').resize(
        (request['resolution'], request['resolution']), Image.Resampling.BILINEAR)
    base = np.asarray(fallback, dtype=np.float32)[::-1].copy() / 255
    base[..., :3] = linear(base[..., :3])
    photo = linear(np.asarray(image, dtype=np.float32)[::-1] / 255)
    with np.load(folder / 'surface.npz', allow_pickle=False) as surface:
        output, influence = project(surface['points'], surface['triangles'], surface['uv'],
                                    photo, np.asarray(mask, dtype=np.float32)[::-1] / 255, base,
                                    request['camera_scale'], request['strength'])
    gain = np.ones(3)
    neck_report = {'status': 'disabled'}
    if request['match_skin']:
        valid = influence > .95 * request['strength']
        luminance = output[..., :3] @ [.2126, .7152, .0722]
        if np.count_nonzero(valid) < 100:
            raise ValueError('Insufficient fully visible face pixels for tone matching; disable Match Skin Tone')
        low, high = np.percentile(luminance[valid], [25, 75])
        valid &= (luminance >= low) & (luminance <= high)
        old = np.median(base[..., :3][valid], axis=0)
        new = np.median(output[..., :3][valid], axis=0)
        gain = np.clip(new / np.maximum(old, .005), .1, 32)
        chroma = old / max(old.sum(), 1e-8)
        matched = tone(base, gain, chroma)
        output[..., :3] += (matched[..., :3] - base[..., :3]) * (1 - influence[..., None])
        if request['body']:
            body = Image.open(folder / 'body.png').convert('RGBA')
            body.thumbnail((2048, 2048), Image.Resampling.BILINEAR)
            body = np.asarray(body, dtype=np.float32)[::-1].copy() / 255
            body[..., :3] = linear(body[..., :3])
            body = tone(body, gain, chroma)
            if request.get('blend_neck'):
                from neck_color_core import blend
                with np.load(folder / 'neck.npz', allow_pickle=False) as neck:
                    with np.load(folder / 'surface.npz', allow_pickle=False) as surface:
                        output, neck_report = blend(output, body, neck['points'], surface['triangles'], surface['uv'],
                            neck['seam'], neck['head_uv'], neck['head_ids'], neck['body_uv'], neck['body_ids'], protection=influence)
            save(body, folder / 'body-color.png')
    save(output, folder / 'color.png')
    Image.fromarray(np.rint(np.clip(influence[::-1], 0, 1) * 255).astype(np.uint8)).save(folder / 'mask.png')
    return {'status': 'complete', 'seconds': time.perf_counter() - start,
            'resolution': request['resolution'], 'projected_texels': int(np.count_nonzero(influence > .1)),
            'skin_tone_gain': gain.tolist(), 'matched_body': request['body'],
            'neck_blend': neck_report, 'neck_warning': request.get('neck_warning', '')}


if __name__ == '__main__':
    path = Path(sys.argv[1])
    request = json.loads(path.read_text(encoding='utf-8'))
    watch_parent(request['parent'])
    try:
        result = run(request, path.parent)
    except Exception as exc:  # noqa: BLE001 - publish worker failures to its owner.
        result = {'status': 'failed', 'error': str(exc)}
    (path.parent / 'result.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    if result['status'] != 'complete':
        raise SystemExit(1)
