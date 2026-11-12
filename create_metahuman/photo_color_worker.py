"""Local photo face mask and visible color transfer, without Blender or network."""

import json
import sys
import time
from pathlib import Path

import numpy as np
from photo_color_core import balance_view, complete_skin_tone, feature_alignment, merge_view, project_samples, register_view, texture_gutters, raster_protection, uv_coverage, feather_projection
from photo_fit_core import MEDIAPIPE_68
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


def exclude_eye_apertures(mask, landmarks):
    """Keep photographed eyeballs off skin that moves when the lids close."""
    from PIL import Image, ImageDraw, ImageFilter

    width, height = mask.size
    allowed = Image.new('L', mask.size, 255)
    draw = ImageDraw.Draw(allowed)
    for start in (36, 42):
        indices = MEDIAPIPE_68[start:start + 6]
        draw.polygon([(landmarks[index].x * width, landmarks[index].y * height)
                      for index in indices], fill=0)
    radius = max(2, round(min(width, height) * .004))
    allowed = allowed.filter(ImageFilter.MinFilter(2 * radius + 1)).filter(
        ImageFilter.GaussianBlur(max(1, radius / 2)))
    return Image.fromarray(np.minimum(np.asarray(mask), np.asarray(allowed)))


def exclude_nostril_apertures(image, mask, landmarks):
    """Keep photographed cavity shadows off the outside of the fitted nose.

    Only compact dark components inside the landmark-defined nasal base are
    removed. Native skin supplies the actual aperture; alar creases, freckles
    and the surrounding photographed skin remain available for projection.
    """
    from PIL import Image, ImageDraw, ImageFilter
    from scipy.ndimage import label, binary_dilation, binary_propagation, distance_transform_edt

    width,height=image.size
    points=np.array([(p.x*width,p.y*height) for p in landmarks])[list(MEDIAPIPE_68)]
    base=points[31:36];span=float(np.linalg.norm(base[-1]-base[0]))
    if span<8:return mask,0
    region=Image.new('L',image.size)
    # Tip and lower nose landmarks bound the search, even in oblique views.
    center=base.mean(0)
    across=(base[-1]-base[0])/span
    up=np.array([-across[1],across[0]])
    center=center+(points[30]-center)*.12
    angles=np.linspace(0,2*np.pi,48,endpoint=False)
    polygon=(center+np.cos(angles)[:,None]*across*span*.55
             +np.sin(angles)[:,None]*up*span*.35)
    ImageDraw.Draw(region).polygon([tuple(p) for p in polygon],fill=255)
    area=np.asarray(region)>0
    rgb=np.asarray(image,dtype=np.float32)/255
    light=rgb@[.2126,.7152,.0722]
    if area.sum()<16:return mask,0
    reference=float(np.percentile(light[area],75))
    components,count=label(area & (light<reference*.43))
    holes=np.zeros_like(area)
    for index in range(1,count+1):
        component=components==index;size=int(component.sum())
        if max(2,span*span*.001)<=size<=span*span*.16:holes|=component
    removed=int(holes.sum())
    if not removed:return mask,0
    # Removing only the black core leaves a photographed nostril rim beside
    # the fitted opening. Follow its connected penumbra, bounded by both the
    # nasal-base landmarks and distance to the detected aperture. This keeps
    # alar creases elsewhere on the nose available for transfer.
    near = distance_transform_edt(~holes) <= max(1., span * .07)
    penumbra = area & near & (light < reference * .60)
    holes = binary_propagation(holes, mask=holes | penumbra)
    radius=max(1,round(span*.025))
    holes=binary_dilation(holes,iterations=radius)
    allowed=Image.fromarray(np.where(holes,0,255).astype(np.uint8)).filter(
        ImageFilter.GaussianBlur(max(.7,radius*.45)))
    return Image.fromarray(np.minimum(np.asarray(mask),np.asarray(allowed))),removed


def fine_hair_mask(image, hair, landmarks=None):
    """Exclude thin dark strands connected to confident hair, retaining skin marks."""
    from scipy.ndimage import (grey_closing, binary_dilation, binary_propagation,
                               distance_transform_edt, gaussian_filter, sobel)
    from PIL import Image, ImageDraw

    rgb = np.asarray(image, dtype=np.float32) / 255
    luma = rgb @ np.array([.2126, .7152, .0722], dtype=np.float32)
    width, height = image.size
    radius = max(2, round(min(width, height) * .003))
    background = grey_closing(luma, size=2 * radius + 1)
    contrast = background - luma
    gx, gy = sobel(luma, axis=1) / 8, sobel(luma, axis=0) / 8
    sigma = max(1., radius * .8)
    xx = gaussian_filter(gx * gx, sigma)
    yy = gaussian_filter(gy * gy, sigma)
    xy = gaussian_filter(gx * gy, sigma)
    coherence = np.hypot(xx - yy, 2 * xy) / np.maximum(xx + yy, 1e-12)
    # Weak candidates must form a coherent line. Strong dark strand junctions
    # can still connect it to the hair mass without treating pores as strands.
    strands = ((contrast > np.maximum(.015, background * .045))
               & ((coherence > .4) | (contrast > np.maximum(.035, background * .14))))
    protected = Image.new('L', image.size)
    if landmarks is not None:
        draw = ImageDraw.Draw(protected)
        for start, stop in ((17, 22), (22, 27), (48, 68)):
            points = np.array([(landmarks[i].x * width, landmarks[i].y * height)
                               for i in MEDIAPIPE_68[start:stop]])
            low, high = points.min(0), points.max(0)
            margin = np.array([width * .012, height * .012])
            draw.rectangle(tuple(low - margin) + tuple(high + margin), fill=255)
    protected = np.asarray(protected) > 0
    anchors = (np.asarray(hair) > .5) & ~protected
    if not anchors.any():
        return Image.new('L', image.size, 255), 0
    near = distance_transform_edt(~anchors) <= min(width, height) * .06
    strands &= near & ~protected
    seeds = strands & binary_dilation(anchors, iterations=max(1, round(min(width, height) * .004)))
    connected = binary_propagation(seeds, structure=np.ones((3, 3)), mask=strands)
    expanded = binary_dilation(connected, iterations=max(1, round(min(width, height) * .0015)))
    expanded[protected] = False
    feather = gaussian_filter(expanded.astype(np.float32), .7)
    feather[protected] = 0
    allowed = np.rint((1 - np.clip(feather, 0, 1)) * 255).astype(np.uint8)
    return Image.fromarray(allowed), int(connected.sum())


def repair_hair_color(image, mask, allowed):
    """Complete only hair-connected skin pixels with trusted photo donors."""
    from scipy.ndimage import distance_transform_edt
    from PIL import Image
    from texture_region_completion import complete_color

    source = np.asarray(image)
    confidence = np.asarray(mask)
    allowed = np.asarray(allowed)
    region = (1 - allowed.astype(np.float32) / 255) * (confidence > 32)
    region[[0, -1], :] = 0
    region[:, [0, -1]] = 0
    donors = (confidence > 192) & (allowed > 250)
    if not region.any() or not donors.any():
        return image, 0
    # Non-skin and rejected hairs must not donate their color to the fill.
    # This temporary donor extension is never written outside the repair mask.
    nearest = distance_transform_edt(~donors, return_distances=False, return_indices=True)
    prepared = source.copy()
    prepared[~donors] = source[nearest[0][~donors], nearest[1][~donors]]
    proposal, _ = complete_color(prepared, region, radius=max(2., min(image.size) * .004))
    output = source.copy()
    active = region > 0
    weight = region[active, None]
    output[active] = np.rint(source[active] * (1 - weight) + proposal[active] * weight).clip(0, 255).astype(np.uint8)
    return Image.fromarray(output), int(np.count_nonzero(np.any(output != source, axis=2)))


def repair_nostril_shadows(image, mask, landmarks):
    """Infer local skin only inside detected source cavity shadows.

    Completing the rejected shadow before transfer avoids replacing a whole
    alar territory with differently pigmented native skin. The mesh supplies
    the actual opening. Every source pixel outside the soft mask stays exact.
    """
    from PIL import Image
    allowed, count = exclude_nostril_apertures(
        image, Image.new('L', image.size, 255), landmarks)
    if not count:
        return image, 0
    return repair_hair_color(image, mask, allowed)


def skin_mask(image, mask, model, landmarks=None, repair_hair=False):
    """Restrict a landmark mask to confidently segmented visible skin."""
    if not model.is_file():
        return mask, 'LANDMARK_OVAL', 0, image
    import mediapipe as mp
    from mediapipe.tasks import python
    from mediapipe.tasks.python import vision
    from PIL import Image, ImageFilter

    options = vision.ImageSegmenterOptions(
        base_options=python.BaseOptions(model_asset_path=str(model)),
        output_confidence_masks=True)
    with vision.ImageSegmenter.create_from_options(options) as segmenter:
        labels = segmenter.labels
        indices = [labels.index(name) for name in ('face-skin', 'body-skin')]
        result = segmenter.segment(mp.Image(
            image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(image)))
        confidence = sum(result.confidence_masks[index].numpy_view().squeeze().copy()
                         for index in indices)
        hair = result.confidence_masks[labels.index('hair')].numpy_view().squeeze().copy()
    confidence = np.clip((confidence - .55) / .30, 0, 1)
    semantic = Image.fromarray(np.rint(confidence * 255).astype(np.uint8))
    radius = max(2, round(min(image.size) * .003))
    semantic = semantic.filter(ImageFilter.MinFilter(2 * radius + 1))
    feather = semantic.filter(ImageFilter.GaussianBlur(max(2, min(image.size) * .015)))
    semantic = np.asarray(semantic, dtype=np.float32) * np.asarray(feather, dtype=np.float32) / 255
    combined = np.minimum(np.asarray(mask), semantic).astype(np.uint8)
    if repair_hair:
        allowed, _ = fine_hair_mask(image, hair, landmarks)
        image, count = repair_hair_color(image, Image.fromarray(combined), allowed)
        return Image.fromarray(combined), 'SEMANTIC_SKIN_HAIR_REPAIR', count, image
    return Image.fromarray(combined), 'SEMANTIC_SKIN', 0, image


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
    mask = exclude_eye_apertures(mask, faces[0])
    image, nostril_pixels = repair_nostril_shadows(image, mask, faces[0])
    segmentation_model = Path(request['detector']).parent / 'selfie_multiclass_256x256.tflite'
    mask, mask_method, hair_pixels, image = skin_mask(
        image, mask, segmentation_model, faces[0], request.get('repair_hair', False))
    fallback = Image.open(folder / 'fallback.png').convert('RGBA').resize(
        (request['resolution'], request['resolution']), Image.Resampling.BILINEAR)
    base = np.asarray(fallback, dtype=np.float32)[::-1].copy() / 255
    base[..., :3] = linear(base[..., :3])
    photo = linear(np.asarray(image, dtype=np.float32)[::-1] / 255)
    with np.load(folder / 'surface.npz', allow_pickle=False) as surface:
        protection = surface['skin_protection'] if 'skin_protection' in surface else None
        protected_eyelids = int(np.count_nonzero(surface['eyelid_protection'] > .1)) if 'eyelid_protection' in surface else (int(np.count_nonzero(protection > .1)) if protection is not None else 0)
        protected_nostrils = int(np.count_nonzero(surface['nostril_protection'] > .1)) if 'nostril_protection' in surface else 0
        blink_mask = (raster_protection(surface['blink_uv'], surface['blink_weights'], base.shape[:2])
                      if 'blink_uv' in surface else None)
        alignment_report = {}
        primary_warp = None
        if 'view_landmarks' in surface:
            expected = surface['view_landmarks'][:, :2] / request['camera_scale'] * [
                max(width, height) / width, max(width, height) / height] + .5
            observed = np.array([(faces[0][i].x, 1 - faces[0][i].y) for i in MEDIAPIPE_68])
            primary_warp, alignment_report = feature_alignment(expected, observed)
        primary, primary_weight = project_samples(
            surface['points'], surface['triangles'], surface['uv'], photo,
            np.asarray(mask, dtype=np.float32)[::-1] / 255, base.shape[:2],
            request['camera_scale'], warp=primary_warp, vertex_protection=protection,
            texture_protection=blink_mask)
        influence = primary_weight * request['strength']
        output = base.copy()
        output[..., :3] += (primary - output[..., :3]) * influence[..., None]
        supporting_views = []
        views = request.get('supporting_photos', [])
        if views:
            from PIL import ImageOps

            original_coverage = influence.copy()
            color, confidence = primary.copy(), primary_weight.copy()
            with vision.FaceLandmarker.create_from_options(options) as detector:
                for view in views:
                    slot = view['slot']
                    record = {'slot': slot, 'name': view['name'], 'status': 'rejected',
                              'generated': bool(view.get('generated', False))}
                    try:
                        if type(slot) is not int or not 0 <= slot < 4:
                            raise ValueError('Invalid supporting photo slot')
                        with Image.open(folder / f'support-{slot}.input') as raw:
                            if raw.width * raw.height > 32_000_000:
                                raise ValueError('Supporting photo exceeds 32 megapixels')
                            supporting = ImageOps.exif_transpose(raw).convert('RGB')
                            supporting.thumbnail((4096, 4096))
                        detected = detector.detect(mp.Image(
                            image_format=mp.ImageFormat.SRGB,
                            data=np.ascontiguousarray(supporting))).face_landmarks
                        if len(detected) != 1:
                            raise ValueError('Supporting photo needs exactly one visible face')
                        sw, sh = supporting.size
                        pixels = np.array([(p.x * sw, p.y * sh) for p in detected[0]])
                        eye = np.linalg.norm(pixels[33] - pixels[263])
                        face_height = np.linalg.norm(pixels[168] - pixels[152])
                        baseline = max(eye, face_height * .55)
                        if baseline < 80 or np.linalg.norm(pixels[13] - pixels[14]) / baseline > .08:
                            raise ValueError('Use a close, neutral, closed-mouth supporting photo')
                        target = (pixels[list(MEDIAPIPE_68)] - [sw / 2, sh / 2]) * [1, -1] / max(sw, sh)
                        center, rotation, shift, scale, alignment = register_view(surface['view_landmarks'], target)
                        points = (surface['points'] - center) @ rotation.T
                        points[:, :2] += shift
                        aligned_landmarks = (surface['view_landmarks'] - center) @ rotation.T
                        aligned_landmarks[:, :2] += shift
                        expected = aligned_landmarks[:, :2] / scale * [max(sw, sh) / sw, max(sw, sh) / sh] + .5
                        observed = pixels[list(MEDIAPIPE_68)] / [sw, sh] * [1, -1] + [0, 1]
                        warp, feature_report = feature_alignment(expected, observed, alignment['feature_landmarks'])
                        support_mask = Image.new('L', supporting.size)
                        ImageDraw.Draw(support_mask).polygon(
                            [tuple(pixels[i]) for i in oval], fill=255)
                        radius = max(2, round(min(sw, sh) * .008))
                        support_mask = support_mask.filter(ImageFilter.MinFilter(2 * radius + 1)).filter(
                            ImageFilter.GaussianBlur(radius))
                        support_mask = exclude_eye_apertures(support_mask, detected[0])
                        supporting, support_nostril_pixels = repair_nostril_shadows(
                            supporting, support_mask, detected[0])
                        support_mask, support_mask_method, support_hair_pixels, supporting = skin_mask(
                            supporting, support_mask, segmentation_model, detected[0], request.get('repair_hair', False))
                        colors, weights = project_samples(
                            points, surface['triangles'], surface['uv'],
                            linear(np.asarray(supporting, dtype=np.float32)[::-1] / 255),
                            np.asarray(support_mask, dtype=np.float32)[::-1] / 255,
                            base.shape[:2], scale, warp=warp, vertex_protection=protection,
                            texture_protection=blink_mask)
                        colors, exposure_gain, overlap = balance_view(
                            primary, primary_weight, colors, weights)
                        confidence = merge_view(color, confidence, colors, weights, primary_weight)
                        record.update(status='accepted', **alignment, exposure_gain=exposure_gain,
                                      calibrated_texels=overlap, feature_alignment=feature_report,
                                      mask_method=support_mask_method, fine_hair_repaired_pixels=support_hair_pixels,
                                      nostril_aperture_pixels=support_nostril_pixels)
                    except (ValueError, OSError) as exc:
                        record['reason'] = str(exc)
                    supporting_views.append(record)
            if not any(view['status'] == 'accepted' for view in supporting_views):
                reasons = '. '.join(view['reason'] for view in supporting_views)
                raise ValueError('No supporting photo could be used. ' + reasons)
            influence = confidence * request['strength']
            output[..., :3] = base[..., :3] * (1 - influence[..., None]) + color * influence[..., None]
            added_texels = int(np.count_nonzero((influence > .1) & (original_coverage <= .1)))
        else:
            added_texels = 0
    gain = np.ones(3)
    matched = base.copy()
    neck_report = {'status': 'disabled'}
    completion_report = {'status': 'disabled'}
    if request['match_skin']:
        valid = primary_weight > .95
        luminance = output[..., :3] @ [.2126, .7152, .0722]
        if np.count_nonzero(valid) < 100:
            raise ValueError('Insufficient fully visible face pixels for tone matching; disable Match Skin Tone')
        low, high = np.percentile(luminance[valid], [25, 75])
        valid &= (luminance >= low) & (luminance <= high)
        old = np.median(base[..., :3][valid], axis=0)
        new = np.median(output[..., :3][valid], axis=0)
        gain = np.clip(new / np.maximum(old, .005), .1, 32)
        chroma = old / max(old.sum(), 1e-8)
        with np.load(folder / 'surface.npz', allow_pickle=False) as surface:
            matched, completion_report = complete_skin_tone(
                base, output, influence, surface['triangles'], surface['uv'], gain, points=surface['points'])
        output[..., :3] += (matched[..., :3] - base[..., :3]) * (1 - influence[..., None])
        if request['body']:
            body = Image.open(folder / 'body.png').convert('RGBA')
            body.thumbnail((request['resolution'], request['resolution']), Image.Resampling.BILINEAR)
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
    with np.load(folder / 'surface.npz', allow_pickle=False) as surface:
        coverage = uv_coverage(surface['uv'], output.shape[:2])
        output, influence, edge_report = feather_projection(
            output, matched, influence, coverage, max(1, request['resolution'] // 1024))
        if 'nostril_protection' in surface:
            # A thin aperture needs a longer transition than a broad cheek.
            # Restrict it to native anatomical weights, retaining pore detail
            # elsewhere and never extending photo color into the cavity.
            region = raster_protection(surface['uv'],
                surface['nostril_protection'][surface['triangles']], output.shape[:2])
            region = np.clip(region * 4, 0, 1)
            output, influence, regional = feather_projection(
                output, matched, influence, coverage,
                max(1, request['resolution'] * 3 // 1024), region=region)
            edge_report['nostril_transition'] = regional
        texture_gutters((output, matched), surface['uv'], coverage=coverage)
    save(output, folder / 'color.png')
    save(matched, folder / 'clean.png')
    Image.fromarray(np.rint(np.clip(influence[::-1], 0, 1) * 255).astype(np.uint8)).save(folder / 'mask.png')
    return {'status': 'complete', 'seconds': time.perf_counter() - start,
            'resolution': request['resolution'], 'projected_texels': int(np.count_nonzero(influence > .1)),
            'skin_tone_gain': gain.tolist(), 'matched_body': request['body'],
            'supporting_views': supporting_views, 'added_view_texels': added_texels,
            'primary_feature_alignment': alignment_report,
            'mask_method': mask_method,
            'hair_repair_enabled': bool(request.get('repair_hair', False)),
            'fine_hair_repaired_pixels': hair_pixels,
            'eye_apertures_masked': True,
            'nostril_aperture_pixels': nostril_pixels,
            'nostril_shadow_method': 'LOCAL_SKIN_COMPLETION_NATIVE_RIM',
            'protected_eyelid_vertices': protected_eyelids,
            'protected_nostril_vertices': protected_nostrils,
            'blink_protected_texels': int(np.count_nonzero(blink_mask > .1)) if blink_mask is not None else 0,
            'tone_completion': completion_report,
            'projection_edge_fade': edge_report,
            'clean_reference': True,
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
