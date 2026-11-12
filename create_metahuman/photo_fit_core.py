"""Bounded neutral-head fitting from explicitly corresponding image landmarks.

This estimates a starting shape, not an identity or unseen facial geometry.
The head coefficients are fitted; eyeball and dental identity stay untouched.
"""

import numpy as np

# Anatomical correspondence from the standard sparse-68 ordering to MediaPipe.
# Jaw contours receive less weight because their apparent location is view dependent.
MEDIAPIPE_68 = (
    234, 93, 132, 58, 172, 136, 150, 149, 152, 378, 379, 365, 397, 288, 361, 323, 454,
    70, 63, 105, 66, 107, 336, 296, 334, 293, 300,
    168, 6, 197, 1, 98, 97, 2, 326, 327,
    33, 160, 158, 133, 153, 144, 362, 385, 387, 263, 373, 380,
    61, 40, 37, 0, 267, 270, 291, 321, 314, 17, 84, 91,
    78, 81, 13, 311, 308, 402, 14, 178,
)


def rotation(angles):
    pitch, yaw, roll = angles
    a, b, c = np.cos(angles)
    x, y, z = np.sin((pitch, yaw, roll))
    rx = np.array(((1, 0, 0), (0, a, -x), (0, x, a)))
    ry = np.array(((b, 0, y), (0, 1, 0), (-y, 0, b)))
    rz = np.array(((c, -z, 0), (z, c, 0), (0, 0, 1)))
    return rz @ ry @ rx


def review_rendered_landmarks(reference, candidate):
    """Compare independently detected frontal photo/render landmarks.

    Translation and uniform scale align iris centers. No rotation, reflection,
    affine warp or facial deformation is applied. Errors are normalized by
    inter-iris distance, so use the same view and neutral expression. This is
    a review diagnostic, not recovered geometry or an overall likeness score.
    """
    normalized = []
    for points in (reference, candidate):
        points = np.asarray(points, dtype=float)
        if points.shape != (478, 2) or not np.isfinite(points).all():
            raise ValueError('Use 478 finite, independently detected pixel landmarks for each image')
        eyes = points[[468, 473]]
        distance = float(np.linalg.norm(eyes[1] - eyes[0]))
        if distance < 20:
            raise ValueError('Use a closer image with at least 20 pixels between detected irises')
        normalized.append((points - eyes.mean(axis=0)) / distance)
    regions = dict(jaw=range(17), brows=range(17, 27), nose=range(27, 36),
                   eyes=range(36, 48), mouth=range(48, 68))
    delta = (normalized[1] - normalized[0])[list(MEDIAPIPE_68)]
    return dict(method='IRIS_CENTER_UNIFORM_SCALE',
                regions={name: dict(rms_interiris=float(np.sqrt(np.mean(np.sum(delta[list(indices)] ** 2, axis=1)))))
                         for name, indices in regions.items()},
                limitation='Use frontal neutral images. Detection, lighting, camera projection and expression affect these measurements; lower regional error alone does not establish better likeness.')


def project(points, camera):
    return np.exp(camera[3]) * (points @ rotation(camera[:3]).T)[:, :2] + camera[4:6]


def primary_corridor(projected, baseline, target):
    """Protect a frontal fit while allowing corrections toward its photograph.

    The segment from each accepted coordinate to its observed target is free.
    Excursions outside it retain the old anchor penalty. Return its derivative
    as well, so the joint solve uses the same piecewise objective and Jacobian.
    """
    lower, upper = np.minimum(baseline, target), np.maximum(baseline, target)
    return projected - np.clip(projected, lower, upper), ((projected <= lower) | (projected >= upper)).astype(float)


def region_errors(projected, baseline, target, normalizer):
    regions = [('jaw', 0, 17), ('brows', 17, 27), ('nose', 27, 36),
               ('left_eye', 36, 42), ('right_eye', 42, 48), ('mouth', 48, 68)]
    result = {}
    for name, start, end in regions:
        def error(points):
            return float(np.sqrt(np.mean(np.sum((points[start:end] - target[start:end]) ** 2, axis=1))) / normalizer)
        result[name] = dict(before=error(baseline), after=error(projected))
    return result


def eye_shape_operator(target):
    """Compare local lid contours independently of each eye's image position.

    Reference eye width fixes the normalization; predicted widths cannot shrink
    the penalty. Centering retains width, aperture and corner slope while the
    ordinary observations still determine eye placement and separation.
    """
    target=np.asarray(target,dtype=float)
    operator=np.zeros((24,136))
    for eye,start in enumerate((36,42)):
        width=float(np.linalg.norm(target[start+3]-target[start]))
        if not np.isfinite(width) or width<1e-8:
            raise ValueError('The primary photo does not resolve both eye contours')
        block=np.kron(np.eye(6)-np.ones((6,6))/6,np.eye(2))
        operator[eye*12:(eye+1)*12,start*2:(start+6)*2]=block*.5/width
    return operator


def landmarks(template, basis, definition):
    data = np.asarray(definition)
    if data.shape != (68, 6) or not np.isfinite(data).all():
        raise ValueError('The GNM sparse-68 landmark definition is invalid')
    ids = data[:, ::2].astype(np.int64)
    weights = data[:, 1::2].copy()
    if (ids < 0).any() or (ids >= len(template)).any() or (weights < 0).any() or (weights.sum(1) <= 0).any():
        raise ValueError('The landmark definition does not match this GNM model')
    weights /= weights.sum(1)[:, None]
    return (np.sum(template[ids] * weights[:, :, None], axis=1),
            np.sum(basis[:170, ids] * weights[None, :, :, None], axis=2))


def fit(template, basis, definition, target, initial=None):
    """Fit near-frontal weak-perspective camera and bounded head coefficients."""
    from scipy.optimize import least_squares

    target = np.asarray(target, dtype=np.float64)
    if target.shape != (68, 2) or not np.isfinite(target).all():
        raise ValueError('Photo fitting needs 68 finite, corresponding image points')
    mean, modes = landmarks(template, basis, definition)
    eye_distance = np.linalg.norm(target[36] - target[45])
    if eye_distance < 1e-5:
        raise ValueError('The photo landmarks do not resolve both eyes')
    initial = np.zeros(253) if initial is None else np.asarray(initial, dtype=np.float64)
    if initial.shape != (253,) or not np.isfinite(initial).all():
        raise ValueError('The initial head parameters are invalid')
    weights = np.ones(68)
    weights[:17] = .25
    weights[17:27] = .40
    weights[48:] = .50
    weights[60:] = .20
    scale = np.linalg.norm(target[36] - target[45]) / np.linalg.norm(mean[36] - mean[45])
    camera = np.array((0., 0., 0., np.log(scale), 0., 0.))
    camera[4:6] = target.mean(0) - scale * mean[:, :2].mean(0)
    factor = np.sqrt(weights[:, None]) / eye_distance
    eye_shape=eye_shape_operator(target)
    camera_fit = least_squares(lambda c: ((project(mean, c) - target) * factor).ravel(), camera,
                               bounds=([-.52, -.52, -.70, -8, -20, -20], [.52, .52, .70, 12, 20, 20]),
                               loss='soft_l1', f_scale=.025, max_nfev=80)
    camera = camera_fit.x
    before = float(np.sqrt(np.mean(np.sum((project(mean, camera) - target) ** 2, axis=1))))
    prior = .015

    def residual(vector):
        points = mean + np.tensordot(vector[6:], modes, axes=(0, 0))
        return np.concatenate((((project(points, vector[:6]) - target) * factor).ravel(),
                               eye_shape@(project(points, vector[:6])-target).ravel(),
                               prior * vector[6:]))

    def jacobian(vector):
        points = mean + np.tensordot(vector[6:], modes, axes=(0, 0))
        current = project(points, vector[:6])
        result = np.zeros((136 + 24 + 170, 176))
        raw=np.zeros((136,176))
        for index in range(6):
            camera = vector[:6].copy()
            camera[index] += 1e-6
            raw[:,index]=((project(points,camera)-current)/1e-6).ravel()
        projected = np.exp(vector[3]) * (modes @ rotation(vector[:3]).T)[:, :, :2]
        raw[:,6:]=projected.reshape(170,136).T
        result[:136]=raw*np.repeat(factor[:,0],2)[:,None]
        result[136:160]=eye_shape@raw
        result[160:, 6:] = np.eye(170) * prior
        return result

    solved = least_squares(residual, np.r_[camera, np.clip(initial[:170], -2.5, 2.5)], jac=jacobian,
                           bounds=(np.r_[[-.52, -.52, -.70, -8, -20, -20], np.full(170, -2.5)],
                                   np.r_[[.52, .52, .70, 12, 20, 20], np.full(170, 2.5)]),
                           loss='soft_l1', f_scale=.025, max_nfev=80, ftol=1e-6, xtol=1e-6, gtol=1e-6)
    coefficients = initial.copy()
    coefficients[:170] = solved.x[6:]
    predicted = project(mean + np.tensordot(coefficients[:170], modes, axes=(0, 0)), solved.x[:6])
    after = float(np.sqrt(np.mean(np.sum((predicted - target) ** 2, axis=1))))
    eye_before=float(np.sqrt(np.mean((eye_shape@(project(mean,camera)-target).ravel())**2)))
    eye_after=float(np.sqrt(np.mean((eye_shape@(predicted-target).ravel())**2)))
    if not solved.success or not np.isfinite(coefficients).all() or after >= before:
        raise ValueError('Photo fitting did not converge to an improved shape; use another neutral reference')
    if np.max(np.abs(solved.x[:2])) > .49:
        raise ValueError('Use a near-frontal reference with both cheeks and eyes visible')
    if after / eye_distance > .12:
        raise ValueError('The landmarks do not fit a neutral head closely enough; use a clearer neutral reference')
    return {'coefficients': coefficients.tolist(), 'camera': solved.x[:6].tolist(),
            'projected': predicted.tolist(), 'target': target.tolist(),
            'before_rms': before, 'after_rms': after, 'eye_distance': float(eye_distance),
            'bounded_head_coefficients': int(np.sum(np.abs(coefficients[:170]) > 2.49)),
            'primary_eye_shape':dict(before=eye_before,after=eye_after,method='CENTERED_REFERENCE_WIDTH'),
            'evaluations': int(solved.nfev)}


def fit_multiview(template, basis, definition, views, initial, initial_camera, current_landmarks=None):
    """Joint bounded identity/camera refinement, protecting the primary front.

    Generated supports carry one quarter of a photograph's observation weight.
    Camera-only fits establish the baseline before any identity change is judged.
    Primary landmarks may move toward their photograph instead of freezing an
    existing mismatch. Regional checks prevent a global win from hiding worse eyes.
    """
    from scipy.optimize import least_squares
    from scipy.spatial.transform import Rotation
    if __package__:
        from .photo_color_core import register_view
    else:
        from photo_color_core import register_view

    initial = np.asarray(initial, dtype=float)
    if (initial.shape != (253,) or not np.isfinite(initial).all()
            or np.max(np.abs(initial[:170])) > 2.50001 or not 2 <= len(views) <= 5):
        raise ValueError('Multi-view fitting needs a bounded photo identity and one to four supporting views')
    mean, modes = landmarks(template, basis, definition)
    baseline = mean + np.tensordot(initial[:170], modes, axes=(0, 0))
    if current_landmarks is not None:
        current_landmarks = np.asarray(current_landmarks, dtype=float)
        if current_landmarks.shape != (68, 3) or not np.isfinite(current_landmarks).all():
            raise ValueError('Current generated landmarks are invalid')
        baseline = current_landmarks
    cameras, factors, targets, records = [], [], [], []
    bounds = ([-.62, -1.62, -.85, -8, -20, -20], [.62, 1.62, .85, 12, 20, 20])
    for index, view in enumerate(views):
        target = np.asarray(view['target'], dtype=float)
        if target.shape != (68, 2) or not np.isfinite(target).all():
            raise ValueError('Each view needs 68 finite landmarks')
        weights = np.ones(68)
        weights[:17], weights[17:27], weights[48:], weights[60:] = .15, .4, .5, .2
        if index == 0:
            camera = np.asarray(initial_camera, dtype=float)
            if camera.shape != (6,) or not np.isfinite(camera).all():
                raise ValueError('The primary photo camera is invalid')
            normalizer = np.linalg.norm(target[36] - target[45])
        else:
            try:
                center, matrix, shift, scale, alignment = register_view(baseline, target)
            except ValueError as exc:
                raise ValueError(f"{view.get('name', index)}: {exc}") from exc
            camera = np.r_[Rotation.from_matrix(matrix).as_euler('xyz'), np.log(1 / scale),
                           (shift - (center @ matrix.T)[:2]) / scale]
            visible = np.zeros(68)
            visible[alignment['used_landmarks']] = 1
            weights *= visible
            normalizer = max(np.linalg.norm(target[36] - target[45]),
                             np.linalg.norm(target[27] - target[8]) * .55)
        if normalizer < 1e-5:
            raise ValueError('The face in a supporting view is too small')
        trust = .25 if index and view.get('generated') else 1.
        factor = np.sqrt(weights[:, None] * trust) / normalizer
        solved_camera = least_squares(lambda c: ((project(baseline, c) - target) * factor).ravel(),
                                      np.clip(camera, np.array(bounds[0]) + 1e-7, np.array(bounds[1]) - 1e-7),
                                      bounds=bounds, loss='soft_l1', f_scale=.025, max_nfev=80)
        if not solved_camera.success:
            raise ValueError('A supporting camera could not be fitted reliably')
        cameras.append(solved_camera.x)
        factors.append(factor)
        targets.append(target)
        records.append(dict(name=view.get('name', str(index)), generated=bool(view.get('generated')),
                            observation_weight=trust, used_landmarks=np.flatnonzero(weights).tolist()))
    anchor = project(baseline, cameras[0])
    anchor_factor = np.sqrt(5.) / max(np.linalg.norm(targets[0][36] - targets[0][45]), 1e-8)
    prior = .025
    count = len(views)
    eye_shape=eye_shape_operator(targets[0])

    def points_for(vector):
        return baseline + np.tensordot(vector[:170] - initial[:170], modes, axes=(0, 0))

    def residual(vector):
        points = points_for(vector)
        camera_values = vector[170:].reshape(count, 6)
        observations = [((project(points, camera) - target) * factor).ravel()
                        for camera, target, factor in zip(camera_values, targets, factors)]
        return np.r_[np.concatenate(observations),
                     eye_shape@(project(points,camera_values[0])-targets[0]).ravel(),
                     (primary_corridor(project(points, camera_values[0]), anchor, targets[0])[0] * anchor_factor).ravel(),
                     prior * (vector[:170] - initial[:170])]

    def jacobian(vector):
        points = points_for(vector)
        camera_values = vector[170:].reshape(count, 6)
        result = np.zeros((136 * (count + 1) + 24 + 170, 170 + 6 * count))
        anchor_start=136*count+24
        for index, (camera, factor) in enumerate(zip(camera_values, factors)):
            current = project(points, camera)
            anchor_slope = primary_corridor(current, anchor, targets[0])[1] if index == 0 else None
            identity = np.exp(camera[3]) * (modes @ rotation(camera[:3]).T)[:, :, :2]
            rows = slice(index * 136, (index + 1) * 136)
            result[rows, :170] = (identity * factor[None]).reshape(170, 136).T
            for axis in range(6):
                shifted = camera.copy()
                shifted[axis] += 1e-6
                derivative = (project(points, shifted) - current) / 1e-6
                result[rows, 170 + 6 * index + axis] = (derivative * factor).ravel()
                if index == 0:
                    result[136*count:136*count+24,170+axis]=eye_shape@derivative.ravel()
                    result[anchor_start:anchor_start+136, 170 + axis] = (derivative * anchor_slope * anchor_factor).ravel()
            if index == 0:
                result[136*count:136*count+24,:170]=eye_shape@identity.reshape(170,136).T
                result[anchor_start:anchor_start+136, :170] = (identity * anchor_slope[None] * anchor_factor).reshape(170, 136).T
        result[-170:, :170] = np.eye(170) * prior
        return result

    seed = np.r_[initial[:170], np.asarray(cameras).ravel()]
    lower = np.r_[np.maximum(-2.5, initial[:170] - .6), np.tile(bounds[0], count)]
    upper = np.r_[np.minimum(2.5, initial[:170] + .6), np.tile(bounds[1], count)]
    solved = least_squares(residual, np.clip(seed, lower, upper), jac=jacobian, bounds=(lower, upper),
                           loss='soft_l1', f_scale=.025, max_nfev=100, ftol=1e-6, xtol=1e-6, gtol=1e-6)
    points = points_for(solved.x)
    fitted_cameras = solved.x[170:].reshape(count, 6)
    before, after = 0., 0.
    for index, (target, factor, record) in enumerate(zip(targets, factors, records)):
        old = ((project(baseline, cameras[index]) - target) * factor).ravel()
        new = ((project(points, fitted_cameras[index]) - target) * factor).ravel()
        record.update(before_normalized_rms=float(np.sqrt(np.mean(old ** 2))),
                      after_normalized_rms=float(np.sqrt(np.mean(new ** 2))),
                      camera_before=cameras[index].tolist(),
                      camera=fitted_cameras[index].tolist())
        before += float(old @ old)
        after += float(new @ new)
    drift = float(np.sqrt(np.mean((project(points, fitted_cameras[0]) - anchor) ** 2)) * anchor_factor / np.sqrt(5.))
    predicted = project(points, fitted_cameras[0])
    outward = float(np.sqrt(np.mean(primary_corridor(predicted, anchor, targets[0])[0] ** 2)) * anchor_factor / np.sqrt(5.))
    regions = region_errors(predicted, anchor, targets[0], np.linalg.norm(targets[0][36] - targets[0][45]))
    eye_before=float(np.sqrt(np.mean((eye_shape@(anchor-targets[0]).ravel())**2)))
    eye_after=float(np.sqrt(np.mean((eye_shape@(predicted-targets[0]).ravel())**2)))
    before_by_eye=np.sqrt(np.mean((eye_shape@(anchor-targets[0]).ravel()).reshape(2,12)**2,axis=1))
    after_by_eye=np.sqrt(np.mean((eye_shape@(predicted-targets[0]).ravel()).reshape(2,12)**2,axis=1))
    if not solved.success or not np.isfinite(solved.x).all() or after >= before * .995:
        raise ValueError('Supporting views did not improve shape beyond camera-only alignment; the source is retained')
    if outward > .01 or records[0]['after_normalized_rms'] > records[0]['before_normalized_rms'] * 1.05 + .002:
        raise ValueError('Supporting views would change the primary likeness too much; the source is retained')
    worsened = [name for name, value in regions.items() if value['after'] > value['before'] * 1.05 + .002]
    if worsened:
        raise ValueError('Supporting views would worsen primary ' + ', '.join(worsened) + '; the source is retained')
    if np.any(after_by_eye>before_by_eye*1.05+.002):
        raise ValueError('Supporting views would worsen the primary eye contours; the source is retained')
    coefficients = initial.copy()
    coefficients[:170] = solved.x[:170]
    return dict(coefficients=coefficients.tolist(), camera=fitted_cameras[0].tolist(),
                projected=predicted.tolist(), target=targets[0].tolist(),
                before_rms=float(np.sqrt(np.mean(np.sum((anchor - targets[0]) ** 2, axis=1)))),
                after_rms=float(np.sqrt(np.mean(np.sum((predicted - targets[0]) ** 2, axis=1)))),
                eye_distance=float(np.linalg.norm(targets[0][36] - targets[0][45])),
                method='JOINT_MULTIVIEW_IDENTITY', views=records, primary_drift_normalized=drift,
                primary_outward_drift_normalized=outward, primary_regions=regions,
                primary_protection='TARGET_DIRECTED_CORRIDOR',
                primary_eye_shape=dict(before=eye_before,after=eye_after,method='CENTERED_REFERENCE_WIDTH',
                                       per_eye={name:dict(before=float(a),after=float(b))
                                                for name,a,b in zip(('image_left','image_right'),before_by_eye,after_by_eye)}),
                camera_only_baseline=True, weighted_error_before=before, weighted_error_after=after,
                bounded_head_coefficients=int(np.count_nonzero(np.abs(coefficients[:170]) > 2.49)),
                evaluations=int(solved.nfev))
