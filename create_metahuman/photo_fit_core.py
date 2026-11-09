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


def project(points, camera):
    return np.exp(camera[3]) * (points @ rotation(camera[:3]).T)[:, :2] + camera[4:6]


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
    camera_fit = least_squares(lambda c: ((project(mean, c) - target) * factor).ravel(), camera,
                               bounds=([-.52, -.52, -.70, -8, -20, -20], [.52, .52, .70, 12, 20, 20]),
                               loss='soft_l1', f_scale=.025, max_nfev=80)
    camera = camera_fit.x
    before = float(np.sqrt(np.mean(np.sum((project(mean, camera) - target) ** 2, axis=1))))
    prior = .015

    def residual(vector):
        points = mean + np.tensordot(vector[6:], modes, axes=(0, 0))
        return np.concatenate((((project(points, vector[:6]) - target) * factor).ravel(),
                               prior * vector[6:]))

    def jacobian(vector):
        points = mean + np.tensordot(vector[6:], modes, axes=(0, 0))
        current = project(points, vector[:6])
        result = np.zeros((136 + 170, 176))
        for index in range(6):
            camera = vector[:6].copy()
            camera[index] += 1e-6
            result[:136, index] = (((project(points, camera) - current) / 1e-6) * factor).ravel()
        projected = np.exp(vector[3]) * (modes @ rotation(vector[:3]).T)[:, :, :2]
        result[:136, 6:] = (projected * factor[None]).reshape(170, 136).T
        result[136:, 6:] = np.eye(170) * prior
        return result

    solved = least_squares(residual, np.r_[camera, np.clip(initial[:170], -2.5, 2.5)], jac=jacobian,
                           bounds=(np.r_[[-.52, -.52, -.70, -8, -20, -20], np.full(170, -2.5)],
                                   np.r_[[.52, .52, .70, 12, 20, 20], np.full(170, 2.5)]),
                           loss='soft_l1', f_scale=.025, max_nfev=80, ftol=1e-6, xtol=1e-6, gtol=1e-6)
    coefficients = initial.copy()
    coefficients[:170] = solved.x[6:]
    predicted = project(mean + np.tensordot(coefficients[:170], modes, axes=(0, 0)), solved.x[:6])
    after = float(np.sqrt(np.mean(np.sum((predicted - target) ** 2, axis=1))))
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
            'evaluations': int(solved.nfev)}
