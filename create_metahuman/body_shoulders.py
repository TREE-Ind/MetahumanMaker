"""Reference-anchored trapezius volume and clavicle definition."""

import numpy as np


def field(joint, parameters, front_axis):
    """A continuous collar field shared by the body and lower head mesh."""
    up = joint('neck_01') - joint('pelvis')
    length = np.linalg.norm(up)
    if length < 1e-8:
        raise ValueError('Shoulder shaping requires distinct pelvis and neck joints')
    up /= length
    neck = joint('neck_01')

    def displacement(points):
        fade = np.clip((neck - points) @ up / (length * .10), 0, 1)
        fade = fade * fade * (3 - 2 * fade)
        return deform(points, joint, 1, parameters, front_axis, fade) - points
    return displacement


def deform(points, joint, weight, parameters, front_axis, neck_fade):
    directions = {'X': (1, 0, 0), '-X': (-1, 0, 0),
                  'Y': (0, 1, 0), '-Y': (0, -1, 0)}
    up = joint('neck_01') - joint('pelvis')
    length = np.linalg.norm(up)
    if length < 1e-8:
        raise ValueError('Shoulder shaping requires distinct pelvis and neck joints')
    up /= length
    front = joint.direction(directions.get(front_axis, (0, -1, 0)))
    front -= np.dot(front, up) * up
    magnitude = np.linalg.norm(front)
    if magnitude < 1e-8:
        raise ValueError('Shoulder shaping cannot determine the front of this reference')
    front /= magnitude
    delta = np.zeros_like(points)
    for side in ('l', 'r'):
        shoulder = joint('upperarm_' + side)
        for name, start, radius, amount, direction, sign in (
            ('traps', joint('neck_01'), .85, .15, up, -1),
            ('clavicles', joint('clavicle_' + side), .85, .15, front, 1),
        ):
            value = parameters.get(name, 0)
            if not value:
                continue
            axis = shoulder - start
            span = np.linalg.norm(axis)
            if span < 1e-8:
                raise ValueError('Shoulder shaping requires distinct neck, clavicle and shoulder joints')
            axis /= span
            offset = points - start
            along = offset @ axis
            radial = offset - along[:, None] * axis
            # A compact capsule follows the actual collar line and fades at
            # both ends. Lower torso, arms and the neckline are not displaced.
            t = np.clip(along / span, 0, 1)
            belly = np.sin(np.pi * t) ** 2
            distance = np.linalg.norm(radial, axis=1) / (radius * span)
            falloff = np.clip(1 - distance ** 2, 0, 1) ** 2
            depth = radial @ front
            sector = np.clip(.5 + sign * depth / max(.20 * span, 1e-8), 0, 1)
            sector *= sector * (3 - 2 * sector)
            strength = amount * span * value * belly * falloff * sector * weight * neck_fade
            delta += strength[:, None] * direction
    return points + delta
