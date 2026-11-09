"""Posture, maturity and asymmetry on compatible reference-weighted spines."""
import numpy as np

from . import body_pose

SPINE = tuple('spine_' + str(index).zfill(2) for index in range(1, 6))


def weights(obj, bones):
    """Assign each skin influence to its nearest spine ancestor.

    Arms and hands follow their shoulder's spine transform rigidly. Pelvis and
    leg influences have no spine ancestor and therefore stay anchored.
    """
    columns = {name: index for index, name in enumerate(SPINE)}
    groups = {}
    for group in obj.vertex_groups:
        name = body_pose._name(group.name)
        seen = set()
        while name and name not in seen:
            if name in columns:
                groups[group.index] = columns[name]
                break
            seen.add(name)
            name = bones.get(name, {}).get('parent', '')
    result = np.zeros((len(obj.data.vertices), len(SPINE)))
    for vertex in obj.data.vertices:
        for item in vertex.groups:
            column = groups.get(item.group)
            if column is not None:
                result[vertex.index, column] += max(item.weight, 0)
    # Overweight custom skin cannot amplify translation; unassigned weight
    # remains at rest, preserving the pelvis transition of normalized skins.
    result /= np.maximum(result.sum(axis=1), 1)[:, None]
    return result


def transforms(joint, value, front_axis='-Y', *, age=0, asymmetry=0):
    pelvis, neck = joint('pelvis'), joint('neck_01')
    up = neck - pelvis
    length = np.linalg.norm(up)
    if length < 1e-8:
        raise ValueError('Spine shaping requires distinct pelvis and neck reference joints')
    up /= length
    directions = {'X': (1, 0, 0), '-X': (-1, 0, 0),
                  'Y': (0, 1, 0), '-Y': (0, -1, 0)}
    front = joint.direction(directions.get(front_axis, (0, -1, 0)))
    front -= np.dot(front, up) * up
    magnitude = np.linalg.norm(front)
    if magnitude < 1e-8:
        raise ValueError('Spine shaping cannot determine the front of this reference')
    front /= magnitude
    axis = np.cross(up, front)
    x, y, z = axis
    skew = np.array(((0, -z, y), (z, 0, -x), (-y, x, 0)))
    x, y, z = front
    roll_skew = np.array(((0, -z, y), (z, 0, -x), (-y, x, 0)))
    transform = np.eye(4)
    result = []
    # Distribute flexion, a small maturity compression and signed shoulder
    # imbalance through the actual spine, with the lower body anchored.
    # Compose rigid transforms before blending skin, rather than rotating
    # vertices independently according to their world-space height.
    for name, fraction, side in zip(SPINE, (.10, .15, .20, .25, .30), (-.15, -.15, .15, .40, .75)):
        angle = (.25 * float(value) + .07 * float(age)) * fraction
        rotation = np.eye(3) + np.sin(angle) * skew + (1 - np.cos(angle)) * (skew @ skew)
        if asymmetry:
            angle = .06 * float(asymmetry) * side
            roll = np.eye(3) + np.sin(angle) * roll_skew + (1 - np.cos(angle)) * (roll_skew @ roll_skew)
            rotation = rotation @ roll
        pivot = joint(name)
        local = np.eye(4)
        local[:3, :3] = rotation
        local[:3, 3] = pivot - rotation @ pivot
        if age:
            local[:3, 3] -= up * (length * .025 * float(age) * fraction)
        transform = transform @ local
        result.append(transform.copy())
    return result


def apply(points, skin, matrices):
    output = points.copy()
    for index, transform in enumerate(matrices):
        posed = points @ transform[:3, :3].T + transform[:3, 3]
        output += (posed - points) * skin[:, index, None]
    return output


def deform(points, joint, skin, value, front_axis='-Y'):
    return apply(points, skin, transforms(joint, value, front_axis))
