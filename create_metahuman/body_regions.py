"""Joint-anchored secondary body controls on compatible weighted templates."""
import numpy as np
from mathutils import Vector

from . import anatomy, body_pose, body_posture, body_shoulders
from . import mesh_ops as m


def _bindings(obj, template):
    if not template or m.topology(obj) != m.topology(template):
        return {}
    skeleton = body_pose.reference_skeleton(template)
    bones = {body_pose._name(name): {'head': item['head'], 'parent': body_pose._name(item['parent'])}
             for name, item in (skeleton or {}).get('bones', {}).items()}
    if not bones:
        return {}
    names = {group.index: body_pose._name(group.name) for group in obj.vertex_groups}
    roots = [digit + '_01_' + side for side in ('l', 'r')
             for digit in ('thumb', 'index', 'middle', 'ring', 'pinky')]
    roots += ['foot_l', 'foot_r', 'upperarm_l', 'upperarm_r', 'pelvis', 'spine_01', 'clavicle_l', 'clavicle_r']
    members = {}
    for root in roots:
        if root not in bones:
            continue
        group_ids = set()
        for group, name in names.items():
            seen = set()
            while name and name not in seen:
                # A muscle belly uses upper-arm/twist/corrective weights only.
                # Lower-arm and hand descendants must not lengthen or swell.
                if root.startswith('upperarm_') and name.startswith(('lowerarm_', 'hand_')):
                    break
                if root.startswith('clavicle_') and name.startswith('upperarm_'):
                    break
                # Torso controls blend through pelvis/spine skin, excluding
                # articulated limbs and the neck/head branches.
                if root == 'pelvis' and name.startswith(('thigh_', 'clavicle_', 'neck_')):
                    break
                if name == root:
                    group_ids.add(group)
                    break
                seen.add(name)
                name = bones.get(name, {}).get('parent', '')
        if group_ids:
            members[root] = group_ids
    weights = {root: np.zeros(len(obj.data.vertices)) for root in members}
    by_group = {}
    for root, groups in members.items():
        for group in groups:
            by_group.setdefault(group, []).append(root)
    for vertex in obj.data.vertices:
        for item in vertex.groups:
            for root in by_group.get(item.group, ()):
                weights[root][vertex.index] += item.weight
    reference = m.coordinates(template)
    return {root: (np.clip(weight, 0, 1), reference, bones, template.matrix_world.copy())
            for root, weight in weights.items() if np.count_nonzero(weight > .2) >= 8}


def _frame(binding, output):
    weight, reference, bones, matrix = binding
    ids = np.flatnonzero(weight > .2)
    w = weight[ids] ** 2
    w /= w.sum()
    a, b = reference[ids], output[ids]
    ca, cb = w @ a, w @ b
    x, y = a - ca, b - cb
    affine, _, rank, _ = np.linalg.lstsq(x * np.sqrt(w[:, None]), y * np.sqrt(w[:, None]), rcond=1e-5)
    if rank < 3 or not np.isfinite(affine).all():
        return None

    def joint(name):
        point = np.asarray(matrix @ Vector(bones[name]['head']))
        return (point - ca) @ affine + cb
    joint.direction = lambda direction: np.asarray(matrix.to_3x3() @ Vector(direction)) @ affine
    return joint


def _torso_neck_fade(template, output, up, length, neck):
    explicit = template.vertex_groups.get('CM_Neck')
    ring = [vertex.index for vertex in template.data.vertices
            if explicit and any(item.group == explicit.index and item.weight > .5 for item in vertex.groups)] if explicit else []
    if not ring:
        loops, _ = anatomy.geometry.welded_boundary_loops(
            m.coordinates(template, world=False), [p.vertices[:] for p in template.data.polygons])
        # Choose in the fitted anatomical axis, independently of world rotation.
        ring = max(loops, key=lambda indices: np.mean(output[indices] @ up)) if loops else None
    limit = np.min(output[ring] @ up) if ring is not None else np.dot(neck, up)
    fade = np.clip((limit - output @ up) / max(length * .12, 1e-8), 0, 1)
    return fade * fade * (3 - 2 * fade)


def deform(obj, template, points, parameters, *, follow=None):
    """Keep legacy regional shaping when joint/weight correspondence is unavailable.

    Finger length stretches each weighted digit from its own knuckle. Feet scale
    in the ground plane around their ankles; neither control stretches the arm or
    shifts the whole foot away from the body origin. Upper-arm muscles bulge on
    opposite sides of the humerus, without moving the shoulder/elbow pivots.
    """
    requested = {key for key in ('fingers', 'feet', 'ankles', 'biceps', 'triceps', 'back', 'abdomen', 'glutes', 'posture', 'traps', 'clavicles', 'age', 'asymmetry') if parameters.get(key, 0)}
    if not requested:
        return anatomy.body_deform(points, parameters)
    bindings = _bindings(obj, template)
    frames = {root: _frame(binding, points) for root, binding in bindings.items()}
    frames = {root: frame for root, frame in frames.items() if frame is not None}
    digits = [root for root in frames if '_01_' in root
              and root.replace('_01_', '_03_') in bindings[root][2]]
    feet = [root for root in ('foot_l', 'foot_r') if root in frames]
    handled = set()
    if len(digits) == 10:
        handled.add('fingers')
    if len(feet) == 2:
        handled.update(('feet', 'ankles'))
    arms = [root for root in ('upperarm_l', 'upperarm_r') if root in frames
            and root.replace('upperarm_', 'lowerarm_') in bindings[root][2]]
    if len(arms) == 2:
        handled.update(('biceps', 'triceps'))
    if 'pelvis' in frames and all(name in bindings['pelvis'][2] for name in
                                 ('spine_01', 'spine_02', 'spine_03', 'spine_04', 'spine_05', 'neck_01')):
        handled.update(('back', 'abdomen', 'glutes'))
    if 'spine_01' in frames and all(name in bindings['spine_01'][2]
                                  for name in (*body_posture.SPINE, 'pelvis', 'neck_01')):
        handled.update(('posture', 'asymmetry'))
        if {'biceps', 'triceps', 'abdomen', 'glutes'}.issubset(handled):
            handled.add('age')
    if all(root in frames for root in ('pelvis', 'spine_01', 'clavicle_l', 'clavicle_r')) and all(
            name in bindings['spine_01'][2] for name in ('pelvis', 'neck_01', 'upperarm_l', 'upperarm_r')):
        handled.update(('traps', 'clavicles'))
    if 'age' in requested and 'age' in handled:
        # Age is a relative maturity blend, not a chronological-age estimate.
        # Keep existing artist controls additive, using their anchored tissue
        # fields instead of moving a broad band of vertices in world Z.
        parameters = dict(parameters)
        for name, amount in (('abdomen', .18), ('glutes', -.12), ('biceps', -.12), ('triceps', -.12)):
            parameters[name] = float(np.clip(parameters.get(name, 0) + amount * parameters['age'], -1, 1))
            requested.add(name)
    handled.intersection_update(requested)
    remaining = {key: value for key, value in parameters.items() if key not in handled}
    output = anatomy.body_deform(points, remaining)
    # Refit after the other regional controls, so joint anchors follow this pass.
    frames = {root: _frame(binding, output) for root, binding in bindings.items() if root in frames}
    delta = np.zeros_like(output)
    if handled.intersection(('back', 'abdomen', 'glutes')):
        joint = frames['pelvis']
        if joint is None:
            raise ValueError('Torso shaping lost its reference frame; reset the current body pass')
        pelvis, neck = joint('pelvis'), joint('neck_01')
        up = neck - pelvis
        length = np.linalg.norm(up)
        if length < 1e-8:
            raise ValueError('Torso shaping requires distinct pelvis and neck joints')
        up /= length
        directions = {'X': (1, 0, 0), '-X': (-1, 0, 0),
                      'Y': (0, 1, 0), '-Y': (0, -1, 0)}
        forward = joint.direction(directions.get(template.get('cm_front_axis', '-Y'), (0, -1, 0)))
        forward -= np.dot(forward, up) * up
        magnitude = np.linalg.norm(forward)
        if magnitude < 1e-8:
            raise ValueError('Torso shaping cannot determine the front of this reference')
        forward /= magnitude
        regions = (
            ('back', (joint('spine_03') + joint('spine_05')) * .5,
             abs(np.dot(neck - joint('spine_02'), up)) * .55, -1, .22),
            ('abdomen', (joint('spine_01') + joint('spine_03')) * .5,
             abs(np.dot(joint('spine_04') - pelvis, up)) * .60, 1, .28),
            ('glutes', pelvis - up * length * .08, length * .22, -1, .32),
        )
        weight = bindings['pelvis'][0] * _torso_neck_fade(template, output, up, length, neck)
        for name, center, spread, sign, amount in regions:
            if name not in handled:
                continue
            if spread < 1e-8:
                raise ValueError('Torso shaping requires distinct reference spine joints')
            offset = output - center
            along = offset @ up
            radial = offset - along[:, None] * up
            depth = radial @ forward
            cosine = depth / np.maximum(np.linalg.norm(radial, axis=1), 1e-8)
            side = np.clip((sign * cosine + .05) / .85, 0, 1)
            side = side * side * (3 - 2 * side)
            belly = np.clip(1 - (along / spread) ** 2, 0, 1) ** 2
            # Depth-only shaping leaves the limb axes and opposite torso side
            # in place, with smooth regional and skin-weight transitions.
            strength = amount * parameters[name] * weight * belly * side * depth
            delta += strength[:, None] * forward
    if handled.intersection(('biceps', 'triceps')):
        directions = {'X': (1, 0, 0), '-X': (-1, 0, 0),
                      'Y': (0, 1, 0), '-Y': (0, -1, 0)}
        forward = directions.get(template.get('cm_front_axis', '-Y'), (0, -1, 0))
        for root in arms:
            joint = frames[root]
            if joint is None:
                raise ValueError('Upper-arm shaping lost its reference frame; reset the current body pass')
            pivot = joint(root)
            axis = joint(root.replace('upperarm_', 'lowerarm_')) - pivot
            length = np.linalg.norm(axis)
            if length < 1e-8:
                raise ValueError('Upper-arm shaping requires distinct shoulder and elbow joints')
            axis /= length
            anterior = joint.direction(forward)
            anterior -= np.dot(anterior, axis) * axis
            if np.linalg.norm(anterior) < 1e-8:
                raise ValueError('Upper-arm shaping cannot determine the front of this reference arm')
            anterior /= np.linalg.norm(anterior)
            offset = output - pivot
            along = offset @ axis
            radial = offset - along[:, None] * axis
            angle = (radial @ anterior) / np.maximum(np.linalg.norm(radial, axis=1), 1e-8)
            t = np.clip(along / length, 0, 1)
            # A smooth belly fades to zero at both joints. The skin weights
            # blend between arm and torso without a sharp spatial boundary.
            belly = np.sin(np.pi * t) ** 2 * bindings[root][0]
            for name, sign in (('biceps', 1), ('triceps', -1)):
                if name not in handled:
                    continue
                side = np.clip((sign * angle + .15) / .9, 0, 1)
                side = side * side * (3 - 2 * side)
                delta += radial * (.30 * parameters.get(name, 0) * belly * side)[:, None]
    if 'fingers' in handled:
        for root in digits:
            joint = frames[root]
            if joint is None:
                raise ValueError('Finger shaping lost its reference frame; reset the current body pass')
            pivot = joint(root)
            axis = joint(root.replace('_01_', '_03_')) - pivot
            length = np.linalg.norm(axis)
            if length < 1e-8:
                raise ValueError('Finger shaping requires a nonzero reference digit length')
            axis /= length
            along = np.maximum((output - pivot) @ axis, 0)
            delta += along[:, None] * axis * (.22 * parameters.get('fingers', 0) * bindings[root][0][:, None])
    if handled.intersection(('feet', 'ankles')):
        for root in feet:
            joint = frames[root]
            if joint is None:
                raise ValueError('Foot shaping lost its reference frame; reset the current body pass')
            pivot = joint(root)
            offset = output - pivot
            weight = bindings[root][0]
            # Use the fitted reference ground plane. A rotated character must
            # not grow along world XY or use world Z to locate the ankle band.
            up = np.cross(joint.direction((1, 0, 0)), joint.direction((0, 1, 0)))
            magnitude = np.linalg.norm(up)
            if magnitude < 1e-8:
                raise ValueError('Foot shaping requires a non-degenerate reference ground plane')
            up /= magnitude
            along = offset @ up
            radial = offset - along[:, None] * up
            delta += radial * (.22 * parameters.get('feet', 0) * weight[:, None])
            # Ankle girth uses a short radial band around the ankle, including
            # the calf transition, without widening the toes or moving the sole.
            height = max(np.ptp(output @ up), 1e-8)
            band = np.exp(-.5 * (along / (height * .022)) ** 2)
            radius = np.linalg.norm(radial, axis=1)
            band *= np.exp(-.5 * (radius / (height * .045)) ** 4)
            delta += radial * (.18 * parameters.get('ankles', 0) * band[:, None])
    output += delta
    shoulder_field = None
    shoulder_up = None
    before_shoulders = output
    if handled.intersection(('traps', 'clavicles')):
        joint = _frame(bindings['spine_01'], output)
        if joint is None:
            raise ValueError('Shoulder shaping lost its reference frame; reset the current body pass')
        weight = np.clip(sum(bindings[root][0] for root in ('pelvis', 'clavicle_l', 'clavicle_r')), 0, 1)
        shoulder_up = joint('neck_01') - joint('pelvis')
        shoulder_up /= np.linalg.norm(shoulder_up)
        shoulder_field = body_shoulders.field(joint,
                                             {key: parameters[key] for key in handled.intersection(('traps', 'clavicles'))},
                                             template.get('cm_front_axis', '-Y'))
        output = output + shoulder_field(output) * weight[:, None]
    upper_transform = np.eye(4)
    if handled.intersection(('posture', 'age', 'asymmetry')):
        # Fit the spine before shoulder bulges so skinning does not reinterpret
        # local collar volume as a changed skeletal frame.
        joint = _frame(bindings['spine_01'], before_shoulders)
        if joint is None:
            raise ValueError('Spine shaping lost its reference frame; reset the current body pass')
        skin = body_posture.weights(obj, bindings['spine_01'][2])
        matrices = body_posture.transforms(joint, parameters.get('posture', 0) if 'posture' in handled else 0,
                                           template.get('cm_front_axis', '-Y'),
                                           age=parameters.get('age', 0) if 'age' in handled else 0,
                                           asymmetry=parameters.get('asymmetry', 0) if 'asymmetry' in handled else 0)
        output = body_posture.apply(output, skin, matrices)
        if shoulder_field is not None and follow is not None:
            before_shoulders = body_posture.apply(before_shoulders, skin, matrices)
        upper_transform = matrices[-1]
        if follow is not None:
            up = joint('neck_01') - joint('pelvis')
            follow['up'] = upper_transform[:3, :3] @ (up / np.linalg.norm(up))
    if shoulder_field is not None and follow is not None:
        follow['shoulder_delta'] = output - before_shoulders
        follow['up'] = upper_transform[:3, :3] @ shoulder_up
        inverse = np.linalg.inv(upper_transform)

        def head_delta(points):
            unposed = points @ inverse[:3, :3].T + inverse[:3, 3]
            return shoulder_field(unposed) @ upper_transform[:3, :3].T
        follow['head_delta'] = head_delta
    return output
