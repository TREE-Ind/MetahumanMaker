"""Connected head/body registration with scatter back to untouched part topology."""

import json
from types import SimpleNamespace

import bpy
import numpy as np

from . import anatomy, geometry
from . import mesh_ops as m
from .workflow import group_weights, neck_ring, write_group


def morph_parts(scene):
    s, f = scene.cm_settings, scene.cm_features
    objects = [s.result]
    if s.part == 'FULL_BODY' and not (s.source and s.source.get('cm_gnm')):
        objects.append(f.body_result)
    return [obj for obj in dict.fromkeys(objects) if obj and obj.type=='MESH' and obj.data.shape_keys and 'Wrapped' in obj.data.shape_keys.key_blocks]


def morph_get(features):
    objects = morph_parts(features.id_data)
    return objects[0].data.shape_keys.key_blocks['Wrapped'].value if objects else 1.


def morph_set(features, value):
    scene = features.id_data
    if features.body_editing or features.full_sculpt or features.restore_editing or features.expression_active:
        scene.cm_settings.status = 'Finish the active edit or return to Neutral before changing the wrap morph.'
        return
    value = float(np.clip(value, 0, 1))
    if not np.isfinite(value):
        return
    objects = morph_parts(scene)
    keys = [(obj.data.shape_keys.key_blocks['Wrapped'], obj.data.shape_keys.key_blocks['Wrapped'].value) for obj in objects]
    try:
        for key, _previous in keys:
            key.value = value
        if features.full_head and any((features.eye_left, features.eye_right, features.teeth)):
            from . import workflow
            with bpy.context.temp_override(scene=scene, view_layer=scene.view_layers[0]):
                workflow.update_accessories(bpy.context, scene.cm_settings, features)
    except (ValueError, RuntimeError) as exc:
        for key, previous in keys:
            key.value = previous
        scene.cm_settings.status = 'Wrap morph was retained: '+str(exc)


def physical(points):
    tolerance = max(np.linalg.norm(np.ptp(points, axis=0))*1e-6, 1e-9)
    _keys, inverse = np.unique(np.rint(points/tolerance).astype(np.int64), axis=0, return_inverse=True)
    return inverse


def build(context, s, f):
    """Build only temporary solver geometry; preserve all original UV copies."""
    head, body = s.template, f.body_template
    m.check_inputs(s)
    m.check_mesh(body, 'Body template')
    if head == body or body == s.source:
        raise ValueError('Choose separate head, body and source meshes')
    if body.modifiers or len(body.data.vertices)>250000:
        raise ValueError('Prepare a static base-resolution body template before wrapping')
    h, b = m.coordinates(head), m.coordinates(body)
    hi, bi = anatomy.match_neck(h, neck_ring(head), b, neck_ring(body, False))
    span = float(np.linalg.norm(np.ptp(np.vstack((h, b)), axis=0)))
    neck_gap = float(np.linalg.norm(h[hi]-b[bi], axis=1).max())
    if neck_gap > max(span*.02, 1e-5):
        raise ValueError('Head/body reference necks are too far apart. Align compatible templates before a connected wrap')
    hm, bm = physical(h), physical(b)
    head_count = int(hm.max())+1
    raw = np.concatenate((hm, bm+head_count))
    parent = np.arange(int(raw.max())+1)
    parent[bm[bi]+head_count] = hm[hi]
    _ids, mapping = np.unique(parent[raw], return_inverse=True)
    points = np.vstack((h, b))
    count = int(mapping.max())+1
    counts = np.bincount(mapping, minlength=count)
    rest = np.column_stack([np.bincount(mapping, weights=points[:, axis], minlength=count)/counts for axis in range(3)])
    pin = np.concatenate((m.pin_weights(head), m.pin_weights(body)))
    # A single exact pin chooses the reference of its physical vertex family.
    # Conflicting exact seam pins cannot satisfy continuity; refuse before solve.
    fixed = pin >= 1-1e-7
    fixed_count = np.bincount(mapping[fixed], minlength=count)
    if fixed.any():
        centers = np.column_stack([np.bincount(mapping[fixed], weights=points[fixed, axis], minlength=count)/np.maximum(fixed_count, 1)
            for axis in range(3)])
        if np.max(np.linalg.norm(points[fixed]-centers[mapping[fixed]], axis=1)) > max(span*1e-7, 1e-8):
            raise ValueError('Exact pins disagree at a shared neck/seam vertex. Align the templates or remove conflicting pins')
        rest[fixed_count>0] = centers[fixed_count>0]
    faces = []
    skipped_faces = 0
    for obj, offset in ((head, 0), (body, len(h))):
        for polygon in obj.data.polygons:
            face = mapping[np.asarray(polygon.vertices, dtype=int)+offset].tolist()
            if len(set(face)) < 3:
                skipped_faces += 1
                continue
            faces.append(face)
    if not faces:
        raise ValueError('Templates have no usable physical faces')
    mesh = bpy.data.meshes.new('CM Connected Wrap Graph')
    mesh.from_pydata(rest, [], faces)
    mesh.update()
    obj = bpy.data.objects.new('CM Connected Wrap Graph', mesh)
    context.scene.collection.objects.link(obj)
    try:
        for name, values in (
                ('CM_Pin', pin),
                ('CM_Inner_Parts', np.concatenate((group_weights(head, 'CM_Inner_Parts'), group_weights(body, 'CM_Inner_Parts'))))):
            combined = np.zeros(count)
            np.maximum.at(combined, mapping, values)
            write_group(obj, name, combined)
        ear_groups = {group.index for group in head.vertex_groups
                      if group.name.startswith('FACIAL_') and '_Ear' in group.name}
        ear_weights = np.array([min(1.,sum(item.weight for item in vertex.groups if item.group in ear_groups))
                                for vertex in head.data.vertices])
        if head.vertex_groups.get('CM_Ears'):
            ear_weights = group_weights(head,'CM_Ears')
        if ear_weights.any():
            combined = np.zeros(count)
            np.maximum.at(combined,mapping[:len(h)],ear_weights)
            write_group(obj,'CM_Ears',combined)
        # Preserve paired patch labels on the physical connected solver graph.
        # Without this, full-body wrapping silently discards body constraints.
        for name in sorted({g.name for part in (head, body) for g in part.vertex_groups
                            if g.name.startswith('CM_Fit_')}):
            values = np.concatenate((group_weights(head,name),group_weights(body,name)))
            combined = np.zeros(count)
            np.maximum.at(combined,mapping,values)
            write_group(obj,name,combined)
        if head.vertex_groups.get('CM_WrapMask') or body.vertex_groups.get('CM_WrapMask'):
            values = np.concatenate((group_weights(head, 'CM_WrapMask', default=1), group_weights(body, 'CM_WrapMask', default=1)))
            combined = np.ones(count)
            np.minimum.at(combined, mapping, values)
            write_group(obj, 'CM_WrapMask', combined)
        limits = np.full(count, np.inf)
        local = np.concatenate((np.full(len(h), np.linalg.norm(np.ptp(h, axis=0))*s.max_distance),
            np.full(len(b), np.linalg.norm(np.ptp(b, axis=0))*s.max_distance)))
        np.minimum.at(limits, mapping, local)
        report = {'solver': 'CONNECTED_V1', 'landmarks': 0, 'neck_points': len(hi),
            'reference_neck_gap': neck_gap, 'solver_vertices': count,
            'head_vertices': len(h), 'body_vertices': len(b), 'skipped_degenerate_solver_faces': skipped_faces}
        return obj, mapping, hi, bi, limits, report
    except Exception:
        bpy.data.objects.remove(obj, do_unlink=True)
        bpy.data.meshes.remove(mesh)
        raise


def solve(context, s, f):
    from .guided import signature
    controls = list(s.landmarks)
    for pair in controls:
        if not pair.source or not pair.template:
            raise ValueError('Complete each landmark pair before wrapping')
        if pair.source.parent != s.source or pair.template.parent not in (s.template, f.body_template):
            raise ValueError('Landmarks belong to another source/template')
    if controls:
        if not 3 <= len(controls) <= 512:
            raise ValueError('Use at least three spread landmark pairs across the whole character (maximum 512)')
        geometry.similarity(np.array([pair.template.matrix_world.translation[:] for pair in controls]),
            np.array([pair.source.matrix_world.translation[:] for pair in controls]))
    proxies = []
    graph = result = None
    source_render = s.source.hide_render
    try:
        with m.selection_state(context):
            graph, mapping, hi, bi, limits, report = build(context, s, f)
            pairs = []
            for pair in controls:
                marker = bpy.data.objects.new('CM Joint Landmark', None)
                context.scene.collection.objects.link(marker)
                marker.parent = graph
                marker.location = pair.template.matrix_world.translation
                marker.hide_set(True)
                proxies.append(marker)
                pairs.append(SimpleNamespace(source=pair.source, template=marker, name=pair.name))
            context.view_layer.update()
            settings = SimpleNamespace(**{name: getattr(s, name) for name in (
                'character_name', 'iterations', 'projection_strength', 'smoothing', 'landmark_radius', 'landmark_regularization', 'max_distance')})
            settings.source, settings.template = s.source, graph
            settings.part, settings.result, settings.landmarks = 'FULL_BODY', None, pairs
            face_pairs = [pair for pair in pairs if pair.name.startswith('FACE_')]
            pose_pairs = [pair for pair in pairs if pair.name.startswith('POSE_')]
            if len(face_pairs) >= 3 and len(pose_pairs) >= 3:
                settings.landmarks = [pair for pair in pairs if pair not in face_pairs]
                head = m.coordinates(s.template)
                graph_points = m.coordinates(graph)
                height = float(np.ptp(head[:, 2]))
                neck_top = float(head[hi, 2].max())
                settings.local_landmarks = (
                    np.array([pair.template.matrix_world.translation[:] for pair in face_pairs]),
                    np.array([pair.source.matrix_world.translation[:] for pair in face_pairs]))
                settings.local_landmark_radius = max(height * s.landmark_radius, height * .1)
                settings.local_landmark_weights = np.clip(
                    (graph_points[:, 2] - neck_top) / max(height * .25, 1e-8), 0, 1)
                report['facial_landmarks'] = len(face_pairs)
            settings.projection_limits = limits
            if getattr(f, 'preserve_mouth_interior', True):
                from .wrap_interiors import mouth_mask
                interior = mouth_mask(s.template)
                mask = np.zeros(len(graph.data.vertices), dtype=bool)
                np.logical_or.at(mask, mapping[:len(interior)], interior)
                settings.mouth_interior = mask
                region = np.zeros(len(graph.data.vertices), dtype=bool)
                region[mapping[:len(interior)]] = True
                head_points = m.coordinates(s.template)
                neck_top = float(head_points[hi, 2].max())
                region &= m.coordinates(graph)[:, 2] > neck_top + float(np.ptp(head_points[:, 2])) * .1
                settings.mouth_guard_region = region
                report['preserved_mouth_vertices'] = int(interior.sum())
            report['landmarks'] = len(pairs)
            result = m.wrap(context, settings)
            if result.get('cm_source_ear_regions'):
                report['source_ear_regions'] = int(result['cm_source_ear_regions'])
            if result.get('cm_region_projection'):
                report['region_projection'] = json.loads(result['cm_region_projection'])
            if result.get('cm_conservative_surface'):
                report['conservative_surface'] = json.loads(result['cm_conservative_surface'])
            positions = m.evaluated_coordinates(context, result)
            output = positions[mapping]
            split = len(s.template.data.vertices)
            if not np.isfinite(output).all():
                raise ValueError('Joint solver returned invalid coordinates; previous parts retained')
            gap = float(np.linalg.norm(output[hi]-output[split+bi], axis=1).max())
            if gap > max(np.linalg.norm(np.ptp(output, axis=0))*1e-7, 1e-7):
                raise ValueError('Connected neck correspondence failed; previous parts retained')
            report['output_neck_gap'] = gap
            report['surface_rms'] = settings.last_error
            return output[:split], output[split:], report, result.get('cm_source_signature') or signature(s.source)
    finally:
        s.source.hide_render = source_render
        for marker in proxies:
            bpy.data.objects.remove(marker, do_unlink=True)
        for obj in (result, graph):
            if obj:
                data = obj.data
                bpy.data.objects.remove(obj, do_unlink=True)
                if not data.users:
                    bpy.data.meshes.remove(data)


def part(context, template, vertices, name, role, source_signature, report):
    from .guided import signature
    obj = m.duplicate(context, template, name)
    try:
        obj.shape_key_add(name='Basis')
        key = obj.shape_key_add(name='Wrapped')
        key.data.foreach_set('co', m.to_local(obj, vertices).astype(np.float32).ravel())
        key.value = 1
        obj['cm_template_topology'] = m.topology(template)
        if m.topology(obj) != obj['cm_template_topology']:
            raise ValueError('Part topology changed; previous character retained')
        obj['cm_template_signature'] = signature(template)
        obj['cm_source_signature'] = source_signature
        obj['cm_role'], obj['cm_part'] = 'RESULT', role
        if 'conservative_surface' in report:
            obj['cm_conservative_surface'] = json.dumps(report['conservative_surface'])
        elif 'cm_conservative_surface' in obj:
            del obj['cm_conservative_surface']
        obj['cm_joint_wrap'] = json.dumps(report)
        if 'region_projection' in report:
            obj['cm_region_projection'] = json.dumps(report['region_projection'])
        if role == 'BODY':
            obj['cm_body_source'] = 'SCAN'
        else:
            obj['cm_neck_attached'] = True
        return obj
    except Exception:
        data = obj.data
        bpy.data.objects.remove(obj, do_unlink=True)
        if not data.users:
            bpy.data.meshes.remove(data)
        raise


def wrap(context, s, f):
    from . import workflow
    m.check_inputs(s)
    if context.mode != 'OBJECT' or f.body_editing or f.full_sculpt or f.restore_editing or f.expression_active:
        raise ValueError('Finish the current edit or expression before wrapping a full character')
    old = s.result, f.head_result, f.body_result, s.stage, s.last_error
    actors = list(dict.fromkeys(obj for obj in (s.source, s.template, f.body_template, *old[:3], f.eye_left, f.eye_right, f.teeth) if obj))
    visibility = [(obj, obj.hide_get(), obj.hide_render) for obj in actors]
    accessory_state = [(obj, obj.matrix_world.copy(), {key: list(obj[key]) if key=='cm_follow_original' and key in obj else obj.get(key)
        for key in ('cm_follow_original', 'cm_follow_anchors')})
        for obj in (f.eye_left, f.eye_right, f.teeth) if obj]
    made = []
    try:
        head, body, report, source_signature = solve(context, s, f)
        new_head = part(context, s.template, head, s.character_name+'_Head', 'HEAD', source_signature, report)
        made.append(new_head)
        new_body = part(context, f.body_template, body, s.character_name+'_Body', 'BODY', source_signature, report)
        made.append(new_body)
        s.result, f.head_result, f.body_result = new_head, new_head, new_body
        if f.full_head and accessory_state:
            workflow.update_accessories(context, s, f)
        for obj in (s.source, s.template, f.body_template, *old[:3]):
            if obj and obj not in made:
                obj.hide_set(True);obj.hide_render = True
        for obj in made:
            obj.hide_set(False);obj.hide_render = False
        s.stage, s.last_error = 'REFINE', report['surface_rms']
        m.activate(context, new_head)
        s.status = 'Full character fitted together with '+str(report['neck_points'])+' connected neck points; original head/body topology and UVs retained.'
    except Exception:
        s.result, f.head_result, f.body_result, s.stage, s.last_error = old
        for obj, matrix, properties in accessory_state:
            obj.matrix_world = matrix
            for key, value in properties.items():
                if value is None:
                    if key in obj:del obj[key]
                else:
                    obj[key] = value
        for obj, hidden, render in visibility:
            obj.hide_set(hidden);obj.hide_render = render
        for obj in made:
            data = obj.data
            bpy.data.objects.remove(obj, do_unlink=True)
            if not data.users:bpy.data.meshes.remove(data)
        raise
