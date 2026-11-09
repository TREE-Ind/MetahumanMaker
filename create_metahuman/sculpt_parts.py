"""Direct, additive sculpt access to the accepted character's individual parts."""
import bpy
import numpy as np
from bpy.app.handlers import persistent

from . import mesh_ops as m

ACTIVE = 'cm_individual_sculpt_part'
PARTS = [('HEAD', 'Head', ''), ('EYE_LEFT', 'Left Eye', ''),
         ('EYE_RIGHT', 'Right Eye', ''), ('TEETH', 'Teeth', ''), ('BODY', 'Body', '')]
_WAITING = {}


def available(scene):
    s, f = scene.cm_settings, scene.cm_features
    result = {}
    if s.part != 'BODY' and s.result:
        result['HEAD'] = s.result
        for role in ('EYE_LEFT', 'EYE_RIGHT', 'TEETH'):
            if getattr(f, role.lower()):
                result[role] = getattr(f, role.lower())
    body = f.body_result if s.part == 'FULL_BODY' else s.result if s.part == 'BODY' else None
    if body:
        result['BODY'] = body
    return result


def additive_key(obj, name):
    if not obj.data.shape_keys:
        obj.shape_key_add(name='Basis')
    keys = obj.data.shape_keys
    if not keys.use_relative:
        raise ValueError('Individual sculpting requires relative shape keys')
    cache = 'cm_current_' + name.lower().replace(' ', '_') + '_key'
    key = keys.key_blocks.get(obj.get(cache, '')) or keys.key_blocks.get(name)
    if key and key.lock_shape:
        raise ValueError('Unlock the ' + key.name + ' shape key before sculpting it')
    # Keep an earlier pass at its authored strength. A fresh full-strength pass
    # remains editable even when the artist muted or reduced the old pass.
    if key and (key.mute or abs(key.value - 1) > 1e-6):
        key = None
    if key is None:
        key = obj.shape_key_add(name=name, from_mix=False)
        key.relative_key = keys.key_blocks[0]
        # A zero delta preserves every existing key and its current strength.
        points = np.empty(len(key.data) * 3, dtype=np.float32)
        key.relative_key.data.foreach_get('co', points)
        key.data.foreach_set('co', points)
        key.value = 1
    obj[cache] = key.name
    obj.active_shape_key_index = keys.key_blocks.find(key.name)
    return key


def start(context, s, part='HEAD', *, detail=False):
    from . import (
        bake_jobs,
        compare_views,
        detail_jobs,
        scan_jobs,
        workspace_layout,
        wrinkle_jobs,
    )
    f = context.scene.cm_features
    if context.mode != 'OBJECT' or f.full_sculpt or f.restore_editing:
        raise ValueError('Finish the active sculpt or repair before choosing another part')
    if f.body_editing:
        raise ValueError('Apply or cancel the body pass before sculpting')
    if f.expression_active:
        raise ValueError('Return the expression to Neutral before sculpting individual parts')
    if any(module.running() for module in (bake_jobs, detail_jobs, scan_jobs, wrinkle_jobs)):
        raise ValueError('Finish or cancel the current geometry or map operation before sculpting')
    parts = available(context.scene)
    # Preserve the historical default for body-only scripts calling cm.sculpt().
    target = parts.get('BODY' if s.part == 'BODY' and part == 'HEAD' else part)
    m.check_mesh(target, {key: label for key, label, _ in PARTS}.get(part, 'Character part'))
    if detail and part not in {'HEAD', 'BODY'}:
        raise ValueError('Sculpt detail layers belong to the Head or Body; choose base sculpt for eyes and teeth')
    expression = target.data.shape_keys.key_blocks.get('Expression Check') if target.data.shape_keys else None
    if expression and expression.value:
        raise ValueError('Return Expression Check to Neutral before sculpting this part')
    if target.data.shape_keys and not target.data.shape_keys.use_relative:
        raise ValueError('Individual sculpting requires relative shape keys')
    session = compare_views.recover(context.window) if context.window else None
    if session:
        _WAITING[context.window.as_pointer()] = {'window': context.window, 'scene': context.scene,
                                               'part': part, 'detail': detail, 'target': target,
                                               'settled': False, 'ticks': 100}
        if not bpy.app.timers.is_registered(resume):
            bpy.app.timers.register(resume, first_interval=.1)
        s.status = 'Closing comparison before sculpting the selected part...'
        return
    if workspace_layout.start(context, lambda ready: start(ready, s, part, detail=detail)):
        return
    obj = target
    if detail:
        from . import detail_layers, scan_projection
        obj = f.body_detail if s.part == 'FULL_BODY' and part == 'BODY' else s.detail
        m.check_mesh(obj, 'Sculpt detail')
        scan_projection.require_applied(obj)
        if not bpy.app.background and detail_jobs.start(context, s, f, 'SCULPT', part):
            return
        # Returned sculpt adoption can replace the pointer even when it needs
        # no background transport. Continue on that working layer, not its archive.
        obj = f.body_detail if s.part == 'FULL_BODY' and part == 'BODY' else s.detail
        m.check_mesh(obj, 'Sculpt detail')
        scan_projection.require_applied(obj)
        detail_layers.sync(context, target, obj)
    from .guided import focus
    companions = [obj]
    if part == 'HEAD' and s.part != 'BODY':
        companions += [f.eye_left, f.eye_right, f.teeth]
    elif part == 'BODY' and s.part == 'FULL_BODY':
        companions += [s.result, f.eye_left, f.eye_right, f.teeth]
    focus(context, companions, f.front_axis)
    m.activate(context, obj)
    if detail:
        keys = obj.data.shape_keys.key_blocks if obj.data.shape_keys else None
        key = keys.get('Scan Detail') if keys else None
        additive_key(obj, 'Scan Detail' if key else 'Sculpt Detail')
    else:
        additive_key(obj, 'Sculpt')
        obj[ACTIVE] = 'BODY' if s.part == 'BODY' else part
    bpy.ops.object.mode_set(mode='SCULPT')
    label = next(label for key, label, _ in PARTS if key == ('BODY' if s.part == 'BODY' else part))
    s.status = label + ' sculpt active. Finish Sculpt returns to the character.'


def finish(context, obj):
    if not obj or ACTIVE not in obj:
        return
    part = obj[ACTIVE]
    del obj[ACTIVE]
    if available(context.scene).get(part) != obj:
        return
    from .guided import focus
    parts = available(context.scene)
    focus(context, list(parts.values()), context.scene.cm_features.front_axis)
    context.scene.cm_settings.status = 'Sculpt finished.'


def resume():
    from . import compare_views
    for key, job in list(_WAITING.items()):
        try:
            window, scene = job['window'], job['scene']
            if window not in list(bpy.context.window_manager.windows) or window.scene != scene:
                _WAITING.pop(key, None)
                continue
            part = 'BODY' if scene.cm_settings.part == 'BODY' and job['part'] == 'HEAD' else job['part']
            if available(scene).get(part) != job['target']:
                _WAITING.pop(key, None)
                continue
            job['ticks'] -= 1
            if not job['ticks']:
                raise ValueError('Comparison did not finish closing. Close it before sculpting this part')
            session = compare_views.recover(window)
            if session:
                if not session.get('closing') and compare_views.adjacent(*session['areas']):
                    with bpy.context.temp_override(window=window, area=None, region=None):
                        compare_views.close_views(bpy.context)
                continue
            if not job['settled']:
                job['settled'] = True  # Allow Close's deferred visibility restoration.
                continue
            _WAITING.pop(key, None)
            with bpy.context.temp_override(window=window, area=None, region=None):
                start(bpy.context, scene.cm_settings, job['part'], detail=job['detail'])
        except (ReferenceError, RuntimeError, ValueError) as error:
            _WAITING.pop(key, None)
            try:
                job['scene'].cm_settings.status = str(error)
                job['scene'].cm_features.last_action_error = True
            except ReferenceError:
                pass
    return .1 if _WAITING else None


@persistent
def detach(_unused=None):
    _WAITING.clear()
    if bpy.app.timers.is_registered(resume):
        bpy.app.timers.unregister(resume)


def register():
    for handler in (bpy.app.handlers.load_pre, bpy.app.handlers.undo_pre):
        if detach not in handler:
            handler.append(detach)


def unregister():
    detach()
    for handler in (bpy.app.handlers.load_pre, bpy.app.handlers.undo_pre):
        if detach in handler:
            handler.remove(detach)


class CM_MT_sculpt_parts(bpy.types.Menu):
    bl_label = 'Sculpt Character Part'

    def draw(self, context):
        parts = available(context.scene)
        for key, label, _ in PARTS:
            if key in parts:
                self.layout.operator('cm.sculpt', text=label, icon='SCULPTMODE_HLT').part = key


CLASSES = (CM_MT_sculpt_parts,)
