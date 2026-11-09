"""Recover incomplete screens only after Blender activates the healthy donor."""
import time

import bpy
from bpy.app.handlers import persistent

_JOBS = {}


def coverage(screen):
    areas = list(screen.areas)
    if not areas:
        return 0
    width = max(a.x + a.width for a in areas) - min(a.x for a in areas)
    height = max(a.y + a.height for a in areas) - min(a.y for a in areas)
    return sum(a.width * a.height for a in areas) / max(width * height, 1)


def start(context, callback):
    if bpy.app.background or not context.window or context.mode != 'OBJECT':
        return False
    window = context.window
    key = window.as_pointer()
    if key in _JOBS:
        _JOBS[key]['callback'] = callback
        return True
    if coverage(window.screen) > .95:
        return False
    candidates = []
    for item in bpy.data.workspaces:
        if item == window.workspace or getattr(item, 'use_pin_scene', False):
            continue
        for screen in item.screens:
            views = [area for area in screen.areas if area.type == 'VIEW_3D']
            if len(views) == 1 and coverage(screen) > .98:
                candidates.append(item)
                break
    if not candidates:
        context.scene.cm_settings.status = 'No complete 3D workspace is available for layout recovery'
        return True
    candidates.sort(key=lambda item: (not bool(item.get('cm_working_layout')), item.name != 'Modeling', item.name))
    donor = candidates[0]
    old_view = next((area.spaces.active for area in window.screen.areas if area.type == 'VIEW_3D'), None)
    shading = {name: getattr(old_view.shading, name) for name in ('type', 'use_scene_world', 'use_scene_lights')} if old_view else {}
    job = {'window': window, 'original': window.workspace, 'scene': window.scene, 'layer': window.view_layer,
           'donor': donor, 'target': donor if donor.get('cm_working_layout') else None,
           'remaining_donors':candidates[1:], 'incomplete':None,
           'phase': 'ACTIVATE',
           'shading': shading, 'callback': callback, 'deadline': time.monotonic() + 10, 'stable': None}
    _JOBS[key] = job
    if not bpy.app.timers.is_registered(poll):
        bpy.app.timers.register(poll, first_interval=.05)
    return True


def poll():
    for key, job in list(_JOBS.items()):
        try:
            window = job['window']
            if window not in list(bpy.context.window_manager.windows):
                _JOBS.pop(key, None)
                continue
            if window.scene != job['scene'] or window.workspace not in {job['original'], job['donor'], job['target']}:
                _JOBS.pop(key, None)  # Respect navigation or file changes during the handoff.
                continue
            if time.monotonic() > job['deadline']:
                raise RuntimeError('Blender did not activate a complete working layout')
            if job['phase'] == 'ACTIVATE':
                # A temp_override restores screen context on exit. Switch only
                # from the standalone timer, after the originating UI/MCP call.
                window.workspace = job['donor']
                job['phase'] = 'TARGET' if job['target'] else 'DONOR'
                continue
            expected = job['donor'] if job['phase'] == 'DONOR' else job['target']
            if window.workspace != expected or window.screen not in list(expected.screens):
                continue
            if coverage(window.screen)<.98:
                # An inactive screen can look complete at its old dimensions,
                # then expose missing areas when activated in this window.
                # Let resizing settle, then try another donor instead of waiting
                # out the deadline on the same damaged layout.
                size=(window.screen.as_pointer(),tuple((a.x,a.y,a.width,a.height) for a in window.screen.areas))
                previous=job.get('incomplete')
                count=previous[1]+1 if previous and previous[0]==size else 1
                job['incomplete']=(size,count)
                if count<3:continue
                if not job['remaining_donors']:
                    raise RuntimeError('No candidate workspace remained complete after activation')
                donor=job['remaining_donors'].pop(0)
                job.update(donor=donor,target=donor if donor.get('cm_working_layout') else None,
                           phase='TARGET' if donor.get('cm_working_layout') else 'DONOR',stable=None,incomplete=None)
                window.workspace=donor
                continue
            job['incomplete']=None
            dimensions = (window.screen.as_pointer(), tuple((a.x, a.y, a.width, a.height) for a in window.screen.areas))
            if job['stable'] != dimensions:
                job['stable'] = dimensions
                continue
            if job['phase'] == 'DONOR':
                if bpy.context.window != window or bpy.context.workspace != expected:
                    continue
                previous = set(bpy.data.workspaces)
                job['donor_screen'] = window.screen.name
                # Use the actual active window, without a screen override that
                # would restore the old layout after this operator activates its copy.
                if bpy.ops.workspace.duplicate() != {'FINISHED'}:
                    raise RuntimeError('Could not duplicate the healthy working layout')
                created = set(bpy.data.workspaces) - previous
                if len(created) != 1:
                    raise RuntimeError('Could not identify the recovered workspace')
                job['target'] = created.pop()
                job['target']['cm_working_layout'] = True
                job['target']['cm_recovery_donor_screen'] = job['donor_screen']
                job['phase'], job['stable'] = 'TARGET', None
                window.workspace = job['target']
                continue
            original, target = job['original'], job['target']
            if original.get('cm_working_layout') and original.name in {'Create MH','MetaHuman Maker'}:
                original.name = 'MetaHuman Maker (Previous Layout)'
            target.name = 'MetaHuman Maker'
            window.scene, window.view_layer = job['scene'], job['layer']
            for area in window.screen.areas:
                if area.type == 'VIEW_3D':
                    area.spaces.active.show_region_ui = True
                    for name, value in job['shading'].items():
                        setattr(area.spaces.active.shading, name, value)
            _JOBS.pop(key, None)
            job['scene'].cm_settings.status = 'Working view recovered.'
            with bpy.context.temp_override(window=window, area=None, region=None):
                job['callback'](bpy.context)
        except (ReferenceError, RuntimeError, ValueError) as error:
            _JOBS.pop(key, None)
            try:
                job['scene'].cm_settings.status = 'Layout recovery did not complete: ' + str(error)
                job['scene'].cm_features.last_action_error = True
            except ReferenceError:
                pass
    return .05 if _JOBS else None


@persistent
def detach(_unused=None):
    _JOBS.clear()
    if bpy.app.timers.is_registered(poll):
        bpy.app.timers.unregister(poll)


def register():
    for handler in (bpy.app.handlers.load_pre, bpy.app.handlers.undo_pre):
        if detach not in handler:
            handler.append(detach)


def unregister():
    detach()
    for handler in (bpy.app.handlers.load_pre, bpy.app.handlers.undo_pre):
        if detach in handler:
            handler.remove(detach)
