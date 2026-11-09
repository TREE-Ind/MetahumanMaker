"""Editable native raw-control combinations for artist and agent review."""
from contextlib import contextmanager
import math
import re

import bpy
from bpy.props import EnumProperty, FloatProperty, StringProperty

PREFIX = 'CTRL_expressions.'
REGIONS = [('ALL', 'All', 'All native controls'),
           ('EYES', 'Eyes', 'Eyelids and gaze'), ('BROWS', 'Brows', 'Brow controls'),
           ('MOUTH', 'Mouth', 'Lips and mouth'), ('JAW', 'Jaw', 'Jaw controls'),
           ('NOSE', 'Nose', 'Nose controls'), ('CHEEKS', 'Cheeks', 'Cheek controls'),
           ('TONGUE', 'Tongue', 'Tongue controls'), ('OTHER', 'Other', 'Other native controls')]
PAGE_SIZE = 12


@contextmanager
def syncing():
    from . import native_face_review as review
    previous = review._SYNCING
    review._SYNCING = True
    try:
        yield
    finally:
        review._SYNCING = previous


def region(name):
    name = name.removeprefix(PREFIX).lower()
    # Native cheek controls include mouthCheekBlow/Suck.
    if 'cheek' in name: return 'CHEEKS'
    for prefix, group in [('eye', 'EYES'), ('brow', 'BROWS'), ('mouth', 'MOUTH'),
                          ('jaw', 'JAW'), ('nose', 'NOSE'), ('tongue', 'TONGUE')]:
        if name.startswith(prefix): return group
    return 'OTHER'


def label(name):
    words = re.sub(r'(?<=[a-z0-9])(?=[A-Z])', ' ', name.removeprefix(PREFIX))
    return words[:1].upper() + words[1:]


def opposite(name):
    return name[:-1] + ('R' if name.endswith('L') else 'L') if name.endswith(('L', 'R')) else None


def ensure(settings):
    """Bind the draft to this review session, preserving saved drafts on reopen."""
    from .native_face_review import result
    state = result(settings)
    names = state.get('raw_controls', [])
    if not names or len(names) != len(set(names)) or not all(isinstance(name, str) and name.startswith(PREFIX) for name in names):
        raise ValueError('Connect an Unreal character with native facial controls first')
    owner = state.get('origin', '')
    if not owner: raise ValueError('Connect an Unreal facial review session first')
    if settings.mixer_owner == owner and [item.name for item in settings.mixer_controls] == names:
        return
    initial = state.get('controls', {})
    with syncing():
        settings.mixer_controls.clear()
        for name in names:
            item = settings.mixer_controls.add()
            item.name = name
            item.value = float(initial.get(name, initial.get(name.removeprefix(PREFIX), 0.)))
        settings.mixer_owner = owner
        settings.mixer_page = 0


def values(settings):
    ensure(settings)
    return {item.name: float(item.value) for item in settings.mixer_controls if item.value != 0.}


def set_values(settings, controls):
    """Atomically replace a draft. Accept exact raw names or unprefixed aliases."""
    ensure(settings)
    if not isinstance(controls, dict): raise ValueError('Facial controls must be an object')
    names = {item.name for item in settings.mixer_controls}
    prepared = {}
    for name, value in controls.items():
        if not isinstance(name, str): raise ValueError('Facial control names must be strings')
        key = name if name.startswith(PREFIX) else PREFIX + name
        if key not in names or key in prepared or type(value) not in (int, float) or not math.isfinite(value) or not 0. <= value <= 1.:
            raise ValueError('Use distinct supported native controls with finite values from 0 to 1')
        prepared[key] = value
    with syncing():
        for item in settings.mixer_controls: item.value = prepared.get(item.name, 0.)


def update_value(self, context):
    from . import native_face_review as review
    if review._SYNCING or not context: return
    scene = self.id_data
    settings = scene.cm_face_review
    if settings.mixer_pair:
        other = settings.mixer_controls.get(opposite(self.name) or '')
        if other:
            with syncing(): other.value = self.value
    if settings.input_mode == 'MIXER': review.queue_apply(settings, context)


def update_mode(self, context):
    from . import native_face_review as review
    previous = self.get('cm_previous_input_mode', 'PRESET')
    self['cm_previous_input_mode'] = self.input_mode
    if review._SYNCING: return
    try:
        if self.input_mode == 'MIXER':
            # Transfer the currently selected draft, including unapplied edits.
            selected = review.controls(self, mode=previous) if previous != 'MIXER' else values(self)
            set_values(self, selected)
        review.queue_apply(self, context)
    except (ValueError, KeyError, IndexError) as error:
        self.status = str(error)


def visible_rows(settings):
    """Filter pairs together so active-only never hides an active opposite side."""
    items = {item.name: item for item in settings.mixer_controls}
    seen = set()
    rows = []
    words = settings.mixer_search.casefold().split()
    for name, item in items.items():
        if name in seen: continue
        pair = opposite(name)
        row = sorted([item, items[pair]], key=lambda value: value.name) if pair in items else [item]
        seen.update(value.name for value in row)
        if settings.mixer_region != 'ALL' and region(name) != settings.mixer_region: continue
        if settings.mixer_active_only and not any(value.value for value in row): continue
        if words and not all(word in ' '.join((label(value.name)+' '+value.name).casefold() for value in row) for word in words): continue
        rows.append(row)
    return rows


class CM_FaceMixerControl(bpy.types.PropertyGroup):
    value: FloatProperty(name='Strength', default=0., min=0., max=1., precision=3, update=update_value)


class CM_OT_native_face_mixer(bpy.types.Operator):
    bl_idname = 'cm.native_face_mixer'
    bl_label = 'Edit Native Face Mix'
    action: EnumProperty(items=[('RESET_ALL', 'Neutral', 'Clear every native face control'),
                                ('RESET_VISIBLE', 'Reset Shown', 'Clear the current filtered controls on every page'),
                                ('PREVIOUS', 'Previous', 'Previous page of controls'),
                                ('NEXT', 'Next', 'Next page of controls')])

    def execute(self, context):
        from . import native_face_review as review
        settings = context.scene.cm_face_review
        try:
            ensure(settings)
            rows = visible_rows(settings)
            if self.action in {'NEXT', 'PREVIOUS'}:
                current = min(settings.mixer_page, max(0, (len(rows)-1)//PAGE_SIZE))
                settings.mixer_page = max(0, min(max(0, (len(rows)-1)//PAGE_SIZE), current + (1 if self.action == 'NEXT' else -1)))
                return {'FINISHED'}
            selected = list(settings.mixer_controls) if self.action == 'RESET_ALL' else [item for row in rows for item in row]
            with syncing():
                for item in selected: item.value = 0.
            if settings.input_mode == 'MIXER': review.queue_apply(settings, context)
            settings.status = 'Neutral face mix.' if self.action == 'RESET_ALL' else 'Filtered face controls reset; other controls retained.'
        except (ValueError, RuntimeError) as error:
            settings.status = str(error); self.report({'ERROR'}, str(error)); return {'CANCELLED'}
        return {'FINISHED'}


def draw(layout, settings):
    layout = layout.column()
    layout.prop(settings, 'mixer_region', text='Region')
    layout.prop(settings, 'mixer_search', text='', icon='VIEWZOOM')
    row = layout.row(); row.prop(settings, 'mixer_pair'); row.prop(settings, 'mixer_active_only')
    rows = visible_rows(settings)
    page = min(settings.mixer_page, max(0, (len(rows)-1)//PAGE_SIZE))
    for items in rows[page*PAGE_SIZE:(page+1)*PAGE_SIZE]:
        row = layout.row(align=True)
        if len(items) == 2:
            row.label(text=label(items[0].name[:-1]))
            for item in items: row.prop(item, 'value', text=item.name[-1], slider=True)
        else: row.prop(items[0], 'value', text=label(items[0].name), slider=True)
    if not rows: layout.label(text='No matching controls.')
    if len(rows) > PAGE_SIZE:
        row = layout.row(align=True)
        previous = row.row(); previous.enabled = page > 0; previous.operator('cm.native_face_mixer', text='', icon='TRIA_LEFT').action = 'PREVIOUS'
        row.label(text=f'{page+1} / {(len(rows)-1)//PAGE_SIZE+1}')
        following = row.row(); following.enabled = (page+1)*PAGE_SIZE < len(rows); following.operator('cm.native_face_mixer', text='', icon='TRIA_RIGHT').action = 'NEXT'
    row = layout.row(align=True)
    row.operator('cm.native_face_mixer', text='Neutral', icon='LOOP_BACK').action = 'RESET_ALL'
    row.operator('cm.native_face_mixer', text='Reset Shown').action = 'RESET_VISIBLE'
    layout.label(text=f'{sum(item.value != 0. for item in settings.mixer_controls)} active / {len(settings.mixer_controls)} native controls')
