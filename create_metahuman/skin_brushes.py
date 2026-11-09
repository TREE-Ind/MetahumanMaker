"""Independent procedural sculpt stamps, installed as native Blender brush assets."""

import json
import os
import time
from pathlib import Path

import bpy
import numpy as np

LIBRARY = 'MetaHuman Maker Skin'
CATALOG = 'c4ada637-98d0-40aa-8519-9b4fca7b40c2'
FILENAME = 'create_mh_skin_v1.blend'
_PENDING = None
PRESETS = (
    ('PoresFine', 'Fine Pores', 'Dense, small recessed pores'),
    ('PoresLarge', 'Large Pores', 'Larger recessed pores'),
    ('PoresSparse', 'Sparse Pores', 'Scattered recessed pores'),
    ('CrowFeet', "Crow's Feet", 'Radiating fine lines'),
    ('Forehead', 'Forehead Lines', 'Three softly curved horizontal folds'),
    ('LipLines', 'Lip Lines', 'Short parallel lip creases'),
    ('Scar', 'Scar', 'Irregular raised scar with short branches'),
    ('Crease', 'Crease', 'Single curved crease'),
    ('SkinGrain', 'Skin Grain', 'Fine, softly connected skin grain'),
    ('StretchMarks', 'Stretch Marks', 'Tapered, broken parallel lines'),
    ('WrinkleSoft', 'Soft Wrinkle', 'Broad curved wrinkle'),
    ('WrinkleSharp', 'Sharp Wrinkle', 'Narrow curved wrinkle'),
    ('Freckles', 'Small Bumps', 'Scattered round raised details; this sculpts height, not pigment'),
    ('Crackle', 'Crackle', 'Irregular connected polygonal creases'),
)


def folder():
    return Path(bpy.utils.user_resource('DATAFILES')) / 'create_metahuman' / 'brushes'


def available():
    return (folder() / FILENAME).is_file()


def ensure_library():
    """Also called on registration in factory-startup sculpt author windows."""
    if not available():
        return None
    path = folder().resolve()
    libraries = bpy.context.preferences.filepaths.asset_libraries
    for library in libraries:
        if Path(bpy.path.abspath(library.path)).resolve() == path:
            if library.name == 'Create MH Skin':
                library.name = LIBRARY
            from .blender_compat import asset_import_method
            asset_import_method(library)
            return library
    library = libraries.new(name=LIBRARY, directory=str(path))
    from .blender_compat import asset_import_method
    asset_import_method(library)
    return library


def pattern(identifier, size=256):
    """Height stamps with a soft zero border; no downloaded vendor images."""
    index = next(i for i, p in enumerate(PRESETS) if p[0] == identifier)
    rng = np.random.default_rng(937 + index)
    y, x = np.mgrid[-1:1:complex(size), -1:1:complex(size)]
    stamp = np.zeros_like(x)
    if index in (0, 1, 2, 8, 12):
        count, width = {0: (170, .017), 1: (55, .038), 2: (24, .023),
                        8: (450, .025), 12: (32, .045)}[index]
        for cx, cy in rng.uniform(-1, 1, (count, 2)):
            stamp += rng.uniform(.4, 1) * np.exp(-((x-cx)**2+(y-cy)**2)/width**2)
    elif index == 3:
        for angle in (-.65, -.3, .05, .4, .75):
            along = (x+.65)*np.cos(angle) + y*np.sin(angle)
            across = y*np.cos(angle) - (x+.65)*np.sin(angle)
            stamp += np.exp(-(across/.025)**2) * np.clip(along, 0, 1)
    elif index == 4:
        for offset in (-.42, 0, .42):
            stamp += np.exp(-((y-offset-.1*x*x)/.045)**2)
    elif index == 5:
        for offset in np.linspace(-.8, .8, 13):
            stamp += np.exp(-((x-offset-.025*np.sin(y*9))/.019)**2) * np.exp(-(y/.7)**6)
    elif index == 6:
        stamp = np.exp(-((x-.08*np.sin(y*7))/.045)**2)
        for offset in (-.55, -.2, .15, .5):
            stamp += .5*np.exp(-((y-offset-.4*x)/.025)**2)*np.exp(-(x/.18)**4)
    elif index == 9:
        for offset in (-.6, -.2, .2, .6):
            stamp += np.exp(-((x-offset-.05*np.sin(y*6+offset*9))/.025)**2) * np.clip(1-y*y, 0, 1) * (.65+.35*np.sin(y*15+offset*7))
    elif index == 13:
        seeds = rng.uniform(-1.3, 1.3, (35, 2))
        distances = np.sort(np.stack([(x-cx)**2+(y-cy)**2 for cx, cy in seeds]), axis=0)
        stamp = np.exp(-((distances[1]-distances[0])/.018)**2)
    else:
        width = {7: .03, 10: .1, 11: .016}[index]
        stamp = np.exp(-((x-.15*np.sin(y*3))/width)**2)
    stamp *= np.clip(1-x*x-y*y, 0, 1)**2
    return (stamp / max(float(stamp.max()), 1e-8)).astype(np.float32)


def make_image(identifier, destination):
    values = pattern(identifier)
    image = bpy.data.images.new('CM Skin ' + identifier, width=values.shape[0], height=values.shape[1], alpha=False)
    image.colorspace_settings.name = 'Non-Color'
    rgba = np.ones((*values.shape, 4), dtype=np.float32)
    rgba[:, :, :3] = values[:, :, None]
    image.pixels.foreach_set(rgba.ravel())
    image.filepath_raw = str(destination)
    image.file_format = 'PNG'
    image.save()
    image.pack()
    return image


def install():
    if bpy.context.mode != 'SCULPT':
        raise ValueError('Enter Sculpt Detail or a wrinkle layer before installing skin brushes')
    destination = folder()
    destination.mkdir(parents=True, exist_ok=True)
    asset_file = destination / FILENAME
    if not asset_file.exists():
        brushes, textures, images, names = [], [], [], {}
        pending = destination / ('create_mh_skin_v1.pending-' + str(os.getpid()) + '.blend')
        try:
            for identifier, label, description in PRESETS:
                image = make_image(identifier, destination / (identifier + '.png'))
                images.append(image)
                texture = bpy.data.textures.new('CM Skin ' + identifier, type='IMAGE')
                textures.append(texture)
                texture.image = image
                brush = bpy.data.brushes.new('CM Skin ' + label, mode='SCULPT')
                brushes.append(brush)
                names[identifier] = brush.name
                from . import blender_compat
                blender_compat.sculpt_brush_type(brush,'DRAW')
                brush.stroke_method = 'ANCHORED'
                brush.strength = .12
                brush.size = blender_compat.brush_size(60)
                brush.direction = 'ADD' if identifier in {'Scar', 'Freckles'} else 'SUBTRACT'
                brush.use_frontface = True
                brush.texture = texture
                brush.texture_slot.map_mode = 'AREA_PLANE'
                brush.texture_sample_bias = 0
                blender_compat.brush_falloff(brush,'SMOOTH')
                brush.asset_mark()
                brush.asset_data.catalog_id = CATALOG
                brush.asset_data.author = 'MetaHuman Maker — independent procedural pack'
                brush.asset_data.description = description + '. Drag to size the stamp; Ctrl reverses height.'
                brush.asset_data.tags.new('Skin')
                with bpy.context.temp_override(id=brush):
                    bpy.ops.ed.lib_id_load_custom_preview(filepath=str(destination / (identifier + '.png')))
            bpy.data.libraries.write(str(pending), set(brushes), fake_user=True, compress=True)
            (destination / 'blender_assets.cats.txt').write_text(
                '# MetaHuman Maker procedural sculpt brush catalog\nVERSION 1\n\n' + CATALOG + ':MetaHuman Maker Skin:MetaHuman Maker Skin\n', encoding='utf-8')
            (destination / 'pack.json').write_text(json.dumps({'version': 1, 'brushes': names, 'source': 'independently generated procedural stamps'}, indent=2), encoding='utf-8')
            os.replace(pending, asset_file)
        finally:
            for brush in brushes:
                bpy.data.brushes.remove(brush)
            for texture in textures:
                bpy.data.textures.remove(texture)
            for image in images:
                bpy.data.images.remove(image)
            if pending.exists():
                pending.unlink()
    library = ensure_library()
    bpy.ops.wm.save_userpref()
    return library


def activate(context, identifier):
    library = ensure_library()
    if not library:
        library = install()
    names = json.loads((folder() / 'pack.json').read_text(encoding='utf-8'))['brushes']
    # Native asset activation starts an asynchronous library scan on first use.
    # Keep Blender's native asset reference so edits can be saved in its shelf.
    try:
        # Packed brush libraries in 5.x resolve through native asset invocation.
        execution = 'INVOKE_DEFAULT' if bpy.app.version >= (5, 0, 0) else 'EXEC_DEFAULT'
        result = bpy.ops.brush.asset_activate(execution, asset_library_type='CUSTOM', asset_library_identifier=library.name,
            relative_asset_identifier=FILENAME + '/Brush/' + names[identifier])
    except RuntimeError:
        return False
    brush = context.tool_settings.sculpt.brush
    return result == {'FINISHED'} and brush is not None and brush.name == names[identifier]


def finish_activation():
    global _PENDING
    request = _PENDING
    if not request:
        return None
    try:
        context = bpy.context
        if (context.scene != request['scene'] or context.mode != 'SCULPT'
                or context.object != request['object']
                or context.scene.cm_features.skin_brush != request['identifier']):
            _PENDING = None
            return None
        area = request['area']
        if area not in list(request['window'].screen.areas):
            _PENDING = None
            return None
        with context.temp_override(window=request['window'], area=area,
                region=next(r for r in area.regions if r.type == 'WINDOW')):
            if activate(context, request['identifier']):
                context.scene.cm_settings.status = stamp_hint(request['identifier'])
                _PENDING = None
                return None
        if time.monotonic() > request['deadline']:
            context.scene.cm_settings.status = 'Brush library did not finish loading. Check its path in Preferences > File Paths > Asset Libraries, then use the brush again.'
            _PENDING = None
            return None
    except (ReferenceError, RuntimeError, ValueError, OSError, KeyError, StopIteration):
        _PENDING = None
        return None
    return .2


def stamp_hint(identifier):
    label = next(p[1] for p in PRESETS if p[0] == identifier)
    return label + ': drag to stamp, F for size, Shift to smooth, Ctrl to reverse height.'


class CM_OT_skin_brush(bpy.types.Operator):
    bl_idname = 'cm.skin_brush'
    bl_label = 'Use Skin Brush'
    bl_description = 'Activate the selected procedural stamp on the current sculpt; preserves the active layer and mask'
    action: bpy.props.EnumProperty(items=[('USE', 'Use', ''), ('INSTALL', 'Install', '')])

    def execute(self, context):
        global _PENDING
        try:
            if self.action == 'INSTALL':
                install()
                context.scene.cm_settings.status = '14 skin brush assets installed. Enter Sculpt Detail or a wrinkle layer, choose a brush and drag to stamp.'
            else:
                if context.mode != 'SCULPT':
                    raise ValueError('Enter Sculpt Detail or Sculpt Selected Wrinkle Layer first')
                identifier = context.scene.cm_features.skin_brush
                if activate(context, identifier):
                    _PENDING = None
                    context.scene.cm_settings.status = stamp_hint(identifier)
                else:
                    window = context.window or context.window_manager.windows[0]
                    area = context.area if context.area and context.area.type == 'VIEW_3D' else next(a for a in window.screen.areas if a.type == 'VIEW_3D')
                    _PENDING = {'scene': context.scene, 'object': context.object, 'identifier': identifier,
                        'window': window, 'area': area, 'deadline': time.monotonic() + 10}
                    context.scene.cm_settings.status = 'Loading skin brush assets…'
                    if not bpy.app.timers.is_registered(finish_activation):
                        bpy.app.timers.register(finish_activation, first_interval=.2)
        except (RuntimeError, ValueError, OSError, KeyError, TypeError) as exc:
            self.report({'ERROR'}, str(exc))
            return {'CANCELLED'}
        return {'FINISHED'}


def draw(layout, context):
    box = layout.box()
    box.label(text='Skin Brushes', icon='BRUSH_DATA')
    if context.mode=='SCULPT':
        brush=context.tool_settings.sculpt.brush
        if brush:
            box.label(text=brush.name)
            from .blender_compat import unified_paint
            unified=unified_paint(context,'sculpt')
            box.prop(unified if unified.use_unified_size else brush,'size',text='Size')
            box.prop(unified if unified.use_unified_strength else brush,'strength',text='Strength')
        if context.object and context.object.type=='MESH':
            row=box.row(align=True);row.label(text='Mirror')
            for axis in ('x','y','z'):row.prop(context.object.data,'use_mirror_'+axis,text=axis.upper(),toggle=True)
        box.label(text='Drag to sculpt · F size · Shift smooth')
    if not available():
        row = box.row()
        row.enabled = context.mode == 'SCULPT'
        row.operator('cm.skin_brush', text='Install 14 Skin Brushes', icon='ASSET_MANAGER').action = 'INSTALL'
        if context.mode != 'SCULPT':
            box.label(text='Enter a sculpt layer to install brushes.')
        return
    row = box.row(align=True)
    row.prop(context.scene.cm_features, 'skin_brush', text='')
    use = row.row(align=True)
    use.enabled = context.mode == 'SCULPT'
    use.operator('cm.skin_brush', text='Use Brush').action = 'USE'
    if context.mode != 'SCULPT':
        box.label(text='Enter a sculpt layer to use a brush.')
        return
    box.label(text='Shelf catalog: MetaHuman Maker Skin')


def draw_active(layout,context):
    """Keep the controls for the active owned sculpt in one short panel."""
    if context.mode!='SCULPT' or not context.object or context.object.type!='MESH':return False
    s,f=context.scene.cm_settings,context.scene.cm_features
    roles=[('Body' if s.part=='BODY' else 'Head',s.result),('Body',f.body_result),('Left Eye',f.eye_left),('Right Eye',f.eye_right),
           ('Teeth',f.teeth),('Head Detail',s.detail),('Body Detail',f.body_detail),
           ('WM1',f.wm1),('WM2',f.wm2),('WM3',f.wm3),('Full Character',f.full_sculpt)]
    label=next((name for name,obj in roles if obj and obj==context.object),None)
    if label is None:return False
    box=layout.box();box.label(text='Sculpt: '+label,icon='SCULPTMODE_HLT')
    row=box.row(align=True)
    if f.full_sculpt and context.object==f.full_sculpt:
        row.operator('cm.feature',text='Apply Sculpt',icon='CHECKMARK').action='FULL_SCULPT_APPLY'
        row.operator('cm.feature',text='Cancel',icon='X').action='FULL_SCULPT_CANCEL'
    else:row.operator('cm.finish_sculpt',text='Finish Sculpt',icon='CHECKMARK')
    box.operator('cm.focus_character',text='Frame Sculpt',icon='VIEWZOOM')
    draw(layout,context)
    layout.label(text='Ctrl+Z: undo a stroke.')
    layout.label(text='Finish before changing steps.')
    return True


def register():
    bpy.utils.register_class(CM_OT_skin_brush)
    ensure_library()


def unregister():
    global _PENDING
    _PENDING = None
    if bpy.app.timers.is_registered(finish_activation):
        bpy.app.timers.unregister(finish_activation)
    bpy.utils.unregister_class(CM_OT_skin_brush)
