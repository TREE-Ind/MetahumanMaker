"""Prepare a complete source/template workflow from one imported character."""
from pathlib import Path

import bpy

from . import character_project, guided
from . import mesh_ops as m


def read_meshes(context, filename):
    previous = set(bpy.data.objects)
    extension = Path(filename).suffix.lower()
    if extension == '.obj':
        result = bpy.ops.wm.obj_import(filepath=str(filename))
    elif extension == '.fbx':
        result = bpy.ops.import_scene.fbx(filepath=str(filename), use_custom_normals=True)
    elif extension in {'.glb', '.gltf'}:
        result = bpy.ops.import_scene.gltf(filepath=str(filename))
    else:
        raise ValueError('Supported formats: OBJ, FBX, GLB and GLTF')
    if result != {'FINISHED'}:
        raise ValueError('Blender could not import the selected file')
    return [obj for obj in bpy.data.objects if obj not in previous and obj.type == 'MESH']


def start(context, s, f, filename, mode, *, placement='AUTO'):
    filename = Path(filename).resolve()
    if not filename.is_file():
        raise ValueError('Choose an existing source mesh file')
    if mode not in {'FULL_HEAD', 'HEAD_ONLY', 'FULL_BODY'}:
        raise ValueError('Choose Wrap Full Head, Wrap Head Only or Wrap Full Body')
    if placement not in {'AUTO', 'FIT', 'KEEP'}:
        raise ValueError('Choose Automatic, Fit to References or Keep Imported placement')
    library = guided.character_library()
    if not library:
        raise ValueError('Prepare the local MetaHuman character library before guided import, or use Advanced to import your own templates')

    def create(context, s, f):
        # The preceding character stays available for Undo/reuse. Reset pointers
        # before importing so no old result/accessory is mistaken for this source.
        for obj in context.scene.objects:
            obj.hide_set(True)
            obj.hide_render = True
        for name in ('source', 'template', 'result', 'detail'):
            setattr(s, name, None)
        for name in ('head_result', 'body_result', 'body_template', 'body_detail',
                     'eye_left', 'eye_right', 'teeth'):
            setattr(f, name, None)
        meshes = read_meshes(context, filename)
        if len(meshes) != 1:
            raise ValueError(f'Guided import needs one character mesh; found {len(meshes)}. Export a single head or body without separate hair, clothing or accessories.')
        source = meshes[0]
        m.check_mesh(source, 'Imported source')
        if not len(source.data.polygons):
            raise ValueError('The source needs surface faces to wrap')
        s.source = source
        s.part = 'FULL_BODY' if mode == 'FULL_BODY' else 'HEAD'
        f.full_head = mode != 'HEAD_ONLY'
        f.input_mode = 'SCAN'
        from . import native_body
        if not f.is_property_set('body_backend') and native_body.available(f):
            f.body_backend = 'NATIVE'
        guided.load_character_template(context, s, f, library, include_body=mode == 'FULL_BODY')
        from . import surface_identity
        native = surface_identity.recognized(source, s, f) if placement == 'AUTO' else False
        if placement == 'FIT' or (placement == 'AUTO' and not native):
            guided.align_for_detection(context, s, f)
            source['cm_source_alignment'] = 'FIT'
        else:
            source['cm_source_alignment'] = 'METAHUMAN' if native else 'KEEP'
        source['cm_source_file'] = str(filename)
        source['cm_source_import_mode'] = mode
        s.stage, f.ui_stage = 'WRAP', 'WRAP'
        m.activate(context, source)

    previous_folder = s.project_dir
    guided.project_folder(s)
    try:
        character_project.start(context, s, f, create)
    except BaseException:
        s.project_dir = previous_folder
        raise
    s.status = ('Recognized MetaHuman UV surface; imported size and placement retained. '
                if s.source.get('cm_source_alignment') == 'METAHUMAN' else '') + ('Ready for direct native conform; no landmark detection needed.' if s.source.get('cm_source_alignment')=='METAHUMAN' else 'Source and matching references ready. Review alignment, then Auto Wrap or place landmark pairs.')
    if not bpy.app.background and context.window:
        from . import compare_views
        area = next((a for a in context.window.screen.areas if a.type == 'VIEW_3D'), None)
        if area:
            try:
                with context.temp_override(window=context.window, area=area,
                        region=next(r for r in area.regions if r.type == 'WINDOW')):
                    compare_views.open_views(context)
                    # Newly split areas can share their old coordinates until
                    # the first window event. Resolve the actual left/right
                    # panes after that event, then reveal the correct controls.
                    window, scene, source = context.window, context.scene, s.source
                    attempts = 0

                    def finish_layout():
                        nonlocal attempts
                        try:
                            if (window.scene != scene or scene.cm_settings.source != source
                                    or scene.cm_features.ui_stage != 'WRAP'):
                                return None
                            current = compare_views.recover(window)
                            if not current:
                                return None
                            left, right = current['areas']
                            attempts += 1
                            if not compare_views.adjacent(left, right):
                                return .1 if attempts < 10 else None
                            left.spaces.active.show_region_ui = True
                            right.spaces.active.show_region_ui = False
                            left.spaces.active.shading.type = 'SOLID'
                            right.spaces.active.shading.type = 'MATERIAL'
                            region = next((r for r in left.regions if r.type == 'UI'), None)
                            if not region or region.is_property_readonly('active_panel_category'):
                                return .1 if attempts < 10 else None
                            region.active_panel_category = 'MetaHuman Maker'
                            with bpy.context.temp_override(window=window):
                                compare_views.frame(bpy.context, current)
                            return None
                        except ReferenceError:
                            return None
                        except (ValueError, RuntimeError) as exc:
                            scene.cm_settings.status = 'Imported. Open Compare Source / Result: ' + str(exc)
                            return None

                    interval = finish_layout()
                    if interval is not None:
                        bpy.app.timers.register(finish_layout, first_interval=interval)
            except (ValueError, RuntimeError) as exc:
                s.status = 'Imported. Open Compare Source / Result to review alignment: ' + str(exc)
