"""Small public-API differences between Blender 4.3 and 5.x authoring."""
import bpy


def eevee_engine():
    return 'BLENDER_EEVEE' if bpy.app.version>=(5,0,0) else 'BLENDER_EEVEE_NEXT'


def is_eevee(engine):
    return engine in {'BLENDER_EEVEE','BLENDER_EEVEE_NEXT'}


def unified_paint(context,mode):
    settings=context.tool_settings
    paint=getattr(settings,mode)
    unified=getattr(paint,'unified_paint_settings',None)
    return unified if unified is not None else settings.unified_paint_settings


def sculpt_brush_type(brush,value):
    name='sculpt_brush_type' if hasattr(brush,'sculpt_brush_type') else 'sculpt_tool'
    setattr(brush,name,value)


def brush_falloff(brush,value):
    name='curve_distance_falloff_preset' if hasattr(brush,'curve_distance_falloff_preset') else 'curve_preset'
    setattr(brush,name,value)


def brush_size(radius):
    # Blender 5 changed the stored native brush size from radius to diameter.
    return radius*2 if bpy.app.version>=(5,0,0) else radius


def asset_import_method(library):
    # Newer asset systems replace append/reuse with a packed asset library.
    # Static RNA enum items still include the legacy value in 5.2, although
    # the setter rejects it; select the documented version's actual mode.
    method='PACK' if bpy.app.version>=(5,0,0) else 'APPEND_REUSE'
    # Setting this property invalidates native asset loading, even when its
    # value stays the same. Retries must let the existing library scan finish.
    if library.import_method != method:
        library.import_method=method


def render_image_format(settings,value):
    if hasattr(settings,'media_type'):settings.media_type='IMAGE'
    settings.file_format=value
