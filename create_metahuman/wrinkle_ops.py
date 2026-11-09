"""File dialogs for reusable wrinkle sculpt bases."""

import bpy
from bpy.props import StringProperty
from bpy_extras.io_utils import ImportHelper, ExportHelper
from . import wrinkles


class CM_OT_save_wrinkle_base(bpy.types.Operator, ExportHelper):
    bl_idname = "cm.save_wrinkle_base"
    bl_label = "Save Wrinkle Base"
    filename_ext = ".blend"
    filter_glob: StringProperty(default="*.blend", options={"HIDDEN"})

    def execute(self, context):
        try:
            wrinkles.save_layer(
                context,
                context.scene.cm_settings,
                context.scene.cm_features,
                self.filepath,
            )
            context.scene.cm_settings.status = (
                "Custom wrinkle base saved with neutral sculpt and inverse-pose data."
            )
            return {"FINISHED"}
        except Exception as exc:
            context.scene.cm_settings.status = str(exc)
            self.report({"ERROR"}, str(exc))
            return {"CANCELLED"}


class CM_OT_load_wrinkle_base(bpy.types.Operator, ImportHelper):
    """Retarget saved Blender folds, or load an OBJ/FBX sculpt matching this posed base"""

    bl_idname = "cm.load_wrinkle_base"
    bl_label = "Load Custom Wrinkle Base"
    bl_options = {"REGISTER", "UNDO"}
    filter_glob: StringProperty(default="*.blend;*.obj;*.fbx", options={"HIDDEN"})

    def execute(self, context):
        try:
            wrinkles.load_layer(
                context,
                context.scene.cm_settings,
                context.scene.cm_features,
                self.filepath,
            )
            return {"FINISHED"}
        except Exception as exc:
            context.scene.cm_settings.status = str(exc)
            self.report({"ERROR"}, str(exc))
            return {"CANCELLED"}


CLASSES = (CM_OT_save_wrinkle_base, CM_OT_load_wrinkle_base)
