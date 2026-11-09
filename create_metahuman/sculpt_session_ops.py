"""Separate sculpt-session actions shown in the workflow panel."""

import bpy
from . import sculpt_session


class CM_OT_sculpt_session(bpy.types.Operator):
    bl_idname = "cm.sculpt_session"
    bl_label = "Sculpting Session"
    action: bpy.props.EnumProperty(
        items=[
            ("OPEN", "Open", ""),
            ("RESUME", "Resume", ""),
            ("RETURN", "Save & Return", ""),
        ],
        default="OPEN",
    )

    def execute(self, context):
        s, f = context.scene.cm_settings, context.scene.cm_features
        try:
            if self.action == "OPEN":
                sculpt_session.start(context, s, f)
            elif self.action == "RESUME":
                sculpt_session.resume(context, s, f)
            else:
                sculpt_session.prepare_return(context, s, f)
            return {"FINISHED"}
        except Exception as exc:
            f.sculpt_session_status = str(exc)
            f.last_action_error = True
            self.report({"ERROR"}, str(exc))
            return {"CANCELLED"}


CLASSES = (CM_OT_sculpt_session,)
