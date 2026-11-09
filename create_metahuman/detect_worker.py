"""Optional Python process for local MediaPipe inference; never imports Blender."""

import json
from pathlib import Path
import sys


def detect(request):
    import mediapipe as mp
    from mediapipe.tasks import python
    from mediapipe.tasks.python import vision

    kind = request.get("kind", "FACE")
    model = Path(request["models"]) / (
        "face_landmarker.task" if kind == "FACE" else "pose_landmarker_lite.task"
    )
    options = python.BaseOptions(model_asset_path=str(model))
    if kind == "FACE":
        detector = vision.FaceLandmarker.create_from_options(
            vision.FaceLandmarkerOptions(
                base_options=options,
                min_face_detection_confidence=0.35,
                min_face_presence_confidence=0.35,
            )
        )
        selected = [1, 4, 33, 133, 263, 362, 61, 291, 152, 10, 168, 17, 234, 454]
    else:
        detector = vision.PoseLandmarker.create_from_options(
            vision.PoseLandmarkerOptions(
                base_options=options,
                min_pose_detection_confidence=0.35,
                min_pose_presence_confidence=0.35,
            )
        )
        selected = [0, 7, 8, 11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28, 31, 32]
    output = {}
    with detector:
        for role, filename in request["images"].items():
            result = detector.detect(mp.Image.create_from_file(filename))
            faces = result.face_landmarks if kind == "FACE" else result.pose_landmarks
            if not faces:
                raise ValueError(
                    f"No {kind.lower()} detected in {role}. Check Front Axis or use Manual landmarks."
                )
            output[role] = {
                str(i): [faces[0][i].x, faces[0][i].y]
                for i in selected
                if kind == "FACE" or (faces[0][i].visibility or 0) >= 0.45
            }
    return output


if __name__ == "__main__":
    request = json.loads(Path(sys.argv[1]).read_text())
    try:
        result = {"status": "complete", "landmarks": detect(request)}
    except Exception as exc:
        result = {"status": "failed", "error": str(exc)}
    Path(sys.argv[2]).write_text(json.dumps(result), encoding="utf-8")
    if result["status"] != "complete":
        raise SystemExit(1)
