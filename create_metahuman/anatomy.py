"""Editable regional deformations and seam matching, independent of a body model."""

import numpy as np
from . import geometry

BODY_CONTROLS = (
    ("main", "gender", "Gender"),
    ("main", "mass", "Body Mass"),
    ("main", "definition", "Definition"),
    ("main", "height", "Height"),
    ("main", "age", "Age"),
    ("upper", "shoulders", "Shoulders"),
    ("upper", "chest", "Chest"),
    ("upper", "back", "Back"),
    ("upper", "neck", "Neck"),
    ("upper", "traps", "Trapezius"),
    ("arms", "upper_arm", "Upper Arm Girth"),
    ("arms", "forearm", "Forearm Girth"),
    ("arms", "biceps", "Biceps"),
    ("arms", "triceps", "Triceps"),
    ("arms", "arm_length", "Arm Length"),
    ("arms", "armpit", "Armpit Height"),
    ("lower", "waist", "Waist"),
    ("lower", "abdomen", "Abdomen"),
    ("lower", "hips", "Hips"),
    ("lower", "glutes", "Glutes"),
    ("lower", "thighs", "Thighs"),
    ("lower", "calves", "Calves"),
    ("lower", "leg_length", "Leg Length"),
    ("hands_feet", "hands", "Hand Size"),
    ("hands_feet", "feet", "Foot Size"),
    ("hands_feet", "fingers", "Finger Length"),
    ("details", "posture", "Posture"),
    ("details", "asymmetry", "Asymmetry"),
    ("details", "clavicles", "Clavicles"),
    ("details", "knees", "Knees"),
    ("details", "ankles", "Ankles"),
)


def body_deform(vertices, parameters):
    base = geometry.points(vertices)
    result = base.copy()
    lo, hi = base.min(0), base.max(0)
    height = max(hi[2] - lo[2], 1e-8)
    lateral = int(np.argmax(hi[:2] - lo[:2]))
    depth = 1 - lateral
    center = (lo + hi) * 0.5
    side = base[:, lateral] - center[lateral]
    front = base[:, depth] - center[depth]
    z = (base[:, 2] - lo[2]) / height
    width = max(np.ptp(side), 1e-8)
    arm = np.clip((np.abs(side) / width - 0.16) / 0.16, 0, 1)
    torso = 1 - arm
    band = lambda level, spread: np.exp(-0.5 * ((z - level) / spread) ** 2)
    horizontal = np.zeros(len(base))
    sagittal = np.zeros(len(base))
    vertical = np.zeros(len(base))
    p = lambda key: float(parameters.get(key, 0))
    horizontal += 0.16 * p("mass") + 0.07 * p("gender") * band(0.80, 0.1)
    sagittal += 0.18 * p("mass") + 0.10 * p("definition") * np.cos(z * 28) * torso
    horizontal += 0.17 * p("shoulders") * band(0.82, 0.07)
    sagittal += 0.22 * p("chest") * band(0.76, 0.07) * torso * (front < 0)
    sagittal += 0.20 * p("back") * band(0.76, 0.1) * torso * (front >= 0)
    horizontal += 0.20 * p("neck") * band(0.91, 0.04) * torso
    vertical += 0.015 * p("traps") * band(0.86, 0.045) * torso
    for name, level, spread, amount in [
        ("waist", 0.57, 0.07, 0.20),
        ("hips", 0.47, 0.07, 0.18),
        ("thighs", 0.35, 0.10, 0.16),
        ("calves", 0.18, 0.065, 0.18),
        ("knees", 0.26, 0.025, 0.14),
        ("ankles", 0.07, 0.025, 0.14),
    ]:
        horizontal += amount * p(name) * band(level, spread) * torso
    sagittal += 0.25 * p("abdomen") * band(0.59, 0.075) * torso * (front < 0)
    sagittal += 0.22 * p("glutes") * band(0.46, 0.06) * torso * (front >= 0)
    for name, level, amount in [
        ("upper_arm", 0.73, 0.18),
        ("forearm", 0.63, 0.18),
        ("biceps", 0.74, 0.2),
        ("triceps", 0.76, 0.2),
    ]:
        influence = band(level, 0.065) * arm
        sagittal += amount * p(name) * influence
        vertical += 0.02 * p(name) * influence
    horizontal += 0.12 * p("arm_length") * arm
    vertical += 0.02 * p("armpit") * band(0.76, 0.045) * np.clip(arm * 2, 0, 1)
    hands = arm * band(0.55, 0.09)
    horizontal += 0.10 * (p("hands") + p("fingers")) * hands
    sagittal += 0.18 * p("hands") * hands
    feet = band(0.025, 0.035)
    sagittal += 0.24 * p("feet") * feet
    horizontal += 0.12 * p("feet") * feet
    vertical += 0.07 * p("leg_length") * np.minimum(z, 0.5)
    vertical -= 0.015 * p("age") * band(0.85, 0.12)
    sagittal += 0.10 * p("clavicles") * band(0.84, 0.022) * torso
    result[:, lateral] += side * horizontal + 0.005 * height * p("asymmetry") * band(
        0.6, 0.25
    )
    result[:, depth] += front * sagittal + 0.035 * height * p("posture") * band(
        0.80, 0.18
    )
    result[:, 2] += height * vertical
    result[:, 2] = lo[2] + (result[:, 2] - lo[2]) * (1 + 0.15 * p("height"))
    return result


def match_neck(head, head_indices, body, body_indices):
    """Return exact bijective boundary matches without changing either mesh's topology."""
    head, body = geometry.points(head), geometry.points(body)
    a, b = np.asarray(head_indices, dtype=int), np.asarray(body_indices, dtype=int)
    if len(a) != len(b) or len(a) < 3:
        raise ValueError(
            "Head/body neck rings must have the same count (at least 3 vertices). Define CM_Neck on each mesh."
        )
    center_a, center_b = head[a].mean(0), body[b].mean(0)
    order_a = np.argsort(np.arctan2(head[a, 1] - center_a[1], head[a, 0] - center_a[0]))
    order_b = np.argsort(np.arctan2(body[b, 1] - center_b[1], body[b, 0] - center_b[0]))
    a, b = a[order_a], b[order_b]
    candidates = [
        np.roll(order, shift) for order in (b, b[::-1]) for shift in range(len(b))
    ]
    mapped = min(candidates, key=lambda indices: np.sum((head[a] - body[indices]) ** 2))
    return a, mapped


def cavity_values(vertices, normals, edges, radius=0.01,*,triangles=None):
    """Signed concave curvature on physical adjacency, independent of UV splits."""
    vertices, normals = geometry.points(vertices), geometry.points(normals)
    if normals.shape!=vertices.shape:raise ValueError('Cavity normals must match the surface')
    edges=np.asarray(edges)
    if edges.ndim!=2 or edges.shape[1]!=2 or edges.dtype.kind not in 'iu' or np.any(edges<0) or np.any(edges>=len(vertices)):raise ValueError('Invalid cavity edges')
    physical,weld=np.unique(vertices,axis=0,return_inverse=True)
    joined=np.unique(np.sort(weld[edges],axis=1),axis=0);joined=joined[joined[:,0]!=joined[:,1]]
    shared=np.zeros_like(physical)
    if triangles is None:
        np.add.at(shared,weld,normals)
    else:
        triangles=np.asarray(triangles)
        if triangles.ndim!=2 or triangles.shape[1]!=3 or triangles.dtype.kind not in 'iu' or np.any(triangles<0) or np.any(triangles>=len(vertices)):raise ValueError('Invalid cavity triangles')
        corners=weld[triangles];p=physical[corners];area=np.cross(p[:,1]-p[:,0],p[:,2]-p[:,0])
        for corner in range(3):np.add.at(shared,corners[:,corner],area)
    shared/=np.maximum(np.linalg.norm(shared,axis=1,keepdims=True),1e-30)
    laplacian=geometry.neighbor_average(physical,joined)-physical
    curvature=np.sum(laplacian*shared,axis=1)
    return np.clip(curvature/max(radius,1e-8),0,1)[weld]
