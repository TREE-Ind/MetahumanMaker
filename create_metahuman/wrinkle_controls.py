"""Choose sparse native wrinkle composites from measured RigLogic map outputs."""

import math


def derive(data):
    names = data.get("animated_maps", [])
    poses = data.get("poses", [])
    if not names or not poses:
        raise ValueError("Native wrinkle bases need Bridge 0.5 expression data")
    values = [pose.get("animated_maps", []) for pose in poses]
    if any(
        len(row) != len(names) or not all(math.isfinite(v) and 0 <= v <= 1 for v in row)
        for row in values
    ):
        raise ValueError("Invalid animated-map output data")
    normal = [i for i, name in enumerate(names) if name.startswith("head_wm")]
    result = []
    for slot in ["WM1", "WM2", "WM3"]:
        target = [
            i
            for i in normal
            if names[i].startswith("head_" + slot.lower() + "_normal.")
        ]
        if not target:
            raise ValueError("Native data is missing " + slot + " region drivers")
        other = [i for i in normal if i not in target]
        maxima = {i: max(row[i] for row in values) for i in target}
        if any(v <= 0 for v in maxima.values()):
            raise ValueError("Native region has no measured driver")
        coverage = {i: 0 for i in target}
        selected = []
        while any(coverage[i] < 0.95 * maxima[i] for i in target):
            scores = []
            for index, row in enumerate(values):
                gain = sum(
                    min(row[i], max(maxima[i] - coverage[i], 0)) / maxima[i]
                    for i in target
                )
                scores.append(
                    gain / (1 + 4 * sum(row[i] for i in other))
                    if index not in selected and poses[index]["controls"]
                    else 0
                )
            index = max(range(len(scores)), key=scores.__getitem__)
            if scores[index] < 1e-6:
                raise ValueError("Cannot cover native wrinkle regions for " + slot)
            selected.append(index)
            coverage = {i: max(coverage[i], values[index][i]) for i in target}
        controls = {
            name: value
            for index in selected
            for name, value in poses[index]["controls"].items()
        }
        result.append({"name": slot + " Native Composite", "controls": controls})
    return result
