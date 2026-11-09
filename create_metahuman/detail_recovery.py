"""Recover old baked detail from a verified chain of archived subdivision levels."""

import re
from dataclasses import dataclass, field

import bpy
import numpy as np

from . import mesh_ops as m


@dataclass
class LegacyReference:
    objects: list
    parents: list
    explicit: set = field(default_factory=set)
    basis: dict = field(default_factory=dict)
    visible: dict = field(default_factory=dict)
    references: dict = field(default_factory=dict)

    def identity(self):
        return (
            tuple(obj.as_pointer() for obj in self.objects),
            tuple(tuple(p) for p in self.parents),
            tuple(sorted(self.explicit)),
        )


@dataclass
class RecoveredReference:
    reference: np.ndarray
    steps: list
    proof: list = field(default_factory=list)


def level(obj):
    return int(obj.get("cm_detail_level", 1))


def _name(name):
    return re.sub(r"\.\d{3,}$", "", name)


def _compatible(candidate, child):
    if (
        candidate == child
        or candidate.type != "MESH"
        or "cm_detail_level" not in candidate
    ):
        return False
    from .detail_layers import subdivision_type
    if subdivision_type(candidate)!=subdivision_type(child):return False
    if level(candidate) != level(child) - 1 or not candidate.data.shape_keys:
        return False
    if candidate.get("cm_role") in {"WRINKLE", "SOURCE", "TEMPLATE"} or candidate.get(
        "cm_wrinkle_native"
    ):
        return False
    for name in ("cm_library_role", "cm_part"):
        if candidate.get(name) and child.get(name) and candidate[name] != child[name]:
            return False
    expected = (
        len(candidate.data.vertices)
        + len(candidate.data.edges)
        + len(candidate.data.polygons)
    )
    return expected == len(child.data.vertices)


def describe(context, detail):
    """Find recorded/named predecessor candidates; geometry proof runs in the worker."""
    if not detail.data.shape_keys:
        raise ValueError(
            "The saved sculpt has no original Basis; retain it for reprojection"
        )
    result = LegacyReference([], [])
    indices = {}

    def visit(obj):
        if obj in indices:
            return indices[obj]
        index = len(result.objects)
        indices[obj] = index
        result.objects.append(obj)
        result.parents.append([])
        if obj.modifiers:
            raise ValueError(
                "Apply or remove active detail modifiers before recovering its saved surface"
            )
        from . import detail_layers as dl

        if dl.has_reference(obj) or level(obj) == 1:
            return index
        previous = obj.get("cm_detail_previous")
        if isinstance(previous, bpy.types.Object):
            if previous.name not in context.scene.objects or not _compatible(
                previous, obj
            ):
                raise ValueError(
                    "The selected previous sculpt level is incompatible; retain it and choose its actual predecessor"
                )
            candidates = [previous]
            result.explicit.add(index)
        else:
            suffix = "_LV" + str(level(obj))
            name = _name(obj.name)
            parent_name = _name(name[: -len(suffix)]) if name.endswith(suffix) else None
            candidates = [
                candidate
                for candidate in context.scene.objects
                if parent_name
                and _name(candidate.name) == parent_name
                and _compatible(candidate, obj)
            ]
        result.parents[index] = [visit(candidate) for candidate in candidates]
        return index

    visit(detail)
    return result


def _points(collection):
    result = np.empty((len(collection), 3), dtype=np.float32)
    collection.foreach_get("co", result.ravel())
    return result.astype(float)


def basis_for(spec, index):
    if index in spec.basis:
        return spec.basis[index]
    return _points(spec.objects[index].data.shape_keys.key_blocks[0].data)


def visible_for(context, spec, index):
    if index in spec.visible:
        return spec.visible[index]
    obj = spec.objects[index]
    return m.to_local(obj, m.evaluated_coordinates(context, obj))


def _world(obj, values):
    matrix = np.asarray(obj.matrix_world)
    return values @ matrix[:3, :3].T + matrix[:3, 3]


def _tolerance(world):
    return max(2e-7, float(np.linalg.norm(np.ptp(world, axis=0))) * 5e-7)


def resolve(context, spec, current_target, progress=None):
    """Prove each baked Basis against its predecessor before deriving a reference."""
    from . import detail_layers as dl

    memo = {}

    def recover(index):
        if index in memo:
            return memo[index]
        obj = spec.objects[index]
        reference = spec.references.get(index)
        if reference is None:
            reference = dl.read_reference(obj)
        if reference is not None:
            result = RecoveredReference(reference, dl.steps_for(obj))
            memo[index] = result
            return result
        basis = basis_for(spec, index)
        if level(obj) == 1:
            result = RecoveredReference(basis, [1])
            memo[index] = result
            return result
        if progress:
            progress("Checking saved sculpt level " + str(level(obj)) + "...")
        world_basis = _world(obj, basis)
        tolerance = _tolerance(world_basis)
        matches = []
        for parent_index in spec.parents[index]:
            parent = spec.objects[parent_index]
            visible = visible_for(context, spec, parent_index)
            recreated = dl.surface(
                context, parent, 1, target_mesh=obj.data, local_positions=visible,subdivision_type=dl.subdivision_type(obj)
            )
            match_error = float(np.max(np.linalg.norm(recreated - world_basis, axis=1)))
            if match_error > tolerance:
                continue
            previous = recover(parent_index)
            if progress:
                progress(
                    "Restoring the original surface at level " + str(level(obj)) + "..."
                )
            reference_world = dl.surface(
                context,
                parent,
                1,
                target_mesh=obj.data,
                local_positions=previous.reference,
                subdivision_type=dl.subdivision_type(obj),
            )
            matches.append(
                RecoveredReference(
                    m.to_local(obj, reference_world),
                    previous.steps + [1],
                    previous.proof
                    + [
                        {
                            "parent_index": parent_index,
                            "child_index": index,
                            "maximum_error_m": match_error,
                        }
                    ],
                )
            )
        if matches:
            result = matches[0]
            for other in matches[1:]:
                if (
                    result.steps != other.steps
                    or np.max(
                        np.linalg.norm(
                            _world(obj, other.reference)
                            - _world(obj, result.reference),
                            axis=1,
                        )
                    )
                    > tolerance
                ):
                    raise ValueError(
                        "Several previous layers match but have different original surfaces; select the actual Previous Layer"
                    )
            bound = obj.get("cm_detail_bound_steps")
            if bound is not None and list(bound) != result.steps:
                raise ValueError(
                    "The missing reference used a different subdivision history; restore its original reference from a saved project"
                )
            memo[index] = result
            return result
        # An old layer created directly at LV2/LV3 can have a pristine Basis.
        # Accept that only when it matches the current accepted subdivision;
        # never reinterpret a baked sculpt as the neutral surface.
        if index == 0 and not spec.parents[index] and index not in spec.explicit:
            target = current_target() if callable(current_target) else current_target
            if (
                np.max(np.linalg.norm(_world(obj, target) - world_basis, axis=1))
                <= tolerance
            ):
                result = RecoveredReference(
                    target,
                    [level(obj)],
                    [
                        {
                            "direct_basis": True,
                            "maximum_error_m": float(
                                np.max(
                                    np.linalg.norm(
                                        _world(obj, target) - world_basis, axis=1
                                    )
                                )
                            ),
                        }
                    ],
                )
                memo[index] = result
                return result
        raise ValueError(
            "The previous sculpt does not reproduce this saved Basis. Choose its unchanged Previous Layer or reopen a project containing it; sculpt retained"
        )

    return recover(0)
