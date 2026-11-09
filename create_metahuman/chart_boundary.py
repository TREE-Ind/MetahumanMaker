"""Bounded interior optimization followed by independently checked boundary pinning."""
import numpy as np


def solve(rest,faces,initial,boundary,positions,*,cancelled=None):
    import igl
    from .intrinsic_disk import signed_areas,boundary_is_simple
    if not boundary_is_simple(positions,np.arange(len(positions))):raise ValueError('Original source boundary crosses itself')
    current=np.array(initial,dtype=float,copy=True);attempts=[]
    for penalty in (1e4,1e6,1e8,1e10):
        if cancelled and cancelled():raise RuntimeError('Attachment reconstruction cancelled')
        slim=igl.slim_precompute(np.asfortranarray(rest),np.asfortranarray(faces,dtype=np.int32),np.asfortranarray(current),
            igl.MappingEnergyType.SYMMETRIC_DIRICHLET,np.asarray(boundary,dtype=np.int32),np.asfortranarray(positions),penalty)
        result=igl.slim_solve(slim,30)
        if not np.isfinite(result).all() or np.any(signed_areas(result,faces)<=1e-14) or not boundary_is_simple(result,boundary):raise ValueError('Boundary continuation failed chart orientation or simplicity')
        error=float(np.linalg.norm(result[boundary]-positions,axis=1).max(initial=0))
        pinned=result.copy();pinned[boundary]=positions
        areas=signed_areas(pinned,faces);valid=bool(np.all(areas>1e-14) and boundary_is_simple(pinned,boundary))
        attempts.append(dict(penalty=penalty,maximum_boundary_error_before_pin=error,pinned_chart_valid=valid))
        if valid:
            return pinned,dict(method='SOURCE_BOUNDARY_CONTINUATION_AND_VERIFIED_PIN_V1',attempts=attempts,
                original_source_boundary_exact=bool(np.array_equal(pinned[boundary],positions)),boundary_vertices=len(boundary),minimum_pinned_uv_area=float(areas.min()))
        current=result
    raise ValueError('No oriented interior supports the original source boundary: '+str(attempts))
