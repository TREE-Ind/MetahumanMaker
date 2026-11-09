"""Native quadratic interval residuals that retain every supplied anchor.

For uniform knots x_i and linear cardinal hats h_i, write a residual as
sum_i h_i(t) (t/x_i - 1) G_i.  Choose G_0=0 and
G_i=-4*x_i/step * sum_{j<i} D_j.  Inside interval j this is exactly
4*u*(1-u)*D_j, with u=(t-x_j)/step.  Thus each midpoint displacement
D_j has compact interval support and every anchor remains unchanged.

The native implementation uses paired scalar and driver-multiplied solvers.
Non-blink controls can share an independent blink fade; a blink residual uses
gate=None, retaining its own anchors without an additional fade factor.
No runtime Python is required.
This basis preserves anchors; it does not certify the resulting geometry or
continuous motion. Callers must seal sources and validate compiled results.
"""
import re,uuid
from pathlib import Path
import numpy as np
from .native_corrective_gates import complementary_family


def _inputs(knots, fields):
    knots=np.asarray(knots,dtype=float);fields=np.asarray(fields,dtype=float)
    if knots.ndim!=1 or not 3<=len(knots)<=33 or len(knots)%4==0 or not np.isfinite(knots).all():
        raise ValueError('Supply 3 through 33 scalar knots in a non-quaternion-packed layout')
    step=float(knots[1]-knots[0])
    if knots[0]!=0 or knots[-1]!=1 or step<1/64 or not np.allclose(np.diff(knots),step,atol=1e-12,rtol=0):
        raise ValueError('Interval residuals require uniform knots from zero through one')
    if fields.ndim!=3 or fields.shape[0]!=len(knots)-1 or fields.shape[2]!=3 or not fields.shape[1] or not np.isfinite(fields).all():
        raise ValueError('Supply one finite canonical Nx3 midpoint displacement per interval')
    return knots,fields,step


def coefficients(knots, midpoint_fields):
    """Return anchor coefficients; keep the zero anchor structurally empty."""
    knots,fields,step=_inputs(knots,midpoint_fields)
    result=np.zeros((len(knots),*fields.shape[1:]))
    result[1:]=(-4*knots[1:]/step)[:,None,None]*np.cumsum(fields,axis=0)
    return result


def evaluate(knots, midpoint_fields, strength):
    """Evaluate the intended compact residual, independently of native weights."""
    knots,fields,step=_inputs(knots,midpoint_fields)
    if not np.isscalar(strength) or not np.isfinite(strength) or not 0<=strength<=1:
        raise ValueError('Choose a finite interval strength from zero through one')
    index=min(int(np.searchsorted(knots,strength,side='right')-1),len(fields)-1)
    u=(float(strength)-knots[index])/step
    return 4*u*(1-u)*fields[index]


def family(folder, midpoint_fields, *, knots, mesh, driver, gate=None,
           coordinate_linear, namespace=None):
    """Encode engine-space midpoint fields as bounded native pose requests.

    Output is a plan for write_pose_groups, not a staging manifest. The caller
    supplies canonical identity mappings, source seals and physical net-motion
    bounds. Individual cancelling coefficients can exceed the net displacement;
    maximum_encoded_offset_cm describes those fields, not allowable motion.
    """
    from .native_pose_sculpt_solver import _encode_small_fields
    folder=Path(folder).resolve();knots,fields,step=_inputs(knots,midpoint_fields)
    matrix=np.asarray(coordinate_linear,dtype=float)
    if matrix.shape!=(3,3) or not np.isfinite(matrix).all() or np.linalg.svd(matrix,compute_uv=False).min()<1e-4:
        raise ValueError('Supply the sealed, nonsingular native-to-engine linear transform')
    if not folder.is_dir():raise ValueError('Choose a fresh existing field directory')
    namespace=namespace or 'CM_interval_'+uuid.uuid4().hex[:8]
    if not re.fullmatch(r'CM_[A-Za-z0-9_]{1,27}',namespace):
        raise ValueError('Use a CM_ interval namespace of at most 30 characters')
    values=coefficients(knots,fields);inverse=np.linalg.inv(matrix);payloads={};encoding=[];maximum=0.
    for index in range(1,len(knots)):
        parts,proof=_encode_small_fields(values[index]);encoding.append(dict(strength=float(knots[index]),**proof))
        for part,payload in enumerate(parts):
            if np.any(np.linalg.norm(payload,axis=1)>1e-7):payloads.setdefault(part,{})[index]=payload
    targets=[];solvers=[]
    for part,members in sorted(payloads.items()):
        for mode in ('driver','constant'):
            rows=[]
            for index,payload in sorted(members.items()):
                ids=np.flatnonzero(np.linalg.norm(payload,axis=1)>1e-7)
                name=f'{namespace}_p{part}_{mode}_{index}.f32'
                native=(payload[ids]@inverse)*(1 if mode=='driver' else -1)
                with (folder/name).open('xb') as stream:stream.write(native.astype('<f4').tobytes())
                rows.append(dict(value=float(knots[index]),indices=ids.tolist(),file=name))
                maximum=max(maximum,float(np.linalg.norm(payload[ids],axis=1).max()))
            zeros=[float(value) for index,value in enumerate(knots[1:],1) if index not in members]
            plan=complementary_family(folder,rows,mesh=mesh,driver=driver,gate=gate,
                radius=step,namespace=namespace+f'_p{part}_'+('d' if mode=='driver' else 'c'),
                zero_values=zeros,multiply_driver=mode=='driver')
            targets.extend(plan['targets']);solvers.extend(plan['solvers'])
    if not targets:raise ValueError('The interval fields are empty; retain the preceding proposal')
    return dict(targets=targets,solvers=solvers,field_encoding=encoding,
        method='ANCHOR_PRESERVING_QUADRATIC_INTERVAL_RESIDUAL',
        maximum_encoded_offset_cm=maximum,
        maximum_midpoint_offset_cm=float(np.linalg.norm(fields,axis=2).max()),
        anchor_strengths=knots.tolist(),native_motion_accepted=False)
