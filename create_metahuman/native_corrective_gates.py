"""Build paired native corrective fields that fade with an independent control.

The two solvers share interpolation coordinates and kernels. Their opposite
geometry, multiplied by driver and driver*gate, evaluates F(driver)*(1-gate).
No runtime Python or per-character vertex lists are required.
"""
import math,re,struct,uuid,json
from pathlib import Path


def gated_grid_family(folder,samples,*,mesh,drivers,multipliers,radius=.5,namespace=None):
    """Interpolate a complete native control grid with selected zero-axis gates.

    Coordinates not listed as multipliers may have corrective geometry at zero.
    This permits gaze/squint corrections conditioned on blink, including full
    closure, while retaining exact zeros on the pure gaze and squint axes.
    """
    folder=Path(folder).resolve();samples=list(samples);drivers=list(drivers);multipliers=list(multipliers);namespace=namespace or 'CM_grid_'+uuid.uuid4().hex[:12]
    if not folder.is_dir() or not re.fullmatch(r'CM_[A-Za-z0-9_]{1,40}',namespace) or not re.fullmatch(r'[A-Za-z0-9_]+_lod0_mesh',mesh):raise ValueError('Choose an existing request folder and canonical LOD0 mesh/CM namespace')
    if not 1<=len(drivers)<=16 or len(set(drivers))!=len(drivers) or any(not re.fullmatch(r'CTRL_expressions\.[A-Za-z0-9_]+',n) for n in drivers) or not multipliers or len(set(multipliers))!=len(multipliers) or not set(multipliers)<=set(drivers):raise ValueError('Choose distinct native grid controls and a nonempty subset of multiplicative gates')
    if type(radius) not in (int,float) or not math.isfinite(radius) or radius<=0 or not 2<=len(samples)<=64 or (len(drivers)*len(samples))%4==0:raise ValueError('Choose a finite radius and a UE-compatible native scalar grid')
    gate_indices=[drivers.index(n) for n in multipliers];plans={}
    for row in samples:
        values=row['values']
        if len(values)!=len(drivers) or any(type(v) not in (int,float) or not math.isfinite(v) or not 0<=v<=1 for v in values):raise ValueError('Grid knots must contain finite strengths from zero through one')
        point=tuple(map(float,values))
        if point in plans:raise ValueError('Grid knots must be distinct')
        product=math.prod(point[i] for i in gate_indices)
        if not product:
            if set(row)!={'values'}:raise ValueError('Zero-gated knots must not contain corrective geometry')
            plans[point]=None;continue
        if product<1/64:raise ValueError('The grid knot product exceeds the native output-scale bound')
        name=row['file'];ids=list(row['indices']);file=(folder/name).resolve()
        if Path(name).name!=name or not name.endswith('.f32') or file.parent!=folder or not file.is_file():raise ValueError('Grid fields must be local float32 files')
        data=file.read_bytes()
        if not ids or len(data)!=12*len(ids) or len(set(ids))!=len(ids) or any(type(i) is not int or not 0<=i<2**32 for i in ids) or any(not math.isfinite(v[0]) for v in struct.iter_unpack('<f',data)):raise ValueError('Supply finite grid fields at distinct canonical vertex IDs')
        plans[point]=dict(file=name,indices=ids)
    import itertools
    axes=[sorted({p[i] for p in plans}) for i in range(len(drivers))]
    if any(a[0]!=0 or a[-1]!=1 for a in axes) or set(plans)!=set(itertools.product(*axes)):raise ValueError('Capture a complete Cartesian control grid including zero and full activation')
    targets=[];poses=[]
    for index,(point,field) in enumerate(sorted(plans.items())):
        pose=dict(name=f'{namespace}_pose_{index:02d}',values=list(point))
        if field:
            channel=f'{namespace}_{index:02d}';targets.append(dict(mesh=mesh,channel=channel,**field));pose.update(channel=channel,multipliers=multipliers,scale=1/math.prod(point[i] for i in gate_indices))
        poses.append(pose)
    return dict(targets=targets,solvers=[dict(name=namespace+'_solver',inputs=drivers,function='LINEAR',radius=float(radius),poses=poses)],zero_controls=multipliers,minimum_bridge='0.27.0',native_motion_accepted=False)


def interaction_family(folder,samples,*,mesh,drivers,radius=.5,namespace=None):
    """Build a complete two-control grid, structurally zero on either axis.

    Axis samples have values only; interior samples also supply indices/file.
    Multiplying each interior output by both controls retains exact axis zeros
    between interpolation knots. The native writer still verifies knot values.
    Interior target shapes are interpolation constraints at the supplied knots;
    a closed shape at those knots does not certify closure between them.
    """
    folder=Path(folder).resolve();samples=list(samples);drivers=list(drivers);namespace=namespace or 'CM_inter_'+uuid.uuid4().hex[:12]
    if not folder.is_dir() or not re.fullmatch(r'CM_[A-Za-z0-9_]{1,40}',namespace) or not re.fullmatch(r'[A-Za-z0-9_]+_lod0_mesh',mesh):raise ValueError('Choose an existing request folder and canonical LOD0 mesh/CM namespace')
    if len(drivers)!=2 or len(set(drivers))!=2 or any(not re.fullmatch(r'CTRL_expressions\.[A-Za-z0-9_]+',name) for name in drivers):raise ValueError('Choose two distinct native interaction controls')
    if type(radius) not in (int,float) or not math.isfinite(radius) or radius<=0 or not 4<=len(samples)<=64 or (2*len(samples))%4==0:raise ValueError('Choose a finite radius and a grid compatible with UE scalar interpolation')
    plans={}
    for row in samples:
        values=row['values']
        if len(values)!=2 or any(type(v) not in (int,float) or not math.isfinite(v) or not 0<=v<=1 for v in values):raise ValueError('Interaction knots must contain two finite strengths from zero through one')
        point=tuple(map(float,values))
        if point in plans:raise ValueError('Interaction grid knots must be distinct')
        if 0 in point:
            if set(row)!={'values'}:raise ValueError('Axis knots must not contain corrective geometry')
            plans[point]=None;continue
        if math.prod(point)<1/64:raise ValueError('Interaction knot product exceeds the native output-scale limit')
        name=row['file'];ids=list(row['indices']);file=(folder/name).resolve()
        if Path(name).name!=name or not name.endswith('.f32') or file.parent!=folder or not file.is_file():raise ValueError('Interaction fields must be local float32 files')
        data=file.read_bytes()
        if not ids or len(data)!=12*len(ids) or len(set(ids))!=len(ids) or any(type(i) is not int or not 0<=i<2**32 for i in ids) or any(not math.isfinite(v[0]) for v in struct.iter_unpack('<f',data)):raise ValueError('Supply finite interaction fields at distinct canonical vertex IDs')
        plans[point]=dict(file=name,indices=ids)
    axes=[sorted({p[i] for p in plans}) for i in range(2)]
    if any(axis[0]!=0 or len(axis)<2 for axis in axes) or set(plans)!={(a,b) for a in axes[0] for b in axes[1]}:raise ValueError('Capture the complete Cartesian control grid, including both zero axes')
    targets=[];poses=[]
    for index,(point,field) in enumerate(sorted(plans.items())):
        pose=dict(name=f'{namespace}_pose_{index:02d}',values=list(point))
        if field:
            channel=f'{namespace}_{index:02d}';targets.append(dict(mesh=mesh,channel=channel,**field));pose.update(channel=channel,multipliers=drivers,scale=1/math.prod(point))
        poses.append(pose)
    return dict(targets=targets,solvers=[dict(name=namespace+'_solver',inputs=drivers,function='LINEAR',radius=float(radius),poses=poses)],minimum_bridge='0.27.0',zero_on_either_axis=True,native_motion_accepted=False)


def complementary_family(folder,samples,*,mesh,driver,gate,radius=.5,namespace=None,zero_values=(),multiply_driver=True):
    """Return targets/solvers for a CM_DNA_POSE_CORRECTIVES_V1 request.

    Samples contain value, indices and a local float32 XYZ file. The caller
    retains responsibility for source DNA provenance, canonical vertex counts,
    compiled fields, physical bounds and native appearance acceptance. Optional
    zero-valued knots retain existing driver poses without empty morph targets.
    With gate=None, emit only the scalar solver. This supports a control's own
    interval residual without multiplying it by an independent blink fade.
    """
    folder=Path(folder).resolve();samples=list(samples);zero_values=list(zero_values);namespace=namespace or 'CM_fade_'+uuid.uuid4().hex[:12]
    if not folder.is_dir() or not re.fullmatch(r'CM_[A-Za-z0-9_]{1,40}',namespace):raise ValueError('Choose an existing request folder and a short CM_ namespace')
    if not re.fullmatch(r'[A-Za-z0-9_]+_lod0_mesh',mesh):raise ValueError('Choose an explicit canonical LOD0 mesh')
    if driver==gate or any(not re.fullmatch(r'CTRL_expressions\.[A-Za-z0-9_]+',name) for name in ([driver,gate] if gate is not None else [driver])):raise ValueError('Choose a native expression driver and an optional distinct fade control')
    if type(radius) not in (float,int) or not math.isfinite(radius) or radius<=0:raise ValueError('Choose a positive finite interpolation radius')
    if not 1<=len(samples)<=32 or len(samples)+len(zero_values)>32 or (len(samples)+len(zero_values)+1)%4==0:raise ValueError('Supply up to 32 strengths; the neutral plus supplied scalar knots cannot total a multiple of four on UE5.8')
    plans=[];seen=set()
    for row in samples:
        value=row['value'];name=row['file'];indices=list(row['indices'])
        if type(value) not in (float,int) or not math.isfinite(value) or not 1/64<=value<=1 or value in seen:raise ValueError('Use distinct finite sample strengths from 1/64 through 1')
        if Path(name).name!=name or not name.endswith('.f32'):raise ValueError('Canonical field files must stay beside their request')
        file=(folder/name).resolve()
        if file.parent!=folder or not file.is_file():raise ValueError('A canonical field is missing or outside the request folder')
        data=file.read_bytes()
        if not indices or len(data)!=len(indices)*12 or len(set(indices))!=len(indices) or any(type(i) is not int or not 0<=i<2**32 for i in indices) or any(not math.isfinite(v[0]) for v in struct.iter_unpack('<f',data)):raise ValueError('Supply unique canonical indices and corresponding finite float32 XYZ offsets')
        seen.add(value);plans.append(dict(value=float(value),file=name,indices=indices,data=data))
    zeros=[]
    for value in zero_values:
        if type(value) not in (float,int) or not math.isfinite(value) or not 1/64<=value<=1 or value in seen:raise ValueError('Use distinct zero-anchor strengths that do not overlap corrective samples')
        seen.add(value);zeros.append(float(value))
    plans.sort(key=lambda row:row['value']);targets=[];solvers=[];negative_files=[]
    for mode in (['on','cancel'] if gate is not None else ['on']):
        poses=[dict(name=f'{namespace}_{mode}_neutral',values=[0.])]
        for index,row in enumerate(plans):
            channel=f'{namespace}_{mode}_{index:02d}';file=row['file']
            if mode=='cancel':
                file=channel+'.f32';destination=folder/file
                if destination.exists():raise ValueError('The complementary field destination already exists')
                negative=bytearray(row['data'])
                for offset in range(3,len(negative),4):negative[offset]^=0x80
                negative_files.append((destination,bytes(negative)))
            targets.append(dict(mesh=mesh,channel=channel,indices=list(row['indices']),file=file))
            pose=dict(name=channel+'_pose',values=[row['value']],channel=channel)
            multipliers=([driver] if multiply_driver else [])+([gate] if mode=='cancel' else [])
            if multipliers:pose.update(multipliers=multipliers,scale=1/row['value'] if multiply_driver else 1.)
            poses.append(pose)
        poses.extend(dict(name=f'{namespace}_{mode}_zero_{index:02d}',values=[value]) for index,value in enumerate(sorted(zeros)))
        poses.sort(key=lambda row:row['values'])
        solver=dict(name=f'{namespace}_{mode}_solver',inputs=[driver],function='LINEAR',radius=float(radius),poses=poses)
        if mode=='cancel':solver['gate_values']={gate:1.}
        solvers.append(solver)
    written=[]
    try:
        for destination,data in negative_files:
            with destination.open('xb') as stream:stream.write(data)
            written.append((destination,data))
    except Exception:
        for destination,data in written:
            if destination.is_file() and destination.read_bytes()==data:destination.unlink()
        raise
    return dict(targets=targets,solvers=solvers,minimum_bridge='0.27.0',external_gate_controls=int(gate is not None),identity=f'F({driver}) * (1 - {gate})' if gate is not None else f'F({driver})',native_motion_accepted=False)


def endpoint_family(folder,sample,*,mesh,driver,namespace=None,radius=.5):
    """One intermediate scalar correction with unchanged zero/full endpoints."""
    return endpoint_bank(folder,[sample],mesh=mesh,driver=driver,namespace=namespace,radius=radius)


def endpoint_bank(folder,samples,*,mesh,driver,namespace=None,radius=.5,zero_values=()):
    """Interpolate authored scalar fields, retaining neutral/full and zero knots.

    A complete scalar cardinal basis supplies piecewise linear interpolation.
    Endpoint poses have no output channel. There is no extra driver multiplier
    or cancellation field to exaggerate a neighboring strength. Native center
    and compiled-motion verification remain required by the caller.
    """
    import numpy as np
    from .native_pose_sculpt_solver import _encode_small_fields
    folder=Path(folder).resolve();namespace=namespace or 'CM_endpoint_'+uuid.uuid4().hex[:8]
    samples=list(samples);zeros=list(zero_values)
    if (not folder.is_dir() or not re.fullmatch(r'CM_[A-Za-z0-9_]{1,40}',namespace)
            or not re.fullmatch(r'[A-Za-z0-9_]+_lod0_mesh',mesh)
            or not re.fullmatch(r'CTRL_expressions\.[A-Za-z0-9_]+',driver)):
        raise ValueError('Choose a local field directory, canonical LOD0 mesh and native scalar driver')
    if type(radius) not in (int,float) or not math.isfinite(radius) or radius<=0 or not 1<=len(samples)<=31:
        raise ValueError('Supply one through 31 scalar fields and a positive radius')
    plans={};encoding=[]
    for row in samples:
        value=row['value'];name=row['file'];ids=list(row['indices'])
        if type(value) not in (int,float) or not math.isfinite(value) or not 1/64<=value<=63/64 or value in plans:
            raise ValueError('Use distinct intermediate strengths from 1/64 through 63/64')
        path=(folder/name).resolve()
        if Path(name).name!=name or path.parent!=folder or not name.endswith('.f32') or not path.is_file():
            raise ValueError('Keep each scalar field beside its request')
        body=path.read_bytes()
        if (not ids or len(set(ids))!=len(ids) or any(type(i) is not int or not 0<=i<2**32 for i in ids)
                or len(body)!=12*len(ids)):
            raise ValueError('Use distinct canonical IDs and matching float32 XYZ values')
        field=np.frombuffer(body,dtype='<f4').reshape(-1,3).astype(float)
        if not np.isfinite(field).all():raise ValueError('The scalar field contains nonfinite values')
        parts,proof=_encode_small_fields(field);encoding.append(dict(strength=value,**proof))
        plans[float(value)]=(ids,parts)
    for value in zeros:
        if type(value) not in (int,float) or not math.isfinite(value) or not 0<value<=1 or value in plans:
            raise ValueError('Use distinct retained strengths outside the authored samples')
        plans[float(value)]=None
    plans[0.]=None;plans[1.]=None
    if len(plans)>33 or len(plans)%4==0:raise ValueError('The scalar knot count cannot exceed 33 or be a multiple of four on UE 5.8')
    targets=[];solvers=[];files=[]
    for part in range(max(len(row[1]) for row in plans.values() if row)):
        poses=[]
        for number,(value,row) in enumerate(sorted(plans.items())):
            pose=dict(name=f'{namespace}_p{part}_k{number}',values=[value])
            if row and part<len(row[1]):
                ids,parts=row;payload=parts[part];selected=np.flatnonzero(np.linalg.norm(payload,axis=1)>1e-7)
                if len(selected):
                    channel=f'{namespace}_on_p{part}_k{number}';filename=channel+'.f32';files.append((folder/filename,payload[selected].astype('<f4').tobytes()))
                    targets.append(dict(mesh=mesh,channel=channel,file=filename,indices=[ids[i] for i in selected]));pose['channel']=channel
            poses.append(pose)
        if any('channel' in pose for pose in poses):solvers.append(dict(name=f'{namespace}_p{part}_solver',inputs=[driver],function='LINEAR',radius=float(radius),poses=poses))
    if not targets:raise ValueError('The endpoint correction contains no displacement')
    if any(path.exists() for path,_ in files):raise ValueError('Retain the existing endpoint correction files')
    written=[]
    try:
        for path,data in files:
            with path.open('xb') as stream:stream.write(data)
            written.append((path,data))
    except Exception:
        for path,data in written:
            if path.is_file() and path.read_bytes()==data:path.unlink()
        raise
    return dict(targets=targets,solvers=solvers,encoding=encoding,minimum_bridge='0.27.0',
                identity='SCALAR_CARDINAL_FIELDS_WITH_ZERO_ENDPOINTS',retained_endpoints=[0.,1.],native_motion_accepted=False)


def write_pose_groups(folder,targets,solvers,maximum_offset):
    """Keep each solver with its fields while satisfying per-request SDK limits.

    One request retains the legacy filename. Larger plans use up to eight sealed
    groups consumed in order by the native stager; no solver is split or dropped.
    """
    folder=Path(folder);by_channel={row['channel']:row for row in targets}
    if len(by_channel)!=len(targets):raise ValueError('Use distinct native channels across pose groups')
    groups=[];current=[];channels=set();seen=set()
    for solver in solvers:
        owned={pose['channel'] for pose in solver['poses'] if pose.get('channel')}
        if not owned or not owned<=by_channel.keys() or owned&seen or len(owned)>64:raise ValueError('Keep distinct solver-owned fields inside the native request limit')
        if current and (len(channels|owned)>64 or len(current)==16):groups.append((current,channels));current=[];channels=set()
        current.append(solver);channels.update(owned);seen.update(owned)
    if current:groups.append((current,channels))
    if seen!=by_channel.keys() or not 1<=len(groups)<=8:raise ValueError('The native pose plan needs one through eight complete groups')
    names=[]
    for index,(members,selected) in enumerate(groups):
        name='pose-request.json' if len(groups)==1 else f'pose-request-{index:02d}.json'
        data=dict(schema='CM_DNA_POSE_CORRECTIVES_V1',maximum_offset=maximum_offset,targets=[row for row in targets if row['channel'] in selected],solvers=members)
        with (folder/name).open('x') as stream:json.dump(data,stream)
        names.append(name)
    return names
