"""Named inspection combinations of native controls, evaluated by RigLogic."""


def native_curve_key(name):
    """Match raw, dotted and Unreal curve names using native FName casing."""
    if not isinstance(name,str) or not name:raise ValueError('Facial control names must be nonempty strings')
    key=name.lower().replace('.','_')
    return key if key.startswith('ctrl_expressions_') else 'ctrl_expressions_'+key


def lid_capture_poses(raw_controls,mode='LIDS',current=None):
    """Independent, complete native eye grids compatible with calibration APIs."""
    import math
    available={name.removeprefix('CTRL_expressions.') for name in raw_controls}
    rows=[dict(name='Neutral',controls={})]
    if mode=='CURRENT':
        if not isinstance(current,dict):raise ValueError('Choose explicit current facial controls')
        values={}
        for key,value in current.items():
            if not isinstance(key,str):raise ValueError('Facial control names must be strings')
            name=key.removeprefix('CTRL_expressions.')
            if name in values:raise ValueError('Do not repeat a facial control with different prefixes')
            if name not in available or type(value) not in (int,float) or not math.isfinite(value) or not 0<=value<=1:
                raise ValueError('Capture controls must be supported finite values from 0 to 1')
            values[name]=value
        values={name:value for name,value in values.items() if value}
        if values:rows.append(dict(name='CurrentControls',controls=values))
        return rows
    gaze={'GAZE_UP':'eyeLookUp','GAZE_DOWN':'eyeLookDown','GAZE_LEFT':'eyeLookLeft','GAZE_RIGHT':'eyeLookRight'}.get(mode)
    if mode not in {'LIDS','LIDS_FINE','LIDS_DENSE','LIDS_INTERVALS','BLINK_FINE','BLINK_DENSE'} and not gaze:raise ValueError('Choose a supported eye capture set')
    knots=(0.,.5,1.)
    squint_knots=tuple(i/16 for i in range(17)) if mode=='LIDS_INTERVALS' else tuple(i/8 for i in range(9)) if mode=='LIDS_DENSE' else (0.,.25,.5,.75,1.) if mode=='LIDS_FINE' else knots
    for side in ('L','R'):
        for i,squint in enumerate(squint_knots):
            for j,blink in enumerate(tuple(i/8 for i in range(9)) if mode=='BLINK_DENSE' else (0.,.25,.5,.75,1.) if mode=='BLINK_FINE' else knots):
                for k,look in enumerate(knots if gaze else (0.,)):
                    values={key:value for key,value in ((('eyeSquintInner'+side),squint),(('eyeBlink'+side),blink)) if value}
                    if gaze and look:values[gaze+side]=look
                    if not values:continue
                    missing=set(values)-available
                    if missing:raise ValueError('This face is missing capture controls: '+', '.join(sorted(missing)))
                    rows.append(dict(name='Eye'+side+str(i)+str(j)+str(k),controls=values))
    return rows


def verify_lid_capture(capture,source_mesh):
    """Verify native identity, isolation and all requested independent controls."""
    return verify_pose_capture(capture,source_mesh,required_lod=0)


def verify_pose_capture(capture,source_mesh,*,required_lod=None):
    """Verify evaluated controls and the actual requested native LOD per pose."""
    import json,math
    from pathlib import Path
    if not capture.get('poses'):raise ValueError('Native pose capture is empty')
    for row in capture['poses']:
        lod=row.get('lod',required_lod)
        if type(lod) is not int or lod<0 or (required_lod is not None and lod!=required_lod):
            raise ValueError('Native capture has an invalid requested LOD: '+row['name'])
        meta=json.loads(Path(row['file']).read_text())
        if (meta.get('source_mesh')!=source_mesh or meta.get('dna_mode')!='ASSEMBLED_EMBEDDED'
                or meta.get('material_instances_isolated') is not True or type(meta.get('lod')) is not int or meta['lod']!=lod):
            raise ValueError('Native identity, LOD or material isolation did not match capture '+row['name'])
        curves=meta.get('raw_curves')
        # Native curve exports are sparse: an empty dictionary is a valid
        # evaluated neutral, while missing nonzero requests still fail below.
        if not isinstance(curves,dict):raise ValueError('Eye capture has invalid native control data: '+row['name'])
        actual={native_curve_key(name):value for name,value in curves.items() if name.lower().startswith(('ctrl_expressions.','ctrl_expressions_'))}
        expected={native_curve_key(name):value for name,value in row['controls'].items()}
        if len(expected)!=len(row['controls']) or any(type(value) not in (int,float) or not math.isfinite(value) or not 0<=value<=1 for value in expected.values()):
            raise ValueError('Native capture has repeated or invalid requested controls: '+row['name'])
        if 'raw_controls' in meta and not expected.keys()<={native_curve_key(name) for name in meta['raw_controls']}:
            raise ValueError('Native capture requests unsupported face controls: '+row['name'])
        if any(not isinstance(value,(int,float)) or not math.isfinite(value) for value in actual.values()):
            raise ValueError('Eye capture has invalid evaluated controls: '+row['name'])
        if any(abs(actual.get(name,0.)-value)>1e-6 for name,value in expected.items()) or any(abs(value-expected.get(name,0.))>1e-6 for name,value in actual.items()):
            raise ValueError('Native controls did not match the independent eye sample '+row['name'])
    capture.update(controls_verified=True,material_isolation_verified=True,lods_verified=True)
    return capture


def inspection_poses(raw_controls):
    prefix = 'CTRL_expressions.'
    available = set(raw_controls)

    def paired(name, value=1):
        return {name + side: value for side in ('L', 'R')}

    def lips(name):
        return {name + corner: 1 for corner in ('UL', 'UR', 'DL', 'DR')}

    recipes = [
        ('Blink Both', paired('eyeBlink')),
        ('Eyes Wide', paired('eyeWiden')),
        ('Squint Both', {**paired('eyeSquintInner'), **paired('eyeCheekRaise', .5)}),
        ('Look Left', paired('eyeLookLeft')),
        ('Look Right', paired('eyeLookRight')),
        ('Look Up', paired('eyeLookUp')),
        ('Look Down', paired('eyeLookDown')),
        ('Blink Looking Up', {**paired('eyeLookUp'), **paired('eyeBlink')}),
        ('Blink Looking Down', {**paired('eyeLookDown'), **paired('eyeBlink')}),
        ('Brows Raised', {**paired('browRaiseIn'), **paired('browRaiseOuter')}),
        ('Brows Lowered', paired('browDown')),
        ('Smile Both', {**paired('mouthCornerPull'), **paired('eyeCheekRaise', .35)}),
        ('Frown Both', paired('mouthCornerDepress')),
        ('Lips Puckered', lips('mouthLipsPurse')),
        ('Lips Funnel', lips('mouthFunnel')),
        ('Cheeks Puffed', paired('mouthCheekBlow')),
        ('Nose Wrinkled', paired('noseWrinkle')),
        ('Open Smile', {'jawOpen': .6, **paired('mouthCornerPull')}),
        ('Open Pucker', {'jawOpen': .6, **lips('mouthLipsPurse')}),
    ]
    result = []
    for name, controls in recipes:
        native = {prefix + key: value for key, value in controls.items()}
        # Custom DNA may omit controls. Never silently publish a half-pose.
        if native.keys() <= available:
            result.append({'name': name, 'controls': native})
    return result
