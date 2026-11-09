"""Factorized native eye-control samples for evaluated-motion calibration."""

CONTROLS=('eyeBlink','eyeLookUp','eyeLookDown','eyeLookLeft','eyeLookRight','eyeSquintInner','eyeWiden')


def states():
    """Twenty active states plus Neutral; seven scalar inputs, 21 RBF nodes."""
    rows=[dict(name='Neutral',values=[0.]*7)]
    def add(name,**controls):rows.append(dict(name=name,values=[float(controls.get(key,0.)) for key in CONTROLS]))
    add('BlinkHalf',eyeBlink=.5);add('Blink',eyeBlink=1.)
    for gaze in ('Up','Down','Left','Right'):add(gaze,**{'eyeLook'+gaze:1.})
    for gaze in ('Up','Down','Left','Right'):
        for suffix,value in [('HalfBlink',.5),('Blink',1.)]:add(gaze+suffix,eyeBlink=value,**{'eyeLook'+gaze:1.})
    add('SquintHalf',eyeSquintInner=.5);add('Squint',eyeSquintInner=1.);add('WidenHalf',eyeWiden=.5);add('Widen',eyeWiden=1.)
    add('UpWiden',eyeLookUp=1.,eyeWiden=1.);add('DownWiden',eyeLookDown=1.,eyeWiden=1.)
    return rows


def samples():
    """Separate each side's motion and the additional bilateral interaction.

    Native scalar RBF arrays use 7*21 and 14*61 values. Neither resembles a
    packed-quaternion array in UE 5.8's DNA coordinate conversion.
    """
    rows=[dict(name='Neutral',controls={},factor='NEUTRAL',state='Neutral')]
    for factor,sides in [('L',('L',)),('R',('R',)),('Both',('L','R'))]:
        for row in states()[1:]:
            controls={key+side:value for key,value in zip(CONTROLS,row['values']) if value for side in sides}
            rows.append(dict(name=factor+row['name'],controls=controls,factor=factor,state=row['name']))
    return rows


def motion_channel(side,state):return 'CM_lod_motion_'+side+'_'+state.lower()


def interaction_channel(state):return 'CM_lod_interaction_'+state.lower()


def driver_request(*,root_mesh='head_lod0_mesh',maximum_offset=15.,radius=.8):
    """Native factorized motion plus bilateral residual, zero geometry at LOD0.

    Independent side solvers always retain their unilateral correction. The
    joint solver learns only the additional bilateral term, with explicit
    zero-output anchors at all unilateral samples.
    """
    targets=[];solvers=[]
    def target(channel):targets.append(dict(mesh=root_mesh,channel=channel,indices=[0],file='driver_zero.f32'))
    for side in ('L','R'):
        poses=[]
        for row in states():
            pose=dict(name='CM_lod_motion_pose_'+side+'_'+row['name'],values=row['values'])
            if row['name']!='Neutral':pose['channel']=motion_channel(side,row['name']);target(pose['channel'])
            poses.append(pose)
        solvers.append(dict(name='CM_lod_motion_solver_'+side,radius=radius,inputs=['CTRL_expressions.'+key+side for key in CONTROLS],poses=poses))
    poses=[]
    for sample in samples():
        pose=dict(name='CM_lod_interaction_pose_'+sample['name'],values=[sample['controls'].get(key+side,0.) for side in ('L','R') for key in CONTROLS])
        if sample['factor']=='Both':pose['channel']=interaction_channel(sample['state']);target(pose['channel'])
        poses.append(pose)
    solvers.append(dict(name='CM_lod_interaction_solver',radius=radius,inputs=['CTRL_expressions.'+key+side for side in ('L','R') for key in CONTROLS],poses=poses))
    return dict(schema='CM_DNA_POSE_CORRECTIVES_V1',maximum_offset=maximum_offset,targets=targets,solvers=solvers)
