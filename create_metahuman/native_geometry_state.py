"""Track only geometry actually included in a completed native delivery."""
import json

KEY='cm_native_geometry_delivery'
REVIEW='cm_native_character_review'


def read(scene,key):
    raw=scene.get(key,'')
    if not raw:return {}
    try:return json.loads(raw)
    except (TypeError,ValueError):return {}


def capture(context,roles):
    from . import mesh_ops
    from .vertex_contract import fingerprint
    s,f=context.scene.cm_settings,context.scene.cm_features
    objects={'head':s.result,'eye_left':f.eye_left,'eye_right':f.eye_right,'teeth':f.teeth}
    return {'unit_scale':context.scene.unit_settings.scale_length,
            'roles':{role:fingerprint(mesh_ops.evaluated_coordinates(context,objects[role]))
                     for role in roles if objects.get(role)}}


def known_problem(scene):
    review=read(scene,REVIEW)
    if review.get('asset')==scene.cm_features.unreal_character and review.get('native_refresh_needed'):
        return 'The linked UE character predates local geometry changes. Create a Head Revision before assembling the updated character.'
    return ''


def check(context):
    scene=context.scene;problem=known_problem(scene)
    if problem:return problem
    state=read(scene,KEY)
    if state.get('asset')!=scene.cm_features.unreal_character or not state.get('witness'):return ''
    if not state.get('within_native_precision',False):
        return 'Native evaluation changed the delivered head geometry. Review the head revision geometry report before assembly.'
    witness=state['witness']
    if capture(context,witness['roles'])!=witness:
        return 'Local head or anatomy changed after UE delivery. Create a Head Revision before assembling the updated character.'
    return ''


def adopt(scene,receipt):
    """A status/map receipt cannot acknowledge a geometry transfer."""
    if receipt.get('action')!='HEAD_REVISION' or receipt.get('status')!='complete':return
    witness=receipt.get('local_geometry_witness')
    if not witness or not witness.get('roles'):return
    scene[KEY]=json.dumps({'schema':1,'asset':receipt['character'],'witness':witness,
                          'request_sha256':receipt['request_sha256'],
                          'within_native_precision':receipt.get('head_geometry',{}).get('within_native_precision') is True})
    # The preceding asset's review remains historical; it must not label the
    # new asset as stale. check() still compares the actual current geometry.


def guard(context):
    problem=check(context)
    if problem:
        context.scene.cm_features.unreal_assembly_ready=False
        raise ValueError(problem)
