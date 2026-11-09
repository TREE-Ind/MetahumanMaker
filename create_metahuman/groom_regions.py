"""Growth regions anchored to native face weights, with editable-mask priority."""
import numpy as np


def masks(context, head, points):
    from . import workflow
    categories = {}
    for group in head.vertex_groups:
        name = group.name.lower()
        category = ('upper' if 'lipupper' in name and 'skin' in name else
                    'lower' if 'liplower' in name and 'skin' in name else
                    'chin' if 'chin' in name else
                    'eye' if 'eyelidupper' in name else
                    'forehead' if 'forehead' in name else None)
        if category and name.startswith('facial_'):
            categories[group.index] = category
    weights = {key: np.zeros(len(points)) for key in ('upper','lower','chin','eye','forehead')}
    # Visit deform weights once; do not scan every vertex for every facial joint.
    for vertex in head.data.vertices:
        for item in vertex.groups:
            key = categories.get(item.group)
            if key:
                weights[key][vertex.index] = max(weights[key][vertex.index], item.weight)
    centers = {}
    for key, value in weights.items():
        if np.count_nonzero(value > .05) >= 3:
            centers[key] = np.average(points[value > .05], axis=0, weights=value[value > .05])
    axis_name = context.scene.cm_features.front_axis
    axis = 1 if axis_name in {'Y','-Y'} else 0
    lateral = 1-axis
    sign = -1 if axis_name.startswith('-') else 1
    lo, hi = points.min(0), points.max(0)
    extent = np.maximum(hi-lo,1e-8)
    front = sign*(points[:,axis]-np.mean(points[:,axis])) > 0
    z = points[:,2]
    native = all(key in centers for key in ('upper','lower','chin','eye','forehead'))
    if native:
        eye, mouth, chin = centers['eye'][2], centers['upper'][2], centers['chin'][2]
        face_height = max(eye-chin,.15*extent[2])
        brow = eye+.12*face_height
        middle = (lo[lateral]+hi[lateral])*.5
        side = np.abs(points[:,lateral]-middle)
        # The head mesh can include a wide upper chest. Measure face width from
        # facial skin weights rather than treating the shoulders as cheek width.
        face_width=min(extent[lateral],max(.25*extent[lateral],
            2.8*np.percentile(side[weights['forehead']>.05],90)))
        result = {
            'HAIR': z > np.where(front,brow+.20*face_height,eye-.12*face_height),
            'BROWS': front & (z>brow-.07*face_height) & (z<brow+.10*face_height)
                     & (side>.06*face_width) & (side<.34*face_width),
            'MUSTACHE': front & (weights['upper']>.055) & (z>mouth-.08*face_height)
                        & (z<mouth+.17*face_height),
            'BEARD': front & (z>chin-.22*face_height)
                     & (z<np.where(side<.18*face_width,centers['lower'][2]-.02*face_height,mouth+.06*face_height))
                     & (side<.40*face_width),
        }
    else:
        normalized = (z-lo[2])/extent[2]
        result = {key:(normalized>low)&(normalized<high)&(front if key!='HAIR' else True)
                  for key,(low,high) in {'HAIR':(.72,1.1),'BROWS':(.56,.76),
                                        'MUSTACHE':(.32,.48),'BEARD':(.14,.43)}.items()}
    for region in result:
        name = 'CM_Groom_'+region
        if head.vertex_groups.get(name):
            result[region] = workflow.group_weights(head,name)>.5
    return result, 'NATIVE_FACE_WEIGHTS' if native else 'SPATIAL_ESTIMATE'
