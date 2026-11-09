"""Continue packed material AO through explicitly obsolete anatomical shading."""
import numpy as np


def repair(source, mask):
    from .skin_region_continuation import normal_reference
    if source.dtype!=np.uint8 or source.ndim!=3 or source.shape[2]!=4:
        raise ValueError('Use a byte RGBA bent-normal/AO texture; alpha stores AO')
    if mask.shape!=source.shape[:2]:raise ValueError('AO repair selection must match the packed texture')
    alpha=np.repeat(source[:,:,3,None],3,axis=2)
    filled,report=normal_reference(alpha,mask,scalar=True)
    result=source.copy();result[:,:,3]=filled[:,:,0]
    assert np.array_equal(result[:,:,:3],source[:,:,:3])
    assert np.array_equal(result[mask==0],source[mask==0])
    return result,dict(report,method='PACKED_AO_BOUNDARY_CONTINUATION',bent_normal_rgb_bytes_exact=True)
