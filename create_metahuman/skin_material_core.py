"""Art-directed micro material response from the final linear skin albedo.

Color is not measured height or reflectance. Geometry cavity and the source
roughness remain baselines; a bounded shared color residual modifies them.
"""
import numpy as np
from PIL import Image
from scipy import ndimage


def height_cavity_response(height,strength):
    """Black-neutral artistic microcavity; height already embeds paint protection."""
    if height.ndim!=2 or not np.isfinite(height).all() or not np.isfinite(strength) or not 0<=strength<=1:
        raise ValueError('Invalid retained height cavity response')
    return np.maximum(-np.tanh(height/.6),0)*(.18*strength)


def retain_packed_channels(scalar,accepted):
    """Replace the scalar R channel while retaining native G/B/A paint."""
    if accepted.dtype!=np.uint8 or accepted.ndim!=3 or accepted.shape[2] not in (3,4) or scalar.shape!=accepted.shape[:2]:
        raise ValueError('Packed material channels must match the scalar surface')
    result=accepted.copy()
    for y in range(0,len(result),128):
        result[y:y+128,:,0]=np.rint(np.clip(scalar[y:y+128],0,1)*255).astype(np.uint8)
    return result


def preserve_region_base(candidate,accepted,mask,*,protect=None,valid=None,strength=1,normal=False,restore_only=False):
    """Use the accepted surface outside an edit; field detail is added once later.

    Accepted pixels are uint8 to bound memory while processing full 8K maps.
    Final normal replay also restores zero-weight texels exactly because normal
    renormalization and height derivatives can otherwise change their encoding.
    """
    if accepted.dtype!=np.uint8 or accepted.shape!=candidate.shape or mask.shape!=candidate.shape[:2]:
        raise ValueError('Accepted surface and selection must match the candidate')
    retained=selected=0
    for y in range(0,len(candidate),128):
        end=min(len(candidate),y+128);base=accepted[y:end].astype(np.float32)/255
        gate=np.clip(mask[y:end],0,1)*strength
        if protect is not None:gate*=1-np.clip(protect[y:end],0,1)
        if valid is not None:gate*=np.clip(valid[y:end],0,1)
        zero=gate<=0;block=candidate[y:end]
        retained+=int(zero.sum());selected+=int((~zero).sum())
        if not restore_only:
            weight=gate[...,None] if block.ndim==3 else gate
            block*=weight;block+=base*(1-weight)
            if normal:
                vectors=block*2-1
                vectors/=np.maximum(np.linalg.norm(vectors,axis=2,keepdims=True),1e-8)
                block[:]=(vectors+1)*.5
        block[zero]=base[zero]
    return {'accepted_outside_selection_texels':retained,'selected_texels':selected,
            'outside_selection_copied_exactly':True,'restore_only':restore_only}


def split_surface(color,mask,*,balance=.7,height_strength=.7,radius=2,protect=None,valid=None,progress=None):
    """Move fine achromatic shading to a shared, bounded artistic height field.

    Scale RGB together to retain chromatic pigmentation and sharp edge positions.
    No spatial blur is applied to the output color. This is an artistic
    decomposition of generated skin, not measured intrinsic reflectance.
    """
    if not 0<=balance<=1 or not 0<=height_strength<=1 or not .5<=radius<=32:
        raise ValueError('Invalid generated surface decomposition controls')
    h,w=color.shape[:2];height=np.zeros((h,w),np.float32);maximum=0.
    halo=int(np.ceil(max(10,radius*6)*4))+2
    for y in range(0,h,128):
        end=min(h,y+128);low=max(0,y-halo);high=min(h,end+halo);offset=y-low
        rgb=np.log(np.maximum(color[low:high],.002));lum=rgb@np.asarray([.2126,.7152,.0722],np.float32)
        fine=ndimage.gaussian_filter(lum,.65)-ndimage.gaussian_filter(lum,max(3,radius*2))
        middle=ndimage.gaussian_filter(lum,max(3,radius*2))-ndimage.gaussian_filter(lum,max(10,radius*6))
        band=fine+middle
        chroma=rgb-lum[...,None];chroma-=ndimage.gaussian_filter(chroma,(4,4,0))
        pigment=1/(1+(np.linalg.norm(chroma,axis=2)/.09)**4)
        gate=np.clip(mask[low:high],0,1)*pigment
        if protect is not None:gate*=1-np.clip(protect[low:high],0,1)
        if valid is not None:
            inside=ndimage.distance_transform_edt(np.pad(valid[low:high]>.5,1))[1:-1,1:-1]
            gate*=np.clip((inside-2)/4,0,1)
        a=slice(offset,offset+end-y)
        shared=band[a]*gate[a]
        # Bound inferred depth; color-inferred pigment must not become deep pits.
        height[y:end]=np.clip(radius*height_strength*(.6*np.tanh(fine[a]/.13)+1.2*np.tanh(middle[a]/.16))*gate[a],-4,4).astype(np.float32)
        correction=np.clip(shared*balance,-.32,.32)
        color[y:end]*=np.exp(-correction[...,None])
        maximum=max(maximum,float(abs(correction).max()))
        if progress:progress(end,h,'Separating generated color and surface response')
    np.clip(color,0,1,out=color)
    return height,{'method':'shared_generated_skin_microfield','balance':balance,'height_strength':height_strength,
        'radius_output_texels':radius,'maximum_log_color_correction':maximum,
        'height_max_output_texels':float(abs(height).max()),'physical_decomposition_recovered':False,
        'output_color_spatially_blurred':False}


def derive(color,mask,roughness,cavity,*,strength=1,protect=None,progress=None,height=None):
    if not 0<=strength<=1 or color.ndim!=3 or color.shape[2]!=3:raise ValueError('Invalid material derivation controls')
    h,w=color.shape[:2]
    if mask.shape!=(h,w):raise ValueError('Material selection must match albedo')
    def plane(value):
        if value.ndim==3:value=value[...,0]
        target=(max(w,value.shape[1]),max(h,value.shape[0]))
        if value.shape!=(target[1],target[0]):value=np.asarray(Image.fromarray(value).resize(target,Image.Resampling.BILINEAR),dtype=np.float32)
        return value.copy()
    result_rough,result_cavity=plane(roughness),plane(cavity)
    mixed=result_rough.shape!=(h,w) or result_cavity.shape!=(h,w)
    response=np.empty((h,w),np.float32) if mixed else None
    maximum=0.
    for y in range(0,h,128):
        end=min(h,y+128);low=max(0,y-24);high=min(h,end+24);offset=y-low
        if height is None:
            rgb=np.log(np.maximum(color[low:high],.002))
            lum=rgb@np.asarray([.2126,.7152,.0722],np.float32)
            fine=ndimage.gaussian_filter(lum,.6)-ndimage.gaussian_filter(lum,4)
            chroma=rgb-lum[...,None];chroma-=ndimage.gaussian_filter(chroma,(4,4,0))
            pigment=1/(1+(np.linalg.norm(chroma,axis=2)/.08)**4)
            residual=(np.tanh(fine/.12)*pigment)[offset:offset+end-y]
        else:residual=np.tanh(height[y:end]/.6)
        # Shared height already contains selection, paint protection and UV
        # coverage. Applying those again would square the feathered selection.
        gate=strength if height is not None else np.clip(mask[y:end],0,1)*strength
        if height is None and protect is not None:gate*=1-np.clip(protect[y:end],0,1)
        residual=residual*gate
        delta=-residual*(.10 if height is not None else .06)
        if result_rough.shape==(h,w):result_rough[y:end]+=delta
        # Native MetaHuman cavity is black-neutral (not white-neutral AO).
        if result_cavity.shape==(h,w):result_cavity[y:end]+=np.maximum(-residual,0)*(.18 if height is not None else .12)
        if mixed:response[y:end]=residual
        maximum=max(maximum,float(abs(delta).max()))
        if progress:progress(end,h,'Deriving roughness and microcavity')
    for kind,result in (('roughness',result_rough),('cavity',result_cavity)):
        if result.shape!=(h,w):
            field=np.asarray(Image.fromarray(response).resize((result.shape[1],result.shape[0]),Image.Resampling.BILINEAR),dtype=np.float32)
            for y in range(0,len(result),128):
                block=field[y:y+128]
                result[y:y+128]+= -block*(.10 if height is not None else .06) if kind=='roughness' else np.maximum(-block,0)*(.18 if height is not None else .12)
            del field
        np.clip(result,0,1,out=result)
    return result_rough,result_cavity,{'method':'bounded_chroma_gated_albedo_microresponse','strength':strength,
        'maximum_roughness_delta':maximum,'retains_geometry_cavity':True,'physical_material_recovered':False,
        'shared_normal_height_field':height is not None,
        'roughness_resolution':[result_rough.shape[1],result_rough.shape[0]],
        'cavity_resolution':[result_cavity.shape[1],result_cavity.shape[0]]}
