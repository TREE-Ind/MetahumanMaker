"""Interpret inferred skin roughness as detail around native regional response."""
from pathlib import Path
import bpy
import numpy as np


def profiles(scene, *, roles=None):
    from . import map_provenance
    s,f=scene.cm_settings,scene.cm_features
    result={};root=Path(bpy.path.abspath(s.project_dir))/'Maps'
    entries=map_provenance.load(scene)['entries']
    for role in (('HEAD','BODY') if s.part=='FULL_BODY' else (s.part,)):
        if roles is not None and role not in roles:continue
        prefix=s.character_name+('_Body' if role=='BODY' and s.part=='FULL_BODY' else '')
        path=root/(prefix+'_ROUGHNESS.png')
        if not path.is_file():continue
        record=entries.get(map_provenance.identity(scene,'ROUGHNESS_'+role),{}).get('artifacts',{}).get(path.name,{})
        if record:
            import hashlib
            with path.open('rb') as stream:current=hashlib.file_digest(stream,'sha256').hexdigest()
            if record.get('sha256')!=current:record={}
        from . import skin_projection_history
        if skin_projection_history.current_projection(scene,role,'ROUGHNESS'):record={}
        choice=f.roughness_response_mode
        mode='SKIN_DETAIL' if choice=='SKIN_DETAIL' or (choice=='AUTO' and record.get('recipe')=='ALBEDO_MATERIAL') else 'ABSOLUTE'
        if mode=='ABSOLUTE':result[role]={'mode':mode};continue
        obj=f.body_result if role=='BODY' and s.part=='FULL_BODY' else s.result
        if not obj or not obj.data.uv_layers.active:raise ValueError('Native skin roughness detail needs target UVs')
        uv=np.empty((len(obj.data.uv_layers.active.data),2),np.float32)
        obj.data.uv_layers.active.data.foreach_get('uv',uv.ravel())
        if not len(uv):raise ValueError('Skin roughness has no UV samples')
        uv=uv[np.linspace(0,len(uv)-1,min(4096,len(uv)),dtype=np.int32)]%1
        # Apply/Revert can leave unused packed images with the same filepath.
        # Calibrate from the saved bytes being delivered, never an older buffer.
        image=bpy.data.images.load(str(path),check_existing=False)
        try:
            image.colorspace_settings.name='Non-Color'
            w,h=image.size
            x=np.clip((uv[:,0]*w).astype(int),0,w-1);y=np.clip((uv[:,1]*h).astype(int),0,h-1)
            # RNA indexed reads can copy the entire image for every sample.
            # One bulk read is bounded to 1 GiB for an 8K RGBA float image.
            pixels=np.empty(len(image.pixels),np.float32)
            image.pixels.foreach_get(pixels)
            values=pixels[(y*w+x)*image.channels].copy()
            del pixels
            values=values[np.isfinite(values)&(values>=.02)&(values<=1)]
            if len(values)<32:raise ValueError('Not enough valid skin roughness samples for native response')
            center=float(np.median(values))
            result[role]={'mode':mode,'center':center,'samples':len(values),'estimator':'median_target_uv_samples',
                          'finish_roughness':f.skin_finish_roughness,
                          'source_recipe':record.get('recipe','ARTIST')}
        finally:
            if image.users==0:bpy.data.images.remove(image)
    return result
