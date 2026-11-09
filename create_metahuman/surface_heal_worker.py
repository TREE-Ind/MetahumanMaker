"""Owned background surface-color reference generation; never edits project maps."""
from pathlib import Path
import sys,json,hashlib,os,traceback
import numpy as np
from PIL import Image
from surface_heal_core import heal
from worker_lifetime import watch_parent


def main():
    request_path=Path(sys.argv[1]);folder=request_path.parent;request=json.loads(request_path.read_text());watch_parent(request['parent'])
    def report(value):
        path=folder/'report.json';temporary=path.with_suffix('.tmp');temporary.write_text(json.dumps(value,indent=2));temporary.replace(path)
    try:
        for name,digest in request['inputs'].items():
            if Path(name).name!=name:raise ValueError('Invalid surface-heal input filename')
            with (folder/name).open('rb') as stream:actual=hashlib.file_digest(stream,'sha256').hexdigest()
            if actual!=digest:raise ValueError('Surface-heal snapshot changed')
        original=np.asarray(Image.open(folder/'source.png').convert('RGBA'));source=original[:,:,:3];mask=np.asarray(Image.open(folder/'mask.png').convert('L'),dtype=np.float32)/255
        channel=request.get('channel','ALBEDO');component=request.get('component','GRAY');encoding='SRGB'
        if channel=='ROUGHNESS':
            encoding='LINEAR'
            if component=='GRAY':
                if not np.array_equal(source[:,:,0],source[:,:,1]) or not np.array_equal(source[:,:,0],source[:,:,2]):raise ValueError('Select R, G or B for packed roughness; grayscale mode retains no unrelated color channels')
                scalar=source[:,:,0]
            elif component in 'RGB' and len(component)==1:scalar=source[:,:,'RGB'.index(component)]
            else:raise ValueError('Choose a supported roughness component')
            source=np.repeat(scalar[:,:,None],3,axis=2)
        elif channel!='ALBEDO':raise ValueError('Choose albedo or roughness for surface healing')
        with np.load(folder/'surface.npz',allow_pickle=False) as data:
            output,details,mask=heal(source.astype(np.float32)/255,mask,data['points'],data['faces'],data['uv'],padding=12,margin=request.get('margin',0),feather=request.get('feather',0),encoding=encoding,progress=lambda phase:report(dict(status='running',phase=phase)))
        encoded=original.copy();result=np.rint(output*255).astype('uint8')
        if channel=='ROUGHNESS' and component!='GRAY':encoded[:,:,'RGB'.index(component)]=result[:,:,0]
        else:encoded[:,:,:3]=result
        encoded[mask<=0]=original[mask<=0]
        Image.fromarray(encoded).save(folder/'reference.png')
        Image.fromarray(np.rint(mask*65535).astype('uint16')).save(folder/'effective-mask.png')
        details.update(channel=channel,component=component,outside_mask_bytes_exact=bool(np.array_equal(encoded[mask<=0],original[mask<=0])),physical_roughness_recovered=False)
        if channel=='ROUGHNESS' and component!='GRAY':
            retained=[i for i in range(4) if i!='RGB'.index(component)];details['unrelated_packed_channels_exact']=bool(np.array_equal(encoded[:,:,retained],original[:,:,retained]))
        with (folder/'reference.png').open('rb') as stream:digest=hashlib.file_digest(stream,'sha256').hexdigest()
        with (folder/'effective-mask.png').open('rb') as stream:mask_digest=hashlib.file_digest(stream,'sha256').hexdigest()
        report(dict(status='complete',sha256=digest,mask_sha256=mask_digest,details=details))
    except Exception as exc:
        report(dict(status='failed',error=str(exc),traceback=traceback.format_exc()));raise


if __name__=='__main__':main()
