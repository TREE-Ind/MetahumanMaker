"""Local bounded source-eye projection worker, with no Blender imports."""
import json
import sys
import time
import traceback
from pathlib import Path
from worker_lifetime import watch_parent


def main():
    folder = Path(sys.argv[1]).resolve()
    request = json.loads((folder/'request.json').read_text())
    watch_parent(request['parent'])
    def write(report):
        tmp = folder/'report.tmp';tmp.write_text(json.dumps(report,indent=2));tmp.replace(folder/'report.json')
    report = dict(status='running',eyes=[],started=time.time())
    try:
        try:
            import numpy as np
            from PIL import Image
            from eye_color_core import project
        except ModuleNotFoundError as exc:
            raise ValueError('Source eye appearance needs NumPy, SciPy and Pillow in the configured local inference Python. Set up the optional surface solver environment, then retry.') from exc
        with np.load(folder/'input.npz',allow_pickle=False) as archive:
            data = {key:archive[key] for key in archive.files}
        for eye in range(2):
            def progress(value):
                report.update(eye=eye,progress=(eye+value)/2);write(report)
            pixels, row = project(data,request['materials'],eye,size=request['resolution'],progress=progress,
                                  hidden_pigment=request.get('hidden_pigment','SCLERA_ONLY'))
            color=pixels[:,:,:3]
            encoded = np.where(color <= .0031308,color*12.92,1.055*np.maximum(color,0)**(1/2.4)-.055)
            filename = 'eye_'+str(eye)+'.png'
            Image.fromarray(np.rint(np.clip(encoded[::-1],0,1)*255).astype(np.uint8)).save(folder/filename)
            finish=np.stack((np.ones(pixels.shape[:2],np.float32),pixels[:,:,3],pixels[:,:,4]),axis=2)
            finish_name='eye_'+str(eye)+'_MR.png'
            Image.fromarray(np.rint(np.clip(finish[::-1],0,1)*255).astype(np.uint8)).save(folder/finish_name)
            row.update(finish_file=finish_name,roughness_range=[float(pixels[:,:,3].min()),float(pixels[:,:,3].max())],
                       metallic_range=[float(pixels[:,:,4].min()),float(pixels[:,:,4].max())])
            row['file'] = filename;report['eyes'].append(row)
            del pixels,encoded
        report.update(status='complete',progress=1.,finished=time.time());write(report)
    except Exception as exc:
        report.update(status='failed',error=str(exc),traceback=traceback.format_exc(),finished=time.time());write(report)
        raise


if __name__ == '__main__':main()
