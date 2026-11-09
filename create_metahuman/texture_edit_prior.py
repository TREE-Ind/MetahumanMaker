"""Optional pretrained reference-editing prior; never publishes character maps."""
import hashlib
import json
from pathlib import Path
import subprocess

import numpy as np
from PIL import Image
from texture_restore_backend import run_owned


def verify(backend,verified=None):
    backend=Path(backend).resolve()
    manifest=json.loads((backend/'provision.json').read_text(encoding='utf-8'))
    if manifest.get('status')!='verified-local-not-benchmarked':
        raise ValueError('Finish provisioning the optional texture editor first')
    for record in manifest['assets']:
        if record['name']=='runtime.zip':continue
        path=backend/record['name']
        if verified is None:
            with path.open('rb') as stream:
                actual=hashlib.file_digest(stream,'sha256').hexdigest()
        else:
            # A parent patch solve has just fingerprinted all model bytes and
            # repeats that check before returning any publishable candidate.
            actual=verified.get(record['name'])
        if actual!=record['sha256']:
            raise ValueError('Texture editor weights changed: '+record['name'])
    if not (backend/'runtime/sd-cli.exe').is_file():
        raise ValueError('The texture editor runtime is missing')
    return manifest


def propose(source,folder,*,backend,prompt,seed=42,device='auto',max_vram=4,cache=None,verified=None,init_strength=None):
    backend,folder=Path(backend).resolve(),Path(folder).resolve()
    manifest=verify(backend,verified)
    if device=='auto':
        listing=subprocess.run([str(backend/'runtime/sd-cli.exe'),'--list-devices'],
            capture_output=True,text=True,encoding='utf-8',errors='replace',timeout=20,
            creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0)).stdout
        choices=[line.split('\t',1) for line in listing.splitlines() if line.lower().startswith('vulkan') and '\t' in line]
        device=next((name.lower() for name,desc in choices if 'nvidia' in desc.lower()),
                    choices[0][0].lower() if choices else 'cpu')
    if not prompt.strip():raise ValueError('Describe the intended skin detail before editing')
    if init_strength is not None and not 0<init_strength<=1:raise ValueError('Initialization strength must be in (0,1]')
    folder.mkdir(exist_ok=False)
    h,w=source.shape[:2]
    if min(h,w)<64 or max(h,w)>1024 or w%16 or h%16:
        raise ValueError('Reference patches must be 64..1024 pixels, in multiples of 16')
    Image.fromarray(np.rint(np.clip(source,0,1)*255).astype('uint8')).save(folder/'reference.png')
    (folder/'prompt.txt').write_text(prompt,encoding='utf-8')
    key=inputs=None
    if cache is not None:
        import texture_prior_cache
        runtime_sha=hashlib.sha256((backend/'runtime/sd-cli.exe').read_bytes()).hexdigest()
        key,inputs=texture_prior_cache.identity(source,manifest,prompt,seed,runtime_sha,
            {'strength':init_strength,'resize':'bicubic','scale':2} if init_strength is not None else None)
        cached=texture_prior_cache.fetch(cache,key,folder/'proposal.png')
        if cached:
            with Image.open(folder/'proposal.png') as image:
                if image.size!=(w*2,h*2):raise ValueError('Cached proposal has an invalid size')
                result=np.asarray(image.convert('RGB'),dtype=np.float32)/255
            (folder/'inference.log').write_text('Reused verified local model proposal '+key,encoding='utf-8')
            (folder/'report.json').write_text(json.dumps(cached,indent=2),encoding='utf-8')
            return result,cached
    command=[str(backend/'runtime/sd-cli.exe'),
        '--diffusion-model',str(backend/'flux-2-klein-4b-Q4_0.gguf'),
        '--llm',str(backend/'Qwen3-4B-Q4_K_M.gguf'),
        '--vae',str(backend/'full_encoder_small_decoder.safetensors'),
        '-r',str(folder/'reference.png'),'--prompt-file',str(folder/'prompt.txt'),
        '-o',str(folder/'proposal.png'),'-W',str(w*2),'-H',str(h*2),
        '--steps','4','--cfg-scale','1','--sampling-method','euler','-s',str(seed),
        '--backend','diffusion='+device+',vae='+device+',te=cpu',
        '--max-vram',str(max_vram),'--offload-to-cpu','--vae-tiling','-t','4']
    if init_strength is not None:
        Image.open(folder/'reference.png').resize((w*2,h*2),Image.Resampling.BICUBIC).save(folder/'initial.png')
        command+=['--init-img',str(folder/'initial.png'),'--strength',str(init_strength)]
    report={'schema':1,'status':'running','backend_revision':manifest['assets'][-1]['revision'],
        'source_model':manifest['source_model'],'quantized':True,'seed':seed,'prompt':prompt,
        'device':device,'max_vram_gib':max_vram,'true_controlnet_used':False,
        'reference_conditioning':True,'physical_albedo_recovered':False,'applied_to_character':False}
    report['initialization_strength']=init_strength
    (folder/'report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    try:
        run_owned(command,folder/'inference.log')
        with Image.open(folder/'proposal.png') as image:
            if image.size!=(w*2,h*2):raise ValueError('The editor returned a different patch resolution')
            result=np.asarray(image.convert('RGB'),dtype=np.float32)/255
        report.update(status='ready',output_sha256=hashlib.sha256((folder/'proposal.png').read_bytes()).hexdigest())
        report['cache_hit']=False
        if cache is not None:
            texture_prior_cache.publish(cache,key,inputs,folder/'proposal.png',report)
        (folder/'report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
        return result,report
    except BaseException as exc:
        report.update(status='failed',error=str(exc))
        (folder/'report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
        raise
