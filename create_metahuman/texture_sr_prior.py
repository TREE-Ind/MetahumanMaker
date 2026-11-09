"""Local PiSA-SR prior using the existing Torch/Diffusers environment.

Independent adapter for the authors' residual inference, with sequential GPU
residency. Never writes production textures. No network/model resolution.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import time


def expected(folder):
    """Read declared local identity cheaply on Blender's main thread."""
    folder=Path(folder).resolve()
    manifest=json.loads((folder/'provision.json').read_text())
    if manifest.get('status')!='verified-local-not-benchmarked':raise ValueError('Provision the pinned local PiSA-SR components first')
    result={}
    for item in manifest['assets']:
        path=(folder/item['file']).resolve()
        if not path.is_relative_to(folder) or not path.is_file() or path.stat().st_size!=item['bytes']:
            raise ValueError('Missing or changed local SR component: '+item['file'])
        if item['file'] in result or len(item['sha256'])!=64:raise ValueError('Invalid local SR manifest')
        result[item['file']]=item['sha256']
    if 'pisa_sr.pkl' not in result:raise ValueError('The local PiSA-SR adapter is missing')
    return result


def verify(folder):
    manifest=json.loads((folder/'provision.json').read_text())
    if manifest.get('status')!='verified-local-not-benchmarked':
        raise ValueError('Provision and verify the pinned SR components first')
    for item in manifest['assets']:
        path=(folder/item['file']).resolve()
        if not path.is_relative_to(folder.resolve()):raise ValueError('Invalid model path')
        with path.open('rb') as stream:actual=hashlib.file_digest(stream,'sha256').hexdigest()
        if actual!=item['sha256']:raise ValueError('Model changed: '+item['file'])
    return manifest


def run(args):
    os.environ.update(HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',HF_HUB_DISABLE_TELEMETRY='1')
    import numpy as np
    from PIL import Image
    import torch
    from diffusers import AutoencoderKL,UNet2DConditionModel
    from transformers import CLIPTextModel,CLIPTokenizer
    torch.set_num_threads(4)
    if not torch.cuda.is_available():raise RuntimeError('CUDA required')
    folder=Path(args.model).resolve();manifest=verify(folder);base=folder/'base'
    if getattr(args,'expected',None) is not None and expected(folder)!=args.expected:
        raise ValueError('The local SR model identity changed after submission')
    output=Path(args.output).resolve();output.mkdir(parents=True,exist_ok=True)
    started=time.monotonic();report={'model':'csslc/PiSA-SR','base_revision':manifest['base_revision'],
        'adapter_sha256':manifest['assets'][-1]['sha256'],'pixel_strength':args.pixel,
        'semantic_strength':args.semantic,'seed':args.seed,'accepted':False,
        'implementation':'independent Diffusers adapter, streamed LoRA merge; sequential GPU residency'}
    def stage(value):
        from resource_budget import process_memory
        report['process_memory']=process_memory()
        report.update(stage=value,elapsed_seconds=time.monotonic()-started)
        (output/'report.json').write_text(json.dumps(report,indent=2));print(value,flush=True)
        if getattr(args,'progress',None):args.progress(0,1,'PiSA-SR: '+value)
    stage('loading-local-models')
    kwargs=dict(local_files_only=True,use_safetensors=True,variant='fp16')
    tokenizer=CLIPTokenizer.from_pretrained(base/'tokenizer',local_files_only=True)
    text=CLIPTextModel.from_pretrained(base/'text_encoder',torch_dtype=torch.float16,**kwargs).eval()
    with torch.inference_mode():
        tokens=tokenizer(args.prompt,max_length=tokenizer.model_max_length,padding='max_length',truncation=True,return_tensors='pt')
        text.to('cuda');embeddings=text(tokens.input_ids.to('cuda'))[0];text.to('cpu');del text
        torch.cuda.empty_cache()
    unet=UNet2DConditionModel.from_pretrained(base/'unet',torch_dtype=torch.float16,**kwargs).eval()
    vae=AutoencoderKL.from_pretrained(base/'vae',torch_dtype=torch.float32,**kwargs).eval()
    unet.requires_grad_(False);vae.requires_grad_(False);vae.enable_tiling()
    adapter=torch.load(folder/'pisa_sr.pkl',map_location='cpu',weights_only=True)
    weights=adapter['state_dict_unet'];parameters=dict(unet.named_parameters())
    consumed=set();groups={};expanded_bytes=0
    for kind in ('pix','sem'):
        group=[];names=set();rank=adapter['lora_rank_unet_'+kind]
        for key,a in weights.items():
            if '.lora_A.' not in key or not key.endswith('_'+kind+'.weight'):continue
            bkey=key.replace('.lora_A.','.lora_B.');b=weights[bkey]
            name=key.split('.lora_A.')[0]+'.weight'
            if name not in parameters:raise ValueError('Unknown LoRA layer: '+name)
            if name in names:raise ValueError('Ambiguous adapter group: '+name)
            names.add(name);group.append((name,a,b,rank));consumed.update((key,bkey))
            expanded_bytes+=parameters[name].numel()*4
        groups[kind]=group
    if consumed!=set(weights):raise ValueError('Unconsumed adapter tensors: '+str(set(weights)-consumed))
    report['adapter_tensor_count']=len(consumed)
    report['streamed_lora_merge']=True
    report['expanded_adapter_bytes_not_materialized']=expanded_bytes
    def merge(kind):
        # Keep only low-rank factors; two full FP32 expanded adapter dictionaries
        # would compete with the image/latent buffers for several GiB of commit.
        with torch.no_grad():
            for name,a,b,rank in groups[kind]:
                parameter=parameters[name]
                delta=(b.float().reshape(b.shape[0],b.shape[1])@a.float().reshape(a.shape[0],-1)).reshape(parameter.shape)*(8/rank)
                parameter.add_(delta.to(device=parameter.device,dtype=parameter.dtype))
    stage('merging-pixel-prior');merge('pix')
    del groups['pix']
    image=Image.open(args.input).convert('RGB');size=(image.width*args.scale,image.height*args.scale)
    if any(x%8 for x in size):raise ValueError('Output dimensions must be multiples of eight')
    enlarged=image.resize(size,Image.Resampling.BICUBIC)
    control=torch.from_numpy(np.array(enlarged).transpose(2,0,1).copy()).unsqueeze(0).float().div_(127.5).sub_(1)
    report.update(input_size=image.size,output_size=size)
    stage('encoding')
    with torch.inference_mode():
        vae.to('cuda');torch.manual_seed(args.seed)
        latent=vae.encode(control.to('cuda')).latent_dist.sample()*vae.config.scaling_factor
        latent=latent.to(torch.float16);vae.to('cpu');del control;torch.cuda.empty_cache()
        unet.to('cuda');timestep=torch.tensor([1],device='cuda',dtype=torch.long)
        def predict():
            # Small overlapping latent tiles bound attention/activation memory.
            tile=min(args.tile,*latent.shape[-2:]);stride=max(8,tile-32)
            def starts(length):return sorted(set(list(range(0,max(1,length-tile+1),stride))+[length-tile]))
            yy,xx=torch.meshgrid(torch.linspace(-1,1,tile,device='cuda'),torch.linspace(-1,1,tile,device='cuda'),indexing='ij')
            window=torch.exp(-3*(xx*xx+yy*yy))[None,None]
            total=torch.zeros_like(latent,dtype=torch.float32);denom=torch.zeros_like(latent[:,:1],dtype=torch.float32)
            rows,columns=starts(latent.shape[-2]),starts(latent.shape[-1]);count=0
            for y in rows:
                for x in columns:
                    prediction=unet(latent[:,:,y:y+tile,x:x+tile],timestep,encoder_hidden_states=embeddings).sample
                    total[:,:,y:y+tile,x:x+tile]+=prediction.float()*window
                    denom[:,:,y:y+tile,x:x+tile]+=window
                    count+=1
                    if getattr(args,'progress',None) and (count%8==0 or count==len(rows)*len(columns)):
                        args.progress(count,len(rows)*len(columns),'PiSA-SR: '+report['stage'])
            return (total/denom).to(latent.dtype)
        stage('pixel-residual')
        pixel=predict()
        if args.semantic:
            stage('semantic-residual')
            merge('sem')
            semantic=predict();pred=args.pixel*pixel+args.semantic*(semantic-pixel)
        else:pred=args.pixel*pixel
        del unet,parameters,groups,weights,adapter;torch.cuda.empty_cache()
        stage('decoding')
        vae.to('cuda');decode_input=(latent-pred).float()/vae.config.scaling_factor
        if max(size)>1024:
            from texture_sr_decode import decode
            report['decoder']=decode(vae,decode_input,output/'proposal.png',size,getattr(args,'progress',None))
        else:
            result=vae.decode(decode_input).sample
            if not torch.isfinite(result).all():raise ValueError('Non-finite SR output')
            array=(result[0].clamp(-1,1).add(1).mul(127.5).round().byte().cpu().permute(1,2,0).numpy())
            Image.fromarray(array).save(output/'proposal.png')
    report['peak_allocated_gpu_bytes']=torch.cuda.max_memory_allocated()
    del vae,latent,pixel,embeddings
    torch.cuda.empty_cache()
    stage('proposal-ready-for-review')
    return report


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--model',default='models/albedo-superresolution/pisa-sr')
    parser.add_argument('--input',required=True);parser.add_argument('--output',required=True)
    parser.add_argument('--scale',type=int,choices=(2,4),default=2)
    parser.add_argument('--pixel',type=float,default=1);parser.add_argument('--semantic',type=float,default=1)
    parser.add_argument('--seed',type=int,default=42);parser.add_argument('--tile',type=int,default=64)
    parser.add_argument('--prompt',default='')
    run(parser.parse_args())
