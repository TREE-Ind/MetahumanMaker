"""Optional local NCNN execution and bounded native-resolution skin candidates."""
import ctypes
import hashlib
import os
from pathlib import Path
import subprocess
import tempfile

import numpy as np
from PIL import Image
from scipy import ndimage
from texture_restore_core import restore, linear, reduce,prepare_model_input


def run_owned(command, log):
    """Cancel the owned inference process, including when this worker exits."""
    job = None
    kernel = None
    if os.name == 'nt':
        from ctypes import wintypes

        class Basic(ctypes.Structure):
            _fields_ = [('process_time', ctypes.c_int64), ('job_time', ctypes.c_int64),
                        ('flags', wintypes.DWORD), ('min_working', ctypes.c_size_t),
                        ('max_working', ctypes.c_size_t), ('active', wintypes.DWORD),
                        ('affinity', ctypes.c_size_t), ('priority', wintypes.DWORD),
                        ('scheduling', wintypes.DWORD)]

        class Extended(ctypes.Structure):
            _fields_ = [('basic', Basic), ('io', ctypes.c_uint64 * 6),
                        ('process_memory', ctypes.c_size_t), ('job_memory', ctypes.c_size_t),
                        ('peak_process', ctypes.c_size_t), ('peak_job', ctypes.c_size_t)]

        kernel = ctypes.WinDLL('kernel32', use_last_error=True)
        kernel.CreateJobObjectW.argtypes = [ctypes.c_void_p, wintypes.LPCWSTR]
        kernel.CreateJobObjectW.restype = wintypes.HANDLE
        kernel.SetInformationJobObject.argtypes = [wintypes.HANDLE, ctypes.c_int, ctypes.c_void_p, wintypes.DWORD]
        kernel.AssignProcessToJobObject.argtypes = [wintypes.HANDLE, wintypes.HANDLE]
        kernel.CloseHandle.argtypes = [wintypes.HANDLE]
        job = kernel.CreateJobObjectW(None, None)
        limits = Extended()
        limits.basic.flags = 0x2000  # JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
        if not job or not kernel.SetInformationJobObject(job, 9, ctypes.byref(limits), ctypes.sizeof(limits)):
            if job:
                kernel.CloseHandle(job)
            raise OSError(ctypes.get_last_error(), 'Cannot own the inference process')
    process = None
    try:
        with log.open('wb') as stream:
            process = subprocess.Popen(command, stdout=stream, stderr=subprocess.STDOUT,
                                       creationflags=0x08000000 if os.name == 'nt' else 0)
            if job and not kernel.AssignProcessToJobObject(job, int(process._handle)):
                raise OSError(ctypes.get_last_error(), 'Cannot attach inference to its cancellation owner')
            status = process.wait()
            if status:
                raise RuntimeError(f'Inference exited with {status}; see {log.name}')
    finally:
        if process and process.poll() is None:
            process.terminate()
            process.wait()
        if job:
            kernel.CloseHandle(job)


def fingerprint(backend,prior='RESTORE',progress=None):
    backend = Path(backend).resolve()
    names = ('realesrgan-ncnn-vulkan.exe', 'models/realesrgan-x4plus.bin',
             'models/realesrgan-x4plus.param')
    if prior=='EDIT':
        names=('runtime/sd-cli.exe','flux-2-klein-4b-Q4_0.gguf','Qwen3-4B-Q4_K_M.gguf',
               'full_encoder_small_decoder.safetensors')
    result = {};done=0;reported=0
    total=sum((backend/name).stat().st_size for name in names)
    if progress:progress(0,max(1,(total+1048575)//1048576),'Verifying texture models (MiB)')
    for name in names:
        with (backend/name).open('rb') as stream:
            digest=hashlib.sha256()
            while True:
                chunk=stream.read(8*1024**2)
                if not chunk:break
                digest.update(chunk);done+=len(chunk)
                if progress and (done-reported>=64*1024**2 or done==total):
                    progress((done+1048575)//1048576,max(1,(total+1048575)//1048576),'Verifying texture models (MiB)')
                    reported=done
            result[name]=digest.hexdigest()
    return result


def candidate(source, mask, valid, *, backend, expected, folder, strength=.65,
              invention=.35, gpu=-1, protect=None, progress=None,prior='RESTORE',brief='Fine organic skin detail',preserve_structure=True,patterns=False):
    """Use 2x model detail locally, resolve back to the delivered atlas size.

    Never allocate a whole 8K learned atlas. Overlapping 512px crops provide
    context; only selected, UV-covered, unprotected skin receives replacement.
    """
    backend, folder = Path(backend).resolve(), Path(folder).resolve()
    if prior not in {'RESTORE','EDIT'}:raise ValueError('Unknown texture prior')
    if fingerprint(backend,prior,progress) != expected:
        raise ValueError('The restoration backend changed since submission')
    if not 0 <= invention <= 1:
        raise ValueError('Texture invention must be between zero and one')
    h, w = source.shape[:2]
    allow = np.clip(mask, 0, 1)*np.clip(valid, 0, 1)
    border_distance=ndimage.distance_transform_edt(np.pad(valid>.5,1))[1:-1,1:-1]
    allow*=np.clip((border_distance-2)/6,0,1)
    if protect is not None:
        allow *= 1-np.clip(protect, 0, 1)
    original = linear(source)
    if invention == 0 or strength == 0:
        return original,np.zeros((h,w),np.float32),{
            'method':'LEARNED','invention':invention,'inference_skipped':True,
            'generated_detail_is_ground_truth':False,'physical_albedo_recovered':False,
            'uv_seam_continuity_verified':False,'outside_selected_skin_max_delta':0.}
    sums = np.zeros_like(original)
    weights = np.zeros((h, w), np.float32)
    evidence = np.zeros((h, w), np.float32)

    def starts(length):
        return list(range(0, max(length-512, 0), 384))+[max(length-512, 0)]

    tiles = [(x, y) for y in starts(h) for x in starts(w)
             if np.any(allow[y:y+512, x:x+512] > 0)]
    if not tiles:
        raise ValueError('The reconstruction mask contains no unprotected skin')
    details = []
    with tempfile.TemporaryDirectory(prefix='learned-', dir=folder) as scratch_name:
        scratch = Path(scratch_name)
        for index, (x, y) in enumerate(tiles):
            if progress:progress(index,len(tiles))
            at = (slice(y, min(y+512, h)), slice(x, min(x+512, w)))
            src = source[at]
            rows, cols = src.shape[:2]
            context=src;conditioning=None
            if patterns and prior=='EDIT':
                context,conditioning=prepare_model_input(src,valid=valid[at],
                    protect=protect[at] if protect is not None else None)
            if prior=='EDIT':
                from texture_edit_prior import propose
                prompt=('Restore this exact skin texture patch. Preserve existing wrinkle, scar and pigmentation positions. '
                    'Remove checker patterns and projection smearing. Flat evenly illuminated base color, '
                    'no added shadows, highlights, objects or text. Permitted detail: '+brief)
                # Small edge crops are padded for the model's 16px token grid,
                # then cropped back before source constraints are applied.
                pad_h=(-rows)%16;pad_w=(-cols)%16
                padded=np.pad(context,((0,pad_h),(0,pad_w),(0,0)),mode='edge')
                proposal,prior_report=propose(padded,scratch/('prior-'+str(index)),backend=backend,prompt=prompt,verified=expected,
                    cache=folder.parent/'.prior-cache')
                proposal=proposal[:rows*2,:cols*2]
            else:
                Image.fromarray(np.rint(src*255).astype('uint8')).save(scratch/'input.png')
                command = [str(backend/'realesrgan-ncnn-vulkan.exe'), '-i', str(scratch/'input.png'),
                           '-o', str(scratch/'raw.png'), '-m', str(backend/'models'),
                           '-n', 'realesrgan-x4plus', '-s', '4', '-t', '64', '-j', '1:1:1']
                if gpu >= 0:
                    command += ['-g', str(gpu)]
                run_owned(command, folder/'inference.log')
                with Image.open(scratch/'raw.png') as raw:
                    if raw.size != (cols*4, rows*4):
                        raise ValueError('The restoration model returned an unexpected size')
                    proposal = np.asarray(raw.convert('RGB').resize((cols*2, rows*2),
                                          Image.Resampling.LANCZOS), dtype=np.float32)/255
            result, residual, info = restore(src, proposal, valid=valid[at],
                protect=protect[at] if protect is not None else None,
                strength=strength*invention, detail_limit=.025+.225*invention,
                periodic=mask[at], artifact_repair=True,
                trusted_band=.025+(.075 if prior!='EDIT' or preserve_structure else .025)*(1-invention),
                pattern_exception=patterns and prior=='EDIT')
            native = reduce(linear(result), 2)
            feather = np.maximum(np.hanning(rows), .01)[:, None]*np.maximum(np.hanning(cols), .01)[None, :]
            sums[at] += native*feather[:, :, None]
            weights[at] += feather
            local_evidence = residual.reshape(rows, 2, cols, 2).mean(axis=(1, 3))
            evidence[at] += local_evidence*feather
            details.append(dict(info, x=x, y=y))
            if prior=='EDIT':details[-1]['editing_prior']=prior_report
            if conditioning:details[-1]['model_context']=conditioning
            if progress:
                progress(index+1, len(tiles))
    covered = weights > 0
    sums[covered] /= weights[covered, None]
    evidence[covered] /= weights[covered]
    result = original+(sums-original)*allow[:, :, None]
    result[allow <= 0] = original[allow <= 0]
    evidence *= allow
    if fingerprint(backend,prior,progress) != expected:
        raise ValueError('The restoration backend changed during inference')
    return np.clip(result, 0, 1), evidence, {
        'method': 'LEARNED', 'backend': expected, 'tiles': details,
        'backend_verification':'worker_sha256_before_and_after',
        'prior':prior,'detail_brief':brief if prior=='EDIT' else None,
        'invention':invention,'cleanup_strength':strength,
        'output_resolution': [w, h], 'prior_scale': 2,
        'generated_detail_is_ground_truth': False, 'physical_albedo_recovered': False,
        'uv_seam_continuity_verified': False, 'native_atlas_resolution_retained': True,
        'uv_border_pixels_retained':True,
        'outside_selected_skin_max_delta': float(np.abs(result[allow <= 0]-original[allow <= 0]).max())
            if np.any(allow <= 0) else 0.}
