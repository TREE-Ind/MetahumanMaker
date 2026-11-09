"""SHA-256 of contiguous writable buffers without a full-size bytes copy.

Windows CNG avoids the slow bulk SHA implementation in some Blender builds.
The digest remains standard SHA-256; other platforms use hashlib directly.
"""
import ctypes
from functools import lru_cache
import hashlib
import os


@lru_cache(maxsize=1)
def _cng():
    if os.name!='nt':return None
    try:
        api=ctypes.WinDLL('bcrypt.dll')
        handle=ctypes.c_void_p;ulong=ctypes.c_ulong
        signatures={
            'BCryptOpenAlgorithmProvider':[ctypes.POINTER(handle),ctypes.c_wchar_p,ctypes.c_wchar_p,ulong],
            'BCryptCreateHash':[handle,ctypes.POINTER(handle),handle,ulong,handle,ulong,ulong],
            'BCryptHashData':[handle,handle,ulong,ulong],
            'BCryptFinishHash':[handle,handle,ulong,ulong],
            'BCryptDestroyHash':[handle],
            'BCryptCloseAlgorithmProvider':[handle,ulong]}
        for name,args in signatures.items():
            function=getattr(api,name);function.argtypes=args;function.restype=ctypes.c_long
        return api
    except (OSError,AttributeError):return None


def sha256_buffer(data,metadata=b''):
    view=memoryview(data).cast('B')
    api=_cng()
    if api is not None and not view.readonly:
        algorithm=ctypes.c_void_p();handle=ctypes.c_void_p()
        def check(status):
            if status!=0:raise OSError('Windows SHA-256 failed: '+hex(status&0xffffffff))
        try:
            check(api.BCryptOpenAlgorithmProvider(ctypes.byref(algorithm),'SHA256',None,0))
            check(api.BCryptCreateHash(algorithm,ctypes.byref(handle),None,0,None,0,0))
            if view.nbytes:
                buffer=(ctypes.c_ubyte*view.nbytes).from_buffer(view)
                address=ctypes.addressof(buffer)
                for offset in range(0,view.nbytes,1024**3):
                    check(api.BCryptHashData(handle,address+offset,min(1024**3,view.nbytes-offset),0))
            if metadata:
                suffix=ctypes.create_string_buffer(metadata)
                check(api.BCryptHashData(handle,ctypes.addressof(suffix),len(metadata),0))
            output=(ctypes.c_ubyte*32)()
            check(api.BCryptFinishHash(handle,ctypes.addressof(output),32,0))
            return bytes(output).hex()
        except OSError:
            pass  # Standard implementation remains the portability fallback.
        finally:
            if handle.value:api.BCryptDestroyHash(handle)
            if algorithm.value:api.BCryptCloseAlgorithmProvider(algorithm,0)
    digest=hashlib.sha256(view)
    digest.update(metadata)
    return digest.hexdigest()
