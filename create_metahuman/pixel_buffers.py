"""Owned disk-backed scratch buffers for large, sequential image operations."""
from contextlib import contextmanager
from pathlib import Path
import tempfile
import numpy as np


@contextmanager
def scratch(shape,*,directory=None):
    count=int(np.prod(shape))
    if count<=4096*4096*4:
        yield np.empty(shape,dtype=np.float32)
        return
    if directory is not None:Path(directory).mkdir(parents=True,exist_ok=True)
    from . import resource_budget
    resource_budget.require_disk('large image scratch',directory or tempfile.gettempdir(),count*4+2*resource_budget.GIB)
    with tempfile.TemporaryDirectory(prefix='cm-pixels-',dir=directory) as folder:
        pixels=np.memmap(Path(folder)/'scratch.bin',dtype=np.float32,mode='w+',shape=shape)
        try:yield pixels
        finally:
            # Windows cannot unlink an open mapping. Callers must finish Blender
            # foreach_get/set and release derived views before leaving the scope.
            pixels.flush();pixels._mmap.close()


@contextmanager
def image_pixels(image,*,directory=None):
    with scratch((image.size[1],image.size[0],4),directory=directory) as pixels:
        image.pixels.foreach_get(pixels.ravel())
        yield pixels
