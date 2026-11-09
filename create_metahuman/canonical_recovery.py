"""Source-registered clean anatomy recovery for collapsed regional wraps.

External NumPy/SciPy worker API. This preserves canonical finger, toe and ear
structure rather than projecting already folded vertices again. It proposes
geometry only: callers must validate and commit with native_surface_commit,
then inspect source fidelity. Similarity registration cannot reproduce every
source articulation or anatomical proportion.
"""
import numpy as np
from scipy.spatial import cKDTree
from .geometry import similarity
from .region_reconstruction import stitch


def recover(data, regions, *, collar_rings=8, iterations=500, progress=None):
    rest=np.asarray(data['base'],dtype=float)
    current=np.asarray(data['current'],dtype=float)
    source=np.asarray(data['source'],dtype=float)
    if rest.shape!=current.shape or rest.ndim!=2 or rest.shape[1]!=3:
        raise ValueError('Recovery needs matching canonical and current vertex orders')
    if not all(np.isfinite(p).all() for p in (rest,current,source)):
        raise ValueError('Recovery coordinates must be finite')
    if not 0<=collar_rings<=16 or not 1<=iterations<=2000:
        raise ValueError('Choose bounded recovery collar and iteration settings')
    _,weld=np.unique(rest,axis=0,return_inverse=True)
    counts=np.bincount(weld)
    def physical(points):
        return np.column_stack([np.bincount(weld,weights=points[:,a])/counts for a in range(3)])
    base=physical(rest);before=physical(current)
    tri=weld[np.asarray(data['tri'],dtype=np.int64)]
    edges=np.unique(np.sort(np.vstack((tri[:,[0,1]],tri[:,[1,2]],tri[:,[2,0]])),axis=1),axis=0)
    result=before.copy();reports=[]
    for index,name in enumerate(regions):
        if progress:progress(name,index/max(len(regions),1))
        values=np.asarray(data['weights_'+name],dtype=float)
        if values.shape!=(len(rest),) or not np.isfinite(values).all() or np.any((values<0)|(values>1)):
            raise ValueError('Recovery needs valid canonical region weights: '+name)
        weights=np.zeros(len(base));np.maximum.at(weights,weld,values)
        core=weights>.7;patch=weights>.001
        if np.count_nonzero(core)<3:raise ValueError('Recovery region has no usable core: '+name)
        for _ in range(int(collar_rings)):
            patch[np.unique(edges[np.any(patch[edges],axis=1)])]=True
        faces=np.asarray(data['source_faces_'+name],dtype=np.int64)
        if not len(faces):raise ValueError('Mark the matching source surface: '+name)
        src=source[np.unique(np.asarray(data['st'])[faces])]
        tree=cKDTree(src)
        aligned=base+(np.median(src,axis=0)-np.median(base[core],axis=0))
        for _ in range(30):
            distance,nearest=tree.query(aligned[core])
            keep=distance<np.quantile(distance,.85)
            moving=aligned[core][keep];target=src[nearest[keep]]
            rotation,scale,_=similarity(moving,target)
            scale=float(np.clip(scale,.98,1.02))
            # Keep the centroids paired after limiting scale. Reusing the
            # unconstrained translation makes recovery depend on world origin.
            translation=target.mean(0)-moving.mean(0)@rotation*scale
            aligned=aligned@rotation*scale+translation
        result,report=stitch(result,aligned,edges,core,patch,iterations=iterations)
        report.update(region=name,method='SOURCE_REGISTERED_CANONICAL_RECOVERY',
            source_rms=float(np.sqrt(np.mean(tree.query(aligned[core])[0]**2))),
            source_geometry_unchanged=True,native_validation_required=True,
            source_articulation_exact=False,review_required=True)
        reports.append(report)
    return current+(result-before)[weld],reports
