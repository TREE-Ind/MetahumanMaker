"""Continue accepted color across an explicit repair mask on the physical mesh.

This positive-weight harmonic fill invents low-frequency color only. It does not
recover hidden source skin, synthesize pores, or classify clothing automatically.
"""
import numpy as np
from scipy import sparse,ndimage
from scipy.sparse.linalg import spsolve
from scipy.sparse.csgraph import connected_components,dijkstra
try:
    from .texture_surface import pixels,sample,validate
except ImportError:
    from texture_surface import pixels,sample,validate


def linear(values):
    return np.where(values<=.04045,values/12.92,((values+.055)/1.055)**2.4)


def encoded(values):
    return np.where(values<=.0031308,values*12.92,1.055*np.maximum(values,0)**(1/2.4)-.055)


def heal(source,mask,points,faces,uv,*,padding=12,margin=0.,feather=0.,encoding='SRGB',progress=None):
    source=np.asarray(source,dtype=np.float32);mask=np.asarray(mask,dtype=np.float32)
    original_points,original_uv=points,uv
    _checked,faces,_uv=validate(points,faces,uv)
    points=np.asarray(original_points,dtype=np.float64);uv=np.asarray(original_uv,dtype=np.float64)
    if source.ndim!=3 or source.shape[2]!=3 or mask.shape!=source.shape[:2] or not np.isfinite(source).all() or not np.isfinite(mask).all():raise ValueError('Surface heal needs matching finite RGB and repair-mask planes')
    if source.min()<0 or source.max()>1 or mask.min()<0 or mask.max()>1 or not np.any(mask>0):raise ValueError('Choose a nonempty repair mask and normalized color values')
    if not 0<=padding<=64:raise ValueError('Surface heal padding must be between 0 and 64 texels')
    if encoding not in {'SRGB','LINEAR'}:raise ValueError('Choose sRGB color or linear data encoding')
    h,w=mask.shape
    if uv.min()<-1e-6 or uv.max()>1+1e-6:raise ValueError('Normalize a single target UV tile before healing')
    if progress:progress('Solving shared surface color')
    physical,weld=np.unique(points,axis=0,return_inverse=True);tri=weld[faces];count=len(physical)
    edges=np.concatenate((tri[:,[0,1]],tri[:,[1,2]],tri[:,[2,0]]));edges=np.unique(np.sort(edges,axis=1),axis=0);edges=edges[edges[:,0]!=edges[:,1]]
    length=np.linalg.norm(physical[edges[:,0]]-physical[edges[:,1]],axis=1)
    if not len(edges) or np.any(length<=0):raise ValueError('The repair surface has no usable physical adjacency')
    weight=1/length;rows=np.concatenate((edges[:,0],edges[:,1]));cols=np.concatenate((edges[:,1],edges[:,0]));weights=np.tile(weight,2)
    adjacency=sparse.coo_matrix((weights,(rows,cols)),shape=(count,count)).tocsr()
    active=np.zeros(count,dtype=bool);active[tri.ravel()]=True
    if not np.isfinite([margin,feather]).all() or margin<0 or feather<0:raise ValueError('Choose finite nonnegative surface margins')
    if margin or feather:
        seeds=np.unique(tri.ravel()[sample(mask,uv.reshape(-1,2))>=.5])
        if not len(seeds):raise ValueError('This repair mask has no mesh-vertex seeds; use zero surface margin for a tiny region')
        metric=sparse.coo_matrix((np.tile(length,2),(rows,cols)),shape=(count,count)).tocsr();distance=dijkstra(metric,directed=False,indices=seeds,min_only=True)
        mask=mask.copy()
        for triangle,vertices in zip(uv,tri):
            if not np.any(distance[vertices]<=margin+feather):continue
            raster=pixels(triangle*[w,-h]+[-.5,h-.5],(w,h))
            if raster is not None:
                y,x,bary=raster;field=bary@distance[vertices];influence=np.clip((margin+feather-field)/max(feather,1e-12),0,1);influence[field<=margin]=1;influence=influence*influence*(3-2*influence);mask[y,x]=np.maximum(mask[y,x],influence)
    values=sample(source,uv.reshape(-1,2));values=linear(values) if encoding=='SRGB' else values;repair=sample(mask,uv.reshape(-1,2));trusted=repair<=1e-6;ids=tri.ravel()
    sums=np.zeros((count,3));n=np.bincount(ids[trusted],minlength=count);np.add.at(sums,ids[trusted],values[trusted]);known=n>0;colors=sums/np.maximum(n[:,None],1)
    _,component=connected_components(adjacency,directed=False)
    anchored=np.bincount(component[known],minlength=int(component.max())+1)>0
    if np.any(active&~anchored[component]):raise ValueError('A masked connected surface has no accepted color anchors; reduce the mask or provide a reference')
    unknown=active&~known
    if np.any(unknown):
        laplacian=sparse.diags(np.asarray(adjacency.sum(axis=1)).ravel())-adjacency
        matrix=laplacian[unknown][:,unknown];rhs=-(laplacian[unknown][:,known]@colors[known]);colors[unknown]=spsolve(matrix,rhs)
    if not np.isfinite(colors[active]).all():raise ValueError('Surface color solve failed; accepted pixels retained')
    colors=np.clip(colors,0,1);output=source.copy();covered=np.zeros((h,w),dtype=bool);changed=np.zeros((h,w),dtype=bool)
    if progress:progress('Reprojecting healed surface color')
    for index,(triangle,vertices) in enumerate(zip(uv,tri)):
        raster=pixels(triangle*[w,-h]+[-.5,h-.5],(w,h))
        if raster is None:continue
        y,x,bary=raster;covered[y,x]=True;selected=mask[y,x]>0
        if not np.any(selected):continue
        y,x,bary=y[selected],x[selected],bary[selected];proposal=bary@colors[vertices];proposal=encoded(proposal) if encoding=='SRGB' else proposal;amount=mask[y,x,None];proposal=source[y,x]+(proposal-source[y,x])*amount
        repeated=changed[y,x]
        if np.any(repeated) and np.max(np.abs(output[y[repeated],x[repeated]]-proposal[repeated]))>1e-3:raise ValueError('Overlapping selected UV faces disagree; repair the layout before healing')
        output[y,x]=proposal;changed[y,x]=True
    padded=0
    if padding and np.any((mask>0)&~covered):
        distance,nearest=ndimage.distance_transform_edt(~covered,return_indices=True)
        y,x=np.nonzero((mask>0)&~covered&(distance<=padding));proposal=output[nearest[0,y,x],nearest[1,y,x]];amount=mask[y,x,None];output[y,x]=source[y,x]+(proposal-source[y,x])*amount;padded=len(y);changed[y,x]=True
    output[mask<=0]=source[mask<=0]
    return np.clip(output,0,1),dict(method='PHYSICAL_SURFACE_HARMONIC_COLOR',encoding=encoding,physical_vertices=count,unknown_vertices=int(unknown.sum()),accepted_vertex_anchors=int(known.sum()),surface_texels=int(covered.sum()),selected_surface_texels=int(np.sum((mask>0)&covered)),padding_texels=padded,surface_margin=margin,surface_feather=feather,vertex_seed_geodesic_growth=bool(margin or feather),outside_mask_max_delta=float(np.max(np.abs(output[mask<=0]-source[mask<=0]))) if np.any(mask<=0) else 0.,physical_albedo_recovered=False,generated_detail=False),mask
