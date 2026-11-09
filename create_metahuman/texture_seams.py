"""Measure matching 3D edges across split UV charts, without image heuristics."""
import numpy as np
from scipy import ndimage
from scipy import sparse
from scipy.sparse.linalg import lsmr


def reconcile(image,edges,centers,*,eligible,domain,max_delta=.2,radius=4,progress=None,evidence=None):
    """Balance an owned linear-RGB candidate across actual paired UV edges.

    Unknowns occupy a narrow editable band. Bilinear sample constraints couple
    the charts; spatial regularization tapers corrections into their interiors.
    Zero eligibility (including protected paint) is an exact fixed boundary.
    The candidate array is updated in place; no files or source maps are read.
    """
    if image.ndim!=3 or image.shape[2]!=3 or eligible.shape!=image.shape[:2] or domain.shape!=eligible.shape:
        raise ValueError('Seam balancing needs matching RGB, eligibility and UV coverage')
    if not image.flags.c_contiguous or image.dtype.kind!='f' or image.min()<0 or image.max()>1:
        raise ValueError('Seam balancing requires a contiguous linear float image in 0..1')
    if evidence is not None and evidence.shape!=eligible.shape:
        raise ValueError('Seam evidence must match the color map')
    if not all(np.isfinite(v).all() for v in (image,eligible,domain)) or not 0<=max_delta<=.25:
        raise ValueError('Seam balancing inputs must be finite and correction bounded')
    if not 1<=radius<=8:raise ValueError('Seam band radius must be 1..8 texels')
    h,w=eligible.shape
    points=samples(edges,centers,(w,h))
    report={'paired_edges':len(edges),'paired_samples':len(points),'linear_correction_limit':max_delta,
            'band_radius_texels':radius,'invented_surface_detail':False,'protected_pixels_changed':0}
    if not len(points) or max_delta==0:
        return report|{'status':'skipped','editable_samples':0,'changed_texels':0}
    xy=np.clip(points.astype(np.float64),[0,0],[w-1,h-1])
    lo=np.floor(xy).astype(np.int64);hi=np.minimum(lo+1,[w-1,h-1]);fraction=xy-lo
    x,y=fraction[...,0],fraction[...,1]
    ids=np.stack([lo[...,1]*w+lo[...,0],lo[...,1]*w+hi[...,0],
                  hi[...,1]*w+lo[...,0],hi[...,1]*w+hi[...,0]],axis=-1)
    coefficients=np.stack([(1-x)*(1-y),x*(1-y),(1-x)*y,x*y],axis=-1)
    coefficients[:,1]*=-1
    ids=ids.reshape(-1,8);coefficients=coefficients.reshape(-1,8)
    gate=np.clip(eligible,0,1).ravel()
    valid=np.asarray(domain).ravel()>.5
    active=np.any((gate[ids]>0)&valid[ids]&(np.abs(coefficients)>1e-10),axis=1)
    ids=ids[active];coefficients=coefficients[active]
    report['editable_samples']=int(active.sum())
    if not len(ids):return report|{'status':'no_selected_seams','changed_texels':0}
    editable=(gate[ids]>0)&valid[ids]&(np.abs(coefficients)>1e-10)
    unknown=np.unique(ids[editable])
    # Expand only through covered editable texels, avoiding a full-atlas
    # distance field or any modification of unrelated UV padding.
    for _ in range(radius):
        x=unknown%w
        adjacent=np.concatenate([unknown,unknown[x>0]-1,unknown[x<w-1]+1,
                                 unknown[unknown>=w]-w,unknown[unknown<(h-1)*w]+w])
        adjacent=np.unique(adjacent)
        unknown=adjacent[(gate[adjacent]>0)&valid[adjacent]]
    count=len(unknown)
    columns=np.searchsorted(unknown,ids)
    present=(columns<count)&(unknown[np.minimum(columns,count-1)]==ids)
    rows=np.broadcast_to(np.arange(len(ids))[:,None],ids.shape)
    constraint=sparse.coo_matrix((coefficients[present]*gate[ids[present]],
        (rows[present],columns[present])),shape=(len(ids),count)).tocsr()
    mismatch=np.sum(image.reshape(-1,3)[ids]*coefficients[...,None],axis=1)
    # Corrections are smooth inside each chart, with zero correction outside
    # the band/selection. UV background is not a zero-color boundary condition.
    row_parts=[];col_parts=[];value_parts=[];row_count=0
    local=np.arange(count)
    for delta,inside in [(1,unknown%w<w-1),(w,unknown<(h-1)*w),
                         (-1,unknown%w>0),(-w,unknown>=w)]:
        origin=local[inside];neighbor=unknown[inside]+delta
        found=np.searchsorted(unknown,neighbor)
        paired=(found<count)&(unknown[np.minimum(found,count-1)]==neighbor)
        # Each undirected interior pair appears once. Fixed covered neighbors
        # anchor the outer edge of the narrow correction band.
        choose=(paired&(delta>0))|(~paired&valid[neighbor])
        origin=origin[choose];found=found[choose];paired=paired[choose]
        these=np.arange(row_count,row_count+len(origin));row_count+=len(origin)
        row_parts.append(these);col_parts.append(origin);value_parts.append(gate[unknown[origin]])
        row_parts.append(these[paired]);col_parts.append(found[paired]);value_parts.append(-gate[unknown[found[paired]]])
    smooth=sparse.coo_matrix((np.concatenate(value_parts),
        (np.concatenate(row_parts),np.concatenate(col_parts))),shape=(row_count,count)).tocsr()
    system=sparse.vstack([constraint*8,smooth*.15,sparse.eye(count,format='csr')*.015],format='csr')
    correction=np.empty((count,3),np.float64);iterations=[]
    for channel in range(3):
        if progress:progress(channel,3,'Matching UV seam colors')
        rhs=np.concatenate([-8*mismatch[:,channel],np.zeros(row_count+count)])
        solution=lsmr(system,rhs,atol=1e-6,btol=1e-6,maxiter=200)
        correction[:,channel]=np.clip(solution[0],-max_delta,max_delta)*gate[unknown]
        iterations.append(int(solution[2]))
    before=image.reshape(-1,3)[unknown].copy()
    image.reshape(-1,3)[unknown]=np.clip(before+correction,0,1)
    delta=image.reshape(-1,3)[unknown]-before
    if evidence is not None:
        evidence.ravel()[unknown]=np.maximum(evidence.ravel()[unknown],np.linalg.norm(delta,axis=1))
    # Unknowns were explicitly restricted to strictly positive eligibility.
    remaining=mismatch+constraint@(delta/gate[unknown,None])
    if progress:progress(3,3,'Matching UV seam colors')
    return report|{'status':'balanced','unknown_texels':count,'iterations':iterations,
        'changed_texels':int(np.any(delta!=0,axis=1).sum()),
        'max_linear_delta':float(np.abs(delta).max()),
        'paired_linear_error_before_p95':float(np.percentile(np.abs(mismatch).mean(axis=1),95)),
        'paired_linear_error_after_p95':float(np.percentile(np.abs(remaining).mean(axis=1),95)),
        'paired_linear_error_after_max':float(np.abs(remaining).mean(axis=1).max()),
        'all_seams_verified_seamless':False}


def samples(edges, centers, size):
    edges=np.asarray(edges,dtype=np.float32)
    centers=np.asarray(centers,dtype=np.float32)
    if edges.shape!=(len(edges),2,2,2) or centers.shape!=(len(edges),2,2):
        raise ValueError('UV seam correspondence has invalid dimensions')
    if not np.isfinite(edges).all() or not np.isfinite(centers).all():
        raise ValueError('UV seam correspondence must be finite')
    w,h=size
    transform=np.array([w,-h],dtype=np.float32)
    offset=np.array([-.5,h-.5],dtype=np.float32)
    pixel_edges=edges*transform+offset
    pixel_centers=centers*transform+offset
    pairs=[]
    for edge,center in zip(pixel_edges,pixel_centers):
        count=max(2,int(np.ceil(np.linalg.norm(edge[:,1]-edge[:,0],axis=1).max()))+1)
        count=min(count,1024)
        t=(np.arange(count,dtype=np.float32)+.5)/count
        points=edge[:,0,None,:]+(edge[:,1]-edge[:,0])[:,None,:]*t[None,:,None]
        tangent=edge[:,1]-edge[:,0]
        inward=np.stack([-tangent[:,1],tangent[:,0]],axis=1)
        toward=center-edge.mean(axis=1)
        inward*=np.where(np.sum(inward*toward,axis=1)>=0,1.,-1.)[:,None]
        inward/=np.maximum(np.linalg.norm(inward,axis=1,keepdims=True),1e-8)
        # Sample just inside the charts, avoiding padding/background texels.
        points+=inward[:,None,:]*.5
        pairs.append(points.transpose(1,0,2))
    return np.concatenate(pairs) if pairs else np.empty((0,2,2),np.float32)


def read(image, points):
    values=[]
    for side in range(2):
        xy=points[:,side]
        if image.ndim==2:
            value=ndimage.map_coordinates(image,[xy[:,1],xy[:,0]],order=1,mode='nearest')
        else:
            value=np.stack([ndimage.map_coordinates(image[...,c],[xy[:,1],xy[:,0]],order=1,
                                mode='nearest') for c in range(image.shape[2])],axis=1)
        values.append(value)
    return np.stack(values,axis=1)


def read_color(image,points):
    """Linear-light bilinear sampling of sRGB color texels."""
    from texture_restore_core import linear,encoded
    h,w=image.shape[:2]
    xy=np.clip(points.astype(np.float64),[0,0],[w-1,h-1])
    lo=np.floor(xy).astype(np.int64);hi=np.minimum(lo+1,[w-1,h-1]);fraction=xy-lo
    x,y=fraction[...,0],fraction[...,1]
    ids=np.stack([lo[...,1]*w+lo[...,0],lo[...,1]*w+hi[...,0],
                  hi[...,1]*w+lo[...,0],hi[...,1]*w+hi[...,0]],axis=-1)
    weights=np.stack([(1-x)*(1-y),x*(1-y),(1-x)*y,x*y],axis=-1)
    texels=linear(image.reshape(-1,3)[ids])
    return np.clip(encoded(np.sum(texels*weights[...,None],axis=-2)),0,1)


def inspect(source, candidate, edges, centers, *, eligible=None):
    """Report actual edge correspondence; measurements are not visual approval."""
    if source.shape!=candidate.shape or source.ndim!=3 or source.shape[2]!=3:
        raise ValueError('Seam inspection needs matching RGB images')
    points=samples(edges,centers,(source.shape[1],source.shape[0]))
    if not len(points):
        return {'paired_edges':0,'paired_samples':0,'continuity_measured':False}
    before=read_color(source,points);after=read_color(candidate,points)
    old=np.abs(before[:,0]-before[:,1]).mean(axis=1)
    new=np.abs(after[:,0]-after[:,1]).mean(axis=1)
    change=after-before
    introduced=np.abs(change[:,0]-change[:,1]).mean(axis=1)
    affected=np.ones(len(points),dtype=bool) if eligible is None else read(eligible,points).max(axis=1)>0
    def stats(values):
        return {'mean':float(values.mean()),'p95':float(np.percentile(values,95)),
                'max':float(values.max())} if values.size else {'mean':0.,'p95':0.,'max':0.}
    return {'paired_edges':len(edges),'paired_samples':len(points),'continuity_measured':True,
            'sample_units':'encoded RGB 0..1 after sRGB decode and bilinear filtering, half-texel inside matching chart edges',
            'source_seam_difference':stats(old),'candidate_seam_difference':stats(new),
            'introduced_seam_difference':stats(introduced),
            'seam_regression':stats(np.maximum(new-old,0)),
            'affected_samples':int(affected.sum()),'affected_introduced_difference':stats(introduced[affected]),
            'full_surface_visual_approval':False}


def retain_regressed_anchors(source,candidate,edges,centers,*,tolerance=2/255,max_passes=32):
    """Keep accepted texels at unresolved paired seam footprints, with feathering.

    Inputs/output are encoded RGB. This never invents a seam color or relaxes
    the inspection gate. Restore only offending bilinear footprints and their
    three-texel transition; accepted color is the explicit fallback.
    """
    from texture_restore_core import linear,encoded
    h,w=source.shape[:2];points=samples(edges,centers,(w,h))
    chosen=set();failures=[]
    for _ in range(max_passes):
        before=read_color(source,points);after=read_color(candidate,points)
        old=np.abs(before[:,0]-before[:,1]).mean(axis=1)
        new=np.abs(after[:,0]-after[:,1]).mean(axis=1)
        bad=new-old>tolerance
        failures.append(int(bad.sum()))
        if not bad.any():break
        xy=np.clip(points[bad].reshape(-1,2),[0,0],[w-1,h-1]);lo=np.floor(xy).astype(np.int64)
        for dx,dy in ((0,0),(1,0),(0,1),(1,1)):
            xs=np.minimum(lo[:,0]+dx,w-1);ys=np.minimum(lo[:,1]+dy,h-1)
            chosen.update((ys*w+xs).tolist())
        if len(chosen)>w*h*.01:raise ValueError('Unresolved seam anchors exceed the narrow restoration budget')
        # Sparse three-pixel neighborhoods avoid allocating another full 8K RGB.
        ids=np.fromiter(chosen,dtype=np.int64);ys,xs=ids//w,ids%w
        weights={}
        for dy in range(-3,4):
            for dx in range(-3,4):
                radius=max(abs(dx),abs(dy));weight=1 if radius==0 else (4-radius)/4
                yy=np.clip(ys+dy,0,h-1);xx=np.clip(xs+dx,0,w-1)
                for ident in (yy*w+xx).tolist():weights[ident]=max(weights.get(ident,0),weight)
        ids=np.fromiter(weights,dtype=np.int64);blend=np.asarray(list(weights.values()),np.float32)
        y,x=ids//w,ids%w
        candidate[y,x]=encoded(linear(candidate[y,x])*(1-blend[:,None])+linear(source[y,x])*blend[:,None])
        # All chosen footprint texels are exact accepted values, including
        # overlaps with adjacent samples that the sparse feather touched.
        ids=np.fromiter(chosen,dtype=np.int64);candidate[ids//w,ids%w]=source[ids//w,ids%w]
    return {'source':'accepted_albedo','initial_failing_samples':failures[0] if failures else 0,
            'failing_samples_by_pass':failures,'accepted_anchor_texels':len(chosen),
            'inspection_tolerance':tolerance,'inspection_gate_relaxed':False}
