"""Frontal source-eye color transport with native target UV rasterization."""
import numpy as np
from scipy.ndimage import distance_transform_edt
from scipy.spatial import ConvexHull,Delaunay
from scipy.spatial import cKDTree


def sclera_continuation(atlas, marked, covered):
    """Continue the explicitly marked sclera, never an occluding skin edge.

    Source selections describe sclera, while the enclosed pupil is sampled
    from the complete source. Only interiors of marked facets may seed the
    hidden globe. Robust color consensus rejects painted lid/skin fringe;
    it makes no assumption that sclera is white (black eyes work too).
    """
    distance = distance_transform_edt(np.pad(marked & covered,1))[1:-1,1:-1]
    interior = distance >= max(1., float(distance.max()) * .15)
    if interior.sum() < 24:
        raise ValueError('Mark a usable sclera interior before continuing hidden eye color')
    values = atlas[interior]
    # Coarse RGB modes are only a seed. Refine using actual linear colors,
    # retaining measured roughness and metallic from the same observations.
    encoded = np.where(values[:,:3] <= .0031308, values[:,:3]*12.92,
                       1.055*np.maximum(values[:,:3],0)**(1/2.4)-.055)
    bins = np.clip((encoded*12).astype(int),0,11)
    labels = bins[:,0]*144+bins[:,1]*12+bins[:,2]
    histogram = np.bincount(labels,minlength=1728)
    peak = int(histogram.argmax())
    seed = np.median(encoded[labels == peak],axis=0)
    difference = np.linalg.norm(encoded-seed,axis=1)
    consensus = difference <= .18
    if consensus.sum() < 24:
        raise ValueError('Marked sclera has no stable color consensus; refine the sclera selection')
    color = np.median(values[consensus],axis=0)
    # Trust the pupil-bearing observed aperture; reject fringe contamination
    # only in a narrow edge band. Hidden texels all receive the same measured
    # sclera base, avoiding radial stripes and duplicate iris/pupil fragments.
    return color, dict(method='MARKED_SCLERA_INTERIOR_COLOR_CONSENSUS',
        interior_samples=int(interior.sum()),consensus_samples=int(consensus.sum()),
        consensus_fraction=float(consensus.mean()),linear_color=color[:3].tolist())


def decontaminate_fringe(atlas,marked,covered,distance,color,transition):
    """Remove chromatic skin bleed from the marked sclera's aperture fringe.

    Only the uncertain boundary band is eligible. Unmarked pupil facets and
    dark pigment are protected, as is every observed interior texel.
    """
    rgb=atlas[:,:,:3];total=rgb.sum(2);base=float(color[:3].sum())
    chroma=rgb/np.maximum(total[:,:,None],1e-8);seed=color[:3]/max(base,1e-8)
    deviation=np.linalg.norm(chroma-seed,axis=2)
    eligible=marked&covered&(distance<transition*3)&(total>base*.12)
    strength=np.clip((deviation-.06)/.06,0,1)
    band=np.clip((transition*3-distance)/(transition*2),0,1);band=band*band*(3-2*band)
    weight=np.where(eligible,strength*band,0)
    result=atlas*(1-weight[:,:,None])+color*weight[:,:,None]
    return result,dict(method='MARKED_SCLERA_CHROMATIC_FRINGE_REPAIR',affected_pixels=int(np.sum(weight>0)),maximum_weight=float(weight.max()),pupil_facets_protected=True,observed_interior_unchanged=True)


def fragments(triangle, size):
    triangle = np.asarray(triangle, dtype=float) * size - .5
    low = np.maximum(np.ceil(triangle.min(0)).astype(int), 0)
    high = np.minimum(np.floor(triangle.max(0)).astype(int), size - 1)
    if np.any(high < low):return None
    x, y = np.meshgrid(np.arange(low[0], high[0]+1), np.arange(low[1], high[1]+1))
    a, b = triangle[1]-triangle[0], triangle[2]-triangle[0]
    determinant = a[0]*b[1]-a[1]*b[0]
    if abs(determinant) < 1e-12:return None
    dx, dy = x-triangle[0, 0], y-triangle[0, 1]
    u = (dx*b[1]-dy*b[0])/determinant
    v = (a[0]*dy-a[1]*dx)/determinant
    keep = (u >= -1e-9) & (v >= -1e-9) & (u+v <= 1+1e-9)
    return x[keep], y[keep], np.c_[1-u[keep]-v[keep], u[keep], v[keep]]


def sample(pixels, uv, wrap='REPEAT'):
    height, width = pixels.shape[:2]
    xy = np.asarray(uv) * [width, height] - .5
    cells = np.floor(xy).astype(np.int64); fraction = xy-cells
    result = np.zeros((len(uv), pixels.shape[2]), dtype=np.float32)
    for dx, dy, weight in ((0,0,(1-fraction[:,0])*(1-fraction[:,1])),
                           (1,0,fraction[:,0]*(1-fraction[:,1])),
                           (0,1,(1-fraction[:,0])*fraction[:,1]),
                           (1,1,fraction[:,0]*fraction[:,1])):
        x, y = cells[:,0]+dx, cells[:,1]+dy
        if wrap == 'REPEAT':x, y = x % width, y % height
        else:x, y = np.clip(x,0,width-1), np.clip(y,0,height-1)
        result += pixels[y,x] * weight[:,None]
    return result


def project(data, materials, eye, *, size=2048, atlas_size=1024, progress=None, diagnostic=None, hidden_pigment='SCLERA_ONLY'):
    if hidden_pigment not in {'SCLERA_ONLY','FIT_DARK_ELLIPSE'}:raise ValueError('Unknown hidden-eye pigment policy')
    source = np.asarray(data['source_points_'+str(eye)], dtype=float)
    source_uv = data['source_uv_'+str(eye)]
    source_material = data['source_material_'+str(eye)]
    points = np.asarray(data['eye_points_'+str(eye)], dtype=float)
    triangles = data['eye_triangles_'+str(eye)]
    uv = data['eye_uv_'+str(eye)]
    seed_points=np.asarray(data['source_seed_'+str(eye)])
    seeds=seed_points[:,[0,2]]
    low,high=seeds.min(0),seeds.max(0)
    extent = high-low
    if np.any(extent <= 0):raise ValueError('Marked source eye has zero frontal area')
    hull=seeds[ConvexHull(seeds).vertices]
    gx,gy=np.meshgrid((np.arange(atlas_size)+.5)/atlas_size,(np.arange(atlas_size)+.5)/atlas_size)
    domain=(Delaunay((hull-low)/extent).find_simplex(np.c_[gx.ravel(),gy.ravel()])>=0).reshape(atlas_size,atlas_size)
    del gx,gy
    atlas = np.zeros((atlas_size,atlas_size,5), np.float32)
    marked = np.zeros((atlas_size,atlas_size),bool)
    seed_distance = cKDTree(seed_points).query(source.reshape(-1,3))[0].reshape(-1,3)
    seed_faces = np.all(seed_distance <= max(float(np.linalg.norm(extent))*1e-7,1e-10),axis=1)
    depth = np.full((atlas_size,atlas_size), np.inf)
    for i, triangle in enumerate(source):
        row = fragments((triangle[:,[0,2]]-low)/extent, atlas_size)
        if row is None:continue
        x, y, bary = row
        d = bary@triangle[:,1]; take = (d < depth[y,x]) & domain[y,x]
        if not take.any():continue
        x,y,bary,d = x[take],y[take],bary[take],d[take]
        material = materials[int(source_material[i])]
        query = bary@source_uv[i]
        if 'image' in material:
            rgb = sample(data[material['image']], query, material['wrap']) * material['gain']
        else:rgb = np.broadcast_to(material['color'],(len(x),3))
        values = np.empty((len(x),5),np.float32);values[:,:3]=rgb
        for channel,name in ((3,'roughness'),(4,'metallic')):
            specification=material[name]
            values[:,channel] = (sample(data[specification['image']],query,specification['wrap'])[:,specification['channel']]
                                  if 'image' in specification else specification['value'])
        atlas[y,x] = values; depth[y,x] = d;marked[y,x] = seed_faces[i]
    covered = np.isfinite(depth)
    if diagnostic:diagnostic(atlas,covered)
    if covered.sum() < atlas_size**2*.05:
        raise ValueError('Marked source eye does not form a usable frontal patch')
    missing=int(np.sum(domain & ~covered))
    if missing>max(24,int(domain.sum()*.002)):
        raise ValueError('Original source faces do not cover the marked eye aperture, including its pupil; include the complete source eye geometry')
    edge_color, continuation = sclera_continuation(atlas,marked,covered)
    aperture_distance = distance_transform_edt(np.pad(domain,1))[1:-1,1:-1]
    transition = max(2.,float(np.sqrt(domain.sum()))*.025)
    pigment=None; pigment_report=dict(status='disabled',policy=hidden_pigment)
    if hidden_pigment=='FIT_DARK_ELLIPSE':
        if __package__:
            from .ocular_pigment import fit as fit_pigment, evaluate as sample_pigment
        else:
            from ocular_pigment import fit as fit_pigment, evaluate as sample_pigment
        pigment,pigment_report=fit_pigment(atlas,domain,covered,marked,edge_color,low,extent,transition)
    atlas,fringe_repair=decontaminate_fringe(atlas,marked,covered,aperture_distance,edge_color,transition)
    observed_weight = np.clip(aperture_distance/transition,0,1)
    observed_weight *= observed_weight*(3-2*observed_weight)
    # This edge feather affects source aperture fringe only. The interior
    # image (including pupil) remains byte-for-byte numerically unchanged.
    atlas = atlas*observed_weight[:,:,None]+edge_color*(1-observed_weight[:,:,None])
    atlas[~covered] = edge_color
    output = np.zeros((size,size,5),np.float32); mask = np.zeros((size,size),bool)
    overlap = 0.; observed = frontal = 0
    # Dense iris vertices bias a vertex-count median toward the cornea. Fit
    # the retained posterior native sphere to define the hemisphere instead.
    if 'eye_volume_center_'+str(eye) in data:
        globe_center=np.asarray(data['eye_volume_center_'+str(eye)],dtype=float)
        radius=float(data['eye_volume_depth_radius_'+str(eye)])
        if globe_center.shape!=(3,) or not np.isfinite(globe_center).all():raise ValueError('Source-eye volume frame is invalid')
        center=float(globe_center[1]);hemisphere_method='FITTED_VOLUME_FRAME'
    else:
        posterior = points[points[:,1]>=np.median(points[:,1])]
        origin = posterior.mean(0); q = posterior-origin
        fitted = np.linalg.lstsq(np.c_[q*2,np.ones(len(q))],np.sum(q*q,axis=1),rcond=None)[0]
        globe_center = origin+fitted[:3]
        center = float(globe_center[1]); radius = float(np.median(np.linalg.norm(posterior-globe_center,axis=1)));hemisphere_method='POSTERIOR_SPHERE_FIT'
    if not np.isfinite(radius) or radius<=1e-8:raise ValueError('The fitted eye has no usable posterior hemisphere')
    if np.any(uv < -1e-6) or np.any(uv > 1+1e-6):
        raise ValueError('Source-eye color requires target UVs inside the native 0–1 tile')
    for i, indices in enumerate(triangles):
        row = fragments(uv[i], size)
        if row is None:continue
        x,y,bary = row; p = bary@points[indices]
        query = (p[:,[0,2]]-low)/extent
        rgb = sample(atlas, query, 'EXTEND')
        in_bounds = np.all((query >= 0)&(query <= 1),axis=1)
        rgb[~in_bounds] = edge_color
        if pigment is not None:
            trust=sample(observed_weight[:,:,None],query,'EXTEND')[:,0]
            trust[~in_bounds]=0
            inferred=sample_pigment(p[:,[0,2]],pigment,edge_color,float(max(extent))/atlas_size*1.5)
            rgb+=(inferred-edge_color)*(1-trust[:,None])
        weight = np.clip((center+radius*.05-p[:,1])/(radius*.1),0,1)
        weight = weight*weight*(3-2*weight)
        rgb = rgb*weight[:,None]+edge_color*(1-weight[:,None])
        old = mask[y,x]
        if old.any():overlap = max(overlap,float(np.max(abs(output[y[old],x[old]]-rgb[old]))))
        xy = np.clip((query*atlas_size).astype(np.int64),0,atlas_size-1)
        front = weight>.99
        observed += int(np.sum(front & in_bounds & covered[xy[:,1],xy[:,0]])); frontal += int(front.sum())
        output[y,x] = rgb; mask[y,x] = True
        if progress and i % 128 == 0:progress(i/max(len(triangles),1))
    if overlap > .02:
        raise ValueError('Native eye UV overlaps require incompatible colors; retain the previous eye material')
    if mask.sum() < size**2*.01 or not np.isfinite(output).all():
        raise ValueError('Eye UV rasterization produced no usable color chart')
    if not frontal or observed<24:raise ValueError('Source-eye projection has no observed front hemisphere; supply the fitted globe volume frame')
    nearest = distance_transform_edt(~mask, return_distances=False, return_indices=True)
    output[~mask] = output[nearest[0][~mask],nearest[1][~mask]]
    return output, dict(method='MARKED_SOURCE_FRONTAL_COLOR_TO_NATIVE_UV', eye=eye, resolution=size,
        atlas_resolution=atlas_size, source_faces=len(source), target_triangles=len(triangles),
        aperture_domain_pixels=int(domain.sum()),uncovered_aperture_pixels=missing,
        source_patch_usage='Marked sclera defines aperture; original enclosed pupil/surface facets supply color',
        covered_uv_pixels=int(mask.sum()), maximum_overlap_color_error=overlap,
        frontal_observed_fraction=observed/max(frontal,1), posterior_color=edge_color[:3].tolist(),
        posterior_roughness=float(edge_color[3]),posterior_metallic=float(edge_color[4]),
        unobserved_color='Measured marked-sclera consensus on all unobserved globe regions',
        sclera_continuation=continuation,aperture_edge_feather_pixels=transition,
        sclera_fringe_repair=fringe_repair,
        hidden_pigment=pigment_report,
        hemisphere_method=hemisphere_method,
        source_geometry_unchanged=True, target_geometry_unchanged=True, invented_visible_detail=pigment is not None)
