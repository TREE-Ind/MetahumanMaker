"""UV-chart witnesses for transferring native corrective fields across LODs.

Bindings retain canonical DNA identities and barycentric surface witnesses.
Anatomical distance bounds reject overlapping or incompatible UV charts;
there is no positional nearest-vertex fallback or silent seam averaging.
"""
import numpy as np
from .surface_oracle import SurfaceOracle


def bind(reference, target, *, uv_tolerance, maximum_anatomical_distance, allow_zero_outside_support=False, allow_anatomy_chart_projection=False):
    """Bind two matching native anatomy charts, independent of fitted shape.

    Each input contains positions, layout_positions, layout_uv and triangles
    in the DNA vertex-layout domain. Distances use the positions' own units.
    """
    if not np.isfinite(uv_tolerance) or uv_tolerance<=0 or not np.isfinite(maximum_anatomical_distance) or maximum_anatomical_distance<=0:
        raise ValueError('LOD transfer needs explicit positive chart/anatomy bounds')
    def chart(data):
        positions=np.asarray(data['positions'],dtype=float);ids=np.asarray(data['layout_positions']);uv=np.asarray(data['layout_uv'],dtype=float);tri=np.asarray(data['triangles'])
        if positions.ndim!=2 or positions.shape[1]!=3 or ids.ndim!=1 or uv.shape!=(len(ids),2) or tri.ndim!=2 or tri.shape[1]!=3 or ids.dtype.kind not in 'iu' or tri.dtype.kind not in 'iu' or not len(ids) or not len(tri):raise ValueError('Invalid native LOD layout arrays')
        if ids.min()<0 or ids.max()>=len(positions) or tri.min()<0 or tri.max()>=len(ids) or not np.isfinite(positions).all() or not np.isfinite(uv).all():raise ValueError('LOD chart contains invalid geometry or identities')
        if not np.all(np.bincount(ids,minlength=len(positions))>0):raise ValueError('A DNA position has no UV layout witness')
        return positions,ids,uv,tri
    source,sids,suv,tri=chart(reference);destination,dids,duv,dtri=chart(target);anatomy_tri=tri
    corners=suv[tri];edge=corners[:,1:]-corners[:,:1];area=np.abs(edge[:,0,0]*edge[:,1,1]-edge[:,0,1]*edge[:,1,0]);tri=tri[area>1e-14]
    if not len(tri):raise ValueError('The reference LOD has no usable UV triangles')
    # Native decimation often retains original positions exactly, even where
    # UV boundaries shift. Only unique byte-identical anatomy positions are
    # anchors; coincident distinct source IDs require chart witnesses instead.
    buckets={}
    for i,point in enumerate(source):buckets.setdefault(point.tobytes(),[]).append(i)
    unique={key:ids[0] for key,ids in buckets.items() if len(ids)==1}
    exact=np.asarray([unique.get(point.tobytes(),-1) for point in destination[dids]],dtype=np.int64);anchored=exact>=0;remaining=np.flatnonzero(~anchored)
    source_ids=np.zeros((len(dids),3),dtype=np.int64);weights=np.zeros((len(dids),3));source_ids[anchored]=exact[anchored,None];weights[anchored,0]=1.;maximum_uv=0.
    edges=np.vstack((dtri[:,[0,1]],dtri[:,[1,2]],dtri[:,[2,0]]));length=np.linalg.norm(duv[edges[:,0]]-duv[edges[:,1]],axis=1);guard=np.full(len(dids),uv_tolerance)
    np.maximum.at(guard,edges[:,0],length);np.maximum.at(guard,edges[:,1],length)
    mapped=anchored.copy();anatomy_projected=0;boundary_projected=0;maximum_chart_projection=0.
    if len(remaining):
        witness=SurfaceOracle(np.c_[suv,np.zeros(len(suv))],tri).query(np.c_[duv[remaining],np.zeros(len(remaining))],max_distance=uv_tolerance)
        valid=remaining[witness['observed']];source_ids[valid]=sids[tri[witness['source_triangle_ids'][witness['observed']]]];weights[valid]=witness['barycentric'][witness['observed']];mapped[valid]=True;maximum_uv=float(witness['residual'][witness['observed']].max(initial=0))
    if allow_anatomy_chart_projection:
        reconstructed=np.einsum('ni,nij->nj',weights,source[source_ids]);gap=np.linalg.norm(reconstructed-destination[dids],axis=1);missing=np.flatnonzero(~mapped|(gap>maximum_anatomical_distance))
        if len(missing):
            # Decimation may shift a UV border. Project onto the native
            # anatomical surface, requiring compatible normals and a UV
            # witness inside that target vertex's adjacent-edge footprint.
            # This is triangle correspondence, never a nearest-vertex alias.
            corners=destination[dids[dtri]];normal=np.cross(corners[:,1]-corners[:,0],corners[:,2]-corners[:,0]);normals=np.zeros_like(destination)
            for corner in range(3):np.add.at(normals,dids[dtri[:,corner]],normal)
            normals/=np.maximum(np.linalg.norm(normals,axis=1)[:,None],1e-300)
            proof=SurfaceOracle(source[sids],anatomy_tri).query(destination[dids[missing]],max_distance=maximum_anatomical_distance,query_normals=normals[dids[missing]],normal_limit=.2)
            selected=np.flatnonzero(proof['observed'])
            if len(selected):
                f=anatomy_tri[proof['source_triangle_ids'][selected]];bary=proof['barycentric'][selected];chart_points=np.einsum('ni,nij->nj',bary,suv[f]);drift=np.linalg.norm(chart_points-duv[missing[selected]],axis=1);valid=drift<=guard[missing[selected]]
                indices=missing[selected[valid]];source_ids[indices]=sids[f[valid]];weights[indices]=bary[valid];mapped[indices]=True;anatomy_projected=len(indices);maximum_chart_projection=float(drift[valid].max(initial=0))
        missing=np.flatnonzero(~mapped)
        if len(missing):
            # Native LOD UV borders can move across the reference border.
            # A boundary witness is admissible only within the target's own
            # one-ring UV footprint AND the explicit anatomical distance.
            # Record this projection separately from exact chart samples.
            proof=SurfaceOracle(np.c_[suv,np.zeros(len(suv))],tri).query(np.c_[duv[missing],np.zeros(len(missing))],max_distance=float(guard[missing].max()))
            selected=np.flatnonzero(proof['observed'])
            if len(selected):
                f=tri[proof['source_triangle_ids'][selected]];bary=proof['barycentric'][selected];points=np.einsum('ni,nij->nj',bary,source[sids[f]]);distance=np.linalg.norm(points-destination[dids[missing[selected]]],axis=1);drift=proof['residual'][selected]
                valid=(distance<=maximum_anatomical_distance)&(drift<=guard[missing[selected]]);indices=missing[selected[valid]];source_ids[indices]=sids[f[valid]];weights[indices]=bary[valid];mapped[indices]=True;boundary_projected=len(indices);maximum_chart_projection=max(maximum_chart_projection,float(drift[valid].max(initial=0)))
    unmapped=np.flatnonzero(~mapped)
    if len(unmapped) and not allow_zero_outside_support:raise ValueError(f"{len(unmapped)}/{len(duv)} lower-LOD layouts leave the native chart after {int(anchored.sum())} exact anchors; retain current correctives")
    reconstructed=np.einsum('ni,nij->nj',weights[mapped],source[source_ids[mapped]]);distance=np.linalg.norm(reconstructed-destination[dids[mapped]],axis=1)
    if distance.max(initial=0)>maximum_anatomical_distance:raise ValueError(f'A UV witness maps to incompatible anatomy ({distance.max():g} exceeds {maximum_anatomical_distance:g}); retain the current lower LOD')
    # A caller may retain zero displacement outside a compact corrective's
    # support. Each unmapped site must be farther than its entire adjacent UV
    # edge neighborhood from EVERY changed source face, checked per field.
    # This never invents a witness or fills a changed region with zero.
    return dict(source_vertices=len(source),target_vertices=len(destination),source_indices=source_ids,barycentric=weights,target_indices=dids,
                unmapped_layouts=unmapped,unmapped_uv=duv[unmapped],support_guard_radii=guard[unmapped],reference_uv=suv,reference_triangles=tri,reference_layout_positions=sids,
                anatomy_projected_layouts=anatomy_projected,boundary_projected_layouts=boundary_projected,maximum_projected_uv_drift=maximum_chart_projection,byte_exact_layout_anchors=int(anchored.sum()),maximum_uv_residual=maximum_uv,maximum_anatomical_residual=float(distance.max(initial=0)))


def transfer(offsets,binding,*,duplicate_tolerance):
    """Sample a canonical displacement field and verify all seam witnesses."""
    offsets=np.asarray(offsets,dtype=float)
    if offsets.shape!=(binding['source_vertices'],3) or not np.isfinite(offsets).all() or not np.isfinite(duplicate_tolerance) or duplicate_tolerance<0:raise ValueError('Invalid corrective field or seam bound')
    unmapped=binding['unmapped_layouts']
    if len(unmapped):
        tri=binding['reference_triangles'];active=np.any(np.any(offsets[binding['reference_layout_positions'][tri]]!=0,axis=2),axis=1)
        if active.any():
            uv=binding['reference_uv'];query=binding['unmapped_uv'];radius=binding['support_guard_radii'];support=SurfaceOracle(np.c_[uv,np.zeros(len(uv))],tri,face_ids=np.flatnonzero(active)).query(np.c_[query,np.zeros(len(query))],max_distance=float(radius.max()))
            if np.any(support['observed']&(support['residual']<=radius)):
                raise ValueError('An unmapped LOD chart touches the corrective support; retaining zero there would lose detail')
    values=np.einsum('ni,nij->nj',binding['barycentric'],offsets[binding['source_indices']]);ids=binding['target_indices'];n=binding['target_vertices'];counts=np.bincount(ids,minlength=n)
    if np.any(counts==0):raise ValueError('The LOD binding omits canonical vertices')
    result=np.column_stack([np.bincount(ids,weights=values[:,axis],minlength=n)/counts for axis in range(3)]);disagreement=float(np.linalg.norm(values-result[ids],axis=1).max(initial=0))
    if disagreement>duplicate_tolerance:raise ValueError(f'UV seam witnesses disagree on the transferred corrective ({disagreement:g} exceeds {duplicate_tolerance:g})')
    uncertainty=0.;alternatives=binding.get('canonical_seam_alternatives',np.empty((0,4)))
    if len(alternatives):
        vertices=alternatives[:,0].astype(np.int64);first=alternatives[:,1].astype(np.int64);last=alternatives[:,2].astype(np.int64);t=alternatives[:,3,None]
        choices=offsets[first]*(1-t)+offsets[last]*t;uncertainty=float(np.linalg.norm(choices-result[vertices],axis=1).max(initial=0))
        if uncertainty>duplicate_tolerance:raise ValueError(f'Native seam-curve alternatives disagree on the corrective ({uncertainty:g} exceeds {duplicate_tolerance:g})')
    raw_disagreement=disagreement
    if 'raw_chart_source_indices' in binding:
        raw=np.einsum('ni,nij->nj',binding['raw_chart_barycentric'],offsets[binding['raw_chart_source_indices']]);average=np.column_stack([np.bincount(ids,weights=raw[:,axis],minlength=n)/counts for axis in range(3)])
        raw_disagreement=float(np.linalg.norm(raw-average[ids],axis=1).max(initial=0))
    seams=binding.get('canonical_seam_records',np.empty((0,5)))
    return result,dict(method='NATIVE_ANATOMY_AND_CHART_SURFACE_WITNESSES',vertices=n,layout_witnesses=len(ids),canonical_seam_vertices=len(seams),maximum_curve_witness_uncertainty=uncertainty,maximum_seam_curve_distance=float(seams[:,-1].max(initial=0)) if len(seams) else 0.,raw_chart_split_disagreement=raw_disagreement,byte_exact_layout_anchors=int(binding['byte_exact_layout_anchors']),anatomy_projected_layouts=int(binding['anatomy_projected_layouts']),boundary_projected_layouts=int(binding['boundary_projected_layouts']),maximum_projected_uv_drift=float(binding['maximum_projected_uv_drift']),verified_zero_outside_support=len(unmapped),maximum_split_disagreement=disagreement,maximum_uv_residual=float(binding['maximum_uv_residual']),maximum_anatomical_residual=float(binding['maximum_anatomical_residual']),nearest_vertex_aliases=False,native_motion_accepted=False)


def transport_evaluated_motion(reference_neutral,reference_posed,target_neutral,binding,*,duplicate_tolerance):
    """Transport complete evaluated motion, retaining the target's neutral.

    All positions use one shared evaluated coordinate frame. This includes
    native baseline motion; a corrective-only field cannot account for changes
    in joints, skin weights and compiled baseline shapes between LODs.
    """
    neutral=np.asarray(reference_neutral,dtype=float);posed=np.asarray(reference_posed,dtype=float);target=np.asarray(target_neutral,dtype=float)
    if neutral.shape!=posed.shape or target.shape!=(binding['target_vertices'],3) or not np.isfinite(neutral).all() or not np.isfinite(posed).all() or not np.isfinite(target).all():raise ValueError('Evaluated motion transport needs finite corresponding native frames')
    delta,proof=transfer(posed-neutral,binding,duplicate_tolerance=duplicate_tolerance)
    return target+delta,dict(**proof,motion_space='COMPLETE_EVALUATED_NATIVE_MOTION',neutral_geometry_preserved=True)
