"""Bind a lower-LOD UV seam to the native canonical seam curve.

Two texture charts share one geometric degree of freedom at a seam. Sampling
each chart independently can disagree after decimation. This module uses the
actual reference seam edges as the shared witness, retaining the original UV
candidates for auditing. It does not certify a native animated pose.
"""
import numpy as np


def seam_curves(reference, *, uv_tolerance):
    ids=np.asarray(reference['layout_positions']);uv=np.asarray(reference['layout_uv'],dtype=float);tri=np.asarray(reference['triangles']);positions=np.asarray(reference['positions'],dtype=float)
    if not np.isfinite(uv_tolerance) or uv_tolerance<=0:raise ValueError('Choose a positive seam chart tolerance')
    buckets={}
    for a,b in np.vstack((tri[:,[0,1]],tri[:,[1,2]],tri[:,[2,0]])):
        first,last=int(ids[a]),int(ids[b])
        if first==last:continue
        if first>last:first,last,a,b=last,first,b,a
        buckets.setdefault((first,last),[]).append(uv[[a,b]])
    edges=[];charts=[]
    for edge,rows in buckets.items():
        rows=np.asarray(rows)
        if len(rows)<2 or np.max(np.abs(rows-rows[0]))<=uv_tolerance:continue
        if np.linalg.norm(positions[edge[1]]-positions[edge[0]])<=1e-12:continue
        edges.append(edge);charts.append(rows)
    # A seam junction can carry three or more chart coordinates. No single
    # incident edge represents all of them, so retain the native topological
    # junction as a point witness rather than dropping one of its charts.
    layouts={}
    for vertex,coordinate in zip(ids,uv):layouts.setdefault(int(vertex),[]).append(coordinate)
    for vertex,coordinates in layouts.items():
        unique=np.unique(np.asarray(coordinates),axis=0)
        if len(unique)>=3:
            edges.append((vertex,vertex));charts.append(np.repeat(unique[:,None,:],2,axis=1))
    return dict(edges=np.asarray(edges,dtype=np.int64).reshape(-1,2),charts=charts,positions=positions)


def bind_seam_curves(reference,target,binding,*,uv_tolerance,maximum_distance,curves=None,geometric_uncertainty=0.):
    """Choose curve witnesses using physical distance and all incident charts.

    Equidistant witnesses are retained as explicit alternatives. Transfer must
    bound their field disagreement before using the closest witness. At a
    junction, the same canonical endpoint is equivalent on incident edges.
    No field values are consulted when constructing the geometric binding.
    """
    if not np.isfinite(maximum_distance) or maximum_distance<=0:raise ValueError('Choose a positive anatomical seam bound')
    if not np.isfinite(geometric_uncertainty) or geometric_uncertainty<0:raise ValueError('Geometric uncertainty must be finite and nonnegative')
    curves=seam_curves(reference,uv_tolerance=uv_tolerance) if curves is None else curves
    positions=np.asarray(target['positions'],dtype=float);ids=np.asarray(target['layout_positions']);uv=np.asarray(target['layout_uv'],dtype=float);tri=np.asarray(target['triangles']);edges=curves['edges'];source=curves['positions']
    if len(ids)!=len(binding['target_indices']) or not np.array_equal(ids,binding['target_indices']):raise ValueError('Seam binding belongs to a different native layout')
    result=dict(binding);result['source_indices']=np.array(binding['source_indices'],copy=True);result['barycentric']=np.array(binding['barycentric'],copy=True)
    result['raw_chart_source_indices']=np.array(binding['source_indices'],copy=True);result['raw_chart_barycentric']=np.array(binding['barycentric'],copy=True)
    incident=np.vstack((tri[:,[0,1]],tri[:,[1,2]],tri[:,[2,0]]));length=np.linalg.norm(uv[incident[:,0]]-uv[incident[:,1]],axis=1);footprint=np.full(len(ids),uv_tolerance)
    np.maximum.at(footprint,incident[:,0],length);np.maximum.at(footprint,incident[:,1],length)
    groups={}
    for layout,vertex in enumerate(ids):groups.setdefault(int(vertex),[]).append(layout)
    records=[];unresolved=[];resolved=[];alternatives=[]
    if len(edges):
        a=source[edges[:,0]];delta=source[edges[:,1]]-a;length2=np.einsum('ij,ij->i',delta,delta);length2=np.where(length2>0,length2,1.)
    # These native anatomy buffers define the geometry; their float32 storage
    # is not measurement noise. Rank exact buffered positions in float64 and
    # reserve ties for arithmetic uncertainty, plus any caller-declared input
    # uncertainty. A blanket float32-ULP window can merge distinct branches.
    precision=max(64*np.finfo(np.float64).eps*max(float(np.abs(source).max()),1.),1e-12)+geometric_uncertainty
    for vertex,layouts in groups.items():
        layouts=np.asarray(layouts,dtype=int)
        if len(layouts)<2 or np.max(np.abs(uv[layouts]-uv[layouts[0]]))<=uv_tolerance:continue
        if not len(edges):unresolved.append(vertex);continue
        t=np.clip(np.einsum('ij,ij->i',positions[vertex]-a,delta)/length2,0,1);points=a+t[:,None]*delta;distance=np.linalg.norm(points-positions[vertex],axis=1);candidates=np.flatnonzero(distance<=maximum_distance);valid=[]
        for edge in candidates:
            charts=curves['charts'][edge];samples=charts[:,0]*(1-t[edge])+charts[:,1]*t[edge]
            chart_distance=np.linalg.norm(uv[layouts,None,:]-samples[None,:,:],axis=2).min(axis=1)
            if np.all(chart_distance<=footprint[layouts]+uv_tolerance):valid.append(int(edge))
        if not valid:unresolved.append(vertex);continue
        valid=np.asarray(valid);valid=valid[np.argsort(distance[valid],kind='stable')];chosen=int(valid[0]);ambiguous=False
        for alternative in valid[1:]:
            if distance[alternative]-distance[chosen]>precision:break
            # A geometrically close tie on another branch needs an actual
            # pose constraint; do not choose its field by index ordering.
            shared=np.intersect1d(edges[chosen],edges[alternative])
            equivalent=any(np.linalg.norm(points[chosen]-source[v])<=precision and np.linalg.norm(points[alternative]-source[v])<=precision for v in shared)
            if not equivalent:ambiguous=True;break
        if ambiguous:
            for alternative in valid[1:]:
                if distance[alternative]-distance[chosen]>precision:break
                alternatives.append((vertex,int(edges[alternative,0]),int(edges[alternative,1]),float(t[alternative])))
        result['source_indices'][layouts]=[edges[chosen,0],edges[chosen,1],edges[chosen,1]];result['barycentric'][layouts]=[1-t[chosen],t[chosen],0.]
        resolved.extend(layouts.tolist());records.append((vertex,int(edges[chosen,0]),int(edges[chosen,1]),float(t[chosen]),float(distance[chosen])))
    resolved=np.asarray(resolved,dtype=np.int64);old=np.asarray(binding['unmapped_layouts']);keep=~np.isin(old,resolved)
    result['unmapped_layouts']=old[keep];result['unmapped_uv']=np.asarray(binding['unmapped_uv'])[keep];result['support_guard_radii']=np.asarray(binding['support_guard_radii'])[keep]
    result['canonical_seam_records']=np.asarray(records,dtype=float).reshape(-1,5);result['unresolved_seam_vertices']=np.asarray(unresolved,dtype=np.int64)
    result['canonical_seam_alternatives']=np.asarray(alternatives,dtype=float).reshape(-1,4)
    return result,dict(method='CANONICAL_NATIVE_SEAM_CURVE',resolved_vertices=len(records),junction_vertices=sum(row[1]==row[2] for row in records),uncertain_vertices=len({row[0] for row in alternatives}),unresolved_vertices=len(unresolved),maximum_curve_distance=max((row[-1] for row in records),default=0.),geometric_tie_tolerance=precision,declared_geometry_uncertainty=geometric_uncertainty,native_motion_accepted=False)
