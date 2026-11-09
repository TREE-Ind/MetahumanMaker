"""Bounded anatomical lash-root fitting with explicit guide repair.

Render strand shape, widths, colors and group membership remain authored.
Guide replacement is opt-in and uses coverage-selected authored render curves;
it is reported separately from transporting existing guides.
"""
import copy,hashlib,json
import numpy as np


def fit(data,rims,*,maximum_root_offset,clearance=0.,outward=(0.,1.,0.),rebuild_guides=False):
    if not np.isfinite(maximum_root_offset) or maximum_root_offset<=0 or not np.isfinite(clearance) or clearance<0:raise ValueError('Choose finite physical lash fitting bounds')
    direction=np.asarray(outward,dtype=float)
    if direction.shape!=(3,) or not np.isfinite(direction).all() or np.linalg.norm(direction)<1e-12:raise ValueError('Choose a finite outward root direction')
    direction/=np.linalg.norm(direction);starts=[];ends=[];uv_starts=[];uv_ends=[];roles=[]
    for role,rim in rims.items():
        positions=np.asarray(rim['positions'],dtype=float);uv=np.asarray(rim['uv'],dtype=float)
        if positions.ndim!=2 or positions.shape[1]!=3 or len(positions)<4 or uv.shape!=(len(positions),2) or not np.isfinite(positions).all() or not np.isfinite(uv).all():raise ValueError('Supply ordered anatomical rim positions and their skin-chart UVs')
        starts.extend(positions);ends.extend(np.roll(positions,-1,axis=0));uv_starts.extend(uv);uv_ends.extend(np.roll(uv,-1,axis=0));roles.extend([role]*len(positions))
    a=np.asarray(starts);edge=np.asarray(ends)-a;den=np.sum(edge*edge,axis=1)
    if not len(a) or np.any(den<=1e-18):raise ValueError('An anatomical lash rim has a collapsed edge')
    curves=data.get('curves',[])
    if not curves:raise ValueError('Supply authored source strands')
    roots=[]
    for curve in curves:
        points=np.asarray(curve['points'],dtype=float)
        if points.ndim!=2 or points.shape[1]!=3 or len(points)<2 or not np.isfinite(points).all():raise ValueError('An authored lash curve is invalid')
        roots.append(points[0])
    roots=np.asarray(roots);parameter=np.clip(np.einsum('nvi,vi->nv',roots[:,None]-a,edge)/den,0,1);q=a+parameter[:,:,None]*edge;distance=np.linalg.norm(q-roots[:,None],axis=2);segment=distance.argmin(1);rows=np.arange(len(curves));anchor=q[rows,segment]+clearance*direction;offset=anchor-roots;movement=np.linalg.norm(offset,axis=1);guide=np.asarray([bool(c.get('guide',0)) for c in curves]);checked=~guide if rebuild_guides else np.ones(len(curves),dtype=bool)
    if not (~guide).any() or movement[~guide].max(initial=0)>maximum_root_offset:raise ValueError('Visible lash roots exceed the bounded anatomical fit; refine the source alignment')
    if not rebuild_guides and movement[guide].max(initial=0)>maximum_root_offset:raise ValueError('Detached guide roots exceed the fit bound; enable Rebuild Guides from Strands to propose replacement guides')
    result=copy.deepcopy(data);uv_a=np.asarray(uv_starts);uv_b=np.asarray(uv_ends);t=parameter[rows,segment];fitted=[];original_guides=[]
    for i,curve in enumerate(result['curves']):
        if guide[i] and rebuild_guides:original_guides.append(curve);continue
        curve['points']=(np.asarray(curve['points'])+offset[i]).tolist();curve['root_uv']=(uv_a[segment[i]]*(1-t[i])+uv_b[segment[i]]*t[i]).tolist();fitted.append(curve)
    replaced=0
    if rebuild_guides:
        # Select complete render curves, never fabricate straight guides.
        # Farthest-point coverage includes root position and sampled curl.
        groups=sorted({int(c['group']) for c in original_guides})
        for group in groups:
            render=[c for c in fitted if not c.get('guide') and int(c['group'])==group];count=sum(int(c['group'])==group for c in original_guides)
            if count>len(render) or not render:raise ValueError('A groom group has too few authored render strands to rebuild its guides')
            features=[]
            for curve in render:
                points=np.asarray(curve['points']);sample=np.linspace(0,len(points)-1,4);features.append(np.column_stack([np.interp(sample,np.arange(len(points)),points[:,axis]) for axis in range(3)]).ravel())
            features=np.asarray(features);chosen=[int(np.argmax(np.linalg.norm(features-features.mean(0),axis=1)))];nearest=np.sum((features-features[chosen[0]])**2,axis=1)
            for _ in range(1,count):
                nearest[chosen]=-np.inf;index=int(nearest.argmax());chosen.append(index);nearest=np.minimum(nearest,np.sum((features-features[index])**2,axis=1))
            for index in chosen:
                curve=copy.deepcopy(render[index]);curve['guide']=1;fitted.append(curve);replaced+=1
    result['curves']=fitted
    report=dict(method='ANATOMICAL_LID_ROOT_TRANSPORT',render_curves=int((~guide).sum()),guides_rebuilt=replaced,guide_repair_requested=bool(rebuild_guides),authored_render_shapes_preserved=True,maximum_render_root_movement=float(movement[~guide].max(initial=0)),source_guide_sha256=hashlib.sha256(json.dumps(original_guides,sort_keys=True).encode()).hexdigest() if original_guides else None,native_animation_accepted=False)
    return result,report
