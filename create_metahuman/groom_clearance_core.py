"""Root-pinned elastic strand clearance against a caller-owned surface oracle.

The oracle returns nearest point, outward normal and region weight. Its inside
predicate must disambiguate concave/open-sheet nearest-normal false positives.
No topology, roots, widths, UVs, groups or materials are authored here.

Optional routing(point, reference) supplies a target point and outward direction.
Optional safety(point, reference) supplies the nearest point, outward direction
and permitted signed gap relative to the original sample. Surrounding-surface
violations take precedence over regional improvement when selecting a result.
"""
import numpy as np


def repair(points,lengths,nearest,inside,bounds,*,clearance=.00012,root_guard=.001,
           maximum_offset=.03,maximum_length_change=.15,stiffness=12.,passes=10,
           cancelled=lambda:False,progress=lambda *args:None,routing=None,safety=None):
    points=np.asarray(points);lengths=np.asarray(lengths)
    if points.ndim!=2 or points.shape[1]!=3 or points.dtype.kind!='f' or not np.isfinite(points).all():
        raise ValueError('Supply finite ordered groom positions')
    if lengths.ndim!=1 or lengths.dtype.kind not in 'iu' or np.any(lengths<2) or lengths.sum()!=len(points):
        raise ValueError('Supply complete strand point counts')
    if any(not np.isfinite(v) or v<0 for v in (clearance,root_guard,maximum_offset,maximum_length_change,stiffness)) or not maximum_offset or not 1<=passes<=32:
        raise ValueError('Choose finite bounded clearance settings')
    bounds=[(np.asarray(a,float),np.asarray(b,float)) for a,b in bounds]
    if not bounds or any(a.shape!=(3,) or b.shape!=(3,) or not np.isfinite([a,b]).all() or np.any(a>b) for a,b in bounds):
        raise ValueError('Supply finite surface-region bounds')
    tolerance=max(1e-6,min(.00005,max(clearance/4,1e-6)))
    offsets=np.r_[0,np.cumsum(lengths)];result=points.copy();arcs={}
    def arc(i):
        if i not in arcs:arcs[i]=np.r_[0,np.cumsum(np.linalg.norm(np.diff(points[offsets[i]:offsets[i+1]],axis=0),axis=1))]
        return arcs[i]
    def check_cancel():
        if cancelled():raise InterruptedError('Groom clearance cancelled; original strands retained')
    def in_bounds(values,margin=0.):
        mask=np.zeros(len(values),bool)
        for low,high in bounds:mask|=((values>=low-margin)&(values<=high+margin)).all(1)
        return mask
    def surface_query(point):
        hit=nearest(point)
        if hit is None:return None
        q,n,weight=hit
        if weight<.05:return None
        q,n=np.asarray(q,float),np.asarray(n,float)
        if n.shape!=(3,) or not np.isfinite(n).all() or np.linalg.norm(n)<1e-12:return None
        n=n/np.linalg.norm(n);distance=float(np.dot(point-q,n))
        if distance>=clearance-tolerance:return None
        if distance<0 and not inside(point):return None
        return n,distance
    def query(point,reference=None):
        rows=[]
        hit=surface_query(point)
        if hit is not None:rows.append(hit)
        if routing is not None:
            route=routing(point,point if reference is None else reference)
            if route is not None:
                q,n=map(lambda v:np.asarray(v,float),route)
                size=float(np.linalg.norm(n))
                if q.shape!=(3,) or n.shape!=(3,) or not np.isfinite([q,n]).all() or size<1e-12:
                    raise ValueError('The routing envelope returned an invalid contact')
                n=n/size;distance=float(np.dot(point-q,n))
                if distance<clearance-tolerance:rows.append((n,distance))
        if safety is not None and reference is not None and np.linalg.norm(point-reference)>tolerance:
            guard=safety(point,reference)
            if guard is not None:
                q,n,allowed=guard;q,n=np.asarray(q,float),np.asarray(n,float)
                size=float(np.linalg.norm(n))
                if q.shape!=(3,) or n.shape!=(3,) or not np.isfinite([q,n]).all() or not np.isfinite(allowed) or size<1e-12:
                    raise ValueError('The surrounding-surface guard returned an invalid contact')
                n=n/size;distance=float(np.dot(point-q,n))+clearance-allowed
                if distance<clearance-tolerance:rows.append((n,distance,True))
        return rows
    selected=np.flatnonzero(in_bounds(points));owners=np.searchsorted(offsets,selected,side='right')-1
    candidates=set()
    for count,(index,owner) in enumerate(zip(selected,owners)):
        if count%2048==0:check_cancel();progress('Inspecting strand clearance',count,len(selected))
        if arc(owner)[index-offsets[owner]]<=root_guard:continue
        hit=query(points[index])
        if any(row[1]<-tolerance for row in hit):candidates.add(int(owner))
    # Catch coarse strands passing through the region between sampled vertices.
    ends=np.ones(len(points)-1,bool);ends[offsets[1:-1]-1]=False
    mids=(points[:-1]+points[1:])*.5
    selected=np.flatnonzero(ends & in_bounds(mids));owners=np.searchsorted(offsets,selected,side='right')-1
    for count,(index,owner) in enumerate(zip(selected,owners)):
        if count%2048==0:check_cancel();progress('Inspecting strand segments',count,len(selected))
        if owner in candidates:continue
        j=index-offsets[owner]
        if (arc(owner)[j]+arc(owner)[j+1])*.5<=root_guard:continue
        hit=query(mids[index])
        if any(row[1]<-tolerance for row in hit):candidates.add(int(owner))
    del mids
    before_total=after_total=0;maximum_depth_before=maximum_depth_after=0.;changed_curves=0;unresolved=[];length_changes=[]
    kernels={}
    for count,owner in enumerate(sorted(candidates)):
        check_cancel();progress('Relaxing penetrating strands',count,len(candidates))
        start,end=offsets[owner:owner+2];original=points[start:end].astype(float);n=len(original);travel=arc(owner)
        free=(travel>root_guard)&in_bounds(original,maximum_offset+.01);free[0]=False;ids=np.flatnonzero(free)
        def contacts(q):
            rows=[]
            for i,p in enumerate(q):
                if travel[i]<=root_guard or (not in_bounds(p[None])[0] and np.linalg.norm(p-original[i])<=tolerance):continue
                hit=query(p,original[i])
                for contact in hit:rows.append((i,i,0.,contact[0],contact[1],len(contact)>2))
            for i in range(n-1):
                if (travel[i]+travel[i+1])*.5<=root_guard:continue
                p=(q[i]+q[i+1])*.5
                reference=(original[i]+original[i+1])*.5
                if not in_bounds(p[None])[0] and np.linalg.norm(p-reference)<=tolerance:continue
                hit=query(p,reference)
                for contact in hit:rows.append((i,i+1,.5,contact[0],contact[1],len(contact)>2))
            return rows
        def score(rows):
            gaps=[clearance-row[4] for row in rows]
            # Never trade a new/worsened surrounding penetration for a better
            # regional score. An infeasible curve keeps its original geometry.
            surrounding=sum(max(clearance-r[4]-tolerance,0.)**2 for r in rows if r[5])
            return (surrounding,sum(gap*gap for gap in gaps),max(gaps,default=0.),len(rows))
        q=original.copy();initial=contacts(q);best=q.copy();best_rows=initial;best_score=score(initial)
        before_total+=len(initial);maximum_depth_before=max(maximum_depth_before,max((-r[4] for r in initial),default=0.))
        if len(ids):
            key=(n,tuple(ids))
            if key not in kernels:
                second=np.diff(np.eye(n),n=2,axis=0)
                kernel=np.eye(n)+stiffness*(second.T@second)
                kernels[key]=np.linalg.inv(kernel[np.ix_(ids,ids)])
            inverse=kernels[key];lookup=np.full(n,-1,int);lookup[ids]=np.arange(len(ids))
            original_length=float(travel[-1])
            for iteration in range(passes):
                rows=contacts(q)
                if not rows:break
                coefficients=[];normals=[];required=[]
                for contact in rows:
                    a,b,t,normal,distance=contact[:5]
                    row=np.zeros(len(ids))
                    for vertex,weight in ((a,1-t),(b,t)):
                        if lookup[vertex]>=0:row[lookup[vertex]]+=weight
                    if not row.any():continue
                    coefficients.append(row);normals.append(normal)
                    required.append(clearance-distance+np.dot(row@(q[ids]-original[ids]),normal))
                if not coefficients:break
                f=np.array(coefficients);normals=np.array(normals);required=np.array(required)
                green=inverse@f.T;dual=(f@green)*(normals@normals.T)
                diagonal=np.maximum(dual.diagonal(),1e-12);multipliers=np.zeros(len(required))
                for _ in range(80):
                    previous=multipliers.copy()
                    for j in range(len(required)):
                        multipliers[j]=max(0.,multipliers[j]+(required[j]-dual[j]@multipliers)/diagonal[j])
                    if np.max(np.abs(multipliers-previous),initial=0)<1e-8:break
                displacement=green@(normals*multipliers[:,None])
                largest=float(np.linalg.norm(displacement,axis=1).max(initial=0))
                if largest>maximum_offset:displacement*=maximum_offset/largest
                proposal=original.copy();proposal[ids]+=displacement
                accepted=None
                for blend in (1.,.75,.5,.25,.125):
                    trial=q+(proposal-q)*blend
                    total=float(np.linalg.norm(np.diff(trial,axis=0),axis=1).sum())
                    if abs(total-original_length)<=maximum_length_change*max(original_length,1e-12):accepted=trial;break
                if accepted is None:break
                q=accepted;current=contacts(q);current_score=score(current)
                if current_score<best_score:best=q.copy();best_rows=current;best_score=current_score
                if not current:break
        after_total+=len(best_rows);maximum_depth_after=max(maximum_depth_after,max((-r[4] for r in best_rows),default=0.))
        if best_rows:unresolved.append(dict(curve=owner,remaining_samples=len(best_rows),depth=max((-r[4] for r in best_rows),default=0.)))
        if not np.array_equal(best,original):
            result[start:end]=best.astype(points.dtype);changed_curves+=1
            length_changes.append(abs(float(np.linalg.norm(np.diff(best,axis=0),axis=1).sum())-travel[-1])/max(travel[-1],1e-12))
    changed=np.flatnonzero(np.any(result!=points,axis=1))
    if cancelled and cancelled():raise InterruptedError('Groom clearance cancelled')
    roots_exact=result[offsets[:-1]].tobytes()==points[offsets[:-1]].tobytes()
    if not roots_exact or not np.isfinite(result).all():raise RuntimeError('Clearance violated a fixed root or produced invalid positions')
    report=dict(method='ROOT_PINNED_ELASTIC_SURFACE_BARRIER',candidate_curves=len(candidates),changed_curves=changed_curves,changed_points=len(changed),violation_samples_before=before_total,violation_samples_after=after_total,maximum_depth_before=float(maximum_depth_before),maximum_depth_after=float(maximum_depth_after),maximum_point_movement=float(np.linalg.norm(result[changed]-points[changed],axis=1).max(initial=0)),maximum_relative_length_change=float(max(length_changes,default=0.)),roots_byte_exact=roots_exact,unresolved_curves=unresolved,clearance=clearance,root_guard=root_guard,maximum_offset=maximum_offset,native_motion_accepted=False,sampled_segment_clearance=True)
    report.update(routing_constraints=routing is not None,surrounding_guard=safety is not None,
                  surrounding_tolerance=tolerance,sampling='VERTICES_AND_SEGMENT_MIDPOINTS')
    return result,report
