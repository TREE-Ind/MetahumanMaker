"""Reconstruct marked outer ears while retaining an attached head collar."""
import numpy as np
from scipy import sparse
from scipy.spatial import ConvexHull,cKDTree
from .ocular_surface import _chart,FrontSurface
from .surface_charts import boundary_loop,ordered_angles,collar_weights,map_disk_patch,disk_region_faces
from .source_contours import convex_outline


def fit(data,options,*,progress=None,cancelled=None):
    if options.get('ear_method','INTRINSIC')=='PROJECTED':
        return fit_projected(data,options,progress=progress,cancelled=cancelled)
    if options.get('ear_method','INTRINSIC')!='INTRINSIC':raise ValueError('Unknown ear reconstruction method')
    reference=np.asarray(data['parameter_reference']);current=np.asarray(data['current']);source=np.asarray(data['source']);tri=np.asarray(data['triangles']);st=np.asarray(data['source_triangles'])
    centers=source[st].mean(1);head_tree=cKDTree(current);distance,nearest=head_tree.query(centers)
    if 'body_current' in data:eligible=distance<=cKDTree(data['body_current']).query(centers)[0]
    else:
        if np.ptp(source[:,2])>np.ptp(current[:,2])*2:raise ValueError('A full-body source needs its accepted body reference; use the full-body workflow or a head-only source')
        eligible=np.ones(len(st),bool)
    _,source_weld=np.unique(source,axis=0,return_inverse=True);physical_faces=source_weld[st]
    planned=current.copy();movable=np.zeros(len(current),bool);reports=[]
    for ear in range(2):
        if cancelled and cancelled():raise RuntimeError('Ear reconstruction cancelled')
        region=np.asarray(data['ear_region_'+str(ear)])>.05;core=np.asarray(data['ear_weights_'+str(ear)])>.95
        if np.any(core&~region):raise ValueError('Include the complete native ear and its supporting collar in the result region')
        native_faces=np.flatnonzero(np.all(region[tri],axis=1));source_core=np.asarray(data['source_ear_'+str(ear)])>.25;seed=np.all(source_core[st],axis=1)
        if not seed.any():raise ValueError('Mark a complete connected source ear surface')
        selected=np.flatnonzero((region[nearest]&eligible)|seed)
        patch=physical_faces[selected];edges,counts=np.unique(np.sort(np.vstack((patch[:,[0,1]],patch[:,[1,2]],patch[:,[2,0]])),axis=1),axis=0,return_counts=True)
        try:
            if np.any(counts>2) or len(np.unique(patch))-len(edges)+len(patch)!=1:raise ValueError('Not a disk')
            boundary_loop(patch)
            source_faces=selected;regularization=dict(method='ALREADY_MANIFOLD_DISK',faces=len(selected))
        except ValueError:
            try:source_faces,regularization=disk_region_faces(source,st,selected)
            except ValueError as exc:raise ValueError('The source ear and supporting head patch must form one connected surface disk. Refine the source ear selection; geometry was retained. '+str(exc)) from exc
        outward=np.asarray(data['eye_basis'])[:,0]*float(data['ear_sign_'+str(ear)])
        native_tip=reference[np.flatnonzero(core)[np.argmax(reference[core]@outward)]];points=source[source_core];source_tip=points[np.argmax(points@outward)]
        if progress:progress('Mapping curved source ear '+str(ear+1),ear/2)
        candidate,report,record=map_disk_patch(reference,planned,tri,source,st,native_faces,source_faces,
            forward=(0.,0.,1.),collar_rings=5,chart_optimizer='harmonic',native_distal_point=native_tip,source_distal_point=source_tip,
            native_child_influence=core.astype(float),source_child_influence=seed.astype(float),seam_mode='harmonic_displacement')
        changed=np.any(candidate!=planned,axis=1)
        if np.any(movable&changed):raise ValueError('The marked ear reconstruction regions overlap')
        planned=candidate;movable|=changed
        report.update(solver='INTRINSIC_CURVED_EAR_AND_HEAD_COLLAR_V1',accepted=True,ear=ear,source_region=regularization,
            source_head_faces=int(eligible.sum()),protected_boundary=True,three_dimensional_validation_required=True)
        reports.append(report)
    return planned,reports,dict(before=current,planned=planned,surface_movable=movable)


def fit_projected(data,options,*,progress=None,cancelled=None):
    reference=np.asarray(data['parameter_reference']);current=np.asarray(data['current']);source=np.asarray(data['source']);tri=np.asarray(data['triangles']);st=np.asarray(data['source_triangles']);planned=current.copy();movable=np.zeros(len(current),bool);reports=[]
    source_centers=source[st].mean(1)
    if 'body_current' in data:
        eligible=cKDTree(current).query(source_centers)[0]<=cKDTree(data['body_current']).query(source_centers)[0]
    else:
        if np.ptp(source[:,2])>np.ptp(current[:,2])*2:raise ValueError('A full-body source needs its body reference to separate head and arm surfaces; use the full-body workflow or a head-only source')
        eligible=np.ones(len(st),bool)
    if not eligible.any():raise ValueError('No source surfaces belong to the fitted head')
    front_basis=np.asarray(data['eye_basis'])
    for k in range(2):
        if cancelled and cancelled():raise RuntimeError('Ear reconstruction cancelled')
        sign=float(data['ear_sign_'+str(k)]);local=np.array([[0.,-sign,0.],[sign,0.,0.],[0.,0.,1.]]);basis=front_basis@local;ref=reference@basis;cur=current@basis;src=source@basis
        tolerance=max(float(np.linalg.norm(np.ptp(ref,axis=0)))*1e-6,1e-9);_,w=np.unique(np.rint(ref/tolerance).astype(np.int64),axis=0,return_inverse=True);count=np.bincount(w);n=len(count)
        physical=lambda p:np.column_stack([np.bincount(w,weights=p[:,j])/count for j in range(3)])
        b=physical(ref);v=physical(cur);faces=w[tri];region=np.zeros(n,bool);core=np.zeros(n,bool);np.logical_or.at(region,w,data['ear_region_'+str(k)]>.05);np.logical_or.at(core,w,data['ear_weights_'+str(k)]>.95)
        if np.any(core&~region):raise ValueError('The result ear region must include the native ear and its supporting collar')
        rim=boundary_loop(faces[np.all(core[faces],1)]);points=src[data['source_ear_'+str(k)]>.25]
        if len(points)<12:raise ValueError('Mark a complete source ear surface')
        hull=points[ConvexHull(points[:,[0,2]]).vertices];angles,order_report=ordered_angles(b[rim]);desired=convex_outline(hull,angles)
        e=np.unique(np.sort(np.vstack((faces[:,[0,1]],faces[:,[1,2]],faces[:,[2,0]])),axis=1),axis=0);e=e[e[:,0]!=e[:,1]];adj=sparse.coo_matrix((np.ones(len(e)*2),(np.r_[e[:,0],e[:,1]],np.r_[e[:,1],e[:,0]])),shape=(n,n)).tocsr()
        weights=collar_weights(adj,region,core)
        ids,xy,outer,report=_chart(b,v,faces,adj,rim,desired,region=region,cancelled=cancelled,progress=(lambda value:progress('Reconstructing ear '+str(k+1),(k+value)/2)) if progress else None)
        out=v.copy();out[ids[:,None],[0,2]]=xy;changed=np.setdiff1d(ids,outer);queries=out[changed][:,[0,2]]
        original_depth=FrontSurface(cur,tri)(queries);source_depth=FrontSurface(src,st[eligible])(queries);out[changed,1]=original_depth+(source_depth-original_depth)*weights[changed]
        mask=np.isin(w,changed)
        if np.any(movable&mask):raise ValueError('The marked ear reconstruction regions overlap')
        planned[mask]=out[w[mask]]@basis.T;movable|=mask
        report.update(order_report,solver='ORDERED_SOURCE_OUTER_EAR_WITH_SURFACE_COLLAR',accepted=True,ear=k,source_head_faces=int(eligible.sum()),protected_boundary=True);reports.append(report)
    return planned,reports,dict(before=current,planned=planned,surface_movable=movable)
