"""Global contact recovery for locally generated fairing proposals."""
import numpy as np


def run(reference,tasks,guard,local_runner,*,iterations,cancelled=None,progress=None):
    from . import region_projection as rp
    reference=np.array(reference,dtype=float,copy=True);tri=guard['triangles'];weld=guard['weld']
    tolerance=float(np.linalg.norm(np.ptp(reference,axis=0))*1e-6)
    allowed=rp.crossings(reference,tri,weld,pairs=True,tolerance=tolerance)
    p=reference[tri];before_normal=np.cross(p[:,1]-p[:,0],p[:,2]-p[:,0])
    area_floor=np.sum(before_normal**2,axis=1)*.01;reference_quality=rp.triangle_quality(reference,tri)
    faces=weld[tri];edges=np.unique(np.sort(np.vstack((faces[:,[0,1]],faces[:,[1,2]],faces[:,[2,0]])),axis=1),axis=0)
    pinned=set();attempts=[];reports=[]
    projector=None
    if guard.get('source_budget') is not None:
        from .regional_source_projection import Projector
        projector=Projector(guard['source_budget'],reference)
    def defects(output):
        pairs=rp.crossings(output,tri,weld,pairs=True,tolerance=tolerance)
        p=output[tri];normal=np.cross(p[:,1]-p[:,0],p[:,2]-p[:,0])
        bad=np.flatnonzero((np.einsum('ij,ij->i',normal,before_normal)<area_floor)|(rp.triangle_quality(output,tri)<reference_quality*.35))
        return pairs-allowed,bad
    def preserve(output,new,bad):
        unsafe_faces=np.union1d(np.asarray(list(new),dtype=np.int64).ravel(),bad)
        affected=np.unique(faces[unsafe_faces]);affected=np.union1d(affected,edges[np.any(np.isin(edges,affected),axis=1)])
        moving=np.unique(weld[np.any(output!=reference,axis=1)])
        fresh=set(int(v) for v in np.intersect1d(affected,moving))-pinned
        pinned.update(fresh);return len(fresh)
    def finish(output,reports):
        receipt=dict(method='GLOBAL_CONTACT_LOCALIZED_POLISH_V1',passes=attempts,pinned_physical_vertices=len(pinned),
            whole_mesh_checked=True,new_pairs=0,degraded_faces=0,minimum_quality_fraction=.35,original_reference_retained=True,native_validation_required=True)
        for report in reports:
            report['global_contact_recovery']=receipt
            if projector is not None:report['saved_source_projection']=projector.report()
        return output,reports
    for index in range(4):
        if cancelled and cancelled():raise RuntimeError('Regional polish cancelled; previous geometry retained')
        prepared=[]
        for task in tasks:
            families=np.unique(weld[task['ids']]);influence=task['influence'].copy()
            influence[np.isin(families,list(pinned))]=0
            prepared.append(dict(task,influence=influence))
        def update(name,value):
            if progress:progress('Polish pass '+str(index+1)+': '+name,(index+value)/4)
        output,reports=local_runner(reference,prepared,iterations=iterations,cancelled=cancelled,progress=update,mode='POLISH')
        if projector is not None:
            active=np.unique(weld[np.any(output!=reference,axis=1)])
            output=projector(output,active,cancelled=cancelled)
        new,bad=defects(output)
        row=dict(pass_index=index+1,new_proper_contacts=len(new),degraded_faces=len(bad),pinned_physical_vertices=len(pinned));attempts.append(row)
        if not new and not len(bad):
            return finish(output,reports)
        row['newly_pinned_physical_vertices']=preserve(output,new,bad)
        if not row['newly_pinned_physical_vertices']:break
    # Check the last preserved field before discarding all remaining motion.
    # Restoring whole physical families cannot worsen another vertex's source
    # distance; changed triangle joins still require global checks every time.
    for index in range(4):
        if cancelled and cancelled():raise RuntimeError('Regional polish cancelled; previous geometry retained')
        output[np.isin(weld,list(pinned))]=reference[np.isin(weld,list(pinned))]
        new,bad=defects(output)
        row=dict(stage='FINAL_PRESERVATION',pass_index=len(attempts)+1,new_proper_contacts=len(new),degraded_faces=len(bad),pinned_physical_vertices=len(pinned));attempts.append(row)
        if not new and not len(bad):return finish(output,reports)
        row['newly_pinned_physical_vertices']=preserve(output,new,bad)
        if not row['newly_pinned_physical_vertices']:break
    receipt=dict(method='GLOBAL_CONTACT_LOCALIZED_POLISH_V1',passes=attempts,pinned_physical_vertices=len(pinned),
        whole_mesh_checked=True,new_pairs=0,original_reference_retained=True,native_validation_required=True,candidate_rejected=True)
    for report in reports:
        report['global_contact_recovery']=receipt;report['accepted']=False;report['reason']='No globally contact-safe local polish was found.'
        if projector is not None:report['saved_source_projection']=projector.report()
        report['planned_after']=report['after'];report['after']=dict(report['before']);report['max_displacement']=0.
    return reference.copy(),reports
