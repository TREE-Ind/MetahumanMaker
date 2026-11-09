"""Anatomical contact refinement before native inverse-skin calibration.

All topology uses canonical DNA identities. The caller supplies pristine
anatomy, visibility-calibrated margins and explicitly fitted globe volumes.
No character identity, positional welding or private vertex lists are used.
"""
import numpy as np
from .lod_lid_contact import refine_blink
from .ocular_attachments import bind


class LidPoseRefiner:
    def __init__(self, anatomy, domain, volumes, meeting, *, engine_to_frame, input_sha256=None, maximum_attachment_cm=15.,allow_partial_candidate=False):
        self.anatomy=anatomy;self.domain={row['name']:row for row in domain};self.volumes=volumes;self.meeting=meeting;self.matrix=np.asarray(engine_to_frame,dtype=float);self.input_sha256=dict(input_sha256 or {});self.maximum_attachment_cm=float(maximum_attachment_cm)
        if self.matrix.shape!=(3,3) or not np.isfinite(self.matrix).all() or abs(np.linalg.det(self.matrix))<1e-15 or self.maximum_attachment_cm<=0:raise ValueError('Supply an invertible physical coordinate frame and attachment bound')
        self.inverse=np.linalg.inv(self.matrix);self.allow_partial_candidate=bool(allow_partial_candidate)

    def __call__(self,lod,goals,names):
        if lod not in self.anatomy:return goals,dict(status='not_refined',reason='No accepted anatomical visibility boundary for this LOD',native_motion_accepted=False)
        a=self.anatomy[lod];head_name=f'head_lod{lod}_mesh';eye_names=dict(left=f'eyeLeft_lod{lod}_mesh',right=f'eyeRight_lod{lod}_mesh')
        if head_name not in goals or any(name not in goals for name in eye_names.values()):raise ValueError('Lid calibration needs corresponding head and both globe pose families')
        lookup={name:i for i,name in enumerate(names)};neutral=lookup['Neutral'];original=np.asarray(goals[head_name]);head=original.copy();rows=[]
        controls={name:self.domain[name]['controls'] for name in names}
        for name in names:
            control=controls[name];closures={role:control.get('eyeBlink'+side,0.) for role,side in [('left','L'),('right','R')]}
            if not any(closures.values()):continue
            opened={key:value for key,value in control.items() if not key.startswith('eyeBlink')};base=next((key for key in names if controls[key]==opened),None)
            if base is None:raise ValueError('Capture the corresponding open-gaze pose before blink refinement')
            i=lookup[name];globes={role:dict(positions=goals[mesh][i]@self.matrix,triangles=a['eye_triangles'][role]) for role,mesh in eye_names.items()}
            try:corrected,proof=refine_blink(original[i]@self.matrix,original[lookup[base]]@self.matrix,a['positions'],a['triangles'],a['structure'],globes,self.volumes,self.meeting,closures)
            except ValueError as error:
                if not self.allow_partial_candidate:raise ValueError(f'LOD {lod}, {name}: {error}') from error
                rows.append(dict(pose=name,status='rejected',error=str(error),retained='UNREFINED_COMPLETE_MOTION_TARGET'));continue
            head[i]=corrected@self.inverse;rows.append(dict(pose=name,proof=proof))
        result=dict(goals);result[head_name]=head;accessories=[]
        # Keep the native accessory's existing motion and standoff. Only the
        # extra head correction is transported; no reset to archetype shape.
        for mesh,points in goals.items():
            if not mesh.startswith(('eyeshell_','eyelashes_','eyeEdge_')):continue
            binding=bind(original[neutral],a['triangles'],points[neutral],maximum_distance=self.maximum_attachment_cm)
            delta=np.einsum('nv,knvj->knj',binding['barycentric'],(head-original)[:,binding['triangles']]);maximum=float(np.linalg.norm(delta,axis=2).max(initial=0))
            if maximum>self.maximum_attachment_cm:raise ValueError('A native ocular attachment exceeds its correction bound')
            result[mesh]=points+delta;accessories.append(dict(mesh=mesh,method='ADDITIONAL_HEAD_DISPLACEMENT',maximum_offset_cm=maximum,authored_motion_retained=True))
        rejected=sum(row.get('status')=='rejected' for row in rows)
        return result,dict(status='partial_candidate' if rejected else 'refined_targets',rejected_poses=rejected,poses=rows,accessories=accessories,neutral_unchanged=bool(np.array_equal(head[neutral],original[neutral])),native_motion_accepted=False)
