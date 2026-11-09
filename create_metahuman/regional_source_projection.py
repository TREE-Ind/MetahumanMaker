"""Proposal-side projection into saved source limits; native checks remain final."""
import numpy as np


class Projector:
    def __init__(self,data,reference):
        from .source_budget import read_policies
        self.source=data['source'];self.triangles=data['source_triangles'];self.weld=data['weld']
        self.policies=read_policies(data);self.count=np.bincount(self.weld);self.trees={}
        self.native_reserve=max(float(np.linalg.norm(np.ptp(reference,axis=0)))*2e-6,1e-10)
        physical=self.physical(reference);self.spread=np.zeros(len(physical))
        np.maximum.at(self.spread,self.weld,np.linalg.norm(reference-physical[self.weld],axis=1))
        self.queries=0;self.maximum_correction=0.;self.calls=0

    def physical(self,points):
        return np.column_stack([np.bincount(self.weld,weights=points[:,axis])/self.count for axis in range(3)])

    def __call__(self,points,active,*,cancelled=None):
        from mathutils import Vector
        from mathutils.bvhtree import BVHTree
        from .vertex_contract import witnesses,audit_witnesses
        before=self.physical(points);current=before.copy();self.calls+=1
        for _ in range(3):
            corrected=False
            for policy in self.policies:
                if cancelled and cancelled():raise RuntimeError('Regional polish cancelled; previous geometry retained')
                ids=active[policy['caps'][active]>0]
                if not len(ids):continue
                key=policy['key'];faces=policy['faces']
                if key not in self.trees:self.trees[key]=BVHTree.FromPolygons(self.source.tolist(),self.triangles[faces].tolist(),all_triangles=True)
                nearest=np.array([self.trees[key].find_nearest(Vector(point))[2] for point in current[ids]],dtype=np.int64)
                record=witnesses(current[ids],self.source,self.triangles,faces[nearest]);audit_witnesses(record,self.source,self.triangles)
                caps=policy['caps'][ids];inner=caps-np.minimum(caps*.25,self.native_reserve+self.spread[ids]*2)
                distance=record['residual'];outside=distance>inner;self.queries+=len(ids)
                if not outside.any():continue
                selected=ids[outside];centers=record['surface_points'][outside]
                current[selected]=centers+(current[selected]-centers)*(inner[outside]/np.maximum(distance[outside],1e-30))[:,None]
                corrected=True
            if not corrected:break
        delta=current-before;self.maximum_correction=max(self.maximum_correction,float(np.linalg.norm(delta,axis=1).max(initial=0)))
        return np.asarray(points)+delta[self.weld]

    def report(self):
        return dict(method='SAVED_SOURCE_POLICY_POLISH_PROJECTION_V1',calls=self.calls,queried_vertices=self.queries,
            policies_queried=len(self.trees),maximum_world_correction=self.maximum_correction,
            original_source_budgets_unchanged=True,proposal_only=True,native_validation_required=True)
