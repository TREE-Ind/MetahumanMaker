"""Positive fixed-area samples inside target triangles, not just vertices."""
import numpy as np

# Positive seven-point triangle quadrature, including the centroid. Keep the
# sample identities and canonical areas fixed throughout an operation.
BARY=np.array([[1/3,1/3,1/3],[.059715871789770,.470142064105115,.470142064105115],
    [.470142064105115,.059715871789770,.470142064105115],[.470142064105115,.470142064105115,.059715871789770],
    [.797426985353087,.101286507323456,.101286507323456],[.101286507323456,.797426985353087,.101286507323456],
    [.101286507323456,.101286507323456,.797426985353087]])
WEIGHT=np.array([.225,.132394152788506,.132394152788506,.132394152788506,.125939180544827,.125939180544827,.125939180544827])


class InteriorSamples:
    def __init__(self,reference,triangles,movable,metric):
        reference=np.asarray(reference);triangles=np.asarray(triangles);metric=np.asarray(metric)
        if metric.shape!=reference.shape or not np.isfinite(metric).all():raise ValueError('Interior samples need the original canonical area metric')
        self.ids=np.flatnonzero(np.any(movable[triangles],axis=1))
        self.triangles=triangles[self.ids].copy()
        p=metric[self.triangles];area=np.linalg.norm(np.cross(p[:,1]-p[:,0],p[:,2]-p[:,0]),axis=1)*.5
        if not len(area) or np.any(area<=0):raise ValueError('Interior source measurement needs nondegenerate canonical triangles')
        self.weights=(area[:,None]*WEIGHT[None]).ravel();self.area=float(area.sum())
        self.normalized=self.weights/self.weights.sum()

    def points(self,current):
        return np.einsum('ki,nij->nkj',BARY,current[self.triangles]).reshape(-1,3)

    def measure(self,current,query):
        distance=query(self.points(current))['residual']
        return dict(rms_m=float(np.sqrt(self.normalized@(distance*distance))),max_m=float(distance.max()),
            triangles=len(self.ids),samples=len(distance),canonical_area_m2=self.area)

    def energy(self,current,query):
        points=self.points(current);record=query(points);delta=points-record['surface_points']
        value=.5*float(self.normalized@np.sum(delta*delta,axis=1))
        sample_grad=(self.normalized[:,None]*delta).reshape(-1,len(BARY),3)
        corner_grad=np.einsum('ki,nkj->nij',BARY,sample_grad)
        gradient=np.zeros_like(current)
        for corner in range(3):np.add.at(gradient,self.triangles[:,corner],corner_grad[:,corner])
        return value,gradient


def native_query(tree,source,triangles,policy,points):
    """Refine every BVH range candidate in float64; no fixed-k face search."""
    from mathutils import Vector
    from .vertex_contract import closest_triangle_points
    guard=max(float(np.max(np.abs(source),initial=0)),float(np.max(np.abs(points),initial=0)),1.)*np.finfo(np.float32).eps*256
    owners=[];faces=[]
    for i,point in enumerate(points):
        hit=tree.find_nearest(Vector(point))
        if hit[2] is None:raise ValueError('Missing native triangle-interior source witness')
        facet=source[triangles[policy[int(hit[2])]]][None]
        upper=float(closest_triangle_points(point[None],facet)[2][0])
        candidates=tree.find_nearest_range(Vector(point),upper+guard)
        local={int(value[2]) for value in candidates if value[2] is not None};local.add(int(hit[2]))
        owners.extend([i]*len(local));faces.extend(policy[list(local)])
    owners=np.asarray(owners,dtype=np.int64);faces=np.asarray(faces,dtype=np.int64)
    closest,_,distance=closest_triangle_points(points[owners],source[triangles[faces]])
    order=np.lexsort((faces,distance,owners));first=np.r_[True,owners[order][1:]!=owners[order][:-1]];chosen=order[first]
    if not np.array_equal(owners[chosen],np.arange(len(points))):raise ValueError('Incomplete native triangle-interior source coverage')
    return dict(surface_points=closest[chosen],residual=distance[chosen],source_triangle_ids=faces[chosen])
