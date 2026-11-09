"""Fixed original-surface observations for a convex reverse-coverage bound.

Each bound follows an original barycentric target point. Distance to that
point bounds distance to the complete target surface from above. This is a
proposal constraint; independent quadrature and native checks remain required.
"""
import numpy as np
from scipy import sparse


def contract(reference,triangles,reverse,*,companion_vertices=None,companion_triangles=None,reserve=0.,follow=False):
    reference=np.asarray(reference,dtype=float);tri=np.asarray(triangles,dtype=np.int64)
    points=reference;faces=tri
    if companion_vertices is not None:
        points=np.vstack((points,np.asarray(companion_vertices,dtype=float)))
        faces=np.vstack((faces,np.asarray(companion_triangles,dtype=np.int64)+len(reference)))
    ids=np.asarray(reverse['target_triangle_ids'],dtype=np.int64);bary=np.asarray(reverse['target_barycentric'],dtype=float);samples=np.asarray(reverse['source_samples'],dtype=float)
    if ids.ndim!=1 or not len(ids) or bary.shape!=(len(ids),3) or samples.shape!=bary.shape or np.any(ids<0) or np.any(ids>=len(faces)) or not np.isfinite(bary).all() or not np.isfinite(samples).all() or np.any(bary<-1e-12) or not np.allclose(bary.sum(1),1.,rtol=0,atol=1e-12) or not np.isfinite(reserve) or reserve<0:raise ValueError('Invalid original reverse-coverage observations')
    vertices=faces[ids];rows=np.repeat(np.arange(len(ids)),3);columns=vertices.ravel();values=bary.ravel();body=columns<len(reference)
    B=sparse.coo_matrix((values[body],(rows[body],columns[body])),shape=(len(ids),len(reference))).tocsr()
    fixed=np.zeros_like(samples)
    for corner in range(3):
        use=vertices[:,corner]>=len(reference)
        fixed[use]+=bary[use,corner,None]*points[vertices[use,corner]]
    target=samples-fixed;residual=np.linalg.norm(B@reference-target,axis=1);maximum=float(residual.max())
    result=dict(matrix=B,target=target,radii=np.full(len(ids),maximum+reserve),
        report=dict(method='ORIGINAL_TARGET_BARYCENTRIC_REVERSE_COVERAGE_BALLS_V1',samples=len(ids),original_maximum_m=maximum,reserve_m=float(reserve),original_source_samples_unchanged=True,nearest_surface_revalidation_required=True))
    if follow:
        result['transport']=CoverageTransport(len(reference),faces,samples,points[len(reference):])
        result['report']=dict(result['report'],method='FIXED_SOURCE_SAMPLES_TRANSPORTED_TARGET_COVERAGE_V1',original_radius_unchanged=True)
    return result


def affine(current,free,contract):
    matrix,target=contract['transport'](current) if 'transport' in contract else (contract['matrix'],contract['target'])
    return dict(matrix=sparse.kron(matrix[:,free],sparse.eye(3),format='csr'),
        offset=matrix@current-target,radii=contract['radii'])


class CoverageTransport:
    """Refresh only target witnesses for fixed original source observations.

    Each witness remains a point on a current target triangle, so its ball
    still bounds nearest-target distance from above. No sample, scope, or
    original radius is recomputed from an intermediate result.
    """
    def __init__(self,count,triangles,samples,companion):
        self.count=count;self.triangles=np.array(triangles,copy=True)
        self.samples=np.array(samples,copy=True);self.companion=np.array(companion,copy=True)

    def __call__(self,current):
        import igl
        from .vertex_contract import closest_triangle_points
        points=np.vstack((current,self.companion));tree=igl.AABB();tree.init(points,self.triangles)
        squared,ids,_=tree.squared_distance(points,self.triangles,self.samples)
        vertices=self.triangles[ids];_,bary,_=closest_triangle_points(self.samples,points[vertices])
        rows=np.repeat(np.arange(len(ids)),3);columns=vertices.ravel();body=columns<self.count
        B=sparse.coo_matrix((bary.ravel()[body],(rows[body],columns[body])),shape=(len(ids),self.count)).tocsr()
        fixed=np.zeros_like(self.samples)
        for corner in range(3):
            use=vertices[:,corner]>=self.count
            fixed[use]+=bary[use,corner,None]*points[vertices[use,corner]]
        target=self.samples-fixed
        residual=np.linalg.norm(B@current-target,axis=1)
        if not np.allclose(residual,np.sqrt(np.maximum(squared,0)),atol=1e-10,rtol=1e-9):raise ValueError('Transported coverage witness disagrees with its target triangle')
        return B,target
