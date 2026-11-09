"""Source-normal defect regions on canonical physical surface topology."""
import numpy as np


def angle_quality(cosines):
    """Unweighted face-angle distribution; distinct from fold validity."""
    values=np.asarray(cosines,dtype=float)
    if values.ndim!=1 or not np.isfinite(values).all():raise ValueError('Invalid source-normal angular samples')
    if not len(values):return dict(faces=0)
    degrees=np.degrees(np.arccos(np.clip(values,-1,1)));q=np.percentile(degrees,[50,90,95,99,100])
    return dict(faces=len(values),median_degrees=float(q[0]),p90_degrees=float(q[1]),p95_degrees=float(q[2]),p99_degrees=float(q[3]),maximum_degrees=float(q[4]),faces_over_30_degrees=int((degrees>30).sum()))


def analyze(points,triangles,eligible,source,source_triangles,source_face_ids,*,parameter_reference=None,minimum_cosine=.25,minimum_coherence=.8,find_clusters=True):
    """Rank coherent orientation defects without a character-specific region.

    Original source facets provide the normal field. Canonical reference
    areas rank connected clusters so an already collapsed triangle does not
    make its own defect disappear from the score. Results are diagnostic;
    endpoint and native checks still decide whether a repair is acceptable.
    """
    points=np.asarray(points,dtype=float);triangles=np.asarray(triangles);eligible=np.asarray(eligible,dtype=bool);source=np.asarray(source,dtype=float);source_triangles=np.asarray(source_triangles);source_face_ids=np.asarray(source_face_ids)
    if points.ndim!=2 or points.shape[1]!=3 or not np.isfinite(points).all() or eligible.shape!=(len(points),) or source_face_ids.shape!=(len(points),):raise ValueError('Invalid source-defect vertex arrays')
    if triangles.dtype.kind not in 'iu' or triangles.ndim!=2 or triangles.shape[1]!=3 or not len(triangles) or triangles.min()<0 or triangles.max()>=len(points):raise ValueError('Invalid physical defect triangles')
    if source.ndim!=2 or source.shape[1]!=3 or not np.isfinite(source).all() or source_triangles.dtype.kind not in 'iu' or source_triangles.ndim!=2 or source_triangles.shape[1]!=3 or source_triangles.min()<0 or source_triangles.max()>=len(source) or source_face_ids.dtype.kind not in 'iu' or np.any((source_face_ids<-1)|(source_face_ids>=len(source_triangles))):raise ValueError('Invalid original source facets')
    if not -1<minimum_cosine<1 or not 0<minimum_coherence<=1:raise ValueError('Invalid source-normal confidence thresholds')
    rest=points if parameter_reference is None else np.asarray(parameter_reference,dtype=float)
    if rest.shape!=points.shape or not np.isfinite(rest).all():raise ValueError('Canonical defect reference must match the physical surface')
    observed=(source_face_ids>=0)&eligible;normals=np.zeros_like(points);facet=source[source_triangles[source_face_ids[observed]]]
    cross=np.cross(facet[:,1]-facet[:,0],facet[:,2]-facet[:,0]);normals[observed]=cross/np.maximum(np.linalg.norm(cross,axis=1)[:,None],1e-30)
    face_field=normals[triangles].sum(1);coherence=np.linalg.norm(face_field,axis=1)/3
    current=points[triangles];current_cross=np.cross(current[:,1]-current[:,0],current[:,2]-current[:,0]);current_area=np.linalg.norm(current_cross,axis=1)
    cosine=np.einsum('ij,ij->i',current_cross,face_field)/np.maximum(current_area*np.linalg.norm(face_field,axis=1),1e-30)
    corner_agreement=np.einsum('fki,fi->fk',normals[triangles],current_cross/np.maximum(current_area[:,None],1e-30))
    outward=np.all(corner_agreement>=.25,axis=1)
    # A curved, observed patch can lose mean-normal coherence while every
    # original corner direction still proves outward orientation. This adds
    # certified observations; it never drops an old protected face. Such an
    # added face has mean cosine >= .25, so cannot hide a severe defect.
    supported=np.all(observed[triangles],axis=1)&((coherence>=minimum_coherence)|outward)&(current_area>0)
    bad=supported&(cosine<minimum_cosine)
    r=rest[triangles];area=np.linalg.norm(np.cross(r[:,1]-r[:,0],r[:,2]-r[:,0]),axis=1)*.5
    mass=np.zeros(len(points));bad_mass=np.zeros(len(points));back_mass=np.zeros(len(points));back=supported&(cosine<0)
    for corner in range(3):
        np.add.at(mass,triangles[supported,corner],area[supported]/3)
        np.add.at(bad_mass,triangles[bad,corner],area[bad]/3)
        np.add.at(back_mass,triangles[back,corner],area[back]/3)
    vertices=np.flatnonzero(bad_mass>0);clusters=[]
    if len(vertices) and find_clusters:
        from scipy import sparse
        from scipy.sparse.csgraph import connected_components
        lookup=np.full(len(points),-1,int);lookup[vertices]=np.arange(len(vertices));faces=lookup[triangles[bad]]
        edges=np.unique(np.sort(np.vstack((faces[:,[0,1]],faces[:,[1,2]],faces[:,[2,0]])),axis=1),axis=0)
        graph=sparse.coo_matrix((np.ones(len(edges)*2),(np.r_[edges[:,0],edges[:,1]],np.r_[edges[:,1],edges[:,0]])),shape=(len(vertices),len(vertices))).tocsr()
        count,labels=connected_components(graph,directed=False)
        for label in range(count):
            ids=vertices[labels==label];score=(bad_mass[ids]+2*back_mass[ids])/np.maximum(mass[ids],1e-30)
            seed=int(ids[np.argmax(score)])
            component_faces=labels[faces[:,0]]==label
            back_faces=component_faces&(cosine[bad]<0)
            back_area=float(back_mass[ids].sum());total_area=float(bad_mass[ids].sum())
            clusters.append(dict(seed=seed,vertices=len(ids),severe_faces=int(component_faces.sum()),reference_area_m2=total_area,back_reference_area_m2=back_area,
                back_facing_faces=int(back_faces.sum()),repair_score_m2=total_area+2*back_area,
                seed_bad_area_fraction=float(bad_mass[seed]/max(mass[seed],1e-30)),seed_back_area_fraction=float(back_mass[seed]/max(mass[seed],1e-30))))
        clusters.sort(key=lambda row:row['repair_score_m2'],reverse=True)
    return dict(supported_faces=int(supported.sum()),outward_certified_faces=int((supported&outward&(coherence<minimum_coherence)).sum()),source_back_facing_faces=int((supported&(cosine<0)).sum()),severe_alignment_faces=int(bad.sum()),clusters=clusters),dict(face_cosine=cosine,face_supported=supported,bad_mass=bad_mass)


def repair_clusters(diagnostics):
    """Retain every observed severe component, including isolated triangles.

    A single sharply misaligned triangle is three vertices. It must not
    disappear merely because it is not reversed, or because surrounding
    correctly aligned faces dilute its vertex mass. Source observation and
    severity have already been established by ``analyze``.
    Source coherence and complete endpoint/native gates still apply.
    """
    candidates=[row for row in diagnostics['clusters'] if
        row['vertices']>=3 and row['reference_area_m2']>0]
    # Rank the area actually affected, including the existing extra weight
    # for reversed area. Broad pinching must not wait behind every isolated
    # reversed triangle now that nonlinear repair can handle either kind.
    return sorted(candidates,key=lambda row:(row.get('repair_score_m2',0.),row.get('back_reference_area_m2',0.)),reverse=True)


def compare(before,after,triangles,eligible,source,source_triangles,before_faces,after_faces,*,parameter_reference):
    """Compare a fixed canonical-area source-normal scope.

    Faces losing source-normal coherence receive the maximum error instead
    of disappearing from the measurement. This is an alignment diagnostic,
    not a certificate of global injectivity or semantic correspondence.
    """
    from .vertex_contract import resolve_facet_ties
    before_faces=np.asarray(before_faces);after_faces=np.asarray(after_faces).copy()
    ids=np.flatnonzero(np.asarray(eligible,dtype=bool)&(before_faces>=0)&(after_faces>=0))
    after_faces[ids],choice=resolve_facet_ties(np.asarray(after)[ids],source,source_triangles,before_faces[ids],after_faces[ids])
    _,old=analyze(before,triangles,eligible,source,source_triangles,before_faces,parameter_reference=parameter_reference,find_clusters=False)
    _,new=analyze(after,triangles,eligible,source,source_triangles,after_faces,parameter_reference=parameter_reference,find_clusters=False)
    scope=old['face_supported']
    if not scope.any():raise ValueError('No coherent original source-normal scope is available')
    rest=np.asarray(parameter_reference)[triangles];weight=np.linalg.norm(np.cross(rest[:,1]-rest[:,0],rest[:,2]-rest[:,0]),axis=1)[scope]
    old_cos=np.clip(old['face_cosine'][scope],-1,1);new_cos=np.where(new['face_supported'][scope],np.clip(new['face_cosine'][scope],-1,1),-1)
    old_back=old_cos<0;new_back=new_cos<0;old_severe=old_cos<.25;new_severe=new_cos<.25
    return dict(method='FIXED_CANONICAL_AREA_SOURCE_ALIGNMENT_V2',source_facet_choice=choice,faces=int(scope.sum()),
        before_angle_quality=angle_quality(old_cos),after_angle_quality=angle_quality(new_cos),
        before_energy=float(np.average(1-old_cos,weights=weight)),after_energy=float(np.average(1-new_cos,weights=weight)),
        before_back_faces=int((old_cos<0).sum()),after_back_faces=int((new_cos<0).sum()),
        new_back_faces=int((new_back&~old_back).sum()),resolved_back_faces=int((old_back&~new_back).sum()),
        new_back_reference_area_m2=float(weight[new_back&~old_back].sum()*.5),
        before_severe_faces=int((old_cos<.25).sum()),after_severe_faces=int((new_cos<.25).sum()),
        new_severe_faces=int((new_severe&~old_severe).sum()),resolved_severe_faces=int((old_severe&~new_severe).sum()),
        before_severe_reference_area_m2=float(weight[old_severe].sum()*.5),after_severe_reference_area_m2=float(weight[new_severe].sum()*.5),
        after_unmeasurable_faces=int((~new['face_supported'][scope]).sum()))
