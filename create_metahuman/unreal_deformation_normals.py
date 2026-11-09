"""Explicit, reversible tangent policy for substantially deformed rig revisions.

An archetype's vertex-color mask can retain normals from the wrong surface
after a large corrective. Apply this only to an owned revision's selected
skin sections, not to shared artist assets or every material on the face.
The caller owns saving the revision and its returned restoration record.
"""


def configure(mesh, sections, *, owner_folder):
    import unreal
    if not isinstance(mesh,unreal.SkeletalMesh) or not owner_folder.startswith('/Game/') or not mesh.get_path_name().startswith(owner_folder.rstrip('/')+'/'):
        raise ValueError('Choose a skeletal mesh inside the owned revision folder')
    sections=list(sections)
    if not sections or any(len(row)!=2 or any(type(v) is not int or v<0 for v in row) for row in sections) or len(set(map(tuple,sections)))!=len(sections):
        raise ValueError('Supply distinct nonnegative LOD and skin-section pairs')
    api=unreal.get_editor_subsystem(unreal.SkeletalMeshEditorSubsystem);before=[]
    for lod,section in sections:
        recompute=api.get_section_recompute_tangent(mesh,lod,section);mask=api.get_section_recompute_tangents_vertex_mask_channel(mesh,lod,section)
        if recompute is None or mask is None:raise ValueError('A requested skin section is missing')
        before.append(dict(lod=lod,section=section,recompute=bool(recompute),mask=int(mask)))
    record=dict(schema=1,mesh=mesh.get_path_name(),owner_folder=owner_folder,previous=before,policy='FULL_DEFORMED_SURFACE_NORMALS',saved=False)
    try:
        for row in before:
            lod,section=row['lod'],row['section']
            if not row['recompute'] and not api.set_section_recompute_tangent(mesh,lod,section,True):raise RuntimeError('Could not enable revision tangent recomputation')
            # ESkinVertexColorChannel::None means recompute the entire section.
            if row['mask']!=3 and not api.set_section_recompute_tangents_vertex_mask_channel(mesh,lod,section,3):raise RuntimeError('Could not remove the obsolete archetype normal mask')
        return record
    except Exception:
        restore(mesh,record);raise


def restore(mesh,record):
    import unreal
    if record.get('schema')!=1 or mesh.get_path_name()!=record.get('mesh') or not mesh.get_path_name().startswith(record.get('owner_folder','').rstrip('/')+'/'):
        raise ValueError('The normal restoration record belongs to another revision')
    api=unreal.get_editor_subsystem(unreal.SkeletalMeshEditorSubsystem)
    for row in record['previous']:
        lod,section=row['lod'],row['section']
        if api.get_section_recompute_tangents_vertex_mask_channel(mesh,lod,section)!=row['mask']:
            if not api.set_section_recompute_tangents_vertex_mask_channel(mesh,lod,section,row['mask']):raise RuntimeError('Could not restore the revision normal mask')
        if api.get_section_recompute_tangent(mesh,lod,section)!=row['recompute']:
            if not api.set_section_recompute_tangent(mesh,lod,section,row['recompute']):raise RuntimeError('Could not restore revision tangent recomputation')
