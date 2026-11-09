"""Exact source-face authority alongside existing paired vertex masks."""
import hashlib
import numpy as np

PREFIX='CM_SourceFaces_'


def attribute_name(region):
    name=PREFIX+str(region)
    return name if len(name)<=63 else PREFIX+hashlib.sha256(str(region).encode()).hexdigest()[:32]


def values(obj,region):
    attribute=obj.data.attributes.get(attribute_name(region))
    if attribute is None:return None
    if attribute.domain!='FACE' or attribute.data_type!='BOOLEAN':raise ValueError('Source region needs a Boolean face attribute: '+str(region))
    result=np.empty(len(obj.data.polygons),dtype=bool);attribute.data.foreach_get('value',result)
    return result


def write(obj,region,face_ids):
    """Store polygon IDs without changing vertices, topology or UVs.

    The paired CM_Fit_ vertex group remains required. Source face authority
    intersects that group; marking again replaces the exact face selection.
    """
    ids=np.asarray(face_ids)
    if ids.ndim!=1 or ids.dtype.kind not in 'iu' or not len(ids) or ids.min()<0 or ids.max()>=len(obj.data.polygons):raise ValueError('Select valid complete source faces')
    previous=values(obj,region);attribute=obj.data.attributes.get(attribute_name(region))
    if attribute is None:attribute=obj.data.attributes.new(name=attribute_name(region),type='BOOLEAN',domain='FACE')
    selected=np.zeros(len(obj.data.polygons),dtype=bool);selected[ids]=True
    try:attribute.data.foreach_set('value',selected);obj.data.update()
    except BaseException:
        if previous is None:obj.data.attributes.remove(attribute)
        else:attribute.data.foreach_set('value',previous)
        raise
    return int(selected.sum())


def triangle_mask(obj,region):
    selected=values(obj,region)
    if selected is None:return None
    return selected[np.fromiter((face.polygon_index for face in obj.data.loop_triangles),dtype=np.int64,count=len(obj.data.loop_triangles))]


def stamp(obj):
    digest=hashlib.sha256()
    for attribute in sorted((a for a in obj.data.attributes if a.name.startswith(PREFIX)),key=lambda a:a.name):
        digest.update(attribute.name.encode());digest.update(attribute.domain.encode());digest.update(attribute.data_type.encode())
        if attribute.domain!='FACE' or attribute.data_type!='BOOLEAN':raise ValueError('Invalid exact source-face region')
        selected=np.empty(len(attribute.data),dtype=bool);attribute.data.foreach_get('value',selected);digest.update(selected.tobytes())
    return digest.digest()
