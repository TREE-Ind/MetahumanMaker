"""Persistent per-source-policy distance limits for successive regional fits.

The ledger is a Blender ID property, so geometry and limits share project
save and Undo/Redo. No source positions or character-specific rules are stored.
"""
import base64
import json
import zlib
import numpy as np
from .vertex_contract import fingerprint

PROPERTY='cm_source_distance_budget'


def _context(data):
    return dict(version=1,vertices=int(np.max(data['weld']))+1,
        geometry={key:fingerprint(data[key]) for key in
            ('source','source_triangles','parameter_reference','triangles','weld','budget_target_matrix')})


def _load(text,context):
    if not text:return dict(context=context,policies={})
    value=json.loads(text)
    if value.get('context')!=context:return dict(context=context,policies={})
    if not isinstance(value.get('policies'),dict):raise ValueError('Invalid saved source-distance budget')
    return value


def _key(faces):return fingerprint(np.sort(np.asarray(faces,dtype=np.int64)))['sha256']


def _decode(encoded,n):
    raw=zlib.decompress(base64.b64decode(encoded,validate=True))
    caps=np.frombuffer(raw,dtype='<f8').copy()
    if caps.shape!=(n,) or not np.isfinite(caps).all() or np.any(caps<0):raise ValueError('Invalid saved source-distance limits')
    return caps


def _encode(caps):return base64.b64encode(zlib.compress(np.asarray(caps,dtype='<f8').tobytes())).decode('ascii')


def observed_support(entry,n):
    """Read explicit source observations, independently of distance limits.

    An older distance-only policy cannot prove how its limits were created:
    local Polish also creates valid limits without observing supported skin.
    Keep those limits, but require a new Accurate Fit to certify projection.
    """
    encoded=entry.get('observed')
    if encoded is None:return np.zeros(n,dtype=bool)
    raw=zlib.decompress(base64.b64decode(encoded,validate=True))
    values=np.frombuffer(raw,dtype=np.uint8)
    if values.shape!=(n,) or np.any(values>1):raise ValueError('Invalid saved source-observation support')
    result=values.astype(bool)
    if np.any(result&(_decode(entry['caps'],n)<=0)):raise ValueError('Source observations require certified distance limits')
    return result


def _encode_observed(values):
    return base64.b64encode(zlib.compress(np.asarray(values,dtype=np.uint8).tobytes())).decode('ascii')


def read_policies(data):
    """Read matching immutable policies for a proposal-side projection."""
    context=json.loads(bytes(data['source_budget_context']).decode())
    previous=bytes(data['source_budget_previous']).decode();ledger=_load(previous,context)
    result=[]
    for key,entry in ledger['policies'].items():
        faces=np.frombuffer(zlib.decompress(base64.b64decode(entry['faces'],validate=True)),dtype='<i8').copy()
        if _key(faces)!=key:raise ValueError('Persistent source policy fingerprint mismatch')
        result.append(dict(key=key,faces=faces,caps=_decode(entry['caps'],context['vertices'])))
    return result


def snapshot(previous,data):
    """Export only matching source/topology/policy limits to a worker snapshot."""
    context=_context(data);ledger=_load(previous,context);n=context['vertices']
    output=dict(source_budget_context=np.frombuffer(json.dumps(context,sort_keys=True).encode(),dtype=np.uint8).copy(),
        source_budget_previous=np.frombuffer(previous.encode(),dtype=np.uint8).copy())
    for key in data:
        if key.startswith('source_faces_'):
            index=key.removeprefix('source_faces_');encoded=ledger['policies'].get(_key(data[key]))
            caps=_decode(encoded['caps'],n) if encoded else np.zeros(n)
            # Nested policies share valid distance bounds. A narrower new
            # policy is conservative; a broader one gets a necessary bound.
            # Native checks still certify distance to the original policy.
            for entry in ledger['policies'].values():
                if entry is encoded:continue
                old_faces=np.frombuffer(zlib.decompress(base64.b64decode(entry['faces'],validate=True)),dtype='<i8')
                if np.isin(data[key],old_faces).all() or np.isin(old_faces,data[key]).all():
                    inherited=_decode(entry['caps'],n)
                    caps=np.where((caps>0)&(inherited>0),np.minimum(caps,inherited),np.maximum(caps,inherited))
            output['inherited_source_caps_'+index]=caps
    return output


def constrain(data,ids,patches,public_caps,solver_caps):
    """Intersect new budgets with prior limits without adding numerical padding.

    A point can start exactly at its inherited cap. Native storage retraction
    reserves precision inside that same cap if rounding would exceed it.
    """
    ids=np.asarray(ids,dtype=np.int64);patches=np.asarray(patches,dtype=np.int64)
    public=np.array(public_caps,dtype=float,copy=True);inner=np.array(solver_caps,dtype=float,copy=True)
    if public.shape!=ids.shape or inner.shape!=ids.shape or patches.shape!=ids.shape:raise ValueError('Invalid inherited source-budget layout')
    for index in np.unique(patches):
        inherited=data.get('inherited_source_caps_'+str(index))
        if inherited is None:continue
        selected=patches==index;caps=np.asarray(inherited,dtype=float)[ids[selected]]
        if not np.isfinite(caps).all() or np.any(caps<0):raise ValueError('Invalid inherited source-distance caps')
        public[selected]=np.where(caps>0,np.minimum(public[selected],caps),public[selected])
    inner=np.minimum(inner,public)
    if not np.isfinite(public).all() or np.any(public<=0) or np.any(inner<=0):raise ValueError('Source-distance caps must be positive')
    return public,inner


def updated(previous,proof,*,certify_observations=False):
    """Prepare a monotone ledger before the native transaction changes anything."""
    context=json.loads(bytes(proof['source_budget_context']).decode());ledger=_load(previous,context);n=context['vertices']
    ids=proof.get('constrained_ids',proof['selected']);caps=np.asarray(proof['maximum_offset'],dtype=float);patches=proof['patch_ids']
    if ids.ndim!=1 or patches.shape!=ids.shape or caps.shape!=ids.shape or np.any(ids<0) or np.any(ids>=n) or not np.isfinite(caps).all() or np.any(caps<=0):raise ValueError('Invalid certified source-distance caps')
    for index in np.unique(patches):
        key=_key(proof['policy_faces_'+str(index)]);encoded=ledger['policies'].get(key)
        values=_decode(encoded['caps'],n) if encoded else np.zeros(n);use=patches==index;vertices=ids[use]
        observed=observed_support(encoded,n) if encoded else np.zeros(n,dtype=bool)
        if np.any((values[vertices]>0)&(caps[use]>values[vertices]+1e-15)):raise ValueError('Fit attempts to increase a persistent source-distance limit')
        values[vertices]=np.where(values[vertices]>0,np.minimum(values[vertices],caps[use]),caps[use])
        if certify_observations:observed[vertices]=True
        faces=np.asarray(proof['policy_faces_'+str(index)],dtype='<i8')
        ledger['policies'][key]=dict(caps=_encode(values),faces=base64.b64encode(zlib.compress(faces.tobytes())).decode('ascii'),observed=_encode_observed(observed))
    return json.dumps(ledger,sort_keys=True,separators=(',',':'))


def validate_native(actual,proof):
    """Check all prior applicable policies, including newly inferred vertices."""
    from mathutils.bvhtree import BVHTree
    from .vertex_contract import witnesses,audit_witnesses
    from .source_surface_interior import native_query
    if 'source_budget_context' not in proof:return dict(applicable=False,checked_vertices=0)
    context=json.loads(bytes(proof['source_budget_context']).decode())
    previous=bytes(proof.get('source_budget_previous',np.empty(0,dtype=np.uint8))).decode()
    ledger=_load(previous,context);weld=proof['weld'];n=len(np.bincount(weld))
    editable=np.isin(weld,proof['selected']);checked=0;worst=0.;cache=proof.setdefault('_native_source_trees',{})
    for key,entry in ledger['policies'].items():
        caps=_decode(entry['caps'],n);use=editable&(caps[weld]>0)
        if not use.any():continue
        faces=np.frombuffer(zlib.decompress(base64.b64decode(entry['faces'],validate=True)),dtype='<i8')
        if _key(faces)!=key:raise ValueError('Persistent source policy fingerprint mismatch')
        tree_key='budget_'+key
        if tree_key not in cache:cache[tree_key]=BVHTree.FromPolygons(proof['source'].tolist(),proof['source_triangles'][faces].tolist(),all_triangles=True)
        # A float32 BVH face chosen at the physical-family average can be
        # wrong for a stored canonical seam copy near an edge. Enumerate the
        # BVH range conservatively, then compare original facets in float64
        # at each actual stored coordinate, as in native interior validation.
        chosen=native_query(cache[tree_key],proof['source'],proof['source_triangles'],faces,actual[use])
        record=witnesses(actual[use],proof['source'],proof['source_triangles'],chosen['source_triangle_ids'])
        audit_witnesses(record,proof['source'],proof['source_triangles']);excess=record['residual']-caps[weld[use]]
        if np.any(excess>1e-12):raise ValueError('Stored fit exceeds persistent source-detail bounds')
        checked+=int(use.sum());worst=max(worst,float(excess.max(initial=0)))
    return dict(applicable=bool(ledger['policies']),checked_vertices=checked,maximum_excess_m=worst,prior_policy_limits_preserved=True)


def storage_constraints(data,selected,weld):
    """Prepare exact prior-policy queries for external storage-cell repair."""
    if 'source_budget_context' not in data:return []
    import igl
    from .vertex_contract import witnesses
    context=json.loads(bytes(data['source_budget_context']).decode())
    previous=bytes(data.get('source_budget_previous',np.empty(0,dtype=np.uint8))).decode()
    ledger=_load(previous,context);weld=np.asarray(weld);n=len(np.bincount(weld))
    editable=np.isin(weld,selected);constraints=[]
    source=data['source'];triangles=data['source_triangles']
    for key,entry in ledger['policies'].items():
        caps=_decode(entry['caps'],n);ids=np.flatnonzero(editable&(caps[weld]>0))
        if not len(ids):continue
        faces=np.frombuffer(zlib.decompress(base64.b64decode(entry['faces'],validate=True)),dtype='<i8')
        if _key(faces)!=key:raise ValueError('Persistent source policy fingerprint mismatch')
        policy=triangles[faces];tree=igl.AABB();tree.init(source,policy)
        def query(points,tree=tree,policy=policy,faces=faces):
            _,local,_=tree.squared_distance(source,policy,points)
            return witnesses(points,source,triangles,faces[local])['residual']
        constraints.append((ids,caps[weld[ids]],query))
    return constraints
