"""Immutable, project-owned recipes for replaying color-derived skin shading."""
import hashlib
import json
from pathlib import Path


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream,'sha256').hexdigest()


def resolve(root,record):
    path=(root/record['file']).resolve()
    if not path.is_relative_to(root) or digest(path)!=record['sha256']:
        raise ValueError('A retained skin-normal input changed or left the character project')
    return path


def reference(root,path,sha=None):
    path=Path(path).resolve()
    if not path.is_relative_to(root):raise ValueError('Retain skin-normal inputs inside the character project')
    actual=digest(path)
    if sha and actual!=sha:raise ValueError('The retained skin-normal history changed')
    return {'file':path.relative_to(root).as_posix(),'sha256':actual}


def recover(root,normal_sha,stamp):
    """Recover only an exact applied-output/input chain, never guess a baseline."""
    retained=[]
    for path in (root/'SkinLayers').glob('*/bakes/*.json'):
        record=json.loads(path.read_text(encoding='utf-8'));recipe=record.get('recipe',{})
        if (record.get('normal_sha256')==normal_sha and recipe.get('baseline',{}).get('file')
            and all(recipe.get('stamp',{}).get(k)==stamp.get(k) for k in ('role','geometry','uv'))):
            validate(root,recipe,stamp);retained.append(recipe)
    if retained:
        if len({json.dumps(v,sort_keys=True) for v in retained})!=1:raise ValueError('Retained bake normal history is ambiguous')
        return json.loads(json.dumps(retained[0]))
    jobs={}
    for path in (root/'SkinSolves').glob('*/state.json'):
        try:
            state=json.loads(path.read_text(encoding='utf-8'))
            sha=state.get('outputs',{}).get(state.get('prefix','')+'_NORMAL.png')
            old=state.get('stamp',{})
            if (state.get('status')=='applied' and sha and
                all(old.get(key)==stamp.get(key) for key in ('role','geometry','uv'))):
                jobs.setdefault(sha,[]).append((path.parent,state))
        except (OSError,ValueError):
            continue
    chain=[];seen=set();current=normal_sha
    while current in jobs:
        if current in seen or len(jobs[current])!=1:
            raise ValueError('Skin-normal history is ambiguous; retain the current maps')
        seen.add(current)
        folder,state=jobs[current][0]
        if state.get('normal_recipe'):
            result=json.loads(json.dumps(state['normal_recipe']))
            chain.reverse()
            break
        request=json.loads((folder/'request.json').read_text(encoding='utf-8'))
        chain.append((folder,request))
        current=request['inputs']['normal']['sha256']
    else:
        if not chain:return None
        folder,request=chain[-1]
        base=request['inputs']['normal']
        result={'schema':1,'baseline':reference(root,folder/base['file'],base['sha256']),
                'steps':[],'stamp':{key:stamp[key] for key in ('role','geometry','uv','sculpt') if key in stamp}}
        chain.reverse()
    for folder,request in chain:
        inputs=request['inputs'];steps=[]
        def retained(key):
            value=inputs[key]
            return reference(root,folder/value['file'],value['sha256'])
        if request.get('surface_repair'):
            steps.append({'kind':'REPAIR','mask':retained('normal_mask'),
                          'reference':retained('reference_normal'),'strength':request.get('strength',1.)})
        if request.get('relief_strength'):
            steps.append({'kind':'RELIEF','mask':retained('mask'),
                          'strength':request['relief_strength'],'radius':request.get('relief_radius',3.)})
            result['previous_albedo_sha256']=digest(folder/'candidate-albedo.png')
        if steps:steps[-1]['quantize_after']=True
        result['steps'].extend(steps)
    return result


def validate(root,recipe,stamp):
    if recipe.get('schema')!=1 or not recipe.get('steps'):
        raise ValueError('The skin-normal recipe is incomplete')
    for key,value in recipe['stamp'].items():
        if stamp.get(key)!=value:
            raise ValueError('The retained skin-normal baseline predates the current geometry, UVs or sculpt')
    resolve(root,recipe['baseline'])
    for step in recipe['steps']:
        if step['kind']=='RETAIN':continue
        if step['kind']=='SURFACE_PATCH':
            for key in ('before','after'):resolve(root,step[key])
            for key in ('cavity_before','cavity_after'):
                if key in step:resolve(root,step[key])
            continue
        if step['kind']=='HEIGHT':
            resolve(root,step['field']);continue
        if step['kind'] not in {'REPAIR','RELIEF'}:raise ValueError('Unknown skin-normal recipe operation')
        resolve(root,step['mask'])
        if step['kind']=='REPAIR':resolve(root,step['reference'])
        if step.get('color'):resolve(root,step['color'])


def original_baseline(root,normal_sha,stamp,recipe=None):
    """Follow exact immutable shading history to its pre-repair neutral normal."""
    root=Path(root).resolve();current=normal_sha;seen=set();path=None
    while True:
        if current in seen:raise ValueError('Skin-neutral history contains a cycle; retain the current maps')
        seen.add(current)
        current_recipe=recipe if recipe is not None else recover(root,current,stamp)
        recipe=None
        if not current_recipe:return path
        validate(root,current_recipe,stamp)
        basis=current_recipe.get('neutral_baseline') or current_recipe['baseline']
        if basis['sha256']==current and (current_recipe.get('neutral_baseline') or
                any(step['kind']!='RETAIN' for step in current_recipe['steps'])):
            return resolve(root,basis)  # Verified byte-identical no-op surface.
        if basis['sha256']==current and all(step['kind']=='RETAIN' for step in current_recipe['steps']):
            # Older regional material jobs sealed their exact result for
            # replay. Their separately retained clean material input is the
            # original neutral; never treat the self-reference as geometry.
            candidates=[]
            for state_path in (root/'SkinSolves').glob('*/state.json'):
                try:state=json.loads(state_path.read_text(encoding='utf-8'))
                except (OSError,ValueError):continue
                if (state.get('status')!='applied' or not state.get('preserve_material_region') or
                    state.get('outputs',{}).get(state.get('prefix','')+'_NORMAL.png')!=current):continue
                if state.get('normal_recipe')!=current_recipe:raise ValueError('Sealed skin-neutral history disagrees with the accepted normal recipe')
                original=state.get('material_baselines',{}).get('NORMAL')
                request=json.loads((state_path.parent/'request.json').read_text(encoding='utf-8'))
                result=json.loads((state_path.parent/'result.json').read_text(encoding='utf-8'))
                if (not original or request.get('inputs',{}).get('normal',{}).get('sha256')!=original['sha256'] or
                    result.get('outputs',{}).get('candidate-normal.png')!=current):
                    raise ValueError('The original material neutral is not verified by its solve history')
                resolve(root,original);candidates.append(original)
            if not candidates or len({item['sha256'] for item in candidates})!=1:
                raise ValueError('The original sealed skin neutral is missing or ambiguous; choose its saved Original Neutral Image')
            basis=candidates[0]
        path=resolve(root,basis)
        if basis['sha256']==current:return path
        current=basis['sha256']
