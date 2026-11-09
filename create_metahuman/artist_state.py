"""Pure conflict planning for native artist properties; unknown fields survive."""
import copy
import json
from pathlib import Path


def compare(previous,current):
    """Describe preserved native edits without interpreting or applying them."""
    def properties(data):
        result={}
        for scope in ('reflected','actor_reflected'):
            for obj in (data.get(scope) or {}).get('objects',[]):
                result[(scope,obj['object'],'$class')]=obj['class']
                for name,value in obj['properties'].items():result[(scope,obj['object'],name)]=value
        for component in data.get('live_components',[]):
            identity=component['component']
            result[('live_components',identity,'$class')]=component['class']
            def material_fields(value,path):
                if isinstance(value,dict) and value:
                    for name,child in value.items():material_fields(child,path+'/'+name.replace('~','~0').replace('/','~1'))
                else:result[('live_components',identity,path)]=value
            for index,material in enumerate(component['materials']):material_fields(material,'Material['+str(index)+']')
        return result
    before=properties(previous or {});after=properties(current);changes=[]
    for key in sorted(set(before)|set(after)):
        if key not in before or key not in after or before[key]!=after[key]:
            changes.append({'scope':key[0],'object':key[1],'property':key[2],
                'change':'added' if key not in before else 'removed' if key not in after else 'modified',
                'before':before.get(key),'after':after.get(key)})
    old={r['relative_file']:r['sha256'] for r in (previous or {}).get('native_files',[])}
    new={r['relative_file']:r['sha256'] for r in current['native_files']}
    packages=[{'file':name,'change':'added' if name not in old else 'removed' if name not in new else 'modified'}
        for name in sorted(set(old)|set(new)) if old.get(name)!=new.get(name)]
    return {'baseline_available':previous is not None,'character':current['character'],'actor':current.get('actor',''),
        'capture_scope':current.get('capture_scope','character'),'property_changes':changes,'native_file_changes':packages,
        'objects':len((current.get('reflected') or {}).get('objects',[]))+len((current.get('actor_reflected') or {}).get('objects',[])),
        'native_files':len(current['native_files']),'external_dependencies':current['external_dependencies'],
        'visual_application':'not applied; unknown native settings remain preserved for an adapter or native restore',
        'limitations':current.get('limitations',[])}


def adopt_snapshot(scene,manifest):
    """Link preserved UE edits into the Blender project without changing meshes."""
    path=Path(manifest).resolve();data=json.loads(path.read_text(encoding='utf-8'))
    f=scene.cm_features
    if data.get('generator')!='CreateMetaHuman.UnrealArtistState' or data.get('status')!='complete':raise ValueError('The native artist-state snapshot is incomplete')
    if data.get('character')!=f.unreal_character:raise ValueError('Artist edits belong to another Unreal character')
    if f.unreal_character_project and Path(data['project']).resolve()!=Path(f.unreal_character_project).resolve():raise ValueError('Artist edits belong to another Unreal project')
    blob_root=path.parent.parent/'native-package-blobs'
    for record in data['native_files']:
        relative=Path(record['relative_file'])
        if relative.is_absolute() or '..' in relative.parts or relative.suffix not in {'.uasset','.umap','.uexp','.ubulk','.uptnl'}:raise ValueError('Invalid native package destination in artist state')
        blob=(path.parent/record['blob']).resolve()
        if blob.parent!=blob_root.resolve() or not blob.is_file():raise ValueError('A preserved native asset is missing or outside its snapshot store')
    previous=None
    if f.unreal_artist_snapshot:
        old_path=Path(f.unreal_artist_snapshot)
        if old_path.is_file():
            candidate=json.loads(old_path.read_text(encoding='utf-8'))
            if candidate.get('character')==data['character'] and candidate.get('project')==data['project']:previous=candidate
    review=compare(previous,data)
    f.unreal_artist_review=json.dumps(review)
    f.unreal_artist_snapshot=str(path)
    return {'objects':review['objects'],'files':review['native_files'],'changes':len(review['property_changes'])}


def merge(base,local,remote):
    """Three-way merge objects, treating arrays atomically and retaining conflicts.

    Conflicting values stay local until explicitly resolved. Presence is tracked
    separately from null, so deletions and explicit null remain distinguishable.
    This planner never writes native assets or silently resolves an artist edit.
    """
    missing=object();conflicts=[]
    def present(value):return {'present':value is not missing,'value':copy.deepcopy(value) if value is not missing else None}
    def same(a,b):return a is b if a is missing or b is missing else a==b
    def visit(b,l,r,path):
        if same(l,r):return copy.deepcopy(l) if l is not missing else missing
        if same(l,b):return copy.deepcopy(r) if r is not missing else missing
        if same(r,b):return copy.deepcopy(l) if l is not missing else missing
        if isinstance(l,dict) and isinstance(r,dict) and (isinstance(b,dict) or b is missing):
            result={};original={} if b is missing else b
            for key in sorted(set(original)|set(l)|set(r)):
                value=visit(original.get(key,missing),l.get(key,missing),r.get(key,missing),path+[key])
                if value is not missing:result[key]=value
            return result
        conflicts.append({'path':path,'base':present(b),'local':present(l),'unreal':present(r)})
        return copy.deepcopy(l) if l is not missing else missing
    result=visit(base,local,remote,[])
    return {'value':None if result is missing else result,'conflicts':conflicts,'ready':not conflicts}
