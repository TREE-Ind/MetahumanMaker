"""Reuse authored native LOD geometry only after full strand-group comparison."""
from pathlib import Path
import json
import re
import subprocess

KEY = 'CreateMetaHuman.GroomRepresentationSources'


def _deform(source, source_group, groom, group, python, folder):
    """Fit private imported representations; reject unsupported or poor fits."""
    import unreal
    if not python or not Path(python).is_file():return {},dict(status='unavailable',reason='numerical_python_not_configured')
    export=getattr(unreal.CreateMHGroomLibrary,'export_groom_lod_deformation',None)
    apply=getattr(unreal.CreateMHGroomLibrary,'apply_groom_lod_deformation',None)
    if not export or not apply:return {},dict(status='unavailable',reason='rebuild_bridge_for_lod_deformation')
    folder=Path(folder);folder.mkdir(exist_ok=False)
    captured=json.loads(export(source,source_group,groom,group,str(folder/'input.json')))
    if captured.get('status')!='complete':return {},captured
    with (folder/'worker.log').open('w',encoding='utf-8') as log:
        process=subprocess.run([python,str(Path(__file__).with_name('groom_lod_warp.py')),
            str(folder/'input.json'),str(folder/'result.json')],stdout=log,stderr=subprocess.STDOUT,
            timeout=120,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
    if process.returncode or not (folder/'result.json').is_file():
        return {},dict(status='failed',reason='lod_worker_failed',log=str(folder/'worker.log'))
    result=json.loads((folder/'result.json').read_text())
    if result.get('status')!='complete' or result.get('schema')!=1:
        return {},dict(status='failed',reason='invalid_lod_result')
    reports=[dict(source=row['source'],lod=row['lod'],**row['report']) for row in result['meshes']]
    rejected={row['source'] for row in reports if not row['accepted']}
    accepted=[row for row in result['meshes'] if row['source'] not in rejected]
    if not accepted:return {},dict(status='rejected',reason='mesh_quality_gates',meshes=reports)
    # Each retained asset includes all its authored source LODs. Failed assets
    # retain strand fallback rather than partially modifying one static mesh.
    proposal=dict(result,meshes=accepted,accepted=True)
    (folder/'accepted.json').write_text(json.dumps(proposal,separators=(',',':')))
    destination=groom.get_path_name().split('.')[0]+'_LODs/G'+str(group)
    installed=json.loads(apply(str(folder/'accepted.json'),destination))
    if installed.get('status')!='complete':return {},dict(status='failed',native=installed,meshes=reports)
    mapping={row['source']:unreal.load_asset(row['mesh']) for row in installed['meshes']}
    summary=dict(status='complete',meshes=reports,assets=installed['meshes'],
        native_validation=dict(original_meshes_unchanged=installed.get('original_meshes_unchanged',False),
            uv_topology_attributes_verified=installed.get('uv_topology_attributes_verified',False)),
        source_groom=result['source_groom'],target_groom=result['target_groom'],
        source_geometry=result['source_geometry'],target_geometry=result['target_geometry'],
        source_meshes=[dict(source=row['source'],lod=row['lod'],signature=row['signature']) for row in result['meshes']])
    receipt=folder/'fit.json';receipt.write_text(json.dumps(summary,indent=2))
    for original,mesh in mapping.items():
        unreal.EditorAssetLibrary.set_metadata_tag(mesh,'MetaHumanMaker.GroomLODSource',original)
        unreal.EditorAssetLibrary.set_metadata_tag(mesh,'MetaHumanMaker.GroomLODFit',str(receipt))
        if not unreal.EditorAssetLibrary.save_loaded_asset(mesh,False):raise RuntimeError('Could not save private groom LOD mesh '+mesh.get_path_name())
    # These are owned numerical transport files, not artist source data. The
    # compact receipt retains quality metrics, input signatures and asset paths.
    for filename in ('input.json','result.json','accepted.json'):(folder/filename).unlink()
    return mapping,dict(summary,receipt=str(receipt))


def validate_sources(values):
    if not isinstance(values, list) or len(values) > 128:
        raise ValueError('Groom representation sources must be a bounded list')
    result = []; seen = set()
    for value in values:
        if (not isinstance(value, dict) or set(value) != {'asset', 'group', 'project'}
                or not isinstance(value['asset'], str) or not value['asset'].startswith('/')
                or not isinstance(value['project'], str) or not value['project']
                or type(value['group']) is not int or value['group'] < 0):
            raise ValueError('Groom representation source needs asset, group and project')
        identity = (value['asset'].casefold(), value['group'], value['project'].casefold())
        if identity not in seen:
            result.append(dict(value)); seen.add(identity)
    return result


def project_path(unreal):
    return str(Path(unreal.Paths.convert_relative_path_to_full(unreal.Paths.get_project_file_path())).resolve())


def export_sources(groom, group):
    import unreal
    text = unreal.EditorAssetLibrary.get_metadata_tag(groom, KEY)
    inherited = {}
    if text:
        data = json.loads(text)
        if data.get('schema') != 1 or not isinstance(data.get('groups'), dict):
            raise ValueError('Unsupported groom representation ancestry')
        inherited = data['groups']
    current = []
    if any(int(row.get_editor_property('group_index')) == group for name in ('hair_groups_cards', 'hair_groups_meshes')
           for row in groom.get_editor_property(name)):
        current.append(dict(asset=groom.get_path_name(), group=group, project=project_path(unreal)))
    return validate_sources(current + inherited.get(str(group), []))


def _slot(text):
    match = re.search(r'\bMaterialSlotName=("(?:\\.|[^"\\])*"|[^,)]+)', text)
    if not match:
        raise ValueError('Native LOD description has no material slot')
    token = match.group(1)
    return (json.loads(token) if token.startswith('"') else token), match


def restore(groom, records, group_ids, slot_sources, material_key, *, lod_python='', job_folder=None):
    import unreal
    library = unreal.EditorAssetLibrary
    compare = getattr(unreal.CreateMHGroomLibrary, 'compare_groom_groups', None)
    indexed = {int(record['id']): record for record in records}
    materials = list(groom.get_editor_property('hair_groups_materials'))
    cards = list(groom.get_editor_property('hair_groups_cards'))
    meshes = list(groom.get_editor_property('hair_groups_meshes'))
    if cards or meshes:
        raise ValueError('Representation restoration requires a fresh strand import')
    raw = library.get_metadata_tag(groom, material_key)
    provenance = json.loads(raw)['slots'] if raw else {}
    ancestry = {}; report = []
    for index, group_id in enumerate(group_ids):
        sources = validate_sources(indexed.get(group_id, {}).get('representation_sources', []))
        ancestry[str(index)] = sources
        row = dict(group=index, reused=False, candidates=[])
        for origin in sources:
            attempt = dict(source=origin); row['candidates'].append(attempt)
            if Path(origin['project']).resolve() != Path(project_path(unreal)):
                attempt['reason'] = 'different_project'; continue
            if not compare:
                attempt['reason'] = 'native_bridge_comparison_unavailable'; break
            source = unreal.load_asset(origin['asset'])
            if not isinstance(source, unreal.GroomAsset) or source == groom:
                attempt['reason'] = 'source_unavailable'; continue
            source_cards = [r for r in source.get_editor_property('hair_groups_cards') if int(r.get_editor_property('group_index')) == origin['group']]
            source_meshes = [r for r in source.get_editor_property('hair_groups_meshes') if int(r.get_editor_property('group_index')) == origin['group']]
            if not source_cards and not source_meshes:
                attempt['reason'] = 'no_authored_representations'; continue
            match = json.loads(compare(source, origin['group'], groom, index))
            attempt['comparison'] = match
            # Imported meshes are immutable shared dependencies. Procedural
            # editor recipes need a private object copy before their reuse.
            if any(r.get_editor_property('imported_mesh') is None for r in source_cards + source_meshes):
                attempt['reason'] = 'imported_mesh_unavailable'; continue
            if any(re.search(r'\bGenerationSettings=(?!None\b)', r.export_text()) for r in source_cards):
                attempt['reason'] = 'private_generator_recipe_required'; continue
            source_materials = {str(m.get_editor_property('slot_name')): m for m in source.get_editor_property('hair_groups_materials')}
            names = [_slot(r.export_text())[0] for r in source_cards + source_meshes]
            if any(name not in source_materials for name in names):
                attempt['reason'] = 'source_material_slot_unavailable'; continue
            fitted={}
            if not match.get('matches'):
                if match.get('reason')!='positions_changed' or job_folder is None:continue
                try:
                    fitted,fit=_deform(source,origin['group'],groom,index,lod_python,
                        Path(job_folder)/('lod-fit-g%d-a%d'%(index,len(row['candidates']))))
                except (OSError,ValueError,RuntimeError,subprocess.SubprocessError) as exc:
                    attempt['deformation']=dict(status='failed',reason=str(exc));continue
                attempt['deformation']=fit
                if not fitted:continue
                source_cards=[r for r in source_cards if r.get_editor_property('imported_mesh').get_path_name() in fitted]
                source_meshes=[r for r in source_meshes if r.get_editor_property('imported_mesh').get_path_name() in fitted]
            source_materials = {str(m.get_editor_property('slot_name')): m for m in source.get_editor_property('hair_groups_materials')}
            names = [_slot(r.export_text())[0] for r in source_cards + source_meshes]
            if any(name not in source_materials for name in names):
                attempt['reason'] = 'source_material_slot_unavailable'; continue
            remap = {}; extra_materials = []; extra_provenance = {}
            for name in dict.fromkeys(names):
                unique = 'CM_Group%d_%s' % (group_id, name)
                material = source_materials[name].copy(); material.set_editor_property('slot_name', unique)
                remap[name] = unique
                if any(str(m.get_editor_property('slot_name')) == unique for m in materials):
                    continue
                extra_materials.append(material)
                extra_provenance[unique] = slot_sources(source).get(name.casefold(), [])
            def converted(values):
                result = []
                for original in values:
                    text = original.export_text(); name, found = _slot(text)
                    copy = original.copy(); copy.import_text(text[:found.start(1)] + json.dumps(remap[name]) + text[found.end(1):])
                    copy.set_editor_property('group_index', index); result.append(copy)
                    if fitted:copy.set_editor_property('imported_mesh',fitted[original.get_editor_property('imported_mesh').get_path_name()])
                return result
            cards.extend(converted(source_cards)); meshes.extend(converted(source_meshes))
            materials.extend(extra_materials); provenance.update(extra_provenance)
            row.update(reused=True, source=origin, cards=len(source_cards), meshes=len(source_meshes), shared_dependencies_unchanged=True)
            if fitted:row.update(deformed=True,private_meshes=[mesh.get_path_name() for mesh in fitted.values()])
            break
        report.append(row)
    if cards or meshes:
        never = unreal.PropertyAccessChangeNotifyMode.NEVER
        groom.set_editor_property('hair_groups_materials', materials, notify_mode=never)
        groom.set_editor_property('hair_groups_cards', cards, notify_mode=never)
        groom.set_editor_property('hair_groups_meshes', meshes, notify_mode=never)
        library.set_metadata_tag(groom, material_key, json.dumps(dict(schema=1, slots=provenance)))
        groom.set_editor_property('hair_groups_lod', list(groom.get_editor_property('hair_groups_lod')),
                                  notify_mode=unreal.PropertyAccessChangeNotifyMode.ALWAYS)
    library.set_metadata_tag(groom, KEY, json.dumps(dict(schema=1, groups=ancestry)))
    return report
