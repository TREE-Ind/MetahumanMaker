"""Receipt-backed local Unreal groom exchange; network threads never access bpy."""
from pathlib import Path
import json
import re
import shutil
import threading
import uuid
import ast

import bpy
from bpy.app.handlers import persistent
from . import bridge, process_ref

_JOB = None


def _review_project(record):
    project=record.get('unreal_project')
    if project:
        if not isinstance(project,str) or Path(project).suffix.lower()!='.uproject':raise ValueError('The saved review has an invalid Unreal project path')
        return Path(project).resolve()
    # Earlier receipts still identify their project's Content files. Infer
    # only an unambiguous existing project, never from an asset name alone.
    roots=set()
    for name in record.get('source_fingerprints',{}):
        file=Path(name)
        if not file.is_absolute():continue
        roots.update(parent.parent for parent in file.parents if parent.name.lower()=='content')
    candidates={file.resolve() for root in roots if root.is_dir() for file in root.glob('*.uproject') if file.is_file()}
    if len(candidates)!=1:raise ValueError('Cannot identify this older review\'s Unreal project; use its original Blender project')
    return next(iter(candidates))


def find_replacement_reviews(context):
    """Bounded local discovery for the recovery picker; no native dispatch."""
    from datetime import datetime
    root=Path(bpy.path.abspath(context.scene.cm_settings.project_dir))/'Grooms/.exchange'
    if not root.is_dir():return []
    linked=context.scene.cm_features.unreal_character_project
    linked=Path(bpy.path.abspath(linked)).resolve() if linked else None
    name=context.scene.cm_settings.unreal_project.strip().casefold()
    entries=[]
    for child in root.iterdir():
        try:
            file=child/'preview.json'
            if file.is_file():entries.append((file.stat().st_mtime,file))
        except OSError:continue
    result=[]
    for modified,file in sorted(entries,reverse=True)[:512]:
        try:
            if file.stat().st_size>4*1024**2:continue
            record=json.loads(file.read_text(encoding='utf-8-sig'))
            if not isinstance(record,dict) or record.get('mode') not in {'PREPARING','PREVIEW','SAVING','SAVED','RESTORING'}:continue
            if (any(not isinstance(record.get(key),str) or not record[key] for key in ('actor','component'))
                    or any(not isinstance(record.get(key,''),str) for key in ('blueprint','actor_label'))):continue
            if Path(record.get('origin','')).resolve()!=file.parent.resolve():continue
            project=_review_project(record)
            if linked and project!=linked or name and project.stem.casefold()!=name:continue
            actor=record.get('actor_label') or record.get('blueprint','').rsplit('/',1)[-1].split('.')[0] or 'Character'
            if not linked and not name:actor=project.stem+': '+actor
            label=f"{actor} / {record.get('component','Hair')} / {record['mode'].title()} / {datetime.fromtimestamp(modified):%b %d %H:%M}"
            result.append(dict(path=str(file.resolve()),label=label,description=record.get('actor','')))
        except (OSError,ValueError,TypeError):continue
    return result


def recover_replacement(context,filepath):
    """Reconnect a saved review; no Unreal action or file write is dispatched."""
    from . import unreal_jobs
    if running() or unreal_jobs.running():raise ValueError('Wait for the current Unreal action before recovering a review')
    pending=context.scene.cm_groom.exchange_receipt
    if pending:
        receipt=Path(bpy.path.abspath(pending))
        if (receipt.parent/'request.json').is_file() and not receipt.is_file():
            resume(context)
            raise ValueError('Observing the previous groom exchange; wait for its receipt before recovering another review')
    file=Path(bpy.path.abspath(str(filepath))).resolve()
    if file.is_dir():file=file/'preview.json'
    if file.name!='preview.json' or not file.is_file() or file.stat().st_size>4*1024**2:
        raise ValueError('Choose the saved review\'s preview.json file, up to 4 MB')
    record=json.loads(file.read_text(encoding='utf-8-sig'))
    if not isinstance(record,dict) or record.get('schema',1)!=1:raise ValueError('Unsupported groom review format')
    if record.get('mode') not in {'PREPARING','PREVIEW','SAVING','SAVED','RESTORING'}:raise ValueError('This groom review has already finished')
    if (not isinstance(record.get('origin'),str) or Path(record['origin']).resolve()!=file.parent
            or any(not isinstance(record.get(key),str) or not record[key] or '\0' in record[key] for key in ('actor','component'))):
        raise ValueError('The saved groom review has an invalid origin or component identity')
    for key in ('before','after'):
        value=record.get(key)
        if (not isinstance(value,dict) or not all(isinstance(value.get(k),str) for k in ('groom','binding'))
                or any(not isinstance(value.get(k),list) or any(not isinstance(v,str) for v in value[k]) for k in ('materials','groups'))
                or type(value.get('use_cards')) is not bool):raise ValueError('The saved groom review has incomplete component states')
    if (not isinstance(record.get('source_fingerprints'),dict) or not record['source_fingerprints']
            or any(not isinstance(k,str) or not isinstance(v,str) or not re.fullmatch('[0-9a-fA-F]{64}',v) for k,v in record['source_fingerprints'].items())):
        raise ValueError('The saved groom review has invalid source guards')
    project=_review_project(record);linked=context.scene.cm_features.unreal_character_project
    if linked and Path(bpy.path.abspath(linked)).resolve()!=project:raise ValueError('This review belongs to a different Unreal project')
    name=context.scene.cm_settings.unreal_project.strip()
    if name and name.casefold()!=project.stem.casefold():raise ValueError('Select this review\'s Unreal project name before reconnecting')
    g=context.scene.cm_groom
    if g.replacement_origin:
        current=Path(bpy.path.abspath(g.replacement_origin)).resolve()
        if current!=file.parent and (current/'preview.json').is_file():
            previous=json.loads((current/'preview.json').read_text(encoding='utf-8-sig'))
            if previous.get('mode') in {'PREPARING','PREVIEW','SAVING','RESTORING'}:
                raise ValueError('Finish or close the current groom review before recovering another')
    g.replacement_origin=str(file.parent);g.replacement_component=record['component'];g.replacement_result=json.dumps(record)
    g.exchange_status='Saved groom review linked. Open its original Unreal level, then Keep or Restore. Save the Blender project to retain this link.'
    context.scene.cm_settings.status=g.exchange_status
    context.scene.cm_features.last_action_error=False
    return record


def running():
    return _JOB is not None


def command(request, folder):
    """Unreal commits success/failure independently of the remote connection."""
    folder = Path(folder)
    lines = [
        'import json, runpy, os, unreal', 'from pathlib import Path',
        'folder = Path(' + json.dumps(str(folder)) + ')',
        "receipt = folder / 'receipt.json'", "report = {}", 'delegated = False', 'try:',
        "    identity = runpy.run_path(str(folder / 'process_ref.py'))['identify'](os.getpid())",
        "    (folder / 'dispatch.json').write_text(json.dumps(identity), encoding='utf-8')",
        "    api = runpy.run_path(str(folder / 'unreal_groom.py'))",
    ]
    if request.get('expected_project'):
        lines.extend([
            '    expected = Path('+json.dumps(request['expected_project'])+').resolve()',
            '    actual = Path(unreal.Paths.convert_relative_path_to_full(unreal.Paths.get_project_file_path())).resolve()',
            "    if os.path.normcase(str(expected)) != os.path.normcase(str(actual)): raise ValueError('Open the linked Unreal project before continuing this groom exchange')",
        ])
    if request['action'] in {'PREVIEW_REPLACE','KEEP_REPLACE','RESTORE_REPLACE','CLOSE_REPLACE'}:
        lines.append("    revision = runpy.run_path(str(folder / 'unreal_groom_revision.py'))")
        if request['action']=='PREVIEW_REPLACE':
            lines.append("    report = revision['preview'](folder, %s, %s, %s, group_policy=%s)" % (json.dumps(request['groom']),json.dumps(request['binding']),json.dumps(request['component']),json.dumps(request.get('group_policy','AUTO'))))
        elif request['action']=='CLOSE_REPLACE':lines.append("    report = revision['close'](%s)" % json.dumps(request['origin']))
        else:lines.append("    report = revision['finish'](%s, restore=%s)" % (json.dumps(request['origin']),request['action']=='RESTORE_REPLACE'))
    elif request['action'] in {'FIT_LASHES','KEEP_LASH_FIT','REVERT_LASH_FIT'}:
        lines.append("    lash = runpy.run_path(str(folder / 'unreal_lash_fit.py'))")
        if request['action']=='FIT_LASHES':
            lines.append("    report = lash['begin'](folder, %s, %s)" % (json.dumps(request['python']),repr(request['lash_settings'])))
            lines.append("    delegated = report.get('receipt_delegated', False)")
        else:lines.append("    report = lash['finish'](%s, restore=%s)" % (json.dumps(request['origin']),request['action']=='REVERT_LASH_FIT'))
    elif request['action'] == 'PUSH':
        operation = "api['import_groom'](str(folder / 'groom.json'), %s, %s, revision=True, character_path=%s, lod_python=%s, simulation_policy=%s)" % (
            json.dumps(request['destination']), json.dumps(request['target']), json.dumps(request.get('character', '')),json.dumps(request.get('lod_python','')),json.dumps(request.get('simulation_policy','SOURCE')))
        lines.append('    report = ' + operation)
    elif request['action']=='CHARACTER_TARGET':
        lines.append("    report = api['character_target'](%s, %s)" % (json.dumps(request['character']), json.dumps(request['target'])))
    elif request['action']=='EQUIP_CREATOR':
        lines.append("    report = api['equip_wardrobe'](%s, %s, slot=%s)" % (json.dumps(request['character']), json.dumps(request['wardrobe']), json.dumps(request.get('wardrobe_slot', 'Hair'))))
    elif request['action']=='APPLY_ACTOR':
        lines.append("    report = api['apply_to_selected_actor'](%s, %s)" % (json.dumps(request['groom']),json.dumps(request['binding'])))
    elif request['action'] == 'SELECT_TARGET':
        lines.append("    report = api['selected_target']()")
    elif request['action'] in {'SELECT_HEAD', 'SELECT_GROOM'}:
        head = request['action'] == 'SELECT_HEAD'
        lines.append("    report = api['export_selected'](%s, str(folder / %s))" % (
            json.dumps('HEAD' if head else 'GROOM'), json.dumps('head.fbx' if head else 'groom.json')))
    else:
        function = 'export_head' if request['action'] == 'GRAB_HEAD' else 'export_groom'
        filename = 'head.fbx' if request['action'] == 'GRAB_HEAD' else 'groom.json'
        lines.extend([
            '    api[%s](%s, str(folder / %s))' % (json.dumps(function),
                json.dumps(request['target'] if function == 'export_head' else request['asset']), json.dumps(filename)),
            "    report = {'file': " + json.dumps(filename) + ", 'source_asset': " +
                json.dumps(request['target'] if function == 'export_head' else request['asset']) + "}",
        ])
    lines.extend([
        "    report['status'] = 'complete'", 'except Exception as exc:',
        "    partial = folder / 'groom.unreal-report.json'",
        "    if partial.is_file(): report = json.loads(partial.read_text(encoding='utf-8-sig'))",
        "    report.update(status='failed', error=str(exc))", 'finally:',
        "    if not delegated:",
        "        temporary = receipt.with_suffix('.tmp')",
        "        temporary.write_text(json.dumps(report), encoding='utf-8')",
        '        temporary.replace(receipt)',
    ])
    code = '\n'.join(lines)
    compile(code, '<MetaHuman Maker Groom Exchange>', 'exec')
    return code


def _send(job):
    request = job['request']
    try:
        payload = job.pop('payload', None)
        if payload is not None:
            with (job['folder'] / 'groom.json').open('w', encoding='utf-8') as stream:
                json.dump(payload, stream)
        bridge.run_python(request['engine'], command(request, job['folder']), request['project'],
                          on_dispatch=job['dispatched'].set)
    except Exception as exc:
        job['error'] = str(exc)


def start(context, action, payload=None):
    global _JOB
    from . import unreal_jobs, skin_native_jobs,groom_clearance
    if groom_clearance.pending():raise ValueError('Apply or discard the local clearance proposal before Unreal groom exchange')
    if running() or unreal_jobs.running() or skin_native_jobs.running():
        raise ValueError('Wait for the current Unreal exchange before starting another')
    if action not in {'PUSH', 'GRAB', 'GRAB_HEAD', 'SELECT_HEAD', 'SELECT_GROOM', 'SELECT_TARGET','APPLY_ACTOR','EQUIP_CREATOR','CHARACTER_TARGET','FIT_LASHES','KEEP_LASH_FIT','REVERT_LASH_FIT','PREVIEW_REPLACE','KEEP_REPLACE','RESTORE_REPLACE','CLOSE_REPLACE'}:
        raise ValueError('Unknown Unreal groom exchange')
    s, g = context.scene.cm_settings, context.scene.cm_groom
    revision={}
    if g.replacement_origin:
        path=Path(bpy.path.abspath(g.replacement_origin))/'preview.json'
        if path.is_file():revision=json.loads(path.read_text())
    if action in {'PUSH','PREVIEW_REPLACE','FIT_LASHES'} and revision.get('mode') in {'PREPARING','PREVIEW','SAVING','RESTORING'}:
        raise ValueError('Keep or restore the original groom preview before another replacement or push')
    if action in {'KEEP_REPLACE','RESTORE_REPLACE','CLOSE_REPLACE'} and revision.get('mode') not in {'PREPARING','PREVIEW','SAVING','SAVED','RESTORING'}:
        raise ValueError('Preview a groom replacement before Keep or Restore')
    if action=='FIT_LASHES':
        from . import resource_budget
        resource_budget.require('native lash fitting',1536*1024**2)
        if not Path(bpy.path.abspath(context.scene.cm_features.inference_python)).is_file():raise ValueError('Configure the local inference Python environment before fitting lashes')
    if action in {'KEEP_LASH_FIT','REVERT_LASH_FIT'}:
        try:lash_result=json.loads(g.lash_fit_result)
        except (ValueError,TypeError):raise ValueError('Fit and preview the character lashes first') from None
        if lash_result.get('mode') not in {'PREVIEW','SAVED'} or not (Path(lash_result.get('origin',''))/'preview.json').is_file():raise ValueError('No active native lash preview is available')
    if not g.last_push_result and g.exchange_result:
        try:previous_result=json.loads(g.exchange_result)
        except ValueError:previous_result={}
        if previous_result.get('status')=='complete' and previous_result.get('groom') and previous_result.get('binding'):
            g.last_push_result=json.dumps(previous_result)
    delivered={}
    if action in {'APPLY_ACTOR', 'EQUIP_CREATOR','PREVIEW_REPLACE'}:
        try:delivered=json.loads(g.last_push_result or g.exchange_result or '{}')
        except ValueError:raise ValueError('Push a groom successfully before applying it to an actor') from None
        if delivered.get('status')!='complete' or not delivered.get('groom') or not delivered.get('binding'):
            raise ValueError('Push a groom and binding successfully before applying them to an actor')
    if action == 'EQUIP_CREATOR':
        if not delivered.get('wardrobe'):
            raise ValueError('Push a groom with a wardrobe item before equipping it in Creator')
        if not context.scene.cm_features.unreal_character.strip():
            raise ValueError('Deliver this character to MetaHuman Creator before equipping hair')
    if action == 'CHARACTER_TARGET' and not context.scene.cm_features.unreal_character.strip():
        raise ValueError('Deliver this character to MetaHuman Creator before preparing its hair target')
    if action=='PUSH' and not g.unreal_target.strip() and not context.scene.cm_features.unreal_character.strip():
        raise ValueError('Deliver this character to Creator, or choose its Face Skeletal Mesh with Use Selected Face as Binding Target')
    if g.exchange_receipt:
        previous = Path(g.exchange_receipt)
        if (previous.parent / 'request.json').is_file() and not previous.is_file():
            resume(context)
            raise ValueError('Resumed the previous Unreal exchange. Wait for its receipt before another request')
    if action == 'PUSH' and not re.fullmatch(r'[A-Za-z_][A-Za-z0-9_]*', g.name):
        raise ValueError('Use letters, numbers and underscores for the Unreal groom name')
    engine = bpy.path.abspath(s.unreal_engine_dir)
    if not (Path(engine) / 'Engine/Plugins/Experimental/PythonScriptPlugin/Content/Python/remote_execution.py').is_file():
        raise ValueError('Choose the Unreal install root containing Engine/')
    review_project=_review_project(revision) if action in {'KEEP_REPLACE','RESTORE_REPLACE'} else None
    folder = Path(bpy.path.abspath(s.project_dir)) / 'Grooms' / '.exchange' / uuid.uuid4().hex
    folder.mkdir(parents=True, exist_ok=False)
    request = {'action': action, 'engine': engine, 'project': s.unreal_project,
               'asset': g.unreal_asset, 'target': g.unreal_target, 'name': g.name,
               'destination': '/Game/CreateMetaHuman/Grooms/' + g.name,
               'unit_scale': context.scene.unit_settings.scale_length,
               'character': context.scene.cm_features.unreal_character}
    linked=context.scene.cm_features.unreal_character_project
    if linked:request['expected_project']=bpy.path.abspath(linked)
    if action=='APPLY_ACTOR':request.update(groom=delivered['groom'],binding=delivered['binding'])
    if action=='PREVIEW_REPLACE':request.update(groom=delivered['groom'],binding=delivered['binding'],component=g.replacement_component.strip() or g.wardrobe_slot,previous_replacement_origin=g.replacement_origin,group_policy='TARGET_DEFAULTS' if g.replacement_group_defaults else 'AUTO')
    if action in {'KEEP_REPLACE','RESTORE_REPLACE','CLOSE_REPLACE'}:
        request['origin']=str(Path(bpy.path.abspath(g.replacement_origin)))
        if review_project:
            request['expected_project']=str(review_project);request['project']=review_project.stem
    if action=='EQUIP_CREATOR':request.update(character=context.scene.cm_features.unreal_character,wardrobe=delivered['wardrobe'],wardrobe_slot=g.wardrobe_slot)
    names={'unreal_groom','unreal_groom_representations','groom_lod_warp','process_ref'}
    if action=='PUSH' and g.fit_lod_meshes:
        request['lod_python']=bpy.path.abspath(context.scene.cm_features.inference_python)
    if action=='PUSH':request['simulation_policy']=g.simulation_policy
    if action in {'PREVIEW_REPLACE','KEEP_REPLACE','RESTORE_REPLACE','CLOSE_REPLACE'}:names.add('unreal_groom_revision')
    if action in {'FIT_LASHES','KEEP_LASH_FIT','REVERT_LASH_FIT'}:
        pending=['unreal_lash_fit','lash_fit_worker'];seen=set()
        while pending:
            name=pending.pop();path=Path(__file__).with_name(name+'.py')
            if name in seen or not path.is_file():continue
            seen.add(name);names.add(name)
            for node in ast.walk(ast.parse(path.read_text(encoding='utf-8'))):
                if isinstance(node,ast.Import):pending.extend(alias.name.split('.')[-1] for alias in node.names)
                elif isinstance(node,ast.ImportFrom) and node.module:pending.append(node.module.split('.')[-1])
        if action=='FIT_LASHES':request.update(python=bpy.path.abspath(context.scene.cm_features.inference_python),lash_settings=dict(component=g.lash_component,rebuild_guides=g.lash_rebuild_guides,maximum_offset_fraction=g.lash_maximum_offset,clearance_fraction=.001))
        else:request['origin']=lash_result['origin']
    for module in names:
        name=module+'.py'
        shutil.copy2(Path(__file__).with_name(name), folder / name)
    (folder / 'request.json').write_text(json.dumps(request), encoding='utf-8')
    job = {'scene': context.scene, 'request': request, 'folder': folder,
           'dispatched': threading.Event(), 'error': None, 'payload': payload}
    job['thread'] = threading.Thread(target=_send, args=(job,), name='CreateMHGroomExchange', daemon=True)
    _JOB = job
    g.exchange_receipt = str(folder / 'receipt.json')
    if action=='PREVIEW_REPLACE':g.replacement_origin=str(folder)
    g.exchange_result = ''
    g.exchange_status = 'Connecting to Unreal. You can keep working in Blender.'
    s.status = g.exchange_status
    job['thread'].start()
    if not bpy.app.timers.is_registered(poll):
        bpy.app.timers.register(poll, first_interval=.2, persistent=True)


def _scene(job):
    scene = job.get('scene')
    try:
        return scene if scene and scene.name in bpy.data.scenes else None
    except ReferenceError:
        return None


def apply_import(context):
    from . import grooming
    job = _JOB
    if not job or _scene(job) != context.scene:
        raise ValueError('No groom export is ready for this scene')
    if not (job['folder'] / 'receipt.json').is_file():
        raise ValueError('Wait for Unreal to finish exporting')
    report = json.loads((job['folder'] / 'receipt.json').read_text(encoding='utf-8'))
    if report.get('status') != 'complete' or context.mode != 'OBJECT':
        raise ValueError('Finish the current edit and confirm a successful Unreal export before importing')
    if context.scene.unit_settings.scale_length != job['request']['unit_scale']:
        raise ValueError('Restore the scene unit scale used for this exchange')
    if job['request']['action'] in {'GRAB_HEAD', 'SELECT_HEAD'}:
        scalp = grooming.import_head(context, job['folder'] / 'head.fbx')
        target = report.get('source_asset') or job['request']['target']
        scalp['cm_unreal_mesh'] = target
        context.scene.cm_groom.unreal_target = target
    elif job['request']['action'] in {'GRAB', 'SELECT_GROOM'}:
        context.scene.cm_groom.groom = grooming.import_curves(context,
            json.loads((job['folder'] / 'groom.json').read_text(encoding='utf-8-sig')), job['request']['name'])
        context.scene.cm_groom.unreal_asset = report.get('source_asset') or job['request']['asset']
    elif job['request']['action'] == 'SELECT_TARGET':
        context.scene.cm_groom.unreal_target = report['source_asset']
    else:
        raise ValueError('This exchange does not import Blender objects')
    context.scene.cm_groom.exchange_applied = str(job['folder'] / 'receipt.json')


def poll():
    global _JOB
    job = _JOB
    if not job:
        return None
    scene = _scene(job)
    folder = job['folder']
    path = folder / 'receipt.json'
    if scene and scene.cm_groom.exchange_receipt != str(path):
        # Undo can restore the previous request. Keep the remote receipt, but do
        # not publish an undone request into the artist's current scene.
        job['scene'] = None
        scene = None
    if not path.is_file():
        thread = job.get('thread')
        if thread and thread.is_alive():
            if scene and job['dispatched'].is_set():
                scene.cm_groom.exchange_status = 'Unreal is processing the groom. Blender remains usable.'
            return .2
        dispatch = folder / 'dispatch.json'
        if job['dispatched'].is_set() or dispatch.is_file():
            try:
                identity = json.loads(dispatch.read_text(encoding='utf-8')) if dispatch.is_file() else None
            except (OSError, ValueError):
                return .5
            if not identity or process_ref.alive(identity):
                if scene:
                    if job['request']['action']=='FIT_LASHES':
                        progress=folder/'progress.json'
                        try:phase=json.loads(progress.read_text()).get('phase','Preparing the selected character lashes')
                        except (OSError,ValueError):phase='Preparing the selected character lashes'
                        scene.cm_groom.exchange_status=phase+'. You can keep working in Blender.'
                    else:scene.cm_groom.exchange_status = 'Waiting for the original Unreal receipt. No duplicate request will be sent.'
                return .5
            report = {'status': 'failed', 'error': 'Unreal exited before confirming. Inspect partial assets before another push.'}
        else:
            report = {'status': 'failed', 'error': job.get('error') or 'No Unreal execution was confirmed. Inspect Unreal before retrying.'}
        path.write_text(json.dumps(report), encoding='utf-8')
    try:
        report = json.loads(path.read_text(encoding='utf-8'))
    except (OSError, ValueError):
        return .5
    action = job['request']['action']
    if scene and report.get('status') == 'complete' and action in {'GRAB', 'GRAB_HEAD', 'SELECT_HEAD', 'SELECT_GROOM', 'SELECT_TARGET'} and scene.cm_groom.exchange_applied != str(path):
        if bpy.context.scene != scene or bpy.context.mode != 'OBJECT':
            scene.cm_groom.exchange_status = 'Unreal export ready. Return to this project and finish the active edit to import.'
            return .5
        if scene.unit_settings.scale_length != job['request']['unit_scale']:
            scene.cm_groom.exchange_status = 'Restore the scene unit scale used when requesting this groom to import it.'
            return .5
        try:
            if bpy.ops.cm.groom(action='APPLY_EXCHANGE') != {'FINISHED'}:
                raise ValueError(scene.cm_settings.status)
            report['blender_imported'] = True
        except Exception as exc:
            report.update(status='failed', error='Unreal export retained; Blender import failed: ' + str(exc))
    if scene:
        g = scene.cm_groom
        g.exchange_result = json.dumps(report)
        scene.cm_features.last_action_error = report.get('status') != 'complete'
        if report.get('status') != 'complete':
            g.exchange_status = 'Exchange failed: ' + report.get('error', 'Inspect Unreal Output Log')
            if action=='PREVIEW_REPLACE':
                state=folder/'preview.json'
                active=json.loads(state.read_text()).get('mode') if state.is_file() else None
                if active not in {'PREPARING','PREVIEW','SAVING','RESTORING'}:g.replacement_origin=job['request'].get('previous_replacement_origin','')
        elif job['request']['action'] == 'PUSH':
            g.last_push_result=json.dumps(report)
            if report.get('target'):
                g.unreal_target=report['target']
            g.exchange_status = 'Groom, binding and wardrobe saved. Use Preview Replacement on Character, or choose its Creator Section and Equip Groom in Creator.'
            if report.get('binding_target_refreshed'):
                g.exchange_status+=" Binding refreshed for the linked character's current head."
            representations=report.get('representations',[])
            cards=sum(row.get('cards',0) for row in representations if row.get('reused'))
            meshes=sum(row.get('meshes',0) for row in representations if row.get('reused'))
            if cards or meshes:g.exchange_status+=f' Preserved {cards} card and {meshes} mesh representations.'
            if any(row.get('deformed') for row in representations):g.exchange_status+=' Edited-strand LOD fits are ready for review in replacement preview.'
            if report.get('strand_lod_fallbacks'):g.exchange_status+=' Strand LOD fallbacks are listed in the exchange receipt.'
        elif action=='CHARACTER_TARGET':
            g.unreal_target=report['source_asset']
            g.exchange_status='Binding target prepared from this Creator character.' + (' Reused the unchanged rest head.' if report.get('reused_source') else '')
        elif action=='EQUIP_CREATOR':
            g.exchange_status='Groom equipped and saved in Creator: '+report.get('slot','Hair')+'.' + (' Open the character in Unreal to preview it.' if not report.get('editor_opened') else '')
        elif action=='APPLY_ACTOR':
            g.exchange_status='Groom applied to '+report.get('actor_label','the selected actor')+'. Save the Unreal level to keep it. This actor edit can be undone in Unreal.'
        elif action in {'PREVIEW_REPLACE','KEEP_REPLACE','RESTORE_REPLACE','CLOSE_REPLACE'}:
            g.replacement_result=json.dumps(report)
            if report.get('origin'):g.replacement_origin=report['origin']
            g.exchange_status={'PREVIEW':'Replacement is previewing on the existing component. Inspect it, then Save Groom in Blueprint or Restore Previous Groom.','SAVED':'Reviewed groom saved in the existing character blueprint. Creator wardrobe selection is separate.','RESTORED':'Previous groom restored; later artist changes are retained.','CLOSED':'Review closed without changing actor or blueprint data. Save the level to retain instance edits.'}.get(report.get('mode'),'Groom replacement review updated.')
        elif action in {'FIT_LASHES','KEEP_LASH_FIT','REVERT_LASH_FIT'}:
            g.lash_fit_result=json.dumps(report)
            if action=='FIT_LASHES':g.lash_fit_receipt=str(path)
            g.exchange_status={'PREVIEW':'Fitted lashes are previewing in Unreal. Review expressions, then Keep in Blueprint or Restore Previous Lashes.','SAVED':'Fitted lashes saved in the character blueprint.','RESTORED':'Previous lashes restored; other artist settings retained.'}.get(report.get('mode'),'Native lash operation complete.')
        elif action == 'SELECT_TARGET':
            g.exchange_status = 'Binding target set to ' + g.unreal_target.rsplit('/',1)[-1].split('.')[0] + '.'
        else:
            if action in {'GRAB_HEAD', 'SELECT_HEAD'}:
                g.exchange_status = 'Head imported. Its Unreal mesh is now the binding target.'
            else:
                g.exchange_status = 'Groom imported from Unreal.'
        scene.cm_settings.status = g.exchange_status
    (folder / 'result.json').write_text(json.dumps(report), encoding='utf-8')
    if report.get('status') == 'complete' and job['request']['action'] == 'PUSH':
        # The native assets and durable receipt now own this result. Retain
        # failed payloads and inbound files needed by saved pending imports.
        payload = folder / 'groom.json'
        if payload.is_file():
            payload.unlink()
    _JOB = None
    for window in bpy.context.window_manager.windows:
        for area in window.screen.areas:
            area.tag_redraw()
    return None


@persistent
def detach(*_):
    if _JOB:
        _JOB['scene'] = None


def resume(context):
    global _JOB
    receipt = Path(context.scene.cm_groom.exchange_receipt)
    request_path = receipt.parent / 'request.json'
    if not request_path.is_file():
        raise ValueError('No saved groom exchange to resume')
    if _JOB:
        if _JOB['folder'] != receipt.parent:
            raise ValueError('Another groom exchange is still being monitored')
        _JOB['scene'] = context.scene
    else:
        dispatched = threading.Event()
        # A saved request can be queued in Unreal even without a dispatch file.
        # Resume observes it; it never resubmits a possibly executing operation.
        dispatched.set()
        _JOB = {'scene': context.scene, 'folder': receipt.parent,
                'request': json.loads(request_path.read_text(encoding='utf-8')),
                'thread': None, 'dispatched': dispatched, 'error': None}
    if not bpy.app.timers.is_registered(poll):
        bpy.app.timers.register(poll, first_interval=.2, persistent=True)


@persistent
def loaded(*_):
    if hasattr(bpy.context.scene, 'cm_groom') and bpy.context.scene.cm_groom.exchange_receipt:
        try:
            g=bpy.context.scene.cm_groom
            processed=Path(bpy.path.abspath(g.exchange_receipt)).parent/'result.json'
            saved=json.loads(g.exchange_result or '{}')
            if (saved.get('status') in {'complete','failed'} and processed.is_file()
                    and json.loads(processed.read_text()).get('status') in {'complete','failed'}):
                # This scene already consumed the exchange. Replaying it can
                # block immediate agent actions and reset newer delivery links.
                return
            resume(bpy.context)
        except (OSError, ValueError) as exc:
            bpy.context.scene.cm_groom.exchange_status = 'Resume the previous groom exchange: ' + str(exc)


def register():
    if detach not in bpy.app.handlers.load_pre:
        bpy.app.handlers.load_pre.append(detach)
    if loaded not in bpy.app.handlers.load_post:
        bpy.app.handlers.load_post.append(loaded)


def unregister():
    if detach in bpy.app.handlers.load_pre:
        bpy.app.handlers.load_pre.remove(detach)
    if loaded in bpy.app.handlers.load_post:
        bpy.app.handlers.load_post.remove(loaded)
    if bpy.app.timers.is_registered(poll):
        bpy.app.timers.unregister(poll)
    detach()
