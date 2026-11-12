"""Offline whole-action memory/ownership checks; no UE, assets or cloud required."""
import json
from pathlib import Path
import runpy
import sys
from types import SimpleNamespace as NS, ModuleType
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
RUNTIME=runpy.run_path(str(ROOT/'create_metahuman/unreal_assembly.py'))
GIB=RUNTIME['GIB']


def action(editing,commits,physical=1,override=False,disks=None,rebuild=None,private=True,wardrobe_issues=None,missing_groom=False):
    calls={'initialized':0,'released':0,'native_api':0,'configured':0,'groom_verifications':0}
    class Character:
        def get_editor_property(self,key):return NS(get_editor_property=lambda _: 'AnimBP')
        def get_path_name(self):return '/Game/Private/Character'
    character=Character()
    blueprint_path='/Game/AssemblyMemoryFixture/Example/BP_Example.BP_Example'
    metadata={'MHExportQuality':'Cinematic','MetaHumanMaker.SourceCharacter':character.get_path_name()}
    class Blueprint:
        def get_path_name(self):return blueprint_path
        def get_editor_property(self,key):return 'ready'
        def generated_class(self):return object()
    blueprint=Blueprint()
    def build_result():
        if rebuild=='stamp':metadata['MHExportQuality']='Cinematic'
        if rebuild=='exception':raise RuntimeError('Native build interrupted')
    class Session:
        def is_object_added_for_editing(self,c):return editing
        def try_add_object_to_edit(self,c):calls['initialized']+=1;return True
        def remove_object_to_edit(self,c):calls['released']+=1
        def can_build_meta_human(self,c,flag):return True
        def build_meta_human(self,*args):
            calls['native_api']+=1;build_result()
    session=Session();unreal=ModuleType('unreal')
    unreal.Paths=NS(get_project_file_path=lambda:str(ROOT/'tests/ue_project/CreateMH_Test.uproject'),project_content_dir=lambda:str(ROOT))
    unreal.load_asset=lambda path:blueprint if path in (blueprint_path,blueprint_path.split('.')[0]) else character
    unreal.MetaHumanCharacter=Character;unreal.Blueprint=Blueprint
    unreal.MetaHumanCharacterEditorSubsystem=object
    unreal.get_editor_subsystem=lambda _:session
    unreal.EditorAssetLibrary=NS(does_directory_exist=lambda _:False,
        get_metadata_tag=lambda obj,key:metadata.get(key,''),
        remove_metadata_tag=lambda obj,key:metadata.pop(key,None),
        set_metadata_tag=lambda obj,key,value:metadata.__setitem__(key,value),
        save_directory=lambda *a,**k:True,sync_browser_to_objects=lambda *a:None)
    unreal.BlueprintEditorLibrary=NS(compile_blueprint=lambda _:None)
    unreal.BlueprintStatus=NS(BS_UP_TO_DATE='ready',BS_UP_TO_DATE_WITH_WARNINGS='warning')
    unreal.AssetRegistryHelpers=NS(get_asset_registry=lambda:NS(get_assets_by_path=lambda *a,**k:[
        NS(asset_class_path=NS(asset_name='SkeletalMesh'),package_name='/Game/AssemblyMemoryFixture/Example/Face',asset_name='Face')]))
    unreal.MetaHumanCharacterEditorBuildParameters=NS
    unreal.MetaHumanDefaultPipelineType=NS(CINEMATIC=NS(value=0),OPTIMIZED=NS(value=1))
    unreal.MetaHumanQualityLevel=NS(CINEMATIC=NS(value=0))
    unreal.SystemLibrary=NS(get_command_line=lambda:'')
    def native(*args):
        calls['native_api']+=1
        if rebuild is None:return False
        build_result();return True
    unreal.CreateMHCharacterLibrary=NS(build_with_pipeline=native)
    count=0
    def budgets(_):
        nonlocal count
        available=commits[min(count,len(commits)-1)]
        disk=disks[min(count,len(disks)-1)] if disks else 40;count+=1
        return dict(free_disk_bytes=disk*GIB,available_memory_bytes=physical*GIB,available_commit_bytes=available*GIB,
            minimum_commit_bytes=2*GIB,recommended_commit_bytes=10*GIB,initialization_reserve_bytes=8*GIB,disk_reserve_bytes=10*GIB)
    def configure(*args):calls['configured']+=1;return object() if private else None,{'material_baking_required':False}
    def helper(path,*args,**kwargs):
        name=Path(path).name
        if name=='unreal_character.py':return {'inspect_character':lambda *a,**k:dict(status='complete',native_initializations=99,assembly={'ready':True})}
        if name=='unreal_assembly_textures.py':return {'configure':configure}
        if name in {'unreal_assembly_eyes.py','unreal_assembly_auxiliary.py'}:return {'preserve':lambda *a:{}}
        raise AssertionError('Unexpected external helper: '+name)
    function=RUNTIME['assemble'];globals_=function.__globals__
    options=dict(root='/Game/AssemblyMemoryFixture',name='Example',profile='CINEMATIC',allow_low_memory=override)
    if rebuild is not None:options['existing_blueprint']=blueprint_path
    def verify(*args,native_stock_retarget=False):
        assert native_stock_retarget is True,'Stock retargeting must follow a verified native actor update'
        calls['groom_verifications']+=1
        if missing_groom:raise RuntimeError('Selected groom missing from native output')
        return {'selected_grooms':0,'verified':[]}
    with patch.dict(sys.modules,{'unreal':unreal}),patch.dict(globals_,{
            'budgets':budgets,'actor_meshes':lambda _: {},
            'selected_grooms':lambda *args:dict(grooms=[],issues=wardrobe_issues or []),
            'verify_assembled_grooms':verify}),patch('runpy.run_path',side_effect=helper):
        report=function(character.get_path_name(),options)
    if rebuild is not None:
        assert metadata['MHExportQuality']=='Cinematic',(rebuild,metadata,report)
    return report,calls


def run():
    cases=[]
    def check(name,editing,commits,expected,physical=1,override=False,disks=None):
        report,calls=action(editing,commits,physical,override,disks)
        for key,value in expected.items():assert calls[key]==value,(name,calls,report)
        assert report['native_initializations']==calls['initialized'],(name,report)
        assert calls['released']==calls['initialized'],(name,calls)
        assert report['build']['native_build_api_called']==bool(calls['native_api'])
        cases.append(dict(case=name,calls=calls,build_state=report['build']['state'],error=report.get('error')))
    check('warm_low_physical_sufficient_commit',True,[12,12,12],{'initialized':0,'native_api':1})
    check('warm_high_physical_low_commit',True,[9],{'initialized':0,'native_api':0},physical=24)
    check('cold_initialization_reserve_required',False,[17],{'initialized':0,'native_api':0})
    check('cold_post_load_pressure_releases_owned_session',False,[20,20,9],{'initialized':1,'native_api':0})
    check('cold_ready_session_initialized_once',False,[20,20,12,11],{'initialized':1,'native_api':1})
    check('warm_pressure_during_pipeline_preparation',True,[12,12,12,9],{'initialized':0,'configured':1,'native_api':0})
    check('disk_pressure_during_pipeline_preparation',True,[12,12,12,12],{'initialized':0,'configured':1,'native_api':0},disks=[40,40,40,9])
    check('explicit_override_keeps_hard_floor',True,[1],{'native_api':0},override=True)
    check('explicit_override_does_not_require_physical_10gib',True,[3,3,3],{'native_api':1},override=True)
    try:RUNTIME['options'](dict(root='/Game/Test',name='Example',allow_low_memory='false'))
    except ValueError:cases.append({'case':'non_boolean_override_refused'})
    else:raise AssertionError('A string must not enable a memory override')
    for private in (False,True):
        for result in ('no_op','exception','stamp'):
            report,calls=action(True,[12],rebuild=result,private=private)
            assert calls['native_api']==1,(result,report)
            assert report['status']==('complete' if result=='stamp' else 'failed'),(result,report)
            assert calls['groom_verifications']==int(result=='stamp'),(result,calls,report)
            assert bool(report['build'].get('native_actor_update_verified'))==(result=='stamp')
            cases.append(dict(case='rebuild_'+result,private_pipeline=private,status=report['status']))
    report,calls=action(True,[12],rebuild='stamp',wardrobe_issues=['Hair: rest mesh is in assembly output'])
    assert report['status']=='failed' and report['build']['state']=='NOT_STARTED',report
    assert calls['native_api']==0 and calls['configured']==0,calls
    cases.append(dict(case='unsafe_wardrobe_stops_before_pipeline_and_dispatch',status=report['status']))
    report,calls=action(True,[12],rebuild='stamp',missing_groom=True)
    assert calls['native_api']==1 and report['status']=='failed' and report['build']['state']=='FAILED',report
    cases.append(dict(case='native_success_marker_cannot_hide_missing_groom',status=report['status']))
    return dict(status='complete',scope='Offline assembly preflight, ownership and rebuild completion with a fake native API',cases=cases,actual_native_builds=0)


if __name__=='__main__':print(json.dumps(run(),indent=2))
