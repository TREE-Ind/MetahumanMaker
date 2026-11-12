#include "CreateMHExpressionLibrary.h"
#include "DNAAsset.h"
#include "DNAAssetUserData.h"
#include "DNA.h"
#include "DNAReader.h"
#include "DNAUtils.h"
#include "SharedRigRuntimeContext.h"
#include "RigInstance.h"
#include "DNAIndexMapping.h"
#include "Engine/SkeletalMesh.h"
#include "Engine/World.h"
#include "Components/SkeletalMeshComponent.h"
#include "Components/LODSyncComponent.h"
#include "GameFramework/Actor.h"
#include "LODSyncInterface.h"
#include "Animation/AnimSequence.h"
#include "Animation/AnimInstance.h"
#include "Animation/AnimSingleNodeInstance.h"
#include "Animation/AnimData/IAnimationDataController.h"
#include "Animation/AnimData/IAnimationDataModel.h"
#include "Dom/JsonObject.h"
#include "Serialization/JsonSerializer.h"
#include "Serialization/JsonReader.h"
#include "Serialization/JsonWriter.h"
#include "Misc/FileHelper.h"
#include "Misc/ScopeExit.h"
#include "Editor.h"
#include "AssetCompilingManager.h"
#include "Rendering/SkeletalMeshRenderData.h"
#include "Rendering/SkeletalMeshLODRenderData.h"
#include "Rendering/SkeletalMeshModel.h"
#include "Rendering/SkeletalMeshLODModel.h"
#include "SkelMeshDNAUtils.h"
#include "Rendering/SkinWeightVertexBuffer.h"
#include "Animation/MorphTarget.h"
#include "Materials/MaterialInstanceDynamic.h"
#include "LevelEditorViewport.h"
#include "Misc/Guid.h"

namespace
{
    struct FCreateMHViewportRefresh
    {
        FLevelEditorViewportClient* Client;
        FText Label;
    };
    TMap<FString,FCreateMHViewportRefresh> CreateMHViewportRefreshes;
    FString ViewportRefreshJSON(const TSharedRef<FJsonObject>& Object)
    {
        FString Text;
        FJsonSerializer::Serialize(Object,TJsonWriterFactory<>::Create(&Text));
        return Text;
    }
}

FString UCreateMHExpressionLibrary::BeginEditorViewportRefresh()
{
    if (!IsInGameThread() || !GEditor || !GCurrentLevelEditingViewportClient) return FString();
    auto* Client=GCurrentLevelEditingViewportClient;
    if (!GEditor->GetLevelViewportClients().Contains(Client) || Client->IsSimulateInEditorViewport()) return FString();
    const FString Token=FGuid::NewGuid().ToString(EGuidFormats::Digits);
    const FText Label=FText::FromString(TEXT("MetaHuman Maker Capture ")+Token);
    auto Result=MakeShared<FJsonObject>();
    Result->SetStringField(TEXT("token"),Token);
    Result->SetBoolField(TEXT("realtime_before"),Client->IsRealtime());
    Client->AddRealtimeOverride(true,Label);
    Client->Invalidate();
    CreateMHViewportRefreshes.Add(Token,{Client,Label});
    Result->SetBoolField(TEXT("realtime_active"),Client->IsRealtime());
    return ViewportRefreshJSON(Result);
}

FString UCreateMHExpressionLibrary::EndEditorViewportRefresh(const FString& Token)
{
    if (!IsInGameThread()) return FString();
    FCreateMHViewportRefresh Refresh;
    if (!CreateMHViewportRefreshes.RemoveAndCopyValue(Token,Refresh)) return FString();
    auto Result=MakeShared<FJsonObject>();
    const bool Open=GEditor && GEditor->GetLevelViewportClients().Contains(Refresh.Client);
    Result->SetBoolField(TEXT("viewport_open"),Open);
    if (Open)
    {
        // Remove only this capture's override, retaining other systems and
        // any realtime preference the artist changed while the capture ran.
        const bool Removed=Refresh.Client->RemoveRealtimeOverride(Refresh.Label,false);
        Result->SetBoolField(TEXT("override_removed"),Removed);
        Result->SetBoolField(TEXT("realtime_after"),Refresh.Client->IsRealtime());
        Refresh.Client->Invalidate();
    }
    Result->SetStringField(TEXT("status"),TEXT("complete"));
    return ViewportRefreshJSON(Result);
}

bool UCreateMHExpressionLibrary::SupportsIsolatedPoseEvaluation() { return true; }
bool UCreateMHExpressionLibrary::SupportsMaterialCurveSampling() { return true; }

bool UCreateMHExpressionLibrary::PrepareControlClip(UAnimSequence* Animation,USkeletalMesh* FaceMesh)
{
    if (!IsInGameThread()||!Animation||!FaceMesh||!FaceMesh->GetSkeleton()||Animation->GetOutermost()==FaceMesh->GetOutermost()||!Animation->GetPathName().StartsWith(TEXT("/Game/"))) return false;
    auto Model=Animation->GetDataModel();if (!Model||Model->GetNumBoneTracks()||!Model->GetFloatCurves().Num()) return false;
    for (const auto& Curve:Model->GetFloatCurves()) if (!Curve.GetName().ToString().StartsWith(TEXT("CTRL_expressions_"),ESearchCase::IgnoreCase)) return false;
    Animation->Modify();Animation->SetSkeleton(FaceMesh->GetSkeleton());Animation->SetPreviewSkeletalMesh(FaceMesh);Animation->MarkPackageDirty();
    return Animation->GetSkeleton()==FaceMesh->GetSkeleton();
}

TArray<FString> UCreateMHExpressionLibrary::GetFaceRawControls(USkeletalMesh* Mesh)
{
    TArray<FString> Result;
    if (!IsInGameThread() || !Mesh) return Result;
    TSharedPtr<IDNAReader> Reader;
    if (auto Data=Cast<UDNAAssetUserData>(Mesh->GetAssetUserDataOfClass(UDNAAssetUserData::StaticClass())))
    {if (Data->DNAAsset) Reader=Data->DNAAsset->GetDNAReader();}
    else if (auto Legacy=Cast<UDNAAsset>(Mesh->GetAssetUserDataOfClass(UDNAAsset::StaticClass()))) Reader=Legacy->GetDNAReader();
    if (Reader) for (uint16 I=0;I<Reader->GetRawControlCount();++I) Result.Add(Reader->GetRawControlName(I));
    return Result;
}

UAnimSequence* UCreateMHExpressionLibrary::CreateControlPreviewClip(USkeletalMesh* SourceMesh,const FString& ControlsJSON)
{
    if (!IsInGameThread() || !SourceMesh || !SourceMesh->GetSkeleton()) return nullptr;
    TSharedPtr<FJsonObject> Controls;
    if (!FJsonSerializer::Deserialize(TJsonReaderFactory<>::Create(ControlsJSON),Controls) || !Controls) return nullptr;
    TSharedPtr<IDNAReader> Reader;
    if (UDNAAssetUserData* Data=Cast<UDNAAssetUserData>(SourceMesh->GetAssetUserDataOfClass(UDNAAssetUserData::StaticClass())))
    {
        if (Data->DNAAsset) Reader=Data->DNAAsset->GetDNAReader();
    }
    else if (UDNAAsset* LegacyData=Cast<UDNAAsset>(SourceMesh->GetAssetUserDataOfClass(UDNAAsset::StaticClass()))) Reader=LegacyData->GetDNAReader();
    if (!Reader || !Reader->GetRawControlCount()) return nullptr;
    TSet<FName> Supported;
    for (uint16 I=0;I<Reader->GetRawControlCount();++I) Supported.Add(FName(*Reader->GetRawControlName(I).Replace(TEXT("."),TEXT("_"))));
    TArray<TPair<FName,float>> Values;
    for (const auto& Item:Controls->Values)
    {
        double Value=0;
        FString Name=FString(Item.Key).Replace(TEXT("."),TEXT("_"));
        if (!Name.StartsWith(TEXT("CTRL_expressions_"))) Name=TEXT("CTRL_expressions_")+Name;
        const FName CurveName(*Name);
        if (!Supported.Contains(CurveName) || !Item.Value->TryGetNumber(Value) || !FMath::IsFinite(Value) || Value<0.0 || Value>1.0) return nullptr;
        if (Values.ContainsByPredicate([CurveName](const auto& Pair){return Pair.Key==CurveName;})) return nullptr;
        Values.Emplace(CurveName,float(Value));
    }
    UAnimSequence* Animation=NewObject<UAnimSequence>(GetTransientPackage(),NAME_None,RF_Transient);
    Animation->SetSkeleton(SourceMesh->GetSkeleton());
    IAnimationDataController& Controller=Animation->GetController();
    Controller.InitializeModel();
    Controller.OpenBracket(FText::FromString(TEXT("Native expression sample")),false);
    Controller.SetFrameRate(FFrameRate(30,1),false);
    Controller.SetNumberOfFrames(FFrameNumber(1),false);
    bool bValid=true;
    for (const auto& Pair:Values)
    {
        const FAnimationCurveIdentifier Curve(Pair.Key,ERawCurveTrackTypes::RCT_Float);
        bValid=Controller.AddCurve(Curve,AACF_DefaultCurve,false) && Controller.SetCurveKeys(Curve,{FRichCurveKey(0.0f,Pair.Value),FRichCurveKey(1.0f/30.0f,Pair.Value)},false) && bValid;
    }
    Controller.NotifyPopulated();
    Controller.CloseBracket(false);
    if (!bValid) {Animation->MarkAsGarbage();return nullptr;}
    return Animation;
}

FString UCreateMHExpressionLibrary::CaptureComponentPreviewState(USkeletalMeshComponent* Component,bool bAllowTransientAnimation)
{
    if (!IsInGameThread() || !Component || !Component->GetSkeletalMeshAsset()) return TEXT("");
    const auto Mode=Component->GetAnimationMode();
    if (Mode!=EAnimationMode::AnimationBlueprint && Mode!=EAnimationMode::AnimationSingleNode) return TEXT("");
    auto State=MakeShared<FJsonObject>();
    State->SetStringField(TEXT("component"),Component->GetPathName());
    State->SetStringField(TEXT("mesh"),Component->GetSkeletalMeshAsset()->GetPathName());
    State->SetStringField(TEXT("animation_class"),Component->GetAnimClass()?Component->GetAnimClass()->GetPathName():TEXT(""));
    State->SetNumberField(TEXT("mode"),int32(Mode));
    State->SetBoolField(TEXT("editor_tick"),Component->GetUpdateAnimationInEditor());
    State->SetBoolField(TEXT("paused"),Component->bPauseAnims);
    State->SetNumberField(TEXT("forced_lod"),Component->GetForcedLOD());
    if (Mode==EAnimationMode::AnimationSingleNode)
    {
        auto Instance=Component->GetSingleNodeInstance();
        if (!Instance) return TEXT("");
        auto Asset=Instance->GetCurrentAsset();
        // A persisted path can be reloaded after GC; a transient artist clip cannot.
        if (Asset && Asset->HasAnyFlags(RF_Transient) && !bAllowTransientAnimation) return TEXT("");
        State->SetStringField(TEXT("animation"),Asset?Asset->GetPathName():TEXT(""));
        State->SetNumberField(TEXT("position"),Instance->GetCurrentTime());
        State->SetNumberField(TEXT("rate"),Instance->GetPlayRate());
        State->SetBoolField(TEXT("playing"),Instance->IsPlaying());
        State->SetBoolField(TEXT("looping"),Instance->IsLooping());
    }
    FString JSON;FJsonSerializer::Serialize(State,TJsonWriterFactory<>::Create(&JSON));return JSON;
}

bool UCreateMHExpressionLibrary::RestoreComponentPreviewState(USkeletalMeshComponent* Component,const FString& StateJSON)
{
    if (!IsInGameThread() || !Component || !Component->GetSkeletalMeshAsset()) return false;
    TSharedPtr<FJsonObject> State;
    if (!FJsonSerializer::Deserialize(TJsonReaderFactory<>::Create(StateJSON),State) || !State) return false;
    FString Path,Mesh,Class,Animation;double Mode=-1,LOD=-1;bool Tick=false,Paused=false;
    if (!State->TryGetStringField(TEXT("component"),Path) || Path!=Component->GetPathName() ||
        !State->TryGetStringField(TEXT("mesh"),Mesh) || Mesh!=Component->GetSkeletalMeshAsset()->GetPathName() ||
        !State->TryGetStringField(TEXT("animation_class"),Class) || !State->TryGetNumberField(TEXT("mode"),Mode) ||
        !State->TryGetNumberField(TEXT("forced_lod"),LOD) || !State->TryGetBoolField(TEXT("editor_tick"),Tick) ||
        !State->TryGetBoolField(TEXT("paused"),Paused) || !FMath::IsFinite(Mode) || !FMath::IsFinite(LOD) ||
        LOD<0 || LOD>Component->GetSkeletalMeshAsset()->GetLODNum() || LOD!=int32(LOD) ||
        (Mode!=int32(EAnimationMode::AnimationBlueprint) && Mode!=int32(EAnimationMode::AnimationSingleNode))) return false;
    UClass* AnimClass=Class.IsEmpty()?nullptr:LoadObject<UClass>(nullptr,*Class);
    if (!Class.IsEmpty() && (!AnimClass || !AnimClass->IsChildOf(UAnimInstance::StaticClass()))) return false;
    double Position=0,Rate=1;bool Playing=false,Looping=false;UAnimationAsset* Asset=nullptr;
    if (Mode==int32(EAnimationMode::AnimationSingleNode))
    {
        if (!State->TryGetStringField(TEXT("animation"),Animation) || !State->TryGetNumberField(TEXT("position"),Position) ||
            !State->TryGetNumberField(TEXT("rate"),Rate) || !State->TryGetBoolField(TEXT("playing"),Playing) ||
            !State->TryGetBoolField(TEXT("looping"),Looping) || !FMath::IsFinite(Position) || Position<0 || !FMath::IsFinite(Rate)) return false;
        Asset=Animation.IsEmpty()?nullptr:LoadObject<UAnimationAsset>(nullptr,*Animation);
        if (!Animation.IsEmpty() && (!Asset || Asset->GetSkeleton()!=Component->GetSkeletalMeshAsset()->GetSkeleton())) return false;
    }
    Component->SetAnimInstanceClass(AnimClass);
    Component->SetAnimationMode(EAnimationMode::Type(int32(Mode)));
    if (Mode==int32(EAnimationMode::AnimationSingleNode))
    {
        Component->SetAnimation(Asset);Component->SetPlayRate(float(Rate));Component->SetPosition(float(Position),false);
        if (auto Instance=Component->GetSingleNodeInstance()) {Instance->SetLooping(Looping);Instance->SetPlaying(Playing);}
    }
    Component->SetForcedLOD(int32(LOD));Component->bPauseAnims=Paused;Component->SetUpdateAnimationInEditor(Tick);
    return true;
}

FString UCreateMHExpressionLibrary::CaptureEditorLODState(AActor* Actor)
{
    if (!IsInGameThread() || !Actor || !Actor->GetWorld() || Actor->GetWorld()->IsGameWorld()) return TEXT("");
    TArray<UActorComponent*> Components;Actor->GetComponents(Components);
    auto Root=MakeShared<FJsonObject>();
    for (UActorComponent* Component:Components) if (auto LOD=Cast<ILODSyncInterface>(Component))
    {
        auto Row=MakeShared<FJsonObject>();
        Row->SetNumberField(TEXT("rendered"),LOD->GetForceRenderedLOD());
        Row->SetNumberField(TEXT("streamed"),LOD->GetForceStreamedLOD());
        Root->SetObjectField(Component->GetPathName(),Row);
    }
    FString JSON;FJsonSerializer::Serialize(Root,TJsonWriterFactory<>::Create(&JSON));return JSON;
}

bool UCreateMHExpressionLibrary::RestoreEditorLODState(AActor* Actor,const FString& StateJSON)
{
    if (!IsInGameThread() || !Actor || !Actor->GetWorld() || Actor->GetWorld()->IsGameWorld()) return false;
    TSharedPtr<FJsonObject> Root;if (!FJsonSerializer::Deserialize(TJsonReaderFactory<>::Create(StateJSON),Root) || !Root) return false;
    TArray<UActorComponent*> Components;Actor->GetComponents(Components);
    struct FRestore {ILODSyncInterface* LOD;int32 Rendered;int32 Streamed;};TArray<FRestore> Rows;
    for (UActorComponent* Component:Components) if (auto LOD=Cast<ILODSyncInterface>(Component))
    {
        const TSharedPtr<FJsonObject>* Row=nullptr;double Rendered=0,Streamed=0;
        if (!Root->TryGetObjectField(Component->GetPathName(),Row) || !Row || !(*Row)->TryGetNumberField(TEXT("rendered"),Rendered)
                || !(*Row)->TryGetNumberField(TEXT("streamed"),Streamed) || !FMath::IsFinite(Rendered) || !FMath::IsFinite(Streamed)
                || Rendered<-1 || Streamed<-1 || Rendered>128 || Streamed>128 || Rendered!=int32(Rendered) || Streamed!=int32(Streamed)) return false;
        Rows.Add({LOD,int32(Rendered),int32(Streamed)});
    }
    if (Rows.Num()!=Root->Values.Num()) return false;
    for (const FRestore& Row:Rows) {Row.LOD->SetForceRenderedLOD(Row.Rendered);Row.LOD->SetForceStreamedLOD(Row.Streamed);}
    return true;
}

bool UCreateMHExpressionLibrary::RefreshEditorLODSync(ULODSyncComponent* Component,int32 ForcedLOD,bool bUpdate)
{
    if (!IsInGameThread() || !Component || !Component->IsRegistered() || !Component->GetWorld() || Component->GetWorld()->IsGameWorld() || ForcedLOD<-2 || ForcedLOD>128) return false;
    // Avoid an editor property change that can reconstruct the Blueprint and
    // invalidate the live components during a temporary pose review.
    if (ForcedLOD>=-1) Component->ForcedLOD=ForcedLOD;
    if (bUpdate) Component->UpdateLOD();return true;
}

bool UCreateMHExpressionLibrary::RefreshEditorPreviewPose(USkeletalMeshComponent* Component)
{
    if (!IsInGameThread() || !Component || !Component->IsRegistered()
            || !Component->GetSkeletalMeshAsset() || !Component->GetWorld()
            || Component->GetWorld()->IsGameWorld()) return false;
    Component->TickAnimation(0.0f,false);
    Component->RefreshBoneTransforms();
    Component->UpdateComponentToWorld();
    Component->MarkRenderTransformDirty();
    Component->MarkRenderDynamicDataDirty();
    return true;
}

bool UCreateMHExpressionLibrary::ExportEngineControlPose(USkeletalMesh* SourceMesh,const FString& ControlsJSON,const FString& OutputFile,USkeletalMeshComponent* SourceComponent,bool bIncludeSkinMatrices,int32 LODIndex)
{
    if (!SourceMesh || LODIndex<0 || LODIndex>=SourceMesh->GetLODNum()) return false;
    TSubclassOf<UAnimInstance> PostProcessOverride;
    if (SourceComponent)
    {
        if (SourceComponent->GetSkeletalMeshAsset()!=SourceMesh || SourceComponent->GetDisablePostProcessBlueprint()) return false;
        PostProcessOverride=SourceComponent->GetPostProcessAnimBPClassToBeUsed();
    }
    auto Animation=CreateControlPreviewClip(SourceMesh,ControlsJSON);
    if (!Animation) return false;
    ON_SCOPE_EXIT {Animation->MarkAsGarbage();};
    return ExportEngineFacePose(SourceMesh,Animation,TEXT(""),OutputFile,0.0f,PostProcessOverride,bIncludeSkinMatrices,LODIndex);
}

bool UCreateMHExpressionLibrary::ExportEngineFacePose(USkeletalMesh* SourceMesh,UAnimSequence* Animation,const FString& DNAFile,const FString& OutputFile,float SampleTime,TSubclassOf<UAnimInstance> PostProcessOverride,bool bIncludeSkinMatrices,int32 LODIndex)
{
    if (!IsInGameThread() || !SourceMesh || !Animation || !FMath::IsFinite(SampleTime) || SampleTime<0 || LODIndex<0 || LODIndex>=SourceMesh->GetLODNum()) return false;
    UWorld* World=GEditor?GEditor->GetEditorWorldContext().World():nullptr;
    if (!World) return false;
    const bool bEmbeddedDNA=DNAFile.IsEmpty();
    // Native evaluation only reads the mesh and its DNA binding. Reusing its
    // compiled render data avoids rebuilding a multi-gigabyte face per sample.
    USkeletalMesh* Mesh=bEmbeddedDNA?SourceMesh:DuplicateObject<USkeletalMesh>(SourceMesh,GetTransientPackage());
    UAnimSequence* Clip=DuplicateObject<UAnimSequence>(Animation,GetTransientPackage());
    USkeletalMeshComponent* Component=NewObject<USkeletalMeshComponent>(World,NAME_None,RF_Transient);
    ON_SCOPE_EXIT {Component->DestroyComponent();if (!bEmbeddedDNA) Mesh->MarkAsGarbage();Clip->MarkAsGarbage();};
    UClass* PostClass=PostProcessOverride?PostProcessOverride.Get():Mesh->GetPostProcessAnimBlueprint().Get();
    TSharedPtr<IDNAReader> EvaluatedDNA;
    UDNA* EvaluatedAsset=nullptr;
    FString DNABinding;
    if (bEmbeddedDNA)
    {
        // Assembled characters already carry the rig definition that produced
        // their actual skeleton and correctives. Do not replace it with a
        // separately exported DNA or force a legacy coordinate conversion.
        UDNAAsset* DNA=Cast<UDNAAsset>(Mesh->GetAssetUserDataOfClass(UDNAAsset::StaticClass()));
        UDNAAssetUserData* NativeDNA=Cast<UDNAAssetUserData>(Mesh->GetAssetUserDataOfClass(UDNAAssetUserData::StaticClass()));
        if (!(NativeDNA && NativeDNA->DNAAsset) && !(DNA && DNA->GetDNAReader())) return false;
        EvaluatedDNA=NativeDNA && NativeDNA->DNAAsset?NativeDNA->DNAAsset->GetDNAReader():DNA->GetDNAReader();
        EvaluatedAsset=NativeDNA && NativeDNA->DNAAsset?NativeDNA->DNAAsset.Get():nullptr;
        DNABinding=NativeDNA && NativeDNA->DNAAsset?NativeDNA->DNAAsset->GetPathName():DNA->GetPathName();
    }
    else
    {
        Mesh->RemoveUserDataOfClass(UDNAAssetUserData::StaticClass());
        Mesh->RemoveUserDataOfClass(UDNAAsset::StaticClass());
        UDNAAsset* DNA=NewObject<UDNAAsset>(Mesh,NAME_None,RF_Transient);
        FDNAConfig Config;Config.CoordinateSystemTransformPolicy=ECoordinateSystemTransformPolicy::Transform;
        const TSharedPtr<IDNAReader> Reader=LoadDNAFromFile(DNAFile,Config);
        if (!Reader) return false;
        DNA->SetDNAReader(Reader,EDNACopyPolicy::Alias,ERigLogicInitPolicy::Defer);
        DNA->RestoreLegacyUEMHCCompatibility();
        EvaluatedDNA=DNA->GetDNAReader();
        DNABinding=DNAFile;
        Mesh->AddAssetUserData(DNA);
        if (!PostProcessOverride) PostClass=LoadClass<UAnimInstance>(nullptr,TEXT("/MetaHumanCharacter/Face/ABP_Face_PostProcess.ABP_Face_PostProcess_C"));
    }
    if (!PostClass)
    {
        UE_LOG(LogTemp,Error,TEXT("Native face evaluation needs the source component's post-process class: %s"),*SourceMesh->GetPathName());
        return false;
    }
    if (!bEmbeddedDNA) Mesh->SetPostProcessAnimBlueprint(PostClass);
    Clip->SetSkeleton(Mesh->GetSkeleton());
    // ROM clips contain archetype bone tracks. A raw-control oracle must start
    // from this character's actual reference pose, not another identity's ROM.
    Clip->GetController().RemoveAllBoneTracks(false);
    FAssetCompilingManager::Get().FinishAllCompilation();
    Component->SetSkeletalMesh(Mesh);
    // Creator's transient mesh can hold the very same MIDs as its live actor.
    // Animation material curves must never write sampled blink/wrinkle weights
    // into those shared instances. Flatten only dynamic parents; preserve the
    // constant parent's static permutation and copy all current uniform values.
    for (int32 Slot=0;Slot<Mesh->GetMaterials().Num();++Slot)
    {
        UMaterialInterface* Original=Component->GetMaterial(Slot);
        if (!Original) continue;
        UMaterialInterface* Parent=Original;
        while (UMaterialInstanceDynamic* Dynamic=Cast<UMaterialInstanceDynamic>(Parent)) Parent=Dynamic->Parent;
        UMaterialInstanceDynamic* Private=UMaterialInstanceDynamic::Create(Parent,Component);
        if (!Private) return false;
        Private->CopyMaterialUniformParameters(Original);
        Component->SetMaterial(Slot,Private);
    }
    Component->SetOverridePostProcessAnimBP(PostClass,false);
    Component->SetForcedLOD(LODIndex+1);
    Component->SetDisablePostProcessBlueprint(false);
    Component->SetAnimationMode(EAnimationMode::AnimationSingleNode);
    Component->RegisterComponentWithWorld(World);
    Component->UpdateLODStatus();
    Component->InitAnim(true);
    Component->PlayAnimation(Clip,false);
    Component->SetPosition(SampleTime,false);
    Component->TickAnimation(0.0f,false);
    Component->RefreshBoneTransforms();
    if (Component->GetPredictedLODLevel()!=LODIndex) {UE_LOG(LogTemp,Error,TEXT("Native pose requested LOD%d but evaluated LOD%d"),LODIndex,Component->GetPredictedLODLevel());return false;}
    UAnimInstance* Post=Component->GetPostProcessInstance();
    if (!Post) return false;
    if (!EvaluatedDNA) return false;
    TSet<FName> SupportedCurves;
    TArray<TSharedPtr<FJsonValue>> RawControls;
    for (uint16 I=0;I<EvaluatedDNA->GetRawControlCount();++I)
    {
        const FString Name=EvaluatedDNA->GetRawControlName(I);
        SupportedCurves.Add(FName(*Name.Replace(TEXT("."),TEXT("_"))));
        RawControls.Add(MakeShared<FJsonValueString>(Name));
    }
    TArray<FName> CurveNames;
    UAnimInstance* Instance=Component->GetAnimInstance();
    if (!Instance) return false;
    Instance->GetAllCurveNames(CurveNames);
    for (FName Name:CurveNames)
        if (Name.ToString().StartsWith(TEXT("CTRL_expressions_")) && !SupportedCurves.Contains(Name))
        {
            UE_LOG(LogTemp,Error,TEXT("Native face evaluation received an unsupported raw control: %s"),*Name.ToString());
            return false;
        }
    const auto Transforms=Component->GetBoneSpaceTransforms();
    if (Transforms.Num()!=Mesh->GetRefSkeleton().GetNum()) return false;
    TSharedRef<FJsonObject> Root=MakeShared<FJsonObject>();
    Root->SetStringField(TEXT("source_mesh"),SourceMesh->GetPathName());
    Root->SetBoolField(TEXT("material_instances_isolated"),true);
    Root->SetNumberField(TEXT("lod"),LODIndex);
    Root->SetStringField(TEXT("dna_mode"),bEmbeddedDNA?TEXT("ASSEMBLED_EMBEDDED"):TEXT("EXTERNAL_LEGACY_COMPATIBILITY"));
    Root->SetStringField(TEXT("dna_binding"),DNABinding);
    Root->SetArrayField(TEXT("raw_controls"),RawControls);
    Root->SetNumberField(TEXT("dna_rbf_solver_count"),EvaluatedDNA->GetRBFSolverCount());
    Root->SetNumberField(TEXT("dna_rbf_pose_control_count"),EvaluatedDNA->GetRBFPoseControlCount());
    bool HasPoseCorrectives=false;
    if (bIncludeSkinMatrices) for (uint16 C=0;C<EvaluatedDNA->GetBlendShapeChannelCount();++C) HasPoseCorrectives|=EvaluatedDNA->GetBlendShapeChannelName(C).StartsWith(TEXT("CM_"));
    if (HasPoseCorrectives && EvaluatedAsset)
    {
        auto Context=EvaluatedAsset->GetRigRuntimeContext();
        if (Context && Context->RigLogic && Context->DNAReader)
        {
            Root->SetBoolField(TEXT("runtime_rbf_enabled"),Context->RigLogic->GetConfiguration().LoadRBFBehavior);
            Root->SetNumberField(TEXT("runtime_rbf_solver_count"),Context->DNAReader->GetRBFSolverCount());
            FRigInstance Probe(Context->RigLogic.Get());Probe.SetLOD(LODIndex);
            for (uint16 I=0;I<EvaluatedDNA->GetRawControlCount();++I)
            {
                const FName Name(*EvaluatedDNA->GetRawControlName(I).Replace(TEXT("."),TEXT("_")));
                float Value=0.f;if (Instance->GetCurveValue(Name,Value)) Probe.SetRawControl(I,Value);
            }
            Context->RigLogic->Calculate(&Probe);const auto Values=Probe.GetBlendShapeOutputs();auto Weights=MakeShared<FJsonObject>();
            for (uint16 C=0;C<EvaluatedDNA->GetBlendShapeChannelCount();++C)
            {
                const FString Name=EvaluatedDNA->GetBlendShapeChannelName(C);
                if (Name.StartsWith(TEXT("CM_"))&&Values.IsValidIndex(C)) Weights->SetNumberField(Name,Values[C]);
            }
            Root->SetObjectField(TEXT("runtime_direct_pose_morphs"),Weights);
            auto Mapping=EvaluatedAsset->GetDNAIndexMapping(Mesh->GetSkeleton(),Mesh);
            Root->SetNumberField(TEXT("runtime_mapped_morph_count"),Mapping&&Mapping->MorphTargetCurvesPerLOD.IsValidIndex(LODIndex)?Mapping->MorphTargetCurvesPerLOD[LODIndex].Num():-1);
            if (!LODIndex) Root->SetNumberField(TEXT("runtime_lod0_mapped_morph_count"),Mapping&&Mapping->MorphTargetCurvesPerLOD.Num()?Mapping->MorphTargetCurvesPerLOD[0].Num():-1);
        }
    }
    Root->SetStringField(TEXT("animation"),Animation->GetPathName());
    Root->SetStringField(TEXT("post_process"),Post->GetClass()->GetPathName());
    Root->SetNumberField(TEXT("time"),SampleTime);
    TArray<TSharedPtr<FJsonValue>> Bones;
    for (int32 I=0;I<Transforms.Num();++I)
    {
        const FTransform& T=Transforms[I];const FVector P=T.GetTranslation();const FQuat Q=T.GetRotation();const FVector S=T.GetScale3D();
        TSharedRef<FJsonObject> Bone=MakeShared<FJsonObject>();
        Bone->SetStringField(TEXT("name"),Mesh->GetRefSkeleton().GetBoneName(I).ToString());
        Bone->SetNumberField(TEXT("parent"),Mesh->GetRefSkeleton().GetParentIndex(I));
        Bone->SetArrayField(TEXT("transform"),{MakeShared<FJsonValueNumber>(P.X),MakeShared<FJsonValueNumber>(P.Y),MakeShared<FJsonValueNumber>(P.Z),MakeShared<FJsonValueNumber>(Q.X),MakeShared<FJsonValueNumber>(Q.Y),MakeShared<FJsonValueNumber>(Q.Z),MakeShared<FJsonValueNumber>(Q.W),MakeShared<FJsonValueNumber>(S.X),MakeShared<FJsonValueNumber>(S.Y),MakeShared<FJsonValueNumber>(S.Z)});
        Bones.Add(MakeShared<FJsonValueObject>(Bone));
    }
    Root->SetArrayField(TEXT("bones"),Bones);
    TSharedRef<FJsonObject> Morphs=MakeShared<FJsonObject>();
    for (int32 I=0;I<Mesh->GetMorphTargets().Num();++I)
    {
        UMorphTarget* Morph=Mesh->GetMorphTargets()[I];
        Morphs->SetNumberField(Morph->GetName(),Component->MorphTargetWeights.IsValidIndex(I)?Component->MorphTargetWeights[I]:0.0f);
    }
    Root->SetObjectField(TEXT("morphs"),Morphs);
    // Use engine skin matrices and its full influence implementation. Morph
    // position deltas are linear and receive the same per-vertex skin matrix.
    const auto* Render=Mesh->GetResourceForRendering();
    FSkinWeightVertexBuffer* Weights=Component->GetSkinWeightBuffer(LODIndex);
    if (!Render || !Render->LODRenderData.IsValidIndex(LODIndex) || !Weights) return false;
    const auto& LOD=Render->LODRenderData[LODIndex];
    if (bIncludeSkinMatrices)
    {
        TArray<FVector3f> Normals;Normals.Reserve(LOD.GetNumVertices());
        for (uint32 V=0;V<LOD.GetNumVertices();++V) Normals.Add(FVector3f(LOD.StaticVertexBuffers.StaticMeshVertexBuffer.VertexTangentZ(V)));
        TArray<uint8> Bytes;Bytes.SetNumUninitialized(Normals.Num()*sizeof(FVector3f));FMemory::Memcpy(Bytes.GetData(),Normals.GetData(),Bytes.Num());
        const FString Filename=OutputFile+TEXT(".neutral_normals.f32");if (!FFileHelper::SaveArrayToFile(Bytes,*Filename)) return false;
        Root->SetStringField(TEXT("neutral_normals_file"),FPaths::GetCleanFilename(Filename));
    }
    TArray<FMatrix44f> Skin;
    Component->GetCurrentRefToLocalMatrices(Skin,LODIndex);
    TArray<FVector3f> Positions;
    USkinnedMeshComponent::ComputeSkinnedPositions(Component,Positions,Skin,LOD,*Weights);
    TArray<FVector3f> Deltas;Deltas.Init(FVector3f::ZeroVector,Positions.Num());
    for (int32 I=0;I<Mesh->GetMorphTargets().Num();++I)
    {
        const float Weight=Component->MorphTargetWeights.IsValidIndex(I)?Component->MorphTargetWeights[I]:0.0f;
        if (FMath::Abs(Weight)<1.e-8) continue;
        for (const FMorphTargetDelta& Delta : Mesh->GetMorphTargets()[I]->GetMorphTargetDeltas(LODIndex))
        {
            if (!Deltas.IsValidIndex(Delta.SourceIdx)) return false;
            Deltas[Delta.SourceIdx]+=Delta.PositionDelta*Weight;
        }
    }
    TArray<TSharedPtr<FJsonValue>> Sections;
    TArray<uint8> SkinLinearBytes;
    if (bIncludeSkinMatrices) SkinLinearBytes.SetNumZeroed(Positions.Num()*9*sizeof(float));
    const FSkeletalMeshLODInfo* LODInfo=Mesh->GetLODInfo(LODIndex);
    for (int32 SectionIndex=0;SectionIndex<LOD.RenderSections.Num();++SectionIndex)
    {
        const auto& Section=LOD.RenderSections[SectionIndex];
        int32 MaterialIndex=Section.MaterialIndex;
        // Reduced LODs can omit saliva/attachments and remap section slots.
        // Resolve the same override used by the native skeletal mesh renderer.
        if (LODInfo && LODInfo->LODMaterialMap.IsValidIndex(SectionIndex)
                && LODInfo->LODMaterialMap[SectionIndex]!=INDEX_NONE)
            MaterialIndex=LODInfo->LODMaterialMap[SectionIndex];
        if (!Mesh->GetMaterials().IsValidIndex(MaterialIndex)) return false;
        TSharedRef<FJsonObject> Record=MakeShared<FJsonObject>();
        Record->SetNumberField(TEXT("material"),MaterialIndex);
        Record->SetNumberField(TEXT("start"),Section.BaseVertexIndex);
        Record->SetNumberField(TEXT("count"),Section.NumVertices);
        Record->SetStringField(TEXT("material_name"),Mesh->GetMaterials()[MaterialIndex].ImportedMaterialSlotName.ToString());
        Sections.Add(MakeShared<FJsonValueObject>(Record));
        for (uint32 V=Section.BaseVertexIndex;V<Section.BaseVertexIndex+Section.NumVertices;++V)
        {
            float Linear[9]={0};
            uint32 Offset=0,Count=0;Weights->GetVertexInfluenceOffsetCount(V,Offset,Count);
            for (uint32 W=0;W<Count;++W)
            {
                const uint32 Bone=Weights->GetBoneIndex(V,W);
                if (!Section.BoneMap.IsValidIndex(Bone) || !Skin.IsValidIndex(Section.BoneMap[Bone])) return false;
                const float Weight=float(Weights->GetBoneWeight(V,W))/65535.0f;
                const auto& Matrix=Skin[Section.BoneMap[Bone]];
                Positions[V]+=FVector3f(Matrix.TransformVector(Deltas[V]))*Weight;
                if (bIncludeSkinMatrices) for (int32 R=0;R<3;++R) for (int32 C=0;C<3;++C) Linear[R*3+C]+=Matrix.M[R][C]*Weight;
            }
            if (bIncludeSkinMatrices) FMemory::Memcpy(SkinLinearBytes.GetData()+V*9*sizeof(float),Linear,9*sizeof(float));
        }
    }
    TArray<uint8> Bytes;Bytes.SetNumUninitialized(Positions.Num()*3*sizeof(float));
    FMemory::Memcpy(Bytes.GetData(),Positions.GetData(),Bytes.Num());
    const FString VerticesFile=OutputFile+TEXT(".vertices.f32");
    if (!FFileHelper::SaveArrayToFile(Bytes,*VerticesFile)) return false;
    Root->SetStringField(TEXT("vertices_file"),FPaths::GetCleanFilename(VerticesFile));
    Root->SetNumberField(TEXT("vertex_count"),Positions.Num());
    if (bIncludeSkinMatrices)
    {
        TArray<uint32> Triangles;const auto* Indices=LOD.MultiSizeIndexContainer.GetIndexBuffer();if (!Indices || Indices->Num()%3) return false;
        Triangles.Reserve(Indices->Num());for (int32 I=0;I<Indices->Num();++I) Triangles.Add(Indices->Get(I));
        TArray<uint8> Data;Data.SetNumUninitialized(Triangles.Num()*sizeof(uint32));FMemory::Memcpy(Data.GetData(),Triangles.GetData(),Data.Num());
        const FString Filename=OutputFile+TEXT(".triangles.u32");if (!FFileHelper::SaveArrayToFile(Data,*Filename)) return false;
        Root->SetStringField(TEXT("triangles_file"),FPaths::GetCleanFilename(Filename));
    }
    Root->SetArrayField(TEXT("sections"),Sections);
    TArray<uint8> NeutralBytes,UVBytes;NeutralBytes.SetNumUninitialized(Bytes.Num());UVBytes.SetNumUninitialized(Positions.Num()*2*sizeof(float));
    for (int32 V=0;V<Positions.Num();++V)
    {
        const auto P=LOD.StaticVertexBuffers.PositionVertexBuffer.VertexPosition(V);
        const auto UV=LOD.StaticVertexBuffers.StaticMeshVertexBuffer.GetVertexUV(V,0);
        FMemory::Memcpy(NeutralBytes.GetData()+V*3*sizeof(float),&P,3*sizeof(float));
        FMemory::Memcpy(UVBytes.GetData()+V*2*sizeof(float),&UV,2*sizeof(float));
    }
    if (!FFileHelper::SaveArrayToFile(NeutralBytes,*(OutputFile+TEXT(".neutral.f32"))) || !FFileHelper::SaveArrayToFile(UVBytes,*(OutputFile+TEXT(".uv.f32")))) return false;
    Root->SetStringField(TEXT("neutral_file"),FPaths::GetCleanFilename(OutputFile+TEXT(".neutral.f32")));
    Root->SetStringField(TEXT("uv_file"),FPaths::GetCleanFilename(OutputFile+TEXT(".uv.f32")));
    if (bIncludeSkinMatrices)
    {
        const FString LinearFile=OutputFile+TEXT(".skin_linear.f32");
        if (!FFileHelper::SaveArrayToFile(SkinLinearBytes,*LinearFile)) return false;
        Root->SetStringField(TEXT("skin_linear_file"),FPaths::GetCleanFilename(LinearFile));
        Root->SetStringField(TEXT("skin_linear_layout"),TEXT("VERTEX_ROW_MAJOR_3X3"));
        // Use the engine's import/DNA correspondence, not a positional nearest
        // neighbor that could merge coincident but distinct anatomy vertices.
        TUniquePtr<FDNAToSkelMeshMap> Mapping(USkelMeshDNAUtils::CreateMapForUpdatingNeutralMesh(Mesh));
        const FSkeletalMeshModel* Imported=Mesh->GetImportedModel();
        if (!Mapping || !Imported || !Imported->LODModels.IsValidIndex(LODIndex) || !Mapping->ImportVtxToDNAMeshIndex.IsValidIndex(LODIndex) || !Mapping->ImportVtxToDNAVtxIndex.IsValidIndex(LODIndex)) return false;
        TArray<int32> DNAMesh,DNAIndex;
        for (int32 V=0;V<Positions.Num();++V)
        {
            // InitBaseMesh has already resolved MeshToImportVertexMap into
            // the imported LOD vertex domain. Applying that map twice shifts
            // boundaries between the head, teeth and accessory sections.
            const int32 I=V;
            if (!Mapping->ImportVtxToDNAMeshIndex[LODIndex].IsValidIndex(I)||!Mapping->ImportVtxToDNAVtxIndex[LODIndex].IsValidIndex(I)) return false;
            DNAMesh.Add(Mapping->ImportVtxToDNAMeshIndex[LODIndex][I]);DNAIndex.Add(Mapping->ImportVtxToDNAVtxIndex[LODIndex][I]);
        }
        Root->SetStringField(TEXT("dna_mapping_mode"),TEXT("IMPORTED_LOD_VERTEX_IDENTITIES"));
        auto SaveIndices=[&](const TArray<int32>& Values,const FString& Suffix,const TCHAR* Field)
        {
            TArray<uint8> Data;Data.SetNumUninitialized(Values.Num()*sizeof(int32));FMemory::Memcpy(Data.GetData(),Values.GetData(),Data.Num());
            const FString Path=OutputFile+Suffix;if (!FFileHelper::SaveArrayToFile(Data,*Path)) return false;
            Root->SetStringField(Field,FPaths::GetCleanFilename(Path));return true;
        };
        if (!SaveIndices(DNAMesh,TEXT(".dna_mesh.i32"),TEXT("dna_mesh_file"))||!SaveIndices(DNAIndex,TEXT(".dna_vertex.i32"),TEXT("dna_vertex_file"))) return false;
    }
    TSharedRef<FJsonObject> Curves=MakeShared<FJsonObject>();
    for (FName Name:CurveNames)
        if (Name.ToString().StartsWith(TEXT("CTRL_expressions_")))
            Curves->SetNumberField(Name.ToString(),Instance->GetCurveValue(Name));
    Root->SetObjectField(TEXT("raw_curves"),Curves);
    TSharedRef<FJsonObject> MaterialScalars=MakeShared<FJsonObject>();
    if (UMaterialInstanceDynamic* SkinMaterial=Cast<UMaterialInstanceDynamic>(Component->GetMaterial(0)))
    {
        TArray<FMaterialParameterInfo> Parameters;TArray<FGuid> Ids;
        SkinMaterial->GetAllScalarParameterInfo(Parameters,Ids);
        for (const FMaterialParameterInfo& Parameter : Parameters)
        {
            if (!Parameter.Name.ToString().StartsWith(TEXT("head_wm"))) continue;
            float Value=0.0f;
            if (!SkinMaterial->GetScalarParameterValue(Parameter,Value) || !FMath::IsFinite(Value)) return false;
            MaterialScalars->SetNumberField(Parameter.Name.ToString(),Value);
        }
    }
    Root->SetObjectField(TEXT("material_scalars"),MaterialScalars);
    FString Text;
    return FJsonSerializer::Serialize(Root,TJsonWriterFactory<>::Create(&Text)) && FFileHelper::SaveStringToFile(Text,*OutputFile,FFileHelper::EEncodingOptions::ForceUTF8WithoutBOM);
}
