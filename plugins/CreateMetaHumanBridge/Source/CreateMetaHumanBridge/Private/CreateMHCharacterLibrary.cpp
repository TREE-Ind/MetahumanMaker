#include "CreateMHCharacterLibrary.h"
#include "MetaHumanCharacter.h"
#include "MetaHumanCharacterEditorSubsystem.h"
#include "Editor.h"
#include "Dom/JsonObject.h"
#include "Serialization/JsonSerializer.h"
#include "Serialization/JsonWriter.h"
#include "MetaHumanCharacterIdentity.h"
#include "MetaHumanRigEvaluatedState.h"
#include "DNAAsset.h"
#include "DNAAssetUserData.h"
#include "DNA.h"
#include "DNAReader.h"
#include "Engine/SkeletalMesh.h"
#include "Misc/FileHelper.h"
#include "Misc/Paths.h"
#include "HAL/FileManager.h"
#include "MetaHumanCollectionPipeline.h"
#include "Subsystem/MetaHumanCharacterBuild.h"
#include "Misc/PackageName.h"

bool UCreateMHCharacterLibrary::BuildWithPipeline(UObject* Object, UObject* PipelineObject, const FString& ParametersJson, bool bValidateOnly)
{
    if (!IsInGameThread() || !GEditor) return false;
    auto* Character=Cast<UMetaHumanCharacter>(Object);
    auto* Pipeline=Cast<UMetaHumanCollectionPipeline>(PipelineObject);
    if (!Character || !Pipeline) return false;
    TSharedPtr<FJsonObject> Values;
    if (!FJsonSerializer::Deserialize(TJsonReaderFactory<>::Create(ParametersJson),Values) || !Values.IsValid()) return false;
    int32 Type=-1,Quality=-1;
    FString Root,Name,Common,Animation;
    if (!Values->TryGetNumberField(TEXT("pipeline_type"),Type) || !Values->TryGetNumberField(TEXT("pipeline_quality"),Quality) ||
        !Values->TryGetStringField(TEXT("root"),Root) || !Values->TryGetStringField(TEXT("name"),Name) ||
        !Values->TryGetStringField(TEXT("common"),Common) || !Values->TryGetStringField(TEXT("animation"),Animation)) return false;
    if (!StaticEnum<EMetaHumanDefaultPipelineType>()->IsValidEnumValue(Type) || !StaticEnum<EMetaHumanQualityLevel>()->IsValidEnumValue(Quality)) return false;
    if (Type!=int32(EMetaHumanDefaultPipelineType::Cinematic) && Type!=int32(EMetaHumanDefaultPipelineType::Optimized)) return false;
    if (!Root.StartsWith(TEXT("/Game/")) || !Common.StartsWith(Root+TEXT("/")) ||
        !FPackageName::IsValidLongPackageName(Root) || !FPackageName::IsValidLongPackageName(Common) || Name.IsEmpty()) return false;
    for (TCHAR CharacterInName : Name) if (!FChar::IsAlnum(CharacterInName) && CharacterInName!=TEXT('_')) return false;
    FMetaHumanCharacterEditorBuildParameters Params;
    Params.PipelineType=EMetaHumanDefaultPipelineType(Type);Params.PipelineQuality=EMetaHumanQualityLevel(Quality);
    Params.AbsoluteBuildPath=Root;Params.NameOverride=Name;Params.CommonFolderPath=Common;Params.AnimationSystemName=FName(*Animation);
    Params.bEnableWardrobeItemValidation=true;Params.PipelineOverride=Pipeline;
    if (bValidateOnly) return true;
    auto* Subsystem=GEditor->GetEditorSubsystem<UMetaHumanCharacterEditorSubsystem>();
    if (!Subsystem || !Subsystem->IsObjectAddedForEditing(Character) || !Subsystem->CanBuildMetaHuman(Character,false)) return false;
    Subsystem->BuildMetaHuman(Character,Params);
    return true;
}

bool UCreateMHCharacterLibrary::RefreshSkinPreview(UObject* Object)
{
    if (!IsInGameThread() || !GEditor) return false;
    auto* Character=Cast<UMetaHumanCharacter>(Object);
    auto* Subsystem=GEditor->GetEditorSubsystem<UMetaHumanCharacterEditorSubsystem>();
    if (!Character || !Subsystem || !Subsystem->IsObjectAddedForEditing(Character)) return false;
    Subsystem->UpdateCharacterPreviewMaterial(Character,Character->PreviewMaterialType);
    return true;
}

FString UCreateMHCharacterLibrary::InspectCharacter(UObject* Object)
{
    if (!IsInGameThread() || !GEditor) return TEXT("{}");
    auto* Character=Cast<UMetaHumanCharacter>(Object);
    auto* Subsystem=GEditor->GetEditorSubsystem<UMetaHumanCharacterEditorSubsystem>();
    if (!Character || !Subsystem) return TEXT("{}");
    const bool bEditing=Subsystem->IsObjectAddedForEditing(Character);
    FText Reason;
    const bool bReady=bEditing && Subsystem->CanBuildMetaHuman(TNotNull<const UMetaHumanCharacter*>(Character),Reason);
    const auto State=Subsystem->GetRiggingState(Character);
    const bool bRigPending=Subsystem->IsAutoRiggingFace(Character) || State==EMetaHumanCharacterRigState::RigPending;
    const bool bTexturesPending=Subsystem->IsRequestingHighResolutionTextures(Character);
    const bool bStoredPrerequisites=State==EMetaHumanCharacterRigState::Rigged &&
        Character->HasHighResolutionTextures() && !bRigPending && !bTexturesPending;
    if (!bEditing)
    {
        if (bRigPending) Reason=FText::FromString(TEXT("Creator is rigging this character."));
        else if (bTexturesPending) Reason=FText::FromString(TEXT("Creator is downloading texture sources."));
        else if (State!=EMetaHumanCharacterRigState::Rigged) Reason=FText::FromString(TEXT("Create the character rig in Creator."));
        else if (!Character->HasHighResolutionTextures()) Reason=FText::FromString(TEXT("Download Texture Sources in Creator before assembly."));
        else Reason=FText::FromString(TEXT("Saved rig and texture sources are available. Assembly will load the character and verify live readiness."));
    }
    TSharedRef<FJsonObject> Report=MakeShared<FJsonObject>();
    Report->SetBoolField(TEXT("ready"),bReady);
    Report->SetBoolField(TEXT("readiness_verified"),bEditing);
    Report->SetBoolField(TEXT("can_request_build"),bEditing ? bReady : bStoredPrerequisites);
    Report->SetStringField(TEXT("inspection_scope"),bEditing ? TEXT("LIVE_EDITOR") : TEXT("SAVED_CHARACTER"));
    Report->SetStringField(TEXT("reason"),Reason.ToString());
    Report->SetStringField(TEXT("rig_state"),State==EMetaHumanCharacterRigState::Rigged ? TEXT("RIGGED") :
        (State==EMetaHumanCharacterRigState::RigPending ? TEXT("PENDING") : TEXT("UNRIGGED")));
    Report->SetBoolField(TEXT("rig_request_pending"),bRigPending);
    Report->SetBoolField(TEXT("texture_request_pending"),bTexturesPending);
    Report->SetBoolField(TEXT("has_texture_sources"),Character->HasHighResolutionTextures());
    // Preview mip preparation is not saved-source validation. Its mismatch
    // query belongs only to an initialized editing session.
    if (bEditing) Report->SetBoolField(TEXT("needs_texture_download"),Character->NeedsToDownloadTextureSources());
    Report->SetBoolField(TEXT("has_face_dna"),Character->HasFaceDNA());
    Report->SetBoolField(TEXT("has_body_dna"),Character->HasBodyDNA());
    FString Result;
    FJsonSerializer::Serialize(Report,TJsonWriterFactory<>::Create(&Result));
    return Result;
}

bool UCreateMHCharacterLibrary::ExportFaceState(UObject* Object, const FString& OutputFile)
{
    if (!IsInGameThread() || !GEditor || OutputFile.IsEmpty()) return false;
    auto* Character=Cast<UMetaHumanCharacter>(Object);
    auto* Subsystem=GEditor->GetEditorSubsystem<UMetaHumanCharacterEditorSubsystem>();
    if (!Character || !Subsystem || !Subsystem->IsObjectAddedForEditing(Character)) return false;
    const USkeletalMesh* FaceMesh=Subsystem->GetFaceEditMesh(Character);
    // The asset-user-data query is not const-qualified in UE; read it only.
    USkeletalMesh* Mesh=const_cast<USkeletalMesh*>(FaceMesh);
    TSharedPtr<IDNAReader> Reader;
    if (auto* Data=Cast<UDNAAssetUserData>(Mesh->GetAssetUserDataOfClass(UDNAAssetUserData::StaticClass())))
    {
        if (Data->DNAAsset) Reader=Data->DNAAsset->GetDNAReader();
    }
    else if (auto* LegacyData=Cast<UDNAAsset>(Mesh->GetAssetUserDataOfClass(UDNAAsset::StaticClass()))) Reader=LegacyData->GetDNAReader();
    if (!Reader || !Reader->GetLODCount()) return false;
    const auto State=Subsystem->GetFaceState(Character);
    const auto Settings=State->GetSettings();
    const FMetaHumanRigEvaluatedState Evaluated=State->Evaluate();
    if (Evaluated.Vertices.IsEmpty()) return false;
    TArray<uint8> Bytes;
    TArray<TSharedPtr<FJsonValue>> Meshes;
    for (const uint16 Index:Reader->GetMeshIndicesForLOD(0))
    {
        const uint32 Count=Reader->GetVertexPositionCount(Index);
        if (Count>1000000 || Bytes.Num()>12000000-static_cast<int32>(Count)*12) return false;
        TSharedRef<FJsonObject> Part=MakeShared<FJsonObject>();
        Part->SetStringField(TEXT("name"),Reader->GetMeshName(Index));
        Part->SetNumberField(TEXT("dna_mesh_index"),Index);
        Part->SetNumberField(TEXT("vertex_count"),Count);
        Part->SetNumberField(TEXT("byte_offset"),Bytes.Num());
        for (uint32 Vertex=0;Vertex<Count;++Vertex)
        {
            const FVector3f Point=State->GetVertex(Evaluated.Vertices,Index,Vertex);
            if (!FMath::IsFinite(Point.X) || !FMath::IsFinite(Point.Y) || !FMath::IsFinite(Point.Z)) return false;
            const float Values[3]={Point.X,Point.Y,Point.Z};
            Bytes.Append(reinterpret_cast<const uint8*>(Values),sizeof(Values));
        }
        Meshes.Add(MakeShared<FJsonValueObject>(Part));
    }
    if (Meshes.IsEmpty()) return false;
    TSharedRef<FJsonObject> Report=MakeShared<FJsonObject>();
    Report->SetStringField(TEXT("schema"),TEXT("CM_FACE_STATE_V1"));
    Report->SetStringField(TEXT("character"),Character->GetPathName());
    Report->SetStringField(TEXT("source"),TEXT("EDITOR_EVALUATED_FACE_STATE"));
    Report->SetStringField(TEXT("coordinate_system"),TEXT("UE_CM"));
    Report->SetStringField(TEXT("dtype"),TEXT("<f4"));
    Report->SetNumberField(TEXT("global_vertex_delta_scale"),Settings.GlobalVertexDeltaScale());
    Report->SetNumberField(TEXT("high_frequency_delta_scale"),Settings.GlobalHighFrequencyScale());
    Report->SetBoolField(TEXT("compatibility_evaluation"),Settings.UseCompatibilityEvaluation());
    Report->SetArrayField(TEXT("meshes"),Meshes);
    const FString Binary=OutputFile+TEXT(".vertices.f32");
    Report->SetStringField(TEXT("vertices_file"),FPaths::GetCleanFilename(Binary));
    Report->SetNumberField(TEXT("bytes"),Bytes.Num());
    FString JSON;
    if (!FJsonSerializer::Serialize(Report,TJsonWriterFactory<>::Create(&JSON))) return false;
    if (!IFileManager::Get().MakeDirectory(*FPaths::GetPath(FPaths::ConvertRelativePathToFull(OutputFile)),true)) return false;
    return FFileHelper::SaveArrayToFile(Bytes,*Binary) &&
        FFileHelper::SaveStringToFile(JSON,*OutputFile,FFileHelper::EEncodingOptions::ForceUTF8WithoutBOM);
}
