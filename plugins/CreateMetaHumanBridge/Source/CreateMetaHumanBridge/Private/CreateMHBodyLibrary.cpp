#include "CreateMHBodyLibrary.h"
#include "MetaHumanCharacterBodyIdentity.h"
#include "MetaHumanRigEvaluatedState.h"
#include "DNAReader.h"
#include "DNAUtils.h"
#include "Dom/JsonObject.h"
#include "Serialization/JsonReader.h"
#include "Serialization/JsonSerializer.h"
#include "Misc/FileHelper.h"
#include "Misc/Paths.h"

namespace
{
TSharedPtr<FMetaHumanCharacterBodyIdentity> BodyModel;
FString LoadedModelPath;
bool WriteJSON(const FString& Path,const TSharedRef<FJsonObject>& Data)
{
    FString Text;
    return FJsonSerializer::Serialize(Data,TJsonWriterFactory<>::Create(&Text)) &&
        FFileHelper::SaveStringToFile(Text,*Path,FFileHelper::EEncodingOptions::ForceUTF8WithoutBOM);
}
}

bool UCreateMHBodyLibrary::EvaluateBodyJSON(const FString& RequestFile,const FString& OutputFile)
{
    FString Text;
    TSharedPtr<FJsonObject> Request;
    if (!FFileHelper::LoadFileToString(Text,*RequestFile) || !FJsonSerializer::Deserialize(TJsonReaderFactory<>::Create(Text),Request)) return false;
    const FString Path=Request->GetStringField(TEXT("model_path"));
    if (!FPaths::FileExists(Path/TEXT("body_model.dna"))) return false;
    if (!BodyModel || LoadedModelPath!=Path)
    {
        auto Model=MakeShared<FMetaHumanCharacterBodyIdentity>();
        // No asset factory, character editor subsystem or texture synthesis is involved.
        // An empty legacy directory avoids loading unused compatibility presets.
        if (!Model->Init(Path,Path/TEXT("__cm_unused_legacy__"))) return false;
        BodyModel=Model;
        LoadedModelPath=Path;
    }
    auto State=BodyModel->CreateState();
    if (!State) return false;
    FString StateFile;
    if (Request->TryGetStringField(TEXT("state_file"),StateFile) && !StateFile.IsEmpty())
    {
        TArray<uint8> StateBytes;
        if (!FFileHelper::LoadFileToArray(StateBytes,*StateFile) ||
            !State->Deserialize(FSharedBuffer::Clone(StateBytes.GetData(),StateBytes.Num()))) return false;
    }
    auto Constraints=State->GetBodyConstraints(false);
    for (auto& Constraint:Constraints) Constraint.bIsActive=false;
    const TSharedPtr<FJsonObject>* Values=nullptr;
    if (Request->TryGetObjectField(TEXT("constraints"),Values))
    {
        for (const auto& Pair:(*Values)->Values)
        {
            const FString Name(Pair.Key.ToView());
            auto Constraint=Constraints.FindByPredicate([&](const auto& Item){return Item.Name.ToString()==Name;});
            double Value;
            if (!Constraint || !Pair.Value->TryGetNumber(Value) || !FMath::IsFinite(Value) || Value<Constraint->MinMeasurement || Value>Constraint->MaxMeasurement) return false;
            Constraint->bIsActive=true;
            Constraint->TargetMeasurement=Value;
        }
        if ((*Values)->Values.Num()) State->EvaluateBodyConstraints(Constraints);
    }
    Constraints=State->GetBodyConstraints(false);
    TSharedRef<FJsonObject> Result=MakeShared<FJsonObject>();
    Result->SetNumberField(TEXT("schema"),1);
    Result->SetStringField(TEXT("coordinates"),TEXT("UE_CENTIMETERS"));
    Result->SetNumberField(TEXT("body_vertices"),BodyModel->GetNumLOD0MeshVertices(false));
    Result->SetNumberField(TEXT("combined_vertices"),BodyModel->GetNumLOD0MeshVertices(true));
    TArray<TSharedPtr<FJsonValue>> OutputConstraints;
    for (int32 I=0;I<Constraints.Num();++I)
    {
        const auto& C=Constraints[I];
        TSharedRef<FJsonObject> Item=MakeShared<FJsonObject>();
        Item->SetStringField(TEXT("name"),C.Name.ToString());
        Item->SetBoolField(TEXT("active"),C.bIsActive);
        Item->SetNumberField(TEXT("target"),C.TargetMeasurement);
        Item->SetNumberField(TEXT("min"),C.MinMeasurement);
        Item->SetNumberField(TEXT("max"),C.MaxMeasurement);
        Item->SetNumberField(TEXT("actual"),State->GetMeasurement(I));
        OutputConstraints.Add(MakeShared<FJsonValueObject>(Item));
    }
    Result->SetArrayField(TEXT("constraints"),OutputConstraints);
    const auto Vertices=State->GetVerticesAndVertexNormals();
    const int32 Count=BodyModel->GetNumLOD0MeshVertices(false);
    TArray<uint8> Bytes;
    Bytes.SetNumUninitialized(Count*3*sizeof(float));
    for (int32 V=0;V<Count;++V)
    {
        const FVector3f Point=State->GetVertex(Vertices.Vertices,0,V);
        if (!FMath::IsFinite(Point.X) || !FMath::IsFinite(Point.Y) || !FMath::IsFinite(Point.Z)) return false;
        FMemory::Memcpy(Bytes.GetData()+V*3*sizeof(float),&Point,sizeof(float)*3);
    }
    const FString VertexFile=OutputFile+TEXT(".vertices.f32");
    if (!FFileHelper::SaveArrayToFile(Bytes,*VertexFile)) return false;
    Result->SetStringField(TEXT("vertices_file"),FPaths::GetCleanFilename(VertexFile));
    FSharedBuffer Serialized;
    if (!State->Serialize(Serialized)) return false;
    const FString OutputState=OutputFile+TEXT(".state");
    if (!FFileHelper::SaveArrayToFile(TArrayView<const uint8>(static_cast<const uint8*>(Serialized.GetData()),Serialized.GetSize()),*OutputState)) return false;
    Result->SetStringField(TEXT("state_file"),FPaths::GetCleanFilename(OutputState));
    bool IncludeLayouts=false;
    Request->TryGetBoolField(TEXT("include_layouts"),IncludeLayouts);
    if (IncludeLayouts)
    {
        FDNAConfig Config;
        Config.CoordinateSystemTransformPolicy=ECoordinateSystemTransformPolicy::Transform;
        Config.MaxLODPerPlatform=FPerPlatformInt(0);
        Config.MinLODPerPlatform=FPerPlatformInt(0);
        auto Reader=LoadDNAFromFile(FPaths::EnginePluginsDir()/TEXT("MetaHuman/MetaHumanCoreTechLib/Content/ArchetypeDNA/SKM_Body.dna"),Config);
        if (!Reader || Reader->GetVertexPositionCount(0)!=Count) return false;
        TArray<TSharedPtr<FJsonValue>> Layouts;
        for (uint32 L=0;L<Reader->GetVertexLayoutCount(0);++L)
        {
            const auto Layout=Reader->GetVertexLayout(0,L);
            const auto UV=Reader->GetVertexTextureCoordinate(0,Layout.TextureCoordinate);
            Layouts.Add(MakeShared<FJsonValueArray>(TArray<TSharedPtr<FJsonValue>>{
                MakeShared<FJsonValueNumber>(Layout.Position),MakeShared<FJsonValueNumber>(UV.U),MakeShared<FJsonValueNumber>(UV.V)}));
        }
        Result->SetArrayField(TEXT("layouts"),Layouts);
    }
    return WriteJSON(OutputFile,Result);
}
