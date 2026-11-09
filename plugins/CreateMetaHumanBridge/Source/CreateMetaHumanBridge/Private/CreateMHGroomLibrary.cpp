#include "CreateMHGroomLibrary.h"
#include "GroomAsset.h"
#include "HairDescription.h"
#include "HairAttributes.h"
#include "AssetRegistry/AssetRegistryModule.h"
#include "Dom/JsonObject.h"
#include "Serialization/JsonReader.h"
#include "Serialization/JsonSerializer.h"
#include "Misc/FileHelper.h"
#include "Misc/PackageName.h"
#include "Modules/ModuleManager.h"

IMPLEMENT_MODULE(FDefaultModuleImpl, CreateMetaHumanBridge)

static TArray<TSharedPtr<FJsonValue>> VectorJSON(const FVector3f& V)
{
    return {MakeShared<FJsonValueNumber>(V.X), MakeShared<FJsonValueNumber>(V.Y), MakeShared<FJsonValueNumber>(V.Z)};
}

bool UCreateMHGroomLibrary::ExportGroomJSON(UGroomAsset* Groom, const FString& FilePath)
{
    if (!Groom) return false;
    FHairDescription Desc = Groom->GetHairDescription();
    if (!Desc.IsValid()) return false;
    auto Positions = Desc.VertexAttributes().GetAttributesRef<FVector3f>(HairAttribute::Vertex::Position);
    auto Counts = Desc.StrandAttributes().GetAttributesRef<int32>(HairAttribute::Strand::VertexCount);
    auto Groups = Desc.StrandAttributes().GetAttributesRef<int32>(HairAttribute::Strand::GroupID);
    auto Guides = Desc.StrandAttributes().GetAttributesRef<int32>(HairAttribute::Strand::Guide);
    auto RootUV = Desc.StrandAttributes().GetAttributesRef<FVector2f>(HairAttribute::Strand::RootUV);
    auto Widths = Desc.VertexAttributes().GetAttributesRef<float>(HairAttribute::Vertex::Width);
    auto Colors = Desc.VertexAttributes().GetAttributesRef<FVector3f>(HairAttribute::Vertex::Color);
    if (!Positions.IsValid() || !Counts.IsValid()) return false;
    TArray<TSharedPtr<FJsonValue>> Curves;
    int32 Offset = 0;
    for (int32 I=0; I<Desc.GetNumStrands(); ++I)
    {
        FStrandID Strand(I);
        TSharedRef<FJsonObject> Curve = MakeShared<FJsonObject>();
        TArray<TSharedPtr<FJsonValue>> Points, CurveWidths, CurveColors;
        for (int32 J=0; J<Counts[Strand]; ++J)
        {
            FVertexID Vertex(Offset++);
            Points.Add(MakeShared<FJsonValueArray>(VectorJSON(Positions[Vertex])));
            CurveWidths.Add(MakeShared<FJsonValueNumber>(Widths.IsValid()? Widths[Vertex] : 0.01f));
            if (Colors.IsValid()) CurveColors.Add(MakeShared<FJsonValueArray>(VectorJSON(Colors[Vertex])));
        }
        Curve->SetArrayField(TEXT("points"), Points);
        Curve->SetArrayField(TEXT("widths"), CurveWidths);
        if (CurveColors.Num()) Curve->SetArrayField(TEXT("colors"), CurveColors);
        Curve->SetNumberField(TEXT("group"), Groups.IsValid()? Groups[Strand] : 0);
        Curve->SetNumberField(TEXT("guide"), Guides.IsValid()? Guides[Strand] : 0);
        if (RootUV.IsValid()) Curve->SetArrayField(TEXT("root_uv"), {MakeShared<FJsonValueNumber>(RootUV[Strand].X), MakeShared<FJsonValueNumber>(RootUV[Strand].Y)});
        Curves.Add(MakeShared<FJsonValueObject>(Curve));
    }
    TSharedRef<FJsonObject> Root = MakeShared<FJsonObject>();
    Root->SetNumberField(TEXT("schema_version"), 1);
    Root->SetStringField(TEXT("coordinates"), TEXT("Unreal centimeters"));
    Root->SetArrayField(TEXT("curves"), Curves);
    FString Text;
    FJsonSerializer::Serialize(Root, TJsonWriterFactory<>::Create(&Text));
    return FFileHelper::SaveStringToFile(Text, *FilePath);
}

UGroomAsset* UCreateMHGroomLibrary::ImportGroomJSON(const FString& FilePath, const FString& AssetPath)
{
    FString Text;
    TSharedPtr<FJsonObject> Root;
    if (!FFileHelper::LoadFileToString(Text, *FilePath) || !FJsonSerializer::Deserialize(TJsonReaderFactory<>::Create(Text), Root)) return nullptr;
    if (!AssetPath.StartsWith(TEXT("/Game/")) || !FPackageName::IsValidLongPackageName(AssetPath) || FindPackage(nullptr, *AssetPath)) return nullptr;
    const TArray<TSharedPtr<FJsonValue>>* Curves;
    if (!Root->TryGetArrayField(TEXT("curves"), Curves) || Curves->Num()==0) return nullptr;
    FHairDescription Desc;
    TSet<int32> GroupIDs;
    for (const auto& Value : *Curves)
    {
        const TSharedPtr<FJsonObject>* Curve;
        if (!Value->TryGetObject(Curve)) return nullptr;
        const TArray<TSharedPtr<FJsonValue>>* Points;
        if (!(*Curve)->TryGetArrayField(TEXT("points"), Points) || Points->Num()<2) return nullptr;
        FStrandID Strand = Desc.AddStrand();
        int32 Group = (*Curve)->GetIntegerField(TEXT("group"));
        GroupIDs.Add(Group);
        SetHairStrandAttribute(Desc, Strand, HairAttribute::Strand::VertexCount, Points->Num());
        SetHairStrandAttribute(Desc, Strand, HairAttribute::Strand::GroupID, Group);
        SetHairStrandAttribute(Desc, Strand, HairAttribute::Strand::Guide, (*Curve)->GetIntegerField(TEXT("guide")));
        const TArray<TSharedPtr<FJsonValue>>* UV;
        if ((*Curve)->TryGetArrayField(TEXT("root_uv"), UV) && UV->Num()==2)
            SetHairStrandAttribute(Desc, Strand, HairAttribute::Strand::RootUV, FVector2f((*UV)[0]->AsNumber(), (*UV)[1]->AsNumber()));
        const TArray<TSharedPtr<FJsonValue>>* Widths = nullptr;
        (*Curve)->TryGetArrayField(TEXT("widths"), Widths);
        for (int32 I=0; I<Points->Num(); ++I)
        {
            const auto& P = (*Points)[I]->AsArray();
            if (P.Num()!=3) return nullptr;
            FVector3f Position(P[0]->AsNumber(), P[1]->AsNumber(), P[2]->AsNumber());
            if (Position.ContainsNaN()) return nullptr;
            FVertexID Vertex = Desc.AddVertex();
            SetHairVertexAttribute(Desc, Vertex, HairAttribute::Vertex::Position, Position);
            SetHairVertexAttribute(Desc, Vertex, HairAttribute::Vertex::Width,
                Widths && Widths->IsValidIndex(I)? float((*Widths)[I]->AsNumber()) : .01f);
            const TArray<TSharedPtr<FJsonValue>>* Colors;
            if ((*Curve)->TryGetArrayField(TEXT("colors"), Colors) && Colors->IsValidIndex(I))
            {
                const auto& C = (*Colors)[I]->AsArray();
                if (C.Num()==3) SetHairVertexAttribute(Desc, Vertex, HairAttribute::Vertex::Color, FVector3f(C[0]->AsNumber(), C[1]->AsNumber(), C[2]->AsNumber()));
            }
        }
    }
    UPackage* Package = CreatePackage(*AssetPath);
    UGroomAsset* Groom = NewObject<UGroomAsset>(Package, *FPackageName::GetLongPackageAssetName(AssetPath), RF_Public|RF_Standalone|RF_Transactional);
    Groom->SetNumGroup(GroupIDs.Num());
    Groom->CommitHairDescription(MoveTemp(Desc), EHairDescriptionType::Source);
    if (!Groom->CacheDerivedDatas()) return nullptr;
    FAssetRegistryModule::AssetCreated(Groom);
    Groom->MarkPackageDirty();
    return Groom;
}
