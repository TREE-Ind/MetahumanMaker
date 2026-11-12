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
#include "Misc/SecureHash.h"
#include "Modules/ModuleManager.h"

IMPLEMENT_MODULE(FDefaultModuleImpl, CreateMetaHumanBridge)

static TArray<TSharedPtr<FJsonValue>> VectorJSON(const FVector3f& V)
{
    return {MakeShared<FJsonValueNumber>(V.X), MakeShared<FJsonValueNumber>(V.Y), MakeShared<FJsonValueNumber>(V.Z)};
}

bool UCreateMHGroomLibrary::InitializeGroomGroups(UGroomAsset* Groom)
{
    if (!Groom) return false;
    const FHairDescription& Desc=Groom->GetHairDescription();
    if (!Desc.IsValid()) return false;
    auto Groups=Desc.StrandAttributes().GetAttributesRef<int32>(HairAttribute::Strand::GroupID);
    TArray<int32> Order;TSet<int32> Seen;
    for (int32 I=0;I<Desc.GetNumStrands();++I)
    {
        const int32 ID=Groups.IsValid()?Groups[FStrandID(I)]:0;
        if (!Seen.Contains(ID)) {Seen.Add(ID);Order.Add(ID);}
    }
    if (Order.Num()!=Groom->GetHairGroupsInfo().Num()) return false;
    for (int32 I=0;I<Order.Num();++I)
    {
        Groom->GetHairGroupsInfo()[I].GroupIndex=I;
        Groom->GetHairGroupsInfo()[I].GroupID=Order[I];
    }
    Groom->MarkPackageDirty();return true;
}

FString UCreateMHGroomLibrary::FingerprintGroom(UGroomAsset* Groom)
{
    TSharedRef<FJsonObject> Report = MakeShared<FJsonObject>();
    Report->SetNumberField(TEXT("schema"), 1);
    auto Finish = [&Report](const TCHAR* Status)
    {
        Report->SetStringField(TEXT("status"), Status);
        FString Text;
        FJsonSerializer::Serialize(Report, TJsonWriterFactory<>::Create(&Text));
        return Text;
    };
    if (!Groom) return Finish(TEXT("missing_groom"));
    FHairDescription Desc = Groom->GetHairDescription();
    if (!Desc.IsValid()) return Finish(TEXT("source_description_unavailable"));
    auto Positions = Desc.VertexAttributes().GetAttributesRef<FVector3f>(HairAttribute::Vertex::Position);
    auto Counts = Desc.StrandAttributes().GetAttributesRef<int32>(HairAttribute::Strand::VertexCount);
    auto Groups = Desc.StrandAttributes().GetAttributesRef<int32>(HairAttribute::Strand::GroupID);
    auto Guides = Desc.StrandAttributes().GetAttributesRef<int32>(HairAttribute::Strand::Guide);
    auto UV = Desc.StrandAttributes().GetAttributesRef<FVector2f>(HairAttribute::Strand::RootUV);
    auto Widths = Desc.VertexAttributes().GetAttributesRef<float>(HairAttribute::Vertex::Width);
    auto Colors = Desc.VertexAttributes().GetAttributesRef<FVector3f>(HairAttribute::Vertex::Color);
    if (!Positions.IsValid() || !Counts.IsValid()) return Finish(TEXT("invalid_description"));
    TMap<int32,int32> Indices;
    const int32 NumGroups = Groom->GetHairGroupsInfo().Num();
    if (!NumGroups) return Finish(TEXT("empty_groom"));
    TArray<TUniquePtr<FSHA1>> Hashes;
    TArray<int32> CurveCounts, PointCounts;
    CurveCounts.SetNumZeroed(NumGroups); PointCounts.SetNumZeroed(NumGroups);
    // Hash scalar attributes explicitly, excluding struct padding and group IDs.
    // Group array order is retained, while native GroupID labels may be remapped.
    const uint8 Flags[] = {uint8(UV.IsValid()),uint8(Widths.IsValid()),uint8(Colors.IsValid())};
    for (int32 I=0; I<NumGroups; ++I)
    {
        const int32 ID = Groom->GetHairGroupsInfo()[I].GroupID;
        if (Indices.Contains(ID)) return Finish(TEXT("duplicate_group_id"));
        Indices.Add(ID,I); Hashes.Add(MakeUnique<FSHA1>());
        Hashes.Last()->Update(Flags,UE_ARRAY_COUNT(Flags));
    }
    int32 Offset=0;
    for (int32 I=0; I<Desc.GetNumStrands(); ++I)
    {
        const FStrandID Strand(I);
        const int32 Count=Counts[Strand], Guide=Guides.IsValid()?Guides[Strand]:0;
        const int32* Index=Indices.Find(Groups.IsValid()?Groups[Strand]:0);
        if (!Index || Count<2 || Offset>Desc.GetNumVertices()-Count) return Finish(TEXT("invalid_curve_layout"));
        FSHA1& Hash=*Hashes[*Index];
        auto Scalar=[&Hash](const auto& Value){Hash.Update(reinterpret_cast<const uint8*>(&Value),sizeof(Value));};
        Scalar(Count); Scalar(Guide); ++CurveCounts[*Index]; PointCounts[*Index]+=Count;
        if (UV.IsValid())
        {
            const FVector2f V=UV[Strand];
            if (V.ContainsNaN()) return Finish(TEXT("invalid_root_uv"));
            Scalar(V.X); Scalar(V.Y);
        }
        for (int32 J=0; J<Count; ++J)
        {
            const FVertexID Vertex(Offset++);
            const FVector3f P=Positions[Vertex];
            const float W=Widths.IsValid()?Widths[Vertex]:.01f;
            if (P.ContainsNaN() || !FMath::IsFinite(W)) return Finish(TEXT("invalid_vertex"));
            Scalar(P.X); Scalar(P.Y); Scalar(P.Z); Scalar(W);
            if (Colors.IsValid())
            {
                const FVector3f C=Colors[Vertex];
                if (C.ContainsNaN()) return Finish(TEXT("invalid_color"));
                Scalar(C.X); Scalar(C.Y); Scalar(C.Z);
            }
        }
    }
    if (Offset!=Desc.GetNumVertices()) return Finish(TEXT("invalid_point_count"));
    TArray<TSharedPtr<FJsonValue>> Rows;
    for (int32 I=0; I<NumGroups; ++I)
    {
        if (!CurveCounts[I]) return Finish(TEXT("empty_group"));
        Hashes[I]->Final(); uint8 Digest[FSHA1::DigestSize]; Hashes[I]->GetHash(Digest);
        TSharedRef<FJsonObject> Row=MakeShared<FJsonObject>();
        Row->SetStringField(TEXT("sha1"),BytesToHex(Digest,FSHA1::DigestSize));
        Row->SetNumberField(TEXT("curves"),CurveCounts[I]); Row->SetNumberField(TEXT("points"),PointCounts[I]);
        Rows.Add(MakeShared<FJsonValueObject>(Row));
    }
    Report->SetArrayField(TEXT("groups"),Rows);
    return Finish(TEXT("complete"));
}

FString UCreateMHGroomLibrary::CompareGroomGroups(UGroomAsset* Source, int32 SourceGroup, UGroomAsset* Target, int32 TargetGroup, bool bComparePositions)
{
    TSharedRef<FJsonObject> Report = MakeShared<FJsonObject>();
    Report->SetBoolField(TEXT("matches"), false);
    auto Finish = [&Report](const TCHAR* Reason)
    {
        Report->SetStringField(TEXT("reason"), Reason);
        FString Text;
        FJsonSerializer::Serialize(Report, TJsonWriterFactory<>::Create(&Text));
        return Text;
    };
    if (!Source || !Target || !Source->GetHairGroupsInfo().IsValidIndex(SourceGroup) || !Target->GetHairGroupsInfo().IsValidIndex(TargetGroup))
        return Finish(TEXT("invalid_group"));
    FHairDescription A = Source->GetHairDescription(), B = Target->GetHairDescription();
    if (!A.IsValid() || !B.IsValid()) return Finish(TEXT("source_description_unavailable"));
    auto AP = A.VertexAttributes().GetAttributesRef<FVector3f>(HairAttribute::Vertex::Position);
    auto BP = B.VertexAttributes().GetAttributesRef<FVector3f>(HairAttribute::Vertex::Position);
    auto AC = A.StrandAttributes().GetAttributesRef<int32>(HairAttribute::Strand::VertexCount);
    auto BC = B.StrandAttributes().GetAttributesRef<int32>(HairAttribute::Strand::VertexCount);
    auto AG = A.StrandAttributes().GetAttributesRef<int32>(HairAttribute::Strand::GroupID);
    auto BG = B.StrandAttributes().GetAttributesRef<int32>(HairAttribute::Strand::GroupID);
    auto AD = A.StrandAttributes().GetAttributesRef<int32>(HairAttribute::Strand::Guide);
    auto BD = B.StrandAttributes().GetAttributesRef<int32>(HairAttribute::Strand::Guide);
    auto AU = A.StrandAttributes().GetAttributesRef<FVector2f>(HairAttribute::Strand::RootUV);
    auto BU = B.StrandAttributes().GetAttributesRef<FVector2f>(HairAttribute::Strand::RootUV);
    auto AW = A.VertexAttributes().GetAttributesRef<float>(HairAttribute::Vertex::Width);
    auto BW = B.VertexAttributes().GetAttributesRef<float>(HairAttribute::Vertex::Width);
    auto AX = A.VertexAttributes().GetAttributesRef<FVector3f>(HairAttribute::Vertex::Color);
    auto BX = B.VertexAttributes().GetAttributesRef<FVector3f>(HairAttribute::Vertex::Color);
    if (!AP.IsValid() || !BP.IsValid() || !AC.IsValid() || !BC.IsValid()) return Finish(TEXT("invalid_description"));
    const int32 AID = Source->GetHairGroupsInfo()[SourceGroup].GroupID, BID = Target->GetHairGroupsInfo()[TargetGroup].GroupID;
    TArray<FIntPoint> AS, BS;
    auto Select = [](const FHairDescription& Desc, const auto& Counts, const auto& Groups, int32 Group, TArray<FIntPoint>& Selected)
    {
        int32 Offset = 0;
        for (int32 I = 0; I < Desc.GetNumStrands(); ++I)
        {
            FStrandID Strand(I);
            const int32 Count = Counts[Strand];
            if (Count < 2 || Offset > Desc.GetNumVertices() - Count) return false;
            if ((Groups.IsValid() ? Groups[Strand] : 0) == Group) Selected.Emplace(I, Offset);
            Offset += Count;
        }
        return Offset == Desc.GetNumVertices();
    };
    if (!Select(A, AC, AG, AID, AS) || !Select(B, BC, BG, BID, BS)) return Finish(TEXT("invalid_counts"));
    Report->SetNumberField(TEXT("source_curves"), AS.Num());
    Report->SetNumberField(TEXT("target_curves"), BS.Num());
    if (!AS.Num() || AS.Num() != BS.Num()) return Finish(TEXT("curve_count_changed"));
    // Ten microns accommodates float/unit round trips, not artistic edits.
    constexpr float PositionTolerance = .001f;
    float MaxPositionError = 0;
    for (int32 I = 0; I < AS.Num(); ++I)
    {
        const FStrandID SA(AS[I].X), SB(BS[I].X);
        if (AC[SA] != BC[SB] || (AD.IsValid() ? AD[SA] : 0) != (BD.IsValid() ? BD[SB] : 0)) return Finish(TEXT("curve_layout_changed"));
        if (AU.IsValid() != BU.IsValid() || (AU.IsValid() && !AU[SA].Equals(BU[SB], 1e-6f))) return Finish(TEXT("root_uv_changed"));
        for (int32 J = 0; J < AC[SA]; ++J)
        {
            const FVertexID VA(AS[I].Y + J), VB(BS[I].Y + J);
            const FVector3f Delta = AP[VA] - BP[VB];
            if (AP[VA].ContainsNaN() || BP[VB].ContainsNaN() || Delta.ContainsNaN()) return Finish(TEXT("invalid_position"));
            MaxPositionError = FMath::Max(MaxPositionError, Delta.Size());
            if (bComparePositions && MaxPositionError > PositionTolerance) return Finish(TEXT("positions_changed"));
            const float WA = AW.IsValid() ? AW[VA] : .01f, WB = BW.IsValid() ? BW[VB] : .01f;
            if (!FMath::IsFinite(WA) || !FMath::IsFinite(WB) || !FMath::IsNearlyEqual(WA, WB, 1e-6f)) return Finish(TEXT("widths_changed"));
            if (AX.IsValid() != BX.IsValid() || (AX.IsValid() && !AX[VA].Equals(BX[VB], 1e-6f))) return Finish(TEXT("colors_changed"));
        }
    }
    Report->SetNumberField(TEXT("max_position_error_cm"), MaxPositionError);
    Report->SetNumberField(TEXT("position_tolerance_cm"), PositionTolerance);
    Report->SetBoolField(TEXT("matches"), true);
    return Finish(TEXT("matched_within_native_roundtrip_tolerance"));
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
    if (!InitializeGroomGroups(Groom)) return nullptr;
    if (!Groom->CacheDerivedDatas()) return nullptr;
    FAssetRegistryModule::AssetCreated(Groom);
    Groom->MarkPackageDirty();
    return Groom;
}
