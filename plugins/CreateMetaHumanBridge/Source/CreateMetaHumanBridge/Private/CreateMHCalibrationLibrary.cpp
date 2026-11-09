#include "CreateMHCalibrationLibrary.h"
#include "DNA.h"
#include "DNAAsset.h"
#include "DNAAssetUserData.h"
#include "DNAReader.h"
#include "DNAUtils.h"
#include "SharedRigRuntimeContext.h"
#include "ReferenceSkeleton.h"
#include "SkelMeshDNAUtils.h"
#include "Engine/SkeletalMesh.h"
#include "Components/SkeletalMeshComponent.h"
#include "GameFramework/Actor.h"
#include "Animation/MorphTarget.h"
#include "Animation/AnimCurveMetadata.h"
#include "Rendering/SkeletalMeshModel.h"
#include "Rendering/SkeletalMeshLODModel.h"
#include "SkeletalMeshAttributes.h"
#include "MeshDescription.h"
#include "StaticMeshOperations.h"
#include "AssetCompilingManager.h"
#include "Dom/JsonObject.h"
#include "Serialization/JsonReader.h"
#include "Serialization/JsonSerializer.h"
#include "Misc/FileHelper.h"
#include "Misc/Paths.h"
#include "Misc/ScopeExit.h"
#include "Misc/SecureHash.h"
#include "HAL/FileManager.h"
#include "dna/BinaryStreamWriter.h"
#include "Internationalization/Regex.h"
#include "UObject/StrongObjectPtr.h"

namespace
{
bool NeutralGeometryBox(USkeletalMesh* Mesh,FBox& Box,TArray<TSharedPtr<FJsonValue>>* LODReports=nullptr,FSHA1* PositionsHash=nullptr)
{
    if (!Mesh) return false;
    const auto Model=Mesh->GetImportedModel();if (!Model||Model->LODModels.IsEmpty()) return false;
    Box=FBox(ForceInit);
    const FBox Imported=Mesh->GetImportedBounds().GetBox().ExpandBy(.001);
    const FBox Extended=Mesh->GetBounds().GetBox().ExpandBy(.001);
    for (int32 LOD=0;LOD<Model->LODModels.Num();++LOD)
    {
        int32 Count=0,OutsideImported=0,OutsideExtended=0;
        for (const auto& Section:Model->LODModels[LOD].Sections) for (const auto& Vertex:Section.SoftVertices)
        {
            const FVector Point(Vertex.Position);if (Point.ContainsNaN()) return false;
            if (PositionsHash) PositionsHash->Update(reinterpret_cast<const uint8*>(&Vertex.Position),sizeof(Vertex.Position));
            Box+=Point;++Count;
            OutsideImported+=!Imported.IsInsideOrOn(Point);OutsideExtended+=!Extended.IsInsideOrOn(Point);
        }
        if (!Count) return false;
        if (LODReports)
        {
            auto Row=MakeShared<FJsonObject>();Row->SetNumberField(TEXT("lod"),LOD);Row->SetNumberField(TEXT("vertices"),Count);
            Row->SetNumberField(TEXT("outside_imported"),OutsideImported);Row->SetNumberField(TEXT("outside_extended"),OutsideExtended);LODReports->Add(MakeShared<FJsonValueObject>(Row));
        }
    }
    return Box.IsValid!=0;
}

bool ExpandNeutralBounds(USkeletalMesh* Mesh)
{
    FBox Geometry(ForceInit);if (!NeutralGeometryBox(Mesh,Geometry)) return false;
    const auto Before=Mesh->GetImportedBounds();FBox Expanded=Before.GetBox();
    // Negative artist extensions may intentionally crop the import box. Keep
    // their values, compensating in the private import box rather than silently
    // resetting the artist's properties.
    Expanded+=Geometry;Expanded+=Geometry.Min+Mesh->GetNegativeBoundsExtension();Expanded+=Geometry.Max-Mesh->GetPositiveBoundsExtension();
    if (!Expanded.Min.Equals(Before.GetBox().Min,1.e-6)||!Expanded.Max.Equals(Before.GetBox().Max,1.e-6))
    {
        FBoxSphereBounds Bounds(Expanded);Bounds.SphereRadius=FMath::Max(Bounds.SphereRadius,Before.SphereRadius);
        Mesh->SetImportedBounds(Bounds);
    }
    return true;
}

bool JSON(const FString& Path,const TSharedRef<FJsonObject>& Value)
{
    FString Text;
    return FJsonSerializer::Serialize(Value,TJsonWriterFactory<>::Create(&Text)) &&
        FFileHelper::SaveStringToFile(Text,*Path,FFileHelper::EEncodingOptions::ForceUTF8WithoutBOM);
}
TSharedPtr<IDNAReader> Read(const FString& Path)
{
    FDNAConfig Config;
    Config.CoordinateSystemTransformPolicy=ECoordinateSystemTransformPolicy::Preserve;
    Config.MaxLODPerPlatform=FPerPlatformInt(0);Config.MinLODPerPlatform=FPerPlatformInt(-1);
    return LoadDNAFromFile(Path,Config);
}
bool Points(const TSharedPtr<FJsonObject>& Row,const FString& Parent,TArray<dna::Vector3>& Values)
{
    FString Name;
    if (!Row->TryGetStringField(TEXT("file"),Name) || Name!=FPaths::GetCleanFilename(Name)) return false;
    TArray<uint8> Bytes;
    if (!FFileHelper::LoadFileToArray(Bytes,*FPaths::Combine(Parent,Name)) || !Bytes.Num() || Bytes.Num()%12) return false;
    Values.SetNumUninitialized(Bytes.Num()/12);
    static_assert(sizeof(dna::Vector3)==12);
    FMemory::Memcpy(Values.GetData(),Bytes.GetData(),Bytes.Num());
    for (const auto& P:Values) if (!FMath::IsFinite(P.x)||!FMath::IsFinite(P.y)||!FMath::IsFinite(P.z)) return false;
    return true;
}
}

bool UCreateMHCalibrationLibrary::ExportEmbeddedCalibrationReference(USkeletalMesh* SourceMesh,const FString& Directory)
{
    if (!SourceMesh || Directory.IsEmpty() || IFileManager::Get().DirectoryExists(*Directory)) return false;
    TSharedPtr<IDNAReader> Reader;
    FString Binding;
    if (const auto Data=Cast<UDNAAssetUserData>(SourceMesh->GetAssetUserDataOfClass(UDNAAssetUserData::StaticClass())))
    {
        if (Data->DNAAsset) {Reader=Data->DNAAsset->GetDNAReader();Binding=Data->DNAAsset->GetPathName();}
    }
    else if (const auto Legacy=Cast<UDNAAsset>(SourceMesh->GetAssetUserDataOfClass(UDNAAsset::StaticClass())))
    {Reader=Legacy->GetDNAReader();Binding=Legacy->GetPathName();}
    if (!Reader || !Reader->Unwrap() || !Reader->GetMeshCount() || !IFileManager::Get().MakeDirectory(*Directory,true)) return false;
    const FString DNAFile=FPaths::Combine(Directory,TEXT("source.dna"));
    const FString Reference=FPaths::Combine(Directory,TEXT("reference.json"));
    SaveDNAToFile(Reader.Get(),EDNADataLayer::All,DNAFile);
    if (IFileManager::Get().FileSize(*DNAFile)<=0 || !ExportCalibrationReference(DNAFile,Reference)) return false;
    auto Report=MakeShared<FJsonObject>();
    Report->SetStringField(TEXT("status"),TEXT("complete"));Report->SetStringField(TEXT("source_mesh"),SourceMesh->GetPathName());
    Report->SetStringField(TEXT("dna_binding"),Binding);Report->SetStringField(TEXT("dna"),TEXT("source.dna"));
    Report->SetStringField(TEXT("reference"),TEXT("reference.json"));Report->SetNumberField(TEXT("source_assets_saved"),0);
    return JSON(FPaths::Combine(Directory,TEXT("embedded-reference.json")),Report);
}

FString UCreateMHCalibrationLibrary::InspectGeometryBounds(USkeletalMesh* Mesh)
{
    if (!IsInGameThread()||!Mesh) return FString();
    FAssetCompilingManager::Get().FinishAllCompilation();FBox Geometry(ForceInit);TArray<TSharedPtr<FJsonValue>> LODs;FSHA1 PositionsHash;
    if (!NeutralGeometryBox(Mesh,Geometry,&LODs,&PositionsHash)) return FString();
    auto Report=MakeShared<FJsonObject>();Report->SetStringField(TEXT("status"),TEXT("complete"));Report->SetStringField(TEXT("mesh"),Mesh->GetPathName());Report->SetArrayField(TEXT("lods"),LODs);Report->SetBoolField(TEXT("neutral_only"),true);
    auto Vector=[](const FVector& Value){TArray<TSharedPtr<FJsonValue>> Result;for (int32 I=0;I<3;++I) Result.Add(MakeShared<FJsonValueNumber>(Value[I]));return Result;};
    Report->SetArrayField(TEXT("geometry_min"),Vector(Geometry.Min));Report->SetArrayField(TEXT("geometry_max"),Vector(Geometry.Max));
    Report->SetArrayField(TEXT("imported_min"),Vector(Mesh->GetImportedBounds().GetBox().Min));Report->SetArrayField(TEXT("imported_max"),Vector(Mesh->GetImportedBounds().GetBox().Max));
    Report->SetArrayField(TEXT("extended_min"),Vector(Mesh->GetBounds().GetBox().Min));Report->SetArrayField(TEXT("extended_max"),Vector(Mesh->GetBounds().GetBox().Max));
    PositionsHash.Final();uint8 Digest[FSHA1::DigestSize];PositionsHash.GetHash(Digest);Report->SetStringField(TEXT("position_sha1"),BytesToHex(Digest,UE_ARRAY_COUNT(Digest)));
    FString Text;FJsonSerializer::Serialize(Report,TJsonWriterFactory<>::Create(&Text));return Text;
}

bool UCreateMHCalibrationLibrary::RefreshMeshCopyBounds(USkeletalMesh* SourceMesh,USkeletalMesh* TargetMesh)
{
    if (!IsInGameThread()||!SourceMesh||!TargetMesh||SourceMesh==TargetMesh||SourceMesh->GetOutermost()==TargetMesh->GetOutermost()||!TargetMesh->GetPathName().StartsWith(TEXT("/Game/"))||SourceMesh->GetSkeleton()!=TargetMesh->GetSkeleton()||SourceMesh->GetLODNum()!=TargetMesh->GetLODNum()) return false;
    FAssetCompilingManager::Get().FinishAllCompilation();TargetMesh->Modify();
    if (!ExpandNeutralBounds(TargetMesh)) return false;
    TargetMesh->MarkPackageDirty();return true;
}

bool UCreateMHCalibrationLibrary::RefreshPreviewComponentBounds(USkeletalMeshComponent* Component)
{
    if (!IsInGameThread()||!Component||Component->IsTemplate()||!Component->GetWorld()||!Component->GetOwner()||!Component->GetOwner()->HasAnyFlags(RF_Transient)) return false;
    FAssetCompilingManager::Get().FinishAllCompilation();Component->InvalidateCachedBounds();Component->UpdateBounds();Component->MarkRenderTransformDirty();return true;
}

bool UCreateMHCalibrationLibrary::ExportCalibrationReference(const FString& DNAFile,const FString& OutputFile)
{
    if (IFileManager::Get().FileExists(*OutputFile)) return false;
    auto Reader=Read(DNAFile);
    if (!Reader || !Reader->Unwrap()) return false;
    auto Root=MakeShared<FJsonObject>();Root->SetStringField(TEXT("schema"),TEXT("CM_DNA_CALIBRATION_REFERENCE_V1"));
    Root->SetStringField(TEXT("coordinates"),TEXT("PRESERVED_DNA"));Root->SetNumberField(TEXT("lod_count"),Reader->GetLODCount());
    const auto System=Reader->GetCoordinateSystem();
    auto Direction=[](EDirection Value)->FString
    {
        switch (Value)
        {
        case EDirection::Left:return TEXT("Left");case EDirection::Right:return TEXT("Right");
        case EDirection::Up:return TEXT("Up");case EDirection::Down:return TEXT("Down");
        case EDirection::Front:return TEXT("Front");case EDirection::Back:return TEXT("Back");
        default:return TEXT("Unknown");
        }
    };
    auto Coordinates=MakeShared<FJsonObject>();Coordinates->SetStringField(TEXT("x"),Direction(System.XAxis));Coordinates->SetStringField(TEXT("y"),Direction(System.YAxis));Coordinates->SetStringField(TEXT("z"),Direction(System.ZAxis));Root->SetObjectField(TEXT("coordinate_system"),Coordinates);
    TArray<TSharedPtr<FJsonValue>> Meshes;
    for (uint16 M=0;M<Reader->GetMeshCount();++M)
    {
        auto Row=MakeShared<FJsonObject>();Row->SetStringField(TEXT("name"),Reader->GetMeshName(M));Row->SetNumberField(TEXT("index"),M);
        TArray<dna::Vector3> Positions;
        for (uint32 V=0;V<Reader->GetVertexPositionCount(M);++V)
        {
            const FVector P=Reader->GetVertexPosition(M,V);Positions.Add({float(P.X),float(P.Y),float(P.Z)});
        }
        TArray<uint8> Bytes;Bytes.SetNumUninitialized(Positions.Num()*12);
        FMemory::Memcpy(Bytes.GetData(),Positions.GetData(),Bytes.Num());
        const FString Filename=OutputFile+FString::Printf(TEXT(".positions_%d.f32"),M);
        if (!FFileHelper::SaveArrayToFile(Bytes,*Filename)) return false;
        Row->SetNumberField(TEXT("vertex_count"),Positions.Num());Row->SetStringField(TEXT("positions_file"),FPaths::GetCleanFilename(Filename));
        TArray<TSharedPtr<FJsonValue>> LODs;
        for (uint16 L=0;L<Reader->GetLODCount();++L) for (auto Active:Reader->GetMeshIndicesForLOD(L)) if (Active==M) LODs.Add(MakeShared<FJsonValueNumber>(L));
        Row->SetArrayField(TEXT("lods"),LODs);
        // Retain the DNA layout domain at UV seams; never merge layouts by
        // position or assume two face LODs have matching vertex indices.
        TArray<int32> LayoutPositions;TArray<float> UV;TArray<uint32> Triangles;
        for (uint32 V=0;V<Reader->GetVertexLayoutCount(M);++V)
        {
            const auto Layout=Reader->GetVertexLayout(M,V);
            if (Layout.Position<0||Layout.Position>=Positions.Num()||Layout.TextureCoordinate<0||uint32(Layout.TextureCoordinate)>=Reader->GetVertexTextureCoordinateCount(M)) return false;
            const auto Coordinate=Reader->GetVertexTextureCoordinate(M,Layout.TextureCoordinate);
            LayoutPositions.Add(Layout.Position);UV.Add(Coordinate.U);UV.Add(Coordinate.V);
        }
        for (uint32 F=0;F<Reader->GetFaceCount(M);++F)
        {
            const auto Face=Reader->GetFaceVertexLayoutIndices(M,F);if (Face.Num()<3) return false;
            for (auto V:Face) if (V>=uint32(LayoutPositions.Num())) return false;
            for (int32 I=1;I+1<Face.Num();++I) {Triangles.Add(Face[0]);Triangles.Add(Face[I]);Triangles.Add(Face[I+1]);}
        }
        auto SaveBuffer=[&](const void* Data,int64 Size,const FString& Suffix,const TCHAR* Field)
        {
            TArray<uint8> Buffer;Buffer.SetNumUninitialized(Size);if (Size) FMemory::Memcpy(Buffer.GetData(),Data,Size);
            const FString Path=OutputFile+FString::Printf(TEXT(".mesh_%d"),M)+Suffix;
            if (!FFileHelper::SaveArrayToFile(Buffer,*Path)) return false;
            Row->SetStringField(Field,FPaths::GetCleanFilename(Path));return true;
        };
        if (!SaveBuffer(LayoutPositions.GetData(),LayoutPositions.Num()*sizeof(int32),TEXT(".layout.i32"),TEXT("layout_positions_file"))||
            !SaveBuffer(UV.GetData(),UV.Num()*sizeof(float),TEXT(".uv.f32"),TEXT("layout_uv_file"))||
            !SaveBuffer(Triangles.GetData(),Triangles.Num()*sizeof(uint32),TEXT(".triangles.u32"),TEXT("layout_triangles_file"))) return false;
        Row->SetStringField(TEXT("topology_space"),TEXT("DNA_VERTEX_LAYOUT"));
        TArray<TSharedPtr<FJsonValue>> Targets;
        for (uint16 T=0;T<Reader->GetBlendShapeTargetCount(M);++T)
        {
            auto Item=MakeShared<FJsonObject>();Item->SetNumberField(TEXT("index"),T);
            Item->SetStringField(TEXT("channel"),Reader->GetBlendShapeChannelName(Reader->GetBlendShapeChannelIndex(M,T)));
            if (Reader->GetBlendShapeChannelName(Reader->GetBlendShapeChannelIndex(M,T)).StartsWith(TEXT("CM_")))
            {
                const auto Indices=Reader->GetBlendShapeTargetVertexIndices(M,T);TArray<dna::Vector3> Deltas;
                for (uint32 I=0;I<Reader->GetBlendShapeTargetDeltaCount(M,T);++I) {const auto D=Reader->GetBlendShapeTargetDelta(M,T,I);Deltas.Add({float(D.X),float(D.Y),float(D.Z)});}
                if (Indices.Num()!=Deltas.Num()) return false;
                auto SaveTarget=[&](const void* Data,int64 Size,const TCHAR* Suffix,const TCHAR* Field)
                {
                    TArray<uint8> Buffer;Buffer.SetNumUninitialized(Size);if (Size) FMemory::Memcpy(Buffer.GetData(),Data,Size);
                    const FString Path=OutputFile+FString::Printf(TEXT(".target_%d_%d"),M,T)+Suffix;
                    if (!FFileHelper::SaveArrayToFile(Buffer,*Path)) return false;
                    Item->SetStringField(Field,FPaths::GetCleanFilename(Path));return true;
                };
                if (!SaveTarget(Indices.GetData(),Indices.Num()*sizeof(uint32),TEXT(".indices.u32"),TEXT("indices_file"))||
                    !SaveTarget(Deltas.GetData(),Deltas.Num()*sizeof(dna::Vector3),TEXT(".deltas.f32"),TEXT("deltas_file"))) return false;
            }
            Item->SetNumberField(TEXT("vertices"),Reader->GetBlendShapeTargetVertexIndices(M,T).Num());Targets.Add(MakeShared<FJsonValueObject>(Item));
        }
        Row->SetArrayField(TEXT("targets"),Targets);Meshes.Add(MakeShared<FJsonValueObject>(Row));
    }
    Root->SetArrayField(TEXT("meshes"),Meshes);return JSON(OutputFile,Root);
}

bool UCreateMHCalibrationLibrary::WriteCalibrationDNA(const FString& DNAFile,const FString& RequestFile,const FString& OutputFile)
{
    if (FPaths::ConvertRelativePathToFull(DNAFile)==FPaths::ConvertRelativePathToFull(OutputFile) || IFileManager::Get().FileExists(*OutputFile)) return false;
    FString Text;TSharedPtr<FJsonObject> Request;
    if (!FFileHelper::LoadFileToString(Text,*RequestFile)||!FJsonSerializer::Deserialize(TJsonReaderFactory<>::Create(Text),Request)||!Request) return false;
    FString Schema;double Bound=0;
    if (!Request->TryGetStringField(TEXT("schema"),Schema)||Schema!=TEXT("CM_DNA_CALIBRATION_V1")||!Request->TryGetNumberField(TEXT("maximum_offset"),Bound)||!FMath::IsFinite(Bound)||Bound<=0) return false;
    auto Reader=Read(DNAFile);if (!Reader || !Reader->Unwrap()) return false;
    TMap<FString,uint16> Meshes;for (uint16 M=0;M<Reader->GetMeshCount();++M) Meshes.Add(Reader->GetMeshName(M),M);
    struct FNeutral {uint16 Mesh;TArray<dna::Vector3> Points;};
    struct FCorrective {uint16 Mesh,Target;TArray<uint32> Indices;TArray<dna::Vector3> Deltas;};
    TArray<FNeutral> Neutrals;TArray<FCorrective> Correctives;TSet<FString> Seen;
    TArray<dna::Vector3> JointTranslations;
    FString JointLedger;
    const FString Parent=FPaths::GetPath(RequestFile);
    const TArray<TSharedPtr<FJsonValue>>* Rows=nullptr;
    if (!Request->TryGetArrayField(TEXT("neutral"),Rows)) return false;
    for (const auto& Value:*Rows)
    {
        auto Row=Value->AsObject();FString Name;if (!Row || !Row->TryGetStringField(TEXT("mesh"),Name)||!Meshes.Contains(Name)||Seen.Contains(Name)) return false;Seen.Add(Name);
        FNeutral Patch;Patch.Mesh=Meshes[Name];
        if (!Points(Row,Parent,Patch.Points)||Patch.Points.Num()!=Reader->GetVertexPositionCount(Patch.Mesh)) return false;
        for (int32 V=0;V<Patch.Points.Num();++V)
        {
            const auto& P=Patch.Points[V];if (FVector::Distance(FVector(P.x,P.y,P.z),Reader->GetVertexPosition(Patch.Mesh,V))>Bound) return false;
        }
        Neutrals.Add(MoveTemp(Patch));
    }
    Seen.Empty();if (!Request->TryGetArrayField(TEXT("correctives"),Rows)) return false;
    for (const auto& Value:*Rows)
    {
        auto Row=Value->AsObject();FString Name,Channel;
        if (!Row||!Row->TryGetStringField(TEXT("mesh"),Name)||!Row->TryGetStringField(TEXT("channel"),Channel)||!Meshes.Contains(Name)||Seen.Contains(Name+TEXT("/")+Channel)) return false;
        Seen.Add(Name+TEXT("/")+Channel);const uint16 M=Meshes[Name];int32 Target=INDEX_NONE;
        for (uint16 T=0;T<Reader->GetBlendShapeTargetCount(M);++T) if (Reader->GetBlendShapeChannelName(Reader->GetBlendShapeChannelIndex(M,T))==Channel) {if (Target!=INDEX_NONE) return false;Target=T;}
        if (Target==INDEX_NONE) return false;
        TArray<dna::Vector3> Offsets;if (!Points(Row,Parent,Offsets)) return false;
        const TArray<TSharedPtr<FJsonValue>>* Indices=nullptr;if (!Row->TryGetArrayField(TEXT("indices"),Indices)||Indices->Num()!=Offsets.Num()) return false;
        TMap<uint32,dna::Vector3> Deltas;const auto Original=Reader->GetBlendShapeTargetVertexIndices(M,Target);
        for (int32 I=0;I<Original.Num();++I) {const FVector D=Reader->GetBlendShapeTargetDelta(M,Target,I);Deltas.Add(Original[I],{float(D.X),float(D.Y),float(D.Z)});}
        TSet<uint32> Vertices;
        for (int32 I=0;I<Offsets.Num();++I)
        {
            double Number=0;if (!(*Indices)[I]->TryGetNumber(Number)||!FMath::IsFinite(Number)||Number<0||Number>=Reader->GetVertexPositionCount(M)||Number!=FMath::FloorToDouble(Number)) return false;
            const uint32 V=uint32(Number);if (Vertices.Contains(V)) return false;Vertices.Add(V);const auto D=Offsets[I];
            if (FVector(D.x,D.y,D.z).Size()>Bound) return false;
            auto& Total=Deltas.FindOrAdd(V,dna::Vector3{0,0,0});Total+=D;
        }
        FCorrective Patch;Patch.Mesh=M;Patch.Target=Target;Deltas.GetKeys(Patch.Indices);Patch.Indices.Sort();
        for (const uint32 V:Patch.Indices) Patch.Deltas.Add(Deltas[V]);Correctives.Add(MoveTemp(Patch));
    }
    if (Request->TryGetArrayField(TEXT("neutral_joints"),Rows)&&Rows->Num())
    {
        auto Legacy=LoadDNAFromFile(DNAFile,FDNAConfig::Legacy(ECoordinateSystemTransformPolicy::Transform));
        if (!Legacy||Legacy->GetJointCount()!=Reader->GetJointCount()) return false;
        // Derive and verify the preserved-DNA -> native local translation
        // frame from this rig, rather than guessing a Maya/UE axis swizzle.
        FMatrix Gram=FMatrix::Identity,Cross=FMatrix::Identity;
        for (int32 I=0;I<3;++I) for (int32 J=0;J<3;++J) Gram.M[I][J]=Cross.M[I][J]=0;
        for (uint16 J=0;J<Reader->GetJointCount();++J)
        {
            if (Reader->GetJointName(J)!=Legacy->GetJointName(J)) return false;
            const FVector A=Reader->GetNeutralJointTranslation(J),B=Legacy->GetNeutralJointTranslation(J);
            for (int32 I=0;I<3;++I) for (int32 K=0;K<3;++K) {Gram.M[I][K]+=A[I]*A[K];Cross.M[I][K]+=A[I]*B[K];}
            JointTranslations.Add({float(A.X),float(A.Y),float(A.Z)});
        }
        if (FMath::Abs(Gram.Determinant())<1.e-12) return false;
        const FMatrix Frame=Gram.Inverse()*Cross,Inverse=Frame.Inverse();
        for (uint16 J=0;J<Reader->GetJointCount();++J)
            if (FVector::Distance(Frame.TransformVector(Reader->GetNeutralJointTranslation(J)),Legacy->GetNeutralJointTranslation(J))>.001) return false;
        Seen.Empty();
        for (const auto& Value:*Rows)
        {
            auto Row=Value->AsObject();FString Name;const TArray<TSharedPtr<FJsonValue>>* Translation=nullptr;
            if (!Row||!Row->TryGetStringField(TEXT("joint"),Name)||Seen.Contains(Name)||!Row->TryGetArrayField(TEXT("legacy_local_translation"),Translation)||Translation->Num()!=3) return false;
            Seen.Add(Name);int32 Found=INDEX_NONE;
            for (uint16 J=0;J<Reader->GetJointCount();++J) if (Reader->GetJointName(J)==Name) Found=J;
            if (Found==INDEX_NONE) return false;FVector Native;
            for (int32 I=0;I<3;++I) {double Number=0;if (!(*Translation)[I]->TryGetNumber(Number)||!FMath::IsFinite(Number)) return false;Native[I]=Number;}
            if (FVector::Distance(Native,Legacy->GetNeutralJointTranslation(Found))>Bound) return false;
            const FVector Raw=Inverse.TransformVector(Native);JointTranslations[Found]={float(Raw.X),float(Raw.Y),float(Raw.Z)};
        }
        TArray<TSharedPtr<FJsonValue>> Names;for (const auto& Name:Seen) Names.Add(MakeShared<FJsonValueString>(Name));
        if (!FJsonSerializer::Serialize(Names,TJsonWriterFactory<>::Create(&JointLedger))) return false;
    }
    auto Stream=dna::FileStream::create(TCHAR_TO_UTF8(*OutputFile),trio::AccessMode::Write,trio::OpenMode::Binary);
    if (!Stream) return false;
    auto Writer=dna::BinaryStreamWriter::create(Stream);
    ON_SCOPE_EXIT {if (Writer) dna::BinaryStreamWriter::destroy(Writer);dna::FileStream::destroy(Stream);};
    if (!Writer) return false;
    Writer->setFrom(Reader->Unwrap());
    if (JointTranslations.Num()) Writer->setNeutralJointTranslations(JointTranslations.GetData(),JointTranslations.Num());
    if (!JointLedger.IsEmpty()) Writer->setMetaData("CreateMetaHuman.CalibratedJoints",TCHAR_TO_UTF8(*JointLedger));
    for (const auto& Patch:Neutrals) Writer->setVertexPositions(Patch.Mesh,Patch.Points.GetData(),Patch.Points.Num());
    for (const auto& Patch:Correctives) {Writer->setBlendShapeTargetVertexIndices(Patch.Mesh,Patch.Target,Patch.Indices.GetData(),Patch.Indices.Num());Writer->setBlendShapeTargetDeltas(Patch.Mesh,Patch.Target,Patch.Deltas.GetData(),Patch.Deltas.Num());}
    Writer->write();return dna::Status::isOk();
}

bool UCreateMHCalibrationLibrary::ApplyCalibrationToMeshCopy(USkeletalMesh* SourceMesh,USkeletalMesh* TargetMesh,const FString& DNAFile)
{
    if (!IsInGameThread()||!SourceMesh||!TargetMesh||SourceMesh==TargetMesh||SourceMesh->GetOutermost()==TargetMesh->GetOutermost()||!TargetMesh->GetPathName().StartsWith(TEXT("/Game/"))||TargetMesh->GetSkeleton()!=SourceMesh->GetSkeleton()||TargetMesh->GetLODNum()!=SourceMesh->GetLODNum()) return false;
    auto Previous=USkelMeshDNAUtils::GetDNAReader(TargetMesh);if (!Previous) return false;
    auto CalibrationConfig=Previous->GetConfig();CalibrationConfig.Layers|=static_cast<int32>(EDNADataLayer::RBFBehavior);
    auto Calibrated=LoadDNAFromFile(DNAFile,CalibrationConfig);if (!Calibrated||Calibrated->GetJointCount()!=Previous->GetJointCount()||Calibrated->GetMeshCount()!=Previous->GetMeshCount()) return false;
    for (uint16 M=0;M<Previous->GetMeshCount();++M) if (Calibrated->GetMeshName(M)!=Previous->GetMeshName(M)||Calibrated->GetVertexPositionCount(M)!=Previous->GetVertexPositionCount(M)) return false;
    TUniquePtr<FDNAToSkelMeshMap> Mapping(USkelMeshDNAUtils::CreateMapForUpdatingNeutralMesh(TargetMesh));if (!Mapping) return false;
    UDNA* DNA=NewObject<UDNA>(TargetMesh,NAME_None,RF_Transactional);
    if (auto OriginalData=Cast<UDNAAssetUserData>(TargetMesh->GetAssetUserDataOfClass(UDNAAssetUserData::StaticClass())))
    {
        if (OriginalData->DNAAsset)
        {
            DNA->DNAConfig=OriginalData->DNAAsset->DNAConfig;
            DNA->RigLogicConfiguration=OriginalData->DNAAsset->RigLogicConfiguration;
            DNA->MetaData=OriginalData->DNAAsset->MetaData;
            DNA->bKeepDNAAfterInitialization=OriginalData->DNAAsset->bKeepDNAAfterInitialization;
            DNA->bUseOptimizedCooking=OriginalData->DNAAsset->bUseOptimizedCooking;
        }
    }
    if (Calibrated->GetRBFSolverCount()>Previous->GetRBFSolverCount()) DNA->RigLogicConfiguration.LoadRBFBehavior=true;
    DNA->SetDNAReader(Calibrated,EDNACopyPolicy::Copy,ERigLogicInitPolicy::Defer);
    if (Previous->IsLegacyWrapped()) DNA->RestoreLegacyUEMHCCompatibility();else DNA->SetDNAReader(Calibrated,EDNACopyPolicy::Copy,ERigLogicInitPolicy::Initialize);
    auto Data=NewObject<UDNAAssetUserData>(TargetMesh,NAME_None,RF_Transactional);Data->DNAAsset=DNA;
    TargetMesh->Modify();TargetMesh->RemoveUserDataOfClass(UDNAAssetUserData::StaticClass());TargetMesh->RemoveUserDataOfClass(UDNAAsset::StaticClass());TargetMesh->AddAssetUserData(Data);
    auto Reader=DNA->GetDNAReader();if (!Reader) return false;
    auto SourceData=Cast<UDNAAssetUserData>(SourceMesh->GetAssetUserDataOfClass(UDNAAssetUserData::StaticClass()));
    auto OldContext=SourceData&&SourceData->DNAAsset?SourceData->DNAAsset->GetRigRuntimeContext():nullptr;
    auto NewContext=DNA->GetRigRuntimeContext();
    const FString JointLedger=Calibrated->GetMetaDataValue(TEXT("CreateMetaHuman.CalibratedJoints"));
    if (!JointLedger.IsEmpty())
    {
        if (!OldContext||!NewContext||!OldContext->DNAReader||!NewContext->DNAReader) return false;
        TArray<TSharedPtr<FJsonValue>> Named;TSet<FString> Requested;
        if (!FJsonSerializer::Deserialize(TJsonReaderFactory<>::Create(JointLedger),Named)) return false;
        for (const auto& Value:Named) {FString Name;if (!Value->TryGetString(Name)||Requested.Contains(Name)) return false;Requested.Add(Name);}
        const auto Old=OldContext->DNAReader,New=NewContext->DNAReader;
        if (Old->GetJointCount()!=New->GetJointCount()) {UE_LOG(LogTemp,Error,TEXT("Calibration joint counts differ"));return false;}
        const auto OldValues=OldContext->RigLogic->GetNeutralJointValues(),NewValues=NewContext->RigLogic->GetNeutralJointValues();
        if (OldValues.Num()!=Old->GetJointCount()*10||NewValues.Num()!=New->GetJointCount()*10) return false;
        TMap<int32,FVector> Changed;
        for (uint16 J=0;J<Old->GetJointCount();++J)
        {
            if (Old->GetJointName(J)!=New->GetJointName(J)) {UE_LOG(LogTemp,Error,TEXT("Calibration joint names differ at %d"),J);return false;}
            if (!Requested.Contains(Old->GetJointName(J))) continue;
            const int32 Offset=J*10;
            const FQuat BeforeRotation(OldValues[Offset+3],OldValues[Offset+4],OldValues[Offset+5],OldValues[Offset+6]),AfterRotation(NewValues[Offset+3],NewValues[Offset+4],NewValues[Offset+5],NewValues[Offset+6]);
            if (BeforeRotation.AngularDistance(AfterRotation)>.0001) {UE_LOG(LogTemp,Error,TEXT("Calibration changed an unsupported joint rotation: %s"),*Old->GetJointName(J));return false;}
            const FVector Before(OldValues[Offset],OldValues[Offset+1],OldValues[Offset+2]),After(NewValues[Offset],NewValues[Offset+1],NewValues[Offset+2]);
            if (FVector::Distance(Before,After)<1.e-5) continue;
            const int32 Bone=TargetMesh->GetRefSkeleton().FindBoneIndex(FName(*New->GetJointName(J)));
            if (Bone==INDEX_NONE) {UE_LOG(LogTemp,Error,TEXT("Calibration joint has no skeletal bone: %s"),*New->GetJointName(J));return false;}
            // Preserve any assembled bind adjustment just as we preserve the
            // assembled morph baseline. Apply only the DNA's requested delta.
            Changed.Add(Bone,TargetMesh->GetRefSkeleton().GetRefBonePose()[Bone].GetTranslation()+(After-Before));
        }
        if (Changed.Num())
        {
            // Only named, changed joint translations are updated. Preserve
            // every other local bind transform and the shared skeleton asset.
            {FReferenceSkeletonModifier Modifier(TargetMesh->GetRefSkeleton(),TargetMesh->GetSkeleton());
             for (const auto& Pair:Changed) {FTransform Pose=TargetMesh->GetRefSkeleton().GetRefBonePose()[Pair.Key];Pose.SetTranslation(Pair.Value);Modifier.UpdateRefPoseTransform(Pair.Key,Pose);}}
            TargetMesh->GetRefBasesInvMatrix().Reset();TargetMesh->CalculateInvRefMatrices();
        }
    }
    USkelMeshDNAUtils::UpdateBaseMesh(TargetMesh,Reader.Get(),Mapping.Get(),ELodUpdateOption::All);
    auto SourceReader=USkelMeshDNAUtils::GetDNAReader(SourceMesh);
    if (!SourceReader || SourceReader->GetMeshCount()!=Reader->GetMeshCount()) return false;
    TSet<int32> ChangedMeshes;
    for (uint16 M=0;M<Reader->GetMeshCount();++M)
    {
        if (SourceReader->GetMeshName(M)!=Reader->GetMeshName(M)||SourceReader->GetVertexPositionCount(M)!=Reader->GetVertexPositionCount(M)) return false;
        for (uint32 V=0;V<Reader->GetVertexPositionCount(M);++V)
            if (Reader->GetVertexPosition(M,V)!=SourceReader->GetVertexPosition(M,V)) {ChangedMeshes.Add(M);break;}
    }
    // Updating LODModel/render buffers alone is transient: a later editor
    // rebuild reads the old neutral mesh description. Persist positions only
    // for DNA meshes explicitly different from the source rig.
    auto Imported=TargetMesh->GetImportedModel();
    for (int32 LOD=0;LOD<Imported->LODModels.Num();++LOD)
    {
        FMeshDescription* Description=TargetMesh->GetMeshDescription(LOD);if (!Description) return false;
        auto& Model=Imported->LODModels[LOD];TArray<FSoftSkinVertex> Positions;Model.GetVertices(Positions);
        FSkeletalMeshAttributes Attributes(*Description);auto VertexPositions=Attributes.GetVertexPositions();
        TMap<int32,FVector3d> Sums;TMap<int32,int32> Counts;
        for (int32 V=0;V<Positions.Num();++V)
        {
            if (!Mapping->ImportVtxToDNAMeshIndex.IsValidIndex(LOD)||!Mapping->ImportVtxToDNAMeshIndex[LOD].IsValidIndex(V)) return false;
            if (!ChangedMeshes.Contains(Mapping->ImportVtxToDNAMeshIndex[LOD][V])) continue;
            if (!Model.MeshToImportVertexMap.IsValidIndex(V)) return false;
            const int32 Import=Model.MeshToImportVertexMap[V];if (!Description->IsVertexValid(FVertexID(Import))) return false;
            Sums.FindOrAdd(Import,FVector3d::ZeroVector)+=FVector3d(Positions[V].Position);++Counts.FindOrAdd(Import,0);
        }
        for (const auto& Pair:Sums) VertexPositions.Set(FVertexID(Pair.Key),FVector3f(Pair.Value/double(Counts[Pair.Key])));
        if (Sums.Num())
        {
            // New globe positions need new neutral tangent frames. Compute
            // on a private description copy, then copy back only instances
            // belonging to changed geometry. Artist normals elsewhere survive.
            FMeshDescription Recomputed=*Description;
            FStaticMeshOperations::ComputeTriangleTangentsAndNormals(Recomputed);
            FStaticMeshOperations::ComputeTangentsAndNormals(Recomputed,EComputeNTBsFlags::Normals|EComputeNTBsFlags::Tangents|EComputeNTBsFlags::UseMikkTSpace|EComputeNTBsFlags::BlendOverlappingNormals|EComputeNTBsFlags::IgnoreDegenerateTriangles);
            FSkeletalMeshAttributes Updated(Recomputed);auto Normals=Attributes.GetVertexInstanceNormals();auto Tangents=Attributes.GetVertexInstanceTangents();auto Signs=Attributes.GetVertexInstanceBinormalSigns();
            for (const auto& Pair:Sums) for (const FVertexInstanceID VI:Description->GetVertexVertexInstanceIDs(FVertexID(Pair.Key)))
            {
                Normals.Set(VI,Updated.GetVertexInstanceNormals().Get(VI));Tangents.Set(VI,Updated.GetVertexInstanceTangents().Get(VI));Signs.Set(VI,Updated.GetVertexInstanceBinormalSigns().Get(VI));
            }
            TargetMesh->CommitMeshDescription(LOD);
        }
    }
    // Do not regenerate assembled morphs from exported DNA: the native build
    // can carry additional adjustments that are absent from that export.
    USkelMeshDNAUtils::RebuildRenderData(TargetMesh);FAssetCompilingManager::Get().FinishAllCompilation();
    if (!ExpandNeutralBounds(TargetMesh)) return false;
    TargetMesh->MarkPackageDirty();return true;
}

bool UCreateMHCalibrationLibrary::RetireGeneratedMorphs(USkeletalMesh* SourceMesh,USkeletalMesh* TargetMesh,const TArray<FName>& MorphNames)
{
    if (!IsInGameThread() || !SourceMesh || !TargetMesh || SourceMesh==TargetMesh ||
        SourceMesh->GetOutermost()==TargetMesh->GetOutermost() || !TargetMesh->GetPathName().StartsWith(TEXT("/Game/")) ||
        SourceMesh->GetSkeleton()!=TargetMesh->GetSkeleton() || SourceMesh->GetLODNum()!=TargetMesh->GetLODNum() ||
        MorphNames.IsEmpty() || MorphNames.Num()>2048) return false;
    const FRegexPattern Pattern(TEXT("^eye(Left|Right)_lod[0-7]_mesh__CM_[A-Za-z0-9_]+$"));
    TSet<FName> Selected;TArray<TStrongObjectPtr<UMorphTarget>> Removed;
    for (const auto Name:MorphNames)
    {
        FRegexMatcher Match(Pattern,Name.ToString());
        auto Original=SourceMesh->FindMorphTarget(Name);auto Copy=TargetMesh->FindMorphTarget(Name);
        if (!Match.FindNext() || Selected.Contains(Name) || !Original || !Copy || Copy==Original || Copy->GetOuter()!=TargetMesh || Copy->IsRooted()) return false;
        Selected.Add(Name);Removed.Emplace(Copy);
    }
    for (int32 LOD=0;LOD<TargetMesh->GetLODNum();++LOD) if (!TargetMesh->HasMeshDescription(LOD)) return false;
    TargetMesh->Modify();
    for (const auto& OwnedMorph:Removed) {auto Morph=OwnedMorph.Get();Morph->Modify();TargetMesh->UnregisterMorphTarget(Morph,false);}
    // Remove every import attribute before rebuilding. The stock per-target
    // editor method rebuilds while unregistering each morph; hundreds of
    // generated RBF targets otherwise cause hundreds of redundant rebuilds.
    for (int32 LOD=0;LOD<TargetMesh->GetLODNum();++LOD)
    {
        auto Description=TargetMesh->GetMeshDescription(LOD);if (!Description) return false;
        FSkeletalMeshAttributes Attributes(*Description);const auto Existing=Attributes.GetMorphTargetNames();bool Changed=false;
        for (const auto Name:Selected) if (Existing.Contains(Name))
        {
            if (!Changed) TargetMesh->ModifyMeshDescription(LOD);
            Attributes.UnregisterMorphTargetAttribute(Name);Changed=true;
        }
        if (Changed)
        {
            USkeletalMesh::FCommitMeshDescriptionParams Params;
            Params.bUpdateMorphTargets=false;Params.bUpdateSkinWeightProfiles=false;Params.bUpdateVertexAttributes=false;Params.bUpdateVertexColors=false;
            if (!TargetMesh->CommitMeshDescription(LOD,Params)) return false;
        }
    }
    TargetMesh->InitMorphTargets();USkelMeshDNAUtils::RebuildRenderData(TargetMesh);FAssetCompilingManager::Get().FinishAllCompilation();
    for (const auto Name:Selected) if (TargetMesh->FindMorphTarget(Name) || !SourceMesh->FindMorphTarget(Name)) return false;
    for (const auto& OwnedMorph:Removed)
    {
        auto Morph=OwnedMorph.Get();
        Morph->ClearFlags(RF_Standalone);Morph->Rename(nullptr,GetTransientPackage(),REN_DoNotDirty|REN_DontCreateRedirectors);Morph->MarkAsGarbage();
    }
    TargetMesh->MarkPackageDirty();return true;
}

bool UCreateMHCalibrationLibrary::ApplyCompiledCorrectiveOffsets(USkeletalMesh* SourceMesh,USkeletalMesh* TargetMesh,const FString& RequestFile)
{
    if (!IsInGameThread()||!SourceMesh||!TargetMesh||SourceMesh==TargetMesh||SourceMesh->GetOutermost()==TargetMesh->GetOutermost()||!TargetMesh->GetPathName().StartsWith(TEXT("/Game/"))||TargetMesh->GetSkeleton()!=SourceMesh->GetSkeleton()) return false;
    FString Text;TSharedPtr<FJsonObject> Request;
    if (!FFileHelper::LoadFileToString(Text,*RequestFile)||!FJsonSerializer::Deserialize(TJsonReaderFactory<>::Create(Text),Request)||!Request) return false;
    FString Schema;double Bound=0;const TArray<TSharedPtr<FJsonValue>>* Rows=nullptr;
    if (!Request->TryGetStringField(TEXT("schema"),Schema)||Schema!=TEXT("CM_COMPILED_CORRECTIVE_V1")||!Request->TryGetNumberField(TEXT("maximum_offset_cm"),Bound)||!FMath::IsFinite(Bound)||Bound<=0||!Request->TryGetArrayField(TEXT("patches"),Rows)) return false;
    bool AllowNew=false;Request->TryGetBoolField(TEXT("allow_new_dna_targets"),AllowNew);
    auto TargetReader=USkelMeshDNAUtils::GetDNAReader(TargetMesh);
    auto NewTargetExists=[&](const FName& Name,int32 LOD)
    {
        if (!AllowNew||!TargetReader||LOD>=TargetReader->GetLODCount()) return false;
        for (const uint16 M:TargetReader->GetMeshIndicesForLOD(LOD))
            for (uint16 T=0;T<TargetReader->GetBlendShapeTargetCount(M);++T)
            {
                const FString Channel=TargetReader->GetBlendShapeChannelName(TargetReader->GetBlendShapeChannelIndex(M,T));
                if (Channel.StartsWith(TEXT("CM_"))&&FName(*(TargetReader->GetMeshName(M)+TEXT("__")+Channel))==Name) return true;
            }
        return false;
    };
    const FSkeletalMeshModel* Imported=TargetMesh->GetImportedModel();if (!Imported) return false;
    struct FPatch {FName Name;int32 LOD;TArray<uint32> Indices;TArray<dna::Vector3> Offsets;};
    TArray<FPatch> Patches;TSet<FString> Seen;
    for (const auto& Value:*Rows)
    {
        auto Row=Value->AsObject();FString Name;double LOD=0;
        if (!Row||!Row->TryGetStringField(TEXT("morph"),Name)||!Row->TryGetNumberField(TEXT("lod"),LOD)||!FMath::IsFinite(LOD)||LOD<0||LOD!=FMath::FloorToDouble(LOD)||!Imported->LODModels.IsValidIndex(int32(LOD))) return false;
        const FString Key=Name+FString::Printf(TEXT("/%d"),int32(LOD));if (Seen.Contains(Key)) return false;Seen.Add(Key);
        FPatch Patch;Patch.Name=FName(*Name);Patch.LOD=int32(LOD);
        if ((!SourceMesh->FindMorphTarget(Patch.Name)&&!NewTargetExists(Patch.Name,Patch.LOD))||!Points(Row,FPaths::GetPath(RequestFile),Patch.Offsets)) return false;
        const TArray<TSharedPtr<FJsonValue>>* Indices=nullptr;
        if (!Row->TryGetArrayField(TEXT("indices"),Indices)||Indices->Num()!=Patch.Offsets.Num()) return false;
        TSet<uint32> Vertices;
        for (int32 I=0;I<Indices->Num();++I)
        {
            double V=0;if (!(*Indices)[I]->TryGetNumber(V)||!FMath::IsFinite(V)||V<0||V!=FMath::FloorToDouble(V)||V>=Imported->LODModels[Patch.LOD].NumVertices||Vertices.Contains(uint32(V))) return false;
            const auto D=Patch.Offsets[I];if (FVector(D.x,D.y,D.z).Size()>Bound) return false;
            Patch.Indices.Add(uint32(V));Vertices.Add(uint32(V));
        }
        Patches.Add(MoveTemp(Patch));
    }
    TMap<FName,UMorphTarget*> Owned;
    TargetMesh->Modify();
    for (const auto& Patch:Patches)
    {
        UMorphTarget* Morph=Owned.FindRef(Patch.Name);
        if (!Morph)
        {
            UMorphTarget* Original=SourceMesh->FindMorphTarget(Patch.Name);
            Morph=TargetMesh->FindMorphTarget(Patch.Name);
            if (!Morph || Morph==Original || Morph->GetOuter()!=TargetMesh)
            {
                if (Morph) TargetMesh->UnregisterMorphTarget(Morph,false);
                Morph=Original?DuplicateObject<UMorphTarget>(Original,TargetMesh,Patch.Name):NewObject<UMorphTarget>(TargetMesh,Patch.Name,RF_Transactional);
            }
            if (!Morph || Morph==Original) return false;
            Morph->BaseSkelMesh=TargetMesh;if (Original) Morph->GetMorphLODModels()=Original->GetMorphLODModels();
            Owned.Add(Patch.Name,Morph);
        }
        TMap<uint32,FMorphTargetDelta> Deltas;
        if (Morph->GetMorphLODModels().IsValidIndex(Patch.LOD)) for (const auto& Delta:Morph->GetMorphLODModels()[Patch.LOD].Vertices) Deltas.Add(Delta.SourceIdx,Delta);
        for (int32 I=0;I<Patch.Indices.Num();++I)
        {
            const uint32 V=Patch.Indices[I];auto* Existing=Deltas.Find(V);
            if (!Existing) {FMorphTargetDelta NewDelta;NewDelta.SourceIdx=V;NewDelta.PositionDelta=FVector3f::ZeroVector;NewDelta.TangentZDelta=FVector3f::ZeroVector;Existing=&Deltas.Add(V,NewDelta);}
            const auto D=Patch.Offsets[I];Existing->PositionDelta+=FVector3f(D.x,D.y,D.z);
        }
        TArray<uint32> Order;Deltas.GetKeys(Order);Order.Sort();TArray<FMorphTargetDelta> Values;
        for (const uint32 V:Order) Values.Add(Deltas[V]);
        Morph->PopulateDeltas(Values,Patch.LOD,Imported->LODModels[Patch.LOD].Sections,true,false,0.f);
        // A newly created accessory morph has no data until PopulateDeltas.
        // Registering it earlier triggers UE's HasValidData ensure. Reject
        // genuinely empty targets here before registering them for a rebuild.
        if (!Morph->HasValidData()||!TargetMesh->RegisterMorphTarget(Morph,false)) return false;
        // Skeletal mesh PostEditChange rebuilds compiled morphs from the mesh
        // description. Persist the same assembled baseline + offset there,
        // rather than allowing that rebuild to discard our CPU-model edit.
        FMeshDescription* Description=TargetMesh->GetMeshDescription(Patch.LOD);
        if (!Description) return false;
        FSkeletalMeshAttributes Attributes(*Description);
        if (!Attributes.GetMorphTargetNames().Contains(Patch.Name)) Attributes.RegisterMorphTargetAttribute(Patch.Name,false);
        auto PositionDelta=Attributes.GetVertexMorphPositionDelta(Patch.Name);
        if (!PositionDelta.IsValid()) return false;
        for (const FVertexID V:Description->Vertices().GetElementIDs()) PositionDelta.Set(V,FVector3f::ZeroVector);
        const auto& VertexMap=Imported->LODModels[Patch.LOD].MeshToImportVertexMap;
        TMap<int32,FVector3d> Sums;TMap<int32,int32> Counts;
        for (const auto& Delta:Values)
        {
            if (!VertexMap.IsValidIndex(Delta.SourceIdx)) return false;
            const int32 Import=VertexMap[Delta.SourceIdx];if (!Description->IsVertexValid(FVertexID(Import))) return false;
            Sums.FindOrAdd(Import,FVector3d::ZeroVector)+=FVector3d(Delta.PositionDelta);++Counts.FindOrAdd(Import,0);
        }
        for (const auto& Pair:Sums) PositionDelta.Set(FVertexID(Pair.Key),FVector3f(Pair.Value/double(Counts[Pair.Key])));
        TargetMesh->CommitMeshDescription(Patch.LOD);
    }
    if (Patches.Num())
    {
        TargetMesh->InitMorphTargetsAndRebuildRenderData();FAssetCompilingManager::Get().FinishAllCompilation();
        if (AllowNew)
        {
            // Curve values remain ordinary attributes unless metadata marks
            // them as morph targets. Keep that metadata on the private mesh;
            // never change a shared MetaHuman skeleton to register our curves.
            auto Metadata=TargetMesh->GetAssetUserData<UAnimCurveMetaData>();
            if (!Metadata||Metadata->GetOuter()!=TargetMesh)
            {
                Metadata=Metadata?DuplicateObject<UAnimCurveMetaData>(Metadata,TargetMesh):NewObject<UAnimCurveMetaData>(TargetMesh,NAME_None,RF_Transactional);
                TargetMesh->RemoveUserDataOfClass(UAnimCurveMetaData::StaticClass());TargetMesh->AddAssetUserData(Metadata);
            }
            for (const auto& Patch:Patches) {Metadata->AddCurveMetaData(Patch.Name,false);Metadata->SetCurveMetaDataMorphTarget(Patch.Name,true);}
            // A rebuild may resolve DNA mappings before new morphs exist.
            // Reinitialize the private runtime/mapping pair after registration.
            auto Data=Cast<UDNAAssetUserData>(TargetMesh->GetAssetUserDataOfClass(UDNAAssetUserData::StaticClass()));
            if (!Data||!Data->DNAAsset||Data->DNAAsset->GetOuter()!=TargetMesh) return false;
            auto Current=Data->DNAAsset->GetDNAReader();if (!Current) return false;
            if (Current->IsLegacyWrapped()) Data->DNAAsset->RestoreLegacyUEMHCCompatibility();
            else Data->DNAAsset->SetDNAReader(Current,EDNACopyPolicy::Alias,ERigLogicInitPolicy::Initialize);
        }
        if (!ExpandNeutralBounds(TargetMesh)) return false;
        TargetMesh->MarkPackageDirty();
    }
    return true;
}
