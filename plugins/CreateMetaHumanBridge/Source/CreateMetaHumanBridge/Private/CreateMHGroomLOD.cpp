#include "CreateMHGroomLibrary.h"
#include "GroomAsset.h"
#include "GroomComponent.h"
#include "GroomInstance.h"
#include "NiagaraComponent.h"
#include "NiagaraSystem.h"
#include "NiagaraSystemInstanceController.h"
#include "Engine/World.h"
#include "HairStrandsInterface.h"
#include "RenderingThread.h"
#include "HairDescription.h"
#include "HairAttributes.h"
#include "Engine/StaticMesh.h"
#include "StaticMeshAttributes.h"
#include "MeshDescription.h"
#include "Serialization/MemoryWriter.h"
#include "Serialization/JsonReader.h"
#include "Serialization/JsonSerializer.h"
#include "Dom/JsonObject.h"
#include "Misc/FileHelper.h"
#include "Misc/PackageName.h"
#include "Misc/SecureHash.h"
#include "AssetRegistry/AssetRegistryModule.h"

FString UCreateMHGroomLibrary::CaptureGroomRepresentationState(UGroomComponent* Component)
{
    if (!IsInGameThread() || !Component || !Component->IsRegistered()) return TEXT("");
    struct FState { FHairGroupInstance* Instance=nullptr; EHairGeometryType Geometry=EHairGeometryType::NoneGeometry; float LOD=-1; bool Visible=false; };
    TArray<FState> States;
    for (uint32 Index=0;Index<Component->GetGroupCount();++Index) {FState State;State.Instance=Component->GetGroupInstance(Index);States.Add(State);}
    // Group selection belongs to the rendering thread. Read it there, then
    // wait before serializing; a forced request is not evidence of selection.
    ENQUEUE_RENDER_COMMAND(MHCaptureGroomRepresentation)([&States](FRHICommandListImmediate& RHICmdList)
    {
        for (FState& State:States) if (State.Instance)
        {
            State.Geometry=State.Instance->GeometryType;
            if (const FHairGroupPublicData* Data=State.Instance->HairGroupPublicData)
            {State.LOD=Data->GetLODIndex();State.Visible=Data->GetLODVisibility();}
        }
    });
    FlushRenderingCommands();
    auto Root=MakeShared<FJsonObject>();Root->SetNumberField(TEXT("requested_lod"),Component->GetForcedLOD());
    Root->SetBoolField(TEXT("component_visible"),Component->IsVisible());
    TArray<TSharedPtr<FJsonValue>> Groups;
    for (int32 Index=0;Index<States.Num();++Index)
    {
        const FState& State=States[Index];auto Row=MakeShared<FJsonObject>();
        const TCHAR* Geometry=State.Geometry==EHairGeometryType::Strands?TEXT("STRANDS"):State.Geometry==EHairGeometryType::Cards?TEXT("CARDS"):State.Geometry==EHairGeometryType::Meshes?TEXT("MESHES"):TEXT("NONE");
        Row->SetNumberField(TEXT("group"),Index);Row->SetStringField(TEXT("geometry"),Geometry);
        Row->SetNumberField(TEXT("selected_lod"),State.LOD);Row->SetBoolField(TEXT("lod_visible"),State.Visible);
        Row->SetBoolField(TEXT("instance_available"),State.Instance!=nullptr);Groups.Add(MakeShared<FJsonValueObject>(Row));
    }
    Root->SetArrayField(TEXT("groups"),Groups);
    // Report live solver state separately from asset enable flags. Desired
    // age, a forced LOD and an active component alone do not prove playback.
    TArray<TSharedPtr<FJsonValue>> Simulation;
    for (int32 Index=0;Index<Component->NiagaraComponents.Num();++Index)
    {
        auto Row=MakeShared<FJsonObject>();Row->SetNumberField(TEXT("group"),Index);
        UNiagaraComponent* Niagara=Component->NiagaraComponents[Index];
        Row->SetBoolField(TEXT("component_available"),IsValid(Niagara));
        if (IsValid(Niagara))
        {
            Row->SetStringField(TEXT("component"),Niagara->GetPathName());
            Row->SetStringField(TEXT("system"),Niagara->GetAsset()?Niagara->GetAsset()->GetPathName():TEXT(""));
            Row->SetBoolField(TEXT("registered"),Niagara->IsRegistered());
            Row->SetBoolField(TEXT("active"),Niagara->IsActive());
            Row->SetBoolField(TEXT("paused"),Niagara->IsPaused());
            auto Controller=Niagara->GetSystemInstanceController();
            const bool ValidController=Controller.IsValid()&&Controller->IsValid();
            Row->SetBoolField(TEXT("controller_available"),ValidController);
            if (ValidController)
            {
                Controller->WaitForConcurrentTickAndFinalize();
                Row->SetNumberField(TEXT("age_seconds"),Controller->GetAge());
                Row->SetStringField(TEXT("execution_state"),StaticEnum<ENiagaraExecutionState>()->GetNameStringByValue(static_cast<int64>(Controller->GetActualExecutionState())));
            }
        }
        Simulation.Add(MakeShared<FJsonValueObject>(Row));
    }
    Root->SetArrayField(TEXT("runtime_simulation"),Simulation);
    if (Component->GetWorld())Root->SetNumberField(TEXT("world_time_seconds"),Component->GetWorld()->GetTimeSeconds());
    FString Result;FJsonSerializer::Serialize(Root,TJsonWriterFactory<>::Create(&Result));return Result;
}

namespace MHGroomLOD
{
using FObject = TSharedRef<FJsonObject>;
static FString Text(const FObject& Object)
{
    FString Result;
    FJsonSerializer::Serialize(Object,TJsonWriterFactory<>::Create(&Result));
    return Result;
}
static TSharedPtr<FJsonObject> Parse(const FString& Value)
{
    TSharedPtr<FJsonObject> Object;
    FJsonSerializer::Deserialize(TJsonReaderFactory<>::Create(Value),Object);
    return Object;
}
static FObject Failure(const TCHAR* Reason)
{
    FObject Object=MakeShared<FJsonObject>();
    Object->SetStringField(TEXT("status"),TEXT("failed"));Object->SetStringField(TEXT("error"),Reason);
    return Object;
}
static TSharedPtr<FJsonValue> Point(const FVector3f& V)
{
    return MakeShared<FJsonValueArray>(TArray<TSharedPtr<FJsonValue>>{
        MakeShared<FJsonValueNumber>(V.X),MakeShared<FJsonValueNumber>(V.Y),MakeShared<FJsonValueNumber>(V.Z)});
}
static TSharedPtr<FJsonValue> UV(const FVector2f& V)
{
    return MakeShared<FJsonValueArray>(TArray<TSharedPtr<FJsonValue>>{MakeShared<FJsonValueNumber>(V.X),MakeShared<FJsonValueNumber>(V.Y)});
}
static FString Signature(FMeshDescription& Mesh)
{
    TArray<uint8> Bytes;FMemoryWriter Writer(Bytes,true);Mesh.Serialize(Writer);
    FSHAHash Hash;FSHA1::HashBuffer(Bytes.GetData(),Bytes.Num(),Hash.Hash);
    return Hash.ToString();
}
static FString Invariants(FMeshDescription Mesh)
{
    FStaticMeshAttributes Attr(Mesh);auto P=Attr.GetVertexPositions();auto N=Attr.GetVertexInstanceNormals();auto T=Attr.GetVertexInstanceTangents();
    for (FVertexID ID:Mesh.Vertices().GetElementIDs()) P[ID]=FVector3f::ZeroVector;
    for (FVertexInstanceID ID:Mesh.VertexInstances().GetElementIDs()) {N[ID]=FVector3f::ZeroVector;T[ID]=FVector3f::ZeroVector;}
    return Signature(Mesh);
}
static TSharedPtr<FJsonObject> GroupFingerprint(UGroomAsset* Groom,int32 Group)
{
    TSharedPtr<FJsonObject> Root=Parse(UCreateMHGroomLibrary::FingerprintGroom(Groom));
    const TArray<TSharedPtr<FJsonValue>>* Groups=nullptr;
    if (!Root || Root->GetStringField(TEXT("status"))!=TEXT("complete") || !Root->TryGetArrayField(TEXT("groups"),Groups) || !Groups->IsValidIndex(Group)) return nullptr;
    return (*Groups)[Group]->AsObject();
}
static bool SameFingerprint(const TSharedPtr<FJsonObject>& A,const TSharedPtr<FJsonObject>& B)
{
    return A && B && A->GetStringField(TEXT("sha1"))==B->GetStringField(TEXT("sha1"))
        && A->GetIntegerField(TEXT("curves"))==B->GetIntegerField(TEXT("curves"))
        && A->GetIntegerField(TEXT("points"))==B->GetIntegerField(TEXT("points"));
}
static bool ReadPoint(const TSharedPtr<FJsonValue>& Value,FVector3f& Out)
{
    const TArray<TSharedPtr<FJsonValue>>* V=nullptr;
    if (!Value || !Value->TryGetArray(V) || V->Num()!=3) return false;
    double X,Y,Z;
    if (!(*V)[0]->TryGetNumber(X) || !(*V)[1]->TryGetNumber(Y) || !(*V)[2]->TryGetNumber(Z)) return false;
    Out=FVector3f(X,Y,Z);return !Out.ContainsNaN();
}
struct FRotation
{
    FVector3f R[3];
    FVector3f Apply(const FVector3f& V) const {return FVector3f(R[0].Dot(V),R[1].Dot(V),R[2].Dot(V));}
    bool Valid() const
    {
        return FMath::Abs(R[0].SquaredLength()-1)<.002f && FMath::Abs(R[1].SquaredLength()-1)<.002f
            && FMath::Abs(R[2].SquaredLength()-1)<.002f && FMath::Abs(R[0].Dot(R[1]))<.002f
            && FMath::Abs(R[1].Dot(R[2]))<.002f && FMath::Abs(R[2].Dot(R[0]))<.002f
            && R[0].Dot(R[1].Cross(R[2]))>.998f;
    }
};
struct FChange
{
    UStaticMesh* Source=nullptr;int32 LOD=0;FString Signature,Invariants;
    TMap<int32,int32> Indices;
    TArray<FVector3f> Old,Positions;TArray<FRotation> Rotations;
};
}

FString UCreateMHGroomLibrary::ExportGroomLODDeformation(UGroomAsset* Source,int32 SourceGroup,UGroomAsset* Target,int32 TargetGroup,const FString& FilePath,int32 MaxControls)
{
    using namespace MHGroomLOD;
    auto Fail=[](const TCHAR* Reason){return Text(Failure(Reason));};
    if (MaxControls<512 || MaxControls>262144) return Fail(TEXT("unsupported_control_budget"));
    TSharedPtr<FJsonObject> Comparison=Parse(CompareGroomGroups(Source,SourceGroup,Target,TargetGroup,false));
    if (!Comparison || !Comparison->GetBoolField(TEXT("matches"))) return Fail(TEXT("requires_matching_position_only_curve_layout"));
    FHairDescription A=Source->GetHairDescription(),B=Target->GetHairDescription();
    auto AP=A.VertexAttributes().GetAttributesRef<FVector3f>(HairAttribute::Vertex::Position);
    auto BP=B.VertexAttributes().GetAttributesRef<FVector3f>(HairAttribute::Vertex::Position);
    auto AC=A.StrandAttributes().GetAttributesRef<int32>(HairAttribute::Strand::VertexCount);
    auto BC=B.StrandAttributes().GetAttributesRef<int32>(HairAttribute::Strand::VertexCount);
    auto AG=A.StrandAttributes().GetAttributesRef<int32>(HairAttribute::Strand::GroupID);
    auto BG=B.StrandAttributes().GetAttributesRef<int32>(HairAttribute::Strand::GroupID);
    auto AU=A.StrandAttributes().GetAttributesRef<FVector2f>(HairAttribute::Strand::RootUV);
    auto Select=[](const FHairDescription& Desc,const auto& Counts,const auto& Groups,int32 Group)
    {
        TArray<FIntPoint> Rows;int32 Offset=0;
        for (int32 I=0;I<Desc.GetNumStrands();++I)
        {
            if ((Groups.IsValid()?Groups[FStrandID(I)]:0)==Group) Rows.Emplace(I,Offset);
            Offset+=Counts[FStrandID(I)];
        }
        return Rows;
    };
    TArray<FIntPoint> AS=Select(A,AC,AG,Source->GetHairGroupsInfo()[SourceGroup].GroupID);
    TArray<FIntPoint> BS=Select(B,BC,BG,Target->GetHairGroupsInfo()[TargetGroup].GroupID);
    // Keep dense neighborhoods along representative curves rather than one
    // unrelated sample from almost every strand. Reserve half the curve
    // budget for actual edited curves so a small local repair is not missed.
    TArray<int32> Edited;
    for (int32 I=0;I<AS.Num();++I)
    {
        for (int32 J=0;J<AC[FStrandID(AS[I].X)];++J)
        {
            if (!AP[FVertexID(AS[I].Y+J)].Equals(BP[FVertexID(BS[I].Y+J)],.001f)) {Edited.Add(I);break;}
        }
    }
    const int32 CurveBudget=FMath::Min(MaxControls/32,AS.Num());
    TSet<int32> Chosen;
    const int32 EditedBudget=FMath::Min(Edited.Num(),CurveBudget/2);
    for (int32 I=0;I<EditedBudget;++I) Chosen.Add(Edited[int32(int64(I)*Edited.Num()/EditedBudget)]);
    const int32 UniformBudget=CurveBudget-Chosen.Num();
    for (int32 I=0;I<UniformBudget;++I) Chosen.Add(int32(int64(I)*AS.Num()/UniformBudget));
    TArray<int32> Selected=Chosen.Array();Selected.Sort();
    TArray<TSharedPtr<FJsonValue>> Controls,Targets,Roots,RootPositions;
    for (int32 I:Selected)
    {
        const int32 Count=AC[FStrandID(AS[I].X)],Samples=FMath::Min(32,Count);
        for (int32 K=0;K<Samples;++K)
        {
            const int32 J=int32(int64(K)*(Count-1)/(Samples-1));
            Controls.Add(Point(AP[FVertexID(AS[I].Y+J)]));Targets.Add(Point(BP[FVertexID(BS[I].Y+J)]));
            if (AU.IsValid())
            {
                Roots.Add(UV(AU[FStrandID(AS[I].X)]));RootPositions.Add(Point(AP[FVertexID(AS[I].Y)]));
            }
        }
    }
    TSharedPtr<FJsonObject> SourceFP=GroupFingerprint(Source,SourceGroup),TargetFP=GroupFingerprint(Target,TargetGroup);
    if (!SourceFP || !TargetFP) return Fail(TEXT("groom_group_fingerprint_unavailable"));
    FObject Request=MakeShared<FJsonObject>();Request->SetNumberField(TEXT("schema"),1);
    Request->SetStringField(TEXT("source_groom"),Source->GetPathName());Request->SetStringField(TEXT("target_groom"),Target->GetPathName());
    Request->SetNumberField(TEXT("source_group"),SourceGroup);Request->SetNumberField(TEXT("target_group"),TargetGroup);
    Request->SetObjectField(TEXT("source_geometry"),SourceFP);
    Request->SetObjectField(TEXT("target_geometry"),TargetFP);
    Request->SetArrayField(TEXT("source_controls"),Controls);Request->SetArrayField(TEXT("target_controls"),Targets);
    Request->SetStringField(TEXT("control_sampling"),TEXT("DENSE_EDIT_AWARE_CURVES"));
    if (Roots.Num())
    {
        Request->SetArrayField(TEXT("control_root_uv"),Roots);Request->SetArrayField(TEXT("control_root_positions"),RootPositions);
    }
    TArray<TSharedPtr<FJsonValue>> Meshes;TSet<UStaticMesh*> Exported;
    auto Capture=[&](UStaticMesh* Mesh,int32 GroomLOD,bool bCards)->bool
    {
        if (!Mesh) return false;
        if (Exported.Contains(Mesh)) return true;
        Exported.Add(Mesh);
        if (!Mesh->GetNumSourceModels()) return false;
        for (int32 LOD=0;LOD<Mesh->GetNumSourceModels();++LOD)
        {
            FMeshDescription* Desc=Mesh->GetMeshDescription(LOD);
            if (!Desc || !Desc->Vertices().Num() || !Desc->Triangles().Num()) return false;
            FObject Row=MakeShared<FJsonObject>();Row->SetStringField(TEXT("source"),Mesh->GetPathName());
            Row->SetNumberField(TEXT("lod"),LOD);Row->SetStringField(TEXT("signature"),Signature(*Desc));
            Row->SetStringField(TEXT("uv_topology_signature"),Invariants(*Desc));
            FStaticMeshConstAttributes Attr(*Desc);auto Positions=Attr.GetVertexPositions();
            TMap<int32,int32> Index;TArray<TSharedPtr<FJsonValue>> IDs,Points,Triangles,RootUV;
            const FHairCardsBulkData* Bulk=nullptr;
            if (bCards && LOD==0 && Source->GetHairGroupsPlatformData().IsValidIndex(SourceGroup))
            {
                const auto& Cards=Source->GetHairGroupsPlatformData()[SourceGroup].Cards.LODs;
                if (Cards.IsValidIndex(GroomLOD) && Cards[GroomLOD].BulkData.IsValid()) Bulk=&Cards[GroomLOD].BulkData;
            }
            bool bRootUV=Bulk && AU.IsValid() && Bulk->Header.NumUVs>0;
            for (const FVertexID Vertex:Desc->Vertices().GetElementIDs())
            {
                Index.Add(Vertex.GetValue(),Points.Num());IDs.Add(MakeShared<FJsonValueNumber>(Vertex.GetValue()));Points.Add(Point(Positions[Vertex]));
                if (bRootUV)
                {
                    int32 ID=Vertex.GetValue(),UVIndex=ID*int32((Bulk->Header.NumUVs+1)/2);
                    if (!Bulk->Positions.IsValidIndex(ID) || !Bulk->UVs.IsValidIndex(UVIndex)
                        || !FVector3f(Bulk->Positions[ID]).Equals(Positions[Vertex],.00001f)) bRootUV=false;
                    else RootUV.Add(UV(FVector2f(Bulk->UVs[UVIndex].X,Bulk->UVs[UVIndex].Y)));
                }
            }
            for (const FTriangleID Triangle:Desc->Triangles().GetElementIDs())
            {
                TArray<TSharedPtr<FJsonValue>> Values;
                for (FVertexID Vertex:Desc->GetTriangleVertices(Triangle)) Values.Add(MakeShared<FJsonValueNumber>(Index[Vertex.GetValue()]));
                Triangles.Add(MakeShared<FJsonValueArray>(Values));
            }
            Row->SetArrayField(TEXT("vertex_ids"),IDs);Row->SetArrayField(TEXT("positions"),Points);Row->SetArrayField(TEXT("triangles"),Triangles);
            if (bRootUV && RootUV.Num()==Points.Num()) Row->SetArrayField(TEXT("root_uv"),RootUV);
            Meshes.Add(MakeShared<FJsonValueObject>(Row));
        }
        return true;
    };
    for (const auto& Row:Source->GetHairGroupsCards())
        if (Row.GroupIndex==SourceGroup && !Capture(Row.ImportedMesh,Row.LODIndex,true)) return Fail(TEXT("card_source_mesh_unavailable"));
    for (const auto& Row:Source->GetHairGroupsMeshes())
        if (Row.GroupIndex==SourceGroup && !Capture(Row.ImportedMesh,Row.LODIndex,false)) return Fail(TEXT("helmet_source_mesh_unavailable"));
    if (!Meshes.Num()) return Fail(TEXT("no_imported_lod_meshes"));
    Request->SetArrayField(TEXT("meshes"),Meshes);
    if (!FFileHelper::SaveStringToFile(Text(Request),*FilePath,FFileHelper::EEncodingOptions::ForceUTF8WithoutBOM)) return Fail(TEXT("request_write_failed"));
    FObject Report=MakeShared<FJsonObject>();Report->SetStringField(TEXT("status"),TEXT("complete"));
    Report->SetNumberField(TEXT("controls"),Controls.Num());Report->SetNumberField(TEXT("mesh_lods"),Meshes.Num());
    Report->SetStringField(TEXT("request"),FilePath);return Text(Report);
}

FString UCreateMHGroomLibrary::ApplyGroomLODDeformation(const FString& FilePath,const FString& DestinationFolder)
{
    using namespace MHGroomLOD;
    auto Fail=[](const TCHAR* Reason){return Text(Failure(Reason));};
    if (!DestinationFolder.StartsWith(TEXT("/Game/")) || !FPackageName::IsValidLongPackageName(DestinationFolder)) return Fail(TEXT("use_private_project_destination"));
    FString Input;
    if (!FFileHelper::LoadFileToString(Input,*FilePath)) return Fail(TEXT("result_missing"));
    TSharedPtr<FJsonObject> Root=Parse(Input);
    if (!Root || Root->GetIntegerField(TEXT("schema"))!=1 || Root->GetStringField(TEXT("status"))!=TEXT("complete") || !Root->GetBoolField(TEXT("accepted"))) return Fail(TEXT("solver_did_not_accept_meshes"));
    UGroomAsset* Source=LoadObject<UGroomAsset>(nullptr,*Root->GetStringField(TEXT("source_groom")));
    UGroomAsset* Target=LoadObject<UGroomAsset>(nullptr,*Root->GetStringField(TEXT("target_groom")));
    int32 SG=Root->GetIntegerField(TEXT("source_group")),TG=Root->GetIntegerField(TEXT("target_group"));
    if (!SameFingerprint(GroupFingerprint(Source,SG),Root->GetObjectField(TEXT("source_geometry")))
        || !SameFingerprint(GroupFingerprint(Target,TG),Root->GetObjectField(TEXT("target_geometry")))) return Fail(TEXT("groom_changed_since_capture"));
    TSharedPtr<FJsonObject> Comparison=Parse(CompareGroomGroups(Source,SG,Target,TG,false));
    if (!Comparison || !Comparison->GetBoolField(TEXT("matches"))) return Fail(TEXT("curve_layout_changed_since_capture"));
    const float Maximum=FMath::Max(2.f*float(Comparison->GetNumberField(TEXT("max_position_error_cm"))),.00001f);
    TArray<FChange> Changes;TMap<UStaticMesh*,FString> Destinations;TSet<FString> Rows;
    const TArray<TSharedPtr<FJsonValue>>& Meshes=Root->GetArrayField(TEXT("meshes"));
    if (!Meshes.Num() || Meshes.Num()>256) return Fail(TEXT("unsupported_mesh_count"));
    for (const auto& Value:Meshes)
    {
        const TSharedPtr<FJsonObject> Row=Value->AsObject();
        if (!Row || !Row->GetObjectField(TEXT("report"))->GetBoolField(TEXT("accepted"))) return Fail(TEXT("mesh_quality_failed"));
        FChange Change;Change.Source=LoadObject<UStaticMesh>(nullptr,*Row->GetStringField(TEXT("source")));Change.LOD=Row->GetIntegerField(TEXT("lod"));
        if (!Change.Source) return Fail(TEXT("source_mesh_missing"));
        if (Change.LOD<0 || Change.LOD>=Change.Source->GetNumSourceModels()) return Fail(TEXT("invalid_source_mesh_lod"));
        FString Key=Change.Source->GetPathName()+TEXT(":")+FString::FromInt(Change.LOD);
        if (Rows.Contains(Key)) return Fail(TEXT("duplicate_mesh_lod"));Rows.Add(Key);
        FMeshDescription* Desc=Change.Source->GetMeshDescription(Change.LOD);
        if (!Desc || Signature(*Desc)!=Row->GetStringField(TEXT("signature"))) return Fail(TEXT("source_mesh_changed_since_capture"));
        Change.Signature=Signature(*Desc);Change.Invariants=Invariants(*Desc);
        const auto& IDs=Row->GetArrayField(TEXT("vertex_ids"));const auto& Points=Row->GetArrayField(TEXT("positions"));const auto& Rotations=Row->GetArrayField(TEXT("rotations"));
        if (IDs.Num()!=Desc->Vertices().Num() || Points.Num()!=IDs.Num() || Rotations.Num()!=IDs.Num()) return Fail(TEXT("vertex_count_changed"));
        FStaticMeshConstAttributes Attr(*Desc);auto Old=Attr.GetVertexPositions();
        for (int32 I=0;I<IDs.Num();++I)
        {
            double Number=IDs[I]->AsNumber();int32 ID=int32(Number);
            if (Number!=ID || !Desc->Vertices().IsValid(FVertexID(ID)) || Change.Indices.Contains(ID)) return Fail(TEXT("vertex_identity_changed"));
            FVector3f P;if (!ReadPoint(Points[I],P) || FVector3f::Dist(P,Old[FVertexID(ID)])>Maximum+.000001f) return Fail(TEXT("invalid_vertex_movement"));
            const TArray<TSharedPtr<FJsonValue>>* Matrix=nullptr;FRotation R;
            if (!Rotations[I]->TryGetArray(Matrix) || Matrix->Num()!=3) return Fail(TEXT("invalid_rotation"));
            for (int32 Axis=0;Axis<3;++Axis) if (!ReadPoint((*Matrix)[Axis],R.R[Axis])) return Fail(TEXT("invalid_rotation"));
            if (!R.Valid()) return Fail(TEXT("rotation_contains_scale_or_reflection"));
            Change.Indices.Add(ID,I);Change.Old.Add(Old[FVertexID(ID)]);Change.Positions.Add(P);Change.Rotations.Add(R);
        }
        for (const FTriangleID Triangle:Desc->Triangles().GetElementIDs())
        {
            const auto Vertices=Desc->GetTriangleVertices(Triangle);
            const int32 A=Change.Indices[Vertices[0].GetValue()],B=Change.Indices[Vertices[1].GetValue()],C=Change.Indices[Vertices[2].GetValue()];
            FVector3f N=(Change.Old[B]-Change.Old[A]).Cross(Change.Old[C]-Change.Old[A]);
            FVector3f M=(Change.Positions[B]-Change.Positions[A]).Cross(Change.Positions[C]-Change.Positions[A]);
            if (N.Length()>1e-10f)
            {
                FVector3f Expected=(Change.Rotations[A].Apply(N)+Change.Rotations[B].Apply(N)+Change.Rotations[C].Apply(N))/3.f;
                if (M.Length()<N.Length()*.1f || Expected.GetSafeNormal().Dot(M.GetSafeNormal())<.2f) return Fail(TEXT("collapsed_or_inverted_triangle"));
            }
            const int32 EdgeIDs[3]={A,B,C};
            for (int32 E=0;E<3;++E)
            {
                int32 X=EdgeIDs[E],Y=EdgeIDs[(E+1)%3];float Before=FVector3f::Dist(Change.Old[X],Change.Old[Y]);
                float After=FVector3f::Dist(Change.Positions[X],Change.Positions[Y]);
                if (Before>1e-8f && (After<Before*.5f || After>Before*1.5f)) return Fail(TEXT("strained_mesh_edge"));
            }
        }
        if (!Destinations.Contains(Change.Source))
        {
            FString Name=Change.Source->GetName()+TEXT("_Wrapped_")+Row->GetStringField(TEXT("signature")).Left(8);
            FString Path=DestinationFolder+TEXT("/")+Name;
            if (FindPackage(nullptr,*Path) || FPackageName::DoesPackageExist(Path)) return Fail(TEXT("destination_already_exists"));
            for (const auto& Pair:Destinations) if (Pair.Value==Path) return Fail(TEXT("destination_name_collision"));
            Destinations.Add(Change.Source,Path);
        }
        Changes.Add(MoveTemp(Change));
    }
    // Refuse partial source-LOD replacement before creating any assets.
    for (const auto& Pair:Destinations)
        for (int32 LOD=0;LOD<Pair.Key->GetNumSourceModels();++LOD)
            if (!Rows.Contains(Pair.Key->GetPathName()+TEXT(":")+FString::FromInt(LOD))) return Fail(TEXT("missing_source_mesh_lod"));
    TMap<UStaticMesh*,UStaticMesh*> Copies;
    auto Discard=[&Copies]()
    {
        for (auto& Pair:Copies)
        {
            Pair.Value->ClearFlags(RF_Public|RF_Standalone);Pair.Value->MarkAsGarbage();
        }
    };
    for (const auto& Pair:Destinations)
    {
        UPackage* Package=CreatePackage(*Pair.Value);
        UStaticMesh* Copy=Cast<UStaticMesh>(StaticDuplicateObject(Pair.Key,Package,*FPackageName::GetLongPackageAssetName(Pair.Value)));
        if (!Copy) {Discard();return Fail(TEXT("private_mesh_copy_failed"));}
        Copies.Add(Pair.Key,Copy);Copy->SetAssetImportData(nullptr);
    }
    for (const FChange& Change:Changes)
    {
        UStaticMesh* Copy=Copies[Change.Source];
        FMeshDescription* Desc=Copy->CreateMeshDescription(Change.LOD,*Change.Source->GetMeshDescription(Change.LOD));
        FStaticMeshAttributes Attr(*Desc);auto Positions=Attr.GetVertexPositions();auto Normals=Attr.GetVertexInstanceNormals();auto Tangents=Attr.GetVertexInstanceTangents();
        for (const auto& Pair:Change.Indices) Positions[FVertexID(Pair.Key)]=Change.Positions[Pair.Value];
        for (const FVertexInstanceID Instance:Desc->VertexInstances().GetElementIDs())
        {
            int32 I=Change.Indices[Desc->GetVertexInstanceVertex(Instance).GetValue()];
            Normals[Instance]=Change.Rotations[I].Apply(Normals[Instance]).GetSafeNormal();
            Tangents[Instance]=Change.Rotations[I].Apply(Tangents[Instance]).GetSafeNormal();
        }
        auto& Settings=Copy->GetSourceModel(Change.LOD).BuildSettings;
        Settings.bRecomputeNormals=false;Settings.bRecomputeTangents=false;
        Copy->CommitMeshDescription(Change.LOD);
    }
    for (const auto& Pair:Copies)
    {
        TArray<FText> Errors;Pair.Value->Build(true,&Errors);
        if (Errors.Num()) {Discard();return Fail(TEXT("private_mesh_build_failed"));}
    }
    for (const FChange& Change:Changes)
    {
        if (Signature(*Change.Source->GetMeshDescription(Change.LOD))!=Change.Signature)
            {Discard();return Fail(TEXT("original_mesh_changed_during_build"));}
        if (Invariants(*Copies[Change.Source]->GetMeshDescription(Change.LOD))!=Change.Invariants)
            {Discard();return Fail(TEXT("uv_topology_or_other_attributes_changed"));}
    }
    TArray<TSharedPtr<FJsonValue>> Output;
    for (const auto& Pair:Copies)
    {
        Pair.Value->MarkPackageDirty();FAssetRegistryModule::AssetCreated(Pair.Value);
        FObject Row=MakeShared<FJsonObject>();Row->SetStringField(TEXT("source"),Pair.Key->GetPathName());Row->SetStringField(TEXT("mesh"),Pair.Value->GetPathName());
        Output.Add(MakeShared<FJsonValueObject>(Row));
    }
    FObject Report=MakeShared<FJsonObject>();Report->SetStringField(TEXT("status"),TEXT("complete"));Report->SetArrayField(TEXT("meshes"),Output);
    Report->SetBoolField(TEXT("original_meshes_unchanged"),true);
    Report->SetBoolField(TEXT("uv_topology_attributes_verified"),true);return Text(Report);
}
