#include "CreateMHExpressionLibrary.h"
#include "DNAUtils.h"
#include "DNAReader.h"
#include "RigLogic.h"
#include "RigInstance.h"
#include "DNAAsset.h"
#include "SharedRigRuntimeContext.h"
#include "UObject/StrongObjectPtr.h"
#include "Dom/JsonObject.h"
#include "Serialization/JsonReader.h"
#include "Serialization/JsonSerializer.h"
#include "Misc/FileHelper.h"
#include "Misc/Paths.h"

namespace
{
TSharedPtr<FJsonValue> Point(const FVector& V)
{
    return MakeShared<FJsonValueArray>(TArray<TSharedPtr<FJsonValue>>{
        MakeShared<FJsonValueNumber>(V.X), MakeShared<FJsonValueNumber>(V.Y), MakeShared<FJsonValueNumber>(V.Z)});
}

bool WriteJSON(const FString& Path, const TSharedRef<FJsonObject>& Value)
{
    FString Text;
    return FJsonSerializer::Serialize(Value, TJsonWriterFactory<>::Create(&Text)) &&
        FFileHelper::SaveStringToFile(Text, *Path, FFileHelper::EEncodingOptions::ForceUTF8WithoutBOM);
}

TArray<FMatrix> JointMatrices(const IDNAReader& Reader, TArrayView<const float> Neutral, TArrayView<const float> Delta)
{
    // Same quaternion layout/composition as UE 5.8 FAnimNode_RigLogic::UpdateJoints.
    TArray<FMatrix> Matrices;
    TArray<uint8> Visited;
    Matrices.SetNum(Reader.GetJointCount());
    Visited.Init(0, Matrices.Num());
    TFunction<void(int32)> Visit = [&](int32 J)
    {
        check(Visited[J] != 1);
        if (Visited[J] == 2) return;
        Visited[J] = 1;
        const int32 I = J * 10;
        const FQuat Rotation = FQuat(Neutral[I+3], Neutral[I+4], Neutral[I+5], Neutral[I+6]) *
            (Delta.Num() ? FQuat(Delta[I+3], Delta[I+4], Delta[I+5], Delta[I+6]) : FQuat::Identity);
        const auto At = [&](int32 K) { return Neutral[I+K] + (Delta.Num() ? Delta[I+K] : 0.0f); };
        const FTransform Local(Rotation, FVector(At(0),At(1),At(2)), FVector(At(7),At(8),At(9)));
        Matrices[J] = Local.ToMatrixWithScale();
        const int32 Parent = Reader.GetJointParentIndex(J);
        if (Parent != J && Parent >= 0 && Parent < Matrices.Num())
        {
            Visit(Parent);
            Matrices[J] *= Matrices[Parent];
        }
        Visited[J] = 2;
    };
    for (int32 J=0; J<Matrices.Num(); ++J) Visit(J);
    return Matrices;
}
}

bool UCreateMHExpressionLibrary::ExportExpressionReferenceJSON(const FString& DNAFile, const FString& RequestFile, const FString& OutputFile)
{
    FString RequestText;
    TSharedPtr<FJsonObject> Request;
    if (!FFileHelper::LoadFileToString(RequestText,*RequestFile) ||
        !FJsonSerializer::Deserialize(TJsonReaderFactory<>::Create(RequestText), Request) || !Request.IsValid()) return false;
    const TArray<TSharedPtr<FJsonValue>>* Poses = nullptr;
    if (!Request->TryGetArrayField(TEXT("poses"),Poses)) return false;
    FDNAConfig Config;
    Config.CoordinateSystemTransformPolicy = ECoordinateSystemTransformPolicy::Transform;
    Config.MaxLODPerPlatform = FPerPlatformInt(0);
    Config.MinLODPerPlatform = FPerPlatformInt(0);
    TSharedPtr<IDNAReader> Reader = LoadDNAFromFile(DNAFile,Config);
    if (!Reader) return false;
    bool bMetaHumanCompatibility=false;
    Request->TryGetBoolField(TEXT("metahuman_compatibility"),bMetaHumanCompatibility);
    TStrongObjectPtr<UDNAAsset> NativeDNA;
    TSharedPtr<FRigLogic> NativeRig;
    if (bMetaHumanCompatibility)
    {
        NativeDNA.Reset(NewObject<UDNAAsset>(GetTransientPackage(),NAME_None,RF_Transient));
        NativeDNA->SetDNAReader(Reader,EDNACopyPolicy::Alias,ERigLogicInitPolicy::Defer);
        NativeDNA->RestoreLegacyUEMHCCompatibility();
        Reader=NativeDNA->GetDNAReader();
        NativeRig=NativeDNA->GetRigRuntimeContext()->RigLogic;
    }
    TSharedRef<FJsonObject> Root = MakeShared<FJsonObject>();
    Root->SetNumberField(TEXT("schema"),2);
    Root->SetStringField(TEXT("coordinates"),TEXT("UE_CENTIMETERS"));
    Root->SetBoolField(TEXT("metahuman_compatibility"),bMetaHumanCompatibility);
    Root->SetStringField(TEXT("joint_value_convention"),bMetaHumanCompatibility?TEXT("MHC_LEGACY_LOCAL"):TEXT("TRANSFORMED_DNA_LOCAL"));
    Root->SetStringField(TEXT("dna"),DNAFile);
    Root->SetNumberField(TEXT("joint_count"),Reader->GetJointCount());
    TArray<TSharedPtr<FJsonValue>> JointNames,JointParents;
    for (uint16 J=0;J<Reader->GetJointCount();++J)
    {
        JointNames.Add(MakeShared<FJsonValueString>(Reader->GetJointName(J)));
        JointParents.Add(MakeShared<FJsonValueNumber>(Reader->GetJointParentIndex(J)));
    }
    Root->SetArrayField(TEXT("joint_names"),JointNames);
    Root->SetArrayField(TEXT("joint_parents"),JointParents);
    Root->SetNumberField(TEXT("joint_group_count"),Reader->GetJointGroupCount());
    TMap<FString,int32> Controls;
    TArray<TSharedPtr<FJsonValue>> ControlNames, Meshes, Results;
    for (uint16 I=0; I<Reader->GetRawControlCount(); ++I)
    {
        Controls.Add(Reader->GetRawControlName(I),I);
        ControlNames.Add(MakeShared<FJsonValueString>(Reader->GetRawControlName(I)));
    }
    Root->SetArrayField(TEXT("raw_controls"),ControlNames);
    TArray<TSharedPtr<FJsonValue>> AnimatedMapNames;
    for (uint16 I=0; I<Reader->GetAnimatedMapCount(); ++I)
        AnimatedMapNames.Add(MakeShared<FJsonValueString>(Reader->GetAnimatedMapName(I)));
    Root->SetArrayField(TEXT("animated_maps"),AnimatedMapNames);
    const auto MeshIndices = Reader->GetMeshIndicesForLOD(0);
    for (const uint16 M : MeshIndices)
    {
        TSharedRef<FJsonObject> Mesh = MakeShared<FJsonObject>();
        Mesh->SetNumberField(TEXT("index"),M);
        Mesh->SetStringField(TEXT("name"),Reader->GetMeshName(M));
        Mesh->SetNumberField(TEXT("blendshape_count"),Reader->GetBlendShapeTargetCount(M));
        TArray<TSharedPtr<FJsonValue>> Neutral, Layouts;
        for (uint32 V=0; V<Reader->GetVertexPositionCount(M); ++V) Neutral.Add(Point(Reader->GetVertexPosition(M,V)));
        for (uint32 L=0; L<Reader->GetVertexLayoutCount(M); ++L)
        {
            const FVertexLayout Layout = Reader->GetVertexLayout(M,L);
            const auto UV = Reader->GetVertexTextureCoordinate(M,Layout.TextureCoordinate);
            Layouts.Add(MakeShared<FJsonValueArray>(TArray<TSharedPtr<FJsonValue>>{
                MakeShared<FJsonValueNumber>(Layout.Position),MakeShared<FJsonValueNumber>(UV.U),MakeShared<FJsonValueNumber>(UV.V)}));
        }
        Mesh->SetArrayField(TEXT("neutral"),Neutral);
        Mesh->SetArrayField(TEXT("layouts"),Layouts);
        const int32 InfluenceCount = Reader->GetMaximumInfluencePerVertex(M);
        TArray<uint8> WeightBytes, JointBytes;
        WeightBytes.Init(0,Reader->GetVertexPositionCount(M)*InfluenceCount*sizeof(float));
        JointBytes.Init(0,Reader->GetVertexPositionCount(M)*InfluenceCount*sizeof(uint16));
        for (uint32 V=0; V<Reader->GetVertexPositionCount(M); ++V)
        {
            const auto Weights = Reader->GetSkinWeightsValues(M,V);
            const auto Joints = Reader->GetSkinWeightsJointIndices(M,V);
            if (Weights.Num()>InfluenceCount || Weights.Num()!=Joints.Num()) return false;
            FMemory::Memcpy(WeightBytes.GetData()+V*InfluenceCount*sizeof(float),Weights.GetData(),Weights.Num()*sizeof(float));
            FMemory::Memcpy(JointBytes.GetData()+V*InfluenceCount*sizeof(uint16),Joints.GetData(),Joints.Num()*sizeof(uint16));
        }
        const FString WeightsFile = OutputFile+FString::Printf(TEXT(".weights_%d.f32"),M);
        const FString JointsFile = OutputFile+FString::Printf(TEXT(".joints_%d.u16"),M);
        if (!FFileHelper::SaveArrayToFile(WeightBytes,*WeightsFile) || !FFileHelper::SaveArrayToFile(JointBytes,*JointsFile)) return false;
        Mesh->SetNumberField(TEXT("influence_count"),InfluenceCount);
        Mesh->SetStringField(TEXT("weights_file"),FPaths::GetCleanFilename(WeightsFile));
        Mesh->SetStringField(TEXT("joints_file"),FPaths::GetCleanFilename(JointsFile));
        Meshes.Add(MakeShared<FJsonValueObject>(Mesh));
    }
    Root->SetArrayField(TEXT("meshes"),Meshes);
    if (Poses->Num())
    {
        if (!Reader->GetRawControlCount() || !Reader->GetJointGroupCount())
        {
            Root->SetStringField(TEXT("error"),TEXT("DNA contains no evaluable facial behavior"));
            WriteJSON(OutputFile,Root);
            return false;
        }
        if (!NativeRig) NativeRig=MakeShared<FRigLogic>(Reader.Get());
        FRigLogic& Rig=*NativeRig;
        const auto Neutral = Rig.GetNeutralJointValues();
        if (Neutral.Num() != Reader->GetJointCount()*10) return false;
        TArray<TSharedPtr<FJsonValue>> NeutralValues;
        for (const float Value : Neutral) NeutralValues.Add(MakeShared<FJsonValueNumber>(Value));
        Root->SetArrayField(TEXT("neutral_joint_values"),NeutralValues);
        auto Bind = JointMatrices(*Reader,Neutral,{});
        // MHC's geometry wrapper swizzles Maya geometry into UE while its
        // behavior retains Legacy-space joint frames. Bring both bind and
        // animated world matrices into the geometry frame before skinning.
        const FMatrix LegacyToGeometry(FPlane(1,0,0,0),FPlane(0,0,-1,0),FPlane(0,1,0,0),FPlane(0,0,0,1));
        if (bMetaHumanCompatibility) for (auto& Matrix : Bind) Matrix*=LegacyToGeometry;
        int32 PoseIndex = 0;
        for (const auto& PoseValue : *Poses)
        {
            const auto Pose = PoseValue->AsObject();
            if (!Pose.IsValid()) return false;
            const TSharedPtr<FJsonObject>* Requested = nullptr;
            if (!Pose->TryGetObjectField(TEXT("controls"),Requested)) return false;
            FRigInstance Instance(&Rig);
            Instance.SetLOD(0);
            for (const auto& Pair : (*Requested)->Values)
            {
                const int32* Index = Controls.Find(FString(Pair.Key.ToView()));
                double Value = 0;
                if (!Index || !Pair.Value->TryGetNumber(Value) || !FMath::IsFinite(Value) || Value<0 || Value>1) return false;
                Instance.SetRawControl(*Index,Value);
            }
            Rig.Calculate(&Instance);
            TArray<uint8> JointValueBytes;
            const auto JointValues=Instance.GetJointOutputs();
            JointValueBytes.SetNumUninitialized(JointValues.Num()*sizeof(float));
            FMemory::Memcpy(JointValueBytes.GetData(),JointValues.GetData(),JointValueBytes.Num());
            const FString JointValueFile=OutputFile+FString::Printf(TEXT(".local_joints_%d.f32"),PoseIndex);
            if (!FFileHelper::SaveArrayToFile(JointValueBytes,*JointValueFile)) return false;
            auto Animated = JointMatrices(*Reader,Neutral,Instance.GetJointOutputs());
            if (bMetaHumanCompatibility) for (auto& Matrix : Animated) Matrix*=LegacyToGeometry;
            TArray<FMatrix> Skin;
            for (int32 J=0; J<Bind.Num(); ++J) Skin.Add(Bind[J].Inverse()*Animated[J]);
            TArray<uint8> MatrixBytes;
            MatrixBytes.SetNumUninitialized(Skin.Num()*16*sizeof(float));
            for (int32 J=0; J<Skin.Num(); ++J)
                for (int32 R=0; R<4; ++R)
                    for (int32 C=0; C<4; ++C)
                    {
                        const float Value = float(Skin[J].M[R][C]);
                        FMemory::Memcpy(MatrixBytes.GetData()+(J*16+R*4+C)*sizeof(float),&Value,sizeof(Value));
                    }
            const FString MatrixFile = OutputFile+FString::Printf(TEXT(".skin_%d.f32"),PoseIndex);
            if (!FFileHelper::SaveArrayToFile(MatrixBytes,*MatrixFile)) return false;
            const auto Blend = Instance.GetBlendShapeOutputs();
            TArray<TSharedPtr<FJsonValue>> PoseMeshes;
            for (const uint16 M : MeshIndices)
            {
                TArray<FVector> Points;
                for (uint32 V=0; V<Reader->GetVertexPositionCount(M); ++V) Points.Add(Reader->GetVertexPosition(M,V));
                for (uint16 B=0; B<Reader->GetBlendShapeTargetCount(M); ++B)
                {
                    const uint16 Channel = Reader->GetBlendShapeChannelIndex(M,B);
                    if (!Blend.IsValidIndex(Channel) || FMath::Abs(Blend[Channel])<1.e-8) continue;
                    const auto Indices = Reader->GetBlendShapeTargetVertexIndices(M,B);
                    for (int32 D=0; D<Indices.Num(); ++D) Points[Indices[D]] += Blend[Channel]*Reader->GetBlendShapeTargetDelta(M,B,D);
                }
                TArray<uint8> VertexBytes;
                VertexBytes.SetNumUninitialized(Points.Num()*3*sizeof(float));
                for (int32 V=0; V<Points.Num(); ++V)
                {
                    const auto Weights = Reader->GetSkinWeightsValues(M,V);
                    const auto Joints = Reader->GetSkinWeightsJointIndices(M,V);
                    FVector P = FVector::ZeroVector;
                    if (!Weights.Num()) P = Points[V];
                    for (int32 W=0; W<Weights.Num(); ++W) P += FVector(Skin[Joints[W]].TransformPosition(Points[V]))*Weights[W];
                    const float XYZ[3] = {float(P.X),float(P.Y),float(P.Z)};
                    FMemory::Memcpy(VertexBytes.GetData()+V*3*sizeof(float),XYZ,sizeof(XYZ));
                }
                TSharedRef<FJsonObject> Mesh = MakeShared<FJsonObject>();
                Mesh->SetNumberField(TEXT("index"),M);
                const FString VertexFile = OutputFile + FString::Printf(TEXT(".pose_%d_mesh_%d.f32"),PoseIndex,M);
                if (!FFileHelper::SaveArrayToFile(VertexBytes,*VertexFile)) return false;
                Mesh->SetStringField(TEXT("vertices_file"),FPaths::GetCleanFilename(VertexFile));
                Mesh->SetNumberField(TEXT("vertex_count"),Points.Num());
                PoseMeshes.Add(MakeShared<FJsonValueObject>(Mesh));
            }
            TSharedRef<FJsonObject> Result = MakeShared<FJsonObject>();
            Result->SetStringField(TEXT("name"),Pose->GetStringField(TEXT("name")));
            Result->SetStringField(TEXT("skin_matrices_file"),FPaths::GetCleanFilename(MatrixFile));
            Result->SetStringField(TEXT("joint_values_file"),FPaths::GetCleanFilename(JointValueFile));
            Result->SetObjectField(TEXT("controls"),*Requested);
            TArray<TSharedPtr<FJsonValue>> AnimatedMapValues;
            for (const float Value : Instance.GetAnimatedMapOutputs())
                AnimatedMapValues.Add(MakeShared<FJsonValueNumber>(Value));
            Result->SetArrayField(TEXT("animated_maps"),AnimatedMapValues);
            Result->SetArrayField(TEXT("meshes"),PoseMeshes);
            Results.Add(MakeShared<FJsonValueObject>(Result));
            ++PoseIndex;
        }
    }
    Root->SetArrayField(TEXT("poses"),Results);
    return WriteJSON(OutputFile,Root);
}
