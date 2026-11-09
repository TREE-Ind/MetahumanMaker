#include "CreateMHCalibrationLibrary.h"
#include "DNAUtils.h"
#include "DNAReader.h"
#include "RigLogic.h"
#include "RigInstance.h"
#include "Dom/JsonObject.h"
#include "Serialization/JsonReader.h"
#include "Serialization/JsonSerializer.h"
#include "Misc/FileHelper.h"
#include "Misc/Paths.h"
#include "Misc/ScopeExit.h"
#include "HAL/FileManager.h"
#include "dna/BinaryStreamWriter.h"
#include "dna/Reader.h"

bool UCreateMHCalibrationLibrary::WritePoseCorrectiveDNA(const FString& DNAFile,const FString& RequestFile,const FString& OutputFile)
{
    if (FPaths::ConvertRelativePathToFull(DNAFile)==FPaths::ConvertRelativePathToFull(OutputFile)||IFileManager::Get().FileExists(*OutputFile)) return false;
    FString Text;TSharedPtr<FJsonObject> Request;
    if (!FFileHelper::LoadFileToString(Text,*RequestFile)||!FJsonSerializer::Deserialize(TJsonReaderFactory<>::Create(Text),Request)||!Request) return false;
    FString Schema;double Bound=0;const TArray<TSharedPtr<FJsonValue>>* Targets=nullptr;const TArray<TSharedPtr<FJsonValue>>* Solvers=nullptr;
    if (!Request->TryGetStringField(TEXT("schema"),Schema)||Schema!=TEXT("CM_DNA_POSE_CORRECTIVES_V1")||!Request->TryGetNumberField(TEXT("maximum_offset"),Bound)||!FMath::IsFinite(Bound)||Bound<=0||
        !Request->TryGetArrayField(TEXT("targets"),Targets)||Targets->Num()<1||Targets->Num()>64||!Request->TryGetArrayField(TEXT("solvers"),Solvers)||Solvers->Num()<1||Solvers->Num()>16) return false;
    FDNAConfig Config;Config.CoordinateSystemTransformPolicy=ECoordinateSystemTransformPolicy::Preserve;Config.MaxLODPerPlatform=FPerPlatformInt(0);Config.MinLODPerPlatform=FPerPlatformInt(-1);
    auto Reader=LoadDNAFromFile(DNAFile,Config);if (!Reader||!Reader->Unwrap()) return false;auto R=Reader->Unwrap();
    const uint32 ControlBase=R->getRawControlCount()+R->getPSDCount()+R->getMLControlCount()+R->getRBFPoseControlCount();
    if (!R->getLODCount()||ControlBase+Targets->Num()>65535||R->getBlendShapeChannelCount()+Targets->Num()>65535||R->getRBFSolverCount()+Solvers->Num()>65535) return false;
    struct FTarget {FString Name;uint16 Mesh,Target,Channel,Control;TArray<uint32> Indices;TArray<dna::Vector3> Deltas;};
    struct FPose {FString Name;TArray<float> Values;TArray<uint16> Multipliers;float Scale=1.f;int32 Target=INDEX_NONE;};
    struct FSolver {FString Name;TArray<uint16> Inputs;TMap<uint16,float> ExternalGates;TArray<FPose> Poses;float Radius=0;dna::RBFFunctionType Function=dna::RBFFunctionType::Gaussian;};
    TArray<FTarget> Patches;TArray<FSolver> Prepared;TMap<FString,int32> Names;TMap<uint16,uint16> MeshTargetCounts;TSet<int32> Driven;
    for (const auto& Value:*Targets)
    {
        auto Row=Value->AsObject();FString Mesh,Name,File;
        if (!Row||!Row->TryGetStringField(TEXT("mesh"),Mesh)||!Row->TryGetStringField(TEXT("channel"),Name)||!Name.StartsWith(TEXT("CM_"))||Names.Contains(Name)||!Row->TryGetStringField(TEXT("file"),File)||File!=FPaths::GetCleanFilename(File)) return false;
        for (uint16 C=0;C<R->getBlendShapeChannelCount();++C) if (FString(UTF8_TO_TCHAR(R->getBlendShapeChannelName(C).data()))==Name) return false;
        int32 M=INDEX_NONE;for (uint16 I=0;I<R->getMeshCount();++I) if (FString(UTF8_TO_TCHAR(R->getMeshName(I).data()))==Mesh) M=I;
        if (M==INDEX_NONE) return false;
        FTarget Patch;Patch.Name=Name;Patch.Mesh=M;Patch.Target=MeshTargetCounts.FindOrAdd(M,R->getBlendShapeTargetCount(M))++;
        Patch.Channel=R->getBlendShapeChannelCount()+Patches.Num();Patch.Control=ControlBase+Patches.Num();
        TArray<uint8> Bytes;const TArray<TSharedPtr<FJsonValue>>* Indices=nullptr;
        if (!FFileHelper::LoadFileToArray(Bytes,*FPaths::Combine(FPaths::GetPath(RequestFile),File))||!Bytes.Num()||Bytes.Num()%12||!Row->TryGetArrayField(TEXT("indices"),Indices)||Indices->Num()!=Bytes.Num()/12) return false;
        Patch.Deltas.SetNumUninitialized(Bytes.Num()/12);FMemory::Memcpy(Patch.Deltas.GetData(),Bytes.GetData(),Bytes.Num());TSet<uint32> Seen;
        for (int32 I=0;I<Indices->Num();++I)
        {
            double N=0;const auto D=Patch.Deltas[I];
            if (!(*Indices)[I]->TryGetNumber(N)||!FMath::IsFinite(N)||N<0||N>=R->getVertexPositionCount(M)||N!=FMath::FloorToDouble(N)||Seen.Contains(uint32(N))||!FMath::IsFinite(D.x)||!FMath::IsFinite(D.y)||!FMath::IsFinite(D.z)||FVector(D.x,D.y,D.z).Size()>Bound) return false;
            Seen.Add(uint32(N));Patch.Indices.Add(uint32(N));
        }
        Names.Add(Name,Patches.Num());Patches.Add(MoveTemp(Patch));
    }
    TSet<FString> SolverNames,PoseNames;int32 NewPoseCount=0;
    for (const auto& Value:*Solvers)
    {
        auto Row=Value->AsObject();FSolver Solver;double Radius=0;const TArray<TSharedPtr<FJsonValue>>* Inputs=nullptr;const TArray<TSharedPtr<FJsonValue>>* Poses=nullptr;
        if (!Row||!Row->TryGetStringField(TEXT("name"),Solver.Name)||!Solver.Name.StartsWith(TEXT("CM_"))||SolverNames.Contains(Solver.Name)||!Row->TryGetNumberField(TEXT("radius"),Radius)||!FMath::IsFinite(Radius)||Radius<=0||!Row->TryGetArrayField(TEXT("inputs"),Inputs)||Inputs->Num()<1||Inputs->Num()>16||!Row->TryGetArrayField(TEXT("poses"),Poses)||Poses->Num()<2||Poses->Num()>64) return false;
        for (uint16 I=0;I<R->getRBFSolverCount();++I) if (FString(UTF8_TO_TCHAR(R->getRBFSolverName(I).data()))==Solver.Name) return false;
        SolverNames.Add(Solver.Name);Solver.Radius=Radius;
        FString Function;
        if (Row->HasField(TEXT("function")))
        {
            if (!Row->TryGetStringField(TEXT("function"),Function)) return false;
            if (Function==TEXT("LINEAR")) Solver.Function=dna::RBFFunctionType::Linear;
            else if (Function!=TEXT("GAUSSIAN")) return false;
        }
        for (const auto& Input:*Inputs)
        {
            FString Name;if (!Input->TryGetString(Name)) return false;int32 Found=INDEX_NONE;
            for (uint16 I=0;I<R->getRawControlCount();++I) if (FString(UTF8_TO_TCHAR(R->getRawControlName(I).data()))==Name) Found=I;
            if (Found==INDEX_NONE||Solver.Inputs.Contains(uint16(Found))) return false;Solver.Inputs.Add(Found);
        }
        // A gate is not a distance coordinate. Keeping it outside the RBF
        // domain lets paired positive/negative fields cancel structurally at
        // gate=1, instead of changing their interpolation as the gate moves.
        if (Row->HasField(TEXT("gate_values")))
        {
            const TSharedPtr<FJsonObject>* Gates=nullptr;
            if (!Row->TryGetObjectField(TEXT("gate_values"),Gates)||(*Gates)->Values.Num()<1||(*Gates)->Values.Num()+Solver.Inputs.Num()>16) return false;
            for (const auto& Pair:(*Gates)->Values)
            {
                double Center=0;int32 Found=INDEX_NONE;
                for (uint16 I=0;I<R->getRawControlCount();++I) if (FString(UTF8_TO_TCHAR(R->getRawControlName(I).data()))==FString(Pair.Key.ToView())) Found=I;
                if (Found==INDEX_NONE||Solver.Inputs.Contains(uint16(Found))||!Pair.Value->TryGetNumber(Center)||!FMath::IsFinite(Center)||Center<=0||Center>1) return false;
                Solver.ExternalGates.Add(uint16(Found),float(Center));
            }
        }
        for (const auto& Entry:*Poses)
        {
            auto PoseRow=Entry->AsObject();FPose Pose;const TArray<TSharedPtr<FJsonValue>>* Values=nullptr;FString Channel;
            if (!PoseRow||!PoseRow->TryGetStringField(TEXT("name"),Pose.Name)||!Pose.Name.StartsWith(TEXT("CM_"))||PoseNames.Contains(Pose.Name)||!PoseRow->TryGetArrayField(TEXT("values"),Values)||Values->Num()!=Solver.Inputs.Num()) return false;
            for (uint16 I=0;I<R->getRBFPoseCount();++I) if (FString(UTF8_TO_TCHAR(R->getRBFPoseName(I).data()))==Pose.Name) return false;
            PoseNames.Add(Pose.Name);
            for (const auto& Number:*Values) {double N=0;if (!Number->TryGetNumber(N)||!FMath::IsFinite(N)||N<0||N>1) return false;Pose.Values.Add(N);}
            for (const auto& Other:Solver.Poses) if (Other.Values==Pose.Values) return false;
            if (PoseRow->TryGetStringField(TEXT("channel"),Channel)) {if (!Names.Contains(Channel)) return false;Pose.Target=Names[Channel];Driven.Add(Pose.Target);}
            double Scale=1.;if (PoseRow->HasField(TEXT("scale"))&&!PoseRow->TryGetNumberField(TEXT("scale"),Scale)) return false;
            if (!FMath::IsFinite(Scale)||Scale<=0||Scale>64) return false;Pose.Scale=Scale;
            const TArray<TSharedPtr<FJsonValue>>* Multipliers=nullptr;
            if (PoseRow->HasField(TEXT("multipliers")))
            {
                if (!PoseRow->TryGetArrayField(TEXT("multipliers"),Multipliers)||Multipliers->Num()<1||Multipliers->Num()>Solver.Inputs.Num()+Solver.ExternalGates.Num()||Pose.Target==INDEX_NONE) return false;
                double Center=Scale;
                for (const auto& Multiplier:*Multipliers)
                {
                    FString Name;if (!Multiplier->TryGetString(Name)) return false;int32 Input=INDEX_NONE;uint16 Raw=0;double GateCenter=0;
                    for (int32 I=0;I<Solver.Inputs.Num();++I) if (FString(UTF8_TO_TCHAR(R->getRawControlName(Solver.Inputs[I]).data()))==Name) Input=I;
                    if (Input!=INDEX_NONE) {Raw=Solver.Inputs[Input];GateCenter=Pose.Values[Input];}
                    else
                    {
                        bool Found=false;
                        for (const auto& Gate:Solver.ExternalGates) if (FString(UTF8_TO_TCHAR(R->getRawControlName(Gate.Key).data()))==Name) {Raw=Gate.Key;GateCenter=Gate.Value;Found=true;break;}
                        if (!Found) return false;
                    }
                    if (Pose.Multipliers.Contains(Raw)) return false;
                    Pose.Multipliers.Add(Raw);Center*=GateCenter;
                }
                // Keep the existing native center validation meaningful: a
                // gated pose must still produce unit output at its own knot.
                if (FMath::Abs(Center-1.)>1.e-6) return false;
            }
            else if (Scale!=1.) return false;
            Solver.Poses.Add(MoveTemp(Pose));++NewPoseCount;
        }
        // UE 5.8's coordinate converter treats any RBF target-value array
        // divisible by four as packed quaternions, even for scalar Euclidean
        // solvers. Refuse that layout rather than silently swizzling controls.
        // Callers can supply a meaningful additional zero-output anchor.
        if ((Solver.Inputs.Num()*Solver.Poses.Num())%4==0)
        {
            UE_LOG(LogTemp,Error,TEXT("Scalar RBF requires a non-quaternion-packed pose layout on UE 5.8; add a zero-output anchor: %s"),*Solver.Name);
            return false;
        }
        Prepared.Add(MoveTemp(Solver));
    }
    if (Driven.Num()!=Patches.Num()||R->getRBFPoseCount()+NewPoseCount>65535) return false;
    const auto OldInput=R->getBlendShapeChannelInputIndices();const auto OldOutput=R->getBlendShapeChannelOutputIndices();const auto OldLODs=R->getBlendShapeChannelLODs();
    // Native behavior uses prefix LOD bounds. Append only after its complete
    // LOD0 prefix; lower LOD prefixes and all old row/control indices stay exact.
    if (OldInput.size()!=OldOutput.size()||OldLODs.size()!=R->getLODCount()||OldLODs[0]!=OldInput.size()||OldInput.size()+Patches.Num()>65535) return false;
    auto Stream=dna::FileStream::create(TCHAR_TO_UTF8(*OutputFile),trio::AccessMode::Write,trio::OpenMode::Binary);if (!Stream) return false;
    auto W=dna::BinaryStreamWriter::create(Stream);ON_SCOPE_EXIT {if (W) dna::BinaryStreamWriter::destroy(W);dna::FileStream::destroy(Stream);};if (!W) return false;W->setFrom(R);
    TArray<uint16> BInputs,BOutputs,BLODs;for (auto V:OldInput) BInputs.Add(V);for (auto V:OldOutput) BOutputs.Add(V);for (auto V:OldLODs) BLODs.Add(V);BLODs[0]+=Patches.Num();
    uint32 Mapping=R->getMeshBlendShapeChannelMappingCount();
    for (int32 I=0;I<Patches.Num();++I)
    {
        const auto& P=Patches[I];W->setBlendShapeChannelName(P.Channel,TCHAR_TO_UTF8(*P.Name));W->setBlendShapeChannelIndex(P.Mesh,P.Target,P.Channel);
        W->setBlendShapeTargetVertexIndices(P.Mesh,P.Target,P.Indices.GetData(),P.Indices.Num());W->setBlendShapeTargetDeltas(P.Mesh,P.Target,P.Deltas.GetData(),P.Deltas.Num());
        W->setMeshBlendShapeChannelMapping(Mapping++,P.Mesh,P.Channel);W->setRBFPoseControlName(R->getRBFPoseControlCount()+I,TCHAR_TO_UTF8(*P.Name));BInputs.Add(P.Control);BOutputs.Add(P.Channel);
    }
    W->setBlendShapeChannelInputIndices(BInputs.GetData(),BInputs.Num());W->setBlendShapeChannelOutputIndices(BOutputs.GetData(),BOutputs.Num());W->setBlendShapeChannelLODs(BLODs.GetData(),BLODs.Num());
    for (uint16 LOD=0;LOD<R->getLODCount();++LOD)
    {
        TArray<uint16> Channels,Indices;for (auto C:R->getBlendShapeChannelIndicesForLOD(LOD)) Channels.Add(C);for (auto S:R->getRBFSolverIndicesForLOD(LOD)) Indices.Add(S);
        if (!LOD) {for (const auto& P:Patches) Channels.Add(P.Channel);for (int32 S=0;S<Prepared.Num();++S) Indices.Add(R->getRBFSolverCount()+S);}
        W->setBlendShapeChannelIndices(LOD,Channels.GetData(),Channels.Num());W->setLODBlendShapeChannelMapping(LOD,LOD);W->setRBFSolverIndices(LOD,Indices.GetData(),Indices.Num());W->setLODRBFSolverMapping(LOD,LOD);
    }
    uint16 PoseIndex=R->getRBFPoseCount();
    for (int32 I=0;I<Prepared.Num();++I)
    {
        const auto& S=Prepared[I];const uint16 Index=R->getRBFSolverCount()+I;TArray<uint16> PoseIndices;TArray<float> Values;
        W->setRBFSolverName(Index,TCHAR_TO_UTF8(*S.Name));W->setRBFSolverRawControlIndices(Index,S.Inputs.GetData(),S.Inputs.Num());
        for (const auto& P:S.Poses)
        {
            PoseIndices.Add(PoseIndex);Values.Append(P.Values);W->setRBFPoseName(PoseIndex,TCHAR_TO_UTF8(*P.Name));W->setRBFPoseScale(PoseIndex,1.f);W->setRBFPoseInputControlIndices(PoseIndex,P.Multipliers.GetData(),P.Multipliers.Num());
            if (P.Target!=INDEX_NONE) {const uint16 Control=Patches[P.Target].Control;const float Weight=P.Scale;W->setRBFPoseOutputControlIndices(PoseIndex,&Control,1);W->setRBFPoseOutputControlWeights(PoseIndex,&Weight,1);}
            else {W->setRBFPoseOutputControlIndices(PoseIndex,nullptr,0);W->setRBFPoseOutputControlWeights(PoseIndex,nullptr,0);}
            ++PoseIndex;
        }
        W->setRBFSolverPoseIndices(Index,PoseIndices.GetData(),PoseIndices.Num());W->setRBFSolverRawControlValues(Index,Values.GetData(),Values.Num());W->setRBFSolverType(Index,dna::RBFSolverType::Interpolative);
        // The native factory applies the DNA rotation-unit conversion to all
        // radii, including Euclidean ones. Store the requested scalar radius
        // in that convention so runtime interpolation keeps its intended width.
        const float Radius=R->getRotationUnit()==dna::RotationUnit::degrees?FMath::RadiansToDegrees(S.Radius):S.Radius;
        W->setRBFSolverRadius(Index,Radius);W->setRBFSolverAutomaticRadius(Index,dna::AutomaticRadius::Off);W->setRBFSolverWeightThreshold(Index,1.e-6f);W->setRBFSolverDistanceMethod(Index,dna::RBFDistanceMethod::Euclidean);
        W->setRBFSolverNormalizeMethod(Index,dna::RBFNormalizeMethod::OnlyNormalizeAboveOne);W->setRBFSolverFunctionType(Index,S.Function);W->setRBFSolverTwistAxis(Index,dna::TwistAxis::X);
    }
    W->write();if (!dna::Status::isOk()) return false;
    // Check the serialized drivers with the native evaluator before allowing
    // the caller to bind this DNA to any candidate mesh.
    auto Check=LoadDNAFromFile(OutputFile,Config);if (!Check) return false;
    FRigLogic Rig(Check.Get());TArray<TSharedPtr<FJsonValue>> Samples;bool Valid=true;
    int32 LinearSolvers=0;
    for (int32 I=0;I<Prepared.Num();++I)
    {
        if (Prepared[I].Function==dna::RBFFunctionType::Linear) ++LinearSolvers;
        if (Check->Unwrap()->getRBFSolverFunctionType(R->getRBFSolverCount()+I)!=Prepared[I].Function) Valid=false;
    }
    for (const auto& S:Prepared) for (const auto& P:S.Poses)
    {
        FRigInstance Instance(&Rig);Instance.SetLOD(0);
        for (int32 I=0;I<S.Inputs.Num();++I) Instance.SetRawControl(S.Inputs[I],P.Values[I]);
        for (const auto& Gate:S.ExternalGates) Instance.SetRawControl(Gate.Key,Gate.Value);
        Rig.Calculate(&Instance);auto Row=MakeShared<FJsonObject>();Row->SetStringField(TEXT("pose"),P.Name);auto Weights=MakeShared<FJsonObject>();
        const auto Output=Instance.GetBlendShapeOutputs();
        for (int32 I=0;I<Patches.Num();++I)
        {
            const auto& Patch=Patches[I];const float Actual=Output.IsValidIndex(Patch.Channel)?Output[Patch.Channel]:-1.f;
            Weights->SetNumberField(Patch.Name,Actual);
            // Other solvers can also be active at these same control values;
            // only check targets belonging to the sampled solver here.
            bool Owned=false;for (const auto& Other:S.Poses) Owned|=Other.Target==I;
            if (Owned&&FMath::Abs(Actual-(P.Target==I?1.f:0.f))>1.e-4f) Valid=false;
        }
        Row->SetObjectField(TEXT("morph_weights"),Weights);Samples.Add(MakeShared<FJsonValueObject>(Row));
    }
    int32 GatedPoses=0,ZeroGateSamples=0;
    for (const auto& S:Prepared) for (const auto& P:S.Poses) if (P.Multipliers.Num())
    {
        ++GatedPoses;
        for (const auto Gate:P.Multipliers)
        {
            FRigInstance Instance(&Rig);Instance.SetLOD(0);
            for (const auto Input:S.Inputs) Instance.SetRawControl(Input,.37f);
            for (const auto& External:S.ExternalGates) Instance.SetRawControl(External.Key,.37f);
            Instance.SetRawControl(Gate,0.f);Rig.Calculate(&Instance);++ZeroGateSamples;
            const auto Output=Instance.GetBlendShapeOutputs();const auto Channel=Patches[P.Target].Channel;
            if (!Output.IsValidIndex(Channel)||FMath::Abs(Output[Channel])>1.e-7f) Valid=false;
        }
    }
    int32 ExternalGateControls=0;for (const auto& S:Prepared) ExternalGateControls+=S.ExternalGates.Num();
    auto Receipt=MakeShared<FJsonObject>();Receipt->SetBoolField(TEXT("native_driver_centers_valid"),Valid);Receipt->SetArrayField(TEXT("samples"),Samples);Receipt->SetNumberField(TEXT("multiplicatively_gated_poses"),GatedPoses);Receipt->SetNumberField(TEXT("native_zero_gate_samples"),ZeroGateSamples);Receipt->SetNumberField(TEXT("linear_solvers"),LinearSolvers);Receipt->SetNumberField(TEXT("external_gate_controls"),ExternalGateControls);
    FString ReceiptText;if (!FJsonSerializer::Serialize(Receipt,TJsonWriterFactory<>::Create(&ReceiptText))||!FFileHelper::SaveStringToFile(ReceiptText,*(OutputFile+TEXT(".drivers.json")))) return false;
    return Valid;
}
