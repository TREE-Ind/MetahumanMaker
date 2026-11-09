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

bool UCreateMHCalibrationLibrary::WriteDrivenMeshTargetsDNA(const FString& DNAFile,const FString& RequestFile,const FString& OutputFile)
{
    if (FPaths::ConvertRelativePathToFull(DNAFile)==FPaths::ConvertRelativePathToFull(OutputFile)||IFileManager::Get().FileExists(*OutputFile)) return false;
    FString Text,Schema;TSharedPtr<FJsonObject> Request;double Bound=0;const TArray<TSharedPtr<FJsonValue>>* Rows=nullptr;
    if (!FFileHelper::LoadFileToString(Text,*RequestFile)||!FJsonSerializer::Deserialize(TJsonReaderFactory<>::Create(Text),Request)||!Request||
        !Request->TryGetStringField(TEXT("schema"),Schema)||Schema!=TEXT("CM_DNA_DRIVEN_TARGETS_V1")||!Request->TryGetNumberField(TEXT("maximum_offset"),Bound)||!FMath::IsFinite(Bound)||Bound<=0||
        !Request->TryGetArrayField(TEXT("targets"),Rows)||Rows->Num()<1||Rows->Num()>4096) return false;
    bool Activate=false;
    if (Request->HasField(TEXT("activate_owned_drivers"))&&!Request->TryGetBoolField(TEXT("activate_owned_drivers"),Activate)) return false;
    FDNAConfig Config;Config.CoordinateSystemTransformPolicy=ECoordinateSystemTransformPolicy::Preserve;Config.MaxLODPerPlatform=FPerPlatformInt(0);Config.MinLODPerPlatform=FPerPlatformInt(-1);
    auto Reader=LoadDNAFromFile(DNAFile,Config);if (!Reader||!Reader->Unwrap()) return false;auto R=Reader->Unwrap();
    const uint32 ControlBase=R->getRawControlCount()+R->getPSDCount()+R->getMLControlCount();
    struct FPatch {uint16 Mesh,Target,Channel,LOD;TArray<uint32> Indices;TArray<dna::Vector3> Deltas;};
    TArray<FPatch> Patches;TSet<FString> Seen;TMap<uint16,uint16> Counts,ChannelLOD,SolverLOD;
    const auto Inputs=R->getBlendShapeChannelInputIndices();const auto Outputs=R->getBlendShapeChannelOutputIndices();
    if (Inputs.size()!=Outputs.size()||R->getMeshBlendShapeChannelMappingCount()+Rows->Num()>65535) return false;
    for (const auto& Value:*Rows)
    {
        auto Row=Value->AsObject();FString Mesh,Channel,File;double LOD=0;
        if (!Row||!Row->TryGetStringField(TEXT("mesh"),Mesh)||!Row->TryGetStringField(TEXT("channel"),Channel)||!Channel.StartsWith(TEXT("CM_"))||
            !Row->TryGetStringField(TEXT("file"),File)||File!=FPaths::GetCleanFilename(File)||Seen.Contains(Mesh+TEXT("/")+Channel)) return false;
        if (Row->HasField(TEXT("lod"))&&!Row->TryGetNumberField(TEXT("lod"),LOD)) return false;
        if (!FMath::IsFinite(LOD)||LOD<0||LOD>=R->getLODCount()||LOD!=FMath::FloorToDouble(LOD)) return false;
        int32 M=INDEX_NONE,C=INDEX_NONE;
        for (uint16 I=0;I<R->getMeshCount();++I) if (FString(UTF8_TO_TCHAR(R->getMeshName(I).data()))==Mesh) M=I;
        for (uint16 I=0;I<R->getBlendShapeChannelCount();++I) if (FString(UTF8_TO_TCHAR(R->getBlendShapeChannelName(I).data()))==Channel) C=I;
        if (M==INDEX_NONE||C==INDEX_NONE) return false;
        bool ActiveMesh=false,ActiveChannel=false,OwnedDriver=false;
        for (auto I:R->getMeshIndicesForLOD(uint16(LOD))) ActiveMesh|=I==M;
        for (auto I:R->getBlendShapeChannelIndicesForLOD(uint16(LOD))) ActiveChannel|=I==C;
        for (int32 I=0;I<int32(Inputs.size());++I) if (Outputs[I]==C&&Inputs[I]>=ControlBase&&Inputs[I]<ControlBase+R->getRBFPoseControlCount())
            OwnedDriver|=FString(UTF8_TO_TCHAR(R->getRBFPoseControlName(Inputs[I]-ControlBase).data()))==Channel;
        if (!ActiveMesh||(!ActiveChannel&&!Activate)||!OwnedDriver) return false;
        if (Activate) {uint16& Last=ChannelLOD.FindOrAdd(uint16(C),uint16(LOD));Last=FMath::Max(Last,uint16(LOD));}
        for (uint16 T=0;T<R->getBlendShapeTargetCount(M);++T) if (R->getBlendShapeChannelIndex(M,T)==C) return false;
        uint16& Count=Counts.FindOrAdd(M,R->getBlendShapeTargetCount(M));if (Count==65535) return false;
        FPatch Patch;Patch.Mesh=M;Patch.Target=Count++;Patch.Channel=C;Patch.LOD=uint16(LOD);
        TArray<uint8> Bytes;const TArray<TSharedPtr<FJsonValue>>* Indices=nullptr;
        if (!FFileHelper::LoadFileToArray(Bytes,*FPaths::Combine(FPaths::GetPath(RequestFile),File))||!Bytes.Num()||Bytes.Num()%12||!Row->TryGetArrayField(TEXT("indices"),Indices)||Indices->Num()!=Bytes.Num()/12) return false;
        Patch.Deltas.SetNumUninitialized(Bytes.Num()/12);FMemory::Memcpy(Patch.Deltas.GetData(),Bytes.GetData(),Bytes.Num());TSet<uint32> Vertices;
        for (int32 I=0;I<Indices->Num();++I)
        {
            double N=0;const auto D=Patch.Deltas[I];
            if (!(*Indices)[I]->TryGetNumber(N)||!FMath::IsFinite(N)||N<0||N>=R->getVertexPositionCount(M)||N!=FMath::FloorToDouble(N)||Vertices.Contains(uint32(N))||
                !FMath::IsFinite(D.x)||!FMath::IsFinite(D.y)||!FMath::IsFinite(D.z)||FVector(D.x,D.y,D.z).Size()>Bound) return false;
            Vertices.Add(uint32(N));Patch.Indices.Add(uint32(N));
        }
        Seen.Add(Mesh+TEXT("/")+Channel);Patches.Add(MoveTemp(Patch));
    }
    if (Activate)
    {
        // Only the append-only CM_ solvers that actually own these controls
        // may be activated. Never enable an unrelated native facial solver.
        for (const auto& Entry:ChannelLOD)
        {
            bool Found=false;
            for (uint16 S=0;S<R->getRBFSolverCount();++S)
            {
                if (!FString(UTF8_TO_TCHAR(R->getRBFSolverName(S).data())).StartsWith(TEXT("CM_"))) continue;
                bool Owns=false;
                for (auto P:R->getRBFSolverPoseIndices(S)) for (auto Control:R->getRBFPoseOutputControlIndices(P))
                    for (int32 I=0;I<int32(Inputs.size());++I) Owns|=Outputs[I]==Entry.Key&&Inputs[I]==Control;
                if (Owns) {Found=true;uint16& Last=SolverLOD.FindOrAdd(S,Entry.Value);Last=FMath::Max(Last,Entry.Value);}
            }
            if (!Found) return false;
        }
    }
    auto Stream=dna::FileStream::create(TCHAR_TO_UTF8(*OutputFile),trio::AccessMode::Write,trio::OpenMode::Binary);if (!Stream) return false;
    auto W=dna::BinaryStreamWriter::create(Stream);ON_SCOPE_EXIT {if (W) dna::BinaryStreamWriter::destroy(W);dna::FileStream::destroy(Stream);};if (!W) return false;
    W->setFrom(R);uint16 Mapping=R->getMeshBlendShapeChannelMappingCount();
    for (const auto& P:Patches)
    {
        W->setBlendShapeChannelIndex(P.Mesh,P.Target,P.Channel);W->setBlendShapeTargetVertexIndices(P.Mesh,P.Target,P.Indices.GetData(),P.Indices.Num());
        W->setBlendShapeTargetDeltas(P.Mesh,P.Target,P.Deltas.GetData(),P.Deltas.Num());W->setMeshBlendShapeChannelMapping(Mapping++,P.Mesh,P.Channel);
    }
    if (Activate)
    {
        // Blend-shape input LODs are prefixes. Extending owned rows therefore
        // requires a stable regrouping by last active LOD, not extending the
        // prefix over every unrelated LOD0-only native row.
        const auto OldLODs=R->getBlendShapeChannelLODs();
        if (OldLODs.size()!=R->getLODCount()||!OldLODs.size()||OldLODs[0]!=Inputs.size()||Inputs.size()>65535) return false;
        TArray<uint16> LastLOD;TArray<int32> Order;
        for (int32 I=0;I<int32(Inputs.size());++I)
        {
            uint16 Last=0;
            for (uint16 L=0;L<R->getLODCount();++L) {if (L&&OldLODs[L]>OldLODs[L-1]) return false;if (I<OldLODs[L]) Last=L;}
            if (const uint16* Extra=ChannelLOD.Find(Outputs[I])) Last=FMath::Max(Last,*Extra);
            LastLOD.Add(Last);Order.Add(I);
        }
        Order.Sort([&](int32 A,int32 B){return LastLOD[A]==LastLOD[B]?A<B:LastLOD[A]>LastLOD[B];});
        TArray<uint16> NewInputs,NewOutputs,NewLODs;
        for (auto I:Order) {NewInputs.Add(Inputs[I]);NewOutputs.Add(Outputs[I]);}
        for (uint16 L=0;L<R->getLODCount();++L)
        {
            uint16 Count=0;for (auto Last:LastLOD) if (Last>=L) ++Count;NewLODs.Add(Count);
            TArray<uint16> Channels,Solvers;
            for (auto V:R->getBlendShapeChannelIndicesForLOD(L)) Channels.Add(V);
            for (auto V:R->getRBFSolverIndicesForLOD(L)) Solvers.Add(V);
            for (const auto& Entry:ChannelLOD) if (Entry.Value>=L) Channels.AddUnique(Entry.Key);
            for (const auto& Entry:SolverLOD) if (Entry.Value>=L) Solvers.AddUnique(Entry.Key);
            W->setBlendShapeChannelIndices(L,Channels.GetData(),Channels.Num());W->setLODBlendShapeChannelMapping(L,L);
            W->setRBFSolverIndices(L,Solvers.GetData(),Solvers.Num());W->setLODRBFSolverMapping(L,L);
        }
        W->setBlendShapeChannelInputIndices(NewInputs.GetData(),NewInputs.Num());W->setBlendShapeChannelOutputIndices(NewOutputs.GetData(),NewOutputs.Num());W->setBlendShapeChannelLODs(NewLODs.GetData(),NewLODs.Num());
    }
    W->write();if (!dna::Status::isOk()) return false;
    auto Checked=LoadDNAFromFile(OutputFile,Config);if (!Checked||!Checked->Unwrap()) return false;auto C=Checked->Unwrap();
    if (R->getBlendShapeChannelCount()!=C->getBlendShapeChannelCount()||R->getRBFPoseControlCount()!=C->getRBFPoseControlCount()||R->getRBFSolverCount()!=C->getRBFSolverCount()||R->getRBFPoseCount()!=C->getRBFPoseCount()||R->getJointCount()!=C->getJointCount()||R->getMeshCount()!=C->getMeshCount()) return false;
    for (const auto& P:Patches)
    {
        bool Found=false;
        for (auto I:C->getMeshBlendShapeChannelMappingIndicesForLOD(P.LOD)) {auto Pair=C->getMeshBlendShapeChannelMapping(I);Found|=Pair.meshIndex==P.Mesh&&Pair.blendShapeChannelIndex==P.Channel;}
        if (!Found||C->getBlendShapeChannelIndex(P.Mesh,P.Target)!=P.Channel) return false;
    }
    // Evaluate the existing corrective centers before/after serialization.
    // Native channels/joints stay exact. Explicitly activated CM_ channels
    // must reproduce their own LOD0 outputs at the same control values.
    FRigLogic Before(Reader.Get()),After(Checked.Get());float Maximum=0;int32 Samples=0;
    auto Compare=[&](uint16 LOD,const TArray<uint16>& Controls,const TArray<float>& Values)
    {
        if (Controls.Num()!=Values.Num()) return false;
        FRigInstance A(&Before),B(&After),Reference(&Before);A.SetLOD(LOD);B.SetLOD(LOD);Reference.SetLOD(0);
        for (int32 I=0;I<Controls.Num();++I) {A.SetRawControl(Controls[I],Values[I]);B.SetRawControl(Controls[I],Values[I]);Reference.SetRawControl(Controls[I],Values[I]);}
        Before.Calculate(&A);After.Calculate(&B);Before.Calculate(&Reference);++Samples;
        auto AV=A.GetBlendShapeOutputs(),BV=B.GetBlendShapeOutputs();if (AV.Num()!=BV.Num()) return false;
        auto Expected=Reference.GetBlendShapeOutputs();if (Expected.Num()!=AV.Num()) return false;
        for (int32 I=0;I<AV.Num();++I)
        {
            const uint16* Last=ChannelLOD.Find(uint16(I));const float Want=Last&&LOD<=*Last?Expected[I]:AV[I];
            Maximum=FMath::Max(Maximum,FMath::Abs(Want-BV[I]));
        }
        auto AJ=A.GetJointOutputs(),BJ=B.GetJointOutputs();if (AJ.Num()!=BJ.Num()) return false;
        for (int32 I=0;I<AJ.Num();++I) Maximum=FMath::Max(Maximum,FMath::Abs(AJ[I]-BJ[I]));
        return FMath::IsFinite(Maximum)&&Maximum<=1.e-6f;
    };
    for (uint16 LOD=0;LOD<(Activate?R->getLODCount():1);++LOD)
    {
      if (!Compare(LOD,TArray<uint16>{},TArray<float>{})) return false;
      for (uint16 S=0;S<R->getRBFSolverCount();++S)
      {
        if (!FString(UTF8_TO_TCHAR(R->getRBFSolverName(S).data())).StartsWith(TEXT("CM_"))) continue;
        auto Controls=R->getRBFSolverRawControlIndices(S);auto Poses=R->getRBFSolverPoseIndices(S);auto Values=R->getRBFSolverRawControlValues(S);
        if (Values.size()!=Controls.size()*Poses.size()) return false;
        for (uint16 P=0;P<Poses.size();++P)
        {
            TArray<uint16> Input;TArray<float> Sample;
            for (uint16 I=0;I<Controls.size();++I) {Input.Add(Controls[I]);Sample.Add(Values[P*Controls.size()+I]);}
            if (!Compare(LOD,Input,Sample)) return false;
        }
      }
    }
    auto Receipt=MakeShared<FJsonObject>();Receipt->SetBoolField(TEXT("existing_driver_outputs_preserved"),!Activate);Receipt->SetBoolField(TEXT("unrequested_driver_outputs_preserved"),true);Receipt->SetBoolField(TEXT("native_joint_outputs_preserved"),true);Receipt->SetBoolField(TEXT("lod0_driver_outputs_preserved"),true);Receipt->SetBoolField(TEXT("activated_outputs_match_lod0"),Activate);Receipt->SetNumberField(TEXT("activated_owned_channels"),ChannelLOD.Num());Receipt->SetNumberField(TEXT("new_mesh_targets"),Patches.Num());Receipt->SetNumberField(TEXT("new_channels"),0);Receipt->SetNumberField(TEXT("native_samples"),Samples);Receipt->SetNumberField(TEXT("maximum_driver_difference"),Maximum);
    FString Result;return FJsonSerializer::Serialize(Receipt,TJsonWriterFactory<>::Create(&Result))&&FFileHelper::SaveStringToFile(Result,*(OutputFile+TEXT(".drivers.json")));
}
