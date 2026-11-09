#include "CreateMHArtistLibrary.h"
#include "Components/SkinnedMeshComponent.h"
#include "Engine/SkeletalMesh.h"
#include "Materials/MaterialInterface.h"
#include "UObject/UnrealType.h"
#include "UObject/UObjectHash.h"
#include "Dom/JsonObject.h"
#include "Serialization/JsonSerializer.h"
#include "Serialization/JsonWriter.h"
#include "MaterialBakingStructures.h"
#include "IMaterialBakingModule.h"
#include "ImageUtils.h"
#include "Misc/FileHelper.h"
#include "Misc/Paths.h"
#include "HAL/FileManager.h"
#include "Modules/ModuleManager.h"
#include "StaticMeshAttributes.h"
#include "ShaderCompiler.h"
#include "AssetCompilingManager.h"
#include "RenderingThread.h"
#include "Misc/App.h"
#include "DynamicRHI.h"
#include "Engine/Texture.h"
#include "UObject/StrongObjectPtr.h"
#include "Materials/MaterialInstanceDynamic.h"

bool UCreateMHArtistLibrary::SupportsStableMaterialBaking() { return true; }

FString UCreateMHArtistLibrary::SnapshotObject(UObject* Object)
{
    if (!IsInGameThread() || !IsValid(Object)) return TEXT("{}");
    TSharedRef<FJsonObject> Root=MakeShared<FJsonObject>();
    Root->SetNumberField(TEXT("schema"),1);
    Root->SetStringField(TEXT("root"),Object->GetPathName());
    TArray<UObject*> Objects{Object};
    GetObjectsWithOuter(Object,Objects,EGetObjectsFlags::IncludeNestedObjects);
    TArray<TSharedPtr<FJsonValue>> Records;
    for (UObject* Item : Objects)
    {
        if (!IsValid(Item) || Item->HasAnyFlags(RF_Transient)) continue;
        TSharedRef<FJsonObject> Record=MakeShared<FJsonObject>();
        Record->SetStringField(TEXT("object"),Item->GetPathName());
        Record->SetStringField(TEXT("class"),Item->GetClass()->GetPathName());
        Record->SetStringField(TEXT("package"),Item->GetPackage()->GetName());
        Record->SetStringField(TEXT("outermost_package"),Item->GetOutermost()->GetName());
        Record->SetBoolField(TEXT("package_dirty"),Item->GetPackage()->IsDirty() || Item->GetOutermost()->IsDirty());
        TSharedRef<FJsonObject> Properties=MakeShared<FJsonObject>();
        for (TFieldIterator<FProperty> It(Item->GetClass()); It; ++It)
        {
            const FProperty* Field=*It;
            if (Field->HasAnyPropertyFlags(CPF_Transient|CPF_Deprecated|CPF_SkipSerialization)) continue;
            TSharedRef<FJsonObject> Value=MakeShared<FJsonObject>();
            Value->SetStringField(TEXT("type"),Field->GetCPPType());
            bool bLarge=Field->GetSize()>65536;
            if (const auto* Array=CastField<FArrayProperty>(Field))
            {
                FScriptArrayHelper Helper(Array,Array->ContainerPtrToValuePtr<void>(Item));
                Value->SetNumberField(TEXT("count"),Helper.Num());
                bLarge|=int64(Helper.Num())*Array->Inner->GetSize()>65536;
            }
            if (!bLarge)
            {
                FString Text;
                Field->ExportText_InContainer(0,Text,Item,nullptr,Item,PPF_None);
                if (Text.Len()<=262144) Value->SetStringField(TEXT("text"),Text);
                else bLarge=true;
            }
            Value->SetStringField(TEXT("preservation"),bLarge?TEXT("native_package_payload"):TEXT("reflected_text_and_native_package"));
            Properties->SetObjectField(Field->GetName(),Value);
        }
        Record->SetObjectField(TEXT("properties"),Properties);
        Records.Add(MakeShared<FJsonValueObject>(Record));
    }
    Root->SetArrayField(TEXT("objects"),Records);
    FString Text;
    FJsonSerializer::Serialize(Root,TJsonWriterFactory<>::Create(&Text));
    return Text;
}

bool UCreateMHArtistLibrary::BakeMaterialProperty(USkinnedMeshComponent* Component, int32 MaterialSlot, const FString& Property, int32 Size, const FString& Filename)
{
    if (!IsInGameThread() || !IsValid(Component) || Size<64 || Size>8192 || !FMath::IsPowerOfTwo(Size) || FPaths::FileExists(Filename)) return false;
    if (!FApp::CanEverRender() || GUsingNullRHI)
    {
        UE_LOG(LogTemp,Error,TEXT("CreateMH appearance baking requires rendering; commandlets must use -AllowCommandletRendering without -NullRHI"));
        return false;
    }
    USkeletalMesh* Mesh=Cast<USkeletalMesh>(Component->GetSkinnedAsset());
    UMaterialInterface* Material=Component->GetMaterial(MaterialSlot);
    if (!Mesh || !Material || MaterialSlot<0 || !Mesh->GetMeshDescription(0))
    {
        UE_LOG(LogTemp,Error,TEXT("CreateMH appearance: missing mesh, material, or LOD0 MeshDescription (mesh=%s material=%s slot=%d description=%d)"),*GetNameSafe(Mesh),*GetNameSafe(Material),MaterialSlot,Mesh&&Mesh->GetMeshDescription(0)!=nullptr);
        return false;
    }
    EMaterialProperty Kind;
    if (Property==TEXT("basecolor")) Kind=MP_BaseColor;
    else if (Property==TEXT("emissive_color")) Kind=MP_EmissiveColor;
    else if (Property==TEXT("roughness")) Kind=MP_Roughness;
    else if (Property==TEXT("specular")) Kind=MP_Specular;
    else if (Property==TEXT("normal")) Kind=MP_Normal;
    else if (Property==TEXT("metallic")) Kind=MP_Metallic;
    else if (Property==TEXT("ambient_occlusion")) Kind=MP_AmbientOcclusion;
    else if (Property==TEXT("opacity_mask")) Kind=MP_OpacityMask;
    else return false;
    // MaterialBaking caches a proxy's uniform expressions by interface pointer.
    // A later bake of the same MID can otherwise reuse its previous parameters.
    // Preserve the constant parent/static permutation, flatten dynamic parents,
    // and copy current uniforms into a unique transient interface per bake.
    UMaterialInterface* Parent=Material;
    while (UMaterialInstanceDynamic* Dynamic=Cast<UMaterialInstanceDynamic>(Parent)) Parent=Dynamic->Parent;
    if (!Parent) return false;
    TStrongObjectPtr<UMaterialInstanceDynamic> FrozenMaterial(UMaterialInstanceDynamic::Create(Parent,GetTransientPackage()));
    if (!FrozenMaterial.IsValid()) return false;
    FrozenMaterial->CopyMaterialUniformParameters(Material);
    Material=FrozenMaterial.Get();
    FMaterialData Settings;
    Settings.Material=Material;Settings.PropertySizes.Add(Kind,FIntPoint(Size,Size));
    Settings.bPerformShrinking=false;Settings.bTangentSpaceNormal=true;
    FMeshData Geometry;
    Geometry.MeshDescription=Mesh->GetMeshDescription(0);
    const FStaticMeshConstAttributes Attributes(*Geometry.MeshDescription);
    const auto SlotNames=Attributes.GetPolygonGroupMaterialSlotNames();
    const auto UVs=Attributes.GetVertexInstanceUVs();
    if (!Mesh->GetMaterials().IsValidIndex(MaterialSlot)) return false;
    const FSkeletalMaterial& Slot=Mesh->GetMaterials()[MaterialSlot];
    for (const FPolygonGroupID Group : Geometry.MeshDescription->PolygonGroups().GetElementIDs())
    {
        const FName Name=SlotNames[Group];
        UE_LOG(LogTemp,Display,TEXT("CreateMH appearance polygon group %d (%s), requested %d (%s / %s)"),Group.GetValue(),*Name.ToString(),MaterialSlot,*Slot.ImportedMaterialSlotName.ToString(),*Slot.MaterialSlotName.ToString());
        if (Name==Slot.ImportedMaterialSlotName || Name==Slot.MaterialSlotName) Geometry.MaterialIndices.Add(Group.GetValue());
    }
    if (Geometry.MaterialIndices.IsEmpty()) return false;
    int32 Triangles=0;FVector2f MinUV(FLT_MAX,FLT_MAX),MaxUV(-FLT_MAX,-FLT_MAX);
    for (const FTriangleID Triangle : Geometry.MeshDescription->Triangles().GetElementIDs())
    {
        if (!Geometry.MaterialIndices.Contains(Geometry.MeshDescription->GetTrianglePolygonGroup(Triangle).GetValue())) continue;
        ++Triangles;
        for (const FVertexInstanceID Vertex : Geometry.MeshDescription->GetTriangleVertexInstances(Triangle))
        {
            const auto UV=UVs.Get(Vertex,0);MinUV.X=FMath::Min(MinUV.X,UV.X);MinUV.Y=FMath::Min(MinUV.Y,UV.Y);MaxUV.X=FMath::Max(MaxUV.X,UV.X);MaxUV.Y=FMath::Max(MaxUV.Y,UV.Y);
        }
    }
    UE_LOG(LogTemp,Display,TEXT("CreateMH appearance selected %d triangles, UV0 range %s to %s"),Triangles,*MinUV.ToString(),*MaxUV.ToString());
    if (!Triangles) return false;
    // MetaHuman body UVs occupy UDIM 1002. A fixed 0..1 baking box produces
    // uncovered magenta even though the material and geometry are correct.
    const FVector2D Tile(FMath::FloorToDouble(MinUV.X+1.e-5),FMath::FloorToDouble(MinUV.Y+1.e-5));
    if (MaxUV.X>Tile.X+1.00001 || MaxUV.Y>Tile.Y+1.00001)
    {
        UE_LOG(LogTemp,Error,TEXT("CreateMH appearance: material spans multiple UDIM tiles; export them separately"));return false;
    }
    Geometry.TextureCoordinateBox=FBox2D(Tile,Tile+FVector2D(1,1));
    Geometry.PrimitiveData=FPrimitiveData(Component);
    // The editor may still be compiling the freshly loaded parent material.
    // ExportMaterialProxy requires its shader map to be fully initialized.
    FAssetCompilingManager::Get().FinishAllCompilation();
    if (GShaderCompilingManager) GShaderCompilingManager->FinishAllCompilation();
    // A compiled texture may still have only coarse streaming mips resident.
    // UV baking has no screen-space footprint to request the required mips.
    // Force only this material's textures during the bake, then restore every
    // transient residency flag without touching artist asset settings.
    struct FResidentTextures
    {
        struct FEntry { TStrongObjectPtr<UTexture> Texture; bool Force; bool IgnoreBias; };
        TArray<FEntry> Entries;
        ~FResidentTextures()
        {
            for (FEntry& Entry : Entries)
            {
                Entry.Texture->bForceMiplevelsToBeResident=Entry.Force;
                Entry.Texture->bIgnoreStreamingMipBias=Entry.IgnoreBias;
            }
        }
    } Resident;
    TArray<UTexture*> Textures;
    Material->GetUsedTextures(Textures);
    for (UTexture* Texture : Textures)
    {
        if (!IsValid(Texture)) continue;
        Resident.Entries.Add({TStrongObjectPtr<UTexture>(Texture),bool(Texture->bForceMiplevelsToBeResident),bool(Texture->bIgnoreStreamingMipBias)});
        Texture->bForceMiplevelsToBeResident=true;
        Texture->bIgnoreStreamingMipBias=true;
    }
    for (FResidentTextures::FEntry& Entry : Resident.Entries) Entry.Texture->WaitForStreaming();
    FlushRenderingCommands();
    TArray<FMaterialData*> Materials{&Settings};TArray<FMeshData*> Meshes{&Geometry};TArray<FBakeOutput> Outputs;
    FModuleManager::LoadModuleChecked<IMaterialBakingModule>(TEXT("MaterialBaking")).BakeMaterials(Materials,Meshes,Outputs);
    const bool bConstant=Outputs.Num()==1 && Outputs[0].PropertyData.Contains(Kind) && Outputs[0].PropertySizes[Kind]==FIntPoint(1,1) && Outputs[0].PropertyData[Kind].Num()==1;
    if (bConstant)
    {
        const FColor Pixel=Outputs[0].PropertyData[Kind][0];
        UE_LOG(LogTemp,Display,TEXT("CreateMH appearance constant result: %s"),*Pixel.ToString());
        if (Pixel.R==255 && Pixel.G==0 && Pixel.B==255)
        {
            UE_LOG(LogTemp,Error,TEXT("CreateMH appearance: uncovered magenta output rejected"));return false;
        }
        Outputs[0].PropertyData[Kind].Init(Pixel,Size*Size);
        Outputs[0].PropertySizes[Kind]=FIntPoint(Size,Size);
    }
    if (Outputs.Num()!=1 || !Outputs[0].PropertyData.Contains(Kind) || Outputs[0].PropertySizes[Kind]!=FIntPoint(Size,Size))
    {
        const FIntPoint Actual=Outputs.Num()==1?Outputs[0].PropertySizes.FindRef(Kind):FIntPoint::ZeroValue;
        UE_LOG(LogTemp,Error,TEXT("CreateMH appearance: native baker returned unexpected outputs (count=%d, requested=%d, actual=%dx%d)"),Outputs.Num(),Size,Actual.X,Actual.Y);
        return false;
    }
    const auto& Pixels=Outputs[0].PropertyData[Kind];
    if (Pixels.Num()!=Size*Size){UE_LOG(LogTemp,Error,TEXT("CreateMH appearance: incomplete native pixels %d"),Pixels.Num());return false;}
    TArray64<uint8> PNG;
    FImageUtils::PNGCompressImageArray(Size,Size,Pixels,PNG);
    if (!IFileManager::Get().MakeDirectory(*FPaths::GetPath(Filename),true)) return false;
    if (!FFileHelper::SaveArrayToFile(PNG,*Filename)) return false;
    TSharedRef<FJsonObject> Receipt=MakeShared<FJsonObject>();
    Receipt->SetStringField(TEXT("property"),Property);
    Receipt->SetBoolField(TEXT("constant_native_output"),bConstant);
    Receipt->SetNumberField(TEXT("emissive_scale"),Outputs[0].EmissiveScale);
    Receipt->SetBoolField(TEXT("linear_color"),Outputs[0].PropertyIsLinearColor.FindRef(Kind));
    Receipt->SetBoolField(TEXT("texture_residency_waited"),true);
    Receipt->SetBoolField(TEXT("material_parameters_snapshotted"),true);
    Receipt->SetNumberField(TEXT("resident_texture_count"),Resident.Entries.Num());
    Receipt->SetArrayField(TEXT("uv_box"),{MakeShared<FJsonValueNumber>(Tile.X),MakeShared<FJsonValueNumber>(Tile.Y),MakeShared<FJsonValueNumber>(Tile.X+1),MakeShared<FJsonValueNumber>(Tile.Y+1)});
    FString Text;FJsonSerializer::Serialize(Receipt,TJsonWriterFactory<>::Create(&Text));
    return FFileHelper::SaveStringToFile(Text,*(Filename+TEXT(".json")),FFileHelper::EEncodingOptions::ForceUTF8WithoutBOM);
}
