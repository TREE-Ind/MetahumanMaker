#include "CreateMHSkinLibrary.h"
#include "Materials/Material.h"
#include "Materials/MaterialInstanceConstant.h"
#include "Materials/MaterialExpressionBreakMaterialAttributes.h"
#include "Materials/MaterialExpressionMultiply.h"
#include "Materials/MaterialExpressionAdd.h"
#include "MaterialShared.h"
#include "StaticParameterSet.h"
#include "ShaderCompiler.h"
#include "AssetCompilingManager.h"
#include "UObject/StrongObjectPtr.h"
#include "UObject/UnrealType.h"
#include "Engine/TextureRenderTarget2D.h"
#include "Engine/Texture2D.h"
#include "Engine/Canvas.h"
#include "Kismet/KismetRenderingLibrary.h"
#include "Editor.h"
#include "RenderingThread.h"
#include "Misc/FileHelper.h"
#include "Misc/Paths.h"
#include "HAL/FileManager.h"
#include "ImageUtils.h"
#include "AssetToolsModule.h"
#include "IAssetTools.h"
#include "Modules/ModuleManager.h"

bool UCreateMHSkinLibrary::BakeBaseColor(UMaterialInterface* Material, const FString& OutputFile, int32 Resolution, const FString& ScratchDirectory, FVector2D UVOrigin)
{
    return BakeSurfaceMap(Material,OutputFile,Resolution,ScratchDirectory,UVOrigin,TEXT("BaseColor"));
}

bool UCreateMHSkinLibrary::BakeSurfaceMap(UMaterialInterface* Material, const FString& OutputFile, int32 Resolution, const FString& ScratchDirectory, FVector2D UVOrigin, FName Map)
{
    const bool bNormal=Map==TEXT("Normal"), bRoughness=Map==TEXT("Roughness");
    if (!bNormal && !bRoughness && Map!=TEXT("BaseColor")) return false;
    if (!IsInGameThread() || !GEditor || !IsValid(Material) || Resolution<64 || Resolution>8192 ||
        OutputFile.IsEmpty() || !ScratchDirectory.StartsWith(TEXT("/Game/CreateMHSkinScratch/Job_")) ||
        !FPaths::GetExtension(OutputFile).Equals(TEXT("png"), ESearchCase::IgnoreCase)) return false;
    UWorld* World=GEditor->GetEditorWorldContext().World();
    UMaterial* Original=Material->GetMaterial();
    if (!World || !Original) return false;
    if (bNormal && !Original->bTangentSpaceNormal) return false;

    // Capture a private copy of the native BaseColor graph. Preserve all static
    // and uniform settings; no lighting, actor edit or original material edit.
    // AssetTools retains the editor-only expression graph when duplicating.
    auto& Assets=FModuleManager::LoadModuleChecked<FAssetToolsModule>(TEXT("AssetTools")).Get();
    const FString Name=TEXT("BakeColor_")+FGuid::NewGuid().ToString(EGuidFormats::Digits);
    TStrongObjectPtr<UMaterial> Copy(Cast<UMaterial>(Assets.DuplicateAsset(Name,ScratchDirectory,Original)));
    if (!Copy) return false;
    UMaterialEditorOnlyData* Data=Copy->GetEditorOnlyData();
    if (!Data) { UE_LOG(LogTemp,Error,TEXT("Create MH: private skin material has no editor graph")); return false; }
    if (Copy->bUseMaterialAttributes)
    {
        if (!Data->MaterialAttributes.Expression) { UE_LOG(LogTemp,Error,TEXT("Create MH: private skin graph has no material attributes")); return false; }
        auto* Break=NewObject<UMaterialExpressionBreakMaterialAttributes>(Copy.Get(),NAME_None,RF_Transient);
        Break->MaterialAttributes=Data->MaterialAttributes;
        Copy->GetExpressionCollection().AddExpression(Break);
        Data->EmissiveColor.Connect(bNormal ? 8 : (bRoughness ? 3 : 0),Break);
        Data->MaterialAttributes.Expression=nullptr;
        Copy->bUseMaterialAttributes=false;
    }
    else
    {
        const FExpressionInput& Input=bNormal ? static_cast<const FExpressionInput&>(Data->Normal) :
            (bRoughness ? static_cast<const FExpressionInput&>(Data->Roughness) : static_cast<const FExpressionInput&>(Data->BaseColor));
        if (!Input.Expression) return false;
        static_cast<FExpressionInput&>(Data->EmissiveColor)=Input;
    }
    if (bNormal)
    {
        auto* Scale=NewObject<UMaterialExpressionMultiply>(Copy.Get(),NAME_None,RF_Transient);
        auto* Bias=NewObject<UMaterialExpressionAdd>(Copy.Get(),NAME_None,RF_Transient);
        Scale->A=Data->EmissiveColor; Scale->ConstB=0.5f;
        Bias->A.Connect(0,Scale); Bias->ConstB=0.5f;
        Copy->GetExpressionCollection().AddExpression(Scale);
        Copy->GetExpressionCollection().AddExpression(Bias);
        Data->EmissiveColor.Connect(0,Bias);
    }
    Data->EmissiveColor.UseConstant=false;
    // UE 5.8's setter validates the *old* enum, which can be the special
    // FromMaterialExpression value on native skin. Normalize our private copy
    // before calling it, avoiding a handled engine ensure on every new editor.
    auto* ModelProperty=FindFProperty<FByteProperty>(UMaterial::StaticClass(),TEXT("ShadingModel"));
    if (!ModelProperty) return false;
    ModelProperty->SetPropertyValue_InContainer(Copy.Get(),static_cast<uint8>(MSM_Unlit));
    Copy->SetShadingModel(MSM_Unlit);
    Copy->bUsedAsSpecialEngineMaterial=false;
    Copy->SetUsageByFlag(MATUSAGE_SkeletalMesh,false);
    Copy->SetUsageByFlag(MATUSAGE_NiagaraSprites,false);
    Copy->SetUsageByFlag(MATUSAGE_NiagaraRibbons,false);
    Copy->SetUsageByFlag(MATUSAGE_NiagaraMeshParticles,false);
    Copy->SetFeatureLevelToCompile(GMaxRHIFeatureLevel,true);
    Copy->PostEditChange();
    Copy->ForceRecompileForRendering();
    TStrongObjectPtr<UMaterialInstanceConstant> Instance(NewObject<UMaterialInstanceConstant>(GetTransientPackage(),NAME_None,RF_Transient));
    Instance->SetParentEditorOnly(Copy.Get(),false);
    Instance->CopyMaterialUniformParametersEditorOnly(Material,true);
    // We flattened the parent hierarchy onto the private base material. Values
    // inherited by the source (notably IsBody) must become explicit overrides;
    // the ordinary uniform-copy helper retains their original override flags.
    FStaticParameterSet Static;
    Material->GetStaticParameterValues(Static);
    for (FStaticSwitchParameter& Parameter:Static.StaticSwitchParameters) Parameter.bOverride=true;
    for (FStaticComponentMaskParameter& Parameter:Static.EditorOnly.StaticComponentMaskParameters) Parameter.bOverride=true;
    Instance->UpdateStaticPermutation(Static);
    Instance->SetFeatureLevelToCompile(GMaxRHIFeatureLevel,true);
    Instance->PostEditChange();
    Instance->ForceRecompileForRendering();
    FAssetCompilingManager::Get().FinishAllCompilation();
    if (GShaderCompilingManager) GShaderCompilingManager->FinishAllCompilation();
    FlushRenderingCommands();
    const FMaterialResource* Resource=Instance->GetMaterialResource(GMaxRHIShaderPlatform);
    if (!Resource || !Resource->IsGameThreadShaderMapComplete()) { UE_LOG(LogTemp,Error,TEXT("Create MH: private skin material shader did not compile")); return false; }
    TArray<UTexture*> Textures;
    Material->GetUsedTextures(Textures,TOptional<EMaterialQualityLevel::Type>(),TOptional<EShaderPlatform>(GMaxRHIShaderPlatform));
    for (UTexture* Texture:Textures)
    {
        if (UTexture2D* Texture2D=Cast<UTexture2D>(Texture))
        {
            Texture2D->SetForceMipLevelsToBeResident(60.0f);
            Texture2D->StreamIn(Texture2D->GetNumMips(),true);
            Texture2D->WaitForStreaming();
        }
    }
    FlushRenderingCommands();

    TStrongObjectPtr<UTextureRenderTarget2D> Target(UKismetRenderingLibrary::CreateRenderTarget2D(
        World,Resolution,Resolution,RTF_RGBA8,FLinearColor(0,0,0,1)));
    if (!Target) return false;
    Target->TargetGamma=bNormal || bRoughness ? 1.0f : 2.2f;
    Target->UpdateResourceImmediate(true);
    UCanvas* Canvas=nullptr;
    FVector2D CanvasSize;
    FDrawToRenderTargetContext Context;
    UKismetRenderingLibrary::BeginDrawCanvasToRenderTarget(World,Target.Get(),Canvas,CanvasSize,Context);
    if (Canvas) Canvas->K2_DrawMaterial(Instance.Get(),FVector2D::ZeroVector,CanvasSize,UVOrigin,FVector2D::UnitVector);
    UKismetRenderingLibrary::EndDrawCanvasToRenderTarget(World,Context);
    TArray<FColor> Pixels;
    FReadSurfaceDataFlags Flags(RCM_UNorm);
    Flags.SetLinearToGamma(false);
    const bool bRead=Target->GameThread_GetRenderTargetResource()->ReadPixels(Pixels,Flags);
    Target->ReleaseResource();
    FlushRenderingCommands();
    if (!bRead || Pixels.Num()!=Resolution*Resolution) return false;
    bool bNonBlack=false, bNotAllMagenta=false;
    for (FColor& Pixel:Pixels)
    {
        bNonBlack|=Pixel.R!=0 || Pixel.G!=0 || Pixel.B!=0;
        bNotAllMagenta|=Pixel.R!=255 || Pixel.G!=0 || Pixel.B!=255;
        Pixel.A=255;
    }
    if (!bNonBlack || !bNotAllMagenta) { UE_LOG(LogTemp,Error,TEXT("Create MH: private skin material returned blank pixels")); return false; }
    TArray64<uint8> PNG;
    FImageUtils::PNGCompressImageArray(Resolution,Resolution,Pixels,PNG);
    if (PNG.IsEmpty()) return false;
    IFileManager::Get().MakeDirectory(*FPaths::GetPath(OutputFile),true);
    return FFileHelper::SaveArrayToFile(PNG,*OutputFile);
}
