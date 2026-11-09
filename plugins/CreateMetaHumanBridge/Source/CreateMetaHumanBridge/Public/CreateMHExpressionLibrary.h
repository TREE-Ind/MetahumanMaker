#pragma once
#include "Kismet/BlueprintFunctionLibrary.h"
#include "CreateMHExpressionLibrary.generated.h"

class USkeletalMesh;
class UAnimSequence;
class UAnimInstance;
class USkeletalMeshComponent;

/** Local face evaluation and explicit, private preview-clip preparation. */
UCLASS()
class CREATEMETAHUMANBRIDGE_API UCreateMHExpressionLibrary : public UBlueprintFunctionLibrary
{
    GENERATED_BODY()
public:
    /** Native pose exports include evaluated animated skin material scalars. */
    UFUNCTION(BlueprintPure, Category="Create MetaHuman")
    static bool SupportsMaterialCurveSampling();
    /** Temporary pose components use private material instances, including curve-driven wrinkles. */
    UFUNCTION(BlueprintPure, Category="Create MetaHuman")
    static bool SupportsIsolatedPoseEvaluation();

    /** Bind a caller-owned project control clip to the face skeleton. Rejects
     * bone animation and non-control curves. Never saves or edits a skeleton. */
    UFUNCTION(BlueprintCallable, Category="Create MetaHuman")
    static bool PrepareControlClip(UAnimSequence* Animation, USkeletalMesh* FaceMesh);

    /** Create an unsaved constant raw-control clip using this mesh's embedded DNA. */
    UFUNCTION(BlueprintCallable, Category="Create MetaHuman")
    static UAnimSequence* CreateControlPreviewClip(USkeletalMesh* FaceMesh, const FString& ControlsJSON);

    /** Inspect supported raw controls without evaluating/exporting geometry. */
    UFUNCTION(BlueprintCallable, Category="Create MetaHuman")
    static TArray<FString> GetFaceRawControls(USkeletalMesh* FaceMesh);

    /** Capture editor playback settings. Unsupported/custom or unsaved animation
     * configurations are rejected rather than promising an inexact restore. */
    UFUNCTION(BlueprintCallable, Category="Create MetaHuman")
    static FString CaptureComponentPreviewState(USkeletalMeshComponent* Component, bool bAllowTransientAnimation=false);

    /** Restore a captured component's playback settings without saving assets. */
    UFUNCTION(BlueprintCallable, Category="Create MetaHuman")
    static bool RestoreComponentPreviewState(USkeletalMeshComponent* Component, const FString& StateJSON);

    /** Request: {poses:[{name:string,controls:{rawControlName:0..1}}]}; empty poses inspect DNA only. */
    UFUNCTION(BlueprintCallable, Category="Create MetaHuman")
    static bool ExportExpressionReferenceJSON(const FString& DNAFile, const FString& RequestFile, const FString& OutputFile);

    /** Evaluate face post-processing on transient animation/components. Empty DNAFile reads
     * the compiled mesh with its native DNA binding and post-process class. A nonempty path
     * uses the explicit legacy-compatible external-DNA evaluation route. */
    UFUNCTION(BlueprintCallable, Category="Create MetaHuman")
    static bool ExportEngineFacePose(USkeletalMesh* SourceMesh, UAnimSequence* Animation, const FString& DNAFile, const FString& OutputFile, float SampleTime=0.0f, TSubclassOf<UAnimInstance> PostProcessOverride=nullptr, bool bIncludeSkinMatrices=false, int32 LODIndex=0);

    /** Evaluate constant native raw controls from a JSON object using transient
     * animation data. No ROM/template animation or persistent assets are needed. */
    UFUNCTION(BlueprintCallable, Category="Create MetaHuman")
    static bool ExportEngineControlPose(USkeletalMesh* SourceMesh, const FString& ControlsJSON, const FString& OutputFile, USkeletalMeshComponent* SourceComponent=nullptr, bool bIncludeSkinMatrices=false, int32 LODIndex=0);
};
