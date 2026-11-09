#pragma once
#include "Kismet/BlueprintFunctionLibrary.h"
#include "CreateMHCalibrationLibrary.generated.h"

class USkeletalMesh;

/** Explicit local calibration staging. Never requests a service rig or saves an asset. */
UCLASS()
class CREATEMETAHUMANBRIDGE_API UCreateMHCalibrationLibrary : public UBlueprintFunctionLibrary
{
    GENERATED_BODY()
public:
    /** Read-only neutral geometry/bounds inspection across every imported LOD.
     * Does not certify animated morph/bone envelopes or save assets. */
    UFUNCTION(BlueprintCallable, Category="Create MetaHuman")
    static FString InspectGeometryBounds(USkeletalMesh* Mesh);

    /** Expand only a separate project's mesh copy to contain its neutral geometry.
     * Retains authored bounds extensions and never changes the source or saves. */
    UFUNCTION(BlueprintCallable, Category="Create MetaHuman")
    static bool RefreshMeshCopyBounds(USkeletalMesh* SourceMesh, USkeletalMesh* TargetMesh);

    /** Recompute cached bounds only on a caller-owned transient review actor.
     * Allows checking a bounds-scale change without waiting for editor ticks. */
    UFUNCTION(BlueprintCallable, Category="Create MetaHuman")
    static bool RefreshPreviewComponentBounds(class USkeletalMeshComponent* Component);

    /** Export the selected face's embedded DNA and preserved-coordinate reference
     * into a fresh local directory. Never modifies or saves native assets. */
    UFUNCTION(BlueprintCallable, Category="Create MetaHuman")
    static bool ExportEmbeddedCalibrationReference(USkeletalMesh* SourceMesh, const FString& Directory);

    /** Export full-LOD positions and existing target names in preserved DNA coordinates. */
    UFUNCTION(BlueprintCallable, Category="Create MetaHuman")
    static bool ExportCalibrationReference(const FString& DNAFile, const FString& OutputFile);

    /** Copy the complete DNA, changing only explicitly named neutral arrays and additive
     * existing corrective deltas. Output must be new; input is never overwritten. */
    UFUNCTION(BlueprintCallable, Category="Create MetaHuman")
    static bool WriteCalibrationDNA(const FString& DNAFile, const FString& RequestFile, const FString& OutputFile);

    /** Append explicitly named corrective targets driven by native interpolative
     * RBF solvers. Existing rig behavior and targets are copied without changes.
     * New targets/solvers are enabled at LOD0 only. Output must be new. */
    UFUNCTION(BlueprintCallable, Category="Create MetaHuman")
    static bool WritePoseCorrectiveDNA(const FString& DNAFile, const FString& RequestFile, const FString& OutputFile);

    /** Add mesh targets using already-owned CM_ corrective channels. Channel/control
     * identities and native joints remain unchanged. Default behavior preserves LOD
     * activation. An explicit activate_owned_drivers request extends only their owned
     * CM_ solvers/channels through the requested LODs, validated against LOD0 output.
     * Useful for revision accessories and lower-LOD correction transfer. */
    UFUNCTION(BlueprintCallable, Category="Create MetaHuman")
    static bool WriteDrivenMeshTargetsDNA(const FString& DNAFile, const FString& RequestFile, const FString& OutputFile);

    /** Bind calibrated DNA to a separate, unsaved mesh copy and update all LOD geometry.
     * Assembled morph data is retained; source DNA/skeleton/skin weights are not modified. */
    UFUNCTION(BlueprintCallable, Category="Create MetaHuman")
    static bool ApplyCalibrationToMeshCopy(USkeletalMesh* SourceMesh, USkeletalMesh* TargetMesh, const FString& DNAFile);

    /** Add explicit rest-space deltas to private copies of existing assembled morphs.
     * Keeps their assembled position baseline and unrelated LOD data. Native rebuilds
     * may recompute tangent/normal data. An explicit allow_new_dna_targets request
     * can add CM_ targets already defined in the private DNA, with mesh-owned
     * morph curve metadata. Never changes the shared skeleton. Never saves. */
    UFUNCTION(BlueprintCallable, Category="Create MetaHuman")
    static bool ApplyCompiledCorrectiveOffsets(USkeletalMesh* SourceMesh, USkeletalMesh* TargetMesh, const FString& RequestFile);

    /** Retire explicitly named CM_ globe morphs on an owned mesh copy. Removes
     * their import attributes and compiled objects in one rebuild; source,
     * native pupil targets and unrelated morphs are never modified. Never saves. */
    UFUNCTION(BlueprintCallable, Category="Create MetaHuman")
    static bool RetireGeneratedMorphs(USkeletalMesh* SourceMesh, USkeletalMesh* TargetMesh, const TArray<FName>& MorphNames);
};
