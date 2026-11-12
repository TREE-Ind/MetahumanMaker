#pragma once
#include "Kismet/BlueprintFunctionLibrary.h"
#include "CreateMHGroomLibrary.generated.h"
class UGroomAsset;
class UGroomComponent;
class UGroomBindingAsset;
class UBlueprint;

UCLASS()
class CREATEMETAHUMANBRIDGE_API UCreateMHGroomLibrary : public UBlueprintFunctionLibrary
{
    GENERATED_BODY()
public:
    UFUNCTION(BlueprintCallable, Category="Create MetaHuman")
    static bool ExportGroomJSON(UGroomAsset* Groom, const FString& FilePath);
    UFUNCTION(BlueprintCallable, Category="Create MetaHuman")
    static UGroomAsset* ImportGroomJSON(const FString& FilePath, const FString& AssetPath);
    UFUNCTION(BlueprintCallable, Category="Create MetaHuman")
    static FString CompareGroomGroups(UGroomAsset* Source, int32 SourceGroup, UGroomAsset* Target, int32 TargetGroup, bool bComparePositions = true);

    /** Ordered source curve data, captured before an assembly can replace assets. */
    UFUNCTION(BlueprintCallable, Category = "MetaHuman Maker|Groom")
    static FString FingerprintGroom(UGroomAsset* Groom);

    UFUNCTION(BlueprintCallable, Category = "MetaHuman Maker|Groom")
    static bool InitializeGroomGroups(UGroomAsset* Groom);

    UFUNCTION(BlueprintCallable, Category = "MetaHuman Maker|Groom")
    static FString ExportGroomLODDeformation(UGroomAsset* Source, int32 SourceGroup, UGroomAsset* Target, int32 TargetGroup, const FString& FilePath, int32 MaxControls = 65536);

    UFUNCTION(BlueprintCallable, Category = "MetaHuman Maker|Groom")
    static FString ApplyGroomLODDeformation(const FString& FilePath, const FString& DestinationFolder);

    /** Renderer geometry and live Niagara controller age/state. Authored flags and requested LOD are not playback evidence. */
    UFUNCTION(BlueprintCallable, Category = "MetaHuman Maker|Groom")
    static FString CaptureGroomRepresentationState(UGroomComponent* Component);

    UFUNCTION(BlueprintCallable, Category = "MetaHuman Maker|Groom")
    static UGroomComponent* FindBlueprintGroomTemplate(UBlueprint* Blueprint, FName ComponentName);

    UFUNCTION(BlueprintCallable, Category = "MetaHuman Maker|Groom")
    static UGroomComponent* EnsureLocalBlueprintGroomTemplate(UBlueprint* Blueprint, FName ComponentName);

    UFUNCTION(BlueprintCallable, Category = "MetaHuman Maker|Groom")
    static bool RemoveBlueprintGroomOverride(UBlueprint* Blueprint, FName ComponentName, UGroomAsset* ExpectedGroom, UGroomBindingAsset* ExpectedBinding);

    /** Authored template settings outside groom replacement's controlled fields. */
    UFUNCTION(BlueprintCallable, Category = "MetaHuman Maker|Groom")
    static FString SnapshotGroomTemplate(UGroomComponent* Component, bool bArchetype = false);
};
