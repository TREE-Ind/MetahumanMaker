#pragma once
#include "Kismet/BlueprintFunctionLibrary.h"
#include "CreateMHArtistLibrary.generated.h"

class USkinnedMeshComponent;

/** Read-only preservation and native material evaluation for artist round trips. */
UCLASS()
class CREATEMETAHUMANBRIDGE_API UCreateMHArtistLibrary : public UBlueprintFunctionLibrary
{
    GENERATED_BODY()
public:
    /** Current-parameter snapshots and resident-mip evaluation are supported. */
    UFUNCTION(BlueprintPure, Category="Create MetaHuman")
    static bool SupportsStableMaterialBaking();

    /** Reflect all persistent properties and owned subobjects. Large payloads remain in native packages. */
    UFUNCTION(BlueprintCallable, Category="Create MetaHuman")
    static FString SnapshotObject(UObject* Object);

    /** Bake one live material property over its real LOD0 UVs. Never modifies the material or mesh. */
    UFUNCTION(BlueprintCallable, Category="Create MetaHuman")
    static bool BakeMaterialProperty(USkinnedMeshComponent* Component, int32 MaterialSlot, const FString& Property, int32 Size, const FString& Filename);
};
