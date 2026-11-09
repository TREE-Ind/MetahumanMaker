#pragma once
#include "Kismet/BlueprintFunctionLibrary.h"
#include "CreateMHSkinLibrary.generated.h"

class UMaterialInterface;

UCLASS()
class CREATEMETAHUMANBRIDGE_API UCreateMHSkinLibrary : public UBlueprintFunctionLibrary
{
    GENERATED_BODY()
public:
    /** Bake the actual native BaseColor graph in UV space, without editing the material. */
    UFUNCTION(BlueprintCallable, Category="Create MetaHuman")
    static bool BakeBaseColor(UMaterialInterface* Material, const FString& OutputFile, int32 Resolution, const FString& ScratchDirectory, FVector2D UVOrigin);

    /** Capture BaseColor, tangent-space Normal or linear Roughness from the final native graph. */
    UFUNCTION(BlueprintCallable, Category="Create MetaHuman")
    static bool BakeSurfaceMap(UMaterialInterface* Material, const FString& OutputFile, int32 Resolution, const FString& ScratchDirectory, FVector2D UVOrigin, FName Map);
};
