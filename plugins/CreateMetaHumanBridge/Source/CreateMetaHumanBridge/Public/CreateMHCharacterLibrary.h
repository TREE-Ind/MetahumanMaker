#pragma once
#include "Kismet/BlueprintFunctionLibrary.h"
#include "CreateMHCharacterLibrary.generated.h"

UCLASS()
class CREATEMETAHUMANBRIDGE_API UCreateMHCharacterLibrary : public UBlueprintFunctionLibrary
{
    GENERATED_BODY()
public:
    /** Inspect saved prerequisites or a live session without initializing a character.
     * Closed assets report saved eligibility separately from verified live readiness.
     * Never requests rigging, textures or a build.
     */
    UFUNCTION(BlueprintCallable, Category="Create MetaHuman")
    static FString InspectCharacter(UObject* Character);

    /** Read evaluated LOD0 face state in canonical DNA vertex order (UE cm).
     * Requires an already editable character. Never conforms, commits or rigs.
     * Writes JSON metadata plus little-endian float32 XYZ binary coordinates.
     */
    UFUNCTION(BlueprintCallable, Category="Create MetaHuman")
    static bool ExportFaceState(UObject* Character, const FString& OutputFile);

    /** Refresh native preview actors from the character's saved material mode.
     * Requires an editable character; never conforms, rigs or requests textures.
     */
    UFUNCTION(BlueprintCallable, Category="Create MetaHuman")
    static bool RefreshSkinPreview(UObject* Character);

    /** Pass a private pipeline through the native build boundary. Validation
     * only checks configuration; execution requires an editable ready character.
     * Returning true means dispatched, not that assembly output is complete.
     */
    UFUNCTION(BlueprintCallable, Category="Create MetaHuman")
    static bool BuildWithPipeline(UObject* Character, UObject* Pipeline, const FString& ParametersJson, bool bValidateOnly = false);
};
