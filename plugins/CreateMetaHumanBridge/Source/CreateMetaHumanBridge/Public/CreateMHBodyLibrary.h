#pragma once
#include "Kismet/BlueprintFunctionLibrary.h"
#include "CreateMHBodyLibrary.generated.h"

UCLASS()
class CREATEMETAHUMANBRIDGE_API UCreateMHBodyLibrary : public UBlueprintFunctionLibrary
{
    GENERATED_BODY()
public:
    /** Evaluate a caller-owned native body state; does not edit any character asset. */
    UFUNCTION(BlueprintCallable, Category="Create MetaHuman")
    static bool EvaluateBodyJSON(const FString& RequestFile, const FString& OutputFile);
};
