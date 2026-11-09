#pragma once
#include "Kismet/BlueprintFunctionLibrary.h"
#include "CreateMHGroomLibrary.generated.h"
class UGroomAsset;

UCLASS()
class CREATEMETAHUMANBRIDGE_API UCreateMHGroomLibrary : public UBlueprintFunctionLibrary
{
    GENERATED_BODY()
public:
    UFUNCTION(BlueprintCallable, Category="Create MetaHuman")
    static bool ExportGroomJSON(UGroomAsset* Groom, const FString& FilePath);
    UFUNCTION(BlueprintCallable, Category="Create MetaHuman")
    static UGroomAsset* ImportGroomJSON(const FString& FilePath, const FString& AssetPath);
};
