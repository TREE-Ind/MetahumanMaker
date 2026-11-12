#include "CreateMHGroomLibrary.h"
#include "GroomComponent.h"
#include "GroomAsset.h"
#include "GroomBindingAsset.h"
#include "Engine/Blueprint.h"
#include "Engine/BlueprintGeneratedClass.h"
#include "Engine/SimpleConstructionScript.h"
#include "Engine/SCS_Node.h"
#include "Engine/InheritableComponentHandler.h"
#include "Kismet2/BlueprintEditorUtils.h"
#include "UObject/UnrealType.h"
#include "UObject/UObjectHash.h"
#include "Dom/JsonObject.h"
#include "Serialization/JsonSerializer.h"

namespace MHGroomBlueprint
{
static USCS_Node* FindNode(UBlueprint* Blueprint,FName Name)
{
    if (!Blueprint || !Blueprint->GeneratedClass || Name.IsNone()) return nullptr;
    USCS_Node* Found=nullptr;
    for (UClass* Class=Blueprint->GeneratedClass;Class;Class=Class->GetSuperClass())
    {
        UBlueprint* Owner=Cast<UBlueprint>(Class->ClassGeneratedBy);
        if (!Owner || !Owner->SimpleConstructionScript) continue;
        for (USCS_Node* Node:Owner->SimpleConstructionScript->GetAllNodes())
            if (Node && Node->GetVariableName()==Name && Cast<UGroomComponent>(Node->ComponentTemplate))
            {if (Found && Found!=Node) return nullptr;Found=Node;}
    }
    return Found;
}
static bool Editable(UBlueprint* Blueprint)
{
    return IsInGameThread() && Blueprint && Blueprint->GetOutermost()->GetName().StartsWith(TEXT("/Game/"));
}
}

UGroomComponent* UCreateMHGroomLibrary::FindBlueprintGroomTemplate(UBlueprint* Blueprint,FName ComponentName)
{
    if (!IsInGameThread()) return nullptr;
    USCS_Node* Node=MHGroomBlueprint::FindNode(Blueprint,ComponentName);
    return Node?Cast<UGroomComponent>(Node->GetActualComponentTemplate(Cast<UBlueprintGeneratedClass>(Blueprint->GeneratedClass))):nullptr;
}

UGroomComponent* UCreateMHGroomLibrary::EnsureLocalBlueprintGroomTemplate(UBlueprint* Blueprint,FName ComponentName)
{
    if (!MHGroomBlueprint::Editable(Blueprint)) return nullptr;
    UGroomComponent* Current=FindBlueprintGroomTemplate(Blueprint,ComponentName);
    if (!Current) return nullptr;
    if (Current->GetOutermost()==Blueprint->GetOutermost()) return Current;
    USCS_Node* Node=MHGroomBlueprint::FindNode(Blueprint,ComponentName);
    const FComponentKey Key(Node);
    if (!Key.IsValid() || !Blueprint->ParentClass || !Blueprint->ParentClass->IsChildOf(Key.GetComponentOwner())) return nullptr;
    Blueprint->Modify();
    UInheritableComponentHandler* Handler=Blueprint->GetInheritableComponentHandler(true);
    if (!Handler) return nullptr;
    Handler->Modify();
    UGroomComponent* Local=Cast<UGroomComponent>(Handler->CreateOverridenComponentTemplate(Key));
    if (!Local || Local->GetOutermost()!=Blueprint->GetOutermost()) return nullptr;
    FBlueprintEditorUtils::MarkBlueprintAsModified(Blueprint);
    return Local;
}

bool UCreateMHGroomLibrary::RemoveBlueprintGroomOverride(UBlueprint* Blueprint,FName ComponentName,UGroomAsset* ExpectedGroom,UGroomBindingAsset* ExpectedBinding)
{
    if (!MHGroomBlueprint::Editable(Blueprint)) return false;
    USCS_Node* Node=MHGroomBlueprint::FindNode(Blueprint,ComponentName);
    if (!Node || Node->ComponentTemplate->GetOutermost()==Blueprint->GetOutermost()) return false;
    UInheritableComponentHandler* Handler=Blueprint->GetInheritableComponentHandler(false);
    const FComponentKey Key(Node);
    UGroomComponent* Local=Handler?Cast<UGroomComponent>(Handler->GetOverridenComponentTemplate(Key)):nullptr;
    if (!Local || Local->GetOutermost()!=Blueprint->GetOutermost() || Local->GroomAsset!=ExpectedGroom || Local->BindingAsset!=ExpectedBinding) return false;
    Blueprint->Modify();Handler->Modify();Handler->RemoveOverridenComponentTemplate(Key);
    FBlueprintEditorUtils::MarkBlueprintAsModified(Blueprint);
    return Handler->GetOverridenComponentTemplate(Key)==nullptr;
}

FString UCreateMHGroomLibrary::SnapshotGroomTemplate(UGroomComponent* Component,bool bArchetype)
{
    if (!IsInGameThread() || !IsValid(Component)) return TEXT("");
    if (bArchetype) Component=Cast<UGroomComponent>(Component->GetArchetype());
    if (!IsValid(Component)) return TEXT("");
    const TSet<FName> Controlled{TEXT("GroomAsset"),TEXT("BindingAsset"),TEXT("OverrideMaterials"),TEXT("GroomGroupsDesc"),TEXT("bUseCards")};
    const FString RootPath=Component->GetPathName();
    TArray<UObject*> Objects{Component};GetObjectsWithOuter(Component,Objects,EGetObjectsFlags::IncludeNestedObjects);
    auto Root=MakeShared<FJsonObject>();auto Records=MakeShared<FJsonObject>();Root->SetNumberField(TEXT("schema"),1);
    for (UObject* Item:Objects)
    {
        if (!IsValid(Item) || Item->HasAnyFlags(RF_Transient)) continue;
        auto Record=MakeShared<FJsonObject>();auto Properties=MakeShared<FJsonObject>();
        Record->SetStringField(TEXT("class"),Item->GetClass()->GetPathName());
        for (TFieldIterator<FProperty> It(Item->GetClass());It;++It)
        {
            const FProperty* Field=*It;
            if (!Field->HasAnyPropertyFlags(CPF_Edit) || Field->HasAnyPropertyFlags(CPF_EditConst|CPF_Transient|CPF_Deprecated|CPF_SkipSerialization)
                    || (Item==Component && Controlled.Contains(Field->GetFName()))) continue;
            FString Value;Field->ExportText_InContainer(0,Value,Item,nullptr,Item,PPF_None);
            if (Value.Len()>1048576) return TEXT("");
            // Normalize only the exact self path or its owned subobjects,
            // not external objects with a similarly prefixed name.
            if (Value==RootPath) Value=TEXT("$SELF");
            for (const TCHAR* Suffix:{TEXT("'"),TEXT("\""),TEXT("."),TEXT(":")})
                Value.ReplaceInline(*(RootPath+Suffix),*(FString(TEXT("$SELF"))+Suffix));
            Properties->SetStringField(Field->GetName(),Value);
        }
        Record->SetObjectField(TEXT("properties"),Properties);
        Records->SetObjectField(Item==Component?TEXT(""):Item->GetPathName(Component),Record);
    }
    Root->SetObjectField(TEXT("objects"),Records);
    FString Result;FJsonSerializer::Serialize(Root,TJsonWriterFactory<>::Create(&Result));return Result;
}
