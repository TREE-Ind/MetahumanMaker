using UnrealBuildTool;
public class CreateMetaHumanBridge : ModuleRules
{
    public CreateMetaHumanBridge(ReadOnlyTargetRules Target) : base(Target)
    {
        PCHUsage = PCHUsageMode.UseExplicitOrSharedPCHs;
        PublicDependencyModuleNames.AddRange(new string[] { "Core", "CoreUObject", "Engine", "HairStrandsCore", "StaticMeshDescription" });
        PrivateDependencyModuleNames.AddRange(new string[] { "Renderer", "Niagara" });
        PrivateDependencyModuleNames.AddRange(new string[] { "UnrealEd", "AssetRegistry", "AssetTools", "Json", "JsonUtilities", "MeshDescription", "SkeletalMeshDescription", "MaterialBaking", "RigLogicModule", "RigLogicLib", "MetaHumanCoreTechLib", "RHI", "RenderCore", "MetaHumanCharacter", "MetaHumanCharacterEditor", "MetaHumanCharacterPalette", "MetaHumanSDKRuntime" });
    }
}
