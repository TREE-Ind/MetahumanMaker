# Installation

[Back to README](../README.md) · [First character](quickstart.md) · [Troubleshooting](troubleshooting.md)

## Choose your installation route

Use the **release add-on ZIP** for Blender. GitHub's automatic source-code ZIP and the source checkout contain development files and cannot be installed directly as the add-on.

| Component | Launch target |
| --- | --- |
| OS | Windows x64 |
| Blender | 5.2.x; exercised on 5.2.3; add-on minimum 5.2.0 |
| Unreal | 5.8.x; exercised on 5.8.2 |
| Editor bridge | MetaHuman Maker Bridge 0.29.0 |
| Worker / repository Python | Separate Python 3.12 environment |
| MetaHuman data | Your own Creator Core Data and licensed reference content |

An Unreal build/patch mismatch can require rebuilding the bridge. Other platforms are not certified by the Windows launch package. Resource use depends on source density, subdivisions, texture resolution and optional backends; start at modest settings and use the add-on's memory/disk estimates.

## 1. Install the Blender add-on

1. Download [`metahuman_maker-0.346.0.zip`](https://github.com/TREE-Ind/MetahumanMaker/releases/download/v0.346.0/metahuman_maker-0.346.0.zip) from the [public preview release](https://github.com/TREE-Ind/MetahumanMaker/releases/tag/v0.346.0).
2. In Blender, open **Edit → Preferences → Add-ons → Install from Disk**.
3. Select the ZIP, enable **MetaHuman Maker**, then open the 3D View sidebar with **N** and choose **MetaHuman Maker**.
4. Set a writable **Project Folder** and a character name. Save through the add-on as you work.

The ZIP includes the Windows groom Alembic helper. Do not extract an entire Git checkout into Blender's add-on directory. When upgrading, finish active operations, save the character project, install the new ZIP, and restart Blender if modules from an earlier version remain loaded.

## 2. Prepare the numerical worker

From a source checkout with Python 3.12:

```powershell
python tools/setup_surface_solver.py
```

This creates a separate `.venv` and installs the surface solver's dependencies. Point the add-on's worker Python setting to `.venv/Scripts/python.exe`. Keep worker dependencies out of Blender's and Unreal's embedded Python environments.

Basic wrapping does not require installing every optional model. Photo landmarks/GNM support can be provisioned with `python tools/setup_models.py`; that command installs packages and downloads data. See [optional backends](#optional-learned-backends) before enabling learned skin tools.

## 3. Prepare your reference library

Native conform needs a matching MetaHuman template with the original LOD0 topology and UVs. A source mesh can have arbitrary topology; the template cannot.

1. Obtain your own licensed neutral head/body references through your MetaHuman tools.
2. In the Start panel's **Reference Library** section, assign the **Head Template** and **Body Template**, and the corresponding left eye, right eye and teeth when available.
3. Set the facing axis and choose **Save Assigned Templates as Library** with a new `.blend` filename.
4. Reuse it with **Choose Reference Library**.

The library stores static copies, packed still-image materials and available body-pose metadata. Original source objects remain intact. Complex shaders that rely on other scene objects need standalone baked materials before packing. A regular `.blend` is not a reference library merely because it contains meshes. Missing roles are reported; matching references are required for their respective workflows.

Reference libraries, DNA, MetaHuman textures and model weights are not distributed with the repository. The procedural demo is useful for learning the interface; it does not supply native MetaHuman content.

## 4. Configure Unreal exchange

1. Close the target Unreal project.
2. Extract the bridge release ZIP so your project has `Plugins/CreateMetaHumanBridge/CreateMetaHumanBridge.uplugin` and the matching `Binaries/Win64` files.
3. Open the project, enable **MetaHuman Maker Bridge**, **Python Editor Script Plugin**, and the required MetaHuman plugins. Enable **Alembic Groom Importer** for groom exchange.
4. Enable **Python Remote Execution** in the Python plugin settings and restart the editor when prompted.
5. Install the optional MetaHuman Creator Core Data for your engine installation.
6. In Blender, set the Unreal engine directory, the `.uproject` and the compiled bridge folder.

The bridge's plugin declaration enables its required native dependencies. Keep Python remote execution on a trusted local connection: it permits scripts to execute inside your editor. A second unrelated Unreal project is not a substitute for the linked project.

For Anatomical body shaping, the bridge folder must contain its `.uplugin` and compiled binary, and the engine must have its optional body data installed. The add-on launches an isolated native worker; wait for **Anatomical preview ready** before applying.

## Optional learned backends

Inspect a pinned download plan first:

```powershell
python tools/provision_optional.py pisa-sr
python tools/provision_optional.py marigold
python tools/provision_optional.py texture-editor
```

These commands report download and disk requirements. Add `--download` to fetch a chosen profile. They do not configure Torch/CUDA or select the backend inside Blender. Models can be several gigabytes; provision only what you plan to use. Existing weights keep their upstream terms.

| Profile / tool | Purpose |
| --- | --- |
| Fast Restoration | Local Real-ESRGAN restoration prior; configure its runtime/model paths in the skin tools. |
| `pisa-sr` | Diffusion super-resolution prior with a separately configured CUDA Python environment. |
| `texture-editor` | Quantized reference-editing model/runtime, selected through the skin solver. |
| `marigold` | Optional material decomposition. |

See [skin reconstruction](workflows.md#skin-projection-repair-and-reconstruction) and [third-party notices](../THIRD_PARTY_NOTICES.md).

## Build from source

Source users additionally need Visual Studio 2022 C++ build tools, a compatible Windows SDK and the installed UE third-party SDKs.

```powershell
python tools/build_groom_alembic.py --engine 'C:/Program Files/Epic Games/UE_5.8'
python tools/build.py --addon-only
```

This produces `dist/metahuman_maker-0.346.0.zip`. The native helper must exist before packaging; source archives deliberately omit executables. See the [groom helper README](../create_metahuman/native/README.md).

Build the bridge against your own engine:

```powershell
& 'C:/Program Files/Epic Games/UE_5.8/Engine/Build/BatchFiles/RunUAT.bat' BuildPlugin -Plugin="$PWD/plugins/CreateMetaHumanBridge/CreateMetaHumanBridge.uplugin" -Package="$PWD/build/CreateMetaHumanBridge-UE5.8" -TargetPlatforms=Win64
```

Copy that packaged plugin into your project's `Plugins/CreateMetaHumanBridge` folder. The output directory itself contains the `.uplugin`; preserve this nesting. Do not copy engine SDKs into the repository. Continue with the [quick start](quickstart.md).

## Existing projects and scripts

The product and Blender sidebar are now **MetaHuman Maker**. The Python module `create_metahuman`, `cm.*` operators, `cm_settings` / `cm_features` scene properties, and Unreal plugin module `CreateMetaHumanBridge` retain their identifiers for compatibility. Keep the Unreal plugin folder named `CreateMetaHumanBridge`; its displayed name is **MetaHuman Maker Bridge**. Existing workspaces and skin brush libraries are reused. The compiled bridge also retains its legacy Blueprint category names; callable node identities and behavior are unchanged.

## Published packages

[Blender installer](https://github.com/TREE-Ind/MetahumanMaker/releases/download/v0.346.0/metahuman_maker-0.346.0.zip) ? [UE 5.8 bridge](https://github.com/TREE-Ind/MetahumanMaker/releases/download/v0.346.0/MetaHumanMakerBridge-UE5.8-0.29.0.zip) ? [SHA256 checksums](https://github.com/TREE-Ind/MetahumanMaker/releases/download/v0.346.0/SHA256SUMS.txt) ? [Release notes and source archive](https://github.com/TREE-Ind/MetahumanMaker/releases/tag/v0.346.0)
