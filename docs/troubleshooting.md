# Troubleshooting

[Installation](installation.md) · [Known limitations](known-issues.md) · [Report a bug](../CONTRIBUTING.md#reporting-a-bug)

| Symptom | Check / next step |
| --- | --- |
| No **MetaHuman Maker** sidebar | Install the add-on ZIP, enable it in Preferences and press **N** in a 3D View. A GitHub source archive is not the installer. |
| Add-on fails to enable | Check Blender's version and the error text. The launch minimum is 5.2.0. Restart after replacing an older loaded package. |
| Worker cannot start | Select the separate worker Python created by the setup tool. Do not point it at an incomplete environment or Blender's embedded Python. |
| Asked to choose a model/reference | A reference library is your licensed topology/data; a learned model is an optional backend dependency. Configure the requested one rather than downloading everything. |
| Reference library refused | Use the add-on's assigned-template library saver. Check tagged roles, UVs, facing and required head/body/eye/teeth parts. |
| Anatomical mode unavailable | Verify the compiled bridge folder, matching engine and optional MetaHuman body data. The source-only plugin folder is insufficient. |
| Native body preview still starting | First engine/worker initialization can take time. Observe the existing job and wait for **Anatomical preview ready**. |
| Unreal not found / wrong editor | Open the linked `.uproject`, set the engine path, enable Python remote execution, and check the bridge build/engine match. |
| Duplicate or pending native request | Resume/observe the original request's receipt. Do not submit a second rig or texture request just because the client timed out. |
| Insufficient memory or disk | Use **Memory and Disk Estimate → Check Locally**, reduce source density/subdivision/map size, or close heavy applications. Preserve the saved project before cleanup. |
| Stale final maps | Geometry, UVs or bake inputs changed. Update the relevant maps or explicitly adopt genuinely authored custom maps; do not suppress the checks. |
| Generative skin does not run | Configure only the selected backend's runtime/model paths and worker environment. Provisioning model files does not also install a compatible CUDA/Torch environment. |
| Skin artifacts or unexpected marks | Compare the preserved projection; reduce invention, add protection/repair masks and inspect UV seams, nostrils, eyelids, lips and ears. |
| Albedo changed but other maps did not | Review the material derivation/bake settings and freshness. A color repair does not automatically rebuild every dependent map. |
| Expressions expose eye/lid defects | Return to neutral and inspect the actual rig/attachments. Use regional/corrective workflows cautiously; broader motion and lower LODs remain experimental. |
| Creator updated, assembled copy did not | Reassemble the updated Creator character to a fresh destination, then inspect that output. |
| Web bundle differs from Unreal | Browser shading and sampled facial combinations are approximations. Confirm whether you used static GLB or the separate native bundle workflow. |

## Recovery before retrying

Keep the project folder, original inputs and action receipts. Finish or cancel active sculpt/paint/body sessions through the add-on. Reopen the saved project to restore its state and observe pending native actions. A preview, a saved character and an assembled copy are distinct stages.

## A useful bug report

Include the OS, Blender/add-on/UE/bridge versions, the exact action, expected/actual behavior and relevant error text. Add front/profile/back screenshots for fit problems and before/after material crops for texture problems. Use a small redistributable or procedural reproduction. Redact private paths and remove credentials; do not upload licensed templates, DNA, weights or customer scans.
