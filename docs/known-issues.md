# Support scope and known limitations

[README](../README.md) · [Troubleshooting](troubleshooting.md) · [Release notes](release-notes.md)

This is a source preview with locally exercised workflows, not certification of all inputs or render combinations.

| Area | Current boundary |
| --- | --- |
| Platform | The launch package targets Windows x64, Blender 5.2.x and UE 5.8.x. Source CI on Linux validates packaging, not native platform support. |
| Source fitting | Strong stylization, missing regions and difficult hands/feet/ears can need regional authoring. Source and template have different topology; their full vertex buffers are not expected to be identical. Automatic landmark detection and ordinary Wrap numerical fitting run in cancellable workers. Final mesh/attachment publication still runs on Blender's main thread and can pause the UI on complex retained characters. |
| Photo fitting | A starting identity from a limited view. Likeness, occluded surfaces, body design and hair still require authoring. |
| Body controls | Anatomical primary controls use the installed native model; secondary regional controls are sculpt adjustments. |
| Skin reconstruction | Added generative detail is invented. Source constraints/protection masks bound the edit but do not make it a measured scan. |
| Derived materials | Albedo-driven relief and roughness/cavity contributions are artistic inference, not recovered physical depth or reflectance. |
| Authored animated maps | All three 4K wrinkle/color slots have been exercised in the native skin shader. Custom-map neutral preservation, continuous motion and lower-LOD appearance still need review on the intended character. |
| Native shaded screenshots | Capture temporarily enables realtime/background rendering with the current bridge. High-resolution viewport screenshots can still vary between unchanged frames. Compare repeated baselines before attributing pixel differences to a texture or shader edit; quantitative neutral-map preservation remains unverified. |
| 8K / dense sculpt layers | High memory/disk costs; worker estimates are best effort. Use 4K/modest detail for the first complete workflow. |
| Eyes and lids | Corrective capture/staging is experimental. Strong squint, mixed gaze/blink, groom overlap, broader motion and lower LODs can expose defects. Check your actual result. |
| Native services | Rigging and texture requests depend on your Epic account/content access, installed plugins and service availability. |
| Artist round-trip | Supported appearance/property/package transfer is available. Arbitrary custom shaders, dependencies or every artist change are not guaranteed lossless. |
| Groom replacement review | The current rebuilt bridge can create child-only overrides for inherited Blueprint SCS grooms. Restore removes only review-owned overrides and retains pre-existing ones; parent templates remain unchanged. Native C++ templates and arbitrary construction scripts are not covered. Instance and blueprint group controls follow source identity separately; unknown legacy captures retain ordered compatibility. Ambiguous controls need the explicit incoming-settings choice. Live transient actors require their original editor session and persistent groom resources. If later artist edits or ownership conflict with a snapshot, close the review to retain them. Continuous animation still requires review on the actual character. |
| Grooms | Fresh captures can reuse matching same-project imported cards/meshes or optionally fit private copies for position-only strand edits. Numerical gates can reject individual source meshes; rejected LODs retain reported strand fallback and its rendering cost. Attribute/layout changes, legacy captures, missing meshes, different projects and unsupported procedural recipes remain unsupported for fitting. This is not new card generation or atlas rebaking. Actual representation, silhouette, intended-distance transitions and animated deformation require review. Surface-clearance proposals measure sampled contacts; pinned roots and movement limits can leave unresolved strands. Continuous/animated clearance is not certified. Static exchange does not generate simulation caches. |
| Browser delivery | Native shader appearance, strand rendering and sampled facial controls are approximations. Static GLB contains neither native DNA nor a complete rig/groom bundle. |

Native assembly verifies active selected skeletal groom source data and bound output attributes. Installed presets can receive Unreal's native RBF head adaptation; receipts distinguish this from source-coordinate equality. Custom groom positions retain strict comparison. This does not certify other wardrobe types or preserve arbitrary manual blueprint changes. Unsafe groom source dependencies stop before assembly; keep the input groom, wardrobe item, binding and rest meshes outside the output folder and rebuild the current bridge when using the source preview.

For a final character, review the saved neutral, multiple viewing angles, representative expressions/gaze, the dressed silhouette, grooms and intended LODs in the target renderer. A successful binding or geometry audit alone does not establish visual quality.
