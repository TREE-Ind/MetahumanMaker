# Release notes

## Source updates after v0.346.0

- Shaded Unreal captures now request realtime rendering through a scoped bridge override and temporarily allow background rendering. Completion, errors and screenshot timeouts release that capture's override and restore the prior performance setting; camera and pose are retained. Capture receipts record the warmup and release state. The realtime override requires the rebuilt bridge; older bridges retain the background-rendering fallback. A warmed viewport is not a guarantee of temporal image convergence.

- Push Groom now refreshes a rest-head snapshot owned by the linked character before binding. A head revision can therefore reuse the normal Push/Equip/Assemble workflow instead of silently binding to the previous face. Explicit targets owned by an artist or another character remain explicit, and strict custom-groom position checks remain unchanged.

- Added groom-group filtering to guide/strand selection, available group IDs, and **Remove Selected Curves** in the guided and advanced hair panels. Removal keeps linked copies independent and retains surviving curve attributes; the real 82,589-curve Updo workflow exercised group selection, whole-group removal and native transport export.

- Added optional native iris palette and detail groups: primary/secondary hue, secondary lightness, blending, rotation, shadow detail and limbal-ring size/softness. Each group is disabled by default; legacy requests retain these native values. Inputs use the installed Creator ranges and hue remains a palette coordinate rather than RGB/HSV.

- Photo fits now provide separate observed iris-color references, with pupil/lid/highlight filtering and small/obscured-eye rejection. The photo panel shows reference-bound sRGB values without changing eye materials or native settings. These cues retain photographic lighting and are not recovered albedo or calibrated Creator controls.

- Multi-view shape refinement now permits frontal landmarks to move toward the primary reference instead of penalizing all changes from the accepted fit equally. Separate jaw, brow, nose, eye and mouth checks reject regional regressions; reports distinguish total movement from movement outside the permitted correction corridor. Generated supports retain their lower weight and head coefficients remain bounded. This is landmark fitting, not a guarantee of likeness.

- Photo shape fitting now includes a centered contour term for each eye, in addition to the existing positional landmarks. It fits eyelid aperture, width and corner slope without using eye placement as a substitute for shape. Multi-view proposals also check each primary eye contour for regression. Head coefficient bounds, regional checks, generated-view weighting and untouched tooth/eyeball identity parameters are retained; reports expose the contour errors separately from total landmark error.
- Added camera-matched supporting-photo overlays for multi-view face proposals. Source/result comparison retains the selected reference and opacity; original reference bytes bind each camera so replaced images cannot silently reuse stale calibration. Primary, three-quarter and profile switching and view restoration were exercised on the saved photo demo.

- Added optional native **Pupil Dilation** and **Iris Lightness** controls to Creator eye appearance. Older eye requests retain the artist's existing values. Pupil input uses the native 0.85–1.2 range.
- Added **Crown Shape** in guided and advanced grooming: **Crown Lift** adjusts upper scalp strand height and **Crown Sweep** moves raised lengths horizontally around unchanged root anchors. Widths, colors, guide/group attributes and root UVs are retained. The operation supports Blender Undo and keeps linked data independent. Review scalp clearance and the whole silhouette before publishing; this is a shaping tool, not a collision solver.
- Native assembly now distinguishes installed preset hair adapted by Unreal's RBF head fitting from a changed source. After a fresh native build, preset adaptation still requires unchanged source data, matching ordered strand attributes and a valid bound target. Receipts identify adapted positions separately; custom groom positions retain strict comparison and every result needs visual review.

- Restoring or rolling back a live skin preview now updates material slots without triggering Blueprint construction scripts. This retains the actor's existing components and LODSync references during expression review instead of invalidating the pending restoration state.

- Ordinary Wrap numerical fitting now runs in an owned, cancellable Blender worker for heads, bodies and connected head/body fits. Frozen geometry and exact sampled landmark positions retain the direct solver's numerical inputs; changed geometry, settings or landmark positions refuse publication. Cancellation retains the accepted character and detected points. Final mesh/attachment publication remains on the main thread and can pause complex retained characters.

- Wrap publication now refuses newly collapsed triangles. Automatic head candidate-validation failures can retry once with temporary conservative protection; invalid inputs and other operation types stop normally. Recovery is recorded on the result and in successful Auto Wrap receipts, shown in Refine, and leaves the user's saved setting intact. Both failed attempts retain the accepted character.

- Exposed **Conservative Fit** and head **Front Face Only** protection in the main Wrap step, with a facing selector for manual frontal-shell fitting and clearer tooltips. Saved values and solver behavior are unchanged; these common choices no longer require Advanced settings.

- Project-owned expression reference reuse now hashes buffers incrementally and checks destination space for missing files and rollback copies before staging them. Matching content-addressed resources remain reusable without another copy.

- Save Project now checks destination space before packing images, writing dirty project textures or creating its folder. Its estimate covers whole-file geometry and external images scheduled for packing. A real-head save/resume pass retained geometry, an artist sculpt key, packed paint pixels, roles and units; an injected low-space refusal retained the unsaved image and geometry.

- Retained SIMPLE sculpt reuse now preserves the owned shading helper and accepts its managed normal-transfer modifier. Unrelated live modifiers remain blocked. Fixed first-time import of geometry-only donors without map files, and refreshed the destination view layer after rollback cleanup so the original failure is reported instead of a stale selection error.

- Fixed cross-project part reuse reading stale saved world transforms. Donor scenes now evaluate their parent/constraint transforms before parts are detached, preventing an unintended offset during import. A real photo-head/warrior-body reuse pass retained the head's saved positions and the existing body's geometry and maps.

- Custom body saves now reject head-only results and unfinished edits, check destination space, and stage the OBJ/native-state companion before replacing previous files. Publication failures restore previous outputs; blocked restoration retains backups with their location in the error. The real body save/load workflow retained all 32,334 vertices within OBJ precision.

- Shaded capture receipts now include the piloted camera's exposure inputs, override flags and blend weight to support consistent human/agent reviews. Unavailable camera diagnostics leave the image capture usable; these inputs do not represent combined volume/viewport exposure.

- Native face captures now record evaluated wrinkle/color material drivers and bound animated textures. The panel shows the last captured active-driver count. These readings establish material inputs; inspect a correctly exposed image before accepting skin appearance. Fixed shaded capture failing after its screenshot because groom registration was queried through an unavailable Unreal Python method.

- Sculpt detail updates now check destination space for input/output carriers and next-level geometry before staging files. Retained wrinkle updates size their scene/layer allowance instead of relying solely on a fixed 2 GiB threshold. Requests retain the estimates. Pre-launch failures clean owned carriers, and failed wrinkle launches restore the preceding pending markers. These estimates do not reserve disk space.

- Groom renderer capture now reports live Niagara controller state and simulation age separately from authored physics flags. Shaded review receipts include these readings with the rebuilt bridge, allowing an agent to distinguish enabled settings from an advancing solver. This does not measure GPU particle displacement or certify collision quality.

- Added Push Physics choices for captured settings, static hair, Groom Rods and Groom Springs. Explicit modes change simulation flags and solver selection on the new groom revision, including authored LOD overrides. Source force/constraint settings remain intact, and receipts report the chosen policy and resulting flags. Animated collision and appearance still need character-specific review.
- Smart Paint viewport strokes now follow connected skin across UV seams rather than stamping an atlas disk. Nearby disconnected surfaces and unrelated UV islands are excluded; the cursor follows the surface brush radius. Shared atlas texels remain shared. Agents can use the same surface selection through `PaintSession.stamp_surface` with ordinary stroke history and Apply/Cancel.

- Added authored-template protection before removing review-created groom overrides. Persistent editable settings and owned subobjects are compared independently from the groom/binding/material-slot/group fields, preventing Restore from discarding later transform or other component edits. Snapshot/ownership persistence and guarded restoration passed a saved-level/editor-restart workflow.

- Added **Recover Saved Groom Review**, with project-local character/component/state/date choices and file browsing. Reconnection preserves the last Push and changes no assets; malformed, finished, foreign or conflicting active reviews are refused. New receipts retain the exact Unreal project, and Keep/Restore target it explicitly. Child override ownership and restoration also passed a real saved-level/editor-restart workflow.

- Groom Keep can create a child-only override for inherited Blueprint SCS components. Restore removes a review-created override rather than freezing parent values into the child, and retains pre-existing overrides. Read-only defaults lookup, parent-source guards and saved ownership markers protect parent assets and later artist edits. Rebuild the matching bridge for these APIs.

- Added explicit atlas-mask strokes to the Smart Paint Python session API, including UV-layout checks and normal stroke undo/redo. Agent-authored selections use the same reference correction and Apply/Cancel lifecycle as brush strokes.

- Added viewport-only Beauty, Albedo, Shape and Surface Normals inspection in LookDev. Inspection retains material graphs/maps/geometry and restores prior viewport preferences when returning to Beauty or disabling LookDev. Background studio renders explicitly require Beauty.

- Added optional same-project imported card/helmet fitting for position-only strand edits. Root-aware local rigid proposals and shape regularization feed native stale-input, topology/UV and distortion checks before private mesh creation. Rejected source meshes retain explicit strand fallback. This does not generate cards or rebake atlases; visual and animated acceptance remain required.
- Groom replacement can prepare a private assembled-face binding while retaining the Creator-rest-head binding and durable recovery records. Native JSON import now initializes distinct group IDs from raw curve data instead of leaving duplicate group identifiers.
- Added renderer-thread groom representation inspection, separate from forced LOD requests, and included it in shaded facial capture receipts with the matching rebuilt bridge.
- Groom captures retain asset LOD mode and automatic bias. Blending conflicting asset policies stops before import; legacy records keep the prior default policy. Manual/discrete representation review remains distinct from project automatic/continuous strand LOD behavior.

- Native assembly now inspects active selected groom dependencies before character initialization, rejects missing sources and groom inputs in the replaceable output folder, and rechecks inputs before dispatch. A completed native actor update must also contain unchanged pre-build source fingerprints and distinct bound components matching those sources within native round-trip tolerance. Missing or changed grooms fail the assembly receipt instead of reporting success. Rebuild the bridge from this source tree for the new fingerprint API. These checks do not certify other wardrobe types, animation or appearance.

- Groom replacement now maps component group controls by source identity through reordered blends, independently for instance and blueprint defaults. Known added sources use incoming settings; unknown legacy captures retain bounded ordering. Ambiguous controls stop before mutation, with **Use Incoming Group Settings** as an explicit alternative stored in the review. Receipts record the mappings. Blueprint updates stage complete material/group state before native initialization and validate the group result. Live transient preview actors can now be reacquired in their original editor world instead of producing a misleading level error.

- Fresh groom captures now track native card/mesh sources. Push preserves imported representations, atlases and material slots for same-project groups whose complete strand geometry/attributes match, including reordered Blender blends. The rebuilt bridge compares full source descriptions with a 0.001 cm position tolerance. Changed groups and unsupported sources retain explicit strand fallbacks; this does not deform cards to match sculpted strands. Exchange status now points artists to replacement preview and reports preserved representations.

- Photo projection now fades inward from unsupported texels to the clean fallback over a narrow, resolution-scaled band. The fade follows covered UV pixels, leaves pixels beyond that band unchanged, preserves image alpha, and records its radius/affected texel count in the projection report and applied image metadata. Native nostril-thickness territories receive a longer anatomical transition to soften jagged inner-edge color changes without broad face smoothing.

- Photo-color visibility now compares surface depths at the same photo-pixel centers and blends occlusion coverage across the sampling footprint. This replaces angle-amplified depth tolerance, which could transfer color through thin rims onto hidden cavity surfaces, and reduces hard pixel steps at occlusion boundaries.

- Separate sculpt sessions now check destination disk space before saving/copying a character and again before opening or resuming the author window. Estimates account for private scenes, map copies and writable images; a failed check retains existing recovery files.

- Added native groom material provenance across Blender blends and group renumbering. Replacement follows source asset/group/slot ancestry before legacy shader matching, preserving distinct per-group instance overrides when base shaders are shared. Known added sources keep their authored material. Instance replacement stages complete material/group state before native groom initialization to avoid partial editor reconstruction.

- Added receipt-backed **Preview Replacement on Character**, **Save Groom in Blueprint**, and **Restore Previous Groom** controls. Replacement targets the existing component instead of adding duplicate hair. Actor controls and blueprint state are recorded separately; interrupted saves/restores retain a recovery journal. Closing a conflicted review can retain current artist edits without changing assets.
- Fixed completed groom exchanges being revived on project load, which could block immediate operations and replace newer saved delivery links. Linked-project checks now also guard native groom commands.
- Groom replacement preserves assigned material bindings rather than strand-renderer fallback values for card/mesh slots. Genuinely new slots use authored target materials; ambiguous matches remain explicit. Existing ordered group controls survive replacement, with native defaults for added groups. Shared inherited component templates are protected from unintended edits.

- Fixed reviewed groom replacement failing when exchanged groups have different material-slot counts. Assigned artist materials map by slot identity or an unambiguous shared source material; unrelated source assets remain intact. Static strand imports now report strand fallbacks for source card/mesh LODs whose geometry is absent from the exchange.

- Added optional behind-ear strand routing using the rear ear envelope and original-contact eligibility. A whole-head guard prioritizes avoiding new or worsened surrounding contacts, with unchanged curves retained when movement/length limits make the route infeasible. Review static native placement and expressions before keeping a partial proposal.

- Added asynchronous, root-pinned groom surface-clearance proposals with bounded movement/length change, saved recovery, cancellation, explicit partial-result handling and sparse geometry reverts. Static sampled contact reduction requires visual review and does not guarantee ear tucking or animated clearance. Native imported root coordinates retain exact export precision when roots and transform are unchanged.


- Fixed native review freezing coarse hair/clothing when forcing only the face LOD. Explicit face selection now follows native character mappings with streaming active; render/streaming requests for skeletal meshes and grooms are captured and restored. Temporary sync changes use native field setters to avoid Blueprint reconstruction. Shaded capture checks requested LOD availability.
- Unified raw, dotted and Unreal underscore control aliases in capture verification, accepted explicit zero-valued neutral controls, and rejected invalid or unknown requests.
- Added **Capture Current UE View** to native facial review: receipt-backed 1080p viewport images with applied control metadata, process ownership/recovery and an Open Image action. Camera, playback and assets are retained.
- Fixed facial review remaining visually neutral when the Unreal viewport is not ticking: the matching native bridge now evaluates component animation, face post-processing and bone/render updates explicitly. Playback restoration also refreshes the restored pose. Shaded capture waits for rendering to settle and verifies evaluated control curves before recording an image.
- Protected compact photographed nostril shadows during multi-view skin transfer so they cannot stamp a second opening onto the outside of the fitted nose. Native skin supplies the fitted aperture; original photos remain untouched.
- Native pose capture now verifies each evaluated LOD, face identity, material isolation and control sample. Eye calibration still requires LOD0. Reduced-LOD capture resolves the renderer's material-slot overrides, including eyes when attachment sections are omitted.
- Added **Refine Shape from Supporting Photos** with joint identity/camera fitting, a camera-only baseline, protected frontal projection and lower weight for generated references. Applying proposes a source without replacing the wrapped/native character; packed references, feature controls and eye/dental identities are retained.
- Added up to four packed supporting photos, including explicit generated-reference provenance, for multi-view face texture projection.
- Added profile-aware registration, bounded facial-feature alignment, exposure matching, skin segmentation, surface tone continuation and UV gutters.
- Added 4K photo projection and input retention, with larger memory/time allowances and bounded-grid exposure correction. Existing projects keep the 2K default. Larger atlases do not invent source detail.
- Protected photographed eye apertures and native eyelid skin, including skin revealed by a compatible native blink reference. Older contaminated photo bases may need the saved clean skin preset reapplied.
- Limited generated-head wrapping to compatible facing surfaces with collision/distortion backtracking, and verified that in-place Unreal assembly actually updates the existing actor.
- Added **Retain Native Ear Detail** for generated heads, matching ear placement/size while retaining native folds. Marked-ear projection also checks surface orientation during wrap and refinement.
- Added optional **Repair Hair on Skin** to photo transfer, combining hair-connected line detection, brow/lip protection and local color completion from trusted photo skin. Original references stay untouched; inferred repairs and per-view counts are recorded.

## 0.346.0 — Public source preview

**Add-on 0.346.0 · Unreal bridge 0.29.0 · Windows x64 · Blender 5.2.x / UE 5.8.x**

### Character tools

Guided source/reference setup, mesh wrapping, photo-assisted starting faces, regional/anatomical body shaping, retained sculpt/detail layers, skin projection and paint repair, optional source-guided restoration/generative skin, coordinated material maps, groom exchange, native Creator delivery and rig/source/assembly orchestration.

Agent operation uses Blender Python through a connected bridge such as Blender MCP. Scene state, asynchronous workers, saved projects and durable native action receipts support inspect/edit/review workflows. The language model/agent is supplied separately.

### Latest changes

- Renamed the product, Blender UI, installer archives and TREE launch film to **MetaHuman Maker**. Existing project and agent identifiers remain compatible.

- Corrected the native **Masculine ↔ Feminine** control direction and made its value an absolute normalized trait position. New passes seed the saved native position; measurement controls retain relative behavior.
- Replaced the public front page with TREE Industries branding, real workflow screenshots, an MCP editing clip and a focused guide structure.
- Added installation, first-character, workflow, agent, troubleshooting, limitation and release documentation.
- Expanded the clean-source inventory to include curated documentation media and community files, with media/link/source-boundary checks.
- Kept optional models, native character content, local projects and credentials outside the public source tree.

### Validation scope

Local development workflows were exercised on Blender 5.2.3 and Unreal Engine 5.8.2. The release's source/archive checks and packaged Blender smoke check are recorded with its prepared artifacts. Source CI does not certify native Linux support or every character/LOD/expression combination.

The clothed Genie, photo study, Orc and alien illustrate real workflows. Their underlying assets and licensed reference content are not bundled. This preview retains experimental eye/lid calibration and artist review requirements; see [known limitations](known-issues.md).

Original project code remains MIT licensed, including commercial use. Dependencies and user content retain their own terms; see [third-party notices](../THIRD_PARTY_NOTICES.md).
