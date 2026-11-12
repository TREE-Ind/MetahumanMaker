# Agents and automation

[README](../README.md) · [Installation](installation.md) · [Architecture](development.md#architecture)

MetaHuman Maker supplies character-authoring operations, saved project state and asynchronous worker/native jobs. Connect your own trusted agent through Blender MCP or another Blender Python execution bridge. No LLM, hosted chat service or universal agent endpoint is bundled.

## The operating loop

**Inspect → propose an edit → invoke the add-on → observe the job → review the result → apply/save.**

Use the same operators and project roles as the artist-facing UI. Keep the accepted character available for direct editing. Visual review is still part of the work: inspect front/profile/back, then representative expressions before accepting a rigged result.

### Inspect the current project

Execute inside Blender:

```python
import bpy, json

scene = bpy.context.scene
s, f = scene.cm_settings, scene.cm_features
print(json.dumps({
    'character': s.character_name,
    'status': s.status,
    'source': s.source.name if s.source else None,
    'head': s.result.name if s.result else None,
    'body': f.body_result.name if f.body_result else None,
    'body_editing': f.body_editing,
    'native_character': f.unreal_character,
}))
```

### Import a source

Configure the reference library and worker first. Set `source_path` to the absolute path of your source mesh, then:

```python
import bpy

bpy.ops.cm.import_mesh(
    filepath=source_path,
    role='source',
    prepare_workflow=True,
    wrap_mode='FULL_BODY',
    placement='AUTO',
)
```

Choose head/full-body scope to match the reference and review alignment before wrapping. An import operator succeeding does not establish likeness or fit quality.

### Make an anatomical body change

Start in a project with an accepted body, configured bridge and installed native body data:

```python
import bpy

f = bpy.context.scene.cm_features
f.body_backend = 'NATIVE'
bpy.ops.cm.feature(action='BODY_PREVIEW')
```

In a subsequent agent request, set the desired position:

```python
f = bpy.context.scene.cm_features
next(c for c in f.body_controls if c.key == 'gender').value = 0.7
```

The native **Masculine ↔ Feminine** axis is absolute: negative masculine, positive feminine. Other measurement adjustments are relative to the current pass baseline. Poll between bridge requests:

```python
from create_metahuman import native_body

print(native_body.can_apply())
```

Once ready, inspect the silhouette. Apply only the intended result:

```python
import bpy

bpy.ops.cm.feature(action='BODY_APPLY')
bpy.ops.cm.save_project()
```

Do not block Blender's main thread with sleep/poll loops. Let its timers update and make later bridge calls. Preview first, inspect, then apply or use the UI's Cancel; do not bypass the active session by editing saved native state blobs.

## Delivery readiness and native actions

```python
import bpy
from create_metahuman import guided, unreal_jobs

s, f = bpy.context.scene.cm_settings, bpy.context.scene.cm_features
print(guided.check_delivery(bpy.context, s, f))
print(unreal_jobs.running())
```

After resolving delivery problems, `bpy.ops.cm.export_bundle(push=True)` saves and exports the character through the ordinary delivery checks. Existing destinations are protected; use the revision/fresh-destination workflow rather than overwriting artist work.

Native rigging, texture requests and assembly retain requests/receipts. If observation times out or Blender reopens, inspect the existing action before issuing another one. A lost client connection is not proof that Unreal stopped working.

After connecting native facial review and applying controls, use `bpy.ops.cm.native_face_review(action='CAPTURE_VIEW')` to capture the currently framed Unreal level viewport at 1080p. Observe the original `cm_face_review.receipt`; its `shaded_capture` records the image path, applied controls and verified evaluated curves. Inspect that image before accepting a pose. Capture retains the current camera/playback and saves no native assets. Restore previous playback with the existing `RESTORE` action when finished. Use the matching rebuilt native bridge so poses evaluate even when the editor viewport is not ticking.

## Practical agent boundaries

An automatic head wrap can retry once after a `mesh_ops.FitValidationError`, using temporary conservative protection. Other exception types and already conservative/manual/full-body fits are not retried. A recovered result records JSON in `result['cm_wrap_recovery']` with `attempts`, `method`, `primary_error` and `review_required`; Auto Wrap receipts expose it as `fit_recovery` after successful completion. Inspect likeness and difficult regions before accepting it. The guard checks newly collapsed triangles, not all intersections or pre-existing template defects.

In an interactive Blender session, ordinary Wrap stages numerical fitting in an owned Blender worker. Observe `auto_wrap_jobs.running()` and `auto_wrap_jobs.fitting()`; use `auto_wrap_jobs.cancel()` for either detection or fitting. Completed detection points stay available after fit cancellation. Receipts include `detection_worker_pid` and `fit_worker_pid`; a successful operator return while the worker runs is not fit completion. Changed geometry, settings or landmark positions refuse publication. Final mesh/attachment publication still runs on the main thread; a retained production head can pause there for several seconds. Background Python operators remain synchronous unless `auto_wrap_jobs.start_fit(context, guided=...)` is explicitly used.

`bpy.ops.cm.save_project()` checks destination space before image packing and file writes. `resource_budget.project_save_storage(scene)` estimates the whole-file carrier and external-image packing allowance; it is not a disk reservation. Resume the resulting `project.blend` with `bpy.ops.cm.project_file(filepath=..., part='RESUME')` only after saving the current work. Retain the project directory and check missing-image warnings; a saved scene can still contain unavailable external textures.

Use `bpy.ops.cm.save_custom_body(filepath=...)` for an accepted body after finishing edits. Keep the OBJ and `.cm-body.json` companion together for anatomical state. It rejects a head-only result and checks destination space before staging. Failure during publication restores old outputs; if that restoration fails, retain the named recovery directory. Import from Project remains the route for part maps and retained sculpt layers. A successful OBJ round trip does not certify neck color, expression deformation or Unreal appearance.

Native view receipts include `shaded_capture.skin_expression_state`: supported head material path, evaluated `values`, `active_count`, `driver_count` and `animated_textures`. Compare these inputs between neutral and expressions on the same material. An `available=False` record means the diagnostic is unsupported, not a neutral result. Active inputs do not prove visible wrinkles or color changes; inspect correctly exposed images and retain any failed capture receipt separately from a later successful one.

`shaded_capture.camera_state` records the piloted camera, exact-view flag, post-process blend weight and exposure settings/override flags. A free viewport or unsupported camera reports `available=False` without failing the image capture. Camera inputs exclude volume and viewport exposure overrides. Keep the same framing/exposure and return to neutral after expressions before attributing initial softness to texture content.

Use the normal export/texture-refresh path for full WM/CM maps. It invokes `skin_bake_layers.native_wrinkle_baseline` when needed, then `unreal_native_animated.prepare` to bind conversion to a verified neutral normal/albedo and source hashes. Do not pick a convenient reference normal independently: retained skin edits can make its basis differ from the full maps' authored neutral. Add-on CM outputs are full color maps; native numeric deltas need explicit DELTA encoding. A successful six-map bake/encoding check is separate from native animated material appearance acceptance.

Detail and retained-wrinkle update requests include `storage_budget` estimates. Low-space refusal occurs before creating their job folders or setting wrinkle pending markers. `resource_budget.scene_snapshot_storage(scene)` is a cheap scene-carrier estimate without mesh evaluation or pixel readback; it is not a disk reservation or a bake budget. On a preparation/launch exception, inspect the original job state rather than creating duplicate requests: pre-launch carriers are cleaned, while a live worker's inputs remain protected until exit.

Set `bpy.context.scene.cm_groom.simulation_policy` before the normal groom `PUSH`: `SOURCE` (default), `STATIC`, `RODS` or `SPRINGS`. The request snapshots the policy; changing the next-Push UI does not change an in-flight request. `SOURCE` retains captured group physics and LOD flags. Explicit modes affect all groups/LODs on the new revision only; other force/constraint settings remain captured. Native receipts include `simulation_policy` and per-group `simulation` authored flags. These describe configuration, not observed Niagara motion or collision acceptance. Review the actual character before accepting physics.

With the current rebuilt bridge, `CreateMHGroomLibrary.capture_groom_representation_state(component)` also includes `runtime_simulation`: live Niagara system paths, registered/active/paused flags, controller availability, execution state and elapsed simulation age. Compare two captures during normal playback to detect an advancing solver; desired age and asset enable flags alone are insufficient. The inspector waits for pending native work without stepping time or changing simulation settings. Controller advancement does not verify GPU particle positions, collisions or finished hairstyle quality. Normal shaded native view receipts include these diagnostics in `rendered_grooms`.

Check runtime physics in normal Unreal Simulate/Play mode. The still editor world can retain zero-age controllers even when the same groom advances normally in simulation. Treat that result as inconclusive for runtime delivery. Read-only runtime component inspection is separate from the add-on's editor-world replacement and restore actions; stop simulation before those editor mutations. Component simulation overrides and project scalability can also affect the final behavior.

For a local surface repair, `PaintSession.stamp_surface(world_point, polygon_index, erase=False)` uses the session's world-space geometry and base polygon indices. Call `begin()` before one or more stamps and `end()` after the stroke, then inspect `publish()` before applying. The footprint crosses connected triangle edges within the brush sphere, including UV seams, without jumping to disconnected nearby surfaces or unrelated atlas islands. Radius uses local texel density; this is a spherical surface selection, not an exact geodesic brush. Shared atlas texels cannot be edited independently. The older `stamp(uv)` remains an explicit atlas-space operation. Use normal session Undo/Redo and `finish(False)` to discard a proposal.

Groom Keep supports inherited Blueprint SCS components with the current source bridge. Preview records `blueprint_inherited`; Keep can create a child-only override and reports `blueprint_result.local_override_created`. Restore removes that review-owned override so later parent changes continue to inherit, or restores the preceding state of an existing local override. The durable `blueprint_override_token` must match component metadata before deletion. Do not fabricate ownership markers or bypass a conflict: use Close Review / Retain Current Edits when later artist ownership or properties changed. Parent Blueprint files join the source guards for inherited reviews.

Owned child removal also checks `blueprint_template_after`, a native snapshot of persistent editable settings and owned subobjects outside the controlled groom fields. A changed transform or other authored setting refuses deletion. `CreateMHGroomLibrary.snapshot_groom_template(component, archetype=False)` supplies this read-only snapshot; self paths are normalized across template ownership/reloading. External asset references remain paths, not deep graph snapshots. Do not rewrite snapshots to force a restore over an artist edit.

Reconnect a missing review link with `bpy.ops.cm.recover_groom_review(filepath='/path/to/review/preview.json')`. It validates the record/project and updates Blender's review link without dispatching an Unreal action or changing the last Push result. `groom_jobs.find_replacement_reviews(bpy.context)` lists locally saved candidates for the normal picker. Recovery refuses finished/foreign reviews and replacement of an unresolved current review; it observes an older pending exchange rather than resubmitting it. Open the recorded Unreal level before invoking the normal Keep/Restore actions. New receipts retain the exact Unreal project; those actions target that project rather than a possibly different scene selector.

Smart Paint also accepts an explicit atlas mask through `PaintSession.apply_mask(weights, expected_uv=uv_signature)`. Use finite 0–1 weights with the same dimensions and lower-left image origin as Blender pixel buffers. Supply the current UV signature from `skin_projection.uv_layout_signature(obj)`; a changed layout is refused. The mask becomes one stroke using the session's selected reference/correction settings, so normal stroke undo/redo, publishing and Apply/Cancel remain available. This does not decide which skin needs repair or infer a replacement texture. Finish with `session.finish(False)` to discard a candidate, or review it before accepting.

With LookDev enabled, set `scene.cm_features.lookdev_inspection` to `DIFFUSE_COLOR`, `SHAPE`, `NORMAL` or `COMBINED` for albedo, geometry, shaded-normal or beauty inspection. This updates 3D viewports without changing material graphs, maps or geometry. Return to `COMBINED` when finished; background studio previews require Beauty. A color-only artifact is not evidence that the mesh needs smoothing. Compare the same framing across modes and confirm the result in Unreal before declaring a native skin defect fixed.

For photo sources, add supporting views with `bpy.ops.cm.add_supporting_photo(filepath=..., generated=True)` when they were generated. `bpy.ops.cm.refine_photo_views()` starts the asynchronous shape proposal; monitor `cm_features.photo_fit_status`, then use `bpy.ops.cm.apply_photo_fit()` only when ready. Applying retains the wrapped/native character, supporting images, current feature controls, body-placement record and paired wrap anchors. Inspect the new source before **Update Face**, color projection and native redelivery. A changed photo, calibration, trust flag, model or geometry invalidates the pending proposal. Generated references are inferred evidence and have lower weight than photographs.

- Operate on explicit source/template/result roles; preserve artist originals and accepted paint.
- Use supported regional tools and protect unrelated anatomy. Record what changed when making a substantial solver adjustment.
- Choose an invention budget with the character's design requirements. Protect identifying marks before a generative pass.
- Treat geometry, UVs and map provenance as linked. Resolve stale maps through the normal update path; explicit adoption of custom maps is an authoring decision.
- Inspect actual geometry and images, not only a successful return code. Report defects and uncertainty at the scope you checked.
- Save project state and use cancellation/recovery paths instead of launching duplicate workers or creating many experimental copies.
- Groom clearance uses `bpy.ops.cm.groom(action='SOLVE_CLEARANCE')`, followed by `APPLY_CLEARANCE`, `REVERT_CLEARANCE`, `RESUME_CLEARANCE` or `CANCEL_CLEARANCE`. Configure `scene.cm_groom.clearance_head`, `clearance_region` and the physical limits first. Poll `groom_clearance.running()` / `ready()` and inspect `scene.cm_groom.clearance_report`. Applying unresolved proposals requires setting `clearance_accept_partial` for that proposal after review. Collision counts do not establish artistic or animated acceptance; inspect native front/profile/back views and preserve the original groom when placement gets worse.
- For ear tucking, set `clearance_region='EARS'`, `clearance_route='TUCK_EARS'` and `clearance_back_axis` in the collider's local frame (standard native heads: `(0, 1, 0)`). Nearest-surface mode remains `SURFACE`. Compare sampled whole-head contacts as well as the ear view; the surrounding guard prioritizes avoiding worsened contacts, so an infeasible strand can keep its original shape. Older saved nearest-surface proposals retain default routing settings when resumed.
- After Push, select the existing character actor in Unreal and call `bpy.ops.cm.groom(action='PREVIEW_REPLACE')`. `replacement_component` overrides the default component name from `wardrobe_slot`. Poll `groom_jobs.running()` / `poll()` and inspect `exchange_result`. Review the actual character, then call `KEEP_REPLACE` to save its blueprint or `RESTORE_REPLACE` to restore the preceding actor and any blueprint change made by this review. `replacement_origin` retains the recovery folder. `CLOSE_REPLACE` closes tracking without overwriting current artist edits. It does not save the level or undo a saved blueprint. `APPLY_ACTOR` is the separate-component layering action and should not be used to replace existing hair.
- Inspect Push receipt fields `representations` and `strand_lod_fallbacks`. Fresh captures can retain same-project imported card/mesh geometry for fully compared, matching strand groups. For position-only edits, `scene.cm_groom.fit_lod_meshes` enables private mesh fitting through the configured numerical worker. Rows with `deformed=True` list `private_meshes`; inspect their candidate fit receipts and rejected LODs. The worker's motion-retention metric measures agreement with its proposed deformation, not likeness. Changed attributes/layout, unavailable origins and unsupported procedural recipes retain fallback. Rebuild the current bridge. Atlases remain shared native dependencies, not portable content embedded in strand JSON.
- Native `CreateMHGroomLibrary.capture_groom_representation_state(component)` reads renderer-selected `geometry`, `selected_lod` and `lod_visible` per group; `requested_lod` is separate. Current shaded face captures include these records as `rendered_grooms`. Inspect the image too: valid geometry selection is not proof of acceptable hair shape or lighting. A Creator-rest binding may require a private assembled-face binding; replacement receipts record `binding_retarget` and `owned_bindings`. Retain those assets and the review folder for recovery rather than deleting them after Keep.
- Fresh native captures also retain independent `group_sources` ancestry. Inspect replacement receipt `group_controls` and, after Keep, `blueprint_group_controls`; component width/shadow/other controls follow source groups rather than their new indices. Set `scene.cm_groom.replacement_group_defaults=True` before a new preview only when incoming asset settings are intended. The stored review policy owns Keep even if this next-preview control later changes. Ambiguous unequal controls require that explicit choice; legacy captures without ancestry retain ordered compatibility.

For operator definitions, begin with [guided workflow](../create_metahuman/guided.py), [native body jobs](../create_metahuman/native_body.py), [Unreal jobs](../create_metahuman/unreal_jobs.py) and [native groom review](../create_metahuman/unreal_groom_revision.py). These are Python interfaces, not a promise that every experimental helper has a polished artist workflow.

For native material comparisons, retain repeated unchanged screenshots as a rendering-noise control. A successful Shaded Capture and zero expression drivers do not prove neutral appearance equality. Check `shaded_capture.render_warmup` for active/released viewport overrides, then compare A/B changes against unchanged-baseline variation. Use the final renderer when the viewport baseline is unstable.
For photo eye matching, compare the reference and a frontal neutral render with independent image landmarks using `photo_eye_core.iris_guidance` and `compare_iris_observations`. Inspect each eye's observed color, aperture and iris-to-eye width. Palette parameter readback is insufficient for assembled characters with baked iris textures: update Creator eyes, reassemble the existing Blueprint, and review the rendered result. Keep lighting and framing fixed for before/after comparisons; image-space color differences are not recovered albedo or an overall likeness score.

Use `photo_fit_core.review_rendered_landmarks` for a delivered face, passing independent 478-point pixel detections for the reference and frontal neutral render. It reports regional error after iris-center translation/scale alignment, without rotating or warping the images. Preserve the images with the report and judge the actual likeness; source-solver error reductions alone do not establish that the final native character improved.