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

## Practical agent boundaries

- Operate on explicit source/template/result roles; preserve artist originals and accepted paint.
- Use supported regional tools and protect unrelated anatomy. Record what changed when making a substantial solver adjustment.
- Choose an invention budget with the character's design requirements. Protect identifying marks before a generative pass.
- Treat geometry, UVs and map provenance as linked. Resolve stale maps through the normal update path; explicit adoption of custom maps is an authoring decision.
- Inspect actual geometry and images, not only a successful return code. Report defects and uncertainty at the scope you checked.
- Save project state and use cancellation/recovery paths instead of launching duplicate workers or creating many experimental copies.

For operator definitions, begin with [guided workflow](../create_metahuman/guided.py), [native body jobs](../create_metahuman/native_body.py) and [Unreal jobs](../create_metahuman/unreal_jobs.py). These are Python interfaces, not a promise that every experimental helper has a polished artist workflow.
