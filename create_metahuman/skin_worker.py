"""Reproject the user's local licensed reference skin onto canonical GNM UVs.

Run in an isolated Blender process. No licensed images are distributed.
"""
from pathlib import Path
import hashlib
import json
import sys
import os
import bpy
import numpy as np

assert bpy.app.background, 'Skin preparation must use an isolated background process'

root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root))
import create_metahuman
from create_metahuman import automation, head_model, mesh_ops as m, worker_lifetime

request = json.loads(Path(sys.argv[sys.argv.index('--') + 1]).read_text(encoding='utf-8'))
if request.get('parent_pid') and request['parent_pid'] != os.getpid():
    worker_lifetime.watch_parent(request['parent_pid'])

def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()

inputs = {str(Path(request['model_dir']) / 'gnm_head.npz'): '', str(Path(request['library'])): ''}
for name in ('native_albedo', 'native_normal', 'native_cavity', 'native_roughness', 'body_albedo', 'body_normal', 'body_cavity', 'body_roughness'):
    if request.get(name):
        inputs[str(Path(request[name]).resolve())] = ''
inputs = {path: digest(path) for path in inputs}

create_metahuman.register()
bpy.context.preferences.filepaths.save_version = 0
bpy.context.scene.cycles.samples = 4
for obj in list(bpy.data.objects):
    bpy.data.objects.remove(obj, do_unlink=True)
s, f = bpy.context.scene.cm_settings, bpy.context.scene.cm_features
folder = Path(request['folder']).resolve()
folder.mkdir(parents=True, exist_ok=True)

def progress(message):
    print(message, flush=True)
    try:
        pending = folder / 'progress.tmp'
        pending.write_text(message, encoding='utf-8')
        pending.replace(folder / 'progress.txt')
    except OSError:
        pass

progress('Aligning the local reference skin...')
s.project_dir = str(folder)
s.character_name = 'GNM_Local_Skin'
resolution = int(request.get('resolution', 2048))
s.map_size = str(resolution)
s.bake_distance = .006
s.iterations = 30
s.max_distance = .5
f.wrap_quality = 'ACCURATE'
f.front_axis = '-Y'
f.source_front_axis = '-Y'
f.full_head = False
f.bake_device = 'CPU'
f.model_dir = request['model_dir']
f.inference_python = request['inference_python']
template = head_model.create_head(bpy.context, s, f)
s.template = template
canonical = m.coordinates(template).copy()
library = Path(request['library'])
assert library
with bpy.data.libraries.load(str(library), link=False) as (available, loaded):
    loaded.objects = available.objects
source = next(o for o in loaded.objects if o and o.get('cm_library_role') == 'head')
from create_metahuman import skin_projection
body_uv = None
if request.get('body_albedo'):
    body = next(o for o in loaded.objects if o and o.get('cm_library_role') == 'body')
    body_uv = skin_projection.uv_layout_signature(body)
for obj in loaded.objects:
    if obj and obj != source and not obj.users_collection:
        bpy.data.objects.remove(obj, do_unlink=True)
bpy.context.collection.objects.link(source)
source.parent = None
# Native synthesized textures share the licensed reference's MetaHuman UVs.
# Replace images on this private worker copy, never on the artist's library.
for slot in source.material_slots:
    if not slot.material or not slot.material.use_nodes:
        continue
    for node in slot.material.node_tree.nodes:
        key = {'ALBEDO': 'native_albedo', 'NORMAL': 'native_normal'}.get(node.get('cm_map_kind'))
        if key and request.get(key):
            node.image = bpy.data.images.load(request[key], check_existing=False)
            if key == 'native_normal':
                node.image.colorspace_settings.name = 'Non-Color'
            # Opaque Unreal render targets export RGB with a zero alpha plane.
            # These are opaque skin maps, including when used outside Blender.
            node.image.alpha_mode = 'NONE'
s.source = source
bpy.context.view_layer.update()
m.align_source(bpy.context, s, bounds=True)
automation.auto_landmarks(bpy.context, s, f)
m.align_source(bpy.context, s)
source_points, template_points = m.landmark_points(s)
progress('Building the GNM skin projection surface...')
wrapped = m.wrap(bpy.context, s)
_, distances = m.nearest(m.source_bvh(bpy.context, source), m.evaluated_coordinates(bpy.context, wrapped))
print('FIT_DIAGNOSTIC', json.dumps({'distance_quantiles': np.quantile(distances, [0,.5,.9,.99,1]).tolist(),
    'source_bounds': [m.coordinates(source).min(0).tolist(), m.coordinates(source).max(0).tolist()],
    'target_bounds': [m.evaluated_coordinates(bpy.context, wrapped).min(0).tolist(), m.evaluated_coordinates(bpy.context, wrapped).max(0).tolist()],
    'keys': {k.name:k.value for k in wrapped.data.shape_keys.key_blocks}}), flush=True)
if request.get('project_surface', True):
    # Densify only the private projection receiver. SIMPLE retains linear GNM
    # UV interpolation; the actual generated head and its UVs stay unchanged.
    evaluated = wrapped.evaluated_get(bpy.context.evaluated_depsgraph_get())
    mesh = bpy.data.meshes.new_from_object(evaluated, preserve_all_data_layers=True, depsgraph=bpy.context.evaluated_depsgraph_get())
    projection = bpy.data.objects.new('GNM_Skin_Projection', mesh)
    bpy.context.collection.objects.link(projection)
    projection.matrix_world = wrapped.matrix_world.copy()
    wrapped.hide_set(True)
    wrapped.hide_render = True
    wrapped = projection
    s.result = projection
    m.activate(bpy.context, projection)
    modifier = projection.modifiers.new('Projection sampling', 'SUBSURF')
    modifier.subdivision_type = 'SIMPLE'
    modifier.levels = modifier.render_levels = 1
    bpy.ops.object.modifier_apply(modifier=modifier.name)
    projected, _ = m.nearest(m.source_bvh(bpy.context, source), m.evaluated_coordinates(bpy.context, wrapped))
    wrapped.data.vertices.foreach_set('co', m.to_local(wrapped, projected).ravel())
    wrapped.data.update()
    bpy.context.view_layer.update()
assert np.array_equal(canonical, m.coordinates(template))
assert request.get('project_surface', True) or m.topology(wrapped) == m.topology(template)
progress('Baking the local skin onto GNM UVs...')
path = m.bake(bpy.context, s, 'ALBEDO')
image = bpy.data.images.load(str(path), check_existing=False)
pixels = np.asarray(image.pixels[:], dtype=np.float32).reshape(-1, 4)
nonempty = pixels[:, :3].max(axis=1) > .002
# Detect actual missed rays independently of legitimately black source pixels.
progress('Checking skin coverage...')
original_material = source.data.materials[0]
assert len(source.data.materials) == 1
white = bpy.data.materials.new('GNM_Projection_Coverage')
white.use_nodes = True
white.node_tree.nodes.get('Principled BSDF').inputs['Base Color'].default_value = (1, 1, 1, 1)
source.data.materials[0] = white
original_name = s.character_name
try:
    s.character_name += '_ProjectionCoverage'
    coverage_path = m.bake(bpy.context, s, 'CAVITY')
finally:
    source.data.materials[0] = original_material
    s.character_name = original_name
coverage_image = bpy.data.images.load(str(coverage_path), check_existing=False)
hits = np.asarray(coverage_image.pixels[:], dtype=np.float32).reshape(-1, 4)[:, :3].max(axis=1) > .5
mask_image = bpy.data.images.new('GNM_UV_Coverage', width=resolution, height=resolution, float_buffer=True)
mask_material = bpy.data.materials.new('GNM_UV_Coverage')
mask_material.use_nodes = True
nodes = mask_material.node_tree.nodes
emitter = nodes.new('ShaderNodeEmission')
emitter.inputs['Color'].default_value = (1, 1, 1, 1)
mask_material.node_tree.links.new(emitter.outputs[0], nodes.get('Material Output').inputs['Surface'])
target = nodes.new('ShaderNodeTexImage')
target.image = mask_image
nodes.active = target
receiver = m.duplicate(bpy.context, template, 'GNM_UV_Coverage')
receiver.data.materials.clear()
receiver.data.materials.append(mask_material)
m.activate(bpy.context, receiver)
scene = bpy.context.scene
scene.render.engine = 'CYCLES'
scene.cycles.samples = 1
scene.render.bake.use_selected_to_active = False
scene.render.bake.margin = 0
bpy.ops.object.bake(type='EMIT')
mask = np.asarray(mask_image.pixels[:], dtype=np.float32).reshape(-1, 4)[:, 0] > .5
missing = mask & ~hits
from create_metahuman import skin_projection
source_image = next(n.image for n in original_material.node_tree.nodes if n.type == 'TEX_IMAGE' and n.image and n.get('cm_map_kind') == 'ALBEDO')
progress('Completing small skin gaps from the source surface...')
fallback = skin_projection.repair(bpy.context, source, wrapped, source_image, image, missing)
image.save()
from create_metahuman.skin_library import save_image_copy
native_path = folder / 'Maps/Native_Reference_ALBEDO.png'
save_image_copy(source_image, native_path, opaque=True)
normal_source = next((n.image for n in original_material.node_tree.nodes
    if n.type == 'TEX_IMAGE' and n.image and n.get('cm_map_kind') == 'NORMAL'), None)
native_normal_path = None
if normal_source:
    native_normal_path = folder / 'Maps/Native_Reference_NORMAL.png'
    save_image_copy(normal_source, native_normal_path, opaque=True)
diagnostic = np.zeros((len(mask), 4), dtype=np.float32)
diagnostic[mask, :3] = 1
diagnostic[missing, :3] = (1, 0, 0)
diagnostic[:, 3] = 1
mask_image.pixels.foreach_set(diagnostic.ravel())
mask_image.filepath_raw = str(folder / 'coverage-diagnostic.png')
mask_image.file_format = 'PNG'
mask_image.save()
bpy.data.objects.remove(receiver, do_unlink=True)
report = {'source_library': str(library), 'landmarks': len(s.landmarks),
    'resolution': resolution,
    'source_landmarks': source_points.tolist(), 'template_landmarks': template_points.tolist(),
    'vertices': len(wrapped.data.vertices), 'canonical_unchanged': True,
    'topology': m.topology(template), 'map': str(path),
    'sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
    'covered_fraction': float(nonempty.mean()), 'uv_coverage_fraction': float(mask.mean()),
    'dark_skin_texels': int((mask & ~nonempty).sum()),
    'missing_skin_texels': int(missing.sum()), 'uv_texels': int(mask.sum()),
    'missing_skin_fraction': float(missing.sum() / max(1, mask.sum())), 'wrap_rms': float(s.last_error)}
report['surface_transfer_fallback'] = fallback
report['uv_sha256'] = head_model.uv_signature(template)
report['native_reference'] = {'map': str(native_path), 'sha256': digest(native_path),
    'uv_layout': skin_projection.uv_layout_signature(source)}
if native_normal_path:
    report['native_normal'] = {'map': str(native_normal_path), 'sha256': digest(native_normal_path)}
report['extra_maps'] = {}
for role in ('native_cavity', 'native_roughness', 'body_albedo', 'body_normal', 'body_cavity', 'body_roughness'):
    if request.get(role):
        extra = bpy.data.images.load(request[role], check_existing=False)
        if not role.endswith('albedo'):
            extra.colorspace_settings.name = 'Non-Color'
        path = folder / 'Maps' / (role + '.png')
        save_image_copy(extra, path, opaque=True)
        report['extra_maps'][role] = {'map': str(path), 'sha256': digest(path)}
        if role == 'body_albedo':
            report['extra_maps'][role]['uv_layout'] = body_uv
if any(digest(path) != value for path, value in inputs.items()):
    raise ValueError('The reference library or GNM model changed during preparation; previous skin retained')
report['inputs'] = inputs
if request.get('native_synthesis'):
    report['native_synthesis'] = request['native_synthesis']
if request.get('native_sampling'):
    report['native_sampling'] = request['native_sampling']
report['schema_version'] = 1
(folder / 'projection.json').write_text(json.dumps(report, indent=2))
f.generated_skin_image = image
f.generated_skin = 'IMAGE'
image.pack()
source.hide_set(True)
wrapped.hide_set(True)
template.hide_set(False)
m.activate(bpy.context, template)
if request.get('save_review'):
    bpy.ops.wm.save_as_mainfile(filepath=str(folder / 'review.blend'), compress=True)
progress('Local GNM skin ready.')
print('GNM_SKIN_PROJECTION_REVIEW_READY', json.dumps({k:v for k,v in report.items() if 'landmarks' not in k}), flush=True)
