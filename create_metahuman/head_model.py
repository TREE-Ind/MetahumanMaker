"""GNM neutral-pose identity evaluator using the public Google NPZ schema.

Model weights stay in a user-selected directory; no Google model is bundled.
"""

from pathlib import Path
import json
import hashlib
import numpy as np
import bpy
from . import mesh_ops as m

_CACHE = {}


def update_skin(self, context):
    """Change only the generated source's shading, including during face mixing."""
    if not context or not self.generated_head:
        return
    try:
        apply_skin(self.generated_head, self)
    except (ValueError, ReferenceError) as exc:
        context.scene.cm_settings.status = str(exc)
    if context.screen:
        for area in context.screen.areas:
            area.tag_redraw()


def validate_skin_image(image):
    # has_data describes resident pixels; a valid newly loaded file may be lazy.
    # Accessing the pixel array requests decoding without copying the full image.
    if not len(image.pixels) or min(image.size) < 1:
        raise ValueError('The selected skin image has no readable pixels')


def uv_signature(obj):
    uv = obj.data.uv_layers.get('GNM_Skin_UV')
    if not uv:
        raise ValueError('The generated head is missing its GNM UV layout')
    values = np.empty((len(uv.data), 2), dtype=np.float32)
    uv.data.foreach_get('uv', values.ravel())
    return hashlib.sha256(values.tobytes()).hexdigest()


def _link(links, source, target):
    if not any(link.from_socket == source for link in target.links):
        links.new(source, target)


def apply_skin(obj, f):
    """Retain native GNM UV skin images through identity updates and project saves."""
    if not obj.get('cm_gnm') or not obj.data.uv_layers.get('GNM_Skin_UV'):
        raise ValueError('Generate a GNM head before choosing its skin')
    if f.generated_skin == 'LOCAL':
        from . import skin_library
        image = skin_library.local_image(f)
        if image and image.get('cm_gnm_uv_hash') != uv_signature(obj):
            raise ValueError('This local skin uses a different GNM UV layout; prepare it again')
    else:
        image = f.generated_skin_image if f.generated_skin == 'IMAGE' else None
    if image:
        validate_skin_image(image)
    material = obj.data.materials[0] if obj.data.materials else None
    new_material = not material or not material.get('cm_gnm_skin')
    if not material or not material.get('cm_gnm_skin'):
        material = bpy.data.materials.new('GNM_Skin')
        material['cm_gnm_skin'] = True
        if obj.data.materials:
            obj.data.materials[0] = material
        else:
            obj.data.materials.append(material)
    elif material.users > 1:
        material = material.copy()
        obj.data.materials[0] = material
    if not material.use_nodes:
        material.use_nodes = True
    nodes, links = material.node_tree.nodes, material.node_tree.links
    shader = next((n for n in nodes if n.type == 'BSDF_PRINCIPLED'), None)
    if shader is None:
        shader = nodes.new('ShaderNodeBsdfPrincipled')
    output = next((n for n in nodes if n.type == 'OUTPUT_MATERIAL' and n.is_active_output), None)
    if output is None:
        output = nodes.new('ShaderNodeOutputMaterial')
    _link(links, shader.outputs['BSDF'], output.inputs['Surface'])
    texture = nodes.get('CM_GNM_Skin_Image')
    if texture is None:
        texture = nodes.new('ShaderNodeTexImage')
        texture.name = 'CM_GNM_Skin_Image'
        texture.label = 'GNM Skin Albedo'
        texture.location = (-600, 0)
    if texture.image != image:
        texture.image = image
    uv = nodes.get('CM_GNM_Skin_UV')
    if uv is None:
        uv = nodes.new('ShaderNodeUVMap')
        uv.name = 'CM_GNM_Skin_UV'
        uv.location = (-800, 0)
    if uv.uv_map != 'GNM_Skin_UV':
        uv.uv_map = 'GNM_Skin_UV'
    _link(links, uv.outputs['UV'], texture.inputs['Vector'])
    tone = nodes.get('CM_GNM_Skin_Tone')
    if tone is None:
        tone = nodes.new('ShaderNodeMixRGB')
        tone.name = 'CM_GNM_Skin_Tone'
        tone.label = 'Skin Tone'
        tone.blend_type = 'MULTIPLY'
        tone.inputs[0].default_value = 1
        tone.location = (-300, 0)
    gain = 2 ** f.generated_skin_tone
    tone.inputs[2].default_value = (gain, gain, gain, 1)
    if image:
        _link(links, texture.outputs['Color'], tone.inputs[1])
    else:
        for link in list(tone.inputs[1].links):
            links.remove(link)
        tone.inputs[1].default_value = (*f.skin_color, 1)
    _link(links, tone.outputs[0], shader.inputs['Base Color'])
    if new_material:
        shader.inputs['Roughness'].default_value = .48
    material.diffuse_color = (*[min(1, value * gain) for value in f.skin_color], 1)
    return material


def draw_skin(layout, f):
    layout.prop(f, 'generated_skin', text='Skin')
    if f.generated_skin == 'IMAGE':
        row = layout.row(align=True)
        row.prop(f, 'generated_skin_image', text='')
        row.operator('cm.load_generated_skin', text='', icon='FILE_FOLDER')
        if not f.generated_skin_image:
            layout.label(text='Choose an albedo made for the GNM UV layout.', icon='INFO')
    elif f.generated_skin == 'LOCAL':
        from . import skin_library, skin_native_jobs
        layout.prop(f, 'generated_skin_preset', text='Preset')
        if not f.generated_local_skin_image:
            layout.label(text='Prepare once from your local character.', icon='INFO')
        busy = skin_library.running() or skin_native_jobs.running()
        row = layout.row(align=True)
        row.enabled = not busy
        row.operator('cm.generate_native_skin', text='Create Skin', icon='ADD')
        if f.generated_skin_preset == 'REFERENCE':
            row.operator('cm.prepare_local_skin', text='Prepare Reference', icon='FILE_REFRESH')
        if busy:
            layout.operator('cm.cancel_local_skin', text='Cancel Preparation', icon='CANCEL')
        if f.skin_library_status:
            layout.label(text=f.skin_library_status)
    else:
        layout.prop(f, 'skin_color', text='Color')
    layout.prop(f, 'generated_skin_tone', text='Tone')
    if f.generated_skin == 'IMAGE' and f.generated_skin_image:
        layout.operator('cm.keep_skin', text='Keep in Skin Library', icon='ADD')


def load_model(folder):
    path = Path(bpy.path.abspath(folder)) / "gnm_head.npz"
    if not path.is_file():
        from .setup_paths import candidates

        path = next(
            (
                Path(item["model_dir"]) / "gnm_head.npz"
                for item in candidates()
                if item.get("model_dir")
                and (Path(item["model_dir"]) / "gnm_head.npz").is_file()
            ),
            path,
        )
    if not path.is_file():
        raise ValueError(
            "Choose the model folder containing gnm_head.npz; run tools/setup_models.py first"
        )
    key = (str(path.resolve()), path.stat().st_mtime)
    if key not in _CACHE:
        with np.load(path, allow_pickle=False) as archive:
            data = {
                name: archive[name]
                for name in (
                    "template_vertex_positions",
                    "vertex_identity_basis",
                    "quads",
                    "quad_uvs",
                    "vertex_groups",
                    "vertex_group_names",
                )
            }
        if data["vertex_identity_basis"].shape[0] != 253:
            raise ValueError(
                "Expected the public GNM v3 head with 253 identity parameters"
            )
        _CACHE.clear()
        _CACHE[key] = data
    return _CACHE[key]


def coefficients(f):
    values = np.zeros(253, dtype=np.float32)
    for item in f.identity_coefficients:
        if 0 <= item.index < 253:
            values[item.index] = item.value
    return values


def sample_identity(f, seed=None):
    from .setup_paths import resolve

    resolve(f)
    rng = np.random.default_rng(f.identity_seed if seed is None else seed)
    decoder = Path(bpy.path.abspath(f.model_dir)) / "identity_decoder.npz"
    if decoder.is_file():
        with np.load(decoder, allow_pickle=False) as archive:
            z = rng.normal(size=(1, int(archive["latent_dim"]))).astype(np.float32)
            condition = np.zeros((1, 6), dtype=np.float32)
            condition[0, 0:2] = [1 - f.gender, f.gender]
            condition[0, 2 + int(f.ethnicity)] = 1
            vector = np.concatenate([z, condition], axis=1)
            for i in range(int(archive["layers"])):
                vector = vector @ archive[f"kernel_{i}"] + archive[f"bias_{i}"]
                activation = str(archive[f"activation_{i}"])
                if activation == "relu":
                    vector = np.maximum(vector, 0)
                elif activation == "tanh":
                    vector = np.tanh(vector)
                elif activation not in {"linear", "none"}:
                    raise ValueError(
                        "Unsupported identity decoder activation: " + activation
                    )
            vector = vector[0]
        if vector.shape != (253,):
            raise ValueError("Identity decoder must output 253 coefficients")
    else:
        raise ValueError(
            "Semantic decoder not prepared. Run tools/convert_gnm_decoder.py with the optional Python environment."
        )
    return np.clip(vector, -3, 3)


def update_coefficients(f, values):
    f.identity_coefficients.clear()
    for i, value in enumerate(values):
        item = f.identity_coefficients.add()
        item.index, item.value = i, float(value)


def evaluate(data, identity):
    return np.asarray(
        data["template_vertex_positions"], dtype=np.float32
    ) + np.tensordot(identity, data["vertex_identity_basis"], axes=(0, 0))


def create_head(context,s,f,randomize=False,invalidate=True):
    """Stage a whole identity before replacing a generated source's geometry."""
    if context.mode!='OBJECT' or f.body_editing or f.full_sculpt or f.restore_editing:
        raise ValueError('Finish the active edit before generating or refining a face')
    old=f.generated_head;original=old.data if old and old.get('cm_gnm') else None
    if original and (not old.is_editable or not original.is_editable):
        raise ValueError('Make the generated source local and editable before refining its identity')
    from . import character_project,resource_budget
    resource_budget.require('generate face identity',128*resource_budget.MIB+2*resource_budget.meshes([old]))
    fields={name:getattr(f,name) for name in ('generated_head','generated_local_skin_image','front_axis')}
    source=s.source
    identities=[(item.index,item.value) for item in f.identity_coefficients]
    face=[(item.key,item.label,item.category,item.value) for item in f.face_controls]
    properties={name:character_project._stored(old.get(name)) for name in
                ('cm_identity','cm_gnm_model_scale','cm_gnm_generated_geometry','cm_front_axis')} if original else {}
    before=set(bpy.data.user_map());staged=None
    with m.selection_state(context,on_error=True):
        try:
            if original:
                staged=original.copy();old.data=staged
            result=_create_head(context,s,f,randomize,invalidate)
            if original:
                if result==old and original.users<=int(original.use_fake_user):
                    # Keep the stable source Mesh ID when it has no linked users.
                    points=np.empty(len(staged.vertices)*3,dtype=np.float32)
                    staged.vertices.foreach_get('co',points)
                    materials=list(staged.materials)
                    old.data=original;original.vertices.foreach_set('co',points)
                    while len(original.materials)>len(materials):original.materials.pop(index=len(original.materials)-1)
                    for index,material in enumerate(materials):
                        if index<len(original.materials):original.materials[index]=material
                        else:original.materials.append(material)
                    for polygon in original.polygons:polygon.use_smooth=True
                    original.update()
                elif result!=old:old.data=original
            return result
        except Exception:
            if original:old.data=original
            s.source=source
            for name,value in fields.items():setattr(f,name,value)
            f.identity_coefficients.clear()
            for index,value in identities:
                item=f.identity_coefficients.add();item.index=index;item.value=value
            f.face_controls.clear()
            for key,label,category,value in face:
                item=f.face_controls.add();item.key=key;item.label=label;item.category=category;item['value']=value
            for name,value in properties.items():
                if value is None:
                    if name in old:del old[name]
                else:old[name]=value
            added=set(bpy.data.user_map())-before
            if added:bpy.data.batch_remove(ids=added)
            staged=None
            context.view_layer.update()
            raise
        finally:
            if staged and staged.users<=int(staged.use_fake_user):bpy.data.meshes.remove(staged)


def _create_head(context, s, f, randomize=False, invalidate=True):
    data = load_model(f.model_dir)
    if randomize:
        update_coefficients(f, sample_identity(f))
    elif not len(f.identity_coefficients):
        update_coefficients(f, np.zeros(253))
    raw = evaluate(data, coefficients(f))
    names = [str(n) for n in data["vertex_group_names"]]
    regions = {
        "nose": "nose_region",
        "mouth": "expression_basis_mouth_nose_ears",
        "eyes": "expression_basis_left_eye",
        "brows": "middle_brow_region",
        "cheeks": "left_cheek_region",
        "ears": "ears",
        "forehead": "forehead_region",
        "chin": "chin_region",
        "jaw": "skin_exterior",
    }
    if not len(f.face_controls):
        for name in regions:
            item = f.face_controls.add()
            item.key = name
            item.label = name.title()
    for index, item in enumerate(f.face_controls):
        if abs(item.value) < 1e-8:
            continue
        region = regions[item.key]
        if region not in names:
            continue
        mask = data["vertex_groups"][names.index(region)].copy()
        for left, right in [("left_eye", "right_eye"), ("left_cheek", "right_cheek")]:
            other = region.replace(left, right)
            if other != region and other in names:
                mask = np.maximum(mask, data["vertex_groups"][names.index(other)])
        alternate = np.zeros(253, dtype=np.float32)
        alternate[:170] = (
            np.random.default_rng(731 + index).normal(size=170).astype(np.float32)
            * 0.1
            * item.value
        )
        delta = np.tensordot(alternate, data["vertex_identity_basis"], axes=(0, 0))
        raw += delta * mask[:, None]
    if "skin" not in names:
        raise ValueError("GNM model lacks its skin group")
    keep = np.flatnonzero(data["vertex_groups"][names.index("skin")] > 0.5)
    remap = np.full(len(raw), -1, dtype=np.int32)
    remap[keep] = np.arange(len(keep))
    quads = np.asarray(data["quads"], dtype=np.int32)
    face_mask = np.all(remap[quads] >= 0, axis=1)
    faces = remap[quads[face_mask]]
    # Google model coordinates: Y up, Z forward. Blender: Z up, -Y forward.
    template = np.asarray(data["template_vertex_positions"])[keep]
    scale = f.generated_head_height / max(np.ptp(template[:, 1]), 1e-8)
    raw = raw[keep]
    vertices = np.column_stack((raw[:, 0], -raw[:, 2], raw[:, 1])) * scale
    old = f.generated_head
    if old and old.get("cm_gnm") and len(old.data.vertices) == len(vertices):
        obj = old
        obj.data.vertices.foreach_set("co", vertices.ravel())
        obj.data.update()
    else:
        mesh = bpy.data.meshes.new("GNM_Head_Surface")
        mesh.from_pydata(vertices.tolist(), [], faces.tolist())
        mesh.update()
        obj = bpy.data.objects.new("GNM_Head", mesh)
        context.collection.objects.link(obj)
        uv = mesh.uv_layers.new(name="GNM_Skin_UV")
        uv.data.foreach_set("uv", np.asarray(data["quad_uvs"])[face_mask].ravel())
        obj["cm_gnm"] = True
        from .workflow import write_group

        for target, groups in {
            "CM_Ears": ["ears"],
            "CM_Eyes": ["eye_sockets"],
            "CM_Inner_Parts": ["mouth_sock"],
            "CM_Upper_Orbital_Rim": ["left_brow_region", "right_brow_region"],
            "CM_Head": ["skin_exterior"],
        }.items():
            available = [
                data["vertex_groups"][names.index(g)][keep]
                for g in groups
                if g in names
            ]
            if available:
                write_group(obj, target, np.maximum.reduce(available))
        f.generated_head = obj
    obj["cm_identity"] = json.dumps(coefficients(f).tolist())
    obj['cm_gnm_model_scale'] = float(scale)
    from . import photo_review
    if photo_review._VIEW and photo_review._VIEW['source'] == obj:
        photo_review.camera_for(context, obj)
    obj['cm_gnm_generated_geometry'] = hashlib.sha256(np.asarray(vertices, dtype=np.float32).tobytes()).hexdigest()
    apply_skin(obj, f)
    for polygon in obj.data.polygons:
        polygon.use_smooth = True
    s.source = obj
    if invalidate and s.landmarks:
        bpy.ops.cm.clear_landmarks()
    if "cm_front_axis" not in obj:
        obj["cm_front_axis"] = "-Y"
    if not s.template:
        f.front_axis = obj["cm_front_axis"]
    m.activate(context, obj)
    s.status = "Actual GNM head generated locally from 253 identity coefficients. Source ready for wrapping."
    return obj


def identity_bank(f):
    """Twelve deterministic identities drawn through the learned semantic decoder."""
    return np.stack(
        [sample_identity(f, f.identity_seed + index * 7919) for index in range(12)]
    )


def mix_identities(context, s, f):
    seeds = identity_bank(f)
    weights = np.asarray(f.identity_mix, dtype=np.float32)
    denominator = np.abs(weights).sum()
    if denominator < 1e-8:
        weights = np.full(12, 1 / 12)
        denominator = 1
        f.identity_mix = weights
    update_coefficients(f, weights @ seeds / denominator)
    return create_head(context, s, f)
