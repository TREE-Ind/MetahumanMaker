from pathlib import Path
from contextlib import contextmanager
import bpy
import numpy as np
from . import anatomy, mesh_ops as m


@contextmanager
def skin_part(s, f, role=None):
    """Scope existing texture tools to the chosen part, restoring scene pointers."""
    saved = s.result, s.detail, s.character_name
    body = s.part == "FULL_BODY" and (role or f.texture_part) == "BODY"
    try:
        if body:
            m.check_mesh(f.body_result, "Body result")
            s.result, s.detail = f.body_result, f.body_detail
            s.character_name += "_Body"
        yield
    finally:
        if body:
            f.body_detail = s.detail
        s.result, s.detail, s.character_name = (
            saved[0],
            saved[1] if body else s.detail,
            saved[2],
        )


def save_material_maps(s, obj, role):
    """Save library images to this character, without sharing editable image buffers."""
    folder = Path(bpy.path.abspath(s.project_dir)) / "Maps"
    folder.mkdir(parents=True, exist_ok=True)
    prefix = s.character_name + ("_Body" if role == "BODY" else "")
    for material in obj.data.materials:
        if not material or not material.use_nodes:
            continue
        # Duplicating the material keeps painting from modifying the template.
        copy = material.copy()
        obj.data.materials[list(obj.data.materials).index(material)] = copy
        for node in copy.node_tree.nodes:
            kind = node.get("cm_map_kind")
            if node.type == "TEX_IMAGE" and node.image and kind:
                image = copy_image_pixels(node.image)
                image.filepath_raw = str(folder / (prefix + "_" + kind + ".png"))
                image.file_format = "PNG"
                image.save()
                node.image = image
                if kind == "NORMAL":
                    reference = (
                        folder.parent / "ReferenceMaps" / (prefix + "_NORMAL.png")
                    )
                    reference.parent.mkdir(exist_ok=True)
                    import shutil

                    shutil.copy2(image.filepath_raw, reference)
                    obj["cm_reference_normal"] = str(reference)
                    packed = bpy.data.images.load(str(reference), check_existing=False)
                    packed.colorspace_settings.name = "Non-Color"
                    packed.pack()
                    obj["cm_reference_normal_image"] = packed


def reference_normal(s):
    packed = s.result.get("cm_reference_normal_image")
    if isinstance(packed, bpy.types.Image):
        return packed
    path = s.result.get("cm_reference_normal")
    if not path or not Path(path).is_file():
        current = (
            Path(bpy.path.abspath(s.project_dir))
            / "Maps"
            / (s.character_name + "_NORMAL.png")
        )
        if not current.is_file():
            return None
        path = current.parent.parent / "ReferenceMaps" / current.name
        path.parent.mkdir(exist_ok=True)
        import shutil
        from . import skin_bake_layers
        neutral=skin_bake_layers.reference_normal(s.result)
        shutil.copy2(neutral or current, path)
        s.result["cm_reference_normal"] = str(path)
    packed = bpy.data.images.load(str(path), check_existing=False)
    packed.colorspace_settings.name = "Non-Color"
    packed.pack()
    s.result["cm_reference_normal_image"] = packed
    return packed


def adopt_project_maps(s, obj, role, original_name, source_folder, *, _images=None, _copy_files=True):
    """Bring a reused part's maps into the current project along with its mesh."""
    import shutil
    images = {} if _images is None else _images

    destination = Path(bpy.path.abspath(s.project_dir)) / "Maps"
    destination.mkdir(parents=True, exist_ok=True)
    old = original_name + ("_Body" if role == "BODY" else "")
    new = s.character_name + ("_Body" if role == "BODY" else "")
    for path in source_folder.iterdir() if _copy_files and source_folder.is_dir() else ():
        if not path.is_file() or path.suffix.lower() != '.png' or not path.name.startswith(old + '_'):
            continue
        if role == "HEAD" and path.name.startswith(original_name + "_Body_"):
            continue
        output = destination / (new + path.name[len(old) :])
        if path.resolve() != output.resolve():
            shutil.copy2(path, output)
    # Packed material images are still usable if the old project folder moved.
    for index, original in enumerate(list(obj.data.materials)):
        if not original or not original.use_nodes:
            continue
        material = original.copy()
        obj.data.materials[index] = material
        for node in material.node_tree.nodes:
            if node.type != "TEX_IMAGE" or not node.image:
                continue
            filename = Path(node.image.filepath).name
            kind = node.get("cm_map_kind")
            if filename.startswith(old + "_"):
                name = new + filename[len(old) :]
            elif kind:
                name = new + "_" + kind + ".png"
            else:
                continue
            if Path(name).name != name or '/' in name or '\\' in name:
                raise ValueError('Reused material has an invalid map filename')
            output = destination / name
            key = (str(output), node.image.colorspace_settings.name, node.image.alpha_mode)
            image = images.get(key)
            if image is None:
                image = (
                    bpy.data.images.load(str(output), check_existing=False)
                    if output.is_file() and not node.image.is_dirty
                    else copy_image_pixels(node.image)
                )
                image.colorspace_settings.name = node.image.colorspace_settings.name
                image.alpha_mode = node.image.alpha_mode
                image.filepath_raw = str(output)
                image.file_format = "PNG"
                image.save()
                image.pack()
                images[key] = image
            node.image = image
    source_project=Path(source_folder).resolve().parent
    if source_project!=Path(bpy.path.abspath(s.project_dir)).resolve():
        obj['cm_skin_source_project']=str(source_project)
        for key in ('cm_skin_bake_layers','cm_keep_skin_bake_layers','cm_skin_bake_expected_normal'):
            if key in obj:del obj[key]


def final_parts(s,f):
    """Resolve immutable final-map part selection without changing scene roles."""
    if s.part!='FULL_BODY':return (s.part,)
    scope=getattr(f,'final_map_scope','BOTH')
    if scope not in {'HEAD','BODY','BOTH'}:raise ValueError('Choose Head, Body or both for final maps')
    return ('HEAD','BODY') if scope=='BOTH' else (scope,)


def final_maps(context, s, f):
    from . import skin_bake_layers
    from . import scan_projection
    roles=final_parts(s,f)
    scan_projection.require_applied(s.detail if 'HEAD' in roles or s.part!='FULL_BODY' else None,
        f.body_detail if s.part=='FULL_BODY' and 'BODY' in roles else None)
    from . import detail_layers
    detail_layers.sync_character(context, s, f,roles=roles)
    skin_bake_layers.preflight(context.scene,roles=roles)
    context.scene['cm_bake_skin_layer_results']='[]'
    from . import map_provenance
    track=not context.scene.get('cm_bake_worker')
    if track:map_provenance.prepare(context.scene,'FINAL_MAPS')
    for role in roles:
        with skin_part(s, f, role):
            folder = Path(bpy.path.abspath(s.project_dir)) / "Maps"
            normal_path = folder / (s.character_name + "_NORMAL.png")
            from . import normal_composition
            normal_composition.validate(s.result,s.detail)
            geometry_only=normal_composition.geometry_only(s.result)
            generated=not geometry_only and role=='HEAD' and generated_skin_source(s,f)
            if generated:
                transfer_skin_normal(context,s,f)
            if f.add_pores and s.detail:
                add_pores(context, s, f)
            if s.detail:
                base = None if geometry_only else reference_normal(s)
                if base or geometry_only:
                    with geometry_normals_only(s.detail):
                        m.bake(context, s, "NORMAL")
                else:
                    m.bake(context, s, "NORMAL")
                if base:
                    from . import surface_maps

                    sculpt = bpy.data.images.load(
                        str(normal_path), check_existing=False
                    )
                    sculpt.colorspace_settings.name = "Non-Color"
                    reference = copy_image_pixels(base,colorspace='Non-Color')
                    try:
                        composite_normal_images(reference,sculpt,directory=folder)
                        sculpt.filepath_raw = str(normal_path)
                        sculpt.file_format = "PNG"
                        sculpt.save()
                    finally:
                        bpy.data.images.remove(sculpt)
                        bpy.data.images.remove(reference)
                bake_cavity(context, s, f,record=False)
                skin_bake_layers.replay(context,s)
            elif s.source and s.result.get("cm_body_source") != "LIBRARY":
                if not generated:m.bake(context, s, "NORMAL")
                bake_cavity(context, s, f,record=False)
                skin_bake_layers.replay(context,s)
            elif not normal_path.is_file():
                raise ValueError(
                    "Create a Sculpt Layer before baking final maps for " + role.lower()
                )
            if (folder / (s.character_name + "_ALBEDO.png")).is_file():
                if not any(s.result.data.materials):
                    m.preview_material(context,s)
                else:
                    from . import bake_jobs
                    files=[{'name':s.character_name+'_'+kind+'.png'} for kind in ('NORMAL','CAVITY')
                        if (folder/(s.character_name+'_'+kind+'.png')).is_file()]
                    bake_jobs.refresh_maps(context.scene,folder,files)
            if s.detail:
                s.detail.hide_set(True)
    from .guided import focus

    focus(
        context,
        [
            s.result,
            f.body_result if s.part == "FULL_BODY" else None,
            f.eye_left,
            f.eye_right,
            f.teeth,
        ],
        f.front_axis,
    )
    s.status = "Final maps ready using each part's normal composition setting."
    if track:
        map_provenance.record(context.scene,map_provenance.capture(context.scene,'FINAL_MAPS'))
        skin_bake_layers.accept_bake(context.scene,__import__('json').loads(context.scene.get('cm_bake_skin_layer_results','[]')))


def preview_skin(context, s, f):
    with skin_part(s, f):
        return m.preview_material(context, s)


@contextmanager
def geometry_normals_only(obj):
    """Bake geometry separately so base skin normals are not counted twice."""
    materials = list(obj.data.materials)
    temporary = []
    try:
        for index, original in enumerate(materials):
            if not original:
                continue
            material = original.copy()
            temporary.append(material)
            if material.use_nodes:
                for node in material.node_tree.nodes:
                    normal = node.inputs.get("Normal")
                    if normal:
                        for link in list(normal.links):
                            material.node_tree.links.remove(link)
            obj.data.materials[index] = material
        yield
    finally:
        obj.data.materials.clear()
        for material in materials:
            obj.data.materials.append(material)
        for material in temporary:
            bpy.data.materials.remove(material)


def generated_native_reference(s, f,*,color_independent=False):
    """Shared identity/UV proof, independent of color edits."""
    if s.source != f.generated_head or (not color_independent and f.generated_skin != 'LOCAL') or not s.template or not s.template.get('cm_template_library'):
        return None
    image = f.generated_local_skin_image
    if not image:
        return None
    from . import head_model,skin_projection
    native = image.get('cm_native_skin_reference')
    if not isinstance(native, bpy.types.Image) or native.get('cm_native_uv_layout') != skin_projection.uv_layout_signature(s.result):
        return None
    if image.get('cm_gnm_uv_hash') != head_model.uv_signature(s.source):
        return None
    return native


def generated_skin_source(s,f):
    return bool(s.source and s.source==f.generated_head and s.source.get('cm_gnm'))


def generated_reference_normal(context,s,f):
    """Reuse native pores after color edits, but honor changed normal sources."""
    import hashlib
    native=generated_native_reference(s,f,color_independent=True)
    normal=native.get('cm_native_normal_reference') if native else None
    if not isinstance(normal,bpy.types.Image) or min(normal.size)<1:return None
    if s.source.modifiers:return None
    evaluated=s.source.evaluated_get(context.evaluated_depsgraph_get())
    mesh=evaluated.to_mesh()
    try:
        points=np.empty(len(mesh.vertices)*3,dtype=np.float32)
        mesh.vertices.foreach_get('co',points)
        if hashlib.sha256(points.tobytes()).hexdigest()!=s.source.get('cm_gnm_generated_geometry'):return None
    finally:evaluated.to_mesh_clear()
    if len(s.source.data.materials)!=1:return None
    material=s.source.data.materials[0]
    if not material or not material.use_nodes or not material.get('cm_gnm_skin'):return None
    nodes=material.node_tree.nodes
    shader=next((n for n in nodes if n.type=='BSDF_PRINCIPLED'),None)
    output=next((n for n in nodes if n.type=='OUTPUT_MATERIAL' and n.is_active_output),None)
    if not shader or not output or len(output.inputs['Surface'].links)!=1:return None
    if output.inputs['Surface'].links[0].from_socket!=shader.outputs['BSDF']:return None
    if output.inputs['Displacement'].is_linked:return None
    for name in ('Normal','Coat Normal','Tangent'):
        socket=shader.inputs.get(name)
        if socket and (socket.is_linked or np.any(np.asarray(socket.default_value)!=0)):return None
    return normal


def generated_reference_albedo(s, f):
    """Retain native albedo only when its color path is still unchanged."""
    native=generated_native_reference(s,f)
    image=f.generated_local_skin_image
    if native is None or image.is_dirty:return None
    from . import skin_library
    if image.get('cm_gnm_pixels_hash') != skin_library.pixel_digest(image):return None
    if len(s.source.data.materials) != 1:
        return None
    material = s.source.data.materials[0]
    if not material or not material.use_nodes or not material.get('cm_gnm_skin'):
        return None
    nodes = material.node_tree.nodes
    texture, tone, uv = (nodes.get(name) for name in ('CM_GNM_Skin_Image', 'CM_GNM_Skin_Tone', 'CM_GNM_Skin_UV'))
    shader = next((n for n in nodes if n.type == 'BSDF_PRINCIPLED'), None)
    if not all((texture, tone, uv, shader)) or texture.image != image or tone.blend_type != 'MULTIPLY':
        return None
    if shader.inputs['Normal'].is_linked:
        return None  # Artist normal detail requires the actual source projection.
    def linked(socket, output):
        return len(socket.links) == 1 and socket.links[0].from_socket == output
    output = next((n for n in nodes if n.type == 'OUTPUT_MATERIAL' and n.is_active_output), None)
    if not output or not linked(output.inputs['Surface'], shader.outputs['BSDF']):
        return None
    if not (linked(shader.inputs['Base Color'], tone.outputs[0]) and linked(tone.inputs[1], texture.outputs['Color']) and
            linked(texture.inputs['Vector'], uv.outputs['UV']) and uv.uv_map == 'GNM_Skin_UV' and
            not tone.inputs[0].is_linked and tone.inputs[0].default_value == 1 and not tone.inputs[2].is_linked and
            np.allclose(tone.inputs[2].default_value[:3], [2 ** f.generated_skin_tone] * 3)):
        return None
    return native


def copy_image_pixels(image,*,colorspace=None):
    """Blender image copies must include unsaved painted pixels."""
    if image.source not in {'FILE','GENERATED'}:
        raise ValueError('Use a still, single-image texture before copying painted pixels: '+image.name)
    width,height=map(int,image.size)
    count=len(image.pixels)
    if min(width,height)<1 or count!=width*height*4:
        raise ValueError('Image has no readable RGBA pixels: '+image.name)
    from . import resource_budget
    resource_budget.require('copy painted texture',32*resource_budget.MIB+count*8)
    values=np.empty(count,dtype=np.float32);image.pixels.foreach_get(values)
    # A copied FILE ID can have no buffer, especially when its path changes.
    # Allocate our own buffer before copying pixels; never reload the artist's ID.
    output=bpy.data.images.new(image.name+'_Copy',width=width,height=height,alpha=True,float_buffer=image.is_float)
    try:
        output.colorspace_settings.name=colorspace or image.colorspace_settings.name
        output.alpha_mode=image.alpha_mode;output.file_format=image.file_format;output.filepath_raw=image.filepath_raw
        output.use_fake_user=image.use_fake_user
        for key in image.keys():output[key]=image[key]
        output.pixels.foreach_set(values);output.update()
        return output
    except Exception:
        bpy.data.images.remove(output)
        raise


def copy_reference_albedo(image, s, gain, kind='ALBEDO'):
    from . import skin_projection
    output = copy_image_pixels(image)
    try:
        size = int(s.map_size)
        if list(output.size) != [size, size]:
            output.scale(size, size)
        if gain != 1:
            pixels = np.asarray(output.pixels[:], dtype=np.float32).reshape(-1, 4)
            colors = skin_projection.linear(pixels[:, :3]) if output.colorspace_settings.name == 'sRGB' else pixels[:, :3]
            colors = np.clip(colors * gain, 0, 1)
            pixels[:, :3] = skin_projection.encoded(colors) if output.colorspace_settings.name == 'sRGB' else colors
            output.pixels.foreach_set(pixels.ravel())
        folder = Path(bpy.path.abspath(s.project_dir)) / 'Maps'
        folder.mkdir(parents=True, exist_ok=True)
        output.filepath_raw = str(folder / (s.character_name + '_' + kind + '.png'))
        output.file_format = 'PNG'
        output.save()
    finally:
        if not output.users:
            bpy.data.images.remove(output)


def bake_skin(context, s, f):
    from . import skin_bake
    plan=skin_bake.plan(s,f)
    original_size=s.map_size
    try:
      s.map_size=f.skin_map_size
      for item in plan:
        role=item['role']
        with skin_part(s, f, role):
            native = generated_reference_albedo(s, f) if role == 'HEAD' and s.template else None
            if item.get('albedo',True):
                if native:
                    copy_reference_albedo(native, s, 2 ** f.generated_skin_tone)
                    s.result['cm_albedo_source'] = 'LOCAL_NATIVE_UV'
                else:
                    m.bake(context, s, 'ALBEDO')
                    s.result['cm_albedo_source'] = 'SOURCE_PROJECTION'
                if 'cm_skin_source_albedo' in s.result:del s.result['cm_skin_source_albedo']
                f.skin_solver_source=None
            if item.get('roughness'):m.bake(context,s,'ROUGHNESS')
            recipe=transfer_skin_normal(context,s,f) if item['normal'] else None
            if not context.scene.get('cm_bake_worker'):m.preview_material(context, s)
        if recipe and not context.scene.get('cm_bake_worker'):
            from . import map_provenance
            key='FINAL_'+role
            map_provenance.record(context.scene,{key:map_provenance.signature(context.scene,key)},
                outputs={item['prefix']+'_NORMAL.png'},recipe='SOURCE_GNM_NATIVE' if recipe=='GNM_NATIVE_COPY' else 'SOURCE_SKIN_NORMAL')
    finally:s.map_size=original_size


def transfer_skin_normal(context,s,f):
    """Refresh only the immutable skin baseline, independently of albedo/detail."""
    normal=generated_reference_normal(context,s,f) if s.template else None
    if normal:
        copy_reference_albedo(normal,s,1,kind='NORMAL')
        s.result['cm_reference_normal_image']=normal
        s.result['cm_normal_source']='LOCAL_NATIVE_UV'
        return 'GNM_NATIVE_COPY'
    detail=s.detail
    try:
        s.detail=None
        path=m.bake(context,s,'NORMAL')
    finally:s.detail=detail
    capture_reference_normal(s,path)
    s.result['cm_normal_source']='SOURCE_PROJECTION'
    return 'SKIN_NORMAL'


def refresh_wrinkle_skin(context,s,f):
    """Refresh a packed source baseline without replacing any delivered map."""
    import tempfile
    project=s.project_dir
    root=Path(bpy.path.abspath(project)).resolve()
    jobs=root/'ReferenceMaps'/'.jobs';jobs.mkdir(parents=True,exist_ok=True)
    try:
        with tempfile.TemporaryDirectory(prefix='skin-',dir=jobs) as temporary:
            s.project_dir=temporary
            transfer_skin_normal(context,s,f)
            image=s.result.get('cm_reference_normal_image')
            if isinstance(image,bpy.types.Image) and image.get('cm_owned_skin_baseline'):
                logical=root/'ReferenceMaps'/(s.character_name+'_SOURCE_NORMAL.png')
                image.filepath_raw=str(logical)
                s.result['cm_reference_normal']=str(logical)
    finally:s.project_dir=project


def capture_reference_normal(s,path):
    import shutil
    destination=Path(bpy.path.abspath(s.project_dir))/'ReferenceMaps'/Path(path).name
    destination.parent.mkdir(parents=True,exist_ok=True)
    shutil.copy2(path,destination)
    image=bpy.data.images.load(str(destination),check_existing=False)
    image.colorspace_settings.name='Non-Color';image.pack()
    image['cm_owned_skin_baseline']=True
    old=s.result.get('cm_reference_normal_image')
    s.result['cm_reference_normal_image']=image;s.result['cm_reference_normal']=str(destination)
    if isinstance(old,bpy.types.Image) and old.get('cm_owned_skin_baseline') and not old.users:
        bpy.data.images.remove(old)
    return image


def bake_cavity(context, s, f,*,record=True):
    obj = s.detail or s.result
    m.check_mesh(obj, "Cavity source")
    world = m.evaluated_coordinates(context, obj)
    evaluated = obj.evaluated_get(context.evaluated_depsgraph_get())
    mesh = evaluated.to_mesh()
    try:
        if m.topology_data(mesh)!=m.topology(obj):raise ValueError('Apply topology-changing modifiers before baking cavity from this sculpt')
        mesh.calc_loop_triangles();triangles=np.empty((len(mesh.loop_triangles),3),dtype=np.int64);mesh.loop_triangles.foreach_get('vertices',triangles.ravel())
        edges=np.empty((len(mesh.edges),2),dtype=np.int64);mesh.edges.foreach_get('vertices',edges.ravel())
        if obj.matrix_world.to_3x3().determinant()<0:triangles=triangles[:,::-1]
        normal_matrix = obj.matrix_world.to_3x3().inverted().transposed()
        normals = np.array([normal_matrix @ v.normal for v in mesh.vertices])
        normals /= np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1e-12)
    finally:
        evaluated.to_mesh_clear()
    radius = np.linalg.norm(np.ptp(world, axis=0)) * f.cavity_radius
    values = anatomy.cavity_values(world, normals, edges, radius,triangles=triangles)
    # Curvature is encoded in an emission material, then projected through the same bake path.
    clone = m.duplicate(context, obj, "CM_CavitySource")
    clone.data.vertices.foreach_set("co", m.to_local(clone, world).ravel())
    color = clone.data.color_attributes.new(
        name="CM_Cavity", type="FLOAT_COLOR", domain="POINT"
    )
    rgba = np.column_stack((values, values, values, np.ones(len(values))))
    color.data.foreach_set("color", rgba.ravel())
    material = bpy.data.materials.new("CM_CurvatureBake")
    material.use_nodes = True
    nodes = material.node_tree.nodes
    vertex_color = nodes.new("ShaderNodeVertexColor")
    vertex_color.layer_name = color.name
    material.node_tree.links.new(
        vertex_color.outputs["Color"], nodes.get("Principled BSDF").inputs["Base Color"]
    )
    clone.data.materials.clear()
    clone.data.materials.append(material)
    source = s.source
    try:
        s.source = clone
        # A separate output tag prevents overwriting the user's albedo.
        result=m.bake(context, s, "CAVITY")
    finally:
        s.source = source
        data=clone.data
        bpy.data.objects.remove(clone, do_unlink=True)
        if not data.users:bpy.data.meshes.remove(data)
        bpy.data.materials.remove(material)
    if record and not context.scene.get('cm_bake_worker'):
        from . import map_provenance
        map_provenance.record_single(context.scene,result,'CAVITY')
    return result


def correction(pixels, exposure=0.0, gamma=1.0, saturation=1.0):
    rgb = np.maximum(pixels[..., :3] * (2**exposure), 0) ** (1 / max(gamma, 0.01))
    luminance = rgb @ np.array([0.2126, 0.7152, 0.0722])
    result = pixels.copy()
    result[..., :3] = np.clip(
        luminance[..., None] + (rgb - luminance[..., None]) * saturation, 0, 1
    )
    return result


def resample_image(image, width, height):
    # Accessing pixels loads file images lazily; has_data may be false for a valid new load.
    raw = np.empty(len(image.pixels), dtype=np.float32)
    image.pixels.foreach_get(raw)
    if not len(raw) or not image.size[0] or not image.size[1]:
        raise ValueError("Reference image has no readable pixel data")
    pixels = raw.reshape(image.size[1], image.size[0], 4)
    if tuple(image.size) == (width, height):
        return pixels
    rows = np.minimum(
        (np.arange(height) * image.size[1] / height).astype(int), image.size[1] - 1
    )
    columns = np.minimum(
        (np.arange(width) * image.size[0] / width).astype(int), image.size[0] - 1
    )
    return pixels[rows[:, None], columns[None, :]]


def composite_normal_images(reference,sculpt,*,directory=None):
    """Nearest-resample/RNM with one disk plane and a single reference read."""
    from . import pixel_buffers,surface_maps,resource_budget
    width,height=sculpt.size
    rw,rh=reference.size
    if min(width,height,rw,rh)<1:
        raise ValueError('Normal composition needs readable reference and sculpt images')
    # RNA pixel slices can copy the complete native image, so reading rows that
    # way becomes quadratic at 8K. Hold the one bulk reference read in RAM; the
    # worker's full image-memory budget remains separate from its disk reserve.
    resource_budget.require('normal reference pixels',rw*rh*16+64*resource_budget.MIB)
    base=np.empty((rh,rw,4),dtype=np.float32)
    reference.pixels.foreach_get(base.ravel())
    with pixel_buffers.image_pixels(sculpt,directory=directory) as detail:
        columns=np.minimum((np.arange(width)*rw/width).astype(int),rw-1)
        for first in range(0,height,32):
            last=min(first+32,height);rows=np.minimum((np.arange(first,last)*rh/height).astype(int),rh-1)
            # RNM copies each input block before writing it, so detail can own
            # both the incoming sculpt and its result without another 8K plane.
            surface_maps.reorient_normals(base[rows[:,None],columns[None,:]],detail[first:last],out=detail[first:last])
        sculpt.pixels.foreach_set(detail.ravel())


def wrinkle_layer(context, s, f):
    m.check_mesh(s.detail or s.result, "Wrinkle base")
    from . import wrinkles, expressions
    existing = getattr(f, f.wrinkle_slot.lower())
    if existing:
        m.activate(context, existing)
        return existing
    if f.wrinkle_native_bases:
        data = expressions.bank(f)
        if f.wrinkle_slot not in data.get('wrinkle_poses', {}):
            raise ValueError('Prepare Native Expressions with Bridge 0.5 to create native wrinkle bases')
        pose = data['poses'][data['wrinkle_poses'][f.wrinkle_slot]]
        obj = wrinkles.posed_layer(context, s, f, data, pose, f.wrinkle_slot)
    else:
        obj = wrinkles.snapshot(context, s.detail or s.result, s.character_name + '_' + f.wrinkle_slot)
        obj.shape_key_add(name="Basis")
        obj.shape_key_add(name="Wrinkle Sculpt").value = 1
        obj.active_shape_key_index = 1
        wrinkles._vector_attribute(obj,'.cm_wm_neutral',np.asarray([v.co[:] for v in obj.data.vertices]))
        wrinkles.seal_source(context,s,obj)
    obj["cm_role"] = "WRINKLE"
    setattr(f, f.wrinkle_slot.lower(), obj)
    m.activate(context, obj)
    from .guided import focus
    focus(context,[obj,*obj.children])
    s.status = f"{f.wrinkle_slot} base ready. Sculpt folds in its protected area, then bake this slot."
    return obj


def bake_wrinkle(context, s, f, slot=None,*,refresh_skin=True):
    from . import wrinkles
    wrinkles.refresh_source_state(context,s,f)
    slot = slot or f.wrinkle_slot
    obj = getattr(f, slot.lower())
    m.check_mesh(obj, slot + " wrinkle layer")
    if obj.get('cm_wrinkle_update_pending'):
        raise ValueError('Update the retained wrinkle sculpt before baking this slot')
    from . import map_provenance
    track=not context.scene.get('cm_bake_worker')
    if track:map_provenance.prepare(context.scene,'WRINKLES',[slot])
    detail, name, map_size = s.detail, s.character_name, s.map_size
    neutral = None
    try:
        s.map_size = '1024' if f.wrinkle_map_size == 'DEFAULT' else f.wrinkle_map_size
        if refresh_skin and obj.get('cm_wrinkle_native') and generated_skin_source(s,f):
            refresh_wrinkle_skin(context,s,f)
        if obj.get('cm_wrinkle_native'):
            from . import wrinkles
            neutral = wrinkles.neutralized_copy(context, obj)
        reference = reference_normal(s) if neutral else None
        s.detail = neutral or obj
        s.character_name = name + "_" + slot
        with geometry_normals_only(s.detail):
            path = m.bake(context, s, "NORMAL")
        if neutral:
            if reference:
                image = bpy.data.images.load(str(path), check_existing=False)
                try:
                    image.colorspace_settings.name = 'Non-Color'
                    composite_normal_images(reference,image,directory=path.parent)
                    image.save()
                finally:
                    bpy.data.images.remove(image)
        destination = path.with_name(name + "_" + slot + ".png")
        path.replace(destination)
        if f.bloodflow:
            s.character_name=name
            bake_bloodflow(context,s,f,slot)
        result=destination
    finally:
        s.detail, s.character_name, s.map_size = detail, name, map_size
        if neutral:
            mesh = neutral.data
            bpy.data.objects.remove(neutral, do_unlink=True)
            bpy.data.meshes.remove(mesh)
    if track:map_provenance.record(context.scene,map_provenance.capture(context.scene,'WRINKLES',[slot]))
    return result


def require_saved_bloodflow_albedo(s):
    base=Path(bpy.path.abspath(s.project_dir))/'Maps'/(s.character_name+'_ALBEDO.png')
    if not base.is_file():raise ValueError('Save the head albedo before baking expression color')
    if any(image.is_dirty and image.filepath and Path(bpy.path.abspath(image.filepath)).resolve()==base.resolve() for image in bpy.data.images):raise ValueError('Save the edited head albedo before baking expression color')
    return base


def bake_bloodflow(context,s,f,slot=None):
    """Generate only expression color from saved albedo, independent of WM geometry."""
    from . import wrinkles,surface_maps
    slot=slot or f.wrinkle_slot
    if slot not in {'WM1','WM2','WM3'} or s.part=='BODY':raise ValueError('Choose a head expression-color slot')
    base=require_saved_bloodflow_albedo(s)
    mask=getattr(f,slot.lower()+'_mask') or wrinkles.ensure_mask(s,f,slot)
    image=bpy.data.images.load(str(base),check_existing=False);cm=None
    try:
        image.colorspace_settings.name='sRGB';image.alpha_mode='NONE'
        width,height=image.size;pixels=resample_image(image,width,height)
        influence=resample_image(mask,width,height)[...,:3].mean(2)
        surface_maps.tint_bloodflow_in_place(pixels,influence,f.bloodflow_strength)
        cm=bpy.data.images.new(s.character_name+'_CM'+slot[-1],width=width,height=height,alpha=False)
        cm.colorspace_settings.name='sRGB';np.clip(pixels,0,1,out=pixels);cm.pixels.foreach_set(pixels.ravel())
        output=base.with_name(s.character_name+'_CM'+slot[-1]+'.png');cm.filepath_raw=str(output);cm.file_format='PNG';cm.save()
        return output
    finally:
        if cm and not cm.users:bpy.data.images.remove(cm)
        if not image.users:bpy.data.images.remove(image)


def save_brush_alphas(context, s, f):
    from . import skin_brushes
    folder = Path(bpy.path.abspath(s.project_dir)) / "Brushes"
    folder.mkdir(parents=True, exist_ok=True)
    for identifier, _label, _description in skin_brushes.PRESETS:
        image = skin_brushes.make_image(identifier, folder / (identifier + '.png'))
        texture = bpy.data.textures.new('CM_' + identifier, type='IMAGE')
        texture.image = image
    s.status = '14 procedural skin alphas exported to Brushes. Install Skin Brushes in Detail for sculpting assets.'


def add_pores(context, s, f):
    from .workflow import key_from_world, group_weights

    obj = s.detail
    m.check_mesh(obj, "Detail mesh for pores")
    previous = (
        obj.data.shape_keys.key_blocks.get("Procedural Pores")
        if obj.data.shape_keys
        else None
    )
    previous_value = previous.value if previous else 0
    evaluated=None
    try:
        if previous:previous.value=0
        context.view_layer.update()
        evaluated=obj.evaluated_get(context.evaluated_depsgraph_get())
        mesh=evaluated.to_mesh()
        if len(mesh.vertices)!=len(obj.data.vertices):
            raise ValueError('Remove topology-changing modifiers before generating pores')
        local=np.empty((len(mesh.vertices),3),dtype=np.float64)
        normals=np.empty_like(local)
        mesh.vertices.foreach_get('co',local.ravel())
        mesh.vertices.foreach_get('normal',normals.ravel())
        matrix=np.asarray(evaluated.matrix_world)
        vertices=local@matrix[:3,:3].T+matrix[:3,3]
        # Row-vector normals use inverse(linear), equivalent to the normal
        # matrix's inverse transpose in Blender's column-vector convention.
        normals=normals@np.linalg.inv(matrix[:3,:3])
        normals/=np.maximum(np.linalg.norm(normals,axis=1)[:,None],1e-12)
        if not np.isfinite(vertices).all() or not np.isfinite(normals).all():
            raise ValueError('Pore generation requires a finite sculpt surface')
    finally:
        if evaluated is not None:evaluated.to_mesh_clear()
        if previous:previous.value=previous_value
        context.view_layer.update()
    p = vertices / max(f.pore_spacing, 1e-8)
    cell = np.floor(p)
    phases = np.column_stack(
        [
            np.mod(np.sin(cell @ np.asarray(coeff)) * 43758.5453, 1)
            for coeff in [
                (127.1, 311.7, 74.7),
                (269.5, 183.3, 246.1),
                (113.5, 271.9, 124.6),
            ]
        ]
    )
    amplitude = (
        -np.exp(-np.sum((p - cell - phases) ** 2, axis=1) / 0.025) * f.pore_strength
    )
    amplitude *= group_weights(obj, "CM_DetailMask", default=1)
    key_from_world(obj, "Procedural Pores", vertices + normals * amplitude[:, None])
