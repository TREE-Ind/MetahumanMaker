"""Self-contained static GLB delivery, staged separately from Unreal bundles."""

import json
import re
import struct
from pathlib import Path
from tempfile import TemporaryDirectory

import bpy
import numpy as np
from mathutils import Matrix

from . import (
    character_project,
    export_delivery,
    map_provenance,
    mesh_ops,
    web_materials,
    resource_budget,
    roughness_response,
    web_preview,
)

GENERATOR = 'CreateMetaHuman.StaticGLB'


def parts(scene):
    s, f = scene.cm_settings, scene.cm_features
    if s.part == 'FULL_BODY':
        result = {'Head': f.head_result or s.result, 'Body': f.body_result}
    else:
        result = {'Body' if s.part == 'BODY' else 'Head': s.result}
    if any(obj is None for obj in result.values()):
        raise ValueError('Complete the accepted character geometry before exporting a GLB')
    if s.part != 'BODY' and (f.full_head or s.part == 'FULL_BODY'):
        for role, field in (('EyeLeft', 'eye_left'), ('EyeRight', 'eye_right'), ('Teeth', 'teeth')):
            obj = getattr(f, field)
            if obj is None and f.full_head:
                raise ValueError('Load the missing full-head part before GLB export: ' + role)
            if obj is not None:
                result[role] = obj
    return result


def preflight(context):
    s, f = context.scene.cm_settings, context.scene.cm_features
    character_project.preflight(context, s, f)
    selected = parts(context.scene)
    for obj in selected.values():
        mesh_ops.check_mesh(obj, 'Accepted character part')
        if obj.get('cm_individual_sculpt_part'):
            raise ValueError('Finish the individual sculpt before exporting the character')
        keys = obj.data.shape_keys
        expression = keys.key_blocks.get('Expression Check') if keys else None
        if expression and expression.value:
            raise ValueError('Return Expression Check to Neutral before exporting a static GLB')
    for obj in (s.detail, f.body_detail if s.part == 'FULL_BODY' else None):
        if obj and (not obj.get('cm_scan_applied', True) or obj.get('cm_head_update_pending')):
            raise ValueError('Apply or update the retained sculpt before exporting the character')
    scopes = tuple('FINAL_' + key.upper() for key in selected if key in {'Head', 'Body'})
    problems = map_provenance.problems(context.scene, keys=scopes, suffixes=('_NORMAL.png',))
    if problems:
        raise ValueError(problems[0])
    return selected


def document(path):
    with path.open('rb') as stream:
        magic, version, size = struct.unpack('<4sII', stream.read(12))
        if magic != b'glTF' or version != 2 or size != path.stat().st_size:
            raise ValueError('The GLB exporter did not produce a valid version-2 container')
        length, kind = struct.unpack('<I4s', stream.read(8))
        if kind != b'JSON' or length > size - 20:
            raise ValueError('The GLB is missing its asset description')
        return json.loads(stream.read(length))


def image_sizes(path, asset, limit):
    result = []
    with path.open('rb') as stream:
        stream.seek(12)
        json_size = struct.unpack('<I', stream.read(4))[0]
        stream.seek(20 + json_size)
        length, kind = struct.unpack('<I4s', stream.read(8))
        if kind != b'BIN\0':
            raise ValueError('GLB images have no embedded binary payload')
        binary = stream.tell()
        for image in asset.get('images', []):
            view = asset['bufferViews'][image['bufferView']]
            offset, size = view.get('byteOffset', 0), view['byteLength']
            if view.get('buffer', 0) != 0 or offset < 0 or size < 24 or offset + size > length:
                raise ValueError('GLB texture lies outside its embedded buffer')
            stream.seek(binary + offset)
            header = stream.read(24)
            if image.get('mimeType') != 'image/png' or header[:8] != b'\x89PNG\r\n\x1a\n' or header[12:16] != b'IHDR':
                raise ValueError('GLB export must embed the converted PNG image buffers')
            width, height = struct.unpack('>II', header[16:24])
            if min(width, height) < 1 or max(width, height) > limit:
                raise ValueError('An embedded GLB image exceeded the selected texture limit')
            result.append({'name': image.get('name', ''), 'width': width, 'height': height})
    return result


def export(context):
    s = context.scene.cm_settings
    selected = preflight(context)
    limit=int(s.web_texture_size)
    images={node.image for obj in selected.values() for mat in obj.data.materials if mat and mat.use_nodes
            for node in mat.node_tree.nodes if node.type=='TEX_IMAGE' and node.image}
    source_bytes=sum(int(image.size[0])*int(image.size[1])*16 for image in images)
    target_bytes=sum(min(limit,int(image.size[0]))*min(limit,int(image.size[1]))*16 for image in images)
    largest=max((int(image.size[0])*int(image.size[1])*16 for image in images),default=0)
    resource_budget.require('static GLB export',512*resource_budget.MIB+source_bytes+2*largest+4*target_bytes+
                            resource_budget.meshes(selected.values()))
    resource_budget.require_disk('static GLB export',bpy.path.abspath(s.project_dir),256*resource_budget.MIB+2*target_bytes)
    profiles={}
    if s.web_skin_finish:
        for role,value in roughness_response.profiles(context.scene).items():
            if value['mode']!='SKIN_DETAIL':continue
            prefix=s.character_name+('_Body' if role=='BODY' and s.part=='FULL_BODY' else '')
            profiles[role.title()]={**value,'path':str((Path(bpy.path.abspath(s.project_dir))/'Maps'/(prefix+'_ROUGHNESS.png')).resolve())}
    name = re.sub(r'[^A-Za-z0-9_]', '_', s.character_name).strip('_')
    if not name:
        raise ValueError('Enter a character name')
    parent = Path(bpy.path.abspath(s.project_dir)).resolve() / 'Export' / 'Web'
    destination = parent / name
    if destination.resolve() != destination.absolute():
        raise ValueError('GLB output must remain inside the project without linked folders')
    destination.mkdir(parents=True, exist_ok=True)
    previous = destination / 'manifest.json'
    names = [name + '.glb', 'load-character.js', 'README.txt', 'preview.html', 'manifest.json']
    if previous.exists():
        if json.loads(previous.read_text(encoding='utf-8')).get('generator') != GENERATOR:
            raise ValueError('This Web export folder contains an unrelated manifest; choose another character name')
    elif any((destination / item).exists() for item in names):
        raise ValueError('The Web export would replace unowned files; choose another character name')
    factory = web_materials.Materials(int(s.web_texture_size))
    temporary = []
    report = []
    unit = context.scene.unit_settings.scale_length
    if not np.isfinite(unit) or unit <= 0:
        raise ValueError('Choose a positive scene unit scale before GLB export')
    try:
        with TemporaryDirectory(prefix='.glb-', dir=parent) as directory:
            stage = Path(directory)
            with mesh_ops.selection_state(context):
                try:
                    for obj in context.selected_objects:
                        obj.select_set(False)
                    depsgraph = context.evaluated_depsgraph_get()
                    for role, original in selected.items():
                        evaluated = original.evaluated_get(depsgraph)
                        data = bpy.data.meshes.new_from_object(evaluated, preserve_all_data_layers=True, depsgraph=depsgraph)
                        obj = bpy.data.objects.new('CM_Web_' + role, data)
                        temporary.append((obj, data))
                        context.scene.collection.objects.link(obj)
                        obj.matrix_world = Matrix.Scale(unit, 4) @ evaluated.matrix_world
                        values = np.empty(len(data.vertices) * 3, dtype=np.float32)
                        data.vertices.foreach_get('co', values)
                        if not len(data.polygons) or not np.isfinite(values).all():
                            raise ValueError('Accepted character has invalid export geometry: ' + original.name)
                        originals = list(data.materials)
                        data.materials.clear()
                        for material in originals:
                            data.materials.append(factory.material(material,role=role,profile=profiles.get(role)))
                        if data.materials and any(m.node_tree and any(n.type == 'TEX_IMAGE' for n in m.node_tree.nodes)
                                                  for m in data.materials) and not data.uv_layers:
                            raise ValueError('A textured character part has no UV layout: ' + original.name)
                        obj.select_set(True)
                        # Selection export still filters through a local viewport's membership.
                        for area in context.screen.areas if context.screen else ():
                            if area.type == 'VIEW_3D' and area.spaces.active.local_view:
                                obj.local_view_set(area.spaces.active, True)
                        report.append({'role': role, 'node': obj.name, 'source_vertices': len(data.vertices),
                                       'source_polygons': len(data.polygons)})
                    context.view_layer.objects.active = temporary[0][0]
                    result = bpy.ops.export_scene.gltf(filepath=str(stage / names[0]), export_format='GLB',
                        use_selection=True, use_active_scene=True, use_visible=False, export_animations=False, export_skins=False,
                        export_morph=False, export_cameras=False, export_lights=False, export_extras=False,
                        export_yup=True, export_texcoords=True, export_normals=True, export_tangents=True,
                        export_vertex_color='NONE', export_image_format='AUTO', export_keep_originals=False,
                        will_save_settings=False)
                    if result != {'FINISHED'}:
                        raise RuntimeError('GLB export did not finish; the preceding delivery was retained')
                finally:
                    for obj, data in temporary:
                        bpy.data.objects.remove(obj, do_unlink=True)
                        bpy.data.meshes.remove(data)
            asset = document(stage / names[0])
            if asset.get('animations') or asset.get('skins'):
                raise ValueError('Static GLB unexpectedly contains animation or skinning data')
            if any('uri' in item for item in asset.get('buffers', [])) or any('bufferView' not in item for item in asset.get('images', [])):
                raise ValueError('GLB export left external texture or buffer dependencies')
            actual = {node.get('name') for node in asset.get('nodes', []) if 'mesh' in node}
            if actual != {part['node'] for part in report}:
                raise ValueError('The GLB contains unexpected meshes or is missing an accepted character part')
            sizes = image_sizes(stage / names[0], asset, factory.limit)
            (stage / names[1]).write_text(
                "import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js';\n\n"
                "export async function loadCharacter(scene, url = new URL('./" + names[0] + "', import.meta.url)) {\n"
                "  const gltf = await new GLTFLoader().loadAsync(String(url));\n"
                "  scene.add(gltf.scene);\n  return gltf.scene;\n}\n", encoding='utf-8')
            (stage / names[2]).write_text(
                'Static character GLB for Three.js / glTF 2.0 viewers.\n'
                'The GLB embeds its images and uses meters with Y up. Serve it over HTTP.\n'
                'load-character.js shows integration into an existing Three.js scene.\n'
                'For an interactive preview, run python -m http.server 8000 --bind 127.0.0.1 in this folder, then open http://127.0.0.1:8000/preview.html.\n'
                'Lighting studio includes six portrait presets, eight local CC0 environments, key/fill/rim softboxes, exposure, rotation, temperature and saved lighting files.\n'
                'The preview fetches pinned Three.js modules from jsDelivr and needs an internet connection. No character file is uploaded.\n'
                'Geometry is the accepted head/body/accessories with current shape mixes evaluated.\n'
                'Separate dense sculpt meshes are represented through their current baked skin maps.\n'
                'PBR shading approximates Blender skin; subsurface scattering and studio lighting differ.\n'
                'Use Skin Finish normalizes accepted derived roughness; native regional masks and the Creator slider curve are not reproduced.\n'
                'Native eye shader colors are flattened to a portable UV image; normals use OpenGL orientation.\n'
                'This static file contains no MetaHuman DNA, skeletal rig, facial animation, animated wrinkle/blood-flow drivers, or groom strands.\n'
                'Use separate native Unreal assembly and groom workflows for those features.\n', encoding='utf-8')
            manifest = {'schema_version': 1, 'generator': GENERATOR, 'format': 'GLB', 'mesh': names[0],
                        'support_files': names[1:4], 'name': name, 'geometry': 'STATIC', 'units': 'meters',
                        'up_axis': 'Y', 'parts': report, 'texture_limit': factory.limit,
                        'embedded_images': len(asset.get('images', [])), 'image_sizes': sizes,
                        'materials': len(asset.get('materials', []))}
            manifest['skin_finish']={'enabled':s.web_skin_finish,'materials':factory.calibrated}
            web_preview.write(stage/names[3],names[0])
            manifest['support_files'].extend(['lighting-studio.js',*[p.relative_to(stage).as_posix() for p in sorted((stage/'lighting').iterdir()) if p.is_file()]])
            owned=export_delivery.files(json.loads(previous.read_text(encoding='utf-8'))) if previous.is_file() else set()
            for relative in export_delivery.files(manifest)-owned:
                if (destination/relative).exists():raise ValueError('The Web export would replace an unowned file: '+relative.as_posix())
            (stage / 'manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
            export_delivery.publish(stage, destination)
    finally:
        factory.close()
    return destination
