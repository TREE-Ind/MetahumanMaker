"""Private conventional texture inputs for MetaHuman virtual-texture baking.

UV-only material renders cannot reliably warm streaming VT pages. Evaluate the
same MetaHuman material permutation with full source pixels in transient textures.
Artist assets, static switches and texture streaming settings remain untouched.
"""
import hashlib
import runpy
import struct
import uuid
from pathlib import Path
from tempfile import TemporaryDirectory


class BakeInputs:
    def __init__(self, directory):
        self.budget=runpy.run_path(str(Path(__file__).with_name('resource_budget.py')))
        self.budget['require_disk']('private virtual texture bake inputs',directory,512*1024**2)
        self.scratch = TemporaryDirectory(prefix='.web-bake-inputs-', dir=directory)
        self.textures = {}
        self.keepalive = []

    def close(self):
        self.keepalive.clear()
        self.textures.clear()
        self.scratch.cleanup()

    def conventional_texture(self, asset_path, world):
        import unreal
        if asset_path in self.textures:
            return self.textures[asset_path]
        source = unreal.load_asset(asset_path) or unreal.find_object(None, asset_path)
        if not isinstance(source, unreal.Texture2D):
            raise ValueError('Virtual material input must be a Texture2D: ' + asset_path)
        if not source.get_editor_property('virtual_texture_streaming'):
            self.textures[asset_path] = source
            return source
        path = Path(self.scratch.name) / (hashlib.sha256(asset_path.encode()).hexdigest() + '.png')
        task = unreal.AssetExportTask()
        task.object = source
        task.filename = str(path)
        task.exporter = unreal.TextureExporterPNG()
        task.automated = True
        task.prompt = False
        task.replace_identical = False
        if not unreal.Exporter.run_asset_export_task(task):
            raise RuntimeError('Cannot read full virtual texture source: ' + asset_path)
        with path.open('rb') as stream:header=stream.read(24)
        if header[:8]!=b'\x89PNG\r\n\x1a\n':raise ValueError('Private virtual texture source is not PNG')
        width,height=struct.unpack('>II',header[16:24])
        if min(width,height)<1 or max(width,height)>8192:raise ValueError('Virtual texture source exceeds the 8K bake input limit')
        self.budget['require']('private resident texture conversion',width*height*16+256*1024**2)
        texture = unreal.RenderingLibrary.import_file_as_texture2d(world, str(path))
        if not texture or texture.get_editor_property('virtual_texture_streaming'):
            raise RuntimeError('Cannot create private resident texture: ' + asset_path)
        texture.set_editor_property('srgb', source.get_editor_property('srgb'))
        texture.set_editor_property('compression_settings', source.get_editor_property('compression_settings'))
        self.textures[asset_path] = texture
        self.keepalive.append(texture)
        return texture

    def material(self, live, record, world, *, separate_micro=False):
        import unreal
        switches = record.get('switches', {})
        virtual = switches.get('Use Virtual Texture') is True
        micro = separate_micro and switches.get('Use Micro Skin Details') is True
        if not virtual and not micro:
            return live, {'virtual_inputs_resolved': False, 'micro_details_separate': False}
        parent = live
        while isinstance(parent, unreal.MaterialInstanceDynamic):
            parent = parent.get_editor_property('parent')
        lib = unreal.MaterialEditingLibrary
        constant = unreal.new_object(unreal.MaterialInstanceConstant, name='CM_WebResident_' + uuid.uuid4().hex)
        if not constant:
            raise RuntimeError('Cannot create private native bake permutation')
        self.keepalive.append(constant)
        lib.set_material_instance_parent(constant, parent)
        for key, value in switches.items():
            lib.set_material_instance_static_switch_parameter_value(constant, key, value)
        if virtual:
            lib.set_material_instance_static_switch_parameter_value(constant, 'Use Virtual Texture', False)
        if micro:
            lib.set_material_instance_static_switch_parameter_value(constant, 'Use Micro Skin Details', False)
        lib.update_material_instance(constant)
        material = unreal.MaterialLibrary.create_dynamic_material_instance(world, constant)
        self.keepalive.append(material)
        # Read the live uniforms, including a caller's neutral expression state.
        for key in record.get('scalars', {}):
            value = live.get_scalar_parameter_value(key) if isinstance(live, unreal.MaterialInstanceDynamic) else record['scalars'][key]
            material.set_scalar_parameter_value(key, value)
        for key in record.get('vectors', {}):
            value = live.get_vector_parameter_value(key) if isinstance(live, unreal.MaterialInstanceDynamic) else unreal.LinearColor(*record['vectors'][key])
            material.set_vector_parameter_value(key, value)
        for key, value in record.get('textures', {}).items():
            if value:
                texture = unreal.load_asset(value['asset']) or unreal.find_object(None, value['asset'])
                material.set_texture_parameter_value(key, texture)
        resolved = []
        if virtual:
            for key, value in record.get('textures', {}).items():
                if not key.endswith(' VT') or not value:
                    continue
                ordinary = key[:-3]
                if ordinary not in record['textures']:
                    raise ValueError('No conventional material input for virtual parameter: ' + key)
                material.set_texture_parameter_value(ordinary, self.conventional_texture(value['asset'], world))
                resolved.append(key)
            if not resolved:
                raise ValueError('Virtual material has no supported conventional parameter pairs')
        return material, {'virtual_inputs_resolved': virtual, 'resolved_virtual_parameters': resolved,
                          'micro_details_separate': micro}
