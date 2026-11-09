"""Keep local Unreal synthesis off Blender's UI thread, then project in Blender."""
from pathlib import Path
import json
import shutil
import threading
import uuid
import bpy
from . import bridge

_JOB = None


def running():
    return _JOB is not None


def _send(job, engine, project):
    script, request = str(job['folder'] / 'unreal_skin.py'), str(job['folder'] / 'native-request.json')
    try:
        # Separate calls let Unreal finish compiling capture materials and update
        # texture resources before they are sampled. Never use TextureExporterPNG
        # on synthesized float images: UE 5.8 asserts on their source format.
        for action in ('prepare', 'capture'):
            if action == 'capture' and job['cancelled']:
                action = 'cancel'
            command = 'import runpy; runpy.run_path(' + json.dumps(script) + ')[' + repr(action) + '](' + json.dumps(request) + ')'
            bridge.run_python(engine, command, project)
    except Exception as exc:
        job['error'] = str(exc)
    finally:
        job['done'].set()


def start(context, label, u, v, face_texture_index, body_texture_index=0, show_top_underwear=True, target='GENERATED'):
    global _JOB
    from . import skin_library, unreal_jobs, groom_jobs, guided, setup_paths
    if running() or skin_library.running() or unreal_jobs.running() or groom_jobs.running():
        raise ValueError('Wait for the current skin or Unreal operation to finish')
    s, f = context.scene.cm_settings, context.scene.cm_features
    setup_paths.resolve(f)
    if not guided.character_library():
        raise ValueError('Prepare the licensed local character library first')
    engine = bpy.path.abspath(s.unreal_engine_dir)
    if not (Path(engine) / 'Engine/Plugins/Experimental/PythonScriptPlugin/Content/Python/remote_execution.py').is_file():
        raise ValueError('Choose the Unreal install root in Export settings')
    from . import resource_budget
    resource_budget.require('local skin synthesis',resource_budget.GIB)
    root = skin_library.cache_root(f)
    root.mkdir(parents=True, exist_ok=True)
    if shutil.disk_usage(root).free < 2 * 1024**3:
        raise ValueError('Skin generation needs 2 GiB of free disk space')
    folder = root / '.jobs' / uuid.uuid4().hex[:12]
    folder.mkdir(parents=True, exist_ok=False)
    shutil.copy2(Path(__file__).with_name('unreal_skin.py'), folder / 'unreal_skin.py')
    (folder / 'native-request.json').write_text(json.dumps({'label': label.strip() or 'Native Skin',
        'u': u, 'v': v, 'face_texture_index': face_texture_index,
        'body_texture_index': body_texture_index, 'show_top_underwear': show_top_underwear}), encoding='utf-8')
    job = {'folder': folder, 'root': root, 'scene': context.scene, 'done': threading.Event(), 'error': '',
           'cancelled': False, 'target': target}
    _JOB = job
    job['thread'] = threading.Thread(target=_send, args=(job, engine, s.unreal_project), name='CreateMHSkin', daemon=True)
    f.skin_library_status = 'Synthesizing locally in Unreal; Blender remains available...'
    job['thread'].start()
    if not bpy.app.timers.is_registered(poll):
        bpy.app.timers.register(poll, first_interval=.2, persistent=True)


def poll():
    global _JOB
    job = _JOB
    if not job:
        return None
    if not job['done'].is_set():
        return .3
    _JOB = None
    scene = job['scene']
    if not scene:
        return None
    if job['cancelled']:
        scene.cm_features.skin_library_status = 'Skin generation canceled; previous skin retained.'
        scene.cm_features.last_action_error = False
        scene.cm_settings.status = scene.cm_features.skin_library_status
        return None
    try:
        from . import skin_library
        if job['root'] != skin_library.cache_root(scene.cm_features):
            raise ValueError('Model location changed; the captured skin remains in ' + str(job['folder']))
        receipt = job['folder'] / 'native-capture.json'
        if not receipt.is_file():
            raise ValueError('Native skin generation failed: ' + job['error'])
        native = json.loads(receipt.read_text(encoding='utf-8'))
        keys = [key for key in ('native_albedo', 'native_normal', 'native_cavity', 'native_roughness',
                               'body_albedo', 'body_normal', 'body_cavity', 'body_roughness') if key in native]
        for key in keys:
            path = Path(native[key]).resolve()
            if not path.is_relative_to(job['folder'].resolve()) or skin_library.digest(path) != native[key + '_sha256']:
                raise ValueError('Captured skin changed before projection')
        with bpy.context.temp_override(scene=scene):
            inputs = {k: native[k] for k in (*keys, 'label')}
            inputs['native_synthesis'] = native.get('synthesis', {})
            inputs['native_sampling'] = {role: {'uv_origin': value['uv_origin']}
                                         for role, value in native.get('surface_sampling', {}).items()}
            inputs['target'] = job['target']
            skin_library.start(bpy.context, native=inputs, folder=job['folder'])
    except Exception as exc:
        scene.cm_features.skin_library_status = str(exc)
        scene.cm_settings.status = str(exc)
        scene.cm_features.last_action_error = True
    return None


def detach():
    if _JOB:
        _JOB['scene'] = None


class CM_OT_generate_native_skin(bpy.types.Operator):
    bl_idname = 'cm.generate_native_skin'
    bl_label = 'Create Local Skin'
    bl_description = 'Create matching head/body skin from the installed local MetaHuman model; keep Unreal open'

    label: bpy.props.StringProperty(name='Name', default='My Native Skin')
    u: bpy.props.FloatProperty(name='Tone U', min=0, max=1, default=.5)
    v: bpy.props.FloatProperty(name='Tone V', min=0, max=1, default=.5)
    face_texture_index: bpy.props.EnumProperty(name='Surface Detail', items=[('0', 'Variation 1', ''), ('1', 'Variation 2', '')])
    body_texture_index: bpy.props.IntProperty(name='Body Surface', min=0, max=8, default=0,
        description='Native MetaHuman body detail variant, from 0 to 8')
    show_top_underwear: bpy.props.BoolProperty(name='Top Underwear', default=True)
    target: bpy.props.EnumProperty(items=[('GENERATED', 'Generated Head', ''), ('CHARACTER', 'Full Character', '')],
                                   default='GENERATED', options={'HIDDEN'})

    def invoke(self, context, event):
        return context.window_manager.invoke_props_dialog(self, width=380)

    def draw(self, context):
        self.layout.prop(self, 'label')
        self.layout.prop(context.scene.cm_settings, 'unreal_project', text='UE Project')
        engine = Path(bpy.path.abspath(context.scene.cm_settings.unreal_engine_dir))
        if not (engine / 'Engine').is_dir():
            self.layout.prop(context.scene.cm_settings, 'unreal_engine_dir', text='Unreal Install')
        self.layout.prop(self, 'u', slider=True)
        self.layout.prop(self, 'v', slider=True)
        self.layout.prop(self, 'face_texture_index')
        self.layout.prop(self, 'body_texture_index')
        self.layout.prop(self, 'show_top_underwear')
        self.layout.label(text='Uses the installed UE local skin model.', icon='INFO')
        self.layout.label(text='Keep the chosen Unreal project open.')

    def execute(self, context):
        try:
            start(context, self.label, self.u, self.v, int(self.face_texture_index),
                  self.body_texture_index, self.show_top_underwear, self.target)
            return {'FINISHED'}
        except Exception as exc:
            self.report({'ERROR'}, str(exc))
            return {'CANCELLED'}


CLASSES = (CM_OT_generate_native_skin,)
