"""Release working buffers only after an owned skin solve was discarded."""
import json
from pathlib import Path
import re
import stat

FILES={
    'albedo.png','reference.png','mask.png','normal.png','reference_normal.png',
    'normal_mask.png','accepted.png','protect.png','candidate-albedo.png',
    'candidate-normal.png','artifact-evidence.png','relief-mask.png','retained-normal.png',
    'material_roughness.png','material_cavity.png','candidate-roughness.png','candidate-cavity.png',
    'region_normal.png','region_roughness.png','region_cavity.png','valid.png','valid.npz',
    'material-height.npy','anatomy-protect.png','surface-prior-preview.png',
    'baseline-normal.png','baseline-cavity.png','baseline-roughness.png','baseline-albedo.png',
}


def _plain(path):
    info=path.lstat()
    if path.is_symlink() or getattr(info,'st_file_attributes',0)&getattr(stat,'FILE_ATTRIBUTE_REPARSE_POINT',0x400):
        raise ValueError('A linked path is not eligible for skin-buffer cleanup')
    return info


def _folder(project,folder):
    project=Path(project).resolve();folder=Path(folder).absolute()
    if folder.parent!=project/'SkinSolves' or not re.fullmatch('[a-f0-9]{12}',folder.name):
        raise ValueError('Skin cleanup needs an owned job directly inside this project')
    for path in (project,project/'SkinSolves',folder):_plain(path)
    if folder.resolve()!=folder:raise ValueError('The skin job leaves its project')
    return project,folder


def _json(path):
    if _plain(path).st_size>2*1024**2:raise ValueError('Skin cleanup metadata is too large')
    return json.loads(path.read_text(encoding='utf-8'))


def ready(project,folder,*,worker_stopped=False):
    project,folder=_folder(project,folder)
    state=_json(folder/'state.json')
    if state.get('schema')!=1 or state.get('status')!='discarded' or Path(state.get('project','')).resolve()!=project:
        raise ValueError('Only discarded jobs from this project can be cleaned')
    if worker_stopped:return
    if (folder/'staged.json').exists():
        from . import skin_solve_staged
        if skin_solve_staged.alive(folder) is not False:
            raise ValueError('The original staged worker is active or its identity is uncertain')
        if not (folder/'cancel').is_file():raise ValueError('The discarded staged job has no cancellation marker')
    else:
        # Legacy direct workers have no durable child identity. A discarded
        # state is written only after their process exits; additionally retain
        # buffers while the original authoring PID is still present.
        from . import process_ref
        request=_json(folder/'request.json')
        parent=request.get('parent')
        if not isinstance(parent,int) or parent<=0:raise ValueError('The original direct-worker parent is unknown')
        if process_ref.identify(parent):raise ValueError('The original authoring process may still own these buffers')


def buffers(project,folder,*,worker_stopped=False):
    ready(project,folder,worker_stopped=worker_stopped)
    folder=Path(folder);files=[];directories=[]
    def collect(path):
        info=_plain(path)
        if stat.S_ISREG(info.st_mode):
            files.append(dict(path=path.relative_to(folder).as_posix(),bytes=info.st_size,mtime_ns=info.st_mtime_ns))
        elif stat.S_ISDIR(info.st_mode):
            for child in sorted(path.iterdir()):collect(child)
            directories.append(path.relative_to(folder).as_posix())
        else:raise ValueError('A special file is not eligible for cleanup')
    for path in sorted(folder.iterdir()):
        name=path.name
        selected=(name in FILES or re.fullmatch(r'(?:bloodflow_|candidate-)CM[123]\.png',name)
                  or re.fullmatch(r'(?:surface-guide-|step_).+\.png',name))
        scratch=(name=='sr-prior' or name.startswith(('learned-','surface-prior-')))
        if selected:
            if not stat.S_ISREG(_plain(path).st_mode):raise ValueError('An expected skin image is not a regular file')
            collect(path)
        elif scratch:
            if not stat.S_ISDIR(_plain(path).st_mode):raise ValueError('An expected skin scratch directory changed type')
            collect(path)
    return dict(job=folder.name,files=files,directories=directories,bytes=sum(row['bytes'] for row in files))


def prune(project,folder,*,worker_stopped=False):
    """Recheck the complete plan before deleting any file; keep all receipts."""
    project,folder=_folder(project,folder)
    plan=buffers(project,folder,worker_stopped=worker_stopped)
    ready(project,folder,worker_stopped=worker_stopped)
    for row in plan['files']:
        path=folder/row['path'];info=_plain(path)
        if info.st_size!=row['bytes'] or info.st_mtime_ns!=row['mtime_ns']:
            raise ValueError('Skin buffers changed during cleanup; inspect the original job')
    removed=0
    for row in plan['files']:
        path=folder/row['path']
        if path.resolve().is_relative_to(folder) and path.resolve()!=folder:
            path.unlink();removed+=row['bytes']
        else:raise ValueError('A cleanup buffer left its owned job')
    for name in plan['directories']:(folder/name).rmdir()
    return dict(job=folder.name,removed_bytes=removed,removed_files=len(plan['files']))


def inspect(project,protected=()):
    project=Path(project).resolve();root=project/'SkinSolves'
    report=dict(schema=1,project=str(project),jobs=[],skipped=[],eligible_bytes=0,eligible_files=0)
    if not root.exists():return report
    _plain(root)
    for folder in sorted(root.iterdir()):
        if not re.fullmatch('[a-f0-9]{12}',folder.name):continue
        try:
            if any(Path(path).resolve().is_relative_to(folder.resolve()) for path in protected):
                raise ValueError('This job is referenced by the current Blender project')
            plan=buffers(project,folder)
            if plan['files']:report['jobs'].append(plan)
        except (OSError,ValueError,KeyError) as exc:
            report['skipped'].append(dict(job=folder.name,reason=str(exc)))
    report['eligible_bytes']=sum(job['bytes'] for job in report['jobs'])
    report['eligible_files']=sum(len(job['files']) for job in report['jobs'])
    return report


def clean(project,protected=()):
    report=inspect(project,protected);results=[]
    for job in report['jobs']:results.append(prune(project,Path(project)/'SkinSolves'/job['job']))
    return dict(schema=1,project=report['project'],jobs=results,
                removed_bytes=sum(row['removed_bytes'] for row in results),
                removed_files=sum(row['removed_files'] for row in results),skipped=report['skipped'])
