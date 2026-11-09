"""Build our Windows writer using the installed open-source UE third-party SDKs."""
import os
from pathlib import Path
import subprocess
import shutil
import argparse

root = Path(__file__).resolve().parents[1]
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--engine',type=Path,default=Path(os.environ.get('UE_ENGINE_DIR','C:/Program Files/Epic Games/UE_5.8')),
                    help='UE installation root or Engine directory; defaults to UE_ENGINE_DIR or the standard UE 5.8 location')
args=parser.parse_args()
engine=args.engine.expanduser().resolve()
if (engine/'Engine').is_dir():engine=engine/'Engine'
sdk = engine / 'Source/ThirdParty'
alembic = sdk / 'Alembic/Deploy/alembic-1.8.9'
imath = sdk / 'Imath/Deploy/Imath-3.2.2'
if not alembic.is_dir() or not imath.is_dir():
    raise SystemExit('The engine directory must contain the UE 5.8 Alembic 1.8.9 and Imath 3.2.2 SDKs. Use --engine to choose your installation.')
vswhere = Path('C:/Program Files (x86)/Microsoft Visual Studio/Installer/vswhere.exe')
vs = subprocess.check_output([str(vswhere), '-latest', '-products', '*', '-requires',
                              'Microsoft.VisualStudio.Component.VC.Tools.x86.x64',
                              '-property', 'installationPath'], text=True).strip()
if not vs:
    raise RuntimeError('Install the Visual Studio C++ build tools')
vcvars = Path(vs) / 'VC/Auxiliary/Build/vcvars64.bat'
environment_text = subprocess.check_output(
    'cmd.exe /d /s /c ""' + str(vcvars) + '" >nul && set"', text=True)
environment = {key.upper(): value for key, value in os.environ.items()}
for line in environment_text.splitlines():
    key, separator, value = line.partition('=')
    if separator and key:
        environment[key.upper()] = value
build = root / 'build/groom-alembic-native'
build.mkdir(parents=True, exist_ok=True)
output = root / 'create_metahuman/native/cm_groom_alembic.exe'
source = root / 'create_metahuman/native/groom_alembic_writer.cpp'
compiler = shutil.which('cl.exe', path=environment['PATH'])
if not compiler:
    raise RuntimeError('The Visual Studio environment did not expose cl.exe')
args = [compiler, '/nologo', '/std:c++17', '/O2', '/EHsc', '/MD',
        '/I' + str(alembic / 'include'), '/I' + str(imath / 'include'),
        '/I' + str(imath / 'include/Imath'), str(source), '/Fe:' + str(output),
        '/Fo:' + str(build / 'writer.obj'), '/link',
        str(alembic / 'VS2015/x64/lib/Alembic.lib'),
        str(imath / 'VS2015/x64/lib/Imath-3_2.lib')]
subprocess.run(args, env=environment, cwd=build, check=True)
print('Built', output, output.stat().st_size, 'bytes')
