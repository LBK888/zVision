# Build with: python -m PyInstaller --noconfirm packaging/windows.spec
import os
import sys
from pathlib import Path

# Resolve Windows system DLLs from Windows, never an unrelated tool's PATH.
# A Poppler ICU DLL named icuuc.dll otherwise shadows Windows ICU used by Qt.
system_root = Path(os.environ.get('SystemRoot', r'C:\Windows'))
os.environ['PATH'] = os.pathsep.join(str(p) for p in [
    Path(sys.executable).parent, Path(sys.base_prefix),
    system_root / 'System32', system_root, system_root / 'System32' / 'Wbem',
])
from PyInstaller.utils.hooks import collect_data_files, collect_submodules, copy_metadata

root = Path(SPECPATH).parent
assets = Path(os.environ.get('UVC_BUILD_ASSETS', str(root / '.cache' / 'build-assets')))
ffmpeg = Path(os.environ.get('UVC_FFMPEG', str(assets / 'ffmpeg.exe')))
models = Path(os.environ.get('UVC_MODEL_DIR', str(root / '.cache' / 'models')))
required = [ffmpeg, models / 'yolo26n-seg.pt', models / 'yolo26n-pose.pt']
for path in required:
    if not path.is_file():
        raise FileNotFoundError(f'Build resource not found: {path}. See packaging/README.md')

datas = [(str(path), 'models') for path in required[1:]]
datas += [(str(root / 'profiles'), 'profiles')]
datas += [(str(root / 'README.md'), 'documentation'), (str(root / 'readme_TW.md'), 'documentation')]
datas += [(str(root / 'THIRD_PARTY.md'), 'documentation')]
datas += collect_data_files('ultralytics')
for distribution in ['ultralytics', 'torch', 'torchvision', 'PyQt6', 'opencv-python', 'numpy', 'scipy', 'openpyxl', 'pyserial', 'cv2-enumerate-cameras']:
    datas += copy_metadata(distribution)
if (assets / 'licenses').is_dir():
    datas += [(str(assets / 'licenses'), 'licenses')]

hidden = collect_submodules('ultralytics.nn')
hidden += ['ultralytics.models.yolo', 'ultralytics.models.yolo.segment', 'ultralytics.models.yolo.pose',
           'scipy.special._ufuncs_cxx', 'cv2_enumerate_cameras.windows_backend']
a = Analysis(
    [str(root / 'packaging' / 'entrypoint.py')],
    pathex=[str(root)],
    binaries=[(str(ffmpeg), 'tools')],
    datas=datas,
    hiddenimports=hidden,
    hookspath=[],
    excludes=['PySide6', 'PyQt5', 'PySide2', 'tkinter', 'tensorflow', 'jax', 'jaxlib',
              'IPython', 'notebook', 'jupyter', 'sphinx', 'pytest', 'torchaudio',
              'pandas.tests', 'numpy.tests', 'scipy.tests'],
    noarchive=False,
)
pyz = PYZ(a.pure)
exe = EXE(pyz, a.scripts, a.binaries, a.datas, [],
          name='UVC-Lab-0.1.0-Windows-x64', debug=False, bootloader_ignore_signals=False,
          strip=False, upx=False, console=False, disable_windowed_traceback=False,
          runtime_tmpdir=None)
