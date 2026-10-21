# Windows one-file release

Build on Windows x64. The CPU release includes the two default YOLO26 weights
and FFmpeg; it does not need Python installed on the target computer.

## Prepare

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install torch==2.11.0 torchvision==0.26.0 --index-url https://download.pytorch.org/whl/cpu
.\.venv\Scripts\python.exe -m pip install -e '.[ai,test]' 'pyinstaller==6.22.3'
```

Run `tools/smoke_yolo.py` to download `yolo26n-seg.pt` and `yolo26n-pose.pt` to
`.cache/models/`. Put FFmpeg at `.cache/build-assets/ffmpeg.exe`, or set
`UVC_FFMPEG` to its absolute path. Include its distribution license and source/
build information with the release notices. FFprobe is not required.

## Build and verify

```powershell
$env:PYINSTALLER_CONFIG_DIR = "$PWD\.cache\pyinstaller"
.\.venv\Scripts\python.exe -m PyInstaller --clean --noconfirm --workpath build/windows packaging/windows.spec
```

Output: `dist/UVC-Lab-0.1.0-Windows-x64.exe`.
The spec isolates DLL search paths from unrelated developer tools. Keep
`--clean` when rebuilding after a change to native dependencies.

Use `Start-Process -Wait` for the windowed executable in scripts. Its explicit
self-test writes a structured report and data to the chosen directory:

```powershell
$exe = (Resolve-Path dist/UVC-Lab-0.1.0-Windows-x64.exe).Path
Start-Process -FilePath $exe -ArgumentList '--self-test', 'G:\uvc-release-test' -Wait -WindowStyle Hidden
Get-Content G:\uvc-release-test\self-test.json
```

It checks frozen camera-helper subprocesses, all three Qt languages, simulated
acquisition, MP4/AVI, Excel, and actual YOLO26 segmentation/pose fusion. A passing
report has `ok: true` and `frozen: true`. It does not validate physical cameras or
motors.

The executable is not UPX-packed or code-signed. Publish SHA-256 checksums, source,
build recipes, model origins, dependency versions, and third-party notices with
the release. The bootloader extracts native dependencies on startup.

Data/logs live under `%LOCALAPPDATA%\UVC Lab`, not the extraction directory.
`UVC_LAB_DATA_DIR` redirects this for testing. Exact bundled model basenames
resolve to bundled resources; missing custom paths are not silently substituted.
