"""Collect installed component license files and provenance for a binary release."""
from pathlib import Path
import importlib.metadata as metadata
import json
import os
import shutil
import subprocess
from zipfile import ZipFile, ZIP_DEFLATED

ROOT = Path(__file__).resolve().parents[1]
dest = ROOT/".cache"/"build-assets"/"licenses"
dest.mkdir(parents=True, exist_ok=True)
names = ["ultralytics", "torch", "torchvision", "PyQt6", "PyQt6-Qt6", "opencv-python", "numpy", "scipy",
         "openpyxl", "pyserial", "cv2-enumerate-cameras", "pillow", "matplotlib", "pyinstaller"]
provenance = []
for name in names:
    distribution = metadata.distribution(name)
    component = dest/name
    component.mkdir(exist_ok=True)
    copied = []
    for relative in distribution.files or []:
        parts = [p.lower() for p in relative.parts]
        filename = relative.name.lower()
        license_like = filename.startswith(("license", "copying", "notice")) or "licenses" in parts
        if not license_like:
            continue
        path = Path(distribution.locate_file(relative))
        if path.is_file():
            safe_relative = Path(*[p for p in relative.parts if p not in ["..", "."]])
            target = component/safe_relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, target)
            copied.append(str(safe_relative).replace("\\", "/"))
    provenance.append({"name": name, "version": distribution.version, "license": distribution.metadata.get("License"),
                       "project_urls": distribution.metadata.get_all("Project-URL"), "files": copied})
ffmpeg = Path(os.environ.get("UVC_FFMPEG", shutil.which("ffmpeg") or "C:/ffmpeg/bin/ffmpeg.exe"))
version = subprocess.run([str(ffmpeg), "-version"], capture_output=True, check=True).stdout.decode("utf-8", errors="replace")
(dest/"FFmpeg-version-and-build.txt").write_text(version, encoding="utf-8")
license_path = ffmpeg.parent.parent/"LICENSE.txt"
if license_path.is_file():
    shutil.copy2(license_path, dest/"FFmpeg-LICENSE.txt")
(dest/"component-versions.json").write_text(json.dumps(provenance, ensure_ascii=False, indent=2), encoding="utf-8")
shutil.copy2(ROOT/"THIRD_PARTY.md", dest/"THIRD_PARTY.md")
output = ROOT/"dist"/"UVC-Lab-0.1.0-third-party-notices.zip"
output.parent.mkdir(exist_ok=True)
with ZipFile(output, "w", ZIP_DEFLATED) as bundle:
    for path in dest.rglob("*"):
        if path.is_file():
            bundle.write(path, path.relative_to(dest).as_posix())
print(output)
