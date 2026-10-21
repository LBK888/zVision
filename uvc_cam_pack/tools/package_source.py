"""Create a clean, verifiable source distribution without models or run data."""
from pathlib import Path
import hashlib
from zipfile import ZipFile, ZIP_DEFLATED

root = Path(__file__).resolve().parents[1]
destination = root/"dist"
destination.mkdir(exist_ok=True)
archive = destination/"uvc-lab-0.1.0-source.zip"
files = [root/name for name in ["README.md", "readme_TW.md", "THIRD_PARTY.md", "pyproject.toml", "requirements.txt", "start.bat", "install.ps1", ".gitignore"]]
for directory in ["uvc_lab", "tests", "tools", "docs", "examples", "profiles", "packaging"]:
    files.extend(p for p in (root/directory).rglob("*") if p.is_file() and "__pycache__" not in p.parts and p.suffix != ".pyc")
with ZipFile(archive, "w", ZIP_DEFLATED) as bundle:
    for path in sorted(set(files)):
        if path.suffix == ".py":
            compile(path.read_text(encoding="utf-8-sig"), str(path), "exec")
        bundle.write(path, "uvc-lab/"+path.relative_to(root).as_posix())
with ZipFile(archive) as bundle:
    assert bundle.testzip() is None
    assert not any(".cache/" in p or "runs/" in p or "upstream/" in p or ".venv/" in p for p in bundle.namelist())
digest = hashlib.sha256(archive.read_bytes()).hexdigest()
archive.with_suffix(".zip.sha256").write_text(digest+"  "+archive.name+"\n", encoding="ascii")
print(archive)
print("sha256", digest)
print("bytes", archive.stat().st_size)
