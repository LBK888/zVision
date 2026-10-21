"""Resource and subprocess paths shared by source and one-file distributions."""
from __future__ import annotations

from dataclasses import asdict
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile


def frozen() -> bool:
    return bool(getattr(sys, "frozen", False))


def resources() -> Path:
    return Path(getattr(sys, "_MEIPASS", Path(__file__).resolve().parents[1]))


def data_directory() -> Path:
    if os.environ.get("UVC_LAB_DATA_DIR"):
        return Path(os.environ["UVC_LAB_DATA_DIR"]).expanduser().resolve()
    if frozen():
        return Path(os.environ.get("LOCALAPPDATA", str(Path.home())))/"UVC Lab"
    return Path.cwd()


def bootstrap() -> None:
    if not frozen():
        return
    home = data_directory()
    home.mkdir(parents=True, exist_ok=True)
    cache = home/"cache"
    (cache/"ultralytics").mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("UVC_LAB_CACHE", str(cache))
    os.environ.setdefault("YOLO_CONFIG_DIR", str(cache/"ultralytics"))
    tool_directory = resources()/"tools"
    os.environ["PATH"] = str(tool_directory)+os.pathsep+os.environ.get("PATH", "")
    os.chdir(home)
    # Windowed builds have no console. Give dependencies a real text stream.
    if sys.stdout is None or sys.stderr is None:
        (home/"logs").mkdir(exist_ok=True)
        log = (home/"logs"/"application.log").open("a", encoding="utf-8", buffering=1)
        if sys.stdout is None:
            sys.stdout = log
        if sys.stderr is None:
            sys.stderr = log


def model_path(configured: str) -> str:
    path = Path(configured).expanduser()
    if path.is_file():
        return str(path)
    # Only exact official basenames use bundled fallbacks. Missing custom paths
    # must not silently select a different model with the same basename.
    if path.name == configured:
        bundled = resources()/"models"/path.name
        if bundled.is_file():
            return str(bundled)
        cached = Path(os.environ.get("UVC_LAB_CACHE", ".cache"))/"models"/path.name
        if cached.is_file():
            return str(cached)
    return configured


def camera_command(action: str, camera=None) -> dict | list:
    with tempfile.TemporaryDirectory(prefix="uvc-camera-") as temp:
        result_path = Path(temp)/"result.json"
        if frozen():
            args = [sys.executable]
        else:
            args = [sys.executable, "-m", "uvc_lab"]
        args += ["--camera-tool", action, "--result-json", str(result_path)]
        if camera is not None:
            args += ["--camera-json", json.dumps(asdict(camera))]
        result = subprocess.run(args, capture_output=True, timeout=90,
                                creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
        if result_path.is_file():
            payload = json.loads(result_path.read_text(encoding="utf-8"))
            if not payload["ok"]:
                raise RuntimeError(payload["error"])
            if result.returncode:
                raise RuntimeError(f"Camera helper failed with exit code {result.returncode}")
            return payload["result"]
        error = (result.stderr or b"").decode("utf-8", errors="replace")
        raise RuntimeError(error[-3000:] or f"Camera helper produced no result (exit {result.returncode})")


def execute_camera_tool(action: str, camera_json: str | None, result_path: Path) -> int:
    from .camera import benchmark_resolution, discover_cameras, enumerate_modes
    from .config import CameraConfig
    from .storage import atomic_json
    try:
        if action == "scan":
            result = discover_cameras()
        else:
            camera = CameraConfig(**json.loads(camera_json or "{}"))
            if action == "modes":
                modes, evidence = enumerate_modes(camera)
                result = {"modes": [asdict(m) for m in modes], "evidence": evidence}
            elif action == "benchmark":
                result = benchmark_resolution(camera)
            else:
                raise ValueError("Unknown camera action")
        atomic_json(result_path, {"ok": True, "result": result})
        return 0
    except Exception as error:
        atomic_json(result_path, {"ok": False, "error": str(error)})
        return 1
