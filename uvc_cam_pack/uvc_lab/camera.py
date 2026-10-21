"""Camera negotiation reports evidence; advertised modes are never measured FPS."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from copy import deepcopy
import json
import os
import platform
import re
import shutil
import subprocess
import time
from typing import Callable

import cv2
import numpy as np

from .config import CameraConfig


def run_tool(args: list[str], timeout: float = 15) -> str:
    result = subprocess.run(args, capture_output=True, timeout=timeout,
                            creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
    return (result.stdout + result.stderr).decode("utf-8", errors="replace")


def backend_id(name: str = "auto") -> int:
    return {"dshow": cv2.CAP_DSHOW, "msmf": cv2.CAP_MSMF, "v4l2": cv2.CAP_V4L2,
            "avfoundation": cv2.CAP_AVFOUNDATION, "auto": cv2.CAP_DSHOW if os.name == "nt" else cv2.CAP_ANY}[name]


@dataclass(frozen=True)
class CameraMode:
    width: int
    height: int
    fps: float
    fourcc: str
    evidence: str = "advertised"
    min_fps: float | None = None


def parse_dshow_modes(output: str) -> list[CameraMode]:
    found = set()
    modes = []
    for line in output.splitlines():
        codec = re.search(r"(?:vcodec|pixel_format)=([^ ]+)", line)
        if not codec:
            continue
        fourcc = {"mjpeg": "MJPG", "yuyv422": "YUY2", "uyvy422": "UYVY", "nv12": "NV12",
                  "rgb24": "RGB3", "bgr24": "BGR3", "h264": "H264", "hevc": "HEVC"}.get(codec[1].lower())
        if not fourcc:
            continue
        endpoints = re.findall(r"s=(\d+)x(\d+)\s+fps=([\d.]+)", line)
        for w, h, fps in endpoints:
            key = int(w), int(h), float(fps), fourcc
            if key not in found:
                found.add(key)
                modes.append(CameraMode(*key, min_fps=float(endpoints[0][2])))
    return sorted(modes, key=lambda m: (m.width*m.height, m.fps, m.fourcc), reverse=True)


def parse_v4l2_modes(output: str) -> list[CameraMode]:
    modes, fourcc, size = [], None, None
    for line in output.splitlines():
        match = re.search(r"\[\d+\]: '(.{4})'", line)
        if match:
            fourcc = match[1]
        match = re.search(r"Size: Discrete (\d+)x(\d+)", line)
        if match:
            size = tuple(map(int, match.groups()))
        match = re.search(r"\(([\d.]+) fps\)", line)
        if match and fourcc and size:
            modes.append(CameraMode(*size, float(match[1]), fourcc))
    return list(dict.fromkeys(modes))


def discover_cameras() -> list[dict]:
    try:
        from cv2_enumerate_cameras import enumerate_cameras
        api = cv2.CAP_DSHOW if os.name == "nt" else cv2.CAP_ANY
        return [{"source": c.index, "name": c.name, "device_path": getattr(c, "path", ""),
                 "backend": "dshow" if os.name == "nt" else "auto"} for c in enumerate_cameras(api)]
    except ImportError:
        pass
    if os.name == "nt" and shutil.which("ffmpeg"):
        output = run_tool(["ffmpeg", "-hide_banner", "-list_devices", "true", "-f", "dshow", "-i", "dummy"])
        names = re.findall(r'"([^"]+)" \(video\)', output)
        return [{"source": i, "name": name, "device_path": "", "backend": "dshow"} for i, name in enumerate(names)]
    if platform.system() == "Linux":
        from pathlib import Path
        return [{"source": str(p), "name": p.name, "device_path": str(p), "backend": "v4l2"}
                for p in sorted(Path("/dev").glob("video*"))]
    # This is an explicit fallback. Running in a subprocess bounds slow drivers.
    devices = []
    for index in range(8):
        cap = cv2.VideoCapture(index, backend_id())
        if cap.isOpened():
            devices.append({"source": index, "name": f"Camera {index}", "device_path": "", "backend": "auto"})
        cap.release()
    return devices


def enumerate_modes(config: CameraConfig) -> tuple[list[CameraMode], str]:
    if str(config.source).startswith("demo"):
        return [CameraMode(1280, 720, 30, "MJPG", "simulation"), CameraMode(1920, 1080, 22, "MJPG", "simulation"),
                CameraMode(3264, 2448, 10, "MJPG", "simulation")], "simulation"
    if os.name == "nt" and shutil.which("ffmpeg"):
        device = config.device_path or config.name
        if device.startswith("\\\\?\\"):
            device = "@device_pnp_"+device
        device = device.replace("@device:pnp:", "@device_pnp_")
        output = run_tool(["ffmpeg", "-hide_banner", "-list_options", "true", "-f", "dshow", "-i", f"video={device}"])
        modes = parse_dshow_modes(output)
        if modes:
            return modes, output
    if platform.system() == "Linux" and shutil.which("v4l2-ctl"):
        device = str(config.source) if isinstance(config.source, str) else f"/dev/video{config.source}"
        output = run_tool(["v4l2-ctl", "-d", device, "--list-formats-ext"])
        modes = parse_v4l2_modes(output)
        if modes:
            return modes, output
    # Read-back verification rather than presenting common modes as capabilities.
    modes = []
    cap = cv2.VideoCapture(config.source, backend_id(config.backend))
    try:
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open camera {config.id}")
        for w, h in [(640, 480), (1280, 720), (1280, 800), (1600, 1200), (1920, 1080), (2592, 1944), (3264, 2448)]:
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*config.fourcc))
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
            cap.set(cv2.CAP_PROP_FPS, config.fps)
            ok, frame = cap.read()
            if ok and frame.shape[:2] == (h, w):
                mode = CameraMode(w, h, cap.get(cv2.CAP_PROP_FPS) or config.fps, config.fourcc, "probe-readback")
                if mode not in modes:
                    modes.append(mode)
    finally:
        cap.release()
    return modes, "Fallback probe: this is a verified subset, not a complete driver mode enumeration."


class DemoCamera:
    def __init__(self, config: CameraConfig):
        self.config = config
        self.index = 0
        self.deadline = time.monotonic()
        self.report = {"simulation": True, "width": config.width, "height": config.height, "fps": config.fps,
                       "fourcc": config.fourcc, "controls": {}, "warnings": []}

    def read(self):
        wait = self.deadline-time.monotonic()
        if wait > 0:
            time.sleep(wait)
        self.deadline = max(self.deadline+1/self.config.fps, time.monotonic())
        self.index += 1
        h, w = self.config.height, self.config.width
        image = np.full((h, w, 3), (42, 48, 43), np.uint8)
        for r in range(2):
            for c in range(3):
                center = (int((c+.5)*w/3), int((r+.5)*h/2))
                cv2.circle(image, center, int(min(w/3, h/2)*.36), (130, 143, 125), 2)
                angle = self.index*.05+(r*3+c)
                p = (center[0]+int(np.sin(angle)*w/32), center[1]+int(np.cos(angle)*h/25))
                cv2.ellipse(image, p, (max(3, w//65), max(2, h//65)), self.index*3, 0, 360, (90, 184, 180), -1)
        cv2.putText(image, f"SIMULATION / {self.config.id} / {self.index}", (20, 35), cv2.FONT_HERSHEY_SIMPLEX,
                    .7, (210, 225, 215), 1, cv2.LINE_AA)
        return True, image

    def release(self):
        pass

    def control_snapshot(self):
        return {"simulation": True}


def open_camera(config: CameraConfig):
    if str(config.source).startswith("demo"):
        return DemoCamera(config)
    source = config.source
    if os.name == "nt" and config.device_path:
        from cv2_enumerate_cameras import enumerate_cameras
        candidates = enumerate_cameras(backend_id(config.backend))
        matched = [c for c in candidates if getattr(c, "path", "").casefold() == config.device_path.casefold()]
        if len(matched) != 1:
            raise RuntimeError(f"Saved physical device unavailable or ambiguous: {config.id}")
        source = matched[0].index
    cap = cv2.VideoCapture(source, backend_id(config.backend))
    if not cap.isOpened():
        cap.release()
        raise RuntimeError(f"Cannot open {config.id}: {config.source}")
    report = {"simulation": False, "backend": cap.getBackendName(), "resolved_source": source,
              "device_path": config.device_path, "controls": {}, "warnings": []}
    try:
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*config.fourcc))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.height)
        cap.set(cv2.CAP_PROP_FPS, config.fps)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        is_v4l = "V4L" in report["backend"]
        auto_on = config.auto_exposure_on if config.auto_exposure_on is not None else (3 if is_v4l else .75)
        auto_off = config.auto_exposure_off if config.auto_exposure_off is not None else (1 if is_v4l else .25)
        ae, af, awb = (False, False, False) if config.scientific_lock else (config.auto_exposure, config.auto_focus, config.auto_white_balance)
        settings = [("auto_exposure", cv2.CAP_PROP_AUTO_EXPOSURE, auto_on if ae else auto_off),
                    ("auto_focus", cv2.CAP_PROP_AUTOFOCUS, float(af)),
                    ("auto_white_balance", cv2.CAP_PROP_AUTO_WB, float(awb))]
        if not ae:
            settings.extend([("exposure", cv2.CAP_PROP_EXPOSURE, config.exposure), ("gain", cv2.CAP_PROP_GAIN, config.gain)])
        if not af:
            settings.append(("focus", cv2.CAP_PROP_FOCUS, config.focus))
        if not awb:
            settings.append(("white_balance", cv2.CAP_PROP_WB_TEMPERATURE, config.white_balance))
        for name, prop, value in settings:
            accepted = cap.set(prop, value)
            actual = cap.get(prop)
            verified = accepted and abs(actual-value) <= max(.01, abs(value)*.01)
            report["controls"][name] = {"requested": value, "set_accepted": accepted, "readback": actual, "verified": verified}
            if not verified:
                report["warnings"].append(f"{name}: requested {value}, readback {actual}, accepted={accepted}")
        report.update(width=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), height=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                      fps=cap.get(cv2.CAP_PROP_FPS), fourcc=decode_fourcc(cap.get(cv2.CAP_PROP_FOURCC)))
        if (report["width"], report["height"]) != (config.width, config.height):
            raise RuntimeError(f"{config.id}: driver substituted resolution {report['width']}x{report['height']}")
        if report["fourcc"] != config.fourcc:
            report["warnings"].append(f"Pixel format substituted: {config.fourcc} -> {report['fourcc']}")
        if config.strict_controls and report["warnings"]:
            raise RuntimeError("Unverified camera settings: "+"; ".join(report["warnings"]))
        return OpenCVCamera(cap, report)
    except BaseException:
        cap.release()
        raise


def decode_fourcc(value: float) -> str:
    v = int(value)
    return "".join(chr((v >> (8*i)) & 255) for i in range(4))


class OpenCVCamera:
    def __init__(self, cap, report):
        self.cap, self.report = cap, report

    def read(self):
        return self.cap.read()

    def release(self):
        self.cap.release()

    def control_snapshot(self):
        properties = {"exposure": cv2.CAP_PROP_EXPOSURE, "gain": cv2.CAP_PROP_GAIN,
                      "focus": cv2.CAP_PROP_FOCUS, "white_balance": cv2.CAP_PROP_WB_TEMPERATURE,
                      "auto_exposure": cv2.CAP_PROP_AUTO_EXPOSURE, "auto_focus": cv2.CAP_PROP_AUTOFOCUS,
                      "auto_white_balance": cv2.CAP_PROP_AUTO_WB}
        return {key: self.cap.get(prop) for key, prop in properties.items()}


def measure_camera(config: CameraConfig, seconds: float = 8, stop: Callable[[], bool] = lambda: False) -> dict:
    cam = open_camera(config)
    try:
        warmup_end = time.monotonic()+config.warmup_s
        stamps = []
        failures = 0
        start = None
        while not stop():
            ok, frame = cam.read()
            now = time.monotonic()
            if now < warmup_end:
                continue
            if start is None:
                start = now
            if ok and frame is not None:
                if frame.shape[:2] != (config.height, config.width):
                    raise RuntimeError("Frame dimensions disagree with selected resolution")
                stamps.append(now)
            else:
                failures += 1
                if failures > 10:
                    raise RuntimeError("Repeated camera read failure")
            if now-start >= seconds:
                break
        if len(stamps) < 2:
            raise RuntimeError("Insufficient frames for benchmark")
        gaps = np.diff(stamps)
        fps = (len(stamps)-1)/(stamps[-1]-stamps[0])
        return {"camera": config.id, "requested": asdict(config), "negotiated": cam.report,
                "measured_fps": fps, "conservative_fps": float(1/np.percentile(gaps, 95)),
                "read_gap_p95_ms": float(np.percentile(gaps, 95)*1000), "frames": len(stamps),
                "read_failures": failures, "duration_s": stamps[-1]-stamps[0],
                "scope": "capture only; run system benchmark for processing, disk, USB contention and preview cost",
                "timestamp_basis": "host monotonic read completion; sensor timestamps unavailable"}
    finally:
        cam.release()


def benchmark_resolution(config: CameraConfig, seconds: float = 8) -> dict:
    requested = deepcopy(config)
    candidate = deepcopy(config)
    advertised = None
    note = ""
    try:
        modes, note = enumerate_modes(config)
        rates = [m.fps for m in modes if m.width == config.width and m.height == config.height and m.fourcc == config.fourcc]
        if rates:
            advertised = max(rates)
            candidate.fps = max(candidate.fps, advertised)
    except Exception as error:
        note = "Mode enumeration unavailable: "+str(error)
    result = measure_camera(candidate, seconds)
    result.update(original_preset=asdict(requested), advertised_max_fps=advertised,
                  tested_fps_request=candidate.fps, mode_enumeration_evidence=note,
                  suggested_fps=min(result["measured_fps"], result["conservative_fps"])*.9,
                  interpretation="Measured capture ceiling at highest advertised/requested rate for selected resolution, pixel format and exposure; finite test, not a hardware guarantee")
    return result


if __name__ == "__main__":
    import sys
    action = sys.argv[1]
    if action == "scan":
        print(json.dumps(discover_cameras(), ensure_ascii=False))
    else:
        conf = CameraConfig(**json.loads(sys.argv[2]))
        if action == "modes":
            modes, evidence = enumerate_modes(conf)
            print(json.dumps({"modes": [asdict(x) for x in modes], "evidence": evidence}, ensure_ascii=False))
        elif action == "benchmark":
            print(json.dumps(benchmark_resolution(conf), ensure_ascii=False))
