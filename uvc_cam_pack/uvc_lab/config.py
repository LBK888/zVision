from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields
import json
import math
from pathlib import Path
import re
import secrets
from typing import Any


@dataclass
class CameraConfig:
    id: str = "camera-0"
    source: int | str = 0
    name: str = "UVC camera"
    device_path: str = ""
    backend: str = "auto"
    width: int = 1280
    height: int = 720
    fps: float = 30.0
    fourcc: str = "MJPG"
    auto_exposure: bool = True
    exposure: float = -6.0
    auto_focus: bool = True
    focus: float = 0.0
    auto_white_balance: bool = True
    white_balance: float = 4500.0
    gain: float = 0.0
    scientific_lock: bool = False
    strict_controls: bool = False
    warmup_s: float = 2.5
    controls_interval_s: float = 1.0
    auto_exposure_on: float | None = None
    auto_exposure_off: float | None = None


@dataclass
class ROI:
    name: str = "plate"
    # Normalized coordinates in the rectified full image, x/y/width/height.
    rect: list[float] = field(default_factory=lambda: [0, 0, 1, 1])
    shape: str = "rectangle"
    rows: int = 1
    cols: int = 1


@dataclass
class GeometryConfig:
    rois: list[ROI] = field(default_factory=lambda: [ROI()])
    # Clockwise TL, TR, BR, BL in normalized sensor coordinates.
    perspective: list[list[float]] = field(default_factory=list)
    output_size: list[int] = field(default_factory=list)
    stabilize: bool = False
    max_shift_px: float = 12.0
    min_phase_response: float = 0.12
    # Per-camera overrides allow different plates and alignment on each camera.
    per_camera: dict[str, dict] = field(default_factory=dict)


@dataclass
class CaptureConfig:
    mode: str = "timelapse"
    interval_s: float = 600.0
    duration_s: float | None = 3600.0
    live: bool = True
    preview_fps: float = 8.0
    queue_memory_mb: int = 128
    overflow: str = "stop"
    min_free_gb: float = 1.0
    max_frame_age_s: float = 2.0


@dataclass
class StorageConfig:
    directory: str = "runs"
    raw: bool = True
    processed: bool = True
    image_format: str = "png"
    jpeg_quality: int = 97
    video: bool = False
    video_container: str = "mp4"
    video_fps: float = 20.0
    video_crf: int = 15
    video_bitrate_mbps: float = 40.0
    segment_s: float = 300.0
    annotated_video: bool = True


@dataclass
class ModelConfig:
    path: str = "yolo26n-seg.pt"
    task: str = "segment"
    confidence: float = 0.25
    classes: list[int] = field(default_factory=list)
    # Map model-local class IDs to a common semantic label, e.g. {"0": "animal"}.
    class_map: dict[str, str] = field(default_factory=dict)


@dataclass
class AnalysisConfig:
    models: list[ModelConfig] = field(default_factory=lambda: [ModelConfig()])
    device: str = "cpu"
    imgsz: int = 640
    consensus: str = "none"  # none, temporal, rotation
    consensus_iou: float = 0.3
    fusion_iou: float = 0.3
    tracker: str = "ukf"
    iou_threshold: float = 0.3
    max_age: int = 30
    min_hits: int = 3
    trail_length: int = 60
    gating_px: float = 1000.0
    area_weight: float = 0.2
    high_threshold: float = 0.5
    low_threshold: float = 0.1
    class_similarity: dict[str, float] = field(default_factory=dict)
    sequence_interval_s: float = 600.0
    pixel_size_um: float | None = None
    overlays: dict[str, bool] = field(default_factory=lambda: {
        "boxes": True, "labels": True, "masks": True, "keypoints": True,
        "trails": True, "velocity": False, "timestamp": True,
    })
    font_scale: float = 0.6
    line_width: int = 2
    timestamp_color: list[int] = field(default_factory=lambda: [255, 255, 255])
    redraw_original: bool = False
    input_images: str = "processed"
    use_capture_geometry: bool = True


@dataclass
class HardwareConfig:
    enabled: bool = False
    host: str = "127.0.0.1"
    port: int = 8765
    token: str = field(default_factory=lambda: secrets.token_urlsafe(24))
    targets: list[str] = field(default_factory=lambda: ["A1", "A2"])
    settle_s: float = 1.0
    timeout_s: float = 120.0
    flush_frames: int = 3
    sensor_port: str = ""
    sensor_baud: int = 115200
    sensor_interval_s: float = 5.0
    sensor_start_command: str = "START\n"


@dataclass
class AppConfig:
    schema_version: int = 1
    language: str = "zh_TW"
    experiment: str = "experiment"
    notes: str = ""
    cameras: list[CameraConfig] = field(default_factory=lambda: [CameraConfig()])
    geometry: GeometryConfig = field(default_factory=GeometryConfig)
    capture: CaptureConfig = field(default_factory=CaptureConfig)
    storage: StorageConfig = field(default_factory=StorageConfig)
    analysis: AnalysisConfig = field(default_factory=AnalysisConfig)
    hardware: HardwareConfig = field(default_factory=HardwareConfig)

    def to_dict(self, redact: bool = False) -> dict:
        result = asdict(self)
        if redact:
            result["hardware"]["token"] = "<redacted>"
        return result

    def save(self, path: str | Path, redact: bool = False) -> None:
        from .storage import atomic_json
        atomic_json(Path(path), self.to_dict(redact))

    @classmethod
    def from_dict(cls, data: dict) -> AppConfig:
        def make(kind, values):
            unknown = set(values) - {f.name for f in fields(kind)}
            if unknown:
                raise ValueError(f"Unknown {kind.__name__} fields: {sorted(unknown)}")
            return kind(**values)
        values = dict(data)
        if values.get("schema_version", 1) != 1:
            raise ValueError("Unsupported configuration schema")
        values["cameras"] = [make(CameraConfig, x) for x in values.get("cameras", [{}])]
        geo = dict(values.get("geometry", {}))
        geo["rois"] = [make(ROI, x) for x in geo.get("rois", [{}])]
        values["geometry"] = make(GeometryConfig, geo)
        for key, kind in [("capture", CaptureConfig), ("storage", StorageConfig), ("hardware", HardwareConfig)]:
            section = dict(values.get(key, {}))
            if key == "hardware" and section.get("token") == "<redacted>":
                section["token"] = secrets.token_urlsafe(24)
            values[key] = make(kind, section)
        ai = dict(values.get("analysis", {}))
        ai["models"] = [make(ModelConfig, x) for x in ai.get("models", [{}])]
        values["analysis"] = make(AnalysisConfig, ai)
        obj = make(cls, values)
        obj.validate()
        return obj

    @classmethod
    def load(cls, path: str | Path) -> AppConfig:
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))

    def validate(self) -> None:
        def positive(value: Any, name: str, allow_zero: bool = False):
            if not isinstance(value, (float, int)) or not math.isfinite(value) or value < 0 or (value == 0 and not allow_zero):
                raise ValueError(f"{name}: expected a finite positive number")
        def choice(value, allowed, name):
            if value not in allowed:
                raise ValueError(f"{name}: expected one of {allowed}")
        def unit(v, name):
            if not isinstance(v, (float, int)) or not math.isfinite(v) or not 0 <= v <= 1:
                raise ValueError(f"{name}: expected 0..1")
        if not self.cameras or len({c.id for c in self.cameras}) != len(self.cameras):
            raise ValueError("Camera IDs must be nonempty and unique")
        device_keys = [(c.backend, c.device_path or str(c.source)) for c in self.cameras]
        if len(set(device_keys)) != len(device_keys):
            raise ValueError("Each physical camera can only be added once")
        choice(self.language, ["zh_TW", "en", "ja"], "language")
        for camera in self.cameras:
            if not re.fullmatch(r"[A-Za-z0-9_-]+", camera.id):
                raise ValueError("Camera ID must contain only letters, digits, _ or -")
            for name in ["width", "height", "fps"]:
                positive(getattr(camera, name), name)
            if not isinstance(camera.width, int) or not isinstance(camera.height, int):
                raise ValueError("Resolution must be integer pixels")
            if len(camera.fourcc) != 4:
                raise ValueError("fourcc must have four characters")
            positive(camera.warmup_s, "warmup_s", True)
            positive(camera.controls_interval_s, "controls_interval_s", True)
            for name in ["exposure", "focus", "gain", "white_balance"]:
                if not math.isfinite(getattr(camera, name)):
                    raise ValueError(f"{name} must be finite")
        choice(self.capture.mode, ["timelapse", "continuous"], "mode")
        positive(self.capture.interval_s, "interval_s")
        if self.capture.duration_s is not None:
            positive(self.capture.duration_s, "duration_s")
        for name in ["preview_fps", "queue_memory_mb", "max_frame_age_s"]:
            positive(getattr(self.capture, name), name)
        choice(self.capture.overflow, ["stop", "drop"], "overflow")
        positive(self.capture.min_free_gb, "min_free_gb", True)
        s = self.storage
        if not any([s.raw, s.processed, s.video]):
            raise ValueError("Select at least one image or video output")
        choice(s.image_format, ["png", "tiff", "jpg"], "image_format")
        choice(s.video_container, ["mp4", "avi"], "video_container")
        positive(s.video_fps, "video_fps")
        positive(s.segment_s, "segment_s")
        positive(s.video_bitrate_mbps, "video_bitrate_mbps")
        if not 1 <= s.jpeg_quality <= 100 or not 0 <= s.video_crf <= 51:
            raise ValueError("Invalid image/video quality")
        validate_geometry(self.geometry)
        for override in self.geometry.per_camera.values():
            validate_geometry(geometry_override(self.geometry, override))
        a = self.analysis
        choice(a.consensus, ["none", "temporal", "rotation"], "consensus")
        choice(a.tracker, ["ukf", "sort", "bytetrack"], "tracker")
        choice(a.input_images, ["processed", "raw"], "input_images")
        for name in ["consensus_iou", "fusion_iou", "iou_threshold", "area_weight", "high_threshold", "low_threshold"]:
            unit(getattr(a, name), name)
        if min(a.consensus_iou, a.fusion_iou, a.iou_threshold) <= 0:
            raise ValueError("Object matching IoU thresholds must be greater than zero")
        if a.low_threshold >= a.high_threshold:
            raise ValueError("ByteTrack low threshold must be below high threshold")
        for name in ["imgsz", "max_age", "min_hits", "trail_length", "gating_px", "sequence_interval_s", "font_scale", "line_width"]:
            positive(getattr(a, name), name)
        for name in ["imgsz", "max_age", "min_hits", "trail_length", "line_width"]:
            if not isinstance(getattr(a, name), int):
                raise ValueError(f"{name} must be an integer")
        if len(a.timestamp_color) != 3 or any(not isinstance(v, int) or not 0 <= v <= 255 for v in a.timestamp_color):
            raise ValueError("timestamp_color must contain three BGR bytes")
        for pair, similarity in a.class_similarity.items():
            if pair.count("|") != 1:
                raise ValueError("Class similarity keys require labelA|labelB")
            unit(similarity, "class_similarity")
        if a.pixel_size_um is not None:
            positive(a.pixel_size_um, "pixel_size_um")
        if not a.models:
            raise ValueError("At least one model is required")
        for m in a.models:
            choice(m.task, ["detect", "segment", "pose"], "model task")
            unit(m.confidence, "confidence")
            if not m.path.strip():
                raise ValueError("Empty model path")
            if any(not isinstance(v, int) or v < 0 for v in m.classes):
                raise ValueError("Model class filters must be nonnegative integer IDs")
        h = self.hardware
        if h.enabled and (len(self.cameras) != 1 or self.capture.mode != "timelapse"):
            raise ValueError("Moving-camera handshake requires one camera in timelapse mode")
        if not 0 <= h.port <= 65535 or len(h.token) < 16:
            raise ValueError("API port invalid or token shorter than 16 characters")
        if not h.targets or len(set(h.targets)) != len(h.targets):
            raise ValueError("Target names must be nonempty and unique")
        if any(not re.fullmatch(r"[A-Za-z0-9_-]+", t) for t in h.targets):
            raise ValueError("Target names must contain only letters, digits, _ or -")
        positive(h.settle_s, "settle_s", True)
        positive(h.timeout_s, "timeout_s")
        positive(h.sensor_interval_s, "sensor_interval_s")
        if not isinstance(h.flush_frames, int) or h.flush_frames < 1:
            raise ValueError("flush_frames must be an integer >= 1")
        if h.sensor_port and h.sensor_baud <= 0:
            raise ValueError("Invalid serial baud rate")


def geometry_override(base: GeometryConfig, override: dict) -> GeometryConfig:
    values = asdict(base)
    values.update(override)
    values["per_camera"] = {}
    values["rois"] = [x if isinstance(x, ROI) else ROI(**x) for x in values["rois"]]
    return GeometryConfig(**values)


def validate_geometry(g: GeometryConfig) -> None:
    if not g.rois or len({r.name for r in g.rois}) != len(g.rois):
        raise ValueError("ROI names must be unique and nonempty")
    for r in g.rois:
        if not re.fullmatch(r"[A-Za-z0-9_-]+", r.name) or len(r.rect) != 4:
            raise ValueError("Invalid ROI name or rectangle")
        x, y, w, h = r.rect
        if not all(math.isfinite(v) for v in r.rect) or min(x, y) < 0 or min(w, h) <= 0 or x+w > 1.000001 or y+h > 1.000001:
            raise ValueError("ROI must fit within normalized image coordinates")
        if not isinstance(r.rows, int) or not isinstance(r.cols, int) or not 1 <= r.rows <= 128 or not 1 <= r.cols <= 128:
            raise ValueError("Grid rows/columns must be integers in 1..128")
        if r.shape not in ["rectangle", "ellipse"]:
            raise ValueError("Invalid ROI shape")
    if not math.isfinite(g.max_shift_px) or g.max_shift_px <= 0 or not 0 <= g.min_phase_response <= 1:
        raise ValueError("Invalid stabilization parameters")
    if g.output_size and (len(g.output_size) != 2 or any(not isinstance(x, int) or x < 2 for x in g.output_size)):
        raise ValueError("output_size requires two integer pixel dimensions >= 2")
    if g.perspective:
        import cv2
        import numpy as np
        p = np.asarray(g.perspective, np.float32)
        if p.shape != (4, 2) or not np.isfinite(p).all() or (p < 0).any() or (p > 1).any():
            raise ValueError("Perspective requires four normalized points")
        if not cv2.isContourConvex(p) or cv2.contourArea(p) < 0.001:
            raise ValueError("Perspective points must form a convex quadrilateral: TL, TR, BR, BL")
