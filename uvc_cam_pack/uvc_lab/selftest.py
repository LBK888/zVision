"""Explicit --self-test mode validates the distributed executable without a camera."""
from __future__ import annotations

from dataclasses import asdict
import os
from pathlib import Path
import shutil
import sqlite3
import traceback


def run_self_test(directory: Path) -> int:
    from .storage import atomic_json
    directory = directory.resolve()
    directory.mkdir(parents=True, exist_ok=True)
    report = {"ok": False, "checks": {}}
    try:
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
        from PyQt6.QtWidgets import QApplication
        from .acquisition import CaptureSession
        from .analysis import YoloPipeline, fuse_models
        from .camera import DemoCamera
        from .config import AppConfig, CameraConfig, ModelConfig
        from .runtime import camera_command, frozen, resources
        from .storage import VideoSink, read_image, write_image
        from .ui import MainWindow
        import cv2

        report["frozen"] = frozen()
        report["checks"]["ffmpeg"] = shutil.which("ffmpeg")
        if not report["checks"]["ffmpeg"]:
            raise RuntimeError("Bundled FFmpeg is unavailable")
        config = AppConfig()
        config.cameras = [CameraConfig(id="demo-1", source="demo-1", width=160, height=120, warmup_s=0)]
        report["checks"]["camera_subprocess"] = camera_command("modes", config.cameras[0])
        app = QApplication.instance() or QApplication([])
        for language in ["zh_TW", "en", "ja"]:
            config.language = language
            window = MainWindow(config)
            window.show()
            app.processEvents()
            window.grab().save(str(directory/f"ui-{language}.png"))
            window.close()
        report["checks"]["qt_languages"] = ["zh_TW", "en", "ja"]
        config.storage.directory = str(directory/"runs")
        config.storage.video = True
        config.capture.interval_s = .1
        config.capture.duration_s = .4
        config.capture.min_free_gb = 0
        session = CaptureSession(config)
        output = session.run()
        with sqlite3.connect(output/"records.sqlite") as db:
            count = db.execute("SELECT COUNT(*) FROM frames").fetchone()[0]
        if count < 2 or not (output/"report.xlsx").is_file():
            raise RuntimeError("Capture or Excel output failed")
        report["checks"]["capture"] = {"path": str(output), "frames": count}
        camera = DemoCamera(config.cameras[0])
        image = camera.read()[1]
        config.storage.video_container = "avi"
        sink = VideoSink(directory/"avi-test", config, (160, 120))
        sink.write(image, 0)
        report["checks"]["avi"] = sink.close()

        # Use the installed Ultralytics sample to exercise real segmentation/pose.
        import ultralytics
        sample = Path(ultralytics.__file__).parent/"assets"/"bus.jpg"
        image = read_image(sample) if sample.is_file() else image
        config.analysis.models = [ModelConfig(), ModelConfig(path="yolo26n-pose.pt", task="pose")]
        config.analysis.imgsz = 320
        pipeline = YoloPipeline(config.analysis)
        predictions = pipeline.infer(image)
        objects = fuse_models(predictions, config.analysis.fusion_iou)
        if sample.is_file() and not any(d.get("area_px2") and d.get("keypoints") for d in objects):
            raise RuntimeError("Real segmentation + pose fusion failed")
        report["checks"]["yolo26"] = {"objects": len(objects), "models": pipeline.provenance}
        report["ok"] = True
    except Exception:
        report["error"] = traceback.format_exc()
    atomic_json(directory/"self-test.json", report)
    return 0 if report["ok"] else 1
