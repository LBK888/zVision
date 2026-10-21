from copy import deepcopy
import json
from pathlib import Path
import shutil
import sqlite3
import threading
import queue
import time
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import numpy as np
import pytest

from uvc_lab.acquisition import CaptureSession
from uvc_lab.analysis import AnalysisJob, redraw_cache
from uvc_lab.config import AppConfig, CameraConfig, HardwareConfig, ModelConfig, ROI
from uvc_lab.hardware import Handshake
from uvc_lab.storage import read_image, write_image, export_excel, VideoSink


def configuration(tmp_path):
    c = AppConfig()
    c.cameras = [CameraConfig(id="demo-1", source="demo-1", width=160, height=120, warmup_s=0)]
    c.storage.directory = str(tmp_path/"runs")
    c.capture.interval_s = .15
    c.capture.duration_s = .5
    c.capture.min_free_gb = 0
    c.analysis.min_hits = 1
    return c


def test_multicamera_capture_preserves_raw_and_metadata(tmp_path):
    c = configuration(tmp_path)
    c.cameras.append(CameraConfig(id="demo-2", source="demo-2", width=160, height=120, warmup_s=0))
    c.geometry.rois = [ROI(rows=2, cols=3)]
    job = CaptureSession(c)
    root = job.run()
    assert job.saved >= 6
    assert job.dropped == 0
    assert (root/"report.xlsx").exists()
    settings = json.loads((root/"settings.json").read_text(encoding="utf-8"))
    assert settings["hardware"]["token"] == "<redacted>"
    with sqlite3.connect(root/"records.sqlite") as db:
        streams = list(db.execute("SELECT DISTINCT stream FROM frames"))
        assert len(streams) == 12
        meta = json.loads(db.execute("SELECT metadata FROM frames LIMIT 1").fetchone()[0])
        assert "sensor_to_tile" in meta
        raw, processed = db.execute("SELECT raw_path,processed_path FROM frames LIMIT 1").fetchone()
        assert read_image(root/raw).shape[:2] == (120, 160)
        assert read_image(root/processed).shape[:2] == (60, 53)


def test_writer_failure_marks_error_and_releases_workers(tmp_path, monkeypatch):
    c = configuration(tmp_path)
    job = CaptureSession(c)
    def fail(packet):
        raise OSError("Injected disk write failure")
    monkeypatch.setattr(job, "_save", fail)
    with pytest.raises(RuntimeError, match="Injected disk write failure"):
        job.run()
    assert not any(w.is_alive() for w in job.workers)
    manifest = json.loads((job.store.path/"manifest.json").read_text())
    assert manifest["state"] == "error"


def test_duration_limits_wait_for_unresponsive_motion_controller(tmp_path):
    c = configuration(tmp_path)
    c.hardware.enabled = True
    c.hardware.port = 0
    c.hardware.timeout_s = 30
    c.capture.duration_s = .25
    job = CaptureSession(c)
    start = time.monotonic()
    job.run()
    assert time.monotonic()-start < 2
    assert job.saved == 0


def test_manual_stop_is_cancelled_and_frames_are_finalized(tmp_path):
    c = configuration(tmp_path)
    c.capture.duration_s = None
    job = CaptureSession(c)
    timer = threading.Timer(.35, job.stop)
    timer.start()
    root = job.run()
    timer.join()
    assert json.loads((root/"manifest.json").read_text())["state"] == "cancelled"
    assert job.saved > 0


def test_queue_overflow_stops_but_drains_previously_accepted_frames(tmp_path, monkeypatch):
    c = configuration(tmp_path)
    c.capture.mode = "continuous"
    c.capture.duration_s = 3
    job = CaptureSession(c)
    job.queue = queue.Queue(maxsize=1)
    original = job._save
    def slow_save(packet):
        time.sleep(.15)
        original(packet)
    monkeypatch.setattr(job, "_save", slow_save)
    with pytest.raises(RuntimeError, match="queue full"):
        job.run()
    assert job.saved >= 2
    assert job.dropped == 1
    assert job.discarded == 0


def test_environment_excel_keeps_units_and_chart(tmp_path):
    from uvc_lab.storage import RunStore
    from openpyxl import load_workbook
    c = configuration(tmp_path)
    store = RunStore(c)
    store.sensor(1000, 23.5, 60.5, "Temperature: 23.5, Humidity: 60.5")
    store.sensor(1005, 23.6, 61.0, "Temperature: 23.6, Humidity: 61.0")
    store.finish("completed", {})
    path = export_excel(store.path)
    book = load_workbook(path)
    assert book["Environment"]["B2"].value == 23.5
    assert book["Environment"]["C3"].value == 61.0
    assert len(book["Environment"]._charts) == 1
    book.close()


def test_api_rejects_stale_arrivals_and_requires_bearer_auth():
    cfg = HardwareConfig(enabled=True, port=0)
    api = Handshake(cfg)
    address = api.start()
    url = f"http://127.0.0.1:{address[1]}"
    try:
        with pytest.raises(HTTPError) as error:
            urlopen(url+"/v1/state", timeout=2)
        assert error.value.code == 401
        request_id = api.request("A1")
        with pytest.raises(ValueError):
            api.arrive("old", "A1")
        request = Request(url+"/v1/arrived", data=json.dumps({"request_id": request_id, "target": "A1"}).encode(),
                          headers={"Authorization": "Bearer "+cfg.token, "Content-Type": "application/json"})
        state = json.loads(urlopen(request, timeout=2).read())
        assert state["phase"] == "settling"
        assert not state["can_move"]
        arrival = api.arrival_time
        api.arrive(request_id, "A1")
        assert api.arrival_time == arrival
        api.complete({"sequence": 1})
        assert api.snapshot()["can_move"]
        api.acknowledge(request_id, "A1")
        api.wait_acknowledgement(threading.Event())
        api.request("A2")
        with pytest.raises(ValueError):
            api.arrive(request_id, "A1")
    finally:
        api.stop()


def test_moving_target_handshake_uses_fresh_frames_and_persisted_files(tmp_path):
    c = configuration(tmp_path)
    c.hardware.enabled = True
    c.hardware.port = 0
    c.hardware.settle_s = .03
    c.hardware.timeout_s = 3
    c.hardware.flush_frames = 2
    c.capture.interval_s = .5
    c.capture.duration_s = .6
    job = CaptureSession(c)
    outcomes = []
    errors = []
    def controller():
        seen = set()
        while not job.stop_event.is_set():
            api = job.handshake
            if api:
                state = api.snapshot()
                if state["phase"] == "awaiting_arrival" and state["request_id"] not in seen:
                    seen.add(state["request_id"])
                    api.arrive(state["request_id"], state["target"])
                if state["phase"] == "capture_complete" and state["request_id"] not in {x[0] for x in outcomes}:
                    assert state["can_move"]
                    for path in state["capture"]["files"]:
                        assert (job.store.path/path).is_file()
                    outcomes.append((state["request_id"], state["target"]))
                    api.acknowledge(state["request_id"], state["target"])
            time.sleep(.002)
    thread = threading.Thread(target=controller, daemon=True)
    thread.start()
    root = job.run()
    thread.join(2)
    assert {target for _, target in outcomes} == {"A1", "A2"}
    with sqlite3.connect(root/"records.sqlite") as db:
        targets = {json.loads(x[0])["target"] for x in db.execute("SELECT metadata FROM frames")}
    assert targets == {"A1", "A2"}


class FakePipeline:
    def __init__(self, config):
        self.config = config
        self.provenance = [{"test_double": True}]

    def infer(self, image, valid_mask=None):
        result = []
        for index, model in enumerate(self.config.models):
            result.append([{"label": "animal", "bbox": [10, 10, 30, 30], "confidence": .9,
                            "model": str(index), "area_px2": 350 if model.task == "segment" else None,
                            "polygon": [[10, 10], [30, 10], [30, 30], [10, 30]] if model.task == "segment" else None,
                            "keypoints": [[20, 20, .8]] if model.task == "pose" else None}])
        return result


def test_sequence_analysis_exports_fused_data_and_reuses_cache(tmp_path):
    c = configuration(tmp_path)
    c.analysis.models = [ModelConfig(), ModelConfig(path="pose-test.pt", task="pose")]
    c.analysis.consensus = "temporal"
    c.storage.annotated_video = False
    for camera in ["cameraA", "cameraB"]:
        for index in range(3):
            write_image(tmp_path/"input"/camera/f"frame{index}.png", np.zeros((60, 80, 3), np.uint8))
    job = AnalysisJob(c, tmp_path/"input", pipeline_factory=FakePipeline)
    root = job.run()
    with sqlite3.connect(root/"records.sqlite") as db:
        records = list(db.execute("SELECT stream,detections,tracks FROM frames ORDER BY stream,sequence"))
    assert len(records) == 6
    for stream, ds, ts in records:
        detections, tracks = json.loads(ds), json.loads(ts)
        assert len(detections) == 1
        assert detections[0]["area_px2"] == 350
        assert detections[0]["keypoints"] == [[20, 20, .8]]
        assert len(detections[0]["models"]) == 2
        assert tracks[0]["id"] == 1
    if shutil.which("ffmpeg"):
        output = redraw_cache(root, c)
        assert len(list(output.rglob("*.mp4"))) == 2


def test_raw_only_capture_replays_recorded_roi_geometry(tmp_path):
    c = configuration(tmp_path)
    c.storage.processed = False
    c.geometry.rois = [ROI(rows=2, cols=3)]
    root = CaptureSession(c).run()
    analysis_config = configuration(tmp_path)
    analysis_config.storage.annotated_video = False
    # Deliberately leave analysis_config.geometry at its default 1x1.
    output = AnalysisJob(analysis_config, root, pipeline_factory=FakePipeline).run()
    with sqlite3.connect(output/"records.sqlite") as db:
        assert db.execute("SELECT COUNT(DISTINCT stream) FROM frames").fetchone()[0] == 6
        processed = db.execute("SELECT processed_path FROM frames LIMIT 1").fetchone()[0]
        assert read_image(output/processed).shape[:2] == (60, 53)
    analysis_config.analysis.input_images = "raw"
    analysis_config.analysis.use_capture_geometry = False
    reprocessed = AnalysisJob(analysis_config, root, pipeline_factory=FakePipeline).run()
    with sqlite3.connect(reprocessed/"records.sqlite") as db:
        assert db.execute("SELECT COUNT(DISTINCT stream) FROM frames").fetchone()[0] == 1
        processed = db.execute("SELECT processed_path FROM frames LIMIT 1").fetchone()[0]
        assert read_image(reprocessed/processed).shape[:2] == (120, 160)


@pytest.mark.skipif(not shutil.which("ffmpeg"), reason="FFmpeg not installed")
@pytest.mark.parametrize("container", ["mp4", "avi"])
def test_video_timestamp_resampling_and_finalization(tmp_path, container):
    c = configuration(tmp_path)
    c.storage.video_container = container
    c.storage.video_fps = 10
    c.capture.mode = "continuous"
    video = VideoSink(tmp_path/"video", c, (80, 60))
    image = np.zeros((60, 80, 3), np.uint8)
    for timestamp in [0, .05, .1, .4]:
        video.write(image, timestamp)
    summary = video.close()
    assert summary["frames"] == 5
    assert summary["duplicates"] == 2
    assert summary["dropped_by_resampler"] == 1
    assert not list(tmp_path.glob("*.partial.*"))
    import cv2
    cap = cv2.VideoCapture(str(tmp_path/f"video_0000.{container}"))
    assert cap.get(cv2.CAP_PROP_FRAME_COUNT) == 5
    assert cap.get(cv2.CAP_PROP_FPS) == pytest.approx(10)
    cap.release()
