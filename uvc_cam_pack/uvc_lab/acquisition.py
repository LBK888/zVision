from __future__ import annotations

from dataclasses import asdict, dataclass, field
import queue
import shutil
import threading
import time
from pathlib import Path

import cv2
import numpy as np

from .camera import open_camera
from .config import AppConfig, geometry_override
from .geometry import GeometryProcessor
from .hardware import Handshake, SensorWorker
from .storage import RunStore, VideoSink, estimate_storage, export_excel, write_image


@dataclass
class Packet:
    camera: str
    sequence: int
    timestamp: float
    monotonic: float
    image: np.ndarray
    target: str = "fixed"
    controls: dict = field(default_factory=dict)
    completion: threading.Event = field(default_factory=threading.Event)
    error: str | None = None
    outputs: list[str] = field(default_factory=list)


class CameraWorker(threading.Thread):
    def __init__(self, config, stop, start_gate, sink, status):
        super().__init__(daemon=True, name="capture-"+config.id)
        self.config, self.stop_event, self.start_gate = config, stop, start_gate
        self.sink, self.status = sink, status
        self.condition = threading.Condition()
        self.latest = None
        self.ready = threading.Event()
        self.error = None
        self.report = {}
        self.sequence = 0
        self.read_frames = 0
        self.first_read = None
        self.last_read = None
        self.started = None
        self.continuous = False
        self.last_controls = {}
        self.last_controls_time = -float("inf")

    def run(self):
        camera = None
        try:
            camera = open_camera(self.config)
            self.report = camera.report
            until = time.monotonic()+self.config.warmup_s
            while not self.stop_event.is_set():
                ok, image = camera.read()
                now, utc = time.monotonic(), time.time()
                if self.stop_event.is_set():
                    break
                if not ok or image is None:
                    raise RuntimeError(f"{self.config.id}: camera read failed")
                if image.shape[:2] != (self.config.height, self.config.width):
                    raise RuntimeError(f"{self.config.id}: unexpected frame dimensions {image.shape}")
                if now < until:
                    continue
                self.ready.set()
                if not self.start_gate.is_set():
                    continue
                if self.started is None:
                    self.started = now
                self.sequence += 1
                self.read_frames += 1
                self.first_read = self.first_read or now
                self.last_read = now
                packet = Packet(self.config.id, self.sequence, utc, now, image)
                if self.config.controls_interval_s and now-self.last_controls_time >= self.config.controls_interval_s:
                    self.last_controls = {"readback_monotonic": now, **camera.control_snapshot()}
                    self.last_controls_time = now
                packet.controls = self.last_controls
                with self.condition:
                    self.latest = packet
                    self.condition.notify_all()
                if self.continuous:
                    self.sink(packet)
        except Exception as error:
            self.error = str(error)
            self.status("camera_error", {"camera": self.config.id, "error": str(error)})
            self.stop_event.set()
        finally:
            self.ready.set()
            if camera:
                camera.release()
            with self.condition:
                self.condition.notify_all()

    def fresh(self, after: float, min_sequence: int = 0, timeout: float = 5) -> Packet:
        deadline = time.monotonic()+timeout
        with self.condition:
            while not self.stop_event.is_set():
                p = self.latest
                if p and p.monotonic >= after and p.sequence > min_sequence:
                    return Packet(p.camera, p.sequence, p.timestamp, p.monotonic, p.image, controls=p.controls)
                if time.monotonic() >= deadline:
                    raise TimeoutError(f"{self.config.id}: no fresh frame")
                self.condition.wait(.1)
        raise InterruptedError("Capture stopped")


class CaptureSession:
    def __init__(self, config: AppConfig, status=lambda kind, data: None, preview=lambda stream, image: None):
        self.config = AppConfig.from_dict(config.to_dict())
        self.status_callback, self.preview_callback = status, preview
        self.stop_event = threading.Event()
        self.start_gate = threading.Event()
        self.live = config.capture.live
        self.preview_only = False
        largest = max(c.width*c.height*3 for c in config.cameras)
        count = max(2, int(config.capture.queue_memory_mb*1024**2/largest))
        self.queue = queue.Queue(maxsize=count)
        self.store = None
        self.workers = []
        self.handshake = None
        self.writer_error = None
        self.write_failed = False
        self.writer = None
        self.sensor = None
        self.saved = 0
        self.saved_by_camera = {}
        self.dropped = 0
        self.discarded = 0
        self.videos = {}
        self.preview_times = {}
        self.geometry = {}
        self.started_monotonic = None
        self.saving_time = 0.0
        self.first_saved = None
        self.last_saved = None
        self.user_stopped = False
        self.duration_timer = None

    def event(self, kind, data):
        if self.store:
            self.store.event(kind, data)
        self.status_callback(kind, data)

    def stop(self):
        self.user_stopped = True
        self.stop_event.set()

    def enqueue(self, packet):
        if self.preview_only:
            self.emit_preview(packet.camera, packet.image)
            return
        try:
            self.queue.put_nowait(packet)
        except queue.Full:
            self.dropped += 1
            self.event("queue_overflow", {"camera": packet.camera, "sequence": packet.sequence, "policy": self.config.capture.overflow, "dropped": self.dropped})
            if self.config.capture.overflow == "stop":
                self.writer_error = "Writer queue full: capture stopped to preserve scientific integrity"
                self.stop_event.set()
            packet.error = "queue_overflow"
            packet.completion.set()

    def emit_preview(self, stream, image):
        if not self.live:
            return
        now = time.monotonic()
        if now-self.preview_times.get(stream, -1e9) < 1/self.config.capture.preview_fps:
            return
        self.preview_times[stream] = now
        h, w = image.shape[:2]
        factor = min(1, 960/w)
        self.preview_callback(stream, cv2.resize(image, (round(w*factor), round(h*factor))))

    def _save(self, packet):
        start = time.monotonic()
        cfg, s = self.config, self.config.storage
        if shutil.disk_usage(self.store.path).free < cfg.capture.min_free_gb*1e9:
            raise OSError("Low disk space reserve reached")
        key = (packet.camera, packet.target)
        if key not in self.geometry:
            geo = geometry_override(cfg.geometry, cfg.geometry.per_camera.get(packet.camera, {}))
            self.geometry[key] = GeometryProcessor(geo)
        relative = Path("images")/packet.camera/packet.target
        filename = f"{packet.sequence:010d}_{int(packet.timestamp*1e6)}"
        raw_path = ""
        if s.raw:
            raw_path = (relative/"raw"/(filename+"."+s.image_format)).as_posix()
            write_image(self.store.path/raw_path, packet.image, s.jpeg_quality)
            packet.outputs.append(raw_path)
        tiles, geometry = self.geometry[key].apply(packet.image)
        for tile in tiles:
            stream = f"{packet.camera}/{packet.target}/{tile.key}"
            processed_path = ""
            if s.processed:
                processed_path = (relative/tile.key/(filename+"."+s.image_format)).as_posix()
                write_image(self.store.path/processed_path, tile.image, s.jpeg_quality)
                packet.outputs.append(processed_path)
            if s.video:
                if stream not in self.videos:
                    self.videos[stream] = VideoSink(self.store.path/"videos"/stream/"capture", cfg, (tile.image.shape[1], tile.image.shape[0]))
                self.videos[stream].write(tile.image, packet.monotonic)
                if cfg.hardware.enabled:
                    # Complete the current target's file before allowing movement.
                    self.videos[stream]._close_segment()
                    packet.outputs.append(self.videos[stream].video_path.relative_to(self.store.path).as_posix())
            # Lossless geometric validity masks are saved once per stream; stabilization
            # border differences are represented by the per-frame transform as well.
            dynamic_mask = self.geometry[key].config.stabilize
            mask_path = (Path("masks")/stream/((filename if dynamic_mask else "valid")+".png")).as_posix()
            if not (self.store.path/mask_path).exists():
                write_image(self.store.path/mask_path, tile.valid_mask)
            meta = {"camera": packet.camera, "target": packet.target, "roi": tile.key,
                    "camera_controls": packet.controls,
                    "sensor_to_tile": tile.sensor_to_tile, "bounds": tile.bounds, "valid_mask": mask_path,
                    "geometry": geometry, "source_size": [packet.image.shape[1], packet.image.shape[0]]}
            meta["tile_size"] = [tile.image.shape[1], tile.image.shape[0]]
            self.store.frame(stream, packet.sequence, packet.timestamp, packet.monotonic, raw_path, processed_path, meta)
            self.emit_preview(stream, tile.image)
        self.saved += 1
        self.saved_by_camera[packet.camera] = self.saved_by_camera.get(packet.camera, 0)+1
        self.first_saved = self.first_saved or packet.monotonic
        self.last_saved = packet.monotonic
        self.saving_time += time.monotonic()-start
        packet.completion.set()

    def _writer_loop(self):
        while True:
            packet = self.queue.get()
            try:
                if packet is None:
                    return
                if self.write_failed:
                    packet.error = self.writer_error
                    self.discarded += 1
                    self.event("frame_not_saved", {"camera": packet.camera, "sequence": packet.sequence, "reason": self.writer_error})
                else:
                    self._save(packet)
            except Exception as error:
                packet.error = str(error)
                self.writer_error = str(error)
                self.write_failed = True
                self.discarded += 1
                self.event("writer_error", {"error": str(error), "camera": packet.camera, "sequence": packet.sequence})
                self.stop_event.set()
            finally:
                if packet is not None:
                    packet.completion.set()
                self.queue.task_done()

    def run(self, preview_only=False) -> Path | None:
        cfg = self.config
        cfg.validate()
        self.preview_only = preview_only
        error = None
        result = None
        if not preview_only:
            estimate = estimate_storage(cfg)
            if estimate["sufficient"] is False:
                raise OSError("Estimated storage exceeds free space after reserve")
            self.store = RunStore(cfg)
            result = self.store.path
            self.event("storage_estimate", estimate)
            self.writer = threading.Thread(target=self._writer_loop, daemon=True, name="writer")
            self.writer.start()
        try:
            for cam in cfg.cameras:
                worker = CameraWorker(cam, self.stop_event, self.start_gate, self.enqueue, self.event)
                worker.continuous = cfg.capture.mode == "continuous" or preview_only
                self.workers.append(worker)
                worker.start()
            deadline = time.monotonic()+max(c.warmup_s for c in cfg.cameras)+20
            while not all(w.ready.is_set() for w in self.workers):
                if self.stop_event.wait(.1):
                    break
                if time.monotonic() >= deadline:
                    raise TimeoutError("Camera startup timed out")
            for w in self.workers:
                self.event("camera_negotiated", {"camera": w.config.id, **w.report})
                if w.error:
                    raise RuntimeError(w.error)
            if self.stop_event.is_set():
                raise InterruptedError("Capture cancelled during startup")
            if cfg.hardware.sensor_port and self.store:
                self.sensor = SensorWorker(cfg.hardware, self.store, self.stop_event)
                self.sensor.start()
            if cfg.hardware.enabled and not preview_only:
                self.handshake = Handshake(cfg.hardware, self.event)
                address = self.handshake.start()
                self.event("api_started", {"host": address[0], "port": address[1]})
            self.started_monotonic = time.monotonic()
            if not preview_only and cfg.capture.duration_s is not None:
                self.duration_timer = threading.Timer(cfg.capture.duration_s, self.stop_event.set)
                self.duration_timer.daemon = True
                self.duration_timer.start()
            self.start_gate.set()
            self.event("capture_started", {"preview_only": preview_only})
            next_due = self.started_monotonic
            last_stats = 0
            while not self.stop_event.is_set():
                now = time.monotonic()
                if not preview_only and cfg.capture.duration_s is not None and now-self.started_monotonic >= cfg.capture.duration_s:
                    break
                if now-last_stats > 1:
                    self.status_callback("statistics", self.statistics())
                    last_stats = now
                if preview_only or cfg.capture.mode == "continuous":
                    self.stop_event.wait(.03)
                    continue
                if now < next_due:
                    self.stop_event.wait(min(.1, next_due-now))
                    # Timelapse preview samples the latest frame without enqueuing it.
                    for w in self.workers:
                        if w.latest:
                            self.emit_preview(w.config.id, w.latest.image)
                    continue
                targets = cfg.hardware.targets if self.handshake else ["fixed"]
                for target in targets:
                    if self.stop_event.is_set():
                        break
                    if self.handshake:
                        request_id = self.handshake.request(target)
                        arrival = self.handshake.wait_arrival(self.stop_event)
                        if self.stop_event.wait(max(0, arrival+cfg.hardware.settle_s-time.monotonic())):
                            break
                        worker = self.workers[0]
                        sequence = worker.latest.sequence if worker.latest else 0
                        packet = worker.fresh(arrival+cfg.hardware.settle_s, sequence+cfg.hardware.flush_frames-1,
                                              timeout=cfg.hardware.timeout_s)
                        self.handshake.capturing()
                        packets = [packet]
                    else:
                        # Common deadline, independently running cameras; record actual skew.
                        packets = [w.fresh(next_due, timeout=cfg.capture.max_frame_age_s+5) for w in self.workers]
                        skew = max(p.monotonic for p in packets)-min(p.monotonic for p in packets)
                        self.event("capture_cycle", {"scheduled_monotonic": next_due, "actual_skew_s": skew})
                    for packet in packets:
                        packet.target = target
                        self.enqueue(packet)
                    if self.handshake:
                        packet = packets[0]
                        while not packet.completion.wait(.1):
                            if self.writer_error:
                                raise RuntimeError(self.writer_error)
                        if packet.error:
                            raise RuntimeError(packet.error)
                        self.handshake.complete({"request_id": request_id, "camera": packet.camera, "sequence": packet.sequence,
                                                 "unix_s": packet.timestamp, "files": packet.outputs})
                        self.handshake.wait_acknowledgement(self.stop_event)
                # Advance against the original monotonic epoch; skip missed slots.
                next_due += cfg.capture.interval_s
                late = time.monotonic()-next_due
                if late >= 0:
                    skipped = int(late//cfg.capture.interval_s)+1
                    self.event("schedule_overrun", {"skipped_cycles": skipped, "late_s": late})
                    next_due += skipped*cfg.capture.interval_s
            if any(w.error for w in self.workers):
                raise RuntimeError("; ".join(w.error for w in self.workers if w.error))
            if self.writer_error:
                raise RuntimeError(self.writer_error)
        except InterruptedError:
            pass
        except Exception as caught:
            error = caught
            self.event("session_error", {"error": str(caught)})
        finally:
            if self.duration_timer:
                self.duration_timer.cancel()
            self.stop_event.set()
            if self.handshake:
                self.handshake.stop("error" if error else "stopped")
            for worker in self.workers:
                worker.join(timeout=5)
                if worker.is_alive():
                    error = error or RuntimeError(f"Driver blocked on {worker.config.id}; restart application before reusing camera")
            if self.sensor:
                self.sensor.join(timeout=3)
            if self.writer:
                self.queue.put(None)
                self.writer.join(timeout=60)
                if self.writer.is_alive():
                    # Never close a database while its writer still owns it.
                    self.store.manifest.update(state="incomplete", error="Writer did not stop; application restart required")
                    self.store.flush_manifest()
                    raise RuntimeError("Writer did not stop; output is marked incomplete for recovery")
            video_summary = {}
            for stream, video in self.videos.items():
                try:
                    video_summary[stream] = video.close()
                except Exception as caught:
                    error = error or caught
            error = error or (RuntimeError(self.writer_error) if self.writer_error else None)
            if self.store:
                summary = self.statistics()
                summary["video"] = video_summary
                summary["preview_enabled"] = self.live
                summary["negotiated_cameras"] = {w.config.id: w.report for w in self.workers}
                summary["error"] = str(error) if error else None
                self.store.finish("error" if error else ("cancelled" if self.user_stopped else "completed"), summary)
                try:
                    export_excel(self.store.path)
                except Exception as caught:
                    self.status_callback("export_error", {"error": str(caught)})
        if error:
            raise error
        self.status_callback("completed", {"path": str(result) if result else ""})
        return result

    def statistics(self):
        elapsed = time.monotonic()-self.started_monotonic if self.started_monotonic else 0
        cameras = {}
        for w in self.workers:
            interval = (w.last_read-w.first_read) if w.first_read and w.last_read else 0
            fps = (w.read_frames-1)/interval if interval else 0
            saved = self.saved_by_camera.get(w.config.id, 0)
            writer_capacity = saved/self.saving_time if self.saving_time else 0
            cameras[w.config.id] = {"frames": w.read_frames, "fps": fps, "saved": saved,
                                    "writer_capacity_fps": writer_capacity,
                                    "suggested_continuous_fps": min(fps, writer_capacity)*.9 if saved else 0}
        return {"elapsed_s": elapsed, "saved": self.saved, "dropped": self.dropped, "discarded_after_write_error": self.discarded, "queue": self.queue.qsize(),
                "queue_capacity": self.queue.maxsize, "cameras": cameras,
                "writer_mean_ms": self.saving_time/self.saved*1000 if self.saved else 0}
