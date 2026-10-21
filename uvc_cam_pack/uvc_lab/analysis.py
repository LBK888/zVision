from __future__ import annotations

from collections import defaultdict
from copy import deepcopy
from dataclasses import asdict, dataclass
import json
import os
import base64
import zlib
from pathlib import Path
import re
import sqlite3
import threading
import time

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment

from .config import AppConfig, AnalysisConfig, ModelConfig, geometry_override
from .geometry import GeometryProcessor, transform_points
from .storage import RunStore, VideoSink, atomic_json, export_excel, file_hash, read_image, write_image
from .tracking import StreamTracker, iou


def match_detections(anchors: list[dict], others: list[dict], threshold: float) -> list[tuple[int, int]]:
    if not anchors or not others:
        return []
    cost = np.full((len(anchors), len(others)), 1e6)
    for i, a in enumerate(anchors):
        for j, b in enumerate(others):
            score = iou(a["bbox"], b["bbox"])
            if a["label"] == b["label"] and score >= threshold:
                cost[i, j] = 1-score
    rows, cols = linear_sum_assignment(cost)
    return [(int(i), int(j)) for i, j in zip(rows, cols) if cost[i, j] < 1e6]


def consensus(views: list[list[dict]], threshold: float, center_index: int = 0) -> list[dict]:
    """At least two distinct views must match a class AND object, one-to-one.

    A center miss supported by both neighbors is explicitly inferred. Geometry
    from another time is never reported as a measured center area or pose.
    """
    anchors = deepcopy(views[center_index])
    for d in anchors:
        d["support_views"] = [center_index]
        d["observed_in_center"] = True
    remaining = [i for i in range(len(views)) if i != center_index]
    used = {i: set() for i in remaining}
    for index in remaining:
        for i, j in match_detections(anchors, views[index], threshold):
            anchors[i]["support_views"].append(index)
            used[index].add(j)
    accepted = [d for d in anchors if len(d["support_views"]) >= 2]
    # Temporal neighbor agreement with no center observation is a positive
    # classification only, not a measured segmentation/pose at this time.
    if len(remaining) == 2:
        a, b = remaining
        left = [(j, d) for j, d in enumerate(views[a]) if j not in used[a]]
        right = [(j, d) for j, d in enumerate(views[b]) if j not in used[b]]
        for i, j in match_detections([d for _, d in left], [d for _, d in right], threshold):
            x, y = left[i][1], right[j][1]
            accepted.append({"label": x["label"], "bbox": ((np.array(x["bbox"])+y["bbox"])/2).tolist(),
                             "confidence": min(x["confidence"], y["confidence"]), "support_views": [a, b],
                             "observed_in_center": False, "inferred": True, "model": x.get("model", ""),
                             "area_px2": None, "polygon": None, "keypoints": None})
    for d in accepted:
        d["votes"] = len(d["support_views"])
    return accepted


def undo_rotation(detection: dict, rotation: int, original_size: tuple[int, int]) -> dict:
    d = deepcopy(detection)
    w, h = original_size
    if rotation == 0:
        return d
    def matrix(edge=False):
        offset = 0 if edge else 1
        if rotation == 90:  # inverse of clockwise
            return [[0, 1, 0], [-1, 0, h-offset], [0, 0, 1]]
        return [[0, -1, w-offset], [1, 0, 0], [0, 0, 1]]
    x1, y1, x2, y2 = d["bbox"]
    corners = np.array(transform_points([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], matrix(True)))
    d["bbox"] = [float(corners[:, 0].min()), float(corners[:, 1].min()), float(corners[:, 0].max()), float(corners[:, 1].max())]
    if d.get("polygon"):
        d["polygon"] = transform_points(d["polygon"], matrix())
    if d.get("keypoints"):
        points = np.array(d["keypoints"], dtype=float)
        points[:, :2] = transform_points(points[:, :2], matrix())
        d["keypoints"] = points.tolist()
    if d.get("mask_bits"):
        binary = decode_mask(d["mask_bits"]).astype(np.uint8)
        restored = cv2.rotate(binary, cv2.ROTATE_90_COUNTERCLOCKWISE if rotation == 90 else cv2.ROTATE_90_CLOCKWISE)
        d["mask_bits"] = encode_mask(restored > 0)
    return d


def fuse_models(per_model: list[list[dict]], threshold: float) -> list[dict]:
    """Associate segmentation and pose as attributes of the same object."""
    objects = []
    for detections in per_model:
        matches = match_detections(objects, detections, threshold)
        matched = set()
        for i, j in matches:
            source = detections[j]
            objects[i].setdefault("measurements", []).append(deepcopy(source))
            # Prefer a measured mask and pose from the center observation.
            for attr in ["polygon", "area_px2", "keypoints", "mask_bits"]:
                if source.get(attr) is not None:
                    objects[i][attr] = deepcopy(source[attr])
            objects[i]["models"].append(source["model"])
            matched.add(j)
        for j, source in enumerate(detections):
            if j not in matched:
                item = deepcopy(source)
                item["measurements"] = [deepcopy(source)]
                item["models"] = [source["model"]]
                objects.append(item)
    return objects


class YoloPipeline:
    def __init__(self, config: AnalysisConfig):
        cache = Path(os.environ.get("UVC_LAB_CACHE", str(Path.cwd()/".cache"))).resolve()/"ultralytics"
        cache.mkdir(parents=True, exist_ok=True)
        os.environ.setdefault("YOLO_CONFIG_DIR", str(cache))
        from ultralytics import YOLO
        from .runtime import model_path
        self.config = config
        self.models = [YOLO(model_path(m.path), task=m.task) for m in config.models]
        for m, model in zip(config.models, self.models):
            if model.task != m.task:
                raise ValueError(f"Model task mismatch: {m.path}: configured {m.task}, actual {model.task}")
        self.provenance = [{"config": asdict(m), "resolved_path": str(getattr(model, "ckpt_path", None) or m.path),
                            "sha256": file_hash(getattr(model, "ckpt_path", None) or m.path), "names": model.names}
                           for m, model in zip(config.models, self.models)]

    def infer_model(self, index: int, image: np.ndarray, valid_mask=None) -> list[dict]:
        cfg, model = self.config.models[index], self.models[index]
        # Preserve low-confidence evidence for ByteTrack's second association.
        conf = min(cfg.confidence, self.config.low_threshold) if self.config.tracker == "bytetrack" else cfg.confidence
        results = model.predict(image, conf=conf, classes=cfg.classes or None, imgsz=self.config.imgsz,
                                device=self.config.device, verbose=False, retina_masks=True)
        detections = []
        for result in results:
            if result.boxes is None:
                continue
            boxes = result.boxes.xyxy.cpu().numpy()
            scores = result.boxes.conf.cpu().numpy()
            ids = result.boxes.cls.cpu().numpy().astype(int)
            polygons = result.masks.xy if result.masks is not None else None
            masks = result.masks.data.cpu().numpy() if result.masks is not None else None
            kps = result.keypoints.data.cpu().numpy() if result.keypoints is not None else None
            for j, (bbox, score, cls_id) in enumerate(zip(boxes, scores, ids)):
                if valid_mask is not None:
                    cx = min(image.shape[1]-1, max(0, int((bbox[0]+bbox[2])/2)))
                    cy = min(image.shape[0]-1, max(0, int((bbox[1]+bbox[3])/2)))
                    if valid_mask[cy, cx] == 0:
                        continue
                label = cfg.class_map.get(str(cls_id), str(result.names[cls_id]))
                polygon = polygons[j].tolist() if polygons is not None else None
                # Count binary pixels at the actual tile resolution; no polygon
                # area approximation or arbitrary buffer applied to biology data.
                area = None
                mask_bits = None
                if masks is not None:
                    binary = cv2.resize(masks[j], (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST) > .5
                    if valid_mask is not None:
                        binary &= valid_mask > 0
                    area = int(np.count_nonzero(binary))
                    contours, _ = cv2.findContours(binary.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    polygon = max(contours, key=cv2.contourArea).reshape(-1, 2).tolist() if contours else None
                    mask_bits = encode_mask(binary)
                detections.append({"label": label, "class_id": int(cls_id), "bbox": bbox.tolist(),
                                   "confidence": float(score), "model": str(index)+":"+Path(cfg.path).name,
                                   "positive": bool(score >= cfg.confidence),
                                   "task": cfg.task, "polygon": polygon, "area_px2": area,
                                   "mask_bits": mask_bits,
                                   "keypoints": kps[j].tolist() if kps is not None else None})
        return detections

    def infer(self, image, valid_mask=None) -> list[list[dict]]:
        per_model = []
        self.last_evidence = []
        for index in range(len(self.models)):
            central = self.infer_model(index, image, valid_mask)
            evidence = {"model": str(index), "mode": self.config.consensus, "views": [deepcopy(central)]}
            if self.config.consensus == "rotation":
                cw = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
                ccw = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
                size = (image.shape[1], image.shape[0])
                cw_mask = cv2.rotate(valid_mask, cv2.ROTATE_90_CLOCKWISE) if valid_mask is not None else None
                ccw_mask = cv2.rotate(valid_mask, cv2.ROTATE_90_COUNTERCLOCKWISE) if valid_mask is not None else None
                views = [central,
                         [undo_rotation(d, 90, size) for d in self.infer_model(index, cw, cw_mask)],
                         [undo_rotation(d, -90, size) for d in self.infer_model(index, ccw, ccw_mask)]]
                evidence["views"] = deepcopy(views)
                low_evidence = [d for d in central if not d["positive"]]
                views = [[d for d in view if d["positive"]] for view in views]
                central = consensus(views, self.config.consensus_iou)+low_evidence
                # Rotations are the same timepoint; geometry is still observed.
                for d in central:
                    if d.get("inferred"):
                        d["observed_in_center"] = False
                        candidates = [other for vi in d["support_views"] for other in views[vi]
                                      if other["label"] == d["label"] and iou(other["bbox"], d["bbox"]) >= self.config.consensus_iou]
                        if candidates:
                            best = max(candidates, key=lambda other: other["confidence"])
                            for attr in ["bbox", "polygon", "area_px2", "keypoints", "mask_bits", "confidence"]:
                                d[attr] = deepcopy(best.get(attr))
                            d["inferred"] = False
                        d["inference_basis"] = "rotated_views_same_time"
                    d["positive"] = bool(d.get("votes", 0) >= 2)
            per_model.append(central)
            self.last_evidence.append(evidence)
        return per_model


def encode_mask(binary):
    return {"encoding": "zlib-packbits-big", "shape": list(binary.shape),
            "data": base64.b64encode(zlib.compress(np.packbits(binary).tobytes())).decode("ascii")}


def decode_mask(record):
    bits = np.frombuffer(zlib.decompress(base64.b64decode(record["data"])), dtype=np.uint8)
    return np.unpackbits(bits)[:np.prod(record["shape"])].reshape(record["shape"]).astype(bool)


def color_for(index: int) -> tuple[int, int, int]:
    return ((index*67+70) % 200+40, (index*41+110) % 200+40, (index*97+50) % 200+40)


def render(image: np.ndarray, detections: list[dict], tracks: list[dict], timestamp: float, config: AnalysisConfig, highlight=None) -> np.ndarray:
    canvas = image.copy()
    o, font, thickness = config.overlays, config.font_scale, config.line_width
    for d in detections:
        color = color_for(d.get("track_id", sum(map(ord, d["label"]))))
        if d.get("track_id") == highlight:
            color = (0, 235, 255)
        x1, y1, x2, y2 = map(lambda v: int(round(v)), d["bbox"])
        if o.get("masks") and d.get("polygon"):
            poly = np.array(d["polygon"], np.int32)
            tint = canvas.copy()
            if d.get("mask_bits") and d["mask_bits"]["shape"] == list(canvas.shape[:2]):
                tint[decode_mask(d["mask_bits"])] = color
            else:
                cv2.fillPoly(tint, [poly], color)
            canvas = cv2.addWeighted(canvas, .7, tint, .3, 0)
            cv2.polylines(canvas, [poly], True, color, thickness)
        if o.get("boxes"):
            cv2.rectangle(canvas, (x1, y1), (x2, y2), color, thickness)
        if o.get("labels"):
            label = f"{d['label']} {d['confidence']:.2f}"
            if "track_id" in d:
                label += f" #{d['track_id']}"
            if "votes" in d:
                label += f" [{d['votes']}/3]"
            cv2.putText(canvas, label, (x1, max(15, y1-5)), cv2.FONT_HERSHEY_SIMPLEX, font, color, thickness, cv2.LINE_AA)
        if o.get("keypoints") and d.get("keypoints"):
            for p in d["keypoints"]:
                if len(p) < 3 or p[2] >= .3:
                    cv2.circle(canvas, (round(p[0]), round(p[1])), max(2, thickness+1), color, -1)
    for t in tracks:
        color = color_for(t["id"])
        if o.get("trails") and len(t["trail"]) >= 2:
            cv2.polylines(canvas, [np.array(t["trail"], np.int32)], False, color, thickness)
        if o.get("velocity"):
            a = ((t["bbox"][0]+t["bbox"][2])/2, (t["bbox"][1]+t["bbox"][3])/2)
            # Arrow represents one second of movement; numerical value is px/s.
            b = (a[0]+t["velocity_px_s"][0], a[1]+t["velocity_px_s"][1])
            cv2.arrowedLine(canvas, tuple(map(round, a)), tuple(map(round, b)), color, thickness)
    if o.get("timestamp"):
        text = f"t = {timestamp:.3f} s"
        cv2.putText(canvas, text, (14, max(24, canvas.shape[0]-18)), cv2.FONT_HERSHEY_SIMPLEX, font,
                    tuple(config.timestamp_color), thickness, cv2.LINE_AA)
    return canvas


@dataclass
class InputFrame:
    path: Path
    timestamp: float
    sequence: int
    stream: str
    metadata: dict
    processed: bool = False


def natural_key(path):
    return [int(part) if part.isdigit() else part.lower() for part in re.split(r"(\d+)", str(path))]


def discover_sequences(root: Path, interval: float, prefer_raw=False, reprocess_raw=False) -> dict[str, list[InputFrame]]:
    root = root.resolve()
    db_path = root/"records.sqlite"
    if db_path.exists():
        groups = defaultdict(list)
        seen_raw = set()
        with sqlite3.connect(db_path) as db:
            for stream, seq, ts, raw, processed, metadata in db.execute("SELECT stream,sequence,timestamp,raw_path,processed_path,metadata FROM frames ORDER BY stream,timestamp,id"):
                selected = (raw or processed) if prefer_raw else (processed or raw)
                if not selected:
                    continue
                p = root/selected
                if p.is_file():
                    meta = json.loads(metadata)
                    meta["original_path"] = str(root/raw) if raw and (root/raw).is_file() else ""
                    if meta.get("valid_mask"):
                        meta["valid_mask_path"] = str(root/meta["valid_mask"])
                    is_processed = bool(processed) and selected == processed
                    if not is_processed and reprocess_raw:
                        # Several prior ROI rows can refer to the same sensor image.
                        key = str(p.resolve())
                        if key in seen_raw:
                            continue
                        seen_raw.add(key)
                        stream = f"{meta.get('camera', stream.split('/')[0])}/{meta.get('target', 'fixed')}"
                        for field in ["roi", "sensor_to_tile", "tile_size", "valid_mask", "valid_mask_path", "valid_region_bits"]:
                            meta.pop(field, None)
                    groups[stream].append(InputFrame(p, ts, seq, stream, meta, is_processed))
        return dict(groups)
    groups = defaultdict(list)
    files = sorted((p for p in root.rglob("*") if p.suffix.lower() in [".jpg", ".jpeg", ".png", ".tif", ".tiff"]), key=natural_key)
    for p in files:
        if "_anno_frame_" in p.name and p.with_name(p.name.replace("_anno_frame_", "_frame_")).exists():
            continue
        relative = p.relative_to(root)
        # Never combine camera folders into one timeline or tracker.
        key = str(relative.parent).replace("\\", "/") if relative.parent != Path(".") else "sequence"
        camera_prefix = re.match(r"(cam\d+)_", p.name, re.IGNORECASE)
        if camera_prefix:
            key = (key+"/" if key != "sequence" else "")+camera_prefix[1].lower()
        number = len(groups[key])
        # Recognize original TrackBox ..._frame_<unix>.tiff; otherwise use an
        # explicit experimental interval rather than file creation timestamps.
        stamp = re.search(r"(?:frame_|_)(1\d{9}(?:\.\d+)?)$", p.stem)
        ts = float(stamp[1]) if stamp else number*interval
        groups[key].append(InputFrame(p, ts, number, key, {"timestamp_basis": "filename_unix" if stamp else "configured_interval", "original_path": str(p)}))
    return dict(groups)


class AnalysisJob:
    def __init__(self, config: AppConfig, root: Path, status=lambda kind, data: None, preview=lambda stream, image: None,
                 pipeline_factory=YoloPipeline):
        self.config = AppConfig.from_dict(config.to_dict())
        self.root = root.resolve()
        self.status, self.preview = status, preview
        self.stop_event = threading.Event()
        self.pipeline_factory = pipeline_factory
        self.store = None

    def stop(self):
        self.stop_event.set()

    def run(self, infer=True):
        self.config.validate()
        # Discover inputs before creating the output directory inside the tree.
        groups = discover_sequences(self.root, self.config.analysis.sequence_interval_s,
                                    self.config.analysis.input_images == "raw", not self.config.analysis.use_capture_geometry)
        if not groups:
            raise ValueError("No readable image sequences found")
        self.store = RunStore(self.config, "analysis" if infer else "timelapse_video")
        store = self.store
        pipeline = None
        videos = {}
        count = 0
        error = None
        try:
            if infer:
                self.status("loading_models", {})
                pipeline = self.pipeline_factory(self.config.analysis)
                atomic_json(store.path/"models.json", pipeline.provenance)
            cfg = self.config
            total = sum(len(xs) for xs in groups.values())
            for group, inputs in groups.items():
                if self.stop_event.is_set():
                    break
                transforms = {}
                trackers = {}
                pending = []
                previous = None
                origin = inputs[0].timestamp

                def prepare(source):
                    image = read_image(source.path)
                    if source.processed:
                        mask_path = source.metadata.get("valid_mask_path")
                        mask = read_image(mask_path)[:, :, 0] if mask_path and Path(mask_path).is_file() else None
                        if mask is None and source.metadata.get("valid_region_bits"):
                            mask = decode_mask(source.metadata["valid_region_bits"]).astype(np.uint8)*255
                        tiles = [(source.stream, image, source.metadata, mask)]
                    elif cfg.analysis.use_capture_geometry and source.metadata.get("sensor_to_tile") and source.metadata.get("tile_size"):
                        # Replay the actual recorded per-frame transform, including
                        # stabilization, instead of estimating a new transform.
                        matrix = np.asarray(source.metadata["sensor_to_tile"], np.float64)
                        size = tuple(source.metadata["tile_size"])
                        tile = cv2.warpPerspective(image, matrix, size)
                        mask = cv2.warpPerspective(np.full(image.shape[:2], 255, np.uint8), matrix, size, flags=cv2.INTER_NEAREST)
                        path = source.metadata.get("valid_mask_path")
                        if path and Path(path).is_file():
                            mask = cv2.bitwise_and(mask, read_image(path)[:, :, 0])
                        elif source.metadata.get("valid_region_bits"):
                            mask = cv2.bitwise_and(mask, decode_mask(source.metadata["valid_region_bits"]).astype(np.uint8)*255)
                        tile[mask == 0] = 0
                        tiles = [(source.stream, tile, source.metadata, mask)]
                    else:
                        # A raw capture record already refers to a particular ROI.
                        # Recompute using the original saved capture geometry when
                        # available; otherwise apply the current geometry to imports.
                        camera = source.metadata.get("camera", group.split("/")[0])
                        if camera not in transforms:
                            geo = geometry_override(cfg.geometry, cfg.geometry.per_camera.get(camera, {}))
                            transforms[camera] = GeometryProcessor(geo)
                        split, report = transforms[camera].apply(image)
                        desired_roi = source.metadata.get("roi")
                        tiles = [(group+"/"+t.key if not desired_roi else source.stream, t.image,
                                  {**source.metadata, "sensor_to_tile": t.sensor_to_tile, "bounds": t.bounds, "geometry": report}, t.valid_mask)
                                 for t in split if not desired_roi or t.key == desired_roi]
                    prepared = {}
                    for stream, tile, metadata, mask in tiles:
                        predictions = pipeline.infer(tile, mask) if pipeline else []
                        metadata = {**metadata, "input_path": str(source.path), "input_timestamp": source.timestamp,
                                    "model_evidence": deepcopy(getattr(pipeline, "last_evidence", []))}
                        if mask is not None:
                            metadata["valid_region_bits"] = encode_mask(mask > 0)
                        metadata["tile_size"] = [tile.shape[1], tile.shape[0]]
                        prepared[stream] = (tile, metadata, predictions)
                    return prepared

                def consume(source, current, before, after):
                    nonlocal count
                    for stream, (image, metadata, per_model) in current.items():
                        if cfg.analysis.consensus == "temporal" and pipeline:
                            accepted_models = []
                            for mi in range(len(cfg.analysis.models)):
                                prev = before.get(stream, (None, None, [[] for _ in per_model]))[2][mi] if before else []
                                nex = after.get(stream, (None, None, [[] for _ in per_model]))[2][mi] if after else []
                                views = [[d for d in view if d.get("positive", True)] for view in [prev, per_model[mi], nex]]
                                accepted = consensus(views, cfg.analysis.consensus_iou, 1)
                                for d in accepted:
                                    d["positive"] = True
                                accepted += [d for d in per_model[mi] if not d.get("positive", True)]
                                accepted_models.append(accepted)
                            per_model = accepted_models
                            metadata = {**metadata, "temporal_neighbors": [before is not None, after is not None]}
                            metadata["temporal_neighbor_frames"] = [
                                neighbor.get(stream, (None, {}, []))[1].get("input_path") if neighbor else None
                                for neighbor in [before, after]]
                        ds = fuse_models(per_model, cfg.analysis.fusion_iou) if pipeline else []
                        for d in ds:
                            d["positive"] = any(m.get("positive", True) for m in d.get("measurements", [d]))
                            accepted = [m for m in d.get("measurements", [d]) if m.get("positive", True)]
                            if accepted:
                                d["bbox"] = accepted[0]["bbox"]
                                d["confidence"] = accepted[0]["confidence"]
                                for attr in ["polygon", "area_px2", "keypoints", "mask_bits"]:
                                    d[attr] = next((m[attr] for m in accepted if m.get(attr) is not None), None)
                        # Preserve both processed coordinates and sensor coordinates.
                        matrix = metadata.get("sensor_to_tile")
                        inverse = np.linalg.inv(matrix) if matrix is not None else None
                        for d in ds:
                            if inverse is not None:
                                x1, y1, x2, y2 = d["bbox"]
                                d["sensor_corners"] = transform_points([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], inverse)
                            if cfg.analysis.pixel_size_um is not None and d.get("area_px2") is not None:
                                # Calibration applies to processed image pixels.
                                d["area_um2"] = d["area_px2"]*cfg.analysis.pixel_size_um**2
                        if stream not in trackers:
                            trackers[stream] = StreamTracker(cfg.analysis)
                        tracks = trackers[stream].update(ds, source.timestamp) if pipeline else []
                        # Save a lossless base for future overlay editing, without rerunning YOLO.
                        tile_path = Path("base")/stream/f"{source.sequence:010d}.png"
                        write_image(store.path/tile_path, image)
                        meta = {**metadata, "input_path": str(source.path), "input_sha256": file_hash(source.path),
                                "elapsed_s": source.timestamp-origin, "coordinates": "processed tile pixels"}
                        store.frame(stream, source.sequence, source.timestamp, source.timestamp,
                                    metadata.get("original_path") or str(source.path), tile_path.as_posix(), meta, ds, tracks)
                        annotated = render(image, [d for d in ds if d.get("positive", True)], tracks, source.timestamp-origin, cfg.analysis)
                        if cfg.storage.annotated_video or not infer:
                            if stream not in videos:
                                video_cfg = AppConfig.from_dict(cfg.to_dict())
                                video_cfg.capture.mode = "timelapse"
                                videos[stream] = VideoSink(store.path/"videos"/stream/"analysis", video_cfg, (image.shape[1], image.shape[0]))
                            videos[stream].write(annotated, source.timestamp)
                        self.preview(stream, annotated)
                    count += 1
                    self.status("analysis_progress", {"current": count, "total": total, "stream": group})

                # A rolling three-frame window bounds image/inference memory.
                for source in inputs:
                    if self.stop_event.is_set():
                        break
                    prepared = prepare(source)
                    if pending:
                        prev_source, prev_prepared = pending[0]
                        consume(prev_source, prev_prepared, previous, prepared)
                        previous = prev_prepared
                        pending.clear()
                    pending.append((source, prepared))
                if pending and not self.stop_event.is_set():
                    source, prepared = pending[0]
                    consume(source, prepared, previous, None)
            with store.lock:
                recorded = store.db.execute("SELECT COUNT(*) FROM frames").fetchone()[0]
            if recorded == 0 and not self.stop_event.is_set():
                raise ValueError("No output ROI frames were produced. Check image source and geometry settings.")
        except Exception as caught:
            error = caught
            store.event("analysis_error", {"error": str(caught)})
        finally:
            summaries = {}
            for stream, video in videos.items():
                try:
                    summaries[stream] = video.close()
                except Exception as caught:
                    error = error or caught
            store.finish("error" if error else ("cancelled" if self.stop_event.is_set() else "completed"),
                         {"input_frames": count, "input_root": str(self.root), "videos": summaries, "error": str(error) if error else None})
            export_excel(store.path)
        if error:
            raise error
        return store.path


def redraw_cache(run: Path, config: AppConfig, stop: threading.Event | None = None, status=lambda kind, data: None) -> Path:
    output = run/("redraw_"+time.strftime("%Y%m%d_%H%M%S"))
    output.mkdir(exist_ok=False)
    config.save(output/"settings.json", redact=True)
    videos = {}
    with sqlite3.connect(run/"records.sqlite") as db:
        rows = db.execute("SELECT stream,timestamp,processed_path,metadata,detections,tracks FROM frames ORDER BY stream,timestamp,id")
        try:
            for count, (stream, timestamp, path, meta, ds, tracks) in enumerate(rows, 1):
                if stop and stop.is_set():
                    break
                metadata = json.loads(meta)
                detections, track_records = json.loads(ds), json.loads(tracks)
                if config.analysis.redraw_original:
                    original = metadata.get("original_path")
                    if not original:
                        raise ValueError("Original raw image is unavailable for this run")
                    image = read_image(original)
                    inverse = np.linalg.inv(metadata.get("sensor_to_tile", np.eye(3)))
                    detections, track_records = project_overlays(detections, track_records, inverse, (image.shape[1], image.shape[0]))
                else:
                    image = read_image(run/path)
                frame = render(image, [d for d in detections if d.get("positive", True)], track_records, metadata.get("elapsed_s", timestamp), config.analysis)
                if stream not in videos:
                    cfg = AppConfig.from_dict(config.to_dict())
                    cfg.capture.mode = "timelapse"
                    videos[stream] = VideoSink(output/stream/"redraw", cfg, (image.shape[1], image.shape[0]))
                videos[stream].write(frame, timestamp)
                status("redraw_progress", {"frames": count})
        finally:
            for sink in videos.values():
                sink.close()
    return output


def project_overlays(detections, tracks, inverse, output_size=None):
    ds, ts = deepcopy(detections), deepcopy(tracks)
    def box_in_sensor(box):
        x1, y1, x2, y2 = box
        p = np.array(transform_points([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], inverse))
        return [float(p[:, 0].min()), float(p[:, 1].min()), float(p[:, 0].max()), float(p[:, 1].max())]
    for d in ds:
        # Full-resolution mask remains stored in tile space; projected polygons
        # are used on originals to avoid silently interpreting its pixel grid.
        if d.get("mask_bits") and output_size is not None:
            mask = cv2.warpPerspective(decode_mask(d["mask_bits"]).astype(np.uint8), np.asarray(inverse, np.float64), output_size, flags=cv2.INTER_NEAREST)
            d["mask_bits"] = encode_mask(mask > 0)
        else:
            d.pop("mask_bits", None)
        d["bbox"] = box_in_sensor(d["bbox"])
        if d.get("polygon"):
            d["polygon"] = transform_points(d["polygon"], inverse)
        if d.get("keypoints"):
            p = np.array(d["keypoints"], float)
            p[:, :2] = transform_points(p[:, :2], inverse)
            d["keypoints"] = p.tolist()
    for t in ts:
        center = [(t["bbox"][0]+t["bbox"][2])/2, (t["bbox"][1]+t["bbox"][3])/2]
        velocity_tip = np.array(center)+t["velocity_px_s"]
        points = np.array(transform_points([center, velocity_tip], inverse))
        t["velocity_px_s"] = (points[1]-points[0]).tolist()
        t["bbox"] = box_in_sensor(t["bbox"])
        t["trail"] = transform_points(t["trail"], inverse) if t["trail"] else []
    return ds, ts
