from __future__ import annotations

from collections import deque
import numpy as np
from scipy.optimize import linear_sum_assignment

from .config import AnalysisConfig
from .vendor.tracker import create_tracker


def iou(a, b) -> float:
    left, top = max(a[0], b[0]), max(a[1], b[1])
    right, bottom = min(a[2], b[2]), min(a[3], b[3])
    intersection = max(0, right-left)*max(0, bottom-top)
    union = max(0, a[2]-a[0])*max(0, a[3]-a[1])+max(0, b[2]-b[0])*max(0, b[3]-b[1])-intersection
    return intersection/union if union > 0 else 0.0


class StreamTracker:
    """One tracker per camera/target/ROI; semantic IDs never mix model taxonomies."""
    def __init__(self, config: AnalysisConfig):
        similarities = {}
        # Numeric semantic IDs are assigned in label arrival order. The user map
        # uses labels; translate it when labels become known.
        options = dict(max_age=config.max_age, min_hits=config.min_hits, iou_threshold=config.iou_threshold,
                       max_trail_len=config.trail_length, gating_threshold=config.gating_px,
                       area_weight=config.area_weight, class_similarity_map=similarities)
        if config.tracker == "bytetrack":
            options.update(high_thresh=config.high_threshold, low_thresh=config.low_threshold)
        self.tracker = create_tracker(config.tracker, options)
        # The upstream history grows without bound. Records are streamed to SQLite.
        self.tracker.tracking_history = deque(maxlen=1)
        self.labels = {}
        self.config = config
        self.previous = None
        self.sequence = 0
        original = self.tracker._compute_cost_matrix
        def gated(tracks, detections):
            cost = original(tracks, detections)
            # Gate BEFORE assignment; invalid edges cannot displace valid matches.
            cost[cost > 1-config.iou_threshold] = 1e6
            return cost
        self.tracker._compute_cost_matrix = gated

    def update(self, detections: list[dict], timestamp: float) -> list[dict]:
        if self.previous is not None and timestamp <= self.previous:
            raise ValueError("Tracking timestamps must be strictly increasing within a stream")
        dt = timestamp-self.previous if self.previous is not None else self.config.sequence_interval_s
        self.previous = timestamp
        self.sequence += 1
        normalized = []
        for d in detections:
            label = d["label"]
            if label not in self.labels:
                self.labels[label] = len(self.labels)
            box = d["bbox"]
            if np.isfinite(box).all() and box[2] > box[0] and box[3] > box[1]:
                normalized.append({"bbox": box, "class_id": self.labels[label], "confidence": d["confidence"],
                                   "mask": d.get("polygon"), "keypoints": d.get("keypoints")})
        similarities = {}
        for key, value in self.config.class_similarity.items():
            a, b = key.split("|", 1)
            if a in self.labels and b in self.labels:
                similarities[(self.labels[a], self.labels[b])] = float(value)
        self.tracker.class_similarity_map = similarities
        tracks = self.tracker.update(normalized, self.sequence, dt=dt)
        inverse = {v: k for k, v in self.labels.items()}
        records = []
        for track in tracks:
            if track.time_since_update:
                # Predictions are not observations and must not be counted as positive.
                continue
            if hasattr(track, "custom_kf"):
                velocity = [float(track.state[4]), float(track.state[5])]
            else:
                velocity = [float(track.state[4]), float(track.state[5])]
            records.append({"id": track.id, "label": inverse[track.class_id], "bbox": track.bbox.tolist(),
                            "confidence": float(track.confidence), "age": track.age, "hits": track.hits,
                            "velocity_px_s": velocity, "trail": [list(map(float, p)) for p in track.trail],
                            "predicted_center": list(track.pred_pos) if track.pred_pos else None})
        # Attach IDs one-to-one to measurements for exports/hover labels.
        if records and detections:
            cost = np.ones((len(records), len(detections)))*1e6
            for i, track in enumerate(records):
                for j, det in enumerate(detections):
                    if track["label"] == det["label"]:
                        cost[i, j] = 1-iou(track["bbox"], det["bbox"])
            for i, j in zip(*linear_sum_assignment(cost)):
                if cost[i, j] <= 1-self.config.iou_threshold:
                    detections[j]["track_id"] = records[i]["id"]
                    records[i]["positive"] = detections[j].get("positive", True)
        return records
