from __future__ import annotations

from dataclasses import dataclass
import cv2
import numpy as np

from .config import GeometryConfig


@dataclass
class Tile:
    key: str
    image: np.ndarray
    # Maps sensor pixel coordinates to this output tile (homogeneous 3x3).
    sensor_to_tile: list[list[float]]
    bounds: list[int]
    valid_mask: np.ndarray


class GeometryProcessor:
    def __init__(self, config: GeometryConfig):
        self.config = config
        self.reference = None

    def apply(self, raw: np.ndarray) -> tuple[list[Tile], dict]:
        cfg = self.config
        h, w = raw.shape[:2]
        transform = np.eye(3, dtype=np.float64)
        image = raw
        report = {"stabilization_applied": False, "phase_response": None, "shift_px": [0.0, 0.0]}
        # Fixed first-frame reference avoids indefinite cumulative drift.
        if cfg.stabilize:
            scale = min(1, 640/w)
            gray = cv2.resize(cv2.cvtColor(raw, cv2.COLOR_BGR2GRAY), None, fx=scale, fy=scale).astype(np.float32)
            if self.reference is None or self.reference.shape != gray.shape:
                self.reference = gray.copy()
            else:
                window = cv2.createHanningWindow((gray.shape[1], gray.shape[0]), cv2.CV_32F)
                shift, response = cv2.phaseCorrelate(self.reference, gray, window)
                dx, dy = shift[0]/scale, shift[1]/scale
                report.update(phase_response=float(response), shift_px=[float(dx), float(dy)])
                if np.isfinite([dx, dy]).all() and response >= cfg.min_phase_response and np.hypot(dx, dy) <= cfg.max_shift_px:
                    transform[0, 2], transform[1, 2] = -dx, -dy
                    report["stabilization_applied"] = True
                else:
                    report["stabilization_rejected"] = True
        out_w, out_h = w, h
        if cfg.perspective:
            source = np.asarray(cfg.perspective, np.float32)*[w-1, h-1]
            if cfg.output_size:
                out_w, out_h = cfg.output_size
            else:
                out_w = max(2, round(max(np.linalg.norm(source[1]-source[0]), np.linalg.norm(source[2]-source[3]))))
                out_h = max(2, round(max(np.linalg.norm(source[3]-source[0]), np.linalg.norm(source[2]-source[1]))))
            dest = np.float32([[0, 0], [out_w-1, 0], [out_w-1, out_h-1], [0, out_h-1]])
            transform = cv2.getPerspectiveTransform(source.astype(np.float32), dest) @ transform
        valid = np.full((h, w), 255, np.uint8)
        if cfg.stabilize or cfg.perspective:
            image = cv2.warpPerspective(raw, transform, (out_w, out_h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
            valid = cv2.warpPerspective(valid, transform, (out_w, out_h), flags=cv2.INTER_NEAREST)
        tiles = []
        for roi in cfg.rois:
            x, y, rw, rh = roi.rect
            x0, y0 = round(x*out_w), round(y*out_h)
            x1, y1 = min(out_w, round((x+rw)*out_w)), min(out_h, round((y+rh)*out_h))
            # Integer edges partition every pixel exactly once, including remainders.
            xs, ys = np.linspace(x0, x1, roi.cols+1, dtype=int), np.linspace(y0, y1, roi.rows+1, dtype=int)
            if np.min(np.diff(xs)) < 1 or np.min(np.diff(ys)) < 1:
                raise ValueError("ROI grid has more cells than available pixels")
            for row in range(roi.rows):
                for col in range(roi.cols):
                    left, top, right, bottom = xs[col], ys[row], xs[col+1], ys[row+1]
                    crop = image[top:bottom, left:right].copy()
                    mask = valid[top:bottom, left:right].copy()
                    if roi.shape == "ellipse":
                        ellipse = np.zeros(mask.shape, np.uint8)
                        ch, cw = mask.shape
                        cv2.ellipse(ellipse, ((cw-1)//2, (ch-1)//2), (max(1, (cw-1)//2), max(1, (ch-1)//2)), 0, 0, 360, 255, -1)
                        mask = cv2.bitwise_and(mask, ellipse)
                        crop[mask == 0] = 0
                    local = np.array([[1, 0, -left], [0, 1, -top], [0, 0, 1]], np.float64) @ transform
                    key = roi.name if roi.rows*roi.cols == 1 else f"{roi.name}_r{row+1:02d}_c{col+1:02d}"
                    tiles.append(Tile(key, crop, local.tolist(), [int(left), int(top), int(right), int(bottom)], mask))
        report["sensor_to_rectified"] = transform.tolist()
        return tiles, report


def transform_points(points, matrix):
    p = np.asarray(points, np.float64).reshape(-1, 2)
    homogeneous = np.column_stack([p, np.ones(len(p))]) @ np.asarray(matrix).T
    return (homogeneous[:, :2]/homogeneous[:, 2:3]).tolist()
