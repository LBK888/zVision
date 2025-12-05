"""
多相機時間序列影像分析工具 (Multi-Camera Timelapse Analyzer)
支援 YOLO 模型分析、區域裁切、影片生成及報表輸出

version: 1.0.0 2025/11/27
author: Bo-Kai Liao, GitHub: lbk888
"""

import sys
import os
import re
from pathlib import Path
from datetime import datetime
from copy import deepcopy
import cv2
import numpy as np
import pandas as pd
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QPushButton, QLabel, QFileDialog, 
                             QListWidget, QSpinBox, QColorDialog, QProgressBar,
                             QGroupBox, QGridLayout, QCheckBox, QScrollArea,
                             QMessageBox, QComboBox, QDoubleSpinBox, QAbstractItemView,
                             QDialog, QDialogButtonBox, QFormLayout, QSlider,
                             QTableWidget, QTableWidgetItem, QHeaderView)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QRect, QTimer, QPoint
from PyQt5.QtGui import QImage, QPixmap, QPainter, QPen, QColor
from ultralytics import YOLO
from yolo_tracker_v2 import YOLOTracker, create_tracker
import traceback


# tracker 初始化 (Global default, will be replaced in thread)
tracker = create_tracker('ukf', {
    'max_age': 30,
    'min_hits': 3,
    'iou_threshold': 0.3
})


class ClassSimilarityDialog(QDialog):
    def __init__(self, current_map=None, model_classes=None, parent=None):
        super().__init__(parent)
        self.setWindowTitle("類別相似度設定")
        self.resize(600, 500)
        self.similarity_map = dict(current_map) if current_map else {}
        # model_classes: dict {id: name}
        self.model_classes = model_classes if model_classes else {}
        self.sorted_classes = sorted(self.model_classes.items(), key=lambda x: x[0])
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout(self)
        
        # 說明
        info_label = QLabel("設定不同類別之間的相似度 (0.0 ~ 1.0)。\n若兩類別相似度高，Tracker 會更容易將其視為同一物體。")
        info_label.setWordWrap(True)
        layout.addWidget(info_label)

        # 快速設定按鈕
        btn_group = QGroupBox("快速設定")
        btn_layout = QHBoxLayout()
        
        btn_strict = QPushButton("嚴格匹配 (清空)")
        btn_strict.setToolTip("清空所有相似度設定，僅相同類別ID可匹配")
        btn_strict.clicked.connect(self.set_strict)
        
        btn_loose = QPushButton("全部寬鬆 (All 1.0)")
        btn_loose.setToolTip("將所有類別組合設為 1.0 (視為相同)")
        btn_loose.clicked.connect(self.set_loose)
        
        btn_layout.addWidget(btn_strict)
        btn_layout.addWidget(btn_loose)
        btn_group.setLayout(btn_layout)
        layout.addWidget(btn_group)
        
        # 新增規則區域
        add_group = QGroupBox("新增/修改規則")
        add_layout = QGridLayout()
        
        add_layout.addWidget(QLabel("類別 A:"), 0, 0)
        self.combo_cls_a = QComboBox()
        self.populate_combo(self.combo_cls_a)
        add_layout.addWidget(self.combo_cls_a, 0, 1)
        
        add_layout.addWidget(QLabel("類別 B:"), 0, 2)
        self.combo_cls_b = QComboBox()
        self.populate_combo(self.combo_cls_b)
        add_layout.addWidget(self.combo_cls_b, 0, 3)
        
        add_layout.addWidget(QLabel("相似度:"), 1, 0)
        self.spin_score = QDoubleSpinBox()
        self.spin_score.setRange(0.0, 1.0)
        self.spin_score.setSingleStep(0.1)
        self.spin_score.setValue(0.5)
        add_layout.addWidget(self.spin_score, 1, 1)
        
        btn_add = QPushButton("加入/更新")
        btn_add.clicked.connect(self.add_rule)
        add_layout.addWidget(btn_add, 1, 3)
        
        add_group.setLayout(add_layout)
        layout.addWidget(add_group)
        
        # 規則列表 (Table)
        layout.addWidget(QLabel("目前規則列表:"))
        self.table = QTableWidget()
        self.table.setColumnCount(4)
        self.table.setHorizontalHeaderLabels(["類別 A", "類別 B", "相似度", "操作"])
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        layout.addWidget(self.table)
        
        # 載入目前規則
        self.refresh_table()
        
        # 底部按鈕
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def populate_combo(self, combo):
        combo.clear()
        if not self.model_classes:
            combo.addItem("未知 (請先選擇模型)", -1)
            return
        for cls_id, name in self.sorted_classes:
            combo.addItem(f"{name} ({cls_id})", cls_id)

    def set_strict(self):
        self.similarity_map = {}
        self.refresh_table()

    def set_loose(self):
        reply = QMessageBox.question(self, "確認", 
                                   "這將會把所有類別組合的相似度設為 1.0，\n"
                                   "若類別數量較多會產生大量規則，確定要執行嗎？",
                                   QMessageBox.Yes | QMessageBox.No)
        if reply == QMessageBox.Yes:
            self.similarity_map = {}
            ids = [x[0] for x in self.sorted_classes]
            for i in ids:
                for j in ids:
                    if i != j:
                        # 為了避免重複，只存 key 小的在前? 
                        # Tracker 通常查 (a,b) 或 (b,a)，這裡我們存雙向或單向皆可，
                        # 視 Tracker 實作而定。假設 Tracker 查 (min, max)。
                        # 為了保險，我們存 (min, max)
                        k = tuple(sorted((i, j)))
                        self.similarity_map[k] = 1.0
            self.refresh_table()

    def add_rule(self):
        if not self.model_classes:
            return
            
        id_a = self.combo_cls_a.currentData()
        id_b = self.combo_cls_b.currentData()
        score = self.spin_score.value()
        
        if id_a == id_b:
            QMessageBox.warning(self, "無效", "類別 A 與 B 不能相同")
            return
            
        # 存為 tuple(sorted((id_a, id_b))) 以保持一致性
        key = tuple(sorted((id_a, id_b)))
        self.similarity_map[key] = score
        self.refresh_table()

    def delete_rule(self, key):
        if key in self.similarity_map:
            del self.similarity_map[key]
            self.refresh_table()

    def refresh_table(self):
        self.table.setRowCount(0)
        for (id_a, id_b), score in self.similarity_map.items():
            row = self.table.rowCount()
            self.table.insertRow(row)
            
            name_a = self.model_classes.get(id_a, f"ID {id_a}")
            name_b = self.model_classes.get(id_b, f"ID {id_b}")
            
            self.table.setItem(row, 0, QTableWidgetItem(name_a))
            self.table.setItem(row, 1, QTableWidgetItem(name_b))
            self.table.setItem(row, 2, QTableWidgetItem(str(score)))
            
            btn_del = QPushButton("刪除")
            # 使用 closure 綁定 key
            btn_del.clicked.connect(lambda checked, k=(id_a, id_b): self.delete_rule(k))
            self.table.setCellWidget(row, 3, btn_del)

    def get_map(self):
        return self.similarity_map

DEFAULT_OVERLAY_CONFIG = {
    'bbox': True,
    'probs': True,
    'masks': False,
    'kpt_line': False,
    'font_scale': 0.6,
    'track_id': True,
    'track_age': False,
    'velocity': False,
    'track_confidence': False,
    'trail': True,
    'pred_position': False
}

DETECTION_COLORS = [
    (255, 99, 71),
    (135, 206, 235),
    (60, 179, 113),
    (238, 130, 238),
    (255, 215, 0),
    (255, 140, 0),
    (0, 191, 255),
    (199, 21, 133),
    (154, 205, 50),
    (106, 90, 205)
]


def _tensor_to_numpy(data):
    if data is None:
        return None
    value = data
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    return np.array(value)


def imread_safe(path):
    """
    Safe image read for Windows with non-ASCII paths.
    """
    try:
        stream = np.fromfile(path, dtype=np.uint8)
        img = cv2.imdecode(stream, cv2.IMREAD_COLOR)
        return img
    except Exception:
        return None


def add_timestamp(img, text, config):
    """在圖片上添加時間戳記"""
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = config['size']
    color = config['color']  # BGR
    thickness = max(1, int(font_scale * 2))
    
    # 計算文字大小
    (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, thickness)
    
    # 在右上角繪製文字背景
    padding = 10
    x = img.shape[1] - text_width - padding
    y = padding + text_height
    
    # 半透明背景
    overlay = img.copy()
    cv2.rectangle(overlay, 
                 (x - 5, y - text_height - 5), 
                 (x + text_width + 5, y + baseline + 5), 
                 (0, 0, 0), -1)
    img = cv2.addWeighted(overlay, 0.6, img, 0.4, 0)
    
    # 繪製文字
    cv2.putText(img, text, (x, y), font, font_scale, color, thickness, cv2.LINE_AA)
    
    return img


def convert_result_to_detections(result, model_name, crop_offset=(0, 0)):
    detections = []
    boxes = getattr(result, "boxes", None)
    if boxes is None or len(boxes) == 0:
        return detections
    
    xyxy = _tensor_to_numpy(boxes.xyxy)
    confs = _tensor_to_numpy(boxes.conf)
    cls_ids = _tensor_to_numpy(boxes.cls)
    names = getattr(result, "names", None)
    
    mask_polys = None
    if hasattr(result, "masks") and result.masks is not None:
        mask_polys = result.masks.xy
    
    keypoints_xy = None
    keypoints_conf = None
    if hasattr(result, "keypoints") and result.keypoints is not None:
        keypoints_xy = _tensor_to_numpy(result.keypoints.xy)
        if hasattr(result.keypoints, "conf") and result.keypoints.conf is not None:
            keypoints_conf = _tensor_to_numpy(result.keypoints.conf)
    
    for idx in range(len(xyxy)):
        bbox = xyxy[idx].tolist()
        global_bbox = [
            bbox[0] + crop_offset[0],
            bbox[1] + crop_offset[1],
            bbox[2] + crop_offset[0],
            bbox[3] + crop_offset[1]
        ]
        cls_id = int(cls_ids[idx]) if cls_ids is not None else 0
        class_name = names[cls_id] if names and cls_id in names else f"class_{cls_id}"
        det = {
            'bbox': bbox,
            'bbox_global': global_bbox,
            'class_id': cls_id,
            'class_name': class_name,
            'confidence': float(confs[idx]) if confs is not None else 0.0,
            'probability': float(confs[idx]) if confs is not None else 0.0,
            'model_name': model_name
        }
        
        if mask_polys is not None and idx < len(mask_polys):
            det['masks'] = [poly.tolist() for poly in mask_polys[idx]]
        
        if keypoints_xy is not None and idx < len(keypoints_xy):
            kp_list = []
            for kp_idx, kp in enumerate(keypoints_xy[idx]):
                kp_entry = {
                    'x': float(kp[0]),
                    'y': float(kp[1]),
                    'conf': float(keypoints_conf[idx][kp_idx]) if keypoints_conf is not None else None
                }
                kp_list.append(kp_entry)
            det['keypoints'] = kp_list
        
        detections.append(det)
    return detections


def serialize_tracked_object(obj):
    bbox = obj.bbox.tolist() if isinstance(obj.bbox, np.ndarray) else list(obj.bbox)
    trail_points = [(float(pt[0]), float(pt[1])) for pt in obj.trail] if obj.trail else []
    velocity = (float(obj.state[4]), float(obj.state[5])) if obj.state is not None and len(obj.state) >= 6 else (0.0, 0.0)
    pred_pos = obj.pred_pos if obj.pred_pos else (0.0, 0.0)
    return {
        'id': obj.id,
        'bbox': bbox,
        'confidence': float(obj.confidence),
        'age': obj.age,
        'time_since_update': obj.time_since_update,
        'velocity': velocity,
        'trail': trail_points,
        'class_id': obj.class_id,
        'pred_pos': pred_pos
    }


class AnnotationRenderer:
    def __init__(self, overlay_config):
        self.overlay_config = overlay_config or DEFAULT_OVERLAY_CONFIG.copy()
    
    def render(self, frame, detections, tracking_data, track_color_fn=None, highlight_track_id=None):
        cfg = self.overlay_config
        canvas = frame.copy()
        track_map = {track['id']: track for track in tracking_data} if tracking_data else {}
        drawn_track_ids = set()
        font_scale = cfg.get('font_scale', 0.6)
        thickness = max(1, int(font_scale * 2))
        
        # If highlighting, we might want to dim others or just draw the highlight on top.
        # User request: "hover... show object+track... show path with transparency... start/end frame"
        # Implies we should draw the highlighted track specially.
        
        # Draw normal detections/tracks first
        for det in detections or []:
            bbox = det.get('bbox')
            track_id = det.get('track_id')
            
            # If highlighting, maybe skip drawing others or draw them normally?
            # Let's draw normally, and draw highlight on top.
            
            if cfg.get('bbox', True) and bbox:
                x1, y1, x2, y2 = [int(round(v)) for v in bbox]
                color = self._get_detection_color(det.get('class_id', 0))
                cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 2)
                
                label_parts = [det.get('class_name', 'object')]
                if cfg.get('probs', True):
                    label_parts.append(f"{det.get('confidence', 0.0):.2f}")
                
                track_info = track_map.get(track_id)
                track_label = self._build_track_label(track_info, cfg)
                if track_label:
                    drawn_track_ids.add(track_id)
                    label_parts.append(track_label)
                
                label_text = " | ".join(label_parts)
                cv2.putText(
                    canvas,
                    label_text,
                    (x1, max(20, y1 - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    font_scale,
                    color,
                    thickness,
                    cv2.LINE_AA
                )
            
            # Masks, Keypoints... (Keep existing logic)
            if cfg.get('masks') and det.get('masks'):
                mask_overlay = canvas.copy()
                color = self._get_detection_color(det.get('class_id', 0))
                for poly in det['masks']:
                    pts = np.array(poly, dtype=np.int32)
                    if pts.ndim == 1: pts = pts.reshape(-1, 2)
                    if len(pts) >= 3:
                        pts = pts.reshape(-1, 1, 2)
                        cv2.fillPoly(mask_overlay, [pts], color)
                canvas = cv2.addWeighted(mask_overlay, 0.35, canvas, 0.65, 0)
                
            if det.get('keypoints'):
                kp_points = []
                for kp in det['keypoints']:
                    kp_pt = (int(round(kp['x'])), int(round(kp['y'])))
                    kp_points.append(kp_pt)
                    cv2.circle(canvas, kp_pt, 2, (0, 255, 255), -1)
                if cfg.get('kpt_line') and len(kp_points) > 1:
                    for start, end in zip(kp_points[:-1], kp_points[1:]):
                        cv2.line(canvas, start, end, (0, 255, 255), 1)

        # Draw Trails (Normal)
        if tracking_data and cfg.get('trail'):
            for track in tracking_data:
                if track.get('id') == highlight_track_id: continue # Draw later
                if track.get('time_since_update', 0) > 0: continue
                
                trail_pts = track.get('trail') or []
                if len(trail_pts) < 2: continue
                
                color = self._get_track_color(track['id'], track_color_fn)
                converted = [(int(round(x)), int(round(y))) for x, y in trail_pts]
                for start, end in zip(converted[:-1], converted[1:]):
                    cv2.line(canvas, start, end, color, 2)

        # Draw Highlighted Track
        if highlight_track_id is not None and track_map.get(highlight_track_id):
            track = track_map[highlight_track_id]
            trail_pts = track.get('trail') or []
            color = self._get_track_color(track['id'], track_color_fn)
            
            # Draw gradient trail
            if len(trail_pts) >= 2:
                converted = [(int(round(x)), int(round(y))) for x, y in trail_pts]
                n_pts = len(converted)
                for i in range(n_pts - 1):
                    start = converted[i]
                    end = converted[i+1]
                    # Alpha: newest (end) = 0.5, oldest (start) = 0.0
                    # i=0 (oldest) → alpha=0.0, i=n-2 (newest) → alpha~0.5
                    alpha = 0.5 * (i + 1) / n_pts
                    
                    overlay = canvas.copy()
                    cv2.line(overlay, start, end, color, 4) # Thicker line
                    canvas = cv2.addWeighted(overlay, alpha, canvas, 1 - alpha, 0)

            # Draw Start/End Frame info
            # We don't have start/end frame in track info directly here unless we store it.
            # But we can show "Track ID: X" and maybe current frame info.
            # The user asked for "Start and End frame". 
            # `TrackedObject` has `age`. Start frame = Current Frame - Age + 1 (approx if continuous).
            # Or we need to store start_frame in TrackedObject.
            # `yolo_tracker_v2.py` TrackedObject doesn't have start_frame explicitly, but we can infer or add it.
            # For now, I will show Age.
            
            bbox = track.get('bbox')
            if bbox:
                x1, y1, x2, y2 = [int(round(v)) for v in bbox]
                cv2.rectangle(canvas, (x1, y1), (x2, y2), (255, 255, 255), 3) # Highlight box
                
                # Calculate Start/End Frame
                # Assuming track['frame_id'] is the last update frame.
                # If we are rendering a specific frame, 'current_frame' is implied by the track state at that time.
                # But 'track' here comes from 'tracking_data' which is the snapshot at that frame.
                # So track['frame_id'] should be the current frame index (or close to it if missed).
                end_frame = track.get('frame_id', 0)
                age = track.get('age', 0)
                start_frame = end_frame - age + 1
                
                info_text = f"ID:{track['id']} | Frames: {start_frame}-{end_frame} ({age})"
                cv2.putText(canvas, info_text, (x1, y1 - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        return canvas
    
    @staticmethod
    def _get_detection_color(class_id):
        return DETECTION_COLORS[class_id % len(DETECTION_COLORS)]
    
    @staticmethod
    def _get_track_color(track_id, track_color_fn=None):
        if track_color_fn:
            return track_color_fn(track_id)
        return (60 + (37 * track_id) % 190,
                80 + (53 * track_id) % 160,
                100 + (29 * track_id) % 150)
    
    @staticmethod
    def _build_track_label(track_info, cfg, include_prefix=False):
        if not track_info:
            return ""
        parts = []
        if cfg.get('track_id', True):
            parts.append(f"ID {track_info['id']}")
        if cfg.get('track_age'):
            parts.append(f"Age {track_info.get('age', 0)}")
        if cfg.get('velocity'):
            vx, vy = track_info.get('velocity', (0.0, 0.0))
            parts.append(f"Vel {vx:.1f},{vy:.1f}")
        if cfg.get('track_confidence'):
            parts.append(f"Conf {track_info.get('confidence', 0.0):.2f}")
        return " | ".join(parts)

class ImageLabel(QLabel):
    """可選取區域的圖片標籤"""
    selection_changed = pyqtSignal(object)
    mouse_moved = pyqtSignal(QPoint) # New signal
    
    def __init__(self):
        super().__init__()
        self.setMouseTracking(True)
        self.start_point = None
        self.end_point = None
        self.selecting = False
        self.crop_rect = None
        self.scale_factor_x = 1.0
        self.scale_factor_y = 1.0
        self.base_pixmap = None
        self.display_pixmap = None
        self.is_playback_mode = False
        self.pixmap_offset = QPoint(0, 0)
        
    def set_image(self, image_path):
        """載入並顯示圖片"""
        pixmap = QPixmap(image_path)
        if pixmap.isNull():
            return False
        self.is_playback_mode = False
        self.base_pixmap = pixmap
        self.start_point = None
        self.end_point = None
        self.crop_rect = None
        self.selection_changed.emit(self.crop_rect)
        self.update_display_pixmap()
        return True

    def set_playback_mode(self, enabled: bool):
        self.is_playback_mode = enabled
        if enabled:
            self.selecting = False
            self.start_point = None
            self.end_point = None
        self.redraw_selection_overlay()

    def clear_selection(self):
        self.start_point = None
        self.end_point = None
        self.crop_rect = None
        self.selection_changed.emit(self.crop_rect)
        self.redraw_selection_overlay()

    def update_display_pixmap(self):
        if self.base_pixmap is None:
            return
        target_width = max(1, self.width())
        target_height = max(1, self.height())
        scaled_pixmap = self.base_pixmap.scaled(
            target_width,
            target_height,
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation
        )
        self.display_pixmap = scaled_pixmap
        if scaled_pixmap.width() > 0:
            self.scale_factor_x = self.base_pixmap.width() / scaled_pixmap.width()
        else:
            self.scale_factor_x = 1.0
        if scaled_pixmap.height() > 0:
            self.scale_factor_y = self.base_pixmap.height() / scaled_pixmap.height()
        else:
            self.scale_factor_y = 1.0
        offset_x = max(0, (self.width() - scaled_pixmap.width()) // 2)
        offset_y = max(0, (self.height() - scaled_pixmap.height()) // 2)
        self.pixmap_offset = QPoint(offset_x, offset_y)
        self.redraw_selection_overlay()

    def redraw_selection_overlay(self):
        if self.display_pixmap is None:
            return
        base = self.display_pixmap.copy()
        if self.start_point and self.end_point and not self.is_playback_mode:
            painter = QPainter(base)
            painter.setPen(QPen(Qt.red, 2))
            rect = QRect(self.start_point, self.end_point)
            painter.drawRect(rect)
            painter.end()
        self.setPixmap(base)
    
    def resizeEvent(self, event):
        super().resizeEvent(event)
        if self.base_pixmap is not None:
            self.update_display_pixmap()
            if self.crop_rect and not self.is_playback_mode:
                self.restore_selection_from_crop()

    def restore_selection_from_crop(self):
        if self.crop_rect is None or self.display_pixmap is None or self.base_pixmap is None:
            return
        scale_x = self.display_pixmap.width() / self.base_pixmap.width()
        scale_y = self.display_pixmap.height() / self.base_pixmap.height()
        x1 = int(self.crop_rect[0] * scale_x)
        y1 = int(self.crop_rect[1] * scale_y)
        x2 = int(self.crop_rect[2] * scale_x)
        y2 = int(self.crop_rect[3] * scale_y)
        self.start_point = QPoint(x1, y1)
        self.end_point = QPoint(x2, y2)
        self.redraw_selection_overlay()

    def show_frame(self, frame):
        """顯示影片幀並停用選取"""
        if frame is None:
            return
        self.set_playback_mode(True)
        if len(frame.shape) == 3 and frame.shape[2] == 3:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, ch = frame_rgb.shape
            bytes_per_line = ch * w
            qimg = QImage(frame_rgb.data, w, h, bytes_per_line, QImage.Format_RGB888).copy()
        else:
            h, w = frame.shape[:2]
            qimg = QImage(frame.data, w, h, w, QImage.Format_Grayscale8).copy()
        self.base_pixmap = QPixmap.fromImage(qimg)
        self.update_display_pixmap()
        self.selection_changed.emit(None)

    def _map_to_pixmap_point(self, point: QPoint):
        if self.display_pixmap is None:
            return None
        px = point.x() - self.pixmap_offset.x()
        py = point.y() - self.pixmap_offset.y()
        if px < 0 or py < 0 or px >= self.display_pixmap.width() or py >= self.display_pixmap.height():
            return None
        return QPoint(px, py)
    
    def mousePressEvent(self, event):
        if self.is_playback_mode:
            return
        if event.button() == Qt.LeftButton:
            mapped = self._map_to_pixmap_point(event.pos())
            if mapped is None:
                return
            self.start_point = mapped
            self.end_point = mapped
            self.selecting = True
    
    def mouseMoveEvent(self, event):
        # Always emit mouse move if mapped
        mapped = self._map_to_pixmap_point(event.pos())
        if mapped:
            # Convert to original image coordinates
            orig_x = int(mapped.x() * self.scale_factor_x)
            orig_y = int(mapped.y() * self.scale_factor_y)
            self.mouse_moved.emit(QPoint(orig_x, orig_y))

        if self.is_playback_mode:
            return
        if self.selecting:
            if mapped is None:
                return
            self.end_point = mapped
            self.update_selection()
    
    def mouseReleaseEvent(self, event):
        if self.is_playback_mode:
            return
        if event.button() == Qt.LeftButton:
            self.selecting = False
            mapped = self._map_to_pixmap_point(event.pos())
            if mapped is not None:
                self.end_point = mapped
            self.update_selection()
            self.save_crop_rect()
    
    def update_selection(self):
        if self.start_point and self.end_point:
            self.redraw_selection_overlay()
    
    def save_crop_rect(self):
        """儲存裁切區域（轉換為原始圖片座標）"""
        if self.start_point and self.end_point and self.base_pixmap is not None:
            x1 = int(min(self.start_point.x(), self.end_point.x()) * self.scale_factor_x)
            y1 = int(min(self.start_point.y(), self.end_point.y()) * self.scale_factor_y)
            x2 = int(max(self.start_point.x(), self.end_point.x()) * self.scale_factor_x)
            y2 = int(max(self.start_point.y(), self.end_point.y()) * self.scale_factor_y)
            x1 = max(0, min(self.base_pixmap.width(), x1))
            y1 = max(0, min(self.base_pixmap.height(), y1))
            x2 = max(0, min(self.base_pixmap.width(), x2))
            y2 = max(0, min(self.base_pixmap.height(), y2))
            if x2 - x1 > 0 and y2 - y1 > 0:
                self.crop_rect = (x1, y1, x2, y2)
            else:
                self.crop_rect = None
            self.selection_changed.emit(self.crop_rect)
    
    def get_crop_rect(self):
        return self.crop_rect
    
    def reset_selection(self):
        """重置選取區域"""
        self.clear_selection()

    def set_crop_rect(self, crop_rect):
        self.crop_rect = crop_rect
        if crop_rect is None:
            self.clear_selection()
            return
        if self.display_pixmap is None:
            return
        self.restore_selection_from_crop()
        self.selection_changed.emit(self.crop_rect)


class ProcessThread(QThread):
    """處理執行緒"""
    progress = pyqtSignal(int)
    status = pyqtSignal(str)
    finished = pyqtSignal(bool, str)
    
    def __init__(self, task_type, params):
        super().__init__()
        self.task_type = task_type
        self.params = params
        self.track_colors = {}
        self.overlay_config = deepcopy(self.params.get('overlay_config', DEFAULT_OVERLAY_CONFIG))
        self.output_payload = None
    
    def run(self):
        try:
            if self.task_type == 'create_video':
                self.create_timelapse_video()
            elif self.task_type == 'analyze_video':
                self.create_analyzed_video()
            elif self.task_type == 'redraw_video':
                self.redraw_cached_video()
            elif self.task_type == 'redraw_original_video':
                self.redraw_original_video()
            self.finished.emit(True, "處理完成！")
        except Exception as e:
            
            
            print(traceback.format_exc())
            
            
            
            
            self.finished.emit(False, f"錯誤: {str(e)}")
    
    def create_timelapse_video(self):
        """創建時間序列影片"""
        images = self.params['images']
        output_path = self.params['output_path']
        crop_rect = self.params['crop_rect']
        fps = self.params['fps']
        timestamp_config = self.params['timestamp_config']
        
        self.status.emit("讀取圖片中...")
        

        # 讀取第一張圖片以獲取尺寸
        first_img = imread_safe(images[0])
        if first_img is None:
            raise ValueError(f"無法讀取第一張圖片: {images[0]}")
            
        if crop_rect:
            x1, y1, x2, y2 = crop_rect
            first_img = first_img[y1:y2, x1:x2]
        
        height, width = first_img.shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        # 計算時間戳記
        timestamps = self.extract_timestamps(images)
        start_time = timestamps[0] if timestamps else 0
        
        total = len(images)
        for idx, img_path in enumerate(images):
            img = imread_safe(img_path)
            if img is None:
                print(f"Warning: Failed to load image {img_path}, skipping.")
                continue
                
            if crop_rect:
                x1, y1, x2, y2 = crop_rect
                img = img[y1:y2, x1:x2]
            
            # 添加時間戳記
            if idx < len(timestamps):
                elapsed_min = (timestamps[idx] - start_time) / 60
                timestamp_text = f"{elapsed_min:.1f} min"
                img = add_timestamp(img, timestamp_text, timestamp_config)
            
            out.write(img)
            progress = int((idx + 1) / total * 100)
            self.progress.emit(progress)
            self.status.emit(f"處理中: {idx+1}/{total}")
        
        out.release()
    
    def create_analyzed_video(self):
        """創建分析標註影片"""
        images = self.params['images']
        output_path = self.params['output_path']
        crop_rect = self.params['crop_rect']
        fps = self.params['fps']
        timestamp_config = self.params['timestamp_config']
        models = self.params['models']
        excel_path = self.params['excel_path']
        
        self.status.emit("載入 YOLO 模型中...")
        

        yolo_models = [YOLO(path) for path in models]
        first_img = imread_safe(images[0])
        if first_img is None:
            raise ValueError(f"無法讀取第一張圖片: {images[0]}")
            
        if crop_rect:
            x1, y1, x2, y2 = crop_rect
            first_img = first_img[y1:y2, x1:x2]
        
        height, width = first_img.shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        tracker_type = self.params.get('tracker_type', 'ukf')
        tracker_config = self.params.get('tracker_config', {})
        # Merge class similarity map if provided
        if 'class_similarity_map' in self.params:
            tracker_config['class_similarity_map'] = self.params['class_similarity_map']
            
        local_tracker = create_tracker(tracker_type, tracker_config)
        
        self.track_colors = {}
        overlay_renderer = AnnotationRenderer(deepcopy(self.overlay_config))
        
        timestamps = self.extract_timestamps(images)
        start_time = timestamps[0] if timestamps else 0
        total = len(images)
        all_detections = []
        frame_cache = []
        
        for idx, img_path in enumerate(images):
            img = imread_safe(img_path)
            if img is None:
                print(f"Warning: Failed to load image {img_path}, skipping.")
                continue
            
            if crop_rect:
                x1, y1, x2, y2 = crop_rect
                img = img[y1:y2, x1:x2]
                crop_offset = (x1, y1)
            else:
                crop_offset = (0, 0)
            
            frame_detections = []
            for model_idx, model in enumerate(yolo_models):
                results = model(img, verbose=False)
                for result in results:
                    dets = convert_result_to_detections(result, os.path.basename(models[model_idx]), crop_offset)
                    frame_detections.extend(dets)
            
            tracked_objects = local_tracker.update(frame_detections, idx)
            track_records = [serialize_tracked_object(obj) for obj in tracked_objects]
            
            annotated_img = overlay_renderer.render(img.copy(), frame_detections, track_records, self.get_track_color)
            if idx < len(timestamps):
                elapsed_min = (timestamps[idx] - start_time) / 60
                timestamp_text = f"{elapsed_min:.1f} min"
                annotated_img = add_timestamp(annotated_img, timestamp_text, timestamp_config)
            
            out.write(annotated_img)
            progress = int((idx + 1) / total * 100)
            self.progress.emit(progress)
            self.status.emit(f"分析中: {idx+1}/{total}")
            
            detection_rows = self.build_detection_rows(frame_detections, track_records, timestamps, idx, img_path)
            all_detections.extend(detection_rows)
            frame_cache.append({
                'image_path': img_path,
                'crop_rect': crop_rect,
                'detections': deepcopy(frame_detections),
                'tracking': deepcopy(track_records),
                'timestamp': timestamps[idx] if idx < len(timestamps) else None
            })
        
        out.release()
        
        if all_detections:
            df = pd.DataFrame(all_detections)
            tracking_df = pd.DataFrame(local_tracker.tracking_history) if local_tracker.tracking_history else None
            with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
                df.to_excel(writer, index=False, sheet_name='Detections')
                if tracking_df is not None and not tracking_df.empty:
                    tracking_df.to_excel(writer, index=False, sheet_name='TrackingHistory')
            self.status.emit(f"Excel 報表已儲存: {excel_path}")
        
        self.output_payload = {
            'frames': frame_cache,
            'excel_rows': deepcopy(all_detections),
            'tracking_history': deepcopy(local_tracker.tracking_history),
            'timestamps': timestamps,
            'start_time': start_time,
            'crop_rect': crop_rect
        }

    
    def build_detection_rows(self, detections, track_records, timestamps, frame_idx, img_path):
        rows = []
        timestamp_value = timestamps[frame_idx] if frame_idx < len(timestamps) else 0
        datetime_value = datetime.fromtimestamp(timestamp_value).strftime('%Y-%m-%d %H:%M:%S') if timestamp_value else ''
        track_map = {track['id']: track for track in track_records if track.get('id') is not None}
        
        for det in detections:
            bbox_global = det.get('bbox_global') or det.get('bbox')
            if not bbox_global or len(bbox_global) < 4:
                continue
            x1g, y1g, x2g, y2g = bbox_global
            track_id = det.get('track_id')
            track_info = track_map.get(track_id)
            trail_count = len(track_info.get('trail', [])) if track_info else 0
            velocity = track_info.get('velocity', (0.0, 0.0)) if track_info else (0.0, 0.0)
            pred_pos = track_info.get('pred_pos', (0.0, 0.0)) if track_info else (0.0, 0.0)
            
            rows.append({
                '時間戳記': timestamp_value,
                '日期時間': datetime_value,
                '圖片檔名': os.path.basename(img_path),
                '模型': det.get('model_name'),
                '類別': det.get('class_name'),
                '追蹤ID': track_id,
                'x1': x1g,
                'y1': y1g,
                'x2': x2g,
                'y2': y2g,
                '信心度': det.get('confidence', 0.0),
                'bbox面積': (x2g - x1g) * (y2g - y1g),
                '軌跡點數': trail_count,
                'velocity_x': velocity[0],
                'velocity_y': velocity[1],
                'pred_x': pred_pos[0],
                'pred_y': pred_pos[1],
                'has_mask': 1 if det.get('masks') else 0,
                'keypoint_count': len(det.get('keypoints', []))
            })
        return rows
    
    def redraw_cached_video(self):
        analysis_data = self.params.get('analysis_data')
        if not analysis_data:
            raise ValueError("缺少分析快取資料，無法重新繪製。")
        
        frames = analysis_data.get('frames', [])
        if not frames:
            raise ValueError("分析資料中沒有可用的幀資訊。")
        
        output_path = self.params['output_path']
        fps = self.params['fps']
        timestamp_config = self.params['timestamp_config']
        excel_path = self.params['excel_path']
        timestamps = analysis_data.get('timestamps', [])
        start_time = analysis_data.get('start_time', timestamps[0] if timestamps else 0)
        
        first_frame = frames[0]
        base_img = cv2.imread(first_frame['image_path'])
        crop_rect = first_frame.get('crop_rect')
        if crop_rect:
            x1, y1, x2, y2 = crop_rect
            base_img = base_img[y1:y2, x1:x2]
        height, width = base_img.shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        overlay_renderer = AnnotationRenderer(deepcopy(self.overlay_config))
        total = len(frames)
        
        for idx, frame_info in enumerate(frames):
            frame_img = cv2.imread(frame_info['image_path'])
            frame_crop = frame_info.get('crop_rect')
            if frame_crop:
                x1, y1, x2, y2 = frame_crop
                frame_img = frame_img[y1:y2, x1:x2]
            
            detections = deepcopy(frame_info.get('detections', []))
            tracking = deepcopy(frame_info.get('tracking', []))
            
            annotated_img = overlay_renderer.render(frame_img.copy(), detections, tracking, self.get_track_color)
            timestamp_value = frame_info.get('timestamp')
            if timestamp_value is None and idx < len(timestamps):
                timestamp_value = timestamps[idx]
            if timestamp_value is not None:
                elapsed_min = (timestamp_value - start_time) / 60
                timestamp_text = f"{elapsed_min:.1f} min"
                annotated_img = add_timestamp(annotated_img, timestamp_text, timestamp_config)
            
            out.write(annotated_img)
            progress = int((idx + 1) / total * 100)
            self.progress.emit(progress)
            self.status.emit(f"重新繪製中: {idx+1}/{total}")
        
        out.release()
        
        excel_rows = analysis_data.get('excel_rows', [])
        tracking_history = analysis_data.get('tracking_history', [])
        if excel_rows:
            df = pd.DataFrame(excel_rows)
            tracking_df = pd.DataFrame(tracking_history) if tracking_history else None
            with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
                df.to_excel(writer, index=False, sheet_name='Detections')
                if tracking_df is not None and not tracking_df.empty:
                    tracking_df.to_excel(writer, index=False, sheet_name='TrackingHistory')
            self.status.emit(f"Excel 報表已儲存: {excel_path}")
        
        self.output_payload = deepcopy(analysis_data)
    
    def redraw_original_video(self):
        """
        將分析追蹤完成的結果，使用選擇的繪製選項，
        繪製到原始未裁切的圖片上，然後製作成影片檔
        """
        analysis_data = self.params.get('analysis_data')
        if not analysis_data:
            raise ValueError("缺少分析快取資料，無法重新繪製。")
        
        frames = analysis_data.get('frames', [])
        if not frames:
            raise ValueError("分析資料中沒有可用的幀資訊。")
        
        output_path = self.params['output_path']
        fps = self.params['fps']
        timestamp_config = self.params['timestamp_config']
        excel_path = self.params['excel_path']
        timestamps = analysis_data.get('timestamps', [])
        start_time = analysis_data.get('start_time', timestamps[0] if timestamps else 0)
        
        # 讀取第一張原始圖片以獲取尺寸 (未裁切的完整原圖)
        first_frame = frames[0]
        base_img = imread_safe(first_frame['image_path'])
        if base_img is None:
            raise ValueError(f"無法讀取第一張圖片: {first_frame['image_path']}")
        
        height, width = base_img.shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        overlay_renderer = AnnotationRenderer(deepcopy(self.overlay_config))
        total = len(frames)
        
        for idx, frame_info in enumerate(frames):
            # 讀取原始完整圖片 (不裁切，使用完整未選取區域的原圖)
            frame_img = imread_safe(frame_info['image_path'])
            if frame_img is None:
                continue
            
            # 獲取裁切區域資訊
            frame_crop = frame_info.get('crop_rect')
            
            # 深拷貝偵測和追蹤資料
            detections = deepcopy(frame_info.get('detections', []))
            tracking = deepcopy(frame_info.get('tracking', []))
            
            # 如果有裁切區域，需要調整偵測框和追蹤框的座標
            # 因為原始分析時是在裁切後的圖片上進行的
            # 現在要繪製到原始圖片上，需要加上裁切偏移
            if frame_crop:
                x1_offset, y1_offset, _, _ = frame_crop
                
                # 調整所有偵測框的座標
                for det in detections:
                    if 'bbox' in det:
                        bbox = det['bbox']
                        det['bbox'] = [
                            bbox[0] + x1_offset,
                            bbox[1] + y1_offset,
                            bbox[2] + x1_offset,
                            bbox[3] + y1_offset
                        ]
                    # masks 和 keypoints 也需要調整
                    if 'masks' in det and det['masks']:
                        adjusted_masks = []
                        for poly in det['masks']:
                            # poly is a flat list: [x1, y1, x2, y2, ...]
                            adjusted_poly = []
                            for i in range(0, len(poly), 2):
                                if i + 1 < len(poly):
                                    adjusted_poly.append(poly[i] + x1_offset)
                                    adjusted_poly.append(poly[i+1] + y1_offset)
                            adjusted_masks.append(adjusted_poly)
                        det['masks'] = adjusted_masks
                    
                    if 'keypoints' in det and det['keypoints']:
                        for kp in det['keypoints']:
                            kp['x'] += x1_offset
                            kp['y'] += y1_offset
                
                # 調整所有追蹤框的座標
                for track in tracking:
                    if 'bbox' in track:
                        bbox = track['bbox']
                        track['bbox'] = [
                            bbox[0] + x1_offset,
                            bbox[1] + y1_offset,
                            bbox[2] + x1_offset,
                            bbox[3] + y1_offset
                        ]
                    # 調整軌跡點
                    if 'trail' in track and track['trail']:
                        adjusted_trail = []
                        for pt in track['trail']:
                            adjusted_trail.append((pt[0] + x1_offset, pt[1] + y1_offset))
                        track['trail'] = adjusted_trail
            
            # 在原始完整圖片上繪製標註
            annotated_img = overlay_renderer.render(frame_img.copy(), detections, tracking, self.get_track_color)
            
            # 添加時間戳記
            timestamp_value = frame_info.get('timestamp')
            if timestamp_value is None and idx < len(timestamps):
                timestamp_value = timestamps[idx]
            if timestamp_value is not None:
                elapsed_min = (timestamp_value - start_time) / 60
                timestamp_text = f"{elapsed_min:.1f} min"
                annotated_img = add_timestamp(annotated_img, timestamp_text, timestamp_config)
            
            out.write(annotated_img)
            progress = int((idx + 1) / total * 100)
            self.progress.emit(progress)
            self.status.emit(f"重新繪製到原始圖片: {idx+1}/{total}")
        
        out.release()
        
        # 儲存 Excel 報表
        excel_rows = analysis_data.get('excel_rows', [])
        tracking_history = analysis_data.get('tracking_history', [])
        if excel_rows:
            df = pd.DataFrame(excel_rows)
            tracking_df = pd.DataFrame(tracking_history) if tracking_history else None
            with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
                df.to_excel(writer, index=False, sheet_name='Detections')
                if tracking_df is not None and not tracking_df.empty:
                    tracking_df.to_excel(writer, index=False, sheet_name='TrackingHistory')
            self.status.emit(f"Excel 報表已儲存: {excel_path}")
        
        self.output_payload = deepcopy(analysis_data)
    
    def extract_timestamps(self, image_paths):
        """從檔名提取時間戳記"""
        timestamps = []
        for path in image_paths:
            # 尋找檔名中的時間戳記 (例如: cam1_frame_1234567890.tiff)
            match = re.search(r'_(\d+)\.(tiff?|jpg|png|jpeg)', os.path.basename(path))
            if match:
                timestamps.append(int(match.group(1)))
            else:
                timestamps.append(0)
        return timestamps
    


    def get_track_color(self, track_id):
        """取得穩定且可重現的追蹤色彩"""
        if track_id in self.track_colors:
            return self.track_colors[track_id]
        color = (
            60 + (37 * track_id) % 190,
            80 + (53 * track_id) % 160,
            100 + (29 * track_id) % 150
        )
        self.track_colors[track_id] = color
        return color


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("多相機時間序列影像分析工具")
        self.setGeometry(100, 100, 1200, 800)
        
        self.folder_path = None
        self.image_files = []
        self.model_files = []
        self.crop_rect = None
        self.timestamp_color = (0, 255, 0)  # BGR: 綠色
        self.timestamp_size = 1.0
        self.overlay_config = deepcopy(DEFAULT_OVERLAY_CONFIG)
        self.overlay_checkboxes = {}
        self.saved_crop_rect = None
        self.last_analysis_data = None
        self.loaded_models = {}
        self.video_timer = QTimer(self)
        self.video_timer.timeout.connect(self.update_video_frame)
        self.video_capture = None
        self.current_video_path = None
        self.btn_redraw_video = None
        
        # New Attributes
        self.tracker_type = 'ukf'
        self.class_similarity_map = None
        self.total_frames = 0
        self.current_frame_idx = 0
        self.is_paused = False
        self.current_video_source = 'raw' # 'raw' or 'analysis'
        self.track_colors = {} # Store track colors for consistent visualization
        
        self.init_ui()
    
    def init_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)
        
        # 左側控制面板
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        
        # 資料夾選擇
        folder_group = QGroupBox("1. 選擇資料夾")
        folder_layout = QVBoxLayout()
        
        btn_select_folder = QPushButton("選擇資料夾")
        btn_select_folder.clicked.connect(self.select_folder)
        folder_layout.addWidget(btn_select_folder)
        
        self.lbl_folder = QLabel("未選擇資料夾")
        self.lbl_folder.setWordWrap(True)
        folder_layout.addWidget(self.lbl_folder)
        
        folder_group.setLayout(folder_layout)
        left_layout.addWidget(folder_group)
        
        # 圖片列表
        image_group = QGroupBox("2. 圖片列表")
        image_layout = QVBoxLayout()
        
        self.lbl_image_count = QLabel("圖片數量: 0")
        image_layout.addWidget(self.lbl_image_count)
        
        # 相機過濾器
        filter_layout = QHBoxLayout()
        filter_layout.addWidget(QLabel("過濾相機:"))
        self.combo_camera = QComboBox()
        self.combo_camera.addItem("全部")
        self.combo_camera.currentTextChanged.connect(self.filter_images)
        filter_layout.addWidget(self.combo_camera)
        image_layout.addLayout(filter_layout)
        
        self.list_images = QListWidget()
        self.list_images.itemClicked.connect(self.preview_image)
        image_layout.addWidget(self.list_images)
        
        image_group.setLayout(image_layout)
        left_layout.addWidget(image_group)
        
        # YOLO 模型選擇
        model_group = QGroupBox("3. YOLO 模型 (可多選)")
        model_layout = QVBoxLayout()
        
        self.lbl_model_count = QLabel("模型數量: 0")
        model_layout.addWidget(self.lbl_model_count)
        
        self.list_models = QListWidget()
        self.list_models.setSelectionMode(QAbstractItemView.SingleSelection)
        self.list_models.itemSelectionChanged.connect(self.on_model_selection_changed)
        model_layout.addWidget(self.list_models)
        
        btn_test_model = QPushButton("測試模型預覽")
        btn_test_model.clicked.connect(self.preview_model_prediction)
        model_layout.addWidget(btn_test_model)
        
        model_group.setLayout(model_layout)
        left_layout.addWidget(model_group)
        
        # 時間戳記設定
        timestamp_group = QGroupBox("4. 時間戳記設定")
        timestamp_layout = QGridLayout()
        
        timestamp_layout.addWidget(QLabel("文字大小:"), 0, 0)
        self.spin_timestamp_size = QDoubleSpinBox()
        self.spin_timestamp_size.setRange(0.5, 3.0)
        self.spin_timestamp_size.setValue(1.0)
        self.spin_timestamp_size.setSingleStep(0.1)
        self.spin_timestamp_size.valueChanged.connect(self.update_timestamp_size)
        timestamp_layout.addWidget(self.spin_timestamp_size, 0, 1)
        
        self.btn_color = QPushButton("選擇顏色")
        self.btn_color.clicked.connect(self.select_timestamp_color)
        self.btn_color.setStyleSheet("background-color: rgb(0, 255, 0);")
        timestamp_layout.addWidget(self.btn_color, 1, 0, 1, 2)
        
        timestamp_group.setLayout(timestamp_layout)
        timestamp_group.setLayout(timestamp_layout)
        left_layout.addWidget(timestamp_group)

        # Tracker Settings
        tracker_group = QGroupBox("5. 追蹤器設定")
        tracker_layout = QVBoxLayout()
        
        tracker_sel_layout = QHBoxLayout()
        tracker_sel_layout.addWidget(QLabel("演算法:"))
        self.combo_tracker = QComboBox()
        self.combo_tracker.addItems(['UKF (Default)', 'SORT', 'ByteTrack'])
        self.combo_tracker.currentTextChanged.connect(self.on_tracker_changed)
        tracker_sel_layout.addWidget(self.combo_tracker)
        tracker_layout.addLayout(tracker_sel_layout)
        
        btn_similarity = QPushButton("設定類別相似度 (Class Similarity)")
        btn_similarity.clicked.connect(self.open_similarity_dialog)
        tracker_layout.addWidget(btn_similarity)
        
        tracker_group.setLayout(tracker_layout)
        left_layout.addWidget(tracker_group)
        
        # 標註繪製選項
        overlay_group = QGroupBox("6. 標註繪製選項")
        overlay_layout = QGridLayout()
        overlay_group.setLayout(overlay_layout)
        
        def add_overlay_checkbox(label, key, row, col):
            checkbox = QCheckBox(label)
            checkbox.setChecked(self.overlay_config.get(key, False))
            checkbox.stateChanged.connect(lambda state, opt=key: self.update_overlay_option(opt, state))
            overlay_layout.addWidget(checkbox, row, col)
            self.overlay_checkboxes[key] = checkbox
        
        add_overlay_checkbox("顯示 BBox", 'bbox', 0, 0)
        add_overlay_checkbox("顯示信心度", 'probs', 0, 1)
        add_overlay_checkbox("顯示遮罩(seg)", 'masks', 0, 2)
        add_overlay_checkbox("關節線(pose)", 'kpt_line', 1, 0)
        add_overlay_checkbox("顯示追蹤ID", 'track_id', 1, 1)
        add_overlay_checkbox("追蹤長度", 'track_age', 1, 2)
        add_overlay_checkbox("速度向量", 'velocity', 2, 0)
        add_overlay_checkbox("追蹤信心", 'track_confidence', 2, 1)
        add_overlay_checkbox("軌跡", 'trail', 2, 2)
        add_overlay_checkbox("預測位置", 'pred_position', 3, 2)
        
        overlay_layout.addWidget(QLabel("字體大小"), 3, 0)
        self.spin_overlay_font = QDoubleSpinBox()
        self.spin_overlay_font.setRange(0.3, 3.0)
        self.spin_overlay_font.setSingleStep(0.1)
        self.spin_overlay_font.setValue(self.overlay_config.get('font_scale', 0.6))
        self.spin_overlay_font.valueChanged.connect(self.update_overlay_font)
        overlay_layout.addWidget(self.spin_overlay_font, 3, 1)
        
        left_layout.addWidget(overlay_group)
        
        # 影片設定
        video_group = QGroupBox("7. 影片設定")
        video_layout = QGridLayout()
        
        video_layout.addWidget(QLabel("FPS:"), 0, 0)
        self.spin_fps = QSpinBox()
        self.spin_fps.setRange(1, 60)
        self.spin_fps.setValue(10)
        video_layout.addWidget(self.spin_fps, 0, 1)
        
        video_group.setLayout(video_layout)
        left_layout.addWidget(video_group)
        
        # 執行按鈕
        action_group = QGroupBox("8. 執行")
        action_layout = QVBoxLayout()
        
        btn_create_video = QPushButton("生成時間序列影片")
        btn_create_video.clicked.connect(self.create_timelapse_video)
        action_layout.addWidget(btn_create_video)
        
        btn_analyze_video = QPushButton("生成分析標註影片 + Excel")
        btn_analyze_video.clicked.connect(self.create_analyzed_video)
        action_layout.addWidget(btn_analyze_video)
        
        self.btn_redraw_video = QPushButton("重新繪製上次分析")
        self.btn_redraw_video.clicked.connect(self.redraw_analyzed_video)
        self.btn_redraw_video.setEnabled(False)
        action_layout.addWidget(self.btn_redraw_video)
        
        self.btn_play_analysis = QPushButton("播放分析結果 (互動預覽)")
        self.btn_play_analysis.clicked.connect(self.play_analysis_result)
        self.btn_play_analysis.setEnabled(False)
        action_layout.addWidget(self.btn_play_analysis)
        
        # ✅ Feature 4: 新增「重新繪製到原圖」按鈕
        self.btn_redraw_to_original = QPushButton("重新繪製到原圖上")
        self.btn_redraw_to_original.clicked.connect(self.redraw_to_original_images)
        self.btn_redraw_to_original.setEnabled(False)
        action_layout.addWidget(self.btn_redraw_to_original)
        
        action_group.setLayout(action_layout)
        left_layout.addWidget(action_group)
        
        # 進度條
        self.progress_bar = QProgressBar()
        left_layout.addWidget(self.progress_bar)
        
        self.lbl_status = QLabel("就緒")
        left_layout.addWidget(self.lbl_status)
        
        left_layout.addStretch()
        
        # 右側預覽面板
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        
        preview_label = QLabel("圖片預覽與區域選取")
        preview_label.setAlignment(Qt.AlignCenter)
        right_layout.addWidget(preview_label)
        
        instruction_label = QLabel("拖曳滑鼠選取要裁切的區域（可選）")
        instruction_label.setAlignment(Qt.AlignCenter)
        right_layout.addWidget(instruction_label)
        
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        
        self.image_preview = ImageLabel()
        self.image_preview.setAlignment(Qt.AlignCenter)
        self.image_preview.setMinimumSize(800, 600)
        self.image_preview.selection_changed.connect(self.on_selection_changed)
        self.image_preview.mouse_moved.connect(self.on_mouse_move)
        scroll_area.setWidget(self.image_preview)
        
        right_layout.addWidget(scroll_area)
        
        # Video Controls
        video_controls = QHBoxLayout()
        
        self.btn_play_pause = QPushButton("暫停")
        self.btn_play_pause.clicked.connect(self.toggle_playback)
        self.btn_play_pause.setEnabled(False)
        video_controls.addWidget(self.btn_play_pause)
        
        self.slider_video = QSlider(Qt.Horizontal)
        self.slider_video.sliderPressed.connect(self.pause_video)
        self.slider_video.sliderReleased.connect(self.resume_video)
        self.slider_video.valueChanged.connect(self.seek_video)
        self.slider_video.setEnabled(False)
        video_controls.addWidget(self.slider_video)
        
        self.spin_frame = QSpinBox()
        self.spin_frame.setRange(0, 0)
        self.spin_frame.valueChanged.connect(self.change_frame)
        self.spin_frame.setEnabled(False)
        video_controls.addWidget(self.spin_frame)
        
        self.lbl_total_frames = QLabel("/ 0")
        video_controls.addWidget(self.lbl_total_frames)
        
        right_layout.addLayout(video_controls)
        
        btn_reset_selection = QPushButton("取消選取區域")
        btn_reset_selection.clicked.connect(self.reset_selection)
        right_layout.addWidget(btn_reset_selection)
        
        # 添加到主佈局
        main_layout.addWidget(left_panel, 1)
        main_layout.addWidget(right_panel, 2)
    
    def start_video_preview(self, video_path):
        """在預覽區播放指定影片"""
        if not video_path or not os.path.exists(video_path):
            return
        self.stop_video_preview()
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            QMessageBox.warning(self, "警告", f"無法開啟影片: {video_path}")
            return
        self.video_capture = cap
        fps = cap.get(cv2.CAP_PROP_FPS)
        if not fps or fps <= 0:
            fps = 24
        
        self.total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.current_frame_idx = 0
        self.is_paused = False
        
        # Update Controls
        self.slider_video.setRange(0, self.total_frames - 1)
        self.slider_video.setValue(0)
        self.slider_video.setEnabled(True)
        
        self.spin_frame.setRange(0, self.total_frames - 1)
        self.spin_frame.setValue(0)
        self.spin_frame.setEnabled(True)
        
        self.lbl_total_frames.setText(f"/ {self.total_frames - 1}")
        self.btn_play_pause.setText("暫停")
        self.btn_play_pause.setEnabled(True)
        
        interval = max(30, int(1000 / fps))
        self.current_video_path = video_path
        self.video_timer.start(interval)
        self.update_video_frame()  # 立即顯示第一幀

    def update_video_frame(self):
        if self.current_video_source == 'raw':
            if self.video_capture is None:
                self.stop_video_preview()
                return
                
            if not self.is_paused:
                if self.current_frame_idx >= self.total_frames:
                    self.current_frame_idx = 0
                    self.video_capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
                
                ret, frame = self.video_capture.read()
                if ret:
                    self.image_preview.show_frame(frame)
                    self.current_frame_idx += 1
                    self._update_controls()
                else:
                    self.current_frame_idx = 0
                    self.video_capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
        
        elif self.current_video_source == 'analysis':
            if not self.last_analysis_data:
                self.stop_video_preview()
                return
            
            frames = self.last_analysis_data.get('frames', [])
            if not frames:
                self.stop_video_preview()
                return
                
            if not self.is_paused:
                if self.current_frame_idx >= len(frames):
                    self.current_frame_idx = 0
                
                self._render_analysis_frame(self.current_frame_idx)
                self.current_frame_idx += 1
                self._update_controls()
            else:
                # If paused, we might still need to re-render if hover changed
                self._render_analysis_frame(self.current_frame_idx)

    def _update_controls(self):
        self.slider_video.blockSignals(True)
        self.slider_video.setValue(self.current_frame_idx - 1 if self.current_frame_idx > 0 else 0)
        self.slider_video.blockSignals(False)
        
        self.spin_frame.blockSignals(True)
        self.spin_frame.setValue(self.current_frame_idx - 1 if self.current_frame_idx > 0 else 0)
        self.spin_frame.blockSignals(False)

    def _render_analysis_frame(self, idx):
        frames = self.last_analysis_data.get('frames', [])
        if idx < 0 or idx >= len(frames):
            return
            
        frame_info = frames[idx]
        img_path = frame_info['image_path']
        if not os.path.exists(img_path):
            return
            

            
        img = imread_safe(img_path)
        if img is None:
            return
            
        crop_rect = frame_info.get('crop_rect')
        if crop_rect:
            x1, y1, x2, y2 = crop_rect
            img = img[y1:y2, x1:x2]
            
        detections = frame_info.get('detections', [])
        tracking = frame_info.get('tracking', [])
        
        renderer = AnnotationRenderer(self.get_overlay_config_copy())
        annotated_img = renderer.render(img, detections, tracking, self.get_track_color, self.hover_track_id)
        
        # Add timestamp
        timestamps = self.last_analysis_data.get('timestamps', [])
        start_time = self.last_analysis_data.get('start_time', 0)
        timestamp_value = frame_info.get('timestamp')
        if timestamp_value is None and idx < len(timestamps):
            timestamp_value = timestamps[idx]
            
        if timestamp_value is not None:
            elapsed_min = (timestamp_value - start_time) / 60
            timestamp_text = f"{elapsed_min:.1f} min"
            annotated_img = add_timestamp(annotated_img, timestamp_text, self.get_timestamp_config())
            
        self.image_preview.show_frame(annotated_img)

    def play_analysis_result(self):
        if not self.last_analysis_data:
            return
        
        self.stop_video_preview()
        self.current_video_source = 'analysis'
        frames = self.last_analysis_data.get('frames', [])
        self.total_frames = len(frames)
        self.current_frame_idx = 0
        self.is_paused = False
        
        # Update Controls
        self.slider_video.setRange(0, self.total_frames - 1)
        self.slider_video.setValue(0)
        self.slider_video.setEnabled(True)
        
        self.spin_frame.setRange(0, self.total_frames - 1)
        self.spin_frame.setValue(0)
        self.spin_frame.setEnabled(True)
        
        self.lbl_total_frames.setText(f"/ {self.total_frames - 1}")
        self.btn_play_pause.setText("暫停")
        self.btn_play_pause.setEnabled(True)
        
        fps = self.spin_fps.value()
        interval = max(30, int(1000 / fps))
        self.video_timer.start(interval)
        self.update_video_frame()

    def on_mouse_move(self, point):
        if self.current_video_source != 'analysis' or not self.last_analysis_data:
            return
            
        frames = self.last_analysis_data.get('frames', [])
        # Use current frame index (or previous one since update increments it)
        idx = max(0, self.current_frame_idx - 1)
        if idx >= len(frames):
            return
            
        frame_info = frames[idx]
        detections = frame_info.get('detections', [])
        
        # Check if mouse is inside any bbox
        # Note: point is in original image coordinates (cropped image coordinates if cropped)
        # The detections bbox are relative to the cropped image in frame_info['detections']
        # because we stored them that way in ProcessThread.
        
        hover_id = None
        min_dist = float('inf')
        
        x, y = point.x(), point.y()
        
        for det in detections:
            bbox = det.get('bbox')
            if not bbox: continue
            x1, y1, x2, y2 = bbox
            if x1 <= x <= x2 and y1 <= y <= y2:
                # Inside bbox
                track_id = det.get('track_id')
                if track_id is not None:
                    # If multiple overlaps, pick smallest area or center dist?
                    # Pick the one with center closest to mouse
                    cx, cy = (x1+x2)/2, (y1+y2)/2
                    dist = (x-cx)**2 + (y-cy)**2
                    if dist < min_dist:
                        min_dist = dist
                        hover_id = track_id
        
        if self.hover_track_id != hover_id:
            self.hover_track_id = hover_id
            # If paused, force update to show highlight immediately
            if self.is_paused:
                self.update_video_frame()

    def stop_video_preview(self):
        if self.video_timer.isActive():
            self.video_timer.stop()
        if self.video_capture is not None:
            self.video_capture.release()
            self.video_capture = None
        self.current_video_path = None
        self.image_preview.set_playback_mode(False)
        self.current_video_source = 'raw'
        self.hover_track_id = None
        
        # Reset Controls
        self.slider_video.setEnabled(False)
        self.spin_frame.setEnabled(False)
        self.btn_play_pause.setEnabled(False)
        self.lbl_total_frames.setText("/ 0")

    def toggle_playback(self):
        self.is_paused = not self.is_paused
        self.btn_play_pause.setText("播放" if self.is_paused else "暫停")

    def pause_video(self):
        self.is_paused = True
        self.btn_play_pause.setText("播放")

    def resume_video(self):
        # Only resume if it was playing before? Or just resume?
        # Usually slider release resumes playback.
        self.is_paused = False
        self.btn_play_pause.setText("暫停")

    def seek_video(self, frame_idx):
        if self.current_video_source == 'raw':
            if self.video_capture is None:
                return
            self.current_frame_idx = frame_idx
            self.video_capture.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = self.video_capture.read()
            if ret:
                self.image_preview.show_frame(frame)
                # Sync spinbox
                self.spin_frame.blockSignals(True)
                self.spin_frame.setValue(frame_idx)
                self.spin_frame.blockSignals(False)
        elif self.current_video_source == 'analysis':
            # Analysis playback mode
            self.current_frame_idx = frame_idx
            self._render_analysis_frame(frame_idx)
            # Sync spinbox
            self.spin_frame.blockSignals(True)
            self.spin_frame.setValue(frame_idx)
            self.spin_frame.blockSignals(False)

    def change_frame(self, frame_idx):
        if self.current_video_source == 'raw' and self.video_capture is None:
            return
        elif self.current_video_source == 'analysis' and not self.last_analysis_data:
            return
        # Similar to seek but triggered by spinbox
        self.seek_video(frame_idx)
        self.slider_video.blockSignals(True)
        self.slider_video.setValue(frame_idx)
        self.slider_video.blockSignals(False)

    def on_tracker_changed(self, text):
        if 'SORT' in text:
            self.tracker_type = 'sort'
        elif 'ByteTrack' in text:
            self.tracker_type = 'bytetrack'
        else:
            self.tracker_type = 'ukf'
        print(f"Tracker changed to: {self.tracker_type}")

    def on_model_selection_changed(self):
        selected_models = self.get_selected_model_paths()
        if not selected_models:
            return
            
        model_path = selected_models[0]
        try:
            model = self.load_model_instance(model_path)
            # Check if current map is valid
            if self.class_similarity_map:
                valid_ids = set(model.names.keys())
                invalid_keys = []
                for (id_a, id_b) in self.class_similarity_map.keys():
                    if id_a not in valid_ids or id_b not in valid_ids:
                        invalid_keys.append((id_a, id_b))
                
                if invalid_keys:
                    QMessageBox.warning(self, "警告", 
                        "更換模型後，目前的類別相似度設定包含無效的類別ID，\n"
                        "系統將自動重置相似度設定為預設值 (嚴格匹配)。")
                    self.class_similarity_map = {}
        except Exception as e:
            print(f"Error loading model for validation: {e}")

    def open_similarity_dialog(self):
        model_classes = {}
        selected_models = self.get_selected_model_paths()
        if selected_models:
            try:
                model = self.load_model_instance(selected_models[0])
                model_classes = model.names
            except Exception as e:
                QMessageBox.warning(self, "警告", f"無法讀取模型類別: {e}")
        
        dialog = ClassSimilarityDialog(self.class_similarity_map, model_classes, self)
        if dialog.exec_() == QDialog.Accepted:
            self.class_similarity_map = dialog.get_map()
            print(f"Class Similarity Map updated: {self.class_similarity_map}")

    def select_folder(self):
        """選擇資料夾"""
        folder = QFileDialog.getExistingDirectory(self, "選擇資料夾")
        if folder:
            self.stop_video_preview()
            self.folder_path = folder
            self.lbl_folder.setText(f"已選擇: {folder}")
            self.saved_crop_rect = None
            self.crop_rect = None
            self.last_analysis_data = None
            if self.btn_redraw_video:
                self.btn_redraw_video.setEnabled(False)
            self.load_files()
    
    def load_files(self):
        """載入圖片、影片和模型檔案"""
        if not self.folder_path:
            return
        
        # 載入圖片和影片
        self.image_files = []
        image_extensions = ['.tiff', '.tif', '.jpg', '.jpeg', '.png']
        video_extensions = ['.mp4', '.avi', '.mov', '.mkv']  # ✅ Feature 1: 新增影片支援
        
        for file in os.listdir(self.folder_path):
            # 圖片檔案
            if any(file.lower().endswith(ext) for ext in image_extensions):
                # 排除標註圖片
                if '_anno_' not in file and '_annotated' not in file:
                    self.image_files.append(os.path.join(self.folder_path, file))
            # 影片檔案 
            elif any(file.lower().endswith(ext) for ext in video_extensions):
                # 影片檔案代表一台相機的資料
                if '_anno_' not in file and '_annotated' not in file:
                    self.image_files.append(os.path.join(self.folder_path, file))
        
        self.image_files.sort()
        self.lbl_image_count.setText(f"檔案數量: {len(self.image_files)} (圖片+影片)")
        
        # 🔧 Bug Fix 1: 更新相機過濾器 (包含影片檔案)
        cameras = set()
        for file_path in self.image_files:
            basename = os.path.basename(file_path)
            # 從圖片和影片檔名提取相機編號
            match = re.search(r'cam(\d+)', basename)
            if match:
                cameras.add(f"cam{match.group(1)}")
            # 也支援直接用影片檔名作為相機名稱
            elif any(basename.lower().endswith(ext) for ext in ['.mp4', '.avi', '.mov', '.mkv']):
                # 使用不含副檔名的檔名作為相機名稱
                name_without_ext = os.path.splitext(basename)[0]
                cameras.add(name_without_ext)
        
        self.combo_camera.clear()
        self.combo_camera.addItem("全部")
        for cam in sorted(cameras):
            self.combo_camera.addItem(cam)
        
        self.filter_images()
        
        # ✅ Feature 2: 載入模型 (資料夾 + 腳本目錄)
        self.model_files = []
        # 從資料夾載入
        for file in os.listdir(self.folder_path):
            if file.endswith('.pt'):
                self.model_files.append(os.path.join(self.folder_path, file))
        
        # 從腳本所在目錄載入
        script_dir = os.path.dirname(os.path.abspath(__file__))
        if os.path.exists(script_dir):
            for file in os.listdir(script_dir):
                if file.endswith('.pt'):
                    model_path = os.path.join(script_dir, file)
                    # 避免重複添加
                    if model_path not in self.model_files:
                        self.model_files.append(model_path)
        
        self.list_models.clear()
        for model in self.model_files:
            item_name = os.path.basename(model)
            # 如果來自腳本目錄，添加標記
            if os.path.dirname(model) == script_dir:
                item_name = f"[本地] {item_name}"
            self.list_models.addItem(item_name)
        
        self.lbl_model_count.setText(f"模型數量: {len(self.model_files)}")
    
    def filter_images(self):
        """根據相機過濾圖片"""
        self.list_images.clear()
        camera_filter = self.combo_camera.currentText()
        
        filtered_images = []
        if camera_filter == "全部":
            filtered_images = self.image_files
        else:
            filtered_images = [img for img in self.image_files if camera_filter in os.path.basename(img)]
        
        for img in filtered_images:
            self.list_images.addItem(os.path.basename(img))
    
    def preview_image(self, item):
        """預覽選中的圖片或影片"""
        img_name = item.text()
        img_path = os.path.join(self.folder_path, img_name)
        self.stop_video_preview()
        
        # Auto-switch camera filter if needed
        # Assuming camera name is part of the filename, e.g., "cam1_..."
        # We check if the current filter matches the file. If not, try to find a matching filter.
        current_filter = self.combo_camera.currentText()
        if current_filter != "全部" and current_filter not in img_name:
            # Try to find a better filter
            for i in range(self.combo_camera.count()):
                filter_text = self.combo_camera.itemText(i)
                if filter_text != "全部" and filter_text in img_name:
                    self.combo_camera.setCurrentIndex(i)
                    break
        
        # 🔧 Bug Fix 2: 支援影片預覽
        video_extensions = ['.mp4', '.avi', '.mov', '.mkv']
        if any(img_path.lower().endswith(ext) for ext in video_extensions):
            # 如果是影片檔，提取第一幀顯示
            try:
                cap = cv2.VideoCapture(img_path)
                ret, frame = cap.read()
                cap.release()
                if ret and frame is not None:
                    self.image_preview.show_frame(frame)
                    self.apply_saved_crop_rect()
                else:
                    QMessageBox.warning(self, "錯誤", "無法讀取影片第一幀。")
            except Exception as e:
                QMessageBox.warning(self, "錯誤", f"影片預覽失敗: {str(e)}")
        else:
            # 圖片檔案
            self.image_preview.set_image(img_path)
            self.apply_saved_crop_rect()

    def get_track_color(self, track_id):
        """取得穩定且可重現的追蹤色彩 (MainWindow version)"""
        if track_id in self.track_colors:
            return self.track_colors[track_id]
        color = (
            60 + (37 * track_id) % 190,
            80 + (53 * track_id) % 160,
            100 + (29 * track_id) % 150
        )
        self.track_colors[track_id] = color
        return color
    
    def reset_selection(self):
        """重置選取區域"""
        self.image_preview.reset_selection()
        self.crop_rect = None
        self.saved_crop_rect = None
        QMessageBox.information(self, "提示", "已重置選取區域")

    def on_selection_changed(self, crop_rect):
        """同步 ImageLabel 的裁切結果"""
        if crop_rect is not None:
            self.crop_rect = crop_rect
            self.saved_crop_rect = crop_rect
        else:
            self.crop_rect = None

    def apply_saved_crop_rect(self):
        if self.saved_crop_rect and not self.image_preview.is_playback_mode:
            self.image_preview.set_crop_rect(self.saved_crop_rect)

    def update_overlay_option(self, key, state):
        self.overlay_config[key] = bool(state)

    def update_overlay_font(self, value):
        self.overlay_config['font_scale'] = float(value)

    def get_overlay_config_copy(self):
        return deepcopy(self.overlay_config)

    def get_timestamp_config(self):
        return {
            'color': self.timestamp_color,
            'size': self.timestamp_size
        }

    def get_selected_model_paths(self):
        """獲取選中的模型路徑"""
        selected = []
        for item in self.list_models.selectedItems():
            # 🔧 Bug Fix 3: 使用實際路徑而非顯示名稱
            # 獲取該項目在清單中的索引，然後從 model_files 獲取實際路徑
            index = self.list_models.row(item)
            if 0 <= index < len(self.model_files):
                selected.append(self.model_files[index])
        return selected

    def get_current_image_path(self):
        current_item = self.list_images.currentItem()
        if not current_item:
            return None
        return os.path.join(self.folder_path, current_item.text())

    def load_model_instance(self, model_path):
        model = self.loaded_models.get(model_path)
        if model is None:
            model = YOLO(model_path)
            self.loaded_models[model_path] = model
        return model

    def preview_model_prediction(self):
        image_path = self.get_current_image_path()
        if not image_path:
            QMessageBox.warning(self, "提示", "請先在圖片列表選擇一張圖片。")
            return
        model_paths = self.get_selected_model_paths()
        if not model_paths:
            QMessageBox.warning(self, "提示", "請至少選擇一個 YOLO 模型。")
            return
        model_path = model_paths[0]
        try:
            model = self.load_model_instance(model_path)
        except Exception as exc:
            QMessageBox.critical(self, "錯誤", f"模型載入失敗: {exc}")
            return
        
        img = cv2.imread(image_path)
        crop_rect = self.image_preview.get_crop_rect() or self.saved_crop_rect
        if crop_rect:
            x1, y1, x2, y2 = crop_rect
            img = img[y1:y2, x1:x2]
        results = model(img, verbose=False)
        detections = []
        for result in results:
            detections.extend(convert_result_to_detections(result, os.path.basename(model_path)))
        renderer = AnnotationRenderer(self.get_overlay_config_copy())
        annotated_img = renderer.render(img.copy(), detections, [], None)
        self.display_preview_frame(annotated_img)

    def display_preview_frame(self, frame):
        """顯示預覽幀（用於模型測試）"""
        if frame is None:
            return
        self.stop_video_preview()
        if frame.ndim == 2:
            h, w = frame.shape
            qimg = QImage(frame.data, w, h, w, QImage.Format_Grayscale8).copy()
        else:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, ch = frame_rgb.shape
            bytes_per_line = ch * w
            qimg = QImage(frame_rgb.data, w, h, bytes_per_line, QImage.Format_RGB888).copy()
        # ✅ Feature 3: 設置為播放模式，避免顯示紅色選取框
        self.image_preview.set_playback_mode(True)
        self.image_preview.base_pixmap = QPixmap.fromImage(qimg)
        self.image_preview.update_display_pixmap()
        # 不調用 apply_saved_crop_rect()，避免繪製錯誤的紅框

    def redraw_analyzed_video(self):
        if not self.last_analysis_data:
            QMessageBox.warning(self, "提示", "目前沒有可重新繪製的分析結果。")
            return
        output_path, _ = QFileDialog.getSaveFileName(
            self, "重新輸出分析影片", self.folder_path, "MP4 Files (*.mp4)"
        )
        if not output_path:
            return
        excel_path = output_path.replace('.mp4', '_analysis.xlsx')
        params = {
            'analysis_data': deepcopy(self.last_analysis_data),
            'output_path': output_path,
            'fps': self.spin_fps.value(),
            'timestamp_config': self.get_timestamp_config(),
            'excel_path': excel_path,
            'overlay_config': self.get_overlay_config_copy()
        }
        self.run_process('redraw_video', params)
    
    def redraw_to_original_images(self):
        """重新繪製分析結果到原始圖片上 (未裁切的完整大小圖片) 並生成影片"""
        if not self.last_analysis_data:
            QMessageBox.warning(self, "提示", "目前沒有可重新繪製的分析結果。")
            return
        
        frames = self.last_analysis_data.get('frames', [])
        if not frames:
            QMessageBox.warning(self, "錯誤", "分析資料中沒有可用的幀資訊。")
            return
        
        # 選擇輸出影片路徑
        output_path, _ = QFileDialog.getSaveFileName(
            self, "儲存原始圖片分析影片", self.folder_path, "MP4 Files (*.mp4)"
        )
        
        if not output_path:
            return
        
        # Excel 檔案路徑
        excel_path = output_path.replace('.mp4', '_original_analysis.xlsx')
        
        params = {
            'analysis_data': self.last_analysis_data,
            'output_path': output_path,
            'fps': self.spin_fps.value(),
            'timestamp_config': self.get_timestamp_config(),
            'excel_path': excel_path,
            'overlay_config': self.get_overlay_config_copy()
        }
        
        self.run_process('redraw_original_video', params)

    def select_timestamp_color(self):
        """選擇時間戳記顏色"""
        color = QColorDialog.getColor()
        if color.isValid():
            # BGR 格式
            self.timestamp_color = (color.blue(), color.green(), color.red())
            self.btn_color.setStyleSheet(f"background-color: {color.name()};")
    
    def update_timestamp_size(self, value):
        """更新時間戳記大小"""
        self.timestamp_size = value
    
    def get_selected_images(self):
        """獲取當前過濾後的圖片列表"""
        camera_filter = self.combo_camera.currentText()
        if camera_filter == "全部":
            return self.image_files
        else:
            return [img for img in self.image_files if camera_filter in os.path.basename(img)]
    
    def create_timelapse_video(self):
        """創建時間序列影片"""
        images = self.get_selected_images()
        if not images:
            QMessageBox.warning(self, "警告", "沒有找到圖片！")
            return
        
        # 獲取裁切區域
        selected_crop = self.image_preview.get_crop_rect() or self.saved_crop_rect
        self.crop_rect = selected_crop
        
        output_path, _ = QFileDialog.getSaveFileName(
            self, "儲存影片", self.folder_path, "MP4 Files (*.mp4)"
        )
        
        if not output_path:
            return
        
        params = {
            'images': images,
            'output_path': output_path,
            'crop_rect': selected_crop,
            'fps': self.spin_fps.value(),
            'timestamp_config': self.get_timestamp_config()
        }
        
        self.run_process('create_video', params)
    
    def create_analyzed_video(self):
        """創建分析標註影片"""
        images = self.get_selected_images()
        if not images:
            QMessageBox.warning(self, "警告", "沒有找到圖片！")
            return
        
        # 獲取選中的模型
        selected_models = self.get_selected_model_paths()
        
        if not selected_models:
            QMessageBox.warning(self, "警告", "請至少選擇一個 YOLO 模型！")
            return
        
        # 獲取裁切區域
        selected_crop = self.image_preview.get_crop_rect() or self.saved_crop_rect
        self.crop_rect = selected_crop
        
        output_path, _ = QFileDialog.getSaveFileName(
            self, "儲存分析影片", self.folder_path, "MP4 Files (*.mp4)"
        )
        
        if not output_path:
            return
        
        # Excel 檔案路徑
        excel_path = output_path.replace('.mp4', '_analysis.xlsx')
        
        params = {
            'images': images,
            'output_path': output_path,
            'crop_rect': selected_crop,
            'fps': self.spin_fps.value(),
            'timestamp_config': self.get_timestamp_config(),
            'models': selected_models,
            'excel_path': excel_path,
            'overlay_config': self.get_overlay_config_copy(),
            'tracker_type': self.tracker_type,
            'tracker_config': {
                'max_age': 30,
                'min_hits': 3,
                'iou_threshold': 0.3
            },
            'class_similarity_map': self.class_similarity_map
        }
        
        self.run_process('analyze_video', params)
    
    def run_process(self, task_type, params):
        """執行處理任務"""
        self.stop_video_preview()
        if task_type in ('analyze_video', 'redraw_video') and self.btn_redraw_video:
            self.btn_redraw_video.setEnabled(False)
        self.progress_bar.setValue(0)
        self.thread = ProcessThread(task_type, params)
        self.thread.progress.connect(self.progress_bar.setValue)
        self.thread.status.connect(self.lbl_status.setText)
        self.thread.finished.connect(self.on_process_finished)
        self.thread.start()
    
    def on_process_finished(self, success, message):
        """處理完成回調"""
        if success:
            QMessageBox.information(self, "完成", message)
            if self.thread and self.thread.task_type in ('analyze_video', 'redraw_video'):
                payload = getattr(self.thread, 'output_payload', None)
                if payload:
                    self.last_analysis_data = payload
                    if self.btn_redraw_video:
                        self.btn_redraw_video.setEnabled(True)
                    if self.btn_redraw_to_original:
                        self.btn_redraw_to_original.setEnabled(True)
                    if self.btn_play_analysis:
                        self.btn_play_analysis.setEnabled(True)
                output_path = self.thread.params.get('output_path')
                if output_path:
                    self.start_video_preview(output_path)
        else:
            QMessageBox.critical(self, "錯誤", message)
        self.lbl_status.setText("就緒")
        self.thread = None


def main():
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()
