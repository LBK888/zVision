from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict
import json
import os
from pathlib import Path
import sqlite3
import subprocess
import sys
import threading
import traceback

import cv2
import numpy as np
from PyQt6.QtCore import Qt, QThread, QTimer, pyqtSignal, QPointF, QRectF
from PyQt6.QtGui import QColor, QFont, QFontDatabase, QImage, QPainter, QPen, QPolygonF
from PyQt6.QtWidgets import (
    QApplication, QCheckBox, QComboBox, QDialog, QDialogButtonBox, QDoubleSpinBox,
    QFileDialog, QFormLayout, QFrame, QGridLayout, QGroupBox, QHBoxLayout, QLabel,
    QLineEdit, QListWidget, QMainWindow, QMessageBox, QPlainTextEdit, QProgressBar,
    QPushButton, QScrollArea, QSizePolicy, QSlider, QSpinBox, QSplitter,
    QStackedWidget, QTableWidget, QTableWidgetItem, QVBoxLayout, QWidget,
)

from .acquisition import CaptureSession
from .analysis import AnalysisJob, discover_sequences, redraw_cache, render
from .config import AppConfig, CameraConfig, ModelConfig, ROI, geometry_override
from .geometry import GeometryProcessor
from .i18n import tr
from .storage import atomic_json, estimate_storage, export_excel, read_image


STYLE = """
QWidget { color: #24343b; font-family: 'Microsoft JhengHei UI', 'Yu Gothic UI', 'Segoe UI'; font-size: 13px; }
QMainWindow, QDialog { background: #f4f6f3; }
QFrame#sidebar { background: #20383c; border: 0; }
QFrame#sidebar QLabel { color: #f1f6f3; background: transparent; }
QLabel#brand { font-size: 23px; font-weight: 700; letter-spacing: 1px; }
QLabel#muted { color: #6c7f7a; }
QLabel#title { font-size: 25px; font-weight: 600; }
QLabel#metric { font-size: 24px; font-weight: 600; color: #226b60; }
QLabel#badge { color: #257362; background: #dcece5; border-radius: 5px; padding: 6px 11px; }
QListWidget#navigation { background: transparent; border: none; color: #bacdc7; outline: none; }
QListWidget#navigation::item { padding: 16px 11px; margin: 3px 0; border-radius: 6px; }
QListWidget#navigation::item:selected { background: #345751; color: white; }
QListWidget#navigation::item:hover { background: #2d494a; }
QGroupBox { background: #ffffff; border: 1px solid #dce3de; border-radius: 8px; margin-top: 20px; padding: 20px 15px 14px; font-weight: 600; }
QGroupBox::title { subcontrol-origin: margin; left: 16px; padding: 0 5px; }
QPushButton { background: #ffffff; border: 1px solid #cdd9d3; border-radius: 5px; padding: 9px 13px; }
QPushButton:hover { border-color: #388376; background: #eaf3ee; }
QPushButton:pressed { background: #dcebe3; }
QPushButton:disabled { color: #9ba8a2; background: #edf0ec; border-color: #dfe5df; }
QPushButton#primary { background: #276e61; border-color: #276e61; color: white; font-weight: 600; }
QPushButton#primary:hover { background: #205e53; }
QPushButton#danger { color: #a35234; border-color: #d6b9a8; }
QPushButton#primary:disabled { background: #a7bdb3; border-color: #a7bdb3; }
QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox, QPlainTextEdit { background: #fafcf9; border: 1px solid #cedad2; border-radius: 4px; padding: 7px; selection-background-color: #c1e3d5; }
QLineEdit:focus, QSpinBox:focus, QDoubleSpinBox:focus, QComboBox:focus { border: 1px solid #2d8372; }
QComboBox::drop-down { border: 0; width: 22px; }
QCheckBox { padding: 5px 0; }
QCheckBox::indicator { width: 16px; height: 16px; }
QScrollArea { border: 0; background: transparent; }
QScrollArea > QWidget > QWidget { background: #f4f6f3; }
QTableWidget, QListWidget { background: #fff; border: 1px solid #dce3de; border-radius: 5px; gridline-color: #e5ebe5; }
QHeaderView::section { background: #eaf0e9; padding: 8px; border: 0; }
QProgressBar { background: #e3ebe4; border: 0; border-radius: 3px; height: 7px; text-align: center; }
QProgressBar::chunk { background: #398775; border-radius: 3px; }
QSplitter::handle { background: #dde5de; width: 1px; }
QToolTip { background: #20383c; color: white; padding: 7px; border: none; }
"""


class Canvas(QWidget):
    selectionChanged = pyqtSignal(object)
    perspectiveChanged = pyqtSignal(object)
    hovered = pyqtSignal(object)

    def __init__(self, empty_text):
        super().__init__()
        self.setMinimumSize(340, 260)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setMouseTracking(True)
        self.empty_text = empty_text
        self.image = None
        self.mode = "rectangle"
        self.selection = None
        self.points = []
        self.start_point = None
        self.drag_point = None
        self.rows, self.cols = 1, 1
        self.editable = True

    def set_image(self, image):
        self.image = np.ascontiguousarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        self.update()

    def image_rect(self):
        if self.image is None:
            return QRectF(0, 0, self.width(), self.height())
        h, w = self.image.shape[:2]
        scale = min((self.width()-24)/w, (self.height()-24)/h)
        return QRectF((self.width()-w*scale)/2, (self.height()-h*scale)/2, w*scale, h*scale)

    def normalized(self, point):
        rect = self.image_rect()
        return [min(1, max(0, (point.x()-rect.x())/rect.width())), min(1, max(0, (point.y()-rect.y())/rect.height()))]

    def display_point(self, normalized):
        r = self.image_rect()
        return QPointF(r.x()+normalized[0]*r.width(), r.y()+normalized[1]*r.height())

    def paintEvent(self, event):
        p = QPainter(self)
        p.fillRect(self.rect(), QColor("#15292d"))
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        if self.image is None:
            p.setPen(QColor("#a6c0b6"))
            p.drawText(self.rect().adjusted(25, 25, -25, -25), Qt.AlignmentFlag.AlignCenter | Qt.TextFlag.TextWordWrap, self.empty_text)
            return
        h, w = self.image.shape[:2]
        p.drawImage(self.image_rect(), QImage(self.image.data, w, h, self.image.strides[0], QImage.Format.Format_RGB888))
        p.setPen(QPen(QColor("#f0c56c"), 2))
        if self.selection:
            x, y, rw, rh = self.selection
            a, b = self.display_point([x, y]), self.display_point([x+rw, y+rh])
            r = QRectF(a, b)
            if self.mode == "ellipse":
                p.drawEllipse(r)
            else:
                p.drawRect(r)
            for col in range(1, self.cols):
                px = r.x()+r.width()*col/self.cols
                p.drawLine(QPointF(px, r.y()), QPointF(px, r.bottom()))
            for row in range(1, self.rows):
                py = r.y()+r.height()*row/self.rows
                p.drawLine(QPointF(r.x(), py), QPointF(r.right(), py))
        if self.points:
            polygon = QPolygonF([self.display_point(x) for x in self.points])
            p.setPen(QPen(QColor("#80d9c2"), 2))
            p.drawPolygon(polygon) if len(self.points) == 4 else p.drawPolyline(polygon)
            for i, point in enumerate(polygon):
                p.setBrush(QColor("#80d9c2"))
                p.drawEllipse(point, 5, 5)
                p.drawText(point+QPointF(8, -8), str(i+1))

    def mousePressEvent(self, event):
        if self.image is None or not self.editable or not self.image_rect().contains(event.position()):
            return
        point = self.normalized(event.position())
        if self.mode == "perspective":
            for i, p in enumerate(self.points):
                if (self.display_point(p)-event.position()).manhattanLength() < 18:
                    self.drag_point = i
                    return
            if len(self.points) == 4:
                self.points = []
            self.points.append(point)
            if len(self.points) == 4:
                self.perspectiveChanged.emit(self.points)
        else:
            self.start_point = point
            self.selection = [*point, 0, 0]
        self.update()

    def mouseMoveEvent(self, event):
        if self.image is None:
            return
        point = self.normalized(event.position())
        self.hovered.emit(point)
        if self.drag_point is not None:
            self.points[self.drag_point] = point
            self.update()
        if self.start_point:
            a = self.start_point
            self.selection = [min(a[0], point[0]), min(a[1], point[1]), abs(a[0]-point[0]), abs(a[1]-point[1])]
            self.update()

    def mouseReleaseEvent(self, event):
        if self.drag_point is not None:
            self.drag_point = None
            self.perspectiveChanged.emit(self.points)
        if self.start_point:
            self.start_point = None
            if self.selection[2] > .001 and self.selection[3] > .001:
                self.selectionChanged.emit(self.selection)


class Task(QThread):
    statusEvent = pyqtSignal(object)
    preview = pyqtSignal(object)
    result = pyqtSignal(object)
    failed = pyqtSignal(str)

    def __init__(self, function):
        super().__init__()
        self.function = function

    def run(self):
        try:
            self.result.emit(self.function(self))
        except Exception:
            self.failed.emit(traceback.format_exc())


class MainWindow(QMainWindow):
    def __init__(self, config=None):
        super().__init__()
        if not QFontDatabase.families() and os.name == "nt":
            for font in ["msjh.ttc", "segoeui.ttf", "YuGothR.ttc"]:
                path = Path(os.environ.get("WINDIR", "C:/Windows"))/"Fonts"/font
                if path.exists():
                    QFontDatabase.addApplicationFont(str(path))
        self.config = config or AppConfig()
        self.current_camera_index = 0
        first = self.config.cameras[0]
        self.geo = geometry_override(self.config.geometry, self.config.geometry.per_camera.get(first.id, {}))
        self.task = None
        self.job = None
        self.hardware_blocked = False
        self.freeze_preview = False
        self.last_run = None
        self.input_root = None
        self.current_image = None
        self.stream_frames = {}
        self.review_run = None
        self.review_rows = []
        self.review_record = None
        self.preview_frames = {}
        self.preview_lock = threading.Lock()
        self.preview_timer = QTimer(self)
        self.preview_timer.timeout.connect(self.drain_preview)
        self.preview_timer.start(100)
        self.playback = QTimer(self)
        self.playback.timeout.connect(self.next_frame)
        self.resize(1500, 960)
        self.setMinimumSize(1100, 720)
        self.build()

    def t(self, key):
        return tr(key, self.config.language)

    def build(self):
        self.setWindowTitle(self.t("app"))
        self.setStyleSheet(STYLE)
        central = QWidget()
        root = QHBoxLayout(central)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)
        sidebar = QFrame()
        sidebar.setObjectName("sidebar")
        sidebar.setFixedWidth(234)
        nav_layout = QVBoxLayout(sidebar)
        nav_layout.setContentsMargins(18, 30, 18, 22)
        brand = QLabel("UVC LAB")
        brand.setObjectName("brand")
        nav_layout.addWidget(brand)
        tagline = QLabel(self.t("tagline"))
        tagline.setWordWrap(True)
        nav_layout.addWidget(tagline)
        nav_layout.addSpacing(28)
        self.nav = QListWidget()
        self.nav.setObjectName("navigation")
        for key in ["devices", "geometry", "plan", "analysis", "hardware", "review"]:
            self.nav.addItem(self.t(key))
        nav_layout.addWidget(self.nav, 1)
        self.language = QComboBox()
        for name, code in [("繁體中文（台灣）", "zh_TW"), ("English", "en"), ("日本語", "ja")]:
            self.language.addItem(name, code)
        self.language.setCurrentIndex(self.language.findData(self.config.language))
        self.language.currentIndexChanged.connect(self.change_language)
        nav_layout.addWidget(self.language)
        root.addWidget(sidebar)
        main = QWidget()
        outer = QVBoxLayout(main)
        outer.setContentsMargins(24, 24, 24, 16)
        header = QHBoxLayout()
        self.page_title = QLabel(self.t("devices"))
        self.page_title.setObjectName("title")
        header.addWidget(self.page_title, 1)
        self.badge = QLabel(self.t("ready"))
        self.badge.setObjectName("badge")
        header.addWidget(self.badge)
        outer.addLayout(header)
        toolbar = QHBoxLayout()
        self.action_buttons = []
        for key, handler in [("load", self.load_settings), ("save", self.save_settings), ("advanced", self.edit_json)]:
            toolbar.addWidget(self.button(key, handler))
        toolbar.addStretch()
        self.live = QCheckBox(self.t("live"))
        self.live.setChecked(self.config.capture.live)
        self.live.toggled.connect(self.toggle_live)
        toolbar.addWidget(self.live)
        outer.addLayout(toolbar)
        split = QSplitter()
        self.pages = QStackedWidget()
        self.page_controls = []
        for creator in [self.devices_page, self.geometry_page, self.plan_page, self.analysis_page, self.hardware_page, self.review_page]:
            widget = creator()
            area = QScrollArea()
            area.setWidgetResizable(True)
            area.setWidget(widget)
            area.setMinimumWidth(420)
            self.pages.addWidget(area)
            self.page_controls.append(widget)
        split.addWidget(self.pages)
        preview = QWidget()
        pv = QVBoxLayout(preview)
        pv.setContentsMargins(18, 0, 0, 0)
        label = QLabel(self.t("preview_title"))
        label.setFont(QFont("", 13, QFont.Weight.DemiBold))
        pv.addWidget(label)
        self.stream_selector = QComboBox()
        self.stream_selector.setToolTip(self.t("select_stream"))
        self.stream_selector.currentTextChanged.connect(self.select_preview)
        pv.addWidget(self.stream_selector)
        self.canvas = Canvas(self.t("empty_preview"))
        self.canvas.perspectiveChanged.connect(self.set_perspective)
        self.canvas.hovered.connect(self.hover_object)
        pv.addWidget(self.canvas, 1)
        self.preview_caption = QLabel(self.t("empty_preview").split("\n")[1])
        self.preview_caption.setObjectName("muted")
        self.preview_caption.setWordWrap(True)
        pv.addWidget(self.preview_caption)
        self.metrics = {}
        metrics = QGridLayout()
        for i, key in enumerate(["elapsed", "saved", "dropped", "queue"]):
            block = QWidget()
            box = QVBoxLayout(block)
            box.setContentsMargins(8, 12, 8, 8)
            title = QLabel(self.t(key))
            title.setObjectName("muted")
            value = QLabel("—")
            value.setObjectName("metric")
            box.addWidget(title)
            box.addWidget(value)
            metrics.addWidget(block, i//2, i%2)
            self.metrics[key] = value
        pv.addLayout(metrics)
        self.progress = QProgressBar()
        self.progress.setRange(0, 100)
        self.progress.setValue(0)
        self.progress.setTextVisible(False)
        pv.addWidget(self.progress)
        split.addWidget(preview)
        split.setSizes([650, 540])
        outer.addWidget(split, 1)
        footer = QHBoxLayout()
        self.preview_button = self.button("preview", self.start_preview)
        self.check_button = self.button("preflight", self.preflight)
        self.start_button = self.button("start", self.start_capture, "primary")
        self.stop_button = self.button("stop", self.stop_job, "danger")
        self.stop_button.setEnabled(False)
        for button in [self.preview_button, self.check_button, self.start_button, self.stop_button]:
            footer.addWidget(button)
        outer.addLayout(footer)
        root.addWidget(main, 1)
        self.setCentralWidget(central)
        self.nav.currentRowChanged.connect(self.change_page)
        self.nav.setCurrentRow(0)
        self.update_camera_list()

    def button(self, key, handler, style=None):
        b = QPushButton(self.t(key))
        if style:
            b.setObjectName(style)
        b.clicked.connect(handler)
        return b

    def page(self, hint=None):
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setContentsMargins(0, 0, 12, 20)
        layout.setSpacing(14)
        if hint:
            self.hint(layout, hint)
        return widget, layout

    def hint(self, layout, key):
        label = QLabel(self.t(key))
        label.setWordWrap(True)
        label.setObjectName("muted")
        layout.addWidget(label)
        return label

    def group(self, layout, key):
        group = QGroupBox(self.t(key))
        form = QFormLayout(group)
        form.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow)
        form.setVerticalSpacing(10)
        layout.addWidget(group)
        return form

    def resolve(self, path):
        names = path.split(".")
        if names[0] == "camera":
            obj = self.config.cameras[self.current_camera_index]
        elif names[0] == "geo":
            obj = self.geo
        else:
            obj = self.config
            names.insert(0, "config")
        for name in names[1:-1]:
            obj = getattr(obj, name)
        return obj, names[-1]

    def get_value(self, path):
        obj, name = self.resolve(path)
        return getattr(obj, name)

    def set_value(self, path, value):
        obj, name = self.resolve(path)
        setattr(obj, name, value)

    def line(self, form, key, path):
        edit = QLineEdit(str(self.get_value(path)))
        edit.textChanged.connect(lambda value: self.set_value(path, value))
        form.addRow(self.t(key), edit)
        return edit

    def spin(self, form, key, path, low=0, high=1e6, decimals=2, factor=1):
        edit = QDoubleSpinBox()
        edit.setDecimals(decimals)
        edit.setRange(low, high)
        edit.setValue((self.get_value(path) or 0)/factor)
        edit.valueChanged.connect(lambda value: self.set_value(path, int(value*factor) if decimals == 0 else value*factor))
        form.addRow(self.t(key), edit)
        return edit

    def check(self, form, key, path):
        box = QCheckBox(self.t(key))
        box.setChecked(self.get_value(path))
        box.toggled.connect(lambda checked: self.set_value(path, checked))
        form.addRow(box)
        return box

    def combo(self, form, key, path, options):
        edit = QComboBox()
        for value, label in options:
            edit.addItem(self.t(label) if label else str(value), value)
        edit.setCurrentIndex(max(0, edit.findData(self.get_value(path))))
        edit.currentIndexChanged.connect(lambda i: self.set_value(path, edit.itemData(i)))
        form.addRow(self.t(key), edit)
        return edit

    def devices_page(self):
        widget, layout = self.page("fps_hint")
        tools = QHBoxLayout()
        tools.addWidget(self.button("scan", self.scan_cameras))
        tools.addWidget(self.button("demo", self.use_demo))
        layout.addLayout(tools)
        self.camera_selector = QComboBox()
        self.camera_selector.currentIndexChanged.connect(self.change_camera)
        layout.addWidget(self.camera_selector)
        actions = QHBoxLayout()
        actions.addWidget(self.button("add", self.add_camera))
        actions.addWidget(self.button("remove", self.remove_camera))
        layout.addLayout(actions)
        self.camera_details = QWidget()
        self.camera_form_layout = QVBoxLayout(self.camera_details)
        self.camera_form_layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.camera_details)
        layout.addWidget(self.button("system_test", self.system_benchmark))
        self.measurement = QPlainTextEdit()
        self.measurement.setReadOnly(True)
        self.measurement.setMaximumHeight(180)
        layout.addWidget(self.measurement)
        layout.addStretch()
        return widget

    def camera_fields(self):
        while self.camera_form_layout.count():
            item = self.camera_form_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        form = self.group(self.camera_form_layout, "camera")
        self.line(form, "name", "camera.name")
        source = self.line(form, "source", "camera.source")
        source.textChanged.disconnect()
        source.textChanged.connect(lambda x: self.set_value("camera.source", int(x) if x.isdigit() else x))
        self.combo(form, "backend", "camera.backend", [(x, None) for x in ["auto", "dshow", "msmf", "v4l2", "avfoundation"]])
        form.addRow(self.button("modes", self.read_modes))
        self.mode_selector = QComboBox()
        self.mode_selector.currentIndexChanged.connect(self.select_mode)
        form.addRow(self.t("mode"), self.mode_selector)
        self.width_control = self.spin(form, "width", "camera.width", 2, 20000, 0)
        self.height_control = self.spin(form, "height", "camera.height", 2, 20000, 0)
        self.fps_control = self.spin(form, "fps", "camera.fps", .01, 1000, 3)
        self.fourcc_control = self.combo(form, "format", "camera.fourcc", [(x, None) for x in ["MJPG", "YUY2", "UYVY", "NV12", "RGB3", "BGR3", "H264", "HEVC"]])
        form.addRow(self.button("benchmark", self.benchmark_camera))
        controls = self.group(self.camera_form_layout, "controls")
        lock = self.check(controls, "lock", "camera.scientific_lock")
        auto_controls = [self.check(controls, key, path) for key, path in [("auto_exposure", "camera.auto_exposure"), ("auto_focus", "camera.auto_focus"), ("auto_wb", "camera.auto_white_balance")]]
        def update_locked(checked):
            for control in auto_controls:
                control.setEnabled(not checked)
        lock.toggled.connect(update_locked)
        update_locked(lock.isChecked())
        for key, path, low, high in [("exposure", "exposure", -10000, 1e6), ("focus", "focus", -1000, 1e6),
                                      ("wb", "white_balance", 0, 20000), ("gain", "gain", -100, 1e6)]:
            self.spin(controls, key, "camera."+path, low, high)
        self.spin(controls, "warmup", "camera.warmup_s", 0, 120)
        self.check(controls, "strict", "camera.strict_controls")

    def geometry_page(self):
        widget, layout = self.page("roi_hint")
        layout.addWidget(self.button("open_image", self.open_image))
        self.geometry_mode = QComboBox()
        for key in ["rectangle", "ellipse", "perspective"]:
            self.geometry_mode.addItem(self.t(key), key)
        self.geometry_mode.currentIndexChanged.connect(self.geometry_mode_changed)
        layout.addWidget(self.geometry_mode)
        form = self.group(layout, "geometry")
        self.roi_name = QLineEdit("plate")
        form.addRow(self.t("roi_name"), self.roi_name)
        self.roi_rows, self.roi_cols = QSpinBox(), QSpinBox()
        for control in [self.roi_rows, self.roi_cols]:
            control.setRange(1, 128)
            control.setValue(1)
            control.valueChanged.connect(self.grid_changed)
        form.addRow(self.t("rows"), self.roi_rows)
        form.addRow(self.t("cols"), self.roi_cols)
        presets = QHBoxLayout()
        for rows, cols in [(1, 1), (2, 3), (3, 4), (8, 12)]:
            b = QPushButton(f"{cols} × {rows}")
            b.clicked.connect(lambda checked=False, r=rows, c=cols: (self.roi_rows.setValue(r), self.roi_cols.setValue(c)))
            presets.addWidget(b)
        form.addRow(presets)
        form.addRow(self.button("add_roi", self.add_roi))
        form.addRow(self.button("full_roi", self.full_roi))
        self.roi_list = QListWidget()
        self.roi_list.setMaximumHeight(140)
        self.roi_list.currentRowChanged.connect(self.select_roi)
        form.addRow(self.roi_list)
        form.addRow(self.button("remove", self.remove_roi))
        self.stabilize_control = self.check(form, "stabilize", "geo.stabilize")
        self.max_shift_control = self.spin(form, "max_shift", "geo.max_shift_px", .1, 100)
        form.addRow(self.button("clear_perspective", self.clear_perspective))
        form.addRow(self.button("show_corrected", self.show_corrected))
        self.per_camera_box = QCheckBox(self.t("per_camera"))
        self.per_camera_box.setChecked(self.config.cameras[self.current_camera_index].id in self.config.geometry.per_camera)
        form.addRow(self.per_camera_box)
        form.addRow(self.button("geometry_save", self.save_geometry))
        self.update_roi_list()
        layout.addStretch()
        return widget

    def plan_page(self):
        widget, layout = self.page()
        f = self.group(layout, "plan")
        self.line(f, "experiment", "experiment")
        notes = QPlainTextEdit(self.config.notes)
        notes.setMaximumHeight(90)
        notes.textChanged.connect(lambda: setattr(self.config, "notes", notes.toPlainText()))
        f.addRow(self.t("notes"), notes)
        self.combo(f, "capture_mode", "capture.mode", [("timelapse", "timelapse"), ("continuous", "continuous")])
        self.spin(f, "interval", "capture.interval_s", .01, 1e9, 3)
        self.duration = QDoubleSpinBox()
        self.duration.setRange(.01, 1e8)
        self.duration.setValue((self.config.capture.duration_s or 3600)/60)
        self.duration.valueChanged.connect(lambda value: setattr(self.config.capture, "duration_s", value*60) if not self.unlimited.isChecked() else None)
        self.unlimited = QCheckBox(self.t("unlimited"))
        self.unlimited.setChecked(self.config.capture.duration_s is None)
        self.duration.setEnabled(not self.unlimited.isChecked())
        def unlimited(checked):
            self.config.capture.duration_s = None if checked else self.duration.value()*60
            self.duration.setEnabled(not checked)
        self.unlimited.toggled.connect(unlimited)
        f.addRow(self.t("duration"), self.duration)
        f.addRow(self.unlimited)
        s = self.group(layout, "output")
        self.output_edit = self.line(s, "output", "storage.directory")
        s.addRow(self.button("browse", self.choose_output))
        self.check(s, "raw", "storage.raw")
        self.check(s, "processed", "storage.processed")
        self.combo(s, "image_format", "storage.image_format", [(x, None) for x in ["png", "tiff", "jpg"]])
        self.check(s, "video", "storage.video")
        self.combo(s, "container", "storage.video_container", [(x, None) for x in ["mp4", "avi"]])
        self.spin(s, "video_fps", "storage.video_fps", .01, 500, 3)
        self.spin(s, "crf", "storage.video_crf", 0, 51, 0)
        self.spin(s, "segment", "storage.segment_s", 1, 86400)
        self.spin(s, "reserve", "capture.min_free_gb", 0, 1e6)
        self.hint(layout, "space_hint")
        layout.addWidget(self.button("space", self.preflight))
        self.space_label = QLabel("—")
        self.space_label.setWordWrap(True)
        layout.addWidget(self.space_label)
        layout.addStretch()
        return widget

    def analysis_page(self):
        widget, layout = self.page("model_hint")
        self.models = QTableWidget(0, 3)
        self.models.setHorizontalHeaderLabels([self.t("model_path"), self.t("task"), self.t("confidence")])
        self.models.horizontalHeader().setStretchLastSection(True)
        self.models.setColumnWidth(0, 245)
        self.models.setMaximumHeight(160)
        self.models.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        layout.addWidget(self.models)
        form = self.group(layout, "add_model")
        self.model_path = QLineEdit("yolo26n-pose.pt")
        form.addRow(self.t("model_path"), self.model_path)
        form.addRow(self.button("browse", self.choose_model))
        self.model_task = QComboBox()
        self.model_task.addItems(["detect", "segment", "pose"])
        self.model_task.setCurrentText("pose")
        form.addRow(self.t("task"), self.model_task)
        buttons = QHBoxLayout()
        buttons.addWidget(self.button("add_model", self.add_model))
        buttons.addWidget(self.button("remove", self.remove_model))
        form.addRow(buttons)
        for model in self.config.analysis.models:
            self.insert_model(model)
        form = self.group(layout, "analysis")
        self.combo(form, "input_images", "analysis.input_images", [("processed", "input_processed"), ("raw", "input_raw")])
        self.check(form, "use_capture_geometry", "analysis.use_capture_geometry")
        self.line(form, "device", "analysis.device")
        self.combo(form, "consensus", "analysis.consensus", [(key, key) for key in ["none", "temporal", "rotation"]])
        self.spin(form, "vote_iou", "analysis.consensus_iou", .01, 1, 2)
        self.combo(form, "tracker", "analysis.tracker", [(x, None) for x in ["ukf", "sort", "bytetrack"]])
        self.spin(form, "iou", "analysis.iou_threshold", .01, 1, 2)
        self.spin(form, "max_age", "analysis.max_age", 1, 100000, 0)
        self.spin(form, "min_hits", "analysis.min_hits", 1, 1000, 0)
        self.spin(form, "sequence_interval", "analysis.sequence_interval_s", .001, 1e9, 3)
        self.spin(form, "calibration", "analysis.pixel_size_um", 0, 1e6, 6)
        self.check(form, "annotated_video", "storage.annotated_video")
        self.hint(layout, "consensus_hint")
        layout.addWidget(self.button("import_sequence", self.choose_sequence))
        self.sequence_label = QLabel(str(self.input_root or "—"))
        self.sequence_label.setWordWrap(True)
        layout.addWidget(self.sequence_label)
        layout.addWidget(self.button("run_analysis", lambda: self.start_analysis(True), "primary"))
        layout.addWidget(self.button("make_video", lambda: self.start_analysis(False)))
        layout.addStretch()
        return widget

    def hardware_page(self):
        widget, layout = self.page("api_hint")
        form = self.group(layout, "hardware")
        self.check(form, "api_enabled", "hardware.enabled")
        self.line(form, "host", "hardware.host")
        self.spin(form, "port", "hardware.port", 1, 65535, 0)
        token = self.line(form, "token", "hardware.token")
        token.setEchoMode(QLineEdit.EchoMode.PasswordEchoOnEdit)
        targets = QLineEdit(", ".join(self.config.hardware.targets))
        targets.textChanged.connect(lambda value: setattr(self.config.hardware, "targets", [x.strip() for x in value.split(",") if x.strip()]))
        form.addRow(self.t("targets"), targets)
        self.spin(form, "settle", "hardware.settle_s", 0, 3600)
        self.spin(form, "timeout", "hardware.timeout_s", .1, 86400)
        self.line(form, "sensor", "hardware.sensor_port")
        self.log = QPlainTextEdit()
        self.log.setReadOnly(True)
        self.log.setMaximumBlockCount(1500)
        self.log.setMinimumHeight(260)
        layout.addWidget(QLabel(self.t("logs")))
        layout.addWidget(self.log, 1)
        return widget

    def review_page(self):
        widget, layout = self.page()
        layout.addWidget(self.button("open_run", self.open_review))
        self.review_label = QLabel(str(self.review_run or "—"))
        self.review_label.setWordWrap(True)
        layout.addWidget(self.review_label)
        self.review_stream = QComboBox()
        self.review_stream.currentTextChanged.connect(self.load_review_stream)
        layout.addWidget(self.review_stream)
        self.seek = QSlider(Qt.Orientation.Horizontal)
        self.seek.setRange(0, 0)
        self.seek.valueChanged.connect(self.show_review_frame)
        layout.addWidget(self.seek)
        self.frame_label = QLabel("0 / 0")
        layout.addWidget(self.frame_label)
        layout.addWidget(self.button("play", self.toggle_playback))
        form = self.group(layout, "review")
        for key in self.config.analysis.overlays:
            box = QCheckBox(self.t(key))
            box.setChecked(self.config.analysis.overlays[key])
            box.toggled.connect(lambda value, key=key: self.update_overlay(key, value))
            form.addRow(box)
        self.check(form, "redraw_original", "analysis.redraw_original")
        self.object_details = QPlainTextEdit()
        self.object_details.setReadOnly(True)
        self.object_details.setMaximumHeight(170)
        layout.addWidget(self.object_details)
        layout.addWidget(self.button("redraw", self.redraw))
        layout.addWidget(self.button("export_excel", self.export_report))
        layout.addStretch()
        return widget

    def collect(self):
        self.save_geometry(silent=True)
        models = []
        for row in range(self.models.rowCount()):
            original = self.config.analysis.models[row] if row < len(self.config.analysis.models) else ModelConfig()
            m = deepcopy(original)
            m.path = self.models.item(row, 0).text().strip()
            m.task = self.models.item(row, 1).text().strip()
            m.confidence = float(self.models.item(row, 2).text())
            models.append(m)
        self.config.analysis.models = models
        if self.config.analysis.pixel_size_um == 0:
            self.config.analysis.pixel_size_um = None
        self.config.validate()
        return self.config

    def change_language(self):
        try:
            self.collect()
            language = self.language.currentData()
            self.config.language = language
            page = self.nav.currentRow()
            self.build()
            self.nav.setCurrentRow(page)
        except Exception as error:
            self.show_error(str(error))

    def change_page(self, index):
        if index < 0:
            return
        self.pages.setCurrentIndex(index)
        self.page_title.setText(self.nav.item(index).text())
        self.canvas.editable = index == 1 and (not self.task or self.is_preview_job())
        if index != 5:
            self.playback.stop()
        if index == 1:
            self.canvas.mode = self.geometry_mode.currentData()

    def update_camera_list(self):
        self.camera_selector.blockSignals(True)
        self.camera_selector.clear()
        for camera in self.config.cameras:
            self.camera_selector.addItem(f"{camera.id}  ·  {camera.name}")
        self.current_camera_index = min(self.current_camera_index, len(self.config.cameras)-1)
        self.camera_selector.setCurrentIndex(self.current_camera_index)
        self.camera_selector.blockSignals(False)
        self.camera_fields()
        self.load_camera_geometry()

    def load_camera_geometry(self):
        camera = self.config.cameras[self.current_camera_index]
        self.geo = geometry_override(self.config.geometry, self.config.geometry.per_camera.get(camera.id, {}))
        self.per_camera_box.setChecked(camera.id in self.config.geometry.per_camera)
        for control, value in [(self.stabilize_control, self.geo.stabilize), (self.max_shift_control, self.geo.max_shift_px)]:
            control.blockSignals(True)
            control.setChecked(value) if isinstance(control, QCheckBox) else control.setValue(value)
            control.blockSignals(False)
        self.canvas.points = deepcopy(self.geo.perspective)
        self.canvas.selection = None
        self.update_roi_list()

    def change_camera(self, index):
        if index < 0 or index >= len(self.config.cameras):
            return
        self.save_geometry(silent=True)
        self.current_camera_index = index
        camera = self.config.cameras[index]
        self.load_camera_geometry()
        self.camera_fields()
        if camera.id in self.stream_frames:
            self.show_image(self.stream_frames[camera.id], camera.id)

    def add_camera(self):
        self.save_geometry(silent=True)
        ids = {c.id for c in self.config.cameras}
        n = 0
        while f"camera-{n}" in ids:
            n += 1
        self.config.cameras.append(CameraConfig(id=f"camera-{n}", source=n))
        self.current_camera_index = len(self.config.cameras)-1
        self.update_camera_list()

    def remove_camera(self):
        if len(self.config.cameras) > 1:
            self.config.cameras.pop(self.current_camera_index)
            self.update_camera_list()

    def scan_cameras(self):
        def done(devices):
            if not devices:
                self.measurement.setPlainText(self.t("no_camera"))
                return
            self.config.cameras = [CameraConfig(id=f"camera-{i}", **d) for i, d in enumerate(devices)]
            self.current_camera_index = 0
            self.update_camera_list()
        self.start_task(lambda task: self.camera_subprocess("scan"), done)

    def use_demo(self):
        if self.task:
            return
        self.config.cameras = [CameraConfig(id=f"demo-{i+1}", source=f"demo-{i+1}", name=self.t("simulated"), warmup_s=0) for i in range(2)]
        self.current_camera_index = 0
        self.update_camera_list()
        self.measurement.setPlainText(self.t("simulated"))

    def camera_subprocess(self, action, camera=None):
        from .runtime import camera_command
        return camera_command(action, camera)

    def read_modes(self):
        try:
            camera = deepcopy(self.collect().cameras[self.current_camera_index])
            def done(result):
                self.mode_selector.clear()
                for mode in result["modes"]:
                    self.mode_selector.addItem(f"{mode['width']} × {mode['height']} · {mode['fps']:g} FPS · {mode['fourcc']} [{mode['evidence']}]", mode)
                self.measurement.setPlainText(result["evidence"])
            self.start_task(lambda task: self.camera_subprocess("modes", camera), done)
        except Exception as error:
            self.show_error(str(error))

    def select_mode(self, index):
        mode = self.mode_selector.itemData(index)
        if mode:
            self.width_control.setValue(mode["width"])
            self.height_control.setValue(mode["height"])
            self.fps_control.setValue(mode["fps"])
            self.fourcc_control.setCurrentText(mode["fourcc"])

    def benchmark_camera(self):
        try:
            camera = deepcopy(self.collect().cameras[self.current_camera_index])
            def done(result):
                summary = self.t("benchmark_summary").format(request=result["tested_fps_request"], actual=result["measured_fps"], suggested=result["suggested_fps"])
                self.measurement.setPlainText(summary+"\n\n"+json.dumps(result, ensure_ascii=False, indent=2))
                output = Path(self.config.storage.directory)/"benchmarks"
                output.mkdir(parents=True, exist_ok=True)
                from datetime import datetime
                atomic_json(output/f"{camera.id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json", result)
            self.start_task(lambda task: self.camera_subprocess("benchmark", camera), done)
        except Exception as error:
            self.show_error(str(error))

    def system_benchmark(self):
        try:
            cfg = deepcopy(self.collect())
            cfg.capture.mode = "continuous"
            cfg.capture.duration_s = 8
            cfg.hardware.enabled = False
            cfg.experiment += "_system-benchmark"
            self.launch_capture(cfg, False)
        except Exception as error:
            self.show_error(str(error))

    def geometry_mode_changed(self):
        if hasattr(self, "canvas"):
            self.canvas.mode = self.geometry_mode.currentData()
            self.canvas.update()

    def grid_changed(self):
        if hasattr(self, "canvas"):
            self.canvas.rows, self.canvas.cols = self.roi_rows.value(), self.roi_cols.value()
            self.canvas.update()

    def update_roi_list(self):
        if not hasattr(self, "roi_list"):
            return
        self.roi_list.blockSignals(True)
        self.roi_list.clear()
        for roi in self.geo.rois:
            self.roi_list.addItem(f"{roi.name}  ·  {roi.cols} × {roi.rows}  ·  {self.t(roi.shape)}")
        self.roi_list.blockSignals(False)

    def select_roi(self, index):
        if 0 <= index < len(self.geo.rois):
            roi = self.geo.rois[index]
            self.roi_name.setText(roi.name)
            self.roi_rows.setValue(roi.rows)
            self.roi_cols.setValue(roi.cols)
            self.geometry_mode.setCurrentIndex(self.geometry_mode.findData(roi.shape))
            self.canvas.selection = list(roi.rect)
            self.canvas.update()

    def add_roi(self):
        if not self.canvas.selection:
            self.show_error(self.t("select_roi"))
            return
        shape = self.geometry_mode.currentData()
        if shape == "perspective":
            shape = "rectangle"
        roi = ROI(self.roi_name.text().strip(), list(self.canvas.selection), shape, self.roi_rows.value(), self.roi_cols.value())
        existing = next((i for i, r in enumerate(self.geo.rois) if r.name == roi.name), None)
        if existing is not None:
            self.geo.rois[existing] = roi
        else:
            self.geo.rois.append(roi)
        self.update_roi_list()

    def remove_roi(self):
        index = self.roi_list.currentRow()
        if 0 <= index < len(self.geo.rois):
            self.geo.rois.pop(index)
            self.update_roi_list()

    def full_roi(self):
        self.canvas.selection = [0, 0, 1, 1]
        self.canvas.update()

    def set_perspective(self, points):
        self.geo.perspective = deepcopy(points)

    def clear_perspective(self):
        self.geo.perspective = []
        self.canvas.points = []
        self.canvas.update()

    def show_corrected(self):
        if self.current_image is None:
            return
        try:
            geo = deepcopy(self.geo)
            geo.rois = [ROI()]
            geo.stabilize = False
            corrected, report = GeometryProcessor(geo).apply(self.current_image)
            self.canvas.set_image(corrected[0].image)
            self.freeze_preview = True
            self.preview_caption.setText(self.t("corrected_snapshot"))
            self.canvas.points = []
            self.geometry_mode.setCurrentIndex(0)
        except Exception as error:
            self.show_error(str(error))

    def save_geometry(self, checked=False, silent=False):
        if not hasattr(self, "per_camera_box"):
            return
        camera = self.config.cameras[self.current_camera_index]
        if self.per_camera_box.isChecked():
            values = asdict(self.geo)
            values.pop("per_camera", None)
            self.config.geometry.per_camera[camera.id] = values
        else:
            overrides = self.config.geometry.per_camera
            self.config.geometry = deepcopy(self.geo)
            self.config.geometry.per_camera = overrides
            self.config.geometry.per_camera.pop(camera.id, None)
        if not silent and hasattr(self, "log"):
            self.log.appendPlainText(self.t("geometry_save")+": "+camera.id)

    def open_image(self):
        path, _ = QFileDialog.getOpenFileName(self, self.t("open_image"), "", "Images (*.png *.jpg *.jpeg *.tif *.tiff)")
        if path:
            try:
                self.show_image(read_image(path), Path(path).name)
            except Exception as error:
                self.show_error(str(error))

    def choose_output(self):
        path = QFileDialog.getExistingDirectory(self, self.t("output"), self.config.storage.directory)
        if path:
            self.output_edit.setText(path)

    def choose_model(self):
        path, _ = QFileDialog.getOpenFileName(self, self.t("model_path"), "", "Models (*.pt *.onnx *.engine)")
        if path:
            self.model_path.setText(path)

    def insert_model(self, model):
        row = self.models.rowCount()
        self.models.insertRow(row)
        for col, text in enumerate([model.path, model.task, str(model.confidence)]):
            self.models.setItem(row, col, QTableWidgetItem(text))

    def add_model(self):
        model = ModelConfig(path=self.model_path.text(), task=self.model_task.currentText())
        self.config.analysis.models.append(model)
        self.insert_model(model)

    def remove_model(self):
        row = self.models.currentRow()
        if row >= 0:
            self.models.removeRow(row)
            if row < len(self.config.analysis.models):
                self.config.analysis.models.pop(row)

    def choose_sequence(self):
        path = QFileDialog.getExistingDirectory(self, self.t("import_sequence"), str(self.input_root or self.last_run or Path.cwd()))
        if path:
            self.input_root = Path(path)
            self.sequence_label.setText(path)

    def preflight(self):
        try:
            cfg = self.collect()
            result = estimate_storage(cfg)
            text = self.t("space_summary").format(free=result["free_bytes"]/1e9, reserve=result["reserve_bytes"]/1e9,
                                                   rate=result["estimated_bytes_per_second"]*3600/1e9, hours=result["estimated_hours_available"])
            if result["estimated_total_bytes"] is not None:
                text += "\n"+self.t("space_need").format(need=result["estimated_total_bytes"]/1e9)
            text += "\n"+self.t("space_unlimited" if result["unlimited"] else ("space_ok" if result["sufficient"] else "space_bad"))
            self.space_label.setText(text)
            QMessageBox.information(self, self.t("preflight"), text)
        except Exception as error:
            self.show_error(str(error))

    def toggle_live(self, checked):
        self.freeze_preview = False
        self.config.capture.live = checked
        if isinstance(self.job, CaptureSession):
            self.job.live = checked
        if not checked:
            with self.preview_lock:
                self.preview_frames.clear()

    def push_preview(self, stream, image):
        # Bounded mailbox: queued Qt signals must not retain unbounded full frames.
        if not self.config.capture.live:
            return
        with self.preview_lock:
            self.preview_frames[stream] = image

    def drain_preview(self):
        with self.preview_lock:
            pending = self.preview_frames
            self.preview_frames = {}
        for stream, image in pending.items():
            self.stream_frames[stream] = image
            if self.stream_selector.findText(stream) < 0:
                self.stream_selector.addItem(stream)
            if self.stream_selector.currentText() == stream and not self.freeze_preview:
                self.show_image(image, stream)

    def select_preview(self, stream):
        if not self.task or self.is_preview_job():
            camera_id = stream.split("/")[0]
            for index, camera in enumerate(self.config.cameras):
                if camera.id == camera_id and index != self.current_camera_index:
                    self.camera_selector.setCurrentIndex(index)
                    break
        if stream in self.stream_frames:
            self.show_image(self.stream_frames[stream], stream)

    def show_image(self, image, stream):
        self.current_image = image
        self.canvas.set_image(image)
        suffix = " · "+self.t("simulated") if stream.startswith("demo") else ""
        self.preview_caption.setText(f"{stream} · {image.shape[1]} × {image.shape[0]}{suffix}")

    def start_preview(self):
        try:
            cfg = deepcopy(self.collect())
            self.live.setChecked(True)
            cfg.capture.live = True
            self.launch_capture(cfg, True)
        except Exception as error:
            self.show_error(str(error))

    def start_capture(self):
        try:
            self.launch_capture(deepcopy(self.collect()), False)
        except Exception as error:
            self.show_error(str(error))

    def launch_capture(self, cfg, preview_only):
        if self.hardware_blocked:
            self.show_error(self.t("restart_required"))
            return
        if self.task:
            self.show_error(self.t("busy"))
            return
        task = Task(lambda task: self.job.run(preview_only))
        self.job = CaptureSession(cfg, lambda key, data: task.statusEvent.emit((key, data)), self.push_preview)
        self.job.preview_only = preview_only
        self.freeze_preview = False
        self.start_task_object(task, self.capture_finished)

    def capture_finished(self, path):
        if path:
            self.last_run = Path(path)
            self.input_root = Path(path)
            self.sequence_label.setText(str(path))
            self.log.appendPlainText(self.t("done")+": "+str(path))
            manifest = json.loads((Path(path)/"manifest.json").read_text(encoding="utf-8"))
            self.measurement.setPlainText(json.dumps(manifest["summary"], ensure_ascii=False, indent=2))

    def start_analysis(self, infer):
        if self.task:
            return
        try:
            cfg = deepcopy(self.collect())
            if self.input_root is None:
                self.choose_sequence()
            if self.input_root is None:
                return
            task = Task(lambda task: self.job.run(infer))
            self.job = AnalysisJob(cfg, self.input_root, lambda key, data: task.statusEvent.emit((key, data)), self.push_preview)
            self.start_task_object(task, self.analysis_finished)
        except Exception as error:
            self.show_error(str(error))

    def analysis_finished(self, path):
        self.last_run = Path(path)
        self.set_review_run(Path(path))
        self.nav.setCurrentRow(5)

    def start_task(self, function, done):
        if self.task:
            self.show_error(self.t("busy"))
            return
        self.job = None
        self.start_task_object(Task(function), done)

    def start_task_object(self, task, done):
        if self.task:
            self.show_error(self.t("busy"))
            return
        self.playback.stop()
        self.task = task
        task.statusEvent.connect(self.on_event)
        task.result.connect(done)
        task.failed.connect(self.task_error)
        task.finished.connect(self.task_finished)
        for index, widget in enumerate(self.page_controls[:5]):
            widget.setEnabled(index == 1 and self.is_preview_job())
        self.language.setEnabled(False)
        for button in [self.start_button, self.preview_button, self.check_button]:
            button.setEnabled(False)
        self.stop_button.setEnabled(self.job is not None)
        self.badge.setText(self.t("running"))
        self.canvas.editable = self.nav.currentRow() == 1 and self.is_preview_job()
        task.start()

    def is_preview_job(self):
        return isinstance(self.job, CaptureSession) and self.job.preview_only

    def stop_job(self):
        if self.job:
            self.job.stop()
        self.stop_button.setEnabled(False)

    def on_event(self, event):
        key, data = event
        if key == "statistics":
            self.metrics["elapsed"].setText(f"{data['elapsed_s']:.1f} s")
            self.metrics["saved"].setText(str(data["saved"]))
            self.metrics["dropped"].setText(str(data["dropped"]))
            self.metrics["queue"].setText(f"{data['queue']} / {data['queue_capacity']}")
            return
        if key == "analysis_progress":
            self.progress.setValue(round(100*data["current"]/max(1, data["total"])))
        self.log.appendPlainText(key+"  "+json.dumps(data, ensure_ascii=False))

    def task_error(self, text):
        if "Driver blocked" in text or "Writer did not stop" in text:
            self.hardware_blocked = True
        self.log.appendPlainText(text)
        self.show_error(text[-1800:])

    def task_finished(self):
        task = self.task
        self.task = None
        self.job = None
        for widget in self.page_controls:
            widget.setEnabled(True)
        self.language.setEnabled(True)
        for button in [self.start_button, self.preview_button, self.check_button]:
            button.setEnabled(not self.hardware_blocked)
        self.stop_button.setEnabled(False)
        self.badge.setText(self.t("ready"))
        self.canvas.editable = self.nav.currentRow() == 1
        task.deleteLater()

    def show_error(self, text):
        QMessageBox.critical(self, self.t("error"), text)

    def save_settings(self):
        if self.task:
            return
        try:
            cfg = self.collect()
            path, _ = QFileDialog.getSaveFileName(self, self.t("save"), "uvc-lab-settings.json", "JSON (*.json)")
            if path:
                cfg.save(path)
        except Exception as error:
            self.show_error(str(error))

    def load_settings(self):
        if self.task:
            return
        path, _ = QFileDialog.getOpenFileName(self, self.t("load"), "", "JSON (*.json)")
        if path:
            try:
                cfg = AppConfig.load(path)
                self.config = cfg
                self.current_camera_index = 0
                first = cfg.cameras[0]
                self.geo = geometry_override(cfg.geometry, cfg.geometry.per_camera.get(first.id, {}))
                self.build()
            except Exception as error:
                self.show_error(str(error))

    def edit_json(self):
        if self.task:
            return
        try:
            self.collect()
        except Exception as error:
            self.show_error(str(error))
            return
        dialog = QDialog(self)
        dialog.setWindowTitle(self.t("advanced"))
        dialog.resize(850, 760)
        layout = QVBoxLayout(dialog)
        label = QLabel(self.t("json_hint"))
        label.setWordWrap(True)
        layout.addWidget(label)
        edit = QPlainTextEdit(json.dumps(self.config.to_dict(), ensure_ascii=False, indent=2))
        edit.setFont(QFont("Consolas", 10))
        layout.addWidget(edit)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Apply | QDialogButtonBox.StandardButton.Cancel)
        buttons.button(QDialogButtonBox.StandardButton.Apply).setText(self.t("apply"))
        buttons.rejected.connect(dialog.reject)
        def apply():
            try:
                cfg = AppConfig.from_dict(json.loads(edit.toPlainText()))
                self.config = cfg
                self.current_camera_index = 0
                first = cfg.cameras[0]
                self.geo = geometry_override(cfg.geometry, cfg.geometry.per_camera.get(first.id, {}))
                dialog.accept()
                self.build()
            except Exception as error:
                self.show_error(str(error))
        buttons.button(QDialogButtonBox.StandardButton.Apply).clicked.connect(apply)
        layout.addWidget(buttons)
        dialog.exec()

    def open_review(self):
        path = QFileDialog.getExistingDirectory(self, self.t("open_run"), str(self.last_run or Path.cwd()))
        if path:
            try:
                self.set_review_run(Path(path))
            except Exception as error:
                self.show_error(str(error))

    def set_review_run(self, path):
        if not (path/"records.sqlite").is_file():
            raise ValueError("records.sqlite not found")
        self.playback.stop()
        self.review_run = path
        self.review_label.setText(str(path))
        with sqlite3.connect(path/"records.sqlite") as db:
            streams = [x[0] for x in db.execute("SELECT DISTINCT stream FROM frames ORDER BY stream")]
        self.review_stream.clear()
        self.review_stream.addItems(streams)

    def load_review_stream(self, stream):
        self.playback.stop()
        if self.review_run is None or not stream:
            return
        with sqlite3.connect(self.review_run/"records.sqlite") as db:
            self.review_rows = [row[0] for row in db.execute("SELECT id FROM frames WHERE stream=? ORDER BY timestamp,id", (stream,))]
        self.seek.setRange(0, max(0, len(self.review_rows)-1))
        self.seek.setValue(0)
        self.show_review_frame(0)

    def show_review_frame(self, index):
        if self.review_run is None or not 0 <= index < len(self.review_rows):
            return
        try:
            with sqlite3.connect(self.review_run/"records.sqlite") as db:
                row = db.execute("SELECT stream,timestamp,raw_path,processed_path,metadata,detections,tracks FROM frames WHERE id=?", (self.review_rows[index],)).fetchone()
            stream, timestamp, raw, processed, meta, ds, tracks = row
            path = self.review_run/(processed or raw)
            image = read_image(path)
            metadata, ds, tracks = json.loads(meta), json.loads(ds), json.loads(tracks)
            elapsed = metadata.get("elapsed_s", timestamp)
            self.review_record = image, ds, tracks, elapsed
            self.canvas.set_image(render(image, [d for d in ds if d.get("positive", True)], tracks, elapsed, self.config.analysis))
            self.current_image = image
            self.frame_label.setText(f"{self.t('frame')} {index+1} / {len(self.review_rows)} · {stream}")
            self.preview_caption.setText(str(path))
        except Exception as error:
            self.playback.stop()
            self.log.appendPlainText(str(error))

    def hover_object(self, point):
        if self.nav.currentRow() != 5 or self.review_record is None:
            return
        image, detections, tracks, timestamp = self.review_record
        x, y = point[0]*image.shape[1], point[1]*image.shape[0]
        found = next((d for d in detections if d["bbox"][0] <= x <= d["bbox"][2] and d["bbox"][1] <= y <= d["bbox"][3]), None)
        self.object_details.setPlainText(json.dumps(found, ensure_ascii=False, indent=2) if found else "")
        self.canvas.set_image(render(image, detections, tracks, timestamp, self.config.analysis, found.get("track_id") if found else None))

    def update_overlay(self, key, value):
        self.config.analysis.overlays[key] = value
        self.show_review_frame(self.seek.value())

    def toggle_playback(self):
        if self.playback.isActive():
            self.playback.stop()
        elif self.review_rows:
            self.playback.start(max(1, round(1000/self.config.storage.video_fps)))

    def next_frame(self):
        if self.seek.value() >= self.seek.maximum():
            self.playback.stop()
        else:
            self.seek.setValue(self.seek.value()+1)

    def redraw(self):
        if self.task:
            return
        if self.review_run is None:
            return
        try:
            cfg = deepcopy(self.collect())
            path = self.review_run
            stop = threading.Event()
            class RedrawStop:
                def stop(self):
                    stop.set()
            task = Task(lambda task: redraw_cache(path, cfg, stop, lambda key, data: task.statusEvent.emit((key, data))))
            self.job = RedrawStop()
            self.start_task_object(task, lambda path: self.log.appendPlainText(self.t("done")+": "+str(path)))
        except Exception as error:
            self.show_error(str(error))

    def export_report(self):
        if self.review_run:
            self.start_task(lambda task: export_excel(self.review_run), lambda path: self.log.appendPlainText(str(path)))

    def closeEvent(self, event):
        if self.task and self.task.isRunning():
            self.stop_job()
            event.ignore()
            self.task.finished.connect(self.close)
        else:
            event.accept()
