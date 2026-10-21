from pathlib import Path
import pytest
from PyQt6.QtWidgets import QApplication

from uvc_lab.config import AppConfig
from dataclasses import asdict
from uvc_lab.config import GeometryConfig, ROI
from uvc_lab.i18n import TEXT
from uvc_lab.ui import MainWindow


@pytest.fixture(scope="module")
def app():
    return QApplication.instance() or QApplication([])


@pytest.mark.parametrize("language", ["zh_TW", "en", "ja"])
def test_all_pages_build_and_configuration_is_roundtrippable(app, language):
    cfg = AppConfig(language=language)
    window = MainWindow(cfg)
    window.show()
    app.processEvents()
    for index in range(6):
        window.nav.setCurrentRow(index)
        app.processEvents()
        assert window.pages.currentIndex() == index
    assert window.collect().language == language
    assert window.canvas.image is None
    window.close()


def test_roi_editor_and_model_list_are_used_by_runtime(app):
    window = MainWindow()
    window.nav.setCurrentRow(1)
    window.canvas.selection = [.1, .2, .7, .6]
    window.roi_name.setText("plate")
    window.roi_rows.setValue(2)
    window.roi_cols.setValue(3)
    window.add_roi()
    cfg = window.collect()
    assert cfg.geometry.rois[0].rows == 2
    assert cfg.geometry.rois[0].cols == 3
    window.model_path.setText("yolo26n-pose.pt")
    window.add_model()
    assert len(window.collect().analysis.models) == 2
    window.close()


def test_translations_cover_all_product_strings():
    for key, variants in TEXT.items():
        assert len(variants) == 3
        assert all(text.strip() for text in variants), key


def test_camera_geometry_override_survives_load_and_language_rebuild(app):
    cfg = AppConfig()
    cfg.geometry.per_camera["camera-0"] = asdict(GeometryConfig(rois=[ROI(rows=3, cols=4)], stabilize=True))
    window = MainWindow(cfg)
    assert window.geo.rois[0].rows == 3
    assert window.stabilize_control.isChecked()
    assert window.collect().geometry.per_camera["camera-0"]["rois"][0]["cols"] == 4
    window.language.setCurrentIndex(1)
    assert window.collect().geometry.per_camera["camera-0"]["rois"][0]["cols"] == 4
    window.close()


def test_capture_job_runs_through_qt_without_blocking_live_controls(app, tmp_path):
    from PyQt6.QtTest import QTest
    import time
    cfg = AppConfig()
    cfg.cameras[0].source = "demo"
    cfg.cameras[0].width, cfg.cameras[0].height = 160, 120
    cfg.cameras[0].warmup_s = 0
    cfg.capture.interval_s = .1
    cfg.capture.duration_s = .35
    cfg.capture.min_free_gb = 0
    cfg.storage.directory = str(tmp_path)
    window = MainWindow(cfg)
    errors = []
    window.show_error = errors.append
    window.start_capture()
    assert window.task is not None
    assert window.live.isEnabled()
    window.live.setChecked(False)
    deadline = time.monotonic()+5
    while window.task is not None and time.monotonic() < deadline:
        QTest.qWait(20)
    assert window.task is None
    assert not errors
    assert window.last_run is not None
    assert (window.last_run/"manifest.json").exists()
    window.close()
