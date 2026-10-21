"""Render the actual Qt interface with labeled simulation data for visual QA."""
import os
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
from pathlib import Path
from PyQt6.QtWidgets import QApplication
from uvc_lab.camera import DemoCamera
from uvc_lab.config import AppConfig, CameraConfig, ROI
from uvc_lab.ui import MainWindow

root = Path(__file__).resolve().parents[1]
app = QApplication([])
for language in ["zh_TW", "en", "ja"]:
    config = AppConfig(language=language)
    config.cameras = [CameraConfig(id="demo-1", source="demo-1", name="SIMULATION", warmup_s=0)]
    config.geometry.rois = [ROI(rect=[.06, .08, .88, .84], rows=2, cols=3)]
    window = MainWindow(config)
    window.nav.setCurrentRow(1)
    window.show()
    app.processEvents()
    frame = DemoCamera(config.cameras[0]).read()[1]
    window.stream_frames["demo-1"] = frame
    window.stream_selector.addItem("demo-1")
    window.roi_list.setCurrentRow(0)
    app.processEvents()
    window.grab().save(str(root/"docs"/f"ui-{language}.png"))
    window.close()
