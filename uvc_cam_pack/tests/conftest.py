import os
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("YOLO_CONFIG_DIR", str(Path(__file__).resolve().parents[1]/".cache"/"ultralytics"))

