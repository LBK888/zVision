"""Real YOLO26 segmentation + pose smoke test; downloads official weights once."""
from pathlib import Path
import json
import os
import argparse
import importlib.util

from uvc_lab.analysis import AnalysisJob
from uvc_lab.camera import DemoCamera
from uvc_lab.config import AppConfig, CameraConfig, ModelConfig
from uvc_lab.storage import write_image

ROOT = Path(__file__).resolve().parents[1]
os.chdir(ROOT)
parser = argparse.ArgumentParser()
parser.add_argument("--consensus", choices=["none", "rotation", "temporal"], default="none")
parser.add_argument("--synthetic", action="store_true")
args = parser.parse_args()
config = AppConfig()
config.experiment = "real_yolo26_smoke"
config.storage.directory = str(ROOT/"runs"/"validation")
config.storage.annotated_video = True
config.analysis.imgsz = 320
config.analysis.consensus = args.consensus
config.analysis.min_hits = 1
config.analysis.models = [ModelConfig(path=str(ROOT/".cache"/"models"/"yolo26n-seg.pt")),
                          ModelConfig(path=str(ROOT/".cache"/"models"/"yolo26n-pose.pt"), task="pose")]
camera = DemoCamera(CameraConfig(source="demo", width=320, height=240))
inputs = ROOT/".cache"/"smoke-input"
from uvc_lab.storage import read_image
sample = Path(importlib.util.find_spec("ultralytics").origin).parent/"assets"/"bus.jpg"
image = read_image(sample) if sample.is_file() and not args.synthetic else None
for i in range(3):
    write_image(inputs/f"frame{i}.png", image if image is not None else camera.read()[1])
job = AnalysisJob(config, inputs, lambda key, data: print(key, json.dumps(data), flush=True))
output = job.run()
print("OUTPUT", output)
import sqlite3
with sqlite3.connect(output/"records.sqlite") as db:
    records = list(db.execute("SELECT detections FROM frames"))
    detections = [d for row in records for d in json.loads(row[0])]
    fused = [d for d in detections if len(d["models"]) >= 2]
    print("DETECTIONS", len(detections), "FUSED", len(fused))
    if image is not None and args.consensus != "rotation":
        assert detections, "Real sample should produce detections"
        assert any(d.get("area_px2") and d.get("keypoints") for d in fused), "Expected segmentation + pose fusion on real sample"
    if args.consensus == "rotation":
        print("Rotation consensus may reject all upright-trained sample detections; inspect recorded model_evidence.")
