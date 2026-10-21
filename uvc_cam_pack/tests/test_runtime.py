import json
from pathlib import Path

from uvc_lab.config import CameraConfig
from uvc_lab.runtime import camera_command, execute_camera_tool, model_path


def test_model_resources_do_not_replace_missing_custom_paths(tmp_path, monkeypatch):
    import uvc_lab.runtime as runtime
    (tmp_path/"models").mkdir()
    bundled = tmp_path/"models"/"yolo26n-seg.pt"
    bundled.write_bytes(b"test resource")
    monkeypatch.setattr(runtime, "resources", lambda: tmp_path)
    assert model_path("yolo26n-seg.pt") == str(bundled)
    assert model_path("missing/yolo26n-seg.pt") == "missing/yolo26n-seg.pt"


def test_camera_helper_uses_structured_file_results():
    result = camera_command("modes", CameraConfig(source="demo"))
    assert result["evidence"] == "simulation"
    assert result["modes"][0]["width"] > 0


def test_camera_helper_failure_is_written_to_result_file(tmp_path):
    output = tmp_path/"result.json"
    assert execute_camera_tool("unknown", "{}", output) == 1
    result = json.loads(output.read_text(encoding="utf-8"))
    assert result["ok"] is False
    assert "Unknown camera action" in result["error"]
