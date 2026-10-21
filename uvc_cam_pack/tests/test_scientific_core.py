from copy import deepcopy
import json
import math
from pathlib import Path

import cv2
import numpy as np
import pytest

from uvc_lab.analysis import consensus, discover_sequences, fuse_models, match_detections, undo_rotation
from uvc_lab.camera import parse_dshow_modes, parse_v4l2_modes
from uvc_lab.config import AppConfig, GeometryConfig, ROI
from uvc_lab.geometry import GeometryProcessor, transform_points
from uvc_lab.hardware import parse_sensor_line
from uvc_lab.storage import estimate_storage, read_image, write_image
from uvc_lab.tracking import StreamTracker
from uvc_lab.vendor.tracker import UnscentedKalmanFilter


def det(box=(10, 20, 30, 40), label="animal", model="seg", **extra):
    return {"bbox": list(box), "label": label, "confidence": .9, "model": model, **extra}


def test_config_roundtrip_and_reject_mistyped_fields(tmp_path):
    cfg = AppConfig()
    cfg.geometry.rois = [ROI("plate", [.1, .1, .8, .8], "ellipse", 2, 3)]
    cfg.save(tmp_path/"設定.json")
    restored = AppConfig.load(tmp_path/"設定.json")
    assert cfg.to_dict() == restored.to_dict()
    invalid = cfg.to_dict()
    invalid["capture"]["fpss"] = 20
    with pytest.raises(ValueError, match="Unknown"):
        AppConfig.from_dict(invalid)
    cfg.save(tmp_path/"redacted.json", redact=True)
    restored = AppConfig.load(tmp_path/"redacted.json")
    assert restored.hardware.token != cfg.hardware.token
    assert restored.geometry.rois[0].rect == cfg.geometry.rois[0].rect


@pytest.mark.parametrize("change", [
    lambda c: setattr(c.capture, "interval_s", 0),
    lambda c: setattr(c.capture, "duration_s", -1),
    lambda c: setattr(c.cameras[0], "fps", float("nan")),
    lambda c: setattr(c.geometry.rois[0], "rect", [.5, 0, .6, 1]),
    lambda c: setattr(c.geometry.rois[0], "rows", 0),
    lambda c: setattr(c.geometry, "perspective", [[0, 0], [1, 1], [1, 0], [0, 1]]),
    lambda c: setattr(c.analysis, "consensus", "any-two-classes"),
    lambda c: setattr(c.hardware, "targets", ["../escape"]),
])
def test_invalid_scientific_configuration(change):
    c = AppConfig()
    change(c)
    with pytest.raises(ValueError):
        c.validate()


def test_sigma_points_reconstruct_correlated_covariance():
    k = UnscentedKalmanFilter(dim_x=2, dim_z=2)
    p = np.array([[2., 1.], [1., 3.]])
    sigma = k.generate_sigma_points(np.zeros(2), p)
    np.testing.assert_allclose((sigma.T*k.Wc)@sigma, p, atol=1e-4)


@pytest.mark.parametrize("algorithm", ["ukf", "sort", "bytetrack"])
def test_tracker_timestamp_units_and_isolation(algorithm):
    cfg = AppConfig().analysis
    cfg.tracker = algorithm
    cfg.min_hits = 1
    a, b = StreamTracker(cfg), StreamTracker(cfg)
    for t, x in [(0, 0), (10, 1), (20, 2)]:
        tracks = a.update([det((x, 0, x+20, 20))], t)
        assert len(tracks) == 1
        assert tracks[0]["id"] == 1
    b_tracks = b.update([det((100, 100, 120, 120))], 0)
    assert b_tracks[0]["id"] == 1
    assert len(a.tracker.tracking_history) <= 1
    if algorithm != "ukf":
        assert a.tracker.tracked_objects[0].custom_kf.F[0, 4] == 10
    with pytest.raises(ValueError, match="strictly increasing"):
        a.update([], 20)


def test_missing_observation_not_counted_as_detection():
    cfg = AppConfig().analysis
    cfg.min_hits = 1
    tracker = StreamTracker(cfg)
    assert tracker.update([det()], 0)
    assert tracker.update([], 1) == []


def test_consensus_needs_same_object_and_distinct_views():
    assert consensus([[det()], [det((100, 200, 130, 240))], []], .3) == []
    assert consensus([[det(), det()], [], []], .3) == []
    assert consensus([[det()], [det(label="other")], []], .3) == []
    accepted = consensus([[det(), det((100, 200, 130, 240))], [det()], []], .3)
    assert len(accepted) == 1
    assert accepted[0]["votes"] == 2


def test_temporal_center_miss_is_inferred_not_measured_area():
    accepted = consensus([[det(area_px2=12)], [], [det(area_px2=16)]], .3, 1)
    assert len(accepted) == 1
    assert accepted[0]["inferred"]
    assert accepted[0]["area_px2"] is None


def test_temporal_boundary_never_duplicates_an_image():
    assert consensus([[], [det()], []], .3, 1) == []
    assert consensus([[], [det()], [det()]], .3, 1)[0]["votes"] == 2


@pytest.mark.parametrize("rotation", [90, -90])
def test_rotation_inverse_non_square_box_mask_pose(rotation):
    w, h = 100, 60
    original_box = [10, 20, 30, 40]
    rotated_box = [h-40, 10, h-20, 30] if rotation == 90 else [20, w-30, 40, w-10]
    point = [12, 23]
    rotated_point = [h-1-point[1], point[0]] if rotation == 90 else [point[1], w-1-point[0]]
    source = det(rotated_box, polygon=[rotated_point, rotated_point, rotated_point], keypoints=[rotated_point+[.8]])
    restored = undo_rotation(source, rotation, (w, h))
    np.testing.assert_allclose(restored["bbox"], original_box)
    np.testing.assert_allclose(restored["polygon"][0], point)
    np.testing.assert_allclose(restored["keypoints"][0], point+[.8])


def test_model_fusion_keeps_attributes_and_semantics():
    ds = fuse_models([[det(area_px2=35, polygon=[[1, 2], [3, 4], [2, 4]])],
                      [det(model="pose", keypoints=[[10, 20, .9]])]], .3)
    assert len(ds) == 1
    assert ds[0]["area_px2"] == 35
    assert ds[0]["keypoints"] == [[10, 20, .9]]
    assert len(ds[0]["measurements"]) == 2
    assert len(fuse_models([[det()], [det(label="different")]], .3)) == 2


def test_mask_roundtrip_keeps_holes_and_rotation():
    from uvc_lab.analysis import encode_mask, decode_mask
    binary = np.zeros((30, 50), bool)
    binary[3:20, 4:35] = True
    binary[8:13, 10:15] = False
    np.testing.assert_array_equal(decode_mask(encode_mask(binary)), binary)
    cw = cv2.rotate(binary.astype(np.uint8), cv2.ROTATE_90_CLOCKWISE) > 0
    d = det((0, 0, 20, 40), mask_bits=encode_mask(cw))
    restored = undo_rotation(d, 90, (50, 30))
    np.testing.assert_array_equal(decode_mask(restored["mask_bits"]), binary)


def test_bytetrack_low_confidence_recovers_old_track_without_new_positive():
    cfg = AppConfig().analysis
    cfg.tracker = "bytetrack"
    cfg.min_hits = 1
    tracker = StreamTracker(cfg)
    assert tracker.update([det()], 0)[0]["id"] == 1
    low = det(confidence=.2, positive=False)
    observed = tracker.update([low], 1)
    assert observed[0]["id"] == 1
    assert observed[0]["positive"] is False
    other = StreamTracker(cfg)
    assert other.update([det(confidence=.2, positive=False)], 0) == []


def test_grid_partition_has_no_missing_remainder_pixels():
    image = np.arange(11*13*3, dtype=np.uint8).reshape(11, 13, 3)
    tiles, _ = GeometryProcessor(GeometryConfig(rois=[ROI(rows=2, cols=3)])).apply(image)
    assert sum(t.image.shape[0]*t.image.shape[1] for t in tiles) == 11*13
    reconstructed = np.vstack([np.hstack([t.image for t in tiles[:3]]), np.hstack([t.image for t in tiles[3:]])])
    np.testing.assert_array_equal(reconstructed, image)


def test_ellipse_mask_and_sensor_coordinate_transform():
    image = np.full((100, 200, 3), 255, np.uint8)
    tiles, _ = GeometryProcessor(GeometryConfig(rois=[ROI(rect=[.25, .2, .5, .6], shape="ellipse")])).apply(image)
    tile = tiles[0]
    assert tile.image[0, 0].sum() == 0
    assert tile.image[30, 50].sum() == 765
    np.testing.assert_allclose(transform_points([[50, 20]], tile.sensor_to_tile), [[0, 0]])


def test_perspective_and_translation_are_measurable_and_raw_is_untouched():
    image = np.zeros((180, 240, 3), np.uint8)
    rng = np.random.default_rng(10)
    image[:] = rng.integers(0, 255, image.shape, dtype=np.uint8)
    copy = image.copy()
    geometry = GeometryProcessor(GeometryConfig(stabilize=True, max_shift_px=10))
    geometry.apply(image)
    shifted = cv2.warpAffine(image, np.float32([[1, 0, 3], [0, 1, 2]]), (240, 180))
    tiles, report = geometry.apply(shifted)
    assert report["stabilization_applied"]
    np.testing.assert_allclose(report["shift_px"], [3, 2], atol=.35)
    np.testing.assert_array_equal(image, copy)
    geo = GeometryConfig(perspective=[[.1, .1], [.9, .15], [.8, .9], [.2, .9]], output_size=[100, 80])
    tiles, report = GeometryProcessor(geo).apply(image)
    points = np.asarray(geo.perspective)*[239, 179]
    np.testing.assert_allclose(transform_points(points, report["sensor_to_rectified"]), [[0, 0], [99, 0], [99, 79], [0, 79]], atol=1e-4)


def test_unlimited_space_and_multiple_targets():
    c = AppConfig()
    finite = estimate_storage(c, 10**12)
    c.hardware.enabled = True
    multi = estimate_storage(c, 10**12)
    assert multi["estimated_total_bytes"] >= finite["estimated_total_bytes"]*2
    c.capture.duration_s = None
    infinite = estimate_storage(c, 10**12)
    assert infinite["sufficient"] is None
    assert infinite["estimated_total_bytes"] is None
    assert infinite["estimated_hours_available"] > 0


def test_unicode_image_io_and_natural_sequence_order(tmp_path):
    image = np.zeros((16, 24, 3), np.uint8)
    for name in ["frame10.png", "frame2.png", "frame1.png"]:
        write_image(tmp_path/"相機A"/name, image)
    write_image(tmp_path/"相機B"/"frame1.png", image)
    np.testing.assert_array_equal(read_image(tmp_path/"相機A"/"frame1.png"), image)
    groups = discover_sequences(tmp_path, 600)
    assert len(groups) == 2
    assert [f.path.name for f in groups["相機A"]] == ["frame1.png", "frame2.png", "frame10.png"]
    assert [f.timestamp for f in groups["相機A"]] == [0, 600, 1200]


def test_original_trackbox_same_directory_cameras_stay_separate(tmp_path):
    image = np.zeros((10, 10, 3), np.uint8)
    for cam in [1, 2]:
        for ts in [1700000000, 1700000600]:
            write_image(tmp_path/f"cam{cam}_frame_{ts}.tiff", image)
            write_image(tmp_path/f"cam{cam}_anno_frame_{ts}.tiff", image)
    groups = discover_sequences(tmp_path, 1)
    assert set(groups) == {"cam1", "cam2"}
    assert all(len(items) == 2 for items in groups.values())
    assert groups["cam1"][1].timestamp-groups["cam1"][0].timestamp == 600


def test_original_overlay_coordinates_are_reprojected():
    from uvc_lab.analysis import project_overlays
    ds = [det((10, 20, 30, 40), polygon=[[10, 20], [30, 20], [30, 40]], keypoints=[[10, 20, .8]])]
    ts = [{"id": 1, "bbox": [10, 20, 30, 40], "velocity_px_s": [2, 3], "trail": [[10, 20]]}]
    projected, projected_tracks = project_overlays(ds, ts, [[1, 0, 100], [0, 1, 50], [0, 0, 1]])
    assert projected[0]["bbox"] == [110, 70, 130, 90]
    assert projected[0]["keypoints"] == [[110, 70, .8]]
    assert projected_tracks[0]["trail"] == [[110, 70]]
    assert projected_tracks[0]["velocity_px_s"] == [2, 3]


def test_driver_modes_are_not_hardcoded_imx179_claims():
    text = "[dshow] vcodec=mjpeg min s=1920x1080 fps=5 max s=1920x1080 fps=22\n[dshow] pixel_format=yuyv422 min s=640x480 fps=10 max s=640x480 fps=30"
    modes = parse_dshow_modes(text)
    assert any(m.width == 1920 and m.fps == 22 and m.fourcc == "MJPG" for m in modes)
    assert not any(m.width == 3264 for m in modes)
    v4 = "[0]: 'MJPG'\n Size: Discrete 1280x720\n Interval: Discrete 0.033s (30.000 fps)"
    assert parse_v4l2_modes(v4)[0].fps == 30


def test_resolution_benchmark_asks_for_advertised_ceiling_without_mutating_preset(monkeypatch):
    import uvc_lab.camera as module
    c = AppConfig().cameras[0]
    c.fps = 10
    monkeypatch.setattr(module, "enumerate_modes", lambda config: ([module.CameraMode(1280, 720, 30, "MJPG")], "test"))
    def measure(config, seconds):
        assert config.fps == 30
        return {"measured_fps": 22, "conservative_fps": 20}
    monkeypatch.setattr(module, "measure_camera", measure)
    result = module.benchmark_resolution(c)
    assert result["tested_fps_request"] == 30
    assert result["suggested_fps"] == 18
    assert c.fps == 10


@pytest.mark.parametrize("line,expected", [
    ('{"temperature": 23.4, "humidity": 60}', (23.4, 60)),
    ("Temperature: 23.4 C, Humidity: 60 %", (23.4, 60)),
    ("温度: -2.1, 湿度: 45.6", (-2.1, 45.6)),
    ("broken", None), ('{"temperature": 20, "humidity": 200}', None),
])
def test_sensor_formats(line, expected):
    assert parse_sensor_line(line) == expected
