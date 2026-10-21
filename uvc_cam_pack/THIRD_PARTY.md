# Third-party sources

`uvc_lab/vendor/tracker.py` is adapted from LBK888/zVision,
`yolo_tracker_GUI/yolo_tracker_v2.py`, commit
`cf58cf4a6b77f35a624219544b1a60394c04dcf8`.
The source README describes the project as MIT licensed; this fetched revision
does not contain a root LICENSE file. Original authorship is retained here.
Local changes correct UKF sigma-point orientation, use real `dt` in SORT /
ByteTrack, remove eager Ultralytics imports, and defer an optional pandas import.
The wrapper adds stream isolation, bounded history, pre-assignment gating and
semantic class mapping. This is the upstream lightweight implementation, not a
claim of conformance to the official ByteTrack reference implementation.

TrackBox source reviewed: LBK888/TrackBox,
`TrackBox_recodring`, commit `f5fbf652e63349fd2edeb9331293fe0bb72773f4`.
Its acquisition logic was reimplemented, rather than copied into the runtime.
The fetched revision does not contain a root LICENSE file.

Ultralytics models/code use their own licensing terms:
[Ultralytics license](https://www.ultralytics.com/license).
Model weights are separate runtime downloads and are not included in the source
archive. PyQt6, OpenCV, FFmpeg and the other dependencies retain their respective
licenses; this document does not relicense them.

The Windows one-file release additionally bundles the two official default
weights (`yolo26n-seg.pt`, `yolo26n-pose.pt`), CPU PyTorch/torchvision, PyQt6,
FFmpeg, and their runtime dependencies. The matching release provides a
third-party-notices archive with installed license texts and component versions.
Build recipes are in `packaging/`; model SHA-256 values are recorded in inference
reports. The source archive itself excludes all binary dependencies and weights.

Upstream component sources:

- [Ultralytics](https://github.com/ultralytics/ultralytics),
  [official model assets](https://github.com/ultralytics/assets/releases/tag/v8.4.0)
- [PyTorch](https://github.com/pytorch/pytorch),
  [torchvision](https://github.com/pytorch/vision)
- [PyQt6 / Riverbank](https://www.riverbankcomputing.com/software/pyqt/),
  [Qt source](https://code.qt.io/cgit/qt/)
- [FFmpeg source for the bundled revision](https://github.com/FFmpeg/FFmpeg/tree/33b215d155),
  [BtbN Windows build recipes](https://github.com/BtbN/FFmpeg-Builds)
- [OpenCV](https://github.com/opencv/opencv),
  [NumPy](https://github.com/numpy/numpy),
  [SciPy](https://github.com/scipy/scipy)

The FFmpeg version/build configuration is included in the release notices.
