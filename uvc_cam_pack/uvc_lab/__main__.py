from __future__ import annotations

import argparse
from pathlib import Path
import sys

from .config import AppConfig


def main():
    from .runtime import bootstrap, execute_camera_tool
    bootstrap()
    parser = argparse.ArgumentParser(description="UVC Lab scientific imaging workstation")
    parser.add_argument("--config", type=Path)
    parser.add_argument("--demo", action="store_true", help="Use clearly labeled synthetic cameras")
    parser.add_argument("--capture", action="store_true", help="Run acquisition without a GUI")
    parser.add_argument("--analyze", type=Path, help="Analyze a sequence or acquisition run without a GUI")
    parser.add_argument("--camera-tool", choices=["scan", "modes", "benchmark"], help=argparse.SUPPRESS)
    parser.add_argument("--camera-json", help=argparse.SUPPRESS)
    parser.add_argument("--result-json", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--self-test", type=Path, help="Write a packaged-runtime self-test report to this directory")
    args = parser.parse_args()
    if args.camera_tool:
        if args.result_json is None:
            parser.error("--camera-tool requires --result-json")
        sys.exit(execute_camera_tool(args.camera_tool, args.camera_json, args.result_json))
    if args.self_test:
        from .selftest import run_self_test
        sys.exit(run_self_test(args.self_test))
    config = AppConfig.load(args.config) if args.config else AppConfig()
    if args.demo:
        from .config import CameraConfig
        config.cameras = [CameraConfig(id="demo-1", source="demo-1", name="SIMULATION", warmup_s=0)]
    if args.capture:
        from .acquisition import CaptureSession
        job = CaptureSession(config, lambda key, data: print(key, data) if key != "statistics" else None)
        import signal
        signal.signal(signal.SIGINT, lambda *_: job.stop())
        print(job.run())
    elif args.analyze:
        from .analysis import AnalysisJob
        job = AnalysisJob(config, args.analyze, lambda key, data: print(key, data))
        import signal
        signal.signal(signal.SIGINT, lambda *_: job.stop())
        print(job.run())
    else:
        from PyQt6.QtWidgets import QApplication
        from .ui import MainWindow
        app = QApplication(sys.argv[:1])
        window = MainWindow(config)
        window.show()
        sys.exit(app.exec())


if __name__ == "__main__":
    main()
