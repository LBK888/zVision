"""HTTP handshake for external motion controllers; this module never drives motors."""
from __future__ import annotations

from collections import deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import hmac
import json
import re
import threading
import time
import uuid

from .config import HardwareConfig


class Handshake:
    def __init__(self, config: HardwareConfig, event=lambda kind, data: None):
        self.config = config
        self.event = event
        self.condition = threading.Condition()
        self.state = {"phase": "idle", "request_id": None, "target": None, "can_move": False, "revision": 0}
        self.history = deque(maxlen=256)
        self.arrival_time = None
        self.stopped = False
        self.server = None
        self.thread = None

    def snapshot(self):
        with self.condition:
            return {**self.state, "events": list(self.history)}

    def transition(self, phase, **values):
        with self.condition:
            self.state.update(phase=phase, **values, revision=self.state["revision"]+1)
            snapshot = {**self.state, "unix_s": time.time()}
            self.history.append(snapshot)
            self.event("hardware_"+phase, snapshot)
            self.condition.notify_all()

    def request(self, target: str) -> str:
        with self.condition:
            if self.stopped:
                raise InterruptedError("Hardware handshake stopped")
            request_id = uuid.uuid4().hex
            self.arrival_time = None
            self.transition("awaiting_arrival", target=target, request_id=request_id, can_move=True, capture=None)
            return request_id

    def arrive(self, request_id: str, target: str):
        with self.condition:
            if request_id != self.state["request_id"] or target != self.state["target"]:
                raise ValueError("Stale request ID or incorrect target")
            # Retry-safe: a repeated arrival cannot reset settling or complete a new request.
            if self.state["phase"] in ["settling", "capturing", "capture_complete"]:
                return self.snapshot()
            if self.state["phase"] != "awaiting_arrival" or self.stopped:
                raise ValueError("No active arrival request")
            self.arrival_time = time.monotonic()
            self.transition("settling", can_move=False)
            return self.snapshot()

    def wait_arrival(self, stop: threading.Event) -> float:
        deadline = time.monotonic()+self.config.timeout_s
        with self.condition:
            while self.arrival_time is None:
                if stop.is_set() or self.stopped:
                    raise InterruptedError("Stopped while waiting for hardware")
                remaining = deadline-time.monotonic()
                if remaining <= 0:
                    self.transition("timeout", can_move=False)
                    raise TimeoutError("Motion controller arrival timed out")
                self.condition.wait(min(.2, remaining))
            arrival = self.arrival_time
        return arrival

    def capturing(self):
        self.transition("capturing", can_move=False)

    def complete(self, capture: dict):
        self.transition("capture_complete", can_move=True, capture=capture)

    def acknowledge(self, request_id: str, target: str):
        with self.condition:
            if request_id != self.state["request_id"] or target != self.state["target"]:
                raise ValueError("Stale request ID or incorrect target")
            if self.state["phase"] == "acknowledged":
                return self.snapshot()
            if self.state["phase"] != "capture_complete":
                raise ValueError("Capture has not completed")
            self.transition("acknowledged", can_move=True)
            return self.snapshot()

    def wait_acknowledgement(self, stop: threading.Event):
        deadline = time.monotonic()+self.config.timeout_s
        with self.condition:
            while self.state["phase"] != "acknowledged":
                if stop.is_set() or self.stopped:
                    raise InterruptedError("Stopped while waiting for controller acknowledgement")
                if time.monotonic() >= deadline:
                    self.transition("timeout", can_move=False)
                    raise TimeoutError("Controller did not acknowledge capture completion")
                self.condition.wait(.1)

    def start(self):
        owner = self
        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def response(self, code, body):
                data = json.dumps(body, ensure_ascii=False).encode("utf-8")
                self.send_response(code)
                self.send_header("Content-Type", "application/json; charset=utf-8")
                self.send_header("Content-Length", str(len(data)))
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                self.wfile.write(data)

            def authorized(self):
                token = self.headers.get("Authorization", "")
                if not hmac.compare_digest(token, "Bearer "+owner.config.token):
                    self.response(401, {"error": "unauthorized"})
                    return False
                return True

            def do_GET(self):
                if not self.authorized():
                    return
                if self.path != "/v1/state":
                    self.response(404, {"error": "not_found"})
                else:
                    self.response(200, owner.snapshot())

            def do_POST(self):
                if not self.authorized():
                    return
                if self.path not in ["/v1/arrived", "/v1/acknowledge"]:
                    self.response(404, {"error": "not_found"})
                    return
                try:
                    length = int(self.headers.get("Content-Length", "0"))
                    if not 0 < length <= 4096:
                        raise ValueError("Invalid body size")
                    self.connection.settimeout(3)
                    data = json.loads(self.rfile.read(length))
                    operation = owner.arrive if self.path == "/v1/arrived" else owner.acknowledge
                    state = operation(data["request_id"], data["target"])
                    self.response(200, state)
                except (ValueError, KeyError, TypeError, OSError) as error:
                    self.response(409, {"error": str(error)})
        self.server = ThreadingHTTPServer((self.config.host, self.config.port), Handler)
        self.server.daemon_threads = True
        self.thread = threading.Thread(target=self.server.serve_forever, kwargs={"poll_interval": .1}, daemon=True)
        self.thread.start()
        return self.server.server_address

    def stop(self, phase="stopped"):
        with self.condition:
            self.stopped = True
            self.transition(phase, can_move=False)
        if self.server:
            self.server.shutdown()
            self.server.server_close()
            self.thread.join(timeout=2)
            self.server = None


def parse_sensor_line(line: str) -> tuple[float, float] | None:
    try:
        data = json.loads(line)
        temp, humidity = float(data["temperature"]), float(data["humidity"])
    except (ValueError, TypeError, KeyError):
        values = re.findall(r"[:：]\s*(-?\d+(?:\.\d+)?)", line)
        if len(values) < 2:
            return None
        temp, humidity = map(float, values[:2])
    import math
    if not math.isfinite(temp) or not math.isfinite(humidity) or not 0 <= humidity <= 100:
        return None
    return temp, humidity


class SensorWorker(threading.Thread):
    def __init__(self, config: HardwareConfig, store, stop: threading.Event):
        super().__init__(daemon=True, name="environment-sensor")
        self.config, self.store, self.stop_event = config, store, stop
        self.error = None

    def run(self):
        try:
            import serial
            with serial.Serial(self.config.sensor_port, self.config.sensor_baud, timeout=.2) as port:
                if self.config.sensor_start_command:
                    port.write(self.config.sensor_start_command.encode("utf-8"))
                previous = -float("inf")
                while not self.stop_event.is_set():
                    line = port.readline().decode("utf-8", errors="replace").strip()
                    parsed = parse_sensor_line(line)
                    now = time.monotonic()
                    if parsed and now-previous >= self.config.sensor_interval_s:
                        self.store.sensor(time.time(), *parsed, line)
                        previous = now
        except Exception as error:
            self.error = str(error)
            self.store.event("sensor_error", {"error": str(error)})
