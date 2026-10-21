"""Reference external PC / Raspberry Pi controller. No motor command is included.

python examples/controller_pc.py --url http://CAMERA_PC:8765 --token TOKEN
Replace move_and_wait() with the controller's actual positioning implementation.
Use --simulate ONLY with a simulated camera for protocol validation.
"""
import argparse
import json
import time
from urllib.request import Request, urlopen


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://127.0.0.1:8765")
    parser.add_argument("--token", required=True)
    parser.add_argument("--simulate", action="store_true")
    args = parser.parse_args()

    def api(path, body=None):
        request = Request(args.url+path, data=json.dumps(body).encode() if body is not None else None,
                          headers={"Authorization": "Bearer "+args.token, "Content-Type": "application/json"})
        with urlopen(request, timeout=5) as response:
            return json.load(response)

    def move_and_wait(target):
        if not args.simulate:
            raise RuntimeError("Implement your controller's move + arrival confirmation before use")
        print("SIMULATION: arrival at", target, flush=True)
        time.sleep(.3)

    arrived = set()
    while True:
        state = api("/v1/state")
        phase = state["phase"]
        body = {"request_id": state["request_id"], "target": state["target"]}
        if phase == "awaiting_arrival" and state["can_move"] and state["request_id"] not in arrived:
            move_and_wait(state["target"])
            api("/v1/arrived", body)
            arrived.add(state["request_id"])
        elif phase == "capture_complete":
            print("Capture saved:", state["capture"], flush=True)
            api("/v1/acknowledge", body)
        elif phase in ["stopped", "error", "timeout"]:
            print("Session ended:", phase)
            return
        time.sleep(.1)


if __name__ == "__main__":
    main()
