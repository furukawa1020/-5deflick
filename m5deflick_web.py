from __future__ import annotations

import argparse
import json
import os
import threading
import time
from collections import deque
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace
from urllib.parse import unquote, urlparse

import cv2
import numpy as np

import m5deflick as core


ROOT = Path(__file__).resolve().parent
WEB_ROOT = ROOT / "web"


class WebState:
    def __init__(self, calibration_path: str):
        self.lock = threading.RLock()
        self.frame_condition = threading.Condition(self.lock)
        self.calibration_path = calibration_path
        self.calibration = core.Calibration.load(calibration_path)
        self.calibration_points: list[tuple[float, float]] = []
        self.recalibrating = self.calibration is None
        self.frame_size = [0, 0]
        self.status = "starting"
        self.last_label = ""
        self.output_text = ""
        self.events: deque[dict[str, object]] = deque(maxlen=80)
        self.camera_jpeg: bytes | None = None
        self.board_jpeg: bytes | None = None
        self.mask_jpeg: bytes | None = None
        self.reset_background_requested = False
        self.reset_zones_requested = False
        self.settings_version = 0
        self.send_unicode = False
        self.settings = {
            "zone_mode": "color",
            "direction_mode": "side",
            "deadzone": 0.20,
            "key_inflate": 0.50,
            "min_motion_area": 650,
            "motion_threshold": 18,
            "arm_mode": "combined",
            "cooldown": 0.32,
            "settle": 0.10,
            "bg_alpha": 0.006,
            "color_min_area": 500,
            "color_alpha": 0.28,
            "color_search_inflate": 0.65,
        }

    def snapshot(self) -> dict[str, object]:
        with self.lock:
            return {
                "status": self.status,
                "calibrated": self.calibration is not None and not self.recalibrating,
                "recalibrating": self.recalibrating,
                "calibrationPoints": len(self.calibration_points),
                "frameSize": self.frame_size,
                "lastLabel": self.last_label,
                "outputText": self.output_text,
                "events": list(self.events)[-24:],
                "sendUnicode": self.send_unicode,
                "settings": dict(self.settings),
            }


class WebOutput:
    def __init__(self, state: WebState):
        self.state = state
        self.windows_input = None

    def emit_text(self, text: str) -> None:
        with self.state.lock:
            self.state.output_text += text
            self.state.events.appendleft({"kind": "text", "value": text, "at": time.time()})
            send_unicode = self.state.send_unicode
        if send_unicode:
            self.try_windows_input("text", text)

    def emit_key(self, key_name: str) -> None:
        with self.state.lock:
            if key_name == "backspace":
                self.state.output_text = self.state.output_text[:-1]
            self.state.events.appendleft({"kind": "key", "value": key_name, "at": time.time()})
            send_unicode = self.state.send_unicode
        if send_unicode:
            self.try_windows_input("key", key_name)

    def try_windows_input(self, kind: str, value: str) -> None:
        try:
            if kind == "text":
                self._windows().emit_text(value)
            else:
                self._windows().emit_key(value)
        except Exception as exc:
            with self.state.lock:
                self.state.send_unicode = False
                self.state.status = f"unicode input disabled: {exc}"
                self.state.events.appendleft({"kind": "key", "value": "input-error", "at": time.time()})

    def _windows(self) -> core.WindowsInput:
        if self.windows_input is None:
            self.windows_input = core.WindowsInput()
        return self.windows_input


class Processor:
    def __init__(self, state: WebState, args: argparse.Namespace):
        self.state = state
        self.args = args
        self.stop_event = threading.Event()
        self.source = None
        self.output = WebOutput(state)
        self.pipeline_version = -1
        self.detector = None
        self.color_tracker = None
        self.tracker = None
        self.processing_args = None

    def start(self) -> None:
        threading.Thread(target=self.run, name="m5deflick-processor", daemon=True).start()

    def run(self) -> None:
        try:
            self.source = core.open_source(self.args)
            self.rebuild_pipeline()
            while not self.stop_event.is_set():
                ok, frame = self.source.read()
                if not ok or frame is None:
                    self.set_status("waiting for frame")
                    time.sleep(0.15)
                    continue
                self.process_frame(frame)
        except Exception as exc:
            self.set_status(f"error: {exc}")
        finally:
            if self.source is not None:
                self.source.release()

    def rebuild_pipeline(self) -> None:
        settings = self.get_settings()
        self.processing_args = SimpleNamespace(**settings)
        self.detector = core.MotionDetector(
            self.processing_args.min_motion_area,
            self.processing_args.bg_alpha,
            self.processing_args.motion_threshold,
            self.processing_args.arm_mode,
        )
        self.color_tracker = core.ColorZoneTracker(self.processing_args)
        self.tracker = core.EventTracker(self.processing_args, self.output)
        with self.state.lock:
            self.pipeline_version = self.state.settings_version

    def get_settings(self) -> dict[str, object]:
        with self.state.lock:
            return dict(self.state.settings)

    def process_frame(self, frame: np.ndarray) -> None:
        with self.state.lock:
            if self.pipeline_version != self.state.settings_version:
                needs_rebuild = True
            else:
                needs_rebuild = False
        if needs_rebuild:
            self.rebuild_pipeline()

        h, w = frame.shape[:2]
        with self.state.lock:
            self.state.frame_size = [w, h]
            calibration = self.state.calibration
            recalibrating = self.state.recalibrating
            points = list(self.state.calibration_points)
            reset_background = self.state.reset_background_requested
            reset_zones = self.state.reset_zones_requested
            self.state.reset_background_requested = False
            self.state.reset_zones_requested = False

        if reset_background:
            self.detector.reset()
        if reset_zones:
            self.color_tracker.reset()

        camera_display = draw_camera_frame(frame, calibration, points, recalibrating)
        board_display = np.full((core.BOARD_SIZE, core.BOARD_SIZE, 3), 246, dtype=np.uint8)
        mask = np.zeros((core.BOARD_SIZE, core.BOARD_SIZE), dtype=np.uint8)
        label = ""

        if calibration is not None and not recalibrating:
            warped = cv2.warpPerspective(frame, calibration.matrix, (core.BOARD_SIZE, core.BOARD_SIZE))
            allow_bg_update = not self.tracker.active
            zones = self.color_tracker.update(warped, allow_bg_update)
            centroid, area, mask = self.detector.detect(warped, allow_bg_update)
            label = self.tracker.update(centroid, area, time.monotonic(), zones)
            board_display = core.draw_overlay(warped, centroid, label, zones)
            self.set_status("running")
        else:
            draw_pending_board(board_display, points)
            self.set_status("calibrating")

        camera_jpeg = encode_jpeg(camera_display, 82)
        board_jpeg = encode_jpeg(board_display, 82)
        mask_jpeg = encode_jpeg(cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR), 80)
        with self.state.frame_condition:
            self.state.camera_jpeg = camera_jpeg
            self.state.board_jpeg = board_jpeg
            self.state.mask_jpeg = mask_jpeg
            self.state.last_label = label or self.state.last_label
            self.state.frame_condition.notify_all()

    def set_status(self, status: str) -> None:
        with self.state.lock:
            self.state.status = status


def draw_camera_frame(
    frame: np.ndarray,
    calibration: core.Calibration | None,
    points: list[tuple[float, float]],
    recalibrating: bool,
) -> np.ndarray:
    display = frame.copy()
    active_points = points if recalibrating else (calibration.points if calibration else [])
    pts = np.array(active_points, dtype=np.int32)
    if len(pts) >= 2:
        cv2.polylines(display, [pts], isClosed=len(pts) == 4, color=(16, 96, 230), thickness=3)
    for index, point in enumerate(pts):
        cv2.circle(display, tuple(point), 8, (16, 96, 230), -1)
        cv2.putText(display, str(index + 1), (point[0] + 10, point[1] + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (16, 96, 230), 2)
    return display


def draw_pending_board(canvas: np.ndarray, points: list[tuple[float, float]]) -> None:
    cv2.rectangle(canvas, (150, 150), (874, 874), (210, 210, 210), 2)
    cv2.putText(canvas, f"calibration {len(points)}/4", (310, 520), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (80, 80, 80), 3)


def encode_jpeg(frame: np.ndarray, quality: int) -> bytes:
    ok, buffer = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        return b""
    return buffer.tobytes()


class Handler(SimpleHTTPRequestHandler):
    server: "AppServer"

    def do_GET(self) -> None:
        path = urlparse(self.path).path
        if path == "/":
            self.serve_file(WEB_ROOT / "index.html", "text/html; charset=utf-8")
        elif path == "/api/state":
            self.send_json(self.server.state.snapshot())
        elif path.startswith("/stream/"):
            self.stream_frames(path.rsplit("/", 1)[-1])
        else:
            self.serve_static(path)

    def do_POST(self) -> None:
        path = urlparse(self.path).path
        payload = self.read_json()
        if path == "/api/calibration/reset":
            with self.server.state.lock:
                self.server.state.calibration = None
                self.server.state.calibration_points = []
                self.server.state.recalibrating = True
                self.server.state.reset_background_requested = True
                self.server.state.reset_zones_requested = True
            self.send_json({"ok": True})
        elif path == "/api/calibration/undo":
            with self.server.state.lock:
                if self.server.state.calibration_points:
                    self.server.state.calibration_points.pop()
            self.send_json({"ok": True})
        elif path == "/api/calibration/click":
            self.add_calibration_point(payload)
        elif path == "/api/background/reset":
            with self.server.state.lock:
                self.server.state.reset_background_requested = True
            self.send_json({"ok": True})
        elif path == "/api/output/clear":
            with self.server.state.lock:
                self.server.state.output_text = ""
                self.server.state.events.clear()
            self.send_json({"ok": True})
        elif path == "/api/settings":
            self.update_settings(payload)
        else:
            self.send_error(404)

    def add_calibration_point(self, payload: dict[str, object]) -> None:
        try:
            x = float(payload["x"])
            y = float(payload["y"])
        except (KeyError, TypeError, ValueError):
            self.send_error(400, "x and y are required")
            return
        with self.server.state.lock:
            if not self.server.state.recalibrating:
                self.server.state.calibration_points = []
                self.server.state.calibration = None
                self.server.state.recalibrating = True
            if len(self.server.state.calibration_points) < 4:
                self.server.state.calibration_points.append((x, y))
            if len(self.server.state.calibration_points) == 4:
                calibration = core.Calibration(list(self.server.state.calibration_points))
                calibration.save(self.server.state.calibration_path)
                self.server.state.calibration = calibration
                self.server.state.recalibrating = False
                self.server.state.reset_background_requested = True
                self.server.state.reset_zones_requested = True
        self.send_json({"ok": True})

    def update_settings(self, payload: dict[str, object]) -> None:
        allowed = {
            "zone_mode": str,
            "direction_mode": str,
            "deadzone": float,
            "key_inflate": float,
            "min_motion_area": int,
            "motion_threshold": int,
            "arm_mode": str,
            "cooldown": float,
            "settle": float,
            "send_unicode": bool,
        }
        with self.server.state.lock:
            for key, caster in allowed.items():
                if key not in payload:
                    continue
                value = payload[key]
                if key == "send_unicode":
                    self.server.state.send_unicode = bool(value)
                    continue
                if key in ("zone_mode", "direction_mode", "arm_mode"):
                    self.server.state.settings[key] = str(value)
                else:
                    self.server.state.settings[key] = caster(value)
            self.server.state.settings_version += 1
            self.server.state.reset_background_requested = True
        self.send_json({"ok": True})

    def stream_frames(self, name: str) -> None:
        if name not in {"camera", "board", "mask"}:
            self.send_error(404)
            return
        self.send_response(200)
        self.send_header("Age", "0")
        self.send_header("Cache-Control", "no-cache, private")
        self.send_header("Pragma", "no-cache")
        self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
        self.end_headers()
        last_frame = None
        while True:
            with self.server.state.frame_condition:
                self.server.state.frame_condition.wait(timeout=1.0)
                frame = getattr(self.server.state, f"{name}_jpeg")
            if not frame or frame == last_frame:
                continue
            last_frame = frame
            try:
                self.wfile.write(b"--frame\r\n")
                self.wfile.write(b"Content-Type: image/jpeg\r\n")
                self.wfile.write(f"Content-Length: {len(frame)}\r\n\r\n".encode("ascii"))
                self.wfile.write(frame)
                self.wfile.write(b"\r\n")
            except (BrokenPipeError, ConnectionResetError):
                break

    def serve_static(self, path: str) -> None:
        relative = unquote(path.lstrip("/"))
        target = (WEB_ROOT / relative).resolve()
        if WEB_ROOT.resolve() not in target.parents and target != WEB_ROOT.resolve():
            self.send_error(403)
            return
        if not target.is_file():
            self.send_error(404)
            return
        content_type = "application/octet-stream"
        if target.suffix == ".css":
            content_type = "text/css; charset=utf-8"
        elif target.suffix == ".js":
            content_type = "application/javascript; charset=utf-8"
        elif target.suffix == ".html":
            content_type = "text/html; charset=utf-8"
        self.serve_file(target, content_type)

    def serve_file(self, path: Path, content_type: str) -> None:
        data = path.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def read_json(self) -> dict[str, object]:
        length = int(self.headers.get("content-length", "0") or 0)
        if length == 0:
            return {}
        return json.loads(self.rfile.read(length).decode("utf-8"))

    def send_json(self, data: dict[str, object]) -> None:
        body = json.dumps(data, ensure_ascii=False).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format: str, *args: object) -> None:
        return


class AppServer(ThreadingHTTPServer):
    def __init__(self, address: tuple[str, int], state: WebState):
        super().__init__(address, Handler)
        self.state = state


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Browser UI for M5Deflick.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8787)
    parser.add_argument("--source", default="unitv2")
    parser.add_argument("--source-url", default="")
    parser.add_argument("--unitv2-host", default="10.254.239.1")
    parser.add_argument("--calibration", default=core.CALIBRATION_FILE)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    state = WebState(args.calibration)
    processor_args = argparse.Namespace(
        source=args.source,
        source_url=args.source_url,
        unitv2_host=args.unitv2_host,
    )
    processor = Processor(state, processor_args)
    processor.start()
    server = AppServer((args.host, args.port), state)
    print(f"M5Deflick UI: http://{args.host}:{args.port}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        processor.stop_event.set()
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
