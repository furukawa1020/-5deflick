from __future__ import annotations

import argparse
import json
import os
import shutil
import tempfile
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
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".mplcache"))


def default_hand_model_path() -> str:
    source = ROOT / "models" / "hand_landmarker.task"
    target_dir = Path(tempfile.gettempdir()) / "m5deflick"
    target = target_dir / "hand_landmarker.task"
    try:
        target_dir.mkdir(parents=True, exist_ok=True)
        if source.exists() and (not target.exists() or source.stat().st_mtime > target.stat().st_mtime):
            shutil.copyfile(source, target)
        if target.exists():
            return str(target)
    except OSError:
        pass
    return str(source)


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
        self.pending_marker_sample: tuple[float, float] | None = None
        self.marker_hsv: tuple[int, int, int] | None = None
        self.settings_version = 0
        self.send_unicode = False
        self.settings = {
            "input_mode": "red_glove",
            "zone_mode": "layout",
            "direction_mode": "side",
            "deadzone": 0.20,
            "key_inflate": 0.80,
            "min_motion_area": 650,
            "motion_threshold": 18,
            "arm_mode": "combined",
            "cooldown": 0.28,
            "settle": 0.10,
            "max_hold": 0.18,
            "bg_alpha": 0.006,
            "color_min_area": 500,
            "color_alpha": 0.28,
            "color_search_inflate": 0.65,
            "hand_model_path": default_hand_model_path(),
            "hand_confidence": 0.35,
            "marker_min_area": 120,
            "marker_hue_margin": 14,
            "marker_sat_margin": 70,
            "marker_val_margin": 80,
            "target_fps": 16,
            "preview_width": 520,
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
                "markerReady": self.marker_hsv is not None,
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


class MediaPipeHandDetector:
    def __init__(self, model_path: str, confidence: float):
        import mediapipe as mp
        from mediapipe.tasks.python import vision

        if not os.path.exists(model_path):
            raise RuntimeError(f"MediaPipe hand model not found: {model_path}")
        self.mp = mp
        self.vision = vision
        base_options = mp.tasks.BaseOptions(model_asset_path=model_path)
        options = vision.HandLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.VIDEO,
            num_hands=2,
            min_hand_detection_confidence=confidence,
            min_hand_presence_confidence=confidence,
            min_tracking_confidence=0.35,
        )
        self.landmarker = vision.HandLandmarker.create_from_options(options)
        self.started_at = time.monotonic()

    def close(self) -> None:
        self.landmarker.close()

    def detect(self, frame: np.ndarray) -> tuple[tuple[float, float] | None, list[tuple[float, float]]]:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = self.mp.Image(image_format=self.mp.ImageFormat.SRGB, data=rgb)
        timestamp_ms = int((time.monotonic() - self.started_at) * 1000)
        result = self.landmarker.detect_for_video(mp_image, timestamp_ms)
        if not result.hand_landmarks:
            return None, []

        h, w = frame.shape[:2]
        hand = min(result.hand_landmarks, key=lambda landmarks: palm_y(landmarks))
        points = [(float(landmark.x * w), float(landmark.y * h)) for landmark in hand]
        palm_indices = (0, 5, 9, 13, 17)
        palm = (
            sum(points[index][0] for index in palm_indices) / len(palm_indices),
            sum(points[index][1] for index in palm_indices) / len(palm_indices),
        )
        return palm, points


def palm_y(landmarks) -> float:
    return sum(landmarks[index].y for index in (0, 5, 9, 13, 17)) / 5.0


class MarkerDetector:
    def __init__(self, settings: SimpleNamespace):
        self.settings = settings
        self.hsv: tuple[int, int, int] | None = None

    def set_hsv(self, hsv: tuple[int, int, int] | None) -> None:
        self.hsv = hsv

    def sample_from_frame(self, frame: np.ndarray, point: tuple[float, float]) -> tuple[int, int, int]:
        x = int(round(point[0]))
        y = int(round(point[1]))
        height, width = frame.shape[:2]
        x1, x2 = max(0, x - 6), min(width, x + 7)
        y1, y2 = max(0, y - 6), min(height, y + 7)
        patch = frame[y1:y2, x1:x2]
        if patch.size == 0:
            raise ValueError("marker sample is outside the camera frame")
        hsv_patch = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
        median = np.median(hsv_patch.reshape(-1, 3), axis=0)
        self.hsv = (int(median[0]), int(median[1]), int(median[2]))
        return self.hsv

    def detect(self, frame: np.ndarray) -> tuple[tuple[float, float] | None, float, np.ndarray]:
        mask = np.zeros(frame.shape[:2], dtype=np.uint8)
        if self.hsv is None:
            return None, 0.0, mask
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        h, s, v = self.hsv
        hue_margin = int(self.settings.marker_hue_margin)
        sat_margin = int(self.settings.marker_sat_margin)
        val_margin = int(self.settings.marker_val_margin)
        lower_sv = (max(35, s - sat_margin), max(35, v - val_margin))
        upper_sv = (255, min(255, v + val_margin))

        ranges: list[tuple[int, int]] = []
        low_h = h - hue_margin
        high_h = h + hue_margin
        if low_h < 0:
            ranges.append((0, high_h))
            ranges.append((180 + low_h, 179))
        elif high_h > 179:
            ranges.append((low_h, 179))
            ranges.append((0, high_h - 180))
        else:
            ranges.append((low_h, high_h))

        for hue_low, hue_high in ranges:
            lower = np.array((hue_low, lower_sv[0], lower_sv[1]), dtype=np.uint8)
            upper = np.array((hue_high, upper_sv[0], upper_sv[1]), dtype=np.uint8)
            mask |= cv2.inRange(hsv, lower, upper)

        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((17, 17), np.uint8))
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None, 0.0, mask
        contour = max(contours, key=cv2.contourArea)
        area = float(cv2.contourArea(contour))
        if area < self.settings.marker_min_area:
            return None, area, mask
        moments = cv2.moments(contour)
        if moments["m00"] == 0:
            return None, area, mask
        centroid = (moments["m10"] / moments["m00"], moments["m01"] / moments["m00"])
        return centroid, area, mask


class RedGloveDetector:
    def __init__(self, settings: SimpleNamespace):
        self.settings = settings
        self.open_kernel = np.ones((5, 5), np.uint8)
        self.close_kernel = np.ones((19, 19), np.uint8)

    def detect(self, frame: np.ndarray) -> tuple[tuple[float, float] | None, float, np.ndarray]:
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        lower_red = cv2.inRange(hsv, np.array((0, 70, 45), dtype=np.uint8), np.array((13, 255, 255), dtype=np.uint8))
        upper_red = cv2.inRange(hsv, np.array((166, 70, 45), dtype=np.uint8), np.array((179, 255, 255), dtype=np.uint8))
        mask = cv2.bitwise_or(lower_red, upper_red)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self.open_kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, self.close_kernel)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None, 0.0, mask
        contour = max(contours, key=cv2.contourArea)
        area = float(cv2.contourArea(contour))
        if area < self.settings.marker_min_area:
            return None, area, mask
        moments = cv2.moments(contour)
        if moments["m00"] == 0:
            return None, area, mask
        centroid = (moments["m10"] / moments["m00"], moments["m01"] / moments["m00"])
        return centroid, area, mask


class LatestFrameSource:
    def __init__(
        self,
        args: argparse.Namespace,
        prepare_source,
        set_status,
        stop_event: threading.Event,
    ):
        self.args = args
        self.prepare_source = prepare_source
        self.set_status = set_status
        self.stop_event = stop_event
        self.condition = threading.Condition()
        self.source = None
        self.frame: np.ndarray | None = None
        self.sequence = 0
        self.closed = False
        self.thread = threading.Thread(target=self.run, name="m5deflick-frame-reader", daemon=True)
        self.thread.start()

    def run(self) -> None:
        while not self.stop_event.is_set() and not self.closed:
            if self.source is None:
                try:
                    self.set_status("connecting to UnitV2")
                    self.prepare_source()
                    self.source = core.open_source(self.args)
                except Exception as exc:
                    self.set_status(f"waiting for UnitV2: {exc}")
                    self.stop_event.wait(1.0)
                    continue
            try:
                ok, frame = self.source.read()
            except Exception as exc:
                self.set_status(f"reconnecting: {exc}")
                self.release_source()
                self.stop_event.wait(0.35)
                continue
            if not ok or frame is None:
                self.set_status("waiting for frame")
                self.stop_event.wait(0.03)
                continue
            with self.condition:
                self.frame = frame
                self.sequence += 1
                self.condition.notify_all()

    def read(self, last_sequence: int, timeout: float = 1.0) -> tuple[bool, np.ndarray | None, int]:
        deadline = time.monotonic() + timeout
        with self.condition:
            while self.sequence == last_sequence and not self.stop_event.is_set() and not self.closed:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    break
                self.condition.wait(remaining)
            if self.frame is None or self.sequence == last_sequence:
                return False, None, last_sequence
            return True, self.frame.copy(), self.sequence

    def release_source(self) -> None:
        if self.source is not None:
            try:
                self.source.release()
            except Exception:
                pass
            self.source = None

    def release(self) -> None:
        self.closed = True
        with self.condition:
            self.condition.notify_all()
        self.release_source()


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
        self.hand_detector = None
        self.marker_detector = None
        self.red_glove_detector = None
        self.processing_args = None
        self.last_processed_at = 0.0

    def start(self) -> None:
        threading.Thread(target=self.run, name="m5deflick-processor", daemon=True).start()

    def run(self) -> None:
        try:
            self.rebuild_pipeline()
            self.source = LatestFrameSource(self.args, self.prepare_source, self.set_status, self.stop_event)
            last_sequence = 0
            while not self.stop_event.is_set():
                ok, frame, last_sequence = self.source.read(last_sequence)
                if not ok or frame is None:
                    continue
                now = time.monotonic()
                target_interval = 1.0 / max(float(getattr(self.processing_args, "target_fps", 12)), 1.0)
                if now - self.last_processed_at < target_interval:
                    continue
                self.last_processed_at = now
                self.process_frame(frame)
        finally:
            if self.hand_detector is not None:
                self.hand_detector.close()
            if self.source is not None:
                self.source.release()

    def rebuild_pipeline(self) -> None:
        settings = self.get_settings()
        self.processing_args = SimpleNamespace(**settings)
        if self.hand_detector is not None:
            self.hand_detector.close()
            self.hand_detector = None
        self.detector = core.MotionDetector(
            self.processing_args.min_motion_area,
            self.processing_args.bg_alpha,
            self.processing_args.motion_threshold,
            self.processing_args.arm_mode,
        )
        self.color_tracker = core.ColorZoneTracker(self.processing_args)
        self.tracker = core.EventTracker(self.processing_args, self.output)
        self.marker_detector = MarkerDetector(self.processing_args)
        self.red_glove_detector = RedGloveDetector(self.processing_args)
        with self.state.lock:
            self.marker_detector.set_hsv(self.state.marker_hsv)
        if self.processing_args.input_mode == "mediapipe":
            self.hand_detector = MediaPipeHandDetector(
                self.processing_args.hand_model_path,
                self.processing_args.hand_confidence,
            )
        with self.state.lock:
            self.pipeline_version = self.state.settings_version

    def prepare_source(self) -> None:
        target = self.args.source_url or self.args.unitv2_host
        parsed = urlparse(target if "://" in target else "http://" + target)
        if parsed.hostname == "10.254.239.1":
            core.prepare_unitv2_camera_stream("http://10.254.239.1")

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
            pending_marker_sample = self.state.pending_marker_sample
            marker_hsv = self.state.marker_hsv
            self.state.reset_background_requested = False
            self.state.reset_zones_requested = False
            self.state.pending_marker_sample = None

        if reset_background:
            self.detector.reset()
        if reset_zones:
            self.color_tracker.reset()
        if marker_hsv is not None:
            self.marker_detector.set_hsv(marker_hsv)
        if pending_marker_sample is not None:
            sampled_hsv = self.marker_detector.sample_from_frame(frame, pending_marker_sample)
            with self.state.lock:
                self.state.marker_hsv = sampled_hsv
                self.state.events.appendleft({"kind": "key", "value": "marker-set", "at": time.time()})

        camera_display = draw_camera_frame(frame, calibration, points, recalibrating)
        board_display = np.full((core.BOARD_SIZE, core.BOARD_SIZE, 3), 246, dtype=np.uint8)
        mask = np.zeros((core.BOARD_SIZE, core.BOARD_SIZE), dtype=np.uint8)
        label = ""

        if calibration is not None and not recalibrating:
            warped = cv2.warpPerspective(frame, calibration.matrix, (core.BOARD_SIZE, core.BOARD_SIZE))
            allow_bg_update = not self.tracker.active
            zones = self.color_tracker.update(warped, allow_bg_update)

            if self.processing_args.input_mode == "red_glove":
                camera_centroid, area, mask = self.red_glove_detector.detect(frame)
                if camera_centroid is not None:
                    cv2.circle(camera_display, tuple(map(int, camera_centroid)), 18, (0, 0, 255), 4)
                    cv2.circle(mask, tuple(map(int, camera_centroid)), 18, 255, 4)
                centroid = transform_point(camera_centroid, calibration.matrix)
                if centroid is not None and core.nearest_zone(centroid, zones, inflate=self.processing_args.key_inflate) is None:
                    centroid = None
                label = self.tracker.update(centroid, area if centroid is not None else 0.0, time.monotonic(), zones)
                self.set_status("running: red glove" if camera_centroid is not None else "running: no red glove")
            elif self.processing_args.input_mode == "marker":
                camera_centroid, area, mask = self.marker_detector.detect(frame)
                if camera_centroid is not None:
                    cv2.circle(camera_display, tuple(map(int, camera_centroid)), 18, (0, 0, 255), 4)
                    cv2.circle(mask, tuple(map(int, camera_centroid)), 18, 255, 4)
                centroid = transform_point(camera_centroid, calibration.matrix)
                if centroid is not None and core.nearest_zone(centroid, zones, inflate=self.processing_args.key_inflate) is None:
                    centroid = None
                label = self.tracker.update(centroid, area if centroid is not None else 0.0, time.monotonic(), zones)
                self.set_status("running: marker" if self.marker_detector.hsv is not None else "running: sample marker")
            elif self.processing_args.input_mode == "mediapipe":
                hand_palm, hand_points = self.hand_detector.detect(frame)
                draw_hand_points(camera_display, hand_points)
                board_points = transform_points(hand_points, calibration.matrix)
                mask = draw_hand_mask(board_points)
                centroid = transform_point(hand_palm, calibration.matrix) if hand_palm is not None else None
                if centroid is not None and core.nearest_zone(centroid, zones, inflate=self.processing_args.key_inflate) is None:
                    centroid = None
                label = self.tracker.update(centroid, 4500.0 if centroid is not None else 0.0, time.monotonic(), zones)
                self.set_status("running: hand" if hand_palm is not None else "running: no hand")
            else:
                centroid, area, mask = self.detector.detect(warped, allow_bg_update)
                label = self.tracker.update(centroid, area, time.monotonic(), zones)
                self.set_status("running")
            board_display = core.draw_overlay(warped, centroid, label, zones)
        else:
            draw_pending_board(board_display, points)
            self.set_status("calibrating")

        preview_width = int(getattr(self.processing_args, "preview_width", 640))
        camera_jpeg = encode_jpeg(fit_preview(camera_display, preview_width), 72)
        board_jpeg = encode_jpeg(fit_preview(board_display, preview_width), 72)
        mask_jpeg = encode_jpeg(fit_preview(cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR), preview_width), 68)
        with self.state.frame_condition:
            self.state.camera_jpeg = camera_jpeg
            self.state.board_jpeg = board_jpeg
            self.state.mask_jpeg = mask_jpeg
            self.state.last_label = label or self.state.last_label
            self.state.frame_condition.notify_all()

    def set_status(self, status: str) -> None:
        with self.state.lock:
            self.state.status = status


HAND_EDGES = (
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (0, 9), (9, 10), (10, 11), (11, 12),
    (0, 13), (13, 14), (14, 15), (15, 16),
    (0, 17), (17, 18), (18, 19), (19, 20),
)


def transform_point(point: tuple[float, float] | None, matrix: np.ndarray) -> tuple[float, float] | None:
    if point is None:
        return None
    pts = np.array([[[point[0], point[1]]]], dtype=np.float32)
    mapped = cv2.perspectiveTransform(pts, matrix)[0][0]
    return (float(mapped[0]), float(mapped[1]))


def transform_points(points: list[tuple[float, float]], matrix: np.ndarray) -> list[tuple[float, float]]:
    if not points:
        return []
    pts = np.array([[points]], dtype=np.float32)
    mapped = cv2.perspectiveTransform(pts, matrix)[0]
    return [(float(point[0]), float(point[1])) for point in mapped]


def draw_hand_points(display: np.ndarray, points: list[tuple[float, float]]) -> None:
    if not points:
        return
    for start, end in HAND_EDGES:
        cv2.line(display, tuple(map(int, points[start])), tuple(map(int, points[end])), (44, 190, 92), 2)
    for point in points:
        cv2.circle(display, tuple(map(int, point)), 4, (44, 190, 92), -1)


def draw_hand_mask(points: list[tuple[float, float]]) -> np.ndarray:
    mask = np.zeros((core.BOARD_SIZE, core.BOARD_SIZE), dtype=np.uint8)
    if not points:
        return mask
    for start, end in HAND_EDGES:
        cv2.line(mask, tuple(map(int, points[start])), tuple(map(int, points[end])), 255, 8)
    for point in points:
        cv2.circle(mask, tuple(map(int, point)), 10, 255, -1)
    return cv2.dilate(mask, np.ones((13, 13), np.uint8), iterations=1)


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


def fit_preview(frame: np.ndarray, width: int) -> np.ndarray:
    height, current_width = frame.shape[:2]
    if current_width <= width:
        return frame
    new_height = max(1, int(round(height * width / current_width)))
    return cv2.resize(frame, (width, new_height), interpolation=cv2.INTER_AREA)


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
        elif path == "/api/marker/sample":
            self.sample_marker(payload)
        elif path == "/api/marker/clear":
            with self.server.state.lock:
                self.server.state.marker_hsv = None
                self.server.state.pending_marker_sample = None
                self.server.state.events.appendleft({"kind": "key", "value": "marker-clear", "at": time.time()})
            self.send_json({"ok": True})
        elif path == "/api/background/reset":
            with self.server.state.lock:
                self.server.state.reset_background_requested = True
            self.send_json({"ok": True})
        elif path == "/api/output/clear":
            with self.server.state.lock:
                self.server.state.output_text = ""
                self.server.state.events.clear()
            self.send_json({"ok": True})
        elif path == "/api/output/test":
            self.test_output()
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

    def sample_marker(self, payload: dict[str, object]) -> None:
        try:
            x = float(payload["x"])
            y = float(payload["y"])
        except (KeyError, TypeError, ValueError):
            self.send_error(400, "x and y are required")
            return
        with self.server.state.lock:
            self.server.state.pending_marker_sample = (x, y)
            self.server.state.settings["input_mode"] = "marker"
            self.server.state.settings_version += 1
        self.send_json({"ok": True})

    def update_settings(self, payload: dict[str, object]) -> None:
        allowed = {
            "input_mode": str,
            "zone_mode": str,
            "direction_mode": str,
            "deadzone": float,
            "key_inflate": float,
            "min_motion_area": int,
            "motion_threshold": int,
            "arm_mode": str,
            "cooldown": float,
            "settle": float,
            "max_hold": float,
            "send_unicode": bool,
            "hand_confidence": float,
            "marker_min_area": int,
            "marker_hue_margin": int,
            "target_fps": int,
            "preview_width": int,
        }
        with self.server.state.lock:
            for key, caster in allowed.items():
                if key not in payload:
                    continue
                value = payload[key]
                if key == "send_unicode":
                    self.server.state.send_unicode = bool(value)
                    continue
                if key in ("input_mode", "zone_mode", "direction_mode", "arm_mode"):
                    self.server.state.settings[key] = str(value)
                else:
                    self.server.state.settings[key] = caster(value)
            self.server.state.settings_version += 1
            self.server.state.reset_background_requested = True
        self.send_json({"ok": True})

    def test_output(self) -> None:
        value = "あ"
        send_unicode = False
        with self.server.state.lock:
            self.server.state.output_text += value
            self.server.state.events.appendleft({"kind": "text", "value": value, "at": time.time()})
            send_unicode = self.server.state.send_unicode
        if send_unicode:
            try:
                core.WindowsInput().emit_text(value)
            except Exception as exc:
                with self.server.state.lock:
                    self.server.state.send_unicode = False
                    self.server.state.status = f"unicode input disabled: {exc}"
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
            except (BrokenPipeError, ConnectionAbortedError, ConnectionResetError):
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
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        try:
            self.wfile.write(data)
        except (BrokenPipeError, ConnectionAbortedError, ConnectionResetError):
            pass

    def read_json(self) -> dict[str, object]:
        length = int(self.headers.get("content-length", "0") or 0)
        if length == 0:
            return {}
        return json.loads(self.rfile.read(length).decode("utf-8"))

    def send_json(self, data: dict[str, object]) -> None:
        body = json.dumps(data, ensure_ascii=False).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        try:
            self.wfile.write(body)
        except (BrokenPipeError, ConnectionAbortedError, ConnectionResetError):
            pass

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
