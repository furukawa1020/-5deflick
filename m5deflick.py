from __future__ import annotations

import argparse
import collections
import ctypes
import json
import math
import os
import re
import sys
import time
from dataclasses import dataclass
from typing import Iterable, Optional
from urllib.parse import urljoin, urlparse

import cv2
import numpy as np

try:
    import requests
except ImportError:  # pragma: no cover - surfaced as a clear runtime error.
    requests = None


BOARD_SIZE = 1024
CALIBRATION_FILE = "m5deflick_calibration.json"


@dataclass(frozen=True)
class Zone:
    name: str
    kind: str
    x1: int
    y1: int
    x2: int
    y2: int

    @property
    def center(self) -> tuple[float, float]:
        return ((self.x1 + self.x2) * 0.5, (self.y1 + self.y2) * 0.5)

    @property
    def width(self) -> float:
        return float(self.x2 - self.x1)

    @property
    def height(self) -> float:
        return float(self.y2 - self.y1)

    def contains(self, point: tuple[float, float], inflate: float = 0.0) -> bool:
        px, py = point
        margin_x = self.width * inflate
        margin_y = self.height * inflate
        return (
            self.x1 - margin_x <= px <= self.x2 + margin_x
            and self.y1 - margin_y <= py <= self.y2 + margin_y
        )


KANA_ZONES = [
    Zone("あ", "kana", 252, 188, 430, 366),
    Zone("か", "kana", 448, 196, 618, 367),
    Zone("さ", "kana", 638, 198, 799, 369),
    Zone("た", "kana", 255, 384, 432, 563),
    Zone("な", "kana", 451, 386, 619, 561),
    Zone("は", "kana", 640, 386, 801, 560),
    Zone("ま", "kana", 252, 579, 431, 758),
    Zone("や", "kana", 450, 581, 619, 756),
    Zone("ら", "kana", 639, 581, 801, 756),
]

SPECIAL_ZONES = [
    Zone("backspace", "key", 790, 735, 990, 944),
    Zone("escape", "key", 852, 160, 968, 270),
]

ZONES = KANA_ZONES + SPECIAL_ZONES

FLICK_MAP = {
    "あ": {"center": "あ", "left": "い", "up": "う", "right": "え", "down": "お"},
    "か": {"center": "か", "left": "き", "up": "く", "right": "け", "down": "こ"},
    "さ": {"center": "さ", "left": "し", "up": "す", "right": "せ", "down": "そ"},
    "た": {"center": "た", "left": "ち", "up": "つ", "right": "て", "down": "と"},
    "な": {"center": "な", "left": "に", "up": "ぬ", "right": "ね", "down": "の"},
    "は": {"center": "は", "left": "ひ", "up": "ふ", "right": "へ", "down": "ほ"},
    "ま": {"center": "ま", "left": "み", "up": "む", "right": "め", "down": "も"},
    "や": {"center": "や", "left": "ゆ", "up": "ゆ", "right": "よ", "down": "よ"},
    "ら": {"center": "ら", "left": "り", "up": "る", "right": "れ", "down": "ろ"},
}

ZONE_COLORS = {
    "あ": "yellow",
    "か": "cyan",
    "さ": "pink",
    "た": "orange",
    "な": "green",
    "は": "yellow",
    "ま": "cyan",
    "や": "blue",
    "ら": "magenta",
}

HSV_RANGES = {
    "yellow": [((18, 70, 70), (42, 255, 255))],
    "orange": [((4, 70, 70), (24, 255, 255))],
    "green": [((40, 55, 55), (82, 255, 255))],
    "cyan": [((78, 45, 55), (104, 255, 255))],
    "blue": [((95, 55, 45), (125, 255, 255))],
    "pink": [((145, 45, 70), (179, 255, 255))],
    "magenta": [((135, 45, 55), (172, 255, 255))],
}


class Cv2Source:
    def __init__(self, source: str | int):
        if isinstance(source, int) and os.name == "nt":
            self.capture = cv2.VideoCapture(source, cv2.CAP_DSHOW)
        else:
            self.capture = cv2.VideoCapture(source)

    def read(self) -> tuple[bool, Optional[np.ndarray]]:
        return self.capture.read()

    def release(self) -> None:
        self.capture.release()


class MjpegSource:
    def __init__(self, url: str):
        if requests is None:
            raise RuntimeError("requests is required for HTTP/MJPEG sources. Run: pip install -r requirements.txt")
        self.url = url
        self.response = None
        self.iterator = None
        self.buffer = bytearray()
        self._connect()

    def _connect(self) -> None:
        self.release()
        self.response = requests.get(self.url, stream=True, timeout=(3, 8))
        self.response.raise_for_status()
        self.iterator = self.response.iter_content(chunk_size=4096)
        self.buffer.clear()

    def read(self) -> tuple[bool, Optional[np.ndarray]]:
        assert self.iterator is not None
        for _ in range(120):
            try:
                chunk = next(self.iterator)
            except StopIteration:
                self._connect()
                continue
            except Exception:
                time.sleep(0.2)
                self._connect()
                continue

            if not chunk:
                continue
            self.buffer.extend(chunk)
            start = self.buffer.find(b"\xff\xd8")
            end = self.buffer.find(b"\xff\xd9", start + 2)
            if start != -1 and end != -1:
                jpg = bytes(self.buffer[start : end + 2])
                del self.buffer[: end + 2]
                frame = cv2.imdecode(np.frombuffer(jpg, dtype=np.uint8), cv2.IMREAD_COLOR)
                if frame is not None:
                    return True, frame
        return False, None

    def release(self) -> None:
        if self.response is not None:
            self.response.close()
        self.response = None
        self.iterator = None


class WindowsInput:
    INPUT_KEYBOARD = 1
    KEYEVENTF_KEYUP = 0x0002
    KEYEVENTF_UNICODE = 0x0004
    VK_BACK = 0x08
    VK_ESCAPE = 0x1B

    class KEYBDINPUT(ctypes.Structure):
        _fields_ = [
            ("wVk", ctypes.c_ushort),
            ("wScan", ctypes.c_ushort),
            ("dwFlags", ctypes.c_ulong),
            ("time", ctypes.c_ulong),
            ("dwExtraInfo", ctypes.POINTER(ctypes.c_ulong)),
        ]

    class INPUT_UNION(ctypes.Union):
        _fields_ = [("ki", KEYBDINPUT)]

    class INPUT(ctypes.Structure):
        _fields_ = [("type", ctypes.c_ulong), ("union", INPUT_UNION)]

    def __init__(self) -> None:
        if os.name != "nt":
            raise RuntimeError("--output unicode is currently implemented for Windows only.")

    def emit_text(self, text: str) -> None:
        for char in text:
            self._send_unicode_char(char)

    def emit_key(self, key_name: str) -> None:
        vk = {"backspace": self.VK_BACK, "escape": self.VK_ESCAPE}[key_name]
        self._send_vk(vk, key_up=False)
        self._send_vk(vk, key_up=True)

    def _send_unicode_char(self, char: str) -> None:
        code_point = ord(char)
        if code_point > 0xFFFF:
            for unit in char.encode("utf-16-le"):
                self._send_scan(unit)
            return
        self._send_scan(code_point)

    def _send_scan(self, scan_code: int) -> None:
        self._send_keyboard(w_vk=0, w_scan=scan_code, flags=self.KEYEVENTF_UNICODE)
        self._send_keyboard(w_vk=0, w_scan=scan_code, flags=self.KEYEVENTF_UNICODE | self.KEYEVENTF_KEYUP)

    def _send_vk(self, vk: int, key_up: bool) -> None:
        flags = self.KEYEVENTF_KEYUP if key_up else 0
        self._send_keyboard(w_vk=vk, w_scan=0, flags=flags)

    def _send_keyboard(self, w_vk: int, w_scan: int, flags: int) -> None:
        extra = ctypes.c_ulong(0)
        keyboard = self.KEYBDINPUT(w_vk, w_scan, flags, 0, ctypes.pointer(extra))
        event = self.INPUT(self.INPUT_KEYBOARD, self.INPUT_UNION(ki=keyboard))
        sent = ctypes.windll.user32.SendInput(1, ctypes.pointer(event), ctypes.sizeof(event))
        if sent != 1:
            raise ctypes.WinError()


class Output:
    def __init__(self, mode: str):
        self.mode = mode
        self.windows_input = WindowsInput() if mode == "unicode" else None

    def emit_text(self, text: str) -> None:
        if self.mode == "none":
            return
        if self.mode == "print":
            print(text, flush=True)
            return
        assert self.windows_input is not None
        self.windows_input.emit_text(text)
        print(text, flush=True)

    def emit_key(self, key_name: str) -> None:
        if self.mode == "none":
            return
        if self.mode == "print":
            print(f"<{key_name}>", flush=True)
            return
        assert self.windows_input is not None
        self.windows_input.emit_key(key_name)
        print(f"<{key_name}>", flush=True)


@dataclass
class Calibration:
    points: list[tuple[float, float]]

    @property
    def matrix(self) -> np.ndarray:
        source = np.float32(self.points)
        target = np.float32(
            [
                [0, 0],
                [BOARD_SIZE - 1, 0],
                [BOARD_SIZE - 1, BOARD_SIZE - 1],
                [0, BOARD_SIZE - 1],
            ]
        )
        return cv2.getPerspectiveTransform(source, target)

    def save(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as handle:
            json.dump({"points": self.points}, handle, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, path: str) -> Optional["Calibration"]:
        if not os.path.exists(path):
            return None
        with open(path, "r", encoding="utf-8") as handle:
            data = json.load(handle)
        points = data.get("points")
        if not isinstance(points, list) or len(points) != 4:
            return None
        return cls([(float(x), float(y)) for x, y in points])


class ColorZoneTracker:
    def __init__(self, args: argparse.Namespace):
        self.enabled = args.zone_mode == "color"
        self.min_area = args.color_min_area
        self.alpha = args.color_alpha
        self.search_inflate = args.color_search_inflate
        self.zones_by_name = {zone.name: zone for zone in ZONES}

    @property
    def zones(self) -> list[Zone]:
        return [self.zones_by_name[zone.name] for zone in ZONES]

    def reset(self) -> None:
        self.zones_by_name = {zone.name: zone for zone in ZONES}

    def update(self, warped: np.ndarray, allow_update: bool) -> list[Zone]:
        if not self.enabled or not allow_update:
            return self.zones

        hsv = cv2.cvtColor(warped, cv2.COLOR_BGR2HSV)
        for base_zone in KANA_ZONES:
            detected = self._detect_colored_zone(hsv, base_zone)
            if detected is None:
                continue
            previous = self.zones_by_name[base_zone.name]
            self.zones_by_name[base_zone.name] = blend_zone(previous, detected, self.alpha)
        return self.zones

    def _detect_colored_zone(self, hsv: np.ndarray, base_zone: Zone) -> Optional[Zone]:
        current_zone = self.zones_by_name[base_zone.name]
        x1, y1, x2, y2 = inflated_bounds(current_zone, self.search_inflate)
        roi = hsv[y1:y2, x1:x2]
        if roi.size == 0:
            return None

        color_name = ZONE_COLORS[base_zone.name]
        mask = np.zeros(roi.shape[:2], dtype=np.uint8)
        for lower, upper in HSV_RANGES[color_name]:
            mask |= cv2.inRange(roi, np.array(lower, dtype=np.uint8), np.array(upper, dtype=np.uint8))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((17, 17), np.uint8))

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None
        contour = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(contour)
        if area < self.min_area:
            return None

        x, y, w, h = cv2.boundingRect(contour)
        if w < 35 or h < 35:
            return None
        return Zone(base_zone.name, base_zone.kind, x1 + x, y1 + y, x1 + x + w, y1 + y + h)


def inflated_bounds(zone: Zone, inflate: float) -> tuple[int, int, int, int]:
    margin_x = int(zone.width * inflate)
    margin_y = int(zone.height * inflate)
    return (
        max(0, zone.x1 - margin_x),
        max(0, zone.y1 - margin_y),
        min(BOARD_SIZE, zone.x2 + margin_x),
        min(BOARD_SIZE, zone.y2 + margin_y),
    )


def blend_zone(previous: Zone, detected: Zone, alpha: float) -> Zone:
    def mix(old: int, new: int) -> int:
        return int(round(old * (1.0 - alpha) + new * alpha))

    return Zone(
        previous.name,
        previous.kind,
        mix(previous.x1, detected.x1),
        mix(previous.y1, detected.y1),
        mix(previous.x2, detected.x2),
        mix(previous.y2, detected.y2),
    )


class MotionDetector:
    def __init__(self, min_motion_area: int, bg_alpha: float):
        self.min_motion_area = min_motion_area
        self.bg_alpha = bg_alpha
        self.background: Optional[np.ndarray] = None

    def reset(self) -> None:
        self.background = None

    def detect(self, warped: np.ndarray, allow_background_update: bool) -> tuple[Optional[tuple[float, float]], float, np.ndarray]:
        gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (15, 15), 0)
        if self.background is None:
            self.background = gray.astype(np.float32)
            return None, 0.0, np.zeros_like(gray)

        background_u8 = cv2.convertScaleAbs(self.background)
        diff = cv2.absdiff(gray, background_u8)
        _, mask = cv2.threshold(diff, 35, 255, cv2.THRESH_BINARY)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
        mask = cv2.dilate(mask, np.ones((13, 13), np.uint8), iterations=2)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            if allow_background_update:
                cv2.accumulateWeighted(gray, self.background, self.bg_alpha)
            return None, 0.0, mask

        contour = max(contours, key=cv2.contourArea)
        area = float(cv2.contourArea(contour))
        if area < self.min_motion_area:
            if allow_background_update:
                cv2.accumulateWeighted(gray, self.background, self.bg_alpha)
            return None, area, mask

        moments = cv2.moments(contour)
        if moments["m00"] == 0:
            return None, area, mask
        centroid = (moments["m10"] / moments["m00"], moments["m01"] / moments["m00"])
        return centroid, area, mask


class EventTracker:
    def __init__(self, args: argparse.Namespace, output: Output):
        self.args = args
        self.output = output
        self.active = False
        self.points: list[tuple[float, tuple[float, float], float]] = []
        self.last_motion_at = 0.0
        self.last_emit_at = 0.0
        self.last_label = ""

    def update(self, centroid: Optional[tuple[float, float]], area: float, now: float) -> str:
        if centroid is not None:
            if not self.active and now - self.last_emit_at >= self.args.cooldown:
                self.active = True
                self.points = []
            if self.active:
                self.points.append((now, centroid, area))
                self.last_motion_at = now
            return self.last_label

        if self.active and now - self.last_motion_at >= self.args.settle:
            label = self._classify_and_emit()
            self.active = False
            self.points = []
            self.last_emit_at = now
            self.last_label = label
        return self.last_label

    def _classify_and_emit(self) -> str:
        if not self.points:
            return ""

        candidates: list[tuple[Zone, float, tuple[float, float]]] = []
        for _, point, area in self.points:
            zone = nearest_zone(point, inflate=self.args.key_inflate)
            if zone is not None:
                candidates.append((zone, area, point))

        if not candidates:
            return ""

        zone = choose_zone(candidates)
        if zone.kind == "key":
            self.output.emit_key(zone.name)
            return f"{zone.name}"

        direction = classify_direction(zone, self.points, self.args.direction_mode, self.args.deadzone)
        text = FLICK_MAP[zone.name].get(direction, FLICK_MAP[zone.name]["center"])
        self.output.emit_text(text)
        return f"{zone.name}:{direction}->{text}"


def choose_zone(candidates: list[tuple[Zone, float, tuple[float, float]]]) -> Zone:
    score: dict[str, float] = collections.defaultdict(float)
    zones_by_name = {zone.name: zone for zone, _, _ in candidates}
    for zone, area, _ in candidates:
        score[zone.name] += area
    return zones_by_name[max(score, key=score.get)]


def nearest_zone(point: tuple[float, float], inflate: float) -> Optional[Zone]:
    hits = [zone for zone in ZONES if zone.contains(point, inflate=inflate)]
    if not hits:
        return None
    px, py = point
    return min(hits, key=lambda zone: (zone.center[0] - px) ** 2 + (zone.center[1] - py) ** 2)


def classify_direction(
    zone: Zone,
    points: list[tuple[float, tuple[float, float], float]],
    mode: str,
    deadzone: float,
) -> str:
    zone_points = [point for _, point, _ in points if zone.contains(point, inflate=0.8)]
    if not zone_points:
        zone_points = [point for _, point, _ in points]

    if mode == "motion" and len(zone_points) >= 2:
        first = zone_points[0]
        last = zone_points[-1]
        dx = (last[0] - first[0]) / max(zone.width, 1.0)
        dy = (last[1] - first[1]) / max(zone.height, 1.0)
    else:
        cx, cy = zone.center
        first = zone_points[0]
        dx = (first[0] - cx) / max(zone.width * 0.5, 1.0)
        dy = (first[1] - cy) / max(zone.height * 0.5, 1.0)

    if abs(dx) < deadzone and abs(dy) < deadzone:
        return "center"
    if abs(dx) >= abs(dy):
        return "left" if dx < 0 else "right"
    return "up" if dy < 0 else "down"


def draw_overlay(warped: np.ndarray, centroid: Optional[tuple[float, float]], label: str) -> np.ndarray:
    canvas = warped.copy()
    for zone in KANA_ZONES:
        cv2.rectangle(canvas, (zone.x1, zone.y1), (zone.x2, zone.y2), (30, 210, 30), 3)
        cv2.circle(canvas, tuple(map(int, zone.center)), 5, (30, 210, 30), -1)
    for zone in SPECIAL_ZONES:
        cv2.rectangle(canvas, (zone.x1, zone.y1), (zone.x2, zone.y2), (255, 120, 30), 3)
    if centroid is not None:
        cv2.circle(canvas, tuple(map(int, centroid)), 20, (0, 0, 255), 4)
    if label:
        cv2.putText(canvas, label, (24, 52), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3, cv2.LINE_AA)
    return canvas


def collect_calibration(source, path: str) -> Calibration:
    clicked: list[tuple[float, float]] = []
    window_name = "M5Deflick calibration"
    labels = ["top-left", "top-right", "bottom-right", "bottom-left"]

    def on_mouse(event, x, y, _flags, _param) -> None:
        if event == cv2.EVENT_LBUTTONDOWN and len(clicked) < 4:
            clicked.append((float(x), float(y)))

    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(window_name, on_mouse)

    while len(clicked) < 4:
        ok, frame = source.read()
        if not ok or frame is None:
            raise RuntimeError("Could not read a frame while calibrating.")
        display = frame.copy()
        for index, point in enumerate(clicked):
            cv2.circle(display, tuple(map(int, point)), 8, (0, 0, 255), -1)
            cv2.putText(display, str(index + 1), tuple(map(int, point)), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
        prompt = f"Click board corner: {labels[len(clicked)]} ({len(clicked) + 1}/4)"
        cv2.putText(display, prompt, (24, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2, cv2.LINE_AA)
        cv2.imshow(window_name, display)
        key = cv2.waitKey(20) & 0xFF
        if key in (27, ord("q")):
            raise KeyboardInterrupt
        if key == ord("u") and clicked:
            clicked.pop()

    cv2.destroyWindow(window_name)
    calibration = Calibration(clicked)
    calibration.save(path)
    return calibration


def discover_unitv2_url(host: str) -> Optional[str]:
    if requests is None:
        raise RuntimeError("requests is required to auto-discover UnitV2 streams. Run: pip install -r requirements.txt")

    bases = normalize_unitv2_bases(host)
    candidates: list[str] = []
    for base in bases:
        candidates.extend(discover_urls_from_html(base))
        for path in (
            "/video",
            "/video_feed",
            "/stream",
            "/mjpeg",
            "/mjpg",
            "/cam.mjpg",
            "/stream.mjpg",
            "/capture",
            "/?action=stream",
        ):
            candidates.append(urljoin(base + "/", path.lstrip("/")))

    seen: set[str] = set()
    for candidate in candidates:
        if candidate in seen:
            continue
        seen.add(candidate)
        if looks_like_jpeg_stream(candidate):
            return candidate
    return None


def normalize_unitv2_bases(host: str) -> list[str]:
    host = host.strip()
    if not host:
        host = "10.254.239.1"
    if "://" not in host:
        host = "http://" + host
    parsed = urlparse(host)
    hostname = parsed.hostname or "10.254.239.1"
    scheme = parsed.scheme or "http"
    ports = [parsed.port] if parsed.port else [80, 8000, 8080]
    bases = []
    for port in ports:
        netloc = hostname if port == 80 else f"{hostname}:{port}"
        bases.append(f"{scheme}://{netloc}")
    return bases


def discover_urls_from_html(base: str) -> list[str]:
    urls: list[str] = []
    try:
        response = requests.get(base, timeout=(2, 3))
        response.raise_for_status()
    except Exception:
        return urls

    attrs = re.findall(r"""(?:src|href)\s*=\s*["']([^"']+)["']""", response.text, flags=re.IGNORECASE)
    keywords = ("stream", "mjpeg", "mjpg", "video", "camera", "capture", "jpg", "jpeg")
    for attr in attrs:
        if any(keyword in attr.lower() for keyword in keywords):
            urls.append(urljoin(base + "/", attr))
    return urls


def looks_like_jpeg_stream(url: str) -> bool:
    try:
        response = requests.get(url, stream=True, timeout=(2, 3))
        content_type = response.headers.get("content-type", "").lower()
        chunk = next(response.iter_content(chunk_size=4096), b"")
        response.close()
    except Exception:
        return False
    return (
        response.status_code < 400
        and ("multipart" in content_type or "jpeg" in content_type or b"\xff\xd8" in chunk)
    )


def open_source(args: argparse.Namespace):
    if args.source_url:
        return MjpegSource(args.source_url) if args.source_url.startswith("http") else Cv2Source(args.source_url)

    if args.source == "unitv2":
        discovered = discover_unitv2_url(args.unitv2_host)
        if not discovered:
            raise RuntimeError(
                "Could not auto-discover the UnitV2 stream URL. "
                "Open http://10.254.239.1 in a browser, find the stream URL, then pass --source-url."
            )
        print(f"Using UnitV2 stream: {discovered}", flush=True)
        return MjpegSource(discovered)

    if args.source.isdigit():
        return Cv2Source(int(args.source))
    if args.source.startswith("http"):
        return MjpegSource(args.source)
    return Cv2Source(args.source)


def run(args: argparse.Namespace) -> int:
    source = open_source(args)
    output = Output(args.output)
    try:
        calibration = None if args.recalibrate else Calibration.load(args.calibration)
        if calibration is None:
            calibration = collect_calibration(source, args.calibration)

        detector = MotionDetector(args.min_motion_area, args.bg_alpha)
        tracker = EventTracker(args, output)

        cv2.namedWindow("M5Deflick board", cv2.WINDOW_NORMAL)
        cv2.namedWindow("M5Deflick camera", cv2.WINDOW_NORMAL)

        while True:
            ok, frame = source.read()
            if not ok or frame is None:
                print("No frame. Retrying...", file=sys.stderr)
                time.sleep(0.2)
                continue

            matrix = calibration.matrix
            warped = cv2.warpPerspective(frame, matrix, (BOARD_SIZE, BOARD_SIZE))
            allow_bg_update = not tracker.active
            centroid, _area, mask = detector.detect(warped, allow_bg_update)
            label = tracker.update(centroid, _area, time.monotonic())

            board_display = draw_overlay(warped, centroid, label)
            camera_display = draw_camera_overlay(frame, calibration.points)
            mask_display = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
            combined = np.hstack([cv2.resize(board_display, (512, 512)), cv2.resize(mask_display, (512, 512))])

            cv2.imshow("M5Deflick board", combined)
            cv2.imshow("M5Deflick camera", camera_display)

            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q")):
                break
            if key == ord("r"):
                calibration = collect_calibration(source, args.calibration)
                detector.reset()
            if key == ord("b"):
                detector.reset()
    finally:
        source.release()
        cv2.destroyAllWindows()
    return 0


def draw_camera_overlay(frame: np.ndarray, points: Iterable[tuple[float, float]]) -> np.ndarray:
    display = frame.copy()
    pts = np.array(list(points), dtype=np.int32)
    if len(pts) == 4:
        cv2.polylines(display, [pts], isClosed=True, color=(0, 0, 255), thickness=3)
        for point in pts:
            cv2.circle(display, tuple(point), 7, (0, 0, 255), -1)
    return display


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Real-world kana flick input with M5UnitV2/camera motion detection.")
    parser.add_argument("--source", default="unitv2", help="unitv2, camera index like 0, a video file, or an HTTP MJPEG URL.")
    parser.add_argument("--source-url", default="", help="Direct stream URL. Overrides --source.")
    parser.add_argument("--unitv2-host", default="10.254.239.1", help="UnitV2 host/IP used when --source unitv2.")
    parser.add_argument("--calibration", default=CALIBRATION_FILE, help="Calibration JSON path.")
    parser.add_argument("--recalibrate", action="store_true", help="Ignore saved calibration and click four corners again.")
    parser.add_argument("--output", choices=("print", "unicode", "none"), default="print", help="Output mode.")
    parser.add_argument("--direction-mode", choices=("side", "motion"), default="side", help="How flick direction is classified.")
    parser.add_argument("--deadzone", type=float, default=0.22, help="Center deadzone as a ratio of key size.")
    parser.add_argument("--key-inflate", type=float, default=0.45, help="How far outside a key still counts as that key.")
    parser.add_argument("--min-motion-area", type=int, default=1400, help="Minimum moving blob area in warped board pixels.")
    parser.add_argument("--cooldown", type=float, default=0.35, help="Seconds to ignore after an emitted input.")
    parser.add_argument("--settle", type=float, default=0.12, help="Seconds of no motion before classifying a punch.")
    parser.add_argument("--bg-alpha", type=float, default=0.015, help="Background adaptation speed.")
    return parser.parse_args(argv)


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    try:
        return run(args)
    except KeyboardInterrupt:
        return 130
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
