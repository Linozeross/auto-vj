from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass, field
from collections.abc import Callable, Sequence
from pathlib import Path

import cv2
import numpy as np
from dotenv import load_dotenv

from modules.artnet_renderer import StaticArtNetSender
from modules.mapping_detection import DetectionRegion, DetectionResult, LedDetection, analyze_led_frames, colorize_debug_image

load_dotenv()

WINDOW_TITLE = "AutoVJ Mapping Lab"
DEBUG_WINDOW_TITLE = "AutoVJ Mapping Lab Debug"
DEFAULT_CAMERA_INDEX = 0
CAMERA_PROBE_MAX_INDEX = 6
DEFAULT_CAMERA_WIDTH = 1920
DEFAULT_CAMERA_HEIGHT = 1080
CAP_PROP_WIDTH = 3
CAP_PROP_HEIGHT = 4
QUIT_KEY = "q"
ESC_KEY_CODE = 27
LIST_CAMERAS_FLAG = "--list"
CAMERA_FLAG = "--camera"
MAX_INDEX_FLAG = "--max-index"
CAMERA_LABEL_PREFIX = "Camera"
ARTNET_IP_ENV_VAR = "ARTNET_IP"
LED_COUNT_ENV_VAR = "LED_COUNT"
MAPPING_START_LED_ENV_VAR = "MAPPING_START_LED"
DEFAULT_ARTNET_IP = "192.168.178.102"
DEFAULT_LED_COUNT = 280
HUMAN_LED_INDEX_OFFSET = 1
DEFAULT_START_LED_NUMBER = 150
DEFAULT_START_LED_INDEX = DEFAULT_START_LED_NUMBER - HUMAN_LED_INDEX_OFFSET
AUTO_START_LED_INDEX = 0
NO_CAMERAS_FOUND_TEXT = "No cameras found."
AVAILABLE_CAMERAS_TEXT = "Available cameras:"
OPENING_CAMERA_TEXT = "Opening"
QUIT_HELP_TEXT = "Press 'q' or Esc to quit."
OPEN_FAILED_TEXT = "OpenCV could not open camera"
FRAME_READ_FAILED_TEXT = "OpenCV opened the camera but no frame arrived."
FREEZE_KEY = "f"
CLEAR_POINTS_KEY = "c"
NEXT_LED_KEY = "n"
PREVIOUS_LED_KEY = "p"
TRIGGER_LED_KEY = "t"
MEASURE_LED_KEY = "m"
AUTO_MEASURE_KEY = "a"
DEBUG_VIEW_KEY = "d"
EXPORT_KEY = "e"
CLEAR_REGION_KEY = "x"
LIVE_STATUS_TEXT = "LIVE"
FROZEN_STATUS_TEXT = "FROZEN"
AUTO_STATUS_TEXT = "AUTO"
POINTS_LABEL_PREFIX = "Points"
MOUSE_CLICK_EVENT = cv2.EVENT_LBUTTONDOWN
MOUSE_RIGHT_BUTTON_DOWN = cv2.EVENT_RBUTTONDOWN
MOUSE_RIGHT_BUTTON_UP = cv2.EVENT_RBUTTONUP
MOUSE_MOVE_EVENT = cv2.EVENT_MOUSEMOVE
STATUS_TEXT_X = 20
STATUS_TEXT_Y = 30
POINTS_TEXT_Y = 60
LED_TEXT_Y = 90
KEYS_TEXT_Y = 120
KEYS_TEXT_LINE_HEIGHT = 24
DEBUG_STATUS_TEXT_Y = 150
ROI_STATUS_TEXT_Y = 180
POINT_RADIUS_PX = 8
POINT_TEXT_OFFSET_X = 12
POINT_TEXT_OFFSET_Y = 12
TEXT_FONT_SCALE = 0.8
TEXT_THICKNESS = 2
POINT_TEXT_FONT_SCALE = 0.6
POINT_TEXT_THICKNESS = 2
STATUS_TEXT_COLOR = (0, 255, 0)
FROZEN_TEXT_COLOR = (0, 200, 255)
POINT_COLOR = (255, 255, 0)
LED_TEXT_COLOR = (255, 200, 0)
ACTIVE_LED_COLOR = (255, 255, 255)
CAPTURE_SETTLE_FRAMES = 3
FRAME_MEDIAN_STACK_SIZE = 3
MAPPING_POINT_LABEL_PREFIX = "LED"
ARTNET_DISABLED_TEXT = "ArtNet trigger disabled."
MEASURE_FAILED_TEXT = "Measurement failed."
MEASURE_NO_DETECTION_TEXT = "No detection found."
AUTO_RUNNING_TEXT = "Auto measure running..."
AUTO_FINISHED_TEXT = "Auto measure finished."
EXPORT_COMPLETE_TEXT = "Mapping exported."
EXPORT_FILENAME = "mapping_lab_export.json"
MAPPING_FORMAT_VERSION = "1.0"
DEBUG_PANEL_SCALE = 0.45
DEBUG_PANEL_BORDER_PX = 12
DEBUG_LABEL_X = 16
DEBUG_LABEL_Y = 28
DEBUG_LABEL_FONT_SCALE = 0.8
DEBUG_LABEL_THICKNESS = 2
DEBUG_LABEL_COLOR = (255, 255, 255)
ROI_COLOR = (0, 255, 255)
ROI_TEXT_COLOR = (0, 255, 255)
KEYS_TEXT_LINE_ONE = "Keys: t LED on/off | m measure | a auto-run | d debug"
KEYS_TEXT_LINE_TWO = "Keys: n/p next-prev | e export | x clear ROI | q quit"
CAMERA_BACKEND = getattr(cv2, "CAP_AVFOUNDATION", cv2.CAP_ANY) if sys.platform == "darwin" else cv2.CAP_ANY


@dataclass(slots=True)
class OverlayPoint:
    label: str
    x: int
    y: int
    led_index: int | None = None
    confidence: float | None = None


@dataclass(slots=True)
class MappingSession:
    frozen_frame: object | None = None
    overlay_points: list[OverlayPoint] = field(default_factory=list)
    current_led_index: int = DEFAULT_START_LED_INDEX
    led_preview_enabled: bool = False
    auto_measure_running: bool = False
    debug_view_enabled: bool = True
    last_debug_frame: np.ndarray | None = None
    frame_size: tuple[int, int] | None = None
    detection_region: DetectionRegion | None = None
    pending_region_start: tuple[int, int] | None = None
    pending_region_end: tuple[int, int] | None = None

    @property
    def is_frozen(self) -> bool:
        return self.frozen_frame is not None

    def toggle_freeze(self, frame: object) -> None:
        if self.is_frozen:
            self.frozen_frame = None
            return
        self.frozen_frame = frame.copy() if hasattr(frame, "copy") else frame

    def add_point(self, x: int, y: int) -> None:
        self.overlay_points.append(OverlayPoint(str(len(self.overlay_points) + 1), x, y))

    def set_led_point(self, led_index: int, x: int, y: int, confidence: float) -> None:
        label = f"{MAPPING_POINT_LABEL_PREFIX} {led_index + HUMAN_LED_INDEX_OFFSET}"
        self.overlay_points = [point for point in self.overlay_points if point.label != label]
        self.overlay_points.append(OverlayPoint(label, x, y, led_index=led_index, confidence=confidence))

    def clear_points(self) -> None:
        self.overlay_points.clear()

    def set_current_led(self, led_index: int, total_leds: int) -> None:
        self.current_led_index = max(0, min(total_leds - 1, led_index))

    def move_current_led(self, step: int, total_leds: int) -> None:
        self.set_current_led(self.current_led_index + step, total_leds)

    def set_detection_region(self, x0: int, y0: int, x1: int, y1: int) -> None:
        self.detection_region = DetectionRegion(
            x0=min(x0, x1),
            y0=min(y0, y1),
            x1=max(x0, x1),
            y1=max(y0, y1),
        )
        self.pending_region_start = None
        self.pending_region_end = None

    def clear_detection_region(self) -> None:
        self.detection_region = None
        self.pending_region_start = None
        self.pending_region_end = None


def probe_camera_indices(
    max_index: int = CAMERA_PROBE_MAX_INDEX,
    capture_factory: Callable[[int, int], object] | None = None,
) -> list[int]:
    """Return camera indices that OpenCV can open."""
    factory = capture_factory or cv2.VideoCapture
    indices: list[int] = []

    for index in range(max_index):
        capture = factory(index, CAMERA_BACKEND)
        try:
            if capture is not None and capture.isOpened():
                indices.append(index)
        finally:
            if capture is not None:
                capture.release()

    return indices


def choose_camera_index(requested_index: int | None, available_indices: Sequence[int]) -> int | None:
    """Pick the requested camera when available, otherwise use the first detected one."""
    if requested_index is not None and requested_index in available_indices:
        return requested_index
    if requested_index is not None:
        return None
    if available_indices:
        if DEFAULT_CAMERA_INDEX in available_indices:
            return DEFAULT_CAMERA_INDEX
        return int(available_indices[0])
    return None


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plain OpenCV camera preview for Mapping Lab.")
    parser.add_argument(CAMERA_FLAG, type=int, default=None, help="Camera index to open.")
    parser.add_argument(MAX_INDEX_FLAG, type=int, default=CAMERA_PROBE_MAX_INDEX, help="Highest camera index probe range.")
    parser.add_argument(LIST_CAMERAS_FLAG, action="store_true", help="List detected camera indices and exit.")
    return parser.parse_args(argv)


def _open_capture(camera_index: int) -> cv2.VideoCapture:
    capture = cv2.VideoCapture(camera_index, CAMERA_BACKEND)
    capture.set(CAP_PROP_WIDTH, DEFAULT_CAMERA_WIDTH)
    capture.set(CAP_PROP_HEIGHT, DEFAULT_CAMERA_HEIGHT)
    return capture


def _print_available_cameras(indices: Sequence[int]) -> None:
    if not indices:
        print(NO_CAMERAS_FOUND_TEXT)
        return
    print(AVAILABLE_CAMERAS_TEXT)
    for index in indices:
        print(f"- {CAMERA_LABEL_PREFIX} {index}")


def _status_color(is_frozen: bool) -> tuple[int, int, int]:
    return FROZEN_TEXT_COLOR if is_frozen else STATUS_TEXT_COLOR


def _draw_overlay(frame: object, session: MappingSession) -> object:
    output = frame.copy() if hasattr(frame, "copy") else frame
    if isinstance(output, np.ndarray) and session.frame_size is None:
        session.frame_size = (int(output.shape[1]), int(output.shape[0]))
    status_text = AUTO_STATUS_TEXT if session.auto_measure_running else (FROZEN_STATUS_TEXT if session.is_frozen else LIVE_STATUS_TEXT)
    point_count_text = f"{POINTS_LABEL_PREFIX}: {len(session.overlay_points)}"
    led_status = "ON" if session.led_preview_enabled else "OFF"
    current_led_text = f"{MAPPING_POINT_LABEL_PREFIX} {session.current_led_index + HUMAN_LED_INDEX_OFFSET}: {led_status}"
    debug_status_text = f"Debug: {'ON' if session.debug_view_enabled else 'OFF'}"
    roi_status_text = "ROI: ON" if session.detection_region is not None else "ROI: OFF"

    cv2.putText(
        output,
        status_text,
        (STATUS_TEXT_X, STATUS_TEXT_Y),
        cv2.FONT_HERSHEY_SIMPLEX,
        TEXT_FONT_SCALE,
        _status_color(session.is_frozen),
        TEXT_THICKNESS,
    )
    cv2.putText(
        output,
        point_count_text,
        (STATUS_TEXT_X, POINTS_TEXT_Y),
        cv2.FONT_HERSHEY_SIMPLEX,
        POINT_TEXT_FONT_SCALE,
        POINT_COLOR,
        POINT_TEXT_THICKNESS,
    )
    cv2.putText(
        output,
        current_led_text,
        (STATUS_TEXT_X, LED_TEXT_Y),
        cv2.FONT_HERSHEY_SIMPLEX,
        POINT_TEXT_FONT_SCALE,
        LED_TEXT_COLOR,
        POINT_TEXT_THICKNESS,
    )
    cv2.putText(
        output,
        KEYS_TEXT_LINE_ONE,
        (STATUS_TEXT_X, KEYS_TEXT_Y),
        cv2.FONT_HERSHEY_SIMPLEX,
        POINT_TEXT_FONT_SCALE,
        LED_TEXT_COLOR,
        POINT_TEXT_THICKNESS,
    )
    cv2.putText(
        output,
        KEYS_TEXT_LINE_TWO,
        (STATUS_TEXT_X, KEYS_TEXT_Y + KEYS_TEXT_LINE_HEIGHT),
        cv2.FONT_HERSHEY_SIMPLEX,
        POINT_TEXT_FONT_SCALE,
        LED_TEXT_COLOR,
        POINT_TEXT_THICKNESS,
    )
    cv2.putText(
        output,
        debug_status_text,
        (STATUS_TEXT_X, DEBUG_STATUS_TEXT_Y),
        cv2.FONT_HERSHEY_SIMPLEX,
        POINT_TEXT_FONT_SCALE,
        LED_TEXT_COLOR,
        POINT_TEXT_THICKNESS,
    )
    cv2.putText(
        output,
        roi_status_text,
        (STATUS_TEXT_X, ROI_STATUS_TEXT_Y),
        cv2.FONT_HERSHEY_SIMPLEX,
        POINT_TEXT_FONT_SCALE,
        ROI_TEXT_COLOR,
        POINT_TEXT_THICKNESS,
    )

    region = session.detection_region
    if region is not None:
        cv2.rectangle(output, (region.x0, region.y0), (region.x1, region.y1), ROI_COLOR, TEXT_THICKNESS)
    elif session.pending_region_start is not None and session.pending_region_end is not None:
        start_x, start_y = session.pending_region_start
        end_x, end_y = session.pending_region_end
        cv2.rectangle(output, (start_x, start_y), (end_x, end_y), ROI_COLOR, 1)

    for point in session.overlay_points:
        cv2.circle(output, (point.x, point.y), POINT_RADIUS_PX, POINT_COLOR, TEXT_THICKNESS)
        cv2.putText(
            output,
            point.label,
            (point.x + POINT_TEXT_OFFSET_X, point.y - POINT_TEXT_OFFSET_Y),
            cv2.FONT_HERSHEY_SIMPLEX,
            POINT_TEXT_FONT_SCALE,
            POINT_COLOR,
            POINT_TEXT_THICKNESS,
        )

    return output


def _handle_mouse_click(event: int, x: int, y: int, _flags: int, session: MappingSession) -> None:
    if event == MOUSE_CLICK_EVENT:
        session.add_point(x, y)
    elif event == MOUSE_RIGHT_BUTTON_DOWN:
        session.pending_region_start = (x, y)
        session.pending_region_end = (x, y)
    elif event == MOUSE_MOVE_EVENT and session.pending_region_start is not None:
        session.pending_region_end = (x, y)
    elif event == MOUSE_RIGHT_BUTTON_UP and session.pending_region_start is not None:
        start_x, start_y = session.pending_region_start
        session.set_detection_region(start_x, start_y, x, y)


def _read_settled_frame(capture: cv2.VideoCapture, frame_count: int = CAPTURE_SETTLE_FRAMES) -> object | None:
    captured_frames: list[np.ndarray] = []
    for _ in range(frame_count):
        ok, frame = capture.read()
        if ok and frame is not None:
            captured_frames.append(frame)
    if not captured_frames:
        return None
    stacked_frames = np.stack(captured_frames[-FRAME_MEDIAN_STACK_SIZE:], axis=0)
    return np.median(stacked_frames, axis=0).astype(np.uint8)


def _artnet_settings() -> tuple[str, int]:
    return (
        os.environ.get(ARTNET_IP_ENV_VAR, DEFAULT_ARTNET_IP),
        int(os.environ.get(LED_COUNT_ENV_VAR, str(DEFAULT_LED_COUNT))),
    )


def _mapping_start_led_index(total_leds: int) -> int:
    requested_start_number = int(os.environ.get(MAPPING_START_LED_ENV_VAR, str(DEFAULT_START_LED_NUMBER)))
    requested_index = requested_start_number - HUMAN_LED_INDEX_OFFSET
    return max(0, min(total_leds - 1, requested_index))


def _apply_led_preview(sender: StaticArtNetSender | None, session: MappingSession) -> None:
    if sender is None:
        return
    if session.led_preview_enabled:
        sender.show_led(session.current_led_index, ACTIVE_LED_COLOR)
    else:
        sender.blackout()


def _annotate_debug_panel(image: np.ndarray, label: str) -> np.ndarray:
    annotated = image.copy()
    cv2.putText(
        annotated,
        label,
        (DEBUG_LABEL_X, DEBUG_LABEL_Y),
        cv2.FONT_HERSHEY_SIMPLEX,
        DEBUG_LABEL_FONT_SCALE,
        DEBUG_LABEL_COLOR,
        DEBUG_LABEL_THICKNESS,
    )
    return annotated


def _resize_debug_panel(image: np.ndarray) -> np.ndarray:
    height = max(1, int(image.shape[0] * DEBUG_PANEL_SCALE))
    width = max(1, int(image.shape[1] * DEBUG_PANEL_SCALE))
    return cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)


def _build_debug_montage(
    baseline_frame: np.ndarray,
    active_frame: np.ndarray,
    detection_result: DetectionResult,
) -> np.ndarray:
    active_overlay = active_frame.copy()
    detection = detection_result.detection
    if detection is not None:
        cv2.circle(active_overlay, (detection.x, detection.y), POINT_RADIUS_PX, POINT_COLOR, TEXT_THICKNESS)
    panels = [
        _annotate_debug_panel(_resize_debug_panel(active_overlay), "Active"),
        _annotate_debug_panel(_resize_debug_panel(colorize_debug_image(detection_result.artifacts.difference)), "Difference"),
        _annotate_debug_panel(_resize_debug_panel(colorize_debug_image(detection_result.artifacts.mask)), "Mask"),
        _annotate_debug_panel(_resize_debug_panel(colorize_debug_image(detection_result.artifacts.hot_mask)), "Hot Core"),
    ]
    top_row = cv2.hconcat(panels[:2])
    bottom_row = cv2.hconcat(panels[2:])
    border = np.full((DEBUG_PANEL_BORDER_PX, top_row.shape[1], 3), 0, dtype=np.uint8)
    return cv2.vconcat([top_row, border, bottom_row])


def _show_debug_window(session: MappingSession) -> None:
    if session.debug_view_enabled and session.last_debug_frame is not None:
        cv2.imshow(DEBUG_WINDOW_TITLE, session.last_debug_frame)
    else:
        cv2.destroyWindow(DEBUG_WINDOW_TITLE)


def _mapping_export_payload(session: MappingSession, total_leds: int) -> dict[str, object]:
    width, height = session.frame_size or (0, 0)
    sorted_points = sorted(
        (point for point in session.overlay_points if point.led_index is not None),
        key=lambda point: point.led_index if point.led_index is not None else -1,
    )
    leds = []
    for point in sorted_points:
        leds.append(
            {
                "index": int(point.led_index),
                "number": int(point.led_index) + HUMAN_LED_INDEX_OFFSET,
                "x": int(point.x),
                "y": int(point.y),
                "u": float(point.x / max(width - 1, 1)) if width else 0.0,
                "v": float(point.y / max(height - 1, 1)) if height else 0.0,
                "confidence": float(point.confidence or 0.0),
            }
        )
    return {
        "format": "auto_vj_mapping",
        "version": MAPPING_FORMAT_VERSION,
        "frame": {"width": width, "height": height},
        "artnet": {"ip": _artnet_settings()[0], "led_count": total_leds},
        "leds": leds,
    }


def _export_mapping(session: MappingSession, total_leds: int) -> Path:
    export_path = Path(EXPORT_FILENAME)
    export_path.write_text(json.dumps(_mapping_export_payload(session, total_leds), indent=2))
    return export_path


def _measure_current_led(
    capture: cv2.VideoCapture,
    sender: StaticArtNetSender | None,
    session: MappingSession,
) -> LedDetection | None:
    if sender is None:
        print(ARTNET_DISABLED_TEXT)
        return None

    sender.blackout()
    baseline_frame = _read_settled_frame(capture)
    if baseline_frame is None:
        print(MEASURE_FAILED_TEXT)
        return None

    sender.show_led(session.current_led_index, ACTIVE_LED_COLOR)
    active_frame = _read_settled_frame(capture)
    if active_frame is None:
        sender.blackout()
        print(MEASURE_FAILED_TEXT)
        return None

    detection_result = analyze_led_frames(baseline_frame, active_frame, region=session.detection_region)
    session.last_debug_frame = _build_debug_montage(baseline_frame, active_frame, detection_result)
    _show_debug_window(session)
    detection = detection_result.detection
    if detection is None:
        session.led_preview_enabled = True
        _apply_led_preview(sender, session)
        print(MEASURE_NO_DETECTION_TEXT)
        return None

    session.set_led_point(session.current_led_index, detection.x, detection.y, detection.confidence)
    session.led_preview_enabled = True
    _apply_led_preview(sender, session)
    return detection


def _auto_measure_range(
    capture: cv2.VideoCapture,
    sender: StaticArtNetSender | None,
    session: MappingSession,
    total_leds: int,
) -> None:
    if sender is None:
        print(ARTNET_DISABLED_TEXT)
        return

    session.auto_measure_running = True
    start_led_index = AUTO_START_LED_INDEX
    session.set_current_led(start_led_index, total_leds)
    print(f"{AUTO_RUNNING_TEXT} Starting at LED {start_led_index + HUMAN_LED_INDEX_OFFSET}.")
    try:
        for led_index in range(start_led_index, total_leds):
            session.set_current_led(led_index, total_leds)
            detection = _measure_current_led(capture, sender, session)
            if detection is not None:
                print(f"{MAPPING_POINT_LABEL_PREFIX} {led_index + HUMAN_LED_INDEX_OFFSET} -> ({detection.x}, {detection.y})")
            frame = _read_settled_frame(capture)
            if frame is not None:
                cv2.imshow(WINDOW_TITLE, _draw_overlay(frame, session))
                _show_debug_window(session)
                cv2.waitKey(1)
    finally:
        session.auto_measure_running = False
        print(AUTO_FINISHED_TEXT)


def run_mapping_lab(argv: Sequence[str] | None = None) -> None:
    args = _parse_args(argv)
    available_indices = probe_camera_indices(max_index=args.max_index)

    if args.list:
        _print_available_cameras(available_indices)
        return

    camera_index = choose_camera_index(args.camera, available_indices)
    if camera_index is None:
        _print_available_cameras(available_indices)
        if args.camera is not None:
            print(f"{OPEN_FAILED_TEXT} {args.camera}.")
        return

    print(f"{OPENING_CAMERA_TEXT} {CAMERA_LABEL_PREFIX} {camera_index}. {QUIT_HELP_TEXT}")
    capture = _open_capture(camera_index)
    artnet_ip, total_leds = _artnet_settings()
    sender = StaticArtNetSender(artnet_ip, total_leds) if total_leds > 0 else None
    session = MappingSession()
    session.set_current_led(_mapping_start_led_index(total_leds), total_leds)

    if not capture.isOpened():
        print(f"{OPEN_FAILED_TEXT} {camera_index}.")
        capture.release()
        return

    cv2.namedWindow(WINDOW_TITLE)
    cv2.setMouseCallback(WINDOW_TITLE, _handle_mouse_click, session)

    try:
        while True:
            if session.is_frozen:
                frame = session.frozen_frame
            else:
                ok, frame = capture.read()
                if not ok or frame is None:
                    print(FRAME_READ_FAILED_TEXT)
                    break

            if sender is not None and session.led_preview_enabled:
                sender.show_led(session.current_led_index, ACTIVE_LED_COLOR)

            display_frame = _draw_overlay(frame, session)
            cv2.imshow(WINDOW_TITLE, display_frame)
            _show_debug_window(session)
            key_code = cv2.waitKey(1) & 0xFF
            if key_code == ord(QUIT_KEY) or key_code == ESC_KEY_CODE:
                break
            if key_code == ord(FREEZE_KEY) and frame is not None:
                session.toggle_freeze(frame)
            if key_code == ord(CLEAR_POINTS_KEY):
                session.clear_points()
            if key_code == ord(NEXT_LED_KEY):
                session.move_current_led(1, total_leds)
                _apply_led_preview(sender, session)
            if key_code == ord(PREVIOUS_LED_KEY):
                session.move_current_led(-1, total_leds)
                _apply_led_preview(sender, session)
            if key_code == ord(TRIGGER_LED_KEY):
                session.led_preview_enabled = not session.led_preview_enabled
                _apply_led_preview(sender, session)
            if key_code == ord(MEASURE_LED_KEY):
                detection = _measure_current_led(capture, sender, session)
                if detection is not None:
                    print(
                        f"{MAPPING_POINT_LABEL_PREFIX} {session.current_led_index + HUMAN_LED_INDEX_OFFSET}"
                        f" -> ({detection.x}, {detection.y})"
                    )
            if key_code == ord(AUTO_MEASURE_KEY):
                _auto_measure_range(capture, sender, session, total_leds)
            if key_code == ord(DEBUG_VIEW_KEY):
                session.debug_view_enabled = not session.debug_view_enabled
                _show_debug_window(session)
            if key_code == ord(EXPORT_KEY):
                export_path = _export_mapping(session, total_leds)
                print(f"{EXPORT_COMPLETE_TEXT} {export_path}")
            if key_code == ord(CLEAR_REGION_KEY):
                session.clear_detection_region()
    finally:
        if sender is not None:
            sender.blackout()
            sender.close()
        capture.release()
        cv2.destroyAllWindows()
