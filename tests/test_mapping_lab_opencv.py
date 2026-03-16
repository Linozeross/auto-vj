from __future__ import annotations

import numpy as np

from modules.mapping_lab import (
    AUTO_START_LED_INDEX,
    DEFAULT_START_LED_INDEX,
    FRAME_MEDIAN_STACK_SIZE,
    MOUSE_MOVE_EVENT,
    MOUSE_CLICK_EVENT,
    MOUSE_RIGHT_BUTTON_DOWN,
    MOUSE_RIGHT_BUTTON_UP,
    MappingSession,
    _mapping_export_payload,
    _artnet_settings,
    _mapping_start_led_index,
    _read_settled_frame,
    _handle_mouse_click,
    choose_camera_index,
    probe_camera_indices,
)


class _FakeCapture:
    def __init__(self, opened: bool) -> None:
        self._opened = opened

    def isOpened(self) -> bool:
        return self._opened

    def release(self) -> None:
        return None


class _FakeFrameCapture:
    def __init__(self, frames: list[np.ndarray]) -> None:
        self._frames = frames
        self._index = 0

    def read(self) -> tuple[bool, np.ndarray | None]:
        if self._index >= len(self._frames):
            return False, None
        frame = self._frames[self._index]
        self._index += 1
        return True, frame


def test_probe_camera_indices_returns_only_openable_indices() -> None:
    open_indices = {0, 2, 5}

    def factory(index: int, _backend: int) -> _FakeCapture:
        return _FakeCapture(index in open_indices)

    assert probe_camera_indices(max_index=6, capture_factory=factory) == [0, 2, 5]


def test_choose_camera_index_prefers_default_or_first_available() -> None:
    assert choose_camera_index(None, [1, 3]) == 1
    assert choose_camera_index(3, [1, 3]) == 3
    assert choose_camera_index(9, [1, 3]) is None


def test_mapping_session_freeze_toggle_and_overlay_points() -> None:
    frame = [[1, 2], [3, 4]]
    session = MappingSession()

    assert session.current_led_index == DEFAULT_START_LED_INDEX

    session.toggle_freeze(frame)
    _handle_mouse_click(MOUSE_CLICK_EVENT, 10, 20, 0, session)
    _handle_mouse_click(MOUSE_CLICK_EVENT, 30, 40, 0, session)

    assert session.is_frozen is True
    assert [(point.label, point.x, point.y) for point in session.overlay_points] == [
        ("1", 10, 20),
        ("2", 30, 40),
    ]

    session.clear_points()
    session.toggle_freeze(frame)

    assert session.is_frozen is False and session.overlay_points == []


def test_mapping_start_led_index_clamps_to_total_led_count() -> None:
    assert _mapping_start_led_index(280) == DEFAULT_START_LED_INDEX
    assert _mapping_start_led_index(120) == 119


def test_mapping_session_defaults_to_manual_mode() -> None:
    session = MappingSession()

    assert session.auto_measure_running is False and session.led_preview_enabled is False


def test_artnet_settings_use_default_led_count() -> None:
    artnet_ip, led_count = _artnet_settings()

    assert artnet_ip and led_count == 280


def test_auto_mode_starts_from_first_led() -> None:
    assert AUTO_START_LED_INDEX == 0


def test_read_settled_frame_uses_median_of_recent_frames() -> None:
    frames = [
        np.full((2, 2, 3), 10, dtype=np.uint8),
        np.full((2, 2, 3), 20, dtype=np.uint8),
        np.full((2, 2, 3), 30, dtype=np.uint8),
        np.full((2, 2, 3), 200, dtype=np.uint8),
    ]
    capture = _FakeFrameCapture(frames)

    frame = _read_settled_frame(capture, frame_count=FRAME_MEDIAN_STACK_SIZE)

    assert frame is not None and int(frame[0, 0, 0]) == 20


def test_mapping_export_payload_contains_normalized_led_positions() -> None:
    session = MappingSession()
    session.frame_size = (200, 100)
    session.set_led_point(4, 50, 25, 0.8)

    payload = _mapping_export_payload(session, 280)

    assert payload["format"] == "auto_vj_mapping"
    assert payload["frame"] == {"width": 200, "height": 100}
    assert payload["leds"][0]["index"] == 4
    assert payload["leds"][0]["number"] == 5
    assert payload["leds"][0]["u"] > 0.0
    assert payload["leds"][0]["v"] > 0.0


def test_right_mouse_drag_sets_detection_region() -> None:
    session = MappingSession()

    _handle_mouse_click(MOUSE_RIGHT_BUTTON_DOWN, 90, 70, 0, session)
    _handle_mouse_click(MOUSE_MOVE_EVENT, 20, 10, 0, session)
    _handle_mouse_click(MOUSE_RIGHT_BUTTON_UP, 20, 10, 0, session)

    assert session.detection_region is not None
    assert (
        session.detection_region.x0,
        session.detection_region.y0,
        session.detection_region.x1,
        session.detection_region.y1,
    ) == (20, 10, 90, 70)
