from __future__ import annotations

import numpy as np

from modules.mapping_detection import (
    DetectionConfig,
    DetectionRegion,
    analyze_led_frames,
    compute_difference_mask,
    detect_led_from_frames,
)

FRAME_HEIGHT = 80
FRAME_WIDTH = 100
SPOT_RADIUS_PX = 4
BRIGHT_SPOT_VALUE = 255
DARK_VALUE = 0
EXPECTED_X = 40
EXPECTED_Y = 30
THRESHOLD_VALUE = 10


def _blank_frame() -> np.ndarray:
    return np.zeros((FRAME_HEIGHT, FRAME_WIDTH, 3), dtype=np.uint8)


def test_compute_difference_mask_highlights_new_bright_spot() -> None:
    reference_frame = _blank_frame()
    active_frame = _blank_frame()
    yy, xx = np.ogrid[:FRAME_HEIGHT, :FRAME_WIDTH]
    mask = (xx - EXPECTED_X) ** 2 + (yy - EXPECTED_Y) ** 2 <= SPOT_RADIUS_PX ** 2
    active_frame[mask] = BRIGHT_SPOT_VALUE

    difference_mask = compute_difference_mask(
        reference_frame,
        active_frame,
        DetectionConfig(threshold_value=THRESHOLD_VALUE),
    )

    assert int(difference_mask[EXPECTED_Y, EXPECTED_X]) == BRIGHT_SPOT_VALUE


def test_detect_led_from_frames_returns_centroid_of_brightest_blob() -> None:
    reference_frame = _blank_frame()
    active_frame = _blank_frame()
    yy, xx = np.ogrid[:FRAME_HEIGHT, :FRAME_WIDTH]
    active_mask = (xx - EXPECTED_X) ** 2 + (yy - EXPECTED_Y) ** 2 <= SPOT_RADIUS_PX ** 2
    active_frame[active_mask] = BRIGHT_SPOT_VALUE

    detection = detect_led_from_frames(
        reference_frame,
        active_frame,
        DetectionConfig(threshold_value=THRESHOLD_VALUE),
    )

    assert detection is not None and abs(detection.x - EXPECTED_X) <= 1 and abs(detection.y - EXPECTED_Y) <= 1


def test_detect_led_from_frames_returns_none_when_no_change_exists() -> None:
    reference_frame = _blank_frame()
    active_frame = np.full((FRAME_HEIGHT, FRAME_WIDTH, 3), DARK_VALUE, dtype=np.uint8)

    detection = detect_led_from_frames(reference_frame, active_frame)

    assert detection is None


def test_detect_led_from_frames_prefers_weighted_hot_center_over_bleed_tail() -> None:
    reference_frame = _blank_frame()
    active_frame = _blank_frame()
    active_frame[EXPECTED_Y, EXPECTED_X] = BRIGHT_SPOT_VALUE
    active_frame[EXPECTED_Y, EXPECTED_X + 1] = 200
    active_frame[EXPECTED_Y, EXPECTED_X + 2] = 120
    active_frame[EXPECTED_Y, EXPECTED_X + 3] = 80

    detection = detect_led_from_frames(
        reference_frame,
        active_frame,
        DetectionConfig(blur_kernel_size=3, threshold_value=10, min_area_px=1),
    )

    assert detection is not None and abs(detection.x - EXPECTED_X) <= 1


def test_analyze_led_frames_returns_debug_artifacts() -> None:
    reference_frame = _blank_frame()
    active_frame = _blank_frame()
    active_frame[EXPECTED_Y, EXPECTED_X] = BRIGHT_SPOT_VALUE

    result = analyze_led_frames(
        reference_frame,
        active_frame,
        DetectionConfig(blur_kernel_size=3, threshold_value=10, min_area_px=1),
    )

    assert result.detection is not None
    assert result.artifacts.difference.shape == (FRAME_HEIGHT, FRAME_WIDTH)
    assert result.artifacts.mask.shape == (FRAME_HEIGHT, FRAME_WIDTH)
    assert result.artifacts.hot_mask.shape == (FRAME_HEIGHT, FRAME_WIDTH)


def test_detect_led_from_frames_respects_detection_region() -> None:
    reference_frame = _blank_frame()
    active_frame = _blank_frame()
    active_frame[EXPECTED_Y, EXPECTED_X] = BRIGHT_SPOT_VALUE
    allowed_region = DetectionRegion(x0=0, y0=0, x1=20, y1=20)

    detection = detect_led_from_frames(
        reference_frame,
        active_frame,
        DetectionConfig(blur_kernel_size=3, threshold_value=10, min_area_px=1),
        region=allowed_region,
    )

    assert detection is None
