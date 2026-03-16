from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

DEFAULT_BLUR_KERNEL_SIZE = 9
DEFAULT_THRESHOLD_VALUE = 56
DEFAULT_MIN_AREA_PX = 9
DEFAULT_MAX_VALUE = 255.0
CONTOUR_RETRIEVAL_MODE = cv2.RETR_EXTERNAL
CONTOUR_APPROXIMATION_MODE = cv2.CHAIN_APPROX_SIMPLE
COLOR_FRAME_DIMENSIONS = 3
SINGLE_CHANNEL_COUNT = 1
MORPH_KERNEL_SIZE = 3
WEIGHT_EPSILON = 1e-6
HOT_PIXEL_RATIO = 0.8
HEATMAP_COLORMAP = cv2.COLORMAP_TURBO
DEBUG_MAX_VALUE = 255


@dataclass(slots=True)
class DetectionConfig:
    blur_kernel_size: int = DEFAULT_BLUR_KERNEL_SIZE
    threshold_value: int = DEFAULT_THRESHOLD_VALUE
    min_area_px: int = DEFAULT_MIN_AREA_PX


@dataclass(slots=True)
class LedDetection:
    x: int
    y: int
    area_px: int
    peak_value: float
    confidence: float


@dataclass(slots=True)
class DetectionArtifacts:
    difference: np.ndarray
    mask: np.ndarray
    hot_mask: np.ndarray


@dataclass(slots=True)
class DetectionResult:
    detection: LedDetection | None
    artifacts: DetectionArtifacts


@dataclass(slots=True)
class DetectionRegion:
    x0: int
    y0: int
    x1: int
    y1: int


def compute_difference_mask(
    reference_frame: np.ndarray,
    active_frame: np.ndarray,
    config: DetectionConfig | None = None,
) -> np.ndarray:
    """Return a thresholded grayscale mask of bright changes."""
    current_config = config or DetectionConfig()
    reference_gray = _to_grayscale(reference_frame)
    active_gray = _to_grayscale(active_frame)
    frame_delta = cv2.absdiff(active_gray, reference_gray)
    blurred_delta = cv2.GaussianBlur(
        frame_delta,
        (current_config.blur_kernel_size, current_config.blur_kernel_size),
        0,
    )
    _threshold, mask = cv2.threshold(
        blurred_delta,
        current_config.threshold_value,
        int(DEFAULT_MAX_VALUE),
        cv2.THRESH_BINARY,
    )
    morph_kernel = np.ones((MORPH_KERNEL_SIZE, MORPH_KERNEL_SIZE), dtype=np.uint8)
    opened_mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, morph_kernel)
    return opened_mask


def detect_led_from_frames(
    reference_frame: np.ndarray,
    active_frame: np.ndarray,
    config: DetectionConfig | None = None,
    region: DetectionRegion | None = None,
) -> LedDetection | None:
    return analyze_led_frames(reference_frame, active_frame, config, region).detection


def analyze_led_frames(
    reference_frame: np.ndarray,
    active_frame: np.ndarray,
    config: DetectionConfig | None = None,
    region: DetectionRegion | None = None,
) -> DetectionResult:
    """Detect the brightest changed blob between a baseline and active frame."""
    current_config = config or DetectionConfig()
    difference = active_frame_delta(reference_frame, active_frame)
    mask = compute_difference_mask(reference_frame, active_frame, current_config)
    if region is not None:
        region_mask = np.zeros_like(mask)
        region_mask[region.y0:region.y1, region.x0:region.x1] = int(DEFAULT_MAX_VALUE)
        difference = cv2.bitwise_and(difference, region_mask)
        mask = cv2.bitwise_and(mask, region_mask)
    best_hot_mask = np.zeros_like(mask)
    contours, _hierarchy = cv2.findContours(
        mask,
        CONTOUR_RETRIEVAL_MODE,
        CONTOUR_APPROXIMATION_MODE,
    )
    best_detection: LedDetection | None = None

    for contour in contours:
        area_px = int(cv2.contourArea(contour))
        if area_px < current_config.min_area_px:
            continue

        contour_mask = np.zeros_like(mask)
        cv2.drawContours(contour_mask, [contour], -1, int(DEFAULT_MAX_VALUE), thickness=cv2.FILLED)
        peak_value = float(cv2.minMaxLoc(difference, mask=contour_mask)[1])
        hot_pixel_threshold = peak_value * HOT_PIXEL_RATIO
        hot_region_mask = np.where(
            (contour_mask > 0) & (difference >= hot_pixel_threshold),
            int(DEFAULT_MAX_VALUE),
            0,
        ).astype(np.uint8)
        active_mask = hot_region_mask if np.any(hot_region_mask) else contour_mask
        weighted_pixels = difference.astype(np.float32) * (active_mask.astype(np.float32) / DEFAULT_MAX_VALUE)
        total_weight = float(weighted_pixels.sum())
        if total_weight <= WEIGHT_EPSILON:
            continue

        y_coords, x_coords = np.indices(weighted_pixels.shape)
        x = int(round(float((weighted_pixels * x_coords).sum()) / total_weight))
        y = int(round(float((weighted_pixels * y_coords).sum()) / total_weight))
        confidence = min(1.0, peak_value / DEFAULT_MAX_VALUE)
        candidate = LedDetection(
            x=x,
            y=y,
            area_px=area_px,
            peak_value=peak_value,
            confidence=confidence,
        )
        if best_detection is None or candidate.peak_value > best_detection.peak_value:
            best_detection = candidate
            best_hot_mask = active_mask

    return DetectionResult(
        detection=best_detection,
        artifacts=DetectionArtifacts(
            difference=difference,
            mask=mask,
            hot_mask=best_hot_mask,
        ),
    )


def active_frame_delta(reference_frame: np.ndarray, active_frame: np.ndarray) -> np.ndarray:
    """Return the grayscale absolute difference without thresholding."""
    reference_gray = _to_grayscale(reference_frame)
    active_gray = _to_grayscale(active_frame)
    return cv2.absdiff(active_gray, reference_gray)


def _to_grayscale(frame: np.ndarray) -> np.ndarray:
    if frame.ndim == COLOR_FRAME_DIMENSIONS and frame.shape[2] != SINGLE_CHANNEL_COUNT:
        return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    if frame.ndim == COLOR_FRAME_DIMENSIONS and frame.shape[2] == SINGLE_CHANNEL_COUNT:
        return frame[:, :, 0].copy()
    return frame.copy()


def colorize_debug_image(image: np.ndarray) -> np.ndarray:
    if image.ndim == COLOR_FRAME_DIMENSIONS:
        return image.copy()
    normalized = cv2.normalize(image, None, 0, DEBUG_MAX_VALUE, cv2.NORM_MINMAX).astype(np.uint8)
    return cv2.applyColorMap(normalized, HEATMAP_COLORMAP)
