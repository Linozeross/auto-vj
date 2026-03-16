from __future__ import annotations

import math
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass

import cv2
import numpy as np

from modules.mapping_loader import MappingData, MappingLed

CANVAS_WIDTH = 640
CANVAS_HEIGHT = 360
DEFAULT_RENDER_FPS = 30
DEFAULT_BRIGHTNESS = 1.0
DEFAULT_WAVE_SPEED = 0.35
DEFAULT_WAVE_FREQUENCY = 1.2
DEFAULT_SPIRAL_SPEED = 0.22
DEFAULT_SPIRAL_ARMS = 5.0
DEFAULT_SPIRAL_RING_SCALE = 18.0
DEFAULT_SPIRAL_TWIST = 7.5
DEFAULT_BEAM_SPEED = 0.42
DEFAULT_BEAM_SLOPE = 0.85
DEFAULT_BEAM_WIDTH = 0.08
DEFAULT_BEAM_REPEAT = 4.5
DEFAULT_BEAM_GLOW = 0.18
DEFAULT_PREVIEW_LED_RADIUS = 5
DEFAULT_PREVIEW_BG = (8, 10, 16)
DEFAULT_PREVIEW_TEXT = (240, 240, 240)
PREVIEW_LABEL_X = 16
PREVIEW_LABEL_Y = 28
PREVIEW_FONT_SCALE = 0.8
PREVIEW_TEXT_THICKNESS = 2
COLOR_SCALE_MAX = 255


@dataclass(slots=True)
class Canvas2D:
    width: int = CANVAS_WIDTH
    height: int = CANVAS_HEIGHT


class Effect2D(ABC):
    @abstractmethod
    def color_at(self, u: float, v: float, t: float) -> list[int]:
        ...


class HorizontalWave2D(Effect2D):
    def __init__(
        self,
        speed: float = DEFAULT_WAVE_SPEED,
        frequency: float = DEFAULT_WAVE_FREQUENCY,
        brightness: float = DEFAULT_BRIGHTNESS,
    ) -> None:
        self._speed = speed
        self._frequency = frequency
        self._brightness = brightness

    def color_at(self, u: float, v: float, t: float) -> list[int]:
        phase = u * self._frequency - t * self._speed
        wave = (math.sin(2 * math.pi * phase) + 1.0) / 2.0
        return _gradient_color(wave, self._brightness)


class VerticalWave2D(Effect2D):
    def __init__(
        self,
        speed: float = DEFAULT_WAVE_SPEED,
        frequency: float = DEFAULT_WAVE_FREQUENCY,
        brightness: float = DEFAULT_BRIGHTNESS,
    ) -> None:
        self._speed = speed
        self._frequency = frequency
        self._brightness = brightness

    def color_at(self, u: float, v: float, t: float) -> list[int]:
        phase = v * self._frequency - t * self._speed
        wave = (math.sin(2 * math.pi * phase) + 1.0) / 2.0
        return _gradient_color(wave, self._brightness)


class SpiralRings2D(Effect2D):
    def __init__(
        self,
        speed: float = DEFAULT_SPIRAL_SPEED,
        arms: float = DEFAULT_SPIRAL_ARMS,
        ring_scale: float = DEFAULT_SPIRAL_RING_SCALE,
        twist: float = DEFAULT_SPIRAL_TWIST,
        brightness: float = DEFAULT_BRIGHTNESS,
    ) -> None:
        self._speed = speed
        self._arms = arms
        self._ring_scale = ring_scale
        self._twist = twist
        self._brightness = brightness

    def color_at(self, u: float, v: float, t: float) -> list[int]:
        x = (u - 0.5) * 2.0
        y = (v - 0.5) * 2.0
        radius = math.sqrt(x * x + y * y)
        angle = math.atan2(y, x)
        spiral_phase = (
            angle * self._arms
            + radius * self._twist
            - t * self._speed * 2.0 * math.pi
        )
        rings = 0.5 + 0.5 * math.sin(spiral_phase + radius * self._ring_scale)
        flare = max(0.0, 1.0 - radius)
        energy = min(1.0, rings * 0.8 + flare * 0.5)
        return _spectral_color(energy, rings, self._brightness)


class DiagonalBeams2D(Effect2D):
    def __init__(
        self,
        speed: float = DEFAULT_BEAM_SPEED,
        slope: float = DEFAULT_BEAM_SLOPE,
        width: float = DEFAULT_BEAM_WIDTH,
        repeat: float = DEFAULT_BEAM_REPEAT,
        glow: float = DEFAULT_BEAM_GLOW,
        brightness: float = DEFAULT_BRIGHTNESS,
    ) -> None:
        self._speed = speed
        self._slope = slope
        self._width = width
        self._repeat = repeat
        self._glow = glow
        self._brightness = brightness

    def color_at(self, u: float, v: float, t: float) -> list[int]:
        diagonal_position = (u * self._slope - v) * self._repeat - t * self._speed
        wrapped_position = diagonal_position - math.floor(diagonal_position)
        distance_to_beam = min(wrapped_position, 1.0 - wrapped_position)
        core = math.exp(-((distance_to_beam / max(self._width, 1e-6)) ** 2))
        shimmer = 0.5 + 0.5 * math.sin(2.0 * math.pi * (u * 1.7 + v * 0.9 - t * 0.35))
        energy = min(1.0, core + shimmer * self._glow)
        return _beam_color(energy, shimmer, self._brightness)


def render_led_frame(effect: Effect2D, mapping: MappingData, t: float) -> list[int]:
    rgb_values = [0] * mapping.artnet.led_count * 3
    for led in mapping.leds:
        color = effect.color_at(led.u, led.v, t)
        start = led.index * 3
        rgb_values[start:start + 3] = color
    return rgb_values


def render_preview(effect: Effect2D, mapping: MappingData, t: float, label: str) -> np.ndarray:
    canvas = np.full((CANVAS_HEIGHT, CANVAS_WIDTH, 3), DEFAULT_PREVIEW_BG, dtype=np.uint8)
    for led in mapping.leds:
        color = effect.color_at(led.u, led.v, t)
        x = int(round(led.u * max(CANVAS_WIDTH - 1, 1)))
        y = int(round(led.v * max(CANVAS_HEIGHT - 1, 1)))
        cv2.circle(canvas, (x, y), DEFAULT_PREVIEW_LED_RADIUS, color, thickness=cv2.FILLED)
    cv2.putText(
        canvas,
        label,
        (PREVIEW_LABEL_X, PREVIEW_LABEL_Y),
        cv2.FONT_HERSHEY_SIMPLEX,
        PREVIEW_FONT_SCALE,
        DEFAULT_PREVIEW_TEXT,
        PREVIEW_TEXT_THICKNESS,
    )
    return canvas


def _gradient_color(value: float, brightness: float) -> list[int]:
    clamped_value = max(0.0, min(1.0, value))
    scaled = clamped_value * brightness
    return [
        int(COLOR_SCALE_MAX * scaled),
        int(COLOR_SCALE_MAX * (1.0 - abs(clamped_value - 0.5) * 2.0) * brightness),
        int(COLOR_SCALE_MAX * (1.0 - scaled)),
    ]


def _spectral_color(energy: float, accent: float, brightness: float) -> list[int]:
    clamped_energy = max(0.0, min(1.0, energy))
    clamped_accent = max(0.0, min(1.0, accent))
    return [
        int(COLOR_SCALE_MAX * clamped_energy * brightness),
        int(COLOR_SCALE_MAX * (0.25 + clamped_accent * 0.75) * brightness),
        int(COLOR_SCALE_MAX * (1.0 - clamped_energy * 0.65) * brightness),
    ]


def _beam_color(energy: float, shimmer: float, brightness: float) -> list[int]:
    clamped_energy = max(0.0, min(1.0, energy))
    clamped_shimmer = max(0.0, min(1.0, shimmer))
    return [
        int(COLOR_SCALE_MAX * (0.15 + clamped_energy * 0.85) * brightness),
        int(COLOR_SCALE_MAX * (clamped_energy * 0.45 + clamped_shimmer * 0.35) * brightness),
        int(COLOR_SCALE_MAX * (clamped_energy * 0.95) * brightness),
    ]


def monotonic_seconds() -> float:
    return time.monotonic()
