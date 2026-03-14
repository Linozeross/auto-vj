import math
import colorsys
from abc import ABC, abstractmethod
from typing import Any, Literal
from pydantic import BaseModel


class Effect(ABC):
    @abstractmethod
    def get_color(self, t: float, led_index: int, total_leds: int) -> list[int]:
        """Return [r, g, b] for a given led at time t (seconds since activation)."""
        ...

    def on_beat(self, bpm: float, beat_number: int) -> None:
        """Beat-sync hook — no-op by default."""
        pass


class SolidColor(Effect):
    def __init__(self, r: int = 255, g: int = 255, b: int = 255, **_):
        self.r = int(r)
        self.g = int(g)
        self.b = int(b)

    def get_color(self, t: float, led_index: int, total_leds: int) -> list[int]:
        return [self.r, self.g, self.b]


class ColorWave(Effect):
    def __init__(self, speed: float = 1.0, saturation: float = 1.0, brightness: float = 1.0, **_):
        self.speed = float(speed)
        self.saturation = float(saturation)
        self.brightness = float(brightness)

    def get_color(self, t: float, led_index: int, total_leds: int) -> list[int]:
        hue = (t * self.speed + led_index / total_leds) % 1.0
        r, g, b = colorsys.hsv_to_rgb(hue, self.saturation, self.brightness)
        return [int(r * 255), int(g * 255), int(b * 255)]


class Pulse(Effect):
    def __init__(self, r: int = 255, g: int = 255, b: int = 255, rate_hz: float = 1.0, **_):
        self.r = int(r)
        self.g = int(g)
        self.b = int(b)
        self.rate_hz = float(rate_hz)

    def get_color(self, t: float, led_index: int, total_leds: int) -> list[int]:
        brightness = (math.sin(2 * math.pi * self.rate_hz * t) + 1) / 2
        return [int(self.r * brightness), int(self.g * brightness), int(self.b * brightness)]


class Rainbow(Effect):
    def __init__(self, speed: float = 1.0, **_):
        self.speed = float(speed)

    def get_color(self, t: float, led_index: int, total_leds: int) -> list[int]:
        hue = (led_index / total_leds + t * self.speed) % 1.0
        r, g, b = colorsys.hsv_to_rgb(hue, 1.0, 1.0)
        return [int(r * 255), int(g * 255), int(b * 255)]


class Chase(Effect):
    def __init__(self, r: int = 255, g: int = 255, b: int = 255, speed: float = 1.0, tail: int = 5, **_):
        self.r = int(r)
        self.g = int(g)
        self.b = int(b)
        self.speed = float(speed)
        self.tail = int(tail)

    def get_color(self, t: float, led_index: int, total_leds: int) -> list[int]:
        head = (t * self.speed * total_leds) % total_leds
        distance = (head - led_index) % total_leds
        if distance < self.tail:
            fade = 1.0 - distance / self.tail
            return [int(self.r * fade), int(self.g * fade), int(self.b * fade)]
        return [0, 0, 0]


# GPT structured output schema
class EffectCommand(BaseModel):
    effect: Literal["solid", "color_wave", "pulse", "rainbow", "chase"]
    params: dict[str, Any]


EFFECT_REGISTRY: dict[str, type[Effect]] = {
    "solid": SolidColor,
    "color_wave": ColorWave,
    "pulse": Pulse,
    "rainbow": Rainbow,
    "chase": Chase,
}


def effect_from_dict(d: dict) -> Effect:
    effect_name = d["effect"]
    cls = EFFECT_REGISTRY[effect_name]
    return cls(**d.get("params", {}))
