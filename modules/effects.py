import math
import random
import time
import colorsys
from abc import ABC, abstractmethod
from typing import Any, Literal
from pydantic import BaseModel


# ── Base classes ───────────────────────────────────────────────────────────────

class Effect(ABC):
    @abstractmethod
    def get_color(self, t: float, led_index: int, total_leds: int) -> list[int]:
        """Return [r, g, b] for a given LED at time t (seconds since activation)."""
        ...

    def on_beat(self, bpm: float, beat_number: int) -> None:
        """Beat-sync hook — no-op by default."""
        pass


class Filter(Effect):
    """Decorator base: wraps an inner Effect, modifies output, propagates on_beat."""
    def __init__(self, inner: Effect):
        self._inner = inner

    def get_color(self, t: float, led_index: int, total_leds: int) -> list[int]:
        return self._inner.get_color(t, led_index, total_leds)

    def on_beat(self, bpm: float, beat_number: int) -> None:
        self._inner.on_beat(bpm, beat_number)


# ── Filters ────────────────────────────────────────────────────────────────────

class GammaFilter(Filter):
    """Apply gamma correction: (v/255)^(1/gamma) * 255 per channel."""
    def __init__(self, inner: Effect, gamma: float = 2.2, **_):
        super().__init__(inner)
        self._inv_gamma = 1.0 / float(gamma)

    def get_color(self, t: float, led_index: int, total_leds: int) -> list[int]:
        rgb = self._inner.get_color(t, led_index, total_leds)
        return [int((v / 255.0) ** self._inv_gamma * 255) for v in rgb]


class MirrorFilter(Filter):
    """Fold the strip in half — left side mirrored onto right side."""
    def get_color(self, t: float, led_index: int, total_leds: int) -> list[int]:
        half = total_leds // 2
        idx = led_index if led_index < half else total_leds - 1 - led_index
        return self._inner.get_color(t, idx, half)


class ReverseFilter(Filter):
    """Reverse LED order."""
    def get_color(self, t: float, led_index: int, total_leds: int) -> list[int]:
        return self._inner.get_color(t, total_leds - 1 - led_index, total_leds)


class DimFilter(Filter):
    """Scale overall brightness."""
    def __init__(self, inner: Effect, brightness: float = 0.5, **_):
        super().__init__(inner)
        self._brightness = max(0.0, min(1.0, float(brightness)))

    def get_color(self, t: float, led_index: int, total_leds: int) -> list[int]:
        rgb = self._inner.get_color(t, led_index, total_leds)
        return [int(v * self._brightness) for v in rgb]


# ── Original effects ───────────────────────────────────────────────────────────

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
    def __init__(self, r: int = 255, g: int = 255, b: int = 255,
                 speed: float = 1.0, tail: int = 5, **_):
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


# ── New effects ────────────────────────────────────────────────────────────────

class Meteor(Effect):
    """Shooting star with exponential decay tail, optional bounce."""
    def __init__(self, r: int = 255, g: int = 200, b: int = 100,
                 speed: float = 1.0, tail_length: int = 20,
                 tail_decay: float = 0.85, bounce: bool = True, **_):
        self.r = int(r)
        self.g = int(g)
        self.b = int(b)
        self.speed = float(speed)
        self.tail_length = int(tail_length)
        self.tail_decay = float(tail_decay)
        self.bounce = bool(bounce)

    def get_color(self, t: float, led_index: int, total_leds: int) -> list[int]:
        if self.bounce:
            head = (math.sin(t * self.speed * math.pi) + 1) / 2 * (total_leds - 1)
        else:
            head = (t * self.speed * total_leds) % total_leds

        dist = head - led_index
        if -2 <= dist < 0:
            # bright head (2 LEDs wide)
            return [self.r, self.g, self.b]
        elif 0 <= dist < self.tail_length:
            brightness = self.tail_decay ** dist
            return [int(self.r * brightness), int(self.g * brightness), int(self.b * brightness)]
        return [0, 0, 0]


class Twinkle(Effect):
    """Deterministic star field — each LED twinkles independently."""
    def __init__(self, r: int = 255, g: int = 255, b: int = 255,
                 density: float = 0.3, speed: float = 1.0, **_):
        self.r = int(r)
        self.g = int(g)
        self.b = int(b)
        self.density = float(density)
        self.speed = float(speed)
        self._phases: list[tuple[bool, float]] = []

    def _ensure_phases(self, total_leds: int) -> None:
        if len(self._phases) == total_leds:
            return
        self._phases = []
        for i in range(total_leds):
            rng = random.Random(i * 2654435761)
            active = rng.random() <= self.density
            phase = rng.uniform(0, 2 * math.pi)
            self._phases.append((active, phase))

    def get_color(self, t: float, led_index: int, total_leds: int) -> list[int]:
        self._ensure_phases(total_leds)
        active, phase = self._phases[led_index]
        if not active:
            return [0, 0, 0]
        brightness = (math.sin(2 * math.pi * self.speed * t + phase) + 1) / 2
        return [int(self.r * brightness), int(self.g * brightness), int(self.b * brightness)]


class Fire(Effect):
    """Classic fire simulation with heat diffusion and palette mapping."""
    def __init__(self, cooling: float = 0.07, sparking: float = 0.12,
                 reverse: bool = False, **_):
        self.cooling = float(cooling)
        self.sparking = float(sparking)
        self.reverse = bool(reverse)
        self._heat: list[float] = []
        self._last_t: float = -1.0

    def _update(self, total_leds: int, dt: float) -> None:
        if len(self._heat) != total_leds:
            self._heat = [0.0] * total_leds

        # Step 1: Cool every cell
        for i in range(total_leds):
            cool = random.uniform(0, self.cooling * dt * 60)
            self._heat[i] = max(0.0, self._heat[i] - cool)

        # Step 2: Heat diffuses upward (from base toward tip)
        for i in range(total_leds - 1, 1, -1):
            self._heat[i] = (self._heat[i - 1] + self._heat[i - 2] * 2) / 3.0

        # Step 3: Randomly ignite base
        for i in range(min(3, total_leds)):
            if random.random() < self.sparking:
                self._heat[i] = min(1.0, self._heat[i] + random.uniform(0.5, 1.0))

    @staticmethod
    def _heat_to_rgb(heat: float) -> list[int]:
        if heat < 0.33:
            return [int(heat / 0.33 * 255), 0, 0]
        elif heat < 0.66:
            f = (heat - 0.33) / 0.33
            return [255, int(f * 255), 0]
        else:
            f = (heat - 0.66) / 0.34
            return [255, 255, int(f * 255)]

    def get_color(self, t: float, led_index: int, total_leds: int) -> list[int]:
        if abs(t - self._last_t) > 1e-4:
            dt = max(0.001, t - self._last_t) if self._last_t >= 0 else 0.016
            self._last_t = t
            self._update(total_leds, dt)

        if not self._heat:
            return [0, 0, 0]
        idx = led_index if not self.reverse else (total_leds - 1 - led_index)
        return self._heat_to_rgb(self._heat[idx])


class Plasma(Effect):
    """Psychedelic overlapping sine waves mapped to hue."""
    def __init__(self, speed: float = 1.0, scale: float = 1.0, **_):
        self.speed = float(speed)
        self.scale = float(scale)

    def get_color(self, t: float, led_index: int, total_leds: int) -> list[int]:
        pos = led_index / total_leds
        wave1 = math.sin(pos * self.scale * 6 * math.pi + t * self.speed)
        wave2 = math.sin(pos * self.scale * 4 * math.pi + t * self.speed * 0.7)
        hue = ((wave1 + wave2) / 2 + 1) / 2
        r, g, b = colorsys.hsv_to_rgb(hue, 1.0, 1.0)
        return [int(r * 255), int(g * 255), int(b * 255)]


class Larson(Effect):
    """KITT scanner — Gaussian bright peak bouncing back and forth."""
    def __init__(self, r: int = 255, g: int = 0, b: int = 0,
                 speed: float = 1.0, width: int = 5, **_):
        self.r = int(r)
        self.g = int(g)
        self.b = int(b)
        self.speed = float(speed)
        self.sigma = float(width) / 2.0

    def get_color(self, t: float, led_index: int, total_leds: int) -> list[int]:
        center = (math.sin(t * self.speed * math.pi) + 1) / 2 * (total_leds - 1)
        dist = abs(led_index - center)
        brightness = math.exp(-(dist ** 2) / (2 * self.sigma ** 2))
        return [int(self.r * brightness), int(self.g * brightness), int(self.b * brightness)]


class BeatFlash(Effect):
    """Flashes to full brightness on each beat and decays exponentially between beats."""
    def __init__(self, r: int = 255, g: int = 255, b: int = 255,
                 decay: float = 0.3, **_):
        self.r = int(r)
        self.g = int(g)
        self.b = int(b)
        self.decay = float(decay)
        self._last_beat_time: float = -1e9  # far in the past → near-zero brightness

    def on_beat(self, _bpm: float, _beat_number: int) -> None:
        self._last_beat_time = time.monotonic()

    def get_color(self, _t: float, led_index: int, total_leds: int) -> list[int]:
        dt = time.monotonic() - self._last_beat_time
        brightness = math.exp(-self.decay * dt * 10)
        return [int(self.r * brightness), int(self.g * brightness), int(self.b * brightness)]


# ── GPT schema ─────────────────────────────────────────────────────────────────

class FilterCommand(BaseModel):
    type: Literal["gamma", "mirror", "reverse", "dim"]
    params: dict[str, Any] = {}


class EffectCommand(BaseModel):
    effect: Literal[
        "solid", "color_wave", "pulse", "rainbow", "chase",
        "meteor", "twinkle", "fire", "plasma", "larson", "beat_flash",
    ]
    params: dict[str, Any] = {}
    filters: list[FilterCommand] = []
    bpm: float | None = None


# ── Registries & factory ───────────────────────────────────────────────────────

EFFECT_REGISTRY: dict[str, type[Effect]] = {
    "solid":       SolidColor,
    "color_wave":  ColorWave,
    "pulse":       Pulse,
    "rainbow":     Rainbow,
    "chase":       Chase,
    "meteor":      Meteor,
    "twinkle":     Twinkle,
    "fire":        Fire,
    "plasma":      Plasma,
    "larson":      Larson,
    "beat_flash":  BeatFlash,
}

FILTER_REGISTRY: dict[str, type[Filter]] = {
    "gamma":   GammaFilter,
    "mirror":  MirrorFilter,
    "reverse": ReverseFilter,
    "dim":     DimFilter,
}


def effect_from_dict(d: dict) -> Effect:
    effect: Effect = EFFECT_REGISTRY[d["effect"]](**d.get("params", {}))
    for f_spec in d.get("filters", []):
        filter_cls = FILTER_REGISTRY[f_spec["type"]]
        effect = filter_cls(effect, **f_spec.get("params", {}))
    return effect
