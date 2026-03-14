import math
import random
import time
import colorsys
from abc import ABC, abstractmethod
from typing import Any, Callable, Literal
from pydantic import BaseModel


# ── Audio level source (injected from tui.py) ──────────────────────────────────

_audio_level_fn: Callable[[], float] = lambda: 0.0


def set_audio_level_source(fn: Callable[[], float]) -> None:
    global _audio_level_fn
    _audio_level_fn = fn


# ── Color palettes ─────────────────────────────────────────────────────────────

# Each palette is a list of (r, g, b) waypoints interpolated by _palette_color().
PALETTES: dict[str, list[tuple[int, int, int]]] = {
    "ember":   [(0, 0, 0), (120, 0, 0), (255, 50, 0), (255, 160, 20), (255, 240, 160)],
    "ocean":   [(0, 0, 20), (0, 20, 80), (0, 90, 180), (0, 190, 220), (180, 240, 255)],
    "forest":  [(0, 10, 0), (0, 60, 10), (10, 140, 20), (80, 210, 40), (190, 255, 140)],
    "violet":  [(10, 0, 20), (70, 0, 120), (180, 0, 190), (220, 80, 210), (255, 200, 255)],
    "sunset":  [(20, 0, 5), (180, 20, 0), (255, 70, 0), (255, 150, 60), (180, 80, 180)],
    "ice":     [(0, 0, 40), (0, 50, 140), (60, 150, 220), (160, 215, 255), (255, 255, 255)],
    "lava":    [(0, 0, 0), (90, 0, 0), (200, 0, 0), (255, 80, 0), (255, 210, 0)],
}


def _palette_color(palette: list[tuple[int, int, int]], pos: float) -> list[int]:
    """Linearly interpolate along a palette. pos: 0.0 → 1.0."""
    pos = max(0.0, min(1.0, pos))
    n = len(palette) - 1
    scaled = pos * n
    idx = int(scaled)
    frac = scaled - idx
    if idx >= n:
        return list(palette[-1])
    r1, g1, b1 = palette[idx]
    r2, g2, b2 = palette[idx + 1]
    return [
        int(r1 + (r2 - r1) * frac),
        int(g1 + (g2 - g1) * frac),
        int(b1 + (b2 - b1) * frac),
    ]


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
    def __init__(self, r: int = 255, g: int = 255, b: int = 255,
                 rate: float = 1.0, rate_hz: float | None = None, **_):
        self.r = int(r)
        self.g = int(g)
        self.b = int(b)
        self.rate = float(rate_hz if rate_hz is not None else rate)

    def get_color(self, t: float, led_index: int, total_leds: int) -> list[int]:
        brightness = (math.sin(2 * math.pi * self.rate * t) + 1) / 2
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


# ── Palette & beat & audio effects ────────────────────────────────────────────

class PaletteWave(Effect):
    """Scrolls a named color palette along the strip — no harsh primaries."""
    def __init__(self, palette: str = "ember", speed: float = 1.0,
                 stretch: float = 1.0, **_):
        self._palette = PALETTES.get(palette, PALETTES["ember"])
        self.speed = float(speed)
        self.stretch = float(stretch)  # > 1 = wider gradient repeat

    def get_color(self, t: float, led_index: int, total_leds: int) -> list[int]:
        pos = (led_index / total_leds * self.stretch + t * self.speed) % 1.0
        return _palette_color(self._palette, pos)


class BeatPulse(Effect):
    """
    Sharp flash at beat boundary (t % 1.0 == 0) with exponential decay until
    next beat — fully driven by beat-time t, no wall-clock drift.
    """
    BEAT_SHARPNESS_DEFAULT = 4.0

    def __init__(self, r: int = 255, g: int = 100, b: int = 0,
                 sharpness: float = BEAT_SHARPNESS_DEFAULT, **_):
        self.r = int(r)
        self.g = int(g)
        self.b = int(b)
        self.sharpness = float(sharpness)  # higher = faster decay within beat

    def get_color(self, t: float, led_index: int, total_leds: int) -> list[int]:
        phase = t % 1.0  # 0.0 at beat, 1.0 just before next beat
        brightness = math.exp(-self.sharpness * phase)
        return [int(self.r * brightness), int(self.g * brightness), int(self.b * brightness)]


class AudioPulse(Effect):
    """All LEDs scale with microphone amplitude — reacts to bass hits and vocals."""
    AUDIO_SENSITIVITY_DEFAULT = 2.0

    def __init__(self, r: int = 255, g: int = 60, b: int = 0,
                 sensitivity: float = AUDIO_SENSITIVITY_DEFAULT, **_):
        self.r = int(r)
        self.g = int(g)
        self.b = int(b)
        self.sensitivity = float(sensitivity)

    def get_color(self, t: float, led_index: int, total_leds: int) -> list[int]:
        level = min(1.0, _audio_level_fn() * self.sensitivity)
        return [int(self.r * level), int(self.g * level), int(self.b * level)]


class AudioWave(Effect):
    """
    Audio amplitude radiates outward from strip center as a colored ring.
    Uses a named palette so colors stay rich.
    """
    AUDIO_SENSITIVITY_DEFAULT = 2.0

    def __init__(self, palette: str = "ember",
                 sensitivity: float = AUDIO_SENSITIVITY_DEFAULT, **_):
        self._palette = PALETTES.get(palette, PALETTES["ember"])
        self.sensitivity = float(sensitivity)

    def get_color(self, t: float, led_index: int, total_leds: int) -> list[int]:
        level = min(1.0, _audio_level_fn() * self.sensitivity)
        center = total_leds / 2.0
        dist = abs(led_index - center) / center  # 0 at center, 1 at edges
        # light up from center outward proportional to level
        if dist > level:
            return [0, 0, 0]
        brightness = 1.0 - (dist / max(level, 1e-4))
        color = _palette_color(self._palette, dist)
        return [int(c * brightness) for c in color]


# ── GPT schema ─────────────────────────────────────────────────────────────────

class FilterCommand(BaseModel):
    type: Literal["gamma", "mirror", "reverse", "dim"]
    params: dict[str, Any] = {}


class EffectCommand(BaseModel):
    effect: Literal[
        "solid", "color_wave", "pulse", "rainbow", "chase",
        "meteor", "twinkle", "fire", "plasma", "larson", "beat_flash",
        "palette_wave", "beat_pulse", "audio_pulse", "audio_wave",
    ]
    params: dict[str, Any] = {}
    filters: list[FilterCommand] = []
    bpm: float | None = None


# ── Registries & factory ───────────────────────────────────────────────────────

EFFECT_REGISTRY: dict[str, type[Effect]] = {
    "solid":        SolidColor,
    "color_wave":   ColorWave,
    "pulse":        Pulse,
    "rainbow":      Rainbow,
    "chase":        Chase,
    "meteor":       Meteor,
    "twinkle":      Twinkle,
    "fire":         Fire,
    "plasma":       Plasma,
    "larson":       Larson,
    "beat_flash":   BeatFlash,
    "palette_wave": PaletteWave,
    "beat_pulse":   BeatPulse,
    "audio_pulse":  AudioPulse,
    "audio_wave":   AudioWave,
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
