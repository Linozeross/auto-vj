"""Tests for modules/effects.py — effects, filters, factory, schema."""
import pytest
import modules.effects as fx
from modules.effects import (
    # Effects
    SolidColor, ColorWave, Pulse, Rainbow, Chase,
    Meteor, Twinkle, Plasma, Larson, PaletteWave, BeatPulse,
    StripSolid, StripChase,
    # Filters
    GammaFilter, MirrorFilter, ReverseFilter, DimFilter, StripFilter,
    # Factory & registries
    effect_from_dict, EFFECT_REGISTRY, FILTER_REGISTRY,
    # Schema
    EffectCommand, FilterCommand,
    # Strip helpers
    led_to_strip,
)
# Note: SolidColor, ColorWave, Meteor, Larson and GammaFilter/MirrorFilter/ReverseFilter
# are kept as classes but removed from registries — imported here for class-level tests only.


# ── Helpers ───────────────────────────────────────────────────────────────────

def valid_rgb(color: list[int]) -> bool:
    return len(color) == 3 and all(isinstance(v, int) and 0 <= v <= 255 for v in color)


# ── Registry completeness ─────────────────────────────────────────────────────

def test_all_effects_in_registry():
    expected = {
        "solid", "pulse", "rainbow", "chase", "twinkle", "plasma",
        "palette_wave", "beat_pulse", "strip_solid", "strip_chase",
    }
    assert set(EFFECT_REGISTRY.keys()) == expected


def test_all_filters_in_registry():
    assert set(FILTER_REGISTRY.keys()) == {"dim", "strip"}


# ── All effects produce valid RGB ─────────────────────────────────────────────

@pytest.mark.parametrize("name,params", [
    ("pulse",        {"r": 255, "g": 0, "b": 0, "rate_hz": 2.0}),
    ("pulse",        {"r": 255, "g": 0, "b": 0, "rate": 2.0}),
    ("rainbow",      {"speed": 3.0}),
    ("chase",        {"r": 0, "g": 0, "b": 255, "speed": 2.0, "tail": 8}),
    ("twinkle",      {"r": 255, "g": 255, "b": 255, "density": 0.5, "speed": 1.0}),
    ("plasma",       {"speed": 1.0, "scale": 1.0}),
    ("palette_wave", {"palette": "ocean", "speed": 1.0}),
    ("palette_wave", {"palette": [[0, 0, 0], [255, 0, 0], [255, 200, 0]], "speed": 1.0}),
    ("beat_pulse",   {"r": 255, "g": 100, "b": 0, "sharpness": 4.0}),
])
def test_effect_produces_valid_rgb(name, params):
    e = effect_from_dict({"effect": name, "params": params})
    for led in range(0, 100, 10):
        c = e.get_color(1.0, led, 100)
        assert valid_rgb(c), f"{name} LED {led}: invalid RGB {c}"


# ── SolidColor ─────────────────────────────────────────────────────────────────

def test_solid_color_is_constant():
    e = SolidColor(r=10, g=20, b=30)
    assert e.get_color(0.0, 0, 100) == [10, 20, 30]
    assert e.get_color(99.9, 99, 100) == [10, 20, 30]


# ── Pulse ──────────────────────────────────────────────────────────────────────

def test_pulse_half_brightness_at_t0():
    e = Pulse(r=200, g=100, b=50, rate_hz=1.0)
    # sin(0) = 0 → brightness = 0.5
    c = e.get_color(0.0, 0, 100)
    assert c == [100, 50, 25]


def test_pulse_max_brightness_at_quarter_period():
    e = Pulse(r=200, g=0, b=0, rate_hz=1.0)
    # t = 0.25 → sin(π/2) = 1 → brightness = 1.0
    c = e.get_color(0.25, 0, 100)
    assert c[0] == 200


# ── Larson ─────────────────────────────────────────────────────────────────────

def test_larson_peak_near_center_at_t0():
    e = Larson(r=255, g=0, b=0, speed=1.0, width=5)
    colors = [e.get_color(0.0, i, 100) for i in range(100)]
    peak = max(range(100), key=lambda i: colors[i][0])
    assert 45 <= peak <= 55, f"Larson peak at t=0 should be ~49-50, got {peak}"


def test_larson_gaussian_falloff():
    e = Larson(r=255, g=0, b=0, speed=1.0, width=5)
    c_center = e.get_color(0.0, 50, 100)[0]
    c_edge = e.get_color(0.0, 0, 100)[0]
    assert c_center > c_edge


# ── on_beat hook propagation ──────────────────────────────────────────────────

def test_on_beat_is_noop_for_stateless_effects():
    for cls in [SolidColor, Rainbow, Plasma]:
        e = cls()
        e.on_beat(120.0, 1)  # should not raise


# ── Meteor ────────────────────────────────────────────────────────────────────

def test_meteor_bounce_stays_in_bounds():
    e = Meteor(r=255, g=255, b=255, speed=1.0, bounce=True)
    for t in [0.0, 0.25, 0.5, 0.75, 1.0]:
        for led in range(100):
            c = e.get_color(t, led, 100)
            assert valid_rgb(c)


# ── Twinkle ───────────────────────────────────────────────────────────────────

def test_twinkle_deterministic():
    e = Twinkle(r=255, g=255, b=255, density=0.5, speed=1.0)
    c1 = e.get_color(1.0, 5, 100)
    c2 = e.get_color(1.0, 5, 100)
    assert c1 == c2, "Same t and LED index should give same color"


def test_twinkle_density_zero_all_off():
    e = Twinkle(r=255, g=255, b=255, density=0.0)
    for i in range(100):
        assert e.get_color(0.0, i, 100) == [0, 0, 0]


# ── BeatPulse ─────────────────────────────────────────────────────────────────

def test_beat_pulse_bright_at_beat_boundary():
    e = BeatPulse(r=255, g=0, b=0, sharpness=4.0)
    # t=0.0 → phase=0 → brightness = exp(0) = 1.0
    c = e.get_color(0.0, 0, 100)
    assert c[0] == 255


def test_beat_pulse_decays_within_beat():
    e = BeatPulse(r=255, g=0, b=0, sharpness=4.0)
    c_start = e.get_color(0.0, 0, 100)[0]
    c_mid = e.get_color(0.5, 0, 100)[0]
    assert c_mid < c_start


# ── Filters ───────────────────────────────────────────────────────────────────

def test_gamma_filter_darkens_midtones():
    base = SolidColor(128, 128, 128)
    filtered = GammaFilter(base, gamma=2.2)
    c = filtered.get_color(0, 0, 100)
    assert all(v > 128 for v in c), f"Gamma should brighten midtones, got {c}"


def test_dim_filter_scales_correctly():
    base = SolidColor(200, 100, 50)
    filtered = DimFilter(base, brightness=0.5)
    c = filtered.get_color(0, 0, 100)
    assert c == [100, 50, 25]


def test_reverse_filter_flips_order():
    base = Chase(r=255, g=0, b=0, speed=0.0, tail=10)
    rev = ReverseFilter(base)
    c_normal = base.get_color(0, 0, 100)
    c_reversed = rev.get_color(0, 99, 100)
    assert c_normal == c_reversed


def test_mirror_filter_symmetric():
    base = Rainbow(speed=0.0)
    mirrored = MirrorFilter(base)
    assert mirrored.get_color(0, 0, 100) == mirrored.get_color(0, 99, 100)
    assert mirrored.get_color(0, 10, 100) == mirrored.get_color(0, 89, 100)


def test_filter_propagates_on_beat():
    base = SolidColor(255, 0, 0)
    dim = DimFilter(base, brightness=0.8)
    dim.on_beat(120, 1)  # should not raise


# ── effect_from_dict factory ──────────────────────────────────────────────────

def test_effect_from_dict_basic():
    e = effect_from_dict({"effect": "pulse", "params": {"r": 255, "g": 0, "b": 0}})
    assert isinstance(e, Pulse)


def test_effect_from_dict_with_filters():
    e = effect_from_dict({
        "effect": "rainbow",
        "params": {"speed": 1.0},
        "filters": [
            {"type": "dim", "params": {"brightness": 0.5}},
        ],
    })
    c = e.get_color(0.0, 0, 100)
    assert valid_rgb(c)
    raw = Rainbow(speed=1.0).get_color(0.0, 0, 100)
    assert max(c) <= max(raw)


def test_effect_from_dict_empty_filters():
    e = effect_from_dict({"effect": "pulse", "params": {}, "filters": []})
    assert isinstance(e, Pulse)


def test_effect_from_dict_unknown_effect_raises():
    with pytest.raises(KeyError):
        effect_from_dict({"effect": "not_a_real_effect", "params": {}})


# ── Pydantic schema ───────────────────────────────────────────────────────────

def test_effect_command_defaults():
    cmd = EffectCommand(effect="rainbow")
    assert cmd.params == {}
    assert cmd.filters == []
    assert cmd.bpm is None


def test_effect_command_with_filters():
    cmd = EffectCommand(
        effect="pulse",
        params={"r": 255, "g": 0, "b": 0},
        filters=[FilterCommand(type="dim", params={"brightness": 0.5})],
        bpm=128.0,
    )
    assert cmd.effect == "pulse"
    assert cmd.bpm == 128.0
    assert cmd.filters[0].type == "dim"


def test_filter_command_invalid_type():
    with pytest.raises(Exception):
        FilterCommand(type="not_a_filter")


# ── Strip layout helper ────────────────────────────────────────────────────────

def test_led_to_strip_maps_correctly():
    sizes = [25, 25, 25, 25]
    assert led_to_strip(0, sizes) == (0, 0, 25)
    assert led_to_strip(24, sizes) == (0, 24, 25)
    assert led_to_strip(25, sizes) == (1, 0, 25)
    assert led_to_strip(75, sizes) == (3, 0, 25)


def test_led_to_strip_unequal_strips():
    sizes = [10, 20, 30]
    assert led_to_strip(0, sizes) == (0, 0, 10)
    assert led_to_strip(10, sizes) == (1, 0, 20)
    assert led_to_strip(30, sizes) == (2, 0, 30)


# ── StripFilter ────────────────────────────────────────────────────────────────

def test_strip_filter_isolate_same_position_same_color(monkeypatch):
    monkeypatch.setattr(fx, "STRIP_SIZES", [25, 25])
    inner = Rainbow(speed=0.0)
    f = StripFilter(inner, mode="isolate")
    # LED 0 (strip 0, pos 0) and LED 25 (strip 1, pos 0) — same local position → same color
    assert f.get_color(0, 0, 50) == f.get_color(0, 25, 50)


def test_strip_filter_phase_offsets_time(monkeypatch):
    monkeypatch.setattr(fx, "STRIP_SIZES", [25, 25])
    inner = Pulse(r=255, g=0, b=0, rate=1.0)
    f = StripFilter(inner, mode="phase", phase_shift=0.25)
    # strip 0 (t=0, brightness=0.5) vs strip 1 (t=0.25, brightness=1.0) → different
    assert f.get_color(0, 0, 50) != f.get_color(0, 25, 50)


# ── StripSolid ────────────────────────────────────────────────────────────────

def test_strip_solid_distinct_hues_per_strip(monkeypatch):
    monkeypatch.setattr(fx, "STRIP_SIZES", [25, 25])
    e = StripSolid(speed=0.0)
    assert e.get_color(0, 0, 50) != e.get_color(0, 25, 50)


def test_strip_solid_uniform_within_strip(monkeypatch):
    monkeypatch.setattr(fx, "STRIP_SIZES", [25, 25])
    e = StripSolid(speed=0.0)
    # All LEDs within strip 0 should be the same color
    colors = [e.get_color(0, i, 50) for i in range(25)]
    assert all(c == colors[0] for c in colors)


# ── StripChase ────────────────────────────────────────────────────────────────

def test_strip_chase_head_is_full_brightness(monkeypatch):
    monkeypatch.setattr(fx, "STRIP_SIZES", [25, 25, 25, 25])
    e = StripChase(r=255, g=0, b=0, speed=1.0, tail=0)
    assert e.get_color(0, 0, 100) == [255, 0, 0]


def test_strip_chase_non_tail_strip_is_dark(monkeypatch):
    monkeypatch.setattr(fx, "STRIP_SIZES", [25, 25, 25, 25])
    e = StripChase(r=255, g=0, b=0, speed=1.0, tail=0)
    assert e.get_color(0, 50, 100) == [0, 0, 0]
