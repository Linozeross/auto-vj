"""Tests for modules/sequences.py — Sequence and factories."""
import pytest
from modules.effects import SolidColor, Pulse
from modules.sequences import (
    Sequence, sequence_from_dict, sequence_from_effect_cmd, looping_sequence_from_effect_cmd,
    BEATS_PER_BAR, DEFAULT_DURATION_BARS, PHRASE_BEATS,
)


# ── Helpers ────────────────────────────────────────────────────────────────────

def solid(r=255, g=0, b=0) -> SolidColor:
    return SolidColor(r=r, g=g, b=b)


def make_seq(durations: list[float], repeats: int = 1) -> Sequence:
    effects = [solid(i * 10, 0, 0) for i in range(len(durations))]
    return Sequence(list(zip(effects, durations)), repeats=repeats)


# ── 1. Single-step get_color delegates correctly ───────────────────────────────

def test_single_step_delegates_get_color():
    e = solid(100, 200, 50)
    seq = Sequence([(e, 8.0)])
    assert seq.get_color(0.0, 0, 10) == [100, 200, 50]
    assert seq.get_color(4.0, 5, 10) == [100, 200, 50]


# ── 2. Multi-step: correct step selected by t ──────────────────────────────────

def test_multi_step_correct_step_by_t():
    e0 = solid(100, 0, 0)
    e1 = solid(0, 200, 0)
    seq = Sequence([(e0, 4.0), (e1, 4.0)])
    # t=0 → step 0
    assert seq.get_color(0.0, 0, 10) == [100, 0, 0]
    # t=3.9 → still step 0
    assert seq.get_color(3.9, 0, 10) == [100, 0, 0]
    # t=4.0 → step 1
    assert seq.get_color(4.0, 0, 10) == [0, 200, 0]
    # t=7.9 → still step 1
    assert seq.get_color(7.9, 0, 10) == [0, 200, 0]


# ── 3. Steps cycle within repeats ─────────────────────────────────────────────

def test_steps_cycle_within_repeats():
    e0 = solid(100, 0, 0)
    e1 = solid(0, 200, 0)
    seq = Sequence([(e0, 4.0), (e1, 4.0)], repeats=2)
    # Second repeat: t=8 → step 0 again
    assert seq.get_color(8.0, 0, 10) == [100, 0, 0]
    # t=12 → step 1 in second repeat
    assert seq.get_color(12.0, 0, 10) == [0, 200, 0]


# ── 4. Last frame frozen when done ────────────────────────────────────────────

def test_last_frame_frozen_when_done():
    e0 = solid(100, 0, 0)
    e1 = solid(0, 200, 0)
    seq = Sequence([(e0, 4.0), (e1, 4.0)], repeats=1)
    # At exactly total_beats (8.0), sequence is done → freeze last step (e1)
    color_at_end = seq.get_color(8.0, 0, 10)
    color_far_past = seq.get_color(100.0, 0, 10)
    # Both should be step 1 color (last step)
    assert color_at_end == [0, 200, 0]
    assert color_far_past == [0, 200, 0]


# ── 5. is_done: false before end, true at end ─────────────────────────────────

def test_is_done_false_before_end():
    seq = make_seq([8.0], repeats=1)
    assert not seq.is_done(0.0)
    assert not seq.is_done(7.99)


def test_is_done_true_at_end():
    seq = make_seq([8.0], repeats=1)
    assert seq.is_done(8.0)
    assert seq.is_done(100.0)


def test_is_done_respects_repeats():
    seq = make_seq([4.0], repeats=3)
    assert not seq.is_done(11.9)
    assert seq.is_done(12.0)


# ── 6. on_beat propagates to all steps ────────────────────────────────────────

def test_on_beat_propagates_to_all_steps():
    called = []

    class TrackingEffect(SolidColor):
        def __init__(self, tag):
            super().__init__(255, 0, 0)
            self.tag = tag
        def on_beat(self, bpm, beat_number):
            called.append(self.tag)

    e0 = TrackingEffect("a")
    e1 = TrackingEffect("b")
    seq = Sequence([(e0, 4.0), (e1, 4.0)])
    seq.on_beat(120.0, 1)
    assert "a" in called
    assert "b" in called


# ── 7. sequence_from_dict builds correct structure ────────────────────────────

def test_sequence_from_dict_correct_structure():
    cmd = {
        "steps": [
            {"effect": "solid", "params": {"r": 255, "g": 0, "b": 0}, "duration_bars": 4},
            {"effect": "pulse", "params": {"r": 0, "g": 255, "b": 0}, "duration_bars": 2},
        ],
        "repeats": 2,
        "name": "test-seq",
    }
    seq = sequence_from_dict(cmd)
    assert seq.name == "test-seq"
    assert seq.repeats == 2
    assert len(seq._steps) == 2
    assert seq.total_beats == (4 + 2) * BEATS_PER_BAR
    # Step 0 should return solid red
    c = seq.get_color(0.0, 0, 10)
    assert c == [255, 0, 0]


def test_sequence_from_dict_default_duration():
    cmd = {"steps": [{"effect": "rainbow"}], "repeats": 1, "name": ""}
    seq = sequence_from_dict(cmd)
    assert seq.total_beats == DEFAULT_DURATION_BARS * BEATS_PER_BAR


# ── 8. sequence_from_effect_cmd wraps as 1-step ───────────────────────────────

def test_sequence_from_effect_cmd_single_step():
    seq = sequence_from_effect_cmd({"effect": "solid", "params": {"r": 10, "g": 20, "b": 30}})
    assert len(seq._steps) == 1
    assert seq.total_beats == DEFAULT_DURATION_BARS * BEATS_PER_BAR
    assert seq.get_color(0.0, 0, 10) == [10, 20, 30]
    assert seq.name == "solid"


def test_sequence_from_effect_cmd_default_repeats():
    seq = sequence_from_effect_cmd({"effect": "rainbow"})
    assert seq.repeats == 1


def test_looping_sequence_from_effect_cmd_uses_large_repeat_count():
    seq = looping_sequence_from_effect_cmd({"effect": "rainbow"})
    assert seq.repeats > 1
    assert not seq.is_done(DEFAULT_DURATION_BARS * BEATS_PER_BAR)


# ── 9. Empty sequence returns black ──────────────────────────────────────────

def test_empty_sequence_returns_black():
    seq = Sequence([])
    assert seq.get_color(0.0, 0, 10) == [0, 0, 0]


# ── 10. repeats=0 clamped to 1 ───────────────────────────────────────────────

def test_repeats_zero_clamped_to_one():
    seq = make_seq([4.0], repeats=0)
    assert seq.repeats == 1
