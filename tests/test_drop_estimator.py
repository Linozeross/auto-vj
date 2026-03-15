"""Tests for modules/drop_estimator.py (novelty-based algorithm)."""
import types
import numpy as np
import pytest

from modules.drop_estimator import (
    SAMPLERATE,
    N_MELS,
    StructureEstimate,
    _compute_fingerprint,
    _infer_period,
)
from modules.gui_app import (
    VJWorker,
    STRUCTURE_CONFIDENCE_MIN,
    STRUCTURE_BEATS_WINDOW,
    AUTO_TRIGGER_AHEAD_BARS,
)
from modules.sequences import BEATS_PER_BAR, PHRASE_BEATS


def _stub_worker(last_structure: StructureEstimate | None) -> object:
    """Minimal stub that exercises VJWorker queue-helper methods without Qt."""
    stub = types.SimpleNamespace(_last_structure=last_structure)
    stub._should_advance_queue = types.MethodType(VJWorker._should_advance_queue, stub)
    stub._should_trigger_auto_refill = types.MethodType(VJWorker._should_trigger_auto_refill, stub)
    return stub


def _sine(freq_hz: float = 440.0, duration_secs: float = 4.0) -> np.ndarray:
    t = np.linspace(0, duration_secs, int(SAMPLERATE * duration_secs), endpoint=False)
    return (0.5 * np.sin(2 * np.pi * freq_hz * t)).astype(np.float32)


# ── _compute_fingerprint ───────────────────────────────────────────────────────

def test_compute_fingerprint_shape():
    fp = _compute_fingerprint(_sine())
    assert fp is not None
    assert fp.shape == (N_MELS,)


def test_compute_fingerprint_unit_norm():
    fp = _compute_fingerprint(_sine())
    assert fp is not None
    assert abs(float(np.linalg.norm(fp)) - 1.0) < 1e-6


def test_compute_fingerprint_silent_returns_none():
    silent = np.zeros(SAMPLERATE * 3, dtype=np.float32)
    assert _compute_fingerprint(silent) is None


def test_compute_fingerprint_too_short_returns_none():
    assert _compute_fingerprint(np.zeros(64, dtype=np.float32)) is None


# ── _infer_period ──────────────────────────────────────────────────────────────

def test_infer_period_consistent_32():
    change_beats = [0.0, 32.0, 64.0, 96.0]
    period, confidence = _infer_period(change_beats)
    assert period == 32
    assert confidence == pytest.approx(1.0)


def test_infer_period_consistent_16():
    change_beats = [0.0, 16.0, 32.0, 48.0]
    period, confidence = _infer_period(change_beats)
    assert period == 16
    assert confidence == pytest.approx(1.0)


def test_infer_period_too_few_events():
    period, confidence = _infer_period([64.0])
    assert period is None
    assert confidence == pytest.approx(0.0)


def test_infer_period_noisy_still_finds_dominant():
    # 3 intervals of 32 beats, 1 of 30 (snaps to 32) → consistent
    change_beats = [0.0, 32.0, 64.0, 94.0, 126.0]
    period, confidence = _infer_period(change_beats)
    assert period == 32


# ── StructureEstimate ──────────────────────────────────────────────────────────

def test_structure_estimate_fields():
    est = StructureEstimate(
        beats_to_change=16.0,
        change_period=32,
        beats_since_change=16.0,
        novelty=0.25,
        confidence=0.8,
    )
    assert est.beats_to_change == 16.0
    assert est.change_period == 32
    assert est.novelty == pytest.approx(0.25)
    assert est.confidence == pytest.approx(0.8)


def test_structure_estimate_none_countdown():
    est = StructureEstimate(
        beats_to_change=None,
        change_period=None,
        beats_since_change=0.0,
        novelty=0.0,
        confidence=0.0,
    )
    assert est.beats_to_change is None
    assert est.change_period is None


def test_structure_estimate_next_change_beat_defaults_none():
    est = StructureEstimate(
        beats_to_change=8.0,
        change_period=16,
        beats_since_change=8.0,
        novelty=0.1,
        confidence=0.8,
    )
    assert est.next_change_beat is None


def test_structure_estimate_next_change_beat_set():
    est = StructureEstimate(
        beats_to_change=8.0,
        change_period=16,
        beats_since_change=8.0,
        novelty=0.1,
        confidence=0.8,
        next_change_beat=100.0,
    )
    assert est.next_change_beat == pytest.approx(100.0)


# ── _should_advance_queue ──────────────────────────────────────────────────────

def test_should_advance_queue_high_confidence_triggers():
    est = StructureEstimate(
        beats_to_change=0.5,
        change_period=16,
        beats_since_change=15.5,
        novelty=0.2,
        confidence=0.8,
        next_change_beat=100.5,
    )
    stub = _stub_worker(est)
    # beat_number not at phrase boundary (e.g. 7), but drop is imminent
    assert stub._should_advance_queue(7) is True


def test_should_advance_queue_low_confidence_falls_back_to_phrase():
    est = StructureEstimate(
        beats_to_change=0.5,
        change_period=16,
        beats_since_change=15.5,
        novelty=0.2,
        confidence=0.3,   # below STRUCTURE_CONFIDENCE_MIN
        next_change_beat=100.5,
    )
    stub = _stub_worker(est)
    assert stub._should_advance_queue(16) is True   # phrase boundary → True
    assert stub._should_advance_queue(7) is False   # not phrase boundary → False


def test_should_advance_queue_no_estimate_falls_back_to_phrase():
    stub = _stub_worker(None)
    assert stub._should_advance_queue(16) is True
    assert stub._should_advance_queue(5) is False


# ── _should_trigger_auto_refill ────────────────────────────────────────────────

def test_should_trigger_auto_refill_low_remaining_beats():
    stub = _stub_worker(None)
    # 1 bar remaining — below the 4-bar threshold
    assert stub._should_trigger_auto_refill(BEATS_PER_BAR * 1) is True


def test_should_trigger_auto_refill_drop_approaching():
    est = StructureEstimate(
        beats_to_change=8.0,  # within 4 bars (16 beats)
        change_period=32,
        beats_since_change=24.0,
        novelty=0.2,
        confidence=0.8,
        next_change_beat=200.0,
    )
    stub = _stub_worker(est)
    # Plenty of remaining beats, but a drop is coming soon
    assert stub._should_trigger_auto_refill(BEATS_PER_BAR * 10) is True


def test_should_trigger_auto_refill_drop_far_away():
    est = StructureEstimate(
        beats_to_change=30.0,   # far in the future
        change_period=32,
        beats_since_change=2.0,
        novelty=0.1,
        confidence=0.9,
        next_change_beat=300.0,
    )
    stub = _stub_worker(est)
    assert stub._should_trigger_auto_refill(BEATS_PER_BAR * 10) is False
