"""Tests for modules/drop_estimator.py (novelty-based algorithm)."""
import numpy as np
import pytest

from modules.drop_estimator import (
    SAMPLERATE,
    N_MELS,
    StructureEstimate,
    _compute_fingerprint,
    _infer_period,
)


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
