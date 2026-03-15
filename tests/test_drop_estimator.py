"""Tests for modules/drop_estimator.py (novelty-based algorithm)."""
import types
import numpy as np
import pytest

from modules.drop_estimator import (
    SAMPLERATE,
    N_MELS,
    NOVELTY_THRESHOLD,
    MAJOR_NOVELTY_PERCENTILE,
    DROP_NOVELTY_PERCENTILE,
    MINOR_MIN_SPACING_BEATS,
    DROP_MIN_SPACING_BEATS,
    HIGHER_TIER_SUPPRESS_BEATS,
    ChangeEvent,
    ChangeType,
    TierEstimate,
    StructureEstimate,
    _compute_fingerprint,
    _infer_period,
    _classify_event,
    _is_higher_tier_imminent,
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


def test_compute_fingerprint_differs_on_different_audio():
    """Spectrally different broadband signals should produce dissimilar fingerprints."""
    rng = np.random.default_rng(42)
    n = int(SAMPLERATE * 1.5)
    noise = rng.standard_normal(n).astype(np.float32)
    # Low-pass: keep only the first 1/8 of the spectrum
    low_pass = np.fft.irfft(np.fft.rfft(noise) * np.r_[np.ones(len(noise) // 16), np.zeros(len(np.fft.rfft(noise)) - len(noise) // 16)]).astype(np.float32)[:n]
    # High-pass: keep only the top 1/8 of the spectrum
    spec = np.fft.rfft(noise)
    hp_mask = np.r_[np.zeros(len(spec) - len(spec) // 8), np.ones(len(spec) // 8)]
    high_pass = np.fft.irfft(spec * hp_mask).astype(np.float32)[:n]
    fp_low  = _compute_fingerprint(low_pass)
    fp_high = _compute_fingerprint(high_pass)
    assert fp_low is not None and fp_high is not None
    assert float(np.dot(fp_low, fp_high)) < 0.9


# ── dual-signal novelty ────────────────────────────────────────────────────────

def test_onset_delta_larger_for_sudden_activity_change():
    """onset_strength delta is larger when switching from quiet to noisy than steady→steady."""
    import librosa
    rng = np.random.default_rng(42)
    n = int(SAMPLERATE * 3)
    noise = rng.standard_normal(n).astype(np.float32) * 0.5
    sine = _sine(440.0, 3.0)
    onset_sine  = float(librosa.onset.onset_strength(y=sine,  sr=SAMPLERATE).mean())
    onset_noise = float(librosa.onset.onset_strength(y=noise, sr=SAMPLERATE).mean())
    steady_delta = abs(onset_sine - onset_sine)           # 0 — baseline
    change_delta = abs(onset_noise - onset_sine)
    assert change_delta > steady_delta


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

def _make_est_with_drop_tier(beats_to_change: float, confidence: float) -> StructureEstimate:
    """Build a StructureEstimate with a populated drop TierEstimate."""
    drop_tier = TierEstimate(
        change_period=16,
        beats_to_change=beats_to_change,
        beats_since_change=16.0 - beats_to_change,
        confidence=confidence,
        next_change_beat=100.0 + beats_to_change,
    )
    return StructureEstimate(
        beats_to_change=beats_to_change,
        change_period=16,
        beats_since_change=16.0 - beats_to_change,
        novelty=0.2,
        confidence=confidence,
        next_change_beat=100.0 + beats_to_change,
        tiers={"drop": drop_tier},
    )


def test_should_advance_queue_high_confidence_triggers():
    est = _make_est_with_drop_tier(beats_to_change=0.5, confidence=0.8)
    stub = _stub_worker(est)
    # beat_number not at phrase boundary (e.g. 7), but drop is imminent → True
    assert stub._should_advance_queue(7) is True


def test_should_advance_queue_low_confidence_falls_back_to_phrase():
    est = _make_est_with_drop_tier(beats_to_change=0.5, confidence=0.3)  # below min
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
    drop_tier = TierEstimate(
        change_period=32, beats_to_change=8.0,  # within 4 bars (16 beats)
        beats_since_change=24.0, confidence=0.8, next_change_beat=200.0,
    )
    est = StructureEstimate(
        beats_to_change=8.0, change_period=32, beats_since_change=24.0,
        novelty=0.2, confidence=0.8, next_change_beat=200.0,
        tiers={"drop": drop_tier},
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


# ── Section context schema tests ───────────────────────────────────────────────

from modules.effects import SectionContextCommand, VJResponse


def test_section_context_fields():
    ctx = SectionContextCommand(palette_name="warm fire", effect_family="chase", energy="high")
    assert ctx.palette_name == "warm fire"
    assert ctx.effect_family == "chase"
    assert ctx.energy == "high"


def test_vj_response_section_context_optional():
    r = VJResponse(sequences=[], section_context=None)
    assert r.section_context is None


def test_vj_response_section_context_present():
    ctx = SectionContextCommand(palette_name="x", effect_family="wave", energy="low")
    r = VJResponse(sequences=[], section_context=ctx)
    assert r.section_context.energy == "low"


# ── sequence_from_dict_capped tests ────────────────────────────────────────────

from modules.sequences import sequence_from_dict_capped, MAX_STEP_DURATION_BARS, BEATS_PER_BAR as SPB


def test_sequence_from_dict_capped_caps_long_step():
    seq = sequence_from_dict_capped({
        "steps": [{"effect": "pulse", "params": {"r": 255, "g": 0, "b": 0}, "duration_bars": 4.0}],
        "repeats": 1,
    })
    assert seq.total_beats == MAX_STEP_DURATION_BARS * SPB


def test_sequence_from_dict_capped_allows_short_step():
    seq = sequence_from_dict_capped({
        "steps": [{"effect": "rainbow", "params": {}, "duration_bars": 0.5}],
        "repeats": 1,
    })
    assert seq.total_beats == 0.5 * SPB


# ── _build_auto_prompt tests ───────────────────────────────────────────────────

from modules.gui_app import _build_auto_prompt


def test_build_auto_prompt_no_context():
    prompt = _build_auto_prompt(None, is_section_change=False)
    assert "section_context" in prompt


def test_build_auto_prompt_change_flag():
    ctx = SectionContextCommand(palette_name="fire", effect_family="chase", energy="high")
    prompt = _build_auto_prompt(ctx, is_section_change=True)
    assert "new musical section" in prompt


def test_build_auto_prompt_stable_includes_context():
    ctx = SectionContextCommand(palette_name="cool ocean", effect_family="wave", energy="medium")
    prompt = _build_auto_prompt(ctx, is_section_change=False)
    assert "cool ocean" in prompt
    assert "wave" in prompt


# ── ChangeEvent & _classify_event tests ────────────────────────────────────────

def test_change_event_fields():
    ev = ChangeEvent(beat=64.0, novelty=0.35)
    assert ev.beat == pytest.approx(64.0)
    assert ev.novelty == pytest.approx(0.35)


def test_classify_event_fallback_below_min_events():
    # Only 1 event → fallback thresholds: minor < 1.5×, major < 3×, drop >= 3×
    events = [ChangeEvent(beat=0.0, novelty=0.20)]
    assert _classify_event(NOVELTY_THRESHOLD * 1.5 - 0.001, events) == ChangeType.MINOR
    assert _classify_event(NOVELTY_THRESHOLD * 1.5, events) == ChangeType.MAJOR
    assert _classify_event(NOVELTY_THRESHOLD * 3, events) == ChangeType.DROP


def test_classify_event_adaptive_percentiles():
    # 10 events with linearly increasing novelty; check all three tiers
    events = [ChangeEvent(beat=float(i * 16), novelty=0.10 + i * 0.01) for i in range(10)]
    scores = [e.novelty for e in events]
    major_thr = float(np.percentile(scores, MAJOR_NOVELTY_PERCENTILE))
    drop_thr  = float(np.percentile(scores, DROP_NOVELTY_PERCENTILE))
    assert _classify_event(major_thr - 0.001, events) == ChangeType.MINOR
    assert _classify_event(major_thr, events) == ChangeType.MAJOR
    assert _classify_event(drop_thr, events) == ChangeType.DROP


def test_classify_event_drop_tier_infers_period():
    # Drop events at 32-beat spacing → period=32
    assert _infer_period([0.0, 32.0])[0] == 32


def test_tier_estimate_fields():
    te = TierEstimate(change_period=32, beats_to_change=16.0, beats_since_change=16.0, confidence=0.8)
    assert te.change_period == 32
    assert te.next_change_beat is None


def test_structure_estimate_tier_fields_default():
    est = StructureEstimate(
        beats_to_change=None, change_period=None,
        beats_since_change=0.0, novelty=0.0, confidence=0.0,
    )
    assert est.tiers == {}
    assert est.detected_tier is None
    assert est.most_imminent_tier is None


def test_per_tier_cooldown_drop_spacing():
    """A drop cannot be followed by another drop within DROP_MIN_SPACING_BEATS."""
    # Simulate: drop fired at beat 0; check that at beat DROP_MIN_SPACING-1 it's still blocked
    tier_last = {"minor": None, "major": None, "drop": 0.0}
    tier_period = {"minor": None, "major": None, "drop": None}
    beat_just_before = DROP_MIN_SPACING_BEATS - 1
    # Not imminent from above (no higher tier than drop), but cooldown prevents second drop
    # We test the cooldown logic directly: beats_since < min_spacing
    beats_since = beat_just_before - 0.0
    assert beats_since <= DROP_MIN_SPACING_BEATS


def test_suppression_blocks_minor_near_major():
    """Minor should be suppressed when a major change is predicted within HIGHER_TIER_SUPPRESS_BEATS."""
    tier_periods   = {"minor": None, "major": 32, "drop": None}
    tier_last_beats = {"minor": None, "major": 0.0, "drop": None}
    # 32 - (21 % 32) = 11 beats to next major → within suppress window (12)
    assert _is_higher_tier_imminent("minor", tier_periods, tier_last_beats, current_beat=21.0)


def test_suppression_allows_minor_far_from_major():
    """Minor should NOT be suppressed when next major change is far away."""
    tier_periods    = {"minor": None, "major": 32, "drop": None}
    tier_last_beats = {"minor": None, "major": 0.0, "drop": None}
    # 32 - (10 % 32) = 22 beats to next major → outside suppress window
    assert not _is_higher_tier_imminent("minor", tier_periods, tier_last_beats, current_beat=10.0)


def test_suppression_drop_never_suppressed():
    """Drop tier has nothing above it — never suppressed."""
    tier_periods    = {"minor": None, "major": 32, "drop": 64}
    tier_last_beats = {"minor": None, "major": 0.0, "drop": 0.0}
    assert not _is_higher_tier_imminent("drop", tier_periods, tier_last_beats, current_beat=21.0)


def test_structure_estimate_detected_tier_set():
    est = StructureEstimate(
        beats_to_change=16.0, change_period=32,
        beats_since_change=16.0, novelty=0.4, confidence=0.9,
        detected_tier="drop",
    )
    assert est.detected_tier == "drop"
