"""Real-time song structure estimation via spectral novelty.

Algorithm:
  Every ANALYSIS_INTERVAL_SECS, compute a mel-spectrogram fingerprint of the
  last FINGERPRINT_WINDOW_SECS of audio.  The novelty score is the cosine
  distance between the current fingerprint and one COMPARISON_OFFSET steps ago.
  A novelty spike above NOVELTY_THRESHOLD → a drop just occurred.

  The inferred drop period is used to derive major and minor change periods:
      drop_period   — detected from high-novelty event spacing
      major_period  = drop_period // MAJOR_PERIOD_DIVISOR  (e.g. 64 → 32)
      minor_period  = drop_period // MINOR_PERIOD_DIVISOR  (e.g. 64 → 16)

  All three tiers share the last drop beat as grid anchor:
      tier.beats_to_change = derived_period - (current_beat - last_drop_beat) % derived_period

  StructureEstimate.tiers contains TierEstimate objects for all three tiers.
  Flat fields (beats_to_change, confidence, next_change_beat) are populated
  from the most imminent tier for backwards compatibility.
"""
from __future__ import annotations

import threading
from collections import deque
from dataclasses import dataclass, field
from typing import Callable

import librosa
import numpy as np

# ── Constants ──────────────────────────────────────────────────────────────────

SAMPLERATE = 44100               # must match beat_detector.py
HOP_SIZE = 512                   # ~11.6 ms per hop
N_MELS = 64                      # mel bands for spectral fingerprint
FINGERPRINT_WINDOW_SECS = 1.0    # audio window per fingerprint
ANALYSIS_INTERVAL_SECS  = 0.10   # analysis timer interval
FINGERPRINT_HISTORY_LEN = 16     # steps kept in fingerprint history deque
COMPARISON_OFFSET       = 10     # compare vs ≈1 s ago (10 × 0.10 s)

NOVELTY_EMA_ALPHA = 0.35         # EMA weight for new novelty values (lower = smoother)
NOVELTY_THRESHOLD = 0.20         # smoothed novelty threshold → drop detected
NOVELTY_DISPLAY_MAX = NOVELTY_THRESHOLD * 2  # novelty value that fills the bar
DROP_MIN_SPACING_BEATS  = 32     # min beats between drops (≈8 bars)
PHRASE_CANDIDATES = [8, 16, 32, 64]  # valid EDM phrase lengths in beats
MAX_CHANGE_HISTORY = 8           # drop events used for period inference
CONFIDENCE_REQUIRED_RATIO = 0.6  # fraction of intervals that must agree

MAJOR_PERIOD_DIVISOR = 2         # major_period = drop_period // MAJOR_PERIOD_DIVISOR
MINOR_PERIOD_DIVISOR = 4         # minor_period = drop_period // MINOR_PERIOD_DIVISOR

_BUFFER_SAMPLES = int(SAMPLERATE * FINGERPRINT_WINDOW_SECS)

_TIER_NAMES = ("minor", "major", "drop")


# ── Data structures ────────────────────────────────────────────────────────────

@dataclass
class ChangeEvent:
    beat: float     # Link beat position at detection
    novelty: float  # raw novelty at detection


@dataclass
class TierEstimate:
    change_period: int | None         # inferred (or derived) phrase period for this tier
    beats_to_change: float | None     # countdown to next change in this tier
    beats_since_change: float         # beats since last event in this tier
    confidence: float                 # period inference confidence 0.0–1.0
    next_change_beat: float | None = None  # absolute beat of next expected change


@dataclass
class StructureEstimate:
    # Flat fields — populated from the most imminent tier (backwards compatible)
    beats_to_change: float | None  # countdown to next change; None = not established
    change_period: int | None      # inferred phrase period (16/32/64); None = unknown
    beats_since_change: float      # beats since last detected change
    novelty: float                 # current smoothed novelty score 0.0–1.0
    confidence: float              # 0.0–1.0 (consistency of detected changes)
    next_change_beat: float | None = None  # absolute beat position of next change
    # Tier-aware fields
    detected_tier: str | None = None        # "drop" when a drop was just detected, else None
    most_imminent_tier: str | None = None   # tier of soonest known upcoming change
    tiers: dict = field(default_factory=dict)  # dict[str, TierEstimate]


# ── Helpers ────────────────────────────────────────────────────────────────────

def _compute_fingerprint(audio: np.ndarray) -> np.ndarray | None:
    """Mel spectrogram → time-averaged → L2-normalised 64-float vector."""
    if len(audio) < HOP_SIZE * 4:
        return None
    mel = librosa.feature.melspectrogram(
        y=audio, sr=SAMPLERATE, n_mels=N_MELS, hop_length=HOP_SIZE
    )
    vec = mel.mean(axis=1).astype(np.float64)
    norm = float(np.linalg.norm(vec))
    if norm < 1e-8:
        return None
    return vec / norm


def _infer_period(
    change_beats: list[float],
) -> tuple[int | None, float]:
    """
    From a list of change beat positions, infer the phrase period and confidence.

    Returns (period_beats, confidence) where confidence is the fraction of
    inter-change intervals that agree with the inferred period.
    """
    if len(change_beats) < 2:
        return None, 0.0

    intervals = [
        change_beats[i + 1] - change_beats[i]
        for i in range(len(change_beats) - 1)
    ]

    # Snap each interval to nearest phrase candidate
    snapped = []
    for interval in intervals:
        best = min(PHRASE_CANDIDATES, key=lambda c: abs(c - interval))
        snapped.append(best)

    if not snapped:
        return None, 0.0

    # Most common snapped value
    counts: dict[int, int] = {}
    for s in snapped:
        counts[s] = counts.get(s, 0) + 1
    period = max(counts, key=lambda k: counts[k])
    confidence = counts[period] / len(snapped)

    if confidence < CONFIDENCE_REQUIRED_RATIO:
        return None, confidence
    return period, confidence


def _derive_tier_periods(
    drop_period: int | None,
) -> dict[str, int | None]:
    """Derive minor and major periods from the drop period using fixed divisors."""
    if drop_period is None:
        return {"drop": None, "major": None, "minor": None}
    major = drop_period // MAJOR_PERIOD_DIVISOR
    minor = drop_period // MINOR_PERIOD_DIVISOR
    return {
        "drop": drop_period,
        "major": major if major >= 1 else None,
        "minor": minor if minor >= 1 else None,
    }


# ── Main class ─────────────────────────────────────────────────────────────────

class DropEstimator:
    """
    Consumes raw audio chunks (via feed_audio from MicBeatDetector's fan-out),
    computes spectral novelty, detects drops (biggest structural changes),
    infers a drop period, and derives major/minor periods as halves/quarters.

    Lifecycle mirrors MicBeatDetector: start() / stop() / pause() / resume().
    """

    def __init__(self, link_clock) -> None:
        self._link_clock = link_clock
        self._callback: Callable[[StructureEstimate], None] | None = None

        # Rolling audio buffer
        self._buffer: deque[np.ndarray] = deque()
        self._buffer_lock = threading.Lock()

        # Fingerprint history (L2-normalised mel vectors)
        self._fingerprint_history: deque[np.ndarray] = deque(maxlen=FINGERPRINT_HISTORY_LEN)

        # Drop event tracking
        self._drop_events: list[ChangeEvent] = []
        self._drop_last_beat: float | None = None
        self._drop_period: int | None = None

        self._last_novelty: float = 0.0       # raw
        self._smoothed_novelty: float = 0.0   # EMA-smoothed, used for display + threshold

        # Analysis timer
        self._timer: threading.Timer | None = None
        self._running = False
        self._paused = False

        # Thread-safe last estimate
        self._last_estimate = StructureEstimate(
            beats_to_change=None,
            change_period=None,
            beats_since_change=0.0,
            novelty=0.0,
            confidence=0.0,
        )
        self._estimate_lock = threading.Lock()

    # ── Public API ─────────────────────────────────────────────────────────────

    def set_callback(self, cb: Callable[[StructureEstimate], None]) -> None:
        self._callback = cb

    def get_estimate(self) -> StructureEstimate:
        with self._estimate_lock:
            return self._last_estimate

    def feed_audio(self, chunk: np.ndarray) -> None:
        """Called from MicBeatDetector's fan-out with each audio chunk (float32 mono)."""
        if not self._running or self._paused:
            return
        with self._buffer_lock:
            self._buffer.append(chunk.copy())
            total = sum(len(c) for c in self._buffer)
            while len(self._buffer) > 1 and total - len(self._buffer[0]) >= _BUFFER_SAMPLES:
                total -= len(self._buffer.popleft())

    def start(self) -> None:
        if self._running:
            return
        self._running = True
        self._paused = False
        self._schedule_analysis()

    def stop(self) -> None:
        self._running = False
        self._paused = False
        self._cancel_timer()
        with self._buffer_lock:
            self._buffer.clear()
        self._fingerprint_history.clear()
        self._drop_events.clear()
        self._drop_last_beat = None
        self._drop_period = None
        self._last_novelty = 0.0
        self._smoothed_novelty = 0.0

    def pause(self) -> None:
        self._paused = True
        self._cancel_timer()

    def resume(self) -> None:
        if not self._running:
            return
        self._paused = False
        self._schedule_analysis()

    # ── Internals ──────────────────────────────────────────────────────────────

    def _schedule_analysis(self) -> None:
        self._timer = threading.Timer(ANALYSIS_INTERVAL_SECS, self._analyse)
        self._timer.daemon = True
        self._timer.start()

    def _cancel_timer(self) -> None:
        if self._timer is not None:
            self._timer.cancel()
            self._timer = None

    def _analyse(self) -> None:
        if not self._running or self._paused:
            return

        with self._buffer_lock:
            if not self._buffer:
                self._schedule_analysis()
                return
            audio = np.concatenate(list(self._buffer))

        fingerprint = _compute_fingerprint(audio)
        if fingerprint is None:
            if self._running and not self._paused:
                self._schedule_analysis()
            return

        # Cosine distance vs COMPARISON_OFFSET steps ago (≈1 s)
        self._fingerprint_history.append(fingerprint)
        novelty = 0.0
        if len(self._fingerprint_history) > COMPARISON_OFFSET:
            past = self._fingerprint_history[-1 - COMPARISON_OFFSET]
            novelty = max(0.0, min(1.0, float(1.0 - np.dot(fingerprint, past))))
        self._last_novelty = novelty
        self._smoothed_novelty = (
            NOVELTY_EMA_ALPHA * novelty
            + (1.0 - NOVELTY_EMA_ALPHA) * self._smoothed_novelty
        )

        current_beat = self._link_clock.beat

        # Detect drops: novelty spike with cooldown
        detected_tier: str | None = None
        if self._smoothed_novelty > NOVELTY_THRESHOLD:
            beats_since_drop = (
                current_beat - self._drop_last_beat
                if self._drop_last_beat is not None
                else float("inf")
            )
            if beats_since_drop > DROP_MIN_SPACING_BEATS:
                event = ChangeEvent(beat=current_beat, novelty=self._last_novelty)
                self._drop_events.append(event)
                if len(self._drop_events) > MAX_CHANGE_HISTORY:
                    self._drop_events.pop(0)
                self._drop_last_beat = current_beat
                detected_tier = "drop"

                # Re-infer drop period from accumulated events
                drop_beats = [e.beat for e in self._drop_events]
                period, _ = _infer_period(drop_beats)
                if period is not None:
                    self._drop_period = period

        # Derive per-tier periods from the inferred drop period
        tier_periods = _derive_tier_periods(self._drop_period)

        # Drop confidence from period inference
        drop_beats_list = [e.beat for e in self._drop_events]
        _, drop_confidence = _infer_period(drop_beats_list)

        # Build per-tier estimates (all anchored to last drop beat)
        tiers: dict[str, TierEstimate] = {}
        for tier_name in _TIER_NAMES:
            period = tier_periods[tier_name]
            beats_since = (
                (current_beat - self._drop_last_beat) % period
                if (period and self._drop_last_beat is not None)
                else 0.0
            )
            btc = (
                float(period - beats_since)
                if (period and self._drop_last_beat is not None)
                else None
            )
            tiers[tier_name] = TierEstimate(
                change_period=period,
                beats_to_change=btc,
                beats_since_change=beats_since,
                confidence=drop_confidence,
                next_change_beat=(current_beat + btc) if btc is not None else None,
            )

        # Most imminent tier (smallest known beats_to_change)
        imminent_pairs = [(n, te) for n, te in tiers.items() if te.beats_to_change is not None]
        most_imminent_tier: str | None = None
        if imminent_pairs:
            most_imminent_tier = min(imminent_pairs, key=lambda x: x[1].beats_to_change)[0]

        # Flat fields from most imminent tier for backwards compatibility
        if most_imminent_tier is not None:
            imminent = tiers[most_imminent_tier]
            beats_to_change    = imminent.beats_to_change
            confidence         = imminent.confidence
            next_change_beat   = imminent.next_change_beat
            beats_since_change = imminent.beats_since_change
            change_period      = imminent.change_period
        else:
            beats_to_change    = None
            confidence         = 0.0
            next_change_beat   = None
            beats_since_change = 0.0
            change_period      = None

        estimate = StructureEstimate(
            beats_to_change=beats_to_change,
            change_period=change_period,
            beats_since_change=beats_since_change,
            novelty=self._smoothed_novelty,
            confidence=confidence,
            next_change_beat=next_change_beat,
            detected_tier=detected_tier,
            most_imminent_tier=most_imminent_tier,
            tiers=tiers,
        )
        with self._estimate_lock:
            self._last_estimate = estimate

        if self._callback is not None:
            self._callback(estimate)

        if self._running and not self._paused:
            self._schedule_analysis()
