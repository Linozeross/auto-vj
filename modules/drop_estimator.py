"""Real-time song structure estimation via spectral novelty.

Algorithm:
  Every ANALYSIS_INTERVAL_SECS, compute a mel-spectrogram fingerprint of the
  last FINGERPRINT_WINDOW_SECS of audio.  The novelty score is the cosine
  distance between the current fingerprint and one COMPARISON_OFFSET steps ago.
  A novelty spike above NOVELTY_THRESHOLD → a structural change just occurred.

  Each detected change is classified into one of three tiers based on its
  novelty magnitude relative to the song's own history:
      minor — fills, subtle variations (bottom 50% of event novelty)
      major — section transitions, build-ups (50th–80th percentile)
      drop  — climactic moments (top 20%)

  Each tier maintains its own change event history and runs _infer_period
  independently, producing separate countdowns:
      tier.beats_to_change = tier_period - (current_beat - last_tier_beat) % tier_period

  StructureEstimate.tiers contains all three TierEstimate objects.
  Flat fields (beats_to_change, confidence, next_change_beat) are populated
  from the most imminent tier for backwards compatibility.
"""
from __future__ import annotations

import threading
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable

import librosa
import numpy as np

# ── Constants ──────────────────────────────────────────────────────────────────

SAMPLERATE = 44100               # must match beat_detector.py
HOP_SIZE = 512                   # ~11.6 ms per hop
N_MELS = 64                      # mel bands for spectral fingerprint
FINGERPRINT_WINDOW_SECS = 1.5    # audio window used per fingerprint (shorter → faster divergence)
ANALYSIS_INTERVAL_SECS = 0.25    # analysis timer interval
FINGERPRINT_HISTORY_LEN = 16     # fingerprints kept in history deque
COMPARISON_OFFSET = 4            # compare current fingerprint vs N steps ago (≈1 s)

NOVELTY_EMA_ALPHA = 0.35         # EMA weight for new novelty values (lower = smoother)
NOVELTY_THRESHOLD = 0.10         # smoothed novelty threshold → change detected
NOVELTY_DISPLAY_MAX = NOVELTY_THRESHOLD * 2  # novelty value that fills the bar
MIN_CHANGE_SPACING_BEATS = 6     # cooldown: ignore changes closer than this (must be < smallest phrase candidate)
PHRASE_CANDIDATES = [8, 16, 32, 64]  # valid EDM phrase lengths in beats
MAX_CHANGE_HISTORY = 8           # change events per tier used for period inference
CONFIDENCE_REQUIRED_RATIO = 0.6  # fraction of intervals that must agree

NOVELTY_HISTORY_MAX_EVENTS = 200  # long-term change event history (~256 bars)
MIN_EVENTS_FOR_ADAPTIVE    = 2    # need this many events before adaptive threshold kicks in
MAJOR_NOVELTY_PERCENTILE   = 50   # events above 50th percentile = major or drop
DROP_NOVELTY_PERCENTILE    = 80   # events above 80th percentile = drop

_BUFFER_SAMPLES = int(SAMPLERATE * FINGERPRINT_WINDOW_SECS)

_TIER_NAMES = ("minor", "major", "drop")


# ── Data structures ────────────────────────────────────────────────────────────

class ChangeType(str, Enum):
    MINOR = "minor"
    MAJOR = "major"
    DROP  = "drop"


@dataclass
class ChangeEvent:
    beat: float     # Link beat position at detection
    novelty: float  # raw novelty at detection


@dataclass
class TierEstimate:
    change_period: int | None         # inferred phrase period for this tier
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
    detected_tier: str | None = None        # tier of change just detected this cycle
    most_imminent_tier: str | None = None   # tier of soonest known upcoming change
    tiers: dict = field(default_factory=dict)  # dict[str, TierEstimate]


# ── Helpers ────────────────────────────────────────────────────────────────────

def _compute_fingerprint(audio: np.ndarray) -> np.ndarray | None:
    """Log-mel spectrogram → time-averaged → L2-normalised 64-float vector."""
    if len(audio) < HOP_SIZE * 4:
        return None
    mel = librosa.feature.melspectrogram(
        y=audio, sr=SAMPLERATE, n_mels=N_MELS, hop_length=HOP_SIZE
    )
    mel_db = librosa.power_to_db(mel, ref=np.max)  # log scale: amplifies subtle differences
    vec = mel_db.mean(axis=1).astype(np.float64)
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


def _classify_event(novelty: float, all_events: list[ChangeEvent]) -> ChangeType:
    """Classify a novelty spike into minor / major / drop based on the song's history.

    Uses two adaptive percentile thresholds computed from all_events.
    Falls back to fixed multiples of NOVELTY_THRESHOLD when too few events exist.
    """
    if len(all_events) < MIN_EVENTS_FOR_ADAPTIVE:
        if novelty >= NOVELTY_THRESHOLD * 3:   return ChangeType.DROP
        if novelty >= NOVELTY_THRESHOLD * 1.5: return ChangeType.MAJOR
        return ChangeType.MINOR
    scores = [e.novelty for e in all_events]
    drop_thr  = float(np.percentile(scores, DROP_NOVELTY_PERCENTILE))
    major_thr = float(np.percentile(scores, MAJOR_NOVELTY_PERCENTILE))
    if novelty >= drop_thr:  return ChangeType.DROP
    if novelty >= major_thr: return ChangeType.MAJOR
    return ChangeType.MINOR


# ── Main class ─────────────────────────────────────────────────────────────────

class DropEstimator:
    """
    Consumes raw audio chunks (via feed_audio from MicBeatDetector's fan-out),
    computes spectral novelty, detects structural changes, classifies them into
    three tiers (minor / major / drop), infers an independent phrase period per
    tier, and fires a callback with a StructureEstimate every analysis cycle.

    Lifecycle mirrors MicBeatDetector: start() / stop() / pause() / resume().
    """

    def __init__(self, link_clock) -> None:
        self._link_clock = link_clock
        self._callback: Callable[[StructureEstimate], None] | None = None

        # Rolling audio buffer
        self._buffer: deque[np.ndarray] = deque()
        self._buffer_lock = threading.Lock()

        # Fingerprint history (L2-normalised mel vectors)
        self._fingerprint_history: deque[np.ndarray] = deque(
            maxlen=FINGERPRINT_HISTORY_LEN
        )

        # Change event tracking — long-term (all tiers combined, for adaptive threshold)
        self._change_events: list[ChangeEvent] = []
        self._last_change_beat: float | None = None  # cooldown guard

        # Per-tier tracking
        self._tier_events: dict[str, list[ChangeEvent]] = {n: [] for n in _TIER_NAMES}
        self._tier_last_beat: dict[str, float | None]   = {n: None for n in _TIER_NAMES}
        self._tier_period: dict[str, int | None]         = {n: None for n in _TIER_NAMES}

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
        self._change_events.clear()
        self._last_change_beat = None
        for name in _TIER_NAMES:
            self._tier_events[name].clear()
            self._tier_last_beat[name] = None
            self._tier_period[name] = None
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

        # Compute novelty vs fingerprint COMPARISON_OFFSET steps ago
        novelty = 0.0
        self._fingerprint_history.append(fingerprint)
        if len(self._fingerprint_history) > COMPARISON_OFFSET:
            past = self._fingerprint_history[-1 - COMPARISON_OFFSET]
            raw = float(1.0 - np.dot(fingerprint, past))
            novelty = max(0.0, min(1.0, raw))
        self._last_novelty = novelty
        self._smoothed_novelty = (
            NOVELTY_EMA_ALPHA * novelty
            + (1.0 - NOVELTY_EMA_ALPHA) * self._smoothed_novelty
        )

        current_beat = self._link_clock.beat

        # Detect and classify change events
        beats_since_last = (
            current_beat - self._last_change_beat
            if self._last_change_beat is not None
            else float("inf")
        )
        detected_tier: str | None = None
        if self._smoothed_novelty > NOVELTY_THRESHOLD and beats_since_last > MIN_CHANGE_SPACING_BEATS:
            self._last_change_beat = current_beat
            event = ChangeEvent(beat=current_beat, novelty=self._last_novelty)
            self._change_events.append(event)
            if len(self._change_events) > NOVELTY_HISTORY_MAX_EVENTS:
                self._change_events.pop(0)

            tier = _classify_event(event.novelty, self._change_events)
            detected_tier = tier.value
            self._tier_last_beat[tier.value] = current_beat
            self._tier_events[tier.value].append(event)
            if len(self._tier_events[tier.value]) > MAX_CHANGE_HISTORY:
                self._tier_events[tier.value].pop(0)

            # Infer period independently per tier
            tier_beats = [e.beat for e in self._tier_events[tier.value]]
            period, _ = _infer_period(tier_beats)
            if period is not None:
                self._tier_period[tier.value] = period

        # Build per-tier estimates
        tiers: dict[str, TierEstimate] = {}
        for tier_name in _TIER_NAMES:
            last_beat = self._tier_last_beat[tier_name]
            period    = self._tier_period[tier_name]
            tier_beats_list = [e.beat for e in self._tier_events[tier_name]]
            _, tier_confidence = _infer_period(tier_beats_list)
            beats_since = (current_beat - last_beat) if last_beat is not None else 0.0
            btc = float(period - beats_since % period) if (period and last_beat is not None) else None
            tiers[tier_name] = TierEstimate(
                change_period=period,
                beats_to_change=btc,
                beats_since_change=beats_since,
                confidence=tier_confidence,
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
