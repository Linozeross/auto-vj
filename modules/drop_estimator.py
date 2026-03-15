"""Real-time song structure estimation via spectral novelty.

Algorithm:
  Every ANALYSIS_INTERVAL_SECS, compute a mel-spectrogram fingerprint of the
  last FINGERPRINT_WINDOW_SECS of audio.  The novelty score is the cosine
  distance between the current fingerprint and one COMPARISON_OFFSET steps ago.
  A novelty spike above NOVELTY_THRESHOLD → a structural change just occurred.

  Change events are accumulated and their inter-beat intervals are snapped to
  the nearest EDM phrase length (16 / 32 / 64 beats) to infer the change period.
  The countdown to the next change is then:
      beats_to_change = change_period - (current_beat - last_change_beat) % change_period

Upgrade path:
  The `StructureEstimate.novelty` field exposes the raw signal so a future
  classifier can label change *type* (drop / breakdown / verse) without
  touching the detection logic.
"""
from __future__ import annotations

import threading
from collections import deque
from dataclasses import dataclass
from typing import Callable

import librosa
import numpy as np

# ── Constants ──────────────────────────────────────────────────────────────────

SAMPLERATE = 44100               # must match beat_detector.py
HOP_SIZE = 512                   # ~11.6 ms per hop
N_MELS = 64                      # mel bands for spectral fingerprint
FINGERPRINT_WINDOW_SECS = 3.0    # audio window used per fingerprint
ANALYSIS_INTERVAL_SECS = 0.25     # analysis timer interval
FINGERPRINT_HISTORY_LEN = 16     # fingerprints kept in history deque
COMPARISON_OFFSET = 4            # compare current fingerprint vs N steps ago (≈4 s)

NOVELTY_EMA_ALPHA = 0.35         # EMA weight for new novelty values (lower = smoother)
NOVELTY_THRESHOLD = 0.15         # smoothed novelty threshold → change detected
NOVELTY_DISPLAY_MAX = NOVELTY_THRESHOLD * 2  # novelty value that fills the bar (0.30)
MIN_CHANGE_SPACING_BEATS = 8     # cooldown: ignore changes closer than this
PHRASE_CANDIDATES = [16, 32, 64] # valid EDM phrase lengths in beats
MAX_CHANGE_HISTORY = 8           # change events used for period inference
CONFIDENCE_REQUIRED_RATIO = 0.6  # fraction of intervals that must agree

_BUFFER_SAMPLES = int(SAMPLERATE * FINGERPRINT_WINDOW_SECS)


# ── Data structures ────────────────────────────────────────────────────────────

@dataclass
class StructureEstimate:
    beats_to_change: float | None  # countdown to next change; None = not established
    change_period: int | None      # inferred phrase period (16/32/64); None = unknown
    beats_since_change: float      # beats since last detected change
    novelty: float                 # current novelty score 0.0–1.0
    confidence: float              # 0.0–1.0 (consistency of detected changes)


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


# ── Main class ─────────────────────────────────────────────────────────────────

class DropEstimator:
    """
    Consumes raw audio chunks (via feed_audio from MicBeatDetector's fan-out),
    computes spectral novelty, detects structural changes, infers the phrase
    period, and fires a callback with a StructureEstimate every analysis cycle.

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

        # Change event tracking
        self._change_beats: list[float] = []
        self._last_change_beat: float | None = None
        self._change_period: int | None = None
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
        self._change_beats.clear()
        self._last_change_beat = None
        self._change_period = None
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

        # Detect change event using smoothed novelty to avoid false triggers
        beats_since_last = (
            current_beat - self._last_change_beat
            if self._last_change_beat is not None
            else float("inf")
        )
        if self._smoothed_novelty > NOVELTY_THRESHOLD and beats_since_last > MIN_CHANGE_SPACING_BEATS:
            self._last_change_beat = current_beat
            self._change_beats.append(current_beat)
            if len(self._change_beats) > MAX_CHANGE_HISTORY:
                self._change_beats.pop(0)
            period, _ = _infer_period(self._change_beats)
            if period is not None:
                self._change_period = period

        # Recompute beats_since_change and countdown
        beats_since_change = (
            current_beat - self._last_change_beat
            if self._last_change_beat is not None
            else 0.0
        )

        beats_to_change: float | None = None
        if self._change_period is not None and self._last_change_beat is not None:
            remainder = beats_since_change % self._change_period
            beats_to_change = float(self._change_period - remainder)

        _, confidence = _infer_period(self._change_beats)

        estimate = StructureEstimate(
            beats_to_change=beats_to_change,
            change_period=self._change_period,
            beats_since_change=beats_since_change,
            novelty=self._smoothed_novelty,
            confidence=confidence,
        )
        with self._estimate_lock:
            self._last_estimate = estimate

        if self._callback is not None:
            self._callback(estimate)

        if self._running and not self._paused:
            self._schedule_analysis()
