"""Real-time BPM detection from microphone using energy autocorrelation."""
from __future__ import annotations

import threading
from collections import deque
from typing import Callable

import numpy as np
import sounddevice as sd

SAMPLERATE = 44100
HOP_SIZE = 512         # ~11.6 ms per hop at 44.1 kHz
BUFFER_SECS = 20       # seconds of audio kept for analysis
ANALYSIS_INTERVAL = 1  # re-estimate BPM every N seconds
BPM_MIN = 60.0
BPM_MAX = 200.0
BPM_OCTAVE_TOLERANCE = 0.08  # fraction: if new ≈ prev/2 or prev*2, correct it


def _estimate_bpm(audio: np.ndarray, samplerate: int = SAMPLERATE, hop_size: int = HOP_SIZE) -> float | None:
    """Estimate BPM from a mono float32 audio array via onset-strength autocorrelation."""
    n_hops = len(audio) // hop_size
    if n_hops < 8:
        return None

    # Energy per hop (emphasise low-mids via simple diff == high-pass energy)
    frames = audio[:n_hops * hop_size].reshape(n_hops, hop_size)
    energy = (frames.astype(np.float64) ** 2).mean(axis=1)

    # Onset strength: positive energy rises
    onset = np.maximum(0.0, np.diff(energy))
    if onset.sum() == 0:
        return None

    # Normalise
    onset /= onset.max()

    # Autocorrelation of onset envelope
    ac = np.correlate(onset, onset, mode="full")[len(onset) - 1:]

    fps = samplerate / hop_size  # hops per second
    lo = max(1, int(fps * 60.0 / BPM_MAX))
    hi = int(fps * 60.0 / BPM_MIN)
    if hi >= len(ac):
        hi = len(ac) - 1
    if lo >= hi:
        return None

    peak = lo + int(np.argmax(ac[lo : hi + 1]))
    bpm = 60.0 * fps / peak
    return float(np.clip(bpm, BPM_MIN, BPM_MAX))


class MicBeatDetector:
    """
    Listens to the default input device and periodically calls `on_bpm`
    with a refined BPM estimate.  Designed to run alongside PTT recording;
    call pause() / resume() to avoid conflicting sounddevice streams.

    Additional consumers (e.g. DropEstimator) can subscribe to raw audio
    chunks via add_chunk_subscriber() — each chunk is forwarded synchronously
    in the sounddevice callback thread.
    """

    def __init__(self, on_bpm: Callable[[float], None]) -> None:
        self._on_bpm = on_bpm
        self._buffer: deque[np.ndarray] = deque()
        self._buffer_samples = SAMPLERATE * BUFFER_SECS
        self._stream: sd.InputStream | None = None
        self._analysis_lock = threading.Lock()
        self._timer: threading.Timer | None = None
        self._running = False
        self._chunk_subscribers: list[Callable[[np.ndarray], None]] = []
        self._prev_bpm: float | None = None

    # ── public API ────────────────────────────────────────────────────────────

    def add_chunk_subscriber(self, cb: Callable[[np.ndarray], None]) -> None:
        """Register a callback to receive every raw audio chunk (float32 mono).

        Called from the sounddevice callback thread — keep cb fast or hand off
        to another thread/buffer internally.
        """
        self._chunk_subscribers.append(cb)

    def start(self) -> None:
        if self._running:
            return
        self._running = True
        self._open_stream()
        self._schedule_analysis()

    def stop(self) -> None:
        self._running = False
        self._cancel_timer()
        self._close_stream()
        self._buffer.clear()
        self._prev_bpm = None

    def pause(self) -> None:
        """Temporarily close the mic stream (e.g. during PTT recording)."""
        self._cancel_timer()
        self._close_stream()

    def resume(self) -> None:
        """Re-open stream after pause."""
        if not self._running:
            return
        self._open_stream()
        self._schedule_analysis()

    # ── internals ─────────────────────────────────────────────────────────────

    def _open_stream(self) -> None:
        if self._stream is not None:
            return
        self._stream = sd.InputStream(
            samplerate=SAMPLERATE,
            channels=1,
            dtype="float32",
            blocksize=HOP_SIZE,
            callback=self._audio_callback,
        )
        self._stream.start()

    def _close_stream(self) -> None:
        if self._stream is not None:
            try:
                self._stream.stop()
                self._stream.close()
            except Exception:
                pass
            self._stream = None

    def _audio_callback(self, indata: np.ndarray, frames: int, time, status) -> None:
        chunk = indata[:, 0].copy()
        with self._analysis_lock:
            self._buffer.append(chunk)
            # Trim to rolling window
            total = sum(len(c) for c in self._buffer)
            while total - len(self._buffer[0]) >= self._buffer_samples:
                total -= len(self._buffer.popleft())
        for cb in self._chunk_subscribers:
            cb(chunk)

    def _schedule_analysis(self) -> None:
        self._timer = threading.Timer(ANALYSIS_INTERVAL, self._analyse)
        self._timer.daemon = True
        self._timer.start()

    def _cancel_timer(self) -> None:
        if self._timer is not None:
            self._timer.cancel()
            self._timer = None

    def _analyse(self) -> None:
        if not self._running:
            return
        with self._analysis_lock:
            if not self._buffer:
                self._schedule_analysis()
                return
            audio = np.concatenate(list(self._buffer))

        bpm = _estimate_bpm(audio)
        if bpm is not None:
            if self._prev_bpm is not None:
                ratio = bpm / self._prev_bpm
                if abs(ratio - 0.5) < BPM_OCTAVE_TOLERANCE:
                    bpm *= 2.0   # was half — double back
                elif abs(ratio - 2.0) < BPM_OCTAVE_TOLERANCE:
                    bpm *= 0.5   # was double — halve back
            self._prev_bpm = bpm
            self._on_bpm(bpm)

        if self._running:
            self._schedule_analysis()
