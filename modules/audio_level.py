"""Real-time microphone amplitude tracker (RMS, auto-normalized)."""
from __future__ import annotations

import threading

import numpy as np
import sounddevice as sd

SAMPLERATE = 44100
HOP_SIZE = 512
SMOOTHING = 0.80          # exponential smoothing: higher = slower response
MAX_DECAY = 0.9998        # per-sample decay of observed peak (auto-gain)
MIN_PEAK = 0.01           # floor to avoid div-by-zero on silence


class AudioLevel:
    """
    Keeps a continuously updated `level` property in 0.0–1.0.
    Uses dynamic peak normalization so it self-calibrates to room volume.
    Thread-safe; call start() once, pause()/resume() around PTT recording.
    """

    def __init__(self) -> None:
        self._level: float = 0.0
        self._peak: float = MIN_PEAK
        self._stream: sd.InputStream | None = None
        self._lock = threading.Lock()

    # ── public API ─────────────────────────────────────────────────────────────

    @property
    def level(self) -> float:
        with self._lock:
            return self._level

    def start(self) -> None:
        if self._stream is not None:
            return
        self._stream = sd.InputStream(
            samplerate=SAMPLERATE,
            channels=1,
            dtype="float32",
            blocksize=HOP_SIZE,
            callback=self._callback,
        )
        self._stream.start()

    def stop(self) -> None:
        if self._stream is not None:
            try:
                self._stream.stop()
                self._stream.close()
            except Exception:
                pass
            self._stream = None
        with self._lock:
            self._level = 0.0

    # aliases used by tui.py for symmetry with MicBeatDetector
    pause = stop

    def resume(self) -> None:
        self.start()

    # ── internal ───────────────────────────────────────────────────────────────

    def _callback(self, indata: np.ndarray, frames: int, time, status) -> None:
        rms = float(np.sqrt(np.mean(indata[:, 0] ** 2)))
        with self._lock:
            # Slowly decay peak estimate so it re-calibrates after quiet periods
            self._peak = max(MIN_PEAK, self._peak * (MAX_DECAY ** frames), rms)
            normalized = min(1.0, rms / self._peak)
            self._level = SMOOTHING * self._level + (1.0 - SMOOTHING) * normalized
