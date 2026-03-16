"""Real-time microphone amplitude tracker (RMS, auto-normalized)."""
from __future__ import annotations

import os
import threading

import numpy as np
import sounddevice as sd
from dotenv import load_dotenv

load_dotenv()

AUDIO_DEVICE_NAME = os.getenv("AUDIO_DEVICE", "").strip()


def _resolve_device(name: str) -> int | None:
    """Return the first device index whose name contains *name* (case-insensitive).

    Returns None when *name* is empty (use sounddevice default).
    Raises RuntimeError if a name was given but no matching device was found.
    """
    if not name:
        return None
    for i, dev in enumerate(sd.query_devices()):
        if name.lower() in dev["name"].lower() and dev["max_input_channels"] > 0:
            return i
    raise RuntimeError(
        f"AUDIO_DEVICE '{name}' not found. "
        "Run 'python -c \"import sounddevice as sd; [print(i, d) for i, d in enumerate(sd.query_devices())]\"' "
        "to list available devices."
    )


AUDIO_DEVICE_ID: int | None = _resolve_device(AUDIO_DEVICE_NAME)

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
            device=AUDIO_DEVICE_ID,
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
