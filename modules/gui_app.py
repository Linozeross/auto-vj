"""
VJ GUI — PyQt6 miniature control surface.

Layout:
  ┌ status bar ── state pill · BPM · Link ──────────────────┐
  │ effect display ─ name · params · filters ───────────────│
  ├ CONTROLS ──── Speed · Brightness · BPM knobs ───────────┤
  ├ FAVORITES ─── 3×3 preset grid (keys 1-9) ───────────────┤
  │                               [★ Star current effect]   │
  └─────────────────────────────────────────────────────────┘
"""

from __future__ import annotations

import asyncio
import copy
import json
import os
import platform
import random
import sys
import threading
import time
import traceback
from pathlib import Path

from PyQt6.QtCore import Qt, QThread, pyqtSignal, QTimer
from PyQt6.QtGui import QKeyEvent, QAction, QPainter, QColor, QPen, QFont
from PyQt6.QtWidgets import (
    QApplication,
    QDial,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMenu,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QVBoxLayout,
    QWidget,
)
from dotenv import load_dotenv
from openai import OpenAI, BadRequestError as OpenAIBadRequestError
from pydantic import ValidationError

import numpy as np
import sounddevice as sd

from modules.artnet_renderer import ArtNetRenderer, MultiRenderer
from modules.beat_detector import MicBeatDetector
from modules.bpm import BpmMode, LinkClock
from modules.drop_estimator import DropEstimator, StructureEstimate, NOVELTY_THRESHOLD, NOVELTY_DISPLAY_MAX
from modules.effects import VJResponse
from modules.recorder import SAMPLE_RATE, CHANNELS, audio_to_wav_b64
from modules.sequences import Sequence, sequence_from_dict, sequence_from_effect_cmd, PHRASE_BEATS, BEATS_PER_BAR

load_dotenv()

# ── Constants ──────────────────────────────────────────────────────────────────

ARTNET_IP = os.environ.get("ARTNET_IP", "192.168.178.102")
LED_COUNT = int(os.environ.get("LED_COUNT", "100"))
ARTNET_NODES_RAW = os.environ.get("ARTNET_NODES", "")

PRESETS_FILE = Path("presets.json")
NUM_PRESETS = 9

SPEED_MIN = 0.1
SPEED_MAX = 4.0
BRIGHTNESS_MIN = 0.05
BRIGHTNESS_MAX = 1.0
BPM_GUI_MIN = 20.0
BPM_GUI_MAX = 300.0
KNOB_STEPS = 1000

TAP_RESET_GAP_SECS = 2.0

QUEUE_WIDGET_HEIGHT = 52
QUEUE_ITEM_WIDTH = 72
QUEUE_LOOP_GAP = 2

SYSTEM_PROMPT = open("system_prompt.txt").read().strip()

AUTO_TRIGGER_AHEAD_BARS = 4      # start generating this many bars before queue runs out
STRUCTURE_CONFIDENCE_MIN = 0.5   # minimum drop-estimator confidence to use for transition timing
STRUCTURE_BEATS_WINDOW = 1.0     # advance queue when beats_to_change falls within this many beats
DROP_MARKER_COLOR = "#ffcc00"    # colour of the predicted-drop marker in the queue widget
AUTO_PROMPTS = [
    "keep the energy going with something fresh",
    "mix it up — new effect, different vibe",
    "evolve the mood, keep the crowd moving",
    "surprise me with something unexpected but fitting",
    "change the scene, same energy level",
]

# Effects that use "rate" instead of "speed"
RATE_EFFECTS = {"pulse"}

NOVELTY_COLORS: dict[str, str] = {
    "active":  "#00e5ff",   # period established, countdown running
    "pending": "#1a3a4a",   # no period yet
    "hot":     "#ffcc00",   # novelty spike (change just detected)
}

STATE_COLORS: dict[str, str] = {
    "WAITING":      "#1a3a4a",
    "RECORDING":    "#ff2200",
    "TRANSCRIBING": "#ff9900",
    "THINKING":     "#0088ff",
    "LIVE":         "#00ff88",
}

EFFECT_COLORS: dict[str, str] = {
    "pulse":        "#ff00aa",
    "rainbow":      "#ffee00",
    "chase":        "#00ff44",
    "twinkle":      "#00eeff",
    "plasma":       "#cc00ff",
    "palette_wave": "#ff6600",
    "beat_pulse":   "#ffcc00",
}

# Qt QSS accepts only a single font-family value (no comma fallback lists).
_MONO_FONT = "Menlo" if platform.system() == "Darwin" else "Consolas"

def _build_qss(font: str) -> str:
    return (
        f'QMainWindow, QWidget {{ background-color: #030a0e; color: #88ccdd;'
        f' font-family: "{font}"; font-size: 12px; }}\n'
        'QGroupBox { border: 1px solid #0d3344; border-radius: 0px; margin-top: 12px;'
        ' color: #00e5ff; font-size: 10px; font-weight: bold; padding: 4px;'
        ' letter-spacing: 2px; }\n'
        'QGroupBox::title { subcontrol-origin: margin; left: 6px; padding: 0 6px;'
        ' background-color: #030a0e; }\n'
        'QLabel#state-pill { border-radius: 0px; padding: 2px 10px;'
        ' font-size: 11px; font-weight: bold; letter-spacing: 3px; }\n'
        'QPushButton#preset-btn { background-color: #050d14; color: #0d3344;'
        ' border: 1px solid #0a1e2a; border-radius: 0px; padding: 4px 2px;'
        ' font-size: 11px; min-height: 48px; text-align: center; letter-spacing: 1px; }\n'
        'QPushButton#preset-btn[filled="true"] { color: #88ccdd; border-color: #0d3344; }\n'
        'QPushButton#preset-btn:hover { border-color: #005566; }\n'
        'QPushButton#preset-btn:pressed { background-color: #071520; }\n'
        'QPushButton#star-btn { background-color: #050d14; color: #2a4455;'
        ' border: 1px solid #0a1e2a; border-radius: 0px; padding: 4px 14px;'
        ' letter-spacing: 1px; }\n'
        'QPushButton#star-btn:hover { border-color: #ff9900; color: #ff9900; }\n'
    )

DARK_QSS = _build_qss(_MONO_FONT)


# ── Helpers ────────────────────────────────────────────────────────────────────

def _parse_artnet_nodes(raw: str) -> list[tuple[str, int]]:
    if raw.strip():
        nodes = []
        for part in raw.split(","):
            part = part.strip()
            if ":" in part:
                ip, count = part.rsplit(":", 1)
                nodes.append((ip.strip(), int(count)))
            else:
                nodes.append((part, LED_COUNT))
        return nodes
    return [(ARTNET_IP, LED_COUNT)]


def _fmt_effect(cmd: dict) -> str:
    name = cmd.get("effect", "?")
    params = cmd.get("params", {})
    filters = cmd.get("filters", [])
    p = "  ".join(f"{k}={v}" for k, v in params.items()) if params else "—"
    f = ("  [" + "+".join(f["type"] for f in filters) + "]") if filters else ""
    return f"{name}  {p}{f}"


def _apply_speed(cmd: dict, speed: float) -> dict:
    """Return a copy of cmd with speed (or rate for pulse) updated."""
    cmd = copy.deepcopy(cmd)
    key = "rate" if cmd.get("effect") in RATE_EFFECTS else "speed"
    cmd.setdefault("params", {})[key] = round(speed, 3)
    return cmd


def _apply_brightness(cmd: dict, brightness: float) -> dict:
    """Return a copy of cmd with a dim filter added/updated."""
    cmd = copy.deepcopy(cmd)
    filters = cmd.setdefault("filters", [])
    for f in filters:
        if f["type"] == "dim":
            f["params"]["brightness"] = round(brightness, 3)
            return cmd
    filters.append({"type": "dim", "params": {"brightness": round(brightness, 3)}})
    return cmd


def _record_gui(stop_event: threading.Event,
                on_start: "Callable[[], None] | None" = None) -> np.ndarray:
    """Record from mic until stop_event is set. No pynput — controlled by Qt key events."""
    if on_start:
        on_start()
    frames: list[np.ndarray] = []
    with sd.InputStream(samplerate=SAMPLE_RATE, channels=CHANNELS, dtype="int16") as stream:
        while not stop_event.is_set():
            data, _ = stream.read(1024)
            frames.append(data.copy())
    if not frames:
        return np.zeros(0, dtype=np.int16)
    return np.concatenate(frames, axis=0)


def _strip_json_fences(raw: str) -> str:
    """Strip markdown code fences that some models wrap around JSON."""
    raw = raw.strip()
    if raw.startswith("```"):
        raw = raw.split("\n", 1)[-1]  # drop opening fence line
        raw = raw.rsplit("```", 1)[0]  # drop closing fence
    return raw.strip()


def _text_to_response(text: str, history: list[dict], client: OpenAI) -> tuple[VJResponse, list[dict]]:
    """Send a text prompt (no audio) to GPT and return a VJResponse."""
    messages = [{"role": "system", "content": SYSTEM_PROMPT}] + history + [
        {"role": "user", "content": text},
    ]
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
    )
    raw = _strip_json_fences(response.choices[0].message.content)
    new_history = history + [
        {"role": "user", "content": text},
        {"role": "assistant", "content": raw},
    ]
    vj_response = VJResponse.model_validate(json.loads(raw))
    return vj_response, new_history


def _chat_to_response(audio: np.ndarray, history: list[dict], client: OpenAI) -> tuple[VJResponse, list[dict]]:
    b64 = audio_to_wav_b64(audio)
    messages = [{"role": "system", "content": SYSTEM_PROMPT}] + history + [
        {"role": "user", "content": [
            {"type": "input_audio", "input_audio": {"data": b64, "format": "wav"}},
        ]},
    ]
    response = client.chat.completions.create(
        model="gpt-4o-audio-preview",
        modalities=["text"],
        messages=messages,
    )
    raw = _strip_json_fences(response.choices[0].message.content)
    new_history = history + [
        {"role": "user", "content": "[voice command]"},
        {"role": "assistant", "content": raw},
    ]
    vj_response = VJResponse.model_validate(json.loads(raw))
    return vj_response, new_history


# ── Presets persistence ────────────────────────────────────────────────────────

def load_presets() -> list[dict | None]:
    if PRESETS_FILE.exists():
        try:
            data = json.loads(PRESETS_FILE.read_text())
            slots = data if isinstance(data, list) else []
            slots = (slots + [None] * NUM_PRESETS)[:NUM_PRESETS]
            return slots
        except Exception:
            pass
    return [None] * NUM_PRESETS


def save_presets(slots: list[dict | None]) -> None:
    PRESETS_FILE.write_text(json.dumps(slots, indent=2))


# ── Worker: asyncio + VJ logic in a background QThread ────────────────────────

class VJWorker(QThread):
    status_changed      = pyqtSignal(str)
    effect_activated    = pyqtSignal(str, object)   # (prompt, cmd_dict)
    bpm_changed         = pyqtSignal(float)
    link_status_changed = pyqtSignal(float, int)    # (tempo, peers)
    bpm_mode_changed    = pyqtSignal(str)
    queue_status_changed = pyqtSignal(int)           # queue_length
    worker_error        = pyqtSignal(str)            # fatal error message
    auto_mode_changed   = pyqtSignal(bool)
    structure_changed   = pyqtSignal(object)         # StructureEstimate

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._loop: asyncio.AbstractEventLoop | None = None
        self._link_clock: LinkClock | None = None
        self._renderer: MultiRenderer | None = None
        self._current_cmd: dict | None = None
        self._history: list[dict] = []
        self._bpm_mode: BpmMode = BpmMode.LINK
        self._tap_times: list[float] = []
        self._beat_detector: MicBeatDetector | None = None
        self._drop_estimator: DropEstimator | None = None
        self._client: OpenAI | None = None
        # Sequence queue state (asyncio-only, no lock needed)
        self._seq_queue: list[Sequence] = []
        self._seq_current: Sequence | None = None
        self._seq_start_beat: float = 0.0
        # PTT coordination
        self._ptt_start: asyncio.Event | None = None
        self._stop_recording: threading.Event = threading.Event()
        # Auto mode
        self._auto_mode: bool = False
        self._auto_generating: bool = False
        # Latest structure estimate (written from estimator timer thread, read from asyncio loop)
        self._last_structure: StructureEstimate | None = None

    def run(self) -> None:
        try:
            self._loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self._loop)
            self._loop.run_until_complete(self._main())
        except asyncio.CancelledError:
            pass  # clean shutdown via shutdown()
        except Exception:
            msg = traceback.format_exc()
            print(msg, file=sys.stderr)
            self.worker_error.emit(msg)

    def shutdown(self) -> None:
        """Cancel all asyncio tasks and stop the event loop. Call before wait()."""
        self._stop_recording.set()
        if self._beat_detector:
            self._beat_detector.stop()
            if self._drop_estimator:
                self._drop_estimator.stop()
        if self._loop and self._loop.is_running():
            def _cancel() -> None:
                if self._link_clock is not None:
                    self._link_clock.stop()
                for task in asyncio.all_tasks(self._loop):
                    task.cancel()
            self._loop.call_soon_threadsafe(_cancel)

    async def _main(self) -> None:
        self._client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
        self._link_clock = LinkClock()

        def _on_tempo(t: float) -> None:
            self.link_status_changed.emit(t, self._link_clock.num_peers)

        def _on_peers(n: int) -> None:
            self.link_status_changed.emit(self._link_clock.tempo, n)

        self._link_clock.set_tempo_callback(_on_tempo)
        self._link_clock.set_num_peers_callback(_on_peers)
        self._link_clock.start()

        def _on_mic_bpm(bpm: float) -> None:
            self._link_clock.set_bpm(bpm)
            self.bpm_changed.emit(bpm)

        self._beat_detector = MicBeatDetector(on_bpm=_on_mic_bpm)

        def _on_structure(estimate: StructureEstimate) -> None:
            self._last_structure = estimate  # GIL-atomic write; read by _queue_watcher
            self.structure_changed.emit(estimate)

        self._drop_estimator = DropEstimator(link_clock=self._link_clock)
        self._drop_estimator.set_callback(_on_structure)
        self._beat_detector.add_chunk_subscriber(self._drop_estimator.feed_audio)

        # Beat detector and drop estimator are started only in MIC mode to avoid
        # holding a sounddevice stream that conflicts with PTT recording.
        # (Start is triggered by cycle_bpm_mode when user switches to MIC.)

        nodes = _parse_artnet_nodes(ARTNET_NODES_RAW)
        self._renderer = MultiRenderer([
            ArtNetRenderer(ip, leds, link_clock=self._link_clock)
            for ip, leds in nodes
        ])

        await asyncio.gather(
            self._ptt_loop(),
            self._renderer.render_loop(),
            self._queue_watcher(),
        )

    # ── Sequence queue management ──────────────────────────────────────────────

    async def _enqueue_async(self, seq: Sequence) -> None:
        """Start immediately if nothing is playing, otherwise add to queue."""
        if self._seq_current is None:
            self._seq_current = seq
            self._seq_start_beat = self._link_clock.beat
            self._link_clock.attach_effect(seq)
            self._renderer.set_effect(seq)
        else:
            self._seq_queue.append(seq)
        self.queue_status_changed.emit(len(self._seq_queue))

    async def _replace_current_async(self, seq: Sequence) -> None:
        """Immediately swap the current sequence (knob tweaks — bypass phrase boundary)."""
        self._seq_current = seq
        self._seq_start_beat = self._link_clock.beat
        self._link_clock.attach_effect(seq)
        self._renderer.set_effect(seq)

    async def _queue_watcher(self) -> None:
        """Wake on sub-beat intervals, advance the queue at phrase boundaries, and trigger auto-refill."""
        while True:
            tempo = self._link_clock.tempo if self._link_clock else 120.0
            sleep_secs = 60.0 / max(tempo, 20.0) / 4.0
            await asyncio.sleep(sleep_secs)

            # Advance queue at phrase boundaries (or at a predicted structural change)
            if self._seq_current is not None and self._seq_queue:
                seq_t = self._link_clock.beat - self._seq_start_beat
                beat_number = self._link_clock._beat_number
                if self._seq_current.is_done(seq_t) and self._should_advance_queue(beat_number):
                    next_seq = self._seq_queue.pop(0)
                    self._seq_current = next_seq
                    self._seq_start_beat = self._link_clock.beat
                    self._link_clock.attach_effect(next_seq)
                    self._renderer.set_effect(next_seq)
                    self.queue_status_changed.emit(len(self._seq_queue))

            # Auto-refill: trigger when queue is empty and current is within
            # AUTO_TRIGGER_AHEAD_BARS bars of ending (accounts for GPT latency)
            if self._auto_mode and not self._auto_generating and len(self._seq_queue) == 0:
                if self._seq_current is None:
                    should_trigger = True
                else:
                    seq_t = self._link_clock.beat - self._seq_start_beat
                    remaining_beats = (
                        self._seq_current.total_beats * self._seq_current.repeats - seq_t
                    )
                    should_trigger = self._should_trigger_auto_refill(remaining_beats)
                if should_trigger:
                    self._auto_generating = True
                    asyncio.ensure_future(self._auto_queue_refill())

    def _should_advance_queue(self, beat_number: int) -> bool:
        """Return True if it is appropriate to advance the sequence queue now.

        Uses the drop estimator when confident; falls back to phrase-boundary logic.
        """
        est = self._last_structure
        if (est is not None
                and est.beats_to_change is not None
                and est.confidence >= STRUCTURE_CONFIDENCE_MIN
                and est.beats_to_change <= STRUCTURE_BEATS_WINDOW):
            return True
        return beat_number % PHRASE_BEATS == 0

    def _should_trigger_auto_refill(self, remaining_beats: float) -> bool:
        """Return True if auto-refill should start generating now."""
        if remaining_beats / BEATS_PER_BAR < AUTO_TRIGGER_AHEAD_BARS:
            return True
        est = self._last_structure
        if (est is not None
                and est.beats_to_change is not None
                and est.confidence >= STRUCTURE_CONFIDENCE_MIN
                and est.beats_to_change <= AUTO_TRIGGER_AHEAD_BARS * BEATS_PER_BAR):
            return True
        return False

    def get_current_beat(self) -> float:
        """Thread-safe read of the current Link beat position."""
        if self._link_clock is None:
            return 0.0
        return self._link_clock.beat

    async def _auto_queue_refill(self) -> None:
        """Generate a new sequence via text prompt and enqueue it."""
        try:
            prompt = random.choice(AUTO_PROMPTS)
            vj_response, self._history = await asyncio.to_thread(
                _text_to_response, prompt, self._history, self._client
            )
            if vj_response.bpm is not None:
                bpm = float(vj_response.bpm)
                self._link_clock.set_bpm(bpm)
                self.bpm_changed.emit(bpm)
            if vj_response.sequences:
                for s in vj_response.sequences:
                    await self._enqueue_async(sequence_from_dict(s.model_dump()))
                first_step = vj_response.sequences[0].steps[0]
                cmd = {"effect": first_step.effect, "params": first_step.params,
                       "filters": [f.model_dump() for f in first_step.filters]}
                self._current_cmd = cmd
                self.effect_activated.emit("[auto]", cmd)
        except Exception as exc:
            print(f"[auto-vj] auto-refill failed: {exc}", file=sys.stderr)
        finally:
            self._auto_generating = False

    def toggle_auto_mode(self) -> None:
        if self._loop:
            self._loop.call_soon_threadsafe(self._toggle_auto_mode_async)

    def _toggle_auto_mode_async(self) -> None:
        self._auto_mode = not self._auto_mode
        self.auto_mode_changed.emit(self._auto_mode)

    async def _ptt_loop(self) -> None:
        self._ptt_start = asyncio.Event()

        while True:
            self.status_changed.emit("WAITING")

            await self._ptt_start.wait()
            self._ptt_start.clear()

            self.status_changed.emit("RECORDING")
            self._stop_recording.clear()

            def _on_recording_start() -> None:
                if self._bpm_mode is BpmMode.MIC and self._beat_detector:
                    self._beat_detector.pause()
                    if self._drop_estimator:
                        self._drop_estimator.pause()

            audio = await asyncio.to_thread(_record_gui, self._stop_recording, _on_recording_start)

            if self._bpm_mode is BpmMode.MIC and self._beat_detector:
                self._beat_detector.resume()
                if self._drop_estimator:
                    self._drop_estimator.resume()

            if audio.size == 0:
                continue

            self.status_changed.emit("THINKING")
            try:
                vj_response, self._history = await asyncio.to_thread(
                    _chat_to_response, audio, self._history, self._client
                )
            except (ValidationError, ValueError, OpenAIBadRequestError) as exc:
                print(f"[auto-vj] GPT response rejected: {exc}", file=sys.stderr)
                continue

            if vj_response.bpm is not None:
                bpm = float(vj_response.bpm)
                self._link_clock.set_bpm(bpm)
                self.bpm_changed.emit(bpm)

            if vj_response.sequences:
                for s in vj_response.sequences:
                    await self._enqueue_async(sequence_from_dict(s.model_dump()))
                first_step = vj_response.sequences[0].steps[0]
                cmd = {"effect": first_step.effect, "params": first_step.params,
                       "filters": [f.model_dump() for f in first_step.filters]}
                self._current_cmd = cmd
                self.status_changed.emit("LIVE")
                self.effect_activated.emit("[voice command]", cmd)

    # ── Thread-safe calls from Qt ──────────────────────────────────────────────

    def activate_preset(self, cmd: dict) -> None:
        """Called from Qt thread — enqueue effect (phrase-aligned switch)."""
        if self._loop:
            asyncio.run_coroutine_threadsafe(
                self._activate_and_notify(cmd), self._loop
            )

    async def _activate_and_notify(self, cmd: dict) -> None:
        seq = sequence_from_effect_cmd(cmd)
        await self._enqueue_async(seq)
        self._current_cmd = cmd
        self.status_changed.emit("LIVE")
        self.effect_activated.emit("preset", cmd)
        if cmd.get("bpm") is not None:
            self._link_clock.set_bpm(float(cmd["bpm"]))
            self.bpm_changed.emit(float(cmd["bpm"]))

    def apply_speed(self, speed: float) -> None:
        """Immediate knob tweak — replaces current sequence without waiting for phrase boundary."""
        if self._current_cmd and self._loop:
            cmd = _apply_speed(self._current_cmd, speed)
            self._current_cmd = cmd
            seq = sequence_from_effect_cmd(cmd)
            asyncio.run_coroutine_threadsafe(self._replace_current_async(seq), self._loop)

    def apply_brightness(self, brightness: float) -> None:
        """Immediate knob tweak — replaces current sequence without waiting for phrase boundary."""
        if self._current_cmd and self._loop:
            cmd = _apply_brightness(self._current_cmd, brightness)
            self._current_cmd = cmd
            seq = sequence_from_effect_cmd(cmd)
            asyncio.run_coroutine_threadsafe(self._replace_current_async(seq), self._loop)

    def set_bpm(self, bpm: float) -> None:
        if self._link_clock:
            self._link_clock.set_bpm(bpm)
            self.bpm_changed.emit(bpm)

    def ptt_press(self) -> None:
        if self._loop and self._ptt_start:
            self._loop.call_soon_threadsafe(self._ptt_start.set)

    def ptt_release(self) -> None:
        self._stop_recording.set()

    def tap_beat(self) -> None:
        if self._bpm_mode is not BpmMode.TAP or not self._link_clock:
            return
        now = time.monotonic()
        if self._tap_times and now - self._tap_times[-1] > TAP_RESET_GAP_SECS:
            self._tap_times = []
        self._tap_times.append(now)
        if len(self._tap_times) < 2:
            return
        intervals = [self._tap_times[i + 1] - self._tap_times[i] for i in range(len(self._tap_times) - 1)]
        bpm = 60.0 / (sum(intervals) / len(intervals))
        self.set_bpm(bpm)

    def cycle_bpm_mode(self) -> None:
        order = [BpmMode.LINK, BpmMode.TAP, BpmMode.MIC]
        next_mode = order[(order.index(self._bpm_mode) + 1) % len(order)]

        if self._bpm_mode is BpmMode.MIC and self._beat_detector:
            self._beat_detector.stop()
            if self._drop_estimator:
                self._drop_estimator.stop()

        self._bpm_mode = next_mode
        self._tap_times = []

        if self._bpm_mode is BpmMode.MIC and self._beat_detector:
            self._beat_detector.start()
            if self._drop_estimator:
                self._drop_estimator.start()

        self.bpm_mode_changed.emit(next_mode.value)

    @property
    def current_cmd(self) -> dict | None:
        return self._current_cmd

    def get_queue_snapshot(self) -> dict:
        """Thread-safe snapshot of queue state for UI polling (GIL protects simple reads)."""
        current_name = ""
        current_effect = ""
        seq_phase = 0.0
        current_repeats = 1

        if self._seq_current is not None:
            current_name = self._seq_current.name or ""
            current_effect = current_name
            current_repeats = self._seq_current.repeats
            if self._link_clock is not None:
                seq_t = self._link_clock.beat - self._seq_start_beat
                total = self._seq_current.total_beats * self._seq_current.repeats
                seq_phase = min(seq_t / total, 1.0) if total > 0 else 0.0

        queue_items = [{"name": seq.name or "", "effect": seq.name or "", "repeats": seq.repeats} for seq in self._seq_queue]

        # Pass drop marker position to the queue widget (None if estimate not confident)
        next_change_beat = None
        est = self._last_structure
        if (est is not None
                and est.next_change_beat is not None
                and est.confidence >= STRUCTURE_CONFIDENCE_MIN):
            next_change_beat = est.next_change_beat

        seq_total_beats = (
            self._seq_current.total_beats * self._seq_current.repeats
            if self._seq_current is not None else 0
        )

        return {
            "current_name": current_name,
            "current_effect": current_effect,
            "seq_phase": seq_phase,
            "current_repeats": current_repeats,
            "queue": queue_items,
            "next_change_beat": next_change_beat,
            "seq_start_beat": self._seq_start_beat,
            "seq_total_beats": seq_total_beats,
        }


# ── KnobWidget ─────────────────────────────────────────────────────────────────

class KnobWidget(QWidget):
    value_changed = pyqtSignal(float)

    def __init__(self, label: str, min_val: float, max_val: float,
                 default: float, fmt: str = ".2f", parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._min = min_val
        self._max = max_val
        self._fmt = fmt

        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(2)

        self._dial = QDial()
        self._dial.setMinimum(0)
        self._dial.setMaximum(KNOB_STEPS)
        self._dial.setNotchesVisible(True)
        self._dial.setFixedSize(68, 68)
        self._dial.setWrapping(False)

        self._top_label = QLabel(label)
        self._top_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._top_label.setStyleSheet("color: #1a3a4a; font-size: 10px; letter-spacing: 2px;")

        self._val_label = QLabel()
        self._val_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._val_label.setStyleSheet("color: #00e5ff; font-size: 11px; font-weight: bold;")

        layout.addWidget(self._dial, alignment=Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self._top_label)
        layout.addWidget(self._val_label)

        self.set_value(default)
        self._dial.valueChanged.connect(self._on_dial)

    def _on_dial(self, raw: int) -> None:
        v = self._min + (raw / KNOB_STEPS) * (self._max - self._min)
        self._update_label(v)
        self.value_changed.emit(v)

    def _update_label(self, v: float) -> None:
        self._val_label.setText(format(v, self._fmt))

    def set_value(self, v: float) -> None:
        raw = int((v - self._min) / (self._max - self._min) * KNOB_STEPS)
        self._dial.blockSignals(True)
        self._dial.setValue(max(0, min(KNOB_STEPS, raw)))
        self._dial.blockSignals(False)
        self._update_label(v)

    def get_value(self) -> float:
        return self._min + (self._dial.value() / KNOB_STEPS) * (self._max - self._min)


# ── PresetButton ───────────────────────────────────────────────────────────────

class PresetButton(QPushButton):
    recalled    = pyqtSignal(int)
    remove_req  = pyqtSignal(int)

    def __init__(self, slot: int, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._slot = slot
        self._preset: dict | None = None
        self.setObjectName("preset-btn")
        self.setProperty("filled", "false")
        self._refresh()
        self.clicked.connect(lambda: self.recalled.emit(self._slot) if self._preset else None)

    def set_preset(self, preset: dict | None) -> None:
        self._preset = preset
        self.setProperty("filled", "true" if preset else "false")
        self.style().unpolish(self)
        self.style().polish(self)
        self._refresh()

    def _refresh(self) -> None:
        slot_num = self._slot + 1
        if self._preset:
            name = self._preset.get("label") or self._preset.get("cmd", {}).get("effect", "?")
            color = EFFECT_COLORS.get(self._preset.get("cmd", {}).get("effect", ""), "#88ccdd")
            self.setText(f"{slot_num}\n{name}")
            self.setStyleSheet(
                f"QPushButton#preset-btn {{ color: {color};"
                " background-color: #050d14; border: 1px solid #0a3044;"
                " border-radius: 0px; min-height: 48px; font-size: 11px; letter-spacing: 1px; }"
                f"QPushButton#preset-btn:hover {{ border-color: {color}88; }}"
                f"QPushButton#preset-btn:pressed {{ background-color: #071520; }}"
            )
        else:
            self.setText(f"{slot_num}\n—")
            self.setStyleSheet(
                "QPushButton#preset-btn { color: #0d2233; background-color: #050d14;"
                " border: 1px solid #08151f; border-radius: 0px; min-height: 48px;"
                " font-size: 11px; }"
                "QPushButton#preset-btn:hover { border-color: #0d3344; }"
            )

    def contextMenuEvent(self, event) -> None:  # type: ignore[override]
        if self._preset:
            menu = QMenu(self)
            clear_act = QAction("Remove preset", self)
            clear_act.triggered.connect(lambda: self.remove_req.emit(self._slot))
            menu.addAction(clear_act)
            menu.exec(event.globalPos())


# ── SequenceQueueWidget ────────────────────────────────────────────────────────

class SequenceQueueWidget(QWidget):
    """DJ-style timeline showing current sequence progress + queued sequences."""

    def __init__(self, worker: "VJWorker", parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._worker = worker
        self.setFixedHeight(QUEUE_WIDGET_HEIGHT)
        self.setMinimumWidth(200)
        self._timer = QTimer(self)
        self._timer.setInterval(50)
        self._timer.timeout.connect(self.update)
        self._timer.start()

    def paintEvent(self, event) -> None:  # type: ignore[override]
        snap = self._worker.get_queue_snapshot()
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, False)

        w = self.width()
        h = self.height()
        MARGIN = 4
        GAP = 4

        queue = snap["queue"]
        n_queue = len(queue)
        queue_total = sum(
            item["repeats"] * QUEUE_ITEM_WIDTH + (item["repeats"] - 1) * QUEUE_LOOP_GAP + GAP
            for item in queue
        ) if n_queue else 0
        current_w = max(100, w - MARGIN * 2 - queue_total)

        block_x = MARGIN
        block_y = MARGIN
        block_h = h - MARGIN * 2

        current_name = snap["current_name"]
        current_effect = snap["current_effect"]
        seq_phase = snap["seq_phase"]
        current_repeats = snap["current_repeats"]

        if current_name:
            color_hex = EFFECT_COLORS.get(current_effect, "#888888")
            base_color = QColor(color_hex)

            playhead_x = block_x + int(current_w * seq_phase)

            # Played region
            played = QColor(base_color)
            played.setAlpha(80)
            painter.fillRect(block_x, block_y, max(0, playhead_x - block_x), block_h, played)

            # Remaining region
            remain = QColor(base_color)
            remain.setAlpha(30)
            painter.fillRect(playhead_x, block_y, current_w - (playhead_x - block_x), block_h, remain)

            # Loop dividers (dim vertical lines at each repeat boundary)
            if current_repeats > 1:
                painter.setPen(QPen(QColor(color_hex), 1))
                for loop_i in range(1, current_repeats):
                    div_x = block_x + int(current_w * loop_i / current_repeats)
                    painter.drawLine(div_x, block_y, div_x, block_y + block_h - 1)

            # Border
            painter.setPen(QPen(QColor(color_hex), 1))
            painter.drawRect(block_x, block_y, current_w - 1, block_h - 1)

            # Playhead
            painter.setPen(QPen(QColor("#00e5ff"), 2))
            painter.drawLine(playhead_x, block_y, playhead_x, block_y + block_h - 1)

            # Drop marker — predicted structural change from the drop estimator
            next_change_beat = snap.get("next_change_beat")
            seq_start_beat = snap.get("seq_start_beat", 0.0)
            seq_total = snap.get("seq_total_beats", 0)
            if next_change_beat is not None and seq_total > 0:
                drop_offset = next_change_beat - seq_start_beat
                if 0 < drop_offset < seq_total:
                    drop_x = block_x + int(drop_offset / seq_total * current_w)
                    painter.setPen(QPen(QColor(DROP_MARKER_COLOR), 2, Qt.PenStyle.DotLine))
                    painter.drawLine(drop_x, block_y, drop_x, block_y + block_h - 1)

            # Label
            painter.setPen(QColor(color_hex))
            painter.setFont(QFont(_MONO_FONT, 10, QFont.Weight.Bold))
            painter.drawText(block_x + 6, block_y, current_w - 12, block_h,
                             Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft,
                             current_name)
        else:
            painter.fillRect(block_x, block_y, current_w, block_h, QColor("#050d14"))
            painter.setPen(QPen(QColor("#0a1e2a"), 1))
            painter.drawRect(block_x, block_y, current_w - 1, block_h - 1)

        # Queue items
        x = block_x + current_w + GAP
        for item in queue:
            effect = item["effect"]
            name = item["name"] or effect
            repeats = item["repeats"]
            color_hex = EFFECT_COLORS.get(effect, "#888888")

            for loop_i in range(repeats):
                fill = QColor(color_hex)
                fill.setAlpha(120)
                painter.fillRect(x, block_y, QUEUE_ITEM_WIDTH, block_h, fill)

                painter.setPen(QPen(QColor(color_hex), 1))
                painter.drawRect(x, block_y, QUEUE_ITEM_WIDTH - 1, block_h - 1)

                painter.setPen(QColor("#88ccdd"))
                painter.setFont(QFont(_MONO_FONT, 9))
                label = name[:8] if loop_i == 0 else f"×{loop_i + 1}"
                painter.drawText(x + 4, block_y, QUEUE_ITEM_WIDTH - 8, block_h,
                                 Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft,
                                 label)
                x += QUEUE_ITEM_WIDTH + QUEUE_LOOP_GAP

            x += GAP - QUEUE_LOOP_GAP

        painter.end()


# ── Main window ────────────────────────────────────────────────────────────────

class VJMainWindow(QMainWindow):
    def __init__(self, worker: VJWorker) -> None:
        super().__init__()
        self._worker = worker
        self._presets: list[dict | None] = load_presets()
        self._bpm_mode = "link"
        self._knobs_enabled = False

        self.setWindowTitle("[ AUTO-VJ ]")
        self.setMinimumWidth(360)
        self.setStyleSheet(DARK_QSS)

        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)
        root.setContentsMargins(10, 10, 10, 10)
        root.setSpacing(8)

        root.addLayout(self._build_status_bar())
        root.addWidget(self._build_effect_box())
        root.addWidget(self._build_structure_box())
        root.addWidget(self._build_queue_box())
        root.addWidget(self._build_controls_box())
        root.addWidget(self._build_presets_box())

        # Wire worker signals
        worker.status_changed.connect(self._on_status)
        worker.effect_activated.connect(self._on_effect)
        worker.bpm_changed.connect(self._on_bpm)
        worker.link_status_changed.connect(self._on_link)
        worker.bpm_mode_changed.connect(self._on_bpm_mode)
        worker.queue_status_changed.connect(self._on_queue_status)
        worker.worker_error.connect(self._on_worker_error)
        worker.auto_mode_changed.connect(self._on_auto_mode)
        worker.structure_changed.connect(self._on_structure)

        self._bpm_knob_timer = QTimer(self)
        self._bpm_knob_timer.setSingleShot(True)
        self._bpm_knob_timer.setInterval(80)
        self._bpm_knob_timer.timeout.connect(self._flush_bpm_knob)

        self._speed_timer = QTimer(self)
        self._speed_timer.setSingleShot(True)
        self._speed_timer.setInterval(80)
        self._speed_timer.timeout.connect(self._flush_speed_knob)

        self._brightness_timer = QTimer(self)
        self._brightness_timer.setSingleShot(True)
        self._brightness_timer.setInterval(80)
        self._brightness_timer.timeout.connect(self._flush_brightness_knob)

        # Latest structure estimate — updated via signal, countdown recomputed every 50 ms
        self._last_structure_est: StructureEstimate | None = None
        self._countdown_timer = QTimer(self)
        self._countdown_timer.setInterval(50)
        self._countdown_timer.timeout.connect(self._update_countdown)
        self._countdown_timer.start()

        self._knob_speed.value_changed.connect(lambda _: self._speed_timer.start())
        self._knob_brightness.value_changed.connect(lambda _: self._brightness_timer.start())
        self._knob_bpm.value_changed.connect(lambda _: self._bpm_knob_timer.start())

    # ── Layout builders ────────────────────────────────────────────────────────

    def _build_status_bar(self) -> QHBoxLayout:
        row = QHBoxLayout()
        row.setSpacing(8)

        self._state_pill = QLabel("WAITING")
        self._state_pill.setObjectName("state-pill")
        self._state_pill.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._state_pill.setFixedWidth(110)
        self._apply_state_style("WAITING")

        self._bpm_label = QLabel("BPM: —")
        self._bpm_label.setStyleSheet("color: #ff9900; font-weight: bold; letter-spacing: 1px;")

        self._link_label = QLabel("LINK: —")
        self._link_label.setStyleSheet("color: #1a3a4a; letter-spacing: 1px;")

        self._structure_label = QLabel("~?b")
        self._structure_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._structure_label.setStyleSheet(
            f"QLabel {{ color: {NOVELTY_COLORS['pending']}; font-size: 10px;"
            f" letter-spacing: 1px; }}"
        )

        self._mode_label = QLabel("LINK")
        self._mode_label.setStyleSheet("color: #1a3a4a; font-size: 10px; letter-spacing: 1px;")
        self._mode_label.setToolTip("BPM mode — press M to cycle")

        mode_btn = QPushButton("M")
        mode_btn.setFixedSize(24, 24)
        mode_btn.setToolTip("Cycle BPM mode (link / tap / mic)")
        mode_btn.setStyleSheet(
            "QPushButton { background:#050d14; border:1px solid #0a1e2a;"
            " border-radius:0px; color:#1a3a4a; font-size:10px; }"
            "QPushButton:hover { color:#00e5ff; border-color:#005566; }"
        )
        mode_btn.clicked.connect(self._worker.cycle_bpm_mode)

        self._auto_pill = QPushButton("AUTO")
        self._auto_pill.setFixedSize(52, 24)
        self._auto_pill.setCheckable(True)
        self._auto_pill.setToolTip("Auto-refill queue with AI prompts (A)")
        self._auto_pill.setStyleSheet(
            "QPushButton { background:#050d14; border:1px solid #0a1e2a;"
            " border-radius:0px; color:#1a3a4a; font-size:10px; font-weight:bold; letter-spacing:2px; }"
            "QPushButton:checked { background:#ff990022; border-color:#ff990088; color:#ff9900; }"
            "QPushButton:hover { border-color:#005566; }"
        )
        self._auto_pill.clicked.connect(self._worker.toggle_auto_mode)

        row.addWidget(self._state_pill)
        row.addWidget(self._bpm_label)
        row.addWidget(self._link_label)
        row.addWidget(self._structure_label)
        row.addStretch()
        row.addWidget(self._mode_label)
        row.addWidget(mode_btn)
        row.addWidget(self._auto_pill)
        return row

    def _build_effect_box(self) -> QGroupBox:
        box = QGroupBox("[ EFFECT ]")
        layout = QVBoxLayout(box)
        layout.setContentsMargins(8, 10, 8, 6)
        self._effect_label = QLabel("—")
        self._effect_label.setStyleSheet("color: #1a3a4a; font-size: 11px; letter-spacing: 1px;")
        self._effect_label.setWordWrap(True)
        layout.addWidget(self._effect_label)
        return box

    def _build_structure_box(self) -> QGroupBox:
        box = QGroupBox("[ STRUCTURE ]")
        layout = QVBoxLayout(box)
        layout.setContentsMargins(8, 10, 8, 6)
        layout.setSpacing(4)

        # Row 1: countdown + period + confidence
        row1 = QHBoxLayout()
        self._structure_countdown = QLabel("~?b")
        self._structure_countdown.setStyleSheet(
            f"color: {NOVELTY_COLORS['pending']}; font-size: 16px; font-weight: bold; letter-spacing: 2px;"
        )
        self._structure_period = QLabel("period: --")
        self._structure_period.setStyleSheet("color: #1a3a4a; font-size: 10px; letter-spacing: 1px;")
        self._structure_conf = QLabel("conf: --")
        self._structure_conf.setStyleSheet("color: #1a3a4a; font-size: 10px; letter-spacing: 1px;")
        row1.addWidget(self._structure_countdown)
        row1.addStretch()
        row1.addWidget(self._structure_period)
        row1.addSpacing(10)
        row1.addWidget(self._structure_conf)

        # Row 2: novelty bar + value
        row2 = QHBoxLayout()
        novelty_lbl = QLabel("novelty")
        novelty_lbl.setStyleSheet("color: #1a3a4a; font-size: 10px; letter-spacing: 1px;")
        self._novelty_bar = QProgressBar()
        self._novelty_bar.setRange(0, 100)
        self._novelty_bar.setValue(0)
        self._novelty_bar.setTextVisible(False)
        self._novelty_bar.setFixedHeight(6)
        self._novelty_bar.setStyleSheet(
            "QProgressBar { border: 1px solid #0d3344; background: #030a0e; border-radius: 0px; }"
            f"QProgressBar::chunk {{ background: {NOVELTY_COLORS['pending']}; }}"
        )
        self._novelty_value = QLabel("0.00")
        self._novelty_value.setStyleSheet("color: #1a3a4a; font-size: 10px; letter-spacing: 1px;")
        row2.addWidget(novelty_lbl)
        row2.addSpacing(6)
        row2.addWidget(self._novelty_bar)
        row2.addSpacing(6)
        row2.addWidget(self._novelty_value)

        layout.addLayout(row1)
        layout.addLayout(row2)
        return box

    def _build_queue_box(self) -> QGroupBox:
        box = QGroupBox("[ QUEUE ]")
        layout = QVBoxLayout(box)
        layout.setContentsMargins(8, 10, 8, 6)
        self._queue_widget = SequenceQueueWidget(self._worker)
        layout.addWidget(self._queue_widget)
        return box

    def _build_controls_box(self) -> QGroupBox:
        box = QGroupBox("[ CONTROLS ]")
        row = QHBoxLayout(box)
        row.setContentsMargins(8, 14, 8, 6)
        row.setSpacing(16)

        self._knob_speed      = KnobWidget("SPEED",      SPEED_MIN,      SPEED_MAX,      1.0, ".2f")
        self._knob_brightness = KnobWidget("BRIGHTNESS", BRIGHTNESS_MIN, BRIGHTNESS_MAX, 1.0, ".2f")
        self._knob_bpm        = KnobWidget("BPM",        BPM_GUI_MIN,    BPM_GUI_MAX,    120.0, ".0f")

        for knob in (self._knob_speed, self._knob_brightness, self._knob_bpm):
            row.addWidget(knob)
        return box

    def _build_presets_box(self) -> QGroupBox:
        box = QGroupBox("[ FAVORITES  1-9 ]")
        vl = QVBoxLayout(box)
        vl.setContentsMargins(8, 14, 8, 8)
        vl.setSpacing(6)

        grid = QGridLayout()
        grid.setSpacing(4)
        self._preset_btns: list[PresetButton] = []

        for i in range(NUM_PRESETS):
            btn = PresetButton(i)
            btn.set_preset(self._presets[i])
            btn.recalled.connect(self._recall_preset)
            btn.remove_req.connect(self._remove_preset)
            self._preset_btns.append(btn)
            grid.addWidget(btn, i // 3, i % 3)

        vl.addLayout(grid)

        star_row = QHBoxLayout()
        star_row.addStretch()
        self._star_btn = QPushButton("★  Star current")
        self._star_btn.setObjectName("star-btn")
        self._star_btn.setToolTip("Save active effect to first empty slot (1-9)")
        self._star_btn.clicked.connect(self._star_current)
        star_row.addWidget(self._star_btn)
        vl.addLayout(star_row)
        return box

    # ── Slot handlers ──────────────────────────────────────────────────────────

    def _apply_state_style(self, state: str) -> None:
        color = STATE_COLORS.get(state, "#1a3a4a")
        self._state_pill.setStyleSheet(
            f"QLabel#state-pill {{ background-color: {color}22; color: {color};"
            f" border: 1px solid {color}88; border-radius: 0px;"
            f" padding: 2px 10px; font-weight: bold; font-size: 11px; letter-spacing: 3px; }}"
        )
        self._state_pill.setText(state)

    def _on_status(self, state: str) -> None:
        self._apply_state_style(state)

    def _on_effect(self, prompt: str, cmd: object) -> None:
        cmd = dict(cmd)  # type: ignore[arg-type]
        name = cmd.get("effect", "?")
        color = EFFECT_COLORS.get(name, "#88ccdd")
        text = _fmt_effect(cmd)
        self._effect_label.setText(
            f"<span style='color:{color};font-weight:bold;letter-spacing:2px;'>{name.upper()}</span>"
            f"  <span style='color:#1a3a4a;font-size:10px;'>{text[len(name):].strip()}</span>"
        )
        self._knobs_enabled = True
        params = cmd.get("params", {})
        speed_key = "rate" if name in RATE_EFFECTS else "speed"
        if speed_key in params:
            self._knob_speed.set_value(float(params[speed_key]))
        for f in cmd.get("filters", []):
            if f["type"] == "dim":
                self._knob_brightness.set_value(float(f.get("params", {}).get("brightness", 1.0)))

    def _on_queue_status(self, queue_length: int) -> None:
        pass  # queue widget self-updates via its 50ms timer

    def _on_auto_mode(self, active: bool) -> None:
        self._auto_pill.setChecked(active)

    def _on_bpm(self, bpm: float) -> None:
        self._bpm_label.setText(f"BPM: {bpm:.1f}")
        self._knob_bpm.set_value(bpm)

    def _on_link(self, tempo: float, peers: int) -> None:
        peers_str = f"{peers} peer{'s' if peers != 1 else ''}" if peers else "solo"
        self._link_label.setText(f"LINK {tempo:.1f}  [{peers_str}]")

    def _on_bpm_mode(self, mode: str) -> None:
        self._bpm_mode = mode
        self._mode_label.setText(mode.upper())

    def _on_structure(self, est: object) -> None:
        e: StructureEstimate = est  # type: ignore[assignment]
        self._last_structure_est = e  # countdown labels updated by _countdown_timer

        # Period + confidence
        if e.change_period is not None:
            self._structure_period.setText(f"period: {e.change_period}b")
            self._structure_period.setStyleSheet(
                f"color: {NOVELTY_COLORS['active']}; font-size: 10px; letter-spacing: 1px;"
            )
        else:
            self._structure_period.setText("period: --")
            self._structure_period.setStyleSheet("color: #1a3a4a; font-size: 10px; letter-spacing: 1px;")

        conf_pct = int(e.confidence * 100)
        self._structure_conf.setText(f"conf: {conf_pct}%")
        conf_color = NOVELTY_COLORS["active"] if conf_pct >= 60 else "#1a3a4a"
        self._structure_conf.setStyleSheet(f"color: {conf_color}; font-size: 10px; letter-spacing: 1px;")

        # Novelty bar — scaled so NOVELTY_DISPLAY_MAX fills 100%, threshold at 50%
        bar_pct = int(min(e.novelty / NOVELTY_DISPLAY_MAX, 1.0) * 100)
        self._novelty_bar.setValue(bar_pct)
        bar_color = NOVELTY_COLORS["hot"] if e.novelty >= NOVELTY_THRESHOLD else NOVELTY_COLORS["active"]
        self._novelty_bar.setStyleSheet(
            "QProgressBar { border: 1px solid #0d3344; background: #030a0e; border-radius: 0px; }"
            f"QProgressBar::chunk {{ background: {bar_color}; }}"
        )
        self._novelty_value.setText(f"{e.novelty:.2f}")
        self._novelty_value.setStyleSheet(f"color: {bar_color}; font-size: 10px; letter-spacing: 1px;")

    def _update_countdown(self) -> None:
        """Recompute the beats-to-change countdown every 50 ms for smooth display."""
        e = self._last_structure_est
        if e is None or e.next_change_beat is None:
            self._structure_label.setText("~?b")
            self._structure_countdown.setText("~?b")
            return
        beats_left = max(0.0, e.next_change_beat - self._worker.get_current_beat())
        cd_color = NOVELTY_COLORS["hot"] if e.novelty >= NOVELTY_THRESHOLD else NOVELTY_COLORS["active"]
        text = f"~{int(beats_left)}b"
        self._structure_label.setText(text)
        self._structure_label.setStyleSheet(
            f"QLabel {{ color: {cd_color}; font-size: 10px; letter-spacing: 1px; }}"
        )
        self._structure_countdown.setText(text)
        self._structure_countdown.setStyleSheet(
            f"color: {cd_color}; font-size: 16px; font-weight: bold; letter-spacing: 2px;"
        )

    def _on_worker_error(self, msg: str) -> None:
        self._apply_state_style("WAITING")
        self._effect_label.setText(
            f"<span style='color:#ff2200;letter-spacing:2px;'>[ SYSTEM FAILURE — CHECK TERMINAL ]</span>"
        )
        dlg = QMessageBox(self)
        dlg.setWindowTitle("auto-vj — worker error")
        dlg.setIcon(QMessageBox.Icon.Critical)
        dlg.setText("The VJ worker thread crashed.")
        dlg.setDetailedText(msg)
        dlg.exec()

    # ── Knob flush ─────────────────────────────────────────────────────────────

    def _flush_speed_knob(self) -> None:
        if self._knobs_enabled:
            self._worker.apply_speed(self._knob_speed.get_value())

    def _flush_brightness_knob(self) -> None:
        if self._knobs_enabled:
            self._worker.apply_brightness(self._knob_brightness.get_value())

    def _flush_bpm_knob(self) -> None:
        self._worker.set_bpm(self._knob_bpm.get_value())

    # ── Presets ────────────────────────────────────────────────────────────────

    def _recall_preset(self, slot: int) -> None:
        preset = self._presets[slot]
        if preset:
            self._worker.activate_preset(preset["cmd"])

    def _remove_preset(self, slot: int) -> None:
        self._presets[slot] = None
        self._preset_btns[slot].set_preset(None)
        save_presets(self._presets)

    def _star_current(self) -> None:
        cmd = self._worker.current_cmd
        if not cmd:
            self._star_btn.setText("★  (no active effect)")
            QTimer.singleShot(1500, lambda: self._star_btn.setText("★  Star current"))
            return
        slot = next((i for i, p in enumerate(self._presets) if p is None), None)
        if slot is None:
            self._star_btn.setText("★  All slots full!")
            QTimer.singleShot(1500, lambda: self._star_btn.setText("★  Star current"))
            return
        label = cmd.get("effect", "?")
        self._presets[slot] = {"label": label, "cmd": copy.deepcopy(cmd)}
        self._preset_btns[slot].set_preset(self._presets[slot])
        save_presets(self._presets)
        self._star_btn.setText(f"★  Saved to slot {slot + 1}")
        QTimer.singleShot(1500, lambda: self._star_btn.setText("★  Star current"))

    # ── Keyboard ───────────────────────────────────────────────────────────────

    def keyPressEvent(self, event: QKeyEvent) -> None:
        key = event.key()
        if Qt.Key.Key_1 <= key <= Qt.Key.Key_9:
            self._recall_preset(key - Qt.Key.Key_1)
            return
        if key == Qt.Key.Key_Space and not event.isAutoRepeat():
            self._worker.ptt_press()
            return
        if key == Qt.Key.Key_M:
            self._worker.cycle_bpm_mode()
            return
        if key == Qt.Key.Key_A:
            self._worker.toggle_auto_mode()
            return
        if key in (Qt.Key.Key_Return, Qt.Key.Key_Enter):
            self._worker.tap_beat()
            return
        super().keyPressEvent(event)

    def keyReleaseEvent(self, event: QKeyEvent) -> None:
        if event.key() == Qt.Key.Key_Space and not event.isAutoRepeat():
            self._worker.ptt_release()
            return
        super().keyReleaseEvent(event)


# ── Entry point ────────────────────────────────────────────────────────────────

def run_gui() -> None:
    app = QApplication(sys.argv)
    app.setApplicationName("auto-vj")

    worker = VJWorker()
    window = VJMainWindow(worker)
    window.show()
    worker.start()

    def _on_quit() -> None:
        worker.shutdown()
        worker.wait(3000)

    app.aboutToQuit.connect(_on_quit)
    sys.exit(app.exec())
