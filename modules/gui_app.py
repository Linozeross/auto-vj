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
    QPushButton,
    QVBoxLayout,
    QWidget,
)
from dotenv import load_dotenv
from openai import OpenAI
from pydantic import ValidationError

import numpy as np
import sounddevice as sd

from modules.artnet_renderer import ArtNetRenderer, MultiRenderer
from modules.beat_detector import MicBeatDetector
from modules.bpm import BpmMode, LinkClock
from modules.effects import VJResponse
from modules.recorder import SAMPLE_RATE, CHANNELS, audio_to_wav_b64
from modules.sequences import Sequence, sequence_from_dict, sequence_from_effect_cmd, PHRASE_BEATS

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

SYSTEM_PROMPT = open("system_prompt.txt").read().strip()

# Effects that use "rate" instead of "speed"
RATE_EFFECTS = {"pulse"}

STATE_COLORS: dict[str, str] = {
    "WAITING":      "#555555",
    "RECORDING":    "#cc3333",
    "TRANSCRIBING": "#b8a020",
    "THINKING":     "#2090b8",
    "LIVE":         "#2ea850",
}

EFFECT_COLORS: dict[str, str] = {
    "pulse":        "#cc44cc",
    "rainbow":      "#d4c040",
    "chase":        "#40c860",
    "twinkle":      "#60e0e0",
    "plasma":       "#d060e0",
    "palette_wave": "#e08020",
    "beat_pulse":   "#d09000",
}

# Qt QSS accepts only a single font-family value (no comma fallback lists).
_MONO_FONT = "Menlo" if platform.system() == "Darwin" else "Consolas"

def _build_qss(font: str) -> str:
    return (
        f'QMainWindow, QWidget {{ background-color: #0e0e0e; color: #d4d4d4;'
        f' font-family: "{font}"; font-size: 12px; }}\n'
        'QGroupBox { border: 1px solid #222; border-radius: 4px; margin-top: 10px;'
        ' color: #444; font-size: 10px; padding: 4px; }\n'
        'QGroupBox::title { subcontrol-origin: margin; left: 8px; padding: 0 4px; }\n'
        'QLabel#state-pill { border-radius: 3px; padding: 2px 10px;'
        ' font-size: 11px; font-weight: bold; }\n'
        'QPushButton#preset-btn { background-color: #141414; color: #555;'
        ' border: 1px solid #252525; border-radius: 4px; padding: 4px 2px;'
        ' font-size: 11px; min-height: 48px; text-align: center; }\n'
        'QPushButton#preset-btn[filled="true"] { color: #d4d4d4; border-color: #333; }\n'
        'QPushButton#preset-btn:hover { border-color: #444; }\n'
        'QPushButton#preset-btn:pressed { background-color: #202020; }\n'
        'QPushButton#star-btn { background-color: #141414; color: #888;'
        ' border: 1px solid #252525; border-radius: 4px; padding: 4px 14px; }\n'
        'QPushButton#star-btn:hover { border-color: #c08020; color: #e09820; }\n'
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
    raw = response.choices[0].message.content
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
        self._client: OpenAI | None = None
        # Sequence queue state (asyncio-only, no lock needed)
        self._seq_queue: list[Sequence] = []
        self._seq_current: Sequence | None = None
        self._seq_start_beat: float = 0.0
        # PTT coordination
        self._ptt_start: asyncio.Event | None = None
        self._stop_recording: threading.Event = threading.Event()

    def run(self) -> None:
        try:
            self._loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self._loop)
            self._loop.run_until_complete(self._main())
        except Exception:
            msg = traceback.format_exc()
            print(msg, file=sys.stderr)
            self.worker_error.emit(msg)

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
        """Wake on sub-beat intervals and advance the queue at phrase boundaries."""
        while True:
            tempo = self._link_clock.tempo if self._link_clock else 120.0
            sleep_secs = 60.0 / max(tempo, 20.0) / 4.0
            await asyncio.sleep(sleep_secs)

            if self._seq_current is None or not self._seq_queue:
                continue

            seq_t = self._link_clock.beat - self._seq_start_beat
            beat_number = self._link_clock._beat_number

            if self._seq_current.is_done(seq_t) and beat_number % PHRASE_BEATS == 0:
                next_seq = self._seq_queue.pop(0)
                self._seq_current = next_seq
                self._seq_start_beat = self._link_clock.beat
                self._link_clock.attach_effect(next_seq)
                self._renderer.set_effect(next_seq)
                self.queue_status_changed.emit(len(self._seq_queue))

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

            audio = await asyncio.to_thread(_record_gui, self._stop_recording, _on_recording_start)

            if self._bpm_mode is BpmMode.MIC and self._beat_detector:
                self._beat_detector.resume()

            if audio.size == 0:
                continue

            self.status_changed.emit("THINKING")
            try:
                vj_response, self._history = await asyncio.to_thread(
                    _chat_to_response, audio, self._history, self._client
                )
            except (ValidationError, ValueError) as exc:
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

        self._bpm_mode = next_mode
        self._tap_times = []

        if self._bpm_mode is BpmMode.MIC and self._beat_detector:
            self._beat_detector.start()

        self.bpm_mode_changed.emit(next_mode.value)

    @property
    def current_cmd(self) -> dict | None:
        return self._current_cmd

    def get_queue_snapshot(self) -> dict:
        """Thread-safe snapshot of queue state for UI polling (GIL protects simple reads)."""
        current_name = ""
        current_effect = ""
        phrase_phase = 0.0

        if self._seq_current is not None:
            current_name = self._seq_current.name or ""
            current_effect = current_name

        if self._link_clock is not None:
            beat_number = self._link_clock._beat_number
            beat_frac = self._link_clock.beat_phase()
            phrase_phase = ((beat_number % PHRASE_BEATS) + beat_frac) / PHRASE_BEATS

        queue_items = [{"name": seq.name or "", "effect": seq.name or ""} for seq in self._seq_queue]

        return {
            "current_name": current_name,
            "current_effect": current_effect,
            "phrase_phase": phrase_phase,
            "queue": queue_items,
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
        self._top_label.setStyleSheet("color: #555; font-size: 10px;")

        self._val_label = QLabel()
        self._val_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._val_label.setStyleSheet("color: #bbb; font-size: 11px; font-weight: bold;")

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
            color = EFFECT_COLORS.get(self._preset.get("cmd", {}).get("effect", ""), "#d4d4d4")
            self.setText(f"{slot_num}\n{name}")
            self.setStyleSheet(
                f"QPushButton#preset-btn {{ color: {color};"
                " background-color: #141414; border: 1px solid #333;"
                " border-radius: 4px; min-height: 48px; font-size: 11px; }"
                f"QPushButton#preset-btn:hover {{ border-color: #666; }}"
                f"QPushButton#preset-btn:pressed {{ background-color: #202020; }}"
            )
        else:
            self.setText(f"{slot_num}\n—")
            self.setStyleSheet(
                "QPushButton#preset-btn { color: #2a2a2a; background-color: #141414;"
                " border: 1px solid #1e1e1e; border-radius: 4px; min-height: 48px;"
                " font-size: 11px; }"
                "QPushButton#preset-btn:hover { border-color: #333; }"
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
        queue_total = n_queue * (QUEUE_ITEM_WIDTH + GAP) if n_queue else 0
        current_w = max(100, w - MARGIN * 2 - queue_total - (GAP if n_queue else 0))

        block_x = MARGIN
        block_y = MARGIN
        block_h = h - MARGIN * 2

        current_name = snap["current_name"]
        current_effect = snap["current_effect"]
        phrase_phase = snap["phrase_phase"]

        if current_name:
            color_hex = EFFECT_COLORS.get(current_effect, "#888888")
            base_color = QColor(color_hex)

            playhead_x = block_x + int(current_w * phrase_phase)

            # Played region
            played = QColor(base_color)
            played.setAlpha(80)
            painter.fillRect(block_x, block_y, max(0, playhead_x - block_x), block_h, played)

            # Remaining region
            remain = QColor(base_color)
            remain.setAlpha(30)
            painter.fillRect(playhead_x, block_y, current_w - (playhead_x - block_x), block_h, remain)

            # Border
            painter.setPen(QPen(QColor(color_hex), 1))
            painter.drawRect(block_x, block_y, current_w - 1, block_h - 1)

            # Playhead
            painter.setPen(QPen(QColor("#ffffff"), 2))
            painter.drawLine(playhead_x, block_y, playhead_x, block_y + block_h - 1)

            # Label
            painter.setPen(QColor(color_hex))
            painter.setFont(QFont(_MONO_FONT, 10, QFont.Weight.Bold))
            painter.drawText(block_x + 6, block_y, current_w - 12, block_h,
                             Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft,
                             current_name)
        else:
            painter.fillRect(block_x, block_y, current_w, block_h, QColor("#111111"))
            painter.setPen(QPen(QColor("#222222"), 1))
            painter.drawRect(block_x, block_y, current_w - 1, block_h - 1)

        # Queue items
        x = block_x + current_w + GAP
        for item in queue:
            effect = item["effect"]
            name = item["name"] or effect
            color_hex = EFFECT_COLORS.get(effect, "#888888")

            fill = QColor(color_hex)
            fill.setAlpha(120)
            painter.fillRect(x, block_y, QUEUE_ITEM_WIDTH, block_h, fill)

            painter.setPen(QPen(QColor(color_hex), 1))
            painter.drawRect(x, block_y, QUEUE_ITEM_WIDTH - 1, block_h - 1)

            painter.setPen(QColor("#dddddd"))
            painter.setFont(QFont(_MONO_FONT, 9))
            painter.drawText(x + 4, block_y, QUEUE_ITEM_WIDTH - 8, block_h,
                             Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft,
                             name[:8])
            x += QUEUE_ITEM_WIDTH + GAP

        painter.end()


# ── Main window ────────────────────────────────────────────────────────────────

class VJMainWindow(QMainWindow):
    def __init__(self, worker: VJWorker) -> None:
        super().__init__()
        self._worker = worker
        self._presets: list[dict | None] = load_presets()
        self._bpm_mode = "link"
        self._knobs_enabled = False

        self.setWindowTitle("auto-vj")
        self.setMinimumWidth(360)
        self.setStyleSheet(DARK_QSS)

        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)
        root.setContentsMargins(10, 10, 10, 10)
        root.setSpacing(8)

        root.addLayout(self._build_status_bar())
        root.addWidget(self._build_effect_box())
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
        self._bpm_label.setStyleSheet("color: #70b8e0; font-weight: bold;")

        self._link_label = QLabel("Link: —")
        self._link_label.setStyleSheet("color: #555;")

        self._mode_label = QLabel("link")
        self._mode_label.setStyleSheet("color: #444; font-size: 10px;")
        self._mode_label.setToolTip("BPM mode — press M to cycle")

        mode_btn = QPushButton("M")
        mode_btn.setFixedSize(24, 24)
        mode_btn.setToolTip("Cycle BPM mode (link / tap / mic)")
        mode_btn.setStyleSheet(
            "QPushButton { background:#1a1a1a; border:1px solid #2a2a2a;"
            " border-radius:3px; color:#555; font-size:10px; }"
            "QPushButton:hover { color:#aaa; border-color:#444; }"
        )
        mode_btn.clicked.connect(self._worker.cycle_bpm_mode)

        row.addWidget(self._state_pill)
        row.addWidget(self._bpm_label)
        row.addWidget(self._link_label)
        row.addStretch()
        row.addWidget(self._mode_label)
        row.addWidget(mode_btn)
        return row

    def _build_effect_box(self) -> QGroupBox:
        box = QGroupBox("EFFECT")
        layout = QVBoxLayout(box)
        layout.setContentsMargins(8, 10, 8, 6)
        self._effect_label = QLabel("—")
        self._effect_label.setStyleSheet("color: #666; font-size: 11px;")
        self._effect_label.setWordWrap(True)
        layout.addWidget(self._effect_label)
        return box

    def _build_queue_box(self) -> QGroupBox:
        box = QGroupBox("QUEUE")
        layout = QVBoxLayout(box)
        layout.setContentsMargins(8, 10, 8, 6)
        self._queue_widget = SequenceQueueWidget(self._worker)
        layout.addWidget(self._queue_widget)
        return box

    def _build_controls_box(self) -> QGroupBox:
        box = QGroupBox("CONTROLS")
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
        box = QGroupBox("FAVORITES  (1-9)")
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
        color = STATE_COLORS.get(state, "#555")
        self._state_pill.setStyleSheet(
            f"QLabel#state-pill {{ background-color: {color}22; color: {color};"
            f" border: 1px solid {color}66; border-radius: 3px;"
            f" padding: 2px 10px; font-weight: bold; font-size: 11px; }}"
        )
        self._state_pill.setText(state)

    def _on_status(self, state: str) -> None:
        self._apply_state_style(state)

    def _on_effect(self, prompt: str, cmd: object) -> None:
        cmd = dict(cmd)  # type: ignore[arg-type]
        name = cmd.get("effect", "?")
        color = EFFECT_COLORS.get(name, "#d4d4d4")
        text = _fmt_effect(cmd)
        self._effect_label.setText(
            f"<span style='color:{color};font-weight:bold;'>{name}</span>"
            f"  <span style='color:#555;font-size:10px;'>{text[len(name):].strip()}</span>"
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

    def _on_bpm(self, bpm: float) -> None:
        self._bpm_label.setText(f"BPM: {bpm:.1f}")
        self._knob_bpm.set_value(bpm)

    def _on_link(self, tempo: float, peers: int) -> None:
        peers_str = f"{peers} peer{'s' if peers != 1 else ''}" if peers else "solo"
        self._link_label.setText(f"Link {tempo:.1f}  [{peers_str}]")

    def _on_bpm_mode(self, mode: str) -> None:
        self._bpm_mode = mode
        self._mode_label.setText(mode)

    def _on_worker_error(self, msg: str) -> None:
        self._apply_state_style("WAITING")
        self._effect_label.setText(
            f"<span style='color:#cc3333;'>Worker crashed — see terminal</span>"
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

    sys.exit(app.exec())
