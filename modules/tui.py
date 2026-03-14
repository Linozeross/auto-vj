from __future__ import annotations

import asyncio
import json
import os
import time

from openai import OpenAI
from dotenv import load_dotenv
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.message import Message
from textual.widgets import DataTable, Footer, Header, Label, Static
from textual.containers import Horizontal, Vertical

from modules.artnet_renderer import ArtNetRenderer, MultiRenderer
from modules.audio_level import AudioLevel
from modules.beat_detector import MicBeatDetector
from modules.bpm import BpmMode, LinkClock
from modules.effects import effect_from_dict, set_audio_level_source
from modules.recorder import record_until_release, transcribe

load_dotenv()

ARTNET_IP = os.environ.get("ARTNET_IP", "192.168.178.102")
LED_COUNT = int(os.environ.get("LED_COUNT", "100"))
# Comma-separated list of ip:led_count pairs, e.g. "192.168.178.102:100,192.168.178.114:60"
# Falls back to ARTNET_IP + LED_COUNT when not set.
ARTNET_NODES_RAW = os.environ.get("ARTNET_NODES", "")


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
SYSTEM_PROMPT = open("system_prompt.txt").read().strip()
TAP_RESET_GAP_SECS = 2.0   # reset tap history if gap between taps exceeds this

# ── State colours ──────────────────────────────────────────────────────────────
STATE_STYLES: dict[str, str] = {
    "WAITING":      "dim white",
    "RECORDING":    "bold red",
    "TRANSCRIBING": "bold yellow",
    "THINKING":     "bold cyan",
    "LIVE":         "bold green",
}

EFFECT_COLORS: dict[str, str] = {
    "solid":        "white",
    "color_wave":   "cyan",
    "pulse":        "magenta",
    "rainbow":      "yellow",
    "chase":        "green",
    "meteor":       "bright_white",
    "twinkle":      "bright_cyan",
    "fire":         "red",
    "plasma":       "bright_magenta",
    "larson":       "bright_red",
    "beat_flash":   "bright_yellow",
    "palette_wave": "orange3",
    "beat_pulse":   "gold1",
    "audio_pulse":  "deep_pink2",
    "audio_wave":   "medium_orchid",
}


# ── Messages ───────────────────────────────────────────────────────────────────
class StatusChanged(Message):
    def __init__(self, state: str) -> None:
        super().__init__()
        self.state = state


class EffectActivated(Message):
    def __init__(self, prompt: str, cmd: dict) -> None:
        super().__init__()
        self.prompt = prompt
        self.cmd = cmd


class BpmChanged(Message):
    def __init__(self, bpm: float) -> None:
        super().__init__()
        self.bpm = bpm


class LinkStatusChanged(Message):
    def __init__(self, tempo: float, peers: int) -> None:
        super().__init__()
        self.tempo = tempo
        self.peers = peers


class BpmModeChanged(Message):
    def __init__(self, mode: BpmMode) -> None:
        super().__init__()
        self.mode = mode


# ── Helpers ────────────────────────────────────────────────────────────────────
def _fmt_params(params: dict) -> str:
    return "  ".join(f"{k}={v}" for k, v in params.items())


def _fmt_effect_label(cmd: dict) -> str:
    name = cmd.get("effect", "?")
    params = cmd.get("params", {})
    filters = cmd.get("filters", [])
    color = EFFECT_COLORS.get(name, "white")
    p_str = _fmt_params(params) if params else "—"
    f_str = ""
    if filters:
        f_str = "  [dim italic]+" + "+".join(f["type"] for f in filters) + "[/]"
    return f"[bold {color}]{name}[/]  [dim]{p_str}[/]{f_str}"


# ── TUI App ────────────────────────────────────────────────────────────────────
class VJApp(App):
    CSS = """
    Screen {
        layout: vertical;
    }

    #body {
        layout: horizontal;
        height: 1fr;
    }

    #left {
        width: 2fr;
        border: solid $primary-darken-2;
        padding: 0 1;
    }

    #right {
        width: 1fr;
        border: solid $primary-darken-2;
        padding: 1 2;
    }

    .section-title {
        color: $text-muted;
        text-style: bold;
        margin-bottom: 1;
    }

    #status {
        text-style: bold;
        margin-bottom: 1;
    }

    #effect-label {
        margin-top: 1;
    }

    #bpm-label {
        margin-top: 1;
        color: $accent;
    }

    #link-label {
        margin-top: 1;
        color: $success;
    }

    #mode-label {
        margin-top: 1;
        color: $text-muted;
    }

    DataTable {
        height: 1fr;
    }
    """

    BINDINGS = [
        Binding("ctrl+c", "quit", "Quit", priority=True),
        Binding("m", "cycle_bpm_mode", "BPM mode", priority=False),
        Binding("enter", "tap_beat", "Tap beat", priority=True),
    ]

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        with Horizontal(id="body"):
            with Vertical(id="left"):
                yield Label("Command History", classes="section-title")
                yield DataTable(id="history")
            with Vertical(id="right"):
                yield Label("Status", classes="section-title")
                yield Label("WAITING", id="status")
                yield Static("[dim]No effect yet[/]", id="effect-label")
                yield Static("[dim]BPM: —[/]", id="bpm-label")
                yield Static("[dim]Link: —[/]", id="link-label")
                yield Static("[dim]Mode: link  (m to cycle)[/]", id="mode-label")
        yield Footer()

    def on_mount(self) -> None:
        table = self.query_one("#history", DataTable)
        table.add_columns("Prompt", "Effect", "Params")
        table.cursor_type = "row"

        self._bpm_mode: BpmMode = BpmMode.LINK
        self._tap_times: list[float] = []

        client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
        self._link_clock = LinkClock()

        def _on_tempo(t: float) -> None:
            self.post_message(LinkStatusChanged(t, self._link_clock.num_peers))

        def _on_peers(n: int) -> None:
            self.post_message(LinkStatusChanged(self._link_clock.tempo, n))

        self._link_clock.set_tempo_callback(_on_tempo)
        self._link_clock.set_num_peers_callback(_on_peers)
        self._link_clock.start()

        def _on_mic_bpm(bpm: float) -> None:
            self._link_clock.set_bpm(bpm)
            self.post_message(BpmChanged(bpm))

        self._beat_detector = MicBeatDetector(on_bpm=_on_mic_bpm)

        # Audio level tracker — runs in LINK/TAP mode; paused in MIC mode to
        # avoid competing with MicBeatDetector on the same input device.
        self._audio_level = AudioLevel()
        self._audio_level.start()
        set_audio_level_source(lambda: self._audio_level.level)

        nodes = _parse_artnet_nodes(ARTNET_NODES_RAW)
        renderer = MultiRenderer([
            ArtNetRenderer(ip, leds, link_clock=self._link_clock)
            for ip, leds in nodes
        ])

        self.run_worker(self._ptt_loop(renderer, client), exclusive=False)
        self.run_worker(renderer.render_loop(), exclusive=False)

    async def _ptt_loop(self, renderer: ArtNetRenderer, client: OpenAI) -> None:
        history: list[dict] = []

        while True:
            self.post_message(StatusChanged("WAITING"))

            def _on_recording_start() -> None:
                if self._bpm_mode is BpmMode.MIC:
                    self._beat_detector.pause()
                else:
                    self._audio_level.pause()

            audio = await asyncio.to_thread(record_until_release, _on_recording_start)
            if self._bpm_mode is BpmMode.MIC:
                self._beat_detector.resume()
            else:
                self._audio_level.resume()

            if audio.size == 0:
                continue

            self.post_message(StatusChanged("TRANSCRIBING"))
            text = await asyncio.to_thread(transcribe, audio, client)
            if not text.strip():
                continue

            self.post_message(StatusChanged("THINKING"))
            cmd, history = await asyncio.to_thread(_chat_to_effect, text, history, client)

            if cmd.get("bpm") is not None:
                bpm = float(cmd["bpm"])
                self._link_clock.set_bpm(bpm)
                self.post_message(BpmChanged(bpm))

            effect = effect_from_dict(cmd)
            self._link_clock.attach_effect(effect)
            renderer.set_effect(effect)
            self.post_message(StatusChanged("LIVE"))
            self.post_message(EffectActivated(text, cmd))

    # ── Message handlers ───────────────────────────────────────────────────────
    def on_status_changed(self, msg: StatusChanged) -> None:
        label = self.query_one("#status", Label)
        style = STATE_STYLES.get(msg.state, "white")
        label.update(f"[{style}]{msg.state}[/]")

    def on_effect_activated(self, msg: EffectActivated) -> None:
        table = self.query_one("#history", DataTable)
        effect_name = msg.cmd.get("effect", "?")
        params_str = _fmt_params(msg.cmd.get("params", {}))
        filters = msg.cmd.get("filters", [])
        if filters:
            params_str += "  [dim][" + "+".join(f["type"] for f in filters) + "][/]"
        table.add_row(msg.prompt, effect_name, params_str)
        table.move_cursor(row=table.row_count - 1)
        self.query_one("#effect-label", Static).update(_fmt_effect_label(msg.cmd))

    def on_bpm_changed(self, msg: BpmChanged) -> None:
        self.query_one("#bpm-label", Static).update(f"[bold]BPM: {msg.bpm:.1f}[/]")

    def on_link_status_changed(self, msg: LinkStatusChanged) -> None:
        peers_str = f"{msg.peers} peer{'s' if msg.peers != 1 else ''}" if msg.peers else "solo"
        self.query_one("#link-label", Static).update(f"[bold]Link: {msg.tempo:.1f} BPM  [{peers_str}][/]")

    def on_bpm_mode_changed(self, msg: BpmModeChanged) -> None:
        self.query_one("#mode-label", Static).update(
            f"[dim]Mode: [bold]{msg.mode.value}[/bold]  (m to cycle)[/]"
        )

    def check_action(self, action: str, parameters: tuple) -> bool | None:
        if action == "tap_beat":
            return self._bpm_mode is BpmMode.TAP
        return True

    # ── actions ────────────────────────────────────────────────────────────────

    def action_cycle_bpm_mode(self) -> None:
        order = [BpmMode.LINK, BpmMode.TAP, BpmMode.MIC]
        next_mode = order[(order.index(self._bpm_mode) + 1) % len(order)]

        # Stop mic detector when leaving MIC mode; restore audio level tracker
        if self._bpm_mode is BpmMode.MIC:
            self._beat_detector.stop()
            self._audio_level.resume()

        self._bpm_mode = next_mode
        self._tap_times = []

        # Start mic detector when entering MIC mode; pause audio level to avoid
        # competing for the same input device
        if self._bpm_mode is BpmMode.MIC:
            self._audio_level.pause()
            self._beat_detector.start()

        self.post_message(BpmModeChanged(next_mode))

    def action_tap_beat(self) -> None:
        if self._bpm_mode is not BpmMode.TAP:
            return
        now = time.monotonic()
        # Reset if the user paused for more than 2 seconds since the last tap
        if self._tap_times and now - self._tap_times[-1] > TAP_RESET_GAP_SECS:
            self._tap_times = []
        self._tap_times.append(now)
        if len(self._tap_times) < 2:
            return
        intervals = [self._tap_times[i+1] - self._tap_times[i] for i in range(len(self._tap_times)-1)]
        bpm = 60.0 / (sum(intervals) / len(intervals))
        self._link_clock.set_bpm(bpm)
        self.post_message(BpmChanged(bpm))


# ── GPT helper (runs in thread) ────────────────────────────────────────────────
def _chat_to_effect(text: str, history: list[dict], client: OpenAI) -> tuple[dict, list[dict]]:
    messages = [{"role": "system", "content": SYSTEM_PROMPT}] + history + [{"role": "user", "content": text}]
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
        response_format={"type": "json_object"},
    )
    raw = response.choices[0].message.content
    new_history = history + [
        {"role": "user", "content": text},
        {"role": "assistant", "content": raw},
    ]
    return json.loads(raw), new_history
