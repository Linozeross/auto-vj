# AGENTS.md

This file provides guidance to Codex (Codex.ai/code) when working with code in this repository.

## Project

Push-to-talk VJ lighting controller: hold Space to record, release to transcribe (Whisper) → GPT returns structured LED effect command → rendered live to WLED via ArtNet. Features a Textual TUI, conversation history context, composable filter pipeline, and Ableton Link BPM sync.

## Feature branching workflow

Each new feature is developed on a dedicated git branch (e.g. `feature/my-feature`). When the feature is complete, ask the user whether to save anything to memory, commit, and merge into main. Do not do any of these without explicit confirmation.

## Commands

```bash
# Install dependencies
uv sync

# Run
uv run python main.py

# Run tests
uv run pytest tests/ -v

# Add a dependency
uv add <package>

# Quick effect smoke-test (no hardware)
python -c "from modules.effects import effect_from_dict; e = effect_from_dict({'effect':'fire','params':{}}); print(e.get_color(0.1, 5, 100))"

# Check for Ableton Link peers on the network
uv run python test_link.py
```

Copy `.env.example` to `.env` and fill in all three vars before running.

## Testing policy

Whenever you change or add code, also create or update a corresponding test in `tests/`. Keep tests simple — a single `assert` covering the new behaviour is enough. Run `uv run pytest tests/ -v` to verify before finishing.

## Code style

**Named constants for all magic values**: any literal number or string used in logic must be extracted into a named constant at the top of the file (or module level). Do not inline magic values in functions or conditions. Example: `TAP_RESET_GAP_SECS = 2.0` used as `if gap > TAP_RESET_GAP_SECS`, never `if gap > 2.0`.

## Architecture

```
auto-vj/
├── main.py                    # 2-line entry point: VJApp().run()
├── system_prompt.txt          # GPT schema: all effects, filters, BPM field
├── tests/
│   ├── test_effects.py        # effect + filter + factory + schema tests
│   └── test_bpm.py            # LinkClock async tests (pytest-asyncio)
└── modules/
    ├── recorder.py            # PTT + Whisper
    ├── effects.py             # Effect ABC, Filter ABC, all effects, registries, factory
    ├── bpm.py                 # LinkClock — Ableton Link beat source, fires on_beat() at tempo
    ├── artnet_renderer.py     # ArtNetRenderer — multi-universe ArtNet send loop
    └── tui.py                 # Textual TUI app + GPT chat helper
```

---

### modules/recorder.py
- `record_until_release()` — `pynput` Space hold/release, `sounddevice` int16 at 16 kHz
- `transcribe(audio, client)` — WAV in-memory → `whisper-1`
- Both synchronous; called via `asyncio.to_thread()` in tui.py

---

### modules/effects.py

#### Base classes
- `Effect` (ABC) — `get_color(t, led_index, total_leds) -> [r,g,b]`, `on_beat(bpm, beat_number)` no-op hook
- `Filter(Effect)` — decorator base; wraps an inner `Effect`, propagates `on_beat`

#### Concrete effects (11 total)
| Name | Class | Key params |
|---|---|---|
| `solid` | `SolidColor` | `r, g, b` |
| `color_wave` | `ColorWave` | `speed, saturation, brightness` |
| `pulse` | `Pulse` | `r, g, b, rate` (`rate_hz` accepted as alias) |
| `rainbow` | `Rainbow` | `speed` |
| `chase` | `Chase` | `r, g, b, speed, tail` |
| `meteor` | `Meteor` | `r, g, b, speed, tail_length, tail_decay, bounce` |
| `twinkle` | `Twinkle` | `r, g, b, density, speed` — deterministic per-LED phase |
| `fire` | `Fire` | `cooling, sparking, reverse` — stateful heat array, updated once/frame |
| `plasma` | `Plasma` | `speed, scale` — overlapping sine waves → HSV |
| `larson` | `Larson` | `r, g, b, speed, width` — KITT Gaussian scanner |
| `beat_flash` | `BeatFlash` | `r, g, b, decay` — flashes on `on_beat()`, decays between |

#### Concrete filters (4 total)
| Name | Class | Effect |
|---|---|---|
| `gamma` | `GammaFilter` | `(v/255)^(1/gamma)*255` per channel |
| `mirror` | `MirrorFilter` | symmetric reflection from center |
| `reverse` | `ReverseFilter` | flip LED order |
| `dim` | `DimFilter` | scale all channels by `brightness` factor |

#### GPT schema (Pydantic)
```python
class FilterCommand(BaseModel):
    type: Literal["gamma", "mirror", "reverse", "dim"]
    params: dict[str, Any] = {}

class EffectCommand(BaseModel):
    effect: Literal["solid", "color_wave", "pulse", "rainbow", "chase",
                    "meteor", "twinkle", "fire", "plasma", "larson", "beat_flash"]
    params: dict[str, Any] = {}
    filters: list[FilterCommand] = []
    bpm: float | None = None
```

#### Registries & factory
- `EFFECT_REGISTRY: dict[str, type[Effect]]`
- `FILTER_REGISTRY: dict[str, type[Filter]]`
- `effect_from_dict(d)` — instantiates effect, wraps with filters in order (first listed = innermost)

---

### modules/bpm.py
- `LinkClock` — wraps `aalink.Link`; connects to Ableton Link session on the network
- `start()` / `stop()` — call from async context (Textual `on_mount`); starts `_tick_loop`
- `_tick_loop()` — `await link.sync(1)` fires at every beat boundary, then calls `effect.on_beat(tempo, beat_number)`
- `set_bpm(bpm)` — clamps to 20–300, writes to `link.tempo` (propagates to all Link peers)
- `attach_effect(effect)` — called after each `effect_from_dict` in `_ptt_loop`
- `beat` property — current Link beat position (monotonically increasing float)
- `tempo` property — current BPM from Link (may differ from what was set if a peer is connected)
- `num_peers` property — number of connected Ableton Link peers (0 = solo)
- `beat_phase() -> float` — 0.0→1.0 within current beat, read from `link.phase`
- `set_tempo_callback(cb)` / `set_num_peers_callback(cb)` — register callbacks for Link events (fired from C++ thread — use `post_message`, not `query_one`)

---

### modules/artnet_renderer.py
- `ArtNetRenderer(ip, total_leds, fps=100, link_clock=None)`
- `set_effect(effect)` — swap effect; stores `beat_offset = link_clock.beat` so each new effect starts at `t≈0`
- `render_loop()` — async; `t = link_clock.beat - beat_offset` per frame (beat-time); falls back to `time.monotonic()` if no clock (used in tests); calls `get_color(t, global_led_index, total_leds)` per LED per frame
- **Beat-time `t`**: `t` is in **beats** (not seconds). `speed=1.0` in any effect = one cycle per beat. Effects automatically track BPM changes with no code changes.
- **Multi-universe**: strips >170 LEDs automatically use consecutive ArtNet universes (0, 1, 2, ...)

---

### modules/tui.py
- `VJApp(textual.App)` — Textual TUI; two-panel layout: history table (left) + status/effect/BPM/Link (right)
- `_ptt_loop(renderer, client)` — PTT → transcribe → GPT → `effect_from_dict` → `link_clock.attach_effect` → `renderer.set_effect`; maintains `history: list[dict]` for GPT conversation context
- `_chat_to_effect(text, history, client)` — sends `[system] + history + [user]` to GPT; returns `(cmd_dict, new_history)`
- Messages: `StatusChanged` (`WAITING/RECORDING/TRANSCRIBING/THINKING/LIVE`), `EffectActivated`, `BpmChanged`, `LinkStatusChanged`
- `LinkStatusChanged(tempo, peers)` — posted from `set_tempo_callback` / `set_num_peers_callback`; updates Link status label
- Link callbacks run in aalink's C++ thread — always use `post_message()`, never call `query_one()` directly from them
- `EFFECT_COLORS` — maps each effect name to a Textual/Rich color string

---

### main.py
```python
from modules.tui import VJApp
if __name__ == "__main__":
    VJApp().run()
```

---

### system_prompt.txt
Instructs GPT to return JSON only, matching `EffectCommand` schema. Contains:
- All 11 effect names with parameter ranges
- All 4 filter types
- BPM field instructions (syncs to Ableton Link)
- **Speed semantics**: all `speed` / `rate` values are **cycles-per-beat** (1.0 = one cycle per beat). Speed mapping: `slow→0.25`, `medium→1.0`, `fast→2.0`, `very fast→4.0`
- Color name → RGB table, preset table (police, rave, chill, fire, ocean, strobe, scanner, meteor, sparkle, beat, plasma)
- History context instruction (resolves relative commands like "make it faster")

---

## Key config

| Variable | File | Default |
|---|---|---|
| `PTT_KEY` | `modules/recorder.py` | `keyboard.Key.space` |
| `SAMPLE_RATE` | `modules/recorder.py` | 16000 Hz |
| `ARTNET_IP` | `.env` | `192.168.178.102` |
| `LED_COUNT` | `.env` | `100` |
| `LEDS_PER_UNIVERSE` | `modules/artnet_renderer.py` | `170` |
| Chat model | `modules/tui.py` `_chat_to_effect()` | `gpt-4o-mini` |
| BPM clamp | `modules/bpm.py` `set_bpm()` | 20–300 BPM |
| Link initial BPM | `modules/bpm.py` `LinkClock.__init__` | 120 BPM |

## Testing

```bash
uv run pytest tests/ -v          # all 47 tests
uv run pytest tests/test_effects.py -v   # effects + filters only
uv run pytest tests/test_bpm.py -v       # LinkClock async tests
```

Test suite uses `pytest` + `pytest-asyncio` (strict mode, configured in `pyproject.toml`).

## macOS note

`pynput` requires accessibility permissions. If you see "This process is not trusted!", add your terminal app to **System Settings → Privacy & Security → Accessibility**.
