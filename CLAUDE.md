# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

Push-to-talk VJ lighting controller: hold Space to record, release to transcribe (Whisper) → GPT returns structured LED effect command → rendered live to WLED via ArtNet. Features a Textual TUI, conversation history context, composable filter pipeline, and BPM clock.

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
```

Copy `.env.example` to `.env` and fill in all three vars before running.

## Architecture

```
auto-vj/
├── main.py                    # 2-line entry point: VJApp().run()
├── system_prompt.txt          # GPT schema: all effects, filters, BPM field
├── tests/
│   ├── test_effects.py        # effect + filter + factory + schema tests
│   └── test_bpm.py            # BpmClock async tests (pytest-asyncio)
└── modules/
    ├── recorder.py            # PTT + Whisper
    ├── effects.py             # Effect ABC, Filter ABC, all effects, registries, factory
    ├── bpm.py                 # BpmClock — fires on_beat() at tempo
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
| `pulse` | `Pulse` | `r, g, b, rate_hz` |
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
- `BpmClock` — async tick loop that calls `effect.on_beat(bpm, beat_number)` at the right interval
- `set_bpm(bpm)` — clamps to 20–300 BPM, cancels and restarts tick task immediately
- `start()` / `stop()` — call from async context (Textual `on_mount`)
- `attach_effect(effect)` — called after each `effect_from_dict` in `_ptt_loop`
- `beat_phase(t) -> float` — 0.0→1.0 within current beat

---

### modules/artnet_renderer.py
- `ArtNetRenderer(ip, total_leds, fps=100)`
- `set_effect(effect)` — swap effect, reset `t=0`
- `render_loop()` — async; auto-splits across universes at 170 LEDs/universe; calls `get_color(t, global_led_index, total_leds)` per LED per frame
- **Multi-universe**: strips >170 LEDs automatically use consecutive ArtNet universes (0, 1, 2, ...)

---

### modules/tui.py
- `VJApp(textual.App)` — Textual TUI; two-panel layout: history table (left) + status/effect/BPM (right)
- `_ptt_loop(renderer, client)` — PTT → transcribe → GPT → `effect_from_dict` → `bpm_clock.attach_effect` → `renderer.set_effect`; maintains `history: list[dict]` for GPT conversation context
- `_chat_to_effect(text, history, client)` — sends `[system] + history + [user]` to GPT; returns `(cmd_dict, new_history)`
- Messages: `StatusChanged` (`WAITING/RECORDING/TRANSCRIBING/THINKING/LIVE`), `EffectActivated`, `BpmChanged`
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
- BPM field instructions
- Color name → RGB table, speed mapping, preset table (police, rave, chill, fire, ocean, strobe, scanner, meteor, sparkle, beat, plasma)
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

## Testing

```bash
uv run pytest tests/ -v          # all 45 tests
uv run pytest tests/test_effects.py -v   # effects + filters only
uv run pytest tests/test_bpm.py -v       # BpmClock async tests
```

Test suite uses `pytest` + `pytest-asyncio` (strict mode, configured in `pyproject.toml`).

## macOS note

`pynput` requires accessibility permissions. If you see "This process is not trusted!", add your terminal app to **System Settings → Privacy & Security → Accessibility**.
