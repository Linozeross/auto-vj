# auto-vj

Voice-controlled VJ lighting system. Hold Space, describe a vibe, release — GPT returns a structured light show and it plays live on your LED strips via ArtNet/WLED, perfectly locked to your Ableton Link session.

![Blade Runner aesthetic GUI](https://placeholder)

## How it works

1. **Hold Space** (or the PTT button in the GUI) to record your voice
2. **Release** — audio is transcribed by Whisper
3. **GPT-4o-mini** interprets the description and returns a sequence of LED effects as JSON
4. Effects render live over **ArtNet** to WLED at 100 fps, beat-time locked to **Ableton Link**

Conversation history is maintained, so relative commands work: *"make it faster"*, *"warmer colours"*, *"add a drop"*.

## Features

- **Push-to-talk voice control** — Whisper transcription, GPT effect generation
- **Sequence queue** — GPT returns multi-step sequences (buildup → drop → cooldown); sequences switch at 4-bar phrase boundaries for clean DJ cuts
- **Ableton Link sync** — all effect speeds are in cycles-per-beat; BPM changes automatically re-tempo everything
- **Auto-mode** — AI automatically refills the sequence queue when it runs low
- **Multi-universe ArtNet** — strips >170 LEDs span multiple ArtNet universes automatically
- **Two UIs**: a Textual TUI (`main.py`) and a PyQt6 GUI (`main_gui.py`) with knobs, presets, and Blade Runner aesthetics
- **BPM modes**: Ableton Link (network sync), tap tempo, or microphone beat detection
- **Composable filter pipeline** — stack `dim`, `gamma`, `mirror`, `reverse` on any effect

## Requirements

- Python 3.13+
- [uv](https://docs.astral.sh/uv/) package manager
- OpenAI API key (for Whisper + GPT-4o-mini)
- One or more [WLED](https://kno.wled.ge/) devices on your local network
- macOS: terminal app needs **Accessibility permissions** (System Settings → Privacy & Security → Accessibility) for push-to-talk via pynput

## Setup

```bash
# Clone and install
git clone https://github.com/Linozeross/auto-vj.git
cd auto-vj
uv sync

# Configure
cp .env.example .env
# Edit .env — set your OPENAI_API_KEY and WLED IP/LED count
```

**.env** options:

```env
OPENAI_API_KEY=sk-...

# Single WLED target
ARTNET_IP=192.168.1.100
LED_COUNT=100

# Multiple WLED targets (overrides ARTNET_IP / LED_COUNT)
# ARTNET_NODES=192.168.1.100:100,192.168.1.101:60
```

## Running

```bash
# PyQt6 GUI (recommended)
uv run python main_gui.py

# Textual TUI
uv run python main.py
```

### GUI controls

| Control | Action |
|---|---|
| Hold **Space** | Record voice |
| Click **PTT** button | Same as Space |
| **Preset buttons** (1–9) | Instant preset recall (phrase-aligned) |
| Right-click preset | Save current effect to slot |
| **BPM knob** | Adjust tempo (propagates to Ableton Link) |
| **M key** | Cycle BPM mode: Link → Tap → Mic |
| **Tap** button | Tap tempo |
| **Auto** toggle | Enable AI auto-refill mode |

## Effects

All `speed` / `rate` values are **cycles-per-beat** — `1.0` = one cycle per beat at current BPM. Effects automatically track tempo changes.

| Effect | Params | Description |
|---|---|---|
| `pulse` | `r, g, b, rate` (0.1–8.0) | Breathing glow / strobe |
| `rainbow` | `speed` (0.1–4.0) | Full-spectrum hue sweep |
| `chase` | `r, g, b, speed, tail` | Moving beam |
| `twinkle` | `r, g, b, density, speed` | Deterministic star field |
| `plasma` | `speed, scale` | Psychedelic overlapping sine waves → HSV |
| `palette_wave` | `palette, speed` | Mood colour flow (`ember` / `ocean` / `violet` / `sunset`) |
| `beat_pulse` | `r, g, b, sharpness` | Sharp flash on every beat |

## Filters

Stack filters on any effect. Applied innermost-first (left to right in JSON).

| Filter | Params | Effect |
|---|---|---|
| `dim` | `brightness` (0.0–1.0) | Scale all channels |
| `gamma` | `gamma` | Perceptual gamma correction |
| `mirror` | — | Symmetric reflection from center |
| `reverse` | — | Flip LED order |

## Voice command examples

> *"Blue ocean waves, slow and chill"*
> *"Build up to a rave drop — start slow, end with fast rainbow"*
> *"Red and white police lights at 140 BPM"*
> *"Make it warmer"* (resolves relative to current effect)
> *"Add a drop after this buildup"*

**Shorthands** GPT understands:

| Word | Effect |
|---|---|
| strobe | white pulse rate=4.0 |
| police | red chase speed=2.0 |
| rave | rainbow speed=2.0 |
| sparkle | white twinkle density=0.3 |
| beat | orange beat_pulse sharpness=4.0 |
| chill | ocean palette_wave + dim 0.6 |
| fire | ember palette_wave speed=0.5 |

## Architecture

```
auto-vj/
├── main.py                  # TUI entry point
├── main_gui.py              # GUI entry point
├── system_prompt.txt        # GPT schema & effect reference
├── presets.json             # Saved GUI presets
├── modules/
│   ├── effects.py           # Effect + Filter ABCs, all implementations, factory
│   ├── sequences.py         # Sequence(Effect) — phrase-aligned multi-step playback
│   ├── bpm.py               # LinkClock — Ableton Link beat source
│   ├── artnet_renderer.py   # Multi-universe ArtNet send loop (100 fps)
│   ├── recorder.py          # PTT + Whisper transcription
│   ├── tui.py               # Textual TUI app
│   └── gui_app.py           # PyQt6 GUI app
└── tests/
    ├── test_effects.py      # Effect + filter + factory + schema tests
    └── test_bpm.py          # LinkClock async tests
```

**Beat-time rendering**: `t` passed to every `get_color()` call is in **beats**, not seconds. This means `speed=1.0` is always one cycle per beat regardless of BPM — the renderer handles the mapping automatically.

**Sequence switching**: sequences queue up and switch at 16-beat (4-bar) phrase boundaries, matching how DJs structure music. The sequence queue is visible in the GUI with a playhead.

**Ableton Link**: tempo and beat position come from [aalink](https://github.com/nicholasgasior/aalink) (Python bindings for the Link SDK). Any Link-compatible app on the same network (Ableton Live, Traktor, djay, etc.) will sync automatically.

## Testing

```bash
uv run pytest tests/ -v           # all tests
uv run pytest tests/test_effects.py -v   # effects + filters
uv run pytest tests/test_bpm.py -v       # LinkClock async tests

# Quick effect smoke-test (no hardware)
python -c "from modules.effects import effect_from_dict; e = effect_from_dict({'effect':'pulse','params':{'r':255,'g':0,'b':0,'rate':1.0}}); print(e.get_color(0.5, 5, 100))"

# Check for Ableton Link peers on the network
uv run python test_link.py
```

## Dependencies

| Package | Purpose |
|---|---|
| `openai` | Whisper transcription + GPT-4o-mini |
| `sounddevice` | Microphone recording |
| `numpy` | Audio processing, BPM detection |
| `pynput` | Global keyboard hook for PTT (TUI) |
| `pyartnet` | ArtNet UDP protocol |
| `aalink` | Ableton Link integration |
| `textual` | TUI framework |
| `pyqt6` | GUI framework |
| `pydantic` | GPT response schema validation |
| `python-dotenv` | .env loading |

## License

MIT
