# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

Push-to-talk VJ lighting controller: hold Space to record, release to transcribe (Whisper) → GPT returns structured LED effect command → rendered live to WLED via ArtNet.

## Commands

```bash
# Install dependencies
uv sync

# Run
uv run python main.py

# Add a dependency
uv add <package>

# Test effect system without hardware
python -c "from modules.effects import effect_from_dict; e = effect_from_dict({'effect':'rainbow','params':{'speed':2}}); print(e.get_color(0.5, 10, 100))"
```

Copy `.env.example` to `.env` and fill in all three vars before running.

## Architecture

```
auto-vj/
├── main.py                    # async orchestrator (TaskGroup: render_loop + ptt_loop)
├── system_prompt.txt          # effect JSON schema instructions for GPT
└── modules/
    ├── recorder.py            # PTT + Whisper (record_until_release, transcribe)
    ├── effects.py             # Effect base class, concrete effects, GPT schema, factory
    └── artnet_renderer.py     # ArtNetRenderer: set_effect + render_loop
```

### modules/recorder.py
- `record_until_release()` — `pynput` Space hold/release, `sounddevice` int16 at 16 kHz
- `transcribe(audio, client)` — WAV in-memory → `whisper-1`
- Both synchronous; called via `asyncio.to_thread()` in main.py

### modules/effects.py
- `Effect` (ABC) — `get_color(t, led_index, total_leds) -> [r,g,b]`, `on_beat()` hook
- Concrete: `SolidColor`, `ColorWave`, `Pulse`, `Rainbow`, `Chase`
- `EffectCommand` (Pydantic) — structured output schema for GPT
- `effect_from_dict(d)` — factory

### modules/artnet_renderer.py
- `ArtNetRenderer(ip, total_leds, fps=100)`
- `set_effect(effect)` — swap effect, reset t=0
- `render_loop()` — async; computes all LED colors each frame, sends via pyartnet

### main.py
- `chat_to_effect(text)` — GPT structured parse → `{"effect": ..., "params": ...}`
- `ptt_loop(renderer)` — record → transcribe → chat_to_effect → set_effect
- `main()` — TaskGroup runs render_loop + ptt_loop concurrently

**System prompt** (`system_prompt.txt`) instructs GPT to return only EffectCommand JSON.

## Key config

| Variable | File | Default |
|---|---|---|
| `PTT_KEY` | `modules/recorder.py` | `keyboard.Key.space` |
| `SAMPLE_RATE` | `modules/recorder.py` | 16000 Hz |
| `ARTNET_IP` | `.env` | `192.168.178.102` |
| `LED_COUNT` | `.env` | `100` |
| Chat model | `main.py` `chat_to_effect()` | `gpt-4o-mini` |
| System prompt | `system_prompt.txt` | — |
