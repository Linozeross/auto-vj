"""Tests for modules/bpm.py — BpmClock."""
import asyncio
import pytest
from modules.bpm import BpmClock
from modules.effects import BeatFlash


@pytest.mark.asyncio
async def test_bpm_clock_fires_expected_beats():
    clock = BpmClock()
    flash = BeatFlash(r=255, g=0, b=0, decay=0.3)
    clock.attach_effect(flash)
    clock.set_bpm(300)  # max=300 → 5 beats/second (0.2s interval)
    clock.start()
    await asyncio.sleep(1.1)  # expect ~5 beats
    clock.stop()
    assert 4 <= clock.beat_number <= 7


@pytest.mark.asyncio
async def test_bpm_clock_set_bpm_restarts():
    clock = BpmClock()
    clock.start()
    clock.set_bpm(120)
    await asyncio.sleep(0.1)
    clock.stop()
    assert clock._bpm == 120.0


@pytest.mark.asyncio
async def test_bpm_clock_clamps_bpm():
    clock = BpmClock()
    clock.set_bpm(5)     # below min
    assert clock._bpm == 20.0
    clock.set_bpm(9999)  # above max
    assert clock._bpm == 300.0


@pytest.mark.asyncio
async def test_bpm_clock_attach_effect_receives_on_beat():
    clock = BpmClock()
    flash = BeatFlash(r=255, g=0, b=0, decay=0.3)
    clock.attach_effect(flash)
    clock.set_bpm(300)  # 0.2s per beat
    clock.start()
    await asyncio.sleep(0.4)  # wait for at least 1 beat
    clock.stop()
    c = flash.get_color(0, 0, 100)
    assert c[0] > 0, "BeatFlash should have been triggered by BpmClock"


@pytest.mark.asyncio
async def test_bpm_clock_stop_halts_beats():
    clock = BpmClock()
    clock.set_bpm(600)
    clock.start()
    await asyncio.sleep(0.1)
    clock.stop()
    count_at_stop = clock.beat_number
    await asyncio.sleep(0.3)
    assert clock.beat_number == count_at_stop, "Beat count should not increase after stop"


def test_beat_phase_range():
    clock = BpmClock()
    clock.set_bpm(120)
    for t in [0.0, 0.1, 0.25, 0.49, 0.5]:
        phase = clock.beat_phase(t)
        assert 0.0 <= phase < 1.0, f"beat_phase({t}) = {phase} out of [0, 1)"
