"""Tests for modules/bpm.py — LinkClock."""
import asyncio
import pytest
from modules.bpm import LinkClock
from modules.effects import BeatFlash


@pytest.mark.asyncio
async def test_link_clock_fires_beats():
    clock = LinkClock()
    flash = BeatFlash(r=255, g=0, b=0, decay=0.3)
    clock.attach_effect(flash)
    clock.set_bpm(300)  # 300 BPM → 5 beats/second
    clock.start()
    await asyncio.sleep(1.3)  # expect ~5 beats (first sync waits up to 1 beat boundary)
    clock.stop()
    assert 3 <= clock._beat_number <= 8


@pytest.mark.asyncio
async def test_link_clock_set_bpm():
    clock = LinkClock()
    clock.set_bpm(140)
    assert clock.tempo == pytest.approx(140.0, abs=0.1)


@pytest.mark.asyncio
async def test_link_clock_clamps_bpm():
    clock = LinkClock()
    clock.set_bpm(5)     # below min
    assert clock.tempo == pytest.approx(20.0, abs=0.1)
    clock.set_bpm(9999)  # above max
    assert clock.tempo == pytest.approx(300.0, abs=0.1)


@pytest.mark.asyncio
async def test_link_clock_attach_effect_receives_on_beat():
    clock = LinkClock()
    flash = BeatFlash(r=255, g=0, b=0, decay=0.3)
    clock.attach_effect(flash)
    clock.set_bpm(300)  # 0.2s per beat
    clock.start()
    await asyncio.sleep(0.4)  # wait for at least 1 beat
    clock.stop()
    c = flash.get_color(0, 0, 100)
    assert c[0] > 0, "BeatFlash should have been triggered by on_beat()"


@pytest.mark.asyncio
async def test_link_clock_stop_halts_beats():
    clock = LinkClock()
    clock.set_bpm(300)
    clock.start()
    await asyncio.sleep(0.3)
    clock.stop()
    count_at_stop = clock._beat_number
    await asyncio.sleep(0.3)
    assert clock._beat_number == count_at_stop, "Beat count should not increase after stop"


@pytest.mark.asyncio
async def test_beat_phase_range():
    clock = LinkClock()
    clock.set_bpm(120)
    clock.start()
    await asyncio.sleep(0.05)
    for _ in range(5):
        phase = clock.beat_phase()
        assert 0.0 <= phase < 1.0, f"beat_phase() = {phase} out of [0, 1)"
        await asyncio.sleep(0.01)
    clock.stop()


@pytest.mark.asyncio
async def test_link_clock_beat_property():
    clock = LinkClock()
    b0 = clock.beat
    assert isinstance(b0, float)
    assert b0 >= 0.0
    clock.start()
    await asyncio.sleep(0.1)
    b1 = clock.beat
    assert b1 > b0, "beat should increase over time"
    clock.stop()
