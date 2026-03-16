import asyncio
import time
from dataclasses import asdict, dataclass
from typing import Any, Callable

import pyartnet
from modules.effects import Effect, SolidColor

LEDS_PER_UNIVERSE = 170
DEFAULT_ARTNET_FPS = 40


@dataclass(slots=True)
class RendererStats:
    ip: str
    target_fps: int
    actual_fps: float = 0.0
    frame_time_ms: float = 0.0
    compute_time_ms: float = 0.0
    send_time_ms: float = 0.0
    frames_rendered: int = 0
    frames_sent: int = 0
    frames_skipped: int = 0
    universes: int = 0
    payload_bytes: int = 0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class MultiRenderer:
    """Fan-out wrapper: broadcasts the same effect to multiple ArtNetRenderer instances."""

    def __init__(self, renderers: list["ArtNetRenderer"]) -> None:
        self._renderers = renderers

    def set_effect(self, effect: Effect) -> None:
        for r in self._renderers:
            r.set_effect(effect)

    def get_stats(self) -> list[dict[str, Any]]:
        return [r.get_stats() for r in self._renderers]

    async def render_loop(self) -> None:
        await asyncio.gather(*(r.render_loop() for r in self._renderers))


class ArtNetRenderer:
    def __init__(
        self,
        ip: str,
        total_leds: int,
        fps: int = DEFAULT_ARTNET_FPS,
        link_clock=None,
        stats_callback: Callable[[dict[str, Any]], None] | None = None,
    ):
        self._ip = ip
        self._total_leds = total_leds
        self._fps = fps
        self._effect: Effect = SolidColor(0, 0, 0)
        self._link_clock = link_clock
        self._beat_offset: float = link_clock.beat if link_clock else time.monotonic()
        self._stats_callback = stats_callback
        self._stats = RendererStats(ip=ip, target_fps=fps)

    def set_effect(self, effect: Effect) -> None:
        self._effect = effect
        self._beat_offset = self._link_clock.beat if self._link_clock else time.monotonic()

    def get_stats(self) -> dict[str, Any]:
        return self._stats.to_dict()

    async def render_loop(self) -> None:
        node = pyartnet.ArtNetNode.create(self._ip, max_fps=self._fps)
        async with node:
            # Build universe/channel list — split across universes for >170 LEDs
            universes = []
            remaining = self._total_leds
            universe_idx = 0
            led_offset = 0
            while remaining > 0:
                count = min(remaining, LEDS_PER_UNIVERSE)
                u = node.add_universe(universe_idx)
                ch = u.add_channel(start=1, width=count * 3)
                universes.append((u, ch, led_offset, count, None))
                remaining -= count
                led_offset += count
                universe_idx += 1
            self._stats.universes = len(universes)
            self._stats.payload_bytes = self._total_leds * 3

            interval = 1.0 / self._fps
            last_stats_at = time.perf_counter()
            loop_started_at = last_stats_at
            while True:
                frame_started_at = time.perf_counter()
                if self._link_clock:
                    t = self._link_clock.beat - self._beat_offset
                else:
                    t = time.monotonic() - self._beat_offset
                effect = self._effect
                compute_done_at = frame_started_at
                frames_sent_this_tick = 0
                next_universes = []
                for u, ch, offset, count, previous_values in universes:
                    values: list[int] = []
                    for i in range(count):
                        values.extend(effect.get_color(t, offset + i, self._total_leds))
                    compute_done_at = time.perf_counter()
                    if values == previous_values:
                        self._stats.frames_skipped += 1
                        next_universes.append((u, ch, offset, count, previous_values))
                        continue
                    ch.set_values(values)
                    u.send_data()
                    frames_sent_this_tick += 1
                    next_universes.append((u, ch, offset, count, values))
                universes = next_universes

                frame_finished_at = time.perf_counter()
                self._stats.frames_rendered += 1
                self._stats.frames_sent += frames_sent_this_tick
                self._stats.compute_time_ms = (compute_done_at - frame_started_at) * 1000.0
                self._stats.send_time_ms = (frame_finished_at - compute_done_at) * 1000.0
                self._stats.frame_time_ms = (frame_finished_at - frame_started_at) * 1000.0

                if frame_finished_at - last_stats_at >= 1.0:
                    elapsed = max(frame_finished_at - loop_started_at, 1e-6)
                    self._stats.actual_fps = self._stats.frames_rendered / elapsed
                    if self._stats_callback is not None:
                        self._stats_callback(self._stats.to_dict())
                    last_stats_at = frame_finished_at

                sleep_for = interval - (frame_finished_at - frame_started_at)
                if sleep_for > 0:
                    await asyncio.sleep(sleep_for)
                else:
                    await asyncio.sleep(0)
