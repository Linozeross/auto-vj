import asyncio
import time
import pyartnet
from modules.effects import Effect, SolidColor

LEDS_PER_UNIVERSE = 170


class ArtNetRenderer:
    def __init__(self, ip: str, total_leds: int, fps: int = 100, link_clock=None):
        self._ip = ip
        self._total_leds = total_leds
        self._fps = fps
        self._effect: Effect = SolidColor(0, 0, 0)
        self._link_clock = link_clock
        self._beat_offset: float = link_clock.beat if link_clock else time.monotonic()

    def set_effect(self, effect: Effect) -> None:
        self._effect = effect
        self._beat_offset = self._link_clock.beat if self._link_clock else time.monotonic()

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
                universes.append((u, ch, led_offset, count))
                remaining -= count
                led_offset += count
                universe_idx += 1

            interval = 1.0 / self._fps
            while True:
                if self._link_clock:
                    t = self._link_clock.beat - self._beat_offset
                else:
                    t = time.monotonic() - self._beat_offset
                effect = self._effect
                for u, ch, offset, count in universes:
                    values = []
                    for i in range(count):
                        values.extend(effect.get_color(t, offset + i, self._total_leds))
                    ch.set_values(values)
                    u.send_data()
                await asyncio.sleep(interval)
