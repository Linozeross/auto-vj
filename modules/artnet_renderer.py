import asyncio
import time
import pyartnet
from modules.effects import Effect, SolidColor


class ArtNetRenderer:
    def __init__(self, ip: str, total_leds: int, fps: int = 100):
        self._ip = ip
        self._total_leds = total_leds
        self._fps = fps
        self._effect: Effect = SolidColor(0, 0, 0)
        self._start_time: float = time.monotonic()

    def set_effect(self, effect: Effect) -> None:
        self._effect = effect
        self._start_time = time.monotonic()

    async def render_loop(self) -> None:
        node = pyartnet.ArtNetNode.create(self._ip, max_fps=self._fps)
        async with node:
            universe = node.add_universe(0)
            channel = universe.add_channel(start=1, width=self._total_leds * 3)

            interval = 1.0 / self._fps
            while True:
                t = time.monotonic() - self._start_time
                values = []
                effect = self._effect
                for i in range(self._total_leds):
                    values.extend(effect.get_color(t, i, self._total_leds))
                channel.set_values(values)
                universe.send_data()
                await asyncio.sleep(interval)
