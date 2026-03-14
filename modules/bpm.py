import asyncio
import time
from modules.effects import Effect


class BpmClock:
    def __init__(self) -> None:
        self._bpm: float = 120.0
        self._effect: Effect | None = None
        self._task: asyncio.Task | None = None
        self._beat_number: int = 0
        self._running: bool = False

    @property
    def beat_number(self) -> int:
        return self._beat_number

    def beat_phase(self, t: float) -> float:
        """0.0→1.0 position within the current beat."""
        interval = 60.0 / self._bpm
        return (t % interval) / interval

    def attach_effect(self, effect: Effect) -> None:
        self._effect = effect

    def set_bpm(self, bpm: float) -> None:
        self._bpm = max(20.0, min(300.0, bpm))
        self._beat_number = 0
        if self._task and not self._task.done():
            self._task.cancel()
        if self._running:
            self._task = asyncio.ensure_future(self._tick_loop())

    def start(self) -> None:
        self._running = True
        self._task = asyncio.ensure_future(self._tick_loop())

    def stop(self) -> None:
        self._running = False
        if self._task and not self._task.done():
            self._task.cancel()

    async def _tick_loop(self) -> None:
        while self._running:
            await asyncio.sleep(60.0 / self._bpm)
            self._beat_number += 1
            if self._effect is not None:
                self._effect.on_beat(self._bpm, self._beat_number)
