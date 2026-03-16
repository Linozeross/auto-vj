import asyncio
from enum import Enum
from typing import Callable

import aalink

from modules.effects import Effect


class BpmMode(Enum):
    LINK = "link"
    TAP  = "tap"
    MIC  = "mic"
    AUDIO = "audio"


class LinkClock:
    def __init__(self) -> None:
        self._link = aalink.Link(120.0)
        self._link.enabled = True
        self._link.quantum = 1.0
        self._link.playing = True
        self._effect: Effect | None = None
        self._task: asyncio.Task | None = None
        self._beat_number: int = 0
        self._running: bool = False

    @property
    def beat(self) -> float:
        """Current Link beat position (monotonically increasing)."""
        return self._link.beat

    @property
    def tempo(self) -> float:
        return self._link.tempo

    @property
    def num_peers(self) -> int:
        return self._link.num_peers

    def beat_phase(self, _t: float = 0.0) -> float:
        """0.0→1.0 position within the current beat."""
        return self._link.phase % 1.0

    def attach_effect(self, effect: Effect) -> None:
        self._effect = effect

    def set_bpm(self, bpm: float) -> None:
        self._link.tempo = max(20.0, min(300.0, bpm))

    def set_tempo_callback(self, cb: Callable[[float], None]) -> None:
        self._link.set_tempo_callback(cb)

    def set_num_peers_callback(self, cb: Callable[[int], None]) -> None:
        self._link.set_num_peers_callback(cb)

    def start(self) -> None:
        self._running = True
        self._task = asyncio.ensure_future(self._tick_loop())

    def stop(self) -> None:
        self._running = False
        if self._task and not self._task.done():
            self._task.cancel()

    async def _tick_loop(self) -> None:
        while self._running:
            await self._link.sync(1)
            self._beat_number += 1
            if self._effect is not None:
                self._effect.on_beat(self._link.tempo, self._beat_number)
