"""Sequence — a time-ordered list of effects played over musical phrases."""
from __future__ import annotations

from modules.effects import Effect, effect_from_dict

BEATS_PER_BAR: int = 4
DEFAULT_DURATION_BARS: float = 4.0
MAX_STEP_DURATION_BARS: float = 1.0
DEFAULT_REPEATS: int = 1
PHRASE_BEATS: int = BEATS_PER_BAR * 4   # 4 bars = 16 beats
MIN_REPEATS: int = 1
STEP_FREEZE_EPSILON: float = 1e-9


class Sequence(Effect):
    """
    Composite effect that cycles through steps in order.

    Each step is (effect, duration_beats). After playing through all steps
    `repeats` times, is_done() returns True and get_color() freezes on the
    last frame until the queue advances to the next sequence.
    """

    def __init__(
        self,
        steps: list[tuple[Effect, float]],
        repeats: int = DEFAULT_REPEATS,
        name: str = "",
    ) -> None:
        self._steps = steps
        self.total_beats: float = sum(d for _, d in steps) if steps else 0.0
        self.repeats: int = max(MIN_REPEATS, repeats)
        self.name: str = name

        # Cumulative beat positions: [0, d0, d0+d1, ...]
        self._starts: list[float] = []
        acc = 0.0
        for _, d in steps:
            self._starts.append(acc)
            acc += d

    def _active_idx_and_t(self, t: float) -> tuple[int, float]:
        """Return (step_index, t_within_step) for the given sequence time t."""
        clamped = min(t, self.total_beats * self.repeats - STEP_FREEZE_EPSILON)
        local_t = clamped % self.total_beats if self.total_beats > 0 else 0.0
        idx = 0
        for i, start in enumerate(self._starts):
            if local_t >= start:
                idx = i
        return idx, local_t - self._starts[idx]

    def get_color(self, t: float, led_index: int, total_leds: int) -> list[int]:
        if not self._steps:
            return [0, 0, 0]
        idx, t_within = self._active_idx_and_t(t)
        return self._steps[idx][0].get_color(t_within, led_index, total_leds)

    def on_beat(self, bpm: float, beat_number: int) -> None:
        # Propagate to all steps (safe — most on_beat are no-ops)
        for effect, _ in self._steps:
            effect.on_beat(bpm, beat_number)

    def is_done(self, t: float) -> bool:
        return t >= self.total_beats * self.repeats


# ── Factories ──────────────────────────────────────────────────────────────────

def sequence_from_dict(seq_cmd: dict) -> Sequence:
    """Build a Sequence from a SequenceCommand dict (as produced by GPT)."""
    steps = []
    for step in seq_cmd["steps"]:
        effect = effect_from_dict({
            "effect": step["effect"],
            "params": step.get("params", {}),
            "filters": step.get("filters", []),
        })
        duration_beats = step.get("duration_bars", DEFAULT_DURATION_BARS) * BEATS_PER_BAR
        steps.append((effect, duration_beats))
    return Sequence(
        steps=steps,
        repeats=seq_cmd.get("repeats", DEFAULT_REPEATS),
        name=seq_cmd.get("name", ""),
    )


def sequence_from_dict_capped(seq_cmd: dict, max_bars: float = MAX_STEP_DURATION_BARS) -> Sequence:
    """Like sequence_from_dict but caps each step's duration_bars to max_bars."""
    steps = []
    for step in seq_cmd["steps"]:
        effect = effect_from_dict({
            "effect": step["effect"],
            "params": step.get("params", {}),
            "filters": step.get("filters", []),
        })
        capped_bars = min(float(step.get("duration_bars", DEFAULT_DURATION_BARS)), max_bars)
        steps.append((effect, capped_bars * BEATS_PER_BAR))
    return Sequence(
        steps=steps,
        repeats=seq_cmd.get("repeats", DEFAULT_REPEATS),
        name=seq_cmd.get("name", ""),
    )


def sequence_from_effect_cmd(cmd: dict) -> Sequence:
    """Wrap a single EffectCommand dict as a 1-step Sequence (for presets/knobs)."""
    effect = effect_from_dict(cmd)
    return Sequence(
        steps=[(effect, DEFAULT_DURATION_BARS * BEATS_PER_BAR)],
        name=cmd.get("effect", ""),
    )
