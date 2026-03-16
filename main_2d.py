from __future__ import annotations

import argparse
import time
from dataclasses import dataclass

import cv2

from modules.artnet_renderer import DEFAULT_ARTNET_FPS, StaticArtNetSender
from modules.mapping_loader import DEFAULT_MAPPING_PATH, load_mapping
from modules.render_2d import (
    DiagonalBeams2D,
    DEFAULT_RENDER_FPS,
    HorizontalWave2D,
    SpiralRings2D,
    VerticalWave2D,
    monotonic_seconds,
    render_led_frame,
    render_preview,
)

QUIT_KEY = "q"
ESC_KEY_CODE = 27
PREVIEW_WINDOW_TITLE = "AutoVJ 2D Preview"
HORIZONTAL_EFFECT_NAME = "horizontal"
VERTICAL_EFFECT_NAME = "vertical"
SPIRAL_EFFECT_NAME = "spiral"
BEAMS_EFFECT_NAME = "beams"
TELEMETRY_INTERVAL_SECS = 1.0


@dataclass(slots=True)
class LoopTelemetry:
    started_at: float
    last_report_at: float
    frames: int = 0
    render_ms_total: float = 0.0
    send_ms_total: float = 0.0
    preview_ms_total: float = 0.0
    loop_ms_total: float = 0.0

    def record(self, render_ms: float, send_ms: float, preview_ms: float, loop_ms: float) -> None:
        self.frames += 1
        self.render_ms_total += render_ms
        self.send_ms_total += send_ms
        self.preview_ms_total += preview_ms
        self.loop_ms_total += loop_ms

    def should_report(self, now: float) -> bool:
        return now - self.last_report_at >= TELEMETRY_INTERVAL_SECS

    def report(self, now: float) -> None:
        elapsed = max(now - self.started_at, 1e-6)
        avg_render = self.render_ms_total / max(self.frames, 1)
        avg_send = self.send_ms_total / max(self.frames, 1)
        avg_preview = self.preview_ms_total / max(self.frames, 1)
        avg_loop = self.loop_ms_total / max(self.frames, 1)
        fps = self.frames / elapsed
        print(
            "2D telemetry:"
            f" fps={fps:.1f}"
            f" render={avg_render:.2f}ms"
            f" send={avg_send:.2f}ms"
            f" preview={avg_preview:.2f}ms"
            f" loop={avg_loop:.2f}ms"
        )
        self.last_report_at = now


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render a minimal 2D effect over a saved LED mapping.")
    parser.add_argument("--mapping", default=str(DEFAULT_MAPPING_PATH), help="Path to mapping JSON.")
    parser.add_argument(
        "--effect",
        choices=[HORIZONTAL_EFFECT_NAME, VERTICAL_EFFECT_NAME, SPIRAL_EFFECT_NAME, BEAMS_EFFECT_NAME],
        default=BEAMS_EFFECT_NAME,
        help="2D effect direction.",
    )
    parser.add_argument("--fps", type=int, default=DEFAULT_RENDER_FPS, help="Preview/send FPS.")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    mapping = load_mapping(args.mapping)
    if args.effect == HORIZONTAL_EFFECT_NAME:
        effect = HorizontalWave2D()
    elif args.effect == VERTICAL_EFFECT_NAME:
        effect = VerticalWave2D()
    elif args.effect == SPIRAL_EFFECT_NAME:
        effect = SpiralRings2D()
    else:
        effect = DiagonalBeams2D()
    sender = StaticArtNetSender(mapping.artnet.ip, mapping.artnet.led_count, fps=DEFAULT_ARTNET_FPS)
    frame_interval = 1.0 / max(args.fps, 1)
    started_at = monotonic_seconds()
    telemetry = LoopTelemetry(started_at=started_at, last_report_at=started_at)

    try:
        while True:
            loop_started_at = monotonic_seconds()
            t = loop_started_at - started_at
            render_started_at = monotonic_seconds()
            rgb_values = render_led_frame(effect, mapping, t)
            render_done_at = monotonic_seconds()
            sender.send_frame(rgb_values)
            send_done_at = monotonic_seconds()
            preview = render_preview(effect, mapping, t, f"2D {args.effect}")
            preview_done_at = monotonic_seconds()
            cv2.imshow(PREVIEW_WINDOW_TITLE, preview)
            key_code = cv2.waitKey(1) & 0xFF
            if key_code == ord(QUIT_KEY) or key_code == ESC_KEY_CODE:
                break
            loop_done_at = monotonic_seconds()
            telemetry.record(
                render_ms=(render_done_at - render_started_at) * 1000.0,
                send_ms=(send_done_at - render_done_at) * 1000.0,
                preview_ms=(preview_done_at - send_done_at) * 1000.0,
                loop_ms=(loop_done_at - loop_started_at) * 1000.0,
            )
            if telemetry.should_report(loop_done_at):
                telemetry.report(loop_done_at)
            elapsed = loop_done_at - loop_started_at
            sleep_for = frame_interval - elapsed
            if sleep_for > 0:
                time.sleep(sleep_for)
    finally:
        sender.blackout()
        sender.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
