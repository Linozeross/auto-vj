from __future__ import annotations

from modules.mapping_loader import MappingArtNet, MappingData, MappingFrame, MappingLed
from modules.render_2d import DiagonalBeams2D, HorizontalWave2D, SpiralRings2D, VerticalWave2D, render_led_frame, render_preview


def _mapping() -> MappingData:
    return MappingData(
        format="auto_vj_mapping",
        version="1.0",
        frame=MappingFrame(width=200, height=100),
        artnet=MappingArtNet(ip="127.0.0.1", led_count=3),
        leds=[
            MappingLed(index=0, number=1, x=0, y=0, u=0.0, v=0.0, confidence=1.0),
            MappingLed(index=1, number=2, x=100, y=50, u=0.5, v=0.5, confidence=1.0),
            MappingLed(index=2, number=3, x=199, y=99, u=1.0, v=1.0, confidence=1.0),
        ],
    )


def test_render_led_frame_outputs_rgb_for_each_led() -> None:
    values = render_led_frame(HorizontalWave2D(), _mapping(), 0.0)

    assert len(values) == 9 and any(values)


def test_render_preview_returns_canvas_image() -> None:
    preview = render_preview(VerticalWave2D(), _mapping(), 0.0, "demo")

    assert preview.shape[0] > 0 and preview.shape[1] > 0 and preview.shape[2] == 3


def test_spiral_effect_generates_nonzero_colors() -> None:
    values = render_led_frame(SpiralRings2D(), _mapping(), 0.5)

    assert any(values)


def test_diagonal_beams_effect_generates_nonzero_colors() -> None:
    values = render_led_frame(DiagonalBeams2D(), _mapping(), 0.5)

    assert any(values)
