from __future__ import annotations

import json
from pathlib import Path

from modules.mapping_loader import load_mapping


def test_load_mapping_reads_export_payload(tmp_path: Path) -> None:
    mapping_path = tmp_path / "mapping.json"
    mapping_path.write_text(
        json.dumps(
            {
                "format": "auto_vj_mapping",
                "version": "1.0",
                "frame": {"width": 200, "height": 100},
                "artnet": {"ip": "192.168.0.2", "led_count": 4},
                "leds": [
                    {"index": 0, "number": 1, "x": 10, "y": 20, "u": 0.05, "v": 0.2, "confidence": 0.9},
                    {"index": 1, "number": 2, "x": 50, "y": 60, "u": 0.25, "v": 0.6, "confidence": 0.8},
                ],
            }
        )
    )

    mapping = load_mapping(mapping_path)

    assert mapping.artnet.led_count == 4
    assert mapping.leds[1].number == 2
