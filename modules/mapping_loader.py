from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

MAPPING_FORMAT_NAME = "auto_vj_mapping"
DEFAULT_MAPPING_PATH = Path("mapping_lab_export.json")


@dataclass(slots=True)
class MappingLed:
    index: int
    number: int
    x: int
    y: int
    u: float
    v: float
    confidence: float


@dataclass(slots=True)
class MappingFrame:
    width: int
    height: int


@dataclass(slots=True)
class MappingArtNet:
    ip: str
    led_count: int


@dataclass(slots=True)
class MappingData:
    format: str
    version: str
    frame: MappingFrame
    artnet: MappingArtNet
    leds: list[MappingLed]


def load_mapping(path: Path | str = DEFAULT_MAPPING_PATH) -> MappingData:
    mapping_path = Path(path)
    payload = json.loads(mapping_path.read_text())
    if payload.get("format") != MAPPING_FORMAT_NAME:
        raise ValueError(f"Unsupported mapping format: {payload.get('format')}")
    frame_payload = payload["frame"]
    artnet_payload = payload["artnet"]
    led_payloads = payload["leds"]
    return MappingData(
        format=str(payload["format"]),
        version=str(payload["version"]),
        frame=MappingFrame(width=int(frame_payload["width"]), height=int(frame_payload["height"])),
        artnet=MappingArtNet(ip=str(artnet_payload["ip"]), led_count=int(artnet_payload["led_count"])),
        leds=[
            MappingLed(
                index=int(led_payload["index"]),
                number=int(led_payload["number"]),
                x=int(led_payload["x"]),
                y=int(led_payload["y"]),
                u=float(led_payload["u"]),
                v=float(led_payload["v"]),
                confidence=float(led_payload["confidence"]),
            )
            for led_payload in led_payloads
        ],
    )
