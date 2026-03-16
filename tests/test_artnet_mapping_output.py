from __future__ import annotations

import struct

from modules.artnet_renderer import (
    ARTNET_HEADER,
    ARTNET_OPCODE_DMX,
    ARTNET_PROTOCOL_VERSION,
    DEFAULT_TEST_RGB,
    LEDS_PER_UNIVERSE,
    StaticArtNetSender,
)

TOTAL_LEDS = 175
FIRST_LED_INDEX = 0
SECOND_UNIVERSE_LED_INDEX = 171
SECOND_UNIVERSE_RELATIVE_LED_INDEX = SECOND_UNIVERSE_LED_INDEX - LEDS_PER_UNIVERSE
DMX_VALUES_PER_LED = 3
HEADER_LENGTH = 18


class _FakeSocket:
    def __init__(self, *_args, **_kwargs) -> None:
        self.sent_packets: list[tuple[bytes, tuple[str, int]]] = []
        self.closed = False

    def setsockopt(self, *_args, **_kwargs) -> None:
        return None

    def sendto(self, packet: bytes, addr: tuple[str, int]) -> None:
        self.sent_packets.append((packet, addr))

    def close(self) -> None:
        self.closed = True


def test_static_artnet_sender_splits_frame_across_universes() -> None:
    fake_socket = _FakeSocket()
    sender = StaticArtNetSender(
        ip="127.0.0.1",
        total_leds=TOTAL_LEDS,
        socket_factory=lambda *_args, **_kwargs: fake_socket,
    )

    sender.show_leds([FIRST_LED_INDEX, SECOND_UNIVERSE_LED_INDEX], DEFAULT_TEST_RGB)

    assert len(fake_socket.sent_packets) == 2
    first_packet, first_addr = fake_socket.sent_packets[0]
    second_packet, second_addr = fake_socket.sent_packets[1]
    assert first_addr == ("127.0.0.1", 6454)
    assert second_addr == ("127.0.0.1", 6454)
    assert first_packet[:8] == ARTNET_HEADER
    assert struct.unpack("<H", first_packet[8:10])[0] == ARTNET_OPCODE_DMX
    assert struct.unpack(">H", first_packet[10:12])[0] == ARTNET_PROTOCOL_VERSION
    assert first_packet[HEADER_LENGTH:HEADER_LENGTH + 3] == bytes(DEFAULT_TEST_RGB)
    second_universe_values = second_packet[HEADER_LENGTH:]
    second_universe_led_offset = SECOND_UNIVERSE_RELATIVE_LED_INDEX * DMX_VALUES_PER_LED
    assert second_universe_values[second_universe_led_offset:second_universe_led_offset + DMX_VALUES_PER_LED] == bytes(DEFAULT_TEST_RGB)
