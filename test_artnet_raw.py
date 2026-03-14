"""Raw UDP ArtNet DMX packet — no library, straight socket send."""
import socket
import struct
import time

TARGET_IP = "192.168.178.114"
ARTNET_PORT = 6454
LED_COUNT = 280
LEDS_PER_UNIVERSE = 170


def artnet_dmx_packet(universe: int, data: bytes) -> bytes:
    length = len(data)
    header = (
        b"Art-Net\x00"          # ID (8 bytes, null-terminated)
        + struct.pack("<H", 0x5000)  # OpCode: ArtDMX (little-endian)
        + struct.pack(">H", 14)      # ProtVer 14 (big-endian)
        + b"\x00"                    # Sequence (0 = disabled)
        + b"\x00"                    # Physical
        + struct.pack("<H", universe) # Universe (little-endian)
        + struct.pack(">H", length)  # Length (big-endian)
    )
    return header + data


sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)

remaining = LED_COUNT
universe_idx = 0
universes = []
while remaining > 0:
    count = min(remaining, LEDS_PER_UNIVERSE)
    universes.append((universe_idx, count))
    remaining -= count
    universe_idx += 1

print(f"Sending raw ArtNet DMX to {TARGET_IP}:{ARTNET_PORT}, {len(universes)} universe(s), {LED_COUNT} LEDs total")
print("Watch for RED on your LEDs for 3 seconds...")

for _ in range(30):
    for u_idx, count in universes:
        pkt = artnet_dmx_packet(u_idx, bytes([255, 0, 0] * count))
        sock.sendto(pkt, (TARGET_IP, ARTNET_PORT))
    time.sleep(0.1)

print("Sending OFF...")
for u_idx, count in universes:
    sock.sendto(artnet_dmx_packet(u_idx, bytes([0] * count * 3)), (TARGET_IP, ARTNET_PORT))
sock.close()
print("Done.")
