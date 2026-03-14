"""Minimal ArtNet test: flash :114 red for 3 seconds."""
import asyncio
import pyartnet

TARGET_IP = "192.168.178.114"
LED_COUNT = 100


async def main() -> None:
    print(f"Connecting to {TARGET_IP}:6454 ...")
    node = pyartnet.ArtNetNode.create(TARGET_IP, max_fps=25)
    async with node:
        u = node.add_universe(0)
        ch = u.add_channel(start=1, width=LED_COUNT * 3)

        # All LEDs red
        red = [255, 0, 0] * LED_COUNT
        ch.set_values(red)
        u.send_data()
        print("Sent RED — check your LEDs")
        await asyncio.sleep(3)

        # All LEDs off
        ch.set_values([0] * LED_COUNT * 3)
        u.send_data()
        print("Sent OFF")
        await asyncio.sleep(0.5)

    print("Done.")


asyncio.run(main())
