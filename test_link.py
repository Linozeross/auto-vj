"""Quick test: detect Ableton Link peers on the network."""
import asyncio
from aalink import Link


async def main():
    link = Link(120)
    link.enabled = True

    print("Listening for Ableton Link peers... (5 seconds)")
    await asyncio.sleep(1)

    peers = link.num_peers
    print(f"Peers found: {peers}")

    if peers > 0:
        print(f"SUCCESS: {peers} Link peer(s) on the network!")
        print(f"Current BPM: {link.tempo:.2f}")

        print("\nWaiting for next beat...")
        beat = await asyncio.wait_for(link.sync(1), timeout=4.0)
        print(f"Beat received at: {beat}")
    else:
        print("No Link peers found. Is Ableton Live running with Link enabled?")


asyncio.run(main())
