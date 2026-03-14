import asyncio
import os
from openai import OpenAI
from dotenv import load_dotenv

from modules.recorder import record_until_release, transcribe
from modules.effects import EffectCommand, effect_from_dict
from modules.artnet_renderer import ArtNetRenderer

load_dotenv()

client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
ARTNET_IP = os.environ.get("ARTNET_IP", "192.168.178.102")
LED_COUNT = int(os.environ.get("LED_COUNT", "100"))
SYSTEM_PROMPT = open("system_prompt.txt").read().strip()


def chat_to_effect(text: str) -> dict:
    response = client.beta.chat.completions.parse(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": text},
        ],
        response_format=EffectCommand,
    )
    cmd = response.choices[0].message.parsed
    return {"effect": cmd.effect, "params": cmd.params}


async def ptt_loop(renderer: ArtNetRenderer) -> None:
    while True:
        audio = await asyncio.to_thread(record_until_release)
        print("Transcribing...")
        text = await asyncio.to_thread(transcribe, audio, client)
        print(f"You: {text}")
        print("Thinking...")
        effect_cmd = await asyncio.to_thread(chat_to_effect, text)
        print(f"Effect: {effect_cmd}")
        renderer.set_effect(effect_from_dict(effect_cmd))


async def main() -> None:
    print("=== Auto VJ ===")
    renderer = ArtNetRenderer(ARTNET_IP, LED_COUNT)
    try:
        async with asyncio.TaskGroup() as tg:
            tg.create_task(renderer.render_loop())
            tg.create_task(ptt_loop(renderer))
    except* KeyboardInterrupt:
        print("\nBye.")


if __name__ == "__main__":
    asyncio.run(main())
