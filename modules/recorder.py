import io
import time
import wave
from collections.abc import Callable
import numpy as np
import sounddevice as sd
from pynput import keyboard
from openai import OpenAI

SAMPLE_RATE = 16000
CHANNELS = 1
PTT_KEY = keyboard.Key.space


def record_until_release(on_recording_start: Callable[[], None] | None = None) -> np.ndarray:
    """Block until PTT key is held, record while held, return audio array.

    on_recording_start is called just before the audio stream opens (i.e. when
    Space is pressed), allowing callers to pause conflicting streams.
    """
    frames = []
    recording = False
    done = False

    def on_press(key):
        nonlocal recording
        if key == PTT_KEY and not recording:
            recording = True
            print("Recording... (release to send)")

    def on_release(key):
        nonlocal done
        if key == PTT_KEY:
            done = True
            return False  # stop listener

    with keyboard.Listener(on_press=on_press, on_release=on_release):
        print(f"Hold [{PTT_KEY}] to talk")
        while not recording:
            time.sleep(0.01)
        if on_recording_start is not None:
            on_recording_start()
        with sd.InputStream(samplerate=SAMPLE_RATE, channels=CHANNELS, dtype="int16") as stream:
            while not done:
                data, _ = stream.read(1024)
                frames.append(data.copy())

    if not frames:
        return np.zeros(0, dtype=np.int16)
    return np.concatenate(frames, axis=0)


def transcribe(audio: np.ndarray, client: OpenAI) -> str:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(2)  # int16 = 2 bytes
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes(audio.tobytes())
    buf.seek(0)
    buf.name = "recording.wav"
    result = client.audio.transcriptions.create(model="whisper-1", file=buf)
    return result.text
