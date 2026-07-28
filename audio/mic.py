"""Host-mic recorder for push-to-talk STT (sounddevice -> temp WAV)."""

from __future__ import annotations

import tempfile
import wave
from pathlib import Path

import numpy as np
import sounddevice as sd

SAMPLE_RATE = 16_000
CHANNELS = 1
DEFAULT_SECONDS = 5.0


def _input_device_index() -> int:
    """Pick a PortAudio device with at least one input channel."""
    devices = sd.query_devices()
    # Prefer the configured default input when it is valid.
    default_in, _ = sd.default.device
    if isinstance(default_in, int) and default_in >= 0:
        info = devices[default_in]
        if int(info["max_input_channels"]) > 0:
            return default_in

    for i, info in enumerate(devices):
        if int(info["max_input_channels"]) > 0:
            return i

    raise RuntimeError(
        "No host microphone found. PortAudio only sees output devices "
        "(speakers/headphones). Enable a mic in Windows Sound settings "
        "(or connect one), then retry."
    )


def record_to_wav(seconds: float = DEFAULT_SECONDS) -> Path:
    """Record from a host microphone and write a mono WAV.

    Returns the path to a temporary .wav file. Caller should delete it when done.
    """
    device = _input_device_index()
    frames = int(seconds * SAMPLE_RATE)
    audio = sd.rec(
        frames,
        samplerate=SAMPLE_RATE,
        channels=CHANNELS,
        dtype="float32",
        device=device,
    )
    sd.wait()

    pcm = np.clip(audio.flatten(), -1.0, 1.0)
    pcm_i16 = (pcm * 32767).astype(np.int16)

    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    tmp.close()
    path = Path(tmp.name)

    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(2)  # int16
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes(pcm_i16.tobytes())

    return path
