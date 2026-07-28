# Model size comparison notes: docs/research/stt-model-size.md
from __future__ import annotations

import logging
from pathlib import Path

import whisper

logger = logging.getLogger("audio.stt")

_model = None


def get_model():
    """Lazy singleton — loading tiny once avoids multi-minute cold starts per request."""
    global _model
    if _model is None:
        _model = whisper.load_model("tiny")
    return _model


def speech_to_text(audio_path: str) -> str:
    path = Path(audio_path)
    size = path.stat().st_size if path.is_file() else 0
    logger.info("transcribe path=%s size=%s bytes", path, size)

    model = get_model()
    # fp16=False is more reliable on CPU; lower no_speech_threshold so quiet phone
    # clips are still attempted instead of returning "".
    result = model.transcribe(
        str(path),
        language="en",
        fp16=False,
        temperature=0.0,
        condition_on_previous_text=False,
        no_speech_threshold=0.2,
        logprob_threshold=-1.0,
        compression_ratio_threshold=2.8,
    )
    text = (result.get("text") or "").strip()
    logger.info("transcribe result=%r", text)
    return text


if __name__ == "__main__":
    print(
        speech_to_text(
            r"C:\Users\yj.park\Repo\helicopter\src\interface\telegram_audio.ogg"
        )
    )
