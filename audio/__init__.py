"""Audio processing — STT, TTS, and microphone capture."""

from __future__ import annotations

from typing import Any

__all__ = ["speech_to_text", "text_to_speech"]


def __getattr__(name: str) -> Any:
    if name == "speech_to_text":
        from audio.stt import speech_to_text

        return speech_to_text
    if name == "text_to_speech":
        from audio.tss import text_to_speech

        return text_to_speech
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
