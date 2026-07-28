from kokoro import KPipeline
import sounddevice as sd

SAMPLE_RATE = 24000
_pipeline: KPipeline | None = None


def get_pipeline() -> KPipeline:
    """Lazy singleton — init once, reuse for every message."""
    global _pipeline
    if _pipeline is None:
        _pipeline = KPipeline(lang_code="a")
    return _pipeline


def text_to_speech(text: str, speed: float = 1.2, *, block: bool = True) -> None:
    pipeline = get_pipeline()
    _, _, audio = next(pipeline(text, voice="af_heart", speed=speed))
    wav = audio.numpy() if hasattr(audio, "numpy") else audio
    sd.play(wav, SAMPLE_RATE)
    if block:
        sd.wait()


if __name__ == "__main__":
    text_to_speech("salam")
    text_to_speech("Ready for the next message.")
