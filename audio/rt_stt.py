"""RealtimeSTT for the chat app — same recorder config as realtime_stt.py.

Host mic (AirPods Hands-Free preferred). Browser only receives transcripts
over the websocket; it does not feed PCM.
"""

from __future__ import annotations

import asyncio
import logging
import threading
import uuid
from dataclasses import dataclass

import pyaudio
from RealtimeSTT import AudioToTextRecorder

logger = logging.getLogger("audio.rt_stt")

SttEvent = dict[str, str]


@dataclass(slots=True)
class StreamSession:
    session_id: str
    loop: asyncio.AbstractEventLoop
    queue: asyncio.Queue[SttEvent]


def _find_input_device() -> int:
    """Prefer AirPods Hands-Free mic (same logic as realtime_stt.py)."""
    pa = pyaudio.PyAudio()
    try:
        handsfree_16k = None
        handsfree_any = None
        for i in range(pa.get_device_count()):
            info = pa.get_device_info_by_index(i)
            name = info.get("name", "")
            if info.get("maxInputChannels", 0) <= 0:
                continue
            if "Hands-Free" not in name or "bthhfenum" in name:
                continue
            if int(info.get("defaultSampleRate", 0)) == 16000 and handsfree_16k is None:
                handsfree_16k = i
            elif handsfree_any is None:
                handsfree_any = i
        chosen = handsfree_16k if handsfree_16k is not None else handsfree_any
        if chosen is not None:
            info = pa.get_device_info_by_index(chosen)
            logger.info(
                "Using input device %s: %s (%s Hz)",
                chosen,
                info["name"],
                int(info["defaultSampleRate"]),
            )
            return chosen
        default = pa.get_default_input_device_info()
        logger.warning(
            "No AirPods Hands-Free mic found; using default %s: %s",
            default["index"],
            default["name"],
        )
        return int(default["index"])
    finally:
        pa.terminate()


class RealtimeSttManager:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._ready = threading.Event()
        self._running = True
        self._recorder: AudioToTextRecorder | None = None
        self._worker: threading.Thread | None = None
        self._session: StreamSession | None = None

    def warmup(self) -> None:
        self._ensure_started()

    def _ensure_started(self) -> None:
        with self._lock:
            if self._worker and self._worker.is_alive():
                return
            self._worker = threading.Thread(target=self._run_worker, daemon=True)
            self._worker.start()

    def _run_worker(self) -> None:
        logger.info("Initializing RealtimeSTT (realtime_stt.py config)")
        try:
            unknown_sentence_detection_pause = 0.7
            input_device_index = _find_input_device()
            # Same recorder_config as realtime_stt.py
            self._recorder = AudioToTextRecorder(
                spinner=False,
                model="large-v2",
                download_root=None,
                input_device_index=input_device_index,
                realtime_model_type="tiny.en",
                language="en",
                silero_sensitivity=0.4,
                webrtc_sensitivity=2,
                post_speech_silence_duration=unknown_sentence_detection_pause,
                min_length_of_recording=1.1,
                min_gap_between_recordings=0,
                enable_realtime_transcription=True,
                realtime_processing_pause=1,
                on_realtime_transcription_update=self._on_realtime,
                silero_deactivity_detection=True,
                early_transcription_on_silence=0,
                realtime_transcription_use_syllable_boundaries=True,
                realtime_boundary_detector_sensitivity=0.6,
                realtime_boundary_followup_delays=(0.5,),
                beam_size=5,
                beam_size_realtime=3,
                no_log_file=True,
                initial_prompt_realtime=(
                    "End incomplete sentences with ellipses.\n"
                    "Examples:\n"
                    "Complete: The sky is blue.\n"
                    "Incomplete: When the sky...\n"
                    "Complete: She walked home.\n"
                    "Incomplete: Because he...\n"
                ),
                silero_use_onnx=True,
                faster_whisper_vad_filter=False,
            )
            self._ready.set()
            logger.info("RealtimeSTT recorder ready")
        except Exception:
            logger.exception("Failed to initialize RealtimeSTT recorder")
            self._running = False
            self._ready.set()
            return

        while self._running:
            try:
                text = self._recorder.text().strip()
                if text:
                    self._emit("final", text)
            except Exception:
                logger.exception("Recorder loop failed")

    def _on_realtime(self, text: str) -> None:
        if text:
            self._emit("realtime", text)

    def _emit(self, event_type: str, text: str) -> None:
        with self._lock:
            session = self._session
        if session is None:
            return
        future = asyncio.run_coroutine_threadsafe(
            session.queue.put({"type": event_type, "text": text}),
            session.loop,
        )
        try:
            future.result(timeout=0.5)
        except Exception:
            logger.debug("Dropping %s STT event for disconnected client", event_type)

    def attach_session(self, loop: asyncio.AbstractEventLoop) -> StreamSession:
        self._ensure_started()
        self._ready.wait(timeout=180.0)
        if self._recorder is None:
            raise RuntimeError("RealtimeSTT recorder is unavailable")
        session = StreamSession(
            session_id=str(uuid.uuid4()),
            loop=loop,
            queue=asyncio.Queue(),
        )
        with self._lock:
            self._session = session
        return session

    def detach_session(self, session_id: str) -> None:
        with self._lock:
            if self._session and self._session.session_id == session_id:
                self._session = None


_manager = RealtimeSttManager()


def warmup() -> None:
    _manager.warmup()


def attach_session(loop: asyncio.AbstractEventLoop) -> StreamSession:
    return _manager.attach_session(loop)


def detach_session(session_id: str) -> None:
    _manager.detach_session(session_id)
