"""Realtime one-object tracker driven by text prompts.

Supports webcam indices, local files, and YouTube URLs (via yt-dlp).
Open-vocab detection runs once on acquire; a light OpenCV tracker follows.
"""

from __future__ import annotations

import logging
import os
import re
import threading
import time
from dataclasses import asdict, dataclass
from typing import Any

import cv2
import numpy as np
import torch
from PIL import Image
from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

logger = logging.getLogger("perception.realtime_tracker")

# Same live used by the interface VideoBackground.
DEFAULT_YOUTUBE_LIVE = "https://www.youtube.com/watch?v=J7ZrIDvqlic"


@dataclass
class TrackingState:
    active: bool = False
    status: str = "idle"
    prompt: str = ""
    source: str = "0"
    bbox: list[float] | None = None
    frame_width: int | None = None
    frame_height: int | None = None
    score: float | None = None
    fps: float | None = None
    updated_at_s: float | None = None
    last_error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def resolve_video_source(source: int | str) -> tuple[int | str, bool]:
    """Return (opencv_source, is_seekable_file)."""
    if isinstance(source, int):
        return source, False
    text = str(source).strip()
    if text.isdigit():
        return int(text), False
    if _is_youtube_url(text):
        return _youtube_stream_url(text), False
    return text, not text.lower().startswith(("http://", "https://"))


def _is_youtube_url(url: str) -> bool:
    return bool(
        re.search(
            r"(youtube\.com|youtu\.be|youtube-nocookie\.com)",
            url,
            flags=re.IGNORECASE,
        )
    )


def _youtube_stream_url(url: str) -> str:
    try:
        from yt_dlp import YoutubeDL
    except ImportError as exc:
        raise RuntimeError(
            "yt-dlp is required for YouTube sources. pip install yt-dlp"
        ) from exc

    opts = {
        "quiet": True,
        "no_warnings": True,
        "format": "best[height<=720]/best[height<=1080]/best",
    }
    with YoutubeDL(opts) as ydl:
        info = ydl.extract_info(url, download=False)
        if info is None:
            raise RuntimeError(f"yt-dlp returned no info for {url}")
        direct = info.get("url")
        if direct:
            logger.info("Resolved YouTube stream for %s", url)
            return str(direct)
        formats = info.get("formats") or []
        for fmt in reversed(formats):
            if fmt.get("url") and fmt.get("vcodec") not in (None, "none"):
                return str(fmt["url"])
    raise RuntimeError(f"Could not resolve playable stream URL for {url}")


class RealtimePromptTracker:
    def __init__(
        self,
        *,
        source: int | str = DEFAULT_YOUTUBE_LIVE,
        detector_id: str = "IDEA-Research/grounding-dino-tiny",
        box_threshold: float = 0.25,
        text_threshold: float = 0.25,
        min_start_score: float = 0.15,
    ) -> None:
        self.source_label = str(source)
        self.detector_id = detector_id
        self.box_threshold = box_threshold
        self.text_threshold = text_threshold
        self.min_start_score = min_start_score

        self._lock = threading.Lock()
        self._state = TrackingState(source=self.source_label)
        self._jpeg_lock = threading.Lock()
        self._latest_jpeg: bytes | None = None

        self._run_flag = threading.Event()
        self._preview_thread: threading.Thread | None = None
        self._acquire_request: str | None = None
        self._stop_track_request = False
        self._acquire_busy = False

        self._tracker = None
        self._processor = None
        self._detector = None
        self._device = self._preferred_device()
        self._last_relock_s = 0.0
        # MIL drifts under pans; re-detect often (CPU DINO ~0.5–1s).
        self._relock_interval_s = 0.9

        self._start_preview()
        threading.Thread(target=self._warmup_detector, daemon=True).start()

    def start_tracking(self, prompt: str) -> TrackingState:
        text = prompt.strip().lower()
        if not text:
            return self._set_state(status="empty prompt", active=False)
        if not text.endswith("."):
            text += "."

        # Drop any previous OpenCV tracker before acquiring a new target.
        with self._lock:
            self._tracker = None
            self._stop_track_request = False
            self._acquire_request = text
        self._last_relock_s = 0.0
        return self._set_state(
            active=False,
            status=f'acquiring "{text}"',
            prompt=text,
            bbox=None,
            score=None,
            fps=None,
            updated_at_s=time.time(),
            last_error=None,
        )

    def stop_tracking(self) -> TrackingState:
        with self._lock:
            self._stop_track_request = True
            self._acquire_request = None
            self._tracker = None
        return self._set_state(
            active=False,
            status="stopped",
            bbox=None,
            score=None,
            fps=None,
            updated_at_s=time.time(),
        )

    def snapshot(self) -> TrackingState:
        with self._lock:
            return TrackingState(**self._state.to_dict())

    def latest_jpeg(self) -> bytes | None:
        with self._jpeg_lock:
            return self._latest_jpeg

    def mjpeg_frames(self, fps: float = 12.0):
        interval = 1.0 / max(1.0, fps)
        while True:
            frame = self.latest_jpeg()
            if frame is not None:
                yield (
                    b"--frame\r\n"
                    b"Content-Type: image/jpeg\r\n\r\n" + frame + b"\r\n"
                )
            time.sleep(interval)

    def _start_preview(self) -> None:
        if self._preview_thread and self._preview_thread.is_alive():
            return
        self._run_flag.set()
        self._preview_thread = threading.Thread(target=self._preview_loop, daemon=True)
        self._preview_thread.start()

    def _preview_loop(self) -> None:
        while self._run_flag.is_set():
            try:
                self._run_source_once()
            except Exception as exc:  # noqa: BLE001
                with self._lock:
                    self._tracker = None
                self._set_state(
                    active=False,
                    status="preview error",
                    bbox=None,
                    last_error=str(exc),
                    updated_at_s=time.time(),
                )
                logger.warning("Preview loop error: %s", exc)
                time.sleep(2.0)

    def _open_capture(self) -> tuple[cv2.VideoCapture, bool]:
        opencv_source, is_seekable = resolve_video_source(self.source_label)
        cap = cv2.VideoCapture(opencv_source)
        if not cap.isOpened():
            raise RuntimeError(f"Could not open video source: {self.source_label}")
        return cap, is_seekable

    def _run_source_once(self) -> None:
        cap, is_seekable = self._open_capture()
        # Live streams buffer many frames — always prefer the newest.
        try:
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        except Exception:  # noqa: BLE001
            pass
        frame_count = 0
        start = time.perf_counter()
        fail_streak = 0
        try:
            while self._run_flag.is_set():
                ok, frame = _read_latest_frame(cap, flush=5 if not is_seekable else 0)
                if not ok or frame is None:
                    fail_streak += 1
                    if is_seekable:
                        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        continue
                    if fail_streak >= 30:
                        raise RuntimeError("live stream stalled; reconnecting")
                    time.sleep(0.05)
                    continue
                fail_streak = 0

                h, w = frame.shape[:2]
                max_side = 540
                if max(h, w) > max_side:
                    scale = max_side / max(h, w)
                    frame = cv2.resize(
                        frame,
                        (int(w * scale), int(h * scale)),
                        interpolation=cv2.INTER_AREA,
                    )
                fh, fw = frame.shape[:2]

                with self._lock:
                    acquire = self._acquire_request
                    self._acquire_request = None
                    if self._stop_track_request:
                        self._tracker = None
                        self._stop_track_request = False
                    acquire_busy = self._acquire_busy
                    active = self._state.active
                    prompt = self._state.prompt

                if acquire and not acquire_busy:
                    frame_copy = frame.copy()
                    with self._lock:
                        self._acquire_busy = True
                    threading.Thread(
                        target=self._acquire_worker,
                        args=(frame_copy, acquire, False),
                        daemon=True,
                    ).start()
                elif (
                    active
                    and prompt
                    and not acquire_busy
                    and (time.time() - self._last_relock_s) >= self._relock_interval_s
                ):
                    # MIL cannot follow strong camera pans; re-detect on a timer.
                    self._last_relock_s = time.time()
                    frame_copy = frame.copy()
                    with self._lock:
                        self._acquire_busy = True
                    threading.Thread(
                        target=self._acquire_worker,
                        args=(frame_copy, prompt, True),
                        daemon=True,
                    ).start()

                bbox_xyxy = None
                with self._lock:
                    tracker = self._tracker
                    active = self._state.active
                    prompt = self._state.prompt

                if tracker is not None and active:
                    ok_track, tracked = tracker.update(frame)
                    if ok_track:
                        bbox_xyxy = bbox_xywh_to_xyxy(tracked)
                        frame_count += 1
                        fps = frame_count / max(1e-6, time.perf_counter() - start)
                        self._set_state(
                            active=True,
                            status=f'tracking "{prompt}"',
                            bbox=[float(v) for v in bbox_xyxy],
                            frame_width=int(fw),
                            frame_height=int(fh),
                            fps=float(fps),
                            updated_at_s=time.time(),
                        )
                    else:
                        with self._lock:
                            self._tracker = None
                        self._set_state(
                            active=False,
                            status="target lost",
                            bbox=None,
                            updated_at_s=time.time(),
                        )

                # Overlay is drawn in the app UI from /tracking/state — not baked in.
                self._publish_jpeg(frame)
                time.sleep(0.005)
        finally:
            cap.release()

    def _warmup_detector(self) -> None:
        try:
            self._ensure_detector()
            logger.info("Detector ready on %s", self._device)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Detector warmup failed: %s", exc)

    def _acquire_worker(
        self, frame: np.ndarray, prompt: str, relock: bool = False
    ) -> None:
        try:
            self._acquire_on_frame(frame, prompt, relock=relock)
        except Exception as exc:  # noqa: BLE001
            logger.exception("Acquire failed for prompt=%r relock=%s", prompt, relock)
            if not relock:
                with self._lock:
                    self._tracker = None
                self._set_state(
                    active=False,
                    status="acquire error",
                    last_error=_short_error(exc),
                    updated_at_s=time.time(),
                )
        finally:
            with self._lock:
                self._acquire_busy = False

    def _acquire_on_frame(
        self, frame: np.ndarray, prompt: str, *, relock: bool = False
    ) -> None:
        if not relock:
            self._set_state(
                active=False,
                status=f'acquiring "{prompt}"',
                prompt=prompt,
                updated_at_s=time.time(),
                last_error=None,
            )
        try:
            bbox_xywh, score = self._detect_prompt(frame, prompt)
        except Exception as exc:  # noqa: BLE001
            logger.exception("Detector failed for prompt=%r", prompt)
            if not relock:
                self._set_state(
                    active=False,
                    status="acquire error",
                    last_error=_short_error(exc),
                    updated_at_s=time.time(),
                )
            return

        if bbox_xywh is None or score < self.min_start_score:
            if not relock:
                with self._lock:
                    self._tracker = None
                self._set_state(
                    active=False,
                    status=f'no "{prompt}" found',
                    score=float(score),
                    last_error=f"detection score={score:.3f} on {self.source_label}",
                    updated_at_s=time.time(),
                )
            return

        tracker = _make_tracker()
        box_i = (
            int(bbox_xywh[0]),
            int(bbox_xywh[1]),
            int(bbox_xywh[2]),
            int(bbox_xywh[3]),
        )
        tracker.init(frame, box_i)
        with self._lock:
            self._tracker = tracker
        self._last_relock_s = time.time()
        fh, fw = frame.shape[:2]
        self._set_state(
            active=True,
            status=f'tracking "{prompt}"',
            prompt=prompt,
            bbox=[float(v) for v in bbox_xywh_to_xyxy(box_i)],
            frame_width=int(fw),
            frame_height=int(fh),
            score=float(score),
            updated_at_s=time.time(),
            last_error=None,
        )

    def _publish_jpeg(self, frame_bgr: np.ndarray) -> None:
        ok, buf = cv2.imencode(".jpg", frame_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 60])
        if not ok:
            return
        with self._jpeg_lock:
            self._latest_jpeg = buf.tobytes()

    def _detect_prompt(
        self, frame_bgr: np.ndarray, prompt: str
    ) -> tuple[list[float] | None, float]:
        processor, detector = self._ensure_detector()
        # Downscale for one-shot detect — acquire is rare; follow uses OpenCV.
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(rgb)
        max_side = 640
        sx = sy = 1.0
        if max(image.size) > max_side:
            scale = max_side / max(image.size)
            image = image.resize(
                (int(image.width * scale), int(image.height * scale)),
                Image.BILINEAR,
            )
            sx = frame_bgr.shape[1] / image.width
            sy = frame_bgr.shape[0] / image.height

        inputs = processor(images=image, text=prompt, return_tensors="pt")
        inputs = {k: v.to(self._device) for k, v in inputs.items()}
        try:
            with torch.no_grad():
                outputs = detector(**inputs)
        except torch.cuda.OutOfMemoryError:
            logger.warning("CUDA OOM during detect; falling back to CPU")
            self._move_detector_to_cpu()
            processor, detector = self._ensure_detector()
            inputs = {k: v.to(self._device) for k, v in inputs.items()}
            with torch.no_grad():
                outputs = detector(**inputs)

        results = processor.post_process_grounded_object_detection(
            outputs,
            inputs["input_ids"],
            threshold=self.box_threshold,
            text_threshold=self.text_threshold,
            target_sizes=[(image.height, image.width)],
        )[0]
        boxes = results["boxes"].detach().cpu().numpy()
        scores = results["scores"].detach().cpu().numpy()
        if len(boxes) == 0:
            return None, 0.0
        best = int(np.argmax(scores))
        x0, y0, x1, y1 = [float(v) for v in boxes[best]]
        x0, x1 = x0 * sx, x1 * sx
        y0, y1 = y0 * sy, y1 * sy
        w = max(1.0, x1 - x0)
        h = max(1.0, y1 - y0)
        return [x0, y0, w, h], float(scores[best])

    def _move_detector_to_cpu(self) -> None:
        self._device = "cpu"
        if self._detector is not None:
            self._detector.to("cpu")
            self._detector.eval()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _ensure_detector(self):
        if self._processor is None or self._detector is None:
            preferred = self._preferred_device()
            self._processor = AutoProcessor.from_pretrained(self.detector_id)
            try:
                if preferred == "cuda" and torch.cuda.is_available():
                    torch.cuda.empty_cache()
                self._detector = AutoModelForZeroShotObjectDetection.from_pretrained(
                    self.detector_id
                ).to(preferred)
                self._device = preferred
            except torch.cuda.OutOfMemoryError:
                logger.warning(
                    "CUDA OOM loading detector; using CPU (slower acquire, OK once)"
                )
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                self._detector = AutoModelForZeroShotObjectDetection.from_pretrained(
                    self.detector_id
                ).to("cpu")
                self._device = "cpu"
            self._detector.eval()
            logger.info("Detector ready on %s", self._device)
        return self._processor, self._detector

    @staticmethod
    def _preferred_device() -> str:
        # Override: HELI_DETECT_DEVICE=cpu|cuda
        forced = os.getenv("HELI_DETECT_DEVICE", "").strip().lower()
        if forced in {"cpu", "cuda"}:
            return forced
        # Default CPU on laptops so TTS/STT can keep the GPU.
        if os.getenv("HELI_DETECT_FORCE_CUDA", "").strip() in {"1", "true", "yes"}:
            return "cuda" if torch.cuda.is_available() else "cpu"
        return "cpu"

    def _set_state(self, **kwargs: Any) -> TrackingState:
        with self._lock:
            for key, value in kwargs.items():
                setattr(self._state, key, value)
            return TrackingState(**self._state.to_dict())


def _read_latest_frame(
    cap: cv2.VideoCapture, flush: int = 0
) -> tuple[bool, np.ndarray | None]:
    """Read a frame, discarding older buffered frames on live sources."""
    ok, frame = cap.read()
    if not ok or frame is None:
        return False, None
    for _ in range(max(0, flush)):
        more = cap.grab()
        if not more:
            break
        ok2, frame2 = cap.retrieve()
        if ok2 and frame2 is not None:
            frame = frame2
    return True, frame


def _short_error(exc: BaseException, limit: int = 220) -> str:
    text = str(exc).strip() or exc.__class__.__name__
    if "out of memory" in text.lower() or "cuda" in text.lower() and "memory" in text.lower():
        return (
            "GPU out of memory while acquiring target. "
            "Detector will retry on CPU after server restart; "
            "or close other GPU apps and try again."
        )
    if len(text) <= limit:
        return text
    return text[: limit - 3] + "..."


def _make_tracker():
    candidates = (
        "TrackerCSRT_create",
        "TrackerKCF_create",
        "TrackerMIL_create",
    )
    legacy = getattr(cv2, "legacy", None)
    for name in candidates:
        if hasattr(cv2, name):
            return getattr(cv2, name)()
        if legacy is not None and hasattr(legacy, name):
            return getattr(legacy, name)()
    raise RuntimeError("No OpenCV tracker available (need contrib build with tracking).")


def bbox_xywh_to_xyxy(bbox_xywh) -> tuple[float, float, float, float]:
    x, y, w, h = [float(v) for v in bbox_xywh]
    return x, y, x + w, y + h


def _draw_bbox(frame: np.ndarray, bbox_xyxy, label: str) -> np.ndarray:
    out = frame.copy()
    x0, y0, x1, y1 = [int(v) for v in bbox_xyxy]
    cv2.rectangle(out, (x0, y0), (x1, y1), (0, 200, 255), 2)
    cv2.putText(
        out,
        label,
        (x0, max(24, y0 - 8)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 200, 255),
        2,
        cv2.LINE_AA,
    )
    return out
