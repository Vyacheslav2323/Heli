"""Frame-upload multi-object detector/tracker for low-latency live video UX."""

from __future__ import annotations

import logging
import os
import re
import threading
import time
from dataclasses import asdict, dataclass, field
from typing import Any

import cv2
import numpy as np
import torch
from PIL import Image
from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

logger = logging.getLogger("perception.frame_detector")
CLICK_PROMPT_SENTINEL = "__click__."


@dataclass
class TrackedTarget:
    index: int
    bbox: list[float]  # xyxy
    score: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class TrackingState:
    active: bool = False
    status: str = "idle"
    prompt: str = ""
    source: str = "app-upload"
    bbox: list[float] | None = None  # selected / primary box (compat)
    targets: list[TrackedTarget] = field(default_factory=list)
    selected_index: int | None = None  # None = track all
    mode: str = "all"  # "all" | "single"
    frame_width: int | None = None
    frame_height: int | None = None
    score: float | None = None
    fps: float | None = None
    updated_at_s: float | None = None
    last_error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        return d


class FramePromptTracker:
    """Text-prompt tracker that runs only on frames uploaded by the app."""

    def __init__(
        self,
        *,
        detector_id: str = "IDEA-Research/grounding-dino-tiny",
        box_threshold: float = 0.25,
        text_threshold: float = 0.25,
        min_start_score: float = 0.15,
        redetect_interval_s: float = 0.5,
    ) -> None:
        self.detector_id = detector_id
        self.box_threshold = box_threshold
        self.text_threshold = text_threshold
        self.min_start_score = min_start_score
        self.redetect_interval_s = redetect_interval_s

        self._lock = threading.Lock()
        self._state = TrackingState()
        self._flow_prev_gray: np.ndarray | None = None
        self._flow_pts: np.ndarray | None = None
        self._flow_ids: list[int] = []
        self._flow_wh: list[tuple[float, float]] = []
        self._flow_scores: list[float] = []
        self._last_detect_s = 0.0
        self._last_frame_s = 0.0
        self._locked_bbox: list[float] | None = None
        self._detect_busy = False
        self._command_epoch = 0
        self._device = self._preferred_device()
        self._processor = None
        self._detector = None

        threading.Thread(target=self._warmup_detector, daemon=True).start()

    def start_tracking(
        self,
        prompt: str,
        *,
        selected_index: int | None = None,
    ) -> TrackingState:
        normalized = _normalize_prompt(prompt)
        if not normalized:
            return self._set_state(active=False, status="empty prompt", prompt="")
        mode = "single" if selected_index is not None else "all"
        with self._lock:
            self._command_epoch += 1
            self._clear_flow_locked()
            self._locked_bbox = None
            self._last_detect_s = 0.0
            self._detect_busy = False
        label = (
            f'armed "{normalized}" #{selected_index} (waiting for frame)'
            if selected_index is not None
            else f'armed "{normalized}" all (waiting for frame)'
        )
        return self._set_state(
            active=False,
            status=label,
            prompt=normalized,
            bbox=None,
            targets=[],
            selected_index=selected_index,
            mode=mode,
            frame_width=None,
            frame_height=None,
            fps=None,
            score=None,
            updated_at_s=time.time(),
            last_error=None,
        )

    def start_from_bbox(
        self,
        bbox_xyxy: list[float] | tuple[float, float, float, float],
        *,
        label: str = "selection",
    ) -> TrackingState:
        x0, y0, x1, y1 = [float(v) for v in bbox_xyxy]
        if x1 <= x0 or y1 <= y0:
            raise ValueError("bbox must be [x0,y0,x1,y1] with positive area")
        target = TrackedTarget(index=1, bbox=[x0, y0, x1, y1], score=1.0)
        with self._lock:
            self._command_epoch += 1
            self._clear_flow_locked()
            self._locked_bbox = list(target.bbox)
            self._last_detect_s = 0.0
            self._detect_busy = False
        return self._set_state(
            active=True,
            status=f'tracking "{label}"',
            prompt=CLICK_PROMPT_SENTINEL,
            bbox=list(target.bbox),
            targets=[target],
            selected_index=1,
            mode="single",
            score=1.0,
            updated_at_s=time.time(),
            last_error=None,
        )

    def select_index(self, index: int) -> TrackingState:
        """Switch to a specific detection index from the current class."""
        if index < 1:
            return self._set_state(
                last_error="index must be >= 1",
                updated_at_s=time.time(),
            )
        with self._lock:
            prompt = self._state.prompt
            targets = list(self._state.targets)
            self._command_epoch += 1
            self._clear_flow_locked()
            self._locked_bbox = None
            self._last_detect_s = 0.0
            self._detect_busy = False
        if not prompt:
            return self._set_state(
                active=False,
                status="idle",
                last_error="No active class — say e.g. 'track bear' first.",
                updated_at_s=time.time(),
            )
        match = next((t for t in targets if t.index == index), None)
        if match is not None:
            self._locked_bbox = list(match.bbox)
        return self._set_state(
            active=True if match else False,
            status=(
                f'tracking "{prompt}" #{index}'
                if match
                else f'armed "{prompt}" #{index} (waiting for frame)'
            ),
            selected_index=index,
            mode="single",
            bbox=list(match.bbox) if match else None,
            score=float(match.score) if match else None,
            updated_at_s=time.time(),
            last_error=None if match else f"Index {index} not in last detections",
        )

    def track_all(self) -> TrackingState:
        with self._lock:
            prompt = self._state.prompt
            self._command_epoch += 1
            self._clear_flow_locked()
            self._locked_bbox = None
            self._last_detect_s = 0.0
            self._detect_busy = False
        if not prompt:
            return self._set_state(
                active=False,
                status="idle",
                last_error="No active class — say e.g. 'track bear' first.",
                updated_at_s=time.time(),
            )
        return self._set_state(
            active=False,
            status=f'armed "{prompt}" all (waiting for frame)',
            selected_index=None,
            mode="all",
            bbox=None,
            updated_at_s=time.time(),
            last_error=None,
        )

    def stop_tracking(self) -> TrackingState:
        with self._lock:
            self._command_epoch += 1
            self._clear_flow_locked()
            self._locked_bbox = None
            self._last_detect_s = 0.0
            self._last_frame_s = 0.0
            self._detect_busy = False
        return self._set_state(
            active=False,
            status="stopped",
            prompt="",
            bbox=None,
            targets=[],
            selected_index=None,
            mode="all",
            fps=None,
            score=None,
            updated_at_s=time.time(),
            last_error=None,
        )

    def snapshot(self) -> TrackingState:
        with self._lock:
            return _coerce_state(self._state.to_dict())

    def detect_once(self, frame_bgr: np.ndarray, prompt: str) -> TrackingState:
        normalized = _normalize_prompt(prompt)
        if not normalized:
            raise ValueError("prompt is required")
        detections = self._detect_all(frame_bgr, normalized)
        fh, fw = frame_bgr.shape[:2]
        targets = _index_detections(detections)
        if not targets:
            return TrackingState(
                active=False,
                status=f'no "{normalized}" found',
                prompt=normalized,
                source="app-upload",
                bbox=None,
                targets=[],
                selected_index=None,
                mode="all",
                frame_width=int(fw),
                frame_height=int(fh),
                score=0.0,
                fps=None,
                updated_at_s=time.time(),
                last_error=None,
            )
        return TrackingState(
            active=True,
            status=f'detected {len(targets)} "{normalized}"',
            prompt=normalized,
            source="app-upload",
            bbox=list(targets[0].bbox),
            targets=targets,
            selected_index=None,
            mode="all",
            frame_width=int(fw),
            frame_height=int(fh),
            score=float(targets[0].score),
            fps=None,
            updated_at_s=time.time(),
            last_error=None,
        )

    def track_on_frame(
        self,
        frame_bgr: np.ndarray,
        *,
        prompt: str | None = None,
    ) -> TrackingState:
        """Fast path: OpenCV / last boxes every call. DINO redetect runs async."""
        normalized = _normalize_prompt(prompt or "")
        with self._lock:
            epoch = self._command_epoch
            if normalized:
                self._state.prompt = normalized
            current_prompt = self._state.prompt
            selected_index = self._state.selected_index
            mode = self._state.mode
            locked = list(self._locked_bbox) if self._locked_bbox else None
            prev_frame_s = self._last_frame_s
            now = time.time()
            self._last_frame_s = now
            detect_due = (now - self._last_detect_s) >= self.redetect_interval_s
            detect_busy = self._detect_busy
            has_targets = bool(self._state.targets)

        if not current_prompt:
            return self._set_state(
                active=False,
                status="idle",
                bbox=None,
                targets=[],
                updated_at_s=time.time(),
                last_error=None,
            )

        fh, fw = frame_bgr.shape[:2]
        fps = None
        if prev_frame_s > 0:
            dt = max(1e-3, now - prev_frame_s)
            fps = 1.0 / dt

        # If a newer chat/click command landed, abort this stale frame.
        with self._lock:
            if epoch != self._command_epoch:
                return _coerce_state(self._state.to_dict())

        # Fast LK optical flow follow (MIL is too slow for multi-target).
        followed = self._update_flow(frame_bgr)
        with self._lock:
            if epoch != self._command_epoch:
                return _coerce_state(self._state.to_dict())
        if not followed and has_targets:
            with self._lock:
                if epoch != self._command_epoch:
                    return _coerce_state(self._state.to_dict())
                seed = list(self._state.targets)
            if mode == "single" and selected_index is not None:
                seed = [t for t in seed if t.index == selected_index] or seed
            self._init_flow(frame_bgr, seed)
            followed = self._update_flow(frame_bgr) or seed

        if followed:
            with self._lock:
                if epoch != self._command_epoch:
                    return _coerce_state(self._state.to_dict())
                self._state.targets = followed
                if mode == "single" and selected_index is not None:
                    hit = next((t for t in followed if t.index == selected_index), None)
                    if hit is not None:
                        self._locked_bbox = list(hit.bbox)

        is_click_mode = current_prompt == CLICK_PROMPT_SENTINEL
        need_detect = (not is_click_mode) and ((not has_targets) or (not followed) or detect_due)
        if need_detect and not detect_busy:
            frame_copy = frame_bgr.copy()
            with self._lock:
                if epoch != self._command_epoch:
                    return _coerce_state(self._state.to_dict())
                self._detect_busy = True
                self._last_detect_s = time.time()
            threading.Thread(
                target=self._redetect_worker,
                args=(frame_copy, current_prompt, mode, selected_index, locked, epoch),
                daemon=True,
            ).start()

        with self._lock:
            if epoch != self._command_epoch:
                return _coerce_state(self._state.to_dict())
            targets = list(self._state.targets)
            locked_now = list(self._locked_bbox) if self._locked_bbox else None
            selected_index = self._state.selected_index
            mode = self._state.mode
            # Always prefer the live prompt — never clobber a newer command.
            live_prompt = self._state.prompt

        if mode == "single" and selected_index is not None:
            hit = next((t for t in targets if t.index == selected_index), None)
            bbox = list(hit.bbox) if hit else locked_now
            active = bbox is not None
            status_prompt = (
                "selection" if live_prompt == CLICK_PROMPT_SENTINEL else live_prompt
            )
            return self._set_state(
                active=active,
                status=(
                    f'tracking "{status_prompt}" #{selected_index}'
                    if active
                    else f'armed "{status_prompt}" #{selected_index} (waiting for frame)'
                ),
                prompt=live_prompt,
                bbox=bbox,
                targets=targets,
                selected_index=selected_index,
                mode="single",
                frame_width=int(fw),
                frame_height=int(fh),
                fps=float(fps) if fps is not None else None,
                updated_at_s=time.time(),
                last_error=None,
            )

        active = bool(targets)
        return self._set_state(
            active=active,
            status=(
                f'tracking {len(targets)} "{live_prompt}"'
                if active
                else f'armed "{live_prompt}" all (waiting for frame)'
            ),
            prompt=live_prompt,
            bbox=list(targets[0].bbox) if targets else None,
            targets=targets,
            selected_index=None,
            mode="all",
            frame_width=int(fw),
            frame_height=int(fh),
            score=float(targets[0].score) if targets else None,
            fps=float(fps) if fps is not None else None,
            updated_at_s=time.time(),
            last_error=None if active or detect_busy or need_detect else "no detections",
        )

    def _redetect_worker(
        self,
        frame_bgr: np.ndarray,
        prompt: str,
        mode: str,
        selected_index: int | None,
        locked: list[float] | None,
        epoch: int,
    ) -> None:
        try:
            with self._lock:
                if epoch != self._command_epoch:
                    return
            detections = self._detect_all(frame_bgr, prompt)
            targets = _index_detections(detections)
            fh, fw = frame_bgr.shape[:2]
            with self._lock:
                if epoch != self._command_epoch or self._state.prompt != prompt:
                    return
            if not targets:
                with self._lock:
                    if epoch != self._command_epoch:
                        return
                    self._clear_flow_locked()
                    self._locked_bbox = None
                self._set_state(
                    active=False,
                    status=f'no "{prompt}" found',
                    prompt=prompt,
                    bbox=None,
                    targets=[],
                    frame_width=int(fw),
                    frame_height=int(fh),
                    score=0.0,
                    updated_at_s=time.time(),
                    last_error="no detections",
                )
                return

            if mode == "all" or selected_index is None:
                with self._lock:
                    if epoch != self._command_epoch or self._state.prompt != prompt:
                        return
                self._init_flow(frame_bgr, targets)
                self._set_state(
                    active=True,
                    status=f'tracking {len(targets)} "{prompt}"',
                    prompt=prompt,
                    bbox=list(targets[0].bbox),
                    targets=targets,
                    selected_index=None,
                    mode="all",
                    frame_width=int(fw),
                    frame_height=int(fh),
                    score=float(targets[0].score),
                    updated_at_s=time.time(),
                    last_error=None,
                )
                return

            chosen = _pick_sticky_target(targets, selected_index, locked)
            with self._lock:
                if epoch != self._command_epoch or self._state.prompt != prompt:
                    return
            if chosen is None:
                with self._lock:
                    self._clear_flow_locked()
                    self._locked_bbox = None
                self._set_state(
                    active=False,
                    status=f'no "{prompt}" #{selected_index}',
                    prompt=prompt,
                    bbox=None,
                    targets=targets,
                    selected_index=selected_index,
                    mode="single",
                    frame_width=int(fw),
                    frame_height=int(fh),
                    score=None,
                    updated_at_s=time.time(),
                    last_error=f"index {selected_index} missing",
                )
                return

            # Flow-track all detections; selected index is highlighted by mode.
            self._init_flow(frame_bgr, targets)
            with self._lock:
                if epoch != self._command_epoch or self._state.prompt != prompt:
                    return
                self._locked_bbox = list(chosen.bbox)
            self._set_state(
                active=True,
                status=f'tracking "{prompt}" #{chosen.index}',
                prompt=prompt,
                bbox=list(chosen.bbox),
                targets=targets,
                selected_index=chosen.index,
                mode="single",
                frame_width=int(fw),
                frame_height=int(fh),
                score=float(chosen.score),
                updated_at_s=time.time(),
                last_error=None,
            )
        except Exception as exc:  # noqa: BLE001
            logger.exception("Async redetect failed")
            with self._lock:
                if epoch != self._command_epoch:
                    return
            self._set_state(
                last_error=str(exc),
                updated_at_s=time.time(),
            )
        finally:
            with self._lock:
                if epoch == self._command_epoch:
                    self._detect_busy = False

    def _clear_flow_locked(self) -> None:
        self._flow_prev_gray = None
        self._flow_pts = None
        self._flow_ids = []
        self._flow_wh = []
        self._flow_scores = []

    def _init_flow(self, frame_bgr: np.ndarray, targets: list[TrackedTarget]) -> None:
        if not targets:
            with self._lock:
                self._clear_flow_locked()
            return
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        ids: list[int] = []
        wh: list[tuple[float, float]] = []
        scores: list[float] = []
        pts: list[list[list[float]]] = []
        for t in targets:
            x0, y0, x1, y1 = t.bbox
            cx = 0.5 * (x0 + x1)
            cy = 0.5 * (y0 + y1)
            pts.append([[cx, cy]])
            ids.append(t.index)
            wh.append((max(1.0, x1 - x0), max(1.0, y1 - y0)))
            scores.append(float(t.score))
        with self._lock:
            self._flow_prev_gray = gray
            self._flow_pts = np.array(pts, dtype=np.float32)
            self._flow_ids = ids
            self._flow_wh = wh
            self._flow_scores = scores

    def _update_flow(self, frame_bgr: np.ndarray) -> list[TrackedTarget]:
        with self._lock:
            prev = self._flow_prev_gray
            p0 = None if self._flow_pts is None else self._flow_pts.copy()
            ids = list(self._flow_ids)
            wh = list(self._flow_wh)
            scores = list(self._flow_scores)
        if prev is None or p0 is None or len(p0) == 0:
            return []

        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        p1, st, _err = cv2.calcOpticalFlowPyrLK(
            prev,
            gray,
            p0,
            None,
            winSize=(31, 31),
            maxLevel=3,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 20, 0.03),
        )
        if p1 is None or st is None:
            return []

        fh, fw = frame_bgr.shape[:2]
        updated: list[TrackedTarget] = []
        keep_pts: list[list[list[float]]] = []
        keep_ids: list[int] = []
        keep_wh: list[tuple[float, float]] = []
        keep_scores: list[float] = []
        for i, ok in enumerate(st.reshape(-1)):
            if int(ok) != 1:
                continue
            cx, cy = float(p1[i, 0, 0]), float(p1[i, 0, 1])
            w, h = wh[i]
            x0 = max(0.0, min(fw - 1.0, cx - w / 2.0))
            y0 = max(0.0, min(fh - 1.0, cy - h / 2.0))
            x1 = max(x0 + 1.0, min(float(fw), x0 + w))
            y1 = max(y0 + 1.0, min(float(fh), y0 + h))
            updated.append(
                TrackedTarget(index=ids[i], bbox=[x0, y0, x1, y1], score=scores[i])
            )
            keep_pts.append([[cx, cy]])
            keep_ids.append(ids[i])
            keep_wh.append((w, h))
            keep_scores.append(scores[i])

        updated.sort(key=lambda t: t.index)
        with self._lock:
            self._flow_prev_gray = gray
            self._flow_pts = (
                np.array(keep_pts, dtype=np.float32) if keep_pts else None
            )
            self._flow_ids = keep_ids
            self._flow_wh = keep_wh
            self._flow_scores = keep_scores
        return updated

    def _warmup_detector(self) -> None:
        try:
            self._ensure_detector()
            logger.info("Frame detector ready on %s", self._device)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Frame detector warmup failed: %s", exc)

    def _detect_all(
        self, frame_bgr: np.ndarray, prompt: str
    ) -> list[tuple[list[float], float]]:
        """Return all detections as (xywh, score), filtered by min score."""
        processor, detector = self._ensure_detector()
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
            logger.warning("CUDA OOM during detect; moving detector to CPU")
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
        out: list[tuple[list[float], float]] = []
        for box, score in zip(boxes, scores, strict=False):
            sc = float(score)
            if sc < self.min_start_score:
                continue
            x0, y0, x1, y1 = [float(v) for v in box]
            x0, x1 = x0 * sx, x1 * sx
            y0, y1 = y0 * sy, y1 * sy
            w = max(1.0, x1 - x0)
            h = max(1.0, y1 - y0)
            out.append(([x0, y0, w, h], sc))
        return out

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
                logger.warning("CUDA OOM loading detector; falling back to CPU")
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                self._detector = AutoModelForZeroShotObjectDetection.from_pretrained(
                    self.detector_id
                ).to("cpu")
                self._device = "cpu"
            self._detector.eval()
        return self._processor, self._detector

    @staticmethod
    def _preferred_device() -> str:
        forced = os.getenv("HELI_DETECT_DEVICE", "").strip().lower()
        if forced in {"cpu", "cuda"}:
            return forced
        if os.getenv("HELI_DETECT_FORCE_CUDA", "").strip() in {"1", "true", "yes"}:
            return "cuda" if torch.cuda.is_available() else "cpu"
        return "cpu"

    def _set_state(self, **kwargs: Any) -> TrackingState:
        with self._lock:
            for key, value in kwargs.items():
                if key == "targets":
                    value = _coerce_targets(value)
                setattr(self._state, key, value)
            return _coerce_state(self._state.to_dict())


def _coerce_targets(value: Any) -> list[TrackedTarget]:
    if not value:
        return []
    out: list[TrackedTarget] = []
    for item in value:
        if isinstance(item, TrackedTarget):
            out.append(item)
        elif isinstance(item, dict):
            out.append(TrackedTarget(**item))
    return out


def _coerce_state(data: dict[str, Any]) -> TrackingState:
    data = dict(data)
    data["targets"] = _coerce_targets(data.get("targets"))
    return TrackingState(**data)


def decode_upload_image(raw: bytes) -> np.ndarray:
    arr = np.frombuffer(raw, dtype=np.uint8)
    frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if frame is None:
        raise ValueError("Invalid image upload")
    return frame


def bbox_xywh_to_xyxy(bbox_xywh) -> tuple[float, float, float, float]:
    x, y, w, h = [float(v) for v in bbox_xywh]
    return x, y, x + w, y + h


def parse_track_command(user_text: str) -> dict[str, Any] | None:
    """
    Parse track/follow chat commands.

    Examples:
      track bear
      track bears / track all bears
      track bear 3
      track 3          (select index in current session)
      track all
      stop tracking
    """
    text = user_text.strip().lower()
    if not text:
        return None

    if text in {"stop tracking", "stop track", "cancel tracking"}:
        return {"action": "stop"}

    if text in {"tracking status", "track status", "status tracking"}:
        return {"action": "status"}

    if text in {"track all", "follow all"}:
        return {"action": "all"}

    # track 3 / follow 12
    m = re.fullmatch(r"(?:track|follow)\s+(\d+)", text)
    if m:
        return {"action": "select", "index": int(m.group(1))}

    for marker in ("track ", "follow "):
        if not text.startswith(marker):
            continue
        rest = text[len(marker) :].strip()
        if not rest:
            return {"action": "help"}

        # track all bears / track bears
        if rest in {"all", "all of them"}:
            return {"action": "all"}
        m_all = re.fullmatch(r"all\s+(.+)", rest)
        if m_all:
            return {
                "action": "start",
                "prompt": m_all.group(1).strip(),
                "selected_index": None,
            }
        # track bear 3 / track bear #3
        m_idx = re.fullmatch(r"(.+?)\s+#?(\d+)", rest)
        if m_idx:
            prompt = m_idx.group(1).strip()
            if prompt in {"all"}:
                return {"action": "all"}
            return {
                "action": "start",
                "prompt": prompt,
                "selected_index": int(m_idx.group(2)),
            }
        # track bears -> treat as all of class bear
        prompt = rest
        if prompt.endswith("s") and len(prompt) > 3 and " " not in prompt:
            # crude plural: bears -> bear (keep seagulls etc. as-is if needed)
            singular = prompt[:-1]
            return {"action": "start", "prompt": singular, "selected_index": None}
        return {"action": "start", "prompt": prompt, "selected_index": None}

    # Bare object phrase from chat/voice: "keyboard", "red mug", etc.
    if text in {"stop", "cancel"}:
        return {"action": "stop"}
    if text in {"status"}:
        return {"action": "status"}
    return {"action": "start", "prompt": text, "selected_index": None}


def _index_detections(
    detections: list[tuple[list[float], float]],
) -> list[TrackedTarget]:
    """Sort L→R then T→B and assign 1-based indices."""
    items: list[tuple[float, float, list[float], float]] = []
    for xywh, score in detections:
        x, y, w, h = xywh
        cx = x + w / 2.0
        cy = y + h / 2.0
        items.append((cx, cy, [x, y, x + w, y + h], score))
    items.sort(key=lambda t: (t[0], t[1]))
    return [
        TrackedTarget(index=i + 1, bbox=box, score=score)
        for i, (_cx, _cy, box, score) in enumerate(items)
    ]


def _pick_sticky_target(
    targets: list[TrackedTarget],
    selected_index: int,
    locked_bbox: list[float] | None,
) -> TrackedTarget | None:
    if locked_bbox is not None and targets:
        best = max(targets, key=lambda t: _iou(t.bbox, locked_bbox))
        if _iou(best.bbox, locked_bbox) >= 0.15:
            return best
    return next((t for t in targets if t.index == selected_index), None)


def _iou(a: list[float], b: list[float]) -> float:
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    ix0, iy0 = max(ax0, bx0), max(ay0, by0)
    ix1, iy1 = min(ax1, bx1), min(ay1, by1)
    iw, ih = max(0.0, ix1 - ix0), max(0.0, iy1 - iy0)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = max(0.0, ax1 - ax0) * max(0.0, ay1 - ay0)
    area_b = max(0.0, bx1 - bx0) * max(0.0, by1 - by0)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def _normalize_prompt(text: str) -> str:
    out = text.strip().lower()
    if not out:
        return ""
    if not out.endswith("."):
        out += "."
    return out


def _clip_box_xywh(
    bbox_xywh: list[float],
    frame_w: int,
    frame_h: int,
) -> tuple[int, int, int, int]:
    x, y, w, h = bbox_xywh
    x = max(0, min(frame_w - 1, int(x)))
    y = max(0, min(frame_h - 1, int(y)))
    max_w = max(1, frame_w - x)
    max_h = max(1, frame_h - y)
    w = max(1, min(max_w, int(w)))
    h = max(1, min(max_h, int(h)))
    return x, y, w, h
