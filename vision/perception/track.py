"""Single-object SAM 2 video tracker.

Acquire once (point / box / mask) and then track that one target through video
without running open-vocabulary detection on every frame.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Optional

import cv2
import numpy as np
import torch
from transformers import Sam2VideoModel, Sam2VideoProcessor


@dataclass(frozen=True)
class TrackedFrame:
    frame_idx: int
    bbox: tuple[float, float, float, float] | None
    mask: np.ndarray | None
    score: float


class SingleObjectTracker:
    """Thin SAM 2 session wrapper for one object.

    The acquisition stage (click / text detector / etc.) is deliberately out of
    scope for this class. Caller provides one seed input once, then consumes the
    tracking iterator.
    """

    def __init__(
        self,
        model_id: str = "facebook/sam2.1-hiera-tiny",
        *,
        obj_id: int = 1,
        device: str | None = None,
        dtype: torch.dtype = torch.float16,
    ) -> None:
        self.model_id = model_id
        self.obj_id = int(obj_id)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = dtype if self.device == "cuda" else torch.float32

        self.processor = Sam2VideoProcessor.from_pretrained(model_id)
        self.model = Sam2VideoModel.from_pretrained(
            model_id,
            torch_dtype=self.dtype,
        ).to(self.device)
        self.model.eval()

        self._session = None
        self._start_frame_idx: int | None = None
        self._previous_area_ratio: float | None = None

    def start(
        self,
        video_path: str | Path,
        init_frame_idx: int = 0,
        *,
        point: tuple[float, float] | None = None,
        box: tuple[float, float, float, float] | None = None,
        mask: np.ndarray | None = None,
    ) -> None:
        if sum(v is not None for v in (point, box, mask)) != 1:
            raise ValueError("Provide exactly one of point, box, or mask.")

        video_path = Path(video_path)
        if not video_path.exists():
            raise FileNotFoundError(f"Video not found: {video_path}")

        # Avoid torchvision/torchcodec decoding issues by decoding via OpenCV here.
        frames_rgb = _read_video_rgb_frames(video_path)
        self._session = self.processor.init_video_session(
            video=frames_rgb,
            inference_device=self.device,
            dtype=self.dtype,
        )
        self._start_frame_idx = int(init_frame_idx)

        if point is not None:
            px, py = point
            self.processor.add_inputs_to_inference_session(
                self._session,
                frame_idx=self._start_frame_idx,
                obj_ids=[self.obj_id],
                input_points=[[[[float(px), float(py)]]]],
                input_labels=[[[1]]],
            )
            return

        if box is not None:
            x0, y0, x1, y1 = box
            self.processor.add_inputs_to_inference_session(
                self._session,
                frame_idx=self._start_frame_idx,
                obj_ids=[self.obj_id],
                input_boxes=[[[float(x0), float(y0), float(x1), float(y1)]]],
            )
            return

        self.processor.add_inputs_to_inference_session(
            self._session,
            frame_idx=self._start_frame_idx,
            obj_ids=[self.obj_id],
            input_masks=[mask.astype(bool)],
        )

    def track(
        self,
        *,
        max_frames: int | None = None,
        reverse: bool = False,
        show_progress: bool = False,
    ) -> Iterator[TrackedFrame]:
        if self._session is None or self._start_frame_idx is None:
            raise RuntimeError("Call start(...) before track().")

        iterator = self.model.propagate_in_video_iterator(
            self._session,
            start_frame_idx=self._start_frame_idx,
            max_frame_num_to_track=max_frames,
            reverse=reverse,
            show_progress_bar=show_progress,
        )
        for output in iterator:
            yield self._to_tracked_frame(output)

    def is_lost(
        self,
        tracked: TrackedFrame,
        *,
        min_score: float = -4.0,
        min_area_ratio: float = 0.0025,
        max_area_drop_ratio: float = 0.0,
    ) -> bool:
        """Conservative lost heuristic for one-object follow.

        - score too low -> lost
        - absolute area too small -> lost
        - abrupt area collapse vs previous frame -> likely lost/occluded
        """
        if tracked.mask is None:
            return True
        if tracked.score < float(min_score):
            return True

        area_ratio = float(np.count_nonzero(tracked.mask)) / float(tracked.mask.size)
        if area_ratio < float(min_area_ratio):
            return True

        if max_area_drop_ratio > 0.0 and self._previous_area_ratio is not None:
            if area_ratio < self._previous_area_ratio * float(max_area_drop_ratio):
                self._previous_area_ratio = area_ratio
                return True

        self._previous_area_ratio = area_ratio
        return False

    def _to_tracked_frame(self, output) -> TrackedFrame:
        frame_idx = int(output.frame_idx if output.frame_idx is not None else -1)
        pred_masks = output.pred_masks
        scores = output.object_score_logits
        obj_ids = output.object_ids or []

        if pred_masks is None or len(obj_ids) == 0:
            return TrackedFrame(frame_idx=frame_idx, bbox=None, mask=None, score=float("-inf"))

        if torch.is_tensor(pred_masks):
            masks_np = pred_masks.detach().float().cpu().numpy()
        else:
            masks_np = np.asarray(pred_masks)
        # Expected shape: [num_objects, 1, H, W] or [num_objects, H, W]
        if masks_np.ndim == 4:
            masks_np = masks_np[:, 0, :, :]
        mask = (masks_np[0] > 0.0).astype(np.uint8)
        bbox = _bbox_from_mask(mask)

        score = float("-inf")
        if scores is not None:
            if torch.is_tensor(scores):
                score_arr = scores.detach().float().cpu().numpy()
            else:
                score_arr = np.asarray(scores)
            score = float(np.ravel(score_arr)[0])

        return TrackedFrame(frame_idx=frame_idx, bbox=bbox, mask=mask, score=score)


def _bbox_from_mask(mask: np.ndarray) -> tuple[float, float, float, float] | None:
    ys, xs = np.where(mask > 0)
    if xs.size == 0 or ys.size == 0:
        return None
    return float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max())


def _read_video_rgb_frames(video_path: Path) -> list[np.ndarray]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")
    frames: list[np.ndarray] = []
    while True:
        ok, frame_bgr = cap.read()
        if not ok:
            break
        frames.append(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
    cap.release()
    if not frames:
        raise RuntimeError(f"No frames decoded from video: {video_path}")
    return frames
