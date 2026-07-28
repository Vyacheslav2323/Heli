"""Smoke-test acquire-once + SAM2 single-object tracking on a local video.

Usage examples:
  .\\.venv\\Scripts\\python.exe -m perception.test_track_single_object ^
      --video "C:\\path\\to\\video.mp4" --acquire click --x 820 --y 430

  .\\.venv\\Scripts\\python.exe -m perception.test_track_single_object ^
      --video "C:\\path\\to\\video.mp4" --acquire text --prompt "horse."
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image

from perception.track import SingleObjectTracker, TrackedFrame


def _make_preview_clip(
    src: Path, dst: Path, max_side: int = 720, max_frames: int = 120
) -> Path:
    dst.parent.mkdir(parents=True, exist_ok=True)
    vf = (
        f"scale='if(gt(iw,ih),min({max_side},iw),-2)':"
        f"'if(gt(ih,iw),min({max_side},ih),-2)',"
        f"select=lt(n\\,{max_frames})"
    )
    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(src),
        "-vf",
        vf,
        "-vsync",
        "vfr",
        "-an",
        str(dst),
    ]
    subprocess.run(cmd, check=True, capture_output=True)
    return dst


def _first_frame(video_path: Path) -> np.ndarray:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise RuntimeError(f"Video has no frames: {video_path}")
    return frame


def _acquire_mask_from_click(
    frame_bgr: np.ndarray,
    *,
    click_x: float,
    click_y: float,
    sam_checkpoint: Path,
) -> np.ndarray:
    if not sam_checkpoint.exists():
        raise FileNotFoundError(f"MobileSAM checkpoint missing: {sam_checkpoint}")

    from mobile_sam import SamPredictor, sam_model_registry

    device = "cuda" if torch.cuda.is_available() else "cpu"
    sam = sam_model_registry["vit_t"](checkpoint=str(sam_checkpoint))
    sam.to(device=device)
    sam.eval()

    predictor = SamPredictor(sam)
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    predictor.set_image(rgb)
    point_coords = np.array([[float(click_x), float(click_y)]], dtype=np.float32)
    point_labels = np.array([1], dtype=np.int32)
    masks, scores, _ = predictor.predict(
        point_coords=point_coords,
        point_labels=point_labels,
        multimask_output=True,
    )
    return masks[int(np.argmax(scores))].astype(np.uint8)


def _acquire_box_from_text(
    frame_bgr: np.ndarray,
    *,
    prompt: str,
    detector_id: str,
    box_threshold: float,
    text_threshold: float,
) -> tuple[np.ndarray, float]:
    normalized_prompt = prompt.strip().lower()
    if not normalized_prompt.endswith("."):
        normalized_prompt += "."

    device = "cuda" if torch.cuda.is_available() else "cpu"
    from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

    processor = AutoProcessor.from_pretrained(detector_id)
    detector = AutoModelForZeroShotObjectDetection.from_pretrained(detector_id).to(device)
    detector.eval()

    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(rgb)
    inputs = processor(images=image, text=normalized_prompt, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = detector(**inputs)

    results = processor.post_process_grounded_object_detection(
        outputs,
        inputs.input_ids,
        threshold=box_threshold,
        text_threshold=text_threshold,
        target_sizes=[(image.height, image.width)],
    )[0]
    boxes = results["boxes"].detach().cpu().numpy()
    scores = results["scores"].detach().cpu().numpy()
    if len(boxes) == 0:
        raise RuntimeError(f'No detections for prompt "{normalized_prompt}" on frame 0.')
    best_idx = int(np.argmax(scores))
    return boxes[best_idx], float(scores[best_idx])


def _overlay_one(frame_bgr: np.ndarray, tracked: TrackedFrame | None, lost: bool = False) -> np.ndarray:
    out = frame_bgr.astype(np.float32)
    if tracked is not None and tracked.mask is not None:
        mask = tracked.mask.astype(bool)
        if mask.shape[:2] != out.shape[:2]:
            mask = cv2.resize(
                mask.astype(np.uint8),
                (out.shape[1], out.shape[0]),
                interpolation=cv2.INTER_NEAREST,
            ).astype(bool)
        color = np.array([0.0, 200.0, 255.0], dtype=np.float32)
        out[mask] = out[mask] * 0.45 + color * 0.55
        if tracked.bbox is not None:
            x0, y0, x1, y1 = [int(v) for v in tracked.bbox]
            cv2.rectangle(out, (x0, y0), (x1, y1), (0, 200, 255), 2)
            tag = f"id=1 score={tracked.score:.3f}"
            if lost:
                tag += " LOST?"
            cv2.putText(
                out,
                tag,
                (x0, max(20, y0 - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 200, 255),
                2,
                cv2.LINE_AA,
            )
    return np.clip(out, 0, 255).astype(np.uint8)


def _write_overlay_video(
    video_path: Path,
    tracked_by_frame: dict[int, TrackedFrame],
    lost_flags: dict[int, bool],
    out_path: Path,
    preview_dir: Path,
    preview_stride: int = 15,
) -> int:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    preview_dir.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

    frame_idx = 0
    written = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        tracked = tracked_by_frame.get(frame_idx)
        rendered = _overlay_one(frame, tracked, lost=lost_flags.get(frame_idx, False))
        writer.write(rendered)
        if frame_idx % preview_stride == 0:
            cv2.imwrite(str(preview_dir / f"frame_{frame_idx:04d}.jpg"), rendered)
        frame_idx += 1
        written += 1

    cap.release()
    writer.release()
    return written


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--video",
        type=Path,
        default=Path(r"C:\Users\yj.park\Downloads\206294.mp4"),
    )
    parser.add_argument("--acquire", choices=["click", "text"], required=True)
    parser.add_argument("--x", type=float, default=None, help="Click X pixel (for --acquire click).")
    parser.add_argument("--y", type=float, default=None, help="Click Y pixel (for --acquire click).")
    parser.add_argument("--prompt", type=str, default="horse.", help="Text prompt for --acquire text.")
    parser.add_argument(
        "--detector",
        default="IDEA-Research/grounding-dino-tiny",
        help="HF detector id for text acquisition.",
    )
    parser.add_argument(
        "--sam-checkpoint",
        type=Path,
        default=Path("data/perception_tests/mobilesam_206294/weights/mobile_sam.pt"),
    )
    parser.add_argument(
        "--tracker-model",
        default="facebook/sam2.1-hiera-tiny",
        help="HF SAM2 video model id.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("data/perception_tests/track_single_206294"),
    )
    parser.add_argument("--box-threshold", type=float, default=0.25)
    parser.add_argument("--text-threshold", type=float, default=0.25)
    parser.add_argument("--max-side", type=int, default=720)
    parser.add_argument("--max-frames", type=int, default=120)
    parser.add_argument(
        "--skip-preview-clip",
        action="store_true",
        help="Run on original video instead of downscaled preview clip.",
    )
    args = parser.parse_args()

    if not args.video.exists():
        print(f"Video not found: {args.video}", file=sys.stderr)
        return 1

    if args.acquire == "click" and (args.x is None or args.y is None):
        print("--acquire click requires --x and --y.", file=sys.stderr)
        return 1

    args.out_dir.mkdir(parents=True, exist_ok=True)

    work_video = args.video
    if not args.skip_preview_clip:
        work_video = args.out_dir / "preview_input.mp4"
        print(
            f"Making preview clip ({args.max_side}px, max {args.max_frames} frames) ..."
        )
        _make_preview_clip(args.video, work_video, max_side=args.max_side, max_frames=args.max_frames)

    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    frame0 = _first_frame(work_video)

    acquire_t0 = time.perf_counter()
    acquire_payload: dict[str, object] = {}
    if args.acquire == "click":
        print(f"Acquire mode: click @ ({args.x}, {args.y}) -> MobileSAM")
        mask0 = _acquire_mask_from_click(
            frame0,
            click_x=float(args.x),
            click_y=float(args.y),
            sam_checkpoint=args.sam_checkpoint,
        )
        acquire_payload = {"mask": mask0}
    else:
        print(f'Acquire mode: text "{args.prompt}" -> Grounding DINO')
        box0, det_score = _acquire_box_from_text(
            frame0,
            prompt=args.prompt,
            detector_id=args.detector,
            box_threshold=args.box_threshold,
            text_threshold=args.text_threshold,
        )
        acquire_payload = {"box": tuple(float(v) for v in box0)}
        print(f"Frame0 best detection score: {det_score:.3f}")
    acquire_sec = time.perf_counter() - acquire_t0

    tracker = SingleObjectTracker(model_id=args.tracker_model)
    tracker.start(work_video, init_frame_idx=0, **acquire_payload)

    print("Tracking one object with SAM2 ...")
    track_t0 = time.perf_counter()
    tracked_by_frame: dict[int, TrackedFrame] = {}
    lost_flags: dict[int, bool] = {}
    track_scores: list[float] = []
    bboxes: list[tuple[float, float, float, float]] = []
    lost_events = 0
    for tracked in tracker.track(show_progress=True):
        tracked_by_frame[tracked.frame_idx] = tracked
        if tracked.score != float("-inf"):
            track_scores.append(tracked.score)
        if tracked.bbox is not None:
            bboxes.append(tracked.bbox)
        lost = tracker.is_lost(tracked)
        lost_flags[tracked.frame_idx] = lost
        if lost:
            lost_events += 1
    track_sec = time.perf_counter() - track_t0

    out_mp4 = args.out_dir / f"overlay_{args.acquire}.mp4"
    preview_dir = args.out_dir / f"preview_frames_{args.acquire}"
    written_frames = _write_overlay_video(work_video, tracked_by_frame, lost_flags, out_mp4, preview_dir)

    tracked_frames = len(tracked_by_frame)
    tracker_fps = (tracked_frames / track_sec) if track_sec else 0.0

    summary = {
        "source_video": str(args.video),
        "work_video": str(work_video),
        "acquire_mode": args.acquire,
        "tracker_model": args.tracker_model,
        "tracked_frames": tracked_frames,
        "written_frames": written_frames,
        "acquire_sec": round(acquire_sec, 3),
        "track_sec": round(track_sec, 3),
        "tracker_only_fps": round(tracker_fps, 2),
        "mean_score": round(float(np.mean(track_scores)) if track_scores else float("nan"), 4),
        "min_score": round(float(np.min(track_scores)) if track_scores else float("nan"), 4),
        "lost_events": lost_events,
        "bbox_coverage": len(bboxes),
        "overlay": str(out_mp4),
        "preview_dir": str(preview_dir),
    }
    if args.acquire == "click":
        summary["click_xy"] = [args.x, args.y]
    else:
        summary["prompt"] = args.prompt
        summary["detector"] = args.detector
        summary["box_threshold"] = args.box_threshold
        summary["text_threshold"] = args.text_threshold

    summary_path = args.out_dir / f"summary_{args.acquire}.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"\nDone. Open: {out_mp4}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
