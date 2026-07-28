"""Smoke-test SAM 3 open-vocab video segmentation on a local MP4.

Usage:
  .\\.venv\\Scripts\\python.exe -m perception.test_sam3_video ^
      --video "C:\\path\\to\\video.mp4" --prompt horse

Requires:
  - CUDA torch
  - `pip install -e vision/third_party/sam3`
  - Hugging Face access to facebook/sam3 + `hf auth login`
"""

from __future__ import annotations

import argparse
import colorsys
import json
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
import torch


def _make_preview_clip(
    src: Path, dst: Path, max_side: int = 720, max_frames: int = 90
) -> Path:
    """Downscale + trim so an 8GB laptop GPU can finish a smoke test."""
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


def _color_for_id(obj_id: int) -> tuple[float, float, float]:
    h = (obj_id * 0.61803398875) % 1.0
    r, g, b = colorsys.hsv_to_rgb(h, 0.85, 1.0)
    return r, g, b


def _overlay_masks(frame_bgr: np.ndarray, outputs: dict) -> np.ndarray:
    """Blend SAM binary masks onto a BGR frame."""
    if outputs is None:
        return frame_bgr
    masks = outputs.get("out_binary_masks")
    obj_ids = outputs.get("out_obj_ids")
    if masks is None or obj_ids is None or len(obj_ids) == 0:
        return frame_bgr

    if torch.is_tensor(masks):
        masks = masks.detach().cpu().numpy()
    if torch.is_tensor(obj_ids):
        obj_ids = obj_ids.detach().cpu().numpy()

    out = frame_bgr.astype(np.float32)
    h, w = out.shape[:2]
    for i, obj_id in enumerate(obj_ids):
        mask = masks[i]
        if mask.dtype != np.bool_:
            mask = mask > 0.5
        if mask.shape[:2] != (h, w):
            mask = cv2.resize(
                mask.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST
            ).astype(bool)
        if not mask.any():
            continue
        color = np.array(_color_for_id(int(obj_id)), dtype=np.float32) * 255.0
        out[mask] = out[mask] * 0.45 + color * 0.55
        ys, xs = np.where(mask)
        x0, x1 = int(xs.min()), int(xs.max())
        y0, y1 = int(ys.min()), int(ys.max())
        cv2.rectangle(
            out, (x0, y0), (x1, y1), tuple(map(float, color)), 2
        )
        cv2.putText(
            out,
            f"id={int(obj_id)}",
            (x0, max(20, y0 - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            tuple(map(float, color)),
            2,
            cv2.LINE_AA,
        )
    return np.clip(out, 0, 255).astype(np.uint8)


def _write_overlay_video(
    video_path: Path,
    outputs_per_frame: dict[int, dict],
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
    writer = cv2.VideoWriter(
        str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h)
    )
    frame_idx = 0
    written = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        overlaid = _overlay_masks(frame, outputs_per_frame.get(frame_idx))
        writer.write(overlaid)
        if frame_idx % preview_stride == 0:
            cv2.imwrite(str(preview_dir / f"frame_{frame_idx:04d}.jpg"), overlaid)
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
    parser.add_argument("--prompt", type=str, default="horse")
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("data/perception_tests/sam3_206294"),
    )
    parser.add_argument("--max-side", type=int, default=720)
    parser.add_argument("--max-frames", type=int, default=90)
    parser.add_argument(
        "--skip-preview-clip",
        action="store_true",
        help="Run on the original video (may OOM on 8GB GPUs).",
    )
    args = parser.parse_args()

    if not args.video.exists():
        print(f"Video not found: {args.video}", file=sys.stderr)
        return 1

    args.out_dir.mkdir(parents=True, exist_ok=True)
    work_video = args.video
    if not args.skip_preview_clip:
        work_video = args.out_dir / "preview_input.mp4"
        print(
            f"Making preview clip ({args.max_side}px, max {args.max_frames} frames)..."
        )
        _make_preview_clip(
            args.video, work_video, max_side=args.max_side, max_frames=args.max_frames
        )

    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(
            f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB"
        )

    from sam3.model_builder import build_sam3_video_predictor

    print("Loading SAM 3 video predictor (downloads HF checkpoint on first run)...")
    try:
        predictor = build_sam3_video_predictor()
    except Exception as exc:  # noqa: BLE001 — surface HF/auth errors clearly
        msg = str(exc).lower()
        print("\nFailed to load SAM 3 checkpoint.", file=sys.stderr)
        print(exc, file=sys.stderr)
        if "401" in msg or "gated" in msg or "403" in msg or "authorized" in msg:
            print(
                "\nSAM 3 weights are gated on Hugging Face.\n"
                "1) Request access: https://huggingface.co/facebook/sam3\n"
                "2) Then run:  hf auth login\n"
                "3) Re-run this script.",
                file=sys.stderr,
            )
        return 2

    print(f"Starting session on {work_video} ...")
    response = predictor.handle_request(
        request=dict(type="start_session", resource_path=str(work_video))
    )
    session_id = response["session_id"]
    print(f"session_id={session_id}")

    print(f'Adding text prompt "{args.prompt}" on frame 0...')
    response = predictor.handle_request(
        request=dict(
            type="add_prompt",
            session_id=session_id,
            frame_index=0,
            text=args.prompt,
        )
    )
    frame0_out = response["outputs"]
    n0 = len(frame0_out.get("out_obj_ids", [])) if frame0_out else 0
    print(f"Frame 0 detections for '{args.prompt}': {n0}")

    print("Propagating through video...")
    outputs_per_frame: dict[int, dict] = {0: frame0_out}
    for response in predictor.handle_stream_request(
        request=dict(type="propagate_in_video", session_id=session_id)
    ):
        outputs_per_frame[response["frame_index"]] = response["outputs"]

    out_mp4 = args.out_dir / "overlay.mp4"
    preview_dir = args.out_dir / "preview_frames"
    n_frames = _write_overlay_video(
        work_video, outputs_per_frame, out_mp4, preview_dir
    )

    summary = {
        "source_video": str(args.video),
        "work_video": str(work_video),
        "prompt": args.prompt,
        "frame0_objects": n0,
        "propagated_frames": len(outputs_per_frame),
        "written_frames": n_frames,
        "overlay_mp4": str(out_mp4),
        "preview_dir": str(preview_dir),
    }
    summary_path = args.out_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"\nDone. Open: {out_mp4}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
