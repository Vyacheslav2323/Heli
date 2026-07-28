"""Smoke-test Grounding DINO open-vocab detection on a local video.

Usage:
  .\\.venv\\Scripts\\python.exe -m perception.test_grounding_dino ^
      --video "C:\\path\\to\\video.mp4" --prompt "horse."
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image


def _draw_detections(frame_bgr: np.ndarray, boxes, scores, labels) -> np.ndarray:
    out = frame_bgr.copy()
    for box, score, label in zip(boxes, scores, labels):
        x0, y0, x1, y1 = [int(v) for v in box]
        color = (0, 200, 255)
        cv2.rectangle(out, (x0, y0), (x1, y1), color, 2)
        text = f"{label} {float(score):.2f}"
        cv2.putText(
            out,
            text,
            (x0, max(20, y0 - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            2,
            cv2.LINE_AA,
        )
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--video",
        type=Path,
        default=Path(r"C:\Users\yj.park\Downloads\206294.mp4"),
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="horse.",
        help='Lowercase grounded phrases ending with "." e.g. "horse. pole."',
    )
    parser.add_argument(
        "--model-id",
        default="IDEA-Research/grounding-dino-tiny",
        help="HF model id (tiny|base).",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("data/perception_tests/groundingdino_206294"),
    )
    parser.add_argument("--box-threshold", type=float, default=0.25)
    parser.add_argument("--text-threshold", type=float, default=0.25)
    parser.add_argument(
        "--stride",
        type=int,
        default=1,
        help="Run detection every N frames (copy previous boxes in between).",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=0,
        help="If >0, stop after this many frames.",
    )
    parser.add_argument(
        "--max-side",
        type=int,
        default=800,
        help="Resize longest side before inference (0 = native).",
    )
    args = parser.parse_args()

    if not args.video.exists():
        print(f"Video not found: {args.video}", file=sys.stderr)
        return 1

    prompt = args.prompt.strip().lower()
    if not prompt.endswith("."):
        prompt += "."

    args.out_dir.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

    print(f"Loading {args.model_id} ...")
    processor = AutoProcessor.from_pretrained(args.model_id)
    model = AutoModelForZeroShotObjectDetection.from_pretrained(args.model_id).to(
        device
    )
    model.eval()

    cap = cv2.VideoCapture(str(args.video))
    if not cap.isOpened():
        print(f"Could not open video: {args.video}", file=sys.stderr)
        return 1

    fps_in = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out_path = args.out_dir / "overlay.mp4"
    writer = cv2.VideoWriter(
        str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps_in, (w, h)
    )

    n_frames = 0
    n_dets = 0
    per_frame_counts: list[int] = []
    last_boxes = last_scores = last_labels = []
    t0 = time.perf_counter()
    infer_sec = 0.0

    print(f'Prompt: "{prompt}"  stride={args.stride}  max_side={args.max_side}')
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if args.max_frames and n_frames >= args.max_frames:
            break

        run_infer = n_frames % args.stride == 0
        if run_infer:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(rgb)
            if args.max_side and max(image.size) > args.max_side:
                scale = args.max_side / max(image.size)
                image = image.resize(
                    (int(image.width * scale), int(image.height * scale)),
                    Image.BILINEAR,
                )
                sx = w / image.width
                sy = h / image.height
            else:
                sx = sy = 1.0

            inputs = processor(images=image, text=prompt, return_tensors="pt").to(
                device
            )
            t_inf0 = time.perf_counter()
            with torch.no_grad():
                outputs = model(**inputs)
            infer_sec += time.perf_counter() - t_inf0

            results = processor.post_process_grounded_object_detection(
                outputs,
                inputs.input_ids,
                threshold=args.box_threshold,
                text_threshold=args.text_threshold,
                target_sizes=[(image.height, image.width)],
            )[0]

            boxes = results["boxes"].detach().cpu().numpy()
            scores = results["scores"].detach().cpu().numpy()
            labels = results["labels"]
            if isinstance(labels, torch.Tensor):
                labels = [str(x) for x in labels.tolist()]
            else:
                labels = [str(x) for x in labels]

            # Map boxes back to original frame size if we resized for inference.
            if sx != 1.0 or sy != 1.0:
                boxes = boxes.copy()
                boxes[:, [0, 2]] *= sx
                boxes[:, [1, 3]] *= sy

            last_boxes, last_scores, last_labels = boxes, scores, labels

        count = len(last_boxes)
        n_dets += count
        per_frame_counts.append(count)
        writer.write(_draw_detections(frame, last_boxes, last_scores, last_labels))
        n_frames += 1
        if n_frames % 30 == 0:
            print(f"  frame {n_frames}  dets={count}")

    cap.release()
    writer.release()
    elapsed = time.perf_counter() - t0
    fps = n_frames / elapsed if elapsed else 0.0
    infer_fps = (
        (n_frames / args.stride) / infer_sec if infer_sec and args.stride else 0.0
    )

    summary = {
        "source_video": str(args.video),
        "model_id": args.model_id,
        "prompt": prompt,
        "box_threshold": args.box_threshold,
        "text_threshold": args.text_threshold,
        "stride": args.stride,
        "max_side": args.max_side,
        "frames": n_frames,
        "elapsed_sec": round(elapsed, 2),
        "wall_fps": round(fps, 2),
        "infer_fps_approx": round(infer_fps, 2),
        "avg_detections_per_frame": round((n_dets / n_frames) if n_frames else 0, 2),
        "max_detections_in_frame": max(per_frame_counts) if per_frame_counts else 0,
        "frame0_detections": per_frame_counts[0] if per_frame_counts else 0,
        "overlay": str(out_path),
    }
    (args.out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))
    print(f"\nDone. Open: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
