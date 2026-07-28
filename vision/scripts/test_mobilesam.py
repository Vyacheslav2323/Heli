"""Grounding DINO boxes + MobileSAM masks on a local video.

Usage:
  .\\.venv\\Scripts\\python.exe -m perception.test_mobilesam ^
      --video "C:\\path\\to\\video.mp4" --prompt "horse."
"""

from __future__ import annotations

import argparse
import colorsys
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image


def _color_for_id(i: int) -> tuple[int, int, int]:
    h = (i * 0.61803398875) % 1.0
    r, g, b = colorsys.hsv_to_rgb(h, 0.85, 1.0)
    return int(b * 255), int(g * 255), int(r * 255)  # BGR


def _overlay_masks(
    frame_bgr: np.ndarray,
    masks: list[np.ndarray],
    boxes: np.ndarray | None = None,
    labels: list[str] | None = None,
    scores: np.ndarray | None = None,
) -> np.ndarray:
    out = frame_bgr.astype(np.float32)
    for i, mask in enumerate(masks):
        if mask.dtype != np.bool_:
            mask = mask > 0.5
        if not mask.any():
            continue
        color = np.array(_color_for_id(i), dtype=np.float32)
        out[mask] = out[mask] * 0.45 + color * 0.55
        if boxes is not None and i < len(boxes):
            x0, y0, x1, y1 = [int(v) for v in boxes[i]]
            cv2.rectangle(out, (x0, y0), (x1, y1), tuple(map(int, color)), 2)
            tag = labels[i] if labels and i < len(labels) else f"id={i}"
            if scores is not None and i < len(scores):
                tag = f"{tag} {float(scores[i]):.2f}"
            cv2.putText(
                out,
                tag,
                (x0, max(20, y0 - 6)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                tuple(map(int, color)),
                2,
                cv2.LINE_AA,
            )
    return np.clip(out, 0, 255).astype(np.uint8)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--video",
        type=Path,
        default=Path(r"C:\Users\yj.park\Downloads\206294.mp4"),
    )
    parser.add_argument("--prompt", type=str, default="horse.")
    parser.add_argument(
        "--detector",
        default="IDEA-Research/grounding-dino-tiny",
        help="HF Grounding DINO id used to propose boxes.",
    )
    parser.add_argument(
        "--sam-checkpoint",
        type=Path,
        default=Path("data/perception_tests/mobilesam_206294/weights/mobile_sam.pt"),
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("data/perception_tests/mobilesam_206294"),
    )
    parser.add_argument("--box-threshold", type=float, default=0.3)
    parser.add_argument("--text-threshold", type=float, default=0.25)
    parser.add_argument("--max-objects", type=int, default=8)
    parser.add_argument("--stride", type=int, default=2)
    parser.add_argument("--max-side", type=int, default=800)
    parser.add_argument("--max-frames", type=int, default=0)
    args = parser.parse_args()

    if not args.video.exists():
        print(f"Video not found: {args.video}", file=sys.stderr)
        return 1
    if not args.sam_checkpoint.exists():
        print(f"MobileSAM weights missing: {args.sam_checkpoint}", file=sys.stderr)
        return 1

    prompt = args.prompt.strip().lower()
    if not prompt.endswith("."):
        prompt += "."

    args.out_dir.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    from mobile_sam import SamPredictor, sam_model_registry
    from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

    print(f"Loading detector {args.detector} ...")
    processor = AutoProcessor.from_pretrained(args.detector)
    detector = AutoModelForZeroShotObjectDetection.from_pretrained(args.detector).to(
        device
    )
    detector.eval()

    print(f"Loading MobileSAM from {args.sam_checkpoint} ...")
    sam = sam_model_registry["vit_t"](checkpoint=str(args.sam_checkpoint))
    sam.to(device=device)
    sam.eval()
    predictor = SamPredictor(sam)

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
    n_masks_total = 0
    per_frame_counts: list[int] = []
    last_masks: list[np.ndarray] = []
    last_boxes = np.zeros((0, 4))
    last_labels: list[str] = []
    last_scores = np.zeros((0,))
    t0 = time.perf_counter()
    det_sec = 0.0
    sam_sec = 0.0

    print(
        f'Prompt="{prompt}" stride={args.stride} max_objects={args.max_objects} '
        f"max_side={args.max_side}"
    )

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if args.max_frames and n_frames >= args.max_frames:
            break

        if n_frames % args.stride == 0:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(rgb)
            if args.max_side and max(image.size) > args.max_side:
                scale = args.max_side / max(image.size)
                image_small = image.resize(
                    (int(image.width * scale), int(image.height * scale)),
                    Image.BILINEAR,
                )
                sx = w / image_small.width
                sy = h / image_small.height
            else:
                image_small = image
                sx = sy = 1.0

            inputs = processor(
                images=image_small, text=prompt, return_tensors="pt"
            ).to(device)
            t_d0 = time.perf_counter()
            with torch.no_grad():
                outputs = detector(**inputs)
            det_sec += time.perf_counter() - t_d0

            results = processor.post_process_grounded_object_detection(
                outputs,
                inputs.input_ids,
                threshold=args.box_threshold,
                text_threshold=args.text_threshold,
                target_sizes=[(image_small.height, image_small.width)],
            )[0]

            boxes = results["boxes"].detach().cpu().numpy()
            scores = results["scores"].detach().cpu().numpy()
            labels = results["labels"]
            if isinstance(labels, torch.Tensor):
                labels = [str(x) for x in labels.tolist()]
            else:
                labels = [str(x) for x in labels]

            if sx != 1.0 or sy != 1.0:
                boxes = boxes.copy()
                boxes[:, [0, 2]] *= sx
                boxes[:, [1, 3]] *= sy

            # Keep top-k by score.
            if len(scores) > args.max_objects:
                keep = np.argsort(-scores)[: args.max_objects]
                boxes = boxes[keep]
                scores = scores[keep]
                labels = [labels[i] for i in keep]

            # MobileSAM expects RGB uint8 full-res frame.
            t_s0 = time.perf_counter()
            predictor.set_image(rgb)
            masks: list[np.ndarray] = []
            for box in boxes:
                box_xyxy = np.array(box, dtype=np.float32)
                mask, _iou, _ = predictor.predict(
                    box=box_xyxy,
                    multimask_output=False,
                )
                masks.append(mask[0])
            sam_sec += time.perf_counter() - t_s0

            last_masks, last_boxes, last_labels, last_scores = (
                masks,
                boxes,
                labels,
                scores,
            )

        n_masks_total += len(last_masks)
        per_frame_counts.append(len(last_masks))
        writer.write(
            _overlay_masks(frame, last_masks, last_boxes, last_labels, last_scores)
        )
        n_frames += 1
        if n_frames % 30 == 0:
            print(f"  frame {n_frames}  masks={len(last_masks)}")

    cap.release()
    writer.release()
    elapsed = time.perf_counter() - t0
    processed = max(1, (n_frames + args.stride - 1) // args.stride)

    summary = {
        "source_video": str(args.video),
        "pipeline": "grounding-dino-tiny -> MobileSAM (box prompts)",
        "prompt": prompt,
        "detector": args.detector,
        "sam_checkpoint": str(args.sam_checkpoint),
        "box_threshold": args.box_threshold,
        "max_objects": args.max_objects,
        "stride": args.stride,
        "max_side": args.max_side,
        "frames": n_frames,
        "elapsed_sec": round(elapsed, 2),
        "wall_fps": round(n_frames / elapsed if elapsed else 0, 2),
        "detector_fps_approx": round(processed / det_sec if det_sec else 0, 2),
        "mobilesam_fps_approx": round(processed / sam_sec if sam_sec else 0, 2),
        "avg_masks_per_frame": round(
            (n_masks_total / n_frames) if n_frames else 0, 2
        ),
        "frame0_masks": per_frame_counts[0] if per_frame_counts else 0,
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
