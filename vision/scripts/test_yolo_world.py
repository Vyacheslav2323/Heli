"""Smoke-test YOLO-World open-vocab detection (+ optional tracking).

Usage:
  .\\.venv\\Scripts\\python.exe -m perception.test_yolo_world ^
      --video "C:\\path\\to\\video.mp4" --classes horse
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--video",
        type=Path,
        default=Path(r"C:\Users\yj.park\Downloads\206294.mp4"),
    )
    parser.add_argument(
        "--classes",
        nargs="+",
        default=["horse"],
        help="Open-vocab class prompts (space-separated).",
    )
    parser.add_argument(
        "--model",
        default="yolov8s-world.pt",
        help="YOLO-World checkpoint (s/m/l-world.pt).",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("data/perception_tests/yoloworld_206294"),
    )
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument(
        "--track",
        action="store_true",
        help="Use BoT-SORT tracking for stable IDs across frames.",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=0,
        help="If >0, only process this many frames (via ffmpeg trim).",
    )
    args = parser.parse_args()

    if not args.video.exists():
        print(f"Video not found: {args.video}", file=sys.stderr)
        return 1

    args.out_dir.mkdir(parents=True, exist_ok=True)

    import torch
    from ultralytics import YOLO

    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    print(f"Loading {args.model} ...")
    model = YOLO(args.model)
    model.set_classes(list(args.classes))
    print(f"Classes: {args.classes}")

    source = str(args.video)
    # Absolute project path — Ultralytics otherwise nests under a global runs/ cwd.
    project_dir = str(args.out_dir.resolve())
    predict_kwargs = dict(
        source=source,
        imgsz=args.imgsz,
        conf=args.conf,
        device=0 if torch.cuda.is_available() else "cpu",
        save=True,
        project=project_dir,
        name="run",
        exist_ok=True,
        stream=True,
        verbose=False,
    )

    t0 = time.perf_counter()
    n_frames = 0
    n_dets = 0
    per_frame_counts: list[int] = []

    if args.track:
        print("Running track() ...")
        results = model.track(**predict_kwargs, persist=True)
    else:
        print("Running predict() ...")
        results = model.predict(**predict_kwargs)

    for r in results:
        n_frames += 1
        count = 0 if r.boxes is None else len(r.boxes)
        n_dets += count
        per_frame_counts.append(count)
        if args.max_frames and n_frames >= args.max_frames:
            break

    elapsed = time.perf_counter() - t0
    fps = n_frames / elapsed if elapsed > 0 else 0.0
    avg_dets = (n_dets / n_frames) if n_frames else 0.0

    # Ultralytics writes under out_dir/run/
    run_dir = args.out_dir / "run"
    overlay_candidates = list(run_dir.glob("*.mp4")) + list(run_dir.glob("*.avi"))
    overlay = str(overlay_candidates[0]) if overlay_candidates else None

    summary = {
        "source_video": str(args.video),
        "model": args.model,
        "classes": args.classes,
        "track": args.track,
        "imgsz": args.imgsz,
        "conf": args.conf,
        "frames": n_frames,
        "elapsed_sec": round(elapsed, 2),
        "fps": round(fps, 2),
        "avg_detections_per_frame": round(avg_dets, 2),
        "max_detections_in_frame": max(per_frame_counts) if per_frame_counts else 0,
        "frame0_detections": per_frame_counts[0] if per_frame_counts else 0,
        "overlay": overlay,
        "run_dir": str(run_dir),
    }
    summary_path = args.out_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    if overlay:
        print(f"\nDone. Open: {overlay}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
