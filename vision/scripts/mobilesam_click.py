"""Browser click UI for MobileSAM (positive / negative points).

Open the local Gradio URL, click the image, refine with extra clicks.

Usage:
  .\\.venv\\Scripts\\python.exe -m perception.mobilesam_click ^
      --video "C:\\Users\\yj.park\\Downloads\\206294.mp4"
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import gradio as gr
import numpy as np
import torch


def _overlay(frame_bgr: np.ndarray, mask: np.ndarray | None) -> np.ndarray:
    if mask is None or not np.any(mask):
        return frame_bgr
    out = frame_bgr.astype(np.float32)
    color = np.array([0, 200, 255], dtype=np.float32)
    m = mask.astype(bool)
    out[m] = out[m] * 0.45 + color * 0.55
    return np.clip(out, 0, 255).astype(np.uint8)


def _draw_points(frame_bgr: np.ndarray, points: list[tuple[int, int, int]]) -> np.ndarray:
    out = frame_bgr.copy()
    for x, y, label in points:
        color = (0, 255, 0) if label == 1 else (0, 0, 255)
        cv2.circle(out, (int(x), int(y)), 6, color, -1, cv2.LINE_AA)
        cv2.circle(out, (int(x), int(y)), 8, (255, 255, 255), 1, cv2.LINE_AA)
    return out


def _bgr_to_rgb(frame_bgr: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--video",
        type=Path,
        default=Path(r"C:\Users\yj.park\Downloads\206294.mp4"),
    )
    parser.add_argument("--image", type=Path, default=None)
    parser.add_argument(
        "--sam-checkpoint",
        type=Path,
        default=Path("data/perception_tests/mobilesam_206294/weights/mobile_sam.pt"),
    )
    parser.add_argument("--start-frame", type=int, default=0)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7860)
    args = parser.parse_args()

    if not args.sam_checkpoint.exists():
        print(f"Missing MobileSAM weights: {args.sam_checkpoint}", file=sys.stderr)
        return 1

    device = "cuda" if torch.cuda.is_available() else "cpu"
    from mobile_sam import SamPredictor, sam_model_registry

    print(f"Loading MobileSAM on {device} ...")
    sam = sam_model_registry["vit_t"](checkpoint=str(args.sam_checkpoint))
    sam.to(device=device)
    sam.eval()
    predictor = SamPredictor(sam)

    frames: list[np.ndarray] = []
    if args.image is not None:
        img = cv2.imread(str(args.image))
        if img is None:
            print(f"Could not read image: {args.image}", file=sys.stderr)
            return 1
        frames = [img]
    else:
        if not args.video.exists():
            print(f"Video not found: {args.video}", file=sys.stderr)
            return 1
        cap = cv2.VideoCapture(str(args.video))
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frames.append(frame)
        cap.release()
        print(f"Loaded {len(frames)} frames")

    if not frames:
        print("No frames loaded.", file=sys.stderr)
        return 1

    state = {
        "idx": max(0, min(args.start_frame, len(frames) - 1)),
        "points": [],  # (x, y, label)
        "mask": None,
    }

    def _encode_image() -> np.ndarray:
        rgb = _bgr_to_rgb(frames[state["idx"]])
        predictor.set_image(rgb)
        return rgb

    def _render() -> np.ndarray:
        view = _overlay(frames[state["idx"]], state["mask"])
        view = _draw_points(view, state["points"])
        return _bgr_to_rgb(view)

    def _predict() -> None:
        if not state["points"]:
            state["mask"] = None
            return
        coords = np.array([[x, y] for x, y, _ in state["points"]], dtype=np.float32)
        labels = np.array([lab for _, _, lab in state["points"]], dtype=np.int32)
        masks, scores, _ = predictor.predict(
            point_coords=coords,
            point_labels=labels,
            multimask_output=True,
        )
        state["mask"] = masks[int(np.argmax(scores))]

    _encode_image()

    def on_select(mode: str, evt: gr.SelectData):
        # Gradio image select gives (x, y) in image pixel coords.
        x, y = evt.index
        label = 1 if mode == "Positive (include)" else 0
        state["points"].append((int(x), int(y), label))
        _predict()
        status = (
            f"frame {state['idx']}/{len(frames) - 1}  "
            f"points={len(state['points'])}  "
            f"last={'pos' if label == 1 else 'neg'} @ ({x},{y})"
        )
        return _render(), status

    def on_frame_change(frame_idx: int):
        state["idx"] = int(frame_idx)
        state["points"] = []
        state["mask"] = None
        _encode_image()
        return (
            _render(),
            f"frame {state['idx']}/{len(frames) - 1} — click an object",
        )

    def on_reset():
        state["points"] = []
        state["mask"] = None
        return _render(), f"frame {state['idx']}/{len(frames) - 1} — reset"

    def on_undo():
        if state["points"]:
            state["points"].pop()
            _predict()
        return (
            _render(),
            f"frame {state['idx']}/{len(frames) - 1}  points={len(state['points'])}",
        )

    with gr.Blocks(title="MobileSAM click") as demo:
        gr.Markdown(
            "## MobileSAM click\n"
            "Choose **Positive** or **Negative**, then click the image. "
            "Green = include, red = exclude."
        )
        with gr.Row():
            mode = gr.Radio(
                ["Positive (include)", "Negative (exclude)"],
                value="Positive (include)",
                label="Click mode",
            )
            frame_slider = gr.Slider(
                0,
                len(frames) - 1,
                value=state["idx"],
                step=1,
                label="Frame",
            )
        image = gr.Image(
            value=_render(),
            label="Click on the image",
            type="numpy",
            interactive=False,
        )
        status = gr.Textbox(
            value=f"frame {state['idx']}/{len(frames) - 1} — click an object",
            label="Status",
            interactive=False,
        )
        with gr.Row():
            reset_btn = gr.Button("Reset points")
            undo_btn = gr.Button("Undo last point")

        image.select(on_select, inputs=[mode], outputs=[image, status])
        frame_slider.release(on_frame_change, inputs=[frame_slider], outputs=[image, status])
        reset_btn.click(on_reset, outputs=[image, status])
        undo_btn.click(on_undo, outputs=[image, status])

    print(f"\nOpen in browser: http://{args.host}:{args.port}\n")
    demo.launch(server_name=args.host, server_port=args.port, inbrowser=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
