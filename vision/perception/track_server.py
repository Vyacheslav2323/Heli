from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Any

import numpy as np
import torch
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Local imports: this file is launched as a script (same pattern as mobilesam_server.py).
from frame_detector import (
    FramePromptTracker,
    TrackingState,
    decode_upload_image,
    parse_track_command,
)
from mobilesam_server import MobileSamService, SegmentPoint

DEFAULT_CHECKPOINT = Path("vision/third_party/MobileSAM/weights/mobile_sam.pt")


class CommandRequest(BaseModel):
    text: str


tracker = FramePromptTracker()
sam_service = MobileSamService(
    Path(os.getenv("MOBILESAM_CHECKPOINT", str(DEFAULT_CHECKPOINT)))
)
app = FastAPI(title="Thin Client Track API", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5174", "http://127.0.0.1:5174"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _state_to_dict(state: TrackingState) -> dict[str, Any]:
    return state.to_dict()


def _mask_to_bbox(mask: np.ndarray) -> list[float] | None:
    ys, xs = np.where(mask)
    if len(xs) == 0 or len(ys) == 0:
        return None
    x0 = float(xs.min())
    y0 = float(ys.min())
    x1 = float(xs.max() + 1)
    y1 = float(ys.max() + 1)
    if x1 <= x0 or y1 <= y0:
        return None
    return [x0, y0, x1, y1]


@app.get("/health")
def health() -> dict[str, Any]:
    checkpoint_exists = sam_service._checkpoint.exists()
    detector_ready = tracker._detector is not None and tracker._processor is not None
    device = (
        tracker._device
        if tracker._detector is not None
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    # Server is up once process is listening; detector may still be warming.
    return {
        "ok": True,
        "ready": True,
        "detector_ready": detector_ready,
        "checkpoint_ok": checkpoint_exists,
        "device": device,
        "checkpoint": str(sam_service._checkpoint),
    }


@app.get("/state")
def state() -> dict[str, Any]:
    return {"state": _state_to_dict(tracker.snapshot())}


@app.post("/command")
def command(payload: CommandRequest) -> dict[str, Any]:
    parsed = parse_track_command(payload.text)
    if parsed is None:
        return {
            "ok": False,
            "state": _state_to_dict(tracker.snapshot()),
            "message": "No tracking command detected.",
        }

    action = parsed.get("action")
    if action == "start":
        # Hard stop first so in-flight frames/redectects cannot restore the old target.
        tracker.stop_tracking()
        state = tracker.start_tracking(
            str(parsed.get("prompt", "")),
            selected_index=parsed.get("selected_index"),
        )
    elif action == "select":
        state = tracker.select_index(int(parsed["index"]))
    elif action == "all":
        state = tracker.track_all()
    elif action == "stop":
        state = tracker.stop_tracking()
    elif action == "status":
        state = tracker.snapshot()
    elif action == "help":
        state = tracker.snapshot()
        return {
            "ok": False,
            "state": _state_to_dict(state),
            "message": "Try: track <object>, track <object> <index>, track all, stop tracking.",
        }
    else:
        state = tracker.snapshot()
        return {
            "ok": False,
            "state": _state_to_dict(state),
            "message": f"Unsupported action: {action}",
        }

    return {"ok": True, "state": _state_to_dict(state)}


@app.post("/frame")
async def frame(
    image: UploadFile = File(...),
    prompt: str | None = Form(default=None),
) -> dict[str, Any]:
    raw = await image.read()
    frame_bgr = decode_upload_image(raw)
    state = tracker.track_on_frame(frame_bgr, prompt=prompt)
    return {"ok": True, "state": _state_to_dict(state)}


@app.post("/click")
async def click(
    image: UploadFile = File(...),
    x: float = Form(...),
    y: float = Form(...),
    label: str = Form(default="selection"),
) -> dict[str, Any]:
    raw = await image.read()
    frame_bgr = decode_upload_image(raw)
    points = [SegmentPoint(x=float(x), y=float(y), label=1)]
    try:
        mask, score, _device = sam_service.segment(frame_bgr, points)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"MobileSAM click failed: {exc}") from exc

    bbox = _mask_to_bbox(mask)
    if bbox is None:
        raise HTTPException(status_code=404, detail="No segment found at click point")
    state = tracker.start_from_bbox(bbox, label=label.strip() or "selection")
    state.score = float(score)
    return {"ok": True, "bbox": bbox, "state": _state_to_dict(state)}


def main() -> int:
    parser = argparse.ArgumentParser(description="Frame tracker FastAPI server")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7861)
    parser.add_argument("--sam-checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    args = parser.parse_args()

    sam_service._checkpoint = args.sam_checkpoint

    import uvicorn

    uvicorn.run(app, host=args.host, port=args.port)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
