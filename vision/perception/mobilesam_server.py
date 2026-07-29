from __future__ import annotations

import argparse
import base64
import json
import os
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field


DEFAULT_CHECKPOINT = Path("vision/third_party/MobileSAM/weights/mobile_sam.pt")


class SegmentPoint(BaseModel):
    x: float
    y: float
    label: int = Field(default=1, ge=0, le=1)


class SegmentJsonRequest(BaseModel):
    image_base64: str
    points: list[SegmentPoint]


@dataclass
class SamState:
    predictor: Any
    device: str


class MobileSamService:
    def __init__(self, checkpoint: Path) -> None:
        self._checkpoint = checkpoint
        self._state: SamState | None = None

    def _load(self) -> SamState:
        if self._state is not None:
            return self._state
        if not self._checkpoint.exists():
            raise RuntimeError(f"Missing MobileSAM checkpoint: {self._checkpoint}")

        from mobile_sam import SamPredictor, sam_model_registry

        device = "cuda" if torch.cuda.is_available() else "cpu"
        sam = sam_model_registry["vit_t"](checkpoint=str(self._checkpoint))
        sam.to(device=device)
        sam.eval()
        self._state = SamState(predictor=SamPredictor(sam), device=device)
        return self._state

    def segment(self, image_bgr: np.ndarray, points: list[SegmentPoint]) -> tuple[np.ndarray, float, str]:
        state = self._load()
        rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        state.predictor.set_image(rgb)

        if not points:
            raise ValueError("At least one point is required")

        coords = np.array([[p.x, p.y] for p in points], dtype=np.float32)
        labels = np.array([p.label for p in points], dtype=np.int32)
        masks, scores, _ = state.predictor.predict(
            point_coords=coords,
            point_labels=labels,
            multimask_output=True,
        )

        best_idx = int(np.argmax(scores))
        mask = masks[best_idx]
        if mask.dtype != np.bool_:
            mask = mask > 0.5
        return mask.astype(bool), float(scores[best_idx]), state.device


service = MobileSamService(Path(os.getenv("MOBILESAM_CHECKPOINT", str(DEFAULT_CHECKPOINT))))
app = FastAPI(title="MobileSAM API", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5174", "http://127.0.0.1:5174"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health() -> dict[str, Any]:
    checkpoint_exists = service._checkpoint.exists()
    if not checkpoint_exists:
        return {"ok": False, "ready": False, "device": "cpu", "checkpoint": str(service._checkpoint)}

    device = service._state.device if service._state else ("cuda" if torch.cuda.is_available() else "cpu")
    return {"ok": True, "ready": True, "device": device, "checkpoint": str(service._checkpoint)}


@app.post("/segment")
async def segment(
    image: UploadFile | None = File(default=None),
    points: str | None = Form(default=None),
    payload: SegmentJsonRequest | None = None,
) -> dict[str, Any]:
    image_bgr: np.ndarray
    req_points: list[SegmentPoint]

    if payload is not None:
        try:
            raw = base64.b64decode(payload.image_base64, validate=True)
        except Exception as exc:  # noqa: BLE001
            raise HTTPException(status_code=400, detail=f"Invalid image_base64: {exc}") from exc
        req_points = payload.points
    else:
        if image is None or points is None:
            raise HTTPException(status_code=400, detail="Provide either JSON payload or multipart image+points")
        raw = await image.read()
        try:
            parsed_points = json.loads(points)
            req_points = [SegmentPoint.model_validate(item) for item in parsed_points]
        except Exception as exc:  # noqa: BLE001
            raise HTTPException(status_code=400, detail=f"Invalid points JSON: {exc}") from exc

    arr = np.frombuffer(raw, dtype=np.uint8)
    image_bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise HTTPException(status_code=400, detail="Could not decode image")

    try:
        mask, score, device = service.segment(image_bgr, req_points)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"MobileSAM failed: {exc}") from exc

    rgba = np.zeros((mask.shape[0], mask.shape[1], 4), dtype=np.uint8)
    rgba[mask] = (0, 200, 255, 150)

    ok, encoded = cv2.imencode(".png", rgba)
    if not ok:
        raise HTTPException(status_code=500, detail="Failed to encode mask")

    return {
        "mask": base64.b64encode(encoded.tobytes()).decode("ascii"),
        "score": score,
        "width": int(mask.shape[1]),
        "height": int(mask.shape[0]),
        "device": device,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="MobileSAM FastAPI server")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--sam-checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    args = parser.parse_args()

    service._checkpoint = args.sam_checkpoint

    import uvicorn

    uvicorn.run(app, host=args.host, port=args.port)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
