# Research — State of the Art: Video Perception, Reconstruction & Digital Twins

**Date:** 2026-07-23 · Supports roadmap Phases 3–5. Re-survey before each phase kicks off.

## 1. Detection / Segmentation / Tracking from Video

**Leading candidate: Meta SAM 3.1** (released 2026-03, [facebookresearch/sam3](https://github.com/facebookresearch/sam3), checkpoints `facebook/sam3.1` on Hugging Face)

- Unified foundation model: detect, segment, and **track** objects in images and video from **open-vocabulary text prompts** ("track the red car"), points, boxes, or exemplars — no per-class training.
- SAM 3.1 adds *Object Multiplex*: up to 16 objects tracked in a single forward pass, ~32 FPS medium-density scenes on an H100, ~7x faster than SAM 3 at high object counts.
- 848M params — fine on a desktop GPU for offline pipelines; too heavy for small onboard computers today.

**Lightweight / edge alternatives:**

- **YOLO26 (Ultralytics)** — real-time detection/segmentation/pose optimized for edge deployment; closed vocabulary but fast enough for onboard use.
- **DINO-X / Grounding-DINO family** — open-world detection when text-prompted detection (without masks) is enough.

**Recommendation:** SAM 3.1 for the offline perception pipeline (recorded sim/flight video); evaluate YOLO26 later when onboard real-time perception is needed.

## 2. Depth & Geometry from Monocular/Multi-view Video

**Leading candidate: Depth Anything 3 (DA3)** (ByteDance, [ByteDance-Seed/Depth-Anything-3](https://github.com/ByteDance-Seed/Depth-Anything-3), Apache 2.0)

One model family covers most of our geometry needs:

- Monocular relative depth (`DA3Mono-Large`, 0.35B params)
- **Metric** depth for real-world scale (`DA3Metric-Large`)
- Multi-view consistent depth + **camera pose estimation** (main series)
- Direct **3D Gaussian prediction** for novel view synthesis

This is significant for us: DA3 can bootstrap reconstruction from plain video
**without a depth camera**, and its pose estimation reduces dependence on
classical SfM (COLMAP) for offline pipelines.

## 3. 3D Reconstruction & SLAM (Environment Mapping)

The field has converged on **3D Gaussian Splatting (3DGS)** as the reconstruction
representation — photorealistic, fast to render, and increasingly real-time.

| Approach | When to use | Notes |
|---|---|---|
| **Offline: COLMAP/DA3 poses + 3DGS** (e.g. Nerfstudio `splatfacto`, gsplat) | Phase 4 start — reconstruct from recorded sim/flight video | Most mature path; quality-first, not real-time |
| **AeroGS** (CVPR 2026) | Pose-free reconstruction of dynamic UAV footage | SOTA for aerial video without known poses; handles scale imbalance between distant terrain and near objects |
| **3DGS-SLAM onboard: VIGS-Fusion** (ICAR 2025, [code](https://github.com/AbdoullahNdoye/VIGS-Fusion)) | Real-time onboard mapping (later phases) | First real-time 3DGS SLAM fully onboard a 6-inch UAV; needs RGB-D + IMU; ~30x faster tracking than prior 3DGS SLAM |
| **LD3DGS-SLAM** | Long-distance outdoor UAV mapping | Monocular 3DGS SLAM with GNSS-aided localization |
| Classical: ORB-SLAM3 / VINS-Fusion | Robust pose tracking baseline | Sparse maps only; still useful as pose backbone feeding 3DGS |

**Recommendation:** Phase 4 starts **offline**: recorded video → DA3 (poses+depth) or COLMAP → 3DGS reconstruction. Real-time onboard SLAM (VIGS-Fusion-style) is a later optimization, and influences companion-computer choice (Jetson-class GPU).

## 4. Digital Twin Generation (reconstruction → operational twin)

Emerging recipe, aligned with our pipeline:

1. **Geometry** from 3DGS reconstruction (optionally meshed via TSDF/Poisson for physics/collision).
2. **Semantics** by fusing SAM 3.1 open-vocabulary masks into the 3D representation (language-embedded / semantic gaussians — active research area, several open implementations).
3. **Queryability**: scene graph or vector store over labeled objects so the mission planner and the LLM can ask "where is the landing pad?" against the twin.
4. Twin formats worth watching: **OpenUSD** for scene interchange (NVIDIA Omniverse ecosystem), glTF for lightweight export.

## 5. Practical Pipeline Proposal (Phases 3–5)

```
recorded video (sim or flight)
   ├─► SAM 3.1  ──────────────► detections / masks / tracks
   ├─► Depth Anything 3 ─────► depth maps + camera poses
   └─► 3DGS training (gsplat / Nerfstudio) ─► reconstruction
                                    │
              masks + reconstruction fusion
                                    ▼
                  semantic digital twin (queryable)
```

All stages run offline on a desktop GPU against recorded data first — matching
the north-star principle "everything is replayable". Onboard/real-time variants
are a hardware-driven optimization deferred to later phases.

## 6. Open Questions to Resolve Per Phase

- Phase 3: does simulated camera footage (Gazebo/heli-gym) have enough realism for SAM/DA3, or do we need real video early?
- Phase 4: monocular-only vs adding a depth camera (RealSense/OAK-D) — DA3 metric depth may make monocular sufficient.
- Phase 5: twin storage/query format (scene graph vs 3D-aware vector DB) — prototype both against planner needs.
