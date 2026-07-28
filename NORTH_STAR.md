# North Star — Autonomous Helicopter Platform

> This document is the single source of truth for what this project is trying to become.
> Every design decision, dependency, and line of code should be justifiable against it.

## Project Objective

Develop a **software-centered autonomous helicopter platform** that can:

1. Execute reliable flight commands,
2. Perceive and reconstruct its environment from video,
3. Generate operational **digital twins**, and
4. Later accept **natural-language mission instructions** through voice or text.

Near-term work prioritizes **simulation, computer vision, mapping, mission software,
and data pipelines**. Physical electronics are introduced only as required for RC validation.

## Capability Roadmap

```
Flight stability → Basic autonomous commands → Visual perception → Environment mapping
    → Digital twin generation → Mission execution → Natural-language control
```

| Phase | Capability | Definition of done | Status |
|---|---|---|---|
| 0 | Simulation foundation | PX4 SITL (or equivalent) runs reliably; telemetry visible in QGC; repeatable launch scripts | In progress (`controls/tools/`, `controls/heli-gym/`) |
| 1 | Flight stability | Stable hover + attitude hold in sim; PID/controller tuning validated (`heli-gym/pid`) | In progress |
| 2 | Basic autonomous commands | Scripted takeoff, waypoint, loiter, land executed via MAVLink offboard API, with automated pass/fail checks | — |
| 3 | Visual perception | Detect, segment, and track objects from simulated/recorded video streams | — |
| 4 | Environment mapping | Camera video + pose → consistent 3D reconstruction (point cloud / mesh / gaussians) of the flown area | — |
| 5 | Digital twin generation | Reconstruction is post-processed into a queryable, semantically labeled scene usable for planning and replay | — |
| 6 | Mission execution | Declarative mission spec (JSON/YAML) → validated plan → executed flight → logged evidence | — |
| 7 | Natural-language control | Voice/text → intent → mission spec, with human confirmation gate before execution | — |

Each phase must be demonstrable **in simulation first**. Hardware (RC helicopter,
companion computer, camera) enters only when a phase needs physical validation.

## System Design Principle (non-negotiable)

**The LLM interprets human intent. Deterministic flight-control and mission systems
execute and validate commands. The LLM never directly controls actuators.**

Concretely:

- LLM output is always a **structured mission specification** (typed, schema-validated),
  never raw setpoints, RC channels, or MAVLink messages.
- A deterministic **mission validator** checks every spec against safety constraints
  (geofence, battery, altitude limits, no-fly conditions) before anything flies.
- The flight-control layer (PX4/ArduPilot/custom) is the only component that
  commands actuators, and it works identically whether the mission came from an
  LLM, a script, or a human.

## Architecture: Extreme Modularity

The system is a set of independently developed, independently testable modules
communicating through narrow, versioned interfaces. No module imports another
module's internals; boundary contracts (message schemas) are the API.

```
┌────────────────────────────────────────────────────────────────┐
│  HUMAN INTERFACE LAYER                                         │
│  ┌───────────────┐   ┌───────────────┐                         │
│  │ voice (STT)   │   │ text input    │     — interchangeable   │
│  └──────┬────────┘   └──────┬────────┘                         │
│         └───── plain text ──┘                                  │
├────────────────────────────────────────────────────────────────┤
│  REASONING LAYER (LLM — DeepSeek API)                          │
│  text → structured MissionSpec (JSON, schema-validated)        │
├────────────────────────────────────────────────────────────────┤
│  MISSION LAYER (deterministic)                                 │
│  planner · validator/safety gate · executor · state machine    │
├────────────────────────────────────────────────────────────────┤
│  PERCEPTION & MAPPING LAYER                                    │
│  detection/segmentation · depth · SLAM/reconstruction ·        │
│  digital twin builder                                          │
├────────────────────────────────────────────────────────────────┤
│  FLIGHT CONTROL LAYER                                          │
│  MAVLink/serial adapter → PX4 | ArduPilot | Arduino/custom     │
│  (only this layer touches actuators)                           │
├────────────────────────────────────────────────────────────────┤
│  PLATFORM LAYER                                                │
│  simulation (SITL, heli-gym) · data pipeline (logs, video,     │
│  datasets) · tooling (tools/)                                  │
└────────────────────────────────────────────────────────────────┘
```

Key separations (explicitly requested and enforced):

- **Voice/Text ⟂ LLM reasoning** — STT produces plain text; the reasoning module
  consumes plain text. Either side can be swapped (Whisper → cloud STT,
  DeepSeek → any other LLM) without touching the other.
- **Low-level flight control ⟂ mission planning** — the mission layer emits
  abstract commands (`goto(lat, lon, alt)`, `orbit`, `land`); a thin adapter
  translates them to the concrete flight stack (PX4 MAVLink, ArduPilot, or
  Arduino serial protocol). Swapping flight controllers means swapping one adapter.
- **Perception ⟂ mapping ⟂ twin generation** — each consumes/produces defined
  artifacts (frames → detections; frames+poses → reconstruction; reconstruction →
  labeled twin) and can be run offline on recorded data.

See `docs/ARCHITECTURE.md` for module contracts and the proposed repo layout.

## Stack Decisions

| Concern | Decision | Rationale |
|---|---|---|
| Flight controller | **Undecided — kept behind an adapter interface.** Candidates: PX4, ArduPilot, Betaflight, Arduino/custom. See `docs/decisions/001-flight-controller.md` | Hardware choice deferred; sim work continues on PX4 SITL meanwhile |
| Simulation | PX4 SITL + QGC (current `controls/tools/` setup), `controls/heli-gym` for controller experiments | Already working |
| LLM reasoning | **DeepSeek API** (low cost, OpenAI-compatible) | Explicit project decision; structured-output mission generation |
| Speech-to-text | **Local Whisper (faster-whisper) by default**; note: DeepSeek does not offer an STT API, so STT is a separate module — exactly what the modular design expects | Free, offline-capable, easily swapped |
| Perception & mapping | SOTA survey completed; leading candidates: SAM 3.1 (detect/segment/track), Depth Anything 3 (depth + pose), 3D Gaussian Splatting SLAM / COLMAP+3DGS (reconstruction). See `docs/research/perception-sota.md` | Research-driven; re-evaluate per phase |
| Primary language | Python (existing codebase) | Ecosystem for CV/ML/MAVLink |
| Messaging between modules | Start with plain function/process boundaries + JSON schemas; adopt ROS 2 or ZeroMQ only if/when multi-process onboard deployment demands it | Avoid infrastructure before it pays for itself |

## Guiding Principles

1. **Simulation first.** Nothing goes to hardware that hasn't passed in sim.
2. **Deterministic core, probabilistic edges.** ML/LLM components live at the
   perception and intent boundaries; everything between intent and actuator is
   deterministic and testable.
3. **Everything is replayable.** Flights, video, and telemetry are logged so
   perception/mapping/twin pipelines can be developed offline on recorded data.
4. **Narrow interfaces, swappable parts.** Any single vendor/model/flight stack
   can be replaced without a rewrite.
5. **Safety gates are code, not convention.** The validator between mission spec
   and execution is mandatory and cannot be bypassed by any caller, including the LLM.
