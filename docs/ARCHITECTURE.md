# Architecture

Companion to [`NORTH_STAR.md`](../NORTH_STAR.md). Defines the modules, their
contracts, and the target repository layout.

## Module Map

Each module is independently testable and communicates only through the
declared inputs/outputs. Internals are never imported across module boundaries.

### 1. `audio/` — Human Interface Layer (speech)

| Module | Input | Output | Notes |
|---|---|---|---|
| `audio/stt` | audio stream/file | plain text transcript | Default: faster-whisper / RealtimeSTT. Swappable with any STT API. |
| `audio/tss` | plain text | audio | Local TTS for agent replies. |

Contract: downstream only ever sees `{ "text": str, "source": "voice"|"text", "timestamp": ... }`.
Frontend clients are deferred; these modules are library-only for now.

### 2. `reasoning/` — LLM Intent Interpretation

| Module | Input | Output |
|---|---|---|
| `reasoning/intent` | plain text + current world state summary | `MissionSpec` (JSON, schema-validated) |

- Backed by the **DeepSeek API** (OpenAI-compatible client; model configurable).
- Output must validate against `controls/schemas/mission_spec.schema.json`; invalid output
  is rejected and retried, never passed downstream.
- The reasoning layer has **read-only** access to world state (twin, telemetry
  summaries). It cannot invoke any flight function.

### 3. `mission/` — Deterministic Mission System

| Module | Input | Output |
|---|---|---|
| `mission/planner` | `MissionSpec` + twin/map | ordered list of abstract flight commands |
| `mission/validator` | plan + constraints (geofence, battery, altitude, weather) | approved plan or structured rejection |
| `mission/executor` | approved plan | abstract commands streamed to flight adapter; progress/abort events |

- Pure Python, fully unit-testable without any simulator.
- The validator is the **only** gate to execution — every plan passes through it
  regardless of origin (LLM, script, human).

### 4. `perception/` — Vision

| Module | Input | Output |
|---|---|---|
| `perception/detect` | video frames | detections/segments/tracks (e.g. SAM 3.1) |
| `perception/depth` | frames (+ optional poses) | depth maps, camera poses (e.g. Depth Anything 3) |

### 5. `mapping/` — Reconstruction & Digital Twin

| Module | Input | Output |
|---|---|---|
| `mapping/reconstruct` | frames + poses (+ depth) | 3D reconstruction (gaussians / point cloud / mesh) |
| `mapping/twin` | reconstruction + detections | semantically labeled, queryable scene ("digital twin") |

All perception/mapping modules must run **offline on recorded data** — this is
what makes them developable without flying.

### 6. `flight/` — Flight Control Adapter

| Module | Input | Output |
|---|---|---|
| `flight/adapter` (interface) | abstract commands: `arm`, `takeoff(alt)`, `goto(lat,lon,alt)`, `orbit`, `land`, `abort` | telemetry events, command acks |
| `flight/px4` | ↑ via MAVLink (MAVSDK/pymavlink) | — |
| `flight/ardupilot` | ↑ via MAVLink | — |
| `flight/arduino` | ↑ via custom serial protocol | only if a custom FC is built |

**Only implementations of `flight/adapter` may talk to a flight controller.**
Swapping flight stacks = implementing one adapter. Decision tracked in
`docs/decisions/001-flight-controller.md`.

### 7. `platform/` — Simulation, Data, Tooling

- `controls/tools/` — PX4 SITL orchestration, WSL/QGC bridges.
- `controls/heli-gym/` — controller/PID sandbox.
- `data/` — flight logs, video recordings, datasets; every flight/sim run is replayable.
- `vision/tools/` — vision developer utilities (e.g. glb-viewer).

## Target Repository Layout

```
helicopter/
├── NORTH_STAR.md
├── docs/
│   ├── ARCHITECTURE.md
│   ├── decisions/           # ADRs (001-flight-controller.md, ...)
│   └── research/            # research notes (perception-sota.md, ...)
├── controls/                # helicopter controls
│   ├── flight/              # adapter.py, px4/
│   ├── mission/             # planner/, validator/, executor/
│   ├── schemas/             # MissionSpec + flight command contracts
│   ├── reasoning/           # intent stub (DeepSeek later)
│   ├── heli-gym/            # sim / PID sandbox
│   └── tools/               # PX4 SITL, MAVLink/QGC bridges
├── vision/                  # computer vision
│   ├── perception/          # detect / track library code
│   ├── depth-anything-3/    # DA3 vendor
│   ├── third_party/         # MobileSAM, sam3
│   ├── scripts/             # lab demos / CLIs
│   └── tools/               # glb-viewer, etc.
├── audio/                   # speech processing
│   ├── stt.py, tss.py, ...  # STT / TTS / mic
│   └── RealtimeSTT/         # vendor
├── databases/
│   └── graph/               # scene-graph / twin store (scaffold)
├── data/                    # logs, recordings (gitignored where large)
└── tests/                   # mirrors functional roots (controls/, vision/, ...)
```
Frontend clients are out of tree for now; modules must stay independently
testable. A future client will compose these packages through narrow schemas.

## Dataflow: Natural-Language Mission (target end state)

```
voice ──► STT ──► text ─┐
text input ─────────────┴─► reasoning/intent (DeepSeek)
                                │  MissionSpec (JSON)
                                ▼
                        mission/planner ──► mission/validator ──► human confirm gate
                                                                        │ approved plan
                                                                        ▼
                                                                mission/executor
                                                                        │ abstract commands
                                                                        ▼
                                                                flight/adapter (PX4 | ArduPilot | Arduino)
                                                                        │ MAVLink / serial
                                                                        ▼
                                                        simulator or vehicle ──► telemetry/video ──► data/
                                                                                        │
                                        perception + mapping ◄──────────────────────────┘
                                                │
                                                ▼
                                        digital twin ──► (read-only) reasoning & planner context
```

