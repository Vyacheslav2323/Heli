# Helicopter

**A modular autonomy stack for a software-first helicopter.**

Voice or text becomes a typed mission. A deterministic safety gate decides whether it may fly. A thin adapter talks to PX4. Perception and 3D reconstruction run on recorded video, so the system can be developed without flying.

This is not a finished aircraft. It is the software architecture and the parts that run today.

![Thin Client Composer — reconstructed cubicle in Babylon.js, health strip, chat dock](docs/slides/02-thin-client.png)

## Pipeline

I built a pipeline that uses open-source models and flight software as interchangeable parts:

- **[RealtimeSTT](audio/RealtimeSTT/)** (faster-whisper) for local voice → text
- **[MobileSAM](vision/third_party/MobileSAM/)** for click-to-mask, **Grounding DINO** for open-vocab boxes
- **[Depth Anything 3](vision/depth-anything-3/)** and **[VGGT](vision/vggt/)** for offline video → `scene.glb`
- **PX4** (SITL today) behind a thin flight adapter
- **Babylon.js** for the reconstructed scene in the ops UI

Around those, the repo owns the composition: mission schemas, planner / validator / executor, perception servers, and the Thin Client that puts STT, masks, tracks, and the GLB in one viewport. Whisper, the detector, or PX4 can be swapped without rewriting the rest.

## Stabilization

[heli-gym](controls/heli-gym/) is a 6-DOF helicopter sim (Heffley–Mnich). With every channel at zero — collective, lon/lat cyclic, pedal — the airframe is open-loop and falls over. I closed that loop with cascaded PID on attitude, then position (`controls/heli-gym/pid/`), so the same model holds hover.

| Open loop — all controls at 0 | After PID |
|---|---|
| ![Open-loop heli-gym: controls at 0, helicopter falls](docs/videos/heligym-untuned.gif) | ![Same model after cascaded PID hover](docs/videos/heligym-pid-hover.gif) |

## Architecture

The LLM never flies the aircraft. Every plan — script, human, or future model — hits the same validator.

```mermaid
flowchart LR
  STT[Voice STT] --> PLAIN[plain text]
  TXT[Typed text] --> PLAIN
  PLAIN --> LLM[Reasoning / LLM] --> SPEC[MissionSpec] --> PLAN[Planner] --> GATE[Validator] --> EXEC[Executor] --> ADAPT[FlightAdapter] --> PX4[PX4 / mock]
  GATE --- ERR[Structured issues]
  FR[Frames] --> DET[Detect / SAM / track]
  FR --> REC[DA3 / VGGT to GLB]
  PX4 -.-> FR
```

| Layer | Job | Must not do |
|---|---|---|
| Human interface | STT or typed text → plain string | Emit setpoints or MAVLink |
| Reasoning | Text → schema-validated `MissionSpec` | Call the flight adapter |
| Mission | Plan, validate, execute in order | Skip the safety gate |
| Flight | Abstract commands → PX4 / other FC | Parse natural language |
| Perception / mapping | Frames → tracks, masks, GLB | Arm or move the vehicle |

- Illegal altitude or geofence is a structured reject, not a prompt hope.
- Abort is explicit RTL or land.
- A failed command stops the mission; it does not auto-land.
- Reasoning is not built yet. The contracts already are.

## Live demo

`python client/launch.py` opens the Thin Client on [http://127.0.0.1:5174](http://127.0.0.1:5174). Health strip: STT · GLB · SAM · TRACK.

**Voice.** Say “keyboard.” Faster-whisper writes the transcript into the chat dock; the tracker arms on that text.

![Voice STT: “keyboard” arms tracking on the reconstructed desk](docs/slides/07-stt.png)

**Open-vocab detect.** Type `monitor`. Grounding DINO returns a scored box on the display. Same path works for `notebook`.

![Typed “monitor” / “notebook” — open-vocab boxes on the GLB scene](docs/slides/08-image-recognition.png)

**Track.** After voice or text acquire, the yellow box stays locked as the frame pump updates.

![Voice-acquired keyboard track box on the reconstructed scene](docs/slides/09-tracking.png)

- Full-bleed Babylon.js orbit of `scene.glb`
- Click → MobileSAM mask, reprojected as the camera moves
- Chat dock: mic WebSocket + typed commands

Mission confirm and SITL fly-out are not in this UI yet. That is the next wire-up, not a rewrite.

## Perception and mapping

See first. Fly later. Source video is a real desk / cubicle pass — the same workspace the reconstruction was built from.

| Input frames | 3D result in the viewer |
|---|---|
| ![Desk pass with RC transmitter](docs/slides/04-source-desk.jpg) | ![GLB reconstruction with orbit gizmo](docs/slides/01-glb-viewer.png) |
| ![Cubicle partition and mug](docs/slides/05-source-cubicle.jpg) | ![Same scene, orbited](docs/slides/03-glb-orbit.png) |

```mermaid
flowchart LR
  VID[Phone / cam video] --> FR[Sampled frames]
  FR --> SAM[MobileSAM click masks]
  FR --> DINO[Grounding DINO open-vocab boxes]
  DINO --> FLOW[Optical flow]
  FR --> DA3[Depth Anything 3]
  FR --> VGGT[VGGT experiments]
  DA3 --> GLB[scene.glb]
  VGGT --> GLB
  SAM --> UI[Thin Client]
  FLOW --> UI
  GLB --> UI
```

- One shared SAM GPU process — tracker calls `:7860` over HTTP
- ~6 Hz frame pump with epoch guards so stale boxes are dropped
- DA3 batches large clips and Sim(3)-aligns overlap so the scene stays coherent

## Mission contracts

Missions are data, not prompts. Example: [`examples/missions/north_legs.json`](examples/missions/north_legs.json) — takeoff 3 m → north 200 m → hold 10 s → yaw 90° → north 200 m.

```mermaid
sequenceDiagram
  participant Spec as MissionSpec
  participant Planner as plan_mission
  participant Gate as validate_plan
  participant Exec as MissionExecutor
  participant FC as FlightAdapter
  Spec->>Planner: goals in order
  Planner->>Gate: abstract commands
  Gate-->>Gate: max/min alt, geofence, battery
  alt rejected
    Gate-->>Spec: issues[]
  else approved
    Gate->>Exec: command list
    Exec->>FC: takeoff / goto_ned / hold / yaw / land
    FC-->>Exec: telemetry + acks
  end
```

- Schema v1 with geofence, altitude, battery
- Planner keeps goal order; no injected arm/land
- Validator rejects bad alt / fence in unit tests
- Executor is linear; PX4 adapter + mock exist
- ArduPilot adapter is still an empty slot

Language → spec → human confirm → SITL is the closed loop that is not shipped yet. The contracts are already there.

## Status

```mermaid
flowchart TB
  subgraph now [Built]
    UI[Thin Client: STT · SAM · track · GLB]
    CORE[Planner · validator · executor]
    PX[PX4 adapter + SITL tools]
    MAP[Offline DA3 / VGGT → GLB]
  end
  subgraph next [Next]
    LLM[DeepSeek: text → MissionSpec]
    CONF[Human confirm in the client]
    SITLUI[SITL fly-out from the UI]
  end
  subgraph later [Then]
    TWIN[Queryable digital twin]
    SEM[SAM labels fused into 3D]
    GS[3D Gaussian Splatting]
    AP[ArduPilot adapter]
    HW[Hybrid FC + companion computer]
  end
  UI --> LLM
  CORE --> LLM
  LLM --> CONF
  CONF --> SITLUI
  SITLUI --> TWIN
  MAP --> SEM
  SEM --> TWIN
  TWIN --> LLM
  PX --> AP
  AP --> HW
```

The next product step is not a new model. It is wiring intent → confirm → SITL through contracts that already exist.

## Run the demo

Needs a local Python env with project + MobileSAM deps, Node.js, and `vision/third_party/MobileSAM/weights/mobile_sam.pt`.

```bash
cd vision/tools/glb-viewer && npm install
cd ../../client && npm install
python client/launch.py
```

The launcher starts STT (`:8010`), MobileSAM (`:7860`), the tracker (`:7861`), the GLB viewer (`:5173`), and the Thin Client (`:5174`). Details: [`client/README.md`](client/README.md).

Mission-core tests (no GPU, no vehicle):

```bash
pytest tests/controls/mission
```

## Docs

| Doc | What it is |
|---|---|
| [`NORTH_STAR.md`](NORTH_STAR.md) | Product objective, phases, non-negotiable safety rule |
| [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) | Module contracts and target layout |
| [`PPT.md`](PPT.md) | Speaker notes for the portfolio walkthrough |
| [`client/README.md`](client/README.md) | How to run and probe the Thin Client |
| [`docs/decisions/001-flight-controller.md`](docs/decisions/001-flight-controller.md) | Flight-stack decision still open behind the adapter |

## Third-party licenses

Vendored trees keep their own licenses. Notable ones:

| Tree | License |
|---|---|
| `audio/RealtimeSTT/` | MIT |
| `vision/third_party/MobileSAM/` | Apache-2.0 |
| `vision/depth-anything-3/` | Apache-2.0 |
| `vision/vggt/` | Meta research license |
| `vision/third_party/sam3/` | Meta SAM license |
| `controls/heli-gym/` | MIT |
