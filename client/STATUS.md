# Thin Client Composer — Status

Companion to [`PRODUCT.md`](../PRODUCT.md) and [`DESIGN.md`](../DESIGN.md).
Runbook: [`README.md`](./README.md).

**Status:** Local integration UI is runnable and actively used for validating
STT, GLB, MobileSAM, and object tracking. Not yet wired to mission/flight.

---

## Purpose

Compose four independent module servers into one full-viewport operations UI so
developers can exercise perception and speech without flying or SITL:

| Module | Port | Role |
|--------|------|------|
| RealtimeSTT FastAPI | `:8010` | Voice → text (WebSocket) |
| GLB viewer (Vite) | `:5173` | Serves `scene.glb` + health |
| MobileSAM FastAPI | `:7860` | Click-point segmentation |
| Track FastAPI | `:7861` | Text / click / frame tracking |
| **This client** | `:5174` | UI composer + Vite proxies |

Launch: `python client/launch.py` from repo root.

---

## Layout

```
client/
├── launch.py              # Spawns all five processes; waits on health URLs
├── index.html             # Scene + overlay canvases, health strip, chat dock
├── package.json           # Vite + Babylon.js (@babylonjs/core, loaders)
├── vite.config.ts         # Proxies /stt /sam /track /viewer
└── src/
    ├── main.ts            # Wires modules; styles; alerts; health strip
    ├── scene/glbScene.ts  # Babylon GLB load, orbit gizmo, pan/zoom, capture
    ├── overlay/samOverlay.ts
    ├── track/trackSession.ts
    ├── chat/chatDock.ts
    └── health/monitor.ts
```

Stack: TypeScript, Vite 6, Babylon.js 8. No React — plain DOM + canvas modules.

---

## What is built

### 1. Scene (`GlbScene`)

- Loads `/viewer/scene.glb` via Babylon `SceneLoader`
- Strips Depth Anything 3 camera-frustum meshes (tiny vertex pyramids)
- Fits camera to scene bounds; hemispheric light
- Orbit rotation gizmo (XYZ rings) with keyboard `QWASZX`
- Lateral pan (drag off-gizmo), wheel zoom
- `captureBlob()` for SAM / track frame uploads
- World pick + project helpers so overlays reproject when the camera moves
- Gizmo toggle in UI (`#gizmoCheckbox`)

### 2. Overlay (`SamOverlay`)

- Left-click (no drag, not on gizmo) → capture scene → `POST /sam/segment`
- Draws cyan mask; bakes mask pixels into world points and reprojects on camera move
- Draws yellow tracking bboxes + `#index score%` labels from `TrackSession`
- Clears SAM mask when live track boxes arrive; clears all on track switch/stop

### 3. Tracking (`TrackSession`)

- Chat/voice text → `POST /track/command` (non-control phrases become `track <text>`)
- Click after SAM → `POST /track/click` with scene PNG + point
- Frame pump ~6 Hz (`160 ms`) → `POST /track/frame` while tracking is active/armed
- Epoch guarding so stale responses from a previous target are ignored
- Commands understood (server-side via `parse_track_command`):  
  `track <object>`, `track <index>`, `track all`, `stop` / `stop tracking`, `status`, …

### 4. Chat dock (`ChatDock`)

- Text send + Enter
- Mic → WebSocket `/stt/ws/transcribe` with PCM `s16le` packets
- Live partial transcripts + final voice lines pushed into the message list
- Final voice / typed text both call `onUserText` → tracking
- Exposes `SttLiveStatus` (mic/ws/packets/transcripts) for health

### 5. Health monitor

Polls every 4s:

| Pill | Probe |
|------|--------|
| STT | HTTP `/stt/health` + WS ping/hello + mic device + live audio pipeline |
| GLB | `/viewer/health` |
| SAM | `/sam/health` (`ready`) |
| Track | `/track/health` (`ready`) |

Statuses: `ok` | `degraded` | `down`. Hover shows detail string. Transition to `down`
raises a non-blocking alert.

### 6. Launcher (`launch.py`)

Starts STT, MobileSAM, track server, glb-viewer, and client; prints readiness;
Ctrl+C tears down process group (Windows-aware npm/`cmd.exe` spawn).

---

## Interaction model

```
┌─────────────────────────────────────────────────────────┐
│  Health strip (STT · GLB · SAM · TRACK)     [Gizmo ☑]   │
│                                                         │
│              Full-bleed Babylon GLB scene               │
│              + 2D overlay (mask / track boxes)          │
│                                                         │
│  Alerts (transient)                                     │
├─────────────────────────────────────────────────────────┤
│  Chat messages (text / voice / system)                  │
│  [Start Mic]  [Type a message…………]  [Send]              │
└─────────────────────────────────────────────────────────┘
```

| Input | Effect |
|-------|--------|
| Click scene (no drag) | MobileSAM mask + start click tracking |
| Drag / gizmo / keys | Orbit, pan, zoom (no segment) |
| Type / say object name | `track <name>` |
| `track 2`, `track all`, `stop` | Explicit track control |
| Start Mic | Stream audio; finals become track commands |

---

## Proxy map (`vite.config.ts`)

| Browser path | Upstream |
|--------------|----------|
| `/stt/*` (+ WS) | `127.0.0.1:8010` |
| `/sam/*` | `127.0.0.1:7860` |
| `/track/*` | `127.0.0.1:7861` |
| `/viewer/*` | `127.0.0.1:5173` |

---

## Success criteria (from PRODUCT.md)

| Criterion | State |
|-----------|--------|
| Health pills for STT / GLB / SAM / tracker | Done |
| Voice transcripts in chat when STT active | Done |
| Segmentation overlays on GLB | Done (world-reprojected) |
| Tracking boxes update after text or click | Done (frame pump) |

---

## Not in this client

- Mission planner / validator / executor UI
- Flight / PX4 / SITL control
- LLM intent / DeepSeek reasoning
- Mapping / digital-twin query UI
- Auth, remote deploy, multi-user

These stay behind module boundaries per [`docs/ARCHITECTURE.md`](../docs/ARCHITECTURE.md).
The client only composes local vision + speech servers for developer validation.

---

## Dependencies (runtime)

- Node.js + npm (`client/` and `vision/tools/glb-viewer`)
- Python env with MobileSAM + tracker deps
- Weights: `vision/third_party/MobileSAM/weights/mobile_sam.pt`
- GLB asset served by glb-viewer at `/viewer/scene.glb`
