# Thin Client Composer

Thin client that composes four module servers:

- STT FastAPI (`:8010`)
- GLB viewer (`:5173`)
- MobileSAM FastAPI (`:7860`)
- Text/Click tracker FastAPI (`:7861`)

The UI runs on `http://127.0.0.1:5174`.

## Prerequisites

- Python environment with project dependencies and MobileSAM requirements
- Node.js + npm
- `vision/third_party/MobileSAM/weights/mobile_sam.pt` present

Install frontend dependencies:

```bash
cd vision/tools/glb-viewer && npm install
cd ../../..\client && npm install
```

## Run all services

From repo root:

```bash
python client/launch.py
```

The launcher starts:

1. `audio/RealtimeSTT/example_fastapi_server/server.py`
2. `vision/perception/mobilesam_server.py`
3. `vision/perception/track_server.py`
4. `vision/tools/glb-viewer` Vite server
5. `client` Vite server

## API/Health probes used by client

- STT: `GET /stt/health` (proxied to `:8010/health`)
- GLB: `GET /viewer/health` (proxied to `:5173/health`)
- MobileSAM: `GET /sam/health`, `POST /sam/segment` (proxied to `:7860`)
- Tracker: `GET /track/health`, `POST /track/command`, `POST /track/frame`, `POST /track/click` (proxied to `:7861`)

## Interaction model

- Fullscreen GLB scene as background
- Click on viewport to request MobileSAM mask overlay
- Bottom chat dock for text and voice input (STT websocket stream)
- Typed/voice commands such as `track bear`, `track 2`, `track all`, `stop tracking`
- Or just type/say an object name (e.g. `keyboard`, `monitor`) to start tracking it
- Click-to-select handoff: after segmentation, click seed can lock a tracked box that follows camera motion

## STT health checks

STT is marked healthy only when all of these pass:

1. HTTP /stt/health reports ready
2. WebSocket /stt/ws/transcribe accepts and responds to ping/hello
3. A microphone device is available (permission may still be pending until Start Mic)
4. While mic is active: audio packets are flowing; transcripts update status when generated

Hover the STT pill for details. degraded means transport is up but mic/audio/transcript path is incomplete.
