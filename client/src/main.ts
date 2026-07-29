import { ChatDock } from "./chat/chatDock";
import { startHealthMonitor, type HealthDetails, type HealthSnapshot } from "./health/monitor";
import { SamOverlay } from "./overlay/samOverlay";
import { GlbScene } from "./scene/glbScene";
import { TrackSession, type TrackState } from "./track/trackSession";

function injectStyles(): void {
  const style = document.createElement("style");
  style.textContent = `
    :root {
      color-scheme: dark;
      --panel: rgba(8, 12, 18, 0.78);
      --panel-border: rgba(125, 167, 210, 0.28);
      --text-main: #e8f1fb;
      --text-muted: #8da7c4;
      --ok: #5ae38e;
      --degraded: #f0c35a;
      --down: #ff7070;
      --chat-bg: rgba(3, 8, 13, 0.82);
    }

    * { box-sizing: border-box; }
    html, body, #app { margin: 0; width: 100%; height: 100%; overflow: hidden; }
    body {
      font-family: "Segoe UI", "Noto Sans KR", sans-serif;
      color: var(--text-main);
      background: radial-gradient(circle at 20% 20%, #102336 0%, #04080d 70%);
    }

    #sceneCanvas, #overlayCanvas {
      position: fixed;
      inset: 0;
      width: 100vw;
      height: 100vh;
      display: block;
    }

    #overlayCanvas { z-index: 2; }

    #gizmoToggle {
      position: fixed;
      top: 12px;
      right: 12px;
      z-index: 6;
      display: flex;
      align-items: center;
      gap: 8px;
      color: var(--text-main);
      font-size: 13px;
      line-height: 1;
      background: var(--panel);
      border: 1px solid var(--panel-border);
      border-radius: 10px;
      padding: 8px 12px;
      user-select: none;
      backdrop-filter: blur(8px);
    }

    #gizmoToggle input {
      margin: 0;
      width: 14px;
      height: 14px;
      accent-color: #7da7d2;
      cursor: pointer;
    }

    #healthStrip {
      position: fixed;
      top: 14px;
      left: 50%;
      transform: translateX(-50%);
      z-index: 5;
      display: flex;
      flex-wrap: wrap;
      justify-content: center;
      gap: 8px;
      max-width: min(96vw, 880px);
      padding: 8px 12px;
      border-radius: 16px;
      border: 1px solid var(--panel-border);
      background: var(--panel);
      backdrop-filter: blur(8px);
      font-size: 12px;
    }

    .health-pill {
      padding: 4px 10px;
      border-radius: 999px;
      border: 1px solid rgba(255, 255, 255, 0.22);
      color: var(--text-main);
      max-width: 42vw;
      white-space: nowrap;
      overflow: hidden;
      text-overflow: ellipsis;
    }

    .health-pill.ok { border-color: rgba(90, 227, 142, 0.6); color: var(--ok); }
    .health-pill.degraded { border-color: rgba(240, 195, 90, 0.7); color: var(--degraded); }
    .health-pill.down { border-color: rgba(255, 112, 112, 0.6); color: var(--down); }

    #alerts {
      position: fixed;
      top: 64px;
      left: 50%;
      transform: translateX(-50%);
      z-index: 5;
      display: flex;
      flex-direction: column;
      gap: 8px;
      max-width: min(92vw, 720px);
    }

    .alert {
      background: rgba(108, 24, 24, 0.9);
      border: 1px solid rgba(255, 112, 112, 0.58);
      color: #ffe6e6;
      border-radius: 10px;
      padding: 8px 12px;
      font-size: 13px;
    }

    #chatDock {
      position: fixed;
      left: 0;
      right: 0;
      bottom: 0;
      z-index: 7;
      padding: 12px;
      background: linear-gradient(to top, rgba(2, 6, 10, 0.94), rgba(2, 6, 10, 0.7));
      border-top: 1px solid var(--panel-border);
      backdrop-filter: blur(8px);
    }

    #chatMessages {
      max-height: 22vh;
      overflow-y: auto;
      margin-bottom: 10px;
      display: flex;
      flex-direction: column;
      gap: 6px;
      color: var(--text-main);
      font-size: 14px;
    }

    .msg {
      background: var(--chat-bg);
      border: 1px solid rgba(125, 167, 210, 0.25);
      border-radius: 10px;
      padding: 8px 10px;
    }

    .msg.realtime { color: var(--text-muted); border-style: dashed; }

    #chatControls {
      display: grid;
      grid-template-columns: 120px 1fr 100px;
      gap: 8px;
    }

    #chatControls button,
    #chatControls input {
      height: 40px;
      border-radius: 10px;
      border: 1px solid rgba(125, 167, 210, 0.35);
      background: rgba(10, 20, 31, 0.82);
      color: var(--text-main);
      padding: 0 12px;
      font-size: 14px;
    }

    #chatControls input::placeholder { color: var(--text-muted); }

    @media (max-width: 900px) {
      #gizmoToggle {
        top: 58px;
      }
      #healthStrip {
        max-width: min(96vw, 620px);
      }
    }

    @media (max-width: 720px) {
      #chatControls {
        grid-template-columns: 104px 1fr 84px;
      }
    }
  `;
  document.head.appendChild(style);
}

function pushAlert(message: string): void {
  const alerts = document.getElementById("alerts");
  if (!alerts) {
    return;
  }
  const div = document.createElement("div");
  div.className = "alert";
  div.textContent = message;
  alerts.prepend(div);
  window.setTimeout(() => {
    div.remove();
  }, 7000);
}

function renderHealth(snapshot: HealthSnapshot, details: HealthDetails): void {
  const strip = document.getElementById("healthStrip");
  if (!strip) {
    return;
  }
  const items = ["stt", "glb", "sam", "track"].map((key) => {
    const status = snapshot[key] ?? "down";
    const detail = details[key] ?? "";
    const title = detail.replace(/"/g, "&quot;");
    return `<span class="health-pill ${status}" title="${title}">${key.toUpperCase()}: ${status}</span>`;
  });
  strip.innerHTML = items.join("");
}

injectStyles();

const sceneCanvas = document.getElementById("sceneCanvas") as HTMLCanvasElement;
const overlayCanvas = document.getElementById("overlayCanvas") as HTMLCanvasElement;
const gizmoCheckbox = document.getElementById("gizmoCheckbox") as HTMLInputElement;
const chatMessages = document.getElementById("chatMessages") as HTMLElement;
const chatInput = document.getElementById("chatInput") as HTMLInputElement;
const sendButton = document.getElementById("sendButton") as HTMLButtonElement;
const micButton = document.getElementById("micButton") as HTMLButtonElement;

const scene = new GlbScene(sceneCanvas, (text) => {
  if (text.toLowerCase().includes("failed")) {
    pushAlert(text);
  }
});
scene.setGizmoEnabled(gizmoCheckbox.checked);
gizmoCheckbox.addEventListener("change", () => {
  scene.setGizmoEnabled(gizmoCheckbox.checked);
});

let trackSession: TrackSession | null = null;

const chat = new ChatDock({
  messagesEl: chatMessages,
  inputEl: chatInput,
  sendButtonEl: sendButton,
  micButtonEl: micButton,
  onAlert: pushAlert,
  onUserText: (text) => {
    void trackSession?.handleChatText(text);
  },
});

const overlay = new SamOverlay({
  sceneCanvas,
  overlayCanvas,
  captureSceneBlob: () => scene.captureBlob(),
  pickWorldPoint: (x, y) => scene.pickWorldPoint(x, y),
  projectWorldPoint: (point) => scene.projectWorldPoint(point),
  shouldIgnoreClick: () => scene.consumePointerDownOnGizmo(),
  onAlert: pushAlert,
  onSegmentClick: (x, y) => {
    void trackSession?.handleClickSelection(x, y);
  },
});
overlay.setEnabled(true);

scene.setOnCameraMoved(() => {
  overlay.onCameraMoved();
});

trackSession = new TrackSession({
  captureSceneBlob: () => scene.captureBlob(),
  onAlert: pushAlert,
  onSystemMessage: (text) => chat.appendSystemMessage(text),
  onState: (state: TrackState) => {
    if (state.status === "switching" || state.status === "stopped") {
      overlay.clearAll();
      return;
    }
    if (state.targets.length > 0) {
      // Prefer live track boxes over any leftover SAM mask.
      overlay.clearMaskOnly();
      overlay.setTrackingTargets(
        state.targets.map((target) => ({
          index: target.index,
          bbox: target.bbox,
          score: target.score,
        })),
      );
      return;
    }
    overlay.clearTrackingTargets();
  },
});

startHealthMonitor({
  getSttLiveStatus: () => chat.getLiveStatus(),
  onUpdate: (snapshot, details) => renderHealth(snapshot, details),
  onDownTransition: (module, detail) =>
    pushAlert(`${module.toUpperCase()} unavailable: ${detail}`),
});
