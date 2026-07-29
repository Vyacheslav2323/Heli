export type ModuleHealth = "ok" | "degraded" | "down";
export type HealthSnapshot = Record<string, ModuleHealth>;
export type HealthDetails = Record<string, string>;

export type SttLiveStatus = {
  micOpen: boolean;
  wsOpen: boolean;
  packetsSent: number;
  lastPacketAt: number | null;
  lastTranscriptAt: number | null;
  lastError: string | null;
};

type MonitorOptions = {
  onUpdate: (snapshot: HealthSnapshot, details: HealthDetails) => void;
  onDownTransition: (module: string, detail: string) => void;
  getSttLiveStatus: () => SttLiveStatus;
};

const WS_URL = () => `${window.location.origin.replace(/^http/, "ws")}/stt/ws/transcribe`;

async function probeHttpHealth(url: string): Promise<{ ok: boolean; detail: string; body?: Record<string, unknown> }> {
  try {
    const res = await fetch(url, { cache: "no-store" });
    if (!res.ok) {
      return { ok: false, detail: `http ${res.status}` };
    }
    const body = (await res.json().catch(() => ({}))) as Record<string, unknown>;
    return { ok: true, detail: "http ok", body };
  } catch {
    return { ok: false, detail: "http unreachable" };
  }
}

function withTimeout<T>(promise: Promise<T>, ms: number, label: string): Promise<T> {
  return new Promise<T>((resolve, reject) => {
    const timer = window.setTimeout(() => reject(new Error(`${label} timeout`)), ms);
    promise.then(
      (value) => {
        window.clearTimeout(timer);
        resolve(value);
      },
      (error) => {
        window.clearTimeout(timer);
        reject(error);
      },
    );
  });
}

async function probeSttWebsocket(): Promise<{ ok: boolean; detail: string }> {
  return withTimeout(
    new Promise<{ ok: boolean; detail: string }>((resolve) => {
      const ws = new WebSocket(WS_URL());
      let settled = false;
      let gotHello = false;

      const finish = (ok: boolean, detail: string) => {
        if (settled) {
          return;
        }
        settled = true;
        try {
          ws.close();
        } catch {
          // ignore
        }
        resolve({ ok, detail });
      };

      ws.onopen = () => {
        ws.send(JSON.stringify({ type: "ping" }));
      };

      ws.onmessage = (event) => {
        try {
          const data = JSON.parse(String(event.data)) as { type?: string };
          if (data.type === "hello" || data.type === "ready") {
            gotHello = true;
          }
          if (data.type === "pong") {
            finish(true, "ws ping/pong");
            return;
          }
          // Some servers only greet; treat hello/ready as transport OK if no pong arrives quickly.
          if (gotHello) {
            window.setTimeout(() => {
              if (!settled) {
                finish(true, "ws hello");
              }
            }, 250);
          }
        } catch {
          // ignore malformed
        }
      };

      ws.onerror = () => finish(false, "ws error");
      ws.onclose = () => {
        if (!settled) {
          finish(gotHello, gotHello ? "ws closed after hello" : "ws closed");
        }
      };
    }),
    4000,
    "ws",
  ).catch((error: Error) => ({ ok: false, detail: error.message }));
}

async function probeMicAccess(): Promise<{ ok: boolean; detail: string }> {
  try {
    if (!navigator.mediaDevices?.enumerateDevices) {
      return { ok: false, detail: "no mediaDevices API" };
    }
    const devices = await navigator.mediaDevices.enumerateDevices();
    const hasInput = devices.some((d) => d.kind === "audioinput");
    if (!hasInput) {
      return { ok: false, detail: "no mic device" };
    }

    if (navigator.permissions?.query) {
      try {
        const permission = await navigator.permissions.query({
          name: "microphone" as PermissionName,
        });
        if (permission.state === "denied") {
          return { ok: false, detail: "mic permission denied" };
        }
        if (permission.state === "prompt") {
          return { ok: true, detail: "mic present (permission pending)" };
        }
      } catch {
        // Permissions API may reject microphone on some browsers.
      }
    }
    return { ok: true, detail: "mic available" };
  } catch {
    return { ok: false, detail: "mic probe failed" };
  }
}

function evaluateLivePipeline(live: SttLiveStatus): { ok: boolean; detail: string } {
  if (!live.micOpen && !live.wsOpen) {
    return { ok: true, detail: "idle" };
  }
  if (live.micOpen && !live.wsOpen) {
    return { ok: false, detail: "mic open but ws down" };
  }
  if (live.wsOpen && !live.micOpen) {
    return { ok: false, detail: "ws open but mic closed" };
  }
  const now = Date.now();
  if (live.packetsSent === 0) {
    return { ok: false, detail: "no audio packets yet" };
  }
  if (live.lastPacketAt && now - live.lastPacketAt > 3000) {
    return { ok: false, detail: "audio packets stalled" };
  }
  if (live.lastTranscriptAt && now - live.lastTranscriptAt < 15000) {
    return { ok: true, detail: "audio+transcript flowing" };
  }
  // Transmission is confirmed; generation may be waiting for speech.
  return { ok: true, detail: `audio flowing (${live.packetsSent} pkts), waiting transcript` };
}

async function probeStt(getLive: () => SttLiveStatus): Promise<{ status: ModuleHealth; detail: string }> {
  const http = await probeHttpHealth("/stt/health");
  if (!http.ok) {
    return { status: "down", detail: http.detail };
  }
  const ready = http.body?.ready === true || http.body?.ok === true;
  if (!ready) {
    const startup = Array.isArray(http.body?.startupErrors)
      ? String((http.body?.startupErrors as unknown[])[0] ?? "not ready")
      : "not ready";
    return { status: "down", detail: `server not ready: ${startup}` };
  }

  const [ws, mic] = await Promise.all([probeSttWebsocket(), probeMicAccess()]);
  const live = evaluateLivePipeline(getLive());

  const parts = [`http ready`, ws.detail, mic.detail, live.detail];
  if (!ws.ok) {
    return { status: "down", detail: parts.join(" | ") };
  }
  if (!mic.ok) {
    return { status: "degraded", detail: parts.join(" | ") };
  }
  if (!live.ok) {
    return { status: "degraded", detail: parts.join(" | ") };
  }
  return { status: "ok", detail: parts.join(" | ") };
}

export function startHealthMonitor(options: MonitorOptions): () => void {
  let previous: HealthSnapshot = { stt: "down", glb: "down", sam: "down", track: "down" };
  let previousDetails: HealthDetails = { stt: "", glb: "", sam: "", track: "" };

  const poll = async () => {
    const [stt, glb, sam, track] = await Promise.all([
      probeStt(options.getSttLiveStatus),
      probeHttpHealth("/viewer/health").then((r) => ({
        status: (r.ok ? "ok" : "down") as ModuleHealth,
        detail: r.detail,
      })),
      probeHttpHealth("/sam/health").then((r) => {
        const ready = r.body?.ready === true || r.body?.ok === true;
        if (!r.ok) {
          return { status: "down" as ModuleHealth, detail: r.detail };
        }
        if (!ready) {
          return { status: "degraded" as ModuleHealth, detail: "sam not ready" };
        }
        return { status: "ok" as ModuleHealth, detail: "sam ready" };
      }),
      probeHttpHealth("/track/health").then((r) => {
        const ready = r.body?.ready === true || r.body?.ok === true;
        if (!r.ok) {
          return { status: "down" as ModuleHealth, detail: r.detail };
        }
        if (!ready) {
          return { status: "degraded" as ModuleHealth, detail: "track not ready" };
        }
        return { status: "ok" as ModuleHealth, detail: "track ready" };
      }),
    ]);

    const current: HealthSnapshot = {
      stt: stt.status,
      glb: glb.status,
      sam: sam.status,
      track: track.status,
    };
    const details: HealthDetails = {
      stt: stt.detail,
      glb: glb.detail,
      sam: sam.detail,
      track: track.detail,
    };

    for (const key of Object.keys(current)) {
      const wasGood = previous[key] === "ok" || previous[key] === "degraded";
      if (wasGood && current[key] === "down") {
        options.onDownTransition(key, details[key] || previousDetails[key] || "down");
      }
    }

    previous = current;
    previousDetails = details;
    options.onUpdate(current, details);
  };

  const timer = window.setInterval(() => {
    void poll();
  }, 4000);
  void poll();

  return () => window.clearInterval(timer);
}
