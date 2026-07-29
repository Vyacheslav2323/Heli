export type TrackTarget = {
  index: number;
  bbox: [number, number, number, number];
  score: number;
};

export type TrackState = {
  active: boolean;
  status: string;
  prompt: string;
  source: string;
  bbox: [number, number, number, number] | null;
  targets: TrackTarget[];
  selected_index: number | null;
  mode: "all" | "single";
  frame_width: number | null;
  frame_height: number | null;
  score: number | null;
  fps: number | null;
  updated_at_s: number | null;
  last_error: string | null;
};

type TrackResponse = {
  ok: boolean;
  state: TrackState;
  message?: string;
};

type TrackSessionOptions = {
  captureSceneBlob: () => Promise<Blob>;
  onState: (state: TrackState) => void;
  onAlert: (message: string) => void;
  onSystemMessage: (text: string) => void;
};

const FRAME_INTERVAL_MS = 160;

export class TrackSession {
  private captureSceneBlob: () => Promise<Blob>;
  private onState: (state: TrackState) => void;
  private onAlert: (message: string) => void;
  private onSystemMessage: (text: string) => void;

  private state: TrackState | null = null;
  private frameTimer: number | null = null;
  private frameInFlight = false;
  private commandEpoch = 0;
  private expectedPrompt = "";

  constructor(options: TrackSessionOptions) {
    this.captureSceneBlob = options.captureSceneBlob;
    this.onState = options.onState;
    this.onAlert = options.onAlert;
    this.onSystemMessage = options.onSystemMessage;
  }

  async handleChatText(text: string): Promise<void> {
    const trimmed = text.trim();
    if (!trimmed) {
      return;
    }
    // Any chat/voice utterance is a tracking intent; control phrases stay as-is.
    const commandText = this.normalizeTrackCommand(trimmed);
    const epoch = ++this.commandEpoch;
    this.stopFramePump();
    this.onState({
      active: false,
      status: "switching",
      prompt: "",
      source: "app-upload",
      bbox: null,
      targets: [],
      selected_index: null,
      mode: "all",
      frame_width: null,
      frame_height: null,
      score: null,
      fps: null,
      updated_at_s: Date.now() / 1000,
      last_error: null,
    });
    try {
      const response = await fetch("/track/command", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text: commandText }),
      });
      if (epoch !== this.commandEpoch) {
        return;
      }
      const body = (await response.json()) as TrackResponse;
      if (!response.ok) {
        this.onAlert(`Tracking command failed (${response.status})`);
        return;
      }
      this.consumeState(body.state, body.message, epoch);
    } catch {
      if (epoch === this.commandEpoch) {
        this.onAlert("Track service unavailable");
      }
    }
  }

  async handleClickSelection(x: number, y: number): Promise<void> {
    const epoch = ++this.commandEpoch;
    this.stopFramePump();
    this.onState({
      active: false,
      status: "switching",
      prompt: "",
      source: "app-upload",
      bbox: null,
      targets: [],
      selected_index: null,
      mode: "all",
      frame_width: null,
      frame_height: null,
      score: null,
      fps: null,
      updated_at_s: Date.now() / 1000,
      last_error: null,
    });
    try {
      const imageBlob = await this.captureSceneBlob();
      const form = new FormData();
      form.append("image", imageBlob, "scene.png");
      form.append("x", String(x));
      form.append("y", String(y));
      form.append("label", "selection");
      const response = await fetch("/track/click", {
        method: "POST",
        body: form,
      });
      if (epoch !== this.commandEpoch) {
        return;
      }
      if (!response.ok) {
        const detail = await response.text();
        this.onAlert(`Track click failed (${response.status}): ${detail}`);
        return;
      }
      const body = (await response.json()) as TrackResponse;
      this.consumeState(body.state, "Tracking selection", epoch);
    } catch {
      if (epoch === this.commandEpoch) {
        this.onAlert("Unable to start click tracking");
      }
    }
  }

  stopLocal(): void {
    this.commandEpoch += 1;
    this.stopFramePump();
  }

  private consumeState(state: TrackState, overrideMessage: string | undefined, epoch: number): void {
    if (epoch !== this.commandEpoch) {
      return;
    }
    this.state = state;
    this.expectedPrompt = state.prompt;
    this.onState(state);
    const message = overrideMessage ?? state.status;
    if (message) {
      this.onSystemMessage(message);
    }
    if (this.shouldPump(state)) {
      this.startFramePump();
    } else {
      this.stopFramePump();
    }
  }

  private shouldPump(state: TrackState): boolean {
    if (state.active) {
      return true;
    }
    return state.status.includes("armed") && state.prompt.length > 0;
  }

  private startFramePump(): void {
    if (this.frameTimer !== null) {
      return;
    }
    this.frameTimer = window.setInterval(() => {
      void this.sendFrame();
    }, FRAME_INTERVAL_MS);
    void this.sendFrame();
  }

  private stopFramePump(): void {
    if (this.frameTimer !== null) {
      window.clearInterval(this.frameTimer);
      this.frameTimer = null;
    }
    this.frameInFlight = false;
  }

  private async sendFrame(): Promise<void> {
    if (this.frameInFlight || !this.state) {
      return;
    }
    const epoch = this.commandEpoch;
    this.frameInFlight = true;
    try {
      const imageBlob = await this.captureSceneBlob();
      if (epoch !== this.commandEpoch) {
        return;
      }
      const form = new FormData();
      form.append("image", imageBlob, "scene.png");
      const response = await fetch("/track/frame", {
        method: "POST",
        body: form,
      });
      if (epoch !== this.commandEpoch) {
        return;
      }
      if (!response.ok) {
        this.onAlert(`Track frame failed (${response.status})`);
        this.stopFramePump();
        return;
      }
      const body = (await response.json()) as TrackResponse;
      if (epoch !== this.commandEpoch) {
        return;
      }
      // Ignore frames that still report a different prompt after a switch.
      if (
        this.expectedPrompt &&
        body.state.prompt &&
        body.state.prompt !== this.expectedPrompt &&
        body.state.prompt !== "__click__."
      ) {
        return;
      }
      this.state = body.state;
      this.onState(body.state);
      if (!this.shouldPump(body.state)) {
        this.stopFramePump();
      }
    } catch {
      if (epoch === this.commandEpoch) {
        this.onAlert("Track frame update failed");
        this.stopFramePump();
      }
    } finally {
      if (epoch === this.commandEpoch) {
        this.frameInFlight = false;
      }
    }
  }

  private normalizeTrackCommand(text: string): string {
    const normalized = text.trim().toLowerCase();
    if (
      normalized.startsWith("track ") ||
      normalized.startsWith("follow ") ||
      normalized === "track all" ||
      normalized === "follow all" ||
      normalized === "stop tracking" ||
      normalized === "stop track" ||
      normalized === "cancel tracking" ||
      normalized === "stop" ||
      normalized === "cancel" ||
      normalized === "tracking status" ||
      normalized === "track status" ||
      normalized === "status tracking" ||
      normalized === "status"
    ) {
      return text.trim();
    }
    return `track ${text.trim()}`;
  }
}
