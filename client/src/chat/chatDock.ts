import type { SttLiveStatus } from "../health/monitor";

type Message = {
  source: "text" | "voice" | "system";
  text: string;
  realtime?: boolean;
};

type ChatOptions = {
  messagesEl: HTMLElement;
  inputEl: HTMLInputElement;
  sendButtonEl: HTMLButtonElement;
  micButtonEl: HTMLButtonElement;
  onAlert: (message: string) => void;
  onLiveStatus?: (status: SttLiveStatus) => void;
  onUserText?: (text: string, source: "text" | "voice") => void;
};

function appendMessage(container: HTMLElement, message: Message): void {
  const line = document.createElement("div");
  line.className = `msg msg-${message.source}`;
  const prefix =
    message.source === "voice" ? "Voice" : message.source === "text" ? "Text" : "System";
  line.textContent = `${prefix}: ${message.text}`;
  if (message.realtime) {
    line.classList.add("realtime");
  }
  container.appendChild(line);
  container.scrollTop = container.scrollHeight;
}

function packAudioPacket(metadata: Record<string, unknown>, pcmInt16: Int16Array): ArrayBuffer {
  const metadataBytes = new TextEncoder().encode(JSON.stringify(metadata));
  const out = new Uint8Array(4 + metadataBytes.length + pcmInt16.byteLength);
  new DataView(out.buffer).setUint32(0, metadataBytes.length, true);
  out.set(metadataBytes, 4);
  out.set(new Uint8Array(pcmInt16.buffer), 4 + metadataBytes.length);
  return out.buffer;
}

function float32ToInt16(input: Float32Array): Int16Array {
  const out = new Int16Array(input.length);
  for (let i = 0; i < input.length; i += 1) {
    const s = Math.max(-1, Math.min(1, input[i]));
    out[i] = s < 0 ? s * 32768 : s * 32767;
  }
  return out;
}

export class ChatDock {
  private messagesEl: HTMLElement;
  private inputEl: HTMLInputElement;
  private sendButtonEl: HTMLButtonElement;
  private micButtonEl: HTMLButtonElement;
  private onAlert: (message: string) => void;
  private onLiveStatus?: (status: SttLiveStatus) => void;
  private onUserText?: (text: string, source: "text" | "voice") => void;

  private ws: WebSocket | null = null;
  private audioContext: AudioContext | null = null;
  private processor: ScriptProcessorNode | null = null;
  private mediaStream: MediaStream | null = null;
  private realtimeNode: HTMLElement | null = null;
  private sourceNode: MediaStreamAudioSourceNode | null = null;

  private live: SttLiveStatus = {
    micOpen: false,
    wsOpen: false,
    packetsSent: 0,
    lastPacketAt: null,
    lastTranscriptAt: null,
    lastError: null,
  };

  constructor(options: ChatOptions) {
    this.messagesEl = options.messagesEl;
    this.inputEl = options.inputEl;
    this.sendButtonEl = options.sendButtonEl;
    this.micButtonEl = options.micButtonEl;
    this.onAlert = options.onAlert;
    this.onLiveStatus = options.onLiveStatus;
    this.onUserText = options.onUserText;

    this.sendButtonEl.addEventListener("click", () => this.sendText());
    this.inputEl.addEventListener("keydown", (evt) => {
      if (evt.key === "Enter") {
        evt.preventDefault();
        this.sendText();
      }
    });
    this.micButtonEl.addEventListener("click", () => {
      if (this.ws) {
        void this.stopMic().catch((error) => {
          this.onAlert(`Microphone stop failed: ${String(error)}`);
        });
      } else {
        void this.startMic().catch((error) => {
          this.live.lastError = String(error);
          this.emitLive();
          this.onAlert(`Microphone start failed: ${String(error)}`);
          void this.stopMic();
        });
      }
    });
  }

  getLiveStatus(): SttLiveStatus {
    return { ...this.live };
  }

  appendSystemMessage(text: string): void {
    appendMessage(this.messagesEl, { source: "system", text });
  }

  private emitLive(): void {
    this.onLiveStatus?.(this.getLiveStatus());
  }

  private sendText(): void {
    const text = this.inputEl.value.trim();
    if (!text) {
      return;
    }
    appendMessage(this.messagesEl, { source: "text", text });
    this.onUserText?.(text, "text");
    this.inputEl.value = "";
  }

  private updateRealtime(text: string): void {
    this.live.lastTranscriptAt = Date.now();
    this.emitLive();
    if (!this.realtimeNode) {
      this.realtimeNode = document.createElement("div");
      this.realtimeNode.className = "msg msg-voice realtime";
      this.messagesEl.appendChild(this.realtimeNode);
    }
    this.realtimeNode.textContent = `Voice (live): ${text}`;
    this.messagesEl.scrollTop = this.messagesEl.scrollHeight;
  }

  private finalizeRealtime(finalText: string): void {
    this.live.lastTranscriptAt = Date.now();
    this.emitLive();
    if (this.realtimeNode) {
      this.realtimeNode.remove();
      this.realtimeNode = null;
    }
    if (finalText.trim()) {
      appendMessage(this.messagesEl, { source: "voice", text: finalText });
      this.onUserText?.(finalText, "voice");
    }
  }

  private onWsMessage(event: MessageEvent<string>): void {
    try {
      const data = JSON.parse(event.data) as { type?: string; text?: string; message?: string };
      if (data.type === "error") {
        this.live.lastError = data.message ?? "STT error";
        this.emitLive();
        this.onAlert(`STT error: ${data.message ?? "unknown"}`);
      }
      if (data.type === "realtime" && data.text) {
        this.updateRealtime(data.text);
      }
      if ((data.type === "final" || data.type === "fullSentence") && data.text) {
        this.finalizeRealtime(data.text);
      }
    } catch {
      // ignore non-json messages
    }
  }

  private async openSttSocket(): Promise<WebSocket> {
    const wsUrl = `${window.location.origin.replace(/^http/, "ws")}/stt/ws/transcribe`;
    const ws = new WebSocket(wsUrl);

    await new Promise<void>((resolve, reject) => {
      const timer = window.setTimeout(() => {
        try {
          ws.close();
        } catch {
          // ignore
        }
        reject(new Error("STT websocket timed out (proxy/server). Restart client after fix."));
      }, 5000);

      let sawHello = false;

      ws.onopen = () => {
        // Connection established; wait briefly for hello/ready.
      };

      ws.onmessage = (event) => {
        this.onWsMessage(event as MessageEvent<string>);
        try {
          const data = JSON.parse(String(event.data)) as { type?: string };
          if (data.type === "hello" || data.type === "ready") {
            sawHello = true;
            window.clearTimeout(timer);
            resolve();
          }
        } catch {
          // ignore
        }
      };

      ws.onerror = () => {
        window.clearTimeout(timer);
        reject(new Error("STT websocket connection failed"));
      };

      ws.onclose = () => {
        window.clearTimeout(timer);
        if (!sawHello) {
          reject(new Error("STT websocket closed before ready"));
        }
      };

      // Fallback: if server is slow to greet but socket is open, continue after short wait.
      window.setTimeout(() => {
        if (ws.readyState === WebSocket.OPEN && !sawHello) {
          window.clearTimeout(timer);
          resolve();
        }
      }, 1500);
    });

    // Keep message handler after handshake.
    ws.onmessage = (event) => this.onWsMessage(event as MessageEvent<string>);
    ws.onclose = () => {
      this.ws = null;
      this.live.wsOpen = false;
      this.emitLive();
      this.micButtonEl.textContent = "Start Mic";
    };
    ws.onerror = () => {
      this.live.lastError = "websocket error";
      this.emitLive();
    };

    return ws;
  }

  async startMic(): Promise<void> {
    if (this.ws) {
      return;
    }

    this.micButtonEl.disabled = true;
    this.micButtonEl.textContent = "Connecting?";

    try {
      this.ws = await this.openSttSocket();
      this.live.wsOpen = true;
      this.live.packetsSent = 0;
      this.live.lastPacketAt = null;
      this.live.lastError = null;
      this.emitLive();

      this.ws.send(JSON.stringify({ type: "start" }));

      this.mediaStream = await navigator.mediaDevices.getUserMedia({
        audio: {
          channelCount: 1,
          echoCancellation: true,
          noiseSuppression: true,
        },
      });
      this.live.micOpen = true;
      this.emitLive();

      this.audioContext = new AudioContext();
      if (this.audioContext.state === "suspended") {
        await this.audioContext.resume();
      }
      this.sourceNode = this.audioContext.createMediaStreamSource(this.mediaStream);
      this.processor = this.audioContext.createScriptProcessor(4096, 1, 1);

      this.processor.onaudioprocess = (event) => {
        if (!this.ws || this.ws.readyState !== WebSocket.OPEN) {
          return;
        }
        const channel = event.inputBuffer.getChannelData(0);
        const pcm = float32ToInt16(channel);
        const payload = packAudioPacket(
          {
            sampleRate: this.audioContext?.sampleRate ?? 44100,
            channels: 1,
            format: "pcm_s16le",
            frames: pcm.length,
          },
          pcm,
        );
        this.ws.send(payload);
        this.live.packetsSent += 1;
        this.live.lastPacketAt = Date.now();
        if (this.live.packetsSent === 1 || this.live.packetsSent % 20 === 0) {
          this.emitLive();
        }
      };

      this.sourceNode.connect(this.processor);
      this.processor.connect(this.audioContext.destination);
      this.micButtonEl.textContent = "Stop Mic";
    } finally {
      this.micButtonEl.disabled = false;
    }
  }

  async stopMic(): Promise<void> {
    if (this.ws && this.ws.readyState === WebSocket.OPEN) {
      try {
        this.ws.send(JSON.stringify({ type: "stop" }));
      } catch {
        // ignore
      }
      this.ws.close();
    }

    this.processor?.disconnect();
    this.processor = null;
    this.sourceNode?.disconnect();
    this.sourceNode = null;

    if (this.audioContext) {
      await this.audioContext.close();
      this.audioContext = null;
    }

    this.mediaStream?.getTracks().forEach((track) => track.stop());
    this.mediaStream = null;
    this.ws = null;
    this.live.micOpen = false;
    this.live.wsOpen = false;
    this.emitLive();
    this.micButtonEl.textContent = "Start Mic";
    this.micButtonEl.disabled = false;
  }
}
