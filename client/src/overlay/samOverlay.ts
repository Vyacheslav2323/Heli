export type SegmentPoint = { x: number; y: number; label: 0 | 1 };

export type WorldPoint = { x: number; y: number; z: number };
export type OverlayTrackTarget = {
  index: number;
  bbox: [number, number, number, number];
  score: number;
};

type OverlayOptions = {
  sceneCanvas: HTMLCanvasElement;
  overlayCanvas: HTMLCanvasElement;
  captureSceneBlob: () => Promise<Blob>;
  pickWorldPoint: (x: number, y: number) => WorldPoint | null;
  projectWorldPoint: (point: WorldPoint) => { x: number; y: number } | null;
  onAlert: (message: string) => void;
  shouldIgnoreClick?: () => boolean;
  onSegmentClick?: (x: number, y: number) => void;
};

type PointerState = {
  active: boolean;
  moved: boolean;
  x: number;
  y: number;
  ignore: boolean;
};

const MASK_STRIDE = 4;
const MASK_FILL = "rgba(0, 200, 255, 0.59)";

export class SamOverlay {
  private sceneCanvas: HTMLCanvasElement;
  private overlayCanvas: HTMLCanvasElement;
  private overlayCtx: CanvasRenderingContext2D;
  private captureSceneBlob: () => Promise<Blob>;
  private pickWorldPoint: (x: number, y: number) => WorldPoint | null;
  private projectWorldPoint: (point: WorldPoint) => { x: number; y: number } | null;
  private onAlert: (message: string) => void;
  private shouldIgnoreClick: () => boolean;
  private onSegmentClick: (x: number, y: number) => void;
  private points: SegmentPoint[] = [];
  private trackingTargets: OverlayTrackTarget[] = [];
  private worldMaskPoints: WorldPoint[] = [];
  private worldClickPoint: WorldPoint | null = null;
  private enabled = true;
  private hasMask = false;
  private maskDrawnAt = 0;
  private redrawScheduled = false;
  private lastReprojectCount = 0;
  private pointer: PointerState = {
    active: false,
    moved: false,
    x: 0,
    y: 0,
    ignore: false,
  };

  constructor(options: OverlayOptions) {
    this.sceneCanvas = options.sceneCanvas;
    this.overlayCanvas = options.overlayCanvas;
    const ctx = this.overlayCanvas.getContext("2d");
    if (!ctx) {
      throw new Error("2D overlay context unavailable");
    }
    this.overlayCtx = ctx;
    this.captureSceneBlob = options.captureSceneBlob;
    this.pickWorldPoint = options.pickWorldPoint;
    this.projectWorldPoint = options.projectWorldPoint;
    this.onAlert = options.onAlert;
    this.shouldIgnoreClick = options.shouldIgnoreClick ?? (() => false);
    this.onSegmentClick = options.onSegmentClick ?? (() => {});

    this.overlayCanvas.style.pointerEvents = "none";

    this.sceneCanvas.addEventListener("pointerdown", (evt) => {
      if (!this.enabled || evt.button !== 0) {
        this.pointer.active = false;
        return;
      }
      const rect = this.sceneCanvas.getBoundingClientRect();
      this.pointer = {
        active: true,
        moved: false,
        x: evt.clientX - rect.left,
        y: evt.clientY - rect.top,
        ignore: this.shouldIgnoreClick(),
      };
    });

    this.sceneCanvas.addEventListener("pointermove", (evt) => {
      if (!this.pointer.active) {
        return;
      }
      const rect = this.sceneCanvas.getBoundingClientRect();
      const x = evt.clientX - rect.left;
      const y = evt.clientY - rect.top;
      if (Math.abs(x - this.pointer.x) > 5 || Math.abs(y - this.pointer.y) > 5) {
        this.pointer.moved = true;
      }
    });

    this.sceneCanvas.addEventListener("pointerup", (evt) => {
      if (!this.pointer.active || evt.button !== 0) {
        this.pointer.active = false;
        return;
      }
      const shouldSegment = !this.pointer.moved && !this.pointer.ignore;
      const x = this.pointer.x;
      const y = this.pointer.y;
      this.pointer.active = false;
      if (!shouldSegment) {
        return;
      }
      void this.handleClick(x, y);
    });

    this.sceneCanvas.addEventListener("pointercancel", () => {
      this.pointer.active = false;
    });

    this.resize();
    window.addEventListener("resize", () => this.resize());
  }

  setEnabled(value: boolean): void {
    this.enabled = value;
  }

  setTrackingTargets(targets: OverlayTrackTarget[]): void {
    this.trackingTargets = [...targets];
    this.redrawOverlay();
  }

  clearTrackingTargets(): void {
    if (this.trackingTargets.length === 0) {
      return;
    }
    this.trackingTargets = [];
    this.redrawOverlay();
  }

  /** Drop SAM mask + track boxes so a new chat target can take over cleanly. */
  clearAll(): void {
    this.trackingTargets = [];
    this.worldMaskPoints = [];
    this.worldClickPoint = null;
    this.points = [];
    this.hasMask = false;
    this.overlayCtx.clearRect(0, 0, this.overlayCanvas.width, this.overlayCanvas.height);
  }

  clearMaskOnly(): void {
    this.worldMaskPoints = [];
    this.worldClickPoint = null;
    this.points = [];
    this.hasMask = false;
  }

  /** Called when the 3D camera moves so the mask can reproject. */
  onCameraMoved(): void {
    if (this.trackingTargets.length > 0) {
      return;
    }
    if (!this.hasMask || this.worldMaskPoints.length === 0) {
      return;
    }
    this.scheduleWorldRedraw();
  }

  private scheduleWorldRedraw(): void {
    if (this.redrawScheduled) {
      return;
    }
    this.redrawScheduled = true;
    requestAnimationFrame(() => {
      this.redrawScheduled = false;
      this.redrawFromWorld();
    });
  }

  private resize(): void {
    const width = this.sceneCanvas.clientWidth;
    const height = this.sceneCanvas.clientHeight;
    this.overlayCanvas.width = width;
    this.overlayCanvas.height = height;
    this.overlayCanvas.style.width = `${width}px`;
    this.overlayCanvas.style.height = `${height}px`;
    this.redrawOverlay();
  }

  private redrawOverlay(): void {
    if (this.trackingTargets.length > 0) {
      this.drawTrackingTargets();
      return;
    }
    if (this.worldMaskPoints.length > 0) {
      this.redrawFromWorld();
      return;
    }
    this.overlayCtx.clearRect(0, 0, this.overlayCanvas.width, this.overlayCanvas.height);
    this.redrawPoints();
  }

  private bakeMaskToWorld(image: HTMLImageElement): void {
    const width = this.overlayCanvas.width;
    const height = this.overlayCanvas.height;
    const tmp = document.createElement("canvas");
    tmp.width = width;
    tmp.height = height;
    const tmpCtx = tmp.getContext("2d");
    if (!tmpCtx) {
      return;
    }
    tmpCtx.drawImage(image, 0, 0, width, height);
    const { data } = tmpCtx.getImageData(0, 0, width, height);

    const worldPoints: WorldPoint[] = [];
    for (let y = 0; y < height; y += MASK_STRIDE) {
      for (let x = 0; x < width; x += MASK_STRIDE) {
        const alpha = data[(y * width + x) * 4 + 3];
        if (alpha < 16) {
          continue;
        }
        const world = this.pickWorldPoint(x, y);
        if (world) {
          worldPoints.push(world);
        }
      }
    }
    this.worldMaskPoints = worldPoints;
  }

  private redrawFromWorld(): void {
    this.overlayCtx.clearRect(0, 0, this.overlayCanvas.width, this.overlayCanvas.height);
    this.overlayCtx.fillStyle = MASK_FILL;
    let drawn = 0;
    const half = MASK_STRIDE / 2;
    for (const world of this.worldMaskPoints) {
      const screen = this.projectWorldPoint(world);
      if (!screen) {
        continue;
      }
      this.overlayCtx.fillRect(screen.x - half, screen.y - half, MASK_STRIDE, MASK_STRIDE);
      drawn += 1;
    }
    this.lastReprojectCount = drawn;

    if (this.worldClickPoint) {
      const clickScreen = this.projectWorldPoint(this.worldClickPoint);
      if (clickScreen) {
        this.points = [{ x: clickScreen.x, y: clickScreen.y, label: 1 }];
      }
    }
    this.redrawPoints();
  }

  private drawTrackingTargets(): void {
    this.overlayCtx.clearRect(0, 0, this.overlayCanvas.width, this.overlayCanvas.height);
    this.overlayCtx.lineWidth = 2;
    this.overlayCtx.font = "12px 'Segoe UI', sans-serif";
    for (const target of this.trackingTargets) {
      const [x0, y0, x1, y1] = target.bbox;
      const w = Math.max(1, x1 - x0);
      const h = Math.max(1, y1 - y0);
      this.overlayCtx.strokeStyle = "rgba(255, 223, 102, 0.95)";
      this.overlayCtx.fillStyle = "rgba(255, 223, 102, 0.14)";
      this.overlayCtx.fillRect(x0, y0, w, h);
      this.overlayCtx.strokeRect(x0, y0, w, h);
      const label = `#${target.index} ${(target.score * 100).toFixed(0)}%`;
      this.overlayCtx.fillStyle = "rgba(255, 223, 102, 0.95)";
      this.overlayCtx.fillText(label, x0 + 4, Math.max(12, y0 - 6));
    }
  }

  private drawMask(image: HTMLImageElement): void {
    this.bakeMaskToWorld(image);
    this.hasMask = this.worldMaskPoints.length > 0;
    this.maskDrawnAt = Date.now();
    if (this.hasMask) {
      this.redrawFromWorld();
    } else {
      // Fallback: screen-space draw if picking failed (e.g. empty hit).
      this.overlayCtx.clearRect(0, 0, this.overlayCanvas.width, this.overlayCanvas.height);
      this.overlayCtx.drawImage(image, 0, 0, this.overlayCanvas.width, this.overlayCanvas.height);
      this.hasMask = true;
      this.redrawPoints();
    }
  }

  private redrawPoints(): void {
    for (const point of this.points) {
      this.overlayCtx.beginPath();
      this.overlayCtx.arc(point.x, point.y, 5, 0, Math.PI * 2);
      this.overlayCtx.fillStyle = point.label === 1 ? "#53f36e" : "#f46a6a";
      this.overlayCtx.fill();
      this.overlayCtx.lineWidth = 1;
      this.overlayCtx.strokeStyle = "#ffffff";
      this.overlayCtx.stroke();
    }
  }

  private async ensureSamHealthy(): Promise<boolean> {
    try {
      const response = await fetch("/sam/health", { cache: "no-store" });
      if (!response.ok) {
        this.onAlert("MobileSAM health check failed");
        return false;
      }
      const body = (await response.json()) as { ok?: boolean; ready?: boolean };
      if (!body.ok || !body.ready) {
        this.onAlert("MobileSAM is not ready");
        return false;
      }
      return true;
    } catch {
      this.onAlert("MobileSAM is unavailable");
      return false;
    }
  }

  private async handleClick(x: number, y: number): Promise<void> {
    if (!(await this.ensureSamHealthy())) {
      return;
    }

    const clampedX = Math.max(0, Math.min(this.overlayCanvas.width, x));
    const clampedY = Math.max(0, Math.min(this.overlayCanvas.height, y));
    this.points = [{ x: clampedX, y: clampedY, label: 1 }];
    this.worldClickPoint = this.pickWorldPoint(clampedX, clampedY);
    this.worldMaskPoints = [];
    this.hasMask = false;
    this.trackingTargets = [];

    let sceneBlob: Blob;
    try {
      sceneBlob = await this.captureSceneBlob();
    } catch (error) {
      this.onAlert(`Scene capture failed: ${String(error)}`);
      return;
    }

    const form = new FormData();
    form.append("image", sceneBlob, "scene.png");
    form.append("points", JSON.stringify(this.points));

    let response: Response;
    try {
      response = await fetch("/sam/segment", {
        method: "POST",
        body: form,
      });
    } catch {
      this.onAlert("MobileSAM request failed");
      return;
    }

    if (!response.ok) {
      const detail = await response.text();
      this.onAlert(`MobileSAM failed (${response.status}): ${detail}`);
      return;
    }

    const data = (await response.json()) as { mask: string };
    const image = new Image();
    image.src = `data:image/png;base64,${data.mask}`;
    await image.decode();
    this.drawMask(image);
    this.onSegmentClick(clampedX, clampedY);
  }
}
