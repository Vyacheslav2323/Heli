import {
  AbstractMesh,
  Color3,
  Color4,
  Engine,
  FreeCamera,
  HemisphericLight,
  Matrix,
  MeshBuilder,
  PointerEventTypes,
  Scene,
  StandardMaterial,
  TransformNode,
  Vector3,
} from "@babylonjs/core";
import { SceneLoader } from "@babylonjs/core/Loading/sceneLoader";
import "@babylonjs/loaders/glTF";

type AxisId = "x" | "y" | "z";

const AXIS: Record<AxisId, { axis: Vector3; color: Color3 }> = {
  x: { axis: Vector3.Right(), color: new Color3(0.91, 0.2, 0.14) },
  y: { axis: Vector3.Up(), color: new Color3(0.6, 0.76, 0.2) },
  z: { axis: Vector3.Forward(), color: new Color3(0.23, 0.48, 0.95) },
};

export class GlbScene {
  private engine: Engine;
  private scene: Scene;
  private canvas: HTMLCanvasElement;
  private camera: FreeCamera;
  private onCameraMoved: (() => void) | null = null;

  /** Fixed world-space object/orbit center ? gizmo stays here. */
  private objectCenter = Vector3.Zero();
  /** Look-at point; moves with lateral pan (parallel to camera). */
  private lookAt = Vector3.Zero();
  private gizmoRadius = 1;
  private sceneReady = false;
  private gizmoEnabled = true;

  private gizmoRoot: TransformNode;
  private ringMats: Record<AxisId, StandardMaterial> = {} as Record<
    AxisId,
    StandardMaterial
  >;

  private dragAxis: AxisId | null = null;
  private hoverAxis: AxisId | null = null;
  private lastAngle = 0;
  private held = new Set<string>();
  private pointerDownOnGizmo = false;

  private panActive = false;
  private lastPanX = 0;
  private lastPanY = 0;
  private panSensibility = 1000;

  constructor(canvas: HTMLCanvasElement, onStatus: (text: string) => void) {
    this.canvas = canvas;
    this.engine = new Engine(canvas, true, {
      preserveDrawingBuffer: true,
      stencil: true,
    });
    this.scene = new Scene(this.engine);
    this.scene.clearColor = new Color4(0.06, 0.07, 0.1, 1);
    this.scene.setRenderingAutoClearDepthStencil(1, false, false, false);

    this.camera = new FreeCamera("camera", new Vector3(0, 2, -5), this.scene);
    this.lookAt.copyFrom(this.objectCenter);
    this.camera.setTarget(this.lookAt);
    this.camera.minZ = 0.01;
    this.camera.speed = 0;
    this.camera.angularSensibility = 0;
    this.camera.inputs.clear();

    new HemisphericLight("light", new Vector3(0.4, 1, 0.2), this.scene);

    this.gizmoRoot = new TransformNode("rotateGizmo", this.scene);
    this.gizmoRoot.setEnabled(false);
    this.makeRing("x");
    this.makeRing("y");
    this.makeRing("z");

    this.bindPointerInteraction();
    this.bindKeyboardRotation();
    this.bindWheelZoom();

    SceneLoader.Append(
      "/viewer/",
      "scene.glb",
      this.scene,
      () => {
        this.disposeCameraFrustums();
        this.fitCameraToScene();
        this.sceneReady = true;
        this.applyGizmoVisibility();
        this.syncGizmo();
        onStatus("GLB loaded");
      },
      (evt) => {
        if (evt.lengthComputable && evt.total > 0) {
          const pct = Math.round((evt.loaded / evt.total) * 100);
          onStatus(`Loading GLB ${pct}%`);
        }
      },
      (_scene, message, exception) => {
        onStatus("Failed to load GLB");
        console.error(message, exception);
      },
    );

    this.engine.runRenderLoop(() => {
      const dt = this.engine.getDeltaTime() / 1000;
      const step = 1.8 * dt;

      if (this.gizmoEnabled && !this.dragAxis) {
        if (this.held.has("q")) this.rotateCameraAroundAxis(Vector3.Right(), -step);
        if (this.held.has("w")) this.rotateCameraAroundAxis(Vector3.Right(), step);
        if (this.held.has("a")) this.rotateCameraAroundAxis(Vector3.Up(), -step);
        if (this.held.has("s")) this.rotateCameraAroundAxis(Vector3.Up(), step);
        if (this.held.has("z")) this.rotateCameraAroundAxis(Vector3.Forward(), -step);
        if (this.held.has("x")) this.rotateCameraAroundAxis(Vector3.Forward(), step);
      }

      if (this.gizmoEnabled) {
        const keyAxis =
          (this.held.has("q") || this.held.has("w") ? "x" : null) ||
          (this.held.has("a") || this.held.has("s") ? "y" : null) ||
          (this.held.has("z") || this.held.has("x") ? "z" : null);
        if (!this.dragAxis) {
          this.setRingHighlight(keyAxis as AxisId | null, this.hoverAxis);
        }
        this.syncGizmo();
      }

      this.scene.render();
    });

    window.addEventListener("resize", () => {
      this.engine.resize();
    });
  }

  setGizmoEnabled(enabled: boolean): void {
    this.gizmoEnabled = enabled;
    this.applyGizmoVisibility();
  }

  setOnCameraMoved(callback: (() => void) | null): void {
    this.onCameraMoved = callback;
  }

  isGizmoEnabled(): boolean {
    return this.gizmoEnabled;
  }

  private isSceneMesh(mesh: AbstractMesh): boolean {
    if (!mesh.isEnabled() || mesh.visibility <= 0) {
      return false;
    }
    if (
      mesh.name === "__root__" ||
      mesh.name.startsWith("ring_") ||
      mesh.name.startsWith("ring_hit_")
    ) {
      return false;
    }
    return (mesh.getTotalVertices?.() ?? 0) > 0;
  }

  pickWorldPoint(x: number, y: number): { x: number; y: number; z: number } | null {
    const hit = this.scene.pick(x, y, (mesh) => this.isSceneMesh(mesh));
    if (!hit?.hit || !hit.pickedPoint) {
      return null;
    }
    return {
      x: hit.pickedPoint.x,
      y: hit.pickedPoint.y,
      z: hit.pickedPoint.z,
    };
  }

  projectWorldPoint(point: {
    x: number;
    y: number;
    z: number;
  }): { x: number; y: number } | null {
    const world = new Vector3(point.x, point.y, point.z);
    const renderWidth = this.engine.getRenderWidth();
    const renderHeight = this.engine.getRenderHeight();
    if (renderWidth <= 0 || renderHeight <= 0) {
      return null;
    }
    const transform = this.camera.getTransformationMatrix();
    const viewport = this.camera.viewport.toGlobal(renderWidth, renderHeight);
    const projected = Vector3.Project(world, Matrix.Identity(), transform, viewport);
    if (projected.z < 0 || projected.z > 1) {
      return null;
    }
    const scaleX = this.canvas.clientWidth / renderWidth;
    const scaleY = this.canvas.clientHeight / renderHeight;
    return { x: projected.x * scaleX, y: projected.y * scaleY };
  }

  private notifyCameraMoved(source: string): void {
    this.onCameraMoved?.();
  }

  consumePointerDownOnGizmo(): boolean {
    const value = this.pointerDownOnGizmo;
    this.pointerDownOnGizmo = false;
    return value;
  }

  async captureBlob(): Promise<Blob> {
    return await new Promise<Blob>((resolve, reject) => {
      this.canvas.toBlob((blob) => {
        if (!blob) {
          reject(new Error("Unable to capture scene canvas"));
          return;
        }
        resolve(blob);
      }, "image/png");
    });
  }

  /** Depth Anything 3 embeds inferred cameras as tiny colored pyramids (~16 verts). */
  private disposeCameraFrustums(): void {
    for (const mesh of [...this.scene.meshes]) {
      if (
        mesh.name === "__root__" ||
        mesh.name.startsWith("ring_") ||
        mesh.name.startsWith("ring_hit_")
      ) {
        continue;
      }
      const verts = mesh.getTotalVertices?.() ?? 0;
      if (verts > 0 && verts < 1000) {
        mesh.dispose();
      }
    }
  }

  private fitCameraToScene(): void {
    const meshes = this.scene.meshes.filter(
      (m) =>
        m.name !== "__root__" &&
        !m.name.startsWith("ring_") &&
        !m.name.startsWith("ring_hit_") &&
        (m.getTotalVertices?.() ?? 0) > 0,
    );
    if (meshes.length === 0) {
      return;
    }

    let min = meshes[0].getBoundingInfo().boundingBox.minimumWorld.clone();
    let max = meshes[0].getBoundingInfo().boundingBox.maximumWorld.clone();
    for (const mesh of meshes) {
      const bi = mesh.getBoundingInfo().boundingBox;
      min = Vector3.Minimize(min, bi.minimumWorld);
      max = Vector3.Maximize(max, bi.maximumWorld);
    }

    this.objectCenter = min.add(max).scale(0.5);
    this.lookAt.copyFrom(this.objectCenter);
    const extent = max.subtract(min);
    this.gizmoRadius = Math.max(extent.x, extent.y, extent.z) * 0.5 || 1;
    const radius = Math.max(extent.x, extent.y, extent.z) * 1.5 || 2;
    this.camera.position = this.objectCenter.add(new Vector3(0, radius * 0.35, -radius));
    this.camera.upVector = Vector3.Up();
    this.camera.setTarget(this.lookAt);
  }

  private panCamera(dx: number, dy: number): void {
    // Translate camera and lookAt together for true lateral pan (not orbit).
    const right = this.camera.getDirection(Vector3.Right()).normalize();
    const up = this.camera.getDirection(Vector3.Up()).normalize();
    const dist = Vector3.Distance(this.camera.position, this.lookAt);
    const scale = dist / Math.max(this.panSensibility, 1);
    const offset = right.scale(-dx * scale).add(up.scale(dy * scale));
    this.camera.position.addInPlace(offset);
    this.lookAt.addInPlace(offset);
    this.camera.setTarget(this.lookAt);
    this.notifyCameraMoved("pan");
  }

  private rotateCameraAroundAxis(axis: Vector3, angle: number): void {
    const mat = Matrix.RotationAxis(axis.normalize(), angle);
    const camOffset = this.camera.position.subtract(this.objectCenter);
    const lookOffset = this.lookAt.subtract(this.objectCenter);
    this.camera.position = this.objectCenter.add(Vector3.TransformCoordinates(camOffset, mat));
    this.lookAt = this.objectCenter.add(Vector3.TransformCoordinates(lookOffset, mat));
    this.camera.upVector = Vector3.TransformNormal(this.camera.upVector, mat).normalize();
    this.camera.setTarget(this.lookAt);
    this.notifyCameraMoved("rotate");
  }

  private orientRing(mesh: AbstractMesh, id: AxisId): void {
    if (id === "x") mesh.rotation.z = Math.PI / 2;
    if (id === "z") mesh.rotation.x = Math.PI / 2;
  }

  private makeRing(id: AxisId): void {
    const torus = MeshBuilder.CreateTorus(
      `ring_${id}`,
      { diameter: 2, thickness: 0.04, tessellation: 64 },
      this.scene,
    );
    torus.parent = this.gizmoRoot;
    torus.isPickable = false;
    torus.renderingGroupId = 1;
    this.orientRing(torus, id);

    const mat = new StandardMaterial(`ringMat_${id}`, this.scene);
    mat.diffuseColor = Color3.Black();
    mat.specularColor = Color3.Black();
    mat.emissiveColor = AXIS[id].color.clone();
    mat.disableLighting = true;
    mat.alpha = 0.9;
    mat.disableDepthWrite = true;
    torus.material = mat;
    this.ringMats[id] = mat;

    const hit = MeshBuilder.CreateTorus(
      `ring_hit_${id}`,
      { diameter: 2, thickness: 0.28, tessellation: 64 },
      this.scene,
    );
    hit.parent = this.gizmoRoot;
    hit.isPickable = true;
    hit.visibility = 0;
    hit.renderingGroupId = 1;
    this.orientRing(hit, id);
  }

  private setRingHighlight(id: AxisId | null, hoverId: AxisId | null = null): void {
    for (const ax of ["x", "y", "z"] as AxisId[]) {
      const active = ax === id || ax === hoverId;
      const base = AXIS[ax].color;
      this.ringMats[ax].emissiveColor = active
        ? Color3.Lerp(base, Color3.White(), 0.5)
        : base.clone();
    }
  }

  private syncGizmo(): void {
    this.gizmoRoot.position.copyFrom(this.objectCenter);
    const dist = Vector3.Distance(this.camera.position, this.objectCenter);
    const scale = Math.max(dist * 0.22, this.gizmoRadius * 0.55);
    this.gizmoRoot.scaling.setAll(scale);
  }

  private angleAroundAxis(axis: Vector3, pointerX: number, pointerY: number): number {
    const normal = axis.normalize();
    const ray = this.scene.createPickingRay(
      pointerX,
      pointerY,
      Matrix.Identity(),
      this.camera,
    );

    const denom = Vector3.Dot(ray.direction, normal);
    let fromCenter: Vector3;

    if (Math.abs(denom) > 1e-5) {
      const t = Vector3.Dot(this.objectCenter.subtract(ray.origin), normal) / denom;
      fromCenter = ray.origin.add(ray.direction.scale(t)).subtract(this.objectCenter);
    } else {
      const viewDir = ray.direction;
      fromCenter = Vector3.Cross(normal, Vector3.Cross(viewDir, normal));
    }

    const ref = Math.abs(normal.y) < 0.9 ? Vector3.Up() : new Vector3(1, 0, 0);
    const u = Vector3.Cross(ref, normal).normalize();
    const v = Vector3.Cross(normal, u).normalize();
    return Math.atan2(Vector3.Dot(fromCenter, v), Vector3.Dot(fromCenter, u));
  }

  private meshToAxis(mesh: AbstractMesh | null): AxisId | null {
    if (!mesh) return null;
    const m = /^ring(?:_hit)?_([xyz])$/.exec(mesh.name);
    return m ? (m[1] as AxisId) : null;
  }

  private pickRingAxis(): AxisId | null {
    const hit = this.scene.pick(this.scene.pointerX, this.scene.pointerY, (m) =>
      m.name.startsWith("ring_hit_"),
    );
    return this.meshToAxis(hit?.pickedMesh ?? null);
  }

  private bindWheelZoom(): void {
    this.canvas.addEventListener(
      "wheel",
      (e) => {
        e.preventDefault();
        const dir = this.lookAt.subtract(this.camera.position);
        const dist = dir.length();
        if (dist < 1e-6) return;
        const step = Math.sign(e.deltaY) * Math.max(dist * 0.08, 0.05);
        const nextDist = Math.max(0.05, dist + step);
        this.camera.position = this.lookAt.subtract(dir.normalize().scale(nextDist));
        this.camera.setTarget(this.lookAt);
        this.notifyCameraMoved("zoom");
      },
      { passive: false },
    );
  }

  private bindPointerInteraction(): void {
    this.scene.onPointerObservable.add((pi) => {
      if (!this.sceneReady) return;

      if (pi.type === PointerEventTypes.POINTERMOVE) {
        if (this.dragAxis) {
          this.panActive = false;
          const angle = this.angleAroundAxis(
            AXIS[this.dragAxis].axis,
            this.scene.pointerX,
            this.scene.pointerY,
          );
          let delta = angle - this.lastAngle;
          if (delta > Math.PI) delta -= Math.PI * 2;
          if (delta < -Math.PI) delta += Math.PI * 2;
          this.rotateCameraAroundAxis(AXIS[this.dragAxis].axis, -delta);
          this.lastAngle = this.angleAroundAxis(
            AXIS[this.dragAxis].axis,
            this.scene.pointerX,
            this.scene.pointerY,
          );
          this.setRingHighlight(this.dragAxis);
          return;
        }

        if (this.panActive) {
          const dx = this.scene.pointerX - this.lastPanX;
          const dy = this.scene.pointerY - this.lastPanY;
          this.lastPanX = this.scene.pointerX;
          this.lastPanY = this.scene.pointerY;
          if (dx !== 0 || dy !== 0) {
            this.panCamera(dx, dy);
            this.syncGizmo();
          }
        }

        if (this.gizmoEnabled) {
          this.hoverAxis = this.pickRingAxis();
          this.canvas.style.cursor = this.hoverAxis
            ? "grab"
            : this.panActive
              ? "move"
              : "default";
        }
      }

      if (pi.type === PointerEventTypes.POINTERDOWN && pi.event.button === 0) {
        const ax = this.gizmoEnabled ? this.pickRingAxis() : null;
        this.pointerDownOnGizmo = !!ax;
        if (ax) {
          this.dragAxis = ax;
          this.panActive = false;
          this.lastAngle = this.angleAroundAxis(
            AXIS[ax].axis,
            this.scene.pointerX,
            this.scene.pointerY,
          );
          this.canvas.style.cursor = "grabbing";
          this.setRingHighlight(ax);
          pi.event.preventDefault();
        } else {
          this.panActive = true;
          this.lastPanX = this.scene.pointerX;
          this.lastPanY = this.scene.pointerY;
          this.canvas.style.cursor = "move";
        }
      }

      if (pi.type === PointerEventTypes.POINTERUP) {
        this.dragAxis = null;
        this.panActive = false;
        this.canvas.style.cursor = this.hoverAxis ? "grab" : "default";
      }
    });
  }

  private bindKeyboardRotation(): void {
    window.addEventListener("keydown", (e) => {
      if (!this.gizmoEnabled) {
        return;
      }
      const key = e.key.toLowerCase();
      if ("qwaszx".includes(key)) {
        e.preventDefault();
        this.held.add(key);
      }
    });

    window.addEventListener("keyup", (e) => {
      this.held.delete(e.key.toLowerCase());
    });
  }

  private applyGizmoVisibility(): void {
    this.gizmoRoot.setEnabled(this.gizmoEnabled && this.sceneReady);
    if (!this.gizmoEnabled) {
      this.dragAxis = null;
      this.hoverAxis = null;
      this.canvas.style.cursor = "default";
      this.setRingHighlight(null);
    }
  }
}
