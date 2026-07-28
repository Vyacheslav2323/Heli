import {
  Engine,
  Scene,
  FreeCamera,
  HemisphericLight,
  Vector3,
  Color3,
  Color4,
  Matrix,
  MeshBuilder,
  StandardMaterial,
  TransformNode,
  AbstractMesh,
  PointerEventTypes,
  Tools,
} from "@babylonjs/core";
import { SceneLoader } from "@babylonjs/core/Loading/sceneLoader";
import "@babylonjs/loaders/glTF";

const canvas = document.getElementById("renderCanvas") as HTMLCanvasElement;
const status = document.getElementById("status")!;
const gizmoCheckbox = document.getElementById(
  "gizmoCheckbox",
) as HTMLInputElement;

let gizmoEnabled = true;
let sceneReady = false;

const engine = new Engine(canvas, true, {
  preserveDrawingBuffer: true,
  stencil: true,
});

const scene = new Scene(engine);
scene.clearColor = new Color4(0.1, 0.1, 0.12, 1);
scene.setRenderingAutoClearDepthStencil(1, false, false, false);

const camera = new FreeCamera("camera", new Vector3(0, 2, -5), scene);
camera.setTarget(Vector3.Zero());
camera.minZ = 0.01;
camera.speed = 0;
camera.angularSensibility = 0;
camera.inputs.clear();

let cameraTarget = Vector3.Zero();
let gizmoRadius = 1;

new HemisphericLight("light", new Vector3(0, 1, 0), scene);

/** Rotate camera around world axis through the look-at target. */
function rotateCameraAroundAxis(axis: Vector3, angle: number) {
  const mat = Matrix.RotationAxis(axis.normalize(), angle);
  const offset = camera.position.subtract(cameraTarget);
  camera.position = cameraTarget.add(Vector3.TransformCoordinates(offset, mat));
  camera.upVector = Vector3.TransformNormal(camera.upVector, mat).normalize();
  camera.setTarget(cameraTarget);
}

// --- Blender-style rotation rings ---

type AxisId = "x" | "y" | "z";

const AXIS: Record<AxisId, { axis: Vector3; color: Color3 }> = {
  x: { axis: Vector3.Right(), color: new Color3(0.91, 0.2, 0.14) },
  y: { axis: Vector3.Up(), color: new Color3(0.6, 0.76, 0.2) },
  z: { axis: Vector3.Forward(), color: new Color3(0.23, 0.48, 0.95) },
};

const gizmoRoot = new TransformNode("rotateGizmo", scene);
gizmoRoot.setEnabled(false);

const ringMeshes: Record<AxisId, AbstractMesh> = {} as Record<AxisId, AbstractMesh>;
const ringMats: Record<AxisId, StandardMaterial> = {} as Record<
  AxisId,
  StandardMaterial
>;

/**
 * Babylon CreateTorus revolves around +Y (ring lies in XZ).
 * Orient so the torus axis matches the rotation axis.
 */
function orientRing(mesh: AbstractMesh, id: AxisId) {
  if (id === "x") mesh.rotation.z = Math.PI / 2; // Y → X
  // y: default (around Y)
  if (id === "z") mesh.rotation.x = Math.PI / 2; // Y → Z
}

function makeRing(id: AxisId): AbstractMesh {
  // Visible thin ring
  const torus = MeshBuilder.CreateTorus(
    `ring_${id}`,
    { diameter: 2, thickness: 0.04, tessellation: 64 },
    scene,
  );
  torus.parent = gizmoRoot;
  torus.isPickable = false;
  torus.renderingGroupId = 1;
  orientRing(torus, id);

  const mat = new StandardMaterial(`ringMat_${id}`, scene);
  mat.diffuseColor = Color3.Black();
  mat.specularColor = Color3.Black();
  mat.emissiveColor = AXIS[id].color.clone();
  mat.disableLighting = true;
  mat.alpha = 0.9;
  mat.disableDepthWrite = true;
  torus.material = mat;

  // Invisible fat hit volume — stays put (no hover scale on visual)
  const hit = MeshBuilder.CreateTorus(
    `ring_hit_${id}`,
    { diameter: 2, thickness: 0.28, tessellation: 64 },
    scene,
  );
  hit.parent = gizmoRoot;
  hit.isPickable = true;
  hit.visibility = 0;
  hit.renderingGroupId = 1;
  orientRing(hit, id);

  ringMeshes[id] = torus;
  ringMats[id] = mat;
  return torus;
}

makeRing("x");
makeRing("y");
makeRing("z");

function setRingHighlight(id: AxisId | null, hoverId: AxisId | null = null) {
  for (const ax of ["x", "y", "z"] as AxisId[]) {
    const active = ax === id || ax === hoverId;
    const base = AXIS[ax].color;
    // Brighten only — do not scale (scaling pulled the ring away from the cursor)
    ringMats[ax].emissiveColor = active
      ? Color3.Lerp(base, Color3.White(), 0.5)
      : base.clone();
  }
}

function syncGizmo() {
  gizmoRoot.position.copyFrom(cameraTarget);
  const dist = Vector3.Distance(camera.position, cameraTarget);
  const scale = Math.max(dist * 0.22, gizmoRadius * 0.55);
  gizmoRoot.scaling.setAll(scale);
}

/**
 * Angle of the pointer around `axis`, measured in the ring plane
 * (ray ∩ plane through target). Dragging along the ring follows this angle.
 */
function angleAroundAxis(
  axis: Vector3,
  pointerX: number,
  pointerY: number,
): number {
  const normal = axis.normalize();
  const ray = scene.createPickingRay(
    pointerX,
    pointerY,
    Matrix.Identity(),
    camera,
  );

  const denom = Vector3.Dot(ray.direction, normal);
  let fromCenter: Vector3;

  if (Math.abs(denom) > 1e-5) {
    const t = Vector3.Dot(cameraTarget.subtract(ray.origin), normal) / denom;
    fromCenter = ray.origin.add(ray.direction.scale(t)).subtract(cameraTarget);
  } else {
    const viewDir = ray.direction;
    fromCenter = Vector3.Cross(normal, Vector3.Cross(viewDir, normal));
  }

  const ref =
    Math.abs(normal.y) < 0.9 ? Vector3.Up() : new Vector3(1, 0, 0);
  const u = Vector3.Cross(ref, normal).normalize();
  const v = Vector3.Cross(normal, u).normalize();
  return Math.atan2(Vector3.Dot(fromCenter, v), Vector3.Dot(fromCenter, u));
}

let dragAxis: AxisId | null = null;
let lastAngle = 0;
let hoverAxis: AxisId | null = null;

function applyGizmoVisibility() {
  gizmoRoot.setEnabled(gizmoEnabled && sceneReady);
  if (!gizmoEnabled) {
    dragAxis = null;
    hoverAxis = null;
    canvas.style.cursor = "default";
    setRingHighlight(null);
  }
}

gizmoCheckbox.addEventListener("change", () => {
  gizmoEnabled = gizmoCheckbox.checked;
  applyGizmoVisibility();
});

function meshToAxis(mesh: AbstractMesh | null): AxisId | null {
  if (!mesh) return null;
  const m = /^ring(?:_hit)?_([xyz])$/.exec(mesh.name);
  return m ? (m[1] as AxisId) : null;
}

function pickRingAxis(): AxisId | null {
  const hit = scene.pick(scene.pointerX, scene.pointerY, (m) =>
    m.name.startsWith("ring_hit_"),
  );
  return meshToAxis(hit?.pickedMesh ?? null);
}

scene.onPointerObservable.add((pi) => {
  if (!gizmoEnabled) return;

  if (pi.type === PointerEventTypes.POINTERMOVE) {
    if (dragAxis) {
      const angle = angleAroundAxis(
        AXIS[dragAxis].axis,
        scene.pointerX,
        scene.pointerY,
      );
      let delta = angle - lastAngle;
      if (delta > Math.PI) delta -= Math.PI * 2;
      if (delta < -Math.PI) delta += Math.PI * 2;
      rotateCameraAroundAxis(AXIS[dragAxis].axis, -delta);
      // Re-sample after camera move to avoid ±delta oscillation feedback
      lastAngle = angleAroundAxis(
        AXIS[dragAxis].axis,
        scene.pointerX,
        scene.pointerY,
      );
      setRingHighlight(dragAxis);
      return;
    }
    hoverAxis = pickRingAxis();
    canvas.style.cursor = hoverAxis ? "grab" : "default";
    const keyAxis =
      (held.has("q") || held.has("w") ? "x" : null) ||
      (held.has("a") || held.has("s") ? "y" : null) ||
      (held.has("z") || held.has("x") ? "z" : null);
    setRingHighlight(keyAxis, hoverAxis);
  }

  if (pi.type === PointerEventTypes.POINTERDOWN && pi.event.button === 0) {
    const ax = pickRingAxis();
    if (ax) {
      dragAxis = ax;
      lastAngle = angleAroundAxis(
        AXIS[ax].axis,
        scene.pointerX,
        scene.pointerY,
      );
      canvas.style.cursor = "grabbing";
      setRingHighlight(ax);
      pi.event.preventDefault();
    }
  }

  if (pi.type === PointerEventTypes.POINTERUP) {
    dragAxis = null;
    canvas.style.cursor = hoverAxis ? "grab" : "default";
  }
});

const ROTATE_SPEED = 1.8;
const held = new Set<string>();

window.addEventListener("keydown", (e) => {
  const key = e.key.toLowerCase();
  if ("qwaszx".includes(key)) {
    e.preventDefault();
    held.add(key);
  }
});

window.addEventListener("keyup", (e) => {
  held.delete(e.key.toLowerCase());
});

canvas.addEventListener(
  "wheel",
  (e) => {
    e.preventDefault();
    const dir = cameraTarget.subtract(camera.position);
    const dist = dir.length();
    if (dist < 1e-6) return;
    const step = Math.sign(e.deltaY) * Math.max(dist * 0.08, 0.05);
    const nextDist = Math.max(0.05, dist + step);
    camera.position = cameraTarget.subtract(dir.normalize().scale(nextDist));
  },
  { passive: false },
);

SceneLoader.Append(
  "/",
  "scene.glb",
  scene,
  () => {
    // Depth Anything 3 embeds camera frustums as tiny line meshes (~16 verts).
    // Keep the point cloud / reconstruction; dispose the pyramid markers.
    for (const mesh of [...scene.meshes]) {
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

    const meshes = scene.meshes.filter(
      (m) =>
        m.name !== "__root__" &&
        !m.name.startsWith("ring_") &&
        !m.name.startsWith("ring_hit_") &&
        (m.getTotalVertices?.() ?? 0) > 0,
    );
    if (meshes.length > 0) {
      let min = meshes[0].getBoundingInfo().boundingBox.minimumWorld.clone();
      let max = meshes[0].getBoundingInfo().boundingBox.maximumWorld.clone();
      for (const mesh of meshes) {
        const bi = mesh.getBoundingInfo().boundingBox;
        min = Vector3.Minimize(min, bi.minimumWorld);
        max = Vector3.Maximize(max, bi.maximumWorld);
      }
      cameraTarget = min.add(max).scale(0.5);
      const extent = max.subtract(min);
      gizmoRadius = Math.max(extent.x, extent.y, extent.z) * 0.5 || 1;
      const radius = Math.max(extent.x, extent.y, extent.z) * 1.5 || 2;
      camera.position = cameraTarget.add(new Vector3(0, radius * 0.35, -radius));
      camera.upVector = Vector3.Up();
      camera.setTarget(cameraTarget);
    }
    sceneReady = true;
    applyGizmoVisibility();
    syncGizmo();
    status.textContent =
      "Drag rings · QW=X · AS=Y · ZX=Z · scroll=zoom";
    setTimeout(() => {
      status.style.opacity = "0.5";
    }, 3000);
  },
  (event) => {
    if (event.lengthComputable) {
      const pct = Math.round((event.loaded / event.total) * 100);
      status.textContent = `Loading scene.glb… ${pct}%`;
    }
  },
  (_scene, message, exception) => {
    status.textContent = `Failed to load: ${message}`;
    console.error(message, exception);
  },
);

engine.runRenderLoop(() => {
  const dt = engine.getDeltaTime() / 1000;
  const step = ROTATE_SPEED * dt;

  if (!dragAxis) {
    if (held.has("q")) rotateCameraAroundAxis(Vector3.Right(), -step);
    if (held.has("w")) rotateCameraAroundAxis(Vector3.Right(), step);
    if (held.has("a")) rotateCameraAroundAxis(Vector3.Up(), -step);
    if (held.has("s")) rotateCameraAroundAxis(Vector3.Up(), step);
    if (held.has("z")) rotateCameraAroundAxis(Vector3.Forward(), -step);
    if (held.has("x")) rotateCameraAroundAxis(Vector3.Forward(), step);
  }

  const keyAxis =
    (held.has("q") || held.has("w") ? "x" : null) ||
    (held.has("a") || held.has("s") ? "y" : null) ||
    (held.has("z") || held.has("x") ? "z" : null);
  if (gizmoEnabled && !dragAxis) setRingHighlight(keyAxis, hoverAxis);

  if (gizmoEnabled) syncGizmo();
  scene.render();
});

window.addEventListener("resize", () => {
  engine.resize();
});

(window as unknown as { captureScene: () => void }).captureScene = () => {
  Tools.CreateScreenshot(engine, camera, { precision: 1 });
};
