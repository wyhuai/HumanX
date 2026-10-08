import * as THREE from "three";
import { OrbitControls } from "./vendor/OrbitControls.js";

const EARTH_GRAVITY = [0, 0, -9.80665];
const THROW_ARROW_TIME_SCALE = 0.2;
const MAX_THROW_SPEED = 20;
const PRE_ARROW_COLOR = 0x8bd59f;
const POST_ARROW_COLOR = 0x63d6db;
const GROUND_GRID_OFFSET = 0.001;
// Match the four foot contact sites in g1_29dof_old.xml.
const FOOT_CONTACT_POINTS_LOCAL = [
  [-0.05, 0.025, -0.03],
  [-0.05, -0.025, -0.03],
  [0.12, 0.03, -0.03],
  [0.12, -0.03, -0.03],
];
const GROUND_COLOR_PRESETS = [
  { label: "Slate", hex: 0x3f4b59, css: "#3f4b59" },
  { label: "Light gray", hex: 0xc4c9c6, css: "#c4c9c6" },
  { label: "Warm yellow", hex: 0xb19a4c, css: "#b19a4c" },
  { label: "Blue gray", hex: 0x536f82, css: "#536f82" },
  { label: "Green gray", hex: 0x587363, css: "#587363" },
];
const SYNTH_SETTINGS_STORAGE_KEY = "xgen.hoiEditor.synthSettings.v1";
const DATASET_FOLDER_STORAGE_KEY = "xgen.hoiEditor.datasetFolder.v1";
const DEFAULT_GROUNDING_CORRECTION = true;
const ANCHOR_PALM_CENTER = "palm_center";
const ANCHOR_RIGHT_PALM_NORMAL = "right_palm_normal";
const RIGHT_PALM_NORMAL_OFFSET = 0.05;
const RIGHT_PALM_NORMAL_LOCAL = [1, 0, 0];
const RIGHT_PALM_OBJECT_SIZE = 0.05;

function readPersistentSynthSettings() {
  const defaults = {
    objectType: null,
    objectSizeMode: null,
    objectSize: null,
    yawOnly: null,
    supportSurfaces: null,
    preContactMode: null,
    postContactMode: null,
    anchorMode: null,
  };
  if (typeof window === "undefined") return defaults;
  try {
    const stored = JSON.parse(
      window.localStorage?.getItem(SYNTH_SETTINGS_STORAGE_KEY) || "null",
    );
    if (!stored || typeof stored !== "object") return defaults;
    if (stored.objectType === "box" || stored.objectType === "sphere") {
      defaults.objectType = stored.objectType;
    }
    if (stored.objectSizeMode === "auto" || stored.objectSizeMode === "manual") {
      defaults.objectSizeMode = stored.objectSizeMode;
    }
    const objectSize = Number(stored.objectSize);
    if (Number.isFinite(objectSize) && objectSize > 0.02) {
      defaults.objectSize = objectSize;
    }
    if (typeof stored.yawOnly === "boolean") defaults.yawOnly = stored.yawOnly;
    if (typeof stored.supportSurfaces === "boolean") {
      defaults.supportSurfaces = stored.supportSurfaces;
    }
    if (stored.preContactMode === "hold" || stored.preContactMode === "parabolic") {
      defaults.preContactMode = stored.preContactMode;
    }
    if (stored.postContactMode === "hold" || stored.postContactMode === "parabolic") {
      defaults.postContactMode = stored.postContactMode;
    }
    if (
      stored.anchorMode === ANCHOR_PALM_CENTER
      || stored.anchorMode === ANCHOR_RIGHT_PALM_NORMAL
    ) {
      defaults.anchorMode = stored.anchorMode;
    }
  } catch (_) {
    // Ignore unavailable or malformed browser storage.
  }
  return defaults;
}

function persistSynthSettings() {
  if (typeof window === "undefined") return;
  const sizeMode = state.synth.objectSizeManual ? "manual" : "auto";
  const objectSize = Number(state.synth.objectSize);
  const settings = {
    objectType: state.synth.objectType,
    objectSizeMode: sizeMode,
    objectSize: Number.isFinite(objectSize) && objectSize > 0.02 ? objectSize : null,
    yawOnly: Boolean(state.synth.yawOnly),
    supportSurfaces: Boolean(state.synth.supportSurfaces),
    preContactMode: state.synth.preContactMode,
    postContactMode: state.synth.postContactMode,
    anchorMode: state.synth.anchorMode,
  };
  try {
    window.localStorage.setItem(SYNTH_SETTINGS_STORAGE_KEY, JSON.stringify(settings));
  } catch (_) {
    // Ignore blocked or unavailable browser storage.
  }
}

function defaultSynthState() {
  const persisted = readPersistentSynthSettings();
  const hasPersistedObjectType = persisted.objectType !== null;
  const hasPersistedYawOnly = persisted.yawOnly !== null;
  const hasPersistedSupport = persisted.supportSurfaces !== null;
  const hasPersistedPreMode = persisted.preContactMode !== null;
  const hasPersistedPostMode = persisted.postContactMode !== null;
  const hasPersistedAnchorMode = persisted.anchorMode !== null;
  const anchorMode = persisted.anchorMode || ANCHOR_PALM_CENTER;
  const rightPalmNormalAnchor = anchorMode === ANCHOR_RIGHT_PALM_NORMAL;
  const objectSizeManual = rightPalmNormalAnchor || persisted.objectSizeMode === "manual";
  return {
    contactStart: null,
    contactEnd: null,
    seedFrame: null,
    seedObjectPos: null,
    seedObjectQuat: null,
    objectSize: rightPalmNormalAnchor
      ? RIGHT_PALM_OBJECT_SIZE
      : objectSizeManual
        ? persisted.objectSize
        : null,
    objectSizeAuto: null,
    objectSizeManual,
    objectSizeMode: objectSizeManual ? "manual" : "auto",
    objectType: rightPalmNormalAnchor ? "box" : persisted.objectType || "box",
    objectTypeTouched: rightPalmNormalAnchor || hasPersistedObjectType,
    yawOnly: persisted.yawOnly ?? true,
    yawOnlyTouched: hasPersistedYawOnly,
    supportSurfaces: persisted.supportSurfaces ?? true,
    supportTouched: hasPersistedSupport,
    optimizeContactStage: false,
    optimizeContactTouched: false,
    contactOptimizationPreview: null,
    contactOptimizationMeta: null,
    optimizedContactStartObjectPos: null,
    optimizedContactStartObjectQuat: null,
    optimizedReleaseObjectPos: null,
    optimizedReleaseObjectQuat: null,
    preContactMode: persisted.preContactMode || "hold",
    preContactModeTouched: hasPersistedPreMode,
    preContactVelocity: null,
    preContactGravity: [...EARTH_GRAVITY],
    contactStartFrame: null,
    contactStartObjectPos: null,
    contactStartObjectQuat: null,
    postContactMode: persisted.postContactMode || "hold",
    postContactModeTouched: hasPersistedPostMode,
    anchorMode,
    anchorModeTouched: hasPersistedAnchorMode,
    releaseVelocity: null,
    releaseGravity: [...EARTH_GRAVITY],
    releaseFrame: null,
    releaseObjectPos: null,
    releaseObjectQuat: null,
  };
}

const state = {
  files: [],
  folders: [],
  selectedFolder: null,
  file: null,
  frame: 0,
  totalFrames: 1,
  fps: 50,
  frameData: null,
  trajectory: null,
  playing: false,
  timer: null,
  tab: "frame",
  jointSearch: "",
  jointGroup: "all",
  fileSearch: "",
  originalJointValues: null,
  frameRequest: 0,
  originFloorZ: null,
  model: null,
  modelReady: null,
  groundColorIndex: 0,
  groundingCorrection: false,
  groundingProfile: null,
  objectDrag: null,
  jointDrag: null,
  throwVelocityDrag: null,
  cropStart: 0,
  cropEnd: null,
  cropDrag: null,
  timelineDrag: null,
  synth: defaultSynthState(),
};

const $ = (id) => document.getElementById(id);

let scene;
let camera;
let renderer;
let controls;
let worldRoot;
let contentRoot;
let environmentRoot;
let bodyNodes = [];
let jointMarkers = [];
let objectMesh;
let tableMeshes = [];
let preContactVelocityArrow;
let preContactVelocityHandle;
let preContactTrajectoryLine;
let throwVelocityArrow;
let throwVelocityHandle;
let throwTrajectoryLine;
let raycaster;
let pointer = new THREE.Vector2();
let objectPlane = new THREE.Plane();

function fmt(value, digits = 3) {
  const n = Number(value);
  return Number.isFinite(n) ? n.toFixed(digits) : "--";
}

function showToast(message, error = false) {
  const toast = $("toast");
  toast.textContent = message;
  toast.classList.toggle("error", error);
  toast.classList.add("visible");
  clearTimeout(showToast.timer);
  showToast.timer = setTimeout(() => toast.classList.remove("visible"), 2600);
}

function setSaveProgress(prefix, status, message) {
  const progress = $(`${prefix}Progress`);
  const bar = $(`${prefix}ProgressBar`);
  const label = $(`${prefix}ProgressLabel`);
  const button = $(`${prefix}Button`);
  if (!progress || !bar || !label) return;
  clearTimeout(progress.hideTimer);
  progress.hidden = false;
  progress.dataset.state = status;
  label.textContent = message;
  bar.classList.toggle("indeterminate", status === "saving");
  if (status === "saving") {
    bar.style.width = "42%";
    if (button) button.disabled = true;
    return;
  }
  bar.style.width = status === "success" ? "100%" : "68%";
  if (button) button.disabled = false;
  progress.hideTimer = setTimeout(() => {
    progress.hidden = true;
    bar.classList.remove("indeterminate");
    bar.style.width = "0";
  }, status === "success" ? 2600 : 4200);
}

async function api(url, options = {}) {
  const response = await fetch(url, {
    ...options,
    headers: { "Content-Type": "application/json", ...(options.headers || {}) },
  });
  const payload = await response.json();
  if (!response.ok) throw new Error(payload.error || `Request failed (${response.status})`);
  return payload;
}

function init3D() {
  const mount = $("sceneCanvas");
  scene = new THREE.Scene();
  scene.background = new THREE.Color("#13171c");
  camera = new THREE.PerspectiveCamera(36, 1, 0.01, 100);
  camera.up.set(0, 0, 1);
  camera.position.set(2.2, -3.1, 1.75);
  renderer = new THREE.WebGLRenderer({ antialias: true, powerPreference: "high-performance" });
  renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
  renderer.outputColorSpace = THREE.SRGBColorSpace;
  renderer.shadowMap.enabled = true;
  renderer.shadowMap.type = THREE.PCFSoftShadowMap;
  mount.appendChild(renderer.domElement);

  controls = new OrbitControls(camera, renderer.domElement);
  controls.enableDamping = true;
  controls.dampingFactor = 0.08;
  controls.minDistance = 0.45;
  controls.maxDistance = 8;
  controls.target.set(0, 0, 0.72);

  scene.add(new THREE.HemisphereLight(0xdcecff, 0x26313b, 1.6));
  const keyLight = new THREE.DirectionalLight(0xffffff, 2.6);
  keyLight.position.set(2.5, -3.5, 5);
  keyLight.castShadow = true;
  keyLight.shadow.mapSize.set(1024, 1024);
  scene.add(keyLight);
  const fillLight = new THREE.DirectionalLight(0x9fc8e8, 0.8);
  fillLight.position.set(-3, 2, 2);
  scene.add(fillLight);

  worldRoot = new THREE.Group();
  contentRoot = new THREE.Group();
  environmentRoot = new THREE.Group();
  worldRoot.add(contentRoot, environmentRoot);
  scene.add(worldRoot);
  addEnvironment();
  raycaster = new THREE.Raycaster();

  renderer.domElement.addEventListener("pointerdown", onScenePointerDown);
  renderer.domElement.addEventListener("pointermove", onScenePointerMove);
  renderer.domElement.addEventListener("pointerup", onScenePointerUp);
  renderer.domElement.addEventListener("pointercancel", onScenePointerUp);
  renderer.domElement.addEventListener("wheel", (event) => event.stopPropagation(), { passive: true });
  resize3D();
  animate();
}

function addEnvironment() {
  const grid = new THREE.GridHelper(10, 40, 0x566473, 0x2b333d);
  grid.rotation.x = Math.PI / 2;
  grid.material.transparent = true;
  grid.material.opacity = 0.35;
  grid.name = "world_grid";
  environmentRoot.add(grid);
  const floor = new THREE.Mesh(
    new THREE.PlaneGeometry(10, 10),
    new THREE.MeshStandardMaterial({
      color: GROUND_COLOR_PRESETS[0].hex,
      roughness: 0.88,
      metalness: 0,
    }),
  );
  floor.name = "floor";
  floor.receiveShadow = true;
  environmentRoot.add(floor);
  environmentRoot.userData.grid = grid;
  environmentRoot.userData.floor = floor;
  applyGroundColor();
}

function applyGroundColor() {
  const preset = GROUND_COLOR_PRESETS[state.groundColorIndex] || GROUND_COLOR_PRESETS[0];
  const floor = environmentRoot?.userData.floor;
  if (floor?.material?.color) floor.material.color.setHex(preset.hex);
  const swatch = $("groundColorSwatch");
  if (swatch) swatch.style.backgroundColor = preset.css;
  const button = $("groundColorButton");
  if (button) {
    button.title = `Ground color: ${preset.label} · click to change`;
    button.setAttribute("aria-label", `Ground color: ${preset.label}. Click to change`);
  }
}

function cycleGroundColor() {
  state.groundColorIndex = (state.groundColorIndex + 1) % GROUND_COLOR_PRESETS.length;
  applyGroundColor();
  showToast(`Ground color: ${GROUND_COLOR_PRESETS[state.groundColorIndex].label}`);
}

function animate() {
  requestAnimationFrame(animate);
  controls?.update();
  renderer?.render(scene, camera);
}

function resize3D() {
  if (!renderer) return;
  const mount = $("sceneCanvas");
  const width = Math.max(1, mount.clientWidth);
  const height = Math.max(1, mount.clientHeight);
  renderer.setSize(width, height, false);
  camera.aspect = width / height;
  camera.updateProjectionMatrix();
}

function footGroundZ(data) {
  if (
    Array.isArray(data.ee_pos_w)
    && data.ee_pos_w.length >= 4
    && Array.isArray(data.ee_quat_w)
    && data.ee_quat_w.length >= 4
  ) {
    const contactHeights = [];
    for (const footIndex of [2, 3]) {
      const footPosition = new THREE.Vector3(...data.ee_pos_w[footIndex]);
      const footQuaternion = quatFromWxyz(data.ee_quat_w[footIndex]);
      for (const localPoint of FOOT_CONTACT_POINTS_LOCAL) {
        contactHeights.push(
          footPosition.clone()
            .add(new THREE.Vector3(...localPoint).applyQuaternion(footQuaternion))
            .z,
        );
      }
    }
    const groundZ = Math.min(...contactHeights);
    if (Number.isFinite(groundZ)) return groundZ;
  }
  if (Array.isArray(data.ee_pos_w) && data.ee_pos_w.length >= 4) {
    const footHeights = [2, 3].map((index) => Number(data.ee_pos_w[index][2]) - 0.03);
    const groundZ = Math.min(...footHeights);
    if (Number.isFinite(groundZ)) return groundZ;
  }
  return Number(data.base_pos_w?.[2] || 0);
}

function wristSpan(data) {
  if (!Array.isArray(data.ee_pos_w) || data.ee_pos_w.length < 2) return 0.28;
  const left = new THREE.Vector3(...data.ee_pos_w[0]);
  const right = new THREE.Vector3(...data.ee_pos_w[1]);
  const span = left.distanceTo(right);
  return Number.isFinite(span) && span > 0.02 ? span : 0.28;
}

function objectVisualSize(data) {
  if (data?.has_object) {
    const stored = Number(data.object_size);
    if (Number.isFinite(stored) && stored > 0.02) return stored;
  }
  const manualSize = Number(state.synth.objectSize);
  if (state.synth.objectSizeManual && Number.isFinite(manualSize) && manualSize > 0.02) {
    return manualSize;
  }
  if (
    synthSourceActive()
    && state.synth.seedFrame === state.synth.contactStart
    && Number.isFinite(Number(state.synth.objectSizeAuto))
    && Number(state.synth.objectSizeAuto) > 0.02
  ) {
    return Number(state.synth.objectSizeAuto);
  }
  const stored = Number(data?.object_size);
  if (Number.isFinite(stored) && stored > 0.02) return stored;
  return wristSpan(data);
}

function objectVisualType(data) {
  if (data?.has_object) {
    return data.object_type === "sphere" ? "sphere" : "box";
  }
  const value = state.synth.objectTypeTouched || synthSourceActive()
    ? state.synth.objectType
    : data?.object_type;
  return value === "sphere" ? "sphere" : "box";
}

function syncSynthConfigFromHoiData(data) {
  if (!data?.has_object) return;
  const storedSize = Number(data.object_size);
  state.synth.objectType = data.object_type === "sphere" ? "sphere" : "box";
  state.synth.objectTypeTouched = false;
  state.synth.objectSize = Number.isFinite(storedSize) && storedSize > 0.02
    ? storedSize
    : null;
  state.synth.objectSizeManual = Number.isFinite(storedSize) && storedSize > 0.02;
  state.synth.objectSizeMode = state.synth.objectSizeManual ? "manual" : "auto";
  state.synth.yawOnly = Boolean(data.object_yaw_only);
  state.synth.yawOnlyTouched = false;
  state.synth.supportSurfaces = data.support_enabled !== false;
  state.synth.supportTouched = false;
  state.synth.optimizeContactStage = Boolean(data.object_contact_optimized);
  state.synth.optimizeContactTouched = false;
  state.synth.anchorMode = data.object_anchor_mode || ANCHOR_PALM_CENTER;
  state.synth.anchorModeTouched = false;
  state.synth.preContactMode = data.object_pre_contact_mode || "hold";
  state.synth.preContactModeTouched = false;
  state.synth.postContactMode = data.object_post_contact_mode || "hold";
  state.synth.postContactModeTouched = false;
}

function supportVisualSize(data) {
  const stored = data?.support_size;
  if (Array.isArray(stored) && stored.length >= 3) {
    const values = stored.slice(0, 3).map(Number);
    if (values.every((value) => Number.isFinite(value) && value > 0)) return values;
  }
  if (synthSourceActive() || state.synth.objectSizeManual) {
    const size = objectVisualSize(data);
    return [size, size, 0.02];
  }
  return [0.72, 0.58, 0.12];
}

function supportBottomFromObject(pos, size, thickness) {
  return [pos[0], pos[1], pos[2] - size * 0.5 - thickness];
}

function setWorldOrigin(data) {
  const base = data.base_pos_w;
  if (state.originFloorZ === null) {
    state.originFloorZ = state.groundingCorrection && state.groundingProfile
      ? Number(state.groundingProfile.ground_z)
      : footGroundZ(data);
  }
  const floorZ = state.originFloorZ;
  worldRoot.position.set(-base[0], -base[1], -floorZ);
  const frameCorrection = state.groundingCorrection && state.groundingProfile
    ? Number(state.groundingProfile.correction_z?.[data.frame] || 0)
    : 0;
  contentRoot.position.z = Number.isFinite(frameCorrection) ? frameCorrection : 0;
  environmentRoot.userData.grid.position.z = floorZ + GROUND_GRID_OFFSET;
  environmentRoot.userData.floor.position.z = floorZ;
}

function currentGroundingCorrection(data) {
  if (!state.groundingCorrection || !state.groundingProfile || !data) return 0;
  const correction = Number(state.groundingProfile.correction_z?.[data.frame] || 0);
  return Number.isFinite(correction) ? correction : 0;
}

function fixedWorldPositionForContent(data, position) {
  const fixed = new THREE.Vector3(...position);
  fixed.z -= currentGroundingCorrection(data);
  return fixed;
}

function groundedAnchorPosition(data, position) {
  const grounded = [...position];
  grounded[2] += currentGroundingCorrection(data);
  return grounded;
}

function groundingFrameInfo(frame = state.frame) {
  const profile = state.groundingProfile;
  if (!profile) return null;
  const index = Math.max(0, Math.min(
    Number(profile.total_frames || 1) - 1,
    Math.round(Number(frame) || 0),
  ));
  const support = Boolean(profile.support_flags?.[index]);
  const airborne = Boolean(profile.airborne_flags?.[index]);
  const footHeights = profile.corrected_foot_height_w?.[index] || [];
  const correction = Number(profile.correction_z?.[index] || 0);
  return { support, airborne, footHeights, correction };
}

function quatFromWxyz(value) {
  return new THREE.Quaternion(value[1], value[2], value[3], value[0]);
}

function wxyzFromQuat(quat) {
  return [quat.w, quat.x, quat.y, quat.z];
}

function normalizedWxyz(value) {
  const q = quatFromWxyz(value || [1, 0, 0, 0]).normalize();
  return wxyzFromQuat(q);
}

function yawFromWxyz(value) {
  const [w, x, y, z] = normalizedWxyz(value);
  return Math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z));
}

function yawOnlyWxyz(value) {
  const yaw = yawFromWxyz(value);
  return [Math.cos(yaw * 0.5), 0, 0, Math.sin(yaw * 0.5)];
}

function yawQuatFromAngle(yaw) {
  return [Math.cos(yaw * 0.5), 0, 0, Math.sin(yaw * 0.5)];
}

function normalizeBoxQuatForMode(value) {
  return state.synth.yawOnly ? yawOnlyWxyz(value) : normalizedWxyz(value);
}

function multiplyWxyz(left, right) {
  return wxyzFromQuat(quatFromWxyz(left).multiply(quatFromWxyz(right)).normalize());
}

function invertWxyz(value) {
  return wxyzFromQuat(quatFromWxyz(value).invert().normalize());
}

function rotateVectorWxyz(quat, vector) {
  return new THREE.Vector3(...vector).applyQuaternion(quatFromWxyz(quat)).toArray();
}

function axisAngleWxyz(axisName, radians) {
  const axes = {
    x: new THREE.Vector3(1, 0, 0),
    y: new THREE.Vector3(0, 1, 0),
    z: new THREE.Vector3(0, 0, 1),
  };
  const quat = new THREE.Quaternion().setFromAxisAngle(axes[axisName] || axes.z, radians);
  return wxyzFromQuat(quat);
}

function averageWxyz(left, right) {
  const a = quatFromWxyz(left);
  const b = quatFromWxyz(right);
  if (a.dot(b) < 0) b.set(-b.x, -b.y, -b.z, -b.w);
  return wxyzFromQuat(new THREE.Quaternion(
    a.x + b.x,
    a.y + b.y,
    a.z + b.z,
    a.w + b.w,
  ).normalize());
}

function palmCenterPose(data) {
  const positions = Array.isArray(data.palm_pos_w) && data.palm_pos_w.length >= 2
    ? data.palm_pos_w
    : data.ee_pos_w;
  if (!Array.isArray(positions) || positions.length < 2) {
    return { pos: [0, 0, 0], quat: [1, 0, 0, 0] };
  }
  const leftPos = new THREE.Vector3(...positions[0]);
  const rightPos = new THREE.Vector3(...positions[1]);
  const pos = leftPos.add(rightPos).multiplyScalar(0.5).toArray();
  const quats = Array.isArray(data.palm_quat_w) && data.palm_quat_w.length >= 2
    ? data.palm_quat_w
    : data.ee_quat_w;
  const quat = Array.isArray(quats) && quats.length >= 2
    ? averageWxyz(quats[0], quats[1])
    : [1, 0, 0, 0];
  return { pos, quat };
}

function rightPalmNormalAnchorPose(data) {
  const positions = Array.isArray(data.palm_pos_w) && data.palm_pos_w.length >= 2
    ? data.palm_pos_w
    : data.ee_pos_w;
  if (!Array.isArray(positions) || positions.length < 2) {
    return { pos: [0, 0, 0], quat: [1, 0, 0, 0] };
  }
  const quats = Array.isArray(data.palm_quat_w) && data.palm_quat_w.length >= 2
    ? data.palm_quat_w
    : data.ee_quat_w;
  const quat = Array.isArray(quats) && quats.length >= 2
    ? normalizedWxyz(quats[1])
    : [1, 0, 0, 0];
  const normal = new THREE.Vector3(...RIGHT_PALM_NORMAL_LOCAL).applyQuaternion(quatFromWxyz(quat));
  const pos = new THREE.Vector3(...positions[1])
    .add(normal.multiplyScalar(RIGHT_PALM_NORMAL_OFFSET))
    .toArray();
  return { pos, quat };
}

function anchorPoseForSynthesis(data) {
  if (state.synth.anchorMode === ANCHOR_RIGHT_PALM_NORMAL) {
    const pose = rightPalmNormalAnchorPose(data);
    const pos = groundedAnchorPosition(data, pose.pos);
    return {
      pos,
      quat: state.synth.yawOnly ? yawOnlyWxyz(pose.quat) : pose.quat,
    };
  }
  const pose = palmCenterPose(data);
  const pos = groundedAnchorPosition(data, pose.pos);
  const positions = Array.isArray(data.palm_pos_w) && data.palm_pos_w.length >= 2
    ? data.palm_pos_w
    : data.ee_pos_w;
  if (state.synth.yawOnly && Array.isArray(positions) && positions.length >= 2) {
    const left = positions[0];
    const right = positions[1];
    const dx = Number(right[0]) - Number(left[0]);
    const dy = Number(right[1]) - Number(left[1]);
    if (Number.isFinite(dx) && Number.isFinite(dy) && Math.hypot(dx, dy) > 1e-5) {
      return { pos, quat: yawQuatFromAngle(Math.atan2(dy, dx)) };
    }
  }
  return {
    pos,
    quat: state.synth.yawOnly ? yawOnlyWxyz(pose.quat) : pose.quat,
  };
}

function synthSourceActive() {
  return Boolean(state.frameData && !state.frameData.has_object);
}

function invalidateSynthRelease(phase = "both") {
  state.synth.contactOptimizationPreview = null;
  state.synth.contactOptimizationMeta = null;
  state.synth.optimizeContactStage = false;
  state.synth.optimizedContactStartObjectPos = null;
  state.synth.optimizedContactStartObjectQuat = null;
  state.synth.optimizedReleaseObjectPos = null;
  state.synth.optimizedReleaseObjectQuat = null;
  if (phase === "both" || phase === "pre") {
    state.synth.contactStartFrame = null;
    state.synth.contactStartObjectPos = null;
    state.synth.contactStartObjectQuat = null;
    state.synth.preContactVelocity = null;
  }
  if (phase === "both" || phase === "post") {
    state.synth.releaseFrame = null;
    state.synth.releaseObjectPos = null;
    state.synth.releaseObjectQuat = null;
    state.synth.releaseVelocity = null;
  }
}

function parabolicObjectPosition(origin, velocity, gravity, elapsed) {
  return origin.map((value, index) =>
    Number(value)
    + Number(velocity[index] || 0) * elapsed
    + 0.5 * Number(gravity[index] || 0) * elapsed * elapsed,
  );
}

function invertedParabolicObjectPosition(contactPos, contactVelocity, gravity, elapsed) {
  return contactPos.map((value, index) =>
    Number(value)
    - Number(contactVelocity[index] || 0) * elapsed
    + 0.5 * Number(gravity[index] || 0) * elapsed * elapsed,
  );
}

function parabolicObjectVelocity(velocity, gravity, elapsed) {
  return velocity.map((value, index) =>
    Number(value) + Number(gravity[index] || 0) * elapsed,
  );
}

function invertedParabolicObjectVelocity(contactVelocity, gravity, elapsed) {
  return contactVelocity.map((value, index) =>
    Number(value) - Number(gravity[index] || 0) * elapsed,
  );
}

function hideTrajectoryVisuals() {
  [
    [preContactVelocityArrow, preContactVelocityHandle, preContactTrajectoryLine],
    [throwVelocityArrow, throwVelocityHandle, throwTrajectoryLine],
  ].forEach(([arrow, handle, line]) => {
    if (arrow) arrow.visible = false;
    if (handle) handle.visible = false;
    if (line) line.visible = false;
  });
}

function createTrajectoryVisual(color, arrowName, handleName, handleKind, lineName) {
  const arrow = new THREE.ArrowHelper(
    new THREE.Vector3(1, 0, 0),
    new THREE.Vector3(),
    0.1,
    color,
    0.025,
    0.012,
  );
  arrow.name = arrowName;
  arrow.visible = false;
  contentRoot.add(arrow);

  const handle = new THREE.Mesh(
    new THREE.SphereGeometry(0.055, 16, 10),
    new THREE.MeshStandardMaterial({
      color,
      emissive: color,
      emissiveIntensity: 0.35,
      roughness: 0.35,
      metalness: 0.1,
      depthTest: false,
      depthWrite: false,
    }),
  );
  handle.name = handleName;
  handle.userData.kind = handleKind;
  handle.visible = false;
  handle.renderOrder = 5;
  contentRoot.add(handle);

  const line = new THREE.Line(
    new THREE.BufferGeometry(),
    new THREE.LineDashedMaterial({
      color,
      dashSize: 0.045,
      gapSize: 0.028,
      transparent: true,
      opacity: 0.9,
      depthTest: false,
    }),
  );
  line.name = lineName;
  line.visible = false;
  line.renderOrder = 4;
  contentRoot.add(line);
  return { arrow, handle, line };
}

function ensureThrowVisualization() {
  if (
    preContactVelocityArrow
    && preContactVelocityHandle
    && preContactTrajectoryLine
    && throwVelocityArrow
    && throwVelocityHandle
    && throwTrajectoryLine
  ) return;
  const pre = createTrajectoryVisual(
    PRE_ARROW_COLOR,
    "pre_contact_velocity",
    "pre_contact_velocity_handle",
    "pre-contact-velocity-handle",
    "pre_contact_trajectory",
  );
  preContactVelocityArrow = pre.arrow;
  preContactVelocityHandle = pre.handle;
  preContactTrajectoryLine = pre.line;
  const post = createTrajectoryVisual(
    POST_ARROW_COLOR,
    "throw_velocity",
    "throw_velocity_handle",
    "throw-velocity-handle",
    "throw_trajectory",
  );
  throwVelocityArrow = post.arrow;
  throwVelocityHandle = post.handle;
  throwTrajectoryLine = post.line;
}

function updateTrajectoryVisualization(data) {
  if (!contentRoot || !data) return;
  ensureThrowVisualization();
  hideTrajectoryVisuals();
  const source = synthSourceActive();
  const bounds = synthBounds();
  const startFrame = source
    ? (bounds?.start ?? -1)
    : Number(data.object_contact_start_frame ?? -1);
  const endFrame = source
    ? (bounds?.end ?? -1)
    : Number(data.object_release_frame ?? -1);
  const fps = Math.max(Number(data.fps) || 50, 1e-6);
  const isPrePhase = startFrame >= 0 && data.frame < startFrame;
  const isPostPhase = endFrame >= 0 && data.frame > endFrame;
  let visual;
  let mode;
  let contactPos;
  let boundaryVelocity;
  let gravity;
  let currentPos;
  let currentVelocity;
  let points;

  if (isPrePhase) {
    mode = state.synth.preContactModeTouched
      ? state.synth.preContactMode
      : (source ? state.synth.preContactMode : data.object_pre_contact_mode);
    contactPos = source
      ? state.synth.contactStartObjectPos
      : data.object_contact_start_pos_w;
    boundaryVelocity = source
      ? state.synth.preContactVelocity
      : data.object_pre_contact_velocity_w;
    gravity = source ? state.synth.preContactGravity : (data.object_gravity_w || EARTH_GRAVITY);
    const elapsed = (startFrame - data.frame) / fps;
    if (mode === "parabolic" && Array.isArray(contactPos) && Array.isArray(boundaryVelocity)) {
      currentPos = invertedParabolicObjectPosition(contactPos, boundaryVelocity, gravity, elapsed);
      currentVelocity = invertedParabolicObjectVelocity(boundaryVelocity, gravity, elapsed);
      points = [];
      const stride = Math.max(1, Math.ceil((startFrame + 1) / 120));
      for (let frame = 0; frame <= startFrame; frame += stride) {
        const time = (startFrame - frame) / fps;
        points.push(fixedWorldPositionForContent(
          data,
          invertedParabolicObjectPosition(contactPos, boundaryVelocity, gravity, time),
        ));
      }
      const fixedContactPos = fixedWorldPositionForContent(data, contactPos);
      if (!points.length || points[points.length - 1].distanceTo(fixedContactPos) > 1e-5) {
        points.push(fixedContactPos);
      }
    }
    visual = {
      arrow: preContactVelocityArrow,
      handle: preContactVelocityHandle,
      line: preContactTrajectoryLine,
    };
  } else if (isPostPhase) {
    mode = state.synth.postContactModeTouched
      ? state.synth.postContactMode
      : (source ? state.synth.postContactMode : data.object_post_contact_mode);
    contactPos = source ? state.synth.releaseObjectPos : data.object_release_pos_w;
    boundaryVelocity = source ? state.synth.releaseVelocity : data.object_release_velocity_w;
    gravity = source ? state.synth.releaseGravity : (data.object_gravity_w || EARTH_GRAVITY);
    const elapsed = (data.frame - endFrame) / fps;
    if (mode === "parabolic" && Array.isArray(contactPos) && Array.isArray(boundaryVelocity)) {
      currentPos = parabolicObjectPosition(contactPos, boundaryVelocity, gravity, elapsed);
      currentVelocity = parabolicObjectVelocity(boundaryVelocity, gravity, elapsed);
      points = [];
      const futureFrames = Number(data.total_frames) - endFrame - 1;
      const stride = Math.max(1, Math.ceil((futureFrames + 1) / 120));
      for (let frame = endFrame; frame < Number(data.total_frames); frame += stride) {
        const time = (frame - endFrame) / fps;
        points.push(fixedWorldPositionForContent(
          data,
          parabolicObjectPosition(contactPos, boundaryVelocity, gravity, time),
        ));
      }
      const finalTime = futureFrames / fps;
      const finalPoint = fixedWorldPositionForContent(
        data,
        parabolicObjectPosition(contactPos, boundaryVelocity, gravity, finalTime),
      );
      if (!points.length || points[points.length - 1].distanceTo(finalPoint) > 1e-5) {
        points.push(finalPoint);
      }
    }
    visual = {
      arrow: throwVelocityArrow,
      handle: throwVelocityHandle,
      line: throwTrajectoryLine,
    };
  }

  if (
    mode !== "parabolic"
    || !visual
    || !Array.isArray(currentPos)
    || !Array.isArray(currentVelocity)
    || !points?.length
  ) return;

  const velocity = new THREE.Vector3(...currentVelocity);
  const speed = velocity.length();
  const arrowVector = speed > 1e-5
    ? velocity.clone()
    : new THREE.Vector3(0, 0, 0.08 / THROW_ARROW_TIME_SCALE);
  const fixedCurrentPos = fixedWorldPositionForContent(data, currentPos);
  const arrowTip = fixedCurrentPos.clone().add(
    arrowVector.clone().multiplyScalar(THROW_ARROW_TIME_SCALE),
  );
  visual.handle.position.copy(arrowTip);
  visual.handle.visible = source;
  if (speed > 1e-5) {
    visual.arrow.position.copy(fixedCurrentPos);
    visual.arrow.setDirection(velocity.normalize());
    visual.arrow.setLength(
      Math.max(0.08, speed * THROW_ARROW_TIME_SCALE),
      Math.max(0.02, Math.min(0.08, speed * 0.12)),
      Math.max(0.012, Math.min(0.045, speed * 0.06)),
    );
    visual.arrow.visible = true;
  }
  visual.line.geometry.dispose();
  visual.line.geometry = new THREE.BufferGeometry().setFromPoints(points);
  visual.line.computeLineDistances();
  visual.line.visible = points.length > 1;
}

function rgbaMaterial(rgba) {
  const color = new THREE.Color(rgba[0], rgba[1], rgba[2]);
  const material = new THREE.MeshStandardMaterial({
    color,
    roughness: 0.58,
    metalness: 0.18,
    side: THREE.DoubleSide,
  });
  if (rgba[3] < 0.999) {
    material.transparent = true;
    material.opacity = rgba[3];
    material.depthWrite = false;
  }
  return material;
}

function parseSTL(buffer) {
  const view = new DataView(buffer);
  const hasBinaryHeader = buffer.byteLength >= 84;
  const count = hasBinaryHeader ? view.getUint32(80, true) : 0;
  const binaryLength = 84 + count * 50;
  if (count > 0 && binaryLength <= buffer.byteLength) {
    const positions = new Float32Array(count * 9);
    let offset = 84;
    let out = 0;
    for (let triangle = 0; triangle < count; triangle += 1) {
      offset += 12;
      for (let vertex = 0; vertex < 3; vertex += 1) {
        positions[out++] = view.getFloat32(offset, true);
        positions[out++] = view.getFloat32(offset + 4, true);
        positions[out++] = view.getFloat32(offset + 8, true);
        offset += 12;
      }
      offset += 2;
    }
    return makeGeometry(positions);
  }

  const text = new TextDecoder().decode(buffer);
  const matches = [...text.matchAll(/vertex\s+([-+0-9.eE]+)\s+([-+0-9.eE]+)\s+([-+0-9.eE]+)/gi)];
  const positions = new Float32Array(matches.length * 3);
  matches.forEach((match, index) => {
    positions[index * 3] = Number(match[1]);
    positions[index * 3 + 1] = Number(match[2]);
    positions[index * 3 + 2] = Number(match[3]);
  });
  return makeGeometry(positions);
}

function makeGeometry(positions) {
  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute("position", new THREE.Float32BufferAttribute(positions, 3));
  geometry.computeVertexNormals();
  geometry.computeBoundingSphere();
  return geometry;
}

const meshCache = new Map();
async function loadMesh(file) {
  if (!meshCache.has(file)) {
    meshCache.set(file, fetch(`/g1-meshes/${encodeURIComponent(file)}`).then(async (response) => {
      if (!response.ok) throw new Error(`Unable to load ${file}`);
      return parseSTL(await response.arrayBuffer());
    }));
  }
  return meshCache.get(file);
}

async function loadModel() {
  const manifest = await api("/api/model");
  state.model = manifest;
  $("workspaceStatus").textContent = `Loading G1 mesh model · ${manifest.geoms.length} parts`;
  bodyNodes = manifest.bodies.map((body) => {
    const group = new THREE.Group();
    group.name = body.name || `body_${body.id}`;
    group.userData.bodyId = body.id;
    contentRoot.add(group);
    return group;
  });

  const files = [...new Set(manifest.geoms.map((geom) => geom.file))];
  let loaded = 0;
  const geometries = new Map();
  await Promise.all(files.map(async (file) => {
    geometries.set(file, await loadMesh(file));
    loaded += 1;
    $("workspaceStatus").textContent = `Loading G1 mesh model · ${loaded}/${files.length}`;
  }));

  manifest.geoms.forEach((geom) => {
    // MuJoCo centers/reorients raw STL vertices when compiling <mesh>.
    // Bake that raw-STL -> compiled-mesh transform before applying the geom
    // transform, matching the native MJCF renderer.
    const meshGeometry = geometries.get(geom.file).clone();
    const meshQuat = quatFromWxyz(geom.mesh_quat).invert();
    const meshPos = new THREE.Vector3(...geom.mesh_pos)
      .multiplyScalar(-1)
      .applyQuaternion(meshQuat);
    const meshScale = new THREE.Vector3(...geom.mesh_scale);
    meshGeometry.applyMatrix4(
      new THREE.Matrix4().compose(meshPos, meshQuat, meshScale),
    );
    meshGeometry.computeVertexNormals();
    meshGeometry.computeBoundingSphere();
    const mesh = new THREE.Mesh(meshGeometry, rgbaMaterial(geom.rgba));
    mesh.position.fromArray(geom.pos);
    mesh.quaternion.copy(quatFromWxyz(geom.quat));
    mesh.castShadow = true;
    mesh.receiveShadow = true;
    mesh.userData.meshName = geom.mesh;
    mesh.userData.bodyId = geom.body_id;
    bodyNodes[geom.body_id].add(mesh);
  });

  jointMarkers = manifest.joints.map((joint) => {
    const material = new THREE.MeshStandardMaterial({
      color: joint.name.startsWith("right_") ? 0xf2f4f7 : 0x63d6db,
      emissive: joint.name.startsWith("right_") ? 0x3a424b : 0x164d53,
      emissiveIntensity: 0.75,
      roughness: 0.45,
      metalness: 0.1,
    });
    const marker = new THREE.Mesh(new THREE.SphereGeometry(0.027, 12, 8), material);
    marker.userData.joint = joint;
    marker.renderOrder = 3;
    contentRoot.add(marker);
    return marker;
  });

  objectMesh = new THREE.Mesh(
    new THREE.BoxGeometry(1, 1, 1),
    new THREE.MeshStandardMaterial({
      color: 0xeea35b,
      roughness: 0.5,
      metalness: 0.05,
      transparent: true,
      opacity: 0.92,
    }),
  );
  objectMesh.castShadow = true;
  objectMesh.receiveShadow = true;
  objectMesh.userData.kind = "object";
  contentRoot.add(objectMesh);

  const tableMaterial = new THREE.MeshStandardMaterial({
    color: 0x657487,
    roughness: 0.82,
    metalness: 0.05,
    transparent: true,
    opacity: 0.56,
  });
  tableMeshes = [0, 1].map((index) => {
    const table = new THREE.Mesh(new THREE.BoxGeometry(1, 1, 1), tableMaterial.clone());
    table.material.opacity = index === 0 ? 0.5 : 0.34;
    table.receiveShadow = true;
    table.userData.kind = "table";
    contentRoot.add(table);
    return table;
  });
  setView("orbit");
  if (state.frameData) update3DFrame(state.frameData);
  fitView();
  $("workspaceStatus").textContent = "G1 MuJoCo mesh model ready";
}

function update3DFrame(data) {
  if (!state.model || !data || !bodyNodes.length) return;
  setWorldOrigin(data);
  data.body_pose_w.forEach((pose, bodyId) => {
    const body = bodyNodes[bodyId];
    if (!body) return;
    body.position.set(pose[0], pose[1], pose[2]);
    body.quaternion.copy(quatFromWxyz(pose.slice(3, 7)));
  });
  state.model.joints.forEach((joint, index) => {
    const marker = jointMarkers[index];
    const body = bodyNodes[joint.body_id];
    if (!marker || !body) return;
    marker.position.copy(body.position);
    marker.visible = true;
  });
  const objectType = objectVisualType(data);
  const geometryType = objectMesh.userData.objectType;
  if (geometryType !== objectType) {
    objectMesh.geometry.dispose();
    objectMesh.geometry = objectType === "sphere"
      ? new THREE.SphereGeometry(0.5, 32, 20)
      : new THREE.BoxGeometry(1, 1, 1);
    objectMesh.userData.objectType = objectType;
  }
  objectMesh.position.copy(fixedWorldPositionForContent(data, data.object_pos_w));
  objectMesh.quaternion.copy(quatFromWxyz(data.object_quat_w));
  objectMesh.scale.setScalar(objectVisualSize(data));
  const tableData = supportSurfacePositions(data);
  const tableSize = supportVisualSize(data);
  tableMeshes.forEach((table, index) => {
    const position = tableData[index];
    table.visible = Boolean(position);
    if (!position) return;
    table.scale.set(tableSize[0], tableSize[1], tableSize[2]);
    table.position.copy(fixedWorldPositionForContent(data, [
      position[0],
      position[1],
      position[2] + tableSize[2] * 0.5,
    ]));
  });
  updateTrajectoryVisualization(data);
}

function setView(view) {
  if (!camera || !controls) return;
  const target = new THREE.Vector3(0, 0, 0.72);
  controls.target.copy(target);
  if (view === "top") {
    camera.position.set(0.01, 0.01, 3.6);
    camera.up.set(0, 1, 0);
  } else if (view === "front") {
    camera.position.set(0, -3.4, 0.78);
    camera.up.set(0, 0, 1);
  } else {
    camera.position.set(2.2, -3.1, 1.75);
    camera.up.set(0, 0, 1);
  }
  camera.lookAt(target);
  controls.update();
}

function fitView() {
  if (!contentRoot || !contentRoot.children.length) return;
  const box = new THREE.Box3().setFromObject(contentRoot);
  const center = box.getCenter(new THREE.Vector3());
  const size = box.getSize(new THREE.Vector3());
  const radius = Math.max(size.x, size.y, size.z) * 0.95;
  const distance = Math.max(2.2, radius / Math.tan((camera.fov * Math.PI) / 360));
  controls.target.copy(center);
  camera.position.copy(center).add(new THREE.Vector3(distance * 0.9, -distance, distance * 0.62));
  camera.lookAt(center);
  controls.update();
}

function readDatasetFolderPreference() {
  try {
    return window.localStorage?.getItem(DATASET_FOLDER_STORAGE_KEY) || null;
  } catch (_) {
    return null;
  }
}

function persistDatasetFolder() {
  try {
    window.localStorage?.setItem(
      DATASET_FOLDER_STORAGE_KEY,
      state.selectedFolder || "",
    );
  } catch (_) {
    // Ignore blocked or unavailable browser storage.
  }
}

function renderFolderSelect() {
  const select = $("folderSelect");
  if (!select) return;
  select.innerHTML = state.folders.map((folder) => {
    const value = folder.folder || "";
    const label = folder.folder
      ? `${folder.name}  ·  ${folder.folder}`
      : "All folders";
    return `<option value="${escapeHtml(value)}">${escapeHtml(label)} (${folder.files})</option>`;
  }).join("");
  select.value = state.selectedFolder || "";
}

function escapeHtml(value) {
  return String(value)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

async function loadFolders() {
  const payload = await api("/api/folders");
  state.folders = payload.folders || [];
  const available = new Set(state.folders.map((folder) => folder.folder || ""));
  const stored = readDatasetFolderPreference();
  if (state.selectedFolder !== null && available.has(state.selectedFolder)) {
    // Keep the user's current folder during a refresh.
  } else if (stored !== null && available.has(stored)) {
    state.selectedFolder = stored;
  } else {
    const preferred = state.folders.find((folder) => folder.folder === "robot_only");
    const populated = state.folders
      .filter((folder) => folder.folder && folder.files > 0)
      .sort((a, b) => b.files - a.files)[0];
    state.selectedFolder = preferred?.folder || populated?.folder || "";
  }
  renderFolderSelect();
}

function clearDatasetSelection() {
  stopPlayback();
  state.file = null;
  state.frameData = null;
  state.trajectory = null;
  state.totalFrames = 1;
  state.frame = 0;
  state.originFloorZ = null;
  state.groundingProfile = null;
  state.groundingCorrection = false;
  const groundingToggle = $("groundingCorrectionToggle");
  if (groundingToggle) {
    groundingToggle.checked = false;
    groundingToggle.disabled = true;
  }
  $("trajectoryTitle").textContent = "Select a dataset";
  $("workspaceStatus").textContent = "Select a dataset folder";
  renderFileList();
}

async function loadFiles() {
  if (!state.folders.length) await loadFolders();
  const query = state.selectedFolder
    ? `?folder=${encodeURIComponent(state.selectedFolder)}`
    : "";
  const payload = await api(`/api/files${query}`);
  state.files = payload.files;
  renderFileList();
  const currentVisible = state.files.some((file) => file.file === state.file);
  if (!currentVisible) {
    if (state.files.length) {
      await selectFile(state.files[0].file);
    } else {
      clearDatasetSelection();
    }
  }
}

async function selectDatasetFolder(folder) {
  state.selectedFolder = String(folder || "");
  persistDatasetFolder();
  renderFolderSelect();
  await loadFiles();
}

async function createDatasetFolder() {
  const parent = state.selectedFolder || "";
  const name = window.prompt(
    parent
      ? `Create a new folder inside "${parent}"`
      : "Create a new folder inside data",
  );
  if (!name?.trim()) return;
  const result = await api("/api/folders", {
    method: "POST",
    body: JSON.stringify({ parent, name: name.trim() }),
  });
  state.selectedFolder = result.folder;
  persistDatasetFolder();
  await loadFolders();
  await loadFiles();
  showToast(`Created folder ${result.folder}`);
}

function renderFileList() {
  const query = state.fileSearch.toLowerCase();
  const list = $("fileList");
  list.innerHTML = "";
  const files = state.files.filter((file) => file.file.toLowerCase().includes(query));
  if (!files.length) {
    list.innerHTML = `<div class="empty-state">No NPZ files found</div>`;
    return;
  }
  files.forEach((file) => {
    const hasObject = file.keys.includes("object_pos_w") && file.keys.includes("object_quat_w");
    const kind = hasObject ? "HOI" : "robot-only";
    const item = document.createElement("div");
    item.className = `file-item${file.file === state.file ? " active" : ""}`;
    item.innerHTML = `
      <div class="file-name" title="${file.file}">${file.name}</div>
      <div class="file-meta"><span>${kind}</span><span>${file.frames.toLocaleString()} frames</span><span>${fmt(file.duration, 1)} s</span>${file.dirty ? '<span class="file-dirty">edited</span>' : ""}</div>
    `;
    item.addEventListener("click", () => selectFile(file.file));
    list.appendChild(item);
  });
}

async function selectFile(file) {
  stopPlayback();
  state.file = file;
  state.originalJointValues = null;
  state.synth = defaultSynthState();
  state.cropStart = 0;
  state.cropEnd = null;
  state.cropDrag = null;
  state.originFloorZ = null;
  state.groundingCorrection = DEFAULT_GROUNDING_CORRECTION;
  state.groundingProfile = null;
  const groundingToggle = $("groundingCorrectionToggle");
  if (groundingToggle) {
    groundingToggle.checked = DEFAULT_GROUNDING_CORRECTION;
    groundingToggle.disabled = true;
  }
  renderFileList();
  $("workspaceStatus").textContent = "Reading trajectory";
  try {
    state.trajectory = await api(`/api/trajectory?file=${encodeURIComponent(file)}&limit=900`);
    try {
      state.groundingProfile = await api(`/api/grounding?file=${encodeURIComponent(file)}`);
      state.groundingCorrection = DEFAULT_GROUNDING_CORRECTION;
    } catch (error) {
      state.groundingProfile = null;
      state.groundingCorrection = false;
      showToast(`Grounding profile unavailable: ${error.message}`, true);
    }
    state.totalFrames = state.trajectory.total_frames;
    state.fps = state.trajectory.fps;
    state.cropEnd = Math.max(0, state.totalFrames - 1);
    state.frame = 0;
    await loadFrame(0);
    $("trajectoryTitle").textContent = file.split("/").pop();
    $("workspaceStatus").textContent = state.model
      ? `${state.totalFrames.toLocaleString()} frames · G1 mesh ready`
      : `${state.totalFrames.toLocaleString()} frames · loading G1 mesh`;
  } catch (error) {
    showToast(error.message, true);
    $("workspaceStatus").textContent = "Unable to load trajectory";
  }
}

async function loadFrame(frame) {
  if (!state.file) return;
  frame = Math.max(0, Math.min(state.totalFrames - 1, Math.round(Number(frame) || 0)));
  state.frame = frame;
  const requestId = ++state.frameRequest;
  try {
    const loaded = await api(`/api/frame?file=${encodeURIComponent(state.file)}&frame=${frame}`);
    if (requestId !== state.frameRequest) return;
    if (loaded.has_object) {
      syncSynthConfigFromHoiData(loaded);
    } else {
      if (!state.synth.objectTypeTouched) {
        state.synth.objectType = loaded.object_type || "box";
      }
      if (!state.synth.preContactModeTouched) {
        state.synth.preContactMode = loaded.object_pre_contact_mode || "hold";
      }
      if (!state.synth.postContactModeTouched) {
        state.synth.postContactMode = loaded.object_post_contact_mode || "hold";
      }
    }
    await ensureSynthBoundaryPoses(loaded);
    if (requestId !== state.frameRequest) return;
    applySynthPreview(loaded);
    applyContactOptimizationPreview(loaded);
    state.frameData = loaded;
    state.originalJointValues = [...loaded.joint_pos];
    renderAll();
  } catch (error) {
    showToast(error.message, true);
  }
}

function synthBounds() {
  if (state.synth.contactStart === null || state.synth.contactEnd === null) return null;
  const start = Math.max(0, Math.min(state.totalFrames - 1, Math.round(state.synth.contactStart)));
  const end = Math.max(0, Math.min(state.totalFrames - 1, Math.round(state.synth.contactEnd)));
  return start <= end ? { start, end } : { start: end, end: start };
}

function cropBounds() {
  const last = Math.max(0, state.totalFrames - 1);
  const start = Math.max(0, Math.min(last, Math.round(Number(state.cropStart) || 0)));
  const end = Math.max(start, Math.min(
    last,
    Math.round(state.cropEnd === null ? last : state.cropEnd),
  ));
  return { start, end };
}

function transformRelativePose(data) {
  if (!state.synth.relativePos || !state.synth.relativeQuat) return null;
  const center = anchorPoseForSynthesis(data);
  const offset = rotateVectorWxyz(center.quat, state.synth.relativePos);
  const quat = multiplyWxyz(center.quat, state.synth.relativeQuat);
  return {
    pos: center.pos.map((value, index) => value + offset[index]),
    quat: normalizeBoxQuatForMode(quat),
  };
}

async function ensureSynthBoundaryPoses(data) {
  if (!data || data.has_object) return;
  const bounds = synthBounds();
  if (!bounds || !state.synth.relativePos || !state.synth.relativeQuat) return;
  if (
    state.synth.contactStartFrame === bounds.start
    && state.synth.contactStartObjectPos
    && state.synth.contactStartObjectQuat
    && state.synth.releaseFrame === bounds.end
    && state.synth.releaseObjectPos
    && state.synth.releaseObjectQuat
    && state.synth.preContactVelocity
    && state.synth.releaseVelocity
  ) return;
  const file = state.file;
  const startRequest = data.frame === bounds.start
    ? Promise.resolve(data)
    : api(`/api/frame?file=${encodeURIComponent(file)}&frame=${bounds.start}`);
  const endRequest = data.frame === bounds.end
    ? Promise.resolve(data)
    : api(`/api/frame?file=${encodeURIComponent(file)}&frame=${bounds.end}`);
  const [startData, endData] = await Promise.all([startRequest, endRequest]);
  if (state.file !== file) return;
  const startPose = transformRelativePose(startData);
  const endPose = transformRelativePose(endData);
  if (!startPose || !endPose) return;
  state.synth.contactStartFrame = bounds.start;
  state.synth.contactStartObjectPos = [...startPose.pos];
  state.synth.contactStartObjectQuat = normalizedWxyz(startPose.quat);
  state.synth.preContactVelocity = Array.isArray(startData.object_pre_contact_velocity_w)
    ? startData.object_pre_contact_velocity_w.map(Number)
    : [0, 0, 0];
  state.synth.preContactGravity = Array.isArray(startData.object_gravity_w)
    ? startData.object_gravity_w.map(Number)
    : [...EARTH_GRAVITY];
  state.synth.releaseFrame = bounds.end;
  state.synth.releaseObjectPos = [...endPose.pos];
  state.synth.releaseObjectQuat = normalizedWxyz(endPose.quat);
  state.synth.releaseVelocity = Array.isArray(endData.object_release_velocity_w)
    ? endData.object_release_velocity_w.map(Number)
    : [0, 0, 0];
  state.synth.releaseGravity = Array.isArray(endData.object_gravity_w)
    ? endData.object_gravity_w.map(Number)
    : [...EARTH_GRAVITY];
}

function applySynthPreview(data) {
  if (!data || data.has_object) return;
  const bounds = synthBounds();
  let pose = null;
  if (bounds && state.synth.relativePos && state.synth.relativeQuat) {
    if (
      data.frame < bounds.start
      && state.synth.preContactMode === "parabolic"
      && state.synth.contactStartObjectPos
      && state.synth.preContactVelocity
    ) {
      const elapsed = (bounds.start - data.frame) / Math.max(state.fps, 1e-6);
      pose = {
        pos: invertedParabolicObjectPosition(
          state.synth.optimizedContactStartObjectPos
            || state.synth.contactStartObjectPos,
          state.synth.preContactVelocity,
          state.synth.preContactGravity || EARTH_GRAVITY,
          elapsed,
        ),
        quat: state.synth.optimizedContactStartObjectQuat
          || state.synth.contactStartObjectQuat,
      };
    } else if (data.frame >= bounds.start && data.frame <= bounds.end) {
      pose = transformRelativePose(data);
    } else if (data.frame > bounds.end && state.synth.releaseObjectPos && state.synth.releaseObjectQuat) {
      const elapsed = (data.frame - bounds.end) / Math.max(state.fps, 1e-6);
      pose = {
        pos: state.synth.postContactMode === "parabolic"
          ? parabolicObjectPosition(
            state.synth.optimizedReleaseObjectPos
              || state.synth.releaseObjectPos,
            state.synth.releaseVelocity || [0, 0, 0],
            state.synth.releaseGravity || EARTH_GRAVITY,
            elapsed,
          )
          : (state.synth.optimizedReleaseObjectPos
            || state.synth.releaseObjectPos),
        quat: state.synth.optimizedReleaseObjectQuat
          || state.synth.releaseObjectQuat,
      };
    }
  }
  if (!pose && state.synth.seedObjectPos && state.synth.seedObjectQuat) {
    pose = { pos: state.synth.seedObjectPos, quat: state.synth.seedObjectQuat };
  }
  if (!pose) {
    pose = anchorPoseForSynthesis(data);
  }
  data.object_pos_w = [...pose.pos];
  data.object_quat_w = normalizeBoxQuatForMode(pose.quat);
}

function applyContactOptimizationPreview(data) {
  const preview = state.synth.contactOptimizationPreview;
  if (!preview || !Array.isArray(preview.frames) || !data) return;
  const index = preview.frames.indexOf(data.frame);
  if (index < 0) return;
  if (Array.isArray(preview.joint_pos?.[index])) {
    data.joint_pos = [...preview.joint_pos[index]];
  }
  if (Array.isArray(preview.body_pose_w?.[index])) {
    data.body_pose_w = preview.body_pose_w[index].map((pose) => [...pose]);
  }
  if (Array.isArray(preview.palm_pos_w?.[index])) {
    data.palm_pos_w = preview.palm_pos_w[index].map((value) => [...value]);
  }
  if (Array.isArray(preview.palm_quat_w?.[index])) {
    data.palm_quat_w = preview.palm_quat_w[index].map((value) => [...value]);
  }
  if (Array.isArray(preview.ee_pos_w?.[index])) {
    data.ee_pos_w = preview.ee_pos_w[index].map((value) => [...value]);
  }
  if (Array.isArray(preview.ee_quat_w?.[index])) {
    data.ee_quat_w = preview.ee_quat_w[index].map((value) => [...value]);
  }
  if (Array.isArray(preview.object_pos_w?.[index])) {
    data.object_pos_w = [...preview.object_pos_w[index]];
  }
  if (Array.isArray(preview.object_quat_w?.[index])) {
    data.object_quat_w = [...preview.object_quat_w[index]];
  }
  data.object_contact_optimized = true;
  data.contact_optimization_preview = true;
}

function supportSurfacePositions(data) {
  if (!data) return [null, null];
  if (synthSourceActive()) {
    if (!state.synth.supportSurfaces) return [null, null];
    const [size, , thickness] = supportVisualSize(data);
    const seedPos = state.synth.seedObjectPos || data.object_pos_w;
    const releasePos = state.synth.optimizedReleaseObjectPos
      || state.synth.releaseObjectPos
      || state.synth.seedObjectPos
      || data.object_pos_w;
    return [
      supportBottomFromObject(seedPos, size, thickness),
      supportBottomFromObject(releasePos, size, thickness),
    ];
  }
  const supportEnabled = state.synth.supportTouched
    ? state.synth.supportSurfaces
    : data.support_enabled !== false;
  if (!supportEnabled) return [null, null];
  return [data.table1_pos_w, data.table2_pos_w];
}

function rememberSynthSeed() {
  if (!state.frameData) return;
  if (state.synth.contactStart === null) state.synth.contactStart = state.frame;
  if (state.synth.contactEnd === null) state.synth.contactEnd = state.frame;
  if (state.frame !== state.synth.contactStart) return;
  state.frameData.object_quat_w = normalizeBoxQuatForMode(state.frameData.object_quat_w);
  const center = anchorPoseForSynthesis(state.frameData);
  const inverse = invertWxyz(center.quat);
  const relativeWorld = state.frameData.object_pos_w.map((value, index) => value - center.pos[index]);
  state.synth.seedFrame = state.frame;
  state.synth.seedObjectPos = [...state.frameData.object_pos_w];
  state.synth.seedObjectQuat = normalizeBoxQuatForMode(state.frameData.object_quat_w);
  state.synth.objectSizeAuto = wristSpan(state.frameData);
  if (!state.synth.objectSizeManual || !state.synth.objectSize) {
    state.synth.objectSize = state.synth.objectSizeAuto;
  }
  state.synth.relativePos = rotateVectorWxyz(inverse, relativeWorld);
  state.synth.relativeQuat = multiplyWxyz(inverse, state.synth.seedObjectQuat);
  invalidateSynthRelease();
}

async function refreshSynthSeedForMode() {
  if (!state.frameData || !synthSourceActive()) return;
  if (
    state.synth.contactStart === null
    || !state.synth.seedObjectPos
    || !state.synth.seedObjectQuat
  ) return;
  const file = state.file;
  const start = state.synth.contactStart;
  const startData = state.frame === start
    ? state.frameData
    : await api(`/api/frame?file=${encodeURIComponent(file)}&frame=${start}`);
  if (state.file !== file) return;
  const center = anchorPoseForSynthesis(startData);
  const inverse = invertWxyz(center.quat);
  const seedQuat = normalizeBoxQuatForMode(state.synth.seedObjectQuat);
  const relativeWorld = state.synth.seedObjectPos.map((value, index) => value - center.pos[index]);
  state.synth.seedObjectQuat = seedQuat;
  state.synth.relativePos = rotateVectorWxyz(inverse, relativeWorld);
  state.synth.relativeQuat = multiplyWxyz(inverse, seedQuat);
  invalidateSynthRelease();
}

function alignObjectToPalms() {
  if (!state.frameData) return;
  const pose = anchorPoseForSynthesis(state.frameData);
  if (state.synth.contactStart === null) state.synth.contactStart = state.frame;
  if (state.synth.contactEnd === null || state.synth.contactEnd < state.synth.contactStart) {
    state.synth.contactEnd = state.synth.contactStart;
  }
  state.frameData.object_pos_w = pose.pos;
  state.frameData.object_quat_w = pose.quat;
  if (state.synth.anchorMode === ANCHOR_RIGHT_PALM_NORMAL) {
    state.synth.objectType = "box";
    state.synth.objectTypeTouched = true;
    state.synth.objectSizeAuto = RIGHT_PALM_OBJECT_SIZE;
    state.synth.objectSize = RIGHT_PALM_OBJECT_SIZE;
    state.synth.objectSizeManual = true;
    state.synth.objectSizeMode = "manual";
    state.frameData.object_type = "box";
    state.frameData.object_size = RIGHT_PALM_OBJECT_SIZE;
  } else {
    state.synth.objectSizeAuto = wristSpan(state.frameData);
    if (!state.synth.objectSizeManual || !state.synth.objectSize) {
      state.synth.objectSize = state.synth.objectSizeAuto;
    }
  }
  persistSynthSettings();
  rememberSynthSeed();
  renderObjectFields();
  update3DFrame(state.frameData);
  renderSynthesisPanel();
  const anchorLabel = state.synth.anchorMode === ANCHOR_RIGHT_PALM_NORMAL
    ? "right palm normal +5cm"
    : "palm center";
  showToast(`Object aligned to ${anchorLabel} at frame ${state.frame}`);
}

function renderSynthesisPanel() {
  if (!$("contactStartInput")) return;
  const start = state.synth.contactStart ?? state.frame;
  const end = state.synth.contactEnd ?? state.frame;
  $("contactStartInput").max = state.totalFrames - 1;
  $("contactEndInput").max = state.totalFrames - 1;
  $("contactStartInput").value = start;
  $("contactEndInput").value = end;
  const preModeSelect = $("preContactMode");
  if (preModeSelect) preModeSelect.value = state.synth.preContactMode;
  const modeSelect = $("postContactMode");
  if (modeSelect) modeSelect.value = state.synth.postContactMode;
  const bounds = synthBounds();
  const optimizeButton = $("optimizeContactButton");
  if (optimizeButton) {
    optimizeButton.disabled = Boolean(state.frameData?.has_object)
      || !bounds
      || !state.synth.seedObjectPos;
    optimizeButton.textContent = state.synth.contactOptimizationPreview
      ? "Re-optimize contact"
      : "Optimize contact stage";
  }
  const restoreButton = $("restoreContactButton");
  if (restoreButton) {
    restoreButton.disabled = !state.synth.contactOptimizationPreview;
  }
  const saveButton = $("synthesizeButton");
  const loadedHOI = Boolean(state.frameData?.has_object);
  if (saveButton) {
    saveButton.dataset.action = loadedHOI ? "crop" : "synthesize";
    saveButton.title = loadedHOI
      ? "Save the selected time range as a cropped HOI copy"
      : "Save synthesized full HOI NPZ copy";
    saveButton.innerHTML = loadedHOI
      ? '<span class="button-icon">⤢</span> Crop selected HOI'
      : '<span class="button-icon">⇩</span> Save synthesized HOI';
  }
  const mode = loadedHOI ? "Full HOI loaded" : "Robot-only source";
  const size = state.synth.objectSize || (state.frameData ? wristSpan(state.frameData) : 0);
  const seed = state.synth.seedFrame === state.synth.contactStart
    ? `seed frame <strong>${state.synth.seedFrame}</strong>`
    : "seed not set";
  const releaseSpeed = state.synth.releaseVelocity
    ? Math.hypot(...state.synth.releaseVelocity)
    : 0;
  const preContactSpeed = state.synth.preContactVelocity
    ? Math.hypot(...state.synth.preContactVelocity)
    : 0;
  const preContactLabel = state.synth.preContactMode === "parabolic"
    ? "inverted parabolic"
    : "hold still";
  const postContactLabel = state.synth.postContactMode === "parabolic"
    ? "parabolic throw"
    : "hold still";
  const objectLabel = state.synth.objectType === "sphere" ? "sphere diameter" : "box edge";
  const objectSizeModeLabel = state.synth.objectSizeManual ? "manual" : "auto";
  const anchorLabel = state.synth.anchorMode === ANCHOR_RIGHT_PALM_NORMAL
    ? "right palm normal +5cm"
    : "palm midpoint";
  $("synthSummary").innerHTML = `
    <div><strong>${mode}</strong></div>
    <div>contact ${bounds ? `<strong>${bounds.start}</strong> to <strong>${bounds.end}</strong>` : "not marked"}</div>
    <div>anchor <strong>${anchorLabel}</strong></div>
    <div>${objectLabel} <strong>${fmt(size, 3)} m</strong> · ${objectSizeModeLabel} · ${seed}</div>
    <div>horizontal yaw lock <strong>${state.synth.yawOnly ? "on" : "off"}</strong></div>
    <div>support surfaces <strong>${state.synth.supportSurfaces ? "on" : "off"}</strong></div>
    <div>contact optimization <strong>${
      state.synth.contactOptimizationPreview
        ? "preview active"
        : state.synth.optimizeContactStage
          ? "stored"
          : "off"
    }</strong></div>
    <div>pre-contact <strong>${preContactLabel}</strong>${state.synth.preContactMode === "parabolic" ? ` · speed <strong>${fmt(preContactSpeed, 2)} m/s</strong>` : ""}</div>
    <div>post-contact <strong>${postContactLabel}</strong>${state.synth.postContactMode === "parabolic" ? ` · speed <strong>${fmt(releaseSpeed, 2)} m/s</strong>` : ""}</div>
  `;
  drawTimeline();
}

async function markContactStart() {
  state.synth.contactStart = state.frame;
  if (state.synth.contactEnd === null || state.synth.contactEnd < state.frame) {
    state.synth.contactEnd = state.frame;
  }
  alignObjectToPalms();
  await ensureSynthBoundaryPoses(state.frameData);
  if (synthSourceActive()) applySynthPreview(state.frameData);
  renderSynthesisPanel();
  update3DFrame(state.frameData);
}

async function markContactEnd() {
  state.synth.contactEnd = state.frame;
  if (state.synth.contactStart === null || state.synth.contactStart > state.frame) {
    state.synth.contactStart = state.frame;
  }
  invalidateSynthRelease();
  await ensureSynthBoundaryPoses(state.frameData);
  renderSynthesisPanel();
  update3DFrame(state.frameData);
  drawTimeline();
  showToast(`Contact end set to frame ${state.frame}`);
}

async function goToContactStart() {
  if (state.synth.contactStart === null) {
    state.synth.contactStart = state.frame;
  }
  await loadFrame(state.synth.contactStart);
}

async function synthesizeHOI() {
  if (!state.file || !state.frameData) return;
  const synthesizeButton = $("synthesizeButton");
  if (synthesizeButton?.disabled) return;
  if (state.synth.contactStart === null || state.synth.contactEnd === null) {
    showToast("Mark contact start and end first", true);
    return;
  }
  if (state.synth.seedFrame !== state.synth.contactStart) {
    await goToContactStart();
    alignObjectToPalms();
  }
  const bounds = synthBounds();
  const defaultName = `${state.file.split("/").pop().replace(/\.npz$/, "")}_synth_hoi.npz`;
  const name = window.prompt("Save synthesized HOI as", defaultName);
  if (!name || !bounds) return;
  setSaveProgress("synthesize", "saving", "Generating and saving...");
  try {
    const result = await api("/api/synthesize", {
      method: "POST",
      body: JSON.stringify({
        file: state.file,
        filename: name,
        contact_start: bounds.start,
        contact_end: bounds.end,
        object_pos_w: state.synth.seedObjectPos,
        object_quat_w: state.synth.seedObjectQuat,
        object_size: state.synth.objectSize || objectVisualSize(state.frameData),
        object_type: state.synth.objectType,
        anchor_mode: state.synth.anchorMode,
        yaw_only: state.synth.yawOnly,
        support_surfaces: state.synth.supportSurfaces,
        optimize_contact_stage: Boolean(state.synth.contactOptimizationPreview),
        pre_contact_mode: state.synth.preContactMode,
        pre_contact_velocity_w: state.synth.preContactVelocity,
        post_contact_mode: state.synth.postContactMode,
        release_velocity_w: state.synth.releaseVelocity,
        apply_grounding_correction: Boolean(
          state.groundingCorrection && state.groundingProfile,
        ),
        output_folder: state.selectedFolder || "",
      }),
    });
    await loadFiles();
    showToast(`Saved ${result.file}${result.grounding_correction_applied ? " · grounding baked" : ""}`);
    await selectFile(result.file);
    setSaveProgress("synthesize", "success", "Saved");
  } catch (error) {
    setSaveProgress("synthesize", "error", "Save failed");
    showToast(error.message, true);
  }
}

async function handleSynthesisSave() {
  if (state.frameData?.has_object) {
    return cropTrajectory();
  }
  return synthesizeHOI();
}

function synthesisPayload() {
  const bounds = synthBounds();
  if (!bounds || !state.synth.seedObjectPos || !state.synth.seedObjectQuat) {
    return null;
  }
  return {
    file: state.file,
    contact_start: bounds.start,
    contact_end: bounds.end,
    object_pos_w: state.synth.seedObjectPos,
    object_quat_w: state.synth.seedObjectQuat,
    object_size: state.synth.objectSize || objectVisualSize(state.frameData),
    object_type: state.synth.objectType,
    anchor_mode: state.synth.anchorMode,
    yaw_only: state.synth.yawOnly,
    support_surfaces: state.synth.supportSurfaces,
    apply_grounding_correction: Boolean(
      state.groundingCorrection && state.groundingProfile,
    ),
    pre_contact_mode: state.synth.preContactMode,
    pre_contact_velocity_w: state.synth.preContactVelocity,
    post_contact_mode: state.synth.postContactMode,
    release_velocity_w: state.synth.releaseVelocity,
  };
}

async function optimizeContactStagePreview() {
  if (!state.file || !state.frameData) return;
  if (!synthSourceActive()) {
    showToast("Select a robot-only source to optimize contact", true);
    return;
  }
  if (state.synth.contactStart === null || state.synth.contactEnd === null) {
    showToast("Mark contact start and end first", true);
    return;
  }
  if (state.synth.seedFrame !== state.synth.contactStart) {
    await goToContactStart();
    alignObjectToPalms();
  }
  await ensureSynthBoundaryPoses(state.frameData);
  const payload = synthesisPayload();
  if (!payload) {
    showToast("Set the object anchor at contact start first", true);
    return;
  }
  const button = $("optimizeContactButton");
  if (button) {
    button.disabled = true;
    button.textContent = "Optimizing...";
  }
  try {
    const result = await api("/api/optimize_contact", {
      method: "POST",
      body: JSON.stringify(payload),
    });
    state.synth.contactOptimizationPreview = result;
    state.synth.contactOptimizationMeta = result;
    state.synth.optimizeContactStage = true;
    const startIndex = result.frames.indexOf(payload.contact_start);
    const endIndex = result.frames.indexOf(payload.contact_end);
    if (startIndex >= 0) {
      state.synth.optimizedContactStartObjectPos = [...result.object_pos_w[startIndex]];
      state.synth.optimizedContactStartObjectQuat = [...result.object_quat_w[startIndex]];
    }
    if (endIndex >= 0) {
      state.synth.optimizedReleaseObjectPos = [...result.object_pos_w[endIndex]];
      state.synth.optimizedReleaseObjectQuat = [...result.object_quat_w[endIndex]];
    }
    applySynthPreview(state.frameData);
    applyContactOptimizationPreview(state.frameData);
    renderAll();
    showToast(
      `Contact optimized · joint shift ${fmt(result.contact_optimization_max_joint_shift_rad, 3)} rad`,
    );
  } catch (error) {
    showToast(error.message, true);
  } finally {
    renderSynthesisPanel();
  }
}

async function restoreContactOptimizationPreview() {
  if (!state.synth.contactOptimizationPreview) return;
  invalidateSynthRelease();
  await loadFrame(state.frame);
  showToast("Original contact motion restored");
}

function renderAll() {
  renderFrameMeta();
  renderJointList();
  renderObjectFields();
  renderSynthesisPanel();
  update3DFrame(state.frameData);
  drawTimeline();
  $("timelineSlider").max = Math.max(0, state.totalFrames - 1);
  $("timelineSlider").value = state.frame;
  $("frameInput").value = state.frame;
  renderFileList();
}

function renderFrameMeta() {
  const data = state.frameData;
  if (!data) return;
  const time = `${fmt(data.time, 2)} s`;
  $("sceneFrame").textContent = String(data.frame).padStart(4, "0");
  $("sceneTime").textContent = time;
  $("metricFrame").textContent = `${data.frame} / ${data.total_frames - 1}`;
  $("metricTime").textContent = time;
  $("metricFps").textContent = fmt(data.fps, 0);
  $("metricContacts").textContent = `${data.contact_active.filter(Boolean).length} / ${data.contact_active.length}`;
  $("timelineSummary").textContent = `Frame ${data.frame} · ${time}`;
  $("startTime").textContent = "0.00 s";
  $("endTime").textContent = `${fmt((data.total_frames - 1) / data.fps, 2)} s`;
  $("dirtyBadge").hidden = !data.dirty;
  renderVector($("rootVector"), data.base_pos_w, ["X", "Y", "Z"]);
  const eeNames = ["Left hand", "Right hand", "Left foot", "Right foot", "Head"];
  $("eeList").innerHTML = (data.ee_pos_w || []).map((value, index) => `
    <div class="ee-row"><span>${eeNames[index] || `EE ${index}`}</span><strong>${value.map((n) => fmt(n, 2)).join("  ")}</strong></div>
  `).join("");
  $("contactList").innerHTML = (data.contact_active || []).map((active, index) => `
    <div class="contact-pill${active ? " active" : ""}">${["L foot", "R foot", "L hand", "R hand"][index] || `C${index + 1}`}</div>
  `).join("");
  renderGroundingReadout();
}

function renderGroundingReadout() {
  const toggle = $("groundingCorrectionToggle");
  if (toggle) {
    toggle.checked = state.groundingCorrection;
    toggle.disabled = !state.groundingProfile;
  }
  const readout = $("groundingReadout");
  if (!readout) return;
  if (!state.groundingProfile) {
    readout.textContent = "Profile unavailable";
    return;
  }
  if (!state.groundingCorrection) {
    readout.textContent = "Off";
    return;
  }
  const info = groundingFrameInfo();
  if (!info) {
    readout.textContent = "Loading profile";
    return;
  }
  const phase = info.airborne ? "airborne · correction frozen"
    : info.support ? "supported"
      : "transition";
  const feet = info.footHeights.map((value) => `${fmt(value, 3)} m`).join(" · ");
  readout.innerHTML = `<strong>${phase}</strong> · shift ${fmt(info.correction, 3)} m<br><span>foot height ${feet}</span>`;
}

function setGroundingCorrection(enabled) {
  if (enabled && !state.groundingProfile) {
    showToast("Grounding profile is still loading", true);
    const toggle = $("groundingCorrectionToggle");
    if (toggle) toggle.checked = false;
    return;
  }
  const previousCorrection = currentGroundingCorrection(state.frameData);
  state.groundingCorrection = Boolean(enabled);
  const nextCorrection = currentGroundingCorrection(state.frameData);
  const deltaCorrection = nextCorrection - previousCorrection;
  if (
    synthSourceActive()
    && state.synth.seedFrame === state.synth.contactStart
    && state.synth.seedObjectPos
    && Number.isFinite(deltaCorrection)
    && Math.abs(deltaCorrection) > 1e-9
  ) {
    state.synth.seedObjectPos[2] += deltaCorrection;
    if (state.frameData?.object_pos_w) {
      state.frameData.object_pos_w[2] += deltaCorrection;
    }
  }
  invalidateSynthRelease();
  state.originFloorZ = null;
  renderGroundingReadout();
  if (state.frameData && synthSourceActive()) {
    applySynthPreview(state.frameData);
  }
  if (state.frameData) update3DFrame(state.frameData);
  showToast(state.groundingCorrection
    ? "Grounding correction enabled"
    : "Grounding correction disabled");
}

function renderVector(container, values, labels) {
  container.innerHTML = values.map((value, index) => `
    <div class="vector-value"><span>${labels[index]}</span><strong>${fmt(value, 3)}</strong></div>
  `).join("");
}

function jointMatches(name) {
  const query = state.jointSearch.toLowerCase();
  if (query && !name.toLowerCase().includes(query)) return false;
  if (state.jointGroup === "left" && !name.startsWith("left_")) return false;
  if (state.jointGroup === "right" && !name.startsWith("right_")) return false;
  if (state.jointGroup === "waist" && !name.startsWith("waist_")) return false;
  return true;
}

function renderJointList() {
  const data = state.frameData;
  if (!data) return;
  const list = $("jointList");
  list.innerHTML = "";
  data.joint_names.forEach((name, index) => {
    if (!jointMatches(name)) return;
    const [min, max] = data.joint_ranges[index];
    const row = document.createElement("div");
    row.className = "joint-row";
    row.dataset.jointIndex = String(index);
    row.innerHTML = `
      <div class="joint-row-head">
        <span class="joint-name" title="${name}">${data.friendly_joint_names[index]}</span>
        <input class="joint-value" type="number" step="0.001" min="${min}" max="${max}" value="${fmt(data.joint_pos[index], 3)}" aria-label="${name} value">
      </div>
      <input class="joint-slider" type="range" step="0.001" min="${min}" max="${max}" value="${data.joint_pos[index]}" aria-label="${name} slider">
    `;
    const number = row.querySelector(".joint-value");
    const slider = row.querySelector(".joint-slider");
    const edit = (value) => editJoint(index, value, number, slider);
    number.addEventListener("change", () => edit(number.value));
    slider.addEventListener("input", () => edit(slider.value));
    list.appendChild(row);
  });
}

function setJointPreviewValue(index, rawValue) {
  if (!state.frameData) return;
  if (state.synth.contactOptimizationPreview) {
    state.synth.contactOptimizationPreview = null;
    state.synth.contactOptimizationMeta = null;
    state.synth.optimizeContactStage = false;
  }
  const [min, max] = state.frameData.joint_ranges[index];
  const value = Math.max(min, Math.min(max, Number(rawValue)));
  if (!Number.isFinite(value)) return;
  state.frameData.joint_pos[index] = value;
  const row = document.querySelector(`.joint-row[data-joint-index="${index}"]`);
  if (row) {
    row.querySelector(".joint-value").value = fmt(value, 3);
    row.querySelector(".joint-slider").value = value;
  }
  state.frameData.dirty = true;
  $("dirtyBadge").hidden = false;
  update3DFrame(state.frameData);
}

function editJoint(index, rawValue, number, slider) {
  if (!state.frameData) return;
  setJointPreviewValue(index, rawValue);
  number.value = fmt(state.frameData.joint_pos[index], 3);
  slider.value = state.frameData.joint_pos[index];
  queueEdit({ joint_index: index, joint_value: state.frameData.joint_pos[index] });
}

function renderObjectFields() {
  const data = state.frameData;
  if (!data) return;
  const objectType = objectVisualType(data);
  const objectSize = objectVisualSize(data);
  const yawOnlyToggle = $("yawOnlyToggle");
  if (yawOnlyToggle) yawOnlyToggle.checked = state.synth.yawOnly;
  const supportSurfacesToggle = $("supportSurfacesToggle");
  if (supportSurfacesToggle) supportSurfacesToggle.checked = state.synth.supportSurfaces;
  const objectAnchorMode = $("objectAnchorMode");
  if (objectAnchorMode) objectAnchorMode.value = state.synth.anchorMode;
  const objectTypeSelect = $("objectType");
  if (objectTypeSelect) objectTypeSelect.value = objectType;
  const objectTypeSummary = $("objectTypeSummary");
  if (objectTypeSummary) objectTypeSummary.textContent = objectType;
  const objectSizeLabel = $("objectSizeLabel");
  if (objectSizeLabel) {
    objectSizeLabel.textContent = objectType === "sphere"
      ? "Sphere diameter · m"
      : "Box edge length · m";
  }
  const objectSizeInput = $("objectSizeInput");
  if (objectSizeInput) {
    objectSizeInput.value = fmt(objectSize, 3);
    objectSizeInput.disabled = !state.synth.objectSizeManual;
    objectSizeInput.title = objectType === "sphere"
      ? state.synth.objectSizeManual
        ? "Sphere diameter in meters"
        : "Automatically calculated sphere diameter from wrist span"
      : state.synth.objectSizeManual
        ? "Box edge length in meters"
        : "Automatically calculated box edge length from wrist span";
  }
  const objectSizeMode = $("objectSizeMode");
  if (objectSizeMode) objectSizeMode.value = state.synth.objectSizeManual ? "manual" : "auto";
  document.querySelectorAll("[data-rotate-axis]").forEach((button) => {
    button.disabled = state.synth.yawOnly && button.dataset.rotateAxis !== "z";
  });
  makeTransformFields($("objectPositionFields"), "object-pos", ["X", "Y", "Z"], data.object_pos_w, (index, value) => {
    const next = [...state.frameData.object_pos_w];
    next[index] = value;
    editObject({ object_pos_w: next });
  });
  makeTransformFields($("objectQuaternionFields"), "object-quat", ["W", "X", "Y", "Z"], data.object_quat_w, (index, value) => {
    const next = [...state.frameData.object_quat_w];
    next[index] = value;
    editObject({ object_quat_w: next });
  });
}

function makeTransformFields(container, prefix, labels, values, onChange) {
  container.innerHTML = labels.map((label, index) => `
    <div class="transform-field"><label for="${prefix}-${index}">${label}</label><input id="${prefix}-${index}" type="number" step="0.001" value="${fmt(values[index], 4)}"></div>
  `).join("");
  labels.forEach((_, index) => {
    container.querySelector(`#${prefix}-${index}`).addEventListener("change", (event) => {
      const value = Number(event.target.value);
      if (Number.isFinite(value)) onChange(index, value);
    });
  });
}

function syncObjectInputs() {
  if (!state.frameData) return;
  state.frameData.object_pos_w.forEach((value, index) => {
    const input = $(`object-pos-${index}`);
    if (input) input.value = fmt(value, 4);
  });
  state.frameData.object_quat_w.forEach((value, index) => {
    const input = $(`object-quat-${index}`);
    if (input) input.value = fmt(value, 4);
  });
  const objectSizeInput = $("objectSizeInput");
  if (objectSizeInput) objectSizeInput.value = fmt(objectVisualSize(state.frameData), 3);
}

async function setObjectAnchorMode(mode) {
  const nextMode = mode === ANCHOR_RIGHT_PALM_NORMAL
    ? ANCHOR_RIGHT_PALM_NORMAL
    : ANCHOR_PALM_CENTER;
  state.synth.anchorMode = nextMode;
  state.synth.anchorModeTouched = true;
  if (nextMode === ANCHOR_RIGHT_PALM_NORMAL) {
    state.synth.objectType = "box";
    state.synth.objectTypeTouched = true;
    state.synth.objectSize = RIGHT_PALM_OBJECT_SIZE;
    state.synth.objectSizeManual = true;
    state.synth.objectSizeMode = "manual";
  }
  persistSynthSettings();
  invalidateSynthRelease();
  if (!state.frameData) return;
  if (nextMode === ANCHOR_RIGHT_PALM_NORMAL) {
    state.frameData.object_type = "box";
    state.frameData.object_size = RIGHT_PALM_OBJECT_SIZE;
  }
  if (synthSourceActive()) {
    alignObjectToPalms();
    return;
  } else {
    const pose = anchorPoseForSynthesis(state.frameData);
    state.frameData.object_pos_w = [...pose.pos];
    state.frameData.object_quat_w = normalizeBoxQuatForMode(pose.quat);
    state.frameData.object_anchor_mode = nextMode;
    state.frameData.dirty = true;
    $("dirtyBadge").hidden = false;
    const changes = {
      object_anchor_mode: nextMode,
      object_pos_w: state.frameData.object_pos_w,
      object_quat_w: state.frameData.object_quat_w,
    };
    if (nextMode === ANCHOR_RIGHT_PALM_NORMAL) {
      changes.object_type = "box";
      changes.object_size = RIGHT_PALM_OBJECT_SIZE;
    }
    queueEdit(changes);
    syncObjectInputs();
    renderObjectFields();
    renderSynthesisPanel();
    update3DFrame(state.frameData);
  }
  showToast(nextMode === ANCHOR_RIGHT_PALM_NORMAL
    ? "Anchor: right palm normal +5cm"
    : "Anchor: palm midpoint");
}

function setObjectSize(rawValue) {
  if (!state.frameData) return;
  const value = Number(rawValue);
  if (!Number.isFinite(value) || value <= 0.0) {
    showToast("Object size must be positive", true);
    renderObjectFields();
    return;
  }
  state.synth.objectSize = value;
  state.synth.objectSizeManual = true;
  state.synth.objectSizeMode = "manual";
  persistSynthSettings();
  if (state.frameData.has_object) {
    state.frameData.object_size = value;
    state.frameData.dirty = true;
    $("dirtyBadge").hidden = false;
    queueEdit({ object_size: value });
  }
  renderObjectFields();
  renderSynthesisPanel();
  update3DFrame(state.frameData);
}

function setObjectSizeMode(mode) {
  const previousSize = state.frameData ? objectVisualSize(state.frameData) : 0.28;
  const manual = mode === "manual";
  state.synth.objectSizeManual = manual;
  state.synth.objectSizeMode = manual ? "manual" : "auto";
  if (manual && (!Number.isFinite(Number(state.synth.objectSize)) || state.synth.objectSize <= 0.02)) {
    state.synth.objectSize = state.synth.objectSizeAuto || previousSize;
  }
  persistSynthSettings();
  if (!state.frameData) {
    renderObjectFields();
    return;
  }
  const value = objectVisualSize(state.frameData);
  const storedValue = Number(state.frameData.object_size);
  if (
    state.frameData.has_object
    && (!Number.isFinite(storedValue) || Math.abs(storedValue - value) > 1e-6)
  ) {
    state.frameData.object_size = value;
    state.frameData.dirty = true;
    $("dirtyBadge").hidden = false;
    queueEdit({ object_size: value });
  }
  renderObjectFields();
  renderSynthesisPanel();
  update3DFrame(state.frameData);
  showToast(manual
    ? `Manual object size enabled · ${fmt(value, 3)} m`
    : `Automatic object size enabled · ${fmt(value, 3)} m`);
}

function rotateObject(axisName, direction) {
  if (!state.frameData) return;
  if (state.synth.yawOnly && axisName !== "z") {
    showToast("Horizontal lock keeps roll/pitch at 0");
    return;
  }
  const stepDegrees = Math.max(0.1, Math.min(45, Number($("rotationStepInput").value) || 5));
  const delta = axisAngleWxyz(axisName, direction * stepDegrees * Math.PI / 180);
  editObject({
    object_quat_w: multiplyWxyz(state.frameData.object_quat_w, delta),
  });
}

let editQueue = null;
function queueEdit(changes) {
  clearTimeout(editQueue);
  const file = state.file;
  const frame = state.frame;
  editQueue = setTimeout(async () => {
    try {
      const edited = await api("/api/edit", {
        method: "POST",
        body: JSON.stringify({ file, frame, changes }),
      });
      if (state.file === file && state.frame === frame) {
        state.frameData = edited;
        renderFrameMeta();
        update3DFrame(edited);
        renderFileList();
      }
    } catch (error) {
      showToast(error.message, true);
    }
  }, 80);
}

function editObject(changes) {
  if (!state.frameData) return;
  if (state.synth.contactOptimizationPreview) {
    state.synth.contactOptimizationPreview = null;
    state.synth.contactOptimizationMeta = null;
    state.synth.optimizeContactStage = false;
  }
  if (changes.object_pos_w) state.frameData.object_pos_w = changes.object_pos_w;
  if (changes.object_quat_w) {
    const norm = Math.hypot(...changes.object_quat_w);
    const nextQuat = norm > 1e-6
      ? changes.object_quat_w.map((value) => value / norm)
      : [1, 0, 0, 0];
    state.frameData.object_quat_w = normalizeBoxQuatForMode(nextQuat);
    changes.object_quat_w = state.frameData.object_quat_w;
  }
  state.frameData.dirty = true;
  syncObjectInputs();
  renderFrameMeta();
  update3DFrame(state.frameData);
  if (synthSourceActive()) {
    rememberSynthSeed();
    renderSynthesisPanel();
    return;
  }
  queueEdit(changes);
}

function setObjectType(type) {
  state.synth.objectType = type === "sphere" ? "sphere" : "box";
  state.synth.objectTypeTouched = true;
  persistSynthSettings();
  if (!state.frameData) return;
  state.frameData.object_type = state.synth.objectType;
  renderObjectFields();
  update3DFrame(state.frameData);
  if (synthSourceActive()) {
    renderSynthesisPanel();
    return;
  }
  queueEdit({ object_type: state.synth.objectType });
  showToast(`${state.synth.objectType === "sphere" ? "Sphere" : "Box"} selected`);
}

async function setYawOnly(enabled) {
  state.synth.yawOnly = Boolean(enabled);
  state.synth.yawOnlyTouched = true;
  persistSynthSettings();
  invalidateSynthRelease();
  if (state.frameData) {
    state.frameData.object_quat_w = normalizeBoxQuatForMode(state.frameData.object_quat_w);
    await refreshSynthSeedForMode();
    if (synthSourceActive()) applySynthPreview(state.frameData);
    syncObjectInputs();
    renderObjectFields();
    renderSynthesisPanel();
    update3DFrame(state.frameData);
  }
  showToast(state.synth.yawOnly ? "Horizontal yaw lock enabled" : "Horizontal yaw lock disabled");
}

async function setSupportSurfaces(enabled) {
  state.synth.supportSurfaces = Boolean(enabled);
  state.synth.supportTouched = true;
  persistSynthSettings();
  if (state.frameData) {
    await ensureSynthBoundaryPoses(state.frameData);
    renderObjectFields();
    renderSynthesisPanel();
    update3DFrame(state.frameData);
  }
  showToast(state.synth.supportSurfaces ? "Support surfaces enabled" : "Support surfaces disabled");
}

async function setPreContactMode(mode) {
  state.synth.preContactMode = mode === "parabolic" ? "parabolic" : "hold";
  state.synth.preContactModeTouched = true;
  persistSynthSettings();
  invalidateSynthRelease("pre");
  if (state.frameData) {
    await ensureSynthBoundaryPoses(state.frameData);
    if (synthSourceActive()) applySynthPreview(state.frameData);
    renderObjectFields();
    renderSynthesisPanel();
    update3DFrame(state.frameData);
  }
  showToast(
    state.synth.preContactMode === "parabolic"
      ? "Inverted pre-contact trajectory enabled"
      : "Pre-contact hold enabled",
  );
}

async function setPostContactMode(mode) {
  state.synth.postContactMode = mode === "parabolic" ? "parabolic" : "hold";
  state.synth.postContactModeTouched = true;
  persistSynthSettings();
  invalidateSynthRelease("post");
  if (state.frameData) {
    await ensureSynthBoundaryPoses(state.frameData);
    if (synthSourceActive()) applySynthPreview(state.frameData);
    renderObjectFields();
    renderSynthesisPanel();
    update3DFrame(state.frameData);
  }
  showToast(
    state.synth.postContactMode === "parabolic"
      ? "Parabolic throw preview enabled"
      : "Post-contact hold enabled",
  );
}

function updateTableAndObjectDrag(event) {
  const point = new THREE.Vector3();
  if (!raycaster.ray.intersectPlane(objectPlane, point)) return;
  const local = contentRoot.worldToLocal(point.clone());
  const next = [local.x, local.y, local.z];
  next[2] = state.objectDrag.origin[2];
  state.frameData.object_pos_w = next;
  objectMesh.position.fromArray(next);
  syncObjectInputs();
  renderFrameMeta();
}

function updateTrajectoryVelocityDrag(event) {
  const drag = state.throwVelocityDrag;
  if (!drag) return;
  const point = new THREE.Vector3();
  if (!raycaster.ray.intersectPlane(drag.plane, point)) return;
  const local = contentRoot.worldToLocal(point.clone());
  const delta = local.sub(new THREE.Vector3(...drag.originPos));
  const velocity = delta.multiplyScalar(1 / THROW_ARROW_TIME_SCALE);
  const speed = velocity.length();
  if (speed > MAX_THROW_SPEED) velocity.setLength(MAX_THROW_SPEED);
  if (drag.phase === "pre") {
    const bounds = synthBounds();
    const elapsed = bounds
      ? (bounds.start - state.frame) / Math.max(state.fps, 1e-6)
      : 0;
    const gravity = new THREE.Vector3(...(state.synth.preContactGravity || EARTH_GRAVITY));
    state.synth.preContactVelocity = velocity.add(gravity.multiplyScalar(elapsed)).toArray();
  } else {
    const bounds = synthBounds();
    const elapsed = bounds
      ? (state.frame - bounds.end) / Math.max(state.fps, 1e-6)
      : 0;
    const gravity = new THREE.Vector3(...(state.synth.releaseGravity || EARTH_GRAVITY));
    state.synth.releaseVelocity = velocity.sub(gravity.multiplyScalar(elapsed)).toArray();
  }
  if (state.frameData) {
    applySynthPreview(state.frameData);
    renderSynthesisPanel();
    update3DFrame(state.frameData);
  }
}

function normalizedPointer(event) {
  const rect = renderer.domElement.getBoundingClientRect();
  pointer.x = ((event.clientX - rect.left) / rect.width) * 2 - 1;
  pointer.y = -((event.clientY - rect.top) / rect.height) * 2 + 1;
}

function onScenePointerDown(event) {
  if (!state.frameData || !renderer) return;
  normalizedPointer(event);
  raycaster.setFromCamera(pointer, camera);
  const targets = [
    objectMesh,
    preContactVelocityHandle,
    throwVelocityHandle,
    ...jointMarkers,
  ].filter(Boolean);
  const hit = raycaster.intersectObjects(targets, false)[0];
  if (
    (hit?.object === preContactVelocityHandle || hit?.object === throwVelocityHandle)
    && synthSourceActive()
  ) {
    const phase = hit.object === preContactVelocityHandle ? "pre" : "post";
    const bounds = synthBounds();
    const phaseActive = phase === "pre"
      ? bounds && state.frame < bounds.start && state.synth.preContactMode === "parabolic"
      : bounds && state.frame > bounds.end && state.synth.postContactMode === "parabolic";
    if (!phaseActive || !Array.isArray(state.frameData.object_pos_w)) return;
    const handleWorld = throwVelocityHandle.getWorldPosition(new THREE.Vector3());
    if (phase === "pre") {
      handleWorld.copy(preContactVelocityHandle.getWorldPosition(new THREE.Vector3()));
    }
    const normal = camera.getWorldDirection(new THREE.Vector3()).normalize();
    const plane = new THREE.Plane().setFromNormalAndCoplanarPoint(normal, handleWorld);
    state.throwVelocityDrag = {
      phase,
      originPos: [...state.frameData.object_pos_w],
      plane,
    };
    controls.enabled = false;
    renderer.domElement.setPointerCapture(event.pointerId);
    return;
  }
  if (hit?.object === objectMesh) {
    const worldPosition = objectMesh.getWorldPosition(new THREE.Vector3());
    objectPlane.set(new THREE.Vector3(0, 0, 1), -worldPosition.z);
    state.objectDrag = { origin: [...state.frameData.object_pos_w] };
    controls.enabled = false;
    renderer.domElement.setPointerCapture(event.pointerId);
    updateTableAndObjectDrag(event);
    return;
  }
  if (hit?.object?.userData?.joint) {
    const joint = hit.object.userData.joint;
    const jointIndex = joint.index;
    state.jointDrag = {
      jointIndex,
      startY: event.clientY,
      originValue: state.frameData.joint_pos[jointIndex],
    };
    controls.enabled = false;
    renderer.domElement.setPointerCapture(event.pointerId);
    return;
  }
}

function onScenePointerMove(event) {
  if (state.throwVelocityDrag) {
    normalizedPointer(event);
    raycaster.setFromCamera(pointer, camera);
    updateTrajectoryVelocityDrag(event);
    return;
  }
  if (state.objectDrag) {
    normalizedPointer(event);
    raycaster.setFromCamera(pointer, camera);
    updateTableAndObjectDrag(event);
    return;
  }
  if (state.jointDrag) {
    const delta = (state.jointDrag.startY - event.clientY) * 0.012;
    setJointPreviewValue(state.jointDrag.jointIndex, state.jointDrag.originValue + delta);
  }
}

function onScenePointerUp(event) {
  if (state.throwVelocityDrag) {
    const phase = state.throwVelocityDrag.phase;
    state.throwVelocityDrag = null;
    showToast(`${phase === "pre" ? "Pre-contact" : "Release"} velocity updated`);
  }
  if (state.objectDrag) {
    if (synthSourceActive()) {
      rememberSynthSeed();
      renderSynthesisPanel();
    } else {
      queueEdit({ object_pos_w: state.frameData.object_pos_w });
    }
    state.objectDrag = null;
  }
  if (state.jointDrag) {
    const { jointIndex } = state.jointDrag;
    queueEdit({
      joint_index: jointIndex,
      joint_value: state.frameData.joint_pos[jointIndex],
    });
    state.jointDrag = null;
  }
  controls.enabled = true;
  try { renderer.domElement.releasePointerCapture(event.pointerId); } catch (_) {}
}

function drawTimeline() {
  const maxFrame = Math.max(1, state.totalFrames - 1);
  const percent = Math.max(0, Math.min(1, state.frame / maxFrame));
  $("timelineProgress").style.width = `${percent * 100}%`;
  $("timelineThumb").style.left = `${percent * 100}%`;

  const crop = cropBounds();
  const cropLeft = Math.max(0, Math.min(1, crop.start / maxFrame));
  const cropRight = Math.max(cropLeft, Math.min(1, crop.end / maxFrame));
  $("timelineCropWindow").style.left = `${cropLeft * 100}%`;
  $("timelineCropWindow").style.width = `${Math.max(0.4, (cropRight - cropLeft) * 100)}%`;
  $("timelineCropStart").style.left = `${cropLeft * 100}%`;
  $("timelineCropEnd").style.left = `${cropRight * 100}%`;
  $("timelineCropStart").setAttribute("aria-valuenow", String(crop.start));
  $("timelineCropEnd").setAttribute("aria-valuenow", String(crop.end));
  $("timelineCropStart").setAttribute("aria-valuemax", String(state.totalFrames - 1));
  $("timelineCropEnd").setAttribute("aria-valuemax", String(state.totalFrames - 1));
  $("cropSummary").textContent = `keep ${crop.start}–${crop.end}`;

  let contactStart = null;
  let contactEnd = null;
  const marked = synthBounds();
  if (marked) {
    contactStart = marked.start;
    contactEnd = marked.end;
  } else if (state.trajectory?.indices?.length && state.trajectory?.contact?.length) {
    state.trajectory.indices.forEach((frameIndex, index) => {
      if (state.trajectory.contact[index] > 0.5) {
        contactStart = contactStart === null ? frameIndex : Math.min(contactStart, frameIndex);
        contactEnd = contactEnd === null ? frameIndex : Math.max(contactEnd, frameIndex);
      }
    });
  }

  const contact = $("timelineContact");
  if (contactStart === null || contactEnd === null) {
    contact.style.display = "none";
    return;
  }
  const left = Math.max(0, Math.min(1, contactStart / maxFrame));
  const right = Math.max(left, Math.min(1, contactEnd / maxFrame));
  contact.style.display = "block";
  contact.style.left = `${left * 100}%`;
  contact.style.width = `${Math.max(0.3, (right - left) * 100)}%`;
}

function frameFromTimelinePointer(event) {
  const rect = $("timebar").getBoundingClientRect();
  const ratio = Math.max(0, Math.min(1, (event.clientX - rect.left) / Math.max(rect.width, 1)));
  return Math.round(ratio * Math.max(0, state.totalFrames - 1));
}

function onTimelinePointerDown(event) {
  if (!state.file || state.totalFrames <= 1) return;
  if (event.target.closest(".timebar-crop-handle")) return;
  event.preventDefault();
  state.timelineDrag = { pointerId: event.pointerId };
  event.currentTarget.setPointerCapture?.(event.pointerId);
  loadFrame(frameFromTimelinePointer(event));
}

function onTimelinePointerMove(event) {
  if (!state.timelineDrag || event.pointerId !== state.timelineDrag.pointerId) return;
  loadFrame(frameFromTimelinePointer(event));
}

function onTimelinePointerUp(event) {
  if (!state.timelineDrag || event.pointerId !== state.timelineDrag.pointerId) return;
  const timeline = $("timebar");
  state.timelineDrag = null;
  try {
    timeline.releasePointerCapture?.(event.pointerId);
  } catch (_) {}
}

function onCropPointerDown(event) {
  event.preventDefault();
  event.stopPropagation();
  if (!state.file || state.totalFrames <= 1) return;
  state.cropDrag = {
    side: event.currentTarget.dataset.cropSide,
    pointerId: event.pointerId,
    handle: event.currentTarget,
  };
  event.currentTarget.setPointerCapture?.(event.pointerId);
  drawTimeline();
}

function onCropPointerMove(event) {
  if (!state.cropDrag || event.pointerId !== state.cropDrag.pointerId) return;
  const frame = frameFromTimelinePointer(event);
  const current = cropBounds();
  if (state.cropDrag.side === "start") {
    state.cropStart = Math.min(frame, current.end);
  } else {
    state.cropEnd = Math.max(frame, current.start);
  }
  drawTimeline();
}

function onCropPointerUp(event) {
  if (!state.cropDrag || event.pointerId !== state.cropDrag.pointerId) return;
  const current = cropBounds();
  const handle = state.cropDrag.handle;
  state.cropDrag = null;
  if (state.frame < current.start) {
    loadFrame(current.start);
  } else if (state.frame > current.end) {
    loadFrame(current.end);
  } else {
    drawTimeline();
  }
  try {
    handle?.releasePointerCapture?.(event.pointerId);
  } catch (_) {}
}

async function cropTrajectory() {
  if (!state.file) return;
  const bounds = cropBounds();
  const last = Math.max(0, state.totalFrames - 1);
  if (bounds.start === 0 && bounds.end === last) {
    showToast("Move the crop handles to choose a shorter range", true);
    return;
  }
  const sourceName = state.file.split("/").pop().replace(/\.npz$/, "");
  const defaultName = `${sourceName}_crop_${bounds.start}_${bounds.end}.npz`;
  const name = window.prompt("Save cropped copy as", defaultName);
  if (!name) return;
  const cropButton = $("cropButton");
  if (cropButton?.disabled) return;
  if (cropButton) cropButton.disabled = true;
  setSaveProgress("save", "saving", "Saving cropped copy...");
  try {
    const result = await api("/api/crop", {
      method: "POST",
      body: JSON.stringify({
        file: state.file,
        filename: name,
        start_frame: bounds.start,
        end_frame: bounds.end,
        apply_grounding_correction: Boolean(
          state.groundingCorrection && state.groundingProfile,
        ),
        output_folder: state.selectedFolder || "",
      }),
    });
    await loadFiles();
    showToast(`Saved ${result.file} · ${result.frames} frames${result.grounding_correction_applied ? " · grounding baked" : ""}`);
    await selectFile(result.file);
    setSaveProgress("save", "success", "Saved");
  } catch (error) {
    setSaveProgress("save", "error", "Save failed");
    showToast(error.message, true);
  } finally {
    if (cropButton) cropButton.disabled = false;
  }
}

function startPlayback() {
  state.playing = true;
  $("playButton").textContent = "Ⅱ";
  state.timer = setInterval(() => {
    if (state.frame >= state.totalFrames - 1) state.frame = -1;
    loadFrame(state.frame + 1);
  }, Math.max(16, 1000 / state.fps));
}

function stopPlayback() {
  state.playing = false;
  $("playButton").textContent = "▶";
  clearInterval(state.timer);
  state.timer = null;
}

async function resetCurrent() {
  if (!state.file) return;
  stopPlayback();
  try {
    await api("/api/reset", {
      method: "POST",
      body: JSON.stringify({ file: state.file, frame: state.frame }),
    });
    await selectFile(state.file);
    showToast("Edits discarded for this file");
  } catch (error) {
    showToast(error.message, true);
  }
}

async function saveCopy() {
  if (!state.file) return;
  const saveButton = $("saveButton");
  if (saveButton?.disabled) return;
  const defaultName = `${state.file.split("/").pop().replace(/\.npz$/, "")}_edited.npz`;
  const name = window.prompt("Save edited copy as", defaultName);
  if (!name) return;
  setSaveProgress("save", "saving", "Saving edited copy...");
  try {
    const result = await api("/api/save", {
      method: "POST",
      body: JSON.stringify({
        file: state.file,
        filename: name,
        apply_grounding_correction: Boolean(
          state.groundingCorrection && state.groundingProfile,
        ),
        output_folder: state.selectedFolder || "",
      }),
    });
    await loadFiles();
    showToast(`Saved ${result.file}${result.grounding_correction_applied ? " · grounding baked" : ""}`);
    setSaveProgress("save", "success", "Saved");
  } catch (error) {
    setSaveProgress("save", "error", "Save failed");
    showToast(error.message, true);
  }
}

function setTab(tab) {
  state.tab = tab;
  document.querySelectorAll(".inspector-tab").forEach((button) =>
    button.classList.toggle("active", button.dataset.tab === tab));
  document.querySelectorAll(".tab-content").forEach((content) =>
    content.classList.toggle("active", content.id === `tab-${tab}`));
}

function wireEvents() {
  $("refreshFiles").addEventListener("click", () => loadFolders()
    .then(() => loadFiles())
    .catch((error) => showToast(error.message, true)));
  $("folderSelect").addEventListener("change", (event) => {
    selectDatasetFolder(event.target.value).catch((error) => showToast(error.message, true));
  });
  $("newFolderButton").addEventListener("click", () => {
    createDatasetFolder().catch((error) => showToast(error.message, true));
  });
  $("fileSearch").addEventListener("input", (event) => { state.fileSearch = event.target.value; renderFileList(); });
  $("jointSearch").addEventListener("input", (event) => { state.jointSearch = event.target.value; renderJointList(); });
  $("jointGroup").addEventListener("change", (event) => { state.jointGroup = event.target.value; renderJointList(); });
  $("timelineSlider").addEventListener("input", (event) => loadFrame(event.target.value));
  $("frameInput").addEventListener("change", (event) => loadFrame(event.target.value));
  $("stepBack").addEventListener("click", () => loadFrame(state.frame - 1));
  $("stepForward").addEventListener("click", () => loadFrame(state.frame + 1));
  $("playButton").addEventListener("click", () => state.playing ? stopPlayback() : startPlayback());
  $("fitView").addEventListener("click", fitView);
  $("groundColorButton").addEventListener("click", cycleGroundColor);
  $("groundingCorrectionToggle").addEventListener("change", (event) => {
    setGroundingCorrection(event.target.checked);
  });
  $("resetButton").addEventListener("click", resetCurrent);
  $("saveButton").addEventListener("click", saveCopy);
  $("cropButton").addEventListener("click", () => cropTrajectory().catch((error) => showToast(error.message, true)));
  $("timebar").addEventListener("pointerdown", onTimelinePointerDown);
  $("timelineCropStart").addEventListener("pointerdown", onCropPointerDown);
  $("timelineCropEnd").addEventListener("pointerdown", onCropPointerDown);
  window.addEventListener("pointermove", onTimelinePointerMove);
  window.addEventListener("pointerup", onTimelinePointerUp);
  window.addEventListener("pointercancel", onTimelinePointerUp);
  window.addEventListener("pointermove", onCropPointerMove);
  window.addEventListener("pointerup", onCropPointerUp);
  window.addEventListener("pointercancel", onCropPointerUp);
  $("markStartButton").addEventListener("click", () => markContactStart());
  $("markEndButton").addEventListener("click", () => markContactEnd().catch((error) => showToast(error.message, true)));
  $("goStartButton").addEventListener("click", () => goToContactStart().catch((error) => showToast(error.message, true)));
  $("alignBoxButton").addEventListener("click", alignObjectToPalms);
  $("synthesizeButton").addEventListener("click", () => handleSynthesisSave().catch((error) => showToast(error.message, true)));
  $("optimizeContactButton").addEventListener("click", () => optimizeContactStagePreview().catch((error) => showToast(error.message, true)));
  $("restoreContactButton").addEventListener("click", () => restoreContactOptimizationPreview().catch((error) => showToast(error.message, true)));
  $("yawOnlyToggle").addEventListener("change", (event) => {
    setYawOnly(event.target.checked).catch((error) => showToast(error.message, true));
  });
  $("supportSurfacesToggle").addEventListener("change", (event) => {
    setSupportSurfaces(event.target.checked).catch((error) => showToast(error.message, true));
  });
  $("objectAnchorMode").addEventListener("change", (event) => {
    setObjectAnchorMode(event.target.value).catch((error) => showToast(error.message, true));
  });
  $("objectType").addEventListener("change", (event) => {
    setObjectType(event.target.value);
  });
  $("objectSizeInput").addEventListener("change", (event) => {
    setObjectSize(event.target.value);
  });
  $("objectSizeMode").addEventListener("change", (event) => {
    setObjectSizeMode(event.target.value);
  });
  $("preContactMode").addEventListener("change", (event) => {
    setPreContactMode(event.target.value).catch((error) => showToast(error.message, true));
  });
  $("postContactMode").addEventListener("change", (event) => {
    setPostContactMode(event.target.value).catch((error) => showToast(error.message, true));
  });
  $("contactStartInput").addEventListener("change", (event) => {
    state.synth.contactStart = Math.max(0, Math.min(state.totalFrames - 1, Math.round(Number(event.target.value) || 0)));
    if (state.synth.contactEnd === null) state.synth.contactEnd = state.synth.contactStart;
    invalidateSynthRelease();
    renderSynthesisPanel();
  });
  $("contactEndInput").addEventListener("change", (event) => {
    state.synth.contactEnd = Math.max(0, Math.min(state.totalFrames - 1, Math.round(Number(event.target.value) || 0)));
    if (state.synth.contactStart === null) state.synth.contactStart = state.synth.contactEnd;
    invalidateSynthRelease();
    renderSynthesisPanel();
  });
  document.querySelectorAll("[data-rotate-axis]").forEach((button) => {
    button.addEventListener("click", () => rotateObject(
      button.dataset.rotateAxis,
      Number(button.dataset.rotateDir) || 1,
    ));
  });
  document.querySelectorAll(".segment").forEach((button) => button.addEventListener("click", () => {
    document.querySelectorAll(".segment").forEach((item) => item.classList.toggle("active", item === button));
    setView(button.dataset.view);
  }));
  document.querySelectorAll(".inspector-tab").forEach((button) =>
    button.addEventListener("click", () => setTab(button.dataset.tab)));
  window.addEventListener("resize", () => { resize3D(); drawTimeline(); });
}

init3D();
wireEvents();
state.modelReady = loadModel().catch((error) => {
  showToast(`G1 model failed: ${error.message}`, true);
  $("workspaceStatus").textContent = "G1 mesh model unavailable";
});
loadFolders()
  .then(() => loadFiles())
  .catch((error) => showToast(error.message, true));
