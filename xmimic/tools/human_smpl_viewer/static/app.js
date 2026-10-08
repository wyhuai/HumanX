import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";

const SMPL_BONES = [
  [0, 1], [0, 2], [0, 3], [1, 4], [2, 5], [3, 6], [4, 7], [5, 8],
  [6, 9], [7, 10], [8, 11], [9, 12], [9, 13], [9, 14], [12, 15],
  [13, 16], [14, 17], [16, 18], [17, 19], [18, 20], [19, 21],
  [20, 22], [21, 23],
];

const state = {
  motions: [],
  motion: null,
  frame: 0,
  selectedJoint: 20,
  playing: false,
  lastTick: 0,
  frameCarry: 0,
  viewOffset: new THREE.Vector3(),
};

const el = {
  motionRoot: document.getElementById("motionRoot"),
  motionSelect: document.getElementById("motionSelect"),
  frameCount: document.getElementById("frameCount"),
  fpsValue: document.getElementById("fpsValue"),
  selectedJointName: document.getElementById("selectedJointName"),
  selectedJointPosition: document.getElementById("selectedJointPosition"),
  jointList: document.getElementById("jointList"),
  viewer: document.getElementById("viewer"),
  playButton: document.getElementById("playButton"),
  prevFrameButton: document.getElementById("prevFrameButton"),
  nextFrameButton: document.getElementById("nextFrameButton"),
  frameSlider: document.getElementById("frameSlider"),
  frameNumber: document.getElementById("frameNumber"),
  frameLabel: document.getElementById("frameLabel"),
  speedSelect: document.getElementById("speedSelect"),
  trailToggle: document.getElementById("trailToggle"),
  ballToggle: document.getElementById("ballToggle"),
  orientationToggle: document.getElementById("orientationToggle"),
  curveModeSelect: document.getElementById("curveModeSelect"),
  curveCanvas: document.getElementById("curveCanvas"),
};

const scene = new THREE.Scene();
scene.background = new THREE.Color(0x0b0d0f);

const camera = new THREE.PerspectiveCamera(48, 1, 0.01, 100);
camera.position.set(1.4, 1.1, 1.8);

const renderer = new THREE.WebGLRenderer({ antialias: true });
renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
el.viewer.appendChild(renderer.domElement);

const controls = new OrbitControls(camera, renderer.domElement);
controls.enableDamping = true;
controls.target.set(0.1, 0.2, 0);

scene.add(new THREE.HemisphereLight(0xffffff, 0x30343a, 2.4));

const grid = new THREE.GridHelper(1.8, 18, 0x4d555f, 0x2d333a);
grid.position.y = 0;
scene.add(grid);

const axes = new THREE.AxesHelper(0.45);
scene.add(axes);

const jointGroup = new THREE.Group();
const jointMeshes = [];
const jointGeometry = new THREE.SphereGeometry(0.018, 18, 12);
const jointMaterial = new THREE.MeshStandardMaterial({ color: 0x8fbfff, roughness: 0.7 });
const selectedMaterial = new THREE.MeshStandardMaterial({ color: 0xf2c14e, roughness: 0.6 });

for (let i = 0; i < 24; i += 1) {
  const mesh = new THREE.Mesh(jointGeometry, jointMaterial);
  mesh.userData.jointIndex = i;
  jointMeshes.push(mesh);
  jointGroup.add(mesh);
}
scene.add(jointGroup);

const lineGeometry = new THREE.BufferGeometry();
const linePositions = new Float32Array(SMPL_BONES.length * 2 * 3);
lineGeometry.setAttribute("position", new THREE.BufferAttribute(linePositions, 3));
const skeletonLines = new THREE.LineSegments(
  lineGeometry,
  new THREE.LineBasicMaterial({ color: 0xdce7f5, transparent: true, opacity: 0.85 })
);
scene.add(skeletonLines);

const trailGeometry = new THREE.BufferGeometry();
const trailLine = new THREE.Line(
  trailGeometry,
  new THREE.LineBasicMaterial({ color: 0xf2c14e, transparent: true, opacity: 0.9 })
);
scene.add(trailLine);

const ball = new THREE.Mesh(
  new THREE.SphereGeometry(0.08, 32, 18),
  new THREE.MeshStandardMaterial({ color: 0xe53935, roughness: 0.75 })
);
ball.visible = false;
scene.add(ball);

const rootOrientArrow = new THREE.ArrowHelper(
  new THREE.Vector3(1, 0, 0),
  new THREE.Vector3(0, 0, 0),
  0.28,
  0xf2c14e,
  0.075,
  0.04
);
rootOrientArrow.visible = false;
scene.add(rootOrientArrow);

const raycaster = new THREE.Raycaster();
const pointer = new THREE.Vector2();

function rawToThree(point) {
  return rawToThreeBase(point).sub(state.viewOffset);
}

function rawToThreeBase(point) {
  return new THREE.Vector3(point[0], point[2], -point[1]);
}

function rawVectorToThree(vector) {
  return new THREE.Vector3(vector.x, vector.z, -vector.y);
}

function formatNumber(value) {
  return Number(value).toFixed(3);
}

async function fetchJson(url) {
  const response = await fetch(url);
  const payload = await response.json();
  if (!response.ok) {
    throw new Error(payload.error || response.statusText);
  }
  return payload;
}

async function loadMotionList() {
  const payload = await fetchJson("/api/motions");
  state.motions = payload.motions;
  el.motionRoot.textContent = payload.root;
  el.motionSelect.innerHTML = "";
  for (const motion of state.motions) {
    const option = document.createElement("option");
    option.value = motion.path;
    option.textContent = motion.name;
    el.motionSelect.appendChild(option);
  }
  if (state.motions.length > 0) {
    const preferred = state.motions.find((motion) => motion.path.endsWith("walk_holdingBall/walk_x1.pkl"));
    el.motionSelect.value = preferred ? preferred.path : state.motions[0].path;
    await loadMotion(el.motionSelect.value);
  }
}

async function loadMotion(path) {
  state.playing = false;
  el.playButton.textContent = "Play";
  state.motion = await fetchJson(`/api/motion?path=${encodeURIComponent(path)}`);
  state.frame = 0;
  state.selectedJoint = Math.min(state.selectedJoint, state.motion.num_joints - 1);
  el.frameSlider.max = String(state.motion.num_frames - 1);
  el.frameSlider.value = "0";
  el.frameNumber.max = String(state.motion.num_frames - 1);
  el.frameNumber.value = "0";
  el.frameCount.textContent = String(state.motion.num_frames);
  el.fpsValue.textContent = String(state.motion.fps);
  el.orientationToggle.disabled = !state.motion.human_global_orient_quat_xyzw;
  if (!state.motion.human_global_orient_quat_xyzw) {
    el.orientationToggle.checked = false;
    if (el.curveModeSelect.value === "orient") {
      el.curveModeSelect.value = "joint";
    }
  }
  renderJointList();
  state.viewOffset.copy(computeFrameCenter(state.frame));
  updateFrame();
  fitCameraToFrame(state.frame);
}

function renderJointList() {
  el.jointList.innerHTML = "";
  const names = state.motion?.joint_names || [];
  names.forEach((name, index) => {
    const button = document.createElement("button");
    button.className = "joint-button";
    button.type = "button";
    button.textContent = `${index} ${name}`;
    button.addEventListener("click", () => selectJoint(index));
    el.jointList.appendChild(button);
  });
  updateJointButtons();
}

function updateJointButtons() {
  for (const button of el.jointList.children) {
    const active = Number(button.textContent.split(" ")[0]) === state.selectedJoint;
    button.classList.toggle("active", active);
  }
}

function selectJoint(index) {
  state.selectedJoint = index;
  updateJointButtons();
  updateFrame();
}

function setFrame(nextFrame) {
  const motion = state.motion;
  if (!motion) return;
  const parsed = Number(nextFrame);
  if (!Number.isFinite(parsed)) return;
  state.frame = Math.max(0, Math.min(Math.round(parsed), motion.num_frames - 1));
  state.frameCarry = 0;
  updateFrame();
}

function computeFrameCenter(frameIndex) {
  const motion = state.motion;
  const box = new THREE.Box3();
  const frame = Math.max(0, Math.min(frameIndex, motion.num_frames - 1));
  for (const point of motion.joints[frame]) {
    box.expandByPoint(rawToThreeBase(point));
  }
  const center = new THREE.Vector3();
  box.getCenter(center);
  return center;
}

function fitCameraToFrame(frameIndex) {
  const motion = state.motion;
  if (!motion) return;
  const box = new THREE.Box3();
  const frame = Math.max(0, Math.min(frameIndex, motion.num_frames - 1));
  for (const point of motion.joints[frame]) {
    box.expandByPoint(rawToThree(point));
  }
  const center = new THREE.Vector3();
  const size = new THREE.Vector3();
  box.getCenter(center);
  box.getSize(size);
  const radius = Math.max(size.x, size.y, size.z, 0.6);
  controls.target.copy(center);
  camera.position.copy(center).add(new THREE.Vector3(radius * 1.15, radius * 0.85, radius * 1.55));
  camera.near = 0.01;
  camera.far = Math.max(10, radius * 12);
  camera.updateProjectionMatrix();
  controls.update();
}

function updateFrame() {
  const motion = state.motion;
  if (!motion) return;
  const frame = Math.max(0, Math.min(state.frame, motion.num_frames - 1));
  state.frame = frame;
  el.frameSlider.value = String(frame);
  el.frameNumber.value = String(frame);
  el.frameLabel.textContent = `${frame} / ${motion.num_frames - 1}`;

  const points = motion.joints[frame];
  for (let i = 0; i < points.length; i += 1) {
    jointMeshes[i].position.copy(rawToThree(points[i]));
    jointMeshes[i].material = i === state.selectedJoint ? selectedMaterial : jointMaterial;
    jointMeshes[i].scale.setScalar(i === state.selectedJoint ? 1.55 : 1);
  }

  let offset = 0;
  for (const [a, b] of SMPL_BONES) {
    const pa = rawToThree(points[a]);
    const pb = rawToThree(points[b]);
    linePositions[offset++] = pa.x;
    linePositions[offset++] = pa.y;
    linePositions[offset++] = pa.z;
    linePositions[offset++] = pb.x;
    linePositions[offset++] = pb.y;
    linePositions[offset++] = pb.z;
  }
  lineGeometry.attributes.position.needsUpdate = true;

  const selected = points[state.selectedJoint];
  const selectedName = motion.joint_names[state.selectedJoint] || "joint";
  el.selectedJointName.textContent = `${state.selectedJoint} ${selectedName}`;
  el.selectedJointPosition.textContent = selected.map(formatNumber).join(", ");

  updateTrail();
  updateBall();
  updateRootOrientation();
  drawCurve();
}

function updateTrail() {
  const motion = state.motion;
  const enabled = el.trailToggle.checked && motion;
  trailLine.visible = Boolean(enabled);
  if (!enabled) return;
  const start = Math.max(0, state.frame - 120);
  const positions = [];
  for (let f = start; f <= state.frame; f += 1) {
    const point = rawToThree(motion.joints[f][state.selectedJoint]);
    positions.push(point.x, point.y, point.z);
  }
  trailGeometry.setAttribute("position", new THREE.Float32BufferAttribute(positions, 3));
  trailGeometry.computeBoundingSphere();
}

function updateBall() {
  const motion = state.motion;
  const visible = Boolean(el.ballToggle.checked && motion?.obj_pos);
  ball.visible = visible;
  if (visible) {
    ball.position.copy(rawToThree(motion.obj_pos[state.frame]));
  }
}

function updateRootOrientation() {
  const motion = state.motion;
  const visible = Boolean(el.orientationToggle.checked && motion?.human_global_orient_quat_xyzw);
  rootOrientArrow.visible = visible;
  if (!visible) return;

  const quatXyzw = motion.human_global_orient_quat_xyzw[state.frame];
  const quat = new THREE.Quaternion(quatXyzw[0], quatXyzw[1], quatXyzw[2], quatXyzw[3]).normalize();
  const rawForward = new THREE.Vector3(1, 0, 0).applyQuaternion(quat);
  const direction = rawVectorToThree(rawForward).normalize();
  const origin = rawToThree(motion.joints[state.frame][0]);
  rootOrientArrow.position.copy(origin);
  rootOrientArrow.setDirection(direction);
}

function drawCurve() {
  const motion = state.motion;
  const canvas = el.curveCanvas;
  const rect = canvas.getBoundingClientRect();
  const pixelRatio = Math.min(window.devicePixelRatio || 1, 2);
  const width = Math.max(1, Math.floor(rect.width * pixelRatio));
  const height = Math.max(1, Math.floor(rect.height * pixelRatio));
  if (canvas.width !== width || canvas.height !== height) {
    canvas.width = width;
    canvas.height = height;
  }
  const ctx = canvas.getContext("2d");
  ctx.clearRect(0, 0, width, height);
  ctx.fillStyle = "#101214";
  ctx.fillRect(0, 0, width, height);
  if (!motion) return;

  const padLeft = 54 * pixelRatio;
  const padRight = 18 * pixelRatio;
  const padTop = 24 * pixelRatio;
  const padBottom = 34 * pixelRatio;
  const plotW = width - padLeft - padRight;
  const plotH = height - padTop - padBottom;
  const curveMode = el.curveModeSelect.value;
  const curveIsOrient = curveMode === "orient" && motion.human_global_orient_quat;
  const series = curveIsOrient
    ? [0, 1, 2, 3].map((dim) => motion.human_global_orient_quat.map((quat) => quat[dim]))
    : [0, 1, 2].map((dim) => motion.joints.map((frame) => frame[state.selectedJoint][dim]));
  const values = series.flat();
  const min = Math.min(...values);
  const max = Math.max(...values);
  const span = Math.max(0.001, max - min);
  const yOf = (value) => padTop + (max - value) / span * plotH;
  const xOf = (frame) => padLeft + frame / Math.max(1, motion.num_frames - 1) * plotW;

  ctx.strokeStyle = "#343a40";
  ctx.lineWidth = pixelRatio;
  ctx.beginPath();
  for (let i = 0; i <= 4; i += 1) {
    const y = padTop + (plotH * i) / 4;
    ctx.moveTo(padLeft, y);
    ctx.lineTo(width - padRight, y);
  }
  ctx.stroke();

  const colors = ["#53a6ff", "#ff8f70", "#7ddc84", "#f2c14e"];
  const labels = curveIsOrient ? ["qw", "qx", "qy", "qz"] : ["x", "y", "z"];
  series.forEach((points, dim) => {
    ctx.strokeStyle = colors[dim];
    ctx.lineWidth = 1.6 * pixelRatio;
    ctx.beginPath();
    points.forEach((value, frame) => {
      const x = xOf(frame);
      const y = yOf(value);
      if (frame === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    });
    ctx.stroke();
  });

  const currentX = xOf(state.frame);
  ctx.strokeStyle = "#f2c14e";
  ctx.lineWidth = 1.2 * pixelRatio;
  ctx.beginPath();
  ctx.moveTo(currentX, padTop);
  ctx.lineTo(currentX, height - padBottom);
  ctx.stroke();

  ctx.fillStyle = "#edf0f2";
  ctx.font = `${12 * pixelRatio}px ui-sans-serif, system-ui`;
  const title = curveIsOrient
    ? "human_global_orient_quat raw WXYZ"
    : `${state.selectedJoint} ${motion.joint_names[state.selectedJoint]} raw XYZ`;
  ctx.fillText(title, padLeft, 16 * pixelRatio);
  ctx.fillStyle = "#a5adb5";
  ctx.fillText(max.toFixed(3), 8 * pixelRatio, padTop + 4 * pixelRatio);
  ctx.fillText(min.toFixed(3), 8 * pixelRatio, height - padBottom);
  labels.forEach((label, index) => {
    const x = width - padRight - (90 - index * 28) * pixelRatio;
    ctx.fillStyle = colors[index];
    ctx.fillText(label, x, 16 * pixelRatio);
  });
}

function onPointerDown(event) {
  const rect = renderer.domElement.getBoundingClientRect();
  pointer.x = ((event.clientX - rect.left) / rect.width) * 2 - 1;
  pointer.y = -((event.clientY - rect.top) / rect.height) * 2 + 1;
  raycaster.setFromCamera(pointer, camera);
  const hits = raycaster.intersectObjects(jointMeshes, false);
  if (hits.length > 0) {
    selectJoint(hits[0].object.userData.jointIndex);
  }
}

function resize() {
  const rect = el.viewer.getBoundingClientRect();
  const width = Math.max(1, rect.width);
  const height = Math.max(1, rect.height);
  camera.aspect = width / height;
  camera.updateProjectionMatrix();
  renderer.setSize(width, height, false);
  drawCurve();
}

function tick(time) {
  controls.update();
  if (state.playing && state.motion) {
    const elapsed = state.lastTick ? (time - state.lastTick) / 1000 : 0;
    const speed = Number(el.speedSelect.value);
    state.frameCarry += elapsed * state.motion.fps * speed;
    if (state.frameCarry >= 1) {
      const advance = Math.floor(state.frameCarry);
      state.frameCarry -= advance;
      state.frame = (state.frame + advance) % state.motion.num_frames;
      updateFrame();
    }
  }
  state.lastTick = time;
  renderer.render(scene, camera);
  requestAnimationFrame(tick);
}

el.motionSelect.addEventListener("change", () => loadMotion(el.motionSelect.value));
el.frameSlider.addEventListener("input", () => setFrame(el.frameSlider.value));
el.frameSlider.addEventListener("change", () => setFrame(el.frameSlider.value));
el.frameNumber.addEventListener("input", () => setFrame(el.frameNumber.value));
el.prevFrameButton.addEventListener("click", () => setFrame(state.frame - 1));
el.nextFrameButton.addEventListener("click", () => setFrame(state.frame + 1));
el.playButton.addEventListener("click", () => {
  state.playing = !state.playing;
  state.frameCarry = 0;
  el.playButton.textContent = state.playing ? "Pause" : "Play";
});
el.trailToggle.addEventListener("change", updateFrame);
el.ballToggle.addEventListener("change", updateFrame);
el.orientationToggle.addEventListener("change", updateFrame);
el.curveModeSelect.addEventListener("change", drawCurve);
renderer.domElement.addEventListener("pointerdown", onPointerDown);
window.addEventListener("resize", resize);

resize();
requestAnimationFrame(tick);
loadMotionList().catch((error) => {
  el.motionRoot.textContent = error.message;
  console.error(error);
});
