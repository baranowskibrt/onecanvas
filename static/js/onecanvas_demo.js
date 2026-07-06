// Interactive panoramic-reprojection demo for the OneCanvas website.
//
// Mirrors the results-video glyph (panels_png/_video_scratch/segments/
// 08_results.mp4): a wireframe "canvas" globe sits at a chosen pose; every
// lifted patch feature reprojects onto it along a camera-colored spoke.
// Picking a question flies the globe to that situated viewpoint and re-renders
// the reprojection live, both in 3D and on the unrolled equirectangular strip.
//
// Scene assets are produced by scripts/export_web_scene.py and live under
// static/scenes/<id>/ (scene.json + cloud.bin). All coordinates are in the
// scene's axis-aligned WORLD frame (Z-up); the camera up-vector is set to +Z.

import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';

const SCENES_URL = 'static/scenes/index.json';
const SPHERE_COLOR = 0xb9bcc4;       // gray wireframe (185,188,196)
const AGENT_COLOR = 0xfdc100;        // gold ball (253,193,0)
const ARROW_COLOR = 0x18c850;        // green forward arrow
const FLY_DURATION = 0.7;            // seconds to glide between poses

let renderer, scene, camera, controls, clock;
let cloudPoints = null;
let gizmo = null;                    // rigid group: globe + ball + arrow
let spokes = null, sphereDots = null, endpointPoints = null;
let equirectCanvas = null, equirectCtx = null;

let sceneData = null;                // current scene.json + parsed cloud
let endpointsArr = null;             // Float32Array [3N]
let endpointColorsArr = null;        // Uint8Array [3N]
let zoomPending = 0;                 // accumulated log-zoom from the wheel, eased
                                     // out over a few frames in applyZoom()
let pose = { x: 0, y: 0, z: 0, yaw: 0 };      // current (animated) pose
let target = null;                   // pose we're flying toward
let flyT = 1;                        // 0..1 animation progress

// ---------------------------------------------------------------- bootstrap
export async function initDemo(mountId) {
  const mount = document.getElementById(mountId);
  if (!mount) return;
  const canvas = mount.querySelector('#demo-canvas');
  equirectCanvas = mount.querySelector('#demo-equirect');
  equirectCtx = equirectCanvas.getContext('2d');

  renderer = new THREE.WebGLRenderer({ canvas, antialias: true });
  renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
  scene = new THREE.Scene();
  scene.background = new THREE.Color(0x0a0a0f);
  camera = new THREE.PerspectiveCamera(45, 1, 0.05, 200);
  camera.up.set(0, 0, 1);            // world is Z-up
  controls = new OrbitControls(camera, renderer.domElement);
  controls.enableDamping = true;
  controls.dampingFactor = 0.08;
  controls.rotateSpeed = 0.7;
  controls.enablePan = false;
  // We drive zoom ourselves (see the 'wheel' handler + applyZoom). OrbitControls'
  // own wheel zoom scales the step by |event.deltaY|, which Chrome reports in
  // pixels (~100+/notch) while other browsers report small line-mode values --
  // so the same scroll jumps the whole range in Chrome but eases elsewhere.
  // Normalizing deltaMode + clamping per event makes it gradual everywhere.
  controls.enableZoom = false;
  controls.touches = { ONE: THREE.TOUCH.ROTATE, TWO: THREE.TOUCH.DOLLY_PAN };
  renderer.domElement.addEventListener('wheel', onWheelZoom, { passive: false });
  clock = new THREE.Clock();

  resize(mount);
  window.addEventListener('resize', () => resize(mount));

  const index = await fetch(SCENES_URL).then(r => r.json());
  buildSceneTabs(mount, index.scenes);
  await loadScene(index.scenes[0].id);

  animate();
}

function resize(mount) {
  const canvas = renderer.domElement;
  const w = canvas.clientWidth || mount.clientWidth;
  const h = canvas.clientHeight || 420;
  renderer.setSize(w, h, false);
  camera.aspect = w / h;
  camera.updateProjectionMatrix();
  // Equirect strip: keep crisp on HiDPI.
  const ew = equirectCanvas.clientWidth, eh = equirectCanvas.clientHeight;
  const dpr = Math.min(window.devicePixelRatio, 2);
  equirectCanvas.width = ew * dpr;
  equirectCanvas.height = eh * dpr;
  if (sceneData) drawEquirect();
}

// ----------------------------------------------------------------- scene load
async function loadScene(sceneId) {
  const base = `static/scenes/${sceneId}/`;
  const meta = await fetch(base + 'scene.json').then(r => r.json());
  const buf = await fetch(base + meta.cloud.url).then(r => r.arrayBuffer());
  sceneData = meta;

  // cloud.bin: uint32 magic, uint32 N, float32[3N] xyz, uint8[3N] rgb
  const dv = new DataView(buf);
  const n = dv.getUint32(4, true);
  const positions = new Float32Array(buf, 8, 3 * n);
  const colorsU8 = new Uint8Array(buf, 8 + 12 * n, 3 * n);

  endpointsArr = Float32Array.from(meta.endpoints.flat());
  endpointColorsArr = Uint8Array.from(meta.endpoint_colors.flat());

  clearScene();
  buildCloud(positions, colorsU8);
  buildGizmo(meta.sphere_radius);
  buildSpokes(meta.endpoints.length);
  buildQuestionButtons(meta.questions);

  // Frame the camera on the cloud bounds.
  const c = new THREE.Vector3(...meta.agent_center);
  let r = boundingRadius(positions, c);
  controls.target.copy(c);
  camera.position.copy(c).add(new THREE.Vector3(r * 0.9, -r * 1.1, r * 0.7));
  // Scroll-zoom bounds, scaled to this scene (start distance is ~1.6r). The
  // floor must clear the globe (radius meta.sphere_radius, a small constant ~
  // 0.45) so zooming in never buries the camera inside the sphere on small
  // scenes; the ceiling keeps the whole room in view without flying off.
  const globeR = meta.sphere_radius || 0.45;
  controls.minDistance = Math.max(r * 0.5, globeR * 3.0);
  controls.maxDistance = r * 4.0;
  controls.update();

  // Start at the first question pose.
  setPose(meta.questions[0], false);
  selectQuestion(0);
}

function boundingRadius(positions, center) {
  let m = 0;
  for (let i = 0; i < positions.length; i += 3) {
    const dx = positions[i] - center.x;
    const dy = positions[i + 1] - center.y;
    const dz = positions[i + 2] - center.z;
    m = Math.max(m, Math.hypot(dx, dy, dz));
  }
  return Math.max(2.5, m * 0.85);
}

function clearScene() {
  for (const o of [cloudPoints, gizmo, spokes, sphereDots, endpointPoints]) {
    if (o) { scene.remove(o); o.geometry?.dispose?.(); }
  }
  cloudPoints = gizmo = spokes = sphereDots = endpointPoints = null;
}

function buildCloud(positions, colorsU8) {
  const geo = new THREE.BufferGeometry();
  geo.setAttribute('position', new THREE.BufferAttribute(positions, 3));
  // ScanNet vertex colors are sRGB; three.js outputs sRGB and assumes
  // vertex colors are linear, so raw values come out too bright. Linearize.
  const col = new Float32Array(colorsU8.length);
  for (let i = 0; i < colorsU8.length; i++) {
    const s = colorsU8[i] / 255;
    col[i] = s * s;        // approx sRGB->linear (pow 2.2), cheap version
  }
  geo.setAttribute('color', new THREE.BufferAttribute(col, 3));
  // Opaque with depth-writing so the cloud reads as a solid scene, not a haze.
  // Splat size in world units (attenuated). Bumped to ~match the chunkier,
  // solid-surface look of the room cloud in the paper video; kept just under
  // the feature/sphere dots (0.11-0.13) so those landmarks still read on top.
  const mat = new THREE.PointsMaterial({
    size: 0.11, vertexColors: true, sizeAttenuation: true,
  });
  cloudPoints = new THREE.Points(geo, mat);
  cloudPoints.visible = document.getElementById('demo-show-cloud')?.checked ?? true;
  scene.add(cloudPoints);
}

function buildGizmo(radius) {
  gizmo = new THREE.Group();

  // Wireframe globe: lon meridians + lat rings (matches _pano_wireframe).
  const segPts = [];
  const N = 48;
  for (let lon = 0; lon < 360; lon += 30) {
    const a = lon * Math.PI / 180;
    let prev = null;
    for (let i = 0; i <= N; i++) {
      const t = (i / N) * Math.PI;
      const p = [radius * Math.sin(t) * Math.cos(a),
                 radius * Math.sin(t) * Math.sin(a),
                 radius * Math.cos(t)];
      if (prev) segPts.push(...prev, ...p);
      prev = p;
    }
  }
  for (let lat = -60; lat <= 60; lat += 30) {
    const b = lat * Math.PI / 180;
    const rxy = radius * Math.cos(b), z = radius * Math.sin(b);
    let prev = null;
    for (let i = 0; i <= N; i++) {
      const u = (i / N) * 2 * Math.PI;
      const p = [rxy * Math.cos(u), rxy * Math.sin(u), z];
      if (prev) segPts.push(...prev, ...p);
      prev = p;
    }
  }
  const wfGeo = new THREE.BufferGeometry();
  wfGeo.setAttribute('position',
    new THREE.BufferAttribute(Float32Array.from(segPts), 3));
  gizmo.add(new THREE.LineSegments(wfGeo,
    new THREE.LineBasicMaterial({ color: SPHERE_COLOR, transparent: true, opacity: 0.9 })));

  // Translucent glass fill.
  gizmo.add(new THREE.Mesh(
    new THREE.SphereGeometry(radius * 0.985, 32, 24),
    new THREE.MeshBasicMaterial({ color: SPHERE_COLOR, transparent: true,
      opacity: 0.07, depthWrite: false })));

  // Gold agent ball.
  gizmo.add(new THREE.Mesh(
    new THREE.SphereGeometry(0.12, 24, 18),
    new THREE.MeshBasicMaterial({ color: AGENT_COLOR })));

  // Forward arrow along local +X (group rotates by yaw about Z).
  const arrow = new THREE.Group();
  const len = 0.95, shaftLen = len - 0.26;
  const shaft = new THREE.Mesh(
    new THREE.CylinderGeometry(0.022, 0.022, shaftLen, 12),
    new THREE.MeshBasicMaterial({ color: ARROW_COLOR }));
  shaft.rotation.z = -Math.PI / 2;          // cylinder Y-axis -> +X
  shaft.position.x = shaftLen / 2;
  const cone = new THREE.Mesh(
    new THREE.ConeGeometry(0.075, 0.26, 16),
    new THREE.MeshBasicMaterial({ color: ARROW_COLOR }));
  cone.rotation.z = -Math.PI / 2;
  cone.position.x = shaftLen + 0.13;
  arrow.add(shaft); arrow.add(cone);
  gizmo.add(arrow);

  scene.add(gizmo);
}

function buildSpokes(nEndpoints) {
  // Spokes: 2 verts per endpoint (sphere surface -> 3D feature point).
  const spokeGeo = new THREE.BufferGeometry();
  spokeGeo.setAttribute('position',
    new THREE.BufferAttribute(new Float32Array(nEndpoints * 6), 3));
  const spokeCol = new Float32Array(nEndpoints * 6);
  for (let i = 0; i < nEndpoints; i++) {
    for (let v = 0; v < 2; v++) {
      spokeCol[i * 6 + v * 3] = endpointColorsArr[i * 3] / 255;
      spokeCol[i * 6 + v * 3 + 1] = endpointColorsArr[i * 3 + 1] / 255;
      spokeCol[i * 6 + v * 3 + 2] = endpointColorsArr[i * 3 + 2] / 255;
    }
  }
  spokeGeo.setAttribute('color', new THREE.BufferAttribute(spokeCol, 3));
  spokes = new THREE.LineSegments(spokeGeo,
    new THREE.LineBasicMaterial({ vertexColors: true, transparent: true, opacity: 0.8 }));
  spokes.frustumCulled = false;      // positions change every pose
  spokes.visible = document.getElementById('demo-show-rays')?.checked ?? true;
  scene.add(spokes);

  // Per-endpoint color buffer, reused by both point clouds below.
  const col = new Float32Array(nEndpoints * 3);
  for (let i = 0; i < nEndpoints * 3; i++) col[i] = endpointColorsArr[i] / 255;

  // Colored dot per endpoint where it meets the sphere surface (moves w/ pose).
  const dotGeo = new THREE.BufferGeometry();
  dotGeo.setAttribute('position',
    new THREE.BufferAttribute(new Float32Array(nEndpoints * 3), 3));
  dotGeo.setAttribute('color', new THREE.BufferAttribute(col.slice(), 3));
  sphereDots = new THREE.Points(dotGeo,
    new THREE.PointsMaterial({ size: 0.11, vertexColors: true, sizeAttenuation: true }));
  sphereDots.frustumCulled = false;  // bounding sphere goes stale as it moves
  scene.add(sphereDots);

  // The actual 3D feature locations ("points on the walls"), camera-colored.
  // Fixed in world space, so these stay put when the rays are hidden.
  const wallGeo = new THREE.BufferGeometry();
  wallGeo.setAttribute('position',
    new THREE.BufferAttribute(endpointsArr.slice(), 3));
  wallGeo.setAttribute('color', new THREE.BufferAttribute(col.slice(), 3));
  endpointPoints = new THREE.Points(wallGeo,
    new THREE.PointsMaterial({ size: 0.13, vertexColors: true, sizeAttenuation: true }));
  scene.add(endpointPoints);
}

// ----------------------------------------------------------- pose + reproject
function setPose(q, animate = true) {
  const tgt = { x: q.x, y: q.y, z: sceneData.agent_center[2],
                yaw: q.yaw_deg * Math.PI / 180 };
  if (!animate) {
    pose = { ...tgt }; target = null; flyT = 1; updatePose();
    return;
  }
  // short-arc yaw
  let dy = (tgt.yaw - pose.yaw) % (2 * Math.PI);
  if (dy > Math.PI) dy -= 2 * Math.PI;
  if (dy < -Math.PI) dy += 2 * Math.PI;
  target = { from: { ...pose }, to: tgt, dyaw: dy };
  flyT = 0;
}

function easeInOut(t) { return t < 0.5 ? 2 * t * t : 1 - Math.pow(-2 * t + 2, 2) / 2; }

function updateFly(dt) {
  if (!target) return;
  flyT = Math.min(1, flyT + dt / FLY_DURATION);
  const e = easeInOut(flyT);
  const f = target.from, to = target.to;
  pose.x = f.x + (to.x - f.x) * e;
  pose.y = f.y + (to.y - f.y) * e;
  pose.z = f.z + (to.z - f.z) * e;
  pose.yaw = f.yaw + target.dyaw * e;
  updatePose();
  if (flyT >= 1) target = null;
}

function updatePose() {
  const c = new THREE.Vector3(pose.x, pose.y, pose.z);
  gizmo.position.copy(c);
  gizmo.rotation.z = pose.yaw;
  // Keep the globe centered in view as it glides (snap when not animating).
  controls.target.lerp(c, target ? 0.35 : 1);

  const R = sceneData.sphere_radius;
  const sp = spokes.geometry.attributes.position.array;
  const dp = sphereDots.geometry.attributes.position.array;
  const n = endpointsArr.length / 3;
  for (let i = 0; i < n; i++) {
    const ex = endpointsArr[i * 3], ey = endpointsArr[i * 3 + 1], ez = endpointsArr[i * 3 + 2];
    let dx = ex - c.x, dy = ey - c.y, dz = ez - c.z;
    const inv = 1 / (Math.hypot(dx, dy, dz) || 1e-8);
    const sx = c.x + dx * inv * R, sy = c.y + dy * inv * R, sz = c.z + dz * inv * R;
    sp[i * 6] = sx; sp[i * 6 + 1] = sy; sp[i * 6 + 2] = sz;
    sp[i * 6 + 3] = ex; sp[i * 6 + 4] = ey; sp[i * 6 + 5] = ez;
    dp[i * 3] = sx; dp[i * 3 + 1] = sy; dp[i * 3 + 2] = sz;
  }
  spokes.geometry.attributes.position.needsUpdate = true;
  sphereDots.geometry.attributes.position.needsUpdate = true;
  drawEquirect();
}

// ---------------------------------------------------------- equirect strip
function drawEquirect() {
  const ctx = equirectCtx, W = equirectCanvas.width, H = equirectCanvas.height;
  ctx.clearRect(0, 0, W, H);
  ctx.fillStyle = '#ffffff';
  ctx.fillRect(0, 0, W, H);

  const c = pose;
  // Center the strip on the current heading so "forward" is in the middle.
  const headLon = Math.atan2(Math.cos(c.yaw), Math.sin(c.yaw)) * 180 / Math.PI;
  const LAT_SPAN = 90;                 // full equirectangular: +/-90 deg latitude
  const wrap = d => (((d + 180) % 360) + 360) % 360 - 180;
  const toXY = (lon, lat) => [
    ((wrap(lon - headLon) + 180) / 360) * W,
    ((LAT_SPAN - lat) / (2 * LAT_SPAN)) * H,
  ];

  // Faint lat/lon grid.
  ctx.strokeStyle = 'rgba(185,188,196,0.9)';
  ctx.lineWidth = Math.max(1, W / 800);
  for (let g = -180; g <= 180; g += 30) {
    const x = ((wrap(g) + 180) / 360) * W;
    ctx.beginPath(); ctx.moveTo(x, 0); ctx.lineTo(x, H); ctx.stroke();
  }
  for (let g = -60; g <= 60; g += 30) {
    const y = ((LAT_SPAN - g) / (2 * LAT_SPAN)) * H;
    ctx.beginPath(); ctx.moveTo(0, y); ctx.lineTo(W, y); ctx.stroke();
  }
  // Forward marker (vertical) + horizon marker (horizontal lat=0) form a
  // small green crosshair the agent's eye-level forward direction.
  ctx.strokeStyle = 'rgba(24,200,80,0.85)';
  ctx.lineWidth = Math.max(1.5, W / 500);
  ctx.beginPath(); ctx.moveTo(W / 2, 0); ctx.lineTo(W / 2, H); ctx.stroke();
  ctx.beginPath(); ctx.moveTo(0, H / 2); ctx.lineTo(W, H / 2); ctx.stroke();

  // One colored disk per feature endpoint at its (lon, lat) from this pose.
  const r = Math.max(3, W / 220);
  const n = endpointsArr.length / 3;
  for (let i = 0; i < n; i++) {
    const dx = endpointsArr[i * 3] - c.x;
    const dy = endpointsArr[i * 3 + 1] - c.y;
    const dz = endpointsArr[i * 3 + 2] - c.z;
    const inv = 1 / (Math.hypot(dx, dy, dz) || 1e-8);
    const lon = Math.atan2(dx * inv, dy * inv) * 180 / Math.PI;
    const lat = Math.asin(Math.max(-1, Math.min(1, dz * inv))) * 180 / Math.PI;
    const [x, y] = toXY(lon, lat);
    const cr = endpointColorsArr[i * 3], cg = endpointColorsArr[i * 3 + 1], cb = endpointColorsArr[i * 3 + 2];
    ctx.beginPath();
    ctx.fillStyle = `rgb(${cr},${cg},${cb})`;
    ctx.strokeStyle = `rgb(${cr * 0.55 | 0},${cg * 0.55 | 0},${cb * 0.55 | 0})`;
    ctx.lineWidth = Math.max(1, W / 900);
    ctx.arc(x, y, r, 0, 2 * Math.PI);
    ctx.fill(); ctx.stroke();
  }
}

// ------------------------------------------------------------------- UI
function buildSceneTabs(mount, scenes) {
  const ul = mount.querySelector('#demo-scene-tabs');
  if (!ul) return;
  ul.innerHTML = '';
  if (scenes.length <= 1) { ul.style.display = 'none'; return; }
  scenes.forEach((s, i) => {
    const li = document.createElement('li'); li.className = 'nav-item';
    const a = document.createElement('a');
    a.className = 'nav-link' + (i === 0 ? ' active' : '');
    a.textContent = s.title || s.id;
    a.onclick = async () => {
      ul.querySelectorAll('.nav-link').forEach((b, j) =>
        b.className = 'nav-link' + (j === i ? ' active' : ''));
      await loadScene(s.id);
    };
    li.appendChild(a); ul.appendChild(li);
  });
}

let questionEls = [];
function buildQuestionButtons(questions) {
  const ul = document.getElementById('demo-question-tabs');
  ul.innerHTML = ''; questionEls = [];
  questions.forEach((q, i) => {
    const li = document.createElement('li'); li.className = 'nav-item';
    const a = document.createElement('a');
    a.className = 'nav-link' + (i === 0 ? ' active' : '');
    a.textContent = q.question;
    a.onclick = () => { setPose(q); selectQuestion(i); };
    li.appendChild(a); ul.appendChild(li); questionEls.push(a);
  });
}

function selectQuestion(idx) {
  questionEls.forEach((a, i) =>
    a.className = 'nav-link' + (i === idx ? ' active' : ''));
  const q = sceneData.questions[idx];
  const ans = document.getElementById('demo-answer');
  if (ans) {
    ans.innerHTML = q.answer
      ? `<span class="has-text-grey">Model answer:</span> <b>${q.answer}</b>`
      : '';
  }
  const situ = document.getElementById('demo-situation');
  if (situ) {
    situ.innerHTML = q.situation
      ? `<span class="has-text-grey">Situation:</span> <i>${q.situation}</i>`
      : '';
  }
}

// ------------------------------------------------------------------- loop
// Custom wheel zoom: normalize across browsers (deltaMode + magnitude vary
// wildly -- Chrome pixels vs others line/page), clamp each event's contribution,
// and fold it into zoomPending as a log-zoom. applyZoom() then eases it out over
// a few frames, so a flick still reaches the clamp but a nudge dollies smoothly.
function onWheelZoom(e) {
  e.preventDefault();                                   // claim the wheel; don't scroll the page
  const unit = e.deltaMode === 1 ? 16                   // lines  -> ~16 px
             : e.deltaMode === 2 ? (renderer.domElement.clientHeight || 420)  // pages
             : 1;                                        // already pixels
  const px = Math.max(-120, Math.min(120, e.deltaY * unit));  // clamp one event to [-120,120] px
  zoomPending += (px / 120) * 0.12;                     // <=0.12 log-units/event; down = zoom out
}

// Ease the pending wheel zoom into the camera distance, frame-rate independent.
// Dolly is RELATIVE (multiplies the current distance), so it layers cleanly on
// top of the pose-fly without fighting it, and stays within the per-scene clamps.
function applyZoom(dt) {
  if (!controls || Math.abs(zoomPending) < 1e-4) { zoomPending = 0; return; }
  const ease = zoomPending * Math.min(1, dt * 12);      // apply a fraction this frame
  zoomPending -= ease;
  const offset = camera.position.clone().sub(controls.target);
  const dist = Math.max(controls.minDistance,
                        Math.min(controls.maxDistance, offset.length() * Math.exp(ease)));
  offset.setLength(dist);
  camera.position.copy(controls.target).add(offset);
}

function animate() {
  requestAnimationFrame(animate);
  const dt = clock.getDelta();
  updateFly(dt);
  applyZoom(dt);
  controls.update();
  renderer.render(scene, camera);
}

document.addEventListener('DOMContentLoaded', () => {
  const showCloud = document.getElementById('demo-show-cloud');
  if (showCloud) showCloud.addEventListener('change',
    () => { if (cloudPoints) cloudPoints.visible = showCloud.checked; });
  const showRays = document.getElementById('demo-show-rays');
  if (showRays) showRays.addEventListener('change',
    () => { if (spokes) spokes.visible = showRays.checked; });
  initDemo('demo-root').catch(err => {
    console.error(err);
    const m = document.getElementById('demo-root');
    if (m) m.insertAdjacentHTML('beforeend',
      `<p class="has-text-danger">Interactive demo failed to load: ${err}</p>`);
  });
});
