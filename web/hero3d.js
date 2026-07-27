// ─────────────────────────────────────────────────────────────
// AgroAI — 3D hero: a living, swaying crop field.
// Pure three.js (MIT). No external assets — geometry is generated.
// Theme-aware fog colour reads from the CSS custom property.
// ─────────────────────────────────────────────────────────────
import * as THREE from "three";

const mount = document.getElementById("hero-canvas");
const reduceMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;

/** Read --hero-fog CSS variable and convert to a THREE.Color hex int. */
function fogColorFromCSS() {
  const raw = getComputedStyle(document.documentElement)
    .getPropertyValue("--hero-fog").trim() || "#f6faf6";
  return new THREE.Color(raw);
}

if (mount) {
  const scene = new THREE.Scene();
  const fogCol = fogColorFromCSS();
  scene.fog = new THREE.Fog(fogCol, 12, 30);
  scene.background = fogCol.clone();

  const camera = new THREE.PerspectiveCamera(50, 1, 0.1, 100);
  camera.position.set(0, 3.6, 9);
  camera.lookAt(0, 1.1, 0);

  const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
  renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
  mount.appendChild(renderer.domElement);

  // ── Lighting ──
  scene.add(new THREE.AmbientLight(0x8fffc0, 0.55));
  const key = new THREE.DirectionalLight(0xffffff, 1.1);
  key.position.set(5, 8, 4);
  scene.add(key);
  const rim = new THREE.DirectionalLight(0xfbbf24, 0.5);
  rim.position.set(-6, 3, -4);
  scene.add(rim);

  // ── Ground (subtle wireframe relief) ──
  const groundGeo = new THREE.PlaneGeometry(40, 40, 24, 24);
  const ground = new THREE.Mesh(
    groundGeo,
    new THREE.MeshBasicMaterial({ color: 0x16a34a, wireframe: true, transparent: true, opacity: 0.06 })
  );
  ground.rotation.x = -Math.PI / 2;
  scene.add(ground);

  // ── Crop field (instanced cones, pivoting at the base) ──
  const N = 34, SPACING = 0.62;
  const blade = new THREE.ConeGeometry(0.07, 1, 5);
  blade.translate(0, 0.5, 0); // base at y=0 so it sways from the soil
  const mat = new THREE.MeshStandardMaterial({ roughness: 0.6, metalness: 0.0 });
  const field = new THREE.InstancedMesh(blade, mat, N * N);

  const dummy = new THREE.Object3D();
  const color = new THREE.Color();
  const cGreen = new THREE.Color(0x4ade80);
  const cAmber = new THREE.Color(0xfbbf24);
  const heights = new Float32Array(N * N);
  const phases = new Float32Array(N * N);
  let i = 0;
  const half = ((N - 1) * SPACING) / 2;
  for (let x = 0; x < N; x++) {
    for (let z = 0; z < N; z++) {
      const px = x * SPACING - half;
      const pz = z * SPACING - half;
      const h = 0.55 + Math.random() * 0.9;
      heights[i] = h;
      phases[i] = Math.random() * Math.PI * 2;
      dummy.position.set(px, 0, pz);
      dummy.scale.set(1, h, 1);
      dummy.updateMatrix();
      field.setMatrixAt(i, dummy.matrix);
      color.copy(cGreen).lerp(cAmber, Math.min(1, (h - 0.55) / 0.9) * 0.7 + Math.random() * 0.15);
      field.setColorAt(i, color);
      i++;
    }
  }
  scene.add(field);

  // ── Floating pollen particles ──
  const pCount = 220;
  const pPos = new Float32Array(pCount * 3);
  for (let p = 0; p < pCount; p++) {
    pPos[p * 3] = (Math.random() - 0.5) * 22;
    pPos[p * 3 + 1] = Math.random() * 7;
    pPos[p * 3 + 2] = (Math.random() - 0.5) * 22;
  }
  const pGeo = new THREE.BufferGeometry();
  pGeo.setAttribute("position", new THREE.BufferAttribute(pPos, 3));
  const pollen = new THREE.Points(
    pGeo,
    new THREE.PointsMaterial({ color: 0xfcd34d, size: 0.06, transparent: true, opacity: 0.7 })
  );
  scene.add(pollen);

  // ── Mouse parallax ──
  let mx = 0, my = 0;
  window.addEventListener("pointermove", (e) => {
    mx = (e.clientX / window.innerWidth - 0.5) * 2;
    my = (e.clientY / window.innerHeight - 0.5) * 2;
  });

  function resize() {
    const w = mount.clientWidth || window.innerWidth;
    const h = mount.clientHeight || window.innerHeight;
    renderer.setSize(w, h, false);
    camera.aspect = w / h;
    camera.updateProjectionMatrix();
  }
  window.addEventListener("resize", resize);
  resize();

  const swayM = new THREE.Object3D();
  function renderField(t) {
    for (let k = 0; k < N * N; k++) {
      const a = Math.sin(t * 1.4 + phases[k]) * 0.12;
      const b = Math.cos(t * 1.1 + phases[k]) * 0.08;
      const idxX = Math.floor(k / N), idxZ = k % N;
      swayM.position.set(idxX * SPACING - half, 0, idxZ * SPACING - half);
      swayM.rotation.set(b, 0, a);
      swayM.scale.set(1, heights[k], 1);
      swayM.updateMatrix();
      field.setMatrixAt(k, swayM.matrix);
    }
    field.instanceMatrix.needsUpdate = true;
  }

  let running = true;
  document.addEventListener("visibilitychange", () => { running = !document.hidden; });

  // ── Sync fog + background when theme changes ──
  const themeObs = new MutationObserver(() => {
    const c = fogColorFromCSS();
    scene.fog.color.copy(c);
    scene.background.copy(c);
  });
  themeObs.observe(document.documentElement, { attributes: true, attributeFilter: ["data-theme"] });

  const clock = new THREE.Clock();
  function loop() {
    if (running) {
      const t = clock.getElapsedTime();
      if (!reduceMotion) renderField(t);
      const orbit = reduceMotion ? 0 : t * 0.08;
      camera.position.x = Math.sin(orbit) * 9 + mx * 0.6;
      camera.position.z = Math.cos(orbit) * 9;
      camera.position.y = 3.6 - my * 0.4;
      camera.lookAt(0, 1.1, 0);
      pollen.rotation.y = t * 0.02;
      pollen.position.y = Math.sin(t * 0.3) * 0.3;
      renderer.render(scene, camera);
    }
    requestAnimationFrame(loop);
  }
  renderField(0);
  loop();
}
