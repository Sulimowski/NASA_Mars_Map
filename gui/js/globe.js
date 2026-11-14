// Mars globe: textured planet, thin dusty atmosphere, stars, location pins with HTML labels.
// Touch first: one finger turns the planet, two fingers zoom, a tap on a pin selects it.

import * as THREE from "three";

const DEG = Math.PI / 180;
const FOV = 32;
const SUN = new THREE.Vector3(-0.55, 0.42, 0.72).normalize();

// Same orientation as three.js SphereGeometry UVs with an equirectangular map whose left edge is
// longitude -180: longitude 0 faces +X, -90 faces +Z, north is +Y.
export function latLonToVec3(lat, lon, r = 1) {
  const la = lat * DEG;
  const ph = (lon + 180) * DEG;
  return new THREE.Vector3(-Math.cos(ph) * Math.cos(la) * r, Math.sin(la) * r, Math.sin(ph) * Math.cos(la) * r);
}

const wrapAngle = (a) => ((((a + Math.PI) % (2 * Math.PI)) + 2 * Math.PI) % (2 * Math.PI)) - Math.PI;
const easeInOut = (t) => (t < 0.5 ? 4 * t * t * t : 1 - (-2 * t + 2) ** 3 / 2);
const smooth = (a, b, x) => Math.min(1, Math.max(0, (x - a) / (b - a)));

const PLANET_VS = /* glsl */ `
  varying vec2 vUv; varying vec3 vN; varying vec3 vP;
  void main() {
    vUv = uv;
    vN = normalize(mat3(modelMatrix) * normal);
    vec4 wp = modelMatrix * vec4(position, 1.0);
    vP = wp.xyz;
    gl_Position = projectionMatrix * viewMatrix * wp;
  }`;

const PLANET_FS = /* glsl */ `
  uniform sampler2D map; uniform vec3 sunDir; uniform float fade;
  varying vec2 vUv; varying vec3 vN; varying vec3 vP;
  void main() {
    vec3 n = normalize(vN);
    vec3 v = normalize(cameraPosition - vP);
    vec3 c = texture2D(map, vUv).rgb;
    c = mix(vec3(dot(c, vec3(0.299, 0.587, 0.114))), c, 1.1);
    float day = smoothstep(-0.28, 0.5, dot(n, sunDir));
    vec3 col = c * (0.045 + 1.12 * day);
    float rim = pow(1.0 - max(dot(n, v), 0.0), 2.6);
    col += vec3(0.95, 0.6, 0.4) * rim * (0.03 + 0.2 * day);
    gl_FragColor = vec4(col * fade, 1.0);
  }`;

const ATMO_FS = /* glsl */ `
  uniform vec3 sunDir; uniform float fade;
  varying vec3 vN; varying vec3 vP;
  void main() {
    vec3 n = normalize(vN);
    vec3 v = normalize(cameraPosition - vP);
    float t = clamp(-dot(n, v) / 0.34, 0.0, 1.0);
    float glow = pow(t, 3.2);
    float lit = 0.25 + 0.75 * smoothstep(-0.4, 0.6, dot(n, sunDir));
    gl_FragColor = vec4(vec3(0.98, 0.58, 0.36) * glow * lit * 0.3 * fade, 1.0);
  }`;

export class Globe {
  constructor(canvas, labelLayer, { texture, onSelect, onReady, freeArea }) {
    this.canvas = canvas;
    this.labelLayer = labelLayer;
    this.onSelect = onSelect || (() => {});
    this.freeArea = freeArea || (() => ({ left: 0, top: 0, right: innerWidth, bottom: innerHeight }));

    this.renderer = new THREE.WebGLRenderer({ canvas, antialias: true });
    this.renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
    this.renderer.setClearColor(0x050505, 1);
    this.scene = new THREE.Scene();
    this.camera = new THREE.PerspectiveCamera(FOV, 1, 0.05, 200);

    this.planet = new THREE.Group();
    this.planet.rotation.order = "XYZ";          // yaw about Y first, then pitch about X
    this.scene.add(this.planet);

    this.yaw = 0;
    this.pitch = 0.18;
    this.zoom = 1;
    this.targetZoom = 1;
    this.vel = { yaw: 0, pitch: 0 };
    this.anim = null;
    this.selected = null;
    this.pins = [];
    this.features = [];
    this.view ={ offX: 0, offY: 0, dist: 4.5 };
    this.viewTarget = { ...this.view };
    this.fade = { value: 0 };
    this.clock = new THREE.Clock();
    this.lastInput = -1e9;

    this.#buildStars();
    this.#buildPlanet(texture, onReady);
    this.#bindInput();
    addEventListener("resize", () => this.resize());
    this.resize(true);
    this.renderer.setAnimationLoop(() => this.#tick());
  }

  // ---------- scene ----------

  #buildStars() {
    const n = 2200;
    const pos = new Float32Array(n * 3);
    const col = new Float32Array(n * 3);
    for (let i = 0; i < n; i++) {
      const v = new THREE.Vector3().randomDirection().multiplyScalar(80);
      pos.set([v.x, v.y, v.z], i * 3);
      const b = 0.25 + Math.random() ** 3 * 0.75;
      col.set([b, b * 0.96, b * 0.9], i * 3);
    }
    const g = new THREE.BufferGeometry();
    g.setAttribute("position", new THREE.BufferAttribute(pos, 3));
    g.setAttribute("color", new THREE.BufferAttribute(col, 3));
    const m = new THREE.PointsMaterial({ size: 1.5, sizeAttenuation: false, vertexColors: true, transparent: true });
    this.stars = new THREE.Points(g, m);
    this.scene.add(this.stars);
  }

  #buildPlanet(url, onReady) {
    const tex = new THREE.TextureLoader().load(url, () => {
      this.fadeStart = this.clock.elapsedTime;
      onReady?.();
    });
    tex.anisotropy = this.renderer.capabilities.getMaxAnisotropy();

    const planet = new THREE.Mesh(
      new THREE.SphereGeometry(1, 160, 80),
      new THREE.ShaderMaterial({
        uniforms: { map: { value: tex }, sunDir: { value: SUN }, fade: this.fade },
        vertexShader: PLANET_VS,
        fragmentShader: PLANET_FS,
      }),
    );
    const atmo = new THREE.Mesh(
      new THREE.SphereGeometry(1.06, 96, 48),
      new THREE.ShaderMaterial({
        uniforms: { sunDir: { value: SUN }, fade: this.fade },
        vertexShader: PLANET_VS,
        fragmentShader: ATMO_FS,
        side: THREE.BackSide,
        blending: THREE.AdditiveBlending,
        transparent: true,
        depthWrite: false,
      }),
    );
    this.planetMesh = planet;
    this.planet.add(planet);
    this.scene.add(atmo);                        // a sphere: no need to turn with the planet
  }

  addPin(loc) {
    const local = latLonToVec3(loc.lat, loc.lon, 1.002);
    const group = new THREE.Group();
    group.position.copy(local);
    this.planet.updateMatrixWorld(true);
    group.lookAt(this.planet.localToWorld(local.clone().multiplyScalar(2)));

    // Only an invisible tap target lives in 3D; the mark itself is drawn in HTML so it stays crisp.
    const hit = new THREE.Mesh(new THREE.CircleGeometry(0.05, 16), new THREE.MeshBasicMaterial({ visible: false }));
    hit.userData.id = loc.id;
    group.add(hit);
    this.planet.add(group);

    const label = document.createElement("button");
    label.className = "pin";
    label.setAttribute("aria-label", `${loc.name}, ${loc.place}`);
    label.innerHTML = `<span class="pin-mark"></span><span class="pin-name"></span>`;
    label.querySelector(".pin-name").textContent = loc.name;
    label.addEventListener("click", () => this.onSelect(loc.id));
    this.labelLayer.append(label);

    this.pins.push({ loc, group, hit, label });
    if (this.pins.length === 1) this.yaw = this.#yawFor(loc) - 1.1;   // first pin turns into view
  }

  // A name of a large landform, for orientation. Not tappable.
  addFeature({ name, lat, lon }) {
    const el = document.createElement("div");
    el.className = "feat";
    el.append(document.createElement("span"));
    el.firstChild.textContent = name;
    this.labelLayer.prepend(el);
    this.features.push({ local: latLonToVec3(lat, lon, 1.002), el });
  }

  // ---------- camera ----------

  #yawFor(loc) {
    const p = latLonToVec3(loc.lat, loc.lon);
    return Math.atan2(-p.x, p.z);
  }

  focus(id) {
    const pin = this.pins.find((p) => p.loc.id === id);
    if (!pin) return;
    this.selected = id;
    const yaw = this.yaw + wrapAngle(this.#yawFor(pin.loc) - this.yaw);
    this.#animate({ yaw, pitch: pin.loc.lat * DEG, zoom: 0.78 });
    this.pins.forEach((p) => p.label.classList.toggle("on", p === pin));
  }

  release() {
    this.selected = null;
    this.#animate({ yaw: this.yaw, pitch: this.pitch, zoom: 1 }, 1.0);
    this.pins.forEach((p) => p.label.classList.remove("on"));
  }

  #animate(to, dur = 1.5) {
    this.vel.yaw = this.vel.pitch = 0;
    this.anim = { from: { yaw: this.yaw, pitch: this.pitch, zoom: this.zoom }, to, t0: this.clock.elapsedTime, dur };
  }

  resize(instant = false) {
    const W = innerWidth;
    const H = innerHeight;
    this.renderer.setSize(W, H, false);
    this.camera.aspect = W / H;
    const a = this.freeArea();
    const fw = Math.max(200, a.right - a.left);
    const fh = Math.max(200, a.bottom - a.top);
    const diameter = 0.8 * Math.min(fw, fh);
    this.viewTarget = {
      offX: (a.left + a.right) / 2 - W / 2,
      offY: (a.top + a.bottom) / 2 - H / 2,
      dist: H / (diameter * Math.tan((FOV / 2) * DEG)),
    };
    if (instant) this.view = { ...this.viewTarget };
    this.#applyView();
  }

  #applyView() {
    const W = innerWidth;
    const H = innerHeight;
    this.camera.setViewOffset(W, H, -this.view.offX, -this.view.offY, W, H);
    this.camera.position.set(0, 0, this.view.dist * this.zoom);
    this.camera.lookAt(0, 0, 0);
    this.camera.updateProjectionMatrix();
  }

  // ---------- input ----------

  #bindInput() {
    const el = this.canvas;
    const pts = new Map();
    let moved = 0;
    let pinchDist = 0;
    let lastT = 0;

    const spread = () => {
      const [a, b] = [...pts.values()];
      return Math.hypot(a.x - b.x, a.y - b.y);
    };

    el.addEventListener("pointerdown", (e) => {
      el.setPointerCapture(e.pointerId);
      pts.set(e.pointerId, { x: e.clientX, y: e.clientY });
      this.lastInput = this.clock.elapsedTime;
      this.anim = null;
      this.vel.yaw = this.vel.pitch = 0;
      moved = pts.size > 1 ? 99 : 0;
      if (pts.size === 2) pinchDist = spread();
      lastT = performance.now();
      this.canvas.dispatchEvent(new CustomEvent("globe-touch", { bubbles: true }));
    });

    el.addEventListener("pointermove", (e) => {
      const p = pts.get(e.pointerId);
      if (!p) return;
      const dx = e.clientX - p.x;
      const dy = e.clientY - p.y;
      p.x = e.clientX;
      p.y = e.clientY;
      this.lastInput = this.clock.elapsedTime;
      if (pts.size === 1) {
        const k = 0.0048 * this.zoom;
        const now = performance.now();
        const dt = Math.max(1, now - lastT) / 1000;
        lastT = now;
        this.yaw += dx * k;
        this.pitch = THREE.MathUtils.clamp(this.pitch + dy * k, -1.3, 1.3);
        this.vel.yaw = (dx * k) / dt;
        this.vel.pitch = (dy * k) / dt;
        moved += Math.abs(dx) + Math.abs(dy);
      } else if (pts.size === 2) {
        const d = spread();
        if (pinchDist > 0) this.#zoomBy(pinchDist / d);
        pinchDist = d;
      }
    });

    const up = (e) => {
      if (!pts.has(e.pointerId)) return;
      pts.delete(e.pointerId);
      if (pts.size === 0 && moved < 8) {
        this.vel.yaw = this.vel.pitch = 0;
        this.#tap(e.clientX, e.clientY);
      }
      if (performance.now() - lastT > 80) this.vel.yaw = this.vel.pitch = 0;   // held still before lifting
      if (pts.size < 2) pinchDist = 0;
    };
    el.addEventListener("pointerup", up);
    el.addEventListener("pointercancel", up);

    el.addEventListener("wheel", (e) => {
      e.preventDefault();
      this.anim = null;
      this.lastInput = this.clock.elapsedTime;
      this.#zoomBy(Math.exp(e.deltaY * 0.0012));
    }, { passive: false });
  }

  #zoomBy(f) {
    this.targetZoom = THREE.MathUtils.clamp(this.targetZoom * f, 0.42, 1.6);
  }

  #tap(x, y) {
    const ndc = new THREE.Vector2((x / innerWidth) * 2 - 1, -(y / innerHeight) * 2 + 1);
    const ray = new THREE.Raycaster();
    ray.setFromCamera(ndc, this.camera);
    const hits = ray.intersectObjects([this.planetMesh, ...this.pins.map((p) => p.hit)]);
    const pin = hits.find((h) => h.object.userData.id);
    if (pin && hits[0].object !== this.planetMesh) this.onSelect(pin.object.userData.id);
  }

  // ---------- frame ----------

  #tick() {
    const dt = Math.min(this.clock.getDelta(), 0.05);
    const t = this.clock.elapsedTime;

    if (this.fadeStart !== undefined) this.fade.value = easeInOut(smooth(0, 1.6, t - this.fadeStart));

    if (this.anim) {
      const { from, to, t0, dur } = this.anim;
      const e = easeInOut(Math.min(1, (t - t0) / dur));
      this.yaw = from.yaw + (to.yaw - from.yaw) * e;
      this.pitch = from.pitch + (to.pitch - from.pitch) * e;
      this.zoom = this.targetZoom = from.zoom + (to.zoom - from.zoom) * e;
      if (e >= 1) this.anim = null;
    } else {
      this.yaw += this.vel.yaw * dt;
      this.pitch = THREE.MathUtils.clamp(this.pitch + this.vel.pitch * dt, -1.3, 1.3);
      const decay = Math.exp(-dt * 3.5);
      this.vel.yaw *= decay;
      this.vel.pitch *= decay;
      const idle = smooth(5, 8, t - this.lastInput);
      if (!this.selected) this.yaw += 0.045 * idle * dt;
      this.zoom += (this.targetZoom - this.zoom) * Math.min(1, dt * 9);
    }

    const k = Math.min(1, dt * 6);
    for (const key of ["offX", "offY", "dist"]) this.view[key] += (this.viewTarget[key] - this.view[key]) * k;

    this.planet.rotation.set(this.pitch, this.yaw, 0);
    this.#applyView();
    this.stars.rotation.set(this.pitch * 0.08, this.yaw * 0.08, 0);

    const scale = this.zoom * 0.9 + 0.1;
    for (const p of this.pins) p.group.scale.setScalar(scale);

    this.renderer.render(this.scene, this.camera);
    this.#placeLabels();
  }

  #placeLabels() {
    const W = innerWidth;
    const H = innerHeight;
    const world = new THREE.Vector3();
    const toCam = new THREE.Vector3();
    const normal = new THREE.Vector3();
    // Puts an element over the point in `world` and fades it out as the point turns away.
    const place = (el, from, to) => {
      const facing = toCam.copy(this.camera.position).sub(world).normalize().dot(normal.copy(world).normalize());
      const o = smooth(from, to, facing) * this.fade.value;
      world.project(this.camera);
      const x = (world.x * 0.5 + 0.5) * W;
      const y = (-world.y * 0.5 + 0.5) * H;
      el.style.transform = `translate3d(${x.toFixed(1)}px, ${y.toFixed(1)}px, 0)`;
      el.style.opacity = o.toFixed(3);
      return o;
    };
    for (const p of this.pins) {
      p.group.getWorldPosition(world);
      p.label.style.pointerEvents = place(p.label, 0.05, 0.3) > 0.5 ? "auto" : "none";
    }
    for (const f of this.features) {
      this.planet.localToWorld(world.copy(f.local));
      place(f.el, 0.3, 0.65);
    }
  }
}
