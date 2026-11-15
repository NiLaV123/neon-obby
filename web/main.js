import { RASTER_WGSL, TRACE_WGSL, PT_TEMPORAL_WGSL, PT_PREPASS_WGSL, PT_RESAMPLE_WGSL, PT_DUP_WGSL, DN_ACCUM_WGSL, DN_ATROUS_WGSL, SHADE_WGSL, POST_WGSL } from './shaders.js';
import { SoundLibrary, MovementSounds } from './audio.js';

const $ = (id) => document.getElementById(id);
const canvas = $('c');

// ------------------------------------------------------------- mat4 (column-major)
const m4 = {
  mul(a, b) {
    const o = new Float32Array(16);
    for (let c = 0; c < 4; c++)
      for (let r = 0; r < 4; r++)
        o[c * 4 + r] = a[r] * b[c * 4] + a[4 + r] * b[c * 4 + 1] + a[8 + r] * b[c * 4 + 2] + a[12 + r] * b[c * 4 + 3];
    return o;
  },
  // reversed-Z, infinite far plane: depth = near / viewZ
  persp(fovy, aspect, near) {
    const f = 1 / Math.tan(fovy / 2), o = new Float32Array(16);
    o[0] = f / aspect; o[5] = f; o[11] = -1; o[14] = near;
    return o;
  },
  lookAt(e, t, up) {
    let zx = e[0] - t[0], zy = e[1] - t[1], zz = e[2] - t[2];
    let l = Math.hypot(zx, zy, zz) || 1; zx /= l; zy /= l; zz /= l;
    let xx = up[1] * zz - up[2] * zy, xy = up[2] * zx - up[0] * zz, xz = up[0] * zy - up[1] * zx;
    l = Math.hypot(xx, xy, xz) || 1; xx /= l; xy /= l; xz /= l;
    const yx = zy * xz - zz * xy, yy = zz * xx - zx * xz, yz = zx * xy - zy * xx;
    return new Float32Array([
      xx, yx, zx, 0, xy, yy, zy, 0, xz, yz, zz, 0,
      -(xx * e[0] + xy * e[1] + xz * e[2]), -(yx * e[0] + yy * e[1] + yz * e[2]), -(zx * e[0] + zy * e[1] + zz * e[2]), 1,
    ]);
  },
  inv(m) {
    const [a00, a01, a02, a03, a10, a11, a12, a13, a20, a21, a22, a23, a30, a31, a32, a33] = m;
    const b00 = a00 * a11 - a01 * a10, b01 = a00 * a12 - a02 * a10, b02 = a00 * a13 - a03 * a10;
    const b03 = a01 * a12 - a02 * a11, b04 = a01 * a13 - a03 * a11, b05 = a02 * a13 - a03 * a12;
    const b06 = a20 * a31 - a21 * a30, b07 = a20 * a32 - a22 * a30, b08 = a20 * a33 - a23 * a30;
    const b09 = a21 * a32 - a22 * a31, b10 = a21 * a33 - a23 * a31, b11 = a22 * a33 - a23 * a32;
    const det = 1 / (b00 * b11 - b01 * b10 + b02 * b09 + b03 * b08 - b04 * b07 + b05 * b06);
    return new Float32Array([
      (a11 * b11 - a12 * b10 + a13 * b09) * det, (a02 * b10 - a01 * b11 - a03 * b09) * det,
      (a31 * b05 - a32 * b04 + a33 * b03) * det, (a22 * b04 - a21 * b05 - a23 * b03) * det,
      (a12 * b08 - a10 * b11 - a13 * b07) * det, (a00 * b11 - a02 * b08 + a03 * b07) * det,
      (a32 * b02 - a30 * b05 - a33 * b01) * det, (a20 * b05 - a22 * b02 + a23 * b01) * det,
      (a10 * b10 - a11 * b08 + a13 * b06) * det, (a01 * b08 - a00 * b10 - a03 * b06) * det,
      (a30 * b04 - a31 * b02 + a33 * b00) * det, (a21 * b02 - a20 * b04 - a23 * b00) * det,
      (a11 * b07 - a10 * b09 - a12 * b06) * det, (a00 * b09 - a01 * b07 + a02 * b06) * det,
      (a31 * b01 - a30 * b03 - a32 * b00) * det, (a20 * b03 - a21 * b01 + a22 * b00) * det,
    ]);
  },
};

// ------------------------------------------------------------- sound controls
const sounds = new SoundLibrary();
const movementSounds = new MovementSounds(sounds);
function syncSoundControls() {
  $('mute').textContent = sounds.muted ? 'Sound OFF' : 'Sound ON';
  $('mute').setAttribute('aria-pressed', String(sounds.muted));
  $('volume').value = Math.round(sounds.volume * 100);
}
function toggleSound() {
  sounds.setMuted(!sounds.muted);
  syncSoundControls();
  if (!sounds.muted) void sounds.unlock().then(() => sounds.play('toggle'));
}
$('mute').addEventListener('click', toggleSound);
$('volume').addEventListener('input', (e) => {
  sounds.setVolume(Number(e.target.value) / 100);
  syncSoundControls();
  void sounds.unlock();
});
$('volume').addEventListener('change', () => sounds.play('toggle'));
document.addEventListener('visibilitychange', () => {
  sounds.setActive(!document.hidden);
  movementSounds.reset();
});
syncSoundControls();

function fail(msg) {
  $('err').textContent = msg;
  $('err').style.display = 'block';
  $('start').style.display = 'none';
}

async function main() {
  if (!navigator.gpu) return fail('WebGPU is not available in this browser. Try a recent Chrome, Edge, or Safari 26+.');
  const adapter = await navigator.gpu.requestAdapter({ powerPreference: 'high-performance' });
  if (!adapter) return fail('No WebGPU adapter found.');
  const device = await adapter.requestDevice({
    requiredLimits: {
      maxStorageBufferBindingSize: adapter.limits.maxStorageBufferBindingSize,
      maxBufferSize: adapter.limits.maxBufferSize,
    },
  });
  device.lost.then((i) => fail('GPU device lost: ' + i.message));
  let gpuErrs = 0;
  device.addEventListener('uncapturederror', (e) => { (window.__gpuErrs ||= []).push(e.error.message); if (gpuErrs++ === 0) fail('GPU error: ' + e.error.message); });
  const ctx = canvas.getContext('webgpu');
  const format = navigator.gpu.getPreferredCanvasFormat();
  ctx.configure({ device, format, alphaMode: 'opaque' });

  const bytes = await (await fetch('obby.wasm')).arrayBuffer();
  const { instance } = await WebAssembly.instantiate(bytes, {});
  const w = instance.exports;
  w.init();

  // --------------------------------------------------------- GPU resources
  const MAX_BOXES = 4096;
  const S = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST;
  const UNI_FLOATS = 100;
  const uniBuf = device.createBuffer({ size: UNI_FLOATS * 4, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
  const boxBuf = device.createBuffer({ size: MAX_BOXES * 64, usage: S });
  const nodeBuf = device.createBuffer({ size: MAX_BOXES * 2 * 32, usage: S });
  const primBuf = device.createBuffer({ size: MAX_BOXES * 4, usage: S });

  const mod = (code, label) => {
    const m = device.createShaderModule({ code, label });
    m.getCompilationInfo().then((info) => {
      for (const msg of info.messages) if (msg.type === 'error') fail(`${label}: ${msg.message} (line ${msg.lineNum})`);
    });
    return m;
  };
  const compute = (code, label) => device.createComputePipeline({ layout: 'auto', compute: { module: mod(code, label), entryPoint: 'main' } });
  const rasterMod = mod(RASTER_WGSL, 'raster'), postMod = mod(POST_WGSL, 'post');
  const tracePipe = compute(TRACE_WGSL, 'trace');
  const shadePipe = compute(SHADE_WGSL, 'shade');
  const ptTPipe = compute(PT_TEMPORAL_WGSL, 'pt-temporal');
  const ptPPipe = compute(PT_PREPASS_WGSL, 'pt-prepass');
  const ptRPipe = compute(PT_RESAMPLE_WGSL, 'pt-resample');
  const ptDPipe = compute(PT_DUP_WGSL, 'pt-dupmap');
  const dnAPipe = compute(DN_ACCUM_WGSL, 'dn-accum');
  const dnWPipe = compute(DN_ATROUS_WGSL, 'dn-atrous');
  // a-trous iterations: step 1, 2, 4, 8, 16; the first one feeds the colour history
  const DN_ITERS = 5;
  const dnIterBuf = Array.from({ length: DN_ITERS }, (_, k) => {
    const b = device.createBuffer({ size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    device.queue.writeBuffer(b, 0, new Uint32Array([1 << k, k === 0 ? 1 : 0, 0, 0]));
    return b;
  });
  const dnDummy = device.createTexture({ size: [1, 1], format: 'rgba16float', usage: GPUTextureUsage.STORAGE_BINDING }).createView();

  const rasterPipe = device.createRenderPipeline({
    layout: 'auto',
    vertex: { module: rasterMod, entryPoint: 'vs' },
    fragment: { module: rasterMod, entryPoint: 'fs', targets: [{ format: 'rgba8unorm' }, { format: 'rgba16float' }] },
    primitive: { topology: 'triangle-list', cullMode: 'none' },
    depthStencil: { format: 'depth32float', depthWriteEnabled: true, depthCompare: 'greater' },
  });
  const postPipe = device.createRenderPipeline({
    layout: 'auto',
    vertex: { module: postMod, entryPoint: 'vs' },
    fragment: { module: postMod, entryPoint: 'fs', targets: [{ format }] },
    primitive: { topology: 'triangle-list' },
  });
  const sampler = device.createSampler({ magFilter: 'linear', minFilter: 'linear' });
  const rasterBG = device.createBindGroup({
    layout: rasterPipe.getBindGroupLayout(0),
    entries: [{ binding: 0, resource: { buffer: uniBuf } }, { binding: 1, resource: { buffer: boxBuf } }],
  });

  // Paired spatial reuse textures (ReSTIR PT Enhanced, Sec. 3.1): start from consecutive link
  // indices, apply n_sigma tiled 2x2 random shuffles (every other one offset by one pixel),
  // then pair the two pixels sharing each link. Self-inverting: A -> B and B -> A.
  function makePairTexture(size, sigma, rand) {
    const n = Math.max(1, Math.floor(sigma * sigma / 2 + 1.46 / sigma + 1.76 / sigma ** 2 + 0.656 / sigma ** 3 + 0.5));
    const link = new Int32Array(size * size);
    for (let i = 0; i < link.length; i++) link[i] = i >> 1;
    const idx = [0, 0, 0, 0], vals = [0, 0, 0, 0];
    for (let s = 0; s < n; s++) {
      const off = s & 1;
      for (let by = 0; by < size; by += 2) {
        for (let bx = 0; bx < size; bx += 2) {
          for (let k = 0; k < 4; k++) {
            idx[k] = ((by + (k >> 1) + off) % size) * size + ((bx + (k & 1) + off) % size);
            vals[k] = link[idx[k]];
          }
          for (let k = 3; k > 0; k--) { const r = Math.floor(rand() * (k + 1)); const t = vals[k]; vals[k] = vals[r]; vals[r] = t; }
          for (let k = 0; k < 4; k++) link[idx[k]] = vals[k];
        }
      }
    }
    const first = new Int32Array(link.length / 2).fill(-1);
    const data = new Int8Array(link.length * 2);
    const half = size / 2;
    const wrap = (d) => (d > half ? d - size : d < -half ? d + size : d);
    for (let i = 0; i < link.length; i++) {
      const l = link[i];
      if (first[l] < 0) { first[l] = i; continue; }
      const j = first[l];
      const dx = wrap((j % size) - (i % size)), dy = wrap(Math.floor(j / size) - Math.floor(i / size));
      data[i * 2] = dx; data[i * 2 + 1] = dy;
      data[j * 2] = -dx; data[j * 2 + 1] = -dy;
    }
    const t = device.createTexture({ size: [size, size], format: 'rg8sint', usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST });
    device.queue.writeTexture({ texture: t }, data, { bytesPerRow: size * 2 }, [size, size]);
    return { tex: t, view: t.createView(), size };
  }
  let seedState = 0x9e3779b9;
  const rand = () => { // mulberry32
    seedState = (seedState + 0x6d2b79f5) | 0;
    let t = seedState;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
  // sigma = sqrt(8 / (9 pi)) R; R = 20 px at our render sizes
  const PAIR_SIGMA = Math.sqrt(8 / (9 * Math.PI)) * 20;
  const pairTex = [254, 230, 210].map((sz) => makePairTexture(sz, PAIR_SIGMA, rand));

  let W = 0, H = 0, tex = null, views = null, bufs = null, renderScale = 0.75, clearPT = true;
  let BG = null; // per-parity bind groups
  const RT = GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING;
  const ST = GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.TEXTURE_BINDING;
  const PT_PIXEL_BUDGET = 960 * 540;
  function resize() {
    const dpr = Math.min(devicePixelRatio || 1, 2);
    const cw = Math.max(1, Math.floor(canvas.clientWidth * dpr));
    const ch = Math.max(1, Math.floor(canvas.clientHeight * dpr));
    if (canvas.width !== cw || canvas.height !== ch) { canvas.width = cw; canvas.height = ch; }
    // Path tracing costs ~30 software-BVH rays per pixel, so in PT mode the internal resolution is
    // capped to a pixel budget (~960x540 at the default scale; G scales the budget too).
    let sc = renderScale;
    if (rtOn && ptOn) sc = Math.min(sc, Math.sqrt(PT_PIXEL_BUDGET * (renderScale / 0.75) ** 2 / (cw * ch)));
    const nw = Math.max(1, Math.floor(cw * sc)), nh = Math.max(1, Math.floor(ch * sc));
    if (nw === W && nh === H && tex) return;
    W = nw; H = nh;
    if (tex) for (const t of Object.values(tex)) t.destroy();
    if (bufs) for (const b of Object.values(bufs)) b.destroy();
    const mk = (f, u) => device.createTexture({ size: [W, H], format: f, usage: u });
    // G-buffers ping-pong so the temporal pass can see last frame's primary hits
    tex = {
      albedo0: mk('rgba8unorm', RT), normal0: mk('rgba16float', RT), depth0: mk('depth32float', RT),
      albedo1: mk('rgba8unorm', RT), normal1: mk('rgba16float', RT), depth1: mk('depth32float', RT),
      hist0: mk('rgba16float', ST), hist1: mk('rgba16float', ST), hdr: mk('rgba16float', ST),
      raw: mk('rgba16float', ST), pt: mk('rgba16float', ST),
      dnHist0: mk('rgba16float', ST), dnHist1: mk('rgba16float', ST),
      dnMom0: mk('rgba16float', ST), dnMom1: mk('rgba16float', ST),
      dnA: mk('rgba16float', ST), dnB: mk('rgba16float', ST),
    };
    const n = W * H, SB = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST;
    const mb = (bytes) => device.createBuffer({ size: bytes, usage: SB });
    // two 64-byte reservoir sets (paper: 2 x 64 B / pixel), paired-shift results, dup map
    bufs = { resA: mb(n * 64), resB: mb(n * 64), preFJ: mb(n * 3 * 16), preJac: mb(n * 3 * 4), dup: mb(n * 4) };
    clearPT = true;
    const v = {}; for (const k in tex) v[k] = tex[k].createView();
    views = v;
    const buf = (b) => ({ buffer: b });
    BG = [0, 1].map((i) => {
      const cur = { albedo: i ? v.albedo1 : v.albedo0, normal: i ? v.normal1 : v.normal0, depth: i ? v.depth1 : v.depth0 };
      const prev = { albedo: i ? v.albedo0 : v.albedo1, normal: i ? v.normal0 : v.normal1, depth: i ? v.depth0 : v.depth1 };
      const scene = [
        { binding: 0, resource: buf(uniBuf) },
        { binding: 1, resource: buf(boxBuf) },
        { binding: 2, resource: buf(nodeBuf) },
        { binding: 3, resource: buf(primBuf) },
        { binding: 4, resource: cur.normal },
        { binding: 5, resource: cur.depth },
      ];
      const pairs = (b) => pairTex.map((p, s) => ({ binding: b + s, resource: p.view }));
      return {
        cur,
        trace: device.createBindGroup({ layout: tracePipe.getBindGroupLayout(0), entries: [...scene, { binding: 6, resource: v.raw }] }),
        shade: device.createBindGroup({
          layout: shadePipe.getBindGroupLayout(0),
          entries: [
            ...scene,
            { binding: 6, resource: cur.albedo },
            { binding: 7, resource: v.raw },
            { binding: 8, resource: i ? v.hist1 : v.hist0 },
            { binding: 9, resource: i ? v.hist0 : v.hist1 },
            { binding: 10, resource: v.hdr },
            { binding: 11, resource: v.dnB }, // denoised PT illumination (after 5 a-trous passes)
          ],
        }),
        ptT: device.createBindGroup({
          layout: ptTPipe.getBindGroupLayout(0),
          entries: [
            ...scene,
            { binding: 6, resource: cur.albedo },
            { binding: 7, resource: prev.albedo },
            { binding: 8, resource: prev.normal },
            { binding: 9, resource: prev.depth },
            { binding: 10, resource: buf(bufs.resA) },
            { binding: 11, resource: buf(bufs.resB) },
            { binding: 12, resource: buf(bufs.dup) },
          ],
        }),
        ptP: device.createBindGroup({
          layout: ptPPipe.getBindGroupLayout(0),
          entries: [
            ...scene,
            { binding: 6, resource: cur.albedo },
            { binding: 7, resource: buf(bufs.resB) },
            { binding: 8, resource: buf(bufs.preFJ) },
            { binding: 9, resource: buf(bufs.preJac) },
            ...pairs(10),
          ],
        }),
        ptR: device.createBindGroup({
          layout: ptRPipe.getBindGroupLayout(0),
          entries: [
            // resampling reuses the pre-pass shifts and traces no rays: no box / BVH bindings
            ...scene.filter((e) => e.binding === 0 || e.binding >= 4),
            { binding: 6, resource: cur.albedo },
            { binding: 7, resource: buf(bufs.resB) },
            { binding: 8, resource: buf(bufs.resA) },
            { binding: 9, resource: buf(bufs.preFJ) },
            { binding: 10, resource: buf(bufs.preJac) },
            { binding: 11, resource: v.pt },
            ...pairs(12),
          ],
        }),
        dnA: device.createBindGroup({
          layout: dnAPipe.getBindGroupLayout(0),
          entries: [
            { binding: 0, resource: buf(uniBuf) },
            { binding: 1, resource: cur.normal },
            { binding: 2, resource: cur.depth },
            { binding: 3, resource: cur.albedo },
            { binding: 4, resource: v.pt },
            { binding: 5, resource: i ? v.dnHist1 : v.dnHist0 },
            { binding: 6, resource: i ? v.dnMom1 : v.dnMom0 },
            { binding: 7, resource: i ? v.dnMom0 : v.dnMom1 },
            { binding: 8, resource: v.dnA },
          ],
        }),
        dnW: dnIterBuf.map((ib, k) => device.createBindGroup({
          layout: dnWPipe.getBindGroupLayout(0),
          entries: [
            { binding: 0, resource: buf(uniBuf) },
            { binding: 1, resource: cur.normal },
            { binding: 2, resource: cur.depth },
            { binding: 4, resource: k & 1 ? v.dnB : v.dnA },
            { binding: 5, resource: k & 1 ? v.dnA : v.dnB },
            { binding: 6, resource: k === 0 ? (i ? v.dnHist0 : v.dnHist1) : dnDummy },
            { binding: 7, resource: buf(ib) },
          ],
        })),
        ptD: device.createBindGroup({
          layout: ptDPipe.getBindGroupLayout(0),
          entries: [
            { binding: 0, resource: buf(uniBuf) },
            { binding: 7, resource: buf(bufs.resA) },
            { binding: 8, resource: buf(bufs.dup) },
          ],
        }),
        post: device.createBindGroup({
          layout: postPipe.getBindGroupLayout(0),
          entries: [{ binding: 0, resource: v.hdr }, { binding: 1, resource: buf(uniBuf) }, { binding: 2, resource: sampler }],
        }),
      };
    });
  }

  // --------------------------------------------------------- input
  const keys = new Set();
  let yaw = Math.PI, pitch = 0.32, dist = 18, rtOn = true, ptOn = true, dragging = false, playing = false;
  const SCALES = [1, 0.75, 0.5];
  let scaleIdx = 1; // default 75%: ReSTIR PT Enhanced runs at full render resolution
  addEventListener('keydown', (e) => {
    // Leave native keyboard interaction intact on the audio controls.
    if (e.target.closest?.('#audio-controls') && ['Space', 'Enter', 'ArrowLeft', 'ArrowRight', 'ArrowUp', 'ArrowDown', 'Home', 'End', 'PageUp', 'PageDown'].includes(e.code)) return;
    if (e.target.closest?.('button') && ['Space', 'Enter'].includes(e.code)) return;
    if (playing) void sounds.unlock();
    keys.add(e.code);
    if (e.code === 'Space') e.preventDefault();
    if (e.repeat) return;
    if (e.code === 'KeyR') { rtOn = !rtOn; flashMsg(rtOn ? 'Ray tracing ON' : 'Raster only'); }
    if (e.code === 'KeyP') { ptOn = !ptOn; clearPT = true; flashMsg(ptOn ? 'ReSTIR PT Enhanced ON' : 'Hybrid RT (direct shadows + mirrors, flat ambient)'); }
    if (e.code === 'KeyG') { scaleIdx = (scaleIdx + 1) % SCALES.length; renderScale = SCALES[scaleIdx]; flashMsg(`Render scale ${Math.round(renderScale * 100)}%`); }
    if (e.code === 'KeyK') { w.respawn(); movementSounds.reset(); if (playing) sounds.play('respawn'); }
    if (e.code === 'KeyT') { w.restart(); movementSounds.reset(); if (playing) sounds.play('restart'); $('win').style.display = 'none'; }
    if (e.code === 'KeyH') $('help').classList.toggle('hidden');
    if (e.code === 'KeyM') toggleSound();
    if (playing && ['KeyR', 'KeyP', 'KeyG', 'KeyH'].includes(e.code)) sounds.play('toggle');
  });
  addEventListener('keyup', (e) => keys.delete(e.code));
  addEventListener('blur', () => { keys.clear(); sounds.stopAll(); movementSounds.reset(); });
  canvas.addEventListener('contextmenu', (e) => e.preventDefault());
  canvas.addEventListener('mousedown', (e) => {
    if (playing) void sounds.unlock();
    if (e.button === 2) dragging = true;
    else if (playing && document.pointerLockElement !== canvas) canvas.requestPointerLock?.();
  });
  addEventListener('mouseup', () => (dragging = false));
  addEventListener('mousemove', (e) => {
    if (document.pointerLockElement === canvas || dragging) {
      yaw -= e.movementX * 0.0035;
      pitch = Math.min(1.35, Math.max(-1.2, pitch + e.movementY * 0.0035));
    }
  });
  canvas.addEventListener('wheel', (e) => { e.preventDefault(); dist = Math.min(45, Math.max(4, dist * Math.exp(e.deltaY * 0.001))); }, { passive: false });
  $('start').querySelector('button').addEventListener('click', () => {
    w.start();
    playing = true;
    $('start').style.display = 'none';
    movementSounds.reset();
    void sounds.unlock().then(() => sounds.play('start'));
    canvas.requestPointerLock?.();
  });

  let msgTimer = 0;
  function flashMsg(t) { $('msg').textContent = t; $('msg').style.opacity = 1; msgTimer = 1.6; }

  // --------------------------------------------------------- loop
  const uni = new Float32Array(UNI_FLOATS);
  const sun = (() => { const s = [-0.42, 0.78, -0.46], l = Math.hypot(...s); return s.map((x) => x / l); })();
  let prevVP = null, prevInvVP = null, prevEye = null, last = performance.now(), frameNo = 0, flash = 0, fpsAcc = 0, fpsN = 0, fps = 0;

  function fmtTime(t) { const m = Math.floor(t / 60), s = (t % 60).toFixed(2).padStart(5, '0'); return `${m}:${s}`; }

  // At most 2 frames queued on the GPU; otherwise a slow path-traced frame lets the queue grow
  // without bound and the page freezes behind it.
  let inFlight = 0, gpuMs = 0;
  function tick(now) {
    if (inFlight >= 2) { requestAnimationFrame(tick); return; }
    const dt = Math.min((now - last) / 1000, 0.1); last = now;
    fpsAcc += dt; fpsN++;
    if (fpsAcc > 0.5) { fps = fpsN / fpsAcc; fpsAcc = 0; fpsN = 0; }
    resize();

    const k = (c) => (keys.has(c) ? 1 : 0);
    const mz = playing ? k('KeyW') + k('ArrowUp') - k('KeyS') - k('ArrowDown') : 0;
    const mx = playing ? k('KeyD') + k('ArrowRight') - k('KeyA') - k('ArrowLeft') : 0;
    const jump = playing ? k('Space') : 0;
    if (!playing) yaw += dt * 0.08; // idle orbit behind the title card
    w.frame(dt, mx, mz, jump, yaw, pitch, dist);

    const mem = w.memory.buffer;
    const out = new Float32Array(mem, w.out_ptr(), 32);
    const eye = [out[0], out[1], out[2]], target = [out[3], out[4], out[5]];
    const ev = out[13];
    if (ev & 1) flashMsg(`Checkpoint! Stage ${out[9]}`);
    if (ev & 2) flash = 1;
    if (ev & 4) { $('win').style.display = 'flex'; $('wintime').textContent = fmtTime(out[12]); $('windeaths').textContent = out[11]; }
    if (playing) {
      // Death/win take priority over incidental sounds from the same physics frame.
      if (ev & 2) sounds.play('death');
      else if (ev & 4) sounds.play('win');
      else {
        if (ev & 1) sounds.play('checkpoint');
        if (ev & 8) sounds.play('jump');
        if (ev & 16) sounds.play('bounce');
      }
      if (!out[14]) movementSounds.update({ x: out[6], z: out[8], grounded: Boolean(out[15]), moving: Boolean(mx || mz), events: ev }, dt);
    }

    const nb = w.box_count(), nn = w.node_count(), np = w.prim_count();
    device.queue.writeBuffer(boxBuf, 0, mem, w.boxes_ptr(), nb * 64);
    device.queue.writeBuffer(nodeBuf, 0, mem, w.nodes_ptr(), nn * 32);
    device.queue.writeBuffer(primBuf, 0, mem, w.prims_ptr(), np * 4);

    const view = m4.lookAt(eye, target, [0, 1, 0]);
    const proj = m4.persp(70 * Math.PI / 180, W / H, 0.1);
    const vp = m4.mul(proj, view);
    const invVP = m4.inv(vp);
    uni.set(vp, 0);
    uni.set(invVP, 16);
    uni.set(prevVP || vp, 32);
    uni.set(prevInvVP || invVP, 48);
    uni.set([eye[0], eye[1], eye[2], now / 1000], 64);
    uni.set([sun[0], sun[1], sun[2], frameNo], 68);
    uni.set([W, H, rtOn ? 1 : 0, nb], 72);
    uni.set([flash, 1.0, canvas.width, canvas.height], 76);
    uni.set([rtOn && ptOn ? 1 : 0, 0, 0, 0], 80);
    uni.set([...(prevEye || eye), 0], 84);
    // random flip / mirror / transpose / offset of each pairing texture, every frame (Sec. 3.2)
    pairTex.forEach((p, s) => uni.set([Math.floor(rand() * 8), Math.floor(rand() * p.size), Math.floor(rand() * p.size), 0], 88 + s * 4));
    device.queue.writeBuffer(uniBuf, 0, uni);
    prevVP = vp; prevInvVP = invVP; prevEye = eye;
    flash = Math.max(0, flash - dt * 2.5);

    const par = frameNo & 1;
    const bg = BG[par];
    const enc = device.createCommandEncoder();
    const rp = enc.beginRenderPass({
      colorAttachments: [
        { view: bg.cur.albedo, loadOp: 'clear', storeOp: 'store', clearValue: [0, 0, 0, 0] },
        { view: bg.cur.normal, loadOp: 'clear', storeOp: 'store', clearValue: [0, 0, 0, 0] },
      ],
      depthStencilAttachment: { view: bg.cur.depth, depthClearValue: 0, depthLoadOp: 'clear', depthStoreOp: 'store' },
    });
    rp.setPipeline(rasterPipe);
    rp.setBindGroup(0, rasterBG);
    rp.draw(36, nb);
    rp.end();

    const gx = Math.ceil(W / 8), gy = Math.ceil(H / 8);
    const run = (pipe, group, x = gx, y = gy) => {
      const c = enc.beginComputePass();
      c.setPipeline(pipe);
      c.setBindGroup(0, group);
      c.dispatchWorkgroups(x, y);
      c.end();
    };
    if (rtOn && ptOn) {
      if (clearPT) { enc.clearBuffer(bufs.resA); enc.clearBuffer(bufs.dup); clearPT = false; }
      run(ptTPipe, bg.ptT);   // initial path tree + temporal reuse      resA -> resB
      run(ptPPipe, bg.ptP);   // paired spatial pre-pass (one shift per pair side)
      run(ptRPipe, bg.ptR);   // paired spatial resampling               resB -> resA
      run(ptDPipe, bg.ptD, Math.ceil(W / 16), Math.ceil(H / 16)); // duplication map for next frame
      run(dnAPipe, bg.dnA);   // denoiser: demodulate + temporal accumulation + variance
      bg.dnW.forEach((g) => run(dnWPipe, g)); // denoiser: 5 a-trous passes, dnA <-> dnB, result in dnB
    } else if (rtOn) {
      run(tracePipe, bg.trace);
    }
    run(shadePipe, bg.shade);

    const pp = enc.beginRenderPass({
      colorAttachments: [{ view: ctx.getCurrentTexture().createView(), loadOp: 'clear', storeOp: 'store', clearValue: [0, 0, 0, 1] }],
    });
    pp.setPipeline(postPipe);
    pp.setBindGroup(0, bg.post);
    pp.draw(3);
    pp.end();
    device.queue.submit([enc.finish()]);
    inFlight++;
    const t0 = performance.now();
    device.queue.onSubmittedWorkDone().then(() => { inFlight--; gpuMs = gpuMs * 0.9 + (performance.now() - t0) * 0.1; });
    frameNo++;

    // HUD
    $('stage').textContent = `${out[9]} / ${out[10]}`;
    $('deaths').textContent = out[11];
    $('time').textContent = fmtTime(out[12]);
    $('perf').textContent = `${fps.toFixed(0)} fps · ${gpuMs.toFixed(1)} ms · ${W}×${H} · ${nb} boxes · ${rtOn ? (ptOn ? 'ReSTIR PT Enhanced' : 'hybrid RT') : 'raster'}`;
    if (msgTimer > 0) { msgTimer -= dt; if (msgTimer <= 0) $('msg').style.opacity = 0; }
    requestAnimationFrame(tick);
  }
  requestAnimationFrame(tick);
}

main().catch((e) => fail(String(e?.message || e)));
