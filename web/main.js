import { RASTER_WGSL, TRACE_WGSL, PT_TEMPORAL_WGSL, PT_PREPASS_WGSL, PT_RESAMPLE_WGSL, PT_DUP_WGSL, DN_ACCUM_WGSL, DN_ATROUS_WGSL, SHADE_WGSL, POST_WGSL,
  SHADOW_WGSL, PROBE_TRACE_WGSL, PROBE_UPDATE_WGSL, AO_WGSL, AO_BLUR_WGSL, LIGHT_WGSL, TRANSPARENT_WGSL, BLOOM_DOWN_WGSL, BLOOM_UP_WGSL, WARP_WGSL } from './shaders.js';
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
  // orthographic, WebGPU depth 0..1 (sun shadow maps)
  ortho(r, near, far) {
    const o = new Float32Array(16);
    o[0] = 1 / r; o[5] = 1 / r; o[10] = -1 / (far - near); o[14] = -near / (far - near); o[15] = 1;
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

const add3 = (a, b) => [a[0] + b[0], a[1] + b[1], a[2] + b[2]];
const sub3 = (a, b) => [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
const scale3 = (a, k) => [a[0] * k, a[1] * k, a[2] * k];
const cross3 = (a, b) => [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
const norm3 = (a) => { const l = Math.hypot(...a) || 1; return [a[0] / l, a[1] / l, a[2] / l]; };

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
  // A second, independent device only presents (asynchronous reprojection): its queue doesn't
  // wait behind a long path-traced frame on the scene device. Falls back to the scene device.
  let pdev = null;
  try {
    const a2 = await navigator.gpu.requestAdapter({ powerPreference: 'high-performance' });
    if (a2) pdev = await a2.requestDevice();
  } catch { pdev = null; }
  const twoDev = !!pdev;
  if (!pdev) pdev = device;
  if (twoDev) pdev.addEventListener('uncapturederror', (e) => fail('GPU error (present device): ' + e.error.message));
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
  const UNI_FLOATS = 172;
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
  // ---- raster lighting resources (fixed size)
  const SM_SIZE = 2048, CASCADES = 3;
  const PROBES = 24 * 12 * 32, PRAYS = 48, PROBE_SPACING = 3, MAX_LIGHTS = 256;
  const shadowPipe = device.createRenderPipeline({
    layout: 'auto',
    vertex: { module: mod(SHADOW_WGSL, 'shadow'), entryPoint: 'vs' },
    primitive: { topology: 'triangle-list', cullMode: 'none' },
    depthStencil: { format: 'depth32float', depthWriteEnabled: true, depthCompare: 'less', depthBias: 2, depthBiasSlopeScale: 2.0 },
  });
  const probeTPipe = compute(PROBE_TRACE_WGSL, 'probe-trace');
  const probeUPipe = compute(PROBE_UPDATE_WGSL, 'probe-update');
  const aoPipe = compute(AO_WGSL, 'ao');
  const aoBPipe = compute(AO_BLUR_WGSL, 'ao-blur');
  const lightPipe = compute(LIGHT_WGSL, 'light');
  const bloomDPipe = compute(BLOOM_DOWN_WGSL, 'bloom-down');
  const bloomUPipe = compute(BLOOM_UP_WGSL, 'bloom-up');
  const transpMod = mod(TRANSPARENT_WGSL, 'transparent');
  const premul = { srcFactor: 'one', dstFactor: 'one-minus-src-alpha', operation: 'add' };
  const transpPipe = device.createRenderPipeline({
    layout: 'auto',
    vertex: { module: transpMod, entryPoint: 'vs' },
    fragment: { module: transpMod, entryPoint: 'fs', targets: [{ format: 'rgba16float', blend: { color: premul, alpha: premul } }] },
    primitive: { topology: 'triangle-list', cullMode: 'none' },
    depthStencil: { format: 'depth32float', depthWriteEnabled: false, depthCompare: 'greater' },
  });
  const shadowTex = device.createTexture({ size: [SM_SIZE, SM_SIZE, CASCADES], format: 'depth32float', usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING });
  const shadowLayer = [0, 1, 2].map((l) => shadowTex.createView({ dimension: '2d', baseArrayLayer: l, arrayLayerCount: 1 }));
  const shadowArray = shadowTex.createView({ dimension: '2d-array' });
  const shadowSamp = device.createSampler({ compare: 'less', magFilter: 'linear', minFilter: 'linear' });
  const cascadeBG = [0, 1, 2].map((l) => {
    const b = device.createBuffer({ size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    device.queue.writeBuffer(b, 0, new Uint32Array([l, 0, 0, 0]));
    return device.createBindGroup({
      layout: shadowPipe.getBindGroupLayout(0),
      entries: [{ binding: 0, resource: { buffer: uniBuf } }, { binding: 1, resource: { buffer: boxBuf } }, { binding: 2, resource: { buffer: b } }],
    });
  });
  const probeIrrBuf = device.createBuffer({ size: PROBES * 64 * 16, usage: S });
  const probeDepBuf = device.createBuffer({ size: PROBES * 256 * 8, usage: S });
  const probeStateBuf = device.createBuffer({ size: PROBES * 16, usage: S });
  const rayBuf = device.createBuffer({ size: PROBES * PRAYS * 16, usage: S });
  const lightBuf = device.createBuffer({ size: MAX_LIGHTS * 32, usage: S });
  const probeTraceBG = device.createBindGroup({
    layout: probeTPipe.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: uniBuf } }, { binding: 1, resource: { buffer: boxBuf } },
      { binding: 2, resource: { buffer: nodeBuf } }, { binding: 3, resource: { buffer: primBuf } },
      { binding: 6, resource: { buffer: probeIrrBuf } }, { binding: 7, resource: { buffer: probeDepBuf } },
      { binding: 8, resource: { buffer: probeStateBuf } }, { binding: 9, resource: { buffer: lightBuf } },
      { binding: 10, resource: { buffer: rayBuf } },
    ],
  });
  const probeUpdateBG = device.createBindGroup({
    layout: probeUPipe.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: uniBuf } }, { binding: 1, resource: { buffer: rayBuf } },
      { binding: 2, resource: { buffer: probeIrrBuf } }, { binding: 3, resource: { buffer: probeDepBuf } },
      { binding: 4, resource: { buffer: probeStateBuf } },
    ],
  });
  const transpBG = device.createBindGroup({
    layout: transpPipe.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: uniBuf } }, { binding: 1, resource: { buffer: boxBuf } },
      { binding: 2, resource: shadowArray }, { binding: 3, resource: shadowSamp },
      { binding: 4, resource: { buffer: probeIrrBuf } }, { binding: 5, resource: { buffer: probeDepBuf } },
      { binding: 6, resource: { buffer: probeStateBuf } }, { binding: 7, resource: { buffer: lightBuf } },
    ],
  });
  const bloomPrm = [1, 0].map((first) => {
    const b = device.createBuffer({ size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    device.queue.writeBuffer(b, 0, new Float32Array([first, 0, 0, 0]));
    return b;
  });
  const BLOOM_LEVELS = 6;

  // ---- asynchronous reprojection (timewarp) on the present device
  const warpCode = WARP_WGSL
    .replace('DEPTH_DECL', twoDev ? '@group(0) @binding(2) var srcDepth: texture_2d<f32>;' : '@group(0) @binding(2) var srcDepth: texture_depth_2d;')
    .replace('DEPTH_LOAD', twoDev ? 'textureLoad(srcDepth, dp, 0).r' : 'textureLoad(srcDepth, dp, 0)');
  const warpMod = pdev.createShaderModule({ code: warpCode, label: 'warp' });
  const warpPipe = pdev.createRenderPipeline({
    layout: 'auto',
    vertex: { module: warpMod, entryPoint: 'vs' },
    fragment: { module: warpMod, entryPoint: 'fs', targets: [{ format }] },
    primitive: { topology: 'triangle-list' },
  });
  const warpUni = pdev.createBuffer({ size: 56 * 4, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
  const warpSamp = pdev.createSampler({ magFilter: 'linear', minFilter: 'linear' });
  const OVERSCAN = 1.2;   // async mode renders a wider field of view so fast turns don't show edges
  let warp = null;        // per-resize: source textures, staging buffers, bind groups
  const RASTER_EXPOSURE = 0.62; // hand-tuned: keeps sunlit plastic around 0.8 after ACES

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

  let CW = 0, CH = 0, W = 0, H = 0, tex = null, views = null, bufs = null, renderScale = 0.75, clearPT = true;
  let BG = null, bloomPasses = []; // per-parity bind groups, bloom chain
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
    if (nw === W && nh === H && cw === CW && ch === CH && tex) return;
    W = nw; H = nh; CW = cw; CH = ch;
    if (warp) { for (const t of warp.own) t.destroy(); warp = null; }
    if (tex) for (const t of Object.values(tex)) t.destroy();
    if (bufs) for (const b of Object.values(bufs)) b.destroy();
    const mk = (f, u) => device.createTexture({ size: [W, H], format: f, usage: u });
    // G-buffers ping-pong so the temporal pass can see last frame's primary hits
    tex = {
      albedo0: mk('rgba8unorm', RT), normal0: mk('rgba16float', RT), depth0: mk('depth32float', RT | GPUTextureUsage.COPY_SRC),
      albedo1: mk('rgba8unorm', RT), normal1: mk('rgba16float', RT), depth1: mk('depth32float', RT | GPUTextureUsage.COPY_SRC),
      hist0: mk('rgba16float', ST), hist1: mk('rgba16float', ST), hdr: mk('rgba16float', ST | GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.COPY_SRC),
      raw: mk('rgba16float', ST), pt: mk('rgba16float', ST),
      aoRaw: device.createTexture({ size: [Math.max(1, (W + 1) >> 1), Math.max(1, (H + 1) >> 1)], format: 'r32float', usage: ST }), ao: mk('r32float', ST),
      hdrPrev: mk('rgba16float', GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST),
      dnHist0: mk('rgba16float', ST), dnHist1: mk('rgba16float', ST),
      dnMom0: mk('rgba16float', ST), dnMom1: mk('rgba16float', ST),
      dnA: mk('rgba16float', ST), dnB: mk('rgba16float', ST),
    };
    // bloom pyramids (half resolution, 6 mips): down chain and accumulated up chain
    const bw = Math.max(1, W >> 1), bh = Math.max(1, H >> 1);
    const levels = Math.min(BLOOM_LEVELS, Math.floor(Math.log2(Math.max(1, Math.min(bw, bh)))) + 1);
    const mkPyr = () => device.createTexture({ size: [bw, bh], format: 'rgba16float', mipLevelCount: levels, usage: ST });
    tex.bloomD = mkPyr(); tex.bloomU = mkPyr();
    // finished frames for the timewarp (canvas resolution, ping-pong so a frame is never overwritten
    // while it is still being presented)
    const mkLdr = () => device.createTexture({ size: [CW, CH], format, usage: RT | GPUTextureUsage.COPY_SRC });
    tex.ldr0 = mkLdr(); tex.ldr1 = mkLdr();
    const n = W * H, SB = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST;
    const mb = (bytes) => device.createBuffer({ size: bytes, usage: SB });
    // two 64-byte reservoir sets (paper: 2 x 64 B / pixel), paired-shift results, dup map
    bufs = { resA: mb(n * 64), resB: mb(n * 64), preFJ: mb(n * 3 * 16), preJac: mb(n * 3 * 4), dup: mb(n * 4) };
    clearPT = true;
    const v = {}; for (const k in tex) v[k] = tex[k].createView();
    views = v;
    const buf = (b) => ({ buffer: b });
    const mip = (t, l) => t.createView({ baseMipLevel: l, mipLevelCount: 1 });
    const bD = Array.from({ length: levels }, (_, l) => mip(tex.bloomD, l));
    const bU = Array.from({ length: levels }, (_, l) => mip(tex.bloomU, l));
    bloomPasses = [];
    for (let l = 0; l < levels; l++) {
      const dims = [Math.max(1, bw >> l), Math.max(1, bh >> l)];
      bloomPasses.push({
        pipe: bloomDPipe, dims,
        group: device.createBindGroup({
          layout: bloomDPipe.getBindGroupLayout(0),
          entries: [
            { binding: 0, resource: l === 0 ? v.hdr : bD[l - 1] }, { binding: 1, resource: sampler },
            { binding: 2, resource: bD[l] }, { binding: 3, resource: buf(bloomPrm[l === 0 ? 0 : 1]) },
          ],
        }),
      });
    }
    for (let l = levels - 2; l >= 0; l--) {
      const dims = [Math.max(1, bw >> l), Math.max(1, bh >> l)];
      bloomPasses.push({
        pipe: bloomUPipe, dims,
        group: device.createBindGroup({
          layout: bloomUPipe.getBindGroupLayout(0),
          entries: [
            { binding: 0, resource: l === levels - 2 ? bD[l + 1] : bU[l + 1] }, { binding: 1, resource: sampler },
            { binding: 2, resource: bD[l] }, { binding: 3, resource: bU[l] },
          ],
        }),
      });
    }
    const bloomOut = levels > 1 ? bU[0] : bD[0];
    {
      const cbpr = Math.ceil(CW * 4 / 256) * 256, dbpr = Math.ceil(W * 4 / 256) * 256;
      const wbg = (srcView, depthView) => pdev.createBindGroup({
        layout: warpPipe.getBindGroupLayout(0),
        entries: [{ binding: 0, resource: { buffer: warpUni } }, { binding: 1, resource: srcView },
          { binding: 2, resource: depthView }, { binding: 3, resource: warpSamp }],
      });
      if (twoDev) {
        const sc = pdev.createTexture({ size: [CW, CH], format, usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST });
        const sd = pdev.createTexture({ size: [W, H], format: 'r32float', usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST });
        const stC = device.createBuffer({ size: cbpr * CH, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
        const stD = device.createBuffer({ size: dbpr * H, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
        const g = wbg(sc.createView(), sd.createView());
        warp = { own: [sc, sd], sc, sd, stC, stD, cbpr, dbpr, bg: [g, g] };
      } else {
        warp = { own: [], cbpr, dbpr, bg: [wbg(v.ldr0, v.depth0), wbg(v.ldr1, v.depth1)] };
      }
    }
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
        ao: device.createBindGroup({
          layout: aoPipe.getBindGroupLayout(0),
          entries: [{ binding: 0, resource: buf(uniBuf) }, { binding: 1, resource: cur.normal }, { binding: 2, resource: cur.depth }, { binding: 3, resource: v.aoRaw }],
        }),
        aoB: device.createBindGroup({
          layout: aoBPipe.getBindGroupLayout(0),
          entries: [{ binding: 0, resource: buf(uniBuf) }, { binding: 2, resource: cur.depth }, { binding: 3, resource: v.aoRaw }, { binding: 4, resource: v.ao }],
        }),
        light: device.createBindGroup({
          layout: lightPipe.getBindGroupLayout(0),
          entries: [
            { binding: 0, resource: buf(uniBuf) }, { binding: 1, resource: buf(boxBuf) },
            { binding: 2, resource: cur.normal }, { binding: 3, resource: cur.depth }, { binding: 4, resource: cur.albedo },
            { binding: 5, resource: shadowArray }, { binding: 6, resource: shadowSamp },
            { binding: 7, resource: buf(probeIrrBuf) }, { binding: 8, resource: buf(probeDepBuf) },
            { binding: 9, resource: buf(probeStateBuf) }, { binding: 10, resource: buf(lightBuf) },
            { binding: 11, resource: v.ao }, { binding: 12, resource: v.hdrPrev }, { binding: 13, resource: sampler },
            { binding: 14, resource: v.hdr },
          ],
        }),
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
            { binding: 12, resource: buf(probeIrrBuf) }, { binding: 13, resource: buf(probeDepBuf) },
            { binding: 14, resource: buf(probeStateBuf) }, { binding: 15, resource: buf(lightBuf) },
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
          entries: [{ binding: 0, resource: v.hdr }, { binding: 1, resource: buf(uniBuf) }, { binding: 2, resource: sampler }, { binding: 3, resource: bloomOut }],
        }),
      };
    });
  }

  // --------------------------------------------------------- input
  const keys = new Set();
  let yaw = Math.PI, pitch = 0.32, dist = 18, rtOn = false, ptOn = true, dragging = false, playing = false;
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
    if (e.code === 'KeyR') { rtOn = !rtOn; clearPT = true; flashMsg(rtOn ? (ptOn ? 'ReSTIR PT Enhanced' : 'Hybrid ray tracing') : 'Raster lighting: probes + lights + shadow maps'); }
    if (e.code === 'KeyP') { ptOn = !ptOn; clearPT = true; flashMsg(ptOn ? 'ReSTIR PT Enhanced ON' : 'Hybrid RT: traced shadows + reflections, probe GI, lights'); }
    if (e.code === 'KeyG') { scaleIdx = (scaleIdx + 1) % SCALES.length; renderScale = SCALES[scaleIdx]; flashMsg(`Render scale ${Math.round(renderScale * 100)}%`); }
    if (e.code === 'KeyF') { setAsync(!asyncOn); flashMsg(asyncOn ? `Async reprojection ON${twoDev ? '' : ' (single device)'}` : 'Async reprojection OFF'); }
    if (e.code === 'KeyV') { capIdx = (capIdx + 1) % SCENE_CAPS.length; flashMsg(SCENE_CAPS[capIdx] ? `Scene capped at ${SCENE_CAPS[capIdx]} fps` : 'Scene uncapped'); }
    if (e.code === 'KeyX') {   // freeze: keep warping the last finished frame, render nothing new
      if (!asyncOn) setAsync(true);
      frozen = !frozen;
      flashMsg(frozen ? 'Frame frozen — look around (X to unfreeze)' : 'Unfrozen');
    }
    if (e.code === 'KeyI') { infillOn = !infillOn; flashMsg(`Infill ${infillOn ? 'ON' : 'OFF'}`); }
    if (e.code === 'KeyJ') { showHoles = !showHoles; flashMsg(`Hole tint ${showHoles ? 'ON' : 'OFF'}`); }
    if (e.code === 'KeyK') { w.respawn(); movementSounds.reset(); if (playing) sounds.play('respawn'); }
    if (e.code === 'KeyT') { w.restart(); movementSounds.reset(); if (playing) sounds.play('restart'); $('win').style.display = 'none'; }
    if (e.code === 'KeyH') $('help').classList.toggle('hidden');
    if (e.code === 'KeyM') toggleSound();
    if (playing && ['KeyR', 'KeyP', 'KeyG', 'KeyF', 'KeyV', 'KeyH'].includes(e.code)) sounds.play('toggle');
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

  function renderScene(now, eye, target, outView, fovY, beforeSubmit) {
    const mem = w.memory.buffer;
    const nb = w.box_count(), nn = w.node_count(), np = w.prim_count();
    device.queue.writeBuffer(boxBuf, 0, mem, w.boxes_ptr(), nb * 64);
    device.queue.writeBuffer(nodeBuf, 0, mem, w.nodes_ptr(), nn * 32);
    device.queue.writeBuffer(primBuf, 0, mem, w.prims_ptr(), np * 4);

    const nl = w.light_count();
    device.queue.writeBuffer(lightBuf, 0, mem, w.lights_ptr(), Math.min(nl, MAX_LIGHTS) * 32);

    const view = m4.lookAt(eye, target, [0, 1, 0]);
    const proj = m4.persp(fovY, W / H, 0.1);
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

    // ---- cascaded shadow maps: fit a sphere around each frustum slice, snap to texels
    const fwd = norm3(sub3(target, eye));
    const tanY = Math.tan(fovY / 2), tanX = tanY * W / H;
    const right = norm3(cross3(fwd, [0, 1, 0])), up = cross3(right, fwd);
    const splits = [0.1, dist + 10, (dist + 10) * 2.8, 190];
    const texel = [0, 0, 0, 0], range = [0, 0, 0, 0];
    for (let c = 0; c < CASCADES; c++) {
      const n = splits[c], f = splits[c + 1];
      const pts = [];
      for (const z of [n, f]) for (const sx of [-1, 1]) for (const sy of [-1, 1])
        pts.push(add3(eye, add3(scale3(fwd, z), add3(scale3(right, sx * tanX * z), scale3(up, sy * tanY * z)))));
      const ctr = pts.reduce((a, p) => add3(a, scale3(p, 1 / 8)), [0, 0, 0]);
      let r = 0; for (const p of pts) r = Math.max(r, Math.hypot(...sub3(p, ctr)));
      r = Math.ceil(r * 4) / 4;
      const back = r + 160;
      const lv = m4.lookAt(add3(ctr, scale3(sun, back)), ctr, Math.abs(sun[1]) > 0.95 ? [0, 0, 1] : [0, 1, 0]);
      const lvp = m4.mul(m4.ortho(r, 0, back + r + 40), lv);
      // texel snapping keeps the shadow edges from crawling when the camera moves
      const ox = lvp[12] * SM_SIZE / 2, oy = lvp[13] * SM_SIZE / 2;
      lvp[12] += (Math.round(ox) - ox) * 2 / SM_SIZE;
      lvp[13] += (Math.round(oy) - oy) * 2 / SM_SIZE;
      uni.set(lvp, 100 + c * 16);
      texel[c] = 2 * r / SM_SIZE;
      range[c] = back + r + 40;
    }
    uni.set([splits[1], splits[2], splits[3], 0.022], 148);
    uni.set(texel, 152);
    uni.set(range, 156);
    // ---- DDGI probe volume follows the player, biased towards where the camera looks
    const fwdFlat = norm3([fwd[0], 0, fwd[2]]);
    const pc = add3(target, scale3(fwdFlat, 14));
    uni.set([Math.floor(pc[0] / PROBE_SPACING) - 12, Math.floor(pc[1] / PROBE_SPACING) - 4, Math.floor(pc[2] / PROBE_SPACING) - 16, PROBE_SPACING], 160);
    uni.set([0.95, frameNo, Math.min(nl, MAX_LIGHTS), 0.1], 164);
    uni.set([rtOn ? 0 : 1, 1, 1, RASTER_EXPOSURE], 168);
    device.queue.writeBuffer(uniBuf, 0, uni);
    prevVP = vp; prevInvVP = invVP; prevEye = eye;

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

    if (!rtOn) {
      // sun shadow cascades
      for (let c = 0; c < CASCADES; c++) {
        const sp = enc.beginRenderPass({
          colorAttachments: [],
          depthStencilAttachment: { view: shadowLayer[c], depthClearValue: 1, depthLoadOp: 'clear', depthStoreOp: 'store' },
        });
        sp.setPipeline(shadowPipe);
        sp.setBindGroup(0, cascadeBG[c]);
        sp.draw(36, nb);
        sp.end();
      }
    }

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
    if (!(rtOn && ptOn)) {
      // DDGI probes light both the raster and the hybrid RT modes
      run(probeTPipe, probeTraceBG, Math.ceil(PROBES / 2 * PRAYS / 64), 1); // trace probe rays
      run(probeUPipe, probeUpdateBG, PROBES / 2, 1);                        // blend into probes
    }
    if (rtOn) {
      run(shadePipe, bg.shade);
    } else {
      run(aoPipe, bg.ao, Math.ceil(W / 16), Math.ceil(H / 16)); // half-res AO
      run(aoBPipe, bg.aoB);
      run(lightPipe, bg.light);                                         // deferred lighting
      const tp = enc.beginRenderPass({                                  // forward glass on top
        colorAttachments: [{ view: views.hdr, loadOp: 'load', storeOp: 'store' }],
        depthStencilAttachment: { view: bg.cur.depth, depthReadOnly: true },
      });
      tp.setPipeline(transpPipe);
      tp.setBindGroup(0, transpBG);
      tp.draw(36, nb);
      tp.end();
      enc.copyTextureToTexture({ texture: tex.hdr }, { texture: tex.hdrPrev }, [W, H]); // for next frame's SSR
    }
    for (const b of bloomPasses) run(b.pipe, b.group, Math.ceil(b.dims[0] / 8), Math.ceil(b.dims[1] / 8));

    const pp = enc.beginRenderPass({
      colorAttachments: [{ view: outView, loadOp: 'clear', storeOp: 'store', clearValue: [0, 0, 0, 1] }],
    });
    pp.setPipeline(postPipe);
    pp.setBindGroup(0, bg.post);
    pp.draw(3);
    pp.end();
    if (beforeSubmit) beforeSubmit(enc, par);
    device.queue.submit([enc.finish()]);
    frameNo++;
    return { nb, nl, vp, invVP, par };

  }

  // ---- asynchronous reprojection driver
  const FOV = 70 * Math.PI / 180;
  const FOV_OVERSCAN = 2 * Math.atan(Math.tan(FOV / 2) * OVERSCAN);
  const SCENE_CAPS = [0, 30, 15, 8];   // 0 = as fast as the GPU allows
  let asyncOn = false, capIdx = 0, sceneBusy = false, lastSceneT = -1e9, warpSrc = null, warpInFlight = 0;
  let sceneCount = 0, sceneT0 = performance.now(), sceneFps = 0, lastNb = 0, lastNl = 0;
  const wu = new Float32Array(56);
  let frozen = false, infillOn = true, showHoles = false;

  function startAsyncScene(now, eye, target) {
    const p = frameNo & 1;
    const wp = warp;
    const r = renderScene(now, eye, target, (p ? tex.ldr1 : tex.ldr0).createView(), FOV_OVERSCAN, (enc, par) => {
      if (!twoDev) return;
      enc.copyTextureToBuffer({ texture: par ? tex.ldr1 : tex.ldr0 }, { buffer: wp.stC, bytesPerRow: wp.cbpr }, [CW, CH]);
      enc.copyTextureToBuffer({ texture: par ? tex.depth1 : tex.depth0, aspect: 'depth-only' }, { buffer: wp.stD, bytesPerRow: wp.dbpr }, [W, H]);
    });
    sceneBusy = true;
    // one device: the queue already orders the warp after this frame, so present it right away
    if (!twoDev) warpSrc = { vp: r.vp, invVP: r.invVP, par: r.par, W, H };
    const t0 = performance.now();
    const done = () => {
      warpSrc = { vp: r.vp, invVP: r.invVP, par: r.par, W, H };
      sceneBusy = false;
      sceneCount++;
      gpuMs = gpuMs * 0.8 + (performance.now() - t0) * 0.2;
    };
    device.queue.onSubmittedWorkDone().then(async () => {
      if (!twoDev) { done(); return; }
      // hand the finished frame to the present device (staging read-back, then upload)
      await Promise.all([wp.stC.mapAsync(GPUMapMode.READ), wp.stD.mapAsync(GPUMapMode.READ)]);
      if (wp === warp) {
        pdev.queue.writeTexture({ texture: wp.sc }, wp.stC.getMappedRange(), { bytesPerRow: wp.cbpr }, [CW, CH]);
        pdev.queue.writeTexture({ texture: wp.sd }, wp.stD.getMappedRange(), { bytesPerRow: wp.dbpr }, [W, H]);
      }
      wp.stC.unmap(); wp.stD.unmap();
      if (wp === warp) done(); else sceneBusy = false;
    }).catch(() => { sceneBusy = false; });
    return r;
  }

  function presentWarp(eye, target) {
    if (!warpSrc || warpInFlight >= 2) return;
    const view = m4.lookAt(eye, target, [0, 1, 0]);
    const vp = m4.mul(m4.persp(FOV, W / H, 0.1), view);
    wu.set(vp, 0);
    wu.set(warpSrc.invVP, 16);
    wu.set(m4.mul(warpSrc.vp, m4.inv(vp)), 32);
    wu.set([warpSrc.W, warpSrc.H, CW, CH], 48);
    wu.set([infillOn ? 1 : 0, showHoles ? 1 : 0, 0, 0], 52);
    pdev.queue.writeBuffer(warpUni, 0, wu);
    const enc = pdev.createCommandEncoder();
    const rp = enc.beginRenderPass({
      colorAttachments: [{ view: ctx.getCurrentTexture().createView(), loadOp: 'clear', storeOp: 'store', clearValue: [0, 0, 0, 1] }],
    });
    rp.setPipeline(warpPipe);
    rp.setBindGroup(0, warp.bg[warpSrc.par]);
    rp.draw(3);
    rp.end();
    pdev.queue.submit([enc.finish()]);
    warpInFlight++;
    pdev.queue.onSubmittedWorkDone().then(() => { warpInFlight--; });
  }
  function setAsync(on) {
    asyncOn = on;
    warpSrc = null;
    if (!on) frozen = false;
    if (twoDev) ctx.configure({ device: on ? pdev : device, format, alphaMode: 'opaque' });
  }

  // At most 2 frames queued on the GPU; otherwise a slow path-traced frame lets the queue grow
  // without bound and the page freezes behind it.
  let inFlight = 0, gpuMs = 0;
  function tick(now) {
    if (!asyncOn && inFlight >= 2) { requestAnimationFrame(tick); return; }
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

    let nb = lastNb, nl = lastNl;
    if (!asyncOn) {
      const r = renderScene(now, eye, target, ctx.getCurrentTexture().createView(), FOV, null);
      nb = r.nb; nl = r.nl;
      inFlight++;
      const t0 = performance.now();
      device.queue.onSubmittedWorkDone().then(() => { inFlight--; gpuMs = gpuMs * 0.9 + (performance.now() - t0) * 0.1; });
    } else {
      const cap = SCENE_CAPS[capIdx];
      if (!sceneBusy && !(frozen && warpSrc) && (cap === 0 || now - lastSceneT >= 1000 / cap - 2)) {
        lastSceneT = now;
        const r = startAsyncScene(now, eye, target);
        nb = r.nb; nl = r.nl;
      }
      presentWarp(eye, target);
    }
    lastNb = nb; lastNl = nl;
    flash = Math.max(0, flash - dt * 2.5);

    // HUD
    $('stage').textContent = `${out[9]} / ${out[10]}`;
    $('deaths').textContent = out[11];
    $('time').textContent = fmtTime(out[12]);
    if (now - sceneT0 > 500) { sceneFps = sceneCount * 1000 / (now - sceneT0); sceneCount = 0; sceneT0 = now; }
    const mode = rtOn ? (ptOn ? 'ReSTIR PT Enhanced' : 'hybrid RT') : `raster GI · ${nl} lights`;
    $('perf').textContent = asyncOn
      ? `timewarp ${fps.toFixed(0)} fps · ${frozen ? 'FROZEN' : `scene ${sceneFps.toFixed(0)} fps`}${SCENE_CAPS[capIdx] ? ` (cap ${SCENE_CAPS[capIdx]})` : ''} · ${W}×${H} · ${mode}`
      : `${fps.toFixed(0)} fps · ${gpuMs.toFixed(1)} ms · ${W}×${H} · ${nb} boxes · ${mode}`;
    if (msgTimer > 0) { msgTimer -= dt; if (msgTimer <= 0) $('msg').style.opacity = 0; }
    requestAnimationFrame(tick);
  }
  requestAnimationFrame(tick);
}

main().catch((e) => fail(String(e?.message || e)));
