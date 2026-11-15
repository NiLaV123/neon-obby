import { RASTER_WGSL, TRACE_WGSL, SHADE_WGSL, POST_WGSL } from './shaders.js';

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

// ------------------------------------------------------------- audio blips
let audio = null;
function blip(freq, dur = 0.12, type = 'square', slide = 0, vol = 0.06) {
  if (!audio) return;
  const t = audio.currentTime, o = audio.createOscillator(), g = audio.createGain();
  o.type = type; o.frequency.setValueAtTime(freq, t);
  if (slide) o.frequency.exponentialRampToValueAtTime(freq * slide, t + dur);
  g.gain.setValueAtTime(vol, t); g.gain.exponentialRampToValueAtTime(0.0001, t + dur);
  o.connect(g).connect(audio.destination); o.start(t); o.stop(t + dur);
}

function fail(msg) {
  $('err').textContent = msg;
  $('err').style.display = 'block';
  $('start').style.display = 'none';
}

async function main() {
  if (!navigator.gpu) return fail('WebGPU is not available in this browser. Try a recent Chrome, Edge, or Safari 26+.');
  const adapter = await navigator.gpu.requestAdapter({ powerPreference: 'high-performance' });
  if (!adapter) return fail('No WebGPU adapter found.');
  const device = await adapter.requestDevice();
  device.lost.then((i) => fail('GPU device lost: ' + i.message));
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
  const uniBuf = device.createBuffer({ size: 256, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
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
  const rasterMod = mod(RASTER_WGSL, 'raster'), traceMod = mod(TRACE_WGSL, 'trace');
  const shadeMod = mod(SHADE_WGSL, 'shade'), postMod = mod(POST_WGSL, 'post');

  const rasterPipe = device.createRenderPipeline({
    layout: 'auto',
    vertex: { module: rasterMod, entryPoint: 'vs' },
    fragment: { module: rasterMod, entryPoint: 'fs', targets: [{ format: 'rgba8unorm' }, { format: 'rgba16float' }] },
    primitive: { topology: 'triangle-list', cullMode: 'none' },
    depthStencil: { format: 'depth32float', depthWriteEnabled: true, depthCompare: 'greater' },
  });
  const tracePipe = device.createComputePipeline({ layout: 'auto', compute: { module: traceMod, entryPoint: 'main' } });
  const shadePipe = device.createComputePipeline({ layout: 'auto', compute: { module: shadeMod, entryPoint: 'main' } });
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

  let W = 0, H = 0, tex = null, views = null, traceBG = null, rtBGs = null, postBG = null, renderScale = 1;
  const RT = GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING;
  const ST = GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.TEXTURE_BINDING;
  function resize() {
    const dpr = Math.min(devicePixelRatio || 1, 2);
    const cw = Math.max(1, Math.floor(canvas.clientWidth * dpr));
    const ch = Math.max(1, Math.floor(canvas.clientHeight * dpr));
    if (canvas.width !== cw || canvas.height !== ch) { canvas.width = cw; canvas.height = ch; }
    const nw = Math.max(1, Math.floor(cw * renderScale)), nh = Math.max(1, Math.floor(ch * renderScale));
    if (nw === W && nh === H && tex) return;
    W = nw; H = nh;
    if (tex) for (const t of Object.values(tex)) t.destroy();
    const mk = (f, u) => device.createTexture({ size: [W, H], format: f, usage: u });
    tex = {
      albedo: mk('rgba8unorm', RT), normal: mk('rgba16float', RT), depth: mk('depth32float', RT),
      hist0: mk('rgba16float', ST), hist1: mk('rgba16float', ST), hdr: mk('rgba16float', ST),
      raw: mk('rgba16float', ST),
    };
    const v = {}; for (const k in tex) v[k] = tex[k].createView();
    views = v;
    const scene = [
      { binding: 0, resource: { buffer: uniBuf } },
      { binding: 1, resource: { buffer: boxBuf } },
      { binding: 2, resource: { buffer: nodeBuf } },
      { binding: 3, resource: { buffer: primBuf } },
      { binding: 4, resource: v.normal },
      { binding: 5, resource: v.depth },
    ];
    traceBG = device.createBindGroup({
      layout: tracePipe.getBindGroupLayout(0),
      entries: [...scene, { binding: 6, resource: v.raw }],
    });
    rtBGs = [0, 1].map((i) => device.createBindGroup({
      layout: shadePipe.getBindGroupLayout(0),
      entries: [
        ...scene,
        { binding: 6, resource: v.albedo },
        { binding: 7, resource: v.raw },
        { binding: 8, resource: i ? v.hist1 : v.hist0 },
        { binding: 9, resource: i ? v.hist0 : v.hist1 },
        { binding: 10, resource: v.hdr },
      ],
    }));
    postBG = device.createBindGroup({
      layout: postPipe.getBindGroupLayout(0),
      entries: [{ binding: 0, resource: v.hdr }, { binding: 1, resource: { buffer: uniBuf } }, { binding: 2, resource: sampler }],
    });
  }

  // --------------------------------------------------------- input
  const keys = new Set();
  let yaw = Math.PI, pitch = 0.32, dist = 18, rtOn = true, dragging = false, playing = false;
  const SCALES = [1, 0.75, 0.5];
  let scaleIdx = 0;
  addEventListener('keydown', (e) => {
    keys.add(e.code);
    if (e.code === 'Space') e.preventDefault();
    if (e.repeat) return;
    if (e.code === 'KeyR') { rtOn = !rtOn; flashMsg(rtOn ? 'Ray tracing ON' : 'Raster only'); }
    if (e.code === 'KeyG') { scaleIdx = (scaleIdx + 1) % SCALES.length; renderScale = SCALES[scaleIdx]; flashMsg(`Render scale ${Math.round(renderScale * 100)}%`); }
    if (e.code === 'KeyK') w.respawn();
    if (e.code === 'KeyT') { w.restart(); $('win').style.display = 'none'; }
    if (e.code === 'KeyH') $('help').classList.toggle('hidden');
  });
  addEventListener('keyup', (e) => keys.delete(e.code));
  addEventListener('blur', () => keys.clear());
  canvas.addEventListener('contextmenu', (e) => e.preventDefault());
  canvas.addEventListener('mousedown', (e) => {
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
  $('start').addEventListener('click', () => {
    playing = true;
    $('start').style.display = 'none';
    try { audio = audio || new AudioContext(); } catch { /* no audio */ }
    canvas.requestPointerLock?.();
  });

  let msgTimer = 0;
  function flashMsg(t) { $('msg').textContent = t; $('msg').style.opacity = 1; msgTimer = 1.6; }

  // --------------------------------------------------------- loop
  const uni = new Float32Array(64);
  const sun = (() => { const s = [-0.42, 0.78, -0.46], l = Math.hypot(...s); return s.map((x) => x / l); })();
  let prevVP = null, last = performance.now(), frameNo = 0, flash = 0, fpsAcc = 0, fpsN = 0, fps = 0;

  function fmtTime(t) { const m = Math.floor(t / 60), s = (t % 60).toFixed(2).padStart(5, '0'); return `${m}:${s}`; }

  function tick(now) {
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
    if (ev & 1) { blip(660, 0.1, 'square', 1.5); setTimeout(() => blip(990, 0.15, 'square', 1.3), 90); flashMsg(`Checkpoint! Stage ${out[9]}`); }
    if (ev & 2) { blip(220, 0.35, 'sawtooth', 0.3, 0.08); flash = 1; }
    if (ev & 4) { [523, 659, 784, 1046].forEach((f, i) => setTimeout(() => blip(f, 0.25, 'triangle', 1, 0.08), i * 120)); $('win').style.display = 'flex'; $('wintime').textContent = fmtTime(out[12]); $('windeaths').textContent = out[11]; }
    if (ev & 8) blip(420, 0.09, 'square', 1.6, 0.03);
    if (ev & 16) blip(200, 0.3, 'sine', 4, 0.09);

    const nb = w.box_count(), nn = w.node_count(), np = w.prim_count();
    device.queue.writeBuffer(boxBuf, 0, mem, w.boxes_ptr(), nb * 64);
    device.queue.writeBuffer(nodeBuf, 0, mem, w.nodes_ptr(), nn * 32);
    device.queue.writeBuffer(primBuf, 0, mem, w.prims_ptr(), np * 4);

    const view = m4.lookAt(eye, target, [0, 1, 0]);
    const proj = m4.persp(70 * Math.PI / 180, W / H, 0.1);
    const vp = m4.mul(proj, view);
    uni.set(vp, 0);
    uni.set(m4.inv(vp), 16);
    uni.set(prevVP || vp, 32);
    uni.set([eye[0], eye[1], eye[2], now / 1000], 48);
    uni.set([sun[0], sun[1], sun[2], frameNo], 52);
    uni.set([W, H, rtOn ? 1 : 0, nb], 56);
    uni.set([flash, 1.0, canvas.width, canvas.height], 60);
    device.queue.writeBuffer(uniBuf, 0, uni);
    prevVP = vp;
    flash = Math.max(0, flash - dt * 2.5);

    const enc = device.createCommandEncoder();
    const rp = enc.beginRenderPass({
      colorAttachments: [
        { view: views.albedo, loadOp: 'clear', storeOp: 'store', clearValue: [0, 0, 0, 0] },
        { view: views.normal, loadOp: 'clear', storeOp: 'store', clearValue: [0, 0, 0, 0] },
      ],
      depthStencilAttachment: { view: views.depth, depthClearValue: 0, depthLoadOp: 'clear', depthStoreOp: 'store' },
    });
    rp.setPipeline(rasterPipe);
    rp.setBindGroup(0, rasterBG);
    rp.draw(36, nb);
    rp.end();

    const gx = Math.ceil(W / 8), gy = Math.ceil(H / 8);
    if (rtOn) {
      const tp = enc.beginComputePass();
      tp.setPipeline(tracePipe);
      tp.setBindGroup(0, traceBG);
      tp.dispatchWorkgroups(gx, gy);
      tp.end();
    }
    const cp = enc.beginComputePass();
    cp.setPipeline(shadePipe);
    cp.setBindGroup(0, rtBGs[frameNo & 1]);
    cp.dispatchWorkgroups(gx, gy);
    cp.end();

    const pp = enc.beginRenderPass({
      colorAttachments: [{ view: ctx.getCurrentTexture().createView(), loadOp: 'clear', storeOp: 'store', clearValue: [0, 0, 0, 1] }],
    });
    pp.setPipeline(postPipe);
    pp.setBindGroup(0, postBG);
    pp.draw(3);
    pp.end();
    device.queue.submit([enc.finish()]);
    frameNo++;

    // HUD
    $('stage').textContent = `${out[9]} / ${out[10]}`;
    $('deaths').textContent = out[11];
    $('time').textContent = fmtTime(out[12]);
    $('perf').textContent = `${fps.toFixed(0)} fps · ${W}×${H} · ${nb} boxes · ${rtOn ? 'RT' : 'raster'}`;
    if (msgTimer > 0) { msgTimer -= dt; if (msgTimer <= 0) $('msg').style.opacity = 0; }
    requestAnimationFrame(tick);
  }
  requestAnimationFrame(tick);
}

main().catch((e) => fail(String(e?.message || e)));
