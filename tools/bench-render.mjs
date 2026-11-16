// Run the actual WebGPU renderer offscreen with Dawn, using a deterministic
// camera/physics timeline. Install the optional `webgpu` package to run this.
import assert from 'node:assert/strict';
import fs from 'node:fs';
import path from 'node:path';
import { createRequire } from 'node:module';
import { pathToFileURL } from 'node:url';

const require = createRequire(import.meta.url);
const { create, globals } = await import(pathToFileURL(require.resolve('webgpu')));
Object.assign(globalThis, globals);
const root = path.resolve(process.argv[2] || 'web');
const output = process.argv[3];
const frames = Number(process.env.BENCH_FRAMES || 20);
const width = Number(process.env.BENCH_WIDTH || 960);
const height = Number(process.env.BENCH_HEIGHT || 540);
const mode = process.env.BENCH_MODE || 'raster';
assert.ok(['raster', 'hybrid', 'pt', 'async'].includes(mode));
assert.ok(Number.isInteger(frames) && frames > 0);
const gpu = create([]);
const realNow = performance.now.bind(performance);
let clock = 1000;
let frameStart = null;
// Initialization must use the same clock as tick, including its first dt.
Object.defineProperty(globalThis, 'performance', { value: { now: () => clock + (frameStart === null ? 0 : realNow() - frameStart) }, configurable: true });
Object.defineProperty(globalThis, 'localStorage', { value: undefined, configurable: true });
Object.defineProperty(globalThis, 'navigator', { value: { gpu }, configurable: true });
globalThis.window = globalThis;
globalThis.devicePixelRatio = 1;
const handlers = new Map();
globalThis.addEventListener = (name, callback) => {
  if (!handlers.has(name)) handlers.set(name, []);
  handlers.get(name).push(callback);
};
function press(code) {
  const event = { code, repeat: false, target: { closest: () => null }, preventDefault() {} };
  for (const callback of handlers.get('keydown') || []) callback(event);
  for (const callback of handlers.get('keyup') || []) callback(event);
}
const devices = new Set();
const requestAdapter = gpu.requestAdapter.bind(gpu);
gpu.requestAdapter = async (...args) => {
  const adapter = await requestAdapter(...args);
  if (adapter) {
    const requestDevice = adapter.requestDevice.bind(adapter);
    adapter.requestDevice = async (...options) => {
      const device = await requestDevice(...options);
      devices.add(device);
      return device;
    };
  }
  return adapter;
};
const elements = new Map();
function element(id) {
  if (!elements.has(id)) elements.set(id, {
    textContent: '', value: '', style: {},
    addEventListener() {}, setAttribute() {},
    classList: { toggle() {} },
    querySelector: () => element(`${id}-button`),
  });
  return elements.get(id);
}
globalThis.document = { getElementById: element, addEventListener() {}, hidden: false };
const canvas = element('c');
Object.assign(canvas, { width, height, clientWidth: width, clientHeight: height });
let device, texture;
const context = {
  configure(config) {
    if (device && device !== config.device) { texture?.destroy(); texture = null; }
    device = config.device;
    this.format = config.format;
  },
  getCurrentTexture() {
    if (!texture || texture.width !== canvas.width || texture.height !== canvas.height) {
      texture?.destroy();
      texture = device.createTexture({ size: [canvas.width, canvas.height], format: this.format,
        usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.COPY_SRC });
    }
    return texture;
  },
};
canvas.getContext = () => context;
globalThis.fetch = async (name) => ({ arrayBuffer: async () => {
  const bytes = fs.readFileSync(path.join(root, name));
  return bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength);
} });
let tick, ready;
const initialized = new Promise((resolve) => { ready = resolve; });
globalThis.requestAnimationFrame = (callback) => { tick = callback; ready(); };
await import(pathToFileURL(path.join(root, 'main.js')));
const timeout = setTimeout(() => {
  console.error(element('err').textContent || 'Renderer did not initialize');
  process.exit(1);
}, 30000);
await initialized;
clearTimeout(timeout);
if (mode === 'pt' || mode === 'hybrid') press('KeyR');
if (mode === 'hybrid') press('KeyP');
if (mode === 'async') press('KeyF');
if (process.env.BENCH_AUTO === '0') press('KeyQ');

function checkErrors() {
  assert.equal(element('err').textContent, '', 'renderer error');
  assert.deepEqual(globalThis.__gpuErrs || [], [], 'WebGPU validation errors');
}
const times = [];
for (let frame = 0; frame < frames + 5; frame++) {
  const start = realNow();
  clock = 1000 + frame * 1000 / 60;
  frameStart = realNow();
  if (process.env.BENCH_RESIZE && frame === 10) {
    canvas.clientWidth += 64;
    for (const callback of handlers.get('resize') || []) callback();
  }
  tick(clock);
  await Promise.all([...devices].map((device) => device.queue.onSubmittedWorkDone()));
  if (mode === 'async') await new Promise(setImmediate);
  checkErrors();
  if (frame >= 5) times.push(realNow() - start);
}
times.sort((a, b) => a - b);
const result = {
  viewport: [canvas.width, canvas.height], frames, mode,
  hud: element('perf').textContent,
  meanMs: times.reduce((a, b) => a + b, 0) / times.length,
  medianMs: times[Math.floor(times.length / 2)],
  p95Ms: times[Math.min(times.length - 1, Math.floor(times.length * 0.95))],
};
if (output) {
  fs.writeFileSync(output, JSON.stringify(result, null, 2) + '\n');
  const bytesPerRow = Math.ceil(canvas.width * 4 / 256) * 256;
  const readback = device.createBuffer({ size: bytesPerRow * canvas.height,
    usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
  const encoder = device.createCommandEncoder();
  encoder.copyTextureToBuffer({ texture }, { buffer: readback, bytesPerRow }, [canvas.width, canvas.height]);
  device.queue.submit([encoder.finish()]);
  await readback.mapAsync(GPUMapMode.READ);
  const padded = new Uint8Array(readback.getMappedRange());
  const pixels = new Uint8Array(canvas.width * canvas.height * 4);
  for (let y = 0; y < canvas.height; y++) pixels.set(padded.subarray(y * bytesPerRow, y * bytesPerRow + canvas.width * 4), y * canvas.width * 4);
  fs.writeFileSync(`${output}.rgba`, pixels);
  readback.unmap();
  readback.destroy();
  checkErrors();
}
console.log(JSON.stringify(result));
for (const device of devices) device.destroy();
process.exit(0);
