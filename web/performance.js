// Adjust only internal resolution; G remains the user's maximum render scale.
// Hysteresis and a cooldown avoid reallocating render targets every few frames.
export class RenderBudget {
  constructor({ targetMs = 1000 / 60, cooldownMs = 1000 } = {}) {
    this.targetMs = targetMs;
    this.cooldownMs = cooldownMs;
    this.reset();
  }

  reset(now = 0) {
    this.scale = 1;
    this.averageMs = 0;
    this.samples = 0;
    this.slowSamples = 0;
    this.fastSamples = 0;
    this.lastChange = now;
  }

  observe(milliseconds, now) {
    if (!Number.isFinite(milliseconds) || milliseconds <= 0 || !Number.isFinite(now)) return;
    this.averageMs = this.samples ? this.averageMs * 0.85 + milliseconds * 0.15 : milliseconds;
    this.samples++;
    this.slowSamples = milliseconds > this.targetMs * 1.15 ? this.slowSamples + 1 : 0;
    this.fastSamples = milliseconds < this.targetMs * 0.7 ? this.fastSamples + 1 : 0;
    if (this.samples < 12 || now - this.lastChange < this.cooldownMs) return;
    let next = this.scale;
    if (this.averageMs > this.targetMs * 1.15 && this.slowSamples >= 6) next -= 0.1;
    else if (this.averageMs < this.targetMs * 0.7 && this.fastSamples >= 30) next += 0.05;
    next = Math.round(Math.min(1, Math.max(0.5, next)) * 20) / 20;
    if (next === this.scale) return;
    this.scale = next;
    this.lastChange = now;
    this.samples = 0;
    this.averageMs = 0;
  }
}

export function probeSchedule(frame, stride = 4) {
  // Strides divide the 9,216-probe volume, so every slot is visited exactly
  // once per cycle. Match the old two-frame temporal response at every budget.
  if (![2, 4, 8].includes(stride)) throw new RangeError('Probe stride must be 2, 4, or 8');
  return { count: 9216 / stride, phase: frame % stride, hysteresis: 0.95 ** (stride / 2) };
}
