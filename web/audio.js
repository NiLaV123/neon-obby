// Original procedural effects. Times are scheduled on the audio clock so chords
// and melodies stay in time even when path tracing slows down rendering.
const tone = (frequency, duration, options = {}) => ({ frequency, duration, type: 'sine', gain: 0.1, ...options });
const noise = (duration, options = {}) => ({ noise: true, duration, cutoff: 1500, gain: 0.1, ...options });
const notes = (frequencies, step, { at = 0, ...options } = {}) => frequencies.map((frequency, i) => tone(frequency, 0.24, { at: at + i * step, type: 'triangle', ...options }));

export const SOUND_LIBRARY = {
  start: [...notes([392, 523.25, 659.25, 783.99], 0.075), tone(130.81, 0.42, { gain: 0.14 })],
  jump: [tone(260, 0.18, { end: 780, type: 'triangle', gain: 0.12 }), noise(0.09, { cutoff: 2800, gain: 0.035 })],
  land: [tone(115, 0.14, { end: 45, gain: 0.2 }), noise(0.12, { cutoff: 650, gain: 0.15 })],
  footstep: [noise(0.055, { cutoff: 1350, gain: 0.09 }), tone(165, 0.065, { end: 65, gain: 0.055 })],
  bounce: [tone(140, 0.38, { end: 1150, type: 'triangle', gain: 0.17 }), tone(210, 0.28, { at: 0.04, end: 1725, gain: 0.065 }), noise(0.06, { cutoff: 900, gain: 0.12 })],
  checkpoint: [...notes([523.25, 659.25, 783.99, 1046.5], 0.085, { duration: 0.38, gain: 0.11 }), tone(261.63, 0.55, { gain: 0.1 })],
  death: [noise(0.42, { cutoff: 1800, gain: 0.16 }), tone(220, 0.46, { end: 40, type: 'sawtooth', gain: 0.065 }), ...notes([311.13, 261.63, 196], 0.1, { gain: 0.07 })],
  respawn: [...notes([261.63, 392, 523.25], 0.08, { duration: 0.22, gain: 0.085 }), noise(0.13, { cutoff: 3200, gain: 0.025 })],
  restart: [...notes([783.99, 523.25, 392, 523.25], 0.07, { gain: 0.08 })],
  toggle: [tone(880, 0.055, { type: 'triangle', gain: 0.07 }), tone(1320, 0.065, { at: 0.045, gain: 0.04 })],
  win: [
    ...notes([523.25, 659.25, 783.99, 1046.5, 783.99, 1046.5, 1318.51, 1567.98], 0.12, { duration: 0.38, gain: 0.1 }),
    ...[130.81, 261.63, 329.63, 392].map((f) => tone(f, 0.9, { at: 0.85, type: 'triangle', gain: 0.065, attack: 0.025 })),
    ...notes([1046.5, 1318.51, 1567.98, 2093], 0.11, { at: 1.25, duration: 0.5, gain: 0.025 }),
  ],
};

const SETTINGS_KEY = 'neon-obby-audio';
const MAX_VOICES = 48;
const INTERRUPTS = new Set(['start', 'death', 'respawn', 'restart', 'win']);
const clamp = (value, low, high) => Math.min(high, Math.max(low, value));

export class SoundLibrary {
  constructor({ createContext = () => new AudioContext(), storage, random = Math.random } = {}) {
    this.createContext = createContext;
    this.random = random;
    this.context = null;
    this.voices = new Set();
    this.volume = 0.65;
    this.muted = false;
    this.active = true;
    try {
      this.storage = storage === undefined ? globalThis.localStorage : storage;
      const saved = JSON.parse(this.storage?.getItem(SETTINGS_KEY) || 'null');
      if (Number.isFinite(saved?.volume)) this.volume = clamp(saved.volume, 0, 1);
      if (typeof saved?.muted === 'boolean') this.muted = saved.muted;
    } catch { /* Audio still works when storage is unavailable. */ }
  }

  // Call from a user gesture; browsers may suspend audio again in background tabs.
  async unlock() {
    try {
      if (!this.context || this.context.state === 'closed') {
        this.context = this.createContext();
        const ctx = this.context;
        this.master = ctx.createGain();
        this.master.gain.value = this.muted ? 0 : this.volume;
        const limiter = ctx.createDynamicsCompressor();
        limiter.threshold.value = -14;
        limiter.knee.value = 12;
        limiter.ratio.value = 6;
        limiter.attack.value = 0.003;
        limiter.release.value = 0.15;
        this.master.connect(limiter).connect(ctx.destination);
        this.noise = ctx.createBuffer(1, ctx.sampleRate, ctx.sampleRate);
        const samples = this.noise.getChannelData(0);
        for (let i = 0; i < samples.length; i++) samples[i] = this.random() * 2 - 1;
      }
      if (this.active && this.context.state === 'suspended') await this.context.resume();
      return this.context.state === 'running';
    } catch {
      // An unavailable or blocked audio device must never prevent playing.
      return false;
    }
  }

  save() {
    try { this.storage?.setItem(SETTINGS_KEY, JSON.stringify({ volume: this.volume, muted: this.muted })); } catch { /* Private browsing. */ }
    if (this.master) this.master.gain.setTargetAtTime(this.muted ? 0 : this.volume, this.context.currentTime, 0.015);
  }

  setVolume(volume) {
    if (!Number.isFinite(volume)) return;
    this.volume = clamp(volume, 0, 1);
    this.save();
  }

  setMuted(muted) {
    this.muted = Boolean(muted);
    if (this.muted) this.stopAll();
    this.save();
  }

  setActive(active) {
    this.active = active;
    if (!active) {
      this.stopAll();
      this.context?.suspend().catch(() => {});
    } else if (this.context) {
      void this.unlock();
    }
  }

  stopAll() {
    for (const voice of this.voices) {
      const now = this.context.currentTime;
      voice.envelope.gain.cancelScheduledValues(now);
      voice.envelope.gain.setTargetAtTime(0.0001, now, 0.004);
      voice.source.stop(now + 0.02);
    }
    // The interrupted voices finish their short fade while the replacement cue
    // gets a fresh voice budget (so a busy frame cannot drop the victory sound).
    this.voices.clear();
  }

  play(name, { intensity = 1, pitch = 1, pan = 0 } = {}) {
    const ctx = this.context, layers = SOUND_LIBRARY[name];
    if (!layers || !this.active || this.muted || !this.volume || ctx?.state !== 'running') return;
    if (INTERRUPTS.has(name)) this.stopAll();
    if (name === 'footstep' || name === 'jump') pitch *= 0.94 + this.random() * 0.12;
    const now = ctx.currentTime + 0.005;
    for (const layer of layers) {
      if (this.voices.size >= MAX_VOICES) break;
      const start = now + (layer.at || 0), end = start + layer.duration;
      const source = layer.noise ? ctx.createBufferSource() : ctx.createOscillator();
      if (layer.noise) {
        source.buffer = this.noise;
      } else {
        source.type = layer.type;
        source.frequency.setValueAtTime(layer.frequency * pitch, start);
        if (layer.end) source.frequency.exponentialRampToValueAtTime(layer.end * pitch, end);
      }
      const envelope = ctx.createGain(), panner = ctx.createStereoPanner();
      panner.pan.value = clamp(pan, -1, 1);
      const attack = layer.attack || 0.004;
      envelope.gain.setValueAtTime(0.0001, start);
      envelope.gain.exponentialRampToValueAtTime(Math.max(0.0001, layer.gain * clamp(intensity, 0, 1)), start + attack);
      envelope.gain.exponentialRampToValueAtTime(0.0001, end);
      const nodes = [source, envelope, panner];
      if (layer.noise) {
        const filter = ctx.createBiquadFilter();
        filter.type = 'lowpass';
        filter.frequency.value = layer.cutoff;
        source.connect(filter).connect(envelope);
        nodes.push(filter);
      } else {
        source.connect(envelope);
      }
      envelope.connect(panner).connect(this.master);
      const voice = { source, envelope };
      this.voices.add(voice);
      source.onended = () => {
        for (const node of nodes) node.disconnect();
        this.voices.delete(voice);
      };
      source.start(start);
      source.stop(end + 0.01);
    }
  }
}

// Use distance, not frame count, for footsteps. Grounded transitions produce
// landings; resetting on teleports prevents fake footsteps after respawning.
export class MovementSounds {
  constructor(sounds) {
    this.sounds = sounds;
    this.reset();
  }

  reset() {
    this.previous = null;
    this.distance = 0;
    this.airTime = 0;
    this.left = false;
  }

  update({ x, z, grounded, moving, events }, dt) {
    const previous = this.previous;
    this.previous = { x, z, grounded };
    if (!previous || events & (2 | 4)) {
      this.distance = 0;
      this.airTime = 0;
      return;
    }
    if (!grounded) {
      this.airTime += dt;
      this.distance = 0;
      return;
    }
    if (!previous.grounded && this.airTime > 0.08) {
      this.sounds.play('land', { intensity: clamp(this.airTime / 0.7, 0.3, 1) });
      this.distance = 0;
    } else if (moving && previous.grounded && !(events & (8 | 16))) {
      const distance = Math.hypot(x - previous.x, z - previous.z);
      this.distance += distance < 4 ? distance : 0;
      if (this.distance >= 2.8) {
        this.sounds.play('footstep', { pan: this.left ? -0.2 : 0.2 });
        this.left = !this.left;
        this.distance %= 2.8;
      }
    } else {
      this.distance = 0;
    }
    this.airTime = 0;
  }
}
