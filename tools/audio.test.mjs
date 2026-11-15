import assert from 'node:assert/strict';
import test from 'node:test';
import { SoundLibrary, SOUND_LIBRARY, MovementSounds } from '../web/audio.js';

// A scheduling-only AudioContext lets lifecycle tests run with stock Node.
class Param {
  constructor() { this.value = 0; this.events = []; }
  record(value, time) {
    assert.ok(Number.isFinite(value) && Number.isFinite(time) && time >= 0);
    this.events.push({ value, time });
  }
  setValueAtTime(value, time) { this.record(value, time); }
  exponentialRampToValueAtTime(value, time) { assert.ok(value > 0); this.record(value, time); }
  setTargetAtTime(value, time) { this.record(value, time); }
  cancelScheduledValues(time) { this.events = this.events.filter((event) => event.time < time); }
}

class AudioNode {
  constructor() {
    for (const name of ['gain', 'frequency', 'pan', 'threshold', 'knee', 'ratio', 'attack', 'release']) this[name] = new Param();
  }
  connect(node) { assert.ok(node); return node; }
  disconnect() { this.disconnected = true; }
  start(time) { assert.ok(Number.isFinite(time) && time >= 0); this.startTime = time; }
  stop(time) { assert.ok(Number.isFinite(time) && time >= 0); this.stopTime = time; }
}

class AudioContext {
  constructor() { this.currentTime = 3; this.sampleRate = 8000; this.state = 'suspended'; this.destination = new AudioNode(); this.sources = []; }
  async resume() { this.state = 'running'; }
  async suspend() { this.state = 'suspended'; }
  createGain() { return new AudioNode(); }
  createDynamicsCompressor() { return new AudioNode(); }
  createBiquadFilter() { return new AudioNode(); }
  createStereoPanner() { return new AudioNode(); }
  createBuffer(channels, length) { const data = new Float32Array(length); return { getChannelData: () => data }; }
  createOscillator() { const source = new AudioNode(); this.sources.push(source); return source; }
  createBufferSource() { return this.createOscillator(); }
  finish() { for (const source of this.sources) source.onended?.(); }
}

const setup = (options = {}) => {
  const context = new AudioContext();
  const sounds = new SoundLibrary({ createContext: () => context, storage: null, ...options });
  return { sounds, context };
};

test('audio remains lazy and device failures do not block gameplay', async () => {
  let attempts = 0;
  const sounds = new SoundLibrary({ createContext: () => { attempts++; throw Error('No device'); }, storage: null });
  sounds.play('start');
  assert.equal(attempts, 0);
  assert.equal(await sounds.unlock(), false);
  assert.doesNotThrow(() => sounds.play('jump'));
});

test('every effect schedules layers on the audio clock and releases its nodes', async () => {
  for (const name of Object.keys(SOUND_LIBRARY)) {
    const { sounds, context } = setup();
    await sounds.unlock();
    sounds.play(name);
    assert.ok(context.sources.length > 1, `${name} should have multiple layers`);
    for (const source of context.sources) {
      assert.ok(source.startTime >= context.currentTime);
      assert.ok(source.stopTime > source.startTime);
    }
    context.finish();
    assert.equal(sounds.voices.size, 0);
    assert.ok(context.sources.every((source) => source.disconnected));
  }
});

test('mute cancels queued fanfare notes and preferences survive a reload', async () => {
  let saved = null;
  const storage = { getItem: () => saved, setItem: (key, value) => { saved = value; } };
  const { sounds, context } = setup({ storage });
  await sounds.unlock();
  sounds.play('win');
  assert.ok(context.sources.some((source) => source.startTime > context.currentTime + 1));
  sounds.setVolume(0.3);
  sounds.setMuted(true);
  assert.ok(context.sources.every((source) => source.stopTime <= context.currentTime + 0.02));
  const count = context.sources.length;
  sounds.play('jump');
  assert.equal(context.sources.length, count);
  const reloaded = new SoundLibrary({ storage });
  assert.equal(reloaded.volume, 0.3);
  assert.equal(reloaded.muted, true);
});

test('invalid or unavailable storage and volume values are handled', () => {
  for (const saved of ['invalid', '{"volume":99,"muted":true}', '{"volume":"loud","muted":"yes"}']) {
    const { sounds } = setup({ storage: { getItem: () => saved } });
    assert.ok(sounds.volume >= 0 && sounds.volume <= 1);
    sounds.setVolume(NaN);
    assert.ok(Number.isFinite(sounds.volume));
    sounds.setVolume(-1);
    assert.equal(sounds.volume, 0);
  }
  const { sounds } = setup({ storage: { getItem() { throw Error('Blocked'); }, setItem() { throw Error('Blocked'); } } });
  assert.doesNotThrow(() => sounds.setMuted(true));
});

test('backgrounding stops sounds and foregrounding resumes the existing context', async () => {
  const { sounds, context } = setup();
  await sounds.unlock();
  sounds.play('checkpoint');
  sounds.setActive(false);
  const count = context.sources.length;
  sounds.play('jump');
  assert.equal(context.sources.length, count);
  assert.equal(context.state, 'suspended');
  sounds.setActive(true);
  await sounds.unlock();
  assert.equal(sounds.context, context);
  assert.equal(context.state, 'running');
});

test('voice count is bounded and cleaned-up capacity can be reused', async () => {
  const { sounds, context } = setup();
  await sounds.unlock();
  for (let i = 0; i < 100; i++) sounds.play('footstep');
  assert.ok(context.sources.length <= 48);
  context.finish();
  const count = context.sources.length;
  sounds.play('jump');
  assert.ok(context.sources.length > count);
});

test('a victory fanfare replaces a full voice pool without losing notes', async () => {
  const { sounds, context } = setup();
  await sounds.unlock();
  for (let i = 0; i < 100; i++) sounds.play('footstep');
  const count = context.sources.length;
  sounds.play('win');
  assert.equal(context.sources.length - count, SOUND_LIBRARY.win.length);
  assert.ok(context.sources.slice(0, count).every((source) => source.stopTime <= context.currentTime + 0.02));
});

const movement = () => {
  const played = [];
  return { played, tracker: new MovementSounds({ play: (name, options) => played.push({ name, ...options }) }) };
};
const position = (x = 0, grounded = true, extra = {}) => ({ x, z: 0, grounded, moving: true, events: 0, ...extra });

test('footstep cadence follows distance consistently at different frame rates', () => {
  for (const fps of [20, 60, 144]) {
    const { played, tracker } = movement();
    tracker.update(position(), 1 / fps);
    for (let frame = 1; frame <= fps * 2; frame++) tracker.update(position(frame * 16 / fps), 1 / fps);
    assert.equal(played.length, 11, `${fps} fps`);
    assert.ok(played.every((sound) => sound.name === 'footstep'));
    assert.equal(played[0].pan, -played[1].pan);
  }
});

test('airborne movement produces one landing and no footsteps', () => {
  const { played, tracker } = movement();
  tracker.update(position(), 1 / 60);
  for (let i = 1; i <= 30; i++) tracker.update(position(i / 4, false), 1 / 60);
  tracker.update(position(7.5), 1 / 60);
  tracker.update(position(7.5), 1 / 60);
  assert.equal(played.length, 1);
  assert.equal(played[0].name, 'land');
  assert.ok(played[0].intensity > 0.3);
});

test('death, reset, and passive platform movement do not produce false steps or landings', () => {
  const { played, tracker } = movement();
  tracker.update(position(0, false), 0.1);
  tracker.update(position(1, false), 0.1);
  tracker.update(position(0, true, { events: 2 }), 0.1);
  tracker.reset();
  tracker.update(position(100), 0.1);
  tracker.update(position(110, true, { moving: false }), 0.1);
  tracker.update(position(200), 0.1);
  assert.deepEqual(played, []);
});
